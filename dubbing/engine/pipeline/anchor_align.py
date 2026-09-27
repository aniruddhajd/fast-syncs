"""
Phrase-level anchor alignment (v0.18.1) — AI mode sync
======================================================
Whole-paragraph pieces sync badly: an English paragraph is several phrases
with pauses between them, and one continuous block of dubbed speech drifts
past every one of those pauses. This module aligns the dub at PHRASE level.

  units  = the approved paragraphs split into clauses (, ; : । . ! ? —),
           tiny fragments merged; a unit never crosses a paragraph, so the
           cast (one voice per paragraph) is kept.
  cues   = the English sync-SRT phrases (start, end) — the same
           speech-region phrases the other sync modes use.
  align  = one monotonic dynamic-programming pass (Gale–Church in spirit,
           but on SECONDS): consecutive units are grouped onto consecutive
           cues so that the group's REAL synthesized duration matches the
           phrase span. Moves: 1-4 units x 1-4 cues, plus "English phrase
           with no dub" (fillers, laughter). A group never crosses a
           paragraph.
  hint   = each paragraph's row window (from the translator, or a memory
           re-pairing) is a SOFT prior: a group whose cues fall outside its
           row's window pays a penalty — strong for translator rows (they
           are exact), weak for memory rows (they were paired by length).

Output: groups (u_lo, u_hi, c_lo, c_hi), inclusive indices; each becomes one
placed piece whose window is its cues' span.

v0.20 — the English cues themselves (phrase_cues / cues_from_items)
-------------------------------------------------------------------
The alignment can only be as fine as its cues. The sync SRT's cues are
loudness regions (an absolute -42 dB gate) split by spaCy, which is not
installed — so a fluent 76 s talk came back as 17 phrases, one 18 s long,
and every clause of that stretch shared one 18 s window. AI mode now builds
its cues from the Scribe WORD timestamps: cut at a word gap >= 0.30 s, at a
spectral pause (spectral_vad), at a sentence end, and never let a cue run
past 5 s; cues under 0.8 s join a close neighbour. When the user cut the
English into items on the timeline, those items are the cues
(cues_from_items). This is the segmentation step of prosodic alignment for
automatic dubbing (Federico et al., Interspeech 2020; Virkar et al., ICASSP
2021): the translation is timed against the source's real phrases.

v0.20 — pause markers (clause_units(..., pauses=))
--------------------------------------------------
The AI translator now writes ' | ' where the English pauses (isochrony-aware
translation, arXiv 2112.08548). The markers are stripped from the text
before anyone sees or hears it and kept as char offsets ("pauses" in the
draft rows). When a row carries them, its units are cut THERE first — one
unit per English phrase — so every group boundary the DP can choose falls
on an English pause. The DP still groups neighbouring stretches when their
summed duration fits better (forcing strict 1-to-1 measured worse: single
phrases are too short for per-phrase speech-rate error to cancel).
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

_CLAUSE_SPLIT = re.compile(r"(?<=[,;:।॥.!?—])\s+")
UNIT_MIN_CHARS = 12          # merge fragments shorter than this
MAX_A = 4                    # units per group
MAX_B = 4                    # cues per group
SKIP_COST = 0.6              # base cost of an English phrase left undubbed
SHAPE_COST = 0.12            # mild preference for 1-1 groups
OVERRUN_WEIGHT = 2.0         # running past the pause after is worse
GAP_CAP_S = 1.5              # most of a following pause a group may use


LONG_UNIT_CHARS = 60         # ~5 s of Indic speech: split once more
HALF_MIN_CHARS = 20          # ...but only into halves at least this long


def _split_long(u: str) -> List[str]:
    """A clause longer than LONG_UNIT_CHARS is split at the word gap
    nearest its middle (recursively), so the aligner can put a pause where
    the English speaker pauses instead of running one long block."""
    if len(u) <= LONG_UNIT_CHARS:
        return [u]
    mid = len(u) // 2
    gaps = [i for i, ch in enumerate(u) if ch == " "
            and HALF_MIN_CHARS <= i <= len(u) - HALF_MIN_CHARS]
    if not gaps:
        return [u]
    cut = min(gaps, key=lambda i: abs(i - mid))
    return _split_long(u[:cut].strip()) + _split_long(u[cut:].strip())


def clause_units(text: str, min_chars: int = UNIT_MIN_CHARS,
                 pauses: Optional[Sequence[int]] = None) -> List[str]:
    """Split one paragraph into clause units; merge short fragments into
    their neighbour so no unit is a stray word; split over-long clauses.

    *pauses* (v0.20): char offsets into the whitespace-normalised text where
    the translator marked an English pause. The text is cut at those points
    first and each stretch is one unit (over-long ones still halved) — the
    stretches are NOT merged or re-cut at commas, because each one was
    written to be spoken during one English phrase. Offsets that do not
    fall on a word gap mean the text changed: ignored (normal split)."""
    norm = " ".join((text or "").split())
    cuts = sorted({int(p) for p in (pauses or [])
                   if isinstance(p, int) and 0 < p < len(norm)
                   and norm[p - 1] == " "})
    if cuts:
        segs, prev = [], 0
        for c in cuts:
            segs.append(norm[prev:c].strip())
            prev = c
        segs.append(norm[prev:].strip())
        out_m: List[str] = []
        for s in segs:
            if not s:
                continue
            # A stretch with no letters (a lone dash) rides with its neighbour.
            if out_m and not re.search(r"\w", s):
                out_m[-1] = out_m[-1] + " " + s
            elif out_m and not re.search(r"\w", out_m[-1]):
                out_m[-1] = out_m[-1] + " " + s
            else:
                out_m.append(s)
        return [piece for u in out_m for piece in _split_long(u)]
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(" ".join(
        (text or "").split())) if p.strip()]
    out: List[str] = []
    for p in parts:
        if out and (len(out[-1]) < min_chars or len(p) < min_chars):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1]) < min_chars:
        out[-2] = out[-2] + " " + out.pop()
    return [piece for u in out for piece in _split_long(u)]


def _overlap_frac(a0, a1, b0, b1) -> float:
    span = max(1e-6, a1 - a0)
    return max(0.0, min(a1, b1) - max(a0, b0)) / span


def align_units(unit_durs: Sequence[float], unit_rows: Sequence[int],
                cues: Sequence[Tuple[float, float]],
                row_windows: Sequence[Tuple[float, float]],
                row_hint: Sequence[float]) -> List[Tuple[int, int, int, int]]:
    """Monotonic duration alignment of units onto cues.

    unit_durs[i]  real seconds of unit i (from synthesis)
    unit_rows[i]  row (paragraph) index of unit i
    cues[j]       (start_s, end_s) English phrase, time-ordered
    row_windows[r], row_hint[r]  soft window prior and its weight per row
    Returns [(u_lo, u_hi, c_lo, c_hi)] covering every unit exactly once, in
    order; English cues left undubbed simply appear in no group. With no
    cues (or no feasible path) one group of all units with c_lo = -1."""
    n, m = len(unit_durs), len(cues)
    if n == 0:
        return []
    if m == 0:
        return [(0, n - 1, -1, -1)]
    INF = float("inf")
    band = max(40, m // 6) if n * m > 250_000 else None

    def _in_band(i, j):
        return band is None or abs(j - (i * m) / n) <= band

    def _gap_after(jb):
        return (max(0.0, min(GAP_CAP_S, cues[jb + 1][0] - cues[jb][1]))
                if jb + 1 < m else GAP_CAP_S)

    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur == INF:
                continue
            if j < m:                                   # undubbed phrase
                w = cues[j][1] - cues[j][0]
                c = cur + SKIP_COST + w
                if c < dp[i][j + 1]:
                    dp[i][j + 1], bp[i][j + 1] = c, (0, 1)
            if i == n:
                continue
            row = unit_rows[i]
            rw = row_windows[row] if 0 <= row < len(row_windows) else None
            hint = row_hint[row] if 0 <= row < len(row_hint) else 0.0
            d = 0.0
            for a in range(1, MAX_A + 1):
                k = i + a - 1
                if k >= n or unit_rows[k] != row:
                    break
                d += unit_durs[k]
                for b in range(1, MAX_B + 1):
                    jb = j + b - 1
                    if jb >= m:
                        break
                    if not _in_band(i + a, j + b):
                        continue
                    c0, c1 = cues[j][0], cues[jb][1]
                    win = max(0.05, c1 - c0)
                    if d <= win:
                        cost = win - d
                    else:
                        over, g = d - win, _gap_after(jb)
                        cost = over if over <= g else \
                            g + (over - g) * OVERRUN_WEIGHT
                    cost += SHAPE_COST * (a - 1 + b - 1)
                    if rw:
                        cost += hint * (1.0 - _overlap_frac(
                            c0, c1, rw[0] - 0.4, rw[1] + 0.4))
                    c = cur + cost
                    if c < dp[i + a][j + b]:
                        dp[i + a][j + b], bp[i + a][j + b] = c, (a, b)
    if dp[n][m] == INF:
        return [(0, n - 1, -1, -1)]
    groups: List[Tuple[int, int, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        a, b = bp[i][j]
        if a > 0:
            groups.append((i - a, i - 1, j - b, j - 1))
        i, j = i - a, j - b
    groups.reverse()
    return groups


# ─────────────────────────────────────────────────────────────────────────────
#  v0.20 — English phrase cues from word timestamps
# ─────────────────────────────────────────────────────────────────────────────

PHRASE_GAP_S = 0.30          # a word gap this long ends a phrase
PHRASE_MAX_S = 5.0           # no phrase runs longer than this
PHRASE_MIN_S = 0.8           # shorter phrases join a close neighbour...
PHRASE_MERGE_GAP_S = 0.6     # ...but never across a pause longer than this
PHRASE_ONSET_PULL_S = 0.6    # a region's first word may start this much late
_PC_SENT_END = re.compile(r"[.?!…][\"'”’)\]]*$")
_PC_SOFT_END = re.compile(r"[,;:—–][\"'”’)\]]*$")

Cue = Tuple[float, float, str]


def _pc_words(words) -> List[dict]:
    """Scribe word tokens with text, time-ordered — the same filter the
    subtitle builders use (type == "word"; spacing/audio events dropped)."""
    out = []
    for w in words or []:
        if w.get("type", "word") != "word":
            continue
        t = str(w.get("text", "") or "").strip()
        if not t:
            continue
        s = float(w.get("start", 0.0) or 0.0)
        e = float(w.get("end", s) or s)
        out.append({"text": t, "start": s, "end": max(s, e)})
    return out


def _pc_bucket(ws: List[dict], regions) -> List[List[dict]]:
    """Words into speech regions by _build_subtitle_srt's rule (a word
    belongs to the first region whose end is not before its start; the rest
    go to the last region), so no word is ever lost or duplicated. No
    regions -> one bucket."""
    if not regions:
        return [ws] if ws else []
    buckets: List[List[dict]] = [[] for _ in regions]
    wi = ri = 0
    while wi < len(ws) and ri < len(regions):
        if ws[wi]["start"] <= regions[ri][1]:
            buckets[ri].append(ws[wi])
            wi += 1
        else:
            ri += 1
    while wi < len(ws):
        buckets[-1].append(ws[wi])
        wi += 1
    # Scribe often starts a region's first word late (a soft onset reads as
    # silence to it); the loudness region knows better. Same rule as the
    # sync SRT builder: the first chunk of a region starts at the region —
    # but never before the previous word ends (Scribe may stretch a word
    # over the start of the next region), so cues never overlap.
    prev_end = float("-inf")
    for (r0, _r1), b in zip(regions, buckets):
        if not b:
            continue
        pulled = max(float(r0), prev_end)
        if pulled < b[0]["start"] <= r0 + PHRASE_ONSET_PULL_S:
            b[0]["start"] = pulled
        prev_end = b[-1]["end"]
    return [b for b in buckets if b]


def _pc_paused(a: dict, b: dict, pauses) -> bool:
    """A spectral pause lies between word a and word b. Scribe often lets a
    word's span run over the silence after it, so the test is generous at
    the edges (0.15 s) but the pause's middle must sit between the two
    words' middles — a dip inside one long word is not a phrase break."""
    lo, hi = a["end"] - 0.15, b["start"] + 0.15
    mid_a = (a["start"] + a["end"]) / 2.0
    mid_b = (b["start"] + b["end"]) / 2.0
    for ps, pe in pauses or ():
        if pe <= lo:
            continue
        if ps >= hi:
            break
        if min(pe, hi) - max(ps, lo) >= 0.08 and \
           mid_a < (ps + pe) / 2.0 < mid_b:
            return True
    return False


def _pc_split_long(ws: List[dict], max_s: float, min_s: float
                   ) -> List[List[dict]]:
    """Split a phrase longer than max_s at its best word boundary: the
    widest gap, a sentence/comma bonus, a mild pull toward the middle, and
    both halves at least min_s when possible. Recursive."""
    if len(ws) < 2 or ws[-1]["end"] - ws[0]["start"] <= max_s:
        return [ws]
    best, best_k = None, 1
    total = max(1.0, ws[-1]["end"] - ws[0]["start"])
    for k in range(1, len(ws)):
        left = ws[k - 1]["end"] - ws[0]["start"]
        right = ws[-1]["end"] - ws[k]["start"]
        score = max(0.0, ws[k]["start"] - ws[k - 1]["end"])
        if _PC_SENT_END.search(ws[k - 1]["text"]):
            score += 0.35
        elif _PC_SOFT_END.search(ws[k - 1]["text"]):
            score += 0.2
        score -= 0.4 * abs(left - right) / total
        if left < min_s or right < min_s:
            score -= 1.0
        if best is None or score > best:
            best, best_k = score, k
    return (_pc_split_long(ws[:best_k], max_s, min_s)
            + _pc_split_long(ws[best_k:], max_s, min_s))


def _pc_merge_short(groups: List[List[dict]], hard: List[bool], max_s: float,
                    min_s: float, merge_gap_s: float) -> List[List[dict]]:
    """Join phrases shorter than min_s to the closer neighbour — only across
    a short gap, never across a region boundary (*hard*[i] = a hard break
    before group i), and never into a phrase longer than max_s."""
    groups, hard = list(groups), list(hard)

    def dur(g):
        return g[-1]["end"] - g[0]["start"]

    changed = True
    while changed:
        changed = False
        for i in sorted(range(len(groups)), key=lambda i: dur(groups[i])):
            g = groups[i]
            if dur(g) >= min_s:
                break                                # sorted: none left
            cands = []
            if i > 0 and not hard[i]:
                gap = g[0]["start"] - groups[i - 1][-1]["end"]
                if gap < merge_gap_s and \
                   g[-1]["end"] - groups[i - 1][0]["start"] <= max_s:
                    cands.append((gap, i - 1))
            if i + 1 < len(groups) and not hard[i + 1]:
                gap = groups[i + 1][0]["start"] - g[-1]["end"]
                if gap < merge_gap_s and \
                   groups[i + 1][-1]["end"] - g[0]["start"] <= max_s:
                    cands.append((gap, i + 1))
            if not cands:
                continue
            j = min(cands)[1]
            lo, hi = min(i, j), max(i, j)
            groups[lo] = groups[lo] + groups[hi]
            del groups[hi]
            del hard[hi]
            changed = True
            break
    return groups


def _pc_cue(ws: List[dict]) -> Cue:
    return (round(ws[0]["start"], 3), round(ws[-1]["end"], 3),
            " ".join(w["text"] for w in ws))


def phrase_cues(words, regions=None, pauses=None, gap_s: float = PHRASE_GAP_S,
                max_s: float = PHRASE_MAX_S, min_s: float = PHRASE_MIN_S
                ) -> List[Cue]:
    """English phrase cues [(start_s, end_s, text)] from Scribe words.

    Every word token lands in exactly one cue, in order. A cue ends at a
    region boundary (loudness silence), a word gap >= gap_s, a spectral
    pause (*pauses* from spectral_vad), or after a sentence end; a cue
    longer than max_s is split at its best word boundary; one shorter than
    min_s joins a neighbour across a short gap. Times are the words' own
    Scribe times (first start, last end)."""
    groups: List[List[dict]] = []
    hard: List[bool] = []
    for bucket in _pc_bucket(_pc_words(words), regions):
        cur, first = [bucket[0]], True
        for a, b in zip(bucket, bucket[1:]):
            if (b["start"] - a["end"] >= gap_s or _pc_paused(a, b, pauses)
                    or _PC_SENT_END.search(a["text"])):
                for part in _pc_split_long(cur, max_s, min_s):
                    groups.append(part)
                    hard.append(first)
                    first = False
                cur = [b]
            else:
                cur.append(b)
        for part in _pc_split_long(cur, max_s, min_s):
            groups.append(part)
            hard.append(first)
            first = False
    groups = _pc_merge_short(groups, hard, max_s, min_s, PHRASE_MERGE_GAP_S)
    return [_pc_cue(g) for g in groups if g]


def cues_from_items(words, items, pauses=None, gap_s: float = PHRASE_GAP_S,
                    max_s: float = PHRASE_MAX_S, min_s: float = PHRASE_MIN_S
                    ) -> List[Cue]:
    """English cues from the user's own chunks on the timeline.

    *items*: [(start_s, length_s)] measured from the dubbed audio's 0:00.
    Each word goes to the item it overlaps most (the nearest one when it
    overlaps none), never moving backwards, so no word is lost or doubled.
    An item becomes one cue whose times are its words' times clamped to the
    item; an item longer than max_s is re-split by phrase_cues. Items with
    no words are dropped."""
    spans = sorted((float(s), float(s) + float(n))
                   for s, n in (items or []) if float(n) > 0)
    if not spans:
        return phrase_cues(words, None, pauses, gap_s, max_s, min_s)
    per: List[List[dict]] = [[] for _ in spans]
    k_prev = 0
    for w in _pc_words(words):
        best, best_k = None, k_prev
        for k in range(k_prev, len(spans)):
            s, e = spans[k]
            ov = min(e, w["end"]) - max(s, w["start"])
            # overlap wins (more is better); otherwise the nearest item
            key = -ov if ov > 0 else 1e3 + max(s - w["end"], w["start"] - e)
            if best is None or key < best:
                best, best_k = key, k
            if s > w["end"]:
                break                                # later items only farther
        per[best_k].append(w)
        k_prev = best_k
    out: List[Cue] = []
    for (s, e), grp in zip(spans, per):
        if not grp:
            continue
        if grp[-1]["end"] - grp[0]["start"] > max_s:
            out.extend(phrase_cues(grp, None, pauses, gap_s, max_s, min_s))
            continue
        c0 = min(max(grp[0]["start"], s), grp[0]["end"])
        c1 = max(min(grp[-1]["end"], e), grp[-1]["start"])
        out.append((round(c0, 3), round(max(c0, c1), 3),
                    " ".join(w["text"] for w in grp)))
    return out
