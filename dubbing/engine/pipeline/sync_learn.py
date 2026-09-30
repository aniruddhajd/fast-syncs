"""
Sync feedback loops (v0.28) — the small, pure half
===================================================
Two loops close around the dub stage's sync:

1. A SELF-CORRECTING RUN (dub_engine._sync_loop): after the first placement
   the sync check's report says which pieces are still off; the engine
   relaxes only those, places again and keeps the round only if it is
   better. The scoring and the "who is off" list live here.

2. LEARNING FROM THE USER'S TIMELINE, per language:
   * start bias — where the user drags lines relative to the English
     (pressed "Learn from final dub"): lead = final start - English start.
     The run's median merges into a weighted running mean, clamped, and is
     applied to future placements only after enough samples.
   * speaking speed — measured on EVERY run from the real TTS: characters
     per second of each synced piece before stretching. Used to size the
     fit retry's character budget.

Storage: <AI_LEARNING_DIR>/sync/<lang_key>.json — a separate file from the
text profile, so a bad timing sample can never touch the translation style.
Every read/write is fail-open: a broken file reads as empty, a failed write
is ignored by the caller.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import List, Optional, Sequence, Tuple

from .ai_translator import AI_LEARNING_DIR, _lang_key

SYNC_PROFILE_VERSION = 1
BIAS_MIN_SAMPLES = 30        # samples before a learned bias is applied
BIAS_MIN_RUN = 8             # pieces a single run needs to count
BIAS_OUTLIER_S = 1.5         # |lead| beyond this is a moved-far-away clip
BIAS_CLAMP_S = 0.4           # the learned bias never exceeds +/- this
BIAS_WEIGHT_CAP = 500        # running mean weight cap (keeps it adaptable)
CPS_WEIGHT_CAP = 2000.0      # seconds of speech behind a learned speed
CPS_MIN, CPS_MAX = 4.0, 30.0  # plausible characters-per-second range


# ─── storage ─────────────────────────────────────────────────────────────────

def sync_profile_path(language: str) -> str:
    return os.path.join(AI_LEARNING_DIR, "sync", _lang_key(language) + ".json")


def _empty(language: str) -> dict:
    return {"version": SYNC_PROFILE_VERSION, "language": language,
            "start_bias_s": 0.0, "bias_samples": 0,
            "cps": 0.0, "cps_seconds": 0.0, "voices": {},
            "runs": 0, "updated": ""}


def load_sync_profile(language: str) -> dict:
    """The stored profile merged over an empty one; unknown keys kept."""
    prof = _empty(language)
    try:
        with open(sync_profile_path(language), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            prof.update(data)
    except Exception:
        pass
    if not isinstance(prof.get("voices"), dict):
        prof["voices"] = {}
    return prof


def save_sync_profile(prof: dict) -> str:
    """Atomic write (temp file + os.replace). Returns the path."""
    path = sync_profile_path(prof.get("language") or "")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prof["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


# ─── start bias (learned from the user's final timeline) ────────────────────

def merge_bias(prof: dict, leads: Sequence[float]) -> Tuple[bool, int]:
    """Fold one run's leads (seconds, final start - English start) into
    *prof*. Returns (changed, samples used). Outliers are dropped; a run
    with too few pieces teaches nothing."""
    use = [float(x) for x in leads if abs(float(x)) <= BIAS_OUTLIER_S]
    if len(use) < BIAS_MIN_RUN:
        return False, len(use)
    run = statistics.median(use)
    n_old = max(0, int(prof.get("bias_samples") or 0))
    w_old = min(n_old, BIAS_WEIGHT_CAP)
    old = float(prof.get("start_bias_s") or 0.0)
    new = (old * w_old + run * len(use)) / float(w_old + len(use))
    prof["start_bias_s"] = round(max(-BIAS_CLAMP_S, min(BIAS_CLAMP_S, new)), 4)
    prof["bias_samples"] = n_old + len(use)
    return True, len(use)


def learned_bias(language: str,
                 min_samples: int = BIAS_MIN_SAMPLES) -> Tuple[float, int]:
    """(bias seconds, samples) — 0.0 until *min_samples* are learned."""
    prof = load_sync_profile(language)
    n = int(prof.get("bias_samples") or 0)
    if n < min_samples:
        return 0.0, n
    b = float(prof.get("start_bias_s") or 0.0)
    return max(-BIAS_CLAMP_S, min(BIAS_CLAMP_S, b)), n


def bias_windows(pieces: List[dict], bias_s: float) -> List[dict]:
    """Copies of *pieces* with every window shifted by *bias_s*."""
    if not bias_s:
        return pieces
    out = []
    for p in pieces:
        q = dict(p)
        w = p.get("win")
        if w:
            q["win"] = (float(w[0]) + bias_s, float(w[1]) + bias_s)
        out.append(q)
    return out


def leads_from_final(piece_rows: Sequence[dict],
                     final: Sequence[Tuple[int, float, float, float]]
                     ) -> List[float]:
    """Leads for the final clips that are engine pieces.
    *piece_rows*: the run's <base>_sync_pieces.json rows ({idx, win}).
    *final*: (piece index, timeline start, length, region offset) from the
    learn request — the offset is what the importer added to every time."""
    wins = {}
    for r in piece_rows:
        try:
            if isinstance(r, dict) and r.get("win"):
                wins[int(r["idx"])] = r["win"]
        except Exception:
            pass
    out = []
    for idx, pos, _ln, off in final:
        w = wins.get(int(idx))
        if w:
            out.append(round(float(pos) - float(off) - float(w[0]), 4))
    return out


# ─── speaking speed (measured on every run) ─────────────────────────────────

def merge_cps(prof: dict, voice: Optional[str],
              samples: Sequence[Tuple[int, float]]) -> bool:
    """Fold (characters, raw seconds) samples into the language speed and
    the voice's own speed. Returns True when anything was learned."""
    good = [(c, s) for c, s in samples if s and s > 0.3 and c > 0
            and CPS_MIN <= c / s <= CPS_MAX]
    if len(good) < 3:
        return False
    run = statistics.median(c / s for c, s in good)
    secs = float(sum(s for _c, s in good))

    def fold(node):
        w_old = min(float(node.get("cps_seconds") or 0.0), CPS_WEIGHT_CAP)
        old = float(node.get("cps") or 0.0)
        node["cps"] = round((old * w_old + run * secs) / (w_old + secs), 3) \
            if w_old > 0 and old > 0 else round(run, 3)
        node["cps_seconds"] = round(float(node.get("cps_seconds") or 0.0)
                                    + secs, 2)

    fold(prof)
    if voice:
        fold(prof["voices"].setdefault(str(voice), {}))
    return True


def learned_cps(language: str, voice: Optional[str] = None,
                min_seconds: float = 30.0) -> Optional[float]:
    """The learned characters/second for *voice* (else the language), or
    None until enough speech has been measured."""
    prof = load_sync_profile(language)
    for node in ((prof["voices"].get(str(voice)) if voice else None), prof):
        if isinstance(node, dict) and \
                float(node.get("cps_seconds") or 0.0) >= min_seconds and \
                CPS_MIN <= float(node.get("cps") or 0.0) <= CPS_MAX:
            return float(node["cps"])
    return None


# ─── self-correcting run: who is off, and how bad is it ─────────────────────

def loop_offenders(report: dict, placed: Sequence[dict]) -> List[int]:
    """0-based indices of pieces still off: Un sync, start drift, end drift."""
    bad = {i for i, p in enumerate(placed) if p.get("status") != "synced"}
    for key in ("drift", "end_drift"):
        for item in report.get(key) or ():
            try:
                bad.add(int(item[0]) - 1)
            except Exception:
                pass
    return sorted(i for i in bad if 0 <= i < len(placed))


def loop_score(report: dict, placed: Sequence[dict], tol_s: float) -> float:
    """Lower is better: 1000 per Un sync piece, + ms of start offset beyond
    the tolerance, + half the ms of end offset beyond the end threshold
    (the sync check's max(1 s, 3 x tol)) — a piece that starts on time but
    runs long past its English counts too, just less — + a quarter of the
    summed start offset of every synced piece."""
    unsync = sum(1 for p in placed if p.get("status") != "synced")
    over = sum(max(0.0, abs(float(off)) - tol_s)
               for _n, off in (report.get("drift") or ()))
    end_tol = max(1.0, 3.0 * tol_s)
    end_over = sum(max(0.0, abs(float(off)) - end_tol)
                   for _n, off in (report.get("end_drift") or ()))
    # every synced piece's start offset counts a little, so a round cannot
    # trade on-time starts for better ends unnoticed
    all_start = float(report.get("mean_offset_ms") or 0) *         float(report.get("synced") or 0) / 1000.0
    return (unsync * 1000.0 + over * 1000.0 + end_over * 500.0
            + all_start * 250.0)


def loop_better(new_score: float, new_bad: Sequence[int],
                old_score: float, old_bad: Sequence[int]) -> bool:
    """A round is kept when it scores lower, or scores the same with fewer
    pieces still off."""
    return (new_score < old_score - 1e-6
            or (abs(new_score - old_score) <= 1e-6
                and len(new_bad) < len(old_bad)))
