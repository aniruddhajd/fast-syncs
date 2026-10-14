"""
Skip pauses (v0.29.5) — cut the silent stretches out of one line's audio
=======================================================================
The Regenerate tab's "Shorter (skip pauses)" button. The line keeps the
SAME take — same words, same voice, nothing re-synthesized, no credits —
but every pause inside it is cut down to a short breath and the speech on
either side is joined with a tiny crossfade, so the line plays shorter.

  pauses    spectral_vad.spectral_pauses — relative to the file's own noise
            floor, so it works on quiet and noisy takes alike
  cut       each pause longer than SP_KEEP_S keeps SP_KEEP_S of silence
            (half after the speech before it, half before the speech after
            it) and loses the rest — the "join"; a zero-length join would
            clip consonant tails the VAD frames cannot see
  start     leading silence is KEPT: the chunk is placed by its position on
            the timeline, so moving the first word would break the sync
  end       trailing silence is trimmed to SP_TAIL_S after the last speech
  joins     SP_XFADE_MS crossfade at every cut, so no click at the seam

keep_spans() is pure numpy (unit-testable); skip_pauses_segment() applies
the spans to a pydub AudioSegment (any channel count / sample width).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .spectral_vad import SV_HOP_S, _sv_frames, spectral_pauses, speech_frames

SP_KEEP_S = 0.06       # silence left where a pause was (the join)
SP_MIN_PAUSE_S = 0.15  # shorter gaps are part of normal speech rhythm
SP_TAIL_S = 0.08       # silence kept after the last word
SP_XFADE_MS = 10       # crossfade at every cut


def keep_spans(y, sr: int, keep_s: float = SP_KEEP_S,
               min_pause_s: float = SP_MIN_PAUSE_S,
               tail_s: float = SP_TAIL_S) -> List[Tuple[float, float]]:
    """The [(start_s, end_s)] stretches of *y* to keep, time-ordered.

    No speech found -> the whole clip, unchanged (never return silence
    for a clip the detector could not read)."""
    y = np.asarray(y, dtype=np.float32).ravel()
    dur = y.size / float(sr) if sr else 0.0
    if dur <= 0:
        return []
    times, speech = speech_frames(y, sr)
    if not speech.any():
        return [(0.0, round(dur, 3))]
    last = int(speech.size - 1 - np.argmax(speech[::-1]))
    speech_end = float(times[last] + SV_HOP_S / 2.0)

    half = max(0.0, keep_s) / 2.0
    spans: List[Tuple[float, float]] = []
    cursor = 0.0
    for s, e in spectral_pauses(y, sr, min_pause_s=min_pause_s):
        if e - s <= keep_s + 0.01:
            continue                 # nothing worth cutting
        spans.append((cursor, s + half))
        cursor = e - half
    spans.append((cursor, min(dur, max(cursor, speech_end + tail_s))))
    return [(round(a, 3), round(b, 3)) for a, b in spans if b - a > 0.0005]


def skip_pauses_segment(seg, keep_s: float = SP_KEEP_S,
                        min_pause_s: float = SP_MIN_PAUSE_S,
                        tail_s: float = SP_TAIL_S,
                        xfade_ms: int = SP_XFADE_MS):
    """Cut the pauses out of a pydub AudioSegment.

    Returns (new_segment, info) where info = {"pauses_cut", "removed_s",
    "in_s", "out_s"}. With nothing to cut the segment comes back as-is
    (pauses_cut 0, removed_s possibly > 0 from the trailing trim only)."""
    mono = seg.set_channels(1) if seg.channels > 1 else seg
    y = np.array(mono.get_array_of_samples(), dtype=np.float32)
    y /= float(1 << (8 * mono.sample_width - 1))
    sr = int(seg.frame_rate)
    spans = keep_spans(y, sr, keep_s=keep_s, min_pause_s=min_pause_s,
                       tail_s=tail_s)
    in_s = len(seg) / 1000.0
    if not spans:
        return seg, {"pauses_cut": 0, "removed_s": 0.0,
                     "in_s": in_s, "out_s": in_s}

    out = None
    for a, b in spans:
        piece = seg[int(round(a * 1000)):int(round(b * 1000))]
        if out is None:
            out = piece
            continue
        xf = int(min(xfade_ms, len(out), len(piece)))
        out = out.append(piece, crossfade=xf) if xf > 0 else out + piece
    out_s = len(out) / 1000.0
    return out, {"pauses_cut": len(spans) - 1,
                 "removed_s": round(max(0.0, in_s - out_s), 3),
                 "in_s": round(in_s, 3), "out_s": round(out_s, 3)}


# ── v0.29.7 Longer (add pauses): the mirror — silence INTO the gaps ──────────
# v0.29.8: between WORDS too. TTS speech often runs words together with no
# 0.12 s pause anywhere, so pauses alone left most lines untouched. A word
# gap is a short silent dip (non-speech frames) or a deep energy valley
# (SP_WORD_DIP_DB under the speech level) — syllables inside a word dip far
# less. Real pauses still get the bigger share (SP_PAUSE_WEIGHT).
SP_ADD_MIN_PAUSE_S = 0.12      # a gap this long is a place a speaker breathes
SP_ADD_PER_PAUSE_S = 0.30      # default per pause when no total is asked for
SP_ADD_PER_WORD_S = 0.12       # default per word gap
SP_ADD_MAX_PER_PAUSE_S = 1.20  # never a gap that sounds like a dropout
SP_ADD_MAX_PER_WORD_S = 0.50
SP_PAUSE_WEIGHT = 2.0          # a pause takes twice a word gap's share
SP_WORD_DIP_DB = 12.0          # valley this far under the speech level
SP_WORD_SPACING_S = 0.25       # word gaps at least this far apart
SP_WORD_EDGE_S = 0.12          # never this close to the first/last speech
SP_WORD_FADE_MS = 6            # fade at a word cut (the valley is not silent)


def gap_points(y, sr: int, min_pause_s: float = SP_ADD_MIN_PAUSE_S
               ) -> List[Tuple[float, str]]:
    """[(t_s, 'pause'|'word')] — where silence can go, time-ordered.

    Pauses (spectral_pauses) at their middle; word gaps at the quietest
    frame of a dip between words. Nothing before the first or after the
    last speech, so the first word never moves."""
    y = np.asarray(y, dtype=np.float32).ravel()
    times, energy, _flat = _sv_frames(y, sr)
    _t, speech = speech_frames(y, sr)
    if times.size == 0 or not speech.any():
        return []
    pauses = spectral_pauses(y, sr, min_pause_s=min_pause_s)
    pts: List[Tuple[float, str]] = [(round((s + e) / 2.0, 3), "pause")
                                    for s, e in pauses]
    n = times.size
    first = int(np.argmax(speech))
    last = int(n - 1 - np.argmax(speech[::-1]))
    sm = np.convolve(energy, np.ones(3) / 3.0, mode="same")
    voiced = sm[first:last + 1][speech[first:last + 1]]
    if voiced.size == 0:
        return sorted(pts)
    level = float(np.percentile(voiced, 90))
    w = max(1, int(round(0.06 / SV_HOP_S)))           # +-60 ms local minimum
    edge = int(round(SP_WORD_EDGE_S / SV_HOP_S))
    for i in range(first + edge, last - edge + 1):
        t = float(times[i])
        if any(s <= t <= e for s, e in pauses):
            continue                                   # already a pause point
        lo, hi = max(0, i - w), min(n, i + w + 1)
        if sm[i] > sm[lo:hi].min():
            continue
        if speech[i] and sm[i] > level - SP_WORD_DIP_DB:
            continue                                   # a syllable, not a gap
        if all(abs(t - p) >= SP_WORD_SPACING_S for p, _k in pts):
            pts.append((round(t, 3), "word"))
    return sorted(pts)


def add_pauses_segment(seg, add_s: float = 0.0,
                       min_pause_s: float = SP_ADD_MIN_PAUSE_S,
                       per_pause_s: float = SP_ADD_PER_PAUSE_S,
                       max_per_pause_s: float = SP_ADD_MAX_PER_PAUSE_S,
                       words: bool = True):
    """Put silence between the words (and into the pauses) of a pydub
    AudioSegment.

    *add_s* > 0: that much silence in total, shared by the gap points with a
    pause weighing SP_PAUSE_WEIGHT word gaps (each capped); 0: the per-kind
    defaults. A word cut gets a SP_WORD_FADE_MS fade on both sides, since a
    valley is quiet but not silent. *words* False = pauses only (v0.29.7).

    Returns (new_segment, info) with info = {"pauses" (all gap points),
    "word_gaps", "added_s", "in_s", "out_s"}; nowhere to put silence -> the
    segment unchanged, pauses 0."""
    from .config import _AudioSegment
    mono = seg.set_channels(1) if seg.channels > 1 else seg
    y = np.array(mono.get_array_of_samples(), dtype=np.float32)
    y /= float(1 << (8 * mono.sample_width - 1))
    sr = int(seg.frame_rate)
    pts = gap_points(y, sr, min_pause_s=min_pause_s)
    if not words:
        pts = [p for p in pts if p[1] == "pause"]
    in_s = round(len(seg) / 1000.0, 3)
    none = {"pauses": 0, "word_gaps": 0, "added_s": 0.0, "in_s": in_s,
            "out_s": in_s}
    if not pts:
        return seg, none

    def _cap(kind):
        return max_per_pause_s if kind == "pause" else SP_ADD_MAX_PER_WORD_S

    if add_s and add_s > 0:
        wsum = sum(SP_PAUSE_WEIGHT if k == "pause" else 1.0 for _t, k in pts)
        unit = add_s / wsum
        each = [min(unit * (SP_PAUSE_WEIGHT if k == "pause" else 1.0), _cap(k))
                for _t, k in pts]
    else:
        each = [min(per_pause_s if k == "pause" else SP_ADD_PER_WORD_S, _cap(k))
                for _t, k in pts]

    out, cursor, fade_in = None, 0, False
    for (t, kind), e_s in zip(pts, each):
        ms = int(round(max(0.0, e_s) * 1000))
        if ms < 10:
            continue
        at = int(round(t * 1000))
        piece = seg[cursor:at]
        if fade_in and len(piece) > SP_WORD_FADE_MS:
            piece = piece.fade_in(SP_WORD_FADE_MS)
        fade_in = kind == "word"
        if fade_in and len(piece) > SP_WORD_FADE_MS:
            piece = piece.fade_out(SP_WORD_FADE_MS)
        gap = (_AudioSegment.silent(duration=ms, frame_rate=sr)
               .set_channels(seg.channels).set_sample_width(seg.sample_width))
        out = (piece + gap) if out is None else (out + piece + gap)
        cursor = at
    if out is None:
        return seg, none
    tail = seg[cursor:]
    if fade_in and len(tail) > SP_WORD_FADE_MS:
        tail = tail.fade_in(SP_WORD_FADE_MS)
    out = out + tail
    out_s = round(len(out) / 1000.0, 3)
    return out, {"pauses": len(pts),
                 "word_gaps": sum(1 for _t, k in pts if k == "word"),
                 "added_s": round(out_s - in_s, 3),
                 "in_s": in_s, "out_s": out_s}
