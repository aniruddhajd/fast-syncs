"""
Sync check agent (v0.18) — verify and tighten a placed dub
==========================================================
Runs after placement (anchor sync, and match sync at sentence/clause piece
size). It is deterministic on purpose: whether a line starts on time is
arithmetic, and arithmetic should not cost a model call or vary between
runs.

For every piece that has an English window (win = (start_s, end_s)):

  offset = placed start - English speech start

  1. Late  (offset > tol): pull it earlier, down to the English start, as
     far as the previous piece's end allows.
  2. Early (offset < -tol): push it later, up to the English start, as far
     as the next piece's start allows (never overlaps).
  3. Rescue: an Un sync piece whose window has room between its synced
     neighbours (script order kept, no overlap) is put back on the timeline
     at max(English start, previous end) — if that lands within 2 x tol.
  4. Report: pieces within tolerance, moved, rescued, still drifting, mean
     and max |offset|, and overlaps (there should be none).

Moves never reorder pieces and never create an overlap, so the result is
always at least as good as the placement it was given.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

SYNC_TOL_S = 0.30       # |offset| a listener does not notice as late/early
SYNC_MIN_GAP_S = 0.05   # breathing room kept between consecutive pieces


def sync_check(pieces: Sequence[dict], placed: Sequence[dict],
               durations: Sequence[float], audio_end: float = 0.0,
               tol: float = SYNC_TOL_S, log=None
               ) -> Tuple[List[dict], dict]:
    """(placed', report). *placed* items are {"position", "status"} as
    returned by place_pieces; a new list is returned, the input is not
    modified."""
    say = log or (lambda _m: None)
    out = [dict(p) for p in placed]
    n = len(out)
    dur = [max(0.0, float(d)) for d in durations]
    win = [p.get("win") for p in pieces]

    def _synced(i):
        return out[i]["status"] == "synced"

    def _prev_end(i):
        for j in range(i - 1, -1, -1):
            if _synced(j):
                return out[j]["position"] + dur[j] + SYNC_MIN_GAP_S
        return 0.0

    def _next_start(i):
        for j in range(i + 1, n):
            if _synced(j):
                return out[j]["position"] - SYNC_MIN_GAP_S
        return float("inf") if audio_end <= 0 else audio_end + 2.0

    moved = rescued = 0
    # ── 3. Rescue first, so the tightening pass sees the final neighbours.
    for i in range(n):
        if _synced(i) or not win[i]:
            continue
        a = float(win[i][0])
        start = max(a, _prev_end(i))
        if start - a <= 2 * tol and start + dur[i] <= _next_start(i):
            out[i] = {"position": round(start, 6), "status": "synced"}
            rescued += 1
            say(f"  [CHECK] piece{i + 1} rescued from Un sync -> "
                f"{start:.2f}s (English at {a:.2f}s)")

    # ── 1-2. Tighten late / early pieces, in script (= timeline) order.
    for i in range(n):
        if not _synced(i) or not win[i]:
            continue
        a = float(win[i][0])
        pos = out[i]["position"]
        off = pos - a
        new = pos
        if off > tol:
            new = max(a, _prev_end(i))
        elif off < -tol:
            new = min(a, _next_start(i) - dur[i])
        if abs(new - pos) > 0.005 and abs(new - a) < abs(off):
            out[i]["position"] = round(new, 6)
            moved += 1
            say(f"  [CHECK] piece{i + 1} {'late' if off > 0 else 'early'} "
                f"{off:+.2f}s -> {new - a:+.2f}s")

    # ── 4. Report.
    offs, drift, overlaps = [], [], 0
    ends, end_drift = [], []
    for i in range(n):
        if not _synced(i):
            continue
        if win[i]:
            o = out[i]["position"] - float(win[i][0])
            offs.append(abs(o))
            if abs(o) > tol:
                drift.append((i + 1, round(o, 2)))
            # End alignment: a piece that starts on time but stops long
            # before (or after) its English has drifted INSIDE the piece —
            # the start-only check used to call that "on time".
            eo = out[i]["position"] + dur[i] - float(win[i][1])
            ends.append(abs(eo))
            if abs(eo) > max(1.0, 3 * tol):
                end_drift.append((i + 1, round(eo, 2)))
        nxt = _next_start(i) + SYNC_MIN_GAP_S
        if out[i]["position"] + dur[i] > nxt + 0.001:
            overlaps += 1
    synced = sum(1 for i in range(n) if _synced(i))
    report = {
        "pieces": n, "synced": synced, "unsync": n - synced,
        "within_tol": sum(1 for o in offs if o <= tol),
        "moved": moved, "rescued": rescued, "overlaps": overlaps,
        "mean_offset_ms": int(round(1000 * sum(offs) / len(offs))) if offs else 0,
        "max_offset_ms": int(round(1000 * max(offs))) if offs else 0,
        "drift": drift, "tol_ms": int(round(tol * 1000)),
        "mean_end_offset_ms": int(round(1000 * sum(ends) / len(ends))) if ends else 0,
        "end_drift": end_drift,
    }
    return out, report


def format_sync_check(report: dict) -> str:
    """Human-readable sync check report (<base>_sync_check.txt)."""
    lines = [
        "SYNC CHECK",
        f"pieces {report['pieces']}  synced {report['synced']}  "
        f"unsync {report['unsync']}",
        f"on time (|offset| <= {report['tol_ms']} ms): "
        f"{report['within_tol']} of {report['synced']}",
        f"mean |offset| {report['mean_offset_ms']} ms   "
        f"max |offset| {report['max_offset_ms']} ms",
        f"tightened {report['moved']}   rescued from Un sync "
        f"{report['rescued']}   overlaps {report['overlaps']}",
    ]
    lines.append(f"mean |end offset| {report.get('mean_end_offset_ms', 0)} ms "
                 "(dub end vs English phrase end)")
    if report["drift"]:
        lines.append("")
        lines.append("Start still off by more than the tolerance (piece: "
                     "seconds, + = late):")
        lines += [f"  piece {i}: {o:+.2f}s" for i, o in report["drift"]]
    if report.get("end_drift"):
        lines.append("")
        lines.append("End far from the English phrase end (piece: seconds, "
                     "- = finishes early, + = runs over):")
        lines += [f"  piece {i}: {o:+.2f}s" for i, o in report["end_drift"]]
    return "\n".join(lines) + "\n"
