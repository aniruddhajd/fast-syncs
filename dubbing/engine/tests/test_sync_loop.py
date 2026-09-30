#!/usr/bin/env python3
"""v0.28 sync feedback loops — offline, no network, no audio.

  * the self-correcting run (_sync_loop): a better round is kept, a worse
    one is dropped and undone, round 2 re-voices only when allowed
  * learning the start bias from the user's final timeline (P: lines,
    _sync_pieces.json, the per-language sync profile)
  * the measured speaking speed

    python -m unittest dubbing/engine/tests/test_sync_loop.py -v
"""

import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="dub_syncloop_")
os.environ.setdefault("DUB_STATUS_DIR", os.path.join(_TMP, "status"))
os.environ.setdefault("AI_LEARNING_DIR", os.path.join(_TMP, "ai_learning"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import dub_engine as de                               # noqa: E402
from pipeline import sync_check as sc                 # noqa: E402
from pipeline import sync_learn as sl                 # noqa: E402


class Seg:
    """Stands in for a pydub segment: only its length (ms) matters."""

    def __init__(self, ms):
        self.ms = ms

    def __len__(self):
        return int(self.ms)


def _place(pieces, durs, log=None):
    """Deterministic placement: at the target, never before the previous
    piece has ended (+50 ms)."""
    out, end = [], -1.0
    for p, d in zip(pieces, durs):
        pos = max(float(p["win"][0]), end + 0.05)
        out.append({"position": round(pos, 6), "status": "synced"})
        end = pos + d
    return out


def _fake_pl():
    return types.SimpleNamespace(
        place_pieces=_place, sync_check=sc.sync_check,
        loop_score=sl.loop_score, loop_offenders=sl.loop_offenders,
        loop_better=sl.loop_better)


WINS = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
PIECES = [{"win": w, "en": "e", "text": "t"} for w in WINS]
MAX_AT = 1.25


def _slot(i):
    return WINS[i][1] - WINS[i][0]


def _make_render(segs):
    def render(cap, lead):
        durs, ratios = [], []
        for i, sg in enumerate(segs):
            d = len(sg) / 1000.0
            slot = _slot(i) + lead.get(i, 0.0)
            r = min(d / slot, cap.get(i, MAX_AT)) if d > slot else 1.0
            durs.append(round(d / r, 6))
            ratios.append(r)
        return (None, [], durs, ratios)
    return render


def _run(segs, texts, fit_retry=None, allow=False, render=None):
    ctx = {"base": os.path.join(tempfile.mkdtemp(dir=_TMP), "run"),
           "en_audio_dur": 6.0}
    render = render or _make_render(segs)
    first = render({}, {})
    placed0 = _place(PIECES, first[2])
    changes = []
    out = de._sync_loop(_fake_pl(), ctx, PIECES, segs, texts, changes,
                        first, placed0, render,
                        fit_retry or (lambda todo, b: []), _slot, MAX_AT,
                        allow_revoice=allow)
    return out, placed0, ctx, changes


def _loop_file(ctx):
    with open(ctx["base"] + "_sync_loop.txt", encoding="utf-8") as f:
        return f.read()


class SyncLoop(unittest.TestCase):
    def test_free_round_fixes_a_late_piece_and_is_kept(self):
        segs = [Seg(1900), Seg(2900), Seg(1900)]
        (_r, placed), placed0, ctx, _ = _run(segs, ["a", "b", "c"])
        # first pass: piece 3 starts 0.37 s late (over the 0.3 s tolerance)
        self.assertGreater(placed0[2]["position"] - 4.0, 0.3)
        self.assertLessEqual(placed[2]["position"] - 4.0, 0.3)
        self.assertIn("kept", _loop_file(ctx))

    def test_worse_round_is_dropped(self):
        segs = [Seg(1900), Seg(2900), Seg(1900)]
        good = _make_render(segs)

        def worse(cap, lead):                       # relaxing makes it worse
            r = good({}, {})
            if lead:
                return (None, [], [d + 1.0 for d in r[2]], r[3])
            return r
        (_r, placed), placed0, ctx, _ = _run(segs, ["a", "b", "c"],
                                             render=worse)
        self.assertEqual(placed, placed0)
        self.assertIn("dropped", _loop_file(ctx))

    def test_round_two_revoices_the_long_piece_before(self):
        segs = [Seg(1900), Seg(4000), Seg(1900)]
        texts = ["a", "b long", "c"]
        seen = []

        def fit_retry(todo, budget_of):
            seen.extend(todo)
            for i in todo:
                segs[i] = Seg(1800)
                texts[i] = "b"
            return list(todo)
        (_r, placed), _p0, _ctx, _ = _run(segs, texts, fit_retry, allow=True)
        self.assertEqual(seen, [1])                  # the piece BEFORE
        self.assertLessEqual(placed[2]["position"] - 4.0, 0.3)
        self.assertEqual(texts[1], "b")

    def test_round_two_never_rewords_when_not_allowed(self):
        segs = [Seg(1900), Seg(4000), Seg(1900)]
        called = []
        _run(segs, ["a", "b", "c"], lambda t, b: called.append(t) or [],
             allow=False)
        self.assertEqual(called, [])

    def test_loop_off_returns_the_first_pass(self):
        segs = [Seg(1900), Seg(2900), Seg(1900)]
        with mock.patch.object(de, "_engine_setting",
                               side_effect=lambda k, d, *a, **kw:
                               0 if k == "sync_loop" else d):
            (_r, placed), placed0, _c, _ = _run(segs, ["a", "b", "c"])
        self.assertEqual(placed, placed0)


class _Isolated(unittest.TestCase):
    def setUp(self):
        self._old = sl.AI_LEARNING_DIR
        sl.AI_LEARNING_DIR = tempfile.mkdtemp(dir=_TMP)

    def tearDown(self):
        sl.AI_LEARNING_DIR = self._old


class BiasLearning(_Isolated):
    def test_merge_median_clamp_and_min_run(self):
        prof = sl._empty("Marathi")
        self.assertEqual(sl.merge_bias(prof, [0.1] * 5), (False, 5))
        ok, n = sl.merge_bias(prof, [0.1] * 10 + [5.0])   # outlier dropped
        self.assertTrue(ok)
        self.assertEqual(n, 10)
        self.assertAlmostEqual(prof["start_bias_s"], 0.1)
        sl.merge_bias(prof, [1.4] * 1000)                  # clamped
        self.assertLessEqual(prof["start_bias_s"], sl.BIAS_CLAMP_S)

    def test_bias_needs_enough_samples(self):
        prof = sl._empty("Marathi")
        sl.merge_bias(prof, [0.12] * 10)
        sl.save_sync_profile(prof)
        self.assertEqual(sl.learned_bias("Marathi")[0], 0.0)   # 10 < 30
        sl.merge_bias(prof, [0.12] * 25)
        sl.save_sync_profile(prof)
        b, n = sl.learned_bias("Marathi")
        self.assertAlmostEqual(b, 0.12)
        self.assertEqual(n, 35)

    def test_bias_shifts_windows_only(self):
        out = sl.bias_windows([{"win": (1.0, 2.0), "text": "x"},
                               {"win": None}], 0.1)
        self.assertEqual(out[0]["win"], (1.1, 2.1))
        self.assertIsNone(out[1]["win"])

    def test_parse_p_lines_and_old_requests(self):
        txt = ("BASE: /x/a\nC: 1.0|2.0|नमस्कार|x\n"
               "P: 3|4.100|2.000|0.000\nP: bad\nP: 0|1|1|0\n")
        self.assertEqual(de._parse_learn_pieces(txt), [(3, 4.1, 2.0, 0.0)])
        self.assertEqual(de._parse_learn_pieces("BASE: /x/a\nC: 1|2|t\n"), [])
        _base, chunks = de._parse_learn_request(txt)    # C: parser unchanged
        self.assertEqual(len(chunks), 1)

    def test_learn_sync_final_end_to_end(self):
        d = tempfile.mkdtemp(dir=_TMP)
        base = os.path.join(d, "talk_Marathi")
        de._write_sync_pieces(base, [(i * 2.0, i * 2.0 + 1.5)
                                     for i in range(10)],
                              ["t"] * 10, [1.5] * 10,
                              [{"position": i * 2.0, "status": "synced"}
                               for i in range(10)])
        # the user left every clip 150 ms late, on a region starting at 60 s
        req = "BASE: %s\n" % base + "".join(
            "P: %d|%.3f|1.500|60.000\n" % (i + 1, 60 + i * 2.0 + 0.15)
            for i in range(10))
        line = de._learn_sync_final(base, req, "Marathi")
        self.assertIn("learned timing from 10 clip(s)", line)
        self.assertIn("+150 ms", line)
        prof = sl.load_sync_profile("Marathi")
        self.assertAlmostEqual(prof["start_bias_s"], 0.15, places=3)
        self.assertIn("already learned",
                      de._learn_sync_final(base, req, "Marathi"))

    def test_learn_sync_final_old_run(self):
        d = tempfile.mkdtemp(dir=_TMP)
        line = de._learn_sync_final(os.path.join(d, "x"),
                                    "P: 1|1.0|1.0|0\n", "Marathi")
        self.assertIn("before v0.28", line)


class SpeedLearning(_Isolated):
    def test_merge_and_lookup(self):
        prof = sl._empty("Hindi")
        self.assertFalse(sl.merge_cps(prof, "v1", [(24, 2.0)]))  # too few
        self.assertTrue(sl.merge_cps(prof, "v1",
                                     [(24, 2.0), (36, 3.0), (12, 1.0)] * 6))
        sl.save_sync_profile(prof)
        self.assertAlmostEqual(sl.learned_cps("Hindi", "v1"), 12.0)
        self.assertAlmostEqual(sl.learned_cps("Hindi"), 12.0)
        self.assertIsNone(sl.learned_cps("Tamil"))                # nothing

    def test_engine_learns_raw_speed_not_stretched(self):
        args = types.SimpleNamespace(script_source="ai_test",
                                     language="Hindi")
        pl = types.SimpleNamespace(load_sync_profile=sl.load_sync_profile,
                                   merge_cps=sl.merge_cps,
                                   save_sync_profile=sl.save_sync_profile)
        texts = ["x" * 24] * 6
        durs = [1.6] * 6                 # 2.0 s of speech stretched 1.25x
        de._learn_speed(pl, args, texts, durs, [1.25] * 6,
                        [{"status": "synced"}] * 6, None, "v9")
        self.assertAlmostEqual(sl.learned_cps("Hindi", "v9",
                                              min_seconds=1.0), 12.0)

    def test_prompt_agents_never_learn_speed(self):
        args = types.SimpleNamespace(script_source="prompt_agents",
                                     language="Odia")
        de._learn_speed(None, args, [], [], [], [], None, "v")
        self.assertFalse(os.path.exists(sl.sync_profile_path("Odia")))


if __name__ == "__main__":
    unittest.main()
