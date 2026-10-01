#!/usr/bin/env python3
"""v0.28.2 Speed up removes pauses — offline (synthetic tones, no network).

  * long silences inside a line (and at its end) shrink to a short breath
  * the opening silence and the speech itself are left alone
  * --tighten-pauses writes the shortened clips and the T: answer lines

    python -m unittest dubbing/engine/tests/test_tighten_pauses.py -v
"""

import os
import pathlib
import sys
import tempfile
import types
import unittest

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="dub_tighten_")
os.environ.setdefault("DUB_STATUS_DIR", os.path.join(_TMP, "status"))
os.environ.setdefault("AI_LEARNING_DIR", os.path.join(_TMP, "ai_learning"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import dub_engine as de                               # noqa: E402

try:
    from pydub import AudioSegment
    from pydub.generators import Sine
    HAVE_PYDUB = True
except Exception:                                     # noqa: BLE001
    HAVE_PYDUB = False


def _line():
    """0.3 s lead-in, speech 1 s, pause 0.8 s, speech 1 s, tail 0.6 s."""
    tone = Sine(220).to_audio_segment(duration=1000).apply_gain(-6)
    gap = AudioSegment.silent
    return (gap(duration=300) + tone + gap(duration=800) + tone
            + gap(duration=600))


@unittest.skipUnless(HAVE_PYDUB, "pydub not installed")
class Tighten(unittest.TestCase):
    def test_inner_and_tail_pauses_shrink_speech_kept(self):
        seg = _line()
        out, cut = de._tighten_segment(seg, min_ms=250, keep_ms=120)
        # 800 -> 120 and 600 -> 120: about 1160 ms gone
        self.assertGreater(cut, 1000)
        self.assertLess(cut, 1300)
        self.assertAlmostEqual(len(out), len(seg) - cut, delta=5)
        # the opening silence stays, so the speech starts where it did
        self.assertLess(out[:290].dBFS, -50)
        self.assertGreater(out[300:400].dBFS, -20)

    def test_line_without_long_pauses_is_untouched(self):
        tone = Sine(220).to_audio_segment(duration=2000)
        out, cut = de._tighten_segment(tone, min_ms=250, keep_ms=120)
        self.assertEqual(cut, 0)
        self.assertEqual(len(out), len(tone))

    def test_switch_off(self):
        orig = de._engine_setting
        de._engine_setting = (lambda k, d, *a, **kw:
                              0 if k == "speed_tighten" else d)
        try:
            _out, cut = de._tighten_segment(_line())
        finally:
            de._engine_setting = orig
        self.assertEqual(cut, 0)

    def test_job_writes_clips_and_answer(self):
        d = tempfile.mkdtemp(dir=_TMP)
        src = os.path.join(d, "tts.wav")
        (AudioSegment.silent(duration=1000) + _line()).export(src,
                                                               format="wav")
        req = os.path.join(d, "_tighten.txt")
        with open(req, "w", encoding="utf-8") as f:
            f.write(f"OUT: {os.path.join(d, 'regen')}\n"
                    f"T: {{AAA}}|1.000|{len(_line()) / 1000:.3f}|{src}\n")
        man = {}
        de._run_tighten_pauses(types.SimpleNamespace(text_file=req), man)
        self.assertEqual(man["pause_count"], "1")
        with open(man["pause_txt"], encoding="utf-8") as f:
            line = f.read().strip()
        key, new_len, _cut, wav = line[2:].strip().split("|", 3)
        self.assertEqual(key, "{AAA}")
        self.assertTrue(os.path.isfile(wav))
        self.assertLess(float(new_len), len(_line()) / 1000.0 - 1.0)


if __name__ == "__main__":
    unittest.main()
