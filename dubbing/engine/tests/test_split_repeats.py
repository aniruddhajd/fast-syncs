#!/usr/bin/env python3
"""v0.28.4 split clips never put a sentence in twice — offline.

A clip split (S) in REAPER keeps its whole text on both halves. Learning
from the final dub must count that text once, and a stored script that
repeats a sentence back to back must not be reused.

    python -m unittest dubbing/engine/tests/test_split_repeats.py -v
"""

import os
import pathlib
import sys
import tempfile
import unittest

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="dub_split_")
os.environ.setdefault("DUB_STATUS_DIR", os.path.join(_TMP, "status"))
os.environ.setdefault("AI_LEARNING_DIR", os.path.join(_TMP, "ai_learning"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import dub_engine as de                               # noqa: E402

A = "जेव्हा तुम्ही सर्प दोष म्हणता, तेव्हा सहसा त्याचा परिणाम त्वचेवर होतो।"
B = "तो सापाचा शाप असतो, हे सगळ्यांना माहीत आहे।"
C = "त्याचा एक पैलू थेट जोडलेला आहे त्या गोष्टीशी ज्याला आपण साप म्हणतो।"


class Split(unittest.TestCase):
    def test_split_halves_become_one_clip(self):
        chunks = [(0.0, 3.0, A), (3.0, 3.0, A), (6.5, 2.0, C)]
        out, n = de._merge_split_chunks(chunks)
        self.assertEqual(n, 1)
        self.assertEqual(out[0], (0.0, 6.0, A))
        self.assertEqual(out[1][2], C)

    def test_half_contained_in_neighbour(self):
        chunks = [(0.0, 3.0, A), (3.0, 3.0, A + " " + B), (6.0, 2.0, B)]
        out, n = de._merge_split_chunks(chunks)
        self.assertEqual(n, 2)
        self.assertEqual(out, [(0.0, 8.0, A + " " + B)])

    def test_different_lines_untouched(self):
        chunks = [(0.0, 3.0, A), (3.0, 3.0, C)]
        self.assertEqual(de._merge_split_chunks(chunks), (chunks, 0))

    def test_back_to_back_repeat_detected(self):
        self.assertTrue(de._repeated_sentences(f"{A} {A} {C}"))
        self.assertTrue(de._repeated_sentences(f"{C}\n\n{C}"))
        # the same sentence again later (a speaker repeating himself) is fine
        self.assertFalse(de._repeated_sentences(f"{A} {C} {A}"))
        self.assertFalse(de._repeated_sentences(f"{A} {C}"))


class FinalWindows(unittest.TestCase):
    """v0.28.5: a script reused from a learned final dub is placed where the
    user left its clips (not paired to the English by length)."""

    def setUp(self):
        import types
        self.pl = types.SimpleNamespace(
            AI_LEARNING_DIR=tempfile.mkdtemp(dir=_TMP),
            _lang_key=lambda l: l.lower(),
            _split_translation_paragraphs=lambda t: [
                p.strip() for p in t.split("\n\n") if p.strip()],
            _extract_srt_entries=lambda s: [(0.0, 2.0, "When you say."),
                                            (60.5, 62.0, "Curse."),
                                            (62.2, 64.0, "Not outside.")])

    def test_saved_positions_become_exact_rows(self):
        chunks = [(60.0, 2.0, A), (62.1, 2.0, C)]
        doc = f"{A}\n\n{C}"
        req = "P: 1|60.000|2.000|60.000\nP: 2|62.100|2.000|60.000\n"
        de._save_final_windows(self.pl, "Marathi", doc, chunks, req)
        rows = de._final_window_rows(self.pl, "Marathi", "srt", doc)
        self.assertEqual(len(rows), 2)
        # the region offset (60 s) is taken back out
        self.assertEqual((rows[0]["start"], rows[0]["end"]), (0.0, 2.0))
        self.assertEqual(rows[0]["cues"], [1])
        self.assertEqual(rows[1]["tr"], C)

    def test_changed_script_falls_back(self):
        de._save_final_windows(self.pl, "Marathi", f"{A}\n\n{C}",
                               [(0.0, 2.0, A), (2.0, 2.0, C)], "")
        self.assertIsNone(de._final_window_rows(self.pl, "Marathi", "srt",
                                                f"{A}\n\n{B}"))


if __name__ == "__main__":
    unittest.main()
