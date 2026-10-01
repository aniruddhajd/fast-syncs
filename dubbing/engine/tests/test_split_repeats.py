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


if __name__ == "__main__":
    unittest.main()
