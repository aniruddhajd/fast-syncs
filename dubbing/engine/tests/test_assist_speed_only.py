#!/usr/bin/env python3
"""v0.23 assistant contract — offline, fake model, no network.

Speed up / Slow down send "SPEED ONLY": review_assist must return no
alternative lines (the script cannot change) but keep the speed. Every other
instruction still gets its alternatives.

    python -m unittest dubbing/engine/tests/test_assist_speed_only.py -v
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("AI_LEARNING_DIR",
                      tempfile.mkdtemp(prefix="dub_assist_tests_"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from pipeline import ai_translator as at             # noqa: E402

REPLY = json.dumps({"reply": "done", "alternatives": ["नवी ओळ एक", "नवी ओळ दोन"],
                    "speed": 1.12})


def _ask(instruction):
    with mock.patch.object(at, "_llm_generate", return_value=REPLY):
        return at.review_assist("Hello there.", "नमस्कार, कसे आहात?", 1.5, 2.0,
                                instruction, "Marathi")


class SpeedOnly(unittest.TestCase):
    def test_speed_only_drops_rewrites_keeps_speed(self):
        out = _ask("SPEED ONLY: do not change the script. Set a faster speed.")
        self.assertEqual(out["alternatives"], [])
        self.assertAlmostEqual(out["speed"], 1.12)

    def test_other_instructions_keep_alternatives(self):
        out = _ask("Shorten the length by changing or using fewer words.")
        self.assertEqual(len(out["alternatives"]), 2)


if __name__ == "__main__":
    unittest.main()
