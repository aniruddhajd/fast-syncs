#!/usr/bin/env python3
"""Translator reply shapes — offline, no model.

Regression for a live 25-minute Marathi run that died with
"'int' object is not iterable": the model wrote "cues": 12 instead of
"cues": [12]. Every shape below must parse, and a truly broken paragraph
must be skipped (retry), never crash the run.

    python -m unittest dubbing/engine/tests/test_translator_reply.py -v
"""

import os
import pathlib
import sys
import tempfile
import unittest

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("AI_LEARNING_DIR",
                      tempfile.mkdtemp(prefix="dub_reply_tests_"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from pipeline import ai_translator as at             # noqa: E402

# (id, start, end, text) — the page shape the translator is given.
PAGE = [(10, 0.0, 1.0, "Hello."), (11, 1.2, 2.0, "How are you?"),
        (12, 2.5, 3.5, "Very well.")]


class CueShapes(unittest.TestCase):
    def test_cue_list_shapes(self):
        self.assertEqual(at._cue_list(12), [12])
        self.assertEqual(at._cue_list([10, 11]), [10, 11])
        self.assertEqual(at._cue_list("11-12"), [11, 12])
        self.assertEqual(at._cue_list("10, 11"), ["10", "11"])
        self.assertEqual(at._cue_list(None), [])
        self.assertEqual(at._cue_list({"x": 1}), [])

    def test_bare_int_cues_parse(self):
        paras = [{"cues": [10, 11], "text": "नमस्कार, कसे आहात?"},
                 {"cues": 12, "text": "खूप छान."}]          # the live failure
        rows, _repaired = at._rows_from_reply(paras, PAGE)
        self.assertEqual([r["cues"] for r in rows], [[10, 11], [12]])

    def test_string_range_cues_parse(self):
        paras = [{"cues": "10-11", "text": "नमस्कार."},
                 {"cues": "12", "text": "छान."}]
        rows, _ = at._rows_from_reply(paras, PAGE)
        self.assertEqual([r["cues"] for r in rows], [[10, 11], [12]])

    def test_salvage_accepts_bare_int(self):
        batch = [{"page": PAGE}]
        data = {"passages": [{"n": 1, "paragraphs": [
            {"cues": [10, 11], "text": "अ"}, {"cues": 12, "text": "ब"}]}]}
        self.assertIsNotNone(at._complete_salvage(data, batch))


if __name__ == "__main__":
    unittest.main()
