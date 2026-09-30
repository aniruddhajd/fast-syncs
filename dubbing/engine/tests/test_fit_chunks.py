#!/usr/bin/env python3
"""v0.32 "Write a line for each chunk" — offline, no model.

The Regenerate tab sends several placed chunks; the engine writes one line
per chunk sized to its length. These tests fake the LLM and check the
request parsing, the prompt (English per chunk, char budget, order rule)
and the reply handling (bad shapes skipped, pause markers stripped).

    python -m unittest dubbing/engine/tests/test_fit_chunks.py -v
"""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("AI_LEARNING_DIR",
                      tempfile.mkdtemp(prefix="dub_fit_tests_"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from pipeline import ai_translator as at             # noqa: E402
import dub_engine                                    # noqa: E402

ROWS = [{"k": 1, "len": 2.0, "en": "When you sit here,", "cur": "old one"},
        {"k": 2, "len": 3.0, "en": "you must be at ease.", "cur": ""}]


class ParseRequest(unittest.TestCase):
    def test_parse(self):
        text = ("CPS: 12.50\nPREV: पहले\nNEXT: \n"
                "@@ 1 2.000\nEN: When you sit here,\nCUR: old one\n"
                "@@ 2 3.000\nEN: you must be at ease.\nCUR: \n"
                "@@ bad\nEN: ignored\n")
        rows, cps, prev, nxt = dub_engine._parse_fit_chunks(text)
        self.assertEqual(cps, 12.5)
        self.assertEqual(prev, "पहले")
        self.assertEqual(nxt, "")
        self.assertEqual([(r["k"], r["len"], r["en"]) for r in rows],
                         [(1, 2.0, "When you sit here,"),
                          (2, 3.0, "you must be at ease.")])


class FitChunks(unittest.TestCase):
    def _run(self, reply, cps=10.0):
        seen = {}

        def fake(prompt, model, **kw):
            seen["prompt"] = prompt
            return reply
        with mock.patch.object(at, "_llm_generate", side_effect=fake):
            out = at.fit_chunks(ROWS, "Hindi", cps)
        return out, seen.get("prompt", "")

    def test_prompt_has_budget_english_and_order(self):
        _out, prompt = self._run('{"lines": []}')
        self.assertIn("~20 chars", prompt)          # 2.0 s * 10 cps
        self.assertIn("~30 chars", prompt)
        self.assertIn("When you sit here,", prompt)
        self.assertIn("English order", prompt)

    def test_reply_parsed_and_cleaned(self):
        out, _ = self._run('{"lines": [{"k": 1, "text": "जब आप | यहाँ बैठते हैं,"},'
                           ' {"k": "2", "text": "आप सहज हों।"},'
                           ' {"k": 9, "text": "stray"}, {"text": "no k"}, 5]}')
        self.assertEqual(out, {1: "जब आप यहाँ बैठते हैं,", 2: "आप सहज हों।"})

    def test_pause_guidance_and_fill(self):
        _out, prompt = self._run('{"lines": []}')
        self.assertIn("[pause N s]", prompt)
        self.assertIn("FILLS the chunk's time", prompt)

    def test_copied_pause_tag_never_voiced(self):
        out, _ = self._run('{"lines": [{"k": 1, "text": '
                           '"जब आप [pause 0.6 s] यहाँ बैठते हैं,"}]}')
        self.assertEqual(out, {1: "जब आप यहाँ बैठते हैं,"})

    def test_llm_failure_is_empty(self):
        with mock.patch.object(at, "_llm_generate",
                               side_effect=RuntimeError("down")):
            self.assertEqual(at.fit_chunks(ROWS, "Hindi"), {})

    def test_no_rows(self):
        self.assertEqual(at.fit_chunks([], "Hindi"), {})


if __name__ == "__main__":
    unittest.main()
