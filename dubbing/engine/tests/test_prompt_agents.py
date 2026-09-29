#!/usr/bin/env python3
"""v0.25 "Prompt agents · test" (--script-source prompt_agents) — offline.

The Prompt-chain files run as three agents per page (Translator Step1 ->
Reviewer Step2 -> Punctuator Step3) with the free checks between them; the
Lekhak pipeline, memory and learning are bypassed; the rows keep English
windows for AI mode's anchor sync. A fake model answers every call.

    python -m unittest dubbing/engine/tests/test_prompt_agents.py -v
"""

import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="dub_pagents_")
os.environ.setdefault("DUB_STATUS_DIR", os.path.join(_TMP, "status"))
os.environ.setdefault("AI_LEARNING_DIR", os.path.join(_TMP, "ai_learning"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import dub_engine as de                               # noqa: E402
from pipeline import prompt_agents as pa              # noqa: E402

EN = [(0.0, 2.0, "In 1947 India became free."),
      (2.4, 4.4, "It was a big moment.")]
GOOD = "१९४७ मध्ये भारत स्वतंत्र झाला।\n\nतो एक मोठा क्षण होता।"
NO_NUMBER = "भारत स्वतंत्र झाला।\n\nतो एक मोठा क्षण होता।"


def _prompt(name, lang):
    return {"Step1_Translation_Prompt": "P1", "Step2_Review_Prompt": "P2",
            "Step3_Punctuation_Prompt": "P3"}[name]


class FakeLLM:
    """Answers per step (by static prefix); records every call."""

    def __init__(self, answers):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls = []

    def __call__(self, dynamic, model=None, static_prefix=None, **kw):
        self.calls.append(static_prefix)
        return self.answers[static_prefix].pop(0)


class Agents(unittest.TestCase):
    def run_with(self, answers):
        fake = FakeLLM(answers)
        with mock.patch.object(pa, "_llm_generate", fake), \
             mock.patch.object(pa, "_load_lang_prompt", _prompt):
            out = pa.prompt_agents_translate(EN, "Marathi", "m")
        return out, fake

    def test_three_agents_in_order(self):
        (script, rows, info), fake = self.run_with(
            {"P1": [GOOD], "P2": [GOOD], "P3": [GOOD]})
        self.assertEqual(fake.calls, ["P1", "P2", "P3"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["start"], 0.0)     # English window kept
        self.assertIn("१९४७", script)
        self.assertIn("PROMPT AGENTS REVIEW REPORT", info["report"])

    def test_translator_retried_when_a_number_is_dropped(self):
        (script, _rows, info), fake = self.run_with(
            {"P1": [NO_NUMBER, GOOD], "P2": [GOOD], "P3": [GOOD]})
        self.assertEqual(fake.calls, ["P1", "P1", "P2", "P3"])
        self.assertIn("१९४७", script)
        self.assertIn("retried", " ".join(info["page_outcomes"][0]["log"]))

    def test_reviewer_that_drops_a_number_is_rejected(self):
        (script, _rows, info), _ = self.run_with(
            {"P1": [GOOD], "P2": [NO_NUMBER], "P3": [GOOD]})
        self.assertIn("१९४७", script)
        self.assertIn("reviewer: rejected",
                      " ".join(info["page_outcomes"][0]["log"]))


class Engine(unittest.TestCase):
    def test_source_flags(self):
        a = types.SimpleNamespace(script_source="prompt_agents")
        self.assertEqual(de._ai_source(a), "prompt_agents")   # AI-mode sync
        self.assertTrue(de._is_test(a))                        # never learns
        self.assertFalse(de._house_rules_on(a))                # no rules
        self.assertEqual(de._test_suffix(a), "_PTEST")
        self.assertEqual(de._draft_mode(a), "ptest")

    def test_ptest_folder(self):
        d = tempfile.mkdtemp(dir=_TMP)
        audio = os.path.join(d, "talk.wav")
        with open(audio, "wb") as f:
            f.write(b"RIFF")
        m = {}
        out, base = de._prepare_test_dir(audio, m, de.PTEST_SUFFIX)
        self.assertEqual(out, os.path.join(d, "talk_PTEST"))
        self.assertEqual(base, os.path.join(out, "talk_PTEST"))
        self.assertEqual(m["variant"], "ptest")
        self.assertTrue(de._is_test_base(base))

    def test_needs_only_step1_to_3(self):
        a = types.SimpleNamespace(script_source="prompt_agents",
                                  steps="translate", provided_script=None)
        self.assertEqual(de._required_prompts(a),
                         ["Step1_Translation_Prompt", "Step2_Review_Prompt",
                          "Step3_Punctuation_Prompt"])
        a.steps = "dub"
        self.assertEqual(de._required_prompts(a), [])


if __name__ == "__main__":
    unittest.main()
