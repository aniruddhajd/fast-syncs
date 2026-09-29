#!/usr/bin/env python3
"""v0.24 house rules for "AI · test rules" — offline, no model.

The distilled prompt-mode rules (pipeline/ai_rules/) load for every shipped
language, never open a prompt file, stay small, never tell the translator to
write the three-dot pause mark, and reach the prompt ONLY when
house_rules=True — so AI · learns is byte-for-byte unchanged.

    python -m unittest dubbing/engine/tests/test_house_rules.py -v
"""

import builtins
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("AI_LEARNING_DIR",
                      tempfile.mkdtemp(prefix="dub_rules_tests_"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from pipeline import ai_agents, ai_lang                # noqa: E402
from pipeline import ai_translator as at               # noqa: E402

LANGS = ["Bengali", "Hindi", "Kannada", "Malayalam", "Tamil", "Telugu",
         "Gujarati", "Marathi", "Punjabi", "Assamese", "Odia", "Nepali"]
PROFILE = {"toneDescription": "warm", "grammarRules": "plural honorific",
           "vocabularyMappings": {"grace": "कृपा"},
           "corrections": [{"en": "Hi", "draft": "अ", "final": "ब"}]}

_real_open = builtins.open


def _no_prompts_open(file, *a, **kw):
    p = os.path.abspath(str(file)) if isinstance(file, (str, os.PathLike)) \
        else ""
    if p and os.path.basename(os.path.dirname(p)) == "prompts":
        raise AssertionError("house rules opened a prompt file: " + p)
    return _real_open(file, *a, **kw)


class Rules(unittest.TestCase):
    def setUp(self):
        ai_lang._RULES_CACHE.clear()

    def test_every_language_loads_without_prompt_files(self):
        with mock.patch.object(builtins, "open", _no_prompts_open):
            for lang in LANGS:
                r = ai_lang.house_rules(lang)
                self.assertTrue(r, lang)
                self.assertIn("holistic", r.lower(), lang)   # common part
                self.assertNotIn("<!--", r, lang)            # header stripped

    def test_unknown_language_gets_none(self):
        self.assertEqual(ai_lang.house_rules("Klingon"), "")
        self.assertEqual(ai_lang.house_rules(""), "")

    def test_small_and_never_writes_dots(self):
        for lang in LANGS:
            r = ai_lang.house_rules(lang)
            self.assertLess(len(r), 4000, lang)
            self.assertNotIn("...", r, lang)
            self.assertNotIn("…", r, lang)
            self.assertNotRegex(r, r"\[\d+(\.\d+)?s\]", lang)   # SRT format

    def test_prefix_only_with_flag_and_after_corrections(self):
        off = at.build_static_prefix("Marathi", PROFILE)
        on = at.build_static_prefix("Marathi", PROFILE, house_rules=True)
        self.assertNotIn("HOUSE DUBBING RULES", off)
        self.assertIn("HOUSE DUBBING RULES", on)
        self.assertGreater(on.index("HOUSE DUBBING RULES"),
                           on.index("RECENT HUMAN CORRECTIONS"))
        # the output-format rules still come last
        self.assertGreater(on.index("OUTPUT FORMAT"),
                           on.index("HOUSE DUBBING RULES"))

    def test_agents_only_with_flag(self):
        self.assertNotIn("HOUSE DUBBING", ai_agents.proofer_prefix("Hindi"))
        self.assertIn("HOUSE DUBBING",
                      ai_agents.proofer_prefix("Hindi", house_rules=True))
        g = ai_agents._grammar_prompt("a", "b", "Hindi",
                                      ai_lang.house_rules("Hindi"))
        self.assertIn("house dubbing rules", g.lower())
        self.assertNotIn("house dubbing rules",
                         ai_agents._grammar_prompt("a", "b", "Hindi").lower())


if __name__ == "__main__":
    unittest.main()
