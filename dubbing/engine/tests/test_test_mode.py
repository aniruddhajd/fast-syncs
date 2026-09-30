#!/usr/bin/env python3
"""v0.24 "AI · test rules" (--script-source ai_test) — offline, no network.

The test source is AI mode plus house rules that never shares a folder
with a normal run (v0.26: it learns into the shared AI memory):
  * argparse accepts ai_test with the same guards as ai
  * it counts as an AI source (anchor sync, phrase cues) and is flagged test
  * results go to <src>/<stem>_TEST/<stem>_TEST_*; a re-run from the copy
    reuses it; a NORMAL run from that copy goes to the ordinary <stem> folder
  * v0.26: it learns (_learns); --learn-final accepts a _TEST run and
    refuses only a Prompt-agents _PTEST run

    python -m unittest dubbing/engine/tests/test_test_mode.py -v
"""

import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="dub_testmode_")
os.environ.setdefault("DUB_STATUS_DIR", os.path.join(_TMP, "status"))
os.environ.setdefault("AI_LEARNING_DIR", os.path.join(_TMP, "ai_learning"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import dub_engine as de                               # noqa: E402
from pipeline import config as cfg                    # noqa: E402


def _args(src):
    return types.SimpleNamespace(script_source=src)


def _parse(argv):
    with mock.patch.object(sys, "argv", ["dub_engine.py"] + argv):
        return de._parse_args()


class Source(unittest.TestCase):
    def test_argparse_accepts_ai_test(self):
        a = _parse(["--audio", "x.wav", "--language", "Marathi",
                    "--steps", "translate", "--script-source", "ai_test"])
        self.assertEqual(a.script_source, "ai_test")

    def test_argparse_rejects_provided_script(self):
        with self.assertRaises(SystemExit):
            _parse(["--audio", "x.wav", "--language", "Marathi",
                    "--steps", "translate", "--script-source", "ai_test",
                    "--provided-script", "p.txt"])

    def test_helpers(self):
        self.assertEqual(de._ai_source(_args("ai")), "ai")
        self.assertEqual(de._ai_source(_args("ai_test")), "ai_test")
        self.assertIsNone(de._ai_source(_args("prompt")))
        self.assertTrue(de._is_test(_args("ai_test")))
        self.assertFalse(de._is_test(_args("ai")))


class Folder(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(dir=_TMP)
        self.audio = os.path.join(self.dir, "talk.wav")
        with open(self.audio, "wb") as f:
            f.write(b"RIFF0000WAVE")

    def test_test_folder_and_base(self):
        m = {}
        out, base = de._prepare_test_dir(self.audio, m)
        self.assertEqual(out, os.path.join(self.dir, "talk_TEST"))
        self.assertEqual(base, os.path.join(out, "talk_TEST"))
        self.assertEqual(m["variant"], "test")
        self.assertTrue(os.path.isfile(os.path.join(out, "talk.wav")))
        # a re-run from the copy inside it reuses the same folder
        out2, _ = de._prepare_test_dir(os.path.join(out, "talk.wav"), {})
        self.assertEqual(out2, out)

    def test_normal_run_from_test_copy_goes_beside(self):
        out, _ = de._prepare_test_dir(self.audio, {})
        normal = cfg._prepare_output_dir(os.path.join(out, "talk.wav"))
        self.assertEqual(normal, os.path.join(self.dir, "talk"))

    def test_is_test_base(self):
        _out, base = de._prepare_test_dir(self.audio, {})
        self.assertTrue(de._is_test_base(base))
        self.assertFalse(de._is_test_base(os.path.join(self.dir, "talk",
                                                       "talk")))


def _learn_req(folder):
    d = tempfile.mkdtemp(dir=_TMP)
    os.makedirs(os.path.join(d, folder), exist_ok=True)
    base = os.path.join(d, folder, folder)
    req = os.path.join(d, "req.txt")
    with open(req, "w", encoding="utf-8") as f:
        f.write(f"BASE: {base}\nC: 0.0|2.0|नमस्कार\n")
    return types.SimpleNamespace(text_file=req, language="Marathi"), base


class Learning(unittest.TestCase):
    def test_test_rules_learns_prompt_agents_does_not(self):
        self.assertTrue(de._learns(_args("ai")))
        self.assertTrue(de._learns(_args("ai_test")))
        self.assertFalse(de._learns(_args("prompt_agents")))
        self.assertFalse(de._learns(_args("prompt")))

    def test_learn_final_accepts_test_run(self):
        args, base = _learn_req("a_TEST")
        self.assertFalse(de._is_ptest_base(base))
        sentinel = RuntimeError("reached learning")
        with mock.patch.object(de, "_import_pipeline", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as cm:
                de._run_learn_final(args, {})
        self.assertIs(cm.exception, sentinel)   # past the gate

    def test_learn_final_refuses_prompt_agents_run(self):
        args, base = _learn_req("a_PTEST")
        self.assertTrue(de._is_ptest_base(base))
        with self.assertRaises(RuntimeError) as cm:
            de._run_learn_final(args, {})
        self.assertIn("Prompt agents", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
