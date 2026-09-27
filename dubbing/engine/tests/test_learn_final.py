#!/usr/bin/env python3
"""v0.21 "learn from the final dub" — offline unit tests, no network.

Covers the --learn-final util mode and the review-time learning switch:

  * request parsing (BASE:, C: lines, '|' inside text, BOM, bad lines)
  * English-under-chunk lookup (cue midpoint rule)
  * regen-edit corrections (chains collapsed, undone edits dropped)
  * draft paragraph -> final corrections (and the ambiguous case skipped)
  * truncation guard: a final that lost a number is held back
  * idempotency: the same final timeline is learned once
  * --learn-final end to end through dub_engine.main() with a fake LLM
  * --steps dub no longer learns unless ai_learn_at = "review"

Every LLM call goes to a fake (pipeline.ai_translator._llm_generate is
patched), the learning profile and translation memory live in a temp dir,
and any attempt to open a prompt file fails the test.

    python -m unittest dubbing/engine/tests/test_learn_final.py -v
"""

import builtins
import json
import os
import pathlib
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest import mock

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="dub_learn_final_tests_")
# Never touch the user's learning profile or translation memory.
os.environ["AI_LEARNING_DIR"] = os.path.join(_TMP, "ai_learning")
os.environ["TRANSLATION_MEMORY_DB"] = os.path.join(_TMP, "tm.db")
os.environ["DUB_STATUS_DIR"] = os.path.join(_TMP, "status")
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from pipeline import ai_translator as at             # noqa: E402
from pipeline import tm as tmmod                     # noqa: E402
import dub_engine as de                              # noqa: E402

LEARN_DIR = os.path.join(_TMP, "ai_learning")
TM_DB = os.path.join(_TMP, "tm.db")


def setUpModule():
    # Another test module may have imported the pipeline first (env read at
    # import time) — pin every storage path to the temp dir regardless.
    at.AI_LEARNING_DIR = LEARN_DIR
    tmmod.DB_PATH = TM_DB
    tmmod._local.conn = None
    de.STATUS_DIR = os.path.join(_TMP, "status")


class FakeLLM:
    """Stands in for _llm_generate; records every call."""

    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply or json.dumps({
            "toneDescription": "warm, plain",
            "grammarRules": "use the honorific plural",
            "vocabularyMappings": {"grace": "कृपा"}})

    def __call__(self, prompt, model=None, **kw):
        self.calls.append((prompt, kw))
        return self.reply


_real_open = builtins.open


def _no_prompts_open(file, *a, **kw):
    p = os.path.abspath(str(file)) if isinstance(file, (str, os.PathLike)) \
        else ""
    if p and os.path.basename(os.path.dirname(p)) == "prompts":
        raise AssertionError("AI mode opened a prompt file: " + p)
    return _real_open(file, *a, **kw)


def _srt(cues):
    def ts(x):
        ms = int(round(x * 1000))
        return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms // 60000 % 60,
                                        ms // 1000 % 60, ms % 1000)
    return "\n".join(f"{i}\n{ts(s)} --> {ts(e)}\n{t}\n"
                     for i, (s, e, t) in enumerate(cues, 1))


def _make_run(name, en_cues, draft_rows=None, edits=None):
    """A fake run folder: <tmp>/<name>/<name>{_ai_phrases.srt, ...}."""
    d = os.path.join(_TMP, name)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, name)
    with open(base + "_ai_phrases.srt", "w", encoding="utf-8") as f:
        f.write(_srt(en_cues))
    if draft_rows is not None:
        with open(base + "_ai_draft.json", "w", encoding="utf-8") as f:
            json.dump({"rows": draft_rows}, f, ensure_ascii=False)
    if edits:
        with open(base + "_regen_edits.jsonl", "w", encoding="utf-8") as f:
            for e in edits:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return base


def _write_request(base, chunks, name="_learn_final.txt"):
    path = os.path.join(os.path.dirname(base), name)
    lines = ["BASE: " + base] + [f"C: {s:.3f}|{n:.3f}|{t}"
                                 for (s, n, t) in chunks]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


EN = [(0.0, 2.0, "Grace is available to all."),
      (2.5, 5.0, "The moon is 108 times away."),
      (5.5, 8.0, "Just sit quietly.")]
DRAFT = [
    {"en": EN[0][2], "tr": "कृपा सबके लिए उपलब्ध है।", "start": 0.0,
     "end": 2.0, "cues": [1]},
    {"en": EN[1][2], "tr": "चंद्रमा 108 गुना दूर है।", "start": 2.5,
     "end": 5.0, "cues": [2]},
    {"en": EN[2][2], "tr": "बस शांति से बैठिए।", "start": 5.5, "end": 8.0,
     "cues": [3]},
]


class RequestParsingTest(unittest.TestCase):

    def test_base_chunks_pipes_bom_and_bad_lines(self):
        text = ("\ufeffBASE: C:\\runs\\talk\\talk\n"
                "C: 5.5|2.0|third  line\n"
                "C: 0.000|1.500|first | with a pipe\n"
                "C: nope|1|bad number\n"
                "C: 3.0|1.0\n"
                "C: 2.0|1.0|   \n"
                "junk line\n")
        base, chunks = de._parse_learn_request(text)
        self.assertEqual(base, "C:\\runs\\talk\\talk")
        self.assertEqual(chunks, [(0.0, 1.5, "first | with a pipe"),
                                  (5.5, 2.0, "third line")])


class EnglishUnderTest(unittest.TestCase):

    def test_midpoint_rule_never_splits_a_cue(self):
        self.assertEqual(de._english_under(EN, 0.0, 2.2),
                         "Grace is available to all.")
        # cue 2 (mid 3.75) belongs to the window that holds its midpoint only
        self.assertEqual(de._english_under(EN, 2.0, 3.6), "")
        self.assertEqual(de._english_under(EN, 3.6, 8.0),
                         "The moon is 108 times away. Just sit quietly.")
        self.assertEqual(de._english_under([], 0, 10), "")

    def test_english_source_order(self):
        base = _make_run("ensrc", EN)
        pl = de._import_pipeline()
        entries, name = de._load_final_english(pl, base)
        self.assertEqual(name, "ensrc_ai_phrases.srt")
        self.assertEqual([t for (_s, _e, t) in entries], [c[2] for c in EN])


class CorrectionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pl = de._import_pipeline()

    def test_regen_edits_chain_and_undone_edit(self):
        edits = [
            {"t": 5.5, "len": 2.0, "old": "बस शांति से बैठिए।",
             "new": "बस चुपचाप बैठिए।", "merged": 0, "at": "x"},
            {"t": 5.5, "len": 1.8, "old": "बस चुपचाप बैठिए।",
             "new": "बस चुपचाप बैठ जाइए।", "merged": 0, "at": "y"},
            # undone with Ctrl+Z: its new text is not on the timeline
            {"t": 0.0, "len": 2.0, "old": "कृपा सबके लिए उपलब्ध है।",
             "new": "कृपा हर किसी के लिए है।", "merged": 0, "at": "z"},
        ]
        base = _make_run("regen", EN, DRAFT, edits)
        chunks = [(0.0, 2.0, DRAFT[0]["tr"]), (2.5, 2.5, DRAFT[1]["tr"]),
                  (5.5, 1.8, "बस चुपचाप बैठ जाइए।")]
        got = de._load_regen_edits(base, chunks)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["old"], "बस शांति से बैठिए।")
        self.assertEqual(got[0]["new"], "बस चुपचाप बैठ जाइए।")
        en_entries, _ = de._load_final_english(self.pl, base)
        pairs, corr = de._final_pairs_and_corrections(self.pl, base, chunks,
                                                      en_entries, got)
        self.assertEqual(corr, [{"en": "Just sit quietly.",
                                 "draft": "बस शांति से बैठिए।",
                                 "final": "बस चुपचाप बैठ जाइए।"}])
        # untouched paragraphs pair with their draft English; the
        # regenerated chunk pairs with the English under it
        self.assertIn((EN[0][2], DRAFT[0]["tr"]), pairs)
        self.assertIn((EN[2][2], "बस चुपचाप बैठ जाइए।"), pairs)

    def test_draft_paragraph_to_final_correction(self):
        base = _make_run("draftcorr", EN, DRAFT)
        # paragraph 1 was edited at review time and dubbed in two pieces
        chunks = [(0.0, 1.0, "कृपा सभी के लिए"), (1.0, 1.0, "उपलब्ध है।"),
                  (2.5, 2.5, DRAFT[1]["tr"]), (5.5, 2.0, DRAFT[2]["tr"])]
        en_entries, _ = de._load_final_english(self.pl, base)
        pairs, corr = de._final_pairs_and_corrections(self.pl, base, chunks,
                                                      en_entries, [])
        self.assertEqual(corr, [{"en": EN[0][2],
                                 "draft": "कृपा सबके लिए उपलब्ध है।",
                                 "final": "कृपा सभी के लिए उपलब्ध है।"}])
        self.assertEqual(len(pairs), 3)

    def test_unrelated_text_is_not_a_correction(self):
        base = _make_run("unrelated", EN, DRAFT)
        chunks = [(0.0, 2.0, "पूरी तरह अलग वाक्य जिसका कोई संबंध नहीं"),
                  (2.5, 2.5, DRAFT[1]["tr"]), (5.5, 2.0, DRAFT[2]["tr"])]
        en_entries, _ = de._load_final_english(self.pl, base)
        _pairs, corr = de._final_pairs_and_corrections(self.pl, base, chunks,
                                                       en_entries, [])
        self.assertEqual(corr, [])

    def test_fit_shortening_is_not_a_correction(self):
        base = _make_run("fitshort", EN, DRAFT)
        with open(base + "_ai_fit_changes.txt", "w", encoding="utf-8") as f:
            f.write("Piece 3\nAPPROVED: बस शांति से बैठिए।\n"
                    "SPOKEN:   बस शांति से बैठो।\n")
        chunks = [(0.0, 2.0, DRAFT[0]["tr"]), (2.5, 2.5, DRAFT[1]["tr"]),
                  (5.5, 2.0, "बस शांति से बैठो।")]
        en_entries, _ = de._load_final_english(self.pl, base)
        _pairs, corr = de._final_pairs_and_corrections(self.pl, base, chunks,
                                                       en_entries, [])
        self.assertEqual(corr, [])


class LearnFromFinalTest(unittest.TestCase):

    def setUp(self):
        setUpModule()

    def test_truncation_is_held_back(self):
        fake = FakeLLM()
        corr = [{"en": EN[1][2], "draft": "चंद्रमा 108 गुना दूर है।",
                 "final": "चंद्रमा दूर है।"},
                {"en": EN[2][2], "draft": "बस शांति से बैठिए।",
                 "final": "बस चुपचाप बैठिए।"}]
        pairs = [(EN[1][2], "चंद्रमा दूर है।"), (EN[2][2], "बस चुपचाप बैठिए।")]
        with mock.patch.object(at, "_llm_generate", fake):
            res = at.learn_from_final("Hindi", pairs, corr, "x", EN,
                                      source="t")
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["held_back"], ["चंद्रमा दूर है।"])
        self.assertEqual(res["corrections"], 1)
        self.assertEqual(res["pairs"], 1)
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn("चंद्रमा दूर है", fake.calls[0][0])
        prof = at.load_profile("Hindi")
        self.assertEqual(prof["vocabularyMappings"].get("grace"), "कृपा")
        self.assertTrue(prof["corrections"][-1]["final"] == "बस चुपचाप बैठिए।")

    def test_never_raises_and_llm_error_keeps_memory(self):
        def boom(*a, **k):
            raise RuntimeError("no network in tests")
        with mock.patch.object(at, "_llm_generate", boom):
            res = at.learn_from_final("Tamil", [("Hello there friend.",
                                                 "வணக்கம் நண்பரே")], [])
        self.assertTrue(res["ok"])
        self.assertIn("profile merge skipped", res["error"])
        self.assertEqual(res["tm_stored"], 1)
        with mock.patch.object(at, "_absorb_evidence",
                               side_effect=ValueError("x")):
            res = at.learn_from_final("Tamil", [("a b c", "d e f")], [])
        self.assertFalse(res["ok"])


class EndToEndTest(unittest.TestCase):

    def setUp(self):
        setUpModule()

    def _main(self, req):
        argv = ["dub_engine.py", "--learn-final", "--language", "Hindi",
                "--text-file", req]
        fake = FakeLLM()
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(at, "_llm_generate", fake), \
             mock.patch.object(builtins, "open", _no_prompts_open):
            rc = de.main()
        with open(os.path.join(de.STATUS_DIR, "engine_done.json"),
                  encoding="utf-8") as f:
            man = json.load(f)
        return rc, man, fake

    def test_learn_final_end_to_end_and_idempotent(self):
        edits = [{"t": 5.5, "len": 1.9, "old": "बस शांति से बैठिए।",
                  "new": "बस चुपचाप बैठिए।", "merged": 0,
                  "at": "2026-09-27 10:00:00"}]
        base = _make_run("e2e", EN, DRAFT, edits)
        chunks = [(0.0, 2.0, DRAFT[0]["tr"]), (2.5, 2.5, DRAFT[1]["tr"]),
                  (5.5, 1.9, "बस चुपचाप बैठिए।")]
        req = _write_request(base, chunks)
        rc, man, fake = self._main(req)
        self.assertEqual(rc, 0, man)
        self.assertEqual(set(man), {"status", "error", "learn_txt"})
        self.assertEqual(man["status"], "ok")
        with open(man["learn_txt"], encoding="utf-8") as f:
            out = f.read()
        self.assertIn("SKIPPED: 0", out)
        self.assertIn("PAIRS: 3", out)
        self.assertIn("CORRECTIONS: 1", out)
        self.assertIn("HELD: 0", out)
        self.assertEqual(len(fake.calls), 1)
        with open(base + "_ai_learned_final.json", encoding="utf-8") as f:
            marker = json.load(f)
        for k in ("sha256", "learned_at", "pairs", "corrections",
                  "held_back"):
            self.assertIn(k, marker)
        self.assertTrue(os.path.isfile(os.path.join(LEARN_DIR, "hindi.json")))
        con = sqlite3.connect(TM_DB)
        n_pairs = con.execute("SELECT COUNT(*) FROM pairs WHERE "
                              "language='hindi'").fetchone()[0]
        n_full = con.execute("SELECT COUNT(*) FROM full_docs WHERE "
                             "language='hindi'").fetchone()[0]
        con.close()
        self.assertGreaterEqual(n_pairs, 3)
        self.assertEqual(n_full, 1)          # the timeline covers the English

        # Same final timeline again: skipped, no LLM call.
        rc, man, fake = self._main(req)
        self.assertEqual(rc, 0)
        with open(man["learn_txt"], encoding="utf-8") as f:
            self.assertIn("SKIPPED: 1", f.read())
        self.assertEqual(fake.calls, [])

        # One more fix: only the NEW evidence is learned.
        chunks2 = chunks[:2] + [(5.5, 1.9, "बस चुपचाप बैठ जाइए।")]
        with open(base + "_regen_edits.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": 5.5, "len": 1.9,
                                "old": "बस चुपचाप बैठिए।",
                                "new": "बस चुपचाप बैठ जाइए।", "merged": 0,
                                "at": "2026-09-27 10:05:00"},
                               ensure_ascii=False) + "\n")
        rc, man, fake = self._main(_write_request(base, chunks2))
        self.assertEqual(rc, 0)
        with open(man["learn_txt"], encoding="utf-8") as f:
            out = f.read()
        self.assertIn("PAIRS: 1", out)       # only the changed line
        self.assertIn("CORRECTIONS: 1", out)

    def test_bad_request_writes_error_manifest(self):
        req = os.path.join(_TMP, "_bad_learn.txt")
        with open(req, "w", encoding="utf-8") as f:
            f.write("C: 0|1|text only, no BASE\n")
        rc, man, _fake = self._main(req)
        self.assertEqual(rc, 1)
        self.assertEqual(man["status"], "error")
        self.assertIn("BASE", man["error"])

    def test_registered(self):
        self.assertIn("learn_from_final", de.REQUIRED_FUNCTIONS)
        de._check_symbols(de._import_pipeline())


class StepsDubLearningSwitchTest(unittest.TestCase):

    def _run(self, settings):
        path = os.path.join(_TMP, "engine_settings_%d.json" % len(settings))
        if settings:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(settings, f)
        elif os.path.exists(path):
            os.remove(path)
        calls = []
        notes = []
        with mock.patch.object(de, "ENGINE_SETTINGS_FILE", path), \
             mock.patch.object(de, "_ai_learn",
                               lambda *a: calls.append(a)), \
             mock.patch.object(de, "_note", notes.append):
            de._maybe_ai_learn(types.SimpleNamespace(), types.SimpleNamespace(
                language="Hindi"), "base", "script")
        return calls, notes

    def test_default_waits_for_final(self):
        calls, notes = self._run({})
        self.assertEqual(calls, [])
        self.assertTrue(any("Learn from final dub" in n for n in notes))
        calls, _ = self._run({"ai_learn_at": "final"})
        self.assertEqual(calls, [])

    def test_review_setting_learns_at_dub(self):
        calls, _ = self._run({"ai_learn_at": "review"})
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
