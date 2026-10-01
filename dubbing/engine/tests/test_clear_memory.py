#!/usr/bin/env python3
"""v0.28.6 Clear memory — offline, on TEMPORARY copies only.

  * audio scope forgets one audio's saved script + clip positions, keeps
    the line pairs and other audios
  * language scope wipes the language (pairs, scripts, profiles) after a
    backup, and leaves other languages alone

The memory DB and learning folder are pointed at a temp folder before any
call, so the user's real memory is never opened.

    python -m unittest dubbing/engine/tests/test_clear_memory.py -v
"""

import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="dub_clearmem_")
os.environ.setdefault("DUB_STATUS_DIR", os.path.join(_TMP, "status"))
os.environ.setdefault("AI_LEARNING_DIR", os.path.join(_TMP, "ai_learning"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import dub_engine as de                               # noqa: E402
from pipeline import tm                               # noqa: E402

SRT = ("1\n00:00:00,000 --> 00:00:02,000\nWhen you say Sarpa Dosha.\n\n"
       "2\n00:00:02,500 --> 00:00:04,000\nIt affects the skin.\n")
EN_SRC = "When you say Sarpa Dosha. It affects the skin."


class ClearMemory(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(dir=_TMP)
        self.learn = os.path.join(self.root, "ai_learning")
        os.makedirs(os.path.join(self.learn, "sync"), exist_ok=True)
        # an isolated memory DB — never the user's
        self._db, self._conn = tm.DB_PATH, getattr(tm._local, "conn", None)
        tm.DB_PATH = os.path.join(self.root, "tm.db")
        tm._local.conn = None
        tm.store_full("marathi", EN_SRC, "पहिली ओळ।\n\nदुसरी ओळ।", "a")
        tm.store_full("marathi", "Another talk entirely.", "दुसरं।", "b")
        tm.store_full("hindi", EN_SRC, "पहली पंक्ति।", "c")
        conn = tm._get_conn()
        conn.execute("INSERT INTO pairs (language, en_norm, en_text, "
                     "translation, source, proofed_at) VALUES "
                     "('marathi','x','x','y','a',0)")
        conn.commit()
        real = de._import_pipeline()
        self.pl = types.SimpleNamespace(**vars(real))
        self.pl.translation_memory = tm
        self.pl.AI_LEARNING_DIR = self.learn
        self.pl.profile_path = lambda l: os.path.join(self.learn,
                                                      l.lower() + ".json")
        self.pl.sync_profile_path = lambda l: os.path.join(
            self.learn, "sync", l.lower() + ".json")
        for p in (self.pl.profile_path("Marathi"),
                  self.pl.sync_profile_path("Marathi")):
            with open(p, "w", encoding="utf-8") as f:
                f.write("{}")
        run = os.path.join(self.root, "talk_Marathi")
        os.makedirs(run)
        self.base = os.path.join(run, "talk_Marathi")
        with open(self.base + ".srt", "w", encoding="utf-8") as f:
            f.write(SRT)
        de._save_final_windows(self.pl, "Marathi", "पहिली ओळ।\n\nदुसरी ओळ।",
                               [(0.0, 2.0, "पहिली ओळ।"),
                                (2.5, 1.5, "दुसरी ओळ।")], "")

    def tearDown(self):
        conn = getattr(tm._local, "conn", None)
        if conn is not None:
            conn.close()
        tm.DB_PATH = self._db
        tm._local.conn = self._conn

    def _run(self, scope):
        req = os.path.join(self.root, "_clear.txt")
        with open(req, "w", encoding="utf-8") as f:
            f.write(f"SCOPE: {scope}\nBASE: {self.base}\n")
        man = {}
        with mock.patch.object(de, "_import_pipeline", return_value=self.pl):
            de._run_clear_memory(types.SimpleNamespace(
                text_file=req, language="Marathi"), man)
        with open(man["clear_txt"], encoding="utf-8") as f:
            return f.read()

    def _count(self, sql):
        return tm._get_conn().execute(sql).fetchone()[0]

    def test_audio_scope_forgets_only_this_audio(self):
        out = self._run("audio")
        self.assertIn("1 saved script(s) and 1 set(s)", out)
        self.assertEqual(self._count(
            "select count(*) from full_docs where language='marathi'"), 1)
        self.assertEqual(self._count(
            "select count(*) from full_docs where language='hindi'"), 1)
        self.assertEqual(self._count("select count(*) from pairs"), 1)
        self.assertTrue(os.path.isfile(self.pl.profile_path("Marathi")))

    def test_language_scope_wipes_language_with_backup(self):
        out = self._run("language")
        self.assertIn("all Marathi memory", out)
        self.assertEqual(self._count(
            "select count(*) from full_docs where language='marathi'"), 0)
        self.assertEqual(self._count("select count(*) from pairs"), 0)
        self.assertEqual(self._count(
            "select count(*) from full_docs where language='hindi'"), 1)
        self.assertFalse(os.path.isfile(self.pl.profile_path("Marathi")))
        backup = [ln[8:].strip() for ln in out.splitlines()
                  if ln.startswith("BACKUP:")][0]
        self.assertTrue(os.path.isfile(os.path.join(backup,
                                                    "translation_memory.db")))
        self.assertTrue(os.path.isfile(os.path.join(backup, "marathi.json")))


class SeparateScopes(ClearMemory):
    """v0.29: script memory and timing memory are cleared separately."""

    def test_audio_scope_forgets_only_this_audio(self):
        pass                                    # covered by ClearMemory

    def test_language_scope_wipes_language_with_backup(self):
        pass                                    # covered by ClearMemory

    def test_sync_scope_keeps_script_memory(self):
        out = self._run("sync")
        self.assertIn("the timing Marathi memory", out)
        self.assertFalse(os.path.isfile(self.pl.sync_profile_path("Marathi")))
        self.assertTrue(os.path.isfile(self.pl.profile_path("Marathi")))
        self.assertEqual(self._count("select count(*) from pairs"), 1)

    def test_script_scope_keeps_timing(self):
        out = self._run("script")
        self.assertIn("the script Marathi memory", out)
        self.assertTrue(os.path.isfile(self.pl.sync_profile_path("Marathi")))
        self.assertFalse(os.path.isfile(self.pl.profile_path("Marathi")))
        self.assertEqual(self._count("select count(*) from pairs"), 0)


class LearnMode(unittest.TestCase):
    def test_mode_line(self):
        self.assertEqual(de._learn_mode("BASE: x\nMODE: sync\n"), "sync")
        self.assertEqual(de._learn_mode("MODE: script"), "script")
        self.assertEqual(de._learn_mode("BASE: x\nC: 1|2|t\n"), "both")

    def test_sync_mode_never_touches_the_script_memory(self):
        d = tempfile.mkdtemp(dir=_TMP)
        os.makedirs(os.path.join(d, "a"), exist_ok=True)
        base = os.path.join(d, "a", "a")
        req = os.path.join(d, "req.txt")
        with open(req, "w", encoding="utf-8") as f:
            f.write(f"BASE: {base}\nMODE: sync\nC: 0.0|2.0|नमस्कार मंडळी\n")
        man = {}
        with mock.patch.object(de, "_import_pipeline",
                               side_effect=AssertionError("no script learn")):
            de._run_learn_final(types.SimpleNamespace(
                text_file=req, language="Marathi"), man)
        with open(man["learn_txt"], encoding="utf-8") as f:
            out = f.read()
        self.assertIn("MODE: sync", out)
        self.assertIn("SYNC:", out)


if __name__ == "__main__":
    unittest.main()
