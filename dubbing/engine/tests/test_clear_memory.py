#!/usr/bin/env python3
"""v0.29.3 Clear AI memory — offline, on TEMPORARY copies only.

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


class ClearMemory(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(dir=_TMP)
        self.learn = os.path.join(self.root, "ai_learning")
        os.makedirs(os.path.join(self.learn, "sync"), exist_ok=True)
        os.makedirs(os.path.join(self.learn, "final_windows", "marathi"))
        self._db, self._conn = tm.DB_PATH, getattr(tm._local, "conn", None)
        tm.DB_PATH = os.path.join(self.root, "tm.db")      # never the user's
        tm._local.conn = None
        tm.store_full("marathi", "English one.", "मराठी एक।", "a")
        tm.store_full("hindi", "English one.", "हिंदी एक।", "b")
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
        for p in (self.pl.profile_path("Marathi"),
                  os.path.join(self.learn, "sync", "marathi.json"),
                  self.pl.profile_path("Kannada")):
            with open(p, "w", encoding="utf-8") as f:
                f.write("{}")

    def tearDown(self):
        conn = getattr(tm._local, "conn", None)
        if conn is not None:
            conn.close()
        tm.DB_PATH = self._db
        tm._local.conn = self._conn

    def _count(self, sql):
        return tm._get_conn().execute(sql).fetchone()[0]

    def test_clears_one_language_with_backup(self):
        man = {}
        with mock.patch.object(de, "_import_pipeline", return_value=self.pl):
            de._run_clear_memory(types.SimpleNamespace(language="Marathi"),
                                 man)
        self.assertIn("Cleared all Marathi AI memory: 1 line pair(s), 1 "
                      "saved script(s)", man["clear_summary"])
        self.assertEqual(self._count(
            "select count(*) from full_docs where language='marathi'"), 0)
        self.assertEqual(self._count("select count(*) from pairs"), 0)
        self.assertEqual(self._count(
            "select count(*) from full_docs where language='hindi'"), 1)
        self.assertFalse(os.path.isfile(self.pl.profile_path("Marathi")))
        self.assertTrue(os.path.isfile(self.pl.profile_path("Kannada")))
        self.assertFalse(os.path.exists(os.path.join(self.learn, "sync",
                                                     "marathi.json")))
        b = man["clear_backup"]
        self.assertTrue(os.path.isfile(os.path.join(b, "marathi.json")))
        self.assertTrue(os.path.isfile(os.path.join(b,
                                                    "translation_memory.db")))


if __name__ == "__main__":
    unittest.main()
