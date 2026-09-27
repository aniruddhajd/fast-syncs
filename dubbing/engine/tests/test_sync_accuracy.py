#!/usr/bin/env python3
"""v0.20 sync-accuracy unit tests — offline, no network, no user data.

Covers the pieces AI-mode anchor sync gained in v0.20 and the behaviour
they must not disturb:

  * anchor_align.phrase_cues / cues_from_items  (phase A)
  * spectral_vad.spectral_pauses / onset_after  (phase B)
  * pause markers: ai_translator.split_pause_markers, _rows_from_reply,
    the PAUSES check, marker-free counts, clause_units(pauses=), and
    dub_engine._anchor_rows (markers only on an unchanged paragraph)  (C)
  * regressions: clause_units without markers, salvage of invalid JSON,
    marker-free translator text byte-identical

    python -m unittest dubbing/engine/tests/test_sync_accuracy.py -v
    (or run this file directly; pytest also collects it)

Needs numpy (and pydub for nothing here) — both already engine requirements.
"""

import json
import os
import pathlib
import sys
import tempfile
import types
import unittest

import numpy as np

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="dub_sync_tests_")
# Never touch the user's learning profile or translation memory.
os.environ["AI_LEARNING_DIR"] = os.path.join(_TMP, "ai_learning")
os.environ["TRANSLATION_MEMORY_DB"] = os.path.join(_TMP, "tm.db")
os.environ.setdefault("DUB_STATUS_DIR", os.path.join(_TMP, "status"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from pipeline import anchor_align as aa              # noqa: E402
from pipeline import spectral_vad as sv              # noqa: E402
from pipeline import ai_translator as at             # noqa: E402
from pipeline import ai_checks as ac                 # noqa: E402


def W(text, start, end):
    return {"text": text, "start": start, "end": end, "type": "word"}


def _words_of(cues):
    return [t for (_s, _e, txt) in cues for t in txt.split()]


# A fluent 14 s run-on with small gaps, a sentence end and one long pause.
FLUENT = []
_t = 0.2
for k in range(40):
    w = f"w{k}" + ("." if k == 17 else ",") if k in (17, 29) else f"w{k}"
    FLUENT.append(W(w, round(_t, 3), round(_t + 0.26, 3)))
    _t += 0.26 + (0.9 if k == 24 else 0.06)


class PhraseCuesTest(unittest.TestCase):

    def test_every_word_once_in_order_and_no_cue_over_max(self):
        for max_s in (2.0, 3.5, 5.0):
            cues = aa.phrase_cues(FLUENT, None, None, max_s=max_s)
            self.assertEqual(_words_of(cues), [w["text"] for w in FLUENT])
            for s, e, _t in cues:
                self.assertLessEqual(e - s, max_s + 1e-6)
                self.assertLess(s, e)
            for (_s0, e0, _t0), (s1, _e1, _t1) in zip(cues, cues[1:]):
                self.assertGreaterEqual(s1, e0)       # never overlap

    def test_cuts_at_gap_sentence_end_and_pause(self):
        cues = aa.phrase_cues(FLUENT, None, None, max_s=30.0, min_s=0.0)
        ends = [txt.split()[-1] for (_s, _e, txt) in cues]
        self.assertIn("w17.", ends)                  # sentence end
        self.assertIn("w24", ends)                   # 0.9 s gap
        # a spectral pause between w5 and w6 (no word gap there) is a cut
        mid = (FLUENT[5]["end"] + FLUENT[6]["start"]) / 2
        cues_p = aa.phrase_cues(FLUENT, None, [(mid - 0.07, mid + 0.07)],
                                max_s=30.0, min_s=0.0)
        self.assertIn("w5", [txt.split()[-1] for (_s, _e, txt) in cues_p])
        # ...but a dip in the middle of one word is not
        w = FLUENT[8]
        inside = ((w["start"] + w["end"]) / 2 - 0.05,
                  (w["start"] + w["end"]) / 2 + 0.05)
        cues_i = aa.phrase_cues(FLUENT, None, [inside], max_s=30.0, min_s=0.0)
        self.assertEqual(len(cues_i), len(cues))

    def test_regions_bucket_like_subtitle_builder(self):
        words = [W("a", 0.5, 0.7), W("b", 0.75, 1.0), W("c", 3.0, 3.3),
                 W("", 3.4, 3.5), {"text": "(laughs)", "start": 3.6,
                                   "end": 3.9, "type": "audio_event"},
                 W("d", 9.0, 9.2)]
        regions = [(0.3, 1.1), (2.9, 3.4)]          # d falls past the last
        cues = aa.phrase_cues(words, regions, None, min_s=0.0)
        self.assertEqual(_words_of(cues), ["a", "b", "c", "d"])
        self.assertAlmostEqual(cues[0][0], 0.3)     # onset pulled to region
        # c and d share the last bucket but a 5.7 s gap still cuts them
        self.assertEqual([c[2] for c in cues], ["a b", "c", "d"])

    def test_onset_pull_never_overlaps_previous_word(self):
        # Scribe stretched "This" past the start of the next region
        words = [W("said,", 5.2, 5.6), W("This", 5.7, 6.72),
                 W("branch,", 6.78, 7.2), W("you", 7.25, 7.5)]
        cues = aa.phrase_cues(words, [(5.1, 6.2), (6.47, 7.6)], None,
                              min_s=0.0)
        for (_s0, e0, _t0), (s1, _e1, _t1) in zip(cues, cues[1:]):
            self.assertGreaterEqual(s1, e0)
        self.assertAlmostEqual(cues[1][0], 6.72)

    def test_real_transcripts_never_overlap(self):
        # The cached Scribe runs, when present (read-only; skipped on a
        # fresh clone). Every word once, no overlap, none past max_s.
        import glob
        files = sorted(glob.glob(str(ENGINE_DIR.parent / "data" / "stt_cache"
                                     / "All_Dialogue.wav.*.json")))[:3]
        if not files:
            self.skipTest("no cached transcripts")
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                words = json.load(fh).get("words") or []
            cues = aa.phrase_cues(words, None, None)
            want = [str(w.get("text", "")).strip() for w in words
                    if w.get("type", "word") == "word"
                    and str(w.get("text", "")).strip()]
            self.assertEqual(_words_of(cues), " ".join(want).split())
            for (_s0, e0, _t0), (s1, _e1, _t1) in zip(cues, cues[1:]):
                self.assertGreaterEqual(s1 + 1e-6, e0)
            self.assertTrue(all(e - s <= aa.PHRASE_MAX_S + 1e-6
                                for s, e, _t in cues))

    def test_short_phrase_merges_only_across_short_gap(self):
        words = [W("one", 0.0, 0.3), W("two", 0.35, 0.6),   # 0.6 s cue
                 W("three", 0.95, 1.6), W("four", 1.65, 2.4),
                 W("so", 4.0, 4.3)]                          # far away
        cues = aa.phrase_cues(words, None, None)
        self.assertEqual([c[2] for c in cues], ["one two three four", "so"])

    def test_empty_input(self):
        self.assertEqual(aa.phrase_cues([], None, None), [])
        self.assertEqual(aa.cues_from_items([], [(0, 1), (1, 1)]), [])


class CuesFromItemsTest(unittest.TestCase):

    def test_items_are_boundaries_and_words_kept_once(self):
        items = [(0.0, 2.0), (2.1, 3.0), (6.0, 20.0)]
        cues = aa.cues_from_items(FLUENT, items, max_s=5.0)
        self.assertEqual(_words_of(cues), [w["text"] for w in FLUENT])
        for s, e, _t in cues:
            self.assertLessEqual(e - s, 5.0 + 1e-6)
        # the first item's words never share a cue with the second item's
        first_item = {w["text"] for w in FLUENT
                      if (w["start"] + w["end"]) / 2 < 2.0}
        for _s, _e, txt in cues:
            ws = set(txt.split())
            self.assertTrue(ws <= first_item or not (ws & first_item))

    def test_no_items_falls_back_to_phrases(self):
        self.assertEqual(aa.cues_from_items(FLUENT, []),
                         aa.phrase_cues(FLUENT, None, None))


class SpectralVadTest(unittest.TestCase):
    SR = 16000

    def _signal(self):
        sr = self.SR
        rng = np.random.default_rng(1)

        def tone(sec):
            t = np.arange(int(sec * sr)) / sr
            return 0.3 * (np.sin(2 * np.pi * 180 * t)
                          + 0.5 * np.sin(2 * np.pi * 360 * t)
                          + 0.25 * np.sin(2 * np.pi * 720 * t))
        z = lambda sec: np.zeros(int(sec * sr))          # noqa: E731
        y = np.concatenate([z(0.3), tone(1.0), z(0.5), tone(1.2), z(0.25),
                            tone(0.8), z(0.4)])
        return (y + rng.normal(0, 0.0005, y.size)).astype(np.float32)

    def test_finds_the_silences_within_30ms(self):
        got = sv.spectral_pauses(self._signal(), self.SR)
        want = [(1.3, 1.8), (3.0, 3.25)]
        self.assertEqual(len(got), len(want), got)
        for (gs, ge), (ws, we) in zip(got, want):
            self.assertLess(abs(gs - ws), 0.03, got)
            self.assertLess(abs(ge - we), 0.03, got)

    def test_short_gap_is_not_a_pause(self):
        sr = self.SR
        t = np.arange(int(0.8 * sr)) / sr
        tone = 0.3 * np.sin(2 * np.pi * 200 * t)
        y = np.concatenate([tone, np.zeros(int(0.06 * sr)), tone])
        self.assertEqual(sv.spectral_pauses(y, sr), [])

    def test_onset_after(self):
        y = self._signal()
        on = sv.onset_after(y, self.SR, 1.4)
        self.assertIsNotNone(on)
        self.assertLess(abs(on - 1.8), 0.03)
        self.assertIsNone(sv.onset_after(y, self.SR, 1.4, max_s=0.2))

    def test_silence_and_tiny_input(self):
        self.assertEqual(sv.spectral_pauses(np.zeros(16000), 16000), [])
        self.assertEqual(sv.spectral_pauses(np.zeros(10), 16000), [])
        self.assertIsNone(sv.onset_after(np.zeros(10), 16000, 0.0))


class PauseMarkerTest(unittest.TestCase):

    def test_split_pause_markers(self):
        text, offs = at.split_pause_markers("अ ब | क  ड |इ|  फ")
        self.assertEqual(text, "अ ब क ड इ फ")
        self.assertEqual([text[o:o + 1] for o in offs], ["क", "इ", "फ"])
        self.assertEqual(at.split_pause_markers("| a | b |"), ("a b", [2]))
        raw = "no  markers\nhere"
        self.assertEqual(at.split_pause_markers(raw), (raw, []))

    def test_rows_from_reply_strips_and_keeps_offsets(self):
        page = [(1, 0.0, 1.0, "a"), (2, 1.2, 2.0, "b"), (3, 2.5, 3.0, "c"),
                (4, 3.2, 4.0, "d")]
        rows, repaired = at._rows_from_reply(
            [{"cues": [1, 2], "text": "एक | दोन"},
             {"cues": [3, 4], "text": "तीन | चार"}], page)
        self.assertFalse(repaired)
        self.assertEqual([r["tr"] for r in rows], ["एक दोन", "तीन चार"])
        self.assertEqual(rows[0]["pauses"], [3])
        self.assertNotIn("|", "".join(r["tr"] for r in rows))
        # overlapping claims merge; the second row's offsets shift along
        rows, repaired = at._rows_from_reply(
            [{"cues": [1, 2], "text": "एक | दोन"},
             {"cues": [2, 3, 4], "text": "तीन | चार"}], page)
        self.assertTrue(repaired)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["tr"], "एक दोन तीन चार")
        self.assertEqual([r["tr"][o:o + 1] for o in r["pauses"]], ["द", "च"])

    def test_marker_free_reply_is_unchanged(self):
        page = [(1, 0.0, 1.0, "a")]
        text = "मी म्हणालो,  ‘ही फांदी कापून टाका.’"
        rows, _rep = at._rows_from_reply([{"cues": [1], "text": text}], page)
        self.assertEqual(rows[0]["tr"], text)
        self.assertEqual(rows[0]["pauses"], [])

    def test_salvage_still_reads_invalid_json_with_markers(self):
        raw = ('{"passages": [{"n": 1, "paragraphs": [{"cues": [1, 2], '
               '"text": "मी म्हणालो, "ही | फांदी""}]}]}')
        data = at._complete_salvage(at._salvage_passages(raw), [{"page": [
            (1, 0.0, 1.0, "a"), (2, 1.0, 2.0, "b")]}])
        self.assertIsNotNone(data)
        rows, _ = at._rows_from_reply(data["passages"][0]["paragraphs"],
                                      [(1, 0.0, 1.0, "a"), (2, 1.0, 2.0, "b")])
        self.assertNotIn("|", rows[0]["tr"])
        self.assertEqual(len(rows[0]["pauses"]), 1)

    def test_pauses_check_and_marker_free_counts(self):
        en = "x" * 80
        good = [{"en": en, "tr": "क" * 70, "start": 0.0, "end": 6.0,
                 "cues": [1, 2, 3], "pauses": [20, 40]}]
        codes = [c["code"] for c in ac.run_checks(en, "क" * 70, "Marathi",
                                                    good)]
        self.assertNotIn("PAUSES", codes)
        bad = [dict(good[0], cues=list(range(1, 9)), pauses=[])]
        checks = ac.run_checks(en, "क" * 70, "Marathi", bad)
        pauses = [c for c in checks if c["code"] == "PAUSES"]
        self.assertEqual(len(pauses), 1)
        self.assertEqual(pauses[0]["severity"], "warn")
        # rows without the key (memory tier 1, older drafts) never warn
        old = [{k: v for k, v in bad[0].items() if k != "pauses"}]
        self.assertNotIn("PAUSES", [c["code"] for c in ac.run_checks(
            en, "क" * 70, "Marathi", old)])
        # markers never count toward UNDERSHOOT / TIMING lengths
        short = "क" * 30
        marked = " | ".join(["क" * 10] * 3)
        a = ac.run_checks(en, short, "Marathi", [dict(good[0], tr=short)])
        b = ac.run_checks(en, marked, "Marathi", [dict(good[0], tr=marked)])
        self.assertEqual([c["code"] for c in a if c["code"] != "PAUSES"],
                         [c["code"] for c in b if c["code"] != "PAUSES"])
        self.assertIn("UNDERSHOOT", [c["code"] for c in b])

    def test_meaning_kept_ignores_markers(self):
        orig = "अंतर १०८ पट आहे आणि हे खूप मोठे आहे"
        self.assertTrue(at.meaning_kept(orig, orig.replace(" आणि", " | आणि")))
        self.assertFalse(at.meaning_kept(orig, "अंतर | | | | | | | | | |"))

    def test_passage_block_has_char_budget(self):
        blk = at._passage_block(1, [(1, 0.0, 2.0, "hello there")], None, {},
                                [], "Marathi")
        self.assertIn("~24 chars", blk)              # 2 s x 12 chars/s
        self.assertIn('[1] @0.00s', blk)

    def test_output_rules_ask_for_markers(self):
        prefix = at.build_static_prefix("Marathi", at._empty_profile(
            "Marathi"))
        self.assertIn('" | "', prefix)
        # the fit / assistant prompts (no output rules) never ask for them
        self.assertNotIn('" | "', at.build_static_prefix(
            "Marathi", at._empty_profile("Marathi"), output_rules=False))


class ClauseUnitsTest(unittest.TestCase):
    PARA = ("मी आश्रमात असाच चाललो होतो. तेव्हा एका झाडाकडे पाहून मी "
            "म्हणालो, ‘ही फांदी कापून टाका,’ कारण तिथून लोकांची सतत ये-जा "
            "असते.")

    def test_without_markers_unchanged(self):
        # pinned output of the v0.18.1 splitter
        self.assertEqual(aa.clause_units(self.PARA), [
            "मी आश्रमात असाच चाललो होतो.",
            "तेव्हा एका झाडाकडे पाहून मी म्हणालो,",
            # no split after ",’" — the lookbehind sees the quote mark
            "‘ही फांदी कापून टाका,’ कारण तिथून लोकांची सतत ये-जा असते."])
        self.assertEqual(aa.clause_units(self.PARA, pauses=None),
                         aa.clause_units(self.PARA))

    def test_markers_cut_first_and_are_not_regrouped(self):
        marked = ("मी आश्रमात असाच | चाललो होतो. तेव्हा एका झाडाकडे पाहून मी "
                  "म्हणालो, ‘ही फांदी कापून टाका,’ | कारण तिथून लोकांची सतत "
                  "ये-जा असते.")
        text, offs = at.split_pause_markers(marked)
        units = aa.clause_units(text, pauses=offs)
        self.assertEqual(units[0], "मी आश्रमात असाच")    # marker, not the "."
        self.assertEqual(units[-1], "कारण तिथून लोकांची सतत ये-जा असते.")
        self.assertEqual(" ".join(units), text)

    def test_stale_offsets_are_ignored(self):
        # offsets that do not sit on a word gap = the text changed
        self.assertEqual(aa.clause_units(self.PARA, pauses=[4, 6]),
                         aa.clause_units(self.PARA))


class EngineGlueTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import dub_engine as de
        cls.de = de
        cls.pl = de._import_pipeline()

    def test_new_symbols_registered_and_selfcheck_symbols(self):
        for name in ("phrase_cues", "cues_from_items", "spectral_pauses",
                     "onset_after", "split_pause_markers"):
            self.assertIn(name, self.de.REQUIRED_FUNCTIONS)
        self.de._check_symbols(self.pl)

    def test_cues_to_srt_round_trip(self):
        cues = [(0.13, 2.24, "I was just walking in,"),
                (2.74, 6.72, "in the ashram.")]
        back = self.pl._extract_srt_entries(self.de._cues_to_srt(self.pl,
                                                                  cues))
        self.assertEqual(back, cues)

    def test_region_items_sidecar(self):
        wav = os.path.join(_TMP, "take.wav")
        with open(wav + ".dubregion.json", "w", encoding="utf-8") as f:
            json.dump({"version": 1, "project_pos": 12.0, "items": [
                {"start": 2.0, "len": 3.0}, {"start": 0.0, "len": 1.5}]}, f)
        self.assertEqual(self.de._region_items(wav), [(0.0, 1.5), (2.0, 3.0)])
        self.assertEqual(self.de._region_items(os.path.join(_TMP, "x.wav")),
                         [])

    def test_anchor_rows_keep_markers_only_for_unchanged_text(self):
        args = types.SimpleNamespace(script_source="ai")
        rows = [{"en": "a b", "tr": "एक दोन", "start": 0.0, "end": 2.0,
                 "cues": [1, 2], "pauses": [3]},
                {"en": "c d", "tr": "तीन चार", "start": 2.5, "end": 4.0,
                 "cues": [3, 4], "pauses": [4]}]
        ctx = {"base": os.path.join(_TMP, "nobase"), "ai_rows": rows,
               "script_text": "एक  दोन\n\nतीन बदलले"}
        out = self.de._anchor_rows(self.pl, args, ctx)
        self.assertEqual(out[0]["pauses"], [3])      # whitespace-only change
        self.assertIsNone(out[1]["pauses"])          # reviewer edited it
        # prompt mode never gets anchor rows at all
        self.assertIsNone(self.de._anchor_rows(
            self.pl, types.SimpleNamespace(script_source="prompt"), ctx))


if __name__ == "__main__":
    unittest.main(verbosity=2)
