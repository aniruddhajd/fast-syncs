#!/usr/bin/env python3
"""v0.22 ElevenLabs Dub (--script-source eleven) — offline tests, no network.

Every HTTP request goes to a fake ElevenLabs (pipeline.el_dubbing._urlopen is
patched), so nothing is uploaded and no credits are spent. Covers:

  * language code mapping, resource parsing (speakers, time order, dub times)
  * cast -> one voice per ElevenLabs speaker (majority, warning on conflict)
  * the API key is never sent to a non-ElevenLabs host (render download)
  * an HTTP error surfaces ElevenLabs' own message
  * translate stage end to end: project file, review rows, pre-filled cast
  * dub stage end to end: only edited segments PATCHed, speaker voices set,
    re-dub + render + download, timestamps laid out at the segment times

    python -m unittest dubbing/engine/tests/test_el_dubbing.py -v
"""

import io
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
import urllib.error
import wave
from unittest import mock

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="dub_el_tests_")
os.environ["DUB_STATUS_DIR"] = os.path.join(_TMP, "status")
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from pipeline import el_dubbing as eld               # noqa: E402
import dub_engine as de                              # noqa: E402

VOICE_A = "VoiceAAAAAAAAAAAAAAA"
VOICE_B = "VoiceBBBBBBBBBBBBBBB"
VOICE_DEFAULT = "VoiceDDDDDDDDDDDDDDD"


def _wav_bytes(seconds=5.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def _resource():
    seg = lambda sid, s, e, en, tr: {                      # noqa: E731
        "id": sid, "start_time": s, "end_time": e, "text": en,
        "dubs": {"hi": {"start_time": s, "end_time": e - 0.1, "text": tr,
                        "audio_stale": False}}}
    return {
        "speaker_tracks": {
            "spk_a": {"id": "spk_a", "speaker_name": "Speaker 1",
                      "voices": {"hi": "clonedA"},
                      "segments": ["seg2", "seg1"]},
            "spk_b": {"id": "spk_b", "speaker_name": "Speaker 2",
                      "voices": {"hi": "clonedB"}, "segments": ["seg3"]},
        },
        "speaker_segments": {
            "seg2": seg("seg2", 2.0, 3.0, "How are you?", "आप कैसे हैं?"),
            "seg1": seg("seg1", 0.5, 1.5, "Hello there.", "नमस्ते।"),
            "seg3": seg("seg3", 3.5, 4.5, "Very well.", "बहुत बढ़िया।"),
        },
        "renders": {"r1": {"id": "r1", "language": "hi",
                           "status": "complete",
                           "media_ref": {"url":
                                         "https://storage.example.com/r1"}}},
    }


class _Resp:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeEL:
    """Records every request; answers like the Dubbing Studio API."""

    def __init__(self):
        self.calls = []

    def __call__(self, req, timeout=120):
        url, method = req.full_url, req.get_method()
        body = req.data
        self.calls.append((method, url, dict(req.header_items()), body))
        path = url.split("elevenlabs.io", 1)[-1]
        if url.startswith("https://storage.example.com/"):
            return _Resp(_wav_bytes())
        if method == "POST" and path == "/v1/dubbing":
            return _Resp(json.dumps({"dubbing_id": "dub123",
                                     "expected_duration_sec": 12}).encode())
        if method == "GET" and path == "/v1/dubbing/dub123":
            return _Resp(json.dumps({"status": "dubbed"}).encode())
        if method == "GET" and path == "/v1/dubbing/resource/dub123":
            return _Resp(json.dumps(_resource()).encode())
        if method == "POST" and path.endswith("/render/hi"):
            return _Resp(json.dumps({"render_id": "r1"}).encode())
        if method in ("PATCH", "POST"):
            return _Resp(b"{}")
        raise AssertionError(f"unexpected request {method} {url}")

    def paths(self, method):
        return [u.split("elevenlabs.io", 1)[-1]
                for (m, u, _h, _b) in self.calls if m == method]


class UnitTests(unittest.TestCase):
    def test_lang_code(self):
        self.assertEqual(eld.el_lang_code({"code": "bn-IN"}), "bn")
        self.assertEqual(eld.el_lang_code({"code": "ne-NP"}), "ne")
        self.assertEqual(eld.el_lang_code({}), "")

    def test_parse_resource(self):
        speakers, segs = eld.parse_resource(_resource(), "hi")
        self.assertEqual([s["id"] for s in segs], ["seg1", "seg2", "seg3"])
        self.assertEqual([s["speaker"] for s in segs],
                         ["spk_a", "spk_a", "spk_b"])
        self.assertAlmostEqual(segs[0]["end"], 1.4)       # dub's own end
        self.assertEqual(segs[0]["tr"], "नमस्ते।")
        self.assertEqual({s["el_id"] for s in speakers}, {"spk_a", "spk_b"})

    def test_parse_resource_rejects_empty(self):
        with self.assertRaises(RuntimeError):
            eld.parse_resource({"speaker_tracks": {}}, "hi")

    def test_speaker_voices_majority(self):
        segs = [{"speaker": "a"}, {"speaker": "a"}, {"speaker": "a"},
                {"speaker": "b"}]
        vmap, warns = eld.speaker_voices(
            segs, {1: VOICE_B, 2: VOICE_A, 3: VOICE_A}, VOICE_DEFAULT)
        self.assertEqual(vmap, {"a": VOICE_A, "b": VOICE_DEFAULT})
        self.assertEqual(len(warns), 1)

    def test_key_never_sent_off_elevenlabs(self):
        fake = FakeEL()
        with mock.patch.object(eld, "_urlopen", fake):
            eld.download("https://storage.example.com/r1",
                         os.path.join(_TMP, "x.wav"), "SECRET")
            eld.get_dub("dub123", "SECRET")
        hdrs = {u: {k.lower(): v for k, v in h.items()}
                for (_m, u, h, _b) in fake.calls}
        self.assertNotIn("xi-api-key", hdrs["https://storage.example.com/r1"])
        self.assertEqual(
            hdrs["https://api.elevenlabs.io/v1/dubbing/dub123"]["xi-api-key"],
            "SECRET")

    def test_http_error_message(self):
        def boom(req, timeout=120):
            raise urllib.error.HTTPError(req.full_url, 422, "bad", {},
                                         io.BytesIO(b'{"detail":"no bn"}'))
        with mock.patch.object(eld, "_urlopen", boom):
            with self.assertRaises(RuntimeError) as cm:
                eld.get_dub("dub123", "k")
        self.assertIn("422", str(cm.exception))
        self.assertIn("no bn", str(cm.exception))


class StageTests(unittest.TestCase):
    def setUp(self):
        de.STATUS_DIR = os.path.join(_TMP, "status")
        self.pl = de._import_pipeline()
        self.dir = tempfile.mkdtemp(dir=_TMP)
        self.audio = os.path.join(self.dir, "talk.wav")
        with open(self.audio, "wb") as f:
            f.write(_wav_bytes())
        self.args = types.SimpleNamespace(
            language="Hindi", voice_id=VOICE_DEFAULT, script_source="eleven",
            steps="translate", el_model="eleven_v3", provided_script=None)

    def test_translate_then_dub(self):
        fake = FakeEL()
        manifest, ctx = {}, {"audio_path": self.audio}
        with mock.patch.object(eld, "_urlopen", fake):
            de._stage_translate_eleven(self.pl, self.args, "k", manifest, ctx)
        base = ctx["base"]

        el = json.loads(open(base + de.EL_DUB_SUFFIX, encoding="utf-8").read())
        self.assertEqual(el["dubbing_id"], "dub123")
        self.assertEqual(len(el["segments"]), 3)
        self.assertEqual(len(ctx["el_rows"]), 3)
        self.assertEqual(ctx["punc_result"].count("\n\n"), 2)
        self.assertTrue(os.path.isfile(base + "_sync_en.srt"))
        cast = json.loads(open(base + "_speakers.json",
                               encoding="utf-8").read())
        self.assertEqual(len(cast["speakers"]), 2)
        self.assertEqual(cast["assignments"]["3"]["speaker"], "s2")
        create = [b for (m, u, _h, b) in fake.calls
                  if m == "POST" and u.endswith("/v1/dubbing")][0]
        self.assertIn(b'name="dubbing_studio"\r\n\r\ntrue', create)
        self.assertIn(b'name="target_lang"\r\n\r\nhi', create)

        # Reviewer edits paragraph 2 and casts speaker 2 to voice B.
        paras = ctx["punc_result"].split("\n\n")
        paras[1] = "आप कैसे हो?"
        ctx2 = {"audio_path": self.audio, "out_dir": ctx["out_dir"],
                "base": base, "script_text": "\n\n".join(paras),
                "en_audio_dur": 5.0}
        cast["assignments"] = {"1": {"speaker": "s1", "voice_id": VOICE_A},
                               "2": {"speaker": "s1", "voice_id": VOICE_A},
                               "3": {"speaker": "s2", "voice_id": VOICE_B}}
        with open(base + "_speakers.json", "w", encoding="utf-8") as f:
            json.dump(cast, f)
        fake2 = FakeEL()
        manifest2 = {}
        with mock.patch.object(eld, "_urlopen", fake2):
            de._stage_dub_eleven(self.pl, self.args, "k", manifest2, ctx2,
                                 VOICE_DEFAULT)

        patches = fake2.paths("PATCH")
        self.assertEqual(
            [p for p in patches if "/segment/" in p],
            ["/v1/dubbing/resource/dub123/segment/seg2/hi"])
        spk_bodies = {u.rsplit("/", 1)[-1]: json.loads(b)
                      for (m, u, _h, b) in fake2.calls
                      if m == "PATCH" and "/speaker/" in u}
        self.assertEqual(spk_bodies["spk_a"]["voice_id"], VOICE_A)
        self.assertEqual(spk_bodies["spk_b"]["voice_id"], VOICE_B)
        self.assertIn("/v1/dubbing/resource/dub123/dub", fake2.paths("POST"))
        self.assertTrue(os.path.isfile(manifest2["tts_wav"]))
        ts = open(manifest2["timestamps_txt"], encoding="utf-8").read()
        lines = [ln for ln in ts.splitlines()
                 if ln.strip() and not ln.startswith("[Index]")]
        self.assertEqual(len(lines), 3)
        # piece 1 is cut at the segment's own time AND placed there
        self.assertTrue(lines[0].startswith("[1] [500ms] [1400ms]"), lines[0])
        self.assertIn("[500ms] [synced]", lines[0])
        self.assertEqual(manifest2["synced_count"], "3")

    def test_dub_rejects_paragraph_count_change(self):
        fake = FakeEL()
        manifest, ctx = {}, {"audio_path": self.audio}
        with mock.patch.object(eld, "_urlopen", fake):
            de._stage_translate_eleven(self.pl, self.args, "k", manifest, ctx)
        ctx["script_text"] = "एक\n\nदो"
        with mock.patch.object(eld, "_urlopen", FakeEL()):
            with self.assertRaises(RuntimeError) as cm:
                de._stage_dub_eleven(self.pl, self.args, "k", {}, ctx,
                                     VOICE_DEFAULT)
        self.assertIn("2 paragraph", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
