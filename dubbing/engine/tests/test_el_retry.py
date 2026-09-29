#!/usr/bin/env python3
"""v0.24.2 ElevenLabs voice requests retry transient failures — offline.

A live 43-chunk Marathi dub died on chunk 29 with "TimeoutError: The read
operation timed out" because no ElevenLabs voice request retried. Now a
timeout / 429 / 5xx gets up to 3 tries; a bad key still fails at once; the
final failure is a plain message, not a traceback.

    python -m unittest dubbing/engine/tests/test_el_retry.py -v
"""

import io
import pathlib
import sys
import unittest
import urllib.error
from unittest import mock

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from pipeline import tts                               # noqa: E402


class _Resp:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _seq(*outcomes):
    """A fake _urlopen that raises / returns each outcome in turn."""
    calls = []

    def fake(req, timeout=0):
        o = outcomes[len(calls)]
        calls.append(req)
        if isinstance(o, BaseException):
            raise o
        return _Resp(o)
    fake.calls = calls
    return fake


def _http(code):
    return urllib.error.HTTPError("https://api.elevenlabs.io/x", code, "e",
                                  {}, io.BytesIO(b"{}"))


class Retry(unittest.TestCase):
    def setUp(self):
        self.sleep = mock.patch("time.sleep").start()
        self.addCleanup(mock.patch.stopall)

    def test_two_timeouts_then_audio(self):
        fake = _seq(TimeoutError("read timed out"), TimeoutError("again"),
                    b"MP3")
        with mock.patch.object(tts, "_urlopen", fake):
            out = tts._elevenlabs_tts_post("नमस्कार", "k", "voice12345678",
                                           "eleven_v3")
        self.assertEqual(out, b"MP3")
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_429_then_success(self):
        fake = _seq(_http(429), b"MP3")
        with mock.patch.object(tts, "_urlopen", fake):
            self.assertEqual(tts._el_read(object(), 10), b"MP3")

    def test_three_timeouts_plain_message(self):
        fake = _seq(TimeoutError("t"), TimeoutError("t"), TimeoutError("t"))
        with mock.patch.object(tts, "_urlopen", fake):
            with self.assertRaises(ValueError) as cm:
                tts._elevenlabs_tts_post("x", "k", "voice12345678",
                                         "eleven_v3")
        self.assertIn("did not answer after 3 attempts", str(cm.exception))

    def test_bad_key_is_not_retried(self):
        fake = _seq(_http(401))
        with mock.patch.object(tts, "_urlopen", fake):
            with self.assertRaises(ValueError) as cm:
                tts._elevenlabs_tts_post("x", "k", "voice12345678",
                                         "eleven_v3")
        self.assertIn("401", str(cm.exception))
        self.assertEqual(len(fake.calls), 1)
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
