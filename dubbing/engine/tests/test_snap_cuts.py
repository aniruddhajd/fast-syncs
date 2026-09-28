#!/usr/bin/env python3
"""v0.23.5 chunk cuts land in real pauses — offline, synthetic audio.

A live 52-minute Marathi dub had chunks that stopped mid-word, the word's
tail appearing as a stray blob at the start of the next chunk: eleven_v3's
character times ran early and the old +-250 ms "quietest frame" search cut
inside a word. Here the reported boundary sits 400 ms early, inside speech;
the cut must still fall inside the real pause.

    python -m unittest dubbing/engine/tests/test_snap_cuts.py -v
"""

import math
import os
import pathlib
import struct
import sys
import tempfile
import unittest
import wave

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("DUB_STATUS_DIR", tempfile.mkdtemp(prefix="dub_snap_"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import dub_engine as de                               # noqa: E402

RATE = 16000


def _write(path, parts):
    """parts: [(ms, amplitude)] — a 220 Hz tone, amplitude 0 = silence."""
    frames = bytearray()
    t = 0
    for ms, amp in parts:
        for _ in range(int(RATE * ms / 1000)):
            v = int(amp * math.sin(2 * math.pi * 220 * t / RATE))
            frames += struct.pack("<h", v)
            t += 1
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))


class PauseCut(unittest.TestCase):
    def test_pure_picks_the_pause(self):
        rms = [900] * 30 + [5] * 12 + [900] * 30          # pause at 300-420 ms
        cut = de._pause_cut(rms, 0, 10, target=100)
        self.assertTrue(300 <= cut <= 420, cut)

    def test_pure_no_pause_returns_none(self):
        self.assertIsNone(de._pause_cut([900] * 40, 0, 10, target=200))
        # a 30 ms dip is a gap between syllables, not a pause
        self.assertIsNone(de._pause_cut([900] * 20 + [5] * 3 + [900] * 20,
                                        0, 10, target=200))


class SnapCuts(unittest.TestCase):
    def setUp(self):
        self.pl = de._import_pipeline()
        if not self.pl.PYDUB_AVAILABLE:
            self.skipTest("pydub not available")
        self.path = os.path.join(tempfile.mkdtemp(), "tts.wav")
        # speech 0-1500, pause 1500-1700, speech 1700-3200
        _write(self.path, [(1500, 12000), (200, 0), (1500, 12000)])

    def test_early_timestamp_still_cuts_in_the_pause(self):
        spans = [(0, 1100), (1150, 3200)]          # boundary 400 ms too early
        out, n = de._snap_cuts(self.pl, self.path, spans)
        cut = out[0][1]
        self.assertEqual(out[0][1], out[1][0])     # neighbours share the cut
        self.assertTrue(1500 <= cut <= 1700, cut)
        self.assertEqual(n, 1)

    def test_far_apart_pieces_untouched(self):
        spans = [(0, 1000), (2600, 3200)]          # > apart_ms: real silence
        out, n = de._snap_cuts(self.pl, self.path, spans)
        self.assertEqual(out, spans)
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
