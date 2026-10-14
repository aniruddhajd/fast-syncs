#!/usr/bin/env python3
"""v0.29.5 "Shorter (skip pauses)" — offline tests, synthetic audio only.

  * pause_skip.keep_spans: internal pauses shrink to SP_KEEP_S, leading
    silence stays (sync onset), trailing silence trims to SP_TAIL_S,
    a clip with no speech comes back whole
  * pause_skip.skip_pauses_segment on a real pydub segment (stereo, 16-bit)
  * dub_engine --skip-pauses end to end: --in-start/--in-len section, the
    manifest fields, and "nothing to cut" leaving regen_wav empty

    python -m unittest dubbing/engine/tests/test_pause_skip.py -v

Needs numpy, pydub and ffmpeg — all already engine requirements.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from pipeline import pause_skip as ps                # noqa: E402

SR = 16000


def _voice(sec, f0=180.0, seed=0):
    """Harmonic 'voiced speech' with a little noise — what the VAD calls
    speech (low spectral flatness, energy in 300-3400 Hz)."""
    t = np.arange(int(sec * SR)) / SR
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    rng = np.random.default_rng(seed)
    return (0.25 * y / np.abs(y).max()
            + 0.002 * rng.standard_normal(t.size)).astype(np.float32)


def _quiet(sec, seed=1):
    rng = np.random.default_rng(seed)
    return (0.0005 * rng.standard_normal(int(sec * SR))).astype(np.float32)


def _line(lead=0.30, pause=0.60, tail=0.50):
    """lead | 1.0 s speech | pause | 0.8 s speech | pause | 0.7 s speech | tail"""
    return np.concatenate([_quiet(lead), _voice(1.0), _quiet(pause, 2),
                           _voice(0.8, 200, 3), _quiet(pause, 4),
                           _voice(0.7, 170, 5), _quiet(tail, 6)])


def _write_wav(path, y, channels=1):
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    if channels == 2:
        pcm = np.repeat(pcm, 2)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _wav_secs(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


class KeepSpans(unittest.TestCase):
    def test_pauses_shrink_lead_kept_tail_trimmed(self):
        y = _line()
        spans = ps.keep_spans(y, SR)
        self.assertEqual(len(spans), 3, spans)
        self.assertEqual(spans[0][0], 0.0)           # leading silence kept
        self.assertGreater(spans[0][1], 1.25)        # first phrase whole
        kept = sum(b - a for a, b in spans)
        dur = y.size / SR
        # Two 0.6 s pauses -> ~0.06 s each, 0.5 s tail -> ~0.08 s.
        self.assertGreater(dur - kept, 1.3)
        self.assertLess(dur - kept, 1.75)
        # Every phrase survives: >= the speech it holds.
        self.assertGreater(kept, 0.3 + 1.0 + 0.8 + 0.7)

    def test_short_gaps_left_alone(self):
        y = _line(pause=0.10, tail=0.05)
        spans = ps.keep_spans(y, SR)
        self.assertEqual(len(spans), 1, spans)

    def test_no_speech_returns_whole_clip(self):
        y = _quiet(1.5)
        self.assertEqual(ps.keep_spans(np.zeros_like(y), SR), [(0.0, 1.5)])


class Segment(unittest.TestCase):
    def test_stereo_segment(self):
        from pydub import AudioSegment
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "in.wav"
            _write_wav(p, _line(), channels=2)
            seg = AudioSegment.from_wav(str(p))
            out, info = ps.skip_pauses_segment(seg)
        self.assertEqual(out.channels, 2)
        self.assertEqual(out.frame_rate, SR)
        self.assertEqual(info["pauses_cut"], 2)
        self.assertAlmostEqual(info["removed_s"], info["in_s"] - info["out_s"],
                               places=2)
        self.assertGreater(info["removed_s"], 1.2)


class AddPauses(unittest.TestCase):
    """v0.29.7 Longer (add pauses): silence INTO the pauses."""

    def _seg(self, y, channels=1):
        from pydub import AudioSegment
        d = tempfile.mkdtemp()
        p = pathlib.Path(d) / "in.wav"
        _write_wav(p, y, channels=channels)
        with open(p, "rb") as f:
            return AudioSegment.from_wav(f)

    def test_total_shared_by_pauses(self):
        seg = self._seg(_line(), channels=2)
        out, info = ps.add_pauses_segment(seg, add_s=1.0)
        self.assertEqual(info["pauses"], 2)
        self.assertAlmostEqual(info["added_s"], 1.0, places=2)
        self.assertEqual(out.channels, 2)
        self.assertAlmostEqual(len(out) - len(seg), 1000, delta=2)
        # The first word has not moved: the lead-in is untouched.
        self.assertEqual(out[:300].raw_data, seg[:300].raw_data)

    def test_default_and_cap(self):
        seg = self._seg(_line())
        _out, info = ps.add_pauses_segment(seg)
        self.assertAlmostEqual(info["added_s"], 2 * ps.SP_ADD_PER_PAUSE_S,
                               places=2)
        _out, info = ps.add_pauses_segment(seg, add_s=10.0)
        self.assertAlmostEqual(info["added_s"],
                               2 * ps.SP_ADD_MAX_PER_PAUSE_S, places=2)

    def test_between_words(self):
        # v0.29.8: eight 0.3 s "words" with 50 ms dips — no pause anywhere.
        parts = [_quiet(0.3)]
        for k in range(8):
            parts.append(_voice(0.3, 170 + 10 * k, 10 + k))
            if k < 7:
                parts.append(_quiet(0.05, 30 + k))
        parts.append(_quiet(0.3))
        y = np.concatenate(parts)
        pts = ps.gap_points(y, SR)
        words = [t for t, kind in pts if kind == "word"]
        self.assertGreaterEqual(len(words), 4, pts)
        self.assertFalse([t for t, kind in pts if kind == "pause"], pts)
        # Every word gap sits in (or right at) one of the dips.
        dips = [0.3 + 0.3 * (k + 1) + 0.05 * k for k in range(7)]
        for t in words:
            self.assertLess(min(abs(t - (d + 0.025)) for d in dips), 0.05, t)
        seg = self._seg(y)
        out, info = ps.add_pauses_segment(seg, add_s=1.0)
        self.assertEqual(info["word_gaps"], len(words))
        self.assertAlmostEqual(info["added_s"], 1.0, delta=0.02)
        self.assertEqual(out[:300].raw_data, seg[:300].raw_data)
        _o, info = ps.add_pauses_segment(seg, add_s=1.0, words=False)
        self.assertEqual(info["pauses"], 0)

    def test_no_pause_inside(self):
        seg = self._seg(np.concatenate([_quiet(0.3), _voice(1.5),
                                        _quiet(0.4)]))
        out, info = ps.add_pauses_segment(seg, add_s=1.0)
        self.assertEqual(info["pauses"], 0)
        self.assertEqual(len(out), len(seg))


class EngineMode(unittest.TestCase):
    def _run(self, d, *extra):
        status = pathlib.Path(d) / "status"
        env = dict(os.environ, DUB_STATUS_DIR=str(status))
        r = subprocess.run([sys.executable, str(ENGINE_DIR / "dub_engine.py"),
                            "--skip-pauses", *extra], env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           timeout=120)
        m = json.loads((status / "engine_done.json").read_text("utf-8"))
        return r, m

    def test_section_cut_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            src = pathlib.Path(d) / "tts.wav"
            out = pathlib.Path(d) / "regen" / "chunk_v1.wav"
            # 2 s of other audio before the chunk, 1 s after it.
            line = _line()
            _write_wav(src, np.concatenate([_voice(2.0, 220, 7), line,
                                            _voice(1.0, 230, 8)]))
            r, m = self._run(d, "--in-wav", str(src), "--out-wav", str(out),
                             "--in-start", "2.0",
                             "--in-len", str(line.size / SR))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(m["status"], "ok", m)
            self.assertEqual(m["regen_wav"], str(out))
            self.assertEqual(m["pauses_cut"], "2")
            got = _wav_secs(out)
            self.assertAlmostEqual(line.size / SR - got,
                                   float(m["removed_s"]), places=1)
            self.assertLess(got, line.size / SR - 1.2)

    def test_add_pauses_mode(self):
        with tempfile.TemporaryDirectory() as d:
            src = pathlib.Path(d) / "tts.wav"
            out = pathlib.Path(d) / "regen" / "chunk_pauses_v1.wav"
            line = _line()
            _write_wav(src, np.concatenate([_voice(1.0, 220, 7), line]))
            # _run always passes --skip-pauses; this mode needs its own call.
            status = pathlib.Path(d) / "status"
            env = dict(os.environ, DUB_STATUS_DIR=str(status))
            r = subprocess.run([sys.executable,
                                str(ENGINE_DIR / "dub_engine.py"),
                                "--add-pauses", "--add-s", "0.8",
                                "--in-wav", str(src), "--out-wav", str(out),
                                "--in-start", "1.0",
                                "--in-len", str(line.size / SR)],
                               env=env, capture_output=True, text=True,
                               encoding="utf-8", timeout=120)
            m = json.loads((status / "engine_done.json").read_text("utf-8"))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(m["status"], "ok", m)
            self.assertEqual(m["regen_wav"], str(out))
            self.assertEqual(m["pauses_cut"], "2")
            self.assertAlmostEqual(float(m["added_s"]), 0.8, places=2)
            self.assertAlmostEqual(_wav_secs(out) - line.size / SR, 0.8,
                                   places=2)

    def test_nothing_to_cut(self):
        with tempfile.TemporaryDirectory() as d:
            src = pathlib.Path(d) / "tts.wav"
            out = pathlib.Path(d) / "out.wav"
            _write_wav(src, np.concatenate([_quiet(0.2), _voice(1.5),
                                            _quiet(0.05)]))
            r, m = self._run(d, "--in-wav", str(src), "--out-wav", str(out))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(m["status"], "ok", m)
            self.assertEqual(m["regen_wav"], "")
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
