"""
Spectral (STFT) pause detector (v0.20) — numpy only
===================================================
The English speech regions come from `_detect_regions_from_audio`: a frame
RMS against an ABSOLUTE -42 dBFS threshold. That misses real pauses in two
common cases — a noisy/roomy recording whose floor sits above -42 dB (every
breath gap reads as speech, so a 76 s talk came back as 17 phrases, one of
them 18 s long), and a quiet one where soft syllables fall below it.

This module finds pauses from the SPECTRUM instead, relative to each file's
own noise floor (a short-time Fourier transform — the "can we use Fourier"
question — in the spirit of classic energy + spectral-flatness VADs):

  frames     25 ms Hann windows, 10 ms hop, numpy.fft.rfft
  band       energy in the telephone speech band 300-3400 Hz (dB) — hum,
             rumble and hiss outside it no longer count as speech
  flatness   geometric / arithmetic mean of the band's power spectrum:
             ~1 for noise, low for voiced speech (harmonics)
  floor      10th percentile of the band energy, per file (never below
             peak - 55 dB, so digital-silence padding cannot drag it down)
  level      90th percentile of the frames above the floor gate: the
             file's typical speech loudness
  speech     band energy > max(floor + 9 dB, level - 26 dB), and flatness
             < 0.45 unless the frame is loud (> floor + 20 dB — fricatives
             are flat but loud); hysteresis: once open, speech continues
             down to max(floor + 6 dB, level - 30 dB). The level term is
             what finds pauses in a very quiet studio file, where room tail
             and breath sit 20 dB over the floor but 35 dB under the voice.
  pause      a non-speech run of at least 120 ms between two speech frames

Used by anchor_align.phrase_cues (extra phrase cut points) and, later, by
onset snapping (`onset_after`, deferred phase D). It only ever ADDS cut
points; with no audio the phrase builder works from word gaps alone.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

SV_WIN_S = 0.025             # analysis window
SV_HOP_S = 0.010             # frame hop
SV_BAND_HZ = (300.0, 3400.0) # speech band
SV_OPEN_DB = 9.0             # above the floor: speech starts
SV_CLOSE_DB = 6.0            # above the floor: speech continues (hysteresis)
SV_LOUD_DB = 20.0            # loud enough to be speech whatever the flatness
SV_FLAT_MAX = 0.45           # flatness above this is noise-like
SV_FLOOR_PCT = 10.0          # noise floor percentile
SV_LEVEL_PCT = 90.0          # speech level percentile
SV_LEVEL_OPEN_DB = 26.0      # below the speech level: never opens speech
SV_LEVEL_CLOSE_DB = 30.0     # below the speech level: speech ends
SV_FLOOR_RANGE_DB = 55.0     # floor never below peak - this
SV_MIN_PAUSE_S = 0.12        # shortest pause reported
SV_BLOCK = 4096              # frames per FFT block (bounded memory)


def _sv_frames(y, sr: int):
    """(times_s, band_db, flatness) per frame. times are frame CENTRES."""
    y = np.asarray(y, dtype=np.float32).ravel()
    sr = int(sr)
    win = max(16, int(round(SV_WIN_S * sr)))
    hop = max(1, int(round(SV_HOP_S * sr)))
    if y.size < win:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    n_fr = 1 + (y.size - win) // hop
    nfft = 1 << (win - 1).bit_length()
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (freqs >= SV_BAND_HZ[0]) & (freqs <= min(SV_BAND_HZ[1], sr / 2.0))
    if not band.any():
        band = freqs > 0
    energy = np.empty(n_fr, dtype=np.float64)
    flat = np.empty(n_fr, dtype=np.float64)
    eps = 1e-12
    for b0 in range(0, n_fr, SV_BLOCK):
        b1 = min(n_fr, b0 + SV_BLOCK)
        idx = (np.arange(b0, b1)[:, None] * hop) + np.arange(win)[None, :]
        spec = np.fft.rfft(y[idx] * window, n=nfft, axis=1)
        pw = (spec.real ** 2 + spec.imag ** 2)[:, band] + eps
        energy[b0:b1] = 10.0 * np.log10(pw.sum(axis=1))
        flat[b0:b1] = np.exp(np.log(pw).mean(axis=1)) / pw.mean(axis=1)
    times = (np.arange(n_fr) * hop + win / 2.0) / sr
    return times, energy, flat


def speech_frames(y, sr: int):
    """(times_s, is_speech bool array). Empty arrays for silent/short audio."""
    times, energy, flat = _sv_frames(y, sr)
    if times.size == 0:
        return times, np.zeros(0, dtype=bool)
    peak = float(energy.max())
    valid = energy > peak - 80.0
    floor = float(np.percentile(energy[valid] if valid.any() else energy,
                                SV_FLOOR_PCT))
    floor = max(floor, peak - SV_FLOOR_RANGE_DB)
    loud = energy[energy > floor + SV_OPEN_DB]
    level = float(np.percentile(loud, SV_LEVEL_PCT)) if loud.size else peak
    open_ = (energy > max(floor + SV_OPEN_DB, level - SV_LEVEL_OPEN_DB)) & (
        (flat < SV_FLAT_MAX) | (energy > floor + SV_LOUD_DB))
    keep = energy > max(floor + SV_CLOSE_DB, level - SV_LEVEL_CLOSE_DB)
    speech = np.zeros(times.size, dtype=bool)
    active = False
    for i in range(times.size):
        if active:
            active = bool(keep[i])
        else:
            active = bool(open_[i])
        speech[i] = active
    return times, speech


def spectral_pauses(y, sr: int, min_pause_s: float = SV_MIN_PAUSE_S
                    ) -> List[Tuple[float, float]]:
    """Pauses BETWEEN speech, [(start_s, end_s)], time-ordered. Leading and
    trailing silence are not pauses (nothing to cut there)."""
    times, speech = speech_frames(y, sr)
    if not speech.any():
        return []
    half = SV_HOP_S / 2.0
    first = int(np.argmax(speech))
    last = int(speech.size - 1 - np.argmax(speech[::-1]))
    out: List[Tuple[float, float]] = []
    i = first
    while i <= last:
        if speech[i]:
            i += 1
            continue
        j = i
        while j <= last and not speech[j]:
            j += 1
        s, e = float(times[i] - half), float(times[j - 1] + half)
        if e - s >= min_pause_s:
            out.append((round(s, 3), round(e, 3)))
        i = j
    return out


def onset_after(y, sr: int, t: float, max_s: float = 2.0) -> Optional[float]:
    """Time of the first speech frame at or after *t* (within *max_s*), or
    None. For onset snapping (phase D): trim a piece's leading TTS silence
    so its speech starts exactly where the English speech starts."""
    times, speech = speech_frames(y, sr)
    if times.size == 0:
        return None
    lo = int(np.searchsorted(times, float(t) - SV_HOP_S / 2.0))
    hi = int(np.searchsorted(times, float(t) + float(max_s), side="right"))
    for i in range(max(0, lo), min(times.size, hi)):
        if speech[i]:
            return round(max(float(t), float(times[i] - SV_HOP_S / 2.0)), 3)
    return None
