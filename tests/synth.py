"""Synthetic test media with known ground truth.

* make_song(): a 48 kHz stereo "track" with drums, bass, lead and pads, built
  from seeded sections so the two choruses are bit-identical (repeated phrases
  that can fool a naive correlator).
* make_take(): an MP4 like an OBS recording: 16:9 video + AAC playback audio
  starting at a known song position. Every video frame carries a barcode in
  the centre (the part that survives the 9:16 crop) encoding the take id and
  the song time in milliseconds the performer is "singing" in that frame, so
  exported videos can be checked frame by frame against the timeline.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
from scipy.signal import resample_poly

from fractions import Fraction

from takesync.media import find_tool

SR = 48000
BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT

# barcode layout in a 640x360 source frame (inside the centre 9:16 crop)
SRC_W, SRC_H = 640, 360
GRID_COLS, GRID_ROWS = 4, 6
GRID_X0, GRID_Y0, CELL_W, CELL_H = 226, 30, 47, 50
ID_BITS, TIME_BITS = 4, 20
TIME_BIAS = 200_000  # ms, so pre-roll (negative song time) is representable


def _env(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def _kick():
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    f = 50 + 120 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.12)


def _note(freq, dur, rng, kind="saw"):
    n = int(dur * SR)
    t = np.arange(n) / SR
    if kind == "saw":
        x = 2 * ((t * freq) % 1.0) - 1
    else:
        x = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t + rng.uniform(0, 6))
    return x * _env(n, 0.005, dur * 0.6)


def _section(bars, seed, lead=True):
    rng = np.random.default_rng(seed)
    n = int(bars * BAR * SR)
    out = np.zeros((n, 2))
    kick = _kick()
    scale = 220 * 2 ** (np.array([0, 3, 5, 7, 10, 12, 15]) / 12)
    hats = [rng.standard_normal(int(0.05 * SR)) * _env(int(0.05 * SR), 0.0005, 0.015) for _ in range(4)]
    bassline = rng.choice(scale[:4] / 4, size=8)
    for b in range(int(bars * 4)):
        p = int(b * BEAT * SR)
        k = kick[: n - p]
        out[p:p + len(k)] += 0.8 * k[:, None]
        h = hats[b % 4]
        q = p + int(BEAT * SR / 2)
        if q < n:
            h = h[: n - q]
            out[q:q + len(h), 0] += 0.25 * h
            out[q:q + len(h), 1] += 0.18 * h
        bn = _note(bassline[b % 8], BEAT * 0.9, rng)
        bn = bn[: n - p]
        out[p:p + len(bn)] += 0.3 * bn[:, None]
    if lead:
        for e in range(int(bars * 8)):
            if rng.random() < 0.25:
                continue
            p = int(e * BEAT / 2 * SR)
            ln = _note(rng.choice(scale) * 2, BEAT / 2 * 0.95, rng, kind="sine")[: n - p]
            pan = rng.uniform(0.3, 0.7)
            out[p:p + len(ln), 0] += 0.25 * ln * pan
            out[p:p + len(ln), 1] += 0.25 * ln * (1 - pan)
    return out


def make_song(seed: int = 1, short: bool = False) -> np.ndarray:
    """intro(2) verse(4) chorus(4) verse2(4) chorus(4, exact copy) outro(2) bars @120 BPM = 40 s.
    The chorus itself is a 2-bar phrase repeated twice, so repeats exist at 4 s and 16 s."""
    phrase = _section(2, seed + 30)
    chorus = np.concatenate([phrase, phrase])
    parts = [_section(2, seed + 10, lead=False), _section(4, seed + 20), chorus,
             _section(4, seed + 40), chorus, _section(2, seed + 50, lead=False)]
    if short:
        parts = parts[:3]
    song = np.concatenate(parts)
    return (0.5 * song / np.max(np.abs(song))).astype(np.float32)


def song_slice(song: np.ndarray, start_s: float, dur_s: float, sr: int = SR) -> np.ndarray:
    """song[start, start+dur) at 48 kHz with silence outside the song (start may be negative)."""
    n = int(round(dur_s * sr))
    i0 = int(round(start_s * sr))
    out = np.zeros((n, song.shape[1]), dtype=np.float32)
    a, b = max(0, i0), min(len(song), i0 + n)
    if b > a:
        out[a - i0:b - i0] = song[a:b]
    return out


def encode_bits(take_id: int, song_ms: int) -> list[int]:
    v = (take_id & 0xF) << TIME_BITS | ((song_ms + TIME_BIAS) & ((1 << TIME_BITS) - 1))
    return [(v >> (ID_BITS + TIME_BITS - 1 - i)) & 1 for i in range(ID_BITS + TIME_BITS)]


def decode_value(bits) -> tuple[int, int]:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v >> TIME_BITS, (v & ((1 << TIME_BITS) - 1)) - TIME_BIAS


def cell_centres():
    """(x, y) centres of the barcode cells in source pixel coordinates, row-major."""
    return [(GRID_X0 + c * CELL_W + CELL_W / 2, GRID_Y0 + r * CELL_H + CELL_H / 2)
            for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def make_take(path: str, song: np.ndarray, *, offset: float, duration: float, take_id: int,
              fps=60, audio_sr: int = SR, noise_db: float | None = None,
              audio_bitrate: str = "160k", audio_start: float = 0.0, camera_delay: float = 0.0,
              audio: bool = True, size: tuple[int, int] = (SRC_W, SRC_H), gain: float = 1.0) -> None:
    """Write a fake OBS recording.

    offset:       song time at media time 0 (negative = pre-roll before the song starts)
    audio_start:  the audio stream begins this many seconds after the video (timestamps)
    camera_delay: the picture lags the audio by this much (like a webcam)
    """
    w, h = size
    sx, sy = w / SRC_W, h / SRC_H
    fps = Fraction(fps)
    n_frames = int(round(duration * fps))
    with tempfile.TemporaryDirectory() as td:
        cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0"]
        if audio:
            a = song_slice(song, offset + audio_start, duration - audio_start) * gain
            if noise_db is not None:
                rng = np.random.default_rng(take_id)
                a = a + (10 ** (noise_db / 20) * rng.standard_normal(a.shape)).astype(np.float32)
            if audio_sr != SR:
                g = np.gcd(audio_sr, SR)
                a = resample_poly(a, audio_sr // g, SR // g, axis=0).astype(np.float32)
            raw = os.path.join(td, "a.f32")
            a.astype("<f4").tofile(raw)
            cmd += ["-itsoffset", f"{audio_start:.6f}", "-f", "f32le", "-ar", str(audio_sr), "-ac", "2", "-i", raw,
                    "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", audio_bitrate]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "120", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        base = np.full((h, w), 70, np.uint8)
        base[:, : int(60 * sx)] = 40 + 30 * take_id % 120  # per-take tint at the (cropped-away) edge
        cells = [(int((GRID_X0 + c * CELL_W) * sx), int((GRID_Y0 + r * CELL_H) * sy))
                 for r in range(GRID_ROWS) for c in range(GRID_COLS)]
        cw, ch = int((CELL_W - 6) * sx), int((CELL_H - 6) * sy)
        for k in range(n_frames):
            f = base.copy()
            song_ms = int(round((float(k / fps) + offset - camera_delay) * 1000))
            for bit, (x, y) in zip(encode_bits(take_id, song_ms), cells):
                f[y + 3:y + 3 + ch, x + 3:x + 3 + cw] = 235 if bit else 16
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing synthetic take")


def read_barcodes(frames: np.ndarray, crop_x: int, crop_w: int, crop_h: int, crop_y: int = 0,
                  src_w: int = SRC_W, src_h: int = SRC_H) -> list[tuple[int, int]]:
    """Decode (take_id, song_ms) from small gray output frames (n, H, W)."""
    _, H, W = frames.shape
    sx, sy = src_w / SRC_W, src_h / SRC_H
    pts = []
    for (cx, cy) in cell_centres():
        ox = (cx * sx - crop_x) / crop_w * W
        oy = (cy * sy - crop_y) / crop_h * H
        pts.append((int(round(ox)), int(round(oy))))
    out = []
    for f in frames:
        bits = [1 if f[max(0, y - 1):y + 2, max(0, x - 1):x + 2].mean() > 128 else 0 for x, y in pts]
        out.append(decode_value(bits))
    return out
