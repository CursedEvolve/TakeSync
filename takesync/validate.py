"""Checks run on every export (and available from the CLI)."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .media import decode_audio, decode_gray_frames, find_tool, probe, run_capture
from .sync import ANALYSIS_SR, _local_ncc, _ref_window, quiet_threshold, _rms

BLACK_Y = 22  # mean luma at or below this (TV range black = 16) counts as black


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    severity: str = "error"   # error | warning | info

    def line(self) -> str:
        mark = "PASS" if self.ok else ("WARN" if self.severity == "warning" else "FAIL")
        return f"[{mark}] {self.name}: {self.detail}"


def _packet_pts(path: str, stream: str) -> np.ndarray:
    out = run_capture([find_tool("ffprobe"), "-v", "error", "-select_streams", stream,
                       "-show_entries", "packet=pts_time", "-of", "csv=p=0", path])
    vals = []
    for x in out.decode().split():
        try:
            vals.append(float(x.strip(",")))
        except ValueError:
            pass
    return np.sort(np.array(vals))


def validate_export(path: str, *, n_frames: int, fps: Fraction, width: int, height: int,
                    bed: np.ndarray | None = None, bed_rate: int | None = None,
                    cut_frames: list[int] | None = None,
                    cancel: threading.Event | None = None) -> list[Check]:
    checks: list[Check] = []
    info = probe(path, check_vfr=False)
    f = float(fps)
    frame = 1.0 / f

    ok = info.has_video and info.video_codec == "h264"
    checks.append(Check("video codec", ok, f"{info.video_codec or 'none'}"))
    checks.append(Check("resolution", (info.width, info.height) == (width, height),
                        f"{info.width}x{info.height} (expected {width}x{height})"))
    checks.append(Check("pixel format", info.pix_fmt == "yuv420p", info.pix_fmt))
    checks.append(Check("frame rate", info.fps == fps, f"{info.fps} (expected {fps})"))

    pts = _packet_pts(path, "v:0")
    checks.append(Check("frame count", len(pts) == n_frames, f"{len(pts)} (expected {n_frames})"))
    if len(pts) > 1:
        d = np.diff(pts)
        dev = float(np.max(np.abs(d - frame)))
        checks.append(Check("frame pacing", dev < 0.0011, f"max deviation from {frame * 1000:.3f} ms: {dev * 1000:.3f} ms"))
        checks.append(Check("video starts at 0", abs(pts[0]) < 0.0011, f"first frame at {pts[0] * 1000:.3f} ms"))

    if not info.audio:
        checks.append(Check("audio stream", False, "missing"))
    else:
        a = info.audio[0]
        checks.append(Check("audio stream", a.codec == "aac", f"{a.codec} {a.sample_rate} Hz {a.channels}ch"))
        checks.append(Check("audio starts at 0", abs(a.start) < 0.0011, f"{a.start * 1000:.3f} ms"))
        dur = a.duration or 0.0
        exp = n_frames * frame
        checks.append(Check("audio length", abs(dur - exp) <= frame + 0.03,
                            f"{dur:.3f} s vs video {exp:.3f} s"))
        if bed is not None and bed_rate:
            checks.extend(_check_audio_content(path, a.index, bed, bed_rate, exp, cancel))

    # luma of every frame: black frames, especially around cuts
    frames = decode_gray_frames(path, 36, 64, cancel=cancel)
    luma = frames.reshape(len(frames), -1).mean(axis=1)
    black = np.nonzero(luma <= BLACK_Y)[0]
    checks.append(Check("decoded frames", len(frames) == n_frames, f"{len(frames)} decoded"))
    checks.append(Check("black frames", len(black) == 0,
                        "none" if len(black) == 0 else f"{len(black)} (first at frame {black[0]})", "warning"))
    if cut_frames:
        near = sorted({j for c in cut_frames for j in (c - 1, c, c + 1) if 0 <= j < len(luma)})
        bad = [j for j in near if luma[j] <= BLACK_Y]
        checks.append(Check("frames around cuts", not bad,
                            f"{len(cut_frames)} cuts, {len(near)} frames checked" + (f", black: {bad[:10]}" if bad else "")))
    return checks


def _check_audio_content(path, stream_index, bed, bed_rate, duration, cancel) -> list[Check]:
    """The exported audio must be the bed, at lag 0, with nothing inserted or dropped."""
    from scipy.signal import resample_poly
    out = decode_audio(path, stream_index, ANALYSIS_SR, 1, cancel=cancel)
    ref = bed.mean(axis=1) if bed.ndim == 2 else bed
    g = np.gcd(ANALYSIS_SR, bed_rate)
    ref = resample_poly(ref.astype(np.float64), ANALYSIS_SR // g, bed_rate // g)
    n = int(duration * ANALYSIS_SR)
    out, ref = out[:n], ref[:n]
    q = quiet_threshold(ref)
    w, s = ANALYSIS_SR, int(0.02 * ANALYSIS_SR)
    lags, nccs, dropouts = [], [], 0
    for p in range(0, max(0, len(ref) - w), w):
        y = ref[p:p + w]
        if _rms(y) < q:
            continue
        x = out[p:p + w]
        if _rms(x) < q:
            dropouts += 1
            continue
        v = _local_ncc(y, _ref_window(out, p - s, w + 2 * s))
        k = int(np.argmax(v))
        lags.append((k - s) / ANALYSIS_SR * 1000)
        nccs.append(float(v[k]))
    if not lags:
        return [Check("audio content", True, "reference is silent; nothing to compare", "warning")]
    lags = np.array(lags)
    worst = float(np.max(np.abs(lags)))
    return [
        Check("audio = reference bed", float(np.median(nccs)) > 0.9 and dropouts == 0,
              f"median correlation {np.median(nccs):.3f}, {dropouts} silent dropouts in {len(lags) + dropouts} s"),
        Check("audio offset", worst <= 1.0, f"max lag {worst:.3f} ms over {len(lags)} windows"),
    ]


def report(checks: list[Check]) -> str:
    fails = sum(1 for c in checks if not c.ok and c.severity == "error")
    warns = sum(1 for c in checks if not c.ok and c.severity == "warning")
    head = "EXPORT VALIDATION: " + ("PASSED" if not fails else f"FAILED ({fails} problems)") + (
        f", {warns} warnings" if warns else "")
    return head + "\n" + "\n".join(c.line() for c in checks)


def passed(checks: list[Check]) -> bool:
    return all(c.ok or c.severity != "error" for c in checks)
