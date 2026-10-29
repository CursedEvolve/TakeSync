"""FFmpeg / ffprobe helpers: finding the tools, probing files, running jobs with
progress + cancel, and decoding audio to NumPy.

Timing convention used everywhere in TakeSync
---------------------------------------------
* "media time" of a file = presentation timestamp minus the container's
  start_time (the earliest timestamp of any stream). This is exactly what
  ``ffmpeg -ss`` seeks to, so media time is the only per-file clock we use.
* Audio decoded by :func:`decode_audio` is padded/trimmed so that sample ``k``
  is at media time ``k / sample_rate``, even if the audio stream starts later
  than the video or has gaps.
* "song time" is defined by the reference (see sync.py).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np


class MediaError(Exception):
    """A problem with an input file or an FFmpeg run, with a readable message."""


class Cancelled(Exception):
    """Raised when the user cancels a long-running operation."""


_POPEN_EXTRA = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
_TOOL_CACHE: dict[str, str] = {}


def find_tool(name: str) -> str:
    """Locate ffmpeg/ffprobe: a bundled ./ffmpeg/bin folder first, then PATH."""
    if name in _TOOL_CACHE:
        return _TOOL_CACHE[name]
    exe = name + (".exe" if os.name == "nt" else "")
    root = Path(__file__).resolve().parent.parent
    for cand in (root / "ffmpeg" / "bin" / exe, root / "ffmpeg" / exe):
        if cand.is_file():
            _TOOL_CACHE[name] = str(cand)
            return str(cand)
    found = shutil.which(name)
    if not found:
        raise MediaError(
            f"{name} was not found. Install FFmpeg (see README: 'winget install Gyan.FFmpeg') "
            f"or put {exe} in the TakeSync\\ffmpeg\\bin folder."
        )
    _TOOL_CACHE[name] = found
    return found


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _stderr_reader(stream, sink: deque) -> None:
    for raw in iter(stream.readline, b""):
        sink.append(raw.decode("utf-8", errors="replace").rstrip())
    stream.close()


def run_ffmpeg(args: list[str], *, cancel: threading.Event | None = None,
               on_frame=None, cwd: str | None = None) -> None:
    """Run ffmpeg with ``args`` (no shell). ``on_frame(n)`` gets the number of
    frames written so far. Raises MediaError with the tail of ffmpeg's log on
    failure, Cancelled if ``cancel`` gets set (the process is killed)."""
    cmd = [find_tool("ffmpeg"), "-hide_banner", "-nostdin", "-nostats",
           "-progress", "pipe:1", "-stats_period", "0.25", *args]
    err_tail: deque = deque(maxlen=60)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, cwd=cwd, **_POPEN_EXTRA)
    t = threading.Thread(target=_stderr_reader, args=(proc.stderr, err_tail), daemon=True)
    t.start()
    try:
        for raw in iter(proc.stdout.readline, b""):
            if cancel is not None and cancel.is_set():
                proc.kill()
                proc.wait()
                raise Cancelled()
            line = raw.decode("utf-8", errors="replace").strip()
            if on_frame and line.startswith("frame="):
                try:
                    on_frame(int(line[6:]))
                except ValueError:
                    pass
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        t.join(timeout=2)
    _check_cancel(cancel)
    if proc.returncode != 0:
        tail = "\n".join(list(err_tail)[-15:])
        raise MediaError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")


def run_capture(cmd: list[str], *, cancel: threading.Event | None = None,
                chunk: int = 1 << 20) -> bytes:
    """Run a tool and return its stdout bytes, read in chunks (cancellable)."""
    err_tail: deque = deque(maxlen=40)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, **_POPEN_EXTRA)
    t = threading.Thread(target=_stderr_reader, args=(proc.stderr, err_tail), daemon=True)
    t.start()
    parts = []
    try:
        while True:
            if cancel is not None and cancel.is_set():
                proc.kill()
                raise Cancelled()
            buf = proc.stdout.read(chunk)
            if not buf:
                break
            parts.append(buf)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        t.join(timeout=2)
    if proc.returncode != 0:
        tail = "\n".join(list(err_tail)[-10:])
        raise MediaError(f"{Path(cmd[0]).stem} failed (exit {proc.returncode}):\n{tail}")
    return b"".join(parts)


# --------------------------------------------------------------------------- probing

@dataclass
class AudioStream:
    index: int            # absolute stream index in the file
    codec: str
    sample_rate: int
    channels: int
    start: float          # media time of first audio timestamp
    duration: float | None


@dataclass
class MediaInfo:
    path: str
    duration: float
    format_start: float
    has_video: bool = False
    width: int = 0               # displayed size (after rotation)
    height: int = 0
    rotation: int = 0
    fps: Fraction | None = None  # frame rate we treat the video as
    r_fps: Fraction | None = None
    avg_fps: Fraction | None = None
    vfr_suspect: bool = False
    video_index: int = -1
    video_start: float = 0.0     # media time of the first video frame
    video_duration: float = 0.0
    video_codec: str = ""
    pix_fmt: str = ""
    color: dict = field(default_factory=dict)
    nb_frames: int | None = None
    audio: list[AudioStream] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def video_end(self) -> float:
        return self.video_start + self.video_duration

    def summary(self) -> str:
        parts = [f"{self.name}: {self.duration:.3f}s"]
        if self.has_video:
            fps = f"{float(self.fps):.3f}" if self.fps else "?"
            parts.append(f"{self.width}x{self.height} @ {fps} fps {self.video_codec}/{self.pix_fmt}")
            parts.append(f"video media time {self.video_start:.3f}-{self.video_end:.3f}s")
        else:
            parts.append("no video")
        if self.audio:
            a = self.audio[0]
            parts.append(f"audio {a.codec} {a.sample_rate} Hz {a.channels}ch starts {a.start:.3f}s"
                         + (f" (+{len(self.audio) - 1} more tracks)" if len(self.audio) > 1 else ""))
        else:
            parts.append("NO AUDIO")
        return " | ".join(parts)


def _frac(s: str | None) -> Fraction | None:
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        f = Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return f if f > 0 else None


def _float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


_STANDARD_FPS = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
                 Fraction(30), Fraction(50), Fraction(60000, 1001), Fraction(60),
                 Fraction(120), Fraction(144)]


def nearest_standard_fps(f: float) -> Fraction:
    return min(_STANDARD_FPS, key=lambda s: abs(float(s) - f))


def probe(path: str | os.PathLike, *, check_vfr: bool = True) -> MediaInfo:
    path = str(path)
    if not os.path.isfile(path):
        raise MediaError(f"File not found: {path}")
    out = run_capture([find_tool("ffprobe"), "-v", "error", "-print_format", "json",
                       "-show_format", "-show_streams", path])
    try:
        data = json.loads(out.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise MediaError(f"ffprobe returned unreadable output for {path}: {e}")
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    fstart = _float(fmt.get("start_time"), 0.0) or 0.0
    info = MediaInfo(path=path, duration=_float(fmt.get("duration"), 0.0) or 0.0, format_start=fstart)

    vids = [s for s in streams if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic")]
    if vids:
        v = vids[0]
        info.has_video = True
        info.video_index = int(v["index"])
        info.video_codec = v.get("codec_name", "")
        info.pix_fmt = v.get("pix_fmt", "")
        w, h = int(v.get("width", 0)), int(v.get("height", 0))
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(round(_float(sd["rotation"], 0)))
        if "rotate" in (v.get("tags") or {}):
            rot = int(_float(v["tags"]["rotate"], 0))
        rot %= 360
        info.rotation = rot
        if rot in (90, 270):
            w, h = h, w
        info.width, info.height = w, h
        info.r_fps = _frac(v.get("r_frame_rate"))
        info.avg_fps = _frac(v.get("avg_frame_rate"))
        info.video_start = (_float(v.get("start_time"), fstart) or 0.0) - fstart
        vd = _float(v.get("duration"))
        nbf = v.get("nb_frames")
        info.nb_frames = int(nbf) if nbf and str(nbf).isdigit() else None
        base = info.avg_fps or info.r_fps
        if vd is None and info.nb_frames and base:
            vd = info.nb_frames / float(base)
        if vd is None:
            vd = max(0.0, info.duration - info.video_start)
        info.video_duration = vd
        for key in ("color_space", "color_primaries", "color_transfer", "color_range"):
            if v.get(key) and v[key] != "unknown":
                info.color[key] = v[key]
        if base is None:
            info.warnings.append("Could not read the frame rate.")
        else:
            info.fps = base
            if info.r_fps and info.avg_fps and abs(float(info.r_fps) - float(info.avg_fps)) / float(info.avg_fps) > 0.01:
                info.vfr_suspect = True
            # Treat near-standard rates as the standard rate (59.99 -> 60).
            std = nearest_standard_fps(float(base))
            if abs(float(std) - float(base)) / float(std) < 0.01:
                info.fps = std
        if "10" in info.pix_fmt or "12" in info.pix_fmt:
            info.warnings.append(f"High bit-depth video ({info.pix_fmt}); output will be 8-bit yuv420p.")
        if info.color.get("color_transfer") in ("smpte2084", "arib-std-b67"):
            info.warnings.append("HDR video detected; TakeSync does not tone-map, colours may look washed out.")
        if rot:
            info.warnings.append(f"Rotation metadata {rot} deg; FFmpeg will auto-rotate.")
        if check_vfr:
            _check_vfr(info)
        if info.vfr_suspect:
            info.warnings.append("Variable frame rate suspected; frames are resampled to a constant rate on export.")

    for s in streams:
        if s.get("codec_type") != "audio":
            continue
        info.audio.append(AudioStream(
            index=int(s["index"]), codec=s.get("codec_name", ""),
            sample_rate=int(s.get("sample_rate", 0) or 0), channels=int(s.get("channels", 0) or 0),
            start=(_float(s.get("start_time"), fstart) or 0.0) - fstart,
            duration=_float(s.get("duration"))))
    if not info.audio:
        info.warnings.append("No audio stream: automatic sync impossible, a manual offset is required.")
    return info


def _check_vfr(info: MediaInfo, seconds: float = 6.0) -> None:
    """Look at the first few seconds of packet timestamps for irregular spacing."""
    try:
        out = run_capture([find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+{seconds}",
                           "-show_entries", "packet=pts_time", "-of", "csv=p=0", info.path])
    except MediaError:
        return
    pts = sorted(v for v in (_float(x.strip(",")) for x in out.decode().split()) if v is not None)
    if len(pts) < 10 or not info.fps:
        return
    d = np.diff(np.array(pts))
    nominal = 1.0 / float(info.fps)
    bad = np.mean(np.abs(d - nominal) > 0.25 * nominal)
    if bad > 0.02:
        info.vfr_suspect = True


# --------------------------------------------------------------------------- audio

def decode_audio(path: str, stream_index: int, sample_rate: int, channels: int = 1, *,
                 start_sample: int = 0, num_samples: int | None = None,
                 cancel: threading.Event | None = None) -> np.ndarray:
    """Decode one audio stream to float32, sample k = media time k / sample_rate.

    Gaps/late starts are filled with silence by aresample (async + first_pts=0).
    Only samples [start_sample, start_sample + num_samples) are kept in memory;
    the output is zero padded if the file is shorter. Returns shape (n,) for
    mono, (n, channels) otherwise.
    """
    cmd = [find_tool("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", path,
           "-map", f"0:{stream_index}", "-vn", "-sn", "-dn",
           "-af", f"aresample={sample_rate}:async=1:first_pts=0",
           "-ac", str(channels), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    err_tail: deque = deque(maxlen=40)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, **_POPEN_EXTRA)
    t = threading.Thread(target=_stderr_reader, args=(proc.stderr, err_tail), daemon=True)
    t.start()
    frame_bytes = 4 * channels
    skip = start_sample * frame_bytes
    want = None if num_samples is None else num_samples * frame_bytes
    parts: list[bytes] = []
    got = 0
    pos = 0
    try:
        while True:
            if cancel is not None and cancel.is_set():
                proc.kill()
                raise Cancelled()
            buf = proc.stdout.read(1 << 20)
            if not buf:
                break
            lo, hi = pos, pos + len(buf)
            pos = hi
            if hi <= skip:
                continue
            piece = buf[max(0, skip - lo):]
            if want is not None:
                piece = piece[:max(0, want - got)]
            if piece:
                parts.append(piece)
                got += len(piece)
            if want is not None and got >= want:
                proc.kill()
                break
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        t.join(timeout=2)
    if proc.returncode not in (0, None) and not (want is not None and got >= want):
        tail = "\n".join(list(err_tail)[-10:])
        raise MediaError(f"Could not decode audio from {Path(path).name}:\n{tail}")
    data = b"".join(parts)
    data = data[: len(data) - len(data) % frame_bytes]
    arr = np.frombuffer(data, dtype=np.float32).copy()
    if channels > 1:
        arr = arr.reshape(-1, channels)
    if num_samples is not None and len(arr) < num_samples:
        pad = np.zeros((num_samples - len(arr),) + arr.shape[1:], dtype=np.float32)
        arr = np.concatenate([arr, pad])
    return arr


def decode_gray_frames(path: str, width: int, height: int, *, vf_pre: str = "",
                       cancel: threading.Event | None = None) -> np.ndarray:
    """Decode every video frame of ``path`` scaled to width x height 8-bit luma.
    Used by export validation and the synthetic tests (small sizes only)."""
    chain = (vf_pre + "," if vf_pre else "") + f"scale={width}:{height}:flags=area,format=gray"
    out = run_capture([find_tool("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error",
                       "-i", path, "-map", "0:v:0", "-vf", chain, "-fps_mode", "passthrough",
                       "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], cancel=cancel)
    n = len(out) // (width * height)
    return np.frombuffer(out[: n * width * height], dtype=np.uint8).reshape(n, height, width)
