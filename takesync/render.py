"""FFmpeg rendering: per-segment H.264 encode straight from the source files,
stream-copy concat, one continuous reference audio bed.

Frame selection: output frame j shows song time t = t0 + j/fps. For a take with
offset o (and a camera delay d) that is source media time m = t - o + d. Each
segment is decoded from the original file at full resolution; the fps filter
picks, for every output slot, the source frame nearest to m (round=near), then
the 9:16 crop is taken before the Lanczos scale. Each source frame is encoded
exactly once.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .media import Cancelled, MediaError, decode_audio, run_capture, run_ffmpeg, find_tool
from .planner import CutPlan


@dataclass
class RenderSource:
    path: str
    name: str
    offset: float          # song_time = media_time + offset (audio)
    video_start: float     # media time of first frame
    video_end: float       # media time just after the last frame
    width: int
    height: int
    color: dict = field(default_factory=dict)


@dataclass
class RenderSettings:
    width: int = 1080
    height: int = 1920
    crf: int = 17
    preset: str = "slow"
    audio_bitrate: str = "320k"
    crop_center: float = 0.5        # horizontal centre of the crop, 0..1 of source width
    camera_delay: float = 0.0       # seconds the picture lags the playback audio
    labels: bool = False            # burn take name/time (diagnostics only)

    @classmethod
    def preview(cls, **kw) -> "RenderSettings":
        base = dict(width=540, height=960, crf=26, preset="veryfast", audio_bitrate="128k")
        base.update(kw)
        return cls(**base)


@dataclass
class BedSource:
    path: str
    stream_index: int
    sample_rate: int
    channels: int


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def crop_rect(w: int, h: int, center: float = 0.5, aspect: Fraction = Fraction(9, 16)) -> tuple[int, int, int, int]:
    """Largest 9:16 rectangle (cw, ch, x, y), even-sized, horizontally centred at ``center``."""
    if Fraction(w, h) > aspect:
        ch = _even(h)
        cw = _even(h * aspect)
        x = int(round(center * w - cw / 2))
        x = min(max(0, x), w - cw) // 2 * 2
        y = (h - ch) // 2 // 2 * 2
    else:
        cw = _even(w)
        ch = _even(w / aspect)
        x = (w - cw) // 2 // 2 * 2
        y = (h - ch) // 2 // 2 * 2
    return cw, ch, x, y


def _fps_str(fps: Fraction) -> str:
    return f"{fps.numerator}/{fps.denominator}"


def _timescale(fps: Fraction) -> int:
    return fps.numerator * 256 if fps.denominator == 1 else fps.numerator


def media_time(plan: CutPlan, src: RenderSource, frame: int, camera_delay: float) -> float:
    return plan.frame_time(frame) - src.offset + camera_delay


def check_bounds(plan: CutPlan, sources: list[RenderSource], camera_delay: float = 0.0) -> list[str]:
    """Hard guard: every output frame must map to an existing source frame."""
    errs = []
    half = 0.5 / float(plan.fps)
    for i, s in enumerate(plan.segments):
        src = sources[s.take]
        m0 = media_time(plan, src, s.start, camera_delay)
        m1 = media_time(plan, src, s.end - 1, camera_delay)
        last_frame = src.video_end - 1.0 / float(plan.fps)
        if m0 < src.video_start - half or m1 > last_frame + half or m0 < 0:
            errs.append(f"Segment {i + 1} needs {src.name} media time {m0:.3f}-{m1:.3f} s, "
                        f"but its video only spans {src.video_start:.3f}-{src.video_end:.3f} s.")
    return errs


_SMPTE170 = ("smpte170m", "bt470bg")


def _scale_filter(settings: RenderSettings, src: RenderSource) -> str:
    opts = ["flags=lanczos+accurate_rnd+full_chroma_int"]
    if src.color.get("color_space") in _SMPTE170:
        opts += ["in_color_matrix=bt601", "out_color_matrix=bt709"]
    if src.color.get("color_range") == "pc":
        opts += ["in_range=full", "out_range=limited"]
    return f"scale={settings.width}:{settings.height}:" + ":".join(opts)


def frame_chain(src: RenderSource, fps: Fraction, m_first: float, settings: RenderSettings,
                out_w: int | None = None, out_h: int | None = None) -> tuple[float, str]:
    """(seek_seconds, filter) producing frames starting at the source frame nearest to m_first."""
    k = max(0, min(2, int(math.floor(m_first * float(fps) + 1e-9))))
    seek = m_first - k / float(fps)
    cw, ch, x, y = crop_rect(src.width, src.height, settings.crop_center)
    scale = _scale_filter(settings, src)
    if out_w:
        scale = f"scale={out_w}:{out_h}:flags=lanczos"
    chain = (f"fps={_fps_str(fps)}:start_time=0:round=near,trim=start_frame={k},setpts=PTS-STARTPTS,"
             f"crop={cw}:{ch}:{x}:{y},{scale},setsar=1,format=yuv420p")
    return seek, chain


def _find_font() -> str | None:
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.isfile(p):
            return p
    return None


def fmt_time(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60)}:{t % 60:06.3f}"


def render_video(plan: CutPlan, sources: list[RenderSource], settings: RenderSettings, workdir: str, *,
                 cancel: threading.Event | None = None, progress=None) -> str:
    """Encode every segment and write a concat list. Returns the list file name (in workdir)."""
    errs = check_bounds(plan, sources, settings.camera_delay)
    if errs:
        raise MediaError("Cut list is out of bounds:\n" + "\n".join(errs))
    fps = plan.fps
    total = plan.n_frames
    done = 0
    font = None
    if settings.labels:
        f = _find_font()
        if f:
            shutil.copy(f, os.path.join(workdir, "font.ttf"))
            font = "font.ttf"
    names = []
    for i, seg in enumerate(plan.segments):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        src = sources[seg.take]
        n = seg.end - seg.start
        m0 = media_time(plan, src, seg.start, settings.camera_delay)
        seek, chain = frame_chain(src, fps, m0, settings)
        if settings.labels:
            label = f"{src.name}  |  song {fmt_time(plan.frame_time(seg.start))}"
            with open(os.path.join(workdir, f"label_{i:04d}.txt"), "w", encoding="utf-8") as fh:
                fh.write(label)
            dt = (f"drawtext=textfile=label_{i:04d}.txt:fontcolor=white:fontsize=h/30:box=1:"
                  f"boxcolor=black@0.6:boxborderw=8:x=(w-tw)/2:y=h*0.03")
            if font:
                dt += f":fontfile={font}"
            chain += "," + dt
        name = f"seg_{i:04d}.mp4"
        args = ["-y", "-ss", f"{seek:.6f}", "-i", src.path, "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", chain, "-frames:v", str(n), "-fps_mode", "cfr", "-r", _fps_str(fps),
                "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
                "-profile:v", "high", "-level:v", "4.2", "-pix_fmt", "yuv420p",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
                "-video_track_timescale", str(_timescale(fps)), name]

        def cb(frames, base=done):
            if progress:
                progress(base + min(frames, n), total)
        try:
            run_ffmpeg(args, cancel=cancel, on_frame=cb, cwd=workdir)
        except MediaError:
            if not settings.labels:
                raise
            # drawtext unavailable (no font): render without the label
            args[args.index("-vf") + 1] = chain.rsplit(",drawtext", 1)[0]
            run_ffmpeg(args, cancel=cancel, on_frame=cb, cwd=workdir)
        got = count_frames(os.path.join(workdir, name))
        if got != n:
            raise MediaError(f"Segment {i + 1} from {src.name}: expected {n} frames, got {got}. "
                             "The source may end early or have damaged frames.")
        done += n
        names.append(name)
        if progress:
            progress(done, total)
    with open(os.path.join(workdir, "segments.txt"), "w", encoding="utf-8") as fh:
        for nm in names:
            fh.write(f"file '{nm}'\n")
    return "segments.txt"


def count_frames(path: str) -> int:
    out = run_capture([find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
                       "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path])
    try:
        return int(out.decode().strip().strip(","))
    except ValueError:
        return -1


def build_bed(bed: BedSource, plan: CutPlan, workdir: str, cancel: threading.Event | None = None) -> tuple[str, np.ndarray]:
    """Cut the reference audio for song times [t0, t0 + duration) as one piece.
    Song time == media time of the bed file (reference offset is 0 by definition).
    Returns (raw f32 file name in workdir, the samples)."""
    sr, ch = bed.sample_rate, max(1, min(2, bed.channels))
    start = int(round(plan.t0 * sr))
    count = int(round(plan.n_frames / float(plan.fps) * sr))
    lead = max(0, -start)
    data = decode_audio(bed.path, bed.stream_index, sr, ch, start_sample=max(0, start),
                        num_samples=count - lead, cancel=cancel)
    if lead:
        data = np.concatenate([np.zeros((lead,) + data.shape[1:], np.float32), data])
    data = np.ascontiguousarray(data.reshape(count, ch) if ch > 1 else data.reshape(count, 1))
    data.astype("<f4").tofile(os.path.join(workdir, "bed.f32"))
    return "bed.f32", data


def mux(workdir: str, list_name: str, bed_name: str, bed: BedSource, out_path: str,
        settings: RenderSettings, cancel: threading.Event | None = None) -> None:
    ch = max(1, min(2, bed.channels))
    run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", list_name,
                "-f", "f32le", "-ar", str(bed.sample_rate), "-ac", str(ch), "-i", bed_name,
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", settings.audio_bitrate, "-movflags", "+faststart",
                "-metadata", "comment=Made with TakeSync", os.path.abspath(out_path)],
               cancel=cancel, cwd=workdir)


def render(plan: CutPlan, sources: list[RenderSource], bed: BedSource, settings: RenderSettings,
           out_path: str, *, cancel: threading.Event | None = None, progress=None,
           keep_workdir: str | None = None) -> np.ndarray:
    """Full pipeline into ``out_path`` (written as .partial first, then renamed).
    Returns the audio bed samples (for validation)."""
    out_path = os.path.abspath(out_path)
    workdir = keep_workdir or tempfile.mkdtemp(prefix="takesync_")
    os.makedirs(workdir, exist_ok=True)
    partial = str(Path(out_path).with_suffix(".partial.mp4"))
    try:
        list_name = render_video(plan, sources, settings, workdir, cancel=cancel, progress=progress)
        bed_name, bed_data = build_bed(bed, plan, workdir, cancel)
        mux(workdir, list_name, bed_name, bed, partial, settings, cancel)
        os.replace(partial, out_path)
        return bed_data
    finally:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def grab_frame(src: RenderSource, fps: Fraction, m: float, settings: RenderSettings,
               w: int, h: int, label: str | None = None, cancel=None) -> np.ndarray:
    """One cropped RGB frame (h, w, 3) at media time m, selected exactly like the renderer."""
    seek, chain = frame_chain(src, fps, m, settings, w, h)
    chain = chain.replace(",format=yuv420p", "")
    out = run_capture([find_tool("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error",
                       "-ss", f"{seek:.6f}", "-i", src.path, "-map", "0:v:0", "-vf", chain,
                       "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], cancel=cancel)
    if len(out) < w * h * 3:
        return np.zeros((h, w, 3), np.uint8)
    return np.frombuffer(out[: w * h * 3], np.uint8).reshape(h, w, 3)


def write_png(img: np.ndarray, path: str) -> None:
    h, w, _ = img.shape
    import subprocess
    from .media import _POPEN_EXTRA
    p = subprocess.run([find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "pipe:0", "-frames:v", "1", path],
                       input=np.ascontiguousarray(img).tobytes(), capture_output=True, **_POPEN_EXTRA)
    if p.returncode != 0:
        raise MediaError("Could not write PNG: " + p.stderr.decode(errors="replace")[-300:])
