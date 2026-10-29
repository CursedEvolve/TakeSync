"""Processing state shared by the CLI and the GUI (no UI code here)."""
from __future__ import annotations

import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import sync
from .media import MediaError, MediaInfo, probe
from .planner import CutPlan, detect_onsets, diagnostic_plan, plan_cuts, validate_plan
from .render import (BedSource, RenderSettings, RenderSource, crop_rect, fmt_time, grab_frame,
                     render, write_png)
from .validate import validate_export

MARGIN_FRAMES = 2  # keep this far from the ends of every take's valid footage


@dataclass
class Take:
    info: MediaInfo
    audio: np.ndarray | None = None          # 16 kHz analysis audio
    result: sync.AlignResult | None = None
    manual_offset: float | None = None
    enabled: bool = True

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def offset(self) -> float | None:
        if self.manual_offset is not None:
            return self.manual_offset
        return self.result.offset if self.result else None


@dataclass
class Project:
    takes: list[Take] = field(default_factory=list)
    ref_wav: MediaInfo | None = None          # optional reference audio file
    ref_audio: np.ndarray | None = None       # its 16 kHz analysis audio
    ref_take: int = 0                         # used when ref_wav is None
    camera_delay: float = 0.0
    audio_track: int = 0
    analyzed: bool = False
    log: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ loading
    def load(self, paths: list[str], ref_wav: str | None = None, *, cancel=None, progress=None) -> None:
        seen = set()
        self.takes = []
        for i, p in enumerate(paths):
            key = os.path.normcase(os.path.abspath(p))
            if key in seen:
                continue
            seen.add(key)
            if progress:
                progress(f"Reading {os.path.basename(p)}", i, len(paths))
            info = probe(p)
            if not info.has_video:
                raise MediaError(f"{info.name} has no video stream.")
            self.takes.append(Take(info))
        for i, t in enumerate(self.takes):
            if progress:
                progress(f"Decoding audio of {t.name}", i, len(self.takes))
            t.audio = sync.load_analysis_audio(t.info, self.audio_track, cancel)
        self.set_reference_wav(ref_wav, cancel=cancel)
        self.ref_take = self.auto_reference_take()
        self.analyzed = False

    def set_reference_wav(self, path: str | None, *, cancel=None) -> None:
        if not path:
            self.ref_wav, self.ref_audio = None, None
            return
        info = probe(path, check_vfr=False)
        if not info.audio:
            raise MediaError(f"Reference {info.name} has no audio.")
        self.ref_wav = info
        self.ref_audio = sync.load_analysis_audio(info, 0, cancel)

    def auto_reference_take(self) -> int:
        """Longest take with clearly audible playback audio."""
        best, best_len = 0, -1.0
        for i, t in enumerate(self.takes):
            if t.audio is None:
                continue
            act = sync.active_region(t.audio)
            length = (act[1] - act[0]) if act else 0.0
            if length > best_len:
                best, best_len = i, length
        return best

    # ------------------------------------------------------------------ analysis
    @property
    def reference_audio(self) -> np.ndarray | None:
        if self.ref_wav is not None:
            return self.ref_audio
        return self.takes[self.ref_take].audio if self.takes else None

    @property
    def reference_name(self) -> str:
        if self.ref_wav is not None:
            return self.ref_wav.name
        return self.takes[self.ref_take].name if self.takes else "-"

    def analyze(self, *, cancel=None, progress=None) -> None:
        ref = self.reference_audio
        fps = float(self.output_fps())
        for i, t in enumerate(self.takes):
            if progress:
                progress(f"Aligning {t.name}", i, len(self.takes))
            t.manual_offset = None
            if self.ref_wav is None and i == self.ref_take:
                r = sync.AlignResult(offset=0.0, status="reference", score=1.0, agreement=1.0)
                r.matched = sync.active_region(t.audio) if t.audio is not None else None
                if t.audio is None:
                    r.warnings.append("Reference take has no audio.")
                t.result = r
            elif t.audio is None:
                t.result = sync.AlignResult(offset=None, warnings=["No audio track: set the offset manually."])
            elif ref is None:
                t.result = sync.AlignResult(offset=None, warnings=["Reference has no audio."])
            else:
                t.result = sync.align(ref, t.audio, fps=fps, cancel=cancel)
            t.enabled = t.result.usable
        self.analyzed = True

    def set_manual_offset(self, i: int, offset: float | None) -> None:
        t = self.takes[i]
        if offset is None:
            t.manual_offset = None
            return
        t.manual_offset = offset
        t.result = sync.evaluate_offset(self.reference_audio, t.audio, offset, status="manual")
        t.enabled = True

    def accept(self, i: int, enabled: bool) -> None:
        """User confirms (or excludes) a take, e.g. one flagged low/ambiguous."""
        t = self.takes[i]
        t.enabled = enabled and t.offset is not None

    # ------------------------------------------------------------------ timeline
    def output_fps(self) -> Fraction:
        if not self.takes:
            return Fraction(60)
        if self.ref_wav is None and self.takes[self.ref_take].info.fps:
            return self.takes[self.ref_take].info.fps
        rates = Counter(t.info.fps for t in self.takes if t.info.fps)
        if not rates:
            return Fraction(60)
        top = max(rates.values())
        return max(r for r, c in rates.items() if c == top)

    def usable(self, i: int) -> tuple[float, float] | None:
        """Song-time span of take i with valid footage and matching audio."""
        t = self.takes[i]
        if t.offset is None:
            return None
        o, d = t.offset, self.camera_delay
        margin = MARGIN_FRAMES / float(self.output_fps())
        a, b = t.info.video_start + o - d + margin, t.info.video_end + o - d - margin
        m = t.result.matched if t.result else None
        if m is not None:
            a, b = max(a, m[0]), min(b, m[1])
        elif t.manual_offset is None and t.audio is not None and t.result is not None \
                and t.result.status != "reference":
            return None  # audio never matches the song: nothing trustworthy to use
        return (a, b) if b - a > 1.0 / float(self.output_fps()) else None

    def usable_map(self) -> dict[int, tuple[float, float]]:
        out = {}
        for i, t in enumerate(self.takes):
            if t.enabled:
                iv = self.usable(i)
                if iv:
                    out[i] = iv
        return out

    def bed(self) -> BedSource:
        info = self.ref_wav if self.ref_wav is not None else self.takes[self.ref_take].info
        if not info.audio:
            raise MediaError(f"The reference {info.name} has no audio to use as the soundtrack.")
        a = info.audio[min(self.audio_track if self.ref_wav is None else 0, len(info.audio) - 1)]
        return BedSource(info.path, a.index, a.sample_rate or 48000, a.channels or 2)

    def bed_span(self) -> tuple[float, float]:
        info = self.ref_wav if self.ref_wav is not None else self.takes[self.ref_take].info
        if not info.audio:
            return (0.0, 0.0)
        a = info.audio[0]
        end = a.start + (a.duration if a.duration else info.duration - a.start)
        return (0.0, end)

    def auto_excerpt(self) -> tuple[float, float] | None:
        """Song-time span shared by every enabled take (and the soundtrack)."""
        u = self.usable_map()
        if not u:
            return None
        a = max(v[0] for v in u.values())
        b = min(v[1] for v in u.values())
        s0, s1 = self.bed_span()
        a, b = max(a, s0), min(b, s1)
        return (a, b) if b - a > 0.5 else None

    def union_span(self) -> tuple[float, float] | None:
        u = self.usable_map()
        if not u:
            return None
        s0, s1 = self.bed_span()
        return (max(s0, min(v[0] for v in u.values())), min(s1, max(v[1] for v in u.values())))

    # ------------------------------------------------------------------ planning
    def make_plan(self, start: float, end: float, pacing: str = "natural", seed: int = 1) -> CutPlan:
        u = self.usable_map()
        if not u:
            raise ValueError("No usable takes. Run Analyze and fix the flagged takes first.")
        s0, s1 = self.bed_span()
        warn = []
        if start < s0 or end > s1:
            warn.append(f"Excerpt clipped to the soundtrack ({fmt_time(s0)}-{fmt_time(s1)}).")
            start, end = max(start, s0), min(end, s1)
        onsets, conf = detect_onsets(self.reference_audio, sync.ANALYSIS_SR, start, end)
        plan = plan_cuts(start, end, self.output_fps(), u, onsets, conf, pacing, seed)
        plan.warnings = warn + plan.warnings
        if not conf and len(u) > 1:
            plan.warnings.append("Beat detection was not confident; cutting at regular intervals.")
        return plan

    def check_plan(self, plan: CutPlan) -> list[str]:
        return validate_plan(plan, self.usable_map())

    def sources(self) -> list[RenderSource]:
        return [RenderSource(t.info.path, t.name, t.offset if t.offset is not None else 0.0,
                             t.info.video_start, t.info.video_end, t.info.width, t.info.height, t.info.color)
                for t in self.takes]

    def export(self, plan: CutPlan, out_path: str, settings: RenderSettings, *, cancel=None,
               progress=None, validate: bool = True):
        errs = self.check_plan(plan)
        if errs:
            raise ValueError("Cut list problems:\n" + "\n".join(errs))
        settings.camera_delay = self.camera_delay
        bed_src = self.bed()
        bed = render(plan, self.sources(), bed_src, settings, out_path, cancel=cancel, progress=progress)
        if not validate:
            return []
        if progress:
            progress(plan.n_frames, plan.n_frames)
        return validate_export(out_path, n_frames=plan.n_frames, fps=plan.fps, width=settings.width,
                               height=settings.height, bed=bed, bed_rate=bed_src.sample_rate,
                               cut_frames=plan.cut_frames(), cancel=cancel)

    # ------------------------------------------------------------------ reporting
    def alignment_report(self) -> str:
        lines = [f"Reference: {self.reference_name}" + (" (audio file)" if self.ref_wav else " (take)"),
                 f"Output frame rate: {float(self.output_fps()):.3f} fps   camera delay: {self.camera_delay * 1000:.0f} ms",
                 "Timing: song_time = media_time + offset  (media time = file time from its first timestamp)", ""]
        for i, t in enumerate(self.takes):
            r = t.result
            off = t.offset
            iv = self.usable(i)
            lines.append(f"Take {i + 1}: {t.name}")
            lines.append(f"  {t.info.summary()}")
            if r is None:
                lines.append("  not analysed")
                continue
            lines.append(f"  status: {r.status.upper()}{'  (manual)' if t.manual_offset is not None else ''}"
                         f"   used: {'yes' if t.enabled and iv else 'NO'}")
            lines.append(f"  offset: {'-' if off is None else f'{off * 1000:+.2f} ms ({off:+.4f} s)'}")
            if r.status not in ("reference",):
                lines.append(f"  match score {r.score:.3f}, windows agreeing {r.agreement:.0%} of {r.windows},"
                             f" ambiguity {r.ambiguity:.2f}, drift {r.drift_ms:+.2f} ms")
                for o, s in r.alternatives[:2]:
                    lines.append(f"    next best: {o:+.4f} s (score {s:.3f})")
            lines.append(f"  usable song time: {fmt_time(iv[0]) + ' - ' + fmt_time(iv[1]) if iv else 'none'}")
            for w in t.info.warnings + r.warnings:
                lines.append(f"  WARNING: {w}")
        ex = self.auto_excerpt()
        lines.append("")
        lines.append(f"Shared overlap of all used takes: {fmt_time(ex[0]) + ' - ' + fmt_time(ex[1]) if ex else 'NONE'}"
                     + (f" ({ex[1] - ex[0]:.2f} s)" if ex else ""))
        return "\n".join(lines)

    # ------------------------------------------------------------------ diagnostics
    def contact_sheet(self, path: str, times: list[float], settings: RenderSettings | None = None,
                      cell=(180, 320), cancel=None) -> None:
        """Grid: one row per song time, one column per used take, same frame selection as export."""
        settings = settings or RenderSettings()
        settings.camera_delay = self.camera_delay
        u = self.usable_map()
        cols = sorted(u)
        w, h = cell
        img = np.zeros((len(times) * (h + 4), len(cols) * (w + 4), 3), np.uint8)
        srcs = self.sources()
        fps = self.output_fps()
        for r, t in enumerate(times):
            for c, k in enumerate(cols):
                if not (u[k][0] <= t <= u[k][1]):
                    continue
                m = t - srcs[k].offset + self.camera_delay
                img[r * (h + 4):r * (h + 4) + h, c * (w + 4):c * (w + 4) + w] = grab_frame(
                    srcs[k], fps, m, settings, w, h, cancel=cancel)
        write_png(img, path)

    def diagnostic(self, out_path: str, start: float, end: float, every: float = 2.0, *, cancel=None, progress=None):
        plan = diagnostic_plan(start, end, self.output_fps(), self.usable_map(), every)
        settings = RenderSettings.preview(labels=True)
        return plan, self.export(plan, out_path, settings, cancel=cancel, progress=progress)
