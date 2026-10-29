"""Rule-based cut planning on the song timeline.

A plan is a list of segments in OUTPUT FRAME units: output frame j shows song
time ``t0 + j / fps``. Working in whole frames keeps every cut exact and makes
the plan trivially checkable (contiguous, in bounds, no gaps).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy import signal

# (min, target, max) segment length in seconds
PACING = {"natural": (1.8, 2.6, 3.6), "fast": (0.9, 1.3, 1.9)}
EPS = 1e-6


@dataclass
class Segment:
    start: int   # first output frame (inclusive)
    end: int     # last output frame (exclusive)
    take: int    # index into the project's take list


@dataclass
class CutPlan:
    t0: float                 # song time of output frame 0
    fps: Fraction
    segments: list[Segment]
    warnings: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return self.segments[-1].end if self.segments else 0

    @property
    def duration(self) -> float:
        return self.n_frames / float(self.fps)

    def frame_time(self, j: int) -> float:
        return self.t0 + j / float(self.fps)

    def frame_at(self, song_time: float) -> int:
        return int(round((song_time - self.t0) * float(self.fps)))

    def cut_frames(self) -> list[int]:
        return [s.start for s in self.segments[1:]]

    def to_dict(self) -> dict:
        return {"t0": self.t0, "fps": f"{self.fps.numerator}/{self.fps.denominator}",
                "segments": [[s.start, s.end, s.take] for s in self.segments]}

    @classmethod
    def from_dict(cls, d: dict) -> "CutPlan":
        return cls(t0=float(d["t0"]), fps=Fraction(d["fps"]),
                   segments=[Segment(int(a), int(b), int(k)) for a, b, k in d["segments"]])


def _covers(iv, t):
    return iv[0] - EPS <= t <= iv[1] + EPS


def detect_onsets(x: np.ndarray | None, sr: int, start: float, end: float):
    """Onsets of the reference audio between song times start..end.
    Returns ([(song_time, strength 0..1)], confident)."""
    if x is None:
        return [], False
    a, b = max(0, int(start * sr)), min(len(x), int(end * sr))
    seg = x[a:b].astype(np.float64)
    if len(seg) < sr:
        return [], False
    hop = sr // 100
    _, _, z = signal.stft(seg, fs=sr, nperseg=1024, noverlap=1024 - hop, boundary=None, padded=False)
    mag = np.log1p(100 * np.abs(z))
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    if flux.max() <= 0:
        return [], False
    base = signal.medfilt(flux, 51)
    nov = np.maximum(0.0, flux - base)
    scale = np.percentile(nov, 99.5) or nov.max()
    nov = np.clip(nov / scale, 0, 1)
    peaks, props = signal.find_peaks(nov, height=0.2, distance=12)
    times = a / sr + (peaks + 1) * hop / sr + 512 / sr
    strengths = props["peak_heights"]
    strong = int(np.sum(strengths >= 0.4))
    confident = strong >= max(2, (end - start) / 3.0)
    return list(zip(times.tolist(), strengths.tolist())), confident


def plan_cuts(t0: float, t1: float, fps: Fraction, usable: dict[int, tuple[float, float]],
              onsets=None, onsets_confident: bool = False, pacing: str = "natural",
              seed: int = 1) -> CutPlan:
    """Plan cuts for song times [t0, t1). ``usable`` maps take index -> song-time
    interval where that take has valid, matching footage."""
    f = float(fps)
    warnings: list[str] = []
    if not usable:
        raise ValueError("No usable takes.")
    n = int(round((t1 - t0) * f))
    if n <= 0:
        raise ValueError("Excerpt is empty.")

    def ft(j):
        return t0 + j / f

    def covers(k, j):
        return _covers(usable[k], ft(j))

    # trim to the contiguous range where at least one take has footage
    first = next((j for j in range(n) if any(covers(k, j) for k in usable)), None)
    if first is None:
        raise ValueError("No take has footage inside the chosen excerpt.")
    if first > 0:
        warnings.append(f"No take covers the first {first / f:.2f} s of the excerpt; output starts later.")
        t0, n = ft(first), n - first
    last = next((j for j in range(n) if not any(covers(k, j) for k in usable)), n)
    if last < n:
        warnings.append(f"No take covers song time {ft(last):.2f} s onwards; output shortened by {(n - last) / f:.2f} s.")
        n = last

    takes = sorted(usable)
    if len(takes) == 1:
        warnings.append("Only one usable take: exporting a straight crop without cuts.")
        return CutPlan(t0, fps, [Segment(0, n, takes[0])], warnings)

    def reach(k, j):
        """first frame >= j that take k does NOT cover (k covers j)."""
        b = usable[k][1]
        return min(n, int(math.floor((b - t0) * f + EPS)) + 1)

    mn, tg, mx = (int(round(v * f)) for v in PACING.get(pacing, PACING["natural"]))
    onset_frames = sorted((int(round((t - t0) * f)), s) for t, s in (onsets or []))
    rng = random.Random(seed)
    last_used = {k: None for k in takes}
    segs: list[Segment] = []
    j, prev = 0, None
    while j < n:
        if n - j <= mx:
            c = n
        else:
            lo, hi = j + mn, max(j + mn, min(j + mx, n - mn))
            c = j + tg
            if onsets_confident:
                best, best_score = None, 0.15
                width = max(1.0, 0.35 * (mx - mn))
                for fr, s in onset_frames:
                    if lo <= fr <= hi:
                        score = s * math.exp(-0.5 * ((fr - (j + tg)) / width) ** 2)
                        if score > best_score:
                            best, best_score = fr, score
                if best is not None:
                    c = best
            c = min(max(c, lo), hi)
        cands = [k for k in takes if covers(k, j) and covers(k, c - 1)]
        if not cands:
            here = [k for k in takes if covers(k, j)]
            c = max(reach(k, j) for k in here)
            cands = [k for k in here if reach(k, j) >= c]
        pool = [k for k in cands if k != prev] or cands
        weights = [10.0 if last_used[k] is None else 1.0 + (j - last_used[k]) / f for k in pool]
        choice = rng.choices(pool, weights=weights)[0]
        if segs and choice == segs[-1].take:
            segs[-1].end = c
        else:
            segs.append(Segment(j, c, choice))
        last_used[choice], prev, j = c, choice, c
    return CutPlan(t0, fps, segs, warnings)


def diagnostic_plan(t0: float, t1: float, fps: Fraction, usable: dict[int, tuple[float, float]],
                    every: float = 2.0) -> CutPlan:
    """Round-robin through the takes every ``every`` seconds (for sync review)."""
    f = float(fps)
    n = int(round((t1 - t0) * f))
    step = max(1, int(round(every * f)))
    takes = sorted(usable)
    segs, i = [], 0
    for j in range(0, n, step):
        c = min(n, j + step)
        ok = [k for k in takes if _covers(usable[k], t0 + j / f) and _covers(usable[k], t0 + (c - 1) / f)]
        if not ok:
            break
        k = ok[i % len(ok)]
        i += 1
        if segs and segs[-1].take == k:
            segs[-1].end = c
        else:
            segs.append(Segment(j, c, k))
    if not segs:
        raise ValueError("No take covers the start of the diagnostic range.")
    return CutPlan(t0, fps, segs)


def validate_plan(plan: CutPlan, usable: dict[int, tuple[float, float]]) -> list[str]:
    """Problems that would make the plan unsafe to render (empty list = OK)."""
    errs = []
    if not plan.segments:
        return ["The cut list is empty."]
    if plan.segments[0].start != 0:
        errs.append("The first segment must start at the beginning.")
    for i, s in enumerate(plan.segments):
        name = f"Segment {i + 1}"
        if s.end <= s.start:
            errs.append(f"{name} has no length.")
            continue
        if i and s.start != plan.segments[i - 1].end:
            errs.append(f"{name} does not continue from the previous one (gap or overlap).")
        if s.take not in usable:
            errs.append(f"{name} uses a take that is not usable.")
            continue
        a, b = plan.frame_time(s.start), plan.frame_time(s.end - 1)
        iv = usable[s.take]
        if not (_covers(iv, a) and _covers(iv, b)):
            errs.append(f"{name} ({a:.3f}-{b:.3f} s) is outside take {s.take + 1}'s usable range "
                        f"({iv[0]:.3f}-{iv[1]:.3f} s).")
    return errs
