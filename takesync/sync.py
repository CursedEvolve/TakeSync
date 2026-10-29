"""Audio alignment of takes against a reference.

Timing convention
-----------------
song time   = position on the reference timeline, in seconds. With a reference
              audio file it is the time since that file's first sample; with a
              reference take it is that take's media time (so its offset is 0).
media time  = a file's own clock (see media.py).
offset      = song_time - media_time for a take's audio. A take whose recording
              started 2 s before the song position 0 has offset -2.0; a take that
              starts 30 s into the song has offset +30.0.

Method
------
1. Mono 16 kHz analysis audio (never used for output).
2. Coarse: GCC-PHAT cross-correlation of the full signals at 2 kHz -> a handful
   of candidate offsets (repeated phrases give several strong peaks).
3. Validation: every candidate is checked in up to 24 windows spread over the
   whole overlap at 16 kHz with normalised cross-correlation. A window only
   counts if its own best lag lands within 1 ms of the candidate. The candidate
   with the best mean score wins; a close runner-up means "ambiguous".
4. Refinement: median of the agreeing windows' sub-sample lags. Drift across
   the overlap is measured from the same windows.
5. Coverage: 1 s windows at the final offset mark where the take's audio really
   matches the song (excludes pre-roll, stopped playback, a different song).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.fft import irfft, next_fast_len, rfft

from .media import Cancelled, MediaInfo, decode_audio

ANALYSIS_SR = 16000
COARSE_SR = 2000

# Decision thresholds on 300 Hz high-passed audio (tuned on synthetic AAC
# material, see tests): a clean digital copy scores ~0.97-0.99, the same copy
# with -30 dB noise at 64 kbps ~0.84, a different song with the same kick
# and tempo ~0.37.
FAIL_SCORE = 0.5         # below: does not match the reference
LOW_SCORE = 0.7          # below: low confidence, user must confirm
LOW_AGREEMENT = 0.6
AMBIGUOUS_RATIO = 0.5    # best residual / runner-up residual above this: ambiguous
MATCH_NCC = 0.5          # coverage window counts as matching
HIGHPASS_HZ = 300        # analysis ignores sub-bass/kick, which repeats in every song


@dataclass
class AlignResult:
    offset: float | None                 # seconds, song_time = media_time + offset
    score: float = 0.0                   # 0..1 mean agreeing window correlation
    agreement: float = 0.0               # fraction of usable windows that agree
    windows: int = 0                     # usable (non-silent) windows
    ambiguity: float = 0.0               # runner-up score / best score
    alternatives: list[tuple[float, float]] = field(default_factory=list)  # (offset, score)
    drift_ms: float = 0.0                # change of the lag across the overlap
    status: str = "failed"               # ok | low | ambiguous | failed | reference | manual
    matched: tuple[float, float] | None = None   # song-time span where audio matches
    gaps: list[tuple[float, float]] = field(default_factory=list)   # short mismatches inside
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.offset is not None and self.status in ("ok", "reference", "manual")


def load_analysis_audio(info: MediaInfo, track: int = 0, cancel: threading.Event | None = None) -> np.ndarray | None:
    """Mono 16 kHz, high-passed at HIGHPASS_HZ; sample k = media time k / 16000."""
    if not info.audio:
        return None
    a = info.audio[min(track, len(info.audio) - 1)]
    return prepare(decode_audio(info.path, a.index, ANALYSIS_SR, 1, cancel=cancel))


def prepare(x: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    sos = signal.butter(4, HIGHPASS_HZ, "highpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


# --------------------------------------------------------------------------- helpers

def _frame_rms(x: np.ndarray, sr: int, hop_s: float = 0.1) -> np.ndarray:
    hop = max(1, int(hop_s * sr))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(x[: n * hop].astype(np.float64).reshape(n, hop) ** 2, axis=1))


def quiet_threshold(x: np.ndarray, sr: int = ANALYSIS_SR) -> float:
    """RMS level treated as silence: 35 dB below the loud parts, floor -80 dBFS."""
    r = _frame_rms(x, sr)
    if len(r) == 0:
        return 1e-4
    loud = np.percentile(r, 90)
    return max(1e-4, loud * 10 ** (-35 / 20))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def _local_ncc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normalised correlation of x (len w) against every position of y (len w+2s)."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    w = len(x)
    c = signal.correlate(y, x, mode="valid", method="fft")
    ey = np.concatenate([[0.0], np.cumsum(y * y)])
    e = ey[w:] - ey[:-w]
    ex = float(np.dot(x, x))
    return c / np.sqrt(np.maximum(e * ex, 1e-20))


def _parabolic(v: np.ndarray, k: int) -> float:
    if 0 < k < len(v) - 1:
        a, b, c = v[k - 1], v[k], v[k + 1]
        den = a - 2 * b + c
        if den < 0:
            return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return 0.0


def _overlap(n_ref: int, n_take: int, lag: int) -> tuple[int, int]:
    """Take-sample range overlapping the reference for ref_index = take_index + lag."""
    return max(0, -lag), min(n_take, n_ref - lag)


# --------------------------------------------------------------------------- coarse

def coarse_candidates(ref: np.ndarray, take: np.ndarray, sr: int = ANALYSIS_SR,
                      max_candidates: int = 6, min_overlap_s: float = 2.0) -> list[tuple[int, float]]:
    """Candidate lags (in samples at ``sr``) from GCC-PHAT at COARSE_SR."""
    q = sr // COARSE_SR
    r = signal.resample_poly(ref.astype(np.float64), 1, q)
    t = signal.resample_poly(take.astype(np.float64), 1, q)
    r -= r.mean()
    t -= t.mean()
    n = next_fast_len(len(r) + len(t))
    g = rfft(r, n) * np.conj(rfft(t, n))
    g /= np.abs(g) + 1e-12 * np.max(np.abs(g))
    cc = irfft(g, n)
    lags = np.arange(n)
    lags[lags > n // 2] -= n
    lo, hi = np.maximum(0, -lags), np.minimum(len(t), len(r) - lags)
    ov = (hi - lo).astype(np.float64)
    cc[ov < min_overlap_s * COARSE_SR] = -np.inf
    # rank by raw peak (favours long overlaps) and by overlap-normalised peak
    # (finds short takes); validation decides between them
    picked: list[tuple[int, float]] = []
    sep = int(0.25 * COARSE_SR)
    for ranking in (cc, cc / np.sqrt(np.maximum(ov, 1.0))):
        order = np.argsort(ranking)[::-1]
        added = 0
        for idx in order[:50000]:
            if not np.isfinite(ranking[idx]):
                break
            lag = int(lags[idx])
            if all(abs(lag - p) > sep for p, _ in picked):
                picked.append((lag, float(cc[idx])))
                added += 1
                if added >= max_candidates:
                    break
    # windowed search: a few 4 s pieces of the take, each matched against the
    # whole reference. Catches short partial overlaps that the full-length
    # correlation buries under repeated loops.
    w = 4 * COARSE_SR
    active = _frame_rms(t, COARSE_SR, 0.5) > quiet_threshold(t, COARSE_SR)
    if len(t) > w:
        starts = np.linspace(0, len(t) - w, min(10, max(1, len(t) // w))).astype(int)
        nw = next_fast_len(len(r) + w)
        rf = rfft(r, nw)
        for p0 in starts:
            seg = active[p0 // COARSE_SR * 2:(p0 + w) // COARSE_SR * 2]
            if len(seg) and seg.mean() < 0.5:
                continue
            g = rf * np.conj(rfft(t[p0:p0 + w], nw))
            g /= np.abs(g) + 1e-12 * np.max(np.abs(g))
            c = irfft(g, nw)[: len(r) - w + 1]
            for _ in range(3):
                k = int(np.argmax(c))
                if not np.isfinite(c[k]):
                    break
                lag = k - int(p0)
                if all(abs(lag - p) > sep for p, _ in picked):
                    picked.append((lag, float(c[k])))
                c[max(0, k - sep):k + sep] = -np.inf
    return [(lag * q, v) for lag, v in picked]


# --------------------------------------------------------------------------- validation

def _ref_window(ref: np.ndarray, start: int, length: int) -> np.ndarray:
    """ref[start:start+length] with zeros outside the reference."""
    out = np.zeros(length, dtype=np.float64)
    a, b = max(0, start), min(len(ref), start + length)
    if b > a:
        out[a - start:b - start] = ref[a:b]
    return out


def _window_scan(ref, take, lag, sr, win_s=2.0, max_windows=24, search_s=0.025,
                 q_ref=None, q_take=None):
    """Correlation curves of up to ``max_windows`` take windows against the
    reference around ``lag``. Returns list of (take_pos, ncc_curve, search)."""
    s = int(search_s * sr)
    a, b = _overlap(len(ref), len(take), lag)
    # short overlaps get shorter windows so several independent checks still fit
    w = int(min(win_s * sr, max(0.5 * sr, (b - a) / 5)))
    if b - a < w:
        return []
    count = max(1, min(max_windows, (b - a) // w))
    starts = np.linspace(a, b - w, count).astype(int)
    q_ref = quiet_threshold(ref, sr) if q_ref is None else q_ref
    q_take = quiet_threshold(take, sr) if q_take is None else q_take
    out = []
    for p in starts:
        x = take[p:p + w]
        if _rms(x) < q_take or _rms(ref[p + lag:p + lag + w]) < q_ref:
            continue
        out.append((int(p), _local_ncc(x, _ref_window(ref, p + lag - s, w + 2 * s)), s))
    return out


def _score(scan, lag_shift: float, sr: int, tol_s: float = 0.001):
    """Score the candidate ``lag_shift`` samples away from the scan centre.
    Returns (score, agreement, [(pos, refined_shift, ncc)]) where a window
    agrees when its correlation peak lies within tol of the candidate."""
    if not scan:
        return 0.0, 0.0, []
    tol = max(1, int(round(tol_s * sr)))
    total, agree = 0.0, []
    for p, v, s in scan:
        c = s + int(round(lag_shift))
        lo, hi = max(0, c - tol), min(len(v), c + tol + 1)
        if hi <= lo:
            continue
        k = lo + int(np.argmax(v[lo:hi]))
        peak = float(v[k])
        if k == int(np.argmax(v)) or peak >= float(np.max(v)) - 0.02:
            agree.append((p, (k - s) + _parabolic(v, k), peak))
            total += max(0.0, peak)
    return total / len(scan), len(agree) / len(scan), agree


def align(ref: np.ndarray, take: np.ndarray, sr: int = ANALYSIS_SR, *, fps: float = 60.0,
          cancel: threading.Event | None = None) -> AlignResult:
    """Find the offset of ``take`` against ``ref`` (both mono at ``sr``)."""
    res = AlignResult(offset=None)
    if ref is None or take is None or len(take) < sr or len(ref) < sr:
        res.warnings.append("Audio missing or shorter than 1 s.")
        return res
    q_ref, q_take = quiet_threshold(ref, sr), quiet_threshold(take, sr)
    if _rms(take) < 1e-4:
        res.warnings.append("Take audio is silent.")
        return res
    # quick screen of every candidate with few windows, full check of the best
    quick = []
    for lag, _ in coarse_candidates(ref, take, sr):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        scan = _window_scan(ref, take, lag, sr, max_windows=6, q_ref=q_ref, q_take=q_take)
        quick.append((_score(scan, 0, sr, tol_s=0.004)[0], lag))
    quick.sort(reverse=True)
    evaluated = []
    for _, lag in quick[:6]:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        scan = _window_scan(ref, take, lag, sr, q_ref=q_ref, q_take=q_take)
        # re-centre on the windows' own peaks (coarse lag is only ~1 ms accurate)
        _, _, agree = _score(scan, 0, sr, tol_s=0.004)
        shift = float(np.median([d for _, d, _ in agree])) if agree else 0.0
        sc, ag, agree = _score(scan, shift, sr)
        evaluated.append((sc, ag, lag + shift, scan, agree, lag))
    evaluated = [e for e in evaluated if e[3]]
    if not evaluated:
        res.warnings.append("Not enough non-silent overlapping audio with the reference.")
        return res
    evaluated.sort(key=lambda e: e[0], reverse=True)
    sc, ag, lag, scan, agree, centre = evaluated[0]
    res.windows = len(scan)
    res.score, res.agreement = sc, ag
    others = [e for e in evaluated[1:] if abs(e[2] - lag) > 0.05 * sr]
    res.alternatives = [(e[2] / sr, e[0]) for e in others[:3]]
    if others:
        # compare unexplained energy: a true digital copy leaves ~1% residual,
        # a repeated-but-different bar typically 5-10x more
        r_best, r_next = max(1 - sc, 0.003), max(1 - others[0][0], 0.003)
        res.ambiguity = min(1.0, r_best / r_next)
    if not agree or res.windows < 2:
        res.warnings.append("Not enough non-silent overlapping audio to verify the offset.")
        return res
    lags = np.array([centre + d for _, d, _ in agree])
    pos = np.array([p for p, _, _ in agree])
    res.offset = float(np.median(lags)) / sr
    if len(agree) >= 3 and np.ptp(pos) > 0:
        slope = np.polyfit(pos, lags, 1)[0]
        res.drift_ms = float(slope * np.ptp(pos) / sr * 1000)
    if sc < FAIL_SCORE:
        res.status = "failed"
        res.warnings.append(f"Audio does not match the reference (score {sc:.2f}); wrong song, missing playback "
                            f"audio or heavy noise? Best guess was {res.offset:+.3f} s - set it manually if correct.")
        res.alternatives.insert(0, (res.offset, sc))
        res.offset = None
        return res
    if res.ambiguity > AMBIGUOUS_RATIO:
        res.status = "ambiguous"
        alts = ", ".join(f"{o:+.3f}s (score {s:.2f})" for o, s in res.alternatives[:2])
        res.warnings.append(f"Ambiguous: other offsets match almost as well: {alts}. "
                            "Probably a repeated section - confirm or set the offset manually.")
    elif sc < LOW_SCORE or ag < LOW_AGREEMENT:
        res.status = "low"
        res.warnings.append(f"Low confidence (score {sc:.2f}, {ag:.0%} of windows agree). Check this take.")
    else:
        res.status = "ok"
    half_frame_ms = 500.0 / fps
    if abs(res.drift_ms) > half_frame_ms:
        res.warnings.append(f"Clock drift of {res.drift_ms:+.1f} ms across the take; sync will be off by up to "
                            f"{abs(res.drift_ms) / 2:.1f} ms at the ends.")
    coverage(ref, take, res, sr, q_ref=q_ref, q_take=q_take)
    return res


def evaluate_offset(ref: np.ndarray | None, take: np.ndarray | None, offset: float,
                    sr: int = ANALYSIS_SR, *, status: str = "manual") -> AlignResult:
    """Score a user-supplied offset and compute its coverage."""
    res = AlignResult(offset=offset, status=status)
    if ref is None or take is None:
        res.warnings.append("No audio to verify this offset; coverage = whole video.")
        return res
    lag = int(round(offset * sr))
    scan = _window_scan(ref, take, lag, sr)
    res.windows = len(scan)
    res.score, res.agreement, _ = _score(scan, offset * sr - lag, sr)
    if res.score < LOW_SCORE:
        res.warnings.append(f"Audio agrees poorly at this offset (score {res.score:.2f}).")
    coverage(ref, take, res, sr)
    return res


# --------------------------------------------------------------------------- coverage

def _classify(ref, take, lag, sr, win, hop, a, b, q_ref, q_take):
    """Classify take windows [p, p+win) for p in [a, b): 1 match, 0 both quiet, -1 mismatch."""
    s = int(0.002 * sr)
    res = []
    for p in range(a, max(a, b - win) + 1, hop):
        x = take[p:p + win]
        if len(x) < win:
            break
        y = _ref_window(ref, p + lag - s, win + 2 * s)
        qx = _rms(x) < q_take
        qy = _rms(y[s:-s]) < q_ref
        if qx and qy:
            res.append((p, 0))
        elif qx or qy:
            res.append((p, -1))
        else:
            res.append((p, 1 if np.max(_local_ncc(x, y)) >= MATCH_NCC else -1))
    return res


def coverage(ref, take, res: AlignResult, sr: int = ANALYSIS_SR, q_ref=None, q_take=None,
             max_gap_s: float = 1.0) -> None:
    """Fill res.matched / res.gaps: the song-time span where this take's audio follows the song."""
    q_ref = quiet_threshold(ref, sr) if q_ref is None else q_ref
    q_take = quiet_threshold(take, sr) if q_take is None else q_take
    lag = int(round(res.offset * sr))
    a, b = 0, len(take)  # windows outside the reference count as mismatch
    win, hop = sr, sr // 2
    cls = _classify(ref, take, lag, sr, win, hop, a, b, q_ref, q_take)
    if not any(c == 1 for _, c in cls):
        res.matched = None
        res.warnings.append("No part of this take matches the reference at this offset.")
        return
    # runs without long mismatches; pick the one with most matching windows
    runs, cur, pend = [], [], []
    for p, c in cls:
        if c == -1:
            pend.append((p, c))
            if cur and len(pend) * hop > max_gap_s * sr:
                runs.append(cur)
                cur = []
            continue
        if cur:
            cur.extend(pend)
        pend = []
        cur.append((p, c))
    if cur:
        runs.append(cur)
    best = max(runs, key=lambda r: sum(1 for _, c in r if c == 1))
    matches = [p for p, c in best if c == 1]
    start, end = matches[0], matches[-1] + win
    gaps = []
    for p, c in best:
        if matches[0] < p < matches[-1] and c == -1:
            gaps.append(p)
    # refine edges with short windows
    fw, fh = sr // 10, sr // 20
    fine = _classify(ref, take, lag, sr, fw, fh, max(0, start - win), start + win, q_ref, q_take)
    m = [p for p, c in fine if c == 1]
    if m and m[0] <= start + win:
        start = m[0] if m[0] == 0 else m[0] + fw // 2
    fine = _classify(ref, take, lag, sr, fw, fh, max(0, end - 2 * win), min(len(take), end + win), q_ref, q_take)
    m = [p for p, c in fine if c == 1]
    if m and m[-1] + fw >= end - win:
        end = m[-1] + fw if m[-1] + fw >= len(take) - fh else m[-1] + fw // 2
    res.matched = ((start + lag) / sr, (end + lag) / sr)
    if gaps:
        merged = []
        for p in gaps:
            if merged and p - merged[-1][1] <= hop:
                merged[-1][1] = p + win
            else:
                merged.append([p, p + win])
        res.gaps = [((x + lag) / sr, (y + lag) / sr) for x, y in merged]
        res.warnings.append(f"{len(res.gaps)} short audio mismatch(es) inside the take (noise or a glitch?).")


def active_region(sig: np.ndarray | None, sr: int = ANALYSIS_SR) -> tuple[float, float] | None:
    """Media-time span where the reference take's playback audio is audible."""
    if sig is None:
        return None
    res = AlignResult(offset=0.0, status="reference")
    coverage(sig, sig, res, sr)
    return res.matched
