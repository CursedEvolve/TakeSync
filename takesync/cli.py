"""Command line interface (used for testing and power use; the GUI is takesync.gui).

    python -m takesync inspect  take1.mp4 take2.mp4
    python -m takesync analyze  take*.mp4 [--ref song.wav]
    python -m takesync diagnose take*.mp4 --out diag_folder
    python -m takesync export   take*.mp4 --out final.mp4 [--start 0:12.5 --end 0:45]
    python -m takesync validate final.mp4 --frames 1234 --fps 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction

from .media import Cancelled, MediaError, probe
from .planner import CutPlan
from .project import Project
from .render import RenderSettings, fmt_time
from .validate import passed, report, validate_export


def parse_time(s: str) -> float:
    """'83.5', '1:23.5' or '0:01:23.500' -> seconds."""
    s = s.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-")
    total = 0.0
    for part in s.split(":"):
        total = total * 60 + float(part)
    return sign * total


def _progress(msg, i=None, n=None):
    print(f"  {msg}" + (f" ({i + 1}/{n})" if n else ""), file=sys.stderr)


def _frames_progress():
    last = [-1]

    def cb(done, total):
        pct = int(100 * done / max(1, total))
        if pct != last[0] and pct % 10 == 0:
            last[0] = pct
            print(f"  rendering {pct}%", file=sys.stderr)
    return cb


def _load(args) -> Project:
    p = Project(camera_delay=getattr(args, "camera_delay_ms", 0.0) / 1000.0, audio_track=args.audio_track)
    p.load(args.files, args.ref, progress=_progress)
    if args.ref_take is not None:
        if not 1 <= args.ref_take <= len(p.takes):
            raise SystemExit(f"--ref-take must be 1..{len(p.takes)}")
        p.ref_take = args.ref_take - 1
    p.analyze(progress=_progress)
    for spec in args.offset or []:
        k, _, ms = spec.partition("=")
        p.set_manual_offset(int(k) - 1, float(ms) / 1000.0)
    for k in args.accept or []:
        p.accept(k - 1, True)
    for k in args.exclude or []:
        p.accept(k - 1, False)
    return p


def _excerpt(p: Project, args) -> tuple[float, float]:
    auto = p.auto_excerpt()
    start = parse_time(args.start) if args.start else (auto[0] if auto else None)
    end = parse_time(args.end) if args.end else (auto[1] if auto else None)
    if start is None or end is None:
        span = p.union_span()
        if not span:
            raise SystemExit("No usable takes - see the report above.")
        print("  No region is shared by all takes; using the union. Use --start/--end to choose.", file=sys.stderr)
        start = span[0] if start is None else start
        end = span[1] if end is None else end
    if end <= start:
        raise SystemExit("Excerpt end must be after start.")
    return start, end


def cmd_inspect(args):
    for f in args.files:
        info = probe(f)
        print(info.summary())
        print(f"  container start {info.format_start:.6f}s  video start {info.video_start:.6f}s  "
              f"r_fps {info.r_fps} avg_fps {info.avg_fps} frames {info.nb_frames} vfr_suspect {info.vfr_suspect}")
        for a in info.audio:
            print(f"  audio stream #{a.index}: {a.codec} {a.sample_rate} Hz {a.channels}ch start {a.start:.6f}s "
                  f"duration {a.duration}")
        if info.color:
            print(f"  colour: {info.color}")
        for w in info.warnings:
            print(f"  WARNING: {w}")


def cmd_analyze(args):
    p = _load(args)
    print(p.alignment_report())
    if args.json:
        data = {"reference": p.reference_name, "takes": [
            {"file": t.info.path, "offset": t.offset, "status": t.result.status if t.result else None,
             "score": t.result.score if t.result else None, "usable": p.usable(i), "enabled": t.enabled,
             "warnings": (t.result.warnings if t.result else [])} for i, t in enumerate(p.takes)]}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    return 0 if all(t.enabled for t in p.takes) else 2


def cmd_diagnose(args):
    p = _load(args)
    os.makedirs(args.out, exist_ok=True)
    rep = p.alignment_report()
    print(rep)
    with open(os.path.join(args.out, "alignment_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep + "\n")
    start, end = _excerpt(p, args)
    end = min(end, start + args.length)
    times = [start + (end - start) * (i + 0.5) / args.rows for i in range(args.rows)]
    sheet = os.path.join(args.out, "contact_sheet.png")
    p.contact_sheet(sheet, times)
    print(f"Contact sheet: {sheet}  (rows = song times {', '.join(fmt_time(t) for t in times)}; columns = takes)")
    out = os.path.join(args.out, "diagnostic.mp4")
    plan, checks = p.diagnostic(out, start, end, args.every, progress=_frames_progress())
    print(f"Diagnostic video: {out}  ({plan.duration:.2f} s, switching take every {args.every} s)")
    print(report(checks))
    return 0 if passed(checks) else 1


def cmd_export(args):
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists. Use --overwrite to replace it.")
    p = _load(args)
    print(p.alignment_report())
    bad = [i + 1 for i, t in enumerate(p.takes) if not t.enabled]
    if bad:
        print(f"\nTakes not used: {bad}. Use --accept N to confirm a flagged take or --offset N=ms to set it.",
              file=sys.stderr)
    if args.plan_in:
        with open(args.plan_in, encoding="utf-8") as fh:
            plan = CutPlan.from_dict(json.load(fh))
    else:
        start, end = _excerpt(p, args)
        plan = p.make_plan(start, end, args.pacing, args.seed)
    for w in plan.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"\nCut list ({len(plan.segments)} segments, {plan.duration:.3f} s, starts at song {fmt_time(plan.t0)}):")
    for i, s in enumerate(plan.segments):
        print(f"  {i + 1:3d}  {fmt_time(plan.frame_time(s.start))} - {fmt_time(plan.frame_time(s.end))}  "
              f"take {s.take + 1} ({p.takes[s.take].name})")
    if args.plan_out:
        with open(args.plan_out, "w", encoding="utf-8") as fh:
            json.dump(plan.to_dict(), fh, indent=1)
    settings = RenderSettings.preview() if args.preview else RenderSettings()
    settings.crop_center = args.crop_center
    checks = p.export(plan, args.out, settings, progress=_frames_progress())
    print(f"\nWrote {args.out}")
    print(report(checks))
    return 0 if passed(checks) else 1


def cmd_validate(args):
    checks = validate_export(args.file, n_frames=args.frames, fps=Fraction(args.fps),
                             width=args.width, height=args.height)
    print(report(checks))
    return 0 if passed(checks) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="takesync", description="Sync and auto-cut OBS performance takes.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("files", nargs="+")
        sp.add_argument("--ref", help="reference audio file (WAV/MP3); default: a take")
        sp.add_argument("--ref-take", type=int, help="take number to use as reference (1-based)")
        sp.add_argument("--audio-track", type=int, default=0, help="audio track of the recordings (0 = first)")
        sp.add_argument("--offset", action="append", metavar="N=MS", help="manual offset for take N in ms")
        sp.add_argument("--accept", action="append", type=int, metavar="N", help="use flagged take N anyway")
        sp.add_argument("--exclude", action="append", type=int, metavar="N", help="do not use take N")
        sp.add_argument("--camera-delay-ms", type=float, default=0.0, help="picture lags audio by this much")
        sp.add_argument("--start", help="excerpt start, song time (e.g. 0:12.5)")
        sp.add_argument("--end", help="excerpt end, song time")

    sp = sub.add_parser("inspect", help="show ffprobe metadata")
    sp.add_argument("files", nargs="+")
    sp.set_defaults(fn=cmd_inspect)

    sp = sub.add_parser("analyze", help="align takes and print the report")
    common(sp)
    sp.add_argument("--json")
    sp.set_defaults(fn=cmd_analyze)

    sp = sub.add_parser("diagnose", help="report + contact sheet + alternating diagnostic video")
    common(sp)
    sp.add_argument("--out", required=True, help="output folder")
    sp.add_argument("--every", type=float, default=2.0)
    sp.add_argument("--length", type=float, default=30.0)
    sp.add_argument("--rows", type=int, default=4)
    sp.set_defaults(fn=cmd_diagnose)

    sp = sub.add_parser("export", help="plan cuts and export 1080x1920")
    common(sp)
    sp.add_argument("--out", required=True)
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--pacing", choices=["natural", "fast"], default="natural")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--crop-center", type=float, default=0.5, help="0..1 horizontal crop centre")
    sp.add_argument("--preview", action="store_true", help="fast low-res draft")
    sp.add_argument("--plan-in", help="use this cut list JSON instead of auto planning")
    sp.add_argument("--plan-out", help="save the cut list JSON")
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("validate", help="check an exported file")
    sp.add_argument("file")
    sp.add_argument("--frames", type=int, required=True)
    sp.add_argument("--fps", default="60")
    sp.add_argument("--width", type=int, default=1080)
    sp.add_argument("--height", type=int, default=1920)
    sp.set_defaults(fn=cmd_validate)

    args = ap.parse_args(argv)
    try:
        return args.fn(args) or 0
    except (MediaError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Cancelled:
        print("Cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
