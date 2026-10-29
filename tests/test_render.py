"""End-to-end exports checked frame by frame via the barcodes burned into the
synthetic takes, plus the export validator."""
import os
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np
import pytest

from takesync.media import decode_gray_frames, probe
from takesync.planner import CutPlan, Segment
from takesync.project import Project
from takesync.render import RenderSettings, check_bounds, crop_rect
from takesync.validate import passed, report
from tests import synth
from tests.conftest import TAKES

FAST = dict(preset="ultrafast")


def take_ids(project):
    out = []
    for t in project.takes:
        stem = os.path.splitext(t.name)[0]
        out.append(TAKES[stem][2] if stem in TAKES else None)
    return out


def check_frames(out_path, plan, project, tol_ms=None, src_size=(640, 360), center=0.5):
    """Every output frame must show the planned take at the planned song time."""
    frames = decode_gray_frames(out_path, 108, 192)
    assert len(frames) == plan.n_frames
    cw, ch, x, y = crop_rect(*src_size, center)
    codes = synth.read_barcodes(frames, x, cw, ch, y, *src_size)
    ids = take_ids(project)
    if tol_ms is None:
        tol_ms = 500 / float(plan.fps) + 1.0  # nearest frame + ms rounding of the barcode
    errs = []
    for seg in plan.segments:
        for j in range(seg.start, seg.end):
            tid, ms = codes[j]
            if ids[seg.take] is not None:
                assert tid == ids[seg.take], f"frame {j}: shows take id {tid}, plan says {ids[seg.take]}"
            errs.append(ms - plan.frame_time(j) * 1000)
    errs = np.array(errs)
    assert np.max(np.abs(errs)) <= tol_ms, f"sync error up to {np.max(np.abs(errs)):.2f} ms"
    return errs


def bed_matches_song(bed, song, t0, tol_ms=0.5):
    """The soundtrack must be the song from t0, at lag 0."""
    truth = synth.song_slice(song, t0, len(bed) / synth.SR).mean(axis=1)
    got = bed.mean(axis=1)
    s = 480
    c = np.correlate(np.pad(truth, s)[: len(got) + 2 * s], got, "valid")
    lag = (int(np.argmax(c)) - s) / synth.SR * 1000
    ncc = np.max(c) / np.sqrt(np.dot(truth, truth) * np.dot(got, got))
    assert abs(lag) <= tol_ms and ncc > 0.95, (lag, ncc)


@pytest.fixture(scope="module")
def outdir():
    return tempfile.mkdtemp(prefix="takesync_test_out_")


def test_full_quality_multitake_export(media, song, outdir):
    p = Project()
    p.load([media[n] for n in ["ref", "pos", "neg", "sr44", "late_audio"]])
    p.analyze()
    assert all(t.enabled for t in p.takes), p.alignment_report()
    ex = p.auto_excerpt()
    assert ex is not None
    plan = p.make_plan(ex[0], min(ex[1], ex[0] + 10.0), "fast", seed=5)
    assert len({s.take for s in plan.segments}) >= 3
    out = os.path.join(outdir, "full.mp4")
    checks = p.export(plan, out, RenderSettings())   # the real export settings
    print(report(checks))
    assert passed(checks)
    errs = check_frames(out, plan, p)
    print(f"sync error: max {np.max(np.abs(errs)):.2f} ms, mean {np.mean(np.abs(errs)):.2f} ms")
    info = probe(out)
    assert (info.width, info.height, info.fps) == (1080, 1920, Fraction(60))


def test_reference_wav_bed_is_the_song(media, song, outdir):
    p = Project()
    p.load([media["pos"], media["neg"]], media["wav"])
    p.analyze()
    plan = p.make_plan(8.0, 14.0, "natural", seed=1)
    out = os.path.join(outdir, "wavref.mp4")
    settings = RenderSettings(**FAST)
    from takesync.render import render
    bed = render(plan, p.sources(), p.bed(), settings, out)
    bed_matches_song(bed, song, plan.t0, tol_ms=0.05)
    check_frames(out, plan, p)


def test_cuts_at_take_edges(media, outdir):
    """Segments that start on the first usable frame / end on the last usable frame."""
    p = Project()
    p.load([media["ref"], media["neg"], media["short"]])
    p.analyze()
    fps = p.output_fps()
    u = p.usable_map()
    s0, s1 = u[2]                      # 'short' covers ~12.0-19.5
    t0 = s0 - 1.0
    a = int(round((s0 - t0) * 60)) + 1  # first frame inside 'short'
    b = int((s1 - t0) * 60)             # last frame inside 'short' + 1
    plan = CutPlan(t0, fps, [Segment(0, a, 0), Segment(a, b, 2), Segment(b, b + 60, 1)])
    assert p.check_plan(plan) == []
    assert check_bounds(plan, p.sources()) == []
    out = os.path.join(outdir, "edges.mp4")
    checks = p.export(plan, out, RenderSettings.preview())
    assert passed(checks), report(checks)
    check_frames(out, plan, p)
    # one frame further is rejected before rendering
    bad = CutPlan(t0, fps, [Segment(0, a - 2, 0), Segment(a - 2, b, 2)])
    assert p.check_plan(bad)


def test_cut_at_very_start_of_file(media, outdir):
    """Reference take from media time ~0 (frame 1), cut into a take near its end."""
    p = Project()
    p.load([media["ref"], media["pos"]])
    p.analyze()
    u = p.usable_map()
    t0 = u[0][0]
    plan = p.make_plan(t0, t0 + 4.0)
    assert plan.segments[0].start == 0
    out = os.path.join(outdir, "start.mp4")
    checks = p.export(plan, out, RenderSettings.preview())
    assert passed(checks), report(checks)
    check_frames(out, plan, p)


def test_bounds_guard(media):
    p = Project()
    p.load([media["ref"], media["short"]])
    p.analyze()
    plan = CutPlan(10.0, Fraction(60), [Segment(0, 60, 1)])   # 'short' starts at 12.0
    assert check_bounds(plan, p.sources())
    with pytest.raises(ValueError):
        p.export(plan, "never_written.mp4", RenderSettings.preview())
    assert not os.path.exists("never_written.mp4")


def test_camera_delay(media, outdir):
    p = Project(camera_delay=0.120)
    p.load([media["cam_delay"]], media["wav"])
    p.analyze()
    assert abs(p.takes[0].offset - 6.0) < 0.001
    plan = p.make_plan(7.0, 12.0)
    out = os.path.join(outdir, "camdelay.mp4")
    checks = p.export(plan, out, RenderSettings.preview())
    assert passed(checks), report(checks)
    check_frames(out, plan, p)
    # without the correction the picture is 120 ms late
    p.camera_delay = 0.0
    out2 = os.path.join(outdir, "camdelay_off.mp4")
    p.export(plan, out2, RenderSettings.preview())
    with pytest.raises(AssertionError):
        check_frames(out2, plan, p)


def test_30fps_source(media, outdir):
    p = Project()
    p.load([media["fps30"]], media["wav"])
    p.analyze()
    assert p.output_fps() == 30
    plan = p.make_plan(3.0, 9.0)
    out = os.path.join(outdir, "fps30.mp4")
    checks = p.export(plan, out, RenderSettings.preview())
    assert passed(checks), report(checks)
    check_frames(out, plan, p)


def test_ntsc_rate_source(media, outdir):
    p = Project()
    p.load([media["ntsc"]], media["wav"])
    p.analyze()
    assert p.output_fps() == Fraction(60000, 1001)
    plan = p.make_plan(3.0, 8.0)
    out = os.path.join(outdir, "ntsc.mp4")
    checks = p.export(plan, out, RenderSettings.preview())
    assert passed(checks), report(checks)
    check_frames(out, plan, p)


def test_mixed_frame_rates(media, outdir):
    """60 fps reference with a 30 fps take: 30 fps frames are shown twice, never out of order."""
    p = Project()
    p.load([media["ref"], media["fps30"]])
    p.analyze()
    plan = CutPlan(4.0, Fraction(60), [Segment(0, 120, 0), Segment(120, 300, 1)])
    out = os.path.join(outdir, "mixed.mp4")
    checks = p.export(plan, out, RenderSettings.preview())
    assert passed(checks), report(checks)
    check_frames(out, plan, p, tol_ms=500 / 30 + 1.0)


def test_4k_source_crop_geometry(media, song, outdir):
    src = os.path.join(outdir, "src4k.mp4")
    synth.make_take(src, song, offset=5.0, duration=3.0, take_id=6, size=(3840, 2160))
    p = Project()
    p.load([src], media["wav"])
    p.analyze()
    cw, ch, x, y = crop_rect(3840, 2160)
    assert ch == 2160 and abs(cw / ch - 9 / 16) < 0.002 and x % 2 == 0
    plan = p.make_plan(5.1, 7.8)
    out = os.path.join(outdir, "4k.mp4")
    checks = p.export(plan, out, RenderSettings(**FAST))
    assert passed(checks), report(checks)
    check_frames(out, plan, p, src_size=(3840, 2160))


def test_crop_center_shift(media, outdir):
    """Shifting the crop moves the picture: the barcode decodes only with the matching geometry."""
    cw, ch, x, y = crop_rect(640, 360, 0.47)
    assert x == int(round(0.47 * 640 - cw / 2)) // 2 * 2
    assert crop_rect(640, 360, 0.0)[2] == 0 and crop_rect(640, 360, 1.0)[2] == 640 - cw
    p = Project()
    p.load([media["ref"]])
    p.analyze()
    plan = p.make_plan(2.0, 3.0)
    out = os.path.join(outdir, "shift.mp4")
    p.export(plan, out, RenderSettings.preview(crop_center=0.47))
    check_frames(out, plan, p, center=0.47)


def test_single_input_file(media, outdir):
    p = Project()
    p.load([media["pos"]])
    p.analyze()
    assert p.takes[0].result.status == "reference"
    plan = p.make_plan(*p.auto_excerpt())
    assert len(plan.segments) == 1 and any("one usable take" in w for w in plan.warnings)


def test_no_shared_overlap(media):
    p = Project()
    p.load([media["ref"], media["short"], media["chorus2_plus"]])
    p.analyze()
    assert p.auto_excerpt() is None
    assert p.union_span() is not None


def test_cli_export_and_no_overwrite(media, outdir):
    out = os.path.join(outdir, "cli.mp4")
    cmd = [sys.executable, "-m", "takesync", "export", media["ref"], media["pos"], "--out", out,
           "--start", "0:08", "--end", "0:12", "--preview"]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "EXPORT VALIDATION: PASSED" in r.stdout
    mtime = os.path.getmtime(out)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    assert r.returncode != 0 and "exists" in (r.stdout + r.stderr)
    assert os.path.getmtime(out) == mtime


def test_cli_diagnose(media, outdir):
    out = os.path.join(outdir, "diag")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-m", "takesync", "diagnose", media["ref"], media["pos"], media["neg"],
                        "--out", out, "--length", "8"], capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr
    for f in ("alignment_report.txt", "contact_sheet.png", "diagnostic.mp4"):
        assert os.path.getsize(os.path.join(out, f)) > 0
    assert "EXPORT VALIDATION: PASSED" in r.stdout
    info = probe(os.path.join(out, "contact_sheet.png"))
    assert (info.width, info.height) == (3 * 184, 4 * 324)
