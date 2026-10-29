"""ffprobe inspection, timestamps and timebase handling."""
import os
import subprocess
import tempfile
from fractions import Fraction

import numpy as np
import pytest

from takesync.media import decode_audio, find_tool, probe
from takesync.render import RenderSettings, RenderSource, crop_rect, frame_chain
from tests import synth
from tests.conftest import TAKES


def test_probe_basic(media):
    info = probe(media["ref"])
    assert info.has_video and (info.width, info.height) == (640, 360)
    assert info.fps == Fraction(60) and not info.vfr_suspect
    assert abs(info.video_duration - 30.0) < 0.02 and info.video_start == 0.0
    assert info.audio and info.audio[0].sample_rate == 48000


def test_probe_rates(media):
    assert probe(media["fps30"]).fps == Fraction(30)
    assert probe(media["ntsc"]).fps == Fraction(60000, 1001)
    assert probe(media["sr44"]).audio[0].sample_rate == 44100


def test_probe_no_audio(media):
    info = probe(media["no_audio"])
    assert not info.audio and any("manual" in w for w in info.warnings)


def test_late_audio_start_is_honoured(media, song):
    """Audio that starts 137 ms after the video must land at media time 0.137."""
    info = probe(media["late_audio"])
    assert info.audio[0].start > 0.1
    a = decode_audio(info.path, info.audio[0].index, 48000, 1)
    off = TAKES["late_audio"][0]
    truth = synth.song_slice(song, off, len(a) / 48000).mean(axis=1)
    # padded silence before the stream's first timestamp (0.115 s incl. AAC priming)
    assert np.max(np.abs(a[: int(0.11 * 48000)])) == 0.0
    x = a[48000:3 * 48000]
    s = 960
    c = np.correlate(truth[48000 - s:3 * 48000 + s], x, "valid")
    lag_ms = (int(np.argmax(c)) - s) / 48
    assert abs(lag_ms) <= 1.0, lag_ms   # MP4 edit lists store the delay with 1 ms precision


def test_decode_range_and_padding(media):
    info = probe(media["ref"])
    full = decode_audio(info.path, info.audio[0].index, 16000, 1)
    part = decode_audio(info.path, info.audio[0].index, 16000, 1, start_sample=16000 * 5, num_samples=16000)
    assert np.allclose(part, full[16000 * 5:16000 * 6])
    tail = decode_audio(info.path, info.audio[0].index, 16000, 1, start_sample=len(full) - 100, num_samples=1000)
    assert len(tail) == 1000 and np.all(tail[100:] == 0)


def test_vfr_detection():
    d = tempfile.mkdtemp()
    vfr = os.path.join(d, "vfr.mp4")
    subprocess.run([find_tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=60:d=4",
                    "-vf", "settb=1/1000,setpts='(N/60+0.007*mod(N,2))/TB'", "-fps_mode", "passthrough", "-enc_time_base", "1/1000",
                    "-c:v", "libx264", "-preset", "ultrafast", vfr], check=True)
    assert probe(vfr).vfr_suspect


def test_rotation_metadata():
    d = tempfile.mkdtemp()
    src = os.path.join(d, "land.mp4")
    rot = os.path.join(d, "rot.mp4")
    subprocess.run([find_tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=1",
                    "-c:v", "libx264", "-preset", "ultrafast", src], check=True)
    r = subprocess.run([find_tool("ffmpeg"), "-v", "error", "-y", "-display_rotation", "90", "-i", src,
                        "-c", "copy", rot])
    if r.returncode != 0:
        pytest.skip("this ffmpeg cannot write rotation metadata")
    info = probe(rot)
    assert info.rotation in (90, 270) and (info.width, info.height) == (360, 640)
    # portrait source: the crop keeps the full width and trims height
    cw, ch, x, y = crop_rect(info.width, info.height)
    assert cw == 360 and ch == 640


def test_frame_chain_seek_math():
    src = RenderSource("x.mp4", "x", 0.0, 0.0, 10.0, 3840, 2160)
    seek, chain = frame_chain(src, Fraction(60), 1.2345, RenderSettings())
    assert abs(seek - (1.2345 - 2 / 60)) < 1e-9 and "trim=start_frame=2" in chain
    seek, chain = frame_chain(src, Fraction(60), 0.01, RenderSettings())
    assert seek == 0.01 and "trim=start_frame=0" in chain
    assert "crop=1214:2160:1312:0" in chain and "lanczos" in chain
    seek, chain = frame_chain(src, Fraction(60000, 1001), 1.0, RenderSettings())
    assert "fps=60000/1001" in chain


def test_unicode_and_spaces_in_paths(media, tmp_path):
    from takesync.project import Project
    from takesync.render import RenderSettings as RS
    d = tmp_path / "Mes vidéos  ünï"
    d.mkdir()
    a = d / "take one ✓.mp4"
    b = d / "take two (copy).mp4"
    import shutil
    shutil.copy(media["ref"], a)
    shutil.copy(media["pos"], b)
    p = Project()
    p.load([str(a), str(b)])
    p.analyze()
    assert all(t.enabled for t in p.takes)
    plan = p.make_plan(8.0, 10.0)
    out = d / "export final ✓.mp4"
    from takesync.validate import passed, report
    checks = p.export(plan, str(out), RS.preview())
    assert passed(checks), report(checks)
