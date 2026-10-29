import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from takesync.media import find_tool  # noqa: E402
from tests import synth  # noqa: E402

# name: (offset = song time at media 0, duration, take_id, extra make_take kwargs)
TAKES = {
    "ref": (0.0, 30.0, 0, {}),
    "pos": (7.3456, 20.0, 1, {}),
    "neg": (-2.5, 25.0, 2, {}),
    "late_audio": (3.21, 15.0, 3, {"audio_start": 0.137}),
    "sr44": (1.234, 20.0, 4, {"audio_sr": 44100}),
    "noisy": (5.5, 18.0, 5, {"noise_db": -30, "audio_bitrate": "64k"}),
    "very_noisy": (4.0, 16.0, 6, {"noise_db": -18, "audio_bitrate": "48k"}),
    "short": (12.0, 7.5, 7, {}),
    "chorus2_plus": (24.0, 14.0, 8, {}),
    "inside_repeat": (12.5, 3.0, 9, {}),
    "no_audio": (2.0, 12.0, 10, {"audio": False}),
    "quiet": (3.0, 15.0, 11, {"gain": 0.05}),
    "fps30": (2.0, 12.0, 12, {"fps": 30}),
    "ntsc": (2.0, 10.0, 13, {"fps": "60000/1001"}),
    "cam_delay": (6.0, 14.0, 14, {"camera_delay": 0.120}),
}


@pytest.fixture(scope="session")
def song():
    return synth.make_song()


@pytest.fixture(scope="session")
def media(tmp_path_factory, song):
    """Generate every synthetic take once per test session."""
    d = os.environ.get("TAKESYNC_TEST_CACHE") or str(tmp_path_factory.mktemp("media"))
    os.makedirs(d, exist_ok=True)
    paths = {}
    for name, (off, dur, tid, kw) in TAKES.items():
        p = os.path.join(d, f"{name}.mp4")
        if not os.path.exists(p):
            synth.make_take(p + ".tmp.mp4", song, offset=off, duration=dur, take_id=tid, **kw)
            os.replace(p + ".tmp.mp4", p)
        paths[name] = p
    # a different song entirely
    p = os.path.join(d, "wrong_song.mp4")
    if not os.path.exists(p):
        synth.make_take(p, synth.make_song(seed=99), offset=3.0, duration=15.0, take_id=15)
    paths["wrong_song"] = p
    # long recording: 60 s of pre-roll silence, the whole song, 20 s of silence
    p = os.path.join(d, "long.mp4")
    if not os.path.exists(p):
        synth.make_take(p, song, offset=-60.0, duration=120.0, take_id=1, fps=30, size=(320, 180))
    paths["long"] = p
    # the song itself as WAV and MP3 references
    wav = os.path.join(d, "song.wav")
    if not os.path.exists(wav):
        from scipy.io import wavfile
        wavfile.write(wav, synth.SR, (song * 32767).astype(np.int16))
    paths["wav"] = wav
    mp3 = os.path.join(d, "song.mp3")
    if not os.path.exists(mp3):
        subprocess.run([find_tool("ffmpeg"), "-v", "error", "-y", "-i", wav, "-c:a", "libmp3lame", "-b:a", "192k", mp3],
                       check=True)
    paths["mp3"] = mp3
    return paths


def truth(name):
    return TAKES[name][0]
