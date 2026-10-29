"""Synchronisation against known synthetic offsets."""
import pytest

from takesync.media import probe
from takesync.project import Project
from takesync.sync import align, load_analysis_audio
from tests.conftest import truth

TOL_MS = 1.0  # internal target; one 60 fps frame is 16.7 ms


@pytest.fixture(scope="module")
def audio(media):
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = load_analysis_audio(probe(media[name]))
        return cache[name]
    return get


@pytest.mark.parametrize("name", ["pos", "neg", "late_audio", "sr44", "noisy", "short", "chorus2_plus",
                                  "quiet", "fps30", "ntsc", "cam_delay"])
def test_known_offsets_against_reference_take(audio, name):
    r = align(audio("ref"), audio(name))
    assert r.status == "ok", (r.status, r.warnings)
    err_ms = (r.offset - truth(name)) * 1000
    assert abs(err_ms) <= TOL_MS, f"{name}: error {err_ms:.3f} ms"


def test_very_noisy_is_flagged_never_silently_wrong(audio):
    # -18 dB noise on 48 kbps AAC: far worse than an OBS desktop capture
    r = align(audio("ref"), audio("very_noisy"))
    print("very noisy:", r.status, round(r.score, 3), r.alternatives[:1])
    assert r.status != "ok" or abs(r.offset - truth("very_noisy")) * 1000 <= TOL_MS
    guesses = ([r.offset] if r.offset is not None else []) + [o for o, _ in r.alternatives[:1]]
    assert any(abs(g - truth("very_noisy")) * 1000 <= TOL_MS for g in guesses)


def test_long_recording_with_preroll(audio):
    r = align(audio("ref"), audio("long"))
    assert r.status == "ok"
    assert abs(r.offset - (-60.0)) * 1000 <= TOL_MS
    # coverage: song plays 0..40 s, reference covers 0..30 s
    a, b = r.matched
    assert abs(a - 0.0) < 0.1 and abs(b - 30.0) < 0.1, r.matched


def test_coverage_excludes_preroll_and_ends(audio):
    r = align(audio("ref"), audio("neg"))
    a, b = r.matched                  # song starts at song time 0, take ends at 22.5
    assert -0.06 <= a <= 0.06 and abs(b - 22.5) < 0.06, r.matched


def test_repeated_phrase_is_flagged_ambiguous(audio):
    # 3 s from inside the first half of chorus 1, which is repeated 4 s later
    r = align(audio("ref"), audio("inside_repeat"))
    assert r.status == "ambiguous", (r.status, r.score, r.ambiguity)
    offsets = [r.offset] + [o for o, _ in r.alternatives]
    assert any(abs(o - truth("inside_repeat")) < 0.001 for o in offsets)


def test_partial_overlap_with_repeat_resolved(audio):
    r = align(audio("ref"), audio("chorus2_plus"))
    assert r.status == "ok" and abs(r.offset - 24.0) < 0.001


def test_wrong_song_fails(audio):
    r = align(audio("ref"), audio("wrong_song"))
    assert r.status == "failed" and r.offset is None, (r.status, r.score)


def test_missing_audio(media):
    p = Project()
    p.load([media["ref"], media["no_audio"]])
    p.analyze()
    t = p.takes[1]
    assert t.offset is None and not t.enabled
    assert any("manual" in w.lower() for w in t.result.warnings)
    p.set_manual_offset(1, 2.0)
    assert p.takes[1].enabled and p.usable(1) is not None
    a, b = p.usable(1)
    assert abs(a - 2.0) < 0.05 and abs(b - 14.0) < 0.05


@pytest.mark.parametrize("ref", ["wav", "mp3"])
def test_reference_audio_file(media, ref):
    p = Project()
    p.load([media["pos"], media["neg"], media["chorus2_plus"]], media[ref])
    p.analyze()
    for t, name in zip(p.takes, ["pos", "neg", "chorus2_plus"]):
        assert t.result.status == "ok", (name, t.result.status, t.result.warnings)
        assert abs(t.offset - truth(name)) * 1000 <= TOL_MS, (ref, name, t.offset)


def test_full_song_reference_makes_exact_repeat_ambiguous(media):
    # with the whole song as reference, a take of only chorus 2 also matches chorus 1
    from tests import synth
    import os
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "chorus2.mp4")
    synth.make_take(p, synth.make_song(), offset=28.0, duration=8.0, take_id=3)
    pr = Project()
    pr.load([p], media["wav"])
    pr.analyze()
    r = pr.takes[0].result
    assert r.status == "ambiguous" and not pr.takes[0].enabled


def test_auto_reference_is_longest(media):
    p = Project()
    p.load([media["short"], media["ref"], media["pos"]])
    assert p.ref_take == 1
