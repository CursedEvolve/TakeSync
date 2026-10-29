"""Cut planner rules (pure Python, fast)."""
from fractions import Fraction

import pytest

from takesync.planner import PACING, CutPlan, Segment, plan_cuts, validate_plan

FPS = Fraction(60)


def lengths(plan):
    return [(s.end - s.start) / 60 for s in plan.segments]


def test_contiguous_valid_and_deterministic():
    usable = {0: (0, 60), 1: (0, 60), 2: (0, 60)}
    a = plan_cuts(5.0, 45.0, FPS, usable, seed=7)
    b = plan_cuts(5.0, 45.0, FPS, usable, seed=7)
    assert [vars(s) for s in a.segments] == [vars(s) for s in b.segments]
    assert validate_plan(a, usable) == []
    assert a.n_frames == 40 * 60
    for x, y in zip(a.segments, a.segments[1:]):
        assert x.take != y.take


@pytest.mark.parametrize("pacing", ["natural", "fast"])
def test_pacing_limits(pacing):
    mn, tg, mx = PACING[pacing]
    usable = {0: (0, 100), 1: (0, 100), 2: (0, 100), 3: (0, 100)}
    plan = plan_cuts(0.0, 90.0, FPS, usable, pacing=pacing, seed=3)
    ls = lengths(plan)
    assert min(ls) >= mn - 1e-9 and max(ls) <= mx + 1e-9, ls


def test_fast_cuts_more_than_natural():
    usable = {0: (0, 100), 1: (0, 100), 2: (0, 100)}
    nat = plan_cuts(0, 60, FPS, usable, pacing="natural")
    fast = plan_cuts(0, 60, FPS, usable, pacing="fast")
    assert len(fast.segments) > 1.5 * len(nat.segments)


def test_not_rigid_round_robin():
    usable = {k: (0, 200) for k in range(4)}
    plan = plan_cuts(0, 180, FPS, usable, seed=11)
    seq = [s.take for s in plan.segments]
    rr = all(seq[i] == seq[i % 4] for i in range(len(seq)))
    assert not rr
    assert set(seq) == {0, 1, 2, 3}


def test_prefers_onsets_when_confident():
    usable = {0: (0, 60), 1: (0, 60)}
    onsets = [(t, 1.0) for t in [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]]
    plan = plan_cuts(0, 22, FPS, usable, onsets, True, seed=1)
    for c in plan.cut_frames():
        assert any(abs(c - t * 60) <= 1 for t, _ in onsets), c


def test_takes_ending_early_are_replaced():
    usable = {0: (0, 60), 1: (0, 12.0), 2: (8.0, 60)}
    plan = plan_cuts(0, 30, FPS, usable, seed=2)
    assert validate_plan(plan, usable) == []
    for s in plan.segments:
        a, b = plan.frame_time(s.start), plan.frame_time(s.end - 1)
        assert usable[s.take][0] <= a and b <= usable[s.take][1]


def test_gap_shortens_with_warning():
    usable = {0: (0, 10.0), 1: (0, 10.0)}
    plan = plan_cuts(0, 20, FPS, usable)
    assert plan.n_frames == 601  # frames 0..600 have song time <= 10.0
    assert any("shortened" in w for w in plan.warnings)


def test_late_start_with_warning():
    usable = {0: (3.0, 20.0), 1: (3.0, 20.0)}
    plan = plan_cuts(0, 10, FPS, usable)
    assert abs(plan.t0 - 3.0) < 1 / 60 and any("starts later" in w for w in plan.warnings)


def test_single_take_is_straight_crop():
    plan = plan_cuts(1.0, 9.0, FPS, {3: (0, 10)})
    assert len(plan.segments) == 1 and plan.segments[0].take == 3
    assert any("one usable take" in w for w in plan.warnings)


def test_no_coverage_raises():
    with pytest.raises(ValueError):
        plan_cuts(30, 40, FPS, {0: (0, 10), 1: (0, 10)})


def test_validate_catches_bad_edits():
    usable = {0: (0, 10), 1: (5, 10)}
    plan = CutPlan(0.0, FPS, [Segment(0, 120, 0), Segment(130, 200, 1), Segment(200, 200, 0)])
    errs = validate_plan(plan, usable)
    assert any("gap" in e for e in errs) and any("no length" in e for e in errs)
    plan = CutPlan(0.0, FPS, [Segment(0, 120, 1)])
    assert any("outside" in e for e in validate_plan(plan, usable))


def test_plan_roundtrip_json():
    plan = CutPlan(1.25, Fraction(60000, 1001), [Segment(0, 100, 0), Segment(100, 250, 2)])
    again = CutPlan.from_dict(plan.to_dict())
    assert again.fps == plan.fps and again.t0 == plan.t0 and again.n_frames == 250
