"""Offline tests for E.20's ceiling and deference model and E.21's classes and routing."""

from __future__ import annotations

import pytest

from sysone.analysis.e20 import ceiling_gain, fit_d, predict, spearman


def test_ceiling_is_zero_when_flags_only_hit_right_labels():
    # reviewer disagrees only on items the upstream gets right: flipping can only hurt
    items = [(True, 1.0, "a"), (True, 0.8, "a"), (False, 0.0, "a"), (True, 0.0, "a")]
    assert ceiling_gain(items) == 0.0


def test_ceiling_picks_best_threshold_per_direction():
    # direction a: flagging at 1.0 fixes one mistake, at 0.6 also breaks two right labels
    items = [(False, 1.0, "a"), (True, 0.6, "a"), (True, 0.6, "a"),
             # direction b: flagging at 0.8 fixes one; at 0.4 fixes two but breaks one (also +1)
             (False, 0.4, "b"), (False, 0.8, "b"), (True, 0.4, "b")]
    assert ceiling_gain(items) == pytest.approx((1 + 1) / 6)


def test_fit_d_recovers_parameters():
    d_w, d_r = 0.3, 0.2
    s = [0.0, 0.2, 0.6, 1.0, 0.8]
    right = [(x, predict(x, True, d_w, d_r)) for x in s]
    wrong = [(x, predict(x, False, d_w, d_r)) for x in s]
    got = fit_d(right, wrong)
    assert got == pytest.approx((d_w, d_r))


def test_fit_d_clips_negative_deference_to_zero():
    assert fit_d([(0.5, 0.4)], [(0.5, 0.6)]) == (0.0, 0.0)


def test_spearman():
    assert spearman([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0


sk = pytest.importorskip("sklearn")


def test_review_classes():
    from sysone.analysis.e21 import review_class
    assert review_class(True, True, 0.8) == "gave_in"
    assert review_class(True, False, 0.8) == "held_firm"
    assert review_class(True, True, 0.0) == "genuine"
    assert review_class(True, True, 0.4) is None      # sometimes right, majority wrong: no class
    assert review_class(False, True, 0.2) == "correct_accept"
    assert review_class(False, False, 1.0) is None


def test_route_catch_takes_top_scores_with_tiebreak():
    from sysone.analysis.e21 import route_catch
    score = {("p", "1"): 0.9, ("p", "2"): 0.5, ("p", "3"): 0.5, ("p", "4"): 0.1}
    wrong = {("p", "1"): True, ("p", "2"): False, ("p", "3"): True, ("p", "4"): True}
    tie = {("p", "1"): 0.0, ("p", "2"): 0.1, ("p", "3"): 0.9, ("p", "4"): 0.0}
    # 50% budget routes 2 items: "1" and, by tiebreak, "3" -> catches 2 of 3 mistakes
    assert route_catch(score, wrong, 0.5, tie) == pytest.approx(2 / 3)


def test_e20a_override_rules():
    from sysone.analysis.e20a import first_n, overrides
    assert not overrides("never", ["b", "b"], "a")
    assert overrides("unanimous", ["b", "b"], "a") and not overrides("unanimous", ["b", "a"], "a")
    assert overrides("majority", ["b", "a"], "a") and not overrides("majority", ["a", "a"], "a")
    assert first_n([(2, "c"), (0, "a"), (1, "b")], 2) == ["a", "b"]
