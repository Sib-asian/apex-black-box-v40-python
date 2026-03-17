"""
Pytest unit tests for Apex Black Box V40 core engine.

Tests cover:
- Poisson PMF computation
- xG calculation with DA soft cap
- Lambda blending (sigmoid weighting)
- Score effects (garbage time, post-goal shock, red cards)
- Joint Poisson matrix properties
- Full engine run
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from apex_black_box.core import (
    MatchScore,
    MatchStats,
    OracleEngineV40,
    OracleOutput,
    PreMatchData,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_pre() -> PreMatchData:
    return PreMatchData(
        spread_open=-0.5,
        spread_curr=-0.5,
        total_open=2.5,
        total_curr=2.5,
        is_knockout=False,
    )


@pytest.fixture
def default_score() -> MatchScore:
    return MatchScore(min=45, rec=0, hg=1, ag=0, last_goal=40)


@pytest.fixture
def default_stats() -> MatchStats:
    return MatchStats(
        sot_h=3, mis_h=2, cor_h=3, da_h=10, poss_h=55.0,
        sot_a=1, mis_a=1, cor_a=1, da_a=5,  poss_a=45.0,
    )


@pytest.fixture
def engine(default_pre, default_score, default_stats) -> OracleEngineV40:
    return OracleEngineV40(default_pre, default_score, default_stats)


# ---------------------------------------------------------------------------
# Data-class validation
# ---------------------------------------------------------------------------


class TestMatchScoreValidation:
    def test_valid_score(self):
        s = MatchScore(min=90, rec=3, hg=2, ag=1)
        assert s.min == 90

    def test_invalid_minute_high(self):
        with pytest.raises(ValueError):
            MatchScore(min=121, rec=0, hg=0, ag=0)

    def test_invalid_minute_low(self):
        with pytest.raises(ValueError):
            MatchScore(min=-1, rec=0, hg=0, ag=0)

    def test_negative_goals(self):
        with pytest.raises(ValueError):
            MatchScore(min=10, rec=0, hg=-1, ag=0)

    def test_invalid_red_cards(self):
        with pytest.raises(ValueError):
            MatchScore(min=10, rec=0, hg=0, ag=0, red_h=3)


class TestMatchStatsValidation:
    def test_valid_stats(self):
        s = MatchStats(sot_h=5, mis_h=3, cor_h=4, da_h=20, poss_h=60.0,
                       sot_a=2, mis_a=1, cor_a=2, da_a=10, poss_a=40.0)
        assert s.poss_h == 60.0

    def test_negative_sot(self):
        with pytest.raises(ValueError):
            MatchStats(sot_h=-1, mis_h=0, cor_h=0, da_h=0, poss_h=50.0,
                       sot_a=0, mis_a=0, cor_a=0, da_a=0, poss_a=50.0)

    def test_possession_out_of_range(self):
        with pytest.raises(ValueError):
            MatchStats(sot_h=0, mis_h=0, cor_h=0, da_h=0, poss_h=110.0,
                       sot_a=0, mis_a=0, cor_a=0, da_a=0, poss_a=40.0)


# ---------------------------------------------------------------------------
# Poisson PMF
# ---------------------------------------------------------------------------


class TestPoissonPMF:
    def test_zero_goals(self, engine):
        p = engine.poisson_pmf(lam=1.5, k=0)
        expected = math.exp(-1.5)
        assert abs(p - expected) < 1e-10

    def test_one_goal(self, engine):
        p = engine.poisson_pmf(lam=1.5, k=1)
        expected = 1.5 * math.exp(-1.5)
        assert abs(p - expected) < 1e-10

    def test_probabilities_sum_to_one(self, engine):
        lam = 1.8
        total = sum(engine.poisson_pmf(lam=lam, k=k) for k in range(30))
        assert abs(total - 1.0) < 1e-6

    def test_zero_lambda(self, engine):
        p = engine.poisson_pmf(lam=0.0, k=0)
        assert p == pytest.approx(1.0, abs=1e-10)

    def test_large_k_near_zero(self, engine):
        p = engine.poisson_pmf(lam=1.0, k=20)
        assert p < 1e-10


# ---------------------------------------------------------------------------
# xG calculation
# ---------------------------------------------------------------------------


class TestXGCalculation:
    def test_zero_shots_gives_small_xg(self, engine):
        xg = engine.calc_xg(sot=0, mis=0, da=0)
        assert xg >= 0.0
        assert xg < 0.1

    def test_more_shots_more_xg(self, engine):
        xg_low  = engine.calc_xg(sot=1, mis=1, da=5)
        xg_high = engine.calc_xg(sot=5, mis=3, da=20)
        assert xg_high > xg_low

    def test_da_soft_cap(self, engine):
        """Each DA above 35 should add less than one below 35."""
        xg_35 = engine.calc_xg(sot=0, mis=0, da=35)
        xg_36 = engine.calc_xg(sot=0, mis=0, da=36)
        xg_70 = engine.calc_xg(sot=0, mis=0, da=70)
        # Increment from 35→36 should be smaller than from 34→35
        xg_34 = engine.calc_xg(sot=0, mis=0, da=34)
        delta_below_cap = xg_35 - xg_34
        delta_above_cap = xg_36 - xg_35
        assert delta_above_cap < delta_below_cap

    def test_xg_non_negative(self, engine):
        xg = engine.calc_xg(sot=0, mis=0, da=0)
        assert xg >= 0.0


# ---------------------------------------------------------------------------
# Lambda blending
# ---------------------------------------------------------------------------


class TestLambdaBlending:
    def test_early_game_closer_to_prior(self, engine):
        """At minute 1, the blend should weight prior heavily."""
        rate = 2.0
        prior = 1.5
        blended = engine.blend_lambda(xg_rate=rate, prior=prior, minute=1)
        # Should be closer to prior than to rate
        assert abs(blended - prior) < abs(blended - rate)

    def test_late_game_smaller_output_than_early(self, engine):
        """blend_lambda returns expected goals *remaining*, so later minutes
        should give a smaller absolute output (less time left)."""
        rate = 2.0
        prior = 1.5
        early = engine.blend_lambda(xg_rate=rate, prior=prior, minute=10)
        late  = engine.blend_lambda(xg_rate=rate, prior=prior, minute=80)
        assert late < early

    def test_midpoint_returns_positive(self, engine):
        """At minute 45, the result should be a positive expected goals count."""
        rate = 2.0
        prior = 1.0
        blended = engine.blend_lambda(xg_rate=rate, prior=prior, minute=45)
        assert blended > 0.0

    def test_monotone_decreasing_with_minute(self, engine):
        """Expected goals remaining must decrease monotonically as elapsed minutes
        increase (less time left on the clock)."""
        rate = 2.0
        prior = 1.0
        values = [engine.blend_lambda(xg_rate=rate, prior=prior, minute=m)
                  for m in [10, 30, 45, 60, 80]]
        assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# Score effects
# ---------------------------------------------------------------------------


class TestScoreEffects:
    def test_garbage_time_reduces_leading_lambda(self):
        """A 3-goal lead should reduce the leading team's lambda."""
        pre = PreMatchData(spread_open=0.0, spread_curr=0.0,
                           total_open=2.5, total_curr=2.5, is_knockout=False)
        score = MatchScore(min=75, rec=0, hg=3, ag=0, last_goal=70)
        stats = MatchStats(sot_h=8, mis_h=4, cor_h=5, da_h=25, poss_h=65.0,
                           sot_a=1, mis_a=1, cor_a=1, da_a=5,  poss_a=35.0)
        engine = OracleEngineV40(pre, score, stats)
        lam_h_normal = 2.0
        lam_a_normal = 1.0
        lam_h_eff, _ = engine.apply_score_effects(lam_h_normal, lam_a_normal)
        assert lam_h_eff < lam_h_normal

    def test_red_card_reduces_lambda(self):
        """A home red card should reduce home lambda."""
        pre = PreMatchData(spread_open=0.0, spread_curr=0.0,
                           total_open=2.5, total_curr=2.5, is_knockout=False)
        score = MatchScore(min=60, rec=0, hg=0, ag=0, last_goal=-1, red_h=1)
        stats = MatchStats(sot_h=2, mis_h=1, cor_h=2, da_h=8, poss_h=45.0,
                           sot_a=3, mis_a=2, cor_a=3, da_a=12, poss_a=55.0)
        engine_rc = OracleEngineV40(pre, score, stats)

        score_no_rc = MatchScore(min=60, rec=0, hg=0, ag=0, last_goal=-1, red_h=0)
        engine_no_rc = OracleEngineV40(pre, score_no_rc, stats)

        lam_base = 1.5
        lam_h_rc, _ = engine_rc.apply_score_effects(lam_base, 1.2)
        lam_h_no, _ = engine_no_rc.apply_score_effects(lam_base, 1.2)
        assert lam_h_rc < lam_h_no


# ---------------------------------------------------------------------------
# Joint Poisson matrix
# ---------------------------------------------------------------------------


class TestJointMatrix:
    def test_matrix_shape(self, engine):
        mat = engine.poisson_joint_matrix(lam_h=1.5, lam_a=1.2)
        assert mat.shape == (9, 9)  # 0..8

    def test_matrix_sums_to_one(self, engine):
        mat = engine.poisson_joint_matrix(lam_h=1.5, lam_a=1.2)
        assert abs(mat.sum() - 1.0) < 1e-4

    def test_all_non_negative(self, engine):
        mat = engine.poisson_joint_matrix(lam_h=1.5, lam_a=1.2)
        assert (mat >= 0).all()


# ---------------------------------------------------------------------------
# Full engine run
# ---------------------------------------------------------------------------


class TestFullRun:
    def test_run_returns_oracle_output(self, engine):
        output = engine.run()
        assert isinstance(output, OracleOutput)

    def test_probs_sum_to_one(self, engine):
        output = engine.run()
        total = output.probs["1"] + output.probs["X"] + output.probs["2"]
        assert abs(total - 1.0) < 1e-4

    def test_over_under_complementary(self, engine):
        output = engine.run()
        assert abs(output.probs["Over25"] + output.probs["Under25"] - 1.0) < 1e-4

    def test_confidence_in_range(self, engine):
        output = engine.run()
        assert 0.0 <= output.confidence <= 100.0

    def test_vix_in_range(self, engine):
        output = engine.run()
        assert 0.0 <= output.vix <= 100.0

    def test_all_required_markets_present(self, engine):
        output = engine.run()
        for market in ("1", "X", "2", "Over25", "Under25", "BTTS", "DNB_H", "DNB_A"):
            assert market in output.probs, f"Missing market: {market}"

    def test_all_probs_between_zero_and_one(self, engine):
        output = engine.run()
        for market, p in output.probs.items():
            assert 0.0 <= p <= 1.0, f"Probability out of range for {market}: {p}"

    def test_knockout_has_no_draw(self):
        pre = PreMatchData(spread_open=0.0, spread_curr=0.0,
                           total_open=2.5, total_curr=2.5, is_knockout=True)
        score = MatchScore(min=90, rec=0, hg=1, ag=1, last_goal=85)
        stats = MatchStats(sot_h=4, mis_h=2, cor_h=3, da_h=12, poss_h=50.0,
                           sot_a=4, mis_a=2, cor_a=3, da_a=12, poss_a=50.0)
        engine = OracleEngineV40(pre, score, stats)
        output = engine.run()
        # In knockout, X probability should be 0 (no draw possible at 90')
        assert output.probs["X"] == pytest.approx(0.0, abs=1e-6)
