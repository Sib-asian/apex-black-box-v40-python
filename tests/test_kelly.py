"""
Pytest unit tests for Apex Black Box V40 Kelly Criterion module.

Tests cover:
- Basic Kelly calculation
- VIX adjustments (block < 30, reduce > 60)
- Confidence scaling
- Value bet detection
- Implied probability
- Bankroll updates
- P&L tracking
"""

from __future__ import annotations

import pytest

from apex_black_box.kelly import KellyCriterion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kelly() -> KellyCriterion:
    return KellyCriterion(bankroll=1000.0, max_kelly=0.25, fraction=0.5)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestKellyInit:
    def test_default_parameters(self):
        k = KellyCriterion()
        assert k.bankroll == 1000.0
        assert k.max_kelly == 0.25
        assert k.fraction == 0.5

    def test_invalid_bankroll(self):
        with pytest.raises(ValueError):
            KellyCriterion(bankroll=0.0)

    def test_invalid_max_kelly(self):
        with pytest.raises(ValueError):
            KellyCriterion(max_kelly=0.0)

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            KellyCriterion(fraction=0.0)


# ---------------------------------------------------------------------------
# Implied probability
# ---------------------------------------------------------------------------


class TestImpliedProbability:
    def test_even_money(self, kelly):
        assert kelly.implied_probability(2.0) == pytest.approx(0.5, abs=1e-9)

    def test_short_odds(self, kelly):
        assert kelly.implied_probability(1.5) == pytest.approx(1 / 1.5, abs=1e-9)

    def test_long_odds(self, kelly):
        assert kelly.implied_probability(10.0) == pytest.approx(0.1, abs=1e-9)

    def test_invalid_odds(self, kelly):
        with pytest.raises(ValueError):
            kelly.implied_probability(0.0)


# ---------------------------------------------------------------------------
# Value bet detection
# ---------------------------------------------------------------------------


class TestIsValueBet:
    def test_value_bet_true(self, kelly):
        # prob 0.6 vs implied 0.5 (odds 2.0)
        assert kelly.is_value_bet(prob=0.6, odds=2.0) is True

    def test_value_bet_false(self, kelly):
        # prob 0.4 vs implied 0.5 (odds 2.0)
        assert kelly.is_value_bet(prob=0.4, odds=2.0) is False

    def test_exactly_implied(self, kelly):
        # prob == implied → no edge
        assert kelly.is_value_bet(prob=0.5, odds=2.0) is False


# ---------------------------------------------------------------------------
# Kelly calculation – no edge
# ---------------------------------------------------------------------------


class TestKellyNoEdge:
    def test_negative_edge_returns_zero_stake(self, kelly):
        result = kelly.calculate(prob=0.4, odds=2.0, vix=50.0, confidence=70.0)
        assert result["stake"] == 0.0
        assert result["kelly_full"] == 0.0

    def test_negative_ev(self, kelly):
        result = kelly.calculate(prob=0.4, odds=2.0, vix=50.0, confidence=70.0)
        assert result["ev"] < 0.0


# ---------------------------------------------------------------------------
# Kelly calculation – VIX guard
# ---------------------------------------------------------------------------


class TestKellyVIXGuard:
    def test_vix_below_30_returns_zero(self, kelly):
        result = kelly.calculate(prob=0.6, odds=2.0, vix=20.0, confidence=70.0)
        assert result["stake"] == 0.0
        assert result["kelly_full"] == 0.0

    def test_vix_above_60_reduces_stake(self, kelly):
        result_low_vix  = kelly.calculate(prob=0.6, odds=2.0, vix=50.0, confidence=70.0)
        result_high_vix = kelly.calculate(prob=0.6, odds=2.0, vix=80.0, confidence=70.0)
        assert result_high_vix["stake"] < result_low_vix["stake"]

    def test_vix_at_boundary_30_allows_bet(self, kelly):
        result = kelly.calculate(prob=0.6, odds=2.0, vix=30.0, confidence=70.0)
        assert result["stake"] > 0.0

    def test_vix_just_below_30_blocks(self, kelly):
        result = kelly.calculate(prob=0.6, odds=2.0, vix=29.9, confidence=70.0)
        assert result["stake"] == 0.0


# ---------------------------------------------------------------------------
# Kelly calculation – confidence scaling
# ---------------------------------------------------------------------------


class TestKellyConfidenceScaling:
    def test_higher_confidence_higher_stake(self, kelly):
        low  = kelly.calculate(prob=0.6, odds=2.0, vix=50.0, confidence=40.0)
        high = kelly.calculate(prob=0.6, odds=2.0, vix=50.0, confidence=90.0)
        assert high["stake"] > low["stake"]

    def test_zero_confidence_gives_zero_stake(self, kelly):
        result = kelly.calculate(prob=0.6, odds=2.0, vix=50.0, confidence=0.0)
        assert result["stake"] == 0.0


# ---------------------------------------------------------------------------
# Kelly calculation – stake cap
# ---------------------------------------------------------------------------


class TestKellyStakeCap:
    def test_stake_never_exceeds_max_kelly(self, kelly):
        # Very high edge → would exceed max_kelly without cap
        result = kelly.calculate(prob=0.99, odds=5.0, vix=50.0, confidence=100.0)
        assert result["stake"] <= kelly.bankroll * kelly.max_kelly

    def test_stake_positive_for_value_bet(self, kelly):
        result = kelly.calculate(prob=0.6, odds=2.0, vix=50.0, confidence=70.0)
        assert result["stake"] > 0.0


# ---------------------------------------------------------------------------
# Kelly calculation – returned keys
# ---------------------------------------------------------------------------


class TestKellyReturnedKeys:
    def test_all_keys_present(self, kelly):
        result = kelly.calculate(prob=0.6, odds=2.0, vix=50.0, confidence=70.0)
        for key in ("kelly_full", "kelly_fractional", "stake", "ev", "edge"):
            assert key in result

    def test_fractional_half_of_full(self, kelly):
        result = kelly.calculate(prob=0.6, odds=2.0, vix=50.0, confidence=100.0)
        # With vix=50 (no VIX reduction) and fraction=0.5
        assert result["kelly_fractional"] == pytest.approx(
            result["kelly_full"] * 0.5, rel=1e-5
        )


# ---------------------------------------------------------------------------
# Bankroll management
# ---------------------------------------------------------------------------


class TestBankrollManagement:
    def test_update_bankroll(self, kelly):
        kelly.update_bankroll(2000.0)
        assert kelly.bankroll == 2000.0

    def test_invalid_bankroll_update(self, kelly):
        with pytest.raises(ValueError):
            kelly.update_bankroll(-100.0)

    def test_bet_result_win(self, kelly):
        pnl = kelly.bet_result(stake=100.0, odds=2.0, won=True)
        assert pnl == pytest.approx(100.0, abs=1e-6)
        assert kelly.bankroll == pytest.approx(1100.0, abs=1e-6)

    def test_bet_result_loss(self, kelly):
        pnl = kelly.bet_result(stake=100.0, odds=2.0, won=False)
        assert pnl == pytest.approx(-100.0, abs=1e-6)
        assert kelly.bankroll == pytest.approx(900.0, abs=1e-6)

    def test_max_stake(self, kelly):
        assert kelly.max_stake() == pytest.approx(250.0, abs=1e-6)
