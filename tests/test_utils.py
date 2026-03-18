"""
Tests for apex_black_box.utils — formatting, statistical intervals, edge & Kelly.
"""
from __future__ import annotations

import math
import pytest

from apex_black_box.utils import (
    format_odd,
    format_probability,
    prob_to_fair_odd,
    parse_result,
    wilson_ci,
    calculate_edge,
    kelly_fraction,
)


class TestFormatOdd:
    def test_format_2_5(self):
        assert format_odd(2.5) == "2.50"

    def test_format_integer(self):
        assert format_odd(3) == "3.00"

    def test_format_custom_decimals(self):
        assert format_odd(1.5, decimals=3) == "1.500"

    def test_format_invalid_returns_na(self):
        assert format_odd("not_a_number") == "N/A"

    def test_format_none_returns_na(self):
        assert format_odd(None) == "N/A"  # type: ignore[arg-type]


class TestFormatProbability:
    def test_as_pct_75(self):
        assert format_probability(0.75, as_pct=True) == "75.0%"

    def test_as_pct_0(self):
        assert format_probability(0.0, as_pct=True) == "0.0%"

    def test_as_pct_1(self):
        assert format_probability(1.0, as_pct=True) == "100.0%"

    def test_as_decimal(self):
        result = format_probability(0.75)
        assert result == "0.750"

    def test_invalid_returns_na(self):
        assert format_probability("bad") == "N/A"


class TestProbToFairOdd:
    def test_half_prob(self):
        assert prob_to_fair_odd(0.5) == 2.0

    def test_quarter_prob(self):
        assert prob_to_fair_odd(0.25) == 4.0

    def test_zero_prob_returns_inf(self):
        result = prob_to_fair_odd(0.0)
        assert math.isinf(result)

    def test_negative_returns_inf(self):
        result = prob_to_fair_odd(-0.1)
        assert math.isinf(result)

    def test_value_near_1(self):
        result = prob_to_fair_odd(0.99)
        assert abs(result - 1.010) < 0.01


class TestParseResult:
    def test_valid_result(self):
        assert parse_result("2-1") == (2, 1)

    def test_zero_zero(self):
        assert parse_result("0-0") == (0, 0)

    def test_high_score(self):
        assert parse_result("5-3") == (5, 3)

    def test_invalid_returns_zero_zero(self):
        assert parse_result("bad") == (0, 0)

    def test_empty_string(self):
        assert parse_result("") == (0, 0)

    def test_partial_format(self):
        assert parse_result("1-") == (0, 0)

    def test_whitespace_padded(self):
        assert parse_result(" 2-1 ") == (2, 1)


class TestWilsonCI:
    def test_bounds_around_half(self):
        lo, hi = wilson_ci(0.5, 100)
        assert lo < 0.5 < hi

    def test_clamped_to_01(self):
        lo, hi = wilson_ci(0.5, 100)
        assert lo >= 0.01
        assert hi <= 0.99

    def test_small_sample(self):
        lo, hi = wilson_ci(0.5, 1)
        assert lo < hi

    def test_extreme_probability(self):
        lo, hi = wilson_ci(0.99, 100)
        # hi should be clamped at 0.99; lo should be reasonably high
        assert lo > 0.9
        assert hi <= 0.99

    def test_zero_probability(self):
        lo, hi = wilson_ci(0.0, 100)
        assert lo >= 0.01
        assert hi > lo

    def test_width_decreases_with_n(self):
        """Larger n should produce a narrower CI."""
        lo1, hi1 = wilson_ci(0.5, 30)
        lo2, hi2 = wilson_ci(0.5, 300)
        width1 = hi1 - lo1
        width2 = hi2 - lo2
        assert width2 < width1


class TestCalculateEdge:
    def test_positive_edge(self):
        """Model prob 0.6, odd 2.0 → positive edge."""
        edge = calculate_edge(0.6, 2.0)
        assert edge > 0

    def test_zero_edge_at_fair_odd(self):
        """Model prob = 1/odd after vig removal → near-zero edge."""
        # At odd=2.0, book_fair_prob = (0.5/1.05) ≈ 0.476; if model = 0.476 → edge ≈ 0
        fair_p = (1.0 / 2.0) / 1.05
        edge = calculate_edge(fair_p, 2.0)
        assert abs(edge) < 0.01

    def test_negative_edge(self):
        """Model prob below book prob → negative edge."""
        edge = calculate_edge(0.3, 2.0)
        assert edge < 0

    def test_odd_at_or_below_1(self):
        """Odd ≤ 1.0 should return 0 (no bet)."""
        assert calculate_edge(0.9, 1.0) == 0.0
        assert calculate_edge(0.9, 0.5) == 0.0

    def test_invalid_inputs(self):
        """Invalid inputs should return 0."""
        assert calculate_edge("bad", 2.0) == 0.0  # type: ignore[arg-type]


class TestKellyFraction:
    def test_positive_fraction(self):
        """Model prob 0.6, odd 2.0 → positive Kelly fraction."""
        k = kelly_fraction(0.6, 2.0)
        assert k > 0

    def test_no_bet_scenario(self):
        """When model prob is low and odd is short → 0 (no bet)."""
        k = kelly_fraction(0.3, 1.5)
        assert k == 0.0

    def test_fraction_capped_at_max_kelly(self):
        """Kelly fraction must not exceed max_kelly (default 0.10)."""
        k = kelly_fraction(0.95, 10.0)
        assert k <= 0.10

    def test_full_kelly_scaled_by_fraction(self):
        """Quarter Kelly (default) should be ≤ full Kelly."""
        k_quarter = kelly_fraction(0.6, 2.0, fraction=0.25)
        k_full = kelly_fraction(0.6, 2.0, fraction=1.0)
        assert k_quarter <= k_full

    def test_odd_at_or_below_1(self):
        """Odd ≤ 1.0 → 0 fraction."""
        assert kelly_fraction(0.9, 1.0) == 0.0

    def test_zero_prob(self):
        """prob=0 → 0 fraction (no bet)."""
        assert kelly_fraction(0.0, 2.0) == 0.0

    def test_result_always_nonnegative(self):
        """Kelly fraction is always ≥ 0."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for odd in [1.2, 1.5, 2.0, 3.0, 5.0]:
                k = kelly_fraction(p, odd)
                assert k >= 0, f"Negative Kelly for p={p}, odd={odd}"
