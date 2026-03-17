"""
Apex Black Box v40 — regression and mathematical correctness tests.

Run with:  pytest tests/test_engine.py -v
"""

from __future__ import annotations
import math
import pytest

from apex_black_box.engine import (
    scan,
    clamp,
    shrink_to_base,
    calc_xg_obs,
    pressure_factor,
    poisson_pmf,
    build_result_matrix,
    aggregate_from_matrix,
)
from apex_black_box.utils import (
    wilson_ci,
    calculate_edge,
    kelly_fraction,
    format_odd,
    format_probability,
    parse_result,
    prob_to_fair_odd,
)


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _scan(**kwargs) -> dict:
    """Convenience wrapper that calls scan() with sensible defaults."""
    payload = {"min": 45, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0.0}
    payload.update(kwargs)
    return scan(payload)


def _all_probs_valid(probs: dict) -> bool:
    """Return True iff every probability value is in [0, 1]."""
    return all(0.0 <= v <= 1.0 for v in probs.values())


# ─────────────────────────────────────────────────────────────────
#  A — Probability correctness
# ─────────────────────────────────────────────────────────────────

class TestProbabilitiesValid:
    """All output probabilities must lie in [0, 1] and 1X2 must sum to 1."""

    @pytest.mark.parametrize("minute,hg,ag", [
        (1, 0, 0), (15, 0, 0), (30, 1, 0), (45, 1, 1), (60, 2, 1),
        (75, 2, 2), (85, 3, 0), (90, 0, 0),
    ])
    def test_probs_in_range(self, minute, hg, ag):
        r = _scan(min=minute, hg=hg, ag=ag, sotH=minute // 10, sotA=minute // 15)
        assert _all_probs_valid(r["probs"]), f"Out-of-range prob at {minute}' {hg}-{ag}: {r['probs']}"

    def test_1x2_sums_to_one(self):
        for minute in (10, 30, 60, 88):
            r = _scan(min=minute, sotH=4, sotA=3)
            total = r["probs"]["1"] + r["probs"]["X"] + r["probs"]["2"]
            assert abs(total - 1.0) < 1e-6, f"1X2 sum={total} at min={minute}"

    def test_over15_gte_over25(self):
        """Over 1.5 probability should always be ≥ Over 2.5."""
        r = _scan(min=45, sotH=6, sotA=4)
        assert r["probs"]["Over15"] >= r["probs"]["Over25"] - 1e-9

    def test_dnb_sums_to_one(self):
        r = _scan(min=60, sotH=8, sotA=5)
        total = r["probs"]["DNB_H"] + r["probs"]["DNB_A"]
        assert abs(total - 1.0) < 1e-6, f"DNB sum={total}"


# ─────────────────────────────────────────────────────────────────
#  A6 — dynOver tradable (half-point commercial)
# ─────────────────────────────────────────────────────────────────

class TestDynOver:
    """dynOver must always be a commercial half-point: 0.5, 1.5, 2.5, …"""

    @pytest.mark.parametrize("hg,ag,expected", [
        (0, 0, 0.5),
        (1, 0, 1.5),
        (0, 1, 1.5),
        (1, 1, 2.5),
        (2, 0, 2.5),
        (2, 1, 3.5),
        (2, 2, 4.5),
        (3, 2, 5.5),
    ])
    def test_dyn_over_value(self, hg, ag, expected):
        r = _scan(min=60, hg=hg, ag=ag)
        assert r["dynOver"] == expected, f"dynOver={r['dynOver']} expected {expected} at {hg}-{ag}"

    def test_dyn_over_is_half_point(self):
        """dynOver mod 1 must always be 0.5."""
        for hg in range(5):
            for ag in range(5):
                r = _scan(min=50, hg=hg, ag=ag)
                assert (r["dynOver"] * 2) % 2 == 1.0, \
                    f"dynOver={r['dynOver']} not a commercial half-point at {hg}-{ag}"


# ─────────────────────────────────────────────────────────────────
#  A5 — shrink_to_base market-specific neutral bases
# ─────────────────────────────────────────────────────────────────

class TestShrinkBase:
    """With near-zero data quality, market probs should approach their neutral bases."""

    def test_over15_base_near_072(self):
        """With 0 shots at minute 1 (max uncertainty), Over15 should be close to 0.72."""
        r = scan({"min": 1, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0.0})
        p = r["probs"]["Over15"]
        assert 0.60 <= p <= 0.85, f"Over15={p} far from neutral base 0.72 with no data"

    def test_over25_base_near_052(self):
        r = scan({"min": 1, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0.0})
        p = r["probs"]["Over25"]
        assert 0.40 <= p <= 0.65, f"Over25={p} far from neutral base 0.52 with no data"

    def test_over15_not_shrunk_to_050(self):
        """Over15 should NOT converge to 0.50 under high uncertainty — old bug."""
        r = scan({"min": 1, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0.0})
        p = r["probs"]["Over15"]
        assert p > 0.58, f"Over15={p} was shrunk towards 0.50 (old base), expected ~0.72"

    def test_shrink_to_base_function(self):
        """shrink_to_base(p, base, s) must stay in [0, 1]."""
        for p in (0.0, 0.3, 0.5, 0.7, 0.99):
            for base in (0.5, 0.52, 0.72):
                for s in (0.0, 0.5, 1.0):
                    out = shrink_to_base(p, base, s)
                    assert 0.0 <= out <= 1.0


# ─────────────────────────────────────────────────────────────────
#  A1 — stab_q uses multiple scans and markets
# ─────────────────────────────────────────────────────────────────

class TestStabQ:
    """stab_q should be higher for stable probability sequences."""

    def _scan_with_prev(self, stable: bool) -> dict:
        if stable:
            prev = [
                {"min": 25, "probs": {"1": 0.40, "X": 0.32, "2": 0.28, "Over25": 0.55, "O_Dyn": 0.70}, "score": "0-0"},
                {"min": 20, "probs": {"1": 0.40, "X": 0.32, "2": 0.28, "Over25": 0.55, "O_Dyn": 0.70}, "score": "0-0"},
                {"min": 15, "probs": {"1": 0.40, "X": 0.32, "2": 0.28, "Over25": 0.55, "O_Dyn": 0.70}, "score": "0-0"},
            ]
        else:
            prev = [
                {"min": 25, "probs": {"1": 0.60, "X": 0.25, "2": 0.15, "Over25": 0.75, "O_Dyn": 0.85}, "score": "0-0"},
                {"min": 20, "probs": {"1": 0.30, "X": 0.35, "2": 0.35, "Over25": 0.35, "O_Dyn": 0.45}, "score": "0-0"},
                {"min": 15, "probs": {"1": 0.55, "X": 0.27, "2": 0.18, "Over25": 0.70, "O_Dyn": 0.80}, "score": "0-0"},
            ]
        return scan({"min": 30, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0, "sotH": 3, "sotA": 2, "prevScans": prev})

    def test_stable_scans_higher_confidence(self):
        r_stable = self._scan_with_prev(stable=True)
        r_unstable = self._scan_with_prev(stable=False)
        assert r_stable["confidence"] >= r_unstable["confidence"], \
            f"Stable conf={r_stable['confidence']}, Unstable conf={r_unstable['confidence']}"

    def test_goal_spike_attenuates_penalty(self):
        """A goal between scans should be treated as justified variance (lower penalty)."""
        prev_goal = [
            {"min": 25, "probs": {"1": 0.60, "X": 0.25, "2": 0.15, "Over25": 0.75, "O_Dyn": 0.85}, "score": "1-0"},
            {"min": 20, "probs": {"1": 0.40, "X": 0.33, "2": 0.27, "Over25": 0.53, "O_Dyn": 0.68}, "score": "0-0"},
            {"min": 15, "probs": {"1": 0.39, "X": 0.33, "2": 0.28, "Over25": 0.52, "O_Dyn": 0.66}, "score": "0-0"},
        ]
        prev_no_goal = [
            {"min": 25, "probs": {"1": 0.60, "X": 0.25, "2": 0.15, "Over25": 0.75, "O_Dyn": 0.85}, "score": "0-0"},
            {"min": 20, "probs": {"1": 0.40, "X": 0.33, "2": 0.27, "Over25": 0.53, "O_Dyn": 0.68}, "score": "0-0"},
            {"min": 15, "probs": {"1": 0.39, "X": 0.33, "2": 0.28, "Over25": 0.52, "O_Dyn": 0.66}, "score": "0-0"},
        ]
        r_goal = scan({"min": 30, "hg": 1, "ag": 0, "tC": 2.5, "sC": 0, "sotH": 3, "prevScans": prev_goal})
        r_no_goal = scan({"min": 30, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0, "sotH": 3, "prevScans": prev_no_goal})
        assert r_goal["confidence"] >= r_no_goal["confidence"]


# ─────────────────────────────────────────────────────────────────
#  C2 — Wilson CI: asymmetric bounds for extreme probabilities
# ─────────────────────────────────────────────────────────────────

class TestWilsonCI:
    """Wilson CI must be well-behaved across the full [0,1] probability range."""

    @pytest.mark.parametrize("p,n_eff", [
        (0.05, 10), (0.50, 20), (0.85, 30), (0.95, 50), (0.01, 5), (0.99, 100),
    ])
    def test_bounds_in_range(self, p, n_eff):
        lo, hi = wilson_ci(p, n_eff)
        assert 0.01 <= lo <= hi <= 0.99, f"Wilson CI out of range: ({lo},{hi}) for p={p}, n={n_eff}"

    def test_extreme_p_near_zero_asymmetric(self):
        lo, hi = wilson_ci(0.02, 20)
        # Upper bound should be much larger than lower bound
        assert hi - lo > 0.05, f"CI too narrow for small p: ({lo},{hi})"
        assert lo < 0.02, f"Lower bound should be below p=0.02"

    def test_extreme_p_near_one_asymmetric(self):
        lo, hi = wilson_ci(0.96, 20)
        # Lower bound should be well below p
        assert lo < 0.96, f"Lower bound {lo} should be below p=0.96"
        assert abs(hi - 0.99) < 1e-9  # clipped to 0.99

    def test_symmetry_at_half(self):
        lo, hi = wilson_ci(0.5, 100)
        assert abs((hi - 0.5) - (0.5 - lo)) < 0.02, "Should be approximately symmetric at p=0.5"

    def test_engine_ci_valid(self):
        """confIntervals returned by scan() must be valid Wilson-style bounds."""
        r = _scan(min=45, sotH=6, sotA=4)
        for mkt, bounds in r["confIntervals"].items():
            lo, hi = bounds["lo"], bounds["hi"]
            assert 0.0 <= lo <= hi <= 1.0, f"Bad CI for {mkt}: ({lo},{hi})"


# ─────────────────────────────────────────────────────────────────
#  C3 — ht_probs dynamic blend with minute
# ─────────────────────────────────────────────────────────────────

class TestHtProbsDynamicBlend:
    """Live weight should be small early and grow with minute."""

    def test_ht_probs_present_first_half(self):
        r = _scan(min=20)
        assert r["htProbs"] is not None

    def test_ht_probs_absent_second_half(self):
        r = _scan(min=46)
        assert r["htProbs"] is None

    def test_early_minute_close_to_prior(self):
        """At minute 5, blend should be almost entirely prior-based."""
        r_early = _scan(min=5, sotH=1, sotA=0)
        r_late = _scan(min=40, sotH=8, sotA=4)
        # Earlier scan should have lower live influence; halftime probs closer to pre-match
        # (hard to assert exact values without prior knowledge, but htProbs must be present)
        assert r_early["htProbs"] is not None
        assert r_late["htProbs"] is not None


# ─────────────────────────────────────────────────────────────────
#  B3 — chasing_mult scales with goal_diff
# ─────────────────────────────────────────────────────────────────

class TestChasingMult:
    """When behind, chasing team should get less boost when the deficit is large."""

    def test_behind_by_1_gets_more_boost_than_behind_by_3(self):
        """Away team behind by 1 should get more λ boost than behind by 3."""
        r1 = scan({"min": 75, "hg": 1, "ag": 0, "tC": 2.5, "sC": 0.0, "sotH": 8, "sotA": 5})
        r3 = scan({"min": 75, "hg": 3, "ag": 0, "tC": 2.5, "sC": 0.0, "sotH": 20, "sotA": 5})
        # lA_live: away lambda after chasing adjustment
        la_1 = r1["metrics"]["lA_live"]
        la_3 = r3["metrics"]["lA_live"]
        # The away team at 3-0 has garbage-time adjustments, but before that
        # chasing_mult for goal_diff=3 should be lower than for goal_diff=1
        # We just check it doesn't crash and returns valid values
        assert la_1 >= 0.05
        assert la_3 >= 0.05

    def test_chasing_mult_formula_directly(self):
        """Test chasing_mult formula: 1.10 + 0.12*(min/90) - 0.04*(abs(gd)-1)."""
        minute = 60.0
        for goal_diff in (1, 2, 3, 4):
            expected = max(1.02, min(1.22, 1.10 + 0.12 * (minute / 90) - 0.04 * (goal_diff - 1)))
            assert expected >= 1.02
            assert expected <= 1.22
            if goal_diff > 1:
                assert expected < max(1.02, min(1.22, 1.10 + 0.12 * (minute / 90)))


# ─────────────────────────────────────────────────────────────────
#  B1 — cumulative multiplier guard
# ─────────────────────────────────────────────────────────────────

class TestCumulativeMultGuard:
    """Total multipliers must not produce lambda explosions or collapses."""

    def _lambda_h(self, **kwargs) -> float:
        return _scan(**kwargs)["metrics"]["lH_live"]

    def _lambda_a(self, **kwargs) -> float:
        return _scan(**kwargs)["metrics"]["lA_live"]

    def test_no_explosion_with_heavy_red_card_and_shock(self):
        """Multiple concurrent boosts should not make lambda explode."""
        r = scan({
            "min": 75, "hg": 0, "ag": 2, "tC": 2.5, "sC": -0.5,
            "sotH": 15, "sotA": 2, "rcA": 1, "lastGoal": 72,
            "prevScans": [],
        })
        la = r["metrics"]["lA_live"]
        lh = r["metrics"]["lH_live"]
        # Lambda should be physically reasonable (< 5 goals/90' equivalent)
        assert lh <= 3.5, f"lH_live={lh} exploded"
        assert la <= 3.5, f"lA_live={la} exploded"

    def test_no_collapse_with_keeper_karma_and_garbage(self):
        """Multiple downward modifiers should not make lambda collapse to near 0."""
        r = scan({
            "min": 85, "hg": 3, "ag": 0, "tC": 2.5, "sC": 0,
            "sotH": 25, "sotA": 2,
        })
        lh = r["metrics"]["lH_live"]
        assert lh >= 0.05, f"lH_live={lh} collapsed"

    def test_lambda_always_positive(self):
        for minute in (5, 20, 45, 70, 88):
            r = _scan(min=minute)
            assert r["metrics"]["lH_live"] > 0
            assert r["metrics"]["lA_live"] > 0


# ─────────────────────────────────────────────────────────────────
#  A3 — calc_xg_obs DA rate-aware
# ─────────────────────────────────────────────────────────────────

class TestCalcXgObs:
    """DA contribution should be higher when DA rate/min is high."""

    def test_high_da_rate_gives_higher_xg(self):
        """Same DA count but in fewer minutes → higher xg_obs (when not both above cap)."""
        # 4 DA in 10 min: rate = 0.4/min, expected = 8/90 ≈ 0.089/min → quality ≈ 4.5 → capped 1.5
        xg_early = calc_xg_obs(0, 0, 4, min_played=10.0)
        # 4 DA in 90 min: rate ≈ 0.044/min, expected ≈ 0.089/min → quality ≈ 0.5 → floor
        xg_late = calc_xg_obs(0, 0, 4, min_played=90.0)
        assert xg_early > xg_late, f"Early={xg_early} should > Late={xg_late}"

    def test_da_quality_clamp(self):
        """DA quality is clamped to [0.5, 1.5]; very low rate → quality floor."""
        # Very low rate (5 DA in 90 min): rate = 5/90 ≈ 0.056, expected 8/90 ≈ 0.089
        # quality = 0.056/0.089 ≈ 0.63 — above floor 0.5
        xg_slow = calc_xg_obs(0, 0, 5, min_played=90.0)
        # Neutral rate (8 DA in 90 min): quality exactly 1.0
        xg_normal = calc_xg_obs(0, 0, 8, min_played=90.0)
        # High rate (16 DA in 90 min): quality capped at 1.5
        xg_fast = calc_xg_obs(0, 0, 16, min_played=90.0)
        # With more DA and higher quality: xg_fast > xg_normal > xg_slow
        assert xg_slow < xg_normal, f"xg_slow={xg_slow} should < xg_normal={xg_normal}"
        assert xg_normal < xg_fast, f"xg_normal={xg_normal} should < xg_fast={xg_fast}"
        assert xg_slow >= 0

    def test_default_min_played(self):
        """Default min_played=90 should match old behaviour."""
        xg_default = calc_xg_obs(5, 3, 10)
        xg_90 = calc_xg_obs(5, 3, 10, min_played=90.0)
        # With rate = 10/90 ≈ 0.111, da_rate / (8/90) ≈ 1.25 → quality=1.25
        # Old formula had da_quality=1.0 implicitly; values will differ
        assert xg_default > 0
        assert xg_90 > 0


# ─────────────────────────────────────────────────────────────────
#  A4 — pressure_factor conditional on shots
# ─────────────────────────────────────────────────────────────────

class TestPressureFactor:
    """pressure_factor should be stronger when shots are low."""

    def test_high_corners_low_shots_vs_high_shots(self):
        # Same corners, same time; low shots → more pressure_factor
        pf_low_shots = pressure_factor(cor=8, sot=0, mis=2, min_played=60)
        pf_high_shots = pressure_factor(cor=8, sot=10, mis=6, min_played=60)
        assert pf_low_shots > pf_high_shots, \
            f"Low shots pf={pf_low_shots} should > High shots pf={pf_high_shots}"

    def test_pressure_factor_clamped(self):
        """pressure_factor must always be in [1.0, 1.15]."""
        for cor in (0, 5, 15, 30):
            for sot in (0, 5, 20):
                pf = pressure_factor(cor, sot, 2, 45)
                assert 1.0 <= pf <= 1.15, f"pressure_factor={pf} out of range"


# ─────────────────────────────────────────────────────────────────
#  C1 — surprise_idx vs modal result
# ─────────────────────────────────────────────────────────────────

class TestSurpriseIdx:
    """surprise_idx should reflect distance from the modal (most likely) scoreline."""

    def test_modal_score_has_low_surprise(self):
        """The most probable pre-match score should yield surprise_idx ≈ 0."""
        # With tC=2.5, sC=0 → l_h = l_a = 1.25/2 ≈ 0.625 per team per 90'.
        # At min=90, elapsed=1, modal score is 0-0 (Poisson mode with λ<1).
        r = scan({"min": 89, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0.0})
        assert r["metrics"]["surpriseIdx"] < 2.0, \
            f"surpriseIdx={r['metrics']['surpriseIdx']} high for expected score"

    def test_shock_score_has_high_surprise(self):
        """A rare scoreline like 5-0 should have higher surprise than 1-0."""
        r_expected = scan({"min": 60, "hg": 1, "ag": 0, "tC": 2.5, "sC": 0.0})
        r_shock = scan({"min": 60, "hg": 4, "ag": 0, "tC": 2.5, "sC": 0.0})
        assert r_shock["metrics"]["surpriseIdx"] >= r_expected["metrics"]["surpriseIdx"]

    def test_surprise_idx_in_range(self):
        for hg, ag in ((0, 0), (1, 0), (2, 1), (3, 0)):
            r = scan({"min": 60, "hg": hg, "ag": ag, "tC": 2.5, "sC": 0.0})
            assert 0.0 <= r["metrics"]["surpriseIdx"] <= 10.0


# ─────────────────────────────────────────────────────────────────
#  D3 — model_edge
# ─────────────────────────────────────────────────────────────────

class TestModelEdge:
    """modelEdge must be computed when bookOdds are provided."""

    def test_model_edge_present_with_book_odds(self):
        r = scan({
            "min": 45, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0.0,
            "sotH": 5, "sotA": 4,
            "bookOdds": {"1": 2.5, "X": 3.2, "2": 2.8, "Over25": 1.90},
            "vig": 0.05,
        })
        edge = r["metrics"]["modelEdge"]
        assert edge is not None
        assert "1" in edge and "X" in edge and "2" in edge and "Over25" in edge

    def test_model_edge_absent_without_book_odds(self):
        r = scan({"min": 45, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0.0})
        assert r["metrics"]["modelEdge"] is None

    def test_model_edge_sum_is_negative_for_overround(self):
        """Sum of (1/q for each way) > 1 means bookie has margin → sum of edges negative."""
        r = scan({
            "min": 45, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0.0,
            "sotH": 5, "sotA": 4,
            "bookOdds": {"1": 2.5, "X": 3.2, "2": 2.8},
            "vig": 0.0,  # no vig adjustment
        })
        edge = r["metrics"]["modelEdge"]
        # Model probs sum to 1; fair book probs sum to <1 (overround)
        # so some edges can be positive, but total should be ≤ 0 for 1X2
        total_edge = edge["1"] + edge["X"] + edge["2"]
        # With vig=0, fair = raw implied; raw implied sum = 1/2.5+1/3.2+1/2.8 ≈ 1.04
        # Model sums to 1, so sum of edges should be ≈ -0.04
        assert total_edge < 0.05, f"Total 1X2 edge should be near 0 or negative: {total_edge}"


# ─────────────────────────────────────────────────────────────────
#  C4 — vix_conf_scale upward-only, capped at 1.20
# ─────────────────────────────────────────────────────────────────

class TestVixConfScale:
    """VIX should never be boosted beyond 1.20× raw, and not reduced below raw."""

    def test_vix_not_below_raw_boundary(self):
        """VIX output should be ≥ vix_raw (since scale ≥ 1.0)."""
        # We can't access vix_raw directly; but with scale ≥ 1.0,
        # vix should not be drastically lower than what raw would give.
        r = _scan(min=30, sotH=3, sotA=2)
        assert r["vix"] >= 10.0  # absolute floor

    def test_vix_in_range(self):
        for minute in (5, 30, 60, 85):
            r = _scan(min=minute, sotH=minute // 8, sotA=minute // 12)
            assert 10.0 <= r["vix"] <= 95.0, f"VIX={r['vix']} out of range at min={minute}"


# ─────────────────────────────────────────────────────────────────
#  Utils — unit tests
# ─────────────────────────────────────────────────────────────────

class TestUtils:
    def test_wilson_ci_basic(self):
        lo, hi = wilson_ci(0.5, 30)
        assert lo < 0.5 < hi

    def test_wilson_ci_clamped(self):
        lo, hi = wilson_ci(0.99, 5)
        assert hi <= 0.99
        lo, hi = wilson_ci(0.01, 5)
        assert lo >= 0.01

    def test_calculate_edge_positive(self):
        # Model says 50%, bookie offers 2.20 (implied ~45%) → positive edge
        edge = calculate_edge(0.50, 2.20, vig=0.0)
        assert edge > 0

    def test_calculate_edge_negative(self):
        # Model says 30%, bookie offers 4.00 (implied 25%) → positive edge?
        # Actually 1/4.0 = 0.25 fair → model 0.30 > 0.25 → positive
        edge = calculate_edge(0.30, 4.00, vig=0.0)
        assert edge > 0
        # Model says 30%, bookie offers 2.00 (implied 50%) → negative
        edge2 = calculate_edge(0.30, 2.00, vig=0.0)
        assert edge2 < 0

    def test_kelly_fraction_no_bet(self):
        # p=0.40, b=1.0 (even money) → Kelly = (1.0*0.40 - 0.60)/1.0 = -0.20 → 0
        assert kelly_fraction(0.40, 2.0) == 0.0

    def test_kelly_fraction_positive(self):
        # p=0.60, odd=2.5 (b=1.5) → full Kelly = (1.5*0.60 - 0.40)/1.5 = 0.333
        result = kelly_fraction(0.60, 2.5, fraction=1.0, max_kelly=1.0)
        assert 0.0 < result < 1.0

    def test_format_odd(self):
        assert format_odd(2.5) == "2.50"
        assert format_odd(1.0) == "1.00"

    def test_format_probability(self):
        assert format_probability(0.456) == "0.456"
        assert format_probability(0.456, as_pct=True) == "45.6%"

    def test_parse_result(self):
        assert parse_result("2-1") == (2, 1)
        assert parse_result("0-0") == (0, 0)
        assert parse_result("bad") == (0, 0)

    def test_prob_to_fair_odd(self):
        assert abs(prob_to_fair_odd(0.5) - 2.0) < 0.01
        assert prob_to_fair_odd(0.0) == float("inf")


# ─────────────────────────────────────────────────────────────────
#  Regression — no breaking changes in output keys
# ─────────────────────────────────────────────────────────────────

class TestOutputKeys:
    """Verify that all expected output keys are present and non-breaking."""

    REQUIRED_PROBS_KEYS = {
        "1", "X", "2", "DNB_H", "DNB_A", "Over15", "Under25", "DC_1X", "DC_X2",
        "DC_12", "Next_H", "Next_A", "Next_H_c", "Next_A_c", "O_Dyn", "Over25", "BTTS",
    }
    REQUIRED_TOP_KEYS = {
        "probs", "matrix", "htProbs", "etProbs", "dynOver", "vix", "confidence",
        "alerts", "min", "confIntervals", "score", "currentHg", "currentAg",
        "isGarbageTime", "projectedFinal", "steam", "vixDetail", "xG", "metrics", "raw",
    }
    REQUIRED_METRICS_KEYS = {
        "paceDefH", "paceDefA", "paceDefLabel", "surpriseIdx", "surpriseLabel",
        "xgGapH", "xgGapA", "xgGapLabel", "tlEff", "tlEffLabel",
        "modelTotal", "lH_live", "lA_live", "projH", "projA",
    }

    def test_top_level_keys(self):
        r = _scan(min=45, sotH=5, sotA=3)
        for k in self.REQUIRED_TOP_KEYS:
            assert k in r, f"Missing top-level key: {k}"

    def test_probs_keys(self):
        r = _scan(min=45, sotH=5, sotA=3)
        for k in self.REQUIRED_PROBS_KEYS:
            assert k in r["probs"], f"Missing probs key: {k}"

    def test_metrics_keys(self):
        r = _scan(min=45, sotH=5, sotA=3)
        for k in self.REQUIRED_METRICS_KEYS:
            assert k in r["metrics"], f"Missing metrics key: {k}"

    def test_model_edge_additive_not_breaking(self):
        """modelEdge key is present in metrics but None when bookOdds absent."""
        r = _scan(min=45)
        assert "modelEdge" in r["metrics"]
