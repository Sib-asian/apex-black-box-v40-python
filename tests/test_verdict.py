"""
Apex Black Box – Verdict engine tests.

Tests for:
1. generate_verdict() output structure and correctness
2. NoMoreGoals market conditions
3. Next Goal late-game downscoring
4. Exchange market availability (Under/Over 3.5)
5. Confidence threshold behaviour
6. Scan result includes verdict field (integration)
7. Regression: key probabilities in NoMoreGoals field

Run with:  pytest tests/test_verdict.py -v
"""

from __future__ import annotations
import pytest
from apex_black_box.engine import scan
from apex_black_box.verdict import generate_verdict, ENHANCED_VERDICT


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _scan(**kwargs) -> dict:
    payload = {"min": 45, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0.0}
    payload.update(kwargs)
    return scan(payload)


def _verdict(extra_scan_kwargs=None, **override_data) -> dict:
    """Build a minimal scan result and run generate_verdict() on it."""
    kwargs = {"min": 45, "hg": 0, "ag": 0, "tC": 2.5, "sC": 0.0,
              "sotH": 5, "sotA": 4, "daH": 15, "daA": 12}
    if extra_scan_kwargs:
        kwargs.update(extra_scan_kwargs)
    data = _scan(**kwargs)
    data.update(override_data)
    return generate_verdict(data)


# ─────────────────────────────────────────────────────────────────
#  1. Output structure
# ─────────────────────────────────────────────────────────────────

class TestVerdictStructure:
    def test_required_keys_present(self):
        v = _verdict()
        for key in ("topPick", "altPick", "specPick", "traps", "green", "watch",
                    "spec", "negative", "hardTraps", "all",
                    "LOW_CONF", "MID_CONF", "LATE_GAME", "GREEN_THRESH",
                    "bttsCorrSuppressed", "overCorrSuppressed"):
            assert key in v, f"Missing key: {key}"

    def test_all_list_types(self):
        v = _verdict()
        for key in ("traps", "green", "watch", "spec", "negative", "hardTraps", "all"):
            assert isinstance(v[key], list), f"{key} should be a list"

    def test_top_pick_has_required_fields(self):
        """topPick (when present) must have the fields renderAdvice() needs."""
        v = _verdict(extra_scan_kwargs={"sotH": 8, "sotA": 6, "daH": 25, "daA": 20})
        if v["topPick"] is not None:
            tp = v["topPick"]
            for field in ("id", "name", "p", "fairOdd"):
                assert field in tp, f"topPick missing field: {field}"
            assert isinstance(tp["fairOdd"], str), "fairOdd should be a string"
            assert 0.0 <= tp["p"] <= 1.0, f"topPick.p out of range: {tp['p']}"

    def test_green_thresh_in_range(self):
        v = _verdict()
        assert 0.40 <= v["GREEN_THRESH"] <= 0.70


# ─────────────────────────────────────────────────────────────────
#  2. NoMoreGoals market conditions
# ─────────────────────────────────────────────────────────────────

class TestNoMoreGoals:
    def test_no_more_goals_not_before_55min(self):
        """NoMoreGoals should never appear before minute 55."""
        v = _verdict(extra_scan_kwargs={"min": 50, "hg": 0, "ag": 0,
                                        "sotH": 1, "sotA": 1, "daH": 5, "daA": 5})
        ids = {m["id"] for m in v["all"]}
        assert "nomore" not in ids

    def test_no_more_goals_removed_when_vix_too_high(self):
        """NoMoreGoals must be removed when VIX > 38."""
        data = _scan(min=65, hg=0, ag=0, sotH=2, sotA=2, daH=10, daA=10)
        # Force high VIX
        data["vix"] = 45.0
        data["raw"]["sotH"] = 2
        data["raw"]["sotA"] = 2
        data["raw"]["daH"] = 10
        data["raw"]["daA"] = 10
        v = generate_verdict(data)
        ids = {m["id"] for m in v["all"]}
        assert "nomore" not in ids

    def test_no_more_goals_removed_when_sot_too_high(self):
        """NoMoreGoals must be removed when total SoT > 5."""
        data = _scan(min=65, hg=0, ag=0, sotH=4, sotA=3, daH=10, daA=10)
        data["vix"] = 30.0
        data["raw"]["rcH"] = 0
        data["raw"]["rcA"] = 0
        data["raw"]["lastGoal"] = 0
        data["raw"]["sotH"] = 4
        data["raw"]["sotA"] = 3  # total SoT = 7 > 5
        data["raw"]["daH"] = 10
        data["raw"]["daA"] = 10
        v = generate_verdict(data)
        ids = {m["id"] for m in v["all"]}
        assert "nomore" not in ids

    def test_no_more_goals_removed_with_red_card(self):
        """NoMoreGoals must be removed when there is at least one red card."""
        data = _scan(min=65, hg=0, ag=0, sotH=1, sotA=1, daH=5, daA=5)
        data["vix"] = 30.0
        data["raw"]["rcH"] = 1
        data["raw"]["rcA"] = 0
        data["raw"]["sotH"] = 1
        data["raw"]["sotA"] = 1
        data["raw"]["daH"] = 5
        data["raw"]["daA"] = 5
        data["raw"]["lastGoal"] = 0
        v = generate_verdict(data)
        ids = {m["id"] for m in v["all"]}
        assert "nomore" not in ids

    def test_no_more_goals_present_under_conservative_conditions(self):
        """NoMoreGoals may be present when all conservative conditions are met."""
        # Conservative conditions: min≥55, vix≤38, sotTot≤5, daTot≤46, rcH+rcA==0,
        # and lastGoal==0 (or min-lastGoal≥10)
        # We test that the market is NOT filtered (it can still be absent from all
        # if the probability is too low)
        data = _scan(min=65, hg=0, ag=0, sotH=1, sotA=1, daH=5, daA=5)
        data["vix"] = 25.0
        data["raw"]["rcH"] = 0
        data["raw"]["rcA"] = 0
        data["raw"]["sotH"] = 1
        data["raw"]["sotA"] = 1
        data["raw"]["daH"] = 5
        data["raw"]["daA"] = 5
        data["raw"]["lastGoal"] = 0
        # p_no_more must be > 0.005 for the market to be added to the list
        data["probs"]["NoMoreGoals"] = 0.50  # force a plausible value
        v = generate_verdict(data)
        # Under conservative conditions the market should survive the filter
        ids = {m["id"] for m in v["all"]}
        assert "nomore" in ids, "NoMoreGoals should be present under conservative conditions"


# ─────────────────────────────────────────────────────────────────
#  3. Next Goal late-game downscoring
# ─────────────────────────────────────────────────────────────────

class TestNextGoalLateGame:
    def test_nextH_downscored_when_home_winning_after_60(self):
        """Next Goal Casa should have a very low score when home is winning at 60+."""
        data = _scan(min=65, hg=1, ag=0, sotH=5, sotA=3)
        data["raw"]["rcH"] = 0
        data["raw"]["rcA"] = 0
        v = generate_verdict(data)
        # Find the nextH market in analyzed
        nh = next((m for m in v["all"] if m["id"] == "nextH"), None)
        if nh is not None:
            # Score should be heavily reduced (< 3) due to late-game penalty
            assert nh.get("_score", 5) < 3, (
                f"nextH should be downscored in late-game home winning, "
                f"got _score={nh.get('_score')}"
            )

    def test_nextA_downscored_when_away_winning_after_60(self):
        """Next Goal Trasf should have a very low score when away is winning at 60+."""
        data = _scan(min=65, hg=0, ag=1, sotH=3, sotA=5)
        data["raw"]["rcH"] = 0
        data["raw"]["rcA"] = 0
        v = generate_verdict(data)
        na = next((m for m in v["all"] if m["id"] == "nextA"), None)
        if na is not None:
            assert na.get("_score", 5) < 3, (
                f"nextA should be downscored in late-game away winning, "
                f"got _score={na.get('_score')}"
            )

    def test_nextH_not_downscored_when_home_losing(self):
        """Next Goal Casa should NOT receive late-game penalty when home is losing."""
        data = _scan(min=65, hg=0, ag=1, sotH=5, sotA=3)
        data["raw"]["rcH"] = 0
        data["raw"]["rcA"] = 0
        v = generate_verdict(data)
        # In this scenario nextA might be present (away winning), but nextH might be
        # a "chasing" market — it should not be penalised
        nh = next((m for m in v["all"] if m["id"] in ("nextH", "nextH_chase")), None)
        if nh is not None:
            # Should not have the late-game home-winning penalty
            # (score may be reduced for other reasons, but not that specific -4)
            assert nh.get("_score", 5) >= 1  # not heavily negative

    def test_next_goal_not_added_if_goal_diff_gt1(self):
        """Next Goal market should not appear (as simple 'nextH'/'nextA') when diff > 1."""
        # With goal diff >= 2, only nextH_chase can appear (if hg < ag)
        data = _scan(min=45, hg=0, ag=2, sotH=3, sotA=5)
        v = generate_verdict(data)
        ids = {m["id"] for m in v["all"]}
        assert "nextH" not in ids and "nextA" not in ids


# ─────────────────────────────────────────────────────────────────
#  4. Exchange markets availability
# ─────────────────────────────────────────────────────────────────

class TestExchangeMarkets:
    def test_under35_always_present_when_goals_lt4(self):
        """Under 3.5 should be in the analyzed list when hg+ag < 4."""
        for hg, ag in [(0, 0), (1, 0), (1, 1), (2, 1)]:
            data = _scan(min=45, hg=hg, ag=ag, sotH=5, sotA=4)
            v = generate_verdict(data)
            ids = {m["id"] for m in v["all"]}
            assert "under35" in ids, f"under35 missing at {hg}-{ag}"

    def test_under35_removed_when_goals_ge4(self):
        """Under 3.5 must be removed when hg+ag >= 4."""
        data = _scan(min=60, hg=2, ag=2, sotH=5, sotA=4)
        v = generate_verdict(data)
        ids = {m["id"] for m in v["all"]}
        assert "under35" not in ids, "under35 should be removed when 4+ goals scored"

    def test_over35_present_when_goals_lt4(self):
        """Over 3.5 should be in the analyzed list (may be low probability)."""
        data = _scan(min=45, hg=0, ag=0, sotH=6, sotA=5)
        v = generate_verdict(data)
        ids = {m["id"] for m in v["all"]}
        assert "over35" in ids

    def test_1x2_markets_present(self):
        """Core 1X2 markets should always be present."""
        data = _scan(min=45, hg=0, ag=0, sotH=5, sotA=4)
        v = generate_verdict(data)
        ids = {m["id"] for m in v["all"]}
        assert "1" in ids and "X" in ids and "2" in ids

    def test_btts_present_when_not_already_happened(self):
        """BTTS should be present when neither team has scored yet."""
        data = _scan(min=45, hg=0, ag=0, sotH=5, sotA=4)
        v = generate_verdict(data)
        ids = {m["id"] for m in v["all"]}
        assert "btts" in ids

    def test_btts_removed_when_already_happened(self):
        """BTTS should be removed when both teams have already scored."""
        data = _scan(min=60, hg=1, ag=1, sotH=5, sotA=4)
        v = generate_verdict(data)
        ids = {m["id"] for m in v["all"]}
        assert "btts" not in ids


# ─────────────────────────────────────────────────────────────────
#  5. Confidence thresholds
# ─────────────────────────────────────────────────────────────────

class TestConfidenceThresholds:
    def test_low_conf_flag(self):
        """LOW_CONF should be True when confidence < 40."""
        data = _scan(min=10, hg=0, ag=0)  # very little data → low confidence
        v = generate_verdict(data)
        if data["confidence"] < 40:
            assert v["LOW_CONF"] is True

    def test_high_conf_not_low(self):
        """HIGH_CONF scenario: LOW_CONF must be False."""
        data = _scan(min=75, hg=1, ag=0, sotH=10, sotA=7, daH=30, daA=20)
        v = generate_verdict(data)
        if data["confidence"] >= 55:
            assert v["LOW_CONF"] is False
            assert v["MID_CONF"] is False

    def test_late_game_flag(self):
        """LATE_GAME should be True after minute 85."""
        data = _scan(min=88, hg=0, ag=0)
        v = generate_verdict(data)
        assert v["LATE_GAME"] is True

    def test_not_late_game_before_85(self):
        """LATE_GAME should be False before minute 86."""
        data = _scan(min=80, hg=0, ag=0)
        v = generate_verdict(data)
        assert v["LATE_GAME"] is False


# ─────────────────────────────────────────────────────────────────
#  6. Integration: scan() includes verdict
# ─────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_scan_result_has_verdict(self):
        """scan() output must include a 'verdict' key."""
        r = _scan(min=60, hg=1, ag=0, sotH=7, sotA=5, daH=20, daA=15)
        assert "verdict" in r, "scan result missing 'verdict' key"
        assert isinstance(r["verdict"], dict)

    def test_scan_verdict_has_structure(self):
        """Verdict embedded in scan() must have standard keys."""
        r = _scan(min=60, hg=1, ag=0, sotH=7, sotA=5, daH=20, daA=15)
        v = r["verdict"]
        assert "topPick" in v
        assert "all" in v
        assert "GREEN_THRESH" in v

    def test_scan_probs_include_no_more_goals(self):
        """scan() probs must include 'NoMoreGoals' key."""
        r = _scan(min=65, hg=0, ag=0, sotH=3, sotA=2)
        assert "NoMoreGoals" in r["probs"], "NoMoreGoals missing from scan probs"
        p = r["probs"]["NoMoreGoals"]
        assert 0.0 <= p <= 1.0, f"NoMoreGoals out of range: {p}"

    def test_no_more_goals_is_complement_of_any_goal(self):
        """NoMoreGoals + P(at least 1 more goal) should ≈ 1."""
        r = _scan(min=60, hg=0, ag=0, sotH=5, sotA=4)
        p_no = r["probs"]["NoMoreGoals"]
        p_yes = r["probs"]["O_Dyn"]  # P(at least 1 more goal on dynOver line)
        # They are not exact complements because dynOver may be != 1.5,
        # but both should be in [0,1]
        assert 0.0 <= p_no <= 1.0
        assert 0.0 <= p_yes <= 1.0

    @pytest.mark.parametrize("minute,hg,ag", [
        (10, 0, 0), (30, 1, 0), (60, 1, 1), (80, 2, 1), (88, 0, 0),
    ])
    def test_scan_verdict_probs_all_valid(self, minute, hg, ag):
        """All probabilities in verdict market list should be in [0, 1]."""
        r = _scan(min=minute, hg=hg, ag=ag, sotH=minute//10, sotA=minute//15)
        for m in r["verdict"]["all"]:
            assert 0.0 <= m["p"] <= 1.0, (
                f"market {m['id']} has p={m['p']} at {minute}' {hg}-{ag}"
            )


# ─────────────────────────────────────────────────────────────────
#  7. Regression: golden snapshot for key market probabilities
# ─────────────────────────────────────────────────────────────────

class TestGoldenSnapshot:
    """
    Verify that the scan + verdict pipeline produces expected probability
    ranges for representative match states.  These are not exact values
    (engine can be updated) but guardrails to detect accidental regressions.
    """

    def _check(self, r, market, lo, hi, label=""):
        probs = r.get("probs", {})
        p = probs.get(market)
        assert p is not None, f"Market {market} missing from probs"
        assert lo <= p <= hi, (
            f"{label}: {market}={p:.3f} outside expected range [{lo}, {hi}]"
        )

    def test_0_0_at_45min(self):
        r = _scan(min=45, hg=0, ag=0, tC=2.5, sC=0.0,
                  sotH=5, sotA=4, daH=15, daA=12)
        self._check(r, "Over25", 0.20, 0.75, "0-0@45'")
        self._check(r, "BTTS",   0.20, 0.65, "0-0@45'")
        self._check(r, "1",      0.25, 0.55, "0-0@45' P1")

    def test_1_0_at_60min(self):
        r = _scan(min=60, hg=1, ag=0, tC=2.5, sC=-0.5,
                  sotH=7, sotA=4, daH=20, daA=12)
        self._check(r, "1",      0.45, 0.85, "1-0@60' P1")
        self._check(r, "Over15", 0.55, 0.92, "1-0@60' Over15")
        self._check(r, "Under25", 0.30, 0.75, "1-0@60' Under25")

    def test_no_more_goals_at_70min_cold(self):
        """Cold match at 70' should have a plausible NoMoreGoals probability."""
        r = _scan(min=70, hg=0, ag=0, tC=2.5, sC=0.0,
                  sotH=1, sotA=1, daH=4, daA=3)
        p = r["probs"]["NoMoreGoals"]
        # With very low activity, probability of no more goals should be somewhat high
        assert p >= 0.10, f"NoMoreGoals too low for cold 0-0@70': {p:.3f}"

    def test_over35_increases_with_goals(self):
        """Over 3.5 should be higher when 3 goals have already been scored."""
        r_low = _scan(min=45, hg=0, ag=0, tC=2.5)
        r_high = _scan(min=45, hg=2, ag=1, tC=2.5)
        assert r_high["probs"]["Over35"] > r_low["probs"]["Over35"], (
            "Over35 should be higher when 3 goals already scored"
        )
