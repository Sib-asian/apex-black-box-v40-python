"""
Tests for apex_black_box.steam and engine.build_steam_advice().
"""
from __future__ import annotations

import pytest

from apex_black_box.steam import SteamAnalyzer
from apex_black_box.engine import analyze_steam, build_steam_advice


class TestSteamAnalyzerDetectMovement:
    def _make_analyzer(self, s_o=0.0, s_c=0.0, t_o=2.5, t_c=2.5):
        return SteamAnalyzer({"sO": s_o, "sC": s_c, "tO": t_o, "tC": t_c})

    def test_detect_movement_no_spread(self):
        """No spread/total movement → level 'none' or 'weak'."""
        sa = self._make_analyzer(s_o=0.0, s_c=0.0, t_o=2.5, t_c=2.5)
        result = sa.detect_movement()
        assert isinstance(result, dict)
        assert result.get("level") in ("none", "weak")

    def test_detect_movement_strong_spread(self):
        """Large spread movement (>0.5) should produce 'weak' or 'strong' level."""
        sa = self._make_analyzer(s_o=0.0, s_c=0.75, t_o=2.5, t_c=2.5)
        # s_o=0, s_c=0.75 means a big spread change toward home
        result = sa.detect_movement()
        assert result.get("level") in ("weak", "strong", "reverse")
        # adS should be ~0.75
        assert result.get("adS", 0.0) >= 0.5

    def test_detect_movement_returns_required_keys(self):
        """detect_movement must return all required keys."""
        sa = self._make_analyzer(s_o=-0.5, s_c=0.0, t_o=2.75, t_c=2.5)
        result = sa.detect_movement()
        required = {"level", "dSpread", "dTotal", "adS", "adT", "signals", "lambdaMod"}
        for key in required:
            assert key in result, f"Missing key: {key}"


class TestSteamAnalyzerGenerateAdvice:
    def test_generate_advice_returns_list(self):
        """generate_advice must always return a list."""
        sa = SteamAnalyzer({"sO": 0.0, "sC": -0.5, "tO": 2.5, "tC": 2.75})
        result = sa.generate_advice()
        assert isinstance(result, list)

    def test_generate_advice_nonempty_on_signal(self):
        """Strong spread + total movement should yield at least one advice."""
        sa = SteamAnalyzer({"sO": 0.0, "sC": -1.0, "tO": 2.5, "tC": 3.0})
        # big spread + total movement — should generate advice
        result = sa.generate_advice()
        assert len(result) >= 1, "Expected at least one advice on strong steam signal"

    def test_generate_advice_empty_on_no_signal(self):
        """No steam movement should yield empty or weak-only advice."""
        sa = SteamAnalyzer({"sO": 0.0, "sC": 0.0, "tO": 2.5, "tC": 2.5})
        result = sa.generate_advice()
        # May have a weak-signal warning, but no actionable picks
        for adv in result:
            assert adv.get("strength") in ("DEBOLE", "debole", None), (
                f"Expected only weak advice on no signal, got strength={adv.get('strength')}"
            )


class TestAnalyzeSteam:
    def test_reverse_scenario(self):
        """When spread and total move in opposite directions → level may be 'reverse'."""
        result = analyze_steam(s_o=0.0, s_c=-1.0, t_o=2.5, t_c=2.0)
        # spread goes home (negative), total drops → reverse
        assert result["level"] in ("reverse", "weak", "strong")

    def test_signals_list(self):
        """analyze_steam must always return a 'signals' list."""
        result = analyze_steam(s_o=0.0, s_c=0.0, t_o=2.5, t_c=2.5)
        assert isinstance(result["signals"], list)

    def test_lambda_mod_structure(self):
        """lambdaMod must have 'h' and 'a' keys both > 0."""
        result = analyze_steam(s_o=-0.25, s_c=0.25, t_o=2.5, t_c=2.5)
        lm = result["lambdaMod"]
        assert "h" in lm and "a" in lm
        assert lm["h"] > 0 and lm["a"] > 0


class TestBuildSteamAdvice:
    """Tests for Fix 1 — JS-compatible fields in build_steam_advice()."""

    def _get_advice_for_strong_home(self):
        """Simulate strong home steam movement."""
        steam = analyze_steam(s_o=0.0, s_c=-1.2, t_o=2.5, t_c=2.8)
        return build_steam_advice(steam, t_c=2.8, s_c=-1.2)

    def _get_advice_for_reverse(self):
        """Simulate reverse steam (spread down, total down)."""
        steam = analyze_steam(s_o=0.0, s_c=-0.8, t_o=2.5, t_c=2.0)
        return build_steam_advice(steam, t_c=2.0, s_c=-0.8)

    def test_advice_contains_js_fields(self):
        """Every advice dict must have the required JS-compatible fields."""
        advice = self._get_advice_for_strong_home()
        required_js_fields = {"icon", "cat", "pick", "note", "fairOdd", "prob", "strength"}
        for adv in advice:
            missing = required_js_fields - set(adv.keys())
            assert not missing, f"Advice missing JS fields: {missing}\nadv={adv}"

    def test_backward_compat_fields_present(self):
        """market, reason, confidence_req must still be present for backward compat."""
        advice = self._get_advice_for_strong_home()
        for adv in advice:
            assert "market" in adv, f"Missing 'market' in {adv}"
            assert "reason" in adv, f"Missing 'reason' in {adv}"
            assert "confidence_req" in adv, f"Missing 'confidence_req' in {adv}"

    def test_strength_is_uppercase(self):
        """strength field must be UPPERCASE (FORTE, MEDIO, DEBOLE, etc.)."""
        advice = self._get_advice_for_strong_home()
        for adv in advice:
            s = adv.get("strength", "")
            assert s == s.upper(), (
                f"strength '{s}' is not UPPERCASE in advice: {adv}"
            )

    def test_pick_equals_market(self):
        """pick must equal market for JS renderAdvice compatibility."""
        advice = self._get_advice_for_strong_home()
        for adv in advice:
            assert adv["pick"] == adv["market"], (
                f"pick '{adv['pick']}' != market '{adv['market']}'"
            )

    def test_note_equals_reason(self):
        """note must equal reason for JS renderAdvice compatibility."""
        advice = self._get_advice_for_strong_home()
        for adv in advice:
            assert adv["note"] == adv["reason"], (
                f"note != reason in advice: {adv}"
            )

    def test_fair_odd_format(self):
        """fairOdd must be either a decimal string like '1.85' or '—'."""
        advice = self._get_advice_for_strong_home()
        for adv in advice:
            fo = adv.get("fairOdd", "")
            assert isinstance(fo, str), f"fairOdd must be str, got {type(fo)}"
            if fo != "—":
                # should be parseable as a float
                val = float(fo)
                assert val >= 1.0, f"fairOdd={fo} is below 1.0"

    def test_prob_is_float_in_range(self):
        """prob must be a float in [0, 1]."""
        advice = self._get_advice_for_strong_home()
        for adv in advice:
            p = adv.get("prob")
            assert isinstance(p, float), f"prob must be float, got {type(p)}"
            assert 0.0 <= p <= 1.0, f"prob={p} out of [0,1]"

    def test_icon_nonempty_string(self):
        """icon must be a non-empty string."""
        advice = self._get_advice_for_strong_home()
        for adv in advice:
            icon = adv.get("icon", "")
            assert isinstance(icon, str) and len(icon) > 0, (
                f"icon must be non-empty string, got {repr(icon)}"
            )

    def test_cat_nonempty_uppercase_string(self):
        """cat must be a non-empty UPPERCASE string."""
        advice = self._get_advice_for_strong_home()
        for adv in advice:
            cat = adv.get("cat", "")
            assert isinstance(cat, str) and len(cat) > 0, f"cat empty in {adv}"
            assert cat == cat.upper(), f"cat '{cat}' not UPPERCASE"

    def test_reverse_steam_warning_advice(self):
        """Reverse steam advice should have ATTENZIONE cat and warning icon."""
        advice = self._get_advice_for_reverse()
        # May or may not produce reverse advice depending on exact level
        for adv in advice:
            if adv.get("cat") == "ATTENZIONE":
                assert "⚠️" in adv.get("icon", ""), (
                    f"Expected ⚠️ icon for ATTENZIONE advice: {adv}"
                )

    def test_nonempty_advice_on_strong_signal(self):
        """Strong steam signal must generate at least one advice."""
        advice = self._get_advice_for_strong_home()
        assert len(advice) >= 1, "Expected advice on strong steam"

    def test_btts_advice_fields(self):
        """BTTS advice should have correct icon and cat."""
        # High lambda mods and rising total trigger BTTS Sì
        steam = {"level": "strong", "dSpread": 0.0, "dTotal": 0.5,
                 "adS": 0.1, "adT": 0.5,
                 "lambdaMod": {"h": 1.05, "a": 1.05},
                 "hasSpreadStrong": False}
        advice = build_steam_advice(steam, t_c=3.0, s_c=0.0)
        btts_advs = [a for a in advice if "BTTS Sì" in a.get("market", "")]
        if btts_advs:
            adv = btts_advs[0]
            assert adv["icon"] == "⚽"
            assert adv["cat"] == "BTTS"
