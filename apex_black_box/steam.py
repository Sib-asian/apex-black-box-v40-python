"""
Apex Black Box V40 - Steam (Line-Movement) Analyser
=====================================================
Detects sharp-money steam moves in spread and totals markets and converts
those signals into lambda modifiers and pre-match advice strings.

Steam signals
-------------
- **Reverse line movement (RLM)**: public betting % and line move in opposite
  directions, suggesting sharp action.
- **Total move direction**: line rising = sharps betting OVER (inflate lambdas);
  line falling = sharps betting UNDER (deflate lambdas).
- **Spread move direction**: line moving towards home = sharps on away (or vice versa).

All modifier outputs are multiplicative factors applied to the Poisson lambdas.
"""

from __future__ import annotations

from typing import Any, Dict, List


class SteamAnalyzer:
    """Analyse pre-match betting-line movements for sharp-money signals.

    Parameters
    ----------
    spread_open: Opening spread / Asian handicap line.
    spread_curr: Current spread / Asian handicap line.
    total_open:  Opening over/under goals line.
    total_curr:  Current over/under goals line.
    """

    # Thresholds for classifying movement magnitude
    _SMALL_MOVE: float = 0.25   # Below this: negligible
    _MEDIUM_MOVE: float = 0.75  # Below this: moderate (>= 0.75 is large)
    # (anything >= _MEDIUM_MOVE is classified as large)

    # Lambda modifier magnitudes
    _MOD_SMALL: float = 0.03
    _MOD_MEDIUM: float = 0.07
    _MOD_LARGE: float = 0.12

    def __init__(
        self,
        spread_open: float,
        spread_curr: float,
        total_open: float,
        total_curr: float,
    ) -> None:
        self.spread_open = spread_open
        self.spread_curr = spread_curr
        self.total_open = total_open
        self.total_curr = total_curr

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _magnitude_label(move: float) -> str:
        """Classify the absolute size of a line move."""
        abs_move = abs(move)
        if abs_move < SteamAnalyzer._SMALL_MOVE:
            return "negligible"
        if abs_move < SteamAnalyzer._MEDIUM_MOVE:
            return "small"
        return "large"

    @staticmethod
    def _modifier_for_magnitude(abs_move: float) -> float:
        """Return the adjustment magnitude corresponding to a move size."""
        if abs_move < SteamAnalyzer._SMALL_MOVE:
            return 0.0
        if abs_move < SteamAnalyzer._MEDIUM_MOVE:
            return SteamAnalyzer._MOD_SMALL
        if abs_move < 1.5:
            return SteamAnalyzer._MOD_MEDIUM
        return SteamAnalyzer._MOD_LARGE

    # ------------------------------------------------------------------
    # Public detection methods
    # ------------------------------------------------------------------

    def detect_spread_movement(self) -> Dict[str, Any]:
        """Analyse spread line movement.

        Convention: negative spread favours home (e.g. -1.5 = home -1.5).
        - Spread moving more negative → sharps on HOME.
        - Spread moving less negative / more positive → sharps on AWAY.

        Returns
        -------
        Dict[str, Any]:
            - ``open``       Opening spread.
            - ``current``    Current spread.
            - ``move``       Raw movement (curr - open).
            - ``magnitude``  'negligible' | 'small' | 'large'.
            - ``direction``  'home_favoured' | 'away_favoured' | 'neutral'.
            - ``signal``     Plain-English signal string.
        """
        move = self.spread_curr - self.spread_open
        magnitude = self._magnitude_label(move)

        if abs(move) < self._SMALL_MOVE:
            direction = "neutral"
            signal = "Spread stable – no significant sharp action detected."
        elif move < 0:
            direction = "home_favoured"
            signal = (
                f"Spread moved {abs(move):.2f} units in favour of HOME "
                f"(open {self.spread_open:+.2f} → current {self.spread_curr:+.2f}). "
                "Sharp money likely on HOME."
            )
        else:
            direction = "away_favoured"
            signal = (
                f"Spread moved {move:.2f} units in favour of AWAY "
                f"(open {self.spread_open:+.2f} → current {self.spread_curr:+.2f}). "
                "Sharp money likely on AWAY."
            )

        return {
            "open": self.spread_open,
            "current": self.spread_curr,
            "move": round(move, 3),
            "magnitude": magnitude,
            "direction": direction,
            "signal": signal,
        }

    def detect_total_movement(self) -> Dict[str, Any]:
        """Analyse totals (O/U goals) line movement.

        - Line rising → sharps betting OVER (inflate expected goals).
        - Line falling → sharps betting UNDER (deflate expected goals).

        Returns
        -------
        Dict[str, Any]:
            - ``open``       Opening total.
            - ``current``    Current total.
            - ``move``       Raw movement (curr - open).
            - ``magnitude``  'negligible' | 'small' | 'large'.
            - ``direction``  'over_steam' | 'under_steam' | 'neutral'.
            - ``signal``     Plain-English signal string.
        """
        move = self.total_curr - self.total_open
        magnitude = self._magnitude_label(move)

        if abs(move) < self._SMALL_MOVE:
            direction = "neutral"
            signal = "Totals line stable – no significant sharp action detected."
        elif move > 0:
            direction = "over_steam"
            signal = (
                f"Total rose {move:.2f} goals "
                f"(open {self.total_open:.1f} → current {self.total_curr:.1f}). "
                "OVER steam detected – expect higher scoring."
            )
        else:
            direction = "under_steam"
            signal = (
                f"Total dropped {abs(move):.2f} goals "
                f"(open {self.total_open:.1f} → current {self.total_curr:.1f}). "
                "UNDER steam detected – expect lower scoring."
            )

        return {
            "open": self.total_open,
            "current": self.total_curr,
            "move": round(move, 3),
            "magnitude": magnitude,
            "direction": direction,
            "signal": signal,
        }

    def get_lambda_modifiers(self) -> Dict[str, float]:
        """Translate steam signals into multiplicative lambda adjustments.

        Logic
        -----
        1. **Total movement** → applied symmetrically to both home and away.
           - OVER steam: both lambdas +modifier.
           - UNDER steam: both lambdas -modifier.
        2. **Spread movement** → applied asymmetrically.
           - Home-favoured steam: home lambda +modifier, away lambda -modifier/2.
           - Away-favoured steam: away lambda +modifier, home lambda -modifier/2.

        All adjustments are additive deltas to a base factor of 1.0, then clipped
        to ensure neither modifier drops below 0.80 (avoid unrealistic suppression).

        Returns
        -------
        Dict[str, float]:
            - ``home_mod``: Multiplicative factor for home lambda.
            - ``away_mod``: Multiplicative factor for away lambda.
        """
        total_move = self.total_curr - self.total_open
        spread_move = self.spread_curr - self.spread_open

        total_adj = self._modifier_for_magnitude(abs(total_move))
        spread_adj = self._modifier_for_magnitude(abs(spread_move))

        home_mod = 1.0
        away_mod = 1.0

        # Total steam
        if total_move > self._SMALL_MOVE:           # OVER steam
            home_mod += total_adj
            away_mod += total_adj
        elif total_move < -self._SMALL_MOVE:         # UNDER steam
            home_mod -= total_adj
            away_mod -= total_adj

        # Spread steam
        if spread_move < -self._SMALL_MOVE:          # Favours HOME
            home_mod += spread_adj
            away_mod -= spread_adj * 0.5
        elif spread_move > self._SMALL_MOVE:         # Favours AWAY
            away_mod += spread_adj
            home_mod -= spread_adj * 0.5

        # Safety floor
        home_mod = max(home_mod, 0.80)
        away_mod = max(away_mod, 0.80)

        return {
            "home_mod": round(home_mod, 4),
            "away_mod": round(away_mod, 4),
        }

    def generate_advice(self) -> List[str]:
        """Generate a list of actionable pre-match advice strings.

        Returns
        -------
        List[str]: Advice items based on detected steam moves.
        """
        advice: List[str] = []

        spread_info = self.detect_spread_movement()
        total_info = self.detect_total_movement()

        # Spread advice
        if spread_info["magnitude"] != "negligible":
            advice.append(f"📊 Spread steam [{spread_info['direction']}]: {spread_info['signal']}")

            if spread_info["direction"] == "home_favoured":
                advice.append("✅ Consider HOME markets (1X2, DNB_H, Asian Handicap Home).")
            elif spread_info["direction"] == "away_favoured":
                advice.append("✅ Consider AWAY markets (1X2, DNB_A, Asian Handicap Away).")

        # Total advice
        if total_info["magnitude"] != "negligible":
            advice.append(f"⚽ Total steam [{total_info['direction']}]: {total_info['signal']}")

            if total_info["direction"] == "over_steam":
                advice.append("✅ Consider OVER 2.5 goals and BTTS markets.")
            elif total_info["direction"] == "under_steam":
                advice.append("✅ Consider UNDER 2.5 goals markets.")

        # Combined OVER + HOME steam (e.g. heavy-scoring home favourite)
        if (
            total_info["direction"] == "over_steam"
            and spread_info["direction"] == "home_favoured"
            and spread_info["magnitude"] != "negligible"
            and total_info["magnitude"] != "negligible"
        ):
            advice.append(
                "🔥 Double steam: OVER + HOME. High-scoring home win scenario. "
                "Asian Handicap Home / BTTS & Over may offer value."
            )

        # Contradictory signals warning
        if (
            total_info["direction"] == "over_steam"
            and spread_info["direction"] == "away_favoured"
            and spread_info["magnitude"] != "negligible"
        ):
            advice.append(
                "⚠️ Contradictory signals: OVER steam present while spread favours the away "
                "side (underdog). Fade caution — wait for confirmation before betting."
            )

        if not advice:
            advice.append("ℹ️ No significant steam detected. Proceed with standard pre-match analysis.")

        return advice

    def summary(self) -> Dict[str, Any]:
        """Full steam analysis summary.

        Returns
        -------
        Dict[str, Any]: Combined spread and total analysis with modifiers.
        """
        spread_info = self.detect_spread_movement()
        total_info = self.detect_total_movement()
        mods = self.get_lambda_modifiers()
        adv = self.generate_advice()

        return {
            "spread": spread_info,
            "total": total_info,
            "lambda_modifiers": mods,
            "advice": adv,
        }
