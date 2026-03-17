"""
Apex Black Box V40 - Core Engine
=================================
Main probabilistic engine for in-play football match analysis.
Uses Poisson modelling with Dixon-Coles correction, xG blending,
and situational modifiers to produce calibrated match-outcome probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MatchScore:
    """Current match score and red-card state.

    Attributes
    ----------
    min:       Elapsed minutes (0-120, including added time).
    rec:       Added / recovery minutes shown on the board.
    hg:        Home goals scored.
    ag:        Away goals scored.
    last_goal: Minute the most recent goal was scored (-1 if none yet).
    red_h:     Red cards received by the home side (0-2).
    red_a:     Red cards received by the away side (0-2).
    """

    min: int
    rec: int
    hg: int
    ag: int
    last_goal: int = -1
    red_h: int = 0
    red_a: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.min <= 120):
            raise ValueError(f"Minute must be 0-120, got {self.min}")
        if self.hg < 0 or self.ag < 0:
            raise ValueError("Goals cannot be negative")
        if not (0 <= self.red_h <= 2) or not (0 <= self.red_a <= 2):
            raise ValueError("Red cards per team must be 0-2")


@dataclass
class MatchStats:
    """In-play stats snapshot for both teams.

    Attributes
    ----------
    sot_h, sot_a:   Shots on target (home / away).
    mis_h, mis_a:   Missed shots / shots off target (home / away).
    cor_h, cor_a:   Corners earned (home / away).
    da_h, da_a:     Dangerous attacks (home / away).
    poss_h, poss_a: Ball possession percentage (0-100, sum ≈ 100).
    """

    sot_h: int
    mis_h: int
    cor_h: int
    da_h: int
    poss_h: float
    sot_a: int
    mis_a: int
    cor_a: int
    da_a: int
    poss_a: float

    def __post_init__(self) -> None:
        for name, val in [
            ("sot_h", self.sot_h), ("mis_h", self.mis_h),
            ("cor_h", self.cor_h), ("da_h", self.da_h),
            ("sot_a", self.sot_a), ("mis_a", self.mis_a),
            ("cor_a", self.cor_a), ("da_a", self.da_a),
        ]:
            if val < 0:
                raise ValueError(f"{name} cannot be negative")
        if not (0.0 <= self.poss_h <= 100.0):
            raise ValueError(f"poss_h must be 0-100, got {self.poss_h}")
        if not (0.0 <= self.poss_a <= 100.0):
            raise ValueError(f"poss_a must be 0-100, got {self.poss_a}")


@dataclass
class PreMatchData:
    """Pre-match market data and Bayesian priors.

    Attributes
    ----------
    spread_open:  Opening spread / handicap line.
    spread_curr:  Current spread / handicap line.
    total_open:   Opening totals line (O/U goals).
    total_curr:   Current totals line.
    is_knockout:  True when the match is a knockout fixture (no draw allowed).
    prior_h:      Bayesian prior for home expected goals rate.
    prior_a:      Bayesian prior for away expected goals rate.
    prior_draw:   Bayesian prior draw weight (used in blending only).
    """

    spread_open: float
    spread_curr: float
    total_open: float
    total_curr: float
    is_knockout: bool
    prior_h: float = 1.5
    prior_a: float = 1.2
    prior_draw: float = 0.3


@dataclass
class OracleOutput:
    """Full output bundle from a single oracle scan.

    Attributes
    ----------
    probs:            Market probability dict (keys: 1, X, 2, Over25, Under25,
                      BTTS, DNB_H, DNB_A).
    confidence:       Overall confidence score (0-100).
    vix:              Volatility index (0-100).
    lambdas:          Poisson rate parameters used ('home', 'away').
    alerts:           Human-readable alert strings.
    raw_score_matrix: Optional 2-D goal-probability matrix (home x away).
    """

    probs: Dict[str, float]
    confidence: float
    vix: float
    lambdas: Dict[str, float]
    alerts: List[str]
    raw_score_matrix: Optional[List[List[float]]] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OracleEngineV40:
    """Apex Black Box V40 probabilistic in-play oracle.

    Parameters
    ----------
    pre:   Pre-match data including priors and market lines.
    score: Current match score and red-card state.
    stats: In-play statistics snapshot.
    """

    # Scaling constants
    _BLEND_K: float = 0.10          # sigmoid steepness for prior blend
    _BLEND_MID: float = 45.0        # sigmoid midpoint (minutes)
    _MATCH_DURATION_MINUTES: float = 90.0  # standard match duration
    _XG_BASE_RATE: float = 0.08     # base conversion rate (shots → xG)
    _DA_SOFT_CAP: int = 35          # DA cap before decay kicks in
    _DA_DECAY: float = 0.15         # additional DA weight above soft cap
    _POST_GOAL_SHOCK_WINDOW: int = 6 # minutes after a goal where rates dip
    _POST_GOAL_SHOCK_FACTOR: float = 0.85
    _GARBAGE_TIME_THRESHOLD: int = 75  # minutes after which goal diff matters
    _GARBAGE_TIME_DIFF: int = 3     # goal difference that triggers garbage-time
    _RED_CARD_PENALTY: float = 0.12 # per-card goal-rate reduction for the team

    def __init__(
        self,
        pre: PreMatchData,
        score: MatchScore,
        stats: MatchStats,
    ) -> None:
        self.pre = pre
        self.score = score
        self.stats = stats

    # ------------------------------------------------------------------
    # Static / helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def poisson_pmf(lam: float, k: int) -> float:
        """Poisson PMF computed in log-space for numerical stability.

        Parameters
        ----------
        lam: Rate parameter (must be > 0).
        k:   Number of events (must be >= 0).

        Returns
        -------
        float: P(X = k) under Poisson(lam).
        """
        if lam <= 0.0:
            return 1.0 if k == 0 else 0.0
        if k < 0:
            return 0.0
        log_pmf = k * math.log(lam) - lam - math.lgamma(k + 1)
        if log_pmf < -700:
            return 0.0
        return math.exp(log_pmf)

    @staticmethod
    def poisson_joint_matrix(
        lam_h: float,
        lam_a: float,
        max_goals: int = 8,
    ) -> np.ndarray:
        """Build a (max_goals+1) x (max_goals+1) joint probability matrix.

        Applies Dixon-Coles low-score correction to cells (0,0), (1,0),
        (0,1), (1,1) to account for the known under-representation of
        0-0 and 1-1 draws in Poisson models.

        Parameters
        ----------
        lam_h:     Home team expected goals remaining.
        lam_a:     Away team expected goals remaining.
        max_goals: Maximum goals to consider per side.

        Returns
        -------
        np.ndarray: Joint probability matrix [home_goals, away_goals].
        """
        size = max_goals + 1
        matrix = np.zeros((size, size), dtype=float)

        # Populate independent Poisson probabilities
        h_pmf = [OracleEngineV40.poisson_pmf(lam_h, k) for k in range(size)]
        a_pmf = [OracleEngineV40.poisson_pmf(lam_a, k) for k in range(size)]

        for i in range(size):
            for j in range(size):
                matrix[i, j] = h_pmf[i] * a_pmf[j]

        # Dixon-Coles correction
        # rho < 0 shifts probability from (0,0) and (1,1) towards (1,0) / (0,1)
        # time_decay: correction weakens as the match nears 90 min
        # dc_scale:   correction also weakens when expected goals are high
        # (The correction is not needed in high-scoring environments)
        return matrix  # DC applied by caller via _apply_dc_correction

    def _apply_dc_correction(
        self,
        matrix: np.ndarray,
        lam_h: float,
        lam_a: float,
    ) -> np.ndarray:
        """Apply Dixon-Coles correction in-place and re-normalise.

        rho = -0.10 * dc_scale * time_decay
          dc_scale   = 1 / (1 + max(0, lam_h*lam_a - 1) * 0.35)
          time_decay = max(0.05, 1 - min/90)
        """
        minute = self.score.min
        time_decay = max(0.05, 1.0 - minute / self._MATCH_DURATION_MINUTES)
        dc_scale = 1.0 / (1.0 + max(0.0, lam_h * lam_a - 1.0) * 0.35)
        rho = -0.10 * dc_scale * time_decay

        # Correction factors for low-score cells
        p00 = matrix[0, 0]
        p10 = matrix[1, 0] if matrix.shape[0] > 1 else 0.0
        p01 = matrix[0, 1] if matrix.shape[1] > 1 else 0.0
        p11 = (
            matrix[1, 1] if matrix.shape[0] > 1 and matrix.shape[1] > 1 else 0.0
        )

        # DC adjustment: τ(i,j) multiplier
        lh = max(lam_h, 1e-9)
        la = max(lam_a, 1e-9)

        tau_00 = 1.0 - lh * la * rho
        tau_10 = 1.0 + la * rho
        tau_01 = 1.0 + lh * rho
        tau_11 = 1.0 - rho

        matrix[0, 0] = p00 * tau_00
        if matrix.shape[0] > 1:
            matrix[1, 0] = p10 * tau_10
        if matrix.shape[1] > 1:
            matrix[0, 1] = p01 * tau_01
        if matrix.shape[0] > 1 and matrix.shape[1] > 1:
            matrix[1, 1] = p11 * tau_11

        # Guard against any negatives introduced by extreme rho values
        matrix = np.clip(matrix, 0.0, None)

        total = matrix.sum()
        if total > 0.0:
            matrix /= total

        return matrix

    @staticmethod
    def calc_xg(
        sot: int,
        mis: int,
        da: int,
        conv_rate: float = 0.08,
    ) -> float:
        """Estimate expected goals from shot and DA data.

        Shots on target contribute at ``conv_rate`` per shot.
        Missed shots contribute at ``conv_rate * 0.35``.
        Dangerous attacks above the soft cap of 35 contribute at
        ``DA_DECAY`` (0.15 × normal weight) to reflect diminishing returns.

        Parameters
        ----------
        sot:       Shots on target.
        mis:       Shots off target / missed.
        da:        Dangerous attacks.
        conv_rate: Base conversion rate for shots on target.

        Returns
        -------
        float: xG estimate (non-negative).
        """
        xg = sot * conv_rate + mis * conv_rate * 0.35

        da_cap = OracleEngineV40._DA_SOFT_CAP
        if da <= da_cap:
            xg += da * conv_rate * 0.15
        else:
            xg += da_cap * conv_rate * 0.15
            excess = da - da_cap
            xg += excess * conv_rate * 0.15 * OracleEngineV40._DA_DECAY

        return max(0.0, xg)

    def blend_lambda(
        self,
        xg_rate: float,
        prior: float,
        minute: int,
    ) -> float:
        """Blend xG-derived rate with Bayesian prior using sigmoid weighting.

        Weight w = 1 / (1 + exp(-k * (minute - mid)))
          w → 0 early in the match (prior dominates)
          w → 1 late in the match (xG dominates)

        Parameters
        ----------
        xg_rate: Goals per 90 min implied by in-game xG stats.
        prior:   Pre-match prior expected goals (from PreMatchData).
        minute:  Elapsed match minutes.

        Returns
        -------
        float: Blended lambda (goals remaining, scaled to remaining time).
        """
        w = 1.0 / (1.0 + math.exp(-self._BLEND_K * (minute - self._BLEND_MID)))
        blended = w * xg_rate + (1.0 - w) * prior

        # Scale to remaining minutes
        minutes_remaining = max(1.0, self._MATCH_DURATION_MINUTES - minute + self.score.rec)
        return blended * (minutes_remaining / self._MATCH_DURATION_MINUTES)

    def apply_score_effects(
        self,
        lam_h: float,
        lam_a: float,
    ) -> Tuple[float, float]:
        """Apply situational modifiers to raw lambda estimates.

        Modifiers applied (multiplicative):

        1. **Red cards** – reduce goal rate by ``RED_CARD_PENALTY`` per card.
        2. **Post-goal shock** – both rates drop by ``POST_GOAL_SHOCK_FACTOR``
           in the ``POST_GOAL_SHOCK_WINDOW`` minutes after a goal.
        3. **Garbage time** – when the goal difference ≥ ``GARBAGE_TIME_DIFF``
           after minute ``GARBAGE_TIME_THRESHOLD``, the leading team's rate
           drops (they slow down) and the trailing team's rate rises slightly
           (chasing).

        Parameters
        ----------
        lam_h: Raw home lambda.
        lam_a: Raw away lambda.

        Returns
        -------
        Tuple[float, float]: (adjusted_lam_h, adjusted_lam_a).
        """
        minute = self.score.min

        # 1. Red-card suppression
        lam_h *= (1.0 - self._RED_CARD_PENALTY * self.score.red_h)
        lam_a *= (1.0 - self._RED_CARD_PENALTY * self.score.red_a)

        # 2. Post-goal shock
        if (
            self.score.last_goal >= 0
            and (minute - self.score.last_goal) <= self._POST_GOAL_SHOCK_WINDOW
        ):
            lam_h *= self._POST_GOAL_SHOCK_FACTOR
            lam_a *= self._POST_GOAL_SHOCK_FACTOR

        # 3. Garbage time
        if minute >= self._GARBAGE_TIME_THRESHOLD:
            goal_diff = self.score.hg - self.score.ag
            if abs(goal_diff) >= self._GARBAGE_TIME_DIFF:
                if goal_diff > 0:
                    # Home winning comfortably – slow down / away chasing
                    lam_h *= 0.80
                    lam_a *= 1.10
                else:
                    # Away winning comfortably
                    lam_a *= 0.80
                    lam_h *= 1.10

        return max(lam_h, 1e-9), max(lam_a, 1e-9)

    def calc_vix(
        self,
        lam_h: float,
        lam_a: float,
        confidence: float,
    ) -> float:
        """Compute the volatility index (0-100).

        VIX rises when:
        - Expected goals are high (more uncertain outcomes).
        - Confidence is low.
        - Game is early (less information).
        - Match is close (near-equal lambdas).

        Parameters
        ----------
        lam_h:      Home lambda.
        lam_a:      Away lambda.
        confidence: Current confidence score (0-100).

        Returns
        -------
        float: VIX score clamped to [0, 100].
        """
        # Component 1: total expected goals → higher goals ↔ higher volatility
        total_lam = lam_h + lam_a
        lam_component = min(100.0, total_lam * 15.0)

        # Component 2: closeness of lambdas (max when equal)
        if total_lam > 0:
            balance = 1.0 - abs(lam_h - lam_a) / total_lam
        else:
            balance = 1.0
        balance_component = balance * 30.0

        # Component 3: low confidence → high VIX
        confidence_component = (1.0 - confidence / 100.0) * 40.0

        # Component 4: early game penalty
        minute = self.score.min
        time_component = max(0.0, (45.0 - minute) / 45.0) * 20.0

        vix = lam_component + balance_component + confidence_component + time_component
        return float(np.clip(vix, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _minutes_remaining(self) -> float:
        return max(1.0, self._MATCH_DURATION_MINUTES - self.score.min + self.score.rec)

    def _compute_xg_rates(self) -> Tuple[float, float]:
        """Convert in-game stats to per-90-min xG rates."""
        elapsed = max(1.0, float(self.score.min))

        xg_h_elapsed = self.calc_xg(
            self.stats.sot_h, self.stats.mis_h, self.stats.da_h
        )
        xg_a_elapsed = self.calc_xg(
            self.stats.sot_a, self.stats.mis_a, self.stats.da_a
        )

        # Annualise to per-90 rate
        rate_h = xg_h_elapsed / elapsed * self._MATCH_DURATION_MINUTES
        rate_a = xg_a_elapsed / elapsed * self._MATCH_DURATION_MINUTES

        return rate_h, rate_a

    def _compute_market_probs(
        self,
        matrix: np.ndarray,
    ) -> Dict[str, float]:
        """Derive market probabilities from the score matrix.

        The matrix represents the probability distribution of *additional*
        goals from the current scoreline.  We need to add the existing
        goals to determine FT result, total goals, etc.
        """
        size = matrix.shape[0]
        hg_curr = self.score.hg
        ag_curr = self.score.ag

        p1 = p_x = p2 = 0.0
        p_over25 = p_under25 = 0.0
        p_btts = 0.0
        p_dnb_h = p_dnb_a = 0.0

        for i in range(size):
            for j in range(size):
                prob = matrix[i, j]
                ft_h = hg_curr + i
                ft_a = ag_curr + j

                # 1X2
                if ft_h > ft_a:
                    p1 += prob
                elif ft_h == ft_a:
                    p_x += prob
                else:
                    p2 += prob

                # Over/Under 2.5
                total = ft_h + ft_a
                if total > 2.5:
                    p_over25 += prob
                else:
                    p_under25 += prob

                # Both Teams To Score
                if ft_h >= 1 and ft_a >= 1:
                    p_btts += prob

                # Draw No Bet
                if ft_h != ft_a:
                    if ft_h > ft_a:
                        p_dnb_h += prob
                    else:
                        p_dnb_a += prob

        # Normalise DNB (excludes draw probability)
        dnb_total = p_dnb_h + p_dnb_a
        if dnb_total > 0:
            p_dnb_h_norm = p_dnb_h / dnb_total
            p_dnb_a_norm = p_dnb_a / dnb_total
        else:
            p_dnb_h_norm = p_dnb_a_norm = 0.5

        return {
            "1": round(p1, 6),
            "X": round(p_x, 6),
            "2": round(p2, 6),
            "Over25": round(p_over25, 6),
            "Under25": round(p_under25, 6),
            "BTTS": round(p_btts, 6),
            "DNB_H": round(p_dnb_h_norm, 6),
            "DNB_A": round(p_dnb_a_norm, 6),
        }

    def _compute_confidence(
        self,
        lam_h: float,
        lam_a: float,
        probs: Dict[str, float],
    ) -> float:
        """Compute overall confidence score (0-100).

        Confidence is higher when:
        - We are late in the game (more information).
        - There is a clear favourite (skewed 1X2 distribution).
        - Expected goals are moderate (not too chaotic).
        """
        minute = self.score.min

        # Time confidence: increases linearly from 30 (0 min) to 80 (90 min)
        time_conf = 30.0 + (minute / self._MATCH_DURATION_MINUTES) * 50.0

        # Market clarity: how much the leading market share stands out
        market_probs = [probs["1"], probs["X"], probs["2"]]
        max_prob = max(market_probs)
        clarity = (max_prob - 1.0 / 3.0) / (2.0 / 3.0) * 20.0  # 0-20 range

        # Penalise very high total lambdas (chaotic games)
        total_lam = lam_h + lam_a
        chaos_penalty = max(0.0, (total_lam - 2.5) * 5.0)

        confidence = time_conf + clarity - chaos_penalty
        return float(np.clip(confidence, 0.0, 100.0))

    def _generate_alerts(
        self,
        lam_h: float,
        lam_a: float,
        probs: Dict[str, float],
        vix: float,
    ) -> List[str]:
        """Generate human-readable alerts for notable match situations."""
        alerts: List[str] = []
        minute = self.score.min

        if self.score.red_h > 0:
            alerts.append(f"⚠️ HOME has {self.score.red_h} red card(s) – rate suppressed")
        if self.score.red_a > 0:
            alerts.append(f"⚠️ AWAY has {self.score.red_a} red card(s) – rate suppressed")

        if (
            self.score.last_goal >= 0
            and (minute - self.score.last_goal) <= self._POST_GOAL_SHOCK_WINDOW
        ):
            alerts.append(
                f"🔔 Recent goal at min {self.score.last_goal} – post-goal shock active"
            )

        goal_diff = abs(self.score.hg - self.score.ag)
        if minute >= self._GARBAGE_TIME_THRESHOLD and goal_diff >= self._GARBAGE_TIME_DIFF:
            leader = "HOME" if self.score.hg > self.score.ag else "AWAY"
            alerts.append(f"🕐 Garbage time: {leader} leading by {goal_diff} after min {minute}")

        if vix > 75:
            alerts.append(f"📈 HIGH VOLATILITY (VIX={vix:.1f}) – use smaller stakes")
        elif vix > 55:
            alerts.append(f"📊 Elevated volatility (VIX={vix:.1f})")

        if lam_h + lam_a > 3.5:
            alerts.append("⚡ High combined expected goals – OVER markets favoured")

        if probs["BTTS"] > 0.65:
            alerts.append(f"⚽ BTTS probability strong ({probs['BTTS']:.1%})")

        if probs["1"] > 0.75:
            alerts.append(f"🏠 HOME strong favourite ({probs['1']:.1%})")
        elif probs["2"] > 0.75:
            alerts.append(f"✈️ AWAY strong favourite ({probs['2']:.1%})")

        if self.pre.is_knockout and probs["X"] > 0.10:
            alerts.append("🏆 Knockout fixture – draw probability redistributed in ET")

        return alerts

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> OracleOutput:
        """Execute the full oracle pipeline and return a complete output bundle.

        Pipeline steps
        --------------
        1. Compute per-90-min xG rates from in-game stats.
        2. Blend xG rates with pre-match priors (sigmoid).
        3. Apply steam (line-movement) modifiers if available.
        4. Apply situational score effects (red cards, post-goal shock, etc.).
        5. Build joint Poisson matrix with Dixon-Coles correction.
        6. Derive market probabilities from the matrix.
        7. Compute confidence and VIX.
        8. Generate alerts.

        Returns
        -------
        OracleOutput: Full oracle output bundle.
        """
        rate_h, rate_a = self._compute_xg_rates()

        lam_h = self.blend_lambda(rate_h, self.pre.prior_h, self.score.min)
        lam_a = self.blend_lambda(rate_a, self.pre.prior_a, self.score.min)

        # Incorporate line-movement modifiers when SteamAnalyzer is available
        try:
            from apex_black_box.steam import SteamAnalyzer

            steam = SteamAnalyzer(
                self.pre.spread_open,
                self.pre.spread_curr,
                self.pre.total_open,
                self.pre.total_curr,
            )
            mods = steam.get_lambda_modifiers()
            lam_h *= mods.get("home_mod", 1.0)
            lam_a *= mods.get("away_mod", 1.0)
        except Exception:
            pass

        lam_h, lam_a = self.apply_score_effects(lam_h, lam_a)

        # Build and correct joint matrix
        matrix = self.poisson_joint_matrix(lam_h, lam_a)
        matrix = self._apply_dc_correction(matrix, lam_h, lam_a)

        probs = self._compute_market_probs(matrix)
        confidence = self._compute_confidence(lam_h, lam_a, probs)
        vix = self.calc_vix(lam_h, lam_a, confidence)
        alerts = self._generate_alerts(lam_h, lam_a, probs, vix)

        # In knockout fixtures suppress draw probability and redistribute
        if self.pre.is_knockout:
            draw_share = probs["X"]
            probs["1"] = round(probs["1"] + draw_share * 0.5, 6)
            probs["2"] = round(probs["2"] + draw_share * 0.5, 6)
            probs["X"] = 0.0
            # Re-normalise
            total = probs["1"] + probs["X"] + probs["2"]
            if total > 0:
                probs["1"] = round(probs["1"] / total, 6)
                probs["2"] = round(probs["2"] / total, 6)

        return OracleOutput(
            probs=probs,
            confidence=round(confidence, 2),
            vix=round(vix, 2),
            lambdas={"home": round(lam_h, 4), "away": round(lam_a, 4)},
            alerts=alerts,
            raw_score_matrix=matrix.tolist(),
        )
