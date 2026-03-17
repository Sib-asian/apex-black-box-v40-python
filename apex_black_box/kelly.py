"""
Apex Black Box V40 - Kelly Criterion Staking
=============================================
Fractional Kelly staking with VIX-based and confidence-based adjustments.
All monetary outputs are in the same units as ``bankroll`` (default USD).
"""

from __future__ import annotations

from typing import Dict


class KellyCriterion:
    """Compute optimal fractional-Kelly stakes.

    Parameters
    ----------
    bankroll:   Current bankroll / available capital.
    max_kelly:  Hard cap on stake as a fraction of bankroll (e.g. 0.25 = 25%).
    fraction:   Fractional Kelly divisor (e.g. 0.5 = half-Kelly).
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        max_kelly: float = 0.25,
        fraction: float = 0.5,
    ) -> None:
        if bankroll <= 0:
            raise ValueError(f"bankroll must be positive, got {bankroll}")
        if not (0.0 < max_kelly <= 1.0):
            raise ValueError(f"max_kelly must be in (0, 1], got {max_kelly}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

        self.bankroll = bankroll
        self.max_kelly = max_kelly
        self.fraction = fraction

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def calculate(
        self,
        prob: float,
        odds: float,
        vix: float,
        confidence: float,
    ) -> Dict[str, float]:
        """Calculate Kelly stake components for a single bet.

        Adjustments applied (in order)
        --------------------------------
        1. **Full Kelly** = (edge) / (odds - 1) where edge = prob - 1/odds.
        2. **Confidence scaling**: kelly_full *= (confidence / 100).
        3. **VIX adjustment**:
           - vix < 30  → return zero-stake (market too uncertain).
           - vix > 60  → reduce by 20%.
        4. **Fractional Kelly**: kelly_fractional = kelly_full * self.fraction.
        5. **Stake**: min(kelly_fractional * bankroll, bankroll * max_kelly).
        6. Negative Kelly (no edge) → all outputs are 0.

        Parameters
        ----------
        prob:       Model probability of the outcome (0 < prob < 1).
        odds:       Decimal odds offered by bookmaker (must be > 1.0).
        vix:        Volatility index from the oracle (0-100).
        confidence: Oracle confidence score (0-100).

        Returns
        -------
        Dict[str, float]:
            - ``kelly_full``        Full Kelly fraction (unbounded).
            - ``kelly_fractional``  Fractional Kelly after adjustments.
            - ``stake``             Recommended monetary stake.
            - ``ev``                Expected value per unit staked.
            - ``edge``              Raw edge (prob - implied_prob).
        """
        if odds <= 1.0:
            raise ValueError(f"odds must be > 1.0, got {odds}")
        if not (0.0 < prob < 1.0):
            raise ValueError(f"prob must be in (0, 1), got {prob}")

        implied = self.implied_probability(odds)
        edge = prob - implied
        ev = prob * (odds - 1.0) - (1.0 - prob)

        # No edge → skip
        if edge <= 0.0:
            return {
                "kelly_full": 0.0,
                "kelly_fractional": 0.0,
                "stake": 0.0,
                "ev": round(ev, 6),
                "edge": round(edge, 6),
            }

        # VIX guard – highly volatile markets are too risky to bet
        if vix < 30.0:
            return {
                "kelly_full": 0.0,
                "kelly_fractional": 0.0,
                "stake": 0.0,
                "ev": round(ev, 6),
                "edge": round(edge, 6),
            }

        # Full Kelly formula: f* = (b*p - q) / b  where b = odds - 1
        b = odds - 1.0
        kelly_full = (b * prob - (1.0 - prob)) / b

        # Confidence scaling
        kelly_full *= confidence / 100.0

        # VIX adjustment
        if vix > 60.0:
            kelly_full *= 0.80

        # Ensure non-negative after adjustments
        kelly_full = max(0.0, kelly_full)

        # Fractional Kelly
        kelly_fractional = kelly_full * self.fraction

        # Stake (hard-capped)
        stake = min(
            kelly_fractional * self.bankroll,
            self.bankroll * self.max_kelly,
        )

        return {
            "kelly_full": round(kelly_full, 6),
            "kelly_fractional": round(kelly_fractional, 6),
            "stake": round(stake, 2),
            "ev": round(ev, 6),
            "edge": round(edge, 6),
        }

    def is_value_bet(self, prob: float, odds: float) -> bool:
        """Return True when the model probability exceeds the implied probability.

        Parameters
        ----------
        prob: Model probability (0 < prob < 1).
        odds: Decimal odds (must be > 1.0).

        Returns
        -------
        bool: True if there is positive edge.
        """
        if odds <= 1.0 or prob <= 0.0 or prob >= 1.0:
            return False
        return prob > self.implied_probability(odds)

    @staticmethod
    def implied_probability(odds: float) -> float:
        """Compute implied probability from decimal odds.

        Parameters
        ----------
        odds: Decimal odds (must be > 0).

        Returns
        -------
        float: 1 / odds (raw, no overround removal).
        """
        if odds <= 0.0:
            raise ValueError(f"odds must be positive, got {odds}")
        return 1.0 / odds

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def max_stake(self) -> float:
        """Hard-cap stake (bankroll * max_kelly)."""
        return self.bankroll * self.max_kelly

    def update_bankroll(self, new_bankroll: float) -> None:
        """Update bankroll in-place (e.g., after a settled bet).

        Parameters
        ----------
        new_bankroll: New bankroll value (must be positive).
        """
        if new_bankroll <= 0:
            raise ValueError(f"new_bankroll must be positive, got {new_bankroll}")
        self.bankroll = new_bankroll

    def bet_result(
        self,
        stake: float,
        odds: float,
        won: bool,
    ) -> float:
        """Calculate P&L for a settled bet and update bankroll.

        Parameters
        ----------
        stake: Amount staked.
        odds:  Decimal odds at which the bet was placed.
        won:   True if the bet won.

        Returns
        -------
        float: P&L (positive = profit, negative = loss).
        """
        if won:
            pnl = stake * (odds - 1.0)
        else:
            pnl = -stake
        self.update_bankroll(self.bankroll + pnl)
        return pnl
