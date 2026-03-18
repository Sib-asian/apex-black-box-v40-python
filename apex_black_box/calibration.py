"""
Apex Black Box – Calibration Module
=====================================

Provides:
  • IsotonicCalibrator (M14) — pool-adjacent-violators isotonic regression
    for post-hoc probability calibration of binary markets.
  • CalibrationDashboard — utility class for tracking and analysing bet data.

No external dependencies — pure Python 3.9+.

Usage
-----
    from apex_black_box.calibration import IsotonicCalibrator

    cal = IsotonicCalibrator()
    cal.fit(raw_probs, outcomes)          # lists of floats
    calibrated = [cal.predict(p) for p in new_probs]
"""

from __future__ import annotations

import math
from typing import Sequence


# ─────────────────────────────────────────────────────────────────
#  M14: Isotonic Calibrator (Pool-Adjacent Violators)
# ─────────────────────────────────────────────────────────────────

class IsotonicCalibrator:
    """Isotonic regression calibrator using Pool-Adjacent-Violators (PAV).

    Fits a monotone non-decreasing mapping from raw model probabilities to
    empirical frequencies.  Useful for post-hoc calibration of binary
    market probabilities (e.g. Over25, BTTS) produced by the Oracle Engine.

    Parameters
    ----------
    eps : float
        Small clip applied to predicted probabilities to keep them in
        (eps, 1-eps).  Default 1e-6.

    Example
    -------
    >>> cal = IsotonicCalibrator()
    >>> cal.fit([0.1, 0.4, 0.7, 0.9], [0, 1, 1, 1])
    >>> cal.predict(0.5)
    0.666...
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self._xs: list[float] = []  # sorted breakpoints (raw probs)
        self._ys: list[float] = []  # calibrated values at breakpoints
        self._fitted = False

    # ── fit ──────────────────────────────────────────────────────

    def fit(
        self,
        probs: Sequence[float],
        outcomes: Sequence[float],
    ) -> "IsotonicCalibrator":
        """Fit isotonic regression via Pool-Adjacent-Violators.

        Parameters
        ----------
        probs    : raw predicted probabilities (any order)
        outcomes : binary labels {0, 1} corresponding to each prob
        """
        if len(probs) != len(outcomes):
            raise ValueError("probs and outcomes must have the same length")
        if len(probs) == 0:
            self._xs = []
            self._ys = []
            self._fitted = True
            return self

        # Sort by raw probability
        pairs = sorted(zip(probs, outcomes), key=lambda t: t[0])
        xs = [p for p, _ in pairs]
        ys = [float(o) for _, o in pairs]

        # PAV algorithm — merge adjacent blocks that violate monotonicity
        blocks: list[dict] = [{"sum_y": y, "n": 1, "mean": y, "x_lo": x, "x_hi": x}
                               for x, y in zip(xs, ys)]

        changed = True
        while changed:
            changed = False
            merged: list[dict] = []
            i = 0
            while i < len(blocks):
                b = dict(blocks[i])
                while (merged and merged[-1]["mean"] > b["mean"]):
                    prev = merged.pop()
                    b["sum_y"] += prev["sum_y"]
                    b["n"]     += prev["n"]
                    b["mean"]   = b["sum_y"] / b["n"]
                    b["x_lo"]   = prev["x_lo"]
                    changed = True
                merged.append(b)
                i += 1
            blocks = merged

        # Extract breakpoints and calibrated values
        self._xs = []
        self._ys = []
        for b in blocks:
            self._xs.append(b["x_lo"])
            self._ys.append(b["mean"])
            if b["x_hi"] != b["x_lo"]:
                self._xs.append(b["x_hi"])
                self._ys.append(b["mean"])

        self._fitted = True
        return self

    # ── predict ──────────────────────────────────────────────────

    def predict(self, p: float) -> float:
        """Return calibrated probability for a raw probability *p*.

        Uses piecewise-linear interpolation between fitted breakpoints.
        Falls back to the raw probability if not yet fitted.
        """
        if not self._fitted or len(self._xs) == 0:
            return float(p)

        p = float(p)

        if p <= self._xs[0]:
            val = self._ys[0]
        elif p >= self._xs[-1]:
            val = self._ys[-1]
        else:
            # Linear interpolation
            for i in range(len(self._xs) - 1):
                if self._xs[i] <= p <= self._xs[i + 1]:
                    x0, x1 = self._xs[i], self._xs[i + 1]
                    y0, y1 = self._ys[i], self._ys[i + 1]
                    if x1 == x0:
                        val = (y0 + y1) / 2
                    else:
                        val = y0 + (y1 - y0) * (p - x0) / (x1 - x0)
                    break
            else:
                val = self._ys[-1]

        return max(self.eps, min(1.0 - self.eps, val))

    def __repr__(self) -> str:
        state = f"fitted, {len(self._xs)} breakpoints" if self._fitted else "not fitted"
        return f"IsotonicCalibrator({state})"


# ─────────────────────────────────────────────────────────────────
#  CalibrationDashboard — legacy utility class
# ─────────────────────────────────────────────────────────────────

class CalibrationDashboard:
    """Utility class for tracking and analysing bet data with Brier scoring."""

    def __init__(self) -> None:
        self.bet_data: list[dict] = []
        self.brier_scores: list[float] = []
        self.bias_terms: dict = {}

    def add_bet_data(self, bet: dict) -> None:
        """Add a new betting record for analysis.

        Expected keys: 'prediction' (float) and 'outcome' (0 or 1).
        """
        self.bet_data.append(bet)
        self.brier_scores.append(
            self.calculate_brier_score([bet["prediction"]], [bet["outcome"]])
        )

    def calculate_brier_score(
        self,
        predictions: Sequence[float],
        outcomes: Sequence[float],
    ) -> float:
        """Calculate the Brier score for the given predictions and actual outcomes."""
        n = len(outcomes)
        if n == 0:
            return float("nan")
        return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n

    def analyze_accuracy(self) -> dict:
        """Return mean Brier score and count over stored bet data."""
        if not self.brier_scores:
            return {"mean_brier": float("nan"), "n": 0}
        return {
            "mean_brier": sum(self.brier_scores) / len(self.brier_scores),
            "n": len(self.brier_scores),
        }

    def analyze_biases(self) -> dict:
        """Compute mean prediction vs mean outcome (calibration bias)."""
        if not self.bet_data:
            return {}
        mean_p = sum(b["prediction"] for b in self.bet_data) / len(self.bet_data)
        mean_o = sum(b["outcome"] for b in self.bet_data) / len(self.bet_data)
        return {"mean_prediction": mean_p, "mean_outcome": mean_o, "bias": mean_p - mean_o}

    def generate_recommendations(self) -> list[str]:
        """Generate calibration recommendations based on stored data."""
        recs: list[str] = []
        biases = self.analyze_biases()
        if biases:
            bias = biases.get("bias", 0.0)
            if bias > 0.05:
                recs.append("Model is overconfident — consider downward recalibration.")
            elif bias < -0.05:
                recs.append("Model is underconfident — consider upward recalibration.")
            else:
                recs.append("Calibration looks good.")
        return recs
