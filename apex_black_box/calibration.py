"""
Apex Black Box V40 - Calibration Dashboard
============================================
Tracks model predictions against actual outcomes to assess calibration
quality and surface actionable improvement recommendations.

Terminology
-----------
- **Brier score**: Mean squared error between predicted probabilities and
  binary outcomes.  Lower is better; 0.25 is the baseline for random guessing.
- **Bias**: Average prediction minus average outcome for a market.
  Positive bias → over-predicting; negative bias → under-predicting.
- **Accuracy**: Fraction of bets where prediction > 0.5 matched the outcome.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

import numpy as np


class CalibrationDashboard:
    """Track and analyse prediction calibration for the Apex Black Box V40 model.

    Usage
    -----
    ::

        dash = CalibrationDashboard()
        dash.add_bet_data({
            'market':     'Over25',
            'prediction': 0.62,
            'outcome':    1,         # 1 = event happened, 0 = did not
            'odds':       1.85,
            'stake':      10.0,
        })
        print(dash.summary())
    """

    def __init__(self) -> None:
        self.bet_data: List[Dict[str, Any]] = []
        self.brier_scores: List[float] = []
        # Per-market data storage: market → list of (prediction, outcome)
        self._market_data: DefaultDict[str, List[tuple]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_bet_data(self, bet: Dict[str, Any]) -> None:
        """Add a single bet record to the dashboard.

        Parameters
        ----------
        bet: Dictionary with at minimum the keys:
             - ``market``     (str)   – market name, e.g. 'Over25'.
             - ``prediction`` (float) – model probability in [0, 1].
             - ``outcome``    (int)   – 1 if the event occurred, 0 otherwise.
             - ``odds``       (float) – decimal odds at time of bet.
             - ``stake``      (float) – monetary stake.
        """
        required = {"market", "prediction", "outcome", "odds", "stake"}
        missing = required - set(bet.keys())
        if missing:
            raise ValueError(f"Bet record missing required keys: {missing}")

        self.bet_data.append(bet)
        bs = self.calculate_brier_score([bet["prediction"]], [bet["outcome"]])
        self.brier_scores.append(bs)
        self._market_data[bet["market"]].append(
            (float(bet["prediction"]), int(bet["outcome"]))
        )

    # ------------------------------------------------------------------
    # Brier score
    # ------------------------------------------------------------------

    def calculate_brier_score(
        self,
        predictions: List[float],
        outcomes: List[int],
    ) -> float:
        """Compute the mean Brier score for a list of predictions and outcomes.

        BS = (1/n) * Σ (p_i - o_i)²

        Parameters
        ----------
        predictions: List of model probabilities in [0, 1].
        outcomes:    List of binary outcomes (0 or 1).

        Returns
        -------
        float: Mean Brier score (lower is better).

        Raises
        ------
        ValueError: If the lists are empty or have mismatched lengths.
        """
        if not predictions or not outcomes:
            raise ValueError("predictions and outcomes must be non-empty.")
        if len(predictions) != len(outcomes):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs "
                f"{len(outcomes)} outcomes."
            )
        return float(np.mean((np.array(predictions) - np.array(outcomes)) ** 2))

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def analyze_accuracy(self) -> Dict[str, float]:
        """Compute accuracy statistics across all stored bet data.

        Returns
        -------
        Dict[str, float]:
            - ``mean_brier``        Overall mean Brier score.
            - ``high_conf_accuracy``Accuracy for predictions >= 0.6.
            - ``low_conf_accuracy`` Accuracy for predictions < 0.6.
            - ``overall_accuracy``  Fraction of bets where rounded prediction
                                    matched outcome.
            - ``mean_prediction``   Average predicted probability.
            - ``mean_outcome``      Fraction of bets that won.
            - ``total_bets``        Total number of bet records stored.
            - ``calibration_error`` |mean_prediction - mean_outcome|.
        """
        if not self.bet_data:
            return {
                "mean_brier": 0.0,
                "high_conf_accuracy": 0.0,
                "low_conf_accuracy": 0.0,
                "overall_accuracy": 0.0,
                "mean_prediction": 0.0,
                "mean_outcome": 0.0,
                "total_bets": 0.0,
                "calibration_error": 0.0,
            }

        all_preds = [b["prediction"] for b in self.bet_data]
        all_outcomes = [b["outcome"] for b in self.bet_data]

        mean_brier = self.calculate_brier_score(all_preds, all_outcomes)
        mean_pred = sum(all_preds) / len(all_preds)
        mean_out = sum(all_outcomes) / len(all_outcomes)

        # Overall directional accuracy
        correct = sum(
            1 for p, o in zip(all_preds, all_outcomes) if round(p) == o
        )
        overall_acc = correct / len(all_preds)

        # High-confidence subset (>= 0.60)
        hc_pairs = [(p, o) for p, o in zip(all_preds, all_outcomes) if p >= 0.60]
        if hc_pairs:
            hc_acc = sum(1 for p, o in hc_pairs if round(p) == o) / len(hc_pairs)
        else:
            hc_acc = 0.0

        # Low-confidence subset (< 0.60)
        lc_pairs = [(p, o) for p, o in zip(all_preds, all_outcomes) if p < 0.60]
        if lc_pairs:
            lc_acc = sum(1 for p, o in lc_pairs if round(p) == o) / len(lc_pairs)
        else:
            lc_acc = 0.0

        return {
            "mean_brier": round(mean_brier, 6),
            "high_conf_accuracy": round(hc_acc, 4),
            "low_conf_accuracy": round(lc_acc, 4),
            "overall_accuracy": round(overall_acc, 4),
            "mean_prediction": round(mean_pred, 4),
            "mean_outcome": round(mean_out, 4),
            "total_bets": float(len(self.bet_data)),
            "calibration_error": round(abs(mean_pred - mean_out), 4),
        }

    def analyze_biases(self) -> Dict[str, float]:
        """Compute per-market prediction bias.

        bias_market = mean(predictions) - mean(outcomes) for that market.

        Positive values indicate over-prediction; negative = under-prediction.

        Returns
        -------
        Dict[str, float]: Market → bias value mapping.
                          Also includes an ``overall`` key for the global bias.
        """
        biases: Dict[str, float] = {}

        for market, pairs in self._market_data.items():
            if not pairs:
                biases[market] = 0.0
                continue
            preds, outs = zip(*pairs)
            mean_p = sum(preds) / len(preds)
            mean_o = sum(outs) / len(outs)
            biases[market] = round(mean_p - mean_o, 4)

        # Overall
        all_preds = [b["prediction"] for b in self.bet_data]
        all_outcomes = [b["outcome"] for b in self.bet_data]
        if all_preds:
            biases["overall"] = round(
                sum(all_preds) / len(all_preds) - sum(all_outcomes) / len(all_outcomes),
                4,
            )
        else:
            biases["overall"] = 0.0

        return biases

    def generate_recommendations(self) -> List[str]:
        """Generate calibration improvement recommendations.

        Returns
        -------
        List[str]: Actionable recommendations based on bias and Brier score analysis.
        """
        if not self.bet_data:
            return ["No bet data recorded yet. Add bet records to receive recommendations."]

        recs: List[str] = []
        accuracy = self.analyze_accuracy()
        biases = self.analyze_biases()

        # Brier score commentary
        bs = accuracy["mean_brier"]
        if bs < 0.10:
            recs.append(
                f"✅ Excellent Brier score ({bs:.4f}). Model calibration is very strong."
            )
        elif bs < 0.20:
            recs.append(
                f"👍 Good Brier score ({bs:.4f}). Minor refinements could improve edge detection."
            )
        elif bs < 0.25:
            recs.append(
                f"⚠️ Moderate Brier score ({bs:.4f}). Review probability estimation for high-variance markets."
            )
        else:
            recs.append(
                f"🚨 Poor Brier score ({bs:.4f} ≥ 0.25 baseline). Model may be poorly calibrated; "
                "consider recalibrating priors."
            )

        # Calibration error
        cal_err = accuracy["calibration_error"]
        if cal_err > 0.10:
            if accuracy["mean_prediction"] > accuracy["mean_outcome"]:
                recs.append(
                    f"🔻 Model is systematically over-predicting by {cal_err:.2%}. "
                    "Reduce probability estimates or tighten prior weights."
                )
            else:
                recs.append(
                    f"🔺 Model is systematically under-predicting by {cal_err:.2%}. "
                    "Increase probability estimates or raise prior weights."
                )

        # High-confidence accuracy
        hca = accuracy["high_conf_accuracy"]
        if hca > 0:
            if hca < 0.55:
                recs.append(
                    f"⚠️ High-confidence predictions win only {hca:.1%} of the time. "
                    "The GREEN classification threshold may need raising."
                )
            elif hca >= 0.70:
                recs.append(
                    f"✅ High-confidence predictions win {hca:.1%} of the time. "
                    "GREEN threshold is well-calibrated."
                )

        # Per-market bias warnings
        for market, bias in biases.items():
            if market == "overall":
                continue
            if abs(bias) > 0.08:
                direction = "over-predicted" if bias > 0 else "under-predicted"
                recs.append(
                    f"📊 Market '{market}' is consistently {direction} "
                    f"(bias = {bias:+.2%}). Review feature weights for this market."
                )

        # Insufficient data warning
        if accuracy["total_bets"] < 30:
            recs.append(
                f"ℹ️ Only {int(accuracy['total_bets'])} bets recorded. "
                "Recommendations become more reliable with ≥ 30 samples."
            )

        if not recs:
            recs.append("ℹ️ No significant calibration issues detected.")

        return recs

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a complete calibration summary.

        Returns
        -------
        Dict[str, Any]:
            - ``accuracy``         Output of :meth:`analyze_accuracy`.
            - ``biases``           Output of :meth:`analyze_biases`.
            - ``recommendations``  Output of :meth:`generate_recommendations`.
            - ``brier_scores``     Individual Brier scores per bet.
            - ``total_bets``       Total number of bets stored.
        """
        return {
            "accuracy": self.analyze_accuracy(),
            "biases": self.analyze_biases(),
            "recommendations": self.generate_recommendations(),
            "brier_scores": [round(bs, 6) for bs in self.brier_scores],
            "total_bets": len(self.bet_data),
        }
