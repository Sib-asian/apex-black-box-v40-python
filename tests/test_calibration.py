"""
Tests for apex_black_box.calibration — IsotonicCalibrator and CalibrationDashboard.
"""
from __future__ import annotations

import math
import pytest

from apex_black_box.calibration import IsotonicCalibrator, CalibrationDashboard


class TestIsotonicCalibratorFit:
    def test_monotone_sequence(self):
        """Fit and predict on a perfectly monotone sequence."""
        cal = IsotonicCalibrator()
        probs = [0.1, 0.3, 0.6, 0.8]
        outcomes = [0.0, 0.0, 1.0, 1.0]
        cal.fit(probs, outcomes)
        assert cal._fitted
        # predict should return a value in (0, 1)
        p = cal.predict(0.5)
        assert 0.0 < p < 1.0

    def test_non_monotone_pav_correction(self):
        """PAV must merge blocks that violate monotonicity."""
        cal = IsotonicCalibrator()
        # non-monotone: high prob has low outcome, low prob has high outcome
        probs = [0.2, 0.4, 0.6, 0.8]
        outcomes = [1.0, 0.0, 0.0, 1.0]
        cal.fit(probs, outcomes)
        assert cal._fitted
        # The PAV result must be non-decreasing
        test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
        preds = [cal.predict(p) for p in test_points]
        for i in range(len(preds) - 1):
            assert preds[i] <= preds[i + 1] + 1e-9, (
                f"Isotonic violation: predict({test_points[i]})={preds[i]} > "
                f"predict({test_points[i+1]})={preds[i+1]}"
            )

    def test_fit_empty_list(self):
        """fit([]) should not raise and predict should pass through."""
        cal = IsotonicCalibrator()
        cal.fit([], [])
        assert cal._fitted
        # With empty fit, predict returns raw probability unchanged
        assert cal.predict(0.3) == 0.3
        assert cal.predict(0.7) == 0.7

    def test_predict_not_fitted(self):
        """predict on unfitted calibrator must return input probability."""
        cal = IsotonicCalibrator()
        assert not cal._fitted
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert cal.predict(p) == p

    def test_predict_out_of_range_raises(self):
        """predict() must raise ValueError for p outside [0, 1]."""
        cal = IsotonicCalibrator()
        cal.fit([0.2, 0.8], [0, 1])
        with pytest.raises(ValueError):
            cal.predict(-0.1)
        with pytest.raises(ValueError):
            cal.predict(1.1)

    def test_predict_boundary_values(self):
        """predict(0.0) and predict(1.0) must not raise."""
        cal = IsotonicCalibrator()
        cal.fit([0.1, 0.9], [0.0, 1.0])
        p_lo = cal.predict(0.0)
        p_hi = cal.predict(1.0)
        assert 0.0 < p_lo < 1.0
        assert 0.0 < p_hi < 1.0

    def test_eps_clipping(self):
        """Predicted values must be clipped to [eps, 1-eps]."""
        cal = IsotonicCalibrator(eps=1e-6)
        # Force isotonic output of 0 and 1
        cal.fit([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
        assert cal.predict(0.0) >= 1e-6
        assert cal.predict(1.0) <= 1.0 - 1e-6

    def test_probs_outcomes_length_mismatch(self):
        """Mismatched lengths must raise ValueError."""
        cal = IsotonicCalibrator()
        with pytest.raises(ValueError):
            cal.fit([0.1, 0.5], [1.0])

    def test_monotone_output_after_pav(self):
        """After fit on any input, predict must be non-decreasing."""
        import random
        random.seed(42)
        probs = sorted([random.random() for _ in range(20)])
        outcomes = [float(random.random() > 0.4) for _ in range(20)]
        cal = IsotonicCalibrator()
        cal.fit(probs, outcomes)
        test_pts = [i / 20 for i in range(21)]
        preds = [cal.predict(p) for p in test_pts]
        for i in range(len(preds) - 1):
            assert preds[i] <= preds[i + 1] + 1e-9


class TestCalibrationDashboard:
    def test_add_bet_data_and_brier_score(self):
        """add_bet_data() accumulates records and brier_scores."""
        db = CalibrationDashboard()
        db.add_bet_data({"prediction": 0.7, "outcome": 1})
        db.add_bet_data({"prediction": 0.3, "outcome": 0})
        assert len(db.bet_data) == 2
        assert len(db.brier_scores) == 2
        # Brier for (0.7, 1) = 0.09; Brier for (0.3, 0) = 0.09
        assert abs(db.brier_scores[0] - 0.09) < 1e-9
        assert abs(db.brier_scores[1] - 0.09) < 1e-9

    def test_analyze_accuracy(self):
        """analyze_accuracy returns mean_brier and n."""
        db = CalibrationDashboard()
        db.add_bet_data({"prediction": 0.5, "outcome": 1})
        db.add_bet_data({"prediction": 0.5, "outcome": 0})
        result = db.analyze_accuracy()
        # Brier for 0.5 vs 1 = 0.25; Brier for 0.5 vs 0 = 0.25; mean = 0.25
        assert abs(result["mean_brier"] - 0.25) < 1e-9
        assert result["n"] == 2

    def test_empty_dashboard(self):
        """Empty dashboard returns nan mean_brier and 0 count."""
        db = CalibrationDashboard()
        result = db.analyze_accuracy()
        assert math.isnan(result["mean_brier"])
        assert result["n"] == 0

    def test_calculate_brier_score_direct(self):
        """calculate_brier_score works on arbitrary lists."""
        db = CalibrationDashboard()
        bs = db.calculate_brier_score([0.0, 1.0], [0.0, 1.0])
        assert abs(bs - 0.0) < 1e-9  # perfect predictions
        bs2 = db.calculate_brier_score([1.0, 0.0], [0.0, 1.0])
        assert abs(bs2 - 1.0) < 1e-9  # worst predictions

    def test_analyze_biases(self):
        """analyze_biases computes mean prediction, outcome, and bias."""
        db = CalibrationDashboard()
        db.add_bet_data({"prediction": 0.8, "outcome": 1})
        db.add_bet_data({"prediction": 0.8, "outcome": 0})
        biases = db.analyze_biases()
        assert abs(biases["mean_prediction"] - 0.8) < 1e-9
        assert abs(biases["mean_outcome"] - 0.5) < 1e-9
        assert abs(biases["bias"] - 0.3) < 1e-9
