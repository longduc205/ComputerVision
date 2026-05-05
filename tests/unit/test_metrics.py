"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_performance_drop,
    compute_harmonic_mean,
    compute_all_metrics,
)


class TestPerformanceDrop:
    def test_pd_zero_when_equal(self):
        pd = compute_performance_drop(mAP_ID=0.6, mAP_OOD=0.6)
        assert pd == pytest.approx(0.0)

    def test_pd_positive_when_ood_lower(self):
        pd = compute_performance_drop(mAP_ID=0.8, mAP_OOD=0.4)
        assert pd == pytest.approx(50.0)

    def test_pd_zero_when_id_zero(self):
        pd = compute_performance_drop(mAP_ID=0.0, mAP_OOD=0.0)
        assert pd == 0.0


class TestHarmonicMean:
    def test_hm_equal_to_value_when_equal(self):
        hm = compute_harmonic_mean(mAP_ID=0.6, mAP_OOD=0.6)
        assert hm == pytest.approx(0.6)

    def test_hm_lower_than_arithmetic_mean(self):
        hm = compute_harmonic_mean(mAP_ID=1.0, mAP_OOD=0.5)
        assert hm < (1.0 + 0.5) / 2

    def test_hm_zero_when_sum_zero(self):
        hm = compute_harmonic_mean(mAP_ID=0.0, mAP_OOD=0.0)
        assert hm == 0.0

    def test_hm_correct_formula(self):
        hm = compute_harmonic_mean(mAP_ID=0.8, mAP_OOD=0.4)
        expected = 2 * 0.8 * 0.4 / (0.8 + 0.4)
        assert hm == pytest.approx(expected)


class TestComputeAllMetrics:
    def test_empty_predictions(self):
        metrics = compute_all_metrics([], [])
        assert metrics["mAP"] == 0.0
        assert metrics["mAP_50"] == 0.0

    def test_pd_and_h_computed_when_id_provided(self):
        metrics = compute_all_metrics([], [], mAP_ID=0.6)
        assert "PD" in metrics
        assert "H" in metrics
        assert metrics["PD"] == pytest.approx(100.0)
