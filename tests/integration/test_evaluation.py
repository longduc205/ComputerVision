"""Integration tests for evaluation pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd


class TestAnalyzers:
    def test_analyze_per_class_metrics(self):
        try:
            from src.evaluation.analyzers import analyze_per_class_metrics
        except ImportError:
            pytest.skip("matplotlib/seaborn not available")

        metrics = {1: 0.75, 2: 0.60, 3: 0.82}
        class_names = {1: "building", 2: "vehicle", 3: "tree"}

        with tempfile.TemporaryDirectory() as tmpdir:
            df = analyze_per_class_metrics(metrics, class_names, tmpdir)

            assert len(df) == 3
            assert (Path(tmpdir) / "per_class_ap_bar.png").exists()
            assert (Path(tmpdir) / "per_class_metrics.csv").exists()

    def test_analyze_domain_performance(self):
        try:
            from src.evaluation.analyzers import analyze_domain_performance
        except ImportError:
            pytest.skip("matplotlib/seaborn not available")

        results = {
            "tropical": {"mAP": 0.72, "mAP_50": 0.85},
            "arid": {"mAP": 0.65, "mAP_50": 0.78},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            df = analyze_domain_performance(results, tmpdir)
            assert len(df) == 4
            assert (Path(tmpdir) / "domain_comparison.png").exists()

    def test_save_results_summary(self):
        try:
            from src.evaluation.analyzers import save_results_summary
        except ImportError:
            pytest.skip("pandas not available")

        results = {"metrics": {"mAP": 0.72, "PD": 12.5, "H": 0.68}}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_results_summary(results, tmpdir)
            assert (Path(tmpdir) / "eval_results.json").exists()
            assert (Path(tmpdir) / "metrics_summary.csv").exists()
