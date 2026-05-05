"""Tests for logging utilities."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.utils.logging import ExperimentLogger


class TestExperimentLogger:
    def test_init_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(tmpdir, experiment_name="test_exp")
            assert logger.exp_dir.exists()
            assert (logger.exp_dir / "tensorboard").exists()

    def test_experiment_name_is_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(tmpdir, experiment_name="my_exp")
            assert logger.experiment_name == "my_exp"

    def test_log_scalar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(tmpdir, experiment_name="scalar_test")
            logger.log_scalar("loss/train", 0.5, step=1)
            logger.close()

            log_path = logger.exp_dir / "metrics.json"
            assert log_path.exists()
            with open(log_path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["loss/train"] == 0.5
            assert data[0]["step"] == 1

    def test_log_metrics_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(tmpdir)
            logger.log_metrics({"mAP": 0.75, "PD": 15.2}, step=5)
            logger.close()

            log_path = logger.exp_dir / "metrics.json"
            with open(log_path) as f:
                data = json.load(f)
            assert data[0]["mAP"] == 0.75
            assert data[0]["PD"] == 15.2
