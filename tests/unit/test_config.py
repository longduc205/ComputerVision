"""Tests for config utilities."""

import tempfile
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from src.utils.config import get_config_value, load_config, save_config


class TestConfigUtils:
    def test_get_config_value_exists(self):
        cfg = OmegaConf.create({"a": {"b": {"c": 42}}})
        assert get_config_value(cfg, "a.b.c") == 42

    def test_get_config_value_missing_returns_default(self):
        cfg = OmegaConf.create({"a": 1})
        assert get_config_value(cfg, "x.y.z", default=-1) == -1

    def test_load_config_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.yaml"
            data = {"model": {"backbone": "resnet50"}, "training": {"epochs": 10}}
            with open(config_path, "w") as f:
                yaml.dump(data, f)

            cfg = load_config(config_path)
            assert cfg.model.backbone == "resnet50"
            assert cfg.training.epochs == 10

    def test_save_and_load_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({"key": "value", "number": 42})
            output_path = Path(tmpdir) / "output.yaml"

            save_config(cfg, output_path)
            reloaded = OmegaConf.load(output_path)

            assert reloaded.key == "value"
            assert reloaded.number == 42
