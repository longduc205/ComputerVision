"""Tests for framework adapter."""

import pytest
from unittest.mock import MagicMock, patch

import torch

from src.models.framework_adapter import build_detector, Detector, DetectionOutput
from src.models.mmdet_wrapper import MMDetectionWrapper
from src.models.d2_wrapper import Detectron2Wrapper


class TestDetectionOutput:
    def test_creation(self):
        output = DetectionOutput(
            boxes=torch.zeros(3, 4),
            scores=torch.zeros(3),
            labels=torch.zeros(3, dtype=torch.long),
        )
        assert output.boxes.shape == (3, 4)
        assert output.scores.shape == (3,)
        assert output.labels.shape == (3,)


class TestBuildDetector:
    def test_build_mmdetector(self):
        config = {"framework": "mmdet", "model": {"num_classes": 16}}
        with patch("src.models.mmdet_wrapper.MMDetectionWrapper.__init__", return_value=None):
            wrapper = MMDetectionWrapper(config)
            wrapper._initialized = False
            wrapper.model = None
            detector = build_detector(config)
            assert isinstance(detector, MMDetectionWrapper)

    def test_build_detectron2(self):
        config = {"framework": "detectron2", "model": {"num_classes": 16}}
        with patch("src.models.d2_wrapper.Detectron2Wrapper.__init__", return_value=None):
            wrapper = Detectron2Wrapper(config)
            wrapper._initialized = False
            wrapper.model = None
            detector = build_detector(config)
            assert isinstance(detector, Detectron2Wrapper)

    def test_unknown_framework_raises(self):
        config = {"framework": "unknown"}
        with pytest.raises(ValueError, match="Unknown framework"):
            build_detector(config)


class TestDetectorInterface:
    def test_detector_is_abstract(self):
        """Verify Detector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Detector()
