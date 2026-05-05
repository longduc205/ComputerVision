"""Framework-agnostic detector interface and factory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class DetectionOutput:
    """Standardized detection output."""
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


class Detector(ABC):
    """Abstract detector interface. All framework wrappers implement this."""

    @abstractmethod
    def train(self, train_loader, val_loader, cfg: Dict[str, Any]) -> str:
        """Train detector. Returns checkpoint path."""
        pass

    @abstractmethod
    def evaluate(self, data_loader, cfg: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate detector. Returns metrics dict."""
        pass

    @abstractmethod
    def predict(self, image: torch.Tensor) -> DetectionOutput:
        """Run inference on a single image."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        pass


def build_detector(config: Dict[str, Any]) -> Detector:
    """Factory function: build detector from config.

    Reads config.framework to dispatch to correct implementation.
    Supported: "mmdet", "detectron2".
    """
    framework = config.get("framework", "mmdet")

    if framework == "mmdet":
        from src.models.mmdet_wrapper import MMDetectionWrapper
        return MMDetectionWrapper(config)
    elif framework == "detectron2":
        from src.models.d2_wrapper import Detectron2Wrapper
        return Detectron2Wrapper(config)
    else:
        raise ValueError(f"Unknown framework: {framework}. Supported: mmdet, detectron2")
