"""Base dataset class for satellite imagery object detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DetectionSample:
    """Single detection sample."""
    image: torch.Tensor
    bboxes: torch.Tensor
    labels: torch.Tensor
    image_id: int
    orig_size: tuple[int, int]


class BaseDetectionDataset(Dataset, ABC):
    """Abstract base class for detection datasets.

    All dataset implementations must return DetectionSample objects
    with image (C,H,W), bboxes (N,4) as xyxy, and labels (N,).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 512,
        normalize: bool = True,
        transforms: Optional[Any] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.transforms = transforms
        self._load_annotations()

    @abstractmethod
    def _load_annotations(self) -> None:
        """Load annotations from disk. Populate self.annotations list."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> DetectionSample:
        """Return sample at index."""
        pass

    def __len__(self) -> int:
        return len(self.annotations)

    def _apply_transforms(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply augmentation pipeline to image and annotations."""
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)
            return transformed["image"], transformed["bboxes"], transformed["labels"]
        return image, bboxes, labels

    def _normalize_image(
        self,
        image: np.ndarray,
        mean: tuple[float, ...] = (123.675, 116.28, 103.53),
        std: tuple[float, ...] = (58.395, 57.12, 57.375),
    ) -> torch.Tensor:
        """Normalize image with ImageNet-style values."""
        image = image.astype(np.float32)
        image = (image - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)

    def _coco_to_xyxy(
        self,
        bboxes: np.ndarray,
        format: str = "xywh",
    ) -> np.ndarray:
        """Convert bbox formats to xyxy."""
        if format == "xywh":
            x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            return np.stack([x, y, x + w, y + h], axis=1)
        return bboxes
