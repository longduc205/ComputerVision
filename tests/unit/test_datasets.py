"""Tests for dataset classes."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.datasets.base import BaseDetectionDataset, DetectionSample


class DummyDataset(BaseDetectionDataset):
    """Dummy dataset for testing."""

    def _load_annotations(self) -> None:
        self.annotations = [
            {"image_id": 0, "bboxes": np.array([[10, 10, 50, 50]]), "labels": np.array([1])},
            {"image_id": 1, "bboxes": np.array([[20, 20, 80, 80]]), "labels": np.array([2])},
            {"image_id": 2, "bboxes": np.zeros((0, 4)), "labels": np.array([], dtype=np.int64)},
        ]

    def __getitem__(self, idx: int) -> DetectionSample:
        ann = self.annotations[idx]
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return DetectionSample(
            image=torch.from_numpy(image).permute(2, 0, 1).float(),
            bboxes=torch.from_numpy(ann["bboxes"]).float(),
            labels=torch.from_numpy(ann["labels"]).long(),
            image_id=ann["image_id"],
            orig_size=(512, 512),
        )


class TestBaseDetectionDataset:
    def test_len(self):
        ds = DummyDataset(data_root="dummy")
        assert len(ds) == 3

    def test_getitem_with_bboxes(self):
        ds = DummyDataset(data_root="dummy")
        sample = ds[0]
        assert isinstance(sample, DetectionSample)
        assert sample.image.shape == (3, 512, 512)
        assert sample.bboxes.shape[1] == 4
        assert sample.labels.shape[0] == sample.bboxes.shape[0]

    def test_getitem_empty_bboxes(self):
        ds = DummyDataset(data_root="dummy")
        sample = ds[2]
        assert sample.bboxes.shape[0] == 0
        assert sample.labels.shape[0] == 0

    def test_coco_to_xyxy(self):
        ds = DummyDataset(data_root="dummy")
        bboxes_xywh = np.array([[10, 20, 50, 60]])
        bboxes_xyxy = ds._coco_to_xyxy(bboxes_xywh, format="xywh")
        np.testing.assert_array_almost_equal(bboxes_xyxy, np.array([[10, 20, 60, 80]]))

    def test_collate_fn(self):
        def collate_fn(batch):
            images = torch.stack([s.image for s in batch])
            return images

        ds = DummyDataset(data_root="dummy")
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        assert batch.shape == (2, 3, 512, 512)
