"""Tests for augmentation transforms."""

import numpy as np
import pytest

from src.data.transforms.common import Resize, RandomFlip, Normalize, build_base_transforms
from src.data.transforms.geometric import RandomRotate, RandomCrop
from src.data.transforms.domain_rand import ColorJitter


class TestResize:
    def test_resize_down(self):
        image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
        bboxes = np.array([[100, 100, 300, 300]])
        labels = np.array([1])

        transform = Resize(target_size=512)
        result = transform(image=image, bboxes=bboxes, labels=labels)

        assert result["image"].shape == (512, 512, 3)
        assert result["bboxes"].shape[1] == 4

    def test_resize_no_bboxes(self):
        image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
        bboxes = np.zeros((0, 4), dtype=np.float32)
        labels = np.array([], dtype=np.int64)

        transform = Resize(target_size=512)
        result = transform(image=image, bboxes=bboxes, labels=labels)
        assert result["bboxes"].shape[0] == 0


class TestRandomFlip:
    def test_flip_horizontal(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[20, 20, 80, 80]])
        labels = np.array([1])

        transform = RandomFlip(h_prob=1.0)
        result = transform(image=image, bboxes=bboxes, labels=labels)

        flipped = np.flip(image, axis=1)
        np.testing.assert_array_equal(result["image"], flipped)


class TestRandomRotate:
    def test_rotate_90_preserves_bbox_count(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[20, 20, 80, 80]], dtype=np.float32)
        labels = np.array([1])

        transform = RandomRotate(k_options=[1])
        result = transform(image=image, bboxes=bboxes, labels=labels)

        assert result["image"].shape == (100, 100, 3)
        assert len(result["bboxes"]) == 1


class TestColorJitter:
    def test_jitter_does_not_change_shape(self):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        bboxes = np.array([[50, 50, 150, 150]])
        labels = np.array([1])

        transform = ColorJitter(prob=1.0)
        result = transform(image=image, bboxes=bboxes, labels=labels)

        assert result["image"].shape == (256, 256, 3)
        assert result["bboxes"].shape == (1, 4)

    def test_jitter_prob_zero_returns_unchanged(self):
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        bboxes = np.array([[50, 50, 150, 150]])
        labels = np.array([1])

        transform = ColorJitter(prob=0.0)
        result = transform(image=image, bboxes=bboxes, labels=labels)

        np.testing.assert_array_equal(result["image"], image)
