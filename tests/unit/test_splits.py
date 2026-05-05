"""Tests for split allocator."""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from src.data.splits.allocator import allocate_splits, save_splits, load_splits


class TestAllocateSplits:
    def test_splits_cover_all_images(self):
        annotations = [{"image_id": i, "category_id": 1} for i in range(100)]
        splits = allocate_splits(annotations, train_ratio=0.7, val_ratio=0.15)

        all_ids = set(splits["train"] + splits["val"] + splits["test"])
        assert len(all_ids) == 100

    def test_splits_are_disjoint(self):
        annotations = [{"image_id": i, "category_id": 1} for i in range(50)]
        splits = allocate_splits(annotations, train_ratio=0.7, val_ratio=0.15)

        assert len(set(splits["train"]) & set(splits["val"])) == 0
        assert len(set(splits["train"]) & set(splits["test"])) == 0
        assert len(set(splits["val"]) & set(splits["test"])) == 0

    def test_splits_ratio_approximately_correct(self):
        annotations = [{"image_id": i, "category_id": 1} for i in range(1000)]
        splits = allocate_splits(annotations, train_ratio=0.7, val_ratio=0.15)

        n_train = len(splits["train"])
        n_val = len(splits["val"])
        n_test = len(splits["test"])

        assert 0.69 <= n_train / 1000 <= 0.71
        assert 0.14 <= n_val / 1000 <= 0.16
        assert 0.13 <= n_test / 1000 <= 0.17

    def test_reproducibility_with_same_seed(self):
        annotations = [{"image_id": i, "category_id": 1} for i in range(50)]
        splits1 = allocate_splits(annotations, seed=42)
        splits2 = allocate_splits(annotations, seed=42)

        assert splits1["train"] == splits2["train"]
        assert splits1["val"] == splits2["val"]

    def test_raises_on_insufficient_samples(self):
        annotations = [{"image_id": i, "category_id": 1} for i in range(5)]
        with pytest.raises(ValueError, match="No class has"):
            allocate_splits(annotations, min_samples_per_class=30)


class TestSaveLoadSplits:
    def test_save_and_load_roundtrip(self):
        annotations = [{"image_id": i, "category_id": 1} for i in range(30)]
        splits = allocate_splits(annotations)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_splits(splits, tmpdir, "test_dataset")
            reloaded = load_splits(tmpdir, "test_dataset")

            for key in ["train", "val", "test"]:
                assert reloaded[key] == splits[key]
