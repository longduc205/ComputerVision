"""Data split allocator implementing RWDS Algorithm 1."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def allocate_splits(
    annotations: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    min_samples_per_class: int = 30,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Allocate samples into train/val/test splits per class.

    Implements RWDS Algorithm 1: balanced allocation ensuring each class
    has sufficient samples in each split.

    Args:
        annotations: List of annotation dicts with "image_id" and "category_id".
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        min_samples_per_class: Minimum samples per class to keep it.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys "train", "val", "test" -> list of image_ids.
    """
    import numpy as np
    np.random.seed(seed)

    class_to_images = defaultdict(list)
    for ann in annotations:
        class_to_images[ann["category_id"]].append(ann["image_id"])

    valid_classes = {
        cat_id: img_ids
        for cat_id, img_ids in class_to_images.items()
        if len(img_ids) >= min_samples_per_class
    }

    if len(valid_classes) == 0:
        raise ValueError(f"No class has >= {min_samples_per_class} samples")

    all_image_ids = sorted(set(img_id for img_ids in valid_classes.values() for img_id in img_ids))

    n_train = int(len(all_image_ids) * train_ratio)
    n_val = int(len(all_image_ids) * val_ratio)

    np.random.shuffle(all_image_ids)
    train_ids = set(all_image_ids[:n_train])
    val_ids = set(all_image_ids[n_train : n_train + n_val])
    test_ids = set(all_image_ids[n_train + n_val :])

    return {
        "train": sorted(list(train_ids)),
        "val": sorted(list(val_ids)),
        "test": sorted(list(test_ids)),
    }


def save_splits(
    splits: Dict[str, List[int]],
    output_dir: str | Path,
    dataset_name: str,
) -> None:
    """Save split image IDs to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, image_ids in splits.items():
        filepath = output_dir / f"{dataset_name}_{split_name}.json"
        with open(filepath, "w") as f:
            json.dump(
                {"dataset": dataset_name, "split": split_name, "image_ids": image_ids, "count": len(image_ids)},
                f,
            )


def load_splits(
    input_dir: str | Path,
    dataset_name: str,
) -> Dict[str, List[int]]:
    """Load split image IDs from JSON files."""
    input_dir = Path(input_dir)
    splits = {}
    for split_name in ["train", "val", "test"]:
        filepath = input_dir / f"{dataset_name}_{split_name}.json"
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            splits[split_name] = data["image_ids"]
    return splits
