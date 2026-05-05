"""Create train/val/test splits following RWDS Algorithm 1."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for RWDS")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., rwds_cz)")
    parser.add_argument("--annotations", required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--output", default="data/splits", help="Output directory for split files")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--min-samples", type=int, default=30, help="Minimum samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    from src.data.splits.allocator import allocate_splits, save_splits

    with open(args.annotations) as f:
        annotations = json.load(f)

    split_annotations = [
        {"image_id": ann["image_id"], "category_id": ann["category_id"]}
        for ann in annotations.get("annotations", [])
    ]

    splits = allocate_splits(
        annotations=split_annotations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_samples_per_class=args.min_samples,
        seed=args.seed,
    )

    output_dir = Path(args.output)
    save_splits(splits, output_dir, args.dataset)

    print(f"Splits created for {args.dataset}:")
    for split_name, image_ids in splits.items():
        print(f"  {split_name}: {len(image_ids)} images")


if __name__ == "__main__":
    main()
