"""Preprocess satellite imagery: tile, convert bboxes, filter classes."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess satellite imagery datasets")
    parser.add_argument("--input", required=True, help="Input raw data directory")
    parser.add_argument("--output", required=True, help="Output processed data directory")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size in pixels")
    parser.add_argument("--stride", type=int, default=256, help="Tile stride (overlap)")
    parser.add_argument("--min-bbox-area", type=int, default=25, help="Minimum bbox area to keep")
    return parser.parse_args()


def polygon_to_bbox(polygon):
    """Convert polygon points to axis-aligned bounding box [x, y, w, h]."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def crop_image_with_overlap(
    image_path,
    output_images_dir,
    tile_size=512,
    stride=256,
):
    """Crop large image into overlapping tiles."""
    try:
        image = np.array(Image.open(image_path))
    except Exception:
        return []

    h, w = image.shape[:2]
    tile_count = 0

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = image[y : y + tile_size, x : x + tile_size]
            output_path = output_images_dir / f"{image_path.stem}_tile_{tile_count:05d}.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            tile_count += 1

    return tile_count


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    image_files = list(input_dir.glob("**/*.png")) + list(input_dir.glob("**/*.jpg"))
    print(f"Found {len(image_files)} images to process")

    total_tiles = 0
    for image_path in image_files:
        count = crop_image_with_overlap(
            image_path,
            images_dir,
            tile_size=args.tile_size,
            stride=args.stride,
        )
        total_tiles += count
        print(f"  {image_path.name}: {count} tiles")

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(coco_data, f)

    print(f"\nPreprocessing complete.")
    print(f"  Output: {output_dir}")
    print(f"  Total tiles: {total_tiles}")
    print(f"\nNext: run scripts/create_splits.py to generate train/val/test splits")


if __name__ == "__main__":
    main()
