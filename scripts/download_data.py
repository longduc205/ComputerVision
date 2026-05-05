"""Download xView and xBD datasets from public URLs."""

import argparse
from pathlib import Path


DATASETS = {
    "xview": {
        "instructions": "Register at https://xviewdataset.org/ for download link",
    },
    "xbd": {
        "instructions": "Register at https://xviewdataset.org/ for xBD download",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download satellite imagery datasets")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()), help="Dataset to download")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_info = DATASETS[args.dataset]

    output_dir = Path(args.output) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"\nInstructions: {dataset_info['instructions']}")
    print(f"\nTo download:")
    print(f"  1. Visit {dataset_info['instructions']}")
    print(f"  2. Download the dataset files")
    print(f"  3. Place files in: {output_dir}")
    print(f"\nAfter downloading, run preprocessing:")
    print(f"  python scripts/preprocess_data.py --input {output_dir} --output data/processed")


if __name__ == "__main__":
    main()
