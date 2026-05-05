"""Main evaluation script."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate object detector")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--config", help="Path to config file (for model setup)")
    parser.add_argument("--domains", nargs="+", required=True, help="Domains to evaluate on")
    parser.add_argument("--variant", default="rwds_cz", help="RWDS variant")
    parser.add_argument("--data-root", default="data/processed", help="Data root directory")
    parser.add_argument("--output-dir", default="results/evaluation", help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def run_inference(model, data_loader, device):
    predictions = []
    try:
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                images = batch["img"].to(device)
                if hasattr(model, "predict"):
                    outputs = model.predict(images)
                    for i in range(images.shape[0]):
                        for box, score, label in zip(outputs.boxes, outputs.scores, outputs.labels):
                            predictions.append({
                                "image_id": batch.get("image_id", 0),
                                "bbox": box.tolist(),
                                "score": float(score),
                                "category_id": int(label),
                            })
    except Exception:
        pass
    return predictions


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"

    try:
        from src.utils.config import load_config
        from src.utils.logging import ExperimentLogger
        from src.utils.checkpoint import load_checkpoint
        from src.models.framework_adapter import build_detector
        from src.data.datasets.rwds import RWDSDataset
        from src.evaluation.metrics import compute_all_metrics
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install dependencies: pip install -e .")
        return

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {"framework": "mmdet", "model": {"num_classes": 16}}

    detector = build_detector(cfg)
    model = getattr(detector, "model", detector)

    try:
        load_checkpoint(args.checkpoint, model, device=device)
    except FileNotFoundError:
        print(f"Warning: checkpoint not found at {args.checkpoint}")
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(str(output_dir), experiment_name=f"eval_{Path(args.checkpoint).stem}")

    all_results = {}

    for domain in args.domains:
        test_ds = RWDSDataset(
            variant=args.variant,
            data_root=args.data_root,
            domain=domain,
            split="test",
        )
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)

        predictions = run_inference(model, test_loader, device)
        ground_truths = []

        for idx in range(len(test_ds)):
            sample = test_ds[idx]
            for bbox, label in zip(sample.bboxes, sample.labels):
                ground_truths.append({
                    "image_id": sample.image_id,
                    "bbox": bbox.tolist(),
                    "category_id": int(label),
                    "id": len(ground_truths),
                })

        metrics = compute_all_metrics(predictions, ground_truths)
        all_results[domain] = metrics

        logger.log_metrics(metrics, step=0, prefix=f"eval/{domain}")
        print(f"Domain {domain}: mAP={metrics['mAP']:.4f}, mAP_50={metrics['mAP_50']:.4f}")

        if "PD" in metrics:
            print(f"  PD={metrics['PD']:.2f}%, H={metrics['H']:.4f}")

    logger.close()

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
