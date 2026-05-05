"""Main training script: unified entry point for single/multi-source training."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train object detector")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--single-source", action="store_true", help="Single-source training")
    parser.add_argument("--multi-source", action="store_true", help="Multi-source training")
    parser.add_argument(
        "--dg-technique",
        default="none",
        choices=["none", "grad_reversal", "clip_align", "meta_learning"],
        help="Domain-generalisation technique",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--checkpoint-dir", help="Override checkpoint directory")
    parser.add_argument("--domains", nargs="+", help="Domain list for multi-source training")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from src.utils.config import load_config
        from src.utils.logging import ExperimentLogger
        from src.models.framework_adapter import build_detector
        from src.training.single_source import train_single_source
        from src.training.multi_source import train_multi_source
        from src.training.domain_gen import apply_dg_technique
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install dependencies: pip install -e .")
        sys.exit(1)

    cfg = load_config(args.config)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    checkpoint_dir = args.checkpoint_dir or str(Path("results") / "experiments")
    logger = ExperimentLogger(
        log_dir=str(Path("results") / "experiments"),
        config=dict(cfg),
        git_commit=git_commit,
    )

    cfg["device"] = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    if args.checkpoint_dir:
        cfg["checkpoint_dir"] = args.checkpoint_dir

    detector = build_detector(cfg)

    if args.dg_technique != "none":
        model = getattr(detector, "model", detector)
        apply_dg_technique(model, args.dg_technique, cfg)

    variant = cfg.get("dataset", {}).get("name", "rwds_cz")
    data_root = cfg.get("dataset", {}).get("data_root", "data/processed")

    if args.multi_source and args.domains:
        from src.data.datasets.rwds import RWDSDataset

        dataloaders = []
        val_loaders = []

        for domain in args.domains:
            train_ds = RWDSDataset(
                variant=variant, data_root=data_root, domain=domain, split="train"
            )
            val_ds = RWDSDataset(
                variant=variant, data_root=data_root, domain=domain, split="val"
            )
            batch_size = cfg.get("training", {}).get("batch_size", 4)
            dataloaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4))
            val_loaders.append(DataLoader(val_ds, batch_size=batch_size, num_workers=4))

        checkpoint_paths = train_multi_source(
            detectors=[detector],
            dataloaders=dataloaders,
            val_loaders=val_loaders,
            cfg=cfg,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        from src.data.datasets.rwds import RWDSDataset

        domain = args.domains[0] if args.domains else cfg.get("dataset", {}).get("domain", "tropical")

        train_ds = RWDSDataset(variant=variant, data_root=data_root, domain=domain, split="train")
        val_ds = RWDSDataset(variant=variant, data_root=data_root, domain=domain, split="val")

        batch_size = cfg.get("training", {}).get("batch_size", 4)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

        checkpoint_path = train_single_source(
            detector=detector,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
        )

    logger.log_hyperparameter("best_checkpoint", str(checkpoint_dir))
    logger.close()

    print(f"Training complete. Experiment: {logger.experiment_name}")
    print(f"Logs: {logger.exp_dir}")


if __name__ == "__main__":
    main()
