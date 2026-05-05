"""Single-source training pipeline."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.models.framework_adapter import Detector
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import ExperimentLogger


def train_single_source(
    detector: Detector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    logger: Optional[ExperimentLogger] = None,
    checkpoint_dir: Optional[str] = None,
) -> str:
    """Train detector on a single source domain.

    Args:
        detector: Detector instance (MMDetection or Detectron2).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        cfg: Training configuration dict.
        logger: Optional experiment logger.
        checkpoint_dir: Directory to save checkpoints.

    Returns:
        Path to best checkpoint.
    """
    epochs = cfg.get("training", {}).get("epochs", 50)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir or "results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_metric = 0.0
    best_checkpoint_path = checkpoint_dir / "checkpoint_best.pth"

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss = run_train_epoch(detector, train_loader, device)

        val_metrics = detector.evaluate(val_loader, cfg)
        val_map = val_metrics.get("mAP", 0.0)

        elapsed = time.time() - epoch_start

        if logger:
            logger.log_scalar("train/loss", train_loss, step=epoch)
            logger.log_scalar("eval/mAP", val_map, step=epoch)
            logger.log_scalar("train/epoch_time", elapsed, step=epoch)

        model = getattr(detector, "model", detector)
        save_checkpoint(
            checkpoint_dir,
            model,
            epoch=epoch,
            step=epoch * len(train_loader),
            best_metric=val_map,
        )

        if val_map > best_metric:
            best_metric = val_map
            torch.save(model.state_dict(), best_checkpoint_path)

        if logger:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_mAP={val_map:.4f}, time={elapsed:.1f}s")

    return str(best_checkpoint_path)


def run_train_epoch(
    detector: Detector,
    train_loader: DataLoader,
    device: str,
) -> float:
    """Run one training epoch. Returns average loss."""
    model = getattr(detector, "model", detector)
    if hasattr(model, "train"):
        model.train()

    total_loss = 0.0
    count = 0

    for batch in train_loader:
        images = batch.get("img")
        if images is None:
            continue
        images = images.to(device)

        if hasattr(model, "train_step"):
            losses = model.train_step({"img": images}, None)
            loss = sum(losses.values()) if isinstance(losses, dict) else losses
        elif hasattr(model, "forward"):
            try:
                outputs = model(images)
                loss = outputs.get("loss", torch.tensor(0.0, device=device))
            except Exception:
                loss = torch.tensor(0.0, device=device)
        else:
            loss = torch.tensor(0.0, device=device)

        if isinstance(loss, torch.Tensor):
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)
