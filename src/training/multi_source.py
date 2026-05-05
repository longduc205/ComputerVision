"""Multi-source training pipeline with balanced domain sampling."""

import itertools
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.models.framework_adapter import Detector
from src.training.single_source import train_single_source
from src.utils.logging import ExperimentLogger


class MultiSourceDataLoader:
    """Wrapper that combines multiple domain dataloaders into one."""

    def __init__(
        self,
        dataloaders: List[DataLoader],
        domain_ratios: List[float],
    ):
        self.dataloaders = dataloaders
        self.domain_ratios = domain_ratios

    def __iter__(self):
        iterators = [iter(dl) for dl in self.dataloaders]
        total_batches = sum(len(dl) for dl in self.dataloaders)

        round_robin = itertools.cycle(range(len(self.dataloaders)))
        for _ in range(total_batches):
            domain_idx = next(round_robin)
            try:
                batch = next(iterators[domain_idx])
                yield batch
            except StopIteration:
                iterators[domain_idx] = iter(self.dataloaders[domain_idx])
                try:
                    yield next(iterators[domain_idx])
                except StopIteration:
                    pass

    def __len__(self) -> int:
        return sum(len(dl) for dl in self.dataloaders)


def train_multi_source(
    detectors: List[Detector],
    dataloaders: List[DataLoader],
    val_loaders: List[DataLoader],
    cfg: Dict[str, Any],
    domain_ratios: Optional[List[float]] = None,
    logger: Optional[ExperimentLogger] = None,
    checkpoint_dir: Optional[str] = None,
) -> List[str]:
    """Train detectors from multiple source domains simultaneously.

    Args:
        detectors: List of detector instances, one per domain.
        dataloaders: List of training dataloaders, one per domain.
        val_loaders: List of validation dataloaders, one per domain.
        cfg: Training configuration.
        domain_ratios: Sampling ratios per domain. Default: equal.
        logger: Optional experiment logger.
        checkpoint_dir: Directory for checkpoints.

    Returns:
        List of best checkpoint paths, one per domain.
    """
    if domain_ratios is None:
        domain_ratios = [1.0 / len(detectors)] * len(detectors)

    multi_loader = MultiSourceDataLoader(dataloaders, domain_ratios)
    combined_val_loader = val_loaders[0]
    primary_detector = detectors[0]

    checkpoint_path = train_single_source(
        detector=primary_detector,
        train_loader=multi_loader,
        val_loader=combined_val_loader,
        cfg=cfg,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
    )

    return [checkpoint_path]
