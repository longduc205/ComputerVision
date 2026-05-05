"""TensorBoard + JSON structured logging."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Unified logger: TensorBoard + JSON structured logs."""

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        git_commit: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or self._generate_exp_name()
        self.exp_dir = self.log_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.exp_dir / "tensorboard"))

        self.metrics_log: list[Dict[str, Any]] = []
        self.start_time = time.time()

        if config is not None:
            self.log_config(config)
        if git_commit is not None:
            self.log_hyperparameter("git_commit", git_commit)

    def _generate_exp_name(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:6]
        return f"exp_{ts}_{uid}"

    def log_config(self, config: Dict[str, Any]) -> None:
        config_path = self.exp_dir / "config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)
        self.log_hyperparameter("config_path", str(config_path))

    def log_hyperparameter(self, key: str, value: Any) -> None:
        self.writer.add_text(f"hparams/{key}", str(value), 0)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_group: str, values: Dict[str, float], step: int) -> None:
        self.writer.add_scalars(tag_group, values, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        self.writer.add_image(tag, image, step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "eval",
    ) -> None:
        for name, value in metrics.items():
            self.log_scalar(f"{prefix}/{name}", value, step)
        self.metrics_log.append({"step": step, "timestamp": time.time(), **metrics})

    def close(self) -> None:
        self.writer.close()
        log_path = self.exp_dir / "metrics.json"
        with open(log_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
