"""OmegaConf YAML config management with hierarchical merge."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load and merge YAML config with hierarchical defaults.

    Supports Hydra-style defaults list in config.
    Args:
        config_path: Path to primary config file.
        overrides: List of key=value override strings.
    Returns:
        Merged DictConfig.
    """
    cfg = OmegaConf.load(config_path)

    if "_global_" in cfg and "defaults" in cfg["_global_"]:
        defaults = cfg["_global_"]["defaults"]
        base_configs = []
        for default in defaults:
            default_path = Path(config_path).parent / f"{default}.yaml"
            if default_path.exists():
                base_configs.append(OmegaConf.load(default_path))
        if base_configs:
            merged = OmegaConf.merge(*base_configs, cfg)
            cfg = merged

    if overrides:
        cli_overrides = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_overrides)

    return cfg


def save_config(cfg: DictConfig, output_path: str | Path) -> None:
    """Save resolved config to YAML file."""
    with open(output_path, "w") as f:
        OmegaConf.save(cfg, f)


def get_config_value(cfg: DictConfig, key_path: str, default: Any = None) -> Any:
    """Get nested config value using dot notation. Returns default if missing."""
    try:
        return OmegaConf.select(cfg, key_path, default=default)
    except Exception:
        return default
