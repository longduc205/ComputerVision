"""RWDS (Real-World Distribution Shifts) dataset wrapper."""

from pathlib import Path
from typing import List, Optional

from src.data.datasets.xview import xViewDataset
from src.data.datasets.xbd import xBDDataset


class RWDSDataset:
    """RWDS dataset factory.

    RWDS consists of three sub-datasets:
    - RWDS-CZ: Climate Zone dataset derived from xView
    - RWDS-FR: Flood Region dataset derived from xBD
    - RWDS-HE: Hurricane Event dataset derived from xBD

    This class provides a unified interface to access all RWDS variants
    with proper domain metadata.
    """

    VARIANTS = {
        "rwds_cz": {"source": "xview", "domains": ["tropical", "arid", "temperate"]},
        "rwds_fr": {"source": "xbd", "domains": ["us_flood", "india_flood"]},
        "rwds_he": {"source": "xbd", "domains": ["florence", "michael", "harvey", "matthew"]},
    }

    def __init__(
        self,
        variant: str,
        data_root: str,
        domain: str,
        split: str = "train",
        image_size: int = 512,
        transforms: Optional = None,
    ):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown RWDS variant: {variant}. Available: {list(self.VARIANTS.keys())}")

        if domain not in self.VARIANTS[variant]["domains"]:
            raise ValueError(
                f"Unknown domain '{domain}' for {variant}. "
                f"Available: {self.VARIANTS[variant]['domains']}"
            )

        self.variant = variant
        self.domain = domain
        self.source = self.VARIANTS[variant]["source"]

        domain_root = Path(data_root) / variant / domain
        source_class = xViewDataset if self.source == "xview" else xBDDataset

        self.dataset = source_class(
            data_root=str(domain_root),
            split=split,
            image_size=image_size,
            transforms=transforms,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def list_domains(variant: str) -> List[str]:
        """List available domains for a RWDS variant."""
        return RWDSDataset.VARIANTS[variant]["domains"]
