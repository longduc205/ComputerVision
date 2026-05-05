"""xBD dataset loader (building damage assessment)."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.datasets.base import BaseDetectionDataset, DetectionSample


class xBDDataset(BaseDetectionDataset):
    """xBD dataset for building damage detection.

    xBD contains pre- and post-disaster satellite imagery for building
    damage assessment. Damage levels: no damage, minor damage, major
    damage, destroyed.

    Expected processed structure:
    data_root/
        images/{image_id}.png
        annotations.json  # COCO format with polygon -> bbox conversion
    """

    def _load_annotations(self) -> None:
        annotations_path = Path(self.data_root) / "annotations.json"
        with open(annotations_path) as f:
            coco_data = json.load(f)

        self.images = {img["id"]: img for img in coco_data["images"]}
        self.img_to_anns = {img_id: [] for img_id in self.images}
        for ann in coco_data["annotations"]:
            if ann["image_id"] in self.img_to_anns:
                self.img_to_anns[ann["image_id"]].append(ann)

        self.image_ids = sorted(self.images.keys())

    def __getitem__(self, idx: int) -> DetectionSample:
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]

        image_path = Path(self.data_root) / "images" / f"{image_id}.png"
        image = np.array(Image.open(image_path))

        anns = self.img_to_anns[image_id]
        if len(anns) > 0:
            bboxes = np.array([ann["bbox"] for ann in anns], dtype=np.float32)
            labels = np.array([ann["category_id"] for ann in anns], dtype=np.int64)
            bboxes = self._coco_to_xyxy(bboxes, format="xywh")
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)

        image, bboxes, labels = self._apply_transforms(image, bboxes, labels)
        image_tensor = self._normalize_image(image)

        return DetectionSample(
            image=image_tensor,
            bboxes=torch.from_numpy(bboxes).float(),
            labels=torch.from_numpy(labels).long(),
            image_id=image_id,
            orig_size=(img_info.get("height", 0), img_info.get("width", 0)),
        )
