"""Common deterministic transforms for satellite imagery."""

from typing import List

import cv2
import numpy as np


class Resize:
    """Resize image and bboxes to target size."""

    def __init__(self, target_size: int = 512):
        self.target_size = target_size

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, **kwargs):
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h))

        if len(bboxes) > 0:
            bboxes = bboxes * scale
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)

        padded = np.zeros((self.target_size, self.target_size, 3), dtype=image.dtype)
        padded[:new_h, :new_w] = resized

        return {"image": padded, "bboxes": bboxes, "labels": labels}


class RandomFlip:
    """Random horizontal and/or vertical flip."""

    def __init__(self, h_prob: float = 0.5, v_prob: float = 0.0):
        self.h_prob = h_prob
        self.v_prob = v_prob

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, **kwargs):
        if len(bboxes) > 0:
            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            w = image.shape[1]
            h = image.shape[0]

            if np.random.rand() < self.h_prob:
                image = np.flip(image, axis=1)
                bboxes[:, 0], bboxes[:, 2] = w - x2, w - x1

            if np.random.rand() < self.v_prob:
                image = np.flip(image, axis=0)
                bboxes[:, 1], bboxes[:, 3] = h - y2, h - y1

        return {"image": image, "bboxes": bboxes, "labels": labels}


class Normalize:
    """Normalize image to [0,1] range."""

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, **kwargs):
        image = image.astype(np.float32) / 255.0
        return {"image": image, "bboxes": bboxes, "labels": labels}


def build_base_transforms(image_size: int = 512) -> List:
    """Build standard base transform pipeline."""
    return [
        Resize(target_size=image_size),
        RandomFlip(h_prob=0.5),
        Normalize(),
    ]
