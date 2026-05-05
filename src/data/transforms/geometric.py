"""Geometric augmentation transforms."""

from typing import List

import numpy as np


class RandomRotate:
    """Rotate image by 0, 90, 180, or 270 degrees."""

    def __init__(self, k_options: List[int] = None):
        self.k_options = k_options or [0, 1, 2, 3]

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, **kwargs):
        k = self.k_options[np.random.randint(len(self.k_options))]
        if k == 0:
            return {"image": image, "bboxes": bboxes, "labels": labels}

        h, w = image.shape[:2]
        image = np.rot90(image, k=k)

        if len(bboxes) == 0:
            new_bboxes = bboxes
        else:
            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

            if k == 1:
                new_bboxes = np.stack([y1, w - x2, y2, w - x1], axis=1)
            elif k == 2:
                new_bboxes = np.stack([w - x2, h - y2, w - x1, h - y1], axis=1)
            else:
                new_bboxes = np.stack([h - y2, x1, h - y1, x2], axis=1)

            new_h, new_w = image.shape[:2]
            new_bboxes[:, [0, 2]] = np.clip(new_bboxes[:, [0, 2]], 0, new_w)
            new_bboxes[:, [1, 3]] = np.clip(new_bboxes[:, [1, 3]], 0, new_h)

        return {"image": image, "bboxes": new_bboxes, "labels": labels}


class RandomCrop:
    """Random crop preserving at least one bbox."""

    def __init__(self, min_crop_ratio: float = 0.5, max_attempts: int = 10):
        self.min_crop_ratio = min_crop_ratio
        self.max_attempts = max_attempts

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, **kwargs):
        h, w = image.shape[:2]

        for _ in range(self.max_attempts):
            crop_ratio = np.random.uniform(self.min_crop_ratio, 1.0)
            crop_w = int(w * crop_ratio)
            crop_h = int(h * crop_ratio)
            x = np.random.randint(0, max(1, w - crop_w))
            y = np.random.randint(0, max(1, h - crop_h))

            if len(bboxes) == 0:
                break

            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            cx1, cy1 = np.maximum(x1, x), np.maximum(y1, y)
            cx2, cy2 = np.minimum(x2, x + crop_w), np.minimum(y2, y + crop_h)
            valid = (cx2 > cx1) & (cy2 > cy1)

            if valid.any():
                break

        cropped = image[y : y + crop_h, x : x + crop_w]
        if len(bboxes) > 0 and valid.any():
            new_bboxes = np.stack([cx1 - x, cy1 - y, cx2 - x, cy2 - y], axis=1)[valid]
            new_labels = labels[valid]
        else:
            new_bboxes = np.zeros((0, 4), dtype=bboxes.dtype)
            new_labels = np.array([], dtype=labels.dtype)

        return {"image": cropped, "bboxes": new_bboxes, "labels": new_labels}


class Mosaic:
    """Mosaic augmentation: combine 4 images into one."""

    def __init__(self, dataset, index_list: List[int]):
        self.dataset = dataset
        self.index_list = index_list

    def __call__(self, base_image, base_bboxes, base_labels, **kwargs):
        images = [base_image]
        h, w = base_image.shape[:2]

        extra_indices = np.random.choice(self.index_list, 3, replace=False)
        for idx in extra_indices:
            img = self.dataset[idx]
            if hasattr(img, "image"):
                img = (img.image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(img)

        quarter_h, quarter_w = h // 2, w // 2
        quadrants = [
            images[0][:quarter_h, :quarter_w],
            images[1][:quarter_h, quarter_w:],
            images[2][quarter_h:, :quarter_w],
            images[3][quarter_h:, quarter_w:],
        ]
        mosaic = np.concatenate(
            [np.concatenate(quadrants[:2], axis=1), np.concatenate(quadrants[2:], axis=1)],
            axis=0,
        )
        return {"image": mosaic, "bboxes": base_bboxes, "labels": base_labels}
