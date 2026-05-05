"""Domain randomization and weather simulation transforms."""

from typing import List

import cv2
import numpy as np


class ColorJitter:
    """Color jittering: brightness, contrast, saturation, hue."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        prob: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, **kwargs):
        if np.random.rand() > self.prob:
            return {"image": image, "bboxes": bboxes, "labels": labels}

        image = image.astype(np.float32)

        if self.brightness > 0:
            factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            image = np.clip(image * factor, 0, 255)

        if self.contrast > 0:
            factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            gray = np.mean(image, axis=2, keepdims=True)
            image = np.clip((image - gray) * factor + gray, 0, 255)

        if self.saturation > 0:
            hsv = self._rgb_to_hsv(image)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + np.random.uniform(-self.saturation, self.saturation)), 0, 1)
            image = self._hsv_to_rgb(hsv)

        return {"image": np.clip(image, 0, 255).astype(np.uint8), "bboxes": bboxes, "labels": labels}

    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        rgb = rgb / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        v = np.maximum(np.maximum(r, g), b)
        c = v - np.minimum(np.minimum(r, g), b)
        s = np.where(v == 0, 0, c / v)
        h = np.where(
            c == 0,
            0,
            np.where(v == r, (g - b) / c, np.where(v == g, 2 + (b - r) / c, 4 + (r - g) / c)),
        )
        h = ((h / 6.0) % 1.0) * 360.0
        return np.stack([h, s, v * 255.0], axis=2)

    def _hsv_to_rgb(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[:, :, 0] / 360.0, hsv[:, :, 1], hsv[:, :, 2] / 255.0
        c = v * s
        x = c * (1 - np.abs((h * 6) % 2 - 1))
        m = v - c
        rgb = np.zeros_like(hsv[:, :, :3])
        idx = ((h * 6).astype(int) % 6).astype(int)
        mappings = [(1, 2), (0, 2), (2, 0), (2, 1), (1, 0), (0, 1)]
        for i in range(6):
            mask = idx == i
            if mask.any():
                rgb[mask, mappings[i][0]] = c[mask]
                rgb[mask, mappings[i][1]] = x[mask]
        return (rgb + m[:, :, None]) * 255.0


class GaussianBlur:
    """Gaussian blur simulation."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, prob: float = 0.3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, **kwargs):
        if np.random.rand() > self.prob:
            return {"image": image, "bboxes": bboxes, "labels": labels}
        return {
            "image": cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma),
            "bboxes": bboxes,
            "labels": labels,
        }


class WeatherSimulation:
    """Simulate fog and rain overlay."""

    def __init__(self, fog_prob: float = 0.2, rain_prob: float = 0.1):
        self.fog_prob = fog_prob
        self.rain_prob = rain_prob

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, **kwargs):
        image = image.astype(np.float32)

        if np.random.rand() < self.fog_prob:
            fog_strength = np.random.uniform(0.1, 0.3)
            fog = np.random.uniform(200, 255, (image.shape[0], image.shape[1], 1))
            image = np.clip(image * (1 - fog_strength) + fog * fog_strength, 0, 255)

        if np.random.rand() < self.rain_prob:
            rain_strength = np.random.uniform(0.05, 0.15)
            rain_mask = np.random.rand(*image.shape[:2]) < rain_strength
            image[rain_mask] = np.clip(image[rain_mask] * 0.9, 0, 255)

        return {"image": image.astype(np.uint8), "bboxes": bboxes, "labels": labels}


def build_domain_rand_transforms() -> List:
    """Build domain randomization transform pipeline."""
    return [
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.5),
        GaussianBlur(prob=0.3),
        WeatherSimulation(fog_prob=0.1, rain_prob=0.1),
    ]
