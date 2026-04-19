from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass
class AugmentationConfig:
    image_size: int = 224
    jitter: float = 0.12
    blur_prob: float = 0.2
    flip_prob: float = 0.5


def load_rgb_image(path: str | Path, image_size: int = 224) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[0] == 3:
        r, g, b = image
        return 0.299 * r + 0.587 * g + 0.114 * b
    if image.shape[-1] == 3:
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = image.mean(axis=(-2, -1), keepdims=True)
    std = image.std(axis=(-2, -1), keepdims=True) + 1e-6
    return (image - mean) / std


def random_flip(image: np.ndarray, rng: np.random.Generator, prob: float = 0.5) -> np.ndarray:
    if rng.random() < prob:
        image = image[..., ::-1]
    if rng.random() < prob:
        image = image[..., ::-1, :]
    return image


def random_jitter(image: np.ndarray, rng: np.random.Generator, strength: float = 0.12) -> np.ndarray:
    brightness = 1.0 + rng.uniform(-strength, strength)
    contrast = 1.0 + rng.uniform(-strength, strength)
    shifted = image * brightness
    mean = shifted.mean(axis=(-2, -1), keepdims=True)
    return np.clip((shifted - mean) * contrast + mean, 0.0, 1.0)


def random_blur(image: np.ndarray, rng: np.random.Generator, prob: float = 0.2) -> np.ndarray:
    if rng.random() >= prob:
        return image
    pil = Image.fromarray((np.transpose(np.clip(image, 0.0, 1.0), (1, 2, 0)) * 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.2))))
    return np.transpose(np.asarray(pil, dtype=np.float32) / 255.0, (2, 0, 1))


def random_crop(image: np.ndarray, rng: np.random.Generator, crop_size: int) -> np.ndarray:
    _, height, width = image.shape
    if crop_size >= min(height, width):
        return image
    top = int(rng.integers(0, height - crop_size + 1))
    left = int(rng.integers(0, width - crop_size + 1))
    cropped = image[:, top : top + crop_size, left : left + crop_size]
    pil = Image.fromarray((np.transpose(cropped, (1, 2, 0)) * 255).astype(np.uint8))
    pil = pil.resize((width, height), Image.BILINEAR)
    return np.transpose(np.asarray(pil, dtype=np.float32) / 255.0, (2, 0, 1))


def augment_view(image: np.ndarray, rng: np.random.Generator, cfg: AugmentationConfig) -> np.ndarray:
    out = random_flip(image, rng, prob=cfg.flip_prob)
    out = random_jitter(out, rng, strength=cfg.jitter)
    out = random_blur(out, rng, prob=cfg.blur_prob)
    crop_size = max(32, int(cfg.image_size * rng.uniform(0.7, 1.0)))
    out = random_crop(out, rng, crop_size=crop_size)
    return normalize_image(out)


def feature_vector(image: np.ndarray, bins: int = 16) -> np.ndarray:
    gray = to_grayscale(image)
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0), density=True)
    stats = np.array(
        [
            gray.mean(),
            gray.std(),
            gray.min(),
            gray.max(),
            np.percentile(gray, 25),
            np.percentile(gray, 50),
            np.percentile(gray, 75),
        ],
        dtype=np.float32,
    )
    return np.concatenate([hist.astype(np.float32), stats], axis=0)

