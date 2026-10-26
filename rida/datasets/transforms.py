"""Image normalisation and augmentation for 2-D slices (``H x W x C`` float arrays in ``[0, 1]``)."""
from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["normalize_image", "random_geometry", "strong_intensity", "bezier_intensity",
           "domain_randomize", "to_tensor"]


def _as_hwc(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Map a stored slice to ``[0, 1]``."""
    image = _as_hwc(image).astype(np.float32)
    finite = np.isfinite(image)
    if not finite.all():
        image = np.where(finite, image, 0.0)
    mn, mx = float(image.min()), float(image.max())
    if mx > 1.5:
        image = image / 255.0
    elif mn < 0.0 or mx > 1.0:
        image = (image - mn) / max(mx - mn, 1e-6)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def random_geometry(image: np.ndarray, label: np.ndarray | None):
    """Random flips and 90-degree rotations, applied identically to image and label."""
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
        if label is not None:
            label = np.flip(label, axis=1)
    if random.random() < 0.5:
        image = np.flip(image, axis=0)
        if label is not None:
            label = np.flip(label, axis=0)
    k = random.randint(0, 3)
    if k:
        image = np.rot90(image, k=k, axes=(0, 1))
        if label is not None:
            label = np.rot90(label, k=k, axes=(0, 1))
    return np.ascontiguousarray(image), None if label is None else np.ascontiguousarray(label)


def strong_intensity(image: np.ndarray, *, affine_prob: float = 0.8, gamma_prob: float = 0.5,
                     noise_prob: float = 0.5, cutout_prob: float = 0.5) -> np.ndarray:
    """Photometric perturbation of the strong view."""
    out = image.astype(np.float32, copy=True)
    if random.random() < float(affine_prob):
        out = out * random.uniform(0.75, 1.25) + random.uniform(-0.15, 0.15)
    if random.random() < float(gamma_prob):
        out = np.power(np.clip(out, 0.0, 1.0), random.uniform(0.7, 1.5))
    if random.random() < float(noise_prob):
        out = out + np.random.normal(0.0, random.uniform(0.0, 0.08), size=out.shape).astype(np.float32)
    if random.random() < float(cutout_prob):
        h, w = out.shape[:2]
        cut_h = random.randint(max(1, h // 16), max(2, h // 6))
        cut_w = random.randint(max(1, w // 16), max(2, w // 6))
        y = random.randint(0, max(0, h - cut_h))
        x = random.randint(0, max(0, w - cut_w))
        out[y:y + cut_h, x:x + cut_w, :] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def bezier_intensity(image: np.ndarray) -> np.ndarray:
    p1 = (random.random(), random.random())
    p2 = (random.random(), random.random())
    t = np.linspace(0.0, 1.0, 1024)
    xpts = 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3
    ypts = 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3
    order = np.argsort(xpts)
    xpts, ypts = xpts[order], ypts[order]
    if random.random() < 0.5:
        ypts = ypts[::-1]
    return np.interp(np.clip(image, 0.0, 1.0), xpts, ypts).astype(np.float32)


def domain_randomize(image: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32, copy=True)
    if random.random() < 0.5:
        out = 1.0 - out
    if random.random() < 0.9:
        out = bezier_intensity(out)
    if random.random() < 0.5:
        out = np.power(np.clip(out, 0.0, 1.0), random.uniform(0.4, 2.5))
    if random.random() < 0.3:
        out = out + np.random.normal(0.0, random.uniform(0.0, 0.05), size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float()
