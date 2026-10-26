"""Slice datasets stored as ``.npz`` files.

Each file holds an ``image`` array and a ``label`` array.  Files are named ``<case>_<index>.npz``;
the case prefix groups slices into volumes for evaluation.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import domain_randomize, normalize_image, random_geometry, strong_intensity, to_tensor

__all__ = ["NpzSlices", "case_and_index", "build_slice_loaders", "build_slice_eval_loader"]


def case_and_index(name: str) -> tuple[str, int]:
    if "_" not in name:
        return name, 0
    case, idx = name.rsplit("_", 1)
    try:
        return case, int(idx)
    except ValueError:
        return case, 0


class NpzSlices(Dataset):
    """Labelled source slices, weak/strong target pairs or evaluation slices.

    Args:
        root: domain directory containing the ``train`` and ``test`` splits.
        split: ``"train"`` or ``"test"``.
        mode: ``"source"`` (augmented, labelled), ``"target"`` (unlabelled weak and strong views) or ``"eval"``.
        num_classes: number of classes.
        geom_aug: random flips and rotations.
        strong_aug: probabilities ``(affine, gamma, noise, cutout)`` of the strong view.
    """

    def __init__(self, root: str | Path, split: str, mode: str, num_classes: int, geom_aug: bool = False,
                 domain_rand: bool = False, strong_aug=(0.8, 0.5, 0.5, 0.5)):
        self.root = Path(root)
        self.mode = mode
        self.num_classes = int(num_classes)
        self.geom_aug = bool(geom_aug)
        self.domain_rand = bool(domain_rand)
        self.strong_aug = tuple(float(p) for p in strong_aug)
        self.files = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz files found under {self.root / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        data = np.load(path)
        image = normalize_image(data["image"])
        name = path.stem
        if self.mode == "target":
            if self.geom_aug:
                image, _ = random_geometry(image, None)
            affine, gamma, noise, cutout = self.strong_aug
            strong = strong_intensity(image, affine_prob=affine, gamma_prob=gamma, noise_prob=noise,
                                      cutout_prob=cutout)
            return to_tensor(image), to_tensor(strong), name
        label = np.clip(data["label"].astype(np.int64), 0, self.num_classes - 1)
        if self.mode == "source":
            if self.geom_aug:
                image, label = random_geometry(image, label)
            if self.domain_rand:
                image = domain_randomize(image)
            elif random.random() < 0.5:
                image = strong_intensity(image)
        return to_tensor(image), torch.from_numpy(label.copy()).long(), name


def build_slice_loaders(data_cfg, sched, num_classes: int, seed: int, num_workers: int = 2):
    """Source and target training loaders with independent shuffling streams."""
    source = NpzSlices(data_cfg.source, "train", "source", num_classes, geom_aug=sched.geom_aug,
                       domain_rand=sched.domain_rand)
    target = NpzSlices(data_cfg.target, "train", "target", num_classes, geom_aug=sched.geom_aug,
                       strong_aug=getattr(sched, "strong_aug", (0.0, 0.0, 0.0, 0.0)))
    batch = int(sched.batch_size)
    if batch > min(len(source), len(target)):
        raise ValueError(f"batch_size={batch} exceeds the number of training slices")
    loaders = []
    for dataset, offset in ((source, 1009), (target, 2003)):
        generator = torch.Generator()
        generator.manual_seed(int(seed) + offset)
        loaders.append(DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers,
                                  pin_memory=torch.cuda.is_available(), generator=generator))
    return source, target, loaders[0], loaders[1]


def build_slice_eval_loader(root: str | Path, num_classes: int, batch_size: int, num_workers: int = 2):
    dataset = NpzSlices(root, "test", "eval", num_classes)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, drop_last=False,
                        num_workers=max(1, min(4, int(num_workers))))
    return dataset, loader
