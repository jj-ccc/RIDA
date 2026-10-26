"""Pairs of adjacent electron-microscopy sections.

A sample is a pair of sections ``(i, i + stride)`` of one volume.  Volumes are read either from an
HDF5 stack (MitoEM, dataset key ``main``) or from a directory of images listed in a text file
(VNC III, Lucchi).  Training samples are random native-resolution crops; evaluation visits every
pair at full resolution.
"""
from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import domain_randomize, random_geometry, strong_intensity, to_tensor

__all__ = ["PairedSlices", "build_paired_loaders", "build_paired_eval_loader"]

LUCCHI_PAD = (176, 48)


class PairedSlices(Dataset):
    """Pairs of neighbouring EM slices.

    Args:
        data_cfg: data section of the configuration.
        sched: training schedule section.
        role: ``"source"``, ``"target"`` or ``"eval"``.
    """

    def __init__(self, data_cfg, sched, role: str):
        self.data_cfg, self.sched, self.role = data_cfg, sched, role
        d = data_cfg
        if role == "source":
            image, label, listing, self.stride = d.source_images, d.source_labels, getattr(d, "source_list", None), int(d.source_stride)
            self.crop = int(d.crop_size)
        elif role == "target":
            image, label, listing, self.stride = d.target_images, None, getattr(d, "target_list", None), int(d.target_stride)
            self.crop = int(d.target_crop_size)
        else:
            image, label, listing, self.stride = d.test_images, d.test_labels, getattr(d, "test_list", None), int(d.target_stride)
            self.crop = int(d.target_crop_size)
        self.image_root = Path(image)
        self.label_root = Path(label) if label else None
        self.hdf = self.image_root.suffix.lower() in (".hdf", ".h5", ".hdf5")
        self._hdf_files = {}
        if self.hdf:
            with h5py.File(image, "r") as f:
                self.length, *self.shape = f["main"].shape
        else:
            self.names = [s.strip() for s in Path(listing).read_text().splitlines() if s.strip()]
            self.length = len(self.names)
        self.eval_size = int(d.test_size)
        self.pad = LUCCHI_PAD if role == "eval" and d.kind == "lucchi" else (0, 0)

    def __len__(self) -> int:
        return self.length - self.stride

    def raw(self, index: int, label: bool = False, region=None) -> np.ndarray:
        if self.hdf:
            root = self.label_root if label else self.image_root
            if root not in self._hdf_files:
                self._hdf_files[root] = h5py.File(root, "r")
            if region is None:
                region = (slice(0, self.eval_size),) * 2 if self.role == "eval" else (slice(None),) * 2
            array = self._hdf_files[root]["main"][(index, *region)]
            if self.role == "eval":
                array = array[: self.eval_size, : self.eval_size]
        else:
            root = self.label_root if label else self.image_root
            with Image.open(root / self.names[index]) as img:
                array = np.asarray(img)
        return (array > 0).astype(np.int64) if label else array.astype(np.float32)

    def raw_label(self, index: int) -> np.ndarray:
        """Stored annotation of an evaluation slice, unmodified."""
        if self.hdf:
            if self.label_root not in self._hdf_files:
                self._hdf_files[self.label_root] = h5py.File(self.label_root, "r")
            array = self._hdf_files[self.label_root]["main"][index, : self.eval_size, : self.eval_size]
        else:
            with Image.open(self.label_root / self.names[index]) as img:
                array = np.asarray(img)
        return np.asarray(array)

    @staticmethod
    def normalize(array: np.ndarray) -> np.ndarray:
        return (array - array.min()) / max(float(array.max() - array.min()), 1e-6)

    def __getitem__(self, index: int):
        if self.role != "eval":
            index = random.randrange(len(self))
        region = None
        if self.hdf and self.role != "eval":
            y, x = random.randint(0, self.shape[0] - self.crop), random.randint(0, self.shape[1] - self.crop)
            region = (slice(y, y + self.crop), slice(x, x + self.crop))
        image = np.stack([self.normalize(self.raw(index, region=region)),
                          self.normalize(self.raw(index + self.stride, region=region))], axis=-1)
        label = None
        if self.role == "source":
            label = np.stack([self.raw(index, True, region), self.raw(index + self.stride, True, region)], axis=-1)
        if self.role == "eval":
            py, px = self.pad
            if py or px:
                image = np.pad(image, ((py, py), (px, px), (0, 0)), mode="reflect")
            return to_tensor(image), index
        h, w = image.shape[:2]
        y, x = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
        image = image[y:y + self.crop, x:x + self.crop]
        if label is not None:
            label = label[y:y + self.crop, x:x + self.crop]
        if bool(self.sched.geom_aug):
            image, label = random_geometry(image, label)
        if self.role == "source":
            if bool(self.sched.domain_rand):
                image = domain_randomize(image)
            masks = np.stack([label[..., 0], label[..., 1], label[..., 0] ^ label[..., 1]])
            return to_tensor(image), torch.from_numpy(masks.copy()).long(), index
        affine, gamma, noise, cutout = self.sched.strong_aug
        strong = strong_intensity(image, affine_prob=affine, gamma_prob=gamma, noise_prob=noise, cutout_prob=cutout)
        return to_tensor(image), to_tensor(strong), index


def build_paired_loaders(data_cfg, sched, seed: int, num_workers: int = 2):
    """Source and target loaders that draw random pairs with replacement."""
    source = PairedSlices(data_cfg, sched, "source")
    target = PairedSlices(data_cfg, sched, "target")
    loaders = []
    for dataset, offset in ((source, 1009), (target, 2003)):
        generator = torch.Generator().manual_seed(int(seed) + offset)
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=400000, generator=generator)
        loaders.append(DataLoader(dataset, batch_size=int(sched.batch_size), sampler=sampler, drop_last=True,
                                  num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=generator))
    return source, target, loaders[0], loaders[1]


def build_paired_eval_loader(data_cfg, batch_size: int = 1, num_workers: int = 2):
    dataset = PairedSlices(data_cfg, None, "eval")
    return dataset, DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=num_workers)
