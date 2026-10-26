"""Volumetric evaluation of the abdominal and cardiac benchmarks.

Slices are stacked into volumes in slice order, every organ keeps its largest 3-D connected
component, and Dice and the symmetric average surface distance (ASSD, voxels) are averaged over
cases.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from scipy.ndimage import label as connected_components

from ..datasets.slices import case_and_index
from ..datasets.transforms import normalize_image, to_tensor

__all__ = ["keep_largest_component", "validate_volumes", "test_volumes"]


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return mask
    labels, count = connected_components(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def _dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    denom = int(pred.sum() + target.sum())
    return float("nan") if denom == 0 else float(2.0 * np.logical_and(pred, target).sum() / denom)


@torch.no_grad()
def validate_volumes(model, loader, device, num_classes: int, classes, largest_cc: bool = True) -> dict:
    """Mean Dice over volumes of the evaluation split (used during training)."""
    model.eval()
    grouped: dict[str, list] = {}
    for images, labels, names in loader:
        logits = model(images.to(device))[0]
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        labels_np = labels.numpy().astype(np.uint8)
        for i, name in enumerate(names):
            case, idx = case_and_index(str(name))
            grouped.setdefault(case, []).append((idx, preds[i], labels_np[i]))
    per_class = {c: [] for c in range(1, num_classes)}
    counts = {c: [0, 0, 0] for c in range(1, num_classes)}
    for rows in grouped.values():
        rows.sort(key=lambda x: x[0])
        pred_vol = np.stack([r[1] for r in rows])
        lab_vol = np.stack([r[2] for r in rows])
        for c in range(1, num_classes):
            pred_mask = pred_vol == c
            if largest_cc:
                pred_mask = keep_largest_component(pred_mask)
            gt_mask = lab_vol == c
            if num_classes > 2 and not gt_mask.any():
                continue
            counts[c][0] += int(np.logical_and(pred_mask, gt_mask).sum())
            counts[c][1] += int(pred_mask.sum())
            counts[c][2] += int(gt_mask.sum())
            per_class[c].append(_dice(pred_mask, gt_mask))
    metrics, means = {}, []
    for c in range(1, num_classes):
        values = np.array(per_class[c], dtype=np.float64)
        mean = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
        name = classes[c - 1]
        hit, n_pred, n_gt = counts[c]
        metrics[f"dice_{name}"] = mean
        metrics[f"pred_gt_ratio_{name}"] = n_pred / max(n_gt, 1)
        means.append(mean)
    metrics["mean_dice"] = float(np.mean(means)) if means else 0.0
    metrics["score"] = metrics["mean_dice"]
    return metrics


@torch.no_grad()
def test_volumes(model, root: str, classes, device) -> dict:
    """Dice (%) and ASSD (voxels) per organ and their averages on ``root/test``."""
    from medpy.metric.binary import asd, dc

    model.eval()
    cases: dict[str, list] = {}
    for f in sorted(glob.glob(os.path.join(root, "test", "*.npz"))):
        case, index = case_and_index(os.path.basename(f)[:-4])
        cases.setdefault(case, []).append((index, f))
    dsc = {c: [] for c in range(1, len(classes) + 1)}
    assd = {c: [] for c in range(1, len(classes) + 1)}
    for files in cases.values():
        preds, labs = [], []
        for _, f in sorted(files, key=lambda item: item[0]):
            d = np.load(f)
            x = to_tensor(normalize_image(d["image"])).unsqueeze(0).to(device)
            preds.append(model(x)[0].argmax(1)[0].cpu().numpy().astype(np.uint8))
            labs.append(d["label"].astype(np.uint8))
        pv, lv = np.stack(preds), np.stack(labs)
        for c in dsc:
            p, g = keep_largest_component(pv == c), (lv == c)
            if g.sum() == 0:
                continue
            dsc[c].append(dc(p, g) * 100.0)
            if p.sum() > 0:
                try:
                    assd[c].append(0.5 * (float(asd(p, g)) + float(asd(g, p))))
                except Exception:
                    assd[c].append(np.nan)
            else:
                assd[c].append(np.nan)
    res = {}
    for c, name in enumerate(classes, start=1):
        res[f"Dice_{name}"] = float(np.nanmean(dsc[c])) if dsc[c] else float("nan")
        res[f"ASSD_{name}"] = float(np.nanmean(assd[c])) if assd[c] else float("nan")
    d_vals = [res[f"Dice_{n}"] for n in classes if not np.isnan(res[f"Dice_{n}"])]
    a_vals = [res[f"ASSD_{n}"] for n in classes if not np.isnan(res[f"ASSD_{n}"])]
    res["Dice"] = float(np.mean(d_vals)) if d_vals else float("nan")
    res["ASSD"] = float(np.nanmean(a_vals)) if a_vals else float("nan")
    return res
