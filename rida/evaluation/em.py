"""Pixel-level evaluation of mitochondria segmentation on ``.npz`` slices.

All test pixels are pooled; mAP is computed from the foreground probability, F1, IoU and MCC from
the prediction at threshold 0.5.
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np
import torch

from ..datasets.transforms import normalize_image, to_tensor

__all__ = ["validate_em_slices", "test_em_slices", "pooled_binary_metrics"]


def pooled_binary_metrics(probs: np.ndarray, gts: np.ndarray, threshold: float = 0.5) -> dict:
    """mAP, F1, IoU and MCC (%) of pooled foreground probabilities against binary labels."""
    from sklearn.metrics import average_precision_score

    pred = probs >= threshold
    tp = float(np.count_nonzero(pred & gts))
    pred_pos = float(np.count_nonzero(pred))
    gt_pos = float(np.count_nonzero(gts))
    fp, fn = pred_pos - tp, gt_pos - tp
    tn = float(gts.size) - tp - fp - fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "mAP": 100.0 * float(average_precision_score(gts, probs)) if gts.any() else 0.0,
        "F1": 200.0 * tp / max(pred_pos + gt_pos, 1e-6),
        "IoU": 100.0 * tp / max(tp + fp + fn, 1e-6),
        "MCC": 100.0 * (tp * tn - fp * fn) / den if den > 0 else 0.0,
    }


@torch.no_grad()
def validate_em_slices(model, loader, device) -> dict:
    """Pooled foreground Dice of the evaluation split (used during training)."""
    model.eval()
    inter = pred_sum = gt_sum = 0
    for images, labels, _ in loader:
        pred = model(images.to(device))[0].argmax(dim=1).cpu().numpy() == 1
        gt = labels.numpy() == 1
        inter += int(np.logical_and(pred, gt).sum())
        pred_sum += int(pred.sum())
        gt_sum += int(gt.sum())
    dice = float(2.0 * inter / max(pred_sum + gt_sum, 1))
    return {"dice_mito": dice, "pred_gt_ratio_mito": pred_sum / max(gt_sum, 1), "mean_dice": dice, "score": dice}


@torch.no_grad()
def test_em_slices(model, root: str, device) -> dict:
    model.eval()
    probs, gts = [], []
    for f in sorted(glob.glob(os.path.join(root, "test", "*.npz"))):
        d = np.load(f)
        x = to_tensor(normalize_image(d["image"])).unsqueeze(0).to(device)
        probs.append(torch.softmax(model(x)[0], 1)[0, 1].float().cpu().numpy().ravel())
        gts.append((d["label"].astype(np.uint8) == 1).ravel())
    return pooled_binary_metrics(np.concatenate(probs), np.concatenate(gts))
