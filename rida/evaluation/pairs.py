"""Evaluation of the U-Net on pairs of adjacent sections.

Every section pair ``(i, i + stride)`` is segmented once; the two slice heads contribute the
foreground probability of their slice and overlapping contributions are averaged.  Annotations
are binarised as ``label // 255`` and all pixels of the test volume are pooled.
"""
from __future__ import annotations

import numpy as np
import torch

from ..datasets.em_volumes import build_paired_eval_loader
from .em import pooled_binary_metrics

__all__ = ["predict_volume", "validate_pairs", "test_pairs"]


@torch.no_grad()
def predict_volume(model, loader, device) -> np.ndarray:
    """Foreground probability of every slice of the evaluation volume."""
    model.eval()
    data = loader.dataset
    shape = data.raw(0, True).shape
    sums = np.zeros((data.length, *shape), np.float32)
    counts = np.zeros(data.length, np.int32)
    for image, indices in loader:
        outputs = model(image.to(device))
        probabilities = [torch.softmax(outputs[i], 1)[:, 1].cpu().numpy() for i in (0, 1)]
        py, px = data.pad
        for b, index in enumerate(indices.tolist()):
            for head, at in ((0, index), (1, index + data.stride)):
                prob = probabilities[head][b]
                if py or px:
                    prob = prob[py:py + shape[0], px:px + shape[1]]
                sums[at] += prob
                counts[at] += 1
    return sums / np.maximum(counts, 1)[:, None, None]


@torch.no_grad()
def validate_pairs(model, loader, device) -> dict:
    """Mean slice Dice and pooled Dice of the evaluation volume (used during training)."""
    data = loader.dataset
    probs = predict_volume(model, loader, device)
    per_dice, ratios = [], []
    inter = pred_sum = gt_sum = 0
    smooth = 1e-6 if data.data_cfg.kind == "lucchi" else 0.0
    for i in range(data.length):
        pred = probs[i] >= 0.5
        gt = data.raw(i, True) > 0
        tp, pp, gp = int((pred & gt).sum()), int(pred.sum()), int(gt.sum())
        per_dice.append((2 * tp + smooth) / (pp + gp + smooth) if pp + gp + smooth else 0.0)
        ratios.append(pp / max(gp, 1))
        inter += tp
        pred_sum += pp
        gt_sum += gp
    score = float(np.mean(per_dice))
    return {"dice_mito": score, "pred_gt_ratio_mito": float(np.mean(ratios)), "mean_dice": score, "score": score,
            "pooled_dice": 2 * inter / max(pred_sum + gt_sum, 1)}


@torch.no_grad()
def test_pairs(model, data_cfg, device, num_workers: int = 2) -> dict:
    dataset, loader = build_paired_eval_loader(data_cfg, batch_size=1, num_workers=num_workers)
    probs = predict_volume(model, loader, device)
    gts = np.stack([(dataset.raw_label(i).astype(np.float32) / 255.0).astype(np.uint8) > 0
                    for i in range(dataset.length)])
    return pooled_binary_metrics(probs.reshape(-1), gts.reshape(-1))
