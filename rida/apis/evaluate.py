"""Evaluation entry points of the three protocols."""
from __future__ import annotations

from ..evaluation import test_em_slices, test_pairs, test_volumes

__all__ = ["evaluate_medical", "evaluate_em", "evaluate_pairs", "evaluate"]


def evaluate_medical(cfg, model, device="cuda") -> dict:
    """Abdominal / cardiac: Dice and ASSD per organ on the target test volumes."""
    return test_volumes(model, cfg.data.target, cfg.classes, device)


def evaluate_em(cfg, model, device="cuda") -> dict:
    """VNC III -> Lucchi and MitoEM on single sections: pooled mAP / F1 / MCC / IoU."""
    return test_em_slices(model, cfg.data.target, device)


def evaluate_pairs(cfg, model, device="cuda") -> dict:
    """U-Net on pairs of adjacent sections: pooled mAP / F1 / MCC / IoU."""
    return test_pairs(model, cfg.data, device, int(cfg.runtime.num_workers))


def evaluate(cfg, model, device="cuda") -> dict:
    if bool(getattr(cfg.model, "paired", False)):
        return evaluate_pairs(cfg, model, device)
    if cfg.evaluation == "medical":
        return evaluate_medical(cfg, model, device)
    return evaluate_em(cfg, model, device)
