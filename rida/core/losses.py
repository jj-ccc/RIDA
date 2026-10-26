"""Supervised and self-training objectives."""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["dice_loss", "supervised_loss", "masked_cross_entropy", "soft_cross_entropy", "pseudo_label_loss"]


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Foreground soft Dice loss."""
    probs = torch.softmax(logits, dim=1)[:, 1:]
    target_oh = F.one_hot(target.clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()[:, 1:]
    dims = (0, 2, 3)
    inter = (probs * target_oh).sum(dim=dims)
    denom = probs.sum(dim=dims) + target_oh.sum(dim=dims)
    return 1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5)).mean()


def supervised_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int,
                    class_weight: torch.Tensor | None, dice_weight: float = 1.0) -> torch.Tensor:
    """Weighted cross-entropy plus foreground Dice on labelled source images."""
    ce = F.cross_entropy(logits, target, weight=class_weight)
    return ce + float(dice_weight) * dice_loss(logits, target, num_classes)


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                         class_weight: torch.Tensor | None = None) -> torch.Tensor:
    valid = (target >= 0) & (target != 255) & (mask > 0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits, target.clamp_max(logits.shape[1] - 1), weight=class_weight, reduction="none")
    weights = mask.to(device=logits.device, dtype=loss.dtype)
    denom = weights[valid].sum()
    if float(denom.detach().item()) <= 0.0:
        return logits.sum() * 0.0
    return (loss[valid] * weights[valid]).sum() / denom.clamp_min(1e-6)


def soft_cross_entropy(logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor,
                       class_weight: torch.Tensor | None = None, temperature: float = 1.0) -> torch.Tensor:
    temp = max(float(temperature), 1e-3)
    with torch.no_grad():
        target_prob = torch.softmax(teacher_logits / temp, dim=1)
        hard_target = target_prob.argmax(dim=1)
    loss = -(target_prob * torch.log_softmax(logits / temp, dim=1)).sum(dim=1) * (temp * temp)
    weights = mask.to(device=loss.device, dtype=loss.dtype)
    if class_weight is not None:
        weights = weights * class_weight[hard_target]
    denom = weights.sum()
    if float(denom.detach().item()) <= 0.0:
        return logits.sum() * 0.0
    return (loss * weights).sum() / denom.clamp_min(1e-6)


def pseudo_label_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, pseudo: torch.Tensor,
                      weights: torch.Tensor, target: str, class_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Rank-stability weighted self-training towards the EMA teacher.

    ``target="soft"`` distils the teacher distribution at temperature 4, ``target="hard"`` uses the
    teacher's arg-max label.
    """
    if target == "soft":
        return soft_cross_entropy(student_logits, teacher_logits.detach(), weights, class_weight, 4.0)
    return masked_cross_entropy(student_logits, pseudo.detach(), weights, class_weight)
