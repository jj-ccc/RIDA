"""Training utilities: seeding, schedules, EMA and batch iteration."""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn

__all__ = ["set_seed", "ramp_weight", "linear_schedule", "set_cosine_lr", "set_constant_lr", "update_ema",
           "freeze_batchnorm_stats", "next_batch", "class_weight_tensor"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ramp_weight(it: int, weight: float, ramp_iters: int) -> float:
    """Linear ramp of ``weight`` over the first ``ramp_iters`` steps."""
    if ramp_iters <= 0:
        return float(weight)
    return float(weight) * min(1.0, max(0.0, it / float(ramp_iters)))


def linear_schedule(value, it: int) -> float:
    """``value`` is a constant or ``[start, end, ramp_iters]``."""
    if isinstance(value, (list, tuple)):
        start, end, ramp = float(value[0]), float(value[1]), int(value[2])
        if ramp <= 0:
            return end
        return start + (end - start) * min(1.0, max(0.0, it / float(ramp)))
    return float(value)


def set_cosine_lr(optimizer, base_lr: float, it: int, total: int, warmup: int) -> float:
    """Linear warm-up followed by cosine decay to zero."""
    if warmup > 0 and it < warmup:
        lr = base_lr * it / max(1, warmup)
    else:
        prog = (it - warmup) / max(1, total - warmup)
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, prog)))
    for g in optimizer.param_groups:
        g["lr"] = lr * float(g.get("lr_mult", 1.0))
    return lr


def set_constant_lr(optimizer, base_lr: float) -> float:
    for g in optimizer.param_groups:
        g["lr"] = base_lr * float(g.get("lr_mult", 1.0))
    return base_lr


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
    for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
        ema_buf.copy_(buf)


def freeze_batchnorm_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def next_batch(iterator, loader):
    try:
        return iterator, next(iterator)
    except StopIteration:
        iterator = iter(loader)
        return iterator, next(iterator)


def class_weight_tensor(weights, device: torch.device):
    if weights is None:
        return None
    return torch.tensor(list(weights), dtype=torch.float32, device=device)
