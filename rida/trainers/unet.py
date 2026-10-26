"""U-Net trainer for the abdominal and cardiac benchmarks."""
from __future__ import annotations

from ..models import build_unet
from .slice_trainer import SliceTrainer

__all__ = ["UNetTrainer"]


class UNetTrainer(SliceTrainer):
    """All parameters of the U-Net are trained throughout."""

    def build_model(self):
        return build_unet(self.cfg.model, self.num_classes)
