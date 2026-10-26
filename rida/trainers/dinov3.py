"""DINOv3 trainer."""
from __future__ import annotations

from ..models import build_dinov3
from .slice_trainer import SliceTrainer

__all__ = ["DINOv3Trainer"]


class DINOv3Trainer(SliceTrainer):

    def build_model(self):
        return build_dinov3(self.cfg.model, self.num_classes)

    def set_trainable(self, warmup: bool) -> None:
        for name, param in self.model.named_parameters():
            frozen = warmup and name.startswith("encoder.")
            param.requires_grad_(self.trainable[name] and not frozen)
        self.model.freeze_encoder = True if warmup else bool(self.cfg.model.freeze_encoder)

    def param_groups(self):
        mult = float(self.cfg.train.encoder_lr_mult)
        lr = float(self.cfg.train.lr)
        enc = [p for n, p in self.model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
        rest = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
        if mult == 1.0 or not enc:
            return [{"params": enc + rest, "lr": lr, "lr_mult": 1.0}]
        return [{"params": rest, "lr": lr, "lr_mult": 1.0}, {"params": enc, "lr": lr * mult, "lr_mult": mult}]
