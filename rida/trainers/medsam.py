"""MedSAM trainer: the image encoder is trained through its q/v LoRA adapters and the neck."""
from __future__ import annotations

import torch

from ..models import build_medsam
from .slice_trainer import SliceTrainer

__all__ = ["MedSAMTrainer"]


class MedSAMTrainer(SliceTrainer):

    def build_model(self):
        return build_medsam(self.cfg.model, self.num_classes)

    def set_trainable(self, warmup: bool) -> None:
        frozen = (not warmup) and bool(self.cfg.train.freeze_encoder)
        for name, param in self.model.named_parameters():
            param.requires_grad_(self.trainable[name] and not (frozen and name.startswith("encoder.")))

    def param_groups(self):
        mult = float(self.cfg.train.encoder_lr_mult)
        lr = float(self.cfg.train.lr)
        enc = [p for n, p in self.model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
        rest = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
        if mult == 1.0 or not enc:
            return [{"params": enc + rest, "lr": lr, "lr_mult": 1.0}]
        return [{"params": rest, "lr": lr, "lr_mult": 1.0}, {"params": enc, "lr": lr * mult, "lr_mult": mult}]

    @torch.no_grad()
    def teacher_forward(self, weak: torch.Tensor):
        self.teacher.eval()
        logits, feat, low = self.teacher(weak)
        if not bool(self.cfg.train.flip_teacher):
            return logits, feat, low
        probs = torch.softmax(logits.float(), 1)
        for dims in ((-1,), (-2,), (-2, -1)):
            flipped = self.teacher(torch.flip(weak, dims))[0]
            probs = probs + torch.flip(torch.softmax(flipped.float(), 1), dims)
        return torch.log((probs / 4).clamp_min(1e-7)).to(logits.dtype), feat, low
