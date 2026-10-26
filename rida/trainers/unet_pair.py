"""Trainer of the U-Net on pairs of adjacent electron-microscopy sections.

The network predicts both sections of a pair and their change map.  The supervised and the
self-training terms are averaged over the three heads; the image-anchor similarities are read from the
bottleneck and the rank signatures from the decoder feature of the first section.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core import RankMatcher, dice_loss, pseudo_label_loss, rank_terms, anchor_similarities
from ..core.anchors import ortho_anchors
from ..datasets import build_paired_eval_loader, build_paired_loaders
from ..evaluation import validate_pairs
from ..models import build_pair_unet
from ..utils import class_weight_tensor, linear_schedule, ramp_weight
from .base import BaseTrainer

__all__ = ["PairUNetTrainer"]


class PairUNetTrainer(BaseTrainer):

    def build_model(self):
        return build_pair_unet(self.cfg.model, self.num_classes)

    def build_train_loaders(self, sched):
        return build_paired_loaders(self.cfg.data, sched, int(getattr(sched, "loader_seed", getattr(sched, "seed", self.cfg.seed))), int(self.cfg.runtime.num_workers))

    def build_eval_loader(self):
        return build_paired_eval_loader(self.cfg.data, int(self.cfg.train.eval_batch_size),
                                        int(self.cfg.runtime.num_workers))[1]

    def validate(self, model):
        return validate_pairs(model, self.eval_loader, self.device)

    def _supervised(self, outputs, labels, class_weight):
        ce = torch.stack([F.cross_entropy(outputs[i], labels[:, i], weight=class_weight) for i in range(3)]).mean()
        dice = torch.stack([dice_loss(outputs[i], labels[:, i], 2) for i in range(3)]).mean()
        return ce + float(self.cfg.loss.dice_weight) * dice

    def warmup_step(self, source_batch, class_weight):
        x, labels, _ = source_batch
        outputs = self.model(x.to(self.device))
        sup = self._supervised(outputs, labels.to(self.device), class_weight)
        return sup, {"loss": float(sup.detach()), "sup": float(sup.detach()), "rc": 0.0, "rs": 0.0, "rm": 0.0}

    def _matcher(self) -> RankMatcher:
        if self.matcher is None:
            rida = self.cfg.rida
            self.matcher = RankMatcher(self.num_classes, int(rida.num_anchors), float(rida.gate_quantile),
                                       seed=int(self.cfg.seed))
            if getattr(self, "_pending_matcher", None):
                self.matcher.load_state_dict(self._pending_matcher, self.device)
                self._pending_matcher = None
        return self.matcher

    def train_step(self, source_batch, target_batch, class_weight, it: int, sched):
        rida, dev = self.cfg.rida, self.device
        x, labels, _ = source_batch
        x, labels = x.to(dev), labels.to(dev)
        source = self.model(x)
        sup = self._supervised(source, labels, class_weight)
        total = linear_schedule(sched.sup_weight, it) * sup

        weak_x, strong_x, _ = target_batch
        weak_x = weak_x.to(dev)
        self.teacher.eval()
        with torch.no_grad():
            weak = self.teacher(weak_x)
            pseudo = [torch.softmax(weak[i], dim=1).argmax(dim=1) for i in range(3)]
        strong = self.model(strong_x.to(dev))
        ramp = int(sched.ramp_iters)
        k, q = int(rida.num_anchors), float(rida.gate_quantile)

        idx = ortho_anchors(weak[6].detach(), k)
        loss_rc, stab, keep = rank_terms(anchor_similarities(weak[6].detach(), idx), anchor_similarities(strong[6], idx), q)
        loss_rc = loss_rc * ramp_weight(it, float(rida.lambda_rc), ramp)
        stab = F.interpolate(stab[:, None], size=strong[0].shape[-2:], mode="bilinear", align_corners=False)[:, 0].detach()
        pl_cw = getattr(sched, "pl_class_weights", None)
        if pl_cw is None:
            pl_cw = getattr(self.cfg.data, "pl_class_weights", None)
        pl_weight = class_weight_tensor(pl_cw, dev)
        loss_rs = torch.stack([pseudo_label_loss(strong[i], weak[i], pseudo[i], stab, str(sched.pl_target), pl_weight)
                               for i in range(3)]).mean()
        loss_rs = loss_rs * ramp_weight(it, float(rida.lambda_rs), ramp)

        loss_rm = strong[0].sum() * 0.0
        if float(rida.lambda_rm) > 0.0:
            matcher = self._matcher()
            with torch.no_grad():
                t_src = self.teacher(x)[3]
            matcher.update(it, t_src, labels[:, 0])
            loss_rm, _ = matcher.loss(weak[3].detach(), strong[0])
            loss_rm = loss_rm * ramp_weight(it, float(rida.lambda_rm), ramp)

        total = total + loss_rc + loss_rs + loss_rm
        logs = {"loss": float(total.detach()), "sup": float(sup.detach()), "rc": float(loss_rc.detach()),
                "rs": float(loss_rs.detach()), "rm": float(loss_rm.detach()), "stable": float(keep)}
        return total, logs
