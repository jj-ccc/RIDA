"""RIDA on single-slice segmentation networks (U-Net, DINOv3, MedSAM).

Per training step:

* ``L_sup`` -- cross-entropy + Dice on a labelled source batch;
* ``L_rc``  -- hard order matching of the similarities to orthogonal image anchors between the weak and the strong
  view of a target batch on the trusted pairs (weak-view gap above its ``q``-quantile);
* ``L_rs``  -- self-training towards the EMA teacher, every pixel weighted by its rank stability on
  the same trusted pairs;
* ``L_rm``  -- cross-domain rank matching to the source class rank signatures over the class anchors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core import RankMatcher, pseudo_label_loss, rank_terms, anchor_similarities, supervised_loss
from ..core.anchors import ortho_anchors
from ..datasets import build_slice_eval_loader, build_slice_loaders
from ..evaluation import validate_em_slices, validate_volumes
from ..utils import class_weight_tensor, linear_schedule, ramp_weight
from .base import BaseTrainer

__all__ = ["SliceTrainer"]


class SliceTrainer(BaseTrainer):
    """Shared objective of the single-slice backbones."""

    def build_train_loaders(self, sched):
        return build_slice_loaders(self.cfg.data, sched, self.num_classes, int(getattr(sched, "loader_seed", getattr(sched, "seed", self.cfg.seed))),
                                   int(self.cfg.runtime.num_workers))

    def build_eval_loader(self):
        batch = int(self.cfg.train.eval_batch_size)
        return build_slice_eval_loader(self.cfg.data.target, self.num_classes, batch, int(self.cfg.runtime.num_workers))[1]

    def validate(self, model):
        if self.cfg.evaluation == "medical":
            return validate_volumes(model, self.eval_loader, self.device, self.num_classes, self.cfg.classes,
                                    largest_cc=bool(self.cfg.train.eval_largest_cc))
        return validate_em_slices(model, self.eval_loader, self.device)

    # ------------------------------------------------------------------ teacher
    @torch.no_grad()
    def teacher_forward(self, weak: torch.Tensor):
        self.teacher.eval()
        return self.teacher(weak)

    # ------------------------------------------------------------------ objectives
    def warmup_step(self, source_batch, class_weight):
        src_img, src_lab, _ = source_batch
        src_img, src_lab = src_img.to(self.device), src_lab.to(self.device)
        logits = self.model(src_img)[0]
        sup = supervised_loss(logits, src_lab, self.num_classes, class_weight, float(self.cfg.loss.dice_weight))
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
        src_img, src_lab, _ = source_batch
        src_img, src_lab = src_img.to(dev), src_lab.to(dev)
        src_logits, src_feat, src_low = self.model(src_img)
        sup = supervised_loss(src_logits, src_lab, self.num_classes, class_weight, float(self.cfg.loss.dice_weight))
        total = linear_schedule(sched.sup_weight, it) * sup

        tgt_w, tgt_s, _ = target_batch
        tgt_w, tgt_s = tgt_w.to(dev), tgt_s.to(dev)
        weak_logits, weak_feat, weak_low = self.teacher_forward(tgt_w)
        pseudo = torch.softmax(weak_logits, dim=1).argmax(dim=1)
        strong_logits, strong_feat, strong_low = self.model(tgt_s)
        ramp = int(sched.ramp_iters)

        k, q = int(rida.num_anchors), float(rida.gate_quantile)
        idx = ortho_anchors(weak_low.detach(), k)
        loss_rc, stab, keep = rank_terms(anchor_similarities(weak_low.detach(), idx), anchor_similarities(strong_low, idx), q)
        loss_rc = loss_rc * ramp_weight(it, float(rida.lambda_rc), ramp)
        stab = F.interpolate(stab.unsqueeze(1), size=strong_logits.shape[-2:], mode="bilinear",
                             align_corners=False).squeeze(1).detach()
        pl_cw = getattr(sched, "pl_class_weights", None)
        if pl_cw is None:
            pl_cw = getattr(self.cfg.data, "pl_class_weights", None)
        pl_weight = class_weight_tensor(pl_cw, dev)
        loss_rs = pseudo_label_loss(strong_logits, weak_logits, pseudo, stab, str(sched.pl_target), pl_weight)
        loss_rs = loss_rs * ramp_weight(it, float(rida.lambda_rs), ramp)

        loss_rm = strong_logits.sum() * 0.0
        if float(rida.lambda_rm) > 0.0:
            matcher = self._matcher()
            self.teacher.eval()
            with torch.no_grad():
                _, t_src_feat, _ = self.teacher(src_img)
            matcher.update(it, t_src_feat, src_lab)
            loss_rm, _ = matcher.loss(weak_feat.detach(), strong_logits)
            loss_rm = loss_rm * ramp_weight(it, float(rida.lambda_rm), ramp)

        total = total + loss_rc + loss_rs + loss_rm
        logs = {"loss": float(total.detach()), "sup": float(sup.detach()), "rc": float(loss_rc.detach()),
                "rs": float(loss_rs.detach()), "rm": float(loss_rm.detach()), "stable": float(keep)}
        return total, logs
