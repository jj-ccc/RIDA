"""End-to-end RIDA training.

A run trains one segmentation network end to end on labelled source images and unlabelled target
images with source supervision and the RIDA objective.

Backbone families implement the model construction, the trainable parameters, the data pipeline,
the per-step objective and the validation protocol.
"""
from __future__ import annotations

import copy
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..utils import (Logger, class_weight_tensor, dump_config, freeze_batchnorm_stats, load_checkpoint,
                     load_model_weights, next_batch, save_checkpoint, set_constant_lr, set_cosine_lr, set_seed,
                     update_ema)

__all__ = ["BaseTrainer"]


class BaseTrainer:
    """Shared optimisation loop of all backbone families."""

    def __init__(self, cfg, work_dir: str | Path, device: str = "cuda"):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        (self.work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.logger = Logger(self.work_dir)
        self.device = torch.device(device)
        self.num_classes = int(cfg.data.num_classes)
        self.matcher = None
        self.source_state = None

    # ------------------------------------------------------------------ hooks
    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def set_trainable(self, warmup: bool) -> None:
        """Set the trainable parameters (``warmup``: the first, source-supervised iterations)."""
        for name, param in self.model.named_parameters():
            param.requires_grad_(self.trainable[name])

    def param_groups(self) -> list:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return [{"params": params, "lr": float(self.cfg.train.lr), "lr_mult": 1.0}]

    def build_train_loaders(self, sched):
        raise NotImplementedError

    def build_eval_loader(self):
        raise NotImplementedError

    def warmup_step(self, source_batch, class_weight):
        raise NotImplementedError

    def train_step(self, source_batch, target_batch, class_weight, it: int, sched):
        raise NotImplementedError

    def validate(self, model: nn.Module) -> dict:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    def build_optimizer(self):
        opt = self.cfg.optimizer
        return torch.optim.Adam(self.param_groups(), lr=float(self.cfg.train.lr), betas=tuple(opt.betas),
                                weight_decay=float(self.cfg.train.weight_decay))

    def round_sched(self):
        sched = copy.deepcopy(self.cfg.train)
        if getattr(self.cfg.train, "restart_ramp", None) is not None:
            sched.ramp_iters = int(self.cfg.train.restart_ramp)
        if getattr(self.cfg.train, "restart_pl_class_weights", None) is not None:
            sched.pl_class_weights = list(self.cfg.train.restart_pl_class_weights)
        return sched

    def _save_latest(self, path, step, part, part_step, best, best_iter):
        save_checkpoint(path, iteration=step, part=part, part_iter=part_step,
                        model=self.model.state_dict(), teacher=self.teacher.state_dict(),
                        optimizer=self.optimizer.state_dict(), best_score=best, best_iter=best_iter,
                        warmup_iters=self.warmup_iters, rank_matcher=self.matcher.state_dict() if self.matcher else None,
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                        numpy_rng=np.random.get_state())

    # ------------------------------------------------------------------ main loop
    def run(self, resume: bool = True) -> dict:
        cfg = self.cfg
        dump_config(cfg, self.work_dir / "config.yaml")
        set_seed(int(getattr(cfg.warmup, "seed", cfg.seed)))
        self.model = self.build_model().to(self.device)
        self.trainable = {n: p.requires_grad for n, p in self.model.named_parameters()}
        self.teacher = copy.deepcopy(self.model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        self.optimizer = self.build_optimizer()
        self.warmup_iters = int(cfg.warmup.iters)
        total = int(cfg.train.iters)
        rest = total - self.warmup_iters          # iterations after the warm-up
        best, best_iter, start = -1.0, -1, 0
        latest = self.work_dir / "checkpoints" / "latest.pth"
        source_path = self.work_dir / "checkpoints" / "source_only.pth"

        if resume and latest.exists():
            state = load_checkpoint(latest)
            self.model.load_state_dict(state["model"], strict=True)
            self.teacher.load_state_dict(state["teacher"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            start, best, best_iter = int(state["iteration"]), float(state["best_score"]), int(state["best_iter"])
            self._pending_matcher = state.get("rank_matcher")
            torch.set_rng_state(state["rng"].cpu())
            np.random.set_state(state["numpy_rng"])
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
            self.logger.info(f"resumed from {latest} at iteration {start}")
        self._pending_matcher = getattr(self, "_pending_matcher", None)
        if source_path.exists():
            self.source_state = load_checkpoint(source_path)["state_dict"]

        self.set_trainable(start < self.warmup_iters)
        self.eval_loader = self.build_eval_loader()
        _, _, src_loader, tgt_loader = self.build_train_loaders(cfg.warmup if start < self.warmup_iters else cfg.train)
        src_iter, tgt_iter = iter(src_loader), None

        stop = [False]
        signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
        restart_every = int(getattr(cfg.train, "restart_every", 0) or 0)
        restart_at = restart_every or int(getattr(cfg.train, "restart_at", 0) or 0)
        round_sched = self.round_sched()
        rt = cfg.runtime
        recent: list[dict] = []
        step = start
        while step < total:
            step += 1
            in_warmup = step <= self.warmup_iters
            part = "warmup" if in_warmup else "train"
            part_step = step if in_warmup else step - self.warmup_iters

            if step == self.warmup_iters + 1:
                self.set_trainable(False)
                self.teacher.load_state_dict(self.model.state_dict(), strict=True)
                if bool(cfg.train.optimizer_reset):
                    self.optimizer = self.build_optimizer()
                del src_loader, tgt_loader
                set_seed(int(cfg.seed))
                _, _, src_loader, tgt_loader = self.build_train_loaders(cfg.train)
                src_iter, tgt_iter = iter(src_loader), None
                recent.clear()

            fixed = bool(restart_at) and not in_warmup and part_step > restart_at
            rnd = 0
            if fixed and restart_every:
                rnd = min((part_step - 1) // restart_every, max(1, rest // restart_every - 1))
            elif fixed:
                rnd = 1
            new_round = fixed and part_step - 1 == (rnd * restart_every if restart_every else restart_at)
            if new_round:
                if rnd >= 2:
                    self.teacher.load_state_dict(self.model.state_dict(), strict=True)
                load_model_weights(self.model, {"state_dict": self.source_state})
                self.set_trainable(False)
                self.optimizer = self.build_optimizer()
            if fixed:
                local_step = part_step - (rnd * restart_every if restart_every else restart_at)
                if restart_every:
                    last = (rnd + 2) * restart_every > rest
                    local_total = rest - rnd * restart_every if last else restart_every
                else:
                    local_total = rest - restart_at
            else:
                local_step, local_total = part_step, (self.warmup_iters if in_warmup else rest)
            active = cfg.warmup if in_warmup else (round_sched if fixed else cfg.train)

            class_weight = class_weight_tensor(active.class_weights, self.device)
            if str(active.lr_schedule) == "cosine":
                lr = set_cosine_lr(self.optimizer, float(active.lr), local_step, local_total, int(active.lr_warmup_iters))
            else:
                lr = set_constant_lr(self.optimizer, float(active.lr))
            began = time.perf_counter()
            self.model.train()
            if not in_warmup and bool(active.freeze_bn):
                freeze_batchnorm_stats(self.model)
            self.optimizer.zero_grad(set_to_none=True)
            src_iter, src_batch = next_batch(src_iter, src_loader)
            if in_warmup:
                loss, logs = self.warmup_step(src_batch, class_weight)
            else:
                if tgt_iter is None:
                    tgt_iter = iter(tgt_loader)
                tgt_iter, tgt_batch = next_batch(tgt_iter, tgt_loader)
                loss, logs = self.train_step(src_batch, tgt_batch, class_weight, local_step, active)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at iteration {step}: {logs}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.optimizer.grad_clip))
            self.optimizer.step()
            if not in_warmup and not fixed:
                update_ema(self.teacher, self.model, float(cfg.train.ema_decay))
            logs["sec"] = time.perf_counter() - began
            recent.append(logs)

            if step == 1 or step % int(rt.log_interval) == 0:
                row = {"iter": step, "part": part, "part_iter": part_step, "type": "train", "lr": lr,
                       **{k: float(np.mean([r[k] for r in recent])) for k in recent[-1]}}
                self.logger.scalars(row, keys=["loss", "sup", "rc", "rs", "rm", "sec"])
                recent.clear()

            if step % int(rt.eval_interval) == 0 or step in (self.warmup_iters, total):
                deployed = self.model if (in_warmup or fixed) else self.teacher
                metrics = self.validate(deployed)
                self.logger.record({"iter": step, "part": part, "part_iter": part_step, "type": "valid", **metrics})
                self.logger.info(f"[valid] iteration {step} score={100 * metrics['score']:.2f}")
                if not in_warmup and metrics["score"] > best:
                    best, best_iter = metrics["score"], step
                    save_checkpoint(self.work_dir / "checkpoints" / "best.pth", state_dict=deployed.state_dict(),
                                    iteration=step, metrics=metrics)
                if step == self.warmup_iters:
                    save_checkpoint(source_path, state_dict=self.model.state_dict(),
                                    optimizer=self.optimizer.state_dict(), iteration=step, metrics=metrics)
                    self.source_state = self.model.state_dict()
                    self.source_state = {k: v.detach().cpu().clone() for k, v in self.source_state.items()}
                torch.cuda.empty_cache()

            if step % int(rt.save_interval) == 0 or step == total or stop[0]:
                self._save_latest(latest, step, part, part_step, best, best_iter)
            if stop[0]:
                self.logger.info(f"stopped at iteration {step}; resume with the same command")
                return {"stopped": step}

        summary = {"name": cfg.name, "best_score": best, "best_iter": best_iter, "iters": total}
        (self.work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
