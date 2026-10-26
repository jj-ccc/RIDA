"""Training entry point shared by the command-line tools."""
from __future__ import annotations

from pathlib import Path

import torch

from ..trainers import build_trainer
from ..utils import load_config

__all__ = ["train_setting", "default_work_dir"]


def default_work_dir(cfg) -> Path:
    return Path("work_dirs") / cfg.benchmark / cfg.name[len(cfg.benchmark) + 1:]


def train_setting(config: str, work_dir: str | None = None, device: str = "cuda", overrides: dict | None = None,
                  cpu_threads: int = 3) -> dict:
    """Train one setting end to end.

    Args:
        config: path of the YAML configuration.
        work_dir: output directory (``work_dirs/<benchmark>/<backbone>_<direction>`` by default).
        overrides: nested dictionary merged into the configuration.
    """
    torch.set_num_threads(int(cpu_threads))
    cfg = load_config(config, overrides)
    trainer = build_trainer(cfg, work_dir or default_work_dir(cfg), device=device)
    return trainer.run(resume=True)
