"""Checkpoint input and output."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["save_checkpoint", "load_checkpoint", "load_weights", "extract_state_dict", "load_model_weights"]


def save_checkpoint(path: str | Path, **state) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(str(path), map_location=map_location, weights_only=False)


def load_weights(path: str | Path, map_location="cpu") -> dict:
    """Network weights of a model checkpoint (tensors only)."""
    return extract_state_dict(torch.load(str(path), map_location=map_location, weights_only=True))


def extract_state_dict(state: dict) -> dict:
    for key in ("state_dict", "model_weights", "model"):
        if isinstance(state, dict) and key in state:
            return state[key]
    return state


def load_model_weights(model: nn.Module, state: dict, allow_missing_prefix: str | None = "encoder.") -> None:
    """Load the network weights of a checkpoint into ``model``."""
    sd = extract_state_dict(state)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not (k.endswith("num_batches_tracked")
                                      or (allow_missing_prefix and k.startswith(allow_missing_prefix)))]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing={bad[:8]} unexpected={list(unexpected)[:8]}")
