"""Configuration loading.

The entry configuration of a setting is ``configs/rida/<backbone>/<benchmark>_<direction>.yaml``.  It inherits,
through ``_base_``, ``configs/base/<backbone>/<benchmark>_<direction>.yaml`` -> ``configs/base/<backbone>/<backbone>.yaml``
-> ``configs/base/base.yaml``; a later file overrides an earlier one.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

__all__ = ["load_config", "to_namespace", "to_dict", "deep_update", "dump_config"]

ROOT = Path(__file__).resolve().parents[2]


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for base in bases:
        merged = deep_update(merged, _read((path.parent / base).resolve()))
    return deep_update(merged, data)


def to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj


def _resolve(path: str | None) -> str | None:
    if not path:
        return path
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def load_config(path: str | Path, overrides: dict | None = None) -> SimpleNamespace:
    """Load a setting (with its ``_base_`` chain) and resolve repository-relative paths."""
    path = Path(path).resolve()
    cfg = _read(path)
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg["name"] = cfg.get("name") or f"{cfg['benchmark']}_{cfg['model']['backbone']}_{cfg['direction']}"
    cfg["config_path"] = str(path)
    for key in ("source", "target", "source_images", "source_labels", "source_list", "target_images",
                "target_list", "test_images", "test_labels", "test_list"):
        if key in cfg["data"]:
            cfg["data"][key] = _resolve(cfg["data"][key])
    for key in ("pretrained", "repository"):
        if key in cfg["model"]:
            cfg["model"][key] = _resolve(cfg["model"][key])
    return to_namespace(cfg)


def dump_config(cfg: SimpleNamespace, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(json.loads(json.dumps(to_dict(cfg))), sort_keys=False))
