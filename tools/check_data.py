#!/usr/bin/env python
"""Check that the data of the settings is in place.

Example:
    python tools/check_data.py configs/rida/*/cardiac_*.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rida.utils import load_config  # noqa: E402


def check(cfg) -> list[str]:
    d, issues = cfg.data, []
    if bool(getattr(cfg.model, "paired", False)):
        for key in ("source_images", "source_labels", "target_images", "test_images", "test_labels"):
            if not Path(getattr(d, key)).exists():
                issues.append(f"missing {key}: {getattr(d, key)}")
        for key in ("source_list", "target_list", "test_list"):
            value = getattr(d, key, None)
            if value and not Path(value).exists():
                issues.append(f"missing {key}: {value}")
        return issues
    for key, splits in (("source", ("train",)), ("target", ("train", "test"))):
        for split in splits:
            folder = Path(getattr(d, key)) / split
            n = len(list(folder.glob("*.npz"))) if folder.exists() else 0
            if n == 0:
                issues.append(f"no .npz slices in {folder}")
    return issues


def main():
    parser = argparse.ArgumentParser(description="data layout check")
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    failed = 0
    for path in args.configs:
        cfg = load_config(path)
        issues = check(cfg)
        print(f"[{'ok' if not issues else 'missing'}] {cfg.name}")
        for issue in issues:
            print(f"    {issue}")
        failed += bool(issues)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
