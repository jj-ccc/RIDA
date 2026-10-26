#!/usr/bin/env python
"""Train one setting with RIDA.

Examples:
    python tools/train.py configs/rida/unet/cardiac_ct2mr.yaml
    python tools/train.py configs/rida/dinov3/mitoem_h2r.yaml --work-dir work_dirs/mitoem/dinov3_h2r
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rida.apis import train_setting  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="RIDA training")
    parser.add_argument("config", help="setting configuration (configs/rida/<backbone>/<benchmark>_<direction>.yaml)")
    parser.add_argument("--work-dir", default=None, help="output directory")
    parser.add_argument("--seed", type=int, default=None, help="override the seed of the configuration")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {"seed": args.seed} if args.seed is not None else None
    summary = train_setting(args.config, args.work_dir, args.device, overrides)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
