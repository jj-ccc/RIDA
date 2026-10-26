#!/usr/bin/env python
"""Evaluate a checkpoint with the protocol of its benchmark.

Example:
    python tools/test.py configs/rida/unet/cardiac_ct2mr.yaml checkpoints/rida/cardiac/unet_ct2mr.pth
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rida.apis import evaluate, init_model  # noqa: E402
from rida.utils import load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="RIDA evaluation")
    parser.add_argument("config", help="setting configuration")
    parser.add_argument("checkpoint", help="model checkpoint")
    parser.add_argument("--out", default=None, help="write the metrics to this JSON file")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = init_model(cfg, args.checkpoint, args.device)
    metrics = evaluate(cfg, model, args.device)
    result = {"setting": cfg.name, "checkpoint": str(args.checkpoint), "metrics": metrics}
    print(json.dumps({"setting": cfg.name, **{k: round(v, 2) for k, v in metrics.items()}}))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
