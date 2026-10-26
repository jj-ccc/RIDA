#!/usr/bin/env python
"""Convert a training checkpoint into a release checkpoint.

The release file stores the complete network weights (``state_dict``) and nothing else: no optimizer
state, iteration counter, metrics or other training records.

Example:
    python tools/export_checkpoint.py work_dirs/abdominal/dinov3_ct2mr/checkpoints/best.pth \
        checkpoints/rida/abdominal/dinov3_ct2mr.pth
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rida.utils import extract_state_dict, load_checkpoint, save_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="export a release checkpoint")
    parser.add_argument("checkpoint", help="training checkpoint (e.g. best.pth)")
    parser.add_argument("output", help="release checkpoint")
    args = parser.parse_args()
    sd = {k: v.detach().cpu() for k, v in extract_state_dict(load_checkpoint(args.checkpoint)).items()}
    save_checkpoint(args.output, state_dict=sd)
    print(f"{args.output}: {len(sd)} tensors")


if __name__ == "__main__":
    main()
