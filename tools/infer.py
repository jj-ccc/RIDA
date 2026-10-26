#!/usr/bin/env python
"""Segment the target images of a setting with a RIDA checkpoint.

U-Net, DINOv3 and MedSAM settings: every ``.npz`` slice of ``--input-dir`` (default: the target test
slices of the setting) is segmented and written to ``<output_dir>/<name>.npy`` as a label map.
U-Net settings of the electron-microscopy benchmarks: the target test volume is predicted and written to ``<output_dir>/probability.npy``
(foreground probability) and ``<output_dir>/mask.npy`` (threshold 0.5).

Examples:
    python tools/infer.py configs/rida/medsam/abdominal_ct2mr.yaml checkpoints/rida/abdominal/medsam_ct2mr.pth \
        work_dirs/predictions/abdominal_medsam_ct2mr
    python tools/infer.py configs/rida/unet/mitoem_h2r.yaml checkpoints/rida/mitoem/unet_h2r.pth \
        work_dirs/predictions/mitoem_unet_h2r
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rida.apis import init_model, predict_slice_dir  # noqa: E402
from rida.datasets import build_paired_eval_loader  # noqa: E402
from rida.evaluation import predict_volume  # noqa: E402
from rida.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="RIDA inference")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("output_dir")
    parser.add_argument("--input-dir", default=None, help=".npz slices to segment (U-Net, DINOv3, MedSAM settings)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = init_model(cfg, args.checkpoint, args.device)
    out = Path(args.output_dir)
    if bool(getattr(cfg.model, "paired", False)):
        _, loader = build_paired_eval_loader(cfg.data, batch_size=1, num_workers=int(cfg.runtime.num_workers))
        prob = predict_volume(model, loader, args.device)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "probability.npy", prob.astype(np.float16))
        np.save(out / "mask.npy", (prob >= 0.5).astype(np.uint8))
        print(f"volume {prob.shape} -> {out}")
        return
    input_dir = args.input_dir or str(Path(cfg.data.target) / "test")
    n = predict_slice_dir(model, input_dir, args.output_dir, args.device)
    print(f"{n} slices segmented -> {args.output_dir}")


if __name__ == "__main__":
    main()
