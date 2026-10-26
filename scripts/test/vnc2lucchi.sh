#!/usr/bin/env bash
# Evaluation of the released RIDA checkpoints of the VNC III -> Lucchi subset 1 / subset 2 benchmark.
# usage: bash scripts/test/vnc2lucchi.sh [setting ...]       GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_subset1 unet_subset2 dinov3_subset1 dinov3_subset2 medsam_subset1 medsam_subset2}
mkdir -p work_dirs/results
for s in ${SETTINGS}; do
  b=${s%_*}
  CUDA_VISIBLE_DEVICES=${GPU} python tools/test.py configs/rida/${b}/vnc2lucchi_${s##*_}.yaml checkpoints/rida/vnc2lucchi/${s}.pth \
      --out work_dirs/results/vnc2lucchi_${s}.json
done
