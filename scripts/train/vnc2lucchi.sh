#!/usr/bin/env bash
# End-to-end RIDA training of every setting of the VNC III -> Lucchi subset 1 / subset 2 benchmark.
# usage: bash scripts/train/vnc2lucchi.sh [setting ...]      GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_subset1 unet_subset2 dinov3_subset1 dinov3_subset2 medsam_subset1 medsam_subset2}
for s in ${SETTINGS}; do
  b=${s%_*}
  echo "==> vnc2lucchi/${s}"
  CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py configs/rida/${b}/vnc2lucchi_${s##*_}.yaml --work-dir work_dirs/vnc2lucchi/${s}
done
