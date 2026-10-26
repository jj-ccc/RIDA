#!/usr/bin/env bash
# End-to-end RIDA training of every setting of the VNC III -> Lucchi subset 1 / subset 2 benchmark,
# followed by the evaluation of the resulting checkpoint.
# usage: bash scripts/reproduce/vnc2lucchi.sh [setting ...]  GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_subset1 unet_subset2 dinov3_subset1 dinov3_subset2 medsam_subset1 medsam_subset2}
mkdir -p work_dirs/results
for s in ${SETTINGS}; do
  b=${s%_*}
  echo "==> vnc2lucchi/${s}"
  CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py configs/rida/${b}/vnc2lucchi_${s##*_}.yaml \
      --work-dir work_dirs/reproduce/vnc2lucchi/${s}
  CUDA_VISIBLE_DEVICES=${GPU} python tools/test.py configs/rida/${b}/vnc2lucchi_${s##*_}.yaml \
      work_dirs/reproduce/vnc2lucchi/${s}/checkpoints/best.pth --out work_dirs/results/reproduce_vnc2lucchi_${s}.json
done
