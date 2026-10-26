#!/usr/bin/env bash
# End-to-end RIDA training of every setting of the Cardiac CT <-> MR (MM-WHS) benchmark,
# followed by the evaluation of the resulting checkpoint.
# usage: bash scripts/reproduce/cardiac.sh [setting ...]  GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_ct2mr unet_mr2ct dinov3_ct2mr dinov3_mr2ct medsam_ct2mr medsam_mr2ct}
mkdir -p work_dirs/results
for s in ${SETTINGS}; do
  b=${s%_*}
  echo "==> cardiac/${s}"
  CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py configs/rida/${b}/cardiac_${s##*_}.yaml \
      --work-dir work_dirs/reproduce/cardiac/${s}
  CUDA_VISIBLE_DEVICES=${GPU} python tools/test.py configs/rida/${b}/cardiac_${s##*_}.yaml \
      work_dirs/reproduce/cardiac/${s}/checkpoints/best.pth --out work_dirs/results/reproduce_cardiac_${s}.json
done
