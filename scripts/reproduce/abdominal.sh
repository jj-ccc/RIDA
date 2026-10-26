#!/usr/bin/env bash
# End-to-end RIDA training of every setting of the Abdominal CT <-> MR (BTCV / CHAOS) benchmark,
# followed by the evaluation of the resulting checkpoint.
# usage: bash scripts/reproduce/abdominal.sh [setting ...]  GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_ct2mr unet_mr2ct dinov3_ct2mr dinov3_mr2ct medsam_ct2mr medsam_mr2ct}
mkdir -p work_dirs/results
for s in ${SETTINGS}; do
  b=${s%_*}
  echo "==> abdominal/${s}"
  CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py configs/rida/${b}/abdominal_${s##*_}.yaml \
      --work-dir work_dirs/reproduce/abdominal/${s}
  CUDA_VISIBLE_DEVICES=${GPU} python tools/test.py configs/rida/${b}/abdominal_${s##*_}.yaml \
      work_dirs/reproduce/abdominal/${s}/checkpoints/best.pth --out work_dirs/results/reproduce_abdominal_${s}.json
done
