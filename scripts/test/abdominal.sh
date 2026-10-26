#!/usr/bin/env bash
# Evaluation of the released RIDA checkpoints of the Abdominal CT <-> MR (BTCV / CHAOS) benchmark.
# usage: bash scripts/test/abdominal.sh [setting ...]       GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_ct2mr unet_mr2ct dinov3_ct2mr dinov3_mr2ct medsam_ct2mr medsam_mr2ct}
mkdir -p work_dirs/results
for s in ${SETTINGS}; do
  b=${s%_*}
  CUDA_VISIBLE_DEVICES=${GPU} python tools/test.py configs/rida/${b}/abdominal_${s##*_}.yaml checkpoints/rida/abdominal/${s}.pth \
      --out work_dirs/results/abdominal_${s}.json
done
