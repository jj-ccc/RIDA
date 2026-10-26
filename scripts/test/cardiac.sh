#!/usr/bin/env bash
# Evaluation of the released RIDA checkpoints of the Cardiac CT <-> MR (MM-WHS) benchmark.
# usage: bash scripts/test/cardiac.sh [setting ...]       GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_ct2mr unet_mr2ct dinov3_ct2mr dinov3_mr2ct medsam_ct2mr medsam_mr2ct}
mkdir -p work_dirs/results
for s in ${SETTINGS}; do
  b=${s%_*}
  CUDA_VISIBLE_DEVICES=${GPU} python tools/test.py configs/rida/${b}/cardiac_${s##*_}.yaml checkpoints/rida/cardiac/${s}.pth \
      --out work_dirs/results/cardiac_${s}.json
done
