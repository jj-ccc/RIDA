#!/usr/bin/env bash
# Evaluation of the released RIDA checkpoints of the MitoEM human <-> rat benchmark.
# usage: bash scripts/test/mitoem.sh [setting ...]       GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_h2r unet_r2h dinov3_h2r dinov3_r2h medsam_h2r medsam_r2h}
mkdir -p work_dirs/results
for s in ${SETTINGS}; do
  b=${s%_*}
  CUDA_VISIBLE_DEVICES=${GPU} python tools/test.py configs/rida/${b}/mitoem_${s##*_}.yaml checkpoints/rida/mitoem/${s}.pth \
      --out work_dirs/results/mitoem_${s}.json
done
