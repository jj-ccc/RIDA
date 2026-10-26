#!/usr/bin/env bash
# End-to-end RIDA training of every setting of the MitoEM human <-> rat benchmark,
# followed by the evaluation of the resulting checkpoint.
# usage: bash scripts/reproduce/mitoem.sh [setting ...]  GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_h2r unet_r2h dinov3_h2r dinov3_r2h medsam_h2r medsam_r2h}
mkdir -p work_dirs/results
for s in ${SETTINGS}; do
  b=${s%_*}
  echo "==> mitoem/${s}"
  CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py configs/rida/${b}/mitoem_${s##*_}.yaml \
      --work-dir work_dirs/reproduce/mitoem/${s}
  CUDA_VISIBLE_DEVICES=${GPU} python tools/test.py configs/rida/${b}/mitoem_${s##*_}.yaml \
      work_dirs/reproduce/mitoem/${s}/checkpoints/best.pth --out work_dirs/results/reproduce_mitoem_${s}.json
done
