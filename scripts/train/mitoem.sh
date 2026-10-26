#!/usr/bin/env bash
# End-to-end RIDA training of every setting of the MitoEM human <-> rat benchmark.
# usage: bash scripts/train/mitoem.sh [setting ...]      GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_h2r unet_r2h dinov3_h2r dinov3_r2h medsam_h2r medsam_r2h}
for s in ${SETTINGS}; do
  b=${s%_*}
  echo "==> mitoem/${s}"
  CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py configs/rida/${b}/mitoem_${s##*_}.yaml --work-dir work_dirs/mitoem/${s}
done
