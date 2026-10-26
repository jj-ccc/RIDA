#!/usr/bin/env bash
# End-to-end RIDA training of every setting of the Cardiac CT <-> MR (MM-WHS) benchmark.
# usage: bash scripts/train/cardiac.sh [setting ...]      GPU=<id> selects the device (default 0)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
SETTINGS=${@:-unet_ct2mr unet_mr2ct dinov3_ct2mr dinov3_mr2ct medsam_ct2mr medsam_mr2ct}
for s in ${SETTINGS}; do
  b=${s%_*}
  echo "==> cardiac/${s}"
  CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py configs/rida/${b}/cardiac_${s##*_}.yaml --work-dir work_dirs/cardiac/${s}
done
