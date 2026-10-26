#!/usr/bin/env bash
# Run the settings of one benchmark in parallel, one setting per GPU.
# usage: GPUS="0 1 2 3 4 5" bash scripts/train/multi_gpu.sh <benchmark>
set -euo pipefail
cd "$(dirname "$0")/../.."
BENCH=$1; shift || true
EXTRA=${*:-}
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5}"
mkdir -p work_dirs/logs
i=0
for cfg in configs/rida/*/${BENCH}_*.yaml; do
  st=$(basename "${cfg}" .yaml)
  b=$(basename "$(dirname "${cfg}")")
  s="${b}_${st##*_}"
  gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  echo "==> ${BENCH}/${s} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES=${gpu} nohup python tools/train.py "${cfg}" ${EXTRA} --work-dir "work_dirs/${BENCH}/${s}" \
      > "work_dirs/logs/${BENCH}_${s}.log" 2>&1 &
  i=$((i + 1))
done
wait
