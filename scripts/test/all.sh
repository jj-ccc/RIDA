#!/usr/bin/env bash
# Evaluation of all released RIDA checkpoints and a summary table.
set -euo pipefail
cd "$(dirname "$0")/../.."
for b in abdominal cardiac vnc2lucchi mitoem; do bash scripts/test/${b}.sh; done
python tools/summarize_results.py work_dirs/results/{abdominal,cardiac,vnc2lucchi,mitoem}_*.json
