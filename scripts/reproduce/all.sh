#!/usr/bin/env bash
# End-to-end RIDA training and evaluation of all 24 settings, with a summary of the results.
set -euo pipefail
cd "$(dirname "$0")/../.."
for b in abdominal cardiac vnc2lucchi mitoem; do bash scripts/reproduce/${b}.sh; done
python tools/summarize_results.py work_dirs/results/reproduce_*.json
