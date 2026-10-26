#!/usr/bin/env bash
# Verify the dataset layout of every setting.
set -euo pipefail
cd "$(dirname "$0")/../.."
python tools/check_data.py configs/rida/*/{abdominal,cardiac,vnc2lucchi,mitoem}_*.yaml
