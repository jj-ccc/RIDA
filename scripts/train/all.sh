#!/usr/bin/env bash
# End-to-end RIDA training of all 24 settings, one benchmark after the other.
set -euo pipefail
cd "$(dirname "$0")/../.."
for b in abdominal cardiac vnc2lucchi mitoem; do bash scripts/train/${b}.sh; done
