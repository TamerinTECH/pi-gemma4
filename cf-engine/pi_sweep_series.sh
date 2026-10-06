#!/usr/bin/env bash
set -euo pipefail
base=/mnt/usb/pigemma/cf-engine
prefix=${1:-}
for threads in 4 1 2; do
  bash "$base/pi_sweep.sh" perf "$threads" "${prefix}sweep-t$threads"
done
bash "$base/pi_sweep.sh" quality 4 "${prefix}quality-asm"
