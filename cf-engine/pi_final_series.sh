#!/usr/bin/env bash
set -euo pipefail
base=/mnt/usb/pigemma/cf-engine
# Run only inside pi_job.py: one guarded process tree, no competing Pi jobs.
for tag in exact-operator-3 exact-operator-4; do
  bash "$base/pi_operator.sh" "$tag"
done
for threads in 4 1 2; do
  bash "$base/pi_sweep.sh" perf "$threads" "exact-sweep-t$threads"
done
