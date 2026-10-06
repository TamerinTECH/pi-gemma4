#!/usr/bin/env bash
set -euo pipefail
base=/mnt/usb/pigemma/cf-engine
tag=${1:?unique result tag}
"$base/bin/cf-operator-bench" --plan "$base/artifacts/layer0-tile4/manifest.json" \
  --capture "$base/results/capture/activations/blk.0.ffn_gate_up_exps.weight.json" \
  --out "$base/results/$tag.jsonl" --iterations 61
