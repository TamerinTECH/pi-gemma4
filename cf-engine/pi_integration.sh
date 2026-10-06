#!/usr/bin/env bash
set -euo pipefail
base=/mnt/usb/pigemma/cf-engine
model=/mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf
mode=${1:?reference|validate|cf}
out="$base/results/integration-$mode"
mkdir -p "$out"
extra=()
engine=reference
if [[ "$mode" != reference ]]; then
  engine=cf
  extra+=(--cf-plan "$base/artifacts/layer0-tile4/manifest.json" --cf-kernel dotprod)
fi
if [[ "$mode" == validate ]]; then extra+=(--cf-validate); fi
"$base/bin/cf-run" --engine "$engine" "${extra[@]}" --cf-report "$out/profile.json" --mode run \
  -m "$model" -nr -t 4 -c 512 -b 64 -ub 64 --gen-prompts "$base/integration-prompts.json" --chat --server-stop \
  --requests-out "$out/requests.jsonl" > "$out/stdout.txt" 2> "$out/stderr.txt"
