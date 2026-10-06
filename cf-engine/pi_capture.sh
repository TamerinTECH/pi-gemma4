#!/usr/bin/env bash
set -euo pipefail
base=/mnt/usb/pigemma/cf-engine
out="$base/results/capture"
mkdir -p "$out"
"$base/bin/cf-run" --engine cf --cf-plan "$base/artifacts/layer0-tile4/manifest.json" --cf-kernel asm \
  --cf-report "$out/profile.json" --cf-capture-dir "$out/activations" --mode run \
  -m /mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf \
  -nr -t 4 -c 512 -b 64 -ub 64 --gen-prompts "$base/profile-prompts.json" --chat --server-stop \
  --requests-out "$out/requests.jsonl" > "$out/stdout.txt" 2> "$out/stderr.txt"
