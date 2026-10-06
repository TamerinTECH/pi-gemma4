#!/usr/bin/env bash
set -euo pipefail
base=/mnt/usb/pigemma/cf-engine
out="$base/results/numerical-diagnostic-v1"
mkdir "$out"
cp "$base/diagnostic-inputs/prompts.json" "$out/prompts.json"
"$base/bin/round-robin/cf-run" --engine cf --cf-plan "$base/artifacts/layer0-tile4/manifest.json" --cf-kernel dotprod \
  --cf-schedule reference,reference,dotprod,asm,reference --cf-report "$out/profile.json" \
  --cf-logits "$out/logits.bin" --cf-routes "$out/routes.bin" --mode run \
  -m /mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf \
  -nr -t 4 -c 512 -b 64 -ub 64 --gen-prompts "$out/prompts.json" --chat --server-stop \
  --requests-out "$out/requests.jsonl" > "$out/stdout.txt" 2> "$out/stderr.txt"
