#!/usr/bin/env bash
set -euo pipefail
base=/mnt/usb/pigemma/cf-engine
out="$base/results/profile-1"
mkdir -p "$out"
"$base/bin/cf-run" --engine reference --cf-report "$out/profile.json" --mode run \
  -m /mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf \
  -nr -t 4 -c 512 -b 64 -ub 64 --gen-prompts "$base/profile-prompts.json" --chat --server-stop \
  --requests-out "$out/requests.jsonl" > "$out/stdout.txt" 2> "$out/stderr.txt"
"$base/bin/cf-kernel-test" --bench --iterations 3 > "$out/kernel-bench.json"
