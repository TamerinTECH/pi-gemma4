#!/usr/bin/env bash
set -euo pipefail
base=/mnt/usb/pigemma/cf-engine
kind=${1:?perf|quality}
threads=${2:?thread count}
tag=${3:?unique run tag}
out="$base/results/$tag"
mkdir "$out"
cp "$base/suite/$kind-prompts.json" "$out/prompts.json"
schedule=$(cat "$base/suite/$kind-schedule.txt")
extra=()
if [[ "$kind" == quality ]]; then
  extra+=(--cf-logits "$out/logits.bin" --cf-routes "$out/routes.bin")
fi
"$base/bin/cf-run" --engine cf --cf-plan "$base/artifacts/layer0-tile4/manifest.json" --cf-kernel dotprod \
  --cf-schedule "$schedule" --cf-report "$out/profile.json" "${extra[@]}" --mode run \
  -m /mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf \
  -nr -t "$threads" -c 512 -b 64 -ub 64 --gen-prompts "$base/suite/$kind-prompts.json" --chat --server-stop \
  --requests-out "$out/requests.jsonl" > "$out/stdout.txt" 2> "$out/stderr.txt"
