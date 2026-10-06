#!/usr/bin/env bash
# pi-gemma4 ad-hoc (2026-10-06): per-operation time of one E2B decode pass on the Pi (CF-Engine runner in reference
# mode = pinned llama.cpp a7b94df with per-op timers). Two variants: default repack and -nr. Resident, nice 10.
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: lock busy"; exit 4; }
base=/mnt/usb/pigemma/cf-engine
out=~/pigemma/results/e2b-profile; mkdir -p "$out"
E2B=/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') $(vcgencmd measure_temp) $(vcgencmd get_throttled)"
for mode in repack norepack; do
  extra=""; [ "$mode" = norepack ] && extra="-nr"
  nice -n 10 "$base/bin/cf-run" --engine reference --cf-report "$out/profile_$mode.json" --mode run -m "$E2B" $extra \
    -t 4 -c 1024 -b 64 -ub 64 --gen-prompts ~/pigemma/profile_prompts_e2b.json --chat --server-stop \
    --requests-out "$out/requests_$mode.jsonl" > "$out/stdout_$mode.txt" 2> "$out/stderr_$mode.txt"
  echo "$mode rc=$?"; grep -E "PIGEMMA_SUMMARY|chunk [0-9]" "$out/stderr_$mode.txt" "$out/stdout_$mode.txt" | cut -c1-330 | tail -4
done
echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') $(vcgencmd measure_temp) $(vcgencmd get_throttled)"
echo "=== E2B_PROFILE_DONE"
