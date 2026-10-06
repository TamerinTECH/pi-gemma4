#!/usr/bin/env bash
# Per-operation time of 26B-A4B QAT Q4_0 on the Pi with the weights left mapped (-nr), for the prepacking gate.
# Tool: CF-Engine runner in reference mode (pinned llama.cpp a7b94df with per-op timers), as for the E2B profile.
# One short request (prefill = batched rows, decode = one row). Resident, nice 10, bench lock.
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: lock busy"; exit 4; }
base=/mnt/usb/pigemma/cf-engine
out=~/pigemma/results/fast3/profile26; mkdir -p "$out"
M26=/mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf
h() { echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') $(vcgencmd measure_temp) $(vcgencmd get_throttled) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB"; }
h
cat > "$out/prompts.json" <<'EOF'
[
  {"id": "warmup", "prompt": "Hello", "max_tokens": 8},
  {"id": "en_explain", "prompt": "Explain in a few sentences how a refrigerator keeps food cold.", "max_tokens": 64},
  {"id": "de_explain", "prompt": "Erkläre in wenigen Sätzen, warum der Himmel blau ist.", "max_tokens": 64}
]
EOF
nice -n 10 "$base/bin/cf-run" --engine reference --cf-report "$out/profile_norepack.json" --mode run -m "$M26" -nr \
  -t 4 -c 1024 -b 64 -ub 64 --gen-prompts "$out/prompts.json" --chat --server-stop \
  --requests-out "$out/requests_norepack.jsonl" > "$out/stdout_norepack.txt" 2> "$out/stderr_norepack.txt"
echo "norepack rc=$?"; grep -E "PIGEMMA_SUMMARY" "$out/stderr_norepack.txt" "$out/stdout_norepack.txt" | cut -c1-330 | tail -3
h
echo "=== PROFILE26_DONE $(date -Is)"
