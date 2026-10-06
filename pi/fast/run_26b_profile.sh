#!/usr/bin/env bash
# Per-operation times for the prepacking gate of the 26B-A4B (brief 3, part 6). Tool: CF-Engine runner in reference
# mode (pinned llama.cpp a7b94df with per-op timers), as for the E2B profile of the afternoon.
#   1. 26B QAT Q4_0 with the weights left mapped (-nr): whole prompt in one pass, then one-row decode passes
#   2. the same with 4 rows per pass (-b 4): the shape of a draft-verification pass
#   3. representative dense operations: E2B with 4 rows per pass, weights repacked (default) and left mapped (-nr)
# Resident, nice 10, bench lock.
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: lock busy"; exit 4; }
base=/mnt/usb/pigemma/cf-engine
out=~/pigemma/results/fast3/profile26; mkdir -p "$out"
M26=/mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf
E2B=/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
h() { echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') $(vcgencmd measure_temp) $(vcgencmd get_throttled) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB swapfree=$(awk '/SwapFree/{print $2}' /proc/meminfo)kB $(grep -E 'pswpin|pswpout' /proc/vmstat | tr '\n' ' ')"; }
h
# the 26B fills the page cache of this board: do not start it when a service is down or swap is nearly used up
ok26=1
case "$(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ')" in *inactive*|*failed*|*activating*) ok26=0;; esac
[ "$(awk '/SwapFree/{print $2}' /proc/meminfo)" -ge 800000 ] || ok26=0
cat > "$out/prompts.json" <<'EOF'
[
  {"id": "warmup", "prompt": "Hello", "max_tokens": 8},
  {"id": "en_explain", "prompt": "Explain in a few sentences how a refrigerator keeps food cold.", "max_tokens": 48},
  {"id": "de_explain", "prompt": "Erkläre in wenigen Sätzen, warum der Himmel blau ist.", "max_tokens": 48}
]
EOF
cat > "$out/prompts_b4.json" <<'EOF'
[
  {"id": "warmup", "prompt": "Hello", "max_tokens": 4},
  {"id": "long_en", "prompt": "Read the following note and then answer in one sentence. The city council met on Tuesday evening to discuss the new bicycle lanes along the river road. Several residents asked for better lighting, two shop owners worried about parking, and the head of the school asked for a safe crossing near the main gate. The council agreed to add lights and a crossing, and to review parking after six months. What did the council decide?", "max_tokens": 32}
]
EOF
prof() {  # prof <name> <model> <batch> <prompts> [extra flags]
  local name=$1 model=$2 b=$3 prompts=$4; shift 4
  nice -n 10 "$base/bin/cf-run" --engine reference --cf-report "$out/profile_$name.json" --mode run -m "$model" "$@" \
    -t 4 -c 1024 -b "$b" -ub "$b" --gen-prompts "$prompts" --chat --server-stop \
    --requests-out "$out/requests_$name.jsonl" > "$out/stdout_$name.txt" 2> "$out/stderr_$name.txt"
  echo "$name rc=$?"; grep -E "PIGEMMA_SUMMARY" "$out/stderr_$name.txt" "$out/stdout_$name.txt" | cut -c1-300 | tail -2
  h
}
prof e2b_repack_b4    "$E2B" 4  "$out/prompts_b4.json"
prof e2b_norepack_b4  "$E2B" 4  "$out/prompts_b4.json" -nr
# one 26B load only: the prompt goes in 4-row passes, the answer in one-row passes
if [ "$ok26" = 1 ]; then
  prof 26b_norepack_b4 "$M26" 4 "$out/prompts_b4.json" -nr
else
  echo "SKIPPED the 26B profile: service state or free swap below the limit"
fi
echo "=== PROFILE26_DONE $(date -Is)"
