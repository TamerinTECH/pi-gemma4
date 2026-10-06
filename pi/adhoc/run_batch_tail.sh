#!/usr/bin/env bash
# pi-gemma4 ad-hoc (2026-10-06): cost of batch widths that are NOT a multiple of 4 (llama.cpp repack.cpp sends each
# leftover token row through gemv = one more pass over the weights). E2B QAT Q4_0, 4 threads, resident, nice 10.
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -w 5400 9 || { echo "REFUSED: lock busy"; exit 4; }
R=~/pigemma/results/verified-ceiling; mkdir -p "$R"
E2B=/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') temp=$(vcgencmd measure_temp) $(vcgencmd get_throttled) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB"
nice -n 10 ~/pigemma/llama.cpp/build/bin/llama-bench -m "$E2B" -t 4 -p 420 -n 0 -b 3,4,5,6,7,8,12 -r 2 -o md 2> "$R/bench_e2b_batch_tail.err" | tee "$R/bench_e2b_batch_tail.md"
echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') temp=$(vcgencmd measure_temp) $(vcgencmd get_throttled)"
echo "=== BATCH_TAIL_DONE $(date -Is)"
