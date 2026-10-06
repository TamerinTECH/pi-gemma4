#!/usr/bin/env bash
# pi-gemma4 ad-hoc: how fast can ANY full Gemma 4 go on this Pi 5 (CPU only)?  Goal under test: >= 100 tok/s.
# Smallest full Gemma 4 (E2B, Google QAT Q4_0) and the 26B-A4B MoE, two engines (llama.cpp a7b94df, ik_llama.cpp
# 3d27f5bb), three metrics each: single-stream decode, prompt processing, aggregate throughput of N parallel streams.
# Resident runs only (no memory pressure), nice 10, cool-down pauses, one job at a time (flock).
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi benchmark holds the lock"; exit 4; }
OURS=~/pigemma/llama.cpp/build/bin
IK=~/pigemma/ik_llama.cpp/build/bin
E2B=/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
M26=~/pigemma/models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf
R=~/pigemma/results/speed-ceiling; mkdir -p "$R"
health() { echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') temp=$(vcgencmd measure_temp) $(vcgencmd get_throttled) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB"; }
cool() { sleep "${1:-30}"; }
health
echo "=== 1 llama-bench E2B, ours (default repack) $(date -Is)"
nice -n 10 $OURS/llama-bench -m "$E2B" -t 4 -p 512 -n 128 -r 3 -o md 2> "$R/bench_e2b_ours.err" | tee "$R/bench_e2b_ours.md"
cool; health
echo "=== 2 llama-bench E2B, ik (mmap) $(date -Is)"
nice -n 10 $IK/llama-bench -m "$E2B" -t 4 -p 512 -n 128 -r 3 -o md 2> "$R/bench_e2b_ik.err" | tee "$R/bench_e2b_ik.md"
cool; health
echo "=== 3 llama-bench E2B, ik run-time repack $(date -Is)"
nice -n 10 $IK/llama-bench -m "$E2B" -t 4 -p 512 -n 128 -r 3 -rtr 1 -o md 2> "$R/bench_e2b_ik_rtr.err" | tee "$R/bench_e2b_ik_rtr.md"
cool; health
echo "=== 4 parallel streams E2B, ours $(date -Is)"
python3 par_throughput.py --bin $OURS/llama-server --model "$E2B" --label speed-ceiling/par_e2b_ours --streams 1,2,4,8,16 --n-predict 96
cool; health
echo "=== 5 parallel streams E2B, ik $(date -Is)"
python3 par_throughput.py --bin $IK/llama-server --model "$E2B" --label speed-ceiling/par_e2b_ik --streams 1,2,4,8,16 --n-predict 96
cool 45; health
echo "=== 6 parallel streams 26B-A4B UD-Q3_K_M, ik $(date -Is)"
python3 par_throughput.py --bin $IK/llama-server --model "$M26" --label speed-ceiling/par_26b_ik --streams 1,4,8 --n-predict 64 --extra "--swa-compress"
cool 45; health
echo "=== 7 parallel streams 26B-A4B UD-Q3_K_M, ours (-nr) $(date -Is)"
python3 par_throughput.py --bin $OURS/llama-server --model "$M26" --label speed-ceiling/par_26b_ours --streams 1,4,8 --n-predict 64 --extra "-nr"
health
echo "=== SPEED_CEILING_DONE $(date -Is)"
