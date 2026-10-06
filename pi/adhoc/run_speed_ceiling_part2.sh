#!/usr/bin/env bash
# pi-gemma4 ad-hoc: remaining speed-ceiling blocks (5-7) - distinct ports, --extra=... form (argparse), flock.
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi benchmark holds the lock"; exit 4; }
OURS=~/pigemma/llama.cpp/build/bin
IK=~/pigemma/ik_llama.cpp/build/bin
E2B=/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
M26=~/pigemma/models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf
health() { echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') temp=$(vcgencmd measure_temp) $(vcgencmd get_throttled) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB"; }
health
echo "=== 5 parallel streams E2B, ik $(date -Is)"
python3 par_throughput.py --bin $IK/llama-server --model "$E2B" --label speed-ceiling/par_e2b_ik --streams 1,2,4,8,16 --n-predict 96 --port 18094 --extra=--jinja
sleep 45; health
echo "=== 6 parallel streams 26B-A4B UD-Q3_K_M, ik $(date -Is)"
python3 par_throughput.py --bin $IK/llama-server --model "$M26" --label speed-ceiling/par_26b_ik --streams 1,4,8 --n-predict 64 --port 18095 "--extra=--swa-compress --jinja"
sleep 45; health
echo "=== 7 parallel streams 26B-A4B UD-Q3_K_M, ours (-nr) $(date -Is)"
python3 par_throughput.py --bin $OURS/llama-server --model "$M26" --label speed-ceiling/par_26b_ours --streams 1,4,8 --n-predict 64 --port 18096 --extra=-nr
health
echo "=== SPEED_CEILING_PART3_DONE $(date -Is)"
