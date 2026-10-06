#!/usr/bin/env bash
# pigemma-fast (2026-10-06): reference llama.cpp a7b94df vs the patched copy, E2B QAT Q4_0, 4 threads, resident.
#   A  reference prompt test at several batch widths
#   B  patched, PIGEMMA_PAD_TAIL=1
#   C  patched, + PIGEMMA_CHUNK_KB (64 / 128 / 256 at wide batches, then the best for the sweep)
#   D  patched, + PIGEMMA_STRIP_PREFILL=1 (prompt tokens without output skip the shared-KV layers)
#   E  decode check (tg) reference vs patched with all switches
#   F  server: same prompts, reference vs patched (default batch and -b 64 = prompt batches without any output);
#      full texts are stored for an exact comparison
# nice 10, one job at a time (flock), services only read; aborts when health degrades.
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi benchmark holds the lock"; exit 4; }
REF=~/pigemma/llama.cpp/build/bin
FAST=/mnt/usb/pigemma/fast/bin
E2B=/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
R=~/pigemma/results/fast; mkdir -p "$R"
guard() {
  local svc temp thr mem
  svc=$(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ')
  temp=$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp)
  thr=$(vcgencmd get_throttled | cut -d= -f2)
  mem=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  echo "[health $(date +%T)] $svc temp=${temp}C throttled=$thr memavail=${mem}kB"
  case "$svc" in *inactive*|*failed*|*activating*) echo "ABORT: a service is not active"; exit 9;; esac
  [ "$thr" = "0x0" ] || { echo "ABORT: throttled=$thr"; exit 9; }
  [ "$mem" -gt 1048576 ] || { echo "ABORT: MemAvailable below 1 GiB"; exit 9; }
  awk -v t="$temp" 'BEGIN{exit !(t<78)}' || { echo "ABORT: temperature $temp"; exit 9; }
}
bench() {  # bench <label> <binary-dir> <batch-list> [extra llama-bench args]
  local label=$1 dir=$2 bl=$3; shift 3
  nice -n 10 "$dir/llama-bench" -m "$E2B" -t 4 -p 512 -n 0 -b "$bl" -r 2 -o md "$@" 2> "$R/$label.err" | tee "$R/$label.md" | grep -E "pp|tg"
}
[ -x "$FAST/llama-bench" ] || { echo "REFUSED: no patched binaries in $FAST"; exit 3; }
guard
echo "=== A reference $(date -Is)"
bench A_ref "$REF" 3,8,16,64,512
sleep 20; guard
echo "=== B patched, PAD_TAIL $(date -Is)"
PIGEMMA_PAD_TAIL=1 bench B_pad "$FAST" 2,3,4,6,7,8,16
sleep 20; guard
echo "=== C patched, PAD_TAIL + CHUNK_KB $(date -Is)"
for kb in 64 128 256; do
  echo "--- chunk ${kb} KiB"
  PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=$kb bench C_chunk$kb "$FAST" 8,16,64,512
done
sleep 20; guard
echo "=== D patched, PAD_TAIL + CHUNK_KB 128 + STRIP_PREFILL $(date -Is)"
PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=128 PIGEMMA_STRIP_PREFILL=1 bench D_strip "$FAST" 4,8,16,64,512
echo "--- STRIP_PREFILL alone"
PIGEMMA_STRIP_PREFILL=1 bench D_strip_only "$FAST" 16,64,512
sleep 20; guard
echo "=== E decode check $(date -Is)"
nice -n 10 "$REF/llama-bench" -m "$E2B" -t 4 -p 0 -n 96 -r 2 -o md 2> "$R/E_ref.err" | tee "$R/E_ref.md" | grep -E "tg"
PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=128 PIGEMMA_STRIP_PREFILL=1 nice -n 10 "$FAST/llama-bench" -m "$E2B" -t 4 -p 0 -n 96 -r 2 -o md 2> "$R/E_fast.err" | tee "$R/E_fast.md" | grep -E "tg"
sleep 20; guard
echo "=== F server, long prompts, reasoning off $(date -Is)"
python3 mtp_ab.py --bin "$REF/llama-server" --model "$E2B" --label fast/F_ref --prompts-file ~/pigemma/prompts_long64.json \
  --common "--reasoning-budget 0" --max-tokens 64 --config "base=" --order base --port 18180
PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=128 PIGEMMA_STRIP_PREFILL=1 python3 mtp_ab.py --bin "$FAST/llama-server" --model "$E2B" \
  --label fast/F_fast --prompts-file ~/pigemma/prompts_long64.json --common "--reasoning-budget 0" --max-tokens 64 \
  --config "base=" --config "b64=-b 64 -ub 64" --order base,b64 --port 18185
guard
echo "=== FAST_BENCH_DONE $(date -Is)"
