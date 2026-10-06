#!/usr/bin/env bash
# pigemma-fast, second round (2026-10-06): patched build v3 (pad once per call, layer skip usable with the drafter).
#   1  prompt test at batch widths 2,3,6,7,8 with PAD_TAIL + CHUNK_KB 64
#   2  MTP drafter + layer skip: long prompts, reference server vs patched server (prompt speed, same text)
#   3  MTP drafter on the six standard prompts, reference vs patched (decode speed, same text)
#   4  parallel streams, reference vs patched
# nice 10, one job at a time (flock), services only read; aborts when health degrades.
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi benchmark holds the lock"; exit 4; }
REF=~/pigemma/llama.cpp/build/bin
FAST=/mnt/usb/pigemma/fast/bin
D=/mnt/usb/pigemma/models/e2b-qat-q4_0
E2B=$D/gemma-4-E2B_q4_0-it.gguf
MTP=$D/mtp-gemma-4-E2B-it.gguf
R=~/pigemma/results/fast; mkdir -p "$R"
export_fast() { export PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=64 PIGEMMA_STRIP_PREFILL=1; }
unset_fast() { unset PIGEMMA_PAD_TAIL PIGEMMA_CHUNK_KB PIGEMMA_STRIP_PREFILL; }
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
guard
echo "=== 1 patched, PAD_TAIL + CHUNK 64, no layer skip $(date -Is)"
PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=64 nice -n 10 "$FAST/llama-bench" -m "$E2B" -t 4 -p 504 -n 0 -b 2,3,6,7,8,12 -r 2 -o md 2> "$R/P_pad2.err" | tee "$R/P_pad2.md" | grep -E "pp"
sleep 20; guard
S="-md $MTP --spec-type draft-mtp"
echo "=== 2 MTP + long prompts, reference $(date -Is)"
python3 mtp_ab.py --bin "$REF/llama-server" --model "$E2B" --label fast/G_ref_mtp --prompts-file ~/pigemma/prompts_long64.json \
  --common "--reasoning-budget 0" --config "mtp3=$S" --order mtp3 --port 18190
echo "=== 2 MTP + long prompts, patched $(date -Is)"
export_fast
python3 mtp_ab.py --bin "$FAST/llama-server" --model "$E2B" --label fast/G_fast_mtp --prompts-file ~/pigemma/prompts_long64.json \
  --common "--reasoning-budget 0" --config "mtp3=$S" --order mtp3 --port 18192
unset_fast
sleep 20; guard
echo "=== 3 MTP, six standard prompts, reference $(date -Is)"
python3 mtp_ab.py --bin "$REF/llama-server" --model "$E2B" --label fast/H_ref_mtp --max-tokens 128 \
  --config "base=" --config "mtp3=$S" --order base,mtp3 --port 18194
echo "=== 3 MTP, six standard prompts, patched $(date -Is)"
export_fast
python3 mtp_ab.py --bin "$FAST/llama-server" --model "$E2B" --label fast/H_fast_mtp --max-tokens 128 \
  --config "base=" --config "mtp3=$S" --config "mtp2=$S --spec-draft-n-max 2" --order base,mtp3,mtp2 --port 18197
unset_fast
sleep 30; guard
echo "=== 4 parallel streams, reference $(date -Is)"
python3 par_throughput.py --bin "$REF/llama-server" --model "$E2B" --label fast/I_par_ref --streams 1,2,3,4,8,16 --n-predict 96 --port 18200
sleep 30; guard
echo "=== 4 parallel streams, patched $(date -Is)"
export_fast
python3 par_throughput.py --bin "$FAST/llama-server" --model "$E2B" --label fast/I_par_fast --streams 1,2,3,4,8,16 --n-predict 96 --port 18202
unset_fast
guard
echo "=== FAST_BENCH2_DONE $(date -Is)"
