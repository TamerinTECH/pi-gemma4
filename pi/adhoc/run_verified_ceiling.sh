#!/usr/bin/env bash
# pi-gemma4 ad-hoc (2026-10-06): upper bound of VERIFIED multi-token decoding on the Pi 5, smallest full Gemma 4 (E2B).
# Decode is DRAM-bandwidth bound (one pass over the touched weights per model pass), so the only exact lever left is
# committing several tokens per pass. This run measures how far that can go:
#   1. evaluation rate vs tokens per pass (llama-bench prompt test at batch 1..512)  -> ceiling for a FREE, PERFECT drafter
#   2. QAT 2-bit file (fewer bytes per pass)
#   3. MTP drafter: real acceptance at n_max 2/3/4, and SYNTHETIC 100 % acceptance at n_max 3/7/15 (benchmark-only flag)
#   4. the same on the 2-bit file
#   5. copy-heavy tasks (repeat / fix typos / rename variable) with the n-gram drafter, where acceptance is naturally high
# Full outputs are stored so base vs speculative text can be compared exactly.
# Resident only, nice 10, one job at a time (flock), services are only READ; aborts when health degrades.
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi benchmark holds the lock"; exit 4; }
OURS=~/pigemma/llama.cpp/build/bin
D=/mnt/usb/pigemma/models/e2b-qat-q4_0
E2B=$D/gemma-4-E2B_q4_0-it.gguf
Q2=$D/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf
MTP=$D/mtp-gemma-4-E2B-it.gguf
R=~/pigemma/results/verified-ceiling; mkdir -p "$R"
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
cool() { sleep "${1:-30}"; }
guard
echo "=== 1 eval rate vs tokens per pass, E2B QAT Q4_0 $(date -Is)"
nice -n 10 $OURS/llama-bench -m "$E2B" -t 4 -p 512 -n 0 -b 1,2,4,8,16,32,64,128,512 -r 2 -o md 2> "$R/bench_e2b_batch.err" | tee "$R/bench_e2b_batch.md"
cool; guard
echo "=== 2 llama-bench E2B QAT 2-bit (UD-Q2_K_XL) $(date -Is)"
nice -n 10 $OURS/llama-bench -m "$Q2" -t 4 -p 512 -n 128 -r 3 -o md 2> "$R/bench_e2b_q2.err" | tee "$R/bench_e2b_q2.md"
nice -n 10 $OURS/llama-bench -m "$Q2" -t 4 -p 512 -n 0 -b 4,8,16 -r 2 -o md 2> "$R/bench_e2b_q2_batch.err" | tee "$R/bench_e2b_q2_batch.md"
cool; guard
echo "=== 3 MTP real and synthetic acceptance, E2B QAT Q4_0 $(date -Is)"
S="-md $MTP --spec-type draft-mtp"
python3 mtp_ab.py --bin $OURS/llama-server --model "$E2B" --label verified-ceiling/e2b_q4_mtp --max-tokens 128 \
  --config "base=" --config "mtp3=$S" --config "mtp2=$S --spec-draft-n-max 2" --config "mtp4=$S --spec-draft-n-max 4" \
  --config "synth3=$S --spec-draft-n-max 3 --spec-synth-rates 1,1,1" \
  --config "synth7=$S --spec-draft-n-max 7 --spec-synth-rates 1,1,1,1,1,1,1" \
  --config "synth15=$S --spec-draft-n-max 15 --spec-synth-rates 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" \
  --order base,mtp3,mtp2,mtp4,synth3,synth7,synth15
cool; guard
echo "=== 4 MTP on the 2-bit file $(date -Is)"
python3 mtp_ab.py --bin $OURS/llama-server --model "$Q2" --label verified-ceiling/e2b_q2_mtp --max-tokens 128 \
  --config "base=" --config "mtp3=$S" --order base,mtp3 --port 18130
cool; guard
echo "=== 5 copy-heavy tasks, n-gram drafter, E2B QAT Q4_0, thinking off $(date -Is)"
python3 mtp_ab.py --bin $OURS/llama-server --model "$E2B" --label verified-ceiling/e2b_q4_copy --prompts-file ~/pigemma/prompts_copy.json \
  --common "--reasoning-budget 0" --ctx 4096 --port 18140 \
  --config "base=" --config "ngram=--spec-type ngram-mod" \
  --config "ngram_s=--spec-type ngram-mod --spec-ngram-mod-n-match 8 --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-max 32" \
  --config "mtp3=$S" --config "both=$S --spec-type ngram-mod,draft-mtp" \
  --order base,ngram,ngram_s,mtp3,both
guard
echo "=== VERIFIED_CEILING_DONE $(date -Is)"
