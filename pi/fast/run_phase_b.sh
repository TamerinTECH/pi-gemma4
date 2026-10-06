#!/usr/bin/env bash
# Phase B (brief 3, parts 3-6): matched benchmark, ablation, drafter vocabulary shortcut, model screen.
#   run_phase_b.sh [steps]      steps = comma list out of: matched,ablation,centroids,screen_e2b,screen_e4b,screen_26b,
#                               think_e2b   (default: all in this order)
# Every step is a series of llama-server runs driven by mtp_ab.py (streaming requests, greedy, one stream),
# one job at a time (flock), nice 10, services only read, abort when health degrades.
# Results: ~/pigemma/results/fast3/<step>/<run>/ab.json (+ server logs).
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi benchmark holds the lock"; exit 4; }
steps=",${1:-matched,ablation,centroids,screen_e2b,screen_e4b,screen_26b,think_e2b},"
REF=~/pigemma/llama.cpp/build/bin/llama-server
FAST=/mnt/usb/pigemma/fast/bin-v7/llama-server
P=~/pigemma/fast
D2=/mnt/usb/pigemma/models/e2b-qat-q4_0
D4=/mnt/usb/pigemma/models/e4b-qat-q4_0
D26=/mnt/usb/pigemma/models/26b-qat-q4_0
E2B=$D2/gemma-4-E2B_q4_0-it.gguf
E4B=$D4/gemma-4-E4B_q4_0-it-00001-of-00002.gguf
M26=$D26/gemma-4-26B_q4_0-it-00001-of-00005.gguf
MTP2="-md $D2/mtp-gemma-4-E2B-it.gguf --spec-type draft-mtp"
MTP2C="-md $D2/mtp-gemma-4-E2B-it-centroids.gguf --spec-type draft-mtp"
MTP4="-md $D4/mtp-gemma-4-E4B-it.gguf --spec-type draft-mtp"
MTP26="-md $HOME/pigemma/models/mtp-gemma-4-26B-A4B-it-Q4_0.gguf --spec-type draft-mtp"
ALL="@PIGEMMA_STRIP_PREFILL=1 @PIGEMMA_PAD_TAIL=1 @PIGEMMA_CHUNK_KB=64"
ST="@PIGEMMA_SPEC_STATS=1"
port=18410
guard() {
  local svc temp thr mem
  svc=$(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ')
  temp=$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp)
  thr=$(vcgencmd get_throttled | cut -d= -f2)
  mem=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  echo "[health $(date +%T)] $svc temp=${temp}C throttled=$thr memavail=${mem}kB"
  case "$svc" in *inactive*|*failed*|*activating*) echo "ABORT: a service is not active"; exit 9;; esac
  [ "$thr" = "0x0" ] || { echo "ABORT: throttled=$thr"; exit 9; }
  awk -v t="$temp" 'BEGIN{exit !(t<78)}' || { echo "cooling 60 s (temperature $temp)"; sleep 60; }
}
one() {  # one <label> <binary> <model> <prompts> <config string> [common server flags] [extra mtp_ab args...]
  local label=$1 bin=$2 model=$3 prompts=$4 cfg=$5 common=${6:---reasoning off}; shift 6 2>/dev/null || shift $#
  port=$((port + 1))
  guard
  echo "--- $label $(date +%T)"
  python3 mtp_ab.py --bin "$bin" --model "$model" --label "fast3/$label" --prompts-file "$prompts" --common "$common" \
    --stream --cooldown 10 --config "c=$cfg" --order c --port "$port" "$@" > "results/fast3_$(echo "$label" | tr '/' '_').log" 2>&1
  tail -1 "results/fast3_$(echo "$label" | tr '/' '_').log" | cut -c1-160
}
want() { case "$steps" in *",$1,"*) return 0;; *) return 1;; esac; }
mkdir -p results/fast3
guard

if want matched; then
  echo "=== matched: reference/plain, patched/plain, reference/drafter, patched/drafter, 3 alternated repetitions $(date -Is)"
  for rep in 1 2 3; do
    one "matched/r${rep}_ref_plain"  "$REF"  "$E2B" "$P/prompts_matched.json" ""
    one "matched/r${rep}_fast_plain" "$FAST" "$E2B" "$P/prompts_matched.json" "$ALL"
    one "matched/r${rep}_ref_mtp"    "$REF"  "$E2B" "$P/prompts_matched.json" "$MTP2"
    one "matched/r${rep}_fast_mtp"   "$FAST" "$E2B" "$P/prompts_matched.json" "$ALL $MTP2"
  done
fi

if want ablation; then
  echo "=== ablation: each switch alone, patched binary $(date -Is)"
  for v in "none:" "strip:@PIGEMMA_STRIP_PREFILL=1" "pad:@PIGEMMA_PAD_TAIL=1" "chunk:@PIGEMMA_CHUNK_KB=64" "all:$ALL"; do
    name=${v%%:*}; env=${v#*:}
    one "ablation/plain_$name" "$FAST" "$E2B" "$P/prompts_ablation.json" "$env"
    one "ablation/mtp3_$name"  "$FAST" "$E2B" "$P/prompts_ablation.json" "$env $MTP2"
  done
  one "ablation/mtp2_none" "$FAST" "$E2B" "$P/prompts_ablation.json" "$MTP2 --spec-draft-n-max 2"
  one "ablation/mtp2_pad"  "$FAST" "$E2B" "$P/prompts_ablation.json" "@PIGEMMA_PAD_TAIL=1 $MTP2 --spec-draft-n-max 2"
fi

if want centroids; then
  echo "=== drafter vocabulary shortcut: dense head vs cluster head (32 and 64 clusters) $(date -Is)"
  one "centroids/r1_dense"  "$FAST" "$E2B" "$P/prompts_lang.json" "$ALL $ST $MTP2"
  one "centroids/r1_cent32" "$FAST" "$E2B" "$P/prompts_lang.json" "$ALL $ST @PIGEMMA_MTP_CENTROIDS=1 $MTP2C"
  one "centroids/r1_cent64" "$FAST" "$E2B" "$P/prompts_lang.json" "$ALL $ST @PIGEMMA_MTP_CENTROIDS=64 $MTP2C"
  one "centroids/r2_cent32" "$FAST" "$E2B" "$P/prompts_lang.json" "$ALL $ST @PIGEMMA_MTP_CENTROIDS=1 $MTP2C"
  one "centroids/r2_dense"  "$FAST" "$E2B" "$P/prompts_lang.json" "$ALL $ST $MTP2"
fi

if want screen_e2b; then
  echo "=== model screen, E2B, direct answers, task budgets $(date -Is)"
  one "screen/e2b_ref_plain" "$REF"  "$E2B" "$P/prompts_qa_budget.json" ""
  one "screen/e2b_fast_mtp"  "$FAST" "$E2B" "$P/prompts_qa_budget.json" "$ALL $ST $MTP2"
fi

if want screen_e4b; then
  echo "=== model screen, E4B $(date -Is)"
  guard
  echo "--- E4B prompt test and decode, reference vs patched (layer skip only)"
  nice -n 10 "$(dirname "$REF")/llama-bench" -m "$E4B" -t 4 -p 512 -n 48 -r 2 -o md 2> results/fast3/bench_e4b_ref.err | tee results/fast3/bench_e4b_ref.md | grep -E "pp|tg"
  PIGEMMA_STRIP_PREFILL=1 nice -n 10 "$(dirname "$FAST")/llama-bench" -m "$E4B" -t 4 -p 512 -n 48 -r 2 -o md 2> results/fast3/bench_e4b_fast.err | tee results/fast3/bench_e4b_fast.md | grep -E "pp|tg"
  one "screen/e4b_ref_plain"  "$REF"  "$E4B" "$P/prompts_qa_budget.json" ""
  one "screen/e4b_fast_mtp"   "$FAST" "$E4B" "$P/prompts_qa_budget.json" "$ALL $ST $MTP4"
fi

if want screen_26b; then
  echo "=== model screen, 26B-A4B (weights stay mapped: -nr), without and with its assistant $(date -Is)"
  one "screen/26b_ref_plain" "$REF" "$M26" "$P/prompts_qa_budget.json" "-nr"
  one "screen/26b_ref_mtp"   "$REF" "$M26" "$P/prompts_qa_budget.json" "-nr $MTP26"
fi

if want think_e2b; then
  echo "=== separately labelled: E2B with thinking enabled (default reasoning), 768-token budget $(date -Is)"
  one "think/e2b_fast_mtp" "$FAST" "$E2B" "$P/prompts_qa_think.json" "$ALL $ST $MTP2" "--reasoning on"
fi
guard
echo "=== PHASE_B_DONE $(date -Is)"
