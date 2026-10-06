#!/usr/bin/env bash
# One bounded sweep for the throughput estimate: longer drafts gated by the drafter's confidence.
# E2B fast profile (all switches), dense drafter head, the 50 prompts of the drafter-head comparison.
# Baseline on the same prompts: results/fast3/centroids/r1_dense (3 drafts per pass, no gate).
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi benchmark holds the lock"; exit 4; }
FAST=/mnt/usb/pigemma/fast/bin-v7/llama-server
D2=/mnt/usb/pigemma/models/e2b-qat-q4_0
E2B=$D2/gemma-4-E2B_q4_0-it.gguf
MTP="-md $D2/mtp-gemma-4-E2B-it.gguf --spec-type draft-mtp"
ALL="@PIGEMMA_STRIP_PREFILL=1 @PIGEMMA_PAD_TAIL=1 @PIGEMMA_CHUNK_KB=64 @PIGEMMA_SPEC_STATS=1"
port=18490
for v in "n7_p00:--spec-draft-n-max 7" "n7_p70:--spec-draft-n-max 7 --spec-draft-p-min 0.7" "n7_p90:--spec-draft-n-max 7 --spec-draft-p-min 0.9" "n15_p90:--spec-draft-n-max 15 --spec-draft-p-min 0.9"; do
  name=${v%%:*}; flags=${v#*:}; port=$((port + 1))
  temp=$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp)
  echo "--- draftlen/$name $(date +%T) temp=${temp}C $(vcgencmd get_throttled) $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ')"
  python3 mtp_ab.py --bin "$FAST" --model "$E2B" --label "fast3/draftlen/$name" --prompts-file ~/pigemma/fast/prompts_lang.json \
    --common "--reasoning off" --stream --cooldown 10 --port "$port" --config "c=$ALL $MTP $flags" --order c 2>&1 | grep -E "error|AB_DONE"
done
echo "=== DRAFTLEN_DONE $(date -Is)"
