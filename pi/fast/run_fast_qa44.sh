#!/usr/bin/env bash
# pigemma-fast (2026-10-06): broader same-text check on the 44 held-out questions (EN/HE/DE/code), reasoning off,
# 96 tokens: reference build plain  vs  patched build with all switches + MTP drafter (the "fast profile").
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi benchmark holds the lock"; exit 4; }
REF=~/pigemma/llama.cpp/build/bin
FAST=/mnt/usb/pigemma/fast/bin
D=/mnt/usb/pigemma/models/e2b-qat-q4_0
E2B=$D/gemma-4-E2B_q4_0-it.gguf
S="-md $D/mtp-gemma-4-E2B-it.gguf --spec-type draft-mtp"
COMMON=${1:---reasoning-budget 0}   # server flags for both sides, e.g. "--reasoning off"
TAG=${2:-Q}                         # result label prefix
h() { echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') $(vcgencmd measure_temp) $(vcgencmd get_throttled)"; }
h
echo "=== reference, plain $(date -Is)"
python3 mtp_ab.py --bin "$REF/llama-server" --model "$E2B" --label "fast/${TAG}_ref_plain" --prompts-file ~/pigemma/prompts_qa44.json \
  --common "$COMMON" --config "base=" --order base --port 18400 > ~/pigemma/results/fast/${TAG}_ref.log 2>&1
tail -1 ~/pigemma/results/fast/${TAG}_ref.log
sleep 30; h
echo "=== patched, fast profile $(date -Is)"
PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=64 PIGEMMA_STRIP_PREFILL=1 python3 mtp_ab.py --bin "$FAST/llama-server" --model "$E2B" \
  --label "fast/${TAG}_fast_mtp" --prompts-file ~/pigemma/prompts_qa44.json --common "$COMMON" \
  --config "mtp3=$S" --order mtp3 --port 18402 > ~/pigemma/results/fast/${TAG}_fast.log 2>&1
tail -1 ~/pigemma/results/fast/${TAG}_fast.log
h
echo "=== FAST_QA44_DONE $(date -Is)"
