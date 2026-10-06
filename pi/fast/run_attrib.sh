#!/usr/bin/env bash
# Attribution of the one answer (code-01, 253 tokens) whose text differed between reference/plain and the fast
# profile with drafter in the model screen: which component changes the greedy path?
# Reference build: plain, drafter. Patched build: plain and drafter, each with no switch, each switch alone, all.
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi benchmark holds the lock"; exit 4; }
REF=~/pigemma/llama.cpp/build/bin/llama-server
FAST=/mnt/usb/pigemma/fast/bin-v7/llama-server
D2=/mnt/usb/pigemma/models/e2b-qat-q4_0
E2B=$D2/gemma-4-E2B_q4_0-it.gguf
MTP="-md $D2/mtp-gemma-4-E2B-it.gguf --spec-type draft-mtp"
S="@PIGEMMA_STRIP_PREFILL=1"; P="@PIGEMMA_PAD_TAIL=1"; C="@PIGEMMA_CHUNK_KB=64"
python3 mtp_ab.py --bin "$REF" --model "$E2B" --label fast3/attrib/ref --prompts-file ~/pigemma/fast/prompts_code01.json \
  --common "--reasoning off" --stream --cooldown 5 --port 18470 \
  --config "plain=" --config "mtp=$MTP" --config "plain_again=" 2>&1 | grep -E "error|AB_DONE"
python3 mtp_ab.py --bin "$FAST" --model "$E2B" --label fast3/attrib/fast --prompts-file ~/pigemma/fast/prompts_code01.json \
  --common "--reasoning off" --stream --cooldown 5 --port 18480 \
  --config "plain_none=" --config "plain_all=$S $P $C" --config "mtp_none=$MTP" --config "mtp_strip=$S $MTP" \
  --config "mtp_pad=$P $MTP" --config "mtp_chunk=$C $MTP" --config "mtp_all=$S $P $C $MTP" 2>&1 | grep -E "error|AB_DONE"
echo "=== ATTRIB_DONE $(date -Is)"
