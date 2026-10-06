#!/usr/bin/env bash
# Dense 31B (UD-IQ3_XXS, split parts on the USB stick): no-draft vs MTP drafter, same prompts, greedy.
set -uo pipefail
cd ~/pigemma
D=/mnt/usb/pigemma/models/31b-iq3xxs
M=$D/gemma-4-31B-it-UD-IQ3_XXS-00001-of-00004.gguf
MTP=$D/mtp-gemma-4-31B-it-Q4_0.gguf
echo "=== base $(date -Is)"
python3 serve_measure.py --model "$M" --label 31b-iq3xxs-base --n-predict 48
echo "=== mtp $(date -Is)"
python3 serve_measure.py --model "$M" --label 31b-iq3xxs-mtp --n-predict 48 --extra "-md $MTP --spec-type draft-mtp"
grep -h "draft acceptance" results/31b-iq3xxs-mtp/server.log || true
echo "=== AB_31B_DONE $(date -Is)"
