#!/usr/bin/env bash
# Greedy answers to the 36 QA items on both 26B files on the Pi (scored later on the desktop).
set -uo pipefail
cd ~/pigemma
python3 serve_measure.py --model models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf --label qa-26b-q3km --n-predict 64 --prompts qa_all_prompts.json
python3 serve_measure.py --model /mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf --label qa-26b-qat-q4_0 --n-predict 64 --prompts qa_all_prompts.json
echo QA_BOTH_DONE
