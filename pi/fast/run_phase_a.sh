#!/usr/bin/env bash
# Phase A (2026-10-06 evening): bring the Pi tree to patch v7, then run the correctness scenarios for E2B and E4B.
set -uo pipefail
cd ~/pigemma/fast
while pgrep -f "[b]uild_fast.sh" > /dev/null; do sleep 10; done
echo "=== build v7 $(date -Is)"
nice -n 10 ./build_fast.sh llama-fast-v6 bin-v7 2>&1 | tail -8
echo "=== fastcheck E2B $(date -Is)"
./tests/run_fastcheck.sh llama-fast-v6 /mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf e2b-v7 2>&1 | grep -E "^===|SUMMARY|PAIR|health|REFUSED|error|s7\.spec" | cut -c1-240
rm -f /mnt/pigemma-fast/fastcheck-e2b-v7/*.bin
echo "=== fastcheck E4B $(date -Is)"
./tests/run_fastcheck.sh llama-fast-v6 /mnt/usb/pigemma/models/e4b-qat-q4_0/gemma-4-E4B_q4_0-it-00001-of-00002.gguf e4b-v7 2>&1 | grep -E "^===|SUMMARY|PAIR|health|REFUSED|error|s7\.spec" | cut -c1-240
rm -f /mnt/pigemma-fast/fastcheck-e4b-v7/*.bin
echo "=== PHASE_A_DONE $(date -Is)"
