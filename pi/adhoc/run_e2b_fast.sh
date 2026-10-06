#!/usr/bin/env bash
# pi-gemma4 ad-hoc: fastest single-stream configuration of the smallest full Gemma 4 (E2B) on the Pi.
#  A. fetch the QAT-matched MTP drafter (59 MB) and the QAT 2-bit file (2.19 GB) to the USB stick
#  B. speculative decoding A/B on E2B QAT Q4_0: base, mtp(n_max 3), base, mtp(n_max 3), mtp(n_max 6)
#  C. llama-bench of the 2-bit QAT file (memory traffic floor)
set -uo pipefail
cd ~/pigemma
D=/mnt/usb/pigemma/models/e2b-qat-q4_0
./fetch_to_usb.sh unsloth/gemma-4-E2B-it-qat-GGUF mtp-gemma-4-E2B-it.gguf 59235648 586f2460b909008640981ec34060aa864e03c144fbabfb3173c4335087e4aae0 $D || exit 1
./fetch_to_usb.sh unsloth/gemma-4-E2B-it-qat-GGUF gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf 2186186784 0a5bbc20f91f92da96ab4870fa71b356c45b8500a7b8b9c3e0eb48359b72da28 $D || exit 1
exec 9> ~/pigemma/.bench.lock
flock -w 3600 9 || { echo "REFUSED: lock busy for an hour"; exit 4; }
OURS=~/pigemma/llama.cpp/build/bin
E2B=$D/gemma-4-E2B_q4_0-it.gguf
MTP=$D/mtp-gemma-4-E2B-it.gguf
Q2=$D/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf
health() { echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') temp=$(vcgencmd measure_temp) $(vcgencmd get_throttled) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB"; }
health
echo "=== B MTP A/B on E2B QAT Q4_0 $(date -Is)"
python3 mtp_ab.py --bin $OURS/llama-server --model "$E2B" --label speed-ceiling/e2b_mtp_ab --max-tokens 128 \
  --config "base=" --config "mtp3=-md $MTP --spec-type draft-mtp" --config "mtp6=-md $MTP --spec-type draft-mtp --spec-draft-n-max 6" \
  --order base,mtp3,base,mtp3,mtp6
sleep 30; health
echo "=== C llama-bench E2B QAT 2-bit (UD-Q2_K_XL) $(date -Is)"
nice -n 10 $OURS/llama-bench -m "$Q2" -t 4 -p 512 -n 128 -r 3 -o md 2> ~/pigemma/results/speed-ceiling/bench_e2b_q2.err | tee ~/pigemma/results/speed-ceiling/bench_e2b_q2.md
health
echo "=== E2B_FAST_DONE $(date -Is)"
