#!/usr/bin/env bash
# P1 baseline audit: ONE resident (no memory pressure) reference run of the 26B UD-Q3_K_M on the microSD.
# Same server command as the 2026-10-05 series (llama-server -nr -t 4 -c 4096 -np 1, nice 10), n_predict 48,
# the balloon-8 prompts (en_short, he_short) in the legacy format twice (A ... B) and once in the canonical
# Gemma 4 format; --stream for TTFT / inter-token latency; --cold = fadvise(DONTNEED) on THIS file only.
set -uo pipefail
cd ~/pigemma
LOCK=~/pigemma/.pi_experiment.lock
if [ -e "$LOCK" ]; then echo "REFUSED: $LOCK exists: $(cat $LOCK)"; exit 4; fi
echo "p1-baseline-audit resident reference run, pid $$, started $(date -Is)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo); free=$(df -B1 --output=avail / | tail -1)
echo "pre: MemAvailable_kB=$avail root_free=$free $(vcgencmd get_throttled) $(vcgencmd measure_temp)"
for s in contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base; do echo "pre: $s $(systemctl is-active $s)"; done
[ "$avail" -gt 3145728 ] || { echo "ABORT: MemAvailable < 3 GB"; exit 5; }
[ "$free" -gt 2000000000 ] || { echo "ABORT: root free < 2 GB (1e9)"; exit 5; }
python3 serve_measure.py --model models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf --label p1-resident-ref-26b-q3km \
  --n-predict 48 --prompts /tmp/p1_prompts.json --cold --stream
rc=$?
for s in contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base; do echo "post: $s $(systemctl is-active $s)"; done
echo "post: MemAvailable_kB=$(awk '/MemAvailable/{print $2}' /proc/meminfo) root_free=$(df -B1 --output=avail / | tail -1) $(vcgencmd get_throttled)"
echo "P1_REF_DONE rc=$rc $(date -Is)"
