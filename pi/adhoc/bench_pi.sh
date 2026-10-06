#!/usr/bin/env bash
# pi-gemma4 ad-hoc: measure one GGUF on the real Pi 5 (16 GB, microSD) without disturbing services.
# Usage: bench_pi.sh <model.gguf> <label> [threads] [n_gen]
# Steps: (1) cold load from SD (page cache dropped for THIS FILE only via fadvise DONTNEED, nothing else is evicted)
#        (2) llama-bench tg/pp in -nr (mmap) mode and default (repack) mode when memory allows
#        (3) greedy chat generation EN + HE with llama-completion -nr, timings + text
#        (4) thermals/throttling + memory before/after
# All runs nice -n 10, no mlock, services untouched.
set -euo pipefail
model=$1; label=$2; t=${3:-4}; ngen=${4:-64}
bin=~/pigemma/llama.cpp/build/bin
out=~/pigemma/results/$label; mkdir -p "$out"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$out/run.log"; }
snap() { # name
  { echo "== $1 $(date -Is)"; vcgencmd measure_temp; vcgencmd get_throttled; vcgencmd measure_clock arm;
    grep -E "MemTotal|MemAvailable|^Cached|AnonPages|Mapped|SwapFree" /proc/meminfo; cat /proc/loadavg; } >> "$out/system.log"
}
peak_wait() { # pid -> echoes peak VmHWM kB while waiting
  local pid=$1 peak=0 r
  while kill -0 "$pid" 2>/dev/null; do
    r=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
    [ "${r:-0}" -gt "$peak" ] && peak=$r
    sleep 1
  done
  echo "$peak"
}
log "model=$model size=$(stat -c %s "$model") threads=$t ngen=$ngen"
log "free_root=$(df -B1 --output=avail / | tail -1) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
snap start

# (1) cold start: evict only this file's pages, then time a full load + 8 tokens
python3 - "$model" <<'PY'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
PY
log "cold load + 8 tokens (-nr)"
c0=$(date +%s)
nice -n 10 "$bin/llama-completion" -m "$model" -nr -t "$t" -c 1024 -n 8 --temp 0 -no-cnv -p "Hello" \
  > "$out/cold.stdout" 2> "$out/cold.stderr" &
pid=$!
peak=$(peak_wait $pid)
rc=0; wait $pid || rc=$?
c1=$(date +%s)
log "cold rc=$rc wall_s=$((c1-c0)) peak_rss_kB=$peak"
grep -E "load time|eval time" "$out/cold.stderr" | sed 's/.*common_perf_print: *//' | tee -a "$out/run.log" || true
grep -E "model buffer size" "$out/cold.stderr" | tail -4 | tee -a "$out/run.log" || true
snap after-cold

# (2) warm llama-bench, mmap/no-repack mode
log "llama-bench --repack 0"
nice -n 10 "$bin/llama-bench" -m "$model" -t "$t" -p 64 -n 32 -r 2 --repack 0 -o md 2> "$out/bench-nr.stderr" | tee "$out/bench-nr.md" | tee -a "$out/run.log" || log "bench-nr failed"
snap after-bench-nr
# default (repack) mode only when the anon copy fits with >= 2 GB headroom
avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo); size=$(stat -c %s "$model")
if [ $(( avail_kb*1024 - size )) -gt 2147483648 ]; then
  log "llama-bench --repack 1 (default)"
  nice -n 10 "$bin/llama-bench" -m "$model" -t "$t" -p 64 -n 32 -r 2 --repack 1 -o md 2> "$out/bench-repack.stderr" | tee "$out/bench-repack.md" | tee -a "$out/run.log" || log "bench-repack failed"
else
  log "skip default/repack mode: anon copy would not leave 2 GB (avail=${avail_kb} kB, size=$size)"
fi
snap after-bench

# (3) real chat generation, EN and HE
gen() { # name prompt
  printf '<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n' "$2" > "$out/prompt-$1.txt"
  nice -n 10 "$bin/llama-completion" -m "$model" -nr -t "$t" -c 2048 -n "$ngen" --temp 0 -no-cnv -f "$out/prompt-$1.txt" \
    > "$out/gen-$1.stdout" 2> "$out/gen-$1.stderr" || log "gen $1 rc=$?"
  log "gen $1: $(grep -E 'prompt eval time|  eval time' "$out/gen-$1.stderr" | sed 's/.*common_perf_print: *//' | tr '\n' ' ')"
}
gen en "In three short sentences: why is a mixture-of-experts model faster to run than a dense model of the same size?"
gen he "ענה בעברית בשלושה משפטים קצרים: מה היתרון של מחשב קטן כמו Raspberry Pi?"
snap end
log "free_root=$(df -B1 --output=avail / | tail -1)"
log BENCH_DONE
