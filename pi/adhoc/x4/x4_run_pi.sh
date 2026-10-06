#!/usr/bin/env bash
# X4 resident-bottleneck measurements ON THE REAL PI (context-friend-1). Resident runs only: no balloon,
# no drop_caches, no swapoff, no mlock, no service is touched, nothing is written to / except this kit (< 1 MB).
#
# Kit layout on the Pi (copied by pc/x4/push_kit_to_pi.sh):
#   ~/pigemma/x4/{x4_run_pi.sh,x4_compare.py,x4_prompts.json,x4_plan_*.json,membw.c}
#   /dev/shm/pigemma-x4-bin/{membw,kernbench}          (tmpfs; kernbench is cross-built on the desktop)
# Results: /mnt/usb/pigemma/x4/results (USB stick, survives a reboot) or, if the stick is not mounted and
#   cannot be mounted, /dev/shm/pigemma-x4-results (tmpfs).
#
# Usage:
#   nohup bash ~/pigemma/x4/x4_run_pi.sh all > /dev/shm/x4_all.out 2>&1 &      # everything, ~1.5-2 h
#   bash ~/pigemma/x4/x4_run_pi.sh <step>     # one step: preflight | membw | kernbench | threads | longgen |
#                                             #           repack | qat | udref | membw2
# Every step takes the Pi bench lock (flock -n ~/pigemma/.bench.lock) and refuses to run when another
# llama-* / powerinfer* process exists, a ContextFriend service is not active, MemAvailable < 3 GiB or
# / has less than 2 GB free. Exit codes: 0 ok, 3 guard refused, 75 lock busy, other = step failure.
set -u
KIT="${X4_KIT:-$HOME/pigemma/x4}"
BIN="${X4_BIN:-/dev/shm/pigemma-x4-bin}"
LOCK="$HOME/pigemma/.bench.lock"
SERVICES="contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base"
MIN_AVAIL_KB=$((3 * 1024 * 1024))
MIN_ROOT_FREE=2000000000
STEP="${1:-}"

log() { echo "[x4 $(date +%H:%M:%S)] $*"; }

pick_out() {
  if [ -n "${X4_OUT:-}" ]; then echo "$X4_OUT"; return; fi
  if ! mountpoint -q /mnt/usb && [ -b /dev/sda1 ]; then
    sudo mount -o rw,uid=1000,gid=1000,umask=022,noatime /dev/sda1 /mnt/usb >/dev/null 2>&1 || true
  fi
  if mountpoint -q /mnt/usb && mkdir -p /mnt/usb/pigemma/x4/results 2>/dev/null; then
    echo /mnt/usb/pigemma/x4/results
  else
    mkdir -p /dev/shm/pigemma-x4-results; echo /dev/shm/pigemma-x4-results
  fi
}

state_line() {
  local avail root temp thr
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  root=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
  temp=$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
  thr=$(vcgencmd get_throttled 2>/dev/null)
  echo "MemAvailable_kB=$avail root_free=$root temp_c=$temp $thr freq_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) uptime_s=$(cut -d' ' -f1 /proc/uptime)"
}

guard() {
  local bad=0 s st avail root others
  for s in $SERVICES; do
    st=$(systemctl is-active "$s" 2>/dev/null)
    if [ "$st" != "active" ]; then log "GUARD: service $s is '$st'"; bad=1; fi
  done
  others=$(pgrep -a 'llama|powerinfer' 2>/dev/null | grep -v pgrep || true)
  if [ -n "$others" ]; then log "GUARD: other llama/powerinfer process: $others"; bad=1; fi
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$avail" -lt "$MIN_AVAIL_KB" ]; then log "GUARD: MemAvailable $avail kB < $MIN_AVAIL_KB kB"; bad=1; fi
  root=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
  if [ "$root" -lt "$MIN_ROOT_FREE" ]; then log "GUARD: / free $root B < $MIN_ROOT_FREE B"; bad=1; fi
  log "state: $(state_line)"
  return $bad
}

cool() {  # wait (max 7 min) until the SoC is below $1 degC
  local lim="$1" n=0 t
  while [ $n -lt 84 ]; do
    t=$(awk '{printf "%d", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
    [ "$t" -lt "$lim" ] && break
    sleep 5; n=$((n + 1))
  done
  log "cool-down: waited $((n * 5)) s, now ${t} C (limit $lim)"
}

build_membw() {
  mkdir -p "$BIN"
  if [ ! -x "$BIN/membw" ]; then
    log "building membw natively (gcc $(gcc -dumpversion))"
    nice -n 10 gcc -std=gnu11 -O3 -mcpu=native -Wall -Wextra -pthread -o "$BIN/membw" "$KIT/membw.c" || return 1
    { echo "built $(date -Is) with: gcc -std=gnu11 -O3 -mcpu=native -Wall -Wextra -pthread"; gcc --version | head -1; sha256sum "$BIN/membw" "$KIT/membw.c"; } > "$OUT/membw_BUILD_INFO.txt"
  fi
}

step_membw() {  # $1 = output tag
  local tag="$1"
  build_membw || { log "membw build FAILED"; return 1; }
  cool 60
  log "membw $tag: before: $(state_line)"
  # 1024 MiB arena = far beyond the 2 MiB L3; 256 MiB as a second size; best of 7 passes; threads 1..4
  nice -n 10 "$BIN/membw" --size-mb 1024 --threads 1,2,3,4 --reps 7 --json > "$OUT/membw_${tag}_1024.json" 2> "$OUT/membw_${tag}_1024.err"; local rc1=$?
  nice -n 10 "$BIN/membw" --size-mb 256 --threads 1,2,3,4 --reps 7 --json > "$OUT/membw_${tag}_256.json" 2> "$OUT/membw_${tag}_256.err"; local rc2=$?
  nice -n 10 "$BIN/membw" --size-mb 1024 --threads 4,3,2,1 --reps 7 --json > "$OUT/membw_${tag}_1024_rev.json" 2> "$OUT/membw_${tag}_1024_rev.err"; local rc3=$?
  log "membw $tag: after: $(state_line) rc=$rc1/$rc2/$rc3"
  cat "$OUT/membw_${tag}_1024.json"
  [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ]
}

step_kernbench() {
  if [ ! -x "$BIN/kernbench" ]; then log "kernbench binary missing in $BIN (cross-built on the desktop) - step skipped"; return 0; fi
  cool 60
  log "kernbench: before: $(state_line)"
  nice -n 10 "$BIN/kernbench" --types Q8_0,Q6_K,IQ3_XXS,IQ4_NL,IQ4_NL@704,IQ4_XS,Q4_0 --threads 1,2,3,4 \
      --stream-mb 512 --cache-kb 384 --cache-total-mb 256 --batch-cols 32 --reps 5 --json \
      > "$OUT/kernbench.json" 2> "$OUT/kernbench.err"; local rc=$?
  log "kernbench: after: $(state_line) rc=$rc"
  tail -3 "$OUT/kernbench.err"
  return $rc
}

step_plan() {  # $1 = plan name (threads|longgen|repack|qat|udref)
  local name="$1"
  if [ "$name" = "qat" ] && [ ! -f /mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf ]; then
    log "QAT split parts not found on /mnt/usb - step skipped"; return 0
  fi
  log "plan $name: start: $(state_line)"
  ( cd "$KIT" && python3 "$KIT/x4_compare.py" --plan "$KIT/x4_plan_${name}.json" --out "$OUT/$name" ) 2>&1 | tee "$OUT/${name}.out"
  local rc=${PIPESTATUS[0]}
  log "plan $name: end rc=$rc: $(state_line)"
  if grep -q "PLAN_ABORTED" "$OUT/${name}.out"; then return 4; fi
  return $rc
}

run_step() {
  case "$1" in
    preflight) guard ;;
    membw)     guard && step_membw before ;;
    membw2)    guard && step_membw after ;;
    kernbench) guard && step_kernbench ;;
    threads|longgen|repack|qat|udref) guard && step_plan "$1" ;;
    *) echo "unknown step '$1'"; return 2 ;;
  esac
}

if [ -z "$STEP" ]; then sed -n '2,20p' "$0"; exit 2; fi
if [ "${X4_LOCKED:-0}" != "1" ]; then
  mkdir -p "$(dirname "$LOCK")"
  X4_LOCKED=1 exec flock -n -E 75 "$LOCK" bash "$0" "$@"
fi

OUT="$(pick_out)"
mkdir -p "$OUT"
log "X4 step '$STEP' on $(hostname), kit $KIT, results $OUT"
{ echo "collected: $(date -Is)"; uname -a; echo "page size $(getconf PAGESIZE)"; tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo;
  grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo; cat /proc/cmdline; echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)";
  sha256sum "$HOME/pigemma/llama.cpp/build/bin/llama-server" 2>/dev/null; ls -l "$HOME/pigemma/models" /mnt/usb/pigemma/models/26b-qat-q4_0 2>/dev/null;
  df -B1 / /mnt/usb 2>/dev/null; } > "$OUT/env_${STEP}.txt" 2>&1

if [ "$STEP" = "all" ]; then
  rc_all=0
  for s in preflight membw kernbench threads longgen repack qat udref membw2; do
    log "===== step $s ====="
    run_step "$s"; rc=$?
    echo "$s rc=$rc $(date -Is)" >> "$OUT/steps.log"
    if [ $rc -ne 0 ]; then
      log "step $s failed (rc=$rc) - stopping the sequence"
      rc_all=$rc; break
    fi
    sleep 30
  done
  log "X4_ALL_DONE rc=$rc_all results=$OUT"
  for s in $SERVICES; do log "post: $s $(systemctl is-active "$s" 2>/dev/null)"; done
  log "post: $(state_line)"
  exit $rc_all
fi
run_step "$STEP"; rc=$?
echo "$STEP rc=$rc $(date -Is)" >> "$OUT/steps.log"
log "X4_STEP_DONE $STEP rc=$rc results=$OUT"
exit $rc
