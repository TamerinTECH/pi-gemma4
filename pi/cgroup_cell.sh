#!/usr/bin/env bash
# pi/cgroup_cell.sh - run ONE controlled-memory cell on the real Raspberry Pi inside a transient
# systemd scope (cgroup v2): MemoryMax=<budget>, MemorySwapMax=0, MemoryHigh=infinity, optional
# IOReadBandwidthMax. Pi counterpart of `python -m pigemma.controlled cell` (desktop Docker).
#
# STATUS 2026-10-05: the Pi kernel boots with cgroup_disable=memory and without PSI, so this script
# REFUSES (exit 4) until the operator enables the memory controller (append `cgroup_enable=memory
# cgroup_memory=1 psi=1` to /boot/firmware/cmdline.txt and reboot once - his decision, never done by
# this script). The refusal path is the only path verified on the Pi (read-only check):
#   ssh ... ${PI_HOST:?Set PI_HOST=user@host} 'bash -s -- --check' < pi/cgroup_cell.sh
#
# Usage (once enabled):
#   pi/cgroup_cell.sh --check                      readiness report (JSON), exit 0 ready / 4 refused
#   pi/cgroup_cell.sh --method A|B|C|D --budget-gib 8 --workload tuning|heldout [--repeat N]
#        [--conversations id,id] [--n-predict-cap N] [--ranges FILE] [--io-read-bps BYTES]
#        [--engine-id ID --server-bin PATH [--engine-parent ID] [--server-extra "words"]]
#        [--allow-pinner-on-protected-host] [--cell-id ID] [--dry-run]
#   Methods B/C/D start tools/pinner, whose own guard refuses to mlock on this host (it runs
#   ContextFriend services). --allow-pinner-on-protected-host passes the pinner's
#   --allow-protected-host: use it only with the operator's explicit approval. Inside the scope the locked
#   pages are bounded by MemoryMax; without the flag such a cell fails at setup with the pinner's
#   refusal (nothing is locked).
#
# What it guarantees (brief section 2, Pi safety rules in CLAUDE.md):
#   * the cell runner (pigemma/controlled_cell.py), the pinner and llama-server all start INSIDE the
#     scope, before the model loads, so mapped model pages, locked pages, KV and buffers are charged
#     to MemoryMax; swap is off for the scope (MemorySwapMax=0); memory.high stays "max";
#   * nice -n 10; one Pi performance experiment at a time: flock -n on ~/pigemma/.bench.lock (the
#     Pi lock every stage uses) and a refusal while any other llama-server / llama-completion /
#     llama-cli / powerinfer process runs; refuses when MemAvailable < budget + 3 GB, when / has
#     < 2 GB free, or when a ContextFriend service is not active; a watchdog stops ONLY this
#     cell's scope if a service turns unhealthy, and the runner stops its own llama-server if
#     MemAvailable falls below 3 GB;
#   * no global drop_caches, no swapoff, no balloon, no reboot, no cmdline.txt/config.txt edits:
#     cold starts use the runner's targeted posix_fadvise(DONTNEED) on the model file only.
set -euo pipefail

PIGEMMA_HOME="${PIGEMMA_HOME:-$HOME/pigemma}"
MODEL="${PIGEMMA_MODEL:-$PIGEMMA_HOME/models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf}"
MODEL_SIZE=12728497888
MODEL_SHA256=811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052
LLAMA_SERVER="${PIGEMMA_LLAMA_SERVER:-$PIGEMMA_HOME/llama.cpp/build/bin/llama-server}"
PINNER="${PIGEMMA_PINNER:-$PIGEMMA_HOME/tools/build/pinner}"
CODE_DIR="${PIGEMMA_CODE:-$PIGEMMA_HOME/code}"          # holds pigemma/ (controlled_cell.py, workloads.py, ...) + data/workloads
MAP="${PIGEMMA_MAP:-$CODE_DIR/results/maps/map26_ud_q3_k_m.json}"
SERVICES="${PIGEMMA_SERVICES:-contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base}"
MIN_AVAIL_EXTRA=$((3 * 1000 * 1000 * 1000))   # MemAvailable must stay >= 3 GB
MIN_ROOT_FREE=$((2 * 1000 * 1000 * 1000))
EXIT_REFUSED=4
LOCKFILE="$PIGEMMA_HOME/.bench.lock"          # the Pi lock shared by every stage (CLAUDE.md coordination rules)
# Test hook, honoured ONLY together with --dry-run (a dry run starts nothing): read the kernel
# interface files of the readiness check from a fake tree, so the argument handling, the generated
# config.json and the systemd-run command can be tested off the Pi. A real run always reads the
# real /proc and /sys, and the in-cgroup runner refuses anyway when its limit != the budget.
SYSROOT=""

say() { printf '%s\n' "$*" >&2; }
json_str() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

# ------------------------------------------------------------------------------ readiness checks
check() {
  local ok=1 reasons=()
  local cmdline; cmdline="$(cat "$SYSROOT/proc/cmdline" 2>/dev/null || true)"
  local v2=0; [[ -f "$SYSROOT/sys/fs/cgroup/cgroup.controllers" ]] && v2=1
  local mem_ctrl=0; [[ $v2 == 1 ]] && grep -qw memory "$SYSROOT/sys/fs/cgroup/cgroup.controllers" && mem_ctrl=1
  local mem_enabled; mem_enabled="$(awk '$1=="memory"{print $4}' "$SYSROOT/proc/cgroups" 2>/dev/null || true)"
  local psi=0; [[ -r "$SYSROOT/proc/pressure/memory" ]] && psi=1
  local io_ctrl=0; [[ $v2 == 1 ]] && grep -qw io "$SYSROOT/sys/fs/cgroup/cgroup.controllers" && io_ctrl=1
  local sdrun=0; command -v systemd-run >/dev/null 2>&1 && sdrun=1
  local disabled=0; [[ "$cmdline" == *cgroup_disable=memory* ]] && disabled=1
  if [[ $v2 != 1 ]]; then ok=0; reasons+=("cgroup v2 (unified hierarchy) is not mounted at /sys/fs/cgroup"); fi
  if [[ $disabled == 1 ]]; then ok=0; reasons+=("kernel command line contains cgroup_disable=memory"); fi
  if [[ $mem_ctrl != 1 ]]; then ok=0; reasons+=("the memory controller is not in /sys/fs/cgroup/cgroup.controllers (no memory.max/current/peak/stat/events)"); fi
  if [[ "${mem_enabled:-0}" != 1 ]]; then ok=0; reasons+=("/proc/cgroups reports the memory controller disabled (enabled=${mem_enabled:-?})"); fi
  if [[ $sdrun != 1 ]]; then ok=0; reasons+=("systemd-run not found"); fi
  if [[ $psi != 1 ]]; then reasons+=("WARNING: PSI is off (/proc/pressure absent; boot with psi=1) - memory.pressure will be unavailable"); fi
  local avail_kb; avail_kb="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
  local root_free; root_free="$(df -B1 --output=avail / | tail -1 | tr -d ' ')"
  local svc_json="" s st
  for s in $SERVICES; do
    st="$(systemctl is-active "$s" 2>/dev/null || true)"
    svc_json+="$(json_str "$s"):$(json_str "${st:-unknown}"),"
  done
  local reasons_json="" r
  for r in "${reasons[@]}"; do reasons_json+="$(json_str "$r"),"; done
  printf '{"tool":"pi/cgroup_cell.sh","check":"memory-controller-readiness","check_sysroot":%s,"host":%s,"kernel":%s,"cgroup_v2":%s,"memory_controller":%s,"proc_cgroups_memory_enabled":%s,"cmdline_cgroup_disable_memory":%s,"psi":%s,"io_controller":%s,"systemd_run":%s,"mem_available_bytes":%s,"root_free_bytes":%s,"services":{%s},"ready":%s,"reasons":[%s]}\n' \
    "$(json_str "${SYSROOT:-real}")" "$(json_str "$(hostname)")" "$(json_str "$(uname -r)")" "$v2" "$mem_ctrl" "$(json_str "${mem_enabled:-}")" "$disabled" "$psi" "$io_ctrl" "$sdrun" \
    "$((avail_kb * 1024))" "$root_free" "${svc_json%,}" "$([[ $ok == 1 ]] && echo true || echo false)" "${reasons_json%,}"
  if [[ $ok != 1 ]]; then
    say ""
    say "REFUSED: a controlled-memory cell needs the cgroup v2 memory controller, and this Pi does not have it:"
    for r in "${reasons[@]}"; do say "  - $r"; done
    say "No constrained run was started. Options (the operator decides; this script never changes boot settings):"
    say "  1) append 'cgroup_enable=memory cgroup_memory=1 psi=1' to /boot/firmware/cmdline.txt and reboot once"
    say "     (ContextFriend services restart; reversible by removing the words again);"
    say "  2) a KVM guest with a fixed RAM size (apt install qemu-system-arm + guest image; no reboot);"
    say "  3) keep the desktop Docker emulation (python -m pigemma.controlled cell ...) as the controlled"
    say "     comparison and the Pi as the resident (no-pressure) reference."
    say "An mlocked balloon is NOT an allowed substitute on the live device."
    return $EXIT_REFUSED
  fi
  return 0
}

services_healthy() {
  local s
  for s in $SERVICES; do
    systemctl is-active --quiet "$s" || { say "service $s is not active"; return 1; }
  done
}

# ------------------------------------------------------------------------------ arguments
MODE=run METHOD="" BUDGET_GIB=8 WORKLOAD="" REPEAT=0 CONVS="" CAP="" RANGES="" IO_BPS="" CELL_ID="" DRY=0
ENGINE_ID="a7b94df" ENGINE_PARENT="" SERVER_EXTRA="" PINNER_EXTRA=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) MODE=check ;;
    --method) METHOD="$2"; shift ;;
    --budget-gib) BUDGET_GIB="$2"; shift ;;
    --workload) WORKLOAD="$2"; shift ;;
    --repeat) REPEAT="$2"; shift ;;
    --conversations) CONVS="$2"; shift ;;
    --n-predict-cap) CAP="$2"; shift ;;
    --ranges) RANGES="$2"; shift ;;
    --io-read-bps) IO_BPS="$2"; shift ;;
    --cell-id) CELL_ID="$2"; shift ;;
    --engine-id) ENGINE_ID="$2"; shift ;;
    --engine-parent) ENGINE_PARENT="$2"; shift ;;
    --server-bin) LLAMA_SERVER="$2"; shift ;;
    --server-extra) SERVER_EXTRA="$2"; shift ;;
    --allow-pinner-on-protected-host) PINNER_EXTRA="--allow-protected-host" ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,41p' "${BASH_SOURCE[0]:-/dev/null}" 2>/dev/null || true; exit 0 ;;
    *) say "unknown argument: $1"; exit 1 ;;
  esac
  shift
done

if [[ $DRY == 1 && -n "${PIGEMMA_CHECK_SYSROOT:-}" ]]; then
  SYSROOT="$PIGEMMA_CHECK_SYSROOT"
  say "TEST MODE (--dry-run only): readiness check reads $SYSROOT/proc and $SYSROOT/sys, not this machine's kernel"
fi

# The readiness check ALWAYS runs first; nothing below it executes while the controller is off.
if [[ $MODE == check ]]; then
  check; exit $?
fi
check >/dev/null || exit $EXIT_REFUSED

# ------------------------------------------------------------------------------ cell (enabled Pi only)
[[ "$METHOD" =~ ^[ABCD]$ ]] || { say "--method A|B|C|D required"; exit 1; }
[[ "$WORKLOAD" =~ ^(calibration|tuning|heldout)$ ]] || { say "--workload calibration|tuning|heldout required"; exit 1; }
if [[ "$METHOD" =~ ^[CD]$ && ! -f "$RANGES" ]]; then say "method $METHOD needs --ranges FILE (page 16384 plan)"; exit 1; fi
BUDGET=$(python3 -c "print(int(round(float('$BUDGET_GIB') * 1024**3)))")
[[ -f "$MODEL" && "$(stat -c %s "$MODEL")" == "$MODEL_SIZE" ]] || { say "model missing or wrong size: $MODEL"; exit 1; }
for b in "$LLAMA_SERVER" "$PINNER"; do [[ -x "$b" ]] || { say "missing binary $b"; exit 1; }; done
[[ -f "$CODE_DIR/pigemma/controlled_cell.py" ]] || { say "missing $CODE_DIR/pigemma/controlled_cell.py"; exit 1; }

avail=$(( $(awk '/^MemAvailable:/{print $2}' /proc/meminfo) * 1024 ))
root_free=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
if (( avail < BUDGET + MIN_AVAIL_EXTRA )); then say "REFUSED: MemAvailable $avail < budget + 3 GB"; exit $EXIT_REFUSED; fi
if (( root_free < MIN_ROOT_FREE )); then say "REFUSED: / has $root_free B free < 2 GB"; exit $EXIT_REFUSED; fi
services_healthy || { say "REFUSED: ContextFriend services not healthy before the cell"; exit $EXIT_REFUSED; }

# one Pi performance experiment at a time: no other inference process, and the shared Pi lock
# (matched on the process NAME, 15 chars max, so this script's own arguments can never match)
others="$(pgrep -a 'llama-server|llama-completio|llama-cli|powerinfer' 2>/dev/null || true)"
if [[ -n "$others" ]]; then
  say "REFUSED: another inference process is running on this Pi:"; say "$others"; exit $EXIT_REFUSED
fi
CELL_ID="${CELL_ID:-$(date +%Y%m%d-%H%M%S)-$WORKLOAD-$METHOD-${BUDGET_GIB}g-r$REPEAT-pi}"
OUT="$PIGEMMA_HOME/results/controlled/$CELL_ID"
mkdir -p "$OUT"
exec 9>"$LOCKFILE"
flock -n 9 || { say "REFUSED: another Pi performance run holds $LOCKFILE"; exit $EXIT_REFUSED; }

python3 - "$OUT/config.json" <<PYCFG
import json, shlex, sys
cfg = {
 "schema": "pigemma-controlled-cell v1", "cell_id": "$CELL_ID",
 "platform": "real Raspberry Pi 5 16 GB, systemd-run --scope (cgroup v2 memory controller)",
 "created": "$(date -Iseconds)",
 "method": "$METHOD", "budget_bytes": $BUDGET, "budget_gib": float("$BUDGET_GIB"), "repeat": int("$REPEAT"),
 "workload": {"set": "$WORKLOAD", "conversations": [x for x in "$CONVS".split(",") if x] or None,
              "n_predict_cap": int("$CAP") if "$CAP" else None},
 "model": {"backing_path": "$MODEL", "size": $MODEL_SIZE, "sha256": "$MODEL_SHA256", "map": "$MAP"},
 "storage": {"mode": "direct", "rate_bps": 0.0, "device": "microSD (91.6 MB/s raw sequential, measured)",
             "io_read_bandwidth_max": "$IO_BPS" or None},
 "pinner": {"bin": "$PINNER", "ranges": "$RANGES" or None, "extra": shlex.split("$PINNER_EXTRA")},
 "purpose": "measurement",
 "engine": {"id": "$ENGINE_ID", "parent": "$ENGINE_PARENT" or None, "revision": None, "api": {},
            "profile_source": "pi/cgroup_cell.sh arguments"},
 "variant": "$METHOD" if "$ENGINE_ID" == "a7b94df" else "$ENGINE_ID:$METHOD",
 "server": {"bin": "$LLAMA_SERVER", "flag_overrides": {}, "env": {},
            "threads": 4, "ctx": 4096, "batch": 2048, "ubatch": 512, "ctk": "f16",
            "ctv": "f16", "flash_attn": "auto", "parallel": 1, "cache_ram_mib": 0, "ctx_checkpoints": 32,
            "seed": 0, "warmup": True, "verbosity": 4, "extra": shlex.split("""$SERVER_EXTRA"""), "nice": 0,
            "speculative": "off (no -md / --spec-type; MTP and every draft method disabled)"},
 "warmup_request": {"text": "Count from one to five, separated by commas.", "n_predict": 16},
 "request": {"temperature": 0.0, "seed": 0},
 "sampling": {"interval_s": 0.5, "residency": "per_request"},
 "out_dir": "$OUT", "host_headroom_bytes": $MIN_AVAIL_EXTRA,
 "abort_mem_available_below_bytes": $MIN_AVAIL_EXTRA,
}
json.dump(cfg, open(sys.argv[1], "w"), indent=1)
PYCFG

PROPS=(-p "MemoryMax=$BUDGET" -p "MemorySwapMax=0" -p "MemoryHigh=infinity" -p "MemoryAccounting=yes"
       -p "CPUAccounting=yes" -p "IOAccounting=yes")
if [[ -n "$IO_BPS" ]]; then
  # the whole-disk device of the model's filesystem (a partition's parent; the device itself when unpartitioned)
  src="$(df --output=source "$MODEL" | tail -1)"
  pk="$(lsblk -no PKNAME "$src" 2>/dev/null | head -1 || true)"
  dev="${pk:+/dev/$pk}"; dev="${dev:-$src}"
  [[ -b "$dev" ]] || { say "cannot resolve the block device of $MODEL for IOReadBandwidthMax (got '$dev')"; exit 1; }
  PROPS+=(-p "IOReadBandwidthMax=$dev $IO_BPS")
fi
UNIT="pigemma-cell-$(printf '%s' "$CELL_ID" | tr -c 'a-zA-Z0-9_.-' '-')"
CMD=(sudo systemd-run --scope --quiet --collect --unit="$UNIT" "${PROPS[@]}" --
     nice -n 10 env PYTHONPATH="$CODE_DIR" PYTHONUNBUFFERED=1
     python3 -m pigemma.controlled_cell run --config "$OUT/config.json")
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"; echo >> "$OUT/command.txt"
if [[ $DRY == 1 ]]; then cat "$OUT/command.txt"; exit 0; fi

# watchdog: stop ONLY this cell's scope if a ContextFriend service becomes unhealthy
( while sleep 10; do
    systemctl is-active --quiet "$UNIT.scope" 2>/dev/null || exit 0
    if ! services_healthy; then
      echo "watchdog: service unhealthy -> stopping $UNIT.scope" >> "$OUT/watchdog.log"
      sudo systemctl stop "$UNIT.scope"; exit 0
    fi
  done ) &
WD=$!
rc=0
"${CMD[@]}" > "$OUT/runner.log" 2>&1 || rc=$?
kill "$WD" 2>/dev/null || true
sudo chown -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null || true
services_healthy && echo "services healthy after the cell" >> "$OUT/runner.log" || echo "WARNING: services NOT healthy after the cell" >> "$OUT/runner.log"
echo "{\"cell_id\":\"$CELL_ID\",\"exit\":$rc,\"out\":\"$OUT\"}"
exit $rc
