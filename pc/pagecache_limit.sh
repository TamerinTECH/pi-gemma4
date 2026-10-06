#!/usr/bin/env bash
# =============================================================================
# pc/pagecache_limit.sh - E5/E6: the desktop proxy for a RAM-limited Pi 5.
#
# Runs pigemma-probe in replay mode (-ub 1 -nr, decode-like) under a cgroup
# memory cap (docker --memory = --memory-swap) against the 26B GGUF copied
# into the ext4 Docker volume `pigemma-models`, measures the real bytes read
# from storage per token (/proc/self/io read_bytes, major faults), optionally
# A/B tests hot-set pinning (tools/pinner, mlock in the SAME cgroup), and then
# compares every point with the expert-cache simulator
# (python -m pigemma.cache_sim, LRU + Belady at the matching budget) and the
# time model (cache_sim io-point). Spec: docs/design/manifest.json
# files[pc/pagecache_limit.sh]; experiments E5-pagecache-validation and
# E6-hotset-pinning-proxy (docs/EXPERIMENT_PLAN.md).
#
# One file, two sides:
#   host (Git Bash on Windows, or Linux):   bash pc/pagecache_limit.sh [options]
#   inner (inside the capped container):    bash /work/pc/pagecache_limit.sh --inner [--task run|preflight|copy] ...
# The host starts the inner side through pc/docker_run.sh.
#
# Typical use:
#   bash pc/pagecache_limit.sh --mem 3,4,5,6,8 --tokens traces/26b/all.moetrace --range 0:1500 --out results/h1/validation.json
#   bash pc/pagecache_limit.sh --mem 5,6 --pin --skip-copy --tokens traces/26b/all.moetrace --range 0:1500 --out results/h1/validation_pin.json
#   bash pc/pagecache_limit.sh --dry-run          (prints every gate and command, runs nothing heavy)
#
# Steps (every gate decision is logged as "GATE: ..." on stderr and in results/h1/raw/pagecache_limit-<UTC>.log):
#   0  inputs + gates: trace + <trace>.tokens.bin, map.json (built from the GGUF header when the default
#      results/map26.json is missing), stats.json (--pin), probe/pinner/llama-completion binaries,
#      `pc/docker_run.sh vm-df` -> VM root free >= 16 GB AND free space on the host drive holding the
#      Docker VM disk >= 18 GB (C: when the vhdx location is unknown), only when a copy is needed.
#      Otherwise: "docker builder prune would reclaim ~N GB" (a human decision, never run here) and
#      REFUSED unless --allow-bind-mount (bind-mount proxy: bytes/token = majflt x page_size, a lower
#      bound; notes 'bind-mount proxy: lower-bound bytes, timings not representative').
#   1  once: docker volume create pigemma-models; dd copy of the GGUF (progress on stderr, O_DIRECT
#      writes, written as <name>.part and renamed after the size + SHA-256(first 16 MiB) check).
#   2  per cap M (GB = 1e9 B): docker_run --memory <M*1e9>b --cpus 6 --volume pigemma-models:/models:ro
#      -- bash /work/pc/pagecache_limit.sh --inner ...; the inner side records the cgroup limit,
#      the device read_ahead_kb, runs fio randread bs=3345k O_DIRECT QD1 15 s (tier bandwidth),
#      evicts the GGUF from the page cache (vmtouch -e), optionally starts the pinner and waits for its
#      summary, runs the probe, samples memory.stat every 0.5 s, records memory.events/oom_kill and
#      io.stat, runs a plain llama-completion -n 64 for a natural-generation tok/s, stops the pinner
#      (SIGTERM to its pidfile only) and writes results/h1/io_<M>[_pin].csv + pagecache_<M>[_pin].json.
#   3  host: python -m pigemma.cache_sim <trace> --policies lru,belady --budgets-gb <M - anon_rss -
#      non-expert - 0.1> --token-range A:B --warmup-tokens 200 -> results/h1/sim_e5_<M>g.json
#      (sim_e6_<M>g.json for pinned runs), then `python -m pigemma.cache_sim io-point ... --validation
#      <out>` appends the point (measured + projected sub-block), and E5/E6 results envelopes are
#      written next to it. ram_tok_s for the time model comes from an uncapped reference run
#      (--ref-mem 20 GB, reused when results/h1/pagecache_ref20.json matches) or --ram-tok-s.
#
# Decisions beyond the manifest text (documented here; also reported as open issues):
#  D1 GB means 1e9 bytes everywhere (project convention, catalog and EXPERIMENT_PLAN). docker's "5g"
#     would be 5 GiB, so the cap is passed in bytes: --memory 5000000000b. The cgroup limit seen
#     inside the container is recorded; a mismatch becomes a note.
#  D2 The VM-disk gate uses the drive that really holds docker_data.vhdx (pc/docker_run.sh vm-df:
#     host_vm_disk.drive_avail_bytes; F: on this desktop, not C:). C: is the fallback when the
#     location is unknown. The gate applies only when the copy is needed (a verified copy needs no
#     new space).
#  D3 Order inside the container: fio (O_DIRECT, does not populate the page cache) BEFORE the
#     eviction, so a buffered fio fallback cannot leave cached pages behind. Equivalent to the
#     manifest order (vmtouch -e, fio) for the normal O_DIRECT case.
#  D4 Replay chunking: --replay-chunks <out>/raw/replay_chunks.json (from the trace header) and
#     --chunk-tokens = B-A, so the replay restarts the KV cache exactly at the source chunk starts and
#     the access sequence matches the trace the simulator replays. -c = max(2048, longest chunk piece
#     + 8 rounded up to 256). The replayed routing is saved as traces/h1/replay_<tag>.moetrace and
#     compared with the source trace (set agreement per token and layer); --sim-on-replay simulates
#     the replayed sequence itself.
#  D5 The inner side always exits 0 once it has written its run record (the record holds the probe
#     exit code, OOM and timeouts); the host decides. An OOM stops the remaining caps (EXPERIMENT_PLAN:
#     "Any OOM means repack leaked -> fix before continuing") unless --keep-going.
#  D6 With --pin and no --out, the output is results/h1/validation_pin.json (EXPERIMENT_PLAN E6).
#     Pinned runs are compared with the unpinned points of the same cap found in the output file,
#     --compare files and results/h1/validation.json.
#  D7 The validation file also gets "reference", "driver" and "checks_preview" blocks (E5/E6
#     pre-registered thresholds evaluated mechanically; the human verdict stays in the write-up), and
#     each point gets run_record / io_csv / natural_gen_tok_s / routing_agreement extras.
#  D8 Envelopes (python -m pigemma.results): per point E5 (unpinned) or E6 (pinned)
#     '<exp>_pagecache_cap<M>g[_pin]_{replay,gen,predicted}.json': replay -ub 1 (kind measured,
#     tg = steady tok/s after the warm-up), natural generation (measured) and the time-model
#     prediction (kind projected with its assumptions). The reference run writes E5 ..._ref<G>g_*.
#  D9 Test hooks: PIGEMMA_PAGECACHE_PROBE / _PINNER / _COMPLETION (container paths) replace the three
#     binaries; PIGEMMA_PAGECACHE_VOLUME replaces the volume name. They are forwarded by docker_run.
#  D10 The host takes the bench lock (python -m pigemma.paths lock --hold --min-free-gb 20) for the
#     whole run unless one is inherited (PIGEMMA_LOCK_HELD) or --no-lock/--dry-run is given.
#
# Exit codes (pigemma.paths): 0 every requested run completed (thresholds PASS/FAIL are results, not
# errors); 1 a run failed (probe error, OOM, inner failure) or another error; 2 usage; 3 missing
# input (trace, tokens, map, stats, model, binaries, volume copy with --skip-copy); 4 refused (disk
# gates, bench lock busy / low RAM, docker unavailable). Errors start with "[pagecache_limit".
#
# Requires bash >= 4.4. Inner side: Linux with /proc, python3 (stdlib), optional vmtouch and fio.
# Lint: bash -n pc/pagecache_limit.sh (shellcheck is not installed on this desktop).
# =============================================================================
set -euo pipefail
shopt -u patsub_replacement 2>/dev/null || true

readonly PROG=pagecache_limit
readonly VERSION=1.0.0
readonly EXIT_OK=0 EXIT_ERROR=1 EXIT_USAGE=2 EXIT_MISSING=3 EXIT_REFUSED=4
readonly CONTAINER_ROOT=/work
readonly SHA_PREFIX_BYTES=16777216

# Keep Git Bash (MSYS) from rewriting /work, /models, ... in arguments of native programs.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

UNAME_S=$(uname -s 2>/dev/null || echo unknown)
IS_MSYS=0
case $UNAME_S in MINGW*|MSYS*|CYGWIN*) IS_MSYS=1 ;; esac

# ---------------------------------------------------------------------------
# logging and small helpers (shared by both sides)
# ---------------------------------------------------------------------------
LOG_TAG=$PROG
LOG_FILE=""
TS_UTC=0
ts() { if [ "$TS_UTC" = 1 ]; then date -u +%H:%M:%SZ; else date +%H:%M:%S; fi; }
log() {
    local m
    m="[$LOG_TAG $(ts)] $*"
    printf '%s\n' "$m" >&2
    if [ -n "$LOG_FILE" ]; then printf '%s\n' "$m" >>"$LOG_FILE" 2>/dev/null || true; fi
}
warn() { log "WARNING: $*"; }
gate() { log "GATE: $*"; }
die() {
    local rc=$1
    shift
    log "ERROR (exit $rc): $*"
    exit "$rc"
}
refuse() {
    log "REFUSED (exit $EXIT_REFUSED): $*"
    printf '%s: REFUSED: %s\n' "$PROG" "$*" >&2
    exit "$EXIT_REFUSED"
}
is_uint() { [[ ${1:-} =~ ^[0-9]+$ ]]; }
is_num() { [[ ${1:-} =~ ^[0-9]+([.][0-9]+)?$ ]]; }
is_snum() { [[ ${1:-} =~ ^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; }
# gb_to_bytes 4.5 -> 4500000000 (GB = 1e9 B)
gb_to_bytes() { awk -v m="$1" 'BEGIN { printf "%.0f", m * 1000000000 }'; }
# canon_num 5.0 -> 5, 4.50 -> 4.5
canon_num() { awk -v m="$1" 'BEGIN { printf "%g", m }'; }
json_str() {
    local s=$1 bs='\' dbs='\\' q='"' eq='\"' nl=$'\n' tab=$'\t' cr=$'\r'
    s=${s//"$bs"/"$dbs"}
    s=${s//"$q"/"$eq"}
    s=${s//"$nl"/'\n'}
    s=${s//"$tab"/'\t'}
    s=${s//"$cr"/}
    printf '"%s"' "$s"
}
json_num() { if is_snum "${1:-}"; then printf '%s' "$1"; else printf 'null'; fi; }
# split "--opt=value" into "--opt" "value" (both parsers accept either form)
SPLIT_ARGS=()
split_eq_args() {
    local a
    SPLIT_ARGS=()
    for a in "$@"; do
        case $a in
            --*=*) SPLIT_ARGS+=("${a%%=*}" "${a#*=}") ;;
            *) SPLIT_ARGS+=("$a") ;;
        esac
    done
}
need_val() { if [ "$2" -lt 2 ]; then die "$EXIT_USAGE" "option $1 needs a value (see --help)"; fi; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || { echo "$PROG: cannot resolve the script directory" >&2; exit 3; }

# =============================================================================
# INNER SIDE (runs inside the capped container)
# =============================================================================
INNER_PY=""
IFS= read -r -d '' INNER_PY <<'PY' || true
import json, os, sys

def last_json(path, prefix=""):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    val = None
    for ln in lines:
        s = ln.strip()
        if prefix:
            if not s.startswith(prefix):
                continue
            s = s[len(prefix):]
        elif not s.startswith("{"):
            continue
        try:
            val = json.loads(s)
        except ValueError:
            pass
    return val

def op_fio(path, direct):
    out = {"ok": False, "direct": direct == "1"}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        i = text.find("{")
        doc, _ = json.JSONDecoder().raw_decode(text[i:]) if i >= 0 else (None, 0)
        job = doc["jobs"][0]
        rd = job["read"]
        n = int(rd.get("total_ios") or 0)
        pct = (rd.get("clat_ns") or {}).get("percentile") or {}
        def ms(v):
            return None if v is None else float(v) / 1e6
        out.update({
            "ok": int(job.get("error") or 0) == 0 and float(rd.get("bw_bytes") or 0) > 0,
            "error": int(job.get("error") or 0),
            "bs": (job.get("job options") or {}).get("bs"),
            "bs_bytes": int(rd["io_bytes"]) // n if n else None,
            "bw_bytes": int(rd.get("bw_bytes") or 0),
            "mbps": float(rd.get("bw_bytes") or 0) / 1e6,
            "iops": float(rd.get("iops") or 0),
            "io_bytes": int(rd.get("io_bytes") or 0),
            "total_ios": n,
            "runtime_ms": int(rd.get("runtime") or 0),
            "lat_mean_ms": ms((rd.get("lat_ns") or {}).get("mean")),
            "clat_p50_ms": ms(pct.get("50.000000")),
            "clat_p99_ms": ms(pct.get("99.000000")),
            "fio_version": doc.get("fio version"),
        })
    except Exception as e:  # noqa: BLE001 - any parse problem is reported, never fatal
        out["parse_error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out))

def op_memstats(path):
    agg = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                p = ln.split()
                if len(p) != 5:
                    continue
                ph = p[0]
                try:
                    a, fi, m, u = (int(x) for x in p[1:])
                except ValueError:
                    continue
                d = agg.setdefault(ph, {"n": 0, "max_anon": 0, "max_file": 0, "max_file_mapped": 0, "max_usage": 0})
                d["n"] += 1
                d["max_anon"] = max(d["max_anon"], a)
                d["max_file"] = max(d["max_file"], fi)
                d["max_file_mapped"] = max(d["max_file_mapped"], m)
                d["max_usage"] = max(d["max_usage"], u)
                d["last"] = {"anon": a, "file": fi, "file_mapped": m, "usage": u}
    except OSError:
        pass
    print(json.dumps(agg))

def op_ioinfo(path):
    rows, last = 0, None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            next(f, None)
            for ln in f:
                if ln.strip():
                    rows += 1
                    last = ln.split(",", 1)[0]
    except OSError:
        pass
    print(json.dumps({"rows": rows, "last_token_index": int(last) if last and last.isdigit() else None}))

def op_compact(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        i = text.find("{")
        val, _ = json.JSONDecoder().raw_decode(text[i:]) if i >= 0 else (None, 0)
    except Exception:  # noqa: BLE001
        val = None
    print(json.dumps(val))

def op_argv(*argv):
    print(json.dumps(list(argv)))

def op_assemble(kv, dst):
    out = {}
    with open(kv, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if "\t" not in ln:
                continue
            k, v = ln.split("\t", 1)
            try:
                val = json.loads(v)
            except ValueError:
                val = v
            append = k.endswith("[]")
            if append:
                k = k[:-2]
            parts = k.split(".")
            d = out
            for p in parts[:-1]:
                if not isinstance(d.get(p), dict):
                    d[p] = {}
                d = d[p]
            if append:
                lst = d.get(parts[-1])
                if not isinstance(lst, list):
                    lst = []
                    d[parts[-1]] = lst
                lst.append(val)
            else:
                d[parts[-1]] = val
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, dst)

def op_get(json_text, key):
    try:
        v = json.loads(json_text)
        for k in key.split("."):
            v = v[k] if isinstance(v, dict) else None
    except Exception:  # noqa: BLE001
        v = None
    if v is None:
        print("")
    elif isinstance(v, bool):
        print("1" if v else "0")
    elif isinstance(v, (dict, list)):
        print(json.dumps(v))
    else:
        print(v)

op = sys.argv[1]
args = sys.argv[2:]
if op == "summary":
    print(json.dumps(last_json(args[0], "PIGEMMA_SUMMARY ")))
elif op == "jsonline":
    print(json.dumps(last_json(args[0])))
elif op == "fio":
    op_fio(args[0], args[1])
elif op == "memstats":
    op_memstats(args[0])
elif op == "ioinfo":
    op_ioinfo(args[0])
elif op == "compact":
    op_compact(args[0])
elif op == "argv":
    op_argv(*args)
elif op == "assemble":
    op_assemble(args[0], args[1])
elif op == "get":
    op_get(args[0], args[1])
else:
    sys.exit("unknown op " + op)
PY

inner_py() { python3 -c "$INNER_PY" "$@"; }
jget() { inner_py get "$1" "$2"; }

# record helpers: one "key<TAB>json" line per value; keys are dotted paths, "key[]" appends to a list
REC=""
rec() { printf '%s\t%s\n' "$1" "$2" >>"$REC"; }
rec_str() { rec "$1" "$(json_str "$2")"; }
rec_num() { rec "$1" "$(json_num "${2:-}")"; }
rec_bool() { if [ "${2:-0}" = 1 ] || [ "${2:-}" = true ]; then rec "$1" true; else rec "$1" false; fi; }
rec_note() { rec_str "notes[]" "$1"; log "note: $1"; }
# repo-relative form of a container path (/work/x -> x)
crel() { case $1 in "$CONTAINER_ROOT"/*) printf '%s' "${1#"$CONTAINER_ROOT"/}" ;; *) printf '%s' "$1" ;; esac; }

# ---- cgroup (v2 or v1) ------------------------------------------------------
CG_VER=none CG_DIR=""
cg_detect() {
    CG_VER=none CG_DIR=""
    if [ -f /sys/fs/cgroup/cgroup.controllers ] && [ -f /sys/fs/cgroup/memory.stat ]; then
        CG_VER=v2 CG_DIR=/sys/fs/cgroup
    elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        CG_VER=v1 CG_DIR=/sys/fs/cgroup/memory
    fi
}
cg_limit() {
    local v=""
    case $CG_VER in
        v2) v=$(cat "$CG_DIR/memory.max" 2>/dev/null) || v="" ;;
        v1) v=$(cat "$CG_DIR/memory.limit_in_bytes" 2>/dev/null) || v="" ;;
    esac
    # "max" (v2) or the v1 "unlimited" sentinel (~2^63, page aligned) -> empty
    if ! is_uint "$v" || [ "${#v}" -ge 19 ]; then v=""; fi
    printf '%s' "$v"
}
# cg_memstat -> "anon file file_mapped" (bytes)
cg_memstat() {
    case $CG_VER in
        v2) awk '$1=="anon"{a=$2} $1=="file"{f=$2} $1=="file_mapped"{m=$2} END{printf "%d %d %d\n", a, f, m}' "$CG_DIR/memory.stat" 2>/dev/null || echo "0 0 0" ;;
        v1) awk '$1=="total_rss"{a=$2} $1=="total_cache"{f=$2} $1=="total_mapped_file"{m=$2} END{printf "%d %d %d\n", a, f, m}' "$CG_DIR/memory.stat" 2>/dev/null || echo "0 0 0" ;;
        *) echo "0 0 0" ;;
    esac
}
cg_usage() {
    case $CG_VER in
        v2) cat "$CG_DIR/memory.current" 2>/dev/null || true ;;
        v1) cat "$CG_DIR/memory.usage_in_bytes" 2>/dev/null || true ;;
    esac
}
cg_peak() {
    case $CG_VER in
        v2) cat "$CG_DIR/memory.peak" 2>/dev/null || true ;;
        v1) cat "$CG_DIR/memory.max_usage_in_bytes" 2>/dev/null || true ;;
    esac
}
cg_oom_kill() {
    local v=""
    case $CG_VER in
        v2) v=$(awk '$1=="oom_kill"{print $2}' "$CG_DIR/memory.events" 2>/dev/null) || v="" ;;
        v1) v=$(awk '$1=="oom_kill"{print $2}' "$CG_DIR/memory.oom_control" 2>/dev/null) || v="" ;;
    esac
    printf '%s' "$v"
}
cg_io_rbytes() {
    local f
    if [ "$CG_VER" = v2 ] && [ -r /sys/fs/cgroup/io.stat ]; then
        awk '{for(i=1;i<=NF;i++) if($i ~ /^rbytes=/){split($i,a,"="); s+=a[2]; n++}} END{if(n) printf "%d", s}' /sys/fs/cgroup/io.stat 2>/dev/null || true
        return 0
    fi
    for f in /sys/fs/cgroup/blkio/blkio.throttle.io_service_bytes /sys/fs/cgroup/blkio/blkio.io_service_bytes; do
        if [ -r "$f" ] && [ -s "$f" ]; then
            awk '$2=="Read"{s+=$3; n++} END{if(n) printf "%d", s}' "$f" 2>/dev/null || true
            return 0
        fi
    done
}
rec_cg_snapshot() {  # rec_cg_snapshot KEY
    local s anon file mapped
    s=$(cg_memstat)
    read -r anon file mapped <<<"$s"
    rec_num "$1.anon" "$anon"
    rec_num "$1.file" "$file"
    rec_num "$1.file_mapped" "$mapped"
    rec_num "$1.usage" "$(cg_usage)"
    rec_num "$1.peak" "$(cg_peak)"
    rec_num "$1.oom_kill" "$(cg_oom_kill)"
    rec_num "$1.io_rbytes" "$(cg_io_rbytes)"
}

# ---- device of a file: read_ahead_kb, logical block size, fs type, source ----
DEV_MM="" DEV_RA_KB="" DEV_LBS="" DEV_FSTYPE="" DEV_SOURCE=""
dev_info() {
    local f=$1 d mnt
    DEV_MM="" DEV_RA_KB="" DEV_LBS="" DEV_FSTYPE="" DEV_SOURCE=""
    DEV_MM=$(stat -c '%Hd:%Ld' -- "$f" 2>/dev/null) || DEV_MM=""
    if [ -n "$DEV_MM" ] && [ "${DEV_MM%%:*}" != 0 ]; then
        for d in "/sys/dev/block/$DEV_MM/queue" "/sys/dev/block/$DEV_MM/../queue"; do
            if [ -r "$d/read_ahead_kb" ]; then
                DEV_RA_KB=$(cat "$d/read_ahead_kb" 2>/dev/null) || DEV_RA_KB=""
                DEV_LBS=$(cat "$d/logical_block_size" 2>/dev/null) || DEV_LBS=""
                break
            fi
        done
    fi
    mnt=$(df -P -- "$f" 2>/dev/null | awk 'NR==2 {print $6}') || mnt=""
    if [ -n "$mnt" ] && [ -r /proc/self/mountinfo ]; then
        read -r DEV_FSTYPE DEV_SOURCE < <(awk -v mp="$mnt" '$5==mp {for(i=7;i<=NF;i++) if($i=="-"){print $(i+1), $(i+2); exit}}' /proc/self/mountinfo 2>/dev/null) || true
    fi
}

# ---- children: background + wait, so TERM/INT stay responsive (bash defers traps for foreground children)
INNER_STOP=0
CHILD_PID="" CHILD_RC=0 CHILD_TIMED_OUT=0 CHILD_WALL_S=""
run_child() {  # run_child OUT ERR TIMEOUT_S cmd...
    local out=$1 err=$2 to=$3 t0 t1
    shift 3
    t0=$(date +%s.%N)
    if [ "$to" != 0 ] && command -v timeout >/dev/null 2>&1; then
        timeout --signal=TERM --kill-after=120 "$to" "$@" >"$out" 2>"$err" &
    else
        "$@" >"$out" 2>"$err" &
    fi
    CHILD_PID=$!
    CHILD_RC=0
    while :; do
        if wait "$CHILD_PID"; then CHILD_RC=0; else CHILD_RC=$?; fi
        if kill -0 "$CHILD_PID" 2>/dev/null; then continue; fi  # wait interrupted by a trapped signal
        break
    done
    CHILD_PID=""
    t1=$(date +%s.%N)
    CHILD_WALL_S=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')
    CHILD_TIMED_OUT=0
    if [ "$to" != 0 ] && [ "$CHILD_RC" = 124 ]; then CHILD_TIMED_OUT=1; fi
}

# ---- memory sampler (background; our own child) -----------------------------
SAMPLER_PID=""
sampler_start() {
    if [ "$CG_VER" = none ]; then
        warn "no memory cgroup visible: no memory.stat sampling"
        return 0
    fi
    : >"$WORK/mem.samples"
    (
        trap 'exit 0' TERM INT HUP
        while :; do
            ph=$(cat "$WORK/phase" 2>/dev/null) || ph=idle
            ms=$(cg_memstat)
            us=$(cg_usage)
            printf '%s %s %s\n' "${ph:-idle}" "$ms" "${us:-0}" >>"$WORK/mem.samples"
            sleep "$I_SAMPLE_S"
        done
    ) &
    SAMPLER_PID=$!
}
sampler_stop() {
    if [ -n "$SAMPLER_PID" ]; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
        SAMPLER_PID=""
    fi
}
set_phase() { printf '%s\n' "$1" >"$WORK/phase"; }
# sample_now PHASE - one synchronous memory.stat sample (short phases may fall between two sampler ticks)
sample_now() {
    if [ "$CG_VER" = none ] || [ -z "$WORK" ]; then return 0; fi
    printf '%s %s %s\n' "$1" "$(cg_memstat)" "$(cg_usage || echo 0)" >>"$WORK/mem.samples" 2>/dev/null || true
}

# ---- pinner -------------------------------------------------------------------
PINNER_PID="" PINNER_PIDFILE="" PIN_START_RC="" PIN_STOP_RC="" PIN_HOLDING=0
pinner_start() {
    local t0 st
    PINNER_PIDFILE=$WORK/pinner.pid
    log "pinner: $PINNER_BIN --gguf $I_GGUF --ranges $I_PIN_RANGES --verify --json --pidfile $PINNER_PIDFILE (same cgroup as the probe: locked pages count against the cap)"
    "$PINNER_BIN" --gguf "$I_GGUF" --ranges "$I_PIN_RANGES" --verify --json --pidfile "$PINNER_PIDFILE" \
        >"$WORK/pinner.out" 2>"$WORK/pinner.err" &
    PINNER_PID=$!
    t0=$(date +%s)
    while :; do
        if grep -q '^{' "$WORK/pinner.out" 2>/dev/null; then break; fi
        if ! kill -0 "$PINNER_PID" 2>/dev/null; then break; fi
        if [ $(( $(date +%s) - t0 )) -ge "$I_PIN_TIMEOUT" ]; then
            warn "pinner: no summary line after ${I_PIN_TIMEOUT} s"
            break
        fi
        if [ "$INNER_STOP" = 1 ]; then break; fi
        sleep 0.5
    done
    PIN_START_JSON=$(inner_py jsonline "$WORK/pinner.out")
    st=$(jget "$PIN_START_JSON" status)
    case $st in
        ok|partial) PIN_HOLDING=1 ;;
        *) PIN_HOLDING=0 ;;
    esac
    if [ "$PIN_HOLDING" = 0 ] || ! kill -0 "$PINNER_PID" 2>/dev/null; then
        # refused / interrupted / no summary: it exits (or is stopped) - collect its code
        if kill -0 "$PINNER_PID" 2>/dev/null; then pinner_stop; fi
        if [ -n "$PINNER_PID" ]; then
            PIN_START_RC=0
            wait "$PINNER_PID" || PIN_START_RC=$?
            PINNER_PID=""
        fi
        PIN_HOLDING=0
    else
        PIN_START_RC=running
    fi
    rec "pinner.start" "$PIN_START_JSON"
    rec_str "pinner.start_rc" "$PIN_START_RC"
    rec_num "pinner.start_s" "$(( $(date +%s) - t0 ))"
    log "pinner: status=${st:-none} locked_bytes=$(jget "$PIN_START_JSON" locked_bytes) requested_bytes=$(jget "$PIN_START_JSON" requested_bytes) resident_bytes=$(jget "$PIN_START_JSON" resident_bytes) rc=$PIN_START_RC"
}
pinner_stop() {
    local pid="" rc=0 t0
    if [ -z "$PINNER_PID" ]; then return 0; fi
    if [ -s "$PINNER_PIDFILE" ]; then pid=$(tr -dc '0-9' <"$PINNER_PIDFILE" 2>/dev/null) || pid=""; fi
    if [ -z "$pid" ]; then
        warn "pinner pidfile $PINNER_PIDFILE missing or empty; signalling the pinner this script started (pid $PINNER_PID)"
        pid=$PINNER_PID
    elif [ "$pid" != "$PINNER_PID" ]; then
        warn "pidfile names pid $pid but this script started $PINNER_PID; signalling the pidfile pid only"
    fi
    log "pinner: SIGTERM to pid $pid (pidfile $PINNER_PIDFILE)"
    kill -TERM "$pid" 2>/dev/null || warn "kill -TERM $pid failed (already gone?)"
    t0=$(date +%s)
    while kill -0 "$PINNER_PID" 2>/dev/null; do
        if [ $(( $(date +%s) - t0 )) -ge 120 ]; then
            warn "pinner pid $PINNER_PID still alive 120 s after SIGTERM; leaving it to the container teardown"
            PIN_STOP_RC=timeout
            PINNER_PID=""
            return 0
        fi
        sleep 0.2
    done
    wait "$PINNER_PID" || rc=$?
    PIN_STOP_RC=$rc
    PINNER_PID=""
    log "pinner: exited rc=$rc"
}

# ---- run phases -------------------------------------------------------------
TIER_MBPS=""
inner_fio() {
    local rc=0 direct=1 fj
    if [ "$I_FIO_RUNTIME" = 0 ]; then
        log "fio: skipped (--fio-runtime 0)"
        rec "fio" null
        return 0
    fi
    if ! command -v fio >/dev/null 2>&1; then
        rec_note "fio not installed in the image: tier bandwidth not measured"
        rec "fio" null
        return 0
    fi
    log "fio: randread bs=$I_FIO_BS direct=1 QD1 ${I_FIO_RUNTIME}s on $I_GGUF (tier characterisation)"
    fio --name=tier --filename="$I_GGUF" --rw=randread --bs="$I_FIO_BS" --direct=1 --readonly --iodepth=1 \
        --runtime="$I_FIO_RUNTIME" --time_based --output-format=json >"$WORK/fio.json" 2>"$WORK/fio.err" || rc=$?
    if [ "$rc" -ne 0 ]; then
        warn "fio with --direct=1 failed (rc=$rc: $(tail -n 2 "$WORK/fio.err" 2>/dev/null | tr '\n' ' ')); retrying buffered"
        direct=0 rc=0
        fio --name=tier --filename="$I_GGUF" --rw=randread --bs="$I_FIO_BS" --direct=0 --readonly --iodepth=1 \
            --runtime="$I_FIO_RUNTIME" --time_based --output-format=json >"$WORK/fio.json" 2>>"$WORK/fio.err" || rc=$?
        rec_note "fio ran buffered (O_DIRECT failed): the tier bandwidth may be inflated by the page cache"
    fi
    fj=$(inner_py fio "$WORK/fio.json" "$direct")
    rec "fio" "$fj"
    rec_num "fio.rc" "$rc"
    rec_str "fio.cmd" "fio --name=tier --filename=$I_GGUF --rw=randread --bs=$I_FIO_BS --direct=$direct --readonly --iodepth=1 --runtime=$I_FIO_RUNTIME --time_based --output-format=json"
    if [ "$(jget "$fj" ok)" = 1 ]; then
        TIER_MBPS=$(jget "$fj" mbps)
        log "fio: tier_rand_read_3mb_mbps=$(awk -v v="$TIER_MBPS" 'BEGIN{printf "%.1f", v}') MB/s (bs $(jget "$fj" bs_bytes) B, mean latency $(jget "$fj" lat_mean_ms) ms)"
    else
        rec_note "fio failed (rc=$rc): tier bandwidth unknown, predicted_tok_s will be null"
    fi
}

inner_evict() {
    local before="" after="" tool="" rc=0 a_res="" a_tot=""
    if command -v vmtouch >/dev/null 2>&1; then
        before=$(vmtouch -- "$I_GGUF" 2>/dev/null | awk '/Resident Pages/ {print $3}') || before=""
        vmtouch -e -- "$I_GGUF" >"$WORK/vmtouch.out" 2>&1 || rc=$?
        tool=vmtouch
        after=$(vmtouch -- "$I_GGUF" 2>/dev/null | awk '/Resident Pages/ {print $3}') || after=""
    else
        python3 -c 'import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)' "$I_GGUF" || rc=$?
        tool=posix_fadvise
        rec_note "vmtouch not installed: evicted with posix_fadvise(DONTNEED), residency not measured"
    fi
    rec_str "evict.tool" "$tool"
    rec_num "evict.rc" "$rc"
    rec_str "evict.resident_before" "$before"
    rec_str "evict.resident_after" "$after"
    if [ -n "$after" ]; then
        a_res=${after%%/*}
        a_tot=${after#*/}
        rec_num "evict.resident_pages_after" "$a_res"
        rec_num "evict.total_pages" "$a_tot"
        if is_uint "$a_res" && is_uint "$a_tot" && [ "$a_tot" -gt 0 ] && [ $(( a_res * 100 )) -gt "$a_tot" ]; then
            rec_note "eviction incomplete: $after pages still resident after vmtouch -e (mapped or locked elsewhere?) - the run does not start cold"
        fi
    fi
    log "evict ($tool): resident pages before=${before:-?} after=${after:-?}"
}

inner_preflight() {
    local t
    cg_detect
    kv() { printf '%s=%s\n' "$1" "$2"; }
    kv page_size "$(getconf PAGESIZE 2>/dev/null || true)"
    kv nproc "$(nproc 2>/dev/null || true)"
    kv kernel "$(uname -r 2>/dev/null || true)"
    kv machine "$(uname -m 2>/dev/null || true)"
    kv mem_total_kb "$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
    kv mem_available_kb "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
    kv cgroup_version "$CG_VER"
    kv cgroup_mem_limit "$(cg_limit)"
    for t in vmtouch fio python3 dd sha256sum timeout; do
        if command -v "$t" >/dev/null 2>&1; then kv "have_$t" 1; else kv "have_$t" 0; fi
    done
    local probe=${PIGEMMA_PAGECACHE_PROBE:-$CONTAINER_ROOT/tools/build-linux-x86/bin/pigemma-probe}
    local pinner=${PIGEMMA_PAGECACHE_PINNER:-$CONTAINER_ROOT/tools/build-linux-x86/bin/pinner}
    local comp=${PIGEMMA_PAGECACHE_COMPLETION:-$CONTAINER_ROOT/third_party/llama.cpp/build-linux-x86/bin/llama-completion}
    kv probe_bin "$probe"
    if [ -x "$probe" ]; then kv probe_ok 1; else kv probe_ok 0; fi
    kv pinner_bin "$pinner"
    if [ -x "$pinner" ]; then kv pinner_ok 1; else kv pinner_ok 0; fi
    kv completion_bin "$comp"
    if [ -x "$comp" ]; then kv completion_ok 1; else kv completion_ok 0; fi
    if [ -n "$I_GGUF" ] && [ -f "$I_GGUF" ]; then
        kv gguf_present 1
        kv gguf_size "$(stat -c %s -- "$I_GGUF")"
        if [ "$I_NO_SHA" = 0 ]; then
            kv gguf_sha256_16m "$(head -c "$SHA_PREFIX_BYTES" -- "$I_GGUF" | sha256sum | awk '{print $1}')"
        fi
        dev_info "$I_GGUF"
        kv read_ahead_kb "$DEV_RA_KB"
        kv fstype "$DEV_FSTYPE"
        kv avail_bytes "$(df -P -B1 -- "$(dirname -- "$I_GGUF")" 2>/dev/null | awk 'NR==2 {print $4}')"
    else
        kv gguf_present 0
    fi
}

inner_copy() {
    local size avail need t0 t1 rc=0 got sha dstdir
    [ -n "$I_SRC" ] && [ -n "$I_DST" ] || die "$EXIT_USAGE" "copy needs --src and --dst"
    [ -f "$I_SRC" ] || die "$EXIT_MISSING" "copy: source $I_SRC not found in the container"
    size=$(stat -c %s -- "$I_SRC")
    if [ -n "$I_EXPECT_BYTES" ] && [ "$size" != "$I_EXPECT_BYTES" ]; then
        die "$EXIT_MISSING" "copy: source $I_SRC has $size B, expected $I_EXPECT_BYTES (incomplete download?)"
    fi
    dstdir=$(dirname -- "$I_DST")
    mkdir -p -- "$dstdir"
    avail=$(df -P -B1 -- "$dstdir" | awk 'NR==2 {print $4}')
    need=$(( size + 1000000000 ))
    if is_uint "$avail" && [ "$avail" -lt "$need" ]; then
        refuse "volume filesystem has $avail B free, the copy needs $need B (file + 1 GB)"
    fi
    log "copy: $I_SRC -> $I_DST ($size B): dd bs=16M, O_DIRECT writes, progress below"
    t0=$(date +%s)
    dd if="$I_SRC" of="$I_DST.part" bs=16M iflag=fullblock oflag=direct conv=fsync status=progress || rc=$?
    if [ "$rc" -ne 0 ]; then
        warn "dd with oflag=direct failed (rc=$rc); retrying with buffered writes"
        rc=0
        dd if="$I_SRC" of="$I_DST.part" bs=16M iflag=fullblock conv=fsync status=progress || rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        rm -f -- "$I_DST.part"
        die "$EXIT_ERROR" "copy failed (dd rc=$rc)"
    fi
    got=$(stat -c %s -- "$I_DST.part")
    if [ "$got" != "$size" ]; then
        rm -f -- "$I_DST.part"
        die "$EXIT_ERROR" "copy size mismatch: $got B written, $size B expected"
    fi
    sha=$(head -c "$SHA_PREFIX_BYTES" -- "$I_DST.part" | sha256sum | awk '{print $1}')
    mv -f -- "$I_DST.part" "$I_DST"
    t1=$(date +%s)
    printf 'copied_bytes=%s\nsha256_16m=%s\nseconds=%s\n' "$got" "$sha" "$(( t1 - t0 ))"
}

# --- the measurement run -------------------------------------------------------
WORK="" RECORD_WRITTEN=0 FINALIZING=0
PROBE_BIN="" PINNER_BIN="" COMPLETION_BIN="" PIN_START_JSON="null"
inner_on_signal() {
    INNER_STOP=1
    log "signal received: stopping (current child: ${CHILD_PID:-none})"
    if [ -n "$CHILD_PID" ]; then kill -TERM "$CHILD_PID" 2>/dev/null || true; fi
}
inner_on_exit() {
    local rc=$?
    set +e
    sampler_stop
    pinner_stop
    if [ "$RECORD_WRITTEN" != 1 ] && [ "$FINALIZING" = 0 ] && [ -n "$WORK" ] && [ -d "$WORK" ] && [ -n "$REC" ]; then
        rec_bool interrupted 1
        rec_str error "inner side ended early (rc=$rc)"
        inner_finalize
    fi
    if [ -n "$WORK" ] && [ -d "$WORK" ]; then rm -rf -- "$WORK"; fi
    exit "$rc"
}

inner_finalize() {
    local tag=$I_TAG out=$I_OUT_DIR raw ms
    FINALIZING=1
    raw=$out/raw
    mkdir -p -- "$raw" 2>/dev/null || true
    ms=$(inner_py memstats "$WORK/mem.samples")
    rec "mem_samples" "$ms"
    if [ -s "$WORK/io.csv" ]; then
        cp -f -- "$WORK/io.csv" "$out/io_${tag}.csv"
        rec_str "outputs.io_csv" "$(crel "$out/io_${tag}.csv")"
        rec "probe.io_csv_info" "$(inner_py ioinfo "$WORK/io.csv")"
    fi
    if [ -n "$I_TRACE_DIR" ] && [ -s "$WORK/replay.moetrace" ]; then
        mkdir -p -- "$I_TRACE_DIR"
        local sfx
        cp -f -- "$WORK/replay.moetrace" "$I_TRACE_DIR/replay_${tag}.moetrace"
        for sfx in .tokens.bin .gen.txt .meta.json; do
            if [ -f "$WORK/replay.moetrace$sfx" ]; then cp -f -- "$WORK/replay.moetrace$sfx" "$I_TRACE_DIR/replay_${tag}.moetrace$sfx"; fi
        done
        rec_str "outputs.replay_trace" "$(crel "$I_TRACE_DIR/replay_${tag}.moetrace")"
    fi
    if [ -f "$WORK/probe.err" ]; then
        cp -f -- "$WORK/probe.err" "$raw/pagecache_${tag}.probe.log"
        rec_str "outputs.probe_log" "$(crel "$raw/pagecache_${tag}.probe.log")"
    fi
    if [ -f "$WORK/completion.err" ]; then
        cp -f -- "$WORK/completion.err" "$raw/pagecache_${tag}.completion.log"
        rec_str "outputs.completion_log" "$(crel "$raw/pagecache_${tag}.completion.log")"
        if [ -f "$WORK/completion.out" ]; then cp -f -- "$WORK/completion.out" "$raw/pagecache_${tag}.completion.out.log"; fi
    fi
    if [ -f "$WORK/pinner.out" ] || [ -f "$WORK/pinner.err" ]; then
        cat -- "$WORK/pinner.out" "$WORK/pinner.err" "$WORK/pinner_after.out" 2>/dev/null >"$raw/pagecache_${tag}.pinner.log" || true
        rec_str "outputs.pinner_log" "$(crel "$raw/pagecache_${tag}.pinner.log")"
    fi
    if [ -f "$WORK/fio.json" ]; then
        cat -- "$WORK/fio.json" "$WORK/fio.err" 2>/dev/null >"$raw/pagecache_${tag}.fio.log" || true
    fi
    if [ -f "$WORK/mem.samples" ]; then cp -f -- "$WORK/mem.samples" "$raw/pagecache_${tag}.memsamples.log"; fi
    rec_str "outputs.record" "$(crel "$out/pagecache_${tag}.json")"
    rec_str "outputs.inner_log" "$(crel "$raw/pagecache_${tag}.inner.log")"
    rec_str "finished_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if inner_py assemble "$REC" "$out/pagecache_${tag}.json"; then
        RECORD_WRITTEN=1
        log "run record: $(crel "$out/pagecache_${tag}.json")"
    else
        warn "could not write the run record $out/pagecache_${tag}.json"
    fi
    if [ -f "$WORK/inner.log" ]; then cp -f -- "$WORK/inner.log" "$raw/pagecache_${tag}.inner.log" 2>/dev/null || true; fi
}

inner_run() {
    local a b n_tok cap_lim hostj s oom0 oom1 io0 io1 eta
    [ -n "$I_GGUF" ] && [ -n "$I_TOKENS" ] && [ -n "$I_RANGE" ] && [ -n "$I_OUT_DIR" ] \
        || die "$EXIT_USAGE" "inner run needs --gguf, --tokens-bin, --range and --out-dir"
    [ -f "$I_GGUF" ] || die "$EXIT_MISSING" "model $I_GGUF not found in the container (is the volume mounted?)"
    [ -f "$I_TOKENS" ] || die "$EXIT_MISSING" "replay tokens $I_TOKENS not found"
    command -v python3 >/dev/null 2>&1 || die "$EXIT_MISSING" "python3 is required inside the container"
    PROBE_BIN=${PIGEMMA_PAGECACHE_PROBE:-$CONTAINER_ROOT/tools/build-linux-x86/bin/pigemma-probe}
    PINNER_BIN=${PIGEMMA_PAGECACHE_PINNER:-$CONTAINER_ROOT/tools/build-linux-x86/bin/pinner}
    COMPLETION_BIN=${PIGEMMA_PAGECACHE_COMPLETION:-$CONTAINER_ROOT/third_party/llama.cpp/build-linux-x86/bin/llama-completion}
    [ -x "$PROBE_BIN" ] || die "$EXIT_MISSING" "probe binary $PROBE_BIN missing: run 'bash pc/docker_run.sh build-tools' on the host"
    if [ -n "$I_PIN_RANGES" ]; then
        [ -x "$PINNER_BIN" ] || die "$EXIT_MISSING" "pinner binary $PINNER_BIN missing: run 'bash pc/docker_run.sh build-tools' on the host"
        [ -f "$I_PIN_RANGES" ] || die "$EXIT_MISSING" "ranges file $I_PIN_RANGES not found"
    fi
    case $I_RANGE in
        *:*) a=${I_RANGE%%:*} b=${I_RANGE#*:} ;;
        *) die "$EXIT_USAGE" "--range must be A:B" ;;
    esac
    is_uint "$a" && is_uint "$b" && [ "$b" -gt "$a" ] || die "$EXIT_USAGE" "--range must be A:B with integers 0 <= A < B (the host resolves slices)"
    n_tok=$(( b - a ))
    mkdir -p -- "$I_OUT_DIR/raw" || die "$EXIT_ERROR" "cannot create $I_OUT_DIR/raw"
    WORK=$(mktemp -d /tmp/pigemma-pagecache.XXXXXX) || die "$EXIT_ERROR" "mktemp failed"
    LOG_FILE=$WORK/inner.log
    LOG_TAG="$PROG inner $I_TAG"
    REC=$WORK/record.kv
    : >"$REC"
    set_phase idle
    trap inner_on_signal TERM INT HUP
    trap inner_on_exit EXIT

    rec_str record pigemma.pagecache_run
    rec_num format_version 1
    rec_str generator "pc/pagecache_limit.sh $VERSION"
    rec_str tag "$I_TAG"
    rec_str created_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    rec_str volume "$I_VOLUME_KIND"
    rec_str gguf "$I_GGUF"
    rec_num gguf_size "$(stat -c %s -- "$I_GGUF")"
    rec_str tokens_bin "$(crel "$I_TOKENS")"
    rec_str token_range "$a:$b"
    rec_num n_tokens "$n_tok"
    rec_num threads "$I_THREADS"
    rec_num ctx "$I_CTX"
    rec_num mem_cap_bytes_requested "$I_MEM_CAP"
    rec_num page_size "$(getconf PAGESIZE 2>/dev/null || true)"
    rec_bool pinned "$([ -n "$I_PIN_RANGES" ] && echo 1 || echo 0)"
    hostj=$(python3 -m pigemma.paths host 2>/dev/null >"$WORK/host.json" && inner_py compact "$WORK/host.json") || hostj=null
    rec host "${hostj:-null}"
    rec_str "binaries.probe" "$PROBE_BIN"
    rec_str "binaries.pinner" "$PINNER_BIN"
    rec_str "binaries.completion" "$COMPLETION_BIN"

    # cgroup + device
    cg_detect
    cap_lim=$(cg_limit)
    rec_str "cgroup.version" "$CG_VER"
    rec_num "cgroup.memory_limit_bytes" "$cap_lim"
    gate "cgroup $CG_VER memory limit = ${cap_lim:-none} B (requested ${I_MEM_CAP:-none} B)"
    if [ -n "$I_MEM_CAP" ] && [ -z "$cap_lim" ]; then
        rec_note "no cgroup memory limit visible inside the container: the cap may not have applied"
    fi
    rec_cg_snapshot "cgroup.before"
    dev_info "$I_GGUF"
    rec_str "device.dev" "$DEV_MM"
    rec_num "device.read_ahead_kb" "$DEV_RA_KB"
    rec_num "device.logical_block_size" "$DEV_LBS"
    rec_str "device.fstype" "$DEV_FSTYPE"
    rec_str "device.source" "$DEV_SOURCE"
    log "volume device ${DEV_SOURCE:-?} (${DEV_FSTYPE:-?}, dev ${DEV_MM:-?}): read_ahead_kb=${DEV_RA_KB:-not visible}"

    sampler_start

    # 1. tier bandwidth (O_DIRECT: does not touch the page cache), 2. cold start
    if [ "$INNER_STOP" = 0 ]; then set_phase fio; inner_fio; fi
    if [ "$INNER_STOP" = 0 ]; then set_phase evict; inner_evict; fi

    # 3. optional pinner (same cgroup)
    local run_probe=1
    if [ "$INNER_STOP" = 0 ] && [ -n "$I_PIN_RANGES" ]; then
        set_phase pin
        rec_str "pinner.ranges_file" "$(crel "$I_PIN_RANGES")"
        pinner_start
        if [ "$PIN_HOLDING" != 1 ]; then
            rec_note "pinner did not reach a holding state (status $(jget "$PIN_START_JSON" status), rc $PIN_START_RC): the pinned run is skipped"
            run_probe=0
            rec_bool "probe.skipped" 1
        fi
    fi

    # 4. the probe replay under the cap
    if [ "$INNER_STOP" = 0 ] && [ "$run_probe" = 1 ]; then
        local pargs=(--mode moe-trace -m "$I_GGUF" --replay "$I_TOKENS" --replay-range "$a:$b")
        if [ -n "$I_CHUNKS" ]; then pargs+=(--replay-chunks "$I_CHUNKS"); fi
        if [ -n "$I_CHUNK_TOKENS" ]; then pargs+=(--chunk-tokens "$I_CHUNK_TOKENS"); fi
        pargs+=(-ub 1 -nr -t "$I_THREADS" -c "$I_CTX" --io-csv "$WORK/io.csv" --trace-out "$WORK/replay.moetrace")
        rec "probe.argv" "$(inner_py argv "$PROBE_BIN" "${pargs[@]}")"
        if is_num "$TIER_MBPS"; then
            eta=$(awk -v n="$n_tok" -v t="$TIER_MBPS" -v w="$I_BPT_WORST" 'BEGIN { if (t > 0) printf "%.1f", n * w / 1e6 / t / 60; else print "?" }')
            log "probe: $n_tok tokens at -ub 1; worst case (every expert of every token read at ${TIER_MBPS%%.*} MB/s) ~$eta min, less at larger caps"
        fi
        oom0=$(cg_oom_kill)
        io0=$(cg_io_rbytes)
        set_phase probe
        sample_now probe
        log "probe: $PROBE_BIN ${pargs[*]}"
        run_child "$WORK/probe.out" "$WORK/probe.err" "$I_PROBE_TIMEOUT" "$PROBE_BIN" "${pargs[@]}"
        sample_now probe
        set_phase idle
        oom1=$(cg_oom_kill)
        io1=$(cg_io_rbytes)
        rec_num "probe.exit_code" "$CHILD_RC"
        rec_num "probe.wall_s" "$CHILD_WALL_S"
        rec_bool "probe.timed_out" "$CHILD_TIMED_OUT"
        rec "probe.summary" "$(inner_py summary "$WORK/probe.err")"
        rec_cg_snapshot "cgroup.after_probe"
        rec_num "cgroup.oom_kill_before" "$oom0"
        rec_num "cgroup.oom_kill_after_probe" "$oom1"
        if is_uint "$oom0" && is_uint "$oom1"; then
            rec_num "cgroup.oom_kill_probe_delta" "$(( oom1 - oom0 ))"
            if [ "$oom1" -gt "$oom0" ]; then rec_note "OOM: cgroup oom_kill +$(( oom1 - oom0 )) during the probe run"; fi
        fi
        if is_uint "$io0" && is_uint "$io1"; then rec_num "cgroup.io_rbytes_probe_delta" "$(( io1 - io0 ))"; fi
        if [ "$CHILD_TIMED_OUT" = 1 ]; then
            rec_note "probe stopped by --probe-timeout ${I_PROBE_TIMEOUT}s: io.csv covers only the tokens done until then"
        fi
        log "probe: exit $CHILD_RC after ${CHILD_WALL_S}s (io.csv rows: $(inner_py ioinfo "$WORK/io.csv" | sed 's/.*"rows": \([0-9]*\).*/\1/'))"
        s=$(inner_py summary "$WORK/probe.err")
        if [ "$s" = null ]; then warn "probe printed no PIGEMMA_SUMMARY line (tail: $(tail -n 3 "$WORK/probe.err" 2>/dev/null | tr '\n' ' '))"; fi
    fi

    # 5. natural generation under the same cap (pinner still holding)
    if [ "$INNER_STOP" = 0 ] && [ "$run_probe" = 1 ] && [ "$I_COMP_TOKENS" != 0 ]; then
        if [ ! -x "$COMPLETION_BIN" ]; then
            rec_note "llama-completion not found at $COMPLETION_BIN: natural-generation tok/s not measured"
        elif [ ! -f "$I_COMP_PROMPT" ]; then
            rec_note "completion prompt $I_COMP_PROMPT not found: natural-generation tok/s not measured"
        else
            local cargs=(-m "$I_GGUF" -nr -t "$I_THREADS" -c "$I_COMP_CTX" -n "$I_COMP_TOKENS" --temp 0
                -f "$I_COMP_PROMPT" -no-cnv --no-display-prompt -lv 4)
            rec "completion.argv" "$(inner_py argv "$COMPLETION_BIN" "${cargs[@]}")"
            set_phase completion
            sample_now completion
            log "completion: $COMPLETION_BIN ${cargs[*]}"
            run_child "$WORK/completion.out" "$WORK/completion.err" "$I_PROBE_TIMEOUT" "$COMPLETION_BIN" "${cargs[@]}"
            sample_now completion
            set_phase idle
            rec_num "completion.exit_code" "$CHILD_RC"
            rec_num "completion.wall_s" "$CHILD_WALL_S"
            rec_bool "completion.timed_out" "$CHILD_TIMED_OUT"
            rec_num "cgroup.oom_kill_after_completion" "$(cg_oom_kill)"
            log "completion: exit $CHILD_RC after ${CHILD_WALL_S}s"
        fi
    fi

    # 6. pinner residency after the run, then stop it (SIGTERM to its pidfile only)
    if [ -n "$PINNER_PID" ]; then
        set_phase pin
        "$PINNER_BIN" --gguf "$I_GGUF" --ranges "$I_PIN_RANGES" --verify-only --json >"$WORK/pinner_after.out" 2>>"$WORK/pinner.err" || true
        rec "pinner.verify_after" "$(inner_py jsonline "$WORK/pinner_after.out")"
        pinner_stop
        rec_str "pinner.stop_rc" "$PIN_STOP_RC"
        set_phase idle
    fi
    sampler_stop
    rec_cg_snapshot "cgroup.after"
    if [ "$INNER_STOP" = 1 ]; then rec_bool interrupted 1; fi
    inner_finalize
    trap - EXIT
    sampler_stop
    rm -rf -- "$WORK"
    WORK=""
    if [ "$INNER_STOP" = 1 ]; then exit 130; fi
    exit 0
}

inner_main() {
    local task=run
    I_GGUF="" I_TOKENS="" I_RANGE="" I_CHUNKS="" I_CHUNK_TOKENS="" I_THREADS=6 I_CTX=2048 I_TAG=run
    I_OUT_DIR="" I_TRACE_DIR="" I_PIN_RANGES="" I_PIN_TIMEOUT=900 I_FIO_RUNTIME=15 I_FIO_BS=3345k
    I_COMP_TOKENS=64 I_COMP_PROMPT="" I_COMP_CTX=2048 I_VOLUME_KIND=pigemma-models I_MEM_CAP=""
    I_PROBE_TIMEOUT=0 I_SRC="" I_DST="" I_EXPECT_BYTES="" I_SAMPLE_S=0.5 I_NO_SHA=0 I_BPT_WORST=802897920
    LOG_TAG="$PROG inner"
    TS_UTC=1
    split_eq_args "$@"
    set -- "${SPLIT_ARGS[@]}"
    while [ $# -gt 0 ]; do
        case $1 in
            --inner) shift ;;
            --task) need_val "$1" $#; task=$2; shift 2 ;;
            --gguf) need_val "$1" $#; I_GGUF=$2; shift 2 ;;
            --tokens-bin) need_val "$1" $#; I_TOKENS=$2; shift 2 ;;
            --range) need_val "$1" $#; I_RANGE=$2; shift 2 ;;
            --replay-chunks) need_val "$1" $#; I_CHUNKS=$2; shift 2 ;;
            --chunk-tokens) need_val "$1" $#; I_CHUNK_TOKENS=$2; shift 2 ;;
            --threads) need_val "$1" $#; I_THREADS=$2; shift 2 ;;
            --ctx) need_val "$1" $#; I_CTX=$2; shift 2 ;;
            --tag) need_val "$1" $#; I_TAG=$2; shift 2 ;;
            --out-dir) need_val "$1" $#; I_OUT_DIR=$2; shift 2 ;;
            --trace-dir) need_val "$1" $#; I_TRACE_DIR=$2; shift 2 ;;
            --pin-ranges) need_val "$1" $#; I_PIN_RANGES=$2; shift 2 ;;
            --pin-timeout) need_val "$1" $#; I_PIN_TIMEOUT=$2; shift 2 ;;
            --fio-runtime) need_val "$1" $#; I_FIO_RUNTIME=$2; shift 2 ;;
            --fio-bs) need_val "$1" $#; I_FIO_BS=$2; shift 2 ;;
            --completion-tokens) need_val "$1" $#; I_COMP_TOKENS=$2; shift 2 ;;
            --completion-prompt) need_val "$1" $#; I_COMP_PROMPT=$2; shift 2 ;;
            --completion-ctx) need_val "$1" $#; I_COMP_CTX=$2; shift 2 ;;
            --volume-kind) need_val "$1" $#; I_VOLUME_KIND=$2; shift 2 ;;
            --mem-cap-bytes) need_val "$1" $#; I_MEM_CAP=$2; shift 2 ;;
            --probe-timeout) need_val "$1" $#; I_PROBE_TIMEOUT=$2; shift 2 ;;
            --sample-interval) need_val "$1" $#; I_SAMPLE_S=$2; shift 2 ;;
            --src) need_val "$1" $#; I_SRC=$2; shift 2 ;;
            --dst) need_val "$1" $#; I_DST=$2; shift 2 ;;
            --expect-bytes) need_val "$1" $#; I_EXPECT_BYTES=$2; shift 2 ;;
            --bpt-worst-bytes) need_val "$1" $#; I_BPT_WORST=$2; shift 2 ;;
            --no-sha) I_NO_SHA=1; shift ;;
            -h|--help) host_usage; exit 0 ;;
            *) die "$EXIT_USAGE" "inner: unknown argument '$1'" ;;
        esac
    done
    local v
    for v in I_THREADS I_CTX I_PIN_TIMEOUT I_FIO_RUNTIME I_COMP_TOKENS I_COMP_CTX I_PROBE_TIMEOUT; do
        is_uint "${!v}" || die "$EXIT_USAGE" "inner: $v='${!v}' must be a non-negative integer"
    done
    is_uint "$I_BPT_WORST" || I_BPT_WORST=802897920
    if [ -n "$I_MEM_CAP" ] && ! is_uint "$I_MEM_CAP"; then die "$EXIT_USAGE" "inner: --mem-cap-bytes must be an integer"; fi
    is_num "$I_SAMPLE_S" || die "$EXIT_USAGE" "inner: --sample-interval must be a number"
    [[ $I_TAG =~ ^[A-Za-z0-9._-]+$ ]] || die "$EXIT_USAGE" "inner: --tag '$I_TAG' must be [A-Za-z0-9._-]+"
    if [ "$(uname -s)" != Linux ]; then die "$EXIT_USAGE" "--inner runs inside the Linux container only (use the host form on $(uname -s))"; fi
    case $task in
        preflight) inner_preflight ;;
        copy) inner_copy ;;
        run) inner_run ;;
        *) die "$EXIT_USAGE" "inner: unknown --task '$task' (run, preflight, copy)" ;;
    esac
}

# =============================================================================
# HOST SIDE
# =============================================================================
HOST_PY=""
IFS= read -r -d '' HOST_PY <<'PY' || true
"""Host helper of pc/pagecache_limit.sh (runs from the repo root; uses pigemma.*)."""
import argparse, json, math, os, shlex, sys, time
from pathlib import Path


def emit(**kv):
    for k, v in kv.items():
        if v is None:
            s = ""
        elif isinstance(v, bool):
            s = "1" if v else "0"
        elif isinstance(v, float):
            s = repr(v) if math.isfinite(v) else ""
        else:
            s = str(v)
        print(f"{k}={shlex.quote(s)}")


def clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item") and callable(getattr(o, "item")):
        try:
            return clean(o.item())
        except Exception:  # noqa: BLE001
            return str(o)
    return o


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def save(p, doc):
    p = Path(p)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(clean(doc), indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rel(p):
    from pigemma import paths
    return paths.repo_relative(p)


def op_model(a):
    from pigemma import paths
    p = paths.model_path(a.spec)
    mid = a.spec if a.spec in paths.MODELS else paths.model_id_for(p)
    exists = p.is_file()
    size = p.stat().st_size if exists else None
    expected = paths.MODELS[mid]["bytes"] if mid in paths.MODELS else size
    rp = p.resolve()
    try:
        r = rp.relative_to(paths.REPO_ROOT.resolve()).as_posix()
    except ValueError:
        r = ""
    emit(MODEL_ID=mid or "", MODEL_FILE=rp.as_posix(), MODEL_REL=r, MODEL_BASENAME=p.name,
         MODEL_DIR=rp.parent.as_posix(), MODEL_EXISTS=exists, MODEL_SIZE=size, MODEL_BYTES=expected,
         MODEL_SIZE_OK=bool(exists and (expected is None or size == expected)))


def op_sha(a):
    from pigemma import paths
    emit(MODEL_SHA16=paths.sha256_prefix(a.file)["sha256"])


def op_plan(a):
    from pigemma import cache_sim as cs
    from pigemma import trace as tr
    h = tr.read_header(a.trace)
    with tr.TraceReader(a.trace) as r:
        n, L, K, E, trunc = r.n, r.L, r.K, r.E, bool(r.truncated)
    tok = Path(a.trace + ".tokens.bin")
    n_tok = tok.stat().st_size // 4 if tok.is_file() else None
    lim = n if n_tok is None else min(n, n_tok)
    try:
        ra, rb = cs.parse_token_range(a.range, n)
    except Exception as e:  # noqa: BLE001
        print(f"PLAN_ERROR={shlex.quote(str(e))}")
        return 0
    notes = []
    if n_tok is not None and n_tok != n:
        notes.append(f"{tok.name} holds {n_tok} tokens, the trace {n} records")
    if rb > lim:
        notes.append(f"range end {rb} clipped to {lim}")
        rb = lim
    chunks = [c for c in (h.get("chunks") or []) if isinstance(c, dict)]
    segs, s = [], 0
    for c in chunks:
        ln = int(c.get("n_prompt") or 0) + int(c.get("n_gen") or 0)
        segs.append((s, s + ln))
        s += ln
    pieces = [min(e, rb) - max(b0, ra) for b0, e in segs if min(e, rb) > max(b0, ra)]
    maxseg = max(pieces) if pieces else (rb - ra)
    at_start = (ra == 0) or any(b0 == ra for b0, _ in segs)
    if segs and s != n:
        notes.append(f"header chunks cover {s} tokens, the trace has {n} records")
    if not at_start:
        notes.append(f"range start {ra} is not a source chunk start: the first replay chunk lacks its prefix")
    ctx = max(int(a.ctx_min), ((maxseg + 8 + 255) // 256) * 256)
    out = Path(a.chunks_out)
    if not a.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        save(out, {"chunks": [{"n_prompt": int(c.get("n_prompt") or 0), "n_gen": int(c.get("n_gen") or 0)}
                              for c in chunks],
                   "source_trace": rel(a.trace), "generator": "pc/pagecache_limit.sh",
                   "note": "chunk boundaries of the source trace for pigemma-probe --replay-chunks"})
    m = load(a.map)
    sm = m.get("summary") or {}
    tg = Path(str(h.get("gguf_path") or "").replace("\\", "/")).name
    compat = (int(m.get("n_expert") or 0) == E and int(m.get("n_expert_used") or 0) == K
              and len(m.get("moe_layers") or []) == L)
    emit(N_RECORDS=n, N_TOKENS_BIN=n_tok, RANGE_A=ra, RANGE_B=rb, N_RANGE=rb - ra, MAX_SEG=maxseg, CTX_EFF=ctx,
         A_AT_CHUNK_START=at_start, N_CHUNKS=len(chunks), NONEXPERT_BYTES=int(sm.get("nonexpert_bytes") or 0),
         EXPERT_UNIT=sm.get("expert_unit_bytes"), BPT_WORST=sm.get("expert_bytes_per_token_worst"),
         MAP_GGUF=m.get("gguf_basename"), MAP_SIZE=m.get("file_size"),
         MAP_OK=bool(m.get("gguf_basename") == a.model_basename and int(m.get("file_size") or -1) == int(a.model_bytes)),
         TRACE_GGUF=tg, TRACE_COMPAT=compat, TRACE_L=L, TRACE_K=K, TRACE_E=E, TRACE_TRUNCATED=trunc,
         PLAN_NOTES=" | ".join(notes))


def op_vmdf(a):
    d = load(a.file)
    vm = (d.get("vm_root") or {}).get("avail_bytes")
    hv = d.get("host_vm_disk") or {}
    bc = ""
    for row in d.get("docker_system_df") or []:
        if str(row.get("Type", "")).lower().startswith("build cache"):
            bc = str(row.get("Reclaimable", ""))
    emit(VM_ROOT_AVAIL=vm, HOST_DRIVE_AVAIL=hv.get("drive_avail_bytes"), HOST_DRIVE_DIR=hv.get("dir"),
         HOST_VM_FILE=hv.get("file"), BUILD_CACHE_RECLAIMABLE=bc)


def op_ref(a):
    from pigemma import cache_sim as cs
    rp, io = Path(a.record), Path(a.io_csv)
    if not rp.is_file():
        emit(REF_OK=False, REF_REASON="no reference record")
        return 0
    rec = load(rp)
    probe = rec.get("probe") or {}
    reasons = []
    if a.check:
        if Path(str(rec.get("gguf") or "")).name != a.gguf_basename:
            reasons.append("different GGUF")
        if int(rec.get("gguf_size") or -1) != int(a.gguf_size):
            reasons.append("different GGUF size")
        if rec.get("token_range") != a.range:
            reasons.append(f"token range {rec.get('token_range')} != {a.range}")
        if int(rec.get("threads") or -1) != int(a.threads):
            reasons.append("different thread count")
        if rec.get("volume") != a.volume:
            reasons.append("different volume kind")
    if probe.get("exit_code") not in (0,):
        reasons.append(f"reference probe exit {probe.get('exit_code')}")
    meas = None
    if io.is_file():
        try:
            meas = cs.measure_io(io, a.warmup, page_size=a.page_size, bind_mount=a.volume == "bind-mount")
        except Exception as e:  # noqa: BLE001
            reasons.append(f"io.csv: {e}")
    else:
        reasons.append("no io.csv")
    tok = (meas or {}).get("measured_tok_s")
    if not tok:
        reasons.append("no steady-state tok/s")
    emit(REF_OK=not reasons, REF_REASON="; ".join(reasons), REF_TOK_S=tok,
         REF_BPT=(meas or {}).get("measured_bytes_per_token"), REF_CAP=rec.get("mem_cap_bytes_requested"))


def routing_agreement(src, rep, a0):
    import numpy as np
    from pigemma import trace as tr
    with tr.TraceReader(src) as s, tr.TraceReader(rep) as r:
        n = min(r.n, s.n - a0)
        if n <= 0 or s.L != r.L or s.K != r.K:
            return None
        si = np.sort(np.asarray(s.ids[a0:a0 + n]), axis=2)
        ri = np.sort(np.asarray(r.ids[:n]), axis=2)
        same = (si == ri).all(axis=2)
        ov = (si[..., :, None] == ri[..., None, :]).any(axis=-1).sum(axis=-1) / float(s.K)
        return {"n_tokens": int(n), "set_agreement": float(same.mean()), "mean_overlap": float(ov.mean()),
                "source_trace": rel(src), "replay_trace": rel(rep), "source_offset": int(a0)}


def op_analyze(a):
    from pigemma import cache_sim as cs
    rec = load(a.record)
    notes = []
    probe = rec.get("probe") or {}
    cg = rec.get("cgroup") or {}
    bind = a.volume == "bind-mount"
    meas, rows = None, 0
    io = Path(a.io_csv)
    if io.is_file():
        try:
            d = cs.read_io_csv(io)
            rows = int(d["token_index"].size)
            if rows:
                meas = cs.measure_io(d, a.warmup, page_size=a.page_size, bind_mount=bind)
                rp = (d["rss_anon_kb"] + d["rss_file_kb"]).max()
                meas["rss_peak_bytes"] = int(rp) * 1024
        except Exception as e:  # noqa: BLE001
            notes.append(f"io.csv unreadable: {e}")
    if meas and not meas.get("measured_tokens"):
        notes.append(f"io.csv has no rows after the {a.warmup}-token warm-up")
    msp = ((rec.get("mem_samples") or {}).get("probe") or {})
    anon = (meas or {}).get("anon_rss_bytes_max") or None
    anon_src = "probe RssAnon max (io.csv)"
    if not anon and msp.get("max_anon"):
        anon, anon_src = int(msp["max_anon"]), "cgroup memory.stat anon max during the probe"
    file_cache = msp.get("max_file") or None
    oom_delta = cg.get("oom_kill_probe_delta")
    exit_code = probe.get("exit_code")
    oom = bool(oom_delta) and int(oom_delta) > 0
    if exit_code == 137 and not oom:
        notes.append("probe exit 137 (SIGKILL) without an oom_kill count")
    if probe.get("skipped"):
        notes.append("probe skipped (pinner did not hold)")
    elif exit_code not in (0, None) and not probe.get("timed_out") and not oom:
        notes.append(f"probe exit code {exit_code}")
    if rec.get("interrupted"):
        notes.append("run interrupted")
    for n in rec.get("notes") or []:
        if n not in notes:
            notes.append(n)
    fio = rec.get("fio") or {}
    tier = num(fio.get("mbps")) if fio.get("ok") else None
    ra = (rec.get("device") or {}).get("read_ahead_kb")
    rg = int(ra) * 1024 if isinstance(ra, (int, float)) and ra > 0 else None
    lim = cg.get("memory_limit_bytes")
    if a.cap_bytes and lim is not None and abs(int(lim) - int(a.cap_bytes)) > (1 << 20):
        notes.append(f"cgroup limit {lim} B differs from the requested cap {a.cap_bytes} B")
    budget = None
    if anon is not None and a.cap_bytes:
        budget = int(a.cap_bytes) - int(anon) - int(a.nonexpert) - int(a.slack_bytes)
        if budget <= 0:
            notes.append(f"simulator budget {budget} B <= 0 (cap - anon - non-expert - slack): no expert cache left")
    comp, repack = None, None
    cl = Path(a.completion_log) if a.completion_log else None
    if cl and cl.is_file():
        from pigemma import bench
        comp = bench.parse_completion_stderr(cl.read_text(encoding="utf-8", errors="replace"))
        bs = comp.get("buffer_split") or {}
        if bs:
            repack = any(str(k).startswith("CPU_REPACK") for k in bs)
            if repack:
                notes.append("REPACK DETECTED in the llama-completion buffer split (CPU_REPACK): -nr was not "
                             "effective, the run is not page-cache streaming")
        if not comp.get("has_perf"):
            notes.append("llama-completion printed no perf lines (failed?)")
    agree = None
    if a.replay_trace and Path(a.replay_trace).is_file() and Path(a.source_trace).is_file():
        try:
            agree = routing_agreement(a.source_trace, a.replay_trace, int(a.range_a))
        except Exception as e:  # noqa: BLE001
            notes.append(f"routing agreement not computed: {type(e).__name__}: {e}")
        if agree and agree["set_agreement"] < 0.99:
            notes.append(f"replay routing differs from the source trace on {100 * (1 - agree['set_agreement']):.1f}% "
                         f"of (token, layer) expert sets (mean overlap {agree['mean_overlap']:.3f}); "
                         "--sim-on-replay simulates the replayed sequence exactly")
    pin = rec.get("pinner") or {}
    pst = pin.get("start") if isinstance(pin.get("start"), dict) else {}
    pva = pin.get("verify_after") if isinstance(pin.get("verify_after"), dict) else {}
    if pst and pst.get("status") == "partial":
        notes.append(f"pinner partial lock: {pst.get('locked_bytes')} of {pst.get('requested_bytes')} B")
    ha = {"analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "warmup_tokens": a.warmup,
          "io": meas, "io_rows": rows, "anon_rss_bytes": anon, "anon_rss_source": anon_src,
          "file_cache_bytes": file_cache, "oom": oom, "tier_mbps": tier, "read_granularity": rg,
          "sim_budget_bytes": budget, "sim_budget_formula": "mem_cap - anon_rss - nonexpert_bytes - slack",
          "nonexpert_bytes": int(a.nonexpert), "slack_bytes": int(a.slack_bytes), "completion": comp,
          "repack_detected": repack, "routing_agreement": agree, "notes": notes,
          "pinner_resident_before": pst.get("resident_bytes"), "pinner_resident_after": pva.get("resident_bytes")}
    rec["host_analysis"] = ha
    save(a.record, rec)
    Path(a.notes_out).write_text("".join(n.replace("\n", " ") + "\n" for n in notes), encoding="utf-8")
    emit(IO_ROWS=rows, MEAS_TOKENS=(meas or {}).get("measured_tokens") or 0,
         MEAS_BPT=(meas or {}).get("measured_bytes_per_token"), MEAS_TOK_S=(meas or {}).get("measured_tok_s"),
         ANON_RSS=int(anon) if anon is not None else None, FILE_CACHE=int(file_cache) if file_cache else None,
         OOM=oom, PROBE_EXIT=exit_code, PROBE_SKIPPED=bool(probe.get("skipped")),
         PROBE_TIMED_OUT=bool(probe.get("timed_out")), TIER_MBPS=tier,
         READ_GRAN=rg, SIM_BUDGET_BYTES=budget,
         SIM_BUDGET_GB=(f"{budget / 1e9:.9f}" if budget is not None and budget > 0 else None),
         PIN_LOCKED=pst.get("locked_bytes"), PIN_REQUESTED=pst.get("requested_bytes"), PIN_STATUS=pst.get("status"),
         NAT_TG=(comp or {}).get("tg_tok_s"), REPACK=repack,
         AGREE=(agree or {}).get("set_agreement"), REPLAY_N=(agree or {}).get("n_tokens"))


def op_stubsim(a):
    save(a.out, {"kind": "simulated", "runs": [], "warmup_tokens": a.warmup, "trace": a.trace,
                 "notes": [a.note], "generator": "pc/pagecache_limit.sh (no simulation)"})


def op_failpoint(a):
    from pigemma import cache_sim as cs
    notes = [n for n in (a.note or []) if n]
    pt = {"mem_cap_bytes": int(a.cap_bytes), "pinned": bool(a.pinned), "pin_budget_bytes": a.pin_budget_bytes,
          "pin_locked_bytes": None, "anon_rss_bytes": None, "file_cache_bytes": None,
          "tier_rand_read_3mb_mbps": None, "measured_bytes_per_token": None, "measured_majflt_per_token": None,
          "measured_tok_s": None, "sim_budget_bytes": None, "sim_lru_bytes_per_token": None,
          "sim_belady_bytes_per_token": None, "ratio_sim_lru_over_measured": None, "predicted_tok_s": None,
          "oom": bool(a.oom), "notes": notes + ["no usable io.csv rows: the point carries no measurement"],
          "projected": {"kind": "projected", "predicted_tok_s": None, "assumptions": {}}, "failed": True}
    tr_ = [int(x) for x in a.token_range.split(":")] if a.token_range else None
    cs.write_validation(a.validation, pt, gguf=a.gguf, volume=a.volume, token_range=tr_, warmup_tokens=a.warmup)


def _find_point(doc, cap, pinned):
    for p in doc.get("points") or []:
        if int(p.get("mem_cap_bytes", -1)) == int(cap) and bool(p.get("pinned")) == bool(pinned):
            return p
    return None


def op_finish(a):
    from pigemma import results as rs
    rec = load(a.record)
    ha = rec.get("host_analysis") or {}
    probe = rec.get("probe") or {}
    summ = probe.get("summary") if isinstance(probe.get("summary"), dict) else {}
    meas = ha.get("io") or {}
    comp = ha.get("completion") or {}
    point = None
    if a.validation and Path(a.validation).is_file():
        doc = load(a.validation)
        point = _find_point(doc, a.cap_bytes, a.pinned)
        if point is not None:
            point["run_record"] = rel(a.record)
            point["io_csv"] = rel(a.io_csv) if Path(a.io_csv).is_file() else None
            point["sim_json"] = rel(a.sim) if a.sim and Path(a.sim).is_file() else None
            point["natural_gen_tok_s"] = comp.get("tg_tok_s")
            point["routing_agreement"] = (ha.get("routing_agreement") or {}).get("set_agreement")
            point["pinner_resident_before_bytes"] = ha.get("pinner_resident_before")
            point["pinner_resident_after_bytes"] = ha.get("pinner_resident_after")
            point["repack_detected"] = ha.get("repack_detected")
            save(a.validation, doc)
    if a.no_envelopes:
        return 0
    host = rec.get("host") if isinstance(rec.get("host"), dict) and rec["host"].get("hostname") else None
    common = {"experiment": a.experiment, "host": host, "llama_build": summ.get("llama_build"),
              "llama_commit": summ.get("llama_commit"), "model": a.model, "threads": int(a.threads)}
    label = (f"ref{a.cap_str}g" if a.ref else f"cap{a.cap_str}g")
    pin_sfx, file_sfx = ("+pin", "_pin") if a.pinned else ("", "")
    base = Path(a.out_dir) / f"{a.experiment}_pagecache_{label}{file_sfx}"
    inputs = [rel(x) for x in (a.io_csv, a.record, a.sim) if x and Path(x).is_file()]
    notes0 = list(ha.get("notes") or [])
    cap_txt = f"{int(a.cap_bytes) / 1e9:g} GB" if a.cap_bytes else "no"
    proxy = ("desktop Docker proxy: the slow tier is the Docker VM disk, not a Pi storage device"
             if a.volume != "bind-mount" else "bind-mount proxy: lower-bound bytes, timings not representative")
    extra = {"mem_cap_bytes": int(a.cap_bytes) if a.cap_bytes else None, "volume": a.volume, "pinned": bool(a.pinned),
             "reference_run": bool(a.ref), "run_record": rel(a.record), "cgroup": rec.get("cgroup"),
             "fio": rec.get("fio"), "pinner": rec.get("pinner"), "device": rec.get("device"),
             "routing_agreement": ha.get("routing_agreement"), "validation_point": point,
             "token_range": rec.get("token_range"), "warmup_tokens": int(a.warmup)}
    written = []
    if meas.get("measured_tok_s"):
        f = dict(common, kind="measured", variant=f"replay-ub1@{label}{pin_sfx}",
                 flags=[str(x) for x in (probe.get("argv") or [])[1:]], n_ctx=int(a.ctx),
                 tg={"tok_s": meas["measured_tok_s"], "sd": None, "n": meas.get("measured_tokens")},
                 read_bytes_per_token=meas.get("measured_bytes_per_token"),
                 majflt_per_token=meas.get("measured_majflt_per_token"),
                 rss_peak_mb=(meas.get("rss_peak_bytes") / 1e6) if meas.get("rss_peak_bytes") else None,
                 inputs=inputs,
                 notes=notes0 + [f"teacher-forced replay {rec.get('token_range')} at -ub 1 -nr under a {cap_txt} "
                                 f"cgroup memory cap; tok/s and bytes/token over the tokens after the "
                                 f"{a.warmup}-token warm-up", proxy],
                 extra=extra)
        written.append(rs.write_result(f"{base}_replay.json", **f))
    if comp.get("tg_tok_s"):
        f = dict(common, kind="measured", variant=f"gen{a.completion_tokens}@{label}{pin_sfx}",
                 flags=[str(x) for x in ((rec.get("completion") or {}).get("argv") or [])[1:]],
                 n_ctx=int(comp.get("n_ctx") or a.completion_ctx),
                 tg={"tok_s": comp["tg_tok_s"], "sd": None, "n": comp.get("n_eval")},
                 load_s=comp.get("load_s"),
                 inputs=[rel(x) for x in [(rec.get("outputs") or {}).get("completion_log") or ""] if x],
                 notes=[f"plain llama-completion greedy generation under the same {cap_txt} cap "
                        "(natural-generation tok/s)", proxy] + ([n for n in notes0 if "REPACK" in n]),
                 extra={"mem_cap_bytes": extra["mem_cap_bytes"], "pinned": bool(a.pinned), "run_record": rel(a.record),
                        "parsed": comp})
        if comp.get("pp_tok_s"):
            f["pp"] = {"tok_s": comp["pp_tok_s"], "sd": None, "n": comp.get("n_prompt")}
        bs = comp.get("buffer_split") or {}
        if bs:
            f["buffer_split"] = bs
        written.append(rs.write_result(f"{base}_gen.json", **f))
    if point is not None and point.get("predicted_tok_s"):
        pj = point.get("projected") or {}
        f = dict(common, kind="projected", variant=f"predicted@{label}{pin_sfx}", flags=[], n_ctx=int(a.ctx),
                 host=None,
                 tg={"tok_s": point["predicted_tok_s"], "sd": None, "n": None},
                 read_bytes_per_token=point.get("sim_lru_bytes_per_token"),
                 assumptions=pj.get("assumptions") or {},
                 inputs=inputs + [rel(a.validation)],
                 notes=["time model of cache_sim io-point: 1 / (1/ram_tok_s + sim_lru_bytes/tier + n_miss*access_ms)",
                        proxy],
                 extra={"validation_point": point})
        written.append(rs.write_result(f"{base}_predicted.json", **f))
    for p in written:
        print(rel(p))


def _f(v, fmt="{:.2f}", none="-"):
    v = num(v)
    return none if v is None else fmt.format(v)


def op_checks(a):
    doc = load(a.validation)
    pts = doc.get("points") or []
    pool = [p for p in pts if not p.get("pinned")]
    seen = {Path(a.validation).resolve()}
    for c in a.compare or []:
        cp = Path(c)
        if cp.is_file() and cp.resolve() not in seen:
            seen.add(cp.resolve())
            try:
                pool += [p for p in (load(cp).get("points") or []) if not p.get("pinned")]
            except (OSError, ValueError):
                pass
    unp = [p for p in pts if not p.get("pinned")]
    band = [p for p in unp if 4e9 <= int(p.get("mem_cap_bytes", 0)) <= 8e9 and not p.get("failed")]
    rows5 = []
    for p in band:
        r = num(p.get("ratio_sim_lru_over_measured"))
        ok = r is not None and abs(r - 1) <= 0.30 and p.get("within_belady_lru_band") is True
        rows5.append({"mem_cap_bytes": p["mem_cap_bytes"], "ratio": r, "in_band": p.get("within_belady_lru_band"),
                      "pass": ok})
    tm = []
    for p in unp:
        pr, me = num(p.get("predicted_tok_s")), num(p.get("measured_tok_s"))
        if pr and me:
            tm.append({"mem_cap_bytes": p["mem_cap_bytes"], "predicted_over_measured": pr / me,
                       "pass": abs(pr / me - 1) <= 0.35})
    e5 = {"rule": "PASS if |sim_lru/measured - 1| <= 0.30 for caps 4-8 GB and measured lies within "
                  "[0.8 x Belady, 1.3 x LRU]; time model PASS if |predicted/measured - 1| <= 0.35 on >= 3 caps",
          "simulator_bytes": {"points": rows5, "pass": (all(r["pass"] for r in rows5) if rows5 else None)},
          "time_model": {"points": tm, "n_pass": sum(1 for t in tm if t["pass"]),
                         "pass": sum(1 for t in tm if t["pass"]) >= 3},
          "oom_points": [p["mem_cap_bytes"] for p in pts if p.get("oom")]} if unp else None
    e6rows = []
    for p in [q for q in pts if q.get("pinned")]:
        u = next((q for q in pool if int(q.get("mem_cap_bytes", -1)) == int(p["mem_cap_bytes"])
                  and not q.get("failed")), None)
        tok_r = bpt_red = None
        if u is not None:
            pt_, ut = num(p.get("measured_tok_s")), num(u.get("measured_tok_s"))
            pb, ub = num(p.get("measured_bytes_per_token")), num(u.get("measured_bytes_per_token"))
            tok_r = pt_ / ut if pt_ and ut else None
            bpt_red = (1 - pb / ub) if pb is not None and ub else None
        ok = None if u is None else bool((tok_r is not None and tok_r >= 1.15) or (bpt_red is not None and bpt_red >= 0.20))
        e6rows.append({"mem_cap_bytes": p["mem_cap_bytes"], "unpinned_found": u is not None,
                       "tok_s_ratio_pinned_over_unpinned": tok_r, "bytes_per_token_reduction": bpt_red,
                       "pin_locked_over_budget": (num(p.get("pin_locked_bytes")) / num(p.get("pin_budget_bytes"))
                                                  if num(p.get("pin_locked_bytes")) and num(p.get("pin_budget_bytes")) else None),
                       "pass": ok})
    e6 = {"rule": "PASS if pinned tg >= 1.15x unpinned at 5 GB OR bytes/token reduced >= 20%",
          "points": e6rows,
          "pass_at_5gb": next((r["pass"] for r in e6rows if int(r["mem_cap_bytes"]) == 5000000000), None)} if e6rows else None
    doc["checks_preview"] = {"computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "computed_by": "pc/pagecache_limit.sh (mechanical preview of the pre-registered "
                                            "thresholds; the verdict is written by a human in the write-up)",
                             "E5": e5, "E6": e6}
    if a.ref_record or a.ram_tok_s:
        doc["reference"] = {"ram_tok_s": num(a.ram_tok_s), "source": a.ram_tok_s_source,
                            "run_record": rel(a.ref_record) if a.ref_record and Path(a.ref_record).is_file() else None}
    doc["driver"] = {"script": "pc/pagecache_limit.sh", "version": a.version, "argv": json.loads(a.argv_json),
                     "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save(a.validation, doc)
    hdr = ["cap GB", "pin", "meas MB/tok", "majflt/tok", "sim LRU MB", "sim Bel MB", "sim/meas", "band",
           "meas tok/s", "pred tok/s", "gen tok/s", "tier MB/s", "anon MB", "file MB", "oom"]
    lines = []
    for p in pts:
        lines.append([_f(int(p["mem_cap_bytes"]) / 1e9, "{:g}"), "yes" if p.get("pinned") else "no",
                      _f(num(p.get("measured_bytes_per_token")) / 1e6 if num(p.get("measured_bytes_per_token")) is not None else None, "{:.1f}"),
                      _f(p.get("measured_majflt_per_token"), "{:.0f}"),
                      _f(num(p.get("sim_lru_bytes_per_token")) / 1e6 if num(p.get("sim_lru_bytes_per_token")) is not None else None, "{:.1f}"),
                      _f(num(p.get("sim_belady_bytes_per_token")) / 1e6 if num(p.get("sim_belady_bytes_per_token")) is not None else None, "{:.1f}"),
                      _f(p.get("ratio_sim_lru_over_measured"), "{:.3f}"),
                      {True: "in", False: "OUT"}.get(p.get("within_belady_lru_band"), "-"),
                      _f(p.get("measured_tok_s"), "{:.3f}"), _f(p.get("predicted_tok_s"), "{:.3f}"),
                      _f(p.get("natural_gen_tok_s"), "{:.3f}"), _f(p.get("tier_rand_read_3mb_mbps"), "{:.0f}"),
                      _f(num(p.get("anon_rss_bytes")) / 1e6 if num(p.get("anon_rss_bytes")) is not None else None, "{:.0f}"),
                      _f(num(p.get("file_cache_bytes")) / 1e6 if num(p.get("file_cache_bytes")) is not None else None, "{:.0f}"),
                      "OOM" if p.get("oom") else ("FAILED" if p.get("failed") else "0")])
    w = [max(len(h), *(len(r[i]) for r in lines)) if lines else len(h) for i, h in enumerate(hdr)]
    print(f"== {rel(a.validation)}: measured (Docker proxy) vs simulated vs predicted ==")
    print("  ".join(h.rjust(w[i]) for i, h in enumerate(hdr)))
    for r in lines:
        print("  ".join(c.rjust(w[i]) for i, c in enumerate(r)))
    if a.ram_tok_s:
        print(f"ram_tok_s (time-model calibration) = {num(a.ram_tok_s):.3f} tok/s ({a.ram_tok_s_source})")
    if e5:
        sb, tmd = e5["simulator_bytes"], e5["time_model"]
        print(f"E5 preview: simulator bytes { {True: 'PASS', False: 'FAIL', None: 'n/a (no cap in 4-8 GB)'}[sb['pass']] } "
              f"({sum(1 for r in sb['points'] if r['pass'])}/{len(sb['points'])} caps in 4-8 GB pass); "
              f"time model {'PASS' if tmd['pass'] else 'FAIL'} ({tmd['n_pass']} caps within 35%, need 3); "
              f"OOM points: {len(e5['oom_points'])}")
    if e6:
        for r in e6["points"]:
            print(f"E6 preview @ {int(r['mem_cap_bytes']) / 1e9:g} GB: tok/s ratio {_f(r['tok_s_ratio_pinned_over_unpinned'], '{:.3f}')}, "
                  f"bytes/token reduction {_f(r['bytes_per_token_reduction'], '{:.1%}')} -> "
                  f"{ {True: 'PASS', False: 'no gain', None: 'no unpinned point'}[r['pass']] }")


def main():
    ap = argparse.ArgumentParser(prog="pagecache_limit host helper")
    sub = ap.add_subparsers(dest="op", required=True)
    p = sub.add_parser("model"); p.add_argument("spec")
    p = sub.add_parser("sha"); p.add_argument("file")
    p = sub.add_parser("plan")
    for k in ("--trace", "--range", "--map", "--model-basename", "--model-bytes", "--chunks-out", "--ctx-min"):
        p.add_argument(k, required=True)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("vmdf"); p.add_argument("file")
    p = sub.add_parser("ref")
    for k in ("--record", "--io-csv", "--gguf-basename", "--gguf-size", "--range", "--threads", "--volume"):
        p.add_argument(k, required=True)
    p.add_argument("--warmup", type=int, required=True); p.add_argument("--page-size", type=int, default=4096)
    p.add_argument("--check", action="store_true")
    p = sub.add_parser("analyze")
    for k in ("--record", "--io-csv", "--volume", "--nonexpert", "--slack-bytes", "--notes-out", "--source-trace"):
        p.add_argument(k, required=True)
    p.add_argument("--cap-bytes", type=int, default=0); p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--page-size", type=int, default=4096); p.add_argument("--range-a", type=int, default=0)
    p.add_argument("--replay-trace", default=""); p.add_argument("--completion-log", default="")
    p = sub.add_parser("stubsim")
    p.add_argument("--out", required=True); p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--trace", default=None); p.add_argument("--note", required=True)
    p = sub.add_parser("failpoint")
    for k in ("--validation", "--volume"):
        p.add_argument(k, required=True)
    p.add_argument("--cap-bytes", type=int, required=True); p.add_argument("--pinned", type=int, default=0)
    p.add_argument("--pin-budget-bytes", type=int, default=None); p.add_argument("--oom", type=int, default=0)
    p.add_argument("--gguf", default=None); p.add_argument("--token-range", default=None)
    p.add_argument("--warmup", type=int, required=True); p.add_argument("--note", action="append", default=[])
    p = sub.add_parser("finish")
    for k in ("--record", "--io-csv", "--experiment", "--out-dir", "--model", "--cap-str", "--volume"):
        p.add_argument(k, required=True)
    p.add_argument("--validation", default=""); p.add_argument("--sim", default="")
    p.add_argument("--cap-bytes", type=int, default=0); p.add_argument("--pinned", type=int, default=0)
    p.add_argument("--threads", type=int, required=True); p.add_argument("--ctx", type=int, required=True)
    p.add_argument("--warmup", type=int, required=True); p.add_argument("--completion-tokens", type=int, default=64)
    p.add_argument("--completion-ctx", type=int, default=2048)
    p.add_argument("--ref", action="store_true"); p.add_argument("--no-envelopes", action="store_true")
    p = sub.add_parser("checks")
    p.add_argument("--validation", required=True); p.add_argument("--compare", action="append", default=[])
    p.add_argument("--ref-record", default=""); p.add_argument("--ram-tok-s", default="")
    p.add_argument("--ram-tok-s-source", default=""); p.add_argument("--argv-json", default="[]")
    p.add_argument("--version", default="")
    a = ap.parse_args(sys.argv[1:])
    fn = {"model": op_model, "sha": op_sha, "plan": op_plan, "vmdf": op_vmdf, "ref": op_ref, "analyze": op_analyze,
          "stubsim": op_stubsim, "failpoint": op_failpoint, "finish": op_finish, "checks": op_checks}[a.op]
    return fn(a) or 0


sys.exit(main())
PY

PY_BIN=""
pick_python() {
    local c
    for c in "${PIGEMMA_PYTHON:-}" python python3; do
        if [ -z "$c" ]; then continue; fi
        if command -v "$c" >/dev/null 2>&1 \
            && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            PY_BIN=$c
            return 0
        fi
    done
    return 1
}
# hpy OP ARGS... - the host helper (code on stdin); stdout without CR
hpy() { PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "$PY_BIN" - "$@" <<<"$HOST_PY" | tr -d '\r'; }
# pym ARGS... - python -m ... from the repo root, with the command logged
pym() {
    log "run: python $*"
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "$PY_BIN" "$@"
}
show() {  # show CMD... - print the command (dry run)
    local a out=""
    for a in "$@"; do out+=" $(printf '%q' "$a")"; done
    printf 'DRY-RUN:%s\n' "$out"
}

host_usage() {
    cat <<'USAGE'
pc/pagecache_limit.sh - E5/E6: desktop proxy for a RAM-limited Pi (page cache under a cgroup cap)

Usage:
  bash pc/pagecache_limit.sh [--mem 3,4,5,6,8] [--tokens traces/26b/all.moetrace] [--range 0:1500]
                             [--pin] [--skip-copy] [--allow-bind-mount] [--out results/h1/validation.json]
                             [more options below]
  bash /work/pc/pagecache_limit.sh --inner [--task run|preflight|copy] ...   (inside the container; internal)

Main options (manifest):
  --mem LIST          memory caps in GB (1e9 B), comma separated (default 3,4,5,6,8)
  --tokens TRACE      routing trace whose token stream is replayed (default traces/26b/all.moetrace;
                      TRACE.tokens.bin is replayed, TRACE is simulated)
  --range A:B         record range of the trace (Python slice, default 0:1500)
  --pin               E6: pin the hot set (non-expert + hottest experts, budget cap - 0.6 GB anon -
                      0.3 GB slack) with tools/pinner in the same cgroup before each run
  --skip-copy         do not copy the GGUF; the volume must already hold a verified copy
  --allow-bind-mount  if the disk gates fail, read the GGUF through the repo bind mount instead
                      (lower-bound bytes = majflt x page size; timings not representative)
  --out FILE          validation.json to merge points into (default results/h1/validation.json,
                      results/h1/validation_pin.json with --pin); outputs go to its directory

More options:
  --model ID|PATH     GGUF (default 26b-q4)            --map FILE   map.json (default results/map26.json)
  --stats FILE        stats.json for --pin (default <trace without .moetrace>.stats.json)
  --threads N         probe/completion threads (default 6)   --cpus N   docker --cpus (default 6)
  --ctx N             minimum -c of the replay (default 2048; raised to fit the longest chunk)
  --warmup-tokens N   tokens excluded from the steady state, measured and simulated (default 200)
  --ref-mem GB        cap of the uncapped reference run for ram_tok_s (default 20; 0 = none)
  --ram-tok-s X       use this RAM-speed tok/s instead of a reference run
  --redo-ref          rerun the reference even when results/<dir>/pagecache_ref<G>.json matches
  --fio-runtime S     fio tier measurement per cap (default 15; 0 = skip)   --fio-bs B (default 3345k)
  --completion-tokens N  natural-generation tokens per cap (default 64; 0 = skip)
  --completion-prompt F  prompt file (default data/calibration/en_tech_moe_streaming.txt)
  --pin-anon-gb X     anon allowance in the pin budget (default 0.6)   --pin-slack-gb X (default 0.3)
  --sim-slack-gb X    slack in the simulator budget (default 0.1)
  --read-granularity N  cache_sim --read-granularity (default: measured read_ahead_kb x 1024)
  --sim-on-replay     simulate the replayed routing (traces/h1/replay_<tag>.moetrace) instead of the source
  --probe-timeout S   stop the probe (and the completion) after S s; partial io.csv is kept (default 0 = none)
  --pin-timeout S     wait at most S s for the pinner summary (default 900)
  --compare FILE      extra validation.json with unpinned points for the E6 comparison (repeatable)
  --trace-dir DIR     where the replayed routing traces go (default traces/h1)
  --volume NAME       Docker volume (default pigemma-models, env PIGEMMA_PAGECACHE_VOLUME)
  --min-vm-free-gb X  VM root free-space gate for the copy (default 16)
  --min-host-free-gb X  free space gate on the drive holding the Docker VM disk (default 18)
  --min-free-gb X     host free-RAM gate of the bench lock (default 20)
  --no-lock           do not take results/.bench.lock (another job must not load a model meanwhile)
  --keep-going        continue with the next cap after an OOM or a failed run
  --no-envelopes      do not write E5/E6 results envelopes
  --dry-run           print every gate and command; no docker containers, no copies, no writes
  -h, --help          this help        --version   print the version

Outputs (directory of --out, default results/h1/): validation[_pin].json, io_<cap>[_pin].csv,
pagecache_<cap>[_pin].json (run record), sim_e5_<cap>g.json / sim_e6_<cap>g.json, hotset_<cap>.json +
.ranges.txt (--pin), E5_/E6_pagecache_*.json envelopes, raw/*.log; traces/h1/replay_<tag>.moetrace.
Exit codes: 0 ok, 1 a run failed / error, 2 usage, 3 missing input, 4 refused (gates, lock, docker).
USAGE
}

# repo_rel P - repo-relative POSIX path of P (relative to the caller's cwd, absolute, or Windows form)
REPO_POSIX="" REPO_HOST=""
repo_rel() {
    local p=$1 abs root lp lr bs='\'
    p=${p//"$bs"/\/}
    if [ "$IS_MSYS" = 1 ]; then
        case $p in
            [A-Za-z]:/*) abs=$p ;;
            *) abs=$(cygpath -m -a -- "$p" 2>/dev/null) || return 1 ;;
        esac
        root=$REPO_HOST
        lp=${abs,,}
        lr=${root,,}
    else
        case $p in
            /*) abs=$p ;;
            *) abs=$PWD/$p ;;
        esac
        abs=$(realpath -m -- "$abs" 2>/dev/null) || true
        root=$REPO_POSIX
        lp=$abs
        lr=$root
    fi
    lr=${lr%/}
    case $lp in
        "$lr"/*) printf '%s' "${abs:$(( ${#lr} + 1 ))}" ;;
        *) return 1 ;;
    esac
}
user_path() {  # user_path OPTNAME VALUE -> repo-relative or die
    local r
    r=$(repo_rel "$2") || die "$EXIT_USAGE" "$1 '$2' must be inside the repository ($REPO_POSIX): the container sees only /work"
    printf '%s' "$r"
}

DOCKER_RUN=""
DR_DRY=()
docker_run() {  # docker_run ARGS... (pc/docker_run.sh, --dry-run in dry-run mode)
    bash "$DOCKER_RUN" "${DR_DRY[@]}" "$@"
}

# preflight values (PF_*)
parse_kv_into() {
    local prefix=$1 line k v
    while IFS= read -r line; do
        line=${line%$'\r'}
        case $line in *=*) ;; *) continue ;; esac
        k=${line%%=*}
        v=${line#*=}
        if [[ $k =~ ^[a-z0-9_]+$ ]]; then printf -v "${prefix}${k}" '%s' "$v"; fi
    done
}

host_main() {
    local orig_args=("$@")
    # ---- defaults ----
    MEM_LIST=3,4,5,6,8 TRACE=traces/26b/all.moetrace RANGE=0:1500 PIN=0 SKIP_COPY=0 ALLOW_BIND=0 OUT=""
    MODEL=26b-q4 MAP=results/map26.json MAP_GIVEN=0 STATS="" THREADS=6 CPUS=6 CTX_MIN=2048 WARMUP=200
    REF_MEM=20 RAM_TOK_S="" REDO_REF=0 FIO_RUNTIME=15 FIO_BS=3345k COMP_TOKENS=64
    COMP_PROMPT=data/calibration/en_tech_moe_streaming.txt PIN_ANON_GB=0.6 PIN_SLACK_GB=0.3 SIM_SLACK_GB=0.1
    READ_GRAN_OPT="" SIM_ON_REPLAY=0 PROBE_TIMEOUT=0 PIN_TIMEOUT=900 COMPARE=() TRACE_OUT_DIR=traces/h1
    VOLUME=${PIGEMMA_PAGECACHE_VOLUME:-pigemma-models} MIN_VM_FREE_GB=16 MIN_HOST_FREE_GB=18 MIN_FREE_GB=20
    NO_LOCK=0 KEEP_GOING=0 NO_ENVELOPES=0 DRY_RUN=0

    REPO_POSIX=$(cd -- "$SCRIPT_DIR/.." && pwd) || die "$EXIT_MISSING" "cannot resolve the repository root"
    REPO_HOST=$REPO_POSIX
    if [ "$IS_MSYS" = 1 ]; then REPO_HOST=$(cd -- "$REPO_POSIX" && pwd -W 2>/dev/null) || REPO_HOST=$REPO_POSIX; fi

    split_eq_args "$@"
    set -- "${SPLIT_ARGS[@]}"
    while [ $# -gt 0 ]; do
        case $1 in
            --mem) need_val "$1" $#; MEM_LIST=$2; shift 2 ;;
            --tokens) need_val "$1" $#; TRACE=$(user_path "$1" "$2"); shift 2 ;;
            --range) need_val "$1" $#; RANGE=$2; shift 2 ;;
            --pin) PIN=1; shift ;;
            --skip-copy) SKIP_COPY=1; shift ;;
            --allow-bind-mount) ALLOW_BIND=1; shift ;;
            --out) need_val "$1" $#; OUT=$(user_path "$1" "$2"); shift 2 ;;
            --model) need_val "$1" $#; MODEL=$2; shift 2 ;;
            --map) need_val "$1" $#; MAP=$(user_path "$1" "$2"); MAP_GIVEN=1; shift 2 ;;
            --stats) need_val "$1" $#; STATS=$(user_path "$1" "$2"); shift 2 ;;
            --threads) need_val "$1" $#; THREADS=$2; shift 2 ;;
            --cpus) need_val "$1" $#; CPUS=$2; shift 2 ;;
            --ctx) need_val "$1" $#; CTX_MIN=$2; shift 2 ;;
            --warmup-tokens) need_val "$1" $#; WARMUP=$2; shift 2 ;;
            --ref-mem) need_val "$1" $#; REF_MEM=$2; shift 2 ;;
            --ram-tok-s) need_val "$1" $#; RAM_TOK_S=$2; shift 2 ;;
            --redo-ref) REDO_REF=1; shift ;;
            --fio-runtime) need_val "$1" $#; FIO_RUNTIME=$2; shift 2 ;;
            --fio-bs) need_val "$1" $#; FIO_BS=$2; shift 2 ;;
            --completion-tokens) need_val "$1" $#; COMP_TOKENS=$2; shift 2 ;;
            --completion-prompt) need_val "$1" $#; COMP_PROMPT=$(user_path "$1" "$2"); shift 2 ;;
            --pin-anon-gb) need_val "$1" $#; PIN_ANON_GB=$2; shift 2 ;;
            --pin-slack-gb) need_val "$1" $#; PIN_SLACK_GB=$2; shift 2 ;;
            --sim-slack-gb) need_val "$1" $#; SIM_SLACK_GB=$2; shift 2 ;;
            --read-granularity) need_val "$1" $#; READ_GRAN_OPT=$2; shift 2 ;;
            --sim-on-replay) SIM_ON_REPLAY=1; shift ;;
            --probe-timeout) need_val "$1" $#; PROBE_TIMEOUT=$2; shift 2 ;;
            --pin-timeout) need_val "$1" $#; PIN_TIMEOUT=$2; shift 2 ;;
            --compare) need_val "$1" $#; COMPARE+=("$(user_path "$1" "$2")"); shift 2 ;;
            --trace-dir) need_val "$1" $#; TRACE_OUT_DIR=$(user_path "$1" "$2"); shift 2 ;;
            --volume) need_val "$1" $#; VOLUME=$2; shift 2 ;;
            --min-vm-free-gb) need_val "$1" $#; MIN_VM_FREE_GB=$2; shift 2 ;;
            --min-host-free-gb) need_val "$1" $#; MIN_HOST_FREE_GB=$2; shift 2 ;;
            --min-free-gb) need_val "$1" $#; MIN_FREE_GB=$2; shift 2 ;;
            --no-lock) NO_LOCK=1; shift ;;
            --keep-going) KEEP_GOING=1; shift ;;
            --no-envelopes) NO_ENVELOPES=1; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            -h|--help) host_usage; exit 0 ;;
            --version) printf 'pc/pagecache_limit.sh %s\n' "$VERSION"; exit 0 ;;
            *) die "$EXIT_USAGE" "unknown argument '$1' (see --help)" ;;
        esac
    done

    # ---- validation ----
    local v c
    for v in THREADS CTX_MIN WARMUP FIO_RUNTIME COMP_TOKENS PROBE_TIMEOUT PIN_TIMEOUT; do
        is_uint "${!v}" || die "$EXIT_USAGE" "${v,,} must be a non-negative integer (got '${!v}')"
    done
    [ "$THREADS" -ge 1 ] || die "$EXIT_USAGE" "--threads must be >= 1"
    for v in CPUS REF_MEM PIN_ANON_GB PIN_SLACK_GB SIM_SLACK_GB MIN_VM_FREE_GB MIN_HOST_FREE_GB MIN_FREE_GB; do
        is_num "${!v}" || die "$EXIT_USAGE" "${v,,} must be a non-negative number (got '${!v}')"
    done
    if [ -n "$RAM_TOK_S" ] && ! is_num "$RAM_TOK_S"; then die "$EXIT_USAGE" "--ram-tok-s must be a number"; fi
    if [ -n "$READ_GRAN_OPT" ] && ! is_uint "$READ_GRAN_OPT"; then die "$EXIT_USAGE" "--read-granularity must be an integer"; fi
    [[ $FIO_BS =~ ^[0-9]+[kKmM]?$ ]] || die "$EXIT_USAGE" "--fio-bs '$FIO_BS' must look like 3345k"
    [[ $RANGE =~ ^-?[0-9]*:-?[0-9]*$ ]] || die "$EXIT_USAGE" "--range '$RANGE' must be A:B (Python slice)"
    [[ $VOLUME =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die "$EXIT_USAGE" "--volume '$VOLUME' is not a valid volume name"
    CAPS=()
    local IFS_SAVE=$IFS
    IFS=,
    for c in $MEM_LIST; do
        c=${c// /}
        is_num "$c" || { IFS=$IFS_SAVE; die "$EXIT_USAGE" "--mem: '$c' is not a number of GB"; }
        if [ "$(gb_to_bytes "$c")" -le 0 ]; then IFS=$IFS_SAVE; die "$EXIT_USAGE" "--mem: cap '$c' must be > 0"; fi
        CAPS+=("$(canon_num "$c")")
    done
    IFS=$IFS_SAVE
    [ "${#CAPS[@]}" -gt 0 ] || die "$EXIT_USAGE" "--mem gives no caps"
    case $TRACE in *.tokens.bin) TRACE=${TRACE%.tokens.bin} ;; esac
    if [ -z "$OUT" ]; then
        if [ "$PIN" = 1 ]; then OUT=results/h1/validation_pin.json; else OUT=results/h1/validation.json; fi
    fi
    case $OUT in *.json) ;; *) die "$EXIT_USAGE" "--out must name a .json file (got '$OUT')" ;; esac
    if [ -z "$STATS" ]; then STATS=${TRACE%.moetrace}.stats.json; fi
    OUT_DIR=$(dirname -- "$OUT")
    REF_MEM=$(canon_num "$REF_MEM")

    cd -- "$REPO_POSIX" || die "$EXIT_MISSING" "cannot cd to $REPO_POSIX"
    DOCKER_RUN=$SCRIPT_DIR/docker_run.sh
    [ -f "$DOCKER_RUN" ] || die "$EXIT_MISSING" "pc/docker_run.sh not found next to this script"
    pick_python || die "$EXIT_MISSING" "no Python >= 3.10 found (python / python3 / PIGEMMA_PYTHON)"
    if [ "$DRY_RUN" = 1 ]; then DR_DRY=(--dry-run); fi

    # ---- bench lock (D10) ----
    if [ "$DRY_RUN" = 0 ] && [ "$NO_LOCK" = 0 ] && [ -z "${PIGEMMA_LOCK_HELD:-}" ]; then
        gate "bench lock: re-running under 'python -m pigemma.paths lock --hold --min-free-gb $MIN_FREE_GB' (one model-loading job at a time, >= $MIN_FREE_GB GB free host RAM)"
        local lrc=0
        PYTHONUTF8=1 "$PY_BIN" -m pigemma.paths lock --hold --min-free-gb "$MIN_FREE_GB" -- bash pc/pagecache_limit.sh "${orig_args[@]}" || lrc=$?
        exit "$lrc"
    fi
    if [ -n "${PIGEMMA_LOCK_HELD:-}" ]; then gate "bench lock: held by this run's parent (PIGEMMA_LOCK_HELD)"; fi
    if [ "$NO_LOCK" = 1 ]; then gate "bench lock: NOT taken (--no-lock)"; fi

    if [ "$DRY_RUN" = 0 ]; then
        mkdir -p -- "$OUT_DIR/raw" || die "$EXIT_ERROR" "cannot create $OUT_DIR/raw"
        LOG_FILE=$OUT_DIR/raw/pagecache_limit-$(date -u +%Y%m%dT%H%M%SZ).log
        : >"$LOG_FILE" || LOG_FILE=""
    fi
    log "pc/pagecache_limit.sh $VERSION: caps ${CAPS[*]} GB, pin=$PIN, trace $TRACE [$RANGE], out $OUT, threads $THREADS, cpus $CPUS$([ "$DRY_RUN" = 1 ] && echo ', DRY RUN')"
    log "python: $PY_BIN; repo: $REPO_HOST; log file: ${LOG_FILE:-none}"

    host_step0
    host_step1_copy
    host_reference
    local cap failed=0 stop=0
    for cap in "${CAPS[@]}"; do
        if [ "$stop" = 1 ]; then
            warn "cap $cap GB skipped (stopped after an OOM or failure; --keep-going continues)"
            continue
        fi
        POINT_STATUS=ok
        host_point "$cap" "$PIN"
        case $POINT_STATUS in
            ok) ;;
            oom)
                failed=1
                if [ "$KEEP_GOING" = 0 ]; then
                    warn "OOM at cap $cap GB: per EXPERIMENT_PLAN E5 an OOM means the repack leaked or anon exceeded the cap - stopping (fix first, or pass --keep-going)"
                    stop=1
                fi ;;
            skipped) ;;
            *)
                failed=1
                if [ "$KEEP_GOING" = 0 ]; then
                    warn "run at cap $cap GB failed: stopping (--keep-going continues)"
                    stop=1
                fi ;;
        esac
    done
    host_final
    if [ "$failed" = 1 ]; then exit "$EXIT_ERROR"; fi
    exit "$EXIT_OK"
}

host_step0() {
    local miss=0 rc
    # inputs
    if [ ! -f "$TRACE" ]; then
        if [ "$DRY_RUN" = 1 ]; then warn "trace $TRACE not found (dry run continues with placeholders)"; else die "$EXIT_MISSING" "trace $TRACE not found: run pc/trace_collect.sh (E3) first"; fi
        miss=1
    fi
    if [ ! -f "$TRACE.tokens.bin" ]; then
        if [ "$DRY_RUN" = 1 ]; then warn "tokens $TRACE.tokens.bin not found"; else die "$EXIT_MISSING" "replay tokens $TRACE.tokens.bin not found (written by the probe next to the trace)"; fi
        miss=1
    fi
    if [ "$PIN" = 1 ] && [ ! -f "$STATS" ]; then
        if [ "$DRY_RUN" = 1 ]; then warn "stats $STATS not found"; else die "$EXIT_MISSING" "--pin needs $STATS (python -m pigemma.trace stats $TRACE --out $STATS)"; fi
    fi
    if [ "$COMP_TOKENS" != 0 ] && [ ! -f "$COMP_PROMPT" ]; then
        die "$EXIT_MISSING" "completion prompt $COMP_PROMPT not found"
    fi
    # model
    local mo
    mo=$(hpy model "$MODEL") || die "$EXIT_ERROR" "cannot resolve model '$MODEL'"
    eval "$mo"
    gate "model $MODEL -> $MODEL_FILE ($MODEL_SIZE B, expected $MODEL_BYTES B, size ok=$MODEL_SIZE_OK)"
    if [ "$MODEL_EXISTS" != 1 ] || [ "$MODEL_SIZE_OK" != 1 ]; then
        if [ "$DRY_RUN" = 1 ]; then warn "model file missing or wrong size"; else die "$EXIT_MISSING" "model $MODEL_FILE missing or of the wrong size (python -m pigemma.paths model $MODEL --require shows the download command)"; fi
    fi
    MODEL_SPEC=$MODEL
    if [ -z "$MODEL_ID" ]; then MODEL_SPEC=${MODEL_REL:-$MODEL_FILE}; fi
    # map
    if [ ! -f "$MAP" ]; then
        if [ "$MAP_GIVEN" = 1 ]; then die "$EXIT_MISSING" "map $MAP not found"; fi
        if [ "$DRY_RUN" = 1 ]; then
            show "$PY_BIN" -m pigemma.gguf_map build "$MODEL_SPEC" --out "$MAP" --reuse
        else
            gate "map $MAP missing: building it from the GGUF header (header read only)"
            rc=0
            pym -m pigemma.gguf_map build "$MODEL_SPEC" --out "$MAP" --reuse >&2 || rc=$?
            [ "$rc" = 0 ] || die "$EXIT_MISSING" "gguf_map build failed (rc=$rc)"
        fi
    fi
    # plan (trace header, chunk file, map summary)
    CHUNKS_REL=$OUT_DIR/raw/replay_chunks.json
    N_RECORDS="" RANGE_A=0 RANGE_B=1500 N_RANGE=1500 CTX_EFF=$CTX_MIN NONEXPERT_BYTES=1577172088 MAP_OK=1 TRACE_COMPAT=1
    PLAN_NOTES="" TRACE_GGUF="" A_AT_CHUNK_START=1 MAX_SEG="" PLAN_ERROR="" BPT_WORST=802897920
    if [ -f "$TRACE" ] && [ -f "$MAP" ]; then
        local po
        if [ "$DRY_RUN" = 1 ]; then
            po=$(hpy plan --trace "$TRACE" --range "$RANGE" --map "$MAP" --model-basename "$MODEL_BASENAME" \
                --model-bytes "${MODEL_BYTES:-0}" --chunks-out "$CHUNKS_REL" --ctx-min "$CTX_MIN" --dry-run) || die "$EXIT_ERROR" "planning failed"
        else
            po=$(hpy plan --trace "$TRACE" --range "$RANGE" --map "$MAP" --model-basename "$MODEL_BASENAME" \
                --model-bytes "${MODEL_BYTES:-0}" --chunks-out "$CHUNKS_REL" --ctx-min "$CTX_MIN") || die "$EXIT_ERROR" "planning failed (trace $TRACE, map $MAP)"
        fi
        eval "$po"
        if [ -n "$PLAN_ERROR" ]; then die "$EXIT_USAGE" "--range: $PLAN_ERROR"; fi
        BPT_WORST=${BPT_WORST%%.*}
        is_uint "$BPT_WORST" || BPT_WORST=802897920
        gate "trace $TRACE: $N_RECORDS records (L=$TRACE_L K=$TRACE_K E=$TRACE_E, $N_CHUNKS chunks); replay [$RANGE_A:$RANGE_B) = $N_RANGE tokens, longest chunk piece $MAX_SEG -> -c $CTX_EFF"
        if [ -n "$PLAN_NOTES" ]; then warn "plan: $PLAN_NOTES"; fi
        if [ "$MAP_OK" != 1 ]; then die "$EXIT_USAGE" "map $MAP describes $MAP_GGUF ($MAP_SIZE B), not $MODEL_BASENAME ($MODEL_BYTES B)"; fi
        if [ "$TRACE_COMPAT" != 1 ]; then die "$EXIT_USAGE" "trace $TRACE (L=$TRACE_L K=$TRACE_K E=$TRACE_E) does not match map $MAP"; fi
        if [ -n "$TRACE_GGUF" ] && [ "$TRACE_GGUF" != "$MODEL_BASENAME" ]; then
            warn "the trace was recorded on $TRACE_GGUF, the replay uses $MODEL_BASENAME"
        fi
        if [ "$N_RANGE" -le "$WARMUP" ]; then die "$EXIT_USAGE" "range has $N_RANGE tokens, not more than --warmup-tokens $WARMUP"; fi
    else
        warn "plan uses placeholders (missing trace or map): range $RANGE_A:$RANGE_B, -c $CTX_EFF, non-expert $NONEXPERT_BYTES B"
    fi
    # binaries (host side; the env test hooks are checked in the container)
    local b
    for b in "tools/build-linux-x86/bin/pigemma-probe:PIGEMMA_PAGECACHE_PROBE:1" \
             "tools/build-linux-x86/bin/pinner:PIGEMMA_PAGECACHE_PINNER:$PIN" \
             "third_party/llama.cpp/build-linux-x86/bin/llama-completion:PIGEMMA_PAGECACHE_COMPLETION:$([ "$COMP_TOKENS" != 0 ] && echo 1 || echo 0)"; do
        local path=${b%%:*} rest=${b#*:} envn need
        envn=${rest%%:*}
        need=${rest#*:}
        if [ "$need" != 1 ]; then continue; fi
        if [ -n "${!envn:-}" ]; then gate "binary override $envn=${!envn} (container path)"; continue; fi
        if [ ! -f "$path" ]; then
            if [ "$DRY_RUN" = 1 ]; then warn "binary $path not built"; else die "$EXIT_MISSING" "$path not built: run 'bash pc/docker_run.sh build-tools'"; fi
        fi
    done

    # docker preflight + volume state
    VOLUME_KIND=pigemma-models
    PF_page_size=4096 PF_mem_total_kb="" PF_mem_available_kb="" PF_gguf_present=0 PF_gguf_size="" PF_gguf_sha256_16m=""
    PF_have_vmtouch="" PF_have_fio="" PF_cgroup_version="" PF_read_ahead_kb=""
    VOL_EXISTS=0
    if [ "$DRY_RUN" = 0 ]; then
        if docker volume inspect "$VOLUME" >/dev/null 2>&1; then VOL_EXISTS=1; fi
    fi
    gate "docker volume $VOLUME exists: $([ "$VOL_EXISTS" = 1 ] && echo yes || echo no)"
    MODEL_SHA16=""
    if [ "$DRY_RUN" = 0 ] && [ "$MODEL_EXISTS" = 1 ]; then
        eval "$(hpy sha "$MODEL_FILE")"
    fi
    local pfargs=(--no-stdin)
    if [ "$VOL_EXISTS" = 1 ]; then pfargs+=(--volume "$VOLUME:/models:ro"); fi
    local pfout rc2=0
    if [ "$DRY_RUN" = 1 ]; then
        docker_run "${pfargs[@]}" -- bash /work/pc/pagecache_limit.sh --inner --task preflight --gguf "/models/$MODEL_BASENAME" >&2 || true
    else
        pfout=$(docker_run "${pfargs[@]}" -- bash /work/pc/pagecache_limit.sh --inner --task preflight --gguf "/models/$MODEL_BASENAME") || rc2=$?
        if [ "$rc2" = 4 ]; then refuse "docker is not available (pc/docker_run.sh exit 4)"; fi
        if [ "$rc2" = 3 ]; then die "$EXIT_MISSING" "docker image or repo layout missing (pc/docker_run.sh exit 3)"; fi
        [ "$rc2" = 0 ] || die "$EXIT_ERROR" "container preflight failed (rc=$rc2)"
        parse_kv_into PF_ <<<"$pfout"
    fi
    PAGE_SIZE=${PF_page_size:-4096}
    is_uint "$PAGE_SIZE" || PAGE_SIZE=4096
    gate "container: page size $PAGE_SIZE, VM MemTotal ${PF_mem_total_kb:-?} kB (available ${PF_mem_available_kb:-?} kB), cgroup ${PF_cgroup_version:-?}, vmtouch=${PF_have_vmtouch:-?} fio=${PF_have_fio:-?}"
    if is_uint "${PF_mem_total_kb:-}"; then
        local maxcap=0 cb
        for c in "${CAPS[@]}"; do cb=$(gb_to_bytes "$c"); if [ "$cb" -gt "$maxcap" ]; then maxcap=$cb; fi; done
        if [ "$maxcap" -ge $(( PF_mem_total_kb * 1024 )) ]; then
            warn "largest cap $maxcap B >= VM MemTotal $(( PF_mem_total_kb * 1024 )) B: that cap is not binding"
        fi
    fi
    # copy needed?
    NEED_COPY=1
    if [ "$VOL_EXISTS" = 1 ] && [ "${PF_gguf_present:-0}" = 1 ]; then
        if [ "${PF_gguf_size:-}" = "$MODEL_BYTES" ] && { [ -z "$MODEL_SHA16" ] || [ "${PF_gguf_sha256_16m:-}" = "$MODEL_SHA16" ]; }; then
            NEED_COPY=0
            gate "volume $VOLUME already holds $MODEL_BASENAME ($PF_gguf_size B, SHA-256 of the first 16 MiB matches): no copy, no disk gate"
        else
            warn "volume copy of $MODEL_BASENAME does not verify (size ${PF_gguf_size:-?} vs $MODEL_BYTES, sha ${PF_gguf_sha256_16m:0:12} vs ${MODEL_SHA16:0:12})"
        fi
    fi
    if [ "$SKIP_COPY" = 1 ] && [ "$NEED_COPY" = 1 ] && [ "$DRY_RUN" = 0 ]; then
        die "$EXIT_MISSING" "--skip-copy, but volume $VOLUME holds no verified copy of $MODEL_BASENAME (run once without --skip-copy)"
    fi
    if [ "$NEED_COPY" = 1 ] && [ "$SKIP_COPY" = 0 ]; then host_disk_gate; fi
}

host_disk_gate() {
    local vj vmdf_out rc=0 need_vm need_host vm_ok=1 host_ok=1 drive_desc
    need_vm=$(gb_to_bytes "$MIN_VM_FREE_GB")
    need_host=$(gb_to_bytes "$MIN_HOST_FREE_GB")
    if [ "$DRY_RUN" = 1 ]; then
        bash "$DOCKER_RUN" vm-df --json --dry-run >&2 || true
        gate "disk (dry run): would require VM root free >= $need_vm B and host VM-disk drive free >= $need_host B before copying"
        return 0
    fi
    vj=$(mktemp "${TMPDIR:-/tmp}/pigemma-vmdf.XXXXXX")
    bash "$DOCKER_RUN" vm-df --json >"$vj" 2>/dev/null || rc=$?
    if [ "$rc" != 0 ]; then rm -f -- "$vj"; refuse "pc/docker_run.sh vm-df failed (rc=$rc)"; fi
    local vjp=$vj
    if [ "$IS_MSYS" = 1 ]; then vjp=$(cygpath -m -- "$vj"); fi
    vmdf_out=$(hpy vmdf "$vjp") || { rm -f -- "$vj"; die "$EXIT_ERROR" "cannot parse vm-df output"; }
    rm -f -- "$vj"
    VM_ROOT_AVAIL="" HOST_DRIVE_AVAIL="" HOST_DRIVE_DIR="" HOST_VM_FILE="" BUILD_CACHE_RECLAIMABLE=""
    eval "$vmdf_out"
    drive_desc="drive of ${HOST_VM_FILE:-the Docker VM disk}"
    if [ -z "$HOST_DRIVE_AVAIL" ] && [ "$IS_MSYS" = 1 ]; then
        HOST_DRIVE_AVAIL=$(df -P -B1 /c 2>/dev/null | awk 'NR==2 {print $4}') || HOST_DRIVE_AVAIL=""
        drive_desc="C: (the WSL2 vhdx location is unknown; manifest default)"
    fi
    if ! is_uint "$VM_ROOT_AVAIL" || [ "$VM_ROOT_AVAIL" -lt "$need_vm" ]; then vm_ok=0; fi
    if is_uint "$HOST_DRIVE_AVAIL" && [ "$HOST_DRIVE_AVAIL" -lt "$need_host" ]; then host_ok=0; fi
    gate "disk: VM root free ${VM_ROOT_AVAIL:-?} B (need >= $need_vm) -> $([ $vm_ok = 1 ] && echo ok || echo FAIL); $drive_desc free ${HOST_DRIVE_AVAIL:-unknown} B (need >= $need_host) -> $([ $host_ok = 1 ] && echo ok || echo FAIL)"
    if [ "$vm_ok" = 1 ] && [ "$host_ok" = 1 ]; then return 0; fi
    log "docker builder prune would reclaim ~${BUILD_CACHE_RECLAIMABLE:-20 GB} (Build Cache reclaimable per 'docker system df'); pruning is the operator's decision (the cache belongs to other projects) and is never run by this script"
    if [ "$ALLOW_BIND" = 1 ]; then
        VOLUME_KIND=bind-mount
        NEED_COPY=0
        gate "--allow-bind-mount: falling back to the bind mount $MODEL_DIR -> /models (bytes/token = majflt x page size, a lower bound; timings not representative)"
        return 0
    fi
    refuse "not enough disk for the ext4 volume copy of $MODEL_BASENAME ($MODEL_BYTES B); free space, or pass --allow-bind-mount for the lower-bound proxy"
}

host_step1_copy() {
    if [ "$VOLUME_KIND" = bind-mount ]; then return 0; fi
    if [ "$NEED_COPY" = 0 ]; then return 0; fi
    if [ "$SKIP_COPY" = 1 ]; then return 0; fi
    local src cargs=() rc=0 out
    if [ -n "$MODEL_REL" ]; then
        src=$CONTAINER_ROOT/$MODEL_REL
    else
        src=/src/$MODEL_BASENAME
        cargs+=(--volume "$MODEL_DIR:/src:ro")
    fi
    # --as-root: on a native Linux host docker_run would run as the caller's uid, which cannot write
    # into the root-owned named volume (no effect under Git Bash / Docker Desktop)
    cargs+=(--no-stdin --as-root --volume "$VOLUME:/models")
    if [ "$DRY_RUN" = 1 ]; then
        show docker volume create "$VOLUME"
        docker_run "${cargs[@]}" -- bash /work/pc/pagecache_limit.sh --inner --task copy --src "$src" --dst "/models/$MODEL_BASENAME" --expect-bytes "$MODEL_BYTES" >&2 || true
        return 0
    fi
    gate "copy: $MODEL_BASENAME ($MODEL_BYTES B) into Docker volume $VOLUME (one time, ~10-15 min through the bind mount)"
    docker volume create "$VOLUME" >/dev/null || refuse "docker volume create $VOLUME failed"
    out=$(docker_run "${cargs[@]}" -- bash /work/pc/pagecache_limit.sh --inner --task copy --src "$src" --dst "/models/$MODEL_BASENAME" --expect-bytes "$MODEL_BYTES") || rc=$?
    if [ "$rc" != 0 ]; then
        if [ "$rc" = 4 ]; then refuse "the copy container refused (volume full?)"; fi
        die "$EXIT_ERROR" "volume copy failed (rc=$rc)"
    fi
    local CP_copied_bytes="" CP_sha256_16m="" CP_seconds=""
    parse_kv_into CP_ <<<"$out"
    if [ "$CP_copied_bytes" != "$MODEL_BYTES" ]; then die "$EXIT_ERROR" "copy reported $CP_copied_bytes B, expected $MODEL_BYTES"; fi
    if [ -n "$MODEL_SHA16" ] && [ "$CP_sha256_16m" != "$MODEL_SHA16" ]; then
        die "$EXIT_ERROR" "copy verification failed: SHA-256 of the first 16 MiB differs ($CP_sha256_16m vs $MODEL_SHA16)"
    fi
    gate "copy verified: $CP_copied_bytes B, SHA-256(first 16 MiB) $CP_sha256_16m, ${CP_seconds}s"
}

# host_inner_run TAG CAP_BYTES(or empty) [PIN_RANGES_REL] [FIO_RUNTIME]  -> sets INNER_RC
host_inner_run() {
    local tag=$1 capb=$2 ranges=${3:-} fio_rt=${4:-$FIO_RUNTIME} cname
    cname=pigemma-pagecache-${tag//./-}-$$
    local dargs=(--no-stdin --cpus "$CPUS" --name "$cname")
    if [ -n "$capb" ]; then dargs+=(--memory "${capb}b"); fi
    if [ "$VOLUME_KIND" = bind-mount ]; then dargs+=(--volume "$MODEL_DIR:/models:ro"); else dargs+=(--volume "$VOLUME:/models:ro"); fi
    local iargs=(--inner --task run --gguf "/models/$MODEL_BASENAME" --tokens-bin "$CONTAINER_ROOT/$TRACE.tokens.bin"
        --range "$RANGE_A:$RANGE_B" --threads "$THREADS" --ctx "$CTX_EFF" --tag "$tag"
        --out-dir "$CONTAINER_ROOT/$OUT_DIR" --trace-dir "$CONTAINER_ROOT/$TRACE_OUT_DIR"
        --volume-kind "$VOLUME_KIND" --fio-runtime "$fio_rt" --fio-bs "$FIO_BS"
        --completion-tokens "$COMP_TOKENS" --completion-prompt "$CONTAINER_ROOT/$COMP_PROMPT"
        --probe-timeout "$PROBE_TIMEOUT" --pin-timeout "$PIN_TIMEOUT" --bpt-worst-bytes "${BPT_WORST:-802897920}")
    if [ -f "$CHUNKS_REL" ] || [ "$DRY_RUN" = 1 ]; then
        iargs+=(--replay-chunks "$CONTAINER_ROOT/$CHUNKS_REL" --chunk-tokens "$(( N_RANGE >= 2 ? N_RANGE : 2 ))")
    fi
    if [ -n "$capb" ]; then iargs+=(--mem-cap-bytes "$capb"); fi
    if [ -n "$ranges" ]; then iargs+=(--pin-ranges "$CONTAINER_ROOT/$ranges"); fi
    INNER_RC=0
    if [ "$DRY_RUN" = 1 ]; then
        docker_run "${dargs[@]}" -- bash /work/pc/pagecache_limit.sh "${iargs[@]}" >&2 || true
        return 0
    fi
    log "run [$tag]: bash pc/docker_run.sh ${dargs[*]} -- bash /work/pc/pagecache_limit.sh ${iargs[*]}"
    docker_run "${dargs[@]}" -- bash /work/pc/pagecache_limit.sh "${iargs[@]}" >&2 || INNER_RC=$?
    log "run [$tag]: container exit $INNER_RC"
}

RAM_TOK_S_EFF="" RAM_TOK_S_SRC="" REF_RECORD=""
host_reference() {
    if [ -n "$RAM_TOK_S" ]; then
        RAM_TOK_S_EFF=$RAM_TOK_S
        RAM_TOK_S_SRC="given: --ram-tok-s"
        gate "time model: ram_tok_s = $RAM_TOK_S (--ram-tok-s), no reference run"
        return 0
    fi
    if [ "$REF_MEM" = 0 ]; then
        gate "time model: no reference run (--ref-mem 0): predicted_tok_s will be null"
        return 0
    fi
    local tag=ref$REF_MEM capb ro
    capb=$(gb_to_bytes "$REF_MEM")
    REF_RECORD=$OUT_DIR/pagecache_${tag}.json
    if is_uint "${PF_mem_total_kb:-}" && [ "$capb" -ge $(( PF_mem_total_kb * 1024 - 1000000000 )) ]; then
        warn "reference cap $capb B is not below the VM memory ($(( PF_mem_total_kb * 1024 )) B): running the reference without --memory (uncapped inside the VM)"
        capb=""
    fi
    if [ "$DRY_RUN" = 1 ]; then
        gate "reference: uncapped run at ${REF_MEM} GB for ram_tok_s (reused when $REF_RECORD matches)"
        host_inner_run "$tag" "$capb" "" 0
        return 0
    fi
    if [ "$REDO_REF" = 0 ] && [ -f "$REF_RECORD" ]; then
        ro=$(hpy ref --record "$REF_RECORD" --io-csv "$OUT_DIR/io_${tag}.csv" --gguf-basename "$MODEL_BASENAME" \
            --gguf-size "$MODEL_BYTES" --range "$RANGE_A:$RANGE_B" --threads "$THREADS" --volume "$VOLUME_KIND" \
            --warmup "$WARMUP" --page-size "$PAGE_SIZE" --check) || ro="REF_OK=0"
        local REF_OK=0 REF_REASON="" REF_TOK_S="" REF_BPT="" REF_CAP=""
        eval "$ro"
        if [ "$REF_OK" = 1 ]; then
            RAM_TOK_S_EFF=$REF_TOK_S
            RAM_TOK_S_SRC="measured: reused reference run $REF_RECORD (replay at -ub 1, cap ${REF_MEM} GB, all in page cache)"
            gate "time model: reusing $REF_RECORD -> ram_tok_s = $REF_TOK_S tok/s (--redo-ref reruns it)"
            return 0
        fi
        gate "reference $REF_RECORD not reusable ($REF_REASON): rerunning"
    fi
    gate "reference: replay with a ${REF_MEM} GB cap (the whole GGUF fits in the page cache) -> ram_tok_s for the time model"
    host_inner_run "$tag" "$capb" "" 0
    if [ ! -f "$REF_RECORD" ]; then
        warn "reference run wrote no record (container exit $INNER_RC): predicted_tok_s will be null"
        return 0
    fi
    host_analyze "$tag" "$capb"
    ro=$(hpy ref --record "$REF_RECORD" --io-csv "$OUT_DIR/io_${tag}.csv" --gguf-basename "$MODEL_BASENAME" \
        --gguf-size "$MODEL_BYTES" --range "$RANGE_A:$RANGE_B" --threads "$THREADS" --volume "$VOLUME_KIND" \
        --warmup "$WARMUP" --page-size "$PAGE_SIZE") || ro="REF_OK=0"
    local REF_OK=0 REF_REASON="" REF_TOK_S="" REF_BPT="" REF_CAP=""
    eval "$ro"
    if [ "$REF_OK" = 1 ]; then
        RAM_TOK_S_EFF=$REF_TOK_S
        RAM_TOK_S_SRC="measured: reference run $REF_RECORD (replay at -ub 1, cap ${REF_MEM} GB, all in page cache)"
        gate "time model: ram_tok_s = $REF_TOK_S tok/s (reference read $(awk -v b="${REF_BPT:-0}" 'BEGIN{printf "%.2f", b/1e6}') MB/token after the warm-up)"
        if is_num "${REF_BPT:-}" && awk -v b="$REF_BPT" 'BEGIN{exit !(b > 1e6)}'; then
            warn "the reference still reads ${REF_BPT} B/token after the warm-up: it is not fully RAM-resident; ram_tok_s is an underestimate"
        fi
    else
        warn "reference unusable ($REF_REASON): predicted_tok_s will be null"
    fi
    if [ "$NO_ENVELOPES" = 0 ]; then
        hpy finish --record "$REF_RECORD" --io-csv "$OUT_DIR/io_${tag}.csv" --experiment E5 --out-dir "$OUT_DIR" \
            --model "$MODEL_SPEC" --cap-str "$REF_MEM" --volume "$VOLUME_KIND" --cap-bytes "${capb:-0}" \
            --threads "$THREADS" --ctx "$CTX_EFF" --warmup "$WARMUP" --completion-tokens "$COMP_TOKENS" --ref \
            | while IFS= read -r l; do log "envelope: $l"; done || warn "reference envelopes not written"
    fi
}

# host_analyze TAG CAP_BYTES -> eval's the analysis variables
NOTES=()
host_analyze() {
    local tag=$1 capb=${2:-0} ao nf
    nf=$OUT_DIR/raw/pagecache_${tag}.notes.log
    IO_ROWS=0 MEAS_TOKENS=0 MEAS_BPT="" MEAS_TOK_S="" ANON_RSS="" FILE_CACHE="" OOM=0 PROBE_EXIT="" PROBE_SKIPPED=0 PROBE_TIMED_OUT=0
    TIER_MBPS="" READ_GRAN="" SIM_BUDGET_BYTES="" SIM_BUDGET_GB="" PIN_LOCKED="" PIN_REQUESTED="" PIN_STATUS=""
    NAT_TG="" REPACK="" AGREE="" REPLAY_N=""
    ao=$(hpy analyze --record "$OUT_DIR/pagecache_${tag}.json" --io-csv "$OUT_DIR/io_${tag}.csv" \
        --volume "$VOLUME_KIND" --nonexpert "$NONEXPERT_BYTES" --slack-bytes "$(gb_to_bytes "$SIM_SLACK_GB")" \
        --notes-out "$nf" --source-trace "$TRACE" --cap-bytes "${capb:-0}" --warmup "$WARMUP" \
        --page-size "$PAGE_SIZE" --range-a "$RANGE_A" --replay-trace "$TRACE_OUT_DIR/replay_${tag}.moetrace" \
        --completion-log "$OUT_DIR/raw/pagecache_${tag}.completion.log") || { warn "analysis of $tag failed"; return 1; }
    eval "$ao"
    NOTES=()
    if [ -f "$nf" ]; then mapfile -t NOTES < <(tr -d '\r' <"$nf"); fi
    log "[$tag] io rows $IO_ROWS (steady $MEAS_TOKENS), measured $(awk -v b="${MEAS_BPT:-0}" 'BEGIN{printf "%.1f", b/1e6}') MB/token, $(awk -v t="${MEAS_TOK_S:-0}" 'BEGIN{printf "%.3f", t}') tok/s, anon ${ANON_RSS:-?} B, file cache ${FILE_CACHE:-?} B, tier ${TIER_MBPS:-?} MB/s, OOM=$OOM, gen ${NAT_TG:-?} tok/s, routing agreement ${AGREE:-n/a}"
    if [ "$REPACK" = 1 ]; then warn "[$tag] CPU_REPACK buffer seen: -nr was not effective (see the run record)"; fi
    return 0
}

POINT_STATUS=ok
host_point() {
    local cap=$1 pinned=$2 tag capb pin_budget="" ranges="" hs_json exp simf rc
    capb=$(gb_to_bytes "$cap")
    tag=$cap
    exp=E5
    if [ "$pinned" = 1 ]; then tag=${cap}_pin exp=E6; fi
    simf=$OUT_DIR/sim_$(printf '%s' "$exp" | tr 'E' 'e')_${cap}g.json
    log "==== cap $cap GB ($capb B)$([ "$pinned" = 1 ] && echo ', pinned') ===="
    # hot set for --pin (generated on the host from map + stats)
    if [ "$pinned" = 1 ]; then
        pin_budget=$(awk -v m="$capb" -v a="$(gb_to_bytes "$PIN_ANON_GB")" -v s="$(gb_to_bytes "$PIN_SLACK_GB")" 'BEGIN{printf "%.0f", m - a - s}')
        hs_json=$OUT_DIR/hotset_${cap}.json
        ranges=$OUT_DIR/hotset_${cap}.ranges.txt
        gate "pin budget = cap - ${PIN_ANON_GB} GB anon - ${PIN_SLACK_GB} GB slack = $pin_budget B (non-expert first, then hottest experts), page $PAGE_SIZE"
        if [ "$pin_budget" -le 0 ]; then
            warn "cap $cap GB leaves no pin budget: skipped"
            POINT_STATUS=skipped
            return 0
        fi
        if [ "$DRY_RUN" = 1 ]; then
            show "$PY_BIN" -m pigemma.gguf_map hotset --map "$MAP" --stats "$STATS" --budget-bytes "$pin_budget" --page-size "$PAGE_SIZE" --out "$hs_json" --ranges-out "$ranges"
        else
            rc=0
            pym -m pigemma.gguf_map hotset --map "$MAP" --stats "$STATS" --budget-bytes "$pin_budget" \
                --page-size "$PAGE_SIZE" --out "$hs_json" --ranges-out "$ranges" >&2 || rc=$?
            if [ "$rc" = 4 ]; then
                warn "gguf_map refused the pin budget $pin_budget B at cap $cap GB (smaller than the non-expert set): skipped"
                POINT_STATUS=skipped
                return 0
            fi
            [ "$rc" = 0 ] || { warn "hotset generation failed (rc=$rc)"; POINT_STATUS=failed; return 0; }
        fi
    fi
    host_inner_run "$tag" "$capb" "$ranges"
    if [ "$DRY_RUN" = 1 ]; then
        show "$PY_BIN" -m pigemma.cache_sim "$TRACE" --map "$MAP" --policies lru,belady --budgets-gb "<cap - anon_rss - $NONEXPERT_BYTES B - ${SIM_SLACK_GB} GB>" --token-range "$RANGE_A:$RANGE_B" --warmup-tokens "$WARMUP" --no-envelopes --out "$simf"
        show "$PY_BIN" -m pigemma.cache_sim io-point --io-csv "$OUT_DIR/io_${tag}.csv" --sim "$simf" --mem-cap-bytes "$capb" --anon-rss-bytes "<measured>" --tier-mbps "<fio>" --ram-tok-s "<reference>" --warmup-tokens "$WARMUP" --page-size "$PAGE_SIZE" --volume "$VOLUME_KIND" --token-range "$RANGE_A:$RANGE_B" --validation "$OUT"
        return 0
    fi
    local rec_f=$OUT_DIR/pagecache_${tag}.json
    if [ ! -f "$rec_f" ]; then
        local oomf=0 note="container exit $INNER_RC without a run record"
        if [ "$INNER_RC" = 137 ]; then oomf=1 note="container exited 137 (SIGKILL) without a run record: OOM assumed"; fi
        warn "$note"
        hpy failpoint --validation "$OUT" --volume "$VOLUME_KIND" --cap-bytes "$capb" --pinned "$pinned" \
            ${pin_budget:+--pin-budget-bytes "$pin_budget"} --oom "$oomf" --gguf "${MODEL_REL:-$MODEL_BASENAME}" \
            --token-range "$RANGE_A:$RANGE_B" --warmup "$WARMUP" --note "$note" || warn "could not record the failed point"
        if [ "$oomf" = 1 ]; then POINT_STATUS=oom; else POINT_STATUS=failed; fi
        return 0
    fi
    host_analyze "$tag" "$capb" || { POINT_STATUS=failed; return 0; }
    if [ "$IO_ROWS" = 0 ] || [ "$MEAS_TOKENS" = 0 ]; then
        local fargs=() n
        for n in "${NOTES[@]}"; do fargs+=(--note "$n"); done
        hpy failpoint --validation "$OUT" --volume "$VOLUME_KIND" --cap-bytes "$capb" --pinned "$pinned" \
            ${pin_budget:+--pin-budget-bytes "$pin_budget"} --oom "$OOM" --gguf "${MODEL_REL:-$MODEL_BASENAME}" \
            --token-range "$RANGE_A:$RANGE_B" --warmup "$WARMUP" "${fargs[@]}" || warn "could not record the failed point"
        warn "cap $cap GB: no steady-state io.csv rows (probe exit ${PROBE_EXIT:-?}$([ "$PROBE_SKIPPED" = 1 ] && echo ', skipped')): point recorded as failed"
        if [ "$OOM" = 1 ]; then POINT_STATUS=oom; else POINT_STATUS=failed; fi
        return 0
    fi
    # step 3a: simulator at the matching budget
    local sim_trace=$TRACE sim_range="$RANGE_A:$RANGE_B" rg=${READ_GRAN_OPT:-$READ_GRAN}
    if [ "$SIM_ON_REPLAY" = 1 ]; then
        if [ -f "$TRACE_OUT_DIR/replay_${tag}.moetrace" ]; then
            sim_trace=$TRACE_OUT_DIR/replay_${tag}.moetrace
            sim_range=":"
            gate "simulating the replayed routing $sim_trace (--sim-on-replay)"
        else
            warn "--sim-on-replay: $TRACE_OUT_DIR/replay_${tag}.moetrace missing; simulating the source trace"
        fi
    fi
    if [ -n "$SIM_BUDGET_GB" ]; then
        gate "simulator budget = cap $capb - anon $ANON_RSS - non-expert $NONEXPERT_BYTES - slack $(gb_to_bytes "$SIM_SLACK_GB") = $SIM_BUDGET_BYTES B ($SIM_BUDGET_GB GB)"
        local sargs=(-m pigemma.cache_sim "$sim_trace" --map "$MAP" --policies lru,belady --budgets-gb "$SIM_BUDGET_GB"
            --token-range "$sim_range" --warmup-tokens "$WARMUP" --no-envelopes --out "$simf" --log-level WARNING)
        if [ -n "$rg" ]; then sargs+=(--read-granularity "$rg"); fi
        rc=0
        pym "${sargs[@]}" >&2 || rc=$?
        if [ "$rc" != 0 ]; then warn "cache_sim exit $rc (see above); continuing with what it wrote"; fi
        if [ ! -f "$simf" ]; then
            hpy stubsim --out "$simf" --warmup "$WARMUP" --trace "$sim_trace" --note "cache_sim failed (exit $rc)" || true
        fi
    else
        gate "simulator budget <= 0 or unknown (anon ${ANON_RSS:-?} B): no simulation for this cap"
        hpy stubsim --out "$simf" --warmup "$WARMUP" --trace "$sim_trace" --note "no simulation: cap - anon - non-expert - slack <= 0" || true
    fi
    # step 3b: io-point (appends the point to the validation file)
    local ip=(-m pigemma.cache_sim io-point --io-csv "$OUT_DIR/io_${tag}.csv" --sim "$simf" --mem-cap-bytes "$capb"
        --warmup-tokens "$WARMUP" --page-size "$PAGE_SIZE" --volume "$VOLUME_KIND" --gguf "${MODEL_REL:-$MODEL_BASENAME}"
        --token-range "$RANGE_A:$RANGE_B" --validation "$OUT" --log-level WARNING)
    if [ -n "$ANON_RSS" ]; then ip+=(--anon-rss-bytes "$ANON_RSS"); fi
    if [ -n "$FILE_CACHE" ]; then ip+=(--file-cache-bytes "$FILE_CACHE"); fi
    if [ -n "$TIER_MBPS" ]; then ip+=(--tier-mbps "$TIER_MBPS"); fi
    if [ -n "$RAM_TOK_S_EFF" ]; then ip+=(--ram-tok-s "$RAM_TOK_S_EFF"); fi
    if [ -n "$SIM_BUDGET_BYTES" ] && is_uint "$SIM_BUDGET_BYTES"; then ip+=(--sim-budget-bytes "$SIM_BUDGET_BYTES"); fi
    if [ "$pinned" = 1 ]; then
        ip+=(--pinned --pin-budget-bytes "$pin_budget")
        if is_uint "${PIN_LOCKED:-}"; then ip+=(--pin-locked-bytes "$PIN_LOCKED"); fi
    fi
    if [ "$OOM" = 1 ]; then ip+=(--oom); fi
    local n
    for n in "${NOTES[@]}"; do ip+=(--note "$n"); done
    rc=0
    pym "${ip[@]}" >"$OUT_DIR/raw/pagecache_${tag}.point.log" || rc=$?
    if [ "$rc" != 0 ]; then
        warn "io-point failed (rc=$rc)"
        POINT_STATUS=failed
        return 0
    fi
    hpy finish --record "$rec_f" --io-csv "$OUT_DIR/io_${tag}.csv" --experiment "$exp" --out-dir "$OUT_DIR" \
        --model "$MODEL_SPEC" --cap-str "$cap" --volume "$VOLUME_KIND" --validation "$OUT" --sim "$simf" \
        --cap-bytes "$capb" --pinned "$pinned" --threads "$THREADS" --ctx "$CTX_EFF" --warmup "$WARMUP" \
        --completion-tokens "$COMP_TOKENS" $([ "$NO_ENVELOPES" = 1 ] && echo --no-envelopes) \
        | while IFS= read -r l; do log "envelope: $l"; done || warn "envelopes for $tag not written"
    gate "point (cap $cap GB, pinned $pinned) merged into $OUT"
    if [ "$OOM" = 1 ]; then POINT_STATUS=oom; return 0; fi
    if [ "$PROBE_TIMED_OUT" = 1 ]; then
        log "cap $cap GB: the probe was stopped by --probe-timeout ${PROBE_TIMEOUT}s; the point uses the $MEAS_TOKENS steady tokens it finished"
    elif [ -n "$PROBE_EXIT" ] && [ "$PROBE_EXIT" != 0 ]; then
        POINT_STATUS=failed
        return 0
    fi
    POINT_STATUS=ok
}

host_final() {
    if [ "$DRY_RUN" = 1 ]; then
        show "$PY_BIN" - checks --validation "$OUT" "(summary table + checks_preview)"
        log "dry run complete: nothing was executed"
        return 0
    fi
    if [ ! -f "$OUT" ]; then
        warn "no validation file written ($OUT)"
        return 0
    fi
    local cargs=(checks --validation "$OUT" --version "$VERSION")
    local c
    for c in "${COMPARE[@]}"; do cargs+=(--compare "$c"); done
    if [ "$OUT" != results/h1/validation.json ]; then cargs+=(--compare results/h1/validation.json); fi
    if [ -n "$REF_RECORD" ]; then cargs+=(--ref-record "$REF_RECORD"); fi
    if [ -n "$RAM_TOK_S_EFF" ]; then cargs+=(--ram-tok-s "$RAM_TOK_S_EFF" --ram-tok-s-source "$RAM_TOK_S_SRC"); fi
    local argvj
    argvj='[]'
    if [ "${#HOST_ARGV[@]}" -gt 0 ]; then
        argvj=$(printf '%s\n' "${HOST_ARGV[@]}" | PYTHONUTF8=1 "$PY_BIN" -c 'import json,sys; print(json.dumps([l.rstrip("\r\n") for l in sys.stdin]))' | tr -d '\r') || argvj='[]'
    fi
    cargs+=(--argv-json "$argvj")
    hpy "${cargs[@]}" || warn "summary/checks failed"
    log "done: $OUT (log: ${LOG_FILE:-none})"
}

# =============================================================================
HOST_ARGV=()
main() {
    local a
    for a in "$@"; do
        if [ "$a" = --inner ]; then
            inner_main "$@"
            return
        fi
    done
    HOST_ARGV=("$@")
    host_main "$@"
}
main "$@"
