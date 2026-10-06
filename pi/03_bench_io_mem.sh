#!/usr/bin/env bash
# pi/03_bench_io_mem.sh - measure the two constants every Pi projection depends on and write them where
# the catalog loader picks them up instead of the assumed values:
#   (1) storage read speed at expert-slab granularity (fio, O_DIRECT): sequential 1 MiB reads at QD1/QD8,
#       random expert-slab reads at QD1/QD4, random 16 KiB reads (= one Pi page fault) at QD1 -> IOPS and
#       the per-access latency;
#   (2) memory bandwidth (tools/membw: read / copy / triad at 1, 2, 4 threads).
# Output: results/pi/io_mem.json (manifest data_formats "io_mem.json"). pigemma.feasibility (pi/00) and
# pigemma.cache_sim merge it automatically (--io-mem auto) into the matching catalog device profile and
# mark those fields source 'measured'. The catalog file itself is never modified.
#
# ------------------------------------------------------------------------------------------------ usage
#   bash pi/03_bench_io_mem.sh [--file auto|new|PATH] [--dir DIR] [--size-mb 2048] [--seconds 20] [--quick]
#                              [--out FILE] [--device NAME] [--threads 1,2,4] [--membw-size-mb 512]
#                              [--membw-reps 5] [--ioengine libaio] [--slab-bytes N] [--skip-io] [--skip-mem]
#                              [--require-fio] [--no-busy-check] [--dry-run]
#   bash pi/03_bench_io_mem.sh --check-jobs [same options]   # fio --parse-only of every job (needs fio)
#   bash pi/03_bench_io_mem.sh --from-raw DIR [--out FILE] [--device NAME]   # rebuild io_mem.json from raw outputs
#
#   --file auto     (default) the largest *.gguf >= 1 GiB under --dir, read-only; none -> a new test file
#   --file new      always write a fresh test file of --size-mb MiB in --dir (deleted afterwards)
#   --file PATH     an existing file is read read-only (fio --readonly); a missing path is created as the
#                   test file and deleted afterwards
#   --dir DIR       where the models live (default $MODEL_DIR): the storage that matters
#   --size-mb N     test-file size in MiB (default 2048; --quick 512); with an existing file it caps the dd
#                   fallback read. Creating the file needs free disk >= 1.2 x size (+ KEEP_FREE_GB on /)
#   --seconds S     runtime of each fio job (default 20; --quick 5)
#   --quick         --size-mb 512 --seconds 5 --membw-size-mb 256 --membw-reps 3 (explicit flags win)
#   --out FILE      default results/pi/io_mem.json (results/pi-emu/io_mem.json when PIGEMMA_EMULATED=1)
#   --device NAME   catalog device name to write (default PIGEMMA_DEVICE, else automatic, see below)
#   --threads LIST  membw thread counts (default 1,2,4)
#   --ioengine E    fio engine for the QD>1 jobs (default libaio; falls back to psync = effectively QD1)
#   --slab-bytes N  expert slab size (default: catalog 26b-q4 expert_unit_bytes = 3,345,408 B)
#   --skip-io / --skip-mem   measure only memory / only storage
#   --require-fio   exit 3 when fio is missing instead of the dd fallback (sequential read only)
#   --no-busy-check skip assert_not_busy (loadavg / MemAvailable gate)
#   --dry-run       print the plan, the commands and the precondition verdicts; run and write nothing
#
# ------------------------------------------------------------------------------------------------ outputs
#   results/pi/io_mem.json   {format_version, generator, kind:'measured', host, device, device_source,
#                            collected_at, emulated, protected_override, complete,
#                            storage:{seq_read_mbps_qd1, seq_read_mbps_qd8, rand_read_3mb_mbps_qd1,
#                                     rand_read_3mb_mbps_qd4, rand_read_16k_iops, access_ms},
#                            memory:{read_gbps_by_threads:{'1','2','4'}, copy_gbps, triad_gbps, source, cmd, ...},
#                            skipped, flags, test_file, settings, conditions, host_facts, derived, raw_dir, notes}
#                            Every storage metric is {value, unit, source:'measured', cmd, method, job, detail}.
#                            Metrics that could not be measured are ABSENT (reason in 'skipped'), so the
#                            catalog keeps its assumed value for them. 'derived' is kind 'projected'.
#   results/pi/raw/io_mem-<UTC stamp>/   evidence: facts.json, run.tsv (settings + exact commands),
#                            fio_<job>.json/.stderr, dd_*.txt, membw.json/.stderr (failed runs: failed_*)
#   stdout                   the report: measured table, catalog before/after merge, derived numbers.
#
# ---------------------------------------------------------------------------------------------- exit codes
#   0 io_mem.json written (possibly partial: 'complete': false + warnings, e.g. dd fallback)
#   1 usage / internal error, or the written file does not merge through pigemma.feasibility.load_catalog
#   3 precondition failed (disk, RAM, missing python/fio with --require-fio, file too small) or nothing
#     could be measured
#   4 refused: protected host without PIGEMMA_I_KNOW=1, busy host, or the bench lock is held
#
# ---------------------------------------------------------------------------- design decisions / deviations
# * Expert-slab block size: the manifest says bs=3345k. That is 3,425,280 B (fio k = KiB), not the expert
#   unit, and is not 4 KiB aligned (O_DIRECT needs logical-block alignment). This script uses the catalog's
#   26b-q4 expert_unit_bytes (3,345,408 B = gate_up 2,230,272 + down 1,115,136) rounded UP to a multiple of
#   max(4096, page size): 3,358,720 B on the Pi's 16 KiB pages (what an mmap fault of one slab touches),
#   3,346,432 B on 4 KiB pages. Within 0.4 % of the slab; recorded in settings.
# * --file auto prefers an existing GGUF (read-only, no SD writes, real file layout, too large for any
#   device cache); only without one is a test file written (fio write job with random data, end_fsync).
# * All read jobs use fio --readonly (also on our own test file) and --invalidate=1. When O_DIRECT is not
#   supported (some container / network filesystems) the jobs fall back to buffered reads and every such
#   metric carries a 'buffered' flag (page-cache hits possible -> upper bound).
# * ioengine: libaio (QD>1 needs an async engine); if it cannot load, psync is used and the QD4/QD8 metrics
#   are flagged "effectively QD1". One quick 8 MiB probe read selects the engine/direct combination.
# * access_ms = mean total latency (fio lat_ns.mean) of the 16 KiB QD1 random read. It includes the 16 KiB
#   transfer (0.18 ms at 90 MB/s); detail.access_ms_minus_transfer gives the transfer-free value.
#   Conservative for the time model t_io = miss_bytes / rand_read_3mb + n_miss x access_ms.
# * fio missing (the real Pi has none): prints the apt command (never installs: on context-friend-1 that
#   needs the operator's approval) and measures only the sequential read with dd iflag=direct, flagged. The random
#   and latency metrics stay ASSUMED in the catalog. --require-fio turns this into exit 3.
# * Safety on a protected host (PIGEMMA_I_KNOW=1): fio and dd run under nice -n $NICE_LEVEL + ionice -c3
#   (idle class: serves the benchmark only when the services do no I/O), membw under nice. No mlock.
#   The test file obeys KEEP_FREE_GB on the root filesystem. Nothing else is touched.
# * Device name (io_mem.device) when neither --device nor PIGEMMA_DEVICE is given: on a Raspberry Pi 5
#   (device-tree model, or a '-rpi-2712' kernel) or an emulated Pi, the catalog pi5 device with the same nominal RAM tier and storage
#   class (pigemma.feasibility.nominal_ram_gb / storage_class_for), else 'pi5-<tier>gb-<class>' (e.g.
#   pi5-16gb-microsd for context-friend-1), which pigemma.feasibility synthesizes from the name. Other hosts:
#   'linux-<GB>gb-<class>'. Feasibility also matches io_mem.json to facts by hostname.
# * PIGEMMA_EMULATED=1: the default output is results/pi-emu/io_mem.json so emulated numbers can never be
#   auto-merged as Pi measurements; the file says emulated:true.
# * The bench lock (results/.bench.lock, pigemma.paths) is checked: a held lock (a model-loading run) makes
#   this script refuse (exit 4) because both measurements would be distorted.
# * PIGEMMA_FAKE_PROC (tests) supports --dry-run, --check-jobs and --from-raw; a real measurement is refused.
# * --from-raw re-parses a raw directory (any fio JSON whose job names are the five job names below,
#   dd_seq_read_qd1.txt, membw.json, optional run.tsv / facts.json). It never writes into that directory.
# * Derived numbers (all 'projected'): in-RAM decode ceiling tok/s = mem_bw_read x llama_efficiency /
#   bytes_per_token for every catalog model (efficiency 0.65 ASSUMED until pi/05 writes
#   results/pi/calibration.json); expert streaming: all-miss tok/s, and at the catalog's viable (1.0) and
#   good (2.0) tok/s the miss MB/token the storage can serve and the minimum expert-cache hit rate,
#   using the feasibility time model and ignoring compute time (so they are optimistic bounds).
set -euo pipefail

PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIGEMMA_SCRIPT=03_bench_io_mem.sh
# shellcheck source=lib/common.sh
. "$PI_DIR/lib/common.sh"

B3_VERSION="1.0.0"
B3_DEFAULT_SLAB=3345408 # catalog models[26b-q4].expert_unit_bytes (verified in the GGUF header)
B3_JOBS=(seq_read_qd1 seq_read_qd8 rand_read_3mb_qd1 rand_read_3mb_qd4 rand_read_16k_qd1)
MIB=1048576

# ------------------------------------------------------------------------------------------- options
OPT_FILE=auto
OPT_DIR=""
OPT_SIZE_MB=""
OPT_SECONDS=""
OPT_QUICK=0
OPT_OUT=""
OPT_DEVICE="${PIGEMMA_DEVICE:-}"
OPT_THREADS="1,2,4"
OPT_MEMBW_MB=""
OPT_MEMBW_REPS=""
OPT_ENGINE=libaio
OPT_SLAB=""
SKIP_IO=0
SKIP_MEM=0
REQUIRE_FIO=0
BUSY_CHECK=1
DRY_RUN=0
CHECK_JOBS=0
FROM_RAW=""

usage() {
    cat <<EOF
pi/03_bench_io_mem.sh $B3_VERSION - measure storage read speed (fio) and memory bandwidth (membw) on the Pi
and write results/pi/io_mem.json, which replaces the catalog's ASSUMED storage / mem_bw values.

Usage:
  bash pi/03_bench_io_mem.sh [--file auto|new|PATH] [--dir DIR] [--size-mb 2048] [--seconds 20] [--quick]
                             [--out FILE] [--device NAME] [--threads 1,2,4] [--membw-size-mb 512]
                             [--membw-reps 5] [--ioengine libaio] [--slab-bytes N] [--skip-io] [--skip-mem]
                             [--require-fio] [--no-busy-check] [--dry-run]
  bash pi/03_bench_io_mem.sh --check-jobs [options]     fio --parse-only of every job string (needs fio)
  bash pi/03_bench_io_mem.sh --from-raw DIR [--out FILE] [--device NAME]
                                                         rebuild io_mem.json from a raw directory

  --file auto   largest *.gguf >= 1 GiB under --dir, read-only (default); none found -> new test file
  --file new    write a fresh --size-mb test file in --dir (deleted afterwards)
  --file PATH   existing file: read-only; missing path: created as the test file, deleted afterwards
  --dir DIR     model directory to measure (default MODEL_DIR=$MODEL_DIR)
  --size-mb N   test-file size in MiB (default 2048, --quick 512); needs free disk >= 1.2 x size
  --seconds S   runtime per fio job (default 20, --quick 5)
  --quick       short run: --size-mb 512 --seconds 5 --membw-size-mb 256 --membw-reps 3
  --out FILE    default results/pi/io_mem.json (results/pi-emu/io_mem.json when PIGEMMA_EMULATED=1)
  --device NAME device name written to io_mem.json (default PIGEMMA_DEVICE, else automatic)
  --threads L   membw thread counts (default 1,2,4)
  --membw-size-mb N / --membw-reps N   membw buffer (default 512 MiB) and repetitions (default 5)
  --ioengine E  fio engine (default libaio; automatic fallback to psync)
  --slab-bytes N  expert slab bytes (default catalog 26b-q4 expert_unit_bytes 3345408), rounded up to the page
  --skip-io | --skip-mem   only memory | only storage
  --require-fio exit 3 if fio is missing (default: dd fallback, sequential read only, flagged)
  --no-busy-check   skip the loadavg / MemAvailable gate
  --dry-run     print the plan and precondition verdicts, run nothing, write nothing
  -h, --help | --version

fio jobs (each --output-format=json --direct=1 --readonly --runtime=S --time_based):
  seq_read_qd1 / seq_read_qd8        rw=read     bs=1M      iodepth=1 / 8
  rand_read_3mb_qd1 / _qd4           rw=randread bs=<slab>  iodepth=1 / 4
  rand_read_16k_qd1                  rw=randread bs=16k     iodepth=1   (IOPS + access latency)
membw: tools/build/bin/membw --size-mb 512 --threads 1,2,4 --json

Exit codes: 0 written (maybe partial), 1 usage/internal/merge-check error, 3 precondition failed or nothing
measured, 4 refused (protected host without PIGEMMA_I_KNOW=1, busy host, bench lock held).
EOF
}

_b3_usage_err() {
    printf '%s\n' "Try: bash pi/03_bench_io_mem.sh --help" >&2
    die 1 "usage: $*"
}

parse_args() {
    local opt val has_eq
    while [ $# -gt 0 ]; do
        opt=$1
        val=""
        has_eq=0
        case $opt in
            --*=*)
                val=${opt#*=}
                opt=${opt%%=*}
                has_eq=1
                ;;
        esac
        case $opt in
            --file | --dir | --size-mb | --seconds | --out | --device | --threads | --membw-size-mb | \
                --membw-reps | --ioengine | --slab-bytes | --from-raw)
                if [ "$has_eq" = 0 ]; then
                    if [ $# -lt 2 ]; then _b3_usage_err "$opt needs a value"; fi
                    val=$2
                    shift
                fi
                ;;
            *)
                if [ "$has_eq" = 1 ]; then _b3_usage_err "$opt takes no value"; fi
                ;;
        esac
        case $opt in
            --file) OPT_FILE=$val ;;
            --dir) OPT_DIR=$val ;;
            --size-mb) OPT_SIZE_MB=$val ;;
            --seconds) OPT_SECONDS=$val ;;
            --quick) OPT_QUICK=1 ;;
            --out) OPT_OUT=$val ;;
            --device) OPT_DEVICE=$val ;;
            --threads) OPT_THREADS=$val ;;
            --membw-size-mb) OPT_MEMBW_MB=$val ;;
            --membw-reps) OPT_MEMBW_REPS=$val ;;
            --ioengine) OPT_ENGINE=$val ;;
            --slab-bytes) OPT_SLAB=$val ;;
            --skip-io) SKIP_IO=1 ;;
            --skip-mem) SKIP_MEM=1 ;;
            --require-fio) REQUIRE_FIO=1 ;;
            --no-busy-check) BUSY_CHECK=0 ;;
            --dry-run) DRY_RUN=1 ;;
            --check-jobs) CHECK_JOBS=1 ;;
            --from-raw) FROM_RAW=$val ;;
            -h | --help)
                usage
                exit 0
                ;;
            --version)
                printf 'pi/03_bench_io_mem.sh %s\n' "$B3_VERSION"
                exit 0
                ;;
            *) _b3_usage_err "unknown option '$opt'" ;;
        esac
        shift
    done

    if [ "$OPT_QUICK" = 1 ]; then
        OPT_SIZE_MB=${OPT_SIZE_MB:-512}
        OPT_SECONDS=${OPT_SECONDS:-5}
        OPT_MEMBW_MB=${OPT_MEMBW_MB:-256}
        OPT_MEMBW_REPS=${OPT_MEMBW_REPS:-3}
    fi
    SIZE_MB=${OPT_SIZE_MB:-2048}
    RUN_S=${OPT_SECONDS:-20}
    MEMBW_MB=${OPT_MEMBW_MB:-512}
    MEMBW_REPS=${OPT_MEMBW_REPS:-5}

    _b3_uint_in "$SIZE_MB" 16 1048576 || _b3_usage_err "--size-mb must be an integer 16..1048576 (MiB), got '$SIZE_MB'"
    _b3_uint_in "$RUN_S" 1 3600 || _b3_usage_err "--seconds must be an integer 1..3600, got '$RUN_S'"
    _b3_uint_in "$MEMBW_MB" 1 1048576 || _b3_usage_err "--membw-size-mb must be an integer 1..1048576, got '$MEMBW_MB'"
    _b3_uint_in "$MEMBW_REPS" 1 1000 || _b3_usage_err "--membw-reps must be an integer 1..1000, got '$MEMBW_REPS'"
    [[ $OPT_THREADS =~ ^[1-9][0-9]{0,3}(,[1-9][0-9]{0,3})*$ ]] ||
        _b3_usage_err "--threads must be a comma list of positive integers, got '$OPT_THREADS'"
    [[ $OPT_ENGINE =~ ^[a-z0-9_]+$ ]] || _b3_usage_err "--ioengine must be an engine name, got '$OPT_ENGINE'"
    if [ -n "$OPT_SLAB" ]; then
        _b3_uint_in "$OPT_SLAB" 4096 268435456 || _b3_usage_err "--slab-bytes must be an integer 4096..268435456, got '$OPT_SLAB'"
    fi
    if [ -n "$OPT_DEVICE" ] && ! [[ $OPT_DEVICE =~ ^[A-Za-z0-9._-]+$ ]]; then
        _b3_usage_err "--device must match [A-Za-z0-9._-]+, got '$OPT_DEVICE'"
    fi
    case $OPT_FILE in
        "") _b3_usage_err "--file needs a value (auto, new or a path)" ;;
    esac
    if [ "$SKIP_IO" = 1 ] && [ "$SKIP_MEM" = 1 ]; then
        _b3_usage_err "--skip-io and --skip-mem together leave nothing to measure"
    fi
    if [ -n "$FROM_RAW" ] && { [ "$DRY_RUN" = 1 ] || [ "$CHECK_JOBS" = 1 ]; }; then
        _b3_usage_err "--from-raw cannot be combined with --dry-run or --check-jobs"
    fi
    return 0
}

_b3_uint_in() { # VALUE MIN MAX
    [[ ${1:-} =~ ^[0-9]{1,10}$ ]] || return 1
    [ "$1" -ge "$2" ] && [ "$1" -le "$3" ]
}

# --------------------------------------------------------------------------------------- small helpers
_b3_rt() { # _b3_rt KEY VALUE... -> one "KEY<TAB>VALUE" line in $RAW/run.tsv
    local k=$1 v
    shift
    v="$*"
    v=${v//$'\t'/ }
    v=${v//$'\n'/ }
    v=${v//$'\r'/}
    printf '%s\t%s\n' "$k" "$v" >>"$RAW/run.tsv"
}

_b3_io() { # _b3_io TIMEOUT_S CMD... (nice + ionice -c3 when permitted, coreutils timeout when present)
    local t=$1
    shift
    if have_cmd timeout; then
        run_nice_io timeout "$t" "$@"
    else
        run_nice_io "$@"
    fi
}

_b3_cpu() { # _b3_cpu TIMEOUT_S CMD... (nice, timeout when present)
    local t=$1
    shift
    if have_cmd timeout; then
        run_nice timeout "$t" "$@"
    else
        run_nice "$@"
    fi
}

_b3_py() { # _b3_py ARGS... -> python (>= 3.10) from PIGEMMA_HOME with the repo on PYTHONPATH
    local pp
    pp=$(native_path "$PIGEMMA_HOME")
    if [ -n "${_PG_MSYS:-}" ]; then
        pp="$pp${PYTHONPATH:+;$PYTHONPATH}"
    else
        pp="$pp${PYTHONPATH:+:$PYTHONPATH}"
    fi
    (cd "$PIGEMMA_HOME" && PYTHONPATH="$pp" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 "$PY" "$@")
}

_b3_abs() { # absolute path (no symlink resolution; the file may not exist yet)
    case $1 in
        /* | [A-Za-z]:/* | [A-Za-z]:\\*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$PWD/$1" ;;
    esac
}

_b3_file_size() { # bytes of a regular file
    local s=""
    s=$(stat -c %s -- "$1" 2>/dev/null || true)
    if ! [[ $s =~ ^[0-9]+$ ]]; then
        s=$(wc -c <"$1" 2>/dev/null | tr -d ' ' || true)
    fi
    printf '%s\n' "$s"
}

_b3_tail1() { # last non-empty line of a file (for error messages)
    local f=$1 l=""
    if [ -s "$f" ]; then
        l=$(grep -v '^[[:space:]]*$' "$f" 2>/dev/null | tail -n 1 || true)
    fi
    printf '%s\n' "${l:-no output}"
}

_b3_metrics_of() { # JOB -> io_mem metric name(s)
    case $1 in
        seq_read_qd1) printf 'seq_read_mbps_qd1\n' ;;
        seq_read_qd8) printf 'seq_read_mbps_qd8\n' ;;
        rand_read_3mb_qd1) printf 'rand_read_3mb_mbps_qd1\n' ;;
        rand_read_3mb_qd4) printf 'rand_read_3mb_mbps_qd4\n' ;;
        rand_read_16k_qd1) printf 'rand_read_16k_iops access_ms\n' ;;
    esac
}

_b3_skip_job() { # JOB REASON
    local m
    for m in $(_b3_metrics_of "$1"); do
        _b3_rt "skip.$m" "$2"
    done
}

# ------------------------------------------------------------------------------------- plan / settings
_b3_slab_bytes() { # -> SLAB_UNIT (expert unit bytes) and SLAB_BS (fio bs, page multiple), PAGE
    local unit=$B3_DEFAULT_SLAB cat_v="" align
    if [ -n "$OPT_SLAB" ]; then
        unit=$OPT_SLAB
        SLAB_SOURCE="--slab-bytes"
    else
        SLAB_SOURCE="default $B3_DEFAULT_SLAB (catalog lookup failed)"
        if [ -r "$PIGEMMA_CATALOG" ]; then
            cat_v=$(_b3_py -c 'import json, sys
try:
    c = json.load(open(sys.argv[1], encoding="utf-8-sig"))
    m = [x for x in c.get("models", []) if x.get("id") == "26b-q4"]
    print(int(m[0]["expert_unit_bytes"]))
except Exception:
    print("")' "$(native_path "$PIGEMMA_CATALOG")" 2>/dev/null || true)
            cat_v=${cat_v%$'\r'}
        fi
        if [[ $cat_v =~ ^[0-9]+$ ]] && [ "$cat_v" -ge 4096 ]; then
            unit=$cat_v
            SLAB_SOURCE="catalog models[26b-q4].expert_unit_bytes"
        fi
    fi
    PAGE=$(page_size)
    [[ $PAGE =~ ^[0-9]+$ ]] && [ "$PAGE" -ge 512 ] || PAGE=4096
    align=$PAGE
    if [ "$align" -lt 4096 ]; then align=4096; fi
    SLAB_UNIT=$unit
    SLAB_BS=$(((unit + align - 1) / align * align))
}

_b3_find_gguf() { # largest *.gguf >= 1 GiB under DIR -> path on stdout (empty if none)
    local dir=$1 best="" best_s=0 s f
    [ -d "$dir" ] || return 0
    while IFS= read -r -d '' f; do
        s=$(_b3_file_size "$f")
        if [[ $s =~ ^[0-9]+$ ]] && [ "$s" -ge 1073741824 ] && [ "$s" -gt "$best_s" ]; then
            best=$f
            best_s=$s
        fi
    done < <(find "$dir" -maxdepth 4 -type f -name '*.gguf' -print0 2>/dev/null || true)
    printf '%s\n' "$best"
}

_b3_resolve_file() { # -> FILE FILE_MODE (gguf-readonly|existing-readonly|new) FILE_BYTES MEASURE_DIR
    local dir=${OPT_DIR:-$MODEL_DIR} g
    dir=$(_b3_abs "$dir")
    MEASURE_BASE=$dir
    case $OPT_FILE in
        auto)
            g=$(_b3_find_gguf "$dir")
            if [ -n "$g" ]; then
                FILE=$g
                FILE_MODE=gguf-readonly
            else
                FILE="$dir/.pigemma-io-test.$$.bin"
                FILE_MODE=new
                log "no GGUF >= 1 GiB under $dir: a ${SIZE_MB} MiB test file will be written there (deleted afterwards)"
            fi
            ;;
        new)
            FILE="$dir/.pigemma-io-test.$$.bin"
            FILE_MODE=new
            ;;
        *)
            FILE=$(_b3_abs "$OPT_FILE")
            if [ -d "$FILE" ]; then
                die 1 "--file $OPT_FILE is a directory (use --dir DIR for the model directory)"
            elif [ -e "$FILE" ]; then
                if [ ! -f "$FILE" ]; then
                    die 1 "--file $OPT_FILE is not a regular file (block devices are not measured directly)"
                fi
                if [ ! -r "$FILE" ]; then
                    die 3 "--file $OPT_FILE is not readable"
                fi
                case $FILE in
                    *.gguf | *.GGUF) FILE_MODE=gguf-readonly ;;
                    *) FILE_MODE=existing-readonly ;;
                esac
            else
                FILE_MODE=new
            fi
            ;;
    esac
    MEASURE_DIR=${FILE%/*}
    if [ -z "$MEASURE_DIR" ]; then MEASURE_DIR=/; fi
    if [ "$FILE_MODE" = new ]; then
        FILE_BYTES=$((SIZE_MB * MIB))
    else
        FILE_BYTES=$(_b3_file_size "$FILE")
        [[ $FILE_BYTES =~ ^[0-9]+$ ]] || die 3 "cannot read the size of $FILE"
    fi
    FIO_FILENAME=${FILE//:/\\:} # fio uses ':' as a file-name separator
}

_b3_out_paths() { # -> OUT OUT_DIR CAL_PATH
    local def=results/pi
    if is_emulated; then def=results/pi-emu; fi
    OUT=${OPT_OUT:-$PIGEMMA_HOME/$def/io_mem.json}
    OUT=$(_b3_abs "$OUT")
    case $OUT in
        *.json) ;;
        *) _b3_usage_err "--out must name a .json file, got '$OUT'" ;;
    esac
    OUT_DIR=${OUT%/*}
    CAL_PATH="$OUT_DIR/calibration.json"
    if is_emulated && [ -n "$OPT_OUT" ]; then
        case $OUT in
            */results/pi/io_mem.json) warn "PIGEMMA_EMULATED=1 but --out is results/pi/io_mem.json: pigemma.feasibility will auto-merge EMULATED numbers as if they were Pi measurements" ;;
        esac
    fi
}

_b3_disk_check() { # -> 0 if the new test file fits; sets DISK_MSG (same rule as require_free_disk, factor 1.2)
    local free mount keep=0 need
    free=$(free_disk_bytes "$MEASURE_DIR")
    _pg_df "$(_pg_existing_dir "$MEASURE_DIR")"
    mount=$DF_MOUNT
    if ! [[ $free =~ ^[0-9]+$ ]]; then
        DISK_MSG="free disk unknown for $MEASURE_DIR (df failed)"
        return 1
    fi
    if [ "$mount" = / ]; then
        keep=$(awk -v g="${KEEP_FREE_GB:-2}" 'BEGIN { printf "%.0f", g * 1e9 }')
    fi
    need=$(awk -v b="$FILE_BYTES" -v k="$keep" \
        'BEGIN { a = b * 1.2; c = b + k; n = (a > c) ? a : c; printf "%.0f", n }')
    DISK_MSG="free $(fmt_gb "$free") on ${mount:-$MEASURE_DIR}, need $(fmt_gb "$need") (max(1.2 x $(fmt_gb "$FILE_BYTES"), size + keep-free $(fmt_gb "$keep")))"
    [ "$free" -ge "$need" ]
}

_b3_fio_args() { # JOB -> FIO_ARGS (job options only) + J_RW J_BS J_QD
    case $1 in
        seq_read_qd1) J_RW=read; J_BS=1M; J_QD=1 ;;
        seq_read_qd8) J_RW=read; J_BS=1M; J_QD=8 ;;
        rand_read_3mb_qd1) J_RW=randread; J_BS=$SLAB_BS; J_QD=1 ;;
        rand_read_3mb_qd4) J_RW=randread; J_BS=$SLAB_BS; J_QD=4 ;;
        rand_read_16k_qd1) J_RW=randread; J_BS=16k; J_QD=1 ;;
        *) die 1 "internal: unknown fio job '$1'" ;;
    esac
    FIO_ARGS=(--name="$1" --filename="$FIO_FILENAME" --readonly --rw="$J_RW" --bs="$J_BS" --iodepth="$J_QD"
        --direct="$FIO_DIRECT" --ioengine="$FIO_ENGINE" --runtime="$RUN_S" --time_based --invalidate=1
        --randrepeat=0)
}

_b3_layout_args() { # ENGINE DIRECT -> FIO_ARGS (writes the test file with random data, then fsync)
    FIO_ARGS=(--name=layout --filename="$FIO_FILENAME" --rw=write --bs=1M --size="${SIZE_MB}M" --direct="$2"
        --ioengine="$1" --iodepth=4 --end_fsync=1)
}

_b3_probe_args() { # ENGINE DIRECT -> FIO_ARGS (8 MiB read: does this engine/direct combination work?)
    FIO_ARGS=(--name=probe --filename="$FIO_FILENAME" --readonly --rw=read --bs=1M --io_size=8M --direct="$2"
        --ioengine="$1" --iodepth=1)
}

_b3_cmdline() { # CMD ARGS... -> CMDLINE (shell-quoted only where needed, reproducible)
    local a q out=""
    for a in "$@"; do
        if [[ $a =~ ^[A-Za-z0-9_./=:,+@%-]+$ ]]; then
            q=$a
        else
            printf -v q '%q' "$a"
        fi
        out+="${out:+ }$q"
    done
    CMDLINE=$out
}

_b3_fio_json_ok() { # FILE -> 0 if it holds fio JSON jobs without an error
    [ -s "$1" ] || return 1
    grep -q '"jobs"' "$1" 2>/dev/null || return 1
    if grep -Eq '"error"[[:space:]]*:[[:space:]]*[1-9]' "$1" 2>/dev/null; then
        return 1
    fi
    return 0
}

_b3_run_fio() { # TIMEOUT OUT.json ARGS... -> rc (stderr in OUT minus .json plus .stderr)
    local t=$1 out=$2 rc=0
    shift 2
    _b3_io "$t" fio "$@" --output-format=json --output="$out" 2>"${out%.json}.stderr" || rc=$?
    if [ "$rc" = 0 ] && ! _b3_fio_json_ok "$out"; then
        rc=97
    fi
    return "$rc"
}

# ------------------------------------------------------------------------------------- measurement
_b3_remove_testfile() {
    if [ -n "${CREATED_FILE:-}" ] && [ -e "$CREATED_FILE" ]; then
        if rm -f -- "$CREATED_FILE"; then
            log "deleted the test file $CREATED_FILE"
        else
            warn "could not delete the test file $CREATED_FILE - remove it by hand"
        fi
    fi
    CREATED_FILE=""
}

_b3_create_testfile() {
    local combo e d rc ok=0 t
    mkdir -p -- "$MEASURE_DIR" || die 3 "cannot create $MEASURE_DIR"
    CREATED_FILE=$FILE
    add_cleanup _b3_remove_testfile
    t=$((SIZE_MB / 2 + 300)) # >= 2 MB/s write
    log "writing the ${SIZE_MB} MiB test file $FILE (one-off, random data, fsync; deleted afterwards)"
    if [ "$FIO_OK" = 1 ]; then
        for combo in "$OPT_ENGINE 1" "psync 1" "psync 0"; do
            e=${combo% *}
            d=${combo#* }
            _b3_layout_args "$e" "$d"
            rc=0
            _b3_run_fio "$t" "$RAW/layout.json" "${FIO_ARGS[@]}" || rc=$?
            if [ "$rc" = 0 ]; then
                _b3_cmdline fio "${FIO_ARGS[@]}"
                _b3_rt cmd.layout "$CMDLINE"
                ok=1
                break
            fi
            log "fio layout with ioengine=$e direct=$d failed (rc $rc): $(_b3_tail1 "$RAW/layout.stderr")"
            rm -f -- "$FILE"
        done
    fi
    if [ "$ok" = 0 ]; then
        _b3_cmdline dd if=/dev/urandom of="$FILE" bs=1M count="$SIZE_MB" conv=fsync
        _b3_rt cmd.layout "$CMDLINE"
        rc=0
        _b3_io "$t" env LC_ALL=C dd if=/dev/urandom of="$FILE" bs=1M count="$SIZE_MB" conv=fsync \
            2>"$RAW/dd_layout.txt" || rc=$?
        if [ "$rc" != 0 ]; then
            _b3_remove_testfile
            die 3 "could not write the test file $FILE (dd rc $rc: $(_b3_tail1 "$RAW/dd_layout.txt"))"
        fi
    fi
    FILE_BYTES=$(_b3_file_size "$FILE")
    if [ "$FILE_BYTES" != $((SIZE_MB * MIB)) ]; then
        _b3_remove_testfile
        die 3 "test file has $FILE_BYTES bytes, expected $((SIZE_MB * MIB)) (disk full?)"
    fi
}

_b3_select_fio_mode() { # -> FIO_ENGINE FIO_DIRECT; 1 if fio cannot read the file at all
    local engines=("$OPT_ENGINE") e d rc
    if [ "$OPT_ENGINE" != psync ]; then engines+=(psync); fi
    for d in 1 0; do
        for e in "${engines[@]}"; do
            _b3_probe_args "$e" "$d"
            rc=0
            _b3_run_fio 120 "$RAW/probe.json" "${FIO_ARGS[@]}" || rc=$?
            if [ "$rc" = 0 ]; then
                FIO_ENGINE=$e
                FIO_DIRECT=$d
                if [ "$d" = 0 ]; then
                    warn "O_DIRECT reads fail on this filesystem: fio falls back to BUFFERED reads (page cache invalidated per job; results flagged as upper bounds)"
                fi
                if [ "$e" != "$OPT_ENGINE" ]; then
                    warn "fio ioengine $OPT_ENGINE does not work here: using $e (QD4/QD8 jobs become effectively QD1, flagged)"
                fi
                return 0
            fi
            log "fio probe ioengine=$e direct=$d failed (rc $rc): $(_b3_tail1 "$RAW/probe.stderr")"
        done
    done
    return 1
}

_b3_storage_fio() {
    local job rc
    if ! _b3_select_fio_mode; then
        warn "fio cannot read $FILE with any engine/direct combination: falling back to dd"
        _b3_storage_dd "fio unusable on this file (see raw/probe.stderr)"
        return 0
    fi
    _b3_rt ioengine "$FIO_ENGINE"
    _b3_rt direct "$FIO_DIRECT"
    for job in "${B3_JOBS[@]}"; do
        _b3_fio_args "$job"
        _b3_cmdline fio "${FIO_ARGS[@]}" --output-format=json --output="$RAW/fio_$job.json"
        _b3_rt "cmd.$job" "$CMDLINE"
        _b3_rt "method.$job" fio
        log "fio $job: rw=$J_RW bs=$J_BS iodepth=$J_QD ioengine=$FIO_ENGINE direct=$FIO_DIRECT, ${RUN_S} s"
        rc=0
        _b3_run_fio $((RUN_S * 3 + 120)) "$RAW/fio_$job.json" "${FIO_ARGS[@]}" || rc=$?
        if [ "$rc" != 0 ]; then
            warn "fio $job failed (rc $rc): $(_b3_tail1 "$RAW/fio_$job.stderr")"
            if [ -e "$RAW/fio_$job.json" ]; then mv -f -- "$RAW/fio_$job.json" "$RAW/failed_fio_$job.json"; fi
            _b3_skip_job "$job" "fio job $job failed (rc $rc): $(_b3_tail1 "$RAW/fio_$job.stderr")"
        fi
    done
}

_b3_storage_dd() { # REASON: sequential 1 MiB QD1 read with dd (fio missing / unusable)
    local reason=$1 count t rc method=dd-direct out="$RAW/dd_seq_read_qd1.txt"
    _b3_rt fallback "dd: $reason"
    _b3_skip_job seq_read_qd8 "$reason; dd measures QD1 sequential reads only"
    _b3_skip_job rand_read_3mb_qd1 "$reason; no random-read measurement without fio"
    _b3_skip_job rand_read_3mb_qd4 "$reason; no random-read measurement without fio"
    _b3_skip_job rand_read_16k_qd1 "$reason; no latency/IOPS measurement without fio"
    if ! have_cmd dd; then
        _b3_skip_job seq_read_qd1 "$reason and dd is missing"
        warn "dd is missing too: no storage measurement"
        return 0
    fi
    count=$((FILE_BYTES / MIB))
    if [ "$count" -gt "$SIZE_MB" ]; then count=$SIZE_MB; fi
    if [ "$count" -lt 16 ]; then
        _b3_skip_job seq_read_qd1 "file too small for a dd read ($FILE_BYTES bytes)"
        return 0
    fi
    t=$((count / 2 + 120))
    _b3_cmdline dd if="$FILE" of=/dev/null bs=1M count="$count" iflag=direct
    log "dd fallback: sequential read of $count MiB with O_DIRECT ($FILE)"
    rc=0
    _b3_io "$t" env LC_ALL=C dd if="$FILE" of=/dev/null bs=1M count="$count" iflag=direct 2>"$out" || rc=$?
    if [ "$rc" != 0 ]; then
        warn "dd iflag=direct failed (rc $rc: $(_b3_tail1 "$out")): buffered read after dropping this file's page cache (posix_fadvise DONTNEED, this file only)"
        env LC_ALL=C dd if="$FILE" iflag=nocache count=0 2>/dev/null || true
        _b3_cmdline dd if="$FILE" of=/dev/null bs=1M count="$count" iflag=nocache
        method=dd-buffered-cold
        rc=0
        _b3_io "$t" env LC_ALL=C dd if="$FILE" of=/dev/null bs=1M count="$count" iflag=nocache 2>"$out" || rc=$?
    fi
    _b3_rt cmd.seq_read_qd1 "$CMDLINE"
    _b3_rt method.seq_read_qd1 "$method"
    if [ "$rc" != 0 ]; then
        if [ -e "$out" ]; then mv -f -- "$out" "$RAW/failed_dd_seq_read_qd1.txt"; fi
        _b3_skip_job seq_read_qd1 "dd read failed (rc $rc)"
        warn "dd read failed (rc $rc): no storage measurement"
    fi
}

_b3_storage() {
    if [ "$SKIP_IO" = 1 ]; then
        local j
        for j in "${B3_JOBS[@]}"; do _b3_skip_job "$j" "--skip-io"; done
        return 0
    fi
    if [ "$FIO_OK" = 1 ]; then
        _b3_storage_fio
    else
        _b3_storage_dd "fio is not installed (sudo apt-get install -y fio)"
    fi
}

_b3_memory() {
    local bin avail_kb need_kb rc=0
    if [ "$SKIP_MEM" = 1 ]; then
        _b3_rt skip.memory "--skip-mem"
        return 0
    fi
    bin=$(tools_bin membw)
    if [ ! -x "$bin" ]; then
        _b3_rt skip.memory "membw not built ($bin); build it with: bash pi/02_build_llama.sh --with-tools"
        warn "membw not found at $bin: memory bandwidth NOT measured (build it with: bash pi/02_build_llama.sh --with-tools; PIGEMMA_TOOLS_BIN overrides the directory)"
        return 0
    fi
    avail_kb=$(mem_available_kb)
    need_kb=$(((MEMBW_MB + 768) * 1024))
    if [[ $avail_kb =~ ^[0-9]+$ ]] && [ "$avail_kb" -lt "$need_kb" ]; then
        _b3_rt skip.memory "MemAvailable $((avail_kb / 1024)) MiB < membw ${MEMBW_MB} MiB + 768 MiB headroom"
        warn "not enough available RAM for membw (${MEMBW_MB} MiB + 768 MiB headroom; MemAvailable $((avail_kb / 1024)) MiB): memory bandwidth NOT measured"
        return 0
    fi
    _b3_cmdline "$bin" --size-mb "$MEMBW_MB" --threads "$OPT_THREADS" --reps "$MEMBW_REPS" --json
    _b3_rt cmd.membw "$CMDLINE"
    log "membw: ${MEMBW_MB} MiB, threads $OPT_THREADS, best of $MEMBW_REPS"
    _b3_cpu 900 "$bin" --size-mb "$MEMBW_MB" --threads "$OPT_THREADS" --reps "$MEMBW_REPS" --json \
        >"$RAW/membw.json" 2>"$RAW/membw.stderr" || rc=$?
    if [ "$rc" != 0 ]; then
        if [ -e "$RAW/membw.json" ]; then mv -f -- "$RAW/membw.json" "$RAW/failed_membw.json"; fi
        _b3_rt skip.memory "membw failed (rc $rc): $(_b3_tail1 "$RAW/membw.stderr")"
        warn "membw failed (rc $rc): $(_b3_tail1 "$RAW/membw.stderr")"
    fi
}

_b3_conditions() { # PREFIX -> loadavg / temperature / throttle / governor lines in run.tsv
    local p=$1
    _b3_rt "$p.loadavg1" "$(loadavg1)"
    _b3_rt "$p.mem_available_kb" "$(mem_available_kb)"
    _b3_rt "$p.governor" "$(cpu_governor)"
    if declare -F _pg_cpu_temp >/dev/null; then
        _pg_cpu_temp
        _b3_rt "$p.cpu_temp_c" "$_R"
    fi
    if declare -F _pg_throttled >/dev/null; then
        _pg_throttled
        _b3_rt "$p.throttled" "$_R"
    fi
    _b3_rt "$p.at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

_b3_check_lock() { # -> die 4 when results/.bench.lock is held (a model-loading job runs)
    local rc=0 out
    if is_fake; then
        return 0
    fi
    out=$(_b3_py -m pigemma.paths lock --status 2>/dev/null) || rc=$?
    case $rc in
        0) log "bench lock: free" ;;
        1)
            if [ "$DRY_RUN" = 1 ]; then
                warn "bench lock is HELD (a model-loading job is running): a real run would refuse (exit 4)"
                DRY_FAIL=4
                return 0
            fi
            die 4 "the bench lock (results/.bench.lock) is held by a model-loading job: storage and memory numbers would be distorted. Wait until it finishes. $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-300)"
            ;;
        *) warn "could not read the bench lock status (python -m pigemma.paths lock --status rc $rc): continuing" ;;
    esac
}

# ------------------------------------------------------------------------------------------ assembler
_b3_assemble() { # RAW_DIR FACTS_FILE -> io_mem.json at $OUT; returns the assembler's exit code
    local raw=$1 facts=$2 emu=0
    if is_emulated; then emu=1; fi
    _b3_py - --raw "$(native_path "$raw")" --out "$(native_path "$OUT")" --facts "$(native_path "$facts")" \
        --catalog "$(native_path "$PIGEMMA_CATALOG")" --calibration "$(native_path "$CAL_PATH")" \
        --home "$(native_path "$PIGEMMA_HOME")" --device "$OPT_DEVICE" --version "$B3_VERSION" \
        --emulated "$emu" --override "${PIGEMMA_PROTECTED_OVERRIDE:-0}" <<'PYEOF'
"""pi/03_bench_io_mem.sh assembler: raw fio / dd / membw outputs -> io_mem.json + the printed report."""
import argparse
import datetime as _dt
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

GEN = "pi/03_bench_io_mem.sh"
JOB_METRIC = {
    "seq_read_qd1": "seq_read_mbps_qd1",
    "seq_read_qd8": "seq_read_mbps_qd8",
    "rand_read_3mb_qd1": "rand_read_3mb_mbps_qd1",
    "rand_read_3mb_qd4": "rand_read_3mb_mbps_qd4",
    "rand_read_16k_qd1": "rand_read_16k_iops",
}
STORAGE_KEYS = ("seq_read_mbps_qd1", "seq_read_mbps_qd8", "rand_read_3mb_mbps_qd1", "rand_read_3mb_mbps_qd4",
                "rand_read_16k_iops", "access_ms")
DD_RE = re.compile(r"(\d+)\s+bytes\b.*?copied,\s*([0-9]+(?:[.,][0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*s", re.S)


def warn(msg):
    print(f"[pigemma 03_bench_io_mem.sh assembler] WARNING: {msg}", file=sys.stderr)


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rnd(x, n):
    return None if x is None else round(float(x), n)


def jload(path):
    """First JSON object in a file (fio may print notes before the JSON)."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    i = text.find("{")
    if i < 0:
        raise ValueError(f"{path}: no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(text[i:])
    return obj


def read_tsv(path):
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                out[k.strip()] = v.strip()
    return out


def rel(path, home):
    try:
        return Path(path).resolve().relative_to(Path(home).resolve()).as_posix()
    except (ValueError, OSError):
        return str(path).replace("\\", "/")


def utc_iso(ts):
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


# ------------------------------------------------------------------------------------------------ fio
def fio_jobs(raw, warnings):
    found = {}
    for p in sorted(raw.glob("*.json")):
        if p.name.startswith("failed_"):
            continue
        try:
            obj = jload(p)
        except Exception as e:  # noqa: BLE001 - any unreadable file is reported and skipped
            if p.name.startswith("fio"):
                warnings.append(f"{p.name}: not parsable as fio JSON ({e})")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("jobs"), list):
            continue
        gopts = obj.get("global options") or {}
        for job in obj["jobs"]:
            name = job.get("jobname")
            if name not in JOB_METRIC:
                continue
            opts = dict(gopts)
            opts.update(job.get("job options") or {})
            found[name] = {"job": job, "opts": opts, "file": p.name, "version": obj.get("fio version"),
                           "timestamp": num(obj.get("timestamp"))}
    return found


def fio_stats(ent):
    job = ent["job"]
    r = job.get("read") or {}
    err = int(num(job.get("error")) or 0)
    bw = num(r.get("bw_bytes"))
    if bw is None and num(r.get("bw")) is not None:
        bw = num(r.get("bw")) * 1024.0
    lat_ms = None
    lat = r.get("lat_ns") or {}
    if num(lat.get("mean")) is not None:
        lat_ms = num(lat.get("mean")) / 1e6
    elif num((r.get("lat") or {}).get("mean")) is not None:  # fio 2.x reports usec
        lat_ms = num(r["lat"]["mean"]) / 1e3
    clat = r.get("clat_ns") or {}
    pct = clat.get("percentile") or {}

    def p(key):
        v = num(pct.get(key))
        return None if v is None else v / 1e6

    io_bytes = num(r.get("io_bytes"))
    runtime_ms = num(r.get("runtime"))
    ok = err == 0 and bool(bw) and bw > 0 and bool(io_bytes) and io_bytes > 0
    return {
        "ok": ok, "error": err, "bw_bytes": bw, "iops": num(r.get("iops")), "lat_ms_mean": lat_ms,
        "clat_ms_mean": None if num(clat.get("mean")) is None else num(clat.get("mean")) / 1e6,
        "clat_ms_p50": p("50.000000"), "clat_ms_p99": p("99.000000"),
        "io_bytes": io_bytes, "runtime_s": None if runtime_ms is None else runtime_ms / 1e3,
    }


def fio_cmd(name, ent, tsv):
    cmd = tsv.get(f"cmd.{name}")
    if cmd:
        return cmd
    parts = " ".join(f"--{k}={v}" for k, v in ent["opts"].items())
    return f"fio --name={name} {parts} --output-format=json (reconstructed from the fio JSON job options)"


def fio_flags(ent):
    flags = []
    o = ent["opts"]
    if str(o.get("direct", "1")) in ("0", "false"):
        flags.append("buffered: O_DIRECT unavailable; fio invalidated the file's page cache before the job but "
                     "re-reads inside the job may hit the cache (upper bound)")
    eng = str(o.get("ioengine", ""))
    qd = int(num(o.get("iodepth")) or 1)
    if qd > 1 and eng in ("psync", "sync", "vsync", "pvsync"):
        flags.append(f"ioengine {eng} is synchronous: iodepth {qd} not honoured (effectively QD1)")
    return flags


# -------------------------------------------------------------------------------------------- devices
def auto_device(facts, catalog, feas):
    """(device name, reason) from the facts: catalog pi5 device with the same RAM tier + storage class."""
    kb = num(facts.get("ram_total_kb"))
    ram = kb * 1024.0 if kb else None
    cg = num(facts.get("cgroup_mem_max"))
    if ram and cg and 0 < cg < ram:
        ram = cg
    if feas is not None:
        try:
            st_type, cls, _notes = feas.storage_class_for(facts.get("storage"), catalog)
        except Exception:  # noqa: BLE001
            st_type, cls = "unknown", "unknown"
        tier = feas.nominal_ram_gb(ram) if ram else None
    else:
        st_type = str((facts.get("storage") or {}).get("type") or "unknown")
        cls = {"microsd": "microsd", "usb-ssd": "usb-ssd", "nvme": "nvme-gen2"}.get(st_type, "unknown")
        tier = None
        if ram:
            for sku in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                if ram <= sku * (1 << 30) * 1.02:
                    tier = sku
                    break
    board = str(facts.get("board_model") or "")
    kernel = str(facts.get("kernel") or "").lower()
    # Pi 5 = BCM2712: device-tree model, or the Raspberry Pi kernel flavour (6.x+rpt-rpi-2712), or emulated
    is_pi = ("raspberry pi 5" in board.lower() or "rpi-2712" in kernel or "bcm2712" in kernel
             or facts.get("emulated") is True)
    if not board and "rpi-2712" in kernel:
        board = f"Raspberry Pi 5 (kernel {facts.get('kernel')})"
    if not ram:
        return f"unknown-ram-{cls}", "auto: RAM unknown in the facts"
    if is_pi and tier:
        for d in catalog.get("devices", []) or []:
            if d.get("platform") != "pi5":
                continue
            dram = num(d.get("ram_bytes")) or 0
            dtier = feas.nominal_ram_gb(dram) if feas is not None else None
            if dtier == tier and str((d.get("storage") or {}).get("class") or "") == cls:
                return d["name"], (f"auto: catalog device with the same nominal RAM tier ({tier} GB) and storage "
                                   f"class ({cls})")
        why = "emulated Pi" if facts.get("emulated") is True and "raspberry" not in board.lower() else board
        return f"pi5-{tier}gb-{cls}", (f"auto: {why}, nominal RAM {tier} GB, storage class {cls}; no catalog device "
                                       "with that tier + class, pigemma.feasibility synthesizes the profile from the name")
    gb = int(round(ram / 1e9))
    return f"linux-{gb}gb-{cls}", (f"auto: not a Raspberry Pi 5 (board model {board or 'unknown'}): RAM {gb} GB, "
                                   f"storage class {cls}")


def catalog_device(catalog, name):
    for d in catalog.get("devices", []) or []:
        if isinstance(d, dict) and d.get("name") == name:
            return d
    return None


def metric_val(x):
    if isinstance(x, dict):
        return num(x.get("value"))
    return num(x)


def src_kind(x):
    s = str(x.get("source", "")) if isinstance(x, dict) else ""
    return s.split(":", 1)[0].strip().lower() or "?"


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(prog="pi/03_bench_io_mem.sh assembler")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--version", default="?")
    ap.add_argument("--emulated", default="0")
    ap.add_argument("--override", default="0")
    a = ap.parse_args()

    raw, out, home = Path(a.raw), Path(a.out), Path(a.home)
    logging.basicConfig(level=logging.WARNING, format="[pigemma 03_bench_io_mem.sh assembler] %(levelname)s: %(message)s")
    sys.path.insert(0, str(home))
    try:
        from pigemma import feasibility as feas  # stdlib-only module, runs on the Pi's Python
    except Exception as e:  # noqa: BLE001
        feas = None
        warn(f"pigemma.feasibility not importable ({e}): device auto-naming falls back, merge check skipped")

    warnings = []
    tsv = read_tsv(raw / "run.tsv")
    try:
        catalog = json.loads(Path(a.catalog).read_text(encoding="utf-8-sig"))
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: cannot read the catalog {a.catalog}: {e}", file=sys.stderr)
        return 1
    facts_path = raw / "facts.json" if (raw / "facts.json").is_file() else Path(a.facts)
    try:
        facts = json.loads(facts_path.read_text(encoding="utf-8-sig"))
    except Exception as e:  # noqa: BLE001
        warn(f"facts not readable ({facts_path}: {e}); host details unknown")
        facts = {}

    # ---- storage
    storage, skipped, flags, details = {}, {}, [], {}
    jobs = fio_jobs(raw, warnings)
    fio_version = None
    stamps = []
    for name, metric in JOB_METRIC.items():
        ent = jobs.get(name)
        if ent is None:
            continue
        st = fio_stats(ent)
        fio_version = fio_version or ent.get("version")
        if ent.get("timestamp"):
            stamps.append(ent["timestamp"])
        if not st["ok"]:
            skipped[metric] = f"fio job {name} reported error {st['error']} or no data ({ent['file']})"
            continue
        o = ent["opts"]
        detail = {
            "rw": o.get("rw"), "bs": o.get("bs"), "iodepth": o.get("iodepth"), "ioengine": o.get("ioengine"),
            "direct": o.get("direct"), "iops": rnd(st["iops"], 2), "lat_ms_mean": rnd(st["lat_ms_mean"], 4),
            "clat_ms_p50": rnd(st["clat_ms_p50"], 4), "clat_ms_p99": rnd(st["clat_ms_p99"], 4),
            "io_bytes": None if st["io_bytes"] is None else int(st["io_bytes"]),
            "runtime_s": rnd(st["runtime_s"], 3), "fio_version": ent.get("version"), "raw_file": ent["file"],
        }
        entry_flags = fio_flags(ent)
        base = {"source": "measured", "cmd": fio_cmd(name, ent, tsv),
                "method": tsv.get(f"method.{name}", "fio"), "job": name}
        if name == "rand_read_16k_qd1":
            storage["rand_read_16k_iops"] = dict(value=rnd(st["iops"], 2), unit="IOPS", **base, detail=detail)
            if entry_flags:
                storage["rand_read_16k_iops"]["flags"] = entry_flags
            details["lat16k"] = st["lat_ms_mean"]
            details["lat16k_detail"] = detail
            details["lat16k_base"] = base
            details["lat16k_flags"] = entry_flags
        else:
            storage[metric] = dict(value=rnd(st["bw_bytes"] / 1e6, 3), unit="MB/s", **base, detail=detail)
            if entry_flags:
                storage[metric]["flags"] = entry_flags
    if details.get("lat16k") is not None:
        lat = details["lat16k"]
        seq = metric_val(storage.get("seq_read_mbps_qd1")) or metric_val(storage.get("seq_read_mbps_qd8"))
        xfer = 16384 / (seq * 1e6) * 1e3 if seq else None
        storage["access_ms"] = {
            "value": rnd(lat, 4), "unit": "ms", "source": "measured",
            "cmd": ("derived: mean total latency (fio lat_ns.mean = submission + completion) of job "
                    "rand_read_16k_qd1 (16 KiB random read, QD1); includes the 16 KiB transfer. Job: "
                    + details["lat16k_base"]["cmd"]),
            "method": "derived", "job": "rand_read_16k_qd1", "derived_from": "rand_read_16k_qd1",
            "detail": {"lat_ms_mean": rnd(lat, 4), "clat_ms_p50": details["lat16k_detail"]["clat_ms_p50"],
                       "clat_ms_p99": details["lat16k_detail"]["clat_ms_p99"],
                       "transfer_ms_16k_at_seq": rnd(xfer, 4),
                       "access_ms_minus_transfer": rnd(max(0.0, lat - xfer), 4) if xfer is not None else None},
        }
        if details["lat16k_flags"]:
            storage["access_ms"]["flags"] = details["lat16k_flags"]

    dd_file = raw / "dd_seq_read_qd1.txt"
    if "seq_read_mbps_qd1" not in storage and dd_file.is_file():
        text = dd_file.read_text(encoding="utf-8", errors="replace")
        m = None
        for m in DD_RE.finditer(text):
            pass
        if m:
            nbytes = int(m.group(1))
            secs = num(m.group(2).replace(",", "."))
            if nbytes > 0 and secs and secs > 0:
                method = tsv.get("method.seq_read_qd1", "dd-direct")
                fl = [("dd fallback (fio not installed or unusable): sequential 1 MiB reads at QD1 only; "
                       "random-read, QD>1 and latency metrics were not measured")]
                if method != "dd-direct":
                    fl.append("buffered: dd iflag=direct failed; read after dropping this file's page cache "
                              "(kernel readahead active)")
                storage["seq_read_mbps_qd1"] = {
                    "value": rnd(nbytes / secs / 1e6, 3), "unit": "MB/s", "source": "measured",
                    "cmd": tsv.get("cmd.seq_read_qd1", f"dd bs=1M count={nbytes // 1048576} iflag=direct"),
                    "method": method, "job": "seq_read_qd1",
                    "detail": {"bytes": nbytes, "seconds": secs, "bs": "1M", "iodepth": 1, "raw_file": dd_file.name},
                    "flags": fl,
                }
                flags.append(fl[0])
        else:
            warnings.append(f"{dd_file.name}: no 'N bytes ... copied, S s' line")
    for k, v in tsv.items():
        if k.startswith("skip.") and k[5:] in STORAGE_KEYS and k[5:] not in storage:
            skipped[k[5:]] = v
    for k in STORAGE_KEYS:
        if k not in storage and k not in skipped:
            skipped[k] = "not measured (no raw output found)"
    if tsv.get("fallback"):
        flags.append(f"storage fallback: {tsv['fallback']}")
    for m, e in storage.items():
        for f in e.get("flags", []):
            if f.startswith("buffered") and f not in flags:
                flags.append(f"{m}: {f}")

    # ---- memory
    memory = None
    mb = raw / "membw.json"
    if mb.is_file():
        try:
            mj = jload(mb)
            res = [r for r in mj.get("results", []) if isinstance(r, dict) and num(r.get("threads"))]
            if not res:
                raise ValueError("no results[]")
            byt = {str(int(num(r["threads"]))): rnd(num(r.get("read_gbps")), 3) for r in res}
            cpy = {str(int(num(r["threads"]))): rnd(num(r.get("copy_gbps")), 3) for r in res}
            tri = {str(int(num(r["threads"]))): rnd(num(r.get("triad_gbps")), 3) for r in res}
            best = lambda d: max((v for v in d.values() if v is not None), default=None)  # noqa: E731
            memory = {
                "read_gbps_by_threads": byt, "copy_gbps": best(cpy), "triad_gbps": best(tri),
                "copy_gbps_by_threads": cpy, "triad_gbps_by_threads": tri,
                "unit": "GB/s (1e9 bytes/s), best of reps; copy/triad = best over thread counts",
                "source": "measured", "cmd": tsv.get("cmd.membw", "membw --json (command not recorded)"),
                "method": "membw", "membw_version": mj.get("version"), "size_mb": mj.get("size_mb"),
                "reps": mj.get("reps"), "page_size": mj.get("page_size"), "arch": mj.get("host"),
                "kernel": mj.get("release"), "bytes_counted": mj.get("bytes_counted"), "raw_file": mb.name,
            }
            if best(byt) is None:
                raise ValueError("no read_gbps values")
        except Exception as e:  # noqa: BLE001
            memory = None
            skipped["memory"] = f"membw.json not usable: {e}"
    if memory is None and "memory" not in skipped:
        skipped["memory"] = tsv.get("skip.memory", "not measured (no membw.json)")

    if not storage and memory is None:
        print("ERROR: nothing was measured (no usable fio, dd or membw output in "
              f"{raw}); io_mem.json NOT written. Reasons: {json.dumps(skipped)}", file=sys.stderr)
        return 3

    # ---- identity
    host = str(facts.get("hostname") or tsv.get("hostname") or "unknown")
    if a.device:
        device, dev_reason = a.device, "explicit (--device / PIGEMMA_DEVICE)"
        if catalog_device(catalog, device) is None:
            dev_reason += "; not in the catalog: pigemma.feasibility synthesizes a profile from the name"
    else:
        device, dev_reason = auto_device(facts, catalog, feas)
    emulated = a.emulated == "1" or facts.get("emulated") is True or tsv.get("emulated") == "1"
    override = a.override == "1" or tsv.get("override") == "1" or facts.get("override") == "PIGEMMA_I_KNOW"
    if tsv.get("finished_at"):
        collected = tsv["finished_at"]
    elif stamps:
        collected = utc_iso(max(stamps))
    else:
        collected = str(facts.get("collected_at") or utc_iso(_dt.datetime.now(_dt.timezone.utc).timestamp()))
    complete = all(k in storage for k in STORAGE_KEYS) and memory is not None

    notes = []
    if emulated:
        notes.append("EMULATED (PIGEMMA_EMULATED=1, qemu-user arm64 container): these are NOT Raspberry Pi "
                     "measurements; storage/memory speeds are the desktop's through the emulator")
    if facts.get("in_container") is True and not emulated:
        notes.append("measured inside a container: storage is the container's view (overlay / bind mount)")
    if override:
        notes.append("measured on a protected host with the authorised PIGEMMA_I_KNOW=1 override "
                     "(nice/ionice, services untouched)")
    sto = facts.get("storage") or {}
    for n in sto.get("notes") or []:
        notes.append(f"storage detection: {n}")
    if str(sto.get("fstype") or "").lower() in ("tmpfs", "ramfs", "devtmpfs", "zram"):
        fl = (f"RAM-backed filesystem ({sto.get('fstype')}): the storage numbers measure memory, not a disk; "
              "do not use them for streaming projections")
        flags.append(fl)
        notes.append(fl)
    if fio_version is None and any(storage.get(k, {}).get("method", "").startswith("dd") for k in storage):
        notes.append("fio is not installed: install it (sudo apt-get install -y fio, needs the host owner's "
                     "approval on context-friend-1) and re-run for the random-read and latency metrics")

    test_file = None
    if tsv.get("file"):
        test_file = {"path": rel(tsv["file"], home), "mode": tsv.get("file_mode"),
                     "bytes": int(num(tsv.get("file_bytes")) or 0) or None,
                     "deleted_afterwards": tsv.get("file_mode") in ("new", "created-and-deleted")}
    settings = {
        "seconds_per_job": int(num(tsv.get("seconds")) or 0) or None,
        "size_mb": int(num(tsv.get("size_mb")) or 0) or None,
        "slab_unit_bytes": int(num(tsv.get("slab_unit_bytes")) or 0) or None,
        "slab_unit_source": tsv.get("slab_source"),
        "slab_fio_bs_bytes": int(num(tsv.get("slab_bs")) or 0) or None,
        "page_size": int(num(tsv.get("page_size")) or 0) or (facts.get("page_size") or None),
        "ioengine": tsv.get("ioengine"), "direct": tsv.get("direct"), "fio_version": fio_version,
        "nice": tsv.get("nice"), "ionice": tsv.get("ionice"), "quick": tsv.get("quick") == "1",
        "membw_threads": tsv.get("membw_threads"), "membw_size_mb": tsv.get("membw_size_mb"),
        "membw_reps": tsv.get("membw_reps"),
    }
    conditions = {}
    for ph in ("before", "after"):
        blk = {k.split(".", 1)[1]: v for k, v in tsv.items() if k.startswith(ph + ".")}
        if blk:
            conditions[ph] = blk

    # ---- derived (projected)
    sel = catalog.get("selection") or {}
    viable = num(sel.get("viable_stream_tok_s")) or 1.0
    good = num(sel.get("good_stream_tok_s")) or 2.0
    dev_before = catalog_device(catalog, device)
    cls_metrics = {}
    if dev_before is None:
        classes = catalog.get("storage_classes") or {}
        cls = device.split("gb-", 1)[1] if "gb-" in device else "unknown"
        cls_metrics = dict(classes.get(cls) or classes.get("unknown") or {})
        cls_metrics["mem_bw_gbps"] = {
            "value": float(getattr(feas, "DEFAULT_MEM_BW_GBPS", 10.0)) if feas is not None else 10.0,
            "source": "assumed: pigemma.feasibility default for a synthesized profile (Pi 5 ~10 GB/s)"}
    eff_src = None
    eff = None
    cal_path = Path(a.calibration)
    if cal_path.is_file():
        try:
            cal = json.loads(cal_path.read_text(encoding="utf-8-sig"))
            ent = cal.get("llama_efficiency")
            v = metric_val(ent)
            if v is not None and 0.05 <= v <= 1.5:
                eff = v
                eff_src = (ent.get("source") if isinstance(ent, dict) else None) or f"measured: {cal_path.name}"
        except Exception as e:  # noqa: BLE001
            warnings.append(f"{cal_path}: not usable ({e})")
    if eff is None:
        eff = float(getattr(feas, "LLAMA_EFFICIENCY", 0.65)) if feas is not None else 0.65
        eff_src = (f"assumed: pigemma.feasibility LLAMA_EFFICIENCY ({eff}) until pi/05 writes "
                   f"{rel(cal_path, home)} (tg x bytes/token / membw)")
    if memory is not None:
        bw_thr, bw = max(((t, v) for t, v in memory["read_gbps_by_threads"].items() if v is not None),
                         key=lambda tv: tv[1])
        bw_src = f"measured: memory.read_gbps_by_threads[{bw_thr}] (best thread count)"
    else:
        old = ((dev_before or {}).get("mem_bw_gbps") or cls_metrics.get("mem_bw_gbps")
               or {"value": 10.0, "source": "assumed: Pi 5 ~10 GB/s"})
        bw, bw_src = metric_val(old), str(old.get("source", "assumed"))

    def pick_storage(key, catkey):
        for k in key:
            if k in storage:
                return metric_val(storage[k]), f"measured: {k}"
        old = ((dev_before or {}).get("storage") or {}).get(catkey) or cls_metrics.get(catkey) or {}
        return metric_val(old), str(old.get("source", "assumed: catalog")) if isinstance(old, dict) else "assumed"

    rand, rand_src = pick_storage(("rand_read_3mb_mbps_qd1", "rand_read_3mb_mbps_qd4"), "rand_read_3mb_mbps")
    acc, acc_src = pick_storage(("access_ms",), "access_ms")
    models_out = {}
    stream_out = {}
    for m in catalog.get("models", []) or []:
        if m.get("kind") == "draft":
            continue
        ne = num(m.get("nonexpert_bytes_per_token") or m.get("nonexpert_bytes")) or 0.0
        ex = num(m.get("expert_bytes_per_token_worst")) or 0.0
        bpt = ne + ex
        if bpt <= 0:
            continue
        models_out[m["id"]] = {"kind": m.get("kind"), "bytes_per_token": int(bpt),
                               "ceiling_tok_s": rnd(bw * 1e9 * eff / bpt, 3) if bw else None}
        n_miss = int((num(m.get("n_moe_layers")) or 0) * (num(m.get("n_expert_used")) or 0))
        if ex > 0 and n_miss > 0 and rand and acc is not None:
            unit = ex / n_miss
            per_miss_s = unit / (rand * 1e6) + acc / 1e3
            t_all = n_miss * per_miss_s
            row = {"expert_bytes_per_token_worst": int(ex), "n_miss_per_token_worst": n_miss,
                   "unit_bytes_mean": int(unit), "t_io_all_miss_s": rnd(t_all, 4),
                   "tok_s_all_miss": rnd(1.0 / t_all, 4), "max_miss_mb_per_token": {}, "min_hit_rate": {}}
            for target in (viable, good):
                mmax = min(float(n_miss), (1.0 / target) / per_miss_s)
                row["max_miss_mb_per_token"][str(target)] = rnd(mmax * unit / 1e6, 1)
                row["min_hit_rate"][str(target)] = rnd(max(0.0, 1.0 - mmax / n_miss), 4)
            stream_out[m["id"]] = row
    derived = {
        "kind": "projected",
        "note": ("arithmetic on the measured numbers above, not measurements. Ceiling = mem_bw_read x "
                 "llama_efficiency / bytes_per_token (bytes_per_token = nonexpert_bytes_per_token + "
                 "expert_bytes_per_token_worst from the catalog; KV-cache reads ignored). Streaming uses the "
                 "pigemma.feasibility time model t_io = miss_bytes / rand_read_3mb + n_miss x access_ms and "
                 "ignores compute time, so speeds are optimistic and hit rates are lower bounds."),
        "mem_bw_gbps": {"value": bw, "source": bw_src},
        "llama_efficiency": {"value": eff, "source": eff_src},
        "rand_read_3mb_mbps": {"value": rand, "source": rand_src},
        "access_ms": {"value": acc, "source": acc_src},
        "thresholds_tok_s": {"viable": viable, "good": good, "source": "catalog.selection"},
        "in_ram_ceiling": models_out,
        "streaming": stream_out,
    }

    doc = {
        "format_version": 1,
        "generator": f"{GEN} {a.version}",
        "kind": "measured",
        "host": host,
        "device": device,
        "device_source": dev_reason,
        "collected_at": collected,
        "started_at": tsv.get("started_at"),
        "emulated": bool(emulated),
        "protected_override": bool(override),
        "complete": bool(complete),
        "storage": {k: storage[k] for k in STORAGE_KEYS if k in storage},
    }
    if memory is not None:
        doc["memory"] = memory
    doc.update({
        "skipped": skipped,
        "flags": flags,
        "test_file": test_file,
        "settings": settings,
        "conditions": conditions,
        "host_facts": facts,
        "derived": derived,
        "raw_dir": rel(raw, home),
        "notes": notes,
    })
    if warnings:
        doc["warnings"] = warnings
        for w in warnings:
            warn(w)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(sanitize(doc), indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, out)

    # ---- merge check through the real loader
    merged_dev, merge_err = None, None
    if feas is not None:
        try:
            merged = feas.load_catalog(Path(a.catalog), io_mem=out)
            merged_dev = catalog_device(merged, device)
            if merged_dev is None:
                merge_err = f"device {device} missing after the merge"
            else:
                bad = []
                if any(k in storage for k in ("seq_read_mbps_qd1", "seq_read_mbps_qd8")) and \
                        not str(merged_dev["storage"]["seq_read_mbps"].get("source", "")).startswith("measured"):
                    bad.append("storage.seq_read_mbps")
                if any(k in storage for k in ("rand_read_3mb_mbps_qd1", "rand_read_3mb_mbps_qd4")) and \
                        not str(merged_dev["storage"]["rand_read_3mb_mbps"].get("source", "")).startswith("measured"):
                    bad.append("storage.rand_read_3mb_mbps")
                if "access_ms" in storage and \
                        not str(merged_dev["storage"]["access_ms"].get("source", "")).startswith("measured"):
                    bad.append("storage.access_ms")
                if memory is not None and not str(merged_dev["mem_bw_gbps"].get("source", "")).startswith("measured"):
                    bad.append("mem_bw_gbps")
                if bad:
                    merge_err = "fields not marked measured after the merge: " + ", ".join(bad)
        except Exception as e:  # noqa: BLE001
            merge_err = f"pigemma.feasibility.load_catalog(io_mem=...) failed: {e}"

    # ---- report (stdout)
    P = print

    def f(v, nd=2):
        return "n/a" if v is None else f"{v:.{nd}f}"

    P(f"== pi/03_bench_io_mem: MEASURED storage + memory constants ==")
    P(f"host {host} | device {device} ({dev_reason})")
    P(f"collected_at {collected}{' | EMULATED: not Pi numbers' if emulated else ''}"
      f"{' | PIGEMMA_I_KNOW override' if override else ''}")
    if test_file:
        sz = test_file.get("bytes")
        P(f"file {test_file['path']} ({f(sz / 1e9 if sz else None)} GB, {test_file['mode']}) on "
          f"{sto.get('dev') or '?'} ({sto.get('fstype') or '?'}, storage type {sto.get('type') or '?'})")
    if fio_version:
        P(f"method {fio_version}: ioengine {settings['ioengine'] or '?'}, direct {settings['direct'] or '?'}, "
          f"{settings['seconds_per_job'] or '?'} s per job; expert slab {settings['slab_unit_bytes'] or '?'} B -> "
          f"fio bs {settings['slab_fio_bs_bytes'] or '?'} B (page {settings['page_size'] or '?'})")
    elif any(str(e.get("method", "")).startswith("dd") for e in storage.values()):
        P("method dd fallback (fio not installed or unusable): sequential 1 MiB reads at QD1 only")
    P("")
    P(f"  {'STORAGE (MEASURED)':<26} {'value':>10}  {'unit':<5} detail")
    for k in STORAGE_KEYS:
        e = storage.get(k)
        if e is None:
            P(f"  {k:<26} {'-':>10}  {'':<5} not measured: {skipped.get(k, '?')}")
            continue
        d = e.get("detail") or {}
        if k == "access_ms":
            det = (f"= mean latency of the 16 KiB QD1 random read (p99 {f(d.get('clat_ms_p99'), 3)} ms; "
                   f"minus transfer {f(d.get('access_ms_minus_transfer'), 3)} ms)")
            val = f(e["value"], 4)
        elif e.get("method", "").startswith("dd"):
            det = f"{e['method']}, bs 1M, QD1 ({d.get('bytes')} B in {f(d.get('seconds'), 2)} s)"
            val = f(e["value"], 2)
        else:
            det = (f"bs {d.get('bs')} qd{d.get('iodepth')} {d.get('ioengine')}, lat {f(d.get('lat_ms_mean'), 3)} ms "
                   f"mean, clat p99 {f(d.get('clat_ms_p99'), 3)} ms")
            val = f(e["value"], 1 if k == "rand_read_16k_iops" else 2)
        if e.get("flags"):
            det += " [FLAGGED: " + "; ".join(x.split(":")[0] for x in e["flags"]) + "]"
        P(f"  {k:<26} {val:>10}  {e['unit']:<5} {det}")
    P("")
    if memory is not None:
        P(f"  MEMORY (MEASURED, membw {memory.get('size_mb')} MiB, best of {memory.get('reps')}; GB/s = 1e9 B/s)")
        P(f"  {'threads':>9} {'read':>9} {'copy':>9} {'triad':>9}")
        for t in sorted(memory["read_gbps_by_threads"], key=lambda s: int(s)):
            P(f"  {t:>9} {f(memory['read_gbps_by_threads'][t]):>9} {f(memory['copy_gbps_by_threads'].get(t)):>9} "
              f"{f(memory['triad_gbps_by_threads'].get(t)):>9}")
    else:
        P(f"  MEMORY: not measured ({skipped.get('memory')})")
    P("")
    P(f"  CATALOG DEVICE {device}: before (catalog file) -> after merge (pigemma.feasibility.load_catalog)")
    if merged_dev is not None:
        for key, label in (("seq_read_mbps", "storage.seq_read_mbps"), ("rand_read_3mb_mbps", "storage.rand_read_3mb_mbps"),
                           ("access_ms", "storage.access_ms"), ("mem_bw_gbps", "mem_bw_gbps")):
            if key == "mem_bw_gbps":
                old = (dev_before or {}).get("mem_bw_gbps") or cls_metrics.get("mem_bw_gbps")
                new = merged_dev.get("mem_bw_gbps")
            else:
                old = ((dev_before or {}).get("storage") or {}).get(key) or cls_metrics.get(key)
                new = (merged_dev.get("storage") or {}).get(key)
            P(f"    {label:<28} {f(metric_val(old), 3):>9} ({src_kind(old)})  ->  {f(metric_val(new), 3):>9} "
              f"({src_kind(new)})")
        if dev_before is None:
            P(f"    ('before' = catalog storage class defaults: {device} is not in the catalog and is synthesized)")
    elif merge_err is None:
        P("    merge check skipped (pigemma.feasibility not importable)")
    if merge_err:
        P(f"    MERGE CHECK FAILED: {merge_err}")
    P("")
    P("  DERIVED (PROJECTED: arithmetic on the numbers above, not measurements)")
    P(f"    in-RAM decode ceiling = mem_bw {f(bw)} GB/s [{src_kind({'source': bw_src})}] x llama_efficiency "
      f"{f(eff)} [{src_kind({'source': eff_src})}] / bytes per token")
    P(f"    {'model':<18} {'kind':<6} {'GB/token':>9} {'ceiling tok/s':>14}")
    for mid, r in models_out.items():
        P(f"    {mid:<18} {str(r['kind']):<6} {f(r['bytes_per_token'] / 1e9, 3):>9} {f(r['ceiling_tok_s'], 2):>14}")
    if stream_out:
        P(f"    expert streaming from storage: rand 3 MB = {f(rand, 1)} MB/s [{src_kind({'source': rand_src})}], "
          f"access {f(acc, 3)} ms [{src_kind({'source': acc_src})}]; compute time ignored")
        P(f"    {'model':<18} {'expert MB/tok':>13} {'all-miss tok/s':>15} "
          f"{'miss MB/tok @' + str(viable) + '/' + str(good):>22} {'min hit @' + str(viable) + '/' + str(good):>18}")
        for mid, r in stream_out.items():
            mm = r["max_miss_mb_per_token"]
            hr = r["min_hit_rate"]
            P(f"    {mid:<18} {f(r['expert_bytes_per_token_worst'] / 1e6, 1):>13} {f(r['tok_s_all_miss'], 3):>15} "
              f"{f(mm[str(viable)], 1) + ' / ' + f(mm[str(good)], 1):>22} "
              f"{f(hr[str(viable)], 3) + ' / ' + f(hr[str(good)], 3):>18}")
    elif rand is None:
        P("    expert streaming: no random-read number (measured or catalog) -> not computed")
    P("")
    if skipped:
        P("  NOT MEASURED (catalog keeps the ASSUMED value): " + ", ".join(sorted(skipped)))
    if not complete:
        P("  PARTIAL RESULT: complete=false (see 'skipped' and 'flags' in the JSON)")
    for n in notes:
        P(f"  note: {n}")
    P(f"io_mem.json: {rel(out, home)}")
    P(f"raw outputs: {rel(raw, home)}")
    if merge_err:
        print(f"ERROR: {merge_err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
PYEOF
}

# ------------------------------------------------------------------------------------------ modes
_b3_print_plan() {
    local job
    printf 'plan: host %s, out %s\n' "$(host_name)" "$OUT"
    printf 'plan: file %s (%s, %s bytes) in %s\n' "$FILE" "$FILE_MODE" "$FILE_BYTES" "$MEASURE_DIR"
    printf 'plan: expert slab %s B (%s) -> fio bs %s B (page %s)\n' "$SLAB_UNIT" "$SLAB_SOURCE" "$SLAB_BS" "$PAGE"
    if [ "$SKIP_IO" = 1 ]; then
        printf 'plan: storage skipped (--skip-io)\n'
    elif [ "$FIO_OK" = 1 ]; then
        if [ "$FILE_MODE" = new ]; then
            _b3_layout_args "$OPT_ENGINE" 1
            _b3_cmdline fio "${FIO_ARGS[@]}"
            printf 'plan: %s\n' "$CMDLINE"
        fi
        _b3_probe_args "$OPT_ENGINE" 1
        _b3_cmdline fio "${FIO_ARGS[@]}"
        printf 'plan: %s   (engine/direct probe; falls back to psync / buffered)\n' "$CMDLINE"
        for job in "${B3_JOBS[@]}"; do
            _b3_fio_args "$job"
            _b3_cmdline fio "${FIO_ARGS[@]}" --output-format=json
            printf 'plan: %s\n' "$CMDLINE"
        done
    else
        printf 'plan: fio NOT installed -> dd fallback (sequential read only): dd if=%s of=/dev/null bs=1M iflag=direct\n' "$FILE"
    fi
    if [ "$SKIP_MEM" = 1 ]; then
        printf 'plan: memory skipped (--skip-mem)\n'
    else
        printf 'plan: %s --size-mb %s --threads %s --reps %s --json%s\n' "$(tools_bin membw)" "$MEMBW_MB" \
            "$OPT_THREADS" "$MEMBW_REPS" "$([ -x "$(tools_bin membw)" ] || printf ' (NOT BUILT: would be skipped)')"
    fi
}

mode_check_jobs() {
    local job rc=0 fails=0
    need_cmd fio fio "--check-jobs runs fio --parse-only"
    for job in layout probe "${B3_JOBS[@]}"; do
        case $job in
            layout) _b3_layout_args "$OPT_ENGINE" 1 ;;
            probe) _b3_probe_args "$OPT_ENGINE" 1 ;;
            *) _b3_fio_args "$job" ;;
        esac
        _b3_cmdline fio --parse-only "${FIO_ARGS[@]}" --output-format=json
        rc=0
        fio --parse-only "${FIO_ARGS[@]}" --output-format=json >"${TMPDIR:-/tmp}/.pigemma-fio-parse.$$" 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then
            printf 'OK   %-18s %s\n' "$job" "$CMDLINE"
        else
            printf 'FAIL %-18s %s\n' "$job" "$CMDLINE"
            sed 's/^/     /' "${TMPDIR:-/tmp}/.pigemma-fio-parse.$$" || true
            fails=$((fails + 1))
        fi
    done
    rm -f "${TMPDIR:-/tmp}/.pigemma-fio-parse.$$"
    if [ "$fails" -gt 0 ]; then
        die 1 "$fails fio job string(s) failed fio --parse-only"
    fi
    log "all fio job strings parse ($(fio --version 2>/dev/null || echo fio))"
}

mode_from_raw() {
    local raw facts_tmp="" facts rc=0
    raw=$(_b3_abs "$FROM_RAW")
    [ -d "$raw" ] || die 3 "--from-raw: $FROM_RAW is not a directory"
    _pg_need_python
    _b3_out_paths
    if [ -f "$raw/facts.json" ]; then
        facts="$raw/facts.json"
    else
        mkdir -p -- "$OUT_DIR" || die 1 "cannot create $OUT_DIR"
        facts_tmp="$OUT_DIR/.io_mem_facts.$$.json"
        add_cleanup "rm -f -- '$facts_tmp'"
        log "no facts.json in $raw: collecting this host's facts for the host block"
        MODEL_DIR="${OPT_DIR:-$MODEL_DIR}" collect_facts >"$facts_tmp" || die 1 "collect_facts failed"
        facts=$facts_tmp
    fi
    _b3_assemble "$raw" "$facts" || rc=$?
    return "$rc"
}

mode_measure() {
    local stamp rc=0 why
    DRY_FAIL=0
    assert_safe_host
    if [ "$BUSY_CHECK" = 1 ]; then
        if [ "$DRY_RUN" = 1 ]; then
            log "dry run: busy check skipped (a real run calls assert_not_busy)"
        else
            assert_not_busy
        fi
    fi
    _pg_need_python
    _b3_check_lock
    _b3_out_paths
    _b3_slab_bytes
    _b3_resolve_file
    FIO_OK=0
    if have_cmd fio; then
        FIO_OK=1
    elif [ "$SKIP_IO" = 0 ]; then
        if [ "$REQUIRE_FIO" = 1 ]; then
            die 3 "fio is not installed (--require-fio). Install it yourself with: sudo apt-get install -y fio (pigemma never installs packages; on context-friend-1 this needs the host owner's approval)"
        fi
        warn "fio is not installed: sudo apt-get install -y fio (not done automatically). Falling back to dd: sequential read only; random-read and latency metrics stay ASSUMED"
    fi
    FIO_ENGINE=$OPT_ENGINE
    FIO_DIRECT=1

    # preconditions
    if [ "$SKIP_IO" = 0 ]; then
        if [ "$FILE_MODE" = new ]; then
            if _b3_disk_check; then
                log "free disk OK for the test file: $DISK_MSG"
            else
                why="not enough free disk for the ${SIZE_MB} MiB test file: $DISK_MSG. Use --quick / --size-mb N, or --file <an existing GGUF> (read-only, no writes)"
                if [ "$DRY_RUN" = 1 ]; then
                    warn "$why"
                    DRY_FAIL=3
                else
                    die 3 "$why"
                fi
            fi
        else
            if [ "$FILE_BYTES" -lt $((64 * MIB)) ]; then
                why="$FILE has only $FILE_BYTES bytes: too small for random expert-slab reads (need >= 64 MiB)"
                if [ "$DRY_RUN" = 1 ]; then warn "$why"; DRY_FAIL=3; else die 3 "$why"; fi
            elif [ "$FILE_BYTES" -lt 1073741824 ]; then
                warn "$FILE is smaller than 1 GiB: drive caches may inflate the random-read numbers"
            fi
            log "measuring read-only on $FILE ($(fmt_gb "$FILE_BYTES")), no writes"
        fi
    fi

    if [ "$DRY_RUN" = 1 ]; then
        _b3_print_plan
        if [ "$DRY_FAIL" != 0 ]; then
            die "$DRY_FAIL" "dry run: a real run would stop here (see the warnings above); nothing was run or written"
        fi
        log "dry run: nothing was run or written"
        return 0
    fi
    if is_fake; then
        die 1 "PIGEMMA_FAKE_PROC is set (test fixture root): refusing a real measurement; use --dry-run or --from-raw"
    fi
    case $(uname -s 2>/dev/null || echo unknown) in
        Linux) ;;
        *) die 1 "pi/03 measures Linux block storage and memory: run it on the Pi (or in Docker via pc/docker_run.sh), not on $(uname -s 2>/dev/null || echo this OS)" ;;
    esac

    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    RAW="$OUT_DIR/raw/io_mem-$stamp"
    if [ -e "$RAW" ]; then RAW="$RAW-$$"; fi
    mkdir -p -- "$RAW" || die 1 "cannot create $RAW"
    : >"$RAW/run.tsv"
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    log "raw outputs: $RAW"

    if [ "$SKIP_IO" = 0 ] && [ "$FILE_MODE" = new ]; then
        mkdir -p -- "$MEASURE_DIR" || die 3 "cannot create $MEASURE_DIR"
    fi
    MODEL_DIR="$MEASURE_DIR" collect_facts >"$RAW/facts.json" || die 1 "collect_facts failed"
    _b3_rt generator "pi/03_bench_io_mem.sh $B3_VERSION"
    _b3_rt started_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    _b3_rt hostname "$(host_name)"
    _b3_rt file "$FILE"
    _b3_rt file_mode "$FILE_MODE"
    _b3_rt file_bytes "$FILE_BYTES"
    _b3_rt size_mb "$SIZE_MB"
    _b3_rt seconds "$RUN_S"
    _b3_rt quick "$OPT_QUICK"
    _b3_rt slab_unit_bytes "$SLAB_UNIT"
    _b3_rt slab_source "$SLAB_SOURCE"
    _b3_rt slab_bs "$SLAB_BS"
    _b3_rt page_size "$PAGE"
    _b3_rt nice "${NICE_LEVEL:-10}"
    # resolve ionice support once here (run_nice_io caches it), so its warning reaches the terminal instead of
    # the first fio/dd job's redirected .stderr file
    run_nice_io true || true
    if [ "${_PG_IONICE_OK:-0}" = 1 ]; then
        _b3_rt ionice "idle (-c3) for fio/dd"
    else
        _b3_rt ionice "not available (nice only)"
    fi
    _b3_rt membw_threads "$OPT_THREADS"
    _b3_rt membw_size_mb "$MEMBW_MB"
    _b3_rt membw_reps "$MEMBW_REPS"
    if is_emulated; then _b3_rt emulated 1; else _b3_rt emulated 0; fi
    _b3_rt override "${PIGEMMA_PROTECTED_OVERRIDE:-0}"
    _b3_conditions before

    _b3_memory
    if [ "$SKIP_IO" = 0 ] && [ "$FILE_MODE" = new ]; then
        _b3_create_testfile
        _b3_rt file_mode created-and-deleted
        _b3_rt file_bytes "$FILE_BYTES"
    fi
    _b3_storage
    _b3_remove_testfile
    _b3_conditions after
    _b3_rt finished_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    find "$RAW" -maxdepth 1 -type f -name '*.stderr' -size 0 -delete 2>/dev/null || true

    _b3_assemble "$RAW" "$RAW/facts.json" || rc=$?
    case $rc in
        0) log "done: $OUT (next: bash pi/00_feasibility.sh picks it up automatically)" ;;
        3) die 3 "nothing could be measured: io_mem.json was not written (raw outputs kept in $RAW)" ;;
        *) die "$rc" "assembling $OUT failed (exit $rc); raw outputs kept in $RAW" ;;
    esac
}

main() {
    parse_args "$@"
    pigemma_init
    if [ -n "$FROM_RAW" ]; then
        mode_from_raw
        return
    fi
    if [ "$CHECK_JOBS" = 1 ]; then
        _pg_need_python
        _b3_slab_bytes
        _b3_resolve_file
        FIO_ENGINE=$OPT_ENGINE
        FIO_DIRECT=1
        mode_check_jobs
        return
    fi
    mode_measure
}

main "$@"
