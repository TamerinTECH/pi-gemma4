#!/usr/bin/env bash
# =============================================================================
# pc/trace_collect.sh - E3 driver: MoE expert-routing traces of the 26B on the
# desktop, in Docker (pi-gemma4).
#
# Runs `pigemma-probe --mode moe-trace` (tools/probe/probe.cpp) in the Linux build
# image through pc/docker_run.sh, once per corpus file of data/calibration plus
# once for data/prompts/gen_prompts.json, then validates every trace, writes its
# <trace>.meta.json, merges them into <out>/all.moetrace and computes
# <out>/all.stats.json (+ results/h1/all.*.csv) with pigemma.trace.
# Spec: docs/design/manifest.json files[pc/trace_collect.sh]; experiment
# E3-routing-trace (docs/EXPERIMENT_PLAN.md, section E3).
#
#   bash pc/trace_collect.sh [--smoke|--quick|--full] [--gen N] [--threads 12]
#                            [--probs-top K] [--out traces/26b] [options]
#
# Tiers:
#   --smoke   first 64 tokens of en_tech_moe_streaming.txt + 8 generated tokens
#             (validation only: 72 records, all 30 MoE layers). No merge, no stats.
#   --quick   (default) every data/calibration text/code file, chunk 1024, --gen 128
#             per chunk, plus the 8 gen_prompts.json prompts with --chat --gen 256.
#             Today's 7 corpus files tokenise to 10,101 tokens (MEASURED with
#             llama-tokenize, vocab only, 2026-10-05) -> 13 chunks -> <= 1,664 generated,
#             + ~0.4k prompt prefill and <= 2,048 generated for the prompts: the
#             >= 10,000 prefill check passes by a small margin only.
#   --full    as --quick with chunk 2048 and --gen 256 per chunk.
#
# One probe process per corpus item (every chunk starts from an empty KV cache
# inside the probe). Each run is exactly (Git Bash, from the repo root):
#   bash pc/docker_run.sh --no-stdin --name pigemma-trace-<id>-<pid> -- \
#     python -m pigemma.paths lock --hold -- tools/build-linux-x86/bin/pigemma-probe \
#     --mode moe-trace -m /work/<26B gguf> -f /work/data/calibration/<file> \
#     -nr -t 12 -c 4096 -b 2048 -ub 256 --chunk-tokens 1024 --gen 128 \
#     [--probs-top K] --corpus-id <stem> --trace-out /work/traces/26b/<stem>.moetrace
#   (gen prompts: --gen-prompts /work/data/prompts/gen_prompts.json --chat ... --gen 256)
# -nr keeps the whole GGUF CPU_Mapped, so the 31 GB Docker VM holds the 14.4 GB file
# in its page cache instead of a 14 GB CPU_REPACK copy per run. The first read of the
# GGUF through the /work bind mount takes minutes.
# After each run (host Python):
#   python -m pigemma.trace validate <trace>
#   python -m pigemma.trace meta <trace> --gguf <model> --corpus <file> --summary-file <probe log>
#          --host-json <container fingerprint> --wall-ms <probe wall_ms> --coherent auto
# Then (quick/full):
#   python -m pigemma.trace merge <out>/all.moetrace <the per-item traces of this run>
#   python -m pigemma.trace validate <out>/all.moetrace
#   python -m pigemma.trace stats <out>/all.moetrace --out <out>/all.stats.json --csv-dir results/h1
#   python -m pigemma.results write --out results/h1/ --fields <E3 envelope>
#
# Decisions beyond the manifest text (documented here; also reported as open issues):
#  D1  Bench lock: the host takes results/.bench.lock ONCE for the whole tier by
#      re-running itself under `python -m pigemma.paths lock --hold --min-free-gb 20`.
#      The per-run `lock --hold` inside the container (the spec command) then finds
#      the lock "inherited" (pc/docker_run.sh forwards PIGEMMA_LOCK_HELD and maps
#      PIGEMMA_LOCK_FILE), so no other model-loading job can start between two runs.
#      A held lock refuses with exit 4 before anything starts.
#  D2  Merge inputs are listed explicitly (the per-item traces of this run), not
#      `traces/26b/*.moetrace`: a glob would also take an older all.moetrace or a
#      leftover per-file trace (docs/EXPERIMENT_PLAN.md E3 says the same).
#  D3  The smoke trace is written to <out>/smoke/en_tech_moe_streaming.moetrace so it
#      never mixes with the quick/full traces. It uses the golden fixture's command
#      (--chunk-tokens 64 --gen 8 --tokens 72, plus --max-chunks 1). It is never
#      copied into tests/fixtures automatically; the copy command is printed.
#  D4  Corpus = regular, non-empty files in data/calibration with a text/code suffix
#      (txt c cc cpp cxx h hh hpp py js ts rs go java sh), minus README*, all.* (the
#      pigemma.quality corpus concatenation: it would duplicate every file) and hidden
#      files, sorted in C locale. Corpus id = file stem; colliding or reserved stems
#      (all, smoke, the prompts stem) use the file name with '.' -> '_'.
#  D5  <trace>.meta.json comes from `python -m pigemma.trace meta` on the host:
#      host = the container's fingerprint (captured once per tier in a preflight
#      container), gguf sha256 prefix read natively from the host file, wall_ms = the
#      probe's own PIGEMMA_SUMMARY wall_ms (else the host-measured docker wall time),
#      coherent = pigemma.quality.is_coherent per gen.txt chunk.
#  D6  Acceptance checks print CHECK ... PASS|FAIL lines. quick/full: every run exit 0
#      and valid, merged trace valid, >= 10,000 prefill and >= 2,000 generated
#      records, every chunk with generated text coherent, all MoE layers present
#      (= the E3 "Trace VALID" rule). smoke: valid, 72 records = 64 prefill + 8 gen,
#      all layers. A failed check exits 5; the traces, stats and envelope stay written.
#  D7  quick/full also write one E3 results envelope (results/h1/E3_*.json, kind
#      measured, variant moe-trace-<tier>) with the token-weighted probe tok/s (pp/tg),
#      coherent, and the per-run numbers, checks and E3 preview in extra, so
#      `python -m pigemma.results summarize results/h1 --experiment E3` shows the run.
#      --no-envelope disables it; --smoke never writes one.
#  D8  Before the runs, the outputs that this run will regenerate are deleted (each
#      re-run item's trace + .tokens.bin/.gen.txt/.meta.json, and all.moetrace +
#      sidecars + all.stats.json), so a failed run never leaves a stale file that
#      looks current. Nothing else is deleted.
#  D9  -c is raised (rounded up to 256) when chunk + gen + 8 tokens do not fit; -b
#      smaller than the chunk only logs a note (the probe sub-batches, its D-d).
#  D10 Busy check: refuses (exit 4) while another llama-* / pigemma-probe process or
#      container runs (--ignore-busy overrides). Background CPU over 5 s is recorded
#      in the envelope but not gated: the probe tok/s are a by-product of E3, not its
#      measurement (E0 owns engine speed).
#  D11 Logs (gitignored, results/**/*.log): <log-dir>/trace_collect_<UTC>_<tier>.log
#      (driver), ..._<id>.probe.log (full llama.cpp stderr + PIGEMMA_SUMMARY per run)
#      and ..._<tier>.runs.log (run ledger, JSON lines). The console shows only the
#      probe's own lines and errors (--verbose shows everything).
#  D12 --resume reuses a per-item trace that validates, is not truncated, has a
#      meta.json, and matches the corpus sha256, --chunk-tokens/--gen/--probs-top and
#      the GGUF size; everything else is re-run.
#  D14 --model-volume NAME (env PIGEMMA_TRACE_MODEL_VOLUME): the probe reads the GGUF from
#      the Docker volume NAME mounted read-only at /models (/models/<basename of --model>)
#      instead of the /work bind mount. MEASURED 2026-10-05 (E3): the Docker Desktop VM
#      does NOT keep a bind-mounted file in its page cache between containers (two
#      consecutive containers mmap-touching the same 256 MiB: 31.1 and 30.8 MB/s), and the
#      -nr mmap load reads the 14.4 GB file with 4 KiB faults at 16-31 MB/s = 8-15 min
#      PER probe run; an ext4 volume copy (pc/pagecache_limit.sh step 1, volume
#      pigemma-models) loads from the VM page cache after the first run. The host-side
#      size check and meta (sha256 prefix, size) still use the host file of --model; the
#      preflight checks that the volume copy has the same size.
#  D13 Test hooks: PIGEMMA_TRACE_DOCKER_RUN replaces pc/docker_run.sh (same CLI);
#      PIGEMMA_TRACE_PROBE replaces the container-side probe path;
#      PIGEMMA_TRACE_BGCPU_S sets the background-CPU sample length (default 5, 0 = off).
#
# Exit codes (pigemma.paths conventions + 5):
#   0  all runs succeeded and every acceptance check passed (also --help/--dry-run)
#   1  a probe run, validate, meta, merge, stats or envelope step failed
#   2  usage error (bad option, path outside the repo, dense model)
#   3  missing input: 26B GGUF (or wrong size), probe binary, corpus/prompts file,
#      Python >= 3.10 with numpy, docker image
#   4  refused: bench lock held / low RAM, another llama job running, low disk,
#      docker unavailable. stderr line 'trace_collect: REFUSED: ...'
#   5  traces written but an acceptance check failed (see the CHECK lines)
#   130 interrupted (the probe finishes its current llama_process and closes its files)
# stdout: the plan (--dry-run) or the final report (per-run table, CHECK lines,
# E3 preview, full-tier sizing, outputs). Progress and logs go to stderr.
#
# Requires bash >= 4.4 (Git Bash 5.2 on the desktop, or Linux) and a host Python
# >= 3.10 with numpy (python / python3 / PIGEMMA_PYTHON). Lint: bash -n (shellcheck
# is not installed on this desktop).
# =============================================================================
set -euo pipefail
shopt -u patsub_replacement 2>/dev/null || true

readonly PROG=trace_collect
readonly VERSION=1.0.0
readonly EXIT_OK=0 EXIT_ERROR=1 EXIT_USAGE=2 EXIT_MISSING=3 EXIT_REFUSED=4 EXIT_CHECKS=5
readonly CONTAINER_ROOT=/work
readonly DEFAULT_MODEL=26b-q4
readonly DEFAULT_PROBE=tools/build-linux-x86/bin/pigemma-probe
readonly CORPUS_DIR=data/calibration
readonly DEFAULT_PROMPTS=data/prompts/gen_prompts.json
readonly SMOKE_FILE=en_tech_moe_streaming.txt
readonly SMOKE_PREFILL=64
readonly GOLDEN=tests/fixtures/golden_26b_72tok.moetrace
readonly MIN_PREFILL=10000 MIN_GEN=2000
readonly FULL_CHUNK=2048 FULL_GEN=256
readonly MIN_FREE_DISK_BYTES=1000000000
readonly CORPUS_EXTS=" txt c cc cpp cxx h hh hpp py js ts rs go java sh "

# Keep Git Bash (MSYS) from rewriting /work/... arguments of docker.exe / python.exe.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

UNAME_S=$(uname -s 2>/dev/null || echo unknown)
IS_MSYS=0
case $UNAME_S in MINGW*|MSYS*|CYGWIN*) IS_MSYS=1 ;; esac

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
LOG_FILE=""
log() {
    local m
    m="[$PROG $(date +%H:%M:%S)] $*"
    printf '%s\n' "$m" >&2
    if [ -n "$LOG_FILE" ]; then printf '%s\n' "$m" >>"$LOG_FILE" 2>/dev/null || true; fi
}
warn() { log "WARNING: $*"; }
QUIET_GATES=0
gate() { if [ "$QUIET_GATES" = 1 ]; then return 0; fi; log "GATE: $*"; }
die() {
    local rc=$1
    shift
    log "ERROR (exit $rc): $*"
    exit "$rc"
}
refuse() {
    printf '%s: REFUSED: %s\n' "$PROG" "$*" >&2
    if [ -n "$LOG_FILE" ]; then printf '%s: REFUSED: %s\n' "$PROG" "$*" >>"$LOG_FILE" 2>/dev/null || true; fi
    exit "$EXIT_REFUSED"
}
is_uint() { [[ ${1:-} =~ ^[0-9]+$ ]]; }
quote_cmd() {
    local a out=""
    for a in "$@"; do out+="$(printf '%q' "$a") "; done
    printf '%s' "${out% }"
}
now_ms() {
    local t
    t=$(date +%s%3N 2>/dev/null) || t=""
    if ! is_uint "$t"; then t=$(( $(date +%s) * 1000 )); fi
    printf '%s' "$t"
}

# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------
usage() {
    cat <<'USAGE'
pc/trace_collect.sh - E3: MoE routing traces of the 26B (pigemma-probe --mode moe-trace in Docker)

Usage:
  bash pc/trace_collect.sh [--smoke|--quick|--full] [--gen N] [--threads 12]
                           [--probs-top K] [--out traces/26b] [options]

Tiers (default --quick):
  --smoke          first 64 tokens of data/calibration/en_tech_moe_streaming.txt + --gen 8;
                   validation only (72 records, all MoE layers); writes <out>/smoke/
  --quick          every data/calibration text/code file (chunk 1024, --gen 128 per chunk)
                   + data/prompts/gen_prompts.json (--chat --gen 256); merge + stats
  --full           as --quick with chunk 2048 and --gen 256 per chunk

Options:
  --gen N          generated tokens per corpus chunk (smoke 8, quick 128, full 256)
  --prompt-gen N   generated tokens per gen prompt (default 256)
  --threads N      probe -t (default 12)
  --probs-top K    also record the K most probable experts per layer (default 0 = off)
  --out DIR        trace directory inside the repo (default traces/26b)
  --model ID|PATH  MoE GGUF inside the repo (default 26b-q4)
  --model-volume V the probe reads /models/<basename of --model> from Docker volume V (ro)
                   instead of the bind mount (env PIGEMMA_TRACE_MODEL_VOLUME; see D14)
  --chunk-tokens N override the tier's chunk size (smoke 64, quick 1024, full 2048)
  --ctx N          probe -c (default 4096; raised when chunk + gen + 8 does not fit)
  --batch N        probe -b (default 2048)
  --ubatch N       probe -ub (default 256)
  --files LIST     comma list of corpus files (name or stem) instead of all of them
  --prompts FILE   generation prompts (default data/prompts/gen_prompts.json)
  --no-prompts     skip the generation prompts
  --probe-arg ARG  extra pigemma-probe argument, repeatable (e.g. --probe-arg=--chat-style
                   --probe-arg=gemma4)
  --csv-dir DIR    stats CSV directory (default results/h1)
  --results-dir D  E3 envelope directory (default results/h1)
  --no-envelope    do not write the E3 results envelope
  --log-dir DIR    logs (default results/h1/raw; *.log is gitignored)
  --resume         reuse per-item traces that validate and match the settings
  --keep-going     continue after a failed probe run (the failed item is not merged)
  --no-merge       skip merge + stats + envelope (quick/full)
  --min-free-gb N  bench-lock free-RAM gate in GB (default 20)
  --no-lock        do not take the host bench lock (each probe run still uses lock --hold)
  --ignore-busy    run even while another llama-* / pigemma-probe process runs
  --verbose        show the full llama.cpp output of every run
  --dry-run        print the gates and the exact commands; run and write nothing
  -h, --help       this help
  --version        print the version

Writes (quick/full, default dirs): traces/26b/<stem>.moetrace (+ .tokens.bin, .gen.txt,
.meta.json) per corpus file and gen_prompts.moetrace; traces/26b/all.moetrace (+ sidecars);
traces/26b/all.stats.json; results/h1/all.*.csv; results/h1/E3_*.json;
results/h1/raw/trace_collect_<UTC>_*.log.

Exit codes: 0 ok | 1 a run/step failed | 2 usage | 3 missing input (GGUF, probe, corpus,
Python+numpy, image) | 4 refused (bench lock, busy, disk, docker) | 5 traces written but an
acceptance check failed | 130 interrupted.

Examples:
  bash pc/docker_run.sh build-tools
  bash pc/trace_collect.sh --smoke
  bash pc/trace_collect.sh --quick
  bash pc/trace_collect.sh --quick --dry-run
  bash pc/trace_collect.sh --full --resume
USAGE
}

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
ORIG_ARGS=("$@")
TIER="" GEN="" PROMPT_GEN=256 THREADS=12 PROBS_TOP=0 OUT=traces/26b MODEL_SPEC=$DEFAULT_MODEL
CHUNK="" CTX=4096 BATCH=2048 UBATCH=256 FILES="" PROMPTS=$DEFAULT_PROMPTS NO_PROMPTS=0
CSV_DIR=results/h1 RESULTS_DIR=results/h1 LOG_DIR=results/h1/raw NO_ENVELOPE=0
RESUME=0 KEEP_GOING=0 NO_MERGE=0 MIN_FREE_GB=20 NO_LOCK=0 IGNORE_BUSY=0 VERBOSE=0 DRY_RUN=0
PROBE_EXTRA=()
MODEL_VOLUME=${PIGEMMA_TRACE_MODEL_VOLUME:-}

need_val() {
    if [ "$2" -lt 2 ]; then die "$EXIT_USAGE" "option $1 needs a value (see --help)"; fi
}
set_tier() {
    if [ -n "$TIER" ] && [ "$TIER" != "$1" ]; then
        die "$EXIT_USAGE" "choose one tier: --$TIER and --$1 were both given"
    fi
    TIER=$1
}

while [ $# -gt 0 ]; do
    case $1 in
        --smoke) set_tier smoke; shift ;;
        --quick) set_tier quick; shift ;;
        --full)  set_tier full; shift ;;
        --gen)            need_val "$1" $#; GEN=$2; shift 2 ;;
        --gen=*)          GEN=${1#*=}; shift ;;
        --prompt-gen)     need_val "$1" $#; PROMPT_GEN=$2; shift 2 ;;
        --prompt-gen=*)   PROMPT_GEN=${1#*=}; shift ;;
        --threads|-t)     need_val "$1" $#; THREADS=$2; shift 2 ;;
        --threads=*)      THREADS=${1#*=}; shift ;;
        --probs-top)      need_val "$1" $#; PROBS_TOP=$2; shift 2 ;;
        --probs-top=*)    PROBS_TOP=${1#*=}; shift ;;
        --out)            need_val "$1" $#; OUT=$2; shift 2 ;;
        --out=*)          OUT=${1#*=}; shift ;;
        --model|-m)       need_val "$1" $#; MODEL_SPEC=$2; shift 2 ;;
        --model=*)        MODEL_SPEC=${1#*=}; shift ;;
        --model-volume)   need_val "$1" $#; MODEL_VOLUME=$2; shift 2 ;;
        --model-volume=*) MODEL_VOLUME=${1#*=}; shift ;;
        --chunk-tokens)   need_val "$1" $#; CHUNK=$2; shift 2 ;;
        --chunk-tokens=*) CHUNK=${1#*=}; shift ;;
        --ctx|-c)         need_val "$1" $#; CTX=$2; shift 2 ;;
        --ctx=*)          CTX=${1#*=}; shift ;;
        --batch|-b)       need_val "$1" $#; BATCH=$2; shift 2 ;;
        --batch=*)        BATCH=${1#*=}; shift ;;
        --ubatch|-ub)     need_val "$1" $#; UBATCH=$2; shift 2 ;;
        --ubatch=*)       UBATCH=${1#*=}; shift ;;
        --files)          need_val "$1" $#; FILES=$2; shift 2 ;;
        --files=*)        FILES=${1#*=}; shift ;;
        --prompts)        need_val "$1" $#; PROMPTS=$2; shift 2 ;;
        --prompts=*)      PROMPTS=${1#*=}; shift ;;
        --no-prompts)     NO_PROMPTS=1; shift ;;
        --probe-arg)      need_val "$1" $#; PROBE_EXTRA+=("$2"); shift 2 ;;
        --probe-arg=*)    PROBE_EXTRA+=("${1#*=}"); shift ;;
        --csv-dir)        need_val "$1" $#; CSV_DIR=$2; shift 2 ;;
        --csv-dir=*)      CSV_DIR=${1#*=}; shift ;;
        --results-dir)    need_val "$1" $#; RESULTS_DIR=$2; shift 2 ;;
        --results-dir=*)  RESULTS_DIR=${1#*=}; shift ;;
        --no-envelope)    NO_ENVELOPE=1; shift ;;
        --log-dir)        need_val "$1" $#; LOG_DIR=$2; shift 2 ;;
        --log-dir=*)      LOG_DIR=${1#*=}; shift ;;
        --resume)         RESUME=1; shift ;;
        --keep-going)     KEEP_GOING=1; shift ;;
        --no-merge)       NO_MERGE=1; shift ;;
        --min-free-gb)    need_val "$1" $#; MIN_FREE_GB=$2; shift 2 ;;
        --min-free-gb=*)  MIN_FREE_GB=${1#*=}; shift ;;
        --no-lock)        NO_LOCK=1; shift ;;
        --ignore-busy)    IGNORE_BUSY=1; shift ;;
        --verbose)        VERBOSE=1; shift ;;
        --dry-run|-n)     DRY_RUN=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        --version)        printf 'pc/trace_collect.sh %s\n' "$VERSION"; exit 0 ;;
        *) die "$EXIT_USAGE" "unknown argument '$1' (see --help)" ;;
    esac
done
if [ -z "$TIER" ]; then TIER=quick; fi

# value checks
check_int() {  # check_int NAME VALUE MIN MAX
    if ! is_uint "$2" || [ "$2" -lt "$3" ] || [ "$2" -gt "$4" ]; then
        die "$EXIT_USAGE" "$1 '$2' must be an integer in $3..$4"
    fi
}
if [ -n "$GEN" ]; then check_int --gen "$GEN" 0 100000; fi
if [ -n "$CHUNK" ]; then check_int --chunk-tokens "$CHUNK" 2 1000000; fi
check_int --prompt-gen "$PROMPT_GEN" 0 100000
check_int --threads "$THREADS" 1 1024
check_int --probs-top "$PROBS_TOP" 0 256
check_int --ctx "$CTX" 256 10000000
check_int --batch "$BATCH" 1 10000000
check_int --ubatch "$UBATCH" 1 10000000
if ! [[ $MIN_FREE_GB =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    die "$EXIT_USAGE" "--min-free-gb '$MIN_FREE_GB' must be a non-negative number"
fi

case $TIER in
    smoke) : "${CHUNK:=$SMOKE_PREFILL}"; : "${GEN:=8}" ;;
    quick) : "${CHUNK:=1024}"; : "${GEN:=128}" ;;
    full)  : "${CHUNK:=$FULL_CHUNK}"; : "${GEN:=$FULL_GEN}" ;;
esac

# ---------------------------------------------------------------------------
# repository and path handling
# ---------------------------------------------------------------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) \
    || die "$EXIT_MISSING" "cannot resolve the directory of ${BASH_SOURCE[0]}"
REPO_POSIX=$(cd -- "$SCRIPT_DIR/.." && pwd) || die "$EXIT_MISSING" "cannot resolve the repository root"

# to_posix PATH - Windows drive forms (D:\x, D:/x) to the MSYS form on Git Bash
to_posix() {
    local p=$1 bs='\'
    p=${p//"$bs"/\/}
    case $p in
        [A-Za-z]:/*|[A-Za-z]:)
            if [ "$IS_MSYS" = 1 ] && command -v cygpath >/dev/null 2>&1; then
                p=$(cygpath -u -- "$p" 2>/dev/null) || return 1
            fi ;;
    esac
    printf '%s' "$p"
}
# The caller's working directory (a re-exec under the bench lock runs from the repo root).
ORIG_PWD=${TRACE_COLLECT_ORIG_PWD:-$PWD}
ORIG_PWD=$(to_posix "$ORIG_PWD") || ORIG_PWD=$PWD

# normpath /a/./b/../c -> /a/c (lexical, no symlink resolution)
normpath() {
    local IFS=/ seg
    local -a parts out=()
    read -r -a parts <<<"$1"
    for seg in "${parts[@]}"; do
        case $seg in
            ''|.) ;;
            ..) if [ "${#out[@]}" -gt 0 ]; then unset 'out[${#out[@]}-1]'; fi ;;
            *) out+=("$seg") ;;
        esac
    done
    printf '/%s' "${out[*]}"
}

# repo_rel PATH - repo-relative POSIX path (relative inputs are relative to the caller's
# cwd); fails when PATH is outside the repository (the container only sees /work).
repo_rel() {
    local p abs la lr
    p=$(to_posix "$1") || return 1
    case $p in
        /*) abs=$p ;;
        *) abs=$ORIG_PWD/$p ;;
    esac
    abs=$(normpath "$abs")
    la=$abs
    lr=$REPO_POSIX
    if [ "$IS_MSYS" = 1 ]; then la=${la,,}; lr=${lr,,}; fi
    if [ "$la" = "$lr" ]; then printf '.'; return 0; fi
    case $la in
        "$lr"/*) printf '%s' "${abs:$(( ${#REPO_POSIX} + 1 ))}"; return 0 ;;
    esac
    return 1
}
need_in_repo() {  # need_in_repo OPTNAME PATH -> prints the repo-relative path or dies 2
    local r
    r=$(repo_rel "$2") || die "$EXIT_USAGE" "$1 '$2' is outside the repository $REPO_POSIX (the container only sees /work)"
    printf '%s' "$r"
}
native_path() {  # path for the host Python (Windows form on Git Bash)
    if [ "$IS_MSYS" = 1 ] && command -v cygpath >/dev/null 2>&1; then
        cygpath -m -- "$1" 2>/dev/null || printf '%s' "$1"
    else
        printf '%s' "$1"
    fi
}
free_bytes() {  # free bytes of the filesystem holding PATH (nearest existing parent)
    local p=$1 v
    while [ ! -e "$p" ] && [ "$p" != . ] && [ "$p" != / ]; do p=$(dirname -- "$p"); done
    v=$(df -P -B1 -- "$p" 2>/dev/null | awk 'NR==2 {print $4}') || v=""
    if is_uint "$v"; then printf '%s' "$v"; else printf ''; fi
}
fmt_gb() {
    if is_uint "${1:-}"; then awk -v b="$1" 'BEGIN { printf "%.2f GB", b / 1e9 }'; else printf 'n/a'; fi
}

cd -- "$REPO_POSIX" || die "$EXIT_MISSING" "cannot cd to $REPO_POSIX"

OUT_REL=$(need_in_repo --out "$OUT")
CSV_DIR_REL=$(need_in_repo --csv-dir "$CSV_DIR")
RESULTS_DIR_REL=$(need_in_repo --results-dir "$RESULTS_DIR")
LOG_DIR_REL=$(need_in_repo --log-dir "$LOG_DIR")
PROMPTS_REL=$(need_in_repo --prompts "$PROMPTS")
DOCKER_RUN=${PIGEMMA_TRACE_DOCKER_RUN:-$SCRIPT_DIR/docker_run.sh}
if [ -n "${PIGEMMA_TRACE_DOCKER_RUN:-}" ]; then DOCKER_RUN=$(to_posix "$DOCKER_RUN"); fi
DOCKER_RUN_SHOW=pc/docker_run.sh
if [ -n "${PIGEMMA_TRACE_DOCKER_RUN:-}" ]; then DOCKER_RUN_SHOW=$DOCKER_RUN; fi

# model volume (D14): extra docker_run args + the model path the container sees
VOL_ARGS=()
if [ -n "$MODEL_VOLUME" ]; then
    [[ $MODEL_VOLUME =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die "$EXIT_USAGE" "--model-volume '$MODEL_VOLUME' is not a valid volume name"
    VOL_ARGS=(--volume "$MODEL_VOLUME:/models:ro")
fi
model_c() {
    if [ -n "$MODEL_VOLUME" ]; then printf '/models/%s' "${MODEL_REL##*/}"; else printf '%s/%s' "$CONTAINER_ROOT" "$MODEL_REL"; fi
}

# probe path as the container sees it (relative = under /work)
PROBE_C=${PIGEMMA_TRACE_PROBE:-$DEFAULT_PROBE}
PROBE_HOST=""
case $PROBE_C in
    "$CONTAINER_ROOT"/*) PROBE_HOST=${PROBE_C#"$CONTAINER_ROOT"/} ;;
    /*) PROBE_HOST="" ;;
    *) PROBE_HOST=$PROBE_C ;;
esac

# ---------------------------------------------------------------------------
# host Python and the embedded helper
# ---------------------------------------------------------------------------
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
pym() { PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "$PY_BIN" -m "$@"; }

IFS= read -r -d '' HELPER_PY <<'PY' || true
"""Host-side helper of pc/trace_collect.sh (ops: model, resume-ok, runrec, smoke, finalize, bgcpu)."""
import json
import math
import os
import shlex
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

EXIT_CHECKS = 5


def opts(argv):
    out = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("--") or i + 1 >= len(argv):
            raise SystemExit(f"trace_collect helper: bad argument {a!r}")
        out[a[2:].replace("-", "_")] = argv[i + 1]
        i += 2
    return out


def kv(**d):
    for k, v in d.items():
        print(f"{k}={shlex.quote('' if v is None else str(v))}")


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, f=".2f", none="n/a"):
    v = num(x)
    return none if v is None else format(v, f)


# ---------------------------------------------------------------------------- model
def op_model(o):
    from pigemma import paths
    spec = o["spec"]
    mid = spec if spec in paths.MODELS else None
    cat_hint = ""
    if mid is not None:
        p = paths.model_path(mid)
    else:
        p = Path(spec)
        cat = {}
        try:
            cat = json.loads(Path(paths.CATALOG_PATH).read_text(encoding="utf-8"))
        except Exception:
            pass
        for m in cat.get("models", []):
            if m.get("id") == spec:
                if m.get("local_path"):
                    p = Path(m["local_path"])
                else:
                    cat_hint = (f"catalog id {spec!r} has no local file: download "
                                f"https://huggingface.co/{m.get('repo')}/resolve/main/{m.get('file')} "
                                f"into models/ and pass --model <path>")
                kind = m.get("kind", "")
                if kind:
                    kv(MODEL_CATALOG_KIND=kind)
        mid = paths.model_id_for(p)
    try:
        rel = p.resolve().relative_to(paths.REPO_ROOT.resolve()).as_posix()
    except ValueError:
        kv(MODEL_OUTSIDE=str(p))
        return 2
    ent = paths.MODELS.get(mid) if mid else None
    exists = p.is_file()
    kv(MODEL_REL=rel, MODEL_ID=mid or "", MODEL_KIND=(ent or {}).get("kind", ""),
       MODEL_EXPECT_BYTES=(ent or {}).get("bytes", ""), MODEL_EXISTS=int(exists),
       MODEL_SIZE=p.stat().st_size if exists else "",
       MODEL_HINT=paths.download_hint(mid) if ent else cat_hint)
    return 0


# ---------------------------------------------------------------------------- resume
def _flag(flags, name, default=None):
    for i, a in enumerate(flags):
        if a == name and i + 1 < len(flags):
            return flags[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def op_resume_ok(o):
    from pigemma import trace as tr
    tp = Path(o["trace"])
    if not tp.is_file():
        print("no trace")
        return 1
    rep = tr.validate_report(tp)
    if not rep.ok or rep.truncated:
        print("trace does not validate" + (" (truncated)" if rep.truncated else ""))
        return 1
    h = tr.read_header(tp)
    flags = [str(x) for x in h.get("flags", [])]
    want = {"--gen": o["gen"], "--probs-top": o["probs"]}
    if o["kind"] != "prompts":
        want["--chunk-tokens"] = o["chunk"]
    for name, val in want.items():
        got = _flag(flags, name, "0" if name in ("--gen", "--probs-top") else "1024")
        if str(got) != str(val):
            print(f"{name} {got} != {val}")
            return 1
    if str(h.get("gguf_size_bytes")) != str(o["gguf_size"]):
        print(f"gguf_size_bytes {h.get('gguf_size_bytes')} != {o['gguf_size']}")
        return 1
    if str(h.get("corpus_id")) != o["cid"]:
        print(f"corpus_id {h.get('corpus_id')} != {o['cid']}")
        return 1
    if o["kind"] == "prompts" and "--chat" not in flags:
        print("prompts trace without --chat")
        return 1
    meta = tr.read_meta(tp)
    if not meta:
        print("no meta.json")
        return 1
    shas = [c.get("sha256") for c in meta.get("corpus", []) or []]
    if _sha256(o["corpus"]) not in shas:
        print("corpus file changed since the trace was made")
        return 1
    print(f"reusable ({rep.n_records} records)")
    return 0


# ---------------------------------------------------------------------------- runrec
def _trace_facts(tp):
    from pigemma import trace as tr
    out = {}
    if not tp.is_file():
        return out
    try:
        with tr.TraceReader(tp) as r:
            ph = r.phase
            out.update(n_records=int(r.n), n_prefill=int((ph == 0).sum()), n_gen=int((ph == 1).sum()),
                       truncated=bool(r.truncated), n_layer=int(r.header.get("n_layer", 0) or 0),
                       n_moe_layers=len(r.moe_layers), moe_layers=[int(x) for x in r.moe_layers],
                       chunks=list(r.header.get("chunks", []) or []),
                       record_size=int(r.record_size))
    except Exception as e:  # noqa: BLE001
        out["trace_error"] = f"{type(e).__name__}: {e}"
    return out


def op_runrec(o):
    from pigemma import trace as tr
    tp = Path(o["trace"])
    summary = {}
    lp = Path(o["log"]) if o.get("log") else None
    if lp is not None and lp.is_file():
        summary = tr.parse_probe_summary(lp.read_text(encoding="utf-8", errors="replace")) or {}
    meta = tr.read_meta(tp) if tp.is_file() else None
    if not summary and meta:
        summary = dict(meta.get("probe_summary") or {})
    rec = {"item": o["item"], "kind": o["kind"], "trace": o["trace"], "corpus": o["corpus"],
           "log": o.get("log") or None, "rc": int(o["rc"]), "validate_rc": int(o["validate_rc"]),
           "meta_rc": int(o["meta_rc"]), "host_wall_ms": num(o.get("host_wall_ms")),
           "resumed": o.get("resumed") == "1", "summary": summary}
    rec.update(_trace_facts(tp))
    if meta:
        rec["meta"] = str(tr.meta_path(tp))
        rec["coherent"] = meta.get("coherent")
        rec["coherent_by_chunk"] = meta.get("coherent_by_chunk") or {}
        rec["gguf_sha256_prefix"] = meta.get("gguf_sha256_prefix")
    with open(o["ledger"], "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    s = summary
    valid = {0: "valid", -1: "not checked"}.get(rec["validate_rc"], "INVALID")
    line = (f"{rec['item']}: exit {rec['rc']}{' (resumed)' if rec['resumed'] else ''} | {valid}"
            f" | {rec.get('n_records', 0)} records ({rec.get('n_prefill', 0)} prefill + {rec.get('n_gen', 0)} gen,"
            f" {len(rec.get('chunks', []))} chunks, {rec.get('n_moe_layers', 0)}/{rec.get('n_layer', 0)} layers)"
            f" | probe pp {fmt(s.get('tok_s_prompt'), '.1f')} tok/s, tg {fmt(s.get('tok_s_gen'))} tok/s"
            f" (wall incl. callback: {fmt(s.get('wall_tok_s_prompt'), '.1f')} / {fmt(s.get('wall_tok_s_gen'))})"
            f" | load {fmt((num(s.get('t_load_ms')) or 0) / 1000.0, '.1f')} s"
            f" | callback {fmt(s.get('callback_ms'), '.0f')} ms"
            f" | wall {fmt((num(s.get('wall_ms')) or num(o.get('host_wall_ms')) or 0) / 1000.0, '.0f')} s"
            f" | coherent {rec.get('coherent')}")
    if s.get("error"):
        line += f" | probe error: {s.get('error')}"
    print(line)
    return 0


# ---------------------------------------------------------------------------- smoke
def op_smoke(o):
    import numpy as np
    from pigemma import trace as tr
    tp = Path(o["trace"])
    exp_n, exp_pf = int(o["expect_records"]), int(o["expect_prefill"])
    checks = []
    rep = tr.validate_report(tp) if tp.is_file() else None
    checks.append(("validate", rep is not None and rep.ok,
                   "missing trace" if rep is None else ("clean" if rep.ok else "; ".join(rep.errors[:3]))))
    f = _trace_facts(tp)
    n, pf, g = f.get("n_records", 0), f.get("n_prefill", 0), f.get("n_gen", 0)
    checks.append(("records", n == exp_n, f"{n} records, expected {exp_n}"))
    checks.append(("prefill/gen", pf == exp_pf and g == exp_n - exp_pf,
                   f"{pf} prefill + {g} generated, expected {exp_pf} + {exp_n - exp_pf}"))
    nl, ml = f.get("n_layer", 0), f.get("moe_layers", [])
    checks.append(("layers", nl > 0 and ml == list(range(nl)),
                   f"{len(ml)} MoE layers of n_layer {nl}" + (" (26B: 30)" if nl == 30 else "")))
    ok = all(c[1] for c in checks)
    for name, good, detail in checks:
        print(f"CHECK smoke {name}: {'PASS' if good else 'FAIL'} ({detail})")
    gp = Path(o.get("golden") or "")
    if ok and gp.is_file():
        try:
            with tr.TraceReader(tp) as a, tr.TraceReader(gp) as b:
                m = min(a.n, b.n)
                if a.L == b.L and a.K == b.K and m > 0:
                    same_tok = int(np.count_nonzero(a.token_ids[:m] == b.token_ids[:m]))
                    ia, ib = np.sort(a.ids[:m], axis=2), np.sort(b.ids[:m], axis=2)
                    inter = 0
                    for k in range(a.K):
                        inter += (ia[:, :, k:k + 1] == ib).any(axis=2).sum()
                    overlap = inter / float(m * a.L * a.K)
                    exact = float(np.mean(np.all(ia == ib, axis=2)))
                    print(f"INFO golden {gp.as_posix()}: token ids identical {same_tok}/{m}; expert-set overlap "
                          f"{overlap:.4f}, identical sets {exact:.4f} (threads/ubatch may differ from the golden "
                          f"run; small numeric differences can flip near-tied experts)")
        except Exception as e:  # noqa: BLE001
            print(f"INFO golden comparison skipped: {type(e).__name__}: {e}")
    print(f"SMOKE {'PASS' if ok else 'FAIL'}: {tp.as_posix()}")
    return 0 if ok else EXIT_CHECKS


# ---------------------------------------------------------------------------- finalize
def _load_ledger(p):
    recs = []
    if Path(p).is_file():
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _wavg(recs, nk, tk):
    n = sum(int((r.get("summary") or {}).get(nk) or 0) for r in recs)
    t = sum(num((r.get("summary") or {}).get(tk)) or 0.0 for r in recs)
    return (1000.0 * n / t) if (n > 0 and t > 0) else None


def _sd(vals):
    vals = [v for v in vals if v is not None]
    return statistics.stdev(vals) if len(vals) >= 2 else None


def _e3_preview(st):
    pl = st.get("per_layer") or []
    g = st.get("global") or {}
    cl = st.get("cross_layer") or {}

    def mean_key(d, k):
        vals = [num((x.get(d) or {}).get(k)) for x in pl]
        vals = [v for v in vals if v is not None]
        return statistics.fmean(vals) if vals else None

    cov = {k: mean_key("coverage_at", k) for k in ("8", "16", "32", "64")}
    covw32 = mean_key("coverage_weight_at", "32")
    adj = num(g.get("adjacent_reuse_mean"))
    c32 = cov["32"]
    if c32 is None:
        skew = None
    elif c32 >= 0.70 or (adj is not None and adj >= 0.40):
        skew = "strong"
    elif c32 >= 0.5:
        skew = "moderate"
    else:
        skew = "near-uniform"
    p16, st16, same = (num(cl.get("precision_at_16")), num(cl.get("baseline_static_top16")),
                       num(cl.get("baseline_same_ids")))
    gain = (p16 - st16) if (p16 is not None and st16 is not None) else None
    js = [num(x.get("prefill_vs_gen_js_bits")) for x in pl]
    js = [v for v in js if v is not None]
    js_mean = statistics.fmean(js) if js else None
    ent = [num(x.get("entropy_bits")) for x in pl]
    ent = [v for v in ent if v is not None]
    return {
        "kind": "measured", "n_prefill": st.get("n_prefill"), "n_gen": st.get("n_gen"),
        "coverage_mean": cov, "coverage_weight_32_mean": covw32, "adjacent_reuse_mean": adj,
        "entropy_bits_mean": statistics.fmean(ent) if ent else None,
        "skew_class": skew, "precision_at_16": p16, "baseline_static_top16": st16, "baseline_same_ids": same,
        "prefetch_gain_vs_static": gain, "prefetch_worth_modelling": None if gain is None else gain >= 0.10,
        "c3_prefill_vs_gen_js_bits_mean": js_mean, "c3_pass": None if js_mean is None else js_mean <= 0.10,
        "rules": ("strong if mean coverage@32 >= 0.70 or adjacent reuse >= 0.40; moderate 0.5-0.7; near-uniform "
                  "< 0.5; prefetch worth modelling if precision@16 - static top16 >= 0.10; C3 mean prefill-vs-gen "
                  "JS <= 0.10 bits (docs/EXPERIMENT_PLAN.md E3)"),
    }


def op_finalize(o):
    tier = o["tier"]
    recs = _load_ledger(o["ledger"])
    ok_recs = [r for r in recs if r["rc"] == 0 and r["validate_rc"] == 0]
    merged = Path(o.get("merged") or "")
    mf = _trace_facts(merged) if o.get("merged") and merged.is_file() else {}
    n_prefill = mf.get("n_prefill", sum(r.get("n_prefill", 0) for r in ok_recs))
    n_gen = mf.get("n_gen", sum(r.get("n_gen", 0) for r in ok_recs))
    n_records = mf.get("n_records", sum(r.get("n_records", 0) for r in ok_recs))
    n_chunks = len(mf.get("chunks", [])) if mf else sum(len(r.get("chunks", [])) for r in ok_recs)
    min_pf, min_gen = int(o["min_prefill"]), int(o["min_gen"])

    print("")
    print(f"== per-run probe numbers (MEASURED, Docker on this desktop, -nr, cb_eval recorder active) ==")
    print(f"{'item':34s} {'exit':>4s} {'records':>8s} {'prefill':>8s} {'gen':>6s} {'chunks':>6s} "
          f"{'pp tok/s':>9s} {'tg tok/s':>9s} {'load s':>7s} {'wall s':>7s} coherent")
    for r in recs:
        s = r.get("summary") or {}
        print(f"{r['item'][:34]:34s} {r['rc']:>4d} {r.get('n_records', 0):>8d} {r.get('n_prefill', 0):>8d} "
              f"{r.get('n_gen', 0):>6d} {len(r.get('chunks', [])):>6d} {fmt(s.get('tok_s_prompt'), '.1f'):>9s} "
              f"{fmt(s.get('tok_s_gen')):>9s} {fmt((num(s.get('t_load_ms')) or 0) / 1000.0, '.1f'):>7s} "
              f"{fmt((num(s.get('wall_ms')) or r.get('host_wall_ms') or 0) / 1000.0, '.0f'):>7s} "
              f"{r.get('coherent')}{' (resumed)' if r.get('resumed') else ''}")
    pp = _wavg(ok_recs, "n_p_eval", "t_p_eval_ms")
    tg = _wavg(ok_recs, "n_eval", "t_eval_ms")
    loads = [num((r.get("summary") or {}).get("t_load_ms")) for r in ok_recs]
    loads = [v / 1000.0 for v in loads if v is not None and v > 0]
    cb_total = sum(num((r.get("summary") or {}).get("callback_ms")) or 0.0 for r in ok_recs)
    print(f"aggregate (token-weighted, llama_perf): pp {fmt(pp, '.1f')} tok/s, tg {fmt(tg)} tok/s; "
          f"mean load {fmt(statistics.fmean(loads) if loads else None, '.1f')} s; callback total {cb_total:.0f} ms")

    # ---- acceptance checks
    checks = []
    bad = [r["item"] for r in recs if not (r["rc"] == 0 and r["validate_rc"] == 0 and r["meta_rc"] == 0)]
    checks.append(("runs", bool(recs) and not bad,
                   f"{len(recs) - len(bad)}/{len(recs)} runs exit 0 + valid + meta" + (f"; failed: {', '.join(bad)}" if bad else "")))
    mv = o.get("merged_validate_rc", "")
    if mv == "skip":
        checks.append(("merged trace valid", None, "skipped (--no-merge)"))
    else:
        checks.append(("merged trace valid", mv == "0", "python -m pigemma.trace validate exit " + (mv or "not run")))
    checks.append(("prefill tokens", n_prefill >= min_pf, f"{n_prefill} >= {min_pf}"))
    checks.append(("generated tokens", n_gen >= min_gen, f"{n_gen} >= {min_gen}"))
    n_checked = n_bad = n_nogen = 0
    unknown = []
    bad_chunks = []
    for r in ok_recs:
        by = r.get("coherent_by_chunk") or {}
        chunks = r.get("chunks", [])
        if r.get("coherent") is None and any(int(c.get("n_gen", 0)) > 0 for c in chunks):
            unknown.append(r["item"])
        for i, c in enumerate(chunks):
            if int(c.get("n_gen", 0)) <= 0:
                n_nogen += 1
                continue
            v = by.get(str(i))
            if v is None:
                continue
            n_checked += 1
            if not v:
                n_bad += 1
                bad_chunks.append(f"{r['item']}#{i}")
    coh_ok = n_checked > 0 and n_bad == 0 and not unknown
    detail = f"{n_checked - n_bad}/{n_checked} chunks with generated text coherent"
    if n_nogen:
        detail += f"; {n_nogen} chunk(s) generated nothing (EOG at once)"
    if bad_chunks:
        detail += "; incoherent: " + ", ".join(bad_chunks[:10])
    if unknown:
        detail += "; coherence unknown for " + ", ".join(unknown)
    checks.append(("coherent", coh_ok, detail))
    lay_bad = [r["item"] for r in ok_recs if r.get("moe_layers") != list(range(int(r.get("n_layer", 0) or 0)))]
    checks.append(("all MoE layers", bool(ok_recs) and not lay_bad,
                   "every trace records layers 0..n_layer-1" + (f"; not: {', '.join(lay_bad)}" if lay_bad else "")))
    all_ok = all(c[1] is not False for c in checks)
    print("")
    print("== acceptance checks (E3 'Trace VALID' rule) ==")
    for name, good, d in checks:
        print(f"CHECK {name}: {'SKIP' if good is None else ('PASS' if good else 'FAIL')} ({d})")
    print(f"TRACE {'VALID' if all_ok else 'NOT VALID'}: {o.get('merged') or '(not merged)'}: {n_records} records, "
          f"{n_prefill} prefill + {n_gen} generated, {n_chunks} chunks")

    # ---- E3 preview
    preview = None
    sp = Path(o.get("stats") or "")
    if o.get("stats") and sp.is_file():
        try:
            preview = _e3_preview(json.loads(sp.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            print(f"E3 preview skipped: {type(e).__name__}: {e}")
    if preview:
        c = preview["coverage_mean"]
        print("")
        print(f"== E3 preview (MEASURED on {sp.as_posix()}; the verdict belongs in the write-up) ==")
        print(f"mean coverage @8 {fmt(c['8'], '.3f')} @16 {fmt(c['16'], '.3f')} @32 {fmt(c['32'], '.3f')} "
              f"@64 {fmt(c['64'], '.3f')} (uniform: 0.0625 / 0.125 / 0.25 / 0.5); by weight @32 "
              f"{fmt(preview['coverage_weight_32_mean'], '.3f')}")
        print(f"adjacent reuse mean {fmt(preview['adjacent_reuse_mean'], '.3f')} (uniform 0.0625); entropy mean "
              f"{fmt(preview['entropy_bits_mean'], '.3f')} bits (uniform 7)")
        print(f"skew class: {preview['skew_class']} (strong: coverage@32 >= 0.70 or adjacent reuse >= 0.40; "
              f"moderate 0.5-0.7; near-uniform < 0.5)")
        print(f"cross-layer precision@16 {fmt(preview['precision_at_16'], '.3f')} vs static top16 "
              f"{fmt(preview['baseline_static_top16'], '.3f')} vs same ids {fmt(preview['baseline_same_ids'], '.3f')}"
              f" -> prefetch worth modelling: {preview['prefetch_worth_modelling']} (gain >= 0.10)")
        print(f"C3 mean prefill-vs-gen JS {fmt(preview['c3_prefill_vs_gen_js_bits_mean'], '.3f')} bits -> "
              f"{'PASS' if preview['c3_pass'] else ('FAIL' if preview['c3_pass'] is False else 'n/a')} (<= 0.10)")

    # ---- sizing of the full tier from this run's probe rates
    proj = None
    if tier == "quick" and pp and tg:
        fc, fg, pg = int(o["full_chunk"]), int(o["full_gen"]), int(o["prompt_gen"])
        pf_full = gen_full = 0
        for r in ok_recs:
            ch = r.get("chunks", [])
            if r["kind"] == "prompts":
                pf_full += sum(int(x.get("n_prompt", 0)) for x in ch)
                gen_full += len(ch) * pg
            else:
                text = sum(max(0, int(x.get("n_prompt", 0)) - 1) for x in ch)
                nchunk = math.ceil(text / max(1, fc - 1)) if text else 0
                pf_full += text + nchunk
                gen_full += nchunk * fg
        load = statistics.fmean(loads) if loads else 0.0
        secs = pf_full / pp + gen_full / tg + len(ok_recs) * load
        proj = {"kind": "projected", "tier": "full", "chunk_tokens": fc, "gen_per_chunk": fg,
                "prefill_tokens": pf_full, "gen_tokens_max": gen_full, "seconds": round(secs, 1),
                "assumptions": {"pp_tok_s": {"value": round(pp, 3), "unit": "tok/s",
                                             "source": "measured: this quick run (llama_perf, token-weighted)"},
                                "tg_tok_s": {"value": round(tg, 4), "unit": "tok/s",
                                             "source": "measured: this quick run; longer contexts decode a bit slower"},
                                "load_s": {"value": round(load, 2), "unit": "s",
                                           "source": "measured: mean model load of this run (page cache warm)"}}}
        print("")
        print(f"PROJECTED --full (chunk {fc}, --gen {fg}/chunk): {pf_full} prefill + <= {gen_full} generated tokens, "
              f"~{secs / 60:.0f} min at this run's pp {pp:.1f} / tg {tg:.2f} tok/s + {len(ok_recs)} loads of {load:.0f} s")
    elif tier == "full" and pp and tg:
        print("")
        print(f"full tier measured: {n_prefill} prefill + {n_gen} generated at pp {pp:.1f} / tg {tg:.2f} tok/s")

    # ---- E3 envelope fields
    fo = o.get("fields_out") or ""
    if fo:
        host = None
        hj = o.get("host_json") or ""
        if hj and Path(hj).is_file():
            try:
                host = json.loads(Path(hj).read_text(encoding="utf-8"))
                if not isinstance(host, dict) or not host.get("hostname"):
                    host = None
            except Exception:
                host = None
        sums = [r.get("summary") or {} for r in ok_recs]
        lb = next((s.get("llama_build") for s in sums if s.get("llama_build") is not None), None)
        lc = next((s.get("llama_commit") for s in sums if s.get("llama_commit")), None)
        per_pp = [num(s.get("tok_s_prompt")) for s in sums]
        per_tg = [num(s.get("tok_s_gen")) for s in sums]
        notes = [
            "probe tok/s measured inside Docker Desktop (WSL2 VM) on the /work bind mount with -nr and the "
            "cb_eval recorder active; a by-product of E3, not an engine benchmark (E0 owns engine speed)",
            "pp/tg = token-weighted over the probe runs (llama_perf n_p_eval/t_p_eval, n_eval/t_eval); "
            "sd = spread of the per-run rates; extra.runs has every run",
            f"TRACE {'VALID' if all_ok else 'NOT VALID'} by the E3 rule (extra.checks)",
        ]
        if host is None:
            notes.append("host = the machine that wrote this file: the container fingerprint was not captured")
        else:
            notes.append("host = the container fingerprint (where the probe ran)")
        bg = None
        if o.get("bgcpu") and Path(o["bgcpu"]).is_file():
            try:
                bg = json.loads(Path(o["bgcpu"]).read_text(encoding="utf-8"))
            except Exception:
                bg = None
        if bg and bg.get("mean_pct") is not None:
            notes.append(f"background CPU before the runs: mean {bg.get('mean_pct')} %, max {bg.get('max_pct')} % "
                         f"over {bg.get('seconds')} s (recorded, not gated)")
        fields = {
            "experiment": "E3", "kind": "measured", "model": o["model_rel"],
            "variant": f"moe-trace-{tier}", "flags": shlex.split(o.get("flags") or ""),
            "threads": int(o["threads"]), "n_ctx": int(o["ctx"]),
            "llama_build": lb, "llama_commit": lc,
            "coherent": coh_ok,
            "inputs": [r["trace"] for r in ok_recs] + [x for x in (o.get("merged"), o.get("stats")) if x]
            + sorted({r["corpus"] for r in ok_recs}),
            "notes": notes,
            "extra": {
                "driver": f"pc/trace_collect.sh {o.get('version', '')}", "tier": tier,
                "chunk_tokens": int(o["chunk"]), "gen_per_chunk": int(o["gen"]), "prompt_gen": int(o["prompt_gen"]),
                "n_records": n_records, "n_prefill": n_prefill, "n_gen": n_gen, "n_chunks": n_chunks,
                "callback_ms_total": round(cb_total, 3),
                "trace_valid": all_ok,
                "checks": {name: {"pass": None if good is None else bool(good), "detail": d}
                           for name, good, d in checks},
                "e3_preview": preview, "full_tier_projection": proj, "background_cpu": bg,
                "merged_trace": o.get("merged") or None, "stats": o.get("stats") or None,
                "logs": {"driver": o.get("log_file") or None, "ledger": o["ledger"]},
                "docker_image": o.get("image") or None,
                "runs": [{
                    "item": r["item"], "kind": r["kind"], "trace": r["trace"], "rc": r["rc"],
                    "n_records": r.get("n_records"), "n_prefill": r.get("n_prefill"), "n_gen": r.get("n_gen"),
                    "chunks": r.get("chunks"), "coherent": r.get("coherent"), "resumed": r.get("resumed"),
                    "tok_s_prompt": (r.get("summary") or {}).get("tok_s_prompt"),
                    "tok_s_gen": (r.get("summary") or {}).get("tok_s_gen"),
                    "wall_tok_s_prompt": (r.get("summary") or {}).get("wall_tok_s_prompt"),
                    "wall_tok_s_gen": (r.get("summary") or {}).get("wall_tok_s_gen"),
                    "t_load_ms": (r.get("summary") or {}).get("t_load_ms"),
                    "callback_ms": (r.get("summary") or {}).get("callback_ms"),
                    "wall_ms": (r.get("summary") or {}).get("wall_ms"),
                } for r in recs],
            },
        }
        if host is not None:
            fields["host"] = host
        if pp is not None:
            fields["pp"] = {"tok_s": round(pp, 4), "sd": _sd(per_pp), "n": sum(1 for v in per_pp if v is not None)}
        if tg is not None:
            fields["tg"] = {"tok_s": round(tg, 4), "sd": _sd(per_tg), "n": sum(1 for v in per_tg if v is not None)}
        if loads:
            fields["load_s"] = round(statistics.fmean(loads), 3)
        with open(fo, "w", encoding="utf-8", newline="\n") as f:
            json.dump(fields, f, ensure_ascii=False, indent=1, default=str)
    return 0 if all_ok else EXIT_CHECKS


# ---------------------------------------------------------------------------- bgcpu
def op_bgcpu(o):
    try:
        from pigemma import bench
        d = bench.measure_background_cpu(float(o.get("seconds", "5")))
    except Exception as e:  # noqa: BLE001
        d = {"mean_pct": None, "max_pct": None, "error": f"{type(e).__name__}: {e}"}
    Path(o["out"]).write_text(json.dumps(d), encoding="utf-8")
    print(f"background CPU (all cores) over {d.get('seconds', o.get('seconds'))} s: mean {d.get('mean_pct')} %, "
          f"max {d.get('max_pct')} % (recorded, not gated)")
    return 0


OPS = {"model": op_model, "resume-ok": op_resume_ok, "runrec": op_runrec, "smoke": op_smoke,
       "finalize": op_finalize, "bgcpu": op_bgcpu}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in OPS:
        raise SystemExit(f"trace_collect helper: unknown op {sys.argv[1:2]}")
    sys.exit(OPS[sys.argv[1]](opts(sys.argv[2:])))
PY

hpy() { PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "$PY_BIN" - "$@" <<<"$HELPER_PY"; }

# eval_kv TEXT - evaluate only KEY='value' lines (shlex-quoted by the helper)
eval_kv() {
    local line
    while IFS= read -r line; do
        if [[ $line =~ ^[A-Z_]+= ]]; then eval "$line"; fi
    done <<<"$1"
}

# ---------------------------------------------------------------------------
# corpus plan
# ---------------------------------------------------------------------------
ITEM_KIND=() ITEM_SRC=() ITEM_CID=() ITEM_TRACE=()
CORPUS_ALL=()

discover_corpus() {
    local f base ext list
    [ -d "$CORPUS_DIR" ] || die "$EXIT_MISSING" "corpus directory $CORPUS_DIR not found"
    list=$(for f in "$CORPUS_DIR"/*; do printf '%s\n' "${f##*/}"; done | LC_ALL=C sort)
    while IFS= read -r base; do
        [ -n "$base" ] || continue
        f=$CORPUS_DIR/$base
        [ -f "$f" ] || continue
        case $base in .*|*~|README*|readme*|all.*) continue ;; esac
        ext=${base##*.}
        if [ "$ext" = "$base" ]; then continue; fi
        case $CORPUS_EXTS in *" ${ext,,} "*) ;; *) continue ;; esac
        if [ ! -s "$f" ]; then warn "skipping empty corpus file $f"; continue; fi
        CORPUS_ALL+=("$base")
    done <<<"$list"
}

sanitize_id() {
    local s=$1
    s=${s//[^A-Za-z0-9._-]/_}
    printf '%s' "$s"
}

build_plan() {
    local base stem cid f want w found prompts_stem
    local -A seen=()
    prompts_stem=$(basename -- "$PROMPTS_REL")
    prompts_stem=$(sanitize_id "${prompts_stem%.*}")
    if [ "$TIER" = smoke ]; then
        f=$CORPUS_DIR/$SMOKE_FILE
        ITEM_KIND+=(smoke); ITEM_SRC+=("$f"); ITEM_CID+=("${SMOKE_FILE%.*}")
        ITEM_TRACE+=("$OUT_REL/smoke/${SMOKE_FILE%.*}.moetrace")
        return 0
    fi
    discover_corpus
    local -a chosen=()
    if [ -n "$FILES" ]; then
        local -a wanted=()
        IFS=, read -r -a wanted <<<"$FILES"
        for want in "${wanted[@]}"; do
            w=${want// /}
            [ -n "$w" ] || continue
            found=""
            for base in "${CORPUS_ALL[@]}"; do
                if [ "$w" = "$base" ] || [ "$w" = "${base%.*}" ]; then found=$base; break; fi
            done
            if [ -z "$found" ]; then
                die "$EXIT_USAGE" "--files: '$w' is not a corpus file of $CORPUS_DIR (have: ${CORPUS_ALL[*]})"
            fi
            chosen+=("$found")
        done
    else
        chosen=("${CORPUS_ALL[@]}")
    fi
    if [ "${#chosen[@]}" -eq 0 ] && [ "$NO_PROMPTS" = 1 ]; then
        die "$EXIT_USAGE" "nothing to trace: no corpus file selected and --no-prompts"
    fi
    for base in "${chosen[@]}"; do
        stem=$(sanitize_id "${base%.*}")
        cid=$stem
        case $cid in all|smoke|"$prompts_stem") cid=$(sanitize_id "${base//./_}") ;; esac
        if [ -n "${seen[$cid]+x}" ]; then cid=$(sanitize_id "${base//./_}"); fi
        if [ -n "${seen[$cid]+x}" ]; then die "$EXIT_USAGE" "corpus id collision for $base ($cid)"; fi
        seen[$cid]=1
        ITEM_KIND+=(text); ITEM_SRC+=("$CORPUS_DIR/$base"); ITEM_CID+=("$cid")
        ITEM_TRACE+=("$OUT_REL/$cid.moetrace")
    done
    if [ "$NO_PROMPTS" = 0 ]; then
        ITEM_KIND+=(prompts); ITEM_SRC+=("$PROMPTS_REL"); ITEM_CID+=("$prompts_stem")
        ITEM_TRACE+=("$OUT_REL/$prompts_stem.moetrace")
    fi
}

# probe_cmd I - fills PROBE_CMD with the container command of item I
PROBE_CMD=()
probe_cmd() {
    local i=$1 kind=${ITEM_KIND[$1]} src=${ITEM_SRC[$1]}
    local -a input=() tier=() probs=()
    case $kind in
        text)    input=(-f "$CONTAINER_ROOT/$src"); tier=(--chunk-tokens "$CHUNK" --gen "$GEN") ;;
        smoke)   input=(-f "$CONTAINER_ROOT/$src")
                 tier=(--chunk-tokens "$CHUNK" --gen "$GEN" --tokens "$((CHUNK + GEN))" --max-chunks 1) ;;
        prompts) input=(--gen-prompts "$CONTAINER_ROOT/$src" --chat); tier=(--gen "$PROMPT_GEN") ;;
    esac
    if [ "$PROBS_TOP" -gt 0 ]; then probs=(--probs-top "$PROBS_TOP"); fi
    PROBE_CMD=(python -m pigemma.paths lock --hold -- "$PROBE_C" --mode moe-trace -m "$(model_c)"
               "${input[@]}" -nr -t "$THREADS" -c "$CTX" -b "$BATCH" -ub "$UBATCH" "${tier[@]}" "${probs[@]}"
               "${PROBE_EXTRA[@]}" --corpus-id "${ITEM_CID[$i]}" --trace-out "$CONTAINER_ROOT/${ITEM_TRACE[$i]}")
}

container_name() { printf 'pigemma-trace-%s-%s' "$(sanitize_id "$1")" "$$"; }

# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
MODEL_REL="" MODEL_ID="" MODEL_KIND="" MODEL_EXPECT_BYTES="" MODEL_EXISTS=0 MODEL_SIZE="" MODEL_HINT=""
MODEL_OUTSIDE="" MODEL_CATALOG_KIND=""
GATE_PROBLEMS=()   # "rc|message" entries collected in dry-run mode

gate_fail() {  # gate_fail RC MESSAGE - die (real run) or remember (dry run)
    if [ "$DRY_RUN" = 1 ]; then
        GATE_PROBLEMS+=("$1|$2")
        gate "a real run would stop here (exit $1): $2"
    elif [ "$1" = "$EXIT_REFUSED" ]; then
        refuse "$2"
    else
        die "$1" "$2"
    fi
}

resolve_model() {
    local arg out rc=0
    case $MODEL_SPEC in
        */*|*\\*|*.gguf|*.GGUF)
            arg=$(repo_rel "$MODEL_SPEC") \
                || die "$EXIT_USAGE" "--model '$MODEL_SPEC' is outside the repository $REPO_POSIX (the container only sees /work; put the GGUF under models/)" ;;
        *) arg=$MODEL_SPEC ;;
    esac
    out=$(hpy model --spec "$arg" | tr -d '\r') || rc=$?
    eval_kv "$out"
    if [ "$rc" = 2 ] || [ -n "$MODEL_OUTSIDE" ]; then
        die "$EXIT_USAGE" "--model $MODEL_SPEC resolves to $MODEL_OUTSIDE, outside the repository (the container only sees /work)"
    fi
    if [ "$rc" != 0 ] || [ -z "$MODEL_REL" ]; then
        die "$EXIT_ERROR" "cannot resolve --model $MODEL_SPEC (helper exit $rc)"
    fi
    local kind=${MODEL_KIND:-$MODEL_CATALOG_KIND}
    if [ -n "$kind" ] && [ "$kind" != moe ]; then
        die "$EXIT_USAGE" "--model $MODEL_SPEC is a '$kind' model; moe-trace needs an MoE GGUF (default $DEFAULT_MODEL)"
    fi
}

check_inputs() {
    local i f
    if [ "$MODEL_EXISTS" != 1 ]; then
        gate_fail "$EXIT_MISSING" "model $MODEL_REL is missing${MODEL_HINT:+ - $MODEL_HINT}"
    elif [ -n "$MODEL_EXPECT_BYTES" ] && [ "$MODEL_SIZE" != "$MODEL_EXPECT_BYTES" ]; then
        gate_fail "$EXIT_MISSING" "model $MODEL_REL has $MODEL_SIZE bytes, the registry says $MODEL_EXPECT_BYTES (incomplete download?)${MODEL_HINT:+ - $MODEL_HINT}"
    else
        gate "model ${MODEL_ID:-?} $MODEL_REL ($(fmt_gb "$MODEL_SIZE")) present"
    fi
    if [ -n "$PROBE_HOST" ]; then
        if [ -f "$PROBE_HOST" ]; then
            gate "probe $PROBE_HOST present"
        else
            gate_fail "$EXIT_MISSING" "probe binary $PROBE_HOST not found - build it with: bash pc/docker_run.sh build-tools"
        fi
    else
        gate "probe $PROBE_C is a container path outside /work; checked in the preflight container"
    fi
    for i in "${!ITEM_SRC[@]}"; do
        f=${ITEM_SRC[$i]}
        if [ ! -f "$f" ]; then gate_fail "$EXIT_MISSING" "input file $f not found"; fi
    done
}

lock_state() {  # prints 'state|reason' of results/.bench.lock (or 'unknown|...')
    local out rc=0
    out=$(pym pigemma.paths lock --status 2>/dev/null | tr -d '\r') || rc=$?
    "$PY_BIN" -c 'import json,sys
try:
    d = json.loads(sys.stdin.read())
    h = d.get("holder") or {}
    print("%s|%s (cmd: %s)" % (d.get("state"), d.get("reason"), str(h.get("cmd", ""))[:160]))
except Exception as e:
    print("unknown|%s" % e)' <<<"$out" | tr -d '\r'
}

# Model-loading llama.cpp tools (+ the probe). File-only tools such as llama-gguf-split or
# llama-gguf do not count: they move bytes but load no model (seen 2026-10-05 while a 31B
# GGUF was split for the Pi).
readonly BUSY_TOOLS_RE='pigemma-probe|llama-(cli|server|bench|completion|perplexity|speculative|speculative-simple|simple|simple-chat|debug|eval-callback|imatrix|quantize|run|mtmd-cli|embedding|retrieval|lookup|parallel|batched|passkey|tts)'

busy_check() {  # prints matching process/container lines (empty = idle)
    local out="" dps tl
    if command -v docker >/dev/null 2>&1; then
        if command -v timeout >/dev/null 2>&1; then
            dps=$(timeout 30 docker ps --no-trunc --format '{{.Names}} {{.Command}}' 2>/dev/null | tr -d '\r') || dps=""
        else
            dps=$(docker ps --no-trunc --format '{{.Names}} {{.Command}}' 2>/dev/null | tr -d '\r') || dps=""
        fi
        # one short line per matching container: name + the command after the docker_run shim
        out+=$(grep -E "(^|[/ ])($BUSY_TOOLS_RE)( |\"|\$)" <<<"$dps" \
            | sed -E 's/^([^ ]+) .*pigemma-run /\1 /; s/^([^ ]+) "?/container \1: /' | cut -c1-200 || true)
        if [ -n "$out" ]; then out+=$'\n'; fi
    fi
    if [ "$IS_MSYS" = 1 ] && command -v tasklist >/dev/null 2>&1; then
        tl=$(tasklist /FO CSV /NH 2>/dev/null | tr -d '\r') || tl=""
        out+=$(grep -i -E "^\"($BUSY_TOOLS_RE)(\\.exe)?\"" <<<"$tl" | cut -d, -f1,2 | sed 's/^/process: /' || true)
    elif [ -d /proc ]; then
        out+=$(ps -eo pid=,args= 2>/dev/null \
            | grep -E "(^|/| )($BUSY_TOOLS_RE)( |\$)" \
            | grep -v -E "^ *($$|$PPID) " | sed 's/^/process: /' || true)
    fi
    printf '%s' "$out"
}

# ---------------------------------------------------------------------------
# run helpers
# ---------------------------------------------------------------------------
run_logged() {  # run_logged CMD... - stdout+stderr to the console (stderr) and the driver log
    local ps
    set +e
    "$@" 2>&1 | tr -d '\r' | tee -a "${LOG_FILE:-/dev/null}" >&2
    ps=("${PIPESTATUS[@]}")
    set -e
    return "${ps[0]}"
}
run_quiet() {  # run_quiet CMD... - output only into the driver log
    local ps
    set +e
    "$@" 2>&1 | tr -d '\r' >>"${LOG_FILE:-/dev/null}"
    ps=("${PIPESTATUS[@]}")
    set -e
    return "${ps[0]}"
}
console_filter() {
    if [ "$VERBOSE" = 1 ]; then
        cat >&2
        return 0
    fi
    grep --line-buffered -a -v '^PIGEMMA_SUMMARY ' \
        | grep --line-buffered -a -E '^(pigemma-probe|pigemma\.paths|docker_run)|[Ee][Rr][Rr][Oo][Rr]|[Ff]ailed|REFUSED|Killed|[Oo]ut of memory' >&2 \
        || true
}

TICKER_PID=""
start_ticker() {
    local label=$1
    (
        n=0
        while sleep 10; do
            n=$((n + 10))
            if [ $((n % 60)) -eq 0 ]; then log "... $label still running ($((n / 60)) min)"; fi
        done
    ) &
    TICKER_PID=$!
}
stop_ticker() {
    if [ -n "$TICKER_PID" ]; then
        kill "$TICKER_PID" 2>/dev/null || true
        wait "$TICKER_PID" 2>/dev/null || true
        TICKER_PID=""
    fi
}

TMPD=""
cleanup() {
    stop_ticker
    if [ -n "$TMPD" ] && [ -d "$TMPD" ]; then rm -rf -- "$TMPD"; fi
}
trap cleanup EXIT
INTERRUPTED=0
on_signal() {
    INTERRUPTED=1
    log "interrupt received: the probe finishes its current llama_process and closes its outputs; stopping after this run"
}

rm_trace_outputs() {  # rm_trace_outputs TRACE - the trace and its sidecars (only files this run regenerates)
    local t=$1 f
    for f in "$t" "$t.tokens.bin" "$t.gen.txt" "$t.meta.json" "$t.tmp" "$t.tmp.tokens.bin" "$t.tmp.gen.txt"; do
        if [ -e "$f" ]; then rm -f -- "$f"; fi
    done
}

summary_value() {  # summary_value LOG KEY - numeric value of KEY in the last PIGEMMA_SUMMARY line
    local line
    line=$(grep -a '^PIGEMMA_SUMMARY ' -- "$1" 2>/dev/null | tail -n 1 | tr -d '\r') || line=""
    sed -n "s/.*\"$2\":\([-0-9.eE+]*\)[,}].*/\1/p" <<<"$line" | head -n 1
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
main() {
    RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
    pick_python || die "$EXIT_MISSING" "no Python >= 3.10 found (python / python3 / PIGEMMA_PYTHON)"
    build_plan
    resolve_model

    # context / batch sanity (D9)
    local need=$((CHUNK + GEN + 8))
    if [ "$TIER" != smoke ] && [ "$NO_PROMPTS" = 0 ] && [ $((512 + PROMPT_GEN)) -gt "$need" ]; then
        need=$((512 + PROMPT_GEN))
    fi
    if [ "$CTX" -lt "$need" ]; then
        local newctx=$(( (need + 255) / 256 * 256 ))
        warn "-c $CTX cannot hold a chunk of $CHUNK + $GEN generated tokens: using -c $newctx"
        CTX=$newctx
    fi
    if [ "$BATCH" -lt "$CHUNK" ]; then
        log "note: -b $BATCH < chunk $CHUNK: the probe decodes each chunk in sub-batches of $BATCH (probe D-d)"
    fi

    local n=${#ITEM_CID[@]} i
    local merged="" stats=""
    if [ "$TIER" != smoke ]; then
        merged=$OUT_REL/all.moetrace
        stats=$OUT_REL/all.stats.json
    fi

    if [ "$DRY_RUN" = 1 ]; then
        dry_run_plan "$merged" "$stats"
        return 0
    fi

    # ---- gates of a real run. A process that is about to re-run itself under the bench lock
    # (D1) checks them quietly (failures still stop it); the child logs them once.
    local reexec=0
    if [ -z "${PIGEMMA_LOCK_HELD:-}" ] && [ "$NO_LOCK" = 0 ]; then reexec=1; fi
    QUIET_GATES=$reexec
    if [ "$reexec" = 0 ]; then
        mkdir -p -- "$OUT_REL" "$LOG_DIR_REL" || die "$EXIT_ERROR" "cannot create $OUT_REL / $LOG_DIR_REL"
        if [ "$TIER" = smoke ]; then mkdir -p -- "$OUT_REL/smoke" || die "$EXIT_ERROR" "cannot create $OUT_REL/smoke"; fi
        LOG_FILE=$LOG_DIR_REL/trace_collect_${RUN_TS}_${TIER}.log
        : >"$LOG_FILE" || LOG_FILE=""
        log "pc/trace_collect.sh $VERSION: tier $TIER, $n probe run(s), model $MODEL_REL, out $OUT_REL, -t $THREADS -c $CTX -b $BATCH -ub $UBATCH, chunk $CHUNK, gen $GEN$( [ "$TIER" != smoke ] && [ "$NO_PROMPTS" = 0 ] && printf ', prompt gen %s' "$PROMPT_GEN"), probs-top $PROBS_TOP"
        log "python: $PY_BIN; repo: $REPO_POSIX; driver log: ${LOG_FILE:-none}"
    fi
    "$PY_BIN" -c 'import numpy' >/dev/null 2>&1 \
        || die "$EXIT_MISSING" "$PY_BIN cannot import numpy (needed by pigemma.trace)"
    check_inputs
    local lk_state lk_reason lk
    if [ -n "${PIGEMMA_LOCK_HELD:-}" ]; then
        gate "bench lock: held by this run's parent (PIGEMMA_LOCK_HELD)"
    elif [ "$NO_LOCK" = 1 ]; then
        warn "bench lock: NOT taken on the host (--no-lock); every probe run still takes it in its container"
    else
        lk=$(lock_state)
        lk_state=${lk%%|*}
        lk_reason=${lk#*|}
        if [ "$lk_state" = held ]; then
            refuse "the bench lock results/.bench.lock is held: $lk_reason - one model-loading job at a time"
        fi
    fi
    if [ "$IGNORE_BUSY" = 0 ]; then
        local busy
        busy=$(busy_check)
        if [ -n "$busy" ]; then
            refuse "another llama.cpp / probe job is running (--ignore-busy overrides): $(tr '\n' ';' <<<"$busy")"
        fi
        gate "no other llama.cpp / probe process or container is running"
    fi
    local free
    free=$(free_bytes "$OUT_REL")
    if is_uint "$free" && [ "$free" -lt "$MIN_FREE_DISK_BYTES" ]; then
        refuse "only $(fmt_gb "$free") free for $OUT_REL; need >= $(fmt_gb "$MIN_FREE_DISK_BYTES") for traces and stats"
    fi
    gate "free disk for $OUT_REL: $(fmt_gb "$free")"

    # ---- re-run under the host bench lock for the whole tier (D1)
    if [ "$reexec" = 1 ]; then
        log "taking the bench lock: python -m pigemma.paths lock --hold --min-free-gb $MIN_FREE_GB -- bash pc/trace_collect.sh ${ORIG_ARGS[*]}"
        local lrc=0
        TRACE_COLLECT_ORIG_PWD=$ORIG_PWD PYTHONUTF8=1 "$PY_BIN" -m pigemma.paths lock --hold \
            --min-free-gb "$MIN_FREE_GB" -- bash pc/trace_collect.sh "${ORIG_ARGS[@]}" || lrc=$?
        exit "$lrc"
    fi
    QUIET_GATES=0

    # ---- from here on: we hold the lock (or --no-lock)
    local ledger=$LOG_DIR_REL/trace_collect_${RUN_TS}_${TIER}.runs.log
    : >"$ledger" || die "$EXIT_ERROR" "cannot write $ledger"
    log "run ledger: $ledger"
    TMPD=$(mktemp -d "${TMPDIR:-/tmp}/pigemma-tc.XXXXXX") || die "$EXIT_ERROR" "mktemp -d failed"
    trap on_signal INT TERM

    # ---- preflight container: fingerprint, probe runs, model visible, VM memory
    local chost=$TMPD/container_host.json pre prc=0 kvs
    local preflight_sh
    IFS= read -r -d '' preflight_sh <<'SH' || true
probe=$1
model=$2
printf '@@HOST\n'
python -m pigemma.paths host 2>/dev/null || printf '{}\n'
printf '@@KV\n'
if [ -x "$probe" ]; then echo probe_exec=1; else echo probe_exec=0; fi
v=$("$probe" --version 2>&1 | grep -v '^$' | head -n 1) || v=""
echo "probe_version=$v"
if [ -f "$model" ]; then echo "model_size=$(stat -c %s "$model")"; else echo "model_size="; fi
awk '/^MemTotal:/ {print "mem_total_kb=" $2} /^MemAvailable:/ {print "mem_avail_kb=" $2}' /proc/meminfo
echo "nproc=$(nproc 2>/dev/null || echo 0)"
SH
    log "preflight container: fingerprint, probe binary, model visibility, VM memory"
    if [ -n "$MODEL_VOLUME" ]; then gate "model volume: the probe reads $(model_c) from Docker volume $MODEL_VOLUME (ro), not the bind mount (D14)"; fi
    pre=$(bash "$DOCKER_RUN" --no-stdin "${VOL_ARGS[@]}" -- sh -c "$preflight_sh" sh "$PROBE_C" "$(model_c)" 2>>"${LOG_FILE:-/dev/null}" | tr -d '\r') || prc=$?
    case $prc in
        0) ;;
        "$EXIT_REFUSED") refuse "docker is not available (pc/docker_run.sh exit 4; see $LOG_FILE)" ;;
        "$EXIT_MISSING") die "$EXIT_MISSING" "pc/docker_run.sh: image or repository layout missing (see $LOG_FILE)" ;;
        *) die "$EXIT_ERROR" "preflight container failed (exit $prc; see $LOG_FILE)" ;;
    esac
    awk '/^@@HOST$/ {f=1; next} /^@@KV$/ {f=0} f' <<<"$pre" >"$chost"
    kvs=$(sed -n '/^@@KV$/,$p' <<<"$pre")
    kvget() { sed -n "s/^$1=//p" <<<"$kvs" | head -n 1; }
    if [ "$(kvget probe_exec)" != 1 ]; then
        die "$EXIT_MISSING" "the probe $PROBE_C is not executable inside the container - build it with: bash pc/docker_run.sh build-tools"
    fi
    log "probe: $(kvget probe_version)"
    local csize
    csize=$(kvget model_size)
    if [ "$csize" != "$MODEL_SIZE" ]; then
        die "$EXIT_MISSING" "the container sees $(model_c) with '${csize:-no file}' bytes, the host $MODEL_SIZE (bind mount problem, or the --model-volume copy is missing/incomplete?)"
    fi
    local memkb
    memkb=$(kvget mem_total_kb)
    if is_uint "$memkb"; then
        local memav
        memav=$(kvget mem_avail_kb)
        if ! is_uint "$memav"; then memav=0; fi
        gate "Docker VM: $(fmt_gb $((memkb * 1024))) RAM, $(fmt_gb $((memav * 1024))) available, $(kvget nproc) CPUs"
        if [ $((memkb * 1024)) -lt $((MODEL_SIZE + 4000000000)) ]; then
            warn "the Docker VM RAM cannot keep the $(fmt_gb "$MODEL_SIZE") GGUF in its page cache: every probe run re-reads it through the bind mount"
        fi
    fi
    local chost_native=""
    if "$PY_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); sys.exit(0 if isinstance(d,dict) and d.get("hostname") else 1)' "$(native_path "$chost")" >/dev/null 2>&1; then
        chost_native=$(native_path "$chost")
    else
        warn "could not capture the container fingerprint; meta.json host = this machine"
    fi
    local bg=$TMPD/bgcpu.json bgs=${PIGEMMA_TRACE_BGCPU_S:-5}
    if [[ $bgs =~ ^[0-9]+([.][0-9]+)?$ ]] && [ "$bgs" != 0 ]; then
        hpy bgcpu --seconds "$bgs" --out "$(native_path "$bg")" | tr -d '\r' | while IFS= read -r l; do log "$l"; done || true
    fi

    # ---- stale outputs (D8)
    local -a skip=()
    for i in "${!ITEM_CID[@]}"; do
        skip[i]=0
        if [ "$RESUME" = 1 ]; then
            local why rrc=0
            why=$(hpy resume-ok --trace "${ITEM_TRACE[$i]}" --corpus "${ITEM_SRC[$i]}" --kind "${ITEM_KIND[$i]}" \
                --cid "${ITEM_CID[$i]}" --chunk "$CHUNK" \
                --gen "$( [ "${ITEM_KIND[$i]}" = prompts ] && printf '%s' "$PROMPT_GEN" || printf '%s' "$GEN")" \
                --probs "$PROBS_TOP" --gguf-size "$MODEL_SIZE" 2>>"${LOG_FILE:-/dev/null}" | tr -d '\r') || rrc=$?
            if [ "$rrc" = 0 ]; then
                skip[i]=1
                log "resume: reusing ${ITEM_TRACE[$i]}: $why"
            else
                log "resume: re-running ${ITEM_CID[$i]}: ${why:-not reusable}"
            fi
        fi
        if [ "${skip[i]}" = 0 ]; then rm_trace_outputs "${ITEM_TRACE[$i]}"; fi
    done
    if [ -n "$merged" ]; then
        if [ -e "$merged" ] || [ -e "$stats" ]; then log "removing the previous merged outputs ($merged + sidecars, $stats)"; fi
        rm_trace_outputs "$merged"
        if [ -e "$stats" ]; then rm -f -- "$stats"; fi
    fi

    if [ -z "$MODEL_VOLUME" ]; then
        warn "every probe run reads the $(fmt_gb "$MODEL_SIZE") GGUF through the /work bind mount at ~16-31 MB/s (8-15 min; Docker Desktop does not keep bind-mounted files in the VM page cache between containers, MEASURED 2026-10-05) - use --model-volume pigemma-models (D14)"
    fi

    # ---- probe runs
    local rc vrc mrc t0 t1 wall plog cname any_fail=0 n_ok=0
    local -a ok_traces=() ps cmd
    for i in "${!ITEM_CID[@]}"; do
        local cid=${ITEM_CID[$i]} trace=${ITEM_TRACE[$i]} src=${ITEM_SRC[$i]} kind=${ITEM_KIND[$i]}
        if [ "${skip[i]}" = 1 ]; then
            hpy runrec --ledger "$ledger" --item "$cid" --kind "$kind" --trace "$trace" --corpus "$src" --log "" \
                --rc 0 --validate-rc 0 --meta-rc 0 --host-wall-ms 0 --resumed 1 | tr -d '\r' | tee -a "${LOG_FILE:-/dev/null}" >&2 || true
            ok_traces+=("$trace")
            n_ok=$((n_ok + 1))
            continue
        fi
        plog=$LOG_DIR_REL/trace_collect_${RUN_TS}_${cid}.probe.log
        probe_cmd "$i"
        cname=$(container_name "$cid")
        cmd=(bash "$DOCKER_RUN" --no-stdin "${VOL_ARGS[@]}" --name "$cname" -- "${PROBE_CMD[@]}")
        log "[$((i + 1))/$n] $cid: $(quote_cmd bash "$DOCKER_RUN_SHOW" --no-stdin "${VOL_ARGS[@]}" --name "$cname") -- $(quote_cmd "${PROBE_CMD[@]}")"
        printf '%s\n' "$(quote_cmd bash "$DOCKER_RUN_SHOW" --no-stdin "${VOL_ARGS[@]}" --name "$cname" -- "${PROBE_CMD[@]}")" >"$plog"
        t0=$(now_ms)
        start_ticker "$cid"
        set +e
        "${cmd[@]}" 2>&1 | (trap '' INT; tee -a "$plog") | (trap '' INT; console_filter)
        ps=("${PIPESTATUS[@]}")
        set -e
        stop_ticker
        rc=${ps[0]}
        t1=$(now_ms)
        wall=$(summary_value "$plog" wall_ms)
        if [ -z "$wall" ]; then wall=$((t1 - t0)); fi
        vrc=-1
        mrc=-1
        if [ -f "$trace" ]; then
            vrc=0
            run_logged pym pigemma.trace validate "$trace" || vrc=$?
            mrc=0
            local -a meta_args=(pigemma.trace meta "$trace" --gguf "$MODEL_REL" --corpus "$src"
                                --summary-file "$plog" --wall-ms "$wall" --coherent auto
                                --note "collected by pc/trace_collect.sh $VERSION, tier $TIER, in Docker (${PIGEMMA_IMAGE:-pi-gemma4-build:latest}) through the /work bind mount with -nr"
                                --note "launched as: $(quote_cmd bash "$DOCKER_RUN_SHOW" --no-stdin "${VOL_ARGS[@]}" --name "$cname") -- $(quote_cmd "${PROBE_CMD[@]}")")
            if [ -n "$chost_native" ]; then meta_args+=(--host-json "$chost_native"); fi
            run_logged pym "${meta_args[@]}" || mrc=$?
        else
            warn "$cid: no trace was written ($trace)"
        fi
        hpy runrec --ledger "$ledger" --item "$cid" --kind "$kind" --trace "$trace" --corpus "$src" --log "$plog" \
            --rc "$rc" --validate-rc "$vrc" --meta-rc "$mrc" --host-wall-ms "$((t1 - t0))" \
            | tr -d '\r' | tee -a "${LOG_FILE:-/dev/null}" >&2 || warn "could not record the run in $ledger"
        if [ "$rc" = 0 ] && [ "$vrc" = 0 ] && [ "$mrc" = 0 ]; then
            ok_traces+=("$trace")
            n_ok=$((n_ok + 1))
        else
            any_fail=1
            warn "$cid failed: probe exit $rc, validate exit $vrc, meta exit $mrc (probe log: $plog)"
            if [ "$INTERRUPTED" = 1 ] || [ "$rc" -ge 128 ]; then
                log "interrupted: stopping (completed traces stay; re-run with --resume to continue)"
                exit 130
            fi
            if [ "$KEEP_GOING" = 0 ]; then
                log "stopping after the first failure (--keep-going continues); nothing was merged"
                exit "$EXIT_ERROR"
            fi
        fi
        if [ "$INTERRUPTED" = 1 ]; then
            log "interrupted: stopping (completed traces stay; re-run with --resume to continue)"
            exit 130
        fi
    done

    # ---- smoke: checks only
    if [ "$TIER" = smoke ]; then
        local smrc=0
        hpy smoke --trace "${ITEM_TRACE[0]}" --expect-records "$((CHUNK + GEN))" --expect-prefill "$CHUNK" \
            --golden "$GOLDEN" | tr -d '\r' | tee -a "${LOG_FILE:-/dev/null}" || smrc=$?
        printf 'outputs: %s (+ .tokens.bin .gen.txt .meta.json); logs: %s\n' "${ITEM_TRACE[0]}" "$LOG_FILE"
        if [ "$smrc" = 0 ]; then
            printf 'to refresh the golden fixture (a human decision, never done here):\n  cp %s tests/fixtures/golden_26b_72tok.moetrace; cp %s.tokens.bin tests/fixtures/golden_26b_72tok.moetrace.tokens.bin; cp %s.gen.txt tests/fixtures/golden_26b_72tok.moetrace.gen.txt\n' \
                "${ITEM_TRACE[0]}" "${ITEM_TRACE[0]}" "${ITEM_TRACE[0]}"
        fi
        if [ "$any_fail" = 1 ]; then exit "$EXIT_ERROR"; fi
        exit "$smrc"
    fi

    # ---- merge, validate, stats
    local mvrc="" step_fail=0
    if [ "$NO_MERGE" = 1 ]; then
        log "--no-merge: skipping merge, stats and the envelope"
        merged=""
        stats=""
        mvrc=skip
    elif [ "${#ok_traces[@]}" -eq 0 ]; then
        warn "no valid trace to merge"
        merged=""
        stats=""
        step_fail=1
    else
        log "merge: python -m pigemma.trace merge $merged ${ok_traces[*]}"
        if run_quiet pym pigemma.trace merge "$merged" "${ok_traces[@]}"; then
            mvrc=0
            run_logged pym pigemma.trace validate "$merged" || mvrc=$?
            log "stats: python -m pigemma.trace stats $merged --out $stats --csv-dir $CSV_DIR_REL"
            mkdir -p -- "$CSV_DIR_REL" || true
            if ! run_logged pym pigemma.trace stats "$merged" --out "$stats" --csv-dir "$CSV_DIR_REL"; then
                warn "stats failed (see $LOG_FILE)"
                stats=""
                step_fail=1
            fi
        else
            warn "merge failed (see $LOG_FILE)"
            merged=""
            stats=""
            step_fail=1
        fi
    fi

    # ---- checks, E3 preview, sizing, envelope
    local fields="" frc=0
    if [ "$NO_ENVELOPE" = 0 ] && [ "$NO_MERGE" = 0 ] && [ -n "$merged" ]; then fields=$TMPD/e3_fields.json; fi
    local -a flags=(-nr -t "$THREADS" -c "$CTX" -b "$BATCH" -ub "$UBATCH" --chunk-tokens "$CHUNK" --gen "$GEN")
    if [ "$NO_PROMPTS" = 0 ]; then flags+=(--prompt-gen "$PROMPT_GEN"); fi
    if [ "$PROBS_TOP" -gt 0 ]; then flags+=(--probs-top "$PROBS_TOP"); fi
    flags+=("${PROBE_EXTRA[@]}")
    hpy finalize --tier "$TIER" --ledger "$ledger" --merged "$merged" --merged-validate-rc "$mvrc" \
        --stats "$stats" --min-prefill "$MIN_PREFILL" --min-gen "$MIN_GEN" --chunk "$CHUNK" --gen "$GEN" \
        --prompt-gen "$PROMPT_GEN" --full-chunk "$FULL_CHUNK" --full-gen "$FULL_GEN" \
        --fields-out "$( [ -n "$fields" ] && native_path "$fields")" --model-rel "$MODEL_REL" \
        --host-json "$chost_native" --threads "$THREADS" --ctx "$CTX" --flags "$(quote_cmd "${flags[@]}")" \
        --bgcpu "$(native_path "$bg")" --version "$VERSION" --log-file "$LOG_FILE" \
        --image "${PIGEMMA_IMAGE:-pi-gemma4-build:latest}" | tr -d '\r' | tee -a "${LOG_FILE:-/dev/null}" || frc=$?
    if [ "$frc" != 0 ] && [ "$frc" != "$EXIT_CHECKS" ]; then
        warn "finalize step failed (exit $frc)"
        step_fail=1
    fi
    local envelope=""
    if [ -n "$fields" ] && [ -f "$fields" ]; then
        mkdir -p -- "$RESULTS_DIR_REL" || true
        local wout erc=0
        wout=$(pym pigemma.results write --out "$RESULTS_DIR_REL/" --fields "$(native_path "$fields")" 2>>"${LOG_FILE:-/dev/null}" | tr -d '\r') || erc=$?
        if [ "$erc" = 0 ]; then
            envelope=$(tail -n 1 <<<"$wout")
            local bsl='\'
            envelope=${envelope//"$bsl"/\/}
        else
            warn "writing the E3 envelope failed (exit $erc; see $LOG_FILE)"
            step_fail=1
        fi
    fi

    printf '\noutputs:\n'
    for i in "${ok_traces[@]}"; do printf '  %s (+ .tokens.bin .gen.txt .meta.json)\n' "$i"; done
    if [ -n "$merged" ]; then printf '  %s (+ .tokens.bin .gen.txt .meta.json)\n' "$merged"; fi
    if [ -n "$stats" ]; then printf '  %s\n  %s/%s.*.csv\n' "$stats" "$CSV_DIR_REL" "$(basename -- "${merged%.moetrace}")"; fi
    if [ -n "$envelope" ]; then printf '  %s\n' "$envelope"; fi
    printf '  logs: %s, %s, %s/trace_collect_%s_*.probe.log\n' "$LOG_FILE" "$ledger" "$LOG_DIR_REL" "$RUN_TS"

    if [ "$any_fail" = 1 ] || [ "$step_fail" = 1 ]; then exit "$EXIT_ERROR"; fi
    if [ "$frc" = "$EXIT_CHECKS" ]; then
        log "traces written, but an acceptance check failed (exit $EXIT_CHECKS; see the CHECK lines)"
        exit "$EXIT_CHECKS"
    fi
    log "done: tier $TIER, $n_ok run(s), all checks passed"
    exit "$EXIT_OK"
}

dry_run_plan() {
    local merged=$1 stats=$2 i
    local lk free busy
    printf '# pc/trace_collect.sh %s --%s (DRY RUN: nothing is run, nothing is written)\n' "$VERSION" "$TIER"
    printf '# model %s (%s), -t %s -c %s -b %s -ub %s, chunk %s, gen %s per chunk' \
        "$MODEL_REL" "${MODEL_ID:-?}" "$THREADS" "$CTX" "$BATCH" "$UBATCH" "$CHUNK" "$GEN"
    if [ "$TIER" != smoke ] && [ "$NO_PROMPTS" = 0 ]; then printf ', %s per gen prompt' "$PROMPT_GEN"; fi
    printf ', probs-top %s\n' "$PROBS_TOP"
    check_inputs
    if [ -n "${PIGEMMA_LOCK_HELD:-}" ]; then
        printf '# bench lock: inherited (PIGEMMA_LOCK_HELD)\n'
    elif [ "$NO_LOCK" = 1 ]; then
        printf '# bench lock: not taken on the host (--no-lock)\n'
    else
        lk=$(lock_state)
        printf '# bench lock: %s\n' "${lk%%|*} (${lk#*|})"
        if [ "${lk%%|*}" = held ]; then gate_fail "$EXIT_REFUSED" "the bench lock is held: ${lk#*|}"; fi
        local a
        local -a real_args=()
        for a in "${ORIG_ARGS[@]}"; do
            case $a in --dry-run|-n) ;; *) real_args+=("$a") ;; esac
        done
        printf '# a real run re-executes itself under: python -m pigemma.paths lock --hold --min-free-gb %s -- bash pc/trace_collect.sh %s\n' \
            "$MIN_FREE_GB" "$(quote_cmd "${real_args[@]}")"
    fi
    if [ "$IGNORE_BUSY" = 0 ]; then
        busy=$(busy_check)
        if [ -n "$busy" ]; then gate_fail "$EXIT_REFUSED" "another llama.cpp / probe job is running: $(tr '\n' ';' <<<"$busy")"; fi
    fi
    free=$(free_bytes "$OUT_REL")
    printf '# free disk for %s: %s\n' "$OUT_REL" "$(fmt_gb "$free")"
    printf '# %s probe run(s); per run: the docker command, then validate + meta on the host\n' "${#ITEM_CID[@]}"
    for i in "${!ITEM_CID[@]}"; do
        probe_cmd "$i"
        printf '%s\n' "$(quote_cmd bash "$DOCKER_RUN_SHOW" --no-stdin "${VOL_ARGS[@]}" --name "$(container_name "${ITEM_CID[$i]}")" -- "${PROBE_CMD[@]}")"
        printf '%s\n' "$(quote_cmd python -m pigemma.trace validate "${ITEM_TRACE[$i]}")"
        printf '%s --summary-file %s --host-json %s --wall-ms %s --coherent auto\n' \
            "$(quote_cmd python -m pigemma.trace meta "${ITEM_TRACE[$i]}" --gguf "$MODEL_REL" --corpus "${ITEM_SRC[$i]}")" \
            "$LOG_DIR_REL/trace_collect_<UTC>_${ITEM_CID[$i]}.probe.log" '<container-fingerprint.json>' '<probe-wall-ms>'
    done
    if [ "$TIER" = smoke ]; then
        printf '# smoke checks: validate clean, %s records = %s prefill + %s generated, every MoE layer present; golden comparison with %s (info)\n' \
            "$((CHUNK + GEN))" "$CHUNK" "$GEN" "$GOLDEN"
    elif [ "$NO_MERGE" = 1 ]; then
        printf '# --no-merge: no merge, stats or envelope\n'
    else
        local -a traces=()
        for i in "${!ITEM_TRACE[@]}"; do traces+=("${ITEM_TRACE[$i]}"); done
        printf '%s\n' "$(quote_cmd python -m pigemma.trace merge "$merged" "${traces[@]}")"
        printf '%s\n' "$(quote_cmd python -m pigemma.trace validate "$merged")"
        printf '%s\n' "$(quote_cmd python -m pigemma.trace stats "$merged" --out "$stats" --csv-dir "$CSV_DIR_REL")"
        if [ "$NO_ENVELOPE" = 0 ]; then
            printf '%s --fields %s\n' "$(quote_cmd python -m pigemma.results write --out "$RESULTS_DIR_REL/")" '<e3-envelope-fields.json>'
        fi
        printf '# checks: every run valid, merged valid, >= %s prefill and >= %s generated records, every generated chunk coherent, all MoE layers\n' \
            "$MIN_PREFILL" "$MIN_GEN"
    fi
    if [ "${#GATE_PROBLEMS[@]}" -gt 0 ]; then
        printf '# gates that would stop a real run:\n'
        local p
        for p in "${GATE_PROBLEMS[@]}"; do printf '#   exit %s: %s\n' "${p%%|*}" "${p#*|}"; done
    else
        printf '# all gates pass\n'
    fi
}

main
