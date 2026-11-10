#!/usr/bin/env bash
# pi/02_build_llama.sh - native llama.cpp (+ optional pigemma tools) build on the Pi at the pinned commit.
#
# Step 2 of the Pi runtime (after pi/00_feasibility.sh and pi/01_tune_os.sh). RAM-aware, refuses on
# protected hosts, never installs packages, never stops/kills/restarts anything, never reboots.
#
# ------------------------------------------------------------------------------------------------ usage
#   pi/02_build_llama.sh [--commit a7b94df] [--jobs N] [--with-tools]
#                        [--from-desktop user@host:/path/to/pi-gemma4/third_party/llama.cpp]
#                        [--arch auto|native|pi5|generic] [--targets a,b,...] [--repo-url URL]
#                        [--tools-build-dir DIR] [--rebuild] [--clean-objects] [--min-free-gb 3]
#                        [--min-mem-gb 1.5] [--ignore-lock] [--configure-only] [--dry-run] [--verbose]
#                        [--json-out FILE] | -h/--help | --version
#   Long builds on a Pi over flaky Wi-Fi SSH: run detached and poll the log, e.g.
#       nohup pi/02_build_llama.sh --with-tools > pi/out/build.out 2>&1 &
#
# ------------------------------------------------------------------------------------------------ steps
#  1. pigemma_init (pi/pi_config.env + environment) and assert_safe_host: exit 4 on a protected host
#     unless PIGEMMA_I_KNOW=1 (context-friend-1: authorised; the build then still runs under nice).
#  2. Preconditions (only when something has to be fetched or compiled): cmake, gcc, g++, make (or ninja
#     when CMAKE_GENERATOR / the existing cache says Ninja) and git on PATH (apt hint otherwise, exit 3);
#     free disk >= --min-free-gb (default 3 GB) where LLAMA_DIR (and the tools build dir) live - on the
#     ROOT filesystem require_free_disk adds KEEP_FREE_GB (default 2 GB) on top, the FACTS rule "keep
#     >= 2 GB free on / at all times"; effective MemAvailable >= --min-mem-gb (default 1.5 GB);
#     results/.bench.lock not held by a model-loading job (pigemma.paths; --ignore-lock overrides);
#     one build at a time ($PIGEMMA_OUT/.build.lock).
#  3. Source at the commit in $LLAMA_DIR (default ~/llama.cpp):
#       * already a llama.cpp git checkout at the commit: used as is (no network);
#       * missing/empty dir and the FULL hash is known (the pinned a7b94df expands to
#         a7b94df2c616bc1f62a73b964b4a71cb0dcc488e, or a 40-hex --commit): git init + git fetch --depth 1
#         origin <full> + checkout (~40 MB, the same recipe that worked on the real Pi);
#       * otherwise (a short unpinned hash, or the depth-1 fetch failed): the manifest recipe
#         git clone --filter=blob:none <url> $LLAMA_DIR + checkout <commit>;
#       * an existing checkout at another commit: fetched + checked out (refused if it has local
#         modifications);
#       * --from-desktop SRC: rsync -a --delete (build dirs excluded and protected) from user@host:/path
#         over ssh (BatchMode, 20 s connect timeout; PIGEMMA_RSYNC_RSH overrides), or from a local path
#         (rsync, else a tar pipe). The copied tree's git HEAD must match --commit (exit 3 otherwise).
#     Network steps (git fetch/clone, rsync) run under nice + ionice -c3 and are retried 3 times.
#     A non-llama.cpp directory is never overwritten (exit 3).
#  4. Configure (manifest command):
#       cmake -S $LLAMA_DIR -B $LLAMA_DIR/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
#             -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF
#     --arch auto (default): native, falling back ONCE to the Pi 5 ISA
#       -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
#     when (a) native configure fails, (b) native detection enabled i8mm/sve/sme that /proc/cpuinfo does
#     not list, (c) the build fails with an arch-flag error, or (d) the built binary dies with SIGILL.
#     Under emulation (PIGEMMA_EMULATED=1, or aarch64 /proc/cpuinfo with CPU implementer 0x00 = qemu)
#     auto goes straight to the Pi 5 ISA: native detection there sees qemu's "max" CPU and adds
#     +i8mm+sve+sme, i.e. binaries that SIGILL on a real Pi 5 (MEASURED in pi-gemma4-pi-emu:trixie).
#     On x86 hosts the fallback is generic (-DGGML_NATIVE=OFF); pi5 is aarch64-only.
#  5. Build: nice -n $NICE_LEVEL cmake --build $LLAMA_DIR/build -j $(pick_jobs) --target llama-cli
#     llama-completion llama-bench llama-perplexity llama-server
#     (jobs = min(nproc, floor(MemAvailable_GiB/1.2), 4): 2 on a 4 GB board, 4 on 16 GB; --jobs N wins,
#     with a warning above the RAM-safe value). Compiler output goes to $PIGEMMA_OUT/build-logs/*.log
#     with a progress line every 30 s on stderr (--verbose streams everything).
#     Skipped when the binaries already exist, report the checked-out commit and were built with an
#     acceptable arch mode (--rebuild forces; important after --clean-objects, where make would
#     otherwise recompile everything).
#  6. --with-tools: cmake -S $PIGEMMA_HOME/tools -B $PIGEMMA_HOME/tools/build -DLLAMA_DIR=$LLAMA_DIR
#     <same arch flags as the llama build> -DCMAKE_BUILD_TYPE=Release && cmake --build ... --target
#     pigemma-probe pinner membw (only tools whose source exists). Skipped when up to date (binaries
#     newer than their sources, same commit, same LLAMA_DIR, same arch mode).
#  7. Prints the binary paths, the version line and `llama-bench --help | head -1` (stdout) and records
#     everything in $PIGEMMA_OUT/build.json (also written, with status "failed", when a step fails).
#
# ------------------------------------------------------------------------------------------- exit codes
#   0 built / up to date / configured (--configure-only) / dry run
#   1 usage error, or a failed step (git, rsync, configure, build, binary does not run)
#   3 precondition failed: missing tool, disk, RAM, unusable or foreign LLAMA_DIR, commit not found in
#     a shallow checkout, local modifications in the way, desktop tree at another commit
#   4 refused: protected host (without PIGEMMA_I_KNOW=1), bench lock held, another build running
#
# ---------------------------------------------------------------------------------------- build.json
#   {format_version:1, tool, tool_version, created_at, status ('ok'|'configured'|'failed'), error,
#    argv[], host{hostname, kernel, machine, page_size, nproc, ram_total_kb, ram_available_kb,
#    cpu_model, cpu_implementer, cpu_part, cpu_features[], in_container, emulated, qemu_suspected,
#    protected_override, fake_proc},
#    llama{dir, repo_url, from_desktop, requested_commit, commit, commit_short, verified, shallow, dirty,
#    source_method, source_changed, build_dir, bin_dir, arch{requested, mode, fallback_used,
#    fallback_reason, cmake_flags[], arm_detected_flags, effective_flags[], machine_supports{dotprod,
#    i8mm, sve, sme}}, build_number_override, configure_args[], generator, jobs, rebuilt, targets[{name,
#    path, exists, bytes}], version, build_number, version_commit, llama_bench_help, seconds{source,
#    configure, build}, logs{}},
#    tools: null | {source_dir, build_dir, bin_dir, rebuilt, seconds, targets[...], membw_version,
#    pinner_version, probe_help, build_info (the CMake-written pigemma-tools-build.json, embedded)},
#    toolchain{gcc, gxx, cmake, git, make, ninja}, disk{path, free_before_bytes, free_after_bytes,
#    source_bytes, llama_build_bytes, tools_build_bytes, objects_cleaned, objects_freed_bytes},
#    config{nice_level, keep_free_gb, min_free_gb, min_mem_gb, build_jobs}, warnings[], notes[]}
#
# ----------------------------------------------------------------------------------- design decisions
# * Depth-1 fetch of the full hash instead of the manifest's blob:none clone when possible: a blob:none
#   clone still downloads every commit + tree of llama.cpp over the Pi's ~10 MB/s, packet-dropping
#   Wi-Fi; the manifest recipe stays as the fallback. The manifest's extra `git fetch --depth 1 origin
#   $COMMIT` after a blob:none clone is not run: GitHub rejects abbreviated hashes, and a depth-1 fetch
#   into a full clone would make it shallow (breaking rev-list --count). A missing commit is fetched
#   without --depth instead.
# * Build number: a shallow clone makes llama.cpp report "build 1". When HEAD is exactly the pinned
#   commit in a shallow clone, -DLLAMA_BUILD_NUMBER=11398 is passed so llama-bench/llama-cli report the
#   same b11398 as the desktop binaries (tools/CMakeLists.txt makes the same choice for the probe);
#   otherwise -ULLAMA_BUILD_NUMBER clears any stale cached value. The commit is never overridden.
# * Disk budget: 3 GB is the spec's work budget (shallow source ~0.2 GB + static build of 5 targets
#   + the tools tree, which compiles llama/ggml a second time); on / the KEEP_FREE_GB reserve is added.
#   The measured footprint is recorded in build.json (disk.*) to calibrate this later.
# * Nothing to do (everything up to date) skips the disk/RAM gates: re-running pi/02 on a full SD card
#   only refreshes build.json.
# * --dry-run applies every check (exit codes as a real run) and prints the commands it would run as
#   '+ cmd ...' lines on stdout; it fetches, builds and writes nothing (no build.json, no lock).
# * Test hooks: PIGEMMA_MACHINE overrides `uname -m` for the arch decision; PIGEMMA_HEARTBEAT_S sets
#   the progress interval (default 30); LLAMA_REPO_URL sets the default --repo-url.
#
# shellcheck shell=bash
set -euo pipefail

PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
. "$PI_DIR/lib/common.sh"

B_VERSION="1.0.0"
PINNED_FULL="a7b94df2c616bc1f62a73b964b4a71cb0dcc488e" # = build b11398 (FACTS.md, third_party/llama.cpp HEAD)
PINNED_BUILD=11398
PI5_ARM_ARCH="armv8.2-a+dotprod+fp16" # Cortex-A76: dotprod + fp16, NOT i8mm/SVE/BF16
DEFAULT_TARGETS="llama-cli,llama-completion,llama-bench,llama-perplexity,llama-server"
TOOL_TARGETS=(pigemma-probe pinner membw)
DEFAULT_REPO_URL="https://github.com/ggml-org/llama.cpp"
NET_RETRIES=3
NET_RETRY_DELAY_S=10

# C0 control characters other than \t \n \r (dropped from JSON strings; literal list, locale independent)
_B_CTRL=$'\001\002\003\004\005\006\007\010\013\014\016\017\020\021\022\023\024\025\026\027\030\031\032\033\034\035\036\037'

usage() {
    cat <<EOF
pi/02_build_llama.sh $B_VERSION - native llama.cpp (+ pigemma tools) build at the pinned commit.

Usage: pi/02_build_llama.sh [options]
  --commit C            llama.cpp commit, 7-40 hex digits (default LLAMA_COMMIT=${LLAMA_COMMIT:-a7b94df})
  --jobs N              parallel compile jobs (default: min(nproc, floor(MemAvailable_GiB/1.2), 4))
  --with-tools          also build pigemma-probe, pinner, membw from \$PIGEMMA_HOME/tools
  --from-desktop SRC    copy the llama.cpp tree from SRC instead of git: user@host:/path (rsync over ssh)
                        or a local directory; build dirs are excluded
  --arch MODE           auto (default: native, fallback Pi 5 ISA; Pi 5 ISA under emulation) | native |
                        pi5 (-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=$PI5_ARM_ARCH) | generic
  --targets LIST        llama.cpp targets (default $DEFAULT_TARGETS)
  --repo-url URL        git remote for a fresh checkout (default \$LLAMA_REPO_URL or $DEFAULT_REPO_URL)
  --tools-build-dir D   tools build dir (default \$PIGEMMA_HOME/tools/build; later scripts look in
                        \$PIGEMMA_TOOLS_BIN or tools/build/bin)
  --rebuild             configure + build even when the binaries are up to date
  --clean-objects       delete *.o files after a successful build (saves disk; the next --rebuild is full)
  --min-free-gb N       disk work budget in GB (default 3; on / KEEP_FREE_GB=${KEEP_FREE_GB:-2} is added)
  --min-mem-gb X        minimum effective MemAvailable in GB (default 1.5)
  --ignore-lock         build even while results/.bench.lock is held (distorts that benchmark)
  --configure-only      fetch + configure (shows the detected CPU flags), no compile
  --dry-run             run every check, print the commands, change nothing
  --verbose             stream compiler output to stderr (default: log files + progress every 30 s)
  --json-out FILE       build record (default \$PIGEMMA_OUT/build.json)
  -h, --help            this help;  --version  print the version

Configuration: pi/pi_config.env (LLAMA_DIR=${LLAMA_DIR:-~/llama.cpp}, NICE_LEVEL, KEEP_FREE_GB, BUILD_JOBS,
PIGEMMA_OUT ...); environment values win. CMAKE_GENERATOR=Ninja selects ninja (default: Unix Makefiles).
Exit codes: 0 ok | 1 usage error or failed step | 3 precondition failed (tool, disk, RAM, source) |
            4 refused (protected host, bench lock held, concurrent build).
Example: PIGEMMA_I_KNOW=1 nohup pi/02_build_llama.sh --with-tools > pi/out/build.out 2>&1 &
EOF
}

# =========================================================================================== arguments
ORIG_ARGS=("$@")
COMMIT=""
JOBS_ARG=""
WITH_TOOLS=0
FROM=""
REPO_URL=${LLAMA_REPO_URL:-$DEFAULT_REPO_URL}
ARCH_REQ=auto
TARGETS_CSV=$DEFAULT_TARGETS
TOOLS_BUILD=""
REBUILD=0
CLEAN_OBJ=0
MIN_FREE_GB=3
MIN_MEM_GB=1.5
IGNORE_LOCK=0
DRY=0
CONFIGURE_ONLY=0
VERBOSE=0
JSON_OUT=""

while [ $# -gt 0 ]; do
    opt=$1
    val=""
    has_val=0
    case $opt in
        --*=*)
            val=${opt#*=}
            opt=${opt%%=*}
            has_val=1
            ;;
    esac
    case $opt in
        -h | --help)
            usage
            exit 0
            ;;
        --version)
            printf 'pi/02_build_llama.sh %s\n' "$B_VERSION"
            exit 0
            ;;
        --commit | --jobs | --from-desktop | --repo-url | --arch | --targets | --tools-build-dir | \
            --min-free-gb | --min-mem-gb | --json-out)
            if [ "$has_val" = 0 ]; then
                if [ $# -lt 2 ]; then die 1 "option $opt needs a value (see --help)"; fi
                val=$2
                shift
            fi
            if [ -z "$val" ]; then die 1 "option $opt needs a non-empty value (see --help)"; fi
            case $opt in
                --commit) COMMIT=$val ;;
                --jobs) JOBS_ARG=$val ;;
                --from-desktop) FROM=$val ;;
                --repo-url) REPO_URL=$val ;;
                --arch) ARCH_REQ=$val ;;
                --targets) TARGETS_CSV=$val ;;
                --tools-build-dir) TOOLS_BUILD=$val ;;
                --min-free-gb) MIN_FREE_GB=$val ;;
                --min-mem-gb) MIN_MEM_GB=$val ;;
                --json-out) JSON_OUT=$val ;;
            esac
            ;;
        --with-tools | --rebuild | --clean-objects | --ignore-lock | --configure-only | --dry-run | --verbose)
            if [ "$has_val" = 1 ]; then die 1 "option $opt takes no value (see --help)"; fi
            case $opt in
                --with-tools) WITH_TOOLS=1 ;;
                --rebuild) REBUILD=1 ;;
                --clean-objects) CLEAN_OBJ=1 ;;
                --ignore-lock) IGNORE_LOCK=1 ;;
                --configure-only) CONFIGURE_ONLY=1 ;;
                --dry-run) DRY=1 ;;
                --verbose) VERBOSE=1 ;;
            esac
            ;;
        --)
            shift
            if [ $# -gt 0 ]; then die 1 "unexpected arguments: $* (see --help)"; fi
            break
            ;;
        *) die 1 "unknown option or argument '$opt' (see --help)" ;;
    esac
    shift
done

# --------------------------------------------------------------------------------------- validation
COMMIT=${COMMIT:-$LLAMA_COMMIT}
COMMIT=${COMMIT,,}
if ! [[ $COMMIT =~ ^[0-9a-f]{7,40}$ ]]; then
    die 1 "--commit must be 7-40 hex digits, got '$COMMIT'"
fi
if [ -n "$JOBS_ARG" ]; then
    if ! [[ $JOBS_ARG =~ ^[0-9]+$ ]] || [ "$JOBS_ARG" -lt 1 ]; then
        die 1 "--jobs must be a positive integer, got '$JOBS_ARG'"
    fi
    BUILD_JOBS=$JOBS_ARG
    export BUILD_JOBS
fi
case $ARCH_REQ in
    auto | native | pi5 | generic) ;;
    *) die 1 "--arch must be auto, native, pi5 or generic, got '$ARCH_REQ'" ;;
esac
if ! [[ $MIN_FREE_GB =~ ^[0-9]+([.][0-9]+)?$ ]]; then die 1 "--min-free-gb must be a number, got '$MIN_FREE_GB'"; fi
if ! [[ $MIN_MEM_GB =~ ^[0-9]+([.][0-9]+)?$ ]]; then die 1 "--min-mem-gb must be a number, got '$MIN_MEM_GB'"; fi
TARGETS=()
_tgts=()
IFS=',' read -r -a _tgts <<<"$TARGETS_CSV" || true
for _t in "${_tgts[@]}"; do
    _t=${_t//[[:space:]]/}
    if [ -z "$_t" ]; then continue; fi
    if ! [[ $_t =~ ^[A-Za-z0-9_.+-]+$ ]]; then die 1 "--targets: invalid target name '$_t'"; fi
    TARGETS+=("$_t")
done
if [ "${#TARGETS[@]}" -eq 0 ]; then die 1 "--targets: empty target list"; fi

# ============================================================================================ helpers
WARNINGS=()
NOTES=()
bwarn() {
    WARNINGS+=("$*")
    warn "$*"
}
bnote() {
    NOTES+=("$*")
    log "$*"
}

_abspath() { # _abspath PATH -> absolute path (no symlink resolution, no existence check), no trailing /
    local p=$1
    case $p in
        "~") p=${HOME:-/root} ;;
        "~/"*) p="${HOME:-/root}/${p#\~/}" ;;
        /* | [A-Za-z]:[/\\]*) ;;
        *) p="$PWD/$p" ;;
    esac
    while [ "${#p}" -gt 1 ] && [ "${p%/}" != "$p" ]; do p=${p%/}; done
    printf '%s\n' "$p"
}

_tmo() { # _tmo SECONDS CMD... (coreutils timeout when present)
    local s=$1
    shift
    if have_cmd timeout; then timeout "$s" "$@"; else "$@"; fi
}

_is_llama_tree() { [ -f "$1/CMakeLists.txt" ] && [ -f "$1/include/llama.h" ]; }
_is_empty_dir() { [ -d "$1" ] && [ -z "$(ls -A "$1" 2>/dev/null)" ]; }
_in_list() {
    local x=$1 y
    shift
    for y in "$@"; do
        if [ "$x" = "$y" ]; then return 0; fi
    done
    return 1
}

_bytes_of() { # _bytes_of PATH -> apparent bytes (du -sb, else du -sk x 1024); empty when unknown
    local p=$1 out=""
    if [ ! -e "$p" ]; then
        printf '\n'
        return 0
    fi
    out=$(du -sb "$p" 2>/dev/null | awk 'NR==1 { print $1 }') || out=""
    if ! [[ $out =~ ^[0-9]+$ ]]; then
        out=$(du -sk "$p" 2>/dev/null | awk 'NR==1 { printf "%.0f", $1 * 1024 }') || out=""
    fi
    if ! [[ $out =~ ^[0-9]+$ ]]; then out=""; fi
    printf '%s\n' "$out"
}

_lines() { # number of lines of a file, 0 when absent
    local n=0
    if [ -f "$1" ]; then n=$(wc -l <"$1" | tr -d '[:space:]') || n=0; fi
    printf '%s\n' "${n:-0}"
}

_file_size() { # bytes of a regular file, empty when absent
    if [ -f "$1" ]; then wc -c <"$1" | tr -d '[:space:]'; else printf '\n'; fi
}

_first_line() { # _first_line CMD... -> first output line (stdout+stderr), empty on failure
    local out=""
    out=$(_tmo 20 "$@" 2>&1) || true
    printf '%s\n' "${out%%$'\n'*}"
}

_gb() { awk -v g="$1" 'BEGIN { printf "%.0f", g * 1e9 }'; }

# -------------------------------------------------------------------------------------------- JSON
_js() { # JSON string literal
    local s=${1-}
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\t'/\\t}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//[$_B_CTRL]/}
    printf '"%s"' "$s"
}
_jsn() { if [ -n "${1-}" ]; then _js "$1"; else printf 'null'; fi; }
_jn() { # JSON number (integer with leading zeros removed, or decimal) or null
    local v=${1-}
    if [[ $v =~ ^[0-9]+$ ]]; then
        v=${v#"${v%%[!0]*}"}
        printf '%s' "${v:-0}"
    elif [[ $v =~ ^-?[0-9]+[.][0-9]+$ ]]; then
        printf '%s' "$v"
    else
        printf 'null'
    fi
}
_jb() {
    case ${1-} in
        1 | true | TRUE | ON | on | yes) printf 'true' ;;
        0 | false | FALSE | OFF | off | no) printf 'false' ;;
        *) printf 'null' ;;
    esac
}
_ja() { # JSON array of strings
    local first=1 x
    printf '['
    for x in "$@"; do
        if [ "$first" = 0 ]; then printf ', '; fi
        first=0
        _js "$x"
    done
    printf ']'
}

# ---------------------------------------------------------------------------------- command runner
_x() { # _x LOGFILE CMD... : run CMD with output appended to LOGFILE (or streamed with --verbose)
    local logf=$1 rc=0
    shift
    if [ "$DRY" = 1 ]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    {
        printf '\n$'
        printf ' %q' "$@"
        printf '\n'
    } >>"$logf" 2>/dev/null || true
    if [ "$VERBOSE" = 1 ]; then
        "$@" > >(tee -a "$logf" >&2) 2>&1 || rc=$?
    else
        "$@" >>"$logf" 2>&1 || rc=$?
    fi
    return "$rc"
}

_retry() { # _retry N DELAY_S CMD... (network steps)
    local n=$1 d=$2 i=1 rc=0
    shift 2
    while :; do
        rc=0
        "$@" || rc=$?
        if [ "$rc" = 0 ]; then return 0; fi
        if [ "$i" -ge "$n" ]; then return "$rc"; fi
        warn "attempt $i/$n failed (exit $rc); retrying in ${d} s"
        sleep "$d"
        i=$((i + 1))
    done
}

_tail_log() { # _tail_log FILE [N] -> last lines on stderr
    local f=$1 n=${2:-30}
    if [ -s "$f" ]; then
        printf -- '----- last %s lines of %s -----\n' "$n" "$f" >&2
        tail -n "$n" "$f" >&2 || true
        printf -- '-----\n' >&2
    fi
}

HB_PID=""
_hb_start() { # _hb_start LOGFILE LABEL: progress line on stderr every PIGEMMA_HEARTBEAT_S (default 30) s
    if [ "$VERBOSE" = 1 ] || [ "$DRY" = 1 ]; then return 0; fi
    local logf=$1 label=$2 iv=${PIGEMMA_HEARTBEAT_S:-30} t0=$SECONDS
    if ! [[ $iv =~ ^[0-9]+$ ]] || [ "$iv" -lt 1 ]; then iv=30; fi
    # The subshell must not keep the build lock (fd 9, flock) alive: it closes fd 9, and on TERM it kills
    # its own sleep, so no orphaned `sleep` holds the lock (or anything else) after pi/02 exits. Without
    # this a second pi/02 started within PIGEMMA_HEARTBEAT_S of a finished build was refused (exit 4).
    (
        trap - EXIT
        exec 9>&-
        _hb_sp=""
        trap 'if [ -n "$_hb_sp" ]; then kill "$_hb_sp" 2>/dev/null || true; fi; exit 0' TERM
        while :; do
            sleep "$iv" </dev/null >/dev/null 2>&1 &
            _hb_sp=$!
            wait "$_hb_sp" 2>/dev/null || exit 0
            _hb_sp=""
            last=$(grep -aE '^\[ *[0-9]+%\]|^\[[0-9]+/[0-9]+\]' "$logf" 2>/dev/null | tail -n 1 || true)
            log "$label: running for $((SECONDS - t0)) s${last:+ - ${last:0:150}}"
        done
    ) &
    HB_PID=$!
}
_hb_stop() {
    if [ -n "${HB_PID:-}" ]; then
        kill "$HB_PID" 2>/dev/null || true # our own progress subshell, nothing else
        wait "$HB_PID" 2>/dev/null || true
        HB_PID=""
    fi
    return 0
}
add_cleanup "_hb_stop"

# ============================================================================================ setup
pigemma_init
assert_safe_host
PROTECTED_OVERRIDE=${PIGEMMA_PROTECTED_OVERRIDE:-0}

LLAMA_DIR=$(_abspath "$LLAMA_DIR")
export LLAMA_DIR
BUILD_DIR="$LLAMA_DIR/build"
BIN_DIR="$BUILD_DIR/bin"
TOOLS_SRC="$PIGEMMA_HOME/tools"
TOOLS_BUILD=$(_abspath "${TOOLS_BUILD:-$TOOLS_SRC/build}")
TOOLS_BIN="$TOOLS_BUILD/bin"
JSON_OUT=$(_abspath "${JSON_OUT:-$PIGEMMA_OUT/build.json}")
LOG_DIR="$PIGEMMA_OUT/build-logs"
LOG_SRC="$LOG_DIR/source.log"
LOG_CFG="$LOG_DIR/llama-configure.log"
LOG_BLD="$LOG_DIR/llama-build.log"
LOG_TCFG="$LOG_DIR/tools-configure.log"
LOG_TBLD="$LOG_DIR/tools-build.log"

_HOME_ABS=$(_abspath "${HOME:-/root}")
_REPO_ABS=$(_abspath "$PIGEMMA_HOME")
case $LLAMA_DIR in
    / | "$_HOME_ABS" | "$_REPO_ABS" | "$_REPO_ABS/pi" | "$_REPO_ABS/tools" | "$_REPO_ABS/models")
        die 1 "LLAMA_DIR='$LLAMA_DIR' is not a sensible llama.cpp checkout location (default ~/llama.cpp)"
        ;;
esac
if [ "$LLAMA_DIR" = "$_REPO_ABS/third_party/llama.cpp" ]; then
    bwarn "LLAMA_DIR is the repo's desktop reference checkout third_party/llama.cpp: the build dir lands in third_party/llama.cpp/build"
fi
if [ -n "${PIGEMMA_LLAMA_BIN:-}" ] && [ "$(_abspath "$PIGEMMA_LLAMA_BIN")" != "$BIN_DIR" ]; then
    bnote "PIGEMMA_LLAMA_BIN=$PIGEMMA_LLAMA_BIN is set: later pi scripts use THAT directory, not $BIN_DIR"
fi

# ------------------------------------------------------------------------------------- host / CPU
H_MACHINE=${PIGEMMA_MACHINE:-$(uname -m 2>/dev/null || echo unknown)}
CPU_FEATURES=""
CPU_IMPL=""
CPU_PART=""
CPU_MODEL=""
_read_cpuinfo() {
    local f=/proc/cpuinfo k v line
    if [ -n "${PIGEMMA_FAKE_PROC:-}" ]; then f="$PIGEMMA_FAKE_PROC/cpuinfo"; fi
    if [ ! -r "$f" ]; then return 0; fi
    while IFS= read -r line || [ -n "$line" ]; do
        case $line in *:*) ;; *) continue ;; esac
        k=${line%%:*}
        v=${line#*:}
        k=${k%"${k##*[![:space:]]}"}
        v=${v#"${v%%[![:space:]]*}"}
        case $k in
            Features | flags) if [ -z "$CPU_FEATURES" ]; then CPU_FEATURES=$v; fi ;;
            "CPU implementer") if [ -z "$CPU_IMPL" ]; then CPU_IMPL=${v,,}; fi ;;
            "CPU part") if [ -z "$CPU_PART" ]; then CPU_PART=${v,,}; fi ;;
            "model name" | Model) if [ -z "$CPU_MODEL" ]; then CPU_MODEL=$v; fi ;;
        esac
    done <"$f"
    return 0
}
_read_cpuinfo
_has_feature() { [[ " $CPU_FEATURES " == *" $1 "* ]]; }

IS_ARM64=0
case $H_MACHINE in aarch64 | arm64) IS_ARM64=1 ;; esac
QEMU_SUSPECT=0
if [ "$IS_ARM64" = 1 ] && [ "$CPU_IMPL" = "0x00" ]; then QEMU_SUSPECT=1; fi
EMULATED=0
if is_emulated; then EMULATED=1; fi
FORCE_PI5=0
if [ "$IS_ARM64" = 1 ] && { [ "$EMULATED" = 1 ] || [ "$QEMU_SUSPECT" = 1 ]; }; then FORCE_PI5=1; fi
if [ "$QEMU_SUSPECT" = 1 ] && [ "$EMULATED" = 0 ]; then
    bwarn "/proc/cpuinfo reports CPU implementer 0x00 (qemu's emulated 'max' CPU): treating this host as EMULATED for the arch decision; set PIGEMMA_EMULATED=1 so results are labelled emulated"
fi
if [ "$ARCH_REQ" = pi5 ] && [ "$IS_ARM64" = 0 ]; then
    die 1 "--arch pi5 ($PI5_ARM_ARCH) is aarch64-only; this host is $H_MACHINE (use auto, native or generic)"
fi
if [ "$ARCH_REQ" = native ] && [ "$FORCE_PI5" = 1 ]; then
    bwarn "--arch native under emulation: ggml will detect the emulator's ISA (+i8mm+sve+sme); these binaries SIGILL on a real Pi 5"
fi
if [ "$IS_ARM64" = 1 ]; then FALLBACK_MODE=pi5; else FALLBACK_MODE=generic; fi
FALLBACK_REASON=""
FALLBACK_USED=0
if [ "$ARCH_REQ" = auto ]; then
    if [ "$FORCE_PI5" = 1 ]; then
        ARCH_MODE=pi5
        FALLBACK_REASON="emulated aarch64 host (PIGEMMA_EMULATED=$EMULATED, qemu_suspected=$QEMU_SUSPECT): Pi 5 ISA instead of native detection"
        FALLBACK_USED=1
    else
        ARCH_MODE=native
    fi
else
    ARCH_MODE=$ARCH_REQ
fi

ARCH_FLAGS=()
_arch_flags() { # _arch_flags MODE -> ARCH_FLAGS (1 for an unknown mode)
    case ${1:-} in
        native) ARCH_FLAGS=(-DGGML_NATIVE=ON -UGGML_CPU_ARM_ARCH) ;;
        pi5) ARCH_FLAGS=(-DGGML_NATIVE=OFF "-DGGML_CPU_ARM_ARCH=$PI5_ARM_ARCH") ;;
        generic) ARCH_FLAGS=(-DGGML_NATIVE=OFF -UGGML_CPU_ARM_ARCH) ;;
        *)
            ARCH_FLAGS=()
            return 1
            ;;
    esac
    return 0
}

_mode_ok() { # _mode_ok MODE -> 0 when an existing build in MODE satisfies this run
    local m=${1:-}
    case $ARCH_REQ in
        auto)
            if [ "$FORCE_PI5" = 1 ]; then
                [ "$m" = pi5 ]
            elif [ "$IS_ARM64" = 1 ]; then
                [ "$m" = native ] || [ "$m" = pi5 ]
            else
                [ "$m" = native ] || [ "$m" = generic ]
            fi
            ;;
        *) [ "$m" = "$ARCH_REQ" ] ;;
    esac
}

# ---------------------------------------------------------------------------------- CMake cache
_cache_get() { # _cache_get CACHEFILE KEY -> value (1 when absent)
    local f=$1 k=$2 line
    if [ ! -f "$f" ]; then return 1; fi
    line=$(grep -m1 -E "^${k}(:[A-Za-z]+)?=" "$f" 2>/dev/null) || return 1
    printf '%s\n' "${line#*=}"
}
_cache_mode() { # _cache_mode CACHEFILE -> native|pi5|generic|custom|'' (unknown)
    local f=$1 n a
    n=$(_cache_get "$f" GGML_NATIVE) || n=""
    a=$(_cache_get "$f" GGML_CPU_ARM_ARCH) || a=""
    case ${n^^} in
        ON | 1 | TRUE | YES) printf 'native\n' ;;
        OFF | 0 | FALSE | NO)
            if [ "$a" = "$PI5_ARM_ARCH" ]; then
                printf 'pi5\n'
            elif [ -z "$a" ]; then
                printf 'generic\n'
            else
                printf 'custom\n'
            fi
            ;;
        *) printf '\n' ;;
    esac
}
_cache_generator() { _cache_get "$1/CMakeCache.txt" CMAKE_GENERATOR 2>/dev/null || true; }

# ------------------------------------------------------------------------------------ git helpers
GIT=(git -c 'safe.directory=*' -c advice.detachedHead=false -C "$LLAMA_DIR")
_is_git_tree() { [ -e "$LLAMA_DIR/.git" ]; }
_git_head() { "${GIT[@]}" rev-parse --verify --quiet HEAD 2>/dev/null || true; }
_git_shallow() { "${GIT[@]}" rev-parse --is-shallow-repository 2>/dev/null || true; }
_git_dirty() { # 0 = local modifications to tracked files
    "${GIT[@]}" update-index -q --refresh >/dev/null 2>&1 || true
    ! "${GIT[@]}" diff-index --quiet HEAD -- >/dev/null 2>&1
}
_git_has_origin() { "${GIT[@]}" remote get-url origin >/dev/null 2>&1; }
_commit_matches() { [ -n "${1:-}" ] && [[ ${1,,} == "$COMMIT"* ]]; }

REQ_FULL=""
if [ "${#COMMIT}" -eq 40 ]; then
    REQ_FULL=$COMMIT
elif [[ $PINNED_FULL == "$COMMIT"* ]]; then
    REQ_FULL=$PINNED_FULL
fi

# ------------------------------------------------------------------------------- version helpers
VER_RC=""
VER_OUT=""
VER_BUILD=""
VER_COMMIT=""
_version_of() { # _version_of BINARY -> VER_RC VER_OUT VER_BUILD VER_COMMIT
    local out="" rc=0
    out=$(_tmo 60 "$1" --version 2>&1) || rc=$?
    VER_RC=$rc
    VER_OUT=$out
    VER_BUILD=""
    VER_COMMIT=""
    if [[ $out =~ build[[:space:]]+([0-9]+),[[:space:]]*commit[[:space:]]+([0-9a-fA-F]+) ]]; then
        VER_BUILD=${BASH_REMATCH[1]}
        VER_COMMIT=${BASH_REMATCH[2],,}
    elif [[ $out =~ version:[[:space:]]+([0-9]+)[[:space:]]+\(([0-9a-fA-F]+)\) ]]; then
        VER_BUILD=${BASH_REMATCH[1]}
        VER_COMMIT=${BASH_REMATCH[2],,}
    fi
    return 0
}
_version_bin() { # first target that understands --version (all common_params tools and llama-bench do)
    local t
    for t in llama-cli llama-completion llama-server llama-perplexity llama-bench "${TARGETS[@]}"; do
        if _in_list "$t" "${TARGETS[@]}"; then
            printf '%s\n' "$t"
            return 0
        fi
    done
    printf '\n'
}

UPTODATE_REASON=""
LLAMA_MODE_EXISTING=""
_llama_up_to_date() { # uses HEAD_FULL; sets UPTODATE_REASON when not up to date
    local t mode vb
    for t in "${TARGETS[@]}"; do
        if [ ! -x "$BIN_DIR/$t" ]; then
            UPTODATE_REASON="$BIN_DIR/$t is missing"
            return 1
        fi
    done
    mode=$(_cache_mode "$BUILD_DIR/CMakeCache.txt")
    if ! _mode_ok "$mode"; then
        UPTODATE_REASON="existing build has arch mode '${mode:-unknown}', this run wants '$ARCH_MODE'"
        return 1
    fi
    vb=$(_version_bin)
    _version_of "$BIN_DIR/$vb"
    if [ "$VER_RC" != 0 ]; then
        UPTODATE_REASON="$vb --version exited $VER_RC"
        return 1
    fi
    if [ -z "$VER_COMMIT" ] || [ -z "$HEAD_FULL" ] || [[ $HEAD_FULL != "$VER_COMMIT"* ]]; then
        UPTODATE_REASON="$vb reports commit '${VER_COMMIT:-?}', the checkout is at '${HEAD_FULL:0:12}'"
        return 1
    fi
    LLAMA_MODE_EXISTING=$mode
    return 0
}

TOOL_SRCS_PRESENT=()
_tool_src() {
    case $1 in
        pigemma-probe) printf '%s\n' "$TOOLS_SRC/probe/probe.cpp" ;;
        pinner) printf '%s\n' "$TOOLS_SRC/pinner/pinner.c" ;;
        membw) printf '%s\n' "$TOOLS_SRC/membw/membw.c" ;;
    esac
}
for _t in "${TOOL_TARGETS[@]}"; do
    if [ -f "$(_tool_src "$_t")" ]; then TOOL_SRCS_PRESENT+=("$_t"); fi
done
TOOLS_UPTODATE_REASON=""
_tools_up_to_date() { # _tools_up_to_date WANT_MODE ('' = any mode acceptable for this run)
    local want=${1:-} t src info commit mode ldir
    info="$TOOLS_BIN/pigemma-tools-build.json"
    if [ ! -f "$info" ]; then
        TOOLS_UPTODATE_REASON="$info is missing"
        return 1
    fi
    for t in "${TOOL_SRCS_PRESENT[@]}"; do
        src=$(_tool_src "$t")
        if [ ! -x "$TOOLS_BIN/$t" ]; then
            TOOLS_UPTODATE_REASON="$TOOLS_BIN/$t is missing"
            return 1
        fi
        if [ "$src" -nt "$TOOLS_BIN/$t" ] || [ "$TOOLS_SRC/CMakeLists.txt" -nt "$TOOLS_BIN/$t" ]; then
            TOOLS_UPTODATE_REASON="$t is older than its source"
            return 1
        fi
    done
    commit=$(sed -n 's/.*"llama_commit": *"\([^"]*\)".*/\1/p' "$info" | head -n 1) || commit=""
    if [ -z "$commit" ] || [ -z "$HEAD_FULL" ] || [[ $HEAD_FULL != "${commit,,}"* ]]; then
        TOOLS_UPTODATE_REASON="tools were built against llama.cpp '${commit:-?}', the checkout is at '${HEAD_FULL:0:12}'"
        return 1
    fi
    ldir=$(_cache_get "$TOOLS_BUILD/CMakeCache.txt" LLAMA_DIR) || ldir=""
    if [ "$(_abspath "${ldir:-/nonexistent}")" != "$LLAMA_DIR" ]; then
        TOOLS_UPTODATE_REASON="tools were configured with LLAMA_DIR='${ldir:-?}'"
        return 1
    fi
    mode=$(_cache_mode "$TOOLS_BUILD/CMakeCache.txt")
    if [ -n "$want" ]; then
        if [ "$mode" != "$want" ]; then
            TOOLS_UPTODATE_REASON="tools arch mode '${mode:-unknown}' differs from the llama build's '$want'"
            return 1
        fi
    elif ! _mode_ok "$mode"; then
        TOOLS_UPTODATE_REASON="tools arch mode '${mode:-unknown}' does not fit this run"
        return 1
    fi
    return 0
}

# ============================================================================== tools on PATH
GEN_CACHED=$(_cache_generator "$BUILD_DIR")
need_cmd cmake cmake "configure llama.cpp"
need_cmd gcc build-essential "C compiler"
need_cmd g++ build-essential "C++ compiler"
case "${CMAKE_GENERATOR:-}${GEN_CACHED}" in
    *Ninja*) need_cmd ninja ninja-build "CMake generator Ninja (CMAKE_GENERATOR or the existing build cache)" ;;
    *) need_cmd make build-essential "cmake's default generator (Unix Makefiles)" ;;
esac
need_cmd git git "fetch and verify the llama.cpp commit"
FROM_REMOTE=0
if [ -n "$FROM" ]; then
    case $FROM in
        [A-Za-z]:[/\\]*) FROM_REMOTE=0 ;;
        *:*) if [[ ${FROM%%:*} != */* ]]; then FROM_REMOTE=1; fi ;;
    esac
    if [ "$FROM_REMOTE" = 1 ]; then
        need_cmd rsync rsync "--from-desktop $FROM"
        need_cmd ssh openssh-client "--from-desktop $FROM"
    else
        FROM=$(_abspath "$FROM")
        if ! _is_llama_tree "$FROM"; then
            die 3 "--from-desktop $FROM is not a llama.cpp source tree (CMakeLists.txt / include/llama.h missing)"
        fi
        if [ "$FROM" = "$LLAMA_DIR" ]; then
            die 1 "--from-desktop $FROM is LLAMA_DIR itself"
        fi
    fi
fi
NICE=()
if have_cmd nice; then NICE=(nice -n "$NICE_LEVEL"); fi
NICE_IO=("${NICE[@]}")
if have_cmd ionice && ionice -c3 true >/dev/null 2>&1; then NICE_IO+=(ionice -c3); fi

# ============================================================================ source state / plan
HEAD_FULL=""
SRC_ACTION=""
if [ -e "$LLAMA_DIR" ] && [ ! -d "$LLAMA_DIR" ]; then
    die 3 "LLAMA_DIR=$LLAMA_DIR exists and is not a directory"
fi
if [ -n "$FROM" ]; then
    if [ -d "$LLAMA_DIR" ] && ! _is_empty_dir "$LLAMA_DIR" && ! _is_llama_tree "$LLAMA_DIR"; then
        die 3 "refusing to copy (rsync --delete) into $LLAMA_DIR: it is not empty and not a llama.cpp tree (set LLAMA_DIR to a new directory)"
    fi
    if [ "$FROM_REMOTE" = 1 ] || have_cmd rsync; then SRC_ACTION=rsync; else SRC_ACTION=copy; fi
elif [ ! -d "$LLAMA_DIR" ] || _is_empty_dir "$LLAMA_DIR"; then
    if [ -n "$REQ_FULL" ]; then SRC_ACTION=fetch; else SRC_ACTION=clone; fi
elif _is_git_tree; then
    if ! _is_llama_tree "$LLAMA_DIR"; then
        die 3 "$LLAMA_DIR is a git repository but not a llama.cpp tree (include/llama.h missing): refusing to touch it"
    fi
    HEAD_FULL=$(_git_head)
    if _commit_matches "$HEAD_FULL"; then
        SRC_ACTION=use
    else
        SRC_ACTION=switch
        if _git_dirty; then
            die 3 "$LLAMA_DIR is at ${HEAD_FULL:0:12} with local modifications; refusing to check out $COMMIT over them (commit/stash them, or point LLAMA_DIR at a new directory)"
        fi
    fi
else
    if ! _is_llama_tree "$LLAMA_DIR"; then
        die 3 "$LLAMA_DIR exists, is not empty and is not a llama.cpp tree: refusing to overwrite it (set LLAMA_DIR to a new directory)"
    fi
    SRC_ACTION=unverified
    bwarn "$LLAMA_DIR has llama.cpp sources but no .git: its commit cannot be verified against $COMMIT (recorded as unverified)"
fi
log "llama.cpp source: $LLAMA_DIR, requested commit $COMMIT${REQ_FULL:+ ($REQ_FULL)}, action: $SRC_ACTION${HEAD_FULL:+ (HEAD ${HEAD_FULL:0:12})}"

LLAMA_NEEDED=1
if [ "$SRC_ACTION" = use ] && [ "$REBUILD" = 0 ] && [ "$CONFIGURE_ONLY" = 0 ]; then
    if _llama_up_to_date; then
        LLAMA_NEEDED=0
        log "llama.cpp build is up to date (${VER_OUT%%$'\n'*}; arch mode $LLAMA_MODE_EXISTING): not rebuilt (--rebuild forces)"
    else
        log "llama.cpp needs a build: $UPTODATE_REASON"
    fi
fi
TOOLS_NEEDED=0
if [ "$WITH_TOOLS" = 1 ]; then
    if [ ! -f "$TOOLS_SRC/CMakeLists.txt" ]; then
        die 3 "--with-tools: $TOOLS_SRC/CMakeLists.txt is missing (is PIGEMMA_HOME=$PIGEMMA_HOME the pi-gemma4 repo?)"
    fi
    if [ "${#TOOL_SRCS_PRESENT[@]}" -eq 0 ]; then
        die 3 "--with-tools: none of tools/probe/probe.cpp, tools/pinner/pinner.c, tools/membw/membw.c exists"
    fi
    TOOLS_NEEDED=1
    if [ "$SRC_ACTION" = use ] && [ "$REBUILD" = 0 ] && [ "$CONFIGURE_ONLY" = 0 ]; then
        if _tools_up_to_date "$LLAMA_MODE_EXISTING"; then
            TOOLS_NEEDED=0
            log "pigemma tools are up to date in $TOOLS_BIN: not rebuilt (--rebuild forces)"
        else
            log "pigemma tools need a build: $TOOLS_UPTODATE_REASON"
        fi
    fi
fi
WORK=0
if [ "$SRC_ACTION" != use ] || [ "$LLAMA_NEEDED" = 1 ] || [ "$TOOLS_NEEDED" = 1 ]; then WORK=1; fi

# ==================================================================================== locks / gates
_bench_lock_check() {
    if [ "$IGNORE_LOCK" = 1 ]; then
        bwarn "--ignore-lock: building even if results/.bench.lock is held (a running benchmark would be distorted)"
        return 0
    fi
    local py out st reason cmd pp
    if ! py=$(pigemma_python); then
        bwarn "no Python >= 3.10: the bench lock (results/.bench.lock) was not checked"
        return 0
    fi
    pp=$(native_path "$PIGEMMA_HOME")
    out=$(cd "$PIGEMMA_HOME" && PYTHONPATH="$pp" "$py" -c '
try:
    from pigemma import paths
    st = paths.lock_status()
    h = st.get("holder") or {}
    print("\x1f".join([str(st.get("state")), str(st.get("reason")), str(h.get("cmd", ""))[:200]]))
except Exception as e:
    print("unknown\x1f%s: %s\x1f" % (type(e).__name__, e))
' 2>/dev/null) || out=$'unknown\x1fpython failed\x1f'
    IFS=$'\x1f' read -r st reason cmd <<<"$out" || true
    case ${st:-unknown} in
        held) die 4 "the bench lock is held (${reason:-?}; cmd: ${cmd:-?}): a parallel compile would distort that measurement and compete for its RAM. Wait for it, or pass --ignore-lock" ;;
        free | stale | inherited) log "bench lock: ${st}" ;;
        *) bwarn "could not read the bench lock (${reason:-?}); continuing" ;;
    esac
}

_build_lock() { # one pi/02 at a time per PIGEMMA_OUT
    local lockf="$PIGEMMA_OUT/.build.lock" pid=""
    if have_cmd flock; then
        exec 9>"$lockf" || die 1 "cannot open $lockf"
        flock -n 9 || die 4 "another pi/02_build_llama.sh run holds $lockf"
        return 0
    fi
    if ! mkdir "$lockf.d" 2>/dev/null; then
        pid=$(cat "$lockf.d/pid" 2>/dev/null || true)
        if [[ $pid =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then # liveness probe only, no signal sent
            die 4 "another pi/02_build_llama.sh run (pid $pid) holds $lockf.d"
        fi
        rm -rf -- "$lockf.d"
        mkdir "$lockf.d" 2>/dev/null || die 4 "cannot take the build lock $lockf.d"
    fi
    printf '%s\n' "$$" >"$lockf.d/pid"
    add_cleanup "rm -rf -- $(printf '%q' "$lockf.d")"
}

_bench_lock_check
if [ "$DRY" = 0 ] && [ "$WORK" = 1 ]; then _build_lock; fi

MEM_AVAIL_KB_START=$(mem_available_kb)
FREE_BEFORE=$(free_disk_bytes "$LLAMA_DIR" || true)
if [ "$WORK" = 1 ]; then
    NEED_DISK=$(_gb "$MIN_FREE_GB")
    if [ "$SRC_ACTION" != use ] || [ "$LLAMA_NEEDED" = 1 ]; then
        require_free_disk "$LLAMA_DIR" "$NEED_DISK" "the llama.cpp source + build (${MIN_FREE_GB} GB budget)"
    fi
    if [ "$TOOLS_NEEDED" = 1 ]; then
        require_free_disk "$TOOLS_BUILD" "$NEED_DISK" "the pigemma tools build (${MIN_FREE_GB} GB budget)"
    fi
    require_mem_available "$(_gb "$MIN_MEM_GB")" "a parallel C++ build"
    JOBS=$(pick_jobs)
    log "load: loadavg1 $(loadavg1); build jobs: $JOBS (BUILD_JOBS=${BUILD_JOBS:-auto}); nice -n $NICE_LEVEL"
else
    JOBS=$(pick_jobs)
    log "nothing to fetch or compile: disk/RAM gates skipped"
fi
if [ "$DRY" = 0 ]; then
    mkdir -p "$LOG_DIR" || die 1 "cannot create $LOG_DIR"
fi

# ======================================================================================= source step
SRC_METHOD=$SRC_ACTION
SOURCE_CHANGED=0
T_SOURCE=0
_trunc() { if [ "$DRY" = 0 ]; then : >"$1"; fi; }

_src_fetch_full() { # fresh dir + known full hash: git init + depth-1 fetch; 1 when the fetch failed
    local rc=0
    if [ "$DRY" = 0 ]; then mkdir -p "$LLAMA_DIR" || die 1 "cannot create $LLAMA_DIR"; fi
    _x "$LOG_SRC" "${GIT[@]}" init -q || die 1 "git init failed in $LLAMA_DIR (see $LOG_SRC)"
    if [ "$DRY" = 1 ] || ! _git_has_origin; then
        _x "$LOG_SRC" "${GIT[@]}" remote add origin "$REPO_URL" || die 1 "git remote add failed (see $LOG_SRC)"
    fi
    _retry "$NET_RETRIES" "$NET_RETRY_DELAY_S" _x "$LOG_SRC" "${NICE_IO[@]}" "${GIT[@]}" fetch --depth 1 origin "$REQ_FULL" || rc=$?
    if [ "$rc" != 0 ]; then
        bwarn "git fetch --depth 1 origin $REQ_FULL failed (exit $rc); falling back to git clone --filter=blob:none"
        rm -rf -- "${LLAMA_DIR:?}/.git"
        return 1
    fi
    _x "$LOG_SRC" "${GIT[@]}" checkout -q --detach FETCH_HEAD || die 1 "git checkout FETCH_HEAD failed (see $LOG_SRC)"
    SRC_METHOD=fetch-depth1
    return 0
}

_src_checkout_any() { # checkout COMMIT in an existing (full or shallow) git checkout, fetching if needed
    local full="" rc=0 remote=origin
    if [ "$DRY" = 1 ]; then
        _x "$LOG_SRC" "${GIT[@]}" checkout -q --detach "${REQ_FULL:-$COMMIT}"
        return 0
    fi
    if ! _git_has_origin; then remote=$REPO_URL; fi
    full=$("${GIT[@]}" rev-parse --verify --quiet "${REQ_FULL:-$COMMIT}^{commit}" 2>/dev/null) || full=""
    if [ -z "$full" ]; then
        if [ "$(_git_shallow)" = true ]; then
            if [ -z "$REQ_FULL" ]; then
                die 3 "commit $COMMIT is not in the shallow checkout $LLAMA_DIR and an abbreviated hash cannot be fetched: pass the full 40-hex --commit"
            fi
            _retry "$NET_RETRIES" "$NET_RETRY_DELAY_S" _x "$LOG_SRC" "${NICE_IO[@]}" "${GIT[@]}" fetch --depth 1 "$remote" "$REQ_FULL" || rc=$?
        elif [ -n "$REQ_FULL" ]; then
            _retry "$NET_RETRIES" "$NET_RETRY_DELAY_S" _x "$LOG_SRC" "${NICE_IO[@]}" "${GIT[@]}" fetch "$remote" "$REQ_FULL" || rc=$?
        else
            _retry "$NET_RETRIES" "$NET_RETRY_DELAY_S" _x "$LOG_SRC" "${NICE_IO[@]}" "${GIT[@]}" fetch "$remote" || rc=$?
        fi
        if [ "$rc" != 0 ]; then
            _tail_log "$LOG_SRC" 20
            die 1 "git fetch of $COMMIT failed (exit $rc) in $LLAMA_DIR"
        fi
        full=$("${GIT[@]}" rev-parse --verify --quiet "${REQ_FULL:-$COMMIT}^{commit}" 2>/dev/null) || full=""
        if [ -z "$full" ] && [ -n "$REQ_FULL" ]; then
            full=$("${GIT[@]}" rev-parse --verify --quiet "FETCH_HEAD^{commit}" 2>/dev/null) || full=""
        fi
        if [ -z "$full" ]; then
            die 3 "commit $COMMIT not found in $REPO_URL (checkout $LLAMA_DIR)"
        fi
    fi
    _x "$LOG_SRC" "${GIT[@]}" checkout -q --detach "$full" || {
        _tail_log "$LOG_SRC" 20
        die 1 "git checkout $full failed in $LLAMA_DIR"
    }
}

_src_clone() { # manifest recipe: blob:none clone + checkout
    local rc=0
    if [ "$DRY" = 0 ]; then mkdir -p "$(dirname "$LLAMA_DIR")" || die 1 "cannot create the parent of $LLAMA_DIR"; fi
    _retry "$NET_RETRIES" "$NET_RETRY_DELAY_S" _x "$LOG_SRC" "${NICE_IO[@]}" git -c 'safe.directory=*' clone \
        --filter=blob:none "$REPO_URL" "$LLAMA_DIR" || rc=$?
    if [ "$rc" != 0 ]; then
        _tail_log "$LOG_SRC" 20
        die 1 "git clone --filter=blob:none $REPO_URL $LLAMA_DIR failed (exit $rc); check the network, or use --from-desktop"
    fi
    _src_checkout_any
    SRC_METHOD=clone-blob-none
}

_src_copy() { # --from-desktop: rsync (remote or local) or a tar pipe (local, no rsync)
    local rc=0 rsh=${PIGEMMA_RSYNC_RSH:-ssh -o BatchMode=yes -o ConnectTimeout=20} src=${FROM%/}
    if [ "$DRY" = 0 ]; then mkdir -p "$LLAMA_DIR" || die 1 "cannot create $LLAMA_DIR"; fi
    if [ "$SRC_ACTION" = rsync ]; then
        local args=(rsync -a --delete --exclude=/build/ --exclude='/build-*/')
        if [ "$FROM_REMOTE" = 1 ]; then args+=(-e "$rsh"); fi
        _retry "$NET_RETRIES" "$NET_RETRY_DELAY_S" _x "$LOG_SRC" "${NICE_IO[@]}" "${args[@]}" "$src/" "$LLAMA_DIR/" || rc=$?
        if [ "$rc" != 0 ]; then
            _tail_log "$LOG_SRC" 20
            die 1 "rsync from $FROM failed (exit $rc)"
        fi
        SRC_METHOD=rsync
        return 0
    fi
    bwarn "rsync is not installed: copying $FROM with tar (files deleted at the source are NOT removed from $LLAMA_DIR)"
    if [ "$DRY" = 1 ]; then
        printf '+ (cd %q && tar --exclude=./build --exclude=./build-\\* -cf - .) | (cd %q && tar -xf -)\n' "$src" "$LLAMA_DIR"
    else
        printf '\n$ tar copy %s -> %s\n' "$src" "$LLAMA_DIR" >>"$LOG_SRC"
        (cd "$src" && "${NICE_IO[@]}" tar --exclude=./build --exclude='./build-*' -cf - .) 2>>"$LOG_SRC" |
            (cd "$LLAMA_DIR" && tar -xf -) 2>>"$LOG_SRC" || rc=$?
        if [ "$rc" != 0 ]; then
            _tail_log "$LOG_SRC" 20
            die 1 "copying $FROM to $LLAMA_DIR failed (exit $rc)"
        fi
    fi
    SRC_METHOD=copy
}

_t0=$SECONDS
_trunc "$LOG_SRC"
case $SRC_ACTION in
    use | unverified) ;;
    fetch)
        log "fetching llama.cpp $REQ_FULL (depth 1) from $REPO_URL into $LLAMA_DIR"
        if ! _src_fetch_full; then _src_clone; fi
        SOURCE_CHANGED=1
        ;;
    clone)
        log "cloning $REPO_URL (--filter=blob:none) into $LLAMA_DIR, then checking out $COMMIT"
        _src_clone
        SOURCE_CHANGED=1
        ;;
    switch)
        log "switching $LLAMA_DIR from ${HEAD_FULL:0:12} to $COMMIT"
        _src_checkout_any
        SRC_METHOD=switch
        SOURCE_CHANGED=1
        ;;
    rsync | copy)
        log "copying the llama.cpp tree from $FROM into $LLAMA_DIR ($SRC_ACTION; build dirs excluded)"
        _src_copy
        SOURCE_CHANGED=1
        ;;
esac
T_SOURCE=$((SECONDS - _t0))

# ------------------------------------------------------------------------------------ verify source
VERIFIED=0
SHALLOW=""
DIRTY=""
if [ "$DRY" = 0 ]; then
    if ! _is_llama_tree "$LLAMA_DIR"; then
        die 3 "$LLAMA_DIR is not a llama.cpp source tree after the source step (CMakeLists.txt / include/llama.h missing)"
    fi
    if _is_git_tree; then
        HEAD_FULL=$(_git_head)
        SHALLOW=$(_git_shallow)
        if [ -n "$HEAD_FULL" ]; then
            if _git_dirty; then DIRTY=true; else DIRTY=false; fi
        fi
        if _commit_matches "$HEAD_FULL"; then
            VERIFIED=1
        elif [ "$SRC_ACTION" = rsync ] || [ "$SRC_ACTION" = copy ]; then
            die 3 "the tree copied from $FROM is at ${HEAD_FULL:-an unknown commit}, not $COMMIT; pass --commit ${HEAD_FULL:0:7} to build it anyway"
        else
            die 1 "internal: $LLAMA_DIR is at '${HEAD_FULL:-?}' after the source step, expected $COMMIT"
        fi
        if [ "$DIRTY" = true ]; then
            bwarn "$LLAMA_DIR differs from commit ${HEAD_FULL:0:12} (local modifications, or CRLF line endings from a Windows checkout): the binaries may not match the pinned source exactly"
        fi
    elif [ "$SRC_ACTION" != unverified ]; then
        bwarn "$LLAMA_DIR has no .git after the copy: its commit cannot be verified against $COMMIT"
    fi
else
    if [ "$SRC_ACTION" = fetch ]; then
        HEAD_FULL=$REQ_FULL
        SHALLOW=true
    elif [ -n "$HEAD_FULL" ]; then
        SHALLOW=$(_git_shallow)
    fi
fi
HEAD_SHORT=${HEAD_FULL:0:7}

BN_FLAGS=(-ULLAMA_BUILD_NUMBER -ULLAMA_BUILD_COMMIT)
BN_OVERRIDE=""
if [ "$HEAD_FULL" = "$PINNED_FULL" ] && [ "$SHALLOW" = true ]; then
    BN_FLAGS=("-DLLAMA_BUILD_NUMBER=$PINNED_BUILD" -ULLAMA_BUILD_COMMIT)
    BN_OVERRIDE=$PINNED_BUILD
fi

# ================================================================================== llama.cpp build
CONFIGURE_ARGS=()
T_CONFIGURE=0
T_BUILD=0
LLAMA_REBUILT=0
ARM_DETECTED=""
MISMATCH=""
BUILD_STATUS=ok
BUILD_ERROR=""
LLAMA_FINAL_MODE=""

_configure_llama() { # _configure_llama MODE -> exit code of cmake
    _arch_flags "$1"
    CONFIGURE_ARGS=(-S "$LLAMA_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "${ARCH_FLAGS[@]}"
        -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF "${BN_FLAGS[@]}")
    _x "$LOG_CFG" "${NICE[@]}" cmake "${CONFIGURE_ARGS[@]}"
}

_supports() { # _supports CACHEFILE FEATURE -> true|false|'' (cache GGML_MACHINE_SUPPORTS_<feature>)
    local v
    if v=$(_cache_get "$1" "GGML_MACHINE_SUPPORTS_$2"); then
        case ${v^^} in 1 | ON | TRUE | YES) printf 'true\n' ;; *) printf 'false\n' ;; esac
    else
        printf '\n'
    fi
}

_native_mismatch() { # after a native configure on aarch64: features the cache enabled but cpuinfo lacks
    local f bad=()
    if [ "$IS_ARM64" = 0 ] || [ -z "$CPU_FEATURES" ]; then return 1; fi
    for f in i8mm sve sme; do
        if [ "$(_supports "$BUILD_DIR/CMakeCache.txt" "$f")" = true ] && ! _has_feature "$f"; then
            bad+=("$f")
        fi
    done
    if [ "${#bad[@]}" -gt 0 ]; then
        MISMATCH="${bad[*]}"
        return 0
    fi
    return 1
}

_can_fallback() { [ "$ARCH_REQ" = auto ] && [ "$ARCH_MODE" = native ] && [ "$FALLBACK_USED" = 0 ]; }
_do_fallback() { # _do_fallback REASON
    FALLBACK_USED=1
    FALLBACK_REASON=$1
    ARCH_MODE=$FALLBACK_MODE
    bwarn "arch fallback: $1 -> configuring again with --arch $FALLBACK_MODE"
    if [ "$DRY" = 0 ]; then rm -f -- "$BUILD_DIR/CMakeCache.txt"; fi
}

_bfail() { # _bfail CODE MSG: record status=failed in build.json, then die
    local code=$1
    shift
    _hb_stop
    if [ "$DRY" = 0 ]; then
        BUILD_STATUS=failed
        BUILD_ERROR="$*"
        _write_build_json || true
    fi
    die "$code" "$*"
}

_build_llama() {
    local rc attempt=0 t0 t missing cfg_start=0
    _trunc "$LOG_CFG"
    _trunc "$LOG_BLD"
    while :; do
        attempt=$((attempt + 1))
        if [ "$attempt" -gt 3 ]; then _bfail 1 "internal: too many configure/build attempts"; fi
        _arch_flags "$ARCH_MODE"
        log "configuring llama.cpp (arch mode $ARCH_MODE: ${ARCH_FLAGS[*]}${BN_OVERRIDE:+, build number $BN_OVERRIDE}; log: $LOG_CFG)"
        t0=$SECONDS
        rc=0
        cfg_start=$(_lines "$LOG_CFG")
        _configure_llama "$ARCH_MODE" || rc=$?
        T_CONFIGURE=$((T_CONFIGURE + SECONDS - t0))
        if [ "$rc" != 0 ]; then
            _tail_log "$LOG_CFG" 30
            if _can_fallback; then
                _do_fallback "native configure failed (exit $rc)"
                continue
            fi
            _bfail 1 "cmake configure failed (exit $rc); log: $LOG_CFG"
        fi
        if [ "$DRY" = 0 ]; then
            # only this attempt's part of the (appended) configure log
            ARM_DETECTED=$(tail -n "+$((cfg_start + 1))" "$LOG_CFG" | sed -n 's/.*ARM detected flags: *//p' | tail -n 1) || ARM_DETECTED=""
            if [ -n "$ARM_DETECTED" ]; then log "ggml native detection: $ARM_DETECTED"; fi
            MISMATCH=""
            if [ "$ARCH_MODE" = native ] && _native_mismatch; then
                if _can_fallback; then
                    _do_fallback "native detection enabled '$MISMATCH' but /proc/cpuinfo does not list it (emulator?; detected ${ARM_DETECTED:-?})"
                    ARM_DETECTED=""
                    continue
                fi
                bwarn "native detection enabled '$MISMATCH' which /proc/cpuinfo does not list: the binaries may SIGILL"
            fi
        fi
        if [ "$CONFIGURE_ONLY" = 1 ]; then return 0; fi
        log "building ${TARGETS[*]} with -j $JOBS (log: $LOG_BLD)"
        t0=$SECONDS
        rc=0
        _hb_start "$LOG_BLD" "llama.cpp build"
        _x "$LOG_BLD" "${NICE[@]}" cmake --build "$BUILD_DIR" -j "$JOBS" --target "${TARGETS[@]}" || rc=$?
        _hb_stop
        T_BUILD=$((T_BUILD + SECONDS - t0))
        if [ "$rc" != 0 ]; then
            _tail_log "$LOG_BLD" 40
            if _can_fallback && grep -qaE 'march=|mcpu=|invalid feature modifier|target specific option mismatch|unrecognized command[- ]line option|bad value' "$LOG_BLD" 2>/dev/null; then
                _do_fallback "native build failed with an arch-flag error"
                continue
            fi
            _bfail 1 "llama.cpp build failed (exit $rc) after $((SECONDS - t0)) s; log: $LOG_BLD (an out-of-memory kill shows as 'Killed': retry with --jobs 1)"
        fi
        if [ "$DRY" = 1 ]; then return 0; fi
        missing=()
        for t in "${TARGETS[@]}"; do
            if [ ! -x "$BIN_DIR/$t" ]; then missing+=("$t"); fi
        done
        if [ "${#missing[@]}" -gt 0 ]; then
            _bfail 1 "build finished but binaries are missing in $BIN_DIR: ${missing[*]}"
        fi
        _version_of "$BIN_DIR/$(_version_bin)"
        if [ "$VER_RC" = 132 ]; then
            if _can_fallback; then
                _do_fallback "the native binary died with SIGILL (illegal instruction)"
                continue
            fi
            _bfail 1 "the built binary dies with SIGILL (exit $VER_RC): wrong CPU flags for this machine (try --arch pi5)"
        fi
        if [ "$VER_RC" != 0 ]; then
            _bfail 1 "$(_version_bin) --version exited $VER_RC: ${VER_OUT:0:300}"
        fi
        LLAMA_REBUILT=1
        return 0
    done
}

# ====================================================================================== tools build
TOOLS_REBUILT=0
T_TOOLS=0
TOOLS_CONFIGURE_ARGS=()
_build_tools() {
    local rc=0 t0 mode=${LLAMA_FINAL_MODE:-$ARCH_MODE} t
    case $mode in native | pi5 | generic) ;; *) mode=$ARCH_MODE ;; esac
    _arch_flags "$mode"
    TOOLS_CONFIGURE_ARGS=(-S "$TOOLS_SRC" -B "$TOOLS_BUILD" "-DLLAMA_DIR=$LLAMA_DIR" "${ARCH_FLAGS[@]}" -DCMAKE_BUILD_TYPE=Release)
    _trunc "$LOG_TCFG"
    _trunc "$LOG_TBLD"
    log "configuring pigemma tools in $TOOLS_BUILD (arch mode $mode; log: $LOG_TCFG)"
    t0=$SECONDS
    _x "$LOG_TCFG" "${NICE[@]}" cmake "${TOOLS_CONFIGURE_ARGS[@]}" || rc=$?
    if [ "$rc" != 0 ]; then
        _tail_log "$LOG_TCFG" 30
        _bfail 1 "tools configure failed (exit $rc); log: $LOG_TCFG (a build dir configured with another generator needs: rm $TOOLS_BUILD/CMakeCache.txt)"
    fi
    if [ "$CONFIGURE_ONLY" = 1 ]; then
        T_TOOLS=$((SECONDS - t0))
        return 0
    fi
    log "building ${TOOL_SRCS_PRESENT[*]} with -j $JOBS (log: $LOG_TBLD)"
    _hb_start "$LOG_TBLD" "pigemma tools build"
    _x "$LOG_TBLD" "${NICE[@]}" cmake --build "$TOOLS_BUILD" -j "$JOBS" --target "${TOOL_SRCS_PRESENT[@]}" || rc=$?
    _hb_stop
    T_TOOLS=$((SECONDS - t0))
    if [ "$rc" != 0 ]; then
        _tail_log "$LOG_TBLD" 40
        _bfail 1 "pigemma tools build failed (exit $rc); log: $LOG_TBLD"
    fi
    if [ "$DRY" = 1 ]; then return 0; fi
    for t in "${TOOL_SRCS_PRESENT[@]}"; do
        if [ ! -x "$TOOLS_BIN/$t" ]; then _bfail 1 "tools build finished but $TOOLS_BIN/$t is missing"; fi
    done
    TOOLS_REBUILT=1
}

# ===================================================================================== record helpers
BENCH_HELP=""
MEMBW_VERSION=""
PINNER_VERSION=""
PROBE_HELP=""
OBJ_FREED=""

_collect_versions() {
    local out vb prc=0
    if [ "$CONFIGURE_ONLY" = 1 ]; then return 0; fi
    vb=$(_version_bin)
    if [ -z "$VER_OUT" ] && [ -n "$vb" ] && [ -x "$BIN_DIR/$vb" ]; then
        _version_of "$BIN_DIR/$vb"
    fi
    if _in_list llama-bench "${TARGETS[@]}" && [ -x "$BIN_DIR/llama-bench" ]; then
        out=$(_tmo 60 "$BIN_DIR/llama-bench" --help 2>&1) || true
        BENCH_HELP=${out%%$'\n'*}
    fi
    if [ "$WITH_TOOLS" = 1 ]; then
        if [ -x "$TOOLS_BIN/membw" ]; then MEMBW_VERSION=$(_first_line "$TOOLS_BIN/membw" --version); fi
        if [ -x "$TOOLS_BIN/pinner" ]; then PINNER_VERSION=$(_first_line "$TOOLS_BIN/pinner" --version); fi
        if [ -x "$TOOLS_BIN/pigemma-probe" ]; then
            out=$(_tmo 60 "$TOOLS_BIN/pigemma-probe" --help 2>&1) || prc=$?
            PROBE_HELP=$(printf '%s\n' "$out" | head -n 3) || true
            if [ "$prc" -ge 126 ]; then _bfail 1 "pigemma-probe --help exited $prc: the binary does not run"; fi
        fi
    fi
}

_effective_flags() { # unique -mcpu=/-march= tokens from the generated build files of build dir $1
    local d=$1
    {
        if [ -d "$d/ggml/src" ]; then
            grep -rhoE --include=flags.make -- '-m(cpu|arch)=[^ "]+' "$d/ggml/src" 2>/dev/null || true
        fi
        if [ -f "$d/build.ninja" ]; then grep -hoE -- '-m(cpu|arch)=[^ "]+' "$d/build.ninja" 2>/dev/null || true; fi
    } | sort -u
}

_targets_json() { # _targets_json DIR NAME... -> JSON array of {name, path, exists, bytes}
    local dir=$1 t first=1 p ex
    shift
    printf '['
    for t in "$@"; do
        p="$dir/$t"
        if [ "$first" = 0 ]; then printf ', '; fi
        first=0
        if [ -x "$p" ]; then ex=true; else ex=false; fi
        printf '{"name": %s, "path": %s, "exists": %s, "bytes": %s}' "$(_js "$t")" "$(_js "$p")" "$ex" "$(_jn "$(_file_size "$p")")"
    done
    printf ']'
}

_write_build_json() {
    local tmp dir tools_json="null" info="null" eff=() line pyc="" mode cflags=()
    local cmake_v gcc_v gxx_v git_v make_v="" ninja_v="" src_bytes="" lb_bytes="" tb_bytes="" total="" ic cleaned
    dir=$(dirname "$JSON_OUT")
    mkdir -p "$dir" || return 1
    pyc=$(pigemma_python) || pyc=""
    while IFS= read -r line; do
        if [ -n "$line" ]; then eff+=("$line"); fi
    done < <(_effective_flags "$BUILD_DIR")
    mode=${LLAMA_FINAL_MODE:-$ARCH_MODE}
    if _arch_flags "$mode"; then cflags=("${ARCH_FLAGS[@]}"); fi
    cmake_v=$(_first_line cmake --version)
    gcc_v=$(_first_line gcc -dumpfullversion)
    gxx_v=$(_first_line g++ -dumpfullversion)
    git_v=$(_first_line git --version)
    if have_cmd make; then make_v=$(_first_line make --version); fi
    if have_cmd ninja; then ninja_v=$(_first_line ninja --version); fi
    total=$(_bytes_of "$LLAMA_DIR")
    lb_bytes=$(_bytes_of "$BUILD_DIR")
    if [[ $total =~ ^[0-9]+$ ]] && [[ $lb_bytes =~ ^[0-9]+$ ]]; then src_bytes=$((total - lb_bytes)); fi
    if in_container; then ic=true; else ic=false; fi
    if [ "$CLEAN_OBJ" = 1 ] && [ -n "$OBJ_FREED" ]; then cleaned=true; else cleaned=false; fi
    if [ "$WITH_TOOLS" = 1 ]; then
        tb_bytes=$(_bytes_of "$TOOLS_BUILD")
        if [ -f "$TOOLS_BIN/pigemma-tools-build.json" ]; then
            info=$(cat "$TOOLS_BIN/pigemma-tools-build.json")
            if [ -n "$pyc" ] && ! "$pyc" -c 'import json, sys; json.load(open(sys.argv[1], encoding="utf-8"))' \
                "$(native_path "$TOOLS_BIN/pigemma-tools-build.json")" >/dev/null 2>&1; then
                info="null"
                WARNINGS+=("pigemma-tools-build.json is not valid JSON; not embedded")
            fi
        fi
        tools_json=$(
            printf '{\n'
            printf '    "source_dir": %s,\n' "$(_js "$TOOLS_SRC")"
            printf '    "build_dir": %s,\n' "$(_js "$TOOLS_BUILD")"
            printf '    "bin_dir": %s,\n' "$(_js "$TOOLS_BIN")"
            printf '    "rebuilt": %s,\n' "$(_jb "$TOOLS_REBUILT")"
            printf '    "seconds": %s,\n' "$(_jn "$T_TOOLS")"
            printf '    "configure_args": %s,\n' "$(_ja "${TOOLS_CONFIGURE_ARGS[@]}")"
            printf '    "targets": %s,\n' "$(_targets_json "$TOOLS_BIN" "${TOOL_SRCS_PRESENT[@]}")"
            printf '    "membw_version": %s,\n' "$(_jsn "$MEMBW_VERSION")"
            printf '    "pinner_version": %s,\n' "$(_jsn "$PINNER_VERSION")"
            printf '    "probe_help": %s,\n' "$(_jsn "$PROBE_HELP")"
            printf '    "logs": {"configure": %s, "build": %s},\n' "$(_js "$LOG_TCFG")" "$(_js "$LOG_TBLD")"
            printf '    "build_info": %s\n' "$info"
            printf '  }'
        )
    fi
    tmp="$JSON_OUT.tmp.$$"
    {
        printf '{\n'
        printf '  "format_version": 1,\n'
        printf '  "tool": "pi/02_build_llama.sh",\n'
        printf '  "tool_version": %s,\n' "$(_js "$B_VERSION")"
        printf '  "created_at": %s,\n' "$(_js "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
        printf '  "status": %s,\n' "$(_js "$BUILD_STATUS")"
        printf '  "error": %s,\n' "$(_jsn "$BUILD_ERROR")"
        printf '  "argv": %s,\n' "$(_ja "${ORIG_ARGS[@]}")"
        printf '  "host": {\n'
        printf '    "hostname": %s,\n' "$(_js "$(host_name)")"
        printf '    "kernel": %s,\n' "$(_jsn "$(kernel_release)")"
        printf '    "machine": %s,\n' "$(_js "$H_MACHINE")"
        printf '    "page_size": %s,\n' "$(_jn "$(page_size)")"
        printf '    "nproc": %s,\n' "$(_jn "$(cpu_count)")"
        printf '    "ram_total_kb": %s,\n' "$(_jn "$(mem_total_kb)")"
        printf '    "ram_available_kb": %s,\n' "$(_jn "$MEM_AVAIL_KB_START")"
        printf '    "cpu_model": %s,\n' "$(_jsn "$CPU_MODEL")"
        printf '    "cpu_implementer": %s,\n' "$(_jsn "$CPU_IMPL")"
        printf '    "cpu_part": %s,\n' "$(_jsn "$CPU_PART")"
        # shellcheck disable=SC2086
        printf '    "cpu_features": %s,\n' "$(_ja $CPU_FEATURES)"
        printf '    "in_container": %s,\n' "$ic"
        printf '    "emulated": %s,\n' "$(_jb "$EMULATED")"
        printf '    "qemu_suspected": %s,\n' "$(_jb "$QEMU_SUSPECT")"
        printf '    "protected_override": %s,\n' "$(_jb "$PROTECTED_OVERRIDE")"
        printf '    "fake_proc": %s\n' "$(_jsn "${PIGEMMA_FAKE_PROC:-}")"
        printf '  },\n'
        printf '  "llama": {\n'
        printf '    "dir": %s,\n' "$(_js "$LLAMA_DIR")"
        printf '    "repo_url": %s,\n' "$(_js "$REPO_URL")"
        printf '    "from_desktop": %s,\n' "$(_jsn "$FROM")"
        printf '    "requested_commit": %s,\n' "$(_js "$COMMIT")"
        printf '    "commit": %s,\n' "$(_jsn "$HEAD_FULL")"
        printf '    "commit_short": %s,\n' "$(_jsn "$HEAD_SHORT")"
        printf '    "verified": %s,\n' "$(_jb "$VERIFIED")"
        printf '    "shallow": %s,\n' "$(_jb "$SHALLOW")"
        printf '    "dirty": %s,\n' "$(_jb "$DIRTY")"
        printf '    "source_method": %s,\n' "$(_js "$SRC_METHOD")"
        printf '    "source_changed": %s,\n' "$(_jb "$SOURCE_CHANGED")"
        printf '    "build_dir": %s,\n' "$(_js "$BUILD_DIR")"
        printf '    "bin_dir": %s,\n' "$(_js "$BIN_DIR")"
        printf '    "arch": {\n'
        printf '      "requested": %s,\n' "$(_js "$ARCH_REQ")"
        printf '      "mode": %s,\n' "$(_jsn "$mode")"
        printf '      "fallback_used": %s,\n' "$(_jb "$FALLBACK_USED")"
        printf '      "fallback_reason": %s,\n' "$(_jsn "$FALLBACK_REASON")"
        printf '      "cmake_flags": %s,\n' "$(_ja "${cflags[@]}")"
        printf '      "arm_detected_flags": %s,\n' "$(_jsn "$ARM_DETECTED")"
        printf '      "effective_flags": %s,\n' "$(_ja "${eff[@]}")"
        printf '      "machine_supports": {"dotprod": %s, "i8mm": %s, "sve": %s, "sme": %s}\n' \
            "$(_jb "$(_supports "$BUILD_DIR/CMakeCache.txt" dotprod)")" \
            "$(_jb "$(_supports "$BUILD_DIR/CMakeCache.txt" i8mm)")" \
            "$(_jb "$(_supports "$BUILD_DIR/CMakeCache.txt" sve)")" \
            "$(_jb "$(_supports "$BUILD_DIR/CMakeCache.txt" sme)")"
        printf '    },\n'
        printf '    "build_number_override": %s,\n' "$(_jn "$BN_OVERRIDE")"
        printf '    "configure_args": %s,\n' "$(_ja "${CONFIGURE_ARGS[@]}")"
        printf '    "generator": %s,\n' "$(_jsn "$(_cache_generator "$BUILD_DIR")")"
        printf '    "jobs": %s,\n' "$(_jn "$JOBS")"
        printf '    "rebuilt": %s,\n' "$(_jb "$LLAMA_REBUILT")"
        printf '    "targets": %s,\n' "$(_targets_json "$BIN_DIR" "${TARGETS[@]}")"
        printf '    "version": %s,\n' "$(_jsn "$VER_OUT")"
        printf '    "build_number": %s,\n' "$(_jn "$VER_BUILD")"
        printf '    "version_commit": %s,\n' "$(_jsn "$VER_COMMIT")"
        printf '    "llama_bench_help": %s,\n' "$(_jsn "$BENCH_HELP")"
        printf '    "seconds": {"source": %s, "configure": %s, "build": %s},\n' \
            "$(_jn "$T_SOURCE")" "$(_jn "$T_CONFIGURE")" "$(_jn "$T_BUILD")"
        printf '    "logs": {"source": %s, "configure": %s, "build": %s}\n' \
            "$(_js "$LOG_SRC")" "$(_js "$LOG_CFG")" "$(_js "$LOG_BLD")"
        printf '  },\n'
        printf '  "tools": %s,\n' "$tools_json"
        printf '  "toolchain": {"gcc": %s, "gxx": %s, "cmake": %s, "git": %s, "make": %s, "ninja": %s},\n' \
            "$(_jsn "$gcc_v")" "$(_jsn "$gxx_v")" "$(_jsn "$cmake_v")" "$(_jsn "$git_v")" "$(_jsn "$make_v")" "$(_jsn "$ninja_v")"
        printf '  "disk": {"path": %s, "free_before_bytes": %s, "free_after_bytes": %s, "source_bytes": %s, "llama_build_bytes": %s, "tools_build_bytes": %s, "objects_cleaned": %s, "objects_freed_bytes": %s},\n' \
            "$(_js "$LLAMA_DIR")" "$(_jn "$FREE_BEFORE")" "$(_jn "$(free_disk_bytes "$LLAMA_DIR" || true)")" \
            "$(_jn "$src_bytes")" "$(_jn "$lb_bytes")" "$(_jn "$tb_bytes")" "$cleaned" "$(_jn "$OBJ_FREED")"
        printf '  "config": {"nice_level": %s, "keep_free_gb": %s, "min_free_gb": %s, "min_mem_gb": %s, "build_jobs": %s},\n' \
            "$(_jn "$NICE_LEVEL")" "$(_jn "$KEEP_FREE_GB")" "$(_jn "$MIN_FREE_GB")" "$(_jn "$MIN_MEM_GB")" "$(_js "${BUILD_JOBS:-auto}")"
        printf '  "warnings": %s,\n' "$(_ja "${WARNINGS[@]}")"
        printf '  "notes": %s\n' "$(_ja "${NOTES[@]}")"
        printf '}\n'
    } >"$tmp" || {
        rm -f -- "$tmp"
        return 1
    }
    if [ -n "$pyc" ] && ! "$pyc" -c 'import json, sys; json.load(open(sys.argv[1], encoding="utf-8"))' "$(native_path "$tmp")" >/dev/null 2>&1; then
        warn "internal: the generated build record is not valid JSON; kept as $tmp for inspection"
        return 1
    fi
    mv -f -- "$tmp" "$JSON_OUT" || return 1
    log "build record written: $JSON_OUT"
    return 0
}

# ============================================================================================ run
if [ "$LLAMA_NEEDED" = 1 ]; then
    _build_llama
    if [ "$DRY" = 0 ]; then
        LLAMA_FINAL_MODE=$(_cache_mode "$BUILD_DIR/CMakeCache.txt")
    else
        LLAMA_FINAL_MODE=$ARCH_MODE
    fi
else
    LLAMA_FINAL_MODE=$LLAMA_MODE_EXISTING
    ARCH_MODE=${LLAMA_MODE_EXISTING:-$ARCH_MODE}
    if [ -n "$BN_OVERRIDE" ] && [ -n "$VER_BUILD" ] && [ "$VER_BUILD" != "$BN_OVERRIDE" ]; then
        # e.g. pi/adhoc/build_llama_native.sh: same commit, built without -DLLAMA_BUILD_NUMBER
        bnote "the existing binaries report build $VER_BUILD instead of b$BN_OVERRIDE (shallow clone built without -DLLAMA_BUILD_NUMBER); the commit ${VER_COMMIT:-?} is verified, so they are kept. --rebuild makes them report b$BN_OVERRIDE"
    fi
    if [ -f "$LOG_CFG" ] && [ "$LLAMA_MODE_EXISTING" = native ]; then
        # the last configure in the log was the native one that produced this build
        ARM_DETECTED=$(sed -n 's/.*ARM detected flags: *//p' "$LOG_CFG" | tail -n 1) || ARM_DETECTED=""
    fi
fi
if [ "$CONFIGURE_ONLY" = 1 ]; then BUILD_STATUS=configured; fi

if [ "$WITH_TOOLS" = 1 ]; then
    if [ "$SOURCE_CHANGED" = 1 ] || [ "$REBUILD" = 1 ] || [ "$CONFIGURE_ONLY" = 1 ]; then
        TOOLS_NEEDED=1
    elif [ "$TOOLS_NEEDED" = 0 ] && [ "$LLAMA_REBUILT" = 1 ] && [ "$DRY" = 0 ]; then
        # the llama build may have switched arch mode (fallback): re-check against the final mode
        if ! _tools_up_to_date "$LLAMA_FINAL_MODE"; then
            log "pigemma tools need a build: $TOOLS_UPTODATE_REASON"
            TOOLS_NEEDED=1
        fi
    fi
    if [ "$TOOLS_NEEDED" = 1 ]; then _build_tools; fi
fi

if [ "$CLEAN_OBJ" = 1 ] && [ "$CONFIGURE_ONLY" = 0 ]; then
    if [ "$DRY" = 1 ]; then
        printf '+ find %q -name %q -type f -delete\n' "$BUILD_DIR" '*.o'
        if [ "$WITH_TOOLS" = 1 ]; then printf '+ find %q -name %q -type f -delete\n' "$TOOLS_BUILD" '*.o'; fi
    else
        _before=$(_bytes_of "$BUILD_DIR")
        _tb=0
        if [ "$WITH_TOOLS" = 1 ] && [ -d "$TOOLS_BUILD" ]; then _tb=$(_bytes_of "$TOOLS_BUILD"); fi
        find "$BUILD_DIR" -name '*.o' -type f -delete 2>/dev/null || bwarn "could not delete every object file in $BUILD_DIR"
        if [ "$WITH_TOOLS" = 1 ] && [ -d "$TOOLS_BUILD" ]; then
            find "$TOOLS_BUILD" -name '*.o' -type f -delete 2>/dev/null || bwarn "could not delete every object file in $TOOLS_BUILD"
        fi
        _after=$(_bytes_of "$BUILD_DIR")
        _ta=0
        if [ "$WITH_TOOLS" = 1 ] && [ -d "$TOOLS_BUILD" ]; then _ta=$(_bytes_of "$TOOLS_BUILD"); fi
        OBJ_FREED=$((${_before:-0} - ${_after:-0} + ${_tb:-0} - ${_ta:-0}))
        log "--clean-objects: freed $(fmt_gb "$OBJ_FREED" 3) of object files (a later --rebuild recompiles everything)"
    fi
fi

if [ "$DRY" = 1 ]; then
    printf '\n== DRY RUN: nothing was fetched, built or written ==\n'
    printf '%-17s %s\n' "llama.cpp dir" "$LLAMA_DIR (source action: $SRC_ACTION)"
    printf '%-17s %s\n' "commit" "$COMMIT${REQ_FULL:+ ($REQ_FULL)}"
    printf '%-17s %s\n' "arch" "$ARCH_MODE${FALLBACK_REASON:+ - $FALLBACK_REASON}"
    if [ "$LLAMA_NEEDED" = 1 ]; then _s="needed (-j $JOBS)"; else _s="up to date"; fi
    printf '%-17s %s\n' "llama build" "$_s"
    if [ "$WITH_TOOLS" = 1 ]; then
        if [ "$TOOLS_NEEDED" = 1 ]; then _s="needed -> $TOOLS_BIN"; else _s="up to date"; fi
        printf '%-17s %s\n' "tools build" "$_s"
    fi
    for _t in "${TARGETS[@]}"; do printf '%-17s %s\n' "$_t" "$BIN_DIR/$_t"; done
    printf '%-17s %s (not written)\n' "build.json" "$JSON_OUT"
    exit 0
fi

_collect_versions
_write_build_json || bwarn "could not write $JSON_OUT"

# ----------------------------------------------------------------------------------------- stdout
_mode=${LLAMA_FINAL_MODE:-$ARCH_MODE}
_flags=""
if _arch_flags "$_mode"; then _flags=${ARCH_FLAGS[*]}; fi
if [ "$CONFIGURE_ONLY" = 1 ]; then
    printf '\n== llama.cpp %s configured in %s (--configure-only: nothing compiled) ==\n' "${HEAD_SHORT:-?}" "$BUILD_DIR"
else
    printf '\n== llama.cpp %s in %s ==\n' "${VER_COMMIT:-${HEAD_SHORT:-?}}" "$BIN_DIR"
    for _t in "${TARGETS[@]}"; do
        if [ -x "$BIN_DIR/$_t" ]; then
            printf '%-17s %s\n' "$_t" "$BIN_DIR/$_t"
        else
            printf '%-17s %s  MISSING\n' "$_t" "$BIN_DIR/$_t"
        fi
    done
    printf '%-17s %s\n' "version" "${VER_OUT%%$'\n'*}"
fi
printf '%-17s %s\n' "arch" "${_mode:-?}: ${_flags}${ARM_DETECTED:+ (detected $ARM_DETECTED)}${FALLBACK_REASON:+ [fallback: $FALLBACK_REASON]}"
if [ "$LLAMA_REBUILT" = 1 ]; then
    printf '%-17s %s\n' "build" "rebuilt: source $T_SOURCE s + configure $T_CONFIGURE s + build $T_BUILD s, -j $JOBS, nice -n $NICE_LEVEL"
elif [ "$CONFIGURE_ONLY" = 1 ]; then
    printf '%-17s %s\n' "build" "configured in $T_CONFIGURE s (source $T_SOURCE s)"
else
    printf '%-17s %s\n' "build" "up to date (not rebuilt; --rebuild forces)"
fi
if [ -n "$BENCH_HELP" ]; then
    printf '\n== llama-bench --help | head -1 ==\n%s\n' "$BENCH_HELP"
fi
if [ "$WITH_TOOLS" = 1 ]; then
    printf '\n== pigemma tools: %s ==\n' "$TOOLS_BIN"
    for _t in "${TOOL_TARGETS[@]}"; do
        if _in_list "$_t" "${TOOL_SRCS_PRESENT[@]}"; then
            if [ -x "$TOOLS_BIN/$_t" ]; then
                printf '%-17s %s\n' "$_t" "$TOOLS_BIN/$_t"
            elif [ "$CONFIGURE_ONLY" = 1 ]; then
                printf '%-17s %s  (configured, not compiled)\n' "$_t" "$TOOLS_BIN/$_t"
            else
                printf '%-17s %s  MISSING\n' "$_t" "$TOOLS_BIN/$_t"
            fi
        else
            printf '%-17s (not built: its source does not exist yet)\n' "$_t"
        fi
    done
    if [ -f "$TOOLS_BIN/pigemma-tools-build.json" ]; then
        printf '%-17s %s\n' "build info" "$TOOLS_BIN/pigemma-tools-build.json"
    fi
    if [ "$TOOLS_REBUILT" = 1 ]; then
        _s="rebuilt in $T_TOOLS s"
    elif [ "$CONFIGURE_ONLY" = 1 ]; then
        _s="configured"
    else
        _s="up to date (not rebuilt)"
    fi
    printf '%-17s %s\n' "build" "$_s"
    if [ -n "$MEMBW_VERSION" ]; then printf '%-17s %s\n' "membw" "$MEMBW_VERSION"; fi
    if [ -n "$PINNER_VERSION" ]; then printf '%-17s %s\n' "pinner" "$PINNER_VERSION"; fi
    if [ -n "$PROBE_HELP" ]; then printf '%-17s %s\n' "pigemma-probe" "${PROBE_HELP%%$'\n'*}"; fi
    if [ "$TOOLS_BUILD" != "$(_abspath "$TOOLS_SRC/build")" ]; then
        printf '%-17s %s\n' "hint" "export PIGEMMA_TOOLS_BIN=$TOOLS_BIN (later pi scripts default to tools/build/bin)"
    fi
fi
if [ "$LLAMA_DIR" != "$_HOME_ABS/llama.cpp" ]; then
    printf '%-17s %s\n' "hint" "later pi scripts find these binaries through LLAMA_DIR=$LLAMA_DIR (pi/pi_config.env or the environment)"
fi
printf '\n%-17s %s\n' "build.json" "$JSON_OUT"
if [ "$CONFIGURE_ONLY" = 1 ]; then
    log "DONE: llama.cpp configured (arch ${_mode:-?}); run again without --configure-only to compile"
elif [ "$LLAMA_REBUILT" = 1 ] || [ "$TOOLS_REBUILT" = 1 ]; then
    log "DONE: build OK (${VER_OUT%%$'\n'*}); next step: pi/03_bench_io_mem.sh"
else
    log "DONE: everything up to date; build.json refreshed"
fi
exit 0
