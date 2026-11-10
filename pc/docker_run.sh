#!/usr/bin/env bash
# =============================================================================
# pc/docker_run.sh - the one Docker wrapper every pc/*.sh uses (pi-gemma4).
#
# Runs a command in the Linux build image with this repository bind-mounted at
# /work. Works from Git Bash on Windows (Docker Desktop, WSL2 backend) and from
# a plain Linux host. Spec: docs/design/manifest.json files[pc/docker_run.sh].
#
#   pc/docker_run.sh [options] -- <command...>
#   pc/docker_run.sh [options] build-tools [--clean] [--jobs N] [--targets LIST]
#                                          [--require-all] [--ignore-lock]
#   pc/docker_run.sh vm-df [--bytes | --json]
#   pc/docker_run.sh [--pi-emu] build-image [--tag T]
#
# `pc/docker_run.sh --memory 6g --cpus 6 --volume pigemma-models:/models:ro -- CMD...`
# becomes (repo path derived from this script's location, `pwd -W` on Git Bash):
#
#   MSYS_NO_PATHCONV=1 docker run --rm -i [-t] --init \
#       -v "/path/to/pi-gemma4:/work" -w /work \
#       --memory=6g --memory-swap=6g --cpus=6 -v pigemma-models:/models:ro \
#       -e PIGEMMA_IN_CONTAINER=1 -e PYTHONUTF8=1 ... \
#       pi-gemma4-build:latest /bin/sh -c '<python shim>; exec "$@"' pigemma-run CMD...
#
# Use --dry-run to see the exact command line without running anything.
#
# Decisions beyond the manifest text (documented here, listed as open issues):
#  1. TTY: -t is added only when stdin AND stdout are terminals (the spec tests
#     stdin only). With a redirected stdout, -t would merge the container's
#     stderr into the captured output and add CRLF line endings. Under mintty
#     without ConPTY, docker.exe rejects -t ("the input device is not a TTY").
#     In auto mode the wrapper then retries once without -t. The retry is safe
#     because docker refuses before it creates the container. --tty / --no-tty
#     and PIGEMMA_DOCKER_TTY=auto|0|1 override the automatic choice.
#  2. --init (tini) is the default. Ctrl-C and `docker stop` then reach the
#     command, zombies are reaped, and the exit code is still forwarded.
#     --no-init disables it.
#  3. python shim: the images contain only `python3`, but callers write
#     `python -m pigemma....` (manifest specs of trace_collect.sh, h2_run.ps1, ...).
#     When `python` is missing, the container command runs through
#     /bin/sh -c '...; exec "$@"', which puts a python -> python3 symlink
#     (in /tmp/.pigemma-shim) first on PATH, so nested scripts get it too.
#     --no-shim disables this.
#  4. Container environment:
#     - Always set: PIGEMMA_IN_CONTAINER=1, PYTHONUTF8=1,
#       PYTHONIOENCODING=utf-8, LANG=C.UTF-8 (Hebrew output) and PYTHONPATH=/work.
#     - PYTHONPYCACHEPREFIX=/tmp/pigemma-pycache keeps the bind mount free of
#       cpython-312 caches, which would be root-owned on a Linux host.
#     - PIGEMMA_REPO_HOST is the host path of the repo.
#     - Host PIGEMMA_* variables are forwarded, except host-side settings
#       (image, bash, gguf-py, models/llama/tools bin dirs) and values that look
#       like Windows paths.
#     - PIGEMMA_LOCK_FILE is translated to its /work/... path. A host
#       `python -m pigemma.paths lock --hold -- bash pc/docker_run.sh -- ...`
#       then hands its lock (PIGEMMA_LOCK_HELD token) to a nested
#       `pigemma.paths lock --hold` in the container ("inherited") instead of
#       deadlocking.
#  5. vm-df runs `df -P -B1 /` in the local build image instead of `alpine`:
#     no pull is needed and it is the same overlay filesystem (checked
#     2026-10-05; alpine is the fallback).
#     - Root free space inside the VM is only the room left INSIDE the sparse
#       virtual disk. The disk grows on the Windows drive that holds
#       docker_data.vhdx: CustomWslDistroDir in %APPDATA%\Docker\settings-store.json,
#       which is F: on this desktop, not C:.
#     - vm-df therefore also reports that drive's free bytes.
#     - --bytes prints the conservative min() of the two.
#  6. build-tools:
#     - Builds only the targets whose source file exists (tools/CMakeLists.txt
#       skips missing ones). --require-all or PIGEMMA_REQUIRE_ALL_TOOLS=1 makes
#       a missing source an error.
#     - Refuses while the bench lock (results/.bench.lock) is held, because a
#       12-way compile would distort that benchmark. --ignore-lock overrides.
#     - Serialises concurrent builds with flock on
#       tools/build-linux-x86/.pigemma-build.lock.
#     - Runs the build under `nice -n 10`.
#     - --clean empties the build dir but keeps the lock file.
#  7. On a native Linux host (not WSL, not Git Bash) the container runs as
#     --user $(id -u):$(id -g) with HOME=/tmp, so files written to the bind
#     mount are not root-owned. --as-root disables this.
#  8. --pi-emu means:
#     - image pi-gemma4-pi-emu:trixie, --platform linux/arm64, PIGEMMA_EMULATED=1;
#     - when it exists, Docker volume pigemma-arm64-llama is mounted at
#       /opt/llama (the emulated aarch64 llama.cpp build, see FACTS) and
#       PIGEMMA_LLAMA_BIN=/opt/llama/build/bin is set, so pigemma.paths.bin()
#       finds those binaries.
#     LLAMA_DIR is deliberately NOT set: it would redirect pi/02's own checkout.
#  9. build-image is an extra subcommand. It is the remedy that the
#     "image missing" error names, and it runs only when asked.
#
# Exit codes:
#   0  ok (also --help / --version / --dry-run).
#   2  usage error.
#   3  missing: repo layout, a local pi-gemma4-* image, tool sources, or
#      binaries after a build.
#   4  refused: docker CLI or daemon unavailable, or not in Linux-container
#      mode; bench lock held; another build-tools run in progress.
#   Anything else is forwarded from `docker run`: the container command's own
#   code, 125 = docker run itself failed, 126/127 = command not executable /
#   not found.
# Errors produced by the wrapper itself always print a stderr line that starts
# with "docker_run:".
#
# Requires bash >= 4.4 (Git Bash 5.2 / Debian bash 5.2). Lint: bash -n.
# =============================================================================
set -euo pipefail
shopt -u patsub_replacement 2>/dev/null || true

readonly PROG=docker_run
readonly VERSION=1.0.0
readonly EXIT_ERROR=1 EXIT_USAGE=2 EXIT_MISSING=3 EXIT_REFUSED=4
readonly CONTAINER_ROOT=/work
readonly DEFAULT_BUILD_IMAGE=pi-gemma4-build:latest
readonly DEFAULT_PI_EMU_IMAGE=pi-gemma4-pi-emu:trixie
readonly TOOLS_BUILD_REL=tools/build-linux-x86
readonly ALL_TOOLS=(pigemma-probe pinner membw)

# Stop Git Bash (MSYS) from rewriting /work, -w /work, volume specs, ... in the
# arguments of docker.exe. Harmless on Linux.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
say()  { printf '%s: %s\n' "$PROG" "$*" >&2; }
warn() { printf '%s: WARNING: %s\n' "$PROG" "$*" >&2; }
die()  { local rc=$1; shift; printf '%s: %s\n' "$PROG" "$*" >&2; exit "$rc"; }
vlog() { if [ "${VERBOSE:-0}" = 1 ]; then say "$*"; fi; }

UNAME_S=$(uname -s 2>/dev/null || echo unknown)
IS_MSYS=0
case $UNAME_S in MINGW*|MSYS*|CYGWIN*) IS_MSYS=1 ;; esac
is_wsl() { [ -r /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; }

TMP_FILES=()
cleanup() {
    local f
    for f in "${TMP_FILES[@]}"; do
        if [ -n "$f" ]; then rm -f -- "$f"; fi
    done
}
trap cleanup EXIT
# mk_tmp VAR - create a temp file, remember it for cleanup, store its path in VAR.
mk_tmp() {
    local _f
    _f=$(mktemp "${TMPDIR:-/tmp}/pigemma-docker.XXXXXX") || die "$EXIT_ERROR" "mktemp failed"
    TMP_FILES+=("$_f")
    printf -v "$1" '%s' "$_f"
}

json_str() {
    local s=$1 bs='\' dbs='\\' q='"' eq='\"' nl=$'\n' tab=$'\t' cr=$'\r'
    s=${s//"$bs"/"$dbs"}
    s=${s//"$q"/"$eq"}
    s=${s//"$nl"/'\n'}
    s=${s//"$tab"/'\t'}
    s=${s//"$cr"/}
    printf '"%s"' "$s"
}
json_num_or_null() { if [[ ${1:-} =~ ^[0-9]+$ ]]; then printf '%s' "$1"; else printf 'null'; fi; }

in_list() {
    local x=$1 y
    shift
    for y in "$@"; do
        if [ "$x" = "$y" ]; then return 0; fi
    done
    return 1
}

tool_source() {  # tool name -> source path relative to tools/
    case $1 in
        pigemma-probe) printf 'probe/probe.cpp' ;;
        pinner)        printf 'pinner/pinner.c' ;;
        membw)         printf 'membw/membw.c' ;;
        *) return 1 ;;
    esac
}

# print_cmd ARGS... - shell-quoted `docker ARGS...` (for --dry-run / --verbose)
print_cmd() {
    local a out="MSYS_NO_PATHCONV=1 docker"
    for a in "$@"; do out+=" $(printf '%q' "$a")"; done
    printf '%s\n' "$out"
}

# ---------------------------------------------------------------------------
# repository location (derived from this script, never from the caller's cwd)
# ---------------------------------------------------------------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) \
    || die "$EXIT_MISSING" "cannot resolve the directory of ${BASH_SOURCE[0]}"
REPO_POSIX=$(cd -- "$SCRIPT_DIR/.." && pwd) || die "$EXIT_MISSING" "cannot resolve the repository root"

# host_path_of DIR - the form `docker -v` understands (Windows mixed form on Git Bash)
host_path_of() {
    local p=$1 w=""
    if [ "$IS_MSYS" = 1 ]; then
        w=$(cd -- "$p" 2>/dev/null && pwd -W 2>/dev/null) || w=""
        if [ -z "$w" ] && command -v cygpath >/dev/null 2>&1; then
            w=$(cygpath -m -- "$p" 2>/dev/null) || w=""
        fi
    fi
    if [ -z "$w" ]; then w=$p; fi
    printf '%s' "$w"
}
REPO_HOST=$(host_path_of "$REPO_POSIX")

# to_container_path HOSTPATH - prints /work/... when HOSTPATH is inside the repo, else fails
to_container_path() {
    local p=$1 abs lr lp bs='\'
    p=${p//"$bs"/\/}
    if [ "$IS_MSYS" = 1 ]; then
        case $p in
            [A-Za-z]:/*) abs=$p ;;
            *) abs=$(cygpath -m -a -- "$p" 2>/dev/null) || return 1 ;;
        esac
        lp=${abs,,}
        lr=${REPO_HOST,,}
    else
        case $p in
            /*) abs=$p ;;
            *) abs=$PWD/$p ;;
        esac
        if command -v realpath >/dev/null 2>&1; then
            abs=$(realpath -m -- "$abs" 2>/dev/null) || true
        fi
        lp=$abs
        lr=$REPO_POSIX
    fi
    lr=${lr%/}
    if [ "$lp" = "$lr" ]; then
        printf '%s' "$CONTAINER_ROOT"
        return 0
    fi
    case $lp in
        "$lr"/*) printf '%s/%s' "$CONTAINER_ROOT" "${abs:$(( ${#lr} + 1 ))}"; return 0 ;;
    esac
    return 1
}

# convert_volume SRC:DST[:MODE] - make a host-path source absolute / Windows form;
# named volumes (no slash) and Windows drive paths pass through unchanged.
convert_volume() {
    local spec=$1 src rest re_win='^[A-Za-z]:[\/]' re_split='^([^:]+)(:.+)$'
    if [[ $spec =~ $re_win ]]; then
        printf '%s' "$spec"
        return 0
    fi
    if [[ $spec =~ $re_split ]]; then
        src=${BASH_REMATCH[1]}
        rest=${BASH_REMATCH[2]}
    else
        return 1
    fi
    case $src in
        */*|.|..|\~*) ;;
        *) printf '%s' "$spec"; return 0 ;;   # named volume
    esac
    case $src in
        \~)   src=$HOME ;;
        \~/*) src=$HOME/${src#\~/} ;;
    esac
    case $src in
        /*) ;;
        *) src=$PWD/$src ;;
    esac
    if [ "$IS_MSYS" = 1 ]; then
        src=$(cygpath -m -- "$src" 2>/dev/null) || return 1
    elif command -v realpath >/dev/null 2>&1; then
        src=$(realpath -m -- "$src" 2>/dev/null) || return 1
    fi
    printf '%s%s' "$src" "$rest"
}

# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------
usage() {
    cat <<'USAGE'
pc/docker_run.sh - run a command in the pi-gemma4 Linux build image (repo at /work)

Usage:
  pc/docker_run.sh [options] -- <command...>
  pc/docker_run.sh [options] build-tools [--clean] [--jobs N] [--targets LIST]
                                         [--require-all] [--ignore-lock]
  pc/docker_run.sh vm-df [--bytes | --json]
  pc/docker_run.sh [--pi-emu] build-image [--tag T]

Options (before the subcommand or '--'):
  --memory M         memory cap incl. page cache (e.g. 6g, 4096m); also sets
                     --memory-swap=M, i.e. no swap
  --cpus N           CPU quota (e.g. 6 or 1.5)
  --volume SPEC      extra -v SRC:DST[:ro], repeatable (e.g. pigemma-models:/models:ro);
                     relative / Git-Bash host paths are converted to absolute host paths
  --name N           container name
  --env K[=V]        set (K=V) or forward from the host (K) an environment variable;
                     repeatable; wins over the automatic ones
  --workdir DIR      container working directory (default /work; relative = under /work)
  --image I          image (default $PIGEMMA_IMAGE or pi-gemma4-build:latest)
  --platform P       docker --platform (e.g. linux/arm64)
  --pi-emu           emulated Pi 5: image $PIGEMMA_PI_EMU_IMAGE or pi-gemma4-pi-emu:trixie,
                     --platform linux/arm64, PIGEMMA_EMULATED=1, and (when it exists)
                     volume pigemma-arm64-llama at /opt/llama with
                     PIGEMMA_LLAMA_BIN=/opt/llama/build/bin. Speed is NOT representative.
  --no-llama-volume  with --pi-emu: do not mount the prebuilt aarch64 llama.cpp volume
  --tty | --no-tty   force / suppress docker -t (default: -t only when stdin AND stdout
                     are terminals; retried without -t if mintty cannot provide one)
  --no-stdin         omit -i (use inside `while read` loops so docker does not eat stdin)
  --no-init          omit --init
  --no-shim          do not provide `python` (-> python3) inside the container
  --as-root          native Linux host: do not run as your uid:gid
  --docker-arg A     extra raw `docker run` argument, repeatable
                     (e.g. --docker-arg=--cap-add=SYS_ADMIN)
  --dry-run          print the docker command(s) and exit 0 (no docker calls at all)
  --verbose          log decisions and the docker command line to stderr
  -h, --help         this help
  --version          print the wrapper version

Subcommands:
  build-tools        configure + build tools/CMakeLists.txt in the image into
                     tools/build-linux-x86 (Ninja, -DLLAMA_DIR=/work/third_party/llama.cpp
                     -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=ON, Release),
                     targets pigemma-probe pinner membw (those whose source exists),
                     then print the binary paths and `pigemma-probe --help | head -3`.
      --clean          empty tools/build-linux-x86 first
      --jobs N         parallel jobs (default $PIGEMMA_BUILD_JOBS or nproc in the container)
      --targets LIST   comma-separated subset of pigemma-probe,pinner,membw
      --require-all    fail (exit 3) if any tool source is missing
                       (also PIGEMMA_REQUIRE_ALL_TOOLS=1)
      --ignore-lock    build even while results/.bench.lock is held (distorts that benchmark)
  vm-df              free space of the Docker VM root filesystem (`df -P -B1 /` in a
                     container) + `docker system df` + key=value lines (vm_root_*,
                     host_vm_disk_*, vm_effective_avail_bytes)
      --bytes          print only vm_effective_avail_bytes = min(VM root avail, free bytes
                       on the host drive holding the VM disk, when that is known)
      --json           one JSON object instead of text
  build-image        docker build docker/Dockerfile.build -> pi-gemma4-build:latest
                     (with --pi-emu: docker/Dockerfile.pi-emu -> pi-gemma4-pi-emu:trixie)
      --tag T          image tag to build

Environment:
  PIGEMMA_IMAGE, PIGEMMA_PI_EMU_IMAGE, PIGEMMA_VMDF_IMAGE, PIGEMMA_ARM64_LLAMA_VOLUME,
  PIGEMMA_DOCKER_TTY=auto|0|1, PIGEMMA_DOCKER_STDIN=0|1, PIGEMMA_DOCKER_VERBOSE=1,
  PIGEMMA_BUILD_JOBS=N, PIGEMMA_REQUIRE_ALL_TOOLS=1.
  Inside the container: PIGEMMA_IN_CONTAINER=1, PYTHONUTF8=1, PYTHONPATH=/work,
  PIGEMMA_REPO_HOST, forwarded host PIGEMMA_* (PIGEMMA_LOCK_FILE mapped to /work/...).

Exit codes:
  0 ok | 2 usage | 3 missing (image, sources, binaries, repo layout) |
  4 refused (docker unavailable, bench lock held, concurrent build) |
  otherwise the exit code of `docker run` (= the command's, 125 docker error,
  126/127 not executable / not found). Wrapper errors start with "docker_run:".

Examples:
  bash pc/docker_run.sh -- ls /work/models
  bash pc/docker_run.sh build-tools
  bash pc/docker_run.sh --memory 6g --cpus 6 --volume pigemma-models:/models:ro -- bash /work/pc/pagecache_limit.sh --inner
  bash pc/docker_run.sh -- python -m pigemma.paths lock --hold -- tools/build-linux-x86/bin/membw --json
  bash pc/docker_run.sh --pi-emu -- bash pi/00_feasibility.sh
  bash pc/docker_run.sh vm-df --bytes
USAGE
}

# ---------------------------------------------------------------------------
# global option parsing
# ---------------------------------------------------------------------------
MEM="" CPUS="" NAME="" PLATFORM="" IMAGE_OPT="" WORKDIR=$CONTAINER_ROOT
PI_EMU=0 LLAMA_VOL=1 USE_INIT=1 USE_SHIM=1 DRY_RUN=0 AS_ROOT=0
TTY_MODE=${PIGEMMA_DOCKER_TTY:-auto}
USE_STDIN=${PIGEMMA_DOCKER_STDIN:-1}
VERBOSE=${PIGEMMA_DOCKER_VERBOSE:-0}
VOLUMES=() ENV_OPTS=() DOCKER_ARGS=() CMD=() SUB_ARGS=()
SUBCMD=""

need_val() {  # need_val OPTNAME "$#"
    if [ "$2" -lt 2 ]; then die "$EXIT_USAGE" "option $1 needs a value (see --help)"; fi
}

while [ $# -gt 0 ]; do
    case $1 in
        --memory)       need_val "$1" $#; MEM=$2; shift 2 ;;
        --memory=*)     MEM=${1#*=}; shift ;;
        --cpus)         need_val "$1" $#; CPUS=$2; shift 2 ;;
        --cpus=*)       CPUS=${1#*=}; shift ;;
        --volume)       need_val "$1" $#; VOLUMES+=("$2"); shift 2 ;;
        --volume=*)     VOLUMES+=("${1#*=}"); shift ;;
        --name)         need_val "$1" $#; NAME=$2; shift 2 ;;
        --name=*)       NAME=${1#*=}; shift ;;
        --env)          need_val "$1" $#; ENV_OPTS+=("$2"); shift 2 ;;
        --env=*)        ENV_OPTS+=("${1#*=}"); shift ;;
        --workdir)      need_val "$1" $#; WORKDIR=$2; shift 2 ;;
        --workdir=*)    WORKDIR=${1#*=}; shift ;;
        --image)        need_val "$1" $#; IMAGE_OPT=$2; shift 2 ;;
        --image=*)      IMAGE_OPT=${1#*=}; shift ;;
        --platform)     need_val "$1" $#; PLATFORM=$2; shift 2 ;;
        --platform=*)   PLATFORM=${1#*=}; shift ;;
        --docker-arg)   need_val "$1" $#; DOCKER_ARGS+=("$2"); shift 2 ;;
        --docker-arg=*) DOCKER_ARGS+=("${1#*=}"); shift ;;
        --pi-emu)       PI_EMU=1; shift ;;
        --no-llama-volume) LLAMA_VOL=0; shift ;;
        --tty)          TTY_MODE=1; shift ;;
        --no-tty)       TTY_MODE=0; shift ;;
        --no-stdin)     USE_STDIN=0; shift ;;
        --no-init)      USE_INIT=0; shift ;;
        --no-shim)      USE_SHIM=0; shift ;;
        --as-root)      AS_ROOT=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --verbose)      VERBOSE=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        --version)      printf 'pc/docker_run.sh %s\n' "$VERSION"; exit 0 ;;
        --)             shift; SUBCMD=run; CMD=("$@"); break ;;
        build-tools|vm-df|build-image)
                        SUBCMD=$1; shift; SUB_ARGS=("$@"); break ;;
        -v)             die "$EXIT_USAGE" "'-v' is ambiguous here: use --volume SPEC (or --verbose)" ;;
        -*)             die "$EXIT_USAGE" "unknown option '$1' (see --help)" ;;
        *)              die "$EXIT_USAGE" "unexpected '$1': put the container command after '--' (e.g. pc/docker_run.sh -- ls /work) or use a subcommand: build-tools, vm-df, build-image" ;;
    esac
done

if [ -z "$SUBCMD" ]; then
    usage >&2
    die "$EXIT_USAGE" "nothing to do: give '-- <command...>' or a subcommand"
fi

# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
re_mem='^[0-9]+(\.[0-9]+)?([bB]|[kKmMgGtT][iI]?[bB]?)?$'
re_cpus='^[0-9]+(\.[0-9]+)?$'
re_name='^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'
re_platform='^[a-z0-9]+/[a-z0-9_]+(/[a-z0-9]+)?$'
re_envname='^[A-Za-z_][A-Za-z0-9_]*$'
re_winpath='^[A-Za-z]:[\/]'

if [ -n "$MEM" ] && ! [[ $MEM =~ $re_mem ]]; then
    die "$EXIT_USAGE" "--memory '$MEM' is not a docker size (e.g. 6g, 4096m)"
fi
if [ -n "$CPUS" ]; then
    if ! [[ $CPUS =~ $re_cpus ]] || [[ $CPUS =~ ^0+(\.0*)?$ ]]; then
        die "$EXIT_USAGE" "--cpus '$CPUS' must be a positive number (e.g. 6 or 1.5)"
    fi
fi
if [ -n "$NAME" ] && ! [[ $NAME =~ $re_name ]]; then
    die "$EXIT_USAGE" "--name '$NAME' is not a valid container name"
fi
case $TTY_MODE in auto|0|1) ;; *) die "$EXIT_USAGE" "PIGEMMA_DOCKER_TTY must be auto, 0 or 1 (got '$TTY_MODE')" ;; esac
case $USE_STDIN in 0|1) ;; *) die "$EXIT_USAGE" "PIGEMMA_DOCKER_STDIN must be 0 or 1 (got '$USE_STDIN')" ;; esac
case $VERBOSE in 0|1) ;; *) VERBOSE=1 ;; esac
case $WORKDIR in
    /*) ;;
    "") die "$EXIT_USAGE" "--workdir must not be empty" ;;
    *) WORKDIR=$CONTAINER_ROOT/$WORKDIR ;;
esac

if [ "$PI_EMU" = 1 ] && [ -z "$PLATFORM" ]; then PLATFORM=linux/arm64; fi
if [ -n "$PLATFORM" ] && ! [[ $PLATFORM =~ $re_platform ]]; then
    die "$EXIT_USAGE" "--platform '$PLATFORM' is not of the form os/arch[/variant]"
fi

if [ -n "$IMAGE_OPT" ]; then
    IMAGE=$IMAGE_OPT
elif [ "$PI_EMU" = 1 ]; then
    IMAGE=${PIGEMMA_PI_EMU_IMAGE:-$DEFAULT_PI_EMU_IMAGE}
else
    IMAGE=${PIGEMMA_IMAGE:-$DEFAULT_BUILD_IMAGE}
fi
ARM64_LLAMA_VOLUME=${PIGEMMA_ARM64_LLAMA_VOLUME:-pigemma-arm64-llama}

CONVERTED_VOLUMES=()
for v in "${VOLUMES[@]}"; do
    cv=$(convert_volume "$v") || die "$EXIT_USAGE" "--volume '$v' is not SRC:DST[:MODE]"
    CONVERTED_VOLUMES+=("$cv")
done
VOLUMES=("${CONVERTED_VOLUMES[@]}")

declare -A EXPLICIT_ENV=()
for e in "${ENV_OPTS[@]}"; do
    en=${e%%=*}
    if ! [[ $en =~ $re_envname ]]; then die "$EXIT_USAGE" "--env '$e': '$en' is not a variable name"; fi
    EXPLICIT_ENV[$en]=1
done

if [ ! -d "$REPO_POSIX/pigemma" ] || [ ! -d "$REPO_POSIX/pc" ]; then
    die "$EXIT_MISSING" "repository layout not recognised at $REPO_POSIX (pigemma/ or pc/ missing)"
fi

# ---------------------------------------------------------------------------
# docker checks
# ---------------------------------------------------------------------------
docker_preflight() {
    local hint=""
    if is_wsl; then
        hint=" (this is WSL bash; on the Windows desktop run it from Git Bash instead)"
    fi
    if ! command -v docker >/dev/null 2>&1; then
        die "$EXIT_REFUSED" "REFUSED: the docker CLI is not on PATH - install/start Docker Desktop$hint"
    fi
    local errf out rc=0
    mk_tmp errf
    if command -v timeout >/dev/null 2>&1; then
        out=$(timeout 60 docker info --format '{{.OSType}} {{.ServerVersion}}' 2>"$errf") || rc=$?
    else
        out=$(docker info --format '{{.OSType}} {{.ServerVersion}}' 2>"$errf") || rc=$?
    fi
    out=${out//$'\r'/}
    if [ "$rc" -ne 0 ] || [ -z "${out#* }" ] || [ "${out#* }" = "<no value>" ]; then
        local why
        why=$(tail -n 2 -- "$errf" 2>/dev/null | tr '\r\n' '  ') || why=""
        if [ "$rc" = 124 ]; then why="timed out after 60 s. $why"; fi
        die "$EXIT_REFUSED" "REFUSED: 'docker info' failed (rc=$rc) - is Docker Desktop running?$hint ${why}"
    fi
    case $out in
        linux\ *) vlog "docker server: $out" ;;
        *) die "$EXIT_REFUSED" "REFUSED: docker reports OSType '${out%% *}'; switch Docker Desktop to Linux containers" ;;
    esac
}

image_present() { docker image inspect --format '{{.Id}}' "$1" >/dev/null 2>&1; }

require_image() {
    local img=$1
    if image_present "$img"; then return 0; fi
    case $img in
        "$DEFAULT_BUILD_IMAGE")
            die "$EXIT_MISSING" "image $img not found locally; build it with: bash pc/docker_run.sh build-image" ;;
        "$DEFAULT_PI_EMU_IMAGE")
            die "$EXIT_MISSING" "image $img not found locally; build it with: bash pc/docker_run.sh --pi-emu build-image" ;;
        pi-gemma4-*)
            die "$EXIT_MISSING" "image $img not found locally (pi-gemma4-* images are built locally, never pulled); see: bash pc/docker_run.sh build-image --help" ;;
        *)
            warn "image $img is not present locally; docker run will try to pull it" ;;
    esac
}

# ---------------------------------------------------------------------------
# container environment
# ---------------------------------------------------------------------------
declare -A ENV_VAL=()
ENV_ORDER=()
env_set() {
    if [ -z "${ENV_VAL[$1]+x}" ]; then ENV_ORDER+=("$1"); fi
    ENV_VAL[$1]=$2
}

collect_env() {
    env_set PIGEMMA_IN_CONTAINER 1
    env_set PYTHONUTF8 1
    env_set PYTHONIOENCODING utf-8
    env_set LANG C.UTF-8
    env_set PYTHONPATH "$CONTAINER_ROOT"
    env_set PYTHONPYCACHEPREFIX /tmp/pigemma-pycache
    env_set PIGEMMA_REPO_HOST "$REPO_HOST"

    local name val cp names
    names=$(compgen -e | grep '^PIGEMMA_' || true)
    while IFS= read -r name; do
        if [ -z "$name" ]; then continue; fi
        case $name in
            PIGEMMA_IN_CONTAINER|PIGEMMA_REPO_HOST|PIGEMMA_IMAGE|PIGEMMA_PI_EMU_IMAGE|PIGEMMA_VMDF_IMAGE|\
            PIGEMMA_ARM64_LLAMA_VOLUME|PIGEMMA_DOCKER_*|PIGEMMA_BASH|PIGEMMA_GGUF_PY|PIGEMMA_MODELS_DIR|\
            PIGEMMA_LLAMA_BIN|PIGEMMA_TOOLS_BIN|PIGEMMA_BUILD_JOBS)
                vlog "env $name: host-side setting, not forwarded"
                continue ;;
            PIGEMMA_LOCK_FILE)
                val=${!name}
                if cp=$(to_container_path "$val"); then
                    env_set PIGEMMA_LOCK_FILE "$cp"
                    vlog "env PIGEMMA_LOCK_FILE: $val -> $cp"
                else
                    warn "PIGEMMA_LOCK_FILE=$val is outside the repository; not forwarded (the container uses $CONTAINER_ROOT/results/.bench.lock)"
                fi
                continue ;;
        esac
        val=${!name}
        if [[ $val =~ $re_winpath ]]; then
            vlog "env $name looks like a Windows path; not forwarded"
            continue
        fi
        env_set "$name" "$val"
    done <<<"$names"
}

apply_explicit_env() {
    local e en
    for e in "${ENV_OPTS[@]}"; do
        en=${e%%=*}
        if [ "$e" != "$en" ]; then
            env_set "$en" "${e#*=}"
        elif [ -n "${!en+x}" ]; then
            env_set "$en" "${!en}"
        else
            warn "--env $en: not set on the host; skipped"
        fi
    done
}

# prepare_common - everything the run / build-tools paths share; fills RUN_OPTS and ENTRY
prepare_common() {
    if [ "$DRY_RUN" = 0 ]; then
        docker_preflight
        require_image "$IMAGE"
    fi
    collect_env
    if [ "$PI_EMU" = 1 ]; then
        env_set PIGEMMA_EMULATED 1
        if [ "$LLAMA_VOL" = 1 ]; then
            local have_vol=0
            if [ "$DRY_RUN" = 1 ]; then
                have_vol=1
                vlog "dry-run: existence of volume $ARM64_LLAMA_VOLUME not checked"
            elif docker volume inspect "$ARM64_LLAMA_VOLUME" >/dev/null 2>&1; then
                have_vol=1
            fi
            if [ "$have_vol" = 1 ]; then
                VOLUMES+=("$ARM64_LLAMA_VOLUME:/opt/llama")
                if [ -z "${EXPLICIT_ENV[PIGEMMA_LLAMA_BIN]+x}" ]; then
                    env_set PIGEMMA_LLAMA_BIN /opt/llama/build/bin
                fi
            else
                warn "docker volume $ARM64_LLAMA_VOLUME not found: no prebuilt aarch64 llama.cpp in the emulated container (pi/02_build_llama.sh can build one)"
            fi
        fi
    fi
    apply_explicit_env

    local user_args=()
    if [ "$IS_MSYS" = 0 ] && ! is_wsl && [ "$AS_ROOT" = 0 ] && [ "$(id -u)" != 0 ]; then
        user_args=(--user "$(id -u):$(id -g)")
        if [ -z "${EXPLICIT_ENV[HOME]+x}" ]; then env_set HOME /tmp; fi
    fi

    RUN_OPTS=(--rm)
    if [ "$USE_STDIN" = 1 ]; then RUN_OPTS+=(-i); fi
    if [ "$USE_INIT" = 1 ]; then RUN_OPTS+=(--init); fi
    RUN_OPTS+=(-v "$REPO_HOST:$CONTAINER_ROOT" -w "$WORKDIR")
    if [ -n "$MEM" ]; then RUN_OPTS+=("--memory=$MEM" "--memory-swap=$MEM"); fi
    if [ -n "$CPUS" ]; then RUN_OPTS+=("--cpus=$CPUS"); fi
    local v k
    for v in "${VOLUMES[@]}"; do RUN_OPTS+=(-v "$v"); done
    if [ -n "$NAME" ]; then RUN_OPTS+=(--name "$NAME"); fi
    if [ -n "$PLATFORM" ]; then RUN_OPTS+=(--platform "$PLATFORM"); fi
    RUN_OPTS+=("${user_args[@]}")
    for k in "${ENV_ORDER[@]}"; do RUN_OPTS+=(-e "$k=${ENV_VAL[$k]}"); done
    RUN_OPTS+=("${DOCKER_ARGS[@]}")

    # python -> python3 shim (see header, decision 3); exec keeps the exit code and signals.
    ENTRY=()
    if [ "$USE_SHIM" = 1 ]; then
        # shellcheck disable=SC2016  # expanded by the container's /bin/sh, not here
        ENTRY=(/bin/sh -c 'if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then d=/tmp/.pigemma-shim; if mkdir -p "$d" 2>/dev/null && ln -sf "$(command -v python3)" "$d/python" 2>/dev/null; then PATH="$d:$PATH"; export PATH; fi; fi; exec "$@"' pigemma-run)
    fi

    WANT_TTY=0
    case $TTY_MODE in
        1) WANT_TTY=1 ;;
        auto) if [ -t 0 ] && [ -t 1 ] && [ "$USE_STDIN" = 1 ]; then WANT_TTY=1; fi ;;
    esac
}

# docker_run_exec IMAGE CMD... - docker run "${RUN_OPTS[@]}" [-t] IMAGE CMD...; sets DOCKER_RC
docker_run_exec() {
    DOCKER_RC=0
    local tty_args=()
    if [ "$WANT_TTY" = 1 ]; then tty_args=(-t); fi
    if [ "$DRY_RUN" = 1 ]; then
        print_cmd run "${RUN_OPTS[@]}" "${tty_args[@]}" "$@"
        return 0
    fi
    if [ "$VERBOSE" = 1 ]; then print_cmd run "${RUN_OPTS[@]}" "${tty_args[@]}" "$@" >&2; fi
    if [ "$WANT_TTY" = 1 ] && [ "$TTY_MODE" = auto ] && [ "$IS_MSYS" = 1 ]; then
        # Copy docker's own stderr (not the container's output: with -t that goes to
        # stdout) into errf so the mintty refusal can be recognised. stdout stays the
        # terminal via fd 3; the pipeline is synchronous and PIPESTATUS[0] is docker's.
        local errf ps=()
        mk_tmp errf
        { docker run "${RUN_OPTS[@]}" -t "$@" 2>&1 1>&3 3>&- | tee -- "$errf" >&2; ps=("${PIPESTATUS[@]}"); } 3>&1 || true
        DOCKER_RC=${ps[0]:-1}
        if [ "$DOCKER_RC" -ne 0 ] && grep -q 'not a TTY' -- "$errf" 2>/dev/null; then
            warn "this terminal cannot give docker a TTY (mintty without ConPTY?); retrying without -t (--no-tty skips this)"
            DOCKER_RC=0
            docker run "${RUN_OPTS[@]}" "$@" || DOCKER_RC=$?
        fi
        return 0
    fi
    docker run "${RUN_OPTS[@]}" "${tty_args[@]}" "$@" || DOCKER_RC=$?
    return 0
}

# ---------------------------------------------------------------------------
# subcommand: run  (pc/docker_run.sh [options] -- <command...>)
# ---------------------------------------------------------------------------
cmd_run() {
    if [ "${#CMD[@]}" -eq 0 ]; then
        die "$EXIT_USAGE" "missing command after '--' (e.g. pc/docker_run.sh -- ls /work/models)"
    fi
    prepare_common
    docker_run_exec "$IMAGE" "${ENTRY[@]}" "${CMD[@]}"
    exit "$DOCKER_RC"
}

# ---------------------------------------------------------------------------
# subcommand: build-tools
# ---------------------------------------------------------------------------
# The script that runs INSIDE the container. Args: clean jobs require_all ignore_lock targets...
build_inner_script() {
    IFS= read -r -d '' BUILD_INNER <<'INNER' || true
set -euo pipefail
clean=$1 jobs=$2 require_all=$3 ignore_lock=$4
shift 4
targets=("$@")
root=/work
src=$root/tools
bdir=$root/tools/build-linux-x86
llama=$root/third_party/llama.cpp
lockf=$bdir/.pigemma-build.lock
host_root=${PIGEMMA_REPO_HOST:-<repo>}

note() { printf 'build-tools: %s\n' "$*" >&2; }
fail() { local rc=$1; shift; printf 'docker_run: build-tools: %s\n' "$*" >&2; exit "$rc"; }
in_list() { local x=$1 y; shift; for y in "$@"; do if [ "$x" = "$y" ]; then return 0; fi; done; return 1; }

[ -f "$src/CMakeLists.txt" ] || fail 3 "missing $src/CMakeLists.txt"
[ -f "$llama/CMakeLists.txt" ] || fail 3 "missing llama.cpp sources at $llama"
for t in cmake ninja; do
    command -v "$t" >/dev/null 2>&1 || fail 3 "'$t' not found in the image"
done

# 1. bench lock: never compile on 12 cores under a running benchmark
if [ "$ignore_lock" != 1 ]; then
    if command -v python3 >/dev/null 2>&1; then
        lk=$(cd "$root" && python3 -c '
import sys
sys.path.insert(0, "/work")
try:
    from pigemma import paths
    st = paths.lock_status()
    h = st.get("holder") or {}
    print("\x1f".join([str(st.get("state")), str(st.get("reason")), str(h.get("cmd", ""))[:200]]))
except Exception as e:
    print("unknown\x1f%s: %s\x1f" % (type(e).__name__, e))
' 2>/dev/null) || lk=$'unknown\x1fpython3 failed\x1f'
        IFS=$'\x1f' read -r lk_state lk_reason lk_cmd <<<"$lk" || true
        case ${lk_state:-unknown} in
            held) fail 4 "REFUSED: the bench lock is held ($lk_reason; cmd: ${lk_cmd:-?}). A parallel compile would distort that measurement. Wait for it, or pass --ignore-lock." ;;
            free|stale|inherited) note "bench lock: ${lk_state}" ;;
            *) note "WARNING: could not read the bench lock (${lk_reason:-?}); continuing" ;;
        esac
    else
        note "WARNING: python3 not in the image; bench-lock check skipped"
    fi
fi

# 2. one build-tools at a time per build dir
mkdir -p "$bdir"
if command -v flock >/dev/null 2>&1; then
    exec 9>"$lockf" || fail 1 "cannot open $lockf"
    flock -n 9 || fail 4 "REFUSED: another build-tools run holds $lockf"
else
    note "WARNING: flock not available; concurrent builds are not prevented"
fi

# 3. optional clean (keeps the lock file we hold)
if [ "$clean" = 1 ]; then
    note "--clean: emptying $bdir"
    find "$bdir" -mindepth 1 -maxdepth 1 ! -name .pigemma-build.lock -exec rm -rf -- {} +
fi

if ! [[ $jobs =~ ^[0-9]+$ ]] || [ "$jobs" -lt 1 ]; then
    jobs=$(nproc 2>/dev/null || echo 4)
fi

# 4. configure (always re-run: picks up tool sources that appeared since last time)
cfg=(cmake -S "$src" -B "$bdir" -G Ninja
     -DLLAMA_DIR="$llama"
     -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=ON
     -DCMAKE_BUILD_TYPE=Release)
if [ "$require_all" = 1 ]; then
    cfg+=(-DPIGEMMA_REQUIRE_ALL_TOOLS=ON)
else
    cfg+=(-DPIGEMMA_REQUIRE_ALL_TOOLS=OFF)
fi
note "targets: ${targets[*]}; jobs: $jobs; build dir: $bdir"
note "configure: ${cfg[*]}"
t0=$(date +%s)
rc=0
"${cfg[@]}" || rc=$?
if [ "$rc" -ne 0 ]; then
    fail "$rc" "cmake configure failed (rc=$rc). If $bdir was configured by another generator or source tree, retry with: bash pc/docker_run.sh build-tools --clean"
fi

# 5. build
bld=(cmake --build "$bdir" -j "$jobs" --target "${targets[@]}")
if command -v nice >/dev/null 2>&1; then bld=(nice -n 10 "${bld[@]}"); fi
note "build: ${bld[*]}"
rc=0
"${bld[@]}" || rc=$?
t1=$(date +%s)
if [ "$rc" -ne 0 ]; then
    fail "$rc" "build failed (rc=$rc) after $((t1 - t0)) s"
fi

# 6. report
status=0
printf '\n== pigemma tools: %s/bin (host: %s/tools/build-linux-x86/bin) ==\n' "$bdir" "$host_root"
for t in pigemma-probe pinner membw; do
    case $t in
        pigemma-probe) s=probe/probe.cpp ;;
        pinner)        s=pinner/pinner.c ;;
        membw)         s=membw/membw.c ;;
    esac
    p=$bdir/bin/$t
    if in_list "$t" "${targets[@]}"; then
        if [ -x "$p" ]; then
            printf '%-14s %s\n' "$t" "$p"
        else
            printf '%-14s %s  MISSING after the build\n' "$t" "$p"
            status=3
        fi
    elif [ -f "$src/$s" ]; then
        if [ -x "$p" ]; then
            printf '%-14s %s  (not rebuilt this time: not in --targets)\n' "$t" "$p"
        else
            printf '%-14s (not built: not in --targets)\n' "$t"
        fi
    else
        printf '%-14s (not built: source tools/%s does not exist yet)\n' "$t" "$s"
    fi
done
if [ -f "$bdir/bin/pigemma-tools-build.json" ]; then
    printf '%-14s %s\n' "build info" "$bdir/bin/pigemma-tools-build.json"
fi
printf '%-14s %s s (configure + build, -j %s)\n' "build time" "$((t1 - t0))" "$jobs"

if in_list pigemma-probe "${targets[@]}" && [ -x "$bdir/bin/pigemma-probe" ]; then
    printf '\n== pigemma-probe --help | head -3 ==\n'
    prc=0
    out=$("$bdir/bin/pigemma-probe" --help 2>&1) || prc=$?
    if [ -n "$out" ]; then head -n 3 <<<"$out"; fi
    if [ "$prc" -ge 126 ]; then
        note "ERROR: pigemma-probe --help exited $prc - the binary does not run"
        status=1
    elif [ -z "$out" ]; then
        note "WARNING: pigemma-probe --help printed nothing (rc=$prc)"
    fi
fi
if in_list membw "${targets[@]}" && [ -x "$bdir/bin/membw" ]; then
    mrc=0
    mv=$("$bdir/bin/membw" --version 2>&1) || mrc=$?
    if [ "$mrc" -ge 126 ]; then
        note "ERROR: membw --version exited $mrc - the binary does not run"
        status=1
    elif [ -n "$mv" ]; then
        printf '\n== membw --version ==\n'
        head -n 1 <<<"$mv"
    fi
fi
exit "$status"
INNER
}

cmd_build_tools() {
    local clean=0 jobs=${PIGEMMA_BUILD_JOBS:-0} require_all=0 ignore_lock=0 targets_csv="" t s
    case ${PIGEMMA_REQUIRE_ALL_TOOLS:-} in 1|ON|on|true|TRUE|yes|YES) require_all=1 ;; esac
    set -- "${SUB_ARGS[@]}"
    while [ $# -gt 0 ]; do
        case $1 in
            --clean)        clean=1; shift ;;
            --jobs|-j)      need_val "$1" $#; jobs=$2; shift 2 ;;
            --jobs=*)       jobs=${1#*=}; shift ;;
            -j[0-9]*)       jobs=${1#-j}; shift ;;
            --targets)      need_val "$1" $#; targets_csv=$2; shift 2 ;;
            --targets=*)    targets_csv=${1#*=}; shift ;;
            --require-all)  require_all=1; shift ;;
            --ignore-lock)  ignore_lock=1; shift ;;
            --dry-run)      DRY_RUN=1; shift ;;
            --verbose)      VERBOSE=1; shift ;;
            -h|--help)      usage; exit 0 ;;
            *) die "$EXIT_USAGE" "build-tools: unknown argument '$1' (see --help)" ;;
        esac
    done
    if ! [[ $jobs =~ ^[0-9]+$ ]] || [ "$jobs" -gt 1024 ]; then
        die "$EXIT_USAGE" "build-tools: --jobs '$jobs' must be an integer 1..1024 (0 = nproc)"
    fi
    if [ "$PI_EMU" = 1 ] || { [ -n "$PLATFORM" ] && [ "$PLATFORM" != linux/amd64 ]; }; then
        die "$EXIT_USAGE" "build-tools builds the x86-64 desktop tools (AVX2/AVX-512) into $TOOLS_BUILD_REL; for an aarch64 build run pi/02_build_llama.sh inside the emulated Pi: bash pc/docker_run.sh --pi-emu -- bash pi/02_build_llama.sh ..."
    fi
    if [ ! -f "$REPO_POSIX/tools/CMakeLists.txt" ]; then
        die "$EXIT_MISSING" "build-tools: $REPO_POSIX/tools/CMakeLists.txt does not exist"
    fi
    if [ ! -f "$REPO_POSIX/third_party/llama.cpp/CMakeLists.txt" ]; then
        die "$EXIT_MISSING" "build-tools: llama.cpp sources missing at $REPO_POSIX/third_party/llama.cpp"
    fi

    local targets=() missing=() requested=()
    if [ -n "$targets_csv" ]; then
        IFS=, read -r -a requested <<<"$targets_csv"
        for t in "${requested[@]}"; do
            t=${t// /}
            if [ -z "$t" ]; then continue; fi
            in_list "$t" "${ALL_TOOLS[@]}" || die "$EXIT_USAGE" "build-tools: unknown target '$t' (choose from: ${ALL_TOOLS[*]})"
            s=$(tool_source "$t")
            if [ ! -f "$REPO_POSIX/tools/$s" ]; then
                die "$EXIT_MISSING" "build-tools: target $t requested but tools/$s does not exist"
            fi
            if ! in_list "$t" "${targets[@]}"; then targets+=("$t"); fi
        done
    else
        for t in "${ALL_TOOLS[@]}"; do
            s=$(tool_source "$t")
            if [ -f "$REPO_POSIX/tools/$s" ]; then targets+=("$t"); else missing+=("$t (tools/$s)"); fi
        done
        if [ "${#missing[@]}" -gt 0 ]; then
            if [ "$require_all" = 1 ]; then
                die "$EXIT_MISSING" "build-tools --require-all: missing tool sources: ${missing[*]}"
            fi
            warn "build-tools: skipping tools whose source does not exist yet: ${missing[*]}"
        fi
    fi
    if [ "${#targets[@]}" -eq 0 ]; then
        die "$EXIT_MISSING" "build-tools: no tool source exists yet (expected tools/probe/probe.cpp, tools/pinner/pinner.c, tools/membw/membw.c)"
    fi

    build_inner_script
    prepare_common
    docker_run_exec "$IMAGE" "${ENTRY[@]}" bash -c "$BUILD_INNER" pigemma-build-tools \
        "$clean" "$jobs" "$require_all" "$ignore_lock" "${targets[@]}"
    exit "$DOCKER_RC"
}

# ---------------------------------------------------------------------------
# subcommand: vm-df
# ---------------------------------------------------------------------------
# detect_vm_disk - Git Bash only: where Docker Desktop keeps its data vhdx and how much
# room the Windows drive under it has. Sets VMDISK_DIR (Windows form), VMDISK_FILE,
# VMDISK_BYTES, VMDISK_AVAIL, VMDISK_SOURCE (all may stay empty).
detect_vm_disk() {
    VMDISK_DIR="" VMDISK_FILE="" VMDISK_BYTES="" VMDISK_AVAIL="" VMDISK_SOURCE=""
    if [ "$IS_MSYS" != 1 ] || ! command -v cygpath >/dev/null 2>&1; then return 0; fi
    local f raw win dir="" appdata="" localappdata="" dbs='\\' bs='\'
    if [ -n "${APPDATA:-}" ]; then appdata=$(cygpath -u -- "$APPDATA" 2>/dev/null) || appdata=""; fi
    if [ -n "${LOCALAPPDATA:-}" ]; then localappdata=$(cygpath -u -- "$LOCALAPPDATA" 2>/dev/null) || localappdata=""; fi
    if [ -n "$appdata" ]; then
        for f in "$appdata/Docker/settings-store.json" "$appdata/Docker/settings.json"; do
            if [ ! -f "$f" ]; then continue; fi
            raw=$(grep -o '"CustomWslDistroDir"[[:space:]]*:[[:space:]]*"[^"]*"' -- "$f" 2>/dev/null | head -n 1) || raw=""
            if [ -z "$raw" ]; then continue; fi
            win=${raw#*:}
            win=${win#*\"}
            win=${win%\"}
            win=${win//"$dbs"/"$bs"}
            if [ -n "$win" ]; then
                dir=$(cygpath -u -- "$win" 2>/dev/null) || dir=""
                VMDISK_SOURCE="CustomWslDistroDir in $(cygpath -m -- "$f" 2>/dev/null || printf '%s' "$f")"
                break
            fi
        done
    fi
    if [ -z "$dir" ] && [ -n "$localappdata" ]; then
        dir=$localappdata/Docker/wsl
        VMDISK_SOURCE="Docker Desktop default location (%LOCALAPPDATA%\\Docker\\wsl)"
    fi
    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
        VMDISK_SOURCE=""
        return 0
    fi
    VMDISK_DIR=$(cygpath -m -- "$dir" 2>/dev/null) || VMDISK_DIR=$dir
    for f in "$dir/disk/docker_data.vhdx" "$dir/data/ext4.vhdx" "$dir/docker_data.vhdx"; do
        if [ -f "$f" ]; then
            VMDISK_FILE=$(cygpath -m -- "$f" 2>/dev/null) || VMDISK_FILE=$f
            VMDISK_BYTES=$(stat -c %s -- "$f" 2>/dev/null) || VMDISK_BYTES=""
            break
        fi
    done
    VMDISK_AVAIL=$(df -P -B1 -- "$dir" 2>/dev/null | awk 'NR==2 {print $4}') || VMDISK_AVAIL=""
    if ! [[ $VMDISK_AVAIL =~ ^[0-9]+$ ]]; then VMDISK_AVAIL=""; fi
}

cmd_vm_df() {
    local mode=text
    set -- "${SUB_ARGS[@]}"
    while [ $# -gt 0 ]; do
        case $1 in
            --bytes) mode=bytes; shift ;;
            --json)  mode=json; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            --verbose) VERBOSE=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "$EXIT_USAGE" "vm-df: unknown argument '$1' (see --help)" ;;
        esac
    done
    local img=${PIGEMMA_VMDF_IMAGE:-}
    if [ "$DRY_RUN" = 1 ]; then
        print_cmd run --rm --network none "${img:-$DEFAULT_BUILD_IMAGE}" df -P -B1 /
        if [ "$mode" != bytes ]; then print_cmd system df; fi
        exit 0
    fi
    docker_preflight
    if [ -z "$img" ]; then
        if image_present "$DEFAULT_BUILD_IMAGE"; then
            img=$DEFAULT_BUILD_IMAGE
        elif image_present alpine:latest; then
            img=alpine:latest
        else
            img=alpine:latest
            warn "neither $DEFAULT_BUILD_IMAGE nor alpine:latest is local; docker will pull alpine:latest"
        fi
    fi
    local errf df_out rc=0 line fs size used avail pct mnt
    mk_tmp errf
    df_out=$(docker run --rm --network none "$img" df -P -B1 / 2>"$errf") || rc=$?
    df_out=${df_out//$'\r'/}
    if [ "$rc" -ne 0 ]; then
        die "$EXIT_ERROR" "vm-df: 'docker run $img df -P -B1 /' failed (rc=$rc): $(tail -n 3 -- "$errf" | tr '\r\n' '  ')"
    fi
    line=$(sed -n '2p' <<<"$df_out")
    read -r fs size used avail pct mnt <<<"$line" || true
    if ! [[ ${size:-} =~ ^[0-9]+$ && ${used:-} =~ ^[0-9]+$ && ${avail:-} =~ ^[0-9]+$ ]]; then
        die "$EXIT_ERROR" "vm-df: cannot parse df output: $line"
    fi
    detect_vm_disk
    local eff=$avail
    if [ -n "$VMDISK_AVAIL" ] && [ "$VMDISK_AVAIL" -lt "$eff" ]; then eff=$VMDISK_AVAIL; fi

    case $mode in
        bytes)
            printf '%s\n' "$eff"
            ;;
        text)
            printf '%s\n\n' "$df_out"
            local sysdf src=0
            sysdf=$(docker system df 2>&1) || src=$?
            sysdf=${sysdf//$'\r'/}
            if [ "$src" -ne 0 ]; then warn "docker system df failed (rc=$src)"; fi
            printf '%s\n\n' "$sysdf"
            printf 'vm_root_image=%s\n' "$img"
            printf 'vm_root_filesystem=%s\n' "$fs"
            printf 'vm_root_size_bytes=%s\n' "$size"
            printf 'vm_root_used_bytes=%s\n' "$used"
            printf 'vm_root_avail_bytes=%s\n' "$avail"
            if [ -n "$VMDISK_DIR" ]; then
                printf 'host_vm_disk_dir=%s\n' "$VMDISK_DIR"
                printf 'host_vm_disk_file=%s\n' "$VMDISK_FILE"
                printf 'host_vm_disk_file_bytes=%s\n' "$VMDISK_BYTES"
                printf 'host_vm_disk_drive_avail_bytes=%s\n' "$VMDISK_AVAIL"
                printf 'host_vm_disk_source=%s\n' "$VMDISK_SOURCE"
            fi
            printf 'vm_effective_avail_bytes=%s\n' "$eff"
            say "vm_root_avail is the room inside the VM's sparse virtual disk; it can only grow into the free space of the host drive that holds it (host_vm_disk_*), so gate on vm_effective_avail_bytes."
            ;;
        json)
            local sysdf_json="" l first=1 src=0 sysdf_raw
            sysdf_raw=$(docker system df --format '{{json .}}' 2>/dev/null) || src=$?
            sysdf_raw=${sysdf_raw//$'\r'/}
            if [ "$src" -ne 0 ]; then warn "docker system df failed (rc=$src)"; fi
            while IFS= read -r l; do
                case $l in
                    \{*\}) if [ "$first" = 1 ]; then first=0; else sysdf_json+=", "; fi; sysdf_json+=$l ;;
                esac
            done <<<"$sysdf_raw"
            local host_json=null
            if [ -n "$VMDISK_DIR" ]; then
                host_json="{\"dir\": $(json_str "$VMDISK_DIR"), \"file\": $(json_str "$VMDISK_FILE"), \"file_bytes\": $(json_num_or_null "$VMDISK_BYTES"), \"drive_avail_bytes\": $(json_num_or_null "$VMDISK_AVAIL"), \"source\": $(json_str "$VMDISK_SOURCE")}"
            fi
            printf '{"image": %s, "vm_root": {"filesystem": %s, "size_bytes": %s, "used_bytes": %s, "avail_bytes": %s, "mount": %s}, "host_vm_disk": %s, "vm_effective_avail_bytes": %s, "docker_system_df": [%s], "note": %s}\n' \
                "$(json_str "$img")" "$(json_str "$fs")" "$size" "$used" "$avail" "$(json_str "${mnt:-/}")" \
                "$host_json" "$eff" "$sysdf_json" \
                "$(json_str "vm_root avail is the room inside the VM's sparse virtual disk, which grows on the host drive in host_vm_disk; vm_effective_avail_bytes = min of the two (host side only known from Git Bash).")"
            ;;
    esac
    exit 0
}

# ---------------------------------------------------------------------------
# subcommand: build-image
# ---------------------------------------------------------------------------
cmd_build_image() {
    local tag="" dockerfile plat=()
    set -- "${SUB_ARGS[@]}"
    while [ $# -gt 0 ]; do
        case $1 in
            --tag)   need_val "$1" $#; tag=$2; shift 2 ;;
            --tag=*) tag=${1#*=}; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            --verbose) VERBOSE=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "$EXIT_USAGE" "build-image: unknown argument '$1' (see --help)" ;;
        esac
    done
    if [ -z "$tag" ]; then tag=$IMAGE; fi
    if [ "$PI_EMU" = 1 ]; then
        dockerfile=Dockerfile.pi-emu
        plat=(--platform "${PLATFORM:-linux/arm64}")
    else
        dockerfile=Dockerfile.build
        if [ -n "$PLATFORM" ]; then plat=(--platform "$PLATFORM"); fi
    fi
    if [ ! -f "$REPO_POSIX/docker/$dockerfile" ]; then
        die "$EXIT_MISSING" "build-image: $REPO_POSIX/docker/$dockerfile does not exist"
    fi
    local args=(build "${plat[@]}" -t "$tag" -f "$REPO_HOST/docker/$dockerfile" "$REPO_HOST/docker")
    if [ "$DRY_RUN" = 1 ]; then
        print_cmd "${args[@]}"
        exit 0
    fi
    docker_preflight
    if [ "$VERBOSE" = 1 ]; then print_cmd "${args[@]}" >&2; fi
    local rc=0
    docker "${args[@]}" || rc=$?
    exit "$rc"
}

# ---------------------------------------------------------------------------
case $SUBCMD in
    run)         cmd_run ;;
    build-tools) cmd_build_tools ;;
    vm-df)       cmd_vm_df ;;
    build-image) cmd_build_image ;;
    *)           die "$EXIT_USAGE" "internal: unknown subcommand '$SUBCMD'" ;;
esac
