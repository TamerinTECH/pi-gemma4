# shellcheck shell=bash
# pi/lib/common.sh - shared helpers for every pi/*.sh script of pi-gemma4 (bash >= 4.4).
#
# Runs on Debian 13 aarch64 on a Raspberry Pi 5 (16 KiB pages), inside an arm64/x86 Debian container
# without systemd/sudo/device-tree/boot config (degrades with logged warnings), and under Git Bash for
# the unit tests (tests/test_pi_scripts.py, via PIGEMMA_FAKE_PROC fixtures).
#
# SAFETY: this library only READS /proc, /sys, /boot and command output. It never kills, stops,
# restarts or reconfigures anything, never installs packages and never needs root.
#
# ---------------------------------------------------------------------------------------------- usage
#     #!/usr/bin/env bash
#     set -euo pipefail
#     PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     . "$PI_DIR/lib/common.sh"          # sources pi/pi_config.env, exports the config, defines functions
#     pigemma_init                        # validates the config, mkdir $PIGEMMA_OUT, prints effective values
#     assert_safe_host                    # exit 4 on a protected host unless PIGEMMA_I_KNOW=1
#     require_verdict_ok --allow-slow     # fresh facts.json + pigemma.feasibility gate (exit 3/4 otherwise)
#
# The library can also be executed for quick checks:  bash pi/lib/common.sh <command> [args]
#     facts | write-facts [OUT] | safe-host | not-busy | verdict [opts] | jobs | config | storage
#     fmt-gb BYTES | mem-available-kb | free-disk PATH | version | help
#
# --------------------------------------------------------------------------------------- exit codes
# Shared by every pi script and by pigemma.feasibility:
#     0 ok | 1 usage / internal error | 2 only slow streaming possible | 3 not viable / precondition
#     failed (RAM, disk, missing tool) | 4 refused (protected or busy host)
#
# ----------------------------------------------------------------------------------- public functions
#   log MSG...                 info line on stderr: "[pigemma <script> HH:MM:SS] MSG" (PIGEMMA_QUIET=1 hides)
#   warn MSG...                "WARNING: MSG" on stderr (always shown)
#   die [CODE] MSG...          print a tagged message and exit CODE (default 1). Inside $(...) it only
#                              leaves the subshell: call it at the top level of a script.
#   have_cmd CMD               0 if CMD is on PATH
#   need_cmd CMD [APT_PKG] [WHY]   die 3 with an 'apt-get install' hint when CMD is missing (never installs)
#   banner LINE...             boxed lines on stderr
#   print_config               effective configuration (pi_config.env + environment) on stderr
#   pigemma_init               validate config (die 1), mkdir PIGEMMA_OUT, print_config, emulation notice
#   collect_facts              facts.json (manifest data_formats 'facts.json' + extras) on stdout
#   write_facts [OUT]          collect_facts into OUT (default $PIGEMMA_OUT/facts.json) atomically
#   protected_host_reasons     0 if protected; fills array PG_PROTECTED_REASONS (never exits)
#   assert_safe_host           die 4 if the hostname is in DENY_HOSTS / catalog protected_hosts or a running
#                              service matches PROTECTED_SERVICE_GLOBS / catalog protected_service_globs,
#                              unless PIGEMMA_I_KNOW=1 (then loud warning, PIGEMMA_PROTECTED_OVERRIDE=1)
#   assert_not_busy            die 4 if loadavg1 > PIGEMMA_MAX_LOADAVG (default 2.0 x max(1, nproc/4)) or
#                              effective MemAvailable < 1 GB (same rule as pigemma.feasibility)
#   require_verdict_ok [--want X] [--allow-slow] [--sim F] [--facts F] [--no-collect] [--json OUT]
#                      [--quiet] [-- EXTRA_FEASIBILITY_ARGS...]
#                              writes fresh facts, runs python3 -m pigemma.feasibility --facts
#                              $PIGEMMA_OUT/facts.json --json $PIGEMMA_OUT/feasibility.json; returns 0 on
#                              verdict 0 (or 2 with --allow-slow), else dies with the verdict code.
#                              Sets PIGEMMA_VERDICT_RC and PIGEMMA_FEASIBILITY_JSON.
#   feasibility_env [same options]  like require_verdict_ok but with --emit env, and SOURCES the result into
#                              the current shell: MODEL_ID MODEL_REPO MODEL_FILE MODEL_BYTES MODEL_QUANT
#                              MODEL_FAMILY MODEL_SHA256 MODEL_REVISION MODEL_URL LLAMA_FLAGS FITS_REPACKED
#                              DRAFT_* PIN_BUDGET_BYTES VERDICT VERDICT_EXIT PROJECTED_TOK_S
#   run_feasibility ARGS...    python3 -m pigemma.feasibility ARGS (cwd = PIGEMMA_HOME); returns its code
#   pick_jobs [--auto]         BUILD_JOBS if numeric, else min(nproc, floor(MemAvailable_GiB / 1.2), 4) >= 1
#   fmt_gb BYTES [DEC]         "14.44 GB" (1 GB = 1e9 B); "n/a" for non-numbers.  fmt_gib: GiB (2^30)
#   host_name | kernel_release | page_size | cpu_count | loadavg1 | mem_total_kb | mem_available_kb
#   free_disk_bytes PATH | cpu_governor | hat_product | pciex1_gen_cfg | memlock_limit | boot_config_path
#                              single values on stdout (empty when unknown); all honour PIGEMMA_FAKE_PROC.
#                              mem_available_kb is min(MemAvailable, cgroup memory.max - non-reclaimable use).
#   detect_storage [DIR]       sets STORAGE_DEV STORAGE_DISK STORAGE_TYPE (nvme|usb-ssd|microsd|unknown)
#                              STORAGE_ROTATIONAL STORAGE_LINK_SPEED STORAGE_LINK_WIDTH STORAGE_READ_AHEAD_KB
#                              STORAGE_TRANSPORT STORAGE_FSTYPE STORAGE_MOUNT and array STORAGE_NOTES for DIR
#                              (default MODEL_DIR) in the current shell
#   require_free_disk PATH BYTES [WHAT] [FACTOR]   die 3 unless free >= max(BYTES x FACTOR, BYTES +
#                              KEEP_FREE_GB x 1e9 when PATH is on the root filesystem)
#   require_mem_available BYTES [WHAT]  die 3 unless effective MemAvailable >= BYTES
#   mlock_allowed              0 if model weights may be mlocked (ALLOW_MLOCK=1, or auto on an unprotected host)
#   safe_llama_flags "FLAGS"   FLAGS with '-lm mmap+mlock' -> '-lm mmap', '-lm mlock' -> '-lm none', '--mlock'
#                              dropped when mlock_allowed says no (e.g. LLAMA_FLAGS on context-friend-1)
#   is_emulated | in_container | is_fake   predicates
#   pigemma_python             prints a Python >= 3.10 (PIGEMMA_PYTHON, python3, python); 1 if none
#   native_path PATH           PATH as a native program sees it (Git Bash: cygpath -m), else unchanged
#   json_get FILE KEY.PATH     value from a JSON file (strings raw, null -> empty); 1 if missing
#   llama_bin TOOL | tools_bin TOOL   ${PIGEMMA_LLAMA_BIN:-$LLAMA_DIR/build/bin}/TOOL,
#                              ${PIGEMMA_TOOLS_BIN:-$PIGEMMA_HOME/tools/build/bin}/TOOL
#   run_nice CMD...            nice -n $NICE_LEVEL CMD ;  run_nice_io CMD...  + ionice -c3 when permitted
#   add_cleanup "SHELL CODE"   run at EXIT in reverse order (installs the EXIT trap: do not set your own)
#
# ------------------------------------------------------------------------------- PIGEMMA_FAKE_PROC
# PIGEMMA_FAKE_PROC=<dir> (tests only) replaces every system read by a fixture file and NEVER falls back
# to the real system (missing fixture file = "not available", exactly like a bare container):
#     /proc/<x>                 -> <dir>/<x>            (meminfo, cpuinfo, loadavg, device-tree/hat/product,
#                                                        device-tree/model, 1/comm ...)
#     any other absolute path   -> <dir><path>          (/sys/..., /boot/firmware/config.txt, /.dockerenv)
#     command output            -> <dir>/<name>:
#         hostname, kernel (uname -r), pagesize (getconf PAGESIZE), nproc, ulimit_l (ulimit -l),
#         systemctl.txt  (systemctl list-units --type=service --state=running --no-legend --plain;
#                         absent = systemd not running),
#         findmnt.txt    (findmnt -n -P -T DIR -o SOURCE,FSTYPE,TARGET,MAJ:MIN, or just the SOURCE),
#         lsblk.txt      (lsblk -dn -o NAME,TRAN: "<disk> <tran>" per line),
#         df.txt         (df -B1 -P DIR: header + one line),
#         throttled      (vcgencmd get_throttled), tools/<cmake|gcc|git|python3|fio|curl> (--version)
#     Partition -> disk uses /sys/class/block when present, else name rules (mmcblk0p2 -> mmcblk0,
#     nvme0n1p1 -> nvme0n1, sda1 -> sda).
#
# ------------------------------------------------------------------------------- design decisions
# * facts.json keeps every data_formats key and adds extras that pigemma.feasibility either uses
#   (cgroup_mem_max, emulated, device) or ignores (facts_version, collector, cpu_has, board_model,
#   meminfo_available_kb, services_source, cpu_temp_c, throttled, in_container, boot_config, warnings,
#   storage.{disk, transport, fstype, mount, notes}, model_dir_on_root_fs, fake_proc).
# * ram_available_kb is the EFFECTIVE value min(MemAvailable, cgroup headroom) so a --memory capped
#   container reports what it can really use; the raw MemAvailable is meminfo_available_kb.
# * running_services keeps full unit names ("contextfriend-box.service"); globs match the name with and
#   without the ".service" suffix, case-insensitively: a superset of pigemma.feasibility's rule (fnmatch
#   on the lower-cased full name), so the bash gate is never weaker than the Python one.
# * When systemd is not running (container) the services list is empty, services_source says why, and
#   process names (/proc/*/comm) are scanned for the protected globs as a fallback (real hosts only).
# * The catalog's protected_hosts / protected_service_globs are enforced in addition to DENY_HOSTS /
#   PROTECTED_SERVICE_GLOBS, so the bash gate and pigemma.feasibility always agree.
# * DENY_HOSTS defaults to empty (FACTS.md 2026-10-05: context-friend-1 is now the real target Pi);
#   context-friend-1 is still refused through its running contextfriend-* services (and through
#   catalog.protected_hosts while the catalog lists it) unless PIGEMMA_I_KNOW=1.
# * Usage/internal errors exit 1 (2 is taken by "slow streaming only"), matching pigemma.feasibility.

# Guard against double sourcing (not exported, so child scripts source it again).
if [ -n "${_PIGEMMA_COMMON_LOADED:-}" ]; then
    return 0 2>/dev/null || true
fi
_PIGEMMA_COMMON_LOADED=1

PIGEMMA_COMMON_VERSION="1.0.0"
PIGEMMA_FACTS_VERSION=1

# Exit codes (pigemma.feasibility uses the same numbers).
EXIT_OK=0
EXIT_ERROR=1
EXIT_SLOW=2
EXIT_NOT_VIABLE=3
EXIT_REFUSED=4

# C0 control characters other than \t \n \r (stripped from JSON strings; literal list, locale independent).
_PG_CTRL=$'\001\002\003\004\005\006\007\010\013\014\016\017\020\021\022\023\024\025\026\027\030\031\032\033\034\035\036\037'

# ================================================================================= logging / errors
_pg_emit() { # _pg_emit TAG MSG...
    local tag=$1 ts line
    shift
    printf -v ts '%(%H:%M:%S)T' -1 2>/dev/null || ts="--:--:--"
    line="[pigemma ${PIGEMMA_SCRIPT:-common.sh} ${ts}] ${tag:+$tag: }$*"
    printf '%s\n' "$line" >&2
    if [ -n "${PIGEMMA_LOG_FILE:-}" ]; then
        printf '%s\n' "$line" >>"$PIGEMMA_LOG_FILE" 2>/dev/null || true
    fi
    return 0
}

log() {
    if [ "${PIGEMMA_QUIET:-0}" != 1 ]; then
        _pg_emit "" "$*"
    fi
    return 0
}

warn() {
    _pg_emit "WARNING" "$*"
}

die() { # die [CODE] MSG...
    local code=1 tag
    if [ $# -gt 0 ] && [[ $1 =~ ^[0-9]+$ ]]; then
        code=$1
        shift
    fi
    if [ "${#code}" -gt 3 ] || [ "$code" -gt 255 ]; then
        code=1
    fi
    case $code in
        0) tag="DONE" ;;
        2) tag="STOP (exit 2: only slow streaming possible)" ;;
        3) tag="STOP (exit 3: not viable / precondition failed)" ;;
        4) tag="REFUSED (exit 4)" ;;
        *) tag="ERROR (exit $code)" ;;
    esac
    _pg_emit "$tag" "${*:-no message}"
    exit "$code"
}

banner() { # banner LINE...
    local l bar="##########################################################################"
    printf '%s\n' "$bar" >&2
    for l in "$@"; do
        printf '#  %s\n' "$l" >&2
    done
    printf '%s\n' "$bar" >&2
    return 0
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

need_cmd() { # need_cmd CMD [APT_PKG] [WHY]
    local cmd=${1:?need_cmd: command name required} pkg=${2:-${1:-}} why=${3:-}
    if have_cmd "$cmd"; then
        return 0
    fi
    die 3 "required command '$cmd' not found${why:+ ($why)}. Install it yourself with: sudo apt-get install -y $pkg (pigemma never installs packages without an explicit flag)"
}

_pg_truthy() {
    case ${1:-} in
        1 | [Yy] | [Yy][Ee][Ss] | [Tt][Rr][Uu][Ee] | [Oo][Nn]) return 0 ;;
    esac
    return 1
}

_pg_is_uint() { [[ ${1:-} =~ ^[0-9]+$ ]]; }
_pg_is_num() { [[ ${1:-} =~ ^[0-9]+([.][0-9]+)?$ ]]; }

_pg_trim() { # _pg_trim STRING -> _R (leading/trailing whitespace removed)
    _R=$1
    _R=${_R#"${_R%%[![:space:]]*}"}
    _R=${_R%"${_R##*[![:space:]]}"}
}

_pg_timeout() { # _pg_timeout SECONDS CMD... (plain CMD when coreutils timeout is missing)
    local s=$1
    shift
    if have_cmd timeout; then
        timeout "$s" "$@"
    else
        "$@"
    fi
}

# ======================================================================================== fake root
is_fake() { [ -n "${PIGEMMA_FAKE_PROC:-}" ]; }

_pg_p() { # _pg_p SYSTEM_PATH -> _P (mapped into PIGEMMA_FAKE_PROC when set)
    _P=$1
    if [ -n "${PIGEMMA_FAKE_PROC:-}" ]; then
        case $1 in
            /proc/*) _P="$PIGEMMA_FAKE_PROC/${1#/proc/}" ;;
            /*) _P="$PIGEMMA_FAKE_PROC$1" ;;
        esac
    fi
    return 0
}

_pg_read1() { # _pg_read1 SYSTEM_PATH -> _R = first line, NUL bytes dropped, trimmed; 1 if unreadable/empty
    _R=""
    _pg_p "$1"
    if [ ! -f "$_P" ] || [ ! -r "$_P" ]; then
        return 1
    fi
    local line=""
    IFS= read -r line <"$_P" 2>/dev/null || true
    if [ -z "$line" ] && [ -s "$_P" ]; then
        # device-tree strings end in NUL and have no newline; strip NULs explicitly
        line=$(tr -d '\000' <"$_P" 2>/dev/null || true)
        line=${line%%$'\n'*}
    fi
    _pg_trim "$line"
    [ -n "$_R" ]
}

_pg_fake_cmd() { # _pg_fake_cmd NAME -> _R = whole fixture file <fake>/<NAME>; 1 if absent
    _R=""
    local f="$PIGEMMA_FAKE_PROC/$1"
    if [ ! -f "$f" ]; then
        return 1
    fi
    _R=$(tr -d '\000\r' <"$f")
    return 0
}

# ===================================================================================== JSON encoding
_pg_jstr() { # -> _J = JSON string literal
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    s=${s//[$_PG_CTRL]/}
    _J="\"$s\""
}

_pg_jstrn() { # -> _J = string literal, or null when empty
    if [ -n "${1:-}" ]; then
        _pg_jstr "$1"
    else
        _J=null
    fi
}

_pg_jnum() { # -> _J = JSON number (integer or decimal, leading zeros removed) or null
    local re='^(-?)0*([0-9]+([.][0-9]+)?)$'
    if [[ ${1:-} =~ $re ]]; then
        _J="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
        if [ "$_J" = "-0" ]; then _J=0; fi
    else
        _J=null
    fi
}

_pg_jbool() { # -> _J = true / false / null
    case ${1:-} in
        1 | [Tt][Rr][Uu][Ee] | [Yy][Ee][Ss]) _J=true ;;
        0 | [Ff][Aa][Ll][Ss][Ee] | [Nn][Oo]) _J=false ;;
        *) _J=null ;;
    esac
}

_pg_jarr() { # ITEMS... -> _J = JSON array of strings
    local out="" x
    for x in "$@"; do
        _pg_jstr "$x"
        out+="${out:+, }$_J"
    done
    _J="[$out]"
}

# JSON object writer (pretty, 2-space indent): _pg_put KEY appends the current _J.
_pg_put() { _PG_O+="${_PG_IND}\"$1\": ${_J},"$'\n'; }
_pg_obj_open() {
    _PG_O+="${_PG_IND}\"$1\": {"$'\n'
    _PG_IND+="  "
}
_pg_obj_close() {
    _PG_O=${_PG_O%,$'\n'}$'\n'
    _PG_IND=${_PG_IND%  }
    _PG_O+="${_PG_IND}},"$'\n'
}

# ========================================================================================== host facts
host_name() {
    local h=""
    if is_fake; then
        if _pg_fake_cmd hostname; then
            _pg_trim "${_R%%$'\n'*}"
            h=$_R
        fi
    else
        if [ -r /proc/sys/kernel/hostname ]; then
            IFS= read -r h </proc/sys/kernel/hostname 2>/dev/null || true
        fi
        if [ -z "$h" ]; then h=$(uname -n 2>/dev/null || true); fi
        if [ -z "$h" ]; then h=$(hostname 2>/dev/null || true); fi
    fi
    printf '%s\n' "${h:-unknown}"
}

kernel_release() {
    local k=""
    if is_fake; then
        if _pg_fake_cmd kernel; then
            _pg_trim "${_R%%$'\n'*}"
            k=$_R
        fi
    else
        k=$(uname -r 2>/dev/null || true)
    fi
    printf '%s\n' "$k"
}

page_size() {
    local p=""
    if is_fake; then
        if _pg_fake_cmd pagesize; then
            _pg_trim "${_R%%$'\n'*}"
            p=$_R
        fi
    else
        p=$(getconf PAGESIZE 2>/dev/null || getconf PAGE_SIZE 2>/dev/null || true)
    fi
    if ! _pg_is_uint "$p"; then p=""; fi
    printf '%s\n' "$p"
}

cpu_count() {
    local n=""
    if is_fake; then
        if _pg_fake_cmd nproc; then
            _pg_trim "${_R%%$'\n'*}"
            n=$_R
        fi
        if ! _pg_is_uint "$n"; then
            n=0
            _pg_p /proc/cpuinfo
            if [ -r "$_P" ]; then
                local line
                while IFS= read -r line || [ -n "$line" ]; do
                    case $line in processor*:*) n=$((n + 1)) ;; esac
                done <"$_P"
            fi
        fi
    else
        n=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || true)
    fi
    if ! _pg_is_uint "$n" || [ "$n" -lt 1 ]; then n=1; fi
    printf '%s\n' "$n"
}

loadavg1() {
    local a="" rest
    _pg_p /proc/loadavg
    if [ -r "$_P" ]; then
        read -r a rest <"$_P" 2>/dev/null || true
    fi
    if ! _pg_is_num "$a"; then a=""; fi
    printf '%s\n' "$a"
}

_pg_meminfo() { # sets MI_TOTAL MI_AVAIL MI_SWAP (kB, empty when unknown)
    MI_TOTAL=""
    MI_AVAIL=""
    MI_SWAP=""
    local k v rest mfree="" buf="" cached=""
    _pg_p /proc/meminfo
    if [ -r "$_P" ]; then
        while read -r k v rest || [ -n "${k:-}" ]; do
            case $k in
                MemTotal:) MI_TOTAL=$v ;;
                MemAvailable:) MI_AVAIL=$v ;;
                SwapTotal:) MI_SWAP=$v ;;
                MemFree:) mfree=$v ;;
                Buffers:) buf=$v ;;
                Cached:) cached=$v ;;
            esac
            k=""
        done <"$_P"
    fi
    if ! _pg_is_uint "$MI_TOTAL"; then MI_TOTAL=""; fi
    if ! _pg_is_uint "$MI_SWAP"; then MI_SWAP=""; fi
    if ! _pg_is_uint "$MI_AVAIL"; then
        MI_AVAIL=""
        if _pg_is_uint "$mfree"; then # kernels without MemAvailable: rough MemFree + Buffers + Cached
            _pg_is_uint "$buf" || buf=0
            _pg_is_uint "$cached" || cached=0
            MI_AVAIL=$((mfree + buf + cached))
        fi
    fi
    return 0
}

_pg_cgroup() { # sets CG_MAX (bytes, empty = no finite limit) and CG_HEADROOM (bytes or empty)
    CG_MAX=""
    CG_HEADROOM=""
    local max="" cur="" inact="" stat_file="" inact_key="" k v
    if _pg_read1 /sys/fs/cgroup/memory.max; then # cgroup v2
        max=$_R
        if _pg_read1 /sys/fs/cgroup/memory.current; then cur=$_R; fi
        stat_file=/sys/fs/cgroup/memory.stat
        inact_key=inactive_file
    elif _pg_read1 /sys/fs/cgroup/memory/memory.limit_in_bytes; then # cgroup v1
        max=$_R
        if _pg_read1 /sys/fs/cgroup/memory/memory.usage_in_bytes; then cur=$_R; fi
        stat_file=/sys/fs/cgroup/memory/memory.stat
        inact_key=total_inactive_file
    fi
    # "max" (v2) or a value >= 1e18 (v1 'unlimited' = 9223372036854771712) means no limit
    if ! _pg_is_uint "$max" || [ "${#max}" -gt 18 ]; then
        return 0
    fi
    CG_MAX=$max
    if [ -n "$stat_file" ]; then
        _pg_p "$stat_file"
        if [ -r "$_P" ]; then
            while read -r k v || [ -n "${k:-}" ]; do
                if [ "$k" = "$inact_key" ]; then inact=$v; fi
                k=""
            done <"$_P"
        fi
    fi
    if _pg_is_uint "$cur" && [ "${#cur}" -le 18 ]; then
        _pg_is_uint "$inact" && [ "${#inact}" -le 18 ] || inact=0
        local used=$((cur - inact))
        if [ "$used" -lt 0 ]; then used=0; fi
        CG_HEADROOM=$((max - used))
        if [ "$CG_HEADROOM" -lt 0 ]; then CG_HEADROOM=0; fi
    fi
    return 0
}

mem_total_kb() { # MemTotal in kB (empty when unknown); the cgroup limit is reported separately
    _pg_meminfo
    printf '%s\n' "$MI_TOTAL"
}

mem_available_kb() { # effective: min(MemAvailable, cgroup headroom); 0 when unknown
    _pg_meminfo
    _pg_cgroup
    local a=${MI_AVAIL:-0}
    if [ -n "$CG_HEADROOM" ]; then
        local c=$((CG_HEADROOM / 1024))
        if [ "$c" -lt "$a" ]; then a=$c; fi
    fi
    printf '%s\n' "$a"
}

_pg_arm_part() { # _pg_arm_part IMPLEMENTER PART -> _R = core name
    local impl=${1,,} part=${2,,}
    _R=""
    if [ "$impl" = 0x41 ]; then
        case $part in
            0xd03) _R="Cortex-A53" ;; 0xd04) _R="Cortex-A35" ;; 0xd05) _R="Cortex-A55" ;;
            0xd07) _R="Cortex-A57" ;; 0xd08) _R="Cortex-A72" ;; 0xd09) _R="Cortex-A73" ;;
            0xd0a) _R="Cortex-A75" ;; 0xd0b) _R="Cortex-A76" ;; 0xd0c) _R="Neoverse-N1" ;;
            0xd0d) _R="Cortex-A77" ;; 0xd40) _R="Neoverse-V1" ;; 0xd41) _R="Cortex-A78" ;;
            0xd44) _R="Cortex-X1" ;; 0xd46) _R="Cortex-A510" ;; 0xd47) _R="Cortex-A710" ;;
            0xd48) _R="Cortex-X2" ;; 0xd49) _R="Neoverse-N2" ;; 0xd4b) _R="Cortex-A78C" ;;
            0xd4d) _R="Cortex-A715" ;; 0xd4e) _R="Cortex-X3" ;; 0xd4f) _R="Neoverse-V2" ;;
            0xd80) _R="Cortex-A520" ;; 0xd81) _R="Cortex-A720" ;; 0xd82) _R="Cortex-X4" ;;
        esac
    fi
    if [ -z "$_R" ]; then
        _R="CPU implementer ${1:-?} part ${2:-?}"
    fi
    return 0
}

_pg_cpuinfo() { # sets CPU_MODEL, CPU_FLAGS_STR (space separated) and CPU_BOARD
    CPU_MODEL=""
    CPU_FLAGS_STR=""
    CPU_BOARD=""
    local line key val impl="" part="" mname="" hw=""
    _pg_p /proc/cpuinfo
    if [ -r "$_P" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            case $line in *:*) ;; *) continue ;; esac
            key=${line%%:*}
            val=${line#*:}
            key=${key%"${key##*[![:space:]]}"}
            val=${val#"${val%%[![:space:]]*}"}
            val=${val%"${val##*[![:space:]]}"}
            case $key in
                "model name") if [ -z "$mname" ]; then mname=$val; fi ;;
                Features | flags) if [ -z "$CPU_FLAGS_STR" ]; then CPU_FLAGS_STR=$val; fi ;;
                "CPU implementer") if [ -z "$impl" ]; then impl=$val; fi ;;
                "CPU part") if [ -z "$part" ]; then part=$val; fi ;;
                Model | Hardware) hw=$val ;;
            esac
        done <"$_P"
    fi
    if [ -n "$mname" ]; then
        CPU_MODEL=$mname
    elif [ -n "$part" ]; then
        _pg_arm_part "$impl" "$part"
        CPU_MODEL=$_R
    fi
    CPU_BOARD=$hw
    if _pg_read1 /proc/device-tree/model; then
        CPU_BOARD=$_R
    fi
    return 0
}

_pg_has_flag() { # _pg_has_flag FLAG (uses CPU_FLAGS_STR)
    case " ${CPU_FLAGS_STR} " in *" $1 "*) return 0 ;; esac
    return 1
}

cpu_governor() {
    if _pg_read1 /sys/devices/system/cpu/cpufreq/policy0/scaling_governor; then
        printf '%s\n' "$_R"
    elif _pg_read1 /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; then
        printf '%s\n' "$_R"
    else
        printf '\n'
    fi
}

hat_product() { # /proc/device-tree/hat/product (HAT+ EEPROM), empty when none
    if _pg_read1 /proc/device-tree/hat/product; then
        printf '%s\n' "$_R"
    else
        printf '\n'
    fi
}

boot_config_path() { # PIGEMMA_BOOT_CONFIG, else /boot/firmware/config.txt, else /boot/config.txt; 1 if none
    if [ -n "${PIGEMMA_BOOT_CONFIG:-}" ]; then
        printf '%s\n' "$PIGEMMA_BOOT_CONFIG"
        [ -f "$PIGEMMA_BOOT_CONFIG" ]
        return
    fi
    local c
    for c in /boot/firmware/config.txt /boot/config.txt; do
        _pg_p "$c"
        if [ -f "$_P" ]; then
            printf '%s\n' "$_P"
            return 0
        fi
    done
    return 1
}

pciex1_gen_cfg() { # last active dtparam=...pciex1_gen=N in the boot config; empty when absent
    local cfg line gen="" params
    local re_dt='^[[:space:]]*dtparam[[:space:]]*=(.*)$' re_gen='(^|,)[[:space:]]*pciex1_gen[[:space:]]*=[[:space:]]*([0-9]+)'
    cfg=$(boot_config_path 2>/dev/null) || cfg=""
    if [ -z "$cfg" ] || [ ! -r "$cfg" ]; then
        printf '\n'
        return 0
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%%#*}
        line=${line%$'\r'}
        if [[ $line =~ $re_dt ]]; then
            params=${BASH_REMATCH[1]}
            if [[ $params =~ $re_gen ]]; then
                gen=${BASH_REMATCH[2]}
            fi
        fi
    done <"$cfg"
    printf '%s\n' "$gen"
}

memlock_limit() { # ulimit -l: "unlimited" or kB
    local m=""
    if is_fake; then
        if _pg_fake_cmd ulimit_l; then
            _pg_trim "${_R%%$'\n'*}"
            m=$_R
        fi
    else
        m=$(ulimit -l 2>/dev/null || true)
    fi
    printf '%s\n' "$m"
}

in_container() {
    if _pg_truthy "${PIGEMMA_IN_CONTAINER:-0}"; then return 0; fi
    _pg_p /.dockerenv
    if [ -e "$_P" ]; then return 0; fi
    _pg_p /run/.containerenv
    if [ -e "$_P" ]; then return 0; fi
    return 1
}

is_emulated() { _pg_truthy "${PIGEMMA_EMULATED:-0}"; }

_pg_tool_version() { # _pg_tool_version TOOL -> _R = version string or empty
    _R=""
    local out="" ok=0
    if is_fake; then
        if _pg_fake_cmd "tools/$1"; then
            out=$_R
            ok=1
        fi
    elif have_cmd "$1"; then
        case $1 in
            gcc)
                if out=$(_pg_timeout 10 gcc -dumpfullversion 2>/dev/null); then ok=1
                elif out=$(_pg_timeout 10 gcc -dumpversion 2>/dev/null); then ok=1; fi
                ;;
            *)
                if out=$(_pg_timeout 10 "$1" --version 2>&1); then ok=1; fi
                ;;
        esac
    fi
    _R=""
    if [ "$ok" = 1 ]; then
        out=${out%%$'\n'*}
        local re_ver='([0-9]+([.][0-9]+)+[0-9A-Za-z.+~-]*)'
        if [[ $out =~ $re_ver ]]; then
            _R=${BASH_REMATCH[1]}
        fi
    fi
    return 0
}

_pg_throttled() { # vcgencmd get_throttled -> _R = "0x0" or empty (read-only query)
    _R=""
    local out=""
    if is_fake; then
        if _pg_fake_cmd throttled; then out=$_R; fi
    elif have_cmd vcgencmd; then
        out=$(_pg_timeout 5 vcgencmd get_throttled 2>/dev/null || true)
    fi
    out=${out%%$'\n'*}
    _pg_trim "${out#*=}"
    if ! [[ $_R =~ ^0x[0-9a-fA-F]+$ ]]; then _R=""; fi
    return 0
}

_pg_cpu_temp() { # -> _R = degrees C with one decimal, or empty
    local t
    if _pg_read1 /sys/class/thermal/thermal_zone0/temp && _pg_is_uint "$_R"; then
        t=$_R
        _R="$((t / 1000)).$(((t % 1000) / 100))"
    else
        _R=""
    fi
    return 0
}

# ======================================================================================== services
_pg_glob_match() { # _pg_glob_match NAME GLOB (case-insensitive; NAME with or without .service)
    local n=${1,,} g=${2,,}
    # shellcheck disable=SC2053
    if [[ $n == $g ]] || [[ ${n%.service} == $g ]]; then
        return 0
    fi
    return 1
}

_pg_split_list() { # "a b,c" -> array _L
    local s=${1:-}
    s=${s//,/ }
    _L=()
    read -r -a _L <<<"$s" || true
    return 0
}

_pg_catalog_list() { # _pg_catalog_list KEY -> array _CL from the catalog (1 if no catalog / key)
    _CL=()
    local cat=${PIGEMMA_CATALOG:-$PIGEMMA_HOME/pigemma/data/catalog.json} line items p found=0
    local re='\[([^]]*)\]'
    if [ ! -r "$cat" ]; then
        return 1
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        case $line in *"\"$1\""*:*) ;; *) continue ;; esac
        found=2
        if [[ $line =~ $re ]]; then
            found=1
            items=${BASH_REMATCH[1]}
            local parts=()
            IFS=',' read -r -a parts <<<"$items" || true
            for p in "${parts[@]}"; do
                p=${p//\"/}
                _pg_trim "$p"
                if [ -n "$_R" ]; then _CL+=("$_R"); fi
            done
        fi
        break
    done <"$cat"
    if [ "$found" = 2 ]; then # multi-line array: let Python parse it
        local py out
        if py=$(pigemma_python 2>/dev/null); then
            out=$("$py" -c 'import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8-sig")).get(sys.argv[2]) or []
print("\n".join(str(x) for x in d))' "$(native_path "$cat")" "$1" 2>/dev/null || true)
            while IFS= read -r p; do
                _pg_trim "${p%$'\r'}"
                if [ -n "$_R" ]; then _CL+=("$_R"); fi
            done <<<"$out"
            found=1
        fi
    fi
    [ "$found" = 1 ]
}

_pg_uniq() { # _pg_uniq ITEMS... -> array _U (case-insensitive de-duplication, order kept, empties dropped)
    _U=()
    local x y dup
    for x in "$@"; do
        if [ -z "$x" ]; then continue; fi
        dup=0
        for y in "${_U[@]}"; do
            if [ "${x,,}" = "${y,,}" ]; then
                dup=1
                break
            fi
        done
        if [ "$dup" = 0 ]; then _U+=("$x"); fi
    done
    return 0
}

_pg_protected_globs() { # -> array PG_GLOBS (config + catalog, de-duplicated)
    local all=()
    _pg_split_list "${PROTECTED_SERVICE_GLOBS:-}"
    all=("${_L[@]}")
    if _pg_catalog_list protected_service_globs; then
        all+=("${_CL[@]}")
    fi
    _pg_uniq "${all[@]}"
    PG_GLOBS=("${_U[@]}")
    return 0
}

_pg_services() { # -> array PG_SERVICES (running service units) + PG_SERVICES_SOURCE
    PG_SERVICES=()
    PG_SERVICES_SOURCE=""
    local out="" line tok
    if is_fake; then
        if _pg_fake_cmd systemctl.txt; then
            out=$_R
            PG_SERVICES_SOURCE="systemctl (fake: systemctl.txt)"
        else
            PG_SERVICES_SOURCE="unavailable: systemd is not running (no systemctl.txt in the fake root)"
            return 0
        fi
    else
        if ! have_cmd systemctl; then
            PG_SERVICES_SOURCE="unavailable: systemctl not installed"
        elif [ ! -d /run/systemd/system ]; then
            PG_SERVICES_SOURCE="unavailable: systemd is not running (container?)"
        elif out=$(_pg_timeout 15 systemctl list-units --type=service --state=running --no-legend --plain --no-pager 2>/dev/null); then
            PG_SERVICES_SOURCE="systemctl"
        else
            out=""
            PG_SERVICES_SOURCE="unavailable: systemctl list-units failed"
        fi
        if [ "$PG_SERVICES_SOURCE" != systemctl ]; then
            _pg_proc_comm_scan
            return 0
        fi
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        local toks=()
        read -r -a toks <<<"$line" || true
        for tok in "${toks[@]}"; do
            case $tok in
                *.service)
                    PG_SERVICES+=("$tok")
                    break
                    ;;
            esac
        done
    done <<<"$out"
    return 0
}

_pg_proc_comm_scan() { # fallback when systemd is unavailable: process names matching the protected globs
    local f c g s seen
    _pg_protected_globs
    if [ "${#PG_GLOBS[@]}" -eq 0 ]; then
        return 0
    fi
    for f in /proc/[0-9]*/comm; do
        if [ ! -r "$f" ]; then continue; fi
        c=""
        IFS= read -r c <"$f" 2>/dev/null || true
        if [ -z "$c" ]; then continue; fi
        for g in "${PG_GLOBS[@]}"; do
            if _pg_glob_match "$c" "$g"; then
                seen=0
                for s in "${PG_SERVICES[@]}"; do
                    if [ "$s" = "$c" ]; then seen=1; fi
                done
                if [ "$seen" = 0 ]; then PG_SERVICES+=("$c"); fi
                break
            fi
        done
    done
    PG_SERVICES_SOURCE="$PG_SERVICES_SOURCE; fallback: process names (/proc/*/comm) matching the protected globs"
    return 0
}

# ========================================================================================= storage
_pg_existing_dir() { # nearest existing directory of PATH (fake mode: PATH unchanged)
    local d=${1:-.}
    case $d in
        /* | [A-Za-z]:/* | [A-Za-z]:\\*) ;;
        *) d="$PWD/$d" ;;
    esac
    if is_fake; then
        printf '%s\n' "$d"
        return 0
    fi
    while [ ! -d "$d" ]; do
        case $d in
            */*) d=${d%/*} ;;
            *) d=/ ;;
        esac
        if [ -z "$d" ]; then d=/; fi
    done
    printf '%s\n' "$d"
}

_pg_df() { # _pg_df DIR -> DF_FS DF_AVAIL DF_MOUNT (empty when unknown)
    DF_FS=""
    DF_AVAIL=""
    DF_MOUNT=""
    local out="" line sz used cap
    if is_fake; then
        if _pg_fake_cmd df.txt; then out=$_R; fi
    else
        out=$(df -B1 -P "$1" 2>/dev/null || true)
    fi
    case $out in
        *$'\n'*) line=${out#*$'\n'} ;;
        *) return 0 ;;
    esac
    line=${line%%$'\n'*}
    read -r DF_FS sz used DF_AVAIL cap DF_MOUNT <<<"$line" || true
    if ! _pg_is_uint "$DF_AVAIL"; then DF_AVAIL=""; fi
    return 0
}

free_disk_bytes() { # free_disk_bytes PATH -> bytes available to the user (empty when unknown)
    local d
    d=$(_pg_existing_dir "${1:-$MODEL_DIR}")
    _pg_df "$d"
    printf '%s\n' "$DF_AVAIL"
}

_pg_findmnt() { # _pg_findmnt DIR -> FM_SOURCE FM_FSTYPE FM_TARGET FM_MAJMIN
    FM_SOURCE=""
    FM_FSTYPE=""
    FM_TARGET=""
    FM_MAJMIN=""
    local out=""
    if is_fake; then
        if _pg_fake_cmd findmnt.txt; then out=$_R; fi
    elif have_cmd findmnt; then
        out=$(findmnt -n -P -T "$1" -o SOURCE,FSTYPE,TARGET,MAJ:MIN 2>/dev/null || true)
    fi
    out=${out%%$'\n'*}
    if [ -z "$out" ]; then
        return 0
    fi
    if [[ $out == *'SOURCE="'* ]]; then
        local re_src='(^|[[:space:]])SOURCE="([^"]*)"' re_fs='FSTYPE="([^"]*)"' re_tg='TARGET="([^"]*)"'
        local re_mm='MAJ[:_]MIN="([^"]*)"'
        if [[ $out =~ $re_src ]]; then FM_SOURCE=${BASH_REMATCH[2]}; fi
        if [[ $out =~ $re_fs ]]; then FM_FSTYPE=${BASH_REMATCH[1]}; fi
        if [[ $out =~ $re_tg ]]; then FM_TARGET=${BASH_REMATCH[1]}; fi
        if [[ $out =~ $re_mm ]]; then FM_MAJMIN=${BASH_REMATCH[1]}; fi
        # findmnt -P escapes unsafe characters as \xHH: decode them
        printf -v FM_SOURCE '%b' "$FM_SOURCE"
        printf -v FM_TARGET '%b' "$FM_TARGET"
    else
        read -r FM_SOURCE FM_FSTYPE FM_TARGET FM_MAJMIN <<<"$out" || true
    fi
    return 0
}

_pg_blockname() { # _pg_blockname SOURCE MAJ:MIN -> _R = kernel block name (mmcblk0p2) or empty
    local src=${1%%\[*} mm=${2:-} real=""
    _R=""
    if is_fake; then
        case $src in /dev/*) _R=${src##*/} ;; esac
        return 0
    fi
    if { [ "$src" = /dev/root ] || [ ! -e "$src" ]; } && [ -n "$mm" ] && [ -e "/sys/dev/block/$mm" ]; then
        real=$(readlink -f "/sys/dev/block/$mm" 2>/dev/null || true)
        _R=${real##*/}
        return 0
    fi
    case $src in
        /dev/*)
            real=$(readlink -f "$src" 2>/dev/null || true)
            if [ -z "$real" ]; then real=$src; fi
            _R=${real##*/}
            ;;
    esac
    return 0
}

_pg_parent_disk() { # _pg_parent_disk NAME -> _R = whole-disk name (follows partitions and dm/md slaves)
    local name=$1 depth=0 p link s next
    while [ "$depth" -lt 6 ]; do
        depth=$((depth + 1))
        _pg_p "/sys/class/block/$name"
        p=$_P
        if [ -d "$p" ]; then
            next=""
            if [ -d "$p/slaves" ]; then
                for s in "$p"/slaves/*; do
                    if [ -e "$s" ]; then
                        next=${s##*/}
                        break
                    fi
                done
            fi
            if [ -n "$next" ]; then
                name=$next
                continue
            fi
            if [ -e "$p/partition" ]; then
                link=$(readlink -f "$p" 2>/dev/null || true)
                if [ -n "$link" ]; then
                    link=${link%/*}
                    name=${link##*/}
                fi
            fi
            break
        fi
        # no sysfs entry (fake root, odd container): name rules
        case $name in
            mmcblk*p[0-9]* | nvme*n*p[0-9]* | loop*p[0-9]*) name=${name%p[0-9]*} ;;
            sd*[0-9] | vd*[0-9] | xvd*[0-9] | hd*[0-9]) name=${name%%[0-9]*} ;;
        esac
        break
    done
    _R=$name
    return 0
}

_pg_transport() { # _pg_transport DISK -> _R = lsblk TRAN (usb, sata, nvme, ...) or empty
    _R=""
    local out="" n t rest link
    if is_fake; then
        if _pg_fake_cmd lsblk.txt; then
            out=$_R
            _R=""
            while read -r n t rest || [ -n "${n:-}" ]; do
                if [ "$n" = "$1" ]; then
                    _R=${t:-}
                    break
                fi
                n=""
            done <<<"$out"
        fi
        return 0
    fi
    if have_cmd lsblk; then
        out=$(lsblk -dn -o TRAN "/dev/$1" 2>/dev/null || true)
        _pg_trim "${out%%$'\n'*}"
    fi
    if [ -z "$_R" ]; then
        link=$(readlink -f "/sys/block/$1" 2>/dev/null || true)
        case $link in */usb*) _R=usb ;; esac
    fi
    return 0
}

detect_storage() { # detect_storage [DIR] -> STORAGE_* globals (see the header)
    local want=${1:-$MODEL_DIR} dir src name ctrl f
    STORAGE_DEV=""
    STORAGE_DISK=""
    STORAGE_TYPE=unknown
    STORAGE_ROTATIONAL=""
    STORAGE_LINK_SPEED=""
    STORAGE_LINK_WIDTH=""
    STORAGE_READ_AHEAD_KB=""
    STORAGE_TRANSPORT=""
    STORAGE_FSTYPE=""
    STORAGE_MOUNT=""
    STORAGE_NOTES=()
    dir=$(_pg_existing_dir "$want")
    case $want in
        /* | [A-Za-z]:/* | [A-Za-z]:\\*) ;;
        *) want="$PWD/$want" ;;
    esac
    if [ "$dir" != "$want" ]; then
        STORAGE_NOTES+=("model_dir does not exist yet: measured on its nearest existing parent $dir")
    fi
    _pg_findmnt "$dir"
    if [ -z "$FM_SOURCE" ]; then
        _pg_df "$dir"
        FM_SOURCE=$DF_FS
        FM_TARGET=$DF_MOUNT
    fi
    STORAGE_DEV=$FM_SOURCE
    STORAGE_FSTYPE=$FM_FSTYPE
    STORAGE_MOUNT=$FM_TARGET
    src=${FM_SOURCE%%\[*}
    case $src in
        /dev/*) ;;
        *)
            STORAGE_NOTES+=("model_dir is on '${FM_SOURCE:-?}'${FM_FSTYPE:+ ($FM_FSTYPE)}, not a block device (container overlay / network fs?): storage type 'unknown' -> the catalog's conservative microSD-class numbers apply")
            return 0
            ;;
    esac
    _pg_blockname "$src" "$FM_MAJMIN"
    name=$_R
    if [ -z "$name" ]; then
        STORAGE_NOTES+=("cannot resolve the block device behind $FM_SOURCE: storage type 'unknown'")
        return 0
    fi
    _pg_parent_disk "$name"
    STORAGE_DISK=$_R
    if _pg_read1 "/sys/block/$STORAGE_DISK/queue/rotational"; then STORAGE_ROTATIONAL=$_R; fi
    if _pg_read1 "/sys/block/$STORAGE_DISK/queue/read_ahead_kb"; then STORAGE_READ_AHEAD_KB=$_R; fi
    case $STORAGE_DISK in
        mmcblk*)
            STORAGE_TYPE=microsd
            STORAGE_TRANSPORT=mmc
            if _pg_read1 "/sys/block/$STORAGE_DISK/device/type" && [ "$_R" = MMC ]; then
                STORAGE_NOTES+=("$STORAGE_DISK is eMMC (device/type=MMC): treated as the microsd class")
            fi
            ;;
        nvme*)
            STORAGE_TYPE=nvme
            STORAGE_TRANSPORT=nvme
            ctrl=${STORAGE_DISK%%n[0-9]*}
            if _pg_read1 "/sys/class/nvme/$ctrl/device/current_link_speed"; then STORAGE_LINK_SPEED=$_R; fi
            if _pg_read1 "/sys/class/nvme/$ctrl/device/current_link_width"; then STORAGE_LINK_WIDTH=$_R; fi
            if [ -z "$STORAGE_LINK_SPEED" ]; then
                _pg_p /sys/class/nvme
                for f in "$_P"/*/device/current_link_speed; do
                    if [ -r "$f" ]; then
                        IFS= read -r STORAGE_LINK_SPEED <"$f" || true
                        if [ -r "${f%_speed}_width" ]; then
                            IFS= read -r STORAGE_LINK_WIDTH <"${f%_speed}_width" || true
                        fi
                        STORAGE_NOTES+=("NVMe link read from ${f#"${PIGEMMA_FAKE_PROC:-}"} (controller of $STORAGE_DISK not matched by name)")
                        break
                    fi
                done
            fi
            if [ -z "$STORAGE_LINK_SPEED" ]; then
                STORAGE_NOTES+=("NVMe link speed unknown (no /sys/class/nvme/*/device/current_link_speed)")
            fi
            ;;
        sd*)
            _pg_transport "$STORAGE_DISK"
            STORAGE_TRANSPORT=$_R
            if [ "$STORAGE_TRANSPORT" = usb ]; then
                STORAGE_TYPE=usb-ssd
                if [ "$STORAGE_ROTATIONAL" = 1 ]; then
                    STORAGE_NOTES+=("USB disk reports rotational=1 (many USB-SATA bridges do even for SSDs): if this is a hard disk, expect far below the usb-ssd numbers")
                fi
            else
                STORAGE_NOTES+=("$STORAGE_DISK transport '${STORAGE_TRANSPORT:-?}' is not usb: storage type 'unknown'")
            fi
            ;;
        *)
            STORAGE_NOTES+=("block device $STORAGE_DISK is not mmcblk/nvme/sd: storage type 'unknown'")
            ;;
    esac
    _pg_trim "$STORAGE_LINK_SPEED"
    STORAGE_LINK_SPEED=$_R
    _pg_trim "$STORAGE_LINK_WIDTH"
    STORAGE_LINK_WIDTH=$_R
    return 0
}

# =========================================================================================== facts
collect_facts() { # facts.json on stdout (manifest data_formats 'facts.json' + documented extras)
    local warnings=() host kernel psize nproc la mdir memlock gov hat gen cfg board temp thr dfdir
    local now override="" on_root="" w t
    host=$(host_name)
    kernel=$(kernel_release)
    psize=$(page_size)
    if [ -z "$psize" ]; then warnings+=("page size unknown (getconf PAGESIZE failed)"); fi
    _pg_cpuinfo
    if [ -z "$CPU_FLAGS_STR" ]; then warnings+=("no CPU flags in /proc/cpuinfo"); fi
    nproc=$(cpu_count)
    _pg_meminfo
    _pg_cgroup
    if [ -z "$MI_TOTAL" ]; then warnings+=("MemTotal unknown (/proc/meminfo unreadable)"); fi
    local avail=$MI_AVAIL
    if [ -n "$CG_HEADROOM" ] && [ -n "$avail" ] && [ $((CG_HEADROOM / 1024)) -lt "$avail" ]; then
        avail=$((CG_HEADROOM / 1024))
    fi
    la=$(loadavg1)
    mdir=$MODEL_DIR
    detect_storage "$mdir"
    dfdir=$(_pg_existing_dir "$mdir")
    _pg_df "$dfdir"
    if [ -z "$DF_AVAIL" ]; then warnings+=("free disk unknown (df -B1 -P failed for $dfdir)"); fi
    if [ -n "$DF_MOUNT" ]; then
        if [ "$DF_MOUNT" = / ]; then on_root=1; else on_root=0; fi
    fi
    _pg_services
    case $PG_SERVICES_SOURCE in
        unavailable*) warnings+=("running services unknown: ${PG_SERVICES_SOURCE#unavailable: }") ;;
    esac
    hat=$(hat_product)
    gen=$(pciex1_gen_cfg)
    cfg=$(boot_config_path 2>/dev/null) || cfg=""
    if [ -z "$cfg" ]; then warnings+=("no boot config (/boot/firmware/config.txt) - pciex1_gen_cfg unknown"); fi
    if is_fake && [ -z "${PIGEMMA_BOOT_CONFIG:-}" ]; then cfg=${cfg#"$PIGEMMA_FAKE_PROC"}; fi
    gov=$(cpu_governor)
    memlock=$(memlock_limit)
    board=$CPU_BOARD
    _pg_cpu_temp
    temp=$_R
    _pg_throttled
    thr=$_R
    if _pg_truthy "${PIGEMMA_I_KNOW:-0}"; then override=PIGEMMA_I_KNOW; fi
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)

    for w in "${warnings[@]}" "${STORAGE_NOTES[@]}"; do
        if [ "${PIGEMMA_QUIET:-0}" != 1 ]; then _pg_emit "facts" "$w"; fi
    done

    _PG_O="{"$'\n'
    _PG_IND="  "
    _pg_jnum "$PIGEMMA_FACTS_VERSION"; _pg_put facts_version
    _pg_jstr "pi/lib/common.sh $PIGEMMA_COMMON_VERSION"; _pg_put collector
    _pg_jstr "$host"; _pg_put hostname
    _pg_jstrn "$kernel"; _pg_put kernel
    _pg_jnum "$psize"; _pg_put page_size
    _pg_jstrn "$CPU_MODEL"; _pg_put cpu_model
    local flags=()
    read -r -a flags <<<"$CPU_FLAGS_STR" || true
    _pg_jarr "${flags[@]}"; _pg_put cpu_flags
    _pg_obj_open cpu_has
    for t in asimddp i8mm sve fphp asimdhp bf16 avx2 avx512f; do
        if _pg_has_flag "$t"; then _J=true; else _J=false; fi
        _pg_put "$t"
    done
    _pg_obj_close
    _pg_jnum "$nproc"; _pg_put nproc
    _pg_jnum "$MI_TOTAL"; _pg_put ram_total_kb
    _pg_jnum "$avail"; _pg_put ram_available_kb
    _pg_jnum "$MI_AVAIL"; _pg_put meminfo_available_kb
    _pg_jnum "$MI_SWAP"; _pg_put swap_total_kb
    _pg_jnum "$la"; _pg_put loadavg1
    _pg_jstr "$mdir"; _pg_put model_dir
    _pg_obj_open storage
    _pg_jstrn "$STORAGE_DEV"; _pg_put dev
    _pg_jstr "$STORAGE_TYPE"; _pg_put type
    _pg_jbool "$STORAGE_ROTATIONAL"; _pg_put rotational
    _pg_jstrn "$STORAGE_LINK_SPEED"; _pg_put pcie_link_speed
    _pg_jnum "$STORAGE_LINK_WIDTH"; _pg_put pcie_link_width
    _pg_jnum "$STORAGE_READ_AHEAD_KB"; _pg_put read_ahead_kb
    _pg_jstrn "$STORAGE_DISK"; _pg_put disk
    _pg_jstrn "$STORAGE_TRANSPORT"; _pg_put transport
    _pg_jstrn "$STORAGE_FSTYPE"; _pg_put fstype
    _pg_jstrn "$STORAGE_MOUNT"; _pg_put mount
    _pg_jarr "${STORAGE_NOTES[@]}"; _pg_put notes
    _pg_obj_close
    _pg_jnum "$DF_AVAIL"; _pg_put free_disk_bytes
    _pg_jbool "$on_root"; _pg_put model_dir_on_root_fs
    _pg_jarr "${PG_SERVICES[@]}"; _pg_put running_services
    _pg_jstr "$PG_SERVICES_SOURCE"; _pg_put services_source
    _pg_jstrn "$hat"; _pg_put hat_product
    _pg_jnum "$gen"; _pg_put pciex1_gen_cfg
    _pg_jstrn "$cfg"; _pg_put boot_config
    _pg_jstrn "$gov"; _pg_put governor
    if [ "$memlock" = unlimited ]; then _pg_jstr unlimited; else _pg_jnum "$memlock"; fi
    _pg_put memlock_limit
    _pg_obj_open tools
    for t in cmake gcc git python3 fio curl; do
        _pg_tool_version "$t"
        _pg_jstrn "$_R"
        _pg_put "$t"
    done
    _pg_obj_close
    _pg_jstrn "$override"; _pg_put override
    _pg_jstrn "$board"; _pg_put board_model
    _pg_jnum "$temp"; _pg_put cpu_temp_c
    _pg_jstrn "$thr"; _pg_put throttled
    _pg_jnum "$CG_MAX"; _pg_put cgroup_mem_max
    if in_container; then _J=true; else _J=false; fi
    _pg_put in_container
    if is_emulated; then _J=true; else _J=false; fi
    _pg_put emulated
    _pg_jstrn "${PIGEMMA_DEVICE:-}"; _pg_put device
    _pg_jstrn "${PIGEMMA_FAKE_PROC:-}"; _pg_put fake_proc
    _pg_jarr "${warnings[@]}"; _pg_put warnings
    _pg_jstrn "$now"; _pg_put collected_at
    _PG_O=${_PG_O%,$'\n'}$'\n}'
    printf '%s\n' "$_PG_O"
}

write_facts() { # write_facts [OUT] -> OUT (default $PIGEMMA_OUT/facts.json); sets PIGEMMA_FACTS
    local out=${1:-$PIGEMMA_OUT/facts.json} dir tmp
    case $out in
        */*) dir=${out%/*} ;;
        *) dir=. ;;
    esac
    mkdir -p "${dir:-/}" || die 1 "cannot create directory ${dir:-/}"
    tmp="$out.tmp.$$"
    if ! collect_facts >"$tmp"; then
        rm -f "$tmp"
        die 1 "collect_facts failed"
    fi
    mv -f "$tmp" "$out" || die 1 "cannot write $out"
    PIGEMMA_FACTS=$out
    log "facts written: $out"
}

# ===================================================================================== host gates
protected_host_reasons() { # 0 if protected; fills PG_PROTECTED_REASONS (never exits)
    PG_PROTECTED_REASONS=()
    local host h s g deny=() hits=()
    host=$(host_name)
    _pg_split_list "${DENY_HOSTS:-}"
    deny=("${_L[@]}")
    if _pg_catalog_list protected_hosts; then
        deny+=("${_CL[@]}")
    fi
    for h in "${deny[@]}"; do
        if [ -n "$h" ] && [ "${host,,}" = "${h,,}" ]; then
            PG_PROTECTED_REASONS+=("hostname '$host' is deny-listed (DENY_HOSTS / catalog protected_hosts)")
            break
        fi
    done
    _pg_protected_globs
    _pg_services
    for s in "${PG_SERVICES[@]}"; do
        for g in "${PG_GLOBS[@]}"; do
            if [ -n "$g" ] && _pg_glob_match "$s" "$g"; then
                hits+=("$s")
                break
            fi
        done
    done
    if [ "${#hits[@]}" -gt 0 ]; then
        PG_PROTECTED_REASONS+=("protected services running: ${hits[*]} (globs: ${PG_GLOBS[*]})")
    fi
    [ "${#PG_PROTECTED_REASONS[@]}" -gt 0 ]
}

assert_safe_host() { # die 4 on a protected host unless PIGEMMA_I_KNOW=1
    local host r
    host=$(host_name)
    if protected_host_reasons; then
        if _pg_truthy "${PIGEMMA_I_KNOW:-0}"; then
            PIGEMMA_PROTECTED_OVERRIDE=1
            export PIGEMMA_PROTECTED_OVERRIDE
            for r in "${PG_PROTECTED_REASONS[@]}"; do
                warn "PROTECTED HOST OVERRIDE (PIGEMMA_I_KNOW=1): $r"
            done
            warn "rules still in force on '$host': no service is stopped/restarted/killed; nice -n ${NICE_LEVEL:-10}; no mlock of model weights (ALLOW_MLOCK=${ALLOW_MLOCK:-auto}); keep >= ${KEEP_FREE_GB:-2} GB free on /; no config.txt edits or reboots without explicit approval; results must say PIGEMMA_I_KNOW"
            return 0
        fi
        banner "REFUSED: '$host' is a protected host - nothing was changed." \
            "${PG_PROTECTED_REASONS[@]}" \
            "Every pi/*.sh script refuses here (exit 4)." \
            "Override ONLY with explicit authorisation from the host owner: PIGEMMA_I_KNOW=1"
        die 4 "protected host '$host': ${PG_PROTECTED_REASONS[*]}"
    fi
    PIGEMMA_PROTECTED_OVERRIDE=0
    export PIGEMMA_PROTECTED_OVERRIDE
    case $PG_SERVICES_SOURCE in
        unavailable*) warn "host '$host': running services could not be listed (${PG_SERVICES_SOURCE#unavailable: }); hostname and process names checked only" ;;
        *) log "host check: '$host' is not protected (${#PG_SERVICES[@]} running services checked)" ;;
    esac
    return 0
}

assert_not_busy() { # die 4 when loadavg1 or MemAvailable say the host is busy (pigemma.feasibility rule)
    local la n avail_kb limit
    la=$(loadavg1)
    n=$(cpu_count)
    avail_kb=$(mem_available_kb)
    limit=${PIGEMMA_MAX_LOADAVG:-}
    if ! _pg_is_num "$limit"; then
        limit=$(awk -v n="$n" 'BEGIN { x = n / 4; if (x < 1) x = 1; printf "%.2f", 2.0 * x }')
    fi
    if [ -n "$la" ] && awk -v a="$la" -v l="$limit" 'BEGIN { exit !(a + 0 > l + 0) }'; then
        die 4 "busy host: loadavg1 $la > $limit (PIGEMMA_MAX_LOADAVG overrides); try again when the machine is idle"
    fi
    if _pg_is_uint "$avail_kb" && [ $((avail_kb * 1024)) -lt 1000000000 ]; then
        die 4 "busy host: MemAvailable $(fmt_gb $((avail_kb * 1024))) < 1.00 GB"
    fi
    log "host not busy: loadavg1 ${la:-?} <= $limit, MemAvailable $(fmt_gb $((avail_kb * 1024)))"
}

mlock_allowed() { # 0 = model weights may be mlocked
    case ${ALLOW_MLOCK:-auto} in
        1 | [Yy][Ee][Ss] | [Tt][Rr][Uu][Ee]) return 0 ;;
        0 | [Nn][Oo] | [Ff][Aa][Ll][Ss][Ee]) return 1 ;;
    esac
    if protected_host_reasons; then
        return 1
    fi
    return 0
}

safe_llama_flags() { # safe_llama_flags "FLAGS" -> FLAGS, with mlock removed when mlock_allowed says no
    local flags=${1:-} toks=() out=() i t
    if mlock_allowed; then
        printf '%s\n' "$flags"
        return 0
    fi
    read -r -a toks <<<"$flags" || true
    i=0
    while [ "$i" -lt "${#toks[@]}" ]; do
        t=${toks[$i]}
        case $t in
            -lm | --load-mode)
                out+=("$t")
                i=$((i + 1))
                if [ "$i" -lt "${#toks[@]}" ]; then
                    case ${toks[$i]} in
                        mmap+mlock) out+=(mmap) ;;
                        mlock) out+=(none) ;;
                        *) out+=("${toks[$i]}") ;;
                    esac
                fi
                ;;
            --mlock) ;;
            *) out+=("$t") ;;
        esac
        i=$((i + 1))
    done
    if [ "${out[*]}" != "${toks[*]}" ]; then
        warn "mlock removed from the llama.cpp flags (ALLOW_MLOCK=${ALLOW_MLOCK:-auto} on a protected host: the page cache must stay reclaimable for its services): ${out[*]}"
    fi
    printf '%s\n' "${out[*]}"
}

# ============================================================================== resource preconditions
require_free_disk() { # require_free_disk PATH BYTES [WHAT] [FACTOR]
    local path=${1:?require_free_disk: path required} bytes=${2:?require_free_disk: bytes required}
    local what=${3:-this step} factor=${4:-1} dir keep=0 need
    if ! _pg_is_uint "$bytes"; then die 1 "require_free_disk: BYTES must be an integer, got '$bytes'"; fi
    if ! _pg_is_num "$factor"; then die 1 "require_free_disk: FACTOR must be a number, got '$factor'"; fi
    dir=$(_pg_existing_dir "$path")
    _pg_df "$dir"
    if [ -z "$DF_AVAIL" ]; then
        die 3 "cannot determine free disk space for $dir (df failed) - refusing $what"
    fi
    if [ "$DF_MOUNT" = / ]; then
        keep=$(awk -v g="${KEEP_FREE_GB:-2}" 'BEGIN { printf "%.0f", g * 1e9 }')
    fi
    need=$(awk -v b="$bytes" -v f="$factor" -v k="$keep" \
        'BEGIN { a = b * f; c = b + k; n = (a > c) ? a : c; if (n > int(n)) n = int(n) + 1; printf "%.0f", n }')
    if [ "$DF_AVAIL" -lt "$need" ]; then
        die 3 "not enough free disk for $what on ${DF_MOUNT:-$dir}: free $(fmt_gb "$DF_AVAIL"), need $(fmt_gb "$need") (= max(${factor} x $(fmt_gb "$bytes"), $(fmt_gb "$bytes") + keep-free $(fmt_gb "$keep")))"
    fi
    log "free disk OK for $what: $(fmt_gb "$DF_AVAIL") free on ${DF_MOUNT:-$dir}, need $(fmt_gb "$need")"
}

require_mem_available() { # require_mem_available BYTES [WHAT]
    local bytes=${1:?require_mem_available: bytes required} what=${2:-this step} avail_kb
    if ! _pg_is_uint "$bytes"; then die 1 "require_mem_available: BYTES must be an integer, got '$bytes'"; fi
    avail_kb=$(mem_available_kb)
    if [ $((avail_kb * 1024)) -lt "$bytes" ]; then
        die 3 "not enough available RAM for $what: MemAvailable $(fmt_gb $((avail_kb * 1024))), need $(fmt_gb "$bytes")"
    fi
    log "RAM OK for $what: MemAvailable $(fmt_gb $((avail_kb * 1024))) >= $(fmt_gb "$bytes")"
}

pick_jobs() { # pick_jobs [--auto] -> BUILD_JOBS if numeric, else min(nproc, floor(MemAvailable_GiB/1.2), 4)
    local n avail_kb by_mem jobs=4 force_auto=0
    if [ "${1:-}" = --auto ]; then force_auto=1; fi
    n=$(cpu_count)
    avail_kb=$(mem_available_kb)
    _pg_is_uint "$avail_kb" || avail_kb=0
    by_mem=$((avail_kb * 10 / 12582912)) # floor(kB / (1.2 x 1048576))
    if [ "$n" -lt "$jobs" ]; then jobs=$n; fi
    if [ "$by_mem" -lt "$jobs" ]; then jobs=$by_mem; fi
    if [ "$jobs" -lt 1 ]; then jobs=1; fi
    if [ "$force_auto" = 0 ]; then
        case ${BUILD_JOBS:-auto} in
            auto | "") ;;
            *)
                if _pg_is_uint "$BUILD_JOBS" && [ "$BUILD_JOBS" -ge 1 ]; then
                    if [ "$BUILD_JOBS" -gt "$jobs" ]; then
                        warn "BUILD_JOBS=$BUILD_JOBS exceeds the RAM-safe value $jobs (MemAvailable $(fmt_gib $((avail_kb * 1024)))): the build may OOM"
                    fi
                    jobs=$BUILD_JOBS
                else
                    warn "BUILD_JOBS='$BUILD_JOBS' is neither 'auto' nor a positive integer: using $jobs"
                fi
                ;;
        esac
    fi
    printf '%s\n' "$jobs"
}

fmt_gb() { # fmt_gb BYTES [DECIMALS] -> "14.44 GB" (1e9)
    local b=${1:-} d=${2:-2}
    _pg_is_uint "$d" || d=2
    if [[ $b =~ ^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
        awk -v b="$b" -v d="$d" 'BEGIN { f = "%." d "f GB\n"; printf f, b / 1e9 }'
    else
        printf 'n/a\n'
    fi
}

fmt_gib() { # fmt_gib BYTES [DECIMALS] -> "13.45 GiB" (2^30)
    local b=${1:-} d=${2:-2}
    _pg_is_uint "$d" || d=2
    if [[ $b =~ ^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
        awk -v b="$b" -v d="$d" 'BEGIN { f = "%." d "f GiB\n"; printf f, b / 1073741824 }'
    else
        printf 'n/a\n'
    fi
}

# ============================================================================== python / feasibility
native_path() { # path as a native (non-MSYS) program sees it
    if [ -n "${_PG_MSYS:-}" ] && have_cmd cygpath; then
        cygpath -m "$1"
    else
        printf '%s\n' "$1"
    fi
}

pigemma_python() { # prints a Python >= 3.10 interpreter; 1 if none
    local c
    if [ -n "${PIGEMMA_PYTHON:-}" ]; then
        if "$PIGEMMA_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            printf '%s\n' "$PIGEMMA_PYTHON"
            return 0
        fi
        return 1
    fi
    for c in python3 python; do
        if have_cmd "$c" && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            printf '%s\n' "$c"
            return 0
        fi
    done
    return 1
}

_pg_need_python() { # sets PY or dies 3
    if [ -n "${PY:-}" ] && [ "${_PG_PY_FOR:-}" = "${PIGEMMA_PYTHON:-}" ]; then
        return 0
    fi
    PY=$(pigemma_python) || die 3 "no usable Python >= 3.10 (PIGEMMA_PYTHON='${PIGEMMA_PYTHON:-}', python3, python): sudo apt-get install -y python3"
    _PG_PY_FOR=${PIGEMMA_PYTHON:-}
}

run_feasibility() { # run_feasibility ARGS... -> exit code of python3 -m pigemma.feasibility
    _pg_need_python
    local rc=0 pp
    pp=$(native_path "$PIGEMMA_HOME")
    if [ -n "${_PG_MSYS:-}" ]; then
        pp="$pp${PYTHONPATH:+;$PYTHONPATH}"
    else
        pp="$pp${PYTHONPATH:+:$PYTHONPATH}"
    fi
    (cd "$PIGEMMA_HOME" && PYTHONPATH="$pp" "$PY" -m pigemma.feasibility "$@") || rc=$?
    return "$rc"
}

_pg_verdict() { # _pg_verdict MODE(gate|env) [options] -- shared by require_verdict_ok / feasibility_env
    local mode=$1
    shift
    local want="" sim="" allow_slow=0 facts="" json="" quiet=0 collect=1 extra=() rc=0 tmp=""
    while [ $# -gt 0 ]; do
        case $1 in
            --want) want=${2:-}; shift 2 || die 1 "--want needs a value" ;;
            --want=*) want=${1#*=}; shift ;;
            --sim) sim=${2:-}; shift 2 || die 1 "--sim needs a value" ;;
            --sim=*) sim=${1#*=}; shift ;;
            --facts) facts=${2:-}; collect=0; shift 2 || die 1 "--facts needs a value" ;;
            --facts=*) facts=${1#*=}; collect=0; shift ;;
            --json) json=${2:-}; shift 2 || die 1 "--json needs a value" ;;
            --json=*) json=${1#*=}; shift ;;
            --allow-slow | --allow-2) allow_slow=1; shift ;;
            --quiet) quiet=1; shift ;;
            --no-collect) collect=0; shift ;;
            --) shift; extra=("$@"); break ;;
            *) die 1 "require_verdict_ok/feasibility_env: unknown option '$1'" ;;
        esac
    done
    facts=${facts:-$PIGEMMA_OUT/facts.json}
    json=${json:-$PIGEMMA_OUT/feasibility.json}
    if [ "$collect" = 1 ]; then
        write_facts "$facts"
    fi
    if [ ! -f "$facts" ]; then
        die 3 "facts file not found: $facts (run collect_facts / write_facts first)"
    fi
    local args=(--facts "$(native_path "$facts")" --json "$(native_path "$json")")
    if [ -n "$want" ]; then args+=(--want "$want"); fi
    if [ -n "$sim" ]; then args+=(--sim "$(native_path "$sim")"); fi
    if [ "$quiet" = 1 ]; then args+=(--quiet); fi
    if [ "$mode" = env ]; then
        mkdir -p "$PIGEMMA_OUT" || die 1 "cannot create $PIGEMMA_OUT"
        tmp="$PIGEMMA_OUT/.feasibility_env.$$"
        run_feasibility "${args[@]}" --emit env "${extra[@]}" >"$tmp" || rc=$?
        if [ -s "$tmp" ]; then
            # shellcheck disable=SC1090
            . "$tmp" # KEY='value' lines, shlex-quoted by pigemma.feasibility.emit_env
        fi
        rm -f "$tmp"
    else
        run_feasibility "${args[@]}" "${extra[@]}" || rc=$?
    fi
    PIGEMMA_VERDICT_RC=$rc
    PIGEMMA_FEASIBILITY_JSON=$json
    export PIGEMMA_VERDICT_RC PIGEMMA_FEASIBILITY_JSON
    case $rc in
        0)
            log "feasibility verdict OK (exit 0) -> $json"
            return 0
            ;;
        2)
            if [ "$allow_slow" = 1 ]; then
                warn "feasibility verdict: only slow streaming is possible (exit 2) - continuing because this step allows it"
                return 0
            fi
            die 2 "feasibility verdict: only slow streaming is possible (exit 2); see the table above and $json"
            ;;
        3) die 3 "feasibility verdict: nothing viable on this host (exit 3); see the table above and $json" ;;
        4) die 4 "feasibility verdict: refused (protected or busy host, exit 4); see $json" ;;
        *) die "$rc" "pigemma.feasibility failed (exit $rc): check the facts file $facts and the catalog" ;;
    esac
}

require_verdict_ok() { _pg_verdict gate "$@"; }
feasibility_env() { _pg_verdict env "$@"; }

json_get() { # json_get FILE KEY.PATH -> value (strings raw, null -> empty, objects as JSON); 1 if missing
    local f=${1:?json_get: file required} key=${2:-}
    _pg_need_python
    "$PY" -c 'import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8-sig"))
for k in (sys.argv[2].split(".") if sys.argv[2] else []):
    if isinstance(d, list):
        try:
            d = d[int(k)]
        except (ValueError, IndexError):
            sys.exit(1)
    elif isinstance(d, dict) and k in d:
        d = d[k]
    else:
        sys.exit(1)
if d is None:
    print("")
elif isinstance(d, bool):
    print("true" if d else "false")
elif isinstance(d, (dict, list)):
    print(json.dumps(d, ensure_ascii=False))
else:
    print(d)' "$(native_path "$f")" "$key"
}

# ===================================================================================== misc helpers
llama_bin() { printf '%s\n' "${PIGEMMA_LLAMA_BIN:-$LLAMA_DIR/build/bin}/$1"; }
tools_bin() { printf '%s\n' "${PIGEMMA_TOOLS_BIN:-$PIGEMMA_HOME/tools/build/bin}/$1"; }

run_nice() { nice -n "${NICE_LEVEL:-10}" "$@"; }

run_nice_io() { # nice + ionice -c3 (idle I/O class) when the kernel permits it
    if [ -z "${_PG_IONICE_OK:-}" ]; then
        if have_cmd ionice && ionice -c3 true >/dev/null 2>&1; then
            _PG_IONICE_OK=1
        else
            _PG_IONICE_OK=0
            warn "ionice -c3 not available here: running with nice only"
        fi
    fi
    if [ "$_PG_IONICE_OK" = 1 ]; then
        nice -n "${NICE_LEVEL:-10}" ionice -c3 "$@"
    else
        nice -n "${NICE_LEVEL:-10}" "$@"
    fi
}

_PG_CLEANUP=()
_pg_run_cleanup() {
    local rc=$? i
    for ((i = ${#_PG_CLEANUP[@]} - 1; i >= 0; i--)); do
        eval "${_PG_CLEANUP[$i]}" || true
    done
    return "$rc"
}

add_cleanup() { # add_cleanup "SHELL CODE" (run at EXIT, newest first)
    _PG_CLEANUP+=("$*")
    if [ -z "${_PG_TRAP_SET:-}" ]; then
        trap _pg_run_cleanup EXIT
        _PG_TRAP_SET=1
    fi
}

# ================================================================================ configuration
PIGEMMA_CONFIG_VARS=(PIGEMMA_HOME MODEL_DIR LLAMA_DIR LLAMA_COMMIT PIGEMMA_OUT THREADS CTX QUANT_OVERRIDE
    OS_RESERVE_GIB MIN_FREE_DISK_FACTOR KEEP_FREE_GB NICE_LEVEL DENY_HOSTS PROTECTED_SERVICE_GLOBS
    PIGEMMA_I_KNOW ALLOW_MLOCK TUNE_APPLY TUNE_NVME_GEN3 TUNE_GOVERNOR TUNE_SWAPPINESS TUNE_READAHEAD_KB
    TUNE_MEMLOCK BUILD_JOBS HF_LIMIT_RATE PIN_BUDGET_BYTES)
PIGEMMA_ENV_KNOBS=(PIGEMMA_EMULATED PIGEMMA_FAKE_PROC PIGEMMA_PYTHON
    PIGEMMA_BOOT_CONFIG PIGEMMA_MAX_LOADAVG PIGEMMA_LLAMA_BIN PIGEMMA_TOOLS_BIN PIGEMMA_DEVICE PIGEMMA_LOG_FILE
    PIGEMMA_IN_CONTAINER)

print_config() {
    local v src
    if [ "${_PG_CONFIG_LOADED:-0}" = 1 ]; then
        src=$PIGEMMA_CONFIG
    else
        src="built-in defaults ($PIGEMMA_CONFIG not found)"
    fi
    _pg_emit "" "effective configuration ($src; environment overrides win):"
    for v in "${PIGEMMA_CONFIG_VARS[@]}"; do
        printf '    %-24s = %s\n' "$v" "'${!v-}'" >&2
    done
    printf '    %-24s = %s\n' PIGEMMA_CATALOG "'${PIGEMMA_CATALOG:-}'" >&2
    for v in "${PIGEMMA_ENV_KNOBS[@]}"; do
        if [ -n "${!v:-}" ]; then
            printf '    %-24s = %s   (environment)\n' "$v" "'${!v}'" >&2
        fi
    done
    return 0
}

_pg_validate_config() {
    local bad=() v
    _pg_is_uint "$THREADS" && [ "$THREADS" -ge 1 ] || bad+=("THREADS='$THREADS' (positive integer)")
    _pg_is_uint "$CTX" && [ "$CTX" -ge 1 ] || bad+=("CTX='$CTX' (positive integer)")
    _pg_is_num "$MIN_FREE_DISK_FACTOR" && awk -v f="$MIN_FREE_DISK_FACTOR" 'BEGIN { exit !(f >= 1) }' ||
        bad+=("MIN_FREE_DISK_FACTOR='$MIN_FREE_DISK_FACTOR' (number >= 1)")
    _pg_is_num "$KEEP_FREE_GB" || bad+=("KEEP_FREE_GB='$KEEP_FREE_GB' (number of GB >= 0)")
    _pg_is_uint "$NICE_LEVEL" && [ "$NICE_LEVEL" -le 19 ] || bad+=("NICE_LEVEL='$NICE_LEVEL' (0..19)")
    if [ -n "$OS_RESERVE_GIB" ] && ! _pg_is_num "$OS_RESERVE_GIB"; then
        bad+=("OS_RESERVE_GIB='$OS_RESERVE_GIB' (empty or GiB number)")
    fi
    for v in TUNE_APPLY TUNE_NVME_GEN3 TUNE_MEMLOCK PIGEMMA_I_KNOW; do
        case ${!v} in
            0 | 1 | [Yy][Ee][Ss] | [Nn][Oo] | [Tt][Rr][Uu][Ee] | [Ff][Aa][Ll][Ss][Ee]) ;;
            *) bad+=("$v='${!v}' (0 or 1)") ;;
        esac
    done
    case $ALLOW_MLOCK in auto | 0 | 1) ;; *) bad+=("ALLOW_MLOCK='$ALLOW_MLOCK' (auto, 0 or 1)") ;; esac
    [[ $TUNE_GOVERNOR =~ ^[a-z_]+$ ]] || bad+=("TUNE_GOVERNOR='$TUNE_GOVERNOR' (governor name)")
    _pg_is_uint "$TUNE_SWAPPINESS" && [ "$TUNE_SWAPPINESS" -le 200 ] || bad+=("TUNE_SWAPPINESS='$TUNE_SWAPPINESS' (0..200)")
    _pg_is_uint "$TUNE_READAHEAD_KB" || bad+=("TUNE_READAHEAD_KB='$TUNE_READAHEAD_KB' (kB, integer)")
    if [ "$BUILD_JOBS" != auto ] && ! { _pg_is_uint "$BUILD_JOBS" && [ "$BUILD_JOBS" -ge 1 ]; }; then
        bad+=("BUILD_JOBS='$BUILD_JOBS' (auto or a positive integer)")
    fi
    if [ "$PIN_BUDGET_BYTES" != auto ] && ! [[ $PIN_BUDGET_BYTES =~ ^[0-9]+([.][0-9]+)?([eE][0-9]+)?([KMGT]i?B)?$ ]]; then
        bad+=("PIN_BUDGET_BYTES='$PIN_BUDGET_BYTES' (auto, bytes, 5.2e9, 5.2GB or 2GiB)")
    fi
    if [ -n "$HF_LIMIT_RATE" ] && ! [[ $HF_LIMIT_RATE =~ ^[0-9]+[KkMmGg]?$ ]]; then
        bad+=("HF_LIMIT_RATE='$HF_LIMIT_RATE' (curl --limit-rate, e.g. 5M)")
    fi
    [[ $LLAMA_COMMIT =~ ^[0-9a-f]{7,40}$ ]] || bad+=("LLAMA_COMMIT='$LLAMA_COMMIT' (7-40 hex digits)")
    if [ -n "$QUANT_OVERRIDE" ] && ! [[ $QUANT_OVERRIDE =~ ^[A-Za-z0-9_.-]+$ ]]; then
        bad+=("QUANT_OVERRIDE='$QUANT_OVERRIDE' (quant name such as UD-Q3_K_M)")
    fi
    for v in PIGEMMA_HOME MODEL_DIR LLAMA_DIR PIGEMMA_OUT; do
        if [ -z "${!v}" ]; then bad+=("$v is empty"); fi
    done
    if [ "${#bad[@]}" -gt 0 ]; then
        die 1 "invalid configuration ($PIGEMMA_CONFIG / environment): ${bad[*]}"
    fi
    return 0
}

pigemma_init() { # validate config, create PIGEMMA_OUT, print the effective values
    _pg_validate_config
    mkdir -p "$PIGEMMA_OUT" || die 1 "cannot create PIGEMMA_OUT=$PIGEMMA_OUT"
    if [ "${PIGEMMA_QUIET:-0}" != 1 ]; then
        print_config
    fi
    if is_emulated; then
        log "EMULATED host (PIGEMMA_EMULATED=1): results are labelled emulated; speeds are NOT Raspberry Pi measurements"
    fi
    if is_fake; then
        log "FAKE system root (tests only): PIGEMMA_FAKE_PROC=$PIGEMMA_FAKE_PROC"
    fi
    return 0
}

# ================================================================================== initialisation
_pg_this=${BASH_SOURCE[0]:-$0}
case $_pg_this in
    */*) PIGEMMA_LIB_DIR=$(cd "${_pg_this%/*}" && pwd) ;;
    *) PIGEMMA_LIB_DIR=$(pwd) ;;
esac
PIGEMMA_PI_DIR=$(cd "$PIGEMMA_LIB_DIR/.." && pwd)
if [ -z "${PIGEMMA_HOME:-}" ]; then
    PIGEMMA_HOME=$(cd "$PIGEMMA_PI_DIR/.." && pwd)
fi
if [ -z "${PIGEMMA_SCRIPT:-}" ]; then
    PIGEMMA_SCRIPT=${0##*/}
fi
case ${OSTYPE:-} in
    msys* | cygwin*) _PG_MSYS=1 ;;
    *) _PG_MSYS="" ;;
esac

if [ -n "${PIGEMMA_FAKE_PROC:-}" ]; then
    case $PIGEMMA_FAKE_PROC in
        /* | [A-Za-z]:/* | [A-Za-z]:\\*) ;;
        *) PIGEMMA_FAKE_PROC="$PWD/$PIGEMMA_FAKE_PROC" ;;
    esac
    if [ ! -d "$PIGEMMA_FAKE_PROC" ]; then
        die 1 "PIGEMMA_FAKE_PROC=$PIGEMMA_FAKE_PROC is not a directory"
    fi
fi

PIGEMMA_CONFIG=${PIGEMMA_CONFIG:-$PIGEMMA_PI_DIR/pi_config.env}
_PG_CONFIG_LOADED=0
if [ -f "$PIGEMMA_CONFIG" ]; then
    # shellcheck source=../pi_config.env
    . "$PIGEMMA_CONFIG"
    _PG_CONFIG_LOADED=1
fi

# Built-in defaults for anything the config file left unset (missing or older pi_config.env).
MODEL_DIR=${MODEL_DIR:-$PIGEMMA_HOME/models}
LLAMA_DIR=${LLAMA_DIR:-${HOME:-/root}/llama.cpp}
LLAMA_COMMIT=${LLAMA_COMMIT:-a7b94df}
PIGEMMA_OUT=${PIGEMMA_OUT:-$PIGEMMA_HOME/pi/out}
THREADS=${THREADS:-4}
CTX=${CTX:-4096}
QUANT_OVERRIDE=${QUANT_OVERRIDE:-}
OS_RESERVE_GIB=${OS_RESERVE_GIB:-}
MIN_FREE_DISK_FACTOR=${MIN_FREE_DISK_FACTOR:-1.1}
KEEP_FREE_GB=${KEEP_FREE_GB:-2}
NICE_LEVEL=${NICE_LEVEL:-10}
DENY_HOSTS=${DENY_HOSTS-}
PROTECTED_SERVICE_GLOBS=${PROTECTED_SERVICE_GLOBS-contextfriend* context-friend*}
PIGEMMA_I_KNOW=${PIGEMMA_I_KNOW:-0}
ALLOW_MLOCK=${ALLOW_MLOCK:-auto}
TUNE_APPLY=${TUNE_APPLY:-0}
TUNE_NVME_GEN3=${TUNE_NVME_GEN3:-0}
TUNE_GOVERNOR=${TUNE_GOVERNOR:-performance}
TUNE_SWAPPINESS=${TUNE_SWAPPINESS:-10}
TUNE_READAHEAD_KB=${TUNE_READAHEAD_KB:-2048}
TUNE_MEMLOCK=${TUNE_MEMLOCK:-1}
BUILD_JOBS=${BUILD_JOBS:-auto}
HF_LIMIT_RATE=${HF_LIMIT_RATE:-}
PIN_BUDGET_BYTES=${PIN_BUDGET_BYTES:-auto}
PIGEMMA_CATALOG=${PIGEMMA_CATALOG:-$PIGEMMA_HOME/pigemma/data/catalog.json}

# Children (python3 -m pigemma.feasibility reads THREADS, CTX, OS_RESERVE_GIB, DENY_HOSTS,
# PROTECTED_SERVICE_GLOBS, PIGEMMA_MAX_LOADAVG, PIGEMMA_EMULATED from the environment) see the config.
export "${PIGEMMA_CONFIG_VARS[@]}" PIGEMMA_CONFIG PIGEMMA_CATALOG

# ============================================================================ direct execution (CLI)
_pg_cli_usage() {
    cat <<EOF
pi/lib/common.sh $PIGEMMA_COMMON_VERSION - shared helpers of the pi-gemma4 Pi scripts (normally sourced).

Usage: bash pi/lib/common.sh <command> [args]
  facts                 print facts.json (read-only fact collection) to stdout
  write-facts [OUT]     write facts.json (default \$PIGEMMA_OUT/facts.json)
  safe-host             exit 4 on a protected host (DENY_HOSTS, catalog, protected services) unless PIGEMMA_I_KNOW=1
  not-busy              exit 4 when loadavg1 or MemAvailable say the host is busy
  verdict [opts]        require_verdict_ok: --want X --allow-slow --sim F --facts F --no-collect --json OUT --quiet [-- args]
  jobs [--auto]         parallel build jobs: min(nproc, floor(MemAvailable_GiB/1.2), 4) unless BUILD_JOBS is numeric
  config                print the effective configuration
  storage [DIR]         storage detection for DIR (default MODEL_DIR) as KEY=value lines
  fmt-gb BYTES          format bytes as GB (1e9)
  mem-available-kb      effective MemAvailable in kB (cgroup aware)
  free-disk [PATH]      free bytes on the filesystem holding PATH
  version | help

Exit codes: 0 ok, 1 usage/internal error, 2 only slow streaming possible, 3 not viable / precondition failed,
            4 refused (protected or busy host).
Test hook: PIGEMMA_FAKE_PROC=<dir> reads every /proc, /sys, /boot file and command output from fixture files.
EOF
}

if [ "${BASH_SOURCE[0]:-}" = "$0" ]; then
    set -euo pipefail
    PIGEMMA_SCRIPT=common.sh
    _pg_cmd=${1:-help}
    if [ $# -gt 0 ]; then shift; fi
    case $_pg_cmd in
        help | -h | --help) _pg_cli_usage ;;
        version | --version) printf 'pi/lib/common.sh %s\n' "$PIGEMMA_COMMON_VERSION" ;;
        facts | collect-facts | collect_facts) collect_facts ;;
        write-facts | write_facts) write_facts "$@" ;;
        safe-host | assert-safe-host | assert_safe_host) assert_safe_host ;;
        not-busy | assert-not-busy | assert_not_busy) assert_not_busy ;;
        verdict | require-verdict-ok | require_verdict_ok) require_verdict_ok "$@" ;;
        jobs | pick-jobs | pick_jobs) pick_jobs "$@" ;;
        config | print-config | print_config) print_config ;;
        storage | detect-storage | detect_storage)
            detect_storage "${1:-$MODEL_DIR}"
            for _pg_v in STORAGE_DEV STORAGE_DISK STORAGE_TYPE STORAGE_ROTATIONAL STORAGE_LINK_SPEED \
                STORAGE_LINK_WIDTH STORAGE_READ_AHEAD_KB STORAGE_TRANSPORT STORAGE_FSTYPE STORAGE_MOUNT; do
                printf '%s=%q\n' "$_pg_v" "${!_pg_v}"
            done
            for _pg_n in "${STORAGE_NOTES[@]}"; do
                printf '# note: %s\n' "$_pg_n"
            done
            ;;
        fmt-gb | fmt_gb) fmt_gb "$@" ;;
        mem-available-kb | mem_available_kb) mem_available_kb ;;
        free-disk | free_disk_bytes) free_disk_bytes "${1:-$MODEL_DIR}" ;;
        *)
            _pg_cli_usage >&2
            die 1 "unknown command '$_pg_cmd'"
            ;;
    esac
    exit 0
fi
