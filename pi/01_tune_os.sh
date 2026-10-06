#!/usr/bin/env bash
# pi/01_tune_os.sh - print-first OS tuning of a Raspberry Pi 5 for memory/storage-bound LLM inference.
#
# Usage: pi/01_tune_os.sh [--apply] [--no-pcie]
#        pi/01_tune_os.sh [--apply] [--revert] [--dry-run] [--only LIST] [--skip LIST]
#                         [--persist | --runtime-only] [--allow-boot-edit] [-h|--help] [--version]
#
# The default is a DRY RUN: for every item it prints the current value, the proposed value and the exact
# commands --apply would run, and changes nothing. --apply (or TUNE_APPLY=1 in pi/pi_config.env) runs
# them idempotently, as root or as a user with passwordless sudo (every privileged command is
# `sudo -n ...`), and prints the before and after values. assert_safe_host runs first (exit 4).
#
# ITEMS (the TUNE_* variables of pi/pi_config.env select them; --only / --skip a,b,... as well)
#   (a) PCIe Gen3 x1: append `dtparam=pciex1_gen=3` under [all] to /boot/firmware/config.txt. Only when
#       TUNE_NVME_GEN3=1, an NVMe is present, and nothing shows a Hailo / AI HAT on the lane: not
#       /proc/device-tree/hat/product, not lspci, not PCI vendor 0x1e60, not /dev/hailo*. Otherwise it
#       prints "PCIe lane used by the AI HAT: NVMe impossible without a dual-slot HAT" and skips.
#       The block is appended once (grep guard on its marker), after a backup copy. The script prints
#       REBOOT REQUIRED and never reboots. --no-pcie skips the item.
#   (b) cpufreq governor TUNE_GOVERNOR (performance) on every /sys/devices/system/cpu/cpufreq/policy*,
#       persisted by the oneshot unit pigemma-governor.service. TUNE_GOVERNOR=keep (or none) skips it.
#   (c) vm.swappiness=TUNE_SWAPPINESS (10) and vm.vfs_cache_pressure=TUNE_VFS_CACHE_PRESSURE (50, set
#       through the environment only): live values in /proc/sys/vm, persisted by
#       /etc/sysctl.d/90-pigemma.conf.
#   (d) read_ahead_kb=TUNE_READAHEAD_KB (2048) on the whole block device that holds MODEL_DIR, because
#       expert slabs are 1.1-2.2 MB contiguous. Persisted by /etc/udev/rules.d/90-pigemma-readahead.rules.
#   (e) memlock unlimited (TUNE_MEMLOCK=1): /etc/security/limits.d/90-pigemma.conf (soft + hard) and
#       /etc/systemd/system.conf.d/pigemma.conf DefaultLimitMEMLOCK=infinity. Needs a re-login or reboot.
#   (f) advice only: swap / zram, cooling (Active Cooler; the Pi 5 throttles from 80 C), power, storage.
#
# DESIGN DECISIONS (deviations from the manifest spec are marked DEVIATION)
#   * DEVIATION (c): `sysctl --system` is NOT run. It would re-apply every sysctl.d file and could undo
#     live settings of the production services. Instead the two values are written to
#     /proc/sys/vm/<key>, which is exactly what `sysctl -w` does. The persistence file is the same.
#   * (d) runs only `udevadm control --reload-rules`, never `udevadm trigger`, because the live value is
#     written to sysfs directly. (b) runs `systemctl daemon-reload` + `enable`, not `--now`, because the
#     live governor is written directly. (e) never runs `systemctl daemon-reexec`; it prints the note.
#   * PROTECTED HOST with PIGEMMA_I_KNOW=1 (context-friend-1, docs/design/FACTS.md safety rules):
#       - runtime-only by default: no persistence files (--persist overrides);
#       - (a) also needs --allow-boot-edit, the owner's explicit approval for a config.txt edit;
#       - (e) is skipped unless mlock_allowed (ALLOW_MLOCK=auto forbids mlock of model weights there, D4).
#   * --revert undoes an earlier --apply. It removes the marked config.txt block and the files that
#     carry the "Managed by pi-gemma4" marker, and restores the live values recorded in
#     $PIGEMMA_OUT/tune_os.state, which the first --apply wrote. Like the tune plan, it only prints
#     unless --apply is given.
#   * The config.txt backup (<file>.pigemma-<UTC timestamp>.bak, next to it) is a plain `cp`, not `cp -p`:
#     /boot/firmware is vfat, where preserving owner / mode can fail and would abort the item. --revert
#     deletes exactly the marked block; if the file had no final newline before --apply, the newline added
#     in front of the block stays (harmless for the firmware's parser).
#   * Files are only (over)written when absent or carrying our marker. A foreign file with the same name
#     is left alone with a note. Nothing is ever deleted that does not carry the marker.
#   * config.txt sections: a line counts as effective only if it sits before the first filter, under
#     [all] or under [pi5]. Lines under any other filter ([cm4], [cm5], [gpio..], [tryboot] ...) are
#     ignored. Our block always starts with [all]. `include` files are not followed.
#   * --apply is refused (exit 3) inside a container (it would only change the container, and
#     /proc/sys and /sys are read-only there), on non-Linux hosts, and with PIGEMMA_FAKE_PROC. Each of
#     these is allowed when PIGEMMA_TUNE_ROOT points at a scratch directory (test mode).
#   * vm.swappiness is applied as specified (10). Honest note printed with it: low swappiness keeps
#     ANONYMOUS memory (repacked weights, KV cache) out of swap. It does not make mmap'ed (-nr) weights,
#     which are file pages, more resident. ASSUMED kernel behaviour, not measured here.
#
# TEST HOOKS (environment)
#   PIGEMMA_FAKE_PROC=DIR   fake /proc, /sys, /boot and command outputs (all reads; see pi/lib/common.sh).
#   PIGEMMA_BOOT_CONFIG=F   the boot config file, read AND written directly (common.sh's override).
#   PIGEMMA_TUNE_ROOT=DIR   test sysroot. Every file write lands in DIR/<system path>, and live values are
#                           read back from there first. systemctl / udevadm are printed but never run.
#   TUNE_VFS_CACHE_PRESSURE vm.vfs_cache_pressure target (default 50; not in pi_config.env).
#
# EXIT CODES (pi script convention): 0 ok (plan printed / applied / nothing to do), 1 usage error or
# a failed --apply step, 3 --apply precondition failed (not root and no passwordless sudo, container,
# non-Linux, fake root without PIGEMMA_TUNE_ROOT), 4 refused (protected host, from assert_safe_host).
#
# SAFETY: never reboots, never kills / stops / restarts any service, never installs packages. --apply
# writes only the files named above (plus one config.txt backup copy) and the live values above.

set -euo pipefail
PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
. "$PI_DIR/lib/common.sh"

TUNE_VERSION="1.0.0"
TUNE_VFS_CACHE_PRESSURE=${TUNE_VFS_CACHE_PRESSURE:-50}
T_BEGIN="# >>> pi-gemma4 tune_os"
T_END="# <<< pi-gemma4 tune_os"
T_MARK="Managed by pi-gemma4 pi/01_tune_os.sh"
SED_EXPR='/^# >>> pi-gemma4 tune_os/,/^# <<< pi-gemma4 tune_os/d'
F_SYSCTL=/etc/sysctl.d/90-pigemma.conf
F_UNIT=/etc/systemd/system/pigemma-governor.service
F_UDEV=/etc/udev/rules.d/90-pigemma-readahead.rules
F_LIMITS=/etc/security/limits.d/90-pigemma.conf
F_SDCONF=/etc/systemd/system.conf.d/pigemma.conf
UNIT_NAME=pigemma-governor.service
ITEMS=(a b c d e f)
HAILO_RE='hailo|ai[[:space:]_-]*hat|ai[[:space:]_-]*kit'
LSPCI_RE='hailo|\[1e60:'
AI_HAT_MSG="PCIe lane used by the AI HAT: NVMe impossible without a dual-slot HAT"
TS=$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || echo now)
TODAY=$(date -u +%Y-%m-%d 2>/dev/null || echo today)

# ======================================================================================= usage / args
_t_usage() {
    cat <<EOF
pi/01_tune_os.sh $TUNE_VERSION - print-first OS tuning for the Raspberry Pi 5 (pi-gemma4)

Usage: pi/01_tune_os.sh [--apply] [--no-pcie]
       pi/01_tune_os.sh [--apply] [--revert] [--dry-run] [--only LIST] [--skip LIST]
                        [--persist | --runtime-only] [--allow-boot-edit]

Without --apply nothing is changed: every item shows current vs proposed values and the exact commands.
  --apply            execute the plan (same as TUNE_APPLY=1); needs root or passwordless sudo (sudo -n)
  --dry-run          print the plan only, even when TUNE_APPLY=1
  --revert           plan the undo of an earlier --apply (with --apply: execute it)
  --no-pcie          skip item (a): boot config / PCIe generation
  --only LIST        only these items, e.g. --only b,c (items a-f, see below)
  --skip LIST        skip these items
  --persist          write the persistence files even on a protected host (PIGEMMA_I_KNOW=1)
  --runtime-only     change live values only, write no persistence files (a reboot restores defaults)
  --allow-boot-edit  the owner approved the config.txt edit on a protected host (PIGEMMA_I_KNOW=1)
  -h, --help         this help
  --version          print the version

Items:
  (a) PCIe Gen3 x1 for an NVMe: dtparam=pciex1_gen=3 under [all] in /boot/firmware/config.txt
      (TUNE_NVME_GEN3=1 only; skipped when an AI HAT occupies the lane; REBOOT REQUIRED, never reboots)
  (b) cpufreq governor TUNE_GOVERNOR=${TUNE_GOVERNOR} (keep = skip) + oneshot unit $UNIT_NAME
  (c) vm.swappiness=${TUNE_SWAPPINESS}, vm.vfs_cache_pressure=${TUNE_VFS_CACHE_PRESSURE} + $F_SYSCTL
  (d) read_ahead_kb=${TUNE_READAHEAD_KB} on the disk holding MODEL_DIR + udev rule $F_UDEV
  (e) memlock unlimited (TUNE_MEMLOCK=${TUNE_MEMLOCK}): $F_LIMITS + $F_SDCONF (re-login needed)
  (f) advice only: swap / zram, Active Cooler (throttling from 80 C), power supply, storage

Examples:
  bash pi/01_tune_os.sh                                # dry run: the plan, nothing changed
  bash pi/01_tune_os.sh --apply                        # apply, one 'sudo -n' per privileged command
  TUNE_NVME_GEN3=1 bash pi/01_tune_os.sh --apply       # also the PCIe Gen3 line (NVMe, no AI HAT)
  bash pi/01_tune_os.sh --revert --apply               # undo an earlier --apply
  PIGEMMA_I_KNOW=1 bash pi/01_tune_os.sh --apply --only b,d   # protected host: live values only

Exit codes: 0 ok, 1 usage error or a failed step, 3 --apply precondition failed (not root and no
passwordless sudo, container, non-Linux, fake root without PIGEMMA_TUNE_ROOT), 4 refused (protected host).
Test hooks: PIGEMMA_FAKE_PROC (fake /proc, /sys), PIGEMMA_BOOT_CONFIG (boot config file),
PIGEMMA_TUNE_ROOT=DIR (writes go to DIR/<system path>; systemctl / udevadm are not executed).
EOF
}

OPT_APPLY=0
OPT_DRY=0
OPT_REVERT=0
OPT_NO_PCIE=0
OPT_ONLY=""
OPT_SKIP=""
OPT_PERSIST=""
OPT_ALLOW_BOOT=0
while [ $# -gt 0 ]; do
    case $1 in
        --apply) OPT_APPLY=1 ;;
        --dry-run | -n) OPT_DRY=1 ;;
        --revert) OPT_REVERT=1 ;;
        --no-pcie) OPT_NO_PCIE=1 ;;
        --only | --skip)
            if [ $# -lt 2 ]; then
                _t_usage >&2
                die 1 "$1 needs a list of items (a-f)"
            fi
            if [ "$1" = --only ]; then OPT_ONLY=$2; else OPT_SKIP=$2; fi
            shift
            ;;
        --only=*) OPT_ONLY=${1#*=} ;;
        --skip=*) OPT_SKIP=${1#*=} ;;
        --persist) OPT_PERSIST=1 ;;
        --runtime-only) OPT_PERSIST=0 ;;
        --allow-boot-edit) OPT_ALLOW_BOOT=1 ;;
        -h | --help)
            _t_usage
            exit 0
            ;;
        --version)
            printf 'pi/01_tune_os.sh %s\n' "$TUNE_VERSION"
            exit 0
            ;;
        *)
            _t_usage >&2
            die 1 "unknown argument '$1'"
            ;;
    esac
    shift
done

declare -A SEL=() SEL_WHY=()
for _x in "${ITEMS[@]}"; do
    SEL[$_x]=1
    SEL_WHY[$_x]=""
done

_t_items_list() { # _t_items_list LIST OPTION -> array _IL (letters a-f); die 1 on a bad item
    local s=${1//,/ } t raw=()
    _IL=()
    read -r -a raw <<<"$s" || true
    for t in "${raw[@]}"; do
        case $t in
            [a-f]) _IL+=("$t") ;;
            *) die 1 "$2: unknown item '$t' (items are a b c d e f)" ;;
        esac
    done
    if [ "${#_IL[@]}" -eq 0 ]; then
        die 1 "$2: empty item list"
    fi
    return 0
}

if [ -n "$OPT_ONLY" ]; then
    _t_items_list "$OPT_ONLY" --only
    for _x in "${ITEMS[@]}"; do
        SEL[$_x]=0
        SEL_WHY[$_x]="not selected (--only $OPT_ONLY)"
    done
    for _x in "${_IL[@]}"; do
        SEL[$_x]=1
        SEL_WHY[$_x]=""
    done
fi
if [ -n "$OPT_SKIP" ]; then
    _t_items_list "$OPT_SKIP" --skip
    for _x in "${_IL[@]}"; do
        SEL[$_x]=0
        SEL_WHY[$_x]="skipped (--skip $OPT_SKIP)"
    done
fi
if [ "$OPT_NO_PCIE" = 1 ]; then
    SEL[a]=0
    SEL_WHY[a]="skipped (--no-pcie)"
fi

# =============================================================================== init / host gate
_T_OUT_EXISTED=0
if [ -d "$PIGEMMA_OUT" ]; then _T_OUT_EXISTED=1; fi
pigemma_init
if [ "$EUID" = 0 ] && [ -n "${SUDO_UID:-}" ] && [ "$_T_OUT_EXISTED" = 0 ]; then
    chown "$SUDO_UID:${SUDO_GID:-$SUDO_UID}" "$PIGEMMA_OUT" 2>/dev/null || true
fi
assert_safe_host
OVERRIDE=${PIGEMMA_PROTECTED_OVERRIDE:-0}

APPLY=0
if [ "$OPT_DRY" = 1 ]; then
    APPLY=0
elif [ "$OPT_APPLY" = 1 ] || _pg_truthy "$TUNE_APPLY"; then
    APPLY=1
fi
MODE=tune
if [ "$OPT_REVERT" = 1 ]; then MODE=revert; fi

if [ -n "$OPT_PERSIST" ]; then
    PERSIST=$OPT_PERSIST
elif [ "$OVERRIDE" = 1 ]; then
    PERSIST=0
else
    PERSIST=1
fi

if ! [[ $TUNE_VFS_CACHE_PRESSURE =~ ^[0-9]+$ ]] || [ "${#TUNE_VFS_CACHE_PRESSURE}" -gt 6 ]; then
    die 1 "TUNE_VFS_CACHE_PRESSURE='$TUNE_VFS_CACHE_PRESSURE' must be a non-negative integer"
fi

TUNE_ROOT=${PIGEMMA_TUNE_ROOT:-}
if [ -n "$TUNE_ROOT" ]; then
    case $TUNE_ROOT in
        / | //) die 1 "PIGEMMA_TUNE_ROOT=/ is the real system: unset it instead" ;;
        /* | [A-Za-z]:/* | [A-Za-z]:\\*) ;;
        *) TUNE_ROOT="$PWD/$TUNE_ROOT" ;;
    esac
    TUNE_ROOT=${TUNE_ROOT%/}
fi

if [ "$APPLY" = 1 ]; then
    if [ -n "$TUNE_ROOT" ]; then
        mkdir -p "$TUNE_ROOT" || die 1 "cannot create PIGEMMA_TUNE_ROOT=$TUNE_ROOT"
    elif is_fake; then
        die 3 "--apply with PIGEMMA_FAKE_PROC (test fixture) would change THIS machine from fake facts; set PIGEMMA_TUNE_ROOT=<scratch dir> too, or drop --apply"
    else
        _t_os=$(uname -s 2>/dev/null || echo unknown)
        if [ "$_t_os" != Linux ]; then
            die 3 "--apply needs Linux (this is '$_t_os'): OS tuning is for the Raspberry Pi itself; the dry run works anywhere"
        fi
        if in_container; then
            die 3 "--apply inside a container would only change the container, not the host kernel (/proc/sys and /sys are read-only there): run it on the Pi; the dry run works here"
        fi
    fi
fi

STATE_FILE="$PIGEMMA_OUT/tune_os.state"

# ============================================================================== path / read helpers
_t_target() { # SYS_PATH -> _T: where a write to SYS_PATH lands (test sysroot aware)
    _T=$1
    if [ -n "$TUNE_ROOT" ] && [ "$1" != "${PIGEMMA_BOOT_CONFIG:-}" ]; then
        _T="$TUNE_ROOT$1"
    fi
    return 0
}

_t_view() { # SYS_PATH -> _V: where SYS_PATH is read from (sysroot copy, else the fake / real path)
    if [ -n "${PIGEMMA_BOOT_CONFIG:-}" ] && [ "$1" = "$PIGEMMA_BOOT_CONFIG" ]; then
        _V=$1
        return 0
    fi
    if [ -n "$TUNE_ROOT" ] && [ -e "$TUNE_ROOT$1" ]; then
        _V="$TUNE_ROOT$1"
        return 0
    fi
    _pg_p "$1"
    _V=$_P
    return 0
}

_t_rd() { # SYS_PATH -> _R: first line, NULs and CR removed, trimmed; 1 if unreadable or empty
    _R=""
    _t_view "$1"
    if [ ! -f "$_V" ] || [ ! -r "$_V" ]; then
        return 1
    fi
    local line=""
    line=$(tr -d '\000\r' <"$_V" 2>/dev/null) || line=""
    line=${line%%$'\n'*}
    _pg_trim "$line"
    [ -n "$_R" ]
}

_t_content() { # SYS_PATH -> _C: whole file without trailing newlines and CRs; 1 if absent
    _C=""
    _t_view "$1"
    if [ ! -f "$_V" ]; then
        return 1
    fi
    _C=$(cat "$_V" 2>/dev/null) || return 1
    _C=${_C//$'\r'/}
    return 0
}

_t_isdir() {
    _t_view "$1"
    [ -d "$_V" ]
}

_t_file_state() { # SYS_PATH WANT -> _FS = absent | same | ours (outdated) | foreign
    if ! _t_content "$1"; then
        _FS=absent
    elif [ "$_C" = "$2" ]; then
        _FS=same
    elif [[ $_C == *"$T_MARK"* ]]; then
        _FS=ours
    else
        _FS=foreign
    fi
    return 0
}

_t_fs_word() { # _FS value -> _W (human text)
    case $1 in
        absent) _W="absent" ;;
        same) _W="installed (up to date)" ;;
        ours) _W="installed by an older pi-gemma4 run (outdated)" ;;
        foreign) _W="PRESENT but not written by pi-gemma4" ;;
        *) _W="?" ;;
    esac
    return 0
}

_t_systemd_live() { # 0 when systemctl may be queried about THIS system (no fake root, no test sysroot)
    if is_fake || [ -n "$TUNE_ROOT" ]; then return 1; fi
    have_cmd systemctl && [ -d /run/systemd/system ]
}

_t_q() { printf '%q' "$1"; }
_t_bak() { printf '%s.pigemma-%s.bak' "$1" "$TS"; }

# ================================================================================== state (revert)
declare -A STATE=()

_t_state_load() {
    STATE=()
    local p v
    if [ ! -f "$STATE_FILE" ]; then
        return 0
    fi
    while IFS=$'\t' read -r p v || [ -n "${p:-}" ]; do
        case $p in
            "" | "#"*) ;;
            /*) STATE[$p]=${v%$'\r'} ;;
        esac
        p=""
    done <"$STATE_FILE"
    return 0
}

_t_state_save() {
    local tmp="$STATE_FILE.tmp.$$" keys=() k
    if [ "${#STATE[@]}" -eq 0 ]; then
        rm -f "$STATE_FILE"
        return 0
    fi
    mapfile -t keys < <(printf '%s\n' "${!STATE[@]}" | LC_ALL=C sort)
    {
        printf '# pi-gemma4 pi/01_tune_os.sh: live values before the first --apply (restored by --revert --apply)\n'
        printf '# host %s; format: <system path><TAB><original value>\n' "$(host_name)"
        for k in "${keys[@]}"; do
            printf '%s\t%s\n' "$k" "${STATE[$k]}"
        done
    } >"$tmp" || {
        rm -f "$tmp"
        return 1
    }
    mv -f "$tmp" "$STATE_FILE" || return 1
    if [ "$EUID" = 0 ] && [ -n "${SUDO_UID:-}" ]; then
        chown "$SUDO_UID:${SUDO_GID:-$SUDO_UID}" "$STATE_FILE" 2>/dev/null || true
    fi
    return 0
}

_t_state_record() { # SYS_PATH VALUE: keep the FIRST original value only
    if [ -z "${STATE[$1]+x}" ]; then
        STATE[$1]=$2
        _t_state_save || warn "cannot write $STATE_FILE: --revert will not know the original value of $1 ($2)"
    fi
    return 0
}

_t_state_forget_prefix() { # PREFIX
    local k changed=0
    for k in "${!STATE[@]}"; do
        case $k in
            "$1"*)
                unset "STATE[$k]"
                changed=1
                ;;
        esac
    done
    if [ "$changed" = 1 ]; then
        _t_state_save || warn "cannot update $STATE_FILE"
    fi
    return 0
}

# ============================================================================= item bookkeeping
declare -A I_TITLE=() I_CUR=() I_PROP=() I_STATUS=() I_WHY=() I_NOTES=() I_EXEC=() I_FLAG=()
for _x in "${ITEMS[@]}"; do
    I_TITLE[$_x]=""
    I_CUR[$_x]=""
    I_PROP[$_x]=""
    I_STATUS[$_x]=""
    I_WHY[$_x]=""
    I_NOTES[$_x]=""
    I_EXEC[$_x]=0
    I_FLAG[$_x]=""
done
A_ITEM=()
A_TYPE=()
A_PATH=()
A_ARG=()

_t_act() { # ITEM TYPE SYS_PATH [ARG]: types mkdir write append set restore backup sed_delete rm cmd
    A_ITEM+=("$1")
    A_TYPE+=("$2")
    A_PATH+=("$3")
    A_ARG+=("${4-}")
    return 0
}

_t_nact() { # ITEM -> _N = number of actions
    local i
    _N=0
    for ((i = 0; i < ${#A_ITEM[@]}; i++)); do
        if [ "${A_ITEM[$i]}" = "$1" ]; then _N=$((_N + 1)); fi
    done
    return 0
}

_t_note() { # ITEM TEXT
    I_NOTES[$1]+="${I_NOTES[$1]:+$'\n'}$2"
    return 0
}

_t_skip() {
    I_STATUS[$1]=SKIP
    I_WHY[$1]=$2
    I_EXEC[$1]=0
    return 0
}

_t_unselect() { # keeps the actions for display, never executes them
    I_STATUS[$1]="NOT SELECTED"
    I_WHY[$1]=$2
    I_EXEC[$1]=0
    return 0
}

_t_ok() {
    I_STATUS[$1]=OK
    I_WHY[$1]=$2
    I_EXEC[$1]=0
    return 0
}

_t_finalize() { # ITEM: CHANGE when there are actions, else OK
    if [ -z "${I_STATUS[$1]}" ]; then
        _t_nact "$1"
        if [ "$_N" -gt 0 ]; then
            I_STATUS[$1]=CHANGE
            I_EXEC[$1]=1
        else
            I_STATUS[$1]=OK
            I_EXEC[$1]=0
            if [ -z "${I_WHY[$1]}" ]; then
                if [ "$MODE" = revert ]; then I_WHY[$1]="nothing to revert"; else I_WHY[$1]="already as proposed"; fi
            fi
        fi
    fi
    return 0
}

_t_plan_file() { # ITEM SYS_PATH CONTENT -> mkdir/write actions when needed; _FS = file state
    local x=$1 p=$2 c=$3
    _t_file_state "$p" "$c"
    case $_FS in
        same) return 0 ;;
        foreign)
            _t_note "$x" "$p exists and was NOT written by pi-gemma4: left untouched (merge the setting by hand)"
            return 0
            ;;
    esac
    if ! _t_isdir "${p%/*}"; then
        _t_act "$x" mkdir "${p%/*}"
    fi
    _t_act "$x" write "$p" "$c"
    return 0
}

_t_plan_rm() { # ITEM SYS_PATH: remove our file (marker required)
    if _t_content "$2"; then
        if [[ $_C == *"$T_MARK"* ]]; then
            _t_act "$1" rm "$2"
            return 0
        fi
        _t_note "$1" "$2 exists but was NOT written by pi-gemma4: left untouched"
    fi
    return 0
}

_t_plan_restore() { # ITEM PREFIX: restore live values recorded in the state file
    local x=$1 pre=$2 k keys=() n=0
    if [ "${#STATE[@]}" -gt 0 ]; then
        mapfile -t keys < <(printf '%s\n' "${!STATE[@]}" | LC_ALL=C sort)
    fi
    for k in "${keys[@]}"; do
        case $k in "$pre"*) ;; *) continue ;; esac
        n=$((n + 1))
        if _t_rd "$k" && [ "$_R" = "${STATE[$k]}" ]; then
            continue
        fi
        _t_act "$x" restore "$k" "${STATE[$k]}"
    done
    if [ "$n" = 0 ]; then
        _t_note "$x" "no recorded original live values ($STATE_FILE): live values stay until the next reboot"
    fi
    return 0
}

# =================================================================================== file contents
SYSCTL_CONTENT="# $T_MARK: VM settings for memory-bound LLM inference.
# Undo: bash pi/01_tune_os.sh --revert --apply (or delete this file; kernel defaults return at boot)
vm.swappiness = $TUNE_SWAPPINESS
vm.vfs_cache_pressure = $TUNE_VFS_CACHE_PRESSURE"

LIMITS_CONTENT="# $T_MARK: unlimited locked memory (pinner hot set, llama.cpp -lm mlock).
# Takes effect at the next login (pam_limits); '*' does not cover root. Undo: bash pi/01_tune_os.sh --revert --apply
*    soft    memlock    unlimited
*    hard    memlock    unlimited"

SDCONF_CONTENT="# $T_MARK: unlimited locked memory for services started by systemd.
# Takes effect after a reboot or 'sudo systemctl daemon-reexec' (pi/01 never runs it). Undo: bash pi/01_tune_os.sh --revert --apply
[Manager]
DefaultLimitMEMLOCK=infinity"

# $$f is systemd's escape for a literal $f (single quotes: no bash expansion here)
UNIT_TEMPLATE='# @MARK@: cpufreq governor for LLM inference.
# Undo: bash pi/01_tune_os.sh --revert --apply (or: sudo systemctl disable pigemma-governor.service; delete this file)
[Unit]
Description=pi-gemma4: set the cpufreq governor to @GOV@
After=sysinit.target raspi-config.service
ConditionPathExists=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c '"'"'for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo @GOV@ > "$$f"; done'"'"'

[Install]
WantedBy=multi-user.target'
UNIT_CONTENT=${UNIT_TEMPLATE//@MARK@/$T_MARK}
UNIT_CONTENT=${UNIT_CONTENT//@GOV@/$TUNE_GOVERNOR}

_t_udev_content() { # DISK -> _C
    _C="# $T_MARK: read-ahead on the model device (expert slabs are 1.1-2.2 MB contiguous).
# Undo: bash pi/01_tune_os.sh --revert --apply
ACTION==\"add|change\", SUBSYSTEM==\"block\", ENV{DEVTYPE}==\"disk\", KERNEL==\"$1\", ATTR{queue/read_ahead_kb}=\"$TUNE_READAHEAD_KB\""
    return 0
}

# ================================================================================ (a) PCIe Gen3 x1
_t_boot_scan() { # FILE -> BOOT_GEN BOOT_GEN_LINE BOOT_GEN_SECTION BOOT_BLOCK_BEGIN BOOT_BLOCK_END BOOT_NL_END
    BOOT_GEN=""
    BOOT_GEN_LINE=""
    BOOT_GEN_SECTION=""
    BOOT_BLOCK_BEGIN=""
    BOOT_BLOCK_END=""
    BOOT_NL_END=1
    local f=$1 raw line n=0 sect="" applies=1 params last
    local re_sec='^[[:space:]]*\[([^]]*)\]' re_dt='^[[:space:]]*dtparam[[:space:]]*=(.*)$'
    local re_gen='(^|,)[[:space:]]*pciex1_gen[[:space:]]*=[[:space:]]*([0-9]+)'
    if [ ! -r "$f" ]; then
        return 0
    fi
    while IFS= read -r raw || [ -n "$raw" ]; do
        n=$((n + 1))
        raw=${raw%$'\r'}
        case $raw in
            "$T_BEGIN"*) if [ -z "$BOOT_BLOCK_BEGIN" ]; then BOOT_BLOCK_BEGIN=$n; fi ;;
            "$T_END"*) if [ -n "$BOOT_BLOCK_BEGIN" ] && [ -z "$BOOT_BLOCK_END" ]; then BOOT_BLOCK_END=$n; fi ;;
        esac
        line=${raw%%#*}
        if [[ $line =~ $re_sec ]]; then
            _pg_trim "${BASH_REMATCH[1],,}"
            sect=$_R
            case $sect in
                all | pi5) applies=1 ;;
                *) applies=0 ;;
            esac
            continue
        fi
        if [ "$applies" = 1 ] && [[ $line =~ $re_dt ]]; then
            params=${BASH_REMATCH[1]}
            if [[ $params =~ $re_gen ]]; then
                BOOT_GEN=${BASH_REMATCH[2]}
                BOOT_GEN_LINE=$n
                BOOT_GEN_SECTION=${sect:-top}
            fi
        fi
    done <"$f"
    if [ -s "$f" ]; then
        last=$(tail -c 1 "$f" 2>/dev/null || true)
        if [ -n "$last" ]; then BOOT_NL_END=0; fi
    fi
    return 0
}

_t_gen_of_speed() { # "8.0 GT/s PCIe" -> _G = Gen3
    case $1 in
        2.5*) _G=Gen1 ;;
        5.0* | 5\ *) _G=Gen2 ;;
        8.0* | 8\ *) _G=Gen3 ;;
        16*) _G=Gen4 ;;
        *) _G="" ;;
    esac
    return 0
}

_cur_a() { # scan PCIe / HAT / boot config -> A_* + BOOT_* globals; _R = description
    A_BOARD=""
    A_HAT=""
    A_HAILO=()
    A_NVME=()
    A_LINK=""
    A_WIDTH=""
    A_BOOT=""
    local f d v out line c desc
    if _pg_read1 /proc/device-tree/model; then A_BOARD=$_R; fi
    A_HAT=$(hat_product)
    if [ -n "$A_HAT" ] && [[ ${A_HAT,,} =~ $HAILO_RE ]]; then
        A_HAILO+=("HAT EEPROM product '$A_HAT' (/proc/device-tree/hat/product)")
    fi
    _pg_p /sys/bus/pci/devices
    for f in "$_P"/*/vendor; do
        if [ -f "$f" ] && [ -r "$f" ]; then
            v=$(tr -d '\000\r\n ' <"$f" 2>/dev/null || true)
            if [ "${v,,}" = 0x1e60 ]; then
                d=${f%/vendor}
                A_HAILO+=("PCI device ${d##*/} has vendor id 0x1e60 (Hailo)")
            fi
        fi
    done
    out=""
    if is_fake; then
        if _pg_fake_cmd lspci.txt; then out=$_R; fi
    elif have_cmd lspci; then
        out=$(_pg_timeout 10 lspci -nn 2>/dev/null || true)
    fi
    if [ -n "$out" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            if [[ ${line,,} =~ $LSPCI_RE ]]; then A_HAILO+=("lspci: $line"); fi
        done <<<"$out"
    fi
    _pg_p /dev
    for f in "$_P"/hailo*; do
        if [ -e "$f" ]; then A_HAILO+=("device node /dev/${f##*/}"); fi
    done
    _pg_p /sys/class/nvme
    for d in "$_P"/nvme*; do
        if [ -e "$d" ]; then
            A_NVME+=("${d##*/}")
            if [ -z "$A_LINK" ] && [ -r "$d/device/current_link_speed" ]; then
                A_LINK=$(tr -d '\000\r' <"$d/device/current_link_speed" 2>/dev/null || true)
                A_LINK=${A_LINK%%$'\n'*}
                if [ -r "$d/device/current_link_width" ]; then
                    A_WIDTH=$(tr -d '\000\r\n ' <"$d/device/current_link_width" 2>/dev/null || true)
                fi
            fi
        fi
    done
    if [ -n "${PIGEMMA_BOOT_CONFIG:-}" ]; then
        if [ -f "$PIGEMMA_BOOT_CONFIG" ]; then A_BOOT=$PIGEMMA_BOOT_CONFIG; fi
    else
        for c in /boot/firmware/config.txt /boot/config.txt; do
            _t_view "$c"
            if [ -f "$_V" ]; then
                A_BOOT=$c
                break
            fi
        done
    fi
    BOOT_GEN=""
    BOOT_GEN_LINE=""
    BOOT_GEN_SECTION=""
    BOOT_BLOCK_BEGIN=""
    BOOT_BLOCK_END=""
    BOOT_NL_END=1
    if [ -n "$A_BOOT" ]; then
        _t_view "$A_BOOT"
        _t_boot_scan "$_V"
    fi
    if [ -z "$A_BOOT" ]; then
        desc="boot config: none found"
    elif [ -n "$BOOT_GEN" ]; then
        desc="$A_BOOT: pciex1_gen=$BOOT_GEN (line $BOOT_GEN_LINE, section [$BOOT_GEN_SECTION])"
    else
        desc="$A_BOOT: pciex1_gen not set (Pi 5 default: Gen2)"
    fi
    if [ -n "$BOOT_BLOCK_BEGIN" ]; then desc+=", pi-gemma4 block at line $BOOT_BLOCK_BEGIN"; fi
    if [ "${#A_NVME[@]}" -gt 0 ]; then
        _t_gen_of_speed "$A_LINK"
        desc+="; NVMe: ${A_NVME[*]}, link ${A_LINK:-unknown}${A_WIDTH:+ x$A_WIDTH}${_G:+ ($_G)}"
    else
        desc+="; NVMe: none"
    fi
    desc+="; HAT: ${A_HAT:-none}"
    if [ "${#A_HAILO[@]}" -gt 0 ]; then desc+="; Hailo/AI HAT on PCIe: yes"; fi
    _R=$desc
    return 0
}

_t_link_note() {
    _t_gen_of_speed "$A_LINK"
    if [ -n "$A_LINK" ] && [ "$_G" != Gen3 ]; then
        _t_note a "the config asks for Gen3 but the link runs at '$A_LINK' ($_G): reboot pending, or the SSD / FPC cable fell back"
    fi
    return 0
}

plan_a() {
    local block="" h
    I_TITLE[a]="PCIe Gen3 x1 for the NVMe (dtparam=pciex1_gen=3 under [all] in the boot config)"
    _cur_a
    I_CUR[a]=$_R
    I_PROP[a]="dtparam=pciex1_gen=3 appended once under [all] (marked block, backup first); REBOOT REQUIRED, never rebooted by this script"
    if [ "${SEL[a]}" = 0 ]; then
        _t_skip a "${SEL_WHY[a]}"
        return 0
    fi
    if [ "$MODE" = revert ]; then
        I_PROP[a]="remove the pi-gemma4 block from ${A_BOOT:-the boot config}; REBOOT REQUIRED"
        if [ -n "$BOOT_BLOCK_BEGIN" ] && [ -n "$BOOT_BLOCK_END" ]; then
            I_PROP[a]="remove the pi-gemma4 block (lines $BOOT_BLOCK_BEGIN-$BOOT_BLOCK_END) from $A_BOOT; REBOOT REQUIRED"
            _t_act a backup "$A_BOOT"
            _t_act a sed_delete "$A_BOOT"
            I_FLAG[a]=reboot
            if [ "$OVERRIDE" = 1 ] && [ "$OPT_ALLOW_BOOT" != 1 ]; then
                _t_unselect a "protected host (PIGEMMA_I_KNOW=1): editing the boot config needs the owner's explicit approval -> add --allow-boot-edit"
            fi
        elif [ -n "$BOOT_BLOCK_BEGIN" ]; then
            _t_skip a "the pi-gemma4 begin marker (line $BOOT_BLOCK_BEGIN) has no end marker: edit $A_BOOT by hand"
        fi
        return 0
    fi
    if [ -z "$A_BOOT" ]; then
        _t_skip a "no boot config (/boot/firmware/config.txt) on this host (container / not Raspberry Pi OS): nothing to tune"
        return 0
    fi
    if [ -n "$A_BOARD" ] && ! [[ $A_BOARD == *"Raspberry Pi 5"* || $A_BOARD == *"Compute Module 5"* ]]; then
        _t_skip a "board '$A_BOARD' is not a Raspberry Pi 5 / CM5: the external PCIe x1 setting does not apply"
        return 0
    fi
    if [ "${#A_HAILO[@]}" -gt 0 ]; then
        if [ "${#A_NVME[@]}" -gt 0 ]; then
            _t_skip a "$AI_HAT_MSG (an NVMe is visible too: dual-slot HAT / PCIe switch - the link generation is not changed automatically)"
        else
            _t_skip a "$AI_HAT_MSG"
        fi
        for h in "${A_HAILO[@]}"; do _t_note a "AI HAT evidence: $h"; done
        _t_note a "the AI HAT does not help for 25-30B models (README 'Hardware options'); an NVMe needs the lane or a dual-slot HAT"
        return 0
    fi
    if [ "${#A_NVME[@]}" -eq 0 ]; then
        _t_skip a "no NVMe present (model storage: ${STORAGE_TYPE:-see (d)}): nothing to tune. Recommended upgrade: NVMe HAT + PCIe Gen3 x1 (README 'Hardware options')"
        return 0
    fi
    if [ -n "$BOOT_BLOCK_BEGIN" ]; then
        _t_ok a "already added by pi/01_tune_os.sh (grep guard: marker at line $BOOT_BLOCK_BEGIN)"
        _t_link_note
        return 0
    fi
    if [ "$BOOT_GEN" = 3 ]; then
        _t_ok a "already Gen3 (line $BOOT_GEN_LINE)"
        _t_link_note
        return 0
    fi
    if [ "$BOOT_NL_END" = 0 ]; then block=$'\n'; fi
    block+="$T_BEGIN (pi/01_tune_os.sh, $TODAY): PCIe Gen3 x1 for the NVMe - uncertified by Raspberry Pi.
# Undo: bash pi/01_tune_os.sh --revert --apply (or delete this block), then reboot.
[all]
dtparam=pciex1_gen=3
$T_END"
    _t_act a backup "$A_BOOT"
    _t_act a append "$A_BOOT" "$block"
    I_FLAG[a]=reboot
    _t_note a "Gen3 x1 reads 784-894 MB/s vs Gen2 417-454 MB/s (published, docs/research; not measured here); uncertified: if the NVMe drops out, revert"
    if ! _pg_truthy "$TUNE_NVME_GEN3"; then
        _t_unselect a "TUNE_NVME_GEN3=$TUNE_NVME_GEN3 (set TUNE_NVME_GEN3=1 to enable it)"
        return 0
    fi
    if [ "$OVERRIDE" = 1 ] && [ "$OPT_ALLOW_BOOT" != 1 ]; then
        _t_unselect a "protected host (PIGEMMA_I_KNOW=1): editing the boot config needs the owner's explicit approval -> add --allow-boot-edit"
        return 0
    fi
    return 0
}

# ======================================================================================= (b) governor
_cur_b() {
    B_PATHS=()
    B_VALS=()
    B_AVAIL=""
    B_ENABLED=""
    local base d p desc="" i same=1
    _pg_p /sys/devices/system/cpu/cpufreq
    base=$_P
    for d in "$base"/policy[0-9]*; do
        if [ -d "$d" ]; then B_PATHS+=("/sys/devices/system/cpu/cpufreq/${d##*/}/scaling_governor"); fi
    done
    if [ "${#B_PATHS[@]}" -eq 0 ]; then
        _pg_p /sys/devices/system/cpu
        for d in "$_P"/cpu[0-9]*; do
            if [ -d "$d/cpufreq" ]; then B_PATHS+=("/sys/devices/system/cpu/${d##*/}/cpufreq/scaling_governor"); fi
        done
    fi
    for p in "${B_PATHS[@]}"; do
        if _t_rd "$p"; then B_VALS+=("$_R"); else B_VALS+=("?"); fi
    done
    if [ "${#B_PATHS[@]}" -gt 0 ] && _t_rd "${B_PATHS[0]%/*}/scaling_available_governors"; then
        B_AVAIL=$_R
    fi
    _t_file_state "$F_UNIT" "$UNIT_CONTENT"
    B_UNIT_FS=$_FS
    if _t_systemd_live; then
        B_ENABLED=$(systemctl is-enabled "$UNIT_NAME" 2>/dev/null || true)
    fi
    if [ "${#B_PATHS[@]}" -eq 0 ]; then
        desc="cpufreq not exposed (no /sys/devices/system/cpu/cpufreq/policy*)"
    else
        for ((i = 1; i < ${#B_VALS[@]}; i++)); do
            if [ "${B_VALS[$i]}" != "${B_VALS[0]}" ]; then same=0; fi
        done
        if [ "$same" = 1 ] && [ "${#B_PATHS[@]}" = 1 ]; then
            p=${B_PATHS[0]%/scaling_governor}
            desc="${B_VALS[0]} (${p##*/})"
        elif [ "$same" = 1 ]; then
            p=${B_PATHS[0]%/scaling_governor}
            d=${B_PATHS[${#B_PATHS[@]} - 1]%/scaling_governor}
            desc="${B_VALS[0]} on ${#B_PATHS[@]} policies (${p##*/}..${d##*/})"
        else
            for ((i = 0; i < ${#B_PATHS[@]}; i++)); do
                p=${B_PATHS[$i]%/scaling_governor}
                desc+="${desc:+ }${p##*/}=${B_VALS[$i]}"
            done
        fi
        if [ -n "$B_AVAIL" ]; then desc+="; available: $B_AVAIL"; fi
    fi
    _t_fs_word "$B_UNIT_FS"
    desc+="; $UNIT_NAME: $_W${B_ENABLED:+ ($B_ENABLED)}"
    _R=$desc
    return 0
}

plan_b() {
    local i before_n
    I_TITLE[b]="CPU frequency governor -> $TUNE_GOVERNOR (persisted by $UNIT_NAME)"
    _cur_b
    I_CUR[b]=$_R
    I_PROP[b]="$TUNE_GOVERNOR on every cpufreq policy; $UNIT_NAME (oneshot, enabled) re-applies it at boot"
    if [ "${SEL[b]}" = 0 ]; then
        _t_skip b "${SEL_WHY[b]}"
        return 0
    fi
    if [ "$MODE" = revert ]; then
        I_PROP[b]="disable + remove $UNIT_NAME; restore the recorded governors"
        if [ "$B_UNIT_FS" = same ] || [ "$B_UNIT_FS" = ours ]; then
            _t_act b cmd "" "systemctl disable $UNIT_NAME"
            _t_act b rm "$F_UNIT"
            _t_act b cmd "" "systemctl daemon-reload"
        elif [ "$B_UNIT_FS" = foreign ]; then
            _t_note b "$F_UNIT exists but was NOT written by pi-gemma4: left untouched"
        fi
        _t_plan_restore b /sys/devices/system/cpu/
        return 0
    fi
    case $TUNE_GOVERNOR in
        keep | none)
            _t_skip b "TUNE_GOVERNOR=$TUNE_GOVERNOR: governor left alone"
            return 0
            ;;
    esac
    if [ "${#B_PATHS[@]}" -eq 0 ]; then
        _t_skip b "cpufreq is not exposed here (container / VM): nothing to tune"
        return 0
    fi
    if [ -n "$B_AVAIL" ] && [[ " $B_AVAIL " != *" $TUNE_GOVERNOR "* ]]; then
        _t_skip b "governor '$TUNE_GOVERNOR' is not available here (scaling_available_governors: $B_AVAIL)"
        return 0
    fi
    for ((i = 0; i < ${#B_PATHS[@]}; i++)); do
        if [ "${B_VALS[$i]}" != "$TUNE_GOVERNOR" ]; then
            _t_act b set "${B_PATHS[$i]}" "$TUNE_GOVERNOR"
        fi
    done
    if [ "$PERSIST" = 1 ]; then
        _t_nact b
        before_n=$_N
        _t_plan_file b "$F_UNIT" "$UNIT_CONTENT"
        _t_nact b
        if [ "$_N" -gt "$before_n" ]; then
            _t_act b cmd "" "systemctl daemon-reload"
            _t_act b cmd "" "systemctl enable $UNIT_NAME"
        elif [ "$B_UNIT_FS" = same ] && _t_systemd_live && [ "$B_ENABLED" != enabled ]; then
            _t_act b cmd "" "systemctl enable $UNIT_NAME"
        fi
    else
        I_PROP[b]="$TUNE_GOVERNOR on every cpufreq policy (live only: the boot default returns at reboot)"
        _t_note b "runtime-only (protected host or --runtime-only): no $UNIT_NAME; --persist installs it"
    fi
    if [ "$TUNE_GOVERNOR" = performance ]; then
        _t_note b "performance holds the cores at full clock: more heat - see (f) cooling"
    fi
    return 0
}

# ======================================================================================== (c) sysctl
_cur_c() {
    C_SWAP=""
    C_VFS=""
    if _t_rd /proc/sys/vm/swappiness; then C_SWAP=$_R; fi
    if _t_rd /proc/sys/vm/vfs_cache_pressure; then C_VFS=$_R; fi
    _t_file_state "$F_SYSCTL" "$SYSCTL_CONTENT"
    C_FS=$_FS
    _t_fs_word "$C_FS"
    _R="vm.swappiness=${C_SWAP:-unknown}, vm.vfs_cache_pressure=${C_VFS:-unknown}; $F_SYSCTL: $_W"
    return 0
}

_t_sysctl_overrides() { # notes for files that set our keys and are applied after 90-pigemma.conf
    local dir f b
    for dir in /etc/sysctl.d /run/sysctl.d /usr/local/lib/sysctl.d /usr/lib/sysctl.d /lib/sysctl.d; do
        _pg_p "$dir"
        for f in "$_P"/*.conf; do
            if [ ! -f "$f" ]; then continue; fi
            b=${f##*/}
            if [ "$b" = "${F_SYSCTL##*/}" ]; then continue; fi
            if [[ $b > ${F_SYSCTL##*/} ]] && grep -Eq '^[[:space:]]*vm\.(swappiness|vfs_cache_pressure)[[:space:]]*=' "$f" 2>/dev/null; then
                _t_note c "$dir/$b also sets vm.swappiness / vm.vfs_cache_pressure and is applied AFTER ${F_SYSCTL##*/} at boot: its value wins"
            fi
        done
    done
    _pg_p /etc/sysctl.conf
    if [ -f "$_P" ] && grep -Eq '^[[:space:]]*vm\.(swappiness|vfs_cache_pressure)[[:space:]]*=' "$_P" 2>/dev/null; then
        _t_note c "/etc/sysctl.conf also sets vm.swappiness / vm.vfs_cache_pressure (applied last by 'sysctl --system'): check which value wins"
    fi
    return 0
}

plan_c() {
    local swap_kb
    I_TITLE[c]="vm.swappiness=$TUNE_SWAPPINESS, vm.vfs_cache_pressure=$TUNE_VFS_CACHE_PRESSURE (persisted by $F_SYSCTL)"
    _cur_c
    I_CUR[c]=$_R
    I_PROP[c]="vm.swappiness=$TUNE_SWAPPINESS, vm.vfs_cache_pressure=$TUNE_VFS_CACHE_PRESSURE live (/proc/sys/vm) + $F_SYSCTL"
    if [ "${SEL[c]}" = 0 ]; then
        _t_skip c "${SEL_WHY[c]}"
        return 0
    fi
    if [ "$MODE" = revert ]; then
        I_PROP[c]="remove $F_SYSCTL; restore the recorded live values"
        _t_plan_rm c "$F_SYSCTL"
        _t_plan_restore c /proc/sys/vm/
        return 0
    fi
    if [ "$C_SWAP" != "$TUNE_SWAPPINESS" ]; then
        _t_act c set /proc/sys/vm/swappiness "$TUNE_SWAPPINESS"
    fi
    if [ "$C_VFS" != "$TUNE_VFS_CACHE_PRESSURE" ]; then
        _t_act c set /proc/sys/vm/vfs_cache_pressure "$TUNE_VFS_CACHE_PRESSURE"
    fi
    if [ "$PERSIST" = 1 ]; then
        _t_plan_file c "$F_SYSCTL" "$SYSCTL_CONTENT"
        _t_sysctl_overrides
    else
        I_PROP[c]="vm.swappiness=$TUNE_SWAPPINESS, vm.vfs_cache_pressure=$TUNE_VFS_CACHE_PRESSURE live only (boot defaults return at reboot)"
        _t_note c "runtime-only (protected host or --runtime-only): no $F_SYSCTL; --persist writes it"
    fi
    _t_note c "'sysctl --system' is deliberately not run (it would re-apply every sysctl.d file); writing /proc/sys/vm/<key> is what 'sysctl -w' does"
    _pg_meminfo
    swap_kb=${MI_SWAP:-}
    if [ -n "$swap_kb" ] && [ "$swap_kb" != 0 ]; then
        _t_note c "swap present (SwapTotal $(fmt_gb $((swap_kb * 1024)))): low swappiness keeps ANONYMOUS memory (repacked weights, KV cache) out of swap; mmap'ed weights (-nr) are file pages and are dropped and re-read regardless (ASSUMED kernel behaviour, not measured)"
    else
        _t_note c "no swap configured (SwapTotal ${swap_kb:-unknown} kB): vm.swappiness has no effect until swap exists"
    fi
    return 0
}

# ===================================================================================== (d) read-ahead
_cur_d() {
    D_DISK=""
    D_PATH=""
    D_VAL=""
    D_RULE_FS=absent
    detect_storage "$MODEL_DIR"
    D_DISK=$STORAGE_DISK
    if [ -n "$D_DISK" ]; then
        D_PATH="/sys/block/$D_DISK/queue/read_ahead_kb"
        if _t_rd "$D_PATH"; then D_VAL=$_R; fi
        _t_udev_content "$D_DISK"
        _t_file_state "$F_UDEV" "$_C"
        D_RULE_FS=$_FS
    elif _t_content "$F_UDEV"; then
        if [[ $_C == *"$T_MARK"* ]]; then D_RULE_FS=ours; else D_RULE_FS=foreign; fi
    fi
    _t_fs_word "$D_RULE_FS"
    if [ -n "$D_DISK" ]; then
        _R="MODEL_DIR $MODEL_DIR on ${STORAGE_DEV:-?} ($STORAGE_TYPE, disk $D_DISK, mount ${STORAGE_MOUNT:-?}); read_ahead_kb=${D_VAL:-unknown}; $F_UDEV: $_W"
    else
        _R="MODEL_DIR $MODEL_DIR on ${STORAGE_DEV:-?}${STORAGE_FSTYPE:+ ($STORAGE_FSTYPE)}: not a block device; $F_UDEV: $_W"
    fi
    return 0
}

plan_d() {
    local before_n n
    I_TITLE[d]="read_ahead_kb=$TUNE_READAHEAD_KB on the model device (persisted by a udev rule)"
    _cur_d
    I_CUR[d]=$_R
    I_PROP[d]="read_ahead_kb=$TUNE_READAHEAD_KB on ${D_DISK:-the model device} live + $F_UDEV"
    for n in "${STORAGE_NOTES[@]}"; do _t_note d "storage: $n"; done
    if [ "${SEL[d]}" = 0 ]; then
        _t_skip d "${SEL_WHY[d]}"
        return 0
    fi
    if [ "$MODE" = revert ]; then
        I_PROP[d]="remove $F_UDEV; restore the recorded read_ahead_kb"
        _t_nact d
        before_n=$_N
        _t_plan_rm d "$F_UDEV"
        _t_nact d
        if [ "$_N" -gt "$before_n" ]; then
            _t_act d cmd "" "udevadm control --reload-rules"
        fi
        _t_plan_restore d /sys/block/
        return 0
    fi
    if [ -z "$D_DISK" ]; then
        _t_skip d "MODEL_DIR is not on a block device (container overlay / network fs): read_ahead_kb not changed"
        return 0
    fi
    if [ "$D_VAL" != "$TUNE_READAHEAD_KB" ]; then
        _t_act d set "$D_PATH" "$TUNE_READAHEAD_KB"
    fi
    if [ "$PERSIST" = 1 ]; then
        _t_nact d
        before_n=$_N
        _t_udev_content "$D_DISK"
        _t_plan_file d "$F_UDEV" "$_C"
        _t_nact d
        if [ "$_N" -gt "$before_n" ]; then
            _t_act d cmd "" "udevadm control --reload-rules"
        fi
    else
        I_PROP[d]="read_ahead_kb=$TUNE_READAHEAD_KB on $D_DISK live only (the boot default returns at reboot)"
        _t_note d "runtime-only (protected host or --runtime-only): no udev rule; --persist writes it"
    fi
    if [ "${STORAGE_MOUNT:-}" = / ]; then
        _t_note d "$D_DISK also holds the root filesystem: the read-ahead applies to every file on it (services included)"
    fi
    case $D_DISK in
        sd*) _t_note d "the rule matches KERNEL==$D_DISK; USB disk names (sdX) can change between boots - re-run pi/01 after replugging" ;;
    esac
    return 0
}

# ======================================================================================== (e) memlock
_cur_e() {
    local u w1
    E_ULIMIT=$(memlock_limit)
    _t_file_state "$F_LIMITS" "$LIMITS_CONTENT"
    E_FS1=$_FS
    _t_file_state "$F_SDCONF" "$SDCONF_CONTENT"
    E_FS2=$_FS
    case $E_ULIMIT in
        unlimited) u=unlimited ;;
        "") u=unknown ;;
        *) u="$E_ULIMIT kB" ;;
    esac
    _t_fs_word "$E_FS1"
    w1=$_W
    _t_fs_word "$E_FS2"
    _R="ulimit -l (this shell): $u; $F_LIMITS: $w1; $F_SDCONF: $_W"
    return 0
}

plan_e() {
    I_TITLE[e]="memlock unlimited (limits.d + systemd DefaultLimitMEMLOCK)"
    _cur_e
    I_CUR[e]=$_R
    I_PROP[e]="'* soft/hard memlock unlimited' in $F_LIMITS; DefaultLimitMEMLOCK=infinity in $F_SDCONF"
    if [ "${SEL[e]}" = 0 ]; then
        _t_skip e "${SEL_WHY[e]}"
        return 0
    fi
    if [ "$MODE" = revert ]; then
        I_PROP[e]="remove $F_LIMITS and $F_SDCONF"
        _t_plan_rm e "$F_LIMITS"
        _t_plan_rm e "$F_SDCONF"
        _t_nact e
        if [ "$_N" -gt 0 ]; then I_FLAG[e]=relogin; fi
        return 0
    fi
    if ! _pg_truthy "$TUNE_MEMLOCK"; then
        _t_skip e "TUNE_MEMLOCK=$TUNE_MEMLOCK: memlock limits left alone"
        return 0
    fi
    if ! mlock_allowed; then
        _t_skip e "mlock of model weights is not allowed on this host (ALLOW_MLOCK=$ALLOW_MLOCK on a protected host, deviation D4): memlock limits left alone; ALLOW_MLOCK=1 overrides"
        return 0
    fi
    if [ "$PERSIST" = 0 ]; then
        _t_skip e "persistent-only setting (no live equivalent for running sessions) and this run is runtime-only; --persist writes it"
        return 0
    fi
    _t_plan_file e "$F_LIMITS" "$LIMITS_CONTENT"
    _t_plan_file e "$F_SDCONF" "$SDCONF_CONTENT"
    _t_nact e
    if [ "$_N" -gt 0 ]; then I_FLAG[e]=relogin; fi
    _t_note e "needed by pinner (pi/06) and llama.cpp -lm mlock; RE-LOGIN for shells (pam_limits), reboot or 'sudo systemctl daemon-reexec' for services (not run here)"
    return 0
}

# ========================================================================================= (f) advice
_t_throttle_desc() { # HEX -> _R
    local h=${1#0x} v bits=()
    _R=""
    if ! [[ $h =~ ^[0-9a-fA-F]+$ ]]; then return 0; fi
    v=$((16#$h))
    if ((v & 0x1)); then bits+=("under-voltage NOW"); fi
    if ((v & 0x2)); then bits+=("ARM frequency capped NOW"); fi
    if ((v & 0x4)); then bits+=("throttled NOW"); fi
    if ((v & 0x8)); then bits+=("soft temperature limit NOW"); fi
    if ((v & 0x10000)); then bits+=("under-voltage since boot"); fi
    if ((v & 0x20000)); then bits+=("frequency capping since boot"); fi
    if ((v & 0x40000)); then bits+=("throttling since boot"); fi
    if ((v & 0x80000)); then bits+=("soft temperature limit since boot"); fi
    if [ "${#bits[@]}" -eq 0 ]; then
        _R="0x$h (never throttled, no under-voltage since boot)"
    else
        _R="0x$h: ${bits[*]}"
    fi
    return 0
}

_cur_f() {
    local f n swaps="" zram=() fan="" temp thr desc
    _pg_meminfo
    _pg_p /proc/swaps
    if [ -r "$_P" ]; then
        while read -r f _ || [ -n "${f:-}" ]; do
            case $f in Filename | "") ;; *) swaps+="${swaps:+, }$f" ;; esac
            f=""
        done <"$_P"
    fi
    _pg_p /sys/block
    for f in "$_P"/zram*; do
        if [ -e "$f" ]; then zram+=("${f##*/}"); fi
    done
    F_ZRAM=${zram[*]}
    _pg_p /sys/class/hwmon
    for f in "$_P"/hwmon*; do
        if [ -r "$f/name" ]; then
            n=$(tr -d '\000\r\n' <"$f/name" 2>/dev/null || true)
            case $n in
                *fan*)
                    fan=$n
                    if [ -r "$f/fan1_input" ]; then fan+=" $(tr -d '\000\r\n' <"$f/fan1_input" 2>/dev/null || true) rpm"; fi
                    ;;
            esac
        fi
    done
    _pg_p /sys/devices/platform/cooling_fan
    if [ -z "$fan" ] && [ -d "$_P" ]; then fan="cooling_fan (device tree)"; fi
    F_FAN=$fan
    _pg_cpu_temp
    temp=$_R
    _pg_throttled
    thr=$_R
    F_THR_RAW=$thr
    desc="SwapTotal ${MI_SWAP:-?} kB${swaps:+ ($swaps)}${F_ZRAM:+; zram: $F_ZRAM}; CPU ${temp:-?} C; fan: ${fan:-not detected}"
    if [ -n "$thr" ]; then
        _t_throttle_desc "$thr"
        desc+="; get_throttled $_R"
    fi
    _R=$desc
    return 0
}

plan_f() {
    I_TITLE[f]="advice only: swap / zram, cooling, power supply, storage"
    _cur_f
    I_CUR[f]=$_R
    I_PROP[f]="no changes (advice only)"
    if [ "${SEL[f]}" = 0 ]; then
        _t_skip f "${SEL_WHY[f]}"
        return 0
    fi
    if [ "$MODE" = revert ]; then
        _t_skip f "advice only (nothing to revert)"
        return 0
    fi
    I_STATUS[f]=ADVICE
    I_EXEC[f]=0
    if [ -n "${MI_SWAP:-}" ] && [ "${MI_SWAP:-0}" != 0 ]; then
        _t_note f "swap: model weights must never be swapped. Use -nr (mmap) or leave RAM for the repacked copy; zram (compressed RAM) for the services' idle pages is fine, a swap FILE on the microSD is not"
    else
        _t_note f "swap: none. Fine for inference (an OOM is better than swapping weights); optional zram (e.g. zram-tools) holds idle service pages"
    fi
    if [ -n "$F_FAN" ]; then
        _t_note f "cooling: fan detected ($F_FAN). The Pi 5 throttles from 80 C (hard limit 85 C); watch 'vcgencmd measure_temp' and 'vcgencmd get_throttled' during long decodes"
    else
        _t_note f "cooling: no fan detected here - fit the Active Cooler: decode keeps 4 cores busy for minutes and the Pi 5 throttles from 80 C"
    fi
    if [ -n "$F_THR_RAW" ] && [ "$F_THR_RAW" != 0x0 ]; then
        _t_note f "get_throttled is not 0x0: fix power / cooling before measuring tok/s (bits 0/16 = under-voltage: use the 27 W 5 V / 5 A supply)"
    else
        _t_note f "power: use the official 27 W (5 V / 5 A) supply; under-voltage slows the CPU and can corrupt the SD card"
    fi
    case ${STORAGE_TYPE:-unknown} in
        microsd) _t_note f "storage: microSD (~90 MB/s; 91.6 MB/s measured on context-friend-1): loading a 12.7 GB model takes ~2.5 min and streaming experts from it is not viable; upgrade: NVMe HAT (PCIe Gen3 x1 ~850 MB/s, uncertified) or a USB 3 SSD (~380 MB/s) - catalog values, assumed" ;;
        usb-ssd) _t_note f "storage: USB SSD (~380 MB/s, catalog, assumed): use a blue USB 3 port; NVMe Gen3 x1 (~850 MB/s) is about 2x faster" ;;
        nvme) _t_note f "storage: NVMe on the PCIe x1 lane (Gen2 ~450 MB/s, Gen3 ~850 MB/s; catalog, assumed); see item (a) for Gen3" ;;
        *) _t_note f "storage: type unknown here (container / overlay): the catalog's conservative microSD numbers apply" ;;
    esac
    return 0
}

# ======================================================================================= print / exec
_t_show_body() { # CONTENT -> heredoc body lines
    local line
    while IFS= read -r line || [ -n "$line" ]; do
        printf '    > %s\n' "$line"
    done <<<"$1"
    printf '    > EOF\n'
    return 0
}

_t_sudo() { # SYS_PATH -> array _S: privilege prefix on the real system (empty for root or a user-writable file)
    _S=()
    if [ "$EUID" = 0 ]; then return 0; fi
    if [ -n "${PIGEMMA_BOOT_CONFIG:-}" ] && [ "${1:-}" = "$PIGEMMA_BOOT_CONFIG" ] && [ -w "$1" ]; then
        return 0
    fi
    _S=(sudo -n)
    return 0
}

_t_show() { # IDX [short] -> the exact command on stdout
    local i=$1 short=${2:-} t p a pre n tf
    t=${A_TYPE[$i]}
    p=${A_PATH[$i]}
    a=${A_ARG[$i]}
    _t_sudo "$p"
    pre="${_S[*]}"
    pre="${pre:+$pre }"
    case $t in
        mkdir) printf '    $ %smkdir -p %s\n' "$pre" "$(_t_q "$p")" ;;
        write | append)
            tf=tee
            if [ "$t" = append ]; then tf="tee -a"; fi
            if [ -n "$short" ]; then
                n=$(printf '%s\n' "$a" | wc -l)
                printf '    $ %s%s %s >/dev/null   (%s lines)\n' "$pre" "$tf" "$(_t_q "$p")" "${n// /}"
            else
                printf "    \$ %s%s %s >/dev/null <<'EOF'\n" "$pre" "$tf" "$(_t_q "$p")"
                _t_show_body "$a"
            fi
            ;;
        set | restore) printf '    $ echo %s | %stee %s >/dev/null\n' "$(_t_q "$a")" "$pre" "$(_t_q "$p")" ;;
        backup) printf '    $ %scp %s %s\n' "$pre" "$(_t_q "$p")" "$(_t_q "$(_t_bak "$p")")" ;;
        sed_delete) printf "    \$ %ssed -i '%s' %s\n" "$pre" "$SED_EXPR" "$(_t_q "$p")" ;;
        rm) printf '    $ %srm -f %s\n' "$pre" "$(_t_q "$p")" ;;
        cmd) printf '    $ %s%s\n' "$pre" "$a" ;;
    esac
    return 0
}

_t_materialize() { # SYS_PATH: in the test sysroot, copy the viewed file in before editing it
    _t_target "$1"
    if [ -e "$_T" ]; then return 0; fi
    _pg_p "$1"
    if [ -f "$_P" ]; then
        mkdir -p "${_T%/*}" && cp -p "$_P" "$_T"
        return
    fi
    return 1
}

_t_exec() { # IDX -> 0 ok, 1 failed
    local i=$1 t p a tgt w=()
    t=${A_TYPE[$i]}
    p=${A_PATH[$i]}
    a=${A_ARG[$i]}
    if [ "$t" = set ] && _t_rd "$p"; then
        _t_state_record "$p" "$_R"
    fi
    _t_sudo "$p"
    _t_target "$p"
    tgt=$_T
    if [ -n "$TUNE_ROOT" ]; then
        case $t in
            mkdir) mkdir -p "$tgt" ;;
            write) mkdir -p "${tgt%/*}" && printf '%s\n' "$a" >"$tgt" ;;
            append) _t_materialize "$p" && printf '%s\n' "$a" >>"$tgt" ;;
            set | restore) mkdir -p "${tgt%/*}" && echo "$a" >"$tgt" ;;
            backup) _t_materialize "$p" && cp "$tgt" "$(_t_bak "$tgt")" ;;
            sed_delete) _t_materialize "$p" && sed -i "$SED_EXPR" "$tgt" ;;
            rm) rm -f "$tgt" ;;
            cmd) printf '      (test sysroot: not executed)\n' ;;
            *) false ;;
        esac || return 1
    else
        case $t in
            mkdir) "${_S[@]}" mkdir -p "$p" ;;
            write) printf '%s\n' "$a" | "${_S[@]}" tee "$p" >/dev/null ;;
            append) printf '%s\n' "$a" | "${_S[@]}" tee -a "$p" >/dev/null ;;
            set | restore) echo "$a" | "${_S[@]}" tee "$p" >/dev/null ;;
            backup) "${_S[@]}" cp "$p" "$(_t_bak "$p")" ;;
            sed_delete) "${_S[@]}" sed -i "$SED_EXPR" "$p" ;;
            rm) "${_S[@]}" rm -f "$p" ;;
            cmd)
                read -r -a w <<<"$a" || true
                "${_S[@]}" "${w[@]}"
                ;;
            *) false ;;
        esac || return 1
    fi
    return 0
}

_t_print_item() { # ITEM
    local x=$1 i n line hdr
    printf '\n(%s) %s\n' "$x" "${I_TITLE[$x]}"
    printf '    current : %s\n' "${I_CUR[$x]}"
    printf '    proposed: %s\n' "${I_PROP[$x]}"
    printf '    status  : %s%s\n' "${I_STATUS[$x]}" "${I_WHY[$x]:+ - ${I_WHY[$x]}}"
    if [ -n "${I_NOTES[$x]}" ]; then
        while IFS= read -r line; do
            printf '    note    : %s\n' "$line"
        done <<<"${I_NOTES[$x]}"
    fi
    if [ "${I_FLAG[$x]}" = reboot ] && [ "${I_EXEC[$x]}" = 1 ]; then
        printf '    note    : REBOOT REQUIRED after this change (the boot config is read at boot); this script never reboots\n'
    fi
    _t_nact "$x"
    n=$_N
    if [ "$n" -gt 0 ]; then
        if [ "${I_EXEC[$x]}" = 1 ]; then
            if [ "$APPLY" = 1 ]; then hdr="commands (executed below):"; else hdr="commands (dry run - not executed):"; fi
        else
            hdr="commands (NOT executed: ${I_STATUS[$x],,}):"
        fi
        printf '    %s\n' "$hdr"
        for ((i = 0; i < ${#A_ITEM[@]}; i++)); do
            if [ "${A_ITEM[$i]}" = "$x" ]; then _t_show "$i"; fi
        done
    fi
    return 0
}

_t_rescan() { # ITEM -> _R = fresh current description
    case $1 in
        a) _cur_a ;;
        b) _cur_b ;;
        c) _cur_c ;;
        d) _cur_d ;;
        e) _cur_e ;;
        *) _R="" ;;
    esac
    return 0
}

# ============================================================================================= main
_t_state_load

# (d) runs detect_storage, which (a) and (f) quote; scan it first so every item sees STORAGE_TYPE.
_cur_d
plan_a
plan_b
plan_c
plan_d
plan_e
plan_f
for _x in "${ITEMS[@]}"; do _t_finalize "$_x"; done

_t_ram_kb=$(mem_total_kb)
_t_ram="unknown"
if [ -n "$_t_ram_kb" ]; then _t_ram="MemTotal $_t_ram_kb kB = $(fmt_gib $((_t_ram_kb * 1024)))"; fi
if [ "$APPLY" = 1 ]; then
    _t_mode_word="APPLY: the commands below are executed"
else
    _t_mode_word="DRY RUN: nothing is changed (add --apply to execute)"
fi
if [ "$EUID" = 0 ]; then _t_who=root; else _t_who="user (sudo -n per privileged command)"; fi
if [ "$PERSIST" = 1 ]; then _t_pers=yes; else _t_pers="no (runtime-only)"; fi
printf 'pi-gemma4 OS tuning (pi/01_tune_os.sh %s) - %s - %s\n' "$TUNE_VERSION" "$MODE" "$_t_mode_word"
printf 'host    : %s | %s | kernel %s | page %s B | %s | %s\n' "$(host_name)" "${A_BOARD:-board unknown}" \
    "$(kernel_release)" "$(page_size)" "$_t_ram" "$_t_who"
printf 'options : persistence files %s | items selected:' "$_t_pers"
for _x in "${ITEMS[@]}"; do
    if [ "${SEL[$_x]}" = 1 ]; then printf ' %s' "$_x"; fi
done
printf '\n'
if [ "$OVERRIDE" = 1 ]; then
    printf 'PROTECTED HOST OVERRIDE (PIGEMMA_I_KNOW=1): runtime-only unless --persist; no boot config edit without --allow-boot-edit; no memlock change unless mlock is allowed (ALLOW_MLOCK=%s)\n' "$ALLOW_MLOCK"
fi
if is_emulated; then
    printf 'EMULATED host (PIGEMMA_EMULATED=1): tuning an emulated container changes nothing on a real Pi\n'
fi
if in_container && [ -z "$TUNE_ROOT" ]; then
    printf 'CONTAINER: --apply is refused here (it would only change the container); the plan shows what --apply would do on a real Pi\n'
fi
if [ -n "$TUNE_ROOT" ]; then
    printf 'TEST SYSROOT: PIGEMMA_TUNE_ROOT=%s - writes go below it; systemctl / udevadm are not executed\n' "$TUNE_ROOT"
fi

for _x in "${ITEMS[@]}"; do _t_print_item "$_x"; done

N_CHANGE=0
N_OK=0
N_SKIP=0
N_FAIL=0
N_DONE=0
REBOOT=0
RELOGIN=0
if [ "$APPLY" = 1 ]; then
    _t_need_priv=0
    for ((_i = 0; _i < ${#A_ITEM[@]}; _i++)); do
        if [ "${I_EXEC[${A_ITEM[$_i]}]}" = 1 ] && [ -z "$TUNE_ROOT" ]; then
            _t_sudo "${A_PATH[$_i]}"
            if [ "${#_S[@]}" -gt 0 ]; then _t_need_priv=1; fi
        fi
    done
    if [ "$_t_need_priv" = 1 ]; then
        if ! have_cmd sudo; then
            die 3 "--apply needs root: not root and sudo is not installed (nothing was changed)"
        fi
        if ! sudo -n true 2>/dev/null; then
            die 3 "--apply needs root: 'sudo -n' needs a password here (nothing was changed). Run: sudo -E bash pi/01_tune_os.sh --apply"
        fi
    fi
    _t_n_exec=0
    for _x in "${ITEMS[@]}"; do
        if [ "${I_EXEC[$_x]}" = 1 ]; then _t_n_exec=$((_t_n_exec + 1)); fi
    done
    if [ "$_t_n_exec" -gt 0 ]; then
        printf '\n== applying (%s item(s)) ==\n' "$_t_n_exec"
    else
        printf '\n== applying: nothing to do (every selected item is already as proposed, skipped or not selected) ==\n'
    fi
    for _x in "${ITEMS[@]}"; do
        if [ "${I_EXEC[$_x]}" != 1 ]; then continue; fi
        printf '\n(%s) %s\n' "$_x" "${I_TITLE[$_x]}"
        _t_ok_item=1
        for ((_i = 0; _i < ${#A_ITEM[@]}; _i++)); do
            if [ "${A_ITEM[$_i]}" != "$_x" ]; then continue; fi
            _t_show "$_i" short
            if ! _t_exec "$_i"; then
                printf '      FAILED\n'
                _t_ok_item=0
                break
            fi
        done
        _t_rescan "$_x"
        printf '    before  : %s\n' "${I_CUR[$_x]}"
        printf '    after   : %s\n' "$_R"
        if [ "$_t_ok_item" = 1 ]; then
            I_STATUS[$_x]=APPLIED
            if [ "${I_FLAG[$_x]}" = reboot ]; then REBOOT=1; fi
            if [ "${I_FLAG[$_x]}" = relogin ]; then RELOGIN=1; fi
        else
            I_STATUS[$_x]=FAILED
            warn "item ($_x) failed: the remaining commands of this item were not run"
        fi
        printf '    status  : %s\n' "${I_STATUS[$_x]}"
    done
    if [ "$MODE" = revert ]; then
        # recorded originals of items that were reverted (or needed nothing) are no longer needed
        for _x in b c d; do
            case ${I_STATUS[$_x]} in
                APPLIED | OK)
                    case $_x in
                        b) _t_state_forget_prefix /sys/devices/system/cpu/ ;;
                        c) _t_state_forget_prefix /proc/sys/vm/ ;;
                        d) _t_state_forget_prefix /sys/block/ ;;
                    esac
                    ;;
            esac
        done
    fi
fi

for _x in "${ITEMS[@]}"; do
    case ${I_STATUS[$_x]} in
        CHANGE) N_CHANGE=$((N_CHANGE + 1)) ;;
        APPLIED) N_DONE=$((N_DONE + 1)) ;;
        OK) N_OK=$((N_OK + 1)) ;;
        FAILED) N_FAIL=$((N_FAIL + 1)) ;;
        ADVICE) ;;
        *) N_SKIP=$((N_SKIP + 1)) ;;
    esac
done

printf '\n'
if [ "$APPLY" = 1 ]; then
    printf 'summary : %s applied, %s already OK, %s skipped / not selected, %s failed (%s)\n' \
        "$N_DONE" "$N_OK" "$N_SKIP" "$N_FAIL" "$MODE"
    if [ "$REBOOT" = 1 ]; then
        printf 'REBOOT REQUIRED: the boot config changed; reboot when convenient (this script never reboots; on a protected host only with the owner'"'"'s approval)\n'
    fi
    if [ "$RELOGIN" = 1 ]; then
        printf 'RE-LOGIN REQUIRED: memlock limits apply to new login sessions; services get DefaultLimitMEMLOCK after a reboot or '"'"'sudo systemctl daemon-reexec'"'"' (not run here)\n'
    fi
    if [ "$MODE" = tune ] && [ -f "$STATE_FILE" ]; then
        printf 'state   : original live values recorded in %s (used by --revert --apply)\n' "$STATE_FILE"
    fi
else
    printf 'summary : %s to change, %s already OK, %s skipped / not selected (%s plan)\n' "$N_CHANGE" "$N_OK" "$N_SKIP" "$MODE"
    if [ "${I_EXEC[a]}" = 1 ]; then
        printf 'REBOOT REQUIRED after --apply for item (a) (this script never reboots)\n'
    fi
    if [ "$N_CHANGE" -gt 0 ]; then
        if [ "$MODE" = revert ]; then
            printf 'DRY RUN - nothing was changed. To undo: bash pi/01_tune_os.sh --revert --apply\n'
        else
            printf 'DRY RUN - nothing was changed. To apply: bash pi/01_tune_os.sh --apply   (root or passwordless sudo)\n'
        fi
    else
        printf 'DRY RUN - nothing was changed; nothing to do.\n'
    fi
fi

if [ "$N_FAIL" -gt 0 ]; then
    die 1 "$N_FAIL item(s) failed; see above (completed items stay applied; --revert --apply undoes them)"
fi
exit 0
