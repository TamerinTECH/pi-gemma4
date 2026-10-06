#!/usr/bin/env bash
# shellcheck shell=bash
# pi/00_feasibility.sh - step 0 on any Raspberry Pi (or any Linux box): READ-ONLY fact collection,
# feasibility verdict table and recommendation for the Gemma 4 models of pi-gemma4.
#
# This is the ONLY pi script that may run on an unknown or protected host: it reads /proc, /sys, /boot
# and command output (pi/lib/common.sh collect_facts), runs the pure decision logic
# `python3 -m pigemma.feasibility` (stdlib only) and prints a report. It never stops, starts, kills,
# installs, tunes or downloads anything. The only files it writes are its outputs in $PIGEMMA_OUT
# (default pi/out/): facts.json, feasibility.json and feasibility.txt (a copy of the report).
#
# Usage:
#   pi/00_feasibility.sh [--want 26b|31b|<model-id>] [--json pi/out/feasibility.json]
#                        [--sim results/h1/sim.json] [options] [-- extra pigemma.feasibility args]
#   (run `pi/00_feasibility.sh --help` for every option)
#
# Exit code = verdict code of pigemma.feasibility (the convention of every pi script):
#   0  a model fits in RAM (FITS_*) or streams well (STREAM_OK)  -> heavier scripts may run
#   2  only slow operation (STREAM_SLOW, or a slow dense run)    -> only steps that accept it run
#   3  nothing viable (or no disk for any viable file); also: no Python >= 3.10 to evaluate the facts
#   4  refused: protected host (DENY_HOSTS / catalog protected_hosts / protected services) without
#      PIGEMMA_I_KNOW=1, or busy host (loadavg1 / MemAvailable)
#   1  usage error, unreadable input, or pigemma.feasibility failed
#
# Report layout (stdout; with `--json -` the report goes to stderr and stdout carries the JSON):
#   header line, "==== host facts" (RAM total/available, storage type + link speed + read_ahead_kb, free
#   disk, page size, CPU flags as 'asimddp: yes, i8mm: no', kernel, governor, HAT, protected-host flag),
#   a loud banner on protected / overridden / busy hosts, "==== verdict table" (pigemma.feasibility's
#   own table: per model/quant verdict, projected tok/s and its source, details, recommendation, reasons),
#   "==== summary" (verdict, recommended model/quant/flags, the flags as they will be used on THIS host
#   after the mlock policy, the pi/04 disk rule check with the best alternative that fits the disk,
#   projection inputs, next steps, the fixed hardware recommendation line, output paths).
#
# Design decisions (also listed as open issues for the orchestrator):
# * The protected-host gate is NOT assert_safe_host (that would exit before the report). The bash gate
#   (protected_host_reasons, a superset of pigemma.feasibility's glob rule) is evaluated, a banner is
#   printed, and the verdict still runs so the operator sees the table. If the bash gate refuses while
#   pigemma.feasibility did not (a glob that only matches the name without '.service'), the exit code is
#   raised to 4 so pi/00 is never weaker than the gate every other script applies.
# * --facts FILE evaluates an existing facts.json (another host, a fixture) without collecting; the
#   protected/busy state then comes from the verdict JSON, not from this machine.
# * The disk check in the summary applies pi/04's rule (require_free_disk: max(MIN_FREE_DISK_FACTOR x
#   bytes, bytes + KEEP_FREE_GB when the model dir is on /)) to the recommended file. It does not change
#   the exit code (= the verdict code); it says when pi/04 will refuse and names the best row that fits
#   (e.g. context-friend-1: the 26B Q4_0 fits in RAM but not on its 15.5 GB-free microSD -> UD-Q3_K_M).
# * feasibility.json is written to a temp file first and moved into place only when pigemma.feasibility
#   produced a verdict (exit 0/2/3/4), so a failed run never leaves a half-written verdict behind.
# * The report renderer is an embedded stdlib-only Python program (written to a temp file in
#   $PIGEMMA_OUT and run from there): Python >= 3.10 is required anyway for the verdict.
set -euo pipefail

PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
. "$PI_DIR/lib/common.sh"

PG00_VERSION="1.0.0"
PG00_FALLBACK_RECOMMENDATION="Pi 5 16 GB + NVMe Gen3: 26B-A4B fits in RAM (-nr); 8 GB + NVMe Gen3: stream experts (~40% cached); 8 GB + USB3: weak; 4 GB or microSD: not viable; 31B dense: never good speed on any Pi 5"

usage() {
    cat <<EOF
pi/00_feasibility.sh $PG00_VERSION - read-only Raspberry Pi feasibility verdict for the Gemma 4 models.

Usage: pi/00_feasibility.sh [options] [-- extra pigemma.feasibility args]

  --want X            restrict the rows to a family (26b | 31b) or one catalog model id
  --json OUT          verdict JSON (default \$PIGEMMA_OUT/feasibility.json); '-' = JSON on stdout,
                      the report then goes to stderr
  --sim FILE          results/h1/sim.json: LRU hit rates from the cache simulation replace the assumed curve
  --facts FILE        evaluate an existing facts.json instead of collecting facts on this host
  --facts-out FILE    where the collected facts go (default \$PIGEMMA_OUT/facts.json)
  --catalog FILE      catalog (default \$PIGEMMA_CATALOG = pigemma/data/catalog.json)
  --io-mem V          auto | none | PATH: measured storage/memory numbers (auto = results/pi/io_mem.json if present)
  --calibration V     auto | none | PATH: measured llama efficiency (auto = results/pi/calibration.json if present)
  --device NAME       force a catalog device profile (e.g. pi5-16gb-nvme-gen3)
  --report FILE|none  copy of the printed report (default \$PIGEMMA_OUT/feasibility.txt)
  -q, --quiet         hide the configuration dump and info logs on stderr (the report is still printed)
  -h, --help          this help;  --version  print the version

Configuration: pi/pi_config.env (THREADS, CTX, OS_RESERVE_GIB, MODEL_DIR, DENY_HOSTS,
PROTECTED_SERVICE_GLOBS, ALLOW_MLOCK, MIN_FREE_DISK_FACTOR, KEEP_FREE_GB ...); environment values win.
PIGEMMA_I_KNOW=1 is the authorised override of the protected-host refusal (never of the busy refusal).
PIGEMMA_EMULATED=1 labels the run as emulated. PIGEMMA_FAKE_PROC=<dir> reads a fake system root (tests).

Exit codes: 0 FITS_* or STREAM_OK available | 2 only slow operation | 3 nothing viable (or no Python)
            4 refused (protected or busy host) | 1 usage / input error.
This script never modifies the host; it only writes its outputs into \$PIGEMMA_OUT.
EOF
}

_pg00_usage_err() {
    usage >&2
    die 1 "$*"
}

_pg00_abs() { # absolute form of a user path (relative to the caller's cwd); '-' and empty unchanged
    case ${1:-} in
        "" | - | /* | [A-Za-z]:/* | [A-Za-z]:\\*) printf '%s\n' "${1:-}" ;;
        *) printf '%s/%s\n' "$PWD" "$1" ;;
    esac
}

# ------------------------------------------------------------------------------------- arguments
want=""
json_arg=""
sim=""
facts_in=""
facts_out=""
catalog=""
io_mem=""
calibration=""
device=""
report=""
extra=()
while [ $# -gt 0 ]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        --version)
            printf 'pi/00_feasibility.sh %s (pi/lib/common.sh %s)\n' "$PG00_VERSION" "$PIGEMMA_COMMON_VERSION"
            exit 0
            ;;
        --want | --json | --sim | --facts | --facts-out | --catalog | --io-mem | --calibration | --device | --report)
            if [ $# -lt 2 ] || [ -z "${2:-}" ]; then
                _pg00_usage_err "$1 needs a value"
            fi
            _opt=$1
            _val=$2
            shift 2
            ;;
        --want=* | --json=* | --sim=* | --facts=* | --facts-out=* | --catalog=* | --io-mem=* | --calibration=* | --device=* | --report=*)
            _opt=${1%%=*}
            _val=${1#*=}
            shift
            if [ -z "$_val" ]; then
                _pg00_usage_err "$_opt needs a value"
            fi
            ;;
        -q | --quiet)
            PIGEMMA_QUIET=1
            export PIGEMMA_QUIET
            shift
            continue
            ;;
        --)
            shift
            extra=("$@")
            break
            ;;
        *)
            _pg00_usage_err "unknown option '$1' (see --help)"
            ;;
    esac
    case $_opt in
        --want) want=$_val ;;
        --json) json_arg=$_val ;;
        --sim) sim=$_val ;;
        --facts) facts_in=$_val ;;
        --facts-out) facts_out=$_val ;;
        --catalog) catalog=$_val ;;
        --io-mem) io_mem=$_val ;;
        --calibration) calibration=$_val ;;
        --device) device=$_val ;;
        --report) report=$_val ;;
    esac
done

if [ -n "$want" ] && ! [[ $want =~ ^[A-Za-z0-9_.-]+$ ]]; then
    _pg00_usage_err "--want '$want': expected 26b, 31b or a catalog model id"
fi
if [ -n "$device" ] && ! [[ $device =~ ^[A-Za-z0-9_.:-]+$ ]]; then
    _pg00_usage_err "--device '$device': expected a catalog device name"
fi

pigemma_init

# Resolve every path against the caller's cwd (pigemma.feasibility runs from PIGEMMA_HOME).
json_arg=$(_pg00_abs "$json_arg")
sim=$(_pg00_abs "$sim")
facts_in=$(_pg00_abs "$facts_in")
facts_out=$(_pg00_abs "${facts_out:-$PIGEMMA_OUT/facts.json}")
catalog=$(_pg00_abs "${catalog:-$PIGEMMA_CATALOG}")
case ${report:-} in
    none | NONE) report="" ;;
    "") report="$PIGEMMA_OUT/feasibility.txt" ;;
    *) report=$(_pg00_abs "$report") ;;
esac
json_final=${json_arg:-$PIGEMMA_OUT/feasibility.json}
if [ "$facts_out" = - ]; then
    _pg00_usage_err "--facts-out needs a file path ('-' is not supported)"
fi
for _pair in "sim:$sim" "facts:$facts_in" "catalog:$catalog"; do
    _f=${_pair#*:}
    if [ -n "$_f" ] && [ ! -f "$_f" ]; then
        die 1 "--${_pair%%:*} file not found: $_f"
    fi
done
for _pair in "io-mem:$io_mem" "calibration:$calibration"; do
    _f=${_pair#*:}
    case $_f in
        "" | auto | none) ;;
        *)
            _f=$(_pg00_abs "$_f")
            if [ ! -f "$_f" ]; then
                die 1 "--${_pair%%:*} file not found: $_f"
            fi
            if [ "${_pair%%:*}" = io-mem ]; then io_mem=$_f; else calibration=$_f; fi
            ;;
    esac
done

# Scratch space (inside PIGEMMA_OUT: no assumption about /tmp; removed at exit).
tmpd="$PIGEMMA_OUT/.pg00.$$"
mkdir -p "$tmpd" || die 1 "cannot create $tmpd"
add_cleanup "rm -rf '$tmpd'"

# ------------------------------------------------------------------------------------------ facts
if [ -n "$facts_in" ]; then
    facts_mode=file
    facts_file=$facts_in
    log "using existing facts (no collection on this host): $facts_file"
else
    facts_mode=live
    facts_file=$facts_out
    write_facts "$facts_file"
fi

if ! pigemma_python >/dev/null 2>&1; then
    printf '==== facts.json (%s) ====\n' "$facts_file"
    cat "$facts_file"
    die 3 "no usable Python >= 3.10 to evaluate the facts (PIGEMMA_PYTHON='${PIGEMMA_PYTHON:-}', python3, python): sudo apt-get install -y python3 (pigemma never installs packages itself)"
fi
_pg_need_python

# ---------------------------------------------------------------------------------- report renderer
_pg00_renderer() {
    cat <<'PY'
"""Report renderer of pi/00_feasibility.sh (stdlib only). Modes: host FACTS | protstate FEAS | summary FEAS FACTS."""
import fnmatch
import json
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace", newline="\n")
    except (AttributeError, ValueError, OSError):
        pass

LW = 15
ARM_FLAGS = ("asimddp", "i8mm", "sve", "fphp", "asimdhp", "bf16")
X86_FLAGS = ("avx2", "avx512f")


def env(name, default=""):
    return os.environ.get(name, default)


def num(x):
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def kb2b(x):
    v = num(x)
    return None if v is None else v * 1024.0


def gb(b, d=2):
    v = num(b)
    return "n/a" if v is None else f"{v / 1e9:.{d}f} GB"


def gib(b, d=2):
    v = num(b)
    return "n/a" if v is None else f"{v / 2 ** 30:.{d}f} GiB"


def yesno(v):
    if v is None:
        return "unknown"
    return "yes" if v else "no"


def tok(v):
    v = num(v)
    if v is None:
        return "n/a"
    return f"{v:.3f}" if v < 0.1 else f"{v:.2f}"


def src_label(src):  # same reading as pigemma.feasibility's table column
    s = str(src or "").lower()
    if "assumed" in s:
        return "assumed inputs"
    if "simulated" in s:
        return "simulated hit rates"
    return "measured inputs"


def kind_of(src):
    s = str(src or "").strip().lower()
    for k in ("measured", "simulated", "assumed"):
        if s.startswith(k):
            return k
    return s.split(":", 1)[0] if s else "unknown"


def out(label, text=""):
    print(f"  {label:<{LW}} {text}".rstrip())


def load(path):
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------ host
def cpu_flag_text(f):
    has = f.get("cpu_has") if isinstance(f.get("cpu_has"), dict) else {}
    flags = {str(x).lower() for x in (f.get("cpu_flags") or [])}
    if not flags and not has:
        return "unknown (no CPU flags in /proc/cpuinfo)", ""

    def h(k):
        return bool(has[k]) if k in has else (k in flags)

    is_x86 = bool(flags & {"sse2", "avx", "avx2"}) or bool(has.get("avx2")) or bool(has.get("avx512f"))
    is_arm = bool(flags & {"asimd", "fp", "neon"}) or any(bool(has.get(k)) for k in ARM_FLAGS) or not is_x86
    parts = []
    if is_arm:
        parts += [f"{k}: {'yes' if h(k) else 'no'}" for k in ARM_FLAGS]
    if is_x86:
        parts += [f"{k}: {'yes' if h(k) else 'no'}" for k in X86_FLAGS]
    if is_arm and h("asimddp"):
        hint = "dotprod present: llama.cpp's fast Q4_0/Q8_0 ARM kernels apply (GGML_NATIVE=ON detects them)"
    elif is_arm:
        hint = "no dotprod (asimddp): llama.cpp will be far slower than the Pi 5 projections"
    else:
        hint = "x86 host: not a Pi CPU - projections describe the matched device profile, not this machine"
    return ", ".join(parts), hint


def render_host(path):
    f = load(path)
    st = f.get("storage") if isinstance(f.get("storage"), dict) else {}
    print("==== host facts (read-only) ====")
    board = f.get("board_model")
    out("hostname", f"{f.get('hostname') or 'unknown'}" + (f"   ({board})" if board else ""))
    ps = num(f.get("page_size"))
    ps_txt = "unknown" if ps is None else f"{int(ps)} B ({int(ps) // 1024} KiB)"
    out("kernel", f"{f.get('kernel') or 'unknown'}   page size {ps_txt}")
    extra = [f"governor {f.get('governor') or 'unknown'}"]
    t = num(f.get("cpu_temp_c"))
    if t is not None:
        extra.append(f"temp {t:.1f} C")
    thr = f.get("throttled")
    if thr:
        extra.append(f"throttled {thr}" + (" (never)" if str(thr) == "0x0" else
                                           " (throttling / under-voltage seen: check cooler + PSU)"))
    out("cpu", f"{f.get('cpu_model') or 'unknown'} x {f.get('nproc') or '?'}   " + ", ".join(extra))
    flags_txt, hint = cpu_flag_text(f)
    out("cpu flags", flags_txt)
    if hint:
        out("", hint)
    tot, av, raw = kb2b(f.get("ram_total_kb")), kb2b(f.get("ram_available_kb")), kb2b(f.get("meminfo_available_kb"))
    ram = f"total {gb(tot)} ({gib(tot)}), available {gb(av)}"
    if raw is not None and av is not None and abs(raw - av) > 1024 * 1024:
        ram += f" (MemAvailable {gb(raw)}, capped by the cgroup)"
    ram += f", swap {gb(kb2b(f.get('swap_total_kb')))}"
    cg = num(f.get("cgroup_mem_max"))
    if cg:
        ram += f", cgroup memory limit {gb(cg)}"
    out("ram", ram)
    la = num(f.get("loadavg1"))
    out("loadavg1", "unknown" if la is None else f"{la:.2f}")
    typ = st.get("type") or "unknown"
    bits = []
    for k, label in (("dev", "dev"), ("disk", "disk"), ("transport", "transport")):
        if st.get(k):
            bits.append(f"{label} {st[k]}")
    if st.get("fstype"):
        bits.append(f"{st['fstype']}" + (f" on {st['mount']}" if st.get("mount") else ""))
    elif st.get("mount"):
        bits.append(f"mounted on {st['mount']}")
    out("storage", typ + (f" ({', '.join(bits)})" if bits else ""))
    link = st.get("pcie_link_speed")
    width = st.get("pcie_link_width")
    if typ != "nvme":
        link_txt = "n/a (not NVMe)"
    elif link:
        s = str(link)
        gen = " = Gen3" if s.startswith("8.0") else (" = Gen2" if s.startswith("5.0") else "")
        link_txt = s + (f" x{width}" if width else "") + gen
    else:
        link_txt = "unknown"
    ra = st.get("read_ahead_kb")
    out("", f"PCIe link {link_txt}; read_ahead_kb {ra if ra is not None else 'unknown'}; "
            f"rotational {yesno(st.get('rotational'))}")
    if typ == "unknown":
        out("", "storage type unknown: the catalog's conservative class numbers are used (see device profile)")
    on_root = f.get("model_dir_on_root_fs")
    out("model dir", f"{f.get('model_dir') or 'unknown'}" +
        ("" if on_root is None else f"   (on the root filesystem: {yesno(on_root)})"))
    out("free disk", gb(f.get("free_disk_bytes")) + (f" on {st['mount']}" if st.get("mount") else ""))
    hat = f.get("hat_product")
    gen_cfg = f.get("pciex1_gen_cfg")
    cfg_txt = f"config.txt pciex1_gen={gen_cfg}" if gen_cfg is not None else "config.txt pciex1_gen not set"
    if "boot_config" in f and not f.get("boot_config"):
        cfg_txt = "no boot config (container / not a Pi)"
    elif f.get("boot_config"):
        cfg_txt += f" ({f['boot_config']})"
    out("HAT / PCIe", f"HAT: {hat or 'none detected'}; {cfg_txt}")
    if hat and re.search(r"hailo|ai\s*hat|ai\s*kit", str(hat), re.I):
        out("", "AI HAT on the single PCIe lane: no help for 25-30B models; an NVMe needs a dual-slot HAT")
    ml = f.get("memlock_limit")
    if ml is None:
        ml_txt = "unknown"
    elif str(ml).strip().lower() in ("unlimited", "infinity"):
        ml_txt = "unlimited"
    else:
        ml_txt = f"{ml} kB (ulimit -l)"
    out("memlock", ml_txt)
    tools = f.get("tools") if isinstance(f.get("tools"), dict) else {}
    out("tools", ", ".join(f"{k} {v or 'missing'}" for k, v in tools.items()) or "unknown")
    svcs = [str(s) for s in (f.get("running_services") or [])]
    globs = [g for g in env("PG00_GLOBS").split() if g]

    def match(s):
        n = s.lower()
        return any(fnmatch.fnmatchcase(n, g.lower()) or fnmatch.fnmatchcase(n.removesuffix(".service"), g.lower())
                   for g in globs)

    prot = [s for s in svcs if match(s)]
    src = f.get("services_source")
    out("services", f"{len(svcs)} running" + (f" (source: {src})" if src else ""))
    out("", "protected matches: " + (", ".join(prot) if prot else "none") +
        (f"   (globs: {' '.join(globs)})" if globs else ""))
    ctx = [f"container: {yesno(f.get('in_container'))}", f"emulated: {yesno(f.get('emulated'))}"]
    if f.get("device"):
        ctx.append(f"device: {f['device']}")
    if f.get("override"):
        ctx.append(f"override: {f['override']}")
    if f.get("fake_proc"):
        ctx.append(f"FAKE system root: {f['fake_proc']}")
    out("context", ", ".join(ctx))
    out("collected", f"{f.get('collected_at') or 'unknown'} by {f.get('collector') or 'unknown'}")
    state = env("PG00_PROT_STATE")
    where = "here" if env("PG00_FACTS_MODE") == "live" else "on that host"
    if state == "refused":
        out("protected host", f"YES -> every other pi/*.sh script refuses {where} (exit 4); see the banner below")
    elif state == "overridden":
        out("protected host", "YES, override active (PIGEMMA_I_KNOW=1): heavy scripts run under the safety rules")
    elif state == "no":
        out("protected host", "no")
    else:
        out("protected host", "unknown")
    for r in [x for x in env("PG00_PROT_REASONS").split("\n") if x.strip()]:
        out("", f"- {r}")
    notes = [str(x) for x in (f.get("warnings") or [])] + [str(x) for x in (st.get("notes") or [])]
    if f.get("emulated"):
        notes.insert(0, "EMULATED host: speeds are projections for the device profile, not this machine")
    for i, n in enumerate(notes):
        out("notes" if i == 0 else "", f"- {n}")


# ------------------------------------------------------------------------------------- protstate
def render_protstate(path):
    d = load(path)
    reasons = [str(r) for r in (d.get("refusal_reasons") or [])]
    prot, busy = [], []
    state = "no"
    for r in reasons:
        if r.startswith("REFUSED (protected host): "):
            state = "refused"
            prot.append(r[len("REFUSED (protected host): "):])
        elif r.startswith("protected-host refusal overridden"):
            if state != "refused":
                state = "overridden"
            prot.append(r)
        elif r.startswith("REFUSED (busy): "):
            busy.append(r[len("REFUSED (busy): "):])
    if d.get("host_refused") and state != "refused":
        state = "refused"
    print(f"STATE {state}")
    print(f"BUSY {1 if (d.get('busy') or busy) else 0}")
    for r in prot:
        print(f"R {r}")
    for b in busy:
        print(f"B {b}")


# --------------------------------------------------------------------------------------- summary
def bpw(quant):
    try:
        from pigemma.feasibility import quant_bpw
        return float(quant_bpw(quant))
    except Exception:  # pragma: no cover - only when the package cannot be imported
        m = re.search(r"(\d+)", str(quant or ""))
        return float(m.group(1)) if m else 0.0


def rank(r):  # same order as pigemma.feasibility._rank
    v = r.get("verdict")
    if v in ("FITS_MAPPED", "FITS_REPACKED"):
        return 4
    if v == "STREAM_OK":
        return 3
    if v == "STREAM_SLOW":
        return 2
    if v == "NEVER_GOOD_SPEED" and r.get("runnable"):
        return 1
    return 0


def disk_lines(feas, facts, rec):
    factor = num(env("MIN_FREE_DISK_FACTOR")) or 1.1
    keep_gb = num(env("KEEP_FREE_GB"))
    keep_gb = 2.0 if keep_gb is None else keep_gb
    free = num(facts.get("free_disk_bytes"))
    on_root = facts.get("model_dir_on_root_fs")
    st = facts.get("storage") if isinstance(facts.get("storage"), dict) else {}
    if on_root is None:
        on_root = st.get("mount") == "/"
    keep = keep_gb * 1e9 if on_root else 0.0
    where = "/" if on_root else (st.get("mount") or "the model dir")

    def remaining(r):
        return max(0.0, (num(r.get("bytes")) or 0.0) - (num(r.get("present_bytes")) or 0.0))

    def need(r):
        rem = remaining(r)
        return 0.0 if rem <= 0 else max(factor * rem, rem + keep)

    rem = remaining(rec)
    if rem <= 0:
        out("disk (pi/04)", f"{rec['model_id']} is already in the model dir ({gb(rec.get('present_bytes'))}): "
                            "no download needed")
        return
    n = need(rec)
    if keep > 0:
        rule = f"max({factor:g} x {gb(rem)}, {gb(rem)} + keep-free {gb(keep)})"
    else:
        rule = f"{factor:g} x {gb(rem)}"
    if free is None:
        out("disk (pi/04)", f"{rec['model_id']} needs {rule} = {gb(n)} free on {where}; free disk unknown "
                            "(pi/04 checks again before downloading)")
        return
    if free >= n:
        out("disk (pi/04)", f"OK: {rec['model_id']} needs {rule} = {gb(n)} free on {where}; free {gb(free)}")
        return
    out("disk (pi/04)", f"NOT ENOUGH: {rec['model_id']} needs {rule} = {gb(n)} free on {where}; "
                        f"free {gb(free)} -> pi/04_fetch_model.sh will refuse this file")
    alts = [r for r in feas.get("rows") or [] if r.get("eligible") and rank(r) > 0 and
            r.get("model_id") != rec.get("model_id") and need(r) <= free]
    if alts:
        a = max(alts, key=lambda r: (rank(r), bpw(r.get("quant")), num(r.get("bytes")) or 0, num(r.get("tok_s")) or 0))
        out("", f"best row that also fits the disk: {a['model_id']} ({a.get('quant')}, {gb(a.get('bytes'))}) "
                f"{a.get('verdict')}, projected {tok(a.get('tok_s'))} tok/s "
                f"-> pi/04_fetch_model.sh --model {a.get('family')} --quant {a.get('quant')}")
    else:
        out("", "no viable row fits the free disk")
    out("", "or point MODEL_DIR at a USB 3 SSD / NVMe with more space (see the hardware line)")


def render_summary(feas_path, facts_path):
    rc = int(num(env("PG00_RC")) if num(env("PG00_RC")) is not None else 1)
    try:
        facts = load(facts_path)
    except (OSError, ValueError):
        facts = {}
    feas = None
    if feas_path and os.path.isfile(feas_path):
        try:
            feas = load(feas_path)
        except (OSError, ValueError):
            feas = None
    names = {0: "OK - a model fits in RAM (FITS_*) or streams well (STREAM_OK)",
             2: "SLOW ONLY - only slow streaming (STREAM_SLOW) or a slow dense run is possible",
             3: "NOT VIABLE - nothing viable on this hardware (or no disk for it)",
             4: "REFUSED - protected or busy host; the rows are informational only",
             1: "ERROR - pigemma.feasibility did not produce a verdict"}
    print("==== summary ====")
    out("verdict", f"exit {rc} = {names.get(rc, 'error')}")
    for n in [x for x in env("PG00_EXTRA_VERDICT_NOTES").split("\n") if x.strip()]:
        out("", n)
    if feas is None:
        out("recommended", "none (no verdict JSON; see the errors above)")
        out("hardware", env("PG00_FALLBACK_RECOMMENDATION"))
        return
    for r in feas.get("refusal_reasons") or []:
        out("", r)
    rec = feas.get("recommended")
    if rec:
        out("recommended", f"{rec.get('model_id')} ({rec.get('quant')}, {gb(rec.get('bytes'))}) {rec.get('verdict')}, "
                           f"projected {tok(rec.get('tok_s'))} tok/s [{src_label(rec.get('tok_s_source'))}]")
        flags = rec.get("flags") or ""
        out("llama flags", flags)
        eff = env("PG00_EFFECTIVE_FLAGS")
        if eff and eff != flags:
            out("on this host", f"{eff}   (mlock removed: {env('PG00_MLOCK_WHY')})")
        else:
            out("on this host", "the same flags" + (" (mlock allowed here)" if "mlock" in flags else ""))
        out("download", rec.get("url") or "n/a")
        d = rec.get("draft")
        if d:
            out("draft", f"{d.get('id')} ({d.get('spec_type')}, {gb(d.get('bytes'))}): optional; measure first "
                         f"(-md <file> --spec-type {d.get('spec_type')})")
        disk_lines(feas, facts, rec)
    elif rc == 4:
        out("recommended", "none: host refused (see above)")
    else:
        out("recommended", "none: no viable model/quant on this hardware (see the rows)")
    dev = feas.get("device") or {}
    out("device profile", f"{dev.get('name')}: {dev.get('match_reason')}")
    a = feas.get("assumptions") or {}

    def av(key, unit):
        e = a.get(key) or {}
        v = num(e.get("value"))
        return f"{key} {'n/a' if v is None else f'{v:g}'}{unit} [{kind_of(e.get('source'))}]"

    hit = feas.get("hit_source")
    out("projections", "all tok/s are PROJECTED. hit rates: " +
        ("sim.json LRU (simulated)" if hit == "simulated" else "assumed curve (catalog.selection)"))
    out("", ", ".join([av("mem_bw_gbps", " GB/s"), av("llama_efficiency", ""),
                       av("seq_read_mbps", " MB/s"), av("rand_read_3mb_mbps", " MB/s"), av("access_ms", " ms")]))
    inp = feas.get("inputs") or {}
    out("inputs", f"io_mem: {inp.get('io_mem') or 'none (catalog numbers)'}; calibration: "
                  f"{inp.get('calibration') or 'none (0.65 assumed)'}; sim: {inp.get('sim') or 'none'}")
    state = env("PG00_PROT_STATE")
    if rc == 4 and (feas.get("host_refused") or state == "refused"):
        nxt = ["every other pi/*.sh script refuses on this host (exit 4)",
               "override ONLY with the host owner's explicit authorisation: PIGEMMA_I_KNOW=1 bash pi/00_feasibility.sh"]
    elif rc == 4:
        nxt = ["host busy (loadavg1 / MemAvailable): heavy scripts refuse now; re-run pi/00_feasibility.sh when idle"]
    elif rc == 0:
        nxt = ["heavy steps may run: pi/01_tune_os.sh (dry run first), pi/02_build_llama.sh, pi/03_bench_io_mem.sh,",
               "pi/04_fetch_model.sh, pi/05_run.sh (each re-checks this verdict and the host before working)"]
    elif rc == 2:
        nxt = ["only slow operation: steps that accept it (pi/05_run.sh) run as measurements, not as a usable setup"]
    elif rc == 3:
        nxt = ["heavy steps refuse (exit 3): more RAM (16 GB) and/or USB 3 SSD / NVMe storage (hardware line below),",
               "or a different family / quant (--want, QUANT_OVERRIDE)"]
    else:
        nxt = ["fix the error above and re-run"]
    if hit != "simulated" or kind_of((a.get("rand_read_3mb_mbps") or {}).get("source")) != "measured":
        nxt.append("pi/03_bench_io_mem.sh + E4 (--sim results/h1/sim.json) replace assumed inputs; then re-run pi/00")
    if state == "overridden":
        nxt.append("override active: never stop/restart/kill its services; nice -n 10; no mlock; keep >= "
                   f"{env('KEEP_FREE_GB') or '2'} GB free on /; no config.txt edits / reboots without approval")
    for i, line in enumerate(nxt):
        out("next" if i == 0 else "", line)
    out("hardware", feas.get("recommendation_line") or env("PG00_FALLBACK_RECOMMENDATION"))


def main(argv):
    mode = argv[1] if len(argv) > 1 else ""
    if mode == "host" and len(argv) == 3:
        render_host(argv[2])
    elif mode == "protstate" and len(argv) == 3:
        render_protstate(argv[2])
    elif mode == "summary" and len(argv) == 4:
        render_summary(argv[2], argv[3])
    else:
        print(f"usage: {argv[0]} host FACTS | protstate FEAS | summary FEAS FACTS", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
}

_pg00_renderer >"$tmpd/render.py"

_pg00_py() { # _pg00_py MODE ARGS... -> run the renderer (cwd PIGEMMA_HOME, pigemma importable)
    local pp
    pp=$(native_path "$PIGEMMA_HOME")
    if [ -n "${_PG_MSYS:-}" ]; then
        pp="$pp${PYTHONPATH:+;$PYTHONPATH}"
    else
        pp="$pp${PYTHONPATH:+:$PYTHONPATH}"
    fi
    (cd "$PIGEMMA_HOME" && PYTHONPATH="$pp" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
        "$PY" "$(native_path "$tmpd/render.py")" "$@")
}

# ---------------------------------------------------------------------------- protected-host gate
_pg_protected_globs
PG00_GLOBS="${PG_GLOBS[*]}"
prot_state=no
prot_reasons=()
if [ "$facts_mode" = live ]; then
    if protected_host_reasons; then
        prot_reasons=("${PG_PROTECTED_REASONS[@]}")
        if _pg_truthy "${PIGEMMA_I_KNOW:-0}"; then prot_state=overridden; else prot_state=refused; fi
    fi
    case ${PG_SERVICES_SOURCE:-} in
        unavailable*) warn "running services could not be listed (${PG_SERVICES_SOURCE#unavailable: }): hostname and process names checked only" ;;
    esac
fi

# --------------------------------------------------------------------------------------- verdict
json_tmp="$tmpd/feasibility.json"
table="$tmpd/table.txt"
fargs=(--facts "$(native_path "$facts_file")" --catalog "$(native_path "$catalog")" --json "$(native_path "$json_tmp")")
shown=(--facts "$facts_file" --catalog "$catalog" --json "$json_final")
if [ -n "$want" ]; then fargs+=(--want "$want"); shown+=(--want "$want"); fi
if [ -n "$sim" ]; then fargs+=(--sim "$(native_path "$sim")"); shown+=(--sim "$sim"); fi
case $io_mem in
    "") ;;
    auto | none) fargs+=(--io-mem "$io_mem"); shown+=(--io-mem "$io_mem") ;;
    *) fargs+=(--io-mem "$(native_path "$io_mem")"); shown+=(--io-mem "$io_mem") ;;
esac
case $calibration in
    "") ;;
    auto | none) fargs+=(--calibration "$calibration"); shown+=(--calibration "$calibration") ;;
    *) fargs+=(--calibration "$(native_path "$calibration")"); shown+=(--calibration "$calibration") ;;
esac
if [ -n "$device" ]; then fargs+=(--device "$device"); shown+=(--device "$device"); fi
fargs+=("${extra[@]}")
shown+=("${extra[@]}")

log "running: python3 -m pigemma.feasibility ${shown[*]}"
feas_rc=0
run_feasibility "${fargs[@]}" >"$table" || feas_rc=$?
rc=$feas_rc
case $rc in
    0 | 2 | 3 | 4) ;;
    *)
        warn "pigemma.feasibility failed (exit $rc): no verdict"
        rc=1
        ;;
esac

busy=0
busy_reasons=()
extra_notes=()
if [ "$rc" != 1 ] && [ -s "$json_tmp" ]; then
    ps_out=$(_pg00_py protstate "$(native_path "$json_tmp")") || ps_out=""
    j_state=no
    j_reasons=()
    while IFS= read -r _l || [ -n "$_l" ]; do
        _l=${_l%$'\r'}
        case $_l in
            "STATE "*) j_state=${_l#STATE } ;;
            "BUSY "*) busy=${_l#BUSY } ;;
            "R "*) j_reasons+=("${_l#R }") ;;
            "B "*) busy_reasons+=("${_l#B }") ;;
        esac
    done <<<"$ps_out"
    if [ "$facts_mode" = file ]; then
        prot_state=$j_state
        prot_reasons=("${j_reasons[@]}")
    elif [ "$prot_state" = refused ] && [ "$rc" != 4 ]; then
        # the bash gate (used by every other pi script) is a superset of pigemma.feasibility's glob rule
        extra_notes+=("raised to exit 4: the pi/lib/common.sh host gate refuses this host although pigemma.feasibility did not")
        rc=4
    fi
    mkdir -p "$(dirname "$json_final")" 2>/dev/null || true
    if [ "$json_final" != - ]; then
        mv -f "$json_tmp" "$json_final" || die 1 "cannot write $json_final"
        json_show=$json_final
    else
        json_show="$json_tmp"
    fi
else
    json_show=""
    if [ -f "$json_final" ] && [ "$json_final" != - ]; then
        warn "kept the previous $json_final unchanged (it is NOT from this run)"
    fi
fi

# Flags as they will really be used on this host (mlock policy of pi/lib/common.sh).
eff_flags=""
mlock_why=""
if [ -n "$json_show" ]; then
    rec_flags=$(json_get "$json_show" llama_flags 2>/dev/null || true)
    rec_flags=${rec_flags%$'\r'}
    if [ -n "$rec_flags" ]; then
        case ${ALLOW_MLOCK:-auto} in
            auto)
                if [ "$prot_state" = no ]; then _m=1; else _m=0; fi
                mlock_why="ALLOW_MLOCK=auto on a protected host: the page cache must stay reclaimable for its services"
                ;;
            1) _m=1 ;;
            *)
                _m=0
                mlock_why="ALLOW_MLOCK=${ALLOW_MLOCK}"
                ;;
        esac
        eff_flags=$(ALLOW_MLOCK=$_m safe_llama_flags "$rec_flags" 2>/dev/null) || eff_flags=$rec_flags
    fi
fi

# ---------------------------------------------------------------------------------------- report
host_label=$(host_name)
if [ "$facts_mode" = file ]; then
    host_label=$(json_get "$facts_file" hostname 2>/dev/null || true)
    host_label=${host_label%$'\r'}
fi

_pg00_box() {
    local bar l
    bar=$(printf '#%.0s' {1..100})
    printf '%s\n' "$bar"
    for l in "$@"; do
        printf '#  %s\n' "$l"
    done
    printf '%s\n\n' "$bar"
}

_pg00_banner() {
    local where="THIS HOST" r lines=()
    if [ "$facts_mode" = file ]; then where="THE HOST IN $facts_file"; fi
    case $prot_state in
        refused)
            lines+=("PROTECTED HOST '${host_label:-?}': this report is READ-ONLY - nothing was changed.")
            for r in "${prot_reasons[@]}"; do lines+=("  - $r"); done
            lines+=("EVERY OTHER pi/*.sh SCRIPT REFUSES ON $where (exit 4): 01_tune_os, 02_build_llama,"
                "03_bench_io_mem, 04_fetch_model, 05_run, 06_pin_hotset. The verdict rows are informational only."
                "Override ONLY with the host owner's explicit authorisation: PIGEMMA_I_KNOW=1")
            _pg00_box "${lines[@]}"
            ;;
        overridden)
            lines+=("PROTECTED HOST '${host_label:-?}' - OVERRIDE ACTIVE (PIGEMMA_I_KNOW=1)")
            for r in "${prot_reasons[@]}"; do lines+=("  - $r"); done
            lines+=("Rules still in force: never stop/restart/kill its services; nice -n ${NICE_LEVEL:-10} (downloads also"
                "ionice -c3); no mlock of model weights (ALLOW_MLOCK=${ALLOW_MLOCK:-auto}); keep >= ${KEEP_FREE_GB:-2} GB free on /;"
                "no config.txt edits or reboots without explicit approval; results must say PIGEMMA_I_KNOW.")
            _pg00_box "${lines[@]}"
            ;;
    esac
    if [ "$busy" = 1 ]; then
        lines=("BUSY HOST '${host_label:-?}': heavy scripts refuse until it is idle (PIGEMMA_I_KNOW does not override this)")
        for r in "${busy_reasons[@]}"; do lines+=("  - $r"); done
        _pg00_box "${lines[@]}"
    fi
    return 0
}

emit_report() {
    local now
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)
    printf 'pi-gemma4 feasibility report - pi/00_feasibility.sh %s - %s - READ-ONLY (nothing on this host was changed)\n' \
        "$PG00_VERSION" "${now:-?}"
    if is_emulated; then
        printf 'EMULATED run (PIGEMMA_EMULATED=1): speeds are projections for the device profile, NOT Raspberry Pi measurements\n'
    fi
    if [ "$facts_mode" = file ]; then
        printf 'facts from %s (not collected on this machine)\n' "$facts_file"
    fi
    printf '\n'
    _pg00_py host "$(native_path "$facts_file")" || printf '  (host facts could not be rendered; see %s)\n' "$facts_file"
    printf '\n'
    _pg00_banner
    printf '==== verdict table: python3 -m pigemma.feasibility (every tok/s is PROJECTED) ====\n'
    printf '  command: python3 -m pigemma.feasibility %s\n' "${shown[*]}"
    if [ -s "$table" ]; then
        tr -d '\r' <"$table"
    else
        printf '  (no table: pigemma.feasibility exit %s, see stderr)\n' "$feas_rc"
    fi
    printf '\n'
    _pg00_py summary "$(native_path "${json_show:-$tmpd/none.json}")" "$(native_path "$facts_file")" ||
        printf '  (summary could not be rendered; see %s)\n' "${json_show:-stderr}"
    local outs="facts $facts_file | verdict $json_display"
    if [ -n "$report" ]; then outs+=" | report $report"; fi
    printf '  %-15s %s\n' outputs "$outs"
}

PG00_RC=$rc
PG00_FACTS_MODE=$facts_mode
PG00_PROT_STATE=$prot_state
PG00_PROT_REASONS=$(printf '%s\n' "${prot_reasons[@]}")
PG00_EXTRA_VERDICT_NOTES=$(printf '%s\n' "${extra_notes[@]}")
PG00_EFFECTIVE_FLAGS=$eff_flags
PG00_MLOCK_WHY=$mlock_why
json_display=$json_final
if [ "$json_final" = - ]; then json_display="stdout"; fi
if [ -z "$json_show" ]; then json_display="(not written: no verdict)"; fi
export PG00_RC PG00_FACTS_MODE PG00_PROT_STATE PG00_PROT_REASONS PG00_EXTRA_VERDICT_NOTES PG00_EFFECTIVE_FLAGS \
    PG00_MLOCK_WHY PG00_GLOBS PG00_FALLBACK_RECOMMENDATION \
    MIN_FREE_DISK_FACTOR KEEP_FREE_GB

emit_report >"$tmpd/report.txt"

# A closed stdout (e.g. `| head`) must not stop the report copy or change the exit code.
if [ "$json_final" = - ]; then
    cat "$tmpd/report.txt" >&2 || true
    if [ -n "$json_show" ]; then
        tr -d '\r' <"$json_show" || true
    fi
else
    cat "$tmpd/report.txt" || true
fi
if [ -n "$report" ]; then
    mkdir -p "$(dirname "$report")" 2>/dev/null || true
    if cp "$tmpd/report.txt" "$report.tmp.$$" 2>/dev/null && mv -f "$report.tmp.$$" "$report"; then
        log "report written: $report"
    else
        rm -f "$report.tmp.$$" 2>/dev/null || true
        warn "cannot write the report copy $report"
    fi
fi

case $rc in
    0) log "verdict: exit 0 (OK) -> heavier pi scripts may run" ;;
    2) warn "verdict: exit 2 (only slow operation possible)" ;;
    3) warn "verdict: exit 3 (nothing viable on this host)" ;;
    4) warn "verdict: exit 4 (REFUSED: protected or busy host) - every other pi/*.sh script refuses here" ;;
    *) warn "verdict: exit $rc (error)" ;;
esac
exit "$rc"
