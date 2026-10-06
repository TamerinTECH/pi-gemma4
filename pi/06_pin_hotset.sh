#!/usr/bin/env bash
# pi/06_pin_hotset.sh - hot-set pinning on a Raspberry Pi 5: plan the hot byte ranges of a GGUF from the
# desktop routing statistics, keep them mlocked with tools/pinner, and A/B-test page-cache streaming with
# and without the pinner. Part of pi-gemma4 (docs/design/manifest.json, experiment E6 repeated on a Pi).
#
# Usage (the first line is the manifest's usage string; docs/PI_RUNBOOK.md greps for it):
#   pi/06_pin_hotset.sh start|stop|status|ab [--model <gguf>] [--map results/map26.json] [--stats traces/26b/all.stats.json] [--budget-bytes N]
#   pi/06_pin_hotset.sh --help            (full option list)
#
# Subcommands:
#   start   assert_safe_host -> mlock permission -> no live pinner -> bench lock not held by another job ->
#           model -> map (results/map26.json when it describes this GGUF, else built on the Pi from the
#           local GGUF header: python3 -m pigemma.gguf_map build) -> stats (traces/26b/all.stats.json;
#           without stats only the non-expert tensors are pinned) -> budget -> memlock / RAM checks ->
#             python3 -m pigemma.gguf_map hotset --map $MAP --stats $STATS --budget-bytes $B \
#                 --page-size $(getconf PAGESIZE) --out pi/out/hotset.json
#           -> nohup nice -n $NICE_LEVEL tools/build/bin/pinner --gguf $MODEL --ranges pi/out/hotset.ranges.txt \
#                 --verify --json --pidfile pi/out/pinner.pid > pi/out/pinner.log 2> pi/out/pinner.stderr.log &
#           -> waits for the pinner's one-line JSON summary, writes pi/out/pin_state.json and prints the
#           locked / resident bytes and the expert coverage. The pinner keeps running (daemon) until `stop`,
#           or until --max-hold S seconds have passed (pinner --hold-seconds; 7200 s by default on a protected
#           host run with the owner's override, so a bounded experiment there always expires by itself).
#   stop    SIGTERM to the pid in pi/out/pinner.pid ONLY (and only when that process is a pinner), waits for
#           it to exit (the kernel releases the locks), updates pi/out/pin_state.json. Nothing else is ever
#           signalled. No pinner running = exit 0 (idempotent; pi/05_run.sh --pin calls it at exit).
#   status  VmLck of the pidfile's process + the hot set's page-cache residency (mincore) via
#           `pinner --gguf G --ranges R --verify-only --json` (G and R from pi/out/pin_state.json).
#           Exit 0 when a pinner is running, 3 when none is (LSB "program is not running").
#   ab      A/B: plans the hot set once, then per round runs
#             pi/05_run.sh stream-bench --model M --no-calibrate --no-he            (unpinned)
#             pi/05_run.sh stream-bench --model M --no-calibrate --no-he --pin \
#                 --pin-arg=--hotset --pin-arg=pi/out/hotset.json ...                  (pinned)
#           i.e. pi/05 itself starts this script (`start --hotset ...`) after its own gates and stops the
#           pinner at exit, so "start, stream-bench, stop" happens inside the pinned pi/05 run. The hot set's
#           mincore residency is measured before and after every run (pinner --verify-only). Writes
#           results/pi/pin_ab.json (results/pi-emu/ when PIGEMMA_EMULATED=1) plus a timestamped copy
#           pin_ab_<UTC>.json, with both runs' tok/s and read_bytes_per_token and the ratios. The whole A/B
#           holds the bench lock (python3 -m pigemma.paths lock --hold) so no other model job interleaves.
#
# Budget (bytes of page-aligned hot set; non-expert tensors are part of it and always come first):
#   --budget-bytes N > PIN_BUDGET_BYTES (pi/pi_config.env, unless 'auto') > auto:
#   auto = MemAvailable - KV(ctx) - 0.3e9 compute - 0.5e9 headroom, capped at the file size, where
#   MemAvailable is the effective value of pi/lib/common.sh (cgroup aware), KV(ctx) = map.kv_bytes(CTX,
#   --kv-type q8_0) (the KV type of every pigemma.feasibility flag set), the compute and headroom terms
#   are ASSUMED values (--compute-bytes, --headroom-bytes). N accepts 5200000000, 5.2e9, 5.2GB or 2GiB.
#   The page cache left for streaming the cold experts is therefore ~headroom; that is the design.
#
# Outputs (PIGEMMA_OUT = pi/out by default):
#   pi/out/hotset.json + pi/out/hotset.ranges.txt    the plan (pigemma.gguf_map hotset; TRACE_FORMAT 9)
#   pi/out/map_<gguf stem>.json                      map built on the Pi when results/map26.json does not
#                                                    describe the local GGUF (e.g. a UD-Q3_K_M file)
#   pi/out/pinner.pid, pinner.log (one JSON line), pinner.stderr.log (notes, progress, hints)
#   pi/out/pin_state.json   {format_version, generator, version, created_at, updated_at, state
#                           (running|exited|timeout|stopped|gone), host, emulated, protected_override, pid,
#                           pidfile, log, stderr_log, pinner, argv, exit_code, model{id,path,bytes}, map,
#                           stats, hotset, ranges, plan (planned|given), page_size, max_hold_s, budget{bytes, source,
#                           formula, mem_available_bytes, kv_bytes, kv_source, kv_type, ctx, compute_bytes,
#                           headroom_bytes}, hotset_info{...}, summary (the pinner's JSON line), notes[],
#                           stopped_at, stop_wait_s, vm_lck_bytes_at_stop}
#   pi/out/pin_ab_<UTC>/    ab only: per-run copies of pi/out/last_stream-bench.json, pin_state.json and
#                           the --verify-only JSON before / after every run
#   results/pi/pin_ab.json  ab only (NOT a results envelope: no 'experiment' key, so pigemma.results
#                           summarize skips it; the per-run E8 envelopes of pi/05 are the envelopes):
#                           {format_version, generator, version, created_at, kind:'measured',
#                           experiment_ref, host, emulated, protected_override, model{...}, hotset{...},
#                           rounds, order[], runs[{label, round, exit_code, failed, started_at, ended_at,
#                           result_files[], tg_tok_s, pp_tok_s, read_bytes_per_token, majflt_per_token,
#                           load_s, coherent, mode, effective_flags, pinned_by_pi05, pinner{...}|null,
#                           hot_set_requested_bytes, hot_set_resident_before_bytes,
#                           hot_set_resident_after_bytes, mincore_reliable, last_run_copy}], unpinned{n,
#                           tg_tok_s, read_bytes_per_token, majflt_per_token}, pinned{...}, ratio{
#                           tok_s_pinned_over_unpinned, read_bytes_per_token_pinned_over_unpinned,
#                           bytes_per_token_reduction, majflt_per_token_pinned_over_unpinned},
#                           bytes_per_token_worst, in_ram, threshold_check{rule, tok_s_ratio_min,
#                           bytes_reduction_min, result PASS|FAIL|INCONCLUSIVE, reason}, notes[]}
#   stdout: start = the human summary (--json: pin_state.json); status = the residency summary (--json:
#   JSON); stop = one line; ab = the written files, one path per line (pi/05 envelopes, then pin_ab.json).
#   Everything else (logs, progress, the A/B table) goes to stderr.
#
# Exit codes (pi/lib/common.sh convention): 0 ok | 1 usage / internal error / a run failed / the pinner
# failed or timed out | 2 start: partial pin (some ranges not locked; the pinner holds the rest; pi/05
# accepts it) | 3 precondition failed (model / pinner / map / stats / python missing, memlock limit too
# small, not enough RAM for the hot set, budget smaller than the non-expert set) and status: no pinner
# running | 4 refused (protected host without PIGEMMA_I_KNOW=1, mlock not allowed on this host, a pinner
# already running, bench lock held by another job, the pinner's own protected-host guard).
#
# Safety (docs/design/FACTS.md): mlock of model weights is refused wherever pi/lib/common.sh
# mlock_allowed says no (every protected host, e.g. context-friend-1 with its ContextFriend services, even
# with PIGEMMA_I_KNOW=1, unless ALLOW_MLOCK=1 AND --allow-protected-host are both given explicitly by the
# owner). stop/cleanup signal only the pid of pi/out/pinner.pid after checking that it is a pinner;
# nothing else is stopped, restarted or killed. No sudo, no apt, no config edits, no reboots. The pinner
# runs under nice -n $NICE_LEVEL. Under PIGEMMA_FAKE_PROC (tests) start/ab run only with --dry-run and
# stop never sends a signal.
#
# Design decisions (also reported as open issues):
#   * `ab` runs the pinned stream-bench through pi/05's own --pin instead of `start; pi/05 stream-bench;
#     stop`: pi/05's assert_not_busy refuses when MemAvailable < 1 GB, which is exactly the state the auto
#     budget leaves behind once the pinner holds its pages (MemAvailable - KV - compute - headroom); with
#     --pin pi/05 runs its gates first, then starts this script, and stops the pinner at exit.
#   * The hot set is planned ONCE per A/B (explicit byte budget) and handed to the pinned run with
#     --hotset, so every round pins the same ranges.
#   * status uses `pinner --verify-only` rather than the manifest's `pinner --verify --once`: a second
#     locking pass while the daemon runs would be refused by the pinner's memory guard (the daemon's pages
#     are no longer "available"); --verify-only maps the file and reports mincore residency only.
#   * memlock: the manifest says "requires ulimit -l unlimited"; a finite limit that is >= the hot set, or
#     an effective CAP_IPC_LOCK, is also accepted (with a note); --best-effort lets the pinner lock in file
#     order until the limit.
#   * A --map that exists but describes another GGUF (basename or size) is not used: a map of the local
#     GGUF is built into pi/out/map_<stem>.json (or into the --map path when that file does not exist).
#     The default stats file is optional (missing or inconsistent -> non-expert-only pinning, with a
#     warning); an explicit --stats must exist (exit 3) and fit the map (exit 1).
#   * stop and status do not refuse on protected hosts: they lock nothing and stop releases memory.
#   * Split GGUFs (llama-gguf-split parts, e.g. the QAT Q4_0 26B on the Pi's FAT32 USB stick) are refused
#     (exit 3): the map, the hot set and the pinner all address byte offsets of ONE file.
#   * The unpinned run goes first in odd rounds and second in even rounds (--rounds 2 = ABBA): the first
#     run of a round warms the page cache for the second; pin_ab.json says so.
#   * No memory balloon is started here. A board whose RAM holds the whole model shows no pinning gain
#     (pin_ab.json then says INCONCLUSIVE, in_ram true). For the decisive "pinning vs page cache at the
#     same budget" test on a 16 GB Pi, start the bounded balloon first (sudo python3 pi/adhoc/balloon.py
#     <GiB> <max_s>, self-expiring); the auto budget is computed from the MemAvailable left after it.
#
# Lint: bash -n pi/06_pin_hotset.sh (shellcheck when installed). Tests: tests/test_pi06_pin_hotset.py.

set -euo pipefail

PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIGEMMA_SCRIPT="06_pin_hotset.sh"
# shellcheck source=lib/common.sh
. "$PI_DIR/lib/common.sh"

PI06_VERSION="1.0.0"
PI06_USAGE_LINE="pi/06_pin_hotset.sh start|stop|status|ab [--model <gguf>] [--map results/map26.json] [--stats traces/26b/all.stats.json] [--budget-bytes N]"
DEFAULT_MAP_REL="results/map26.json"
DEFAULT_STATS_REL="traces/26b/all.stats.json"
DEF_COMPUTE_BYTES=300000000
DEF_HEADROOM_BYTES=500000000
PINNER_GUARD_BYTES=536870912 # the pinner refuses when requested + 512 MiB > MemAvailable
KV_TYPES="f32 f16 bf16 q8_0 q5_1 q5_0 q4_1 q4_0 iq4_nl"

usage() {
    cat <<EOF
pi/06_pin_hotset.sh $PI06_VERSION - pin the hot set of a GGUF (non-expert tensors + hottest experts) with
tools/pinner on a Raspberry Pi 5, and A/B-test streaming with and without it.

Usage:
  $PI06_USAGE_LINE

Subcommands:
  start    plan the hot set (python3 -m pigemma.gguf_map hotset) and start the pinner daemon
           (nohup tools/build/bin/pinner --gguf MODEL --ranges pi/out/hotset.ranges.txt --verify --json
           --pidfile pi/out/pinner.pid > pi/out/pinner.log &); waits for its summary and prints the locked /
           resident bytes and the expert coverage
  stop     SIGTERM to the pid in pi/out/pinner.pid only (when it is a pinner); waits for it to exit
  status   VmLck of the running pinner + the hot set's page-cache residency (pinner --verify-only);
           exit 3 when no pinner runs
  ab       pi/05_run.sh stream-bench unpinned, then pinned (pi/05 --pin starts and stops the pinner),
           -> results/pi/pin_ab.json with both tok/s and read_bytes_per_token and the ratios

Options (start, ab; most also apply to status):
  --model GGUF|ID        model file or catalog id (default: the recommended model of pi/out/feasibility.json
                         in MODEL_DIR)
  --map FILE             map.json of the model (default $DEFAULT_MAP_REL; a map that describes another
                         GGUF is replaced by one built from the local file into pi/out/map_<stem>.json; a
                         --map path that does not exist is built there)
  --stats FILE           routing statistics (python -m pigemma.trace stats; default $DEFAULT_STATS_REL,
                         optional: without it only the non-expert tensors are pinned)
  --no-stats             pin the non-expert tensors only
  --budget-bytes N       hot-set budget (5200000000, 5.2e9, 5.2GB, 2GiB); default PIN_BUDGET_BYTES, auto =
                         MemAvailable - KV(ctx) - compute - headroom, capped at the file size
  -c, --ctx N            context size for the KV term (default CTX=$CTX)
  --kv-type T            KV cache type for the KV term (default q8_0; one of: $KV_TYPES)
  --compute-bytes N      compute-buffer term (default 0.3e9, assumed)
  --headroom-bytes N     headroom term (default 0.5e9, assumed)
  --rank count|weight    expert ranking of gguf_map hotset (default count)
  --per-layer-min N      pin the N hottest experts of every layer first (default 0)
  --hotset FILE          start: use this hotset.json (and its .ranges.txt) instead of planning
  --pinner PATH          pinner binary (default: PIGEMMA_TOOLS_BIN or tools/build/bin/pinner)
  --start-timeout S      seconds to wait for the pinner's summary (default 1800; the pinner is stopped on
                         timeout)
  --stop-timeout S       seconds to wait for the pinner to exit after SIGTERM (default 60)
  --best-effort          let the pinner lock in file order until RLIMIT_MEMLOCK (partial pin, exit 2)
  --max-hold S           the pinner releases its locks and exits by itself after S seconds (pinner
                         --hold-seconds; default: until 'stop'; 7200 on a protected host run with the override)
  --allow-protected-host forwarded to the pinner; only with the host owner's explicit approval, and only
                         effective together with ALLOW_MLOCK=1 (never passed automatically)
  --json                 start / status: print JSON on stdout instead of the text summary
  --dry-run              run the gates, print the plan and the exact commands; start no pinner, write nothing
  -h, --help             this text;  --version

ab options:
  --rounds N             A/B rounds (default 1); odd rounds run unpinned first, even rounds pinned first
  -n N                   tokens to generate per run (pi/05 -n; default pi/05's 64)
  --with-he              keep pi/05's Hebrew sample (default: --no-he, one model load per run)
  --results-dir DIR      pi/05 --out (default results/pi; results/pi-emu when PIGEMMA_EMULATED=1)
  --out FILE             the A/B summary (default <results-dir>/pin_ab.json)
  --run-arg ARG          extra pi/05_run.sh argument (repeatable, e.g. --run-arg=--want --run-arg=26b);
                         everything after a bare -- is passed to pi/05 as well
  --no-lock              do not hold the bench lock across the whole A/B (each pi/05 run still takes it)

Examples:
  pi/06_pin_hotset.sh start --model models/gemma-4-26B_q4_0-it.gguf --stats traces/26b/all.stats.json
  pi/06_pin_hotset.sh status
  pi/06_pin_hotset.sh stop
  pi/06_pin_hotset.sh ab --model 26b-q4 --budget-bytes 5.2e9 --rounds 2
  pi/06_pin_hotset.sh start --dry-run --model 26b-ud-q3_k_m
  # memory pressure on a 16 GB board: emulate a smaller one with the bounded balloon first, then
  sudo python3 pi/adhoc/balloon.py 8 3600 &  ;  pi/06_pin_hotset.sh ab --model 26b-ud-q3_k_m

Exit codes: 0 ok, 1 usage / internal error / run or pinner failure, 2 partial pin (start), 3 precondition
failed (memlock, RAM, missing input; status: no pinner running), 4 refused (protected host, mlock not
allowed, pinner already running, bench lock held).
Safety: no mlock where pi/lib/common.sh mlock_allowed says no; stop signals only pi/out/pinner.pid.
EOF
}

# ================================================================================== small helpers
_p06_abspath() { # path relative to the caller's cwd -> absolute
    case $1 in
        /* | [A-Za-z]:[/\\]*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$PWD/$1" ;;
    esac
}

_p06_pp() { # PYTHONPATH value with the repo first
    local pp
    pp=$(native_path "$PIGEMMA_HOME")
    if [ -n "${_PG_MSYS:-}" ]; then
        printf '%s\n' "$pp${PYTHONPATH:+;$PYTHONPATH}"
    else
        printf '%s\n' "$pp${PYTHONPATH:+:$PYTHONPATH}"
    fi
}

_p06_pyrun() { # _p06_pyrun ARGS... -> "$PY" ARGS from the repo root with PYTHONPATH set
    local pp
    pp=$(_p06_pp)
    (cd "$PIGEMMA_HOME" && PYTHONPATH="$pp" PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "$PY" "$@")
}

_p06_join_cmd() { # printable shell command line
    local out="" a
    for a in "$@"; do
        out+="${out:+ }$(printf '%q' "$a")"
    done
    printf '%s\n' "$out"
}

_p06_need_uint() { # _p06_need_uint NAME VALUE [MIN]
    if ! _pg_is_uint "$2" || [ "$2" -lt "${3:-0}" ]; then
        die 1 "$1 must be an integer >= ${3:-0}, got '$2' (see --help)"
    fi
}

_p06_stamp() { date -u +%Y%m%dT%H%M%SZ; }

_p06_fmt() { # _p06_fmt BYTES -> "14.44 GB" (>= 1e9) or "3.9 MB"; "n/a" for non-numbers
    if [[ ${1:-} =~ ^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
        awk -v b="$1" 'BEGIN { if (b + 0 < 1e9 && b + 0 > -1e9) printf "%.1f MB\n", b / 1e6; else printf "%.2f GB\n", b / 1e9 }'
    else
        printf 'n/a\n'
    fi
}

# ============================================================================= python helper (stdlib)
# Commands: resolve-model | check-map | check-stats | budget | hotset-info | pinner-summary | state-write |
#           state-update | report-start | status-report | ab-report
_p06_py() {
    local pp
    pp=$(_p06_pp)
    (
        cd "$PIGEMMA_HOME" || exit 1
        PYTHONPATH="$pp" PYTHONUTF8=1 PYTHONIOENCODING=utf-8 exec "$PY" - "$@" <<'PIGEMMA_PY06_EOF'
"""pi/06_pin_hotset.sh helper: model/map/stats checks, budget, pinner summaries, state file, A/B report.

stdlib only (pigemma.gguf_map is imported for map/KV helpers; it never imports numpy or gguf for these
calls). KEY='value' lines (shlex-quoted) are printed for `eval` in bash; JSON / text where documented.
Exit codes: 0 ok, 1 error, 3 missing input / precondition (raised as Fail(code, message)).
"""
import argparse
import datetime as _dt
import json
import math
import os
import shlex
import sys
import tempfile
from pathlib import Path

REPO = Path.cwd()
PINNER_GUARD = 512 * 1024 * 1024
E6_TOK_S_RATIO = 1.15
E6_BYTES_REDUCTION = 0.20
IN_RAM_FRACTION = 0.01
KV_REF_CTX = 4096
KV_PER_32 = {"f32": 128, "f16": 64, "bf16": 64, "q8_0": 34, "q5_1": 24, "q5_0": 22, "q4_1": 20,
             "q4_0": 18, "iq4_nl": 18}
SIZE_SUFFIXES = (("tib", 1 << 40), ("gib", 1 << 30), ("mib", 1 << 20), ("kib", 1 << 10), ("tb", 10**12),
                 ("gb", 10**9), ("mb", 10**6), ("kb", 10**3), ("b", 1))
HOTSET_INFO_KEYS = ("budget_bytes", "total_bytes", "n_ranges", "n_experts", "n_expert_items_total",
                    "expert_coverage_fraction", "expert_weight_coverage_fraction", "nonexpert_included",
                    "nonexpert_bytes_aligned", "page_size", "page_overhead_bytes", "rank", "stats", "map",
                    "map_sha256", "gguf_basename", "gguf_size", "bytes_by_kind")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace", newline="\n")
    except Exception:  # noqa: BLE001
        pass


class Fail(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def say(msg):
    sys.stderr.write(f"pi/06: {msg}\n")


def q(v):
    return shlex.quote("" if v is None else str(v))


def emit(name, value):
    if isinstance(value, bool):
        value = 1 if value else 0
    print(f"{name}={q(value)}")


def load_json(p):
    with open(p, encoding="utf-8-sig") as fh:
        return json.load(fh)


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def dumps(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def posix(p):
    return Path(p).as_posix() if p not in (None, "") else None


def rel(p):
    if p in (None, ""):
        return None
    pp = Path(p)
    if not pp.is_absolute():
        pp = REPO / pp
    try:
        return pp.resolve().relative_to(REPO.resolve()).as_posix()
    except (ValueError, OSError):
        return pp.as_posix()


def resolve(p):
    pp = Path(p)
    return pp if pp.is_absolute() else REPO / pp


def int_or_none(x):
    if x in (None, "") or isinstance(x, bool):
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def num(x):
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, dict):
        x = x.get("value")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def gb(x, d=2):
    """Bytes as '14.44 GB' (>= 1e9) or '3.9 MB' (1 GB = 1e9 B, 1 MB = 1e6 B)."""
    v = num(x)
    if v is None:
        return "n/a"
    if abs(v) < 1e9:
        return f"{v / 1e6:.1f} MB"
    return f"{v / 1e9:.{d}f} GB"


def pct(x, d=1):
    v = num(x)
    return "n/a" if v is None else f"{100.0 * v:.{d}f}%"


def parse_bytes(s):
    """5200000000, 5.2e9, 5_200_000_000, 5.2GB, 2GiB, 512MiB -> int bytes (GB = 1e9, GiB = 2^30)."""
    t = str(s).strip().replace("_", "").replace(",", "")
    mult = 1
    low = t.lower()
    for suf, mu in SIZE_SUFFIXES:
        if low.endswith(suf) and len(low) > len(suf):
            t, mult = t[: -len(suf)].strip(), mu
            break
    try:
        v = float(t)
    except ValueError:
        raise Fail(1, f"not a byte count: {s!r} (examples: 5200000000, 5.2e9, 5.2GB, 2GiB)") from None
    if not math.isfinite(v) or v < 0:
        raise Fail(1, f"a byte count must be finite and >= 0: {s!r}")
    return int(v * mult)


def last_pinner_json(path):
    """The last complete line of FILE that is a pinner JSON summary, or None."""
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for ln in reversed(text.split("\n")[:-1]):  # complete lines only (a partial last line is ignored)
        s = ln.strip()
        if not (s.startswith("{") and s.endswith("}")):
            continue
        try:
            d = json.loads(s)
        except ValueError:
            continue
        if isinstance(d, dict) and ("locked_bytes" in d or d.get("tool") == "pinner"):
            return d
    return None


# ------------------------------------------------------------------------------------- model
def _catalog_models(path):
    try:
        cat = load_json(path)
    except (OSError, ValueError):
        return []
    return [m for m in cat.get("models", []) or [] if isinstance(m, dict)]


def _candidates(entry, model_dir):
    out = []
    f = str(entry.get("file") or "")
    if model_dir and f:
        d = Path(model_dir)
        out += [d / f, d / Path(f).name]
        repo = str(entry.get("repo") or "")
        if "/" in repo:
            org, name = repo.split("/", 1)
            out.append(d / f"{org}__{name}" / f)
    if entry.get("local_path"):
        out.append(REPO / str(entry["local_path"]))
    seen, uniq = set(), []
    for c in out:
        k = c.as_posix()
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def cmd_resolve_model(a):
    models = _catalog_models(a.catalog) if a.catalog else []
    spec = (a.model or "").strip()
    notes = []
    if not spec:
        feas = None
        if a.feasibility and Path(a.feasibility).is_file():
            try:
                feas = load_json(a.feasibility)
            except (OSError, ValueError):
                feas = None
        rec = (feas or {}).get("recommended") or {}
        spec = str(rec.get("model_id") or "")
        if not spec:
            raise Fail(1, "no --model given and no recommended model in pi/out/feasibility.json (run "
                          "pi/00_feasibility.sh, or pass --model <gguf path | catalog id>)")
        notes.append(f"model from the feasibility recommendation: {spec}")
    looks_path = any(c in spec for c in "/\\") or spec.lower().endswith(".gguf")
    entry = None
    if looks_path:
        path = Path(spec)
        name = path.name
        hits = [m for m in models if Path(str(m.get("file") or "")).name == name]
        hits.sort(key=lambda m: m.get("kind") == "draft")
        entry = hits[0] if hits else None
    else:
        entry = next((m for m in models if m.get("id") == spec), None)
        if entry is None:
            ids = ", ".join(sorted(str(m.get("id")) for m in models if m.get("kind") != "draft"))
            raise Fail(1, f"unknown model id '{spec}' (catalog ids: {ids}); or pass a GGUF path")
        cands = _candidates(entry, a.model_dir)
        if not cands:
            raise Fail(1, f"catalog entry '{spec}' has no file name")
        path = next((c for c in cands if c.is_file()), cands[0])
    if entry is not None and entry.get("kind") == "draft":
        raise Fail(1, f"{path.name} is a speculative-decoding draft sidecar ({entry.get('id')}), not a model "
                      "to pin")
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    cat_bytes = int_or_none((entry or {}).get("bytes"))
    mismatch = bool(exists and cat_bytes and size != cat_bytes and (entry or {}).get("bytes_exact", True))
    if entry is None:
        notes.append(f"{path.name} is not in the catalog (uncatalogued GGUF)")
    emit("MODEL_PATH", path.as_posix())
    emit("MODEL_BASENAME", path.name)
    emit("MODEL_STEM", path.name[:-5] if path.name.lower().endswith(".gguf") else path.name)
    emit("MODEL_ID", (entry or {}).get("id") or "")
    emit("MODEL_KIND", (entry or {}).get("kind") or "")
    emit("MODEL_FAMILY", (entry or {}).get("family") or "")
    emit("MODEL_QUANT", (entry or {}).get("quant") or "")
    emit("MODEL_CATALOG_BYTES", cat_bytes if cat_bytes is not None else "")
    emit("MODEL_EXISTS", exists)
    emit("MODEL_SIZE", size if size is not None else "")
    emit("MODEL_FILE_BYTES", size if size is not None else (cat_bytes if cat_bytes is not None else ""))
    emit("MODEL_SIZE_MISMATCH", mismatch)
    kv = int_or_none((entry or {}).get("kv_bytes_q8_at_4096"))
    if kv is None:
        kv = {"26b": 160_000_000, "31b": 420_000_000}.get(str((entry or {}).get("family") or "").lower(),
                                                          420_000_000)
    emit("MODEL_KV_Q8_4K", kv)
    emit("MODEL_NOTES", " | ".join(notes))
    return 0


# ---------------------------------------------------------------------------------- map / stats
def _map_doc(path):
    p = Path(path)
    if not p.is_file():
        return None, "does not exist"
    try:
        m = load_json(p)
    except (OSError, ValueError) as e:
        return None, f"unreadable ({e})"
    if not isinstance(m, dict) or "tensors" not in m or "file_size" not in m:
        return None, "not a pigemma map.json"
    return m, ""


def cmd_check_map(a):
    m, reason = _map_doc(a.map)
    gname = Path(a.gguf).name
    exp = int_or_none(a.expect_size)
    match = False
    if m is not None:
        bn = m.get("gguf_basename") or Path(str(m.get("gguf") or "")).name
        fs = int_or_none(m.get("file_size"))
        if bn != gname:
            reason = f"built for {bn}, not {gname}"
        elif exp is not None and fs != exp:
            reason = f"built for a {fs}-byte {bn}, the local file is {exp} bytes"
        else:
            match, reason = True, "ok"
    s = (m or {}).get("summary") or {}
    emit("MAP_MATCH", match)
    emit("MAP_REASON", reason)
    emit("MAP_IS_MOE", bool(m and (int_or_none(m.get("n_expert")) or 0) > 0))
    emit("MAP_N_EXPERT", (m or {}).get("n_expert", ""))
    emit("MAP_NONEXPERT", s.get("nonexpert_bytes", ""))
    emit("MAP_BPT_WORST", s.get("bytes_per_token_worst", ""))
    emit("MAP_BASENAME", (m or {}).get("gguf_basename", ""))
    return 0


def cmd_check_stats(a):
    ok, reason = False, ""
    try:
        st = load_json(a.stats)
    except (OSError, ValueError) as e:
        st, reason = None, f"unreadable ({e})"
    m, mreason = _map_doc(a.map)
    if st is not None and m is None:
        reason = f"map {mreason}"
    elif st is not None:
        per_layer = st.get("per_layer") if isinstance(st, dict) else None
        moe = {int(x) for x in m.get("moe_layers") or []}
        n_by_layer = {int(e.get("layer")): int(e.get("n_expert") or m.get("n_expert") or 0)
                      for e in m.get("expert_layers") or [] if isinstance(e, dict) and "layer" in e}
        if not isinstance(per_layer, list) or not per_layer:
            reason = "no per_layer list (not a pigemma.trace stats.json)"
        elif not moe:
            reason = "the map has no MoE layers (dense model)"
        else:
            problems = []
            for j, ent in enumerate(per_layer):
                try:
                    il = int(ent["layer"])
                    n = len(ent["freq"])
                except (KeyError, TypeError, ValueError):
                    problems.append(f"per_layer[{j}] lacks layer/freq")
                    break
                if il not in moe:
                    problems.append(f"layer {il} is not a MoE layer of the map")
                    break
                want = n_by_layer.get(il, int(m.get("n_expert") or 0))
                if n != want:
                    problems.append(f"layer {il} has {n} experts in the stats, {want} in the map")
                    break
            if problems:
                reason = problems[0]
            else:
                ok, reason = True, f"{len(per_layer)} MoE layers, {st.get('n_tokens', '?')} tokens"
    emit("STATS_OK", ok)
    emit("STATS_REASON", reason)
    emit("STATS_N_TOKENS", (st or {}).get("n_tokens", "") if isinstance(st, dict) else "")
    return 0


# ------------------------------------------------------------------------------------- budget
def cmd_budget(a):
    fsz = int_or_none(a.file_size)
    ctx = int(a.ctx)
    spec = (a.spec or "auto").strip()
    kv, kv_src = None, ""
    if a.kv_type not in KV_PER_32:
        raise Fail(1, f"unknown --kv-type {a.kv_type!r}; known: {', '.join(sorted(KV_PER_32))}")
    if spec.lower() != "auto":
        b = parse_bytes(spec)
        src = a.spec_source or "--budget-bytes"
        formula = f"{src} = {spec} = {b:,} B"
    else:
        avail = int(a.avail)
        compute = parse_bytes(a.compute)
        headroom = parse_bytes(a.headroom)
        if a.map and Path(a.map).is_file():
            try:
                from pigemma import gguf_map as gm
                kv = int(gm.GgufMap.load(a.map).kv_bytes(ctx, a.kv_type))
                kv_src = f"measured from the GGUF header (map {rel(a.map)}): kv_bytes(n_ctx={ctx}, {a.kv_type})"
            except Exception as e:  # noqa: BLE001
                say(f"KV size from the map failed ({e}); using the catalog estimate")
                kv = None
        if kv is None:
            base = num(a.kv_fallback) or 420_000_000.0
            kv = int(round(base * max(1.0, ctx / KV_REF_CTX) * KV_PER_32[a.kv_type] / KV_PER_32["q8_0"]))
            kv_src = (f"assumed: catalog kv_bytes_q8_at_4096 {int(base):,} B scaled to n_ctx={ctx}, "
                      f"{a.kv_type} (no map yet)")
        b = avail - kv - compute - headroom
        src = "auto"
        formula = (f"MemAvailable {avail:,} - KV({ctx}, {a.kv_type}) {kv:,} - compute {compute:,} (assumed) - "
                   f"headroom {headroom:,} (assumed) = {b:,} B")
    capped = False
    if fsz and b > fsz:
        b, capped = fsz, True
        formula += f"; capped at the file size {fsz:,} B"
    if b <= 0:
        raise Fail(3, f"no RAM left for a hot set: {formula}. Free memory, lower CTX, or pass --budget-bytes")
    emit("BUDGET_BYTES", b)
    emit("BUDGET_SOURCE", src)
    emit("BUDGET_FORMULA", formula)
    emit("BUDGET_CAPPED", capped)
    emit("KV_BYTES", kv if kv is not None else "")
    emit("KV_SOURCE", kv_src)
    return 0


# ------------------------------------------------------------------------------------ hot set
def cmd_hotset_info(a):
    from pigemma import gguf_map as gm
    p = Path(a.hotset)
    if not p.is_file():
        raise Fail(3, f"hot set not found: {posix(p)}")
    try:
        hs = load_json(p)
    except (OSError, ValueError) as e:
        raise Fail(1, f"cannot read the hot set {posix(p)}: {e}")
    if not isinstance(hs, dict) or "ranges" not in hs or "total_bytes" not in hs:
        raise Fail(1, f"{posix(p)} is not a pigemma hotset.json")
    rng = gm.ranges_txt_path(p)
    if not rng.is_file():
        raise Fail(3, f"ranges file not found next to the hot set: {posix(rng)}")
    gname = Path(a.gguf).name
    bn = hs.get("gguf_basename") or Path(str(hs.get("gguf") or "")).name
    size = int_or_none(hs.get("gguf_size"))
    exp = int_or_none(a.expect_size)
    if exp is not None and size is not None and size != exp:
        raise Fail(1, f"{posix(p)} was planned for a {size}-byte {bn}, but {gname} is {exp} bytes: plan again "
                      "(start without --hotset)")
    emit("HS_RANGES_FILE", posix(rng))
    emit("HS_TOTAL", hs.get("total_bytes"))
    emit("HS_BUDGET", hs.get("budget_bytes", ""))
    emit("HS_N_RANGES", hs.get("n_ranges", len(hs.get("ranges") or [])))
    emit("HS_N_EXPERTS", hs.get("n_experts", ""))
    emit("HS_N_ITEMS", hs.get("n_expert_items_total", ""))
    emit("HS_COVERAGE", "" if hs.get("expert_coverage_fraction") is None else hs["expert_coverage_fraction"])
    emit("HS_NONEXPERT", bool(hs.get("nonexpert_included")))
    emit("HS_PAGE", hs.get("page_size", ""))
    emit("HS_STATS", hs.get("stats") or "")
    emit("HS_NAME_NOTE", "" if not bn or bn == gname else f"planned for {bn}, pinning {gname}")
    return 0


# -------------------------------------------------------------------------------- pinner output
def cmd_pinner_summary(a):
    d = last_pinner_json(a.file)
    emit("PS_FOUND", d is not None)
    d = d or {}
    emit("PS_STATUS", d.get("status", ""))
    emit("PS_EXIT", d.get("exit_code", ""))
    emit("PS_MODE", d.get("mode", ""))
    emit("PS_PID", d.get("pid", ""))
    for key, name in (("requested_bytes", "PS_REQUESTED"), ("locked_bytes", "PS_LOCKED"),
                      ("resident_bytes", "PS_RESIDENT"), ("resident_before_bytes", "PS_RESIDENT_BEFORE"),
                      ("ranges", "PS_RANGES"), ("failed", "PS_FAILED"), ("not_attempted", "PS_NOT_ATTEMPTED"),
                      ("vm_lck_kb", "PS_VMLCK_KB"), ("lock_s", "PS_LOCK_S"), ("verify", "PS_VERIFY")):
        v = d.get(key)
        emit(name, "" if v is None else v)
    lim = d.get("memlock_limit", "")
    emit("PS_MEMLOCK", "unlimited" if (d and lim is None) else lim)
    emit("PS_RELIABLE", d.get("mincore_reliable", ""))
    emit("PS_REASON", d.get("reason") or "")
    return 0


# ------------------------------------------------------------------------------------- state
def _hotset_info(path):
    if not path or not Path(path).is_file():
        return None
    try:
        hs = load_json(path)
    except (OSError, ValueError):
        return None
    return {k: hs.get(k) for k in HOTSET_INFO_KEYS if k in hs}


def cmd_state_write(a):
    st = {
        "format_version": 1, "generator": "pi/06_pin_hotset.sh", "version": a.version,
        "created_at": now_iso(), "updated_at": now_iso(), "state": a.state, "host": a.host or None,
        "emulated": a.emulated == "1", "protected_override": a.override == "1",
        "pid": int_or_none(a.pid), "pidfile": posix(a.pidfile), "log": posix(a.log),
        "stderr_log": posix(a.errlog), "pinner": posix(a.pinner),
        "argv": json.loads(a.argv_json) if a.argv_json else None,
        "exit_code": int_or_none(a.exit_code),
        "model": {"id": a.model_id or None, "path": posix(a.model), "bytes": int_or_none(a.model_bytes)},
        "map": posix(a.map), "stats": posix(a.stats), "hotset": posix(a.hotset), "ranges": posix(a.ranges),
        "plan": a.plan, "page_size": int_or_none(a.page_size), "max_hold_s": int_or_none(a.max_hold) or None,
        "budget": {"bytes": int_or_none(a.budget_bytes), "source": a.budget_source or None,
                   "formula": a.budget_formula or None, "mem_available_bytes": int_or_none(a.mem_available),
                   "kv_bytes": int_or_none(a.kv_bytes), "kv_source": a.kv_source or None,
                   "kv_type": a.kv_type or None, "ctx": int_or_none(a.ctx),
                   "compute_bytes": int_or_none(a.compute), "headroom_bytes": int_or_none(a.headroom)},
        "hotset_info": _hotset_info(a.hotset),
        "summary": last_pinner_json(a.summary_file),
        "notes": list(a.note or []),
    }
    write_atomic(a.out, dumps(st))
    return 0


def cmd_state_update(a):
    p = Path(a.file)
    if not p.is_file():
        return 0
    try:
        st = load_json(p)
    except (OSError, ValueError):
        return 0
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        try:
            st[k] = json.loads(v)
        except ValueError:
            st[k] = v
    st["updated_at"] = now_iso()
    write_atomic(p, dumps(st))
    return 0


def _fmt_summary_lines(st):
    s = st.get("summary") or {}
    hi = st.get("hotset_info") or {}
    bud = st.get("budget") or {}
    model = st.get("model") or {}
    out = []
    state = st.get("state")
    head = f"pinner: {state}"
    if st.get("pid"):
        head += f", pid {st['pid']}"
    if s:
        head += f" (status {s.get('status')}, exit code {s.get('exit_code')}"
        head += ", holds the locks until 'stop')" if state == "running" else ")"
    out.append(head)
    out.append(f"  model      {model.get('path')} ({gb(model.get('bytes'))}{', ' + model['id'] if model.get('id') else ''})")
    out.append(f"  hot set    {st.get('hotset')} -> {st.get('ranges')} ({st.get('plan')})")
    if bud.get("bytes") is not None:
        out.append(f"  budget     {gb(bud.get('bytes'))} ({bud.get('source')}): {bud.get('formula') or '-'}")
    if hi:
        out.append(f"  planned    {gb(hi.get('total_bytes'))} in {hi.get('n_ranges')} ranges, page "
                   f"{hi.get('page_size')}; non-expert {'included' if hi.get('nonexpert_included') else 'excluded'}"
                   f" ({gb(hi.get('nonexpert_bytes_aligned'))} aligned)")
        cov = hi.get("expert_coverage_fraction")
        items = hi.get("n_expert_items_total")
        n = hi.get("n_experts")
        frac = (f" ({100.0 * n / items:.1f}% of the expert slabs)" if isinstance(n, int) and items else "")
        out.append(f"  experts    {n} of {items} expert-layer items{frac}; expert_coverage_fraction "
                   f"{'n/a (no stats)' if cov is None else f'{cov:.4f}'} (share of the routed expert accesses "
                   f"in {hi.get('stats') or 'the stats'})")
    if s:
        vm = s.get("vm_lck_kb")
        out.append(f"  locked     {gb(s.get('locked_bytes'))} of {gb(s.get('requested_bytes'))} requested "
                   f"(VmLck {'n/a' if vm is None else f'{vm:,} kB'}); ranges {s.get('ranges')}, failed "
                   f"{s.get('failed')}, not attempted {s.get('not_attempted')}")
        rel_txt = "" if s.get("mincore_reliable", True) else " (mincore unreliable for this user: n/a)"
        out.append(f"  resident   {gb(s.get('resident_bytes'))} after locking (before: "
                   f"{gb(s.get('resident_before_bytes'))}){rel_txt}")
        lim = s.get("memlock_limit")
        out.append(f"  lock time  {s.get('lock_s')} s; memlock limit {'unlimited' if lim is None else f'{lim:,} B'}")
        if s.get("reason"):
            out.append(f"  reason     {s.get('reason')}")
    out.append(f"  files      log {st.get('log')}, stderr {st.get('stderr_log')}, pidfile {st.get('pidfile')}")
    for n_ in st.get("notes") or []:
        out.append(f"  note       {n_}")
    if state == "running":
        out.append("  stop with  bash pi/06_pin_hotset.sh stop   (status: bash pi/06_pin_hotset.sh status)")
    return out


def cmd_report_start(a):
    st = load_json(a.state)
    if a.json == "1":
        sys.stdout.write(dumps(st))
    else:
        print("\n".join(_fmt_summary_lines(st)))
    return 0


def cmd_status_report(a):
    st = None
    if a.state and Path(a.state).is_file():
        try:
            st = load_json(a.state)
        except (OSError, ValueError):
            st = None
    ver = last_pinner_json(a.verify) if a.verify else None
    running = a.running == "1"
    vm = int_or_none(a.vmlck)
    doc = {"running": running, "pid": int_or_none(a.pid), "vm_lck_bytes": vm if running else None,
           "pidfile": posix(a.pidfile), "verify": ver, "verify_exit_code": int_or_none(a.verify_rc),
           "state": st, "checked_at": now_iso(), "notes": list(a.note or [])}
    if st and running and st.get("pid") not in (None, doc["pid"]):
        doc["notes"].append(f"pin_state.json names pid {st.get('pid')}, the pidfile names {doc['pid']}")
    if a.json == "1":
        sys.stdout.write(dumps(doc))
        return 0
    if not running:
        print(f"pinner: not running ({a.why or 'no live pinner in ' + str(posix(a.pidfile))})")
        if st:
            print(f"  last state  {st.get('state')} (updated {st.get('updated_at')}), model "
                  f"{(st.get('model') or {}).get('path')}, hot set {st.get('hotset')}")
            if st.get("state") == "running":
                print("  note       pin_state.json still says 'running': the pinner exited by itself (--max-hold "
                      "expiry, a signal or a crash; see its stderr log " + str(st.get("stderr_log")) + ")")
    else:
        print(f"pinner: running, pid {a.pid}, VmLck {gb(vm)} locked")
        if st:
            print(f"  model      {(st.get('model') or {}).get('path')}")
            print(f"  hot set    {st.get('hotset')} -> {st.get('ranges')} (started {st.get('created_at')})")
            s = st.get("summary") or {}
            if s:
                print(f"  at start   locked {gb(s.get('locked_bytes'))} of {gb(s.get('requested_bytes'))}, "
                      f"status {s.get('status')}")
    if ver:
        req = num(ver.get("requested_bytes"))
        res = num(ver.get("resident_bytes"))
        share = f" ({100.0 * res / req:.1f}%)" if req and res is not None else ""
        print(f"  resident   {gb(res)} of {gb(req)} hot-set bytes now (mincore){share}"
              + ("" if ver.get("mincore_reliable", True) else "; mincore unreliable for this user"))
        for kind, k in sorted((ver.get("kinds") or {}).items()):
            kr, kq = num(k.get("resident_bytes")), num(k.get("requested_bytes"))
            kshare = f" ({100.0 * kr / kq:.1f}%)" if kq and kr is not None else ""
            print(f"    {kind:<12} {gb(kr)} of {gb(kq)}{kshare}, {k.get('ranges')} ranges")
    elif a.verify:
        print(f"  resident   n/a (pinner --verify-only failed, exit {a.verify_rc}; see {posix(a.verify)}.err)")
    for n_ in doc["notes"]:
        print(f"  note       {n_}")
    return 0


# ----------------------------------------------------------------------------------- A/B report
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _ratio(a_, b_):
    if a_ is None or b_ is None or b_ == 0:
        return None
    return a_ / b_


def _run_record(i, a):
    label = a.run_label[i]
    rec = {"label": label, "round": int_or_none(a.run_round[i]), "exit_code": int_or_none(a.run_rc[i]),
           "started_at": a.run_started[i] or None, "ended_at": a.run_ended[i] or None,
           "result_files": [f for f in (a.run_files[i] or "").split(",") if f],
           "tg_tok_s": None, "pp_tok_s": None, "read_bytes_per_token": None, "majflt_per_token": None,
           "load_s": None, "coherent": None, "failed": True, "mode": None, "effective_flags": None,
           "pinned_by_pi05": None, "pinner": None, "hot_set_requested_bytes": None,
           "hot_set_resident_before_bytes": None, "hot_set_resident_after_bytes": None,
           "mincore_reliable": None, "last_run_copy": rel(a.run_last[i]) if a.run_last[i] else None,
           "notes": []}
    last = None
    if a.run_last[i] and Path(a.run_last[i]).is_file():
        try:
            last = load_json(a.run_last[i])
        except (OSError, ValueError):
            last = None
    if last is None:
        rec["notes"].append("pi/05 wrote no fresh last_stream-bench.json for this run")
    else:
        ms = [m for m in last.get("metrics") or [] if isinstance(m, dict)]
        m = next((x for x in ms if x.get("label") == "stream-bench"), ms[0] if ms else None)
        if m:
            for k in ("tg_tok_s", "pp_tok_s", "read_bytes_per_token", "majflt_per_token", "load_s", "coherent"):
                rec[k] = m.get(k)
            rec["failed"] = bool(m.get("failed")) or rec["tg_tok_s"] is None
            if not rec["result_files"] and m.get("file"):
                rec["result_files"] = [m["file"]]
            env = None
            if m.get("file"):
                try:
                    env = load_json(resolve(m["file"]))
                except (OSError, ValueError):
                    env = None
            p05 = ((env or {}).get("extra") or {}).get("pi05") or {}
            rec["mode"] = p05.get("mode") or last.get("mode")
            rec["effective_flags"] = p05.get("effective_flags") or last.get("effective_flags")
            rec["pinned_by_pi05"] = bool(p05.get("pin")) if p05 else bool(last.get("pinned"))
        else:
            rec["notes"].append("last_stream-bench.json has no metrics")
    if rec["exit_code"] not in (0, None):
        rec["failed"] = True
    vb = last_pinner_json(a.run_res_before[i]) if a.run_res_before[i] else None
    va = last_pinner_json(a.run_res_after[i]) if a.run_res_after[i] else None
    if vb:
        rec["hot_set_requested_bytes"] = vb.get("requested_bytes")
        rec["hot_set_resident_before_bytes"] = vb.get("resident_bytes")
        rec["mincore_reliable"] = vb.get("mincore_reliable")
    if va:
        rec["hot_set_resident_after_bytes"] = va.get("resident_bytes")
        rec["hot_set_requested_bytes"] = rec["hot_set_requested_bytes"] or va.get("requested_bytes")
        if rec["mincore_reliable"] is None:
            rec["mincore_reliable"] = va.get("mincore_reliable")
    if label == "pinned" and a.run_pin_state[i] and Path(a.run_pin_state[i]).is_file():
        try:
            ps = load_json(a.run_pin_state[i])
        except (OSError, ValueError):
            ps = None
        if ps:
            s = ps.get("summary") or {}
            rec["pinner"] = {"pid": ps.get("pid"), "state": ps.get("state"), "status": s.get("status"),
                             "exit_code": s.get("exit_code"), "requested_bytes": s.get("requested_bytes"),
                             "locked_bytes": s.get("locked_bytes"), "resident_bytes": s.get("resident_bytes"),
                             "resident_before_bytes": s.get("resident_before_bytes"),
                             "vm_lck_kb": s.get("vm_lck_kb"), "failed": s.get("failed"),
                             "not_attempted": s.get("not_attempted"), "lock_s": s.get("lock_s"),
                             "memlock_limit": s.get("memlock_limit"), "reason": s.get("reason"),
                             "pin_state_copy": rel(a.run_pin_state[i])}
    return rec


def cmd_ab_report(a):
    n = len(a.run_label)
    for name in ("run_round", "run_rc", "run_last", "run_files", "run_res_before", "run_res_after",
                 "run_pin_state", "run_started", "run_ended"):
        if len(getattr(a, name)) != n:
            raise Fail(1, f"ab-report: --{name.replace('_', '-')} given {len(getattr(a, name))} times, expected {n}")
    runs = [_run_record(i, a) for i in range(n)]
    hs = _hotset_info(a.hotset) or {}
    bpt = int_or_none(a.bpt_worst)
    groups = {}
    for lab in ("unpinned", "pinned"):
        ok = [r for r in runs if r["label"] == lab and not r["failed"]]
        groups[lab] = {"n": len(ok), "n_failed": len([r for r in runs if r["label"] == lab]) - len(ok),
                       "tg_tok_s": _mean([num(r["tg_tok_s"]) for r in ok]),
                       "read_bytes_per_token": _mean([num(r["read_bytes_per_token"]) for r in ok]),
                       "majflt_per_token": _mean([num(r["majflt_per_token"]) for r in ok])}
    u, p = groups["unpinned"], groups["pinned"]
    rb_ratio = _ratio(p["read_bytes_per_token"], u["read_bytes_per_token"])
    ratio = {"tok_s_pinned_over_unpinned": _ratio(p["tg_tok_s"], u["tg_tok_s"]),
             "read_bytes_per_token_pinned_over_unpinned": rb_ratio,
             "bytes_per_token_reduction": None if rb_ratio is None else 1.0 - rb_ratio,
             "majflt_per_token_pinned_over_unpinned": _ratio(p["majflt_per_token"], u["majflt_per_token"])}
    for k, v in list(ratio.items()):
        ratio[k] = None if v is None else round(v, 4)
    in_ram = None
    if u["read_bytes_per_token"] is not None and bpt:
        in_ram = u["read_bytes_per_token"] < IN_RAM_FRACTION * bpt
    notes = [
        "MEASURED on this host: tok/s and read_bytes_per_token come from pi/05_run.sh stream-bench "
        "(pigemma.bench completion: llama-completion timings, /proc/<pid>/io read_bytes during generation / "
        "generated tokens); the hot-set residency is mincore() via pinner --verify-only",
        "order: the first run of every round warms the page cache for the second one (odd rounds run "
        "unpinned first, even rounds pinned first; --rounds 2 gives ABBA)",
        "the budget terms compute and headroom are ASSUMED values (pi/06 --compute-bytes / --headroom-bytes)",
    ]
    if a.emulated == "1":
        notes.append("EMULATED host (PIGEMMA_EMULATED=1): timings are qemu artefacts, not Raspberry Pi numbers")
    if a.override == "1":
        notes.append("protected-host override (PIGEMMA_I_KNOW=1) was active")
    modes = sorted({str(r["mode"]) for r in runs if r["mode"]})
    if modes and any(m_ != "stream_moe" for m_ in modes):
        notes.append(f"pi/05 ran the model in mode {', '.join(modes)}: pinning can only help when the model "
                     "streams from storage (mode stream_moe)")
    flags = " ".join(str(r["effective_flags"] or "") for r in runs)
    if "mlock" in flags:
        notes.append("the llama.cpp flags already mlock the model (-lm ...mlock): both runs keep every page")
    reason, result = "", "INCONCLUSIVE"
    pinned_runs = [r for r in runs if r["label"] == "pinned"]
    if any(r["failed"] for r in runs):
        reason = "at least one run failed; see runs[].exit_code and the pi/05 envelopes"
    elif u["tg_tok_s"] is None or p["tg_tok_s"] is None:
        reason = "a tok/s value is missing"
    elif any(not r["pinned_by_pi05"] for r in pinned_runs):
        reason = "a 'pinned' run had no live pinner according to pi/05 (extra.pi05.pin is null)"
    elif in_ram:
        reason = (f"the unpinned run read {u['read_bytes_per_token']:,.0f} B/token from storage (< "
                  f"{IN_RAM_FRACTION:.0%} of the {bpt:,} B/token the model touches): the model is served from "
                  "RAM, pinning cannot show a gain here (use a smaller-RAM board or a memory balloon)")
    else:
        tr = ratio["tok_s_pinned_over_unpinned"]
        red = ratio["bytes_per_token_reduction"]
        ok_t = tr is not None and tr >= E6_TOK_S_RATIO
        ok_b = red is not None and red >= E6_BYTES_REDUCTION
        result = "PASS" if (ok_t or ok_b) else "FAIL"
        reason = (f"tok/s ratio {'n/a' if tr is None else f'{tr:.3f}'} (>= {E6_TOK_S_RATIO}: {ok_t}); "
                  f"bytes/token reduction {'n/a' if red is None else f'{red:.1%}'} (>= {E6_BYTES_REDUCTION:.0%}: "
                  f"{ok_b})")
    if any((r.get("pinner") or {}).get("status") == "partial" for r in pinned_runs):
        notes.append("the pinner locked only part of the hot set (status partial): see runs[].pinner")
    doc = {
        "format_version": 1, "generator": "pi/06_pin_hotset.sh ab", "version": a.version,
        "created_at": now_iso(), "kind": "measured",
        "experiment_ref": "E6 hot-set pinning, repeated on the Pi (docs/EXPERIMENT_PLAN.md#e6-hot-set-pinning-proxy)",
        "host": a.host or None, "emulated": a.emulated == "1", "protected_override": a.override == "1",
        "argv": json.loads(a.argv_json) if a.argv_json else None,
        "model": {"id": a.model_id or None, "path": posix(a.model), "bytes": int_or_none(a.model_bytes)},
        "hotset": {"json": rel(a.hotset), "ranges": rel(a.ranges), "map": rel(a.map) if a.map else None,
                   "stats": rel(a.stats) if a.stats else None, "budget_bytes": int_or_none(a.budget_bytes),
                   "budget_source": a.budget_source or None, "budget_formula": a.budget_formula or None,
                   **{k: hs.get(k) for k in ("total_bytes", "n_ranges", "n_experts", "n_expert_items_total",
                                             "expert_coverage_fraction", "nonexpert_included",
                                             "nonexpert_bytes_aligned", "page_size", "rank")}},
        "rounds": int_or_none(a.rounds), "order": [f"r{r['round']}:{r['label']}" for r in runs],
        "runs": runs, "unpinned": u, "pinned": p, "ratio": ratio,
        "bytes_per_token_worst": bpt, "in_ram": in_ram,
        "threshold_check": {
            "rule": "E6 (pre-registered, docs/EXPERIMENT_PLAN.md): PASS if pinned tg >= 1.15x unpinned OR "
                    "read bytes/token reduced >= 20%",
            "tok_s_ratio_min": E6_TOK_S_RATIO, "bytes_reduction_min": E6_BYTES_REDUCTION,
            "result": result, "reason": reason},
        "notes": notes,
    }
    text = dumps(doc)
    for out in a.out:
        write_atomic(out, text)
    # human table on stderr
    w = sys.stderr.write
    w("\n==== pin A/B (pi/06_pin_hotset.sh ab) ====\n")
    w(f"model     {a.model_id or '-'}  {posix(a.model)}\n")
    w(f"hot set   budget {gb(a.budget_bytes)} ({a.budget_source}), planned {gb(hs.get('total_bytes'))}, "
      f"{hs.get('n_experts')} experts, expert_coverage_fraction {hs.get('expert_coverage_fraction')}\n")
    w(f"{'run':<5}{'label':<10}{'rc':>4}{'tg tok/s':>11}{'read MB/tok':>13}{'majflt/tok':>12}"
      f"{'hot resident before -> after':>32}\n")
    for r in runs:
        tg = num(r["tg_tok_s"])
        rb = num(r["read_bytes_per_token"])
        mf = num(r["majflt_per_token"])
        w(f"r{r['round']:<4}{r['label']:<10}{str(r['exit_code']):>4}"
          f"{'n/a' if tg is None else f'{tg:.3f}':>11}{'n/a' if rb is None else f'{rb / 1e6:.1f}':>13}"
          f"{'n/a' if mf is None else f'{mf:.1f}':>12}"
          f"{gb(r['hot_set_resident_before_bytes']) + ' -> ' + gb(r['hot_set_resident_after_bytes']):>32}\n")
    tr = ratio["tok_s_pinned_over_unpinned"]
    rr = ratio["read_bytes_per_token_pinned_over_unpinned"]
    w(f"ratio     pinned/unpinned: tok/s {'n/a' if tr is None else f'{tr:.3f}x'}, read bytes/token "
      f"{'n/a' if rr is None else f'{rr:.3f}x'}\n")
    w(f"E6 check  {result}: {reason}\n")
    for n_ in notes:
        w(f"note      {n_}\n")
    return 0 if not any(r["failed"] for r in runs) else 1


# ------------------------------------------------------------------------------------------ main
def main(argv):
    ap = argparse.ArgumentParser(prog="pi06-helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("resolve-model")
    s.add_argument("--model", default="")
    s.add_argument("--model-dir", default="")
    s.add_argument("--catalog", default="")
    s.add_argument("--feasibility", default="")
    s = sub.add_parser("check-map")
    s.add_argument("--map", required=True)
    s.add_argument("--gguf", required=True)
    s.add_argument("--expect-size", default="")
    s = sub.add_parser("check-stats")
    s.add_argument("--stats", required=True)
    s.add_argument("--map", required=True)
    s = sub.add_parser("budget")
    for k in ("--spec", "--spec-source", "--map", "--file-size", "--kv-fallback"):
        s.add_argument(k, default="")
    s.add_argument("--avail", default="0")
    s.add_argument("--ctx", default="4096")
    s.add_argument("--kv-type", default="q8_0")
    s.add_argument("--compute", default="300000000")
    s.add_argument("--headroom", default="500000000")
    s = sub.add_parser("hotset-info")
    s.add_argument("--hotset", required=True)
    s.add_argument("--gguf", required=True)
    s.add_argument("--expect-size", default="")
    s = sub.add_parser("pinner-summary")
    s.add_argument("--file", required=True)
    s = sub.add_parser("state-write")
    for k in ("--out", "--state", "--version", "--host", "--emulated", "--override", "--pid", "--pidfile",
              "--log", "--errlog", "--pinner", "--argv-json", "--exit-code", "--model", "--model-id",
              "--model-bytes", "--map", "--stats", "--hotset", "--ranges", "--plan", "--page-size",
              "--budget-bytes", "--budget-source", "--budget-formula", "--mem-available", "--kv-bytes",
              "--kv-source", "--kv-type", "--ctx", "--compute", "--headroom", "--summary-file", "--max-hold"):
        s.add_argument(k, default="")
    s.add_argument("--note", action="append")
    s = sub.add_parser("state-update")
    s.add_argument("--file", required=True)
    s.add_argument("--set", action="append")
    s = sub.add_parser("report-start")
    s.add_argument("--state", required=True)
    s.add_argument("--json", default="0")
    s = sub.add_parser("status-report")
    for k in ("--state", "--running", "--pid", "--vmlck", "--verify", "--verify-rc", "--pidfile", "--why"):
        s.add_argument(k, default="")
    s.add_argument("--json", default="0")
    s.add_argument("--note", action="append")
    s = sub.add_parser("ab-report")
    for k in ("--version", "--host", "--emulated", "--override", "--argv-json", "--model", "--model-id",
              "--model-bytes", "--map", "--stats", "--hotset", "--ranges", "--budget-bytes", "--budget-source",
              "--budget-formula", "--bpt-worst", "--rounds"):
        s.add_argument(k, default="")
    s.add_argument("--out", action="append", required=True)
    for k in ("--run-label", "--run-round", "--run-rc", "--run-last", "--run-files", "--run-res-before",
              "--run-res-after", "--run-pin-state", "--run-started", "--run-ended"):
        s.add_argument(k, action="append", default=[])
    a = ap.parse_args(argv)
    fn = {"resolve-model": cmd_resolve_model, "check-map": cmd_check_map, "check-stats": cmd_check_stats,
          "budget": cmd_budget, "hotset-info": cmd_hotset_info, "pinner-summary": cmd_pinner_summary,
          "state-write": cmd_state_write, "state-update": cmd_state_update, "report-start": cmd_report_start,
          "status-report": cmd_status_report, "ab-report": cmd_ab_report}[a.cmd]
    try:
        return fn(a)
    except Fail as e:
        say(str(e))
        return e.code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
PIGEMMA_PY06_EOF
    )
}

# ========================================================================================== pinner pid
_p06_pidfile_pid() { # raw pid from the pidfile (may be stale); 1 if none
    local pid=""
    if [ ! -r "$PIDFILE" ]; then return 1; fi
    IFS= read -r pid <"$PIDFILE" || true
    pid=${pid%$'\r'}
    if ! _pg_is_uint "$pid"; then return 1; fi
    printf '%s\n' "$pid"
}

_p06_pid_is_pinner() { # 0 when /proc/PID is a live (non-zombie) process whose comm starts with 'pinner'
    local st=""
    if ! _pg_read1 "/proc/$1/comm"; then return 1; fi
    case $_R in
        pinner*) ;;
        *) return 1 ;;
    esac
    _pg_p "/proc/$1/stat"
    if [ -r "$_P" ]; then
        IFS= read -r st <"$_P" 2>/dev/null || true
        st=${st##*) }
        case ${st%% *} in Z | X) return 1 ;; esac
    fi
    return 0
}

_p06_pinner_pid() { # live pinner named by the pidfile -> pid on stdout; 1 if none
    local pid
    pid=$(_p06_pidfile_pid) || return 1
    _p06_pid_is_pinner "$pid" || return 1
    printf '%s\n' "$pid"
}

_p06_vmlck_bytes() { # VmLck of a pid in bytes (0 when unreadable)
    local k v rest
    _pg_p "/proc/$1/status"
    if [ -r "$_P" ]; then
        while read -r k v rest || [ -n "${k:-}" ]; do
            if [ "$k" = "VmLck:" ] && _pg_is_uint "$v"; then
                printf '%s\n' "$((v * 1024))"
                return 0
            fi
            k=""
        done <"$_P"
    fi
    printf '0\n'
}

_p06_alive() { # our own child pinner still running (not a zombie)?
    local st=""
    kill -0 "$1" 2>/dev/null || return 1
    if [ -r "/proc/$1/stat" ]; then
        IFS= read -r st <"/proc/$1/stat" 2>/dev/null || true
        st=${st##*) }
        case ${st%% *} in Z | X) return 1 ;; esac
    fi
    return 0
}

_p06_cap_ipc_lock() { # 0 when CAP_IPC_LOCK (bit 14) is in this shell's effective capability set
    local k v rest
    _pg_p /proc/self/status
    if [ ! -r "$_P" ]; then return 1; fi
    while read -r k v rest || [ -n "${k:-}" ]; do
        if [ "$k" = "CapEff:" ] && [[ $v =~ ^[0-9a-fA-F]+$ ]]; then
            [ $(((16#$v >> 14) & 1)) = 1 ] && return 0
            return 1
        fi
        k=""
    done <"$_P"
    return 1
}

_p06_log_has_json() { # 0 when FILE holds a complete line starting with '{' (the pinner's summary)
    local l
    [ -s "$1" ] || return 1
    while IFS= read -r l; do # only newline-terminated lines (read fails on a partial last line)
        case $l in '{'*) return 0 ;; esac
    done <"$1"
    return 1
}

# ======================================================================================== gates
_p06_check_bench_lock() { # die 4 when another live job holds results/.bench.lock
    local out st=0
    if [ "$O_DRY" = 1 ]; then return 0; fi
    out=$(_p06_pyrun -m pigemma.paths --log-level WARNING lock --status 2>/dev/null) || st=$?
    case $out in
        *'"state": "held"'*)
            die 4 "the bench lock ${PIGEMMA_LOCK_FILE:-results/.bench.lock} is held by another model-loading job: wait for it (python3 -m pigemma.paths lock --status)"
            ;;
        *'"state": "inherited"'*) log "bench lock: held by the parent job (inherited)" ;;
        *) if [ "$st" -gt 1 ]; then warn "could not read the bench lock state (exit $st); continuing"; fi ;;
    esac
}

_p06_gate_pin() { # safe host, mlock permission, fake root, live pinner, bench lock (start and ab)
    assert_safe_host
    if ! mlock_allowed; then
        case ${ALLOW_MLOCK:-auto} in
            0 | [Nn][Oo] | [Ff][Aa][Ll][Ss][Ee]) die 4 "hot-set pinning mlocks model pages: ALLOW_MLOCK=${ALLOW_MLOCK} forbids it. Nothing was started." ;;
            *) die 4 "hot-set pinning mlocks model pages: not allowed on this protected host (ALLOW_MLOCK=${ALLOW_MLOCK:-auto}: its page cache must stay reclaimable for its services). Nothing was started." ;;
        esac
    fi
    PINNER_PROT_ARGS=()
    if [ "$O_ALLOW_PROT" = 1 ]; then
        PINNER_PROT_ARGS=(--allow-protected-host)
        warn "--allow-protected-host given: the pinner's own protected-host guard is overridden (only with the host owner's explicit approval)"
    elif [ "${PIGEMMA_PROTECTED_OVERRIDE:-0}" = 1 ]; then
        die 4 "protected host with ALLOW_MLOCK=1: the pinner refuses here unless --allow-protected-host is also given (only with the host owner's explicit approval)"
    fi
    if [ "${PIGEMMA_PROTECTED_OVERRIDE:-0}" = 1 ] && [ -z "$O_MAX_HOLD" ]; then
        O_MAX_HOLD=7200
        warn "protected host: the pinner releases its locks by itself after ${O_MAX_HOLD}s (--max-hold; bounded experiment)"
    fi
    if is_fake && [ "$O_DRY" != 1 ]; then
        die 1 "PIGEMMA_FAKE_PROC is set (tests): '$SUBCMD' only runs with --dry-run here (a real pinner would lock memory of THIS machine)"
    fi
    local pid
    if pid=$(_p06_pinner_pid); then
        die 4 "a pinner is already running (pid $pid, pidfile $PIDFILE, VmLck $(_p06_fmt "$(_p06_vmlck_bytes "$pid")")): stop it first with pi/06_pin_hotset.sh stop"
    fi
    _p06_check_bench_lock
}

# ================================================================================== preparation
MODEL_PATH="" MODEL_ID="" MODEL_FILE_BYTES="" MODEL_EXISTS=0 MODEL_BASENAME="" MODEL_STEM=""
PINNER_BIN="" PAGE="" AVAIL_BYTES=0 MAP_PATH="" MAP_PENDING=0 MAP_IS_MOE=0 MAP_BPT_WORST="" STATS_PATH=""
BUDGET_BYTES="" BUDGET_SOURCE="" BUDGET_FORMULA="" KV_BYTES="" KV_SOURCE="" HOTSET_JSON="" RANGES_TXT=""
HS_TOTAL="" PLAN_KIND="planned" STATE_NOTES=()

_p06_resolve_model() {
    local spec=$O_MODEL out rc=0 n
    case $spec in
        */* | *\\* | *.gguf | *.GGUF) spec=$(native_path "$(_p06_abspath "$spec")") ;;
    esac
    out=$(_p06_py resolve-model "--model=$spec" "--model-dir=$(native_path "$MODEL_DIR")" \
        "--catalog=$(native_path "$PIGEMMA_CATALOG")" "--feasibility=$(native_path "$PIGEMMA_OUT/feasibility.json")") || rc=$?
    if [ "$rc" != 0 ]; then die "$rc" "model resolution failed (see the pi/06 message above)"; fi
    eval "$out"
    if [ -n "$MODEL_NOTES" ]; then
        IFS='|' read -r -a _notes <<<"$MODEL_NOTES" || true
        for n in "${_notes[@]}"; do
            _pg_trim "$n"
            if [ -n "$_R" ]; then log "model: $_R"; fi
        done
    fi
    if [ "$MODEL_EXISTS" != 1 ]; then
        local hint="pi/04_fetch_model.sh${MODEL_FAMILY:+ --model $MODEL_FAMILY}${MODEL_QUANT:+ --quant $MODEL_QUANT}"
        if [ "$O_DRY" = 1 ]; then
            warn "model file not found: $MODEL_PATH (dry run continues; download it with $hint)"
        else
            die 3 "model file not found: $MODEL_PATH (MODEL_DIR=$MODEL_DIR); download it with $hint"
        fi
    fi
    if [ "$MODEL_SIZE_MISMATCH" = 1 ]; then
        local msg="$MODEL_PATH is $MODEL_SIZE bytes but the catalog says $MODEL_CATALOG_BYTES for $MODEL_ID: incomplete download? (pi/04_fetch_model.sh resumes and verifies)"
        if [ "$O_DRY" = 1 ]; then warn "$msg"; else die 3 "$msg"; fi
    fi
    if [[ $MODEL_BASENAME =~ -[0-9]{5}-of-[0-9]{5}\.gguf$ ]]; then
        # gguf-split parts: tensor offsets are per part file, the map / hot set / pinner handle one file
        die 3 "$MODEL_BASENAME is one part of a split GGUF (llama-gguf-split): hot-set pinning needs a single-file GGUF (merge the parts with llama-gguf-split --merge onto a filesystem that allows files > 4 GB, or fetch the single file)"
    fi
    if [ "$MODEL_KIND" = dense ]; then
        warn "$MODEL_BASENAME is a dense model: there are no experts to choose, the hot set is all non-expert tensors (the whole file) or nothing"
    fi
    log "model: ${MODEL_ID:-uncatalogued} $MODEL_PATH ($(_p06_fmt "${MODEL_FILE_BYTES:-}"))"
}

_p06_resolve_pinner() { # PINNER_BIN; $1 = 1 when a missing binary is fatal
    local fatal=${1:-1}
    if [ -n "$O_PINNER" ]; then
        PINNER_BIN=$(_p06_abspath "$O_PINNER")
    else
        PINNER_BIN=$(tools_bin pinner)
        if [ ! -f "$PINNER_BIN" ] && [ -z "${PIGEMMA_TOOLS_BIN:-}" ] &&
            [ "$(uname -m 2>/dev/null || true)" = x86_64 ] && [ -f "$PIGEMMA_HOME/tools/build-linux-x86/bin/pinner" ]; then
            PINNER_BIN="$PIGEMMA_HOME/tools/build-linux-x86/bin/pinner"
            log "pinner: using the x86 Docker build $PINNER_BIN (tools/build/bin/pinner does not exist)"
        fi
    fi
    if [ ! -f "$PINNER_BIN" ]; then
        local msg="pinner not found at $PINNER_BIN: build it with pi/02_build_llama.sh --with-tools (or set PIGEMMA_TOOLS_BIN / --pinner)"
        if [ "$fatal" != 1 ]; then return 1; fi
        if [ "$O_DRY" = 1 ]; then warn "$msg (dry run continues)"; else die 3 "$msg"; fi
    fi
    return 0
}

_p06_check_map() { # _p06_check_map MAP -> MAP_MATCH MAP_REASON MAP_IS_MOE MAP_NONEXPERT MAP_BPT_WORST ...
    local out
    out=$(_p06_py check-map "--map=$(native_path "$1")" "--gguf=$(native_path "$MODEL_PATH")" \
        "--expect-size=$MODEL_FILE_BYTES") || die 1 "cannot check the map $1"
    eval "$out"
}

_p06_resolve_map() {
    local cand given=0 target rc=0
    MAP_PENDING=0
    if [ -n "$O_MAP" ]; then
        cand=$(_p06_abspath "$O_MAP")
        given=1
    else
        cand="$PIGEMMA_HOME/$DEFAULT_MAP_REL"
    fi
    target="$PIGEMMA_OUT/map_${MODEL_STEM}.json"
    if [ -f "$cand" ]; then
        _p06_check_map "$cand"
        if [ "$MAP_MATCH" = 1 ]; then
            MAP_PATH=$cand
            log "map: $cand describes $MODEL_BASENAME"
            return 0
        fi
        if [ "$given" = 1 ]; then
            warn "--map $O_MAP does not describe $MODEL_BASENAME ($MAP_REASON): a map of the local GGUF is used instead"
        else
            log "map: $DEFAULT_MAP_REL does not describe $MODEL_BASENAME ($MAP_REASON): using a map of the local GGUF"
        fi
    elif [ "$given" = 1 ]; then
        target=$cand
        log "map: $cand does not exist yet: building it from the local GGUF"
    else
        log "map: $DEFAULT_MAP_REL not found: using a map of the local GGUF"
    fi
    if [ -f "$target" ] && [ "$target" != "$cand" ]; then
        _p06_check_map "$target"
        if [ "$MAP_MATCH" = 1 ]; then
            MAP_PATH=$target
            log "map: reusing $target"
            return 0
        fi
    fi
    MAP_PATH=$target
    local cmd=(-m pigemma.gguf_map build "$(native_path "$MODEL_PATH")" --out "$(native_path "$target")" --page-size "$PAGE")
    if [ "$O_DRY" = 1 ]; then
        printf '%s\n' "$(_p06_join_cmd "$PY" "${cmd[@]}")"
        MAP_PENDING=1
        return 0
    fi
    log "building the map of $MODEL_PATH (reads the GGUF header only) -> $target"
    _p06_pyrun "${cmd[@]}" >&2 || rc=$?
    case $rc in
        0) ;;
        3) die 3 "map build: the GGUF or a dependency is missing (see above)" ;;
        *) die 1 "map build failed (exit $rc): is $MODEL_PATH a complete GGUF? (see above)" ;;
    esac
    _p06_check_map "$target"
    if [ "$MAP_MATCH" != 1 ]; then die 1 "the map just built does not describe $MODEL_BASENAME: $MAP_REASON"; fi
    STATE_NOTES+=("map built on this host from the local GGUF header: $target")
}

_p06_resolve_stats() {
    local cand given=0 out
    STATS_PATH=""
    if [ "$O_NO_STATS" = 1 ]; then
        log "stats: --no-stats: pinning the non-expert tensors only"
        STATE_NOTES+=("--no-stats: non-expert tensors only")
        return 0
    fi
    if [ "$MAP_PENDING" != 1 ] && [ "$MAP_IS_MOE" != 1 ]; then
        log "stats: $MODEL_BASENAME has no experts; routing statistics are not used"
        return 0
    fi
    if [ -n "$O_STATS" ]; then
        cand=$(_p06_abspath "$O_STATS")
        given=1
    else
        cand="$PIGEMMA_HOME/$DEFAULT_STATS_REL"
    fi
    if [ ! -f "$cand" ]; then
        if [ "$given" = 1 ]; then die 3 "stats file not found: $cand"; fi
        warn "no routing statistics at $DEFAULT_STATS_REL (made on the desktop: pc/trace_collect.sh, then python -m pigemma.trace stats ... --out $DEFAULT_STATS_REL; copy it to the Pi): pinning the NON-EXPERT tensors only"
        STATE_NOTES+=("no stats file ($DEFAULT_STATS_REL): non-expert tensors only")
        return 0
    fi
    if [ "$MAP_PENDING" = 1 ]; then # dry run without a map: cannot cross-check yet
        STATS_PATH=$cand
        return 0
    fi
    out=$(_p06_py check-stats "--stats=$(native_path "$cand")" "--map=$(native_path "$MAP_PATH")") || die 1 "cannot check the stats $cand"
    eval "$out"
    if [ "$STATS_OK" = 1 ]; then
        STATS_PATH=$cand
        log "stats: $cand ($STATS_REASON)"
        return 0
    fi
    if [ "$given" = 1 ]; then
        die 1 "--stats $O_STATS does not fit the map of $MODEL_BASENAME: $STATS_REASON"
    fi
    warn "default stats $DEFAULT_STATS_REL do not fit $MODEL_BASENAME ($STATS_REASON): pinning the NON-EXPERT tensors only"
    STATE_NOTES+=("default stats did not fit the map ($STATS_REASON): non-expert tensors only")
}

_p06_budget() {
    local spec="auto" src="auto" out rc=0 mapn=""
    if [ -n "$O_BUDGET" ]; then
        spec=$O_BUDGET
        src="--budget-bytes"
    elif [ "${PIN_BUDGET_BYTES:-auto}" != auto ]; then
        spec=$PIN_BUDGET_BYTES
        src="PIN_BUDGET_BYTES (pi/pi_config.env)"
    fi
    if [ "$MAP_PENDING" != 1 ] && [ -n "$MAP_PATH" ]; then mapn=$(native_path "$MAP_PATH"); fi
    out=$(_p06_py budget "--spec=$spec" "--spec-source=$src" "--avail=$AVAIL_BYTES" "--ctx=$CTX_EFF" \
        "--kv-type=$O_KV_TYPE" "--compute=$O_COMPUTE" "--headroom=$O_HEADROOM" "--file-size=$MODEL_FILE_BYTES" \
        "--map=$mapn" "--kv-fallback=$MODEL_KV_Q8_4K") || rc=$?
    if [ "$rc" != 0 ]; then die "$rc" "hot-set budget: see the pi/06 message above"; fi
    eval "$out"
    log "budget: $(_p06_fmt "$BUDGET_BYTES") ($BUDGET_SOURCE): $BUDGET_FORMULA"
}

_p06_hotset_info() { # _p06_hotset_info HOTSET_JSON -> HS_*
    local out rc=0
    out=$(_p06_py hotset-info "--hotset=$(native_path "$1")" "--gguf=$(native_path "$MODEL_PATH")" \
        "--expect-size=$MODEL_FILE_BYTES") || rc=$?
    if [ "$rc" != 0 ]; then die "$rc" "hot set $1 cannot be used (see the pi/06 message above)"; fi
    eval "$out"
    RANGES_TXT=$HS_RANGES_FILE
    if [ -n "$HS_NAME_NOTE" ]; then warn "hot set: $HS_NAME_NOTE"; fi
}

_p06_plan() {
    local rc=0
    HOTSET_JSON="$PIGEMMA_OUT/hotset.json"
    RANGES_TXT="$PIGEMMA_OUT/hotset.ranges.txt"
    local cmd=(-m pigemma.gguf_map hotset --map "$(native_path "$MAP_PATH")")
    if [ -n "$STATS_PATH" ]; then cmd+=(--stats "$(native_path "$STATS_PATH")"); fi
    cmd+=(--budget-bytes "$BUDGET_BYTES" --page-size "$PAGE" --out "$(native_path "$HOTSET_JSON")")
    if [ "$O_RANK" != count ]; then cmd+=(--rank "$O_RANK"); fi
    if [ "$O_PER_LAYER_MIN" != 0 ]; then cmd+=(--per-layer-min "$O_PER_LAYER_MIN"); fi
    if [ "$O_DRY" = 1 ]; then
        if [ "$MAP_PENDING" != 1 ] && _pg_is_uint "${MAP_NONEXPERT:-}" && [ "$MAP_NONEXPERT" -gt "$BUDGET_BYTES" ]; then
            die 3 "the hot-set budget $(_p06_fmt "$BUDGET_BYTES") is smaller than the mandatory non-expert set of $MODEL_BASENAME ($(_p06_fmt "$MAP_NONEXPERT") before page alignment): free RAM, lower CTX, or pick a smaller quant"
        fi
        printf '%s\n' "$(_p06_join_cmd "$PY" "${cmd[@]}")"
        return 0
    fi
    log "planning the hot set: $(_p06_join_cmd python3 "${cmd[@]}")"
    _p06_pyrun "${cmd[@]}" >&2 || rc=$?
    case $rc in
        0) ;;
        4) die 3 "the hot-set budget $(_p06_fmt "$BUDGET_BYTES") is smaller than the mandatory non-expert set of $MODEL_BASENAME (see the REFUSED line above): free RAM, lower CTX, or pick a smaller quant" ;;
        3) die 3 "gguf_map hotset: an input is missing (see above)" ;;
        *) die 1 "gguf_map hotset failed (exit $rc; see above)" ;;
    esac
    _p06_hotset_info "$HOTSET_JSON"
    HS_TOTAL=${HS_TOTAL:-0}
}

_p06_check_memory() { # memlock limit and the pinner's RAM guard for HS_TOTAL (or the budget in a dry run)
    local req=${HS_TOTAL:-} lim
    if ! _pg_is_uint "$req" || [ "$req" = 0 ]; then req=$BUDGET_BYTES; fi
    if ! _pg_is_uint "$req"; then return 0; fi
    lim=$(memlock_limit)
    if [ "$lim" = unlimited ]; then
        log "memlock: ulimit -l unlimited"
    elif _pg_is_uint "$lim" && [ $((lim * 1024)) -ge "$req" ]; then
        warn "ulimit -l is $lim kB: enough for this hot set ($(_p06_fmt "$req")) but not unlimited (pi/01_tune_os.sh --apply sets memlock unlimited)"
    elif _p06_cap_ipc_lock; then
        log "memlock: ulimit -l ${lim:-unknown} kB, but CAP_IPC_LOCK is effective (no limit applies)"
    elif [ "$O_BEST_EFFORT" = 1 ]; then
        warn "ulimit -l is ${lim:-unknown} kB < hot set $(_p06_fmt "$req"): --best-effort, the pinner locks in file order until the limit (partial pin, exit 2)"
        STATE_NOTES+=("--best-effort with ulimit -l ${lim:-unknown} kB: partial pin expected")
    else
        die 3 "ulimit -l is ${lim:-unknown} kB but the hot set needs $(_p06_fmt "$req") of locked memory: run pi/01_tune_os.sh --apply (TUNE_MEMLOCK=1 installs memlock unlimited) and log in again, or pass --best-effort to lock only what the limit allows"
    fi
    if [ $((req + PINNER_GUARD_BYTES)) -gt "$AVAIL_BYTES" ]; then
        die 3 "not enough available RAM to pin $(_p06_fmt "$req"): MemAvailable $(_p06_fmt "$AVAIL_BYTES") < hot set + 0.54 GB (the pinner's own guard); lower --budget-bytes"
    fi
}

_p06_prepare() { # model, pinner, map, stats, budget, plan (or --hotset), memory checks
    local akb
    _p06_resolve_model
    _p06_resolve_pinner 1
    PAGE=$(page_size)
    if ! _pg_is_uint "$PAGE" || [ "$PAGE" = 0 ]; then
        PAGE=16384
        warn "page size unknown here: aligning to 16384 (the Pi 5 kernel; the pinner re-aligns to the runtime page)"
    fi
    akb=$(mem_available_kb)
    _pg_is_uint "$akb" || akb=0
    AVAIL_BYTES=$((akb * 1024))
    if [ -n "$O_HOTSET" ]; then
        if [ -n "$O_MAP$O_STATS$O_BUDGET" ] || [ "$O_NO_STATS" = 1 ]; then
            warn "--hotset given: --map / --stats / --no-stats / --budget-bytes are ignored (the plan is fixed)"
        fi
        HOTSET_JSON=$(_p06_abspath "$O_HOTSET")
        PLAN_KIND="given (--hotset)"
        _p06_hotset_info "$HOTSET_JSON"
        BUDGET_BYTES=$HS_BUDGET
        BUDGET_SOURCE="--hotset"
        BUDGET_FORMULA="budget recorded in $HOTSET_JSON"
        MAP_PATH="" STATS_PATH=$HS_STATS
        log "hot set: $HOTSET_JSON ($(_p06_fmt "$HS_TOTAL") in $HS_N_RANGES ranges, page $HS_PAGE)"
    else
        _p06_resolve_map
        _p06_resolve_stats
        _p06_budget
        _p06_plan
    fi
    _p06_check_memory
}

# ===================================================================================== start
PINNER_PID=""
PIN_OWNED=0
PINNER_RC=""

_p06_launch_cleanup() { # stop a pinner this run started but did not hand over (interrupt, timeout, error)
    if [ "$PIN_OWNED" = 1 ] && [ -n "$PINNER_PID" ] && _p06_alive "$PINNER_PID"; then
        warn "stopping the pinner this run started (pid $PINNER_PID): the start did not complete"
        kill -TERM "$PINNER_PID" 2>/dev/null || true
        local i
        for ((i = 0; i < 100; i++)); do
            _p06_alive "$PINNER_PID" || break
            sleep 0.2
        done
    fi
    PIN_OWNED=0
}

_p06_write_state() { # _p06_write_state STATE [EXIT_CODE]
    local n nargs=() argv_json
    for n in "${STATE_NOTES[@]}"; do nargs+=("--note=$n"); done
    _pg_jarr "${PINNER_ARGV[@]}"
    argv_json=$_J
    _p06_py state-write "--out=$(native_path "$STATE_FILE")" "--state=$1" "--exit-code=${2:-}" \
        "--version=$PI06_VERSION" "--host=$(host_name)" "--emulated=$EMULATED" \
        "--override=${PIGEMMA_PROTECTED_OVERRIDE:-0}" "--pid=$PINNER_PID" "--pidfile=$(native_path "$PIDFILE")" \
        "--log=$(native_path "$PIN_LOG")" "--errlog=$(native_path "$PIN_ERR")" "--pinner=$(native_path "$PINNER_BIN")" \
        "--argv-json=$argv_json" "--model=$(native_path "$MODEL_PATH")" "--model-id=$MODEL_ID" \
        "--model-bytes=$MODEL_FILE_BYTES" "--map=${MAP_PATH:+$(native_path "$MAP_PATH")}" \
        "--stats=${STATS_PATH:+$(native_path "$STATS_PATH")}" "--hotset=$(native_path "$HOTSET_JSON")" \
        "--ranges=$(native_path "$RANGES_TXT")" "--plan=$PLAN_KIND" "--page-size=$PAGE" \
        "--budget-bytes=$BUDGET_BYTES" "--budget-source=$BUDGET_SOURCE" "--budget-formula=$BUDGET_FORMULA" \
        "--mem-available=$AVAIL_BYTES" "--kv-bytes=$KV_BYTES" "--kv-source=$KV_SOURCE" "--kv-type=$O_KV_TYPE" \
        "--ctx=$CTX_EFF" "--compute=$O_COMPUTE" "--headroom=$O_HEADROOM" \
        "--summary-file=$(native_path "$PIN_LOG")" "--max-hold=$O_MAX_HOLD" "${nargs[@]}" ||
        warn "could not write $STATE_FILE"
}

_p06_wait_summary() { # 0 summary + daemon alive | 1 the pinner exited (PINNER_RC) | 2 timeout
    local t0=$SECONDS last=$SECONDS l out i
    while :; do
        if _p06_log_has_json "$PIN_LOG"; then
            out=$(_p06_py pinner-summary "--file=$(native_path "$PIN_LOG")") || out="PS_FOUND=0"
            eval "$out"
            if [ "$PS_FOUND" = 1 ]; then
                if [ "$PS_MODE" = daemon ] && { [ "$PS_STATUS" = ok ] || [ "$PS_STATUS" = partial ]; }; then
                    sleep 0.3
                    if _p06_alive "$PINNER_PID"; then return 0; fi
                fi
                for ((i = 0; i < 75; i++)); do # refusals / interrupts exit right after the summary
                    _p06_alive "$PINNER_PID" || break
                    sleep 0.2
                done
                if _p06_alive "$PINNER_PID"; then return 0; fi
                PINNER_RC=0
                wait "$PINNER_PID" 2>/dev/null || PINNER_RC=$?
                return 1
            fi
        fi
        if ! _p06_alive "$PINNER_PID"; then
            PINNER_RC=0
            wait "$PINNER_PID" 2>/dev/null || PINNER_RC=$?
            out=$(_p06_py pinner-summary "--file=$(native_path "$PIN_LOG")") || out="PS_FOUND=0"
            eval "$out"
            return 1
        fi
        if [ $((SECONDS - t0)) -ge "$O_START_TIMEOUT" ]; then return 2; fi
        if [ $((SECONDS - last)) -ge 15 ]; then
            last=$SECONDS
            l=""
            if [ -s "$PIN_ERR" ]; then l=$(tail -n 1 "$PIN_ERR" 2>/dev/null || true); fi
            log "pinner still locking ($((SECONDS - t0)) s)${l:+: $l}"
        fi
        sleep 1
    done
}

_p06_launch() {
    local args rc wrc
    args=(--gguf "$(native_path "$MODEL_PATH")" --ranges "$(native_path "$RANGES_TXT")" --verify --json
        --pidfile "$(native_path "$PIDFILE")")
    if [ "$O_BEST_EFFORT" = 1 ]; then args+=(--best-effort); fi
    if [ -n "$O_MAX_HOLD" ] && [ "$O_MAX_HOLD" != 0 ]; then
        args+=(--hold-seconds "$O_MAX_HOLD")
        STATE_NOTES+=("the pinner expires by itself after ${O_MAX_HOLD}s (--max-hold)")
    fi
    args+=("${PINNER_PROT_ARGS[@]}")
    PINNER_ARGV=("$PINNER_BIN" "${args[@]}")
    local cmd=(nohup nice -n "$NICE_LEVEL" "$PINNER_BIN" "${args[@]}")
    if [ "$O_DRY" = 1 ]; then
        printf '%s < /dev/null > %s 2> %s &\n' "$(_p06_join_cmd "${cmd[@]}")" "$(printf '%q' "$PIN_LOG")" "$(printf '%q' "$PIN_ERR")"
        log "dry run: the pinner was not started and nothing was written (budget $(_p06_fmt "$BUDGET_BYTES"), $BUDGET_SOURCE)"
        exit 0
    fi
    mkdir -p "$PIGEMMA_OUT" || die 1 "cannot create $PIGEMMA_OUT"
    if [ -f "$PIN_LOG" ]; then mv -f "$PIN_LOG" "$PIN_LOG.prev" 2>/dev/null || true; fi
    if [ -f "$PIN_ERR" ]; then mv -f "$PIN_ERR" "$PIN_ERR.prev" 2>/dev/null || true; fi
    log "starting the pinner: $(_p06_join_cmd "${cmd[@]}") > $PIN_LOG 2> $PIN_ERR &"
    add_cleanup _p06_launch_cleanup
    "${cmd[@]}" </dev/null >"$PIN_LOG" 2>"$PIN_ERR" &
    PINNER_PID=$!
    PIN_OWNED=1
    log "pinner pid $PINNER_PID: waiting for its summary line (locking $(_p06_fmt "${HS_TOTAL:-$BUDGET_BYTES}"); timeout ${O_START_TIMEOUT}s)"
    wrc=0
    _p06_wait_summary || wrc=$?
    case $wrc in
        0)
            PIN_OWNED=0 # handed over: the daemon keeps running after this script exits
            _p06_write_state running "$PS_EXIT"
            _p06_py report-start "--state=$(native_path "$STATE_FILE")" "--json=$O_JSON" || true
            if [ "$PS_STATUS" = partial ]; then
                warn "partial pin: ${PS_FAILED:-?} ranges failed, ${PS_NOT_ATTEMPTED:-?} not attempted (reason: ${PS_REASON:-see $PIN_ERR}); the pinner holds the rest (exit 2)"
                exit 2
            fi
            exit 0
            ;;
        1)
            PIN_OWNED=0
            _p06_write_state exited "$PINNER_RC"
            if [ "$PS_FOUND" = 1 ]; then
                _p06_py report-start "--state=$(native_path "$STATE_FILE")" "--json=$O_JSON" >&2 || true
            fi
            if [ -s "$PIN_ERR" ]; then
                warn "last pinner messages ($PIN_ERR):"
                tail -n 8 "$PIN_ERR" >&2 || true
            fi
            rc=$PINNER_RC
            case $rc in
                3)
                    case ${PS_REASON,,} in
                        *protected*) die 4 "the pinner refused: ${PS_REASON}" ;;
                        *) die 3 "the pinner refused: ${PS_REASON:-see $PIN_ERR}" ;;
                    esac
                    ;;
                2) die 2 "the pinner locked only part of the hot set and exited (exit 2; see $PIN_ERR)" ;;
                0) die 1 "the pinner exited without holding the locks (exit 0; see $PIN_ERR)" ;;
                *) die 1 "the pinner failed (exit $rc${PS_REASON:+: $PS_REASON}; see $PIN_ERR)" ;;
            esac
            ;;
        *)
            warn "no summary from the pinner after ${O_START_TIMEOUT}s (still locking? slow storage?): stopping it"
            _p06_launch_cleanup
            _p06_write_state timeout ""
            die 1 "pinner start timed out after ${O_START_TIMEOUT}s (raise --start-timeout; progress in $PIN_ERR)"
            ;;
    esac
}

p06_start() {
    _p06_gate_pin
    _p06_prepare
    _p06_launch
}

# ====================================================================================== stop
p06_stop() {
    local pid raw vm t0 out
    raw=$(_p06_pidfile_pid || true)
    if ! pid=$(_p06_pinner_pid); then
        if [ -n "$raw" ]; then
            log "pidfile $PIDFILE names pid $raw, which is not a running pinner (stale): nothing to stop"
        else
            log "no pinner running (no pidfile $PIDFILE)"
        fi
        if [ -n "${PY:-}" ] && [ "$O_DRY" != 1 ] && ! is_fake && [ -f "$STATE_FILE" ]; then
            if [ "$(json_get "$STATE_FILE" state 2>/dev/null || true)" = running ]; then
                _p06_py state-update "--file=$(native_path "$STATE_FILE")" "--set=state=gone" || true
            fi
        fi
        printf 'pinner: not running\n'
        exit 0
    fi
    vm=$(_p06_vmlck_bytes "$pid")
    if is_fake || [ "$O_DRY" = 1 ]; then
        printf 'would send SIGTERM to pinner pid %s (VmLck %s)%s\n' "$pid" "$(_p06_fmt "$vm")" \
            "$(is_fake && printf ' - fake system root, no signal sent' || true)"
        exit 0
    fi
    log "stopping pinner pid $pid (VmLck $(_p06_fmt "$vm")): SIGTERM"
    kill -TERM "$pid" 2>/dev/null || die 1 "could not signal pinner pid $pid (owned by another user?)"
    t0=$SECONDS
    while _p06_pid_is_pinner "$pid"; do
        if [ $((SECONDS - t0)) -ge "$O_STOP_TIMEOUT" ]; then
            die 1 "pinner pid $pid did not exit within ${O_STOP_TIMEOUT}s after SIGTERM (see $PIN_ERR); nothing else was signalled"
        fi
        sleep 0.2
    done
    if [ -n "${PY:-}" ] && [ -f "$STATE_FILE" ]; then
        out="$(_p06_stamp)"
        _p06_py state-update "--file=$(native_path "$STATE_FILE")" "--set=state=stopped" \
            "--set=stopped_at=\"$out\"" "--set=stop_wait_s=$((SECONDS - t0))" "--set=vm_lck_bytes_at_stop=$vm" || true
    fi
    printf 'pinner: stopped (pid %s; %s of locked memory released)\n' "$pid" "$(_p06_fmt "$vm")"
    exit 0
}

# ===================================================================================== status
p06_status() {
    local pid="" raw vm=0 running=0 why="" gguf="" ranges="" vfile="" vrc="" notes=()
    raw=$(_p06_pidfile_pid || true)
    if pid=$(_p06_pinner_pid); then
        running=1
        vm=$(_p06_vmlck_bytes "$pid")
    elif [ -n "$raw" ]; then
        why="pidfile $PIDFILE names pid $raw, which is not a running pinner"
        pid=""
    else
        why="no pidfile $PIDFILE"
    fi
    if [ -f "$STATE_FILE" ]; then
        gguf=$(json_get "$STATE_FILE" model.path 2>/dev/null || true)
        ranges=$(json_get "$STATE_FILE" ranges 2>/dev/null || true)
    fi
    if [ -n "$O_MODEL" ]; then
        _p06_resolve_model
        gguf=$MODEL_PATH
    fi
    if [ -n "$O_HOTSET" ]; then ranges=$(_p06_abspath "$O_HOTSET"); ranges="${ranges%.json}.ranges.txt"; fi
    if [ "$running" = 1 ] && [ -n "$gguf" ] && [ -n "$ranges" ]; then
        if is_fake; then
            notes+=("--note=fake system root: the mincore check (pinner --verify-only) is skipped")
        elif ! _p06_resolve_pinner 0; then
            notes+=("--note=pinner binary not found ($PINNER_BIN): residency not measured")
        elif [ ! -f "$gguf" ] || [ ! -f "$ranges" ]; then
            notes+=("--note=model or ranges file missing ($gguf, $ranges): residency not measured")
        else
            vfile="$PIGEMMA_OUT/.pi06_verify.$$.json"
            add_cleanup "rm -f '$vfile' '$vfile.err'"
            vrc=0
            "$PINNER_BIN" --gguf "$(native_path "$gguf")" --ranges "$(native_path "$ranges")" --verify-only --json \
                >"$vfile" 2>"$vfile.err" || vrc=$?
        fi
    fi
    _p06_py status-report "--state=$(native_path "$STATE_FILE")" "--running=$running" "--pid=$pid" "--vmlck=$vm" \
        "--verify=${vfile:+$(native_path "$vfile")}" "--verify-rc=$vrc" "--pidfile=$(native_path "$PIDFILE")" \
        "--why=$why" "--json=$O_JSON" "${notes[@]}" || die 1 "status report failed"
    if [ "$running" = 1 ]; then exit 0; fi
    exit 3
}

# ========================================================================================= ab
AB_LABELS=() AB_ROUNDS=() AB_RCS=() AB_LASTS=() AB_FILES=() AB_RES_BEFORE=() AB_RES_AFTER=() AB_PIN_STATES=()
AB_STARTED=() AB_ENDED=() AB_ALL_FILES=()

_p06_verify_to() { # _p06_verify_to FILE: hot-set residency (pinner --verify-only --json) into FILE; 1 on failure
    local f=$1 rc=0
    if [ ! -f "$PINNER_BIN" ] || [ ! -f "$RANGES_TXT" ] || [ ! -f "$MODEL_PATH" ]; then return 1; fi
    "$PINNER_BIN" --gguf "$(native_path "$MODEL_PATH")" --ranges "$(native_path "$RANGES_TXT")" --verify-only --json \
        >"$f" 2>"$f.err" || rc=$?
    if [ "$rc" != 0 ]; then
        warn "pinner --verify-only failed (exit $rc, see $f.err): residency not recorded"
        return 1
    fi
    return 0
}

_p06_ab_args() { # _p06_ab_args LABEL -> array _AB (the pi/05_run.sh arguments)
    _AB=(stream-bench --model "$MODEL_PATH" --no-calibrate)
    if [ "$O_WITH_HE" != 1 ]; then _AB+=(--no-he); fi
    if [ -n "$O_N" ]; then _AB+=(-n "$O_N"); fi
    _AB+=(--out "$AB_RESULTS_DIR")
    _AB+=("${O_RUN_ARGS[@]}")
    if [ "$1" = pinned ]; then
        _AB+=(--pin "--pin-arg=--hotset" "--pin-arg=$HOTSET_JSON" "--pin-arg=--start-timeout"
            "--pin-arg=$O_START_TIMEOUT" "--pin-arg=--pinner" "--pin-arg=$PINNER_BIN")
        if [ "$O_BEST_EFFORT" = 1 ]; then _AB+=("--pin-arg=--best-effort"); fi
        if [ -n "$O_MAX_HOLD" ]; then _AB+=("--pin-arg=--max-hold" "--pin-arg=$O_MAX_HOLD"); fi
        if [ "$O_ALLOW_PROT" = 1 ]; then _AB+=("--pin-arg=--allow-protected-host"); fi
    fi
    _AB+=("${O_PASS[@]}")
}

_p06_ab_run() { # _p06_ab_run LABEL ROUND
    local label=$1 round=$2 tag rc=0 out started ended since last="" pstate="" vb="" va="" files="" l
    tag="r${round}_${label}"
    _p06_ab_args "$label"
    if l=$(_p06_pinner_pid); then
        die 4 "a pinner is running before the $label run (pid $l): the A/B needs a clean start; stop it first"
    fi
    if _p06_verify_to "$AB_DIR/$tag.verify_before.json"; then vb="$AB_DIR/$tag.verify_before.json"; fi
    banner "pi/06 ab: round $round, $label run" "bash pi/05_run.sh $(_p06_join_cmd "${_AB[@]}")"
    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    since=$(date +%s)
    out=$(bash "$PI_DIR/05_run.sh" "${_AB[@]}") || rc=$?
    ended=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [ -f "$PIGEMMA_OUT/last_stream-bench.json" ]; then
        local mt
        mt=$(stat -c %Y "$PIGEMMA_OUT/last_stream-bench.json" 2>/dev/null || printf '0')
        if _pg_is_uint "$mt" && [ "$mt" -ge "$since" ]; then
            last="$AB_DIR/$tag.last_stream-bench.json"
            cp -f "$PIGEMMA_OUT/last_stream-bench.json" "$last" || last=""
        fi
    fi
    if [ "$label" = pinned ] && [ -f "$STATE_FILE" ]; then
        pstate="$AB_DIR/$tag.pin_state.json"
        cp -f "$STATE_FILE" "$pstate" || pstate=""
    fi
    if _p06_pinner_pid >/dev/null; then
        warn "a pinner is still running after the $label run: stopping it (pi/06_pin_hotset.sh stop)"
        bash "$PI_DIR/06_pin_hotset.sh" stop >&2 || warn "pi/06 stop failed: check $PIDFILE"
    fi
    if _p06_verify_to "$AB_DIR/$tag.verify_after.json"; then va="$AB_DIR/$tag.verify_after.json"; fi
    while IFS= read -r l || [ -n "$l" ]; do
        l=${l%$'\r'}
        case $l in
            *.json)
                files+="${files:+,}$l"
                AB_ALL_FILES+=("$l")
                ;;
        esac
    done <<<"$out"
    AB_LABELS+=("$label") AB_ROUNDS+=("$round") AB_RCS+=("$rc") AB_LASTS+=("$last") AB_FILES+=("$files")
    AB_RES_BEFORE+=("$vb") AB_RES_AFTER+=("$va") AB_PIN_STATES+=("$pstate") AB_STARTED+=("$started")
    AB_ENDED+=("$ended")
    case $rc in
        0) log "$label run (round $round): ok" ;;
        3 | 4)
            if [ "${#AB_LABELS[@]}" = 1 ]; then
                die "$rc" "the first A/B run was refused by pi/05_run.sh (exit $rc): nothing was measured"
            fi
            warn "$label run (round $round) refused by pi/05_run.sh (exit $rc)"
            ;;
        *) warn "$label run (round $round) failed (pi/05 exit $rc); its envelope carries a 'failed:' note" ;;
    esac
}

_p06_ab_cleanup() { # an A/B never leaves a pinner behind (any live pinner here was started by its pinned run)
    if [ "$AB_STARTED_RUNS" = 1 ] && _p06_pinner_pid >/dev/null 2>&1; then
        warn "A/B interrupted: stopping the pinner its pinned run started"
        bash "$PI_DIR/06_pin_hotset.sh" stop >&2 || true
    fi
}

p06_ab() {
    local r i order stamp rc=0 args=() pp
    _p06_gate_pin
    if [ ! -f "$PI_DIR/05_run.sh" ]; then die 3 "ab needs pi/05_run.sh next to this script ($PI_DIR)"; fi
    _p06_need_uint "--rounds" "$O_ROUNDS" 1
    # hold the bench lock across both runs (each pi/05 run inherits it), unless already held / disabled
    if [ "$O_DRY" != 1 ] && [ "$O_NO_LOCK" != 1 ] && [ -z "${PIGEMMA_LOCK_HELD:-}" ]; then
        log "taking the bench lock for the whole A/B (python3 -m pigemma.paths lock --hold)"
        pp=$(_p06_pp)
        PYTHONPATH="$pp" PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "$PY" -m pigemma.paths --log-level WARNING lock --hold \
            --min-free-gb 1 -- bash "$PI_DIR/06_pin_hotset.sh" "${ORIG_ARGS[@]}" || rc=$?
        exit "$rc"
    fi
    _p06_prepare
    if [ -n "$O_RESULTS_DIR" ]; then
        AB_RESULTS_DIR=$(_p06_abspath "$O_RESULTS_DIR")
    elif [ "$EMULATED" = 1 ]; then
        AB_RESULTS_DIR="$PIGEMMA_HOME/results/pi-emu"
    else
        AB_RESULTS_DIR="$PIGEMMA_HOME/results/pi"
    fi
    if [ -n "$O_AB_OUT" ]; then AB_OUT=$(_p06_abspath "$O_AB_OUT"); else AB_OUT="$AB_RESULTS_DIR/pin_ab.json"; fi
    stamp=$(_p06_stamp)
    if [ "$O_DRY" = 1 ]; then
        for ((r = 1; r <= O_ROUNDS; r++)); do
            if [ $((r % 2)) = 1 ]; then order="unpinned pinned"; else order="pinned unpinned"; fi
            for i in $order; do
                printf '%s\n' "$(_p06_join_cmd "$PINNER_BIN" --gguf "$MODEL_PATH" --ranges "$RANGES_TXT" --verify-only --json)"
                _p06_ab_args "$i"
                printf '%s\n' "$(_p06_join_cmd bash "$PI_DIR/05_run.sh" "${_AB[@]}")"
            done
        done
        log "dry run: would write $AB_OUT (and ${AB_OUT%.json}_$stamp.json); nothing was run"
        exit 0
    fi
    AB_DIR="$PIGEMMA_OUT/pin_ab_$stamp"
    mkdir -p "$AB_DIR" "$AB_RESULTS_DIR" || die 1 "cannot create $AB_DIR / $AB_RESULTS_DIR"
    AB_STARTED_RUNS=1
    add_cleanup _p06_ab_cleanup
    for ((r = 1; r <= O_ROUNDS; r++)); do
        if [ $((r % 2)) = 1 ]; then order="unpinned pinned"; else order="pinned unpinned"; fi
        for i in $order; do
            _p06_ab_run "$i" "$r"
        done
    done
    for ((i = 0; i < ${#AB_LABELS[@]}; i++)); do
        args+=("--run-label=${AB_LABELS[$i]}" "--run-round=${AB_ROUNDS[$i]}" "--run-rc=${AB_RCS[$i]}"
            "--run-last=${AB_LASTS[$i]:+$(native_path "${AB_LASTS[$i]}")}" "--run-files=${AB_FILES[$i]}"
            "--run-res-before=${AB_RES_BEFORE[$i]:+$(native_path "${AB_RES_BEFORE[$i]}")}"
            "--run-res-after=${AB_RES_AFTER[$i]:+$(native_path "${AB_RES_AFTER[$i]}")}"
            "--run-pin-state=${AB_PIN_STATES[$i]:+$(native_path "${AB_PIN_STATES[$i]}")}"
            "--run-started=${AB_STARTED[$i]}" "--run-ended=${AB_ENDED[$i]}")
    done
    _pg_jarr "${ORIG_ARGS[@]}"
    rc=0
    _p06_py ab-report "--out=$(native_path "$AB_OUT")" "--out=$(native_path "${AB_OUT%.json}_$stamp.json")" \
        "--version=$PI06_VERSION" "--host=$(host_name)" "--emulated=$EMULATED" \
        "--override=${PIGEMMA_PROTECTED_OVERRIDE:-0}" "--argv-json=$_J" "--model=$(native_path "$MODEL_PATH")" \
        "--model-id=$MODEL_ID" "--model-bytes=$MODEL_FILE_BYTES" "--map=${MAP_PATH:+$(native_path "$MAP_PATH")}" \
        "--stats=${STATS_PATH:+$(native_path "$STATS_PATH")}" "--hotset=$(native_path "$HOTSET_JSON")" \
        "--ranges=$(native_path "$RANGES_TXT")" "--budget-bytes=$BUDGET_BYTES" "--budget-source=$BUDGET_SOURCE" \
        "--budget-formula=$BUDGET_FORMULA" "--bpt-worst=${MAP_BPT_WORST:-}" "--rounds=$O_ROUNDS" "${args[@]}" || rc=$?
    for i in "${AB_ALL_FILES[@]}"; do printf '%s\n' "$i"; done
    if [ -f "$AB_OUT" ]; then
        printf '%s\n' "$AB_OUT"
        printf '%s\n' "${AB_OUT%.json}_$stamp.json"
    fi
    case $rc in
        0) log "A/B done: $AB_OUT (per-run copies in $AB_DIR)" ;;
        1) warn "A/B finished with a failed run: $AB_OUT says which (per-run copies in $AB_DIR)" ;;
        *) die "$rc" "could not write the A/B summary (exit $rc)" ;;
    esac
    exit "$rc"
}

# =================================================================================== arguments
SUBCMD=""
O_MODEL="" O_MAP="" O_STATS="" O_NO_STATS=0 O_BUDGET="" O_HOTSET="" O_KV_TYPE="q8_0" O_CTX=""
O_COMPUTE=$DEF_COMPUTE_BYTES O_HEADROOM=$DEF_HEADROOM_BYTES O_RANK="count" O_PER_LAYER_MIN=0 O_PINNER=""
O_START_TIMEOUT=1800 O_STOP_TIMEOUT=60 O_BEST_EFFORT=0 O_ALLOW_PROT=0 O_JSON=0 O_DRY=0 O_MAX_HOLD=""
O_ROUNDS=1 O_N="" O_WITH_HE=0 O_RESULTS_DIR="" O_AB_OUT="" O_NO_LOCK=0
O_RUN_ARGS=()
O_PASS=()
ORIG_ARGS=()
PINNER_ARGV=()
PINNER_PROT_ARGS=()
AB_STARTED_RUNS=0

_p06_val() { # _p06_val OPTION "$@" -> dies when the option has no value
    if [ $# -lt 3 ]; then die 1 "option $1 needs a value (see --help)"; fi
}

p06_parse_args() {
    local a
    while [ $# -gt 0 ]; do
        a=$1
        case $a in
            --*=*)
                shift
                set -- "${a%%=*}" "${a#*=}" "$@"
                continue
                ;;
        esac
        case $a in
            -h | --help) usage; exit 0 ;;
            --version) printf 'pi/06_pin_hotset.sh %s\n' "$PI06_VERSION"; exit 0 ;;
            start | stop | status | ab)
                if [ -n "$SUBCMD" ]; then die 1 "more than one subcommand: '$SUBCMD' and '$a'"; fi
                SUBCMD=$a
                shift
                continue
                ;;
            --no-stats) O_NO_STATS=1; shift; continue ;;
            --best-effort) O_BEST_EFFORT=1; shift; continue ;;
            --allow-protected-host) O_ALLOW_PROT=1; shift; continue ;;
            --json) O_JSON=1; shift; continue ;;
            --dry-run) O_DRY=1; shift; continue ;;
            --with-he) O_WITH_HE=1; shift; continue ;;
            --no-lock) O_NO_LOCK=1; shift; continue ;;
            --)
                shift
                O_PASS=("$@")
                break
                ;;
        esac
        case $a in
            --model | -m | --map | --stats | --budget-bytes | --hotset | --kv-type | -c | --ctx | --compute-bytes | \
                --headroom-bytes | --rank | --per-layer-min | --pinner | --start-timeout | --stop-timeout | \
                --rounds | -n | --results-dir | --out | --run-arg | --max-hold)
                _p06_val "$a" "$@"
                ;;
            -*) die 1 "unknown option '$a' (see --help)" ;;
            *) die 1 "unexpected argument '$a' (subcommands: start, stop, status, ab; see --help)" ;;
        esac
        case $a in
            --model | -m) O_MODEL=$2 ;;
            --map) O_MAP=$2 ;;
            --stats) O_STATS=$2 ;;
            --budget-bytes) O_BUDGET=$2 ;;
            --hotset) O_HOTSET=$2 ;;
            --kv-type) O_KV_TYPE=$2 ;;
            -c | --ctx) O_CTX=$2 ;;
            --compute-bytes) O_COMPUTE=$2 ;;
            --headroom-bytes) O_HEADROOM=$2 ;;
            --rank) O_RANK=$2 ;;
            --per-layer-min) O_PER_LAYER_MIN=$2 ;;
            --pinner) O_PINNER=$2 ;;
            --start-timeout) O_START_TIMEOUT=$2 ;;
            --stop-timeout) O_STOP_TIMEOUT=$2 ;;
            --rounds) O_ROUNDS=$2 ;;
            -n) O_N=$2 ;;
            --results-dir) O_RESULTS_DIR=$2 ;;
            --out) O_AB_OUT=$2 ;;
            --run-arg) O_RUN_ARGS+=("$2") ;;
            --max-hold) O_MAX_HOLD=$2 ;;
        esac
        shift 2
    done
}

p06_validate_args() {
    local t ok=0
    if [ -z "$SUBCMD" ]; then die 1 "missing subcommand: start, stop, status or ab (see --help)"; fi
    for t in $KV_TYPES; do
        if [ "$O_KV_TYPE" = "$t" ]; then ok=1; fi
    done
    if [ "$ok" != 1 ]; then die 1 "--kv-type must be one of: $KV_TYPES (got '$O_KV_TYPE')"; fi
    case $O_RANK in count | weight) ;; *) die 1 "--rank must be count or weight (got '$O_RANK')" ;; esac
    _p06_need_uint "--per-layer-min" "$O_PER_LAYER_MIN" 0
    _p06_need_uint "--start-timeout" "$O_START_TIMEOUT" 1
    _p06_need_uint "--stop-timeout" "$O_STOP_TIMEOUT" 1
    _p06_need_uint "--rounds" "$O_ROUNDS" 1
    if [ -n "$O_N" ]; then _p06_need_uint "-n" "$O_N" 1; fi
    if [ -n "$O_CTX" ]; then _p06_need_uint "-c/--ctx" "$O_CTX" 1; fi
    if [ -n "$O_MAX_HOLD" ]; then
        _p06_need_uint "--max-hold" "$O_MAX_HOLD" 0
        if [ "$O_MAX_HOLD" -gt 31536000 ]; then die 1 "--max-hold must be <= 31536000 s (one year)"; fi
    fi
    if [ -n "$O_BUDGET" ] && ! [[ $O_BUDGET =~ ^[0-9][0-9_,]*([.][0-9]+)?([eE][+-]?[0-9]+)?[[:space:]]*([KkMmGgTt]i?[Bb]|[Bb])?$ ]]; then
        die 1 "--budget-bytes: not a byte count '$O_BUDGET' (examples: 5200000000, 5.2e9, 5.2GB, 2GiB)"
    fi
    for t in "$O_COMPUTE" "$O_HEADROOM"; do
        if ! [[ $t =~ ^[0-9][0-9_,]*([.][0-9]+)?([eE][+-]?[0-9]+)?([KkMmGgTt]i?[Bb]|[Bb])?$ ]]; then
            die 1 "--compute-bytes / --headroom-bytes: not a byte count '$t'"
        fi
    done
    case $SUBCMD in
        start | ab) ;;
        *)
            if [ -n "$O_MAP$O_STATS$O_BUDGET" ] || [ "$O_NO_STATS" = 1 ]; then
                warn "$SUBCMD ignores --map / --stats / --no-stats / --budget-bytes"
            fi
            ;;
    esac
    if [ "$SUBCMD" != ab ] && { [ "${#O_RUN_ARGS[@]}" -gt 0 ] || [ "${#O_PASS[@]}" -gt 0 ] || [ -n "$O_AB_OUT$O_RESULTS_DIR$O_N" ]; }; then
        die 1 "--run-arg / -- / --out / --results-dir / -n belong to 'ab' only"
    fi
}

# ======================================================================================== main
p06_main() {
    ORIG_ARGS=("$@")
    p06_parse_args "$@"
    p06_validate_args
    case $SUBCMD in
        stop | status)
            local q=${PIGEMMA_QUIET:-0}
            PIGEMMA_QUIET=1
            pigemma_init
            PIGEMMA_QUIET=$q
            ;;
        *) pigemma_init ;;
    esac
    trap 'exit 130' INT
    trap 'exit 143' TERM
    PIDFILE="$PIGEMMA_OUT/pinner.pid"
    PIN_LOG="$PIGEMMA_OUT/pinner.log"
    PIN_ERR="$PIGEMMA_OUT/pinner.stderr.log"
    STATE_FILE="$PIGEMMA_OUT/pin_state.json"
    CTX_EFF=${O_CTX:-$CTX}
    EMULATED=0
    if is_emulated; then EMULATED=1; fi
    PY=""
    if ! PY=$(pigemma_python); then
        PY=""
        if [ "$SUBCMD" != stop ]; then
            die 3 "no usable Python >= 3.10 (PIGEMMA_PYTHON, python3, python): sudo apt-get install -y python3"
        fi
    fi
    case $SUBCMD in
        start) p06_start ;;
        stop) p06_stop ;;
        status) p06_status ;;
        ab) p06_ab ;;
    esac
}

p06_main "$@"
