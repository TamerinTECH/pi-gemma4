#!/usr/bin/env bash
# pi/05_run.sh - run, benchmark and serve Gemma 4 on a Raspberry Pi 5 with the flags of the feasibility
# verdict (pigemma.feasibility through pi/lib/common.sh). Part of pi-gemma4 (docs/design/manifest.json).
#
# Usage (the first line is the manifest's usage string; docs/PI_RUNBOOK.md greps for it):
#   pi/05_run.sh bench|stream-bench|chat|server [--model <gguf>] [--draft <gguf>] [--variants default,norepack] [--pin] [-p 'text']
#   pi/05_run.sh --help            (full option list)
#
# What each subcommand does (all of them: assert_safe_host -> assert_not_busy -> require_verdict_ok
# --allow-slow -> model selection -> RAM gate -> run):
#   bench         python3 -m pigemma.bench matrix per load variant, i.e. exactly
#                   llama-bench -m MODEL -t 2,3,4 -p 64 -n 32 -r 3 -o json [--repack 0] [-lm ...] [verdict KV/batch flags]
#                 -> one results envelope per (variant, threads, pp|tg) in results/pi/ (the desktop E0 envelope),
#                 the raw llama-bench JSON copied to results/pi/bench_<model>_<variant>_<UTC>.json.
#                 The default (repack) variant is refused whenever FITS_REPACKED is false: llama.cpp would
#                 allocate an anonymous copy of the whole file (14 GB for the 26B Q4_0).
#   stream-bench  python3 -m pigemma.bench completion, i.e.
#                   llama-completion -m MODEL -f data/calibration/en_tech_moe_streaming.txt -n 64 --temp 0 <verdict flags>
#                 which samples /proc/<pid>/io read_bytes, majflt, RssAnon/RssFile while it runs; `vmstat 1` is
#                 logged next to it. -> read_bytes_per_token, majflt_per_token, tok/s. Then a short Hebrew chat
#                 sample (data/prompts/gen_prompts.json gen-he-explain-01). Both outputs are printed.
#   chat          llama-cli -m MODEL <verdict flags> [-md DRAFT --spec-type T]       (interactive; -p 'text' = one turn)
#   server        llama-server -m MODEL <verdict flags> --host 0.0.0.0 --port 8080 [-md DRAFT --spec-type T]
#   --pin         starts pi/06_pin_hotset.sh first (and stops it at exit when this script started it).
#
# Model and flags:
#   * The model is --model (GGUF path or catalog id), else QUANT_OVERRIDE (pi_config.env) within the
#     recommended family, else the verdict's recommendation; when the recommended file is not downloaded,
#     the best-ranked runnable verdict row whose file IS in MODEL_DIR is used (a note says so). Files are
#     looked up only in MODEL_DIR (<dir>/<file>, <dir>/<basename>, <dir>/<org>__<repo>/<file>), exactly the
#     candidates pigemma.feasibility checks. A row the verdict blocked only for lack of download space
#     (NO_DISK) is re-enabled when its file is already present.
#   * Flags come from that model's row in pi/out/feasibility.json (`-nr` whenever not FITS_REPACKED),
#     passed through safe_llama_flags (no mlock on protected hosts). On a protected host run with
#     PIGEMMA_I_KNOW=1, `-nr` is also forced for chat/server/stream-bench and the bench default (repack)
#     variant is refused unless --allow-repack: the anonymous repack copy is no more reclaimable for the
#     host's services than mlocked pages are (design decision, see below).
#   * RAM gate (exit 3): effective MemAvailable >= file bytes + 0.5 GB (model fits in RAM: FITS_*, runnable
#     NEVER_GOOD_SPEED) or >= non-expert bytes + 0.5 GB (MoE streaming), plus the draft's bytes; a live
#     pinner's VmLck is credited (those pages already hold weights).
#
# Outputs:
#   stdout                       one line per written results envelope (bench.py's path line); with --dry-run
#                                the exact command lines instead (nothing runs, no results are written)
#   results/pi/                  envelopes (experiment E8; results/pi-emu/ when PIGEMMA_EMULATED=1), raw/ logs,
#                                bench_<model>_<variant>_<UTC>.json (raw llama-bench JSON)
#   results/pi/calibration.json  {llama_efficiency:{value, source, from:[tg tok/s, bytes/token, mem_bw GB/s]}, ...}
#                                = tg x (nonexpert_bytes_per_token + expert_bytes_per_token_worst) / mem_bw,
#                                written after an in-RAM bench / stream-bench (not when emulated, not with
#                                --no-calibrate, not when the run read > 2% of bytes/token from storage);
#                                mem_bw from results/pi/io_mem.json (pi/03), else the verdict's device profile
#                                (then the source says 'assumed'). Read by pigemma.feasibility / pigemma.cache_sim.
#                                --calibration-out FILE writes it elsewhere (tests, other layouts); previous
#                                values are kept under "history".
#   pi/out/last_run.json         + pi/out/last_<subcommand>.json: the run's model, verdict, flags, result files and
#                                key metrics (pi/06_pin_hotset.sh ab reads these)
#   every envelope gets extra.pi05 {verdict, mode, flags, ram gate, thermals, pinner, vmstat, ...} and notes for a
#   protected-host override (PIGEMMA_I_KNOW=1), removed mlock, emulation and pinning.
#
# Exit codes (pi/lib/common.sh convention): 0 ok | 1 usage / internal error / a run failed (its envelope is
# still written with a 'failed: ...' note) | 3 not viable or precondition failed (verdict, RAM gate, model,
# draft or binary missing, size mismatch, every bench variant refused) | 4 refused (protected or busy host,
# bench lock held, port in use, --pin where mlock is not allowed). A slow-streaming verdict (2) is accepted.
#
# Safety (docs/design/FACTS.md): never stops, restarts or kills anything it did not start (it only stops its
# own vmstat sampler and, with --pin, the pinner it started through pi/06); llama.cpp runs at nice -n
# $NICE_LEVEL; one model-loading job at a time (results/.bench.lock via pigemma.paths / pigemma.bench; chat only
# checks the lock because an interactive session cannot be supervised by the lock holder); no apt, no sudo,
# no config edits, no reboots.
#
# Design decisions (also reported as open issues):
#   * bench uses `python3 -m pigemma.bench matrix` once per variant instead of a bare llama-bench +
#     `pigemma.bench parse-bench`: same llama-bench command line and the same envelope builder
#     (bench_rows_to_fields), plus the bench lock, the idle check, failure envelopes and raw logs.
#     The raw JSON is still copied to results/pi/bench_<model>_<variant>_<UTC>.json.
#   * The envelope variant name is derived from the effective flags with bench.py's vocabulary
#     (norepack, nommap, mlock, kvq8 joined by '+'); flags that no variant covers go to --flags.
#   * -p/-n mean prompt/generated tokens for bench (llama-bench spelling) and prompt text / tokens to
#     generate for stream-bench and chat; server rejects them.
#   * --draft is accepted for chat and server only: -md is not wired into llama-bench/llama-completion at
#     a7b94df. The spec type is explicit (draft-mtp for MTP sidecars): MTP only drafts when
#     --spec-type draft-mtp is given (FACTS.md 2026-10-05).
#   * stream-bench's vmstat log covers the whole llama-completion run (bench lock, idle check, load, prompt,
#     generation); the per-token numbers come from bench.py's /proc/<pid>/io sampling of generation only.
#     vmstat is skipped under PIGEMMA_FAKE_PROC (it would describe the real host, not the fixture).
#   * chat only checks the bench lock (an interactive session is not supervised by a lock holder);
#     server holds it for its whole lifetime via `python3 -m pigemma.paths lock --hold`.
#
# Lint: bash -n pi/05_run.sh (shellcheck when installed). Tests: tests/test_pi05_run.py.

set -euo pipefail

PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIGEMMA_SCRIPT="05_run.sh"
# shellcheck source=lib/common.sh
. "$PI_DIR/lib/common.sh"

PI05_VERSION="1.0.0"
PI05_USAGE_LINE="pi/05_run.sh bench|stream-bench|chat|server [--model <gguf>] [--draft <gguf>] [--variants default,norepack] [--pin] [-p 'text']"
MEM_MARGIN=500000000
DEFAULT_PROMPT_REL="data/calibration/en_tech_moe_streaming.txt"
GEN_PROMPTS_REL="data/prompts/gen_prompts.json"

usage() {
    cat <<EOF
pi/05_run.sh $PI05_VERSION - run Gemma 4 on a Raspberry Pi 5 with the feasibility verdict's llama.cpp flags.

Usage:
  $PI05_USAGE_LINE

Subcommands:
  bench          llama-bench -t 2,3,4 -p 64 -n 32 -r 3 -o json per load variant (via python3 -m pigemma.bench
                 matrix) -> results envelopes + results/pi/bench_<model>_<variant>_<UTC>.json
  stream-bench   llama-completion -n 64 --temp 0 -f $DEFAULT_PROMPT_REL wrapped by
                 python3 -m pigemma.bench completion: read_bytes / majflt per token, RssAnon/RssFile, vmstat 1,
                 tok/s; then a Hebrew chat sample; prints the EN and HE outputs
  chat           llama-cli -m MODEL <verdict flags> [-md DRAFT]      (interactive; -p 'text' = one turn, then exit)
  server         llama-server -m MODEL <verdict flags> --host 0.0.0.0 --port 8080 [-md DRAFT]

Common options:
  --model GGUF|ID        model file or catalog id (default: the verdict's model, see below)
  --want 26b|31b|ID      restrict the verdict to a family or model (pigemma.feasibility --want)
  -t, --threads N        llama.cpp threads for the run flags (THREADS, default 4); bench: a comma list
                         (e.g. -t 2,3,4) is the llama-bench thread list
  -c, --ctx N            context size (CTX, default 4096)
  --flags "FLAGS"        extra llama.cpp flags appended to every command (repeatable; also after --)
  --pin                  start pi/06_pin_hotset.sh first (needs mlock: refused on protected hosts)
  --pin-arg ARG          extra argument for 'pi/06_pin_hotset.sh start' (repeatable)
  --allow-repack         protected host only: allow the anonymous repack copy (default there: -nr only)
  --out DIR              results directory (default results/pi; results/pi-emu when PIGEMMA_EMULATED=1)
  --experiment E         results experiment id (default E8)
  --sim FILE             results/h1/sim.json for the verdict (pigemma.feasibility --sim)
  --feasibility-arg ARG  extra pigemma.feasibility argument (repeatable, e.g. --feasibility-arg=--io-mem
                         --feasibility-arg=none)
  --idle-check-s S       background-CPU check before each run (default 10; 0 disables)
  --idle-max-pct P       idle threshold in percent (default 5)
  --timeout S            timeout per llama.cpp run (bench, stream-bench)
  --no-calibrate         do not write results/pi/calibration.json
  --calibration-out F    calibration file to write (default results/pi/calibration.json, which
                         pigemma.feasibility / cache_sim read with --calibration auto)
  --io-mem FILE          io_mem.json for the calibration's memory bandwidth (default results/pi/io_mem.json)
  --dry-run              run the gates, print the plan and the exact commands; run nothing, write no results
  -h, --help             this text;  --version

bench options:
  --variants LIST        load variants (default: norepack, plus default when FITS_REPACKED and allowed);
                         names: default norepack nommap mlock kvq8 hybrid_negctrl, combined with '+'
  --bench-threads LIST   llama-bench thread list (default 2,3,4; counts above nproc are dropped)
  --pp N | -p N          prompt tokens (default 64)
  --tg N | -n N          generated tokens (default 32)
  -r, --reps N           repetitions (default 3)
  --no-verdict-flags     only the variant flags (+ --flags), not the verdict's -lm / -fa / KV / batch flags

stream-bench options:
  -f, --prompt-file F    prompt file (default $DEFAULT_PROMPT_REL)
  -p, --prompt TEXT      prompt text instead of the file
  -n N                   tokens to generate (default 64)
  --chat                 wrap the prompt in the Gemma 4 chat template
  --no-he                skip the Hebrew sample
  --he-prompt-id ID      Hebrew prompt id in $GEN_PROMPTS_REL (default gen-he-explain-01)
  --he-n N               tokens for the Hebrew sample (default 48)

chat / server options:
  --draft GGUF|ID|auto   speculative-decoding draft (-md); auto = the catalog MTP sidecar of the model
  --spec-type T|none     --spec-type (default: draft-mtp / draft-dflash from the draft; none = let llama.cpp infer)
  --draft-n-max N        --spec-draft-n-max N
  -p TEXT | -f FILE      chat: one turn (adds -st), then exit;  -n N: chat tokens to generate
  --host H, --port P     server address (default 0.0.0.0:8080; the server is reachable from the LAN)

Model selection: --model > QUANT_OVERRIDE (pi/pi_config.env) > the verdict's recommendation; when the
recommended file is not in MODEL_DIR the best runnable verdict row that IS downloaded is used.
Flags: the model's row in pi/out/feasibility.json ('-nr' whenever not FITS_REPACKED), mlock removed on
protected hosts. RAM gate: MemAvailable >= file bytes + 0.5 GB (fits) or non-expert bytes + 0.5 GB (streaming).

Examples:
  pi/05_run.sh bench --variants norepack
  pi/05_run.sh bench --variants default,norepack
  pi/05_run.sh stream-bench
  pi/05_run.sh stream-bench --pin
  pi/05_run.sh chat -p 'Explain a mixture-of-experts model in two sentences.'
  pi/05_run.sh server --draft auto
  PIGEMMA_I_KNOW=1 pi/05_run.sh stream-bench --model 26b-ud-q3_k_m

Outputs: stdout = written results envelopes (one path per line); results/pi/calibration.json after an in-RAM
run; pi/out/last_run.json and pi/out/last_<subcommand>.json.
Exit codes: 0 ok, 1 usage / internal error / a run failed (envelope still written), 3 not viable or
precondition failed (verdict, RAM, model, binary), 4 refused (protected or busy host, bench lock, port in use).
EOF
}

# ================================================================================== small helpers
_p05_abspath() { # path relative to the caller's cwd -> absolute
    case $1 in
        /* | [A-Za-z]:[/\\]*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$PWD/$1" ;;
    esac
}

_p05_pp() { # PYTHONPATH value with the repo first
    local pp
    pp=$(native_path "$PIGEMMA_HOME")
    if [ -n "${_PG_MSYS:-}" ]; then
        printf '%s\n' "$pp${PYTHONPATH:+;$PYTHONPATH}"
    else
        printf '%s\n' "$pp${PYTHONPATH:+:$PYTHONPATH}"
    fi
}

_p05_pyrun() { # _p05_pyrun ARGS... -> "$PY" ARGS from the repo root with PYTHONPATH set
    local pp
    pp=$(_p05_pp)
    (cd "$PIGEMMA_HOME" && PYTHONPATH="$pp" PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "$PY" "$@")
}

_p05_lines() { # _p05_lines TEXT -> array _LINES (non-empty lines, trailing CR removed)
    _LINES=()
    local l
    while IFS= read -r l || [ -n "$l" ]; do
        l=${l%$'\r'}
        if [ -n "$l" ]; then _LINES+=("$l"); fi
    done <<<"${1:-}"
}

_p05_join_cmd() { # printable shell command line
    local out="" a
    for a in "$@"; do
        out+="${out:+ }$(printf '%q' "$a")"
    done
    printf '%s\n' "$out"
}

_p05_thermal() { # "temp_c=<C>;throttled=<0x..>" (empty values when unknown)
    local t="" th=""
    if declare -F _pg_cpu_temp >/dev/null; then
        _pg_cpu_temp
        t=$_R
    fi
    if declare -F _pg_throttled >/dev/null; then
        _pg_throttled
        th=$_R
    fi
    printf 'temp_c=%s;throttled=%s\n' "$t" "$th"
}

_p05_need_uint() { # _p05_need_uint NAME VALUE [MIN]
    if ! _pg_is_uint "$2" || [ "$2" -lt "${3:-0}" ]; then
        die 1 "$1 must be an integer >= ${3:-0}, got '$2' (see --help)"
    fi
}

_p05_need_num() {
    if ! _pg_is_num "$2"; then die 1 "$1 must be a non-negative number, got '$2' (see --help)"; fi
}

# ============================================================================= python helper (stdlib)
# Commands: select | plan-bench | plan-run | annotate | calibrate | summary | last-run | prompt | split
_p05_py() {
    local pp
    pp=$(_p05_pp)
    (
        cd "$PIGEMMA_HOME" || exit 1
        PYTHONPATH="$pp" PYTHONUTF8=1 PYTHONIOENCODING=utf-8 exec "$PY" - "$@" <<'PIGEMMA_PY_EOF'
"""pi/05_run.sh helper: model selection, flag planning, envelope annotation, calibration (stdlib only)."""
import argparse
import datetime as _dt
import json
import math
import os
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path.cwd()
MARGIN = 500_000_000
COMPUTE = 300_000_000
RANK = {"FITS_MAPPED": 4, "FITS_REPACKED": 4, "STREAM_OK": 3, "STREAM_SLOW": 2}
PARTS = ("norepack", "hybrid_negctrl", "nommap", "mlock", "kvq8")
IN_RAM_READ_FRACTION = 0.02  # calibration: a run that read more than 2% of bytes/token was not in RAM

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
    sys.stderr.write(f"pi/05: {msg}\n")


def q(v):
    return shlex.quote("" if v is None else str(v))


def emit(name, value):
    if isinstance(value, bool):
        value = 1 if value else 0
    print(f"{name}={q(value)}")


def load_json(p):
    with open(p, encoding="utf-8-sig") as fh:
        return json.load(fh)


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


def posix(p):
    return Path(p).as_posix() if p not in (None, "") else ""


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


def gb(b):
    v = num(b)
    return "n/a" if v is None else f"{v / 1e9:.2f} GB"


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def resolve_file(f):
    p = Path(f)
    return p if p.is_absolute() else REPO / p


# ------------------------------------------------------------------------------------------ models
def model_candidates(m, model_dir):
    file = str(m.get("file") or "")
    repo = str(m.get("repo") or "")
    cands = []
    if model_dir and file:
        b = Path(model_dir).expanduser()
        cands += [b / file, b / Path(file).name]
        if repo:
            cands.append(b / repo.replace("/", "__") / file)
    lp = m.get("local_path")
    if lp and model_dir:
        # MODEL_DIR is the single source of truth (pigemma.feasibility checks the same candidates)
        cands.append(Path(model_dir).expanduser() / Path(str(lp)).name)
    out = []
    for c in cands:
        if c not in out:
            out.append(c)
    return out


def find_model(m, model_dir):
    cands = model_candidates(m, model_dir)
    for c in cands:
        if c.is_file():
            return c, True
    return (cands[0] if cands else None), False


def by_basename(models, name, draft=False):
    for mid, m in models.items():
        if (m.get("kind") == "draft") != draft:
            continue
        names = {Path(str(m.get("file") or "")).name}
        if m.get("local_path"):
            names.add(Path(str(m["local_path"])).name)
        if name in names:
            return mid
    return None


def norm_quant(s):
    return str(s or "").strip().upper()


def bytes_per_token(m):
    ne = num(m.get("nonexpert_bytes_per_token"))
    if ne is None:
        ne = num(m.get("nonexpert_bytes")) or 0.0
    worst = num(m.get("expert_bytes_per_token_worst"))
    if worst is None:
        worst = (num(m.get("n_moe_layers")) or 0) * (num(m.get("n_expert_used")) or 0) * \
            (num(m.get("expert_unit_bytes")) or 0)
    return int(ne + worst)


def has_nr(tokens):
    return any(t in ("-nr", "--no-repack") for t in tokens)


def cmd_select(a):
    fz = load_json(a.feasibility)
    cat = load_json(a.catalog)
    models = {str(m.get("id")): m for m in cat.get("models", []) or [] if isinstance(m, dict)}
    rows = [r for r in fz.get("rows") or [] if isinstance(r, dict)]
    byid = {str(r.get("model_id")): r for r in rows}
    rec = fz.get("recommended") if isinstance(fz.get("recommended"), dict) else None
    notes = []
    mid = None
    path = None
    spec = (a.model or "").strip()
    runnable_ids = sorted(k for k, m in models.items() if m.get("kind") != "draft")
    if spec:
        if spec in models:
            if models[spec].get("kind") == "draft":
                raise Fail(1, f"--model {spec} is a draft sidecar; pass it with --draft (chat/server)")
            mid = spec
        else:
            p = Path(spec).expanduser()
            if not p.is_file():
                raise Fail(3, f"--model {spec}: no such file and not a catalog model id "
                              f"(ids: {', '.join(runnable_ids)})")
            path = p
            mid = by_basename(models, p.name)
            if mid is None and by_basename(models, p.name, draft=True):
                raise Fail(1, f"--model {spec} is a draft sidecar; pass it with --draft (chat/server)")
            if mid is None:
                notes.append(f"{p.name} is not in the catalog: verdict rows do not cover it")
    elif a.quant_override:
        want_q = norm_quant(a.quant_override)
        fam = None
        if a.want and any(str(r.get("family")) == a.want for r in rows):
            fam = a.want
        elif rec:
            fam = rec.get("family")

        def qmatch(r, strip):
            rq = norm_quant(r.get("quant"))
            if strip:
                return rq.removeprefix("UD-") == want_q.removeprefix("UD-")
            return rq == want_q

        cands = []
        for strip in (False, True):
            cands = [r for r in rows if qmatch(r, strip)]
            if fam and any(str(r.get("family")) == fam for r in cands):
                cands = [r for r in cands if str(r.get("family")) == fam]
            if cands:
                break
        if not cands:
            quants = sorted({str(r.get("quant")) for r in rows})
            raise Fail(3, f"QUANT_OVERRIDE={a.quant_override} matches no verdict row (quants: {', '.join(quants)})")
        mid = str(cands[0].get("model_id"))
        notes.append(f"QUANT_OVERRIDE={a.quant_override} selected {mid}")
    else:
        if rec is None:
            raise Fail(3, "the verdict recommends no model (nothing runnable on this host); see the table above")
        mid = str(rec.get("model_id"))
        _, present = find_model(models.get(mid, {}), a.model_dir)
        if not present:
            alts = []
            for r in rows:
                rid = str(r.get("model_id"))
                rank = RANK.get(str(r.get("verdict")), 1 if (r.get("verdict") == "NEVER_GOOD_SPEED"
                                                             and r.get("runnable")) else 0)
                if rid == mid or rank <= 0 or rid not in models:
                    continue
                if find_model(models[rid], a.model_dir)[1]:
                    alts.append((rank, int(num(r.get("bytes")) or 0), rid))
            if alts:
                alts.sort(reverse=True)
                alt = alts[0][2]
                notes.append(f"the recommended {mid} is not downloaded in {a.model_dir}; using the best "
                             f"downloaded runnable model {alt} ({byid[alt].get('verdict')})")
                mid = alt

    row = byid.get(mid) if mid else None
    entry = models.get(mid) if mid else None
    if mid and row is None:
        raise Fail(3, f"model {mid} is not among the verdict rows (--want {fz.get('want')!r} excluded it?)")
    exists = False
    if path is None and entry is not None:
        path, exists = find_model(entry, a.model_dir)
    elif path is not None:
        exists = path.is_file()
    file_size = path.stat().st_size if exists else None

    from pigemma import feasibility as feas  # stdlib only

    if row is not None:
        verdict = str(row.get("verdict"))
        runnable = bool(row.get("runnable"))
        flags = row.get("flags")
        fits_rep = bool(row.get("fits_repacked"))
        mode = row.get("mode")
        if exists and not runnable and row.get("disk_ok") is False:
            # the verdict refused the row for lack of disk space, but the file is already here
            v2 = str(row.get("verdict_without_disk") or verdict)
            if v2 in (feas.FITS_MAPPED, feas.FITS_REPACKED, feas.STREAM_OK, feas.STREAM_SLOW):
                flags = feas.flags_for(v2, fits_repacked=fits_rep, threads=a.threads, n_ctx=a.ctx)
            elif (v2 == feas.NEVER_GOOD_SPEED and mode == "in_ram"
                  and (num(row.get("tok_s")) or 0.0) >= getattr(feas, "SLOW_TOK_S", 0.05)):
                flags = feas.flags_for(feas.FITS_REPACKED if fits_rep else feas.FITS_MAPPED,
                                       fits_repacked=fits_rep, threads=a.threads, n_ctx=a.ctx)
            else:
                flags = None
            if flags:
                runnable = True
                notes.append(f"the verdict found no room to download {mid} ({verdict}) but {posix(path)} is "
                             f"already present: using {v2}")
                verdict = v2
        if not runnable or not flags:
            raise Fail(3, f"{mid}: verdict {verdict} is not runnable on this host: {row.get('reason')}")
        cat_bytes = int(num(entry.get("bytes")) or num(row.get("bytes")) or 0)
        nonexpert = int(num(entry.get("nonexpert_bytes")) or num(row.get("nonexpert_bytes")) or 0)
        bpt = bytes_per_token(entry)
        kind = str(entry.get("kind") or row.get("kind") or "")
        family = str(entry.get("family") or "")
        quant = str(entry.get("quant") or "")
        tok_s = num(row.get("tok_s"))
        reason = str(row.get("reason") or "")
    else:
        size = int(file_size or 0)
        if size + MARGIN + COMPUTE > a.mem_available:
            raise Fail(3, f"{posix(path)} is not in the catalog and does not fit in RAM ({gb(size)} + 0.8 GB > "
                          f"MemAvailable {gb(a.mem_available)}); streaming needs the catalog's non-expert byte "
                          "count (python3 -m pigemma.gguf_map catalog-refresh, or use a catalog model)")
        verdict = "UNCATALOGUED"
        mode = "in_ram"
        fits_rep = False
        flags = feas.flags_for(feas.FITS_MAPPED, fits_repacked=False, threads=a.threads, n_ctx=a.ctx)
        cat_bytes = 0
        nonexpert = size
        bpt = 0
        kind = "unknown"
        family = ""
        try:
            from pigemma import paths as _paths
            quant = _paths.guess_quant(path)
        except Exception:  # noqa: BLE001
            quant = "unknown"
        tok_s = None
        reason = f"uncatalogued file of {gb(size)} fits in MemAvailable {gb(a.mem_available)}: FITS_MAPPED flags"

    tokens = shlex.split(flags)
    if mode in ("in_ram", "stream_moe") and not fits_rep and not has_nr(tokens):
        tokens.insert(0, "-nr")
        notes.append("added -nr (FITS_REPACKED is false)")
    gate_bytes = cat_bytes if cat_bytes > 0 else int(file_size or 0)
    mismatch = bool(exists and cat_bytes and file_size != cat_bytes)

    emit("SEL_MODEL_ID", mid or "")
    emit("SEL_MODEL_PATH", posix(path))
    emit("SEL_MODEL_EXISTS", exists)
    emit("SEL_FILE_SIZE", file_size if file_size is not None else "")
    emit("SEL_CATALOG_BYTES", cat_bytes)
    emit("SEL_BYTES", gate_bytes)
    emit("SEL_SIZE_MISMATCH", mismatch)
    emit("SEL_NONEXPERT", nonexpert)
    emit("SEL_BPT", bpt)
    emit("SEL_KIND", kind)
    emit("SEL_FAMILY", family)
    emit("SEL_QUANT", quant)
    emit("SEL_VERDICT", verdict)
    emit("SEL_MODE", mode or "")
    emit("SEL_FITS_REPACKED", fits_rep)
    emit("SEL_FLAGS", shlex.join(tokens))
    emit("SEL_PROJECTED_TOK_S", "" if tok_s is None else f"{tok_s:.3f}")
    emit("SEL_REASON", reason)

    d = resolve_draft(a, models, mid, notes)
    emit("SEL_DRAFT_ID", d["id"] if d else "")
    emit("SEL_DRAFT_PATH", d["path"] if d else "")
    emit("SEL_DRAFT_EXISTS", d["exists"] if d else False)
    emit("SEL_DRAFT_BYTES", d["bytes"] if d else 0)
    emit("SEL_DRAFT_SPEC_TYPE", d["spec_type"] if d else "")
    emit("SEL_NOTES", " | ".join(notes))
    return 0


def resolve_draft(a, models, mid, notes):
    spec = (a.draft or "").strip()
    if not spec or spec.lower() == "none":
        return None
    entry = None
    did = None
    path = None
    if spec.lower() == "auto":
        if not mid:
            raise Fail(3, "--draft auto needs a catalog model (the main model is not in the catalog)")
        ids = [str(i) for i in models[mid].get("drafts") or []]
        cands = [models[i] for i in ids if i in models]
        entry = next((d for d in cands if d.get("spec_type") == "draft-mtp"), cands[0] if cands else None)
        if entry is None:
            raise Fail(3, f"--draft auto: the catalog lists no draft for {mid}")
        did = str(entry.get("id"))
    elif spec in models and models[spec].get("kind") == "draft":
        entry = models[spec]
        did = spec
    else:
        p = Path(spec).expanduser()
        if not p.is_file():
            raise Fail(3, f"--draft {spec}: no such file and not a catalog draft id")
        path = p
        did = by_basename(models, p.name, draft=True)
        entry = models.get(did) if did else None
    exists = False
    if path is None:
        path, exists = find_model(entry, a.model_dir)
    else:
        exists = path.is_file()
    if exists:
        size = path.stat().st_size
    else:
        size = int(num((entry or {}).get("bytes")) or 0)
    if a.spec_type:
        st = "" if a.spec_type.lower() == "none" else a.spec_type
    elif entry is not None and entry.get("spec_type"):
        st = str(entry["spec_type"])
    else:
        base = Path(str(path)).name.lower()
        st = "draft-mtp" if base.startswith("mtp") else ("draft-dflash" if base.startswith("dflash") else "")
    if entry is not None and mid and models.get(mid):
        fam_d = str(entry.get("family") or "")
        fam_m = str(models[mid].get("family") or "")
        if fam_d and fam_m and fam_d != fam_m:
            notes.append(f"draft {did} belongs to family {fam_d}, the model to {fam_m}: llama.cpp will likely refuse it")
    return {"id": did or Path(str(path)).name, "path": posix(path), "exists": exists, "bytes": int(size),
            "spec_type": st}


# ------------------------------------------------------------------------------------------- flags
def classify(tokens):
    d = {"lm": None, "fa": None, "ctk": None, "ctv": None, "nr": False, "threads": None, "ctx": None,
         "batch": [], "other": []}
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        key, eq, val = (t.partition("=") if t.startswith("--") else (t, "", ""))

        def take():
            nonlocal i
            if eq:
                return val
            i += 1
            return tokens[i] if i < n else ""

        if key in ("-nr", "--no-repack"):
            d["nr"] = True
        elif key == "--repack":
            d["nr"] = False
        elif key in ("-lm", "--load-mode"):
            d["lm"] = take()
        elif key in ("-fa", "--flash-attn"):
            d["fa"] = take()
        elif key in ("-ctk", "--cache-type-k"):
            d["ctk"] = take()
        elif key in ("-ctv", "--cache-type-v"):
            d["ctv"] = take()
        elif key in ("-t", "--threads"):
            d["threads"] = take()
        elif key in ("-c", "--ctx-size"):
            d["ctx"] = take()
        elif key in ("-b", "--batch-size"):
            d["batch"] += ["-b", take()]
        elif key in ("-ub", "--ubatch-size"):
            d["batch"] += ["-ub", take()]
        else:
            d["other"].append(t)
        i += 1
    return d


def is_kvq8(d):
    return d["fa"] == "on" and d["ctk"] == "q8_0" and d["ctv"] == "q8_0"


def kv_rest(d):
    out = []
    for flag, k in (("-fa", "fa"), ("-ctk", "ctk"), ("-ctv", "ctv")):
        if d[k]:
            out += [flag, d[k]]
    return out


def variant_name(parts):
    return "+".join(p for p in PARTS if p in parts) or "default"


def cmd_plan_run(a):
    d = classify(shlex.split(a.flags or ""))
    parts = []
    rest = []
    if d["nr"]:
        parts.append("norepack")
    if d["lm"] == "none":
        parts.append("nommap")
    elif d["lm"] == "mmap+mlock":
        parts.append("mlock")
    elif d["lm"]:
        rest += ["-lm", d["lm"]]
    if is_kvq8(d):
        parts.append("kvq8")
    else:
        rest += kv_rest(d)
    rest += d["batch"] + d["other"] + shlex.split(a.extra or "")
    emit("PLAN_VARIANT", variant_name(parts))
    emit("PLAN_REST", shlex.join(rest))
    emit("PLAN_THREADS", d["threads"] or "")
    emit("PLAN_CTX", d["ctx"] or "")
    return 0


def cmd_plan_bench(a):
    base = classify(shlex.split(a.flags or ""))
    extra = shlex.split(a.extra or "")
    seen = set()
    for raw in [v.strip() for v in (a.variants or "").split(",") if v.strip()]:
        parts = [p.strip() for p in raw.split("+") if p.strip()]
        bad = [p for p in parts if p not in ("default",) + PARTS]
        if bad or not parts or ("default" in parts and len(parts) > 1):
            raise Fail(1, f"--variants: unknown or malformed variant {raw!r} (names: default, "
                          f"{', '.join(PARTS)}; combine with '+', 'default' stands alone)")
        parts = [p for p in parts if p != "default"]
        rest = []
        if not a.no_verdict_flags:
            lm = base["lm"]
            if lm and not ({"nommap", "mlock"} & set(parts)):
                if lm == "none":
                    parts.append("nommap")
                elif lm == "mmap+mlock" and a.mlock_ok:
                    parts.append("mlock")
                elif lm == "mmap+mlock":
                    rest += ["-lm", "mmap"]
                else:
                    rest += ["-lm", lm]
            if "kvq8" not in parts:
                if is_kvq8(base):
                    parts.append("kvq8")
                else:
                    rest += kv_rest(base)
            rest += base["batch"]
        rest += extra
        anon = ("norepack" not in parts) or ("nommap" in parts)
        name = variant_name(parts)
        if anon and not a.anon_ok:
            print(f"REFUSED\t{raw}\t{a.anon_reason}")
            continue
        if "mlock" in parts and not a.mlock_ok:
            print(f"REFUSED\t{raw}\tmlock of model weights is not allowed on this host (ALLOW_MLOCK / protected host)")
            continue
        if name in seen:
            continue
        seen.add(name)
        print(f"RUN\t{name}\t{shlex.join(rest)}")
    if base["other"] and not a.no_verdict_flags:
        say(f"verdict flags not passed to llama-bench (no llama-bench equivalent): {shlex.join(base['other'])}")
    return 0


# ------------------------------------------------------------------------------------- envelopes
def thermal(s):
    out = {}
    for kv in (s or "").split(";"):
        k, _, v = kv.partition("=")
        k = k.strip()
        if not k:
            continue
        v = v.strip()
        out[k] = (num(v) if k == "temp_c" else (v or None)) if v else None
    return out or None


def vmstat_summary(path):
    p = Path(path)
    if not p.is_file():
        return {"file": rel(p), "n_samples": 0, "note": "vmstat log missing"}
    header = None
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        toks = line.split()
        if not toks:
            continue
        if "bi" in toks and "wa" in toks and "r" in toks:
            header = toks
            continue
        if header and len(toks) == len(header) and all(re.fullmatch(r"-?\d+", t) for t in toks):
            rows.append([int(t) for t in toks])
    out = {"file": rel(p), "n_samples": max(0, len(rows) - 1),
           "note": "vmstat 1 over the whole llama-completion run (load + prompt + generation); the first "
                   "sample (averages since boot) is dropped; bi/bo in KiB/s, us/sy/id/wa in percent"}
    if header and len(rows) > 1:
        data = rows[1:]
        means = {}
        for i, name in enumerate(header):
            if name in ("r", "b", "free", "buff", "cache", "si", "so", "bi", "bo", "us", "sy", "id", "wa", "st"):
                means[name] = round(sum(r[i] for r in data) / len(data), 2)
        out["mean"] = means
        out["max"] = {name: max(r[header.index(name)] for r in data) for name in ("bi", "wa") if name in header}
    return out


def cmd_annotate(a):
    from pigemma import results as rs

    E = os.environ
    raw_dir = E.get("PI05_BENCH_RAW_DIR") or ""
    copied = {}
    rc = 0
    for f in a.files:
        p = resolve_file(f)
        try:
            doc = load_json(p)
        except (OSError, ValueError) as e:
            say(f"cannot read {f}: {e}")
            rc = 1
            continue
        notes = [str(n) for n in (doc.get("notes") or [])]

        def add(n):
            if n and n not in notes:
                notes.append(n)

        extra = dict(doc.get("extra")) if isinstance(doc.get("extra"), dict) else {}
        is_bench = bool(extra.get("token_source"))
        if E.get("PI05_PROTECTED_OVERRIDE") == "1":
            add(f"protected host '{E.get('PI05_HOSTNAME')}' run with PIGEMMA_I_KNOW=1 (authorised override): "
                f"services untouched, nice -n {E.get('NICE_LEVEL', '10')}, no mlock of model weights")
        if E.get("PI05_MLOCK_REMOVED") == "1":
            add("mlock removed from the verdict flags (ALLOW_MLOCK=auto on a protected host: the page cache must "
                "stay reclaimable for its services)")
        if E.get("PI05_FORCED_NR") == "1":
            add("-nr forced on a protected host: the anonymous repack copy is not reclaimable either "
                "(pi/05_run.sh --allow-repack measures it)")
        if E.get("PI05_EMULATED") == "1" and not any("emulated" in n for n in notes):
            add("emulated host (PIGEMMA_EMULATED=1, qemu): a functional proof, not a Raspberry Pi measurement")
        if E.get("PI05_PINNED") == "1":
            add(f"hot set pinned by tools/pinner during this run (pid {E.get('PI05_PIN_PID')}, "
                f"VmLck {gb(E.get('PI05_PIN_BYTES'))})")
        if is_bench and E.get("PI05_MODE") == "stream_moe":
            add("llama-bench feeds random token ids: in streaming mode its tg reflects random expert routing, "
                "not the locality of real text (use stream-bench)")
        try:
            argv = json.loads(E.get("PI05_ARGV_JSON") or "[]")
        except ValueError:
            argv = []
        pin = None
        if E.get("PI05_PINNED") == "1":
            pin = {"pid": E.get("PI05_PIN_PID"), "vm_lck_bytes": int(num(E.get("PI05_PIN_BYTES")) or 0),
                   "started_here": E.get("PI05_PIN_STARTED_HERE") == "1"}
        pi = {
            "script": "pi/05_run.sh", "version": E.get("PI05_VERSION"), "subcommand": E.get("PI05_SUBCMD"),
            "label": a.label or None, "argv": argv, "model_path": E.get("PI05_MODEL_PATH"),
            "verdict": E.get("PI05_VERDICT"), "mode": E.get("PI05_MODE"),
            "projected_tok_s": num(E.get("PI05_PROJECTED_TOK_S")), "verdict_reason": E.get("PI05_REASON"),
            "verdict_flags": E.get("PI05_FLAGS_VERDICT"), "effective_flags": E.get("PI05_FLAGS_EFF"),
            "extra_flags": E.get("PI05_EXTRA_FLAGS") or None,
            "mlock_removed": E.get("PI05_MLOCK_REMOVED") == "1", "forced_nr": E.get("PI05_FORCED_NR") == "1",
            "protected_override": E.get("PI05_PROTECTED_OVERRIDE") == "1",
            "emulated": E.get("PI05_EMULATED") == "1",
            "selection_notes": [n for n in (E.get("PI05_SEL_NOTES") or "").split(" | ") if n],
            "feasibility_json": rel(E.get("PI05_FEASIBILITY")), "facts_json": rel(E.get("PI05_FACTS")),
            "ram_gate": {"need_bytes": int(num(E.get("PI05_GATE_NEED")) or 0),
                         "mem_available_bytes": int(num(E.get("PI05_MEM_AVAILABLE")) or 0),
                         "rule": E.get("PI05_GATE_RULE")},
            "pin": pin,
            "thermal": {"before": thermal(E.get("PI05_THERMAL_BEFORE")),
                        "after": thermal(E.get("PI05_THERMAL_AFTER"))},
        }
        if a.vmstat:
            pi["vmstat"] = vmstat_summary(a.vmstat)
        raw = extra.get("raw_stdout")
        if raw_dir and raw and is_bench:
            src = resolve_file(raw)
            if src in copied:
                pi["raw_bench_json"] = copied[src]
            elif src.is_file():
                try:
                    rows = json.loads(src.read_text(encoding="utf-8", errors="replace"))
                except ValueError:
                    rows = None
                if isinstance(rows, list) and rows:
                    mid = re.sub(r"[^A-Za-z0-9_.+-]", "_", str((doc.get("model") or {}).get("id") or "model"))
                    var = re.sub(r"[^A-Za-z0-9_.+-]", "_", str(doc.get("variant") or "variant"))
                    ts = re.sub(r"[^0-9TZ]", "", str(doc.get("created_at") or ""))[:16] or "na"
                    dst = Path(raw_dir) / f"bench_{mid}_{var}_{ts}.json"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    copied[src] = rel(dst)
                    pi["raw_bench_json"] = rel(dst)
                    say(f"raw llama-bench JSON: {rel(dst)}")
        extra["pi05"] = pi
        doc["notes"] = notes
        doc["extra"] = extra
        errs, _warns = rs.check_result(doc)
        if errs:
            say(f"{rel(p)}: annotation would break the envelope ({'; '.join(errs[:3])}); left unchanged")
            rc = 1
            continue
        write_atomic(p, rs.dumps_result(doc))
    return rc


def rate(b):
    v = num(b.get("tok_s")) if isinstance(b, dict) else None
    return "-" if v is None else f"{v:.2f}"


def cmd_summary(a):
    try:
        from pigemma import bench as _bench
        vis = _bench.visible_text
    except Exception:  # noqa: BLE001
        vis = None
    for f in a.files:
        p = resolve_file(f)
        try:
            doc = load_json(p)
        except (OSError, ValueError) as e:
            say(f"cannot read {f}: {e}")
            continue
        bits = [f"variant {doc.get('variant')}", f"threads {doc.get('threads')}",
                f"pp {rate(doc.get('pp'))} tok/s", f"tg {rate(doc.get('tg'))} tok/s"]
        rb = num(doc.get("read_bytes_per_token"))
        if rb is not None:
            bits.append(f"read {rb / 1e6:.2f} MB/token")
        mf = num(doc.get("majflt_per_token"))
        if mf is not None:
            bits.append(f"majflt {mf:.1f}/token")
        for k, unit in (("load_s", "s"), ("rss_peak_mb", "MB"), ("private_peak_mb", "MB"), ("mapped_mb", "MB")):
            v = num(doc.get(k))
            if v is not None:
                bits.append(f"{k} {v:g} {unit}")
        if doc.get("buffer_split"):
            bits.append("buffers " + ",".join(f"{k}:{v}" for k, v in doc["buffer_split"].items()))
        if doc.get("coherent") is not None:
            bits.append(f"coherent {doc.get('coherent')}")
        sys.stderr.write(f"pi/05 result [{a.label or 'run'}] {rel(p)}\n    " + "; ".join(bits) + "\n")
        for n in doc.get("notes") or []:
            sys.stderr.write(f"    note: {n}\n")
        if a.show_output and doc.get("output_sample"):
            text = str(doc["output_sample"])
            if vis is not None:
                try:
                    text = vis(text)
                except Exception:  # noqa: BLE001
                    pass
            sys.stderr.write(f"----- {a.label or 'output'} (greedy, --temp 0) -----\n{text.strip()}\n"
                             "-----\n")
    return 0


# ------------------------------------------------------------------------------------- calibration
def membw(a):
    if a.io_mem and Path(a.io_mem).is_file():
        try:
            d = load_json(a.io_mem)
        except (OSError, ValueError) as e:
            say(f"cannot read {a.io_mem}: {e}")
            d = {}
        mem = d.get("memory") if isinstance(d, dict) else None
        if isinstance(mem, dict):
            by = mem.get("read_gbps_by_threads")
            vals = [num(v) for v in by.values()] if isinstance(by, dict) else []
            vals = [v for v in vals if v]
            if vals:
                return max(vals), (f"measured: {rel(a.io_mem)} memory.read_gbps_by_threads max (membw, pi/03)"), \
                    [rel(a.io_mem)]
            for k in ("copy_gbps", "triad_gbps"):
                v = num(mem.get(k))
                if v:
                    return v, f"measured: {rel(a.io_mem)} memory.{k} (membw, pi/03)", [rel(a.io_mem)]
    if a.feasibility and Path(a.feasibility).is_file():
        try:
            fz = load_json(a.feasibility)
            mb = (fz.get("device") or {}).get("mem_bw_gbps")
            v = num(mb)
            if v:
                src = str(mb.get("source") if isinstance(mb, dict) else "") or "assumed: verdict device profile"
                return v, src, [rel(a.feasibility)]
        except (OSError, ValueError):
            pass
    return 10.0, "assumed: Pi 5 LPDDR4X ~10 GB/s effective (pigemma.feasibility default)", []


def cmd_calibrate(a):
    E = os.environ

    def skip(why):
        say(f"calibration not written: {why}")
        emit("CAL_WRITTEN", 0)
        emit("CAL_REASON", why)
        return 0

    if a.mode != "in_ram":
        return skip(f"mode {a.mode!r} is not in-RAM (the efficiency is an in-RAM quantity)")
    bpt = int(a.bpt or 0)
    if bpt <= 0:
        return skip("bytes per token unknown (model not in the catalog)")
    cands = []
    for f in a.files:
        p = resolve_file(f)
        try:
            doc = load_json(p)
        except (OSError, ValueError):
            continue
        if doc.get("kind") != "measured":
            continue
        tg = num((doc.get("tg") or {}).get("tok_s")) if isinstance(doc.get("tg"), dict) else None
        if not tg or tg <= 0:
            continue
        rb = num(doc.get("read_bytes_per_token"))
        if rb is not None and rb > IN_RAM_READ_FRACTION * bpt:
            say(f"{rel(p)} read {rb / 1e6:.1f} MB/token from storage (> {IN_RAM_READ_FRACTION:.0%} of "
                f"{bpt / 1e6:.0f} MB/token): not an in-RAM run, not used for calibration")
            continue
        cands.append((doc, p, tg, rb))
    if not cands:
        return skip("no measured tg tok/s from an in-RAM run")

    def key(c):
        doc = c[0]
        nr = "norepack" in str(doc.get("variant") or "").split("+")
        return (doc.get("threads") == a.threads, nr != a.prefer_repack, c[2])

    doc, p, tg, rb = max(cands, key=key)
    bw, bw_src, bw_files = membw(a)
    eff = tg * bpt / (bw * 1e9)
    if not 0.05 <= eff <= 1.5:
        return skip(f"llama_efficiency {eff:.4f} is outside 0.05..1.5 (tg {tg:.3f}, {bpt / 1e9:.3f} GB/token, "
                    f"mem_bw {bw:.2f} GB/s): check the inputs")
    measured_bw = bw_src.strip().lower().startswith("measured")
    prefix = "measured" if measured_bw else "assumed"
    mid = (doc.get("model") or {}).get("id")
    src = (f"{prefix}: pi/05_run.sh {a.run} llama_efficiency = tg {tg:.3f} tok/s ({mid}, variant "
           f"{doc.get('variant')}, threads {doc.get('threads')}, {rel(p)}) x {bpt / 1e9:.4f} GB/token (catalog "
           f"nonexpert_bytes_per_token + expert_bytes_per_token_worst) / mem_bw {bw:.2f} GB/s ({bw_src})")
    if not measured_bw:
        src += "; tg is measured, mem_bw is not (run pi/03_bench_io_mem.sh, then re-run this step)"
    out = Path(a.out)
    history = []
    if out.is_file():
        try:
            prev = load_json(out)
            if isinstance(prev, dict):
                history = [h for h in (prev.get("history") or []) if isinstance(h, dict)]
                if isinstance(prev.get("llama_efficiency"), dict):
                    history.insert(0, {"llama_efficiency": prev["llama_efficiency"],
                                       "created_at": prev.get("created_at"), "inputs": prev.get("inputs")})
        except (OSError, ValueError):
            pass
    cal = {
        "format_version": 1, "generator": "pi/05_run.sh", "created_at": now_iso(),
        "host": E.get("PI05_HOSTNAME"),
        "llama_efficiency": {"value": round(eff, 4), "source": src,
                             "from": [round(tg, 4), bpt, round(bw, 3)]},
        "inputs": {"tg_tok_s": tg, "bytes_per_token": bpt, "mem_bw_gbps": bw, "mem_bw_source": bw_src,
                   "mem_bw_files": bw_files, "model_id": mid, "variant": doc.get("variant"),
                   "threads": doc.get("threads"), "run": a.run, "result_file": rel(p),
                   "read_bytes_per_token": rb, "mode": a.mode, "verdict": E.get("PI05_VERDICT")},
        "notes": ["llama_efficiency = tg x bytes/token / mem_bw: the fraction of the memory bandwidth that "
                  "reaches llama.cpp decode on this host; pigemma.feasibility (--calibration auto) and "
                  "pigemma.cache_sim (Pi profiles) use it instead of the 0.65 assumption",
                  "from = [tg tok/s, bytes/token, mem_bw GB/s]"],
        "history": history[:20],
    }
    write_atomic(out, json.dumps(cal, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    try:
        from pigemma import feasibility as feas
        feas.load_calibration(str(out))
    except Exception as e:  # noqa: BLE001
        say(f"warning: pigemma.feasibility cannot read the calibration just written: {e}")
    say(f"calibration written: {rel(out)}: llama_efficiency {eff:.4f} ({prefix})")
    emit("CAL_WRITTEN", 1)
    emit("CAL_PATH", rel(out))
    emit("CAL_VALUE", f"{eff:.4f}")
    emit("CAL_KIND", prefix)
    return 0


# --------------------------------------------------------------------------------------- last run
def cmd_last_run(a):
    E = os.environ
    metrics = []
    for f in a.files:
        p = resolve_file(f)
        try:
            doc = load_json(p)
        except (OSError, ValueError):
            continue
        metrics.append({
            "file": rel(p), "label": None, "experiment": doc.get("experiment"), "variant": doc.get("variant"),
            "threads": doc.get("threads"),
            "pp_tok_s": num((doc.get("pp") or {}).get("tok_s")) if isinstance(doc.get("pp"), dict) else None,
            "tg_tok_s": num((doc.get("tg") or {}).get("tok_s")) if isinstance(doc.get("tg"), dict) else None,
            "read_bytes_per_token": num(doc.get("read_bytes_per_token")),
            "majflt_per_token": num(doc.get("majflt_per_token")), "load_s": num(doc.get("load_s")),
            "coherent": doc.get("coherent"),
            "failed": any(str(n).startswith("failed") for n in doc.get("notes") or []),
        })
    labels = [x for x in (a.labels or "").split(",")]
    for m, lab in zip(metrics, labels):
        m["label"] = lab or None
    out = {
        "format_version": 1, "generator": "pi/05_run.sh", "created_at": now_iso(),
        "host": E.get("PI05_HOSTNAME"), "subcommand": E.get("PI05_SUBCMD"), "exit_code": a.rc,
        "model": {"id": E.get("PI05_MODEL_ID") or None, "path": E.get("PI05_MODEL_PATH") or None},
        "verdict": E.get("PI05_VERDICT"), "mode": E.get("PI05_MODE"),
        "effective_flags": E.get("PI05_FLAGS_EFF"), "extra_flags": E.get("PI05_EXTRA_FLAGS") or None,
        "pinned": E.get("PI05_PINNED") == "1", "results": [m["file"] for m in metrics], "metrics": metrics,
        "calibration": E.get("PI05_CAL_PATH") or None, "out_dir": rel(E.get("PI05_OUT_DIR")),
        "emulated": E.get("PI05_EMULATED") == "1", "protected_override": E.get("PI05_PROTECTED_OVERRIDE") == "1",
    }
    text = json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    for target in a.out:
        write_atomic(target, text)
    return 0


def cmd_split(a):
    """Shell-quoted words of a flags string (shlex rules), for `eval "arr=($(...))"` in bash."""
    try:
        words = shlex.split(a.text or "")
    except ValueError as e:
        raise Fail(1, f"cannot split the flags {a.text!r}: {e}")
    print(" ".join(shlex.quote(w) for w in words))
    return 0


def cmd_prompt(a):
    items = load_json(a.file)
    if isinstance(items, dict):
        items = items.get("prompts") or []
    for it in items:
        if isinstance(it, dict) and str(it.get("id")) == a.id and it.get("prompt"):
            write_atomic(a.out, str(it["prompt"]))
            return 0
    raise Fail(3, f"prompt id {a.id!r} not found in {a.file}")


def main(argv):
    ap = argparse.ArgumentParser(prog="pi/05 helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    for k in ("--feasibility", "--catalog", "--model-dir", "--model", "--quant-override", "--want", "--draft",
              "--spec-type"):
        s.add_argument(k, default="")
    s.add_argument("--mem-available", type=int, default=0)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--ctx", type=int, default=4096)
    s = sub.add_parser("plan-run")
    s.add_argument("--flags", default="")
    s.add_argument("--extra", default="")
    s = sub.add_parser("plan-bench")
    s.add_argument("--flags", default="")
    s.add_argument("--extra", default="")
    s.add_argument("--variants", default="norepack")
    s.add_argument("--anon-ok", type=int, default=0)
    s.add_argument("--anon-reason", default="")
    s.add_argument("--mlock-ok", type=int, default=0)
    s.add_argument("--no-verdict-flags", type=int, default=0)
    s = sub.add_parser("annotate")
    s.add_argument("--label", default="")
    s.add_argument("--vmstat", default="")
    s.add_argument("files", nargs="*")
    s = sub.add_parser("summary")
    s.add_argument("--label", default="")
    s.add_argument("--show-output", type=int, default=0)
    s.add_argument("files", nargs="*")
    s = sub.add_parser("calibrate")
    s.add_argument("--mode", default="")
    s.add_argument("--bpt", type=int, default=0)
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--prefer-repack", type=int, default=0)
    s.add_argument("--io-mem", default="")
    s.add_argument("--feasibility", default="")
    s.add_argument("--out", required=True)
    s.add_argument("--run", default="")
    s.add_argument("files", nargs="*")
    s = sub.add_parser("last-run")
    s.add_argument("--out", action="append", default=[])
    s.add_argument("--rc", type=int, default=0)
    s.add_argument("--labels", default="")
    s.add_argument("files", nargs="*")
    s = sub.add_parser("split")
    s.add_argument("--text", default="")
    s = sub.add_parser("prompt")
    s.add_argument("--file", required=True)
    s.add_argument("--id", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    funcs = {"select": cmd_select, "plan-run": cmd_plan_run, "plan-bench": cmd_plan_bench,
             "annotate": cmd_annotate, "summary": cmd_summary, "calibrate": cmd_calibrate,
             "last-run": cmd_last_run, "prompt": cmd_prompt, "split": cmd_split}
    try:
        return int(funcs[a.cmd](a) or 0)
    except Fail as e:
        say(str(e))
        return e.code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
PIGEMMA_PY_EOF
    )
}

# =================================================================================== argument parsing
SUBCMD=""
O_MODEL="" O_WANT="" O_DRAFT="" O_SPEC_TYPE="" O_DRAFT_NMAX="" O_VARIANTS="" O_BENCH_THREADS=""
O_PIN=0 O_P="" O_P_SET=0 O_FILE="" O_N="" O_PP="" O_TG="" O_REPS="" O_T="" O_C=""
O_NO_VERDICT_FLAGS=0 O_ALLOW_REPACK=0 O_OUT="" O_EXPERIMENT="E8" O_SIM="" O_IDLE_S=10 O_IDLE_PCT=5
O_TIMEOUT="" O_NO_CALIBRATE=0 O_CAL_OUT="" O_IO_MEM="" O_DRY=0 O_CHAT=0 O_NO_HE=0 O_HE_ID="gen-he-explain-01" O_HE_N=48
O_HOST="0.0.0.0" O_PORT=8080
O_FLAGS=()
O_PIN_ARGS=()
O_FEAS_ARGS=()
_P05_SEEN=()

_p05_val() { # _p05_val OPTION "$@" -> dies when the option has no value
    if [ $# -lt 3 ]; then die 1 "option $1 needs a value (see --help)"; fi
}

p05_parse_args() {
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
            --version) printf 'pi/05_run.sh %s\n' "$PI05_VERSION"; exit 0 ;;
            bench | stream-bench | chat | server)
                if [ -n "$SUBCMD" ]; then die 1 "more than one subcommand: '$SUBCMD' and '$a'"; fi
                SUBCMD=$a
                shift
                continue
                ;;
            --pin) O_PIN=1; _P05_SEEN+=(--pin); shift; continue ;;
            --no-verdict-flags) O_NO_VERDICT_FLAGS=1; _P05_SEEN+=("$a"); shift; continue ;;
            --allow-repack) O_ALLOW_REPACK=1; shift; continue ;;
            --no-calibrate) O_NO_CALIBRATE=1; shift; continue ;;
            --dry-run) O_DRY=1; shift; continue ;;
            --chat) O_CHAT=1; _P05_SEEN+=("$a"); shift; continue ;;
            --no-he) O_NO_HE=1; _P05_SEEN+=("$a"); shift; continue ;;
            --)
                shift
                if [ $# -gt 0 ]; then O_FLAGS+=("$*"); fi
                break
                ;;
        esac
        case $a in
            --model | -m | --want | --draft | -md | --spec-type | --draft-n-max | --variants | --bench-threads | \
                --pin-arg | -p | --prompt | -f | --file | --prompt-file | -n | --n-predict | --pp | --tg | -r | \
                --reps | -t | --threads | -c | --ctx | --ctx-size | --flags | --out | --experiment | --sim | \
                --feasibility-arg | --idle-check-s | --idle-max-pct | --timeout | --io-mem | --he-prompt-id | \
                --he-n | --host | --port | --calibration-out)
                _p05_val "$a" "$@"
                ;;
            -*) die 1 "unknown option '$a' (see --help)" ;;
            *) die 1 "unexpected argument '$a' (subcommands: bench, stream-bench, chat, server; see --help)" ;;
        esac
        _P05_SEEN+=("$a")
        case $a in
            --model | -m) O_MODEL=$2 ;;
            --want) O_WANT=$2 ;;
            --draft | -md) O_DRAFT=$2 ;;
            --spec-type) O_SPEC_TYPE=$2 ;;
            --draft-n-max) O_DRAFT_NMAX=$2 ;;
            --variants) O_VARIANTS=$2 ;;
            --bench-threads) O_BENCH_THREADS=$2 ;;
            --pin-arg) O_PIN_ARGS+=("$2") ;;
            -p | --prompt) O_P=$2; O_P_SET=1 ;;
            -f | --file | --prompt-file) O_FILE=$2 ;;
            -n | --n-predict) O_N=$2 ;;
            --pp) O_PP=$2 ;;
            --tg) O_TG=$2 ;;
            -r | --reps) O_REPS=$2 ;;
            -t | --threads) O_T=$2 ;;
            -c | --ctx | --ctx-size) O_C=$2 ;;
            --flags) O_FLAGS+=("$2") ;;
            --out) O_OUT=$2 ;;
            --experiment) O_EXPERIMENT=$2 ;;
            --sim) O_SIM=$2 ;;
            --feasibility-arg) O_FEAS_ARGS+=("$2") ;;
            --idle-check-s) O_IDLE_S=$2 ;;
            --idle-max-pct) O_IDLE_PCT=$2 ;;
            --timeout) O_TIMEOUT=$2 ;;
            --io-mem) O_IO_MEM=$2 ;;
            --calibration-out) O_CAL_OUT=$2 ;;
            --he-prompt-id) O_HE_ID=$2 ;;
            --he-n) O_HE_N=$2 ;;
            --host) O_HOST=$2 ;;
            --port) O_PORT=$2 ;;
        esac
        shift 2
    done
}

_p05_seen() { # 0 if option $1 was given
    local x
    for x in "${_P05_SEEN[@]}"; do
        if [ "$x" = "$1" ]; then return 0; fi
    done
    return 1
}

_p05_only_for() { # _p05_only_for "SUBCMDS" OPTION...: usage error when an option is used elsewhere
    local allowed=$1 o
    shift
    case " $allowed " in *" $SUBCMD "*) return 0 ;; esac
    for o in "$@"; do
        if _p05_seen "$o"; then
            die 1 "option $o is not used by '$SUBCMD' (only: $allowed; see --help)"
        fi
    done
}

p05_validate_args() {
    if [ -z "$SUBCMD" ]; then
        usage >&2
        die 1 "no subcommand given (bench, stream-bench, chat or server)"
    fi
    _p05_only_for "bench" --variants --bench-threads --pp --tg -r --reps --no-verdict-flags
    _p05_only_for "stream-bench" --chat --no-he --he-prompt-id --he-n
    _p05_only_for "chat server" --draft -md --spec-type --draft-n-max
    _p05_only_for "server" --host --port
    _p05_only_for "stream-bench chat" -f --file --prompt-file --prompt
    if [ "$SUBCMD" = server ] && { [ "$O_P_SET" = 1 ] || [ -n "$O_N" ]; }; then
        die 1 "server takes no -p / -n (send prompts to http://$O_HOST:$O_PORT/completion)"
    fi
    if [ "$SUBCMD" = bench ]; then
        # llama-bench spelling: -p N = prompt tokens, -n N = generated tokens
        if [ "$O_P_SET" = 1 ]; then
            if [ -n "$O_PP" ]; then die 1 "bench: give the prompt tokens once (-p N or --pp N)"; fi
            O_PP=$O_P
        fi
        if [ -n "$O_N" ]; then
            if [ -n "$O_TG" ]; then die 1 "bench: give the generated tokens once (-n N or --tg N)"; fi
            O_TG=$O_N
        fi
        O_PP=${O_PP:-64}
        O_TG=${O_TG:-32}
        O_REPS=${O_REPS:-3}
        _p05_need_uint "--pp/-p" "$O_PP" 0
        _p05_need_uint "--tg/-n" "$O_TG" 0
        _p05_need_uint "-r/--reps" "$O_REPS" 1
        if [ "$O_PP" = 0 ] && [ "$O_TG" = 0 ]; then die 1 "bench: --pp and --tg are both 0: nothing to measure"; fi
    fi
    if [ "$SUBCMD" = stream-bench ] || [ "$SUBCMD" = chat ]; then
        if [ "$O_P_SET" = 1 ] && [ -n "$O_FILE" ]; then die 1 "give either -p TEXT or -f FILE, not both"; fi
        if [ "$O_P_SET" = 1 ] && [ -z "$O_P" ]; then die 1 "-p needs a non-empty prompt text"; fi
    fi
    if [ "$SUBCMD" = stream-bench ]; then
        O_N=${O_N:-64}
        _p05_need_uint "-n" "$O_N" 1
        _p05_need_uint "--he-n" "$O_HE_N" 1
    fi
    if [ "$SUBCMD" = chat ] && [ -n "$O_N" ]; then _p05_need_uint "-n" "$O_N" 1; fi
    if [ -n "$O_DRAFT_NMAX" ]; then _p05_need_uint "--draft-n-max" "$O_DRAFT_NMAX" 1; fi
    if [ -n "$O_SPEC_TYPE" ] && [ -z "$O_DRAFT" ]; then die 1 "--spec-type needs --draft"; fi
    if [ -n "$O_DRAFT_NMAX" ] && [ -z "$O_DRAFT" ]; then die 1 "--draft-n-max needs --draft"; fi
    _p05_need_uint "--port" "$O_PORT" 1
    if [ "$O_PORT" -gt 65535 ]; then die 1 "--port must be 1..65535, got $O_PORT"; fi
    _p05_need_num "--idle-check-s" "$O_IDLE_S"
    _p05_need_num "--idle-max-pct" "$O_IDLE_PCT"
    if [ -n "$O_TIMEOUT" ]; then _p05_need_num "--timeout" "$O_TIMEOUT"; fi
    if ! [[ $O_EXPERIMENT =~ ^(E[0-8]|adhoc)$ ]]; then die 1 "--experiment must be E0..E8 or adhoc, got '$O_EXPERIMENT'"; fi
    if [ -n "$O_T" ]; then
        case $O_T in
            *,*)
                if [ "$SUBCMD" != bench ]; then die 1 "-t LIST (comma list) is only for bench; give one number"; fi
                if [ -n "$O_BENCH_THREADS" ]; then die 1 "give the bench thread list once (-t LIST or --bench-threads)"; fi
                O_BENCH_THREADS=$O_T
                ;;
            *)
                _p05_need_uint "-t/--threads" "$O_T" 1
                THREADS=$O_T
                ;;
        esac
    fi
    if [ -n "$O_C" ]; then
        _p05_need_uint "-c/--ctx" "$O_C" 1
        CTX=$O_C
    fi
    export THREADS CTX
}

# ========================================================================================== pinner
PIN_PID=""
PIN_BYTES=0
PIN_STARTED_HERE=0

_p05_pinner_pid() { # live pinner from $PIGEMMA_OUT/pinner.pid -> pid on stdout; 1 if none
    local f="$PIGEMMA_OUT/pinner.pid" pid=""
    if [ ! -r "$f" ]; then return 1; fi
    IFS= read -r pid <"$f" || true
    pid=${pid%$'\r'}
    if ! _pg_is_uint "$pid"; then return 1; fi
    if ! _pg_read1 "/proc/$pid/comm"; then return 1; fi
    case $_R in
        pinner*) printf '%s\n' "$pid" ;;
        *) return 1 ;;
    esac
}

_p05_vmlck_bytes() { # VmLck of a pid in bytes (0 when unreadable)
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

_p05_refresh_pin() {
    PIN_PID=$(_p05_pinner_pid || true)
    PIN_BYTES=0
    if [ -n "$PIN_PID" ]; then PIN_BYTES=$(_p05_vmlck_bytes "$PIN_PID"); fi
}

_p05_pin_stop() {
    if [ "$PIN_STARTED_HERE" = 1 ]; then
        PIN_STARTED_HERE=0
        log "stopping the pinner this run started (pi/06_pin_hotset.sh stop)"
        bash "$PI_DIR/06_pin_hotset.sh" stop >&2 || warn "pi/06_pin_hotset.sh stop failed: check $PIGEMMA_OUT/pinner.pid"
    fi
}

p05_pin_start() {
    if [ "$O_PIN" != 1 ]; then return 0; fi
    if ! mlock_allowed; then
        die 4 "--pin locks model pages with mlock: not allowed on this host (ALLOW_MLOCK=${ALLOW_MLOCK:-auto}; a protected host keeps its page cache reclaimable)"
    fi
    _p05_refresh_pin
    if [ -n "$PIN_PID" ]; then
        log "a pinner is already running (pid $PIN_PID, VmLck $(fmt_gb "$PIN_BYTES")): using it, it is not stopped by this run"
        return 0
    fi
    local pin_cmd=(bash "$PI_DIR/06_pin_hotset.sh" start --model "$SEL_MODEL_PATH" "${O_PIN_ARGS[@]}")
    if [ "$O_DRY" = 1 ]; then
        _p05_join_cmd "${pin_cmd[@]}"
        return 0
    fi
    if [ ! -f "$PI_DIR/06_pin_hotset.sh" ]; then
        die 3 "--pin needs pi/06_pin_hotset.sh, which is not in $PI_DIR"
    fi
    log "starting the hot-set pinner: $(_p05_join_cmd "${pin_cmd[@]}")"
    local rc=0
    "${pin_cmd[@]}" >&2 || rc=$?
    case $rc in
        0) ;;
        2) warn "pi/06_pin_hotset.sh start: partial pin (exit 2); continuing with what is locked" ;;
        *) die "$rc" "pi/06_pin_hotset.sh start failed (exit $rc)" ;;
    esac
    PIN_STARTED_HERE=1
    add_cleanup _p05_pin_stop
    _p05_refresh_pin
    if [ -z "$PIN_PID" ]; then
        warn "pi/06 reported success but no live pinner was found via $PIGEMMA_OUT/pinner.pid"
    else
        log "pinner running: pid $PIN_PID, VmLck $(fmt_gb "$PIN_BYTES")"
    fi
}

# =========================================================================================== vmstat
VMSTAT_PID=""

_p05_vmstat_stop() {
    if [ -n "${VMSTAT_PID:-}" ]; then
        kill "$VMSTAT_PID" 2>/dev/null || true
        wait "$VMSTAT_PID" 2>/dev/null || true
        VMSTAT_PID=""
    fi
}

_p05_vmstat_start() { # _p05_vmstat_start FILE -> 0 when sampling started
    VMSTAT_PID=""
    if [ "$O_DRY" = 1 ] || is_fake || ! have_cmd vmstat; then
        return 1
    fi
    mkdir -p "$(dirname "$1")" || return 1
    vmstat -n 1 >"$1" 2>/dev/null &
    VMSTAT_PID=$!
    add_cleanup _p05_vmstat_stop
    return 0
}

# ===================================================================================== bookkeeping
RESULT_FILES=()
RESULT_LABELS=()
FINAL_RC=0

_p05_export_context() { # everything the annotate / last-run / calibrate helpers read from the environment
    export PI05_VERSION PI05_SUBCMD="$SUBCMD" PI05_HOSTNAME="$HOSTNAME_NOW" PI05_ARGV_JSON
    export PI05_MODEL_ID="$SEL_MODEL_ID" PI05_MODEL_PATH="$SEL_MODEL_PATH" PI05_VERDICT="$SEL_VERDICT"
    export PI05_MODE="$SEL_MODE" PI05_PROJECTED_TOK_S="$SEL_PROJECTED_TOK_S" PI05_REASON="$SEL_REASON"
    export PI05_FLAGS_VERDICT="$FLAGS_VERDICT" PI05_FLAGS_EFF="$FLAGS_EFF" PI05_EXTRA_FLAGS="$EXTRA_FLAGS"
    export PI05_MLOCK_REMOVED="$MLOCK_REMOVED" PI05_FORCED_NR="$FORCED_NR"
    export PI05_PROTECTED_OVERRIDE="${PIGEMMA_PROTECTED_OVERRIDE:-0}"
    export PI05_EMULATED="$EMULATED" PI05_SEL_NOTES="$SEL_NOTES"
    export PI05_FEASIBILITY="$(native_path "$FEAS_JSON")" PI05_FACTS="$(native_path "${PIGEMMA_FACTS:-$PIGEMMA_OUT/facts.json}")"
    export PI05_GATE_NEED="$NEED" PI05_MEM_AVAILABLE="$AVAIL_BYTES" PI05_GATE_RULE="$GATE_RULE"
    export PI05_OUT_DIR="$(native_path "$OUT_DIR")" PI05_THERMAL_BEFORE="$THERMAL_BEFORE"
    export PI05_PINNED=0 PI05_PIN_PID="" PI05_PIN_BYTES=0 PI05_PIN_STARTED_HERE="$PIN_STARTED_HERE"
    if [ -n "$PIN_PID" ]; then
        export PI05_PINNED=1 PI05_PIN_PID="$PIN_PID" PI05_PIN_BYTES="$PIN_BYTES"
    fi
}

_p05_annotate() { # _p05_annotate LABEL VMSTAT_FILE FILE...
    local label=$1 vm=$2
    shift 2
    if [ $# -eq 0 ]; then return 0; fi
    PI05_THERMAL_AFTER=$(_p05_thermal)
    export PI05_THERMAL_AFTER
    _p05_py annotate "--label=$label" "--vmstat=${vm:+$(native_path "$vm")}" "$@" ||
        warn "could not annotate every result file (see above)"
}

_p05_record() { # _p05_record LABEL FILE...
    local label=$1 f
    shift
    for f in "$@"; do
        RESULT_FILES+=("$f")
        RESULT_LABELS+=("$label")
    done
}

_p05_finish() { # _p05_finish RC: calibration, summaries, last_run.json, stdout list; exits RC
    local rc=$1 f labels="" l
    if [ "$O_DRY" = 1 ]; then
        log "dry run: nothing was run and no results were written"
        exit "$rc"
    fi
    for l in "${RESULT_LABELS[@]}"; do labels+="${labels:+,}$l"; done
    _p05_py last-run "--out=$(native_path "$PIGEMMA_OUT/last_run.json")" \
        "--out=$(native_path "$PIGEMMA_OUT/last_${SUBCMD}.json")" "--rc=$rc" "--labels=$labels" \
        "${RESULT_FILES[@]}" || warn "could not write $PIGEMMA_OUT/last_run.json"
    for f in "${RESULT_FILES[@]}"; do
        printf '%s\n' "$f"
    done
    case $rc in
        0) log "done: ${#RESULT_FILES[@]} result file(s) in $OUT_DIR" ;;
        *) warn "finished with exit $rc: ${#RESULT_FILES[@]} result file(s) in $OUT_DIR (failed runs carry a 'failed:' note)" ;;
    esac
    exit "$rc"
}

_p05_calibrate() { # _p05_calibrate RUN FILE...
    local run=$1 out cal_out rc=0 prefer_repack=1
    shift
    if [ "$O_DRY" = 1 ] || [ $# -eq 0 ]; then return 0; fi
    if [ "$O_NO_CALIBRATE" = 1 ]; then
        log "calibration skipped (--no-calibrate)"
        return 0
    fi
    if [ "$EMULATED" = 1 ]; then
        log "calibration skipped: emulated host (qemu timings are not Raspberry Pi measurements)"
        return 0
    fi
    case " $FLAGS_EFF " in *" -nr "* | *" --no-repack "*) prefer_repack=0 ;; esac
    cal_out=${O_CAL_OUT:+$(_p05_abspath "$O_CAL_OUT")}
    cal_out=${cal_out:-$PIGEMMA_HOME/results/pi/calibration.json}
    out=$(_p05_py calibrate "--mode=$SEL_MODE" "--bpt=$SEL_BPT" "--threads=$THREADS" "--prefer-repack=$prefer_repack" \
        "--io-mem=$(native_path "$IO_MEM")" "--feasibility=$(native_path "$FEAS_JSON")" \
        "--out=$(native_path "$cal_out")" "--run=$run" "$@") || rc=$?
    if [ "$rc" != 0 ]; then
        warn "calibration helper failed (exit $rc)"
        return 0
    fi
    local CAL_WRITTEN=0 CAL_PATH="" CAL_VALUE="" CAL_KIND="" CAL_REASON=""
    eval "$out"
    if [ "$CAL_WRITTEN" = 1 ]; then
        PI05_CAL_PATH=$CAL_PATH
        export PI05_CAL_PATH
        log "llama_efficiency $CAL_VALUE ($CAL_KIND) -> $CAL_PATH (pigemma.feasibility / cache_sim use it automatically)"
    fi
}

_p05_bench_py() { # _p05_bench_py SUBCOMMAND ARGS... -> python3 -m pigemma.bench; sets _OUT and _RC
    _RC=0
    _OUT=$(_p05_pyrun -m pigemma.bench --log-level INFO "$@") || _RC=$?
}

_p05_common_bench_args() { # -> array _ARGS (options shared by matrix / completion)
    _ARGS=(--model "$SEL_MODEL_PATH" --bin "$(native_path "$BIN")" --min-free-gb "$MINFREE_GB"
        --out "$(native_path "$OUT_DIR")" --experiment "$O_EXPERIMENT" --idle-check-s "$O_IDLE_S"
        --idle-max-pct "$O_IDLE_PCT")
    if [ -z "${_PG_MSYS:-}" ]; then _ARGS+=(--nice "$NICE_LEVEL"); fi
    if [ -n "$O_TIMEOUT" ]; then _ARGS+=(--timeout "$O_TIMEOUT"); fi
    if [ "$O_DRY" = 1 ]; then _ARGS+=(--dry-run); fi
}

_p05_handle_rc() { # _p05_handle_rc RC WHAT: refusals and missing inputs stop the run, failures continue
    case $1 in
        0) return 0 ;;
        1)
            FINAL_RC=1
            warn "$2 failed (exit 1): its result envelope carries the failure note"
            ;;
        3) _p05_finish_and_die 3 "$2: missing input (exit 3), see the bench output above" ;;
        4) _p05_finish_and_die 4 "$2 refused (exit 4): bench lock held by another model job, or not enough free RAM" ;;
        *) _p05_finish_and_die "$1" "$2 failed (exit $1)" ;;
    esac
}

_p05_finish_and_die() {
    local code=$1
    shift
    if [ "${#RESULT_FILES[@]}" -gt 0 ] && [ "$O_DRY" != 1 ]; then
        local f
        for f in "${RESULT_FILES[@]}"; do printf '%s\n' "$f"; done
    fi
    die "$code" "$*"
}

# ======================================================================================== subcommands
p05_bench() {
    local variants=$O_VARIANTS bthreads plan rc=0 kind name rest runs=() refused=() anon_ok=0 anon_reason="" mlock_ok=0
    if [ -z "$variants" ]; then
        variants=norepack
        if [ "$SEL_FITS_REPACKED" = 1 ] && [ "$REPACK_OK" = 1 ]; then variants="norepack,default"; fi
    fi
    if [ "$SEL_FITS_REPACKED" = 1 ] && [ "$REPACK_OK" = 1 ]; then
        anon_ok=1
    elif [ "$SEL_FITS_REPACKED" != 1 ]; then
        anon_reason="FITS_REPACKED is false for $SEL_MODEL_ID: the default (repack) / nommap variant would allocate an anonymous copy of the weights ($(fmt_gb "$SEL_BYTES")) that does not fit beside the OS"
    else
        anon_reason="protected host (PIGEMMA_I_KNOW=1): the anonymous repack copy is not reclaimable for the host's services; pass --allow-repack to measure it anyway"
    fi
    if mlock_allowed; then mlock_ok=1; fi
    bthreads=$(_p05_bench_threads "${O_BENCH_THREADS:-2,3,4}")
    plan=$(_p05_py plan-bench "--flags=$FLAGS_EFF" "--extra=$EXTRA_FLAGS" "--variants=$variants" "--anon-ok=$anon_ok" \
        "--anon-reason=$anon_reason" "--mlock-ok=$mlock_ok" "--no-verdict-flags=$O_NO_VERDICT_FLAGS") || rc=$?
    if [ "$rc" != 0 ]; then die "$rc" "bench: could not plan the variants (see above)"; fi
    while IFS=$'\t' read -r kind name rest || [ -n "${kind:-}" ]; do
        kind=${kind%$'\r'}
        rest=${rest%$'\r'}
        case $kind in
            RUN) runs+=("$name"$'\t'"$rest") ;;
            REFUSED)
                refused+=("$name")
                warn "bench variant '$name' refused: $rest"
                ;;
        esac
        kind=""
    done <<<"$plan"
    if [ "${#runs[@]}" -eq 0 ]; then
        die 3 "bench: no variant left to run (refused: ${refused[*]:-none}); try --variants norepack"
    fi
    log "bench plan: model $SEL_MODEL_ID, threads $bthreads, pp $O_PP, tg $O_TG, reps $O_REPS, variants: $(printf '%s ' "${runs[@]%%$'\t'*}")"
    p05_pin_start
    _p05_refresh_pin
    _p05_update_minfree
    _p05_export_context
    local r
    for r in "${runs[@]}"; do
        name=${r%%$'\t'*}
        rest=${r#*$'\t'}
        _p05_common_bench_args
        _ARGS+=(--threads "$bthreads" --pp "$O_PP" --tg "$O_TG" -r "$O_REPS" --variants "$name")
        if [ -n "$rest" ]; then _ARGS+=("--flags=$rest"); fi
        log "llama-bench variant $name${rest:+ (+ $rest)}"
        _p05_bench_py matrix "${_ARGS[@]}"
        if [ "$O_DRY" = 1 ]; then
            printf '%s\n' "$_OUT"
            if [ "$_RC" != 0 ]; then die "$_RC" "bench --dry-run failed for variant $name"; fi
            continue
        fi
        _p05_lines "$_OUT"
        if [ "${#_LINES[@]}" -gt 0 ]; then
            _p05_annotate "bench $name" "" "${_LINES[@]}"
            _p05_record "bench:$name" "${_LINES[@]}"
            _p05_py summary "--label=bench $name" "${_LINES[@]}" || true
        fi
        _p05_handle_rc "$_RC" "llama-bench variant $name"
    done
    _p05_calibrate bench "${RESULT_FILES[@]}"
    if [ "${#refused[@]}" -gt 0 ]; then
        warn "variants refused and not measured: ${refused[*]}"
    fi
    _p05_finish "$FINAL_RC"
}

_p05_bench_threads() { # comma list -> list without counts above nproc
    local list=$1 n out="" t arr=()
    n=$(cpu_count)
    IFS=, read -r -a arr <<<"$list"
    for t in "${arr[@]}"; do
        t=${t// /}
        if [ -z "$t" ]; then continue; fi
        _p05_need_uint "bench thread count" "$t" 1
        if [ "$t" -gt "$n" ]; then
            warn "bench thread count $t > nproc $n: dropped"
            continue
        fi
        out+="${out:+,}$t"
    done
    if [ -z "$out" ]; then out=$n; fi
    printf '%s\n' "$out"
}

p05_stream_bench() {
    local prompt_file plan he_file stamp vm_file="" label_en="EN stream-bench" PLAN_VARIANT="" PLAN_REST=""
    local PLAN_THREADS="" PLAN_CTX="" en_files=() he_files=()
    if [ "$O_P_SET" = 1 ]; then
        prompt_file="$PIGEMMA_OUT/.pi05_prompt.$$.txt"
        printf '%s' "$O_P" >"$prompt_file" || die 1 "cannot write $prompt_file"
        add_cleanup "rm -f '$prompt_file'"
        label_en="stream-bench (-p text)"
    elif [ -n "$O_FILE" ]; then
        prompt_file=$(_p05_abspath "$O_FILE")
        if [ ! -f "$prompt_file" ]; then die 3 "prompt file not found: $prompt_file"; fi
        label_en="stream-bench ($(basename "$prompt_file"))"
    else
        prompt_file="$PIGEMMA_HOME/$DEFAULT_PROMPT_REL"
        if [ ! -f "$prompt_file" ]; then die 3 "default prompt file missing: $prompt_file"; fi
    fi
    plan=$(_p05_py plan-run "--flags=$FLAGS_EFF" "--extra=$EXTRA_FLAGS") || die 1 "stream-bench: could not plan the flags"
    eval "$plan"
    if [ -n "$PLAN_THREADS" ] && [ "$PLAN_THREADS" != "$THREADS" ]; then
        warn "the verdict flags say -t $PLAN_THREADS but THREADS=$THREADS: using $THREADS"
    fi
    log "stream-bench plan: model $SEL_MODEL_ID, variant $PLAN_VARIANT${PLAN_REST:+, + $PLAN_REST}, -n $O_N, prompt $prompt_file"
    p05_pin_start
    _p05_refresh_pin
    _p05_update_minfree
    _p05_export_context
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    vm_file="$OUT_DIR/raw/pi05_${SUBCMD}_${SEL_MODEL_ID:-model}_${stamp}.vmstat.log"
    if ! _p05_vmstat_start "$vm_file"; then vm_file=""; fi
    _p05_common_bench_args
    _ARGS+=(--prompt-file "$(native_path "$prompt_file")" -n "$O_N" --threads "$THREADS" -c "$CTX" --variant "$PLAN_VARIANT")
    if [ -n "$PLAN_REST" ]; then _ARGS+=("--flags=$PLAN_REST"); fi
    if [ "$O_CHAT" = 1 ]; then _ARGS+=(--chat); fi
    _p05_bench_py completion "${_ARGS[@]}"
    _p05_vmstat_stop
    if [ "$O_DRY" = 1 ]; then
        printf '%s\n' "$_OUT"
        if [ "$_RC" != 0 ]; then die "$_RC" "stream-bench --dry-run failed"; fi
    else
        _p05_lines "$_OUT"
        en_files=("${_LINES[@]}")
        if [ "${#en_files[@]}" -gt 0 ]; then
            _p05_annotate "$label_en" "$vm_file" "${en_files[@]}"
            _p05_record "stream-bench" "${en_files[@]}"
        fi
        _p05_handle_rc "$_RC" "stream-bench completion"
    fi
    if [ "$O_NO_HE" != 1 ]; then
        he_file="$PIGEMMA_OUT/.pi05_he_prompt.$$.txt"
        add_cleanup "rm -f '$he_file'"
        if _p05_py prompt "--file=$(native_path "$PIGEMMA_HOME/$GEN_PROMPTS_REL")" "--id=$O_HE_ID" \
            "--out=$(native_path "$he_file")"; then
            _p05_common_bench_args
            _ARGS+=(--prompt-file "$(native_path "$he_file")" -n "$O_HE_N" --threads "$THREADS" -c "$CTX"
                --variant "$PLAN_VARIANT" --chat)
            if [ -n "$PLAN_REST" ]; then _ARGS+=("--flags=$PLAN_REST"); fi
            log "Hebrew sample: prompt $O_HE_ID, -n $O_HE_N, chat template"
            _p05_bench_py completion "${_ARGS[@]}"
            if [ "$O_DRY" = 1 ]; then
                printf '%s\n' "$_OUT"
            else
                _p05_lines "$_OUT"
                he_files=("${_LINES[@]}")
                if [ "${#he_files[@]}" -gt 0 ]; then
                    _p05_annotate "HE sample ($O_HE_ID)" "" "${he_files[@]}"
                    _p05_record "he-sample" "${he_files[@]}"
                fi
                _p05_handle_rc "$_RC" "Hebrew sample completion"
            fi
        else
            warn "Hebrew sample skipped: prompt '$O_HE_ID' not found in $GEN_PROMPTS_REL"
        fi
    fi
    if [ "$O_DRY" != 1 ]; then
        if [ "${#en_files[@]}" -gt 0 ]; then
            _p05_py summary "--label=$label_en" --show-output=1 "${en_files[@]}" || true
        fi
        if [ "${#he_files[@]}" -gt 0 ]; then
            _p05_py summary "--label=HE sample ($O_HE_ID)" --show-output=1 "${he_files[@]}" || true
        fi
        if [ "${#en_files[@]}" -gt 0 ]; then _p05_calibrate stream-bench "${en_files[@]}"; fi
    fi
    _p05_finish "$FINAL_RC"
}

_p05_split() { # _p05_split "FLAGS" -> array _SPLIT (shlex rules, so --flags "--api-key 'a b'" stays one word)
    local out
    _SPLIT=()
    if [ -z "${1// /}" ]; then return 0; fi
    out=$(_p05_py split "--text=$1") || die 1 "cannot split the flags: $1"
    eval "_SPLIT=($out)"
}

_p05_llama_args() { # -> array _LARGS: -m MODEL <effective flags> [draft] [extra]
    _LARGS=(-m "$SEL_MODEL_PATH")
    _p05_split "$FLAGS_EFF"
    _LARGS+=("${_SPLIT[@]}")
    if [ -n "$SEL_DRAFT_PATH" ]; then
        _LARGS+=(-md "$SEL_DRAFT_PATH")
        if [ -n "$SEL_DRAFT_SPEC_TYPE" ]; then _LARGS+=(--spec-type "$SEL_DRAFT_SPEC_TYPE"); fi
        if [ -n "$O_DRAFT_NMAX" ]; then _LARGS+=(--spec-draft-n-max "$O_DRAFT_NMAX"); fi
    fi
    _p05_split "$EXTRA_FLAGS"
    _LARGS+=("${_SPLIT[@]}")
}

p05_chat() {
    local cmd rc=0 st
    _p05_llama_args
    if [ -n "$O_N" ]; then _LARGS+=(-n "$O_N"); fi
    if [ "$O_P_SET" = 1 ]; then
        _LARGS+=(-p "$O_P" -st)
    elif [ -n "$O_FILE" ]; then
        local pf
        pf=$(_p05_abspath "$O_FILE")
        if [ ! -f "$pf" ] && [ "$O_DRY" != 1 ]; then die 3 "prompt file not found: $pf"; fi
        _LARGS+=(-f "$pf" -st)
    fi
    cmd=(nice -n "$NICE_LEVEL" "$BIN" "${_LARGS[@]}")
    if [ "$O_DRY" = 1 ]; then
        p05_pin_start
        _p05_join_cmd "${cmd[@]}"
        _p05_finish 0
    fi
    if [ -z "${PIGEMMA_LOCK_HELD:-}" ]; then
        st=0
        _p05_pyrun -m pigemma.paths --log-level WARNING lock --status >/dev/null 2>&1 || st=$?
        if [ "$st" = 1 ]; then
            die 4 "the bench lock (results/.bench.lock) is held by another model-loading job: wait for it (python3 -m pigemma.paths lock --status)"
        fi
    fi
    p05_pin_start
    log "starting llama-cli (Ctrl-C interrupts generation; /exit or Ctrl-D ends the session): $(_p05_join_cmd "${cmd[@]}")"
    "${cmd[@]}" || rc=$?
    log "llama-cli exited with $rc"
    exit "$rc"
}

_p05_port_in_use() { # 0 when something accepts TCP connections on host:port
    local h=$1 p=$2
    case $h in 0.0.0.0 | "" | "*" | "::") h=127.0.0.1 ;; esac
    _pg_timeout 3 bash -c "exec 3<>/dev/tcp/$h/$p" >/dev/null 2>&1
}

p05_server() {
    local cmd rc=0
    _p05_llama_args
    _LARGS+=(--host "$O_HOST" --port "$O_PORT")
    cmd=(nice -n "$NICE_LEVEL" "$BIN" "${_LARGS[@]}")
    if [ "$O_DRY" = 1 ]; then
        p05_pin_start
        _p05_join_cmd "${cmd[@]}"
        _p05_finish 0
    fi
    if _p05_port_in_use "$O_HOST" "$O_PORT"; then
        die 4 "port $O_PORT is already in use on this host: refusing to start a second server (pick --port)"
    fi
    p05_pin_start
    _p05_refresh_pin
    _p05_update_minfree
    case $O_HOST in
        0.0.0.0 | "::") warn "llama-server listens on all interfaces ($O_HOST:$O_PORT): anyone on the LAN can use it (add --flags '--api-key KEY' to require a key)" ;;
    esac
    log "starting llama-server on http://$O_HOST:$O_PORT (holds the bench lock; Ctrl-C stops it): $(_p05_join_cmd "${cmd[@]}")"
    _p05_pyrun -m pigemma.paths lock --hold --min-free-gb "$MINFREE_GB" -- "${cmd[@]}" || rc=$?
    case $rc in
        4) die 4 "llama-server refused: the bench lock is held by another model-loading job, or the free-RAM check failed" ;;
    esac
    log "llama-server exited with $rc"
    exit "$rc"
}

# ======================================================================================== gates
NEED=0
MINFREE_GB=0
GATE_RULE=""

_p05_compute_need() { # NEED (bytes) and GATE_RULE for the selected model / mode / draft / pinner
    local base draft=0
    if { [ "$SUBCMD" = chat ] || [ "$SUBCMD" = server ]; } && [ -n "$SEL_DRAFT_PATH" ]; then
        draft=${SEL_DRAFT_BYTES:-0}
    fi
    case $SEL_MODE in
        stream_moe)
            base=$((SEL_NONEXPERT + MEM_MARGIN))
            GATE_RULE="streaming: non-expert bytes $(fmt_gb "$SEL_NONEXPERT") + 0.5 GB"
            ;;
        *)
            base=$((SEL_BYTES + MEM_MARGIN))
            GATE_RULE="fits mode: file bytes $(fmt_gb "$SEL_BYTES") + 0.5 GB"
            ;;
    esac
    if [ "$draft" -gt 0 ]; then
        base=$((base + draft))
        GATE_RULE+=" + draft $(fmt_gb "$draft")"
    fi
    NEED=$base
    if [ "${PIN_BYTES:-0}" -gt 0 ]; then
        NEED=$((base - PIN_BYTES))
        if [ "$NEED" -lt $((MEM_MARGIN + draft)) ]; then NEED=$((MEM_MARGIN + draft)); fi
        GATE_RULE+=" - pinned VmLck $(fmt_gb "$PIN_BYTES") (min 0.5 GB)"
    fi
}

_p05_update_minfree() { # bench-lock RAM gate for the current pin state
    _p05_compute_need
    MINFREE_GB=$(awk -v b="$NEED" 'BEGIN { printf "%.2f", b / 1e9 + 0.005 }')
}

p05_ram_gate() {
    _p05_refresh_pin
    _p05_compute_need
    if [ "$AVAIL_BYTES" -lt "$NEED" ]; then
        die 3 "not enough available RAM for $SEL_MODEL_ID ($SEL_MODE): MemAvailable $(fmt_gb "$AVAIL_BYTES") < need $(fmt_gb "$NEED") ($GATE_RULE)"
    fi
    log "RAM gate OK: MemAvailable $(fmt_gb "$AVAIL_BYTES") >= $(fmt_gb "$NEED") ($GATE_RULE)"
    MINFREE_GB=$(awk -v b="$NEED" 'BEGIN { printf "%.2f", b / 1e9 + 0.005 }')
}

# =========================================================================================== main
p05_main() {
    _pg_jarr "$@"
    PI05_ARGV_JSON=$_J
    p05_parse_args "$@"
    p05_validate_args
    pigemma_init
    trap 'exit 130' INT
    trap 'exit 143' TERM
    PY=$(pigemma_python) || die 3 "no usable Python >= 3.10 (PIGEMMA_PYTHON, python3, python): sudo apt-get install -y python3"
    EMULATED=0
    if is_emulated; then EMULATED=1; fi
    HOSTNAME_NOW=$(host_name)

    assert_safe_host
    assert_not_busy

    local vargs=(--allow-slow)
    if [ -n "$O_WANT" ]; then vargs+=(--want "$O_WANT"); fi
    if [ -n "$O_SIM" ]; then vargs+=(--sim "$(_p05_abspath "$O_SIM")"); fi
    if [ "${#O_FEAS_ARGS[@]}" -gt 0 ]; then vargs+=(-- "${O_FEAS_ARGS[@]}"); fi
    require_verdict_ok "${vargs[@]}" >&2
    FEAS_JSON=$PIGEMMA_FEASIBILITY_JSON

    AVAIL_KB=$(mem_available_kb)
    _pg_is_uint "$AVAIL_KB" || AVAIL_KB=0
    AVAIL_BYTES=$((AVAIL_KB * 1024))

    local model_arg=$O_MODEL draft_arg=$O_DRAFT sel rc=0
    case $model_arg in
        */* | *.gguf) model_arg=$(native_path "$(_p05_abspath "$model_arg")") ;;
    esac
    case $draft_arg in
        */* | *.gguf) draft_arg=$(native_path "$(_p05_abspath "$draft_arg")") ;;
    esac
    sel=$(_p05_py select "--feasibility=$(native_path "$FEAS_JSON")" "--catalog=$(native_path "$PIGEMMA_CATALOG")" \
        "--model-dir=$(native_path "$MODEL_DIR")" "--model=$model_arg" "--quant-override=$QUANT_OVERRIDE" \
        "--want=$O_WANT" "--draft=$draft_arg" "--spec-type=$O_SPEC_TYPE" "--mem-available=$AVAIL_BYTES" \
        "--threads=$THREADS" "--ctx=$CTX") || rc=$?
    if [ "$rc" != 0 ]; then die "$rc" "model selection failed (see the pi/05 message above)"; fi
    eval "$sel"
    if [ -n "$SEL_NOTES" ]; then
        local n
        IFS='|' read -r -a _notes <<<"$SEL_NOTES" || true
        for n in "${_notes[@]}"; do
            _pg_trim "$n"
            if [ -n "$_R" ]; then warn "model selection: $_R"; fi
        done
    fi

    # ------------------------------------------------------------------ effective flags
    FLAGS_VERDICT=$SEL_FLAGS
    FLAGS_EFF=$(safe_llama_flags "$FLAGS_VERDICT")
    MLOCK_REMOVED=0
    if [ "$FLAGS_EFF" != "$FLAGS_VERDICT" ]; then MLOCK_REMOVED=1; fi
    REPACK_OK=1
    FORCED_NR=0
    if [ "${PIGEMMA_PROTECTED_OVERRIDE:-0}" = 1 ] && [ "$O_ALLOW_REPACK" != 1 ]; then
        REPACK_OK=0
        case " $FLAGS_EFF " in
            *" -nr "* | *" --no-repack "*) ;;
            *)
                if [ "$SUBCMD" != bench ]; then
                    FLAGS_EFF="-nr $FLAGS_EFF"
                    FORCED_NR=1
                    warn "protected host: -nr forced (the anonymous repack copy is not reclaimable for the host's services; --allow-repack overrides)"
                fi
                ;;
        esac
    fi
    EXTRA_FLAGS="${O_FLAGS[*]}"

    banner "pi/05_run.sh $SUBCMD on $HOSTNAME_NOW$( [ "$EMULATED" = 1 ] && printf ' (EMULATED)')" \
        "model    : ${SEL_MODEL_ID:-uncatalogued} ($SEL_KIND ${SEL_QUANT}) $SEL_MODEL_PATH" \
        "verdict  : $SEL_VERDICT, mode $SEL_MODE, projected ${SEL_PROJECTED_TOK_S:-n/a} tok/s (PROJECTED)" \
        "flags    : $FLAGS_EFF${EXTRA_FLAGS:+  + $EXTRA_FLAGS}" \
        "draft    : ${SEL_DRAFT_PATH:-none}${SEL_DRAFT_SPEC_TYPE:+ (--spec-type $SEL_DRAFT_SPEC_TYPE)}"

    # ------------------------------------------------------------------ preconditions
    p05_ram_gate
    if [ "$SEL_MODEL_EXISTS" != 1 ]; then
        local hint="pi/04_fetch_model.sh${SEL_FAMILY:+ --model $SEL_FAMILY}${SEL_QUANT:+ --quant $SEL_QUANT}"
        if [ "$O_DRY" = 1 ]; then
            warn "model file not found: $SEL_MODEL_PATH (dry run continues; download it with $hint)"
        else
            die 3 "model file not found: $SEL_MODEL_PATH (MODEL_DIR=$MODEL_DIR); download it with $hint"
        fi
    fi
    if [ "$SEL_SIZE_MISMATCH" = 1 ]; then
        local msg="$SEL_MODEL_PATH is $SEL_FILE_SIZE bytes but the catalog says $SEL_CATALOG_BYTES for $SEL_MODEL_ID: incomplete download? (pi/04_fetch_model.sh resumes and verifies)"
        if [ "$O_DRY" = 1 ]; then warn "$msg"; else die 3 "$msg"; fi
    fi
    if [ -n "$SEL_DRAFT_PATH" ] && [ "$SEL_DRAFT_EXISTS" != 1 ]; then
        local dmsg="draft file not found: $SEL_DRAFT_PATH (download it with pi/04_fetch_model.sh --draft)"
        if [ "$O_DRY" = 1 ]; then warn "$dmsg"; else die 3 "$dmsg"; fi
    fi
    local tool
    case $SUBCMD in
        bench) tool=llama-bench ;;
        stream-bench) tool=llama-completion ;;
        chat) tool=llama-cli ;;
        server) tool=llama-server ;;
    esac
    BIN=$(llama_bin "$tool")
    if [ ! -f "$BIN" ] && [ ! -f "$BIN.exe" ]; then
        if [ "$O_DRY" = 1 ]; then
            warn "$tool not found at $BIN (dry run continues; build it with pi/02_build_llama.sh)"
        else
            die 3 "$tool not found at $BIN: build llama.cpp with pi/02_build_llama.sh (or set PIGEMMA_LLAMA_BIN / LLAMA_DIR)"
        fi
    fi

    # ------------------------------------------------------------------ outputs
    if [ -n "$O_OUT" ]; then
        OUT_DIR=$(_p05_abspath "$O_OUT")
    elif [ "$EMULATED" = 1 ]; then
        OUT_DIR="$PIGEMMA_HOME/results/pi-emu"
    else
        OUT_DIR="$PIGEMMA_HOME/results/pi"
    fi
    IO_MEM=${O_IO_MEM:+$(_p05_abspath "$O_IO_MEM")}
    IO_MEM=${IO_MEM:-$PIGEMMA_HOME/results/pi/io_mem.json}
    if [ "$O_DRY" != 1 ] && { [ "$SUBCMD" = bench ] || [ "$SUBCMD" = stream-bench ]; }; then
        # chat / server write no results: do not create the results directory for them
        mkdir -p "$OUT_DIR/raw" || die 1 "cannot create $OUT_DIR"
        PI05_BENCH_RAW_DIR=$(native_path "$OUT_DIR")
        export PI05_BENCH_RAW_DIR
    fi
    THERMAL_BEFORE=$(_p05_thermal)
    PI05_CAL_PATH=""
    export PI05_CAL_PATH

    case $SUBCMD in
        bench) p05_bench ;;
        stream-bench) p05_stream_bench ;;
        chat) p05_chat ;;
        server) p05_server ;;
    esac
}

p05_main "$@"
