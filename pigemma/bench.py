"""pigemma.bench - run llama-bench / llama-completion / llama-server and write results envelopes.

One wrapper for every platform (native Windows binaries, the Linux Docker image, the emulated
arm64 Pi container and the real Pi): it builds the command lines for the experiment variants,
runs the binaries resolved by ``paths.bin``, samples the child's memory and I/O while it runs,
parses the outputs and writes one results-JSON envelope per measurement with
``results.write_result`` (kind='measured'). Speculative decoding is measured through
llama-server's per-request ``timings.draft_n / draft_n_accepted`` (``-md`` is not wired into
llama-bench / llama-completion at this llama.cpp commit).

CLI (``python -m pigemma.bench <cmd> --help``)::

    matrix      --model <id|path> [--threads 4,6,12] [--pp 128] [--tg 32] [-r 3]
                [--variants default,norepack,hybrid_negctrl,nommap] [--kvq8] [--out DIR]
    completion  --model <id|path> (--prompt-file P | --prompt TEXT) [-n 128] [--threads 6]
                [--variant V] [--chat] [-c 4096] [--out DIR]
    server-spec --model <id|path> [--variant plain,mtp,mtp-explicit,dflash,dflash-explicit,ngram]
                [--prompts data/prompts/gen_prompts.json] [-n 200] [--threads 6] [--port 18080]
                [--rounds N] [--load-variant V] [--draft F] [--draft-n-max N] [--auto-explicit]
                [--plain-ref F] [--max-prompts N] [--out DIR]
    parse-bench      <llama-bench.json|-> [--model M] [--variant V] [--out DIR]   (no model run)
    parse-completion <stderr.txt|-> [--stdout F] [--model M] [--variant V] [--out DIR]
    parse-server     <response.json>...                                         (prints JSON)
    variants         [--json]                                     (prints the variant table)

Options shared by matrix / completion / server-spec: ``--flags "<extra llama flags>"`` (or extra
flags after ``--``), ``--experiment E0..E8|adhoc``, ``--min-free-gb N`` (bench-lock RAM gate),
``--timeout S``, ``--dry-run`` (print the command lines, run nothing), ``--no-sha``,
``--bin PATH`` (binary override), ``--nice N`` (POSIX), ``--log-level L``, and the idle-machine
check ``--idle-check-s 10 --idle-max-pct 5 [--idle-wait-s S] [--require-idle]``.
stdout gets one line per written result file (its path); progress and summaries go to stderr.

Exit codes (pigemma.paths): 0 every run succeeded; 1 a run failed (its result is still written
with notes=['failed: ...'] and no tok/s) or an unexpected error; 2 usage; 3 missing model /
binary / prompt / draft file; 4 refused (bench lock busy, not enough free RAM, port in use).

Variants -> flags (exact strings; ``variant_flags()``; combine with '+', e.g. ``norepack+kvq8``)::

    default         llama-bench: (none)              others: (none)
    norepack        llama-bench: --repack 0          others: -nr
    hybrid_negctrl  -ot "ffn_(gate_up|down)_exps=CPU" (negative control: still repacks)
    nommap          -lm none
    mlock           -lm mmap+mlock
    kvq8            -fa on -ctk q8_0 -ctv q8_0

Server variants (server-spec): plain (no draft); mtp / dflash (``-md <catalog draft>``, type
inferred by llama.cpp); mtp-explicit / dflash-explicit (``-md`` + ``--spec-type draft-mtp|
draft-dflash``); ngram (``--spec-type ngram-simple``, no draft). ``--load-variant norepack``
adds a load variant's run flags to the server; the result's variant is then ``mtp+norepack``.

Python API (for pigemma.quality, pi/05 and the tests)::

    variant_flags(variant, tool='bench'|'run') -> list[str]; infer_variant(bench_row) -> str
    parse_bench_json(text) -> list[row]       tolerant: truncated array, JSONL, nan/inf, noise
    parse_completion_stderr(text) -> dict      load_s, pp/tg tok/s, buffers, n_ctx, build, ...
    parse_buffer_split(text) -> {'<buft>_mib': MiB}  only bufts that printed a line
    parse_timings(response) -> dict            one llama-server /completion response
    aggregate_timings(records, spec_configured=None) -> dict  accept_rate, spec_active, ...
    parse_membw_json(text) -> dict             io_mem.json 'memory' block from membw --json
    chat_wrap(user_text) -> str                Gemma 4 prompt with thinking disabled
    visible_text(text) -> str; coherence(text) -> (bool, reason, impl)   (pigemma.quality first)
    load_prompts(path) -> list[{id, prompt, max_tokens?, lang?}]
    run_llama_bench(...) -> BenchRun; run_completion(...) -> CompletionRun
    ServerClient(host, port); LlamaServer(model, ...) (context manager, owns its child only)
    run_prompt_set(server_or_client, items, n_predict=None, ...) -> list[request record]
    bench_rows_to_fields / completion_fields / server_fields -> results.write_result kwargs
    ProcessMonitor(pid) - memory / I/O sampler (procfs on Linux, Win32 API on Windows)

Decisions (where the manifest left room; also reported as open issues):

* Result files: ``results.write_result(<out dir>, ...)`` names them
  ``<experiment>_<model>_<variant>_<threads>_<ts>_<run8>.json`` (the run-id fragment keeps the
  pp and tg rows of one llama-bench call apart). ``--out`` defaults to results/<YYYY-mm-dd>/.
  Raw outputs go to ``<out>/raw/<stem>.{stdout.txt,stderr.log,server.log}`` (``*.log`` is
  git-ignored); their paths are recorded under ``extra``.
* Default experiment: matrix / completion / parse-* use E8 on aarch64 hosts (Pi, emulated Pi)
  and E0 elsewhere; server-spec uses E1. ``--experiment`` overrides.
* pp/tg blocks: ``tok_s`` = llama-bench avg_ts (completion: llama_perf tokens per second;
  server: token-weighted sum(n)/sum(ms) over the requests); ``sd`` = llama-bench stddev_ts
  (server: stdev of the per-request rates; completion: null); ``n`` = number of measurements
  behind the mean (llama-bench repetitions, 1 for a completion run, number of server requests).
  Token counts live under ``extra``.
* llama-completion runs as ``llama-completion -m F -f P -n N --temp 0 -t T -c C -no-cnv
  --no-display-prompt -lv 4 [variant flags] [extra flags]``: ``-no-cnv`` because the tool enters
  interactive conversation mode whenever the GGUF has a chat template (it would wait on stdin
  forever); ``--no-display-prompt`` so stdout is only the generated text; ``-lv 4`` because
  llama.cpp's internal INFO logs - including ``load_tensors: <buft> model buffer size`` - map
  to verbosity 4 (TRACE) and are hidden at the default 3. The prompt is copied (chat-wrapped
  with ``--chat``) to a temp file; the original is never modified.
* Chat wrapping uses the Gemma 4 format read from the 26B GGUF's tokenizer.chat_template:
  ``<|turn>user\\n{text}<turn|>\\n<|turn>model\\n<|channel>thought\\n<channel|>`` (thinking
  disabled; the tokenizer adds BOS). The ``<start_of_turn>`` format in docs/design/FACTS.md is
  Gemma 2/3's (Gemma 4's vocab has no such token). pigemma.quality should reuse ``chat_wrap``.
* Memory, Linux: /proc/<pid>/status VmHWM (rss_peak_mb), RssAnon (private_peak_mb), RssFile
  (mapped_mb); /proc/<pid>/io read_bytes and /proc/<pid>/stat majflt. Windows: GetProcess
  MemoryInfo (PROCESS_MEMORY_COUNTERS_EX2): Peak/WorkingSetSize (rss_peak_mb), PrivateUsage /
  PeakPagefileUsage (private_peak_mb = private committed bytes, the value psutil reports as
  ``private``; memory_full_info() is not used because its USS walk is slow on a 14 GB mapping),
  WorkingSetSize - PrivateWorkingSetSize (mapped_mb = resident shareable / file-backed pages).
  psutil is only a fallback (the Pi may not have it). MB = 1e6 bytes; buffer_split is MiB.
* read_bytes_per_token / majflt_per_token (Linux only) cover the GENERATION phase: /proc is
  snapshotted when the first generated byte reaches stdout and when ``common_perf_print``
  reaches stderr (llama.cpp flushes every log entry), and the delta is divided by the
  ``eval time ... / N runs`` count. Load and prompt deltas are under extra.io.
  io_rbytes_per_token uses the cgroup's io.stat rbytes (v1 blkio fallback) and is only filled
  inside a container (elsewhere the cgroup holds unrelated processes). Windows has no
  per-process page-fault I/O counter; ReadTransferCount (excludes mmap paging) is under extra.
* spec_type_effective: 'explicit:<type>' when --spec-type was passed; 'inferred' when a draft
  was passed without --spec-type and the server's generation_settings['speculative.types']
  names a real type; 'none' otherwise (the Gemma 4 MTP sidecars at b11398 report 'none'). The
  raw string is kept in extra.speculative_types_reported. llama-server omits draft_n when it is
  0, so a speculative variant without draft_n records draft_n = 0 (a measurement).
* identical_to_plain: fraction of prompts whose visible text (thinking channel stripped)
  equals the plain run's; per-prompt booleans and difflib similarity ratios are under extra.
  The plain reference is the plain run of the same invocation, else ``--plain-ref FILE``, else
  the newest E1 plain result for the same model, load variant and n_predict in --out. The plain
  variant records null (it is the reference); spec_active is null for plain.
* server-spec refuses (exit 4) when anything already answers on the port and only ever
  terminates the llama-server it started (terminate, kill after 10 s). Its load_s is the
  server's own log timestamp of ``llama_server: model loaded`` (llama-server only starts
  listening after the load, and a refused connect costs ~2 s on Windows, so health polling alone
  is coarse); the wall time until /health said ok is extra.health_wait_s.
* matrix / completion / server-spec hold ``paths.bench_lock`` for their whole duration (an
  inherited lock from ``paths lock --hold`` passes through); parse-* take no lock.
* Idle-machine check (docs/design/FACTS.md: desktop timings are only valid when total CPU is
  < 5 % for 10 s before starting): before EVERY child (each llama-bench variant, each
  completion, each server start) the total system CPU utilisation is measured for
  ``--idle-check-s`` seconds (default 10; 0 disables) from /proc/stat, GetSystemTimes or psutil
  (inside Docker that is the Docker VM's CPU, scope 'container-vm': check the host before
  launching container runs).
  The measurement is stored in every envelope as extra.background_cpu {mean_pct, max_pct,
  seconds, idle, max_allowed_pct, source}; above ``--idle-max-pct`` a note says the timing is
  not idle-machine quality. ``--idle-wait-s S`` keeps re-measuring for up to S seconds waiting
  for an idle window; ``--require-idle`` refuses (exit 4) instead of running on a busy machine.
* server-spec ``--rounds N`` repeats the variant list N times in order (plain, mtp-explicit,
  plain, mtp-explicit, ...: the interleaved A,B,A,B design E1 needs); one envelope per variant
  and round with extra.round / extra.rounds; identical_to_plain compares with the newest plain
  run (the same round's when plain is in the list).
* llama-bench feeds random token ids: its envelopes carry extra.token_source saying so (engine
  throughput only, never routing / locality conclusions).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import datetime as _dt
import difflib
import hashlib
import http.client
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from pigemma import paths, results

try:  # optional: only a fallback for the Win32 / procfs readers below
    import psutil  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - depends on the interpreter  # noqa: BLE001
    psutil = None  # type: ignore[assignment]

log = logging.getLogger("pigemma.bench")

__all__ = [
    "VARIANT_FLAGS", "HYBRID_OT", "SERVER_VARIANTS", "SERVER_VARIANT_SPEC", "variant_flags",
    "split_variant", "infer_variant", "parse_bench_json", "parse_completion_stderr",
    "parse_buffer_split", "parse_timings", "aggregate_timings", "parse_membw_json", "chat_wrap",
    "visible_text", "strip_thinking", "coherence", "load_prompts", "run_llama_bench",
    "run_completion", "ServerClient", "LlamaServer", "run_prompt_set", "bench_rows_to_fields",
    "completion_fields", "server_fields", "ProcessMonitor", "resolve_model", "resolve_draft",
    "measure_background_cpu", "idle_check", "main",
]

# --------------------------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------------------------
HYBRID_OT = "ffn_(gate_up|down)_exps=CPU"

# variant -> {'bench': llama-bench flags, 'run': llama-completion / llama-server / llama-cli flags}
VARIANT_FLAGS: dict[str, dict[str, tuple[str, ...]]] = {
    "default": {"bench": (), "run": ()},
    "norepack": {"bench": ("--repack", "0"), "run": ("-nr",)},
    "hybrid_negctrl": {"bench": ("-ot", HYBRID_OT), "run": ("-ot", HYBRID_OT)},
    "nommap": {"bench": ("-lm", "none"), "run": ("-lm", "none")},
    "mlock": {"bench": ("-lm", "mmap+mlock"), "run": ("-lm", "mmap+mlock")},
    "kvq8": {"bench": ("-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"),
             "run": ("-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0")},
}
VARIANT_NOTES: dict[str, str] = {
    "default": "llama.cpp defaults: mmap + CPU_REPACK copy of every repackable tensor",
    "norepack": "whole file stays CPU_Mapped (the only page-cache streaming mode at a7b94df)",
    "hybrid_negctrl": "negative control: -ot to the CPU buft still repacks the experts",
    "nommap": "-lm none: weights read into anonymous memory",
    "mlock": "-lm mmap+mlock: mapped and locked (never on the ContextFriend Pi)",
    "kvq8": "flash attention + q8_0 KV cache",
}

# server variant -> (catalog draft_type or None, --spec-type value or None)
SERVER_VARIANT_SPEC: dict[str, tuple[str | None, str | None]] = {
    "plain": (None, None),
    "mtp": ("mtp", None),
    "mtp-explicit": ("mtp", "draft-mtp"),
    "dflash": ("dflash", None),
    "dflash-explicit": ("dflash", "draft-dflash"),
    "ngram": (None, "ngram-simple"),
}
SERVER_VARIANTS: tuple[str, ...] = tuple(SERVER_VARIANT_SPEC)

DEFAULT_PORT = 18080
DEFAULT_N_CTX = 4096
DEFAULT_SAMPLE_INTERVAL_S = 0.25
HEALTH_TIMEOUT_S = 300.0
HEALTH_POLL_S = 0.25          # /health poll period (also the TCP probe timeout)
PORT_PROBE_TIMEOUT_S = 0.5    # 'is anything already listening' check before a start
STOP_GRACE_S = 10.0
WARMUP_TOKENS = 16
DEFAULT_SERVER_N_PREDICT = 200
SIMILAR_THRESHOLD = 0.95
MB = 1_000_000
MiB = 1024 * 1024
STDERR_TAIL_LINES = 6
DEFAULT_PROMPTS = paths.PROMPTS_DIR / "gen_prompts.json"
MAX_SAMPLES = 400_000  # ~28 h at 0.25 s


class BenchError(RuntimeError):
    """A run could not be attempted (bad input); exit 1."""


class MissingInputError(BenchError):
    """Model, binary, prompt or draft file missing; exit 3."""


class RefusedError(BenchError):
    """Refused for safety (port already in use, ...); exit 4."""


class ServerStartError(BenchError):
    """llama-server exited or never became healthy."""


class ServerRequestError(BenchError):
    """A /completion request failed (HTTP error, invalid JSON, timeout)."""


def split_variant(variant: str) -> list[str]:
    """'norepack+kvq8' -> ['norepack', 'kvq8']; ValueError for an unknown part."""
    parts = [p.strip() for p in str(variant).split("+") if p.strip()]
    if not parts:
        raise ValueError("empty variant")
    for p in parts:
        if p not in VARIANT_FLAGS:
            raise ValueError(f"unknown variant {p!r} (known: {', '.join(VARIANT_FLAGS)}; "
                             "combine with '+')")
    return parts


def variant_flags(variant: str, tool: str = "run") -> list[str]:
    """Exact llama.cpp flags of a (possibly '+'-combined) load variant.

    ``tool='bench'`` gives the llama-bench spelling (``--repack 0``), ``tool='run'`` the
    llama-completion / llama-server / llama-cli spelling (``-nr``).
    """
    if tool not in ("bench", "run"):
        raise ValueError("tool must be 'bench' or 'run'")
    out: list[str] = []
    for p in split_variant(variant):
        out.extend(VARIANT_FLAGS[p][tool])
    return out


def infer_variant(row: Mapping[str, Any]) -> str:
    """Load variant implied by a llama-bench JSON row (repack, overrides, load_mode, KV types)."""
    parts: list[str] = []
    if row.get("repack") is False:
        parts.append("norepack")
    ovr = str(row.get("tensor_buft_overrides") or "none")
    if ovr not in ("none", "") and "exps" in ovr:
        parts.append("hybrid_negctrl")
    lm = str(row.get("load_mode") or "auto")
    if lm == "none":
        parts.append("nommap")
    elif lm in ("mlock", "mmap+mlock"):
        parts.append("mlock")
    if str(row.get("type_k") or "") == "q8_0" and str(row.get("type_v") or "") == "q8_0":
        parts.append("kvq8")
    return "+".join(parts) if parts else "default"


def default_experiment(subcommand: str) -> str:
    """E1 for server-spec; E8 on aarch64 hosts (Pi / emulated Pi); else E0."""
    if subcommand == "server-spec":
        return "E1"
    return "E8" if paths.machine() == "aarch64" else "E0"


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
# llama.cpp log prefix: "0.24.337.496 I " (minutes.seconds.ms.us + level letter)
_TS_PREFIX = r"\d+\.\d{2}\.\d{3}\.\d{3}"
_LOG_PREFIX_RE = re.compile(rf"^\s*(?:{_TS_PREFIX}\s+(?:[IWEDT]\s+)?)")
_LOG_LEVEL_E_RE = re.compile(rf"^\s*{_TS_PREFIX}\s+E\s")
_ERR_WORD_RE = re.compile(r"\berror\b|failed|exception|abort|out of memory|cannot|unable to",
                          re.IGNORECASE)


def _now_utc() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _ts_compact(d: _dt.datetime | None = None) -> str:
    return (d or _now_utc()).strftime("%Y%m%dT%H%M%SZ")


def _safe_part(x: Any) -> str:
    s = re.sub(r"[^A-Za-z0-9.+\-]+", "-", str(x)).strip("-.")
    return s or "na"


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for p in str(text).split(","):
        p = p.strip()
        if not p:
            continue
        v = int(p)
        if v < 1:
            raise ValueError(f"values must be >= 1, got {v}")
        out.append(v)
    if not out:
        raise ValueError("empty list")
    return out


def _num(x: Any) -> float | None:
    """Finite float or None."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int_or_none(x: Any) -> int | None:
    v = _num(x)
    return None if v is None else int(v)


_TS_GROUPS_RE = re.compile(r"^\s*(\d+)\.(\d{2})\.(\d{3})\.(\d{3})\s")


def log_time_s(text: str, marker: str) -> float | None:
    """Seconds since the tool's log start of the first line containing ``marker`` (llama.cpp
    prefixes every log line with minutes.seconds.ms.us), or None."""
    for ln in (text or "").splitlines():
        if marker in ln:
            m = _TS_GROUPS_RE.match(ln)
            if m:
                mi, se, ms, us = (int(g) for g in m.groups())
                return round(mi * 60 + se + ms / 1e3 + us / 1e6, 3)
    return None


def _strip_log_prefix(line: str) -> str:
    return _LOG_PREFIX_RE.sub("", line, count=1)


def _tail_lines(text: str, n: int = STDERR_TAIL_LINES) -> list[str]:
    """Last n meaningful lines of a log (timestamp / level prefixes stripped)."""
    keep: list[str] = []
    for ln in text.splitlines():
        s = _strip_log_prefix(ln).strip()
        if not s or s.startswith("load_backend:"):
            continue
        keep.append(s)
    return keep[-n:]


def _err_lines(text: str, n: int = STDERR_TAIL_LINES) -> list[str]:
    """Error-looking lines (level E, 'error', 'failed', ...) plus the line just before the
    first of them (llama.cpp often states the cause there), else the tail of the log."""
    keep: list[tuple[str, bool]] = []
    for ln in text.splitlines():
        raw = ln.rstrip()
        s = _strip_log_prefix(raw).strip()
        if not s or s.startswith("load_backend:"):
            continue
        keep.append((s, bool(_LOG_LEVEL_E_RE.match(raw) or _ERR_WORD_RE.search(s))))
    first = next((i for i, (_, e) in enumerate(keep) if e), None)
    if first is None:
        return [s for s, _ in keep[-n:]]
    sel = [s for i, (s, e) in enumerate(keep) if e or i == first - 1]
    return sel[-n:]


def failure_note(stderr_text: str, *, prefix: str = "failed", extra: str | None = None) -> str:
    """'failed: <reason>; <last error lines of stderr>' as one line of at most 700 chars."""
    lines = _err_lines(stderr_text or "")
    msg = " | ".join(lines) if lines else "(no stderr output)"
    if extra:
        msg = f"{extra}; {msg}"
    s = f"{prefix}: {msg}"
    return s if len(s) <= 700 else s[:697] + "..."


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _rel(p: str | os.PathLike[str] | None) -> str | None:
    return None if p is None else paths.repo_relative(p)


def _cmd_str(argv: Sequence[str]) -> str:
    if paths.is_windows():
        return subprocess.list2cmdline([str(a) for a in argv])
    return shlex.join([str(a) for a in argv])


def _decode(b: bytes | str | None) -> str:
    if b is None:
        return ""
    if isinstance(b, str):
        return b
    return b.decode("utf-8", errors="replace")


def _write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# --------------------------------------------------------------------------------------------
# Catalog / model / draft resolution
# --------------------------------------------------------------------------------------------
_CATALOG: dict[str, Any] | None = None


def load_catalog() -> dict[str, Any]:
    """pigemma/data/catalog.json (cached; {} with a warning when unreadable)."""
    global _CATALOG
    if _CATALOG is None:
        try:
            with open(paths.CATALOG_PATH, encoding="utf-8") as f:
                _CATALOG = json.load(f)
        except (OSError, ValueError) as e:
            log.warning("catalog %s unreadable (%s); draft lookup limited to the paths registry",
                        paths.CATALOG_PATH, e)
            _CATALOG = {}
    return _CATALOG


def catalog_model(model_id: str | None) -> dict[str, Any] | None:
    if not model_id:
        return None
    for m in load_catalog().get("models", []) or []:
        if isinstance(m, Mapping) and m.get("id") == model_id:
            return dict(m)
    return None


def catalog_id_for(path_or_id: str | os.PathLike[str]) -> str | None:
    """Catalog id for an id or for a path whose basename matches a catalog file."""
    s = os.fspath(path_or_id)
    if catalog_model(s):
        return s
    name = Path(s).name
    for m in load_catalog().get("models", []) or []:
        if isinstance(m, Mapping) and m.get("file") and Path(str(m["file"])).name == name:
            return str(m.get("id"))
    return None


def _catalog_candidates(entry: Mapping[str, Any]) -> list[Path]:
    cands: list[Path] = []
    lp = entry.get("local_path")
    file = str(entry.get("file") or "")
    repo = str(entry.get("repo") or "")
    if lp:
        cands.append(paths.REPO_ROOT / str(lp))
    dirs: list[Path] = []
    for var in ("PIGEMMA_MODELS_DIR", "MODEL_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            dirs.append(Path(v).expanduser())
    dirs.append(paths.MODELS_DIR)
    for d in dirs:
        if repo and file:
            cands.append(d / repo.replace("/", "__") / file)
        if file:
            cands.append(d / Path(file).name)
    out: list[Path] = []
    for c in cands:
        if c not in out:
            out.append(c)
    return out


def resolve_model(spec: str | os.PathLike[str]) -> tuple[Path, str | None]:
    """(path, registry/catalog id or None) for a paths.MODELS id, a catalog id or a path."""
    s = os.fspath(spec)
    if s in paths.MODELS:
        return paths.model_path(s), s
    entry = catalog_model(s)
    if entry is not None:
        cands = _catalog_candidates(entry)
        for c in cands:
            if c.is_file():
                return c, s
        return (cands[0] if cands else Path(s)), s
    p = paths.model_path(s)
    return p, (paths.model_id_for(p) or catalog_id_for(p))


def resolve_draft(model_id: str | None, model_path: Path | None, draft_type: str
                  ) -> tuple[Path | None, str | None]:
    """(draft path, draft id) of the catalog draft of type 'mtp'|'dflash' for a main model."""
    mid = model_id or (catalog_id_for(model_path) if model_path else None)
    ids: list[str] = []
    entry = catalog_model(mid) if mid else None
    if entry is not None:
        ids.extend(str(x) for x in entry.get("drafts") or [])
    if mid in paths.MODELS:
        ids.extend(d for d in paths.drafts_for(mid) if d not in ids)
    for did in ids:
        d = catalog_model(did) or {}
        dtype = d.get("draft_type")
        if dtype is None and did in paths.MODELS:
            st = paths.MODELS[did].get("spec_type") or ""
            dtype = st.replace("draft-", "") if st else None
        if dtype != draft_type:
            continue
        if did in paths.MODELS:
            p = paths.model_path(did)
            if p.is_file():
                return p, did
        for c in _catalog_candidates(d):
            if c.is_file():
                return c, did
        p = paths.model_path(did) if did in paths.MODELS else (_catalog_candidates(d) or [None])[0]
        return p, did
    return None, None


def model_block(path: Path, model_id: str | None, *, with_sha: bool = True) -> dict[str, Any]:
    """The envelope ``model`` block (paths.model_identity, catalog id / quant filled in)."""
    blk = paths.model_identity(path, with_sha=with_sha)
    if model_id and model_id not in paths.MODELS and blk.get("id") != model_id:
        blk["id"] = model_id
    entry = catalog_model(blk.get("id"))
    if entry and (blk.get("quant") in (None, "unknown")) and entry.get("quant"):
        blk["quant"] = entry["quant"]
    if blk.get("bytes") is None and entry and isinstance(entry.get("bytes"), int):
        blk["bytes"] = entry["bytes"]
    return blk


# --------------------------------------------------------------------------------------------
# Binaries
# --------------------------------------------------------------------------------------------
def _tool_cmd(tool: str, override: str | os.PathLike[str] | Sequence[str] | None = None
              ) -> list[str]:
    """Command prefix for a llama.cpp tool (``paths.bin``) or an explicit override."""
    if override is not None:
        if isinstance(override, (list, tuple)):
            return [str(x) for x in override]
        return [os.fspath(override)]  # type: ignore[arg-type]
    return [str(paths.bin(tool))]


def _check_cmd(cmd: Sequence[str]) -> None:
    exe = cmd[0]
    if os.path.sep in exe or "/" in exe or (paths.is_windows() and ":" in exe):
        if not Path(exe).is_file():
            raise MissingInputError(f"binary not found: {exe}")
    elif shutil.which(exe) is None:
        raise MissingInputError(f"binary not found on PATH: {exe}")


_VERSION_RES = (
    re.compile(r"\(build (\d+), commit ([0-9a-fA-F]{7,40})\)"),
    re.compile(r"version:\s*(\d+)\s*\(([0-9a-fA-F]{7,40})\)"),
    re.compile(r"\bbuild[ :]+(\d+)\s*\(([0-9a-fA-F]{7,40})\)"),
)
_VERSION_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


def parse_build_info(text: str) -> tuple[int | None, str | None]:
    """(build_number, commit) from '--version' output or a log line; (None, None) if absent."""
    for rx in _VERSION_RES:
        m = rx.search(text or "")
        if m:
            return int(m.group(1)), m.group(2).lower()
    m = re.search(r"\bbuild:\s*([0-9a-fA-F]{7,40})\s*\((\d+)\)", text or "")
    if m:
        return int(m.group(2)), m.group(1).lower()
    return None, None


def binary_version(cmd: Sequence[str], timeout: float = 60.0) -> dict[str, Any]:
    """Run ``<cmd> --version`` once (cached) -> {build_number, commit, raw, ok}."""
    key = tuple(cmd)
    if key in _VERSION_CACHE:
        return _VERSION_CACHE[key]
    info: dict[str, Any] = {"build_number": None, "commit": None, "raw": None, "ok": False}
    try:
        cp = subprocess.run(list(cmd) + ["--version"], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        raw = _decode(cp.stdout) + _decode(cp.stderr)
        bn, commit = parse_build_info(raw)
        info.update(build_number=bn, commit=commit, raw=raw.strip()[-400:], ok=bn is not None)
    except (OSError, subprocess.SubprocessError) as e:
        info["raw"] = f"{type(e).__name__}: {e}"
    _VERSION_CACHE[key] = info
    return info


def _build_note(build_number: Any, commit: Any) -> str | None:
    if build_number is None or commit is None:
        return None
    if str(commit).lower().startswith(paths.LLAMA_COMMIT) and build_number != paths.LLAMA_BUILD:
        return (f"llama.cpp reported build_number {build_number} for the pinned commit "
                f"{paths.LLAMA_COMMIT} (shallow-clone build count; the pinned build is "
                f"b{paths.LLAMA_BUILD})")
    if not str(commit).lower().startswith(paths.LLAMA_COMMIT):
        return f"llama.cpp commit {commit} differs from the pinned {paths.LLAMA_COMMIT}"
    return None


# --------------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------------
_NONFINITE_RE = re.compile(r"(?<=[:\[,\s])-?(?:nan|inf|infinity)(?=\s*[,\]}\n])", re.IGNORECASE)


def _bench_row_norm(row: Mapping[str, Any]) -> dict[str, Any]:
    r = dict(row)
    n_prompt = _int_or_none(r.get("n_prompt")) or 0
    n_gen = _int_or_none(r.get("n_gen")) or 0
    n_depth = _int_or_none(r.get("n_depth")) or 0
    if n_prompt > 0 and n_gen == 0:
        kind, test = "pp", f"pp{n_prompt}"
    elif n_gen > 0 and n_prompt == 0:
        kind, test = "tg", f"tg{n_gen}"
    else:
        kind, test = "pg", f"pp{n_prompt}+tg{n_gen}"
    if n_depth:
        test += f" @ d{n_depth}"
    samples = r.get("samples_ts")
    r.update(
        kind=kind, test=test, n_prompt=n_prompt, n_gen=n_gen, n_depth=n_depth,
        n_threads=_int_or_none(r.get("n_threads")),
        build_number=_int_or_none(r.get("build_number")),
        avg_ts=_num(r.get("avg_ts")), stddev_ts=_num(r.get("stddev_ts")),
        n_samples=len(samples) if isinstance(samples, list) else None,
    )
    if isinstance(r.get("build_commit"), str):
        r["build_commit"] = r["build_commit"].strip().lower()
    if isinstance(r.get("cpu_info"), str):
        r["cpu_info"] = r["cpu_info"].strip()
    return r


def parse_bench_json(text: str | bytes) -> list[dict[str, Any]]:
    """Rows of ``llama-bench -o json`` (or ``-o jsonl``) output, normalised.

    Tolerates a truncated array (llama-bench killed or failing mid-run: the complete rows are
    returned), bare nan/inf values (``std::to_string`` of a NaN stddev), log lines before the
    JSON and JSONL. Each row keeps every llama-bench field and gains kind ('pp'|'tg'|'pg'),
    test ('pp128' / 'tg32' / 'pp128+tg32'), n_samples, and float/None avg_ts / stddev_ts.
    """
    s = _decode(text)
    if not s.strip():
        return []
    s = _NONFINITE_RE.sub("null", s)
    dec = json.JSONDecoder()
    m = re.search(r"^\s*\[\s*$", s, re.MULTILINE) or re.search(r"^\s*\[\s*\{", s, re.MULTILINE)
    rows: list[dict[str, Any]] = []
    if m:
        pos = s.index("[", m.start()) + 1
        while pos < len(s):
            while pos < len(s) and s[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(s) or s[pos] == "]":
                break
            try:
                obj, pos = dec.raw_decode(s, pos)
            except json.JSONDecodeError:
                log.warning("llama-bench JSON truncated after %d complete row(s)", len(rows))
                break
            if isinstance(obj, dict):
                rows.append(obj)
    else:  # JSONL (one object per line) or a single object
        for ln in s.splitlines():
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                obj = json.loads(ln)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    rows = [r for r in rows if "avg_ts" in r or "n_threads" in r or "n_prompt" in r]
    return [_bench_row_norm(r) for r in rows]


_RE_LOAD = re.compile(r"\bload time\s*=\s*([0-9.]+)\s*ms")
_RATE = r"([0-9.]+|-?nan|-?inf)"
_RE_PROMPT = re.compile(
    rf"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)\s*"
    rf"\(\s*{_RATE}\s*ms per token,\s*{_RATE}\s*tokens per second\s*\)", re.IGNORECASE)
_RE_EVAL = re.compile(
    rf"(?<!prompt )\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)\s*"
    rf"\(\s*{_RATE}\s*ms per token,\s*{_RATE}\s*tokens per second\s*\)", re.IGNORECASE)
_RE_TOTAL = re.compile(r"\btotal time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens")
_RE_BUF = re.compile(r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB")
_RE_OTHER_BUF = re.compile(r"\b(\S+)\s+(KV|compute|output|RS|recurrent) buffer size =\s+([\d.]+) MiB")
_RE_GENERATE = re.compile(
    r"generate: n_ctx = (\d+), n_batch = (\d+), n_predict = (-?\d+), n_keep = (\d+)")
_RE_SYSINFO = re.compile(r"system_info: n_threads = (\d+) \(n_threads_batch = (\d+)\)")
_RE_N_CTX_SERVER = re.compile(r"n_ctx_slot = (\d+)")


def parse_buffer_split(text: str) -> dict[str, float]:
    """``load_tensors: <buft> model buffer size = X MiB`` lines -> {'<buft>_mib': sum MiB}.

    Only buffer types that printed a line appear (an absent CPU_REPACK_mib means "no repack
    buffer", which is what the E0 gate checks); llama.cpp prints them only at ``-lv 4``.
    """
    out: dict[str, float] = {}
    for m in _RE_BUF.finditer(text or ""):
        key = f"{m.group(1)}_mib"
        out[key] = round(out.get(key, 0.0) + float(m.group(2)), 2)
    return out


def parse_completion_stderr(text: str) -> dict[str, Any]:
    """Parse llama-completion (or llama-cli) stderr.

    Returns {load_ms, load_s, prompt_ms, n_prompt, pp_tok_s, eval_ms, n_eval, tg_tok_s,
    total_ms, n_total, buffer_split, buffers_other, n_ctx, n_batch, n_predict, n_threads,
    n_threads_batch, build_number, build_commit, has_perf}; missing values are None. The
    first ``common_perf_print`` block wins (llama-completion prints exactly one).
    """
    t = text or ""
    out: dict[str, Any] = dict.fromkeys(
        ("load_ms", "load_s", "prompt_ms", "n_prompt", "pp_tok_s", "eval_ms", "n_eval",
         "tg_tok_s", "total_ms", "n_total", "n_ctx", "n_batch", "n_predict", "n_threads",
         "n_threads_batch", "build_number", "build_commit"))
    if m := _RE_LOAD.search(t):
        out["load_ms"] = float(m.group(1))
        out["load_s"] = round(out["load_ms"] / 1000.0, 3)
    if m := _RE_PROMPT.search(t):
        out["prompt_ms"] = float(m.group(1))
        out["n_prompt"] = int(m.group(2))
        out["pp_tok_s"] = _num(m.group(4))
    if m := _RE_EVAL.search(t):
        out["eval_ms"] = float(m.group(1))
        out["n_eval"] = int(m.group(2))
        out["tg_tok_s"] = _num(m.group(4))
    if m := _RE_TOTAL.search(t):
        out["total_ms"] = float(m.group(1))
        out["n_total"] = int(m.group(2))
    if m := _RE_GENERATE.search(t):
        out["n_ctx"], out["n_batch"], out["n_predict"] = (int(m.group(1)), int(m.group(2)),
                                                         int(m.group(3)))
    if m := _RE_SYSINFO.search(t):
        out["n_threads"], out["n_threads_batch"] = int(m.group(1)), int(m.group(2))
    out["build_number"], out["build_commit"] = parse_build_info(t)
    out["buffer_split"] = parse_buffer_split(t)
    other: dict[str, float] = {}
    for m in _RE_OTHER_BUF.finditer(t):
        key = f"{m.group(2)}_{m.group(1)}_mib"
        other[key] = round(other.get(key, 0.0) + float(m.group(3)), 2)
    out["buffers_other"] = other
    out["has_perf"] = out["eval_ms"] is not None or out["prompt_ms"] is not None
    return out


def parse_timings(resp: Mapping[str, Any]) -> dict[str, Any]:
    """One llama-server /completion response -> flat timing record.

    draft_n / draft_n_accepted are None when the server did not report them (it omits them
    when no token was drafted); accept_rate = accepted / drafted or None.
    """
    t = resp.get("timings") if isinstance(resp, Mapping) else None
    t = t if isinstance(t, Mapping) else {}
    gs = resp.get("generation_settings") if isinstance(resp, Mapping) else None
    gs = gs if isinstance(gs, Mapping) else {}
    draft_n = _int_or_none(t.get("draft_n"))
    acc = _int_or_none(t.get("draft_n_accepted"))
    return {
        "prompt_n": _int_or_none(t.get("prompt_n")),
        "prompt_ms": _num(t.get("prompt_ms")),
        "prompt_per_second": _num(t.get("prompt_per_second")),
        "predicted_n": _int_or_none(t.get("predicted_n")),
        "predicted_ms": _num(t.get("predicted_ms")),
        "predicted_per_second": _num(t.get("predicted_per_second")),
        "cache_n": _int_or_none(t.get("cache_n")),
        "draft_n": draft_n,
        "draft_n_accepted": acc,
        "draft_reported": "draft_n" in t,
        "accept_rate": (acc / draft_n) if (draft_n and acc is not None) else None,
        "speculative_types": gs.get("speculative.types"),
        "stop_type": resp.get("stop_type") if isinstance(resp, Mapping) else None,
        "tokens_predicted": _int_or_none(resp.get("tokens_predicted")) if isinstance(resp, Mapping) else None,
        "tokens_evaluated": _int_or_none(resp.get("tokens_evaluated")) if isinstance(resp, Mapping) else None,
        "truncated": resp.get("truncated") if isinstance(resp, Mapping) else None,
    }


def _spec_types(reported: Any) -> list[str]:
    if not isinstance(reported, str):
        return []
    return [x.strip() for x in reported.split(",") if x.strip() and x.strip() != "none"]


def aggregate_timings(records: Sequence[Mapping[str, Any]],
                      spec_configured: bool | None = None) -> dict[str, Any]:
    """Aggregate per-request timing records (parse_timings output, possibly with 'error').

    tg/pp tok_s are time-weighted with llama-server's own conventions: pp = sum(prompt_n) /
    sum(prompt_ms); tg = sum(predicted_n - 1) / sum(predicted_ms) (the first generated token
    comes from the prompt logits; server-common.h n_gen_steps), so one request reproduces its
    predicted_per_second exactly. sd is the stdev of the per-request rates (None for < 2). With speculation configured: draft sums (absent = 0, the server
    omits a zero), accept_rate = sum(accepted)/sum(drafted) or None, spec_active = every
    successful request drafted > 0. Without speculation, or without any successful request,
    they are None. ``spec_configured=None``
    infers it from the records (draft fields or speculative types reported).
    """
    ok = [r for r in records if not r.get("error") and r.get("predicted_n") is not None]
    if spec_configured is None:
        spec_configured = any(r.get("draft_reported") or _spec_types(r.get("speculative_types"))
                              for r in records)

    def rate(n_key: str, ms_key: str, ps_key: str, free: int) -> dict[str, Any] | None:
        pairs = [(int(r[n_key]), float(r[ms_key])) for r in ok
                 if r.get(n_key) is not None and _num(r.get(ms_key)) is not None]
        tot_n = sum(n for n, _ in pairs)
        steps = sum(max(n - free, 0) for n, _ in pairs)
        tot_ms = sum(ms for _, ms in pairs)
        if not pairs or tot_ms <= 0:
            return None
        rates = [float(r[ps_key]) for r in ok if _num(r.get(ps_key)) is not None]
        sd = statistics.stdev(rates) if len(rates) >= 2 else None
        return {"tok_s": steps / (tot_ms / 1000.0), "sd": sd, "n": len(pairs),
                "tokens": tot_n, "steps": steps, "ms": round(tot_ms, 3)}

    out: dict[str, Any] = {
        "n_requests": len(records), "n_ok": len(ok),
        "tg": rate("predicted_n", "predicted_ms", "predicted_per_second", 1),
        "pp": rate("prompt_n", "prompt_ms", "prompt_per_second", 0),
        "spec_configured": bool(spec_configured),
        "draft_n": None, "draft_n_accepted": None, "accept_rate": None, "spec_active": None,
        "n_drafting": None,
    }
    if spec_configured and ok:  # nothing measured -> leave the draft fields None
        dn = sum(int(r.get("draft_n") or 0) for r in ok)
        da = sum(int(r.get("draft_n_accepted") or 0) for r in ok)
        out.update(
            draft_n=dn, draft_n_accepted=da, accept_rate=(da / dn) if dn > 0 else None,
            spec_active=bool(ok) and all(int(r.get("draft_n") or 0) > 0 for r in ok),
            n_drafting=sum(1 for r in ok if int(r.get("draft_n") or 0) > 0),
        )
    reported = sorted({str(r.get("speculative_types")) for r in records
                       if r.get("speculative_types") is not None})
    out["speculative_types_reported"] = reported[0] if len(reported) == 1 else (reported or None)
    return out


def parse_membw_json(text: str | bytes) -> dict[str, Any]:
    """``membw --json`` output -> the io_mem.json 'memory' block.

    {read_gbps_by_threads: {'1': .., '2': ..}, copy_gbps, triad_gbps (best over thread counts),
    read_gbps_best, threads, size_mb, reps, host, source: 'measured'}.
    """
    d = json.loads(_decode(text))
    res = d.get("results") or []
    if not res:
        raise ValueError("membw JSON has no results")
    by_t = {str(int(r["threads"])): float(r["read_gbps"]) for r in res}
    return {
        "read_gbps_by_threads": by_t,
        "copy_gbps": max(float(r["copy_gbps"]) for r in res),
        "triad_gbps": max(float(r["triad_gbps"]) for r in res),
        "read_gbps_best": max(by_t.values()),
        "threads": [int(r["threads"]) for r in res],
        "size_mb": d.get("size_mb"), "reps": d.get("reps"), "host": d.get("host"),
        "source": "measured",
    }


# --------------------------------------------------------------------------------------------
# Chat format, visible text, coherence (pigemma.quality owns the graders; fallbacks here)
# --------------------------------------------------------------------------------------------
GEMMA4_USER_PREFIX = "<|turn>user\n"
GEMMA4_USER_SUFFIX = "<turn|>\n<|turn>model\n"
GEMMA4_NO_THINK = "<|channel>thought\n<channel|>"

_THINK_RE = re.compile(r"<\|channel>thought.*?(?:<channel\|>|\Z)", re.DOTALL)
_LEADING_CHANNEL_RE = re.compile(r"^\s*(?:<channel\|>|<\|channel>)")
_END_TOKENS_RE = re.compile(r"(?:<end_of_turn>|<turn\|>|<eos>|\[end of text\])\s*$")


def chat_wrap(user_text: str, *, thinking: bool = False) -> str:
    """Gemma 4 single-user-turn prompt (tokenizer.chat_template of the Gemma 4 GGUFs).

    ``<|turn>user\\n{text}<turn|>\\n<|turn>model\\n`` plus ``<|channel>thought\\n<channel|>``
    when thinking is disabled (the template's enable_thinking=false branch). No BOS: llama.cpp
    adds it when tokenising.
    """
    s = GEMMA4_USER_PREFIX + str(user_text).strip() + GEMMA4_USER_SUFFIX
    return s if thinking else s + GEMMA4_NO_THINK


def _quality() -> Any:
    try:
        from pigemma import quality  # noqa: PLC0415 - lazy: quality imports bench
        return quality
    except Exception:  # noqa: BLE001 - not written yet / broken: use the fallbacks
        return None


def _strip_thinking_fallback(text: str) -> tuple[str, int]:
    s = (text or "").replace("\r\n", "\n")
    thinking = 0
    for m in _THINK_RE.finditer(s):
        inner = m.group(0)[len("<|channel>thought"):]
        if inner.endswith("<channel|>"):
            inner = inner[: -len("<channel|>")]
        thinking += len(inner.strip())
    s = _THINK_RE.sub("", s)
    s = _LEADING_CHANNEL_RE.sub("", s)
    while True:
        s2 = _END_TOKENS_RE.sub("", s.rstrip())
        if s2 == s.rstrip():
            break
        s = s2
    return s.strip(), thinking


def strip_thinking(text: str) -> tuple[str, int]:
    """(visible text, thinking chars): pigemma.quality.strip_thinking when available."""
    q = _quality()
    fn = getattr(q, "strip_thinking", None) if q else None
    if callable(fn):
        try:
            r = fn(text)
            if isinstance(r, tuple) and len(r) >= 2:
                return str(r[0]), int(r[1])
            return str(r), 0
        except Exception as e:  # noqa: BLE001
            log.warning("quality.strip_thinking failed (%s); using the bench fallback", e)
    return _strip_thinking_fallback(text)


def visible_text(text: str) -> str:
    """Visible answer text (thinking channel and end markers removed)."""
    q = _quality()
    fn = getattr(q, "visible_text", None) if q else None
    if callable(fn):
        try:
            return str(fn(text))
        except Exception as e:  # noqa: BLE001
            log.warning("quality.visible_text failed (%s); using strip_thinking", e)
    return strip_thinking(text)[0]


def _coherent_fallback(text: str) -> tuple[bool, str]:
    """Rules of the quality spec: >= 60% letters/digits/space/punctuation, no 4-gram repeated
    >= 6 times, no 6-gram repeated > 3 times (word n-grams)."""
    t = (text or "").strip()
    if not t:
        return False, "empty output"
    good = sum(1 for ch in t if ch.isalnum() or ch.isspace()
               or unicodedata.category(ch).startswith("P"))
    frac = good / len(t)
    if frac < 0.6:
        return False, f"only {frac:.0%} of chars are letters/digits/space/punctuation"
    words = t.split()
    for n, limit, strict in ((4, 6, False), (6, 3, True)):
        if len(words) < n:
            continue
        cnt = collections.Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
        gram, c = cnt.most_common(1)[0]
        if (c > limit) if strict else (c >= limit):
            return False, f"{n}-gram {' '.join(gram)[:60]!r} repeated {c} times"
    return True, "ok"


def coherence(text: str) -> tuple[bool, str, str]:
    """(coherent, reason, impl) - pigemma.quality.is_coherent when available ('quality'),
    else the bench fallback ('bench-fallback'). Applied to the visible text."""
    vis = visible_text(text)
    q = _quality()
    fn = getattr(q, "is_coherent", None) if q else None
    if callable(fn):
        try:
            r = fn(vis)
            if isinstance(r, tuple):
                return bool(r[0]), str(r[1]) if len(r) > 1 else "", "quality"
            return bool(r), "", "quality"
        except Exception as e:  # noqa: BLE001
            log.warning("quality.is_coherent failed (%s); using the bench fallback", e)
    ok, reason = _coherent_fallback(vis)
    return ok, reason, "bench-fallback"


def load_prompts(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Prompt items from JSON: a list of strings, a list of {id?, prompt, max_tokens?, lang?}
    (gen_prompts.json / qa_set.json), or an object with a 'prompts' list."""
    p = Path(os.fspath(path))
    if not p.is_file():
        raise MissingInputError(f"prompts file not found: {p}")
    with open(p, encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, Mapping):
        data = data.get("prompts")
    if not isinstance(data, list) or not data:
        raise BenchError(f"{p}: expected a non-empty JSON list of prompts")
    items: list[dict[str, Any]] = []
    for i, it in enumerate(data):
        if isinstance(it, str):
            items.append({"id": f"p{i:02d}", "prompt": it})
        elif isinstance(it, Mapping) and isinstance(it.get("prompt"), str):
            d = dict(it)
            d.setdefault("id", f"p{i:02d}")
            d["id"] = str(d["id"])
            items.append(d)
        else:
            raise BenchError(f"{p}: item {i} is neither a string nor an object with 'prompt'")
    return items


# --------------------------------------------------------------------------------------------
# Process monitor (memory + I/O sampler)
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Sample:
    """One reading of a child process (bytes unless named otherwise; None = unavailable)."""

    t: float
    rss: int | None = None
    hwm: int | None = None
    private: int | None = None
    private_peak: int | None = None
    mapped: int | None = None
    read_bytes: int | None = None
    rchar: int | None = None
    majflt: int | None = None
    cg_rbytes: int | None = None
    sys_pgmajfault: int | None = None
    sys_pgpgin_kb: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


COUNTER_FIELDS = ("read_bytes", "rchar", "majflt", "cg_rbytes", "sys_pgmajfault", "sys_pgpgin_kb")


def _read_kv_kb(p: Path, keys: Iterable[str]) -> dict[str, int]:
    want = set(keys)
    out: dict[str, int] = {}
    with open(p, encoding="ascii", errors="replace") as f:
        for ln in f:
            k, _, v = ln.partition(":")
            if k in want:
                parts = v.split()
                if parts and parts[0].isdigit():
                    out[k] = int(parts[0]) * 1024
    return out


def _read_proc_io(pid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(f"/proc/{pid}/io", encoding="ascii", errors="replace") as f:
        for ln in f:
            k, _, v = ln.partition(":")
            v = v.strip()
            if v.isdigit():
                out[k.strip()] = int(v)
    return out


def _read_proc_majflt(pid: int) -> int | None:
    with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as f:
        s = f.read()
    rest = s[s.rfind(")") + 2:].split()
    return int(rest[9]) if len(rest) > 9 else None


def _read_vmstat() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/vmstat", encoding="ascii", errors="replace") as f:
            for ln in f:
                k, _, v = ln.partition(" ")
                if k in ("pgmajfault", "pgpgin"):
                    out[k] = int(v)
    except OSError:
        pass
    return out


def _cgroup_io_reader(pid: int) -> Callable[[], int | None] | None:
    """Reader of the process cgroup's read bytes (v2 io.stat rbytes, v1 blkio Read)."""
    cands: list[Path] = []
    try:
        with open(f"/proc/{pid}/cgroup", encoding="ascii", errors="replace") as f:
            for ln in f:
                parts = ln.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    cands.append(Path("/sys/fs/cgroup") / parts[2].lstrip("/") / "io.stat")
    except OSError:
        return None
    cands.append(Path("/sys/fs/cgroup/io.stat"))
    for c in cands:
        if c.is_file():
            def read_v2(c: Path = c) -> int | None:
                try:
                    tot = 0
                    for ln in c.read_text(encoding="ascii", errors="replace").splitlines():
                        for tok in ln.split()[1:]:
                            if tok.startswith("rbytes="):
                                tot += int(tok[7:])
                    return tot
                except (OSError, ValueError):
                    return None
            return read_v2
    v1 = next((Path("/sys/fs/cgroup/blkio") / n for n in (
        "blkio.throttle.io_service_bytes_recursive", "blkio.throttle.io_service_bytes",
        "blkio.io_service_bytes_recursive", "blkio.io_service_bytes")
        if (Path("/sys/fs/cgroup/blkio") / n).is_file()), None)
    if v1 is not None:
        def read_v1() -> int | None:
            try:
                tot = 0
                for ln in v1.read_text(encoding="ascii", errors="replace").splitlines():
                    parts = ln.split()
                    if len(parts) == 3 and parts[1] == "Read":
                        tot += int(parts[2])
                return tot
            except (OSError, ValueError):
                return None
        return read_v1
    return None


class _Win32:
    """Lazily bound kernel32 calls (GetProcessMemoryInfo EX2, GetProcessIoCounters)."""

    _inst: "_Win32 | None" = None

    def __init__(self) -> None:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        class PMCEX2(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t), ("PrivateWorkingSetSize", ctypes.c_size_t),
                ("SharedCommitUsage", ctypes.c_ulonglong),
            ]

        class IOC(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        self.ctypes = ctypes
        self.PMCEX2 = PMCEX2
        self.IOC = IOC
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.OpenProcess = k.OpenProcess
        self.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.OpenProcess.restype = wintypes.HANDLE
        self.CloseHandle = k.CloseHandle
        self.CloseHandle.argtypes = [wintypes.HANDLE]
        self.GetPMI = k.K32GetProcessMemoryInfo
        self.GetPMI.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        self.GetPMI.restype = wintypes.BOOL
        self.GetIO = k.GetProcessIoCounters
        self.GetIO.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        self.GetIO.restype = wintypes.BOOL
        self.GetExitCode = k.GetExitCodeProcess
        self.GetExitCode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.GetExitCode.restype = wintypes.BOOL
        self.DWORD = wintypes.DWORD

    @classmethod
    def get(cls) -> "_Win32 | None":
        if cls._inst is None and paths.is_windows():
            try:
                cls._inst = cls()
            except Exception as e:  # noqa: BLE001
                log.warning("Win32 memory API unavailable (%s); falling back to psutil", e)
                return None
        return cls._inst

    def open(self, pid: int) -> int | None:
        h = self.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
        return h or None

    def alive(self, h: int) -> bool:
        code = self.DWORD()
        if not self.GetExitCode(h, self.ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE

    def read(self, h: int) -> dict[str, int | None] | None:
        if not self.alive(h):
            return None
        c = self.PMCEX2()
        c.cb = self.ctypes.sizeof(self.PMCEX2)
        ex2 = bool(self.GetPMI(h, self.ctypes.byref(c), c.cb))
        if not ex2:  # pre-1809 Windows: PROCESS_MEMORY_COUNTERS_EX (no private working set)
            c.cb = self.ctypes.sizeof(self.PMCEX2) - self.ctypes.sizeof(self.ctypes.c_size_t) - 8
            if not self.GetPMI(h, self.ctypes.byref(c), c.cb):
                return None
        io = self.IOC()
        rb = int(io.ReadTransferCount) if self.GetIO(h, self.ctypes.byref(io)) else None
        return {
            "rss": int(c.WorkingSetSize), "hwm": int(c.PeakWorkingSetSize),
            "private": int(c.PrivateUsage), "private_peak": int(c.PeakPagefileUsage),
            "mapped": (int(c.WorkingSetSize) - int(c.PrivateWorkingSetSize)) if ex2 else None,
            "read_bytes_win": rb,
        }


class ProcessMonitor:
    """Samples a child process every ``interval`` seconds in a daemon thread.

    Linux: procfs (VmRSS/VmHWM/RssAnon/RssFile, /proc/<pid>/io, majflt, /proc/vmstat, the
    cgroup's read bytes when in a container). Windows: GetProcessMemoryInfo + IO counters.
    Elsewhere: psutil when importable. ``snapshot(tag)`` takes an immediate reading and keeps
    the first one per tag in ``marks`` (used for the load / prompt / generation phase split).
    """

    def __init__(self, pid: int, *, interval: float = DEFAULT_SAMPLE_INTERVAL_S,
                 t0: float | None = None, cgroup_io: bool | None = None) -> None:
        self.pid = int(pid)
        self.interval = max(0.02, float(interval))
        self.t0 = t0 if t0 is not None else time.monotonic()
        self.samples: list[Sample] = []
        self.marks: dict[str, Sample] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._win = _Win32.get() if paths.is_windows() else None
        self._wh: int | None = None
        self._ps = None
        self.win_read_bytes: list[tuple[float, int]] = []
        if self._win is not None:
            self._wh = self._win.open(self.pid)
            self.source = "win32" if self._wh else "none"
        elif sys.platform.startswith("linux") and Path(f"/proc/{self.pid}").exists():
            self.source = "procfs"
        else:
            self.source = "none"
        if self.source == "none" and psutil is not None:
            try:
                self._ps = psutil.Process(self.pid)
                self.source = "psutil"
            except Exception:  # noqa: BLE001
                self._ps = None
        use_cg = paths.in_container() if cgroup_io is None else cgroup_io
        self._cg = _cgroup_io_reader(self.pid) if (use_cg and self.source == "procfs") else None
        self.errors: list[str] = []

    # -- reading -------------------------------------------------------------------------
    def read(self) -> Sample | None:
        t = time.monotonic() - self.t0
        try:
            if self.source == "procfs":
                return self._read_procfs(t)
            if self.source == "win32" and self._win is not None and self._wh:
                d = self._win.read(self._wh)
                if d is None:
                    return None
                rbw = d.pop("read_bytes_win", None)
                if rbw is not None:
                    self.win_read_bytes.append((t, int(rbw)))
                return Sample(t=t, **{k: v for k, v in d.items()})
            if self.source == "psutil" and self._ps is not None:
                mi = self._ps.memory_info()
                return Sample(t=t, rss=int(mi.rss), hwm=getattr(mi, "peak_wset", None),
                              private=getattr(mi, "private", None),
                              private_peak=getattr(mi, "peak_pagefile", None))
        except (OSError, ValueError, IndexError) as e:
            if len(self.errors) < 5:
                self.errors.append(f"{type(e).__name__}: {e}")
            return None
        except Exception as e:  # noqa: BLE001 - psutil.NoSuchProcess etc.
            if len(self.errors) < 5:
                self.errors.append(f"{type(e).__name__}: {e}")
            return None
        return None

    def _read_procfs(self, t: float) -> Sample | None:
        st = _read_kv_kb(Path(f"/proc/{self.pid}/status"),
                         ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"))
        if not st:  # zombie or gone
            return None
        s = Sample(t=t, rss=st.get("VmRSS"), hwm=st.get("VmHWM"), private=st.get("RssAnon"),
                   mapped=st.get("RssFile"))
        try:
            io = _read_proc_io(self.pid)
            s.read_bytes, s.rchar = io.get("read_bytes"), io.get("rchar")
        except OSError:
            pass
        with contextlib.suppress(OSError, ValueError, IndexError):
            s.majflt = _read_proc_majflt(self.pid)
        vm = _read_vmstat()
        s.sys_pgmajfault, s.sys_pgpgin_kb = vm.get("pgmajfault"), vm.get("pgpgin")
        if self._cg is not None:
            s.cg_rbytes = self._cg()
        return s

    def snapshot(self, tag: str | None = None) -> Sample | None:
        s = self.read()
        if s is None:
            return None
        with self._lock:
            if len(self.samples) < MAX_SAMPLES:
                self.samples.append(s)
            if tag and tag not in self.marks:
                self.marks[tag] = s
        return s

    # -- thread --------------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            self.snapshot()
            self._stop.wait(self.interval)

    def start(self) -> "ProcessMonitor":
        if self.source != "none":
            self._thread = threading.Thread(target=self._run, name=f"pigemma-mon-{self.pid}",
                                            daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._win is not None and self._wh:
            with contextlib.suppress(Exception):
                self._win.CloseHandle(self._wh)
            self._wh = None

    # -- results -------------------------------------------------------------------------
    def _max(self, *fields: str) -> int | None:
        vals = [getattr(s, f) for s in self.samples for f in fields if getattr(s, f) is not None]
        return max(vals) if vals else None

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source": self.source,
                "interval_s": self.interval,
                "n_samples": len(self.samples),
                "rss_peak_bytes": self._max("rss", "hwm"),
                "private_peak_bytes": self._max("private", "private_peak"),
                "mapped_peak_bytes": self._max("mapped"),
                "errors": list(self.errors),
            }

    def counter_at(self, field: str, mark: str | None) -> tuple[float, int] | None:
        """(t, value) of a counter at a mark (last non-None reading at or before it) or at
        the last sample when mark is None."""
        with self._lock:
            if mark is not None:
                m = self.marks.get(mark)
                if m is None:
                    return None
                if getattr(m, field) is not None:
                    return m.t, int(getattr(m, field))
                cands = [s for s in self.samples if s.t <= m.t and getattr(s, field) is not None]
            else:
                cands = [s for s in self.samples if getattr(s, field) is not None]
            if not cands:
                return None
            s = cands[-1]
            return s.t, int(getattr(s, field))

    def phase(self, start: str | None, end: str | None) -> dict[str, Any]:
        """Counter deltas between two marks (start None = process start, counters from 0 for
        the per-process counters; end None = last sample)."""
        out: dict[str, Any] = {}
        t_a = 0.0
        if start is not None and start in self.marks:
            t_a = self.marks[start].t
        t_b = None
        for f in COUNTER_FIELDS:
            b = self.counter_at(f, end)
            if b is None:
                out[f] = None
                continue
            t_b = b[0] if t_b is None else t_b
            if start is None:
                if f.startswith("sys_") or f == "cg_rbytes":
                    first = next((s for s in self.samples if getattr(s, f) is not None), None)
                    out[f] = (b[1] - int(getattr(first, f))) if first is not None else None
                else:
                    out[f] = b[1]
                continue
            a = self.counter_at(f, start)
            out[f] = (b[1] - a[1]) if a is not None else None
        if end is not None and end in self.marks:
            t_b = self.marks[end].t
        elif self.samples:
            t_b = self.samples[-1].t
        out["wall_s"] = round(t_b - t_a, 3) if t_b is not None else None
        return out


# --------------------------------------------------------------------------------------------
# Child process helpers
# --------------------------------------------------------------------------------------------
def _popen_kwargs(nice: int | None) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if nice and not paths.is_windows():
        n = int(nice)

        def _pre() -> None:  # runs in the child before exec
            with contextlib.suppress(OSError):
                os.nice(n)
        kw["preexec_fn"] = _pre
    elif nice:
        log.warning("--nice is ignored on Windows")
    return kw


def _signal_child(proc: subprocess.Popen[Any], sig: int) -> None:
    """Signal our own child without Popen.poll() (which could reap it and lose its rusage)."""
    if proc.returncode is not None:
        return
    try:
        if paths.is_windows():
            proc.kill() if sig == getattr(signal, "SIGKILL", 9) else proc.terminate()
        else:
            os.kill(proc.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _wait_child(proc: subprocess.Popen[Any], timeout: float | None
                ) -> tuple[int | None, dict[str, Any] | None, bool]:
    """Wait for our child; (returncode, rusage or None, timed_out). POSIX uses wait4 so the
    child's own rusage (maxrss, majflt, inblock) is available."""
    timed_out = False
    if hasattr(os, "wait4") and not paths.is_windows():
        deadline = (time.monotonic() + timeout) if timeout else None
        kill_at: float | None = None
        while True:
            try:
                pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
            except ChildProcessError:
                return proc.poll(), None, timed_out
            if pid:
                rc = os.waitstatus_to_exitcode(status)
                proc.returncode = rc
                return rc, {
                    "maxrss_bytes": int(ru.ru_maxrss) * 1024, "majflt": int(ru.ru_majflt),
                    "minflt": int(ru.ru_minflt), "inblock_bytes": int(ru.ru_inblock) * 512,
                    "utime_s": round(ru.ru_utime, 3), "stime_s": round(ru.ru_stime, 3),
                }, timed_out
            now = time.monotonic()
            if deadline is not None and now > deadline:
                timed_out = True
                deadline = None
                _signal_child(proc, signal.SIGTERM)
                kill_at = now + STOP_GRACE_S
            if kill_at is not None and now > kill_at:
                _signal_child(proc, signal.SIGKILL)
                kill_at = None
            time.sleep(0.05)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.terminate()
        try:
            rc = proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
    return rc, None, timed_out


class _Reader(threading.Thread):
    """Drains a child pipe; optional sink file and per-chunk / per-line callbacks."""

    def __init__(self, stream: Any, *, lines: bool, sink: Path | None = None,
                 on_data: Callable[[bytes], None] | None = None) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.lines = lines
        self.on_data = on_data
        self.chunks: list[bytes] = []
        if sink:
            sink.parent.mkdir(parents=True, exist_ok=True)
        self._sink = open(sink, "wb") if sink else None  # noqa: SIM115 - closed in run()

    def run(self) -> None:
        try:
            if self.lines:
                for line in iter(self.stream.readline, b""):
                    self._take(line)
            else:
                read = getattr(self.stream, "read1", None) or self.stream.read
                while True:
                    chunk = read(65536)
                    if not chunk:
                        break
                    self._take(chunk)
        except (OSError, ValueError):
            pass
        finally:
            if self._sink:
                self._sink.close()

    def _take(self, b: bytes) -> None:
        self.chunks.append(b)
        if self._sink:
            self._sink.write(b)
            self._sink.flush()
        if self.on_data:
            try:
                self.on_data(b)
            except Exception as e:  # noqa: BLE001 - never kill the drain
                log.debug("reader callback failed: %s", e)

    def text(self) -> str:
        return _decode(b"".join(self.chunks)).replace("\r\n", "\n")


# --------------------------------------------------------------------------------------------
# llama-bench
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class BenchRun:
    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    rows: list[dict[str, Any]]
    wall_s: float
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and bool(self.rows)


def llama_bench_argv(cmd: Sequence[str], model: str | os.PathLike[str], *, threads: Sequence[int],
                     pp: int, tg: int, reps: int, flags: Sequence[str] = ()) -> list[str]:
    """``llama-bench -m F -t <list> -p PP -n TG -r R -o json [flags]``."""
    return list(cmd) + ["-m", os.fspath(model), "-t", ",".join(str(t) for t in threads),
                        "-p", str(int(pp)), "-n", str(int(tg)), "-r", str(int(reps)),
                        "-o", "json"] + [str(f) for f in flags]


def run_llama_bench(model: str | os.PathLike[str], *, threads: Sequence[int], pp: int, tg: int,
                    reps: int, flags: Sequence[str] = (), binary: Any = None,
                    timeout: float | None = None, nice: int | None = None,
                    stdout_path: Path | None = None, stderr_path: Path | None = None) -> BenchRun:
    """Run one llama-bench process and parse its JSON rows (partial rows kept on failure)."""
    cmd = _tool_cmd("llama-bench", binary)
    argv = llama_bench_argv(cmd, model, threads=threads, pp=pp, tg=tg, reps=reps, flags=flags)
    log.info("run: %s", _cmd_str(argv))
    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, **_popen_kwargs(nice))
    except OSError as e:
        return BenchRun(argv, None, "", "", [], 0.0, error=f"{type(e).__name__}: {e}")
    try:
        out_b, err_b = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.terminate()
        try:
            out_b, err_b = proc.communicate(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            out_b, err_b = proc.communicate()
    except BaseException:
        proc.kill()
        proc.communicate()
        raise
    wall = time.monotonic() - t0
    out, err = _decode(out_b), _decode(err_b).replace("\r\n", "\n")
    if stdout_path:
        _write_text(stdout_path, out)
    if stderr_path:
        _write_text(stderr_path, err)
    rows = parse_bench_json(out)
    return BenchRun(argv, proc.returncode, out, err, rows, round(wall, 3), timed_out=timed_out)


def bench_rows_to_fields(rows: Sequence[Mapping[str, Any]], *, experiment: str,
                         model: Mapping[str, Any] | None, variant: str | None,
                         flags: Sequence[str], extra: Mapping[str, Any] | None = None,
                         notes: Sequence[str] = ()) -> list[dict[str, Any]]:
    """One results.write_result kwargs dict per llama-bench row (kind measured)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        r = r if "kind" in r else _bench_row_norm(r)
        rate = {"tok_s": r.get("avg_ts"), "sd": r.get("stddev_ts"),
                "n": r.get("n_samples")}
        ex: dict[str, Any] = {
            "source": "llama-bench", "test": r.get("test"), "n_prompt": r.get("n_prompt"),
            "n_gen": r.get("n_gen"), "n_depth": r.get("n_depth"),
        }
        for k in ("n_batch", "n_ubatch", "avg_ns", "stddev_ns", "samples_ts", "cpu_info",
                  "backends", "model_filename", "model_type", "model_size", "model_n_params",
                  "type_k", "type_v", "flash_attn", "repack", "load_mode", "lazy_mode",
                  "tensor_buft_overrides", "n_cpu_moe", "n_gpu_layers", "poll", "test_time"):
            if k in r:
                ex[k] = r[k]
        if extra:
            ex.update(extra)
        nts = list(notes)
        if r.get("avg_ts") is None:
            nts.append("llama-bench avg_ts was not a finite number")
        bn = _build_note(r.get("build_number"), r.get("build_commit"))
        if bn:
            nts.append(bn)
        f: dict[str, Any] = {
            "experiment": experiment, "kind": "measured",
            "llama_build": r.get("build_number"), "llama_commit": r.get("build_commit"),
            "model": dict(model) if model is not None else None,
            "variant": variant or infer_variant(r), "flags": [str(x) for x in flags],
            "threads": r.get("n_threads"),
            "n_ctx": (r.get("n_prompt") or 0) + (r.get("n_gen") or 0) + (r.get("n_depth") or 0),
            "extra": ex,
        }
        if r.get("kind") == "pp":
            f["pp"] = rate
        elif r.get("kind") == "tg":
            f["tg"] = rate
        else:
            ex["pg"] = rate
        if nts:
            f["notes"] = nts
        out.append(f)
    return out


def _bench_failure_fields(expected: Iterable[tuple[int, str]], *, experiment: str,
                          model: Mapping[str, Any] | None, variant: str, flags: Sequence[str],
                          run: BenchRun, pp: int, tg: int, build: tuple[Any, Any],
                          extra: Mapping[str, Any]) -> list[dict[str, Any]]:
    why = "timed out" if run.timed_out else (run.error or f"exit code {run.returncode}")
    note = failure_note(run.stderr, extra=f"llama-bench {why}; no row for this test")
    out = []
    for t, kind in expected:
        ex = dict(extra)
        ex.update(source="llama-bench", test=f"{kind}{pp if kind == 'pp' else tg}",
                  returncode=run.returncode, timed_out=run.timed_out)
        out.append({
            "experiment": experiment, "kind": "measured", "llama_build": build[0],
            "llama_commit": build[1], "model": dict(model) if model is not None else None,
            "variant": variant, "flags": [str(x) for x in flags], "threads": t,
            "n_ctx": (pp if kind == "pp" else 0) + (tg if kind == "tg" else 0),
            "notes": [note], "extra": ex,
        })
    return out


# --------------------------------------------------------------------------------------------
# llama-completion
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class CompletionRun:
    argv: list[str]
    returncode: int | None
    stdout_text: str
    stderr_text: str
    wall_s: float
    parsed: dict[str, Any]
    mem: dict[str, Any]
    io: dict[str, Any]
    marks: dict[str, float]
    timed_out: bool = False
    error: str | None = None
    rusage: dict[str, Any] | None = None
    prompt_sha256: str | None = None
    prompt_chars: int = 0
    win_read_bytes: int | None = None

    @property
    def ok(self) -> bool:
        return (self.returncode == 0 and not self.timed_out and self.error is None
                and self.parsed.get("tg_tok_s") is not None)


def completion_argv(cmd: Sequence[str], model: str | os.PathLike[str], prompt_file: str,
                    *, n_predict: int, threads: int, n_ctx: int, flags: Sequence[str] = ()
                    ) -> list[str]:
    """``llama-completion -m F -f P -n N --temp 0 -t T -c C -no-cnv --no-display-prompt -lv 4
    [flags]``."""
    return list(cmd) + ["-m", os.fspath(model), "-f", prompt_file, "-n", str(int(n_predict)),
                        "--temp", "0", "-t", str(int(threads)), "-c", str(int(n_ctx)),
                        "-no-cnv", "--no-display-prompt", "-lv", "4"] + [str(f) for f in flags]


def run_completion(model: str | os.PathLike[str], prompt_text: str, *, n_predict: int = 128,
                   threads: int = 6, n_ctx: int = DEFAULT_N_CTX, flags: Sequence[str] = (),
                   chat: bool = False, binary: Any = None, timeout: float | None = None,
                   sample_interval: float = DEFAULT_SAMPLE_INTERVAL_S, nice: int | None = None,
                   stdout_path: Path | None = None, stderr_path: Path | None = None
                   ) -> CompletionRun:
    """Run llama-completion once with the memory / I/O sampler attached.

    Phase marks (seconds since start): prompt_start ('generate: n_ctx' on stderr), gen_start
    (first generated stdout bytes), gen_end (first 'common_perf_print' line). ``io`` holds
    counter deltas for load / prompt / gen / total and the per-token generation figures.
    """
    cmd = _tool_cmd("llama-completion", binary)
    text = chat_wrap(prompt_text) if chat else prompt_text
    data = text.encode("utf-8", errors="surrogateescape")
    tmpdir = Path(tempfile.mkdtemp(prefix="pigemma-bench-"))
    pfile = tmpdir / "prompt.txt"
    pfile.write_bytes(data)
    argv = completion_argv(cmd, model, str(pfile), n_predict=n_predict, threads=threads,
                           n_ctx=n_ctx, flags=flags)
    log.info("run: %s", _cmd_str(argv))
    t0 = time.monotonic()
    marks: dict[str, float] = {}
    try:
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, **_popen_kwargs(nice))
        except OSError as e:
            return CompletionRun(argv, None, "", "", 0.0, parse_completion_stderr(""), {}, {},
                                 {}, error=f"{type(e).__name__}: {e}",
                                 prompt_sha256=_sha256_bytes(data), prompt_chars=len(text))
        mon = ProcessMonitor(proc.pid, interval=sample_interval, t0=t0).start()
        mon.snapshot("start")

        def on_err(b: bytes) -> None:
            s = b.decode("utf-8", errors="replace")
            if "prompt_start" not in marks and "generate: n_ctx" in s:
                marks["prompt_start"] = time.monotonic() - t0
                mon.snapshot("prompt_start")
            if "gen_end" not in marks and ("common_perf_print" in s or "llama_perf_context_print" in s):
                marks["gen_end"] = time.monotonic() - t0
                mon.snapshot("gen_end")

        def on_out(b: bytes) -> None:
            if "gen_start" not in marks and "prompt_start" in marks and b.strip():
                marks["gen_start"] = time.monotonic() - t0
                mon.snapshot("gen_start")

        r_err = _Reader(proc.stderr, lines=True, sink=stderr_path, on_data=on_err)
        r_out = _Reader(proc.stdout, lines=False, sink=stdout_path, on_data=on_out)
        r_err.start()
        r_out.start()
        try:
            rc, ru, timed_out = _wait_child(proc, timeout)
        except BaseException:
            _signal_child(proc, signal.SIGTERM)
            with contextlib.suppress(Exception):
                proc.wait(timeout=STOP_GRACE_S)
            _signal_child(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
            mon.stop()
            raise
        wall = time.monotonic() - t0
        r_err.join(timeout=10)
        r_out.join(timeout=10)
        mon.stop()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    stderr_text = r_err.text()
    stdout_text = r_out.text()
    parsed = parse_completion_stderr(stderr_text)
    mem = mon.summary()
    if ru and ru.get("maxrss_bytes"):
        mem["rss_peak_bytes"] = max(mem.get("rss_peak_bytes") or 0, int(ru["maxrss_bytes"]))
    io: dict[str, Any] = {
        "load": mon.phase(None, "prompt_start") if "prompt_start" in mon.marks else None,
        "prompt": mon.phase("prompt_start", "gen_start") if "gen_start" in mon.marks else None,
        "gen": mon.phase("gen_start", "gen_end") if ("gen_start" in mon.marks and "gen_end" in mon.marks) else None,
        "total": mon.phase(None, None),
    }
    n_eval = parsed.get("n_eval")
    gen = io.get("gen") or {}

    def per_tok(key: str) -> float | None:
        v = gen.get(key)
        return (v / n_eval) if (v is not None and n_eval) else None

    io["read_bytes_per_token"] = per_tok("read_bytes")
    io["majflt_per_token"] = per_tok("majflt")
    io["cg_rbytes_per_token"] = per_tok("cg_rbytes")
    io["rchar_per_token"] = per_tok("rchar")
    io["sys_pgmajfault_per_token"] = per_tok("sys_pgmajfault")
    win_rb = mon.win_read_bytes[-1][1] if mon.win_read_bytes else None
    return CompletionRun(argv=argv, returncode=rc, stdout_text=stdout_text,
                         stderr_text=stderr_text, wall_s=round(wall, 3), parsed=parsed,
                         mem=mem, io=io, marks={k: round(v, 3) for k, v in marks.items()},
                         timed_out=timed_out, rusage=ru, prompt_sha256=_sha256_bytes(data),
                         prompt_chars=len(text), win_read_bytes=win_rb)


def _mb(b: Any) -> float | None:
    v = _num(b)
    return round(v / MB, 1) if v is not None else None


def completion_fields(run: CompletionRun, *, experiment: str, model: Mapping[str, Any] | None,
                      variant: str, flags: Sequence[str], threads: int, n_ctx: int,
                      n_predict: int, chat: bool, prompt_path: str | None,
                      build: tuple[Any, Any] = (None, None), raw: Mapping[str, Any] | None = None
                      ) -> dict[str, Any]:
    """results.write_result kwargs for one completion run (failed runs: notes, no tok/s)."""
    p = run.parsed
    bn = p.get("build_number") if p.get("build_number") is not None else build[0]
    bc = p.get("build_commit") if p.get("build_commit") is not None else build[1]
    notes: list[str] = []
    out_text = run.stdout_text
    vis, thinking = strip_thinking(out_text)
    coh, reason, impl = coherence(out_text) if out_text.strip() else (None, "no output", "n/a")
    in_ct = paths.in_container()
    extra: dict[str, Any] = {
        "source": "llama-completion", "argv": [str(a) for a in run.argv],
        "prompt_file": prompt_path, "prompt_sha256": run.prompt_sha256,
        "prompt_chars": run.prompt_chars, "chat": chat, "chat_format": "gemma4" if chat else None,
        "n_predict": n_predict, "n_prompt_tokens": p.get("n_prompt"),
        "n_eval_runs": p.get("n_eval"), "prompt_ms": p.get("prompt_ms"),
        "eval_ms": p.get("eval_ms"), "total_ms": p.get("total_ms"),
        "n_ctx_requested": n_ctx, "n_threads_reported": p.get("n_threads"),
        "wall_s": run.wall_s, "returncode": run.returncode, "timed_out": run.timed_out,
        "marks_s": run.marks, "io": run.io, "rusage": run.rusage,
        "mem": run.mem, "buffers_other": p.get("buffers_other"),
        "visible_chars": len(vis), "thinking_chars": thinking,
        "coherence_reason": reason, "coherence_impl": impl,
        "win_read_transfer_bytes": run.win_read_bytes,
    }
    if raw:
        extra.update(raw)
    f: dict[str, Any] = {
        "experiment": experiment, "kind": "measured", "llama_build": bn, "llama_commit": bc,
        "model": dict(model) if model is not None else None, "variant": variant,
        "flags": [str(x) for x in flags], "threads": threads,
        "n_ctx": p.get("n_ctx") if p.get("n_ctx") is not None else n_ctx,
        "load_s": p.get("load_s"),
        "rss_peak_mb": _mb(run.mem.get("rss_peak_bytes")),
        "private_peak_mb": _mb(run.mem.get("private_peak_bytes")),
        "mapped_mb": _mb(run.mem.get("mapped_peak_bytes")),
        "buffer_split": p.get("buffer_split") or None,
        "read_bytes_per_token": run.io.get("read_bytes_per_token"),
        "majflt_per_token": run.io.get("majflt_per_token"),
        "io_rbytes_per_token": run.io.get("cg_rbytes_per_token") if in_ct else None,
        "coherent": coh,
        "output_sample": out_text,
        "extra": extra,
    }
    if run.ok:
        f["pp"] = {"tok_s": p.get("pp_tok_s"), "sd": None, "n": 1}
        f["tg"] = {"tok_s": p.get("tg_tok_s"), "sd": None, "n": 1}
    else:
        why = ("timed out" if run.timed_out else run.error or
               (f"exit code {run.returncode}" if run.returncode != 0 else
                "no 'eval time' line in stderr"))
        notes.append(failure_note(run.stderr_text, extra=f"llama-completion {why}"))
    if not p.get("buffer_split"):
        notes.append("no 'load_tensors: ... model buffer size' lines in stderr (buffer_split n/a)")
    if run.io.get("read_bytes_per_token") is None and not paths.is_windows():
        notes.append("generation-phase /proc read_bytes unavailable (no gen markers or /proc/<pid>/io unreadable)")
    if paths.is_windows():
        notes.append("Windows: read_bytes/majflt per token n/a (no per-process paging-I/O counter)")
    note_b = _build_note(bn, bc)
    if note_b:
        notes.append(note_b)
    if os.environ.get("PIGEMMA_EMULATED", "").strip() == "1":
        notes.append("emulated host (qemu): timings are a functional proof, not a Pi measurement")
    if notes:
        f["notes"] = notes
    return f


# --------------------------------------------------------------------------------------------
# llama-server
# --------------------------------------------------------------------------------------------
class ServerClient:
    """Minimal HTTP client for llama-server (/health, /completion) using http.client."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = int(port)

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None,
                 timeout: float = 10.0) -> tuple[int, Any]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
            headers = {"Content-Type": "application/json"} if data is not None else {}
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            try:
                obj = json.loads(raw.decode("utf-8", errors="replace")) if raw else None
            except ValueError:
                obj = {"_raw": raw[:500].decode("utf-8", errors="replace")}
            return resp.status, obj
        finally:
            conn.close()

    def health(self, timeout: float = 5.0) -> tuple[bool, str]:
        """(ready, detail): ready iff HTTP 200 with {"status": "ok"}."""
        try:
            st, obj = self._request("GET", "/health", timeout=timeout)
        except (OSError, http.client.HTTPException) as e:
            return False, f"{type(e).__name__}"
        if st == 200 and isinstance(obj, Mapping) and obj.get("status") == "ok":
            return True, "ok"
        return False, f"HTTP {st}"

    def answering(self, timeout: float = 1.0) -> bool:
        """True when something accepts TCP connections on host:port."""
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout):
                return True
        except OSError:
            return False

    def complete(self, prompt: str, n_predict: int, *, temperature: float = 0.0,
                 cache_prompt: bool = False, timeout: float | None = None,
                 **extra: Any) -> dict[str, Any]:
        """POST /completion; returns the response JSON or raises ServerRequestError."""
        body = {"prompt": prompt, "n_predict": int(n_predict), "temperature": temperature,
                "cache_prompt": cache_prompt, **extra}
        to = timeout if timeout else 300.0 + 30.0 * max(1, int(n_predict))
        try:
            st, obj = self._request("POST", "/completion", body, timeout=to)
        except (OSError, http.client.HTTPException) as e:
            raise ServerRequestError(f"/completion failed: {type(e).__name__}: {e}") from e
        if st != 200 or not isinstance(obj, Mapping):
            raise ServerRequestError(f"/completion HTTP {st}: {str(obj)[:300]}")
        return dict(obj)


class LlamaServer:
    """Starts and owns ONE llama-server child; never touches any other process.

    ``llama-server -m F -t T -c C -np 1 --host H --port P [-md DRAFT] [--spec-type T]
    [--spec-draft-n-max N] [flags]``. ``start()`` refuses (RefusedError) when anything already
    answers on the port, then polls /health until {"status": "ok"} (load_time_s) or raises
    ServerStartError (child exited / health timeout). ``stop()`` terminates, kills after 10 s.
    """

    def __init__(self, model: str | os.PathLike[str], *, threads: int = 6,
                 n_ctx: int = DEFAULT_N_CTX, port: int = DEFAULT_PORT, host: str = "127.0.0.1",
                 draft: str | os.PathLike[str] | None = None, spec_type: str | None = None,
                 draft_n_max: int | None = None, flags: Sequence[str] = (), binary: Any = None,
                 log_path: Path | None = None, health_timeout: float = HEALTH_TIMEOUT_S,
                 sample_interval: float = DEFAULT_SAMPLE_INTERVAL_S, nice: int | None = None) -> None:
        self.cmd = _tool_cmd("llama-server", binary)
        self.model = os.fspath(model)
        self.threads, self.n_ctx, self.port, self.host = int(threads), int(n_ctx), int(port), host
        self.draft = os.fspath(draft) if draft is not None else None
        self.spec_type, self.draft_n_max = spec_type, draft_n_max
        self.flags = [str(f) for f in flags]
        self.log_path = log_path
        self.health_timeout = float(health_timeout)
        self.sample_interval = sample_interval
        self.nice = nice
        self.client = ServerClient(host, self.port)
        self.proc: subprocess.Popen[Any] | None = None
        self.monitor: ProcessMonitor | None = None
        self.load_time_s: float | None = None
        self.health_wait_s: float | None = None
        self.load_time_source: str | None = None
        self.returncode: int | None = None
        self._log_f: Any = None
        self._tmp_log: Path | None = None

    @property
    def argv(self) -> list[str]:
        a = list(self.cmd) + ["-m", self.model, "-t", str(self.threads), "-c", str(self.n_ctx),
                              "-np", "1", "--host", self.host, "--port", str(self.port)]
        if self.draft:
            a += ["-md", self.draft]
        if self.spec_type:
            a += ["--spec-type", self.spec_type]
        if self.draft_n_max is not None:
            a += ["--spec-draft-n-max", str(int(self.draft_n_max))]
        return a + self.flags

    def log_text(self) -> str:
        p = self.log_path or self._tmp_log
        if p and p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
        return ""

    def start(self) -> "LlamaServer":
        if self.client.answering(timeout=PORT_PROBE_TIMEOUT_S):
            raise RefusedError(f"something already answers on {self.host}:{self.port}; "
                               "refusing to start a second server (pick another --port)")
        if self.log_path is None:
            fd, name = tempfile.mkstemp(prefix="pigemma-server-", suffix=".log")
            os.close(fd)
            self._tmp_log = Path(name)
        lp = self.log_path or self._tmp_log
        assert lp is not None
        lp.parent.mkdir(parents=True, exist_ok=True)
        self._log_f = open(lp, "wb")  # noqa: SIM115 - closed in stop()
        log.info("start: %s", _cmd_str(self.argv))
        t0 = time.monotonic()
        try:
            self.proc = subprocess.Popen(self.argv, stdin=subprocess.DEVNULL, stdout=self._log_f,
                                         stderr=subprocess.STDOUT, **_popen_kwargs(self.nice))
        except OSError as e:
            self._log_f.close()
            raise ServerStartError(f"cannot start llama-server: {type(e).__name__}: {e}") from e
        self.monitor = ProcessMonitor(self.proc.pid, interval=self.sample_interval, t0=t0).start()
        deadline = t0 + self.health_timeout
        detail = "no answer"
        while True:
            if self.proc.poll() is not None:
                self.returncode = self.proc.returncode
                raise ServerStartError(failure_note(self.log_text(), extra=(
                    f"llama-server exited with code {self.proc.returncode} before becoming healthy")))
            # a short TCP probe first: a refused connect costs ~2 s on Windows (SYN retries) and
            # llama-server only listens once the model is loaded
            if self.client.answering(timeout=HEALTH_POLL_S):
                ok, detail = self.client.health()
            else:
                ok, detail = False, "not listening"
            if ok:
                self.health_wait_s = round(time.monotonic() - t0, 3)
                # the server's own timestamp of 'model loaded' is exact; the poll is +-0.5 s
                logged = log_time_s(self.log_text(), "llama_server: model loaded")
                if logged is not None and logged <= self.health_wait_s + 1.0:
                    self.load_time_s, self.load_time_source = logged, "server log 'model loaded'"
                else:
                    self.load_time_s, self.load_time_source = self.health_wait_s, "health poll"
                log.info("server healthy after %.1f s (load %.1f s from %s; pid %d)",
                         self.health_wait_s, self.load_time_s, self.load_time_source,
                         self.proc.pid)
                return self
            if time.monotonic() > deadline:
                raise ServerStartError(failure_note(self.log_text(), extra=(
                    f"llama-server not healthy after {self.health_timeout:.0f} s (last: {detail})")))
            time.sleep(HEALTH_POLL_S)

    def complete(self, prompt: str, n_predict: int, **kw: Any) -> dict[str, Any]:
        if self.proc is not None and self.proc.poll() is not None:
            raise ServerRequestError(f"llama-server exited (code {self.proc.returncode})")
        return self.client.complete(prompt, n_predict, **kw)

    def stop(self) -> int | None:
        p = self.proc
        if p is not None and p.returncode is None:
            p.terminate()
            try:
                p.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                log.warning("llama-server pid %d ignored terminate for %.0f s; killing it",
                            p.pid, STOP_GRACE_S)
                p.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    p.wait(timeout=STOP_GRACE_S)
        if p is not None:
            self.returncode = p.returncode
        if self.monitor is not None:
            self.monitor.stop()
        if self._log_f is not None:
            with contextlib.suppress(OSError):
                self._log_f.close()
            self._log_f = None
        return self.returncode

    def cleanup(self) -> None:
        if self._tmp_log is not None:
            with contextlib.suppress(OSError):
                self._tmp_log.unlink()

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def run_prompt_set(server: Any, items: Sequence[Mapping[str, Any]], *,
                   n_predict: int | None = None, chat: bool = True,
                   warmup_tokens: int = WARMUP_TOKENS, request_timeout: float | None = None,
                   default_n_predict: int = DEFAULT_SERVER_N_PREDICT) -> list[dict[str, Any]]:
    """One warm-up request (``warmup_tokens``, not recorded) then one greedy /completion per
    item (``cache_prompt=false``, temperature 0). ``n_predict`` overrides each item's
    max_tokens (default ``default_n_predict``). ``server`` is a LlamaServer or ServerClient.

    Record keys: id, lang, n_predict, prompt_chars, wall_s, content, error, http_ok + every
    parse_timings key.
    """
    recs: list[dict[str, Any]] = []
    if not items:
        return recs
    if warmup_tokens and warmup_tokens > 0:
        p0 = str(items[0]["prompt"])
        try:
            server.complete(chat_wrap(p0) if chat else p0, int(warmup_tokens),
                            timeout=request_timeout)
        except ServerRequestError as e:
            log.warning("warm-up request failed: %s", e)
    for it in items:
        n = int(n_predict if n_predict is not None else it.get("max_tokens") or default_n_predict)
        prompt = chat_wrap(str(it["prompt"])) if chat else str(it["prompt"])
        rec: dict[str, Any] = {"id": str(it.get("id")), "lang": it.get("lang"), "n_predict": n,
                               "prompt_chars": len(prompt)}
        t0 = time.monotonic()
        try:
            resp = server.complete(prompt, n, timeout=request_timeout)
            rec.update(parse_timings(resp))
            rec["content"] = str(resp.get("content") or "")
            rec["error"] = None
        except ServerRequestError as e:
            rec.update(parse_timings({}))
            rec["content"] = None
            rec["error"] = str(e)[:500]
            log.warning("request %s failed: %s", rec["id"], e)
        rec["wall_s"] = round(time.monotonic() - t0, 3)
        tg = rec.get("predicted_per_second")
        log.info("request %s: %s tokens, %s tok/s, draft %s/%s", rec["id"], rec.get("predicted_n"),
                 f"{tg:.2f}" if tg else "n/a", rec.get("draft_n_accepted"), rec.get("draft_n"))
        recs.append(rec)
    return recs


def compare_to_plain(records: Sequence[Mapping[str, Any]], plain: Mapping[str, str] | None
                     ) -> dict[str, Any]:
    """Per-prompt visible-text identity and similarity against the plain run's contents."""
    if not plain:
        return {"identical_to_plain": None, "identical_by_prompt": None,
                "similarity_by_prompt": None, "n_compared": 0, "n_similar_ge_095": None}
    ident: dict[str, bool] = {}
    sim: dict[str, float] = {}
    for r in records:
        rid = str(r.get("id"))
        c = r.get("content")
        if r.get("error") or c is None or rid not in plain or plain[rid] is None:
            continue
        a, b = visible_text(str(plain[rid])), visible_text(str(c))
        ident[rid] = a == b
        sim[rid] = round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 4)
    n = len(ident)
    return {
        "identical_to_plain": (sum(ident.values()) / n) if n else None,
        "identical_by_prompt": ident or None,
        "similarity_by_prompt": sim or None,
        "n_compared": n,
        "n_similar_ge_095": sum(1 for v in sim.values() if v >= SIMILAR_THRESHOLD) if n else None,
    }


def server_fields(records: Sequence[Mapping[str, Any]], *, experiment: str,
                  model: Mapping[str, Any] | None, variant: str, flags: Sequence[str],
                  threads: int, n_ctx: int, server_variant: str, spec_type: str | None,
                  draft: Mapping[str, Any] | None, load_time_s: float | None,
                  mem: Mapping[str, Any] | None, plain: Mapping[str, str] | None,
                  plain_source: str | None, build: tuple[Any, Any] = (None, None),
                  failure: str | None = None, extra: Mapping[str, Any] | None = None
                  ) -> dict[str, Any]:
    """results.write_result kwargs for one server-spec variant."""
    is_plain = server_variant == "plain"
    spec_configured = not is_plain
    agg = aggregate_timings(records, spec_configured=spec_configured)
    reported = agg.get("speculative_types_reported")
    real_types = _spec_types(reported if isinstance(reported, str) else
                             ",".join(reported or []) if isinstance(reported, list) else None)
    if agg["n_ok"] == 0:
        ste = None  # never ran: nothing is known about the effective type
    elif spec_type:
        ste = f"explicit:{spec_type}"
    elif draft and real_types:
        ste = "inferred"
    else:
        ste = "none"
    notes: list[str] = []
    ok_recs = [r for r in records if not r.get("error") and r.get("content") is not None]
    cmp = compare_to_plain(records, plain) if not is_plain else compare_to_plain([], None)
    coh_list = [coherence(str(r["content"])) for r in ok_recs]
    coherent = all(c[0] for c in coh_list) if coh_list else None
    req_out: list[dict[str, Any]] = []
    for r in records:
        rr = {k: r.get(k) for k in (
            "id", "lang", "n_predict", "prompt_n", "prompt_ms", "prompt_per_second",
            "predicted_n", "predicted_ms", "predicted_per_second", "draft_n",
            "draft_n_accepted", "accept_rate", "speculative_types", "stop_type", "wall_s",
            "error", "content")}
        if r.get("content") is not None:
            v, th = strip_thinking(str(r["content"]))
            rr["visible_chars"], rr["thinking_chars"] = len(v), th
        req_out.append(rr)
    ex: dict[str, Any] = {
        "source": "llama-server", "server_variant": server_variant, "spec_type": spec_type,
        "draft": dict(draft) if draft else None, "speculative_types_reported": reported,
        "n_requests": agg["n_requests"], "n_ok": agg["n_ok"], "n_drafting": agg["n_drafting"],
        "pp_tokens": (agg.get("pp") or {}).get("tokens"),
        "tg_tokens": (agg.get("tg") or {}).get("tokens"),
        "identical_by_prompt": cmp["identical_by_prompt"],
        "similarity_by_prompt": cmp["similarity_by_prompt"],
        "n_compared_to_plain": cmp["n_compared"], "n_similar_ge_095": cmp["n_similar_ge_095"],
        "plain_ref": plain_source, "coherent_by_prompt": {
            str(r["id"]): c[0] for r, c in zip(ok_recs, coh_list)} or None,
        "coherence_impl": coh_list[0][2] if coh_list else None,
        "mem": dict(mem) if mem else None, "requests": req_out,
    }
    if extra:
        ex.update(extra)
    f: dict[str, Any] = {
        "experiment": experiment, "kind": "measured", "llama_build": build[0],
        "llama_commit": build[1], "model": dict(model) if model is not None else None,
        "variant": variant, "flags": [str(x) for x in flags], "threads": threads,
        "n_ctx": n_ctx, "load_s": load_time_s,
        "rss_peak_mb": _mb((mem or {}).get("rss_peak_bytes")),
        "private_peak_mb": _mb((mem or {}).get("private_peak_bytes")),
        "mapped_mb": _mb((mem or {}).get("mapped_peak_bytes")),
        "accept_rate": agg["accept_rate"], "draft_n": agg["draft_n"],
        "draft_n_accepted": agg["draft_n_accepted"],
        "spec_type_effective": ste, "spec_active": None if is_plain else agg["spec_active"],
        "identical_to_plain": None if is_plain else cmp["identical_to_plain"],
        "coherent": coherent,
        "output_sample": str(ok_recs[0]["content"]) if ok_recs else None,
        "extra": ex,
    }
    if failure is None and agg["n_ok"] > 0 and agg.get("tg"):
        f["tg"] = {k: agg["tg"][k] for k in ("tok_s", "sd", "n")}
        if agg.get("pp"):
            f["pp"] = {k: agg["pp"][k] for k in ("tok_s", "sd", "n")}
    elif failure is None:
        failure = "failed: no successful /completion request"
    if failure:
        notes.append(failure if failure.startswith("failed") else f"failed: {failure}")
    n_err = sum(1 for r in records if r.get("error"))
    if n_err and failure is None:
        notes.append(f"{n_err} of {len(records)} requests failed (excluded from the rates)")
    if spec_configured and agg["n_ok"] and not agg["draft_n"]:
        notes.append("speculation configured but timings.draft_n absent/0 on every request: "
                     "no token was drafted")
    if not is_plain and plain is None:
        notes.append("no plain reference available: identical_to_plain n/a")
    if (not is_plain and agg["spec_active"] and cmp["identical_to_plain"] is not None
            and cmp["identical_to_plain"] < 1.0):
        n_diff = sum(1 for v in (cmp["identical_by_prompt"] or {}).values() if not v)
        notes.append(f"active drafter changed the greedy output on {n_diff} of "
                     f"{cmp['n_compared']} prompts (speculative greedy decoding should be lossless)")
    note_b = _build_note(*build)
    if note_b:
        notes.append(note_b)
    if os.environ.get("PIGEMMA_EMULATED", "").strip() == "1":
        notes.append("emulated host (qemu): timings are a functional proof, not a Pi measurement")
    if notes:
        f["notes"] = notes
    return f


def find_plain_reference(out_dir: Path, model_id: str | None, load_suffix: str,
                         n_predict: int | None) -> tuple[dict[str, str], str] | None:
    """Newest E1 'plain<load_suffix>' result for the model in out_dir -> ({id: content}, path)."""
    if not out_dir.is_dir():
        return None
    try:
        docs = results.load_results(out_dir, recursive=False)
    except Exception as e:  # noqa: BLE001
        log.warning("cannot scan %s for a plain reference: %s", out_dir, e)
        return None
    want = "plain" + load_suffix
    best: tuple[str, dict[str, str], str] | None = None
    for d in docs:
        if d.get("experiment") != "E1" or d.get("variant") != want:
            continue
        m = d.get("model") or {}
        if model_id and m.get("id") != model_id:
            continue
        ex = d.get("extra") or {}
        reqs = ex.get("requests") or []
        if not reqs:
            continue
        if n_predict is not None and any(r.get("n_predict") not in (None, n_predict) for r in reqs):
            continue
        contents = {str(r.get("id")): r.get("content") for r in reqs if r.get("content") is not None}
        if not contents:
            continue
        ts = str(d.get("created_at"))
        if best is None or ts > best[0]:
            best = (ts, contents, str(d.get("_path")))
    return (best[1], _rel(best[2]) or best[2]) if best else None


def load_plain_ref_file(p: Path) -> dict[str, str]:
    """{id: content} from a plain result envelope (extra.requests)."""
    with open(p, encoding="utf-8-sig") as f:
        d = json.load(f)
    reqs = ((d.get("extra") or {}).get("requests")) if isinstance(d, Mapping) else None
    if not reqs:
        raise BenchError(f"{p}: no extra.requests (not a server-spec result)")
    return {str(r.get("id")): r.get("content") for r in reqs if r.get("content") is not None}


# --------------------------------------------------------------------------------------------
# Idle-machine check
# --------------------------------------------------------------------------------------------
DEFAULT_IDLE_CHECK_S = 10.0
DEFAULT_IDLE_MAX_PCT = 5.0


def _system_cpu_times() -> tuple[float, float, str] | None:
    """(busy, total, source) cumulative CPU time over all CPUs, or None when unavailable."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/stat", encoding="ascii", errors="replace") as f:
                parts = f.readline().split()
            if parts and parts[0] == "cpu" and len(parts) >= 5:
                v = [float(x) for x in parts[1:9]]  # user nice system idle iowait irq softirq steal
                idle = v[3] + (v[4] if len(v) > 4 else 0.0)
                return sum(v) - idle, sum(v), "procfs"
        except (OSError, ValueError):
            pass
    if paths.is_windows():
        try:
            import ctypes  # noqa: PLC0415
            from ctypes import wintypes  # noqa: PLC0415
            idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern),
                                                     ctypes.byref(user)):
                def secs(t: Any) -> float:
                    return ((int(t.dwHighDateTime) << 32) | int(t.dwLowDateTime)) / 1e7
                total = secs(kern) + secs(user)  # kernel time includes the idle time
                return total - secs(idle), total, "win32"
        except Exception:  # noqa: BLE001
            pass
    if psutil is not None:
        try:
            t = psutil.cpu_times()
            total = float(sum(t))
            idle = float(t.idle) + float(getattr(t, "iowait", 0.0))
            return total - idle, total, "psutil"
        except Exception:  # noqa: BLE001
            pass
    return None


def measure_background_cpu(seconds: float = DEFAULT_IDLE_CHECK_S, step: float = 1.0
                           ) -> dict[str, Any]:
    """Total system CPU utilisation (all cores, percent) over ``seconds``.

    Returns {mean_pct, max_pct (worst ``step``-second window), seconds, n_windows, source};
    the percentages are None when no CPU counters are available.
    """
    first = _system_cpu_times()
    if first is None:
        return {"mean_pct": None, "max_pct": None, "seconds": 0.0, "n_windows": 0, "source": None}
    prev = first
    t0 = time.monotonic()
    worst: float | None = None
    n = 0
    while True:
        remaining = seconds - (time.monotonic() - t0)
        if remaining <= 0:
            break
        time.sleep(min(step, remaining))
        cur = _system_cpu_times()
        if cur is None:
            break
        db, dt = cur[0] - prev[0], cur[1] - prev[1]
        if dt > 0:
            pct = 100.0 * db / dt
            worst = pct if worst is None else max(worst, pct)
            n += 1
        prev = cur
    db, dt = prev[0] - first[0], prev[1] - first[1]
    mean = 100.0 * db / dt if dt > 0 else None
    return {"mean_pct": round(mean, 2) if mean is not None else None,
            "max_pct": round(worst, 2) if worst is not None else None,
            "seconds": round(time.monotonic() - t0, 2), "n_windows": n, "source": first[2],
            # inside Docker /proc/stat covers the Docker VM only: the Windows host's load
            # (browsers, other containers' host side) is invisible from here
            "scope": "container-vm" if paths.in_container() else "host"}


def idle_check(args: argparse.Namespace, label: str) -> dict[str, Any] | None:
    """Measure background CPU before a run (see the module docstring); None when disabled.

    Raises RefusedError with --require-idle when the machine stays busy.
    """
    secs = float(getattr(args, "idle_check_s", 0) or 0)
    if secs <= 0:
        return None
    max_pct = float(getattr(args, "idle_max_pct", DEFAULT_IDLE_MAX_PCT))
    deadline = time.monotonic() + float(getattr(args, "idle_wait_s", 0) or 0)
    while True:
        info = measure_background_cpu(secs)
        mean = info["mean_pct"]
        info["max_allowed_pct"] = max_pct
        info["idle"] = None if mean is None else bool(mean <= max_pct)
        if info["idle"] is not False or time.monotonic() >= deadline:
            break
        log.info("background CPU %.1f%% > %.1f%% before %s; waiting for an idle window",
                 mean, max_pct, label)
    if info["idle"] is False:
        msg = (f"background CPU load {info['mean_pct']:.1f}% (> {max_pct:g}%) in the "
               f"{info['seconds']:.0f} s before {label}")
        if getattr(args, "require_idle", False):
            raise RefusedError(msg + " (--require-idle)")
        log.warning("%s: the timing is not idle-machine quality", msg)
    elif info["idle"] is None:
        log.warning("background CPU load unknown before %s (no CPU counters)", label)
    else:
        log.info("idle check before %s: %.1f%% CPU over %.0f s (<= %g%%)", label,
                 info["mean_pct"], info["seconds"], max_pct)
    return info


def idle_note(info: Mapping[str, Any] | None) -> str | None:
    """The envelope note for a non-idle (or unmeasurable) background, else None."""
    if not info:
        return None
    if info.get("idle") is False:
        return (f"background CPU load {info['mean_pct']:.1f}% (> {info['max_allowed_pct']:g}%) in "
                f"the {info['seconds']:.0f} s before the run: not an idle-machine timing")
    if info.get("idle") is None:
        return "background CPU load unknown (no CPU counters on this host)"
    return None


def _add_note(fields: dict[str, Any], note: str | None) -> None:
    if note:
        fields.setdefault("notes", []).append(note)


# --------------------------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------------------------
def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(name)s: %(levelname)s: %(message)s", stream=sys.stderr)


def _utf8_stdio() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _out_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    return paths.RESULTS_DIR / _dt.date.today().isoformat()


def _split_flags(s: str) -> list[str]:
    """Shell-split one --flags string (non-POSIX rules on Windows so backslashes survive)."""
    if not paths.is_windows():
        return shlex.split(s)
    out = []
    for x in shlex.split(s, posix=False):
        out.append(x[1:-1] if len(x) >= 2 and x[0] == x[-1] and x[0] in "\"'" else x)
    return out


def _extra_flags(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for s in args.flags or []:
        out.extend(_split_flags(s))
    out.extend(getattr(args, "extra", None) or [])
    return out


def _resolve_model_arg(spec: str, *, require: bool) -> tuple[Path, str | None]:
    p, mid = resolve_model(spec)
    if require and not p.is_file():
        hint = ""
        if mid in paths.MODELS:
            hint = "; " + paths.download_hint(mid)
        raise MissingInputError(f"model {spec!r} not found at {p}{hint}")
    return p, mid


def _stem(experiment: str, model: Mapping[str, Any] | None, variant: str, threads: Any) -> str:
    mid = (model or {}).get("id") or "nomodel"
    return "_".join(_safe_part(x) for x in (experiment, mid, variant,
                                            threads if threads is not None else "na",
                                            _ts_compact()))


def _write(out_dir: Path, fields: Mapping[str, Any]) -> Path:
    p = results.write_result(out_dir, **fields)
    print(paths.repo_relative(p), flush=True)
    return p


def _lock_ctx(args: argparse.Namespace, cmd: str) -> Any:
    mf = args.min_free_gb if args.min_free_gb is not None else paths.DEFAULT_MIN_FREE_GB
    return paths.bench_lock(mf, cmd=cmd)


def _summary_line(f: Mapping[str, Any]) -> str:
    def r(k: str) -> str:
        b = f.get(k)
        return f"{b['tok_s']:.2f}" if isinstance(b, Mapping) and b.get("tok_s") is not None else "-"
    bits = [f"{f.get('variant')}", f"t={f.get('threads')}", f"pp={r('pp')}", f"tg={r('tg')}"]
    for k in ("load_s", "rss_peak_mb", "private_peak_mb", "mapped_mb", "accept_rate"):
        if f.get(k) is not None:
            bits.append(f"{k}={f[k]}")
    if f.get("buffer_split"):
        bits.append("buffers=" + ",".join(f"{k}:{v}" for k, v in f["buffer_split"].items()))
    if f.get("notes"):
        bits.append("notes=" + "; ".join(f["notes"])[:200])
    return "  ".join(bits)


# --------------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------------
def cmd_matrix(args: argparse.Namespace) -> int:
    threads = _parse_int_list(args.threads)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.kvq8:
        variants = [v if "kvq8" in v.split("+") else f"{v}+kvq8" for v in variants]
    for v in variants:
        split_variant(v)
    if args.pp <= 0 and args.tg <= 0:
        raise ValueError("--pp and --tg are both 0: nothing to measure")
    extra = _extra_flags(args)
    experiment = args.experiment or default_experiment("matrix")
    cmd = _tool_cmd("llama-bench", args.bin)
    model_path, mid = _resolve_model_arg(args.model, require=not args.dry_run)
    if args.dry_run:
        for v in variants:
            print(_cmd_str(llama_bench_argv(cmd, model_path, threads=threads, pp=args.pp,
                                            tg=args.tg, reps=args.reps,
                                            flags=variant_flags(v, "bench") + extra)))
        return paths.EXIT_OK
    _check_cmd(cmd)
    out_dir = _out_dir(args.out)
    model = model_block(model_path, mid, with_sha=not args.no_sha)
    build = binary_version(cmd)
    expected = [(t, k) for t in threads for k, n in (("pp", args.pp), ("tg", args.tg)) if n > 0]
    rc_all = paths.EXIT_OK
    with _lock_ctx(args, f"pigemma.bench matrix {model.get('id')} {','.join(variants)}"):
        for v in variants:
            flags = variant_flags(v, "bench") + extra
            stem = _stem(experiment, model, v, ",".join(map(str, threads)))
            raw_out = out_dir / "raw" / f"{stem}.llama-bench.stdout.txt"
            raw_err = out_dir / "raw" / f"{stem}.llama-bench.stderr.log"
            log.info("matrix: variant %s (%s)", v, VARIANT_NOTES.get(v.split("+")[0], ""))
            bg = idle_check(args, f"llama-bench {v}")
            run = run_llama_bench(model_path, threads=threads, pp=args.pp, tg=args.tg,
                                  reps=args.reps, flags=flags, binary=args.bin,
                                  timeout=args.timeout, nice=args.nice,
                                  stdout_path=raw_out, stderr_path=raw_err)
            common = {"argv": [str(a) for a in run.argv], "wall_s": run.wall_s,
                      "returncode": run.returncode, "raw_stdout": _rel(raw_out),
                      "raw_stderr": _rel(raw_err), "variant_note": VARIANT_NOTES.get(v),
                      "background_cpu": bg,
                      "token_source": "random token ids (llama-bench): engine throughput only, "
                                      "not routing / locality"}
            fields = bench_rows_to_fields(run.rows, experiment=experiment, model=model,
                                          variant=v, flags=flags, extra=common)
            got = {(r.get("n_threads"), r.get("kind")) for r in run.rows}
            missing = [e for e in expected if e not in got]
            if missing or not run.ok:
                rc_all = paths.EXIT_ERROR
                fields += _bench_failure_fields(
                    missing, experiment=experiment, model=model, variant=v, flags=flags,
                    run=run, pp=args.pp, tg=args.tg,
                    build=(build.get("build_number"), build.get("commit")), extra=common)
                if not missing:  # rows complete but non-zero exit: say so on every row
                    for f in fields:
                        f.setdefault("notes", []).append(
                            failure_note(run.stderr, prefix="warning",
                                         extra=f"llama-bench exit code {run.returncode}"))
            for f in fields:
                _add_note(f, idle_note(bg))
                _write(out_dir, f)
                print(f"pigemma.bench: {_summary_line(f)}", file=sys.stderr)
    return rc_all


def _read_prompt_arg(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.prompt is not None:
        return args.prompt, None
    p = Path(args.prompt_file).expanduser()
    if not p.is_file():
        raise MissingInputError(f"prompt file not found: {p}")
    text = p.read_bytes().decode("utf-8", errors="surrogateescape")
    if text.startswith("﻿"):
        text = text[1:]
    return text, _rel(p)


def cmd_completion(args: argparse.Namespace) -> int:
    split_variant(args.variant)
    extra = _extra_flags(args)
    flags = variant_flags(args.variant, "run") + extra
    experiment = args.experiment or default_experiment("completion")
    cmd = _tool_cmd("llama-completion", args.bin)
    model_path, mid = _resolve_model_arg(args.model, require=not args.dry_run)
    text, prompt_rel = _read_prompt_arg(args)
    if args.dry_run:
        print(_cmd_str(completion_argv(cmd, model_path, "<temp prompt file>", n_predict=args.n,
                                       threads=args.threads, n_ctx=args.ctx, flags=flags)))
        return paths.EXIT_OK
    _check_cmd(cmd)
    out_dir = _out_dir(args.out)
    model = model_block(model_path, mid, with_sha=not args.no_sha)
    build = binary_version(cmd)
    stem = _stem(experiment, model, args.variant, args.threads)
    raw_out = out_dir / "raw" / f"{stem}.completion.stdout.txt"
    raw_err = out_dir / "raw" / f"{stem}.completion.stderr.log"
    with _lock_ctx(args, f"pigemma.bench completion {model.get('id')} {args.variant}"):
        bg = idle_check(args, f"llama-completion {args.variant}")
        run = run_completion(model_path, text, n_predict=args.n, threads=args.threads,
                             n_ctx=args.ctx, flags=flags, chat=args.chat, binary=args.bin,
                             timeout=args.timeout, sample_interval=args.sample_interval,
                             nice=args.nice, stdout_path=raw_out, stderr_path=raw_err)
    f = completion_fields(run, experiment=experiment, model=model, variant=args.variant,
                          flags=flags, threads=args.threads, n_ctx=args.ctx, n_predict=args.n,
                          chat=args.chat, prompt_path=prompt_rel,
                          build=(build.get("build_number"), build.get("commit")),
                          raw={"raw_stdout": _rel(raw_out), "raw_stderr": _rel(raw_err),
                               "background_cpu": bg})
    _add_note(f, idle_note(bg))
    _write(out_dir, f)
    print(f"pigemma.bench: {_summary_line(f)}", file=sys.stderr)
    if args.print_output:
        print(run.stdout_text, file=sys.stderr)
    return paths.EXIT_OK if run.ok else paths.EXIT_ERROR


def _server_variant_plan(variant: str, model_path: Path, mid: str | None, draft_override: str | None
                         ) -> tuple[Path | None, str | None, str | None]:
    """(draft path, draft id, spec_type) for a server variant."""
    dtype, spec_type = SERVER_VARIANT_SPEC[variant]
    if dtype is None:
        return None, None, spec_type
    if draft_override:
        return Path(draft_override).expanduser(), paths.model_id_for(draft_override), spec_type
    dp, did = resolve_draft(mid, model_path, dtype)
    return dp, did, spec_type


def cmd_server_spec(args: argparse.Namespace) -> int:
    variants: list[str] = []
    for v in args.variant.split(","):
        v = v.strip()
        if not v:
            continue
        if v not in SERVER_VARIANT_SPEC:
            raise ValueError(f"unknown server variant {v!r} (known: {', '.join(SERVER_VARIANTS)})")
        if v not in variants:
            variants.append(v)
    if not variants:
        raise ValueError("no server variant given")
    if "plain" in variants:  # the reference first
        variants.remove("plain")
        variants.insert(0, "plain")
    if args.n is not None and args.n <= 0:
        args.n = None  # each item's max_tokens
    load_v = args.load_variant or "default"
    split_variant(load_v)
    load_suffix = "" if load_v == "default" else f"+{load_v}"
    extra = _extra_flags(args)
    base_flags = variant_flags(load_v, "run") + extra
    experiment = args.experiment or default_experiment("server-spec")
    cmd = _tool_cmd("llama-server", args.bin)
    model_path, mid = _resolve_model_arg(args.model, require=not args.dry_run)
    prompts_path = Path(args.prompts).expanduser()
    items = load_prompts(prompts_path)
    if args.max_prompts:
        items = items[: args.max_prompts]
    if args.dry_run:
        for v in [v for _ in range(max(1, int(args.rounds or 1))) for v in variants]:
            dp, _, st = _server_variant_plan(v, model_path, mid, args.draft)
            srv = LlamaServer(model_path, threads=args.threads, n_ctx=args.ctx, port=args.port,
                              draft=dp, spec_type=st, draft_n_max=args.draft_n_max,
                              flags=base_flags, binary=args.bin)
            print(_cmd_str(srv.argv))
            print(f"  then: GET /health until ok; POST /completion x {len(items)} "
                  f"(n_predict {args.n or 'per item'}, temperature 0, cache_prompt false)")
        return paths.EXIT_OK
    _check_cmd(cmd)
    out_dir = _out_dir(args.out)
    model = model_block(model_path, mid, with_sha=not args.no_sha)
    build_info = binary_version(cmd)
    build = (build_info.get("build_number"), build_info.get("commit"))
    plain_contents: dict[str, str] | None = None
    plain_source: str | None = None
    if args.plain_ref:
        plain_contents = load_plain_ref_file(Path(args.plain_ref))
        plain_source = _rel(args.plain_ref)
    rc_all = paths.EXIT_OK
    rounds = max(1, int(args.rounds or 1))
    queue: list[tuple[str, int]] = [(v, r) for r in range(1, rounds + 1) for v in variants]
    done: set[tuple[str, int]] = set()
    with _lock_ctx(args, f"pigemma.bench server-spec {model.get('id')} {','.join(variants)}"):
        while queue:
            v, rnd = queue.pop(0)
            if (v, rnd) in done:
                continue
            done.add((v, rnd))
            env_variant = v + load_suffix
            dp, did, st = _server_variant_plan(v, model_path, mid, args.draft)
            flags = list(base_flags)
            draft_info = None
            if dp is not None:
                flags = ["-md", _rel(dp) or str(dp)] + flags
                draft_info = {"id": did, "path": _rel(dp),
                              "bytes": dp.stat().st_size if dp.is_file() else None}
            if st:
                flags = flags + ["--spec-type", st]
            if args.draft_n_max is not None and (dp is not None or st):
                flags = flags + ["--spec-draft-n-max", str(args.draft_n_max)]
            if plain_contents is None and v != "plain":
                found = find_plain_reference(out_dir, model.get("id"), load_suffix, args.n)
                if found:
                    plain_contents, plain_source = found
            stem = _stem(experiment, model, env_variant, args.threads)
            log_path = out_dir / "raw" / f"{stem}.server.log"
            ex_common = {"argv": None, "server_log": _rel(log_path),
                         "prompts_file": _rel(prompts_path),
                         "prompts_sha256": _sha256_file(prompts_path),
                         "n_predict": args.n, "warmup_tokens": args.warmup_tokens,
                         "port": args.port, "load_variant": load_v, "round": rnd,
                         "rounds": rounds}
            common_kw = dict(experiment=experiment, model=model, variant=env_variant,
                             flags=flags, threads=args.threads, n_ctx=args.ctx,
                             server_variant=v, spec_type=st, draft=draft_info,
                             plain=plain_contents if v != "plain" else None,
                             plain_source=plain_source if v != "plain" else None, build=build)
            if SERVER_VARIANT_SPEC[v][0] is not None and (dp is None or not dp.is_file()):
                note = (f"failed: no '{SERVER_VARIANT_SPEC[v][0]}' draft found for model "
                        f"{model.get('id')!r} ({dp if dp else 'not in the catalog'})")
                f = server_fields([], load_time_s=None, mem=None, failure=note,
                                  extra=ex_common, **common_kw)
                _write(out_dir, f)
                rc_all = max(rc_all, paths.EXIT_ERROR)
                continue
            srv = LlamaServer(model_path, threads=args.threads, n_ctx=args.ctx, port=args.port,
                              draft=dp, spec_type=st, draft_n_max=args.draft_n_max
                              if (dp is not None or st) else None, flags=base_flags,
                              binary=args.bin, log_path=log_path,
                              health_timeout=args.health_timeout,
                              sample_interval=args.sample_interval, nice=args.nice)
            ex_common["argv"] = [str(a) for a in srv.argv]
            records: list[dict[str, Any]] = []
            failure: str | None = None
            bg = idle_check(args, f"llama-server {env_variant} (round {rnd}/{rounds})")
            ex_common["background_cpu"] = bg
            try:
                srv.start()
                records = run_prompt_set(srv, items, n_predict=args.n, chat=not args.raw_prompts,
                                         warmup_tokens=args.warmup_tokens,
                                         request_timeout=args.request_timeout)
            except ServerStartError as e:
                failure = str(e)
            finally:
                srv.stop()
            mem = srv.monitor.summary() if srv.monitor else None
            ex = dict(ex_common)
            ex["server_returncode"] = srv.returncode
            ex["health_wait_s"] = srv.health_wait_s
            ex["load_time_source"] = srv.load_time_source
            f = server_fields(records, load_time_s=srv.load_time_s, mem=mem, failure=failure,
                              extra=ex, **common_kw)
            _add_note(f, idle_note(bg))
            _write(out_dir, f)
            print(f"pigemma.bench: {_summary_line(f)}", file=sys.stderr)
            if failure or "tg" not in f:
                rc_all = max(rc_all, paths.EXIT_ERROR)
            if v == "plain" and records:
                plain_contents = {str(r["id"]): r["content"] for r in records
                                  if r.get("content") is not None}
                plain_source = "this invocation"
            xv = (f"{v}-explicit", rnd)
            if (args.auto_explicit and v in ("mtp", "dflash") and f.get("spec_active") is False
                    and xv not in done and xv not in queue):
                log.info("%s never drafted; --auto-explicit adds %s-explicit", v, v)
                queue.insert(0, xv)
    return rc_all


def _model_for_parse(spec: str | None, rows_path: str | None, no_sha: bool) -> dict[str, Any] | None:
    if spec:
        p, mid = resolve_model(spec)
        return model_block(p, mid, with_sha=not no_sha and p.is_file())
    if rows_path:
        p = Path(rows_path)
        mid = paths.model_id_for(p) or catalog_id_for(p)
        return model_block(paths.model_path(p) if p.exists() else p, mid,
                           with_sha=not no_sha and p.is_file())
    return None


def _read_input(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    p = Path(arg).expanduser()
    if not p.is_file():
        raise MissingInputError(f"input file not found: {p}")
    return p.read_bytes().decode("utf-8", errors="replace")


def cmd_parse_bench(args: argparse.Namespace) -> int:
    text = _read_input(args.input)
    rows = parse_bench_json(text)
    if not rows:
        print("pigemma.bench: no llama-bench rows found in the input", file=sys.stderr)
        return paths.EXIT_ERROR
    model = _model_for_parse(args.model, rows[0].get("model_filename"), args.no_sha)
    flags = _extra_flags(args)
    experiment = args.experiment or default_experiment("parse-bench")
    extra = {"parsed_from": None if args.input == "-" else _rel(args.input)}
    if args.stderr:
        extra["raw_stderr"] = _rel(args.stderr)
    fields = []
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_variant.setdefault(args.variant or infer_variant(r), []).append(r)
    for v, rs in by_variant.items():
        fl = flags or variant_flags(v, "bench")
        fields += bench_rows_to_fields(rs, experiment=experiment, model=model, variant=v,
                                       flags=fl, extra=extra)
    if args.out:
        out_dir = Path(args.out).expanduser()
        for f in fields:
            _write(out_dir, f)
    else:
        docs = [results.build_result(**f) for f in fields]
        print(json.dumps(docs, indent=1, ensure_ascii=False))
    return paths.EXIT_OK


def cmd_parse_completion(args: argparse.Namespace) -> int:
    err = _read_input(args.input)
    parsed = parse_completion_stderr(err)
    if not args.out and not args.model:
        print(json.dumps(parsed, indent=1, ensure_ascii=False))
        return paths.EXIT_OK if parsed["has_perf"] else paths.EXIT_ERROR
    out_text = _read_input(args.stdout) if args.stdout else ""
    run = CompletionRun(argv=[], returncode=0 if parsed["has_perf"] else None,
                        stdout_text=out_text.replace("\r\n", "\n"), stderr_text=err,
                        wall_s=0.0, parsed=parsed, mem={}, io={}, marks={})
    model = _model_for_parse(args.model, None, args.no_sha)
    flags = _extra_flags(args) or variant_flags(args.variant, "run")
    f = completion_fields(run, experiment=args.experiment or default_experiment("parse-completion"),
                          model=model, variant=args.variant, flags=flags,
                          threads=args.threads or parsed.get("n_threads"),
                          n_ctx=parsed.get("n_ctx"), n_predict=parsed.get("n_predict"),
                          chat=False, prompt_path=None,
                          raw={"parsed_from": None if args.input == "-" else _rel(args.input),
                               "source": "llama-completion (parsed log; no memory sampling)"})
    f.pop("rss_peak_mb", None)
    f.pop("private_peak_mb", None)
    f.pop("mapped_mb", None)
    f["notes"] = [n for n in f.get("notes", []) if "read_bytes" not in n]
    if args.out:
        _write(Path(args.out).expanduser(), f)
    else:
        print(json.dumps(results.build_result(**f), indent=1, ensure_ascii=False))
    return paths.EXIT_OK if run.ok else paths.EXIT_ERROR


def cmd_parse_server(args: argparse.Namespace) -> int:
    recs = []
    for a in args.inputs:
        d = json.loads(_read_input(a))
        for i, resp in enumerate(d if isinstance(d, list) else [d]):
            r = parse_timings(resp)
            r["id"] = f"{Path(a).stem}#{i}" if isinstance(d, list) else Path(a).stem
            r["content"] = resp.get("content") if isinstance(resp, Mapping) else None
            recs.append(r)
    agg = aggregate_timings(recs)
    out = {"aggregate": agg,
           "requests": [{k: v for k, v in r.items() if k != "content"} for r in recs]}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return paths.EXIT_OK


def cmd_variants(args: argparse.Namespace) -> int:
    table = {v: {"bench": list(d["bench"]), "run": list(d["run"]), "note": VARIANT_NOTES[v]}
             for v, d in VARIANT_FLAGS.items()}
    srv = {v: {"draft_type": d, "spec_type": s} for v, (d, s) in SERVER_VARIANT_SPEC.items()}
    if args.json:
        print(json.dumps({"load_variants": table, "server_variants": srv}, indent=1))
        return paths.EXIT_OK
    for v, d in table.items():
        print(f"{v:15s} llama-bench: {_cmd_str(d['bench']) or '(none)':45s} others: "
              f"{_cmd_str(d['run']) or '(none)'}")
    for v, d in srv.items():
        print(f"server {v:16s} draft={d['draft_type'] or '-':7s} --spec-type {d['spec_type'] or '-'}")
    return paths.EXIT_OK


# --------------------------------------------------------------------------------------------
# Argument parser / main
# --------------------------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pigemma.bench",
        description="Run llama-bench / llama-completion / llama-server variants and write "
                    "results envelopes (kind=measured). See the module docstring for details.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 ok, 1 a run failed (result written with notes) / error, 2 usage, "
               "3 missing input, 4 refused (bench lock busy, low RAM, port in use).")
    p.add_argument("--log-level", default="INFO", help="logging level (default INFO)")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<cmd>")

    run_common = argparse.ArgumentParser(add_help=False)
    run_common.add_argument("--model", "-m", required=True, help="registry/catalog id or GGUF path")
    run_common.add_argument("--out", help="output directory (default results/<YYYY-mm-dd>/)")
    run_common.add_argument("--experiment", help="E0..E8|adhoc (default: see module docstring)")
    run_common.add_argument("--flags", action="append", default=[],
                            help="extra llama.cpp flags (one string, shell-split; repeatable)")
    run_common.add_argument("--bin", help="binary override (default: paths.bin)")
    run_common.add_argument("--min-free-gb", type=float, default=None,
                            help="bench-lock free-RAM gate in GB (default paths.DEFAULT_MIN_FREE_GB; "
                                 "env PIGEMMA_MIN_FREE_GB wins)")
    run_common.add_argument("--timeout", type=float, default=None,
                            help="seconds before the child is terminated (default: none)")
    run_common.add_argument("--nice", type=int, default=None, help="POSIX niceness for the child")
    run_common.add_argument("--no-sha", action="store_true",
                            help="skip the 16 MiB sha256 prefix of the model")
    run_common.add_argument("--dry-run", action="store_true",
                            help="print the command lines and exit without running anything")
    run_common.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_S,
                            help="memory sampler period in seconds (default 0.25)")
    run_common.add_argument("--idle-check-s", type=float, default=DEFAULT_IDLE_CHECK_S,
                            help="seconds of background-CPU measurement before every run "
                                 "(default 10; 0 disables)")
    run_common.add_argument("--idle-max-pct", type=float, default=DEFAULT_IDLE_MAX_PCT,
                            help="max total CPU %% for an idle machine (default 5)")
    run_common.add_argument("--idle-wait-s", type=float, default=0.0,
                            help="keep re-measuring up to S seconds for an idle window")
    run_common.add_argument("--require-idle", action="store_true",
                            help="refuse (exit 4) instead of running on a busy machine")

    m = sub.add_parser("matrix", parents=[run_common], help="llama-bench variant matrix (E0)")
    m.add_argument("--threads", "-t", default="4,6,12", help="thread list (default 4,6,12)")
    m.add_argument("--pp", type=int, default=128, help="prompt tokens (0 = skip pp)")
    m.add_argument("--tg", type=int, default=32, help="generated tokens (0 = skip tg)")
    m.add_argument("-r", "--reps", type=int, default=3, help="repetitions (default 3)")
    m.add_argument("--variants", default="default,norepack,hybrid_negctrl,nommap",
                   help="comma list of load variants ('+' combines, e.g. norepack+kvq8)")
    m.add_argument("--kvq8", action="store_true", help="add +kvq8 to every variant")
    m.set_defaults(func=cmd_matrix)

    c = sub.add_parser("completion", parents=[run_common],
                       help="one llama-completion run with memory/I-O sampling")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt-file", "-f", help="prompt text file (UTF-8)")
    g.add_argument("--prompt", "-p", help="prompt text")
    c.add_argument("-n", "--n-predict", dest="n", type=int, default=128, help="tokens to generate")
    c.add_argument("--threads", "-t", type=int, default=6)
    c.add_argument("--variant", default="default", help="load variant (default 'default')")
    c.add_argument("--chat", action="store_true", help="wrap the prompt in the Gemma 4 template")
    c.add_argument("-c", "--ctx", type=int, default=DEFAULT_N_CTX, help="context size (4096)")
    c.add_argument("--print-output", action="store_true", help="echo the generated text to stderr")
    c.set_defaults(func=cmd_completion)

    s = sub.add_parser("server-spec", parents=[run_common],
                       help="speculative-decoding variants through llama-server timings (E1)")
    s.add_argument("--variant", default="plain",
                   help=f"comma list of {', '.join(SERVER_VARIANTS)} (plain always runs first)")
    s.add_argument("--load-variant", default="default", help="load variant flags for the server")
    s.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="prompts JSON")
    s.add_argument("--max-prompts", type=int, default=None, help="use only the first N prompts")
    s.add_argument("-n", "--n-predict", dest="n", type=int, default=DEFAULT_SERVER_N_PREDICT,
                   help="n_predict per request (default 200; 0 = each item's max_tokens)")
    s.add_argument("--threads", "-t", type=int, default=6)
    s.add_argument("-c", "--ctx", type=int, default=DEFAULT_N_CTX)
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--draft", help="draft GGUF override (default: catalog draft of the model)")
    s.add_argument("--draft-n-max", type=int, default=None, help="--spec-draft-n-max N")
    s.add_argument("--warmup-tokens", type=int, default=WARMUP_TOKENS)
    s.add_argument("--health-timeout", type=float, default=HEALTH_TIMEOUT_S)
    s.add_argument("--request-timeout", type=float, default=None)
    s.add_argument("--plain-ref", help="plain-variant result JSON to compare against")
    s.add_argument("--raw-prompts", action="store_true", help="do not chat-wrap the prompts")
    s.add_argument("--auto-explicit", action="store_true",
                   help="also run <x>-explicit when mtp/dflash never drafted")
    s.add_argument("--rounds", type=int, default=1,
                   help="repeat the variant list N times, interleaved (A,B,A,B; default 1)")
    s.set_defaults(func=cmd_server_spec)

    pb = sub.add_parser("parse-bench", help="llama-bench JSON file -> envelopes (no model run)")
    pb.add_argument("input", help="llama-bench -o json output file, or - for stdin")
    pb.add_argument("--model", "-m", help="id or path (default: the rows' model_filename)")
    pb.add_argument("--variant", help="variant name (default: inferred from the rows)")
    pb.add_argument("--flags", action="append", default=[], help="flags to record")
    pb.add_argument("--experiment")
    pb.add_argument("--stderr", help="llama-bench stderr log to reference in extra")
    pb.add_argument("--no-sha", action="store_true")
    pb.add_argument("--out", help="write envelopes into this directory (default: print JSON)")
    pb.set_defaults(func=cmd_parse_bench, extra=[])

    pc = sub.add_parser("parse-completion", help="llama-completion stderr -> parsed JSON / envelope")
    pc.add_argument("input", help="stderr log file, or - for stdin")
    pc.add_argument("--stdout", help="the run's stdout (output_sample / coherence)")
    pc.add_argument("--model", "-m")
    pc.add_argument("--variant", default="default")
    pc.add_argument("--threads", type=int, default=None)
    pc.add_argument("--flags", action="append", default=[])
    pc.add_argument("--experiment")
    pc.add_argument("--no-sha", action="store_true")
    pc.add_argument("--out", help="write an envelope into this directory")
    pc.set_defaults(func=cmd_parse_completion, extra=[])

    ps = sub.add_parser("parse-server", help="llama-server /completion JSON(s) -> timing summary")
    ps.add_argument("inputs", nargs="+")
    ps.set_defaults(func=cmd_parse_server)

    pv = sub.add_parser("variants", help="print the variant -> flags table")
    pv.add_argument("--json", action="store_true")
    pv.set_defaults(func=cmd_variants)
    return p


def _install_sigterm() -> Any:
    """SIGTERM -> SystemExit(143) so finally blocks stop our own children. Returns the previous
    handler (restored by main) or None when nothing was installed."""
    if paths.is_windows() or threading.current_thread() is not threading.main_thread():
        return None

    def _h(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)
    try:
        prev = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _h)
        return prev
    except (ValueError, OSError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    _utf8_stdio()
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else paths.EXIT_USAGE
    args.extra = extra if extra else getattr(args, "extra", [])
    _setup_logging(args.log_level)
    prev_term = _install_sigterm()
    try:
        return int(args.func(args))
    except (paths.LockBusyError, paths.InsufficientMemoryError, RefusedError) as e:
        print(f"pigemma.bench: REFUSED: {e}", file=sys.stderr)
        return paths.EXIT_REFUSED
    except (MissingInputError, FileNotFoundError) as e:
        print(f"pigemma.bench: MISSING: {e}", file=sys.stderr)
        return paths.EXIT_MISSING
    except results.ResultSchemaError as e:  # a ValueError: must come first
        print(f"pigemma.bench: result schema error: {e}", file=sys.stderr)
        return paths.EXIT_ERROR
    except ValueError as e:
        print(f"pigemma.bench: usage: {e}", file=sys.stderr)
        return paths.EXIT_USAGE
    except BenchError as e:
        print(f"pigemma.bench: ERROR: {e}", file=sys.stderr)
        return paths.EXIT_ERROR
    except KeyboardInterrupt:
        print("pigemma.bench: interrupted", file=sys.stderr)
        return 130
    finally:
        if prev_term is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGTERM, prev_term)


if __name__ == "__main__":
    sys.exit(main())
