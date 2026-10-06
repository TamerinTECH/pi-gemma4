"""pigemma.controlled - host side of the controlled-memory harness (stage P5, brief sections 2-4).

One CELL = one method (A/B/C/D) x one memory budget x one workload set x one repeat, run in a
FRESH Docker container whose cgroup is the memory budget. The in-container half is
``pigemma.controlled_cell`` (it starts the FUSE throttle, the pinner and llama-server inside the
same cgroup, BEFORE the model loads); this module builds the exact ``docker run`` flags, guards
the desktop (bench lock, other model jobs, disk gates), records the host's background CPU load
during the cell, and plans the pin sets of methods C and D with page-exact accounting.

Desktop platform (documented deviation from the brief's cgroup v2, decided 2026-10-05 19:25):
Docker Desktop's WSL2 VM runs cgroup v1 without blk-throttling or PSI, so
  * memory budget = the container's v1 memory controller (--memory = memory.limit_in_bytes,
    --memory-swap = --memory -> memory.memsw.limit_in_bytes = budget: swap disabled), page cache
    (incl. the FUSE file's pages) and mlocked pages are charged to it;
  * storage rate = tools/fusethrottle (FUSE passthrough, O_DIRECT backing reads, serial-device
    token model at 91.6 MB/s = the Pi microSD) running inside the container;
  * PSI is unavailable: reclaim is read from memory.stat (pgmajfault, failcnt) and /proc/vmstat.

CLI (``python -m pigemma.controlled <cmd> --help``)::

    cell          run one cell (writes results/controlled/<cell_id>/...)
    series        run method/engine variants x repeats as randomized complete blocks (one CPU-lock hold)
    docker-cmd    print the exact docker command of a cell config (reproduction)
    make-ranges   plan the pin set of method C (ranked) or D (random, size-matched to C)
    pin-budget    budget arithmetic: what is left for pinned experts after measured needs
    verify-throttle  run the FUSE throttle verification container
    engines       list the engines the harness can run (built-in a7b94df + engines.json)
    engine-check  run an engine's `--version` / `--help` in the cell image (no model) and report
                  which canonical cell flags it lacks
    cpu-lock      status | acquire | release of results/.cpu.lock (desktop CPU discipline)

Engines (brief UPDATE 2): ``--engine ID`` selects the llama-server build of a cell. ``a7b94df`` is
built in (the pinned reference); every other id comes from results/comparisons/engines.json, which
the comparison stages X1a/X1b/X1c write (schema: ``ENGINE_SCHEMA_DOC`` below, tolerant reader).
A cell of another engine is labelled ``<engine>:<method>`` in the summaries and is compared with
its OWN unmodified parent (profile key ``parent``) as well as with the reference method A.

Exit codes: 0 ok; 1 error; 2 cell ran but FAILED (see config.json "failure"); 4 refused (bench or
CPU lock busy, another model job running, disk gate, background CPU above --max-bg-cpu-pct,
engine not runnable).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pigemma import paths

REPO = paths.REPO_ROOT
RESULTS = REPO / "results" / "controlled"
IMAGE = "pi-gemma4-build:fuse"
VOLUME = "pigemma-models"
MODEL_IN_VOLUME = "/models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf"
MODEL_SIZE = 12728497888
MODEL_SHA256 = "811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052"
MAP_PATH = "results/maps/map26_ud_q3_k_m.json"
LLAMA_SERVER = "/work/third_party/llama.cpp/build-linux-x86/bin/llama-server"
PINNER = "/work/tools/build-linux-x86/bin/pinner"
FUSETHROTTLE = "/work/tools/build-linux-x86/bin/fusethrottle"
PI_MICROSD_BPS = 91.6e6
# one hardware thread per physical core as the VM reports the topology (SMT siblings 0-1, 2-3, ...);
# Hyper-V may still move vCPUs between host cores - a VM cannot pin to host cores.
DEFAULT_CPUSET = "2,4,6,8"
GIB = 1024**3
EXIT_OK, EXIT_ERR, EXIT_CELL_FAIL, EXIT_REFUSED = 0, 1, 2, 4
WARMUP_TEXT = "Count from one to five, separated by commas."
REFERENCE_ENGINE = "a7b94df"
# other names of the SAME unpatched reference binary (X1a registered it as 'llama.cpp-a7b94df'):
# they resolve to the built-in engine, so their cells are plain variants A/B/C/D
REFERENCE_ALIASES = ("llama.cpp-a7b94df",)
ENGINES_FILE = REPO / "results" / "comparisons" / "engines.json"
CPU_LOCK = REPO / "results" / ".cpu.lock"
CPU_LOCK_STALE_S = 6 * 3600
SERIES_DIR = RESULTS / "series"

ENGINE_SCHEMA_DOC = """results/comparisons/engines.json - engine profiles for the controlled harness.

Accepted shapes: a list of profiles, {"engines": [profiles]}, {"engines": {id: profile}} or
{id: profile}. Profile keys (aliases in brackets):
  engine_id [id, name]                     required, unique; 'a7b94df' is reserved (built in)
  binary_path_in_container [server_bin, bin, binary, llama_server]
                                           required: the llama-server binary INSIDE the cell
                                           container; /work/... = this repo (bind mount); a
                                           repo-relative path is prefixed with /work/
  revision [commit, rev]                   source revision of the build
  build_flags [cmake_flags]                recorded only
  status                                   'runnable' (also: ok, built, tested, locally tested) or
                                           'blocked' / 'documentation-only' / 'not applicable';
                                           non-runnable engines are refused
  parent [parent_engine, baseline_engine]  engine id of the UNMODIFIED parent build this engine
                                           must be compared with (hotpin -> hotpin-parent,
                                           ik-prefetch-experts -> ik-plain)
  server_flag_overrides [flag_overrides]   {canonical key: [argv words] | [] | null}: respell or
                                           drop one canonical cell flag (keys: see
                                           pigemma.controlled_cell.SERVER_FLAGS; placeholders
                                           {model} {threads} {ctx} {batch} {ubatch} {ctk} {ctv}
                                           {flash_attn} {parallel} {cache_ram_mib}
                                           {ctx_checkpoints} {seed} {verbosity} {port})
  extra_args [server_args, extra_flags, flags, server_extra]
                                           extra llama-server words (list or one string), e.g. the
                                           engine's own pin/prefetch options; speculative-decoding
                                           flags are rejected
  env                                      {name: value} for the server process
  api                                      {"completion_path": "/completion", "extra_body": {...}}
  docker_volumes [volumes, mounts]         extra `-v` specs (e.g. "pigemma-ik-build:/opt/ik:ro")
  docker_image [image]                     another cell image (must contain fuse3 + python3)
  notes [note], license, method_label      recorded only
"""


# ----------------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------------


def default_config(*, method: str, budget_bytes: int, workload_set: str, repeat: int, cell_id: str,
                   conversations: Sequence[str] | None = None, n_predict_cap: int | None = None,
                   storage_mode: str = "fuse-throttled", rate_bps: float = PI_MICROSD_BPS,
                   ranges: str | None = None, threads: int = 4, ctx: int = 4096, batch: int = 2048,
                   ubatch: int = 512, server_warmup: bool = True, ctx_checkpoints: int = 32,
                   residency: str = "per_request", sample_interval_s: float = 0.5,
                   cpuset: str = DEFAULT_CPUSET, cpus: float = 4.0, op_latency_us: float = 0.0,
                   max_readahead: int = 131072, purpose: str = "measurement",
                   engine: Mapping[str, Any] | None = None,
                   server_extra: Sequence[str] | None = None) -> dict[str, Any]:
    if method not in ("A", "B", "C", "D"):
        raise ValueError(f"method must be A/B/C/D, got {method!r}")
    eng = dict(engine) if engine else builtin_engine()
    extra = [str(x) for x in (eng.get("extra") or [])] + [str(x) for x in (server_extra or [])]
    if method in ("C", "D") and not ranges:
        raise ValueError(f"method {method} needs --ranges (see make-ranges)")
    if workload_set not in ("calibration", "tuning", "heldout"):
        raise ValueError(f"unknown workload set {workload_set!r}")
    if purpose not in ("measurement", "pilot", "smoke"):
        raise ValueError(f"purpose must be measurement/pilot/smoke, got {purpose!r}")
    return {
        "schema": "pigemma-controlled-cell v1",
        "cell_id": cell_id,
        "purpose": purpose,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": "desktop Docker Desktop VM (WSL2 5.15, cgroup v1 memory controller) - documented "
                    "deviation from the brief's cgroup v2; storage rate emulated by FUSE",
        "method": method,
        "method_definition": {
            "A": "plain llama.cpp + Linux page cache; nothing between eviction and server start",
            "B": "pinner started with an EMPTY range list (--allow-empty): same helper, mapping and instrumentation, nothing locked",
            "C": "pinner locks non-expert ranges + calibration-ranked hot experts",
            "D": "pinner locks non-expert ranges + random experts, page-accounted total matched to C",
        }[method],
        "budget_bytes": int(budget_bytes),
        "budget_gib": round(budget_bytes / GIB, 3),
        "repeat": int(repeat),
        "workload": {"set": workload_set, "conversations": list(conversations) if conversations else None,
                     "n_predict_cap": n_predict_cap},
        "model": {"backing_path": MODEL_IN_VOLUME, "size": MODEL_SIZE, "sha256": MODEL_SHA256,
                  "repo": "unsloth/gemma-4-26B-A4B-it-GGUF@c099eb48e663fd284577b04978a94ffccb261841",
                  "map": f"/work/{MAP_PATH}"},
        "storage": {"mode": storage_mode, "rate_bps": rate_bps if storage_mode == "fuse-throttled" else 0.0,
                    "rate_source": "Pi 5 microSD, raw sequential O_DIRECT 4 MiB reads, measured 2026-10-05",
                    "device_model": "serial device, bandwidth only (no IOPS/latency/queue-depth model)",
                    "op_latency_us": op_latency_us, "max_readahead": max_readahead,
                    "mountpoint": "/mnt/pigemma-thr", "bin": FUSETHROTTLE, "trace": True},
        "pinner": {"bin": PINNER, "ranges": ranges},
        "engine": {k: eng.get(k) for k in ("id", "revision", "parent", "status", "build_flags", "notes", "license",
                                           "method_label", "api", "profile_source", "profile_sha256")},
        "variant": variant_label(eng.get("id"), method),
        "server": {"bin": eng["bin"], "flag_overrides": dict(eng.get("flag_overrides") or {}),
                   "env": dict(eng.get("env") or {}),
                   "threads": threads, "ctx": ctx, "batch": batch, "ubatch": ubatch,
                   "ctk": "f16", "ctv": "f16", "flash_attn": "auto", "parallel": 1, "cache_ram_mib": 0,
                   "ctx_checkpoints": ctx_checkpoints, "seed": 0, "warmup": server_warmup, "verbosity": 4, "extra": extra,
                   "speculative": "off (no -md / --spec-type; MTP and every draft method disabled)",
                   "notes": "-nr everywhere (repack would copy 4.1 GB of experts to anonymous memory on x86); "
                            "--cache-ram 0 (the default 8192 MiB host prompt cache would be anonymous memory "
                            "inside the budget and would reuse KV state across cache_prompt=false requests)"},
        "warmup_request": {"text": WARMUP_TEXT, "n_predict": 16},
        "request": {"temperature": 0.0, "seed": 0},
        "sampling": {"interval_s": sample_interval_s, "residency": residency},
        "docker": {"image": eng.get("image") or IMAGE, "cpuset_cpus": cpuset, "cpus": cpus, "volume": VOLUME,
                   "extra_volumes": list(eng.get("volumes") or [])},
        "out_dir": f"/work/results/controlled/{cell_id}",
        "host_headroom_bytes": 2 * GIB,
    }


def docker_argv(cfg: Mapping[str, Any], repo: Path = REPO) -> list[str]:
    """The exact docker command of a cell (identical flags for every method)."""
    b = int(cfg["budget_bytes"])
    d = cfg["docker"]
    name = "pigemma-cell-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", cfg["cell_id"]).lower()
    host_repo = str(repo).replace("\\", "/")
    return ["docker", "run", "--rm", "--name", name[:120],
            "--label", "pigemma.stage=P5-controlled", "--label", f"pigemma.cell={cfg['cell_id']}",
            f"--memory={b}b", f"--memory-swap={b}b", "--memory-swappiness=0",
            f"--cpus={d['cpus']}", f"--cpuset-cpus={d['cpuset_cpus']}",
            "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1:-1",
            "--cap-add", "SYS_ADMIN", "--device", "/dev/fuse", "--security-opt", "apparmor:unconfined",
            "--network", "none",
            "-v", f"{d['volume']}:/models:ro", "-v", f"{host_repo}:/work",
            *[w for v in (d.get("extra_volumes") or []) for w in ("-v", str(v))],
            "-e", "PYTHONPATH=/work", "-e", "PYTHONUNBUFFERED=1", "-w", "/work",
            d["image"], "python3", "-m", "pigemma.controlled_cell", "run",
            "--config", f"{cfg['out_dir']}/config.json"]


def new_cell_id(method: str, budget_bytes: int, workload_set: str, repeat: int, tag: str | None = None,
                engine_id: str | None = None) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    gib = f"{budget_bytes / GIB:g}g"
    parts = [ts, workload_set, method, gib, f"r{repeat}"]
    if engine_id and engine_id != REFERENCE_ENGINE:
        parts.insert(2, re.sub(r"[^a-zA-Z0-9_.-]", "-", engine_id))
    if tag:
        parts.append(re.sub(r"[^a-zA-Z0-9_.-]", "-", tag))
    return "-".join(parts)


# ----------------------------------------------------------------------------------------------
# engines (results/comparisons/engines.json)
# ----------------------------------------------------------------------------------------------

_ENGINE_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("engine_id", "id", "name"),
    "bin": ("binary_path_in_container", "server_bin", "llama_server", "bin", "binary"),
    "revision": ("revision", "commit", "rev"),
    "build_flags": ("build_flags", "cmake_flags"),
    "notes": ("notes", "note"),
    "status": ("status",),
    "parent": ("parent", "parent_engine_id", "parent_engine", "baseline_engine"),
    "flag_overrides": ("server_flag_overrides", "flag_overrides"),
    "extra": ("extra_args", "server_args", "extra_flags", "flags", "server_extra"),
    "env": ("env",),
    "api": ("api",),
    "volumes": ("docker_volumes", "volumes", "mounts"),
    "image": ("docker_image", "image"),
    "license": ("license",),
    "method_label": ("method_label",),
}
_RUNNABLE = ("runnable", "ok", "built", "tested", "locally tested", "locally-tested")
_NOT_RUNNABLE = ("blocked", "documentation", "not applicable", "not-applicable", "failed", "unsupported")


class EngineError(ValueError):
    """Unknown, malformed or non-runnable engine profile."""


def variant_label(engine_id: str | None, method: str) -> str:
    """'A'..'D' for the reference engine, '<engine>:<method>' for every other build."""
    return method if engine_id in (None, "", REFERENCE_ENGINE) else f"{engine_id}:{method}"


def builtin_engine() -> dict[str, Any]:
    return {"id": REFERENCE_ENGINE, "bin": LLAMA_SERVER, "revision": "a7b94df (b11398), unpatched",
            "parent": None, "status": "runnable", "flag_overrides": {}, "extra": [], "env": {}, "api": {},
            "volumes": [], "image": None, "notes": "pinned reference build, third_party/llama.cpp, static CPU x86",
            "profile_source": "built-in (pigemma.controlled.builtin_engine)", "profile_sha256": None}


def normalize_engine(raw: Mapping[str, Any], fallback_id: str | None = None) -> dict[str, Any]:
    import shlex
    out: dict[str, Any] = {}
    for key, aliases in _ENGINE_ALIASES.items():
        for a in aliases:
            if a in raw and raw[a] is not None:
                out[key] = raw[a]
                break
    # X1a's convention: {"harness": {"extra_server_args": [...], "server_flag_overrides": {...}}}
    hz = raw.get("harness") if isinstance(raw.get("harness"), Mapping) else {}
    if not out.get("extra") and hz.get("extra_server_args"):
        out["extra"] = hz["extra_server_args"]
    if not out.get("flag_overrides"):
        for k in ("server_flag_overrides", "flag_overrides"):
            if hz.get(k):
                out["flag_overrides"] = hz[k]
                break
    out["methods"] = list(hz["methods"]) if isinstance(hz.get("methods"), list) else None
    out.setdefault("id", fallback_id)
    if not out.get("id"):
        raise EngineError(f"engine profile without engine_id: {dict(raw)!r:.200}")
    out["id"] = str(out["id"])
    if not out.get("bin"):
        raise EngineError(f"engine {out['id']}: no binary_path_in_container")
    b = str(out["bin"]).replace("\\", "/")
    out["bin"] = b if b.startswith("/") else "/work/" + b.lstrip("./")
    ex = out.get("extra") or []
    out["extra"] = shlex.split(ex) if isinstance(ex, str) else [str(x) for x in ex]
    fo = out.get("flag_overrides") or {}
    if not isinstance(fo, Mapping):
        raise EngineError(f"engine {out['id']}: server_flag_overrides must be an object")
    out["flag_overrides"] = {str(k): (None if v is None else [str(w) for w in ([v] if isinstance(v, str) else v)])
                             for k, v in fo.items()}
    vols = out.get("volumes") or []
    out["volumes"] = [vols] if isinstance(vols, str) else [str(v) for v in vols]
    for k in ("env", "api"):
        if not isinstance(out.get(k) or {}, Mapping):
            raise EngineError(f"engine {out['id']}: {k} must be an object")
        out[k] = dict(out.get(k) or {})
    for k in ("revision", "parent", "status", "build_flags", "notes", "image", "license", "method_label"):
        out.setdefault(k, None)
    return out


def load_engines(path: str | os.PathLike[str] | None = None) -> dict[str, dict[str, Any]]:
    """{id: normalized profile}; always contains the built-in reference engine."""
    engines = {REFERENCE_ENGINE: builtin_engine()}
    p = Path(path) if path else ENGINES_FILE
    if not p.exists():
        return engines
    raw = p.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    sha = hashlib.sha256(raw).hexdigest()
    items: list[tuple[str | None, Any]] = []
    body = doc.get("engines", doc) if isinstance(doc, Mapping) else doc
    if isinstance(body, list):
        items = [(None, x) for x in body]
    elif isinstance(body, Mapping):
        items = [(str(k), v) for k, v in body.items() if isinstance(v, Mapping)]
    else:
        raise EngineError(f"{p}: unrecognised engines file")
    for fid, rawp in items:
        e = normalize_engine(rawp, fid)
        if e["id"] == REFERENCE_ENGINE:
            continue  # reserved: the built-in definition wins (recorded as such)
        if e["id"] in REFERENCE_ALIASES or (e["bin"] == LLAMA_SERVER and not e["extra"] and not e["flag_overrides"]
                                            and not e["env"] and not e.get("parent")):
            e["alias_of"] = REFERENCE_ENGINE  # the same unpatched binary under another name
        if e["id"] in engines:
            raise EngineError(f"{p}: duplicate engine id {e['id']!r}")
        e["profile_source"] = str(p)
        e["profile_sha256"] = sha
        engines[e["id"]] = e
    return engines


def engine_runnable(e: Mapping[str, Any]) -> tuple[bool, str]:
    st = str(e.get("status") or "").strip().lower()
    if any(w in st for w in _NOT_RUNNABLE):
        return False, f"status {e.get('status')!r}"
    if st and not any(st == w or st.startswith(w) for w in _RUNNABLE):
        return False, f"status {e.get('status')!r} is not one of {_RUNNABLE}"
    b = str(e["bin"])
    if b.startswith("/work/") and not (REPO / b[len("/work/"):]).exists():
        return False, f"binary {b} does not exist in the repo"
    return True, "no status given (accepted)" if not st else st


def resolve_engine(engine_id: str | None, path: str | os.PathLike[str] | None = None,
                   server_bin: str | None = None, allow_not_runnable: bool = False) -> dict[str, Any]:
    eid = engine_id or REFERENCE_ENGINE
    engines = load_engines(path)
    if eid not in engines:
        raise EngineError(f"unknown engine {eid!r}; known: {sorted(engines)} (file: {path or ENGINES_FILE})")
    e = dict(engines[eid])
    if e.get("alias_of") == REFERENCE_ENGINE:
        ref = dict(engines[REFERENCE_ENGINE])
        ref["notes"] = f"requested as {eid!r}, an alias of the built-in reference engine; {ref.get('notes') or ''}".strip()
        e = ref
    if server_bin:
        e["bin"] = server_bin if server_bin.startswith("/") else "/work/" + Path(server_bin).as_posix()
        e["notes"] = f"--server-bin override; {e.get('notes') or ''}".strip()
    ok, why = engine_runnable(e)
    if not ok and not allow_not_runnable:
        raise EngineError(f"engine {eid!r} is not runnable: {why}")
    return e


def engine_allows(e: Mapping[str, Any], method: str) -> tuple[bool, str]:
    """A profile may restrict the cell methods of its engine (harness.methods), e.g. an engine with
    its own in-process pinning runs as method A only: the external pinner would pin on top of it."""
    ms = e.get("methods")
    if ms and method not in ms:
        return False, f"engine {e['id']!r} is registered for methods {ms}, not {method}"
    return True, ""


def engine_check(e: Mapping[str, Any], timeout: float = 180) -> dict[str, Any]:
    """Run `<bin> --version` and `<bin> --help` in the cell image (no model, no FUSE) and report
    which canonical cell flags the build's help text does not mention."""
    from pigemma import controlled_cell as cc
    base = ["docker", "run", "--rm", "--network", "none", "--cpus=1",
            "-v", f"{str(REPO).replace(chr(92), '/')}:/work",
            *[w for v in (e.get("volumes") or []) for w in ("-v", str(v))],
            *[w for k, v in (e.get("env") or {}).items() for w in ("-e", f"{k}={v}")],
            e.get("image") or IMAGE]
    rc_v, ver = _run(base + [e["bin"], "--version"], timeout=timeout)
    rc_h, hlp = _run(base + [e["bin"], "--help"], timeout=timeout)
    over = e.get("flag_overrides") or {}
    rows = []
    for key, tmpl in cc.SERVER_FLAGS:
        words = over[key] if key in over else tmpl
        flags = [w for w in (words or []) if str(w).startswith("-") and not str(w).lstrip("-").isdigit()]
        missing = [f for f in flags if not re.search(r"(?<![\w-])" + re.escape(f) + r"(?![\w-])", hlp)]
        rows.append({"key": key, "argv": list(words or []), "dropped": not words, "missing_in_help": missing})
    spec = [f for f in (e.get("extra") or []) if f in cc.SPECULATIVE_FLAGS]
    extra_missing = [f for f in (e.get("extra") or []) if f.startswith("-") and not f.lstrip("-").replace(".", "").isdigit()
                     and not re.search(r"(?<![\w-])" + re.escape(f) + r"(?![\w-])", hlp)]
    missing_keys = [r["key"] for r in rows if r["missing_in_help"]]
    return {"engine": e["id"], "bin": e["bin"], "image": e.get("image") or IMAGE,
            "version_exit": rc_v, "version_text": ver.strip()[-600:], "help_exit": rc_h, "help_chars": len(hlp),
            "flags": rows, "canonical_flags_missing_in_help": missing_keys,
            "extra_args_missing_in_help": extra_missing, "speculative_flags_in_extra": spec,
            "binary_runs": rc_v == 0 or rc_h == 0,
            "ok": (rc_h == 0 or rc_v == 0) and not missing_keys and not extra_missing and not spec,
            "hint": "add server_flag_overrides {key: [...] or null} to the profile for every missing canonical "
                    "flag; a missing flag would make llama-server exit at start (the cell then fails, it "
                    "never silently drops a setting)" if missing_keys else None}


# ----------------------------------------------------------------------------------------------
# desktop CPU lock (results/.cpu.lock; CLAUDE.md "coordination rules", same file format as
# pc/x1a/cpu_lock.sh: one JSON object {owner, purpose, start}, created atomically, stale after 6 h)
# ----------------------------------------------------------------------------------------------


class CpuLockBusy(RuntimeError):
    """results/.cpu.lock is held by another owner (or is stale and must be looked at by a human)."""


def cpu_lock_status(path: Path | None = None) -> dict[str, Any]:
    p = path or CPU_LOCK
    try:
        raw = p.read_text("utf-8", "replace")
        age = time.time() - p.stat().st_mtime
    except OSError:
        return {"held": False, "path": str(p)}
    try:
        doc = json.loads(raw)
    except ValueError:
        doc = {}
    return {"held": True, "path": str(p), "owner": doc.get("owner"), "purpose": doc.get("purpose"),
            "start": doc.get("start") or doc.get("start_utc"), "age_s": round(age, 1),
            "stale": age > CPU_LOCK_STALE_S, "raw": raw.strip()[:400]}


def cpu_lock_acquire(owner: str, purpose: str, wait_s: float = 0.0, poll_s: float = 60.0,
                     path: Path | None = None) -> dict[str, Any]:
    """Create the lock exclusively (O_EXCL). Waits up to wait_s, polling every poll_s. A stale lock
    (> 6 h) is never removed automatically: CpuLockBusy says so."""
    p = path or CPU_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            st = cpu_lock_status(p)
            if st.get("stale"):
                raise CpuLockBusy(f"STALE CPU lock (> 6 h), not removed automatically: {st.get('raw')}")
            if time.time() - t0 >= wait_s:
                raise CpuLockBusy(f"CPU lock held by {st.get('owner')!r} ({st.get('purpose')}) since {st.get('start')}")
            time.sleep(min(poll_s, max(0.5, wait_s - (time.time() - t0))))
            continue
        doc = {"owner": owner, "purpose": purpose,
               "start": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "pid_note": f"python -m pigemma.controlled, pid {os.getpid()}"}
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(doc) + "\n")
        return {"acquired": True, "waited_s": round(time.time() - t0, 1), **doc}


def cpu_lock_release(owner: str, path: Path | None = None) -> bool:
    """Remove the lock only when it is ours. Returns False when there is no lock or it is foreign."""
    p = path or CPU_LOCK
    st = cpu_lock_status(p)
    if not st["held"] or st.get("owner") != owner:
        return False
    try:
        p.unlink()
    except OSError:
        return False
    return True


# ----------------------------------------------------------------------------------------------
# desktop guards
# ----------------------------------------------------------------------------------------------


def _run(argv: Sequence[str], timeout: float = 60) -> tuple[int, str]:
    try:
        r = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 127, repr(e)


def other_model_jobs(own_name: str | None = None) -> list[str]:
    """Running model jobs that would invalidate a timing: our project's containers and native llama."""
    busy: list[str] = []
    rc, out = _run(["docker", "ps", "--format", "{{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}"])
    if rc == 0:
        for line in out.splitlines():
            f = line.split("\t")
            if len(f) >= 3 and (f[1].startswith("pi-gemma4") or f[2].startswith("pigemma")) and f[2] != own_name:
                busy.append(f"container {f[2]} ({f[1]}, {f[3] if len(f) > 3 else ''})")
    if os.name == "nt":
        rc, out = _run(["tasklist", "/FO", "CSV", "/NH"])
        for line in out.splitlines():
            m = re.match(r'"([^"]+)","(\d+)"', line)
            if m and m.group(1).lower().startswith(("llama-", "pigemma")):
                busy.append(f"process {m.group(1)} pid {m.group(2)}")
    return busy


def disk_free() -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for d in ("C:\\", "D:\\", "F:\\") if os.name == "nt" else ("/",):
        try:
            out[d] = round(shutil.disk_usage(d).free / 1e9, 1)
        except OSError:
            out[d] = None
    return out


class HostCpu(threading.Thread):
    """Samples total host CPU % (and the WSL VM process) every second into host_cpu.csv."""

    def __init__(self, path: Path) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.stop_ev = threading.Event()
        self.samples: list[tuple[float, float, float | None]] = []
        self.t0_epoch: float | None = None

    def run(self) -> None:
        try:
            import psutil  # type: ignore
        except ImportError:
            return
        vm = None
        for p in psutil.process_iter(["name"]):
            if (p.info.get("name") or "").lower().startswith("vmmem"):
                vm = p
                break
        ncpu = psutil.cpu_count() or 1
        psutil.cpu_percent(None)
        if vm is not None:
            try:
                vm.cpu_percent(None)
            except Exception:  # noqa: BLE001
                vm = None
        t0 = time.time()
        self.t0_epoch = t0
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(f"# t0_epoch={t0:.3f} (Windows host clock; t_s is relative to it)\n")
            fh.write("t_s,host_cpu_pct,vmmem_cpu_pct_of_host\n")
            while not self.stop_ev.wait(1.0):
                tot = psutil.cpu_percent(None)
                v = None
                if vm is not None:
                    try:
                        v = vm.cpu_percent(None) / ncpu
                    except Exception:  # noqa: BLE001
                        vm = None
                self.samples.append((time.time() - t0, tot, v))
                fh.write(f"{time.time() - t0:.1f},{tot:.1f},{'' if v is None else f'{v:.1f}'}\n")
                fh.flush()

    def summary(self) -> dict[str, Any]:
        xs = [s[1] for s in self.samples]
        vs = [s[2] for s in self.samples if s[2] is not None]
        if not xs:
            return {"n": 0}
        xs_sorted = sorted(xs)
        return {"n": len(xs), "t0_epoch": self.t0_epoch, "host_cpu_pct_mean": round(statistics.mean(xs), 1),
                "host_cpu_pct_median": round(statistics.median(xs), 1),
                "host_cpu_pct_p95": round(xs_sorted[int(0.95 * (len(xs) - 1))], 1),
                "vmmem_cpu_pct_mean": round(statistics.mean(vs), 1) if vs else None,
                "note": "host total includes this cell's own container (<= 4 of 12 logical CPUs = 33 %)"}


def idle_probe(seconds: float = 10.0) -> dict[str, Any]:
    try:
        import psutil  # type: ignore
    except ImportError:
        return {"error": "psutil missing"}
    xs = [psutil.cpu_percent(interval=1.0) for _ in range(int(seconds))]
    return {"seconds": int(seconds), "host_cpu_pct_mean": round(statistics.mean(xs), 1),
            "host_cpu_pct_max": round(max(xs), 1)}


# ----------------------------------------------------------------------------------------------
# cell
# ----------------------------------------------------------------------------------------------


def run_cell(cfg: dict[str, Any], *, use_lock: bool = True, ignore_other_jobs: bool = False,
             max_bg_cpu_pct: float | None = None, timeout_s: float = 4 * 3600, idle_s: float = 10.0,
             min_disk_gb: float = 10.0, cpu_lock: str = "acquire", cpu_lock_owner: str | None = None,
             cpu_lock_wait_s: float = 0.0, bg_wait_s: float = 0.0) -> int:
    """cpu_lock: 'acquire' (take results/.cpu.lock for this cell, wait up to cpu_lock_wait_s),
    'inherit' (the caller already holds it as cpu_lock_owner - a series or a stage) or 'ignore'
    (functional smoke tests only; recorded)."""
    out = RESULTS / cfg["cell_id"]
    argv = docker_argv(cfg)
    name = argv[argv.index("--name") + 1]
    host: dict[str, Any] = {"docker_argv": argv, "host": paths.host_fingerprint() if hasattr(paths, "host_fingerprint") else None}
    host["disk_free_GB_before"] = disk_free()
    for d, gb in host["disk_free_GB_before"].items():
        if gb is not None and d in ("C:\\", "D:\\") and gb < min_disk_gb:
            print(f"REFUSED: {d} has {gb} GB free < {min_disk_gb} GB", file=sys.stderr)
            return EXIT_REFUSED
    # desktop CPU lock first: while another stage holds it (builds, dense numerics, model runs) a
    # cell's timings are not valid and its 4 busy CPUs would disturb that stage
    own_cpu_lock = None
    if cpu_lock == "acquire":
        owner = cpu_lock_owner or f"P5-cell {cfg['cell_id']}"
        try:
            host["cpu_lock"] = cpu_lock_acquire(owner, f"controlled cell {cfg.get('variant', cfg['method'])} "
                                                       f"{cfg['budget_gib']} GiB {cfg['workload']['set']} ({cfg['purpose']})",
                                                wait_s=cpu_lock_wait_s)
            own_cpu_lock = owner
        except CpuLockBusy as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return EXIT_REFUSED
    elif cpu_lock == "inherit":
        st = cpu_lock_status()
        if not st["held"] or st.get("owner") != cpu_lock_owner:
            print(f"REFUSED: --cpu-lock-owner {cpu_lock_owner!r} does not hold results/.cpu.lock "
                  f"(holder: {st.get('owner')!r})", file=sys.stderr)
            return EXIT_REFUSED
        host["cpu_lock"] = {"inherited": True, **{k: st.get(k) for k in ("owner", "purpose", "start")}}
    else:
        host["cpu_lock"] = {"ignored": True, "status_at_start": cpu_lock_status(),
                            "note": "functional run only - timings of this cell are not valid measurements"}
    try:
        return _run_cell_locked(cfg, out, argv, name, host, use_lock=use_lock, ignore_other_jobs=ignore_other_jobs,
                                max_bg_cpu_pct=max_bg_cpu_pct, timeout_s=timeout_s, idle_s=idle_s, bg_wait_s=bg_wait_s)
    finally:
        if own_cpu_lock is not None:
            cpu_lock_release(own_cpu_lock)


def _run_cell_locked(cfg: dict[str, Any], out: Path, argv: list[str], name: str, host: dict[str, Any], *,
                     use_lock: bool, ignore_other_jobs: bool, max_bg_cpu_pct: float | None, timeout_s: float,
                     idle_s: float, bg_wait_s: float) -> int:
    busy = other_model_jobs(own_name=name)
    host["other_model_jobs_at_start"] = busy
    if busy and not ignore_other_jobs:
        print("REFUSED: other model jobs are running: " + "; ".join(busy), file=sys.stderr)
        return EXIT_REFUSED
    t_wait0 = time.time()
    probes = []
    while True:
        host["idle_probe_before"] = idle_probe(idle_s) if idle_s > 0 else None
        bg = (host["idle_probe_before"] or {}).get("host_cpu_pct_mean")
        probes.append(bg)
        if max_bg_cpu_pct is None or bg is None or bg <= max_bg_cpu_pct:
            break
        if time.time() - t_wait0 >= bg_wait_s:
            print(f"REFUSED: background CPU {bg} % > {max_bg_cpu_pct} % (waited {time.time() - t_wait0:.0f} s)",
                  file=sys.stderr)
            return EXIT_REFUSED
        time.sleep(30)
    host["idle_probe_history_pct"] = probes
    host["max_bg_cpu_pct_gate"] = max_bg_cpu_pct
    out.mkdir(parents=True, exist_ok=True)
    rc_img, img = _run(["docker", "image", "inspect", cfg["docker"]["image"], "--format", "{{.Id}}"])
    host["image_id"] = img.strip() if rc_img == 0 else None
    cfg = dict(cfg)
    cfg["host"] = host
    (out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    lock_cm = paths.bench_lock(2.0, cmd=" ".join(argv)) if use_lock else None
    cpu = HostCpu(out / "host_cpu.csv")
    rc = EXIT_ERR
    t0 = time.time()
    try:
        if lock_cm is not None:
            lock_cm.__enter__()
        cpu.start()
        with open(out / "runner.log", "w", encoding="utf-8") as log:
            print("$ " + " ".join(argv), file=log, flush=True)
            p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _run(["docker", "stop", "-t", "30", name], timeout=120)  # our own container only
                rc = p.wait(timeout=120)
                host["timeout"] = True
    except paths.BenchLockError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    finally:
        cpu.stop_ev.set()
        cpu.join(timeout=5)
        if lock_cm is not None:
            try:
                lock_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
    doc = json.loads((out / "config.json").read_text("utf-8"))
    doc.setdefault("host", {}).update({
        "docker_exit_code": rc, "wall_s": round(time.time() - t0, 1),
        "host_cpu_during_cell": cpu.summary(), "host_cpu_t0_epoch": cpu.t0_epoch,
        "disk_free_GB_after": disk_free(),
        "other_model_jobs_at_end": other_model_jobs(own_name=name)})
    (out / "config.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    status = doc.get("status")
    print(json.dumps({"cell_id": cfg["cell_id"], "status": status, "failure": doc.get("failure"),
                      "docker_exit_code": rc, "out": str(out)}))
    return EXIT_OK if status == "ok" and rc == 0 else EXIT_CELL_FAIL


# ----------------------------------------------------------------------------------------------
# pin-set planning (methods C and D), page exact
# ----------------------------------------------------------------------------------------------


def _page_runs(off: int, n: int, page: int, size: int) -> tuple[int, int]:
    a = off // page
    b = min((off + n + page - 1) // page, (size + page - 1) // page)
    return a, b


class PagePlanner:
    """Unique-page accounting over one single-file expert map, with a bitmap of covered pages."""

    def __init__(self, em: Any, page: int) -> None:
        files = em.files
        if len(files) != 1:
            raise ValueError("PagePlanner supports single-file models (the primary comparison file)")
        self.em = em
        self.page = page
        self.size = int(files[0]["size"])
        self.n_pages = (self.size + page - 1) // page
        self.bm = bytearray(self.n_pages)
        self.count = 0
        self.nonexpert_runs = [_page_runs(int(r["offset"]), int(r["length"]), page, self.size)
                               for r in em["nonexpert_ranges"]]
        self.expert_runs: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for x in em["experts"]:
            self.expert_runs[(int(x["layer"]), int(x["expert"]))] = [
                _page_runs(int(r["offset"]), int(r["n_bytes"]), page, self.size) for r in x["ranges"]]

    @staticmethod
    def merged(runs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        """Sort and merge overlapping/touching page runs (an expert's scale page can be the
        first page of its down slice, so runs of one expert may overlap)."""
        out: list[list[int]] = []
        for a, b in sorted(runs):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return [(a, b) for a, b in out]

    def new_pages(self, runs: Iterable[tuple[int, int]]) -> int:
        return sum(self.bm[a:b].count(0) for a, b in self.merged(runs))

    def add(self, runs: Iterable[tuple[int, int]]) -> int:
        added = 0
        for a, b in runs:
            seg = self.bm[a:b]
            z = seg.count(0)
            if z:
                self.bm[a:b] = b"\x01" * (b - a)
                added += z
        self.count += added
        return added


def load_ranking(path: str | os.PathLike[str], n_layer: int, n_expert: int) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Ranked (layer, expert) list from {"ranking": [...]}, {"counts": [[...]]}, or a bare list.

    Refuses a ranking whose metadata names a source set other than 'calibration' (held-out leak)."""
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    meta = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(doc, Mapping):
        for k in ("set", "source_set", "workload_set"):
            if k in doc and str(doc[k]) != "calibration":
                raise ValueError(f"ranking {path} was built from set {doc[k]!r}; only 'calibration' may rank experts")
        meta.update({k: doc[k] for k in ("set", "source_set", "workload_set", "source", "created") if k in doc})
    items: list[tuple[int, int]]
    if isinstance(doc, Mapping) and "ranking" in doc:
        items = []
        for it in doc["ranking"]:
            if isinstance(it, Mapping):
                items.append((int(it["layer"]), int(it["expert"])))
            else:
                items.append((int(it[0]), int(it[1])))
    elif isinstance(doc, Mapping) and "counts" in doc:
        c = doc["counts"]
        if len(c) != n_layer or any(len(r) != n_expert for r in c):
            raise ValueError(f"counts must be {n_layer} x {n_expert}")
        flat = [(-float(c[l][e]), l, e) for l in range(n_layer) for e in range(n_expert)]
        flat.sort()
        items = [(l, e) for _, l, e in flat]
    elif isinstance(doc, list):
        items = [(int(a), int(b)) for a, b in doc]
    else:
        raise ValueError(f"unrecognised ranking format in {path}")
    seen: set[tuple[int, int]] = set()
    out = []
    for k in items:
        if not (0 <= k[0] < n_layer and 0 <= k[1] < n_expert):
            raise ValueError(f"ranking entry {k} out of range")
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out, meta


def plan_pins(em: Any, page: int, method: str, *, ranking: Sequence[tuple[int, int]] | None = None,
              expert_pin_bytes: int | None = None, target_total_pages: int | None = None,
              seed: int = 0) -> dict[str, Any]:
    """Select experts for C (ranked prefix within expert_pin_bytes of NEW unique pages) or D
    (uniform random until the total unique pages - non-expert included - reach C's total)."""
    pp = PagePlanner(em, page)
    nonexpert_pages = pp.add(pp.nonexpert_runs)
    chosen: list[tuple[int, int]] = []
    if method == "C":
        if ranking is None or expert_pin_bytes is None:
            raise ValueError("method C needs a ranking and expert_pin_bytes")
        budget_pages = expert_pin_bytes // page
        used = 0
        for k in ranking:
            runs = pp.expert_runs[k]
            need = pp.new_pages(runs)
            if used + need > budget_pages:
                break  # strict prefix of the ranking
            used += pp.add(runs)
            chosen.append(k)
    elif method == "D":
        if target_total_pages is None:
            raise ValueError("method D needs target_total_pages (C's total unique pages)")
        cand = sorted(pp.expert_runs)
        rng = random.Random(seed)
        rng.shuffle(cand)
        for k in cand:
            runs = pp.expert_runs[k]
            need = pp.new_pages(runs)
            if pp.count + need > target_total_pages:
                continue
            pp.add(runs)
            chosen.append(k)
    else:
        raise ValueError("plan_pins: method must be C or D")
    exact_total = em.pages_for(chosen, page, with_nonexpert=True)
    per_layer = [0] * int(em["n_layer"])
    for l, _ in chosen:
        per_layer[l] += 1
    return {
        "method": method, "page": page, "n_experts": len(chosen), "experts": [list(k) for k in chosen],
        "experts_per_layer": per_layer,
        "nonexpert_pages": nonexpert_pages, "total_unique_pages": pp.count,
        "total_unique_pages_exact_check": exact_total, "check_ok": exact_total == pp.count,
        "total_bytes": pp.count * page, "expert_pages": pp.count - nonexpert_pages,
        "nominal_expert_bytes": sum(int(em.expert(l, e)["bytes"]) for l, e in chosen),
        "nominal_nonexpert_bytes": sum(int(r["length"]) for r in em["nonexpert_ranges"]),
        # what whole-page locking costs beyond the tensor bytes themselves (pages shared by two
        # selected ranges are counted once, so this is the net overhead of the union)
        "alignment_overhead_bytes": pp.count * page - sum(int(r["length"]) for r in em["nonexpert_ranges"])
        - sum(int(em.expert(l, e)["bytes"]) for l, e in chosen),
        "expert_pin_bytes_budget": expert_pin_bytes, "target_total_pages": target_total_pages,
        "diff_pages_vs_target": (pp.count - target_total_pages) if target_total_pages is not None else None,
        "seed": seed if method == "D" else None,
    }


def write_ranges(em: Any, plan: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """pinner ranges file: non-expert runs (kind=nonexpert) then expert runs (kind=expert)."""
    page = int(plan["page"])
    f0 = em.files[0]
    experts = [tuple(k) for k in plan["experts"]]
    ne = em.ranges_for([], page, with_nonexpert=True)
    ex = em.ranges_for(experts, page, with_nonexpert=False) if experts else []
    lines = [f"# pigemma-ranges v1 gguf={f0['name']} size={f0['size']} page={page} budget={plan['total_bytes']}",
             f"# P5 controlled harness, method {plan['method']}: {plan['n_experts']} experts, "
             f"{plan['total_unique_pages']} unique pages (non-expert {plan['nonexpert_pages']}); "
             f"overlapping boundary pages are counted once by the pinner"]
    lines += [f"{o} {n} # kind=nonexpert" for _, o, n in ne]
    lines += [f"{o} {n} # kind=expert" for _, o, n in ex]
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    # union of what the pinner will lock, in pages (must equal the plan)
    bm = bytearray((int(f0["size"]) + page - 1) // page)
    for _, o, n in ne + ex:
        bm[o // page:(o + n + page - 1) // page] = b"\x01" * (((o + n + page - 1) // page) - o // page)
    union = bm.count(1)
    return {"ranges_file": str(path), "n_lines": len(ne) + len(ex), "union_pages": union,
            "union_matches_plan": union == plan["total_unique_pages"]}


# ----------------------------------------------------------------------------------------------
# pin budget arithmetic
# ----------------------------------------------------------------------------------------------


def _read_samples(path: Path) -> list[dict[str, str]]:
    import csv
    with open(path, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(l for l in fh if not l.startswith("#"))]


def measured_needs_from_cell(cell_dir: Path) -> dict[str, Any]:
    """Anonymous / kernel / helper memory measured in a method-A (or B) cell."""
    doc = json.loads((cell_dir / "config.json").read_text("utf-8"))
    rows = _read_samples(cell_dir / "samples.csv")

    def mx(col: str, phase_prefix: str | None = None) -> int | None:
        vals = [int(float(r[col])) for r in rows if r.get(col) not in (None, "") and
                (phase_prefix is None or r["phase"].startswith(phase_prefix))]
        return max(vals) if vals else None
    anon = mx("cg_anon")
    kern = mx("cg_kernel")
    srv_anon_kb = mx("server_anon_kb")
    fuse_rss_kb = mx("fuse_rss_kb")
    return {"source_cell": doc["cell_id"], "source_method": doc["method"], "source_budget_bytes": doc["budget_bytes"],
            "cg_anon_peak_bytes": anon, "cg_kernel_peak_bytes": kern,
            "server_anon_peak_bytes": srv_anon_kb * 1024 if srv_anon_kb else None,
            "fuse_rss_peak_bytes": fuse_rss_kb * 1024 if fuse_rss_kb else None,
            "label": "measured (sampled every interval; true peaks between samples can be higher)"}


def pin_budget(budget_bytes: int, *, anon_bytes: int, kernel_bytes: int, nonexpert_bytes: int,
               miss_reserve_bytes: int, safety_bytes: int) -> dict[str, Any]:
    """Bytes left for pinned EXPERT pages. Every term is explicit; never 'all remaining RAM'."""
    left = budget_bytes - anon_bytes - kernel_bytes - nonexpert_bytes - miss_reserve_bytes - safety_bytes
    return {"budget_bytes": budget_bytes, "anon_bytes": anon_bytes, "kernel_bytes": kernel_bytes,
            "nonexpert_pinned_bytes": nonexpert_bytes, "miss_page_cache_reserve_bytes": miss_reserve_bytes,
            "safety_bytes": safety_bytes, "expert_pin_bytes": max(0, left), "feasible": left > 0}


# ----------------------------------------------------------------------------------------------
# series: variants x repeats as randomized complete blocks (brief section 4: "randomize or
# alternate baseline/treatment order and repeat finalist comparisons at least three times")
# ----------------------------------------------------------------------------------------------


def parse_variant(text: str) -> tuple[str, str]:
    """'A' -> (reference engine, 'A'); 'ik-plain:A' -> ('ik-plain', 'A')."""
    eng, sep, meth = text.strip().rpartition(":")
    if not sep:
        eng, meth = REFERENCE_ENGINE, text.strip()
    if meth not in ("A", "B", "C", "D"):
        raise ValueError(f"variant {text!r}: method must be A/B/C/D")
    return eng or REFERENCE_ENGINE, meth


def series_order(variants: Sequence[str], repeats: int, seed: int) -> list[dict[str, Any]]:
    """Randomized complete blocks: block r (= repeat r) holds every variant exactly once in an
    order drawn from random.Random(seed) - reproducible, and no variant keeps a fixed position."""
    if len(set(variants)) != len(variants):
        raise ValueError("duplicate variants")
    rng = random.Random(seed)
    order = []
    for r in range(repeats):
        block = list(variants)
        rng.shuffle(block)
        for pos, v in enumerate(block):
            order.append({"index": len(order), "repeat": r, "position_in_block": pos, "variant": v})
    return order


def run_series(a: argparse.Namespace) -> int:
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    ranges = {"C": a.ranges_c, "D": a.ranges_d}
    budget = _budget(a)
    sid = a.series_id or time.strftime("%Y%m%d-%H%M%S") + f"-{a.workload}-{budget / GIB:g}g"
    engines = {}
    for v in variants:
        eid, meth = parse_variant(v)
        engines[v] = resolve_engine(eid, a.engines_file)
        ok_m, why_m = engine_allows(engines[v], meth)
        if not ok_m:
            print(f"series: {why_m}", file=sys.stderr)
            return EXIT_ERR
        if meth in ("C", "D") and not ranges[meth]:
            print(f"series: variant {v} needs --ranges-{meth.lower()}", file=sys.stderr)
            return EXIT_ERR
    order = series_order(variants, a.repeats, a.seed)
    for o in order:
        eid, meth = parse_variant(o["variant"])
        o.update({"engine": eid, "method": meth,
                  "cell_id": f"{sid}-{re.sub(r'[^a-zA-Z0-9_.-]', '-', o['variant'])}-r{o['repeat']}"})
    plan = {"schema": "pigemma-controlled-series v1", "series_id": sid,
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "design": "randomized complete blocks (block = repeat; every variant once per block)",
            "seed": a.seed, "variants": variants, "repeats": a.repeats, "budget_bytes": budget,
            "workload": a.workload, "purpose": a.purpose, "storage": a.storage, "rate_bps": a.rate_bps,
            "max_bg_cpu_pct": a.max_bg_cpu_pct, "ranges": ranges, "order": order, "status": "planned"}
    if a.dry_run:
        print(json.dumps(plan, indent=1))
        return EXIT_OK
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = SERIES_DIR / f"{sid}.json"
    if plan_path.exists():
        print(f"series {sid} already exists: {plan_path}", file=sys.stderr)
        return EXIT_ERR

    def save() -> None:
        plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    owner = a.cpu_lock_owner or f"P5-series {sid}"
    inherited = bool(a.cpu_lock_owner)
    if not inherited:
        try:
            plan["cpu_lock"] = cpu_lock_acquire(owner, f"controlled series {sid}: {len(order)} cells", wait_s=a.cpu_lock_wait_s)
        except CpuLockBusy as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return EXIT_REFUSED
    plan["status"] = "running"
    save()
    worst = EXIT_OK
    try:
        for o in order:
            rg = ranges.get(o["method"])
            if rg and not rg.startswith("/"):
                rg = "/work/" + Path(rg).as_posix()
            cfg = default_config(method=o["method"], budget_bytes=budget, workload_set=a.workload, repeat=o["repeat"],
                                 cell_id=o["cell_id"], conversations=a.conversations.split(",") if a.conversations else None,
                                 n_predict_cap=a.n_predict_cap, storage_mode=a.storage, rate_bps=a.rate_bps,
                                 ranges=rg, purpose=a.purpose, engine=engines[o["variant"]])
            cfg["series"] = {"series_id": sid, "index": o["index"], "position_in_block": o["position_in_block"],
                             "seed": a.seed, "plan": str(plan_path)}
            o["started"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
            rc = run_cell(cfg, max_bg_cpu_pct=a.max_bg_cpu_pct, bg_wait_s=a.bg_wait_s, timeout_s=a.timeout_s,
                          cpu_lock="inherit", cpu_lock_owner=owner)
            o["exit_code"] = rc
            o["finished"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
            save()
            if rc != EXIT_OK:
                worst = rc
                if not a.keep_going:
                    plan["status"] = f"stopped at cell {o['index']} ({o['cell_id']}): exit {rc}"
                    return rc
        plan["status"] = "complete" if worst == EXIT_OK else f"complete with failures (worst exit {worst})"
        return worst
    finally:
        save()
        if not inherited:
            cpu_lock_release(owner)


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def _budget(a: argparse.Namespace) -> int:
    if getattr(a, "budget_bytes", None):
        return int(a.budget_bytes)
    return int(round(float(a.budget_gib) * GIB))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pigemma.controlled", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cell", help="run one controlled cell in a fresh container")
    c.add_argument("--method", required=True, choices=["A", "B", "C", "D"])
    c.add_argument("--budget-gib", type=float, default=8.0)
    c.add_argument("--budget-bytes", type=int)
    c.add_argument("--workload", required=True, choices=["calibration", "tuning", "heldout"])
    c.add_argument("--conversations", help="comma-separated conversation ids (default: the whole set, file order)")
    c.add_argument("--n-predict-cap", type=int, help="cap max_tokens per request (smoke tests only)")
    c.add_argument("--repeat", type=int, default=0)
    c.add_argument("--cell-id")
    c.add_argument("--tag")
    c.add_argument("--ranges", help="pinner ranges file (methods C and D), repo-relative or /work/...")
    c.add_argument("--storage", default="fuse-throttled", choices=["fuse-throttled", "fuse-unthrottled", "direct"])
    c.add_argument("--rate-bps", type=float, default=PI_MICROSD_BPS)
    c.add_argument("--op-latency-us", type=float, default=0.0)
    c.add_argument("--cpuset", default=DEFAULT_CPUSET)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--ctx", type=int, default=4096)
    c.add_argument("--batch", type=int, default=2048)
    c.add_argument("--ubatch", type=int, default=512)
    c.add_argument("--ctx-checkpoints", type=int, default=32)
    c.add_argument("--server-warmup", choices=["on", "off"], default="on")
    c.add_argument("--residency", choices=["per_request", "phases", "off"], default="per_request")
    c.add_argument("--no-lock", action="store_true")
    c.add_argument("--ignore-other-jobs", action="store_true")
    c.add_argument("--max-bg-cpu-pct", type=float)
    c.add_argument("--idle-seconds", type=float, default=10.0)
    c.add_argument("--timeout-s", type=float, default=4 * 3600)
    c.add_argument("--purpose", choices=["measurement", "pilot", "smoke"], default="measurement",
                   help="smoke cells are excluded from controlled_summary unless --include-smoke")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--engine", default=REFERENCE_ENGINE,
                   help=f"llama-server build: {REFERENCE_ENGINE} (built in) or an id from the engines file")
    c.add_argument("--engines-file", default=None, help=f"default {ENGINES_FILE}")
    c.add_argument("--server-bin", help="override the engine's binary (path inside the container or repo-relative)")
    c.add_argument("--server-extra", help="extra llama-server words for this cell (one string; recorded)")
    c.add_argument("--allow-engine-status", action="store_true", help="run an engine whose status is not 'runnable'")
    c.add_argument("--cpu-lock-wait-s", type=float, default=0.0,
                   help="wait this long for results/.cpu.lock (poll 60 s); default 0 = refuse when it is held")
    c.add_argument("--cpu-lock-owner", help="the cell runs under a lock this owner already holds (a stage or series)")
    c.add_argument("--no-cpu-lock", action="store_true",
                   help="functional smoke tests only: do not take results/.cpu.lock (recorded in config.json)")
    c.add_argument("--bg-wait-s", type=float, default=0.0,
                   help="with --max-bg-cpu-pct: keep probing this long for an idle host before refusing")

    sr = sub.add_parser("series", help="variants x repeats as randomized complete blocks under one CPU-lock hold")
    sr.add_argument("--variants", required=True, help="comma list: A,B,C,D and/or <engine>:<method>, e.g. A,C,D,ik-plain:A")
    sr.add_argument("--repeats", type=int, default=3)
    sr.add_argument("--seed", type=int, required=True, help="order seed (recorded; choose before looking at results)")
    sr.add_argument("--budget-gib", type=float, default=8.0)
    sr.add_argument("--budget-bytes", type=int)
    sr.add_argument("--workload", required=True, choices=["calibration", "tuning", "heldout"])
    sr.add_argument("--conversations")
    sr.add_argument("--n-predict-cap", type=int)
    sr.add_argument("--ranges-c", help="pinner ranges file of method C")
    sr.add_argument("--ranges-d", help="pinner ranges file of method D")
    sr.add_argument("--storage", default="fuse-throttled", choices=["fuse-throttled", "fuse-unthrottled", "direct"])
    sr.add_argument("--rate-bps", type=float, default=PI_MICROSD_BPS)
    sr.add_argument("--purpose", choices=["measurement", "pilot", "smoke"], default="measurement")
    sr.add_argument("--engines-file", default=None)
    sr.add_argument("--max-bg-cpu-pct", type=float)
    sr.add_argument("--bg-wait-s", type=float, default=600.0)
    sr.add_argument("--timeout-s", type=float, default=4 * 3600)
    sr.add_argument("--cpu-lock-wait-s", type=float, default=0.0)
    sr.add_argument("--cpu-lock-owner", help="run under a lock this owner already holds instead of taking one")
    sr.add_argument("--series-id")
    sr.add_argument("--keep-going", action="store_true", help="continue after a failed/refused cell (default: stop)")
    sr.add_argument("--dry-run", action="store_true", help="print the randomized plan only")

    eg = sub.add_parser("engines", help="list the engines the harness can run")
    eg.add_argument("--engines-file", default=None)
    eg.add_argument("--schema", action="store_true", help="print the engines.json schema")

    ec = sub.add_parser("engine-check", help="run an engine's --version/--help in the cell image; report missing flags")
    ec.add_argument("--engine", required=True)
    ec.add_argument("--engines-file", default=None)
    ec.add_argument("--out", help="write the JSON report here")

    cl = sub.add_parser("cpu-lock", help="status | acquire | release of results/.cpu.lock")
    cl.add_argument("action", choices=["status", "acquire", "release"])
    cl.add_argument("--owner")
    cl.add_argument("--purpose", default="controlled-memory cells")
    cl.add_argument("--wait-s", type=float, default=0.0)

    d = sub.add_parser("docker-cmd", help="print the docker command of a cell config.json")
    d.add_argument("config")

    m = sub.add_parser("make-ranges", help="plan a C or D pin set (page exact) and write the pinner ranges")
    m.add_argument("--map", default=MAP_PATH)
    m.add_argument("--page-size", type=int, default=4096)
    m.add_argument("--method", required=True, choices=["C", "D"])
    m.add_argument("--ranking", help="C: ranking JSON built from calibration.jsonl only")
    m.add_argument("--expert-pin-bytes", type=int, help="C: bytes of NEW unique pages for experts")
    m.add_argument("--match", help="D: the C plan JSON to size-match (total unique pages)")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True, help="output stem: <stem>.ranges.txt and <stem>.plan.json")

    b = sub.add_parser("pin-budget", help="what is left for pinned experts in a budget")
    b.add_argument("--budget-gib", type=float, default=8.0)
    b.add_argument("--budget-bytes", type=int)
    b.add_argument("--from-cell", help="a method-A cell directory (measured anon/kernel memory)")
    b.add_argument("--anon-bytes", type=int, help="override (labelled assumed)")
    b.add_argument("--kernel-bytes", type=int)
    b.add_argument("--map", default=MAP_PATH)
    b.add_argument("--page-size", type=int, default=4096)
    b.add_argument("--miss-reserve-bytes", type=int,
                   help="page cache kept free for expert misses (default: worst-case expert bytes of one token)")
    b.add_argument("--safety-frac", type=float, default=0.05)

    v = sub.add_parser("verify-throttle", help="run the FUSE throttle verification container")
    v.add_argument("--out", default=None)
    v.add_argument("--rate-bps", type=float, default=PI_MICROSD_BPS)
    v.add_argument("--region-mib", type=int, default=512)
    v.add_argument("--random-pages", type=int, default=2000)
    v.add_argument("--memory-gib", type=float, default=4.0)

    a = ap.parse_args(argv)

    if a.cmd == "cell":
        import shlex
        budget = _budget(a)
        try:
            eng = resolve_engine(a.engine, a.engines_file, a.server_bin, allow_not_runnable=a.allow_engine_status)
        except EngineError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return EXIT_REFUSED
        ok_m, why_m = engine_allows(eng, a.method)
        if not ok_m and not a.allow_engine_status:
            print(f"REFUSED: {why_m} (--allow-engine-status overrides)", file=sys.stderr)
            return EXIT_REFUSED
        cell_id = a.cell_id or new_cell_id(a.method, budget, a.workload, a.repeat, a.tag, eng["id"])
        ranges = a.ranges
        if ranges and not ranges.startswith("/"):
            ranges = "/work/" + Path(ranges).as_posix()
        cfg = default_config(method=a.method, budget_bytes=budget, workload_set=a.workload, repeat=a.repeat,
                             cell_id=cell_id, conversations=a.conversations.split(",") if a.conversations else None,
                             n_predict_cap=a.n_predict_cap, storage_mode=a.storage, rate_bps=a.rate_bps,
                             ranges=ranges, threads=a.threads, ctx=a.ctx, batch=a.batch, ubatch=a.ubatch,
                             server_warmup=a.server_warmup == "on", ctx_checkpoints=a.ctx_checkpoints,
                             residency=a.residency, cpuset=a.cpuset, op_latency_us=a.op_latency_us,
                             purpose=a.purpose, engine=eng,
                             server_extra=shlex.split(a.server_extra) if a.server_extra else None)
        if a.dry_run:
            print(json.dumps(cfg, indent=1))
            print(" ".join(docker_argv(cfg)))
            return EXIT_OK
        mode = "ignore" if a.no_cpu_lock else ("inherit" if a.cpu_lock_owner else "acquire")
        return run_cell(cfg, use_lock=not a.no_lock, ignore_other_jobs=a.ignore_other_jobs,
                        max_bg_cpu_pct=a.max_bg_cpu_pct, timeout_s=a.timeout_s, idle_s=a.idle_seconds,
                        cpu_lock=mode, cpu_lock_owner=a.cpu_lock_owner, cpu_lock_wait_s=a.cpu_lock_wait_s,
                        bg_wait_s=a.bg_wait_s)

    if a.cmd == "series":
        try:
            return run_series(a)
        except (EngineError, ValueError) as e:
            print(f"series: {e}", file=sys.stderr)
            return EXIT_ERR

    if a.cmd == "engines":
        if a.schema:
            print(ENGINE_SCHEMA_DOC)
            return EXIT_OK
        try:
            engines = load_engines(a.engines_file)
        except (EngineError, ValueError) as e:
            print(f"engines: {e}", file=sys.stderr)
            return EXIT_ERR
        rows = []
        for e in engines.values():
            ok, why = engine_runnable(e)
            rows.append({"engine_id": e["id"], "runnable": ok, "why": why, "bin": e["bin"], "revision": e.get("revision"),
                         "parent": e.get("parent"), "flag_overrides": e.get("flag_overrides"), "extra_args": e.get("extra"),
                         "env": e.get("env"), "alias_of": e.get("alias_of"), "methods": e.get("methods"),
                         "source": e.get("profile_source")})
        print(json.dumps({"engines_file": str(a.engines_file or ENGINES_FILE),
                          "engines_file_exists": Path(a.engines_file or ENGINES_FILE).exists(), "engines": rows}, indent=1))
        return EXIT_OK

    if a.cmd == "engine-check":
        try:
            eng = resolve_engine(a.engine, a.engines_file, allow_not_runnable=True)
        except EngineError as e:
            print(f"engine-check: {e}", file=sys.stderr)
            return EXIT_ERR
        rep_ = engine_check(eng)
        text = json.dumps(rep_, indent=1)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(text, encoding="utf-8")
        print(text)
        return EXIT_OK if rep_["ok"] else EXIT_CELL_FAIL

    if a.cmd == "cpu-lock":
        if a.action == "status":
            st = cpu_lock_status()
            print(json.dumps(st))
            return EXIT_REFUSED if st["held"] else EXIT_OK
        if not a.owner:
            print("cpu-lock acquire/release need --owner", file=sys.stderr)
            return EXIT_ERR
        if a.action == "acquire":
            try:
                print(json.dumps(cpu_lock_acquire(a.owner, a.purpose, wait_s=a.wait_s)))
                return EXIT_OK
            except CpuLockBusy as e:
                print(f"REFUSED: {e}", file=sys.stderr)
                return EXIT_REFUSED
        ok = cpu_lock_release(a.owner)
        print(json.dumps({"released": ok, "status": cpu_lock_status()}))
        return EXIT_OK if ok else EXIT_ERR

    if a.cmd == "docker-cmd":
        cfg = json.loads(Path(a.config).read_text("utf-8"))
        print(" ".join(docker_argv(cfg)))
        return EXIT_OK

    if a.cmd in ("make-ranges", "pin-budget"):
        from pigemma.expert_map import load_expert_map
        em = load_expert_map(REPO / a.map if not os.path.isabs(a.map) else a.map)

    if a.cmd == "make-ranges":
        stem = Path(a.out)
        stem.parent.mkdir(parents=True, exist_ok=True)
        if a.method == "C":
            if not a.ranking or a.expert_pin_bytes is None:
                print("make-ranges C needs --ranking and --expert-pin-bytes", file=sys.stderr)
                return EXIT_ERR
            ranking, meta = load_ranking(a.ranking, int(em["n_layer"]), int(em["n_expert"]))
            plan = plan_pins(em, a.page_size, "C", ranking=ranking, expert_pin_bytes=a.expert_pin_bytes)
            plan["ranking"] = meta
        else:
            if not a.match:
                print("make-ranges D needs --match <C plan.json>", file=sys.stderr)
                return EXIT_ERR
            cplan = json.loads(Path(a.match).read_text("utf-8"))
            if int(cplan["page"]) != a.page_size:
                print("page size differs from the C plan", file=sys.stderr)
                return EXIT_ERR
            plan = plan_pins(em, a.page_size, "D", target_total_pages=int(cplan["total_unique_pages"]), seed=a.seed)
            plan["matched_to"] = {"plan": str(a.match), "total_unique_pages": cplan["total_unique_pages"]}
        plan["map"] = str(a.map)
        plan["created"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        chk = write_ranges(em, plan, stem.with_suffix(".ranges.txt"))
        plan["ranges"] = chk
        stem.with_suffix(".plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
        print(json.dumps({k: plan[k] for k in ("method", "n_experts", "total_unique_pages", "total_bytes",
                                                "nonexpert_pages", "check_ok", "diff_pages_vs_target")} | chk))
        return EXIT_OK if plan["check_ok"] and chk["union_matches_plan"] else EXIT_ERR

    if a.cmd == "pin-budget":
        budget = _budget(a)
        need = measured_needs_from_cell(Path(a.from_cell)) if a.from_cell else {}
        anon = a.anon_bytes if a.anon_bytes is not None else need.get("cg_anon_peak_bytes")
        kern = a.kernel_bytes if a.kernel_bytes is not None else need.get("cg_kernel_peak_bytes")
        if anon is None or kern is None:
            print("need --from-cell (measured) or --anon-bytes and --kernel-bytes (assumed)", file=sys.stderr)
            return EXIT_ERR
        ne_bytes = em.pages_for([], a.page_size, with_nonexpert=True) * a.page_size
        miss = a.miss_reserve_bytes if a.miss_reserve_bytes is not None else int(em["summary"]["expert_bytes_per_token_worst"])
        res = pin_budget(budget, anon_bytes=int(anon), kernel_bytes=int(kern), nonexpert_bytes=ne_bytes,
                         miss_reserve_bytes=int(miss), safety_bytes=int(a.safety_frac * budget))
        res["labels"] = {"anon_bytes": "measured" if a.anon_bytes is None else "assumed",
                         "kernel_bytes": "measured" if a.kernel_bytes is None else "assumed",
                         "nonexpert_pinned_bytes": "exact (map, unique pages)",
                         "miss_page_cache_reserve_bytes": "assumed default = worst-case expert bytes of one token"
                         if a.miss_reserve_bytes is None else "given",
                         "safety_bytes": f"assumed {a.safety_frac:g} x budget"}
        res["measured_needs"] = need
        print(json.dumps(res, indent=1))
        return EXIT_OK if res["feasible"] else EXIT_ERR

    if a.cmd == "verify-throttle":
        out = a.out or f"results/controlled/throttle-verify-{time.strftime('%Y%m%d-%H%M%S')}"
        mem = int(a.memory_gib * GIB)
        argv = ["docker", "run", "--rm", "--name", "pigemma-p5-throttle-verify",
                f"--memory={mem}b", f"--memory-swap={mem}b", "--memory-swappiness=0", "--cpus=2",
                "--cap-add", "SYS_ADMIN", "--device", "/dev/fuse", "--security-opt", "apparmor:unconfined",
                "--network", "none", "-v", f"{VOLUME}:/models:ro", "-v", f"{str(REPO).replace(chr(92), '/')}:/work",
                "-e", "PYTHONPATH=/work", "-w", "/work", IMAGE, "python3", "-m", "pigemma.controlled_cell",
                "verify-throttle", "--backing", MODEL_IN_VOLUME, "--out", "/work/" + Path(out).as_posix(),
                "--rate-bps", f"{a.rate_bps:.0f}", "--region-bytes", str(a.region_mib * 1024**2),
                "--random-pages", str(a.random_pages)]
        print("$ " + " ".join(argv), flush=True)
        return subprocess.call(argv)
    return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
