"""pigemma.paths - single source of truth for the repo layout and the shared run guards.

What lives here (everything else in pigemma imports it from here):

* Repo layout: ``REPO_ROOT`` (= parent of the package dir) and the well-known sub-directories.
* Model registry ``MODELS`` (id -> dict(path, bytes, kind, quant, draft_of, ...)) for the five
  GGUFs downloaded on the desktop, plus ``model_path`` / ``require_model`` / ``model_identity``.
* Binary resolution per platform: ``bin(tool)`` (llama.cpp tools and the custom tools
  ``pigemma-probe``, ``pinner``, ``membw``) and the pinned llama.cpp version check.
* ``host_fingerprint()`` for the results envelope (data_formats "results JSON envelope").
* ``sha256_prefix()`` - cheap identity of a 17 GB file (hash of the first 16 MiB + size).
* ``free_disk_bytes()``.
* ``bench_lock()`` - the cross-agent lock (``results/.bench.lock``) that serialises every
  model-loading job on this machine, with a free-RAM gate.

Runs on Windows (Python 3.14), inside the Linux Docker image, inside the emulated arm64 Debian
container and on the Pi (Python 3.13). Stdlib only; ``psutil`` is used when importable and every
psutil call has a stdlib fallback (/proc on Linux, ctypes on Windows).

Design decisions (documented here because the manifest left them open):

* GB means 10**9 bytes everywhere in this module (``min_free_gb``, ``PIGEMMA_MIN_FREE_GB``).
* ``PIGEMMA_MIN_FREE_GB`` (env) takes precedence over the ``min_free_gb`` argument, so an operator
  can relax/tighten the RAM gate for every caller without code changes (the Pi needs ~2-4).
* Lock liveness: a lock written on the *same host* (hostname + container flag + Linux boot_id) is
  alive iff its pid is alive (and, when a start-time marker was recorded, the pid was not reused).
  A lock written on a *different host* (Windows host vs Docker container share results/ through
  the bind mount but not a pid namespace) is alive iff its heartbeat is fresh: the holder touches
  the lock file every ``HEARTBEAT_S`` seconds; older than ``PIGEMMA_LOCK_STALE_S`` (default 120 s)
  means stale.
* Children started by ``python -m pigemma.paths lock --hold -- <cmd>`` receive
  ``PIGEMMA_LOCK_HELD=<token>`` and ``PIGEMMA_LOCK_FILE=<path>``; a nested ``bench_lock()`` whose
  env token matches the live lock passes through ("inherited") instead of deadlocking. A second
  ``bench_lock()`` in the *same* process is refused (not re-entrant).
* ``bin()`` checks ``aarch64`` before ``/.dockerenv`` because the emulated arm64 Pi container also
  has ``/.dockerenv`` but must use the native aarch64 build in ``$LLAMA_DIR/build/bin``.
* On Windows ``bin('pigemma-probe'|'pinner'|'membw')`` returns the Docker-built Linux binary under
  tools/build-linux-x86/bin (there is no Windows compiler); run it through pc/docker_run.sh.
* ``--hold`` resolves the command's executable with ``shutil.which`` (PATH order) before spawning.
  On Windows a bare ``bash``/``sh`` that resolves to System32 (the WSL launcher, which is what
  PowerShell finds first) or to nothing is replaced by Git Bash (env ``PIGEMMA_BASH``, next to
  git.exe, or Program Files\\Git\\bin\\bash.exe): pc/*.sh need Git Bash + Docker Desktop.

CLI (``python -m pigemma.paths <cmd> --help`` for details)::

    info                         JSON: repo root, llama version check, host, models, binaries, lock
    host                         JSON host_fingerprint()
    model <id|path> [--require] [--json] [--no-sha]
    bin <tool> [--require]
    sha <id|path> [--n-bytes N]
    free-disk [path] [--gb]
    container-path <path>...     repo path -> /work/... (path inside the Docker image)
    lock --status [--lock-file F]
    lock --hold [--min-free-gb N] [--lock-file F] -- <cmd...>

Exit codes: 0 ok; 1 unexpected error (``lock --status``: lock held); 2 usage; 3 missing file /
precondition failed; 4 refused (lock held by a live job, or not enough free RAM); with
``lock --hold`` the child's exit code is forwarded (127 command not found, 126 not executable,
128+N killed by signal N, 130 interrupted).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

try:  # optional dependency: every use below has a stdlib fallback
    import psutil  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - depends on the interpreter  # noqa: BLE001
    psutil = None  # type: ignore[assignment]

log = logging.getLogger("pigemma.paths")

# --------------------------------------------------------------------------------------------
# Repo layout
# --------------------------------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PACKAGE_DIR.parent
MODELS_DIR: Path = REPO_ROOT / "models"
RESULTS_DIR: Path = REPO_ROOT / "results"
TRACES_DIR: Path = REPO_ROOT / "traces"
DATA_DIR: Path = REPO_ROOT / "data"
CALIBRATION_DIR: Path = DATA_DIR / "calibration"
PROMPTS_DIR: Path = DATA_DIR / "prompts"
DOCS_DIR: Path = REPO_ROOT / "docs"
CATALOG_PATH: Path = PACKAGE_DIR / "data" / "catalog.json"
THIRD_PARTY_DIR: Path = REPO_ROOT / "third_party"
LLAMA_SRC_DIR: Path = THIRD_PARTY_DIR / "llama.cpp"
LLAMA_WIN_DIR: Path = THIRD_PARTY_DIR / "llama.cpp-win"
LLAMA_LINUX_X86_BIN: Path = LLAMA_SRC_DIR / "build-linux-x86" / "bin"
TOOLS_DIR: Path = REPO_ROOT / "tools"
TOOLS_DOCKER_BIN: Path = TOOLS_DIR / "build-linux-x86" / "bin"
TOOLS_NATIVE_BIN: Path = TOOLS_DIR / "build" / "bin"
LOCK_FILE: Path = RESULTS_DIR / ".bench.lock"
CONTAINER_ROOT = "/work"  # where pc/docker_run.sh bind-mounts the repo

# --------------------------------------------------------------------------------------------
# Pinned llama.cpp and units
# --------------------------------------------------------------------------------------------
LLAMA_COMMIT = "a7b94df"
LLAMA_BUILD = 11398

GB = 1_000_000_000
MiB = 1 << 20
SHA_PREFIX_BYTES = 16 * MiB
DEFAULT_MIN_FREE_GB = 20.0
HEARTBEAT_S = 15.0
DEFAULT_STALE_S = 120.0

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_MISSING = 3
EXIT_REFUSED = 4

# Container / kernel interface files (module-level so tests can monkeypatch them).
CONTAINER_MARKERS: tuple[str, ...] = ("/.dockerenv", "/run/.containerenv")
CGROUP_V2_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
CGROUP_V1_MEMORY_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
PROC_MEMINFO = Path("/proc/meminfo")
PROC_CPUINFO = Path("/proc/cpuinfo")

# Custom tools built from tools/CMakeLists.txt (everything else is a llama.cpp binary).
CUSTOM_TOOLS: frozenset[str] = frozenset({"pigemma-probe", "pinner", "membw"})

# --------------------------------------------------------------------------------------------
# Model registry (sizes are the exact byte counts of the files downloaded on 2026-10-04)
# --------------------------------------------------------------------------------------------
MODELS: dict[str, dict[str, Any]] = {
    "26b-q4": {
        "path": "models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf",
        "bytes": 14439363584,
        "kind": "moe",
        "quant": "Q4_0",
        "draft_of": None,
        "family": "26b",
        "repo": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
        "file": "gemma-4-26B_q4_0-it.gguf",
        "spec_type": None,
    },
    "31b-q4": {
        "path": "models/google__gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf",
        "bytes": 17651001568,
        "kind": "dense",
        "quant": "Q4_0",
        "draft_of": None,
        "family": "31b",
        "repo": "google/gemma-4-31B-it-qat-q4_0-gguf",
        "file": "gemma-4-31B_q4_0-it.gguf",
        "spec_type": None,
    },
    "26b-mtp": {
        "path": "models/ggml-org__gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it-Q4_0.gguf",
        "bytes": 251939520,
        "kind": "draft",
        "quant": "Q4_0",
        "draft_of": "26b-q4",
        "family": "26b",
        "repo": "ggml-org/gemma-4-26B-A4B-it-GGUF",
        "file": "mtp-gemma-4-26B-A4B-it-Q4_0.gguf",
        "spec_type": "draft-mtp",
    },
    "26b-dflash": {
        "path": "models/ggml-org__gemma-4-26B-A4B-it-GGUF/dflash-gemma-4-26B-A4B-it-Q8_0.gguf",
        "bytes": 472433824,
        "kind": "draft",
        "quant": "Q8_0",
        "draft_of": "26b-q4",
        "family": "26b",
        "repo": "ggml-org/gemma-4-26B-A4B-it-GGUF",
        "file": "dflash-gemma-4-26B-A4B-it-Q8_0.gguf",
        "spec_type": "draft-dflash",
    },
    "31b-mtp": {
        "path": "models/unsloth__gemma-4-31B-it-qat-GGUF/MTP/mtp-gemma-4-31B-it-Q4_0.gguf",
        "bytes": 279955968,
        "kind": "draft",
        "quant": "Q4_0",
        "draft_of": "31b-q4",
        "family": "31b",
        "repo": "unsloth/gemma-4-31B-it-qat-GGUF",
        "file": "MTP/mtp-gemma-4-31B-it-Q4_0.gguf",
        "spec_type": "draft-mtp",
    },
}

_QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:UD-)?I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32)(?=$|[-.])", re.IGNORECASE
)


class BenchLockError(RuntimeError):
    """Base class for bench-lock refusals."""


class LockBusyError(BenchLockError):
    """Another model-loading job holds results/.bench.lock."""


class InsufficientMemoryError(BenchLockError):
    """Not enough free RAM to start a model-loading job."""


# --------------------------------------------------------------------------------------------
# Platform helpers (read at call time so tests can monkeypatch sys.platform / env)
# --------------------------------------------------------------------------------------------
def is_windows() -> bool:
    """True on native Windows Python (also msys/cygwin builds)."""
    return sys.platform.startswith("win") or sys.platform in ("cygwin", "msys")


def machine() -> str:
    """Normalised CPU architecture: 'aarch64', 'x86_64' or the raw platform.machine()."""
    m = (platform.machine() or "").lower()
    if m in ("aarch64", "arm64", "armv8l", "armv8b"):
        return "aarch64"
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    return m or "unknown"


def in_container() -> bool:
    """True inside a Docker/Podman container (/.dockerenv); env PIGEMMA_IN_CONTAINER=0/1 overrides."""
    env = os.environ.get("PIGEMMA_IN_CONTAINER", "").strip()
    if env in ("0", "1"):
        return env == "1"
    if is_windows():
        return False
    return any(os.path.exists(m) for m in CONTAINER_MARKERS)


def llama_dir() -> Path:
    """$LLAMA_DIR (Pi llama.cpp checkout, built by pi/02_build_llama.sh), default ~/llama.cpp."""
    return Path(os.environ.get("LLAMA_DIR", "").strip() or "~/llama.cpp").expanduser()


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
def _models_dir_overrides() -> list[Path]:
    out: list[Path] = []
    for var in ("PIGEMMA_MODELS_DIR", "MODEL_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            out.append(Path(v).expanduser())
    return out


def _model_candidates(entry: dict[str, Any]) -> list[Path]:
    rel = Path(entry["path"])
    sub = Path(*rel.parts[1:]) if len(rel.parts) > 1 and rel.parts[0] == "models" else rel
    cands: list[Path] = []
    for d in _models_dir_overrides():
        cands += [d / sub, d / Path(entry["file"]).name]
    cands.append(REPO_ROOT / rel)
    return cands


def model_ids(kind: str | None = None) -> list[str]:
    """Registry ids, optionally filtered by kind ('moe', 'dense', 'draft')."""
    return [k for k, v in MODELS.items() if kind is None or v["kind"] == kind]


def drafts_for(model_id: str) -> list[str]:
    """Ids of the draft sidecars registered for a main model id."""
    return [k for k, v in MODELS.items() if v.get("draft_of") == model_id]


def model_path(id_or_path: str | os.PathLike[str]) -> Path:
    """Resolve a registry id or a path to a Path (no existence check, see require_model).

    Ids are looked up in ``$PIGEMMA_MODELS_DIR`` / ``$MODEL_DIR`` (``<dir>/<repo_dir>/<file>`` then
    ``<dir>/<basename>``) and then under REPO_ROOT/models; the first existing candidate wins and the
    primary candidate is returned when none exists. A relative path that does not exist relative to
    the cwd is tried relative to REPO_ROOT.
    """
    s = os.fspath(id_or_path)
    if s in MODELS:
        cands = _model_candidates(MODELS[s])
        for c in cands:
            if c.is_file():
                return c
        return cands[0]
    p = Path(s).expanduser()
    if p.is_absolute() or p.exists():
        return p
    rp = REPO_ROOT / p
    return rp if rp.exists() else p


def model_id_for(path: str | os.PathLike[str]) -> str | None:
    """Registry id for an id, or for a path whose basename equals a registered file (else None)."""
    s = os.fspath(path)
    if s in MODELS:
        return s
    name = Path(s).name
    for k, v in MODELS.items():
        if Path(v["path"]).name == name:
            return k
    return None


def download_hint(model_id: str) -> str:
    """One-line Hugging Face download hint for a registry id."""
    e = MODELS[model_id]
    parts = Path(e["path"]).parts
    local = parts[1] if len(parts) > 2 else ""
    return (
        f"HF repo '{e['repo']}', file '{e['file']}' ({e['bytes']} bytes, ungated): "
        f"hf download {e['repo']} {e['file']} --local-dir models/{local}"
    )


def require_model(id_or_path: str | os.PathLike[str]) -> Path:
    """Return the model's Path; raise FileNotFoundError (with the HF hint) if missing or wrong size.

    A registered file whose size differs from the registry (partial download, other quant) is
    treated as missing. Unregistered paths are only checked for existence.
    """
    p = model_path(id_or_path)
    mid = model_id_for(id_or_path)
    if mid is None:
        mid = model_id_for(p)
    hint = f"; {download_hint(mid)}" if mid else ""
    if not p.is_file():
        log.info("model %s: missing at %s", os.fspath(id_or_path), p)
        raise FileNotFoundError(f"model {os.fspath(id_or_path)!r} not found at {p}{hint}")
    if mid is not None:
        size = p.stat().st_size
        want = int(MODELS[mid]["bytes"])
        if size != want:
            log.info("model %s: size %d != registry %d at %s", mid, size, want, p)
            raise FileNotFoundError(
                f"model {mid!r} at {p} has {size} bytes, expected {want} "
                f"(partial download or a different file){hint}"
            )
    log.info("model %s: ok at %s", os.fspath(id_or_path), p)
    return p


def guess_quant(path: str | os.PathLike[str]) -> str:
    """Quant tag (registry value when registered, else parsed from the file name), or 'unknown'."""
    mid = model_id_for(path)
    if mid:
        return str(MODELS[mid]["quant"])
    m = _QUANT_RE.search(Path(os.fspath(path)).stem)
    return m.group(1).upper() if m else "unknown"


def repo_relative(path: str | os.PathLike[str]) -> str:
    """POSIX path relative to REPO_ROOT when inside the repo, else the path string as given."""
    p = Path(os.fspath(path)).expanduser()
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return str(p)


def model_identity(id_or_path: str | os.PathLike[str], with_sha: bool = True) -> dict[str, Any]:
    """The results-envelope ``model`` block: {id, file, quant, bytes, sha_prefix}.

    ``id`` is the registry id or the file stem; ``file`` is repo-relative when possible; ``bytes`` is
    the on-disk size (registry size when the file is absent, None if unknown); ``sha_prefix`` is the
    sha256 hex of the first 16 MiB (None when the file is absent or with_sha=False).
    """
    p = model_path(id_or_path)
    mid = model_id_for(id_or_path)
    if mid is None:
        mid = model_id_for(p)
    exists = p.is_file()
    if exists:
        size: int | None = p.stat().st_size
    else:
        size = int(MODELS[mid]["bytes"]) if mid else None
    sha = sha256_prefix(p)["sha256"] if (exists and with_sha) else None
    return {
        "id": mid or p.stem,
        "file": repo_relative(p),
        "quant": guess_quant(p),
        "bytes": size,
        "sha_prefix": sha,
    }


# --------------------------------------------------------------------------------------------
# Binaries
# --------------------------------------------------------------------------------------------
_LOGGED_BINS: set[tuple[str, str]] = set()
_VERSION_CHECKED = False


def _strip_exe(tool: str) -> str:
    return tool[:-4] if tool.lower().endswith(".exe") else tool


def bin(tool: str, *, check: bool = False) -> Path:  # noqa: A001 - name fixed by the design manifest
    """Resolve a llama.cpp tool (``llama-bench`` ...) or a custom tool to a Path.

    Order: env ``PIGEMMA_TOOLS_BIN`` (custom tools) / ``PIGEMMA_LLAMA_BIN`` (llama tools) dir;
    Windows -> third_party/llama.cpp-win/<tool>.exe (custom tools: tools/build-linux-x86/bin, a
    Linux binary for use through Docker); Linux aarch64 -> $LLAMA_DIR/build/bin (custom:
    tools/build/bin); Linux x86 container (/.dockerenv) -> third_party/llama.cpp/build-linux-x86/bin
    (= /work/... inside the image; custom: tools/build-linux-x86/bin); other Linux ->
    $LLAMA_DIR/build/bin when it exists else the repo's build-linux-x86 (custom: tools/build/bin
    when it exists else tools/build-linux-x86/bin). ``check=True`` raises FileNotFoundError when
    the binary does not exist.
    """
    name = _strip_exe(tool)
    custom = name in CUSTOM_TOOLS
    win = is_windows()
    arch = machine()
    env_var = "PIGEMMA_TOOLS_BIN" if custom else "PIGEMMA_LLAMA_BIN"
    env_dir = os.environ.get(env_var, "").strip()
    if env_dir:
        d, why = Path(env_dir).expanduser(), env_var
    elif custom:
        if win:
            d, why = TOOLS_DOCKER_BIN, "windows: Docker-built Linux binary, run via pc/docker_run.sh"
        elif arch == "aarch64":
            d, why = TOOLS_NATIVE_BIN, "linux aarch64"
        elif in_container():
            d, why = TOOLS_DOCKER_BIN, "linux container"
        elif TOOLS_NATIVE_BIN.is_dir():
            d, why = TOOLS_NATIVE_BIN, "linux native build"
        else:
            d, why = TOOLS_DOCKER_BIN, "linux fallback"
    else:
        if win:
            d, why = LLAMA_WIN_DIR, "windows"
        elif arch == "aarch64":
            d, why = llama_dir() / "build" / "bin", "linux aarch64 ($LLAMA_DIR)"
        elif in_container():
            d, why = LLAMA_LINUX_X86_BIN, "linux x86 container"
        elif (llama_dir() / "build" / "bin").is_dir():
            d, why = llama_dir() / "build" / "bin", "linux ($LLAMA_DIR)"
        else:
            d, why = LLAMA_LINUX_X86_BIN, "linux fallback"
        _maybe_check_version()
    exe = name + (".exe" if (win and not custom) else "")
    p = d / exe
    key = (name, str(p))
    if key not in _LOGGED_BINS:
        _LOGGED_BINS.add(key)
        log.info("bin %s -> %s (%s%s)", name, p, why, "" if p.exists() else ", MISSING")
    if check and not p.is_file():
        raise FileNotFoundError(f"binary {name!r} not found at {p} ({why})")
    return p


tool_path = bin  # alias that does not shadow the builtin at call sites


def require_bin(tool: str) -> Path:
    """bin(tool, check=True)."""
    return bin(tool, check=True)


def _git_head(repo_dir: Path) -> str | None:
    """Commit sha of HEAD of a git checkout, read from .git without running git (None if unknown)."""
    try:
        gitdir = repo_dir / ".git"
        if gitdir.is_file():  # worktree / submodule: "gitdir: <path>"
            txt = gitdir.read_text(encoding="utf-8", errors="replace").strip()
            if txt.startswith("gitdir:"):
                gitdir = (repo_dir / txt.split(":", 1)[1].strip()).resolve()
        head = (gitdir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.split(":", 1)[1].strip()
        loose = gitdir / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8", errors="replace").strip() or None
        packed = gitdir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.strip().split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return None
    return None


def llama_version_check() -> dict[str, Any]:
    """Compare the local llama.cpp builds with LLAMA_BUILD / LLAMA_COMMIT.

    Sources: third_party/llama.cpp-win/VERSION ('b11398'), the git HEAD of third_party/llama.cpp
    and of $LLAMA_DIR. Returns {expected_build, expected_commit, checked, ok, sources:[{source,
    found_build|found_commit, ok}]}; ok is True when nothing could be checked.
    """
    sources: list[dict[str, Any]] = []
    vf = LLAMA_WIN_DIR / "VERSION"
    if vf.is_file():
        try:
            txt = vf.read_text(encoding="utf-8", errors="replace").strip()
            m = re.search(r"(\d+)", txt)
            found = int(m.group(1)) if m else None
            sources.append({"source": str(vf), "found_build": found, "ok": found == LLAMA_BUILD})
        except OSError as e:
            sources.append({"source": str(vf), "found_build": None, "ok": False, "error": str(e)})
    seen: set[str] = set()
    for d in (LLAMA_SRC_DIR, llama_dir()):
        key = os.path.normcase(str(d))
        if key in seen or not (d / ".git").exists():
            continue
        seen.add(key)
        sha = _git_head(d)
        if sha:
            sources.append(
                {"source": str(d / ".git"), "found_commit": sha[:12], "ok": sha.startswith(LLAMA_COMMIT)}
            )
    return {
        "expected_build": LLAMA_BUILD,
        "expected_commit": LLAMA_COMMIT,
        "checked": bool(sources),
        "ok": all(s["ok"] for s in sources),
        "sources": sources,
    }


def _maybe_check_version() -> None:
    global _VERSION_CHECKED
    if _VERSION_CHECKED:
        return
    _VERSION_CHECKED = True
    res = llama_version_check()
    for s in res["sources"]:
        if not s["ok"]:
            log.warning(
                "llama.cpp version mismatch at %s: found %s, expected build %d / commit %s",
                s["source"], s.get("found_build", s.get("found_commit")), LLAMA_BUILD, LLAMA_COMMIT,
            )


# --------------------------------------------------------------------------------------------
# Host fingerprint
# --------------------------------------------------------------------------------------------
_ARM_PARTS = {
    "0xd03": "Cortex-A53", "0xd04": "Cortex-A35", "0xd05": "Cortex-A55", "0xd07": "Cortex-A57",
    "0xd08": "Cortex-A72", "0xd09": "Cortex-A73", "0xd0a": "Cortex-A75", "0xd0b": "Cortex-A76",
    "0xd0c": "Neoverse-N1", "0xd0d": "Cortex-A77", "0xd40": "Neoverse-V1", "0xd41": "Cortex-A78",
    "0xd44": "Cortex-X1", "0xd46": "Cortex-A510", "0xd47": "Cortex-A710", "0xd49": "Neoverse-N2",
    "0xd4b": "Cortex-A78C", "0xd4f": "Neoverse-V2",
}


def _read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _cpuinfo_fields() -> list[tuple[str, str]]:
    txt = _read_text(PROC_CPUINFO) or ""
    out = []
    for line in txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out.append((k.strip(), v.strip()))
    return out


def cpu_name() -> str:
    """Human CPU name: Windows registry, /proc/cpuinfo 'model name' or ARM part + board, else platform."""
    if is_windows():
        try:
            import winreg  # noqa: PLC0415 - Windows only

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as k:
                v, _ = winreg.QueryValueEx(k, "ProcessorNameString")
                if str(v).strip():
                    return str(v).strip()
        except Exception:  # noqa: BLE001 - fall through to platform.processor()
            pass
        return platform.processor() or platform.machine() or "unknown"
    fields = _cpuinfo_fields()
    for k, v in fields:
        if k.lower() == "model name" and v:
            return v
    d = {k.lower(): v for k, v in fields}
    part = d.get("cpu part", "").lower()
    board = d.get("model") or (_read_text(Path("/proc/device-tree/model")) or "").strip("\x00 \n")
    if part:
        core = _ARM_PARTS.get(part, f"ARM part {part}")
        return f"{core} ({board})" if board else core
    return platform.processor() or board or platform.machine() or "unknown"


def _physical_cores() -> int | None:
    if psutil is not None:
        try:
            n = psutil.cpu_count(logical=False)
            if n:
                return int(n)
        except Exception:  # noqa: BLE001
            pass
    pairs: set[tuple[str, str]] = set()
    phys = core = None
    for k, v in _cpuinfo_fields():
        if k == "physical id":
            phys = v
        elif k == "core id":
            core = v
        if phys is not None and core is not None:
            pairs.add((phys, core))
            phys = core = None
    return len(pairs) if pairs else os.cpu_count()


def ram_bytes() -> tuple[int | None, int | None]:
    """(total, available) system RAM in bytes: psutil, /proc/meminfo, Win32 API or sysconf."""
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return int(vm.total), int(vm.available)
        except Exception:  # noqa: BLE001
            pass
    txt = None if is_windows() else _read_text(PROC_MEMINFO)
    if txt:
        kv: dict[str, int] = {}
        for line in txt.splitlines():
            m = re.match(r"^(\w+):\s+(\d+)", line)
            if m:
                kv[m.group(1)] = int(m.group(2)) * 1024
        total = kv.get("MemTotal")
        avail = kv.get("MemAvailable")
        if avail is None and "MemFree" in kv:
            avail = kv["MemFree"] + kv.get("Buffers", 0) + kv.get("Cached", 0)
        if total:
            return total, avail
    if is_windows():
        try:
            import ctypes  # noqa: PLC0415

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            ms = _MemStatus()
            ms.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):  # type: ignore[attr-defined]
                return int(ms.ullTotalPhys), int(ms.ullAvailPhys)
        except Exception:  # noqa: BLE001
            pass
    try:
        ps = os.sysconf("SC_PAGE_SIZE")
        return int(os.sysconf("SC_PHYS_PAGES") * ps), int(os.sysconf("SC_AVPHYS_PAGES") * ps)
    except (AttributeError, ValueError, OSError):
        return None, None


def page_size() -> int:
    """Runtime page size (16384 on the Pi 5 kernel, 4096 elsewhere); 4096 when unknown (Windows)."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 4096


def cgroup_mem_max() -> int | None:
    """cgroup memory limit in bytes (v2 memory.max, v1 limit_in_bytes); None if absent/unlimited."""
    txt = _read_text(CGROUP_V2_MEMORY_MAX)
    if txt is not None:
        t = txt.strip()
        if not t or t == "max":
            return None
        try:
            return int(t)
        except ValueError:
            return None
    txt = _read_text(CGROUP_V1_MEMORY_LIMIT)
    if txt is not None:
        try:
            v = int(txt.strip())
            return None if v >= (1 << 60) else v
        except ValueError:
            return None
    return None


def _os_name() -> str:
    sysname = platform.system() or sys.platform
    if is_windows():
        ed = ""
        with contextlib.suppress(Exception):
            ed = platform.win32_edition() or ""
        return " ".join(x for x in ("Windows", platform.release(), ed) if x)
    pretty = None
    for f in (Path("/etc/os-release"), Path("/usr/lib/os-release")):
        txt = _read_text(f)
        if txt:
            m = re.search(r'^PRETTY_NAME="?([^"\n]*)"?', txt, re.MULTILINE)
            if m:
                pretty = m.group(1)
                break
    return f"{sysname} ({pretty})" if pretty else sysname


def host_fingerprint() -> dict[str, Any]:
    """Host description for the results envelope (data_formats 'results JSON envelope' host block).

    Keys: hostname, os, kernel, cpu, n_logical, n_physical, ram_total_bytes, ram_avail_bytes,
    page_size, in_container, cgroup_mem_max, python, plus machine ('x86_64'/'aarch64') and
    emulated (env PIGEMMA_EMULATED=1, set for the emulated-Pi container runs).
    """
    total, avail = ram_bytes()
    kernel = platform.version() if is_windows() else platform.release()
    return {
        "hostname": socket.gethostname(),
        "os": _os_name(),
        "kernel": kernel,
        "cpu": cpu_name(),
        "n_logical": os.cpu_count(),
        "n_physical": _physical_cores(),
        "ram_total_bytes": total,
        "ram_avail_bytes": avail,
        "page_size": page_size(),
        "in_container": in_container(),
        "cgroup_mem_max": None if is_windows() else cgroup_mem_max(),
        "python": platform.python_version(),
        "machine": machine(),
        "emulated": os.environ.get("PIGEMMA_EMULATED", "").strip() == "1",
    }


# --------------------------------------------------------------------------------------------
# Files and disk
# --------------------------------------------------------------------------------------------
def sha256_prefix(path: str | os.PathLike[str], n_bytes: int = SHA_PREFIX_BYTES) -> dict[str, Any]:
    """Cheap identity of a big file: sha256 of its first ``n_bytes`` plus its size.

    Returns {'sha256': hex, 'prefix_bytes': bytes actually hashed (= min(n_bytes, size)),
    'size': file size in bytes, 'path': str(path)}.
    """
    if n_bytes <= 0:
        raise ValueError("n_bytes must be > 0")
    p = Path(os.fspath(path))
    size = p.stat().st_size
    h = hashlib.sha256()
    done = 0
    with open(p, "rb") as f:
        while done < n_bytes:
            chunk = f.read(min(MiB, n_bytes - done))
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
    return {"sha256": h.hexdigest(), "prefix_bytes": done, "size": size, "path": str(p)}


def free_disk_bytes(path: str | os.PathLike[str] = ".") -> int:
    """Free bytes on the filesystem holding ``path`` (nearest existing parent if it does not exist)."""
    p = Path(os.fspath(path)).expanduser().absolute()
    while not p.exists() and p.parent != p:
        p = p.parent
    return int(shutil.disk_usage(p).free)


def to_container(path: str | os.PathLike[str]) -> str:
    """Translate a path inside the repo to its path inside the Docker image (/work/...)."""
    p = Path(os.fspath(path)).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    try:
        rel = p.resolve().relative_to(REPO_ROOT)
    except ValueError as e:
        raise ValueError(f"{p} is not inside the repo {REPO_ROOT}") from e
    return CONTAINER_ROOT if str(rel) == "." else f"{CONTAINER_ROOT}/{rel.as_posix()}"


# --------------------------------------------------------------------------------------------
# Bench lock
# --------------------------------------------------------------------------------------------
def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _boot_id() -> str | None:
    if is_windows():
        return None
    txt = _read_text(Path("/proc/sys/kernel/random/boot_id"))
    return txt.strip() if txt else None


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid exists on this host (zombies count as dead)."""
    if pid <= 0:
        return False
    if psutil is not None:
        try:
            if not psutil.pid_exists(pid):
                return False
            try:
                return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                return False
            except psutil.AccessDenied:
                return True
        except Exception:  # noqa: BLE001 - fall back to the OS primitive
            pass
    if is_windows():
        try:
            import ctypes  # noqa: PLC0415

            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return k32.GetLastError() == 5  # ERROR_ACCESS_DENIED -> exists
            try:
                code = ctypes.c_ulong()
                if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return True
                return code.value == 259  # STILL_ACTIVE
            finally:
                k32.CloseHandle(h)
        except Exception:  # noqa: BLE001
            return True  # unknown -> assume alive (the safe side for a lock)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    stat = _read_text(Path(f"/proc/{pid}/stat"))
    if stat and ")" in stat:
        fields = stat.rsplit(")", 1)[1].split()
        if fields and fields[0] == "Z":
            return False
    return True


def _pid_start_marker(pid: int) -> dict[str, Any] | None:
    """Process start-time marker used to detect pid reuse ({'kind', 'value'}) or None."""
    if psutil is not None:
        try:
            return {"kind": "psutil", "value": round(float(psutil.Process(pid).create_time()), 2)}
        except Exception:  # noqa: BLE001
            pass
    stat = _read_text(Path(f"/proc/{pid}/stat")) if not is_windows() else None
    if stat and ")" in stat:
        fields = stat.rsplit(")", 1)[1].split()
        if len(fields) > 19:
            return {"kind": "procstat", "value": fields[19]}  # field 22: starttime in clock ticks
    return None


def _same_start(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict) or a.get("kind") != b.get("kind"):
        return True  # cannot tell -> trust the pid
    if a.get("kind") == "psutil":
        try:
            return abs(float(a["value"]) - float(b["value"])) <= 1.0
        except (KeyError, TypeError, ValueError):
            return True
    return str(a.get("value")) == str(b.get("value"))


def _me() -> dict[str, Any]:
    return {"host": socket.gethostname(), "in_container": in_container(), "boot_id": _boot_id()}


def _default_lock_path() -> Path:
    env = os.environ.get("PIGEMMA_LOCK_FILE", "").strip()
    return Path(env).expanduser() if env else LOCK_FILE


def _stale_after(stale_s: float | None) -> float:
    if stale_s is not None:
        return float(stale_s)
    env = os.environ.get("PIGEMMA_LOCK_STALE_S", "").strip()
    try:
        return float(env) if env else DEFAULT_STALE_S
    except ValueError:
        return DEFAULT_STALE_S


def read_lock(lock_path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """Parsed lock file; None when absent; {'_unparsable': True} when it is not a JSON object."""
    p = Path(os.fspath(lock_path)) if lock_path else _default_lock_path()
    try:
        txt = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return {"_unparsable": True}
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        return {"_unparsable": True}
    return d if isinstance(d, dict) else {"_unparsable": True}


def lock_status(
    lock_path: str | os.PathLike[str] | None = None, stale_s: float | None = None
) -> dict[str, Any]:
    """Assess the lock: {'path', 'state', 'reason', 'holder', 'age_s'}.

    state is 'free' (no file), 'held' (live holder), 'stale' (dead holder, will be replaced) or
    'inherited' (live holder whose token equals env PIGEMMA_LOCK_HELD, i.e. we are its child).
    age_s is the seconds since the holder's last heartbeat (lock file mtime).
    """
    p = Path(os.fspath(lock_path)) if lock_path else _default_lock_path()
    out: dict[str, Any] = {"path": str(p), "state": "free", "reason": "no lock file", "holder": None, "age_s": None}
    info = read_lock(p)
    if info is None:
        return out
    try:
        age = max(0.0, time.time() - p.stat().st_mtime)
    except FileNotFoundError:
        return out
    out["holder"] = info
    out["age_s"] = round(age, 1)
    limit = _stale_after(stale_s)
    if info.get("_unparsable"):
        if age > limit:
            out.update(state="stale", reason=f"unparsable lock, heartbeat age {age:.0f}s > {limit:.0f}s")
        else:
            out.update(state="held", reason="unparsable lock (being written?)")
        return out
    me = _me()
    same_host = (
        info.get("host") == me["host"]
        and bool(info.get("in_container")) == me["in_container"]
        and (not info.get("boot_id") or not me["boot_id"] or info.get("boot_id") == me["boot_id"])
    )
    try:
        pid = int(info.get("pid", -1))
    except (TypeError, ValueError):
        pid = -1
    if same_host:
        if not _pid_alive(pid):
            out.update(state="stale", reason=f"holder pid {pid} on this host is not running")
        elif not _same_start(info.get("pid_start"), _pid_start_marker(pid)):
            out.update(state="stale", reason=f"pid {pid} was reused by another process")
        else:
            out.update(state="held", reason=f"holder pid {pid} on this host is alive")
    elif age > limit:
        out.update(
            state="stale",
            reason=f"holder on other host {info.get('host')!r}, heartbeat age {age:.0f}s > {limit:.0f}s",
        )
    else:
        out.update(
            state="held",
            reason=f"holder on other host {info.get('host')!r}, heartbeat age {age:.0f}s <= {limit:.0f}s",
        )
    held_token = os.environ.get("PIGEMMA_LOCK_HELD", "").strip()
    if out["state"] == "held" and held_token and info.get("token") == held_token:
        out.update(state="inherited", reason=out["reason"] + "; token matches PIGEMMA_LOCK_HELD")
    return out


def _busy_message(st: dict[str, Any]) -> str:
    h = st.get("holder") or {}
    return (
        "another model-loading job is running "
        f"(pid {h.get('pid')} on {h.get('host')!r}, started {h.get('started')}, "
        f"cmd {str(h.get('cmd'))[:200]!r}; {st.get('reason')}; lock {st.get('path')})"
    )


def _effective_min_free_gb(arg: float) -> float:
    env = os.environ.get("PIGEMMA_MIN_FREE_GB", "").strip()
    if env:
        try:
            v = float(env)
        except ValueError:
            log.warning("lock: ignoring invalid PIGEMMA_MIN_FREE_GB=%r", env)
        else:
            if v != float(arg):
                log.info("lock: PIGEMMA_MIN_FREE_GB=%s overrides min_free_gb=%s", env, arg)
            return v
    return float(arg)


class _Heartbeat(threading.Thread):
    """Touches the lock file every ``interval`` seconds so other hosts can see the holder is alive."""

    def __init__(self, path: Path, interval: float) -> None:
        super().__init__(name="pigemma-lock-heartbeat", daemon=True)
        self._path = path
        self._interval = interval
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.wait(self._interval):
            with contextlib.suppress(OSError):
                os.utime(self._path, None)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=5)


def _acquire(path: Path, info: dict[str, Any], stale_s: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(6):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            st = lock_status(path, stale_s)
            if st["state"] == "free":
                continue  # vanished between open() and the read
            if st["state"] != "stale":
                log.info("lock: refused, %s", st["reason"])
                raise LockBusyError(_busy_message(st)) from None
            # Re-read right before removing so a lock that was just replaced is never deleted.
            again = read_lock(path)
            if again is not None and again.get("token") != (st["holder"] or {}).get("token"):
                continue
            log.info("lock: replacing stale lock %s (%s)", path, st["reason"])
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
            f.write("\n")
            f.flush()
            with contextlib.suppress(OSError):
                os.fsync(f.fileno())
        return
    raise LockBusyError(f"could not acquire {path} after 6 attempts (contention)")


def _release(path: Path, token: str) -> None:
    cur = read_lock(path)
    if cur is None:
        log.warning("lock: %s already gone at release", path)
        return
    if cur.get("token") != token:
        log.warning("lock: %s was replaced by another holder; not removing it", path)
        return
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    log.info("lock: released %s", path)


@contextlib.contextmanager
def bench_lock(
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    *,
    cmd: str | None = None,
    lock_path: str | os.PathLike[str] | None = None,
    stale_s: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Serialise model-loading jobs: hold results/.bench.lock for the duration of the block.

    Raises LockBusyError ('another model-loading job is running ...') when a live job holds the
    lock, InsufficientMemoryError when available RAM < min_free_gb * 1e9 (env PIGEMMA_MIN_FREE_GB
    overrides the argument). Both subclass BenchLockError(RuntimeError). A stale lock (dead pid on
    this host, or no heartbeat for PIGEMMA_LOCK_STALE_S seconds from another host) is replaced.
    Lock path: ``lock_path`` or env PIGEMMA_LOCK_FILE or results/.bench.lock. Yields the lock record
    {pid, host, started, cmd, token, in_container, boot_id, platform, pid_start, min_free_gb,
    ram_avail_bytes, heartbeat_s}; an inherited lock yields the parent's record with
    'inherited': True and is neither rewritten nor removed.
    """
    path = Path(os.fspath(lock_path)) if lock_path else _default_lock_path()
    need_gb = _effective_min_free_gb(min_free_gb)
    st = lock_status(path, stale_s)
    if st["state"] == "inherited":
        log.info("lock: inherited from parent (%s)", st["reason"])
        yield {**(st["holder"] or {}), "inherited": True}
        return
    if st["state"] == "held":
        log.info("lock: refused, %s", st["reason"])
        raise LockBusyError(_busy_message(st))
    total, avail = ram_bytes()
    cg = None if is_windows() else cgroup_mem_max()
    if avail is None:
        log.warning("lock: available RAM unknown; skipping the %.1f GB free-RAM check", need_gb)
    elif avail < need_gb * GB:
        msg = (
            f"only {avail / GB:.1f} GB RAM available (total {(total or 0) / GB:.1f} GB), "
            f"need >= {need_gb:g} GB to start a model-loading job (env PIGEMMA_MIN_FREE_GB overrides)"
        )
        log.info("lock: refused, %s", msg)
        raise InsufficientMemoryError(msg)
    else:
        log.info(
            "lock: RAM ok, %.1f GB available >= %.1f GB required%s",
            avail / GB, need_gb, f" (cgroup memory.max {cg / GB:.1f} GB)" if cg else "",
        )
    pid = os.getpid()
    me = _me()
    info: dict[str, Any] = {
        "pid": pid,
        "host": me["host"],
        "started": _utc_now(),
        "cmd": cmd if cmd is not None else shlex.join(sys.argv),
        "token": uuid.uuid4().hex,
        "in_container": me["in_container"],
        "boot_id": me["boot_id"],
        "platform": sys.platform,
        "pid_start": _pid_start_marker(pid),
        "min_free_gb": need_gb,
        "ram_avail_bytes": avail,
        "heartbeat_s": HEARTBEAT_S,
    }
    _acquire(path, info, stale_s)
    log.info("lock: acquired %s (pid %d, cmd %s)", path, pid, str(info["cmd"])[:200])
    hb = _Heartbeat(path, HEARTBEAT_S)
    hb.start()
    try:
        yield info
    finally:
        hb.stop()
        _release(path, info["token"])


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def git_bash() -> Path | None:
    """Git Bash on Windows (env PIGEMMA_BASH, next to git.exe, or Program Files), else None.

    Needed because a bare ``bash`` launched from PowerShell resolves to System32\\bash.exe (WSL),
    which cannot run pc/*.sh (they rely on Git Bash path handling and Docker Desktop).
    """
    cands: list[Path] = []
    env = os.environ.get("PIGEMMA_BASH", "").strip()
    if env:
        cands.append(Path(env).expanduser())
    git = shutil.which("git")
    if git:
        g = Path(git).resolve()
        cands += [g.parent.parent / "bin" / "bash.exe", g.parent.parent / "usr" / "bin" / "bash.exe"]
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramW6432", "")):
        if base:
            cands.append(Path(base) / "Git" / "bin" / "bash.exe")
    for c in cands:
        if c.is_file():
            return c
    return None


def _resolve_exe(exe: str) -> str:
    """PATH lookup for a bare command name; on Windows map bare bash/sh to Git Bash, not WSL."""
    if os.sep in exe or (os.altsep is not None and os.altsep in exe):
        return exe
    found = shutil.which(exe)
    if is_windows() and _strip_exe(exe).lower() in ("bash", "sh"):
        sysroot = os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows"))
        if not found or os.path.normcase(found).startswith(sysroot):
            gb = git_bash()
            if gb is not None:
                log.info("hold: using Git Bash %s instead of %s", gb, found or "(not on PATH)")
                return str(gb)
    if found and os.path.normcase(found) != os.path.normcase(exe):
        log.info("hold: resolved %s -> %s (PATH order)", exe, found)
    return found or exe


def _run_child(cmd: list[str], env: dict[str, str]) -> int:
    """Run cmd, forward SIGTERM/SIGHUP (POSIX), return its exit code (128+N on signal N)."""
    cmd = [_resolve_exe(cmd[0])] + cmd[1:]
    try:
        proc = subprocess.Popen(cmd, env=env)
    except FileNotFoundError:
        sys.stderr.write(f"pigemma.paths: command not found: {cmd[0]}\n")
        return 127
    except OSError as e:
        sys.stderr.write(f"pigemma.paths: cannot execute {cmd[0]}: {e}\n")
        return 126
    log.info("hold: started child pid %d: %s", proc.pid, shlex.join(cmd)[:200])
    previous: dict[int, Any] = {}
    if os.name == "posix":

        def _forward(signum: int, _frame: Any) -> None:
            with contextlib.suppress(OSError):
                proc.send_signal(signum)

        for sig in (signal.SIGTERM, signal.SIGHUP):
            with contextlib.suppress(ValueError, OSError):
                previous[sig] = signal.signal(sig, _forward)
    interrupted = False
    rc: int | None = None
    try:
        while rc is None:
            try:
                rc = proc.wait(timeout=30) if interrupted else proc.wait()
            except KeyboardInterrupt:
                if interrupted:
                    break
                interrupted = True  # the console delivered Ctrl-C to the child as well
                log.info("hold: interrupted; waiting up to 30 s for child pid %d", proc.pid)
            except subprocess.TimeoutExpired:
                break
        if rc is None:  # our own child did not exit after the interrupt
            log.info("hold: terminating child pid %d", proc.pid)
            proc.terminate()
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
    finally:
        for sig, h in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, h)
    if rc < 0:
        rc = 128 + (-rc)
    if interrupted and rc == 0:
        rc = 130
    log.info("hold: child exited with %d", rc)
    return rc


def _cmd_lock(args: argparse.Namespace, child: list[str]) -> int:
    lock_path = Path(args.lock_file).expanduser() if args.lock_file else _default_lock_path()
    if args.status:
        st = lock_status(lock_path)
        print(json.dumps(st, indent=1, default=str))
        return EXIT_OK if st["state"] in ("free", "stale") else EXIT_ERROR
    if not args.hold:
        sys.stderr.write("pigemma.paths lock: use --hold -- <cmd...> or --status\n")
        return EXIT_USAGE
    if not child:
        sys.stderr.write("pigemma.paths lock --hold: missing command after '--'\n")
        return EXIT_USAGE
    min_gb = DEFAULT_MIN_FREE_GB if args.min_free_gb is None else args.min_free_gb
    try:
        with bench_lock(min_gb, cmd=shlex.join(child), lock_path=lock_path) as info:
            env = dict(os.environ)
            env["PIGEMMA_LOCK_HELD"] = str(info.get("token", ""))
            env["PIGEMMA_LOCK_FILE"] = str(lock_path)
            return _run_child(child, env)
    except BenchLockError as e:
        sys.stderr.write(f"pigemma.paths: REFUSED: {e}\n")
        return EXIT_REFUSED


def _info() -> dict[str, Any]:
    models = {}
    for k, v in MODELS.items():
        p = model_path(k)
        size = p.stat().st_size if p.is_file() else None
        models[k] = {"path": str(p), "exists": size is not None, "size_ok": size == v["bytes"], "kind": v["kind"]}
    tools = ["llama-bench", "llama-completion", "llama-cli", "llama-server", "llama-perplexity",
             "pigemma-probe", "pinner", "membw"]
    bins = {}
    for t in tools:
        p = bin(t)
        bins[t] = {"path": str(p), "exists": p.is_file()}
    return {
        "repo_root": str(REPO_ROOT),
        "llama": llama_version_check(),
        "host": host_fingerprint(),
        "models": models,
        "bins": bins,
        "lock": lock_status(),
    }


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m pigemma.paths",
        description="Repo paths, model registry, binaries, host fingerprint and the bench lock.",
        epilog="Exit codes: 0 ok, 1 error (lock --status: held), 2 usage, 3 missing, 4 refused; "
        "lock --hold forwards the child's exit code.",
    )
    ap.add_argument("--log-level", default="INFO", help="logging level on stderr (default INFO)")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")
    sub.add_parser("info", help="JSON summary: repo, llama version, host, models, binaries, lock")
    sub.add_parser("host", help="JSON host_fingerprint()")
    m = sub.add_parser("model", help="print a model path (id or path)")
    m.add_argument("model", help="registry id (" + ", ".join(MODELS) + ") or path")
    m.add_argument("--require", action="store_true", help="exit 3 if missing or the size differs")
    m.add_argument("--json", action="store_true", help="print the results-envelope model block")
    m.add_argument("--no-sha", action="store_true", help="with --json: skip the 16 MiB sha256 prefix")
    b = sub.add_parser("bin", help="print the resolved binary path of a tool")
    b.add_argument("tool", help="llama.cpp tool (llama-bench, ...) or pigemma-probe|pinner|membw")
    b.add_argument("--require", action="store_true", help="exit 3 if the binary does not exist")
    s = sub.add_parser("sha", help="sha256 of the first N bytes + size (JSON)")
    s.add_argument("target", help="registry id or file path")
    s.add_argument("--n-bytes", type=int, default=SHA_PREFIX_BYTES, help="prefix length (default 16 MiB)")
    f = sub.add_parser("free-disk", help="free bytes on the filesystem holding PATH")
    f.add_argument("path", nargs="?", default=str(REPO_ROOT))
    f.add_argument("--gb", action="store_true", help="print GB (1e9) with 2 decimals instead of bytes")
    c = sub.add_parser("container-path", help="repo path(s) -> /work/... inside the Docker image")
    c.add_argument("paths", nargs="+")
    lk = sub.add_parser(
        "lock",
        help="bench lock: --status, or --hold -- <cmd...> (runs cmd under the lock, forwards its exit code)",
    )
    g = lk.add_mutually_exclusive_group()
    g.add_argument("--hold", action="store_true", help="acquire, run the command after '--', release")
    g.add_argument("--status", action="store_true", help="print the lock state; exit 0 free/stale, 1 held")
    lk.add_argument(
        "--min-free-gb", type=float, default=None,
        help=f"free-RAM gate in GB = 1e9 B (default {DEFAULT_MIN_FREE_GB:g}; env PIGEMMA_MIN_FREE_GB wins)",
    )
    lk.add_argument(
        "--lock-file", default=None, help="lock path (default env PIGEMMA_LOCK_FILE or results/.bench.lock)"
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    argv = list(sys.argv[1:] if argv is None else argv)
    child: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, child = argv[:i], argv[i + 1:]
    ap = _build_parser()
    args = ap.parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, str(args.log_level).upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
    if args.cmd is None:
        ap.print_help()
        return EXIT_USAGE
    if child and args.cmd != "lock":
        ap.error("'-- <cmd>' is only valid with 'lock --hold'")
    try:
        if args.cmd == "lock":
            return _cmd_lock(args, child)
        if args.cmd == "info":
            print(json.dumps(_info(), indent=1, default=str))
            return EXIT_OK
        if args.cmd == "host":
            print(json.dumps(host_fingerprint(), indent=1))
            return EXIT_OK
        if args.cmd == "model":
            if args.require:
                try:
                    require_model(args.model)
                except FileNotFoundError as e:
                    sys.stderr.write(f"pigemma.paths: {e}\n")
                    return EXIT_MISSING
            if args.json:
                print(json.dumps(model_identity(args.model, with_sha=not args.no_sha), indent=1))
            else:
                print(model_path(args.model))
            return EXIT_OK
        if args.cmd == "bin":
            p = bin(args.tool)
            print(p)
            if args.require and not p.is_file():
                sys.stderr.write(f"pigemma.paths: binary not found: {p}\n")
                return EXIT_MISSING
            return EXIT_OK
        if args.cmd == "sha":
            p = model_path(args.target)
            if not p.is_file():
                sys.stderr.write(f"pigemma.paths: file not found: {p}\n")
                return EXIT_MISSING
            print(json.dumps(sha256_prefix(p, args.n_bytes), indent=1))
            return EXIT_OK
        if args.cmd == "free-disk":
            n = free_disk_bytes(args.path)
            print(f"{n / GB:.2f}" if args.gb else n)
            return EXIT_OK
        if args.cmd == "container-path":
            for p in args.paths:
                print(to_container(p))
            return EXIT_OK
    except ValueError as e:
        sys.stderr.write(f"pigemma.paths: {e}\n")
        return EXIT_USAGE
    except Exception as e:  # noqa: BLE001 - CLI boundary
        sys.stderr.write(f"pigemma.paths: error: {type(e).__name__}: {e}\n")
        return EXIT_ERROR
    ap.print_help()
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
