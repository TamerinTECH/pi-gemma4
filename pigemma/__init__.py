"""pigemma - measurement harness for Gemma 4 (26B-A4B MoE, 31B dense) on a Raspberry Pi 5.

Every tool is run as ``python -m pigemma.<module>`` from the repository root (``python3`` on the
Pi); nothing is installed into the global interpreter (see docs/design/FACTS.md).

This package init is deliberately stdlib-only and import-cheap: ``pigemma.feasibility`` runs on a
stock Pi Python 3.13 that may have no numpy, so importing ``pigemma`` must not pull in numpy,
gguf or psutil.

Its single side effect makes ``gguf`` (gguf-py, which ships inside llama.cpp and is deliberately
not a pip dependency) importable. When ``importlib.util.find_spec("gguf")`` finds nothing, the
first existing directory among

1. ``$PIGEMMA_GGUF_PY``                          (explicit override)
2. ``<repo>/third_party/llama.cpp/gguf-py``     (desktop, pinned llama.cpp commit a7b94df)
3. ``$LLAMA_DIR/gguf-py``                        (Pi, after pi/02_build_llama.sh)
4. ``~/llama.cpp/gguf-py``                       (Pi default LLAMA_DIR)

that contains ``gguf/__init__.py`` is inserted at ``sys.path[0]``. ``find_spec`` is used instead
of a trial ``import gguf`` so that importing pigemma never imports numpy. Note: gguf-py itself
needs numpy and pyyaml at import time (see pyproject.toml).

Public interface:
    __version__             package version (single source; pyproject reads it statically)
    GGUF_PY_DIR             Path inserted into sys.path at import, or None (gguf already
                            importable, or no candidate directory exists)
    gguf_py_candidates()    ordered candidate directories (existing or not)
    ensure_gguf_importable() -> Path | None   idempotent; what runs at import time
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

__version__ = "0.1.0"

__all__ = ["__version__", "GGUF_PY_DIR", "gguf_py_candidates", "ensure_gguf_importable"]

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent


def gguf_py_candidates() -> list[Path]:
    """Return the ordered list of directories that may hold the ``gguf`` package (gguf-py).

    Environment is read at call time, so tests can monkeypatch ``PIGEMMA_GGUF_PY`` and
    ``LLAMA_DIR``. Directories are returned whether or not they exist.
    """
    out: list[Path] = []
    override = os.environ.get("PIGEMMA_GGUF_PY", "").strip()
    if override:
        out.append(Path(override).expanduser())
    out.append(_REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py")
    llama_dir = os.environ.get("LLAMA_DIR", "").strip()
    if llama_dir:
        out.append(Path(llama_dir).expanduser() / "gguf-py")
    out.append(Path("~/llama.cpp/gguf-py").expanduser())
    # De-duplicate while keeping order (LLAMA_DIR often equals ~/llama.cpp on the Pi).
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = os.path.normcase(str(p))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _gguf_findable() -> bool:
    """True if ``import gguf`` would find a module, without importing it."""
    if "gguf" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("gguf") is not None
    except (ImportError, ValueError):  # broken partial installs
        return False


def ensure_gguf_importable() -> Path | None:
    """Make ``gguf`` importable if it is not already; return the directory added to sys.path.

    Returns None when gguf is already findable or when no candidate directory contains
    ``gguf/__init__.py`` (callers that need gguf then get the normal ImportError and should
    point the user at PIGEMMA_GGUF_PY / LLAMA_DIR). Never raises.
    """
    if _gguf_findable():
        return None
    for cand in gguf_py_candidates():
        try:
            if (cand / "gguf" / "__init__.py").is_file():
                entry = str(cand)
                if entry in sys.path:
                    sys.path.remove(entry)
                sys.path.insert(0, entry)
                importlib.invalidate_caches()
                return cand
        except OSError:  # unreadable candidate (permissions, dead network path)
            continue
    return None


GGUF_PY_DIR: Path | None = ensure_gguf_importable()
