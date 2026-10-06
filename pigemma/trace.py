"""pigemma.trace - reference reader / writer / validator for ``.moetrace`` routing traces.

Normative format: docs/TRACE_FORMAT.md (sections 2-5 and 7). This module is the Python reference
implementation of that document. The C++ probe (tools/probe/probe.cpp) writes the same bytes, and
``pigemma.cache_sim``, ``pigemma.gguf_map hotset`` and ``pigemma.h2.oracle`` read through this module.

What is here
------------
* ``TraceWriter(path, header)``: writes the 32-byte fixed header (``n_records`` = sentinel while
  open), the JSON header with a 16 KiB reserve, fixed-size records and the ``<trace>.tokens.bin``
  sidecar. ``close()`` patches the header in place, or rewrites the file when the final header no
  longer fits, and writes ``n_records`` at offset 16 last (TRACE_FORMAT 3.5).
* ``TraceReader(path)``: memory-mapped reader with ``.header .n .truncated .ids .weights .phase
  .chunk .pos .token_index .token_id .token_ids .probs`` plus chunk ``segments()``.
* ``validate(path) -> list[str]``: errors E1..E14 only. Warnings W1..W7 go to logging.
  ``validate_report(path)`` returns both lists.
* ``merge(paths, out)``: concatenates traces, renumbers ``token_index`` and chunks, writes
  ``tokens.bin``, ``gen.txt`` (renumbered separators) and a merged ``meta.json`` (TRACE_FORMAT 3.11).
* ``stats(reader, train_frac=0.7) -> dict``: the H1 statistics of TRACE_FORMAT 5.2. These are
  frequency skew, coverage, the exact LRU stack-distance histogram and hit curve, token-adjacent
  reuse, working sets, prefill-vs-generated divergence, corpus overlap, the cross-layer predictor
  and the router-probability margins.
* Sidecars: ``write_meta / read_meta / build_meta / read_gen / read_tokens /
  parse_probe_summary``. Activation dumps (H2): ``read_acts_dir(dir) -> Acts`` and
  ``write_acts_dir`` / ``npy_header_f16`` (canonical 128-byte .npy header, TRACE_FORMAT 7.3).

CLI (``python -m pigemma.trace <command> --help``)::

    validate <trace>... [--json] [--quiet]            exit 1 iff any error
    info <trace> [--no-header]                        JSON: header + counts + sidecars
    stats <trace> [--train-frac 0.7] [--item layer-expert|expert] [--out stats.json]
          [--csv-dir DIR] [--no-mmap]
    merge <out> <in...> [--no-gen] [--no-meta]        globs are expanded; <out> is skipped as input
    meta <trace> [--gguf P] [--corpus F...] [--argv-json J] [--wall-ms N] [--summary-file F]
         [--host-json F] [--coherent auto|true|false|null] [--note T...] [--no-sha]
    acts <dir>                                        JSON summary of an activation dump directory

Exit codes: 0 ok; 1 error (validation errors, format errors, incompatible merge inputs);
2 usage; 3 missing input file.

Design decisions (the manifest leaves these open; each is also listed in the build report):

1. **Validator extensions.** Two extra *warnings* are reported (never errors, so they do not change
   the exit code): W8, the fixed-header ``reserved`` field is non-zero; W9, the
   ``tokens.bin`` content differs from the records' ``token_id`` (TRACE_FORMAT 4.2 requires
   them to be equal).
2. **Default header keys.** ``TraceWriter`` completes a partial header with ``make_header``
   defaults. Only ``n_expert``, ``n_expert_used`` and ``moe_layers`` are mandatory. The complete
   header must pass the E3/E4 checks, otherwise ``ValueError``.
3. **Strict writing.** ``TraceWriter(strict=True)`` (the default) refuses a record whose
   ``token_index`` is not the next index, ids that are out of range or duplicated, a phase
   outside {0, 1}, and chunk ids out of sequence. ``strict=False`` writes anything that fits the
   field types; tests use it to build invalid traces for the validator.
4. **Chunk bookkeeping.** Unless the header passed to the writer already contains ``chunks``,
   the writer builds them itself. A new chunk starts whenever ``chunk_id`` changes, and
   ``begin_chunk(source)`` lets the caller name its source. The reader takes segments from the
   header when ``sum(n_prompt + n_gen) == N``. Otherwise it derives them from ``chunk_id``
   changes and ``pos == 0`` restarts, and marks them ``derived``.
5. **``token_ids``** comes from ``<trace>.tokens.bin`` when present. Otherwise it falls back to
   the records' ``token_id`` column, and ``.token_ids_source`` says which was used.
6. **Merge.** The output ``tokens.bin`` is rebuilt from the records' ``token_id``. By contract
   this equals the concatenated input ``tokens.bin``; a mismatching input logs W9. An input path
   equal to ``<out>`` is skipped, because ``merge all.moetrace traces/26b/*.moetrace`` matches
   it. Globs are expanded when the shell did not expand them (PowerShell). Truncated inputs are
   merged with a warning and a header note.
7. **``stats(item=...)``**: ``'layer-expert'`` (default, the contract: an item is (layer, expert),
   L*E items) or ``'expert'`` (expert id regardless of layer; a diagnostic only).
8. **Extra stats keys.** Readers ignore unknown keys (TRACE_FORMAT 1). Top level: ``n_train``,
   ``n_test``, ``n_expert``, ``n_expert_used``, ``moe_layers``, ``item``, ``truncated``.
   ``global``: ``n_accesses``, ``n_items``. ``cross_layer``: ``per_pair``, ``n_train``,
   ``n_test`` and ``mutual_info_bits_null_mean``. The plug-in MI of the contract is biased upward
   by several tenths of a bit at E = 128 and N ~ 16k, even for independent layers (the joint table
   is undersampled, so Miller-Madow does not fix it). The null is the same plug-in MI against a
   fixed-seed permutation of the next layer's top-1, i.e. the "no dependence" level at this N;
   report MI minus null.
9. **Stack distances** are exact. A batched offline algorithm costs
   O(n * (b + U log U / b)) for n accesses, U items and batch b = 256, which takes a few seconds
   for the ~4M accesses of the quick tier. It is cross-checked against a brute-force LRU in
   tests/test_trace.py.
10. **``meta`` subcommand.** ``pc/trace_collect.sh`` can write ``<trace>.meta.json`` from bash
    with it. ``coherent=auto`` uses ``pigemma.quality.is_coherent`` per gen.txt chunk when that
    module exists, and gives null with a note otherwise.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import dataclasses
import datetime as _dt
import functools
import glob
import hashlib
import json
import logging
import math
import os
import re
import struct
import sys
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pigemma import paths

log = logging.getLogger("pigemma.trace")

# --------------------------------------------------------------------------------------------
# Constants (mirror of the docs/TRACE_FORMAT.md section 2 block; tests compare them)
# --------------------------------------------------------------------------------------------
MOETRACE_MAGIC = b"MOETRC01"
MOETRACE_FORMAT_VERSION = 1
MOETRACE_OFF_MAGIC = 0
MOETRACE_OFF_HEADER_JSON_LEN = 8
MOETRACE_OFF_RECORD_SIZE = 12
MOETRACE_OFF_N_RECORDS = 16
MOETRACE_OFF_RESERVED = 24
MOETRACE_OFF_HEADER_JSON = 32
MOETRACE_FIXED_HEADER_BYTES = 32
MOETRACE_HEADER_JSON_ALIGN = 8
MOETRACE_HEADER_PAD_BYTE = b" "
MOETRACE_N_RECORDS_OPEN = 0xFFFFFFFFFFFFFFFF
MOETRACE_HEADER_RESERVE_BYTES = 16384
MOETRACE_RECORD_FIXED_BYTES = 16
MOETRACE_REC_OFF_TOKEN_INDEX = 0
MOETRACE_REC_OFF_POS = 4
MOETRACE_REC_OFF_TOKEN_ID = 8
MOETRACE_REC_OFF_PHASE = 12
MOETRACE_REC_OFF_CHUNK_ID = 13
MOETRACE_REC_OFF_RESERVED = 14
MOETRACE_REC_OFF_LAYERS = 16
MOETRACE_BYTES_PER_LAYER_SLOT = 3
MOETRACE_RECORD_SIZE_FORMULA = "16 + 3*L*K + 3*L*P"
MOETRACE_MAX_EXPERTS = 256
MOETRACE_WEIGHT_SUM_TOL = 0.02
MOETRACE_PHASE_PREFILL = 0
MOETRACE_PHASE_GENERATED = 1
MOETRACE_WEIGHTS_TENSOR = "ffn_moe_weights_norm"
MOETRACE_RECORD_SIZE_26B_P0 = 736
MOETRACE_RECORD_SIZE_26B_P16 = 2176
MOETRACE_BYTES_26B_100K_TOKENS = 73600000
SIDECAR_TOKENS = ".tokens.bin"
SIDECAR_GEN = ".gen.txt"
SIDECAR_META = ".meta.json"
GEN_CHUNK_SEPARATOR = "\n--- chunk {i} ---\n"
NPY_HEADER_BYTES = 128
NPY_MAGIC_AND_VERSION = b"\x93NUMPY\x01\x00"
NPY_HEADER_LEN_FIELD = 118
NPY_DESCR = "<f2"
ACTS_PHASE_FILE = "phase.u8"
ACTS_TOKENS_FILE = "tokens.bin"
ACTS_META_FILE = "meta.json"
PROBE_SUMMARY_PREFIX = "PIGEMMA_SUMMARY "
SHA256_PREFIX_BYTES = 16777216

#: Weight-order tolerance of warning W3 (TRACE_FORMAT 3.9).
WEIGHT_ORDER_TOL = 1e-3
#: stats.json defaults (TRACE_FORMAT 5.2).
COVERAGE_POINTS: tuple[int, ...] = (8, 16, 32, 48, 64, 96)
WORKING_SET_WINDOWS: tuple[int, ...] = (16, 64, 256, 1024)
CORPUS_TOP = 32
PREDICT_AT: tuple[int, ...] = (8, 16)
STATIC_BASELINE_TOP = 16
DEFAULT_TRAIN_FRAC = 0.7
#: Records per block when the validator / merge stream through a big trace.
_BLOCK_RECORDS = 1 << 16
#: Batch size of the stack-distance algorithm (see stack_distances).
_SD_BATCH = 256

_FIXED = struct.Struct("<8sIIQQ")
_GEN_SEP_RE_BYTES = re.compile(rb"\n--- chunk (\d+) ---\n")
_GEN_SEP_RE = re.compile(r"^--- chunk (\d+) ---$", re.M)

# Header schema: every key REQUIRED (TRACE_FORMAT 3.4, deviation D5), in document order.
HEADER_SCHEMA: tuple[tuple[str, str], ...] = (
    ("format_version", "int"),
    ("model_name", "str"),
    ("gguf_path", "str"),
    ("gguf_size_bytes", "int"),
    ("llama_build", "int"),
    ("llama_commit", "str"),
    ("arch", "str"),
    ("n_layer", "int"),
    ("n_expert", "int"),
    ("n_expert_used", "int"),
    ("moe_layers", "int_list"),
    ("n_embd", "int"),
    ("probs_top", "int"),
    ("weights_tensor", "str"),
    ("corpus_id", "str"),
    ("created_at", "str"),
    ("n_ctx", "int"),
    ("n_batch", "int"),
    ("n_ubatch", "int"),
    ("n_threads", "int"),
    ("flags", "str_list"),
    ("chunks", "chunks"),
)
REQUIRED_HEADER_KEYS: tuple[str, ...] = tuple(k for k, _ in HEADER_SCHEMA)
#: Keys that fix the record layout; they may not change between open and close.
STRUCTURAL_KEYS: tuple[str, ...] = ("n_expert", "n_expert_used", "moe_layers", "probs_top")
#: Keys merge inputs must agree on (TRACE_FORMAT 3.11).
MERGE_KEYS: tuple[str, ...] = (
    "arch", "n_layer", "n_expert", "n_expert_used", "moe_layers", "probs_top", "weights_tensor",
    "gguf_size_bytes",
)

__all__ = [
    "MOETRACE_MAGIC", "MOETRACE_N_RECORDS_OPEN", "MOETRACE_HEADER_RESERVE_BYTES",
    "REQUIRED_HEADER_KEYS", "HEADER_SCHEMA",
    "TraceFormatError", "TraceWarning", "MergeError",
    "record_size", "record_dtype", "make_header", "complete_header", "check_header",
    "commit_compatible", "sidecar_path", "tokens_path", "gen_path", "meta_path",
    "TraceWriter", "TraceReader", "read_header", "read_fixed_header",
    "ValidationReport", "validate", "validate_report",
    "merge", "trace_info",
    "stats", "access_items", "stack_distances", "stack_distance_hist", "lru_hit_curve",
    "coverage_curve", "entropy_bits", "gini", "js_divergence_bits", "mutual_info_bits",
    "cross_layer_table", "write_stats_csv",
    "write_meta", "read_meta", "build_meta", "read_gen", "read_tokens", "parse_probe_summary",
    "Acts", "read_acts_dir", "write_acts_dir", "npy_header_f16", "read_npy_header",
    "main",
]


class TraceFormatError(ValueError):
    """The file is not a readable ``.moetrace`` (bad magic, header or record size)."""


class TraceWarning(UserWarning):
    """Non-fatal trace condition (truncated trace, unpatched npy, ...)."""


class MergeError(ValueError):
    """Merge inputs are incompatible (message names the first differing key)."""


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def record_size(L: int, K: int, P: int = 0) -> int:
    """Record size in bytes: ``16 + 3*L*K + 3*L*P`` (TRACE_FORMAT 3.7)."""
    return MOETRACE_RECORD_FIXED_BYTES + 3 * int(L) * int(K) + 3 * int(L) * int(P)


@functools.lru_cache(maxsize=64)
def record_dtype(L: int, K: int, P: int = 0) -> np.dtype:
    """Packed NumPy dtype of one record (TRACE_FORMAT 3.6); ``itemsize == record_size(L, K, P)``."""
    fields: list[Any] = [
        ("token_index", "<u4"), ("pos", "<u4"), ("token_id", "<i4"),
        ("phase", "u1"), ("chunk_id", "u1"), ("reserved", "<u2"),
        ("moe", [("ids", "u1", (int(K),)), ("w", "<f2", (int(K),))], (int(L),)),
    ]
    if P > 0:
        fields.append(("probs", [("id", "u1"), ("p", "<f2")], (int(L), int(P))))
    dt = np.dtype(fields)
    if dt.itemsize != record_size(L, K, P):  # pragma: no cover - numpy packs by default
        raise AssertionError(f"record dtype itemsize {dt.itemsize} != {record_size(L, K, P)}")
    return dt


def _round_up_8(n: int) -> int:
    return (int(n) + 7) // 8 * 8


def utc_now_iso() -> str:
    """ISO 8601 UTC timestamp with ``Z`` suffix (TRACE_FORMAT 1)."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def commit_compatible(a: str, b: str, min_len: int = 7) -> bool:
    """Two llama commits are compatible when one is a prefix of the other (>= 7 hex digits)."""
    a, b = str(a).strip().lower(), str(b).strip().lower()
    if len(a) < min_len or len(b) < min_len:
        return False
    return a.startswith(b) or b.startswith(a)


def sidecar_path(trace: str | os.PathLike[str], suffix: str) -> Path:
    """``<trace><suffix>``: the full trace path plus the suffix (deviation D3)."""
    return Path(os.fspath(trace) + suffix)


def tokens_path(trace: str | os.PathLike[str]) -> Path:
    """``<trace>.tokens.bin``."""
    return sidecar_path(trace, SIDECAR_TOKENS)


def gen_path(trace: str | os.PathLike[str]) -> Path:
    """``<trace>.gen.txt``."""
    return sidecar_path(trace, SIDECAR_GEN)


def meta_path(trace: str | os.PathLike[str]) -> Path:
    """``<trace>.meta.json`` (a path that already ends with .meta.json is returned unchanged)."""
    s = os.fspath(trace)
    return Path(s) if s.endswith(SIDECAR_META) else Path(s + SIDECAR_META)


def _clean(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars/arrays -> Python, non-finite floats -> None, Paths -> str."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    return obj


def _dump_json(obj: Any, path: Path, *, indent: int | None = 1) -> None:
    """Atomically write UTF-8 JSON (no NaN) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(_clean(obj), ensure_ascii=False, allow_nan=False, indent=indent)
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _header_bytes(header: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _clean(dict(header)), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


# --------------------------------------------------------------------------------------------
# Header construction and checks
# --------------------------------------------------------------------------------------------
def make_header(*, n_expert: int, n_expert_used: int, moe_layers: Sequence[int], **fields: Any) -> dict:
    """Build a complete JSON header: defaults for every key of TRACE_FORMAT 3.4, then ``fields``.

    Defaults: format_version 1, llama_build/llama_commit pinned (11398 / a7b94df), arch
    'gemma4', n_layer = max(moe_layers) + 1, probs_top 0, weights_tensor 'ffn_moe_weights_norm',
    corpus_id 'prompt', created_at now, empty strings / zeros / lists elsewhere.
    """
    ml = [int(x) for x in moe_layers]
    h: dict[str, Any] = {
        "format_version": MOETRACE_FORMAT_VERSION,
        "model_name": "",
        "gguf_path": "",
        "gguf_size_bytes": 0,
        "llama_build": int(paths.LLAMA_BUILD),
        "llama_commit": str(paths.LLAMA_COMMIT),
        "arch": "gemma4",
        "n_layer": (max(ml) + 1) if ml else 0,
        "n_expert": int(n_expert),
        "n_expert_used": int(n_expert_used),
        "moe_layers": ml,
        "n_embd": 0,
        "probs_top": 0,
        "weights_tensor": MOETRACE_WEIGHTS_TENSOR,
        "corpus_id": "prompt",
        "created_at": utc_now_iso(),
        "n_ctx": 0,
        "n_batch": 0,
        "n_ubatch": 0,
        "n_threads": 0,
        "flags": [],
        "chunks": [],
    }
    h.update(fields)
    return h


def complete_header(header: Mapping[str, Any]) -> dict:
    """Fill missing keys of a partial header with ``make_header`` defaults (structural keys required)."""
    missing = [k for k in ("n_expert", "n_expert_used", "moe_layers") if k not in header]
    if missing:
        raise ValueError(f"trace header needs {', '.join(missing)}")
    rest = {k: v for k, v in header.items() if k not in ("n_expert", "n_expert_used", "moe_layers")}
    return make_header(
        n_expert=header["n_expert"], n_expert_used=header["n_expert_used"],
        moe_layers=header["moe_layers"], **rest,
    )


def _type_ok(v: Any, kind: str) -> bool:
    if kind == "int":
        return _is_int(v)
    if kind == "str":
        return isinstance(v, str)
    if kind == "int_list":
        return isinstance(v, list) and all(_is_int(x) for x in v)
    if kind == "str_list":
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    if kind == "chunks":
        return isinstance(v, list)
    return False


def check_header(header: Any) -> list[str]:
    """E3/E4 errors of a parsed JSON header (TRACE_FORMAT 3.9). Empty list = structurally valid."""
    if not isinstance(header, dict):
        return ["E3: JSON header is not an object"]
    errs: list[str] = []
    for key, kind in HEADER_SCHEMA:
        if key not in header:
            errs.append(f"E3: header key '{key}' missing")
        elif not _type_ok(header[key], kind):
            errs.append(
                f"E3: header key '{key}' has the wrong JSON type "
                f"(expected {kind}, got {type(header[key]).__name__})"
            )
    if errs:
        return errs
    if header["format_version"] != MOETRACE_FORMAT_VERSION:
        errs.append(f"E3: format_version {header['format_version']} != {MOETRACE_FORMAT_VERSION}")
    for i, c in enumerate(header["chunks"]):
        if not isinstance(c, dict):
            errs.append(f"E3: chunks[{i}] is not an object")
            continue
        for key, kind in (("chunk_id", "int"), ("source", "str"), ("n_prompt", "int"), ("n_gen", "int")):
            if key not in c:
                errs.append(f"E3: chunks[{i}] key '{key}' missing")
            elif not _type_ok(c[key], kind):
                errs.append(f"E3: chunks[{i}].{key} has the wrong JSON type (expected {kind})")
            elif kind == "int" and c[key] < 0:
                errs.append(f"E3: chunks[{i}].{key} is negative ({c[key]})")
        if "pos0" in c and (not _is_int(c["pos0"]) or c["pos0"] < 0):
            errs.append(f"E3: chunks[{i}].pos0 must be an integer >= 0")
    if errs:
        return errs
    E, K, P = header["n_expert"], header["n_expert_used"], header["probs_top"]
    ml, n_layer = header["moe_layers"], header["n_layer"]
    if not 1 <= E <= MOETRACE_MAX_EXPERTS:
        errs.append(f"E4: n_expert {E} outside 1..{MOETRACE_MAX_EXPERTS}")
    if not 1 <= K <= max(E, 1):
        errs.append(f"E4: n_expert_used {K} outside 1..n_expert ({E})")
    if not 0 <= P <= max(E, 0):
        errs.append(f"E4: probs_top {P} outside 0..n_expert ({E})")
    if not ml:
        errs.append("E4: moe_layers is empty")
    else:
        if any(b <= a for a, b in zip(ml, ml[1:])):
            errs.append(f"E4: moe_layers is not strictly increasing: {ml}")
        bad = [x for x in ml if not 0 <= x < n_layer]
        if bad:
            errs.append(f"E4: moe_layers values {bad[:5]} outside [0, n_layer={n_layer})")
    return errs


# --------------------------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------------------------
class TraceWriter:
    """Write a ``.moetrace`` file and its ``<trace>.tokens.bin`` sidecar (TRACE_FORMAT 3.5).

    Usage::

        with TraceWriter("x.moetrace", {"n_expert": 128, "n_expert_used": 8,
                                        "moe_layers": list(range(30))}) as w:
            w.begin_chunk("en_tech")            # optional: names the chunk source
            w.write(i, pos, tok, 0, 0, ids, weights)

    ``header`` may be partial (see ``complete_header``). ``reserve`` is the header slack in bytes
    (16384 by default; 0 when the final header is known at open, e.g. merge). If ``header``
    already holds a non-empty ``chunks`` list, those chunks are final and the writer does not
    track chunks itself. ``close(chunks=...)`` can always override.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        header: Mapping[str, Any],
        *,
        reserve: int = MOETRACE_HEADER_RESERVE_BYTES,
        tokens_sidecar: bool = True,
        strict: bool = True,
    ) -> None:
        hdr = complete_header(header)
        errs = check_header(hdr)
        if errs:
            raise ValueError("invalid trace header: " + "; ".join(errs))
        if reserve < 0:
            raise ValueError("reserve must be >= 0")
        self.path = Path(os.fspath(path))
        self.header: dict[str, Any] = hdr
        self.strict = bool(strict)
        self.L = len(hdr["moe_layers"])
        self.K = int(hdr["n_expert_used"])
        self.E = int(hdr["n_expert"])
        self.P = int(hdr["probs_top"])
        self.record_size = record_size(self.L, self.K, self.P)
        self.dtype = record_dtype(self.L, self.K, self.P)
        self.reserve = int(reserve)
        self.n = 0
        self.closed = False
        self._chunks_fixed = bool(hdr["chunks"])
        self._chunks: list[dict[str, Any]] = [dict(c) for c in hdr["chunks"]]
        self._cur = -1  # index of the chunk records currently go to (tracking mode)
        js = _header_bytes(hdr)
        self.H = max(_round_up_8(len(js) + self.reserve), 8)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w+b")  # noqa: SIM115 - closed in close()
        try:
            self._f.write(_FIXED.pack(MOETRACE_MAGIC, self.H, self.record_size, MOETRACE_N_RECORDS_OPEN, 0))
            self._f.write(js + MOETRACE_HEADER_PAD_BYTE * (self.H - len(js)))
            self._tok = open(tokens_path(self.path), "wb") if tokens_sidecar else None  # noqa: SIM115
        except BaseException:
            self._f.close()
            raise

    # -- context manager ------------------------------------------------------------------
    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"TraceWriter({str(self.path)!r}, n={self.n}, L={self.L}, K={self.K}, E={self.E}, P={self.P})"

    # -- chunks ---------------------------------------------------------------------------
    @property
    def chunks(self) -> list[dict[str, Any]]:
        """Chunk list as it will be written at close (copy)."""
        return [dict(c) for c in self._chunks]

    def begin_chunk(self, source: str | None = None, **extra: Any) -> int:
        """Start a new chunk (tracking mode) and return its full index.

        Following records must carry ``chunk_id == index`` (mod 256). ``extra`` keys (e.g.
        ``prompt_index``) are stored in the chunk object.
        """
        if self._chunks_fixed:
            raise RuntimeError("chunks were given in the header; begin_chunk() is not available")
        idx = len(self._chunks)
        c: dict[str, Any] = {
            "chunk_id": idx,
            "source": str(source) if source is not None else str(self.header["corpus_id"]),
            "n_prompt": 0,
            "n_gen": 0,
        }
        c.update(extra)
        self._chunks.append(c)
        self._cur = idx
        return idx

    def _enter_chunk(self, cid: int) -> None:
        if self._chunks_fixed:
            return
        if self._cur >= 0 and (cid & 0xFF) == (self._cur & 0xFF):
            return
        nxt = len(self._chunks)
        if self.strict and (cid & 0xFF) != (nxt & 0xFF):
            cur = "none" if self._cur < 0 else str(self._cur)
            raise ValueError(
                f"chunk_id {cid} is neither the current chunk ({cur}) nor the next one ({nxt}) mod 256"
            )
        self.begin_chunk()

    def _count(self, cids: np.ndarray, phases: np.ndarray) -> None:
        if self._chunks_fixed or cids.size == 0:
            return
        bounds = np.flatnonzero(cids[1:] != cids[:-1]) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [cids.size]))
        for a, b in zip(starts.tolist(), ends.tolist()):
            self._enter_chunk(int(cids[a]))
            n_gen = int(np.count_nonzero(phases[a:b] == MOETRACE_PHASE_GENERATED))
            c = self._chunks[self._cur]
            c["n_gen"] += n_gen
            c["n_prompt"] += (b - a) - n_gen

    # -- records --------------------------------------------------------------------------
    def _coerce_probs(self, probs: Any, n: int | None) -> tuple[np.ndarray, np.ndarray]:
        """probs -> (ids uint, p float) shaped [(n,) L, P]."""
        shape = (self.L, self.P) if n is None else (n, self.L, self.P)
        if isinstance(probs, np.ndarray) and probs.dtype.names and {"id", "p"} <= set(probs.dtype.names):
            pid, pp = probs["id"], probs["p"]
        elif isinstance(probs, (tuple, list)) and len(probs) == 2:
            pid, pp = probs
        else:
            arr = np.asarray(probs)
            if arr.shape != (*shape, 2):
                raise ValueError(f"probs must be a structured (id,p) array, an (ids, probs) pair or [..., 2]; got {arr.shape}")
            pid, pp = arr[..., 0], arr[..., 1]
        pid, pp = np.asarray(pid), np.asarray(pp, dtype=np.float32)
        if pid.shape != shape or pp.shape != shape:
            raise ValueError(f"probs shape {pid.shape}/{pp.shape} != {shape}")
        return pid, pp

    def _check_ids(self, ids: np.ndarray, what: str) -> None:
        if ids.size == 0:
            return
        if ids.min() < 0 or ids.max() > 255:
            raise ValueError(f"{what} do not fit u8")
        if self.strict:
            if ids.max() >= self.E:
                raise ValueError(f"{what}: id {int(ids.max())} >= n_expert {self.E}")
            s = np.sort(ids, axis=-1)
            if s.shape[-1] > 1 and np.any(s[..., 1:] == s[..., :-1]):
                raise ValueError(f"{what}: duplicate expert ids within a (token, layer)")

    def write(
        self,
        token_index: int | None,
        pos: int,
        token_id: int,
        phase: int,
        chunk_id: int,
        ids: Any,
        weights: Any,
        probs: Any = None,
    ) -> None:
        """Append one record. ``ids``/``weights`` are [L, K]; ``probs`` is [L, P] (id, prob) or None.

        ``token_index=None`` means "the next index". ``chunk_id`` may be the full chunk index;
        the record stores it mod 256. Weights and probabilities are rounded to f16 with
        round-to-nearest-even, the same as ggml_fp32_to_fp16.
        """
        if self.closed:
            raise ValueError("write() on a closed TraceWriter")
        ti = self.n if token_index is None else int(token_index)
        if self.strict:
            if ti != self.n:
                raise ValueError(f"token_index {ti} != next index {self.n}")
            if int(phase) not in (MOETRACE_PHASE_PREFILL, MOETRACE_PHASE_GENERATED):
                raise ValueError(f"phase {phase} not in {{0, 1}}")
        ids_a = np.asarray(ids)
        w_a = np.asarray(weights, dtype=np.float32)
        if ids_a.shape != (self.L, self.K) or w_a.shape != (self.L, self.K):
            raise ValueError(f"ids/weights shape {ids_a.shape}/{w_a.shape} != {(self.L, self.K)}")
        self._check_ids(ids_a, "ids")
        rec = np.zeros(1, dtype=self.dtype)
        rec["token_index"] = ti
        rec["pos"] = int(pos)
        rec["token_id"] = int(token_id)
        rec["phase"] = int(phase)
        rec["chunk_id"] = int(chunk_id) & 0xFF
        rec["moe"]["ids"][0] = ids_a.astype(np.uint8)
        rec["moe"]["w"][0] = w_a.astype(np.float16)
        if self.P > 0:
            if probs is None:
                raise ValueError(f"probs required (probs_top={self.P})")
            pid, pp = self._coerce_probs(probs, None)
            if self.strict:
                self._check_prob_ids(pid)
            else:
                self._check_ids(np.asarray(pid), "prob ids")
            rec["probs"]["id"][0] = pid.astype(np.uint8)
            rec["probs"]["p"][0] = pp.astype(np.float16)
        elif probs is not None:
            raise ValueError("probs given but probs_top == 0")
        self._enter_chunk(int(chunk_id))
        self._f.write(rec.tobytes())
        if self._tok is not None:
            self._tok.write(struct.pack("<i", int(token_id)))
        if not self._chunks_fixed:
            c = self._chunks[self._cur]
            c["n_gen" if int(phase) == MOETRACE_PHASE_GENERATED else "n_prompt"] += 1
        self.n += 1

    def _check_prob_ids(self, pid: np.ndarray) -> None:
        pid = np.asarray(pid)
        if pid.size and (pid.min() < 0 or pid.max() >= self.E):
            raise ValueError(f"prob ids out of range 0..{self.E - 1}")
        s = np.sort(pid, axis=-1)
        if s.shape[-1] > 1 and np.any(s[..., 1:] == s[..., :-1]):
            raise ValueError("duplicate prob ids within a (token, layer)")

    def write_many(
        self,
        ids: Any,
        weights: Any,
        *,
        phase: Any = None,
        pos: Any = None,
        token_ids: Any = None,
        chunk_ids: Any = None,
        probs: Any = None,
    ) -> None:
        """Append n records at once: ``ids``/``weights`` [n, L, K]; per-record arrays of length n.

        Defaults: phase 0, chunk 0, token_ids 0 and ``pos`` counted from 0 inside each
        ``chunk_ids`` run. ``token_index`` continues from the current count.
        """
        ids_a = np.asarray(ids)
        if ids_a.ndim != 3 or ids_a.shape[1:] != (self.L, self.K):
            raise ValueError(f"ids shape {ids_a.shape} != (n, {self.L}, {self.K})")
        n = ids_a.shape[0]
        w_a = np.asarray(weights, dtype=np.float32)
        if w_a.shape != ids_a.shape:
            raise ValueError(f"weights shape {w_a.shape} != {ids_a.shape}")
        rec = np.zeros(n, dtype=self.dtype)
        rec["token_index"] = np.arange(self.n, self.n + n, dtype=np.int64)
        cid = np.zeros(n, np.int64) if chunk_ids is None else np.asarray(chunk_ids, np.int64)
        rec["chunk_id"] = cid & 0xFF
        if pos is None:
            p = np.arange(n, dtype=np.int64)
            if n:
                brk = np.flatnonzero(cid[1:] != cid[:-1]) + 1
                starts = np.zeros(n, np.int64)
                starts[brk] = brk
                p = p - np.maximum.accumulate(starts)
            rec["pos"] = p
        else:
            rec["pos"] = np.asarray(pos)
        rec["token_id"] = 0 if token_ids is None else np.asarray(token_ids)
        rec["phase"] = 0 if phase is None else np.asarray(phase)
        self._check_ids(ids_a, "ids")
        rec["moe"]["ids"] = ids_a.astype(np.uint8)
        rec["moe"]["w"] = w_a.astype(np.float16)
        if self.P > 0:
            if probs is None:
                raise ValueError(f"probs required (probs_top={self.P})")
            pid, pp = self._coerce_probs(probs, n)
            if self.strict:
                self._check_prob_ids(pid)
            else:
                self._check_ids(np.asarray(pid), "prob ids")
            rec["probs"]["id"] = np.asarray(pid).astype(np.uint8)
            rec["probs"]["p"] = pp.astype(np.float16)
        self.write_array(rec, _chunk_ids_full=cid)

    def write_array(self, recs: np.ndarray, *, _chunk_ids_full: np.ndarray | None = None) -> None:
        """Append records given as a structured array of exactly ``self.dtype`` (bulk path)."""
        if self.closed:
            raise ValueError("write_array() on a closed TraceWriter")
        recs = np.ascontiguousarray(recs)
        if recs.dtype != self.dtype:
            raise ValueError(f"record dtype mismatch: {recs.dtype} != {self.dtype}")
        n = recs.shape[0]
        if n == 0:
            return
        if self.strict:
            ti = recs["token_index"].astype(np.int64)
            if not np.array_equal(ti, np.arange(self.n, self.n + n)):
                raise ValueError(f"token_index must continue from {self.n}")
            if np.any(recs["phase"] > 1):
                raise ValueError("phase not in {0, 1}")
            self._check_ids(recs["moe"]["ids"], "ids")
            if self.P > 0:
                self._check_prob_ids(recs["probs"]["id"])
        cids = recs["chunk_id"].astype(np.int64) if _chunk_ids_full is None else _chunk_ids_full
        self._count(cids, recs["phase"])
        self._f.write(recs.tobytes())
        if self._tok is not None:
            self._tok.write(recs["token_id"].astype("<i4").tobytes())
        self.n += n

    def flush(self) -> None:
        """Flush records and tokens to the OS; ``n_records`` stays the sentinel until close()."""
        self._f.flush()
        if self._tok is not None:
            self._tok.flush()

    # -- close ----------------------------------------------------------------------------
    def close(self, chunks: Sequence[Mapping[str, Any]] | None = None, **updates: Any) -> None:
        """Patch the final header (with ``chunks`` and any ``updates``) and ``n_records``.

        If the final JSON no longer fits into H bytes, the file is rewritten as ``<trace>.tmp``
        with a fresh 16 KiB reserve and renamed over the original (TRACE_FORMAT 3.5 step 3).
        """
        if self.closed:
            return
        hdr = dict(self.header)
        hdr.update(updates)
        for k in STRUCTURAL_KEYS:
            if hdr.get(k) != self.header.get(k):
                raise ValueError(f"header key '{k}' cannot change at close (record layout)")
        hdr["chunks"] = [dict(c) for c in (chunks if chunks is not None else self._chunks)]
        errs = check_header(hdr)
        if errs:
            raise ValueError("invalid final header: " + "; ".join(errs))
        self.header = hdr
        js = _header_bytes(hdr)
        if self._tok is not None:
            self._tok.close()
        self._f.flush()
        if len(js) <= self.H:
            self._f.seek(MOETRACE_OFF_HEADER_JSON)
            self._f.write(js + MOETRACE_HEADER_PAD_BYTE * (self.H - len(js)))
            self._f.flush()
            self._f.seek(MOETRACE_OFF_N_RECORDS)
            self._f.write(struct.pack("<Q", self.n))
            self._f.flush()
            self._f.close()
        else:
            self._rewrite(js)
        self.closed = True

    def _rewrite(self, js: bytes) -> None:
        new_h = _round_up_8(len(js) + MOETRACE_HEADER_RESERVE_BYTES)
        tmp = self.path.with_name(self.path.name + ".tmp")
        log.info("final header (%d B) exceeds H=%d; rewriting %s with H=%d", len(js), self.H, self.path, new_h)
        try:
            with open(tmp, "wb") as out:
                out.write(_FIXED.pack(MOETRACE_MAGIC, new_h, self.record_size, MOETRACE_N_RECORDS_OPEN, 0))
                out.write(js + MOETRACE_HEADER_PAD_BYTE * (new_h - len(js)))
                self._f.seek(MOETRACE_FIXED_HEADER_BYTES + self.H)
                left = self.n * self.record_size
                while left > 0:
                    buf = self._f.read(min(left, 1 << 22))
                    if not buf:
                        raise OSError(f"short read while rewriting {self.path}")
                    out.write(buf)
                    left -= len(buf)
                out.flush()
                out.seek(MOETRACE_OFF_N_RECORDS)
                out.write(struct.pack("<Q", self.n))
                out.flush()
        except BaseException:
            self._f.close()
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        self._f.close()
        os.replace(tmp, self.path)
        self.H = new_h


# --------------------------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------------------------
def read_fixed_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The 32 fixed bytes as a dict {magic, header_json_len, record_size, n_records, reserved}."""
    with open(path, "rb") as f:
        b = f.read(MOETRACE_FIXED_HEADER_BYTES)
    if len(b) < MOETRACE_FIXED_HEADER_BYTES:
        raise TraceFormatError(f"{path}: file shorter than the 32-byte fixed header ({len(b)} B)")
    magic, h, r, n, res = _FIXED.unpack(b)
    return {"magic": magic, "header_json_len": h, "record_size": r, "n_records": n, "reserved": res}


def _read_head(path: Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    size = path.stat().st_size
    fx = read_fixed_header(path)
    if fx["magic"] != MOETRACE_MAGIC:
        raise TraceFormatError(f"{path}: bad magic {fx['magic']!r} (expected {MOETRACE_MAGIC!r})")
    h = fx["header_json_len"]
    if h % MOETRACE_HEADER_JSON_ALIGN or h < 2 or MOETRACE_FIXED_HEADER_BYTES + h > size:
        raise TraceFormatError(f"{path}: bad header_json_len {h} (file size {size})")
    with open(path, "rb") as f:
        f.seek(MOETRACE_OFF_HEADER_JSON)
        raw = f.read(h)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise TraceFormatError(f"{path}: JSON header unreadable: {e}") from e
    errs = check_header(header)
    if errs:
        raise TraceFormatError(f"{path}: " + "; ".join(errs))
    return header, fx, size


def read_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse only the JSON header of a trace (cheap; no records are touched)."""
    return _read_head(Path(os.fspath(path)))[0]


def _derive_segments(
    header: Mapping[str, Any], phase: np.ndarray, pos: np.ndarray, cid: np.ndarray
) -> tuple[list[dict[str, Any]], str | None]:
    """Chunk segments [{index, start, end, chunk, derived}] and an inconsistency message or None."""
    n = int(phase.shape[0])
    chunks = header.get("chunks")
    good = isinstance(chunks, list) and all(
        isinstance(c, dict) and _is_int(c.get("n_prompt")) and _is_int(c.get("n_gen"))
        and c["n_prompt"] >= 0 and c["n_gen"] >= 0 for c in chunks
    )
    total = sum(c["n_prompt"] + c["n_gen"] for c in chunks) if good else None
    if good and chunks and total == n:
        segs: list[dict[str, Any]] = []
        s = 0
        for i, c in enumerate(chunks):
            m = c["n_prompt"] + c["n_gen"]
            segs.append({"index": i, "start": s, "end": s + m, "chunk": dict(c), "derived": False})
            s += m
        return segs, None
    if not good:
        msg: str | None = "header chunks are malformed"
    elif total != n:
        msg = (f"sum(n_prompt + n_gen) over header chunks = {total} != n_records {n}"
               + (" (header has no chunks; trace not closed?)" if not chunks else ""))
    else:
        msg = None
    if n == 0:
        return [], msg
    corpus = str(header.get("corpus_id", "prompt"))
    brk = np.flatnonzero((cid[1:] != cid[:-1]) | (pos[1:] == 0)) + 1
    starts = np.concatenate(([0], brk)).tolist()
    ends = np.concatenate((brk, [n])).tolist()
    segs = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        ng = int(np.count_nonzero(phase[a:b] == MOETRACE_PHASE_GENERATED))
        segs.append({
            "index": i, "start": a, "end": b, "derived": True,
            "chunk": {"chunk_id": i, "source": corpus, "n_prompt": (b - a) - ng, "n_gen": ng},
        })
    return segs, msg


class TraceReader:
    """Read a ``.moetrace`` (TRACE_FORMAT 3.10). Records are memory-mapped; columns are cached copies.

    Attributes: ``path, header, header_json_len, record_size, n_records_field, n, truncated,
    trailing_bytes, data_start, file_size, L, K, E, P, moe_layers, dtype, records`` (structured
    memmap, shape [n]). Column properties (copies, cached): ``ids`` uint8 [n,L,K], ``weights``
    float32 [n,L,K], ``phase`` / ``chunk`` uint8 [n], ``pos`` / ``token_index`` uint32 [n],
    ``token_id`` int32 [n] (record column), ``token_ids`` int32 [n] (tokens.bin, else the record
    column; see ``token_ids_source``), ``probs`` = (id uint8 [n,L,P], p float32 [n,L,P]) or None.

    On Windows a live memmap keeps the file open; use ``with TraceReader(p) as r:`` or
    ``close()`` before replacing the file, or pass ``use_mmap=False``.
    """

    def __init__(self, path: str | os.PathLike[str], *, use_mmap: bool = True) -> None:
        self.path_str = os.fspath(path)  # as given by the caller (recorded in stats.json "trace")
        self.path = Path(self.path_str)
        header, fx, size = _read_head(self.path)
        self.header: dict[str, Any] = header
        self.file_size = size
        self.header_json_len = int(fx["header_json_len"])
        self.record_size = int(fx["record_size"])
        self.n_records_field = int(fx["n_records"])
        self.reserved_field = int(fx["reserved"])
        self.moe_layers: list[int] = list(header["moe_layers"])
        self.L = len(self.moe_layers)
        self.K = int(header["n_expert_used"])
        self.E = int(header["n_expert"])
        self.P = int(header["probs_top"])
        expect = record_size(self.L, self.K, self.P)
        if self.record_size != expect:
            raise TraceFormatError(
                f"{self.path}: record_size {self.record_size} != {MOETRACE_RECORD_SIZE_FORMULA} = {expect}"
            )
        self.dtype = record_dtype(self.L, self.K, self.P)
        self.data_start = MOETRACE_FIXED_HEADER_BYTES + self.header_json_len
        data = size - self.data_start
        avail, self.trailing_bytes = divmod(data, self.record_size)
        self.truncated = False
        if self.n_records_field == MOETRACE_N_RECORDS_OPEN:
            n = avail
            self.truncated = True
            warnings.warn(
                f"{self.path}: n_records is the open sentinel (trace not closed); inferred {n} records"
                + (f", ignoring {self.trailing_bytes} trailing bytes" if self.trailing_bytes else ""),
                TraceWarning, stacklevel=2,
            )
        elif self.n_records_field > avail:
            n = avail
            self.truncated = True
            warnings.warn(
                f"{self.path}: n_records {self.n_records_field} but only {avail} complete records in the file",
                TraceWarning, stacklevel=2,
            )
        else:
            n = self.n_records_field
            if avail > n or self.trailing_bytes:
                warnings.warn(
                    f"{self.path}: {avail - n} extra records / {self.trailing_bytes} bytes after n_records ignored",
                    TraceWarning, stacklevel=2,
                )
        self.n = int(n)
        self.use_mmap = bool(use_mmap)
        if self.n == 0:
            self.records: np.ndarray = np.zeros(0, dtype=self.dtype)
        elif use_mmap:
            self.records = np.memmap(self.path, dtype=self.dtype, mode="r", offset=self.data_start, shape=(self.n,))
        else:
            self.records = np.fromfile(self.path, dtype=self.dtype, count=self.n, offset=self.data_start)

    # -- lifecycle ------------------------------------------------------------------------
    def __enter__(self) -> TraceReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (f"TraceReader({str(self.path)!r}, n={self.n}, L={self.L}, K={self.K}, E={self.E}, "
                f"P={self.P}, truncated={self.truncated})")

    def close(self) -> None:
        """Drop the memmap and cached columns (lets Windows release the file)."""
        for k in ("ids", "weights", "phase", "chunk", "pos", "token_index", "token_id", "token_ids", "probs"):
            self.__dict__.pop(k, None)
        self.records = np.zeros(0, dtype=self.dtype)

    # -- columns --------------------------------------------------------------------------
    @functools.cached_property
    def ids(self) -> np.ndarray:
        """uint8 [n, L, K] expert ids in ffn_moe_topk order."""
        return np.array(self.records["moe"]["ids"], dtype=np.uint8, copy=True).reshape(self.n, self.L, self.K)

    @functools.cached_property
    def weights(self) -> np.ndarray:
        """float32 [n, L, K] normalised routing weights (from f16)."""
        return np.array(self.records["moe"]["w"], dtype=np.float32).reshape(self.n, self.L, self.K)

    @functools.cached_property
    def phase(self) -> np.ndarray:
        """uint8 [n]: 0 prefill / teacher-forced, 1 generated."""
        return np.array(self.records["phase"], dtype=np.uint8)

    @functools.cached_property
    def chunk(self) -> np.ndarray:
        """uint8 [n]: the record's chunk_id (chunk index mod 256)."""
        return np.array(self.records["chunk_id"], dtype=np.uint8)

    @functools.cached_property
    def pos(self) -> np.ndarray:
        """uint32 [n]: position inside the chunk."""
        return np.array(self.records["pos"], dtype=np.uint32)

    @functools.cached_property
    def token_index(self) -> np.ndarray:
        """uint32 [n]: record token_index (== arange(n) in a valid trace)."""
        return np.array(self.records["token_index"], dtype=np.uint32)

    @functools.cached_property
    def token_id(self) -> np.ndarray:
        """int32 [n]: the record's token_id column."""
        return np.array(self.records["token_id"], dtype=np.int32)

    @property
    def tokens_path(self) -> Path:
        """``<trace>.tokens.bin``."""
        return tokens_path(self.path)

    @functools.cached_property
    def token_ids(self) -> np.ndarray:
        """int32 [n]: tokens.bin when present (first n values), else the record token_id column."""
        tp = self.tokens_path
        if tp.is_file():
            t = np.fromfile(tp, dtype="<i4")
            if t.shape[0] < self.n:
                warnings.warn(f"{tp}: {t.shape[0]} tokens < {self.n} records; using record token_id",
                              TraceWarning, stacklevel=2)
                self._token_src = "records"
                return self.token_id
            self._token_src = "tokens.bin"
            return t[: self.n].astype(np.int32)
        self._token_src = "records"
        return self.token_id

    @property
    def token_ids_source(self) -> str:
        """'tokens.bin' or 'records' (where ``token_ids`` came from)."""
        _ = self.token_ids
        return self._token_src

    @functools.cached_property
    def probs(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(id uint8 [n, L, P], p float32 [n, L, P]) or None when probs_top == 0."""
        if self.P == 0:
            return None
        pid = np.array(self.records["probs"]["id"], dtype=np.uint8).reshape(self.n, self.L, self.P)
        pp = np.array(self.records["probs"]["p"], dtype=np.float32).reshape(self.n, self.L, self.P)
        return pid, pp

    # -- chunks ---------------------------------------------------------------------------
    @property
    def chunks(self) -> list[dict[str, Any]]:
        """The header's chunk list (may be empty or stale for a truncated trace)."""
        return [dict(c) for c in self.header.get("chunks", [])]

    def segments(self) -> list[dict[str, Any]]:
        """Chunk segments [{index, start, end, chunk, derived}] covering records [0, n).

        Taken from the header when ``sum(n_prompt + n_gen) == n``, else derived from chunk_id
        changes and pos == 0 restarts (``derived: True``, source = corpus_id).
        """
        segs, _ = _derive_segments(self.header, self.phase, self.pos, self.chunk)
        return segs

    def chunk_index(self) -> np.ndarray:
        """int64 [n]: full chunk index of each record (not reduced mod 256)."""
        segs = self.segments()
        lens = [s["end"] - s["start"] for s in segs]
        return np.repeat(np.arange(len(segs), dtype=np.int64), lens)

    def chunk_sources(self) -> list[str]:
        """Per-record chunk source string (length n)."""
        out: list[str] = []
        for s in self.segments():
            out.extend([str(s["chunk"].get("source", ""))] * (s["end"] - s["start"]))
        return out


# --------------------------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class ValidationReport:
    """Result of ``validate_report``: errors (E-codes), warnings (W-codes) and a few counts."""

    path: str
    errors: list[str]
    warnings: list[str]
    n_records: int | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class _Issues:
    """Collect issues per code: count + first message."""

    def __init__(self) -> None:
        self.first: dict[str, str] = {}
        self.count: dict[str, int] = {}

    def add(self, code: str, n: int, msg: str) -> None:
        if n <= 0:
            return
        if code not in self.first:
            self.first[code] = msg
        self.count[code] = self.count.get(code, 0) + int(n)

    def render(self, prefix: str) -> list[str]:
        out = []
        for code in sorted((c for c in self.first if c.startswith(prefix)), key=lambda c: int(c[1:])):
            n = self.count[code]
            out.append(f"{code}: {self.first[code]}" + (f" ({n} occurrences)" if n > 1 else ""))
        return out


def validate_report(path: str | os.PathLike[str]) -> ValidationReport:
    """Run every check of TRACE_FORMAT 3.9 and return errors and warnings separately."""
    p = Path(os.fspath(path))
    rep = ValidationReport(path=os.fspath(path), errors=[], warnings=[])
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            fixed = f.read(MOETRACE_FIXED_HEADER_BYTES)
            if len(fixed) < 8 or fixed[:8] != MOETRACE_MAGIC:
                rep.errors.append(f"E1: bad magic {fixed[:8]!r} (expected {MOETRACE_MAGIC!r})")
                return rep
            if len(fixed) < MOETRACE_FIXED_HEADER_BYTES:
                rep.errors.append(f"E2: file shorter than the 32-byte fixed header ({len(fixed)} B)")
                return rep
            _, h, r, n_field, reserved = _FIXED.unpack(fixed)
            if h % MOETRACE_HEADER_JSON_ALIGN or h < 2 or MOETRACE_FIXED_HEADER_BYTES + h > size:
                rep.errors.append(f"E2: header_json_len {h} invalid (multiple of 8, >= 2, 32+H <= file size {size})")
                return rep
            raw = f.read(h)
    except OSError as e:
        rep.errors.append(f"E1: cannot read {p}: {e}")
        return rep
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        rep.errors.append(f"E2: JSON header is not UTF-8: {e}")
        return rep
    try:
        header, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as e:
        rep.errors.append(f"E2: JSON header does not parse: {e}")
        return rep
    if not isinstance(header, dict):
        rep.errors.append("E2: JSON header is not an object")
        return rep
    if text[end:].strip(" "):
        rep.errors.append("E2: non-space bytes after the JSON object in the header")
        return rep
    herr = check_header(header)
    if herr:
        rep.errors.extend(herr)
        return rep
    W = _Issues()
    if reserved != 0:
        W.add("W8", 1, f"fixed-header reserved field is {reserved}, expected 0")
    L, K, E, P = len(header["moe_layers"]), header["n_expert_used"], header["n_expert"], header["probs_top"]
    ml = header["moe_layers"]
    exp_r = record_size(L, K, P)
    if r != exp_r:
        rep.errors.append(f"E5: record_size {r} != {MOETRACE_RECORD_SIZE_FORMULA} = {exp_r} (L={L}, K={K}, P={P})")
        return rep
    data_start = MOETRACE_FIXED_HEADER_BYTES + h
    data = size - data_start
    avail, rem = divmod(data, r)
    errs = _Issues()
    if rem:
        errs.add("E6", 1, f"{rem} bytes of a truncated record after {avail} complete records")
    if n_field == MOETRACE_N_RECORDS_OPEN:
        W.add("W1", 1, f"n_records is the open sentinel; trace not closed, {avail} records inferred")
        n = avail
        rep.truncated = True
    else:
        if n_field * r != data:
            errs.add("E7", 1, f"n_records {n_field} != (file_size - 32 - H) / record_size = {data / r:g}")
        n = min(n_field, avail)
    rep.n_records = n
    if header["weights_tensor"] != MOETRACE_WEIGHTS_TENSOR:
        W.add("W5", 1, f"weights_tensor is '{header['weights_tensor']}', weight-sum check skipped")
    if header["llama_build"] != paths.LLAMA_BUILD or not commit_compatible(header["llama_commit"], paths.LLAMA_COMMIT):
        W.add("W6", 1, f"llama build {header['llama_build']} / commit '{header['llama_commit']}' "
                       f"differs from {paths.LLAMA_BUILD} / {paths.LLAMA_COMMIT}")
    dt = record_dtype(L, K, P)
    check_w = header["weights_tensor"] == MOETRACE_WEIGHTS_TENSOR
    phase_all = np.zeros(n, np.uint8)
    pos_all = np.zeros(n, np.uint32)
    cid_all = np.zeros(n, np.uint8)
    tok_all = np.zeros(n, np.int32)
    for a in range(0, n, _BLOCK_RECORDS):
        b = min(n, a + _BLOCK_RECORDS)
        rec = np.fromfile(p, dtype=dt, count=b - a, offset=data_start + a * r)
        if rec.shape[0] != b - a:  # pragma: no cover - file changed underneath
            errs.add("E6", 1, f"short read at record {a}")
            break
        _check_block(rec, a, ml, K, E, P, check_w, errs, W)
        phase_all[a:b] = rec["phase"]
        pos_all[a:b] = rec["pos"]
        cid_all[a:b] = rec["chunk_id"]
        tok_all[a:b] = rec["token_id"]
    _check_chunks(header, phase_all, pos_all, cid_all, W)
    tp = tokens_path(p)
    if tp.is_file():
        tsize = tp.stat().st_size
        n_ref = avail if n_field == MOETRACE_N_RECORDS_OPEN else n_field
        if tsize != 4 * n_ref:
            errs.add("E14", 1, f"{tp.name} has {tsize} bytes != 4 * n_records ({4 * n_ref})")
        else:
            toks = np.fromfile(tp, dtype="<i4", count=n)
            bad = np.flatnonzero(toks != tok_all)
            if bad.size:
                i = int(bad[0])
                W.add("W9", int(bad.size), f"tokens.bin[{i}] = {int(toks[i])} != record token_id {int(tok_all[i])}")
    else:
        W.add("W7", 1, f"{tp.name} is absent")
    rep.errors.extend(errs.render("E"))
    rep.warnings.extend(W.render("W"))
    return rep


def _check_block(rec: np.ndarray, a: int, ml: Sequence[int], K: int, E: int, P: int,
                 check_w: bool, errs: _Issues, W: _Issues) -> None:
    m = rec.shape[0]
    ti = rec["token_index"].astype(np.int64)
    bad = np.flatnonzero(ti != np.arange(a, a + m))
    if bad.size:
        i = int(bad[0])
        errs.add("E8", int(bad.size), f"record {a + i} has token_index {int(ti[i])}")
    ph = rec["phase"]
    bad = np.flatnonzero(ph > 1)
    if bad.size:
        i = int(bad[0])
        errs.add("E9", int(bad.size), f"token {a + i}: phase {int(ph[i])} not in {{0, 1}}")
    bad = np.flatnonzero(rec["reserved"] != 0)
    if bad.size:
        i = int(bad[0])
        errs.add("E9", int(bad.size), f"token {a + i}: record reserved field {int(rec['reserved'][i])} != 0")
    ids = rec["moe"]["ids"]  # [m, L, K]
    over = np.argwhere(ids >= E)
    if over.size:
        t, j, k = (int(x) for x in over[0])
        errs.add("E10", int(over.shape[0]), f"layer {ml[j]} token {a + t}: id {int(ids[t, j, k])} >= n_expert {E}")
    if K > 1:
        s = np.sort(ids, axis=2)
        dup = np.argwhere(np.any(s[:, :, 1:] == s[:, :, :-1], axis=2))
        if dup.size:
            t, j = (int(x) for x in dup[0])
            errs.add("E11", int(dup.shape[0]),
                     f"layer {ml[j]} token {a + t}: duplicate expert ids {ids[t, j].tolist()}")
    w = rec["moe"]["w"].astype(np.float64)
    nonfin = ~np.isfinite(w)
    if np.any(nonfin):
        t, j, _k = (int(x) for x in np.argwhere(nonfin)[0])
        errs.add("E12", int(np.count_nonzero(np.any(nonfin, axis=2))),
                 f"layer {ml[j]} token {a + t}: non-finite weight {w[t, j].tolist()}")
    if check_w:
        sums = w.sum(axis=2)
        badsum = np.isfinite(sums) & (np.abs(sums - 1.0) > MOETRACE_WEIGHT_SUM_TOL)
        if np.any(badsum):
            t, j = (int(x) for x in np.argwhere(badsum)[0])
            errs.add("E12", int(np.count_nonzero(badsum)),
                     f"layer {ml[j]} token {a + t}: weights sum to {sums[t, j]:.6g} "
                     f"(|sum - 1| > {MOETRACE_WEIGHT_SUM_TOL})")
    if K > 1:
        inc = np.any(np.diff(w, axis=2) > WEIGHT_ORDER_TOL, axis=2)
        if np.any(inc):
            t, j = (int(x) for x in np.argwhere(inc)[0])
            W.add("W3", int(np.count_nonzero(inc)),
                  f"layer {ml[j]} token {a + t}: weights not non-increasing {np.round(w[t, j], 4).tolist()}")
    if P > 0:
        pid = rec["probs"]["id"]
        pp = rec["probs"]["p"].astype(np.float64)
        over = np.argwhere(pid >= E)
        if over.size:
            t, j, k = (int(x) for x in over[0])
            errs.add("E13", int(over.shape[0]), f"layer {ml[j]} token {a + t}: prob id {int(pid[t, j, k])} >= n_expert {E}")
        if P > 1:
            s = np.sort(pid, axis=2)
            dup = np.argwhere(np.any(s[:, :, 1:] == s[:, :, :-1], axis=2))
            if dup.size:
                t, j = (int(x) for x in dup[0])
                errs.add("E13", int(dup.shape[0]), f"layer {ml[j]} token {a + t}: duplicate prob ids")
        badp = ~np.isfinite(pp) | (pp < 0.0) | (pp > 1.0)
        if np.any(badp):
            t, j, k = (int(x) for x in np.argwhere(badp)[0])
            errs.add("E13", int(np.count_nonzero(badp)), f"layer {ml[j]} token {a + t}: prob {pp[t, j, k]!r} outside [0, 1]")
        if P >= K:
            mism = np.any(np.sort(pid[:, :, :K], axis=2) != np.sort(ids, axis=2), axis=2)
            if np.any(mism):
                t, j = (int(x) for x in np.argwhere(mism)[0])
                W.add("W4", int(np.count_nonzero(mism)),
                      f"layer {ml[j]} token {a + t}: top-{K} prob ids != selected ids")


def _check_chunks(header: Mapping[str, Any], phase: np.ndarray, pos: np.ndarray, cid: np.ndarray, W: _Issues) -> None:
    n = int(phase.shape[0])
    segs, msg = _derive_segments(header, phase, pos, cid)
    if msg:
        W.add("W2", 1, msg)
    if not segs:
        return
    for i, c in enumerate(header.get("chunks") or []):
        if isinstance(c, dict) and c.get("chunk_id") != i:
            W.add("W2", 1, f"chunks[{i}].chunk_id = {c.get('chunk_id')} != {i}")
            break
    lens = np.array([s["end"] - s["start"] for s in segs], dtype=np.int64)
    starts = np.array([s["start"] for s in segs], dtype=np.int64)
    seg_id = np.repeat(np.arange(len(segs), dtype=np.int64), lens)
    if not segs[0]["derived"]:
        bad = np.flatnonzero(cid.astype(np.int64) != (seg_id & 0xFF))
        if bad.size:
            i = int(bad[0])
            W.add("W2", int(bad.size), f"record {i} has chunk_id {int(cid[i])}, expected {int(seg_id[i]) & 0xFF}")
        for s in segs:
            ph = phase[s["start"]:s["end"]]
            ng = int(np.count_nonzero(ph == MOETRACE_PHASE_GENERATED))
            if ng != s["chunk"]["n_gen"]:
                W.add("W2", 1, f"chunk {s['index']}: n_gen {s['chunk']['n_gen']} but {ng} phase-1 records")
                break
    # probe 1.1.0 (server sessions): a chunk that reuses a KV prefix carries "pos0" = the position of
    # its first record (the number of reused tokens); the default is 0 (fresh context)
    pos0 = np.array([int(s["chunk"].get("pos0", 0) or 0) if not s["derived"] else 0 for s in segs], dtype=np.int64)
    exp_pos = np.arange(n, dtype=np.int64) - np.repeat(starts, lens) + np.repeat(pos0, lens)
    bad = np.flatnonzero(pos.astype(np.int64) != exp_pos)
    if bad.size:
        i = int(bad[0])
        W.add("W2", int(bad.size), f"record {i}: pos {int(pos[i])}, expected {int(exp_pos[i])} "
                                   f"(contiguous from the chunk's pos0, default 0)")
    if n > 1:
        viol = np.flatnonzero((phase[:-1] == 1) & (phase[1:] == 0) & (seg_id[:-1] == seg_id[1:]))
        if viol.size:
            i = int(viol[0])
            W.add("W2", int(viol.size), f"record {i + 1}: phase-0 record after a phase-1 record in chunk {int(seg_id[i])}")


def validate(path: str | os.PathLike[str]) -> list[str]:
    """Errors only (strings starting with E1..E14); warnings are logged (TRACE_FORMAT 3.9)."""
    rep = validate_report(path)
    for w in rep.warnings:
        log.warning("%s: %s", rep.path, w)
    return rep.errors


# --------------------------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------------------------
def trace_info(path: str | os.PathLike[str], *, include_header: bool = True) -> dict[str, Any]:
    """Header + counts + sidecar presence of a trace (the ``info`` command)."""
    p = Path(os.fspath(path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with TraceReader(p) as r:
            ph = r.phase
            segs = r.segments()
            out: dict[str, Any] = {
                "path": str(p),
                "file_size": r.file_size,
                "header_json_len": r.header_json_len,
                "record_size": r.record_size,
                "n_records": r.n,
                "n_records_field": "sentinel" if r.n_records_field == MOETRACE_N_RECORDS_OPEN else r.n_records_field,
                "truncated": r.truncated,
                "L": r.L, "K": r.K, "E": r.E, "P": r.P,
                "n_prefill": int(np.count_nonzero(ph == 0)),
                "n_gen": int(np.count_nonzero(ph == 1)),
                "n_chunks": len(segs),
                "chunks_derived": bool(segs and segs[0]["derived"]),
                "sources": sorted({str(s["chunk"].get("source", "")) for s in segs}),
                "sidecars": {
                    "tokens": tokens_path(p).is_file(),
                    "gen": gen_path(p).is_file(),
                    "meta": meta_path(p).is_file(),
                },
            }
            if include_header:
                out["header"] = r.header
    out["warnings"] = [str(w.message) for w in caught]
    return out


# --------------------------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------------------------
def _expand_inputs(paths_in: Iterable[str | os.PathLike[str]]) -> list[str]:
    """Input path strings as given; an unexpanded glob (PowerShell) is expanded and sorted."""
    out: list[str] = []
    for x in paths_in:
        s = os.fspath(x)
        if not os.path.exists(s) and any(ch in s for ch in "*?["):
            hits = sorted(glob.glob(s))
            if not hits:
                raise FileNotFoundError(f"no file matches {s}")
            out.extend(hits)
        else:
            out.append(s)
    return out


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def merge(
    paths_in: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    argv: Sequence[str] | None = None,
    write_gen: bool = True,
    write_meta_file: bool = True,
) -> dict[str, Any]:
    """Merge traces into ``out`` (TRACE_FORMAT 3.11) and return a summary dict.

    Inputs must agree on MERGE_KEYS. ``token_index`` is renumbered, chunks are concatenated
    with chunk ids renumbered from 0 (each chunk gains ``from_source``), record ``chunk_id``
    becomes the global chunk index mod 256, and every other byte is copied unchanged.
    ``tokens.bin``, ``gen.txt`` (renumbered separators) and ``meta.json`` are written next
    to ``out``.
    """
    t0 = time.perf_counter()
    outp = Path(os.fspath(out))
    inputs: list[str] = []
    for pth in _expand_inputs(paths_in):
        if os.path.abspath(pth) == os.path.abspath(outp) or _same_file(Path(pth), outp):
            log.warning("merge: skipping %s (it is the output)", pth)
            continue
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"merge input not found: {pth}")
        inputs.append(pth)
    if not inputs:
        raise MergeError("merge: no input traces")
    notes: list[str] = []
    readers: list[TraceReader] = []
    try:
        for pth in inputs:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rd = TraceReader(pth)
            for w in caught:
                log.warning("merge: %s", w.message)
            if rd.truncated:
                notes.append(f"input {pth} was truncated (not closed); its {rd.n} complete records were merged")
            readers.append(rd)
        first = readers[0]
        for rd in readers[1:]:
            for k in MERGE_KEYS:
                if rd.header.get(k) != first.header.get(k):
                    raise MergeError(
                        f"merge: header key '{k}' differs: {first.path} has {first.header.get(k)!r}, "
                        f"{rd.path} has {rd.header.get(k)!r}"
                    )
        for k in ("n_ctx", "n_batch", "n_ubatch", "n_threads", "gguf_path", "llama_build", "llama_commit"):
            vals = [rd.header.get(k) for rd in readers]
            if any(v != vals[0] for v in vals):
                notes.append(f"inputs differ in {k}: {vals}")
        # chunk plan
        new_chunks: list[dict[str, Any]] = []
        per_input: list[tuple[list[dict[str, Any]], int]] = []
        for si, rd in enumerate(readers):
            segs = rd.segments()
            if segs and segs[0]["derived"]:
                notes.append(f"input {rd.path}: header chunks inconsistent, chunks derived from records")
            off = len(new_chunks)
            for s in segs:
                c = dict(s["chunk"])
                c["chunk_id"] = len(new_chunks)
                c["from_source"] = si
                new_chunks.append(c)
            per_input.append((segs, off))
        hdr = dict(first.header)
        hdr["created_at"] = utc_now_iso()
        hdr["flags"] = [str(a) for a in (argv if argv is not None else
                                         ["pigemma.trace", "merge", os.fspath(out), *inputs])]
        hdr["corpus_id"] = "merged"
        hdr["sources"] = list(inputs)
        hdr["chunks"] = new_chunks
        old_notes = [str(x) for x in hdr.get("notes", [])] if isinstance(hdr.get("notes"), list) else []
        if notes or old_notes:
            hdr["notes"] = old_notes + notes
        else:
            hdr.pop("notes", None)
        tmp = outp.with_name(outp.name + ".tmp")
        n_total = 0
        w = TraceWriter(tmp, hdr, reserve=0, strict=False)
        try:
            for rd, (segs, off) in zip(readers, per_input):
                lens = [s["end"] - s["start"] for s in segs]
                gidx = np.repeat(np.arange(off, off + len(segs), dtype=np.int64), lens)
                tb = rd.tokens_path
                toks = np.fromfile(tb, dtype="<i4") if tb.is_file() else None
                if toks is not None and toks.shape[0] < rd.n:
                    log.warning("merge: W9: %s has fewer tokens than records", tb)
                    toks = None
                for a in range(0, rd.n, _BLOCK_RECORDS):
                    b = min(rd.n, a + _BLOCK_RECORDS)
                    blk = np.array(rd.records[a:b], copy=True)
                    if toks is not None and not np.array_equal(toks[a:b], blk["token_id"]):
                        log.warning("merge: W9: %s differs from the record token_id in records %d..%d", tb, a, b)
                    blk["token_index"] = np.arange(n_total + a, n_total + b, dtype=np.int64)
                    blk["chunk_id"] = gidx[a:b] & 0xFF
                    w.write_array(blk)
                n_total += rd.n
            w.close()
        except BaseException:
            if not w.closed:
                w._f.close()
                if w._tok is not None:
                    w._tok.close()
            for pth in (tmp, tokens_path(tmp)):
                with contextlib.suppress(OSError):
                    pth.unlink()
            raise
        os.replace(tmp, outp)
        os.replace(tokens_path(tmp), tokens_path(outp))
        # gen.txt
        gp = gen_path(outp)
        gen_written = False
        if write_gen:
            parts: list[bytes] = []
            for rd, (_segs, off) in zip(readers, per_input):
                g = gen_path(rd.path)
                if g.is_file():
                    data = g.read_bytes()
                    parts.append(_GEN_SEP_RE_BYTES.sub(
                        lambda m, o=off: GEN_CHUNK_SEPARATOR.format(i=int(m.group(1)) + o).encode("utf-8"), data))
            if parts:
                gp.write_bytes(b"".join(parts))
                gen_written = True
        if not gen_written and gp.exists():
            log.info("merge: removing stale %s", gp)
            gp.unlink()
        meta_written = None
        if write_meta_file:
            meta_written = str(write_meta(outp, _merged_meta(outp, readers, hdr, argv, notes, time.perf_counter() - t0)))
        elif meta_path(outp).exists():
            log.info("merge: removing stale %s", meta_path(outp))
            meta_path(outp).unlink()
        summary = {
            "out": os.fspath(out), "n_records": n_total, "n_chunks": len(new_chunks),
            "sources": list(inputs), "notes": notes,
            "tokens": str(tokens_path(outp)), "gen": str(gp) if gen_written else None, "meta": meta_written,
        }
        log.info("merged %d traces -> %s: %d records, %d chunks", len(inputs), outp, n_total, len(new_chunks))
        return summary
    finally:
        for rd in readers:
            rd.close()


def _merged_meta(outp: Path, readers: Sequence[TraceReader], hdr: Mapping[str, Any],
                 argv: Sequence[str] | None, notes: list[str], wall_s: float) -> dict[str, Any]:
    metas = [read_meta(rd.path) for rd in readers]
    shas = {m.get("gguf_sha256_prefix") for m in metas if m}
    corpus: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in metas:
        for c in (m or {}).get("corpus", []) or []:
            key = json.dumps(c, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                corpus.append(c)
    coh = [m.get("coherent") for m in metas if m]
    coherent = None
    if coh and len(coh) == len(metas) and all(isinstance(c, bool) for c in coh):
        coherent = all(coh)
    return {
        "trace": os.fspath(outp),
        "host": paths.host_fingerprint(),
        "gguf_path": hdr.get("gguf_path"),
        "gguf_size_bytes": hdr.get("gguf_size_bytes"),
        "gguf_sha256_prefix": next(iter(shas)) if len(shas) == 1 else None,
        "corpus": corpus,
        "argv": list(hdr.get("flags", [])) if argv is None else [str(a) for a in argv],
        "wall_ms": round(wall_s * 1000.0, 3),
        "tok_s": {"prompt": None, "gen": None},
        "callback_overhead_ms": None,
        "coherent": coherent,
        "chunks": hdr.get("chunks", []),
        "sources": [{"path": rd.path_str, "meta": m} for rd, m in zip(readers, metas)],
        "notes": ["merged by pigemma.trace merge: wall_ms is the merge time; per-input probe "
                  "timings are under sources[].meta"] + notes,
    }


# --------------------------------------------------------------------------------------------
# Statistics building blocks
# --------------------------------------------------------------------------------------------
def access_items(ids: np.ndarray, n_expert: int, item: str = "layer-expert") -> tuple[np.ndarray, int]:
    """Flatten [N, L, K] ids into the access sequence (t, j, k order) and return (items, n_items).

    'layer-expert': item = j * E + e (slot j, not the model layer number), n_items = L*E.
    'expert': item = e, n_items = E.
    """
    ids = np.asarray(ids)
    if ids.ndim != 3:
        raise ValueError(f"ids must be [N, L, K], got {ids.shape}")
    _n, L, _k = ids.shape
    if item == "layer-expert":
        items = (np.arange(L, dtype=np.int64)[None, :, None] * int(n_expert) + ids.astype(np.int64)).ravel()
        return items, L * int(n_expert)
    if item == "expert":
        return ids.astype(np.int64).ravel(), int(n_expert)
    raise ValueError(f"item must be 'layer-expert' or 'expert', got {item!r}")


def stack_distances(items: Any, n_items: int, *, batch: int = _SD_BATCH) -> np.ndarray:
    """Exact LRU stack distance of every access (0 = first touch), int64 [n].

    d = 1 + the number of distinct other items accessed since the previous access to the same
    item (TRACE_FORMAT 5.2). An access hits in an LRU cache of C items iff 1 <= d <= C.

    Algorithm: process accesses in batches of ``batch``. For access i with previous access p:
    distinct(p, i) = (i - p - 1) - #{r in (p, i) : prev[r] > p}. When p lies before the batch,
    this equals o + A - #{o' < o : prev[o'] > p}, where o is the offset in the batch and
    A = #{items whose last access before the batch is > p}, found with one searchsorted on the
    sorted last-access array. When p lies inside the batch, it is (o - o_p - 1) minus the same
    count. Cost O(n * (batch + n_items * log(n_items) / batch)).
    """
    x_all = np.asarray(items).astype(np.int64, copy=False).ravel()
    n = int(x_all.size)
    n_items = int(n_items)
    dist = np.zeros(n, dtype=np.int64)
    if n == 0:
        return dist
    if x_all.min() < 0 or x_all.max() >= n_items:
        raise ValueError(f"items must lie in [0, {n_items})")
    batch = max(1, int(batch))
    last = np.full(n_items, -1, dtype=np.int64)
    lower = np.tri(batch, batch, -1, dtype=bool)  # lower[o, o2] = o2 < o
    offs = np.arange(batch, dtype=np.int64)
    for start in range(0, n, batch):
        x = x_all[start:start + batch]
        m = int(x.size)
        order = np.argsort(x, kind="stable")
        xs = x[order]
        prev_in = np.full(m, -1, dtype=np.int64)
        if m > 1:
            k = np.flatnonzero(xs[1:] == xs[:-1]) + 1
            prev_in[order[k]] = order[k - 1]
        within = prev_in >= 0
        prev_abs = np.where(within, start + prev_in, last[x])
        cnt = ((prev_abs[None, :] > prev_abs[:, None]) & lower[:m, :m]).sum(axis=1)
        o = offs[:m]
        sorted_last = np.sort(last)
        a_cnt = n_items - np.searchsorted(sorted_last, prev_abs, side="right")
        d = np.where(within, (o - prev_in - 1) - cnt, o + a_cnt - cnt) + 1
        d[prev_abs < 0] = 0
        dist[start:start + m] = d
        is_last = np.ones(m, dtype=bool)
        if m > 1:
            is_last[:-1] = xs[1:] != xs[:-1]
        last[xs[is_last]] = start + order[is_last]
    return dist


def stack_distance_hist(items: Any, n_items: int) -> np.ndarray:
    """int64 histogram of length n_items + 1: hist[0] = cold accesses, hist[d] = distance d."""
    d = stack_distances(items, n_items)
    h = np.bincount(d, minlength=int(n_items) + 1)
    if h.shape[0] != int(n_items) + 1:  # pragma: no cover - distances are bounded by n_items
        raise AssertionError("stack distance exceeds the number of items")
    return h.astype(np.int64)


def lru_hit_curve(hist: Any) -> list[list[float]]:
    """[[C, hit(C)] for C in 1..len(hist)-1], hit(C) = sum(hist[1..C]) / sum(hist) (overall, cold misses counted)."""
    h = np.asarray(hist, dtype=np.int64)
    total = int(h.sum())
    cum = np.cumsum(h[1:])
    if total == 0:
        return [[c, 0.0] for c in range(1, h.shape[0])]
    return [[c, float(cum[c - 1]) / total] for c in range(1, h.shape[0])]


def lru_hit_rate(hist: Any, capacity_items: int) -> float:
    """Overall LRU hit rate at ``capacity_items`` from a stack-distance histogram."""
    h = np.asarray(hist, dtype=np.int64)
    total = int(h.sum())
    c = max(0, min(int(capacity_items), h.shape[0] - 1))
    return float(h[1:c + 1].sum()) / total if total else 0.0


def coverage_curve(counts: Any, ns: Iterable[int]) -> list[float]:
    """For each n: (sum of the n largest counts) / total; 1.0 when n >= len(counts). Non-decreasing."""
    c = np.asarray(counts, dtype=np.float64)
    total = float(c.sum())
    srt = np.sort(c)[::-1]
    cum = np.cumsum(srt)
    out = []
    for n in ns:
        n = int(n)
        if n >= c.shape[0]:
            out.append(1.0)
        elif n <= 0 or total <= 0:
            out.append(0.0)
        else:
            out.append(min(1.0, float(cum[n - 1]) / total))
    return out


def entropy_bits(counts: Any) -> float:
    """Shannon entropy (bits) of a count vector; 0 log 0 = 0."""
    c = np.asarray(counts, dtype=np.float64)
    tot = c.sum()
    if tot <= 0:
        return 0.0
    p = c[c > 0] / tot
    return float(max(0.0, -(p * np.log2(p)).sum()))


def gini(counts: Any) -> float:
    """Gini coefficient sum_i sum_k |x_i - x_k| / (2 n sum x); 0 = uniform."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.shape[0]
    tot = x.sum()
    if n == 0 or tot <= 0:
        return 0.0
    i = np.arange(n, dtype=np.float64)
    return float(((2.0 * i - n + 1.0) * x).sum() / (n * tot))


def js_divergence_bits(a: Any, b: Any) -> float | None:
    """Jensen-Shannon divergence (base 2, [0, 1]) between two count vectors; None if either is empty."""
    p = np.asarray(a, dtype=np.float64)
    q = np.asarray(b, dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(u: np.ndarray, v: np.ndarray) -> float:
        nz = u > 0
        return float((u[nz] * np.log2(u[nz] / v[nz])).sum())

    return float(min(1.0, max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))))


def mutual_info_bits(a: Any, b: Any, n_values: int, *, miller_madow: bool = False) -> float:
    """Mutual information I(a; b) in bits for two integer sequences in [0, n_values).

    Plug-in estimate by default (the stats.json contract). The plug-in estimate is biased
    upward by about (B_ab - B_a - B_b + 1) / (2 N ln 2) bits, where B counts the occupied bins.
    For E = 128 and N ~ 16k that is several tenths of a bit even for independent layers.
    ``miller_madow=True`` subtracts that first-order bias; the result can be slightly negative.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    n = int(a.size)
    if n == 0:
        return 0.0
    joint = np.bincount(a * n_values + b, minlength=n_values * n_values).reshape(n_values, n_values).astype(np.float64)
    joint /= joint.sum()
    pa = joint.sum(1)
    pb = joint.sum(0)
    nz = joint > 0
    denom = np.outer(pa, pb)
    mi = float(max(0.0, (joint[nz] * np.log2(joint[nz] / denom[nz])).sum()))
    if miller_madow:
        bias = (int(nz.sum()) - int((pa > 0).sum()) - int((pb > 0).sum()) + 1) / (2.0 * n * math.log(2.0))
        mi -= bias
    return mi


def _membership(ids2d: np.ndarray, n_expert: int) -> np.ndarray:
    """bool [N, E]: M[t, e] = e in ids2d[t]."""
    n = ids2d.shape[0]
    m = np.zeros((n, n_expert), dtype=bool)
    if n:
        m[np.arange(n)[:, None], ids2d.astype(np.int64)] = True
    return m


def cross_layer_table(ids: np.ndarray, n_train: int, n_expert: int) -> list[np.ndarray]:
    """Per consecutive slot pair j: P_j[e, e'] = P(e' in S_{j+1} | e in S_j) on records [0, n_train).

    Returns L-1 float64 [E, E] matrices (row e is 0 when e never appears in slot j of the train
    split). ``pigemma.cache_sim`` (lru_prefetch) can reuse this table.
    """
    ids = np.asarray(ids)
    _n, L, _k = ids.shape
    out = []
    for j in range(L - 1):
        a = _membership(ids[:n_train, j], n_expert).astype(np.float64)
        b = _membership(ids[:n_train, j + 1], n_expert).astype(np.float64)
        c = a.T @ b
        f = a.sum(0)
        out.append(np.divide(c, f[:, None], out=np.zeros_like(c), where=f[:, None] > 0))
    return out


def _topn(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n highest scores per row; ties go to the lower id."""
    return np.argsort(-scores, axis=-1, kind="stable")[..., :n]


def _cross_layer(ids: np.ndarray, n_train: int, E: int, K: int, moe_layers: Sequence[int]) -> dict[str, Any]:
    n, L, _ = ids.shape
    n_test = n - n_train
    keys = ["precision_at_8", "precision_at_16", "baseline_static_top16", "baseline_same_ids", "mutual_info_bits_mean"]
    if L < 2 or n_train < 1 or n_test < 1:
        return {k: None for k in keys} | {"mutual_info_bits_null_mean": None, "per_pair": [],
                                          "n_train": n_train, "n_test": n_test}
    table = cross_layer_table(ids, n_train, E)
    null_rng = np.random.default_rng(0)  # fixed seed: stats.json is reproducible
    pairs = []
    for j in range(L - 1):
        cur = _membership(ids[n_train:, j], E)
        nxt = _membership(ids[n_train:, j + 1], E)
        score = cur.astype(np.float64) @ table[j]
        res: dict[str, Any] = {"layer": int(moe_layers[j]), "next_layer": int(moe_layers[j + 1])}
        for npred in PREDICT_AT:
            pred = _topn(score, min(npred, E))
            res[f"precision_at_{npred}"] = float(np.take_along_axis(nxt, pred, axis=1).sum(1).mean() / K)
        train_freq_next = _membership(ids[:n_train, j + 1], E).sum(0)
        static = _topn(train_freq_next.astype(np.float64), min(STATIC_BASELINE_TOP, E))
        res["baseline_static_top16"] = float(nxt[:, static].sum(1).mean() / K)
        res["baseline_same_ids"] = float((cur & nxt).sum(1).mean() / K)
        res["mutual_info_bits"] = mutual_info_bits(ids[:, j, 0], ids[:, j + 1, 0], E)
        res["mutual_info_bits_null"] = mutual_info_bits(ids[:, j, 0], null_rng.permutation(ids[:, j + 1, 0]), E)
        pairs.append(res)

    def _mean(k: str) -> float:
        return float(np.mean([p[k] for p in pairs]))

    return {
        "precision_at_8": _mean("precision_at_8"),
        "precision_at_16": _mean("precision_at_16"),
        "baseline_static_top16": _mean("baseline_static_top16"),
        "baseline_same_ids": _mean("baseline_same_ids"),
        "mutual_info_bits_mean": _mean("mutual_info_bits"),
        "mutual_info_bits_null_mean": _mean("mutual_info_bits_null"),
        "per_pair": pairs,
        "n_train": int(n_train),
        "n_test": int(n_test),
    }


def _percentiles(x: np.ndarray, qs: Sequence[float]) -> list[float]:
    return [float(v) for v in np.percentile(x, qs)]


# --------------------------------------------------------------------------------------------
# stats()
# --------------------------------------------------------------------------------------------
def stats(
    reader: TraceReader | str | os.PathLike[str],
    train_frac: float = DEFAULT_TRAIN_FRAC,
    item: str = "layer-expert",
    *,
    coverage_points: Sequence[int] = COVERAGE_POINTS,
    working_set_windows: Sequence[int] = WORKING_SET_WINDOWS,
    corpus_top: int = CORPUS_TOP,
) -> dict[str, Any]:
    """H1 statistics of a trace (TRACE_FORMAT 5.2; schema data_formats 'stats.json').

    ``reader`` may be a TraceReader or a trace path. Raises ValueError on an empty trace or on
    expert ids >= n_expert (validate the trace first).
    """
    own = not isinstance(reader, TraceReader)
    r = TraceReader(reader) if own else reader
    try:
        return _stats(r, train_frac, item, coverage_points, working_set_windows, corpus_top)
    finally:
        if own:
            r.close()


def _stats(r: TraceReader, train_frac: float, item: str, coverage_points: Sequence[int],
           working_set_windows: Sequence[int], corpus_top: int) -> dict[str, Any]:
    n, L, K, E, P = r.n, r.L, r.K, r.E, r.P
    if n == 0:
        raise ValueError(f"{r.path}: trace has no records")
    if not 0.0 <= float(train_frac) <= 1.0:
        raise ValueError(f"train_frac {train_frac} outside [0, 1]")
    ids = r.ids
    if int(ids.max()) >= E:
        raise ValueError(f"{r.path}: expert ids >= n_expert {E}; run `python -m pigemma.trace validate`")
    w = r.weights.astype(np.float64)
    phase = r.phase
    n_train = int(math.floor(float(train_frac) * n))
    ml = r.moe_layers
    flat = (np.arange(L, dtype=np.int64)[None, :, None] * E + ids.astype(np.int64))
    freq = np.bincount(flat.ravel(), minlength=L * E).reshape(L, E)
    wmass = np.bincount(flat.ravel(), weights=w.ravel(), minlength=L * E).reshape(L, E)
    pre = phase == MOETRACE_PHASE_PREFILL
    gen = phase == MOETRACE_PHASE_GENERATED
    per_layer = []
    for j in range(L):
        fj = freq[j]
        mem = _membership(ids[:, j], E)
        ent = entropy_bits(fj)
        entry: dict[str, Any] = {
            "layer": int(ml[j]),
            "freq": [int(v) for v in fj],
            "weight_mass": [float(v) for v in wmass[j]],
            "entropy_bits": ent,
            "effective_experts": float(2.0 ** ent),
            "gini": gini(fj),
            "coverage_at": {str(c): v for c, v in zip(coverage_points, coverage_curve(fj, coverage_points))},
            "coverage_weight_at": {str(c): v for c, v in zip(coverage_points, coverage_curve(wmass[j], coverage_points))},
        }
        if n >= 2:
            inter = (mem[:-1] & mem[1:]).sum(1)
            union = (mem[:-1] | mem[1:]).sum(1)
            entry["adjacent_reuse"] = float(inter.mean() / K)
            entry["jaccard_next_mean"] = float(np.mean(inter / np.maximum(union, 1)))
        else:
            entry["adjacent_reuse"] = None
            entry["jaccard_next_mean"] = None
        ws: dict[str, Any] = {}
        for win in working_set_windows:
            win = int(win)
            if n < win or win <= 0:
                ws[str(win)] = None
                continue
            nw = n // win
            distinct = mem[: nw * win].reshape(nw, win, E).any(axis=1).sum(axis=1)
            med, p95 = _percentiles(distinct, [50, 95])
            ws[str(win)] = {"median": med, "p95": p95}
        entry["working_set"] = ws
        f0 = np.bincount(ids[pre, j].ravel(), minlength=E) if pre.any() else np.zeros(E)
        f1 = np.bincount(ids[gen, j].ravel(), minlength=E) if gen.any() else np.zeros(E)
        entry["prefill_vs_gen_js_bits"] = js_divergence_bits(f0, f1)
        per_layer.append(entry)

    # global
    mem_all = np.zeros((n, E), dtype=bool)
    mem_all[np.repeat(np.arange(n), L * K), ids.reshape(-1).astype(np.int64)] = True
    items, n_items = access_items(ids, E, item)
    hist = stack_distance_hist(items, n_items)
    adj = [e["adjacent_reuse"] for e in per_layer if e["adjacent_reuse"] is not None]
    glob_: dict[str, Any] = {
        "unique_experts_per_token_mean": float(mem_all.sum(1).mean()),
        "items_never_used": int(np.count_nonzero(freq == 0)),
        "lru_hit_curve": lru_hit_curve(hist),
        "stack_distance_hist": [int(v) for v in hist],
        "adjacent_reuse_mean": float(np.mean(adj)) if adj else None,
        "corpus_overlap": _corpus_overlap(r, ids, E, corpus_top),
        "n_accesses": int(items.size),
        "n_items": int(n_items),
    }
    out: dict[str, Any] = {
        "trace": r.path_str,
        "n_tokens": int(n),
        "n_prefill": int(np.count_nonzero(pre)),
        "n_gen": int(np.count_nonzero(gen)),
        "train_frac": float(train_frac),
        "n_train": n_train,
        "n_test": int(n - n_train),
        "n_expert": E,
        "n_expert_used": K,
        "moe_layers": list(ml),
        "item": item,
        "truncated": bool(r.truncated),
        "per_layer": per_layer,
        "global": glob_,
        "cross_layer": _cross_layer(ids, n_train, E, K, ml),
    }
    if P > 0 and r.probs is not None:
        _pid, pp = r.probs
        pblock: dict[str, Any] = {"eighth_expert_mass_mean": float(w[:, :, K - 1].mean())}
        if P >= K + 1:
            margin = (pp[:, :, K - 1] - pp[:, :, K]).astype(np.float64).ravel()
            p10, p50, p90 = _percentiles(margin, [10, 50, 90])
            pblock["margin_p8_p9"] = {"p10": p10, "p50": p50, "p90": p90}
        else:
            pblock["margin_p8_p9"] = None
        out["probs"] = pblock
    return _clean(out)


def _corpus_overlap(r: TraceReader, ids: np.ndarray, E: int, top: int) -> dict[str, float]:
    segs = r.segments()
    src: dict[str, list[tuple[int, int]]] = {}
    for s in segs:
        if s["end"] > s["start"]:
            src.setdefault(str(s["chunk"].get("source", "")), []).append((s["start"], s["end"]))
    names = sorted(src)
    if len(names) < 2:
        return {}
    k = min(int(top), E)
    L = ids.shape[1]
    tops: dict[str, list[set[int]]] = {}
    for name in names:
        rows = np.concatenate([np.arange(a, b) for a, b in src[name]])
        sel = ids[rows]
        tops[name] = []
        for j in range(L):
            f = np.bincount(sel[:, j].ravel().astype(np.int64), minlength=E)
            tops[name].append(set(_topn(f.astype(np.float64), k).tolist()))
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            jac = [len(tops[a][j] & tops[b][j]) / max(1, len(tops[a][j] | tops[b][j])) for j in range(L)]
            out[f"{a}|{b}"] = float(np.mean(jac))
    return out


def write_stats_csv(st: Mapping[str, Any], csv_dir: str | os.PathLike[str], prefix: str) -> list[Path]:
    """Plot-friendly CSVs (not an interchange contract): <prefix>.per_layer.csv, .freq.csv,
    .lru_curve.csv, .cross_layer.csv."""
    d = Path(os.fspath(csv_dir))
    d.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pl = st["per_layer"]
    cov_keys = list(pl[0]["coverage_at"]) if pl else []
    ws_keys = list(pl[0]["working_set"]) if pl else []
    f1 = d / f"{prefix}.per_layer.csv"
    with open(f1, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["layer", "entropy_bits", "effective_experts", "gini"]
                    + [f"coverage_at_{c}" for c in cov_keys] + [f"coverage_weight_at_{c}" for c in cov_keys]
                    + ["adjacent_reuse", "jaccard_next_mean"]
                    + [f"working_set_{x}_median" for x in ws_keys] + [f"working_set_{x}_p95" for x in ws_keys]
                    + ["prefill_vs_gen_js_bits"])
        for e in pl:
            wsd = e["working_set"]
            wr.writerow([e["layer"], e["entropy_bits"], e["effective_experts"], e["gini"]]
                        + [e["coverage_at"][c] for c in cov_keys] + [e["coverage_weight_at"][c] for c in cov_keys]
                        + [e["adjacent_reuse"], e["jaccard_next_mean"]]
                        + [(wsd[x] or {}).get("median") for x in ws_keys] + [(wsd[x] or {}).get("p95") for x in ws_keys]
                        + [e["prefill_vs_gen_js_bits"]])
    written.append(f1)
    f2 = d / f"{prefix}.freq.csv"
    with open(f2, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["layer", "expert", "freq", "weight_mass"])
        for e in pl:
            for x, (c, m) in enumerate(zip(e["freq"], e["weight_mass"])):
                wr.writerow([e["layer"], x, c, m])
    written.append(f2)
    f3 = d / f"{prefix}.lru_curve.csv"
    with open(f3, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["capacity_items", "hit_rate"])
        wr.writerows(st["global"]["lru_hit_curve"])
    written.append(f3)
    f4 = d / f"{prefix}.cross_layer.csv"
    with open(f4, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        cols = ["layer", "next_layer", "precision_at_8", "precision_at_16", "baseline_static_top16",
                "baseline_same_ids", "mutual_info_bits", "mutual_info_bits_null"]
        wr.writerow(cols)
        for p in st["cross_layer"].get("per_pair", []):
            wr.writerow([p.get(c) for c in cols])
    written.append(f4)
    return written


# --------------------------------------------------------------------------------------------
# Sidecars: tokens, gen.txt, meta.json, probe summary
# --------------------------------------------------------------------------------------------
def read_tokens(path: str | os.PathLike[str]) -> np.ndarray:
    """Raw i32 LE token stream (``<trace>.tokens.bin`` or any --replay file) as int32."""
    return np.fromfile(os.fspath(path), dtype="<i4").astype(np.int32)


def read_gen(trace: str | os.PathLike[str]) -> dict[int, str]:
    """Sections of ``<trace>.gen.txt`` keyed by full chunk index (TRACE_FORMAT 4.3). {} if absent."""
    p = gen_path(trace) if not os.fspath(trace).endswith(SIDECAR_GEN) else Path(os.fspath(trace))
    if not p.is_file():
        return {}
    text = p.read_bytes().decode("utf-8", errors="replace")
    ms = list(_GEN_SEP_RE.finditer(text))
    out: dict[int, str] = {}
    for i, m in enumerate(ms):
        nxt = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        body = text[m.end():nxt]
        if body.startswith("\n"):
            body = body[1:]
        if i + 1 < len(ms) and body.endswith("\n"):
            body = body[:-1]
        out[int(m.group(1))] = body
    return out


def parse_probe_summary(text: str) -> dict[str, Any] | None:
    """The last ``PIGEMMA_SUMMARY {json}`` line of probe stderr, parsed; None if absent or broken."""
    found = None
    for line in text.splitlines():
        line = line.strip("\r")
        if line.startswith(PROBE_SUMMARY_PREFIX):
            try:
                obj = json.loads(line[len(PROBE_SUMMARY_PREFIX):])
            except json.JSONDecodeError:
                log.warning("unparsable %s line: %.120s", PROBE_SUMMARY_PREFIX.strip(), line)
                continue
            if isinstance(obj, dict):
                found = obj
    return found


def write_meta(stem: str | os.PathLike[str], meta: Mapping[str, Any]) -> Path:
    """Write ``<trace>.meta.json`` (``stem`` = trace path, or the meta path itself). Returns the path."""
    p = meta_path(stem)
    data = dict(meta)
    if "trace" not in data:
        s = os.fspath(stem)
        data["trace"] = s[: -len(SIDECAR_META)] if s.endswith(SIDECAR_META) else s
    _dump_json(data, p)
    return p


def read_meta(stem: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read ``<trace>.meta.json``; None if it does not exist."""
    p = meta_path(stem)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _host_path(p: str) -> Path:
    """A path as recorded inside Docker (/work/...) mapped to the repo on this host."""
    if p.startswith(paths.CONTAINER_ROOT + "/") and not Path(p).exists():
        return paths.REPO_ROOT / p[len(paths.CONTAINER_ROOT) + 1:]
    return Path(p)


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _coherence(trace: Path) -> tuple[bool | None, dict[str, bool], str | None]:
    sections = {i: t for i, t in read_gen(trace).items() if t.strip()}
    if not sections:
        return None, {}, None
    try:
        from pigemma import quality  # noqa: PLC0415 - optional, level-3 module
        fn = quality.is_coherent
    except (ImportError, AttributeError) as e:
        return None, {}, f"coherent=null: pigemma.quality.is_coherent unavailable ({e})"
    by: dict[str, bool] = {}
    for i, txt in sorted(sections.items()):
        res = fn(txt)
        if isinstance(res, tuple):
            res = res[0]
        if isinstance(res, dict):
            res = res.get("coherent")
        by[str(i)] = bool(res)
    return all(by.values()), by, None


def build_meta(
    trace: str | os.PathLike[str],
    *,
    gguf: str | os.PathLike[str] | None = None,
    corpus: Sequence[str | os.PathLike[str]] = (),
    argv: Sequence[str] | None = None,
    wall_ms: float | None = None,
    summary: Mapping[str, Any] | None = None,
    coherent: bool | None | str = "auto",
    host: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
    with_sha: bool = True,
) -> dict[str, Any]:
    """Assemble the ``<trace>.meta.json`` dict of TRACE_FORMAT 4.4 for a finished trace.

    gguf: default = the header's gguf_path (a /work/... path is mapped to the repo); its
    sha256 prefix (first 16 MiB) is computed when the file is reachable and ``with_sha``.
    summary: the parsed PIGEMMA_SUMMARY object (tok_s and callback overhead come from it).
    coherent: 'auto' (pigemma.quality.is_coherent per gen.txt chunk), True/False, or None.
    host: a host fingerprint captured elsewhere (e.g. inside Docker); default = this host.
    """
    tr = Path(os.fspath(trace))
    hdr = read_header(tr)
    note_list = [str(x) for x in notes]
    if host is None:
        host = paths.host_fingerprint()
        note_list.append("host = the machine that wrote this meta.json")
    gpath = _host_path(os.fspath(gguf)) if gguf is not None else _host_path(str(hdr.get("gguf_path", "")))
    sha = None
    if with_sha and str(gpath) and gpath.is_file():
        info = paths.sha256_prefix(gpath, SHA256_PREFIX_BYTES)
        sha = info["sha256"]
        if hdr.get("gguf_size_bytes") and info["size"] != hdr["gguf_size_bytes"]:
            note_list.append(f"gguf {gpath} size {info['size']} != header gguf_size_bytes {hdr['gguf_size_bytes']}")
    elif with_sha:
        note_list.append(f"gguf_sha256_prefix=null: {gpath} not reachable from this host")
    corpus_out = []
    for c in corpus:
        cp = Path(os.fspath(c))
        if cp.is_file():
            corpus_out.append({"path": str(c), "bytes": cp.stat().st_size, "sha256": _file_sha256(cp)})
        else:
            note_list.append(f"corpus file {c} not found")
    s = dict(summary or {})
    if coherent == "auto":
        coh, by, why = _coherence(tr)
    else:
        coh, by, why = (None if coherent is None else bool(coherent)), {}, None
    if why:
        note_list.append(why)
    meta: dict[str, Any] = {
        "trace": str(tr),
        "host": dict(host),
        "gguf_path": hdr.get("gguf_path"),
        "gguf_size_bytes": hdr.get("gguf_size_bytes"),
        "gguf_sha256_prefix": sha,
        "corpus": corpus_out,
        "argv": [str(a) for a in (argv if argv is not None else hdr.get("flags", []))],
        "wall_ms": None if wall_ms is None else float(wall_ms),
        "tok_s": {"prompt": s.get("tok_s_prompt"), "gen": s.get("tok_s_gen")},
        "callback_overhead_ms": s.get("callback_ms"),
        "coherent": coh,
        "chunks": hdr.get("chunks", []),
    }
    if by:
        meta["coherent_by_chunk"] = by
    if summary is not None:
        meta["probe_summary"] = dict(summary)
    if note_list:
        meta["notes"] = note_list
    return meta


# --------------------------------------------------------------------------------------------
# Activation dumps (H2), TRACE_FORMAT 7
# --------------------------------------------------------------------------------------------
def npy_header_f16(n_rows: int, ne0: int) -> bytes:
    """Canonical 128-byte .npy v1.0 header for a C-order '<f2' array of shape (n_rows, ne0)."""
    d = "{'descr': '<f2', 'fortran_order': False, 'shape': (%d, %d), }" % (int(n_rows), int(ne0))
    body = d.encode("latin1")
    pad = NPY_HEADER_BYTES - len(NPY_MAGIC_AND_VERSION) - 2 - len(body) - 1
    if pad < 0:
        raise ValueError("npy header does not fit 128 bytes")
    out = NPY_MAGIC_AND_VERSION + struct.pack("<H", NPY_HEADER_LEN_FIELD) + body + b" " * pad + b"\n"
    assert len(out) == NPY_HEADER_BYTES
    return out


def read_npy_header(path: str | os.PathLike[str]) -> tuple[str, bool, tuple[int, ...], int]:
    """(descr, fortran_order, shape, data_offset) of a .npy file (versions 1-3)."""
    with open(path, "rb") as f:
        pre = f.read(10)
        if len(pre) < 10 or pre[:6] != b"\x93NUMPY":
            raise ValueError(f"{path}: not a .npy file")
        major = pre[6]
        if major == 1:
            hlen = struct.unpack("<H", pre[8:10])[0]
            off = 10 + hlen
            f.seek(10)
        elif major in (2, 3):
            hlen = struct.unpack("<I", pre[8:10] + f.read(2))[0]
            off = 12 + hlen
            f.seek(12)
        else:
            raise ValueError(f"{path}: unsupported .npy version {major}")
        txt = f.read(hlen).decode("latin1" if major < 3 else "utf-8")
    d = ast.literal_eval(txt.strip())
    return str(d["descr"]), bool(d["fortran_order"]), tuple(int(x) for x in d["shape"]), off


_ACT_FILE_RE = re.compile(r"^L(\d+)\.([A-Za-z0-9_]+)\.npy$")


@dataclasses.dataclass
class Acts:
    """An activation dump directory (``read_acts_dir``)."""

    path: Path
    meta: dict[str, Any]
    n_tokens: int
    layers: list[int]
    kinds: dict[str, int]
    phase: np.ndarray | None
    token_ids: np.ndarray | None

    def file(self, layer: int, kind: str) -> Path:
        """Path of ``L<il>.<kind>.npy``."""
        return self.path / f"L{int(layer)}.{kind}.npy"

    def available(self) -> list[tuple[int, str]]:
        """(layer, kind) pairs that have a file, sorted."""
        out = []
        for p in self.path.glob("L*.npy"):
            m = _ACT_FILE_RE.match(p.name)
            if m:
                out.append((int(m.group(1)), m.group(2)))
        return sorted(out)

    def get(self, layer: int, kind: str) -> np.ndarray:
        """float16 memmap [T, ne0] of one (layer, kind); recovers unpatched files (TRACE_FORMAT 7.3)."""
        p = self.file(layer, kind)
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found (available: {self.available()[:8]})")
        descr, fortran, shape, off = read_npy_header(p)
        if descr != NPY_DESCR or fortran or len(shape) != 2:
            raise ValueError(f"{p}: expected '<f2' C-order 2-D, got descr={descr} fortran={fortran} shape={shape}")
        t_hdr, ne0 = shape
        exp_ne0 = self.kinds.get(kind)
        if exp_ne0 is not None and int(exp_ne0) != ne0:
            warnings.warn(f"{p}: ne0 {ne0} != meta kinds[{kind}] = {exp_ne0}", TraceWarning, stacklevel=2)
        size = p.stat().st_size
        t_file = (size - off) // (2 * ne0) if ne0 else 0
        rows = min(t_hdr, t_file)
        if size != off + t_hdr * ne0 * 2:
            warnings.warn(f"{p}: unpatched npy (header rows {t_hdr}, file holds {t_file}); using {rows}",
                          TraceWarning, stacklevel=2)
        if self.n_tokens and rows > self.n_tokens:
            rows = self.n_tokens
        elif self.n_tokens and rows < self.n_tokens:
            warnings.warn(f"{p}: {rows} rows < meta n_tokens {self.n_tokens}", TraceWarning, stacklevel=2)
        if rows == 0:
            return np.zeros((0, ne0), dtype=np.float16)
        return np.memmap(p, dtype="<f2", mode="r", offset=off, shape=(rows, ne0))


def read_acts_dir(directory: str | os.PathLike[str]) -> Acts:
    """Open an activation dump directory (TRACE_FORMAT 7.2): meta.json, L<il>.<kind>.npy, phase.u8, tokens.bin."""
    d = Path(os.fspath(directory))
    if not d.is_dir():
        raise FileNotFoundError(f"acts directory not found: {d}")
    mp = d / ACTS_META_FILE
    meta: dict[str, Any] = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    if not meta:
        warnings.warn(f"{d}: no meta.json; layers/kinds inferred from file names", TraceWarning, stacklevel=2)
    files = []
    for p in d.glob("L*.npy"):
        m = _ACT_FILE_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), m.group(2), p))
    kinds = {str(k): int(v) for k, v in (meta.get("kinds") or {}).items()}
    for _il, kind, p in files:
        if kind not in kinds:
            with contextlib.suppress(Exception):
                kinds[kind] = read_npy_header(p)[2][1]
    layers = sorted({int(x) for x in meta.get("layers", [])} | {il for il, _k, _p in files})
    phase_file = d / str(meta.get("phase_file", ACTS_PHASE_FILE))
    tok_file = d / str(meta.get("tokens_file", ACTS_TOKENS_FILE))
    phase = np.fromfile(phase_file, dtype=np.uint8) if phase_file.is_file() else None
    toks = np.fromfile(tok_file, dtype="<i4").astype(np.int32) if tok_file.is_file() else None
    n_tokens = meta.get("n_tokens")
    if not _is_int(n_tokens):
        cands = [len(a) for a in (phase, toks) if a is not None]
        n_tokens = min(cands) if cands else 0
    n_tokens = int(n_tokens)
    if phase is not None:
        phase = phase[:n_tokens]
    if toks is not None:
        toks = toks[:n_tokens]
    return Acts(path=d, meta=meta, n_tokens=n_tokens, layers=layers, kinds=kinds, phase=phase, token_ids=toks)


def write_acts_dir(
    directory: str | os.PathLike[str],
    arrays: Mapping[tuple[int, str], Any],
    *,
    phase: Any = None,
    token_ids: Any = None,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write a synthetic activation dump (same bytes as the probe: canonical 128-byte headers).

    ``arrays`` maps (layer, kind) -> [T, ne0] array (stored as '<f2'); all must share T.
    Defaults: phase all 0, token ids 0..T-1. ``meta`` keys override the generated meta.json.
    """
    d = Path(os.fspath(directory))
    d.mkdir(parents=True, exist_ok=True)
    ts = {np.asarray(a).shape[0] for a in arrays.values()}
    if len(ts) > 1:
        raise ValueError(f"all arrays must have the same number of rows, got {sorted(ts)}")
    t = ts.pop() if ts else 0
    kinds: dict[str, int] = {}
    for (il, kind), a in sorted(arrays.items()):
        arr = np.ascontiguousarray(np.asarray(a, dtype="<f2"))
        if arr.ndim != 2:
            raise ValueError(f"(L{il}, {kind}) must be 2-D")
        kinds[kind] = int(arr.shape[1])
        with open(d / f"L{int(il)}.{kind}.npy", "wb") as f:
            f.write(npy_header_f16(arr.shape[0], arr.shape[1]))
            f.write(arr.tobytes())
    ph = np.zeros(t, np.uint8) if phase is None else np.asarray(phase, dtype=np.uint8)
    tk = np.arange(t, dtype="<i4") if token_ids is None else np.asarray(token_ids, dtype="<i4")
    if ph.shape[0] != t or tk.shape[0] != t:
        raise ValueError("phase/token_ids length must equal the number of rows")
    ph.tofile(d / ACTS_PHASE_FILE)
    tk.astype("<i4").tofile(d / ACTS_TOKENS_FILE)
    m: dict[str, Any] = {
        "model": "", "gguf_path": "", "layers": sorted({int(il) for il, _k in arrays}),
        "kinds": kinds, "n_tokens": int(t), "phase_file": ACTS_PHASE_FILE, "tokens_file": ACTS_TOKENS_FILE,
        "chunks": [], "llama_build": int(paths.LLAMA_BUILD), "n_ubatch": 0, "flags": [],
    }
    m.update(meta or {})
    _dump_json(m, d / ACTS_META_FILE)
    return d


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _summary_lines(st: Mapping[str, Any]) -> list[str]:
    pl = st["per_layer"]
    g = st["global"]
    cl = st["cross_layer"]
    u = g.get("n_items") or len(g["stack_distance_hist"]) - 1
    curve = g["lru_hit_curve"]

    def _hit(frac: float) -> str:
        c = max(1, min(u, int(round(frac * u))))
        return f"{curve[c - 1][1]:.3f}@{c}"

    def _fmt(v: Any, f: str = ".3f") -> str:
        return "null" if v is None else format(v, f)

    return [
        f"trace {st['trace']}: {st['n_tokens']} records ({st['n_prefill']} prefill, {st['n_gen']} gen), "
        f"{len(pl)} MoE layers, E={st['n_expert']}, K={st['n_expert_used']}",
        f"entropy mean {np.mean([e['entropy_bits'] for e in pl]):.3f} bits, effective experts mean "
        f"{np.mean([e['effective_experts'] for e in pl]):.1f}, gini mean {np.mean([e['gini'] for e in pl]):.3f}",
        "coverage mean: " + ", ".join(
            f"@{k} {np.mean([e['coverage_at'][k] for e in pl]):.3f}" for k in (pl[0]["coverage_at"] if pl else {})),
        f"adjacent reuse mean {_fmt(g['adjacent_reuse_mean'])}, unique experts/token mean "
        f"{g['unique_experts_per_token_mean']:.1f}, items never used {g['items_never_used']}",
        "LRU overall hit (fraction of items): " + ", ".join(_hit(f) for f in (0.0625, 0.125, 0.25, 0.5, 0.75)),
        f"cross-layer precision@8 {_fmt(cl['precision_at_8'])} @16 {_fmt(cl['precision_at_16'])} | "
        f"static top16 {_fmt(cl['baseline_static_top16'])} | same ids {_fmt(cl['baseline_same_ids'])} | "
        f"MI {_fmt(cl['mutual_info_bits_mean'])} bits (shuffle null {_fmt(cl.get('mutual_info_bits_null_mean'))})",
    ]


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m pigemma.trace",
        description="Reader / validator / merger / statistics for .moetrace expert-routing traces "
                    "(format: docs/TRACE_FORMAT.md).",
        epilog="Exit codes: 0 ok, 1 error (validate: any E-code), 2 usage, 3 missing input.",
    )
    ap.add_argument("--log-level", default="INFO", help="logging level on stderr (default INFO)")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")
    v = sub.add_parser("validate", help="check traces against TRACE_FORMAT 3.9; exit 1 iff any error")
    v.add_argument("traces", nargs="+", help="trace path(s)")
    v.add_argument("--json", action="store_true", help="print the reports as JSON")
    v.add_argument("--quiet", action="store_true", help="do not print warnings")
    i = sub.add_parser("info", help="JSON: header, counts, chunks and sidecars of a trace")
    i.add_argument("trace")
    i.add_argument("--no-header", action="store_true", help="omit the JSON header")
    s = sub.add_parser("stats", help="H1 statistics (stats.json, TRACE_FORMAT 5)")
    s.add_argument("trace")
    s.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC, help="train split fraction (default 0.7)")
    s.add_argument("--item", choices=("layer-expert", "expert"), default="layer-expert",
                   help="cache item for the stack-distance curve (default layer-expert)")
    s.add_argument("--out", help="stats.json path (default: print JSON to stdout)")
    s.add_argument("--csv-dir", help="also write plot CSVs <trace-stem>.*.csv into this directory")
    s.add_argument("--no-mmap", action="store_true", help="read the trace into memory instead of mmap")
    m = sub.add_parser("merge", help="merge traces: merge <out> <in...> (globs expanded; <out> skipped)")
    m.add_argument("out")
    m.add_argument("inputs", nargs="+")
    m.add_argument("--no-gen", action="store_true", help="do not concatenate gen.txt sidecars")
    m.add_argument("--no-meta", action="store_true", help="do not write <out>.meta.json")
    mt = sub.add_parser("meta", help="write <trace>.meta.json (TRACE_FORMAT 4.4)")
    mt.add_argument("trace")
    mt.add_argument("--gguf", help="GGUF path for the sha256 prefix (default: header gguf_path, /work mapped)")
    mt.add_argument("--corpus", nargs="*", default=[], help="input text/prompt files (sha256 of each)")
    mt.add_argument("--argv-json", help="probe argv as a JSON list (default: header flags)")
    mt.add_argument("--wall-ms", type=float, help="wall time of the probe process in ms")
    mt.add_argument("--summary-file", help="file holding the probe stderr (PIGEMMA_SUMMARY line)")
    mt.add_argument("--host-json", help="host fingerprint JSON captured where the probe ran")
    mt.add_argument("--coherent", default="auto", choices=("auto", "true", "false", "null"),
                    help="coherence flag (auto = pigemma.quality.is_coherent per gen.txt chunk)")
    mt.add_argument("--note", action="append", default=[], help="add a note (repeatable)")
    mt.add_argument("--no-sha", action="store_true", help="skip the 16 MiB GGUF sha256 prefix")
    mt.add_argument("--out", help="meta path (default <trace>.meta.json)")
    a = sub.add_parser("acts", help="JSON summary of an activation dump directory")
    a.add_argument("dir")
    return ap


def _cmd_validate(args: argparse.Namespace) -> int:
    any_err = False
    reports = []
    for t in args.traces:
        rep = validate_report(t)
        reports.append(rep)
        any_err |= not rep.ok
        if not args.json:
            status = "OK" if rep.ok else "INVALID"
            extra = f", {rep.n_records} records" if rep.n_records is not None else ""
            print(f"{status}: {t}{extra}" + (" (truncated)" if rep.truncated else ""))
            for e in rep.errors:
                print(f"  {e}")
            if not args.quiet:
                for w in rep.warnings:
                    print(f"  warning {w}")
    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=1, ensure_ascii=False))
    return paths.EXIT_ERROR if any_err else paths.EXIT_OK


def _cmd_stats(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    with TraceReader(args.trace, use_mmap=not args.no_mmap) as r:
        st = stats(r, train_frac=args.train_frac, item=args.item)
    dt = time.perf_counter() - t0
    if args.out:
        _dump_json(st, Path(args.out), indent=None)
        for line in _summary_lines(st):
            print(line)
        print(f"wrote {args.out} ({dt:.1f} s)")
    else:
        print(json.dumps(st, ensure_ascii=False, allow_nan=False))
        for line in _summary_lines(st):
            log.info("%s", line)
    if args.csv_dir:
        stem = Path(args.trace).name
        if stem.endswith(".moetrace"):
            stem = stem[: -len(".moetrace")]
        for p in write_stats_csv(st, args.csv_dir, stem):
            log.info("wrote %s", p)
    return paths.EXIT_OK


def _cmd_meta(args: argparse.Namespace) -> int:
    summary = None
    if args.summary_file:
        summary = parse_probe_summary(Path(args.summary_file).read_text(encoding="utf-8", errors="replace"))
        if summary is None:
            log.warning("no %sline in %s", PROBE_SUMMARY_PREFIX, args.summary_file)
    host = json.loads(Path(args.host_json).read_text(encoding="utf-8")) if args.host_json else None
    argv = json.loads(args.argv_json) if args.argv_json else None
    if argv is not None and not (isinstance(argv, list) and all(isinstance(x, str) for x in argv)):
        raise ValueError("--argv-json must be a JSON list of strings")
    coh: bool | None | str = {"auto": "auto", "true": True, "false": False, "null": None}[args.coherent]
    meta = build_meta(args.trace, gguf=args.gguf, corpus=args.corpus, argv=argv, wall_ms=args.wall_ms,
                      summary=summary, coherent=coh, host=host, notes=args.note, with_sha=not args.no_sha)
    p = write_meta(args.out or args.trace, meta)
    print(p)
    return paths.EXIT_OK


def _cmd_acts(args: argparse.Namespace) -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        acts = read_acts_dir(args.dir)
        files = []
        for il, kind in acts.available():
            arr = acts.get(il, kind)
            files.append({"layer": il, "kind": kind, "shape": list(arr.shape),
                          "finite": bool(np.isfinite(np.asarray(arr, dtype=np.float32)).all())})
            del arr
    out = {
        "dir": str(acts.path), "n_tokens": acts.n_tokens, "layers": acts.layers, "kinds": acts.kinds,
        "phase": None if acts.phase is None else {"n": int(acts.phase.size),
                                                  "n_gen": int(np.count_nonzero(acts.phase == 1))},
        "tokens": None if acts.token_ids is None else int(acts.token_ids.size),
        "files": files, "warnings": [str(w.message) for w in caught],
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return paths.EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the exit code."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    ap = _build_parser()
    try:
        args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else paths.EXIT_USAGE
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, str(args.log_level).upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
    if args.cmd is None:
        ap.print_help()
        return paths.EXIT_USAGE
    logging.captureWarnings(True)
    try:
        return _dispatch(ap, args)
    finally:
        logging.captureWarnings(False)


def _dispatch(ap: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        if args.cmd == "validate":
            return _cmd_validate(args)
        if args.cmd == "info":
            p = Path(args.trace)
            if not p.is_file():
                sys.stderr.write(f"pigemma.trace: file not found: {p}\n")
                return paths.EXIT_MISSING
            print(json.dumps(_clean(trace_info(p, include_header=not args.no_header)), indent=1, ensure_ascii=False))
            return paths.EXIT_OK
        if args.cmd == "stats":
            if not Path(args.trace).is_file():
                sys.stderr.write(f"pigemma.trace: file not found: {args.trace}\n")
                return paths.EXIT_MISSING
            return _cmd_stats(args)
        if args.cmd == "merge":
            res = merge(args.inputs, args.out, argv=["pigemma.trace", "merge", args.out, *args.inputs],
                        write_gen=not args.no_gen, write_meta_file=not args.no_meta)
            print(json.dumps(res, indent=1, ensure_ascii=False))
            return paths.EXIT_OK
        if args.cmd == "meta":
            if not Path(args.trace).is_file():
                sys.stderr.write(f"pigemma.trace: file not found: {args.trace}\n")
                return paths.EXIT_MISSING
            return _cmd_meta(args)
        if args.cmd == "acts":
            return _cmd_acts(args)
    except FileNotFoundError as e:
        sys.stderr.write(f"pigemma.trace: {e}\n")
        return paths.EXIT_MISSING
    except (TraceFormatError, MergeError, ValueError) as e:
        sys.stderr.write(f"pigemma.trace: error: {e}\n")
        return paths.EXIT_ERROR
    except Exception as e:  # noqa: BLE001 - CLI boundary
        sys.stderr.write(f"pigemma.trace: error: {type(e).__name__}: {e}\n")
        return paths.EXIT_ERROR
    ap.print_help()
    return paths.EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
