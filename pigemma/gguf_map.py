"""pigemma.gguf_map - GGUF byte-range planner, per-token traffic accounting and hot-set planner.

What it does (docs/design/manifest.json files[] 'pigemma/gguf_map.py'; formats in
docs/TRACE_FORMAT.md sections 8 and 9):

* ``build_map(path)`` reads a GGUF header and returns a :class:`GgufMap`: every tensor as an
  absolute file byte range, classified into a kind, plus the per-layer expert slabs (expert ``e``
  of an ``ffn_*_exps`` tensor = ``[offset + e*nb_expert, offset + (e+1)*nb_expert)``), the
  architecture keys, and a summary (non-expert / expert bytes, expert unit, worst-case expert
  bytes per token, non-expert bytes per token, KV-cache bytes). ``GgufMap.save`` writes map.json.
* ``plan_hotset(map, stats, budget)`` picks what to keep resident on a small machine: all
  non-expert tensors (mandatory unless disabled; includes the tied output head) and then the
  hottest experts from a trace's ``stats.json`` until the byte budget is used. The result is
  page-aligned, adjacent ranges are merged, and it is written as ``hotset.json`` and
  ``hotset.ranges.txt`` (the format ``tools/pinner`` reads; no JSON parser in C).
* ``catalog_refresh()`` fills exact byte sizes and LFS sha256 (and optionally the exact
  non-expert/expert split from a remote header read) of ``pigemma/data/catalog.json`` entries
  from the Hugging Face listing. PC only (needs ``huggingface_hub`` and network).

CLI (``python -m pigemma.gguf_map <cmd> --help``)::

    build <gguf|model-id> --out map.json [--page-size 16384] [--backend auto|gguf|stdlib]
          [--no-sha] [--reuse] [--json]
    summary --map map.json [--json]
    hotset --map map.json [--stats stats.json] --budget-bytes N [--page-size 16384]
           [--no-nonexpert] [--per-layer-min 0] [--rank count|weight] [--min-hits 0]
           --out hotset.json [--ranges-out hotset.ranges.txt]
    catalog-refresh [--catalog PATH] [--ids a,b] [--all] [--split] [--dry-run] [--out PATH]
                    [--token-file FILE]

Exit codes: 0 ok; 1 error (malformed GGUF, missing KV key, network, inconsistent stats);
2 usage; 3 missing input file (GGUF, map, stats) or missing optional dependency
(huggingface_hub); 4 refused (hot-set budget smaller than the mandatory non-expert set, or a
budget that fits nothing).

Design decisions (the manifest left these open; listed again in the agent report):

* Two header readers. ``backend='gguf'`` is gguf-py's ``GGUFReader`` (the spec's reader; its
  ``ReaderTensor.data_offset`` is absolute). ``backend='stdlib'`` is a small pure-Python parser
  (struct only) of the same header, used automatically when gguf-py cannot be imported: the Pi
  and the emulated arm64 container may have no numpy/pyyaml, and ``pi/06_pin_hotset.sh`` builds
  the map on the Pi. It is also what ``catalog-refresh --split`` uses on an HTTP range stream.
  The unit test asserts that both readers produce identical maps. ``auto`` = gguf-py if
  importable, else stdlib. ``hotset``/``summary`` never import numpy or gguf (stdlib only).
* Kinds: ``attn`` = the ``attn_{q,k,v,o,output,qkv}`` matmul weights only; their q/k norms are
  ``norm`` (TRACE_FORMAT 8.2 allows either). ``blk.N.layer_output_scale.weight`` (a per-layer
  gain, read every token like a norm) is ``norm``; ``rope_freqs.weight`` is ``other``.
* ``nonexpert_bytes_per_token`` = attn + ffn_dense + norm + router + expert_scale + other +
  (``output`` if a separate output tensor exists, else ``token_embd``: the tied head is read
  every token) + one embedding row. ``other`` is added to the spec's formula because those
  tensors (rope_freqs) are also read every token; for Gemma 4 it is < 2 KB. For the tied Gemma 4
  GGUFs this equals ``nonexpert_bytes + embd_row_bytes``. NOTE: catalog.json uses the simpler
  convention ``nonexpert_bytes_per_token == nonexpert_bytes`` (asserted by tests/test_catalog.py),
  so catalog-refresh writes that convention, not the map's value (2,310 B apart for the 26B).
* Non-uniform expert quants (unsloth UD): ``expert_unit_bytes`` is the per-layer mean rounded to
  an int (exact for uniform quants), with ``expert_unit_bytes_min``/``_max``; and
  ``expert_bytes_per_token_worst = n_expert_used * sum over MoE layers of the layer's unit``,
  which equals the spec's ``n_moe_layers*n_expert_used*unit`` for uniform quants and is exact
  for UD quants (matches catalog.json). Each ``expert_layers[]`` entry also carries ``unit_bytes``
  and ``n_expert``.
* KV bytes: V uses ``value_length``/``value_length_swa`` when present (equal to the key lengths
  in Gemma 4, so the spec's ``key_length*2*2`` holds). Global Gemma 4 layers share K and V
  (no attn_v tensor); both are counted, as the spec says (upper bound). ``kv_bytes(n_ctx)`` caps
  sliding layers at ``sliding_window`` tokens (llama.cpp's SWA cache is a little larger:
  window + ubatch, padded), so it is an approximation of the allocated cache.
* Integer ``sliding_window_pattern`` N follows llama.cpp ``set_swa_pattern``: layer il is
  sliding iff ``il % N < N-1``. A bool array is used as is. If ``sliding_window`` is set but
  there is no pattern, every layer is treated as global (upper bound) with a warning.
* Hot-set budget accounting is done on page-aligned, merged bytes, so ``total_bytes <=
  budget_bytes`` always (stronger than the spec's "budget + one page per range"). Aligned range
  ends are clamped to the file size (the pinner grammar requires ``offset + length <= size``).
  If the mandatory non-expert set alone exceeds the budget the planner refuses (exit 4) instead
  of pinning a partial set; ``--no-nonexpert`` is the explicit escape hatch.
* Expert selection is greedy: first ``per_layer_min`` rounds (each layer's r-th hottest expert,
  layers in order), then all experts by descending score (``freq`` or ``weight_mass``), ties by
  (layer, expert). An expert that no longer fits is skipped and smaller ones may still be added.
  Experts with ``freq < min_hits`` are not added by the global phase (default 0 = all experts are
  candidates; never-used experts rank last).
* Merged-range metadata: ``kind`` = common kind or ``"mixed"``, ``layer`` = common layer or
  null, ``expert`` = the id when every part is the same expert id, ``hits`` = sum of the expert
  parts' score (null when the range holds no expert slab), ``n_parts``.
* ``map_sha256`` is the SHA-256 of the map.json bytes; for an in-memory map that was never
  saved, of its canonical JSON (sorted keys).
* The ranges file sits next to the JSON: ``hotset.json`` -> ``hotset.ranges.txt`` (what
  pi/06_pin_hotset.sh and pc/pagecache_limit.sh expect); ``--ranges-out`` overrides it.
* catalog-refresh queries the ``main`` revision and records ``info.sha`` as ``revision``; it
  rewrites catalog.json with a formatter that reproduces the hand-written layout byte for byte
  (2-space indent, short scalar lists inline), so a refresh produces a minimal diff. HF token:
  env HF_TOKEN / HUGGING_FACE_TOKEN, else ``--token-file`` / env PIGEMMA_HF_TOKEN_FILE.
  There is no default token path. The token is never printed. The listed repos are
  ungated, so anonymous access works too.

Public API: build_map, read_gguf_header, GgufHeader, TensorRecord, ArraySummary, GgufMap,
load_map, classify_tensor, aligned, merge_ranges, kv_bytes_for, plan_hotset, write_hotset,
format_ranges_txt, read_ranges_txt, ranges_txt_path, load_stats, format_summary,
catalog_refresh, format_catalog_json, GGML_TYPES, KINDS, exceptions GgufMapError,
MissingKeyError, HotsetError, HotsetBudgetError, main.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import datetime as _dt
import fnmatch
import gc
import hashlib
import json
import logging
import math
import os
import re
import struct
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

from pigemma import __version__, paths

log = logging.getLogger("pigemma.gguf_map")

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
MAP_FORMAT_VERSION = 1
HOTSET_FORMAT_VERSION = 1
DEFAULT_PAGE_SIZE = 16384  # Pi 5 kernel page size; the default planning page (TRACE_FORMAT 8.2)
GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32
RANGES_FIRST_LINE_PREFIX = "# pigemma-ranges v1 "
KV_REFERENCE_CTX = 4096
CATALOG_INLINE_MAX = 100
GENERATOR = f"pigemma.gguf_map/{__version__}"
DEFAULT_HF_TOKEN_FILES: tuple[str, ...] = ()

KINDS: tuple[str, ...] = (
    "token_embd", "output", "attn", "ffn_dense", "norm", "router", "expert_scale", "expert", "other",
)
# Kinds read on every decoded token besides the head and the embedding row.
_PER_TOKEN_KINDS: tuple[str, ...] = ("attn", "ffn_dense", "norm", "router", "expert_scale", "other")
_EXPERT_ROLES: tuple[str, ...] = ("gate_up", "gate", "up", "down")

# ggml tensor types at llama.cpp a7b94df (gguf-py constants.GGML_QUANT_SIZES):
# id -> (name, block_size in elements, bytes per block). Cross-checked by tests/test_gguf_map.py.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}

# KV-cache bytes per 32 values for llama.cpp's -ctk/-ctv types (f16 = 64).
KV_CACHE_BYTES_PER_32: dict[str, int] = {
    "f32": 128, "f16": 64, "bf16": 64, "q8_0": 34, "q5_1": 24, "q5_0": 22, "q4_1": 20,
    "q4_0": 18, "iq4_nl": 18,
}

# GGUF value types (gguf-py GGUFValueType).
_GV_STRING = 8
_GV_ARRAY = 9
_SCALAR_FMT: dict[int, str] = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}
_MAX_DECODE_NUMERIC = 65536  # larger numeric arrays (tokenizer scores) are summarised, not decoded
_MAX_DECODE_STRINGS = 4096   # larger string arrays (vocab, merges) are summarised

# Tensor classification (TRACE_FORMAT 8.2; the first three rules are normative).
_RE_LAYER = re.compile(r"^blk\.(\d+)\.")
_RE_EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$")
_RE_EXPERT_SCALE = re.compile(r"^blk\.(\d+)\.ffn_.*_exps\.scale$")
_RE_ROUTER = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.")
_RE_TOKEN_EMBD = re.compile(r"^token_embd\.")
_RE_OUTPUT = re.compile(r"^output\.weight$")
_RE_ATTN = re.compile(r"^blk\.\d+\.attn_(q|k|v|o|output|qkv)\.(weight|bias)$")
_RE_FFN_DENSE = re.compile(r"^blk\.\d+\.ffn_(gate|up|down|gate_up)(_shexp)?\.(weight|bias)$")
_RE_NORM = re.compile(r"(^|\.)[a-z0-9_]*norm[a-z0-9_]*\.(weight|bias)$")
_RE_LAYER_SCALE = re.compile(r"^blk\.\d+\.layer_output_scale\.weight$")

_MAP_REQUIRED: tuple[str, ...] = (
    "gguf", "file_size", "data_offset", "page_size", "n_layer", "n_expert", "n_expert_used",
    "tensors", "expert_layers", "summary",
)


# --------------------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------------------
class GgufMapError(ValueError):
    """Malformed, truncated or unsupported GGUF file, or an inconsistent map.json."""


class MissingKeyError(GgufMapError):
    """A required GGUF KV key is absent; ``key`` names it."""

    def __init__(self, key: str, where: str) -> None:
        self.key = key
        super().__init__(f"missing GGUF KV key {key!r} in {where}")


class HotsetError(ValueError):
    """Invalid hot-set request (bad arguments, stats inconsistent with the map)."""


class HotsetBudgetError(HotsetError):
    """The budget cannot hold the mandatory part of the hot set (exit code 4 on the CLI)."""


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _align_up(x: int, a: int) -> int:
    return -(-x // a) * a


def _py(v: Any) -> Any:
    """Convert numpy scalars / nested lists of them to plain Python values."""
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    item = getattr(v, "item", None)
    if callable(item):
        return item()
    return v


def _jsonable(v: Any) -> Any:
    """Header value for map.json: ArraySummary (skipped big arrays) becomes None."""
    if isinstance(v, ArraySummary):
        return None
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return _py(v)


def _fmt_int(n: int | None) -> str:
    return "-" if n is None else f"{n:,}"


def _gb(n: int | float | None) -> str:
    return "-" if n is None else f"{n / 1e9:.3f}"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text via a temp file + os.replace (no BOM, LF line endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def _dump_json(obj: Any, indent: int | None = 1) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False, allow_nan=False) + "\n"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# Byte-range helpers
# --------------------------------------------------------------------------------------------
def aligned(offset: int, length: int, page: int = DEFAULT_PAGE_SIZE,
            limit: int | None = None) -> tuple[int, int]:
    """Page-align a byte range: floor the start, ceil the end. Returns ``(offset, length)``.

    ``limit`` (the file size) clamps the aligned end, so the result never runs past the end of
    the file. A zero-length range stays zero-length at its (floored) start.
    """
    if page <= 0:
        raise ValueError(f"page must be > 0, got {page}")
    if offset < 0 or length < 0:
        raise ValueError(f"negative range ({offset}, {length})")
    start = (offset // page) * page
    if length == 0:
        return start, 0
    end = _align_up(offset + length, page)
    if limit is not None:
        if offset + length > limit:
            raise ValueError(f"range ({offset}, {length}) ends past limit {limit}")
        end = min(end, limit)
    return start, end - start


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort ``(offset, length)`` ranges and merge overlapping or touching ones (zero lengths dropped)."""
    items = sorted((int(o), int(n)) for o, n in ranges if int(n) > 0)
    out: list[list[int]] = []
    for o, n in items:
        if out and o <= out[-1][1]:
            out[-1][1] = max(out[-1][1], o + n)
        else:
            out.append([o, o + n])
    return [(a, b - a) for a, b in out]


class _IntervalSet:
    """Disjoint, non-touching half-open intervals kept sorted; tracks the covered byte total."""

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.total = 0

    def uncovered(self, a: int, b: int) -> int:
        """Bytes of [a, b) not yet covered."""
        if b <= a:
            return 0
        covered = 0
        i = bisect.bisect_right(self.ends, a)
        n = len(self.starts)
        while i < n and self.starts[i] < b:
            covered += min(b, self.ends[i]) - max(a, self.starts[i])
            i += 1
        return (b - a) - covered

    def add(self, a: int, b: int) -> None:
        """Insert [a, b), merging with overlapping or touching intervals."""
        if b <= a:
            return
        i = bisect.bisect_left(self.ends, a)  # first interval whose end >= a (touching counts)
        j = i
        na, nb = a, b
        old = 0
        n = len(self.starts)
        while j < n and self.starts[j] <= b:
            na = min(na, self.starts[j])
            nb = max(nb, self.ends[j])
            old += self.ends[j] - self.starts[j]
            j += 1
        self.starts[i:j] = [na]
        self.ends[i:j] = [nb]
        self.total += (nb - na) - old

    def intervals(self) -> list[tuple[int, int]]:
        return list(zip(self.starts, self.ends))


# --------------------------------------------------------------------------------------------
# GGUF header reading (two backends)
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ArraySummary:
    """Stand-in for a large GGUF array value that was skipped, not decoded (e.g. the vocab)."""

    elem_type: int
    count: int


@dataclass
class TensorRecord:
    """One tensor-info entry: ``shape`` in gguf order (ne[0] fastest), ``offset`` absolute."""

    name: str
    type_id: int
    type: str
    shape: list[int]
    n_bytes: int
    offset: int


@dataclass
class GgufHeader:
    """Everything build_map needs from a GGUF header."""

    path: str
    version: int
    file_size: int
    alignment: int
    data_offset: int
    fields: dict[str, Any]
    tensors: list[TensorRecord]
    backend: str


def _tensor_nbytes(dims: Sequence[int], type_id: int, name: str, where: str) -> tuple[str, int]:
    if type_id not in GGML_TYPES:
        raise GgufMapError(f"{where}: tensor {name!r} has unknown ggml type id {type_id}")
    tname, block, tsize = GGML_TYPES[type_id]
    n_el = 1
    for d in dims:
        n_el *= int(d)
    if dims and int(dims[0]) % block != 0:
        raise GgufMapError(f"{where}: tensor {name!r} ne[0]={dims[0]} is not a multiple of the "
                           f"{tname} block size {block}")
    return tname, n_el * tsize // block


class _ChunkReader:
    """Sequential little-endian reader over a binary stream; keeps what it read in one bytearray.

    GGUF headers are a few MB to ~20 MB (the vocab), so holding the header in memory is fine even
    on a 4 GB Pi. Works on any object with ``read(n)`` (local file, HfFileSystem file).
    """

    def __init__(self, f: BinaryIO, label: str, chunk: int = 4 << 20) -> None:
        self.f = f
        self.label = label
        self.chunk = chunk
        self.buf = bytearray()
        self.pos = 0

    def need(self, n: int) -> None:
        short = self.pos + n - len(self.buf)
        while short > 0:
            data = self.f.read(max(self.chunk, short))
            if not data:
                raise GgufMapError(
                    f"{self.label}: unexpected end of file while parsing the GGUF header at byte "
                    f"{self.pos} (truncated or partial download?)")
            self.buf += data
            short = self.pos + n - len(self.buf)

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        self.need(size)
        v = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += size
        return v

    def string(self) -> str:
        (n,) = self.unpack("<Q")
        if n > (1 << 31):
            raise GgufMapError(f"{self.label}: implausible string length {n} at byte {self.pos - 8}")
        self.need(n)
        s = bytes(self.buf[self.pos:self.pos + n]).decode("utf-8", errors="replace")
        self.pos += n
        return s

    def skip(self, n: int) -> None:
        self.need(n)
        self.pos += n

    def skip_strings(self, count: int) -> None:
        buf = self.buf  # bytearray grows in place, so this reference stays valid
        pos = self.pos
        unpack = struct.unpack_from
        for _ in range(count):
            if pos + 8 > len(buf):
                self.pos = pos
                self.need(8)
            (n,) = unpack("<Q", buf, pos)
            pos += 8
            if pos + n > len(buf):
                self.pos = pos
                self.need(n)
            pos += n
        self.pos = pos


def _read_value(r: _ChunkReader, vtype: int, key: str) -> Any:
    if vtype in _SCALAR_FMT:
        return r.unpack("<" + _SCALAR_FMT[vtype])[0]
    if vtype == _GV_STRING:
        return r.string()
    if vtype == _GV_ARRAY:
        (etype,) = r.unpack("<I")
        (count,) = r.unpack("<Q")
        if etype in _SCALAR_FMT:
            fmt = _SCALAR_FMT[etype]
            size = struct.calcsize(fmt)
            if count <= _MAX_DECODE_NUMERIC:
                r.need(size * count)
                vals = list(struct.unpack_from(f"<{count}{fmt}", r.buf, r.pos))
                r.pos += size * count
                return vals
            r.skip(size * count)
            return ArraySummary(etype, count)
        if etype == _GV_STRING:
            if count <= _MAX_DECODE_STRINGS:
                return [r.string() for _ in range(count)]
            r.skip_strings(count)
            return ArraySummary(etype, count)
        if etype == _GV_ARRAY:
            return [_read_value(r, _GV_ARRAY, key) for _ in range(count)]
        raise GgufMapError(f"{r.label}: unknown GGUF array element type {etype} for key {key!r}")
    raise GgufMapError(f"{r.label}: unknown GGUF value type {vtype} for key {key!r}")


def _parse_stdlib(f: BinaryIO, label: str, file_size: int) -> GgufHeader:
    """Pure-Python GGUF v2/v3 header parser (little-endian files only)."""
    r = _ChunkReader(f, label)
    r.need(4)
    magic = bytes(r.buf[0:4])
    r.pos = 4
    if magic != GGUF_MAGIC:
        raise GgufMapError(f"{label}: not a GGUF file (magic {magic!r})")
    (version,) = r.unpack("<I")
    if version not in (2, 3):
        if version & 0xFFFF == 0:
            raise GgufMapError(f"{label}: big-endian GGUF is not supported")
        raise GgufMapError(f"{label}: unsupported GGUF version {version} (need 2 or 3)")
    n_tensors, n_kv = r.unpack("<QQ")
    if n_tensors > 1_000_000 or n_kv > 1_000_000:
        raise GgufMapError(f"{label}: implausible header counts (tensors {n_tensors}, kv {n_kv})")
    fields: dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.string()
        (vtype,) = r.unpack("<I")
        fields[key] = _read_value(r, vtype, key)
    alignment = fields.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
    if not isinstance(alignment, int) or alignment <= 0:
        raise GgufMapError(f"{label}: invalid general.alignment {alignment!r}")
    raw: list[tuple[str, list[int], int, int]] = []
    for _ in range(n_tensors):
        name = r.string()
        (n_dims,) = r.unpack("<I")
        if not 1 <= n_dims <= 8:
            raise GgufMapError(f"{label}: tensor {name!r} has {n_dims} dims")
        dims = [int(d) for d in r.unpack(f"<{n_dims}Q")]
        type_id, rel = r.unpack("<IQ")
        raw.append((name, dims, int(type_id), int(rel)))
    data_offset = _align_up(r.pos, alignment)
    tensors: list[TensorRecord] = []
    for name, dims, type_id, rel in raw:
        tname, nb = _tensor_nbytes(dims, type_id, name, label)
        tensors.append(TensorRecord(name, type_id, tname, dims, nb, data_offset + rel))
    return GgufHeader(label, int(version), int(file_size), int(alignment), int(data_offset),
                      fields, tensors, "stdlib")


def _extract_gguf_py(reader: Any, path: Path) -> GgufHeader:
    """Convert a GGUFReader into plain Python values (no numpy objects survive the return)."""
    from gguf import GGUFValueType

    fields: dict[str, Any] = {}
    version = 3
    for key, fld in reader.fields.items():
        if key == "GGUF.version":
            version = int(_py(fld.contents()))
            continue
        if key.startswith("GGUF."):
            continue
        types = list(fld.types)
        if types and types[0] == GGUFValueType.ARRAY:
            count = len(fld.data)
            sub = int(types[-1])
            limit = _MAX_DECODE_STRINGS if sub == _GV_STRING else _MAX_DECODE_NUMERIC
            if count > limit:
                fields[key] = ArraySummary(sub, count)
                continue
        fields[key] = _py(fld.contents())
    tensors = [TensorRecord(name=str(t.name), type_id=int(t.tensor_type), type=str(t.tensor_type.name),
                            shape=[int(x) for x in t.shape], n_bytes=int(t.n_bytes),
                            offset=int(t.data_offset))
               for t in reader.tensors]
    return GgufHeader(str(path), version, int(path.stat().st_size), int(reader.alignment),
                      int(reader.data_offset), fields, tensors, "gguf")


def _parse_gguf_py(path: Path) -> GgufHeader:
    """Header via gguf-py's GGUFReader (memory-mapped; only header values are converted)."""
    from gguf import GGUFReader  # lazy: needs numpy (+ pyyaml)

    reader = GGUFReader(path, "r")
    try:
        return _extract_gguf_py(reader, path)
    finally:
        # Release the memmap promptly (Windows cannot delete a file that is still mapped).
        del reader
        gc.collect()


def _gguf_py_available() -> bool:
    try:
        import gguf  # noqa: F401
    except Exception as e:  # noqa: BLE001 - numpy/pyyaml missing, broken install
        log.info("gguf-py not importable (%s: %s); using the stdlib GGUF reader", type(e).__name__, e)
        return False
    return True


def read_gguf_header(path: str | os.PathLike[str] | BinaryIO, backend: str = "auto", *,
                     file_size: int | None = None, label: str | None = None) -> GgufHeader:
    """Read a GGUF header. ``path`` may be a file path or a binary stream (stdlib backend only).

    backend: 'auto' (gguf-py if importable, else stdlib), 'gguf' or 'stdlib'. Raises
    FileNotFoundError for a missing file and GgufMapError for malformed/truncated headers.
    """
    if backend not in ("auto", "gguf", "stdlib"):
        raise ValueError(f"backend must be auto|gguf|stdlib, got {backend!r}")
    if hasattr(path, "read"):
        if backend == "gguf":
            raise ValueError("backend 'gguf' needs a file path, not a stream")
        if file_size is None:
            raise ValueError("file_size is required when reading from a stream")
        return _parse_stdlib(path, label or "<stream>", file_size)  # type: ignore[arg-type]
    p = Path(os.fspath(path))  # type: ignore[arg-type]
    if not p.is_file():
        raise FileNotFoundError(f"GGUF file not found: {p}")
    if backend == "auto":
        backend = "gguf" if _gguf_py_available() else "stdlib"
    if backend == "gguf":
        return _parse_gguf_py(p)
    size = p.stat().st_size if file_size is None else file_size
    with open(p, "rb") as f:
        return _parse_stdlib(f, label or str(p), size)


# --------------------------------------------------------------------------------------------
# Classification and KV helpers
# --------------------------------------------------------------------------------------------
def classify_tensor(name: str) -> tuple[str, int | None]:
    """Return ``(kind, layer)`` for a GGUF tensor name (kinds: see KINDS)."""
    m = _RE_LAYER.match(name)
    layer = int(m.group(1)) if m else None
    if _RE_EXPERT.match(name):
        kind = "expert"
    elif _RE_EXPERT_SCALE.match(name):
        kind = "expert_scale"
    elif _RE_ROUTER.match(name):
        kind = "router"
    elif _RE_TOKEN_EMBD.match(name):
        kind = "token_embd"
    elif _RE_OUTPUT.match(name):
        kind = "output"
    elif _RE_ATTN.match(name):
        kind = "attn"
    elif _RE_FFN_DENSE.match(name):
        kind = "ffn_dense"
    elif _RE_NORM.search(name) or _RE_LAYER_SCALE.match(name):
        kind = "norm"
    else:
        kind = "other"
    return kind, layer


def _per_layer_list(v: Any, n_layer: int, key: str, where: str) -> list[int]:
    if isinstance(v, ArraySummary):
        raise GgufMapError(f"{where}: {key} is an oversized array ({v.count} entries)")
    if isinstance(v, list):
        vals = [int(x) for x in v]
        if len(vals) != n_layer:
            raise GgufMapError(f"{where}: {key} has {len(vals)} entries, expected block_count={n_layer}")
        return vals
    return [int(v)] * n_layer


def _sliding_list(pattern: Any, n_layer: int, sliding_window: int | None, where: str) -> list[bool]:
    if pattern is None:
        if sliding_window:
            log.warning("%s: sliding_window=%s but no sliding_window_pattern; treating every layer "
                        "as global (KV upper bound)", where, sliding_window)
        return [False] * n_layer
    if isinstance(pattern, list):
        if len(pattern) != n_layer:
            raise GgufMapError(f"{where}: sliding_window_pattern has {len(pattern)} entries, "
                               f"expected {n_layer}")
        return [bool(x) for x in pattern]
    n = int(pattern)  # llama.cpp set_swa_pattern(n, dense_first=false)
    return [n == 0 or (il % n < n - 1) for il in range(n_layer)]


def kv_bytes_for(head_count_kv: Sequence[int], key_length: int, key_length_swa: int,
                 sliding: Sequence[bool], sliding_window: int | None, n_ctx: int,
                 cache_type: str = "f16", value_length: int | None = None,
                 value_length_swa: int | None = None) -> int:
    """KV-cache bytes for ``n_ctx`` tokens; sliding layers hold at most ``sliding_window`` tokens.

    Per layer and token: n_head_kv * (k_len + v_len) values (K and V both counted: upper bound
    for Gemma 4's K==V global layers). ``cache_type`` is a -ctk/-ctv type (f16, q8_0, ...).
    """
    if cache_type not in KV_CACHE_BYTES_PER_32:
        raise ValueError(f"unknown KV cache type {cache_type!r}; known: {sorted(KV_CACHE_BYTES_PER_32)}")
    if n_ctx < 0:
        raise ValueError("n_ctx must be >= 0")
    per32 = KV_CACHE_BYTES_PER_32[cache_type]
    vl = key_length if value_length is None else value_length
    vls = key_length_swa if value_length_swa is None else value_length_swa
    values = 0
    for hk, sw in zip(head_count_kv, sliding):
        per_tok = hk * ((key_length_swa + vls) if sw else (key_length + vl))
        toks = min(n_ctx, sliding_window) if (sw and sliding_window) else n_ctx
        values += per_tok * toks
    return (values * per32 + 31) // 32


def kv_bytes_per_token_f16(head_count_kv: Sequence[int], key_length: int, key_length_swa: int,
                           sliding: Sequence[bool], value_length: int | None = None,
                           value_length_swa: int | None = None) -> int:
    """sum over layers of n_head_kv * (k_len + v_len) * 2 bytes (spec: ``*2*2`` with v_len == k_len)."""
    vl = key_length if value_length is None else value_length
    vls = key_length_swa if value_length_swa is None else value_length_swa
    return sum(hk * ((key_length_swa + vls) if sw else (key_length + vl)) * 2
               for hk, sw in zip(head_count_kv, sliding))


# --------------------------------------------------------------------------------------------
# GgufMap
# --------------------------------------------------------------------------------------------
class GgufMap(Mapping[str, Any]):
    """A map.json document (TRACE_FORMAT section 8) with helper methods.

    It is a read-only Mapping over the JSON dict, so ``m["summary"]["expert_unit_bytes"]`` and
    ``json.dumps(m.to_dict())`` both work.
    """

    def __init__(self, data: Mapping[str, Any], *, path: str | os.PathLike[str] | None = None) -> None:
        missing = [k for k in _MAP_REQUIRED if k not in data]
        if missing:
            raise GgufMapError(f"not a pigemma map.json (missing keys {missing})")
        self._d: dict[str, Any] = dict(data)
        self.path: Path | None = Path(os.fspath(path)) if path is not None else None
        self._by_name = {t["name"]: t for t in self._d["tensors"]}
        self._by_layer = {int(el["layer"]): el for el in self._d["expert_layers"]}

    # Mapping protocol ---------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        return (f"GgufMap({self.gguf_basename!r}, n_tensors={len(self.tensors)}, "
                f"n_moe_layers={len(self.moe_layers)}, n_expert={self.n_expert})")

    # Properties -----------------------------------------------------------------------------
    @property
    def gguf(self) -> str:
        return str(self._d["gguf"])

    @property
    def gguf_basename(self) -> str:
        return str(self._d.get("gguf_basename") or Path(self.gguf.replace("\\", "/")).name)

    @property
    def file_size(self) -> int:
        return int(self._d["file_size"])

    @property
    def page_size(self) -> int:
        return int(self._d["page_size"])

    @property
    def n_layer(self) -> int:
        return int(self._d["n_layer"])

    @property
    def n_expert(self) -> int:
        return int(self._d["n_expert"])

    @property
    def n_expert_used(self) -> int:
        return int(self._d["n_expert_used"])

    @property
    def tensors(self) -> list[dict[str, Any]]:
        return self._d["tensors"]

    @property
    def expert_layers(self) -> list[dict[str, Any]]:
        return self._d["expert_layers"]

    @property
    def summary(self) -> dict[str, Any]:
        return self._d["summary"]

    @property
    def moe_layers(self) -> list[int]:
        return sorted(self._by_layer)

    @property
    def is_moe(self) -> bool:
        return bool(self._by_layer)

    # Lookups --------------------------------------------------------------------------------
    def tensor(self, name: str) -> dict[str, Any]:
        """The tensors[] entry for ``name`` (KeyError naming the tensor if absent)."""
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"tensor {name!r} not in map of {self.gguf_basename}") from None

    def tensors_of_kind(self, *kinds: str) -> list[dict[str, Any]]:
        return [t for t in self.tensors if t["kind"] in kinds]

    def nonexpert_tensors(self) -> list[dict[str, Any]]:
        return [t for t in self.tensors if t["kind"] != "expert"]

    def expert_layer(self, layer: int) -> dict[str, Any]:
        try:
            return self._by_layer[int(layer)]
        except KeyError:
            raise KeyError(f"layer {layer} has no expert tensors in {self.gguf_basename}") from None

    def expert_parts(self, layer: int) -> list[tuple[str, dict[str, Any]]]:
        """``[(role, {offset, n_bytes, nb_expert}), ...]`` for a MoE layer, roles in file-role order."""
        el = self.expert_layer(layer)
        return [(r, el[r]) for r in _EXPERT_ROLES if r in el]

    def expert_ranges(self, layer: int, expert: int) -> list[tuple[int, int]]:
        """Absolute ``(offset, length)`` of expert ``expert`` in every expert tensor of ``layer``."""
        el = self.expert_layer(layer)
        n_e = int(el.get("n_expert", self.n_expert))
        if not 0 <= int(expert) < n_e:
            raise IndexError(f"expert {expert} out of range 0..{n_e - 1}")
        return [(int(p["offset"]) + int(expert) * int(p["nb_expert"]), int(p["nb_expert"]))
                for _, p in self.expert_parts(layer)]

    def expert_unit_bytes_at(self, layer: int) -> int:
        """Bytes of one expert at ``layer`` (sum over the layer's expert tensors)."""
        return sum(int(p["nb_expert"]) for _, p in self.expert_parts(layer))

    def kv_bytes(self, n_ctx: int, cache_type: str = "f16") -> int:
        """KV-cache bytes at ``n_ctx`` tokens (see kv_bytes_for)."""
        return kv_bytes_for(self._d["head_count_kv"], int(self._d["key_length"]),
                            int(self._d["key_length_swa"]), self._d["sliding_pattern"],
                            self._d.get("sliding_window"), n_ctx, cache_type,
                            self._d.get("value_length"), self._d.get("value_length_swa"))

    # Serialisation --------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dict(self._d)

    def save(self, path: str | os.PathLike[str]) -> Path:
        p = Path(os.fspath(path))
        _atomic_write_text(p, _dump_json(self._d))
        self.path = p
        return p

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GgufMap":
        return cls(data)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "GgufMap":
        p = Path(os.fspath(path))
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise GgufMapError(f"{p}: not a JSON object")
        fv = data.get("format_version", MAP_FORMAT_VERSION)
        if fv != MAP_FORMAT_VERSION:
            raise GgufMapError(f"{p}: map format_version {fv} != {MAP_FORMAT_VERSION}")
        return cls(data, path=p)


def load_map(path: str | os.PathLike[str] | Mapping[str, Any] | GgufMap) -> GgufMap:
    """Load a map.json path, or wrap an already-loaded dict / GgufMap."""
    if isinstance(path, GgufMap):
        return path
    if isinstance(path, Mapping):
        return GgufMap(path)
    return GgufMap.load(path)


# --------------------------------------------------------------------------------------------
# build_map
# --------------------------------------------------------------------------------------------
def _kv(fields: Mapping[str, Any], key: str, where: str, *, required: bool = True,
        default: Any = None) -> Any:
    if key in fields:
        return fields[key]
    if required:
        raise MissingKeyError(key, where)
    return default


def build_map_from_header(hdr: GgufHeader, page_size: int = DEFAULT_PAGE_SIZE, *,
                          gguf_label: str | None = None,
                          sha256_prefix: str | None = None) -> GgufMap:
    """Classify the tensors of a parsed header and compute the map + summary."""
    if page_size <= 0:
        raise ValueError(f"page_size must be > 0, got {page_size}")
    where = hdr.path
    f = hdr.fields
    arch = _kv(f, "general.architecture", where)
    if not isinstance(arch, str) or not arch:
        raise GgufMapError(f"{where}: invalid general.architecture {arch!r}")
    n_layer = int(_kv(f, f"{arch}.block_count", where))
    n_embd = int(_kv(f, f"{arch}.embedding_length", where))
    n_ff: Any = _jsonable(_kv(f, f"{arch}.feed_forward_length", where))
    if isinstance(n_ff, list) and n_ff and all(x == n_ff[0] for x in n_ff):
        n_ff = int(n_ff[0])
    head_count_kv = _per_layer_list(_kv(f, f"{arch}.attention.head_count_kv", where), n_layer,
                                    f"{arch}.attention.head_count_kv", where)
    key_length = int(_kv(f, f"{arch}.attention.key_length", where))
    key_length_swa = int(_kv(f, f"{arch}.attention.key_length_swa", where, required=False,
                             default=key_length))
    value_length = int(_kv(f, f"{arch}.attention.value_length", where, required=False,
                           default=key_length))
    value_length_swa = int(_kv(f, f"{arch}.attention.value_length_swa", where, required=False,
                               default=key_length_swa))
    head_count = _jsonable(_kv(f, f"{arch}.attention.head_count", where, required=False))
    if isinstance(head_count, list):
        head_count = [int(x) for x in head_count]
        if head_count and all(x == head_count[0] for x in head_count):
            head_count = head_count[0]
    sliding_window = _kv(f, f"{arch}.attention.sliding_window", where, required=False)
    sliding_window = int(sliding_window) if sliding_window else None
    sliding = _sliding_list(_kv(f, f"{arch}.attention.sliding_window_pattern", where,
                                required=False), n_layer, sliding_window, where)
    n_expert_kv = int(_kv(f, f"{arch}.expert_count", where, required=False, default=0) or 0)
    n_ff_exp = _kv(f, f"{arch}.expert_feed_forward_length", where, required=False)
    n_ff_exp = int(n_ff_exp) if isinstance(n_ff_exp, (int, float)) else _jsonable(n_ff_exp)
    context_length = _kv(f, f"{arch}.context_length", where, required=False)

    # Tensors -------------------------------------------------------------------------------
    recs = sorted(hdr.tensors, key=lambda t: (t.offset, t.name))
    tensors: list[dict[str, Any]] = []
    prev_end, prev_name = hdr.data_offset, None
    for t in recs:
        if t.offset < hdr.data_offset:
            raise GgufMapError(f"{where}: tensor {t.name!r} offset {t.offset} < data_offset {hdr.data_offset}")
        if t.offset < prev_end and prev_name is not None:
            raise GgufMapError(f"{where}: tensor {t.name!r} overlaps {prev_name!r}")
        end = t.offset + t.n_bytes
        if end > hdr.file_size:
            raise GgufMapError(
                f"{where}: tensor {t.name!r} ends at byte {end} > file size {hdr.file_size} "
                "(truncated or partial download)")
        prev_end, prev_name = end, t.name
        kind, layer = classify_tensor(t.name)
        tensors.append({"name": t.name, "type": t.type, "shape": list(t.shape), "n_bytes": t.n_bytes,
                        "offset": t.offset, "kind": kind, "layer": layer})

    # Expert slabs ---------------------------------------------------------------------------
    per_layer: dict[int, dict[str, Any]] = {}
    tensor_n_expert: set[int] = set()
    for t in tensors:
        if t["kind"] != "expert":
            continue
        m = _RE_EXPERT.match(t["name"])
        assert m is not None
        il, role = int(m.group(1)), m.group(2)
        n_e = int(t["shape"][-1])
        if n_e <= 0 or t["n_bytes"] % n_e != 0:
            raise GgufMapError(f"{where}: expert tensor {t['name']!r} n_bytes {t['n_bytes']} is not "
                               f"divisible by n_expert {n_e}")
        if n_expert_kv and n_e != n_expert_kv:
            raise GgufMapError(f"{where}: expert tensor {t['name']!r} has {n_e} experts but "
                               f"{arch}.expert_count is {n_expert_kv}")
        tensor_n_expert.add(n_e)
        el = per_layer.setdefault(il, {"layer": il})
        if role in el:
            raise GgufMapError(f"{where}: duplicate expert role {role!r} in layer {il}")
        el[role] = {"offset": t["offset"], "n_bytes": t["n_bytes"], "nb_expert": t["n_bytes"] // n_e}
        el["n_expert"] = n_e
    if len(tensor_n_expert) > 1:
        raise GgufMapError(f"{where}: expert tensors disagree on the expert count {sorted(tensor_n_expert)}")
    expert_layers: list[dict[str, Any]] = []
    for il in sorted(per_layer):
        el = per_layer[il]
        roles = [r for r in _EXPERT_ROLES if r in el]
        if "down" not in roles or not ({"gate_up"} <= set(roles) or {"gate", "up"} <= set(roles)):
            log.warning("%s: layer %d has an unusual expert tensor set %s", where, il, roles)
        el["unit_bytes"] = sum(int(el[r]["nb_expert"]) for r in roles)
        ordered = {"layer": il}
        ordered.update({r: el[r] for r in roles})
        ordered["n_expert"] = el["n_expert"]
        ordered["unit_bytes"] = el["unit_bytes"]
        expert_layers.append(ordered)
    if expert_layers:
        n_expert = n_expert_kv or next(iter(tensor_n_expert))
        if not n_expert_kv:
            log.warning("%s: no %s.expert_count key; using %d from the tensor shapes", where, arch, n_expert)
        n_expert_used = int(_kv(f, f"{arch}.expert_used_count", where))
    else:
        if n_expert_kv:
            log.warning("%s: %s.expert_count=%d but no ffn_*_exps tensors; treating as dense",
                        where, arch, n_expert_kv)
        n_expert = 0
        n_expert_used = int(_kv(f, f"{arch}.expert_used_count", where, required=False, default=0) or 0)

    # Summary -------------------------------------------------------------------------------
    bytes_per_kind = {k: 0 for k in KINDS}
    tensors_per_kind = {k: 0 for k in KINDS}
    for t in tensors:
        bytes_per_kind[t["kind"]] += t["n_bytes"]
        tensors_per_kind[t["kind"]] += 1
    tensor_bytes_total = sum(bytes_per_kind.values())
    expert_bytes = bytes_per_kind["expert"]
    nonexpert_bytes = tensor_bytes_total - expert_bytes
    units = [int(el["unit_bytes"]) for el in expert_layers]
    unit_mean = int(round(sum(units) / len(units))) if units else 0
    worst = n_expert_used * sum(units)
    embd = next((t for t in tensors if t["name"] == "token_embd.weight"), None)
    output = next((t for t in tensors if t["kind"] == "output"), None)
    embd_row_bytes = 0
    if embd is not None and len(embd["shape"]) >= 2 and int(embd["shape"][1]) > 0:
        embd_row_bytes = int(embd["n_bytes"]) // int(embd["shape"][1])
    if output is not None:
        head, head_bytes = "output", int(output["n_bytes"])
    elif embd is not None:
        head, head_bytes = "token_embd", int(embd["n_bytes"])
    else:
        head, head_bytes = None, 0
    nonexpert_per_token = sum(bytes_per_kind[k] for k in _PER_TOKEN_KINDS) + head_bytes + embd_row_bytes
    aligned_units: list[int] = []
    for el in expert_layers:
        parts = [el[r] for r in _EXPERT_ROLES if r in el]
        for e in range(int(el["n_expert"])):
            aligned_units.append(sum(
                aligned(int(p["offset"]) + e * int(p["nb_expert"]), int(p["nb_expert"]), page_size)[1]
                for p in parts))
    kv_tok = kv_bytes_per_token_f16(head_count_kv, key_length, key_length_swa, sliding,
                                    value_length, value_length_swa)
    kv_f16_4k = kv_bytes_for(head_count_kv, key_length, key_length_swa, sliding, sliding_window,
                             KV_REFERENCE_CTX, "f16", value_length, value_length_swa)
    kv_q8_4k = kv_bytes_for(head_count_kv, key_length, key_length_swa, sliding, sliding_window,
                            KV_REFERENCE_CTX, "q8_0", value_length, value_length_swa)
    summary = {
        "nonexpert_bytes": nonexpert_bytes,
        "expert_bytes": expert_bytes,
        "expert_unit_bytes": unit_mean,
        "expert_unit_bytes_min": min(units) if units else 0,
        "expert_unit_bytes_max": max(units) if units else 0,
        "expert_unit_bytes_aligned_mean": (int(round(sum(aligned_units) / len(aligned_units)))
                                           if aligned_units else 0),
        "n_moe_layers": len(expert_layers),
        "expert_bytes_per_token_worst": worst,
        "nonexpert_bytes_per_token": nonexpert_per_token,
        "bytes_per_token_worst": nonexpert_per_token + worst,
        "head": head,
        "head_bytes": head_bytes,
        "embd_row_bytes": embd_row_bytes,
        "kv_bytes_per_token_f16": kv_tok,
        "kv_bytes_f16_at_4096": kv_f16_4k,
        "kv_bytes_q8_at_4096": kv_q8_4k,
        "tensor_bytes_total": tensor_bytes_total,
        "file_data_bytes": hdr.file_size - hdr.data_offset,
        "bytes_per_kind": bytes_per_kind,
        "tensors_per_kind": tensors_per_kind,
    }
    label = gguf_label if gguf_label is not None else hdr.path
    data: dict[str, Any] = {
        "format_version": MAP_FORMAT_VERSION,
        "generator": GENERATOR,
        "created_at": _utc_now(),
        "backend": hdr.backend,
        "gguf": label,
        "gguf_basename": Path(str(label).replace("\\", "/")).name,
        "gguf_sha256_prefix": sha256_prefix,
        "gguf_version": hdr.version,
        "file_size": hdr.file_size,
        "data_offset": hdr.data_offset,
        "alignment": hdr.alignment,
        "page_size": int(page_size),
        "arch": arch,
        "n_layer": n_layer,
        "n_expert": n_expert,
        "n_expert_used": n_expert_used,
        "n_embd": n_embd,
        "n_ff": n_ff,
        "n_ff_exp": n_ff_exp,
        "n_head": head_count,
        "head_count_kv": head_count_kv,
        "key_length": key_length,
        "key_length_swa": key_length_swa,
        "value_length": value_length,
        "value_length_swa": value_length_swa,
        "sliding_window": sliding_window,
        "sliding_pattern": sliding,
        "context_length": int(context_length) if isinstance(context_length, int) else None,
        "n_tensors": len(tensors),
        "moe_layers": [el["layer"] for el in expert_layers],
        "tensors": tensors,
        "expert_layers": expert_layers,
        "summary": summary,
    }
    return GgufMap(data)


def build_map(path: str | os.PathLike[str], page_size: int = DEFAULT_PAGE_SIZE, *,
              backend: str = "auto", with_sha: bool = True) -> GgufMap:
    """Read a GGUF header and return its byte-range map (see module docstring).

    ``path`` may also be a registry id of pigemma.paths.MODELS ('26b-q4'). The map records the
    path repo-relative when it is inside the repo, else as given. ``with_sha`` adds the
    SHA-256 of the first 16 MiB (pigemma.paths.sha256_prefix) as ``gguf_sha256_prefix``.
    """
    p = paths.model_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"GGUF file not found: {p}")
    hdr = read_gguf_header(p, backend)
    sha = paths.sha256_prefix(p)["sha256"] if with_sha else None
    gmap = build_map_from_header(hdr, page_size, gguf_label=paths.repo_relative(p), sha256_prefix=sha)
    log.info("map of %s: %d tensors, %d MoE layers, backend %s", p.name, len(gmap.tensors),
             len(gmap.moe_layers), hdr.backend)
    return gmap


def format_summary(gmap: GgufMap | Mapping[str, Any]) -> str:
    """Human-readable summary table of a map (what ``build`` and ``summary`` print)."""
    g = load_map(gmap)
    s = g.summary
    lines = [
        f"GGUF         {g.gguf}",
        f"file_size    {_fmt_int(g.file_size)} B ({_gb(g.file_size)} GB)   data_offset "
        f"{_fmt_int(g['data_offset'])}   alignment {g.get('alignment')}   tensors {len(g.tensors)}",
        f"arch         {g.get('arch')}   layers {g.n_layer}   n_embd {g.get('n_embd')}   n_ff {g.get('n_ff')}   "
        + (f"experts {g.n_expert} (top-{g.n_expert_used}, n_ff_exp {g.get('n_ff_exp')})   "
           f"MoE layers {s['n_moe_layers']}" if s["n_moe_layers"] else "dense (no expert tensors)"),
        "",
        f"{'kind':<14}{'tensors':>8}{'bytes':>18}{'GB':>9}",
    ]
    for k in KINDS:
        n = s["tensors_per_kind"].get(k, 0)
        if n:
            b = s["bytes_per_kind"][k]
            lines.append(f"{k:<14}{n:>8}{b:>18,}{b / 1e9:>9.3f}")
    lines += [
        f"{'non-expert':<14}{'':>8}{s['nonexpert_bytes']:>18,}{s['nonexpert_bytes'] / 1e9:>9.3f}",
        f"{'expert':<14}{'':>8}{s['expert_bytes']:>18,}{s['expert_bytes'] / 1e9:>9.3f}",
        "",
    ]
    if s["n_moe_layers"]:
        lines.append(
            f"expert unit (all expert tensors of one expert in one layer): {s['expert_unit_bytes']:,} B"
            f" (min {s['expert_unit_bytes_min']:,}, max {s['expert_unit_bytes_max']:,}; page-aligned at "
            f"{g.page_size} B: {s['expert_unit_bytes_aligned_mean']:,} B mean)")
    lines += [
        f"per token    non-expert {_gb(s['nonexpert_bytes_per_token'])} GB (head {s['head']} "
        f"{_gb(s['head_bytes'])} GB + one embedding row {s['embd_row_bytes']:,} B)"
        + (f" + experts worst {_gb(s['expert_bytes_per_token_worst'])} GB" if s["n_moe_layers"] else "")
        + f" = {_gb(s['bytes_per_token_worst'])} GB/token",
        f"KV cache     {s['kv_bytes_per_token_f16']:,} B/token f16 (upper bound); at "
        f"{KV_REFERENCE_CTX} ctx: f16 {s['kv_bytes_f16_at_4096'] / 1e6:.1f} MB, "
        f"q8_0 {s['kv_bytes_q8_at_4096'] / 1e6:.1f} MB",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Hot-set planner
# --------------------------------------------------------------------------------------------
@dataclass
class _Part:
    offset: int
    length: int
    kind: str
    layer: int | None
    expert: int | None
    hits: int | float | None


def load_stats(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a stats.json written by ``python -m pigemma.trace stats`` (only per_layer is used)."""
    p = Path(os.fspath(path))
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("per_layer"), list):
        raise HotsetError(f"{p}: not a stats.json (no per_layer list)")
    return data


def _stats_table(gmap: GgufMap, stats: Mapping[str, Any], rank: str
                 ) -> dict[int, tuple[list[int], list[float] | None]]:
    per_layer = stats.get("per_layer")
    if not isinstance(per_layer, list):
        raise HotsetError("stats has no per_layer list")
    moe = set(gmap.moe_layers)
    table: dict[int, tuple[list[int], list[float] | None]] = {}
    for j, ent in enumerate(per_layer):
        try:
            il = int(ent["layer"])
            freq = [int(x) for x in ent["freq"]]
        except (KeyError, TypeError, ValueError) as e:
            raise HotsetError(f"stats per_layer[{j}] lacks layer/freq: {e}") from None
        if il not in moe:
            raise HotsetError(f"stats layer {il} is not a MoE layer of {gmap.gguf_basename} "
                              f"(map MoE layers {gmap.moe_layers[:3]}...{gmap.moe_layers[-3:]})")
        n_e = int(gmap.expert_layer(il).get("n_expert", gmap.n_expert))
        if len(freq) != n_e:
            raise HotsetError(f"stats layer {il} has {len(freq)} freq entries, map has {n_e} experts")
        if any(x < 0 for x in freq):
            raise HotsetError(f"stats layer {il} has negative freq")
        wm_raw = ent.get("weight_mass")
        wm: list[float] | None = None
        if wm_raw is not None:
            wm = [float(x) if x is not None else 0.0 for x in wm_raw]
            if not all(math.isfinite(x) and x >= 0 for x in wm):
                raise HotsetError(f"stats layer {il} has a negative or non-finite weight_mass")
            if len(wm) != n_e:
                raise HotsetError(f"stats layer {il} has {len(wm)} weight_mass entries, map has {n_e}")
        if rank == "weight" and wm is None:
            raise HotsetError(f"--rank weight needs per_layer[].weight_mass (missing at layer {il})")
        if il in table:
            raise HotsetError(f"stats lists layer {il} twice")
        table[il] = (freq, wm)
    missing = sorted(moe - set(table))
    if missing:
        log.warning("stats has no entry for MoE layers %s; their experts rank with freq 0", missing)
    return table


def plan_hotset(gmap: GgufMap | Mapping[str, Any] | str | os.PathLike[str],
                stats: Mapping[str, Any] | None, budget_bytes: int, *,
                page_size: int = DEFAULT_PAGE_SIZE, include_nonexpert: bool = True,
                per_layer_min: int = 0, rank: str = "count", min_hits: int = 0,
                stats_path: str | os.PathLike[str] | None = None,
                map_sha256: str | None = None) -> dict[str, Any]:
    """Plan the resident set: non-expert tensors first, then the hottest experts, within budget.

    Returns the hotset.json dict (TRACE_FORMAT section 9). Budget accounting is on page-aligned,
    merged bytes, so ``total_bytes <= budget_bytes``. Raises HotsetBudgetError when the mandatory
    non-expert set exceeds the budget or nothing fits, HotsetError for bad arguments/stats.
    """
    g = load_map(gmap)
    budget_bytes = int(budget_bytes)
    if budget_bytes < 0:
        raise HotsetError(f"budget_bytes must be >= 0, got {budget_bytes}")
    if page_size <= 0:
        raise HotsetError(f"page_size must be > 0, got {page_size}")
    if rank not in ("count", "weight"):
        raise HotsetError(f"rank must be 'count' or 'weight', got {rank!r}")
    if per_layer_min < 0 or min_hits < 0:
        raise HotsetError("per_layer_min and min_hits must be >= 0")
    if not include_nonexpert and stats is None:
        raise HotsetError("nothing to pin: non-expert tensors excluded and no stats given")
    if stats is not None and not g.is_moe:
        log.warning("%s has no experts; stats are ignored", g.gguf_basename)
    limit = g.file_size
    iset = _IntervalSet()
    parts: list[_Part] = []

    nonexpert_aligned = 0
    if include_nonexpert:
        for t in sorted(g.nonexpert_tensors(), key=lambda t: int(t["offset"])):
            n = int(t["n_bytes"])
            if n <= 0:
                continue
            a, ln = aligned(int(t["offset"]), n, page_size, limit)
            iset.add(a, a + ln)
            parts.append(_Part(int(t["offset"]), n, t["kind"], t["layer"], None, None))
        nonexpert_aligned = iset.total
        if nonexpert_aligned > budget_bytes:
            raise HotsetBudgetError(
                f"budget {budget_bytes:,} B < mandatory non-expert set {nonexpert_aligned:,} B "
                f"(page {page_size}) of {g.gguf_basename}: raise the budget (more free RAM) or pass "
                "--no-nonexpert to pin experts only")

    selected: list[tuple[int, int, int, float | None, int | float]] = []
    chosen: set[tuple[int, int]] = set()
    freq_total = 0
    weight_total = 0.0
    if stats is not None and g.is_moe:
        table = _stats_table(g, stats, rank)
        scores: dict[int, list[tuple[int | float, int, float | None]]] = {}
        for il in g.moe_layers:
            n_e = int(g.expert_layer(il).get("n_expert", g.n_expert))
            freq, wm = table.get(il, ([0] * n_e, None))
            freq_total += sum(freq)
            if wm is not None:
                weight_total += sum(wm)
            row = []
            for e in range(n_e):
                w = wm[e] if wm is not None else None
                sc: int | float = freq[e] if rank == "count" else float(w or 0.0)
                row.append((sc, freq[e], w))
            scores[il] = row

        def try_add(il: int, e: int) -> bool:
            if (il, e) in chosen:
                return True
            raw = g.expert_ranges(il, e)
            rngs = merge_ranges(aligned(o, n, page_size, limit) for o, n in raw)
            delta = sum(iset.uncovered(a, a + ln) for a, ln in rngs)
            if iset.total + delta > budget_bytes:
                return False
            for a, ln in rngs:
                iset.add(a, a + ln)
            sc, fr, w = scores[il][e]
            chosen.add((il, e))
            selected.append((il, e, fr, w, sc))
            for o, n in raw:
                parts.append(_Part(o, n, "expert", il, e, sc))
            return True

        per_layer_order = {il: sorted(range(len(row)), key=lambda e, row=row: (-row[e][0], e))
                           for il, row in scores.items()}
        for r in range(per_layer_min):
            for il in g.moe_layers:
                order = per_layer_order[il]
                if r < len(order):
                    try_add(il, order[r])
        ranking = sorted(((sc, il, e) for il, row in scores.items() for e, (sc, _, _) in enumerate(row)),
                         key=lambda x: (-x[0], x[1], x[2]))
        for sc, il, e in ranking:
            if scores[il][e][1] < min_hits:
                continue
            try_add(il, e)

    if not iset.starts:
        raise HotsetBudgetError(f"budget {budget_bytes:,} B fits nothing (smallest item exceeds it)")

    # Ranges with merged metadata ---------------------------------------------------------------
    groups: list[list[_Part]] = [[] for _ in iset.starts]
    for prt in parts:
        groups[bisect.bisect_right(iset.starts, prt.offset) - 1].append(prt)
    ranges: list[dict[str, Any]] = []
    for (a, b), ps in zip(iset.intervals(), groups):
        kinds = {p.kind for p in ps}
        layers = {p.layer for p in ps}
        exp_parts = [p for p in ps if p.kind == "expert"]
        exp_ids = {p.expert for p in exp_parts}
        hits: int | float | None = None
        if exp_parts:
            hsum = sum(p.hits for p in exp_parts if p.hits is not None)
            hits = int(hsum) if rank == "count" else round(float(hsum), 6)
        ranges.append({
            "offset": a,
            "length": b - a,
            "kind": kinds.pop() if len(kinds) == 1 else ("mixed" if kinds else "other"),
            "layer": layers.pop() if len(layers) == 1 else None,
            "expert": (exp_ids.pop() if (len(exp_ids) == 1 and len(exp_parts) == len(ps)) else None),
            "hits": hits,
            "n_parts": len(ps),
        })
    total = sum(r["length"] for r in ranges)
    assert total == iset.total and total <= budget_bytes
    raw_bytes = sum(p.length for p in parts)
    bytes_by_kind: dict[str, int] = {}
    for p in parts:
        bytes_by_kind[p.kind] = bytes_by_kind.get(p.kind, 0) + p.length
    sel_freq = sum(s[2] for s in selected)
    sel_weight = sum((s[3] or 0.0) for s in selected)
    per_layer_sel: dict[int, int] = {}
    for il, _e, *_ in selected:
        per_layer_sel[il] = per_layer_sel.get(il, 0) + 1

    if map_sha256 is None:
        if g.path is not None and g.path.is_file():
            map_sha256 = _sha256_file(g.path)
        else:
            map_sha256 = hashlib.sha256(
                json.dumps(g.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    stats_label: str | None
    if stats is None:
        stats_label = None
    elif stats_path is not None:
        stats_label = str(stats_path)
    else:
        stats_label = str(stats.get("trace") or "<in-memory>")
    return {
        "format_version": HOTSET_FORMAT_VERSION,
        "generator": GENERATOR,
        "created_at": _utc_now(),
        "gguf": g.gguf,
        "gguf_basename": g.gguf_basename,
        "gguf_size": g.file_size,
        "map": str(g.path) if g.path is not None else None,
        "map_sha256": map_sha256,
        "stats": stats_label,
        "page_size": int(page_size),
        "budget_bytes": budget_bytes,
        "total_bytes": total,
        "raw_bytes": raw_bytes,
        "page_overhead_bytes": total - raw_bytes,
        "nonexpert_included": bool(include_nonexpert),
        "nonexpert_bytes_aligned": nonexpert_aligned,
        "n_experts": len(selected),
        "n_expert_items_total": sum(int(el.get("n_expert", g.n_expert)) for el in g.expert_layers),
        "expert_coverage_fraction": (round(sel_freq / freq_total, 9) if freq_total > 0 else None),
        "expert_weight_coverage_fraction": (round(sel_weight / weight_total, 9)
                                            if weight_total > 0 else None),
        "rank": rank,
        "per_layer_min": int(per_layer_min),
        "min_hits": int(min_hits),
        "n_ranges": len(ranges),
        "bytes_by_kind": bytes_by_kind,
        "per_layer_selected": [{"layer": il, "n": per_layer_sel.get(il, 0)} for il in g.moe_layers],
        "experts_selected": [[il, e, fr] for il, e, fr, _w, _sc in selected],
        "ranges": ranges,
    }


def ranges_txt_path(out_json: str | os.PathLike[str]) -> Path:
    """``hotset.json`` -> ``hotset.ranges.txt`` (same directory)."""
    p = Path(os.fspath(out_json))
    if p.suffix.lower() == ".json":
        return p.with_suffix(".ranges.txt")
    return p.with_name(p.name + ".ranges.txt")


def _token(v: Any) -> str:
    """A value for the ranges.txt header/comments: no whitespace, '-' for None."""
    if v is None:
        return "-"
    return re.sub(r"\s+", "_", str(v)) or "-"


def format_ranges_txt(hotset: Mapping[str, Any]) -> str:
    """Render hotset.ranges.txt (TRACE_FORMAT 9.3): header line, comments, 'offset length # ...'."""
    cov = hotset.get("expert_coverage_fraction")
    stats = hotset.get("stats")
    lines = [
        f"{RANGES_FIRST_LINE_PREFIX}gguf={_token(hotset['gguf_basename'])} size={int(hotset['gguf_size'])} "
        f"page={int(hotset['page_size'])} budget={int(hotset['budget_bytes'])}",
        f"# generator={_token(hotset.get('generator'))} created={_token(hotset.get('created_at'))} "
        f"map_sha256={_token(hotset.get('map_sha256'))}",
        f"# stats={_token(Path(str(stats)).name if stats else None)} rank={_token(hotset.get('rank'))} "
        f"n_experts={int(hotset.get('n_experts', 0))} coverage={_token(cov)} "
        f"nonexpert_included={int(bool(hotset.get('nonexpert_included')))}",
        f"# total_bytes={int(hotset['total_bytes'])} n_ranges={len(hotset['ranges'])} "
        "(sorted by offset, page-aligned, adjacent ranges merged)",
    ]
    for r in hotset["ranges"]:
        lines.append(f"{int(r['offset'])} {int(r['length'])} # kind={_token(r.get('kind'))} "
                     f"layer={_token(r.get('layer'))} expert={_token(r.get('expert'))}")
    return "\n".join(lines) + "\n"


_RANGE_LINE = re.compile(r"^[ \t]*(\d+)[ \t]+(\d+)(?:[ \t]+#(.*))?[ \t]*$")


def read_ranges_txt(source: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse a hotset.ranges.txt (path, or the text itself when it contains a newline).

    Strict, like tools/pinner: line 1 must carry the v1 header; '#' lines are comments; any other
    malformed line raises ValueError naming the line number; ``length > 0`` and
    ``offset + length <= size``. Returns {header:{gguf,size,page,budget,...}, comments:[...],
    ranges:[{offset, length, kind, layer, expert}]}.
    """
    s = os.fspath(source)
    text = s if "\n" in s else Path(s).read_text(encoding="ascii")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or not lines[0].rstrip("\r").startswith(RANGES_FIRST_LINE_PREFIX):
        raise ValueError(f"line 1: missing '{RANGES_FIRST_LINE_PREFIX.strip()}' header")
    header: dict[str, Any] = {}
    for tok in lines[0].rstrip("\r")[len(RANGES_FIRST_LINE_PREFIX):].split():
        k, sep, v = tok.partition("=")
        if not sep:
            raise ValueError(f"line 1: malformed header token {tok!r}")
        header[k] = int(v) if v.isdigit() else v
    for k in ("gguf", "size", "page", "budget"):
        if k not in header:
            raise ValueError(f"line 1: header lacks {k}=")
    size = int(header["size"])
    comments: list[str] = []
    out: list[dict[str, Any]] = []
    for i, line in enumerate(lines[1:], start=2):
        ln = line.rstrip("\r")
        st = ln.strip()
        if not st:
            continue
        if st.startswith("#"):
            comments.append(st)
            continue
        m = _RANGE_LINE.match(ln)
        if not m:
            raise ValueError(f"line {i}: malformed range line {ln!r}")
        off, length = int(m.group(1)), int(m.group(2))
        if length <= 0:
            raise ValueError(f"line {i}: length must be > 0")
        if off + length > size:
            raise ValueError(f"line {i}: range ends at {off + length} > size {size}")
        meta: dict[str, Any] = {"offset": off, "length": length, "kind": None, "layer": None, "expert": None}
        if m.group(3):
            for tok in m.group(3).split():
                k, sep, v = tok.partition("=")
                if sep and k in ("kind", "layer", "expert"):
                    meta[k] = None if v == "-" else (int(v) if k != "kind" and v.lstrip("-").isdigit() else v)
        out.append(meta)
    return {"header": header, "comments": comments, "ranges": out}


def write_hotset(hotset: Mapping[str, Any], out_json: str | os.PathLike[str],
                 ranges_txt: str | os.PathLike[str] | None = None) -> tuple[Path, Path]:
    """Write hotset.json and its ranges.txt; returns both paths."""
    pj = Path(os.fspath(out_json))
    pr = Path(os.fspath(ranges_txt)) if ranges_txt is not None else ranges_txt_path(pj)
    _atomic_write_text(pj, _dump_json(dict(hotset)))
    _atomic_write_text(pr, format_ranges_txt(hotset))
    return pj, pr


# --------------------------------------------------------------------------------------------
# Catalog refresh (PC only)
# --------------------------------------------------------------------------------------------
def _has_dict(o: Any) -> bool:
    if isinstance(o, dict):
        return True
    if isinstance(o, list):
        return any(_has_dict(x) for x in o)
    return False


def format_catalog_json(obj: Any, _ind: int = 0) -> str:
    """JSON in catalog.json's hand-written layout: 2-space indent, short dict-free lists inline."""
    sp = "  " * _ind
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{sp}  {json.dumps(k, ensure_ascii=False)}: {format_catalog_json(v, _ind + 1)}"
                 for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + sp + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if not _has_dict(obj):
            inline = json.dumps(obj, ensure_ascii=False, separators=(", ", ": "), allow_nan=False)
            if len(inline) <= CATALOG_INLINE_MAX:
                return inline
        items = [f"{sp}  {format_catalog_json(v, _ind + 1)}" for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + sp + "]"
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


def _placeholder_file(name: Any) -> bool:
    return not name or not isinstance(name, str) or any(c in name for c in "<>*?")


def _needs_refresh(m: Mapping[str, Any]) -> list[str]:
    reasons = []
    if m.get("bytes_exact") is False:
        reasons.append("bytes_exact=false")
    if _placeholder_file(m.get("file")):
        reasons.append("placeholder file name")
    if not isinstance(m.get("bytes"), int) or isinstance(m.get("bytes"), bool) or not m.get("bytes"):
        reasons.append("bytes missing")
    if not m.get("sha256"):
        reasons.append("sha256 missing")
    return reasons


def _read_token_file(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.strip().removeprefix("export ").strip() in ("HUGGING_FACE_TOKEN", "HF_TOKEN"):
            v = v.strip().strip('"').strip("'")
            if v:
                return v
    return None


def hf_token(token_file: str | os.PathLike[str] | None = None) -> str | None:
    """HF token from env (HF_TOKEN, HUGGING_FACE_TOKEN) or a KEY=VALUE file; None = anonymous."""
    for var in ("HF_TOKEN", "HUGGING_FACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    cands: list[str] = []
    if token_file:
        cands.append(os.fspath(token_file))
    env_file = os.environ.get("PIGEMMA_HF_TOKEN_FILE", "").strip()
    if env_file:
        cands.append(env_file)
    cands.extend(DEFAULT_HF_TOKEN_FILES)
    for c in cands:
        p = Path(c).expanduser()
        if p.is_file():
            tok = _read_token_file(p)
            if tok:
                log.info("using the HF token from %s (value not shown)", p)
                return tok
    return None


def _default_api(token: str | None) -> Any:
    from huggingface_hub import HfApi  # lazy: PC only
    return HfApi(token=token)


def _default_opener(repo: str, file: str, revision: str | None, token: str | None) -> Any:
    from huggingface_hub import HfFileSystem  # lazy: PC only
    fs = HfFileSystem(token=token)
    spec = f"{repo}@{revision}/{file}" if revision else f"{repo}/{file}"
    return fs.open(spec, "rb", block_size=8 << 20)


def _sibling_meta(sib: Any) -> tuple[str, int | None, str | None]:
    name = getattr(sib, "rfilename", None) or (sib.get("rfilename") if isinstance(sib, dict) else None)
    size = getattr(sib, "size", None) if not isinstance(sib, dict) else sib.get("size")
    lfs = getattr(sib, "lfs", None) if not isinstance(sib, dict) else sib.get("lfs")
    sha = None
    if lfs is not None:
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    return str(name), (int(size) if size is not None else None), sha


def _set(entry: dict[str, Any], key: str, value: Any, changes: dict[str, Any],
         source: str | None = None) -> None:
    """Set entry[key] (recording [old, new] in changes) and, when it changed or is unsourced, its source."""
    changed = key not in entry or entry[key] != value or type(entry[key]) is not type(value)
    if changed:
        changes[key] = [entry.get(key), value]
        entry[key] = value
    if source is not None:
        srcs = entry.setdefault("sources", {})
        if (changed or key not in srcs) and srcs.get(key) != source:
            changes[f"sources.{key}"] = [srcs.get(key), source]
            srcs[key] = source


def _split_fields(gmap: GgufMap) -> dict[str, int]:
    s = gmap.summary
    return {
        "data_offset": int(gmap["data_offset"]),
        "n_tensors": len(gmap.tensors),
        "nonexpert_bytes": int(s["nonexpert_bytes"]),
        "expert_bytes": int(s["expert_bytes"]),
        "expert_unit_bytes": int(s["expert_unit_bytes"]),
        "expert_unit_bytes_min": int(s["expert_unit_bytes_min"]),
        "expert_unit_bytes_max": int(s["expert_unit_bytes_max"]),
        "expert_bytes_per_token_worst": int(s["expert_bytes_per_token_worst"]),
        # catalog convention (tests/test_catalog.py): per-token non-expert bytes == nonexpert_bytes
        "nonexpert_bytes_per_token": int(s["nonexpert_bytes"]),
        "n_layer": int(gmap.n_layer),
        "n_moe_layers": int(s["n_moe_layers"]),
        "n_expert": int(gmap.n_expert),
        "n_expert_used": int(gmap.n_expert_used) if gmap.n_expert else 0,
    }


def catalog_refresh(catalog_path: str | os.PathLike[str] = paths.CATALOG_PATH, *,
                    ids: Sequence[str] | None = None, refresh_all: bool = False, split: bool = False,
                    dry_run: bool = False, out_path: str | os.PathLike[str] | None = None,
                    token: str | None = None, api: Any = None,
                    opener: Callable[[str, str, str | None, str | None], Any] | None = None,
                    today: str | None = None) -> dict[str, Any]:
    """Fill exact bytes / LFS sha256 (and with ``split`` the header-derived byte split) from HF.

    Selected entries: ``ids`` if given, else every entry when ``refresh_all``, else the entries
    that need it (bytes_exact false, placeholder file name, bytes or sha256 missing). ``split`` is
    also implied for an entry flagged ``estimated: true``. ``api`` / ``opener`` are injectable for
    tests. Returns a report {selected, changed:{id:{field:[old,new]}}, errors:{id:msg}, written}.
    """
    cpath = Path(os.fspath(catalog_path))
    raw = cpath.read_text(encoding="utf-8")
    cat = json.loads(raw)
    models = cat.get("models")
    if not isinstance(models, list):
        raise GgufMapError(f"{cpath}: no models[] list")
    by_id = {m.get("id"): m for m in models}
    if ids:
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ValueError(f"unknown catalog ids {unknown}; known: {sorted(by_id)}")
        sel = [by_id[i] for i in ids]
    elif refresh_all:
        sel = list(models)
    else:
        sel = [m for m in models if _needs_refresh(m) or (split and m.get("estimated"))]
    today = today or _dt.date.today().isoformat()
    report: dict[str, Any] = {"selected": [m.get("id") for m in sel], "changed": {}, "errors": {},
                              "written": None, "dry_run": dry_run}
    if not sel:
        log.info("catalog-refresh: nothing to refresh (all entries exact; use --all or --ids)")
        return report
    if api is None:
        api = _default_api(token)
    opener = opener or _default_opener
    infos: dict[str, Any] = {}
    for m in sel:
        mid = str(m.get("id"))
        changes: dict[str, Any] = {}
        try:
            repo = str(m["repo"])
            if repo not in infos:
                infos[repo] = api.model_info(repo, files_metadata=True)
            info = infos[repo]
            sibs = [_sibling_meta(s) for s in (getattr(info, "siblings", None) or [])]
            fname = m.get("file")
            if _placeholder_file(fname):
                pattern = m.get("file_pattern") or (fname if isinstance(fname, str) else None)
                if not pattern or "<" in pattern:
                    raise GgufMapError("placeholder file name and no usable file_pattern")
                hits = [s for s in sibs if fnmatch.fnmatchcase(s[0], pattern)
                        or fnmatch.fnmatchcase(Path(s[0]).name, pattern)]
                if len(hits) != 1:
                    raise GgufMapError(f"file_pattern {pattern!r} matches {len(hits)} files in {repo}: "
                                       f"{[h[0] for h in hits][:8]}")
                sib = hits[0]
            else:
                found = [s for s in sibs if s[0] == fname]
                if not found:
                    raise GgufMapError(f"{fname!r} not found in {repo} (main)")
                sib = found[0]
            name, size, sha = sib
            if size is None:
                raise GgufMapError(f"{repo}/{name}: listing has no size (files_metadata missing?)")
            src_listing = f"measured: HF API model_info(files_metadata=True) size ({today}); pigemma.gguf_map catalog-refresh"
            _set(m, "file", name, changes)
            _set(m, "bytes", size, changes, src_listing)
            _set(m, "bytes_exact", True, changes)
            if sha:
                _set(m, "sha256", sha, changes,
                     f"measured: HF LFS sha256 from HfApi.model_info(files_metadata=True) ({today}); "
                     "pigemma.gguf_map catalog-refresh")
            rev = getattr(info, "sha", None)
            if rev:
                _set(m, "revision", rev, changes)
            local = m.get("local_path")
            if local:
                lp = paths.REPO_ROOT / str(local)
                if lp.is_file() and lp.stat().st_size != size:
                    log.warning("%s: local file %s has %d bytes, HF lists %d", mid, lp, lp.stat().st_size, size)
            if split or m.get("estimated"):
                lp = (paths.REPO_ROOT / str(local)) if local else None
                if lp is not None and lp.is_file() and lp.stat().st_size == size:
                    hdr = read_gguf_header(lp, "stdlib")
                    how = f"measured: GGUF header of the local file parsed by pigemma.gguf_map ({today})"
                else:
                    with contextlib.closing(opener(repo, name, rev, token)) as fobj:
                        hdr = read_gguf_header(fobj, "stdlib", file_size=size, label=f"hf://{repo}/{name}")
                    how = (f"measured: GGUF header parsed by pigemma.gguf_map from an HTTP range read of "
                           f"the HF file ({today})")
                gm = build_map_from_header(hdr, DEFAULT_PAGE_SIZE, gguf_label=f"hf://{repo}/{name}")
                for k, v in _split_fields(gm).items():
                    src = how
                    if k == "nonexpert_bytes_per_token":
                        src = ("measured: equals nonexpert_bytes (decode reads every non-expert tensor once "
                               "per token; the tied token_embd is also the output head)")
                    _set(m, k, v, changes, src)
                _set(m, "estimated", False, changes)
        except Exception as e:  # noqa: BLE001 - per-entry isolation; reported and exit 1
            report["errors"][mid] = f"{type(e).__name__}: {e}"
            log.error("catalog-refresh %s: %s", mid, report["errors"][mid])
            continue
        if changes:
            report["changed"][mid] = changes
    if report["changed"] and not dry_run:
        cat["updated"] = today
        target = Path(os.fspath(out_path)) if out_path is not None else cpath
        _atomic_write_text(target, format_catalog_json(cat) + "\n")
        report["written"] = str(target)
    return report


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _parse_bytes(s: str) -> int:
    """Byte count from '5200000000', '5.2e9', '5_200_000_000' or with a GB/GiB/MB/MiB suffix."""
    t = s.strip().replace("_", "").replace(",", "")
    mult = 1
    for suf, mu in (("gib", 1 << 30), ("mib", 1 << 20), ("kib", 1 << 10), ("gb", 10**9), ("mb", 10**6),
                    ("kb", 10**3), ("b", 1)):
        if t.lower().endswith(suf):
            t, mult = t[: -len(suf)], mu
            break
    try:
        v = float(t) if any(c in t for c in ".eE") else int(t)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a byte count: {s!r}") from None
    if v < 0 or (isinstance(v, float) and not math.isfinite(v)):
        raise argparse.ArgumentTypeError(f"byte count must be finite and >= 0: {s!r}")
    return int(v * mult)


def _positive_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {s!r}") from None
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0: {s!r}")
    return v


def _nonneg_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {s!r}") from None
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0: {s!r}")
    return v


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m pigemma.gguf_map",
        description="GGUF byte-range map, per-token traffic accounting, hot-set planner, catalog refresh.",
        epilog="Exit codes: 0 ok, 1 error, 2 usage, 3 missing input/dependency, 4 refused (budget).",
    )
    ap.add_argument("--log-level", default="INFO", help="logging level on stderr (default INFO)")
    # --log-level is also accepted after the subcommand (SUPPRESS keeps the top-level default).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=argparse.SUPPRESS, help="logging level on stderr (default INFO)")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")

    b = sub.add_parser("build", parents=[common],
                       help="read a GGUF header and write map.json (prints the summary table)")
    b.add_argument("gguf", help="GGUF path or registry id (" + ", ".join(paths.MODELS) + ")")
    b.add_argument("--out", required=True, help="map.json to write")
    b.add_argument("--page-size", type=_positive_int, default=DEFAULT_PAGE_SIZE,
                   help=f"page size assumed for alignment figures (default {DEFAULT_PAGE_SIZE}, the Pi kernel)")
    b.add_argument("--backend", choices=("auto", "gguf", "stdlib"), default="auto",
                   help="header reader: gguf-py GGUFReader, the stdlib parser, or auto (gguf-py if importable)")
    b.add_argument("--no-sha", action="store_true", help="skip the SHA-256 of the first 16 MiB")
    b.add_argument("--reuse", action="store_true",
                   help="if --out exists and matches the GGUF (basename, size, page size), keep it")
    b.add_argument("--json", action="store_true", help="print the summary block as JSON instead of the table")

    s = sub.add_parser("summary", parents=[common], help="print the summary table of an existing map.json")
    s.add_argument("--map", required=True, help="map.json")
    s.add_argument("--json", action="store_true", help="print the summary block as JSON")

    h = sub.add_parser("hotset", parents=[common], help="plan the pinned set: non-expert tensors + hottest experts within a budget")
    h.add_argument("--map", required=True, help="map.json from 'build'")
    h.add_argument("--stats", default=None,
                   help="stats.json from 'python -m pigemma.trace stats' (omit: non-expert tensors only)")
    h.add_argument("--budget-bytes", type=_parse_bytes, required=True,
                   help="byte budget for the page-aligned set (e.g. 5200000000, 5.2e9, 5.2GB)")
    h.add_argument("--page-size", type=_positive_int, default=DEFAULT_PAGE_SIZE,
                   help=f"page size for alignment (default {DEFAULT_PAGE_SIZE}; pass $(getconf PAGESIZE))")
    h.add_argument("--no-nonexpert", action="store_true", help="do not include the non-expert tensors")
    h.add_argument("--per-layer-min", type=_nonneg_int, default=0,
                   help="first pin this many hottest experts of every layer (default 0)")
    h.add_argument("--rank", choices=("count", "weight"), default="count",
                   help="rank experts by routing count (freq) or routing weight mass (default count)")
    h.add_argument("--min-hits", type=_nonneg_int, default=0,
                   help="skip experts with freq < N in the global phase (default 0: all are candidates)")
    h.add_argument("--out", required=True, help="hotset.json to write (ranges file next to it)")
    h.add_argument("--ranges-out", default=None, help="ranges file (default: <out stem>.ranges.txt)")

    c = sub.add_parser("catalog-refresh", parents=[common],
                       help="fill exact bytes/sha256 (and --split byte split) of catalog entries from HF")
    c.add_argument("--catalog", default=str(paths.CATALOG_PATH), help="catalog.json (default pigemma/data/catalog.json)")
    c.add_argument("--ids", default=None, help="comma-separated catalog ids to refresh")
    c.add_argument("--all", action="store_true", help="refresh every entry (verification run)")
    c.add_argument("--split", action="store_true",
                   help="also refresh the non-expert/expert split from the GGUF header (local file or HTTP range read)")
    c.add_argument("--dry-run", action="store_true", help="report changes, do not write")
    c.add_argument("--out", default=None, help="write here instead of rewriting --catalog in place")
    c.add_argument("--token-file", default=None,
                   help="KEY=VALUE file holding HUGGING_FACE_TOKEN or HF_TOKEN (env HF_TOKEN wins)")
    return ap


def _cmd_build(args: argparse.Namespace) -> int:
    p = paths.model_path(args.gguf)
    if not p.is_file():
        sys.stderr.write(f"pigemma.gguf_map: GGUF not found: {p}\n")
        return paths.EXIT_MISSING
    out = Path(args.out)
    gmap: GgufMap | None = None
    if args.reuse and out.is_file():
        try:
            old = GgufMap.load(out)
            if (old.gguf_basename == p.name and old.file_size == p.stat().st_size
                    and old.page_size == args.page_size):
                log.info("reusing %s (same basename, size and page size)", out)
                gmap = old
            else:
                log.info("%s does not match %s; rebuilding", out, p.name)
        except (OSError, ValueError) as e:
            log.info("cannot reuse %s (%s); rebuilding", out, e)
    if gmap is None:
        gmap = build_map(p, args.page_size, backend=args.backend, with_sha=not args.no_sha)
        gmap.save(out)
        log.info("wrote %s", out)
    if args.json:
        print(json.dumps(gmap.summary, indent=1))
    else:
        print(format_summary(gmap))
        print(f"map          {out}")
    return paths.EXIT_OK


def _cmd_hotset(args: argparse.Namespace) -> int:
    mp = Path(args.map)
    if not mp.is_file():
        sys.stderr.write(f"pigemma.gguf_map: map not found: {mp} (build it with 'python -m "
                         "pigemma.gguf_map build <gguf> --out <map.json>')\n")
        return paths.EXIT_MISSING
    stats = None
    if args.stats:
        sp = Path(args.stats)
        if not sp.is_file():
            sys.stderr.write(f"pigemma.gguf_map: stats not found: {sp}\n")
            return paths.EXIT_MISSING
        stats = load_stats(sp)
    gmap = GgufMap.load(mp)
    try:
        hs = plan_hotset(gmap, stats, args.budget_bytes, page_size=args.page_size,
                         include_nonexpert=not args.no_nonexpert, per_layer_min=args.per_layer_min,
                         rank=args.rank, min_hits=args.min_hits, stats_path=args.stats)
    except HotsetBudgetError as e:
        sys.stderr.write(f"pigemma.gguf_map: REFUSED: {e}\n")
        return paths.EXIT_REFUSED
    pj, pr = write_hotset(hs, args.out, args.ranges_out)
    cov = hs["expert_coverage_fraction"]
    print(f"hotset       {pj}  ranges {pr}")
    print(f"budget       {hs['budget_bytes']:,} B   total {hs['total_bytes']:,} B ({_gb(hs['total_bytes'])} GB)"
          f"   page {hs['page_size']}   ranges {hs['n_ranges']}   page overhead {hs['page_overhead_bytes']:,} B")
    print(f"non-expert   {'included' if hs['nonexpert_included'] else 'excluded'}"
          f" ({hs['nonexpert_bytes_aligned']:,} B aligned)")
    print(f"experts      {hs['n_experts']} of {hs['n_expert_items_total']} expert-layer items, rank {hs['rank']}"
          f", expert_coverage_fraction {'-' if cov is None else f'{cov:.4f}'}")
    return paths.EXIT_OK


def _cmd_catalog_refresh(args: argparse.Namespace) -> int:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.stderr.write("pigemma.gguf_map: catalog-refresh needs huggingface_hub (PC only; "
                         "pip install huggingface_hub)\n")
        return paths.EXIT_MISSING
    ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
    rep = catalog_refresh(args.catalog, ids=ids, refresh_all=args.all, split=args.split,
                          dry_run=args.dry_run, out_path=args.out, token=hf_token(args.token_file))
    print(f"selected     {', '.join(map(str, rep['selected'])) or '-'}")
    for mid, ch in rep["changed"].items():
        for k, (old, new) in ch.items():
            print(f"changed      {mid}.{k}: {old!r} -> {new!r}")
    for mid, err in rep["errors"].items():
        print(f"error        {mid}: {err}")
    if not rep["changed"]:
        print("result       no changes")
    elif rep["written"]:
        print(f"written      {rep['written']}")
    else:
        print("result       dry run, nothing written")
    return paths.EXIT_ERROR if rep["errors"] else paths.EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    ap = _build_parser()
    args = ap.parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if args.cmd is None:
        ap.print_help()
        return paths.EXIT_USAGE
    try:
        if args.cmd == "build":
            return _cmd_build(args)
        if args.cmd == "summary":
            mp = Path(args.map)
            if not mp.is_file():
                sys.stderr.write(f"pigemma.gguf_map: map not found: {mp}\n")
                return paths.EXIT_MISSING
            g = GgufMap.load(mp)
            print(json.dumps(g.summary, indent=1) if args.json else format_summary(g))
            return paths.EXIT_OK
        if args.cmd == "hotset":
            return _cmd_hotset(args)
        if args.cmd == "catalog-refresh":
            return _cmd_catalog_refresh(args)
    except FileNotFoundError as e:
        sys.stderr.write(f"pigemma.gguf_map: {e}\n")
        return paths.EXIT_MISSING
    except (GgufMapError, HotsetError) as e:
        sys.stderr.write(f"pigemma.gguf_map: error: {e}\n")
        return paths.EXIT_ERROR
    except ValueError as e:
        sys.stderr.write(f"pigemma.gguf_map: {e}\n")
        return paths.EXIT_USAGE
    except Exception as e:  # noqa: BLE001 - CLI boundary
        sys.stderr.write(f"pigemma.gguf_map: error: {type(e).__name__}: {e}\n")
        return paths.EXIT_ERROR
    ap.print_help()
    return paths.EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
