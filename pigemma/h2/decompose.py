"""pigemma.h2.decompose - H2 step 1: base + residual-group decomposition of dense FFN matrices.

Hypothesis H2 (docs/HYPOTHESES.md, experiment E7): for the dense Gemma 4 31B keep a cheap base
approximation ``W_base`` of every FFN matrix in RAM and stream only the residual column groups
``R_g`` that change the current output, ``W = W_base + sum_g R_g``. A group can be skipped
without reading it when the norm bound ``||R_g x_g|| <= ||R_g||_2 * ||x_g||`` says it is small.
This module builds those decompositions from the real Q4_0 weights; ``pigemma.h2.oracle``
measures them against captured activations.

What it does (manifest files[] 'pigemma/h2/decompose.py'; npz layout: docs/TRACE_FORMAT.md 13.3)

* ``dequant_q4_0(buf, n_rows, n_cols)``: ggml Q4_0 -> float32 ``[n_rows, n_cols]``; a block of
  32 weights is an f16 scale ``d`` + 16 bytes, low nibbles are elements 0..15, high nibbles
  16..31, value ``d*(q-8)``; rows are ``ne[1]``, each row ``ne[0]`` contiguous. Bit-exact with
  ``gguf.quants.dequantize`` (unit-tested).
* ``load_matrix(map_or_gguf, tensor_name)``: reads only that tensor's byte range through
  ``np.memmap`` and returns ``W`` float32 ``[n_out, n_in]`` with ``y = W @ x`` (GGUF stores
  ``ne = [n_in, n_out]``). Non-Q4_0 tensors are dequantised with gguf-py.
* ``group_quant(W, bits, block=32)``: symmetric absmax block quantisation along the input
  dimension with an f16 scale per block, ggml style: the element with the largest magnitude
  (sign kept) maps to the most negative code ``-2^(bits-1)``, codes ``-2^(bits-1) .. 2^(bits-1)-1``.
  For ``bits=4`` this is exactly ggml's Q4_0 rule (bit-identical to gguf-py's quantiser), so its
  error equals the Q4_0 reference. Bytes per element ``bits/8 + 2/block`` (q2 0.3125, q3 0.4375,
  q4 0.5625).
* ``lowrank(W, r)``: randomized SVD (``torch.svd_lowrank(q=r+32, niter=3)``; numpy randomized
  range finder when torch is missing); returns ``U`` f16 ``[n_out, r]``, ``S``, ``Vt`` f16
  ``[r, n_in]`` and ``bytes = 2*r*(n_in+n_out)``.
* ``decompose(W, base, group_cols=128, res_bits=4)`` -> :class:`Decomp`: the base, then the
  residual split into input-column groups, each quantised at ``res_bits`` (block 32 along the
  columns of the group; 16 = unquantised), with its spectral norm (largest singular value,
  ``np.linalg.svd(compute_uv=False)`` in float64) and Frobenius norm, and byte accounting.

CLI::

    python -m pigemma.h2.decompose --map traces/h2/map31.json --layers 5,20,35,50 \\
        --mats gate,up,down --bases q2,lowrank256 --group 128 --res-bits 4 --out traces/h2/decomp/
        [--gguf PATH|ID] [--block 32] [--lowrank-backend auto|torch|numpy] [--seed 0]
        [--niter 3] [--threads N] [--force] [--dry-run] [--json OUT|-] [--min-free-gb 2]
        [--lock] [--lock-min-free-gb 4] [--log-level INFO]

writes ``<out>/L<il>.<mat>.<base>.npz`` per (layer, matrix, base) plus ``<out>/index.json``,
logs progress and an ETA after the first matrix to stderr, and prints the bytes accounting vs
Q4_0 (65,028,096 B per 31B FFN matrix) to stdout. Existing npz files with the same parameters
are reused unless ``--force``.

Exit codes: 0 ok; 1 runtime error (bad GGUF bytes, size mismatch, numerical failure);
2 usage (bad base name, layer out of range, bad list); 3 missing input (map, GGUF, tensor);
4 refused (not enough free disk for the outputs, bench lock busy); 130 interrupted.

Design decisions (the manifest left these open; repeated in the agent report)

* Orientation: ``W[n_out, n_in]``; groups are contiguous input-column ranges
  ``[i*g, min((i+1)*g, n_in))`` (the last one may be shorter); all block quantisers run along
  the input dimension, like ggml.
* f16 rounding of the base: the npz stores ``W_base_hat`` as f16, so ``decompose`` rounds the
  base to f16 *before* forming the residual (``R = W - f16(base)``); the in-memory
  ``Decomp.W_base_hat`` (float32) and the stored one are therefore identical, and base +
  residual reconstructs ``W`` with no hidden f16 error in the base. Byte accounting is of the
  compact form (codes + scales, or the two low-rank factors), not of the dense f16 matrix.
* Residual storage: for ``res_bits < 16`` each group holds the dequantised quantised residual
  rounded to f16 (``Group.R_hat`` dtype float16: exactly what the npz stores; the f16 rounding
  is < 1/500 of one quantisation step). ``res_bits == 16`` means "unquantised": the residual is
  kept exact in float32 (in memory AND in the npz; this one mode deviates from TRACE_FORMAT
  13.3's f16 so that ``W_base_hat + sum R_g == W`` holds to float32 rounding) and is accounted
  at 2 B/element (the cost of streaming it as f16).
* ``norm2``/``normF`` are computed in float64 on exactly the stored residual values. ``norm2``
  is the LAPACK largest singular value multiplied by ``1 + NORM2_RTOL`` (1e-6) so that the
  bound stays sound against float32 rounding in consumers; tightness figures move by 1e-6 at
  most.
* Partial blocks (a group or a matrix row whose width is not a multiple of ``block``) are
  zero-padded for quantisation and accounted as full blocks (block formats store whole blocks).
* Base names: ``q<b>`` (b in 2..8), ``lowrank<r>``, ``q<b>+lowrank<r>`` and ``none`` (zero
  base, residual = W; a control). ``q2+lowrank128`` quantises first and then fits the rank-128
  factors to the quantisation error (LQER-style), so its base error is not worse than q2's
  (the fit is a projection of that error; only f16 rounding of the result can add to it).
* With the default ``res_bits=4`` the reconstruction (base + all residual groups) is NOT
  lossless: the q2 residual of a Q4_0 matrix is an integer multiple of the Q4_0 scale, which
  a fresh 4-bit absmax grid cannot represent exactly. ``recon_rel_err`` (Frobenius, stored in
  the npz/index/table) is that floor; the oracle cannot reach an eps below it. Use
  ``--res-bits 8`` (much lower floor) or 16 (exact) when that matters.
* Lowrank seeds: the randomized SVD is seeded (``--seed``, default 0) inside
  ``torch.random.fork_rng`` so runs are reproducible and the global torch RNG is untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import gc
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pigemma import __version__, paths
from pigemma import gguf_map as gm

log = logging.getLogger("pigemma.h2.decompose")

__all__ = [
    "QK4_0", "Q4_0_BLOCK_BYTES", "DEFAULT_BLOCK", "DEFAULT_GROUP_COLS", "DEFAULT_RES_BITS",
    "RES_BITS_CHOICES", "BASE_EXAMPLES", "FFN_MATS", "FFN_INPUT_KIND", "NORM2_RTOL",
    "LOWRANK_OVERSAMPLE", "LOWRANK_NITER", "NPZ_FORMAT_VERSION", "GENERATOR",
    "BaseSpec", "Group", "Decomp", "TensorInfo",
    "parse_base", "bytes_per_elem", "quant_bytes", "residual_bytes", "q4_0_nbytes",
    "dequant_q4_0", "dequant_tensor", "tensor_info", "load_matrix", "resolve_gguf", "ffn_names",
    "group_quant", "lowrank", "decompose", "group_slices", "group_input_norms",
    "group_contributions", "decomp_path", "save_npz", "load_npz", "npz_size_estimate", "main",
]

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
QK4_0 = 32                    # weights per Q4_0 block
Q4_0_BLOCK_BYTES = 18         # f16 scale + 16 bytes of nibbles
DEFAULT_BLOCK = 32
DEFAULT_GROUP_COLS = 128
DEFAULT_RES_BITS = 4
RES_BITS_CHOICES: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 16)
BASE_EXAMPLES: tuple[str, ...] = ("q2", "q3", "lowrank256", "lowrank512", "q2+lowrank128")
FFN_MATS: tuple[str, ...] = ("gate", "up", "down")
# Which captured activation (acts dump, TRACE_FORMAT 7) is the input x of each matrix.
FFN_INPUT_KIND: dict[str, str] = {"gate": "ffn_norm", "up": "ffn_norm", "down": "ffn_geglu"}
NORM2_RTOL = 1e-6             # norm2 is rounded up by this relative amount (see module doc)
LOWRANK_OVERSAMPLE = 32
LOWRANK_NITER = 3
NPZ_FORMAT_VERSION = 1
GENERATOR = f"pigemma.h2.decompose/{__version__}"
_CHUNK_ELEMS = 1 << 22        # ~16 MB of float32 per row chunk in the block quantisers

_RE_Q = re.compile(r"q([2-8])")
_RE_LR = re.compile(r"lowrank([1-9][0-9]*)")
_RE_QLR = re.compile(r"q([2-8])\+lowrank([1-9][0-9]*)")


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _fmt_s(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 5400:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.2f} h"


def _jsonable(v: Any) -> Any:
    """Plain JSON value: numpy scalars/arrays to Python, NaN/inf to None, Paths to str."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, Path):
        return v.as_posix()
    return v


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# Base names and byte accounting
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BaseSpec:
    """A parsed base name: ``q<bits>``, ``lowrank<rank>``, ``q<bits>+lowrank<rank>`` or ``none``."""

    name: str
    bits: int | None
    rank: int | None

    def base_bytes(self, n_out: int, n_in: int, block: int = DEFAULT_BLOCK,
                   factor_itemsize: int = 2) -> int:
        """Stored bytes of this base for an ``[n_out, n_in]`` matrix (compact form)."""
        b = 0
        if self.bits is not None:
            b += quant_bytes(n_out, n_in, self.bits, block)
        if self.rank is not None:
            b += factor_itemsize * self.rank * (n_in + n_out)
        return b


def parse_base(base: str | BaseSpec) -> BaseSpec:
    """Parse a base name (see module doc). Raises ValueError ('unknown base ...') otherwise."""
    if isinstance(base, BaseSpec):
        return base
    s = str(base).strip().lower()
    if s == "none":
        return BaseSpec("none", None, None)
    m = _RE_QLR.fullmatch(s)
    if m:
        return BaseSpec(s, int(m.group(1)), int(m.group(2)))
    m = _RE_Q.fullmatch(s)
    if m:
        return BaseSpec(s, int(m.group(1)), None)
    m = _RE_LR.fullmatch(s)
    if m:
        return BaseSpec(s, None, int(m.group(1)))
    raise ValueError(f"unknown base {base!r}: expected q<2-8>, lowrank<r>, q<b>+lowrank<r> or none "
                     f"(e.g. {', '.join(BASE_EXAMPLES)})")


def bytes_per_elem(bits: int, block: int = DEFAULT_BLOCK) -> float:
    """``bits/8 + 2/block``: q2 0.3125, q3 0.4375, q4 0.5625 (= Q4_0), q8 1.0625 at block 32."""
    return bits / 8.0 + 2.0 / block


def quant_bytes(n_rows: int, n_cols: int, bits: int, block: int = DEFAULT_BLOCK) -> int:
    """Stored bytes of an ``[n_rows, n_cols]`` block-quantised matrix (whole blocks, f16 scale)."""
    nb = _ceil_div(n_cols, block)
    return int(n_rows) * nb * (2 + _ceil_div(block * bits, 8))


def residual_bytes(n_out: int, width: int, res_bits: int, block: int = DEFAULT_BLOCK) -> int:
    """Stored bytes of one residual group ``[n_out, width]`` at ``res_bits`` (16 = f16, 2 B/elem)."""
    if res_bits == 16:
        return 2 * int(n_out) * int(width)
    return quant_bytes(n_out, width, res_bits, block)


def q4_0_nbytes(n_out: int, n_in: int) -> int:
    """Bytes of an ``[n_out, n_in]`` matrix in ggml Q4_0 (65,028,096 for a 31B FFN matrix)."""
    return int(n_out) * _ceil_div(n_in, QK4_0) * Q4_0_BLOCK_BYTES


def group_slices(n_in: int, group_cols: int) -> list[slice]:
    """Contiguous input-column groups ``[i*g, min((i+1)*g, n_in))``; the last may be shorter."""
    if group_cols < 1:
        raise ValueError(f"group_cols must be >= 1, got {group_cols}")
    if n_in < 1:
        raise ValueError(f"n_in must be >= 1, got {n_in}")
    return [slice(c, min(c + group_cols, n_in)) for c in range(0, n_in, group_cols)]


def npz_size_estimate(n_out: int, n_in: int, group_cols: int, res_bits: int) -> int:
    """Approximate size of one decomposition .npz (dense f16 base + stacked residual)."""
    g_max = min(group_cols, n_in)
    n_groups = _ceil_div(n_in, group_cols)
    res_item = 4 if res_bits == 16 else 2
    return 2 * n_out * n_in + n_groups * n_out * g_max * res_item + (1 << 16)


# --------------------------------------------------------------------------------------------
# Dequantisation and matrix loading
# --------------------------------------------------------------------------------------------
def _as_u8(buf: Any) -> np.ndarray:
    if isinstance(buf, np.ndarray):
        return buf.reshape(-1).view(np.uint8)
    return np.frombuffer(buf, dtype=np.uint8)


def dequant_q4_0(buf: Any, n_rows: int, n_cols: int, *, out: np.ndarray | None = None) -> np.ndarray:
    """Dequantise ggml Q4_0 bytes to float32 ``[n_rows, n_cols]`` (bit-exact with gguf-py).

    ``buf``: bytes, bytearray, memoryview or a uint8 numpy array/memmap of exactly
    ``n_rows * n_cols/32 * 18`` bytes. Block layout: f16 ``d``, then 16 bytes; byte ``j`` holds
    element ``j`` in its low nibble and element ``j+16`` in its high nibble; value ``d*(q-8)``.
    Works in row chunks so a 65 MB tensor never needs more than ~32 MB of temporaries.
    """
    n_rows, n_cols = int(n_rows), int(n_cols)
    if n_cols % QK4_0:
        raise ValueError(f"Q4_0 needs n_cols % 32 == 0, got {n_cols}")
    nb = n_cols // QK4_0
    raw = _as_u8(buf)
    expected = n_rows * nb * Q4_0_BLOCK_BYTES
    if raw.size != expected:
        raise ValueError(f"Q4_0 buffer has {raw.size} bytes, expected {expected} for [{n_rows}, {n_cols}]")
    if out is None:
        out = np.empty((n_rows, n_cols), dtype=np.float32)
    elif out.shape != (n_rows, n_cols) or out.dtype != np.float32 or not out.flags.c_contiguous:
        raise ValueError("out must be a C-contiguous float32 array of shape (n_rows, n_cols)")
    blocks = raw.reshape(n_rows, nb, Q4_0_BLOCK_BYTES)
    o = out.reshape(n_rows, nb, QK4_0)
    step = max(1, _CHUNK_ELEMS // max(n_cols, 1))
    eight = np.int8(8)
    for r0 in range(0, n_rows, step):
        r1 = min(n_rows, r0 + step)
        b = blocks[r0:r1]
        d = np.ascontiguousarray(b[..., :2]).view(np.float16).astype(np.float32)  # [r, nb, 1]
        qs = b[..., 2:]
        o[r0:r1, :, :16] = d * ((qs & np.uint8(0x0F)).astype(np.int8) - eight).astype(np.float32)
        o[r0:r1, :, 16:] = d * ((qs >> np.uint8(4)).astype(np.int8) - eight).astype(np.float32)
    return out


def dequant_tensor(buf: Any, ggml_type: str, ne: Sequence[int]) -> np.ndarray:
    """Dequantise one GGUF tensor to float32 ``[prod(ne[1:]), ne[0]]`` (2-D, rows = ne[1..])."""
    ne = [int(x) for x in ne]
    if not ne:
        raise ValueError("empty shape")
    ne0 = ne[0]
    n_rows = 1
    for x in ne[1:]:
        n_rows *= x
    raw = _as_u8(buf)
    t = str(ggml_type).upper()
    if t == "Q4_0":
        return dequant_q4_0(raw, n_rows, ne0)
    if t == "F32":
        return raw.view("<f4").reshape(n_rows, ne0).astype(np.float32, copy=True)
    if t == "F16":
        return raw.view("<f2").reshape(n_rows, ne0).astype(np.float32)
    if t == "BF16":
        u = raw.view("<u2").reshape(n_rows, ne0).astype(np.uint32) << np.uint32(16)
        return u.view(np.float32)
    try:
        import gguf  # noqa: F401 - pigemma/__init__ puts third_party gguf-py on sys.path
        from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
        from gguf.quants import dequantize
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"dequantising {t} needs gguf-py ({type(e).__name__}: {e})") from e
    qt = GGMLQuantizationType[t]
    block, tsize = GGML_QUANT_SIZES[qt]
    if ne0 % block:
        raise ValueError(f"{t}: ne[0]={ne0} is not a multiple of the block size {block}")
    row_bytes = ne0 // block * tsize
    if raw.size != n_rows * row_bytes:
        raise ValueError(f"{t} buffer has {raw.size} bytes, expected {n_rows * row_bytes}")
    rows = raw.reshape(n_rows, row_bytes)
    out = np.empty((n_rows, ne0), dtype=np.float32)
    step = max(1, _CHUNK_ELEMS // max(ne0, 1))
    for r0 in range(0, n_rows, step):
        r1 = min(n_rows, r0 + step)
        out[r0:r1] = dequantize(np.ascontiguousarray(rows[r0:r1]), qt)
    return out


@dataclass(frozen=True)
class TensorInfo:
    """Where a matrix lives: absolute byte range in a GGUF and its orientation."""

    name: str
    type: str
    shape: tuple[int, ...]     # gguf ne order: (n_in, n_out[, 1...])
    n_bytes: int
    offset: int                # absolute file offset
    path: Path
    file_size: int

    @property
    def n_in(self) -> int:
        return int(self.shape[0])

    @property
    def n_out(self) -> int:
        n = 1
        for x in self.shape[1:]:
            n *= int(x)
        return n


_HEADER_CACHE: dict[tuple[str, int, int], gm.GgufHeader] = {}


def _header(p: Path) -> gm.GgufHeader:
    st = p.stat()
    key = (str(p.resolve()), int(st.st_size), int(st.st_mtime_ns))
    hdr = _HEADER_CACHE.get(key)
    if hdr is None:
        hdr = gm.read_gguf_header(p, "stdlib")
        _HEADER_CACHE[key] = hdr
    return hdr


def resolve_gguf(gmap: gm.GgufMap, override: str | os.PathLike[str] | None = None) -> Path:
    """Find the GGUF a map describes and check its size against the map.

    Order: ``override`` (path or registry id); else the map's ``gguf`` field (repo-relative or
    absolute), the registry id of its basename (honours $PIGEMMA_MODELS_DIR / $MODEL_DIR), and
    the basename next to the map file. Raises FileNotFoundError, or ValueError on a size mismatch.
    """
    cands: list[Path] = []
    if override is not None:
        cands.append(paths.model_path(override))
    else:
        cands.append(paths.model_path(gmap.gguf))
        mid = paths.model_id_for(gmap.gguf_basename)
        if mid:
            cands.append(paths.model_path(mid))
        if gmap.path is not None:
            cands.append(gmap.path.parent / gmap.gguf_basename)
    for c in cands:
        if c.is_file():
            size = c.stat().st_size
            if size != gmap.file_size:
                raise ValueError(f"{c}: file size {size} != map file_size {gmap.file_size} "
                                 f"(map built from a different file or a partial download)")
            return c
    raise FileNotFoundError("GGUF of the map not found; tried: " + ", ".join(str(c) for c in cands))


def tensor_info(map_or_gguf: Any, tensor_name: str, *,
                gguf: str | os.PathLike[str] | None = None) -> TensorInfo:
    """Locate ``tensor_name``.

    ``map_or_gguf`` is a GgufMap / map dict / map.json path, or a GGUF path / registry id (header
    read with the stdlib parser and cached). Raises FileNotFoundError, KeyError (tensor absent)
    or ValueError (not a matrix, range outside the file).
    """
    gmap: gm.GgufMap | None = None
    if isinstance(map_or_gguf, (gm.GgufMap, Mapping)):
        gmap = gm.load_map(map_or_gguf)
    else:
        s = os.fspath(map_or_gguf)
        if str(s).lower().endswith(".json"):
            p = Path(s)
            if not p.is_file():
                raise FileNotFoundError(f"map not found: {p}")
            gmap = gm.GgufMap.load(p)
    if gmap is not None:
        t = gmap.tensor(tensor_name)
        p = resolve_gguf(gmap, gguf)
        ttype = str(t["type"])
        shape = tuple(int(x) for x in t["shape"])
        n_bytes, offset = int(t["n_bytes"]), int(t["offset"])
        file_size = gmap.file_size
    else:
        p = paths.model_path(gguf if gguf is not None else map_or_gguf)
        if not p.is_file():
            raise FileNotFoundError(f"GGUF not found: {p}")
        hdr = _header(p)
        rec = next((r for r in hdr.tensors if r.name == tensor_name), None)
        if rec is None:
            raise KeyError(f"tensor {tensor_name!r} not in {p.name}")
        ttype, shape = rec.type, tuple(int(x) for x in rec.shape)
        n_bytes, offset = int(rec.n_bytes), int(rec.offset)
        file_size = hdr.file_size
    if len(shape) < 2 or any(x != 1 for x in shape[2:]):
        raise ValueError(f"tensor {tensor_name!r} has shape {list(shape)}: not a 2-D matrix")
    if offset < 0 or offset + n_bytes > file_size:
        raise ValueError(f"tensor {tensor_name!r} range [{offset}, {offset + n_bytes}) exceeds "
                         f"the file size {file_size}")
    return TensorInfo(tensor_name, ttype, shape, n_bytes, offset, p, file_size)


def load_matrix(map_or_gguf: Any, tensor_name: str, *,
                gguf: str | os.PathLike[str] | None = None) -> np.ndarray:
    """Read only ``tensor_name``'s byte range (np.memmap) and dequantise it.

    Returns float32 ``[n_out, n_in]`` (C order) for ``y = W @ x``; GGUF ``ne = [n_in, n_out]``.
    ``map_or_gguf``: a GgufMap, a map dict, a map.json path, or a GGUF path / registry id;
    ``gguf`` overrides the map's GGUF location.
    """
    info = tensor_info(map_or_gguf, tensor_name, gguf=gguf)
    mm = np.memmap(info.path, dtype=np.uint8, mode="r", offset=info.offset, shape=(info.n_bytes,))
    try:
        W = dequant_tensor(mm, info.type, info.shape)
    finally:
        del mm  # release the mapping now (Windows keeps the file locked while it is mapped)
    return W.reshape(info.n_out, info.n_in)


def ffn_names(layer: int) -> dict[str, str]:
    """GGUF names of the dense FFN matrices of ``layer``: {'gate': ..., 'up': ..., 'down': ...}."""
    il = int(layer)
    return {m: f"blk.{il}.ffn_{m}.weight" for m in FFN_MATS}


# --------------------------------------------------------------------------------------------
# Bases
# --------------------------------------------------------------------------------------------
def _quant_rows(x: np.ndarray, bits: int, block: int) -> np.ndarray:
    """Block-quantise + dequantise float32 rows ``x [rows, n_cols]`` along the last axis."""
    rows, n_cols = x.shape
    nb = _ceil_div(n_cols, block)
    width = nb * block
    if width != n_cols:
        xp = np.zeros((rows, width), dtype=np.float32)
        xp[:, :n_cols] = x
    else:
        xp = np.ascontiguousarray(x, dtype=np.float32)
    blk = xp.reshape(rows, nb, block)
    n = 1 << (bits - 1)
    imax = np.abs(blk).argmax(axis=-1)[..., None]
    mx = np.take_along_axis(blk, imax, axis=-1)            # signed absmax, [rows, nb, 1]
    d = mx / np.float32(-n)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d == 0, np.float32(0), np.float32(1) / d).astype(np.float32)
    q = np.trunc(blk * inv + np.float32(n + 0.5))
    np.clip(q, 0, 2 * n - 1, out=q)
    q -= np.float32(n)
    d16 = d.astype(np.float16).astype(np.float32)
    out = (d16 * q).reshape(rows, width)
    return out[:, :n_cols] if width != n_cols else out


def group_quant(W: np.ndarray, bits: int, block: int = DEFAULT_BLOCK) -> tuple[np.ndarray, int]:
    """Symmetric absmax block quantisation with an f16 scale per block; returns ``(W_hat, bytes)``.

    Blocks of ``block`` run along the last axis (the input dimension of ``W[n_out, n_in]``).
    ggml rule: ``d = max/-2^(bits-1)`` where ``max`` is the block's largest-magnitude element with
    its sign, ``q = clip(trunc(x/d + 2^(bits-1) + 0.5), 0, 2^bits-1)``, value
    ``f16(d) * (q - 2^(bits-1))``. ``bits=4`` reproduces ggml/gguf-py Q4_0 exactly. ``W_hat`` is
    float32 of W's shape; ``bytes = rows * ceil(n_cols/block) * (2 + block*bits/8)``.
    """
    bits, block = int(bits), int(block)
    if not 2 <= bits <= 8:
        raise ValueError(f"bits must be in 2..8, got {bits}")
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    A0 = np.asarray(W)
    A = A0.reshape(1, -1) if A0.ndim == 1 else A0
    if A.ndim != 2:
        raise ValueError(f"group_quant expects a 2-D matrix, got shape {A0.shape}")
    rows, n_cols = A.shape
    out = np.empty((rows, n_cols), dtype=np.float32)
    step = max(1, _CHUNK_ELEMS // max(n_cols, 1))
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        out[r0:r1] = _quant_rows(np.asarray(A[r0:r1], dtype=np.float32), bits, block)
    return out.reshape(A0.shape), quant_bytes(rows, n_cols, bits, block)


def _rsvd_numpy(A: np.ndarray, q: int, niter: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized SVD (Halko et al.): range finder with QR-stabilised power iterations."""
    _, n = A.shape
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, q)).astype(A.dtype, copy=False)
    Q, _ = np.linalg.qr(A @ omega)
    for _ in range(int(niter)):
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)
    B = Q.T @ A
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    return Q @ Ub, S, Vt


def _rsvd_torch(A: np.ndarray, q: int, niter: int, seed: int,
                threads: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch  # lazy: only the low-rank bases need it

    if threads:
        torch.set_num_threads(int(threads))
    t = torch.from_numpy(A)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        U, S, V = torch.svd_lowrank(t, q=int(q), niter=int(niter))
    return U.numpy(), S.numpy(), V.T.contiguous().numpy()


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001 - missing or broken torch -> numpy fallback
        log.info("torch not importable (%s: %s); using the numpy randomized SVD", type(e).__name__, e)
        return False
    return True


def lowrank(W: np.ndarray, r: int, *, backend: str = "auto", seed: int = 0,
            niter: int = LOWRANK_NITER, oversample: int = LOWRANK_OVERSAMPLE,
            factor_dtype: Any = np.float16, threads: int | None = None,
            info: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Rank-``r`` approximation ``W ~= (U * S) @ Vt`` by randomized SVD.

    Returns ``(U [n_out, r] factor_dtype, S float32 [r], Vt [r, n_in] factor_dtype, bytes)`` with
    ``bytes = itemsize * r * (n_in + n_out)`` (``2*r*(n_in+n_out)`` for the default f16 factors;
    the r singular values are assumed folded into one factor when streamed). ``backend``:
    'torch' (``torch.svd_lowrank(q=r+oversample, niter)``), 'numpy' (randomized range finder) or
    'auto' (torch if importable). ``q`` is clamped to ``min(W.shape)``; with ``r = min(W.shape)``
    the factorisation is exact up to rounding. ``info`` (optional dict) receives
    {backend, q, niter, seed, seconds}.
    """
    A = np.asarray(W)
    if A.ndim != 2:
        raise ValueError(f"lowrank expects a 2-D matrix, got shape {A.shape}")
    m, n = A.shape
    r = int(r)
    if not 1 <= r <= min(m, n):
        raise ValueError(f"rank r={r} must be in 1..{min(m, n)} for a [{m}, {n}] matrix")
    A = np.require(A, dtype=np.float32, requirements=["C", "W"])
    q = min(r + int(oversample), m, n)
    if backend not in ("auto", "torch", "numpy"):
        raise ValueError(f"backend must be auto|torch|numpy, got {backend!r}")
    use = backend
    if use == "auto":
        use = "torch" if _torch_available() else "numpy"
    t0 = time.perf_counter()
    if use == "torch":
        U, S, Vt = _rsvd_torch(A, q, niter, seed, threads)
    else:
        U, S, Vt = _rsvd_numpy(A, q, niter, seed)
    U = np.ascontiguousarray(U[:, :r]).astype(factor_dtype)
    S = np.asarray(S[:r], dtype=np.float32).copy()
    Vt = np.ascontiguousarray(Vt[:r, :]).astype(factor_dtype)
    if not (np.isfinite(U).all() and np.isfinite(Vt).all() and np.isfinite(S).all()):
        raise FloatingPointError("low-rank factors are not finite (f16 overflow or SVD failure)")
    nbytes = int(np.dtype(factor_dtype).itemsize) * r * (m + n)
    if info is not None:
        info.update({"backend": use, "q": q, "niter": int(niter), "seed": int(seed),
                     "seconds": round(time.perf_counter() - t0, 3)})
    return U, S, Vt, nbytes


def _lowrank_product(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    return (U.astype(np.float32) * S.astype(np.float32)[None, :]) @ Vt.astype(np.float32)


def _round_f16_inplace(A: np.ndarray) -> None:
    """Round a float32 matrix to the nearest f16 value in place (row chunks; no full f16 copy)."""
    rows = A.shape[0]
    step = max(1, _CHUNK_ELEMS // max(A.shape[1], 1))
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        A[r0:r1] = A[r0:r1].astype(np.float16)


# --------------------------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------------------------
@dataclass
class Group:
    """One residual column group.

    ``R_hat`` is ``[n_out, width]``: float16 for ``res_bits < 16`` (the dequantised quantised
    residual, exactly the stored values) or float32 for 16 (exact). It is a view into
    ``Decomp.R_stack``. ``norm2_g`` (largest singular value, rounded up by NORM2_RTOL) and
    ``normF_g`` are computed in float64 on exactly these values.
    """

    cols: slice
    R_hat: np.ndarray
    bytes_g: int
    norm2_g: float
    normF_g: float

    @property
    def start(self) -> int:
        return int(self.cols.start)

    @property
    def stop(self) -> int:
        return int(self.cols.stop)

    @property
    def width(self) -> int:
        return self.stop - self.start

    def R(self) -> np.ndarray:
        """The residual as float32 ``[n_out, width]`` (a copy)."""
        return self.R_hat.astype(np.float32)


@dataclass
class Decomp:
    """``W ~= W_base_hat + sum_g R_g`` placed at its columns (see module doc)."""

    W_base_hat: np.ndarray                 # float32 [n_out, n_in], values f16-representable
    base_bytes: int
    groups: list[Group]
    total_residual_bytes: int
    meta: dict[str, Any] = field(default_factory=dict)
    R_stack: np.ndarray | None = None      # [G, n_out, g_max], zero-padded, storage dtype

    # Shapes ---------------------------------------------------------------------------------
    @property
    def n_out(self) -> int:
        return int(self.W_base_hat.shape[0])

    @property
    def n_in(self) -> int:
        return int(self.W_base_hat.shape[1])

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def total_bytes(self) -> int:
        return int(self.base_bytes) + int(self.total_residual_bytes)

    @property
    def cols(self) -> np.ndarray:
        """int64 ``[G, 2]`` of ``[start, stop)``."""
        return np.array([[g.start, g.stop] for g in self.groups], dtype=np.int64).reshape(-1, 2)

    @property
    def norm2(self) -> np.ndarray:
        return np.array([g.norm2_g for g in self.groups], dtype=np.float64)

    @property
    def normF(self) -> np.ndarray:
        return np.array([g.normF_g for g in self.groups], dtype=np.float64)

    @property
    def bytes_g(self) -> np.ndarray:
        return np.array([g.bytes_g for g in self.groups], dtype=np.int64)

    # Algebra --------------------------------------------------------------------------------
    def stacked(self) -> np.ndarray:
        """``[G, n_out, g_max]`` residual stack (built once if the Decomp was assembled by hand)."""
        if self.R_stack is not None:
            return self.R_stack
        g_max = max((g.width for g in self.groups), default=0)
        dt = self.groups[0].R_hat.dtype if self.groups else np.float32
        st = np.zeros((len(self.groups), self.n_out, g_max), dtype=dt)
        for i, g in enumerate(self.groups):
            st[i, :, :g.width] = g.R_hat
        self.R_stack = st
        return st

    def reconstruct(self, groups: Iterable[int] | None = None) -> np.ndarray:
        """float32 ``W_base_hat + sum of the selected residual groups`` (all groups by default)."""
        out = np.array(self.W_base_hat, dtype=np.float32, copy=True)
        sel = range(len(self.groups)) if groups is None else groups
        for i in sel:
            g = self.groups[int(i)]
            out[:, g.cols] += g.R_hat.astype(np.float32)
        return out

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """``C[t, g] = R_g @ x_t[cols_g]``, float32 ``[T, G, n_out]`` (X ``[T, n_in]`` or ``[n_in]``)."""
        return group_contributions(self.stacked(), self.cols, X)

    def input_norms(self, X: np.ndarray) -> np.ndarray:
        """``||x_t[cols_g]||``, float64 ``[T, G]``; the bound is ``norm2[g] * input_norms[t, g]``."""
        return group_input_norms(self.cols, X)

    # Persistence ----------------------------------------------------------------------------
    def save(self, path: str | os.PathLike[str], **kw: Any) -> Path:
        """Write the npz (see :func:`save_npz`)."""
        return save_npz(self, path, **kw)

    @classmethod
    def from_npz(cls, path: str | os.PathLike[str]) -> "Decomp":
        """Rebuild a Decomp from an npz written by :func:`save_npz`."""
        z = load_npz(path)
        W_base_hat = np.asarray(z["W_base_hat"]).astype(np.float32)
        R_stack = np.asarray(z["R_hat"])
        cols = np.asarray(z["cols"], dtype=np.int64).reshape(-1, 2)
        norm2 = np.asarray(z["norm2"], dtype=np.float64)
        normF = np.asarray(z["normF"], dtype=np.float64)
        bytes_g = np.asarray(z["bytes_g"], dtype=np.int64)
        groups = []
        for i, (a, b) in enumerate(cols.tolist()):
            groups.append(Group(slice(int(a), int(b)), R_stack[i, :, :int(b) - int(a)], int(bytes_g[i]),
                                float(norm2[i]), float(normF[i])))
        meta: dict[str, Any] = {}
        if "meta_json" in z:
            with contextlib.suppress(Exception):
                meta = json.loads(str(z["meta_json"]))
        for k in ("layer", "mat", "base", "group_cols", "res_bits", "block", "q4_0_bytes", "tensor", "gguf"):
            if k in z:
                meta[k] = z[k]
        return cls(W_base_hat, int(z["base_bytes"]), groups, int(bytes_g.sum()), meta, R_stack)


def group_input_norms(cols: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``||x_t[a:b]||`` for every row of X and every ``[a, b)`` in cols; float64 ``[T, G]``."""
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim == 1:
        Xa = Xa[None, :]
    cols = np.asarray(cols, dtype=np.int64).reshape(-1, 2)
    out = np.empty((Xa.shape[0], cols.shape[0]), dtype=np.float64)
    for i, (a, b) in enumerate(cols.tolist()):
        out[:, i] = np.linalg.norm(Xa[:, a:b], axis=1)
    return out


def group_contributions(R_stack: np.ndarray, cols: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``C[t, g] = R_stack[g, :, :w_g] @ x_t[cols_g]``; float32 ``[T, G, n_out]``.

    ``R_stack`` is the npz ``R_hat`` (``[G, n_out, g_max]``, zero-padded). Memory: one float32
    copy of the stack plus ``T*G*n_out`` floats; batch X (e.g. 32 tokens) for 31B matrices.
    """
    Xa = np.asarray(X, dtype=np.float32)
    if Xa.ndim == 1:
        Xa = Xa[None, :]
    cols = np.asarray(cols, dtype=np.int64).reshape(-1, 2)
    G, _, g_max = R_stack.shape
    if cols.shape[0] != G:
        raise ValueError(f"cols has {cols.shape[0]} groups, R_stack has {G}")
    T = Xa.shape[0]
    Xg = np.zeros((G, g_max, T), dtype=np.float32)
    for i, (a, b) in enumerate(cols.tolist()):
        Xg[i, : b - a, :] = Xa[:, a:b].T
    R32 = R_stack if R_stack.dtype == np.float32 else R_stack.astype(np.float32)
    C = np.matmul(R32, Xg)                 # [G, n_out, T]
    return np.ascontiguousarray(C.transpose(2, 0, 1))


def _norm2_float64(R: np.ndarray) -> float:
    """Largest singular value of R (float64 LAPACK; Gram-matrix eigvalsh fallback)."""
    if R.size == 0:
        return 0.0
    try:
        s = np.linalg.svd(R, compute_uv=False)
        return float(s[0]) if s.size else 0.0
    except np.linalg.LinAlgError:
        gram = R.T @ R if R.shape[0] >= R.shape[1] else R @ R.T
        return float(math.sqrt(max(float(np.linalg.eigvalsh(gram)[-1]), 0.0)))


def decompose(W: np.ndarray, base: str | BaseSpec = "q2", group_cols: int = DEFAULT_GROUP_COLS,
              res_bits: int = DEFAULT_RES_BITS, *, block: int = DEFAULT_BLOCK,
              lowrank_backend: str = "auto", seed: int = 0, niter: int = LOWRANK_NITER,
              oversample: int = LOWRANK_OVERSAMPLE, threads: int | None = None) -> Decomp:
    """Build ``W ~= W_base_hat + sum_g R_g`` for ``W[n_out, n_in]`` (see module doc).

    ``base``: 'q2' | 'q3' | 'lowrank256' | 'lowrank512' | 'q2+lowrank128' (any q<b>, lowrank<r>,
    q<b>+lowrank<r>, or 'none'). ``group_cols``: input columns per residual group (last group
    may be shorter). ``res_bits``: 2..8 = block quantisation of each group's residual along its
    columns (block ``block``, f16 scale), 16 = unquantised (exact float32, accounted as f16).
    Memory: W float32, one float32 base, the residual stack (f16 for res_bits < 16) and one
    group of float64 temporaries.
    """
    spec = parse_base(base)
    res_bits, group_cols, block = int(res_bits), int(group_cols), int(block)
    if res_bits != 16 and not 2 <= res_bits <= 8:
        raise ValueError(f"res_bits must be 2..8 or 16, got {res_bits}")
    W32 = np.asarray(W, dtype=np.float32)
    if W32.ndim != 2:
        raise ValueError(f"decompose expects a 2-D matrix, got shape {W32.shape}")
    if not np.isfinite(W32).all():
        raise ValueError("W has non-finite values")
    n_out, n_in = W32.shape
    t0 = time.perf_counter()

    # 1. Base ---------------------------------------------------------------------------------
    lr_info: dict[str, Any] | None = None
    lowrank_S: np.ndarray | None = None
    base_bytes = 0
    Wb: np.ndarray | None = None
    if spec.bits is not None:
        Wb, qb = group_quant(W32, spec.bits, block)
        base_bytes += qb
    if spec.rank is not None:
        if spec.rank > min(n_out, n_in):
            raise ValueError(f"base {spec.name}: rank {spec.rank} > min(n_out, n_in) = {min(n_out, n_in)}")
        target = W32 if Wb is None else (W32 - Wb)
        target_norm = float(np.linalg.norm(target))
        lr_info = {}
        U, S, Vt, lb = lowrank(target, spec.rank, backend=lowrank_backend, seed=seed, niter=niter,
                               oversample=oversample, threads=threads, info=lr_info)
        del target
        base_bytes += lb
        L = _lowrank_product(U, S, Vt)
        Wb = L if Wb is None else (Wb + L)
        del L, U, Vt
        lowrank_S = S
        lr_info.update({"rank": spec.rank, "factor_dtype": "float16", "bytes": lb,
                        "fitted_to": "W" if spec.bits is None else f"W - q{spec.bits}(W)",
                        "energy_fraction": (float(np.sum(S.astype(np.float64) ** 2)) / target_norm ** 2)
                        if target_norm > 0 else None})
    if Wb is None:
        Wb = np.zeros((n_out, n_in), dtype=np.float32)
    else:
        Wb = np.ascontiguousarray(Wb, dtype=np.float32)
        _round_f16_inplace(Wb)
        if not np.isfinite(Wb).all():
            raise FloatingPointError("the base overflows float16")
    t_base = time.perf_counter() - t0

    # 2. Residual groups ----------------------------------------------------------------------
    slices = group_slices(n_in, group_cols)
    g_max = slices[0].stop - slices[0].start
    store_dtype = np.float32 if res_bits == 16 else np.float16
    R_stack = np.zeros((len(slices), n_out, g_max), dtype=store_dtype)
    groups: list[Group] = []
    ss_W = ss_base = ss_recon = 0.0
    max_abs_recon = 0.0
    total_res = 0
    for i, sl in enumerate(slices):
        width = sl.stop - sl.start
        Wg = W32[:, sl]
        Rg = Wg - Wb[:, sl]                                    # exact residual, float32
        if res_bits == 16:
            R_stack[i, :, :width] = Rg
            bytes_g = residual_bytes(n_out, width, 16, block)
        else:
            Rq, bytes_g = group_quant(Rg, res_bits, block)
            R_stack[i, :, :width] = Rq                         # f16 rounding of d*q
            del Rq
        stored = R_stack[i, :, :width].astype(np.float64)
        if not np.isfinite(stored).all():
            raise FloatingPointError(f"group {i}: residual not finite in {np.dtype(store_dtype).name}")
        normF = float(np.sqrt(np.sum(stored * stored)))
        norm2 = _norm2_float64(stored) * (1.0 + NORM2_RTOL)
        Wg64 = Wg.astype(np.float64)
        Rg64 = Rg.astype(np.float64)
        diff = Rg64 - stored
        ss_W += float(np.sum(Wg64 * Wg64))
        ss_base += float(np.sum(Rg64 * Rg64))
        ss_recon += float(np.sum(diff * diff))
        if diff.size:
            max_abs_recon = max(max_abs_recon, float(np.max(np.abs(diff))))
        del stored, Wg64, Rg64, diff, Rg
        groups.append(Group(sl, R_stack[i, :, :width], int(bytes_g), norm2, normF))
        total_res += int(bytes_g)
    t_total = time.perf_counter() - t0

    W_normF = math.sqrt(ss_W)
    n_el = n_out * n_in
    ratios = [g.norm2_g / g.normF_g for g in groups if g.normF_g > 0]
    meta: dict[str, Any] = {
        "format_version": NPZ_FORMAT_VERSION,
        "generator": GENERATOR,
        "base": spec.name, "base_bits": spec.bits, "base_rank": spec.rank,
        "group_cols": group_cols, "res_bits": res_bits, "block": block,
        "n_out": n_out, "n_in": n_in, "n_groups": len(groups), "g_max": g_max,
        "residual_dtype": np.dtype(store_dtype).name,
        "W_normF": W_normF,
        "base_rel_err": (math.sqrt(ss_base) / W_normF) if W_normF > 0 else 0.0,
        "recon_rel_err": (math.sqrt(ss_recon) / W_normF) if W_normF > 0 else 0.0,
        "recon_max_abs_err": max_abs_recon,
        "base_bytes": int(base_bytes), "total_residual_bytes": int(total_res),
        "base_bytes_per_elem": base_bytes / n_el,
        "residual_bytes_per_elem": total_res / n_el,
        "total_bytes_per_elem": (base_bytes + total_res) / n_el,
        "q4_0_bytes": q4_0_nbytes(n_out, n_in),
        "norm2_rtol": NORM2_RTOL,
        "norm2_over_normF": ({"min": float(np.min(ratios)), "median": float(np.median(ratios)),
                              "max": float(np.max(ratios))} if ratios else None),
        "lowrank": lr_info,
        "seconds": {"base": round(t_base, 3), "residual": round(t_total - t_base, 3),
                    "total": round(t_total, 3)},
    }
    d = Decomp(Wb, int(base_bytes), groups, int(total_res), meta, R_stack)
    if lowrank_S is not None:
        d.meta["_lowrank_S"] = lowrank_S  # saved as the npz array 'lowrank_S', not in meta_json
    return d


# --------------------------------------------------------------------------------------------
# npz files (TRACE_FORMAT 13.3)
# --------------------------------------------------------------------------------------------
def decomp_path(out_dir: str | os.PathLike[str], layer: int, mat: str, base: str) -> Path:
    """``<out_dir>/L<layer>.<mat>.<base>.npz`` (e.g. traces/h2/decomp/L20.down.q2.npz)."""
    return Path(os.fspath(out_dir)) / f"L{int(layer)}.{mat}.{parse_base(base).name}.npz"


def save_npz(decomp: Decomp, path: str | os.PathLike[str], *, layer: int = -1, mat: str = "",
             q4_0_bytes: int | None = None, tensor: str = "", gguf: str = "",
             orig_type: str = "", orig_bytes: int | None = None,
             extra_meta: Mapping[str, Any] | None = None) -> Path:
    """Write the decomposition npz (uncompressed ``np.savez`` to a temp file, then atomic rename).

    Arrays (TRACE_FORMAT 13.3): W_base_hat f16 [n_out, n_in]; R_hat [G, n_out, g_max] (f16, or
    f32 for res_bits 16; a short last group is zero-padded on the right); cols int64 [G, 2];
    norm2, normF float64 [G]; bytes_g int64 [G]; int64 scalars base_bytes, q4_0_bytes, layer,
    group_cols, res_bits, n_out, n_in; 0-d unicode mat, base. Extras: format_version, block,
    n_groups, total_residual_bytes, tensor, gguf, orig_type, orig_bytes, base_rel_err,
    recon_rel_err, recon_max_abs_err, W_normF, norm2_rtol, created_at, generator, meta_json
    (JSON string of Decomp.meta + file info), and lowrank_S float32 [r] for low-rank bases.
    """
    p = Path(os.fspath(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    m = decomp.meta
    q4b = int(q4_0_bytes) if q4_0_bytes is not None else q4_0_nbytes(decomp.n_out, decomp.n_in)
    created = _utc_now()
    meta_out = {k: v for k, v in m.items() if not str(k).startswith("_")}
    meta_out.update({"layer": int(layer), "mat": mat, "tensor": tensor, "gguf": gguf,
                     "orig_type": orig_type, "orig_bytes": orig_bytes, "q4_0_bytes": q4b,
                     "created_at": created})
    if extra_meta:
        meta_out.update(dict(extra_meta))
    nan = float("nan")
    arrays: dict[str, np.ndarray] = {
        "W_base_hat": decomp.W_base_hat.astype(np.float16),
        "R_hat": decomp.stacked(),
        "cols": decomp.cols,
        "norm2": decomp.norm2,
        "normF": decomp.normF,
        "bytes_g": decomp.bytes_g,
        "base_bytes": np.array(int(decomp.base_bytes), dtype=np.int64),
        "q4_0_bytes": np.array(q4b, dtype=np.int64),
        "layer": np.array(int(layer), dtype=np.int64),
        "group_cols": np.array(int(m.get("group_cols", 0)), dtype=np.int64),
        "res_bits": np.array(int(m.get("res_bits", 0)), dtype=np.int64),
        "n_out": np.array(decomp.n_out, dtype=np.int64),
        "n_in": np.array(decomp.n_in, dtype=np.int64),
        "mat": np.array(str(mat)),
        "base": np.array(str(m.get("base", ""))),
        "format_version": np.array(NPZ_FORMAT_VERSION, dtype=np.int64),
        "block": np.array(int(m.get("block", DEFAULT_BLOCK)), dtype=np.int64),
        "n_groups": np.array(decomp.n_groups, dtype=np.int64),
        "total_residual_bytes": np.array(int(decomp.total_residual_bytes), dtype=np.int64),
        "tensor": np.array(str(tensor)),
        "gguf": np.array(str(gguf)),
        "orig_type": np.array(str(orig_type)),
        "orig_bytes": np.array(int(orig_bytes) if orig_bytes is not None else -1, dtype=np.int64),
        "base_rel_err": np.array(float(m.get("base_rel_err", nan)), dtype=np.float64),
        "recon_rel_err": np.array(float(m.get("recon_rel_err", nan)), dtype=np.float64),
        "recon_max_abs_err": np.array(float(m.get("recon_max_abs_err", nan)), dtype=np.float64),
        "W_normF": np.array(float(m.get("W_normF", nan)), dtype=np.float64),
        "norm2_rtol": np.array(float(m.get("norm2_rtol", NORM2_RTOL)), dtype=np.float64),
        "created_at": np.array(created),
        "generator": np.array(GENERATOR),
        "meta_json": np.array(json.dumps(_jsonable(meta_out), sort_keys=True, allow_nan=False)),
    }
    if m.get("_lowrank_S") is not None:
        arrays["lowrank_S"] = np.asarray(m["_lowrank_S"], dtype=np.float32)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, p)
    finally:
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()
    return p


def load_npz(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a decomposition npz into a dict: arrays stay arrays, 0-d values become Python scalars."""
    out: dict[str, Any] = {}
    with np.load(os.fspath(path), allow_pickle=False) as z:
        for k in z.files:
            a = z[k]
            out[k] = a.item() if a.ndim == 0 else a
    return out


def _cached_matches(p: Path, expect: Mapping[str, Any]) -> bool:
    """True when ``p`` is a readable npz whose scalars equal ``expect`` (cheap: lazy npz members)."""
    try:
        with np.load(p, allow_pickle=False) as z:
            files = set(z.files)
            for k, v in expect.items():
                if k not in files:
                    return False
                a = z[k]
                if a.ndim != 0 or a.item() != v:
                    return False
            return {"W_base_hat", "R_hat", "cols", "norm2", "normF", "bytes_g"} <= files
    except Exception as e:  # noqa: BLE001 - partial/corrupt file -> recompute
        log.info("%s: not reusable (%s: %s)", p.name, type(e).__name__, e)
        return False


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _parse_int_list(s: str, what: str) -> list[int]:
    out: list[int] = []
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a:
                raise argparse.ArgumentTypeError(f"{what}: bad range {part!r}")
            out.extend(range(a, b + 1))
        elif part.isdigit():
            out.append(int(part))
        else:
            raise argparse.ArgumentTypeError(f"{what}: {part!r} is not an int or a range a-b")
    if not out:
        raise argparse.ArgumentTypeError(f"{what}: empty list")
    return list(dict.fromkeys(out))


def _parse_str_list(s: str) -> list[str]:
    return list(dict.fromkeys(p.strip() for p in str(s).split(",") if p.strip()))


def _positive_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
    return v


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m pigemma.h2.decompose",
        description=("H2 step 1: decompose dense FFN matrices of a GGUF into a cheap base + residual "
                     "column groups with spectral/Frobenius norms and byte accounting vs Q4_0."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Example:\n  python -m pigemma.h2.decompose --map traces/h2/map31.json --layers 5,20,35,50 "
                "--mats gate,up,down --bases q2,lowrank256 --group 128 --res-bits 4 --out traces/h2/decomp/\n\n"
                "Bases: q<2-8>, lowrank<r>, q<b>+lowrank<r>, none. Residual bits: 2..8 (block quant, "
                "f16 scale) or 16 (unquantised).\n"
                "Outputs: <out>/L<il>.<mat>.<base>.npz (TRACE_FORMAT 13.3) + <out>/index.json.\n"
                "Exit codes: 0 ok, 1 error, 2 usage, 3 missing input (map/GGUF/tensor), 4 refused "
                "(disk space, bench lock), 130 interrupted."))
    ap.add_argument("--map", default=None,
                    help="map.json from 'python -m pigemma.gguf_map build' (e.g. traces/h2/map31.json)")
    ap.add_argument("--gguf", default=None,
                    help="GGUF path or registry id; overrides the map's GGUF location "
                         "(without --map the header is read directly)")
    ap.add_argument("--layers", default="5,20,35,50",
                    help="comma list of layers, ranges a-b allowed (default 5,20,35,50)")
    ap.add_argument("--mats", default="gate,up,down", help="subset of gate,up,down (default all three)")
    ap.add_argument("--bases", default="q2,lowrank256",
                    help="comma list of bases (default q2,lowrank256; also q3, lowrank512, q2+lowrank128, none)")
    ap.add_argument("--group", type=_positive_int, default=DEFAULT_GROUP_COLS,
                    help=f"input columns per residual group (default {DEFAULT_GROUP_COLS})")
    ap.add_argument("--res-bits", type=int, choices=RES_BITS_CHOICES, default=DEFAULT_RES_BITS,
                    help=f"residual bits per element (default {DEFAULT_RES_BITS}; 16 = unquantised)")
    ap.add_argument("--block", type=_positive_int, default=DEFAULT_BLOCK,
                    help=f"quantisation block along the input dimension (default {DEFAULT_BLOCK}, as Q4_0)")
    ap.add_argument("--out", default="traces/h2/decomp/", help="output directory (default traces/h2/decomp/)")
    ap.add_argument("--lowrank-backend", choices=("auto", "torch", "numpy"), default="auto",
                    help="randomized SVD backend (default auto: torch if importable)")
    ap.add_argument("--seed", type=int, default=0, help="randomized SVD seed (default 0)")
    ap.add_argument("--niter", type=int, default=LOWRANK_NITER,
                    help=f"randomized SVD power iterations (default {LOWRANK_NITER})")
    ap.add_argument("--threads", type=_positive_int, default=None,
                    help="torch intra-op threads (default: torch's choice)")
    ap.add_argument("--force", action="store_true", help="recompute even if a matching npz exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, projected bytes and disk use; write nothing")
    ap.add_argument("--json", default=None, metavar="OUT|-",
                    help="also write a JSON run summary (- = stdout; the table then goes to stderr)")
    ap.add_argument("--min-free-gb", type=float, default=2.0,
                    help="refuse (exit 4) unless this much disk stays free after the outputs (default 2)")
    ap.add_argument("--lock", action="store_true",
                    help="hold results/.bench.lock for the run so it never overlaps a model benchmark")
    ap.add_argument("--lock-min-free-gb", type=float, default=4.0,
                    help="available-RAM gate for --lock in GB (default 4; env PIGEMMA_MIN_FREE_GB overrides)")
    ap.add_argument("--log-level", default="INFO", help="logging level on stderr (default INFO)")
    return ap


@dataclass
class _Job:
    layer: int
    mat: str
    base: str
    tensor: str
    path: Path
    n_out: int
    n_in: int
    ttype: str
    n_bytes: int
    cached: bool = False


def _entry_from_decomp(job: _Job, d: Decomp, q4b: int, seconds: float) -> dict[str, Any]:
    m = d.meta
    n_el = d.n_out * d.n_in
    return {
        "file": job.path.name, "layer": job.layer, "mat": job.mat, "base": job.base,
        "tensor": job.tensor, "orig_type": job.ttype, "n_out": d.n_out, "n_in": d.n_in,
        "n_groups": d.n_groups, "group_cols": m["group_cols"], "res_bits": m["res_bits"],
        "block": m["block"], "q4_0_bytes": q4b, "base_bytes": d.base_bytes,
        "total_residual_bytes": d.total_residual_bytes,
        "base_bytes_per_elem": d.base_bytes / n_el, "residual_bytes_per_elem": d.total_residual_bytes / n_el,
        "base_over_q4_0": d.base_bytes / q4b, "total_over_q4_0": d.total_bytes / q4b,
        "base_rel_err": m["base_rel_err"], "recon_rel_err": m["recon_rel_err"],
        "norm2_over_normF_median": (m["norm2_over_normF"] or {}).get("median"),
        "lowrank_energy_fraction": (m.get("lowrank") or {}).get("energy_fraction"),
        "seconds": round(seconds, 2), "cached": False, "created_at": _utc_now(),
    }


def _entry_from_npz(job: _Job) -> dict[str, Any]:
    z = load_npz(job.path)
    n_el = int(z["n_out"]) * int(z["n_in"])
    q4b = int(z["q4_0_bytes"])
    meta: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        meta = json.loads(str(z.get("meta_json", "{}")))
    base_b, res_b = int(z["base_bytes"]), int(np.sum(z["bytes_g"]))
    return {
        "file": job.path.name, "layer": job.layer, "mat": job.mat, "base": job.base,
        "tensor": job.tensor, "orig_type": job.ttype, "n_out": int(z["n_out"]), "n_in": int(z["n_in"]),
        "n_groups": int(np.asarray(z["cols"]).reshape(-1, 2).shape[0]), "group_cols": int(z["group_cols"]),
        "res_bits": int(z["res_bits"]), "block": int(z.get("block", DEFAULT_BLOCK)),
        "q4_0_bytes": q4b, "base_bytes": base_b, "total_residual_bytes": res_b,
        "base_bytes_per_elem": base_b / n_el, "residual_bytes_per_elem": res_b / n_el,
        "base_over_q4_0": base_b / q4b, "total_over_q4_0": (base_b + res_b) / q4b,
        "base_rel_err": float(z.get("base_rel_err", float("nan"))),
        "recon_rel_err": float(z.get("recon_rel_err", float("nan"))),
        "norm2_over_normF_median": (meta.get("norm2_over_normF") or {}).get("median"),
        "lowrank_energy_fraction": (meta.get("lowrank") or {}).get("energy_fraction"),
        "seconds": (meta.get("seconds") or {}).get("total"), "cached": True,
        "created_at": str(z.get("created_at", "")),
    }


def _fmt_num(x: Any, fmt: str) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return format(x, fmt)


def _format_table(entries: Sequence[Mapping[str, Any]]) -> str:
    """The bytes-accounting table printed at the end of a run (and by --dry-run)."""
    hdr = (f"{'layer':>5} {'mat':<4} {'base':<14} {'n_out x n_in':>13} {'G':>4} {'base B/el':>9} "
           f"{'res B/el':>8} {'base/Q4_0':>9} {'tot/Q4_0':>8} {'base_err':>8} {'recon_err':>9} "
           f"{'n2/nF':>6} {'s':>6}  file")
    lines = ["Bytes accounting vs Q4_0 (0.5625 B/elem; 65,028,096 B per 31B FFN matrix). "
             "err = relative Frobenius error vs the original matrix; recon = base + ALL residual groups; "
             "n2/nF = median spectral/Frobenius norm ratio of the groups (bound looseness hint).",
             hdr, "-" * len(hdr)]
    tot_q4 = tot_base = tot_res = 0
    for e in entries:
        shape = f"{e['n_out']}x{e['n_in']}"
        lines.append(
            f"{e['layer']:>5} {e['mat']:<4} {e['base']:<14} {shape:>13} {e['n_groups']:>4} "
            f"{_fmt_num(e['base_bytes_per_elem'], '.4f'):>9} {_fmt_num(e['residual_bytes_per_elem'], '.4f'):>8} "
            f"{_fmt_num(e['base_over_q4_0'], '.3f'):>9} {_fmt_num(e['total_over_q4_0'], '.3f'):>8} "
            f"{_fmt_num(e['base_rel_err'], '.4f'):>8} {_fmt_num(e['recon_rel_err'], '.5f'):>9} "
            f"{_fmt_num(e['norm2_over_normF_median'], '.3f'):>6} {_fmt_num(e['seconds'], '.1f'):>6}  "
            f"{e['file']}{' (cached)' if e.get('cached') else ''}")
        tot_q4 += int(e["q4_0_bytes"])
        tot_base += int(e["base_bytes"])
        tot_res += int(e["total_residual_bytes"])
    if entries and tot_q4 > 0:
        lines.append("-" * len(hdr))
        lines.append(f"total: Q4_0 {tot_q4:,} B | base {tot_base:,} B ({tot_base / tot_q4:.3f}x) | "
                     f"residual {tot_res:,} B ({tot_res / tot_q4:.3f}x) | base + all residual "
                     f"{(tot_base + tot_res) / tot_q4:.3f}x Q4_0")
    return "\n".join(lines)


def _update_index(out_dir: Path, entries: Sequence[Mapping[str, Any]], run: Mapping[str, Any]) -> Path:
    p = out_dir / "index.json"
    old: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        old = json.loads(p.read_text(encoding="utf-8"))
    by_file = {e["file"]: e for e in old.get("entries", []) if isinstance(e, dict) and "file" in e}
    for e in entries:
        by_file[e["file"]] = dict(e)
    doc = {"format_version": NPZ_FORMAT_VERSION, "generator": GENERATOR, "updated_at": _utc_now(),
           "last_run": dict(run),
           "entries": sorted(by_file.values(), key=lambda e: (e["layer"], e["mat"], e["base"]))}
    _atomic_write_text(p, json.dumps(_jsonable(doc), indent=1, allow_nan=False) + "\n")
    return p


def _plan(args: argparse.Namespace) -> tuple[Any, Path, str, list[_Job]]:
    """Resolve inputs and build the job list (raises FileNotFoundError / KeyError / ValueError)."""
    layers = _parse_int_list(args.layers, "--layers")
    mats = _parse_str_list(args.mats)
    bad = [m for m in mats if m not in FFN_MATS]
    if bad or not mats:
        raise argparse.ArgumentTypeError(f"--mats: unknown matrices {bad} (choose from {', '.join(FFN_MATS)})")
    try:
        bases = list(dict.fromkeys(parse_base(b).name for b in _parse_str_list(args.bases)))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--bases: {e}") from None
    if not bases:
        raise argparse.ArgumentTypeError("--bases: empty list")
    out_dir = Path(args.out)
    if args.map:
        mp = Path(args.map)
        if not mp.is_file():
            raise FileNotFoundError(f"map not found: {mp}")
        src: Any = gm.GgufMap.load(mp)
        n_layer = src.n_layer
        gguf_label = src.gguf_basename
    elif args.gguf:
        gp = paths.model_path(args.gguf)
        if not gp.is_file():
            raise FileNotFoundError(f"GGUF not found: {gp}")
        src = gp
        hdr = _header(gp)
        arch = hdr.fields.get("general.architecture")
        n_layer = int(hdr.fields.get(f"{arch}.block_count", 10 ** 9)) if arch else 10 ** 9
        gguf_label = gp.name
    else:
        raise argparse.ArgumentTypeError("give --map (preferred) or --gguf")
    over = [il for il in layers if il >= n_layer]
    if over:
        raise argparse.ArgumentTypeError(f"--layers {over} out of range (model has {n_layer} layers)")
    jobs: list[_Job] = []
    for il in layers:
        names = ffn_names(il)
        for mat in mats:
            info = tensor_info(src, names[mat], gguf=args.gguf if args.map else None)
            for b in bases:
                spec = parse_base(b)
                if spec.rank is not None and spec.rank > min(info.n_out, info.n_in):
                    raise argparse.ArgumentTypeError(
                        f"base {b}: rank {spec.rank} > min(n_out, n_in) = {min(info.n_out, info.n_in)} "
                        f"of {names[mat]}")
                jobs.append(_Job(il, mat, b, names[mat], decomp_path(out_dir, il, mat, b), info.n_out,
                                 info.n_in, info.type, info.n_bytes))
    return src, out_dir, gguf_label, jobs


def _expect_scalars(job: _Job, args: argparse.Namespace, gguf_label: str) -> dict[str, Any]:
    return {"format_version": NPZ_FORMAT_VERSION, "layer": job.layer, "mat": job.mat, "base": job.base,
            "group_cols": int(args.group), "res_bits": int(args.res_bits), "block": int(args.block),
            "n_out": job.n_out, "n_in": job.n_in, "tensor": job.tensor, "gguf": gguf_label}


def _dry_run_rows(jobs: Sequence[_Job], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for j in jobs:
        spec = parse_base(j.base)
        n_el = j.n_out * j.n_in
        q4b = q4_0_nbytes(j.n_out, j.n_in)
        bb = spec.base_bytes(j.n_out, j.n_in, args.block)
        rb = sum(residual_bytes(j.n_out, s.stop - s.start, args.res_bits, args.block)
                 for s in group_slices(j.n_in, args.group))
        rows.append({"file": j.path.name, "layer": j.layer, "mat": j.mat, "base": j.base, "n_out": j.n_out,
                     "n_in": j.n_in, "n_groups": _ceil_div(j.n_in, args.group), "q4_0_bytes": q4b,
                     "base_bytes": bb, "total_residual_bytes": rb, "base_bytes_per_elem": bb / n_el,
                     "residual_bytes_per_elem": rb / n_el, "base_over_q4_0": bb / q4b,
                     "total_over_q4_0": (bb + rb) / q4b, "base_rel_err": None, "recon_rel_err": None,
                     "norm2_over_normF_median": None, "seconds": None, "cached": j.cached})
    return rows


def _run(args: argparse.Namespace) -> int:
    src, out_dir, gguf_label, jobs = _plan(args)
    for j in jobs:
        j.cached = ((not args.force) and j.path.is_file()
                    and _cached_matches(j.path, _expect_scalars(j, args, gguf_label)))
    todo = [j for j in jobs if not j.cached]
    need = sum(npz_size_estimate(j.n_out, j.n_in, args.group, args.res_bits) for j in todo)
    free = paths.free_disk_bytes(out_dir)
    sys.stderr.write(f"pigemma.h2.decompose: {len(jobs)} job(s) ({len(jobs) - len(todo)} cached) from "
                     f"{gguf_label}; group {args.group}, res_bits {args.res_bits}, block {args.block}; "
                     f"out {out_dir}\n")
    sys.stderr.write(f"pigemma.h2.decompose: projected new output {need / 1e9:.2f} GB, "
                     f"free disk {free / 1e9:.1f} GB\n")
    if args.dry_run:
        print("DRY RUN (projected byte accounting; errors are computed only by a real run)")
        print(_format_table(_dry_run_rows(jobs, args)))
        return paths.EXIT_OK
    if todo and free - need < args.min_free_gb * 1e9:
        sys.stderr.write(f"pigemma.h2.decompose: REFUSED: outputs need ~{need / 1e9:.2f} GB but only "
                         f"{free / 1e9:.2f} GB is free at {out_dir} (keeping >= {args.min_free_gb} GB free); "
                         "free space, choose fewer layers/bases, or lower --min-free-gb\n")
        return paths.EXIT_REFUSED
    out_dir.mkdir(parents=True, exist_ok=True)

    run_info: dict[str, Any] = {
        "created_at": _utc_now(), "argv": sys.argv[1:], "map": args.map, "gguf": gguf_label,
        "group_cols": args.group, "res_bits": args.res_bits, "block": args.block,
        "bases": _parse_str_list(args.bases), "seed": args.seed, "niter": args.niter,
        "lowrank_backend": args.lowrank_backend}
    entries: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    n_done, n_todo = 0, len(todo)
    floor_warned = False
    gguf_override = args.gguf if args.map else None
    # Group jobs by matrix so each matrix is read and dequantised once.
    by_matrix: dict[tuple[int, str], list[_Job]] = {}
    for j in jobs:
        by_matrix.setdefault((j.layer, j.mat), []).append(j)
    for (il, mat), mjobs in by_matrix.items():
        W: np.ndarray | None = None
        for j in mjobs:
            if j.cached:
                entries.append(_entry_from_npz(j))
                sys.stderr.write(f"pigemma.h2.decompose: L{il}.{mat}.{j.base}: reusing {j.path.name}\n")
                continue
            t0 = time.perf_counter()
            if W is None:
                W = load_matrix(src, j.tensor, gguf=gguf_override)
                log.info("L%d.%s: loaded %s (%s) -> [%d, %d] in %.1f s", il, mat, j.tensor, j.ttype,
                         W.shape[0], W.shape[1], time.perf_counter() - t0)
            d = decompose(W, j.base, args.group, args.res_bits, block=args.block,
                          lowrank_backend=args.lowrank_backend, seed=args.seed, niter=args.niter,
                          threads=args.threads)
            q4b = q4_0_nbytes(d.n_out, d.n_in)
            save_npz(d, j.path, layer=il, mat=mat, q4_0_bytes=q4b, tensor=j.tensor, gguf=gguf_label,
                     orig_type=j.ttype, orig_bytes=j.n_bytes,
                     extra_meta={"input_kind": FFN_INPUT_KIND.get(mat), "seed": args.seed})
            secs = time.perf_counter() - t0
            e = _entry_from_decomp(j, d, q4b, secs)
            entries.append(e)
            n_done += 1
            sys.stderr.write(
                f"pigemma.h2.decompose: [{n_done}/{n_todo}] L{il}.{mat}.{j.base}: base "
                f"{e['base_bytes_per_elem']:.4f} B/el (err {e['base_rel_err']:.4f}), residual "
                f"{e['residual_bytes_per_elem']:.4f} B/el, recon err {e['recon_rel_err']:.5f}, "
                f"{secs:.1f} s -> {j.path}\n")
            if not floor_warned and args.res_bits != 16 and e["recon_rel_err"] > 0.005:
                floor_warned = True
                sys.stderr.write(
                    f"pigemma.h2.decompose: NOTE: with res_bits={args.res_bits} base + ALL residual groups "
                    f"still differ from the original matrix by {e['recon_rel_err']:.2%} (Frobenius); an "
                    "oracle eps below about that cannot be reached with these residuals "
                    "(--res-bits 8 lowers the floor, 16 is exact)\n")
            left = n_todo - n_done
            if left > 0:
                elapsed = time.perf_counter() - t_start
                tag = "after the first matrix" if n_done == 1 else "running average"
                sys.stderr.write(f"pigemma.h2.decompose: ETA ~{_fmt_s(elapsed / n_done * left)} for {left} "
                                 f"remaining job(s) ({tag}: {elapsed / n_done:.1f} s/job)\n")
            sys.stderr.flush()
            del d
            gc.collect()
            _update_index(out_dir, entries, run_info)
        del W
        gc.collect()
    run_info["wall_s"] = round(time.perf_counter() - t_start, 2)
    idx = _update_index(out_dir, entries, run_info)
    table = _format_table(entries)
    summary = {"tool": "pigemma.h2.decompose", "version": __version__, "created_at": _utc_now(),
               "out_dir": out_dir.as_posix(), "index": idx.as_posix(), "run": run_info, "entries": entries}
    if args.json == "-":
        sys.stderr.write(table + "\n")
        sys.stdout.write(json.dumps(_jsonable(summary), indent=1, allow_nan=False) + "\n")
    else:
        print(table)
        if args.json:
            _atomic_write_text(Path(args.json), json.dumps(_jsonable(summary), indent=1, allow_nan=False) + "\n")
    sys.stderr.write(f"pigemma.h2.decompose: done in {_fmt_s(run_info['wall_s'])}; index {idx}\n")
    return paths.EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the exit code."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    ap = _build_parser()
    args = ap.parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        if args.lock and not args.dry_run:
            cmd = "python -m pigemma.h2.decompose " + " ".join(sys.argv[1:])
            with paths.bench_lock(args.lock_min_free_gb, cmd=cmd):
                return _run(args)
        return _run(args)
    except paths.BenchLockError as e:
        sys.stderr.write(f"pigemma.h2.decompose: REFUSED: {e}\n")
        return paths.EXIT_REFUSED
    except argparse.ArgumentTypeError as e:
        sys.stderr.write(f"pigemma.h2.decompose: usage: {e}\n")
        return paths.EXIT_USAGE
    except FileNotFoundError as e:
        sys.stderr.write(f"pigemma.h2.decompose: missing: {e}\n")
        return paths.EXIT_MISSING
    except KeyError as e:
        sys.stderr.write(f"pigemma.h2.decompose: missing: {e.args[0] if e.args else e}\n")
        return paths.EXIT_MISSING
    except KeyboardInterrupt:
        sys.stderr.write("pigemma.h2.decompose: interrupted\n")
        return 130
    except Exception as e:  # noqa: BLE001 - CLI boundary
        sys.stderr.write(f"pigemma.h2.decompose: error: {type(e).__name__}: {e}\n")
        return paths.EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
