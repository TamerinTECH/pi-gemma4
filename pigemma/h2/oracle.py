"""pigemma.h2.oracle - H2 step 2: oracle residual selection, norm-bound certification and the
iso-byte control on real FFN-input activations (experiment E7, docs/HYPOTHESES.md section 5).

Inputs (manifest files[] 'pigemma/h2/oracle.py'; formats: docs/TRACE_FORMAT.md 7 and 13)

* decomposition npz files ``<decomp>/L<il>.<mat>.<base>.npz`` written by
  :mod:`pigemma.h2.decompose` (``W ~= W_base_hat + sum_g R_g``, groups of input columns, one
  spectral norm ``norm2_g`` per group);
* an activation dump directory (``meta.json``, ``L<il>.ffn_norm.npy`` = input of gate/up,
  ``L<il>.ffn_geglu.npy`` = input of down, ``L<il>.ffn_out.npy`` for the sanity check, ``phase.u8``);
* the Q4_0 GGUF itself: ``y_ref = W_q4 @ x`` uses the exact dequantised Q4_0 matrix (what
  llama.cpp computes), read through ``decompose.load_matrix`` (``--map`` or ``--gguf``, else found
  from the npz ``gguf`` basename or the acts ``meta.json`` ``gguf_path``).

What is measured per (layer, mat, base) and per token (first ``--tokens`` tokens, reported for
phase ``all``, ``prefill`` (phase 0, teacher-forced) and ``gen`` (phase 1, generated))

* ``err_base = ||y_ref - W_base x|| / ||y_ref||``; ``floor = ||y_ref - (W_base + sum_g R_g) x|| /
  ||y_ref||`` (the residual quantisation floor: even reading every group leaves it).
* ORACLE: contributions ``C_g = R_g x_g`` are sorted by ``||C_g||`` descending and added to
  ``W_base x`` until the relative error is ``<= eps``; recorded: groups ``k``, residual byte
  fraction, and the error-vs-bytes curve at byte fractions 0, 0.05, ..., 1.0 (mean, p95).
  A token whose error stays above eps even with every group read (floor >= eps) is
  UNREACHABLE: ``frac_bytes_mean``/``p90``/``k_groups_mean`` cover reachable tokens only (null if
  none), ``unreachable_frac`` says how many were not, and ``frac_bytes_mean_conservative``
  (unreachable counted as 1.0, i.e. "needs every byte and still misses") feeds the verdict.
* BOUND (sound certification, HYPOTHESES 5.6): ``b_g = norm2_g * ||x_g||``; groups sorted by
  ``b_g`` ascending; the certified-skip set is the longest prefix P (largest prefix length; the
  condition is not monotone in general) with ``B = sum_P b_g <= eps * ||y_hat|| / (1 + eps)``,
  ``y_hat = y_ref - sum_P C_g``. Sound: ``||y_ref - y_hat|| <= B`` and ``||y_ref|| >= ||y_hat|| - B``
  give ``||y_ref - y_hat|| / ||y_ref|| <= eps``. The chosen prefix is re-verified with directly
  summed vectors (float64) and shortened if rounding broke the inequality. Recorded: certified
  skip fraction (bytes), realised error ``||sum_P C_g|| / ||y_ref||`` (MUST be <= eps:
  ``soundness_violations`` counts tokens where it is not, relative tolerance 1e-9), the per-group
  check ``||C_g|| <= b_g`` (``bound_violations``, must be 0), certification efficiency =
  certified bytes skipped / oracle bytes skipped at the same eps on the certificate's error basis
  (ratio of sums over the tokens; see the decisions below), tightness ``||C_g|| / b_g``
  (p10/p50/p90 over tokens x groups) and the
  per-token Spearman rank correlation between ``b_g`` and ``||C_g||`` (mean over tokens).
  ``total_err_mean`` is the error a runtime that reads the complement would really see, i.e.
  including the residual quantisation floor; it is NOT a soundness criterion.
* ISO-BYTE control (C5): at byte budgets equal to uniform q2 and q3 full reads and to the
  oracle's mean bytes at eps 0.03, the mean relative error of plain uniform group quantisation
  of ``W_q4`` (``decompose.group_quant``, same block) vs base + oracle (per token, the largest
  oracle prefix that fits the residual budget) vs base + certified (a bound-driven runtime: read
  the groups with the largest ``b_g`` first until the budget is spent). The oracle budget lies
  between integer bit widths; its uniform error is interpolated log-linearly in bytes between
  the bracketing widths (both reported).
* SANITY (C6, down only): ``rel_err(W_down_q4 @ ffn_geglu_t, ffn_out_t)`` median and p95 over
  tokens, pass < 0.03 (llama.cpp quantises activations to Q8 for the matmul); the run aborts
  before any heavy work (exit 1) when a median exceeds 0.1 (orientation/scale bug).

Summary (per base; the headline is the best base): median over the sampled (layer, mat)
cases of the conservative residual fraction at eps 0.03 and of the certification efficiency;
implied bytes/token over the model's layers (60 for the 31B) = n_layers * sum over gate/up/down
of (mean base bytes + mean oracle-selected residual bytes at eps 0.03) + attention 4,770,160,640
+ head 1,156,076,544 (PROJECTED: extrapolated from the sampled layers) vs 17,635,166,448 (Q4_0)
and 8.5e9 (IQ2_XXS); s/token at 850 MB/s (NVMe Gen3, assumed) and tok/s at 10 GB/s in RAM
(assumed); verdict by the pre-registered E7 rule (quoted verbatim in ``RULE_TEXT``), evaluated
in the order written (PROMISING, WEAK, NO_ADVANTAGE, NOT_SUPPORTED), at eps 0.03, for phase
``all`` and phase ``gen``; the final verdict is the worse of the two (control C3).

CLI::

    python -m pigemma.h2.oracle --decomp traces/h2/decomp/ --acts traces/h2/acts/31b/ \\
        --tokens 512 --eps 0.01,0.03,0.10 --out results/h2/oracle.json \\
        --md docs/results/H2_RESULTS.md [--batch 32] [--layers L] [--mats M] [--bases B]
        [--map traces/h2/map31.json | --gguf PATH|ID] [--y-ref gguf|recon]
        [--n-layers N] [--attention-bytes N] [--head-bytes N] [--q4-bytes-per-token N]
        [--iq2-bytes N] [--curve-step 0.05] [--sanity-abort 0.1 | --no-sanity-abort]
        [--results-dir DIR | --no-envelopes] [--no-sha] [--lock] [--lock-min-free-gb 4]
        [--force] [--json OUT|-] [--log-level INFO]
    python -m pigemma.h2.oracle --render-from results/h2/oracle.json --md docs/results/H2_RESULTS.md

Outputs: ``--out`` oracle.json (data_formats 'h2 oracle.json' keys exactly, plus extras), the
markdown (``--md``), and two E7 results envelopes per base (kind projected, variants
``h2-<base>@ram-10GBps`` and ``h2-<base>@nvme-850MBps``, tg.tok_s and read_bytes_per_token) in
``--results-dir`` (default: the directory of ``--out``) so ``python -m pigemma.results summarize
results/h2`` shows E7 (oracle.json itself is not an envelope and is skipped there). stderr:
progress, an ETA after the first matrix; stdout: a summary table (or the JSON with ``--json -``).

Exit codes: 0 ok (pipeline VALID, or validity unknown because no sanity pair was available: a
warning says so); 1 error, sanity abort, or the pipeline is INVALID (sanity median >= 0.03 or a
soundness/bound violation; outputs are still written); 2 usage; 3 missing input (decomp dir or
npz, acts dir or activation file, GGUF, map); 4 refused (bench lock busy, not enough RAM);
130 interrupted.

Decisions the manifest left open (repeated in the agent report)

* Reference of the certification: the manifest's ``y_hat = y_ref - sum_P C_g`` literally (so
  ``realised error = ||sum_P C_g|| / ||y_ref||``, the error caused by skipping). With an exact
  decomposition (``--res-bits 16``) this is the runtime output; with quantised residuals the
  runtime output also carries the floor, reported as ``total_err_mean`` and in the markdown.
  A runtime certificate would additionally need a bound on the floor (HYPOTHESES 5.6).
* Certification efficiency (contract key ``bound[eps].certification_efficiency``): the
  certificate bounds only the error caused by skipping (``||sum_P C_g||``, floor excluded), while
  the oracle's stopping rule includes the residual quantisation floor. With ``--res-bits 4`` the
  floor (~0.028 of ``||y||`` on the real down matrix, HYPOTHESES 5.5) uses up most of eps 3%, so
  the literal ratio would compare two different error notions and could exceed 1. The
  denominator is therefore the oracle evaluated on the certificate's basis: the same
  largest-``||C_g||``-first order, reading until ``||sum of the unread C_g|| <= eps ||y_ref||`` (always
  reachable). The two coincide for exact decompositions (``--res-bits 16``). The literal ratio
  (floor-including oracle, reachable tokens only) is kept as
  ``certification_efficiency_vs_floor_oracle``; the floor itself is judged by the residual
  fraction clause (unreachable tokens count as 1.0) and reported as ``total_err_mean``.
* Arithmetic: ``C_g`` and every norm are float64 (``R`` is converted once per case), so the
  1e-6 rounding-up of ``norm2`` in decompose makes ``||C_g|| <= b_g`` hold with a wide margin.
  ``y_ref``, ``W_base x`` and the uniform-quant outputs are float64 matmuls over row chunks.
* Tokens whose input row is not finite (f16 overflow in the dump) or whose ``y_ref`` is zero
  are dropped and counted (``n_dropped_nonfinite``, ``n_dropped_zero_output``).
* WEAK vs NO_ADVANTAGE precedence follows the written order; every clause's truth value is
  reported so a reader can re-evaluate. Clause (iv) "base+oracle error < uniform-quant error at
  equal bytes" is evaluated at the oracle@0.03 budget (median over cases of uniform/oracle error
  ratio > 1); at the q2 budget a q2 base is the same quantiser as uniform q2, so that budget
  cannot discriminate. NO_ADVANTAGE uses the q3 budget (median ratio <= 1.2).
* Missing matrices (e.g. ``--mats down``): the extrapolation fills an unsampled matrix with the
  mean (base + residual) / Q4_0 byte ratio of the sampled ones and lists it in
  ``mats_extrapolated``. Attention/head/layer constants: the contract values for ``31b-q4``,
  otherwise taken from the model's map/header (``--attention-bytes`` etc. override).
* ``--y-ref recon`` (``W_base + sum R`` instead of the GGUF) is accepted only when asked for; it is
  exact only for ``--res-bits 16`` decompositions and is recorded as ``sampling.y_ref_source``.
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
from pigemma import results as rs
from pigemma import trace as tr
from pigemma.h2 import decompose as dc

log = logging.getLogger("pigemma.h2.oracle")

__all__ = [
    "GENERATOR", "FORMAT_VERSION", "DEFAULT_EPS", "VERDICT_EPS", "DEFAULT_TOKENS", "DEFAULT_BATCH",
    "DEFAULT_CURVE_STEP", "PHASES", "ATTENTION_BYTES_31B", "HEAD_BYTES_31B", "Q4_0_BYTES_PER_TOKEN_31B",
    "N_LAYERS_31B", "IQ2_XXS_BYTES_31B", "NVME_MBPS", "RAM_GBPS", "ASSUMED_LLAMA_EFFICIENCY",
    "SANITY_PASS_MAX", "SANITY_ABORT_MAX", "SOUNDNESS_RTOL", "UNIFORM_BITS", "VERDICTS", "VERDICT_RANK",
    "THRESHOLDS", "RULE_TEXT",
    "OracleError", "SanityAbort", "MissingInputError", "RefusedError", "UsageError",
    "DecompArrays", "TokenStats", "CaseFile",
    "eps_key", "parse_eps", "rankdata", "spearman", "matmul_rows_f64", "decomp_arrays", "recon_rel_err",
    "analyze_case", "uniform_token_errors", "bits_bracket", "uniform_at", "needed_uniform_bits",
    "aggregate_case", "sanity_check", "evaluate_verdict", "base_summary", "summarize",
    "model_constants", "resolve_weight_source", "discover_cases", "run_oracle", "render_markdown",
    "write_envelopes", "main",
]

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
GENERATOR = f"pigemma.h2.oracle/{__version__}"
FORMAT_VERSION = 1
DEFAULT_EPS: tuple[float, ...] = (0.01, 0.03, 0.10)
VERDICT_EPS = 0.03
DEFAULT_TOKENS = 512
DEFAULT_BATCH = 32
DEFAULT_CURVE_STEP = 0.05
PHASES: tuple[str, ...] = ("all", "prefill", "gen")
PHASE_CODES: dict[str, int] = {"prefill": 0, "gen": 1}

# Contract constants (TRACE_FORMAT 13.2, design panel / map31.json): dense 31B Q4_0.
ATTENTION_BYTES_31B = 4_770_160_640
HEAD_BYTES_31B = 1_156_076_544
Q4_0_BYTES_PER_TOKEN_31B = 17_635_166_448
N_LAYERS_31B = 60
IQ2_XXS_BYTES_31B = 8_500_000_000
NVME_MBPS = 850.0             # catalog storage_classes.nvme-gen3.seq_read_mbps (assumed)
RAM_GBPS = 10.0               # catalog devices pi5-*.mem_bw_gbps (assumed)
ASSUMED_LLAMA_EFFICIENCY = 0.65

SANITY_PASS_MAX = 0.03
SANITY_ABORT_MAX = 0.10
SOUNDNESS_RTOL = 1e-9
BOUND_RTOL = 1e-9
UNIFORM_BITS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
ISO_PLAIN_BITS: tuple[int, ...] = (2, 3)
VERDICTS: tuple[str, ...] = ("PROMISING", "WEAK", "NO_ADVANTAGE", "NOT_SUPPORTED")
VERDICT_RANK: dict[str, int] = {"PROMISING": 3, "WEAK": 2, "NO_ADVANTAGE": 1, "NOT_SUPPORTED": 0}
THRESHOLDS: dict[str, float] = {
    "sanity_pass_max": SANITY_PASS_MAX,
    "promising_frac_max": 0.25,
    "promising_eff_min": 0.50,
    "promising_bytes_per_token_max": float(IQ2_XXS_BYTES_31B),
    "weak_frac_max": 0.50,
    "weak_eff_lo": 0.20,
    "weak_eff_hi": 0.50,
    "no_advantage_ratio_max": 1.2,
}
#: The pre-registered E7 thresholds, verbatim from docs/design/manifest.json experiments[E7].
RULE_TEXT = (
    "Pipeline VALID only if sanity median rel_err < 0.03 and soundness_violations == 0. PROMISING only "
    "if at eps 3% the oracle residual fraction <= 25% AND certification efficiency >= 50% AND implied "
    "bytes/token < 8.5 GB AND base+oracle error < uniform-quant error at equal bytes; WEAK if fraction "
    "<= 50% or efficiency 0.2-0.5; NO_ADVANTAGE if uniform q3 error <= 1.2x the oracle error at equal "
    "bytes; NOT_SUPPORTED otherwise. Regardless: report s/token on a Pi (any number > 2 GB/token is "
    "< 0.5 tok/s from NVMe) and that attention + head alone (5.9 GB/token) cap the dense model at "
    "~1.7 tok/s even if it were entirely in RAM."
)
VERDICT_ORDER_NOTE = ("evaluated in the order written (PROMISING, WEAK, NO_ADVANTAGE, NOT_SUPPORTED) at "
                      "eps 0.03 on phase 'all' and on phase 'gen'; the final verdict is the worse of the "
                      "two (control C3)")

_NPZ_RE = re.compile(r"^L(\d+)\.(gate|up|down)\.(.+)\.npz$")
_ROW_CHUNK_ELEMS = 1 << 24    # float64 elements per W row chunk in matmul_rows_f64 (128 MB)


class OracleError(RuntimeError):
    """Runtime failure: inconsistent inputs (npz vs GGUF vs acts), numerical failure."""


class SanityAbort(OracleError):
    """The C6 sanity check failed badly (median rel_err above the abort threshold)."""


class MissingInputError(FileNotFoundError):
    """A required input (decomp dir/npz, acts dir/file, GGUF, map) is missing."""


class RefusedError(RuntimeError):
    """Not enough RAM for the largest case (override with --force)."""


class UsageError(ValueError):
    """Bad command-line arguments (exit 2)."""


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_s(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 5400:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.2f} h"


def _jsonable(v: Any) -> Any:
    """Plain JSON: numpy -> Python, NaN/inf -> None, Path -> POSIX str, tuples -> lists."""
    if isinstance(v, Mapping):
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
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()


def _mean(a: Any) -> float | None:
    x = np.asarray(a, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


def _q(a: Any, q: float) -> float | None:
    x = np.asarray(a, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else None


def _median(values: Iterable[Any]) -> float | None:
    xs = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(xs)) if xs else None


def _rel(path: str | os.PathLike[str] | None) -> str | None:
    return None if path is None else paths.repo_relative(path)


def _host_path(p: str) -> Path:
    """A path recorded inside Docker (/work/...) mapped to the repo when it does not exist as is."""
    s = str(p)
    if s.startswith(paths.CONTAINER_ROOT + "/") and not Path(s).exists():
        return paths.REPO_ROOT / s[len(paths.CONTAINER_ROOT) + 1:]
    return Path(s)


def eps_key(eps: float) -> str:
    """Key of an eps map: two decimals when exact ('0.01', '0.03', '0.10'), else '%g'; 'inf'."""
    e = float(eps)
    if math.isinf(e):
        return "inf"
    if e >= 0 and abs(round(e, 2) - e) < 1e-12:
        return f"{e:.2f}"
    return format(e, "g")


def parse_eps(spec: str | Iterable[Any]) -> tuple[float, ...]:
    """'0.01,0.03,0.10' (or '1%,3%', 'inf') -> sorted unique positive floats. Raises ValueError."""
    parts = str(spec).split(",") if isinstance(spec, str) else list(spec)
    vals: list[float] = []
    for p in parts:
        s = str(p).strip()
        if not s:
            continue
        v = float(s[:-1]) / 100.0 if s.endswith("%") else float(s)
        if math.isnan(v) or v <= 0:
            raise ValueError(f"eps must be > 0, got {s!r}")
        vals.append(v)
    if not vals:
        raise ValueError("empty eps list")
    return tuple(sorted(set(vals)))


def rankdata(a: Any) -> np.ndarray:
    """Ranks 1..n with ties averaged (as scipy.stats.rankdata 'average'), float64."""
    x = np.asarray(a, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.zeros(0)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    bounds = np.flatnonzero(np.concatenate(([True], sx[1:] != sx[:-1], [True])))
    counts = np.diff(bounds)
    avg = (bounds[:-1] + bounds[1:] - 1) / 2.0 + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(avg, counts)
    return ranks


def spearman(a: Any, b: Any) -> float | None:
    """Spearman rank correlation in [-1, 1]; None when either input is constant or n < 2."""
    ra, rb = rankdata(a), rankdata(b)
    if ra.size != rb.size:
        raise ValueError("spearman: inputs differ in length")
    if ra.size < 2:
        return None
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    if den <= 0:
        return None
    return float(min(1.0, max(-1.0, float(ra @ rb) / den)))


def matmul_rows_f64(W: Any, X: Any, *, chunk_elems: int = _ROW_CHUNK_ELEMS) -> np.ndarray:
    """``X @ W.T`` in float64 (``W [n_out, n_in]`` any float dtype, ``X [T, n_in]``) over row chunks."""
    X64 = np.asarray(X, dtype=np.float64)
    if X64.ndim == 1:
        X64 = X64[None, :]
    Wa = np.asarray(W)
    n_out, n_in = Wa.shape
    if X64.shape[1] != n_in:
        raise ValueError(f"X has {X64.shape[1]} columns, W has n_in {n_in}")
    out = np.empty((X64.shape[0], n_out), dtype=np.float64)
    step = max(1, int(chunk_elems) // max(n_in, 1))
    for r0 in range(0, n_out, step):
        r1 = min(n_out, r0 + step)
        out[:, r0:r1] = X64 @ np.asarray(Wa[r0:r1], dtype=np.float64).T
    return out


# --------------------------------------------------------------------------------------------
# Decomposition arrays
# --------------------------------------------------------------------------------------------
@dataclass
class DecompArrays:
    """The arrays of one decomposition the oracle needs (from an npz dict, a path or a Decomp)."""

    W_base: np.ndarray            # [n_out, n_in] (f16 from the npz, f32 from a Decomp)
    R_stack: np.ndarray           # [G, n_out, g_max], zero-padded
    cols: np.ndarray              # int64 [G, 2]
    norm2: np.ndarray             # float64 [G]
    bytes_g: np.ndarray           # int64 [G]
    base_bytes: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_out(self) -> int:
        return int(self.W_base.shape[0])

    @property
    def n_in(self) -> int:
        return int(self.W_base.shape[1])

    @property
    def n_groups(self) -> int:
        return int(self.cols.shape[0])

    @property
    def g_max(self) -> int:
        return int(self.R_stack.shape[2])

    @property
    def total_res(self) -> int:
        return int(np.sum(self.bytes_g))

    @property
    def n_el(self) -> int:
        return self.n_out * self.n_in

    @property
    def block(self) -> int:
        return int(self.meta.get("block") or dc.DEFAULT_BLOCK)

    def reconstruct(self) -> np.ndarray:
        """float32 ``W_base + sum_g R_g`` (the full-read decomposition)."""
        out = np.array(self.W_base, dtype=np.float32, copy=True)
        for i, (a, b) in enumerate(self.cols.tolist()):
            out[:, a:b] += np.asarray(self.R_stack[i, :, : b - a], dtype=np.float32)
        return out


_META_SCALARS = ("layer", "mat", "base", "group_cols", "res_bits", "block", "q4_0_bytes", "n_out", "n_in",
                 "tensor", "gguf", "orig_type", "base_rel_err", "recon_rel_err", "W_normF", "format_version")


def decomp_arrays(d: Any) -> DecompArrays:
    """Normalise a decomposition: DecompArrays, ``decompose.Decomp``, npz dict, or npz path."""
    if isinstance(d, DecompArrays):
        return d
    if isinstance(d, (str, os.PathLike)):
        p = Path(os.fspath(d))
        if not p.is_file():
            raise MissingInputError(f"decomposition npz not found: {p}")
        d = dc.load_npz(p)
    if isinstance(d, dc.Decomp):
        meta = {k: v for k, v in d.meta.items() if not str(k).startswith("_")}
        out = DecompArrays(np.asarray(d.W_base_hat), d.stacked(), d.cols, d.norm2, d.bytes_g,
                           int(d.base_bytes), meta)
    elif isinstance(d, Mapping):
        miss = [k for k in ("W_base_hat", "R_hat", "cols", "norm2", "bytes_g", "base_bytes") if k not in d]
        if miss:
            raise OracleError(f"decomposition npz lacks {miss}")
        meta = {k: d[k] for k in _META_SCALARS if k in d}
        if "meta_json" in d:
            with contextlib.suppress(Exception):
                mj = json.loads(str(d["meta_json"]))
                for k in ("base_bits", "base_rank", "lowrank", "norm2_over_normF", "residual_dtype"):
                    if k in mj:
                        meta[k] = mj[k]
        out = DecompArrays(np.asarray(d["W_base_hat"]), np.asarray(d["R_hat"]),
                           np.asarray(d["cols"], dtype=np.int64).reshape(-1, 2),
                           np.asarray(d["norm2"], dtype=np.float64), np.asarray(d["bytes_g"], dtype=np.int64),
                           int(d["base_bytes"]), meta)
    else:
        raise TypeError(f"decomp_arrays: unsupported type {type(d).__name__}")
    G = out.cols.shape[0]
    if out.W_base.ndim != 2 or out.R_stack.ndim != 3:
        raise OracleError("W_base must be 2-D and R_stack 3-D")
    if out.R_stack.shape[0] != G or out.R_stack.shape[1] != out.n_out:
        raise OracleError(f"R_stack shape {out.R_stack.shape} does not match G={G}, n_out={out.n_out}")
    if out.norm2.shape != (G,) or out.bytes_g.shape != (G,):
        raise OracleError("norm2 / bytes_g must have one entry per group")
    for a, b in out.cols.tolist():
        if not 0 <= a < b <= out.n_in or b - a > out.g_max:
            raise OracleError(f"bad group column range [{a}, {b}) for n_in={out.n_in}, g_max={out.g_max}")
    if not (np.isfinite(out.norm2).all() and (out.norm2 >= 0).all()):
        raise OracleError("norm2 must be finite and >= 0")
    return out


def recon_rel_err(D: DecompArrays, W_ref: Any) -> float:
    """``||W_ref - (W_base + sum R)||_F / ||W_ref||_F`` in float64 over the group columns."""
    W = np.asarray(W_ref)
    if W.shape != (D.n_out, D.n_in):
        raise OracleError(f"W_ref shape {W.shape} != decomposition [{D.n_out}, {D.n_in}]")
    ss_d = ss_w = 0.0
    for i, (a, b) in enumerate(D.cols.tolist()):
        Wg = np.asarray(W[:, a:b], dtype=np.float64)
        diff = Wg - np.asarray(D.W_base[:, a:b], dtype=np.float64) - np.asarray(D.R_stack[i, :, : b - a],
                                                                                dtype=np.float64)
        ss_d += float(np.sum(diff * diff))
        ss_w += float(np.sum(Wg * Wg))
    return math.sqrt(ss_d / ss_w) if ss_w > 0 else 0.0


# --------------------------------------------------------------------------------------------
# Per-token analysis of one case
# --------------------------------------------------------------------------------------------
@dataclass
class TokenStats:
    """Per-token measurements of one (layer, mat, base) case; see :func:`analyze_case`.

    Index ``[T, G+1]`` arrays by prefix length (0 = no group). ``*_ord`` = oracle order (largest
    ``||C_g||`` first, groups READ); ``*_asc`` = bound order (smallest ``b_g`` first, groups SKIPPED).
    """

    eps: tuple[float, ...]
    n_out: int
    n_in: int
    n_groups: int
    block: int
    base_bytes: int
    bytes_g: np.ndarray
    valid: np.ndarray                 # bool [T]: finite input row and ||y_ref|| > 0
    finite: np.ndarray                # bool [T]: finite input row
    phase: np.ndarray | None          # uint8 [T] or None
    y_norm: np.ndarray                # [T]
    err_base: np.ndarray              # [T]
    floor: np.ndarray                 # [T]  ||y_ref - (W_base + sum R) x|| / ||y_ref||
    err_ord: np.ndarray               # [T, G+1] rel error after reading the k largest-||C|| groups
    bytes_ord: np.ndarray             # int64 [T, G+1] residual bytes read in that order
    tot_asc: np.ndarray               # [T, G+1] rel error (incl. floor) when the j smallest-b groups are skipped
    bytes_asc: np.ndarray             # int64 [T, G+1] residual bytes skipped in that order
    k_oracle: np.ndarray              # int64 [T, E] (G for unreachable tokens)
    reach: np.ndarray                 # bool [T, E]
    cert_j: np.ndarray                # int64 [T, E] groups certified-skipped
    cert_skip_bytes: np.ndarray       # int64 [T, E]
    cert_realised: np.ndarray         # [T, E] ||sum_P C|| / ||y_ref|| (must be <= eps)
    cert_total: np.ndarray            # [T, E] ||y_ref - runtime output|| / ||y_ref|| (incl. floor)
    cert_violation: np.ndarray        # bool [T, E]
    bound_violations: np.ndarray      # int64 [T] groups with ||C_g|| > b_g
    tightness: np.ndarray             # float32 [T, G] ||C_g|| / b_g (nan where b_g == 0)
    spearman: np.ndarray              # [T] (nan when undefined)
    skip_err_ord: np.ndarray          # [T, G+1] ||sum of the UNREAD contributions|| / ||y_ref|| (oracle order;
                                      #   the certificate's error basis: residual quantisation floor excluded)
    k_free: np.ndarray                # int64 [T, E] oracle groups read on that basis (always reachable)
    free_skip_bytes: np.ndarray       # int64 [T, E] residual bytes the oracle skips on that basis
    seconds: float = 0.0
    oracle_mask: np.ndarray | None = None   # bool [T, E, G] groups READ by the oracle (details=True)
    cert_mask: np.ndarray | None = None     # bool [T, E, G] groups certified-SKIPPED (details=True)

    @property
    def n_tokens(self) -> int:
        return int(self.valid.shape[0])

    @property
    def total_res(self) -> int:
        return int(np.sum(self.bytes_g))

    @property
    def n_el(self) -> int:
        return self.n_out * self.n_in

    def eps_index(self, eps: float) -> int:
        for i, e in enumerate(self.eps):
            if e == eps or eps_key(e) == eps_key(eps):
                return i
        raise KeyError(f"eps {eps} not analysed (have {self.eps})")

    def mask(self, phase: str = "all") -> np.ndarray:
        """Valid tokens of ``phase`` ('all', 'prefill', 'gen')."""
        if phase == "all":
            return self.valid.copy()
        if phase not in PHASE_CODES:
            raise ValueError(f"unknown phase {phase!r}")
        if self.phase is None:
            return np.zeros_like(self.valid)
        return self.valid & (self.phase == PHASE_CODES[phase])

    def oracle_read_bytes(self, ie: int) -> np.ndarray:
        """Residual bytes the oracle reads per token at eps index ``ie`` (all bytes if unreachable)."""
        k = self.k_oracle[:, ie]
        return np.take_along_axis(self.bytes_ord, k[:, None], axis=1)[:, 0]


def analyze_case(decomp: Any, W_ref: Any, X: Any, *, eps: Sequence[float] = DEFAULT_EPS,
                 batch: int = DEFAULT_BATCH, phase: Any = None, y_ref: np.ndarray | None = None,
                 details: bool = False) -> TokenStats:
    """Oracle selection, certification, tightness and Spearman for every token of ``X``.

    ``decomp``: DecompArrays / decompose.Decomp / npz dict / npz path. ``W_ref``: the reference
    matrix ``[n_out, n_in]`` (``W_q4``). ``X``: ``[T, n_in]`` inputs (any float dtype, e.g. the f16
    memmap rows). ``phase``: uint8 ``[T]`` (0 prefill, 1 gen) or None. ``y_ref``: optional
    precomputed ``X @ W_ref.T`` (float64; rows of non-finite inputs must use zeroed rows).
    ``details=True`` also returns the per-token group sets (``oracle_mask``/``cert_mask``).
    Works in token batches of ``batch``; memory ~ ``8*n_el`` (R in float64) + ``3*batch*G*n_out*8``.
    """
    t_start = time.perf_counter()
    D = decomp_arrays(decomp)
    eps_t = tuple(float(e) for e in eps)
    if not eps_t or any((not e > 0) for e in eps_t):
        raise ValueError(f"eps must be a non-empty list of positive values, got {eps!r}")
    W = np.asarray(W_ref)
    if W.shape != (D.n_out, D.n_in):
        raise OracleError(f"W_ref shape {W.shape} != decomposition [{D.n_out}, {D.n_in}]")
    X32 = np.array(X, dtype=np.float32, copy=True)
    if X32.ndim == 1:
        X32 = X32[None, :]
    if X32.ndim != 2 or X32.shape[1] != D.n_in:
        raise OracleError(f"X shape {X32.shape} does not match n_in={D.n_in}")
    T = X32.shape[0]
    finite = np.isfinite(X32).all(axis=1)
    X32[~finite] = 0.0
    ph = None
    if phase is not None:
        ph = np.asarray(phase, dtype=np.uint8).ravel()
        if ph.shape[0] != T:
            raise ValueError(f"phase has {ph.shape[0]} entries, X has {T} rows")
    batch = max(1, int(batch))
    G, n_out, g_max = D.n_groups, D.n_out, D.g_max
    E = len(eps_t)
    eps_arr = np.array(eps_t, dtype=np.float64)
    r_arr = np.where(np.isinf(eps_arr), 1.0, eps_arr / (1.0 + np.where(np.isinf(eps_arr), 0.0, eps_arr)))
    bytes_g = np.asarray(D.bytes_g, dtype=np.int64)
    total_res = int(bytes_g.sum())
    cols = D.cols.tolist()

    if y_ref is None:
        y_ref = matmul_rows_f64(W, X32)
    else:
        y_ref = np.asarray(y_ref, dtype=np.float64)
        if y_ref.shape != (T, n_out):
            raise ValueError(f"y_ref shape {y_ref.shape} != ({T}, {n_out})")
    y_base = matmul_rows_f64(D.W_base, X32)
    y_norm = np.linalg.norm(y_ref, axis=1)
    valid = finite & np.isfinite(y_norm) & (y_norm > 0)
    safe_yn = np.where(valid, y_norm, 1.0)
    # residual stack as [G, g_max, n_out] float64: C[g] = X_g @ R_g.T comes out row-contiguous
    RT = np.ascontiguousarray(np.asarray(D.R_stack).transpose(0, 2, 1), dtype=np.float64)
    norm2 = np.asarray(D.norm2, dtype=np.float64)

    err_base = np.zeros(T)
    floor = np.zeros(T)
    err_ord = np.zeros((T, G + 1))
    bytes_ord = np.zeros((T, G + 1), dtype=np.int64)
    tot_asc = np.zeros((T, G + 1))
    bytes_asc = np.zeros((T, G + 1), dtype=np.int64)
    k_oracle = np.zeros((T, E), dtype=np.int64)
    reach = np.zeros((T, E), dtype=bool)
    cert_j = np.zeros((T, E), dtype=np.int64)
    cert_skip = np.zeros((T, E), dtype=np.int64)
    cert_real = np.zeros((T, E))
    cert_tot = np.zeros((T, E))
    cert_viol = np.zeros((T, E), dtype=bool)
    bound_viol = np.zeros(T, dtype=np.int64)
    tight = np.full((T, G), np.nan, dtype=np.float32)
    spear = np.full(T, np.nan)
    skip_ord = np.zeros((T, G + 1))
    k_free = np.zeros((T, E), dtype=np.int64)
    free_skip = np.zeros((T, E), dtype=np.int64)
    omask = np.zeros((T, E, G), dtype=bool) if details else None
    cmask = np.zeros((T, E, G), dtype=bool) if details else None

    for t0 in range(0, T, batch):
        t1 = min(T, t0 + batch)
        nb = t1 - t0
        sl = slice(t0, t1)
        Xb = X32[sl].astype(np.float64)
        yr = y_ref[sl]
        e_base = yr - y_base[sl]
        yn = safe_yn[sl]
        # group inputs [G, nb, g_max] (zero-padded short last group) and their norms
        Xg = np.zeros((G, nb, g_max), dtype=np.float64)
        for i, (a, b) in enumerate(cols):
            Xg[i, :, : b - a] = Xb[:, a:b]
        xgn = np.sqrt(np.einsum("gbk,gbk->bg", Xg, Xg))                 # [nb, G]
        bnd = xgn * norm2[None, :]
        C = np.matmul(Xg, RT).transpose(1, 0, 2)                         # [nb, G, n_out] view
        del Xg
        cn = np.sqrt(np.einsum("bgn,bgn->bg", C, C))                     # ||C_g||
        bound_viol[sl] = np.count_nonzero(cn > bnd * (1.0 + BOUND_RTOL), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tight[sl] = np.where(bnd > 0, cn / np.where(bnd > 0, bnd, 1.0), np.nan).astype(np.float32)
        for b in range(nb):
            s = spearman(bnd[b], cn[b])
            spear[t0 + b] = np.nan if s is None else s

        # ---- oracle order: read the largest contributions first --------------------------------
        order = np.argsort(-cn, axis=1, kind="stable")
        S = np.take_along_axis(C, order[:, :, None], axis=1)            # [nb, G, n_out] copy
        np.cumsum(S, axis=1, out=S)                                      # S_k = sum of the k largest
        ctot = S[:, -1, :].copy()                                        # sum of every contribution
        f = e_base - ctot                                                # floor vector
        np.subtract(ctot[:, None, :], S, out=S)                          # U_k = unread contributions
        un = np.sqrt(np.einsum("bgn,bgn->bg", S, S))
        np.add(S, f[:, None, :], out=S)                                  # e_base - S_k = f + U_k
        en = np.sqrt(np.einsum("bgn,bgn->bg", S, S))
        del S
        err_ord[sl, 0] = np.linalg.norm(e_base, axis=1) / yn
        err_ord[sl, 1:] = en / yn[:, None]
        skip_ord[sl, 0] = np.linalg.norm(ctot, axis=1) / yn
        skip_ord[sl, 1:] = un / yn[:, None]
        bytes_ord[sl, 1:] = np.cumsum(bytes_g[order], axis=1)
        err_base[sl] = err_ord[sl, 0]
        fn = np.linalg.norm(f, axis=1)
        floor[sl] = fn / yn
        rows_b = np.arange(nb)
        for ie, e in enumerate(eps_arr):
            ok = err_ord[sl] <= e
            r_ok = ok.any(axis=1)
            k = np.where(r_ok, np.argmax(ok, axis=1), G)
            k_oracle[sl, ie] = k
            reach[sl, ie] = r_ok
            # same oracle order, judged on the certificate's basis (unread contributions only, no
            # floor): always reachable because U_G == 0 exactly
            kf = np.argmax(skip_ord[sl] <= e, axis=1)
            k_free[sl, ie] = kf
            free_skip[sl, ie] = total_res - bytes_ord[sl][rows_b, kf]
            if omask is not None:
                for b in range(nb):
                    omask[t0 + b, ie, order[b, : k[b]]] = True

        # ---- bound order: skip the smallest bounds first ---------------------------------------
        oa = np.argsort(bnd, axis=1, kind="stable")
        A = np.take_along_axis(C, oa[:, :, None], axis=1)
        np.cumsum(A, axis=1, out=A)                                      # A[:, j-1] = sum of the j smallest-b groups
        a2 = np.einsum("bgn,bgn->bg", A, A)
        ya = np.einsum("bgn,bn->bg", A, yr)
        fa = np.einsum("bgn,bn->bg", A, f)
        yn2 = (y_norm[sl] ** 2)[:, None]
        yhat = np.sqrt(np.maximum(np.concatenate([yn2, yn2 - 2.0 * ya + a2], axis=1), 0.0))
        tot = np.sqrt(np.maximum(np.concatenate([(fn ** 2)[:, None], (fn ** 2)[:, None] + 2.0 * fa + a2],
                                                axis=1), 0.0))
        tot_asc[sl] = tot / yn[:, None]
        Bcum = np.concatenate([np.zeros((nb, 1)), np.cumsum(np.take_along_axis(bnd, oa, axis=1), axis=1)], axis=1)
        bytes_asc[sl, 1:] = np.cumsum(bytes_g[oa], axis=1)
        rows = np.arange(nb)
        for ie, (e, r) in enumerate(zip(eps_arr, r_arr)):
            cond = Bcum <= r * yhat                                      # [nb, G+1]; j=0 always holds
            j = G - np.argmax(cond[:, ::-1], axis=1)                     # largest prefix satisfying it
            # direct re-verification with the summed vectors (float64); shorten on rounding failure
            for _ in range(G + 1):
                Asel = np.where((j > 0)[:, None], A[rows, np.maximum(j - 1, 0)], 0.0)
                yh = np.linalg.norm(yr - Asel, axis=1)
                bad = (Bcum[rows, j] > r * yh) & (j > 0)
                if not bad.any():
                    break
                j = np.where(bad, j - 1, j)
            an = np.linalg.norm(Asel, axis=1)
            cert_j[sl, ie] = j
            cert_skip[sl, ie] = bytes_asc[sl][rows, j]
            cert_real[sl, ie] = an / yn
            cert_tot[sl, ie] = np.linalg.norm(f + Asel, axis=1) / yn
            if math.isfinite(e):
                cert_viol[sl, ie] = (an / yn > e * (1.0 + SOUNDNESS_RTOL)) & valid[sl]
            if cmask is not None:
                for b in range(nb):
                    cmask[t0 + b, ie, oa[b, : j[b]]] = True
        del A, C
    bound_viol[~valid] = 0
    return TokenStats(
        eps=eps_t, n_out=n_out, n_in=D.n_in, n_groups=G, block=D.block, base_bytes=int(D.base_bytes),
        bytes_g=bytes_g, valid=valid, finite=finite, phase=ph, y_norm=y_norm, err_base=err_base, floor=floor,
        err_ord=err_ord, bytes_ord=bytes_ord, tot_asc=tot_asc, bytes_asc=bytes_asc, k_oracle=k_oracle,
        reach=reach, cert_j=cert_j, cert_skip_bytes=cert_skip, cert_realised=cert_real, cert_total=cert_tot,
        cert_violation=cert_viol, bound_violations=bound_viol, tightness=tight, spearman=spear,
        skip_err_ord=skip_ord, k_free=k_free, free_skip_bytes=free_skip,
        seconds=time.perf_counter() - t_start, oracle_mask=omask, cert_mask=cmask)


# --------------------------------------------------------------------------------------------
# Iso-byte control: plain uniform group quantisation of the reference matrix
# --------------------------------------------------------------------------------------------
def uniform_token_errors(W_ref: Any, X: Any, bits: int, *, block: int = dc.DEFAULT_BLOCK,
                         y_ref: np.ndarray | None = None) -> np.ndarray:
    """Per-token ``||y_ref - Q_bits(W_ref) x|| / ||y_ref||`` (non-finite input rows zeroed; nan if y=0)."""
    X32 = np.array(X, dtype=np.float32, copy=True)
    if X32.ndim == 1:
        X32 = X32[None, :]
    X32[~np.isfinite(X32).all(axis=1)] = 0.0
    W = np.asarray(W_ref, dtype=np.float32)
    if y_ref is None:
        y_ref = matmul_rows_f64(W, X32)
    Q, _ = dc.group_quant(W, int(bits), int(block))
    yq = matmul_rows_f64(Q, X32)
    del Q
    yn = np.linalg.norm(y_ref, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(yn > 0, np.linalg.norm(y_ref - yq, axis=1) / np.where(yn > 0, yn, 1.0), np.nan)


def bits_bracket(bpe: float, block: int = dc.DEFAULT_BLOCK,
                 bits: Sequence[int] = UNIFORM_BITS) -> tuple[int | None, int | None]:
    """(largest bits with bytes/elem <= bpe, smallest bits with bytes/elem >= bpe)."""
    lo = [b for b in bits if dc.bytes_per_elem(b, block) <= bpe + 1e-12]
    hi = [b for b in bits if dc.bytes_per_elem(b, block) >= bpe - 1e-12]
    return (max(lo) if lo else None), (min(hi) if hi else None)


def uniform_at(bpe: float, errs: Mapping[int, float | None], block: int = dc.DEFAULT_BLOCK) -> dict[str, Any]:
    """Uniform-quant error at ``bpe`` bytes/elem from per-bit mean errors: exact width, or log-linear
    interpolation in bytes between the bracketing widths, or the nearest width outside 2..8 bits."""
    lo, hi = bits_bracket(bpe, block, sorted(errs))
    out: dict[str, Any] = {"uniform_bits_lo": lo, "uniform_bits_hi": hi, "uniform_err_lo": None,
                           "uniform_err_hi": None, "uniform_interpolated": False, "uniform_clamped": False}
    e_lo = errs.get(lo) if lo is not None else None
    e_hi = errs.get(hi) if hi is not None else None
    out["uniform_err_lo"], out["uniform_err_hi"] = e_lo, e_hi
    if lo is not None and lo == hi:
        out["uniform_err"] = e_lo
        return out
    if lo is None or hi is None:
        out["uniform_clamped"] = True
        out["uniform_err"] = e_hi if lo is None else e_lo
        return out
    if e_lo is None or e_hi is None:
        out["uniform_err"] = None
        return out
    x0, x1 = dc.bytes_per_elem(lo, block), dc.bytes_per_elem(hi, block)
    w = (bpe - x0) / (x1 - x0)
    if e_lo > 0 and e_hi > 0:
        val = math.exp((1.0 - w) * math.log(e_lo) + w * math.log(e_hi))
    else:
        val = (1.0 - w) * e_lo + w * e_hi
    out["uniform_err"] = val
    out["uniform_interpolated"] = True
    return out


def _iso_budgets(ts: TokenStats, m: np.ndarray, verdict_eps: float) -> list[tuple[str, float, int | None]]:
    """(label, bytes/elem, exact uniform bits or None) budgets of the iso-byte control."""
    out: list[tuple[str, float, int | None]] = [
        (f"q{b}", dc.bytes_per_elem(b, ts.block), b) for b in ISO_PLAIN_BITS]
    try:
        ie = ts.eps_index(verdict_eps)
    except KeyError:
        return out
    if m.any():
        read = float(ts.oracle_read_bytes(ie)[m].mean())
        out.append((f"oracle@{eps_key(verdict_eps)}", (ts.base_bytes + read) / ts.n_el, None))
    return out


def needed_uniform_bits(ts: TokenStats, *, verdict_eps: float = VERDICT_EPS,
                        phases: Sequence[str] = PHASES) -> set[int]:
    """Uniform bit widths the iso-byte entries of every phase of ``ts`` need."""
    need: set[int] = set(ISO_PLAIN_BITS)
    for ph in phases:
        m = ts.mask(ph)
        for _label, bpe, exact in _iso_budgets(ts, m, verdict_eps):
            if exact is not None:
                need.add(exact)
                continue
            lo, hi = bits_bracket(bpe, ts.block)
            need.update(b for b in (lo, hi) if b is not None)
    return need


# --------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------
def _curve_fracs(step: float) -> list[float]:
    n = max(1, int(round(1.0 / float(step))))
    return [round(i / n, 10) for i in range(n + 1)]


def _prefix_within(cum: np.ndarray, budget: float, total: int) -> np.ndarray:
    """Per row: the largest prefix length whose cumulative bytes are <= budget (index into G+1)."""
    thr = budget + 1e-9 * max(total, 1)
    return np.count_nonzero(cum <= thr, axis=1) - 1


def aggregate_case(ts: TokenStats, phase: str = "all", *, uniform: Mapping[int, np.ndarray] | None = None,
                   curve_step: float = DEFAULT_CURVE_STEP, verdict_eps: float = VERDICT_EPS) -> dict[str, Any] | None:
    """The per_case statistics (data_formats 'h2 oracle.json' + extras) of ``phase``; None if empty."""
    m = ts.mask(phase)
    n = int(m.sum())
    if n == 0:
        return None
    total = ts.total_res
    tot = float(total) if total > 0 else 1.0
    out: dict[str, Any] = {
        "phase": phase, "n_tokens": n,
        "err_base_mean": _mean(ts.err_base[m]), "err_base_p95": _q(ts.err_base[m], 0.95),
        "floor_err_mean": _mean(ts.floor[m]), "floor_err_p95": _q(ts.floor[m], 0.95),
    }
    oracle: dict[str, Any] = {}
    bound: dict[str, Any] = {}
    for ie, e in enumerate(ts.eps):
        key = eps_key(e)
        k = ts.k_oracle[m, ie]
        rch = ts.reach[m, ie]
        read = ts.oracle_read_bytes(ie)[m].astype(np.float64)
        frac = read / tot
        err_at = np.take_along_axis(ts.err_ord[m], k[:, None], axis=1)[:, 0]
        oracle[key] = {
            "frac_bytes_mean": _mean(frac[rch]) if rch.any() else None,
            "frac_bytes_p90": _q(frac[rch], 0.9) if rch.any() else None,
            "k_groups_mean": _mean(k[rch]) if rch.any() else None,
            "frac_bytes_median": _q(frac[rch], 0.5) if rch.any() else None,
            "unreachable_frac": float(1.0 - rch.mean()),
            "n_reachable": int(rch.sum()),
            "frac_bytes_mean_conservative": float(np.where(rch, frac, 1.0).mean()),
            "frac_bytes_p90_conservative": float(np.quantile(np.where(rch, frac, 1.0), 0.9)),
            "read_bytes_mean": float(read.mean()),
            "err_mean": _mean(err_at),
        }
        skip = ts.cert_skip_bytes[m, ie].astype(np.float64)
        osk = tot - read                         # floor-including oracle: unreachable -> skips 0
        den_lit = float(osk[rch].sum())
        fsk = ts.free_skip_bytes[m, ie].astype(np.float64)
        den = float(fsk.sum())
        bound[key] = {
            "certified_frac_bytes_mean": float((skip / tot).mean()),
            "realised_err_mean": _mean(ts.cert_realised[m, ie]),
            "soundness_violations": int(np.count_nonzero(ts.cert_violation[m, ie])),
            # certified bytes skipped / bytes the oracle skips at the same eps on the SAME error
            # basis as the certificate (skipped contributions only, floor excluded); ratio of sums
            "certification_efficiency": (float(skip.sum()) / den) if den > 0 else None,
            "certified_frac_bytes_p90": _q(skip / tot, 0.9),
            "k_groups_mean": _mean(ts.cert_j[m, ie]),
            "realised_err_max": float(np.max(ts.cert_realised[m, ie])),
            "total_err_mean": _mean(ts.cert_total[m, ie]),
            "total_err_over_eps_frac": float(np.mean(ts.cert_total[m, ie] > e)) if math.isfinite(e) else 0.0,
            "oracle_skip_frac_mean": float((fsk / tot).mean()),
            "oracle_skip_frac_basis": "unread contributions only (floor excluded), as the certificate",
            # the literal manifest ratio against the floor-including oracle (tokens it reaches); can
            # exceed 1 when the residual quantisation floor uses up most of eps
            "certification_efficiency_vs_floor_oracle": (float(skip[rch].sum()) / den_lit) if den_lit > 0 else None,
            "oracle_skipped_frac_mean": float((osk / tot).mean()),
        }
    out["oracle"] = oracle
    curve = []
    bo = ts.bytes_ord[m]
    eo = ts.err_ord[m]
    for fr in _curve_fracs(curve_step):
        kk = _prefix_within(bo, fr * total, total)
        ee = np.take_along_axis(eo, kk[:, None], axis=1)[:, 0]
        curve.append([fr, _mean(ee), _q(ee, 0.95)])
    out["curve"] = curve
    out["bound"] = bound
    tt = ts.tightness[m].astype(np.float64).ravel()
    tt = tt[np.isfinite(tt)]
    out["tightness"] = {"p10": _q(tt, 0.10), "p50": _q(tt, 0.50), "p90": _q(tt, 0.90),
                        "n": int(tt.size), "max": float(tt.max()) if tt.size else None}
    sp = ts.spearman[m]
    out["spearman_bound_vs_actual"] = _mean(sp)
    out["spearman_median"] = _q(sp, 0.5)
    out["spearman_n_tokens"] = int(np.isfinite(sp).sum())
    out["bound_violations"] = int(ts.bound_violations[m].sum())
    iso = []
    for label, bpe, exact in _iso_budgets(ts, m, verdict_eps):
        rb = bpe * ts.n_el - ts.base_bytes
        ent: dict[str, Any] = {"label": label, "bytes_per_elem": bpe,
                               "residual_budget_bytes": max(0.0, rb) if rb > -0.5 else None}
        if uniform is not None:
            means = {b: _mean(np.asarray(a)[m]) for b, a in uniform.items()}
            if exact is not None:
                ent.update({"uniform_err": means.get(exact), "uniform_bits": exact})
            else:
                ent.update(uniform_at(bpe, means, ts.block))
        else:
            ent["uniform_err"] = None
        if rb < -0.5:
            ent["base_plus_oracle_err"] = None
            ent["base_plus_certified_err"] = None
            ent["note"] = "the base alone exceeds this budget"
        else:
            rb = max(0.0, rb)
            kk = _prefix_within(bo, rb, total)
            ent["base_plus_oracle_err"] = _mean(np.take_along_axis(eo, kk[:, None], axis=1)[:, 0])
            # bound-driven runtime: read the largest b_g first == skip the smallest-b prefix j
            need_skip = total - rb - 1e-9 * max(total, 1)
            ba = ts.bytes_asc[m]
            jj = np.argmax(ba >= need_skip, axis=1)
            ent["base_plus_certified_err"] = _mean(np.take_along_axis(ts.tot_asc[m], jj[:, None], axis=1)[:, 0])
        iso.append(ent)
    out["iso_byte"] = iso
    return out


# --------------------------------------------------------------------------------------------
# Sanity (C6)
# --------------------------------------------------------------------------------------------
def sanity_check(W_down: Any, geglu: Any, ffn_out: Any, *, layer: int | None = None) -> dict[str, Any]:
    """``rel_err_t = ||W_down @ geglu_t - ffn_out_t|| / ||ffn_out_t||``: median, p95, pass (< 0.03)."""
    G_ = np.array(geglu, dtype=np.float32, copy=True)
    O_ = np.asarray(ffn_out, dtype=np.float64)
    if G_.shape[0] != O_.shape[0]:
        raise OracleError(f"sanity: ffn_geglu has {G_.shape[0]} rows, ffn_out {O_.shape[0]}")
    fin = np.isfinite(G_).all(axis=1) & np.isfinite(O_).all(axis=1)
    G_[~fin] = 0.0
    Y = matmul_rows_f64(W_down, G_)
    on = np.linalg.norm(O_, axis=1)
    ok = fin & (on > 0)
    rel = np.linalg.norm(Y[ok] - O_[ok], axis=1) / on[ok]
    med = _q(rel, 0.5)
    return {"layer": layer, "rel_err_median": med, "rel_err_p95": _q(rel, 0.95),
            "pass": (med is not None and med < SANITY_PASS_MAX), "n_tokens": int(ok.sum()),
            "n_dropped": int((~ok).sum()), "rel_err_mean": _mean(rel),
            "rel_err_max": float(rel.max()) if rel.size else None}


# --------------------------------------------------------------------------------------------
# Verdict and summary
# --------------------------------------------------------------------------------------------
def evaluate_verdict(frac: float | None, eff: float | None, implied_total: float | None,
                     ratio_oracle_budget: float | None, ratio_q3: float | None,
                     *, iq2_bytes: float = IQ2_XXS_BYTES_31B) -> tuple[str, dict[str, Any]]:
    """Apply the pre-registered E7 rule (order as written). Ratios are uniform_err / base+oracle_err."""
    th = THRESHOLDS
    c_frac = frac is not None and frac <= th["promising_frac_max"]
    c_eff = eff is not None and eff >= th["promising_eff_min"]
    c_bytes = implied_total is not None and implied_total < iq2_bytes
    c_beat = ratio_oracle_budget is not None and ratio_oracle_budget > 1.0
    w_frac = frac is not None and frac <= th["weak_frac_max"]
    w_eff = eff is not None and th["weak_eff_lo"] <= eff <= th["weak_eff_hi"]
    no_adv = ratio_q3 is not None and ratio_q3 <= th["no_advantage_ratio_max"]
    clauses = {
        "promising_frac_le_0_25": {"value": frac, "ok": c_frac},
        "promising_eff_ge_0_5": {"value": eff, "ok": c_eff},
        "promising_bytes_lt_iq2": {"value": implied_total, "limit": iq2_bytes, "ok": c_bytes},
        "promising_oracle_beats_uniform": {"value": ratio_oracle_budget, "ok": c_beat,
                                           "meaning": "median uniform_err / base+oracle_err at the oracle@0.03 budget > 1"},
        "weak_frac_le_0_5": {"value": frac, "ok": w_frac},
        "weak_eff_0_2_to_0_5": {"value": eff, "ok": w_eff},
        "no_advantage_q3_le_1_2x": {"value": ratio_q3, "ok": no_adv,
                                    "meaning": "median uniform_q3_err / base+oracle_err at the q3 budget <= 1.2"},
    }
    if c_frac and c_eff and c_bytes and c_beat:
        v = "PROMISING"
    elif w_frac or w_eff:
        v = "WEAK"
    elif no_adv:
        v = "NO_ADVANTAGE"
    else:
        v = "NOT_SUPPORTED"
    return v, clauses


def _worse(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if VERDICT_RANK[a] <= VERDICT_RANK[b] else b


def _iso_entry(case: Mapping[str, Any], label: str) -> Mapping[str, Any] | None:
    return next((e for e in case.get("iso_byte", []) if e.get("label") == label), None)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    if den <= 0:
        return math.inf if num > 0 else None
    return num / den


def base_summary(per_case: Sequence[Mapping[str, Any]], base: str, phase: str, consts: Mapping[str, Any],
                 *, verdict_eps: float = VERDICT_EPS, iq2_bytes: float = IQ2_XXS_BYTES_31B) -> dict[str, Any] | None:
    """Summary numbers + verdict of one base on one phase (None when no case has tokens)."""
    cases = [c for c in per_case if c["base"] == base and c["phase"] == phase]
    if not cases:
        return None
    key = eps_key(verdict_eps)
    frac = _median(c["oracle"][key]["frac_bytes_mean_conservative"] for c in cases if key in c["oracle"])
    eff = _median(c["bound"][key]["certification_efficiency"] for c in cases if key in c["bound"])
    olabel = f"oracle@{key}"
    r_beat = _median(_ratio((_iso_entry(c, olabel) or {}).get("uniform_err"),
                            (_iso_entry(c, olabel) or {}).get("base_plus_oracle_err")) for c in cases)
    r_q3 = _median(_ratio((_iso_entry(c, "q3") or {}).get("uniform_err"),
                          (_iso_entry(c, "q3") or {}).get("base_plus_oracle_err")) for c in cases)
    # per-matrix means over the sampled layers
    per_mat: dict[str, dict[str, float]] = {}
    for mat in dc.FFN_MATS:
        cm = [c for c in cases if c["mat"] == mat and key in c["oracle"]]
        if cm:
            per_mat[mat] = {
                "base_bytes": float(np.mean([c["base_bytes"] for c in cm])),
                "residual_bytes": float(np.mean([c["oracle"][key]["read_bytes_mean"] for c in cm])),
                "q4_0_bytes": float(np.mean([c["bytes_q4_0"] for c in cm])),
                "n_layers_sampled": len(cm),
            }
    missing = [m for m in dc.FFN_MATS if m not in per_mat]
    if per_mat and missing:
        ratio_b = float(np.mean([v["base_bytes"] / v["q4_0_bytes"] for v in per_mat.values()]))
        ratio_r = float(np.mean([v["residual_bytes"] / v["q4_0_bytes"] for v in per_mat.values()]))
        q4m = float(np.mean([v["q4_0_bytes"] for v in per_mat.values()]))
        for mat in missing:
            per_mat[mat] = {"base_bytes": ratio_b * q4m, "residual_bytes": ratio_r * q4m, "q4_0_bytes": q4m,
                            "n_layers_sampled": 0, "extrapolated": True}
    n_layers = int(consts["n_layers"])
    att, head = int(consts["attention_bytes"]), int(consts["head_bytes"])
    ffn_base = n_layers * sum(v["base_bytes"] for v in per_mat.values())
    ffn_res = n_layers * sum(v["residual_bytes"] for v in per_mat.values())
    total = ffn_base + ffn_res + att + head
    s_nvme = total / (NVME_MBPS * 1e6)
    tok_ram = RAM_GBPS * 1e9 / total if total > 0 else None
    resident = att + head + ffn_base
    s_mixed = resident / (RAM_GBPS * 1e9) + ffn_res / (NVME_MBPS * 1e6)
    verdict, clauses = evaluate_verdict(frac, eff, total, r_beat, r_q3, iq2_bytes=iq2_bytes)
    return {
        "base": base, "phase": phase, "n_cases": len(cases),
        "n_tokens": int(max(c["n_tokens"] for c in cases)),
        "frac_0_03_median": frac, "cert_eff_median": eff,
        "frac_0_03_reachable_median": _median(c["oracle"][key]["frac_bytes_mean"] for c in cases
                                              if key in c["oracle"]),
        "unreachable_0_03_median": _median(c["oracle"][key]["unreachable_frac"] for c in cases if key in c["oracle"]),
        "err_base_median": _median(c["err_base_mean"] for c in cases),
        "floor_err_median": _median(c["floor_err_mean"] for c in cases),
        "uniform_over_oracle_at_oracle_budget": r_beat, "uniform_q3_over_oracle_at_q3_budget": r_q3,
        "soundness_violations": int(sum(c["bound"][k]["soundness_violations"] for c in cases for k in c["bound"])),
        "bound_violations": int(sum(c.get("bound_violations", 0) for c in cases)),
        "implied_bytes_per_token": {
            "attention": att, "head": head, "ffn_base": ffn_base, "ffn_residual_at_0_03": ffn_res,
            "total": total, "kind": "projected", "note": "extrapolated from sampled layers",
            "n_layers": n_layers, "per_mat": per_mat, "mats_extrapolated": missing,
            "sampled_layers": sorted({int(c["layer"]) for c in cases}),
        },
        "s_per_token_nvme_850mbps": s_nvme,
        "tok_s_in_ram_10gbps": tok_ram,
        "tok_s_in_ram_10gbps_at_efficiency_0_65": (tok_ram * ASSUMED_LLAMA_EFFICIENCY) if tok_ram else None,
        "resident_base_streamed_residual": {
            "resident_bytes": resident, "streamed_bytes": ffn_res, "s_per_token": s_mixed,
            "tok_s": (1.0 / s_mixed) if s_mixed > 0 else None, "kind": "projected",
            "note": "base + attention + head held in RAM at 10 GB/s, the selected residual streamed at "
                    "850 MB/s, no overlap; the oracle selection itself is not runtime-decidable",
        },
        "verdict": verdict, "clauses": clauses,
    }


def summarize(per_case: Sequence[Mapping[str, Any]], sanity: Sequence[Mapping[str, Any]], consts: Mapping[str, Any],
              *, bases: Sequence[str], verdict_eps: float = VERDICT_EPS,
              iq2_bytes: float = IQ2_XXS_BYTES_31B, q4_bytes: float | None = None) -> dict[str, Any]:
    """The oracle.json ``summary`` block (contract keys from the best base + per-base detail)."""
    per_base: dict[str, Any] = {}
    for base in bases:
        s_all = base_summary(per_case, base, "all", consts, verdict_eps=verdict_eps, iq2_bytes=iq2_bytes)
        if s_all is None:
            continue
        s_gen = base_summary(per_case, base, "gen", consts, verdict_eps=verdict_eps, iq2_bytes=iq2_bytes)
        final = _worse(s_all["verdict"], s_gen["verdict"] if s_gen else None)
        per_base[base] = {"verdict": final,
                          "verdict_by_phase": {"all": s_all["verdict"], "gen": s_gen["verdict"] if s_gen else None},
                          "all": s_all, "gen": s_gen}
    all_cases = [c for c in per_case if c["phase"] == "all"]
    sound_total = int(sum(c["bound"][k]["soundness_violations"] for c in all_cases for k in c["bound"]))
    bound_total = int(sum(c.get("bound_violations", 0) for c in all_cases))
    reasons: list[str] = []
    valid: bool | None = True
    if not sanity or all(s.get("rel_err_median") is None for s in sanity):
        valid = None
        reasons.append("no sanity pair (L<il>.ffn_geglu + L<il>.ffn_out + the down matrix) for the sampled "
                       "layers: validity UNKNOWN")
    for s in sanity:
        if s.get("rel_err_median") is not None and not s.get("pass"):
            valid = False
            reasons.append(f"sanity layer {s['layer']}: median rel_err {s['rel_err_median']:.4f} >= "
                           f"{SANITY_PASS_MAX}")
    if sound_total:
        valid = False
        reasons.append(f"{sound_total} certification soundness violation(s) (must be 0)")
    if bound_total:
        valid = False
        reasons.append(f"{bound_total} per-group bound violation(s) ||C_g|| > norm2_g*||x_g|| (must be 0)")
    if valid:
        reasons.append("sanity median rel_err < 0.03 on every sampled layer and 0 soundness violations")
    best = None
    if per_base:
        best = max(per_base, key=lambda b: (VERDICT_RANK[per_base[b]["verdict"]],
                                            -per_base[b]["all"]["implied_bytes_per_token"]["total"]))
    top = per_base[best]["all"] if best else None
    att, head = int(consts["attention_bytes"]), int(consts["head_bytes"])
    ah = att + head
    implied = dict(top["implied_bytes_per_token"]) if top else {
        "attention": att, "head": head, "ffn_base": None, "ffn_residual_at_0_03": None, "total": None,
        "kind": "projected", "note": "extrapolated from sampled layers"}
    implied["base"] = best
    implied["q4_0_total"] = q4_bytes
    implied["iq2_xxs_total"] = iq2_bytes
    always = [
        (f"s/token on a Pi from NVMe Gen3 at {NVME_MBPS:.0f} MB/s (assumed): "
         f"{top['s_per_token_nvme_850mbps']:.1f} s for base {best} "
         f"({implied['total'] / 1e9:.2f} GB/token); any traffic above 2 GB/token is below 0.5 tok/s from NVMe.")
        if top else "no case analysed",
        (f"attention + head alone = {ah / 1e9:.2f} GB/token: even with zero FFN bytes the dense model is capped "
         f"at {RAM_GBPS * 1e9 / ah:.2f} tok/s at {RAM_GBPS:.0f} GB/s in RAM "
         f"({ASSUMED_LLAMA_EFFICIENCY * RAM_GBPS * 1e9 / ah:.2f} tok/s at the assumed "
         f"{ASSUMED_LLAMA_EFFICIENCY} efficiency)."),
    ]
    if q4_bytes:
        always.append(f"Q4_0 streamed at {NVME_MBPS:.0f} MB/s: {q4_bytes / (NVME_MBPS * 1e6):.1f} s/token; "
                      f"in RAM at {RAM_GBPS:.0f} GB/s: {RAM_GBPS * 1e9 / q4_bytes:.2f} tok/s.")
    return {
        "frac_0_03_median": top["frac_0_03_median"] if top else None,
        "cert_eff_median": top["cert_eff_median"] if top else None,
        "soundness_violations_total": sound_total,
        "implied_bytes_per_token": implied,
        "s_per_token_nvme_850mbps": top["s_per_token_nvme_850mbps"] if top else None,
        "tok_s_in_ram_10gbps": top["tok_s_in_ram_10gbps"] if top else None,
        "verdict": per_base[best]["verdict"] if best else "NOT_SUPPORTED",
        "rule": RULE_TEXT,
        "bound_violations_total": bound_total,
        "pipeline_valid": valid,
        "pipeline": "VALID" if valid else ("INVALID" if valid is False else "UNKNOWN"),
        "pipeline_reasons": reasons,
        "verdict_void": valid is False,
        "best_base": best,
        "verdict_order": VERDICT_ORDER_NOTE,
        "thresholds": dict(THRESHOLDS),
        "attention_plus_head_bytes": ah,
        "attention_plus_head_tok_s_cap_10gbps": RAM_GBPS * 1e9 / ah,
        "always_reported": always,
        "per_base": per_base,
    }


# --------------------------------------------------------------------------------------------
# Inputs: cases, weights, model constants
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CaseFile:
    """One decomposition npz on disk."""

    layer: int
    mat: str
    base: str
    path: Path


def discover_cases(decomp_dir: str | os.PathLike[str], *, layers: Iterable[int] | None = None,
                   mats: Iterable[str] | None = None, bases: Iterable[str] | None = None) -> list[CaseFile]:
    """``L<il>.<mat>.<base>.npz`` files of ``decomp_dir`` (optionally filtered), sorted."""
    d = Path(os.fspath(decomp_dir))
    if not d.is_dir():
        raise MissingInputError(f"decomposition directory not found: {d}")
    lay = None if layers is None else {int(x) for x in layers}
    mt = None if mats is None else set(mats)
    bs = None if bases is None else {dc.parse_base(b).name for b in bases}
    out = []
    for p in d.glob("L*.npz"):
        m = _NPZ_RE.match(p.name)
        if not m:
            continue
        il, mat, base = int(m.group(1)), m.group(2), m.group(3)
        try:
            base = dc.parse_base(base).name
        except ValueError:
            log.warning("%s: unknown base %r, skipped", p.name, base)
            continue
        if (lay is not None and il not in lay) or (mt is not None and mat not in mt) or (bs is not None and base not in bs):
            continue
        out.append(CaseFile(il, mat, base, p))
    mat_order = {m: i for i, m in enumerate(dc.FFN_MATS)}
    return sorted(out, key=lambda c: (c.layer, mat_order[c.mat], c.base))


def _npz_scalars(p: Path) -> dict[str, Any]:
    """Scalar members of a decomposition npz without loading the big arrays."""
    out: dict[str, Any] = {}
    with np.load(p, allow_pickle=False) as z:
        for k in _META_SCALARS + ("base_bytes", "n_groups", "total_residual_bytes"):
            if k in z.files:
                a = z[k]
                if a.ndim == 0:
                    out[k] = a.item()
    return out


def resolve_weight_source(*, gguf: str | None = None, map_path: str | None = None, npz_gguf: str | None = None,
                          acts_meta: Mapping[str, Any] | None = None) -> tuple[Any, Path, gm.GgufMap | None, str]:
    """Where to read ``W_q4`` from: ``(src for decompose.load_matrix, gguf path, map or None, how)``.

    Order: ``--map`` (its GGUF, ``--gguf`` overriding the location); ``--gguf``; the npz ``gguf``
    basename (registry id, then as a path); the acts ``meta.json`` ``gguf_path`` (/work mapped to
    the repo) and ``model``. Raises MissingInputError listing what was tried.
    """
    if map_path:
        mp = Path(map_path)
        if not mp.is_file():
            raise MissingInputError(f"map not found: {mp}")
        gmap = gm.GgufMap.load(mp)
        try:
            p = dc.resolve_gguf(gmap, gguf)
        except FileNotFoundError as e:
            raise MissingInputError(str(e)) from None
        return gmap, p, gmap, "map"
    cands: list[tuple[str, Path]] = []
    if gguf:
        cands.append(("--gguf", paths.model_path(gguf)))
    else:
        for label, val in (("npz gguf", npz_gguf),
                           ("acts meta gguf_path", (acts_meta or {}).get("gguf_path")),
                           ("acts meta model", (acts_meta or {}).get("model"))):
            if not val:
                continue
            mid = paths.model_id_for(str(val))
            if mid:
                cands.append((f"{label} (registry {mid})", paths.model_path(mid)))
            cands.append((label, _host_path(str(val)) if "/" in str(val) or "\\" in str(val)
                          else paths.model_path(str(val))))
    for how, c in cands:
        if c.is_file():
            return c, c, None, how
    tried = ", ".join(f"{h}: {c}" for h, c in cands) or "nothing (no --gguf/--map and no gguf name in the inputs)"
    raise MissingInputError("the Q4_0 GGUF (needed for y_ref = W_q4 @ x) was not found; tried " + tried +
                            ". Pass --gguf PATH|ID or --map map.json (or --y-ref recon for res_bits=16 decompositions).")


def model_constants(gguf_path: Path | None = None, gmap: gm.GgufMap | None = None, *, gguf_name: str | None = None,
                    n_layers: int | None = None, attention_bytes: int | None = None, head_bytes: int | None = None,
                    q4_bytes_per_token: int | None = None) -> dict[str, Any]:
    """n_layers, attention/head bytes and the Q4_0 bytes/token of the model (31B contract values
    for 31b-q4; else read from the map/header; CLI overrides win)."""
    name = gguf_name or (gguf_path.name if gguf_path is not None else (gmap.gguf_basename if gmap else None))
    out: dict[str, Any] = {"n_layers": N_LAYERS_31B, "attention_bytes": ATTENTION_BYTES_31B,
                           "head_bytes": HEAD_BYTES_31B, "q4_bytes_per_token": Q4_0_BYTES_PER_TOKEN_31B,
                           "source": "exact: byte counts of the 31b-q4 GGUF (TRACE_FORMAT 13.2, design panel, "
                                     "traces/h2/map31.json)", "notes": []}
    mid = paths.model_id_for(name) if name else None
    if mid != "31b-q4":
        m = gmap
        if m is None and gguf_path is not None and gguf_path.is_file():
            try:
                m = gm.build_map_from_header(gm.read_gguf_header(gguf_path, "stdlib"))
            except Exception as e:  # noqa: BLE001 - fall back to the 31B constants
                out["notes"].append(f"could not read the header of {gguf_path.name} ({type(e).__name__}: {e}); "
                                    "31B constants used")
                out["source"] = f"assumed: 31b-q4 constants (the header of {gguf_path.name} could not be read)"
        if m is not None:
            s = m.summary
            out.update({"n_layers": int(m.n_layer), "attention_bytes": int(s["bytes_per_kind"].get("attn", 0)),
                        "head_bytes": int(s.get("head_bytes", 0)),
                        "q4_bytes_per_token": int(s.get("tensor_bytes_total", 0)),
                        "source": f"exact: map/header of {m.gguf_basename} (not the 31B: contract constants not applicable)"})
        elif not out["notes"]:
            out["notes"].append(f"model {name!r} not recognised and no header available: 31B constants used")
            out["source"] = f"assumed: 31b-q4 constants (model {name!r} not recognised, no header available)"
    for k, v in (("n_layers", n_layers), ("attention_bytes", attention_bytes), ("head_bytes", head_bytes),
                 ("q4_bytes_per_token", q4_bytes_per_token)):
        if v is not None:
            out[k] = int(v)
            out["notes"].append(f"{k} overridden on the command line: {int(v)}")
    return out


def _model_block(gguf_path: Path | None, gguf_name: str | None, with_sha: bool) -> dict[str, Any]:
    if gguf_path is not None and gguf_path.is_file():
        b = paths.model_identity(gguf_path, with_sha=with_sha)
        b["gguf"] = paths.repo_relative(gguf_path)
        return b
    name = gguf_name or "unknown"
    mid = paths.model_id_for(name)
    return {"id": mid or Path(name).stem, "file": name, "quant": paths.guess_quant(name),
            "bytes": int(paths.MODELS[mid]["bytes"]) if mid else None, "sha_prefix": None, "gguf": None}


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------
def _estimate_case_bytes(n_out: int, n_in: int, G: int, batch: int, T: int) -> int:
    n_el = n_out * n_in
    # W_ref f32 + W_base f16 + R f16 (npz) + R f64 + uniform Q f32 + 3 batch arrays [batch, G, n_out] f64
    return int(n_el * (4 + 2 + 2 + 8 + 4) + 3 * batch * G * n_out * 8 + T * max(n_in, n_out) * 24)


def run_oracle(decomp_dir: str | os.PathLike[str], acts_dir: str | os.PathLike[str], *,
               tokens: int = DEFAULT_TOKENS, eps: Sequence[float] = DEFAULT_EPS, batch: int = DEFAULT_BATCH,
               layers: Iterable[int] | None = None, mats: Iterable[str] | None = None,
               bases: Iterable[str] | None = None, map_path: str | None = None, gguf: str | None = None,
               y_ref_mode: str = "gguf", n_layers: int | None = None, attention_bytes: int | None = None,
               head_bytes: int | None = None, q4_bytes_per_token: int | None = None,
               iq2_bytes: float = IQ2_XXS_BYTES_31B, curve_step: float = DEFAULT_CURVE_STEP,
               sanity_abort: float | None = SANITY_ABORT_MAX, check_memory: bool = True,
               with_sha: bool = True, argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run E7 end to end and return the oracle.json document (nothing is written here)."""
    t_start = time.perf_counter()
    notes: list[str] = []
    eps_t = tuple(sorted(set(float(e) for e in eps)))
    if VERDICT_EPS not in eps_t:
        eps_t = tuple(sorted(set(eps_t) | {VERDICT_EPS}))
        notes.append("eps 0.03 added: the verdict is defined at 3%")
    if y_ref_mode not in ("gguf", "recon"):
        raise ValueError(f"y_ref_mode must be gguf|recon, got {y_ref_mode!r}")
    cases = discover_cases(decomp_dir, layers=layers, mats=mats, bases=bases)
    if not cases:
        raise MissingInputError(f"no L<il>.<mat>.<base>.npz in {decomp_dir} (after filters); run "
                                "python -m pigemma.h2.decompose first")
    try:
        acts = tr.read_acts_dir(acts_dir)
    except FileNotFoundError as e:
        raise MissingInputError(str(e)) from None
    T = min(int(tokens), int(acts.n_tokens))
    if T <= 0:
        raise MissingInputError(f"{acts_dir}: the activation dump holds 0 tokens")
    if T < int(tokens):
        notes.append(f"--tokens {tokens} requested, the dump holds {acts.n_tokens}: using {T}")
    phase = None
    if acts.phase is not None and acts.phase.shape[0] >= T:
        phase = np.asarray(acts.phase[:T], dtype=np.uint8)
    else:
        notes.append("no phase.u8 (or shorter than the tokens used): only phase 'all' is reported")
    available = set(acts.available())
    need = sorted({(c.layer, dc.FFN_INPUT_KIND[c.mat]) for c in cases})
    miss = [f"L{il}.{k}.npy" for il, k in need if (il, k) not in available]
    if miss:
        raise MissingInputError(f"{acts_dir}: activation files missing for the sampled cases: {', '.join(miss)}")

    scal = {c: _npz_scalars(c.path) for c in cases}
    for c, s in scal.items():
        for k, v in (("layer", c.layer), ("mat", c.mat), ("base", c.base)):
            if k in s and s[k] != v:
                raise OracleError(f"{c.path.name}: npz {k}={s[k]!r} does not match its file name ({v!r})")
        kind = dc.FFN_INPUT_KIND[c.mat]
        ne0 = acts.kinds.get(kind)
        if ne0 is not None and "n_in" in s and int(s["n_in"]) != int(ne0):
            raise OracleError(f"{c.path.name}: n_in {s['n_in']} != acts {kind} width {ne0} "
                              "(decomposition and activations come from different models?)")
    npz_gguf = next((str(s["gguf"]) for s in scal.values() if s.get("gguf")), None)

    src: Any = None
    gguf_path: Path | None = None
    gmap: gm.GgufMap | None = None
    how = "recon"
    if y_ref_mode == "gguf":
        src, gguf_path, gmap, how = resolve_weight_source(gguf=gguf, map_path=map_path, npz_gguf=npz_gguf,
                                                          acts_meta=acts.meta)
        if npz_gguf and Path(npz_gguf).name != gguf_path.name:
            log.warning("npz files name %s but y_ref reads %s; the per-matrix consistency check decides",
                        npz_gguf, gguf_path.name)
        log.info("W_q4 from %s (%s)", gguf_path, how)
    else:
        notes.append("y_ref = W_base + sum R (--y-ref recon): exact only for res_bits=16 decompositions")
        if gguf or map_path:
            with contextlib.suppress(Exception):
                _s, gguf_path, gmap, _h = resolve_weight_source(gguf=gguf, map_path=map_path)
    consts = model_constants(gguf_path, gmap, gguf_name=npz_gguf, n_layers=n_layers,
                             attention_bytes=attention_bytes, head_bytes=head_bytes,
                             q4_bytes_per_token=q4_bytes_per_token)
    notes.extend(consts.get("notes", []))

    if check_memory:
        est = max(_estimate_case_bytes(int(s.get("n_out", 0)), int(s.get("n_in", 0)),
                                       int(s.get("n_groups") or 1), batch, T) for s in scal.values())
        _total, avail = paths.ram_bytes()
        if avail is not None and avail < est * 1.2:
            raise RefusedError(f"the largest case needs ~{est / 1e9:.1f} GB of RAM, only {avail / 1e9:.1f} GB is "
                               "available; lower --batch, free memory, or pass --force")

    def load_w(layer: int, mat: str) -> np.ndarray:
        name = dc.ffn_names(layer)[mat]
        return dc.load_matrix(src, name, gguf=gguf if gmap is not None else None)

    # ---- sanity (C6) before any heavy work ----------------------------------------------------
    sanity: list[dict[str, Any]] = []
    for il in sorted({c.layer for c in cases}):
        if (il, "ffn_geglu") not in available or (il, "ffn_out") not in available:
            log.warning("sanity: layer %d has no ffn_geglu/ffn_out pair in %s; skipped", il, acts_dir)
            continue
        if y_ref_mode == "gguf":
            Wd = load_w(il, "down")
        else:
            down = [c for c in cases if c.layer == il and c.mat == "down"]
            if not down:
                log.warning("sanity: --y-ref recon and no down decomposition for layer %d; skipped", il)
                continue
            Wd = decomp_arrays(down[0].path).reconstruct()
        g = acts.get(il, "ffn_geglu")
        o = acts.get(il, "ffn_out")
        ent = sanity_check(Wd, g[:T], o[:T], layer=il)
        del g, o, Wd
        sanity.append(ent)
        sys.stderr.write(f"pigemma.h2.oracle: sanity L{il}: median rel_err {ent['rel_err_median']:.4g}, p95 "
                         f"{ent['rel_err_p95']:.4g} over {ent['n_tokens']} tokens -> "
                         f"{'pass' if ent['pass'] else 'FAIL'} (< {SANITY_PASS_MAX})\n")
    gc.collect()
    if sanity_abort is not None and sanity_abort > 0:
        bad = [s for s in sanity if s["rel_err_median"] is not None and s["rel_err_median"] > sanity_abort]
        if bad:
            raise SanityAbort(
                "sanity check failed: median rel_err(W_down_q4 @ ffn_geglu, ffn_out) = " +
                ", ".join(f"L{s['layer']}: {s['rel_err_median']:.3g}" for s in bad) +
                f" > {sanity_abort} (abort threshold). This is an orientation or scale bug, or the activations "
                "and the GGUF come from different models; nothing was written.")

    # ---- main loop --------------------------------------------------------------------------
    groups: dict[tuple[int, str], list[CaseFile]] = {}
    for c in cases:
        groups.setdefault((c.layer, c.mat), []).append(c)
    per_case: list[dict[str, Any]] = []
    n_total, n_done = len(cases), 0
    t_loop = time.perf_counter()
    dropped_nf: dict[str, int] = {}
    for (il, mat), cs in groups.items():
        kind = dc.FFN_INPUT_KIND[mat]
        mm = acts.get(il, kind)
        X = np.array(mm[:T], dtype=np.float32)
        del mm
        nf = int((~np.isfinite(X).all(axis=1)).sum())
        if nf:
            dropped_nf[f"L{il}.{kind}"] = nf
            log.warning("L%d.%s: %d token row(s) with non-finite values dropped", il, kind, nf)
        Xc = X.copy()
        Xc[~np.isfinite(Xc).all(axis=1)] = 0.0
        W_shared = load_w(il, mat) if y_ref_mode == "gguf" else None
        y_shared = matmul_rows_f64(W_shared, Xc) if W_shared is not None else None
        done: list[tuple[CaseFile, dict[str, Any], TokenStats, np.ndarray, np.ndarray, float]] = []
        for c in cs:
            t0 = time.perf_counter()
            D = decomp_arrays(dc.load_npz(c.path))
            if W_shared is not None:
                if W_shared.shape != (D.n_out, D.n_in):
                    raise OracleError(f"{c.path.name}: decomposition [{D.n_out}, {D.n_in}] != GGUF matrix "
                                      f"{list(W_shared.shape)}")
                rr = recon_rel_err(D, W_shared)
                stored = D.meta.get("recon_rel_err")
                if stored is not None and math.isfinite(float(stored)):
                    if abs(rr - float(stored)) > max(1e-5, 1e-3 * float(stored)):
                        raise OracleError(
                            f"{c.path.name}: recon_rel_err vs the GGUF matrix is {rr:.6g} but the npz recorded "
                            f"{float(stored):.6g}: the decomposition was not made from this GGUF")
                W_c, y_c = W_shared, y_shared
            else:
                W_c = D.reconstruct()
                rr = 0.0
                y_c = matmul_rows_f64(W_c, Xc)
            ts = analyze_case(D, W_c, X, eps=eps_t, batch=batch, phase=phase, y_ref=y_c)
            info = {"base_bytes": int(D.base_bytes), "q4_0_bytes": int(D.meta.get("q4_0_bytes") or
                                                                       dc.q4_0_nbytes(D.n_out, D.n_in)),
                    "group_cols": D.meta.get("group_cols"), "res_bits": D.meta.get("res_bits"),
                    "block": D.block, "n_groups": D.n_groups, "n_out": D.n_out, "n_in": D.n_in,
                    "base_rel_err": D.meta.get("base_rel_err"), "recon_rel_err_weights": rr,
                    "tensor": D.meta.get("tensor") or dc.ffn_names(il)[mat]}
            del D
            done.append((c, info, ts, W_c if y_ref_mode == "recon" else np.empty(0), y_c if y_ref_mode == "recon"
                         else np.empty(0), time.perf_counter() - t0))
            n_done += 1
            ie = ts.eps_index(VERDICT_EPS)
            m_all = ts.mask("all")
            fr = float(np.where(ts.reach[m_all, ie], ts.oracle_read_bytes(ie)[m_all] / max(ts.total_res, 1), 1.0).mean()) \
                if m_all.any() else float("nan")
            sys.stderr.write(f"pigemma.h2.oracle: [{n_done}/{n_total}] L{il}.{mat}.{c.base}: err_base "
                             f"{_f(_mean(ts.err_base[m_all]), '.4f')}, floor {_f(_mean(ts.floor[m_all]), '.4f')}, "
                             f"oracle frac@3% {fr:.3f} (unreachable counted as 1), {ts.seconds:.1f} s\n")
            if n_done == 1 and n_total > 1:
                el = time.perf_counter() - t_loop
                sys.stderr.write(f"pigemma.h2.oracle: ETA ~{_fmt_s(el * (n_total - 1))} for {n_total - 1} remaining "
                                 f"case(s) (after the first matrix: {el:.1f} s/case, excluding the iso-byte quantisation)\n")
            sys.stderr.flush()
            gc.collect()
        # iso-byte: uniform quantisation of the reference matrix, once per (block, bits) for the
        # matrix; per case in recon mode, where every case has its own reference
        t_u = time.perf_counter()
        ucache: dict[tuple[int, int], np.ndarray] = {}
        for c, info, ts, W_r, y_r, secs in done:
            if y_ref_mode == "recon":
                ucache = {}
                W_use, y_use = W_r, y_r
                need_bits = needed_uniform_bits(ts)
            else:
                W_use, y_use = W_shared, y_shared
                need_bits = set().union(*(needed_uniform_bits(d[2]) for d in done if d[2].block == ts.block))
            for bb in sorted(need_bits):
                if (ts.block, bb) not in ucache:
                    ucache[(ts.block, bb)] = uniform_token_errors(W_use, Xc, bb, block=ts.block, y_ref=y_use)
            uniform = {bb: ucache[(ts.block, bb)] for bb in sorted(need_bits)}
            for ph in PHASES:
                agg = aggregate_case(ts, ph, uniform=uniform, curve_step=curve_step)
                if agg is None:
                    continue
                entry = {"layer": c.layer, "mat": c.mat, "base": c.base, "phase": agg.pop("phase"),
                         "n_tokens": agg.pop("n_tokens"), "bytes_q4_0": info["q4_0_bytes"],
                         "base_bytes": info["base_bytes"], "residual_bytes_total": ts.total_res}
                entry.update(agg)
                entry["seconds"] = round(secs, 3)
                entry.update({"n_groups": info["n_groups"], "group_cols": info["group_cols"],
                              "res_bits": info["res_bits"], "block": info["block"], "n_out": info["n_out"],
                              "n_in": info["n_in"], "input_kind": kind, "tensor": info["tensor"],
                              "base_bytes_per_elem": info["base_bytes"] / (info["n_out"] * info["n_in"]),
                              "residual_bytes_per_elem": ts.total_res / (info["n_out"] * info["n_in"]),
                              "base_rel_err_weights": info["base_rel_err"],
                              "recon_rel_err_weights": info["recon_rel_err_weights"],
                              "n_dropped_nonfinite": int((~ts.finite).sum()),
                              "n_dropped_zero_output": int((ts.finite & ~ts.valid).sum())})
                per_case.append(entry)
        log.info("L%d.%s: iso-byte uniform quantisation in %.1f s", il, mat, time.perf_counter() - t_u)
        del X, Xc, W_shared, y_shared, done, ucache
        gc.collect()

    base_names = list(dict.fromkeys(c.base for c in cases))
    summary = summarize(per_case, sanity, consts, bases=base_names, iq2_bytes=iq2_bytes,
                        q4_bytes=consts["q4_bytes_per_token"])
    gc_ = sorted({s.get("group_cols") for s in scal.values() if s.get("group_cols") is not None})
    rb_ = sorted({s.get("res_bits") for s in scal.values() if s.get("res_bits") is not None})
    n_pref = int(np.count_nonzero(phase == 0)) if phase is not None else None
    n_gen = int(np.count_nonzero(phase == 1)) if phase is not None else None
    assumptions = {
        "nvme_seq_read_mbps": rs.assumption(NVME_MBPS, "MB/s", "assumed: catalog storage_classes.nvme-gen3."
                                            "seq_read_mbps (published Pi 5 NVMe Gen3 x1 784-894 MB/s)"),
        "ram_bw_gbps": rs.assumption(RAM_GBPS, "GB/s", "assumed: catalog devices pi5-*.mem_bw_gbps (~10 GB/s "
                                     "effective in llama.cpp decode, LPDDR4X-4267)"),
        "llama_efficiency": rs.assumption(1.0, None, "assumed: 1.0 = bandwidth-only upper bound; 0.65 (catalog, "
                                          "assumed) also shown as tok_s_in_ram_10gbps_at_efficiency_0_65"),
        "attention_bytes": rs.assumption(int(consts["attention_bytes"]), "B", consts["source"]),
        "head_bytes": rs.assumption(int(consts["head_bytes"]), "B", consts["source"]),
        "n_layers": rs.assumption(int(consts["n_layers"]), "layers", consts["source"]),
        "q4_0_bytes_per_token": rs.assumption(int(consts["q4_bytes_per_token"]), "B", consts["source"]),
        "iq2_xxs_bytes": rs.assumption(float(iq2_bytes), "B", "threshold: manifest E7, the 31B IQ2_XXS file (~8.5 GB)"),
    }
    doc: dict[str, Any] = {
        "format_version": FORMAT_VERSION, "generator": GENERATOR, "created_at": _utc_now(), "kind": "measured",
        "experiment_id": "E7",
        "model": _model_block(gguf_path, npz_gguf, with_sha),
        "acts_dir": _rel(acts_dir), "decomp_dir": _rel(decomp_dir),
        "sampling": {
            "layers": sorted({c.layer for c in cases}), "mats": [m for m in dc.FFN_MATS if any(c.mat == m for c in cases)],
            "bases": base_names, "group_cols": gc_[0] if len(gc_) == 1 else gc_,
            "res_bits": rb_[0] if len(rb_) == 1 else rb_, "n_tokens": T, "eps": list(eps_t),
            "tokens_requested": int(tokens), "n_tokens_available": int(acts.n_tokens), "batch": int(batch),
            "phases": {"prefill": n_pref, "gen": n_gen}, "y_ref_source": "gguf" if y_ref_mode == "gguf" else "recon",
            "weights_from": None if gguf_path is None else _rel(gguf_path), "weights_how": how,
            "curve_step": curve_step,
        },
        "sanity": sanity,
        "per_case": per_case,
        "summary": summary,
        "assumptions": assumptions,
        "constants": {k: v for k, v in consts.items() if k != "notes"},
        "inputs": {
            "decomp_files": [_rel(c.path) for c in cases],
            "acts_meta": {k: acts.meta.get(k) for k in ("model", "gguf_path", "n_tokens", "layers", "kinds", "chunks",
                                                         "llama_build", "llama_commit", "n_ctx", "n_ubatch", "flags",
                                                         "f16_nonfinite", "corpus_id", "created_at")
                          if k in acts.meta},
            "map": _rel(map_path) if map_path else None,
            "dropped_nonfinite_rows": dropped_nf,
        },
        "host": paths.host_fingerprint(),
        "argv": list(argv) if argv is not None else None,
        "wall_s": round(time.perf_counter() - t_start, 2),
        "notes": notes,
    }
    return _jsonable(doc)


# --------------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------------
def _f(x: Any, fmt: str = ".3f") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return format(x, fmt)


def _pct(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100.0 * float(x):.{nd}f}%"


def _gb(x: Any) -> str:
    return "n/a" if x is None else f"{float(x) / 1e9:.2f}"


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    def cell(v: Any) -> str:
        s = str(v).replace("|", "/").replace("\n", " ").strip()
        return s or " "
    out = ["| " + " | ".join(cell(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(v) for v in r) + " |")
    return out


def _eps_label(key: str) -> str:
    try:
        v = float(key)
    except ValueError:
        return key
    if math.isinf(v):
        return "inf"
    return f"{100 * v:g}%"


def _frac_cell(o: Mapping[str, Any] | None) -> str:
    if not o:
        return "n/a"
    un = o.get("unreachable_frac") or 0.0
    if o.get("frac_bytes_mean") is None:
        return "unreachable (floor)"
    s = f"{_f(o['frac_bytes_mean'])} / {_f(o.get('frac_bytes_p90'))}"
    if un > 0:
        s += f" (unreachable {_pct(un, 0)})"
    return s


def render_markdown(doc: Mapping[str, Any], *, title: str = "H2 results: dense staged residuals (E7)",
                    source: str | None = None) -> str:
    """Render docs/results/H2_RESULTS.md from an oracle.json document."""
    s = doc.get("summary", {})
    samp = doc.get("sampling", {})
    eps_keys = [eps_key(e) for e in samp.get("eps", [])]
    vk = eps_key(VERDICT_EPS)
    cases_all = [c for c in doc.get("per_case", []) if c.get("phase") == "all"]
    model = doc.get("model") or {}
    L: list[str] = [f"# {title}", ""]
    L.append(f"Generated by `{doc.get('generator', GENERATOR)}` on {doc.get('created_at', '?')}"
             + (f" from `{source}`" if source else "") + ". Model `" + str(model.get("id")) + "` (`"
             + str(model.get("file")) + "`).")
    L.append("")
    L.append("Labels: **MEASURED** = computed from captured activations and the real weights on this host; "
             "**PROJECTED** = extrapolated from the sampled layers to the whole model; **ASSUMED** = a constant "
             "with the source given. Fractions are of the residual bytes; errors are relative L2 errors of the "
             "FFN matrix output against `y_ref = W_q4 x`.")
    L.append("")
    # ---- verdict --------------------------------------------------------------------------
    L += ["## Verdict", ""]
    L.append(f"**Verdict: {s.get('verdict')}** (best base `{s.get('best_base')}`; {s.get('verdict_order', '')}).")
    L.append("")
    pv = s.get("pipeline", "UNKNOWN")
    L.append(f"**Pipeline {pv}**: " + "; ".join(s.get("pipeline_reasons", [])) + "."
             + (" The verdict is VOID until the pipeline is valid." if s.get("verdict_void") else ""))
    L.append("")
    L.append("Pre-registered rule (quoted verbatim from the manifest, E7):")
    L.append("")
    L.append("> " + str(s.get("rule", RULE_TEXT)))
    L.append("")
    rows = []
    for base, pb in (s.get("per_base") or {}).items():
        for ph in ("all", "gen"):
            b = pb.get(ph)
            if not b:
                continue
            cl = b["clauses"]
            imp = b["implied_bytes_per_token"]["total"]

            def yn(ok: bool) -> str:
                return "yes" if ok else "no"
            rows.append([
                f"`{base}`", ph, b["n_tokens"],
                f"{_f(b['frac_0_03_median'])} ({yn(cl['promising_frac_le_0_25']['ok'])})",
                f"{_f(b['cert_eff_median'])} ({yn(cl['promising_eff_ge_0_5']['ok'])})",
                f"{_gb(imp)} ({yn(cl['promising_bytes_lt_iq2']['ok'])})",
                f"{_f(b['uniform_over_oracle_at_oracle_budget'], '.2f')} ({yn(cl['promising_oracle_beats_uniform']['ok'])})",
                yn(cl["weak_frac_le_0_5"]["ok"] or cl["weak_eff_0_2_to_0_5"]["ok"]),
                f"{_f(b['uniform_q3_over_oracle_at_q3_budget'], '.2f')} ({yn(cl['no_advantage_q3_le_1_2x']['ok'])})",
                f"**{b['verdict']}**", pb.get("verdict") if ph == "all" else ""])
    L.append("Clause check at eps 3% (medians over the sampled layer x matrix cases; the fraction counts "
             "unreachable tokens as 1.0):")
    L.append("")
    L += _table(["base", "phase", "tokens", "residual fraction (<= 0.25)", "cert. efficiency (>= 0.50)",
                 "implied GB/token (< 8.5)", "uniform/oracle error at oracle bytes (> 1)",
                 "WEAK: fraction <= 0.50 or efficiency 0.2-0.5", "uniform q3/oracle error at q3 bytes (NO_ADVANTAGE if <= 1.2)",
                 "verdict", "final (worse of all, gen)"], rows)
    L.append("")
    L.append("Always reported:")
    L.append("")
    for a in s.get("always_reported", []):
        L.append(f"- {a}")
    L.append("")
    # ---- inputs ---------------------------------------------------------------------------
    L += ["## Inputs and sampling", ""]
    ph = samp.get("phases") or {}
    rows = [
        ["activations", f"`{doc.get('acts_dir')}`"], ["decompositions", f"`{doc.get('decomp_dir')}`"],
        ["weights (y_ref = W_q4 x)", f"`{samp.get('weights_from')}` ({samp.get('y_ref_source')}, {samp.get('weights_how')})"],
        ["layers", ", ".join(str(x) for x in samp.get("layers", []))], ["matrices", ", ".join(samp.get("mats", []))],
        ["bases", ", ".join(samp.get("bases", []))], ["group columns", samp.get("group_cols")],
        ["residual bits", samp.get("res_bits")], ["tokens", f"{samp.get('n_tokens')} (prefill {ph.get('prefill')}, "
                                                            f"gen {ph.get('gen')}; dump holds {samp.get('n_tokens_available')})"],
        ["eps", ", ".join(_eps_label(k) for k in eps_keys)], ["token batch", samp.get("batch")],
        ["host", (doc.get("host") or {}).get("hostname")], ["wall time", f"{doc.get('wall_s')} s"],
    ]
    L += _table(["item", "value"], rows)
    L.append("")
    # ---- sanity ---------------------------------------------------------------------------
    L += ["## Sanity (C6): W_down_q4 @ ffn_geglu vs captured ffn_out (MEASURED)", ""]
    if doc.get("sanity"):
        L += _table(["layer", "tokens", "rel_err median", "rel_err p95", "pass (< 0.03)"],
                    [[x["layer"], x["n_tokens"], _f(x["rel_err_median"], ".4f"), _f(x["rel_err_p95"], ".4f"),
                      "yes" if x["pass"] else "**no**"] for x in doc["sanity"]])
    else:
        L.append("_No sanity pair available (L<il>.ffn_geglu + L<il>.ffn_out); pipeline validity is unknown._")
    L.append("")
    # ---- oracle ---------------------------------------------------------------------------
    L += ["## Oracle residual byte fraction (MEASURED, phase all)", ""]
    L.append("Cells: mean / p90 over the tokens that reach eps; 'unreachable' = tokens whose error stays above eps "
             "even with every residual group read (the residual quantisation floor).")
    L.append("")
    hdr = ["layer", "matrix", "base", "tokens", "err_base mean", "floor mean"] + \
        [f"fraction @{_eps_label(k)}" for k in eps_keys] + [f"groups @{_eps_label(vk)}", "groups total"]
    rows = []
    for c in cases_all:
        rows.append([c["layer"], c["mat"], c["base"], c["n_tokens"], _f(c["err_base_mean"], ".4f"),
                     _f(c.get("floor_err_mean"), ".4f")] + [_frac_cell(c["oracle"].get(k)) for k in eps_keys] +
                    [_f((c["oracle"].get(vk) or {}).get("k_groups_mean"), ".1f"), c.get("n_groups")])
    L += _table(hdr, rows)
    L.append("")
    # ---- phases ---------------------------------------------------------------------------
    L += ["## Teacher-forced vs generated tokens (C3, MEASURED)", ""]
    by = {(c["layer"], c["mat"], c["base"], c["phase"]): c for c in doc.get("per_case", [])}
    rows = []
    for c in cases_all:
        p0 = by.get((c["layer"], c["mat"], c["base"], "prefill"))
        p1 = by.get((c["layer"], c["mat"], c["base"], "gen"))
        rows.append([c["layer"], c["mat"], c["base"], p0["n_tokens"] if p0 else 0, p1["n_tokens"] if p1 else 0,
                     _frac_cell(p0["oracle"].get(vk)) if p0 else "n/a", _frac_cell(p1["oracle"].get(vk)) if p1 else "n/a",
                     _f(p0["err_base_mean"], ".4f") if p0 else "n/a", _f(p1["err_base_mean"], ".4f") if p1 else "n/a"])
    L += _table(["layer", "matrix", "base", "prefill tokens", "gen tokens", f"fraction @{_eps_label(vk)} prefill",
                 f"fraction @{_eps_label(vk)} gen", "err_base prefill", "err_base gen"], rows)
    L.append("")
    # ---- bound ----------------------------------------------------------------------------
    L += ["## Norm-bound certification (MEASURED, phase all)", ""]
    L.append("Certified skip = the longest prefix of groups by ascending `b_g = norm2_g * ||x_g||` with "
             "`sum b_g <= eps * ||y_hat|| / (1 + eps)`, `y_hat = y_ref - sum C_g` (sound by the triangle inequality). "
             "Realised error = the error caused by the skip (must be <= eps); total error adds the residual "
             "quantisation floor (what a runtime would see). Efficiency = certified bytes skipped / bytes the "
             "oracle skips at the same eps on the same basis (largest-`||C_g||`-first reading until the unread "
             "contributions are <= eps; floor excluded, as in the certificate). The ratio against the "
             "floor-including oracle is in the last column (it can exceed 1 when the floor uses up eps).")
    L.append("")
    rows = []
    for c in cases_all:
        for k in eps_keys:
            b = c["bound"].get(k)
            if not b:
                continue
            rows.append([c["layer"], c["mat"], c["base"], _eps_label(k), _f(b["certified_frac_bytes_mean"]),
                         _f(b.get("oracle_skip_frac_mean")),
                         _f(b["realised_err_mean"], ".4f"), _f(b.get("total_err_mean"), ".4f"),
                         b["soundness_violations"], _f(b["certification_efficiency"]),
                         _f(b.get("certification_efficiency_vs_floor_oracle"))])
    L += _table(["layer", "matrix", "base", "eps", "certified skip fraction", "oracle skip fraction (same basis)",
                 "realised err mean", "total err mean (incl. floor)", "soundness violations",
                 "certification efficiency", "efficiency vs floor-including oracle"], rows)
    L.append("")
    L += ["### Bound tightness", ""]
    L.append("Tightness = `||C_g|| / b_g` over tokens x groups (1 = the bound is exact); Spearman = per-token rank "
             "correlation between `b_g` and `||C_g||`, mean over tokens.")
    L.append("")
    L += _table(["layer", "matrix", "base", "tightness p10 / p50 / p90", "Spearman(b_g, ‖C_g‖)",
                 "bound violations"],
                [[c["layer"], c["mat"], c["base"],
                  f"{_f(c['tightness']['p10'])} / {_f(c['tightness']['p50'])} / {_f(c['tightness']['p90'])}",
                  _f(c["spearman_bound_vs_actual"]), c.get("bound_violations", 0)] for c in cases_all])
    L.append("")
    # ---- iso-byte -------------------------------------------------------------------------
    L += ["## Iso-byte comparison (C5, MEASURED)", ""]
    L.append("At equal bytes per weight: plain uniform group quantisation of `W_q4` vs the base plus the "
             "oracle's residual groups vs the base plus the groups with the largest bounds (a certificate-driven "
             "runtime). The oracle budget is interpolated between integer uniform widths (log-linear).")
    L.append("")
    rows = []
    for c in cases_all:
        for e in c.get("iso_byte", []):
            ub = e.get("uniform_bits")
            ul = f"q{ub}" if ub is not None else (f"q{e.get('uniform_bits_lo')}-q{e.get('uniform_bits_hi')} interp."
                                                   if e.get("uniform_interpolated") else "clamped")
            rows.append([c["layer"], c["mat"], c["base"], e["label"], _f(e["bytes_per_elem"], ".4f"),
                         f"{_f(e.get('uniform_err'), '.4f')} ({ul})", _f(e.get("base_plus_oracle_err"), ".4f"),
                         _f(e.get("base_plus_certified_err"), ".4f")])
    L += _table(["layer", "matrix", "base", "budget", "B/weight", "uniform err", "base+oracle err",
                 "base+bound-ordered err"], rows)
    L.append("")
    # ---- curve ----------------------------------------------------------------------------
    L += ["## Error vs residual bytes read (oracle order, MEASURED, mean over tokens)", ""]
    show = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for c in cases_all:
        cm = {round(float(p[0]), 6): p[1] for p in c.get("curve", [])}
        rows.append([c["layer"], c["mat"], c["base"]] + [_f(cm.get(round(x, 6)), ".4f") for x in show])
    L += _table(["layer", "matrix", "base"] + [f"{int(round(100 * x))}%" for x in show], rows)
    L.append("")
    # ---- extrapolation --------------------------------------------------------------------
    L += ["## Extrapolation to the whole model (PROJECTED)", ""]
    q4 = (doc.get("constants") or {}).get("q4_bytes_per_token")
    iq2 = (s.get("implied_bytes_per_token") or {}).get("iq2_xxs_total")
    rows = []
    for base, pb in (s.get("per_base") or {}).items():
        b = pb.get("all")
        if not b:
            continue
        imp = b["implied_bytes_per_token"]
        rs_ = b["resident_base_streamed_residual"]
        rows.append([f"`{base}`", imp["n_layers"], _gb(imp["ffn_base"]), _gb(imp["ffn_residual_at_0_03"]),
                     _gb(imp["attention"]), _gb(imp["head"]), f"**{_gb(imp['total'])}**",
                     _f(imp["total"] / q4, ".2f") + "x" if q4 else "n/a",
                     _f(imp["total"] / iq2, ".2f") + "x" if iq2 else "n/a",
                     _f(b["s_per_token_nvme_850mbps"], ".1f"), _f(b["tok_s_in_ram_10gbps"], ".2f"),
                     _f(rs_["s_per_token"], ".2f"),
                     ", ".join(imp.get("mats_extrapolated") or []) or "-"])
    L += _table(["base", "layers", "FFN base GB", "FFN residual @3% GB", "attention GB", "head GB", "total GB/token",
                 "vs Q4_0", "vs IQ2_XXS 8.5 GB", "s/token @850 MB/s", "tok/s @10 GB/s",
                 "s/token, base resident + residual streamed", "matrices extrapolated"], rows)
    L.append("")
    L.append(f"Q4_0 reference: {_gb(q4)} GB/token. The residual bytes are the oracle's (not runtime-decidable), "
             "so these totals are a lower bound for any real selector.")
    L.append("")
    L.append("Constants and assumptions (label from the source: EXACT = byte counts read from the model file, "
             "ASSUMED = a modelling assumption, THRESHOLD = pre-registered):")
    L.append("")
    for k, a in (doc.get("assumptions") or {}).items():
        src = str(a.get("source") or "")
        head = src.split(":", 1)[0].strip().lower()
        label = {"exact": "EXACT", "assumed": "ASSUMED", "threshold": "THRESHOLD"}.get(head, "ASSUMED")
        L.append(f"- {label} `{k}` = {a.get('value')} {a.get('unit') or ''}: {src}")
    L.append("")
    # ---- notes ----------------------------------------------------------------------------
    notes = list(doc.get("notes") or [])
    if notes:
        L += ["## Notes", ""]
        L += [f"- {n}" for n in notes]
        L.append("")
    return "\n".join(L).rstrip() + "\n"


# --------------------------------------------------------------------------------------------
# Results envelopes (E7, kind projected)
# --------------------------------------------------------------------------------------------
def write_envelopes(doc: Mapping[str, Any], results_dir: str | os.PathLike[str], *,
                    oracle_json: str | os.PathLike[str] | None = None) -> list[Path]:
    """Two E7 envelopes per base (RAM 10 GB/s and NVMe 850 MB/s projections) for results summarize."""
    s = doc.get("summary") or {}
    model = dict(doc.get("model") or {})
    model.pop("gguf", None)
    for k in rs.MODEL_KEYS:
        model.setdefault(k, None)
    meta = (doc.get("inputs") or {}).get("acts_meta") or {}
    llama_build = meta.get("llama_build") if isinstance(meta.get("llama_build"), int) else paths.LLAMA_BUILD
    llama_commit = meta.get("llama_commit") if isinstance(meta.get("llama_commit"), str) else paths.LLAMA_COMMIT
    samp = doc.get("sampling") or {}
    inputs = [str(x) for x in (oracle_json, doc.get("decomp_dir"), doc.get("acts_dir")) if x]
    written: list[Path] = []
    for base, pb in (s.get("per_base") or {}).items():
        b = pb.get("all")
        if not b:
            continue
        total = b["implied_bytes_per_token"]["total"]
        common_assump = {k: v for k, v in (doc.get("assumptions") or {}).items()
                         if k in ("attention_bytes", "head_bytes", "n_layers", "llama_efficiency")}
        for scen, tok_s, key in (("ram-10GBps", b["tok_s_in_ram_10gbps"], "ram_bw_gbps"),
                                 ("nvme-850MBps", (1.0 / b["s_per_token_nvme_850mbps"])
                                  if b["s_per_token_nvme_850mbps"] else None, "nvme_seq_read_mbps")):
            assump = dict(common_assump)
            if key in (doc.get("assumptions") or {}):
                assump[key] = doc["assumptions"][key]
            fields = dict(
                experiment="E7", kind="projected", llama_build=llama_build, llama_commit=llama_commit, model=model,
                variant=f"h2-{base}@{scen}",
                flags=["--base", base, "--group", str(samp.get("group_cols")), "--res-bits", str(samp.get("res_bits")),
                       "--eps", "0.03", "--tokens", str(samp.get("n_tokens"))],
                threads=None, n_ctx=meta.get("n_ctx") if isinstance(meta.get("n_ctx"), int) else None,
                assumptions=assump, inputs=inputs,
                tg={"tok_s": tok_s, "sd": None, "n": None}, read_bytes_per_token=total,
                notes=[f"PROJECTED from {len(b['implied_bytes_per_token']['sampled_layers'])} sampled layer(s), "
                       f"phase all, oracle selection at eps 3% (an upper bound on any runtime selector)",
                       f"E7 verdict for base {base}: {pb.get('verdict')} (pipeline {s.get('pipeline')})"],
                extra={"verdict": pb.get("verdict"), "verdict_by_phase": pb.get("verdict_by_phase"),
                       "frac_0_03_median": b["frac_0_03_median"], "cert_eff_median": b["cert_eff_median"],
                       "pipeline_valid": s.get("pipeline_valid"), "base": base, "scenario": scen,
                       "implied_bytes_per_token": b["implied_bytes_per_token"]},
            )
            written.append(rs.write_result(Path(os.fspath(results_dir)), **fields))
    return written


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
                raise ValueError(f"{what}: bad range {part!r}")
            out.extend(range(a, b + 1))
        elif part.isdigit():
            out.append(int(part))
        else:
            raise ValueError(f"{what}: {part!r} is not an int or a range a-b")
    if not out:
        raise ValueError(f"{what}: empty list")
    return list(dict.fromkeys(out))


def _parse_bytes(s: str) -> int:
    v = float(str(s).replace("_", ""))
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError(f"bad byte count {s!r}")
    return int(round(v))


def _positive_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
    return v


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m pigemma.h2.oracle",
        description=("H2 step 2 (E7): oracle residual selection, sound norm-bound certification, iso-byte "
                     "control and the implied bytes/token of the dense model, on real FFN-input activations."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Example:\n  python -m pigemma.h2.oracle --decomp traces/h2/decomp/ --acts traces/h2/acts/31b/ "
                "--tokens 512 --eps 0.01,0.03,0.10 --out results/h2/oracle.json --md docs/results/H2_RESULTS.md\n\n"
                "Exit codes: 0 ok (pipeline VALID or UNKNOWN), 1 error / sanity abort / pipeline INVALID, 2 usage, "
                "3 missing input, 4 refused (bench lock, RAM), 130 interrupted."))
    ap.add_argument("--decomp", default=None, help="directory of L<il>.<mat>.<base>.npz (pigemma.h2.decompose)")
    ap.add_argument("--acts", default=None, help="activation dump directory (pigemma-probe --mode act-dump)")
    ap.add_argument("--tokens", type=_positive_int, default=DEFAULT_TOKENS,
                    help=f"use the first N tokens of the dump (default {DEFAULT_TOKENS})")
    ap.add_argument("--eps", default="0.01,0.03,0.10",
                    help="relative output-error targets (default 0.01,0.03,0.10; 0.03 is always included)")
    ap.add_argument("--out", default="results/h2/oracle.json", help="oracle.json path (default results/h2/oracle.json)")
    ap.add_argument("--md", default=None, help="also render the markdown report here (e.g. docs/results/H2_RESULTS.md)")
    ap.add_argument("--batch", type=_positive_int, default=DEFAULT_BATCH,
                    help=f"tokens per batch (memory ~ 3*batch*G*n_out*8 B; default {DEFAULT_BATCH})")
    ap.add_argument("--layers", default=None, help="only these layers (comma list, ranges a-b)")
    ap.add_argument("--mats", default=None, help="only these matrices (subset of gate,up,down)")
    ap.add_argument("--bases", default=None, help="only these bases (e.g. q2,lowrank256)")
    ap.add_argument("--map", default=None, help="map.json of the GGUF (e.g. traces/h2/map31.json)")
    ap.add_argument("--gguf", default=None, help="GGUF path or registry id (default: from the npz / acts meta)")
    ap.add_argument("--y-ref", choices=("gguf", "recon"), default="gguf",
                    help="reference output: exact Q4_0 matrix from the GGUF (default) or W_base + sum R "
                         "(exact only for --res-bits 16 decompositions)")
    ap.add_argument("--n-layers", type=_positive_int, default=None, help="layers for the extrapolation (default: model)")
    ap.add_argument("--attention-bytes", type=_parse_bytes, default=None,
                    help=f"attention bytes/token (default {ATTENTION_BYTES_31B} for 31b-q4, else from the header)")
    ap.add_argument("--head-bytes", type=_parse_bytes, default=None,
                    help=f"head bytes/token (default {HEAD_BYTES_31B} for 31b-q4, else from the header)")
    ap.add_argument("--q4-bytes-per-token", type=_parse_bytes, default=None,
                    help=f"Q4_0 bytes/token for comparison (default {Q4_0_BYTES_PER_TOKEN_31B} for 31b-q4)")
    ap.add_argument("--iq2-bytes", type=_parse_bytes, default=IQ2_XXS_BYTES_31B,
                    help="PROMISING bytes/token limit (default 8.5e9, the 31B IQ2_XXS file)")
    ap.add_argument("--curve-step", type=float, default=DEFAULT_CURVE_STEP,
                    help="byte-fraction step of the error curve (default 0.05)")
    ap.add_argument("--sanity-abort", type=float, default=SANITY_ABORT_MAX,
                    help=f"abort when a sanity median exceeds this (default {SANITY_ABORT_MAX})")
    ap.add_argument("--no-sanity-abort", action="store_true", help="never abort on the sanity check (debugging)")
    ap.add_argument("--results-dir", default=None, help="where the E7 envelopes go (default: the directory of --out)")
    ap.add_argument("--no-envelopes", action="store_true", help="do not write E7 results envelopes")
    ap.add_argument("--no-sha", action="store_true", help="skip the 16 MiB sha256 prefix of the GGUF")
    ap.add_argument("--lock", action="store_true", help="hold results/.bench.lock for the run")
    ap.add_argument("--lock-min-free-gb", type=float, default=4.0,
                    help="available-RAM gate for --lock in GB (default 4; env PIGEMMA_MIN_FREE_GB overrides)")
    ap.add_argument("--force", action="store_true", help="skip the RAM estimate check")
    ap.add_argument("--json", default=None, metavar="OUT|-",
                    help="also write the document here (- = stdout; the table then goes to stderr)")
    ap.add_argument("--render-from", default=None, metavar="ORACLE_JSON",
                    help="only re-render --md from an existing oracle.json (no computation)")
    ap.add_argument("--log-level", default="INFO", help="logging level on stderr (default INFO)")
    return ap


def _summary_table(doc: Mapping[str, Any]) -> str:
    s = doc.get("summary", {})
    vk = eps_key(VERDICT_EPS)
    keys = [eps_key(e) for e in (doc.get("sampling") or {}).get("eps", [])]
    lines = ["E7 oracle (MEASURED fractions/errors on phase all; PROJECTED bytes/token):",
             f"{'layer':>5} {'mat':<4} {'base':<14} {'tok':>5} {'err_base':>8} {'floor':>7} " +
             " ".join(f"{'frac@' + k:>10}" for k in keys) + f" {'eff@' + vk:>9} {'viol':>4}"]
    for c in doc.get("per_case", []):
        if c["phase"] != "all":
            continue
        fr = []
        for k in keys:
            o = c["oracle"].get(k) or {}
            v = o.get("frac_bytes_mean_conservative")
            fr.append(f"{_f(v):>9}" + ("*" if (o.get("unreachable_frac") or 0) > 0 else " "))
        viol = sum(b["soundness_violations"] for b in c["bound"].values()) + int(c.get("bound_violations", 0))
        lines.append(f"{c['layer']:>5} {c['mat']:<4} {c['base']:<14} {c['n_tokens']:>5} {_f(c['err_base_mean'], '.4f'):>8} "
                     f"{_f(c.get('floor_err_mean'), '.4f'):>7} " + " ".join(fr) +
                     f" {_f((c['bound'].get(vk) or {}).get('certification_efficiency')):>9} {viol:>4}")
    lines.append("(* = some tokens unreachable at that eps; counted as 1.0)")
    imp = s.get("implied_bytes_per_token") or {}
    lines.append(f"pipeline {s.get('pipeline')}: " + "; ".join(s.get("pipeline_reasons", [])))
    lines.append(f"verdict {s.get('verdict')} (best base {s.get('best_base')}): frac@3% median "
                 f"{_f(s.get('frac_0_03_median'))}, cert eff median {_f(s.get('cert_eff_median'))}, implied "
                 f"{_gb(imp.get('total'))} GB/token (PROJECTED), {_f(s.get('s_per_token_nvme_850mbps'), '.1f')} s/token "
                 f"@850 MB/s, {_f(s.get('tok_s_in_ram_10gbps'), '.2f')} tok/s @10 GB/s")
    return "\n".join(lines)


def _run(args: argparse.Namespace, argv: Sequence[str]) -> int:
    if args.render_from:
        p = Path(args.render_from)
        if not p.is_file():
            raise MissingInputError(f"oracle.json not found: {p}")
        if not args.md:
            raise UsageError("--render-from needs --md OUT.md")
        doc = json.loads(p.read_text(encoding="utf-8"))
        _atomic_write_text(Path(args.md), render_markdown(doc, source=paths.repo_relative(p)))
        print(Path(args.md).as_posix())
        return paths.EXIT_OK
    if not args.decomp or not args.acts:
        raise UsageError("--decomp and --acts are required (or --render-from)")
    try:
        eps = parse_eps(args.eps)
        layers = _parse_int_list(args.layers, "--layers") if args.layers else None
        bases = [b.strip() for b in args.bases.split(",") if b.strip()] if args.bases else None
        for b in bases or []:
            dc.parse_base(b)
    except ValueError as e:
        raise UsageError(str(e)) from None
    mats = [m.strip() for m in args.mats.split(",") if m.strip()] if args.mats else None
    if mats is not None:
        bad = [m for m in mats if m not in dc.FFN_MATS]
        if bad or not mats:
            raise UsageError(f"--mats: unknown matrices {bad} (choose from {', '.join(dc.FFN_MATS)})")
    if not 0 < args.curve_step <= 1:
        raise UsageError("--curve-step must be in (0, 1]")
    if args.sanity_abort is not None and args.sanity_abort <= 0:
        raise UsageError("--sanity-abort must be > 0 (use --no-sanity-abort to disable)")
    doc = run_oracle(args.decomp, args.acts, tokens=args.tokens, eps=eps, batch=args.batch, layers=layers, mats=mats,
                     bases=bases, map_path=args.map, gguf=args.gguf, y_ref_mode=args.y_ref, n_layers=args.n_layers,
                     attention_bytes=args.attention_bytes, head_bytes=args.head_bytes,
                     q4_bytes_per_token=args.q4_bytes_per_token, iq2_bytes=args.iq2_bytes,
                     curve_step=args.curve_step, sanity_abort=None if args.no_sanity_abort else args.sanity_abort,
                     check_memory=not args.force, with_sha=not args.no_sha, argv=list(argv))
    out = Path(args.out)
    text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    _atomic_write_text(out, text)
    written = [out.as_posix()]
    if args.md:
        _atomic_write_text(Path(args.md), render_markdown(doc, source=paths.repo_relative(out)))
        written.append(Path(args.md).as_posix())
    if not args.no_envelopes:
        rdir = Path(args.results_dir) if args.results_dir else out.parent
        env = write_envelopes(doc, rdir, oracle_json=paths.repo_relative(out))
        written.extend(p.as_posix() for p in env)
    if args.json == "-":
        sys.stderr.write(_summary_table(doc) + "\n")
        sys.stdout.write(text)
    else:
        print(_summary_table(doc))
        if args.json:
            _atomic_write_text(Path(args.json), text)
            written.append(Path(args.json).as_posix())
    for w in written:
        sys.stderr.write(f"pigemma.h2.oracle: wrote {w}\n")
    s = doc["summary"]
    if s.get("pipeline_valid") is False:
        sys.stderr.write("pigemma.h2.oracle: PIPELINE INVALID: " + "; ".join(s.get("pipeline_reasons", [])) +
                         " (the verdict is void)\n")
        return paths.EXIT_ERROR
    if s.get("pipeline_valid") is None:
        sys.stderr.write("pigemma.h2.oracle: WARNING: pipeline validity UNKNOWN: " +
                         "; ".join(s.get("pipeline_reasons", [])) + "\n")
    return paths.EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the exit code."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    argv_l = list(sys.argv[1:] if argv is None else argv)
    ap = _build_parser()
    try:
        args = ap.parse_args(argv_l)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else paths.EXIT_USAGE
    if not logging.getLogger().handlers:
        logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        if args.lock and not args.render_from:
            cmd = "python -m pigemma.h2.oracle " + " ".join(argv_l)
            with paths.bench_lock(args.lock_min_free_gb, cmd=cmd):
                return _run(args, argv_l)
        return _run(args, argv_l)
    except paths.BenchLockError as e:
        sys.stderr.write(f"pigemma.h2.oracle: REFUSED: {e}\n")
        return paths.EXIT_REFUSED
    except RefusedError as e:
        sys.stderr.write(f"pigemma.h2.oracle: REFUSED: {e}\n")
        return paths.EXIT_REFUSED
    except SanityAbort as e:
        sys.stderr.write(f"pigemma.h2.oracle: ABORT: {e}\n")
        return paths.EXIT_ERROR
    except FileNotFoundError as e:   # MissingInputError included
        sys.stderr.write(f"pigemma.h2.oracle: missing: {e}\n")
        return paths.EXIT_MISSING
    except KeyError as e:
        sys.stderr.write(f"pigemma.h2.oracle: missing: {e.args[0] if e.args else e}\n")
        return paths.EXIT_MISSING
    except OracleError as e:
        sys.stderr.write(f"pigemma.h2.oracle: error: {e}\n")
        return paths.EXIT_ERROR
    except (UsageError, argparse.ArgumentTypeError) as e:
        sys.stderr.write(f"pigemma.h2.oracle: usage: {e}\n")
        return paths.EXIT_USAGE
    except KeyboardInterrupt:
        sys.stderr.write("pigemma.h2.oracle: interrupted\n")
        return 130
    except Exception as e:  # noqa: BLE001 - CLI boundary
        sys.stderr.write(f"pigemma.h2.oracle: error: {type(e).__name__}: {e}\n")
        return paths.EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
