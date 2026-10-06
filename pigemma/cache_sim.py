"""pigemma.cache_sim - expert-cache simulator for hypothesis H1 (MoE expert streaming on a Pi 5).

Replays a ``.moetrace`` routing trace (docs/TRACE_FORMAT.md section 3) against an expert cache of a
given byte budget under several replacement policies, reports hit rates and bytes read per token,
and projects decode speed (tok/s) per device profile of ``pigemma/data/catalog.json`` with every
input labelled MEASURED / SIMULATED / ASSUMED. Output: ``sim.json`` + ``sim.csv``
(docs/TRACE_FORMAT.md section 11), optionally a per-token ``sim_series.npz``, and one results
envelope (experiment E4, kind 'projected') per (profile, policy) for ``python -m pigemma.results
summarize``. A second subcommand, ``io-point``, compares a measured page-cache run
(``pigemma-probe --io-csv``) with the simulator and writes ``validation.json`` points (E5/E6,
TRACE_FORMAT section 12), so ``pc/pagecache_limit.sh`` needs no comparison code of its own.

Access model (manifest files[] 'pigemma/cache_sim.py')
-----------------------------------------------------
For token t (trace order), for each MoE slot j (``moe_layers`` order), for each of the K selected
experts e: item ``j*E + e`` (the item id of ``pigemma.trace.access_items``). An item is one expert
of one layer: gate_up + down slabs, both read on a miss. Its CACHE size is the exact unit bytes from
map.json (``expert_layers[].gate_up/down.nb_expert``); its READ cost on a miss is rounded up to
``--read-granularity`` (default 131072 B) per slab to model readahead.

Policies
--------
``lru``          global LRU (OrderedDict).
``lfu``          frequency with exponential decay ``--lfu-decay`` per token (count history kept after
                 eviction), ties evicted oldest-first; always admits.
``static``       pre-filled with the top items by frequency on the first ``--train-split`` of the
                 tokens, never replaced; misses are read but not inserted.
``belady``       offline optimum (Belady/MIN with bypass) from precomputed next-use indices.
``lru_prefetch`` LRU plus a cross-layer predictor: after slot j of token t, the top-N experts of slot
                 j+1 by summed P(e' in S_{j+1} | e in S_j) (``pigemma.trace.cross_layer_table`` on the
                 train split) are read ahead if not cached; prefetch bytes are counted separately.
``lru_perlayer`` the budget split equally per layer, one LRU per layer (control).

Metrics per run (sim.json ``runs[]``): ``steady`` (tokens after ``--warmup-tokens``), ``overall``
(all tokens) and ``test`` (the common out-of-sample window, see below) blocks with hit_rate,
weighted_hit_rate (routing weight), miss_bytes_per_token {mean,p50,p95,max}, n_miss_per_token,
prefetch_bytes_per_token, wasted_prefetch_fraction (+ extras); cold_bytes, per_layer_hit_rate[L],
unique_items_touched, capacity_items, and for profile runs a ``projection``.

Projection (kind 'projected'; ``project()``)::

    expert_cache_bytes = ram - os_reserve - nonexpert_bytes - kv_bytes(n_ctx, q8_0) - compute_buffers
    resident_bytes     = nonexpert_bytes_per_token + hit_bytes_per_token      (expert bytes from RAM)
    t_ram              = resident_bytes / (mem_bw_gbps * 1e9 * llama_efficiency)
    t_io_demand        = miss_bytes / (rand_read_3mb_mbps * 1e6) + n_miss * access_ms / 1e3
    t_io_prefetch      = prefetch_bytes / (rand_read_3mb_mbps * 1e6) + n_prefetch * access_ms / 1e3
    t_io               = t_io_demand + t_io_prefetch
    overlapped         = sum over slots j of min(t_prefetch_j, t_layer),  t_layer = t_ram / n_moe_layers
    tg_serial          = 1 / (t_ram + t_io - overlapped)
    tg_overlap         = 1 / max(t_ram, t_io)
    floor_all_in_ram   = mem_bw * eff / (nonexpert_bytes_per_token + expert_bytes_per_token_worst)
    verdict            = INFEASIBLE (cache <= 0) | FITS_RAM (cache >= all expert bytes)
                         | STREAM_GOOD (tg_overlap >= good 2.0 and tg_serial >= viable 1.0)
                         | STREAM_WEAK (tg_serial >= 0.5) | NOT_VIABLE

llama_efficiency: ``--llama-efficiency X`` > (Pi profiles) results/pi/calibration.json (pi/05,
measured) > a desktop E0 calibration (``--e0-dir``, default results/e0: tg tok/s of a measured E0
envelope x bytes/token / membw read GB/s; 'measured' for the desktop profile, 'assumed:
transferred' for the Pi profiles) > 0.65 assumed. Memory and storage numbers come from the catalog
device (assumed) or, with ``--io-mem``, from a measured results/pi/io_mem.json merged by
``pigemma.feasibility.load_catalog``.

CLI::

    python -m pigemma.cache_sim <trace> --map map.json --out results/h1/sim.json
        [--catalog pigemma/data/catalog.json] [--policies lru,lfu,static,belady,lru_prefetch]
        [--budgets-gb 0.5,1,2,3,4,5,6,8,12.85] [--profiles all|pi5|NAME[,NAME]]
        [--train-split 0.3] [--warmup-tokens 256] [--read-granularity 131072]
        [--read-model round|aligned] [--lfu-decay 0.999] [--prefetch-n K] [--token-range A:B]
        [--io-mem auto|none|PATH] [--calibration auto|none|PATH] [--llama-efficiency X]
        [--e0-dir results/e0] [--no-bypass] [--no-stack-check] [--dense-model 31b-q4]
        [--csv results/h1/sim.csv] [--series results/h1/sim_series.npz]
        [--results-dir DIR | --no-envelopes] [--log-level L]
    python -m pigemma.cache_sim io-point --io-csv io.csv --sim sim.json --mem-cap-bytes N
        [--pinned] [--pin-budget-bytes N] [--pin-locked-bytes N] [--anon-rss-bytes N]
        [--file-cache-bytes N] [--tier-mbps X] [--ram-tok-s X] [--access-ms 0]
        [--warmup-tokens 200] [--page-size 4096] [--volume pigemma-models|bind-mount] [--oom]
        [--sim-budget-bytes N] [--gguf P] [--token-range A:B] [--note T]...
        [--validation results/h1/validation.json]

Exit codes (pigemma.paths): 0 ok; 1 error (bad trace/map, incompatible inputs, a failed
LRU-vs-stack-distance self-check); 2 usage; 3 missing input file.

Design decisions (the manifest left these open; all are also reported as open issues)
--------------------------------------------------------------------------------------
1. **Default runs.** Neither ``--budgets-gb`` nor ``--profiles`` -> both defaults (the 9 budgets and
   every catalog device). Default policies are the five of the CLI spec; ``lru_perlayer`` (a
   control) must be named. Raw budget runs have ``profile: null`` and ``projection: null``.
2. **Windows.** ``steady`` = tokens >= warmup for lru/lfu/belady/lru_perlayer. For the two policies
   that learn from the train split (static, lru_prefetch) ``steady`` starts at
   max(warmup, n_train) so their headline numbers are out-of-sample ("evaluated on the remaining
   tokens"); an injected ``static_items``/``prefetch_table`` (API only) keeps the warmup window.
   Every run also carries a ``test`` block over the SAME window [max(warmup, n_train), n) for
   apples-to-apples comparisons, ``steady_window``/``test_window`` (absolute record indices) and
   ``in_sample``. If the steady window is empty the projection falls back to ``overall`` (noted).
3. **Byte budgets.** Caches are byte-budgeted with exact unit sizes; for uniform quants this is a
   capacity of ``budget // unit`` items (``capacity_items``). Variable per-layer units (UD quants)
   use byte accounting; ``capacity_items`` is then budget / mean unit (``capacity_items_exact``
   false) and Belady is the furthest-next-use heuristic (exact MIN only for uniform sizes).
   Readahead over-read inflates I/O bytes only: the extra pages are not credited as cached.
4. **Belady** allows bypass (an incoming item whose next use is no sooner than the furthest
   resident's is not inserted): the true offline optimum for a cache that may decline insertion,
   i.e. a sound upper bound. ``--no-bypass`` gives forced-insertion MIN.
5. **lru_prefetch.** Predictions with summed score 0 are not issued. Predicted items already cached
   are touched (moved to MRU) before the prefetched ones are inserted, so a prefetch never evicts
   an item it predicts. "Wasted" = prefetched (actually read) items not used by the very next
   slot of the same token. Demand misses are the ``stall`` bytes; ``miss_bytes_per_token`` counts
   demand misses only, ``prefetch_bytes_per_token`` the read-ahead, ``io_bytes_per_token`` both.
6. **tg_overlap / tg_serial.** The spec's ``1/max(t_ram, t_io - overlapped/storage_bw)`` double
   counts overlap when t_io > t_ram. Here: tg_serial = 1/(t_ram + t_io - overlapped) (demand reads
   block; prefetch hides behind at most one layer of compute per slot), tg_overlap =
   1/max(t_ram, t_io) (ideal pipelining, all I/O hidden). For non-prefetch policies both reduce to
   the spec's formulas (overlapped = 0). storage_bw for random 3 MB reads = rand_read_3mb_mbps.
7. **Verdict gap.** tg_serial >= 1.0 with tg_overlap < 2.0 is STREAM_WEAK (the spec's 0.5-1.0 band
   left it unassigned). ``INFEASIBLE`` (TRACE_FORMAT D9) when the profile leaves no expert cache.
8. **Dense line.** ``dense_31b_line`` uses the catalog model ``--dense-model`` (default 31b-q4):
   bytes/token = file data bytes (bytes - data_offset); per profile tok/s in RAM (mem_bw * eff) and
   s/token streamed at ``seq_read_mbps``; the top-level three keys are those of
   pi5-16gb-nvme-gen3 (else the first Pi profile), the rest under ``per_profile``.
9. **Envelopes.** One per (profile, policy) plus one ``dense-stream@<profile>``, written to
   ``--results-dir`` (default: the directory of ``--out``) with deterministic names
   ``E4_<model>_<out stem>_<policy>_<profile>.json`` so a re-run overwrites instead of
   accumulating. tg.tok_s = tg_serial (the conservative number); tg_overlap, the verdict and the
   hit rates go to ``extra``; read_bytes_per_token = demand + prefetch bytes.
10. **Self-check.** Every uniform-size LRU run's ``overall.hit_rate`` is compared with
    ``pigemma.trace`` stack-distance ``lru_hit_rate`` at the same capacity (TRACE_FORMAT 5.2,
    within 1e-9); the result is ``checks.lru_vs_stack_distance`` and a mismatch exits 1.
11. **io-point.** measured bytes/token = sum(read_bytes_delta) / tokens over rows with
    token_index >= warmup (``majflt * page_size`` for bind-mount proxies, a lower bound);
    measured tok/s = 1000 * tokens / sum(wall_ms) (== 1000/mean(wall_ms) at -ub 1).
    predicted tok/s = 1 / (1/ram_tok_s + sim_lru_bytes/tier + n_miss * access_ms), access_ms
    default 0 because a QD1 fio 3 MB randread already contains the per-request latency.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import heapq
import json
import logging
import math
import os
import re
import sys
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pigemma import __version__, paths
from pigemma import feasibility as feas
from pigemma import gguf_map as gm
from pigemma import results as rs
from pigemma import trace as tr

log = logging.getLogger("pigemma.cache_sim")

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
GENERATOR = f"pigemma.cache_sim/{__version__}"
SIM_FORMAT_VERSION = 1
VALIDATION_FORMAT_VERSION = 1

POLICIES: tuple[str, ...] = ("lru", "lfu", "static", "belady", "lru_prefetch", "lru_perlayer")
DEFAULT_POLICIES: tuple[str, ...] = ("lru", "lfu", "static", "belady", "lru_prefetch")
LEARNING_POLICIES = frozenset({"static", "lru_prefetch"})
DEFAULT_BUDGETS_GB: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.85)
DEFAULT_TRAIN_SPLIT = 0.3
DEFAULT_WARMUP_TOKENS = 256
DEFAULT_READ_GRANULARITY = 131072
DEFAULT_LFU_DECAY = 0.999
DEFAULT_LLAMA_EFFICIENCY = 0.65
DEFAULT_N_CTX = 4096
READ_MODELS: tuple[str, ...] = ("round", "aligned")
KV_CACHE_TYPE = "q8_0"

FITS_RAM = "FITS_RAM"
STREAM_GOOD = "STREAM_GOOD"
STREAM_WEAK = "STREAM_WEAK"
NOT_VIABLE = "NOT_VIABLE"
INFEASIBLE = "INFEASIBLE"
VERDICTS: tuple[str, ...] = (FITS_RAM, STREAM_GOOD, STREAM_WEAK, NOT_VIABLE, INFEASIBLE)
DEFAULT_GOOD_TOK_S = 2.0     # catalog.selection.good_stream_tok_s wins (tg_overlap threshold)
DEFAULT_VIABLE_TOK_S = 1.0   # catalog.selection.viable_stream_tok_s wins (tg_serial threshold)
WEAK_TOK_S = 0.5             # manifest: STREAM_WEAK band starts at tg_serial 0.5

BELADY_GAP_THRESHOLD = 1.5   # E4: LRU bytes > 1.5x Belady -> prefetch/static worth implementing
STATIC_WITHIN_FRACTION = 0.10  # E4: static within 10% of LRU -> pinning is the simpler deployment

DENSE_MODEL_ID = "31b-q4"
REFERENCE_PROFILE = "pi5-16gb-nvme-gen3"

SIM_CSV_COLUMNS: tuple[str, ...] = (
    "policy", "budget_gb", "profile", "hit_rate_ss", "weighted_hit_ss", "miss_mb_tok_mean",
    "miss_mb_tok_p95", "tg_serial", "tg_overlap", "verdict",
)
SIM_CSV_HEADER = ",".join(SIM_CSV_COLUMNS)
IO_CSV_HEADER = "token_index,phase,wall_ms,read_bytes_delta,majflt_delta,rss_anon_kb,rss_file_kb"
IO_CSV_COLUMNS: tuple[str, ...] = tuple(IO_CSV_HEADER.split(","))
DEFAULT_IO_WARMUP = 200
DEFAULT_IO_PAGE_SIZE = 4096
VOLUMES: tuple[str, ...] = ("pigemma-models", "bind-mount")
BIND_MOUNT_NOTE = "bind-mount proxy: lower-bound bytes, timings not representative"

DEFAULT_E0_DIR = paths.RESULTS_DIR / "e0"
DEFAULT_PI_CALIBRATION = paths.RESULTS_DIR / "pi" / "calibration.json"
DEFAULT_IO_MEM = paths.RESULTS_DIR / "pi" / "io_mem.json"

_EXPERT_ROLES: tuple[str, ...] = ("gate_up", "gate", "up", "down")
_SUBCOMMANDS = frozenset({"io-point", "simulate"})

__all__ = [
    "simulate", "project", "prepare", "map_info", "profile_budget", "classify", "SimResult",
    "PreparedTrace", "MapInfo", "run_simulation", "write_outputs", "sim_csv_text", "dense_line",
    "efficiency_for_profile", "efficiency_from_e0", "select_profiles", "parse_token_range",
    "parse_budgets_gb", "read_io_csv", "measure_io", "validation_point", "write_validation",
    "static_selection", "predict_next_layer", "main", "POLICIES", "VERDICTS", "CacheSimError",
]


class CacheSimError(ValueError):
    """Invalid or incompatible simulator input (bad map, trace/map mismatch, bad option value)."""


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def _num(x: Any) -> float | None:
    """x as a finite float (int/float/numeric str), else None. Booleans are rejected."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
    elif isinstance(x, str):
        try:
            v = float(x.strip())
        except ValueError:
            return None
    else:
        return None
    return v if math.isfinite(v) else None


def _metric(x: Any) -> float | None:
    """Value of a {value, source} object or a plain number."""
    if isinstance(x, Mapping):
        return _num(x.get("value"))
    return _num(x)


def _src(x: Any, default: str) -> str:
    if isinstance(x, Mapping) and x.get("source"):
        return str(x["source"])
    return default


def _assume(value: Any, unit: str | None, source: str) -> dict[str, Any]:
    """An assumption entry {value, unit, source} (sim.json schema is {value, source}; unit is extra)."""
    return {"value": value, "unit": unit, "source": source}


def _inv(x: float) -> float | None:
    """1/x, or None (= unbounded) when x <= 0."""
    return 1.0 / x if x > 0 else None


def _jsonable(obj: Any) -> Any:
    """numpy -> Python, non-finite floats -> None, tuples -> lists, Paths -> repo-relative strings."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, os.PathLike):
        return paths.repo_relative(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return obj


def _dumps(obj: Any, indent: int | None = 1) -> str:
    return json.dumps(_jsonable(obj), indent=indent, ensure_ascii=False, allow_nan=False) + "\n"


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


def _read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _utc_now() -> str:
    return rs.utc_now_iso()


def _rel(p: str | os.PathLike[str] | None) -> str | None:
    return None if p is None else paths.repo_relative(p)


def _fmt6(v: Any) -> str:
    """CSV cell: empty for None, else up to 6 significant digits."""
    x = _num(v)
    return "" if x is None else f"{x:.6g}"


def parse_token_range(spec: str | Sequence[int | None] | None, n: int) -> tuple[int, int]:
    """'A:B' (Python slice semantics, negatives and empty parts allowed) -> (start, stop) in [0, n]."""
    if spec is None or spec == "" or spec == ":":
        return 0, int(n)
    if isinstance(spec, str):
        m = re.fullmatch(r"\s*(-?\d*)\s*:\s*(-?\d*)\s*", spec)
        if not m:
            raise CacheSimError(f"--token-range must look like A:B (Python slice), got {spec!r}")
        a = int(m.group(1)) if m.group(1) not in ("", "-") else None
        b = int(m.group(2)) if m.group(2) not in ("", "-") else None
    else:
        seq = list(spec)
        if len(seq) != 2:
            raise CacheSimError(f"token range must be [A, B], got {spec!r}")
        a, b = (None if v is None else int(v) for v in seq)
    start, stop, _ = slice(a, b).indices(int(n))
    if stop < start:
        stop = start
    return start, stop


def parse_budgets_gb(text: str | Iterable[Any] | None) -> list[float]:
    """'0.5,1,2' -> [0.5, 1.0, 2.0] (GB = 1e9 B). Negative values are refused."""
    if text is None:
        return []
    items = text.split(",") if isinstance(text, str) else list(text)
    out: list[float] = []
    for it in items:
        s = str(it).strip()
        if not s:
            continue
        v = _num(s)
        if v is None or v < 0:
            raise CacheSimError(f"budget {s!r} is not a non-negative number of GB")
        out.append(v)
    return out


# --------------------------------------------------------------------------------------------
# Map adapter
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class MapInfo:
    """What the simulator needs from a map.json (GgufMap, full dict or a minimal dict).

    A minimal dict needs ``n_expert``, ``n_expert_used`` and ``summary.expert_unit_bytes`` +
    ``summary.n_moe_layers`` (the shape tests/test_trace.py passes). Per-slot lists are in
    ``moe_layers`` order (slot j = j-th MoE layer).
    """

    label: str
    gguf_basename: str | None
    sha256_prefix: str | None
    file_size: int | None
    n_expert: int
    n_expert_used: int
    moe_layers: list[int] | None
    unit_bytes: list[int]
    part_bytes: list[list[int]]
    part_offsets: list[list[int]] | None
    nonexpert_bytes: int
    nonexpert_bytes_per_token: int
    expert_bytes: int
    expert_bytes_per_token_worst: int
    file_data_bytes: int | None
    raw: dict[str, Any] = dataclasses.field(repr=False, default_factory=dict)

    @property
    def n_moe_layers(self) -> int:
        return len(self.unit_bytes)

    @property
    def n_items(self) -> int:
        return self.n_moe_layers * self.n_expert

    @property
    def uniform(self) -> bool:
        return len(set(self.unit_bytes)) <= 1

    @property
    def unit_bytes_mean(self) -> float:
        return float(np.mean(self.unit_bytes)) if self.unit_bytes else 0.0

    def item_sizes(self) -> np.ndarray:
        """int64 [L*E]: cache size of item j*E+e (the layer's unit bytes)."""
        return np.repeat(np.asarray(self.unit_bytes, dtype=np.int64), self.n_expert)

    def read_costs(self, granularity: int, read_model: str = "round") -> tuple[np.ndarray, str]:
        """(int64 [L*E] bytes read on a miss, effective read model).

        'round': each slab rounded up to a multiple of ``granularity``. 'aligned': the granules the
        slab actually touches at its absolute file offset (needs expert_layers offsets; falls back
        to 'round' otherwise).
        """
        g = int(granularity)
        if g < 1:
            raise CacheSimError(f"read granularity must be >= 1, got {granularity}")
        if read_model not in READ_MODELS:
            raise CacheSimError(f"read model must be one of {READ_MODELS}, got {read_model!r}")
        E = self.n_expert
        out = np.zeros(self.n_items, dtype=np.int64)
        eff = read_model
        if read_model == "aligned" and self.part_offsets is None:
            log.warning("read model 'aligned' needs expert slab offsets (map.json expert_layers); using 'round'")
            eff = "round"
        e = np.arange(E, dtype=np.int64)
        for j, parts in enumerate(self.part_bytes):
            if eff == "aligned":
                assert self.part_offsets is not None
                tot = np.zeros(E, dtype=np.int64)
                for off, nb in zip(self.part_offsets[j], parts):
                    start = int(off) + e * int(nb)
                    lo = (start // g) * g
                    hi = -(-(start + int(nb)) // g) * g
                    tot += hi - lo
                out[j * E:(j + 1) * E] = tot
            else:
                out[j * E:(j + 1) * E] = sum(-(-int(nb) // g) * g for nb in parts)
        return out, eff

    def kv_bytes(self, n_ctx: int, cache_type: str = KV_CACHE_TYPE) -> tuple[int, str]:
        """(KV-cache bytes at n_ctx, provenance text)."""
        d = self.raw
        if all(k in d for k in ("head_count_kv", "key_length", "key_length_swa", "sliding_pattern")):
            v = gm.kv_bytes_for(d["head_count_kv"], int(d["key_length"]), int(d["key_length_swa"]),
                                d["sliding_pattern"], d.get("sliding_window"), int(n_ctx), cache_type,
                                d.get("value_length"), d.get("value_length_swa"))
            return int(v), (f"{cache_type} K/V at n_ctx {n_ctx} computed by gguf_map.kv_bytes_for from the "
                            "map's attention header (sliding layers capped at the window)")
        s = d.get("summary") or {}
        per32 = gm.KV_CACHE_BYTES_PER_32.get(cache_type, 64)
        if _num(s.get("kv_bytes_f16_at_4096")) is not None:
            v = int(round(float(s["kv_bytes_f16_at_4096"]) * per32 / 64 * max(1.0, n_ctx / 4096)))
            return v, f"{cache_type} K/V scaled from map summary.kv_bytes_f16_at_4096 (floor at 4096 tokens)"
        if _num(s.get("kv_bytes_per_token_f16")) is not None:
            v = int(round(float(s["kv_bytes_per_token_f16"]) * n_ctx * per32 / 64))
            return v, f"{cache_type} K/V = summary.kv_bytes_per_token_f16 x n_ctx (no sliding-window cap)"
        return 0, "map has no attention header: KV bytes taken as 0"

    def check_trace(self, L: int, K: int, E: int, moe_layers: Sequence[int] | None) -> None:
        """Raise CacheSimError when the trace shape does not match this map."""
        errs = []
        if E != self.n_expert:
            errs.append(f"n_expert trace {E} != map {self.n_expert}")
        if K != self.n_expert_used:
            errs.append(f"n_expert_used trace {K} != map {self.n_expert_used}")
        if L != self.n_moe_layers:
            errs.append(f"MoE layers trace {L} != map {self.n_moe_layers}")
        elif self.moe_layers is not None and moe_layers is not None and list(moe_layers) != list(self.moe_layers):
            errs.append(f"moe_layers differ (trace {list(moe_layers)[:6]}..., map {self.moe_layers[:6]}...)")
        if errs:
            raise CacheSimError(f"trace and map {self.label} are incompatible: " + "; ".join(errs))

    def model_id(self, catalog: Mapping[str, Any] | None = None) -> str | None:
        """Registry / catalog id of the map's GGUF (by file basename), else None."""
        if not self.gguf_basename:
            return None
        mid = paths.model_id_for(self.gguf_basename)
        if mid:
            return mid
        for m in (catalog or {}).get("models", []) or []:
            if Path(str(m.get("file", ""))).name == self.gguf_basename:
                return str(m.get("id"))
        return None


def map_info(m: Any) -> MapInfo:
    """Build a :class:`MapInfo` from a map.json path, a GgufMap, a dict, or a MapInfo."""
    if isinstance(m, MapInfo):
        return m
    label = "<dict>"
    if isinstance(m, (str, os.PathLike)):
        label = paths.repo_relative(m)
        try:
            m = gm.load_map(m)
        except gm.GgufMapError as e:
            raise CacheSimError(str(e)) from e
    elif isinstance(m, gm.GgufMap) and m.path is not None:
        label = paths.repo_relative(m.path)
    if not isinstance(m, Mapping):
        raise CacheSimError(f"map must be a map.json path or mapping, not {type(m).__name__}")
    d = dict(m)
    s = d.get("summary") or {}
    try:
        E = int(d["n_expert"])
        K = int(d["n_expert_used"])
    except (KeyError, TypeError, ValueError) as e:
        raise CacheSimError(f"map {label} lacks n_expert / n_expert_used") from e
    if E <= 0 or K <= 0:
        raise CacheSimError(f"map {label} describes a dense model (n_expert {E}); cache_sim needs a MoE map")
    els = sorted(d.get("expert_layers") or [], key=lambda el: int(el["layer"]))
    part_offsets: list[list[int]] | None
    if els:
        moe_layers: list[int] | None = [int(el["layer"]) for el in els]
        part_bytes: list[list[int]] = []
        part_offsets = []
        for el in els:
            roles = [r for r in _EXPERT_ROLES if r in el]
            if not roles:
                raise CacheSimError(f"map {label}: expert layer {el.get('layer')} has no expert slabs")
            part_bytes.append([int(el[r]["nb_expert"]) for r in roles])
            offs = [el[r].get("offset") for r in roles]
            if part_offsets is not None and all(_num(o) is not None for o in offs):
                part_offsets.append([int(o) for o in offs])
            else:
                part_offsets = None
        unit = [sum(pb) for pb in part_bytes]
    else:
        ml = d.get("moe_layers") or None
        moe_layers = [int(x) for x in ml] if ml else None
        n_moe = _num(s.get("n_moe_layers"))
        n_moe_i = int(n_moe) if n_moe is not None else (len(moe_layers) if moe_layers else 0)
        u = _num(s.get("expert_unit_bytes"))
        if u is None or u <= 0 or n_moe_i <= 0:
            raise CacheSimError(f"map {label} has neither expert_layers nor summary.expert_unit_bytes + n_moe_layers")
        unit = [int(u)] * n_moe_i
        part_bytes = [[int(u)] for _ in range(n_moe_i)]
        part_offsets = None
    nonexpert = int(_num(s.get("nonexpert_bytes")) or 0)
    per_tok = _num(s.get("nonexpert_bytes_per_token"))
    expert_bytes = _num(s.get("expert_bytes"))
    worst = _num(s.get("expert_bytes_per_token_worst"))
    fdb = _num(s.get("file_data_bytes"))
    fsz = _num(d.get("file_size"))
    base = d.get("gguf_basename") or (Path(str(d["gguf"]).replace("\\", "/")).name if d.get("gguf") else None)
    return MapInfo(
        label=label, gguf_basename=base, sha256_prefix=d.get("gguf_sha256_prefix"),
        file_size=int(fsz) if fsz is not None else None, n_expert=E, n_expert_used=K,
        moe_layers=moe_layers, unit_bytes=unit, part_bytes=part_bytes, part_offsets=part_offsets,
        nonexpert_bytes=nonexpert, nonexpert_bytes_per_token=int(per_tok) if per_tok is not None else nonexpert,
        expert_bytes=int(expert_bytes) if expert_bytes else int(sum(unit) * E),
        expert_bytes_per_token_worst=int(worst) if worst else int(K * sum(unit)),
        file_data_bytes=int(fdb) if fdb is not None else None, raw=d,
    )


# --------------------------------------------------------------------------------------------
# Prepared trace
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class PreparedTrace:
    """The routing columns of a trace token range, plus cached derived sequences.

    ``items`` is the access sequence (t, slot j, k order) with item = j*E + e (int64 [n*L*K]).
    """

    path: str
    header: dict[str, Any]
    start: int
    stop: int
    n_records_total: int
    truncated: bool
    L: int
    K: int
    E: int
    moe_layers: list[int]
    ids: np.ndarray
    weights: np.ndarray
    phase: np.ndarray
    items: np.ndarray
    _cache: dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return self.stop - self.start

    @property
    def n_items(self) -> int:
        return self.L * self.E

    @property
    def gguf_basename(self) -> str | None:
        g = self.header.get("gguf_path")
        return Path(str(g).replace("\\", "/")).name if g else None

    @property
    def seq(self) -> list[int]:
        """The access sequence as a Python list (fast iteration in the policy loops)."""
        if "seq" not in self._cache:
            self._cache["seq"] = self.items.tolist()
        return self._cache["seq"]

    @property
    def next_use(self) -> np.ndarray:
        """int64 [n_acc]: index of the next access to the same item, or n_acc when none."""
        if "next_use" not in self._cache:
            x = self.items
            n = int(x.size)
            nu = np.full(n, n, dtype=np.int64)
            if n > 1:
                order = np.argsort(x, kind="stable")
                xs = x[order]
                same = xs[1:] == xs[:-1]
                nu[order[:-1][same]] = order[1:][same]
            self._cache["next_use"] = nu
        return self._cache["next_use"]

    @property
    def first_idx(self) -> np.ndarray:
        """Indices of the first access to each distinct item."""
        if "first_idx" not in self._cache:
            self._cache["first_idx"] = np.unique(self.items, return_index=True)[1] if self.items.size else \
                np.zeros(0, dtype=np.int64)
        return self._cache["first_idx"]

    def gathered(self, key: str, per_item: np.ndarray) -> np.ndarray:
        """per_item[items] reshaped [n, L, K], cached under ``key``."""
        if key not in self._cache:
            self._cache[key] = per_item[self.items].reshape(self.n, self.L, self.K)
        return self._cache[key]

    def window(self, a_rel: int, b_rel: int | None = None) -> list[int]:
        """Absolute record indices [start + a, start + b)."""
        b = self.n if b_rel is None else b_rel
        return [self.start + int(a_rel), self.start + int(b)]


def prepare(trace: Any, token_range: str | Sequence[int | None] | None = None) -> PreparedTrace:
    """Read the routing columns of ``trace`` (path, TraceReader or PreparedTrace) for a token range."""
    if isinstance(trace, PreparedTrace):
        if token_range is None:
            return trace
        a, b = parse_token_range(token_range, trace.n)
        return _subrange(trace, a, b)
    own = not isinstance(trace, tr.TraceReader)
    r = tr.TraceReader(trace) if own else trace
    try:
        a, b = parse_token_range(token_range, r.n)
        if r.L < 1 or r.K < 1 or r.E < 1:
            raise CacheSimError(f"{r.path}: trace has L={r.L}, K={r.K}, E={r.E}")
        ids = np.ascontiguousarray(r.ids[a:b])
        if ids.size and int(ids.max()) >= r.E:
            raise CacheSimError(f"{r.path}: expert ids >= n_expert {r.E}; run `python -m pigemma.trace validate`")
        weights = np.ascontiguousarray(r.weights[a:b], dtype=np.float32)
        weights = np.where(np.isfinite(weights), weights, 0.0).astype(np.float32)
        phase = np.array(r.phase[a:b], dtype=np.uint8)
        items = (np.arange(r.L, dtype=np.int64)[None, :, None] * r.E + ids.astype(np.int64)).ravel()
        return PreparedTrace(path=r.path_str, header=dict(r.header), start=a, stop=b, n_records_total=r.n,
                             truncated=bool(r.truncated), L=r.L, K=r.K, E=r.E, moe_layers=list(r.moe_layers),
                             ids=ids, weights=weights, phase=phase, items=items)
    finally:
        if own:
            r.close()


def _subrange(p: PreparedTrace, a: int, b: int) -> PreparedTrace:
    ids = p.ids[a:b]
    items = (np.arange(p.L, dtype=np.int64)[None, :, None] * p.E + ids.astype(np.int64)).ravel()
    return PreparedTrace(path=p.path, header=p.header, start=p.start + a, stop=p.start + b,
                         n_records_total=p.n_records_total, truncated=p.truncated, L=p.L, K=p.K, E=p.E,
                         moe_layers=p.moe_layers, ids=ids, weights=p.weights[a:b], phase=p.phase[a:b], items=items)


# --------------------------------------------------------------------------------------------
# Policy cores (each returns a hit flag per access)
# --------------------------------------------------------------------------------------------
def _lru_uniform(seq: list[int], cap: int) -> bytearray:
    hit = bytearray(len(seq))
    if cap <= 0:
        return hit
    od: OrderedDict[int, None] = OrderedDict()
    mte = od.move_to_end
    pop = od.popitem
    i = 0
    for x in seq:
        if x in od:
            mte(x)
            hit[i] = 1
        else:
            od[x] = None
            if len(od) > cap:
                pop(False)
        i += 1
    return hit


def _lru_bytes(seq: list[int], sizes: list[int], budget: int) -> bytearray:
    hit = bytearray(len(seq))
    od: OrderedDict[int, int] = OrderedDict()
    mte = od.move_to_end
    pop = od.popitem
    used = 0
    i = 0
    for x in seq:
        if x in od:
            mte(x)
            hit[i] = 1
        else:
            s = sizes[x]
            if s <= budget:
                od[x] = s
                used += s
                while used > budget:
                    used -= pop(False)[1]
        i += 1
    return hit


def _lru_perlayer(seq: list[int], n: int, L: int, K: int, caps: Sequence[int]) -> bytearray:
    hit = bytearray(len(seq))
    ods: list[OrderedDict[int, None]] = [OrderedDict() for _ in range(L)]
    i = 0
    for _t in range(n):
        for j in range(L):
            od = ods[j]
            cap = caps[j]
            for _k in range(K):
                x = seq[i]
                if x in od:
                    od.move_to_end(x)
                    hit[i] = 1
                elif cap > 0:
                    od[x] = None
                    if len(od) > cap:
                        od.popitem(False)
                i += 1
    return hit


def _lfu(seq: list[int], n: int, lk: int, sizes: list[int], budget: int, decay: float) -> bytearray:
    """Decayed-frequency LFU. key = log(count at last access) + lam * t_last is time-invariant."""
    hit = bytearray(len(seq))
    n_items = len(sizes)
    lam = -math.log(decay) if decay < 1.0 else 0.0
    cnt = [0.0] * n_items
    tl = [0] * n_items
    stamp = [-1] * n_items
    resident = bytearray(n_items)
    heap: list[tuple[float, int, int]] = []
    push, popf, lg = heapq.heappush, heapq.heappop, math.log
    used = 0
    n_res = 0
    i = 0
    for t in range(n):
        lt = lam * t
        for _ in range(lk):
            x = seq[i]
            c = cnt[x]
            if c > 0.0 and t != tl[x]:
                c *= decay ** (t - tl[x])
            c += 1.0
            cnt[x] = c
            tl[x] = t
            key = lg(c) + lt
            if resident[x]:
                hit[i] = 1
                stamp[x] = i
                push(heap, (key, i, x))
            else:
                s = sizes[x]
                if s <= budget:
                    while used + s > budget:
                        while True:
                            _k0, i0, y = popf(heap)
                            if resident[y] and stamp[y] == i0:
                                break
                        resident[y] = 0
                        used -= sizes[y]
                        n_res -= 1
                    resident[x] = 1
                    used += s
                    n_res += 1
                    stamp[x] = i
                    push(heap, (key, i, x))
            i += 1
        if len(heap) > 4 * n_res + 4096:
            heap = [(lg(cnt[y]) + lam * tl[y], stamp[y], y) for y in range(n_items) if resident[y]]
            heapq.heapify(heap)
    return hit


def _belady(seq: list[int], next_use: list[int], sizes: list[int], budget: int, bypass: bool) -> bytearray:
    """Belady/MIN: evict the resident item whose next use is furthest away (lazy-deletion heap)."""
    hit = bytearray(len(seq))
    res: dict[int, int] = {}
    heap: list[tuple[int, int]] = []
    push, popf = heapq.heappush, heapq.heappop
    used = 0
    for i, x in enumerate(seq):
        nu = next_use[i]
        if x in res:
            hit[i] = 1
            res[x] = nu
            push(heap, (-nu, x))
        else:
            s = sizes[x]
            if s <= budget:
                insert = True
                while used + s > budget:
                    while True:
                        negnu, y = heap[0]
                        if res.get(y) == -negnu:
                            break
                        popf(heap)
                    if bypass and nu >= -negnu:
                        insert = False
                        break
                    popf(heap)
                    del res[y]
                    used -= sizes[y]
                if insert:
                    res[x] = nu
                    used += s
                    push(heap, (-nu, x))
        if len(heap) > 4 * len(res) + 4096:
            heap = [(-v, k) for k, v in res.items()]
            heapq.heapify(heap)
    return hit


def _lru_prefetch(seq: list[int], pred: list, n: int, L: int, K: int, E: int, sizes: list[int],
                  cost: list[int], budget: int) -> tuple[bytearray, dict[str, np.ndarray]]:
    """LRU + read-ahead of the predicted next-slot experts. pred[t][j] lists experts for slot j."""
    hit = bytearray(len(seq))
    pf_b = [0] * (n * L)
    pf_c = [0] * (n * L)
    w_b = [0] * (n * L)
    w_c = [0] * (n * L)
    od: OrderedDict[int, int] = OrderedDict()
    mte = od.move_to_end
    pop = od.popitem
    used = 0
    i = 0
    for t in range(n):
        pt = pred[t]
        for j in range(L):
            idx = t * L + j
            pfs: list[int] | None = None
            if j:
                jE = j * E
                for e in pt[j]:
                    if e < 0:
                        continue
                    x = jE + e
                    if x in od:
                        mte(x)
                        continue
                    s = sizes[x]
                    if s > budget:
                        continue
                    od[x] = s
                    used += s
                    while used > budget:
                        used -= pop(False)[1]
                    if pfs is None:
                        pfs = []
                    pfs.append(x)
                    pf_b[idx] += cost[x]
                    pf_c[idx] += 1
            base = i
            for _k in range(K):
                x = seq[i]
                if x in od:
                    mte(x)
                    hit[i] = 1
                else:
                    s = sizes[x]
                    if s <= budget:
                        od[x] = s
                        used += s
                        while used > budget:
                            used -= pop(False)[1]
                i += 1
            if pfs:
                acc = seq[base:i]
                for x in pfs:
                    if x not in acc:
                        w_b[idx] += cost[x]
                        w_c[idx] += 1
    arr = {
        "pf_bytes": np.asarray(pf_b, dtype=np.int64).reshape(n, L),
        "pf_items": np.asarray(pf_c, dtype=np.int64).reshape(n, L),
        "wasted_bytes": np.asarray(w_b, dtype=np.int64).reshape(n, L),
        "wasted_items": np.asarray(w_c, dtype=np.int64).reshape(n, L),
    }
    return hit, arr


def static_selection(items_train: np.ndarray, sizes: np.ndarray, budget: int) -> np.ndarray:
    """bool [n_items]: the most frequent items of ``items_train`` that fit in ``budget`` bytes.

    Rank: frequency descending, ties to the lower item id; greedy fill (an item that no longer fits
    is skipped and smaller ones may still be added). Never-seen items rank last but still fill the
    budget (what a pinned hot set would hold).
    """
    n_items = int(sizes.size)
    sel = np.zeros(n_items, dtype=bool)
    if budget <= 0 or n_items == 0:
        return sel
    freq = np.bincount(np.asarray(items_train, dtype=np.int64), minlength=n_items)[:n_items]
    order = np.lexsort((np.arange(n_items), -freq))
    if np.all(sizes == sizes[0]):
        cap = int(budget // int(sizes[0])) if sizes[0] > 0 else n_items
        sel[order[:cap]] = True
        return sel
    used = 0
    for x in order.tolist():
        s = int(sizes[x])
        if used + s <= budget:
            sel[x] = True
            used += s
    return sel


def predict_next_layer(ids: np.ndarray, table: Sequence[np.ndarray], n_pred: int, n_expert: int) -> np.ndarray:
    """int16 [n, L, n_pred]: predicted experts of slot j (rows j >= 1) from slot j-1; -1 = none.

    score(e') = sum over e in S_{j-1} of table[j-1][e, e']; top ``n_pred`` by score, ties to the
    lower id; experts with score 0 are not predicted.
    """
    n, L, _K = ids.shape
    E = int(n_expert)
    n_pred = max(0, min(int(n_pred), E))
    pred = np.full((n, L, n_pred), -1, dtype=np.int16)
    if n == 0 or n_pred == 0:
        return pred
    rows = np.arange(n)[:, None]
    for j in range(L - 1):
        mem = np.zeros((n, E), dtype=np.float64)
        mem[rows, ids[:, j].astype(np.int64)] = 1.0
        score = mem @ np.asarray(table[j], dtype=np.float64)
        top = np.argsort(-score, axis=1, kind="stable")[:, :n_pred]
        val = np.take_along_axis(score, top, axis=1)
        pred[:, j + 1] = np.where(val > 0, top, -1).astype(np.int16)
    return pred


# --------------------------------------------------------------------------------------------
# Simulation result
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class SimResult:
    """One (policy, budget[, profile]) simulation. ``to_dict()`` is the sim.json ``runs[]`` entry.

    ``steady``/``overall``/``test`` are dicts with hit_rate, weighted_hit_rate,
    miss_bytes_per_token {mean,p50,p95,max}, n_miss_per_token, prefetch_bytes_per_token,
    wasted_prefetch_fraction and extras. ``series`` holds per-token numpy arrays (not serialised).
    Item access works too: ``res["overall"]["hit_rate"]``.
    """

    policy: str
    budget_bytes: int
    capacity_items: int
    capacity_items_exact: bool
    profile: str | None
    steady: dict[str, Any]
    overall: dict[str, Any]
    test: dict[str, Any]
    cold_bytes: int
    per_layer_hit_rate: list[float | None]
    unique_items_touched: int
    steady_window: list[int]
    test_window: list[int]
    in_sample: bool
    n_tokens: int
    token_range: list[int]
    params: dict[str, Any]
    info: dict[str, Any]
    notes: list[str]
    wall_s: float
    projection: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    series: dict[str, np.ndarray] | None = dataclasses.field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "policy": self.policy,
            "budget_bytes": int(self.budget_bytes),
            "capacity_items": int(self.capacity_items),
            "profile": self.profile,
            "steady": self.steady,
            "overall": self.overall,
            "cold_bytes": int(self.cold_bytes),
            "per_layer_hit_rate": self.per_layer_hit_rate,
            "projection": self.projection,
            "test": self.test,
            "capacity_items_exact": self.capacity_items_exact,
            "unique_items_touched": int(self.unique_items_touched),
            "steady_window": self.steady_window,
            "test_window": self.test_window,
            "in_sample": self.in_sample,
            "n_tokens": int(self.n_tokens),
            "token_range": self.token_range,
            "budget": self.budget,
            "params": self.params,
            "info": self.info,
            "notes": list(self.notes),
            "wall_s": round(float(self.wall_s), 3),
        }
        return _jsonable(d)

    def __getitem__(self, key: str) -> Any:
        if key == "series":
            return self.series
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _empty_block(window: list[int]) -> dict[str, Any]:
    return {
        "hit_rate": None, "weighted_hit_rate": None,
        "miss_bytes_per_token": {"mean": None, "p50": None, "p95": None, "max": None},
        "n_miss_per_token": None, "prefetch_bytes_per_token": None, "wasted_prefetch_fraction": None,
        "stall_bytes_per_token": None, "hit_bytes_per_token": None, "io_bytes_per_token": None,
        "prefetch_items_per_token": None, "wasted_prefetch_bytes_per_token": None,
        "prefetch_bytes_per_token_by_layer": None, "prefetch_items_per_token_by_layer": None,
        "n_tokens": 0, "n_accesses": 0, "window": window,
    }


def simulate(trace: Any, map: Any, policy: str, budget_bytes: float, *,  # noqa: A002 - spec name
             warmup_tokens: int = DEFAULT_WARMUP_TOKENS, read_granularity: int = DEFAULT_READ_GRANULARITY,
             read_model: str = "round", train_split: float = DEFAULT_TRAIN_SPLIT,
             lfu_decay: float = DEFAULT_LFU_DECAY, token_range: str | Sequence[int | None] | None = None,
             prefetch_table: Sequence[np.ndarray] | None = None, prefetch_n: int | None = None,
             static_items: Iterable[int] | None = None, belady_bypass: bool = True,
             profile: str | None = None, keep_series: bool = True) -> SimResult:
    """Replay ``trace`` against an expert cache of ``budget_bytes`` under ``policy``.

    ``trace``: path, TraceReader or PreparedTrace (prepare once to run many policies cheaply).
    ``map``: map.json path, GgufMap, dict (a minimal dict with n_expert, n_expert_used and
    summary.expert_unit_bytes + n_moe_layers works) or MapInfo. A negative budget is simulated
    as 0. ``prefetch_table``: L-1 [E,E] matrices replacing the train-split cross-layer table.
    ``static_items``: item ids (j*E+e) replacing the train-split static selection.
    """
    t0 = time.perf_counter()
    prep = prepare(trace, token_range)
    mi = map_info(map)
    mi.check_trace(prep.L, prep.K, prep.E, prep.moe_layers)
    pol = str(policy).strip().lower()
    if pol not in POLICIES:
        raise CacheSimError(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")
    if not 0.0 <= float(train_split) <= 1.0:
        raise CacheSimError(f"train_split {train_split} outside [0, 1]")
    if not 0.0 < float(lfu_decay) <= 1.0:
        raise CacheSimError(f"lfu_decay {lfu_decay} outside (0, 1]")
    if int(warmup_tokens) < 0:
        raise CacheSimError("warmup_tokens must be >= 0")
    n, L, K, E = prep.n, prep.L, prep.K, prep.E
    budget = max(0, int(budget_bytes))
    sizes_np = mi.item_sizes()
    cost_np, read_model_eff = mi.read_costs(int(read_granularity), read_model)
    sizes = sizes_np.tolist()
    seq = prep.seq
    n_train = int(math.floor(float(train_split) * n))
    warm = min(int(warmup_tokens), n)
    notes: list[str] = []
    info: dict[str, Any] = {}
    capacity_exact = mi.uniform
    unit0 = mi.unit_bytes[0] if mi.unit_bytes else 0
    capacity = (budget // unit0 if unit0 > 0 else 0) if mi.uniform else int(budget // max(1.0, mi.unit_bytes_mean))
    pf_arr: dict[str, np.ndarray] | None = None

    if pol == "lru":
        hitbuf = _lru_uniform(seq, capacity) if mi.uniform else _lru_bytes(seq, sizes, budget)
        hit = np.frombuffer(bytes(hitbuf), dtype=np.uint8).astype(bool)
    elif pol == "lru_perlayer":
        per = budget // L
        caps = [int(per // u) if u > 0 else 0 for u in mi.unit_bytes]
        hit = np.frombuffer(bytes(_lru_perlayer(seq, n, L, K, caps)), dtype=np.uint8).astype(bool)
        capacity = int(sum(caps))
        capacity_exact = True
        info["budget_per_layer_bytes"] = int(per)
        info["capacity_items_per_layer"] = caps
    elif pol == "lfu":
        hit = np.frombuffer(bytes(_lfu(seq, n, L * K, sizes, budget, float(lfu_decay))), dtype=np.uint8).astype(bool)
    elif pol == "belady":
        nu = prep.next_use.tolist()
        hit = np.frombuffer(bytes(_belady(seq, nu, sizes, budget, bool(belady_bypass))), dtype=np.uint8).astype(bool)
        del nu
        info["bypass"] = bool(belady_bypass)
        if not mi.uniform:
            notes.append("variable expert sizes: belady is the furthest-next-use heuristic, not the exact optimum")
    elif pol == "static":
        if static_items is not None:
            sel = np.zeros(prep.n_items, dtype=bool)
            ids_list = [int(x) for x in static_items]
            if ids_list and (min(ids_list) < 0 or max(ids_list) >= prep.n_items):
                raise CacheSimError(f"static_items must lie in [0, {prep.n_items})")
            sel[ids_list] = True
            if int(sizes_np[sel].sum()) > budget:
                notes.append("injected static_items exceed the budget (simulated as given)")
            info["selection"] = "injected"
        else:
            sel = static_selection(prep.items[: n_train * L * K], sizes_np, budget)
            info["selection"] = f"top items by frequency on the first {n_train} tokens (train split)"
            if n_train == 0:
                notes.append("train split is empty: the static set is arbitrary (item-id order)")
        hit = sel[prep.items]
        info["static_items"] = int(sel.sum())
        info["static_bytes"] = int(sizes_np[sel].sum())
        capacity = int(sel.sum()) if mi.uniform else capacity
    elif pol == "lru_prefetch":
        n_pred = K if prefetch_n is None else int(prefetch_n)
        if prefetch_table is not None:
            table = [np.asarray(t, dtype=np.float64) for t in prefetch_table]
            if len(table) != L - 1 or any(t.shape != (E, E) for t in table):
                raise CacheSimError(f"prefetch_table must be {L - 1} arrays of shape ({E}, {E})")
            info["table"] = "injected"
        else:
            table = tr.cross_layer_table(prep.ids, n_train, E) if L > 1 else []
            info["table"] = f"pigemma.trace.cross_layer_table on the first {n_train} tokens (train split)"
            if n_train == 0:
                notes.append("train split is empty: the predictor has no table, nothing is prefetched")
        pred = predict_next_layer(prep.ids, table, n_pred, E).tolist()
        hitbuf, pf_arr = _lru_prefetch(seq, pred, n, L, K, E, sizes, cost_np.tolist(), budget)
        del pred
        hit = np.frombuffer(bytes(hitbuf), dtype=np.uint8).astype(bool)
        info["prefetch_n"] = n_pred
    else:  # pragma: no cover - guarded above
        raise CacheSimError(pol)

    learned = pol in LEARNING_POLICIES and not (
        (pol == "static" and static_items is not None) or (pol == "lru_prefetch" and prefetch_table is not None))
    ss_a = max(warm, n_train) if learned else warm
    test_a = min(n, max(warm, n_train))
    if learned and n_train > warm:
        notes.append(f"steady window starts after the train split (token {n_train} of {n}): out-of-sample")
    in_sample = bool(pol in LEARNING_POLICIES and not learned)

    # ---- metrics -----------------------------------------------------------------------------
    h3 = hit.reshape(n, L, K)
    cost_acc = prep.gathered(f"cost:{read_model_eff}:{int(read_granularity)}:{mi.label}", cost_np)
    size_acc = prep.gathered(f"size:{mi.label}", sizes_np)
    if "w64" not in prep._cache:
        prep._cache["w64"] = prep.weights.astype(np.float64)
        prep._cache["w_tok"] = prep._cache["w64"].sum(axis=(1, 2))
    ck = f"cost_tok:{read_model_eff}:{int(read_granularity)}:{mi.label}"
    if ck not in prep._cache:
        prep._cache[ck] = cost_acc.sum(axis=(1, 2))
    hits_tok = h3.sum(axis=(1, 2), dtype=np.int64)
    miss_bytes_tok = prep._cache[ck] - (cost_acc * h3).sum(axis=(1, 2))
    hit_bytes_tok = (size_acc * h3).sum(axis=(1, 2))
    n_miss_tok = L * K - hits_tok
    w_hit_tok = (prep._cache["w64"] * h3).sum(axis=(1, 2))
    w_tok = prep._cache["w_tok"]
    if pf_arr is not None:
        pf_b_tok = pf_arr["pf_bytes"].sum(axis=1)
        pf_c_tok = pf_arr["pf_items"].sum(axis=1)
        wb_tok = pf_arr["wasted_bytes"].sum(axis=1)
    else:
        pf_b_tok = pf_c_tok = wb_tok = None

    def block(a: int, b: int) -> dict[str, Any]:
        win = prep.window(a, b)
        if b <= a:
            return _empty_block(win)
        sl = slice(a, b)
        mb = miss_bytes_tok[sl].astype(np.float64)
        nt = b - a
        wsum = float(w_tok[sl].sum())
        out: dict[str, Any] = {
            "hit_rate": float(hits_tok[sl].sum()) / (nt * L * K),
            "weighted_hit_rate": float(w_hit_tok[sl].sum()) / wsum if wsum > 0 else None,
            "miss_bytes_per_token": {
                "mean": float(mb.mean()),
                "p50": float(np.percentile(mb, 50)),
                "p95": float(np.percentile(mb, 95)),
                "max": int(miss_bytes_tok[sl].max()),
            },
            "n_miss_per_token": float(n_miss_tok[sl].mean()),
            "prefetch_bytes_per_token": 0.0,
            "wasted_prefetch_fraction": None,
            "stall_bytes_per_token": float(mb.mean()),
            "hit_bytes_per_token": float(hit_bytes_tok[sl].mean()),
            "io_bytes_per_token": float(mb.mean()),
            "prefetch_items_per_token": 0.0,
            "wasted_prefetch_bytes_per_token": 0.0,
            "prefetch_bytes_per_token_by_layer": None,
            "prefetch_items_per_token_by_layer": None,
            "n_tokens": int(nt),
            "n_accesses": int(nt * L * K),
            "window": win,
        }
        if pf_arr is not None:
            assert pf_b_tok is not None and pf_c_tok is not None and wb_tok is not None
            pfb = float(pf_b_tok[sl].sum())
            out["prefetch_bytes_per_token"] = pfb / nt
            out["prefetch_items_per_token"] = float(pf_c_tok[sl].sum()) / nt
            out["wasted_prefetch_bytes_per_token"] = float(wb_tok[sl].sum()) / nt
            out["wasted_prefetch_fraction"] = float(wb_tok[sl].sum()) / pfb if pfb > 0 else None
            out["io_bytes_per_token"] = float(mb.mean()) + pfb / nt
            out["prefetch_bytes_per_token_by_layer"] = [float(v) for v in pf_arr["pf_bytes"][sl].mean(axis=0)]
            out["prefetch_items_per_token_by_layer"] = [float(v) for v in pf_arr["pf_items"][sl].mean(axis=0)]
        return out

    steady = block(ss_a, n)
    overall = block(0, n)
    test = block(test_a, n)
    if steady["n_tokens"] == 0:
        notes.append(f"steady window is empty (n={n}, start {ss_a}); use the overall block")
        per_layer: list[float | None] = [None] * L
    else:
        per_layer = [float(v) for v in h3[ss_a:].mean(axis=(0, 2))]
    fi = prep.first_idx
    cold_bytes = int(cost_np[prep.items[fi]][~hit[fi]].sum()) if fi.size else 0
    series = None
    if keep_series:
        series = {"miss_bytes": miss_bytes_tok.astype(np.int64), "n_miss": n_miss_tok.astype(np.int32),
                  "hit_bytes": hit_bytes_tok.astype(np.int64)}
        if pf_b_tok is not None:
            series["prefetch_bytes"] = pf_b_tok.astype(np.int64)
    params = {"warmup_tokens": int(warmup_tokens), "read_granularity": int(read_granularity),
              "read_model": read_model_eff, "train_split": float(train_split), "n_train": n_train,
              "lfu_decay": float(lfu_decay) if pol == "lfu" else None}
    return SimResult(
        policy=pol, budget_bytes=budget, capacity_items=int(capacity), capacity_items_exact=bool(capacity_exact),
        profile=profile, steady=steady, overall=overall, test=test, cold_bytes=cold_bytes,
        per_layer_hit_rate=per_layer, unique_items_touched=int(fi.size), steady_window=prep.window(ss_a),
        test_window=prep.window(test_a), in_sample=in_sample, n_tokens=n, token_range=[prep.start, prep.stop],
        params=params, info=info, notes=notes, wall_s=time.perf_counter() - t0, series=series,
    )


# --------------------------------------------------------------------------------------------
# Profiles, calibration, projection
# --------------------------------------------------------------------------------------------
def _resolve_profile(profile: Any, catalog: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(profile, Mapping):
        return dict(profile)
    if isinstance(profile, str):
        cat = catalog if catalog is not None else feas.load_catalog(paths.CATALOG_PATH)
        for d in cat.get("devices", []) or []:
            if d.get("name") == profile:
                return dict(d)
        raise CacheSimError(f"unknown device profile {profile!r}; known: "
                            + ", ".join(str(d.get("name")) for d in cat.get("devices", []) or []))
    raise CacheSimError(f"profile must be a catalog device dict or name, not {type(profile).__name__}")


def select_profiles(catalog: Mapping[str, Any], spec: str | Sequence[str] | None) -> list[dict[str, Any]]:
    """'all' | 'pi5' (or 'pi') | 'desktop' | 'name[,name]' -> catalog device dicts (catalog order)."""
    devices = [dict(d) for d in catalog.get("devices", []) or []]
    if spec is None:
        return []
    names = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec) if str(s).strip()]
    out: list[dict[str, Any]] = []
    for nm in names:
        low = nm.lower()
        if low == "all":
            picked = devices
        elif low in ("pi", "pi5"):
            picked = [d for d in devices if str(d.get("platform", "")).lower() == "pi5"]
        elif low == "desktop":
            picked = [d for d in devices if str(d.get("platform", "")).lower() == "desktop"]
        else:
            picked = [d for d in devices if d.get("name") == nm]
            if not picked:
                raise CacheSimError(f"unknown device profile {nm!r}; known: "
                                    + ", ".join(str(d.get("name")) for d in devices) + " (or all, pi5, desktop)")
        for d in picked:
            if all(o.get("name") != d.get("name") for o in out):
                out.append(d)
    return out


def profile_budget(profile: Any, map: Any, *, catalog: Mapping[str, Any] | None = None,  # noqa: A002
                   n_ctx: int | None = None) -> dict[str, Any]:
    """Expert-cache budget of a device profile for a model map.

    expert_cache_bytes = ram_bytes - os_reserve_bytes - nonexpert_bytes - kv_bytes(n_ctx, q8_0)
    - compute_buffers_bytes. ``budget_bytes`` = max(0, expert_cache_bytes); ``infeasible`` when
    expert_cache_bytes <= 0; ``fits_all_experts`` when it holds every expert byte.
    """
    prof = _resolve_profile(profile, catalog)
    mi = map_info(map)
    srcs = prof.get("sources", {}) or {}
    ram = _num(prof.get("ram_bytes"))
    osr = _num(prof.get("os_reserve_bytes"))
    cb = _num(prof.get("compute_buffers_bytes"))
    if ram is None or osr is None or cb is None:
        raise CacheSimError(f"profile {prof.get('name')!r} lacks ram_bytes / os_reserve_bytes / compute_buffers_bytes")
    ctx = int(n_ctx if n_ctx is not None else (_num(prof.get("n_ctx")) or DEFAULT_N_CTX))
    kv, kv_src = mi.kv_bytes(ctx, KV_CACHE_TYPE)
    cache = int(ram) - int(osr) - int(mi.nonexpert_bytes) - int(kv) - int(cb)
    return {
        "profile": prof.get("name"),
        "expert_cache_bytes": int(cache),
        "budget_bytes": int(max(0, cache)),
        "infeasible": bool(cache <= 0),
        "fits_all_experts": bool(cache >= mi.expert_bytes),
        "ram_bytes": int(ram), "os_reserve_bytes": int(osr), "nonexpert_bytes": int(mi.nonexpert_bytes),
        "kv_bytes": int(kv), "kv_cache_type": KV_CACHE_TYPE, "compute_buffers_bytes": int(cb), "n_ctx": ctx,
        "expert_bytes": int(mi.expert_bytes),
        "sources": {
            "ram_bytes": str(srcs.get("ram_bytes", "assumed: catalog device ram_bytes")),
            "os_reserve_bytes": str(srcs.get("os_reserve_bytes", "assumed: catalog device os_reserve_bytes")),
            "compute_buffers_bytes": str(srcs.get("compute_buffers_bytes", "assumed: catalog compute_buffers_bytes")),
            "n_ctx": str(srcs.get("n_ctx", "assumed: run configuration")),
            "kv_bytes": f"assumed: {kv_src}",
            "nonexpert_bytes": f"measured: GGUF header ({mi.label} summary.nonexpert_bytes)",
        },
    }


def _default_efficiency() -> dict[str, Any]:
    return {"value": DEFAULT_LLAMA_EFFICIENCY,
            "source": "assumed: llama.cpp decode reaches 65% of the memory bandwidth (catalog / feasibility "
                      "default); replace with results/pi/calibration.json (pi/05) or a desktop E0 run + membw",
            "files": []}


def _find_membw(root: Path) -> tuple[float, str] | None:
    """Max read GB/s over every membw --json output (raw, or under an envelope's extra.membw)."""
    best: tuple[float, str] | None = None
    for p in sorted(root.rglob("*.json")):
        try:
            obj = _read_json(p)
        except (OSError, ValueError):
            continue
        cands = []
        if isinstance(obj, Mapping):
            cands.append(obj)
            ex = obj.get("extra")
            if isinstance(ex, Mapping) and isinstance(ex.get("membw"), Mapping):
                cands.append(ex["membw"])
        for c in cands:
            if c.get("tool") != "membw" or not isinstance(c.get("results"), list):
                continue
            vals = [_num(r.get("read_gbps")) for r in c["results"] if isinstance(r, Mapping)]
            vals = [v for v in vals if v is not None and v > 0]
            if vals and (best is None or max(vals) > best[0]):
                best = (max(vals), paths.repo_relative(p))
    return best


def efficiency_from_e0(e0_dir: str | os.PathLike[str], catalog: Mapping[str, Any], *,
                       prefer_model: str | None = None) -> dict[str, Any] | None:
    """Desktop llama_efficiency = tg_measured x bytes_per_token / membw read GB/s (manifest spec).

    Uses measured E0 envelopes (tg.tok_s, model.id in the catalog) and membw --json outputs found
    under ``e0_dir``. The row: ``prefer_model`` (else any MoE, else any model), variant 'default'
    preferred, highest tg. bytes_per_token = nonexpert_bytes_per_token + expert_bytes_per_token_worst
    of the catalog model. Returns None when either half is missing.
    """
    root = Path(e0_dir)
    if not root.is_dir():
        return None
    bw = _find_membw(root)
    if bw is None:
        log.info("%s: no membw --json output found; desktop llama_efficiency not derived", root)
        return None
    try:
        envs = rs.load_results(root, recursive=True)
    except Exception as e:  # noqa: BLE001 - a broken results dir must not stop the simulation
        log.warning("%s: cannot load results envelopes (%s)", root, e)
        return None
    models = {str(m.get("id")): m for m in catalog.get("models", []) or []}
    rows = []
    for d in envs:
        if d.get("experiment") != "E0" or d.get("kind") != "measured":
            continue
        tg = _num((d.get("tg") or {}).get("tok_s")) if isinstance(d.get("tg"), Mapping) else None
        mid = (d.get("model") or {}).get("id") if isinstance(d.get("model"), Mapping) else None
        m = models.get(str(mid))
        if tg is None or tg <= 0 or m is None:
            continue
        bpt = (_num(m.get("nonexpert_bytes_per_token")) or 0) + (_num(m.get("expert_bytes_per_token_worst")) or 0)
        if bpt <= 0:
            continue
        rows.append((d, str(mid), tg, bpt, str(m.get("kind"))))
    if not rows:
        log.info("%s: no measured E0 envelope with tg tok/s for a catalog model", root)
        return None

    def rank(r: tuple) -> tuple:
        d, mid, tg, _bpt, kind = r
        return (mid == prefer_model, kind == "moe", d.get("variant") == "default", tg)

    d, mid, tg, bpt, _kind = max(rows, key=rank)
    eff = tg * bpt / (bw[0] * 1e9)
    src = (f"measured: desktop llama_efficiency = tg {tg:.3f} tok/s ({mid}, variant {d.get('variant')}, "
           f"threads {d.get('threads')}) x {bpt / 1e9:.4f} GB/token / membw read {bw[0]:.2f} GB/s "
           f"({d.get('_path')}, {bw[1]})")
    return {"value": round(eff, 4), "source": src, "files": [str(d.get("_path")), bw[1]], "model": mid}


def _load_pi_calibration(spec: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    p = Path(spec)
    if not p.is_file():
        return None
    val, src = feas.load_calibration(p)
    return {"value": float(val), "source": src, "files": [paths.repo_relative(p)]}


def efficiency_for_profile(profile: Mapping[str, Any], *, explicit: Mapping[str, Any] | None = None,
                           pi_cal: Mapping[str, Any] | None = None,
                           desktop_cal: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """llama_efficiency {value, source, files} for a device profile (see the module docstring)."""
    if explicit is not None:
        return dict(explicit)
    platform = str(profile.get("platform", "")).lower()
    if platform == "desktop":
        if desktop_cal is not None:
            return dict(desktop_cal)
        return _default_efficiency()
    if pi_cal is not None:
        return dict(pi_cal)
    if desktop_cal is not None:
        return {"value": desktop_cal["value"],
                "source": ("assumed: the desktop-measured llama.cpp efficiency is transferred to this profile "
                           f"(not measured here; {desktop_cal['source']})"),
                "files": list(desktop_cal.get("files", []))}
    return _default_efficiency()


def classify(tg_serial: float | None, tg_overlap: float | None, expert_cache_bytes: float, expert_bytes: float,
             *, good: float = DEFAULT_GOOD_TOK_S, viable: float = DEFAULT_VIABLE_TOK_S,
             weak: float = WEAK_TOK_S) -> str:
    """Verdict for one (profile, policy). None tok/s means unbounded (no time per token)."""
    if expert_cache_bytes <= 0:
        return INFEASIBLE
    if expert_cache_bytes >= expert_bytes:
        return FITS_RAM
    ser = math.inf if tg_serial is None else float(tg_serial)
    ovl = math.inf if tg_overlap is None else float(tg_overlap)
    if ovl >= good and ser >= viable:
        return STREAM_GOOD
    if ser >= weak:
        return STREAM_WEAK
    return NOT_VIABLE


def _thresholds(catalog: Mapping[str, Any] | None) -> tuple[float, float]:
    sel = (catalog or {}).get("selection", {}) or {}
    good = _num(sel.get("good_stream_tok_s")) or DEFAULT_GOOD_TOK_S
    viable = _num(sel.get("viable_stream_tok_s")) or DEFAULT_VIABLE_TOK_S
    return good, viable


def project(sim_result: SimResult | Mapping[str, Any], profile: Any, map: Any,  # noqa: A002 - spec name
            calibration: Any = None, *, catalog: Mapping[str, Any] | None = None,
            budget_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Projected decode speed of one simulation on one device profile (kind 'projected').

    ``calibration``: None (0.65 assumed), a float (taken as given), (value, source), or
    {value, source}. Returns the sim.json ``projection`` block (+ extras: t_io_demand_s,
    t_io_prefetch_s, overlapped_s, expert_cache_bytes, thresholds, window, notes).
    """
    run = sim_result.to_dict() if isinstance(sim_result, SimResult) else dict(sim_result)
    mi = map_info(map)
    prof = _resolve_profile(profile, catalog)
    bi = dict(budget_info) if budget_info is not None else profile_budget(prof, mi, catalog=catalog)
    notes: list[str] = []
    if calibration is None:
        cal = _default_efficiency()
    elif isinstance(calibration, Mapping):
        cal = {"value": _num(calibration.get("value")), "source": str(calibration.get("source") or
                                                                      "assumed: llama_efficiency given by the caller")}
    elif isinstance(calibration, (tuple, list)) and len(calibration) == 2:
        cal = {"value": _num(calibration[0]), "source": str(calibration[1])}
    else:
        cal = {"value": _num(calibration), "source": "assumed: llama_efficiency given by the caller"}
    eff = cal["value"]
    if eff is None or eff <= 0:
        raise CacheSimError(f"llama_efficiency must be a positive number, got {cal['value']!r}")
    mem_bw = _metric(prof.get("mem_bw_gbps"))
    st = prof.get("storage") or {}
    rr = _metric(st.get("rand_read_3mb_mbps"))
    acc = _metric(st.get("access_ms"))
    if mem_bw is None or mem_bw <= 0 or rr is None or rr <= 0 or acc is None or acc < 0:
        raise CacheSimError(f"profile {prof.get('name')!r} lacks mem_bw_gbps / storage.rand_read_3mb_mbps / access_ms")
    blk = run.get("steady") or {}
    window = "steady"
    if not blk.get("n_tokens"):
        blk = run.get("overall") or {}
        window = "overall"
        notes.append("steady window empty: projection uses the overall block (includes warm-up)")
    miss = _num((blk.get("miss_bytes_per_token") or {}).get("mean")) or 0.0
    n_miss = _num(blk.get("n_miss_per_token")) or 0.0
    hit_b = _num(blk.get("hit_bytes_per_token")) or 0.0
    pf_b = _num(blk.get("prefetch_bytes_per_token")) or 0.0
    pf_i = _num(blk.get("prefetch_items_per_token")) or 0.0
    pf_bl = blk.get("prefetch_bytes_per_token_by_layer")
    pf_il = blk.get("prefetch_items_per_token_by_layer")
    resident = float(mi.nonexpert_bytes_per_token) + hit_b
    bw_eff = mem_bw * 1e9 * eff
    rr_bps = rr * 1e6
    t_ram = resident / bw_eff
    t_io_demand = miss / rr_bps + n_miss * acc / 1e3
    t_io_prefetch = pf_b / rr_bps + pf_i * acc / 1e3
    t_io = t_io_demand + t_io_prefetch
    n_layers = max(1, mi.n_moe_layers)
    t_layer = t_ram / n_layers
    if pf_b > 0 and isinstance(pf_bl, list) and isinstance(pf_il, list) and len(pf_bl) == len(pf_il):
        overlapped = sum(min((_num(b) or 0.0) / rr_bps + (_num(c) or 0.0) * acc / 1e3, t_layer)
                         for b, c in zip(pf_bl, pf_il))
    else:
        overlapped = min(t_io_prefetch, t_ram)
    tg_serial = _inv(t_ram + t_io - overlapped)
    tg_overlap = _inv(max(t_ram, t_io))
    floor = bw_eff / float(mi.nonexpert_bytes_per_token + mi.expert_bytes_per_token_worst) \
        if (mi.nonexpert_bytes_per_token + mi.expert_bytes_per_token_worst) > 0 else None
    good, viable = _thresholds(catalog)
    verdict = classify(tg_serial, tg_overlap, bi["expert_cache_bytes"], mi.expert_bytes, good=good, viable=viable)
    if int(run.get("budget_bytes", -1)) != int(bi["budget_bytes"]):
        notes.append(f"simulated budget {run.get('budget_bytes')} B differs from the profile's expert cache "
                     f"{bi['budget_bytes']} B")
    if bi["infeasible"]:
        notes.append("no room for an expert cache: non-expert weights + KV + buffers exceed RAM (simulated at 0)")
    srcs = bi.get("sources", {})
    policy = run.get("policy")
    hit_rate = _num(blk.get("hit_rate"))
    assumptions = {
        "mem_bw_gbps": _assume(mem_bw, "GB/s", _src(prof.get("mem_bw_gbps"), "assumed: catalog device mem_bw_gbps")),
        "llama_efficiency": _assume(eff, "fraction", cal["source"]),
        "rand_read_3mb_mbps": _assume(rr, "MB/s", _src(st.get("rand_read_3mb_mbps"), "assumed: catalog storage")),
        "access_ms": _assume(acc, "ms", _src(st.get("access_ms"), "assumed: catalog storage")),
        "os_reserve_bytes": _assume(bi["os_reserve_bytes"], "B", srcs.get("os_reserve_bytes", "assumed")),
        "kv_bytes": _assume(bi["kv_bytes"], "B", srcs.get("kv_bytes", "assumed")),
        "compute_buffers_bytes": _assume(bi["compute_buffers_bytes"], "B", srcs.get("compute_buffers_bytes", "assumed")),
        "ram_bytes": _assume(bi["ram_bytes"], "B", srcs.get("ram_bytes", "assumed")),
        "n_ctx": _assume(bi["n_ctx"], "tokens", srcs.get("n_ctx", "assumed")),
        "nonexpert_bytes_per_token": _assume(int(mi.nonexpert_bytes_per_token), "B",
                                             f"measured: GGUF header ({mi.label} summary.nonexpert_bytes_per_token)"),
        "expert_hit_rate": _assume(hit_rate, "fraction",
                                   f"simulated: {policy} {window} hit rate on the routing trace "
                                   f"(records {(blk.get('window') or [None, None])[0]}..{(blk.get('window') or [None, None])[1]})"),
    }
    return _jsonable({
        "profile": prof.get("name"),
        "resident_bytes_per_token": resident,
        "hit_bytes_per_token": hit_b,
        "miss_bytes_per_token": miss,
        "n_miss_per_token": n_miss,
        "prefetch_bytes_per_token": pf_b,
        "t_ram_s": t_ram,
        "t_io_s": t_io,
        "t_io_demand_s": t_io_demand,
        "t_io_prefetch_s": t_io_prefetch,
        "overlapped_s": overlapped,
        "tg_serial": tg_serial,
        "tg_overlap": tg_overlap,
        "floor_all_in_ram_tok_s": floor,
        "verdict": verdict,
        "kind": "projected",
        "expert_cache_bytes": bi["expert_cache_bytes"],
        "budget_bytes": bi["budget_bytes"],
        "window": window,
        "thresholds": {"good_stream_tok_s": good, "viable_stream_tok_s": viable, "weak_stream_tok_s": WEAK_TOK_S},
        "assumptions": assumptions,
        "notes": notes,
    })


def dense_line(catalog: Mapping[str, Any], profiles: Sequence[Mapping[str, Any]],
               efficiencies: Mapping[str, Mapping[str, Any]] | None = None, *,
               dense_id: str = DENSE_MODEL_ID, dense_map: Any = None,
               reference: str = REFERENCE_PROFILE) -> dict[str, Any] | None:
    """The dense-model reference line: every token reads the whole file (no expert cache helps).

    bytes_per_token = file data bytes (map summary.file_data_bytes, else catalog bytes -
    data_offset). Per profile: tok/s in RAM at mem_bw x llama_efficiency (also without the
    efficiency factor), s/token and tok/s streamed at seq_read_mbps, and whether file + KV + OS +
    buffers fit in RAM. Returns None when the dense model is not in the catalog.
    """
    model = next((m for m in catalog.get("models", []) or [] if m.get("id") == dense_id), None)
    if model is None:
        log.warning("dense model %r not in the catalog; dense_31b_line omitted", dense_id)
        return None
    kv_fn = None
    if dense_map is not None:
        dm = dense_map if isinstance(dense_map, Mapping) else gm.load_map(dense_map)
        s = dm.get("summary") or {}
        bpt = int(_num(s.get("file_data_bytes")) or _num(s.get("nonexpert_bytes_per_token")) or 0)
        bsrc = f"measured: dense map summary.file_data_bytes ({paths.repo_relative(dense_map) if not isinstance(dense_map, Mapping) else '<dict>'})"
        if all(k in dm for k in ("head_count_kv", "key_length", "key_length_swa", "sliding_pattern")):
            def kv_fn(ctx: int) -> int:  # noqa: E306
                return gm.kv_bytes_for(dm["head_count_kv"], int(dm["key_length"]), int(dm["key_length_swa"]),
                                       dm["sliding_pattern"], dm.get("sliding_window"), ctx, KV_CACHE_TYPE,
                                       dm.get("value_length"), dm.get("value_length_swa"))
    else:
        b = _num(model.get("bytes"))
        off = _num(model.get("data_offset"))
        if b is not None and off is not None:
            bpt = int(b - off)
            bsrc = f"measured: catalog {dense_id} bytes - data_offset (file data bytes)"
        else:
            bpt = int(_num(model.get("nonexpert_bytes_per_token")) or _num(model.get("bytes")) or 0)
            bsrc = f"measured: catalog {dense_id} nonexpert_bytes_per_token"
    if bpt <= 0:
        return None
    file_bytes = int(_num(model.get("bytes")) or bpt)
    per: dict[str, Any] = {}
    for prof in profiles:
        name = str(prof.get("name"))
        bw = _metric(prof.get("mem_bw_gbps"))
        st = prof.get("storage") or {}
        seqr = _metric(st.get("seq_read_mbps"))
        cal = (efficiencies or {}).get(name) or _default_efficiency()
        eff = float(cal["value"])
        ctx = int(_num(prof.get("n_ctx")) or DEFAULT_N_CTX)
        kv = int(kv_fn(ctx)) if kv_fn is not None else int(feas.kv_bytes(model, ctx))
        need = file_bytes + kv + int(_num(prof.get("os_reserve_bytes")) or 0) + int(_num(prof.get("compute_buffers_bytes")) or 0)
        fits = bool(_num(prof.get("ram_bytes")) is not None and need <= float(prof["ram_bytes"]))
        tok_ram = bw * 1e9 * eff / bpt if bw else None
        s_tok = bpt / (seqr * 1e6) if seqr else None
        per[name] = {
            "mem_bw_gbps": bw, "llama_efficiency": eff, "seq_read_mbps": seqr,
            "tok_s_at_mem_bw": tok_ram, "tok_s_at_mem_bw_raw": (bw * 1e9 / bpt) if bw else None,
            "s_per_token_at_storage_bw": s_tok, "tok_s_at_storage_bw": (1.0 / s_tok) if s_tok else None,
            "fits_ram": fits, "ram_needed_bytes": need, "kv_bytes": kv,
            "tok_s_effective": tok_ram if fits else ((1.0 / s_tok) if s_tok else None),
            "mode": "in RAM" if fits else "streamed from storage",
            "assumptions": {
                "mem_bw_gbps": _assume(bw, "GB/s", _src(prof.get("mem_bw_gbps"), "assumed: catalog")),
                "llama_efficiency": _assume(eff, "fraction", cal["source"]),
                "seq_read_mbps": _assume(seqr, "MB/s", _src(st.get("seq_read_mbps"), "assumed: catalog")),
            },
        }
    if not per:
        return None
    ref = reference if reference in per else next(
        (str(p.get("name")) for p in profiles if str(p.get("platform", "")).lower() == "pi5"), next(iter(per)))
    return _jsonable({
        "model": dense_id,
        "bytes_per_token": bpt,
        "tok_s_at_mem_bw": per[ref]["tok_s_at_mem_bw"],
        "s_per_token_at_storage_bw": per[ref]["s_per_token_at_storage_bw"],
        "reference_profile": ref,
        "kind": "projected",
        "bytes_source": bsrc,
        "per_profile": per,
        "note": "every decoded token reads the whole dense file: no cache policy changes these numbers",
    })


# --------------------------------------------------------------------------------------------
# Full run (the CLI's engine; also usable from tests)
# --------------------------------------------------------------------------------------------
def _stack_check(prep: PreparedTrace, runs: Sequence[SimResult], uniform: bool) -> dict[str, Any]:
    lru = [r for r in runs if r.policy == "lru"]
    if not lru:
        return {"ran": False, "reason": "no lru runs"}
    if not uniform:
        return {"ran": False, "reason": "variable expert sizes: byte-budgeted LRU is not a fixed-capacity LRU"}
    hist = tr.stack_distance_hist(prep.items, prep.n_items)
    rows = []
    for r in lru:
        ref = tr.lru_hit_rate(hist, r.capacity_items)
        sim = r.overall.get("hit_rate")
        diff = abs(float(sim) - ref) if sim is not None else None
        rows.append({"capacity_items": r.capacity_items, "profile": r.profile, "sim_overall_hit_rate": sim,
                     "stack_distance_hit_rate": ref, "abs_diff": diff})
    diffs = [x["abs_diff"] for x in rows if x["abs_diff"] is not None]
    mx = max(diffs) if diffs else None
    return {"ran": True, "tolerance": 1e-9, "max_abs_diff": mx, "ok": bool(mx is None or mx <= 1e-9), "points": rows}


def _comparisons(runs: Sequence[SimResult]) -> list[dict[str, Any]]:
    """E4 decision inputs per budget point, from the common out-of-sample ``test`` window."""
    groups: dict[tuple, dict[str, SimResult]] = {}
    for r in runs:
        key = (r.profile, r.budget_bytes) if r.profile else (None, r.budget_bytes)
        groups.setdefault(key, {})[r.policy] = r
    out = []
    for (prof, bud), pols in groups.items():
        def miss(p: str) -> float | None:
            r = pols.get(p)
            return _num(((r.test or {}).get("miss_bytes_per_token") or {}).get("mean")) if r else None
        lru, bel, sta = miss("lru"), miss("belady"), miss("static")
        gap = None
        if lru is not None and bel is not None:
            gap = (lru / bel) if bel > 0 else (1.0 if lru == 0 else None)
        st_ratio = None
        if lru is not None and sta is not None:
            st_ratio = (sta / lru) if lru > 0 else (1.0 if sta == 0 else None)
        pf = pols.get("lru_prefetch")
        out.append({
            "profile": prof, "budget_bytes": int(bud), "budget_gb": bud / 1e9,
            "window": next(iter(pols.values())).test_window,
            "lru_miss_bytes_per_token": lru, "belady_miss_bytes_per_token": bel, "static_miss_bytes_per_token": sta,
            "belady_gap": gap,
            "prefetch_or_static_worth_implementing": (gap > BELADY_GAP_THRESHOLD) if gap is not None else None,
            "static_over_lru": st_ratio,
            "static_within_10pct_of_lru": (st_ratio <= 1.0 + STATIC_WITHIN_FRACTION) if st_ratio is not None else None,
            "lru_prefetch_stall_bytes_per_token": _num((pf.test or {}).get("stall_bytes_per_token")) if pf else None,
            "lru_prefetch_io_bytes_per_token": _num((pf.test or {}).get("io_bytes_per_token")) if pf else None,
        })
    out.sort(key=lambda d: (d["profile"] is not None, d["budget_bytes"]))
    return out


def _resolve_io_mem(spec: str | None) -> Path | None:
    if spec is None or str(spec).lower() == "none":
        return None
    if str(spec).lower() == "auto":
        return DEFAULT_IO_MEM if DEFAULT_IO_MEM.is_file() else None
    p = Path(spec)
    if not p.is_file():
        raise FileNotFoundError(f"--io-mem {spec}: no such file")
    return p


def run_simulation(trace_path: str | os.PathLike[str] | PreparedTrace, map_path: Any, *,
                   catalog: str | os.PathLike[str] | Mapping[str, Any] | None = None,
                   policies: Sequence[str] = DEFAULT_POLICIES, budgets_gb: Sequence[float] | None = None,
                   profiles: str | Sequence[str] | None = None, train_split: float = DEFAULT_TRAIN_SPLIT,
                   warmup_tokens: int = DEFAULT_WARMUP_TOKENS, read_granularity: int = DEFAULT_READ_GRANULARITY,
                   read_model: str = "round", lfu_decay: float = DEFAULT_LFU_DECAY, prefetch_n: int | None = None,
                   token_range: str | Sequence[int | None] | None = None, io_mem: str | None = "auto",
                   calibration: str | None = "auto", llama_efficiency: float | None = None,
                   e0_dir: str | os.PathLike[str] | None = None, belady_bypass: bool = True,
                   stack_check: bool = True, dense_model: str = DENSE_MODEL_ID,
                   dense_map: Any = None) -> tuple[dict[str, Any], list[SimResult], PreparedTrace]:
    """Run every (policy, budget) and (policy, profile) simulation; returns (sim.json doc, runs, prep).

    ``calibration``: 'auto' (results/pi/calibration.json for Pi profiles + desktop E0 from
    ``e0_dir``), 'none' (0.65 assumed), or a path to a calibration.json. ``io_mem``: 'auto',
    'none' or a path (merged into the catalog devices as measured).
    """
    t0 = time.perf_counter()
    pol_list = [p.strip().lower() for p in policies if str(p).strip()]
    bad = [p for p in pol_list if p not in POLICIES]
    if bad or not pol_list:
        raise CacheSimError(f"unknown policies {bad or '(none)'}; known: {', '.join(POLICIES)}")
    io_path = _resolve_io_mem(io_mem) if not isinstance(catalog, Mapping) else None
    if isinstance(catalog, Mapping):
        cat = dict(catalog)
        cat_label = str(cat.get("_path", "<dict>"))
    else:
        cat_path = Path(catalog) if catalog is not None else paths.CATALOG_PATH
        cat = feas.load_catalog(cat_path, io_mem=io_path)
        cat_label = paths.repo_relative(cat_path)
    prep = prepare(trace_path, token_range)
    mi = map_info(map_path)
    mi.check_trace(prep.L, prep.K, prep.E, prep.moe_layers)
    model_id = mi.model_id(cat)
    notes: list[str] = []
    if prep.gguf_basename and mi.gguf_basename and prep.gguf_basename != mi.gguf_basename:
        notes.append(f"trace recorded on {prep.gguf_basename} but simulated against {mi.gguf_basename}: "
                     "routing assumed identical across quants (an assumption)")
    if prep.truncated:
        notes.append("trace was not closed (truncated): record count inferred from the file size")
    if budgets_gb is None and profiles is None:
        budgets_gb = list(DEFAULT_BUDGETS_GB)
        profiles = "all"
    budget_list = [float(b) for b in (budgets_gb or [])]
    prof_list = select_profiles(cat, profiles) if profiles else []

    # efficiency per profile
    explicit = None
    pi_cal = desktop_cal = None
    cal_files: list[str] = []
    if llama_efficiency is not None:
        if not llama_efficiency > 0:
            raise CacheSimError("--llama-efficiency must be > 0")
        explicit = {"value": float(llama_efficiency),
                    "source": f"assumed: --llama-efficiency {llama_efficiency} given on the command line", "files": []}
    elif calibration is not None and str(calibration).lower() != "none":
        if str(calibration).lower() == "auto":
            pi_cal = _load_pi_calibration(DEFAULT_PI_CALIBRATION)
            desktop_cal = efficiency_from_e0(e0_dir if e0_dir is not None else DEFAULT_E0_DIR, cat,
                                             prefer_model=model_id)
        else:
            cp = Path(str(calibration))
            if not cp.exists():
                raise FileNotFoundError(f"--calibration {calibration}: no such file")
            if cp.is_dir():
                desktop_cal = efficiency_from_e0(cp, cat, prefer_model=model_id)
            else:
                pi_cal = _load_pi_calibration(cp)
                explicit = pi_cal
            if e0_dir is not None and desktop_cal is None:
                desktop_cal = efficiency_from_e0(e0_dir, cat, prefer_model=model_id)
    effs = {str(p.get("name")): efficiency_for_profile(p, explicit=explicit, pi_cal=pi_cal, desktop_cal=desktop_cal)
            for p in prof_list}
    for c in (pi_cal, desktop_cal, explicit):
        if c:
            cal_files.extend(f for f in c.get("files", []) if f and f not in cal_files)

    common = dict(warmup_tokens=warmup_tokens, read_granularity=read_granularity, read_model=read_model,
                  train_split=train_split, lfu_decay=lfu_decay, prefetch_n=prefetch_n, belady_bypass=belady_bypass)
    runs: list[SimResult] = []
    budgets_info = {str(p.get("name")): profile_budget(p, mi, catalog=cat) for p in prof_list}
    for pol in pol_list:
        for bgb in budget_list:
            r = simulate(prep, mi, pol, int(round(bgb * 1e9)), **common)
            runs.append(r)
            _log_run(r, f"{bgb:g} GB")
        for p in prof_list:
            name = str(p.get("name"))
            bi = budgets_info[name]
            r = simulate(prep, mi, pol, bi["budget_bytes"], profile=name, **common)
            r.budget = {k: v for k, v in bi.items() if k != "sources"}
            r.projection = project(r, p, mi, effs[name], catalog=cat, budget_info=bi)
            runs.append(r)
            _log_run(r, name)
    checks = {"lru_vs_stack_distance": _stack_check(prep, runs, mi.uniform) if stack_check else {"ran": False,
                                                                                                    "reason": "--no-stack-check"}}
    dense = dense_line(cat, prof_list or [dict(d) for d in cat.get("devices", []) or []],
                       effs or {str(d.get("name")): efficiency_for_profile(d, explicit=explicit, pi_cal=pi_cal,
                                                                           desktop_cal=desktop_cal)
                                for d in cat.get("devices", []) or []},
                       dense_id=dense_model, dense_map=dense_map)
    cost_np, rm_eff = mi.read_costs(int(read_granularity), read_model)
    doc: dict[str, Any] = {
        "trace": paths.repo_relative(prep.path),
        "map": mi.label,
        "catalog": cat_label,
        "token_range": [prep.start, prep.stop],
        "warmup_tokens": int(warmup_tokens),
        "read_granularity": int(read_granularity),
        "train_split": float(train_split),
        "runs": [r.to_dict() for r in runs],
        "dense_31b_line": dense,
        "format_version": SIM_FORMAT_VERSION,
        "generator": GENERATOR,
        "created_at": _utc_now(),
        "kind": "simulated",
        "n_tokens": prep.n,
        "n_records_total": prep.n_records_total,
        "trace_gguf": prep.gguf_basename,
        "trace_llama_build": prep.header.get("llama_build"),
        "trace_llama_commit": prep.header.get("llama_commit"),
        "model": {"id": model_id, "file": mi.gguf_basename, "sha_prefix": mi.sha256_prefix, "bytes": mi.file_size},
        "shape": {"n_moe_layers": prep.L, "n_expert": prep.E, "n_expert_used": prep.K, "n_items": prep.n_items},
        "bytes": {
            "unit_bytes": mi.unit_bytes[0] if mi.uniform else None,
            "unit_bytes_per_layer": None if mi.uniform else mi.unit_bytes,
            "read_cost_bytes_mean": float(cost_np.mean()) if cost_np.size else None,
            "read_inflation": float(cost_np.mean() / mi.item_sizes().mean()) if cost_np.size else None,
            "read_model": rm_eff,
            "nonexpert_bytes": mi.nonexpert_bytes, "nonexpert_bytes_per_token": mi.nonexpert_bytes_per_token,
            "expert_bytes": mi.expert_bytes, "expert_bytes_per_token_worst": mi.expert_bytes_per_token_worst,
        },
        "policies": pol_list,
        "budgets_gb": budget_list,
        "profiles": [str(p.get("name")) for p in prof_list],
        "lfu_decay": float(lfu_decay),
        "prefetch_n": prefetch_n if prefetch_n is not None else prep.K,
        "belady_bypass": bool(belady_bypass),
        "llama_efficiency": {k: {"value": v["value"], "source": v["source"]} for k, v in effs.items()},
        "io_mem": _rel(io_path),
        "calibration_files": cal_files,
        "checks": checks,
        "summary": {"comparisons": _comparisons(runs),
                    "rules": {"belady_gap": f"LRU bytes/token > {BELADY_GAP_THRESHOLD}x Belady -> prefetch/static "
                                            "policies worth implementing on the Pi",
                              "static": f"static within {int(STATIC_WITHIN_FRACTION * 100)}% of LRU -> hot-set "
                                        "pinning is the simpler deployment",
                              "verdict": "STREAM_GOOD: tg_overlap >= good and tg_serial >= viable; STREAM_WEAK: "
                                         f"tg_serial >= {WEAK_TOK_S}; FITS_RAM: all experts fit; INFEASIBLE: no room"}},
        "notes": notes,
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    return _jsonable(doc), runs, prep


def _log_run(r: SimResult, label: str) -> None:
    ss = r.steady
    mb = (ss.get("miss_bytes_per_token") or {}).get("mean")
    proj = r.projection or {}
    extra = ""
    if proj:
        ts, to = proj.get("tg_serial"), proj.get("tg_overlap")
        extra = (f", tg_serial {ts:.3f}" if ts is not None else ", tg_serial inf") + \
                (f" tg_overlap {to:.3f}" if to is not None else " tg_overlap inf") + f" -> {proj.get('verdict')}"
    log.info("%-12s %-20s cap %5d items: hit %s (steady), %s MB/token%s [%.1f s]", r.policy, label, r.capacity_items,
             f"{ss['hit_rate']:.4f}" if ss.get("hit_rate") is not None else "n/a",
             f"{mb / 1e6:.1f}" if mb is not None else "n/a", extra, r.wall_s)


# --------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------
def sim_csv_text(doc: Mapping[str, Any]) -> str:
    """sim.csv (TRACE_FORMAT 11.2): exact header, empty cells for null, 6 significant digits."""
    lines = [SIM_CSV_HEADER]
    for run in doc.get("runs", []):
        ss = run.get("steady") or {}
        mb = ss.get("miss_bytes_per_token") or {}
        pr = run.get("projection") or {}
        mean, p95 = _num(mb.get("mean")), _num(mb.get("p95"))
        cells = [
            str(run.get("policy", "")),
            _fmt6(_num(run.get("budget_bytes")) / 1e9 if _num(run.get("budget_bytes")) is not None else None),
            str(run.get("profile") or ""),
            _fmt6(ss.get("hit_rate")),
            _fmt6(ss.get("weighted_hit_rate")),
            _fmt6(mean / 1e6 if mean is not None else None),
            _fmt6(p95 / 1e6 if p95 is not None else None),
            _fmt6(pr.get("tg_serial")),
            _fmt6(pr.get("tg_overlap")),
            str(pr.get("verdict") or ""),
        ]
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def _write_series(path: Path, runs: Sequence[SimResult], prep: PreparedTrace) -> Path:
    arrays: dict[str, np.ndarray] = {"token_index": np.arange(prep.start, prep.stop, dtype=np.int64),
                                     "phase": prep.phase.astype(np.uint8)}
    labels = []
    for i, r in enumerate(runs):
        labels.append(f"{r.policy}|{r.budget_bytes}|{r.profile or ''}")
        for k, v in (r.series or {}).items():
            arrays[f"r{i:03d}_{k}"] = v
    arrays["labels"] = np.asarray(labels, dtype=np.str_)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def _model_block(mi: MapInfo, catalog: Mapping[str, Any], model_id: str | None) -> dict[str, Any]:
    if model_id and model_id in paths.MODELS:
        blk = paths.model_identity(model_id, with_sha=False)
    else:
        m = next((x for x in catalog.get("models", []) or [] if x.get("id") == model_id), None) if model_id else None
        blk = {"id": model_id or (Path(mi.gguf_basename).stem if mi.gguf_basename else None),
               "file": (m or {}).get("local_path") or (m or {}).get("file") or mi.gguf_basename,
               "quant": (m or {}).get("quant") or (paths.guess_quant(mi.gguf_basename) if mi.gguf_basename else None),
               "bytes": (m or {}).get("bytes") or mi.file_size, "sha_prefix": None}
    if mi.sha256_prefix:
        blk["sha_prefix"] = mi.sha256_prefix
    return blk


def write_envelopes(doc: Mapping[str, Any], runs: Sequence[SimResult], mi: MapInfo, catalog: Mapping[str, Any],
                    out_dir: str | os.PathLike[str], *, tag: str, inputs: Sequence[str]) -> list[Path]:
    """One E4 'projected' envelope per profile run, plus one per profile for the dense line."""
    rlog = logging.getLogger("pigemma.results")
    old_level = rlog.level
    if rlog.getEffectiveLevel() < logging.WARNING:
        rlog.setLevel(logging.WARNING)  # one INFO line per envelope would drown the run log
    try:
        return _write_envelopes(doc, runs, mi, catalog, Path(out_dir), tag=tag, inputs=inputs)
    finally:
        rlog.setLevel(old_level)


def _write_envelopes(doc: Mapping[str, Any], runs: Sequence[SimResult], mi: MapInfo, catalog: Mapping[str, Any],
                     out: Path, *, tag: str, inputs: Sequence[str]) -> list[Path]:
    written: list[Path] = []
    model_id = mi.model_id(catalog)
    mblk = _model_block(mi, catalog, model_id)
    build = _num(doc.get("trace_llama_build"))
    commit = doc.get("trace_llama_commit")
    safe = lambda s: re.sub(r"[^A-Za-z0-9.\-]+", "-", str(s)).strip("-.") or "na"  # noqa: E731
    for r in runs:
        if not r.profile or not r.projection:
            continue
        pr = r.projection
        ss = r.steady
        name = f"E4_{safe(model_id or 'model')}_{safe(tag)}_{safe(r.policy)}_{safe(r.profile)}.json"
        io_b = _num(ss.get("io_bytes_per_token"))
        fields = dict(
            experiment="E4", kind="projected", llama_build=int(build) if build is not None else None,
            llama_commit=str(commit) if commit else None, model=dict(mblk), variant=f"{r.policy}@{r.profile}",
            flags=[], threads=None, n_ctx=(r.budget or {}).get("n_ctx"),
            tg={"tok_s": pr.get("tg_serial"), "sd": None, "n": None},
            read_bytes_per_token=io_b,
            assumptions=pr.get("assumptions"),
            inputs=list(inputs),
            notes=[f"verdict {pr.get('verdict')} (tg_serial {_fmt6(pr.get('tg_serial'))}, tg_overlap "
                   f"{_fmt6(pr.get('tg_overlap'))} tok/s); tg.tok_s is tg_serial",
                   "projection for llama.cpp -nr streaming mode; not observed on hardware"] + list(pr.get("notes", [])),
            extra={"sim_policy": r.policy, "profile": r.profile, "verdict": pr.get("verdict"),
                   "tg_overlap": pr.get("tg_overlap"), "tg_serial": pr.get("tg_serial"),
                   "floor_all_in_ram_tok_s": pr.get("floor_all_in_ram_tok_s"),
                   "hit_rate_steady": ss.get("hit_rate"), "weighted_hit_rate_steady": ss.get("weighted_hit_rate"),
                   "miss_bytes_per_token": ss.get("miss_bytes_per_token"),
                   "prefetch_bytes_per_token": ss.get("prefetch_bytes_per_token"),
                   "budget_bytes": r.budget_bytes, "capacity_items": r.capacity_items,
                   "expert_cache_bytes": pr.get("expert_cache_bytes"), "t_ram_s": pr.get("t_ram_s"),
                   "t_io_s": pr.get("t_io_s"), "steady_window": r.steady_window, "sim_json": doc.get("_out")},
        )
        try:
            written.append(rs.write_result(out / name, **_jsonable(fields)))
        except rs.ResultSchemaError as e:
            log.error("envelope %s not written: %s", name, e)
    dense = doc.get("dense_31b_line")
    if isinstance(dense, Mapping):
        did = str(dense.get("model"))
        dblk: dict[str, Any]
        if did in paths.MODELS:
            dblk = paths.model_identity(did, with_sha=False)
        else:
            dm = next((x for x in catalog.get("models", []) or [] if x.get("id") == did), {})
            dblk = {"id": did, "file": dm.get("local_path") or dm.get("file"), "quant": dm.get("quant"),
                    "bytes": dm.get("bytes"), "sha_prefix": None}
        for pname, pp in (dense.get("per_profile") or {}).items():
            name = f"E4_{safe(did)}_{safe(tag)}_dense_{safe(pname)}.json"
            fits = bool(pp.get("fits_ram"))
            fields = dict(
                experiment="E4", kind="projected", llama_build=int(build) if build is not None else None,
                llama_commit=str(commit) if commit else None, model=dblk, variant=f"dense-stream@{pname}",
                flags=[], threads=None, n_ctx=None,
                tg={"tok_s": pp.get("tok_s_effective"), "sd": None, "n": None},
                read_bytes_per_token=0 if fits else dense.get("bytes_per_token"),
                assumptions={**pp.get("assumptions", {}),
                             "bytes_per_token": _assume(dense.get("bytes_per_token"), "B", str(dense.get("bytes_source")))},
                inputs=[str(doc.get("catalog"))],
                notes=[f"dense model {pp.get('mode')}: every token reads {_fmt6((dense.get('bytes_per_token') or 0) / 1e9)} GB",
                       str(dense.get("note"))],
                extra={k: v for k, v in pp.items() if k != "assumptions"} | {"profile": pname},
            )
            try:
                written.append(rs.write_result(out / name, **_jsonable(fields)))
            except rs.ResultSchemaError as e:
                log.error("envelope %s not written: %s", name, e)
    return written


def write_outputs(doc: dict[str, Any], runs: Sequence[SimResult], prep: PreparedTrace, mi: MapInfo,
                  catalog: Mapping[str, Any], *, out: str | os.PathLike[str], csv_path: str | os.PathLike[str] | None = None,
                  series_path: str | os.PathLike[str] | None = None, results_dir: str | os.PathLike[str] | None = None,
                  envelopes: bool = True) -> dict[str, Any]:
    """Write sim.json (+ sim.csv, series .npz, E4 envelopes). Returns {kind: path(s)}."""
    outp = Path(out)
    doc["_out"] = paths.repo_relative(outp)
    written: dict[str, Any] = {}
    if envelopes and any(r.projection for r in runs):
        rd = Path(results_dir) if results_dir is not None else outp.parent
        inputs = [x for x in [doc.get("trace"), doc.get("map"), doc.get("catalog"), doc.get("io_mem"),
                              *doc.get("calibration_files", []), paths.repo_relative(outp)] if x]
        env = write_envelopes(doc, runs, mi, catalog, rd, tag=outp.stem, inputs=inputs)
        written["envelopes"] = [paths.repo_relative(p) for p in env]
        doc["envelopes"] = written["envelopes"]
    doc.pop("_out", None)
    _atomic_write_text(outp, _dumps(doc))
    written["sim"] = paths.repo_relative(outp)
    if csv_path is not None:
        cp = Path(csv_path)
        _atomic_write_text(cp, sim_csv_text(doc))
        written["csv"] = paths.repo_relative(cp)
    if series_path is not None:
        written["series"] = paths.repo_relative(_write_series(Path(series_path), runs, prep))
    return written


def format_table(doc: Mapping[str, Any]) -> str:
    """Compact text table of the runs (stdout of the CLI)."""
    hdr = f"{'policy':<13}{'budget GB':>10}  {'profile':<20}{'cap':>6}{'hit_ss':>8}{'w_hit':>8}{'MB/tok':>9}" \
          f"{'p95 MB':>9}{'tg_ser':>8}{'tg_ovl':>8}  verdict"
    lines = [hdr, "-" * len(hdr)]
    for r in doc.get("runs", []):
        ss = r.get("steady") or {}
        mb = ss.get("miss_bytes_per_token") or {}
        pr = r.get("projection") or {}

        def f(v: Any, w: int, fmt: str) -> str:
            x = _num(v)
            return f"{'-':>{w}}" if x is None else f"{x:>{w}{fmt}}"
        lines.append(
            f"{r.get('policy', ''):<13}{f(r.get('budget_bytes', 0) / 1e9, 10, '.3f')}  {str(r.get('profile') or ''):<20}"
            f"{r.get('capacity_items', 0):>6}{f(ss.get('hit_rate'), 8, '.4f')}{f(ss.get('weighted_hit_rate'), 8, '.4f')}"
            f"{f((_num(mb.get('mean')) or 0) / 1e6 if mb.get('mean') is not None else None, 9, '.1f')}"
            f"{f((_num(mb.get('p95')) or 0) / 1e6 if mb.get('p95') is not None else None, 9, '.1f')}"
            f"{f(pr.get('tg_serial'), 8, '.3f')}{f(pr.get('tg_overlap'), 8, '.3f')}  {pr.get('verdict') or ''}")
    dense = doc.get("dense_31b_line")
    if isinstance(dense, Mapping):
        lines.append("")
        lines.append(f"dense {dense.get('model')}: {(dense.get('bytes_per_token') or 0) / 1e9:.2f} GB/token "
                     f"(PROJECTED, {dense.get('reference_profile')}): "
                     f"{_fmt6(dense.get('tok_s_at_mem_bw'))} tok/s if in RAM, "
                     f"{_fmt6(dense.get('s_per_token_at_storage_bw'))} s/token streamed")
    chk = (doc.get("checks") or {}).get("lru_vs_stack_distance") or {}
    if chk.get("ran"):
        lines.append(f"LRU vs stack-distance curve: max |diff| {chk.get('max_abs_diff')} "
                     f"({'OK' if chk.get('ok') else 'MISMATCH'})")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# E5/E6 page-cache validation (io.csv vs simulator)
# --------------------------------------------------------------------------------------------
def read_io_csv(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Parse a probe --io-csv file (TRACE_FORMAT 12.2) into column arrays. Raises CacheSimError."""
    p = Path(path)
    with open(p, encoding="utf-8-sig", newline="") as f:
        lines = [ln.rstrip("\r\n") for ln in f]
    if not lines or lines[0].strip() != IO_CSV_HEADER:
        raise CacheSimError(f"{p}: first line must be exactly {IO_CSV_HEADER!r}")
    rows = []
    for i, ln in enumerate(lines[1:], start=2):
        if not ln.strip():
            continue
        parts = ln.split(",")
        if len(parts) != len(IO_CSV_COLUMNS):
            raise CacheSimError(f"{p}:{i}: expected {len(IO_CSV_COLUMNS)} fields, got {len(parts)}")
        try:
            rows.append([float(x) for x in parts])
        except ValueError as e:
            raise CacheSimError(f"{p}:{i}: non-numeric field ({e})") from e
    a = np.asarray(rows, dtype=np.float64).reshape(-1, len(IO_CSV_COLUMNS))
    out = {c: a[:, k] for k, c in enumerate(IO_CSV_COLUMNS)}
    for c in IO_CSV_COLUMNS:
        if c != "wall_ms":
            out[c] = out[c].astype(np.int64)
    if out["token_index"].size > 1 and np.any(np.diff(out["token_index"]) <= 0):
        raise CacheSimError(f"{p}: token_index must be strictly increasing")
    return out


def measure_io(io: Mapping[str, np.ndarray] | str | os.PathLike[str], warmup_tokens: int = DEFAULT_IO_WARMUP, *,
               page_size: int = DEFAULT_IO_PAGE_SIZE, bind_mount: bool = False) -> dict[str, Any]:
    """Measured bytes/token, majflt/token and tok/s over rows with token_index >= warmup_tokens.

    A row covers token_index[r] - token_index[r-1] tokens (the first row token_index[0] + 1).
    bind_mount: bytes = majflt x page_size (lower bound; 9p reads are not in read_bytes).
    """
    d = read_io_csv(io) if isinstance(io, (str, os.PathLike)) else io
    ti = np.asarray(d["token_index"], dtype=np.int64)
    if ti.size == 0:
        raise CacheSimError("io.csv has no rows")
    covered = np.empty_like(ti)
    covered[0] = ti[0] + 1
    covered[1:] = np.diff(ti)
    keep = ti >= int(warmup_tokens)
    tokens = int(covered[keep].sum())
    rb = np.asarray(d["majflt_delta"], dtype=np.int64) * int(page_size) if bind_mount else \
        np.asarray(d["read_bytes_delta"], dtype=np.int64)
    wall = float(np.asarray(d["wall_ms"], dtype=np.float64)[keep].sum())
    res: dict[str, Any] = {
        "measured_rows": int(keep.sum()),
        "measured_tokens": tokens,
        "measured_bytes_per_token": float(rb[keep].sum()) / tokens if tokens else None,
        "measured_majflt_per_token": float(np.asarray(d["majflt_delta"])[keep].sum()) / tokens if tokens else None,
        "measured_tok_s": 1000.0 * tokens / wall if tokens and wall > 0 else None,
        "anon_rss_bytes_max": int(np.asarray(d["rss_anon_kb"]).max()) * 1024,
        "file_rss_bytes_max": int(np.asarray(d["rss_file_kb"]).max()) * 1024,
        "warmup_tokens": int(warmup_tokens),
        "bytes_basis": "majflt x page_size (lower bound)" if bind_mount else "read_bytes_delta",
    }
    if tokens == 0:
        log.warning("io.csv: no rows at or after token %d", warmup_tokens)
    return res


def _sim_run(sim: Mapping[str, Any], policy: str, budget: int | None) -> Mapping[str, Any] | None:
    cands = [r for r in sim.get("runs", []) or [] if r.get("policy") == policy and not r.get("profile")]
    if not cands:
        cands = [r for r in sim.get("runs", []) or [] if r.get("policy") == policy]
    if not cands:
        return None
    if budget is None:
        return cands[0]
    return min(cands, key=lambda r: abs(int(r.get("budget_bytes", 0)) - int(budget)))


def validation_point(io_csv: str | os.PathLike[str] | Mapping[str, np.ndarray], sim: Any, *, mem_cap_bytes: int,
                     pinned: bool = False, pin_budget_bytes: int | None = None, pin_locked_bytes: int | None = None,
                     anon_rss_bytes: int | None = None, file_cache_bytes: int | None = None,
                     tier_rand_read_3mb_mbps: float | None = None, ram_tok_s: float | None = None,
                     access_ms: float = 0.0, warmup_tokens: int = DEFAULT_IO_WARMUP,
                     page_size: int = DEFAULT_IO_PAGE_SIZE, volume: str = "pigemma-models", oom: bool = False,
                     sim_budget_bytes: int | None = None, notes: Sequence[str] = ()) -> dict[str, Any]:
    """One validation.json point (TRACE_FORMAT 12.3): measured io.csv vs sim.json LRU/Belady.

    predicted_tok_s = 1 / (1/ram_tok_s + sim_lru_bytes/(tier*1e6) + sim_lru_n_miss*access_ms/1e3)
    (None without ram_tok_s or tier). ``sim``: sim.json path or dict, simulated on the same token
    range with the same warm-up.
    """
    if volume not in VOLUMES:
        raise CacheSimError(f"volume must be one of {VOLUMES}, got {volume!r}")
    simd = _read_json(sim) if isinstance(sim, (str, os.PathLike)) else dict(sim)
    bind = volume == "bind-mount"
    meas = measure_io(io_csv, warmup_tokens, page_size=page_size, bind_mount=bind)
    lru = _sim_run(simd, "lru", sim_budget_bytes)
    bel = _sim_run(simd, "belady", sim_budget_bytes)
    nts = list(notes)
    if lru is None:
        nts.append("sim.json has no lru run")
    if int(simd.get("warmup_tokens", warmup_tokens)) != int(warmup_tokens):
        nts.append(f"sim warm-up {simd.get('warmup_tokens')} != io warm-up {warmup_tokens}")
    sim_lru = _num(((lru or {}).get("steady") or {}).get("miss_bytes_per_token", {}).get("mean")) if lru else None
    sim_lru_n = _num(((lru or {}).get("steady") or {}).get("n_miss_per_token")) if lru else None
    sim_bel = _num(((bel or {}).get("steady") or {}).get("miss_bytes_per_token", {}).get("mean")) if bel else None
    measured = meas["measured_bytes_per_token"]
    ratio = (sim_lru / measured) if (sim_lru is not None and measured) else None
    pred = None
    assumptions: dict[str, Any] = {}
    if ram_tok_s and tier_rand_read_3mb_mbps and sim_lru is not None:
        t = 1.0 / float(ram_tok_s) + sim_lru / (float(tier_rand_read_3mb_mbps) * 1e6) + \
            (sim_lru_n or 0.0) * float(access_ms) / 1e3
        pred = 1.0 / t if t > 0 else None
        assumptions = {
            "ram_tok_s": _assume(float(ram_tok_s), "tok/s", "measured: uncapped (all in RAM) run on the same host"),
            "tier_rand_read_3mb_mbps": _assume(float(tier_rand_read_3mb_mbps), "MB/s",
                                               "measured: fio randread bs=3345k on the model volume"),
            "access_ms": _assume(float(access_ms), "ms", "assumed: per-miss latency already inside the QD1 fio "
                                                         "throughput" if not access_ms else "assumed: given"),
            "sim_lru_bytes_per_token": _assume(sim_lru, "B", "simulated: pigemma.cache_sim lru steady state"),
        }
    else:
        nts.append("predicted_tok_s needs --ram-tok-s and --tier-mbps")
    if bind:
        nts.append(BIND_MOUNT_NOTE)
    if oom:
        nts.append("OOM: the cgroup oom_kill counter increased during the run")
    point = {
        "mem_cap_bytes": int(mem_cap_bytes), "pinned": bool(pinned),
        "pin_budget_bytes": pin_budget_bytes, "pin_locked_bytes": pin_locked_bytes,
        "anon_rss_bytes": anon_rss_bytes if anon_rss_bytes is not None else meas["anon_rss_bytes_max"],
        "file_cache_bytes": file_cache_bytes,
        "tier_rand_read_3mb_mbps": tier_rand_read_3mb_mbps,
        "measured_bytes_per_token": measured,
        "measured_majflt_per_token": meas["measured_majflt_per_token"],
        "measured_tok_s": meas["measured_tok_s"],
        "sim_budget_bytes": int(lru["budget_bytes"]) if lru else sim_budget_bytes,
        "sim_lru_bytes_per_token": sim_lru,
        "sim_belady_bytes_per_token": sim_bel,
        "ratio_sim_lru_over_measured": ratio,
        "predicted_tok_s": pred,
        "oom": bool(oom),
        "notes": nts,
        "projected": {"kind": "projected", "predicted_tok_s": pred, "assumptions": assumptions},
        "measured_rows": meas["measured_rows"], "measured_tokens": meas["measured_tokens"],
        "bytes_basis": meas["bytes_basis"],
        "within_belady_lru_band": (None if (measured is None or sim_lru is None or sim_bel is None)
                                   else bool(0.8 * sim_bel <= measured <= 1.3 * sim_lru)),
    }
    return _jsonable(point)


def write_validation(path: str | os.PathLike[str], point: Mapping[str, Any], *, gguf: str | None = None,
                     volume: str = "pigemma-models", token_range: Sequence[int] | None = None,
                     warmup_tokens: int = DEFAULT_IO_WARMUP) -> Path:
    """Merge one point into validation.json (a point with the same mem_cap_bytes + pinned is replaced)."""
    p = Path(path)
    doc: dict[str, Any] = {}
    if p.is_file():
        try:
            doc = _read_json(p)
        except (OSError, ValueError) as e:
            raise CacheSimError(f"{p}: cannot read existing validation.json ({e})") from e
    doc.setdefault("format_version", VALIDATION_FORMAT_VERSION)
    doc["generator"] = GENERATOR
    doc["kind"] = "measured"
    if gguf is not None:
        doc["gguf"] = gguf
    doc.setdefault("gguf", None)
    doc["volume"] = volume
    if token_range is not None:
        doc["token_range"] = [int(token_range[0]), int(token_range[1])]
    doc.setdefault("token_range", None)
    doc["warmup_tokens"] = int(warmup_tokens)
    pts = [q for q in doc.get("points", []) or []
           if not (int(q.get("mem_cap_bytes", -1)) == int(point["mem_cap_bytes"]) and bool(q.get("pinned")) == bool(point["pinned"]))]
    pts.append(dict(point))
    pts.sort(key=lambda q: (int(q.get("mem_cap_bytes", 0)), bool(q.get("pinned"))))
    doc["points"] = pts
    doc["updated_at"] = _utc_now()
    _atomic_write_text(p, _dumps(doc))
    return p


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _int_arg(s: str) -> int:
    v = _num(s)
    if v is None or v < 0 or not float(v).is_integer():
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {s!r}")
    return int(v)


def _float_arg(s: str) -> float:
    v = _num(s)
    if v is None:
        raise argparse.ArgumentTypeError(f"expected a number, got {s!r}")
    return v


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m pigemma.cache_sim",
        description="Expert-cache simulator (H1): replay a .moetrace at byte budgets under "
                    "lru/lfu/static/belady/lru_prefetch/lru_perlayer, report hit rates and bytes/token, "
                    "and project tok/s per catalog device profile (kind=projected).",
        epilog="Second form: python -m pigemma.cache_sim io-point --help (E5/E6 io.csv vs simulator). "
               "Exit codes: 0 ok, 1 error, 2 usage, 3 missing input file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("trace", help="routing trace (.moetrace), e.g. traces/26b/all.moetrace")
    ap.add_argument("--map", required=True, help="map.json of the GGUF (python -m pigemma.gguf_map build)")
    ap.add_argument("--catalog", default=str(paths.CATALOG_PATH), help="catalog.json (device profiles)")
    ap.add_argument("--out", required=True, help="sim.json output path")
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES), help=f"comma list of {', '.join(POLICIES)}")
    ap.add_argument("--budgets-gb", default=None,
                    help="raw expert-cache budgets in GB (1e9 B), e.g. 0.5,1,2,3,4,5,6,8,12.85 "
                         "(default: these, when --profiles is not given either)")
    ap.add_argument("--profiles", default=None,
                    help="catalog device profiles: all | pi5 | desktop | name[,name] (default: all, when "
                         "--budgets-gb is not given either)")
    ap.add_argument("--train-split", type=float, default=DEFAULT_TRAIN_SPLIT,
                    help="fraction of the tokens used to train static / lru_prefetch")
    ap.add_argument("--warmup-tokens", type=_int_arg, default=DEFAULT_WARMUP_TOKENS,
                    help="tokens excluded from the steady-state metrics")
    ap.add_argument("--read-granularity", type=_int_arg, default=DEFAULT_READ_GRANULARITY,
                    help="bytes; each expert slab read on a miss is rounded up to this (readahead model)")
    ap.add_argument("--read-model", choices=READ_MODELS, default="round",
                    help="round: slab size rounded up; aligned: granules touched at the real file offset")
    ap.add_argument("--lfu-decay", type=float, default=DEFAULT_LFU_DECAY, help="LFU count decay per token")
    ap.add_argument("--prefetch-n", type=_int_arg, default=None, help="experts prefetched per slot (default K)")
    ap.add_argument("--token-range", default=None, help="A:B (Python slice) of the trace records to simulate")
    ap.add_argument("--io-mem", default="auto",
                    help="measured io_mem.json merged into the catalog devices: auto (results/pi/io_mem.json if "
                         "present) | none | PATH")
    ap.add_argument("--calibration", default="auto",
                    help="llama_efficiency: auto (results/pi/calibration.json for Pi profiles, desktop E0 + membw "
                         "from --e0-dir) | none (0.65 assumed) | PATH (calibration.json, or a directory of E0 results)")
    ap.add_argument("--llama-efficiency", type=float, default=None, help="explicit efficiency for every profile")
    ap.add_argument("--e0-dir", default=None, help=f"desktop E0 results directory (default {paths.repo_relative(DEFAULT_E0_DIR)})")
    ap.add_argument("--no-bypass", action="store_true", help="belady without bypass (forced-insertion MIN)")
    ap.add_argument("--no-stack-check", action="store_true", help="skip the LRU vs stack-distance self-check")
    ap.add_argument("--dense-model", default=DENSE_MODEL_ID, help="catalog id of the dense reference line")
    ap.add_argument("--dense-map", default=None, help="optional map.json of the dense model (exact file data bytes, KV)")
    ap.add_argument("--csv", default=None, help="sim.csv output path")
    ap.add_argument("--series", default=None, help="per-token series .npz output path")
    ap.add_argument("--results-dir", default=None, help="where E4 envelopes go (default: the directory of --out)")
    ap.add_argument("--no-envelopes", action="store_true", help="do not write E4 results envelopes")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    return ap


def _build_io_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m pigemma.cache_sim io-point",
        description="Compare a measured page-cache run (pigemma-probe --io-csv) with sim.json (lru/belady) and "
                    "print / merge one validation.json point (E5/E6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--io-csv", required=True, help="io.csv from pigemma-probe --io-csv")
    ap.add_argument("--sim", required=True, help="sim.json simulated on the same token range (raw --budgets-gb run)")
    ap.add_argument("--mem-cap-bytes", type=_int_arg, required=True, help="container memory cap (bytes)")
    ap.add_argument("--pinned", action="store_true", help="the run used the pinner")
    ap.add_argument("--pin-budget-bytes", type=_int_arg, default=None)
    ap.add_argument("--pin-locked-bytes", type=_int_arg, default=None)
    ap.add_argument("--anon-rss-bytes", type=_int_arg, default=None, help="default: max RssAnon of io.csv")
    ap.add_argument("--file-cache-bytes", type=_int_arg, default=None, help="cgroup memory.stat file")
    ap.add_argument("--tier-mbps", type=float, default=None, help="measured fio randread 3 MB MB/s of the volume")
    ap.add_argument("--ram-tok-s", type=float, default=None, help="measured tok/s of the uncapped run (all in RAM)")
    ap.add_argument("--access-ms", type=float, default=0.0, help="extra per-miss latency in the time model")
    ap.add_argument("--warmup-tokens", type=_int_arg, default=DEFAULT_IO_WARMUP)
    ap.add_argument("--page-size", type=_int_arg, default=DEFAULT_IO_PAGE_SIZE, help="for bind-mount majflt bytes")
    ap.add_argument("--volume", choices=VOLUMES, default="pigemma-models")
    ap.add_argument("--oom", action="store_true", help="the cgroup oom_kill counter increased")
    ap.add_argument("--sim-budget-bytes", type=_int_arg, default=None, help="pick the sim run closest to this budget")
    ap.add_argument("--gguf", default=None, help="GGUF path recorded in validation.json")
    ap.add_argument("--token-range", default=None, help="A:B recorded in validation.json")
    ap.add_argument("--note", action="append", default=[], help="extra note (repeatable)")
    ap.add_argument("--validation", default=None, help="validation.json to merge the point into")
    ap.add_argument("--log-level", default="INFO")
    return ap


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level).upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr, force=True)


def _utf8_stdio() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError, OSError):
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _parse(ap: argparse.ArgumentParser, argv: Sequence[str]) -> argparse.Namespace | int:
    try:
        return ap.parse_args(list(argv))
    except SystemExit as e:  # argparse: 0 for --help, 2 for usage errors
        return int(e.code) if isinstance(e.code, int) else paths.EXIT_USAGE


def _main_io_point(argv: Sequence[str]) -> int:
    ns = _parse(_build_io_parser(), argv)
    if isinstance(ns, int):
        return ns
    _setup_logging(ns.log_level)
    for p in (ns.io_csv, ns.sim):
        if not Path(p).is_file():
            log.error("missing input: %s", p)
            return paths.EXIT_MISSING
    try:
        pt = validation_point(ns.io_csv, ns.sim, mem_cap_bytes=ns.mem_cap_bytes, pinned=ns.pinned,
                              pin_budget_bytes=ns.pin_budget_bytes, pin_locked_bytes=ns.pin_locked_bytes,
                              anon_rss_bytes=ns.anon_rss_bytes, file_cache_bytes=ns.file_cache_bytes,
                              tier_rand_read_3mb_mbps=ns.tier_mbps, ram_tok_s=ns.ram_tok_s, access_ms=ns.access_ms,
                              warmup_tokens=ns.warmup_tokens, page_size=ns.page_size, volume=ns.volume, oom=ns.oom,
                              sim_budget_bytes=ns.sim_budget_bytes, notes=ns.note)
        tr_range = None
        if ns.token_range:
            m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", ns.token_range)
            if not m:
                log.error("--token-range must be A:B with non-negative integers")
                return paths.EXIT_USAGE
            tr_range = [int(m.group(1)), int(m.group(2))]
        if ns.validation:
            p = write_validation(ns.validation, pt, gguf=ns.gguf, volume=ns.volume, token_range=tr_range,
                                 warmup_tokens=ns.warmup_tokens)
            log.info("merged point (cap %d B, pinned %s) into %s", pt["mem_cap_bytes"], pt["pinned"], p)
    except (CacheSimError, ValueError, KeyError) as e:
        log.error("%s", e)
        return paths.EXIT_ERROR
    sys.stdout.write(_dumps(pt))
    return paths.EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the exit code."""
    _utf8_stdio()
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "io-point":
        return _main_io_point(args[1:])
    if args and args[0] == "simulate":
        args = args[1:]
    ns = _parse(_build_parser(), args)
    if isinstance(ns, int):
        return ns
    _setup_logging(ns.log_level)
    missing = [p for p in (ns.trace, ns.map, ns.catalog) if not Path(p).is_file()]
    for opt in (ns.dense_map,):
        if opt and not Path(opt).is_file():
            missing.append(opt)
    if ns.io_mem and ns.io_mem.lower() not in ("auto", "none") and not Path(ns.io_mem).is_file():
        missing.append(ns.io_mem)
    if missing:
        for p in missing:
            log.error("missing input file: %s", p)
        return paths.EXIT_MISSING
    try:
        budgets = parse_budgets_gb(ns.budgets_gb) if ns.budgets_gb is not None else None
        policies = [p.strip().lower() for p in ns.policies.split(",") if p.strip()]
        bad = [p for p in policies if p not in POLICIES]
        if bad or not policies:
            raise CacheSimError(f"unknown policies {bad or '(none)'}; known: {', '.join(POLICIES)}")
        if ns.profiles is not None:
            select_profiles(feas.load_catalog(ns.catalog), ns.profiles)
        if ns.llama_efficiency is not None and not ns.llama_efficiency > 0:
            raise CacheSimError("--llama-efficiency must be > 0")
        if not 0.0 <= ns.train_split <= 1.0:
            raise CacheSimError("--train-split must lie in [0, 1]")
        if not 0.0 < ns.lfu_decay <= 1.0:
            raise CacheSimError("--lfu-decay must lie in (0, 1]")
        if ns.read_granularity < 1:
            raise CacheSimError("--read-granularity must be >= 1")
        if ns.token_range is not None:
            parse_token_range(ns.token_range, 1)
    except CacheSimError as e:
        log.error("%s", e)
        return paths.EXIT_USAGE
    except ValueError as e:  # malformed catalog.json
        log.error("%s", e)
        return paths.EXIT_ERROR
    try:
        cat = feas.load_catalog(ns.catalog, io_mem=_resolve_io_mem(ns.io_mem))
        doc, runs, prep = run_simulation(
            ns.trace, ns.map, catalog=cat, policies=policies, budgets_gb=budgets, profiles=ns.profiles,
            train_split=ns.train_split, warmup_tokens=ns.warmup_tokens, read_granularity=ns.read_granularity,
            read_model=ns.read_model, lfu_decay=ns.lfu_decay, prefetch_n=ns.prefetch_n, token_range=ns.token_range,
            io_mem="none", calibration=ns.calibration, llama_efficiency=ns.llama_efficiency, e0_dir=ns.e0_dir,
            belady_bypass=not ns.no_bypass, stack_check=not ns.no_stack_check, dense_model=ns.dense_model,
            dense_map=ns.dense_map)
        doc["catalog"] = paths.repo_relative(ns.catalog)
        io_p = _resolve_io_mem(ns.io_mem)
        doc["io_mem"] = _rel(io_p)
        mi = map_info(ns.map)
        written = write_outputs(doc, runs, prep, mi, cat, out=ns.out, csv_path=ns.csv, series_path=ns.series,
                                results_dir=ns.results_dir, envelopes=not ns.no_envelopes)
    except FileNotFoundError as e:
        log.error("%s", e)
        return paths.EXIT_MISSING
    except (CacheSimError, tr.TraceFormatError, gm.GgufMapError, ValueError, KeyError) as e:
        log.error("%s", e)
        return paths.EXIT_ERROR
    print(format_table(doc))
    for k, v in written.items():
        if k == "envelopes":
            log.info("wrote %d E4 envelopes under %s", len(v), Path(v[0]).parent.as_posix() if v else "-")
        else:
            log.info("wrote %s: %s", k, v)
    chk = (doc.get("checks") or {}).get("lru_vs_stack_distance") or {}
    if chk.get("ran") and not chk.get("ok"):
        log.error("LRU simulation disagrees with the stack-distance curve (max |diff| %s): simulator bug",
                  chk.get("max_abs_diff"))
        return paths.EXIT_ERROR
    return paths.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
