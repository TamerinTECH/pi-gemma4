"""pigemma.feasibility - Raspberry Pi 5 feasibility verdict for the Gemma 4 models (stdlib only).

Pure decision logic: ``facts.json`` (collected read-only by ``pi/lib/common.sh collect_facts``) +
``pigemma/data/catalog.json`` (+ optional measured ``results/pi/io_mem.json``, ``results/h1/sim.json``,
``results/pi/calibration.json``) -> one verdict row per runnable model/quant, a recommended
model + llama.cpp flags, and an exit code that gates every ``pi/*.sh`` script.

This module MUST stay stdlib-only: it runs on a stock Pi Python 3.13 without numpy.

Usage (from the repo root; ``python3`` on the Pi)::

    python3 -m pigemma.feasibility --facts pi/out/facts.json [--catalog pigemma/data/catalog.json]
        [--want 26b|31b|<model-id>] [--sim results/h1/sim.json] [--json pi/out/feasibility.json]
        [--emit env] [--io-mem auto|none|PATH] [--calibration auto|none|PATH] [--device NAME]
        [--threads N] [--ctx N] [--os-reserve-gib X] [--max-loadavg X] [--quiet]

Exit codes (shared by every pi/*.sh)::

    0  some row FITS_* (fits in RAM) or STREAM_OK
    1  error: bad arguments or unreadable/invalid input (argparse errors too; 2 is taken)
    2  only slow operation possible: the best row is STREAM_SLOW (or a runnable NEVER_GOOD_SPEED row)
    3  nothing viable (NOT_VIABLE / NEVER_GOOD_SPEED / NO_DISK everywhere)
    4  refused: protected host (catalog.protected_hosts / protected_service_globs) or busy host

Verdicts per row (``row["verdict"]``)::

    FITS_REPACKED    file fits in RAM even with llama.cpp's anonymous repack copy -> default load mode
    FITS_MAPPED      file fits in RAM only file-backed (-nr / --no-repack keeps it CPU_Mapped)
    STREAM_OK        MoE: non-experts + an expert cache >= 2 GB in RAM, misses from nvme/usb-ssd, >= 1 tok/s
    STREAM_SLOW      MoE streaming at >= 0.3 tok/s that is not STREAM_OK (slow, microSD, small cache, ...)
    NOT_VIABLE       MoE: below 0.3 tok/s, or the non-expert weights + KV + buffers do not fit
    NEVER_GOOD_SPEED dense model: streamed (file_bytes / storage_bw s per token) or in RAM but below
                     viable_stream_tok_s; 'runnable' is true when it fits in RAM at >= 0.3 tok/s
    NO_DISK          a FITS_*/STREAM_* row whose file is not present and free disk < 1.05 x file bytes

Arithmetic (all speeds are PROJECTED; every input carries its source string)::

    usable        = ram_total - OS_RESERVE - kv - COMPUTE_BUFFERS
    FITS_MAPPED   : file_bytes <= usable
    FITS_REPACKED : additionally file_bytes + REPACK_EXTRA <= usable - REPACK_HEADROOM
    MoE stream    : expert_cache      = usable - nonexpert_bytes
                    resident_fraction = expert_cache / expert_bytes
                    hit               = assumed curve(resident_fraction)   (catalog.selection)
                                        or LRU steady hit rate from sim.json at that budget ('simulated')
                    miss_bytes        = (1 - hit) * expert_bytes_per_token_worst
                    n_miss            = (1 - hit) * n_moe_layers * n_expert_used
                    t_io  = miss_bytes / (rand_read_3mb_mbps * 1e6) + n_miss * access_ms / 1e3
                    t_ram = (nonexpert_bytes_per_token + hit * expert_bytes_per_token_worst)
                            / (mem_bw_gbps * 1e9 * llama_efficiency)
                    tok_s = 1 / (t_ram + t_io)
    FITS (any)    : in-RAM bound tok_s = mem_bw * eff / (nonexpert_bytes_per_token + expert_bytes_per_token_worst)
    dense stream  : s_per_token = file_bytes / (seq_read_mbps * 1e6)

Constants (module level, documented with their provenance)::

    OS_RESERVE          catalog device os_reserve_bytes when a catalog device matches, else by nominal
                        RAM tier: 0.7e9 (<= 4 GB), 0.8e9 (8 GB), 0.9e9 (>= 16 GB)        [assumed]
                        overrides: --os-reserve-gib, env OS_RESERVE_GIB (pi/pi_config.env)
    KV_Q8_4K            160e6 B (26B at 4096 ctx, q8_0 K/V) and 420e6 B (31B)           [assumed,
                        from the gguf_map kv_bytes design numbers]; a catalog model field
                        'kv_bytes_q8_at_4096' wins when present. Scaled by max(1, n_ctx/4096):
                        conservative both ways (sliding-window layers stop growing at 1024).
    COMPUTE_BUFFERS     0.3e9 B, or the matched catalog device's compute_buffers_bytes  [assumed]
    MIN_STREAM_CACHE    2.0e9 B expert cache needed for STREAM_OK                       [policy]
    MIN_FREE_RAM_TO_RUN 1.0e9 B MemAvailable below this -> busy refusal (exit 4)        [policy]
    LLAMA_EFFICIENCY    0.65 of the memory bandwidth reaches llama.cpp decode           [assumed;
                        results/pi/calibration.json llama_efficiency overrides, source 'measured']
    BUSY_LOADAVG1       2.0 for a 4-core Pi; scaled by nproc/4 above 4 CPUs (emulated containers on a
                        12-vCPU Docker VM), env PIGEMMA_MAX_LOADAVG / --max-loadavg override  [policy]

Design decisions / deviations (also returned to the orchestrator as open issues):

* facts.json carries no device name. The effective device profile is built from the facts: RAM from
  MemTotal (min with an optional ``cgroup_mem_max`` key, for --memory capped containers), storage class
  from ``storage.type`` + ``pcie_link_speed`` via catalog.nvme_link_speed_class /
  storage_type_default_class. A catalog device is used for os_reserve / compute buffers / n_ctx /
  mem_bw / storage metrics only when (a) ``--device`` or ``facts["device"]`` names it, or (b) the
  merged io_mem.json was measured on this hostname, or (c) it has the same nominal RAM tier AND the
  same storage class. Otherwise the storage metrics come from catalog.storage_classes and the rest
  from the tier rule above.
* Dense rows: the spec makes a dense model that fits FITS_*; with a Pi's ~10 GB/s x 0.65 every 31B
  quant (>= 8.5 GB) projects < 1 tok/s in RAM, so a dense row that fits but projects below
  selection.viable_stream_tok_s is NEVER_GOOD_SPEED with ``runnable: true`` (>= 0.3 tok/s). This keeps
  "31b always NEVER_GOOD_SPEED" true on every Pi profile and still lets ``--want 31b`` on a 16 GB Pi
  run the 31B as a slow measurement (exit 2) instead of refusing it.
* STREAM rows that reach 1 tok/s but miss the STREAM_OK preconditions (cache < 2 GB, storage
  unknown/microSD) are STREAM_SLOW rather than NOT_VIABLE (the literal spec left a gap there).
* NO_DISK overrides FITS_*/STREAM_* only; NEVER_GOOD_SPEED keeps its verdict (runnable -> false).
* Protected-host and busy refusals (exit 4) still compute the rows so the operator sees the table,
  but ``recommended`` is None and ``rows_informational_only`` is true. facts.override ==
  'PIGEMMA_I_KNOW' lifts only the protected-host refusal, never the busy one.
* recommended: best by rank (FITS_* > STREAM_OK > STREAM_SLOW > runnable NEVER_GOOD_SPEED), tie-break
  higher-bit quant (nominal bpw table below), then larger file, then faster projection. FITS_REPACKED
  and FITS_MAPPED share a rank (so the 16 GB Pi picks the QAT Q4_0 with -nr over a smaller quant
  that could repack).
* memlock: ``facts.memlock_limit`` is ``ulimit -l`` (kB or 'unlimited'); a too-small limit adds a
  warning but does not change the FITS_MAPPED flags (llama.cpp warns and continues without mlock).
* io_mem merge: seq_read_mbps = max(qd1, qd8) (sequential readahead), rand_read_3mb_mbps = qd1
  (conservative, falls back to qd4), access_ms as given, mem_bw_gbps = max(read_gbps_by_threads)
  (falls back to copy, then triad). An io_mem device name absent from the catalog is synthesized
  from the name ('pi5-8gb-nvme-gen2' -> 8 GB tier, class nvme-gen2) with a logged warning.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import logging
import math
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # pigemma/__init__.py is stdlib-only and cheap; tolerate running the file stand-alone
    from pigemma import __version__ as _PIGEMMA_VERSION
except Exception:  # pragma: no cover - only when the package is not importable
    _PIGEMMA_VERSION = "0.0.0"

log = logging.getLogger("pigemma.feasibility")

# --------------------------------------------------------------------------------------------- paths
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG: Path = REPO_ROOT / "pigemma" / "data" / "catalog.json"
DEFAULT_IO_MEM: Path = REPO_ROOT / "results" / "pi" / "io_mem.json"
DEFAULT_CALIBRATION: Path = REPO_ROOT / "results" / "pi" / "calibration.json"

# ------------------------------------------------------------------------------------------ exit codes
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_SLOW = 2
EXIT_NOT_VIABLE = 3
EXIT_REFUSED = 4

# ------------------------------------------------------------------------------------------- verdicts
FITS_REPACKED = "FITS_REPACKED"
FITS_MAPPED = "FITS_MAPPED"
STREAM_OK = "STREAM_OK"
STREAM_SLOW = "STREAM_SLOW"
NOT_VIABLE = "NOT_VIABLE"
NEVER_GOOD_SPEED = "NEVER_GOOD_SPEED"
NO_DISK = "NO_DISK"
VERDICTS = (FITS_REPACKED, FITS_MAPPED, STREAM_OK, STREAM_SLOW, NOT_VIABLE, NEVER_GOOD_SPEED, NO_DISK)

# ------------------------------------------------------------------------------------------ constants
GB = 1_000_000_000
GIB = 1 << 30
#: OS reserve by nominal RAM tier (GB): <= 4 -> 0.7e9, 8 -> 0.8e9, >= 16 -> 0.9e9 (assumed).
OS_RESERVE_4GB = 700_000_000
OS_RESERVE_8GB = 800_000_000
OS_RESERVE_16GB = 900_000_000
#: q8_0 K+V cache bytes at n_ctx 4096 per model family (assumed; gguf_map kv_bytes design numbers).
KV_Q8_4K: dict[str, int] = {"26b": 160_000_000, "31b": 420_000_000}
KV_Q8_4K_DEFAULT = 420_000_000  # unknown family: the larger (conservative) value
KV_REF_CTX = 4096
COMPUTE_BUFFERS = 300_000_000
MIN_STREAM_CACHE = 2_000_000_000
MIN_FREE_RAM_TO_RUN = 1_000_000_000
#: FITS_REPACKED needs file + REPACK_EXTRA <= usable - REPACK_HEADROOM (anonymous repack copy).
REPACK_EXTRA = 300_000_000
REPACK_HEADROOM = 1_000_000_000
LLAMA_EFFICIENCY = 0.65
DEFAULT_MEM_BW_GBPS = 10.0  # Pi 5 LPDDR4X effective (assumed) when no catalog device supplies one
SLOW_TOK_S = 0.3
DEFAULT_VIABLE_TOK_S = 1.0  # catalog.selection.viable_stream_tok_s wins
DEFAULT_GOOD_TOK_S = 2.0  # catalog.selection.good_stream_tok_s wins
BUSY_LOADAVG1 = 2.0
BUSY_REF_NPROC = 4
DISK_FACTOR = 1.05
PIN_HEADROOM = 500_000_000  # pi/06 budget = MemAvailable - KV - compute - 0.5e9
DEFAULT_THREADS = 4
DEFAULT_N_CTX = 4096
OVERRIDE_TOKEN = "PIGEMMA_I_KNOW"
STREAM_STORAGE_TYPES = frozenset({"nvme", "usb-ssd"})
KNOWN_STORAGE_TYPES = ("nvme", "usb-ssd", "microsd", "unknown")
RAM_SKUS_GB = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
HF_URL = "https://huggingface.co/{repo}/resolve/main/{file}"

#: llama.cpp flag templates per verdict (spec strings; -t / -c filled from threads / n_ctx).
FLAGS_FITS_MAPPED = "-nr -lm mmap+mlock -t {t} -c {c} -fa on -ctk q8_0 -ctv q8_0"
FLAGS_FITS_REPACKED = "-lm mmap -t {t} -c {c} -fa on -ctk q8_0 -ctv q8_0"
FLAGS_STREAM = "-nr -lm mmap -t {t} -c {c} -ctk q8_0 -ctv q8_0 -b 256 -ub 64"

#: Always appended to the reasons (docs/design/FACTS.md "Hardware options").
RECOMMENDATION_LINE = (
    "Pi 5 16 GB + NVMe Gen3: 26B-A4B fits in RAM (-nr); 8 GB + NVMe Gen3: stream experts (~40% cached); "
    "8 GB + USB3: weak; 4 GB or microSD: not viable; 31B dense: never good speed on any Pi 5"
)

#: Nominal bits per weight, used only to rank quants ("tie-break higher-bit quant").
QUANT_BPW: dict[str, float] = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0, "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_1": 6.0, "Q5_0": 5.5,
    "Q5_K_M": 5.69, "Q5_K_S": 5.54, "Q4_1": 5.0, "Q4_K_M": 4.89, "Q4_K_S": 4.58, "Q4_0": 4.5,
    "IQ4_NL": 4.5, "Q4_K_XL": 4.9, "IQ4_XS": 4.25, "Q3_K_XL": 4.0, "Q3_K_L": 4.27, "Q3_K_M": 3.91,
    "IQ3_M": 3.66, "Q3_K_S": 3.5, "IQ3_S": 3.44, "IQ3_XS": 3.3, "IQ3_XXS": 3.06, "Q2_K_XL": 2.9,
    "Q2_K_L": 2.8, "Q2_K": 2.63, "IQ2_M": 2.7, "IQ2_S": 2.5, "IQ2_XS": 2.31, "IQ2_XXS": 2.06,
    "IQ1_M": 1.75, "IQ1_S": 1.56,
}

__all__ = [
    "load_catalog", "merge_io_mem", "load_facts", "load_calibration", "verdict", "Verdict",
    "resolve_device", "storage_class_for", "interpolate_curve", "kv_bytes", "nominal_ram_gb",
    "quant_bpw", "flags_for", "emit_env", "format_table", "main", "VERDICTS", "RECOMMENDATION_LINE",
]


# ============================================================================================ helpers
def _num(x: Any) -> float | None:
    """Return x as a finite float (accepts int/float/numeric str), else None. Booleans are rejected."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
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


def _src(x: Any, default: str = "assumed: (no source given)") -> str:
    if isinstance(x, Mapping) and x.get("source"):
        return str(x["source"])
    return default


def _kind_of(source: str) -> str:
    """'measured' | 'simulated' | 'assumed' from a provenance string (anything else -> 'assumed')."""
    s = (source or "").strip().lower()
    for k in ("measured", "simulated", "assumed"):
        if s.startswith(k):
            return k
    return "assumed"


def _gb(b: float | int | None) -> str:
    return "n/a" if b is None else f"{b / GB:.2f} GB"


def _read_json(path: str | os.PathLike[str]) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def nominal_ram_gb(ram_bytes: float) -> int:
    """Smallest Pi-style RAM SKU (GB) that holds ram_bytes (MemTotal is a bit below the SKU)."""
    for sku in RAM_SKUS_GB:
        if ram_bytes <= sku * GIB * 1.02:
            return sku
    return RAM_SKUS_GB[-1]


def tier_os_reserve(ram_bytes: float) -> int:
    """OS reserve by nominal tier: 0.7e9 (<= 4 GB), 0.8e9 (8 GB), 0.9e9 (>= 16 GB)."""
    t = nominal_ram_gb(ram_bytes)
    if t <= 4:
        return OS_RESERVE_4GB
    if t < 16:
        return OS_RESERVE_8GB
    return OS_RESERVE_16GB


def quant_bpw(quant: str | None) -> float:
    """Nominal bits per weight of a quant name ('UD-' prefix ignored); used for ranking only."""
    q = (quant or "").upper().strip()
    if q.startswith("UD-"):
        q = q[3:]
    if q in QUANT_BPW:
        return QUANT_BPW[q]
    m = re.search(r"Q(\d)", q)
    return float(m.group(1)) + 0.5 if m else 4.0


def kv_bytes(model: Mapping[str, Any], n_ctx: int) -> int:
    """q8_0 KV cache bytes for a model at n_ctx (conservative linear scale above 4096, floor at 4096)."""
    base = _num(model.get("kv_bytes_q8_at_4096"))
    if base is None:
        base = float(KV_Q8_4K.get(str(model.get("family", "")).lower(), KV_Q8_4K_DEFAULT))
    return int(round(base * max(1.0, float(n_ctx) / KV_REF_CTX)))


def interpolate_curve(points: Sequence[Sequence[float]], x: float) -> float:
    """Piecewise-linear y(x): linear to (0, 0) below the first point, hold the last y above the last."""
    pts = sorted((float(a), float(b)) for a, b in points)
    if not pts:
        raise ValueError("empty curve")
    if x <= 0.0:
        return 0.0
    x0, y0 = pts[0]
    if x <= x0:
        return y0 * x / x0 if x0 > 0 else y0
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if x <= xb:
            return yb if xb <= xa else ya + (yb - ya) * (x - xa) / (xb - xa)
    return pts[-1][1]


def flags_for(verdict_name: str, *, fits_repacked: bool = False, threads: int = DEFAULT_THREADS,
              n_ctx: int = DEFAULT_N_CTX) -> str | None:
    """llama.cpp flags for a verdict (None for verdicts that must not run)."""
    if verdict_name in (FITS_MAPPED, FITS_REPACKED):
        tpl = FLAGS_FITS_REPACKED if fits_repacked else FLAGS_FITS_MAPPED
    elif verdict_name in (STREAM_OK, STREAM_SLOW):
        tpl = FLAGS_STREAM
    else:
        return None
    return tpl.format(t=int(threads), c=int(n_ctx))


# ========================================================================================== catalog
def _find_device(catalog: Mapping[str, Any], name: str) -> dict | None:
    for d in catalog.get("devices", []) or []:
        if isinstance(d, dict) and d.get("name") == name:
            return d
    return None


def _class_metrics(catalog: Mapping[str, Any], cls: str) -> dict:
    classes = catalog.get("storage_classes", {}) or {}
    entry = classes.get(cls) or classes.get("unknown")
    if not isinstance(entry, Mapping):
        raise ValueError(f"catalog.storage_classes has neither {cls!r} nor 'unknown'")
    out = copy.deepcopy(dict(entry))
    out["class"] = cls if cls in classes else "unknown"
    return out


def _class_from_device_name(name: str) -> str:
    n = name.lower()
    if "nvme-gen3" in n or "gen3" in n:
        return "nvme-gen3"
    if "nvme" in n:
        return "nvme-gen2"
    if "usb" in n or "ssd" in n:
        return "usb-ssd"
    if "microsd" in n or "sdcard" in n or re.search(r"(^|-)sd($|-)", n):
        return "microsd"
    return "unknown"


def _synthesize_device(catalog: dict, name: str) -> dict:
    """Build a device profile for an io_mem device name that the catalog lacks (logged)."""
    m = re.search(r"(\d+)\s*gb", name.lower())
    ram_gb = int(m.group(1)) if m else None
    pis = [d for d in catalog.get("devices", []) if d.get("platform") == "pi5"]
    base = None
    if ram_gb is not None:
        base = next((d for d in pis if nominal_ram_gb(_num(d.get("ram_bytes")) or 0) == ram_gb), None)
    base = copy.deepcopy(base or (pis[0] if pis else {}))
    cls = _class_from_device_name(name)
    st = _class_metrics(catalog, cls)
    dev: dict[str, Any] = {
        "name": name,
        "platform": "pi5" if name.lower().startswith("pi5") else base.get("platform", "unknown"),
        "ram_bytes": int(ram_gb * GB) if ram_gb else base.get("ram_bytes"),
        "os_reserve_bytes": tier_os_reserve(ram_gb * GB) if ram_gb else base.get("os_reserve_bytes"),
        "compute_buffers_bytes": base.get("compute_buffers_bytes", COMPUTE_BUFFERS),
        "n_ctx": base.get("n_ctx", DEFAULT_N_CTX),
        "mem_bw_gbps": copy.deepcopy(base.get("mem_bw_gbps") or {
            "value": DEFAULT_MEM_BW_GBPS, "source": "assumed: Pi 5 ~10 GB/s effective (pigemma.feasibility default)"}),
        "storage": {"type": st.get("type", "unknown"), "class": st["class"],
                    "seq_read_mbps": st["seq_read_mbps"], "rand_read_3mb_mbps": st["rand_read_3mb_mbps"],
                    "access_ms": st["access_ms"]},
        "sources": {
            "ram_bytes": "assumed: nominal RAM parsed from the io_mem device name (synthesized profile)",
            "os_reserve_bytes": "assumed: pigemma.feasibility RAM-tier rule (synthesized profile)",
            "compute_buffers_bytes": "assumed: copied from the nearest catalog Pi profile (synthesized)",
            "n_ctx": "assumed: copied from the nearest catalog Pi profile (synthesized)",
        },
        "synthesized": True,
    }
    log.warning("io_mem device %r is not in the catalog; synthesized a profile (RAM %s GB, storage class %s)",
                name, ram_gb if ram_gb is not None else "?", cls)
    catalog.setdefault("devices", []).append(dev)
    return dev


def merge_io_mem(catalog: dict, io_mem: Mapping[str, Any], *, origin: str = "io_mem.json") -> dict:
    """Overwrite the matching device's storage.* and mem_bw_gbps with measured io_mem values (in place).

    Sets each overwritten field's source to a string starting with 'measured'. Returns the device.
    """
    if not isinstance(io_mem, Mapping):
        raise TypeError("io_mem must be a mapping (parsed io_mem.json) or a path")
    name = io_mem.get("device")
    if not name or not isinstance(name, str):
        raise ValueError("io_mem.json has no 'device' name")
    dev = _find_device(catalog, name) or _synthesize_device(catalog, name)
    host = io_mem.get("host", "?")
    when = io_mem.get("collected_at", "?")
    origin = Path(str(origin)).name or "io_mem.json"

    def measured(key: str, entry: Any, unit: str) -> dict:
        val = _metric(entry)
        out: dict[str, Any] = {
            "value": val,
            "source": f"measured: {origin} host={host} collected_at={when} field={key} ({unit})",
        }
        if isinstance(entry, Mapping) and entry.get("cmd"):
            out["cmd"] = str(entry["cmd"])
        return out

    st_in = io_mem.get("storage") or {}
    st = dev.setdefault("storage", {})
    changed: list[str] = []

    seq = [(k, st_in.get(k)) for k in ("seq_read_mbps_qd1", "seq_read_mbps_qd8") if _metric(st_in.get(k))]
    if seq:
        k, e = max(seq, key=lambda kv: _metric(kv[1]) or 0.0)
        if (_metric(e) or 0) > 0:
            st["seq_read_mbps"] = measured(k, e, "MB/s, max of qd1/qd8")
            changed.append("storage.seq_read_mbps")
    for k in ("rand_read_3mb_mbps_qd1", "rand_read_3mb_mbps_qd4"):
        v = _metric(st_in.get(k))
        if v and v > 0:
            st["rand_read_3mb_mbps"] = measured(k, st_in[k], "MB/s, qd1 preferred (conservative)")
            changed.append("storage.rand_read_3mb_mbps")
            break
    v = _metric(st_in.get("access_ms"))
    if v is not None and v >= 0:
        st["access_ms"] = measured("access_ms", st_in["access_ms"], "ms")
        changed.append("storage.access_ms")

    mem = io_mem.get("memory") or {}
    by_thr = mem.get("read_gbps_by_threads") or {}
    cand: list[tuple[str, Any]] = []
    if isinstance(by_thr, Mapping):
        cand = [(f"memory.read_gbps_by_threads[{t}]", e) for t, e in by_thr.items() if (_metric(e) or 0) > 0]
    if cand:
        k, e = max(cand, key=lambda kv: _metric(kv[1]) or 0.0)
        dev["mem_bw_gbps"] = measured(k, e, "GB/s, best thread count")
        changed.append("mem_bw_gbps")
    else:
        for k in ("copy_gbps", "triad_gbps"):
            if (_metric(mem.get(k)) or 0) > 0:
                dev["mem_bw_gbps"] = measured(f"memory.{k}", mem[k], "GB/s")
                changed.append("mem_bw_gbps")
                break
    catalog["_io_mem"] = {"host": host, "device": name, "collected_at": when, "origin": origin,
                          "fields": changed}
    if not changed:
        log.warning("io_mem.json for device %s contained no usable measurements", name)
    return dev


def load_catalog(path: str | os.PathLike[str] | None = None,
                 io_mem: str | os.PathLike[str] | Mapping[str, Any] | None = None) -> dict:
    """Load catalog.json (never rewritten) and optionally merge a measured io_mem.json (path or dict)."""
    cat_path = Path(path) if path is not None else DEFAULT_CATALOG
    catalog = _read_json(cat_path)
    if not isinstance(catalog, dict):
        raise ValueError(f"{cat_path}: catalog must be a JSON object")
    for key in ("models", "devices", "selection"):
        if key not in catalog:
            raise ValueError(f"{cat_path}: catalog lacks {key!r}")
    catalog["_path"] = str(cat_path)
    if io_mem is None:
        return catalog
    if isinstance(io_mem, (str, os.PathLike)):
        data = _read_json(io_mem)
        merge_io_mem(catalog, data, origin=str(io_mem))
    elif isinstance(io_mem, Mapping):
        merge_io_mem(catalog, copy.deepcopy(dict(io_mem)))
    else:
        raise TypeError(f"io_mem must be a path or a mapping, not {type(io_mem).__name__}")
    return catalog


# ============================================================================================ inputs
def load_facts(path: str | os.PathLike[str] | Mapping[str, Any]) -> dict:
    """Load and minimally validate facts.json (path, '-' for stdin, or an already-parsed mapping)."""
    if isinstance(path, Mapping):
        facts = copy.deepcopy(dict(path))
    elif str(path) == "-":
        facts = json.load(sys.stdin)
    else:
        facts = _read_json(path)
    if not isinstance(facts, dict):
        raise ValueError("facts.json must be a JSON object")
    total = _num(facts.get("ram_total_kb"))
    if total is None or total <= 0:
        raise ValueError("facts.json: ram_total_kb missing or not a positive number")
    return facts


def load_calibration(cal: str | os.PathLike[str] | Mapping[str, Any] | None) -> tuple[float, str] | None:
    """Return (llama_efficiency, source) from results/pi/calibration.json, or None."""
    if cal is None:
        return None
    origin = "calibration"
    if not isinstance(cal, Mapping):
        origin = Path(cal).name
        cal = _read_json(cal)
    entry = cal.get("llama_efficiency") if isinstance(cal, Mapping) else None
    val = _metric(entry)
    if val is None or not (0.05 <= val <= 1.5):
        raise ValueError(f"calibration: llama_efficiency missing or out of range (0.05..1.5): {val!r}")
    src = _src(entry, "")
    if not src:
        src = f"measured: {origin} llama_efficiency"
    elif not src.lower().startswith(("measured", "assumed", "simulated")):
        src = f"measured: {origin} ({src})"
    return val, src


# ========================================================================================== device
def storage_class_for(storage: Mapping[str, Any] | None, catalog: Mapping[str, Any]) -> tuple[str, str, list[str]]:
    """(storage type, catalog storage class, notes) for facts['storage']."""
    notes: list[str] = []
    st = dict(storage or {})
    st_type = str(st.get("type") or "unknown").lower()
    if st_type not in KNOWN_STORAGE_TYPES:
        notes.append(f"storage type {st_type!r} not recognised -> 'unknown' (conservative microSD numbers)")
        st_type = "unknown"
    defaults = catalog.get("storage_type_default_class", {}) or {}
    if st_type == "nvme":
        speed = st.get("pcie_link_speed")
        mapping = {k: v for k, v in (catalog.get("nvme_link_speed_class", {}) or {}).items() if k != "source"}
        if speed:
            for key, cls in mapping.items():
                if str(speed).strip().startswith(key):
                    return st_type, cls, notes
            notes.append(f"NVMe link speed {speed!r} not in catalog.nvme_link_speed_class -> default class")
        else:
            notes.append("NVMe link speed unknown -> conservative class "
                         f"{defaults.get('nvme', 'nvme-gen2')} (Gen3 needs dtparam=pciex1_gen=3)")
    if st_type == "unknown":
        notes.append("storage type unknown (container / overlay root?) -> catalog 'unknown' class = microSD numbers")
    return st_type, str(defaults.get(st_type, "unknown")), notes


def _pi_mem_bw(catalog: Mapping[str, Any], tier: int) -> dict:
    devs = [d for d in catalog.get("devices", []) if isinstance(d, dict)]
    pis = [d for d in devs if d.get("platform") == "pi5"]
    for pool in ([d for d in pis if nominal_ram_gb(_num(d.get("ram_bytes")) or 0) == tier], pis):
        for d in pool:
            if isinstance(d.get("mem_bw_gbps"), Mapping) and _metric(d["mem_bw_gbps"]):
                return copy.deepcopy(dict(d["mem_bw_gbps"]))
    return {"value": DEFAULT_MEM_BW_GBPS,
            "source": "assumed: Pi 5 LPDDR4X ~10 GB/s effective (pigemma.feasibility default)"}


def resolve_device(facts: Mapping[str, Any], catalog: Mapping[str, Any], *, device: str | None = None,
                   threads: int | None = None, n_ctx: int | None = None,
                   os_reserve_bytes: int | None = None) -> tuple[dict, list[str]]:
    """Effective device profile for the facts (see module docstring) and the notes explaining it."""
    notes: list[str] = []
    ram_total = float(_num(facts.get("ram_total_kb")) or 0) * 1024.0
    ram_src = "measured: facts.json ram_total_kb (MemTotal)"
    cg = _num(facts.get("cgroup_mem_max"))
    if cg is not None and 0 < cg < ram_total:
        notes.append(f"cgroup memory limit {_gb(cg)} < MemTotal {_gb(ram_total)}: using the cgroup limit")
        ram_total, ram_src = cg, "measured: facts.json cgroup_mem_max (container memory limit)"
    avail_kb = _num(facts.get("ram_available_kb"))
    ram_avail = avail_kb * 1024.0 if avail_kb is not None else None
    if ram_avail is not None and cg is not None and 0 < cg < ram_avail:
        ram_avail = cg
    tier = nominal_ram_gb(ram_total)
    st_type, st_class, st_notes = storage_class_for(facts.get("storage"), catalog)
    notes.extend(st_notes)

    matched: dict | None = None
    how = ""
    want_name = device or facts.get("device")
    if want_name:
        matched = _find_device(catalog, str(want_name))
        if matched is None:
            raise ValueError(f"device {want_name!r} is not in the catalog "
                             f"({', '.join(d.get('name', '?') for d in catalog.get('devices', []))})")
        how = "named explicitly (--device / facts.device)"
    else:
        iom = catalog.get("_io_mem")
        host = str(facts.get("hostname") or "")
        if isinstance(iom, Mapping) and host and str(iom.get("host", "")).lower() == host.lower():
            matched = _find_device(catalog, str(iom.get("device")))
            how = f"io_mem.json was measured on this host ({host})"
        if matched is None:
            for d in catalog.get("devices", []):
                dst = d.get("storage") or {}
                if (nominal_ram_gb(_num(d.get("ram_bytes")) or 0) == tier
                        and str(dst.get("class") or "") == st_class):
                    matched = d
                    how = f"same nominal RAM tier ({tier} GB) and storage class ({st_class})"
                    break

    prof: dict[str, Any] = {
        "name": matched["name"] if matched else f"facts:{facts.get('hostname') or 'host'}",
        "matched_catalog_device": matched["name"] if matched else None,
        "match_reason": how or "no catalog device with this RAM tier + storage class; profile built from the "
                               "facts, catalog.storage_classes and the RAM-tier rule",
        "platform": (matched or {}).get("platform", "pi5"),
        "nominal_ram_gb": tier,
        "ram_total_bytes": int(ram_total),
        "ram_available_bytes": None if ram_avail is None else int(ram_avail),
        "sources": {"ram_total_bytes": ram_src},
    }
    srcs = prof["sources"]
    msrc = (matched or {}).get("sources", {}) or {}

    # OS reserve
    if os_reserve_bytes is not None:
        prof["os_reserve_bytes"] = int(os_reserve_bytes)
        srcs["os_reserve_bytes"] = "assumed: operator override (--os-reserve-gib / OS_RESERVE_GIB)"
    elif matched and _num(matched.get("os_reserve_bytes")) is not None:
        prof["os_reserve_bytes"] = int(_num(matched["os_reserve_bytes"]) or 0)
        srcs["os_reserve_bytes"] = msrc.get("os_reserve_bytes", f"assumed: catalog device {matched['name']}")
    else:
        prof["os_reserve_bytes"] = tier_os_reserve(ram_total)
        srcs["os_reserve_bytes"] = (f"assumed: RAM-tier rule ({tier} GB tier: 0.7e9 <=4 GB, 0.8e9 8 GB, "
                                    "0.9e9 >=16 GB)")
    # compute buffers
    if matched and _num(matched.get("compute_buffers_bytes")) is not None:
        prof["compute_buffers_bytes"] = int(_num(matched["compute_buffers_bytes"]) or 0)
        srcs["compute_buffers_bytes"] = msrc.get("compute_buffers_bytes", f"assumed: catalog device {matched['name']}")
    else:
        prof["compute_buffers_bytes"] = COMPUTE_BUFFERS
        srcs["compute_buffers_bytes"] = "assumed: llama.cpp CPU compute buffers 0.3e9 (pigemma.feasibility)"
    # context / threads
    if n_ctx is not None:
        prof["n_ctx"], srcs["n_ctx"] = int(n_ctx), "assumed: run configuration (--ctx / CTX)"
    elif matched and _num(matched.get("n_ctx")):
        prof["n_ctx"] = int(_num(matched["n_ctx"]) or DEFAULT_N_CTX)
        srcs["n_ctx"] = msrc.get("n_ctx", "assumed: catalog device n_ctx")
    else:
        prof["n_ctx"], srcs["n_ctx"] = DEFAULT_N_CTX, "assumed: default run configuration (4096)"
    nproc = int(_num(facts.get("nproc")) or DEFAULT_THREADS)
    prof["threads"] = int(threads) if threads else max(1, min(DEFAULT_THREADS, nproc))
    # memory bandwidth
    if matched and isinstance(matched.get("mem_bw_gbps"), Mapping) and _metric(matched["mem_bw_gbps"]):
        prof["mem_bw_gbps"] = copy.deepcopy(dict(matched["mem_bw_gbps"]))
    else:
        prof["mem_bw_gbps"] = _pi_mem_bw(catalog, tier)
    # storage metrics
    mst = (matched or {}).get("storage") or {}
    use_dev_storage = bool(matched) and (str(mst.get("class") or "") == st_class or bool(want_name)
                                         or how.startswith("io_mem"))
    if use_dev_storage:
        stor = {k: copy.deepcopy(mst[k]) for k in ("seq_read_mbps", "rand_read_3mb_mbps", "access_ms") if k in mst}
        stor["type"], stor["class"] = str(mst.get("type") or st_type), str(mst.get("class") or st_class)
        if stor["type"] != st_type:
            notes.append(f"device {matched['name']} says storage {stor['type']}, facts say {st_type}: "
                         "using the device profile as requested")
    else:
        cm = _class_metrics(catalog, st_class)
        stor = {k: cm[k] for k in ("seq_read_mbps", "rand_read_3mb_mbps", "access_ms")}
        stor["type"], stor["class"] = st_type, cm["class"]
    for k in ("seq_read_mbps", "rand_read_3mb_mbps", "access_ms"):
        if _metric(stor.get(k)) is None:
            raise ValueError(f"storage metric {k} missing for class {stor['class']}")
    prof["storage"] = stor
    return prof, notes


# ======================================================================================== hit model
class _HitModel:
    """Expert-cache hit rate: the catalog's assumed curve or the LRU curve of a sim.json."""

    def __init__(self, catalog: Mapping[str, Any], sim: Mapping[str, Any] | None, sim_label: str | None):
        sel = catalog.get("selection", {}) or {}
        self.curve = [tuple(map(float, p)) for p in sel.get("stream_hit_rate_curve_assumed", [])]
        if not self.curve:
            raise ValueError("catalog.selection.stream_hit_rate_curve_assumed is empty")
        self.curve_source = str((sel.get("sources", {}) or {}).get(
            "stream_hit_rate_curve_assumed", "assumed: catalog stream_hit_rate_curve_assumed"))
        self.sim_label = sim_label or "sim.json"
        self.sim_items: list[tuple[float, float]] = []
        self.sim_bytes: list[tuple[float, float]] = []
        self.sim_layers: int | None = None
        self.notes: list[str] = []
        if sim is not None:
            self._load_sim(sim)

    @property
    def simulated(self) -> bool:
        return bool(self.sim_items or self.sim_bytes)

    def _load_sim(self, sim: Mapping[str, Any]) -> None:
        by_items: dict[float, list[float]] = {}
        by_bytes: dict[float, list[float]] = {}
        for run in sim.get("runs", []) or []:
            if not isinstance(run, Mapping) or str(run.get("policy", "")).lower() != "lru":
                continue
            block = run.get("steady") or run.get("overall") or {}
            hit = _num(block.get("hit_rate")) if isinstance(block, Mapping) else None
            if hit is None:
                continue
            hit = min(1.0, max(0.0, hit))
            cap = _num(run.get("capacity_items"))
            bud = _num(run.get("budget_bytes"))
            if cap is not None and cap >= 0:
                by_items.setdefault(cap, []).append(hit)
            elif bud is not None and bud >= 0:
                by_bytes.setdefault(bud, []).append(hit)
            pl = run.get("per_layer_hit_rate")
            if isinstance(pl, list) and pl and self.sim_layers is None:
                self.sim_layers = len(pl)
        self.sim_items = sorted((k, sum(v) / len(v)) for k, v in by_items.items())
        self.sim_bytes = sorted((k, sum(v) / len(v)) for k, v in by_bytes.items())
        if not self.simulated:
            self.notes.append(f"{self.sim_label}: no usable 'lru' runs -> falling back to the assumed curve")
            log.warning("%s has no usable lru runs; using the assumed hit-rate curve", self.sim_label)
        else:
            n = len(self.sim_items) + len(self.sim_bytes)
            self.notes.append(f"hit rates from {self.sim_label} (LRU steady state, {n} budget points, "
                              "source 'simulated') replace the assumed curve for MoE rows")

    def hit(self, model: Mapping[str, Any], expert_cache: float) -> tuple[float, str, str]:
        """(hit rate, kind 'assumed'|'simulated', source text) for an expert cache of expert_cache bytes."""
        expert_bytes = float(_num(model.get("expert_bytes")) or 0)
        frac = expert_cache / expert_bytes if expert_bytes > 0 else 0.0
        if frac >= 1.0:
            return 1.0, "assumed", "all experts resident (resident_fraction >= 1)"
        n_layers = int(_num(model.get("n_moe_layers")) or 0)
        if self.simulated and (self.sim_layers is None or self.sim_layers == n_layers):
            if self.sim_items:
                unit = _num(model.get("expert_unit_bytes")) or 0.0
                total_items = n_layers * int(_num(model.get("n_expert")) or 0)
                if unit > 0:
                    items = expert_cache / unit
                    h = 1.0 if total_items and items >= total_items else interpolate_curve(self.sim_items, items)
                    return (min(1.0, max(0.0, h)), "simulated",
                            f"simulated: LRU steady-state hit rate from {self.sim_label} at "
                            f"{items:.0f} expert-layer items ({frac:.3f} of all)")
            if self.sim_bytes:
                h = interpolate_curve(self.sim_bytes, expert_cache)
                return (min(1.0, max(0.0, h)), "simulated",
                        f"simulated: LRU steady-state hit rate from {self.sim_label} at budget {_gb(expert_cache)}")
        h = interpolate_curve(self.curve, frac)
        return (min(1.0, max(0.0, h)), "assumed",
                f"assumed curve at resident_fraction {frac:.3f} ({self.curve_source})")


# ========================================================================================== verdict
@dataclass
class Verdict:
    """Result of :func:`verdict`. ``rows``/``recommended`` are plain JSON-ready dicts."""

    ok: bool
    exit_code: int
    rows: list[dict]
    recommended: dict | None
    reasons: list[str]
    host_refused: bool
    busy: bool = False
    refusal_reasons: list[str] = field(default_factory=list)
    device: dict = field(default_factory=dict)
    host: dict = field(default_factory=dict)
    want: str | None = None
    hit_source: str = "assumed"
    assumptions: dict = field(default_factory=dict)
    recommendation_line: str = RECOMMENDATION_LINE

    @property
    def runnable(self) -> bool:
        """True when a run may proceed (exit 0, or exit 2 if the caller accepts slow operation)."""
        return self.exit_code in (EXIT_OK, EXIT_SLOW)

    @property
    def refused(self) -> bool:
        return self.exit_code == EXIT_REFUSED

    def to_dict(self) -> dict:
        rec = self.recommended
        return {
            "tool": "pigemma.feasibility",
            "version": _PIGEMMA_VERSION,
            "kind": "projected",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ok": self.ok,
            "exit_code": self.exit_code,
            "runnable": self.runnable,
            "host_refused": self.host_refused,
            "busy": self.busy,
            "refusal_reasons": list(self.refusal_reasons),
            "rows_informational_only": self.exit_code == EXIT_REFUSED,
            "want": self.want,
            "hit_source": self.hit_source,
            "host": self.host,
            "device": self.device,
            "assumptions": self.assumptions,
            "rows": self.rows,
            "recommended": rec,
            "llama_flags": rec.get("flags") if rec else None,
            "fits_repacked": bool(rec.get("fits_repacked")) if rec else None,
            "reasons": list(self.reasons),
            "recommendation_line": self.recommendation_line,
        }


def _rank(row: Mapping[str, Any]) -> int:
    v = row.get("verdict")
    if v in (FITS_MAPPED, FITS_REPACKED):
        return 4
    if v == STREAM_OK:
        return 3
    if v == STREAM_SLOW:
        return 2
    if v == NEVER_GOOD_SPEED and row.get("runnable"):
        return 1
    return 0


def _present_bytes(model_dir: str | None, model: Mapping[str, Any]) -> int:
    """Size of an already-downloaded copy of the model under model_dir (0 if none is found)."""
    if not model_dir:
        return 0
    file = str(model.get("file") or "")
    if not file:
        return 0
    base = Path(model_dir).expanduser()
    repo = str(model.get("repo") or "")
    cands = [base / file, base / Path(file).name]
    if repo:
        cands.append(base / repo.replace("/", "__") / file)
    if model.get("local_path"):
        cands.append(base / Path(str(model["local_path"])).name)
    best = 0
    for c in cands:
        try:
            if c.is_file():
                best = max(best, c.stat().st_size)
        except OSError:
            continue
    return best


def _placeholder(model: Mapping[str, Any]) -> bool:
    return "<" in str(model.get("file", "")) or model.get("bytes_exact") is False


def _select_models(catalog: Mapping[str, Any], want: str | None) -> list[dict]:
    runnable = [m for m in catalog.get("models", []) if isinstance(m, dict) and m.get("kind") != "draft"]
    if not want:
        return runnable
    w = want.strip().lower()
    by_family = [m for m in runnable if str(m.get("family", "")).lower() == w]
    if by_family:
        return by_family
    by_id = [m for m in runnable if str(m.get("id", "")).lower() == w]
    if by_id:
        return by_id
    fams = sorted({str(m.get("family")) for m in runnable})
    raise ValueError(f"--want {want!r} matches no family ({', '.join(fams)}) and no model id")


def _draft_for(catalog: Mapping[str, Any], model: Mapping[str, Any]) -> dict | None:
    """The family's MTP sidecar (first draft with spec_type draft-mtp), else the first draft."""
    ids = list(model.get("drafts") or [])
    drafts = {m.get("id"): m for m in catalog.get("models", []) if isinstance(m, dict) and m.get("kind") == "draft"}
    cands = [drafts[i] for i in ids if i in drafts]
    for d in cands:
        if d.get("spec_type") == "draft-mtp":
            return d
    return cands[0] if cands else None


def _evaluate_row(model: dict, dev: Mapping[str, Any], *, eff: float, eff_kind: str, hit_model: _HitModel,
                  viable: float, facts: Mapping[str, Any], check_present: bool) -> dict:
    mid = str(model.get("id"))
    file_bytes = int(_num(model.get("bytes")) or 0)
    nonexpert = int(_num(model.get("nonexpert_bytes")) or 0)
    expert = int(_num(model.get("expert_bytes")) or 0)
    ne_tok = int(_num(model.get("nonexpert_bytes_per_token")) or nonexpert)
    n_items = int(_num(model.get("n_moe_layers")) or 0) * int(_num(model.get("n_expert_used")) or 0)
    worst = _num(model.get("expert_bytes_per_token_worst"))
    if worst is None:
        worst = n_items * (_num(model.get("expert_unit_bytes")) or 0)
    worst = int(worst)
    is_moe = expert > 0

    n_ctx, threads = int(dev["n_ctx"]), int(dev["threads"])
    kv = kv_bytes(model, n_ctx)
    ram = int(dev["ram_total_bytes"])
    osr, comp = int(dev["os_reserve_bytes"]), int(dev["compute_buffers_bytes"])
    usable = ram - osr - kv - comp
    bw = float(_metric(dev["mem_bw_gbps"]) or DEFAULT_MEM_BW_GBPS) * 1e9 * eff
    st = dev["storage"]
    rand = float(_metric(st["rand_read_3mb_mbps"]) or 0) * 1e6
    seq = float(_metric(st["seq_read_mbps"]) or 0) * 1e6
    acc = float(_metric(st["access_ms"]) or 0) / 1e3
    st_type = str(st.get("type", "unknown"))

    row: dict[str, Any] = {
        "model_id": mid, "family": model.get("family"), "kind": model.get("kind"), "quant": model.get("quant"),
        "bytes": file_bytes, "repo": model.get("repo"), "file": model.get("file"),
        "sha256": model.get("sha256"), "revision": model.get("revision"),
        "url": HF_URL.format(repo=model.get("repo"), file=model.get("file")),
        "nonexpert_bytes": nonexpert, "expert_bytes": expert,
        "kv_bytes": kv, "os_reserve_bytes": osr, "compute_buffers_bytes": comp, "usable_bytes": usable,
        "fits_mapped": file_bytes <= usable,
        "fits_repacked": (file_bytes <= usable) and (file_bytes + REPACK_EXTRA <= usable - REPACK_HEADROOM),
        "mode": None, "expert_cache_bytes": None, "resident_fraction": None, "hit_rate": None,
        "hit_kind": None, "hit_source": None, "miss_bytes_per_token": None, "n_miss_per_token": None,
        "t_ram_s": None, "t_io_s": None, "tok_s": None, "s_per_token": None, "tok_s_kind": "projected",
        "tok_s_source": None, "runnable": False, "verdict": None, "verdict_without_disk": None,
        "flags": None, "pin_budget_bytes": None, "disk_ok": None, "present_bytes": 0,
        "placeholder": _placeholder(model), "reason": "", "details": "",
    }
    ram_txt = (f"RAM {_gb(ram)} - OS {_gb(osr)} - KV {_gb(kv)} - buffers {_gb(comp)} = usable {_gb(usable)}")
    io_kind = "measured" if all(_kind_of(_src(st[k])) == "measured"
                                for k in ("rand_read_3mb_mbps", "access_ms")) else "assumed"
    bw_kind = _kind_of(_src(dev["mem_bw_gbps"]))

    if row["fits_mapped"]:
        bytes_tok = ne_tok + worst
        tok = bw / bytes_tok if bytes_tok > 0 else float("inf")
        row.update(mode="in_ram", t_ram_s=1.0 / tok if tok else None, t_io_s=0.0, tok_s=tok,
                   s_per_token=1.0 / tok if tok else None,
                   tok_s_source=f"projected in-RAM bound: {_gb(bytes_tok)}/token at mem_bw x eff "
                                f"(mem_bw {bw_kind}, eff {eff_kind})")
        repack_txt = (f"repack copy needs {_gb(file_bytes + REPACK_EXTRA)} <= {_gb(usable - REPACK_HEADROOM)}: "
                      + ("yes" if row["fits_repacked"] else "no -> -nr (file-backed, no anonymous copy)"))
        fit_v = FITS_REPACKED if row["fits_repacked"] else FITS_MAPPED
        if not is_moe and tok < viable:
            row["verdict"] = NEVER_GOOD_SPEED
            row["runnable"] = tok >= SLOW_TOK_S
            row["reason"] = (f"dense, fits in RAM but reads {_gb(bytes_tok)}/token: {_gb(bytes_tok)} / "
                             f"{bw / 1e9:.2f} GB/s = {1.0 / tok:.2f} s/token ({tok:.2f} tok/s)")
            row["flags"] = flags_for(fit_v, fits_repacked=row["fits_repacked"], threads=threads, n_ctx=n_ctx) \
                if row["runnable"] else None
        else:
            row["verdict"] = fit_v
            row["runnable"] = True
            row["reason"] = (f"fits in RAM ({_gb(file_bytes)} <= usable {_gb(usable)}); "
                             f"{'repack fits' if row['fits_repacked'] else 'repack does not fit -> -nr'}; "
                             f"in-RAM bound {tok:.2f} tok/s")
            row["flags"] = flags_for(fit_v, fits_repacked=row["fits_repacked"], threads=threads, n_ctx=n_ctx)
            row["pin_budget_bytes"] = 0
        row["details"] = f"{ram_txt}; {repack_txt}; t_ram = {_gb(bytes_tok)} / {bw / 1e9:.2f} GB/s"
    elif not is_moe:
        s_tok = file_bytes / seq if seq > 0 else float("inf")
        tok = 1.0 / s_tok if s_tok > 0 else 0.0
        cached = max(0, usable)
        rest = max(0, file_bytes - cached)
        ideal = rest / seq if seq > 0 else float("inf")
        row.update(mode="stream_dense", t_io_s=s_tok, t_ram_s=None, tok_s=tok, s_per_token=s_tok,
                   verdict=NEVER_GOOD_SPEED, runnable=False,
                   tok_s_source=f"projected: file bytes / seq_read_mbps ({_kind_of(_src(st['seq_read_mbps']))})")
        row["reason"] = (f"dense {_gb(file_bytes)} > usable {_gb(usable)}: streams the whole file per token: "
                         f"{_gb(file_bytes)} / {seq / 1e6:.0f} MB/s = {s_tok:.1f} s/token ({tok:.3f} tok/s)")
        row["details"] = (f"{ram_txt}; even an ideal page cache holding {_gb(cached)} leaves {_gb(rest)}/token "
                          f"= {ideal:.1f} s/token")
    else:
        cache = usable - nonexpert
        row["mode"] = "stream_moe"
        row["expert_cache_bytes"] = cache
        if cache <= 0:
            row.update(verdict=NOT_VIABLE, runnable=False, tok_s=0.0,
                       reason=f"non-expert weights {_gb(nonexpert)} + KV + buffers do not fit "
                              f"(usable {_gb(usable)})")
            row["details"] = ram_txt
        else:
            frac = cache / expert if expert else 0.0
            hit, hkind, hsrc = hit_model.hit(model, cache)
            miss = (1.0 - hit) * worst
            n_miss = (1.0 - hit) * n_items
            t_io = (miss / rand if rand > 0 else float("inf")) + n_miss * acc
            t_ram = (ne_tok + hit * worst) / bw
            tok = 1.0 / (t_ram + t_io) if (t_ram + t_io) > 0 else float("inf")
            row.update(resident_fraction=frac, hit_rate=hit, hit_kind=hkind, hit_source=hsrc,
                       miss_bytes_per_token=miss, n_miss_per_token=n_miss, t_ram_s=t_ram, t_io_s=t_io,
                       tok_s=tok, s_per_token=1.0 / tok if tok else None,
                       tok_s_source=f"projected: hit {hkind}; storage {io_kind}; mem_bw {bw_kind}; "
                                    f"eff {eff_kind}")
            ok_pre = cache >= MIN_STREAM_CACHE and st_type in STREAM_STORAGE_TYPES
            if ok_pre and tok >= viable:
                row["verdict"] = STREAM_OK
            elif tok >= SLOW_TOK_S:
                row["verdict"] = STREAM_SLOW
            else:
                row["verdict"] = NOT_VIABLE
            row["runnable"] = row["verdict"] in (STREAM_OK, STREAM_SLOW)
            why = ""
            if row["verdict"] == STREAM_SLOW:
                if tok >= viable and cache < MIN_STREAM_CACHE:
                    why = f"; expert cache < {_gb(MIN_STREAM_CACHE)}"
                elif tok >= viable:
                    why = f"; storage {st_type} not nvme/usb-ssd"
            row["reason"] = (f"streams experts: cache {_gb(cache)} = {frac:.1%} of experts, hit {hit:.3f} "
                             f"({hkind}) -> {miss / 1e6:.0f} MB/token from {st_type}: {tok:.2f} tok/s{why}")
            row["details"] = (f"{ram_txt}; t_io = {miss / 1e6:.1f} MB / {rand / 1e6:.0f} MB/s + {n_miss:.1f} x "
                              f"{acc * 1e3:.2f} ms = {t_io:.3f} s; t_ram = ({_gb(ne_tok)} + {hit:.3f} x "
                              f"{_gb(worst)}) / {bw / 1e9:.2f} GB/s = {t_ram:.3f} s")
            if row["runnable"]:
                row["flags"] = flags_for(row["verdict"], threads=threads, n_ctx=n_ctx)
                avail = dev.get("ram_available_bytes")
                base = int(avail) if avail is not None else ram - osr
                row["pin_budget_bytes"] = int(max(0, min(file_bytes, base - kv - comp - PIN_HEADROOM)))

    # disk
    free = _num(facts.get("free_disk_bytes"))
    present = _present_bytes(facts.get("model_dir"), model) if check_present else 0
    row["present_bytes"] = present
    if present >= file_bytes > 0:
        row["disk_ok"] = True
    elif free is None:
        row["disk_ok"] = None
    else:
        need = DISK_FACTOR * (file_bytes - present)
        row["disk_ok"] = free >= need
        if not row["disk_ok"]:
            row["disk_need_bytes"] = int(need)
    row["verdict_without_disk"] = row["verdict"]
    if row["disk_ok"] is False and row["verdict"] in (FITS_MAPPED, FITS_REPACKED, STREAM_OK, STREAM_SLOW):
        row["verdict"] = NO_DISK
        row["runnable"] = False
        row["flags"] = None
        row["reason"] = (f"needs {_gb(DISK_FACTOR * (file_bytes - present))} free disk "
                         f"({DISK_FACTOR} x {_gb(file_bytes - present)}), have {_gb(free)}; "
                         f"otherwise {row['verdict_without_disk']}: {row['reason']}")
    elif row["disk_ok"] is False and row["verdict"] == NEVER_GOOD_SPEED and row["runnable"]:
        row["runnable"] = False
        row["flags"] = None
        row["reason"] += f"; not enough free disk ({_gb(free)})"
    if row["placeholder"]:
        row["reason"] += "; catalog entry is a placeholder (run gguf_map catalog-refresh)"
    row["eligible"] = _rank(row) >= 1 and not row["placeholder"]
    return row


def _host_summary(facts: Mapping[str, Any], dev: Mapping[str, Any], emulated: bool) -> dict:
    st = dict(facts.get("storage") or {})
    return {
        "hostname": facts.get("hostname"), "kernel": facts.get("kernel"), "page_size": facts.get("page_size"),
        "cpu_model": facts.get("cpu_model"), "nproc": facts.get("nproc"),
        "ram_total_bytes": dev.get("ram_total_bytes"), "ram_available_bytes": dev.get("ram_available_bytes"),
        "swap_total_bytes": int((_num(facts.get("swap_total_kb")) or 0) * 1024),
        "loadavg1": _num(facts.get("loadavg1")), "model_dir": facts.get("model_dir"),
        "storage": {"dev": st.get("dev"), "type": dev["storage"].get("type"), "class": dev["storage"].get("class"),
                    "pcie_link_speed": st.get("pcie_link_speed"), "pcie_link_width": st.get("pcie_link_width"),
                    "read_ahead_kb": st.get("read_ahead_kb"), "rotational": st.get("rotational")},
        "free_disk_bytes": _num(facts.get("free_disk_bytes")), "hat_product": facts.get("hat_product"),
        "pciex1_gen_cfg": facts.get("pciex1_gen_cfg"), "governor": facts.get("governor"),
        "memlock_limit": facts.get("memlock_limit"), "override": facts.get("override"),
        "emulated": bool(emulated), "collected_at": facts.get("collected_at"),
    }


def _host_notes(facts: Mapping[str, Any], dev: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    hat = facts.get("hat_product")
    if hat and re.search(r"hailo|ai\s*hat|ai\s*kit", str(hat), re.I):
        notes.append(f"AI HAT detected ({hat}): it does not help 25-30B models and occupies the single PCIe "
                     "lane; an NVMe drive needs a dual-slot HAT")
    st = dict(facts.get("storage") or {})
    gen_cfg = _num(facts.get("pciex1_gen_cfg"))
    speed = str(st.get("pcie_link_speed") or "")
    if gen_cfg == 3 and dev["storage"].get("type") == "nvme" and speed and not speed.startswith("8.0"):
        notes.append(f"config.txt asks for PCIe Gen3 but the NVMe link runs at {speed} (reboot pending or "
                     "unsupported): projected as Gen2")
    flags = [str(f).lower() for f in (facts.get("cpu_flags") or [])]
    if flags and "asimddp" not in flags:
        notes.append("CPU lacks dotprod (asimddp): llama.cpp Q4_0 kernels will be far slower than projected "
                     "(projections assume a Pi 5 Cortex-A76)")
    gov = facts.get("governor")
    if gov and str(gov) != "performance":
        notes.append(f"cpufreq governor '{gov}' (pi/01_tune_os.sh --apply can set 'performance')")
    return notes


def _memlock_note(facts: Mapping[str, Any], rec: Mapping[str, Any] | None) -> str | None:
    if not rec or "mlock" not in str(rec.get("flags") or ""):
        return None
    lim = facts.get("memlock_limit")
    if lim is None or str(lim).strip().lower() in ("unlimited", "infinity"):
        return None
    kb = _num(lim)
    if kb is None or kb * 1024 >= float(rec.get("bytes") or 0):
        return None
    return (f"memlock limit {int(kb)} kB < model {_gb(rec.get('bytes'))}: '-lm mmap+mlock' cannot lock the "
            "file (llama.cpp warns and continues unlocked); run pi/01_tune_os.sh --apply (TUNE_MEMLOCK=1) "
            "and log in again")


def verdict(facts: Mapping[str, Any] | str | os.PathLike[str],
            catalog: Mapping[str, Any] | str | os.PathLike[str] | None = None,
            want: str | None = None,
            sim: Mapping[str, Any] | str | os.PathLike[str] | None = None,
            *,
            calibration: Mapping[str, Any] | str | os.PathLike[str] | None = None,
            device: str | None = None,
            threads: int | None = None,
            n_ctx: int | None = None,
            os_reserve_bytes: int | None = None,
            extra_deny_hosts: Iterable[str] = (),
            extra_service_globs: Iterable[str] = (),
            max_loadavg1: float | None = None,
            emulated: bool | None = None,
            check_present: bool = True) -> Verdict:
    """Compute the feasibility verdict for facts (dict or path) against the catalog (dict, path or default).

    ``sim`` is a parsed sim.json (or path); ``calibration`` a parsed results/pi/calibration.json (or path).
    Raises ValueError on invalid input (unknown --want / --device, broken catalog).
    """
    facts = load_facts(facts) if not isinstance(facts, Mapping) else load_facts(dict(facts))
    if catalog is None or not isinstance(catalog, Mapping):
        catalog = load_catalog(catalog)
    sim_label = None
    if sim is not None and not isinstance(sim, Mapping):
        sim_label = str(sim)
        sim = _read_json(sim)
    cal = load_calibration(calibration)
    eff, eff_src = (cal if cal else (LLAMA_EFFICIENCY, "assumed: llama.cpp reaches 0.65 of the measured memory "
                                                        "bandwidth in decode (desktop-derived prior); "
                                                        "results/pi/calibration.json overrides"))
    sel = catalog.get("selection", {}) or {}
    viable = float(_num(sel.get("viable_stream_tok_s")) or DEFAULT_VIABLE_TOK_S)
    good = float(_num(sel.get("good_stream_tok_s")) or DEFAULT_GOOD_TOK_S)
    if emulated is None:
        emulated = bool(facts.get("emulated"))

    dev, dev_notes = resolve_device(facts, catalog, device=device, threads=threads, n_ctx=n_ctx,
                                    os_reserve_bytes=os_reserve_bytes)
    hit_model = _HitModel(catalog, sim, sim_label)
    models = _select_models(catalog, want)
    eff_kind = _kind_of(eff_src)
    rows = [_evaluate_row(m, dev, eff=eff, eff_kind=eff_kind, hit_model=hit_model, viable=viable, facts=facts,
                          check_present=check_present) for m in models]

    # ---- refusals
    refusal: list[str] = []
    host = str(facts.get("hostname") or "")
    deny = {h.strip().lower() for h in list(catalog.get("protected_hosts", []) or []) + list(extra_deny_hosts) if h}
    globs = [g.strip() for g in list(catalog.get("protected_service_globs", []) or []) + list(extra_service_globs)
             if g and g.strip()]
    protected: list[str] = []
    if host and host.lower() in deny:
        protected.append(f"hostname '{host}' is a protected production host (catalog.protected_hosts)")
    services = [str(s) for s in (facts.get("running_services") or [])]
    hits = sorted({s for s in services for g in globs if fnmatch.fnmatchcase(s.lower(), g.lower())})
    if hits:
        protected.append(f"protected services running: {', '.join(hits)} (globs {', '.join(globs)})")
    override = str(facts.get("override") or "") == OVERRIDE_TOKEN
    host_refused = bool(protected) and not override
    if host_refused:
        refusal.extend(f"REFUSED (protected host): {p}" for p in protected)
    elif protected:
        refusal.append(f"protected-host refusal overridden by {OVERRIDE_TOKEN}: " + "; ".join(protected))

    busy_msgs: list[str] = []
    nproc = int(_num(facts.get("nproc")) or BUSY_REF_NPROC)
    limit = float(max_loadavg1) if max_loadavg1 is not None else BUSY_LOADAVG1 * max(1.0, nproc / BUSY_REF_NPROC)
    la = _num(facts.get("loadavg1"))
    if la is not None and la > limit:
        busy_msgs.append(f"REFUSED (busy): loadavg1 {la:.2f} > {limit:.2f}")
    avail = dev.get("ram_available_bytes")
    if avail is not None and avail < MIN_FREE_RAM_TO_RUN:
        busy_msgs.append(f"REFUSED (busy): MemAvailable {_gb(avail)} < {_gb(MIN_FREE_RAM_TO_RUN)}")
    busy = bool(busy_msgs)
    refusal.extend(busy_msgs)
    refused = host_refused or busy

    # ---- recommendation + exit code
    eligible = [r for r in rows if r["eligible"]]
    best_rank = max((_rank(r) for r in rows if not r["placeholder"]), default=0)
    if refused:
        code = EXIT_REFUSED
    elif best_rank >= 3:
        code = EXIT_OK
    elif best_rank >= 1:
        code = EXIT_SLOW
    else:
        code = EXIT_NOT_VIABLE
    recommended: dict | None = None
    if not refused and eligible:
        best = max(eligible, key=lambda r: (_rank(r), quant_bpw(r.get("quant")), r.get("bytes") or 0,
                                            r.get("tok_s") or 0.0))
        recommended = copy.deepcopy(best)
        draft = next((m for m in models if m.get("id") == best["model_id"]), None)
        d = _draft_for(catalog, draft) if draft else None
        recommended["draft"] = None if d is None else {
            "id": d.get("id"), "repo": d.get("repo"), "file": d.get("file"), "bytes": d.get("bytes"),
            "sha256": d.get("sha256"), "spec_type": d.get("spec_type"),
            "url": HF_URL.format(repo=d.get("repo"), file=d.get("file")),
            "note": "desktop 2026-10-05 (llama-server): MTP drafts only with an explicit --spec-type draft-mtp "
                    "(auto-detect never drafted; explicit: acceptance 60-69% MoE, 39-63% dense); the 26B DFlash "
                    "draft accepted 28% and was slower - measure before enabling (-md <file> --spec-type ...)",
        }

    # ---- reasons
    reasons: list[str] = []
    if emulated:
        reasons.append("EMULATED host (PIGEMMA_EMULATED): projections describe the profile, not this machine")
    reasons.extend(refusal)
    reasons.append(f"device profile {dev['name']}: {dev['match_reason']}")
    reasons.extend(dev_notes)
    reasons.extend(hit_model.notes)
    if cal:
        reasons.append(f"llama_efficiency {eff:.3f} from calibration ({eff_src})")
    reasons.extend(_host_notes(facts, dev))
    if recommended:
        reasons.append(f"recommended {recommended['model_id']} ({recommended['quant']}, {_gb(recommended['bytes'])}): "
                       f"{recommended['verdict']}, projected {recommended['tok_s']:.2f} tok/s; "
                       f"flags: {recommended['flags']}")
        if recommended.get("tok_s") is not None and recommended["tok_s"] < good and code == EXIT_OK:
            reasons.append(f"projected speed below the pre-registered 'good' threshold ({good:g} tok/s)")
        if avail is not None and recommended["verdict"] in (FITS_MAPPED, FITS_REPACKED) and avail < recommended["bytes"]:
            reasons.append(f"MemAvailable now {_gb(avail)} < model {_gb(recommended['bytes'])}: other processes "
                           "hold RAM; loading will evict page cache and may stall")
        ml = _memlock_note(facts, recommended)
        if ml:
            reasons.append(ml)
    elif refused:
        reasons.append("no recommendation: host refused (rows shown for information only)")
    else:
        reasons.append("no viable model/quant on this hardware (see rows)")
    reasons.append("all tok/s values are PROJECTED (kind 'projected'); inputs labelled assumed are not measurements")
    reasons.append(RECOMMENDATION_LINE)

    assumptions = {
        "mem_bw_gbps": {"value": _metric(dev["mem_bw_gbps"]), "unit": "GB/s", "source": _src(dev["mem_bw_gbps"])},
        "llama_efficiency": {"value": eff, "unit": "fraction", "source": eff_src},
        "rand_read_3mb_mbps": {"value": _metric(dev["storage"]["rand_read_3mb_mbps"]), "unit": "MB/s",
                               "source": _src(dev["storage"]["rand_read_3mb_mbps"])},
        "seq_read_mbps": {"value": _metric(dev["storage"]["seq_read_mbps"]), "unit": "MB/s",
                          "source": _src(dev["storage"]["seq_read_mbps"])},
        "access_ms": {"value": _metric(dev["storage"]["access_ms"]), "unit": "ms",
                      "source": _src(dev["storage"]["access_ms"])},
        "os_reserve_bytes": {"value": dev["os_reserve_bytes"], "unit": "B", "source": dev["sources"]["os_reserve_bytes"]},
        "compute_buffers_bytes": {"value": dev["compute_buffers_bytes"], "unit": "B",
                                  "source": dev["sources"]["compute_buffers_bytes"]},
        "kv_bytes_q8_at_4096": {"value": dict(KV_Q8_4K), "unit": "B",
                                "source": "assumed: q8_0 K/V at n_ctx 4096 (gguf_map kv_bytes design numbers), "
                                          "scaled by max(1, n_ctx/4096)"},
        "hit_rate_curve": {"value": [list(p) for p in hit_model.curve], "unit": "resident fraction -> hit",
                           "source": hit_model.curve_source},
        "thresholds": {"value": {"viable_stream_tok_s": viable, "slow_tok_s": SLOW_TOK_S,
                                 "min_stream_cache_bytes": MIN_STREAM_CACHE,
                                 "min_free_ram_to_run_bytes": MIN_FREE_RAM_TO_RUN, "max_loadavg1": limit,
                                 "disk_factor": DISK_FACTOR},
                       "unit": "mixed", "source": "assumed: pre-registered policy thresholds (docs/HYPOTHESES.md)"},
    }
    return Verdict(ok=code == EXIT_OK, exit_code=code, rows=rows, recommended=recommended, reasons=reasons,
                   host_refused=host_refused, busy=busy, refusal_reasons=refusal, device=dev,
                   host=_host_summary(facts, dev, bool(emulated)), want=want,
                   hit_source="simulated" if hit_model.simulated else "assumed", assumptions=assumptions)


# ========================================================================================== output
def _tok_label(row: Mapping[str, Any]) -> str:
    tok = row.get("tok_s")
    if tok is None:
        return "-"
    src = str(row.get("tok_s_source") or "")
    if "assumed" in src:
        kind = "assumed"
    elif "simulated" in src:
        kind = "simulated"
    else:
        kind = "measured in"
    num = f"{tok:.3f}" if tok < 0.1 else f"{tok:.2f}"
    return f"{num} ({kind})"


def format_table(v: Verdict) -> str:
    """Human-readable report: host line, per-row table, details, recommendation, verdict, hardware line."""
    lines: list[str] = []
    h = v.host
    st = h.get("storage", {})
    lines.append(f"host {h.get('hostname')}: RAM {_gb(h.get('ram_total_bytes'))} total / "
                 f"{_gb(h.get('ram_available_bytes'))} available, storage {st.get('type')} ({st.get('class')}), "
                 f"free disk {_gb(h.get('free_disk_bytes'))}, page {h.get('page_size')}, "
                 f"loadavg1 {h.get('loadavg1')}, profile {v.device.get('name')}")
    if v.exit_code == EXIT_REFUSED:
        bar = "!" * 100
        lines.append(bar)
        for r in v.refusal_reasons:
            lines.append(f"!! {r}")
        lines.append("!! every heavier pi/*.sh script will refuse on this host (exit 4); rows are informational")
        lines.append(bar)
    hdr = f"{'model':<16} {'quant':<11} {'size GB':>7}  {'verdict':<17} {'tok/s (source)':<19} reason"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in v.rows:
        lines.append(f"{r['model_id']:<16} {str(r.get('quant')):<11} {r['bytes'] / GB:>7.2f}  "
                     f"{r['verdict']:<17} {_tok_label(r):<19} {r['reason']}")
    lines.append("details:")
    for r in v.rows:
        lines.append(f"  {r['model_id']}: {r['details']}")
    rec = v.recommended
    if rec:
        lines.append(f"recommended: {rec['model_id']} ({rec['quant']}, {_gb(rec['bytes'])}) {rec['verdict']}, "
                     f"projected {_tok_label(rec)} tok/s")
        lines.append(f"  flags: {rec['flags']}")
        lines.append(f"  url:   {rec['url']}")
        if rec.get("draft"):
            d = rec["draft"]
            lines.append(f"  draft: {d['id']} ({d['spec_type']}, {_gb(d['bytes'])}) - {d['note']}")
    else:
        lines.append("recommended: none")
    lines.append("reasons:")
    for r in v.reasons:
        lines.append(f"  - {r}")
    names = {EXIT_OK: "OK", EXIT_SLOW: "SLOW ONLY", EXIT_NOT_VIABLE: "NOT VIABLE", EXIT_REFUSED: "REFUSED"}
    lines.append(f"verdict: exit {v.exit_code} ({names.get(v.exit_code, '?')})")
    return "\n".join(lines) + "\n"


def emit_env(v: Verdict) -> str:
    """Shell-safe KEY=value lines (every value shlex-quoted) for `eval "$(... --emit env)"`."""
    rec = v.recommended
    d = (rec or {}).get("draft") or {}
    vals: list[tuple[str, Any]] = [
        ("MODEL_ID", rec["model_id"] if rec else ""),
        ("MODEL_REPO", rec["repo"] if rec else ""),
        ("MODEL_FILE", rec["file"] if rec else ""),
        ("MODEL_BYTES", rec["bytes"] if rec else ""),
        ("MODEL_QUANT", rec["quant"] if rec else ""),
        ("MODEL_FAMILY", rec["family"] if rec else ""),
        ("MODEL_SHA256", (rec.get("sha256") or "") if rec else ""),
        ("MODEL_REVISION", (rec.get("revision") or "") if rec else ""),
        ("MODEL_URL", rec["url"] if rec else ""),
        ("LLAMA_FLAGS", (rec.get("flags") or "") if rec else ""),
        ("FITS_REPACKED", (1 if rec.get("fits_repacked") else 0) if rec else ""),
        ("DRAFT_ID", d.get("id") or ""),
        ("DRAFT_REPO", d.get("repo") or ""),
        ("DRAFT_FILE", d.get("file") or ""),
        ("DRAFT_BYTES", d.get("bytes") or ""),
        ("DRAFT_SHA256", d.get("sha256") or ""),
        ("DRAFT_SPEC_TYPE", d.get("spec_type") or ""),
        ("PIN_BUDGET_BYTES", (rec.get("pin_budget_bytes") if rec.get("pin_budget_bytes") is not None else 0)
         if rec else ""),
        ("VERDICT", rec["verdict"] if rec else ("REFUSED" if v.exit_code == EXIT_REFUSED else NOT_VIABLE)),
        ("VERDICT_EXIT", v.exit_code),
        ("PROJECTED_TOK_S", f"{rec['tok_s']:.3f}" if rec and rec.get("tok_s") is not None else ""),
    ]
    out = []
    for k, val in vals:
        s = "" if val is None else str(val)
        out.append(f"{k}={shlex.quote(s)}")
    return "\n".join(out) + "\n"


# ============================================================================================== CLI
class _Parser(argparse.ArgumentParser):
    """argparse with usage errors mapped to exit 1 (exit 2 means STREAM_SLOW for the pi scripts)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="python3 -m pigemma.feasibility",
        description="Raspberry Pi 5 feasibility verdict for the Gemma 4 models (stdlib only, read-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes: 0 FITS_*/STREAM_OK available; 1 usage/input error; 2 only slow (STREAM_SLOW or a\n"
            "runnable NEVER_GOOD_SPEED); 3 nothing viable / NO_DISK; 4 refused (protected or busy host).\n"
            "With --emit env the report goes to stderr and stdout carries only shell-safe KEY=value lines.\n"
            "All tok/s figures are projections; see the 'assumptions' block of --json for every source."),
    )
    p.add_argument("--facts", required=True, help="facts.json from pi/lib/common.sh collect_facts ('-' = stdin)")
    p.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="catalog.json (default: %(default)s)")
    p.add_argument("--want", default=None, help="restrict rows to a family (26b|31b) or one model id")
    p.add_argument("--sim", default=None, help="results/h1/sim.json: LRU hit rates replace the assumed curve")
    p.add_argument("--json", dest="json_out", default=None, help="write the full verdict JSON here ('-' = stdout)")
    p.add_argument("--emit", choices=("env",), default=None, help="print shell-safe KEY=value lines on stdout")
    p.add_argument("--io-mem", default="auto",
                   help="measured io_mem.json to merge: auto (results/pi/io_mem.json if present) | none | PATH")
    p.add_argument("--calibration", default="auto",
                   help="llama_efficiency calibration: auto (results/pi/calibration.json if present) | none | PATH")
    p.add_argument("--device", default=None, help="force a catalog device profile by name")
    p.add_argument("--threads", type=int, default=None, help="threads for the flags (default: env THREADS or 4)")
    p.add_argument("--ctx", type=int, default=None, help="context for KV + flags (default: env CTX or profile)")
    p.add_argument("--os-reserve-gib", type=float, default=None,
                   help="OS reserve override in GiB (default: env OS_RESERVE_GIB, else profile / RAM tier)")
    p.add_argument("--max-loadavg", type=float, default=None,
                   help="busy threshold for loadavg1 (default: env PIGEMMA_MAX_LOADAVG, else 2.0 x max(1, nproc/4))")
    p.add_argument("--emulated", action="store_true", help="label as emulated (also env PIGEMMA_EMULATED=1)")
    p.add_argument("--quiet", action="store_true", help="do not print the report table")
    p.add_argument("--log-level", default="WARNING", help="logging level for stderr (default: %(default)s)")
    return p


def _env_num(env: Mapping[str, str], name: str) -> float | None:
    v = (env.get(name) or "").strip()
    return _num(v) if v else None


def _resolve_optional(arg: str | None, default: Path) -> Path | None:
    if arg is None or arg.strip().lower() == "none" or arg.strip() == "":
        return None
    if arg.strip().lower() == "auto":
        return default if default.is_file() else None
    return Path(arg)


def _write_json(path: str, data: Any) -> None:
    if path == "-":
        sys.stdout.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out)


def _utf8_streams() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    """CLI entry point; returns the verdict exit code (0/2/3/4) or 1 on errors."""
    _utf8_streams()
    env = os.environ if environ is None else environ
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
                        format="%(name)s: %(levelname)s: %(message)s", stream=sys.stderr)
    try:
        io_mem = _resolve_optional(args.io_mem, DEFAULT_IO_MEM)
        cal = _resolve_optional(args.calibration, DEFAULT_CALIBRATION)
        catalog = load_catalog(args.catalog, io_mem=io_mem)
        facts = load_facts(args.facts)
        threads = args.threads or (int(_env_num(env, "THREADS") or 0) or None)
        n_ctx = args.ctx or (int(_env_num(env, "CTX") or 0) or None)
        osr_gib = args.os_reserve_gib if args.os_reserve_gib is not None else _env_num(env, "OS_RESERVE_GIB")
        max_la = args.max_loadavg if args.max_loadavg is not None else _env_num(env, "PIGEMMA_MAX_LOADAVG")
        deny = (env.get("DENY_HOSTS") or "").replace(",", " ").split()
        globs = (env.get("PROTECTED_SERVICE_GLOBS") or "").replace(",", " ").split()
        emulated = bool(args.emulated or (env.get("PIGEMMA_EMULATED") or "").strip() in ("1", "true", "yes")
                        or facts.get("emulated"))
        v = verdict(facts, catalog, want=args.want, sim=args.sim, calibration=cal, device=args.device,
                    threads=threads, n_ctx=n_ctx,
                    os_reserve_bytes=int(osr_gib * GIB) if osr_gib is not None else None,
                    extra_deny_hosts=deny, extra_service_globs=globs, max_loadavg1=max_la, emulated=emulated)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"pigemma.feasibility: ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    data = v.to_dict()
    data["inputs"] = {"facts": str(args.facts), "catalog": str(args.catalog),
                      "io_mem": str(io_mem) if io_mem else None, "calibration": str(cal) if cal else None,
                      "sim": args.sim}
    try:
        if args.json_out:
            _write_json(args.json_out, data)
    except OSError as exc:
        print(f"pigemma.feasibility: ERROR: cannot write {args.json_out}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    report_stream = sys.stderr if (args.emit == "env" or args.json_out == "-") else sys.stdout
    if not args.quiet:
        report_stream.write(format_table(v))
    if args.emit == "env":
        sys.stdout.write(emit_env(v))
    for r in v.refusal_reasons:
        if v.exit_code == EXIT_REFUSED:
            print(f"pigemma.feasibility: {r}", file=sys.stderr)
    return v.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
