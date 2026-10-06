"""pigemma.controlled_summary - paired analysis of controlled-memory cells (stage P5 harness).

Reads results/controlled/<cell_id>/{config.json, requests.jsonl, outputs/*.json} written by
``pigemma.controlled_cell`` and produces:

* **paired comparisons** treatment vs baseline (default A) at the same memory budget, workload set
  and storage condition: pairs are matched by (repeat, request id); per pair the ratio
  treatment/baseline of decode tok/s (client-side), prefill tok/s, TTFT, total request time and
  p99 inter-token latency; the median paired ratio with a percentile bootstrap CI (resampling
  REQUESTS as clusters, all repeats of a request together), overall, per category and per request;
* **memory budget vs throughput** per (method, budget, storage);
* **physical read bytes vs throughput** per request and per (method, budget) group (decode bytes
  read from the emulated device per generated token vs decode tok/s, Spearman rho);
* **setup costs** per cell (pin/prefault, server load, warm-up) and amortized throughput;
* **exactness**: generated token ids of every request vs the method-A reference (first divergence
  index; prompts are compared too, because a follow-up prompt contains earlier replies).

Engines (brief UPDATE 2): a cell's VARIANT is its method letter for the pinned reference build
(a7b94df) and ``<engine>:<method>`` for every other build. Each variant is compared with the
reference method A, and a variant whose engine names a ``parent`` is ALSO compared with the same
method on that unmodified parent build ("vs own parent"), which is the valid comparison for a
patched engine (HotPin vs its parent, ik --prefetch-experts vs ik plain).

Decision flags follow the brief's section 8 project thresholds (1.5x median paired decode speedup
= "promising"; >= 2 tok/s and >= 2x = "strong"); the regression limits for prefill and tail
latency are parameters (the brief does not quantify "substantial") and are printed with the result.

CLI::

    python -m pigemma.controlled_summary summarize [--root results/controlled] [--cells a,b,...]
        [--baseline A] [--out results/controlled/summary.json] [--md results/controlled/summary.md]
    python -m pigemma.controlled_summary exactness [--root ...] [--cells ...]

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "results" / "controlled"

# metric name -> (record key, higher_is_better)
METRICS: dict[str, tuple[str, bool]] = {
    "decode_tok_s": ("decode_tok_s_client", True),
    "prefill_tok_s": ("prefill_tok_s_client", True),
    "ttft_s": ("ttft_s", False),
    "total_s": ("total_s", False),
    "itl_p99_s": ("itl_p99_s", False),
    "itl_median_s": ("itl_median_s", False),
}
PROMISING_SPEEDUP = 1.5
STRONG_SPEEDUP = 2.0
STRONG_ABS_TOK_S = 2.0


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------


def load_cell(d: Path) -> dict[str, Any] | None:
    cfgp = d / "config.json"
    if not cfgp.exists():
        return None
    cfg = json.loads(cfgp.read_text("utf-8"))
    reqs: list[dict[str, Any]] = []
    rp = d / "requests.jsonl"
    if rp.exists():
        for line in rp.read_text("utf-8").splitlines():
            if line.strip():
                reqs.append(json.loads(line))
    return {"dir": d, "config": cfg, "requests": reqs}


def load_cells(root: Path, ids: Sequence[str] | None = None, include_failed: bool = False,
               include_smoke: bool = False) -> list[dict[str, Any]]:
    """Cells under root (directories starting with '_' are dev scratch). Failed cells and
    ``purpose == "smoke"`` cells are skipped unless asked for."""
    out = []
    dirs = [root / i for i in ids] if ids else sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    for d in dirs:
        c = load_cell(d)
        if c is None:
            continue
        if c["config"].get("status") != "ok" and not include_failed:
            continue
        if c["config"].get("purpose") == "smoke" and not include_smoke:
            continue
        out.append(c)
    return out


def condition(cfg: Mapping[str, Any]) -> tuple[Any, ...]:
    """What must be identical for two cells to be paired (method and repeat excluded): budget,
    workload set, storage mode and rate, the per-request token cap and the conversation subset."""
    st = cfg.get("storage", {})
    wl = cfg.get("workload", {})
    convs = wl.get("conversations")
    return (int(cfg["budget_bytes"]), wl["set"], st.get("mode", "?"), float(st.get("rate_bps") or 0.0),
            wl.get("n_predict_cap"), ",".join(convs) if convs else None)


REFERENCE_ENGINE = "a7b94df"
REFERENCE_IDS = (None, "", REFERENCE_ENGINE, "llama.cpp-a7b94df")  # names of the unpatched reference binary


def variant(cfg: Mapping[str, Any]) -> str:
    """'A'..'D' for the reference engine, '<engine>:<method>' for any other build."""
    v = cfg.get("variant")
    if v:
        return str(v)
    eid = (cfg.get("engine") or {}).get("id")
    return cfg["method"] if eid in REFERENCE_IDS else f"{eid}:{cfg['method']}"


def parent_variant(cfg: Mapping[str, Any]) -> str | None:
    """The same method on the engine's unmodified parent build, or None when no parent is named."""
    p = (cfg.get("engine") or {}).get("parent")
    if not p:
        return None
    return cfg["method"] if p in REFERENCE_IDS else f"{p}:{cfg['method']}"


def cond_label(c: tuple[Any, ...]) -> str:
    s = f"{c[0] / 1024**3:g}GiB/{c[1]}/{c[2]}" + (f"@{c[3] / 1e6:g}MB/s" if c[3] else "")
    if len(c) > 4 and c[4]:
        s += f"/cap{c[4]}"
    if len(c) > 5 and c[5]:
        s += f"/subset[{c[5]}]"
    return s


def workload_requests(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [r for r in cell["requests"] if r.get("kind") == "workload"]


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------


def median(xs: Sequence[float]) -> float | None:
    return statistics.median(xs) if xs else None


def bootstrap_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float] = statistics.median,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI of stat(values) (plain resampling of the values)."""
    if len(values) < 2:
        return (None, None)
    rng = random.Random(seed)
    n = len(values)
    bs = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = bs[int(math.floor(alpha / 2 * (n_boot - 1)))]
    hi = bs[int(math.ceil((1 - alpha / 2) * (n_boot - 1)))]
    return (lo, hi)


def cluster_bootstrap_ci(clusters: Mapping[str, Sequence[float]], stat: Callable[[Sequence[float]], float] = statistics.median,
                         n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float | None, float | None]:
    """Bootstrap resampling whole clusters (requests), keeping all repeats of a request together."""
    keys = [k for k, v in clusters.items() if v]
    if len(keys) < 2:
        flat = [x for v in clusters.values() for x in v]
        return bootstrap_ci(flat, stat, n_boot, alpha, seed)
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in keys:
            sample.extend(clusters[keys[rng.randrange(len(keys))]])
        bs.append(stat(sample))
    bs.sort()
    return (bs[int(math.floor(alpha / 2 * (n_boot - 1)))], bs[int(math.ceil((1 - alpha / 2) * (n_boot - 1)))])


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None

    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


# ----------------------------------------------------------------------------------------------
# paired comparisons
# ----------------------------------------------------------------------------------------------


def _index(cells: Sequence[Mapping[str, Any]]) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[str, str]]:
    """({(condition, variant, repeat, rid): request record}, {variant: its parent variant})."""
    idx: dict[tuple[Any, ...], dict[str, Any]] = {}
    parents: dict[str, str] = {}
    for c in cells:
        cfg = c["config"]
        v = variant(cfg)
        pv = parent_variant(cfg)
        if pv and pv != v:
            parents[v] = pv
        hc = _host_cpu_series(c)
        for r in workload_requests(c):
            key = (condition(cfg), v, int(cfg.get("repeat", 0)), r["rid"])
            if key in idx:
                raise ValueError(f"duplicate cell for {key}: {c['dir']} - give every repeat its own --repeat")
            idx[key] = dict(r, _cell=cfg["cell_id"], _host_cpu_pct=_host_cpu_for(hc, r))
    return idx, parents


def _pairs(idx: Mapping[tuple[Any, ...], Mapping[str, Any]], treat: str, base: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for (cond, v, rep, rid), rec in idx.items():
        if v != treat:
            continue
        b = idx.get((cond, base, rep, rid))
        if b is None:
            continue
        out[cond].append({"rid": rid, "repeat": rep, "base": b, "treat": rec,
                          "category": rec.get("category"), "conv_id": rec.get("conv_id"),
                          "turn": rec.get("turn")})
    return out


def pair_requests(cells: Sequence[Mapping[str, Any]], baseline: str = "A") -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """{(condition, treatment variant): [pair, ...]} vs the baseline variant (default: method A of
    the reference engine); pair = {rid, repeat, base, treat, category, conv_id}."""
    idx, _ = _index(cells)
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for v in sorted({k[1] for k in idx}):
        if v == baseline:
            continue
        for cond, ps in _pairs(idx, v, baseline).items():
            out[(cond, v)] = ps
    return out


def pair_requests_vs_parent(cells: Sequence[Mapping[str, Any]], exclude_baseline: str | None = "A",
                            ) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """{(condition, treatment variant, parent variant): [pair, ...]} for every variant whose engine
    names an unmodified parent build. Pairs already covered by the baseline comparison (parent
    variant == exclude_baseline) are left out."""
    idx, parents = _index(cells)
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for v, pv in sorted(parents.items()):
        if pv == exclude_baseline:
            continue
        for cond, ps in _pairs(idx, v, pv).items():
            out[(cond, v, pv)] = ps
    return out


# ----------------------------------------------------------------------------------------------
# background CPU of the host next to every request (host_cpu.csv is sampled by the Windows side)
# ----------------------------------------------------------------------------------------------


def _host_cpu_series(cell: Mapping[str, Any]) -> tuple[float, list[tuple[float, float]]] | None:
    """(t0_epoch, [(t_s, host_cpu_pct), ...]) of a cell, or None when it was not recorded."""
    cfg = cell["config"]
    host = cfg.get("host") or {}
    t0 = host.get("host_cpu_t0_epoch") or (host.get("host_cpu_during_cell") or {}).get("t0_epoch")
    p = Path(cell["dir"]) / "host_cpu.csv"
    if not p.exists():
        return None
    rows: list[tuple[float, float]] = []
    for line in p.read_text("utf-8", "replace").splitlines():
        if line.startswith("#"):
            if t0 is None and "t0_epoch=" in line:
                try:
                    t0 = float(line.split("t0_epoch=")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            continue
        f = line.split(",")
        try:
            rows.append((float(f[0]), float(f[1])))
        except (ValueError, IndexError):
            continue
    if t0 is None or not rows:
        return None
    return float(t0), rows


def _host_cpu_for(series: tuple[float, list[tuple[float, float]]] | None, rec: Mapping[str, Any]) -> float | None:
    """Mean host CPU % (all cores, the cell's own <= 4 of 12 logical CPUs included) over a request.

    The request's start is the VM clock, the samples are the Windows clock: an offset of a second or
    two is possible, so this is a background-load indicator, not an exact attribution."""
    if series is None or rec.get("t_start_epoch") is None or rec.get("total_s") is None:
        return None
    t0, rows = series
    a = float(rec["t_start_epoch"]) - t0
    b = a + float(rec["total_s"])
    xs = [v for t, v in rows if a <= t <= b + 1.0]
    return round(sum(xs) / len(xs), 1) if xs else None


def ratio(t: float | None, b: float | None) -> float | None:
    if t is None or b is None or b == 0:
        return None
    return t / b


def summarize_pairs(pairs: Sequence[Mapping[str, Any]], *, n_boot: int = 2000, seed: int = 0,
                    max_prefill_regression: float = 0.10, max_tail_regression: float = 0.10) -> dict[str, Any]:
    res: dict[str, Any] = {"n_pairs": len(pairs), "n_requests": len({p["rid"] for p in pairs}),
                           "repeats": sorted({p["repeat"] for p in pairs}), "metrics": {}}
    for name, (key, higher) in METRICS.items():
        by_rid: dict[str, list[float]] = defaultdict(list)
        for p in pairs:
            q = ratio(p["treat"].get(key), p["base"].get(key))
            if q is not None:
                by_rid[p["rid"]].append(q)
        flat = [x for v in by_rid.values() for x in v]
        lo, hi = cluster_bootstrap_ci(by_rid, n_boot=n_boot, seed=seed)
        res["metrics"][name] = {"ratio_definition": f"treatment/baseline of {key} ({'higher' if higher else 'lower'} is better)",
                                "n": len(flat), "median_ratio": median(flat), "ci95": [lo, hi],
                                "min": min(flat) if flat else None, "max": max(flat) if flat else None,
                                "share_better": (sum(1 for x in flat if (x > 1) == higher and x != 1) / len(flat)) if flat else None}
    # absolute decode rates
    t_dec = [p["treat"].get("decode_tok_s_client") for p in pairs if p["treat"].get("decode_tok_s_client") is not None]
    b_dec = [p["base"].get("decode_tok_s_client") for p in pairs if p["base"].get("decode_tok_s_client") is not None]
    res["decode_tok_s_treatment_median"] = median(t_dec)
    res["decode_tok_s_baseline_median"] = median(b_dec)
    for side in ("treat", "base"):
        hc = [p[side]["_host_cpu_pct"] for p in pairs if p[side].get("_host_cpu_pct") is not None]
        vo = [p[side]["bg_vm_other_cores_decode"] for p in pairs if p[side].get("bg_vm_other_cores_decode") is not None]
        res[f"background_{'treatment' if side == 'treat' else 'baseline'}"] = {
            "host_cpu_pct_mean_over_requests": round(sum(hc) / len(hc), 1) if hc else None,
            "host_cpu_pct_max_request": max(hc) if hc else None,
            "vm_other_cores_decode_mean": round(sum(vo) / len(vo), 3) if vo else None,
            "note": "host % includes the cell's own <= 33 %; vm_other = CPU cores busy in the VM outside the cell"}
    # per category / per request
    for group in ("category", "rid"):
        g: dict[str, list[float]] = defaultdict(list)
        for p in pairs:
            q = ratio(p["treat"].get("decode_tok_s_client"), p["base"].get("decode_tok_s_client"))
            if q is not None:
                g[str(p[group])].append(q)
        res[f"decode_ratio_by_{group}"] = {k: {"n": len(v), "median": median(v),
                                               "ci95": list(bootstrap_ci(v, n_boot=n_boot, seed=seed)),
                                               "min": min(v), "max": max(v)} for k, v in sorted(g.items())}
    m = res["metrics"]
    dec = m["decode_tok_s"]["median_ratio"]
    pre = m["prefill_tok_s"]["median_ratio"]
    tail = m["itl_p99_s"]["median_ratio"]
    no_prefill_reg = pre is not None and pre >= 1.0 - max_prefill_regression
    no_tail_reg = tail is not None and tail <= 1.0 + max_tail_regression
    res["flags"] = {
        "thresholds": {"promising_speedup": PROMISING_SPEEDUP, "strong_speedup": STRONG_SPEEDUP,
                       "strong_abs_tok_s": STRONG_ABS_TOK_S, "max_prefill_regression": max_prefill_regression,
                       "max_tail_regression": max_tail_regression,
                       "note": "regression limits are harness parameters (the brief does not quantify 'substantial')"},
        "median_decode_speedup": dec,
        "no_prefill_regression": no_prefill_reg, "no_tail_regression": no_tail_reg,
        "promising": bool(dec is not None and dec >= PROMISING_SPEEDUP and no_prefill_reg and no_tail_reg),
        "strong": bool(dec is not None and dec >= STRONG_SPEEDUP and (median(t_dec) or 0) >= STRONG_ABS_TOK_S
                       and no_prefill_reg and no_tail_reg),
        "incremental": bool(dec is not None and 1.0 < dec < PROMISING_SPEEDUP),
        "regression": bool(dec is not None and dec < 1.0),
    }
    return res


# ----------------------------------------------------------------------------------------------
# other tables
# ----------------------------------------------------------------------------------------------


def budget_table(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    g: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for c in cells:
        cfg = c["config"]
        g[(variant(cfg),) + condition(cfg)].append(c)
    rows = []
    for (meth, budget, wset, mode, rate, *_rest), cs in sorted(g.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        dec = [r["decode_tok_s_client"] for c in cs for r in workload_requests(c) if r.get("decode_tok_s_client") is not None]
        ttft = [r["ttft_s"] for c in cs for r in workload_requests(c) if r.get("ttft_s") is not None]
        peaks = [((c["config"].get("effective") or {}).get("max_usage_by_phase_sampled") or {}) for c in cs]
        peak = max((max(p.values()) for p in peaks if p), default=None)
        locked = [(((c["config"].get("effective") or {}).get("setup") or {}).get("pinner_summary") or {}).get("locked_bytes") for c in cs]
        q = sorted(dec)
        pre = [r["prefill_tok_s_client"] for c in cs for r in workload_requests(c) if r.get("prefill_tok_s_client") is not None]
        hostc = [((c["config"].get("host") or {}).get("host_cpu_during_cell") or {}).get("host_cpu_pct_mean") for c in cs]
        hostc = [x for x in hostc if x is not None]
        rows.append({"method": meth, "budget_gib": budget / 1024**3, "workload_set": wset, "storage": mode,
                     "rate_MBps": rate / 1e6 if rate else None, "n_cells": len(cs), "n_requests": len(dec),
                     "decode_tok_s_median": median(dec), "prefill_tok_s_median": median(pre),
                     "decode_tok_s_per_cell_median": [median([r["decode_tok_s_client"] for r in workload_requests(c)
                                                              if r.get("decode_tok_s_client") is not None]) for c in cs],
                     "host_cpu_pct_mean_per_cell": hostc or None,
                     "decode_tok_s_iqr": [q[len(q) // 4], q[(3 * len(q)) // 4]] if len(q) >= 4 else None,
                     "ttft_s_median": median(ttft), "cg_usage_peak_sampled_bytes": peak,
                     "locked_bytes": [x for x in locked if x is not None] or None})
    return rows


def _first(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def read_bytes_table(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pts = []
    for c in cells:
        cfg = c["config"]
        for r in workload_requests(c):
            d = r.get("io_decode") or {}
            p = r.get("io_prefill") or {}
            pts.append({"cell": cfg["cell_id"], "method": variant(cfg), "budget_gib": cfg["budget_bytes"] / 1024**3,
                        "rid": r["rid"], "decode_tok_s": r.get("decode_tok_s_client"),
                        "prefill_tok_s": r.get("prefill_tok_s_client"), "ttft_s": r.get("ttft_s"),
                        "input_tokens_new": r.get("prompt_n_new"),
                        "decode_bytes_per_token": _first(r.get("decode_device_bytes_per_token"),
                                                         r.get("decode_backing_bytes_per_token")),
                        "decode_backing_bytes": _first(d.get("device_read_bytes"), d.get("fuse_backing_bytes")),
                        "prefill_backing_bytes": _first(p.get("device_read_bytes"), p.get("fuse_backing_bytes")),
                        "read_source": r.get("device_read_source") or ("fuse_backing_bytes" if d.get("fuse_backing_bytes") is not None else None),
                        "decode_majflt": d.get("server_majflt"), "output_tokens": r.get("output_tokens")})
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for p in pts:
        groups[(p["method"], p["budget_gib"])].append(p)
    agg = []
    for (m, b), v in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        bpt = [x["decode_bytes_per_token"] for x in v if x["decode_bytes_per_token"] is not None]
        dec = [x["decode_tok_s"] for x in v if x["decode_tok_s"] is not None]
        pfb = [x["prefill_backing_bytes"] for x in v if x["prefill_backing_bytes"] is not None]
        agg.append({"method": m, "budget_gib": b, "n": len(v),
                    "decode_MB_per_token_median": (median(bpt) or 0) / 1e6 if bpt else None,
                    "decode_tok_s_median": median(dec),
                    "prefill_GB_per_request_median": (median(pfb) or 0) / 1e9 if pfb else None})
    rho = spearman([p["decode_bytes_per_token"] for p in pts], [p["decode_tok_s"] for p in pts])
    rho_by = {}
    for (m, b), v in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rho_by[f"{m}@{b:g}GiB"] = spearman([x["decode_bytes_per_token"] for x in v], [x["decode_tok_s"] for x in v])
    return {"points": pts, "by_method_budget": agg, "spearman_bytes_per_token_vs_decode_tok_s": rho,
            "spearman_within_method_budget": rho_by,
            "read_sources": sorted({p["read_source"] for p in pts if p.get("read_source")}),
            "note": "bytes = physical reads of the cell per phase from the most exact source available "
                    "(read_sources): the emulated device's counter in FUSE mode, else cgroup io.stat rbytes, "
                    "else /proc/<pid>/io read_bytes of server+pinner; page-fault counts are never used as bytes"}


def setup_table(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for c in cells:
        cfg = c["config"]
        eff = cfg.get("effective") or {}
        ph = eff.get("phases") or {}
        wl = workload_requests(c)
        toks = sum(int(r.get("output_tokens") or 0) for r in wl)
        work_s = sum(float(r.get("total_s") or 0) for r in wl)
        setup = float(ph.get("setup_s") or 0) + float(ph.get("load_s") or 0) + float(ph.get("warmup_s") or 0)
        la = eff.get("lock_accounting") or {}
        for k in ("setup", "load", "warmup"):  # unified physical-read column where the cell has it
            if ph.get(f"{k}_device_read_bytes") is not None:
                ph = dict(ph, **{f"{k}_backing_bytes": ph[f"{k}_device_read_bytes"]})
        rows.append({"cell": cfg["cell_id"], "method": variant(cfg), "budget_gib": cfg["budget_bytes"] / 1024**3,
                     "requested_lock_bytes": la.get("requested_bytes_page_aligned_union"),
                     "locked_bytes": la.get("pinner_locked_bytes"),
                     "effective_lock_verified": la.get("effective_lock_verified"),
                     "repeat": cfg.get("repeat"), "setup_s": ph.get("setup_s"), "setup_backing_bytes": ph.get("setup_backing_bytes"),
                     "load_s": ph.get("load_s"), "load_backing_bytes": ph.get("load_backing_bytes"),
                     "warmup_s": ph.get("warmup_s"), "warmup_backing_bytes": ph.get("warmup_backing_bytes"),
                     "workload_request_s": round(work_s, 3), "output_tokens": toks,
                     "amortized_tok_s": round(toks / (setup + work_s), 4) if (setup + work_s) > 0 else None,
                     "host_cpu_pct_mean": ((cfg.get("host") or {}).get("host_cpu_during_cell") or {}).get("host_cpu_pct_mean")})
    return rows


# ----------------------------------------------------------------------------------------------
# exactness
# ----------------------------------------------------------------------------------------------


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Index of the first differing token; len(shorter) if one is a prefix; None if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _outputs(cell: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    d = Path(cell["dir"]) / "outputs"
    for r in workload_requests(cell):
        safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in r["rid"])
        p = d / f"{safe}.json"
        if p.exists():
            out[r["rid"]] = json.loads(p.read_text("utf-8"))
    return out


def _same_requests(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Two cells ran the same requests: same set, conversation subset and per-request token cap."""
    wa, wb = a.get("workload", {}), b.get("workload", {})
    return (wa.get("set"), wa.get("n_predict_cap"), tuple(wa.get("conversations") or ())) == \
           (wb.get("set"), wb.get("n_predict_cap"), tuple(wb.get("conversations") or ()))


def exactness(cells: Sequence[Mapping[str, Any]], ref_method: str = "A") -> dict[str, Any]:
    """Every cell's per-request token ids (and output text) vs a reference cell that ran the same
    requests: the method-A cell of the SAME engine when one exists (a cache method must not change
    its own engine's output), else the engine's unmodified parent, else the reference engine's A
    (cross-engine: numerical differences between revisions are expected and are reported, they are
    not a cache failure). Same budget and repeat are preferred. A-vs-A checks determinism."""
    report: dict[str, Any] = {"ref_method": ref_method, "cells": []}
    for c in cells:
        cfg = c["config"]
        eid = (cfg.get("engine") or {}).get("id") or REFERENCE_ENGINE
        own_ref = ref_method if eid in REFERENCE_IDS else f"{eid}:{ref_method}"
        chain = [(own_ref, "same engine")]
        pv = parent_variant(dict(cfg, method=ref_method))
        if pv and pv != own_ref:
            chain.append((pv, "unmodified parent engine"))
        if ref_method not in [x[0] for x in chain]:
            chain.append((ref_method, "reference engine (cross-engine)"))
        ref, kind = None, None
        for want, k in chain:
            cand = [r for r in cells if r is not c and variant(r["config"]) == want and _same_requests(r["config"], cfg)]
            if cand:
                cand.sort(key=lambda r: (r["config"]["budget_bytes"] != cfg["budget_bytes"],
                                         r["config"].get("repeat") != cfg.get("repeat")))
                ref, kind = cand[0], k
                break
        if ref is None:
            continue
        ro, co = _outputs(ref), _outputs(c)
        per = []
        for rid, o in co.items():
            if rid not in ro:
                continue
            fd = first_divergence(ro[rid]["token_ids"], o["token_ids"])
            ta, tb = ro[rid].get("text"), o.get("text")
            per.append({"rid": rid, "identical": fd is None, "first_divergence": fd,
                        "text_identical": (ta == tb) if ta is not None and tb is not None else None,
                        "prompt_identical": ro[rid].get("prompt_sha256") == o.get("prompt_sha256"),
                        "token_ids_source": o.get("token_ids_source"),
                        "n_ref": len(ro[rid]["token_ids"]), "n": len(o["token_ids"])})
        report["cells"].append({"cell": cfg["cell_id"], "method": variant(cfg), "ref_cell": ref["config"]["cell_id"],
                                "ref_variant": variant(ref["config"]), "ref_kind": kind,
                                "n_compared": len(per), "n_identical": sum(1 for p in per if p["identical"]),
                                "all_identical": all(p["identical"] for p in per) if per else None,
                                "divergent": [p for p in per if not p["identical"] or not p["prompt_identical"]
                                              or p["text_identical"] is False]})
    return report


# ----------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------


def summarize(cells: Sequence[Mapping[str, Any]], baseline: str = "A", **kw: Any) -> dict[str, Any]:
    pairs = pair_requests(cells, baseline)
    comps = []
    for (cond, meth), ps in sorted(pairs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        s = summarize_pairs(ps, **kw)
        s.update({"condition": cond_label(cond), "treatment": meth, "baseline": baseline,
                  "comparison": "vs reference baseline"})
        comps.append(s)
    for (cond, meth, pv), ps in sorted(pair_requests_vs_parent(cells, baseline).items()):
        s = summarize_pairs(ps, **kw)
        s.update({"condition": cond_label(cond), "treatment": meth, "baseline": pv,
                  "comparison": "vs own unmodified parent engine"})
        comps.append(s)
    return {"schema": "pigemma-controlled-summary v1", "n_cells": len(cells),
            "cells": [c["config"]["cell_id"] for c in cells],
            "paired": comps, "budget_vs_throughput": budget_table(cells),
            "read_bytes_vs_throughput": read_bytes_table(cells), "setup_costs": setup_table(cells),
            "exactness": exactness(cells, baseline)}


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def to_markdown(s: Mapping[str, Any]) -> str:
    L = ["# Controlled-memory cells - summary", "",
         f"{s['n_cells']} cells. All numbers MEASURED in those cells; ratios are treatment/baseline per paired request.", ""]
    L += ["## Paired comparisons (median paired ratio, 95 % cluster-bootstrap CI over requests)", "",
          "| condition | treatment vs base | pairs | decode x | decode CI | prefill x | TTFT x | ITL p99 x | flags |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in s["paired"]:
        m = c["metrics"]
        fl = c["flags"]
        flag = "strong" if fl["strong"] else "promising" if fl["promising"] else "regression" if fl["regression"] else \
            "incremental" if fl["incremental"] else "-"
        ci = m["decode_tok_s"]["ci95"]
        L.append(f"| {c['condition']} | {c['treatment']} vs {c['baseline']} | {c['n_pairs']} | "
                 f"{_f(m['decode_tok_s']['median_ratio'])} | [{_f(ci[0])}, {_f(ci[1])}] | "
                 f"{_f(m['prefill_tok_s']['median_ratio'])} | {_f(m['ttft_s']['median_ratio'])} | "
                 f"{_f(m['itl_p99_s']['median_ratio'])} | {flag} |")
    L += ["", "Background load next to the paired numbers (host CPU % over the requests, cell included; "
          "CPU cores busy in the VM outside the cell during decode):", "",
          "| condition | treatment vs base | host % treat | host % base | VM-other cores treat | VM-other cores base |",
          "|---|---|---|---|---|---|"]
    for c in s["paired"]:
        bt, bb = c.get("background_treatment") or {}, c.get("background_baseline") or {}
        L.append(f"| {c['condition']} | {c['treatment']} vs {c['baseline']} | {_f(bt.get('host_cpu_pct_mean_over_requests'), 1)} | "
                 f"{_f(bb.get('host_cpu_pct_mean_over_requests'), 1)} | {_f(bt.get('vm_other_cores_decode_mean'))} | "
                 f"{_f(bb.get('vm_other_cores_decode_mean'))} |")
    L += ["", "## Paired decode ratio per workload category (median, bootstrap 95 % CI, min-max)", "",
          "| condition | treatment vs base | category | pairs | decode x | CI | min | max |", "|---|---|---|---|---|---|---|---|"]
    for c in s["paired"]:
        for cat, v in (c.get("decode_ratio_by_category") or {}).items():
            L.append(f"| {c['condition']} | {c['treatment']} vs {c['baseline']} | {cat} | {v['n']} | {_f(v['median'])} | "
                     f"[{_f(v['ci95'][0])}, {_f(v['ci95'][1])}] | {_f(v['min'])} | {_f(v['max'])} |")
    L += ["", "## Memory budget vs throughput", "",
          "| method | budget GiB | storage | cells | requests | decode tok/s median | IQR | TTFT s median | peak cgroup (sampled) GB |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in s["budget_vs_throughput"]:
        L.append(f"| {r['method']} | {r['budget_gib']:g} | {r['storage']} | {r['n_cells']} | {r['n_requests']} | "
                 f"{_f(r['decode_tok_s_median'])} | {r['decode_tok_s_iqr'] or '-'} | {_f(r['ttft_s_median'])} | "
                 f"{_f((r['cg_usage_peak_sampled_bytes'] or 0) / 1e9 if r['cg_usage_peak_sampled_bytes'] else None, 2)} |")
    rb = s["read_bytes_vs_throughput"]
    L += ["", "## Physical reads vs throughput", "",
          f"Spearman rho(decode bytes/token, decode tok/s) = {_f(rb['spearman_bytes_per_token_vs_decode_tok_s'])}", "",
          "| method | budget GiB | requests | decode MB/token median | decode tok/s median |", "|---|---|---|---|---|"]
    for r in rb["by_method_budget"]:
        L.append(f"| {r['method']} | {r['budget_gib']:g} | {r['n']} | {_f(r['decode_MB_per_token_median'], 1)} | {_f(r['decode_tok_s_median'])} |")
    L += ["", "## Setup costs and amortized throughput", "",
          "| cell | method | setup s | setup GB read | load s | load GB read | warm-up s | request s | tokens | amortized tok/s | host CPU % |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["setup_costs"]:
        gb = lambda x: _f(x / 1e9, 2) if x is not None else "-"  # noqa: E731
        L.append(f"| {r['cell']} | {r['method']} | {_f(r['setup_s'], 1)} | {gb(r['setup_backing_bytes'])} | {_f(r['load_s'], 1)} | "
                 f"{gb(r['load_backing_bytes'])} | {_f(r['warmup_s'], 1)} | {_f(r['workload_request_s'], 1)} | {r['output_tokens']} | "
                 f"{_f(r['amortized_tok_s'])} | {_f(r['host_cpu_pct_mean'], 1)} |")
    ex = s["exactness"]
    L += ["", f"## Exactness (token ids vs method {ex['ref_method']} reference)", "",
          "| cell | method | reference | compared | identical | first divergences |", "|---|---|---|---|---|---|"]
    for c in ex["cells"]:
        dv = ", ".join(f"{d['rid']}@{d['first_divergence']}" for d in c["divergent"][:6]) or "-"
        L.append(f"| {c['cell']} | {c['method']} | {c['ref_cell']} | {c['n_compared']} | {c['n_identical']} | {dv} |")
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pigemma.controlled_summary")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("summarize", "exactness"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=str(DEFAULT_ROOT))
        p.add_argument("--cells")
        p.add_argument("--baseline", default="A")
        p.add_argument("--include-failed", action="store_true")
        p.add_argument("--include-smoke", action="store_true")
        p.add_argument("--out")
        p.add_argument("--md")
        p.add_argument("--n-boot", type=int, default=2000)
        p.add_argument("--max-prefill-regression", type=float, default=0.10)
        p.add_argument("--max-tail-regression", type=float, default=0.10)
    a = ap.parse_args(argv)
    cells = load_cells(Path(a.root), a.cells.split(",") if a.cells else None, a.include_failed, a.include_smoke)
    if a.cmd == "exactness":
        rep = exactness(cells, a.baseline)
    else:
        rep = summarize(cells, a.baseline, n_boot=a.n_boot, max_prefill_regression=a.max_prefill_regression,
                        max_tail_regression=a.max_tail_regression)
    text = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    if a.md and a.cmd == "summarize":
        Path(a.md).write_text(to_markdown(rep), encoding="utf-8")
    if not a.out:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
