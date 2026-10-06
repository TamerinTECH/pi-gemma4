"""pigemma.hotrank - static hot-expert ranking from CALIBRATION routing traces, the size-matched
random control, and calibration-vs-held-out routing comparisons (stage T1 of the decision
workflow; brief 2026-10-05 sections 3 and 5).

Inputs are server-session traces written by ``pigemma.gentrace`` (probe 1.1.0): a ``.moetrace``
with one chunk per request and the ``<trace>.requests.jsonl`` sidecar that gives the prefill
``llama_process`` calls of every request (needed to cut the prefill into ubatches).

Access events (what a retained expert can save)
-----------------------------------------------
An expert's weights are read from the page cache once per evaluation step that routes at least
one token to it (ggml ``mul_mat_id`` groups the tokens of a ubatch by expert), so the unit of a
possible fetch is the *evaluation step*, not the token:

* decode event   - a generated-token evaluation (phase 1 record) that selects the expert;
* prefill event  - a prefill ubatch (consecutive ``n_ubatch`` pieces of each llama_process call)
                   in which at least one token selects the expert.

Token-level prefill counts are kept for reference only (they measure compute, not fetches).

Score (brief section 5: "expected avoided fetch cost per retained byte")
-------------------------------------------------------------------------
For expert x = (layer, expert), page size P::

    F(x)        = (w_dec * n_decode_events(x) + w_pre * n_prefill_events(x)) / n_requests
                  expected access events per calibration request            [MEASURED counts]
    retained(x) = unique pages of x's three slices * P                      [exact, from the map]
    fetch(x)    = retained(x)   (bytes read when x is entirely absent)      [exact, from the map]
    runs(x)     = discontiguous page runs of x's weight slices (2 here: down, gate_up; the
                  4-byte scale entry shares one always-hot page per layer)  [exact, from the map]
    c_run       = rate * t_op + readaround_overshoot                        [ASSUMED, documented]
    score(x)    = F(x) * (fetch(x) + runs(x) * c_run) / retained(x)

``w_dec = w_pre = 1`` treats every access event as equally likely to be a miss (ASSUMED: the
static ranking has no cache-state model). Because every expert of layers 0-28 has the same size
in this file and layer 29 is larger, the cost factor is nearly constant within those two groups:
the order is driven by F, and the per-fetch term only discounts layer 29 slightly. Variants
(decode-only frequency, c_run = 0, 16 KiB pages) are stored with their agreement with the primary.

Only a trace of the ``calibration`` set may be ranked (``build`` refuses anything else); the
output carries ``"set": "calibration"``, which ``pigemma.controlled.load_ranking`` checks.

CLI (``python -m pigemma.hotrank <cmd> --help``)::

    build    --trace traces/q3km/calibration.moetrace --map results/maps/map26_ud_q3_k_m.json
             --out results/rankings/hot_q3km_calibration.json
    control  --ranking R --map M --expert-pin-bytes N --out-prefix P [--seed S] [--page-size 4096]
             the C plan (ranked prefix) and the size-matched random D plan (page-accounted), both
             through pigemma.controlled.plan_pins / write_ranges = the harness's own planner
    compare  --ranking R --map M --calibration CAL.moetrace --other NAME=TRACE [--other ...]
             --out results/rankings/routing_overlap_q3km.json
             top-N overlap per layer, rank correlation, capacity curves (C / random D / oracle)

numpy is required (host side).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pigemma import expert_map as xm
from pigemma import trace as tr

__all__ = ["load_events", "geometry", "build_ranking", "set_from_ranking", "hit_rates", "main"]

FORMAT = "pigemma-hot-ranking/1"
DEFAULT_SEED = 20261006
DEFAULT_RATE_BPS = 91.6e6          # Pi 5 microSD, sequential O_DIRECT read, measured 2026-10-05
DEFAULT_T_OP_S = 1.0e-3            # ASSUMED per discontiguous read (rationale: "score.labels" of the ranking file)
DEFAULT_READAROUND = 131072        # mmap read-around window = read_ahead_kb 128 (Pi mmcblk0, FUSE max_readahead)
GIB = 1024**3


def _sha256(p: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# trace -> access events
# ----------------------------------------------------------------------------------------------
def _requests_sidecar(trace: Path) -> dict[int, dict[str, Any]]:
    p = Path(str(trace) + ".requests.jsonl")
    if not p.is_file():
        raise FileNotFoundError(f"{p} (the probe's --requests-out sidecar) is required to cut the prefill into ubatches")
    out = {}
    for ln in p.read_text("utf-8").splitlines():
        if ln.strip():
            d = json.loads(ln)
            out[int(d["chunk_id"])] = d
    return out


def load_events(trace: str | os.PathLike[str]) -> dict[str, Any]:
    """Access events of a server-session trace.

    Returns dict with: header, L, E, K, n_ubatch, requests (list per chunk: id, n_keep,
    n_prefill, n_gen, ubatch sizes, stop), and per chunk boolean arrays
    ``dec[i]`` [n_gen, L, E] (generated-token evaluations) and ``ub[i]`` [n_ubatches, L, E]
    (prefill ubatch presence), plus ``pre_tok[i]`` [L, E] token-level prefill counts."""
    trace = Path(trace)
    with tr.TraceReader(trace, use_mmap=False) as r:
        ids = r.ids.astype(np.int64)            # [n, L, K]
        phase = r.phase
        hdr = dict(r.header)
        chunks = r.chunks
        L, K, E = r.L, r.K, r.E
    side = _requests_sidecar(trace)
    n_ubatch = int(hdr["n_ubatch"])
    n = ids.shape[0]
    member = np.zeros((n, L, E), dtype=bool)
    member[np.arange(n)[:, None, None], np.arange(L)[None, :, None], ids] = True
    reqs, dec, ub, pre_tok = [], [], [], []
    s = 0
    for i, c in enumerate(chunks):
        npre, ngen = int(c["n_prompt"]), int(c["n_gen"])
        sd = side.get(i)
        if sd is None:
            raise ValueError(f"{trace}: no requests.jsonl line for chunk {i}")
        if int(sd["n_prefill"]) != npre or int(sd["n_gen_evaluated"]) != ngen:
            raise ValueError(f"{trace}: chunk {i} header ({npre}, {ngen}) != sidecar ({sd['n_prefill']}, {sd['n_gen_evaluated']})")
        if not np.all(phase[s:s + npre] == 0) or not np.all(phase[s + npre:s + npre + ngen] == 1):
            raise ValueError(f"{trace}: chunk {i} phase layout is not prefill-then-generated")
        calls = [int(x) for x in sd["prefill_calls"]]
        if sum(calls) != npre:
            raise ValueError(f"{trace}: chunk {i} prefill_calls {calls} do not sum to {npre}")
        sizes: list[int] = []
        for cl in calls:                         # llama.cpp split_simple: consecutive n_ubatch pieces
            while cl > 0:
                t = min(n_ubatch, cl)
                sizes.append(t)
                cl -= t
        pres = np.zeros((len(sizes), L, E), dtype=bool)
        o = s
        for j, t in enumerate(sizes):
            pres[j] = member[o:o + t].any(axis=0)
            o += t
        ub.append(pres)
        pre_tok.append(member[s:s + npre].sum(axis=0).astype(np.int64))
        dec.append(member[s + npre:s + npre + ngen].copy())
        reqs.append({"chunk": i, "id": sd.get("id") or c.get("prompt_id") or str(i), "n_keep": int(sd["n_keep"]),
                     "n_prompt_tokens": int(sd["n_prompt_tokens"]), "n_prefill": npre, "n_gen": ngen,
                     "n_sampled": int(sd["n_sampled"]), "stop": sd["stop"], "ubatches": sizes})
        s += npre + ngen
    if s != n:
        raise ValueError(f"{trace}: chunks cover {s} of {n} records")
    return {"trace": str(trace), "header": hdr, "L": L, "E": E, "K": K, "n_ubatch": n_ubatch,
            "requests": reqs, "dec": dec, "ub": ub, "pre_tok": pre_tok}


def counts_of(ev: Mapping[str, Any], idx: Sequence[int] | None = None) -> dict[str, Any]:
    """Event counts over the requests ``idx`` (default all): arrays [L, E] and totals."""
    L, E = ev["L"], ev["E"]
    idx = range(len(ev["requests"])) if idx is None else idx
    dec = np.zeros((L, E), np.int64)
    pre_ub = np.zeros((L, E), np.int64)
    pre_tok = np.zeros((L, E), np.int64)
    n_dec = n_ub = n_pt = n_req = 0
    for i in idx:
        dec += ev["dec"][i].sum(axis=0)
        pre_ub += ev["ub"][i].sum(axis=0)
        pre_tok += ev["pre_tok"][i]
        n_dec += ev["dec"][i].shape[0]
        n_ub += ev["ub"][i].shape[0]
        n_pt += ev["requests"][i]["n_prefill"]
        n_req += 1
    return {"decode": dec, "prefill_ubatch": pre_ub, "prefill_token": pre_tok, "n_requests": n_req,
            "n_decode_events": n_dec, "n_prefill_ubatches": n_ub, "n_prefill_tokens": n_pt}


# ----------------------------------------------------------------------------------------------
# map -> page geometry
# ----------------------------------------------------------------------------------------------
def _merge_runs(pr: Sequence[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for a, b in sorted(pr):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def geometry(em: Any, page: int) -> dict[str, np.ndarray]:
    """Per expert [L, E]: unique pages, page-aligned bytes, nominal bytes and ``runs`` = the
    discontiguous page runs of its WEIGHT slices. The 4-byte per-expert scale entry sits on one
    page shared by all 128 experts of the layer (touched by every token), so it is counted in
    ``pages`` but not as a fetch of its own; ``runs_all`` includes it."""
    L, E = int(em["n_layer"]), int(em["n_expert"])
    pages = np.zeros((L, E), np.int64)
    runs = np.zeros((L, E), np.int64)
    runs_all = np.zeros((L, E), np.int64)
    nominal = np.zeros((L, E), np.int64)
    for x in em["experts"]:
        l, e = int(x["layer"]), int(x["expert"])
        pages[l, e] = int(x["pages"][str(page)]["pages"])
        nominal[l, e] = int(x["bytes"])

        def pr(rs: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
            return [(int(r["offset"]) // page, (int(r["offset"]) + int(r["n_bytes"]) + page - 1) // page) for r in rs]
        allr = _merge_runs(pr(x["ranges"]))
        runs_all[l, e] = len(allr)
        runs[l, e] = len(_merge_runs(pr([r for r in x["ranges"] if not str(r.get("role", "")).endswith("_scale")])))
        if sum(b - a for a, b in allr) != pages[l, e]:
            raise ValueError(f"expert ({l},{e}): page runs {allr} disagree with the map's page count {pages[l, e]}")
    return {"pages": pages, "bytes": pages * page, "runs": runs, "runs_all": runs_all, "nominal": nominal}


# ----------------------------------------------------------------------------------------------
# ranking
# ----------------------------------------------------------------------------------------------
def _order(score: np.ndarray, tie: np.ndarray) -> list[tuple[int, int]]:
    L, E = score.shape
    flat = [(-float(score[l, e]), -int(tie[l, e]), l, e) for l in range(L) for e in range(E)]
    flat.sort()
    return [(l, e) for _, _, l, e in flat]


def build_ranking(counts: Mapping[str, Any], geom: Mapping[str, np.ndarray], *, w_dec: float = 1.0, w_pre: float = 1.0,
                  rate_bps: float = DEFAULT_RATE_BPS, t_op_s: float = DEFAULT_T_OP_S,
                  readaround: int = DEFAULT_READAROUND) -> dict[str, Any]:
    nreq = max(1, int(counts["n_requests"]))
    freq = (w_dec * counts["decode"] + w_pre * counts["prefill_ubatch"]) / nreq
    c_run = rate_bps * t_op_s + readaround
    cost_factor = (geom["bytes"] + geom["runs"] * c_run) / geom["bytes"]
    score = freq * cost_factor
    return {"freq": freq, "cost_factor": cost_factor, "score": score, "c_run_bytes": c_run,
            "order": _order(score, counts["decode"])}


def set_from_ranking(order: Sequence[tuple[int, int]], k: int, L: int, E: int) -> np.ndarray:
    s = np.zeros((L, E), dtype=bool)
    for l, e in order[:k]:
        s[l, e] = True
    return s


def hit_rates(counts: Mapping[str, Any], pinned: np.ndarray, geom: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Static hit rates of a pinned set on a trace's access events (no cache model: an access to an
    unpinned expert is counted as a miss) and the worst-case miss volume per decode token."""
    d, u = counts["decode"], counts["prefill_ubatch"]
    nd = max(1, int(counts["n_decode_events"]))
    return {"decode_access_hit_rate": float(d[pinned].sum() / max(1, d.sum())),
            "prefill_event_hit_rate": float(u[pinned].sum() / max(1, u.sum())),
            "decode_unpinned_bytes_per_token_worst_case": float((d * geom["bytes"])[~pinned].sum() / nd),
            "decode_unpinned_accesses_per_token": float(d[~pinned].sum() / nd)}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x))
        # average ranks of ties
        xs = x[o]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(a.astype(float)), rank(b.astype(float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _topn(x: np.ndarray, n: int) -> set[int]:
    o = sorted(range(len(x)), key=lambda i: (-float(x[i]), i))
    return set(o[:n])


def _prefix_agreement(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]], ks: Sequence[int]) -> dict[str, float]:
    return {str(k): round(len(set(a[:k]) & set(b[:k])) / k, 4) for k in ks}


# ----------------------------------------------------------------------------------------------
# CLI: build
# ----------------------------------------------------------------------------------------------
def _workload_meta(set_name: str) -> dict[str, dict[str, Any]]:
    from pigemma import workloads as wl
    return {c["id"]: {"lang": c.get("lang"), "category": c.get("category"), "max_tokens": c.get("max_tokens"),
                      "context_reuse": c.get("context_reuse")} for c in wl.load_set(set_name)}


def _cmd_build(a: argparse.Namespace) -> int:
    ev = load_events(a.trace)
    hdr = ev["header"]
    ids = [r["id"] for r in ev["requests"]]
    if hdr.get("corpus_id") != "calibration" or not all(str(i).startswith("cal-") for i in ids):
        print(f"REFUSED: {a.trace} is not a calibration trace (corpus_id {hdr.get('corpus_id')!r}, first ids {ids[:3]}); "
              "only the calibration set may rank experts", file=sys.stderr)
        return 4
    from pigemma import workloads as wl
    cal_ids = [f"{c['id']}#{k}" for c in wl.load_set("calibration") for k in range(len(wl.user_turns(c)))]
    if ids != cal_ids:
        print(f"REFUSED: the trace's requests ({len(ids)}) are not exactly calibration.jsonl's requests ({len(cal_ids)})",
              file=sys.stderr)
        return 4
    doc_map = json.loads(Path(a.map).read_text("utf-8"))
    em = xm.load_expert_map(a.map)
    L, E = ev["L"], ev["E"]
    cnt = counts_of(ev)
    g4, g16 = geometry(em, 4096), geometry(em, 16384)
    gp = {4096: g4, 16384: g16}[a.page_size]
    kw = {"w_dec": a.w_dec, "w_pre": a.w_pre, "rate_bps": a.rate_bps, "t_op_s": a.t_op_ms / 1000.0, "readaround": a.readaround}
    prim = build_ranking(cnt, gp, **kw)
    variants = {
        "decode_only": build_ranking(cnt, gp, **{**kw, "w_pre": 0.0}),
        "prefill_events_only": build_ranking(cnt, gp, **{**kw, "w_dec": 0.0}),
        "no_per_fetch_term": build_ranking(cnt, gp, **{**kw, "t_op_s": 0.0, "readaround": 0}),
        "page_16384": build_ranking(cnt, g16, **kw) if a.page_size != 16384 else build_ranking(cnt, g4, **kw),
        "prefill_token_weighted": {"order": _order((cnt["decode"] + cnt["prefill_token"]) * prim["cost_factor"], cnt["decode"])},
    }
    ks = [240, 480, 960, 1440, 1920, 2880]
    agreement = {name: _prefix_agreement(prim["order"], v["order"], ks) for name, v in variants.items()}
    order = prim["order"]
    wl_dir = Path(wl.WORKLOADS_DIR)
    trace_p = Path(a.trace)
    out = {
        "format": FORMAT,
        "set": "calibration",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "trace": str(a.trace).replace("\\", "/"), "trace_sha256": _sha256(trace_p),
            "requests_sidecar_sha256": _sha256(str(trace_p) + ".requests.jsonl"),
            "workload_file": "data/workloads/calibration.jsonl", "workload_sha256": _sha256(wl_dir / "calibration.jsonl"),
            "n_requests": cnt["n_requests"], "n_decode_events": cnt["n_decode_events"],
            "n_prefill_ubatches": cnt["n_prefill_ubatches"], "n_prefill_tokens": cnt["n_prefill_tokens"],
            "probe_version": hdr.get("probe_version"), "llama_commit": hdr.get("llama_commit"),
            "server_stop": hdr.get("server_stop"), "n_ubatch": ev["n_ubatch"], "n_batch": hdr.get("n_batch"),
            "label": "MEASURED: routing of real greedy generations (phase 0 prefill + phase 1 generated tokens) of "
                     "exactly the requests the harness sends for calibration.jsonl; held-out and tuning data not used",
        },
        "model": {"gguf_basename": doc_map.get("gguf_basename"), "file_size": doc_map.get("file_size"),
                  "map": str(a.map).replace("\\", "/"), "map_sha256": _sha256(a.map), "n_layer": L, "n_expert": E,
                  "n_expert_used": ev["K"]},
        "score": {
            "formula": "score = F * (fetch_bytes + runs * c_run) / retained_bytes; "
                       "F = (w_dec * decode_events + w_pre * prefill_ubatch_events) / n_requests; "
                       "retained_bytes = fetch_bytes = unique pages of the expert's slices * page_size; "
                       "c_run = rate_bps * t_op_s + readaround_overshoot_bytes",
            "page_size": a.page_size, "w_dec": a.w_dec, "w_pre": a.w_pre,
            "rate_bps": a.rate_bps, "t_op_s": a.t_op_ms / 1000.0, "readaround_overshoot_bytes": a.readaround,
            "c_run_bytes": prim["c_run_bytes"],
            "labels": {"counts": "measured", "pages / runs": "exact from the map",
                       "w_dec = w_pre = 1": "assumed (every access event equally likely to be a miss)",
                       "t_op_s": "assumed (order of magnitude from the preliminary Pi balloon run; 0 in the desktop FUSE emulation)",
                       "readaround_overshoot_bytes": "assumed = one 128 KiB read-around window per discontiguous run"},
            "cost_factor_by_layer_group": {
                "layers_0_28_min_max": [round(float(prim["cost_factor"][:-1].min()), 5), round(float(prim["cost_factor"][:-1].max()), 5)],
                "layer_29_min_max": [round(float(prim["cost_factor"][-1].min()), 5), round(float(prim["cost_factor"][-1].max()), 5)]},
        },
        "ranking": [[l, e] for l, e in order],
        "scores": [round(float(prim["score"][l, e]), 6) for l, e in order],
        "counts": {"decode": cnt["decode"].tolist(), "prefill_ubatch": cnt["prefill_ubatch"].tolist(),
                   "prefill_token": cnt["prefill_token"].tolist()},
        "geometry": {"page_4096": {"pages": g4["pages"].tolist(), "runs": g4["runs"].tolist()},
                     "page_16384": {"pages": g16["pages"].tolist(), "runs": g16["runs"].tolist()},
                     "nominal_bytes_by_layer": [int(g4["nominal"][l, 0]) for l in range(L)]},
        "variants": {name: [[l, e] for l, e in v["order"]] for name, v in variants.items()},
        "variant_prefix_agreement_with_primary": {"prefix_sizes": ks, **agreement},
        "summary": {
            "experts_never_selected_in_decode": int((cnt["decode"] == 0).sum()),
            "experts_never_in_a_prefill_ubatch": int((cnt["prefill_ubatch"] == 0).sum()),
            "decode_accesses": int(cnt["decode"].sum()), "prefill_ubatch_events": int(cnt["prefill_ubatch"].sum()),
            "top_240_480_960_1920_share_of_decode_accesses": [
                round(float(sum(cnt["decode"][l, e] for l, e in order[:k]) / max(1, cnt["decode"].sum())), 4)
                for k in (240, 480, 960, 1920)],
            "experts_per_layer_in_top_960": np.bincount([l for l, _ in order[:960]], minlength=L).tolist(),
        },
        "random_control": {
            "seed": a.seed,
            "generator": "pigemma.controlled.plan_pins(em, page, 'D', target_total_pages=<C plan total_unique_pages>, seed=seed): "
                         "a seeded uniform shuffle of all (layer, expert) pairs, each added unless it would exceed C's total "
                         "unique pages (non-expert pages included), so D matches C in page-accounted bytes",
            "cli": f"python -m pigemma.hotrank control --ranking <this file> --map {str(a.map).replace(chr(92), '/')} "
                   f"--expert-pin-bytes <N> --seed {a.seed} --out-prefix results/controlled/plans/<name>   "
                   "(equivalent: python -m pigemma.controlled make-ranges --method C ... then --method D --match <C plan> --seed "
                   f"{a.seed})",
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": a.out, "n_ranked": len(order), **{k: out["source"][k] for k in
                      ("n_requests", "n_decode_events", "n_prefill_ubatches", "n_prefill_tokens")},
                      "summary": out["summary"], "agreement": out["variant_prefix_agreement_with_primary"],
                      "cost_factor": out["score"]["cost_factor_by_layer_group"]}, indent=1))
    return 0


# ----------------------------------------------------------------------------------------------
# CLI: control (C plan + size-matched random D plan through the harness's planner)
# ----------------------------------------------------------------------------------------------
def _cmd_control(a: argparse.Namespace) -> int:
    from pigemma import controlled as ct
    em = xm.load_expert_map(a.map)
    ranking, meta = ct.load_ranking(a.ranking, int(em["n_layer"]), int(em["n_expert"]))
    pc = ct.plan_pins(em, a.page_size, "C", ranking=ranking, expert_pin_bytes=a.expert_pin_bytes)
    pc["ranking"] = meta
    pd = ct.plan_pins(em, a.page_size, "D", target_total_pages=pc["total_unique_pages"], seed=a.seed)
    pd["matched_to"] = {"total_unique_pages": pc["total_unique_pages"], "method": "C"}
    res = {}
    for name, plan in (("C", pc), ("D", pd)):
        prefix = Path(f"{a.out_prefix}-{name}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        info = ct.write_ranges(em, plan, Path(str(prefix) + ".ranges.txt"))
        plan["ranges"] = info
        Path(str(prefix) + ".plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
        res[name] = {"plan": str(prefix) + ".plan.json", "ranges": info["ranges_file"], "n_experts": plan["n_experts"],
                     "total_unique_pages": plan["total_unique_pages"], "total_bytes": plan["total_bytes"],
                     "expert_pages": plan["expert_pages"], "check_ok": plan["check_ok"],
                     "union_matches_plan": info["union_matches_plan"]}
    res["D_minus_C_pages"] = pd["total_unique_pages"] - pc["total_unique_pages"]
    res["overlap_experts_C_and_D"] = len({tuple(x) for x in pc["experts"]} & {tuple(x) for x in pd["experts"]})
    res["seed"] = a.seed
    print(json.dumps(res, indent=1))
    return 0


# ----------------------------------------------------------------------------------------------
# CLI: compare (calibration vs other sets)
# ----------------------------------------------------------------------------------------------
def _capacity_curve(em: Any, page: int, ranking: Sequence[tuple[int, int]], budgets: Sequence[int],
                    seeds: Sequence[int], L: int, E: int) -> list[dict[str, Any]]:
    """For each expert-pin budget: the C set (ranked prefix, page exact) and D sets (seeded random,
    matched to C's total unique pages) from the harness's own planner."""
    from pigemma import controlled as ct
    out = []
    for b in budgets:
        pc = ct.plan_pins(em, page, "C", ranking=ranking, expert_pin_bytes=int(b))
        ds = []
        for sd in seeds:
            pd = ct.plan_pins(em, page, "D", target_total_pages=pc["total_unique_pages"], seed=sd)
            ds.append({"seed": sd, "experts": [tuple(x) for x in pd["experts"]], "expert_pages": pd["expert_pages"]})
        out.append({"expert_pin_bytes": int(b), "C_experts": [tuple(x) for x in pc["experts"]],
                    "C_expert_pages": pc["expert_pages"], "C_total_unique_pages": pc["total_unique_pages"], "D": ds})
    return out


def _mask(pairs: Sequence[tuple[int, int]], L: int, E: int) -> np.ndarray:
    m = np.zeros((L, E), dtype=bool)
    for l, e in pairs:
        m[l, e] = True
    return m


def _cmd_compare(a: argparse.Namespace) -> int:
    from pigemma import controlled as ct
    em = xm.load_expert_map(a.map)
    L, E = int(em["n_layer"]), int(em["n_expert"])
    page = a.page_size
    geom = geometry(em, page)
    ranking, meta = ct.load_ranking(a.ranking, L, E)
    cal = load_events(a.calibration)
    ccal = counts_of(cal)
    others: dict[str, dict[str, Any]] = {}
    for spec in a.other:
        name, path = spec.split("=", 1)
        others[name] = load_events(path)
    budgets = [int(x * GIB) for x in np.arange(a.budget_step_gib, 9.75 + 1e-9, a.budget_step_gib)]
    seeds = [a.seed] + [a.seed + i for i in range(1, a.n_seeds)]
    curve = _capacity_curve(em, page, ranking, budgets, seeds, L, E)
    res: dict[str, Any] = {
        "format": "pigemma-routing-overlap/1", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ranking": meta, "page_size": page, "map": str(a.map).replace("\\", "/"),
        "definitions": {
            "top-N overlap": "per layer, |top-N(calibration) & top-N(other)| / N by decode-event count (ties to the lower id); "
                             "chance level = N / 128",
            "decode_access_hit_rate": "share of the other set's decode expert accesses (8 per token per layer) that fall on a "
                                      "pinned expert - a STATIC hit rate without a page-cache model (an unpinned access counts as a miss)",
            "prefill_event_hit_rate": "the same over prefill ubatch events",
            "C": "strict prefix of the calibration ranking within the expert pin budget (page exact, harness planner)",
            "D": f"random experts matched to C's total unique pages, seeds {seeds} (first seed = the ranking file's control seed)",
            "oracle": "prefix of a ranking built from the EVALUATED set itself with the same score - not deployable, an upper "
                      "reference for a static set",
            "uniform": "expected hit rate of a random set = pinned experts / 3840",
        },
        "calibration": {"trace": str(a.calibration).replace("\\", "/"), "n_requests": ccal["n_requests"],
                        "n_decode_events": ccal["n_decode_events"], "n_prefill_ubatches": ccal["n_prefill_ubatches"]},
        "sets": {},
    }
    score_kw = json.loads(Path(a.ranking).read_text("utf-8")).get("score", {})
    kw = {"w_dec": score_kw.get("w_dec", 1.0), "w_pre": score_kw.get("w_pre", 1.0),
          "rate_bps": score_kw.get("rate_bps", DEFAULT_RATE_BPS), "t_op_s": score_kw.get("t_op_s", DEFAULT_T_OP_S),
          "readaround": score_kw.get("readaround_overshoot_bytes", DEFAULT_READAROUND)}
    for name, ev in {"calibration(self)": cal, **others}.items():
        cnt = counts_of(ev)
        own = build_ranking(cnt, geom, **kw)["order"]
        ent: dict[str, Any] = {"trace": ev["trace"].replace("\\", "/"), "n_requests": cnt["n_requests"],
                               "n_decode_events": cnt["n_decode_events"], "n_prefill_ubatches": cnt["n_prefill_ubatches"],
                               "n_prefill_tokens": cnt["n_prefill_tokens"]}
        # per-layer top-N overlap + rank correlation (decode events)
        per_layer = []
        for l in range(L):
            row = {"layer": l, "spearman_decode": round(_spearman(ccal["decode"][l], cnt["decode"][l]), 4),
                   "spearman_prefill_ubatch": round(_spearman(ccal["prefill_ubatch"][l], cnt["prefill_ubatch"][l]), 4)}
            for n in (8, 16, 32, 64):
                row[f"top{n}_overlap_decode"] = round(len(_topn(ccal["decode"][l], n) & _topn(cnt["decode"][l], n)) / n, 4)
                # how much of the other set's decode accesses in this layer fall on calibration's top-N / its own top-N
                tot = max(1, int(cnt["decode"][l].sum()))
                row[f"top{n}_cal_share_of_decode"] = round(float(cnt["decode"][l][sorted(_topn(ccal["decode"][l], n))].sum() / tot), 4)
                row[f"top{n}_own_share_of_decode"] = round(float(cnt["decode"][l][sorted(_topn(cnt["decode"][l], n))].sum() / tot), 4)
            per_layer.append(row)
        ent["per_layer"] = per_layer
        ent["mean_over_layers"] = {k: round(float(np.nanmean([r[k] for r in per_layer])), 4)
                                   for k in per_layer[0] if k != "layer"}
        ent["chance_overlap"] = {f"top{n}": round(n / E, 4) for n in (8, 16, 32, 64)}
        # global prefix agreement of the ranking with the set's own ranking
        ent["global_prefix_agreement_cal_vs_own"] = _prefix_agreement(ranking, own, [240, 480, 960, 1440, 1920, 2880])
        # capacity curve
        rows = []
        for c in curve:
            mc = _mask(c["C_experts"], L, E)
            k = len(c["C_experts"])
            mo = _mask(own[:k], L, E)
            hc, ho = hit_rates(cnt, mc, geom), hit_rates(cnt, mo, geom)
            hds = [hit_rates(cnt, _mask(d["experts"], L, E), geom) for d in c["D"]]
            rows.append({
                "expert_pin_GiB": round(c["expert_pin_bytes"] / GIB, 3), "expert_pin_bytes": c["expert_pin_bytes"],
                "C_n_experts": k, "C_expert_pages": c["C_expert_pages"], "uniform": round(k / (L * E), 4),
                "C_decode_hit": round(hc["decode_access_hit_rate"], 4), "C_prefill_hit": round(hc["prefill_event_hit_rate"], 4),
                "D_decode_hit_seed0": round(hds[0]["decode_access_hit_rate"], 4),
                "D_decode_hit_mean": round(float(np.mean([h["decode_access_hit_rate"] for h in hds])), 4),
                "D_decode_hit_min": round(float(np.min([h["decode_access_hit_rate"] for h in hds])), 4),
                "D_decode_hit_max": round(float(np.max([h["decode_access_hit_rate"] for h in hds])), 4),
                "D_prefill_hit_mean": round(float(np.mean([h["prefill_event_hit_rate"] for h in hds])), 4),
                "D_n_experts_seed0": len(c["D"][0]["experts"]),
                "oracle_decode_hit": round(ho["decode_access_hit_rate"], 4),
                "oracle_prefill_hit": round(ho["prefill_event_hit_rate"], 4),
                "C_decode_unpinned_MB_per_token_worst_case": round(hc["decode_unpinned_bytes_per_token_worst_case"] / 1e6, 2),
                "D_decode_unpinned_MB_per_token_worst_case_seed0": round(hds[0]["decode_unpinned_bytes_per_token_worst_case"] / 1e6, 2),
                "oracle_decode_unpinned_MB_per_token_worst_case": round(ho["decode_unpinned_bytes_per_token_worst_case"] / 1e6, 2),
            })
        ent["capacity_curve"] = rows
        ent["no_pinning_decode_MB_per_token_worst_case"] = round(
            hit_rates(cnt, np.zeros((L, E), bool), geom)["decode_unpinned_bytes_per_token_worst_case"] / 1e6, 2)
        # distinct experts actually used
        ent["distinct_experts_used_decode"] = int((cnt["decode"] > 0).sum())
        ent["distinct_experts_used_any"] = int(((cnt["decode"] + cnt["prefill_ubatch"]) > 0).sum())
        # per language / category (requests grouped by the workload metadata)
        set_name = {"cal": "calibration", "tun": "tuning", "ho": "heldout"}.get(str(ev["requests"][0]["id"]).split("-")[0])
        if set_name:
            wm = _workload_meta(set_name)
            groups: dict[str, list[int]] = {}
            for i, r in enumerate(ev["requests"]):
                m = wm.get(str(r["id"]).split("#")[0], {})
                groups.setdefault(f"lang={m.get('lang')}", []).append(i)
                groups.setdefault(f"category={m.get('category')}", []).append(i)
            gsel = [c for c in curve if round(c["expert_pin_bytes"] / GIB, 3) in a.group_budgets_gib]
            gout = {}
            for gname, idx in sorted(groups.items()):
                gc = counts_of(ev, idx)
                gown = build_ranking(gc, geom, **kw)["order"]
                gout[gname] = {"n_requests": len(idx), "n_decode_events": gc["n_decode_events"], "by_budget": [
                    {"expert_pin_GiB": round(c["expert_pin_bytes"] / GIB, 3),
                     "C_decode_hit": round(hit_rates(gc, _mask(c["C_experts"], L, E), geom)["decode_access_hit_rate"], 4),
                     "D_decode_hit_seed0": round(hit_rates(gc, _mask(c["D"][0]["experts"], L, E), geom)["decode_access_hit_rate"], 4),
                     "oracle_group_decode_hit": round(hit_rates(gc, _mask(gown[:len(c["C_experts"])], L, E), geom)["decode_access_hit_rate"], 4)}
                    for c in gsel]}
            ent["groups"] = gout
            # per request at the selected budgets (for per-workload reporting)
            preq = []
            for i, r in enumerate(ev["requests"]):
                rc = counts_of(ev, [i])
                if rc["n_decode_events"] == 0:
                    continue
                preq.append({"id": r["id"], "n_decode_events": rc["n_decode_events"], "by_budget": [
                    {"expert_pin_GiB": round(c["expert_pin_bytes"] / GIB, 3),
                     "C_decode_hit": round(hit_rates(rc, _mask(c["C_experts"], L, E), geom)["decode_access_hit_rate"], 4),
                     "D_decode_hit_seed0": round(hit_rates(rc, _mask(c["D"][0]["experts"], L, E), geom)["decode_access_hit_rate"], 4)}
                    for c in gsel]})
            ent["per_request"] = preq
        res["sets"][name] = ent
    # split-half reliability of the calibration counts (conversations alternate between halves,
    # so follow-up turns stay together): how much of the cal-vs-other gap is sampling noise
    conv_of = [str(r["id"]).split("#")[0] for r in cal["requests"]]
    conv_ids = list(dict.fromkeys(conv_of))
    half = {c: i % 2 for i, c in enumerate(conv_ids)}
    ia = [i for i, c in enumerate(conv_of) if half[c] == 0]
    ib = [i for i, c in enumerate(conv_of) if half[c] == 1]
    ca, cb = counts_of(cal, ia), counts_of(cal, ib)
    sh = []
    for l in range(L):
        row = {"layer": l, "spearman_decode": round(_spearman(ca["decode"][l], cb["decode"][l]), 4)}
        for n in (8, 16, 32, 64):
            row[f"top{n}_overlap_decode"] = round(len(_topn(ca["decode"][l], n) & _topn(cb["decode"][l], n)) / n, 4)
        sh.append(row)
    ra = build_ranking(ca, geom, **kw)["order"]
    rb = build_ranking(cb, geom, **kw)["order"]
    res["calibration_split_half"] = {
        "how": "calibration conversations alternate between two halves (0, 2, 4, ... / 1, 3, 5, ...)",
        "half_a": {"n_requests": ca["n_requests"], "n_decode_events": ca["n_decode_events"]},
        "half_b": {"n_requests": cb["n_requests"], "n_decode_events": cb["n_decode_events"]},
        "mean_over_layers": {k: round(float(np.nanmean([r[k] for r in sh])), 4) for k in sh[0] if k != "layer"},
        "global_prefix_agreement": _prefix_agreement(ra, rb, [240, 480, 960, 1440, 1920, 2880]),
        "per_layer": sh}
    # do decode and prefill prefer the same experts? (calibration; per layer top-N by decode events
    # vs by prefill TOKEN counts, which unlike ubatch presence is not saturated by long prompts)
    dp = []
    for l in range(L):
        row = {"layer": l, "spearman_decode_vs_prefill_token": round(_spearman(ccal["decode"][l], ccal["prefill_token"][l]), 4)}
        for n in (8, 16, 32, 64):
            row[f"top{n}_overlap"] = round(len(_topn(ccal["decode"][l], n) & _topn(ccal["prefill_token"][l], n)) / n, 4)
        dp.append(row)
    res["calibration_decode_vs_prefill_token_preference"] = {
        "mean_over_layers": {k: round(float(np.nanmean([r[k] for r in dp])), 4) for k in dp[0] if k != "layer"},
        "per_layer": dp}
    # prefill ubatch saturation: how many of the 128 experts of a layer one prefill ubatch touches
    sat = {}
    for name, ev in {"calibration": cal, **others}.items():
        rows = []
        for i, r in enumerate(ev["requests"]):
            for j, t in enumerate(r["ubatches"]):
                rows.append((t, float(ev["ub"][i][j].sum(axis=1).mean())))
        buckets = {}
        for lo, hi in ((1, 4), (5, 31), (32, 63), (64, 127), (128, 255), (256, 512)):
            xs = [m for t, m in rows if lo <= t <= hi]
            if xs:
                buckets[f"{lo}-{hi} tokens"] = {"n_ubatches": len(xs), "mean_experts_per_layer": round(float(np.mean(xs)), 1),
                                                 "min": round(float(np.min(xs)), 1), "max": round(float(np.max(xs)), 1)}
        sat[name] = buckets
    res["prefill_ubatch_experts_touched_per_layer"] = sat
    # external rankings (e.g. HotPin's own calibration ranking from X1b), prefix agreement with ours
    ext = {}
    for spec in a.external:
        name, path = spec.split("=", 1)
        try:
            er, _ = ct.load_ranking(path, L, E)
        except Exception as e:  # noqa: BLE001
            ext[name] = {"error": repr(e)[:200]}
            continue
        ours = json.loads(Path(a.ranking).read_text("utf-8"))
        ent = {"path": path.replace("\\", "/"), "n": len(er),
               "prefix_agreement_with_primary": _prefix_agreement(ranking, er, [240, 480, 960, 1440, 1920]),
               "prefix_agreement_with_decode_only": _prefix_agreement(
                   [tuple(x) for x in ours.get("variants", {}).get("decode_only", [])] or ranking, er,
                   [240, 480, 960, 1440, 1920])}
        for oname, oev in others.items():
            oc = counts_of(oev)
            ent[f"decode_hit_on_{oname}_at_prefix"] = {
                str(k): {"external": round(hit_rates(oc, _mask(er[:k], L, E), geom)["decode_access_hit_rate"], 4),
                         "ours": round(hit_rates(oc, _mask(ranking[:k], L, E), geom)["decode_access_hit_rate"], 4)}
                for k in (480, 960, 1440, 1920)}
        ext[name] = ent
    res["external_rankings"] = ext
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    brief = {n: {"mean_over_layers": e["mean_over_layers"],
                 "curve_sample": [r for r in e["capacity_curve"] if r["expert_pin_GiB"] in a.group_budgets_gib]}
             for n, e in res["sets"].items()}
    brief["calibration_split_half"] = {k: res["calibration_split_half"][k] for k in ("mean_over_layers", "global_prefix_agreement")}
    brief["decode_vs_prefill_token"] = res["calibration_decode_vs_prefill_token_preference"]["mean_over_layers"]
    brief["prefill_ubatch_saturation"] = sat
    brief["external"] = ext
    print(json.dumps(brief, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pigemma.hotrank", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build", help="rank experts from the CALIBRATION trace")
    b.add_argument("--trace", required=True)
    b.add_argument("--map", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--page-size", type=int, default=4096, choices=(4096, 16384))
    b.add_argument("--w-dec", type=float, default=1.0)
    b.add_argument("--w-pre", type=float, default=1.0)
    b.add_argument("--rate-bps", type=float, default=DEFAULT_RATE_BPS)
    b.add_argument("--t-op-ms", type=float, default=DEFAULT_T_OP_S * 1000.0)
    b.add_argument("--readaround", type=int, default=DEFAULT_READAROUND)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed of the random control recorded in the file")
    c = sub.add_parser("control", help="C plan + size-matched random D plan (page-accounted)")
    c.add_argument("--ranking", required=True)
    c.add_argument("--map", required=True)
    c.add_argument("--expert-pin-bytes", type=int, required=True)
    c.add_argument("--out-prefix", required=True)
    c.add_argument("--seed", type=int, default=DEFAULT_SEED)
    c.add_argument("--page-size", type=int, default=4096, choices=(4096, 16384))
    m = sub.add_parser("compare", help="calibration vs other sets: top-N overlap, capacity curves")
    m.add_argument("--ranking", required=True)
    m.add_argument("--map", required=True)
    m.add_argument("--calibration", required=True)
    m.add_argument("--other", action="append", default=[], help="NAME=TRACE (repeatable)")
    m.add_argument("--external", action="append", default=[],
                   help="NAME=RANKING.json: another calibration ranking to compare with (repeatable)")
    m.add_argument("--out", required=True)
    m.add_argument("--page-size", type=int, default=4096, choices=(4096, 16384))
    m.add_argument("--seed", type=int, default=DEFAULT_SEED)
    m.add_argument("--n-seeds", type=int, default=10)
    m.add_argument("--budget-step-gib", type=float, default=0.25)
    m.add_argument("--group-budgets-gib", type=float, nargs="*", default=[1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0])
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if a.cmd == "build":
        return _cmd_build(a)
    if a.cmd == "control":
        return _cmd_control(a)
    if a.cmd == "compare":
        return _cmd_compare(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
