#!/usr/bin/env python3
"""X1b: inspect a HotPin routing dump (format written by its --dump-routing: int32 layer, int32 k, k ids).

What it reports (stdlib only):
  * record/pass structure: the tool writes, per graph evaluation and per layer, one record per token
    of that evaluation, so consecutive same-layer records give the batch width n (n > 1 = prefill
    (u)batch, n == 1 = one generated token);
  * the multi-token records check: at a6647b1 `ffn_moe_topk` is a strided VIEW of the full argsort
    (ggml_argsort_top_k: ne = {k, n_tokens}, nb[1] = n_expert * 4). The dump callback copies
    k*n contiguous int32 from t->data and reads record j at ids[j*k], i.e. it walks ALONG the first
    token's 128-entry argsort row instead of jumping to the next token. Signature: inside one
    prefill evaluation the first 16 records of a layer are 128 DISTINCT expert ids (a permutation
    of 0..127); real top-8 lists of 16 consecutive tokens repeat experts. We count how often the
    signature holds;
  * counts exactly as HotPin's make_hot_file.py sees them (all records) vs counts from the
    single-token evaluations only (generated tokens; unaffected), and how the two rankings differ.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import sys
from pathlib import Path


def load_records(path: Path) -> tuple[list[int], list[tuple[int, ...]], dict]:
    raw = Path(path).read_bytes()
    ints = array.array("i")
    ints.frombytes(raw[: len(raw) // 4 * 4])
    if sys.byteorder != "little":
        ints.byteswap()
    layers: list[int] = []
    ids: list[tuple[int, ...]] = []
    i, n = 0, len(ints)
    ks: dict[int, int] = {}
    while i + 2 <= n:
        layer, k = ints[i], ints[i + 1]
        if not (0 <= layer < 1000 and 1 <= k <= 64 and i + 2 + k <= n):
            raise ValueError(f"bad record at int {i}: layer={layer} k={k}")
        layers.append(layer)
        ids.append(tuple(ints[i + 2:i + 2 + k]))
        ks[k] = ks.get(k, 0) + 1
        i += 2 + k
    info = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "records": len(layers),
            "k_histogram": ks, "trailing_ints": n - i}
    return layers, ids, info


def passes_of(layers: list[int]) -> list[tuple[int, int, int]]:
    """[(first_record_index, layer, run_length)] for consecutive same-layer records."""
    runs = []
    i, n = 0, len(layers)
    while i < n:
        j = i
        while j < n and layers[j] == layers[i]:
            j += 1
        runs.append((i, layers[i], j - i))
        i = j
    return runs


def rank(counts: dict[tuple[int, int], int], first_seen: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    # make_hot_file.py: sorted(counts.items(), key=-count), stable -> ties in first-seen order
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True)
    ap.add_argument("--meta", help="routing_<tag>_meta.json of hotpin_routing_dump.py (token totals cross-check)")
    ap.add_argument("--n-layer", type=int, default=30)
    ap.add_argument("--n-expert", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-counts", help="write {all,single,multi} count matrices to this JSON")
    a = ap.parse_args()
    layers, ids, info = load_records(Path(a.routing))
    runs = passes_of(layers)
    L, E = a.n_layer, a.n_expert
    # group runs into graph evaluations: layers 0..L-1 with one common width
    evals: list[tuple[int, int]] = []   # (index of first run, width)
    bad = 0
    i = 0
    while i < len(runs):
        chunk = runs[i:i + L]
        if len(chunk) == L and [r[1] for r in chunk] == list(range(L)) and len({r[2] for r in chunk}) == 1:
            evals.append((i, chunk[0][2]))
            i += L
        else:
            bad += 1
            i += 1
    widths: dict[int, int] = {}
    for _, w in evals:
        widths[w] = widths.get(w, 0) + 1
    all_c: dict[tuple[int, int], int] = {}
    single_c: dict[tuple[int, int], int] = {}
    multi_c: dict[tuple[int, int], int] = {}
    first_all: dict[tuple[int, int], int] = {}
    first_single: dict[tuple[int, int], int] = {}
    seq = 0
    sig_checked = sig_hit = 0
    perm_checked = perm_hit = 0
    dup_within_record = 0
    tok0_distinct_from_rest = 0
    multi_records = single_records = 0
    for ri, w in evals:
        for (start, layer, n) in runs[ri:ri + L]:
            for j in range(n):
                rec = ids[start + j]
                if len(set(rec)) != len(rec):
                    dup_within_record += 1
                for e in rec:
                    key = (layer, e)
                    all_c[key] = all_c.get(key, 0) + 1
                    if key not in first_all:
                        first_all[key] = seq
                    seq += 1
                    if n == 1:
                        single_c[key] = single_c.get(key, 0) + 1
                        first_single.setdefault(key, seq)
                    else:
                        multi_c[key] = multi_c.get(key, 0) + 1
            if n == 1:
                single_records += 1
            else:
                multi_records += n
                m = min(n, E // len(ids[start]))
                flat = [e for j in range(m) for e in ids[start + j]]
                sig_checked += 1
                if len(set(flat)) == len(flat):
                    sig_hit += 1
                if n >= E // len(ids[start]):
                    perm_checked += 1
                    if sorted(flat) == list(range(E)):
                        perm_hit += 1
    order_all = rank(all_c, first_all)
    order_single = rank(single_c, first_single)
    res: dict = {
        "schema": "pigemma-x1b-hotpin-routing-analysis v1", "routing": str(a.routing), "file": info,
        "runs": len(runs), "graph_evaluations": len(evals), "runs_not_in_a_clean_evaluation": bad,
        "evaluation_width_histogram": {str(k): v for k, v in sorted(widths.items())},
        "records": {"single_token_evaluations": single_records, "multi_token_records": multi_records,
                    "single_token_tokens": single_records // L if L else None,
                    "multi_token_tokens": multi_records // L if L else None},
        "records_with_a_repeated_expert_id": dup_within_record,
        "multi_token_check": {
            "what": "per (prefill evaluation, layer): are the first min(n,16) records pairwise disjoint (8*min(n,16) "
                    "distinct ids)? and for n >= 16: are the first 16 records a permutation of 0..127?",
            "layer_evaluations_checked": sig_checked, "first_records_pairwise_disjoint": sig_hit,
            "n_ge_16_checked": perm_checked, "first_16_records_are_a_permutation_of_all_experts": perm_hit},
        "counts": {
            "all_records": {"unique_experts": len(all_c), "total_hits": sum(all_c.values()),
                            "max": max(all_c.values()) if all_c else 0, "min": min(all_c.values()) if all_c else 0},
            "single_token_only": {"unique_experts": len(single_c), "total_hits": sum(single_c.values()),
                                  "max": max(single_c.values()) if single_c else 0},
            "multi_token_only": {"unique_experts": len(multi_c), "total_hits": sum(multi_c.values()),
                                 "max": max(multi_c.values()) if multi_c else 0,
                                 "min": min(multi_c.values()) if multi_c else 0},
        },
        "ranking_all_vs_single_token_only": {},
    }
    for n in (250, 500, 1000, 1500, 2000, 3000):
        sa, ss = set(order_all[:n]), set(order_single[:n])
        res["ranking_all_vs_single_token_only"][str(n)] = {
            "top_n_overlap": len(sa & ss), "available_single": len(order_single)}
    per_layer_all = [0] * L
    for (l, _), c in all_c.items():
        per_layer_all[l] += c
    res["counts"]["hits_per_layer_all_records"] = per_layer_all
    if a.meta:
        meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))
        reqs = meta.get("requests", [])
        exp_single = sum((r.get("gen_tokens") or 0) - (1 if (r.get("gen_tokens") or 0) >= r["max_tokens"] else 0)
                         for r in reqs)
        res["meta_cross_check"] = {
            "requests": len(reqs), "prompt_tokens_sum": sum(r.get("prompt_tokens") or 0 for r in reqs),
            "gen_tokens_sum": sum(r.get("gen_tokens") or 0 for r in reqs),
            "expected_single_token_evaluations": exp_single,
            "expected_multi_token_tokens": sum(r.get("prompt_tokens") or 0 for r in reqs),
            "single_ok": exp_single == single_records // L,
            "multi_ok": sum(r.get("prompt_tokens") or 0 for r in reqs) == multi_records // L,
            "note": "a 1-token (u)batch would be counted as single; prompts here are >= 27 tokens and the "
                    "ubatch is 512, so a 1-token remainder needs a prompt of 513/1025 tokens"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    if a.save_counts:
        def mat(c: dict) -> list[list[int]]:
            return [[c.get((l, e), 0) for e in range(E)] for l in range(L)]
        Path(a.save_counts).write_text(json.dumps({
            "note": "HotPin routing dump counts; 'all' = what make_hot_file.py ranks; 'single' = single-token "
                    "evaluations (generated tokens); 'multi' = multi-token evaluations (prefill; see the "
                    "multi_token_check in the analysis)",
            "source_set": "calibration", "routing_sha256": info["sha256"], "n_layer": L, "n_expert": E,
            "all": mat(all_c), "single": mat(single_c), "multi": mat(multi_c),
            "order_all": [list(k) for k in order_all], "order_single": [list(k) for k in order_single]}) + "\n",
            encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("graph_evaluations", "evaluation_width_histogram", "records",
                                           "multi_token_check", "counts", "ranking_all_vs_single_token_only")}
                     | ({"meta_cross_check": res["meta_cross_check"]} if a.meta else {}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
