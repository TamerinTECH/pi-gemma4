#!/usr/bin/env python3
"""X1c: compare the greedy outputs of the engines in a pc/x1c/ik_check.py run (stdlib only).

    compare_outputs.py <check.json> <out.json> [--ref a7b94df-reference] [--pairs a:b,c:d]

Reports, per prompt:
  * within-engine determinism (run a == run b == streaming run),
  * for every pair: prompt token ids equal, generated tokens equal, first divergence, max |delta| of the
    chosen-token logprob over the common prefix, top-5 lists equal, top-5 max |delta|.

Three views of "the generated tokens", because the two server families report different things:
  ids       token ids returned by the server (llama.cpp: `tokens` / completion_probabilities[].id).
            ik_llama.cpp's /completion returns NO ids; a list with missing ids is treated as ABSENT and
            never compared (an earlier version of this script compared lists of None and reported
            "identical" - that was a false positive and is fixed here).
  pieces    the per-token strings (llama.cpp: completion_probabilities[].token; ik_llama.cpp:
            completion_probabilities[].content). Available for both families -> used for the
            cross-engine first-divergence index.
  retok     ids from POST /tokenize of the generated text (same procedure for every engine; added by
            ik_check.py v2). Equal text gives equal retok ids by construction, so this only converts
            the text comparison into token units.
Top-5 values: llama.cpp reports logprobs, ik_llama.cpp reports probabilities; a pair is compared on
the value kind both sides have (same-engine pairs: exact float equality is the bit-identity evidence).

Cross-ENGINE differences (ik_llama.cpp vs llama.cpp) are numerical differences between two
implementations and are documented, not treated as a cache failure; SAME-engine pairs
(ik-plain vs ik-prefetch-experts) must be identical because the option must not change the computation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def ids_of(run: dict) -> list:
    ids = run.get("token_ids") or [p.get("id") for p in run.get("probs") or []]
    return list(ids) if ids and all(i is not None for i in ids) else []


def pieces_of(run: dict) -> list:
    ps = [p.get("token") for p in run.get("probs") or []]
    return ps if ps and all(p is not None for p in ps) else []


def retok_of(run: dict) -> list:
    return list(run.get("retokenized_ids") or [])


def _val(x: dict, kind: str):
    return x.get(kind)


def _first_div(a: list, b: list):
    n = min(len(a), len(b))
    div = next((i for i in range(n) if a[i] != b[i]), None)
    if div is None and len(a) != len(b):
        div = n
    return div, (n if div is None else div)


def pair(a: dict, b: dict) -> dict:
    out: dict = {"text_identical": a.get("text") == b.get("text") and a.get("text") is not None}
    ia, ib = ids_of(a), ids_of(b)
    pa_, pb_ = pieces_of(a), pieces_of(b)
    ra, rb = retok_of(a), retok_of(b)
    out["server_ids_available"] = [bool(ia), bool(ib)]
    if ia and ib:
        div, common = _first_div(ia, ib)
        out.update({"ids_identical": div is None, "first_divergence_index_ids": div, "common_prefix_tokens_ids": common})
    else:
        out.update({"ids_identical": None, "first_divergence_index_ids": None, "common_prefix_tokens_ids": None})
    if pa_ and pb_:
        div, common = _first_div(pa_, pb_)
        out.update({"pieces_identical": div is None, "first_divergence_index_pieces": div,
                    "common_prefix_tokens_pieces": common, "n_a": len(pa_), "n_b": len(pb_),
                    "piece_at_divergence": [pa_[div] if div is not None and div < len(pa_) else None,
                                            pb_[div] if div is not None and div < len(pb_) else None]})
    else:
        out.update({"pieces_identical": None, "first_divergence_index_pieces": None, "common_prefix_tokens_pieces": None})
    if ra and rb:
        div, common = _first_div(ra, rb)
        out.update({"retokenized_ids_identical": div is None, "first_divergence_index_retok": div,
                    "common_prefix_tokens_retok": common})
    else:
        out.update({"retokenized_ids_identical": None})
    # probabilities on the common prefix (pieces view when available, else ids view)
    common = out.get("common_prefix_tokens_pieces")
    if common is None:
        common = out.get("common_prefix_tokens_ids") or 0
    pa, pb = a.get("probs") or [], b.get("probs") or []
    n = min(common, len(pa), len(pb))
    for kind in ("logprob", "prob"):
        d_chosen = []
        for i in range(n):
            x, y = _val(pa[i], kind), _val(pb[i], kind)
            if x is None:  # ik: the chosen token's value is the matching top entry
                x = next((t.get(kind) for t in pa[i].get("top") or [] if t.get("token") == pa[i].get("token") and t.get(kind) is not None), None)
            if y is None:
                y = next((t.get(kind) for t in pb[i].get("top") or [] if t.get("token") == pb[i].get("token") and t.get(kind) is not None), None)
            if x is not None and y is not None:
                d_chosen.append(abs(x - y))
        top_same = 0
        top_cmp = 0
        d_top = []
        exact = 0
        for i in range(n):
            ta, tb = pa[i].get("top") or [], pb[i].get("top") or []
            if not ta or not tb or any(t.get(kind) is None for t in ta + tb):
                continue
            top_cmp += 1
            ka = [t.get("id") if t.get("id") is not None else t.get("token") for t in ta]
            kb = [t.get("id") if t.get("id") is not None else t.get("token") for t in tb]
            if ka == kb:
                top_same += 1
                dd = [abs(x[kind] - y[kind]) for x, y in zip(ta, tb)]
                d_top += dd
                exact += int(all(v == 0.0 for v in dd))
        out[f"chosen_{kind}_max_abs_diff_on_common_prefix"] = max(d_chosen) if d_chosen else None
        out[f"chosen_{kind}_positions_compared"] = len(d_chosen)
        out[f"top5_{kind}_positions_compared"] = top_cmp
        out[f"top5_{kind}_lists_same_candidates"] = f"{top_same}/{top_cmp}" if top_cmp else None
        out[f"top5_{kind}_max_abs_diff"] = max(d_top) if d_top else None
        out[f"top5_{kind}_positions_exactly_equal"] = f"{exact}/{top_cmp}" if top_cmp else None
    # a cross-family pair has logprobs on one side and probs on the other: compare exp(logprob) with prob
    if out["chosen_logprob_positions_compared"] == 0 and out["chosen_prob_positions_compared"] == 0:
        d = []
        for i in range(n):
            def p_of(e):
                if e.get("logprob") is not None:
                    return math.exp(e["logprob"])
                return next((t.get("prob") for t in e.get("top") or [] if t.get("token") == e.get("token") and t.get("prob") is not None), None)
            x, y = p_of(pa[i]), p_of(pb[i])
            if x is not None and y is not None:
                d.append(abs(x - y))
        out["chosen_probability_max_abs_diff_cross_family"] = max(d) if d else None
        out["chosen_probability_positions_compared_cross_family"] = len(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("check")
    ap.add_argument("out")
    ap.add_argument("--ref", default="a7b94df-reference")
    ap.add_argument("--pairs", default="")
    a = ap.parse_args()
    chk = json.load(open(a.check, encoding="utf-8"))
    engs = {e["engine_id"]: e for e in chk["engines"]}
    res: dict = {"schema": "pigemma-x1c-compare v2", "check": a.check, "kind": chk.get("kind"),
                 "model_sha256_measured": chk.get("model_sha256_measured"),
                 "views": "ids = token ids from the server (absent for ik_llama.cpp); pieces = per-token strings; "
                          "retok = ids from /tokenize of the generated text; see the module docstring",
                 "engines": {}, "pairs": {}}
    for eid, e in engs.items():
        rows = {}
        for pid, pr in (e.get("requests") or {}).items():
            ra, rb, st = pr.get("a", {}), pr.get("b", {}), pr.get("stream", {})
            ia, ib = ids_of(ra), ids_of(rb)
            pa, pb = ra.get("probs") or [], rb.get("probs") or []
            same_probs = bool(pa) and len(pa) == len(pb) and all(
                x.get("logprob") == y.get("logprob") and x.get("prob") == y.get("prob") and x.get("top") == y.get("top")
                and x.get("token") == y.get("token") for x, y in zip(pa, pb))
            has_values = bool(pa) and any((x.get("logprob") is not None) or any(t.get("prob") is not None or t.get("logprob") is not None
                                                                              for t in x.get("top") or []) for x in pa)
            rows[pid] = {"n_generated": ra.get("tokens_predicted"), "prompt_tokens": len(pr.get("prompt_token_ids") or []),
                         "server_ids_available": bool(ia),
                         "a_equals_b_ids": (ia == ib) if (ia and ib) else None,
                         "a_equals_b_pieces": (pieces_of(ra) == pieces_of(rb)) if (pieces_of(ra) and pieces_of(rb)) else None,
                         "a_equals_b_text": ra.get("text") == rb.get("text") if ra and rb else None,
                         "a_equals_b_values_and_top5_exact": same_probs if has_values else None,
                         "stream_text_equals_a_text": st.get("text") == ra.get("text") if st else None,
                         "stream_ids_equal_a_ids": (st.get("token_ids") == ia) if (st.get("token_ids") and ia) else None,
                         "stream_timestamps_per": st.get("timestamps_per"),
                         "text": ra.get("text")}
        res["engines"][eid] = {"ok": e.get("ok"), "error": e.get("error"), "version": e.get("version_output"),
                               "bin_sha256": e.get("bin_sha256"), "requests": rows}
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",") if ":" in p]
    pairs += [(a.ref, eid) for eid in engs if eid != a.ref and (a.ref, eid) not in pairs and a.ref in engs]
    for x, y in pairs:
        if x not in engs or y not in engs or not engs[x].get("ok") or not engs[y].get("ok"):
            res["pairs"][f"{x} vs {y}"] = {"error": "engine missing or failed"}
            continue
        rows = {}
        for pid in engs[x]["requests"]:
            px, py = engs[x]["requests"][pid], engs[y]["requests"].get(pid, {})
            if "a" not in px or "a" not in py:
                continue
            r = pair(px["a"], py["a"])
            r["prompt_token_ids_identical"] = px.get("prompt_token_ids") == py.get("prompt_token_ids") and bool(px.get("prompt_token_ids"))
            rows[pid] = r
        res["pairs"][f"{x} vs {y}"] = rows
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for eid, e in res["engines"].items():
        print(eid, "ok" if e["ok"] else f"FAILED {e['error']}",
              {p: {k: r[k] for k in ("server_ids_available", "a_equals_b_ids", "a_equals_b_pieces", "a_equals_b_text",
                                     "a_equals_b_values_and_top5_exact", "stream_text_equals_a_text")}
               for p, r in e["requests"].items()})
    keys = ("prompt_token_ids_identical", "text_identical", "ids_identical", "pieces_identical", "first_divergence_index_pieces",
            "common_prefix_tokens_pieces", "retokenized_ids_identical", "first_divergence_index_retok",
            "chosen_logprob_max_abs_diff_on_common_prefix", "top5_logprob_positions_exactly_equal",
            "chosen_prob_max_abs_diff_on_common_prefix", "top5_prob_lists_same_candidates", "top5_prob_positions_exactly_equal",
            "top5_prob_max_abs_diff", "chosen_probability_max_abs_diff_cross_family")
    for k, rows in res["pairs"].items():
        print("==", k)
        for pid, r in rows.items() if isinstance(rows, dict) else []:
            if isinstance(r, dict):
                print("  ", pid, {kk: r[kk] for kk in keys if kk in r and r[kk] is not None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
