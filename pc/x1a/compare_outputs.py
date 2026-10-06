#!/usr/bin/env python3
"""X1a: compare the greedy outputs of the engines recorded by pc/x1a/functional_check.py.

    python pc/x1a/compare_outputs.py results/comparisons/x1a/functional/<run>/functional_check.json

Writes comparison.json next to the input and prints a short table. Differences ACROSS revisions are
reported, not judged: a different token is a numerical difference to document, not a correctness failure.
Within one engine (run a vs run b vs the streaming client) the outputs must be identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def first_div(a: list, b: list) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def pair(pa: dict, pb: dict) -> dict:
    ta, tb = pa["a"]["token_ids"] or [], pb["a"]["token_ids"] or []
    d = first_div(ta, tb)
    common = len(ta) if d is None else d
    qa, qb = pa["a"]["probs"], pb["a"]["probs"]
    dl: list[float] = []
    top_same = 0
    top_max = 0.0
    bitwise_equal_positions = 0
    for i in range(min(common, len(qa), len(qb))):
        la, lb = qa[i].get("logprob"), qb[i].get("logprob")
        if la is not None and lb is not None:
            dl.append(abs(la - lb))
            if la == lb:
                bitwise_equal_positions += 1
        ia = [t["id"] for t in qa[i]["top"]]
        ib = [t["id"] for t in qb[i]["top"]]
        if ia == ib:
            top_same += 1
            for x, y in zip(qa[i]["top"], qb[i]["top"]):
                if x.get("logprob") is not None and y.get("logprob") is not None:
                    top_max = max(top_max, abs(x["logprob"] - y["logprob"]))
    return {
        "tokens_identical": d is None,
        "first_divergence_index": d,
        "n_tokens": [len(ta), len(tb)],
        "text_identical": pa["a"]["text"] == pb["a"]["text"],
        "prompt_token_ids_identical": pa.get("prompt_token_ids") == pb.get("prompt_token_ids"),
        "n_prompt_tokens": [len(pa.get("prompt_token_ids") or []), len(pb.get("prompt_token_ids") or [])],
        "common_prefix_tokens": common,
        "chosen_logprob_max_abs_diff": max(dl) if dl else None,
        "chosen_logprob_mean_abs_diff": (sum(dl) / len(dl)) if dl else None,
        "chosen_logprob_exactly_equal_positions": bitwise_equal_positions,
        "top5_id_lists_identical_positions": top_same,
        "top5_logprob_max_abs_diff_where_ids_match": top_max,
        "logprob_note": "logprobs as printed by llama-server JSON (softmax of the raw logits; greedy sampling)",
    }


def main() -> int:
    src = Path(sys.argv[1])
    fc = json.loads(src.read_text(encoding="utf-8"))
    engs = {e["engine_id"]: e for e in fc["engines"]}
    out: dict = {"schema": "pigemma-x1a-comparison v1", "source": str(src).replace("\\", "/"),
                 "kind": fc.get("kind"), "model_sha256_ok": fc.get("model_sha256_ok"),
                 "model_sha256_measured": fc.get("model_sha256_measured"), "n_predict": fc.get("n_predict"),
                 "prompts": [{k: p[k] for k in ("id", "lang", "category", "set", "prompt_sha256")} for p in fc["prompts"]],
                 "engines": {}, "within_engine": {}, "across_engines": {}}
    for eid, e in engs.items():
        out["engines"][eid] = {
            "ok": e.get("ok"), "error": e.get("error"), "version_output": e.get("version_output"),
            "bin_sha256": e.get("bin_sha256"), "bin_bytes": e.get("bin_bytes"),
            "system_info": e.get("system_info"), "server_ready_s": e.get("server_ready_s"),
            "repack_present": (e.get("server_log") or {}).get("repack_present"),
            "buffers_mib": (e.get("server_log") or {}).get("buffers_mib"),
            "arg_or_error_lines": (e.get("server_log") or {}).get("arg_or_error_lines"),
            "props": e.get("props"), "server_rc": e.get("server_rc"),
            "timings_functional_only": {pid: {tag: r[tag]["timings_functional_only"] for tag in ("a", "b")}
                                        for pid, r in (e.get("requests") or {}).items()},
        }
        w = {}
        for pid, r in (e.get("requests") or {}).items():
            a, b, s = r["a"], r["b"], r["stream"]
            la = [q.get("logprob") for q in a["probs"]]
            lb = [q.get("logprob") for q in b["probs"]]
            w[pid] = {"a_eq_b_tokens": a["token_ids"] == b["token_ids"],
                      "a_eq_stream_tokens": a["token_ids"] == s["token_ids"],
                      "a_eq_b_text": a["text"] == b["text"],
                      "a_eq_b_logprobs_exact": la == lb,
                      "n_tokens": len(a["token_ids"] or []), "stop_type": a.get("stop_type"),
                      "text": a["text"]}
        out["within_engine"][eid] = w
    ids = [e for e in engs if engs[e].get("ok")]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            key = f"{ids[i]} vs {ids[j]}"
            out["across_engines"][key] = {
                pid: pair(engs[ids[i]]["requests"][pid], engs[ids[j]]["requests"][pid])
                for pid in engs[ids[i]]["requests"] if pid in engs[ids[j]]["requests"]}
    dst = src.with_name("comparison.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for eid, e in out["engines"].items():
        print(f"{eid}: ok={e['ok']} ready={e['server_ready_s']} s repack={e['repack_present']} | {e['version_output']!r}")
        for pid, w in out["within_engine"].get(eid, {}).items():
            print(f"   {pid}: n={w['n_tokens']} a==b {w['a_eq_b_tokens']} a==stream {w['a_eq_stream_tokens']} "
                  f"logprobs a==b {w['a_eq_b_logprobs_exact']}")
    for key, v in out["across_engines"].items():
        for pid, p in v.items():
            print(f"{key} | {pid}: tokens_identical={p['tokens_identical']} first_div={p['first_divergence_index']} "
                  f"max|dlogprob|={p['chosen_logprob_max_abs_diff']} exact_eq_pos={p['chosen_logprob_exactly_equal_positions']}"
                  f"/{p['common_prefix_tokens']} top5_same={p['top5_id_lists_identical_positions']}")
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
