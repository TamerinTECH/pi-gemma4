#!/usr/bin/env python3
"""Summarise mtp_ab.py results (ab.json): speed per configuration, model passes, and exact text identity vs the base.

Usage: analyze_verified_ceiling.py <ab.json> [<ab.json> ...] [--out summary.json]
A model pass commits 1 sampled token plus the accepted draft tokens, so passes = generated - accepted.
"""
import argparse
import json
import statistics
import sys


def summarise(path):
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    base = {}
    for run in j["runs"]:
        if run["config"] == "base" and not base:
            base = {r["id"]: r for r in run["requests"]}
    rows = []
    for run in j["runs"]:
        reqs = run.get("requests") or []
        row = {"config": run["config"], "index": run["index"], "error": run.get("error"), "n_requests": len(reqs)}
        if reqs:
            gen = sum(r["gen_n"] or 0 for r in reqs)
            acc = sum(r["draft_n_accepted"] or 0 for r in reqs)
            drafted = sum(r["draft_n"] or 0 for r in reqs)
            decode_s = sum((r["gen_n"] or 0) / r["gen_tok_s"] for r in reqs if r.get("gen_tok_s"))
            wall_s = sum(r["wall_s"] for r in reqs)
            passes = gen - acc
            same = [r["id"] for r in reqs if r["id"] in base and r.get("text_sha256") == base[r["id"]].get("text_sha256")]
            diff = [r["id"] for r in reqs if r["id"] in base and r.get("text_sha256") != base[r["id"]].get("text_sha256")]
            row.update({
                "generated": gen, "drafted": drafted, "accepted": acc,
                "acceptance": round(acc / drafted, 3) if drafted else None,
                "decode_tok_s_total": round(gen / decode_s, 2) if decode_s else None,
                "decode_tok_s_median": round(statistics.median(r["gen_tok_s"] for r in reqs if r.get("gen_tok_s")), 2),
                "decode_tok_s_min": round(min(r["gen_tok_s"] for r in reqs if r.get("gen_tok_s")), 2),
                "decode_tok_s_max": round(max(r["gen_tok_s"] for r in reqs if r.get("gen_tok_s")), 2),
                "request_tok_s_total": round(gen / wall_s, 2) if wall_s else None,
                "prompt_tok_s_median": round(statistics.median(r["prompt_tok_s"] for r in reqs if r.get("prompt_tok_s")), 1),
                "passes": passes, "tokens_per_pass": round(gen / passes, 2) if passes else None,
                "ms_per_pass": round(1000 * decode_s / passes, 1) if passes else None,
                "text_identical_to_base": len(same), "text_differs_from_base": diff,
                "visible_answer_chars": sum(len(r.get("content") or "") for r in reqs),
                "reasoning_chars": sum(len(r.get("reasoning") or "") for r in reqs),
                "finish": sorted({str(r.get("finish")) for r in reqs}),
                "max_temp_c": max((r.get("temp_c") or 0) for r in reqs),
                "per_request": {r["id"]: [r["gen_n"], round(r["gen_tok_s"], 2) if r.get("gen_tok_s") else None,
                                          r.get("draft_n"), r.get("draft_n_accepted")] for r in reqs},
            })
        rows.append(row)
    return {"file": path, "label": j.get("label"), "model": j.get("model"), "configs": j.get("configs"), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = [summarise(p) for p in a.files]
    for s in out:
        print(f"== {s['label']}  ({s['model']})")
        for r in s["rows"]:
            if r.get("error") and not r.get("generated"):
                print(f"  {r['config']:9s} ERROR {r['error']}")
                continue
            print(f"  {r['config']:9s} decode {r['decode_tok_s_total']:6.2f} tok/s (median {r['decode_tok_s_median']}, "
                  f"{r['decode_tok_s_min']}-{r['decode_tok_s_max']}) | request {r['request_tok_s_total']} | "
                  f"tok/pass {r['tokens_per_pass']} | ms/pass {r['ms_per_pass']} | accept {r['acceptance']} | "
                  f"same text {r['text_identical_to_base']}/{r['n_requests']} diff {r['text_differs_from_base']} | "
                  f"answer chars {r['visible_answer_chars']} reasoning {r['reasoning_chars']} | {r['max_temp_c']} C")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
