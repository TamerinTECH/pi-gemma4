#!/usr/bin/env python3
"""Group the per-operation report of the CF-Engine runner (reference mode) into a few classes and show, per phase,
time per pass, modeled weight bytes per pass and the resulting read rate. An operation that already runs at the
board's DRAM read rate (about 12 GB/s) cannot be made faster by another weight layout or kernel.

Usage: analyze_op_profile.py REPORT.json [--gguf MODEL.gguf] [--requests REQUESTS.jsonl] [--out FILE.json]
  --gguf      take the size of every weight tensor from the model file (experts: the routed share only)
  --requests  request log of the runner (prompt tokens per prefill pass)
"""
import argparse
import collections
import json
import re
import sys


def group(op):
    w = op.get("weight_name") or ""
    name = op.get("name") or ""
    if re.search(r"_exps", w):
        return "expert feed-forward (gate/up/down of the routed experts)"
    if re.search(r"ffn_gate_inp|router", w):
        return "expert router"
    if re.search(r"ffn_(gate|up|down)", w):
        return "dense feed-forward"
    if re.search(r"attn_(q|k|v|qkv|output|out)", w):
        return "attention projections"
    if re.search(r"^output\.|token_embd|output_proj|^per_layer|per_layer", w):
        return "output head / embeddings"
    if w:
        return "other weights"
    if re.search(r"kqv|attn|soft_max|flash", name, re.I) or op.get("op") in ("FLASH_ATTN_EXT", "SOFT_MAX"):
        return "attention over the context (no weights)"
    return "other operations without weights"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--out", default="")
    ap.add_argument("--gguf", default="")
    ap.add_argument("--requests", default="")
    a = ap.parse_args()
    with open(a.report, encoding="utf-8") as f:
        j = json.load(f)
    ops = j["ops"]
    sizes, frac = {}, 1.0
    if a.gguf:
        sys.path.insert(0, "third_party/llama.cpp/gguf-py")
        from gguf import GGUFReader  # noqa: PLC0415
        rd = GGUFReader(a.gguf)
        sizes = {t.name: int(t.n_bytes) for t in rd.tensors}
        meta = {k: f.parts[f.data[0]] for k, f in rd.fields.items() if k.endswith(("expert_count", "expert_used_count"))}
        n_exp = next((int(v[0]) for k, v in meta.items() if k.endswith(".expert_count")), 0)
        n_used = next((int(v[0]) for k, v in meta.items() if k.endswith(".expert_used_count")), 0)
        frac = n_used / n_exp if n_exp else 1.0
        print(f"model file: {len(sizes)} tensors, experts {n_used} of {n_exp} routed per token")
    prefill_tokens = 0
    if a.requests:
        with open(a.requests, encoding="utf-8") as f:
            prefill_tokens = sum(json.loads(ln).get("n_prefill", 0) for ln in f if ln.strip())
    out = {"report": a.report, "phases": {}}
    for phase in ("decode", "prefill"):
        sel = [o for o in ops if o.get("phase") == phase]
        if not sel:
            continue
        # passes of this phase = calls of the final output operation
        heads = [o for o in sel if o.get("name") == "result_output"]
        passes = max((o["calls"] for o in heads), default=0) or max(o["calls"] for o in sel)
        tot_ms = sum(o["elapsed_ms"] for o in sel)
        g_ms, g_bytes, g_calls = collections.Counter(), collections.Counter(), collections.Counter()
        for o in sel:
            g = group(o)
            g_ms[g] += o["elapsed_ms"]
            w = o.get("weight_name") or ""
            if w in sizes and phase == "decode":
                g_bytes[g] += int(sizes[w] * (frac if "_exps" in w else 1.0)) * o["calls"]
            else:
                g_bytes[g] += o.get("logical_weight_bytes") or 0
            g_calls[g] += o["calls"]
        per_tok = f", {prefill_tokens} prompt tokens = {tot_ms / prefill_tokens:.1f} ms per token" if phase == "prefill" and prefill_tokens else ""
        print(f"== {phase}: {passes} passes, {tot_ms / passes:.1f} ms of graph time per pass{per_tok}")
        rows = {}
        for g, ms in g_ms.most_common():
            b = g_bytes[g]
            rate = b / (ms / 1000.0) / 1e9 if ms > 0 and b else None
            rows[g] = {"ms_per_pass": round(ms / passes, 2), "share": round(ms / tot_ms, 3),
                       "ms_per_prompt_token": round(ms / prefill_tokens, 2) if phase == "prefill" and prefill_tokens else None,
                       "modeled_weight_mb_per_pass": round(b / passes / 1e6, 1), "read_rate_gb_s": round(rate, 2) if rate else None}
            print(f"  {g:62s} {ms / passes:8.2f} ms  {100 * ms / tot_ms:5.1f} %  {b / passes / 1e6:8.1f} MB  " + (f"{rate:5.2f} GB/s" if rate else "    -"))
        out["phases"][phase] = {"passes": passes, "ms_per_pass": round(tot_ms / passes, 2), "groups": rows,
                                "modeled_weight_mb_per_pass": round(sum(g_bytes.values()) / passes / 1e6, 1)}
    for k in ("peak_rss_kib", "major_faults", "reference_op_ms"):
        out[k] = j.get(k)
    out["request_execution"] = [{x: r.get(x) for x in ("request", "reference_ms", "storage_read_bytes", "major_faults")} for r in j.get("request_execution", [])]
    print("peak RSS MB:", round((j.get("peak_rss_kib") or 0) / 1024), "| major faults:", j.get("major_faults"),
          "| storage bytes read per request:", [r.get("storage_read_bytes") for r in j.get("request_execution", [])])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
