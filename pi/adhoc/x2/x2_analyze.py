#!/usr/bin/env python3
"""X2: summarise x2_compare.py results (runs on the desktop, stdlib only).

  python pi/adhoc/x2/x2_analyze.py tune  results/comparisons/ik_llama_pi/tune
  python pi/adhoc/x2/x2_analyze.py eval  results/comparisons/ik_llama_pi/eval

tune: one line per block + the pre-declared choice (smallest sum of client total latency over the tuning
      requests; ties within 2 % go to -t 4, then (ik) to --swa-compress).
eval: per engine x prompt over the repeats: mean / min / max of prefill tok/s, decode tok/s, TTFT, total
      latency, ITL p99 / max; paired ik/ours ratios per repeat; memory peaks; temperature / throttling;
      prompt-token parity and output comparison (within engine across repeats, and across engines).
Writes <dir>/summary.json and prints Markdown tables.
"""
import json
import os
import statistics
import sys


def load_blocks(d):
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name, "block.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                b = json.load(f)
            b["_dir"] = os.path.join(d, name)
            tp = os.path.join(d, name, "prompt_tokens.json")
            if os.path.isfile(tp):
                with open(tp) as f:
                    b["_prompt_tokens"] = json.load(f)
            out.append(b)
    out.sort(key=lambda b: b.get("started", ""))
    return out


def reqs(b):
    return [r for r in b.get("requests", []) if not r.get("discarded_warmup")]


def fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def msr(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return fmt(float(xs[0]), nd)
    return f"{statistics.mean(xs):.{nd}f} ({min(xs):.{nd}f}-{max(xs):.{nd}f})"


def cv(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2 or statistics.mean(xs) == 0:
        return None
    return statistics.stdev(xs) / statistics.mean(xs)


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def norm_latency(r):
    """Projected time to prefill the prompt and generate the FULL bound: TTFT + (n_predict - 1) / client decode rate."""
    if not r or r.get("ttft_s") is None or not r.get("decode_tok_s_client"):
        return None
    return r["ttft_s"] + (r["n_predict"] - 1) / r["decode_tok_s_client"]


def tune(*dirs):
    blocks = []
    for d in dirs:
        blocks += load_blocks(d)
    rows, per_engine = [], {}
    print("| block | engine | config | load s | " + " | ".join(
        f"{x} prompt tok | {x} gen tok | {x} pp tok/s | {x} tg tok/s (server) | {x} tg tok/s (client) | {x} TTFT s | {x} total s"
        for x in ("tun-01", "tun-06")) +
        " | sum total s (pass-1 rule) | S s (revised rule) | RssAnon peak MiB | temp max C | error |")
    print("|" + "---|" * 23)
    for b in blocks:
        rq = {r["id"]: r for r in reqs(b)}
        tot = sum(r["total_s"] for r in rq.values()) if len(rq) == 2 else None
        nl = [norm_latency(r) for r in rq.values()]
        S = sum(nl) if len(rq) == 2 and all(x is not None for x in nl) else None
        row = {"label": b["label"], "engine": b["engine"], "config": b.get("config"), "load_s": b.get("load_s"),
               "sum_total_s": tot, "S_norm_s": S, "RssAnon_peak_kB": (b.get("block_peaks") or {}).get("RssAnon_max_kB"),
               "temp_max_c": (b.get("block_peaks") or {}).get("temp_max_c"), "error": b.get("error") or b.get("aborted"),
               "throttled_values": (b.get("block_peaks") or {}).get("throttled_values"),
               "freq_min_khz": (b.get("block_peaks") or {}).get("freq_min_khz"),
               "requests": {k: {m: v.get(m) for m in ("prompt_n", "prompt_tok_s", "gen_n", "gen_tok_s", "ttft_s", "total_s",
                                                         "decode_tok_s_client", "itl_median_s", "itl_max_s", "read_bytes",
                                                         "majflt", "cpu_s")}
                            for k, v in rq.items()}}
        rows.append(row)
        cells = []
        for x in ("tun-01", "tun-06"):
            r = rq.get(x, {})
            cells += [fmt(r.get("prompt_n")), fmt(r.get("gen_n")), fmt(r.get("prompt_tok_s")), fmt(r.get("gen_tok_s"), 3),
                      fmt(r.get("decode_tok_s_client"), 3), fmt(r.get("ttft_s")), fmt(r.get("total_s"))]
        an = row["RssAnon_peak_kB"]
        print(f"| {b['label']} | {b['engine']} | {b.get('config')} | {fmt(b.get('load_s'), 1)} | " + " | ".join(cells) +
              f" | {fmt(tot)} | {fmt(S)} | {fmt(an / 1024 if an else None, 0)} | {fmt(row['temp_max_c'], 1)} | {row['error'] or ''} |")
        if S is not None:
            per_engine.setdefault(b["engine"], []).append(row)
    choice = {}
    for eng, rs in per_engine.items():
        best = min(r["S_norm_s"] for r in rs)
        tied = [r for r in rs if r["S_norm_s"] <= best * 1.02]

        def pref(r):
            c = r["config"] or ""
            return (0 if c.startswith("t4") else 1, 0 if (eng != "ik" or "swac" in c) else 1, r["S_norm_s"])
        pick = sorted(tied, key=pref)[0]
        choice[eng] = {"chosen": pick["config"], "label": pick["label"], "S_norm_s": pick["S_norm_s"],
                       "best_S_norm_s": best, "tied_within_2pct": [r["config"] for r in tied],
                       "all_S_norm_s": {r["config"]: round(r["S_norm_s"], 2) for r in rs},
                       "all_sum_total_s_pass1_rule": {r["config"]: round(r["sum_total_s"], 2) for r in rs}}
    print("\nChoice (revised rule S; ties within 2 % -> -t 4, then ik --swa-compress):", json.dumps(choice, indent=1))
    with open(os.path.join(dirs[-1], "tuning_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"dirs": list(dirs), "rows": rows, "choice": choice}, f, indent=1)


METRICS = [("prompt_tok_s", "prefill tok/s (server)", 2), ("gen_tok_s", "decode tok/s (server)", 3),
           ("decode_tok_s_client", "decode tok/s (client)", 3), ("ttft_s", "TTFT s", 2), ("total_s", "total s", 2),
           ("itl_median_s", "ITL median s", 3), ("itl_p90_s", "ITL p90 s", 3), ("itl_p99_s", "ITL p99 s", 3),
           ("itl_max_s", "ITL max s", 3)]


def evaluate(d):
    blocks = load_blocks(d)
    engines = sorted({b["engine"] for b in blocks})
    order = [(b["label"], b["engine"], b.get("started"), b.get("error") or b.get("aborted")) for b in blocks]
    by = {}       # (engine, prompt) -> list of request dicts in block order
    for b in blocks:
        for r in reqs(b):
            by.setdefault((b["engine"], r["id"]), []).append(dict(r, _block=b["label"]))
    pids = []
    for b in blocks:
        for r in reqs(b):
            if r["id"] not in pids:
                pids.append(r["id"])
    summ = {"order": order, "engines": engines, "prompts": pids, "per_prompt": {}, "ratios": {}, "memory": {}, "thermal": {},
            "outputs": {}, "tokenization": {}}
    print("Block order:", " -> ".join(f"{o[0]}" for o in order))
    print()
    print("| prompt | engine | n | prompt tok | gen tok | " + " | ".join(m[1] for m in METRICS) + " | read MB | majflt |")
    print("|" + "---|" * (7 + len(METRICS)))
    for pid in pids:
        for e in engines:
            rs = by.get((e, pid), [])
            if not rs:
                continue
            s = {m[0]: [r.get(m[0]) for r in rs] for m in METRICS}
            s["prompt_n"] = [r.get("prompt_n") for r in rs]
            s["gen_n"] = [r.get("gen_n") for r in rs]
            s["n_events"] = [r.get("n_events") for r in rs]
            s["read_bytes"] = [r.get("read_bytes") for r in rs]
            s["majflt"] = [r.get("majflt") for r in rs]
            s["cpu_s"] = [r.get("cpu_s") for r in rs]
            s["stalls_gt_1s"] = [r.get("stalls_gt_1s") for r in rs]
            s["stalls_gt_2x_median"] = [r.get("stalls_gt_2x_median") for r in rs]
            s["cv"] = {m[0]: cv(s[m[0]]) for m in METRICS}
            summ["per_prompt"].setdefault(pid, {})[e] = s
            print(f"| {pid} | {e} | {len(rs)} | {sorted(set(s['prompt_n']))} | {sorted(set(s['gen_n']))} | " +
                  " | ".join(msr(s[m[0]], m[2]) for m in METRICS) +
                  f" | {msr([x / 1e6 for x in s['read_bytes'] if x is not None], 1)} | {sorted(set(s['majflt']))} |")
    if len(engines) == 2 and "ours" in engines:
        other = [e for e in engines if e != "ours"][0]
        print(f"\nPaired ratios {other} / ours (repeat k of {other} vs repeat k of ours; >1 = {other} larger):\n")
        print("| prompt | " + " | ".join(m[1] for m in METRICS) + " |")
        print("|" + "---|" * (1 + len(METRICS)))
        for pid in pids:
            a, b_ = by.get(("ours", pid), []), by.get((other, pid), [])
            n = min(len(a), len(b_))
            row = {}
            for m in METRICS:
                rr = [b_[i][m[0]] / a[i][m[0]] for i in range(n) if a[i].get(m[0]) and b_[i].get(m[0])]
                row[m[0]] = rr
            summ["ratios"][pid] = row
            print(f"| {pid} | " + " | ".join(msr(row[m[0]], 3) for m in METRICS) + " |")
    # memory + thermal per block
    print("\n| block | engine | load s | RssAnon after load MiB | RssAnon peak MiB | RssFile peak MiB | VmHWM MiB | Pss_Anon end MiB |"
          " MemAvailable min MiB | swap peak MiB | temp start C | temp max C | freq min MHz | throttled values | read MB (load) |")
    print("|" + "---|" * 15)
    for b in blocks:
        pk = b.get("block_peaks") or {}
        al = b.get("after_load") or {}
        ar = b.get("after_requests") or {}

        def mib(v):
            return fmt(v / 1024, 0) if v is not None else "n/a"
        summ["memory"][b["label"]] = {"engine": b["engine"], "load_s": b.get("load_s"), "peaks": pk,
                                      "after_load": al, "after_requests": ar,
                                      "pre_residency": (b.get("pre") or {}).get("model_residency")}
        summ["thermal"][b["label"]] = {"temp_start_c": (b.get("pre") or {}).get("temp_c"), "temp_max_c": pk.get("temp_max_c"),
                                       "freq_min_khz": pk.get("freq_min_khz"), "throttled_values": pk.get("throttled_values"),
                                       "throttled_pre": (b.get("pre") or {}).get("throttled"),
                                       "throttled_post": (b.get("post") or {}).get("throttled")}
        fm = pk.get("freq_min_khz")
        rb = al.get("read_bytes")
        print(f"| {b['label']} | {b['engine']} | {fmt(b.get('load_s'), 1)} | {mib((al.get('status_kB') or {}).get('RssAnon'))} | "
              f"{mib(pk.get('RssAnon_max_kB'))} | {mib(pk.get('RssFile_max_kB'))} | {mib(pk.get('VmHWM_kB'))} | "
              f"{mib((ar.get('smaps_rollup_kB') or {}).get('Pss_Anon'))} | {mib(pk.get('MemAvailable_min_kB'))} | "
              f"{mib(pk.get('VmSwap_max_kB'))} | {fmt((b.get('pre') or {}).get('temp_c'), 1)} | {fmt(pk.get('temp_max_c'), 1)} | "
              f"{fmt(fm / 1000 if fm else None, 0)} | {pk.get('throttled_values')} | {fmt(rb / 1e6 if rb is not None else None, 1)} |")
    # pooled inter-token gaps per engine (all held-out requests, all repeats)
    print()
    print("Pooled inter-token gaps per engine (every streamed event of every measured request):")
    print()
    print("| engine | scope | n gaps | median s | p90 s | p99 s | max s | gaps > 2x median | gaps > 1 s |")
    print("|---|---|---|---|---|---|---|---|---|")
    summ["itl_pooled"] = {}
    for e in engines:
        pools = {"all": [], "short prompts": [], "long prompt": []}
        for b in blocks:
            if b["engine"] != e:
                continue
            for r in reqs(b):
                ep = os.path.join(b["_dir"], f"events-{r['id']}.json")
                if not os.path.isfile(ep):
                    continue
                with open(ep) as f:
                    g = json.load(f).get("gaps_s", [])
                pools["all"] += g
                pools["long prompt" if (r.get("prompt_n") or 0) >= 500 else "short prompts"] += g
        summ["itl_pooled"][e] = {}
        for scope, g in pools.items():
            if not g:
                continue
            sg = sorted(g)

            def q(x):
                return sg[min(len(sg) - 1, max(0, int(round(x / 100.0 * (len(sg) - 1)))))]
            med = q(50)
            st = {"n": len(g), "median_s": med, "p90_s": q(90), "p99_s": q(99), "max_s": sg[-1],
                  "gt_2x_median": sum(1 for x in g if x > 2 * med), "gt_1s": sum(1 for x in g if x > 1.0)}
            summ["itl_pooled"][e][scope] = st
            print(f"| {e} | {scope} | {st['n']} | {st['median_s']:.3f} | {st['p90_s']:.3f} | {st['p99_s']:.3f} | {st['max_s']:.3f} | "
                  f"{st['gt_2x_median']} | {st['gt_1s']} |")
    # tokenisation parity
    print("\nPrompt tokenisation parity (ids from each engine's /tokenize, BOS included):")
    for pid in ["warmup"] + pids:
        seqs = {}
        for b in blocks:
            t = (b.get("_prompt_tokens") or {}).get(pid)
            if isinstance(t, list):
                seqs.setdefault(b["engine"], []).append(tuple(t))
        info = {e: {"n_tokens": sorted({len(x) for x in v}), "stable": len(set(v)) == 1} for e, v in seqs.items()}
        same = len({v[0] for v in seqs.values()}) == 1 if seqs else None
        summ["tokenization"][pid] = {"per_engine": info, "identical_across_engines": same}
        print(f"  {pid}: {info} identical_across_engines={same}")
    # outputs
    print("\nOutputs (greedy):")
    for pid in pids:
        o = {}
        for e in engines:
            rs = by.get((e, pid), [])
            texts = [r.get("output", "") for r in rs]
            toks = [r.get("output_token_ids_retokenized") or [] for r in rs]
            o[e] = {"n_runs": len(rs), "identical_across_repeats": len(set(texts)) == 1, "n_distinct": len(set(texts)),
                    "gen_n": [r.get("gen_n") for r in rs], "chars": [len(t) for t in texts], "first": texts[0] if texts else None,
                    "first_tokens": toks[0] if toks else None,
                    "min_common_prefix_tokens_vs_first": min((common_prefix(toks[0], t) for t in toks), default=None)}
        cross = None
        if len(engines) == 2:
            a, b_ = o[engines[0]], o[engines[1]]
            if a["first"] is not None and b_["first"] is not None:
                ta, tb = a["first_tokens"] or [], b_["first_tokens"] or []
                cross = {"identical_text": a["first"] == b_["first"], "common_prefix_chars": common_prefix(a["first"], b_["first"]),
                         "common_prefix_tokens": common_prefix(ta, tb), "len_tokens": [len(ta), len(tb)]}
        summ["outputs"][pid] = {"per_engine": {e: {k: v for k, v in o[e].items() if k != "first_tokens"} for e in o},
                                "cross_engine": cross}
        print(f"  {pid}: " + "; ".join(f"{e}: repeats identical={o[e]['identical_across_repeats']} gen_n={o[e]['gen_n']}" for e in o) +
              f"; cross-engine={cross}")
    with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"tune": tune, "eval": evaluate}[sys.argv[1]](*sys.argv[2:])
