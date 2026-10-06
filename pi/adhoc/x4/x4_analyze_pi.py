#!/usr/bin/env python3
"""X4: analysis of the Pi-side resident-bottleneck results (desktop, stdlib only, reads files only).

Input : the directory fetched from the Pi by pc/x4/fetch_results_from_pi.sh (membw_*.json, kernbench.json,
        threads/ longgen/ repack/ qat/ udref/ block directories written by x4_compare.py) and
        results/pi-real/x4-resident-bottleneck/existing_evidence.json (bytes per decoded token by quant type).
Output: <pi-dir>/../pi_analysis.json and pi_tables.md (Markdown tables ready for docs/results/RESIDENT_BOTTLENECK_PI.md).
Missing parts are reported as missing, never guessed.

Usage: python pi/adhoc/x4/x4_analyze_pi.py --pi-dir results/pi-real/x4-resident-bottleneck/pi
"""
import argparse
import glob
import json
import os
import statistics

GB = 1e9


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=2):
    return "n/a" if x is None else ("%." + str(nd) + "f") % x


def read_blocks(d):
    out = []
    for path in sorted(glob.glob(os.path.join(d, "*", "block.json"))):
        b = load(path)
        if not b.get("requests"):
            out.append({"label": b.get("label"), "error": b.get("error") or b.get("aborted"), "requests": []})
            continue
        ev_dir = os.path.dirname(path)
        reqs = []
        for q in b["requests"]:
            if q.get("discarded_warmup"):
                continue
            tag = q["id"] if sum(1 for x in b["requests"] if x["id"] == q["id"]) == 1 else "%s.%d" % (q["id"], q.get("repeat_index", 0))
            gaps = []
            ep = os.path.join(ev_dir, "events-%s.json" % tag)
            if os.path.exists(ep):
                gaps = load(ep).get("gaps_s") or []
            reqs.append({
                "id": q["id"], "repeat_index": q.get("repeat_index", 0), "prompt_n": q.get("prompt_n"),
                "prompt_tok_s": q.get("prompt_tok_s"), "gen_n": q.get("gen_n"), "gen_tok_s": q.get("gen_tok_s"),
                "decode_tok_s_client": q.get("decode_tok_s_client"), "ttft_s": q.get("ttft_s"), "total_s": q.get("total_s"),
                "itl_median_s": q.get("itl_median_s"), "itl_p99_s": q.get("itl_p99_s"), "itl_max_s": q.get("itl_max_s"),
                "read_bytes": q.get("read_bytes"), "majflt": q.get("majflt"), "cpu_s": q.get("cpu_s"),
                "phase_prefill": q.get("phase_prefill"), "phase_decode": q.get("phase_decode"),
                "perf_prefill": q.get("perf_prefill"), "perf_decode": q.get("perf_decode"),
                "temp_max_c": (q.get("window") or {}).get("temp_max_c"),
                "freq_min_khz": (q.get("window") or {}).get("freq_min_khz"),
                "MemAvailable_min_kB": (q.get("window") or {}).get("MemAvailable_min_kB"),
                "RssAnon_max_kB": (q.get("window") or {}).get("RssAnon_max_kB"),
                "throttled_after": q.get("throttled_after"), "gaps_s": gaps,
            })
        out.append({"label": b["label"], "engine": b.get("engine"), "config": b.get("config"), "command": b.get("command"),
                    "started": b.get("started"), "load_s": b.get("load_s"), "error": b.get("error"),
                    "after_load": b.get("after_load"), "block_peaks": b.get("block_peaks"), "perf": b.get("perf"),
                    "pre": {k: (b.get("pre") or {}).get(k) for k in ("MemAvailable_kB", "temp_c", "throttled", "model_residency",
                                                                     "prewarm_read_s", "cool_down_wait_s__temp_c")},
                    "post_services": (b.get("post") or {}).get("services"), "requests": reqs})
    return out


def zero(q):
    return q.get("read_bytes") == 0 and q.get("majflt") == 0


def decode_zero(q):
    ph = q.get("phase_decode")
    return ph is not None and ph.get("read_bytes") == 0 and ph.get("majflt") == 0


def summarize_requests(reqs, bytes_per_token):
    """Aggregate a list of request rows (same thread count, same prompt class)."""
    if not reqs:
        return None
    dec = [q["decode_tok_s_client"] for q in reqs if q["decode_tok_s_client"]]
    pp = [q["prompt_tok_s"] for q in reqs if q["prompt_tok_s"]]
    out = {"n": len(reqs), "decode_tok_s_mean": mean(dec), "decode_tok_s_min": min(dec) if dec else None,
           "decode_tok_s_max": max(dec) if dec else None, "prefill_tok_s_mean": mean(pp),
           "prefill_tok_s_min": min(pp) if pp else None, "prefill_tok_s_max": max(pp) if pp else None,
           "itl_median_s_mean": mean([q["itl_median_s"] for q in reqs]),
           "itl_max_s_max": max([q["itl_max_s"] for q in reqs if q["itl_max_s"] is not None], default=None),
           "temp_max_c": max([q["temp_max_c"] for q in reqs if q["temp_max_c"] is not None], default=None),
           "freq_min_khz": min([q["freq_min_khz"] for q in reqs if q["freq_min_khz"] is not None], default=None),
           "read_bytes_total": sum(q["read_bytes"] or 0 for q in reqs),
           "decode_read_bytes_total": sum((q.get("phase_decode") or {}).get("read_bytes") or 0 for q in reqs),
           "decode_majflt_total": sum((q.get("phase_decode") or {}).get("majflt") or 0 for q in reqs),
           "decoded_tokens": sum(q["gen_n"] or 0 for q in reqs)}
    if bytes_per_token and out["decode_tok_s_mean"]:
        out["effective_GB_s"] = round(out["decode_tok_s_mean"] * bytes_per_token / GB, 2)
    # CPU seconds per token, per phase
    dcpu = [(q["phase_decode"]["cpu_s"], q["gen_n"]) for q in reqs if q.get("phase_decode") and q["gen_n"]]
    pcpu = [(q["phase_prefill"]["cpu_s"], q["prompt_n"]) for q in reqs if q.get("phase_prefill") and q["prompt_n"]]
    if dcpu:
        out["decode_cpu_s_per_token"] = round(sum(c for c, _ in dcpu) / sum(n for _, n in dcpu), 4)
    if pcpu:
        out["prefill_cpu_s_per_token"] = round(sum(c for c, _ in pcpu) / sum(n for _, n in pcpu), 4)
    for ph in ("decode", "prefill"):
        ps = [q.get("perf_" + ph) for q in reqs if q.get("perf_" + ph)]
        if ps:
            cyc = sum(p.get("cycles", 0) for p in ps)
            out["perf_" + ph] = {
                "ipc": round(sum(p.get("instructions", 0) for p in ps) / cyc, 3) if cyc else None,
                "stall_backend_fraction": round(sum(p.get("stall_backend", 0) for p in ps) / cyc, 4)
                if cyc and any("stall_backend" in p for p in ps) else None,
                "l3d_refill_bytes_per_token": None, "l2d_refill_bytes_per_token": None,
                "min_running_ratio": min(p.get("min_running_ratio", 1.0) for p in ps)}
            ntok = sum((q["gen_n"] if ph == "decode" else q["prompt_n"]) or 0 for q in reqs if q.get("perf_" + ph))
            if ntok:
                for ev, key in (("l3d_cache_refill", "l3d_refill_bytes_per_token"), ("l2d_cache_refill", "l2d_refill_bytes_per_token")):
                    if any(ev in p for p in ps):
                        out["perf_" + ph][key] = round(sum(p.get(ev, 0) for p in ps) * 64 / ntok)
                out["perf_" + ph]["cycles_per_token"] = round(cyc / ntok)
                out["perf_" + ph]["instructions_per_token"] = round(sum(p.get("instructions", 0) for p in ps) / ntok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-dir", required=True)
    ap.add_argument("--evidence", default=None)
    a = ap.parse_args()
    pi = os.path.abspath(a.pi_dir)
    base = os.path.dirname(pi)
    ev_path = a.evidence or os.path.join(base, "existing_evidence.json")
    ev = load(ev_path) if os.path.exists(ev_path) else {}
    ud = (ev.get("per_token_bytes") or {}).get("ud_q3_k_m") or {}
    qat = (ev.get("per_token_bytes") or {}).get("qat_q4_0") or {}
    out = {"stage": "X4-resident-bottleneck-pi", "pi_dir": pi, "missing": []}
    md = []

    # ---- membw ---------------------------------------------------------------------------------------------
    membw = {}
    for path in sorted(glob.glob(os.path.join(pi, "membw_*.json"))):
        try:
            m = load(path)
        except ValueError:
            continue
        membw[os.path.basename(path)[:-5]] = {"size_mb": m.get("size_mb"), "page_size": m.get("page_size"),
                                              "results": m.get("results")}
    best = {}
    for name, m in membw.items():
        for r in m["results"] or []:
            if (m.get("size_mb") or 0) < 512:
                continue
            b = best.setdefault(r["threads"], {"read_gbps": 0, "copy_gbps": 0, "triad_gbps": 0, "read_all": []})
            for k in ("read_gbps", "copy_gbps", "triad_gbps"):
                b[k] = max(b[k], r[k])
            b["read_all"].append(r["read_gbps"])
    out["membw"] = {"runs": membw, "best_by_threads_arena_ge_512MiB": best}
    if not membw:
        out["missing"].append("membw")
    else:
        md.append("### Memory bandwidth on the Pi (tools/membw, measured)\n")
        md.append("| Threads | read GB/s (best) | read GB/s (all runs) | copy GB/s | triad GB/s |")
        md.append("|---|---|---|---|---|")
        for t in sorted(best):
            b = best[t]
            md.append("| %d | %s | %s | %s | %s |" % (t, fmt(b["read_gbps"]), ", ".join(fmt(x) for x in b["read_all"]),
                                                    fmt(b["copy_gbps"]), fmt(b["triad_gbps"])))
        md.append("")

    # ---- kernbench -----------------------------------------------------------------------------------------
    kb_path = os.path.join(pi, "kernbench.json")
    kern = {}
    if os.path.exists(kb_path) and os.path.getsize(kb_path) > 0:
        kb = load(kb_path)
        for r in kb["results"]:
            key = "%s@%d" % (r["type"].upper(), r["n_in"])
            kern.setdefault(key, {}).setdefault(r["threads"], {})[r["regime"]] = {
                "gbps_best": r["gbps_best"], "gbps_median": r["gbps_median"], "pairs_per_s_best": r["pairs_per_s_best"]}
        out["kernbench"] = {"cpu_features": kb.get("cpu_features"), "stream_mb": kb.get("stream_mb"),
                            "cache_kb": kb.get("cache_kb"), "by_type": kern}
        md.append("### Kernel throughput by quantization type (tools/kernbench, measured, ggml mul_mat path without repack)\n")
        md.append("| Type@row | Threads | stream GB/s (512 MiB matrix) | cached GB/s (384 KiB matrix) | stream / cached | "
                  "stream / membw read | cached batch-32 pairs/s |")
        md.append("|---|---|---|---|---|---|---|")
        for key in kern:
            for t in sorted(kern[key]):
                c = kern[key][t]
                s, ch = (c.get("stream") or {}).get("gbps_best"), (c.get("cached") or {}).get("gbps_best")
                mb = (best.get(t) or {}).get("read_gbps")
                md.append("| %s | %d | %s | %s | %s | %s | %s |" % (
                    key, t, fmt(s), fmt(ch), fmt(s / ch if s and ch else None), fmt(s / mb if s and mb else None),
                    "%.3e" % c["cached_batch"]["pairs_per_s_best"] if c.get("cached_batch") else "n/a"))
        md.append("")
    else:
        out["missing"].append("kernbench")

    # ---- server plans ----------------------------------------------------------------------------------------
    plans = {name: read_blocks(os.path.join(pi, name)) for name in ("threads", "longgen", "repack", "qat", "udref")}
    out["plans_raw_summary"] = {k: [{"label": b["label"], "error": b.get("error"), "n_requests": len(b["requests"])} for b in v]
                                for k, v in plans.items()}

    def by_threads(blocks, bpt, only_zero=False):
        res = {}
        for b in blocks:
            if not b["requests"]:
                continue
            t = int(str(b.get("config") or "t0").lstrip("t").split("-")[0] or 0)
            for cls, pred in (("short", lambda q: (q["prompt_n"] or 0) < 100), ("long", lambda q: (q["prompt_n"] or 0) >= 500)):
                sel = [q for q in b["requests"] if pred(q) and (zero(q) or not only_zero)]
                res.setdefault(t, {}).setdefault(cls, []).extend(sel)
        return {t: {cls: summarize_requests(rs, bpt) for cls, rs in d.items()} for t, d in sorted(res.items())}

    # threads
    if plans["threads"]:
        ts = by_threads(plans["threads"], ud.get("bytes_per_decoded_token"))
        out["thread_scaling"] = ts
        md.append("### Thread scaling, UD-Q3_K_M, -nr, resident (measured; two interleaved passes)\n")
        md.append("| Threads | Decode tok/s short ctx (min-max) | Decode tok/s after 687-tok prompt | Prefill tok/s 687 tok | "
                  "Effective GB/s (short) | membw read GB/s | Decode CPU-s/token | Decode IPC | Decode backend-stall | "
                  "Prefill IPC | Storage read in decode | Temp max C | Freq min MHz |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t, d in ts.items():
            s, lg = d.get("short") or {}, d.get("long") or {}
            pd, pp = (s.get("perf_decode") or {}), (lg.get("perf_prefill") or {})
            md.append("| %d | %s (%s-%s) | %s | %s | %s | %s | %s | %s | %s | %s | %s B | %s | %s |" % (
                t, fmt(s.get("decode_tok_s_mean"), 3), fmt(s.get("decode_tok_s_min"), 3), fmt(s.get("decode_tok_s_max"), 3),
                fmt(lg.get("decode_tok_s_mean"), 3), fmt(lg.get("prefill_tok_s_mean")), fmt(s.get("effective_GB_s")),
                fmt((best.get(t) or {}).get("read_gbps")), fmt(s.get("decode_cpu_s_per_token"), 3), fmt(pd.get("ipc"), 2),
                fmt(pd.get("stall_backend_fraction"), 3), fmt(pp.get("ipc"), 2),
                (s.get("decode_read_bytes_total") or 0) + (lg.get("decode_read_bytes_total") or 0),
                fmt(max([x for x in (s.get("temp_max_c"), lg.get("temp_max_c")) if x is not None], default=None), 1),
                fmt((min([x for x in (s.get("freq_min_khz"), lg.get("freq_min_khz")) if x is not None], default=0) or 0) / 1000, 0)))
        md.append("")
    else:
        out["missing"].append("threads")

    # long generation
    if plans["longgen"]:
        lg_rows = []
        for b in plans["longgen"]:
            for q in b["requests"]:
                g = q["gaps_s"]
                n = len(g)
                first, last = (g[: max(1, n // 5)], g[-max(1, n // 5):]) if n else ([], [])
                lg_rows.append({"block": b["label"], "config": b.get("config"), "prompt_n": q["prompt_n"], "gen_n": q["gen_n"],
                                "decode_tok_s_client": q["decode_tok_s_client"], "phase_decode": q.get("phase_decode"),
                                "phase_prefill": q.get("phase_prefill"), "perf_decode": q.get("perf_decode"),
                                "itl_first_fifth_median_s": statistics.median(first) if first else None,
                                "itl_last_fifth_median_s": statistics.median(last) if last else None,
                                "itl_max_s": q["itl_max_s"], "temp_max_c": q["temp_max_c"], "freq_min_khz": q["freq_min_khz"],
                                "throttled_after": q["throttled_after"]})
        out["long_generation"] = lg_rows
        md.append("### One long generation, UD-Q3_K_M, -nr (measured)\n")
        md.append("| Block | Prompt tok | Generated tok | Decode tok/s | Storage read during DECODE (B) | Major faults in decode | "
                  "ITL first fifth (s) | ITL last fifth (s) | ITL max (s) | Temp max C | Freq min MHz | Throttle flags |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in lg_rows:
            ph = r["phase_decode"] or {}
            md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["block"], r["prompt_n"], r["gen_n"], fmt(r["decode_tok_s_client"], 3), ph.get("read_bytes"), ph.get("majflt"),
                fmt(r["itl_first_fifth_median_s"], 4), fmt(r["itl_last_fifth_median_s"], 4), fmt(r["itl_max_s"], 3),
                fmt(r["temp_max_c"], 1), fmt((r["freq_min_khz"] or 0) / 1000, 0), r["throttled_after"]))
        md.append("")
    else:
        out["missing"].append("longgen")

    # repack vs -nr
    if plans["repack"]:
        rows = []
        for b in plans["repack"]:
            for cls, pred in (("short", lambda q: (q["prompt_n"] or 0) < 100), ("long", lambda q: (q["prompt_n"] or 0) >= 500)):
                allr = [q for q in b["requests"] if pred(q)]
                zr = [q for q in allr if zero(q)]
                rows.append({"block": b["label"], "engine": b.get("engine"), "class": cls, "load_s": b.get("load_s"),
                             "RssAnon_max_kB": (b.get("block_peaks") or {}).get("RssAnon_max_kB"),
                             "MemAvailable_min_kB": (b.get("block_peaks") or {}).get("MemAvailable_min_kB"),
                             "VmSwap_max_kB": (b.get("block_peaks") or {}).get("VmSwap_max_kB"),
                             "n_requests": len(allr), "n_zero_read": len(zr),
                             "read_bytes_per_request": [q["read_bytes"] for q in allr],
                             "zero_read": summarize_requests(zr, ud.get("bytes_per_decoded_token")),
                             "all": summarize_requests(allr, ud.get("bytes_per_decoded_token"))})
        out["repack_vs_nr"] = rows
        md.append("### -nr against default repack, same binary, UD-Q3_K_M, 4 threads (measured; resident = requests with zero reads)\n")
        md.append("| Block | Prompt class | Requests (zero-read) | Storage read per request (GB) | Prefill tok/s (zero-read) | "
                  "Decode tok/s (zero-read) | Anonymous peak (GiB) | MemAvailable min (GiB) | Swap peak (MiB) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            z = r["zero_read"] or {}
            md.append("| %s | %s | %d (%d) | %s | %s | %s | %s | %s | %s |" % (
                r["block"], r["class"], r["n_requests"], r["n_zero_read"],
                ", ".join(fmt((x or 0) / GB, 2) for x in r["read_bytes_per_request"]), fmt(z.get("prefill_tok_s_mean")),
                fmt(z.get("decode_tok_s_mean"), 3), fmt((r["RssAnon_max_kB"] or 0) / 1048576, 2),
                fmt((r["MemAvailable_min_kB"] or 0) / 1048576, 2), fmt((r["VmSwap_max_kB"] or 0) / 1024, 0)))
        md.append("")
    else:
        out["missing"].append("repack")

    # QAT vs UD reference
    if plans["qat"]:
        rows = []
        for name, bpt, lab in (("qat", qat.get("bytes_per_decoded_token"), "QAT Q4_0"), ("udref", ud.get("bytes_per_decoded_token"), "UD-Q3_K_M")):
            for b in plans[name]:
                for cls, pred in (("short", lambda q: (q["prompt_n"] or 0) < 100), ("long", lambda q: (q["prompt_n"] or 0) >= 500)):
                    allr = [q for q in b["requests"] if pred(q)]
                    zr = [q for q in allr if zero(q)]
                    dz = [q for q in allr if decode_zero(q)]
                    rows.append({"file": lab, "block": b["label"], "config": b.get("config"), "class": cls,
                                 "bytes_per_decoded_token": bpt, "n_requests": len(allr), "n_zero_read": len(zr),
                                 "n_decode_zero_read": len(dz), "read_bytes_per_request": [q["read_bytes"] for q in allr],
                                 "zero_read": summarize_requests(zr, bpt), "decode_zero_read": summarize_requests(dz, bpt),
                                 "all": summarize_requests(allr, bpt)})
        out["qat_vs_ud"] = rows
        md.append("### QAT Q4_0 against UD-Q3_K_M in one session, -nr (measured; only requests whose DECODE phase read nothing)\n")
        md.append("| File | Block | Prompt class | Requests (decode zero-read) | Bytes per decoded token (GB) | Decode tok/s | "
                  "Effective GB/s | Prefill tok/s (fully zero-read requests) | Storage read per request (GB) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            dz, z = r["decode_zero_read"] or {}, r["zero_read"] or {}
            md.append("| %s | %s | %s | %d (%d) | %s | %s | %s | %s | %s |" % (
                r["file"], r["block"], r["class"], r["n_requests"], r["n_decode_zero_read"],
                fmt((r["bytes_per_decoded_token"] or 0) / GB, 3), fmt(dz.get("decode_tok_s_mean"), 3), fmt(dz.get("effective_GB_s")),
                fmt(z.get("prefill_tok_s_mean")), ", ".join(fmt((x or 0) / GB, 2) for x in r["read_bytes_per_request"])))
        md.append("")
    else:
        out["missing"].append("qat")

    # ---- time model: per-token decode time predicted from the per-type streaming rates ------------------------------
    if kern and ud.get("by_type"):
        model = {}
        for name, d in (("UD-Q3_K_M", ud), ("QAT Q4_0", qat)):
            if not d.get("by_type"):
                continue
            for t in (1, 2, 3, 4):
                tot_stream = tot_cached = 0.0
                parts, ok = {}, True
                for typ, nbytes in d["by_type"].items():
                    key = "%s@2816" % typ
                    if typ == "F32":
                        rate_s = rate_c = (best.get(t) or {}).get("read_gbps")   # F32 router/norms: assume bandwidth-limited
                    else:
                        c = (kern.get(key) or {}).get(t) or {}
                        rate_s, rate_c = (c.get("stream") or {}).get("gbps_best"), (c.get("cached") or {}).get("gbps_best")
                    if not rate_s or not rate_c:
                        ok = False
                        break
                    parts[typ] = {"bytes": nbytes, "stream_gbps": rate_s, "cached_gbps": rate_c,
                                  "t_stream_s": nbytes / GB / rate_s, "t_cached_s": nbytes / GB / rate_c}
                    tot_stream += nbytes / GB / rate_s
                    tot_cached += nbytes / GB / rate_c
                if ok:
                    model.setdefault(name, {})[t] = {
                        "predicted_s_per_token_stream": round(tot_stream, 4), "predicted_tok_s_stream": round(1 / tot_stream, 3),
                        "compute_only_s_per_token": round(tot_cached, 4), "compute_only_tok_s": round(1 / tot_cached, 3),
                        "memory_share_of_token_time": round(1 - tot_cached / tot_stream, 3), "parts": parts}
        out["time_model"] = {"tag": "derived: sum over quant types of (bytes per decoded token / kernbench rate); "
                             "F32 tensors at the membw read rate; attention over the KV cache, sampling and graph "
                             "overhead are not in the model", "by_file": model}
        md.append("### Decode-time model from the per-type rates (derived) against the measured decode\n")
        md.append("| File | Threads | Predicted tok/s (weights streamed from DRAM) | Predicted tok/s if every weight were in cache "
                  "(arithmetic only) | Share of token time that is memory wait | Measured decode tok/s (short context) |")
        md.append("|---|---|---|---|---|---|")
        for name, d in model.items():
            for t, m in d.items():
                meas = None
                if name == "UD-Q3_K_M":
                    meas = ((out.get("thread_scaling") or {}).get(t) or {}).get("short", {}) or {}
                    meas = meas.get("decode_tok_s_mean")
                md.append("| %s | %d | %s | %s | %s | %s |" % (name, t, fmt(m["predicted_tok_s_stream"], 3),
                                                              fmt(m["compute_only_tok_s"], 3),
                                                              fmt(m["memory_share_of_token_time"], 3), fmt(meas, 3)))
        md.append("")

    with open(os.path.join(base, "pi_analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(base, "pi_tables.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("missing:", out["missing"])
    print("wrote", os.path.join(base, "pi_analysis.json"), "and pi_tables.md")


if __name__ == "__main__":
    main()
