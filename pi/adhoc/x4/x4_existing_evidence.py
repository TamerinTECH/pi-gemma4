#!/usr/bin/env python3
"""X4 (comparisons workflow): resident-decode bottleneck on the Pi - analysis of EXISTING raw results.

Runs on the desktop, stdlib only, reads files only (no model, no Pi access):
  * results/maps/map26_ud_q3_k_m.json, map26_qat_q4_0_split5.json  -> bytes and weights touched per decoded token
  * results/pi-real/{series-26b,qat,p1-resident-ref}/*/run.json    -> earlier Pi runs (serve_measure.py)
  * results/comparisons/ik_llama_pi/raw/{eval,tune,tune2}/*/block.json + events-*.json (X2 driver)

Every number in the output is tagged: "measured" (taken from a raw Pi result), "computed" (arithmetic on the
GGUF tensor table) or "derived" (measured rate x computed bytes). Nothing here is a new measurement.

Usage:  python pi/adhoc/x4/x4_existing_evidence.py [--out results/pi-real/x4-resident-bottleneck]
"""
import argparse
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# bytes per block / weights per block (ggml type table, a7b94df ggml.c)
TYPE_BLOCK = {
    "F32": (4, 1), "F16": (2, 1), "Q4_0": (18, 32), "Q8_0": (34, 32), "Q6_K": (210, 256),
    "IQ3_XXS": (98, 256), "IQ4_NL": (18, 32), "IQ4_XS": (136, 256),
}


def rel(*p):
    return os.path.join(ROOT, *p)


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def weights_of(nbytes, typ):
    b, w = TYPE_BLOCK[typ]
    return nbytes * w // b


def per_token_ud():
    """UD-Q3_K_M: every decoded token reads all non-expert tensors + exactly 8 experts in each of 30 layers."""
    m = load(rel("results", "maps", "map26_ud_q3_k_m.json"))
    tens = {t["name"]: t for t in m["tensors"]}
    by_kind, by_type, w_by_type = {}, {}, {}

    def add(kind, typ, nbytes):
        by_kind[kind] = by_kind.get(kind, 0) + nbytes
        by_type[typ] = by_type.get(typ, 0) + nbytes
        w_by_type[typ] = w_by_type.get(typ, 0) + weights_of(nbytes, typ)

    for t in m["tensors"]:
        if t["kind"] in ("expert", "expert_scale"):
            continue
        kind = "output_head(token_embd)" if t["kind"] == "token_embd" else t["kind"]
        add(kind, t["type"], t["n_bytes"])
    k = m["n_expert_used"]
    per_layer = []
    for L in m["expert_layers"]:
        gu = tens["blk.%d.ffn_gate_up_exps.weight" % L["layer"]]
        dn = tens["blk.%d.ffn_down_exps.weight" % L["layer"]]
        add("expert_gate_up", gu["type"], k * L["gate_up"]["nb_expert"])
        add("expert_down", dn["type"], k * L["down"]["nb_expert"])
        add("expert_scale", "F32", k * 4)
        per_layer.append({"layer": L["layer"], "gate_up_type": gu["type"], "down_type": dn["type"],
                          "bytes_8_experts": k * (L["gate_up"]["nb_expert"] + L["down"]["nb_expert"] + 4)})
    total = sum(by_kind.values())
    exp = sum(v for kk, v in by_kind.items() if kk.startswith("expert"))
    return {
        "file": m["gguf_basename"], "file_bytes": m["file_size"], "source": "results/maps/map26_ud_q3_k_m.json",
        "tag": "computed",
        "bytes_per_decoded_token": total,
        "expert_bytes_per_token": exp,
        "nonexpert_bytes_per_token": total - exp,
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "weights_per_token_by_type": dict(sorted(w_by_type.items(), key=lambda kv: -kv[1])),
        "weights_per_token": sum(w_by_type.values()),
        "expert_layer_types": sorted({(p["gate_up_type"], p["down_type"]) for p in per_layer}),
        "expert_bytes_per_layer_min_max": [min(p["bytes_8_experts"] for p in per_layer),
                                           max(p["bytes_8_experts"] for p in per_layer)],
        "kv_bytes_per_context_token_f16": m["summary"].get("kv_bytes_per_token_f16"),
        "note": "prefill reads the output head once per request (logits only for the last prompt token) and each "
                "weight row once per micro-batch; decode reads everything listed here for every token",
    }


def per_token_qat():
    q = load(rel("results", "maps", "map26_qat_q4_0_split5.json"))
    s = q["summary"]
    by_kind = {}
    for kk, v in s["nonexpert_bytes_by_kind"].items():
        by_kind["output_head(token_embd)" if kk == "token_embd" else kk] = v
    k = q["n_expert_used"]
    exp = 0
    types = set()
    for L in q["layers"]:
        for role, r in L["roles"].items():
            exp += k * r["nb_expert"]
            if role != "down_scale":
                types.add(r["type"])
    by_kind["expert(gate_up+down+scale)"] = exp
    by_type = dict(s["nonexpert_bytes_by_type"])
    by_type["Q4_0"] = by_type.get("Q4_0", 0) + exp - k * 4 * len(q["layers"])
    by_type["F32"] = by_type.get("F32", 0) + k * 4 * len(q["layers"])
    w_by_type = {t: weights_of(v, t) for t, v in by_type.items()}
    total = sum(by_kind.values())
    return {
        "file": "gemma-4-26B_q4_0-it (5 split parts on the USB stick)", "file_bytes": s["file_bytes_total"],
        "source": "results/maps/map26_qat_q4_0_split5.json", "tag": "computed",
        "bytes_per_decoded_token": total,
        "expert_bytes_per_token": exp,
        "nonexpert_bytes_per_token": total - exp,
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "weights_per_token_by_type": dict(sorted(w_by_type.items(), key=lambda kv: -kv[1])),
        "weights_per_token": sum(w_by_type.values()),
        "expert_types": sorted(types),
    }


def serve_measure_runs():
    """Earlier serve_measure.py runs: one row per request."""
    rows = []
    for g in ("results/pi-real/series-26b/*/run.json", "results/pi-real/qat/*/run.json",
              "results/pi-real/p1-resident-ref/*/run.json"):
        for path in sorted(glob.glob(rel(*g.split("/")))):
            r = load(path)
            label = os.path.basename(os.path.dirname(path))
            for q in r.get("requests", []):
                rows.append({
                    "source": os.path.relpath(path, ROOT).replace(os.sep, "/"), "run": label,
                    "started": r.get("started"), "extra": r.get("extra") or "", "balloon_gib": r.get("balloon_gib"),
                    "threads": r.get("threads"), "id": q["id"], "prompt_n": q.get("prompt_n"),
                    "prompt_tok_s": q.get("prompt_tok_s"), "gen_n": q.get("gen_n"), "gen_tok_s": q.get("gen_tok_s"),
                    "read_bytes": q.get("read_bytes"), "majflt": q.get("majflt"), "temp_c": q.get("temp_c"),
                    "pss_file_kB": (q.get("smaps_kB") or {}).get("Pss_File"),
                    "io_decode_read_bytes": (q.get("io_decode") or {}).get("read_bytes"),
                    "io_prefill_read_bytes": (q.get("io_prefill") or {}).get("read_bytes"),
                })
    return rows


def x2_blocks():
    """X2 driver blocks of OUR build (ours / ours-repack) and ik: one row per non-warm-up request."""
    rows, events = [], []
    for sub in ("eval", "tune", "tune2"):
        for path in sorted(glob.glob(rel("results", "comparisons", "ik_llama_pi", "raw", sub, "*", "block.json"))):
            b = load(path)
            d = os.path.dirname(path)
            if b.get("error") or b.get("aborted") or not b.get("requests"):
                continue
            for q in b["requests"]:
                if q.get("discarded_warmup"):
                    continue
                row = {
                    "source": os.path.relpath(path, ROOT).replace(os.sep, "/"), "block": b["label"],
                    "engine": b["engine"], "config": b.get("config"), "started": b.get("started"),
                    "command": b.get("command"), "id": q["id"], "prompt_n": q.get("prompt_n"),
                    "prompt_tok_s": q.get("prompt_tok_s"), "gen_n": q.get("gen_n"), "gen_tok_s": q.get("gen_tok_s"),
                    "decode_tok_s_client": q.get("decode_tok_s_client"), "ttft_s": q.get("ttft_s"),
                    "itl_median_s": q.get("itl_median_s"), "itl_max_s": q.get("itl_max_s"),
                    "read_bytes": q.get("read_bytes"), "majflt": q.get("majflt"), "cpu_s": q.get("cpu_s"),
                    "total_s": q.get("total_s"), "temp_max_c": (q.get("window") or {}).get("temp_max_c"),
                    "freq_min_khz": (q.get("window") or {}).get("freq_min_khz"),
                    "throttled_after": q.get("throttled_after"),
                    "RssAnon_max_kB": (q.get("window") or {}).get("RssAnon_max_kB"),
                    "MemAvailable_min_kB": (q.get("window") or {}).get("MemAvailable_min_kB"),
                }
                rows.append(row)
                ev_path = os.path.join(d, "events-%s.json" % q["id"])
                if os.path.exists(ev_path):
                    e = load(ev_path)
                    events.append({"block": b["label"], "engine": b["engine"], "config": b.get("config"),
                                   "id": q["id"], "prompt_n": q.get("prompt_n"), "gaps_s": e.get("gaps_s") or []})
    return rows, events


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else None)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=rel("results", "pi-real", "x4-resident-bottleneck"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ud, qat = per_token_ud(), per_token_qat()
    sm = serve_measure_runs()
    x2, ev = x2_blocks()
    GB = 1e9

    # ---- 1. zero-read audit ------------------------------------------------------------------------------
    def audit(rows, pred):
        sel = [r for r in rows if pred(r)]
        z = [r for r in sel if r["read_bytes"] == 0 and r["majflt"] == 0]
        return {"requests": len(sel), "zero_read_requests": len(z),
                "decoded_tokens_total": sum(r["gen_n"] or 0 for r in sel),
                "decoded_tokens_in_zero_read_requests": sum(r["gen_n"] or 0 for r in z),
                "prefill_tokens_in_zero_read_requests": sum(r["prompt_n"] or 0 for r in z),
                "longest_single_generation_with_zero_reads": max([r["gen_n"] or 0 for r in z], default=0),
                "read_bytes_min_max": [min(r["read_bytes"] for r in sel), max(r["read_bytes"] for r in sel)] if sel else None,
                "nonzero_requests": [{"run": r.get("run") or r.get("block"), "id": r["id"], "read_bytes": r["read_bytes"],
                                      "majflt": r["majflt"], "gen_tok_s": r["gen_tok_s"]} for r in sel if r not in z]}

    audit_out = {
        "tag": "measured (process /proc/<pid>/io read_bytes and major faults per request)",
        "qat_q4_0_usb_cold_run_1350": audit(sm, lambda r: r["run"] == "26b-qat-q4_0-usb-cold"),
        "ud_q3_k_m_morning_base_runs": audit(sm, lambda r: r["run"] in ("26b-q3km-cold", "26b-q3km-warm-base-r1",
                                                                         "26b-q3km-warm-base-r2")),
        "ud_q3_k_m_morning_mtp_runs": audit(sm, lambda r: r["run"].startswith("26b-q3km-warm-mtp")),
        "ud_q3_k_m_p1_reference_1854": audit(sm, lambda r: r["run"].startswith("p1-resident-ref")),
        "ud_q3_k_m_x2_ours_nr": audit(x2, lambda r: r["engine"] == "ours"),
        "ud_q3_k_m_x2_ours_default_repack": audit(x2, lambda r: r["engine"] == "ours-repack"),
        "ud_q3_k_m_x2_ik": audit(x2, lambda r: r["engine"] == "ik"),
    }
    all_nr_zero = [r for r in sm if r["run"] in ("26b-q3km-cold", "26b-q3km-warm-base-r1", "26b-q3km-warm-base-r2")
                   and r["read_bytes"] == 0] + [r for r in x2 if r["engine"] == "ours" and r["read_bytes"] == 0]
    audit_out["ud_q3_k_m_nr_no_mtp_zero_read_total"] = {
        "requests": len(all_nr_zero), "decoded_tokens": sum(r["gen_n"] for r in all_nr_zero),
        "prefill_tokens": sum(r["prompt_n"] for r in all_nr_zero),
        "longest_single_generation": max(r["gen_n"] for r in all_nr_zero),
        "note": "warm-up requests of the X2 driver are not counted",
    }

    # ---- 2. effective bandwidth = decode tok/s x bytes per decoded token -----------------------------------
    def eff(rate, bpt):
        return round(rate * bpt / GB, 2)

    obs = []
    morning = [r for r in sm if r["run"] in ("26b-q3km-cold", "26b-q3km-warm-base-r1", "26b-q3km-warm-base-r2")]
    short_m = [r["gen_tok_s"] for r in morning if r["prompt_n"] < 100]
    long_m = [r["gen_tok_s"] for r in morning if r["prompt_n"] > 500]
    obs.append({"file": "UD-Q3_K_M", "engine": "a7b94df -nr -t 4", "where": "morning series 10:46-10:58 (13 requests, all zero-read)",
                "context": "short prompts (46-54 tok)", "n": len(short_m), "decode_tok_s_min": min(short_m),
                "decode_tok_s_max": max(short_m), "decode_tok_s_mean": mean(short_m)})
    obs.append({"file": "UD-Q3_K_M", "engine": "a7b94df -nr -t 4", "where": "morning series, 821-token prompt",
                "context": "821-917 tok", "n": len(long_m), "decode_tok_s_min": min(long_m),
                "decode_tok_s_max": max(long_m), "decode_tok_s_mean": mean(long_m)})
    p1z = [r["gen_tok_s"] for r in sm if r["run"].startswith("p1-resident-ref")]
    obs.append({"file": "UD-Q3_K_M", "engine": "a7b94df -nr -t 4", "where": "P1 reference rerun 18:54 (6 requests, 2 zero-read)",
                "context": "short prompts (39-54 tok)", "n": len(p1z), "decode_tok_s_min": min(p1z),
                "decode_tok_s_max": max(p1z), "decode_tok_s_mean": mean(p1z)})
    for eng, cfg, lab in (("ours", "t4", "a7b94df -nr -t 4"), ("ours", "t3", "a7b94df -nr -t 3"),
                          ("ik", "t4-swac", "ik_llama.cpp 3d27f5bb -t 4 --swa-compress"),
                          ("ik", "t3-swac", "ik_llama.cpp 3d27f5bb -t 3 --swa-compress")):
        for ctx_name, pred in (("short prompts (30-40 tok)", lambda r: r["prompt_n"] < 100),
                               ("long prompt (687-884 tok)", lambda r: r["prompt_n"] > 500)):
            sel = [r["decode_tok_s_client"] for r in x2 if r["engine"] == eng and r["config"] == cfg and pred(r)
                   and r["read_bytes"] == 0]
            if sel:
                obs.append({"file": "UD-Q3_K_M", "engine": lab, "where": "X2 raw (eval + tune), client-side rate, zero-read",
                            "context": ctx_name, "n": len(sel), "decode_tok_s_min": min(sel),
                            "decode_tok_s_max": max(sel), "decode_tok_s_mean": mean(sel)})
    for o in obs:
        for k in ("min", "max", "mean"):
            o["effective_GB_s_" + k] = eff(o["decode_tok_s_" + k], ud["bytes_per_decoded_token"])
    qrows = [r for r in sm if r["run"] == "26b-qat-q4_0-usb-cold"]
    qobs = []
    for r in qrows:
        qobs.append({"file": "QAT Q4_0", "engine": "a7b94df -nr -t 4", "id": r["id"], "prompt_n": r["prompt_n"],
                     "decode_tok_s": r["gen_tok_s"], "prompt_tok_s": r["prompt_tok_s"],
                     "read_bytes_in_request": r["read_bytes"], "majflt": r["majflt"],
                     "file_pages_resident_GB": round(r["pss_file_kB"] * 1024 / GB, 2),
                     "effective_GB_s": eff(r["gen_tok_s"], qat["bytes_per_decoded_token"]),
                     "resident": False})

    # ---- 3. thread scaling that already exists (X2 tuning: 3 vs 4 threads) ----------------------------------
    def cell(eng, cfg, pid):
        sel = [r for r in x2 if r["engine"] == eng and r["config"] == cfg and r["id"] == pid and "tune" in r["block"]]
        return sel
    thr = []
    for eng, c4, c3 in (("ours", "t4", "t3"), ("ik", "t4-swac", "t3-swac"), ("ik", "t4", "t3")):
        for pid in ("tun-01", "tun-06"):
            r4, r3 = cell(eng, c4, pid), cell(eng, c3, pid)
            if not r4 or not r3:
                continue
            d4, d3 = mean([r["decode_tok_s_client"] for r in r4]), mean([r["decode_tok_s_client"] for r in r3])
            p4, p3 = mean([r["prompt_tok_s"] for r in r4]), mean([r["prompt_tok_s"] for r in r3])
            thr.append({"engine": eng, "configs": [c4, c3], "prompt": pid, "prompt_n": r4[0]["prompt_n"],
                        "decode_tok_s_4t": d4, "decode_tok_s_3t": d3, "decode_3t_over_4t": round(d3 / d4, 3),
                        "prefill_tok_s_4t": p4, "prefill_tok_s_3t": p3, "prefill_3t_over_4t": round(p3 / p4, 3),
                        "n_blocks_4t": len(r4), "n_blocks_3t": len(r3),
                        "cpu_s_4t": mean([r["cpu_s"] for r in r4]), "cpu_s_3t": mean([r["cpu_s"] for r in r3]),
                        "total_s_4t": mean([r["total_s"] for r in r4]), "total_s_3t": mean([r["total_s"] for r in r3])})

    # ---- 4. inter-token latency vs context length (our build, 4 threads, X2 eval + tune) ---------------------
    xs, ys = [], []
    per_req = []
    for e in ev:
        if e["engine"] != "ours" or e["config"] != "t4":
            continue
        g = e["gaps_s"]
        if not g:
            continue
        for i, gap in enumerate(g):
            xs.append(e["prompt_n"] + i + 1)
            ys.append(gap)
        per_req.append({"block": e["block"], "id": e["id"], "prompt_n": e["prompt_n"], "n_gaps": len(g),
                        "itl_mean_s": round(sum(g) / len(g), 4), "itl_median_s": round(statistics.median(g), 4)})
    fit = None
    if len(xs) > 10:
        a0, b0, r2 = linfit(xs, ys)
        fit = {"tag": "derived (least squares over %d inter-token gaps of our -nr -t 4 blocks)" % len(xs),
               "itl_at_zero_context_s": round(a0, 4), "slope_s_per_context_token": round(b0, 7),
               "slope_ms_per_1000_context_tokens": round(b0 * 1e6, 1), "r2": round(r2, 3),
               "context_range_tokens": [min(xs), max(xs)],
               "kv_bytes_per_context_token_f16": ud["kv_bytes_per_context_token_f16"],
               "implied_extra_bytes_per_decoded_token_at_1000_ctx": ud["kv_bytes_per_context_token_f16"] * 1000,
               "implied_decode_tok_s_at_context": {str(n): round(1.0 / (a0 + b0 * n), 3) for n in (0, 100, 500, 1000, 2000, 4000)},
               "note": "tok/s at contexts above the measured range is an extrapolation, not a measurement"}
    xs_i, ys_i = [], []
    for e in ev:
        if e["engine"] == "ik" and e["config"] == "t4-swac" and e["gaps_s"]:
            for i, gap in enumerate(e["gaps_s"]):
                xs_i.append(e["prompt_n"] + i + 1)
                ys_i.append(gap)
    fit_ik = None
    if len(xs_i) > 10:
        a1, b1, r21 = linfit(xs_i, ys_i)
        fit_ik = {"itl_at_zero_context_s": round(a1, 4), "slope_ms_per_1000_context_tokens": round(b1 * 1e6, 1),
                  "r2": round(r21, 3), "n_gaps": len(xs_i)}

    # ---- 5. thermal: decode rate vs temperature for identical requests ---------------------------------------
    therm = []
    for pid in sorted({r["id"] for r in x2 if r["engine"] == "ours" and r["config"] == "t4"}):
        sel = [r for r in x2 if r["engine"] == "ours" and r["config"] == "t4" and r["id"] == pid]
        therm.append({"prompt": pid, "n": len(sel),
                      "temp_max_c": [r["temp_max_c"] for r in sel],
                      "decode_tok_s_client": [r["decode_tok_s_client"] for r in sel],
                      "freq_min_khz": [r["freq_min_khz"] for r in sel]})
    temps_all = [r["temp_max_c"] for r in x2 if r["temp_max_c"] is not None]
    freq_all = [r["freq_min_khz"] for r in x2 if r["freq_min_khz"] is not None]
    thermal = {"tag": "measured (X2 raw: 1 Hz samples of thermal_zone0, scaling_cur_freq, vcgencmd get_throttled)",
               "per_prompt_ours_t4": therm, "temp_max_over_all_x2_requests_c": max(temps_all),
               "freq_min_over_all_x2_requests_khz": min(freq_all),
               "throttled_values": sorted({r["throttled_after"] for r in x2 if r["throttled_after"]}),
               "morning_series_throttled": "0x0 at the start and end of every run; temp 46-74 C",
               "note": "0x80000 = sticky 'soft temperature limit has occurred' bit; no live bit (0x1-0x8) in any sample"}

    # ---- 6. cross-file check: speed ratio vs byte ratio vs prefill ratio -------------------------------------
    ud_short = mean(short_m)
    qat_best = max(q["decode_tok_s"] for q in qobs)
    qat_long = [q for q in qobs if q["prompt_n"] > 500][0]
    ud_long_pp = mean([r["prompt_tok_s"] for r in morning if r["prompt_n"] > 500])
    cross = {
        "bytes_per_token_ratio_ud_over_qat": round(ud["bytes_per_decoded_token"] / qat["bytes_per_decoded_token"], 3),
        "weights_per_token_ud": ud["weights_per_token"], "weights_per_token_qat": qat["weights_per_token"],
        "decode_ratio_qat_over_ud_measured": [round(qat_best / max(short_m), 3), round(qat_best / min(short_m), 3)],
        "qat_decode_used": {"tok_s": qat_best, "caveat": "least-disturbed QAT request (de_short); it still read 50.3 MB "
                            "from the USB stick (354 major faults) - an upper bound on the damage is 0.72 s of 15.7 s"},
        "prefill_ratio_qat_over_ud_measured_821_tok_prompt": round(qat_long["prompt_tok_s"] / ud_long_pp, 3),
        "prefill_caveat": "the QAT 821-token request read 0.27 GB from the stick during the request",
        "prediction_if_bandwidth_bound": "decode ratio = byte ratio",
        "prediction_if_kernel_bound": "decode ratio = prefill (compute) ratio",
    }

    out = {
        "stage": "X4-resident-bottleneck-pi", "kind": "analysis of existing raw results (no new Pi measurement)",
        "per_token_bytes": {"ud_q3_k_m": ud, "qat_q4_0": qat},
        "zero_read_audit": audit_out,
        "effective_bandwidth": {"tag": "derived = measured decode tok/s x computed bytes per decoded token (decimal GB/s); "
                                "KV-cache and activation traffic not included",
                                "ud_q3_k_m": obs, "qat_q4_0": qobs},
        "thread_scaling_existing": {"tag": "measured (X2 tuning blocks, one block per config, resident, zero reads)",
                                    "rows": thr, "missing": "1 and 2 threads were never run on the Pi"},
        "itl_vs_context": {"ours_t4": fit, "ik_t4_swa_compress": fit_ik, "per_request": per_req},
        "thermal": thermal,
        "cross_file": cross,
        "repack_existing": [r for r in x2 if r["engine"] == "ours-repack"],
    }
    dst = os.path.join(a.out, "existing_evidence.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    # ---- console summary -------------------------------------------------------------------------------------
    def gb(x):
        return "%.3f GB" % (x / GB)
    for name, d in (("UD-Q3_K_M", ud), ("QAT Q4_0", qat)):
        print("== %s: %s per decoded token (experts %s, non-expert %s); weights/token %.3fe9" % (
            name, gb(d["bytes_per_decoded_token"]), gb(d["expert_bytes_per_token"]), gb(d["nonexpert_bytes_per_token"]),
            d["weights_per_token"] / 1e9))
        for kk, v in d["by_kind"].items():
            print("     kind %-28s %14d  %5.1f %%" % (kk, v, 100.0 * v / d["bytes_per_decoded_token"]))
        for kk, v in d["by_type"].items():
            print("     type %-28s %14d  %5.1f %%   weights %.1fe6" % (kk, v, 100.0 * v / d["bytes_per_decoded_token"],
                                                                       d["weights_per_token_by_type"][kk] / 1e6))
    print("== zero-read audit")
    for kk, v in audit_out.items():
        print("   ", kk, {x: y for x, y in v.items() if x not in ("nonzero_requests", "tag")} if isinstance(v, dict) else v)
    print("== effective bandwidth (UD-Q3_K_M)")
    for o in obs:
        print("    %-44s %-28s n=%2d decode %.3f-%.3f (mean %.3f) tok/s -> %.2f-%.2f (mean %.2f) GB/s" % (
            o["engine"], o["context"], o["n"], o["decode_tok_s_min"], o["decode_tok_s_max"], o["decode_tok_s_mean"],
            o["effective_GB_s_min"], o["effective_GB_s_max"], o["effective_GB_s_mean"]))
    print("== effective bandwidth (QAT Q4_0, none resident)")
    for q in qobs:
        print("    %-14s pn=%4d decode %.3f tok/s pp %.2f read %.3f GB majflt %5d file-resident %.2f GB -> %.2f GB/s" % (
            q["id"], q["prompt_n"], q["decode_tok_s"], q["prompt_tok_s"], q["read_bytes_in_request"] / GB, q["majflt"],
            q["file_pages_resident_GB"], q["effective_GB_s"]))
    print("== thread scaling (existing)")
    for t in thr:
        print("    %-5s %-18s %-7s pn=%4d decode 4t %.3f 3t %.3f (x%.3f)  prefill 4t %.2f 3t %.2f (x%.3f) cpu_s 4t %.1f 3t %.1f" % (
            t["engine"], "/".join(t["configs"]), t["prompt"], t["prompt_n"], t["decode_tok_s_4t"], t["decode_tok_s_3t"],
            t["decode_3t_over_4t"], t["prefill_tok_s_4t"], t["prefill_tok_s_3t"], t["prefill_3t_over_4t"],
            t["cpu_s_4t"], t["cpu_s_3t"]))
    print("== ITL vs context:", json.dumps(fit))
    print("   ik:", json.dumps(fit_ik))
    print("== thermal:", json.dumps({k: v for k, v in thermal.items() if k != "per_prompt_ours_t4"}))
    for t in therm:
        print("   ", t)
    print("== cross-file:", json.dumps(cross, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
