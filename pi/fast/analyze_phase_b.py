#!/usr/bin/env python3
"""Tables for the phase-B runs (pi/fast/run_phase_b.sh): matched four configurations, ablation, drafter
vocabulary shortcut, model screen. Reads results/pi-real/fast3/<step>/<run>/ab.json, prints text tables and
writes one JSON summary.

Usage: analyze_phase_b.py [--root results/pi-real/fast3] [--out results/pi-real/fast3/summary.json]
"""
import argparse
import collections
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, ".")


def load(path):
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    run = j["runs"][0]
    run["_label"] = j.get("label")
    return run


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def rng(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    m = statistics.median(xs)
    return f"{m:.{nd}f}" if len(xs) == 1 else f"{m:.{nd}f} ({min(xs):.{nd}f}-{max(xs):.{nd}f})"


def agg(run):
    reqs = run.get("requests") or []
    if not reqs:
        return {"error": run.get("error")}
    gen = sum(r["gen_n"] or 0 for r in reqs)
    gen_s = sum((r.get("gen_ms") or 0) for r in reqs) / 1000.0 or sum((r["gen_n"] or 0) / r["gen_tok_s"] for r in reqs if r.get("gen_tok_s"))
    prm = sum(r["prompt_n"] or 0 for r in reqs)
    prm_s = sum((r.get("prompt_ms") or 0) for r in reqs) / 1000.0
    acc = sum(r.get("draft_n_accepted") or 0 for r in reqs)
    drafted = sum(r.get("draft_n") or 0 for r in reqs)
    passes = gen - acc
    st = run.get("draft_stats") or []
    d_calls, d_ms = sum(x["calls"] for x in st), sum(x["draft_ms"] for x in st)
    return {
        "requests": len(reqs), "gen_tokens": gen, "prompt_tokens": prm,
        "total_wall_s": round(sum(r["wall_s"] for r in reqs), 2),
        "median_wall_s": round(med([r["wall_s"] for r in reqs]), 2),
        "median_ttft_visible_s": med([r.get("ttft_visible_s") for r in reqs]),
        "decode_tok_s": round(gen / gen_s, 2) if gen_s else None,
        "prompt_tok_s": round(prm / prm_s, 1) if prm_s else None,
        "tokens_per_cycle": round(gen / passes, 3) if passes else None,
        "ms_per_cycle": round(1000 * gen_s / passes, 1) if passes else None,
        "accepted": acc, "drafted": drafted, "acceptance": round(acc / drafted, 3) if drafted else None,
        "draft_ms_per_cycle": round(d_ms / d_calls, 2) if d_calls else None,
        "draft_calls": d_calls,
        "peak_rss_mb": round((run.get("mem_kb") or {}).get("VmHWM", 0) / 1024, 0) or None,
        "rss_anon_mb": round((run.get("mem_kb") or {}).get("RssAnon", 0) / 1024, 0) or None,
        "finish": dict(collections.Counter(str(r.get("finish")) for r in reqs)),
        "reasoning_chars": sum(len(r.get("reasoning") or "") for r in reqs),
        "max_temp_c": max((r.get("temp_c") or 0) for r in reqs),
        "load_s": run.get("load_s"),
    }


def same_text(run, base):
    b = {r["id"]: r["text_sha256"] for r in base.get("requests") or []}
    reqs = run.get("requests") or []
    diff = [r["id"] for r in reqs if r["id"] in b and r["text_sha256"] != b[r["id"]]]
    return len(reqs) - len(diff), len(reqs), diff


def grade(run):
    from pigemma import quality_v2 as q
    items = q.load_qa_v2()
    items = items["items"] if isinstance(items, dict) else items
    byid = {it["id"]: it for it in items}
    ok, tot, trunc, rows = collections.Counter(), collections.Counter(), collections.Counter(), []
    for r in run.get("requests") or []:
        it = byid.get(r["id"])
        if not it:
            continue
        vis, _, _ = q.visible_answer(r.get("content"), r.get("reasoning"))
        try:
            g = q.grade_item(it, vis)
            passed = bool(g.get("pass"))
        except Exception:  # noqa: BLE001
            passed = False
        cat = it["category"]
        for key in (("all", "all"), ("lang", it["lang"]), ("cat", cat)):
            tot[key] += 1
            ok[key] += 1 if passed else 0
            trunc[key] += 1 if r.get("finish") == "length" else 0
        rows.append({"id": r["id"], "pass": passed, "finish": r.get("finish"), "gen_n": r.get("gen_n"), "wall_s": r.get("wall_s")})
    out = {"correct": ok[("all", "all")], "total": tot[("all", "all")], "truncated": trunc[("all", "all")],
           "by_lang": {k[1]: f"{ok[k]}/{tot[k]}" for k in tot if k[0] == "lang"},
           "by_category": {k[1]: f"{ok[k]}/{tot[k]} (cut {trunc[k]})" for k in tot if k[0] == "cat"},
           "truncated_and_wrong": sum(1 for x in rows if x["finish"] == "length" and not x["pass"]), "rows": rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/pi-real/fast3")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    runs = {}
    for path in sorted(glob.glob(os.path.join(a.root, "*", "*", "ab.json"))):
        step, name = path.replace("\\", "/").split("/")[-3:-1]
        runs[(step, name)] = load(path)
    summary = {}

    # ---- matched ---------------------------------------------------------------------------------------------
    m = {k[1]: v for k, v in runs.items() if k[0] == "matched"}
    if m:
        print("== matched four configurations (E2B, --reasoning off, 9 prompts, repetitions alternated)")
        cfgs = ["ref_plain", "fast_plain", "ref_mtp", "fast_mtp"]
        base = m.get("r1_ref_plain", {})
        table = {}
        for c in cfgs:
            reps = [agg(m[n]) for n in sorted(m) if n.endswith("_" + c)]
            ident = [same_text(m[n], base) for n in sorted(m) if n.endswith("_" + c)]
            if not reps:
                continue
            table[c] = {
                "reps": len(reps),
                "total_wall_s": rng([r.get("total_wall_s") for r in reps]),
                "median_ttft_visible_s": rng([r.get("median_ttft_visible_s") for r in reps], 3),
                "decode_tok_s": rng([r.get("decode_tok_s") for r in reps]),
                "prompt_tok_s": rng([r.get("prompt_tok_s") for r in reps], 1),
                "tokens_per_cycle": rng([r.get("tokens_per_cycle") for r in reps]),
                "ms_per_cycle": rng([r.get("ms_per_cycle") for r in reps], 1),
                "peak_rss_mb": rng([r.get("peak_rss_mb") for r in reps], 0),
                "same_text_as_ref_plain": f"{sum(i[0] for i in ident)}/{sum(i[1] for i in ident)}",
                "raw": reps,
            }
            t = table[c]
            print(f"  {c:11s} total {t['total_wall_s']} s | first visible {t['median_ttft_visible_s']} s | decode {t['decode_tok_s']} tok/s | "
                  f"prompt {t['prompt_tok_s']} tok/s | tok/cycle {t['tokens_per_cycle']} | ms/cycle {t['ms_per_cycle']} | peak RSS {t['peak_rss_mb']} MB | "
                  f"same text {t['same_text_as_ref_plain']}")
        # paired per request: medians over repetitions
        ids = [r["id"] for r in base.get("requests") or []]
        paired = []
        for pid in ids:
            row = {"id": pid}
            for c in cfgs:
                ws, ts, pn = [], [], None
                for n in sorted(m):
                    if n.endswith("_" + c):
                        for r in m[n].get("requests") or []:
                            if r["id"] == pid:
                                ws.append(r["wall_s"]); ts.append(r.get("ttft_visible_s")); pn = r.get("prompt_n")
                row[c] = {"wall_s": med(ws), "wall_range": [min(ws), max(ws)] if ws else None, "ttft_s": med(ts)}
                row["prompt_n"] = pn
            paired.append(row)
        print("  paired per request, median seconds over repetitions: complete answer (first visible text)")
        regress = []
        for row in paired:
            cells = "  ".join(f"{c}: {row[c]['wall_s']:.2f} ({row[c]['ttft_s'] if row[c]['ttft_s'] is not None else float('nan'):.2f})" for c in cfgs if row[c]["wall_s"] is not None)
            print(f"    {row['id']:13s} prompt {row['prompt_n']:>4} tok | {cells}")
            for ref, fast in (("ref_plain", "fast_plain"), ("ref_mtp", "fast_mtp")):
                if row[ref]["wall_s"] and row[fast]["wall_s"] and row[fast]["wall_s"] > 1.02 * row[ref]["wall_s"]:
                    regress.append((row["id"], ref, row[ref]["wall_s"], fast, row[fast]["wall_s"]))
        print("  regressions (patched slower than its reference by more than 2 % on a request):", regress or "none")
        rp = [x.get("rendered_prompt") for x in m.values() if x.get("rendered_prompt")]
        print("  rendered prompt for 'Hello' (--reasoning off):", repr(rp[0][:160]) if rp else "not recorded", "| identical in all runs:", len(set(rp)) <= 1)
        summary["matched"] = {"table": table, "paired": paired, "regressions": regress, "rendered_prompt": rp[0] if rp else None}

    # ---- ablation --------------------------------------------------------------------------------------------
    ab = {k[1]: v for k, v in runs.items() if k[0] == "ablation"}
    if ab:
        print("== ablation (patched binary, one switch at a time; 3 long + 3 short prompts)")
        summary["ablation"] = {}
        for fam in ("plain", "mtp3", "mtp2"):
            base = ab.get(fam + "_none")
            for n in sorted(ab):
                if not n.startswith(fam + "_"):
                    continue
                s = agg(ab[n])
                ident = same_text(ab[n], base) if base else (None, None, None)
                summary["ablation"][n] = dict(s, same_text=f"{ident[0]}/{ident[1]}")
                print(f"  {n:12s} total {s.get('total_wall_s')} s | first visible {s.get('median_ttft_visible_s')} s | decode {s.get('decode_tok_s')} | "
                      f"prompt {s.get('prompt_tok_s')} tok/s | tok/cycle {s.get('tokens_per_cycle')} | ms/cycle {s.get('ms_per_cycle')} | same text {ident[0]}/{ident[1]}")

    # ---- drafter vocabulary shortcut -------------------------------------------------------------------------
    ce = {k[1]: v for k, v in runs.items() if k[0] == "centroids"}
    if ce:
        print("== drafter vocabulary shortcut (E2B fast profile; dense head vs cluster head)")
        summary["centroids"] = {}
        base = ce.get("r1_dense")
        for n in sorted(ce):
            s = agg(ce[n])
            ident = same_text(ce[n], base) if base else (None, None, None)
            bylang = {}
            for lang in ("en", "he", "de", "code"):
                sub = dict(ce[n], requests=[r for r in ce[n].get("requests") or [] if r["lang"] == lang])
                x = agg(sub)
                bylang[lang] = {k: x.get(k) for k in ("requests", "decode_tok_s", "tokens_per_cycle", "ms_per_cycle", "acceptance", "total_wall_s")}
            summary["centroids"][n] = dict(s, same_text=f"{ident[0]}/{ident[1]}", by_lang=bylang, diff=ident[2])
            print(f"  {n:10s} total {s.get('total_wall_s')} s | decode {s.get('decode_tok_s')} tok/s | tok/cycle {s.get('tokens_per_cycle')} | ms/cycle {s.get('ms_per_cycle')} | "
                  f"draft ms/cycle {s.get('draft_ms_per_cycle')} | accepted {s.get('accepted')}/{s.get('drafted')} = {s.get('acceptance')} | same text {ident[0]}/{ident[1]}")
            print("             " + " | ".join(f"{lang}: {v['decode_tok_s']} tok/s, {v['tokens_per_cycle']} tok/cycle, acc {v['acceptance']}" for lang, v in bylang.items()))

    # ---- model screen ----------------------------------------------------------------------------------------
    for step, title in (("screen", "model screen, direct answers (--reasoning off), task budgets"), ("think", "thinking enabled (separately labelled)")):
        sc = {k[1]: v for k, v in runs.items() if k[0] == step}
        if not sc:
            continue
        print(f"== {title}: 44 held-out questions")
        summary[step] = {}
        for n in sorted(sc):
            s = agg(sc[n])
            g = grade(sc[n])
            ws = sorted(r["wall_s"] for r in sc[n].get("requests") or [])
            p90 = ws[int(0.9 * len(ws))] if ws else None
            summary[step][n] = dict(s, quality={k: v for k, v in g.items() if k != "rows"}, p90_wall_s=p90, rows=g["rows"])
            print(f"  {n:15s} correct {g['correct']}/{g['total']} (cut off {g['truncated']}, cut off and wrong {g['truncated_and_wrong']}) | total {s.get('total_wall_s')} s | "
                  f"median {s.get('median_wall_s')} s, p90 {p90} s | first visible {s.get('median_ttft_visible_s')} s | decode {s.get('decode_tok_s')} tok/s | "
                  f"prompt {s.get('prompt_tok_s')} tok/s | tok/cycle {s.get('tokens_per_cycle')} | peak RSS {s.get('peak_rss_mb')} MB | load {s.get('load_s')} s | {s.get('max_temp_c')} C")
            print(f"                  by language {g['by_lang']} | by category {g['by_category']}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
