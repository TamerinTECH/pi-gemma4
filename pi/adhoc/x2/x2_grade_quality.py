#!/usr/bin/env python3
"""X2: grade the quality-parity diagnostic (desktop; uses pigemma.quality_v2's objective graders).

  python pi/adhoc/x2/x2_grade_quality.py results/comparisons/ik_llama_pi/raw/quality

Reads <dir>/<block>/block.json written by x2_compare.py (raw /completion outputs of the 34 non-code
qa_heldout_v2 items), grades every output with pigemma.quality_v2.grade_item on the visible answer and
writes <dir>/quality_summary.json. Prints per-block totals, per language / category, and every item
where the blocks disagree. Timings of these blocks are NOT performance results.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pigemma import quality_v2 as Q  # noqa: E402


def main(d):
    sys.stdout.reconfigure(encoding="utf-8")
    items = {i["id"]: i for i in Q.load_qa_v2()}
    blocks = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name, "block.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                blocks.append(json.load(f))
    out = {"blocks": {}, "disagreements": []}
    table = {}
    for b in blocks:
        rows = []
        for r in b.get("requests", []):
            if r.get("discarded_warmup") or r["id"] not in items:
                continue
            it = items[r["id"]]
            vis, think, notes = Q.visible_answer(r.get("output"))
            g = Q.grade_item(it, vis)
            rows.append({"id": r["id"], "lang": it["lang"], "category": it["category"], "pass": bool(g["pass"]),
                         "score": g["score"], "reason": g["reason"], "visible": vis, "cleanup": notes, "thinking_chars": think,
                         "gen_n": r.get("gen_n"), "stop": {k: v for k, v in (r.get("stop_fields") or {}).items() if v},
                         "prompt_n": r.get("prompt_n")})
            table.setdefault(r["id"], {})[b["label"]] = rows[-1]

        def agg(key):
            c = collections.defaultdict(lambda: [0, 0])
            for x in rows:
                c[x[key]][1] += 1
                c[x[key]][0] += int(x["pass"])
            return {k: f"{v[0]}/{v[1]}" for k, v in sorted(c.items())}
        pk = b.get("block_peaks") or {}
        out["blocks"][b["label"]] = {"engine": b["engine"], "config": b.get("config"), "command": b.get("command"),
                                     "pass": sum(x["pass"] for x in rows), "total": len(rows),
                                     "mean_score": round(sum(float(x["score"] or 0) for x in rows) / len(rows), 4) if rows else None,
                                     "by_lang": agg("lang"), "by_category": agg("category"),
                                     "n_markup_cleanup": sum(1 for x in rows if x["cleanup"]),
                                     "n_hit_limit": sum(1 for x in rows if "limit" in json.dumps(x["stop"])),
                                     "temp_max_c": pk.get("temp_max_c"), "throttled_values": pk.get("throttled_values"),
                                     "freq_min_khz": pk.get("freq_min_khz"), "error": b.get("error"), "rows": rows}
        o = out["blocks"][b["label"]]
        print(f"{b['label']}: {o['pass']}/{o['total']} pass, mean score {o['mean_score']}, by_lang {o['by_lang']}, "
              f"by_category {o['by_category']}, temp max {o['temp_max_c']}, throttled {o['throttled_values']}, error {o['error']}")
    labels = [b["label"] for b in blocks]
    print()
    for iid, per in table.items():
        ps = {lab: per[lab]["pass"] for lab in labels if lab in per}
        texts = {lab: per[lab]["visible"] for lab in labels if lab in per}
        if len(set(ps.values())) > 1:
            out["disagreements"].append({"id": iid, "pass": ps, "visible": texts,
                                         "reason": {lab: per[lab]["reason"] for lab in per}})
            print(f"DISAGREE {iid}: {ps}")
            for lab in per:
                print(f"   [{lab}] {per[lab]['reason'][:110]} | {per[lab]['visible'][:260]!r}")
    both_fail = [iid for iid, per in table.items() if per and not any(x["pass"] for x in per.values())]
    out["fail_in_every_block_that_ran_it"] = both_fail
    out["identical_output_ours_vs_ik_frozen"] = None
    if len(labels) >= 2:
        a, b_ = labels[0], labels[1]
        same = [iid for iid, per in table.items() if a in per and b_ in per and per[a]["visible"] == per[b_]["visible"]]
        both = [iid for iid, per in table.items() if a in per and b_ in per]
        out["identical_output_ours_vs_ik_frozen"] = {"identical": len(same), "of": len(both), "ids": same}
        print(f"\nidentical visible output {a} vs {b_}: {len(same)}/{len(both)}")
    print("fail in every block:", both_fail)
    with open(os.path.join(d, "quality_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
