#!/usr/bin/env python3
"""X4 audit: per-request storage reads of the earlier Pi 26B runs (reads run.json files only).

Usage: python pi/adhoc/x4/x4_audit_prior_runs.py [--json OUT]
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GLOBS = [
    "results/pi-real/qat/*/run.json",
    "results/pi-real/series-26b/*/run.json",
    "results/pi-real/p1-resident-ref/*/run.json",
]


def main():
    out = []
    for g in GLOBS:
        for path in sorted(glob.glob(os.path.join(ROOT, g))):
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
            label = os.path.basename(os.path.dirname(path))
            run = {
                "label": label,
                "run_json": os.path.relpath(path, ROOT).replace(os.sep, "/"),
                "started": r.get("started"),
                "command": r.get("command"),
                "threads": r.get("threads"),
                "ctx": r.get("ctx"),
                "n_predict": r.get("n_predict"),
                "extra": r.get("extra"),
                "balloon_gib": r.get("balloon_gib"),
                "load_s": r.get("load_s"),
                "load_read_bytes": r.get("load_read_bytes"),
                "load_majflt": r.get("load_majflt"),
                "mem_available_start_kB": (r.get("meminfo_start_kB") or {}).get("MemAvailable"),
                "after_load_smaps_kB": r.get("after_load_smaps_kB"),
                "throttled": [r.get("throttled_start"), r.get("throttled_end")],
                "temp_c": [r.get("temp_start_c"), r.get("temp_end_c")],
                "requests": [],
            }
            for q in r.get("requests", []):
                sm = q.get("smaps_kB") or {}
                run["requests"].append({
                    "id": q.get("id"),
                    "prompt_n": q.get("prompt_n"),
                    "prompt_tok_s": q.get("prompt_tok_s"),
                    "gen_n": q.get("gen_n"),
                    "gen_tok_s": q.get("gen_tok_s"),
                    "read_bytes": q.get("read_bytes"),
                    "majflt": q.get("majflt"),
                    "pss_anon_kB": sm.get("Pss_Anon"),
                    "pss_file_kB": sm.get("Pss_File"),
                    "temp_c": q.get("temp_c"),
                    "draft_n": q.get("draft_n"),
                    "zero_storage_reads": q.get("read_bytes") == 0 and q.get("majflt") == 0,
                })
            out.append(run)
    for run in out:
        print("== %s  started %s  threads %s ctx %s n_predict %s extra %r balloon %s" % (
            run["label"], run["started"], run["threads"], run["ctx"], run["n_predict"], run["extra"], run["balloon_gib"]))
        print("   cmd: %s" % run["command"])
        print("   load_s %s load_read %s load_majflt %s MemAvailable_start_kB %s after_load %s" % (
            run["load_s"], run["load_read_bytes"], run["load_majflt"], run["mem_available_start_kB"], run["after_load_smaps_kB"]))
        print("   throttled %s temp %s" % (run["throttled"], run["temp_c"]))
        for q in run["requests"]:
            print("   %-22s pn=%4s pp=%6.2f gn=%3s tg=%5.3f read=%12s majflt=%6s anon_kB=%s file_kB=%s temp=%s zero=%s" % (
                q["id"], q["prompt_n"], q["prompt_tok_s"] or 0, q["gen_n"], q["gen_tok_s"] or 0, q["read_bytes"],
                q["majflt"], q["pss_anon_kB"], q["pss_file_kB"], q["temp_c"], q["zero_storage_reads"]))
    if "--json" in sys.argv:
        dst = sys.argv[sys.argv.index("--json") + 1]
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", dst)


if __name__ == "__main__":
    main()
