#!/usr/bin/env python3
"""X2: build results/comparisons/ik_llama_pi/X2_RESULT.json (machine-readable stage result for X4/X6).

  python pi/adhoc/x2/x2_make_result.py results/comparisons/ik_llama_pi

Reads raw/eval/summary.json, raw/tune2/tuning_summary.json, raw/quality/quality_summary.json (if present),
the block.json files and build/BUILD_INFO.txt. Every number is MEASURED on the real Pi unless a field
name says projected. Vendor numbers are not included here.
"""
import json
import os
import statistics
import sys


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def rng(xs):
    xs = [x for x in xs if x is not None]
    return [round(min(xs), 4), round(max(xs), 4)] if xs else None


def main(root):
    ev = jl(os.path.join(root, "raw", "eval", "summary.json"))
    tu = jl(os.path.join(root, "raw", "tune2", "tuning_summary.json"))
    qp = os.path.join(root, "raw", "quality", "quality_summary.json")
    qual = jl(qp) if os.path.isfile(qp) else None
    plan = jl(os.path.join(root, "driver", "x2_plan_eval.json"))
    env = open(os.path.join(root, "raw", "env_info.txt"), encoding="utf-8").read().splitlines()
    table = []
    for pid in ev["prompts"]:
        for e in ("ours", "ik"):
            s = ev["per_prompt"][pid][e]
            table.append({
                "prompt": pid, "engine": e, "runs": len(s["prompt_tok_s"]),
                "prompt_tokens": s["prompt_n"][0], "generated_tokens": sorted(set(s["gen_n"])),
                "prefill_tok_s_server_mean": mean(s["prompt_tok_s"]), "prefill_tok_s_server_range": rng(s["prompt_tok_s"]),
                "decode_tok_s_client_mean": mean(s["decode_tok_s_client"]), "decode_tok_s_client_range": rng(s["decode_tok_s_client"]),
                "decode_tok_s_server_mean": mean(s["gen_tok_s"]),
                "ttft_s_mean": mean(s["ttft_s"]), "ttft_s_range": rng(s["ttft_s"]),
                "total_latency_s_mean": mean(s["total_s"]), "total_latency_s_range": rng(s["total_s"]),
                "itl_median_s_mean": mean(s["itl_median_s"]), "itl_p90_s_mean": mean(s["itl_p90_s"]),
                "itl_p99_s_mean": mean(s["itl_p99_s"]), "itl_max_s_max": max(s["itl_max_s"]),
                "stalls_gt_1s": sum(s["stalls_gt_1s"]), "stalls_gt_2x_median": sum(s["stalls_gt_2x_median"]),
                "process_read_bytes_per_request_max": max(s["read_bytes"]), "majflt_max": max(s["majflt"]),
                "cv_percent": {k: (round(v * 100, 2) if v is not None else None) for k, v in s["cv"].items()}})
    ratios = {pid: {k: {"mean": mean(v), "range": rng(v)} for k, v in r.items()} for pid, r in ev["ratios"].items()}
    mem = {}
    for lab, m in ev["memory"].items():
        pk, al = m["peaks"], m["after_load"]
        mem[lab] = {"engine": m["engine"], "load_s": m["load_s"],
                    "RssAnon_after_load_MiB": round(al["status_kB"]["RssAnon"] / 1024), "RssAnon_peak_MiB": round(pk["RssAnon_max_kB"] / 1024),
                    "RssFile_peak_MiB": round(pk["RssFile_max_kB"] / 1024), "VmHWM_MiB": round(pk["VmHWM_kB"] / 1024),
                    "VmSwap_peak_MiB": round(pk["VmSwap_max_kB"] / 1024), "MemAvailable_min_MiB": round(pk["MemAvailable_min_kB"] / 1024),
                    "model_page_cache_residency_before_block": m["pre_residency"]["fraction"],
                    "temp_start_c": ev["thermal"][lab]["temp_start_c"], "temp_max_c": pk["temp_max_c"],
                    "cpu_freq_min_khz": pk["freq_min_khz"], "throttled_values_seen": pk["throttled_values"]}
    res = {
        "schema": "x2-result/1", "stage": "X2-ik-llama-on-pi", "track": "runtime comparison, SAME model file, resident (no memory pressure); feeds tracks A and B, proves nothing about track C",
        "all_numbers": "MEASURED on the real Pi 5 16 GB (context-friend-1) on 2026-10-05 22:30-23:59 CEST unless a key says 'projected'",
        "environment": env,
        "model": {"file": "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf", "bytes": 12728497888,
                  "sha256": "811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052",
                  "source": "unsloth (non-QAT), tensor types: IQ3_XXS x29, IQ4_NL x29, IQ4_XS x1, Q8_0 x206, Q6_K x1, F32 x392 (ik load log)",
                  "storage": "microSD ext4 (/dev/mmcblk0p2); fully in the page cache (mincore 100 %) before every block"},
        "engines": {
            "ours": {"repo": "ggml-org/llama.cpp", "commit": "a7b94df2c616bc1f62a73b964b4a71cb0dcc488e",
                     "binary_sha256": "0386501ee355c3390a2a76b0d9c47f5bbcf1694d366ffdeda467be8d9862e0cb",
                     "build": "native on the Pi 2026-10-05 08:51: Release, GGML_NATIVE=ON -> -mcpu=cortex-a76+crc+crypto+dotprod+noi8mm+nosve+nosme, OpenMP, static libs",
                     "frozen_args": plan["frozen"]["ours"]},
            "ik": {"repo": "ikawrakow/ik_llama.cpp", "commit": "3d27f5bb6348c4d79edded63b03e9a93efb2f80e", "license": "MIT",
                   "binary_sha256": "72b5f8b3e1705363471793672d48295eb0fbe01ebc7c7fee73c403e5e609721c",
                   "build": "native on the Pi 2026-10-05 22:31-22:45 (833 s, -j 3, nice 10): Release, GGML_NATIVE=ON -> -O3 -march=native "
                            "(gcc 14.2 resolves it to -mcpu=cortex-a76+crc+crypto on this board) -flax-vector-conversions -fopenmp, static libs; "
                            "source and objects in /tmp (tmpfs) because / had only 2.145 GB free; only llama-server + llama-bench installed to "
                            "~/pigemma/ik_llama.cpp/build/bin; / free afterwards 2,079,219,712 B",
                   "frozen_args": plan["frozen"]["ik"],
                   "not_used": "-rtr (run-time repack), -ser, --prefetch-experts, -khad, MTP"}},
        "request": plan["common_request"],
        "prompt_format": "pre-rendered GGUF Gemma 4 chat template, thinking off (empty thought block); identical bytes to POST /completion of both "
                         "engines; prompt token ids identical across engines for every prompt (checked through each engine's /tokenize)",
        "load_and_coherence": {
            "ik_loads_exact_gguf": True, "ik_load_s_resident": "3.5-3.9", "ours_load_s_resident": "4.0-4.1",
            "ik_weights_stay_mmapped": "yes: RssFile 12,136 MiB, RssAnon after load 530 MiB with --swa-compress (1,030 MiB without); no repack",
            "ik_output_coherent": "yes in EN/DE/HE on tuning + held-out prompts; see output_comparison and quality_parity for the caveats"},
        "tuning": {"rule": jl(os.path.join(root, "driver", "x2_plan_tune2.json"))["purpose"], "rows": tu["rows"], "choice": tu["choice"]},
        "heldout_evaluation": {"design": plan["purpose"], "block_order": [o[0] for o in ev["order"]], "per_prompt": table,
                               "paired_ratio_ik_over_ours": ratios, "itl_pooled": ev["itl_pooled"], "per_block_memory_thermal": mem,
                               "tokenization": ev["tokenization"], "output_comparison": ev["outputs"]},
        "projected": {"prefill_1000_tokens_s": {"ours": round(1000 / mean(ev["per_prompt"]["ho-13"]["ours"]["prompt_tok_s"]), 1),
                                                "ik": round(1000 / mean(ev["per_prompt"]["ho-13"]["ik"]["prompt_tok_s"]), 1),
                                                "basis": "linear projection from the measured 884-token prompt; not measured"}},
        "quality_parity": qual and {k: ({kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v)
                                    for k, v in {**qual["blocks"], "disagreements": qual["disagreements"],
                                                 "fail_in_every_block_that_ran_it": qual["fail_in_every_block_that_ran_it"],
                                                 "identical_output_ours_vs_ik_frozen": qual["identical_output_ours_vs_ik_frozen"]}.items()},
    }
    notes_p = os.path.join(root, "driver", "x2_notes.json")
    if os.path.isfile(notes_p):
        res.update(jl(notes_p))
    with open(os.path.join(root, "X2_RESULT.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("written", os.path.join(root, "X2_RESULT.json"))


if __name__ == "__main__":
    main(sys.argv[1])
