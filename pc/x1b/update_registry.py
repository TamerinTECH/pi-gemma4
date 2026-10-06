#!/usr/bin/env python3
"""X1b: update results/comparisons/registry.json for the target hotpin-llm (text surgery like pc/x1a).

Only the target hotpin-llm (status, status_reason, runnable_for_exact_gguf, + engine_ids, local_test) and one
appended entry of the top-level 'updates' list are touched; every other line stays byte-identical. The
edited text must parse to exactly the intended content or nothing is written. Numbers are read from the X1b
result files.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P = REPO / "results" / "comparisons" / "registry.json"
X1B = REPO / "results" / "comparisons" / "x1b"
NL = chr(10)


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def case(run: str, cid: str) -> dict:
    return next(c for c in load(X1B / "lock_verify" / run / "lock_verify.json")["cases"] if c["case_id"] == cid)


def main() -> int:
    m0 = os.path.getmtime(P)
    raw = open(P, encoding="utf-8", newline="").read()
    if chr(13) in raw:
        print("REFUSED: registry has CR characters", file=sys.stderr)
        return 4
    d = json.loads(raw)
    t = next(x for x in d["targets"] if x["id"] == "hotpin-llm")
    if t["status"] != "documentation-only":
        print(f"REFUSED: target status is already {t['status']!r}", file=sys.stderr)
        return 4
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    bp = load(REPO / "third_party/llama.cpp-a6647b1/build-linux-x86/BUILD_INFO.json")
    bh = load(REPO / "third_party/hotpin-llm-build-a6647b1/build-linux-x86/BUILD_INFO.json")
    ba = load(REPO / "third_party/hotpin-llm-build-a6647b1/build-linux-aarch64-cross/BUILD_INFO.json")
    an = load(X1B / "routing/routing_original_analysis.json")
    meta = load(X1B / "routing/routing_original_meta.json")
    c_hp, c_rp = case("main", "hotpin-nr-original2gb"), case("main", "hotpin-defaultrepack-original2gb")
    c_lim = case("memlock64m", "hotpin-nr-original2gb-memlock64m")
    c_arm = case("aarch64-emu", "hotpin-aarch64-emu-nr-original2gb")
    mt = an["multi_token_check"]
    t["status"] = "locally tested"
    t["status_reason"] = ("X1b (2026-10-05/06): parent a6647b1 and a6647b1 + HotPin patches built for Linux x86-64 in Docker "
                          "with the pinned-build flags; both load our exact UD-Q3_K_M with -nr and the harness argv; HotPin's "
                          "own profiling + ranking run on calibration.jsonl only; lock accounting verified; output "
                          "bit-identical to its parent. Desktop functional/accounting checks only: no controlled-memory "
                          "cell, no speed result, nothing on the real Pi.")
    t["runnable_for_exact_gguf"] = "yes (measured in X1b, desktop x86 Docker; engine ids hotpin-parent and hotpin)"
    t["engine_ids"] = ["hotpin-parent", "hotpin"]
    t["local_test"] = {
        "stage": "X1b-hotpin",
        "date": now,
        "report": "results/comparisons/x1b/README.md",
        "builds": {
            "parent": {"tree": "third_party/llama.cpp-a6647b1 (unmodified shallow clone)", "seconds_at_j4": bp["build_seconds"],
                       "llama_server_sha256": bp["binaries"]["llama-server"]["sha256"]},
            "hotpin": {"tree": "Docker volume pigemma-x1b-hotpin-a6647b1 (/hp/src) = parent + hotpin_l1_pin.diff + dump_routing_prefetch.diff via git apply",
                       "seconds_at_j4": bh["build_seconds"],
                       "llama_server_sha256": bh["binaries"]["llama-server"]["sha256"],
                       "output": "third_party/hotpin-llm-build-a6647b1 (binaries + notices; AGPL combined work)"},
            "flags": " ".join(bp["cmake_configure_flags"]),
            "harness_engine_check": "ok for both (no flag override needed): results/comparisons/x1b/engine-check-hotpin*.json",
        },
        "profiling_on_calibration": {
            "tool": "HotPin's patched llama-simple-chat --dump-routing (unmodified) + tools/make_hot_file.py (unmodified)",
            "requests": meta["totals"]["requests"], "prompt_token_evaluations": meta["totals"]["prompt_tokens"],
            "generated_tokens": meta["totals"]["gen_tokens"],
            "single_token_evaluations": an["records"]["single_token_tokens"],
            "routing_sha256": meta["routing_sha256"],
            "wall_seconds_incl_4_model_loads": 530,
            "defect_measured": (f"{mt['first_16_records_are_a_permutation_of_all_experts']}/{mt['n_ge_16_checked']} prefill "
                                "layer-evaluations are a permutation of all 128 experts: the dump ignores the stride of "
                                "the ffn_moe_topk view, prompt tokens add a uniform count "
                                f"({an['counts']['multi_token_only']['min']}-{an['counts']['multi_token_only']['max']} per expert), "
                                "and the ranking is in effect built from generated tokens only"),
            "pin_sets": "results/comparisons/x1b/hotfiles/ (original 1-6 GB tool budgets; budget-matched; stride-fixed local adaptation)",
            "budget_semantics_measured": "tool budget N GB locks 0.81-0.82 x N GiB of our file (slice estimate 3.16 MiB vs 2,632,960 B real)",
        },
        "lock_verification": {
            "with_nr_original_2gb": {
                "tool_log_requested_mib": c_hp["server_log"]["tool_pinned_mib_requested"],
                "failed_lock_requests": c_hp["server_log"]["failed_mlock_warnings"],
                "VmLck_kB": c_hp["status_after_load"]["VmLck_kB"],
                "unique_locked_pages_4k": c_hp["locked_vs_predicted"]["actual_locked_model_pages"],
                "exact_match_with_prediction": c_hp["locked_vs_predicted"]["exact_match"],
                "alignment_overlock_pages": c_hp["predicted"]["alignment_overlock_pages"],
                "locked_anonymous_kB": c_hp["smaps_after_load"]["locked_non_model"]["size_kB"]},
            "default_repack_original_2gb": {
                "cpu_repack_buffer_mib": c_rp["server_log"]["buffers_mib"].get("CPU_REPACK model"),
                "locked_gguf_pages": c_rp["locked_vs_predicted"]["actual_locked_model_pages"],
                "locked_anonymous_kB": c_rp["smaps_after_load"]["locked_non_model"]["anonymous_kB"],
                "conclusion": "HotPin must run with -nr; its README launch lines do not pass it"},
            "memlock_64_mib": {
                "tool_log_requested_mib": c_lim["server_log"]["tool_pinned_mib_requested"],
                "failed_lock_requests": c_lim["server_log"]["failed_mlock_warnings"],
                "VmLck_kB": c_lim["status_after_load"]["VmLck_kB"]},
            "lossless_vs_parent": "token ids, chosen logprob and top-5 identical at 64 positions (2 tuning prompts x 32 tokens)",
            "raw": "results/comparisons/x1b/lock_verify/",
        },
        "cross_revision": "a6647b1 vs a7b94df are not numerically identical on the same prompts (tun-01 same ids, logprob diff up to 0.137; tun-02 diverges at token 8): results/comparisons/x1b/cross_revision_a6647b1_vs_a7b94df.json",
        "arm": {
            "aarch64_build": f"builds (cross {ba['compiler']}, -march=armv8.2-a+dotprod+fp16, {ba['build_seconds']} s at -j 4)",
            "aarch64_emulated_lock_check": {
                "where": "qemu-user, Debian 13 arm64 container, 4 KiB pages, load only",
                "failed_lock_requests": c_arm["server_log"]["failed_mlock_warnings"],
                "VmLck_kB": c_arm["status_after_load"]["VmLck_kB"],
                "exact_match_with_prediction": c_arm["locked_vs_predicted"]["exact_match"]},
            "real_pi": "not tested; not a drop-in Pi solution (memlock limit 8192 kB by default, needs -nr, 16 KiB-page sizes only predicted, prefetch layer is Windows-only, no memory cgroup on the live Pi)",
        },
        "not_done": ["controlled-memory cell (no P5 line in the progress log when the stage ended; run instructions in the report)",
                     "any speed measurement", "real-Pi run"],
    }
    upd = {"stage": "X1b-hotpin", "at": now,
           "what": "hotpin-llm: documentation-only -> locally tested (engines hotpin-parent + hotpin runnable for the exact GGUF; "
                   "HotPin's own ranking on calibration.jsonl; lock accounting and repack behaviour measured; aarch64 builds; "
                   "no controlled cell yet)"}
    d["updates"].append(upd)

    lines = raw.split(NL)

    def js(v) -> str:
        return json.dumps(v, ensure_ascii=False)

    def dump(v, indent: int) -> str:
        return json.dumps(v, indent=2, ensure_ascii=False).replace(NL, NL + " " * indent)

    def find(prefix: str, lo: int = 0, hi: int | None = None) -> int:
        hi = len(lines) if hi is None else hi
        hits = [i for i in range(lo, hi) if lines[i].startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit(f"expected exactly one line starting with {prefix!r} in [{lo},{hi}), got {len(hits)}")
        return hits[0]

    b0 = find('      "id": "hotpin-llm"')
    b1 = find('      "id": "ik_llama.cpp"')
    i = find('      "runnable_for_exact_gguf": ', b0, b1)
    lines[i] = '      "runnable_for_exact_gguf": ' + js(t["runnable_for_exact_gguf"]) + ","
    i = find('      "status": ', b0, b1)
    lines[i] = '      "status": ' + js(t["status"]) + ","
    i = find('      "status_reason": ', b0, b1)
    if not lines[i].rstrip().endswith(","):
        raise SystemExit("status_reason is the last key of the target - adapt the surgery")
    lines[i] = '      "status_reason": ' + js(t["status_reason"]) + ","
    lines.insert(i + 1, '      "engine_ids": ' + js(t["engine_ids"]) + ",")
    lines.insert(i + 2, '      "local_test": ' + dump(t["local_test"], 6) + ",")
    text = NL.join(lines)
    tail = NL + "  ]" + NL + "}" + NL
    if not text.endswith(tail):
        raise SystemExit("unexpected file ending (updates list not last)")
    text = text[:-len(tail)] + "," + NL + "    " + dump(upd, 4) + tail
    back = json.loads(text)
    if back != d:
        raise SystemExit("text surgery does not reproduce the intended registry content - nothing written")
    if os.path.getmtime(P) != m0:
        print("REFUSED: registry.json changed while updating - rerun", file=sys.stderr)
        return 4
    tmp = str(P) + ".x1b.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    os.replace(tmp, P)
    print("updated", P, os.path.getsize(P), "bytes; target status =", t["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
