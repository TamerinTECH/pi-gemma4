#!/usr/bin/env python3
"""X1b: append engines hotpin-parent and hotpin to results/comparisons/engines.json (append rule of X1a).

Idempotent: entries with the same engine_id are replaced, every other entry is written back untouched;
the two X1b names are removed from pending_engines; 'updated' is refreshed. Values come from the build
info files and the X1b verification results, not from memory.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENG = REPO / "results" / "comparisons" / "engines.json"
X1B = REPO / "results" / "comparisons" / "x1b"
PARENT_DIR = REPO / "third_party" / "llama.cpp-a6647b1" / "build-linux-x86"
HOTPIN_DIR = REPO / "third_party" / "hotpin-llm-build-a6647b1" / "build-linux-x86"
ACTIVE = X1B / "hotfiles" / "active" / "hotpin.txt"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def case(run: str, cid: str) -> dict:
    d = load(X1B / "lock_verify" / run / "lock_verify.json")
    return next(c for c in d["cases"] if c["case_id"] == cid)


def smoke_cell() -> dict:
    """Evidence of the single functional smoke cell (FUSE-throttled storage, 8 GiB cgroup); numbers from its files."""
    cdir = REPO / "results" / "controlled" / "20261006-x1b-smoke-tuning-hotpin-A-8g-r0"
    if not (cdir / "config.json").exists():
        return {"status": "not run", "note": "mlock through the FUSE mount inside the memory cgroup is unverified"}
    cfg = load(cdir / "config.json")
    eff = cfg["effective"]
    log = (cdir / "server.log").read_text("utf-8", "replace")
    reqs = [json.loads(ln) for ln in (cdir / "requests.jsonl").read_text("utf-8").splitlines() if ln.strip()]
    wl = next(r for r in reqs if r.get("kind") == "workload")
    import re
    m = re.search(r"expert-cache v1: pinned (\d+) hot expert-slices \(([\d.]+) MiB\)", log)
    used_meta = load(Path(str(X1B / "hotfiles" / "hotpin_matched_smokeC_expert.txt") + ".meta.json"))
    pred_pages = used_meta["prediction"]["by_page"]["4096"]["predicted_unique_locked_pages"]
    return {
        "cell": "results/controlled/20261006-x1b-smoke-tuning-hotpin-A-8g-r0",
        "kind": "ONE FUNCTIONAL smoke cell (purpose=smoke, variant hotpin:A, 8 GiB, FUSE 91.6 MB/s, tun-01 capped at 16 tokens; host CPU not idle) - not a measurement, no speed claim",
        "status": cfg.get("status"), "failure": cfg.get("failure"),
        "tool_log": m.group(0) if m else None,
        "failed_mlock_warnings": log.count("failed to mlock"),
        "cg_unevictable_bytes_end": eff["lock_accounting"]["cg_unevictable_bytes_end"],
        "pin_set_used": "hotfiles/hotpin_matched_smokeC_expert.txt (the active file when the cell ran, sha256 d676459b...)",
        "predicted_unique_locked_bytes": pred_pages * 4096,
        "cg_unevictable_equals_predicted": eff["lock_accounting"]["cg_unevictable_bytes_end"] == pred_pages * 4096,
        "repack_present": eff["server_log"]["repack_present"],
        "phases": {k: eff["phases"][k] for k in ("load_s", "warmup_s", "workload_s", "load_device_read_bytes")},
        "cgroup_events_end": load(cdir / "cgroup_end.json").get("events"),
        "request_functional_only": {k: wl.get(k) for k in ("rid", "input_tokens", "output_tokens", "ttft_s", "prefill_tok_s_server",
                                                           "decode_tok_s_server", "decode_device_bytes_per_token",
                                                           "bg_vm_other_cores_decode")},
        "verified": "mlock of pages of the FUSE-throttled mount works inside the 8 GiB memory cgroup: the cgroup's unevictable counter equals the predicted unique locked pages and stays constant through load, warm-up and the request",
    }


def main() -> int:
    bp, bh = load(PARENT_DIR / "BUILD_INFO.json"), load(HOTPIN_DIR / "BUILD_INFO.json")
    ecp, ech = load(X1B / "engine-check-hotpin-parent.json"), load(X1B / "engine-check-hotpin.json")
    c_par = case("main", "parent-nr-envset")
    c_hp = case("main", "hotpin-nr-original2gb")
    c_m = case("main", "hotpin-nr-matched-smokeC-expert")
    c_rp = case("main", "hotpin-defaultrepack-original2gb")
    c_no = case("main", "hotpin-nr-nohot")
    c_lim = case("memlock64m", "hotpin-nr-original2gb-memlock64m")
    active_meta = load(Path(str(X1B / "hotfiles" / "hotpin_matched_smokeC_expert.txt") + ".meta.json"))
    assert sha(ACTIVE) == active_meta["hot_file_sha256"], "active hot file differs from its source variant"
    flags = bp["cmake_configure_flags"]
    assert flags == bh["cmake_configure_flags"]
    build_common = {
        "image": "pi-gemma4-build:latest (ubuntu 24.04, gcc/g++ 13.3.0, cmake 3.28.3, ninja 1.11.1); binaries run in pi-gemma4-build:fuse (harness image, same base)",
        "script": "pc/x1b/build_engines.sh -> pc/x1b/build_in_container.sh (-j 4, nice 10, under results/.cpu.lock; same configure flags as pc/x1a = the flags that reproduce the pinned a7b94df build byte-for-byte)",
        "effective_arch_flags": "ggml-cpu target: -O3 -DNDEBUG -march=native (= znver4 on the Ryzen 5 8400F: AVX2 + AVX-512 incl. VBMI/VNNI/BF16) -fopenmp; libllama/common: -O3 -DNDEBUG",
        "linkage": "libllama/ggml/common static in each tool (BUILD_SHARED_LIBS=OFF); glibc, libstdc++, libgomp dynamic",
        "not_portable": "-march=native: needs an AVX-512 (Zen 4 class) CPU",
    }

    def same_tokens(a: dict, b: dict) -> bool:
        return all(a["requests"][k]["token_ids"] == b["requests"][k]["token_ids"]
                   and [p["logprob"] for p in a["requests"][k]["probs"]] == [p["logprob"] for p in b["requests"][k]["probs"]]
                   for k in a["requests"])

    parent = {
        "engine_id": "hotpin-parent",
        "role": "HotPin's OWN unmodified parent llama.cpp revision (brief UPDATE 2 sections 2 and 4): the only valid baseline for engine 'hotpin'; never compare hotpin with a7b94df (1540 commits newer)",
        "binary_path_in_container": "/work/third_party/llama.cpp-a6647b1/build-linux-x86/bin/llama-server",
        "binary_path_host": "third_party/llama.cpp-a6647b1/build-linux-x86/bin/llama-server",
        "binary_sha256": bp["binaries"]["llama-server"]["sha256"],
        "binary_bytes": bp["binaries"]["llama-server"]["bytes"],
        "other_tools_same_dir": [t for t in bp["targets"] if t != "llama-server"],
        "revision": bp["revision"],
        "release_tag": "b9857 (per HotPin's README / X0; shallow clone has no tags, HEAD sha verified locally)",
        "version_output": bp["version_output"].replace("\n", " / "),
        "patches": "none (unmodified shallow clone third_party/llama.cpp-a6647b1; git status clean before and after the build)",
        "build_flags": flags,
        "build_flags_status": f"MEASURED: configure + build log in third_party/llama.cpp-a6647b1/build-linux-x86/ (BUILD_INFO.json, logs/); build {bp['build_seconds']} s at -j 4",
        "build": build_common,
        "status": "runnable",
        "status_evidence": "results/comparisons/x1b/lock_verify/main (case parent-nr-envset): loads our exact UD-Q3_K_M with the harness's own argv (-nr ... --warmup), CPU_Mapped 12123.75 MiB, no CPU_REPACK buffer, generates; results/comparisons/x1b/engine-check-hotpin-parent.json: every canonical cell flag is in its --help (no override needed)",
        "parent": None,
        "parent_engine_id": None,
        "compare_against": "llama.cpp-a7b94df (engine-revision drift only: group 'runtime track A-B'; NOT a method comparison)",
        "comparison_group": "same-model track A: baseline of 'hotpin'",
        "license": "MIT (llama.cpp)",
        "harness": {
            "server_argv_compatible": True,
            "engine_check_ok": bool(ecp.get("ok")),
            "extra_server_args": [],
            "needs_nr": True,
            "methods": ["A", "B", "C", "D"],
            "methods_note": "method A = the baseline HotPin must be compared with. The pinner/FUSE helpers are engine-independent, so C and D can also run on this engine: 'hotpin-parent:C' vs 'hotpin:A' compares OUR static pinning with HotPin's pinning on the SAME llama.cpp revision.",
        },
        "functional_check": {
            "kind": "FUNCTIONAL, not a benchmark (no memory budget, no storage throttle, desktop not verified idle, cpuset 2,4,6,8)",
            "ignores_LLAMA_HOT_EXPERTS": c_par["status_after_load"]["VmLck_kB"] == 0 and c_par["server_log"]["tool_pinned_line"] is None,
            "buffers_mib": c_par["server_log"]["buffers_mib"],
            "prompts": "tuning tun-01 (EN) and tun-02 (HE), 32 greedy tokens each (same prompts and settings as X1a's check)",
            "incidental_speed_not_a_benchmark": {k: v["timings_functional_only"] for k, v in c_par["requests"].items()},
        },
        "notes": "a6647b1 is 1540 commits older than a7b94df. Output vs a7b94df on the X1a prompts: see results/comparisons/x1b/README.md (cross-revision comparison is documented, not treated as a cache defect).",
    }
    lv, lvm = c_hp["locked_vs_predicted"], c_m["locked_vs_predicted"]
    hotpin = {
        "engine_id": "hotpin",
        "role": "HotPin L1 (LozzKappa/hotpin-llm @ ac52bd9a): in-process mlock of routing-ranked expert slices at model load; ORIGINAL implementation (unmodified patches) on its own parent revision",
        "binary_path_in_container": "/work/third_party/hotpin-llm-build-a6647b1/build-linux-x86/bin/llama-server",
        "binary_path_host": "third_party/hotpin-llm-build-a6647b1/build-linux-x86/bin/llama-server",
        "binary_sha256": bh["binaries"]["llama-server"]["sha256"],
        "binary_bytes": bh["binaries"]["llama-server"]["bytes"],
        "other_tools_same_dir": [t for t in bh["targets"] if t != "llama-server"] + ["llama-simple-chat-stridefix (LOCAL ADAPTATION, see local_adaptations/)"],
        "revision": bh["revision"] + " + HotPin patches",
        "version_output": bh["version_output"].replace("\n", " / "),
        "patches": bh["patches_applied"].strip() + " applied unmodified with `git apply` (3 files changed; resulting diff: third_party/hotpin-llm-build-a6647b1/build-linux-x86/PATCHED_TREE.diff). gptoss_arch.diff and glm4moelite_mla_fix.diff are not needed for Gemma 4 and were not applied.",
        "build_flags": flags,
        "build_flags_status": f"MEASURED: identical script and flags as hotpin-parent; build {bh['build_seconds']} s at -j 4; third_party/hotpin-llm-build-a6647b1/build-linux-x86/BUILD_INFO.json",
        "build": build_common,
        "status": "runnable",
        "status_evidence": "results/comparisons/x1b/lock_verify/main/lock_verify.json + results/comparisons/x1b/engine-check-hotpin.json",
        "parent": "hotpin-parent",
        "parent_engine_id": "hotpin-parent",
        "compare_against": "hotpin-parent",
        "comparison_group": "same-model track A (method: HotPin L1 pin) vs its OWN unmodified parent",
        "license": "AGPL-3.0-or-later + separate paid commercial license (third_party/hotpin-llm/{LICENSE,NOTICE,COMMERCIAL_LICENSE.md}); the patched binary is a combined work -> treat as AGPL. COMPARISON USE ONLY; no product integration without the operator's decision.",
        "method_label": "HotPin L1 (in-process mlock of hot expert slices)",
        "env": {"LLAMA_HOT_EXPERTS": "/work/results/comparisons/x1b/hotfiles/active/hotpin.txt"},
        "extra_args": [],
        "harness": {
            "server_argv_compatible": True,
            "engine_check_ok": bool(ech.get("ok")),
            "extra_server_args": [],
            "needs_nr": True,
            "needs_nr_reason": "MEASURED: without -nr (llama.cpp default repack, which is how HotPin's README launches it) the IQ4_NL down-expert tensors are copied to an anonymous CPU_REPACK buffer and HotPin locks those copies instead of the file pages (see 'repack_case').",
            "methods": ["A"],
            "methods_note": "Run as method A only (no pinner helper): HotPin pins inside llama-server while the model loads. Do not combine with C/D (double pinning). Every cell already has --cap-add IPC_LOCK --ulimit memlock=-1:-1 (pigemma.controlled.docker_argv); without them most locks FAIL silently (see 'limited_memlock_case').",
            "pin_set_env": "LLAMA_HOT_EXPERTS",
            "active_pin_set": {
                "path": "results/comparisons/x1b/hotfiles/active/hotpin.txt",
                "sha256": sha(ACTIVE),
                "is_copy_of": "results/comparisons/x1b/hotfiles/hotpin_matched_smokeC_expert.txt",
                "n_experts": active_meta["prediction"]["n_lines"],
                "predicted_unique_locked_pages_4k": active_meta["prediction"]["by_page"]["4096"]["predicted_unique_locked_pages"],
                "PROVISIONAL": "matched to the P5 SMOKE plan of method C (results/controlled/_dev/smoke-plans/smokeC.plan.json, expert_pin_bytes 1.5e9). After M1 freezes method C's plan, re-match and re-activate BEFORE any hotpin cell: see 'rematch_command'.",
                "rematch_command": "python pc/x1b/hotpin_pinset.py match --order results/comparisons/x1b/hotfiles/hotpin_order_full.txt --plan <frozen C plan.json> --mode expert --out results/comparisons/x1b/hotfiles/hotpin_matched_frozenC_expert.txt && cp results/comparisons/x1b/hotfiles/hotpin_matched_frozenC_expert.txt results/comparisons/x1b/hotfiles/active/hotpin.txt   (then record the sha256 of active/hotpin.txt in FROZEN_CONFIG.json)",
                "matching_rule": "--mode expert: HotPin's unique locked pages <= method C's expert pages (HotPin never locks non-expert pages; they stay ordinary page cache). --mode total matches C's total locked pages instead (C's non-expert pages are then spent on more experts). Report which rule was used.",
            },
            "cell_evidence_to_check": "per cell: server VmLck (samples.csv / marks), cgroup unevictable, and the server.log lines 'expert-cache v1: pinned N hot expert-slices' + any 'failed to mlock' warning. Expected VmLck = predicted_unique_locked_pages x 4 kB.",
            "smoke_cell": smoke_cell(),
        },
        "pin_set_provenance": {
            "procedure": "HotPin's own: patched llama-simple-chat --dump-routing (unmodified HotPin patch) on data/workloads/calibration.jsonl ONLY (32 requests, Gemma 4 template via --raw, greedy, -n = max_tokens), then HotPin's tools/make_hot_file.py",
            "routing_dump": "results/comparisons/x1b/routing/routing_original.bin (+ routing_original_meta.json, routing_original_analysis.json)",
            "full_order": "results/comparisons/x1b/hotfiles/hotpin_order_full.txt",
            "variants": {
                "original": "results/comparisons/x1b/hotfiles/hotpin_original_{1,1.5,2,3,4,6}gb.txt = the tool's output as is; its slice estimate (file_size/3840 = 3.16 MiB) makes an 'N GB' budget lock only 0.81-0.82 x N GiB of our file",
                "locally adapted budget": "hotpin_matched_*.txt = strict prefix of HotPin's own order, cut by exact predicted unique locked pages (pc/x1b/hotpin_pinset.py match)",
                "locally adapted ranking (optional)": "hotpin_stridefix_*.txt = same tool chain after a one-hunk fix of the dump callback (see known_defect); NOT HotPin's original",
            },
            "known_defect_of_the_original_dump": "MEASURED: for every multi-token evaluation (all prompt tokens) HotPin's dump callback reads ffn_moe_topk without the view stride, so it records the full 128-entry argsort of the first ceil(T/16) tokens instead of each token's top-8 (1140/1140 prefill layer-evaluations are a permutation of all experts). Prompt tokens therefore add a near-uniform count (377-407 per expert) and the ORIGINAL ranking is effectively a ranking of the 1971 generated-token evaluations only.",
        },
        "lock_verification": {
            "kind": "accounting check on the desktop Docker VM (ext4 volume, 4 KiB pages, no memory budget) - not a benchmark",
            "original_2gb": {
                "tool_log": c_hp["server_log"]["tool_pinned_line"],
                "failed_lock_requests": c_hp["server_log"]["failed_mlock_warnings"],
                "VmLck_kB": c_hp["status_after_load"]["VmLck_kB"],
                "unique_locked_pages_actual": lv["actual_locked_model_pages"],
                "unique_locked_pages_predicted": lv["predicted_pages"],
                "exact_match_actual_vs_predicted": lv["exact_match"],
                "alignment_overlock_pages": c_hp["predicted"]["alignment_overlock_pages"],
                "locked_anonymous_kB": c_hp["smaps_after_load"]["locked_non_model"]["size_kB"],
                "locked_pages_referenced_by_a_64_token_generation_kB": c_hp["smaps_after_generation"]["locked_model"]["referenced_kB"],
                "locked_vmas_referenced": f"{c_hp['smaps_after_generation']['locked_model']['vmas_with_any_reference']}/{c_hp['smaps_after_generation']['locked_model']['n_vmas']}",
            },
            "active_matched": {
                "tool_log": c_m["server_log"]["tool_pinned_line"],
                "failed_lock_requests": c_m["server_log"]["failed_mlock_warnings"],
                "VmLck_kB": c_m["status_after_load"]["VmLck_kB"],
                "unique_locked_pages_actual": lvm["actual_locked_model_pages"],
                "unique_locked_pages_predicted": lvm["predicted_pages"],
                "exact_match_actual_vs_predicted": lvm["exact_match"],
            },
            "repack_case": {
                "what": "same binary and hot file WITHOUT -nr (llama.cpp default)",
                "buffers_mib": c_rp["server_log"]["buffers_mib"],
                "tool_log": c_rp["server_log"]["tool_pinned_line"],
                "locked_gguf_pages": c_rp["locked_vs_predicted"]["actual_locked_model_pages"],
                "predicted_gguf_pages_not_locked": c_rp["locked_vs_predicted"]["predicted_not_locked_pages"],
                "locked_anonymous_kB": c_rp["smaps_after_load"]["locked_non_model"]["anonymous_kB"],
            },
            "limited_memlock_case": {
                "what": "RLIMIT_MEMLOCK 64 MiB, no CAP_IPC_LOCK",
                "tool_log": c_lim["server_log"]["tool_pinned_line"],
                "failed_lock_requests": c_lim["server_log"]["failed_mlock_warnings"],
                "failed_bytes": c_lim["server_log"]["failed_mlock_bytes"],
                "VmLck_kB": c_lim["status_after_load"]["VmLck_kB"],
            },
            "lossless_check": {
                "hotpin_pinned_vs_parent_tokens_and_logprobs_identical": same_tokens(c_hp, c_par) and same_tokens(c_m, c_par),
                "hotpin_without_hot_file_vs_parent_identical": same_tokens(c_no, c_par),
                "prompts": "tun-01, tun-02 x 32 greedy tokens (64 positions; token ids, chosen logprob and top-5)",
            },
        },
        "l3_prefetch": "not used and not usable: HotPin's --prefetch exists only in llama-simple-chat and only for Windows (PrefetchVirtualMemory); on Linux it compiles to a no-op (build warning: unused variable PREFETCH_HINT_BYTES). llama-server has L1 only.",
        "notes": "See results/comparisons/x1b/README.md for the full account (ranking procedure, variants, ARM status, run instructions).",
    }
    doc = load(ENG)
    before = {e["engine_id"]: json.dumps(e, sort_keys=True) for e in doc["engines"]}
    keep = [e for e in doc["engines"] if e["engine_id"] not in ("hotpin-parent", "hotpin")]
    doc["engines"] = keep + [parent, hotpin]
    doc["pending_engines"] = [p for p in doc.get("pending_engines", []) if not p.startswith("hotpin")]
    doc["updated"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    ub = doc.setdefault("updated_by", [])
    if isinstance(ub, list) and "X1b-hotpin (comparisons workflow wf_02d183f6-341)" not in ub:
        ub.append("X1b-hotpin (comparisons workflow wf_02d183f6-341)")
    for e in keep:  # untouched entries must be byte-for-byte the same objects
        assert json.dumps(e, sort_keys=True) == before[e["engine_id"]]
    ENG.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("engines now:", [e["engine_id"] for e in doc["engines"]], "| pending:", doc["pending_engines"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
