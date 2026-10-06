#!/usr/bin/env python3
"""X1a: update results/comparisons/registry.json for the upstream llama.cpp target.

Only the target llama.cpp-upstream-b11430, our_reference.engine (one added key) and a new top-level
'updates' list are touched. The file is edited as TEXT so that X0's hand formatting of every other
entry stays byte-identical; the edited text must parse to exactly the intended content or nothing is
written.
"""
from __future__ import annotations

import datetime
import json
import os
import sys

P = "results/comparisons/registry.json"
NL = chr(10)
FLAGS = ("-G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=ON "
         "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF "
         "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")


def main() -> int:
    m0 = os.path.getmtime(P)
    with open(P, encoding="utf-8", newline="") as fh:
        raw = fh.read()
    d = json.loads(raw)
    if "updates" in d:
        print("REFUSED: registry already has an 'updates' list - extend this script to append to it", file=sys.stderr)
        return 4
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t = next(x for x in d["targets"] if x["id"] == "llama.cpp-upstream-b11430")
    if t["status"] != "documentation-only":
        print(f"REFUSED: target status is already {t['status']!r}", file=sys.stderr)
        return 4
    t["status"] = "locally tested"
    t["status_reason"] = ("X1a (2026-10-05): built unmodified for Linux x86-64 in Docker with the flags that reproduce "
                          "the pinned a7b94df binaries byte-for-byte; loads our exact UD-Q3_K_M with -nr and the "
                          "harness argv; greedy output bit-identical to a7b94df on 2 tuning prompts x 32 tokens. "
                          "Desktop functional check only: no controlled-memory cell and no Pi build yet.")
    t["runnable_for_exact_gguf"] = "yes (measured in X1a, desktop x86 Docker; engine id llama.cpp-upstream-8345f33)"
    t["engine_ids"] = ["llama.cpp-upstream-8345f33"]
    t["local_test"] = {
        "stage": "X1a-upstream-llama-recent",
        "date": now,
        "worktree": "third_party/llama.cpp-upstream-8345f33 (shallow clone, HEAD 8345f333951c661d166b00e6f9362e553768f292, "
                    "unmodified; git status clean after the build)",
        "build": {
            "flags": FLAGS,
            "flags_note": "X0 suggested GGML_NATIVE=OFF + AVX2 + AVX512; the pinned binaries report AVX512_VBMI/VNNI/BF16 "
                          "= 1, i.e. they were built with GGML_NATIVE=ON. A control rebuild of a7b94df with the flags "
                          "above gave 6/6 byte-identical binaries (sha256), so these ARE the pinned build's flags "
                          "(measured).",
            "seconds_at_j4": 198,
            "control_rebuild_a7b94df_seconds": 193,
            "image": "pi-gemma4-build:latest (gcc 13.3.0, cmake 3.28.3, ninja 1.11.1)",
            "binaries": "third_party/llama.cpp-upstream-8345f33/build-linux-x86/bin/{llama-server,llama-cli,"
                        "llama-completion,llama-bench,llama-perplexity,llama-tokenize}",
            "evidence": ["third_party/llama.cpp-upstream-8345f33/build-linux-x86/BUILD_INFO.json",
                         "third_party/llama.cpp-a7b94df-rebuild-x1a/build-linux-x86/BUILD_INFO.json",
                         "results/comparisons/x1a/build_both.log"],
        },
        "functional_check": {
            "kind": "functional only - not a benchmark (no memory budget, no storage throttle, desktop not verified idle)",
            "model": "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf, sha256 811882f6... measured in the run (Docker volume copy)",
            "server_args": "pigemma.controlled_cell.server_argv on the harness default config: -nr -t 4 -tb 4 -c 4096 "
                           "-b 2048 -ub 512 -ctk f16 -ctv f16 -fa auto -np 1 --cache-ram 0 --ctx-checkpoints 32 "
                           "--seed 0 -lv 4 --no-webui --warmup",
            "prompts": "tuning set: tun-01 (EN, 30 prompt tokens), tun-02 (HE, 39 prompt tokens); 32 greedy tokens each",
            "result_measured": {
                "tokens_identical_to_a7b94df": "64/64",
                "chosen_logprob_max_abs_diff": 0.0,
                "top5_ids_and_logprobs_identical_positions": "64/64",
                "prompt_tokenization_identical": True,
                "chat_template_sha256_identical": True,
                "within_engine_determinism": "a == b == streaming for both; a7b94df in a second process identical too",
                "weights": "CPU_Mapped model buffer 12123.75 MiB on both, no CPU_REPACK buffer",
                "differences": "CPU compute buffer 110.03 MiB vs 101.52 MiB (+8.51 MiB), graph nodes 2647 vs 2644, "
                               "input tensors 13 vs 9; llama-server --help byte-identical",
                "incidental_speed_not_a_benchmark": "decode 9.30-10.23 tok/s vs 9.39-10.33 tok/s (a7b94df); inside "
                                                    "the run-to-run spread",
            },
            "artifacts": ["results/comparisons/x1a/functional/run1/functional_check.json",
                          "results/comparisons/x1a/functional/run1/comparison.json",
                          "results/comparisons/x1a/functional/run1/<engine>/server.log"],
        },
        "engine_profile": "results/comparisons/engines.json (engine llama.cpp-upstream-8345f33, status runnable)",
        "pi_aarch64": "NOT built (stage rule; Pi / has about 2.15 GB free). Matching command recorded in engines.json "
                      "-> pi_aarch64_build and results/comparisons/x1a/README.md.",
        "open": ["controlled-memory cells on this engine (M1/M2, main workflow)", "Pi aarch64 build + resident A-B"],
    }
    t["changes_since_a7b94df"]["measured_locally_x1a"] = (
        "diff -rq of the two local trees (line endings ignored): 109 differing entries; in ggml/src outside the GPU "
        "backends only 4 files differ, all ggml-cpu/spacemit (RISC-V); x86/ARM CPU kernels and ggml-cpu/CMakeLists.txt "
        "identical. The bit-identical-output expectation for #29622 held on the functional check.")
    d["our_reference"]["engine"]["linux_build_flags_reconstructed_x1a"] = (
        FLAGS + " - a rebuild with these flags is byte-identical to third_party/llama.cpp/build-linux-x86/bin "
                "(6/6 binaries, sha256); see third_party/llama.cpp-a7b94df-rebuild-x1a/build-linux-x86/BUILD_INFO.json")
    d["updates"] = [{
        "stage": "X1a-upstream-llama-recent", "at": now,
        "what": "llama.cpp-upstream-b11430: documentation-only -> locally tested (runnable for the exact GGUF); "
                "pinned Linux build flags reconstructed and verified; results/comparisons/engines.json created"}]

    # ---- text surgery that keeps X0's hand formatting; the result must parse to exactly `d` ----
    if chr(13) in raw:
        print("REFUSED: registry has CR characters - unexpected", file=sys.stderr)
        return 4
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

    b0 = find('      "id": "llama.cpp-upstream-b11430"')
    b1 = find('      "id": "hotpin-llm"')
    i = find('        "expected_cpu_effect": ', b0, b1)
    if lines[i].rstrip().endswith(","):
        raise SystemExit("expected_cpu_effect is no longer the last key of changes_since_a7b94df")
    lines[i] = lines[i] + ","
    lines.insert(i + 1, '        "measured_locally_x1a": ' + js(t["changes_since_a7b94df"]["measured_locally_x1a"]))
    b1 += 1
    i = find('      "runnable_for_exact_gguf": ', b0, b1)
    lines[i] = '      "runnable_for_exact_gguf": ' + js(t["runnable_for_exact_gguf"]) + ","
    i = find('      "status": ', b0, b1)
    lines[i] = '      "status": ' + js(t["status"]) + ","
    i = find('      "status_reason": ', b0, b1)
    lines[i] = '      "status_reason": ' + js(t["status_reason"]) + ","
    lines.insert(i + 1, '      "engine_ids": ' + js(t["engine_ids"]) + ",")
    lines.insert(i + 2, '      "local_test": ' + dump(t["local_test"], 6) + ",")
    e0 = find('    "engine": {')
    e1 = find('    "model_files": [')
    i = find('      "status": "locally tested"', e0, e1)
    lines[i] = lines[i] + ","
    lines.insert(i + 1, '      "linux_build_flags_reconstructed_x1a": '
                 + js(d["our_reference"]["engine"]["linux_build_flags_reconstructed_x1a"]))
    text = NL.join(lines)
    tail = NL + "}" + NL
    if not text.endswith(tail):
        raise SystemExit("unexpected file ending")
    text = text[:-len(tail)] + "," + NL + '  "updates": ' + dump(d["updates"], 2) + tail
    back = json.loads(text)
    # key ORDER inside the edited objects differs from dict insertion order only cosmetically; compare content
    if back != d:
        raise SystemExit("text surgery does not reproduce the intended registry content - nothing written")
    if os.path.getmtime(P) != m0:
        print("REFUSED: registry.json changed while updating - rerun", file=sys.stderr)
        return 4
    tmp = P + ".x1a.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    os.replace(tmp, P)
    print("updated", P, os.path.getsize(P), "bytes; target status =", t["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
