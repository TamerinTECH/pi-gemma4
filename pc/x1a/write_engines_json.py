#!/usr/bin/env python3
"""X1a: create results/comparisons/engines.json (engine profiles for the controlled harness).

Run once from the repo root: python pc/x1a/write_engines_json.py
Refuses to overwrite an existing file (later stages APPEND entries by hand or with their own script).
"""
from __future__ import annotations

import datetime
import json
import os
import sys

P = "results/comparisons/engines.json"

FLAGS = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=ON",
         "-DGGML_OPENMP=ON", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_UI=OFF",
         "-DLLAMA_USE_PREBUILT_UI=OFF", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]

COMMON_BUILD = {
    "image": "pi-gemma4-build:latest (ubuntu 24.04, gcc/g++ 13.3.0, cmake 3.28.3, ninja 1.11.1); binaries run in "
             "pi-gemma4-build:fuse (harness image, same base)",
    "script": "pc/x1a/build_both.sh -> pc/x1a/build_engine_in_container.sh (source tree mounted read-only, build "
              "tree on a Docker volume, -j 4, nice 10, under results/.cpu.lock)",
    "effective_arch_flags": "ggml-cpu target: -O3 -DNDEBUG -march=native (= znver4 on the Ryzen 5 8400F: AVX2, "
                            "AVX-512 F/CD/VL/DQ/BW + VBMI/VNNI/BF16) -fopenmp; libllama/common: -O3 -DNDEBUG "
                            "(no -march), CMake default",
    "linkage": "libllama/ggml/common linked statically into each tool (BUILD_SHARED_LIBS=OFF); glibc, libstdc++, "
               "libgcc_s, libm and libgomp stay dynamic (same as the pre-existing a7b94df binaries; the word "
               "static in the stage text means this, not -static)",
    "not_portable": "-march=native: these binaries need an AVX-512 (Zen 4 class) CPU; do not copy them to "
                    "another machine",
}

ENGINES = [
    {
        "engine_id": "llama.cpp-a7b94df",
        "role": "reference: our pinned known-good revision (harness default pigemma.controlled.LLAMA_SERVER)",
        "binary_path_in_container": "/work/third_party/llama.cpp/build-linux-x86/bin/llama-server",
        "binary_path_host": "third_party/llama.cpp/build-linux-x86/bin/llama-server",
        "binary_sha256": "5f60d6845a6833ed6bafc5280d91cc0140af472afc5d5e549f75d918838d6176",
        "binary_bytes": 15679072,
        "other_tools_same_dir": ["llama-cli", "llama-completion", "llama-bench", "llama-perplexity",
                                 "llama-tokenize", "llama-debug", "llama-gguf-split", "(46 tools in total)"],
        "revision": "a7b94df2c616bc1f62a73b964b4a71cb0dcc488e",
        "release_tag": "b11398",
        "version_output": "version: 0.5.0-dev (build 1, commit a7b94df) / built with GNU 13.3.0 for Linux x86_64",
        "patches": "none",
        "build_flags": FLAGS,
        "build_flags_status": "MEASURED: the original configure line was never recorded (binaries built 2026-10-04 "
                              "22:43, CMakeCache not kept). X1a rebuilt a7b94df with exactly these flags and all 6 "
                              "rebuilt binaries (llama-server, llama-cli, llama-completion, llama-bench, "
                              "llama-perplexity, llama-tokenize) are byte-identical (sha256) to the pre-existing "
                              "ones, so these flags reproduce the pinned build. Evidence: "
                              "third_party/llama.cpp-a7b94df-rebuild-x1a/build-linux-x86/BUILD_INFO.json",
        "build": COMMON_BUILD,
        "status": "runnable",
        "status_evidence": "all earlier Docker runs + results/comparisons/x1a/functional/run1 (2 server starts)",
        "parent_engine_id": None,
        "compare_against": None,
        "comparison_group": "same-model track A (reference engine for methods A/B/C/D)",
        "harness": {"server_argv_compatible": True, "extra_server_args": [], "needs_nr": True,
                    "methods": ["A", "B", "C", "D"]},
        "notes": "Unchanged by X1a (third_party/llama.cpp was only read; git status clean; build-linux-x86 still "
                 "holds only bin/).",
    },
    {
        "engine_id": "llama.cpp-upstream-8345f33",
        "role": "recent upstream ggml-org/llama.cpp (brief UPDATE 2 section 2, P0): separates engine/upstream "
                "drift from our method gains",
        "binary_path_in_container": "/work/third_party/llama.cpp-upstream-8345f33/build-linux-x86/bin/llama-server",
        "binary_path_host": "third_party/llama.cpp-upstream-8345f33/build-linux-x86/bin/llama-server",
        "binary_sha256": "103e3cddc57a97872cc638b73cb148b71054dd07e52ecfbc1950d094090b3d35",
        "binary_bytes": 15711200,
        "other_tools_same_dir": ["llama-cli", "llama-completion", "llama-bench", "llama-perplexity",
                                 "llama-tokenize"],
        "revision": "8345f333951c661d166b00e6f9362e553768f292",
        "release_tag": "b11430 (per X0; the shallow clone has no tags, HEAD sha verified locally)",
        "version_output": "version: 0.6.0-dev (build 1, commit 8345f33) / built with GNU 13.3.0 for Linux x86_64",
        "patches": "none (unmodified worktree: shallow clone third_party/llama.cpp-upstream-8345f33, git status "
                   "clean after the build)",
        "build_flags": FLAGS,
        "build_flags_status": "MEASURED: same script, image and flags as the a7b94df control rebuild that reproduced "
                              "the pinned binaries byte-for-byte; build 198 s at -j 4. Evidence: "
                              "third_party/llama.cpp-upstream-8345f33/build-linux-x86/BUILD_INFO.json (+ logs/)",
        "build": COMMON_BUILD,
        "status": "runnable",
        "status_evidence": "results/comparisons/x1a/functional/run1/{functional_check.json,comparison.json}: loads "
                           "our exact UD-Q3_K_M (sha256 811882f6..., measured in the run) with the harness's own "
                           "argv (-nr, -t 4, -c 4096, -b 2048, -ub 512, f16 KV, --cache-ram 0, --ctx-checkpoints "
                           "32, --no-webui, --warmup), CPU_Mapped 12123.75 MiB, no CPU_REPACK buffer; /completion "
                           "non-streaming and the harness streaming client both work",
        "parent_engine_id": None,
        "compare_against": "llama.cpp-a7b94df",
        "comparison_group": "runtime track A-B (same GGUF, different engine revision; NOT a method comparison)",
        "harness": {
            "server_argv_compatible": True, "extra_server_args": [], "needs_nr": True,
            "methods": ["A", "B", "C", "D"],
            "methods_note": "the pinner/FUSE helpers are engine-independent, so every method can run on this "
                            "engine; the minimum useful cell is method A (plain upstream + Linux page cache) next "
                            "to method A on llama.cpp-a7b94df. llama-server --help is byte-identical between the "
                            "two revisions (736 lines, diff = 0).",
        },
        "functional_check": {
            "kind": "FUNCTIONAL, not a benchmark (no memory budget, no storage throttle, desktop not verified "
                    "idle, cpuset 2,4,6,8)",
            "prompts": "tuning set only: tun-01 (EN chat, 30 prompt tokens) and tun-02 (HE chat, 39 prompt tokens), "
                       "rendered with pigemma.workloads.render_gemma4, 32 greedy tokens each (temperature 0, "
                       "seed 0, cache_prompt false)",
            "vs_llama.cpp-a7b94df": {
                "prompt_token_ids_identical": True,
                "generated_token_ids_identical": "2/2 prompts, 32/32 tokens each",
                "text_identical": True,
                "chosen_token_logprob_max_abs_diff": 0.0,
                "top5_candidates": "same 5 ids and exactly equal logprobs at all 64 positions (as printed by the "
                                   "server JSON)",
                "within_engine_determinism": "run a == run b == streaming run for both engines; a7b94df restarted "
                                             "in a second process reproduced the same tokens and logprobs",
                "measured_differences": "graph nodes 2647 vs 2644, input tensors 13 vs 9, CPU compute buffer 110.03 "
                                        "MiB vs 101.52 MiB (+8.51 MiB anonymous memory at -c 4096 -b 2048 -ub 512; "
                                        "attributed to the mixed token/embedding batch inputs of PR #29622 - "
                                        "attribution from the source diff, not bisected); KV log label attn_rot_* "
                                        "-> n_rot_*; nothing else in the load log",
                "incidental_speed_not_a_benchmark": "decode 9.30-10.23 tok/s (upstream) vs 9.39-10.33 tok/s "
                                                    "(a7b94df, two processes); prefill 24.6-28.8 vs 22.2-28.8 "
                                                    "tok/s on 30-39-token prompts; differences are inside the "
                                                    "run-to-run spread - no speed claim either way",
            },
        },
        "source_diff_vs_a7b94df": "measured on the two local trees (diff -rq, line endings ignored): 109 differing "
                                  "entries; in ggml/src outside the GPU backends only 4 files differ, all under "
                                  "ggml-cpu/spacemit (RISC-V); the x86 and ARM CPU kernels and "
                                  "ggml-cpu/CMakeLists.txt are identical; gemma4.cpp differs only in the "
                                  "embedding-scale call (build_inp_embd(tok_embd, sqrtf(n_embd)))",
        "pi_aarch64_build": {
            "status": "recorded only - NOT built in X1a (stage rule); the Pi / has about 2.15 GB free, below what "
                      "a source + build tree needs above the 2 GB gate",
            "matching_native_command": "same as the pinned Pi build (pi/adhoc/build_llama_native.sh, a7b94df, "
                                       "410 s): git init + fetch --depth 1 origin "
                                       "8345f333951c661d166b00e6f9362e553768f292 into a NEW directory (never "
                                       "~/pigemma/llama.cpp), then: nice -n 10 cmake -S <src> -B <src>/build "
                                       "-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF "
                                       "-DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF && nice -n 10 cmake "
                                       "--build <src>/build -j 4 --target llama-cli llama-completion llama-server "
                                       "llama-bench llama-perplexity",
            "expected_flag_resolution": "as in results/pi-real/pi-logs/build-cmake.log for a7b94df: "
                                        "-mcpu=cortex-a76+crc+crypto+dotprod+noi8mm+nosve+nosme on the ggml-cpu "
                                        "target (ggml-cpu/CMakeLists.txt is identical in both revisions; gcc "
                                        "14.2.0) - ASSUMED until built",
            "where": "source + build tree must go to the USB stick (/mnt/usb, FAT32, 35 GB free) or wait until / "
                     "has room; a build on FAT32 is untested",
            "fallback_cross_build": "desktop arm64 container pi-gemma4-pi-emu:trixie with -DGGML_NATIVE=OFF "
                                    "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 (the flags of the existing "
                                    "emulated a7b94df build in volume pigemma-arm64-llama, 1321 s). That -march "
                                    "line has no cortex-a76 tuning, so it does not reproduce the Pi's native "
                                    "binary: use it only as a matched PAIR (both revisions from the container) "
                                    "and never against earlier Pi numbers",
            "expected_cpu_delta": "ASSUMED about 0 (no ARM kernel source change); not measured",
        },
        "notes": "Outputs are bit-identical to a7b94df on the functional check, so M1/M2 can treat this engine as a "
                 "drift control: an A-vs-A difference between the two engines in the controlled cells would be "
                 "noise or the +8.5 MiB compute buffer, not a kernel change. License MIT.",
    },
]

DOC = {
    "schema": "pigemma-engines v1",
    "purpose": "Engine profiles for the controlled-memory harness (pigemma.controlled / controlled_cell): which "
               "llama-server binaries can run a cell on our exact track-A GGUF, how they were built, and against "
               "which engine each one must be compared.",
    "created_by": "X1a-upstream-llama-recent (comparisons workflow wf_02d183f6-341)",
    "updated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "how_to_use": "Set cfg['server']['bin'] of a cell to binary_path_in_container (default_config() puts "
                  "pigemma.controlled.LLAMA_SERVER there) and append harness.extra_server_args to "
                  "cfg['server']['extra']. Paths are inside the harness container (repo bind-mounted at /work, "
                  "image pi-gemma4-build:fuse). A patched engine is compared against its parent_engine_id, an "
                  "unpatched revision against compare_against; never attribute a revision difference to a caching "
                  "method.",
    "status_vocabulary": {
        "runnable": "built, loads our exact GGUF with the harness argv and generates; may be used in M1/M2",
        "blocked": "cannot be run for the exact GGUF; reason recorded",
        "pending": "registered by a later stage, not checked yet",
    },
    "append_rule": "Later stages (X1b: hotpin-parent + hotpin; X1c: ik-plain + ik-prefetch-experts) APPEND entries "
                   "to 'engines', remove their names from 'pending_engines' and update 'updated'; do not rewrite "
                   "existing entries.",
    "model_checked": {
        "file": "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf",
        "sha256": "811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052",
        "bytes": 12728497888,
        "path_in_container": "/models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf (Docker "
                             "volume pigemma-models; same sha256 as models/unsloth__gemma-4-26B-A4B-it-GGUF/ on D: "
                             "per P4's re-check at 22:30)",
    },
    "engines": ENGINES,
    "pending_engines": ["hotpin-parent (X1b)", "hotpin (X1b)", "ik-plain (X1c)", "ik-prefetch-experts (X1c)"],
}


def main() -> int:
    if os.path.exists(P):
        print(f"REFUSED: {P} exists - merge by hand", file=sys.stderr)
        return 4
    with open(P, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(DOC, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print("wrote", P, os.path.getsize(P), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
