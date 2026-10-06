# Comparison targets: compatibility, licensing and local test plan

Stage X0 of the comparisons workflow, written 2026-10-05 ~20:20 CEST, following brief UPDATE 2 §2
(`docs/design/INSTRUCTIONS_2026-10-05_v2.md`). The machine-readable version, with sizes, shas and sources, is
[`results/comparisons/registry.json`](../../results/comparisons/registry.json).

No builds and no model runs were done in this stage. The evidence comes from:

- the GitHub API (compare, PR bodies, file contents);
- read-only shallow clones in `third_party/hotpin-llm` (ac52bd9a) and `third_party/ik_llama.cpp` (3d27f5bb);
- the Hugging Face API and model cards;
- 24 MiB range reads of two GGUF headers;
- arXiv;
- one read-only ssh check of the Pi.

**Status values** (later stages update them): locally tested / documentation-only / blocked / not applicable.
Numbers published by a project appear only in the "vendor-reported" table at the end. None of them are our results.

**Our reference.** llama.cpp `a7b94df` (b11398, 2026-10-04) with these files:

- `gemma-4-26B-A4B-it-UD-Q3_K_M.gguf`: sha256 `811882f6…`, unsloth@c099eb48, non-QAT. Its real tensor types are
  IQ3_XXS and IQ4_NL experts, IQ4_XS + Q8_0 in layer 29, Q8_0 attention and dense FFN, a Q6_K tied head, and F32.
- Google QAT `gemma-4-26B_q4_0-it.gguf`: sha256 `3eca3b8f…`, Q4_0, Q6_K and F32.

## Table

| Pri | Target | License | Comparison group | Gemma 4 + our exact types | ARM / Cortex-A76 | Status (X0) | Planned local test |
|---|---|---|---|---|---|---|---|
| P0 | **Upstream llama.cpp b11430** `8345f333` (2026-10-05 15:55 UTC) | MIT | runtime track A-B | yes; loader unchanged except the embedding-scale path | yes (native build as for a7b94df) | **locally tested** (X1a, 2026-10-05 22:55: runnable on desktop x86; greedy output bit-identical to a7b94df on 2 tuning prompts x 32 tokens; see `results/comparisons/x1a/README.md`). Pi build not done | X1a: Docker build with the same flags as a7b94df; 2 tuning prompts × 32 greedy tokens vs a7b94df (`-nr`); register in `engines.json` |
| P0 | **HotPin** `LozzKappa/hotpin-llm` @ac52bd9a, patches on llama.cpp `a6647b1` (b9857, 2026-07-01) | **AGPL-3.0** + paid commercial license | same-model track A, vs its **own** parent a6647b1 | **yes**: a6647b1 contains PR #21309 (1221 commits after it). Its loader matches our GGUF; `-nr` exists | author tested only Zen5/Windows ("AVX2 or newer"). L1 patch is portable C++. X1b: builds for aarch64 (cross GCC 13.3) and its pin path works under qemu-user; not run on the real Pi; not a drop-in Pi solution (memlock limit, `-nr`, no memory cgroup there) | **locally tested** (X1b, 2026-10-06 00:05: `hotpin-parent` and `hotpin` runnable on desktop x86; HotPin's own profiling run on `calibration.jsonl`; with `-nr` it locks exactly the predicted pages and is bit-identical to its parent; one functional smoke cell in the controlled harness (locks hold in the 8 GiB cgroup), no measurement cell yet; see `results/comparisons/x1b/README.md`) | X1b: build a6647b1 unmodified + patched; run its own routing dump on `calibration.jsonl`; `make_hot_file.py` at method C's budget ("original" and "budget-matched" variants); `llama-server -nr` in the same cgroup; verify `VmLck`/smaps against the expert map |
| P1 | **ik_llama.cpp** @3d27f5bb (Gemma 4 since PR #1581, 86e33fd6, 2026-04-06) | MIT | runtime track A-B | yes; requires fused `ffn_gate_up_exps` (we have it). NEON iqk kernels for IQ3_XXS, IQ4_NL, IQ4_XS, Q8_0, Q6_K, Q4_0 | "ARM_NEON or better"; iqk needs dotprod (A76 has it); Rock 5B (A76) user report; no Pi 5 report | documentation-only | X2: aarch64 build in the desktop arm64 container (the Pi `/` has only 2.15 GB free); resident A/B vs a7b94df on the Pi with the UD-Q3_K_M file. Flags: `-ser` off, `--swa-compress`, no `-rtr`. Optional desktop cell: `--prefetch-experts` |
| P1 | **SmallThinker + PowerInfer `smallthinker` runtime** (Tiiny-AI/PowerInfer main, smallthinker/ 2025-07-28) | runtime MIT, models Apache-2.0 | different-model track B | n/a (own MoE models, ReGLU); mainline llama.cpp also has arch `smallthinker` | vendor publishes Pi 5 numbers (cross-built with clang 19, `+nodotprod`). Our Pi lacks clang, mold, libaio-dev and liburing-dev | documentation-only | X3: build in the arm64 container (clang-19, libaio, liburing); run 4BA0.6B `Q4_0.powerinfer.gguf` (2.29 GB) resident on USB with the same rubric; plain Q4_0 on a7b94df as a control; 21BA3B (12.14 GB, must be split) only if 4B is informative |
| P1 | **Gemma 4 E4B-it QAT Q4_0** (Google, rev 4b4a2c1d, 5.15 GB; fallback **E2B** 3.35 GB) on llama.cpp a7b94df | Apache-2.0 | different-model track B | same family, template and QAT provenance; Q4_0, Q6_K, F16, F32 (header read) | yes (same runtime as our Pi runs) | documentation-only | X3: download to C:, split to < 4 GiB for FAT32, copy to the USB stick; same held-out tasks + rubric as the 26B; cold vs warm |
| P2 | **picoLLM** (Picovoice) | repo Apache-2.0; engine needs an AccessKey | different-model track B | **no Gemma 4** (Gemma 2b/7b, Gemma3 270m, Llama, Mistral, Phi …) | Pi 3/4/5 listed by the vendor | documentation-only | none: there is no authorised AccessKey, and license validation needs the internet, so it is not air-gapped |
| R | **MoE-Infinity** (arXiv 2401.14361) | Apache-2.0 | research-only | no | no (CUDA GPU required) | not applicable | none; transferable ideas → [`PRIOR_ART_CACHE.md`](PRIOR_ART_CACHE.md) (X0 extension) |
| R | **Mira** (arXiv 2609.38090) | paper, no code found | research-only | no | no (single GPU) | documentation-only | none; its HOT+STAGE design + bounded rebalancing is a template for the "cost-aware cache" only if the simulator shows headroom |
| R | **SVD-LLM** (arXiv 2403.07378) | Apache-2.0 | research-only (track C) | not listed (Llama, Mistral, OPT) | PyTorch/GPU | documentation-only | none; D1 already has "base alone" and "static selection" arms |
| R | **TEAL** (arXiv 2408.14690) | MIT | research-only (track C) | no (Llama, Mistral thresholds) | no (Triton/CUDA kernels) | not applicable | none; a TEAL-style magnitude selector is a cheap baseline for D1's online selector |
| R | **LLM in a Flash** (arXiv 2312.11514) | paper, no code | research-only (track C) | no (needs ReLU sparsity) | n/a | not applicable | none; reuse its layout lessons (bundling, read granularity, windowing) |

## Notes per target

**Upstream llama.cpp.** Our pinned a7b94df is only one day older than b11430 (32 commits, 0 behind).

- Nothing changed in the x86 or ARM `ggml-cpu` paths. The only CPU-kernel commit is RISC-V (SpacemiT IME1, #28479).
- MoE code is unchanged.
- One commit touches Gemma 4: #29622 applies the `sqrt(n_embd)` embedding scale through a per-row `ggml_mul`
  instead of `ggml_scale`. It is the same multiplication, so bit-identical output is expected, but X1a must check it.
- Everything else is GPU backends, server code or quantized-KV metadata.

The upstream comparison is therefore a sanity check with an expected CPU delta of about 0 (ASSUMED). It is not a
source of gains. Run it cheaply, and include it in M1/M2 only if the functional check passes. Our Linux a7b94df
build kept no `CMakeCache.txt`. X1a must record explicit flags and rebuild both revisions the same way if the
flags are unclear.

**HotPin.** HotPin is **not blocked**: its parent `a6647b1` already supports Gemma 4, and the author ran
`gemma4:26b-a4b` (unsloth UD-Q4_K_M, a different file from ours). These findings come from reading the source:

- **L1** is about 40 lines in `load_tensors`. It calls `mlock` on `t->data + e·nb[2]` for every listed (layer,
  expert) slice of the `*_exps.weight` tensors.
  - It must run with `-nr`. Otherwise the repacked anonymous copies get locked instead of file pages: IQ4_NL down
    experts on x86, and more tensor types on the A76.
  - Its "pinned N MiB" log counts **requested** bytes. A failed lock only prints a warning, so we need `VmLck` or
    smaps `Locked:` evidence.
  - It rounds the size, not the start address, so a slice can lock up to one extra page.
- **L3** prefetch is Windows-only (`PrefetchVirtualMemory`). On Linux and the Pi it does nothing, and it ignores
  `ffn_gate_up_exps`.
- **Ranking** uses a routing dump from patched `llama-simple-chat` (`ffn_moe_topk` callback). It counts global
  (layer, expert) frequencies and turns a GB budget into a slice count using the estimate `file_size/(30·128)`.
  For our UD file that estimate is 3.31 MB, against a true slice of 2.63 MB, so it fills about 20 % less than the
  budget. X1b must report the actual locked pages and also run a budget-matched variant.
- The shipped `hot-files/gemma4-26b/*` must **not** be used, because they come from another GGUF.
- **License.** AGPLv3 network copyleft plus a paid commercial license. Use it for comparison only and keep it
  isolated in `third_party/hotpin-llm` with its notices. **Licensing question for the operator before any product use.** Our
  `tools/pinner` must stay independent code.
- **Prior art.** Its published Gemma 4 result was measured in RAM and is flat (+0.0 % / −0.1 %). Its +45.5 % gain
  is for gpt-oss-120b on NVMe faster than 1 GB/s. That is consistent with our expectation that pinning only matters
  when the model does not fit in memory.

**HotPin, measured in X1b (2026-10-06).** Both engines build at `a6647b1` with the pinned-build flags and load our
exact GGUF; they are `runnable` in `results/comparisons/engines.json`. Findings that change how it must be compared:

- Its routing dump is wrong for every multi-token evaluation (it ignores the stride of the `ffn_moe_topk` view), so
  prompt tokens add a uniform count and the ranking is in effect built from generated tokens only (1,971 of 8,226
  token evaluations of the calibration run).
- With `-nr` it locks exactly the predicted pages (426,466 pages for its "2 GB" set = 1.63 GiB, 0 failed requests).
  Without `-nr`, 40 % of the locked memory is anonymous repacked copies. With a 64 MiB memlock limit 1,246 of 1,294
  requests fail while its log still reports the full amount.
- Output is bit-identical to its parent. `a6647b1` and `a7b94df` are **not** numerically identical on the same
  prompts, so HotPin is judged only against `hotpin-parent`.
- Details, pin-set variants (original, budget-matched, stride-fixed) and run instructions:
  `results/comparisons/x1b/README.md`.

**ik_llama.cpp.** The source supports our GGUF and our quant types on NEON. Three properties matter for fairness:

1. ik keeps weights mmap-backed by default (`-rtr` is opt-in).
2. ik keeps a full-size SWA KV cache unless `--swa-compress` is set, which is more KV memory than llama.cpp uses at
   the same `-c`.
3. `-ser` (smart expert reduction) changes the computation, so it stays off in track A.

ik already ships an **exact storage-to-RAM expert prefetcher**, `--prefetch-experts` (#2101): `pread` workers on
routed expert slices during decode and lookahead streaming plus `MADV_COLD` during prefill. Its authors report
1.9–2.7× prefill speedup when the model is larger than RAM; the decode change they call noise. This is an existing
solution that any track-A prefetching idea of ours must beat, and the strongest reason to test ik on the desktop
harness, not only on the Pi. Gemma 4 numerics differ from mainline (fused ops, router-scale absorption #1615), so
differences between engines are documented and not called failures.

**PowerInfer / SmallThinker.** These are different models, trained "mainly on English" according to their model
cards, so Hebrew and German are expected to be weak. That is a hypothesis to measure, not a result.

- **Vendor Pi build.** The vendor's Pi 5 toolchain disables dotprod (`-march=armv8.2-a+nodotprod+noi8mm+nosve`).
  A native build would enable it, so we record which one we use.
- **Approximation.** The default "sparse" mode uses a sparse `lm_head` predictor that the README itself calls lossy.
  We report vendor-default outputs and say so.
- **Dense control.** Mainline llama.cpp a7b94df knows the `smallthinker` architecture. The plain Q4_0 GGUF on our
  runtime therefore separates "model choice" from "PowerInfer runtime".
- **Storage.** 21BA3B is 12.14 GB, which needs splitting on the FAT32 stick. Whether the PowerInfer sparse and
  memory-efficient paths accept split GGUFs is unverified. The microSD has no room.

**Smaller multilingual model: Gemma 4 E4B-it QAT Q4_0.** We chose it because it shares the 26B's tokenizer, chat
template, thinking format, Google QAT provenance, Apache-2.0 license and our exact runtime. Only model size changes,
and `pigemma/quality_v2.py` applies unchanged.

The vendor language evidence is "pre-trained on 140+ languages" and "out-of-the-box support for 35+". The MMMLU
score is E4B 76.6 % vs 26B-A4B 86.3 %, and MMMLU covers German but not Hebrew. So Hebrew quality is unknown until
measured.

The header (MEASURED from a range read) shows that 2.31 GB of the 5.15 GB is the per-layer embedding table, which is
looked up, not multiplied. Compute weights are about 2.8 GB, so decode should be far cheaper than the 26B (PROJECTED).
The fallback, E2B (3.35 GB), fits the FAT32 stick as a single file.

These alternatives stay documentation-only:

- Qwen3/Qwen3.5-4B: Apache-2.0, broad multilingual claim.
- Phi-4-mini: MIT, lists Hebrew and German.
- Llama 3.2 3B: no Hebrew among its official languages; Llama license.

**picoLLM.** "Inference 100 % offline", but the AccessKey must be validated online against Picovoice servers, and
the models come only as proprietary compressed files from the Console. It has no Gemma 4. Without credentials
authorised for this project it stays documentation-only, and it does not qualify as an air-gapped option.

**Dense research (track C): how the three differ from our dynamic residual selection.** Our mechanism is
W = W_base + Σ_g R_g. The activation-aware low-rank base W_base is always computed. Each residual block R_g is
read and multiplied only when a cheap bound ‖R_g x_g‖ ≤ ‖R_g‖·‖x_g‖ says it can matter for **this** token.

| Work | Class | What is selected, and when | Exactness / cost | Gemma 4 |
|---|---|---|---|---|
| SVD-LLM | low-rank compression | nothing per token: one truncated factorisation per matrix, chosen offline (whitening + sequential low-rank update) | the residual is discarded for good | not supported (Llama/Mistral/OPT, PyTorch) |
| TEAL | activation sparsity | per token, input entries with small \|x_i\| are zeroed and matching weight columns skipped (per-tensor calibrated thresholds) | approximate, no base; GPU Triton kernels; quantized sparse kernels unsupported | no |
| LLM in a Flash | storage-aware access + ReLU sparsity | per token, a low-rank predictor picks FFN neurons that ReLU leaves non-zero; windowing keeps recent neurons in DRAM; row-column bundling enlarges flash reads | exact only where ReLU zeros are exact; needs ReLU or relufied models | no (Gemma 4 is GELU-tanh) |
| **Ours (D1)** | dynamic residual selection | per token, residual blocks of a base+residual decomposition by an error bound, without reading or computing all corrections | approximate unless all blocks are taken; selection + metadata cost must be counted | applies to the original weights by construction (tested on Gemma 31B layers in D1) |

A resemblance does not transfer support. None of the three runs Gemma 4 or GGUF on a CPU. SVD-LLM corresponds to
our "base alone" arm. A TEAL-style magnitude selector is the obvious cheap baseline for our online selector. LLM in
a Flash's chunking and recency lessons apply to how residual blocks would be laid out on microSD or USB.

## Local test plan (order and gates)

1. **X1a (desktop, CPU lock, -j 4).** Build upstream b11430 in `third_party/llama.cpp-upstream-8345f33` with the
   same flags as a7b94df. Run the functional check on UD-Q3_K_M with `-nr`. Write the result to
   `results/comparisons/engines.json`.
2. **X1b (desktop, CPU lock).** Build HotPin's parent a6647b1 unmodified, and a6647b1 with the HotPin patches.
   Rank with HotPin's own tool on `data/workloads/calibration.jsonl` only.
   - Run two pin-set variants: "original" (its tool's budget) and "budget-matched" (exact unique pages from
     `results/maps/map26_ud_q3_k_m.json`).
   - Verify locked pages and failed locks.
   - Register `hotpin-parent` and `hotpin` so M1/M2 run them in the same cgroup budget as A/B/C/D.
   - An aarch64 build follows only after a desktop result.
3. **X2 (Pi, Pi lock, resident, nice -n 10).** Cross-build ik_llama.cpp in the arm64 container, because the Pi `/`
   has 2.15 GB free.
   - Load our exact UD-Q3_K_M from the microSD.
   - Freeze settings on 2 tuning requests, then run a held-out subset (EN/HE/DE + one ~1000-token prompt) with
     alternating order and 3 repeats against a7b94df `-nr`.
   - Optional desktop harness cell: `--prefetch-experts` vs plain mmap at method A's budget.
4. **X3 (Pi, product track B).** Write the rubric first (`data/workloads/product_rubric.json`). Then run three
   configurations, one model at a time, with first-use and warm sessions separated:
   - the best Gemma 26B configuration;
   - Gemma 4 E4B QAT (or E2B);
   - SmallThinker-4BA0.6B on PowerInfer, plus the plain Q4_0 on a7b94df as a control.
5. **No action:** picoLLM (credentials), MoE-Infinity / TEAL (GPU only), Mira / SVD-LLM / LLM in a Flash (papers;
   used for positioning and D1 baselines).

**Pi limits for all of the above** (read-only check, 20:10):

- `/` has 2,145,472,512 B free, just above the 2 GB gate, so no build trees go on `/`.
- The USB stick has 35.4 GB free but is FAT32: under 4 GiB per file and no symlinks; binaries can run from it.
- There is no clang, mold, libaio-dev or liburing-dev.
- The Pi llama.cpp build has no `llama-gguf-split`, so any splitting happens on the desktop.

## Vendor-reported numbers (NOT our measurements; hardware and settings as stated by the source)

| Source | Device | Model / config | Reported |
|---|---|---|---|
| SmallThinker README / model cards | Raspberry Pi 5 | SmallThinker 4B-A0.6B sparse, Q4_0 (2.24 GiB) | 28.77 tok/s |
| same | Raspberry Pi 5 | SmallThinker 4B, memory limited to 1 GiB | 0.75 tok/s |
| same | Raspberry Pi 5 | SmallThinker 21B-A3B sparse, Q4_0 (11.47 GiB) | 6.61 tok/s |
| same | Raspberry Pi 5 | Qwen3 0.6B / Qwen3 1.7B / Gemma 3n E2B / E4B (their runtime) | 27.44 / 11.08 / 6.66 / 4.01 tok/s |
| same | rk3588 (A76 + A55) | SmallThinker 21B sparse / limited to 8 GB | 10.84 / 8.56 tok/s |
| HotPin `results/all_benchmarks.json` | Ryzen AI 9 HX 370, Windows | gemma4 26B UD-Q4_K_M in RAM: baseline / L1 2 GB / L1 4 GB | 15.53 / 15.53 / 15.52 tok/s |
| HotPin README | same | gpt-oss-120b Q4_K_M (58.5 GB): mmap / L1 pin | 2.64 / 3.84 tok/s (+45.5 %) |
| ik_llama.cpp PR #2101 | i9-14900F + RTX 4080S, NVMe | `--prefetch-experts`, model > RAM, cold | PP ×1.88–2.69; TG "likely noise" |
| Gemma 4 model card | — | MMMLU E4B / E2B / 26B-A4B / 31B | 76.6 / 67.4 / 86.3 / 88.4 % |
| LLM in a Flash | M1 Max (CPU) | OPT 6.7B per-token I/O vs naive | 105 ms vs 2196 ms |
| TEAL | GPU (gpt-fast) | decode at 40 % / 50 % sparsity | 1.53× / 1.8× |

The SmallThinker figures give no prompt length, context size or run count, and use "the number of threads that can
achieve the maximum speed". They are presumably decode rates. X3 reproduces them locally and does not quote them.
