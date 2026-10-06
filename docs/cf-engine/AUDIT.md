# CF-Engine: focused baseline audit

Audited 2026-10-06. This records historical evidence and a new **read-only E2B GGUF header inventory**, not new performance measurements. The 26B-A4B remains the primary large-model target. [Machine-readable audit and all 541 E2B tensor records](../../results/cf-engine/audit.json).

## Corrections that govern this implementation

1. **The approximately 8 tok/s measurement is E2B, not 26B.** The morning 26B measurement is 3.34 server decode tok/s for the unsloth UD-Q3_K_M artifact. That artifact is not the Google QAT Q4_0 chosen for initial CF packing.
2. **1.379 GB/token was computed from tensor categories, not measured at the DRAM interface.** Its categorization also omitted a real 27.5 MB dense projection, described below. Neither the old nor corrected inventory establishes physical traffic.
3. **33.38 tok/s is aggregate model-token throughput across 16 streams.** It uses one common request interval. 2.48 tok/s is a per-stream server decode-window rate, which uses a shorter interval. They are compatible measurements of different intervals.
4. **Visible-answer speed was not measured by that morning harness.** Both retained single-stream E2B and 26B chat samples have empty `content` and nonempty `reasoning_content`. Completion-token counts therefore cannot be called useful-answer tokens.
5. **The retained MTP logs demonstrate engagement and timing, but do not establish identical greedy output.** Preserve the result as preliminary engaged-MTP-runtime evidence, with its correctness claim unverified here. Do not start another speculative experiment before the block path works.

Raw measurements remain unchanged under `results/pi-real/speed-ceiling/`; older design claims in `CLAUDE.md` must be read with these corrections. No raw evidence or old build was replaced by this audit.

## Model identity and inventory

The E2B artifact was read through `ssh pi@context-friend-1` using a stdlib GGUF header parser which reads metadata and tensor descriptors, then stops before weight data. It did not load a model, hash all weights, warm them, or run inference. Path: `/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf`.

| Property | Header observation |
|---|---|
| File bytes | 3,349,516,256 |
| GGUF / tensor count | version 3 / 541 |
| Text tensor logical elements | 4,628,569,635 |
| Quantization inventory | 275 Q4_0; 2 Q6_K; 1 F16; 263 F32 |
| Layers / hidden / context | 35 / 1536 / 131072 |
| PLE table | Q6_K `[8960,262144]`, 1,926,758,400 bytes |
| PLE model projection | F16 `[1536,8960]`, 27,525,120 bytes |
| Tied input/output embedding | Q6_K `[1536,262144]`, 330,301,440 bytes |

This is the text GGUF inventory, not the total multimodal checkpoint parameter count. Google's model card distinguishes E2B's effective parameters from its larger parameter inventory including PLE, and explains that PLE tables are accessed by lookup. The 26B variant instead selects eight of 128 experts plus a shared expert. [Google Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4).

The 26B Google QAT Q4_0 inventory already exists in `results/maps/map26_qat_q4_0_split5.json`: expert gate/up Q4_0 `[2816,1408,128]`, down Q4_0 `[704,2816,128]`, F32 norms/router/scales, and a Q6_K tied head. The morning unsloth 26B inventory is materially different: IQ3_XXS, IQ4_NL, IQ4_XS, Q8_0, Q6_K, F32. See `results/comparisons/ik_llama_pi/X2_RESULT.json` and `results/maps/map26_ud_q3_k_m.json`. Filename precision labels are insufficient.

## Traffic and bandwidth

`results/pi-real/e2b_bytes_per_token.json` computes 1,379,415,180 bytes by adding four non-PLE categories. It excludes the entire 1,954,284,544-byte category labeled “per-layer embedding table (row lookup only)”. Header inspection reconstructs that excluded category exactly:

```
1,926,758,400  per_layer_token_embd.weight   Q6_K lookup table
   27,525,120  per_layer_model_proj.weight  F16 dense projection
        1,024 per_layer_proj_norm.weight   F32 norm
-------------
1,954,284,544
```

In the reference `src/models/gemma4.cpp`, `build_inp_per_layer()` uses `ggml_get_rows` for the table (around line 481), but `project_per_layer_inputs()` uses `ggml_mul_mat` for the full F16 projection (around line 515). A single Q6_K PLE row is 7,350 bytes. Adding that row, the projection and norm yields a **corrected coarse logical source-weight-access estimate of 1,406,948,674 bytes/token**. The tied token-embedding row is 1,260 bytes already inside the full head's address range; counting its separate lookup adds that tiny logical access. The original aggregation script was not retained; its arithmetic and excluded bucket are reconstructible from the artifact.

This estimate still omits KV, intermediates, packing, repeated kernel reads, cache-line rounding and other executor traffic. It does not count cache hits or measure transactions. It is neither a DRAM counter nor a validated trace reconstruction of every load/store.

The preserved `membw` result is a **measured sequential-access bandwidth microbenchmark**, using known bytes divided by elapsed time, not memory-controller counters: a 1 GiB arena, best of five passes, read rates 14.061 / 13.527 / 13.058 / 12.477 decimal GB/s for 1/2/3/4 threads. `tools/membw/membw.c` documents the count; write allocation is excluded from copy/triad accounting. Evidence: `results/pi-real/membw_pi_2026-10-06.json`.

Under the old **assumed** 1.379 GB/token denominator, dividing these rates yields roughly 9.05–10.19 tok/s. The observed rate as a percentage of this quotient is a modeled ratio, not measured DRAM utilization. The more appropriate relationship remains:

```
single-stream committed tokens/s <= effective DRAM bytes/s / DRAM bytes per committed token
```

At the measured microbenchmark rates, 100 committed tokens/s permits about **125–141 MB/token** before other limits. Relative to the old assumption this requires roughly **9.8–11.1x less traffic or more reuse**. This is a research requirement, not a proof that every possible Gemma execution algorithm is physically incapable of 100 tok/s, and not a promise that CF can attain it.

Keep four categories distinct: storage reads (`/proc/PID/io read_bytes`), DRAM transactions (not measured here), CPU cache traffic (not measured here), and modeled logical tensor accesses. Zero storage I/O only supports a storage-residency observation. The old 26B “2.7 GB/token” handover estimate is also superseded by the existing tensor arithmetic in `results/pi-real/x4-resident-bottleneck/existing_evidence.json`: 3,044,428,856 logical bytes for UD-Q3_K_M versus 2,380,055,608 for QAT Q4_0, under that script's coarse assumptions. Neither is a DRAM measurement.

## Timing definitions and what was generated

`pi/adhoc/par_throughput.py` starts all request threads, joins all of them, and divides their total `completion_tokens` by this common wall interval. The E2B sixteen-stream record is **1536 / 46.01 = 33.38 model tokens/s**. Its 2.48 per-stream rate comes separately from the server's `predicted_per_second`. The approximate decode interval implied by 96 / 2.48 is 38.71 s; common-wall per-stream throughput is 96 / 46.01 = 2.09 tok/s. Prompt processing and other request work account for the different windows. Individual wall latencies and TTFT were not retained, so their distribution cannot be reconstructed from this aggregate record. The archive rounded wall time to two decimals.

| Historical E2B run | Server decode model tok/s | Full-request model tok/s | Evidence / limitation |
|---|---:|---:|---|
| llama-bench pp512/tg128 | 7.68 ± 0.10 | — | synthetic random tokens; pp512 61.95 ± 0.33 |
| ik llama-bench mmap | 7.90 ± 0.07 | — | synthetic; pp512 53.13 ± 0.76 |
| Ordinary server, 12 MTP A/B control requests | median 7.015 | median 6.79 | 128 generated tokens/request |
| MTP n_max 3, 12 requests | median 14.747, range 10.912–16.145 | median 13.73 | proposals and accepted proposals recorded separately |
| MTP n_max 6, two requests only | 7.751–7.956 | 7.48–7.76 | interrupted; no complete comparison |

`results/pi-real/speed-ceiling/{par_e2b_ours,par_26b_ours}/sample_output.txt` each records an assistant response containing only reasoning. The table's rates therefore describe model generation, not completed answers or first visible answers. Hidden reasoning is neither silently discarded nor counted as visible product output.

MTP values come from `e2b_fast.out`. `mtp_ab.py` asks for greedy generation but only prints timing and acceptance fields during execution, keeps at most 2,000 content characters and a reasoning-character count in memory, and writes `ab.json` after all configurations. That file is absent from the local retained morning tree following the stop. Even a matching text prefix would not prove all output token IDs, logits, or sampling-distribution equivalence. Correct distribution-preserving speculation requires target verification with the appropriate acceptance algorithm; that theorem does not certify these incomplete local records. [Original speculative-decoding paper](https://arxiv.org/abs/2211.17192).

## Existing preparation and first projection selection

The historical morning audit found no retained operator-time profile: X4 contained tensor-byte arithmetic and end-to-end records. The new CF reference profiles now provide measured selection evidence for **Gemma 4 26B-A4B Google QAT Q4_0**, separate from the earlier 3-bit model results. Both use four threads, the same 24-token calibration prompt, and 21 evaluated decode tokens (22 sampled tokens including EOS). Their named combined gate/up projections are the larger measured MoE projection family in both phases. The initial `blk.0.ffn_gate_up_exps.weight` path is a bounded instance of that family, not the slowest individual layer. [Machine-readable summary and source hashes](../../results/cf-engine/initial-projection-selection.json).

| Initial reference profile | Phase | All 30 named gate/up operations | Share of instrumented graph time |
|---|---|---:|---:|
| Pi `pi-profile-1/profile.json` | Prefill | 71,966.699 ms | 54.498% |
| Pi `pi-profile-1/profile.json` | Decode | 16,982.314 ms | 49.102% |
| Desktop `desktop-profile.json` | Prefill | 739.493 ms | 41.508% |
| Desktop `desktop-profile.json` | Decode | 489.190 ms | 22.110% |

These are **initial storage-active profiles, not repeated warm benchmarks or isolated arithmetic timings**. Both loading logs disable warmup. The Pi process records 24,010,194,432 cumulative storage-read bytes and 13,870 major faults; desktop records 14,447,054,848 bytes and 118 faults. Those counters include loading and cannot attribute I/O to prefill versus decode. Decode following prefill does not prove residency. Instrumented wall times include worker barriers and fused-group attribution. The old profiles lack `weight_name`: unnamed `MUL_MAT_ID` nodes remain unidentified here, and the output head cannot be inferred from `result_output` (a final scaling node). Later weight-aware profiles and alternating warm runs must determine actual optimization headroom. Nothing in this table measures DRAM traffic or visible-answer speed.

The reference `ggml_compute_forward_mul_mat_id()` in `ggml/src/ggml-cpu/ggml-cpu.c` already quantizes each source activation row once before iterating experts, groups token work by expert using `matrix_rows`, and accesses the prepared row using `id % ne11`. Gate/up receives a shared input row; the combined gate/up weight tensor already exists. Down projections use their own post-nonlinearity activation rows. CF must expose an additional packing, tiling, scheduling or arithmetic benefit, rather than claiming baseline input sharing as a new optimization.

Pi tiling starts from 512 KB L2 per Cortex-A76 core and 2 MB shared L3, then requires board measurements. Do not infer i8mm/SVE/SME support from generic ARM64 builds; the archived board probe reports dotprod and FP16, with no i8mm/SVE. [Raspberry Pi 5 product brief](https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf).

## Alternatives: actual status, not aspirational coverage

| Target | Status | Retained evidence and limit |
|---|---|---|
| Known-good llama.cpp `a7b94df` | **Locally tested** | Pi resident baseline; `results/pi-real/speed-ceiling/`. Keep QAT and non-QAT models separate. |
| Compatible upstream `8345f33` | **Locally tested** (desktop functionality) | `results/comparisons/x1a/README.md`: two tuning prompts, 64 generated IDs and printed top-5 probabilities identical; CPU-kernel source unchanged. No matched Pi performance result. |
| ik_llama.cpp `3d27f5b` | **Locally tested** | E2B morning data and `results/comparisons/ik_llama_pi/X2_RESULT.json`: 26B resident, four held-out requests, three alternated repeats. Different greedy outputs; quality parity unresolved. The eight-slot morning 26B server exit is a separate **blocked configuration**, not evidence of general incompatibility. |
| ik expert prefetch | **Partially integrated** | `results/comparisons/x1c/e2-prefetch-8g/check.json`: desktop cgroup-v1 behavior check, unthrottled storage, instrumented; not a benchmark. |
| Arm KleidiAI | **Locally tested at operator level; partially integrated** | CF directly executed pinned v1.24.0 DOTPROD kernels on captured 26B Q4_0 experts; three independent 1/2/4-thread comparisons. See `KLEIDIAI.md` and `results/cf-engine/operator-{1,2,3}.jsonl`. Upstream integration excludes MUL_MAT_ID; no whole-model Kleidi claim. |
| Microsoft T-MAC | **Documentation-only** | LUT/bit-plane mechanism checked. No same-Gemma4-artifact conversion and execution verified locally. Include input-table preparation, scales, conversion and scratch when comparing. |
| HotPin | **Partially integrated** | `results/comparisons/x1b/README.md`: desktop output/pinning checks, one 8 GiB cgroup smoke, aarch64 emulation; no real-Pi benchmark. Source remains isolated under `third_party/`. |
| LiteRT-LM | **Documentation-only** | Official Raspberry Pi/Linux ARM64 support checked; no local CPU `.litertlm` test. Different artifact and quantization must be documented as a product comparison. |

The historical ik matched-file result gives useful headroom context: roughly 2.86x prefill at an 884-token prompt, near parity in short-context decode, and 15% faster decode after that longer prompt. It does not establish a QAT Q4_0 result or equal quality. The archived report explicitly notes numerical output differences and incomplete quality checks.

[ik's repository](https://github.com/ikawrakow/ik_llama.cpp) documents Gemma4 and NEON paths. [KleidiAI](https://github.com/ARM-software/kleidiai) supplies microkernels; [llama.cpp's build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#arm-kleidiai) explicitly limits claims from enabling it. [T-MAC](https://github.com/microsoft/T-MAC) directly computes mixed-precision products through lookup tables and supports multiple low-bit formats; that is a mechanism to compare, not proof of compatible GGUF block semantics.

[HotPin's repository](https://github.com/LozzKappa/hotpin-llm) declares AGPL licensing. Existing local checks additionally found a prompt-routing stride problem, budget labels that did not equal locked bytes, and anonymous repacked pages when used with default repacking. Preserve the existing isolation and compare actual locked physical pages. No HotPin code was incorporated by this audit.

[LiteRT-LM's repository](https://github.com/google-ai-edge/LiteRT-LM) has a Gemma E4B example explicitly using `--backend=gpu`; that example is not a CPU measurement. A future test must select CPU in the [CLI](https://developers.google.com/edge/litert-lm/cli/usage), verify runtime dispatch, and record the `.litertlm` model and quantization identity.

## Memory and research boundaries

The archived Pi boot command line has `cgroup_disable=memory`; its memory controller is unavailable. Recheck live before a constrained experiment. Until corrected through an explicitly authorized device change or an isolated supported environment, **no valid 8 GiB Pi application-budget result exists**. Earlier balloons are historical stress tests, not matched budget tests; do not restart them. Desktop cgroup-v1 + FUSE work is separately labeled, not Pi performance.

Start executor and helpers inside the budget before loading, then audit page ownership. Moving a PID after loading does not move earlier memory charges. Shared file pages can retain their original charge. Count canonical/packed/source cache copies, KV, scratch and metadata jointly. [Linux cgroup memory ownership](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-ownership). An 8 GiB application cap on this 16 GB host is not an 8 GB whole-device claim.

The optional base-plus-correction module remains a later bounded research path. [SVD-LLM](https://arxiv.org/abs/2403.07378) already studies low-rank model compression. Neither modeled logical bytes nor isolated layer error establishes a useful dynamic correction selector. Measure output-head runtime share before investing, keep calibration separate from held-out evaluation, validate all-correction reconstruction, and charge selector work and correction metadata. No algorithmic traffic reduction, energy result or arbitrary-sampling certificate was established by this audit.
