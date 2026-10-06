# pi-gemma4: can a full Gemma 4 run on a Raspberry Pi 5?

**Status 2026-10-05.** llama.cpp is pinned at commit a7b94df (build b11398) on every host. Everything
runs on CPUs: no GPU, no NPU, no cloud, no hidden big machine.

the operator (TamerinTECH) asked whether a *full* Gemma 4 large model can be downloaded and run on a Raspberry
Pi 5 "with good speed and quality". This repository is the measurement harness built to answer that.
**The question is the project; a "yes" is not assumed anywhere.** Every number in this repository
carries one of three labels:

| Label | Meaning |
|---|---|
| **MEASURED** | Observed on named hardware, with the result file that holds it. |
| **PROJECTED** | Computed by `pigemma.feasibility` or `pigemma.cache_sim` from listed inputs; not observed. |
| **ASSUMED** | A constant nobody has measured on the target yet (catalog `source` strings start with `assumed:`), or a literature prior for a different model. |

"Good speed" has a fixed meaning here, chosen before the data: **≥ 2.0 tok/s is good and ≥ 1.0 tok/s
is viable** (`selection.good_stream_tok_s` and `selection.viable_stream_tok_s` in
[pigemma/data/catalog.json](pigemma/data/catalog.json); a policy choice, not a measurement).

## The answer so far

| Question | Answer on 2026-10-05 | Kind |
|---|---|---|
| Does the 26B-A4B MoE run on a Pi 5? | **Yes, on a 16 GB board, from RAM.** 3.17–3.27 tok/s generation with the 3-bit file (UD-Q3_K_M), up to 4.14 tok/s with Google's QAT Q4_0 file, 9.1–9.3 tok/s prompt processing, correct English, Hebrew and German answers | MEASURED on `context-friend-1` ([results/pi-real/README.md](results/pi-real/README.md)) |
| Does the dense 31B run at good speed? | **No.** 0.53 tok/s with the 3-bit file held in RAM, and MTP speculative decoding halved that on the Pi. The Q4_0 file does not fit in 16 GB of RAM, and streaming it costs about 21 s per token | MEASURED and PROJECTED |
| Which Pi 5 configuration should be bought? | **Pi 5 16 GB + NVMe Gen3 x1 + Active Cooler.** The 26B Q4_0 file fits in RAM with `-nr` and loads in about 17 s | PROJECTED: 2.73 tok/s |
| Can an 8 GB Pi stream the experts from storage (H1)? | Projected 1.94 tok/s on NVMe Gen3 with a literature hit-rate curve. Not measured. A measured proxy on microSD (7.5 GB usable RAM) ran at 0.55–0.79 tok/s, storage-bound | PROJECTED; MEASURED proxy |
| Can a cheap base plus streamed residuals rescue the dense 31B (H2)? | The pre-registered expectation is NOT_SUPPORTED. The quick E7 run (2 layers) says **NO_ADVANTAGE**: the oracle needs 96% of the residual bytes at 3% error | MEASURED (quick run); the full E7 is pending |
| Does an AI HAT help? | No. Hailo accelerators cannot hold or run a 25–30B model, and the HAT takes the PCIe lane the NVMe needs | from the research pass ([docs/research/RESEARCH_SUMMARY.md](docs/research/RESEARCH_SUMMARY.md)) |

Most desktop experiments (E0 to E6 and the full E7) have not run yet. Their thresholds are fixed in
[docs/HYPOTHESES.md](docs/HYPOTHESES.md) and [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md); the
[status table](#honest-status-measured--projected--assumed) below says what exists today.

**Non-goals.**

- No GPU, NPU or cloud inference. AI HATs were evaluated and rejected (see [Hardware options](#hardware-options)).
- No training, fine-tuning, distillation or new quantisation method. Only published GGUF files are used:
  Google's QAT Q4_0 files and unsloth's lower-bit quants.
- No patched llama.cpp. Everything runs on stock a7b94df. The custom native code is three small tools
  (a routing/activation probe, an mlock pinner and a memory-bandwidth benchmark).
- No speed claim without a measurement. Projections are labelled as projections.
- Not a chat product. The Pi scripts can run a chat or a server, but the deliverable is the measurement.

## The two models and their byte budgets

On a CPU, generating one token means reading every weight that token uses from memory once. Bytes per
token therefore set the speed limit. The byte counts below are **MEASURED** from the GGUF headers by
[pigemma/gguf_map.py](pigemma/gguf_map.py) ([results/map26.json](results/map26.json),
[traces/h2/map31.json](traces/h2/map31.json)) and stored in
[pigemma/data/catalog.json](pigemma/data/catalog.json). `tests/test_docs.py` checks every row against
the catalog field named in it.

**Gemma 4 26B-A4B (MoE, 25.23B parameters, 3.82B active), Google QAT Q4_0.** 30 layers, each with a
dense FFN (the "shared expert") and 128 routed experts, of which the router picks 8 per token.

| Part | Catalog field | Bytes | Size |
|---|---|---:|---:|
| GGUF file | `models.26b-q4.bytes` | 14,439,363,584 | 14.44 GB |
| Non-expert weights, read for every token | `models.26b-q4.nonexpert_bytes` | 1,577,172,088 | 1.577 GB |
| … of which the tied Q6_K head (`token_embd`; the whole matrix is read for the logits) | `models.26b-q4.breakdown_bytes.head_tied_token_embd_q6_k` | 605,563,904 | 0.606 GB |
| … of which attention | `models.26b-q4.breakdown_bytes.attn` | 624,885,760 | 0.625 GB |
| … of which the dense FFN (shared expert) | `models.26b-q4.breakdown_bytes.dense_ffn_shared_expert` | 301,086,720 | 0.301 GB |
| … of which router, norms and expert scales | `models.26b-q4.breakdown_bytes.router_norms_scales` | 45,635,704 | 0.046 GB |
| One expert in one layer (gate_up + down slab) | `models.26b-q4.expert_unit_bytes` | 3,345,408 | 3.35 MB |
| All experts (30 layers × 128) | `models.26b-q4.expert_bytes` | 12,846,366,720 | 12.85 GB |
| Expert bytes per token, nothing cached (30 × 8 × 3,345,408) | `models.26b-q4.expert_bytes_per_token_worst` | 802,897,920 | 803 MB |
| **Total per token with nothing cached** | `models.26b-q4.nonexpert_bytes_per_token + models.26b-q4.expert_bytes_per_token_worst` | 2,380,070,008 | **2.38 GB** |
| Real-Pi file: unsloth UD-Q3_K_M | `models.26b-ud-q3_k_m.bytes` | 12,728,497,888 | 12.73 GB |
| … its bytes per token (attention and embeddings stay at higher precision) | `models.26b-ud-q3_k_m.nonexpert_bytes_per_token + models.26b-ud-q3_k_m.expert_bytes_per_token_worst` | 3,044,443,256 | 3.04 GB |

**Gemma 4 31B (dense, 30.70B parameters), Google QAT Q4_0.** 60 layers. Every token reads every weight.

| Part | Catalog field | Bytes | Size |
|---|---|---:|---:|
| GGUF file | `models.31b-q4.bytes` | 17,651,001,568 | 17.65 GB |
| **Read per token (every tensor)** | `models.31b-q4.nonexpert_bytes_per_token` | 17,635,166,448 | **17.64 GB** |
| … of which the FFN (60 layers × 3 matrices × 65,028,096 B) | `models.31b-q4.breakdown_bytes.ffn` | 11,705,057,280 | 11.71 GB |
| … of which attention | `models.31b-q4.breakdown_bytes.attn` | 4,770,160,640 | 4.77 GB |
| … of which the tied Q6_K head | `models.31b-q4.breakdown_bytes.head_tied_token_embd_q6_k` | 1,156,076,544 | 1.16 GB |
| Attention + head, which H2 does not touch | `models.31b-q4.breakdown_bytes.attn + models.31b-q4.breakdown_bytes.head_tied_token_embd_q6_k` | 5,926,237,184 | 5.93 GB |
| Real-Pi file: unsloth UD-IQ3_XXS | `models.31b-ud-iq3_xxs.bytes` | 11,837,782,976 | 11.84 GB |
| … its bytes per token | `models.31b-ud-iq3_xxs.nonexpert_bytes_per_token` | 11,821,947,120 | 11.82 GB |

What follows from the tables (PROJECTED with the ASSUMED Pi 5 constants of 10.0 GB/s effective RAM
bandwidth × 0.65 llama.cpp efficiency): the 26B is capped at 6.5 GB/s / 2.38 GB = 2.73 tok/s even with
every expert in RAM, and the 31B at 6.5 GB/s / 17.64 GB = 0.37 tok/s even if it fit. The real Pi already
beat the 26B projection (3.17–3.27 tok/s measured against 2.14 projected for the 3-bit file, roughly
9.7 GB/s of weight traffic), so the assumed efficiency is too pessimistic for that kernel; `pi/03` and
`pi/05` replace both constants with measured ones. The 31B's 3-bit file came in at its projection
(0.53 measured, 0.55 projected). Full derivations: [docs/HYPOTHESES.md, section 2](docs/HYPOTHESES.md#2-bytes-per-token-the-quantity-everything-depends-on)
and [section 3.4](docs/HYPOTHESES.md#34-the-time-model-shared-by-h1-the-dense-line-and-feasibility).

## The one llama.cpp fact everything depends on

**By default, llama.cpp a7b94df copies the Q4_0 weights out of the memory-mapped file.** To use its
faster CPU kernels it *repacks* every Q4_0 tensor into a `CPU_REPACK` buffer, which is anonymous memory
almost the size of the file. Only tensors whose buffer type is the device's default buffer are served
straight from the mmap (`third_party/llama.cpp/src/llama-model.cpp:1849-1870`; the `is_default_buft`
test is at line 1856). That decides what can run where:

1. **With repack, a weight file has to fit in anonymous memory.** For the 26B Q4_0 that is about 14 GB
   that the kernel can neither drop nor re-read: on an 8 GB board the load ends in the OOM killer, and
   on a 16 GB board it does not fit beside the OS (`pigemma.feasibility` requires file + 0.3 GB ≤
   usable − 1.0 GB; for the catalog's 16 GB profile that is 14.74 GB ≤ 13.54 GB, which fails).
2. **`-nr` / `--no-repack` (`--repack 0` in llama-bench) is the only mode that keeps the weights
   file-backed** (`CPU_Mapped`, i.e. page cache that Linux can evict and re-read). It is therefore the
   only page-cache streaming mode at this commit, and the mechanism H1 relies on. Its cost is kernel
   speed, because the repacked kernels are the fast ones. E0 measures that penalty on the desktop and
   `pi/05` measures it on the Pi.
3. **`-ot "ffn_(gate_up|down)_exps=CPU"` does not keep the experts mapped.** An override to the CPU
   buffer type re-runs `select_weight_buft` over `buft_list_cpu`, which still contains `CPU_REPACK`
   (`third_party/llama.cpp/src/llama-model-loader.cpp:1240-1242`), and `-ot` can only name the devices'
   default buffer types (`third_party/llama.cpp/common/arg.cpp:255-260`). It is kept only as the
   negative control `hybrid_negctrl`.
4. Load modes at this commit are `-lm auto|none|mmap|mlock|mmap+mlock|dio`; `--mlock` and `--no-mmap`
   are gone. `-lm none` reads the whole file into anonymous memory, with the same OOM risk as repack.

A related detail for the routing probe: warm-up routes every token to *all* experts
(`third_party/llama.cpp/src/llama-graph.cpp:1473`), so `pigemma-probe` sets `params.warmup = false`.

These are source-reading facts, verified on 2026-10-05; `tests/test_docs.py` re-checks that the cited
lines still say so. **MEASURED** on the real Pi with `-nr`: the 31B UD-IQ3_XXS ran with 12.58 GB of
file-backed page cache and about 0.44 GB of anonymous memory; the 26B UD-Q3_K_M used 0.52 GB anonymous
(0.99 GB after an 821-token context) ([results/pi-real/README.md](results/pi-real/README.md)). The
formal desktop gate (E0: `CPU_Mapped` ≥ 95% of the file, no `CPU_REPACK` line, private peak < 1.5 GB,
and the negative control repacks) ran on 2026-10-05. It **passed for the 26B**: 99.9% mapped, no
`CPU_REPACK` line, 1,236 MB private, and `-ot` repacks 13,134 MiB. It **failed for the 31B on the
private peak only**: 2,533 MB, of which the f16 KV cache at `-c 4096` is 1,520 MiB; no weights were
copied. The `-nr` cost on the desktop is −12 to −17% decode at 4 threads (26B) and −30 to −45% prefill
([docs/results/E0-desktop-baseline-and-repack-gate.md](docs/results/E0-desktop-baseline-and-repack-gate.md)). The flag table
per verdict is in [docs/PI_RUNBOOK.md, section 8](docs/PI_RUNBOOK.md#8-llamacpp-flags-per-verdict-and-why-repack-is-refused).

## Honest status: measured / projected / assumed

The desktop is a Ryzen 5 8400F (6 cores, AVX-512), 64 GB DDR5, no GPU. The real Pi is `context-friend-1`:
Pi 5, 16 GB, Cortex-A76 ×4, 16 KiB pages, microSD only, Active Cooler, production services running.
`_pending_` means not run yet. A dash means the column does not apply.

| Quantity | MEASURED (host, date, file) | PROJECTED (inputs) | ASSUMED (value, source) |
|---|---|---|---|
| Desktop 26B decode, llama-bench tg32 | 15.18 tok/s at 4 threads, 13.52 at 6 (desktop, 2026-10-04, [baseline](results/desktop-baseline-2026-10-04.md)) | — | — |
| Desktop 31B decode, llama-bench tg32, 6 threads | 2.44 tok/s (desktop, 2026-10-04, [baseline](results/desktop-baseline-2026-10-04.md)) | — | — |
| Desktop llama-server generation, 6 threads | 26B 16.03 tok/s, 31B 2.40 tok/s (desktop, 2026-10-05, [baseline](results/desktop-baseline-2026-10-04.md)) | — | — |
| Speculative decoding on the desktop (E1) | MTP drafts only with an explicit `--spec-type draft-mtp`: acceptance 60–69% (26B) and 39–63% (31B). DFlash: 28% acceptance, generation 11.83 vs 16.03 tok/s, i.e. slower ([baseline](results/desktop-baseline-2026-10-04.md)). Speed with MTP on an idle machine: _pending_ | — | — |
| E0 repack gate and the `-nr` speed penalty | Gate PASS 26B, FAIL 31B (private peak 2,533 MB = KV cache + compute buffer, no weight copy). `-nr` decode: 26B 0.83–0.88× at 4 threads and 0.92–0.96× at 6; 31B 0.86–0.90× at 6. Prefill pp128: 0.55–0.70×. Background CPU was 12–62% (desktop, 2026-10-05, [E0](docs/results/E0-desktop-baseline-and-repack-gate.md)) | — | penalty 10–40% (HYPOTHESES 3.2) |
| E2 quality: KLD of `-nr` vs default, Q&A on the desktop | _pending_ (`pc/desktop_bench.ps1 -Stage quality`) | — | — |
| E3 routing trace | smoke trace valid: 72 records (64 prefill + 8 generated), 30 MoE layers, coherent text (desktop, Docker, 2026-10-05, [meta](traces/26b/smoke/en_tech_moe_streaming.moetrace.meta.json)). The `--quick` trace: _pending_ | — | LRU hit vs cache fraction 0.15 → 0.60, 0.25 → 0.74, 0.35 → 0.84, 0.45 → 0.88 (literature prior, catalog `selection`) |
| E4–E6 expert cache, page-cache proxy, pinning | _pending_ | 8 GB + NVMe Gen3: 1.94 tok/s serial, 2.87 overlapped (feasibility, prior hit 0.861) | NVMe Gen3 850 MB/s sequential, 700 MB/s random 3 MB, 0.2 ms |
| 26B UD-Q3_K_M (3-bit) on the 16 GB Pi, from RAM | 3.17–3.27 tok/s generation, 9.1–9.3 tok/s prompt, 148 s cold load from microSD, 0 bytes read from storage during inference (context-friend-1, 2026-10-05, [pi-real](results/pi-real/README.md)) | 2.14 tok/s (feasibility); E8 band 1.7–3.0 tok/s, so the measurement lies above it | 10.0 GB/s × 0.65 efficiency |
| 26B QAT Q4_0 on the same Pi (USB stick, not fully resident beside the services) | 2.36–4.14 tok/s generation, 125 s load (context-friend-1, 2026-10-05, [pi-real](results/pi-real/README.md)) | 2.73 tok/s when fully resident | 10.0 GB/s × 0.65 efficiency |
| 26B UD-Q3_K_M with 7.5 GB usable RAM (an 8 GiB mlocked balloon, about an 8 GB board) | 0.55–0.79 tok/s, storage-bound on the microSD (context-friend-1, 2026-10-05, [pi-real](results/pi-real/README.md)) | — | — |
| MTP speculative decoding on the Pi (26B, interleaved A/B) | mean 3.18 → 3.00 tok/s (−6%); code +34%, Hebrew −29% (context-friend-1, 2026-10-05, [pi-real](results/pi-real/README.md)) | — | — |
| MTP speculative decoding on the Pi (31B UD-IQ3_XXS, A/B) | 0.508 → 0.248–0.276 tok/s, i.e. halved; acceptance 24–31% with the drafter built for the QAT checkpoint (context-friend-1, 2026-10-05, [pi-real](results/pi-real/README.md)) | — | — |
| Q&A on the Pi (36 items: EN/HE/DE facts, numbers, extraction, code) | 35/36 (UD-Q3_K_M) and 34/36 (QAT Q4_0) strict; every miss was thinking-channel markup, not a wrong answer (context-friend-1, 2026-10-05, [pi-real](results/pi-real/README.md)) | — | — |
| 31B UD-IQ3_XXS (3-bit) on the 16 GB Pi, from RAM | 0.53 tok/s generation, 0.63 tok/s prompt, 138 s cold load (context-friend-1, 2026-10-05, [pi-real](results/pi-real/README.md)) | 0.55 tok/s | 10.0 GB/s × 0.65 efficiency |
| 31B QAT Q4_0 on any Pi 5 | — (does not fit 16 GB of RAM) | 20.8 s/token from NVMe Gen3; 0.37 tok/s if it were in RAM | 850 MB/s; 10.0 GB/s × 0.65 |
| 26B Q4_0 on the recommended Pi 5 16 GB + NVMe Gen3 | _pending_ (no such board yet) | 2.73 tok/s, about 17 s load | 10.0 GB/s × 0.65; 850 MB/s |
| Pi memory bandwidth and llama efficiency | _pending_ (`pi/03` membw, `pi/05` calibration) | — | 10.0 GB/s (catalog); 0.65 (`feasibility.LLAMA_EFFICIENCY`) |
| Pi storage | microSD 91.6 MB/s sequential, O_DIRECT 4 MiB (context-friend-1, 2026-10-05, [pi-real](results/pi-real/README.md)) | — | sequential / random 3 MB: microSD 85 / 60, USB 3.0 SSD 380 / 350, NVMe Gen2 430 / 400, NVMe Gen3 850 / 700 MB/s (catalog `storage_classes`) |
| E7 H2 oracle, quick run (layers 20 and 45, 128 tokens, q2 base) | **NO_ADVANTAGE**: the oracle needs 0.964 of the residual bytes at ε = 3%, certification efficiency 0.070, pipeline VALID (sanity 0.0053 and 0.0062) (desktop, 2026-10-05, [H2 quick](docs/results/H2_RESULTS_quick.md)). Full E7: _pending_ | 23.34 GB/token implied (extrapolated from 2 of 60 layers) | NVMe Gen3 850 MB/s gives 27.5 s/token |
| Emulated arm64 Pi build (qemu) | functional only: llama.cpp a7b94df builds and runs for aarch64 in Debian 13 ([pi-emu](results/pi-emu/README.md)); no speed numbers | — | — |

The per-experiment templates (with verdict lines) are in [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md),
and the Pi-only status table is [docs/PI_RUNBOOK.md, section 13](docs/PI_RUNBOOK.md#13-status-on-the-pi-measured-projected-assumed).

## Hardware options

the operator's note: "for the pi, we can consider other setups: more RAM, NVMe drives, AI HAT, etc." The table
gives the **PROJECTED** tok/s and verdict of `pigemma.feasibility` for each Pi 5 configuration, using the
**ASSUMED** catalog constants (10.0 GB/s × 0.65; storage speeds as listed; the prior LRU hit curve for
expert streaming). 26B means the 26B-A4B QAT Q4_0 file (14.44 GB), 31B the dense QAT Q4_0 file
(17.65 GB). The dense column is the whole file divided by the sequential read speed, because the file
does not fit in RAM. **MEASURED** values come from `context-friend-1`.

| Pi 5 RAM | Model storage (sequential / random 3 MB) | 26B-A4B Q4_0 | 31B dense Q4_0 | Verdict |
|---|---|---|---|---|
| 4 GB | microSD (85 / 60 MB/s) | 0.11 NOT_VIABLE | 207.7 s/token | not viable for either model |
| 4 GB | USB 3.0 SSD, NVMe Gen2, NVMe Gen3 | 0.57, 0.65, 0.98 STREAM_SLOW | 46.5, 41.0, 20.8 s/token | not viable: about 1.3 GB of expert cache is too small |
| 8 GB | microSD | 0.44 STREAM_SLOW | 207.7 s/token | not viable. MEASURED proxy: 0.55–0.79 tok/s for the 3-bit 26B with 7.5 GB usable RAM |
| 8 GB | USB 3.0 SSD (380 / 350 MB/s) | 1.46 STREAM_OK | 46.5 s/token | the MoE streams experts, slowly |
| 8 GB | NVMe Gen2 x1 (430 / 400 MB/s, the default link) | 1.57 STREAM_OK | 41.0 s/token | the MoE streams experts |
| 8 GB | NVMe Gen3 x1 (850 / 700 MB/s; `dtparam=pciex1_gen=3`, uncertified) | 1.94 STREAM_OK | 20.8 s/token | the MoE streams experts with about 40% of them cached; **this is where H1 matters** |
| 16 GB | microSD (91.6 MB/s MEASURED on `context-friend-1`) | 2.73 FITS_MAPPED; MEASURED 3.17–3.27 tok/s with the 3-bit file | 3-bit file in RAM: 0.53 tok/s MEASURED | works today. The Q4_0 26B does not fit a 30 GB card with 2 GB kept free, so 3-bit files are used |
| 16 GB | USB 3.0 SSD or NVMe Gen2 | 2.73 FITS_MAPPED | 46.5, 41.0 s/token | good; the drive only shortens the load (about 38 s from a USB 3.0 SSD) |
| 16 GB | NVMe Gen3 x1 + Active Cooler | 2.73 FITS_MAPPED | 20.8 s/token; a 3-bit 31B in RAM runs at about 0.5 tok/s | **recommended**: the 26B-A4B fits in RAM with `-nr` and loads in about 17 s |
| any | + AI HAT (Hailo-8/8L; AI HAT+ 2 with Hailo-10H) | no change | no change | **does not help.** Hailo-8/8L have no DRAM and no LLM support. Hailo-10H handles at most about 2B models, decodes more slowly than the Pi CPU and has no llama.cpp backend. The HAT also takes the only PCIe lane, so an NVMe then needs a dual-slot HAT |
| 2 or more Pis | llama.cpp RPC | — | — | slower than one 16 GB Pi in published measurements |

The 16 GB rows are RAM-bound: storage only changes the load time (14.44 GB at 850 MB/s ≈ 17 s, at
85 MB/s ≈ 170 s). An Active Cooler is required either way: the Pi 5 throttles from 80 °C, and the real
Pi peaked at 72.5 °C over 23 minutes of decode with the cooler (`throttled=0x0`). Price: the Pi 5 16 GB
cost $120 at launch and about $305 by 2026-04, after memory prices rose. The full table with notes,
including the 3-bit quants that fit smaller cards, is in
[docs/PI_RUNBOOK.md, section 3](docs/PI_RUNBOOK.md#3-hardware-options); the arithmetic is in
[docs/HYPOTHESES.md, section 3.6](docs/HYPOTHESES.md#36-ram--storage-grid-projected-by-feasibilityverdict-with-the-prior-hit-curve).

## Hypotheses H1 and H2

**H1, MoE "small active set" (Gemma 4 26B-A4B).** Keep the 1.58 GB of non-expert weights and an LRU
cache of expert slabs (3.35 MB each) in RAM, and read missed experts from storage on demand. The claim:
on a Pi 5 with 8 GB and NVMe Gen3, a 5.16 GB expert cache (40.2% of the experts) gives **≥ 1.0 tok/s
serial and ≥ 2.0 tok/s with I/O overlapped**. That needs an **LRU hit rate ≥ 0.582** on Gemma 4's own
routing, and it needs the real page cache to behave like the simulator. It is tested by E3 (routing
skew on real EN/HE/code text, `pigemma-probe` traces), E4 (cache simulation), E5 (a Docker page-cache
proxy under memory caps), E6 (mlock-pinning the hot set) and finally `pi/05` on hardware. H1 is
falsified for the 8 GB tier if E4's LRU hit at 5.16 GB is below 0.408 (not even viable). The 16 GB
tier does not need H1 at all: the whole model fits in RAM. Details:
[docs/HYPOTHESES.md, section 4](docs/HYPOTHESES.md#48-pre-registered-thresholds-fixed-before-the-data).

**H2, dense "staged residuals" (Gemma 4 31B).** Write each FFN matrix as `W = W_base + Σ_g R_g`, with
groups g of 128 input columns. Keep a cheap base in RAM, and read a residual group only when it could
change the output: `‖R_g x_g‖ ≤ ‖R_g‖₂ · ‖x_g‖₂` lets the runtime skip a group without reading it. An
oracle experiment comes first (E7): on captured real activations, how much of the residual does a
token *actually* need for a 3% relative output error? The claim is **PROMISING** only if, at ε = 3%,
the oracle needs **≤ 25%** of the residual bytes, the norm bound certifies **≥ 50%** of what the oracle
skips, implied traffic is **< 8.5 GB/token** (the 31B IQ2_XXS file) and the result beats uniform
quantisation at equal bytes. The pre-registered expectation was NOT_SUPPORTED: with a q2 base the
traffic is 12.43 GB/token even with no residual read at all. The quick E7 run measured
**NO_ADVANTAGE** (0.964 of the residual bytes needed, certification efficiency 0.070). Details:
[docs/HYPOTHESES.md, section 5](docs/HYPOTHESES.md#510-pre-registered-thresholds-e7) and
[docs/results/H2_RESULTS_quick.md](docs/results/H2_RESULTS_quick.md).

Pre-registered thresholds (fixed on 2026-10-05, before the data; changes only through the
[deviations log](docs/EXPERIMENT_PLAN.md#deviations-log)):

| Exp. | Threshold | Status |
|---|---|---|
| E0 | `-nr` keeps `CPU_Mapped` ≥ 95% of the file with no `CPU_REPACK` line and private peak < 1.5 GB, and the `-ot` negative control repacks. A failure stops all streaming work | 26B **PASS**; 31B **FAIL** (private peak 2,533 MB from KV + compute, weights not copied); speed band PASS for both ([E0](docs/results/E0-desktop-baseline-and-repack-gate.md)) |
| E2 | `-nr` is numerically neutral: KLD mean < 0.001 and same-top-p > 99.5% | _pending_ |
| E3 | Strong skew: mean coverage@32 ≥ 0.70 or adjacent-token expert reuse ≥ 0.40 | smoke trace only |
| E4 | STREAM_GOOD: LRU tg_overlap ≥ 2.0 and tg_serial ≥ 1.0 tok/s (hit ≥ 0.582 on NVMe Gen3) | _pending_ |
| E5 | Simulated LRU bytes within ±30% of the measured page cache at 4–8 GB caps; time model within ±35% | _pending_ |
| E6 | Pinning: ≥ 1.15 × the unpinned tok/s, or ≥ 20% fewer bytes per token, at the 5 GB cap | _pending_ |
| E7 | PROMISING as defined above; NO_ADVANTAGE if uniform q3 error ≤ 1.2 × the oracle error at equal bytes | quick run: NO_ADVANTAGE |
| E8 | Real Pi, 26B UD-Q3_K_M tg inside 1.7–3.0 tok/s means "projection holds" | ad hoc 3.17–3.27: above the band (bandwidth better than assumed); the formal `pi/05` run is pending |

## Plain statements

- **The dense 31B cannot reach good speed on any Raspberry Pi 5 with any known post-training technique.**
  It reads every weight for every token: 17.64 GB at Q4_0. From NVMe Gen3 that is about 21 s per token
  (20.8 s PROJECTED); even if it were all in RAM, which on a 16 GB board it is not, it would run at
  0.37 tok/s at the assumed 10.0 GB/s × 0.65 (0.57 tok/s at 100% efficiency). The 3-bit file that does
  fit in RAM was MEASURED at 0.53 tok/s. Attention plus head alone are 5.93 GB/token, which caps the
  dense model at 1.69 tok/s at 10 GB/s even if the FFN cost nothing.
- **H2 is a measurement, not a promise.** It measures how much of a residual a decision really needs.
  It is not a path to good speed for the dense model on a Pi.
- **The MoE is the realistic tier.** It reads 2.38 GB per token instead of 17.64 GB. On a 16 GB Pi 5 it
  fits in RAM with `-nr`, and the real Pi already runs it at 3.17–3.27 tok/s (3-bit) and up to
  4.14 tok/s (QAT Q4_0), above the 2.0 tok/s "good" line.
- **The recommended configuration is a Pi 5 16 GB + NVMe Gen3 x1 + Active Cooler.** It is the
  configuration where the QAT Q4_0 26B fits in RAM with room for the OS and loads in about 17 s.
- **An 8 GB board is where H1 is needed.** Its numbers are projections until E4, E5 and `pi/05` measure them.
- **Speculative decoding is not a free win on the Pi.** On the MoE each drafted token routes to its own
  8 experts, so verification reads more expert bytes. MEASURED on the Pi: +34% on code, −29% on Hebrew,
  −6% on average. On the 3-bit dense 31B it halved the speed (0.508 → 0.248–0.276 tok/s), because the
  IQ3_XXS kernels are compute-bound on the Cortex-A76 (prompt processing 0.62 tok/s, barely above
  generation), so batched verification saves nothing. Whether a Q4_0- or K-quant-class dense file with a
  matched drafter behaves better is open (E1 on the desktop, then the Pi).

## Quick start: desktop (today)

Requirements: Windows with Python ≥ 3.12, numpy, psutil and pyyaml (`requirements.txt`), pytest, Git
Bash, Docker Desktop with the `pi-gemma4-build:latest` image ([docker/Dockerfile.build](docker/Dockerfile.build)),
the llama.cpp a7b94df Windows binaries in `third_party/llama.cpp-win/`, and the models in `models/`
(gitignored; `python -m pigemma.paths model 26b-q4 --require` prints the download command if one is
missing). There is no install step: everything runs as `python -m pigemma.<module>` from the repository
root, and pytest finds the package through `pythonpath` in [pyproject.toml](pyproject.toml).

```powershell
python -m pytest                                    # unit tests: synthetic GGUFs and traces, no models needed
python -m pigemma.paths info                        # models, binaries, host, bench-lock state
python -m pigemma.paths lock --status               # only one model-loading job at a time
.\pc\desktop_bench.ps1 -Stage bench,spec,quality    # E0 repack gate, E1 speculative decoding, E2 quality
.\pc\h2_run.ps1 -Quick                              # E7 quick (2 layers, 128 tokens)
.\pc\h2_run.ps1                                     # E7 in full (4 layers, 512 tokens)
```

```bash
bash pc/docker_run.sh build-tools                   # pigemma-probe, pinner, membw for Linux x86 (Docker)
bash pc/trace_collect.sh --smoke                    # 72-token routing trace, compared with the golden fixture
bash pc/trace_collect.sh --quick                    # E3: routing trace of the whole corpus (about 25 min)
python -m pigemma.gguf_map build 26b-q4 --out results/map26.json --reuse
python -m pigemma.cache_sim traces/26b/all.moetrace --map results/map26.json --profiles all --e0-dir results/<date> --out results/h1/sim.json --csv results/h1/sim.csv
bash pc/pagecache_limit.sh --mem 3,4,5,6,8 --tokens traces/26b/all.moetrace --range 0:1500 --out results/h1/validation.json
bash pc/pagecache_limit.sh --mem 5,6 --pin --skip-copy --tokens traces/26b/all.moetrace --range 0:1500 --out results/h1/validation_pin.json
python -m pigemma.results summarize results/<date> --md docs/results/DESKTOP_<date>.md
```

`pip install -e .` also works, but only inside a private virtual environment, never into the global
interpreter. Each driver holds the bench lock `results/.bench.lock` (it refuses when another model job
holds it or less than 20 GB of RAM is free) and prints free disk first. Speed numbers are only valid
on an idle machine: background CPU below 5% for 10 s, with no other llama process or Docker job
running. The exact command of every experiment, its runtime and its outputs are in
[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) (overview, run order, E0 to E8).

## Quick start: Raspberry Pi 5 (pi/00 to pi/06)

Copy the repository to the Pi ([docs/PI_RUNBOOK.md, section 5](docs/PI_RUNBOOK.md#5-putting-the-repository-on-the-pi)),
adjust [pi/pi_config.env](pi/pi_config.env) and run the steps in order. Every script re-checks the
host and the verdict before doing anything heavy, and refuses rather than guesses.

```bash
bash pi/00_feasibility.sh                    # read-only verdict: exit 0 or 2 = go on, 3 = stop, 4 = refused
bash pi/01_tune_os.sh                        # dry run: prints the OS tuning plan; --apply needs root
nohup bash pi/02_build_llama.sh --with-tools > pi/out/build.out 2>&1 &
bash pi/03_bench_io_mem.sh --quick           # measured storage + memory -> results/pi/io_mem.json
bash pi/04_fetch_model.sh --dry-run          # what would be downloaded, how big, how long
bash pi/04_fetch_model.sh --bg               # the download, detached, resumable, sha256-verified
bash pi/05_run.sh bench --variants norepack
bash pi/05_run.sh chat -p 'What is the capital of France? Answer with one word.'
bash pi/06_pin_hotset.sh ab --budget-bytes 5.2e9 --rounds 2
bash pi/00_feasibility.sh                    # again: the measured io_mem.json and calibration.json are merged
```

| Step | What it does | Refuses or stops when |
|---|---|---|
| `pi/00_feasibility.sh` | Collects host facts and prints the verdict table: which model and quant fit, with which llama.cpp flags, at what PROJECTED speed | exit 3: nothing viable. Exit 4: protected or busy host |
| `pi/01_tune_os.sh` | Prints, and with `--apply` applies, governor, swappiness, read-ahead, memlock and the optional NVMe Gen3 link | never edits the boot config or reboots without explicit flags; exit 3 without root or inside a container |
| `pi/02_build_llama.sh` | Builds llama.cpp a7b94df natively (`GGML_NATIVE=ON`: dotprod + fp16 on the Cortex-A76) and, with `--with-tools`, the probe, pinner and membw | exit 3: missing compiler, cmake or git (never installs), too little disk or RAM; exit 4: bench lock held |
| `pi/03_bench_io_mem.sh` | Measures storage (fio, or dd when fio is missing) and memory bandwidth (membw) | exit 3: too little disk or RAM; exit 4: busy host or bench lock |
| `pi/04_fetch_model.sh` | Downloads the recommended GGUF (resumable `curl -C -`) and verifies size and sha256 | exit 3: not enough disk (it keeps 2 GB free on `/`) or not viable; it never deletes a file |
| `pi/05_run.sh` | `bench`, `stream-bench`, `chat` or `server` with the verdict's flags; writes results envelopes and calibration | exit 3: RAM check fails or the verdict is not viable; repack is refused unless `--allow-repack` |
| `pi/06_pin_hotset.sh` | Pins the non-expert weights and the hottest experts with mlock and A/B-tests the gain | exit 4: mlock not allowed (every protected host), or a pinner is already running |

Exit codes are the same for every script: 0 ok, 1 error, 2 only slow operation possible, 3 not viable
or a missing precondition, 4 refused. The full procedure, configuration, flags per verdict,
troubleshooting and the usage lines of every script are in [docs/PI_RUNBOOK.md](docs/PI_RUNBOOK.md).

## Safety

`context-friend-1` was first probed as the wrong Pi. Since 2026-10-05 it is the real target Pi (the
board behind that hostname was replaced by a 16 GB Pi 5), and the operator has authorised testing on it. It
still runs production ContextFriend services, so it is a **protected host**:

- Every `pi/` script refuses on it (exit 4), because its hostname is in the catalog's `protected_hosts`
  and its `contextfriend-*` services are running. `PIGEMMA_I_KNOW=1` is the authorised override, and
  results written under it carry a note.
- Even with the override: never stop, restart or kill the ContextFriend or Hailo services; run builds,
  downloads and inference at `nice -n 10` (downloads also at `ionice -c3`); no mlock of model weights
  (`pi/06` refuses there); no anonymous repack copy (`-nr` is forced); keep at least 2 GB free on `/`
  and swap models one at a time by hand; no edit of `/boot/firmware/config.txt`, no reboot and no apt
  without the operator's explicit approval.
- Long jobs run with `nohup` and are polled: the board is on Wi-Fi, which drops packets under load.

On the desktop: one model-loading job at a time (the bench lock), Docker caches are never pruned
(they belong to other projects), and nothing is pip-installed into the global Python. Details:
[docs/PI_RUNBOOK.md, section 2.3](docs/PI_RUNBOOK.md#23-safety-rules-on-context-friend-1) and
[docs/design/FACTS.md](docs/design/FACTS.md).

## Repository map

| Path | What it holds |
|---|---|
| [pigemma/](pigemma/) | The Python package. Run every module as `python -m pigemma.<module>` (`python3` on the Pi). |
| [pigemma/paths.py](pigemma/paths.py) | Repository paths, the model registry, tool binaries, the host fingerprint and the bench lock |
| [pigemma/results.py](pigemma/results.py) | The results envelope (`kind`: measured / projected / assumed), `validate` and `summarize` |
| [pigemma/feasibility.py](pigemma/feasibility.py) | The Pi verdict, stdlib only (it runs on a bare Pi Python) |
| [pigemma/gguf_map.py](pigemma/gguf_map.py) | GGUF byte-range maps (`map.json`), hot-set planning, catalog refresh |
| [pigemma/trace.py](pigemma/trace.py) | The `.moetrace` routing-trace format: reader, writer, validator, merge, statistics |
| [pigemma/cache_sim.py](pigemma/cache_sim.py) | The expert-cache simulator (E4) and page-cache validation points (E5, E6) |
| [pigemma/bench.py](pigemma/bench.py) | Drivers and parsers for llama-bench, llama-completion and llama-server (E0, E1, E8) |
| [pigemma/quality.py](pigemma/quality.py) | KLD, perplexity and Q&A grading (E2) |
| [pigemma/h2/](pigemma/h2/) | H2: `decompose.py` (base + residual groups) and `oracle.py` (oracle and norm-bound certification, E7) |
| [pigemma/data/catalog.json](pigemma/data/catalog.json) | Models, devices and storage classes; every number has a `source` |
| [tools/](tools/) | One CMake tree for three native tools, built in Docker and on the Pi: [probe](tools/probe/probe.cpp) (cb_eval recorder for routing traces, activation dumps and io.csv), [pinner](tools/pinner/pinner.c) (mlock daemon), [membw](tools/membw/membw.c) (memory bandwidth) |
| [pc/](pc/) | Desktop drivers: `desktop_bench.ps1` (E0–E2), `docker_run.sh`, `trace_collect.sh` (E3), `pagecache_limit.sh` (E5, E6), `h2_run.ps1` (E7) |
| [pi/](pi/) | The Pi runtime: `00`–`06`, [pi/lib/common.sh](pi/lib/common.sh), [pi/pi_config.env](pi/pi_config.env); `pi/adhoc/` holds the scripts behind the first real-Pi measurements |
| [data/](data/) | The calibration corpus ([data/calibration/README.md](data/calibration/README.md): EN, HE, DE, code) and the prompt sets ([data/prompts/](data/prompts/)) |
| [docs/](docs/) | Hypotheses, experiment plan, Pi runbook, file formats, design facts and research |
| [results/](results/) | Measured records: JSON and Markdown are tracked, logs and logits are not |
| [traces/](traces/) | Routing traces and H2 activations; the large binaries are gitignored, `map31.json` and `*.stats.json` are tracked |
| [tests/](tests/) | pytest. `pytest` needs no models; `pytest -m integration` needs the models, Docker or native binaries ([fixtures](tests/fixtures/README.md)) |
| [docker/](docker/) | `Dockerfile.build` (Ubuntu 24.04 toolchain) and `Dockerfile.pi-emu` (Debian 13 arm64 under qemu) |
| `third_party/` | llama.cpp at a7b94df: sources, the Linux x86 build, and the official Windows CPU binaries |
| `models/` | The GGUF files (gitignored) |

## Documents

- [docs/HYPOTHESES.md](docs/HYPOTHESES.md): H1 and H2 as falsifiable statements, the byte arithmetic,
  the Pi 5 budgets and every pre-registered threshold.
- [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md): E0 to E8, with procedures, commands, controls,
  results templates and the deviations log.
- [docs/PI_RUNBOOK.md](docs/PI_RUNBOOK.md): the Pi procedure, step by step, with hardware options,
  refusal rules and troubleshooting.
- [docs/TRACE_FORMAT.md](docs/TRACE_FORMAT.md): every file format (`.moetrace`, `map.json`, `hotset.json`,
  results envelope, `sim.json`, `validation.json`, `oracle.json`, ...).
- [docs/design/FACTS.md](docs/design/FACTS.md) and [docs/design/manifest.json](docs/design/manifest.json):
  verified facts and the build manifest (files, data formats, experiments, risks).
- [docs/research/RESEARCH_SUMMARY.md](docs/research/RESEARCH_SUMMARY.md): Pi 5 hardware numbers, MoE
  offloading literature and dense-model literature, with sources.
- Measurements: [results/desktop-baseline-2026-10-04.md](results/desktop-baseline-2026-10-04.md),
  [results/pi-real/README.md](results/pi-real/README.md), [results/pi-emu/README.md](results/pi-emu/README.md),
  [docs/results/H2_RESULTS_quick.md](docs/results/H2_RESULTS_quick.md). The rendered `DESKTOP_<date>.md`,
  `H1_RESULTS.md`, `H2_RESULTS.md` and `PI_<date>.md` appear in [docs/results/](docs/results/) as the
  experiments run.

## Deviations from the manifest spec

- **No `pip install -e .` in the quick start.** The spec's first command was `pip install -e . && pytest`.
  The project rules forbid installing into the global Python, and pytest finds the package without an
  install, so the quick start runs `python -m pytest` and mentions the editable install for a private
  venv only.
- **`context-friend-1` is protected, not "never touched".** The spec says never to run heavy scripts on
  `context-friend-1` because it is deny-listed. Since 2026-10-05 it is the real target Pi, so the rule
  became the protected-host refusal with an authorised override and the safety rules above.
- **The "~1.3 tok/s in RAM" figure for the dense 31B is corrected.** The spec quoted about 1.3 tok/s if
  the 31B were entirely in a 16 GB board's RAM, and about 22 s/token from NVMe Gen3. With the catalog
  constants the in-RAM figure is 0.37 tok/s (0.57 at 100% efficiency; 1.3 tok/s would need about
  23 GB/s of weight traffic), the NVMe Gen3 figure is 20.8 s/token, and the measured 3-bit 31B in RAM
  is 0.53 tok/s.
- **E5 and E6 commands follow the experiment plan.** The spec's single `pagecache_limit.sh --mem 4,5,6,8 --pin`
  line is split into the E5 run (caps 3–8 GB, unpinned) and the E6 run (`--pin` at 5 and 6 GB), as in
  [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md#e4-cache-simulation).
