# E0: desktop baseline and repack gate (2026-10-05)

**Verdict:** speed band **PASS** for both models. Repack gate **PASS for the 26B-A4B** and **FAIL for the
31B**. The 31B fails on one criterion only: its private-bytes peak is 2533 MB against a 1.5 GB limit.
The cause is the 31B's own f16 KV cache (1520 MiB at `-c 4096`) plus the compute buffer (592 MiB). No
weights were copied: there is no `CPU_REPACK` line, and 99.9% of the file stays `CPU_Mapped`. The `-ot`
negative control repacks for both models, as predicted.

**What `-nr` costs on this desktop (x86, AVX-512):**
- Decode (tg): **−12 to −17% at 4 threads** for the 26B and −4 to −8% at 6 threads; −10 to −14% for the
  31B at 6 threads.
- Prefill (pp128): **−30 to −45%**.

Decode is memory-bound, so the slower non-repacked kernels show up mostly in prefill. Greedy text with a
chat prompt is byte-identical with and without `-nr`. Logit-level differences (KLD) are as large as the
differences from changing only the batch size with repack on (26B).

All numbers below are MEASURED on this host unless they are labelled DERIVED. The desktop was **not idle**:
background CPU was 12–62% (Chrome, Discord and Docker Desktop are the operator's processes and were left alone),
and the default-variant runs were under memory pressure (see Caveats).

## What was run

Host: Ryzen 5 8400F (6C/12T, AVX-512), 64 GB DDR5, Windows 11, no GPU. llama.cpp b11398 (a7b94df), the
official Windows CPU binaries in `third_party/llama.cpp-win/` (ggml-cpu-zen4). The models are Google's
QAT Q4_0 GGUFs: `26b-q4` (14,439,363,584 B) and `31b-q4` (17,651,001,568 B). Nothing else loaded a model
during the stage (checked with tasklist and docker ps; the bench lock `results/.bench.lock` was held by
every model job).

1. **Manifest procedure, as defined** (27.5 min; driver log `results/e0-desktop_bench-2026-10-05.log`):
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File pc\desktop_bench.ps1 -Stage bench -Models 26b,31b
   ```
   The driver runs these steps:
   - membw (D5, Docker).
   - `python -m pigemma.bench matrix` with the variants default, norepack (`--repack 0`),
     hybrid_negctrl (`-ot "ffn_(gate_up|down)_exps=CPU"`) and nommap (`-lm none`). The 26B uses
     `-t 4,6,12 -p 128 -n 32 -r 3`; the 31B uses `-t 6,12 -p 128 -n 16 -r 2`.
   - `python -m pigemma.bench completion` per variant: `-n 64 --temp 0 -t 6 -c 4096 -lv 4`, with the
     prompt `data/calibration/en_tech_moe_streaming.txt` (784 tokens, raw text, no chat template).
   - The gate in `results/2026-10-05/E0_GATE.md`. The `results summarize` step writes
     `docs/results/DESKTOP_2026-10-05.md`.

   The thread sweep is the manifest's own list. The idle-machine sweep at 4/6/8/12 threads in
   `results/desktop-baseline-2026-10-04.md` was reused as the speed reference and was not repeated.
2. **Added to pin down the `-nr` penalty** (the background load drifted, so a sequential matrix alone is
   not precise enough): llama-bench runs **interleaved A,B,B,A** (default, norepack, norepack, default).
   This order cancels linear drift. Runs: 26B at 4 threads (block 1), 26B at 4 and 6 threads (block 2),
   and 31B at 6 threads.
   ```bash
   bash results/2026-10-05/e0-diag/run_followup.sh            # chat check + ABBA 26B t4 (-r 5, tg64) + ABBA 31B t6 (-r 3, tg32)
   python -m pigemma.bench matrix --model 26b-q4 --threads 4,6 --pp 128 --tg 64 -r 5 \
     --variants default,norepack,norepack,default --out results/2026-10-05/e0-interleave --experiment E0
   python results/2026-10-05/e0-interleave/penalty_summary.py # -> penalty_summary.json
   ```
3. **Added because the 26B `-nr` completion text diverged.** A KLD check with llama-perplexity against
   the default logits. Settings: `data/calibration/all.txt`, `-c 512 --chunks 3 -t 6` (768 scored
   tokens). Comparisons: `-nr`; the default variant at batch 1 (a noise-floor control); and the default
   variant run twice (a determinism control). A chat-prompt greedy check was also run. The 400 MB
   base-logit files were written to the session scratchpad and deleted afterwards.
   ```bash
   bash results/2026-10-05/e0-diag/run_kld_diag.sh 26b; bash results/2026-10-05/e0-diag/run_kld_diag.sh 26b repeat
   bash results/2026-10-05/e0-diag/run_kld_diag.sh 31b
   PYTHONPATH=. python results/2026-10-05/e0-diag/write_kld_envelopes.py
   ```

## Results

### Speed vs the idle-machine reference (pre-registered ±20%)

| Check | Measured tg tok/s (sd) | Band | Result | Background CPU |
|---|---|---|---|---|
| 26B tg32 @4t, default | 16.31 (0.22) | 12.14–18.22 | **PASS** | 29.8% |
| 26B tg32 @6t, default | 15.55 (0.86) | 10.80–16.20 | **PASS** | 29.8% |
| 31B tg16 @6t, default | 2.33 (0.04) | 1.95–2.93 | **PASS** | 17.5% |

### Repack gate (from the llama-completion load logs, `-lv 4`)

| Check | 26B | 31B | Threshold |
|---|---|---|---|
| norepack: `CPU_Mapped` model buffer | 13,755.4 MiB = 99.9% of the file, **PASS** | 16,818.2 MiB = 99.9%, **PASS** | ≥ 95% of the file |
| norepack: `CPU_REPACK` line | absent, **PASS** | absent, **PASS** | absent |
| norepack: private-bytes peak | 1,236 MB, **PASS** | **2,533 MB, FAIL** | < 1,500 MB |
| C1 hybrid_negctrl: `CPU_REPACK` line | present, 13,133.9 MiB (= default), **PASS** | present, 15,710.6 MiB (= default), **PASS** | present |
| **Gate** | **PASS** | **FAIL** | |

Where the norepack private bytes go, from the same logs:

| | 26B | 31B |
|---|---|---|
| KV cache, f16, `-c 4096` | 80 + 300 = 380 MiB | 320 + 1,200 = **1,520 MiB** |
| compute buffer, output buffer | 538.2 + 1 MiB | 591.5 + 1 MiB |
| KV + compute + output | 919 MiB = 964 MB | 2,113 MiB = 2,215 MB |
| private peak measured | 1,236 MB | 2,533 MB |
| rest (runtime, tokenizer, ...) | ~272 MB | ~318 MB |

For the 31B, the KV cache alone (1,594 MB) is larger than the 1.5 GB threshold. The threshold was
written with the 26B's footprint in mind. It is reported as FAIL and **not** changed here. Any change goes
through the deviations log.

Two more observations from the load logs:
- With repack, the Windows build keeps **both** copies resident. The working-set peaks are 28.8 GB (26B)
  and 36.0 GB (31B). The `CPU_Mapped` line still reports the whole file next to `CPU_REPACK`. So the
  absence of `CPU_REPACK` is the discriminating check, not the `CPU_Mapped` size.
- `-ot` changed nothing: the buffer split, the speed and the output are the same as default.

### llama-bench matrix (sequential, random token ids, engine throughput only)

| Model | Variant | pp128 @4/6/12t | tg @4/6/12t | tg / default |
|---|---|---|---|---|
| 26B | default | 56.4 / 69.7 / 90.9 | 16.31 / 15.55 / 13.89 | 1 |
| 26B | norepack | 37.5 / 45.8 / 53.8 | 14.31 / 14.88 / 12.93 | 0.877 / 0.957 / 0.931 |
| 26B | hybrid_negctrl | 54.1 / 69.5 / 85.8 | 15.29 / 15.86 / 13.61 | 0.938 / 1.020 / 0.980 |
| 26B | nommap | 55.0 / 68.7 / 81.8 | 12.88 / 14.40 / 13.79 | 0.790 / 0.926 / 0.993 |
| 31B | default | – / 11.84 / 13.85 | – / 2.33 / 2.16 | 1 |
| 31B | norepack | – / 6.49 / 8.69 | – / 2.00 / 2.18 | – / 0.859 / 1.008 |
| 31B | hybrid_negctrl | – / 12.37 / 15.69 | – / 2.18 / 2.15 | – / 0.936 / 0.995 |
| 31B | nommap | – / 12.13 / 13.31 | – / 2.02 / 1.55 | – / 0.866 / 0.717 |

hybrid_negctrl computes exactly what default computes: same buffers, byte-identical output. Its spread
around default (−6% to +2%) is therefore the session's noise floor for a single sequential run.

### `-nr` penalty = tok/s(norepack) / tok/s(default)

| Model, threads | tg: sequential matrix | tg: ABBA block(s) | **tg range** | pp128: all estimates | **pp range** |
|---|---|---|---|---|---|
| 26B @4t | 0.877 (16.31 → 14.31) | 0.846 (15.64 → 13.22), 0.835 (13.29 → 11.09) | **0.83–0.88 (−12 to −17%)** | 0.665, 0.703, 0.680 | **0.67–0.70** |
| 26B @6t | 0.957 (15.55 → 14.88) | 0.918 (12.38 → 11.37) | **0.92–0.96 (−4 to −8%)** | 0.658, 0.657 | **0.66** |
| 26B @12t | 0.931 | – | 0.93 | 0.592 | 0.59 |
| 31B @6t | 0.859 (2.33 → 2.00) | 0.902 (1.75 → 1.57) | **0.86–0.90 (−10 to −14%)** | 0.549, 0.618 | **0.55–0.62** |
| 31B @12t | 1.008 | – | 1.01 | 0.627 | 0.63 |

Details and per-repetition samples are in `results/2026-10-05/e0-interleave/penalty_summary.json`.

One ABBA run is excluded as disturbed: 26B default @4t, block 1. Its tg samples were 10.94, 8.56,
13.26, 6.35 and 3.35 tok/s, ending at their minimum. The exclusion rule ("the last sample is the minimum
and is below 50% of the first") was written after seeing the data and is stated here. With that run
included, block 1 would give a meaningless 1.10. Block 1 had no memory sampler. During block 2 the
sampler showed available RAM falling to ~9 GB, with page-ins, during the default (repack) runs, which
is the likely cause.

llama-completion (6 threads, 784-token raw prompt, 64 tokens, single runs) shows larger gaps:
- 26B: tg 9.82 vs 13.01 (0.755), pp 39.7 vs 63.8 (0.62).
- 31B: tg 1.74 vs 1.87 (0.93), pp 7.11 vs 10.84 (0.66).

The `-nr` runs load in 2–4 s because they only map the file. The weight pages are then first touched
during evaluation, so those numbers include first-touch faults. This is a plausible cause, not
verified: Windows has no per-process paging counter here.

### Load, memory and buffers (llama-completion, 6 threads, `-c 4096`)

| Model | Variant | load s | RSS peak MB | private peak MB | CPU_Mapped MiB | CPU_REPACK MiB |
|---|---|---|---|---|---|---|
| 26B | default | 9.92 | 28,831 | 15,034 | 13,755.4 | 13,133.9 |
| 26B | norepack | 2.08 | 13,821 | **1,236** | 13,755.4 | absent |
| 26B | hybrid_negctrl | 19.39 | 28,831 | 15,034 | 13,755.4 | 13,133.9 |
| 26B | nommap | 8.62 | 15,666 | 16,265 | – (CPU 1,198.9) | 13,133.9 |
| 31B | default | 30.68 | 36,021 | 19,040 | 16,818.2 | 15,710.6 |
| 31B | norepack | 3.68 | 19,548 | **2,533** | 16,818.2 | absent |
| 31B | hybrid_negctrl | 13.21 | 36,017 | 19,040 | 16,818.2 | 15,710.6 |
| 31B | nommap | 11.09 | 20,704 | 21,327 | – (CPU 2,210.1) | 15,710.6 |

The load times are warm-cache (the files were in the standby list) and are not cold-disk numbers. The
31B nommap tg of 0.58 tok/s (completion) and 1.55 (matrix @12t) are most likely memory pressure: 21.3 GB
private next to the operator's applications and a 15 GB Docker VM commit. This is not verified, and nommap is not
part of the gate.

### Output identity and numerical neutrality (diagnostic, not a pre-registered E0 metric)

E0 completion texts (greedy, 64 tokens; SHA-256 prefixes of stdout):
- **26B:** default = hybrid_negctrl = nommap (`3407ac0ffc01`). **norepack diverged at the 2nd generated
  token** and fell into a "-thought-thought-..." loop (`coherent=false`).
- **31B:** all four variants are byte-identical (`990a35072d35`).

Chat-prompt check, 26B, 6 threads, 160 tokens. The prompt was "Explain in 3 short sentences why running
a 26B mixture-of-experts language model on a Raspberry Pi is hard, then say 'שלום' in Hebrew." Default
and norepack gave **byte-identical**, correct answers ending in "שלום"
(`results/2026-10-05/e0-diag/chat/`).

KLD against the default logits (repacked, n_batch 512); 768 tokens of `all.txt`; envelopes in
`results/2026-10-05/e0-diag/`:

| Model | Run compared with default b512 | mean KLD | same top-1 | 99.9% KLD | PPL(Q) (default: 236.84 / 201.86) |
|---|---|---|---|---|---|
| 26B | default b512, run again (determinism) | 0.000000 | 100.0% | 0.00004 | 236.84 |
| 26B | **norepack b512** | **0.0872** | **90.2%** | 2.97 | 226.71 |
| 26B | default **b1** (repack on, token-by-token kernels) | 0.0880 | 90.2% | 3.26 | 235.24 |
| 26B | norepack b1 | 0.0753 | 91.1% | 3.40 | 229.65 |
| 31B | **norepack b512** | **0.0148** | **93.9%** | 1.23 | 207.51 |

Reading:
- The engine is deterministic.
- On the 26B, `-nr` moves the logits exactly as much as merely switching the batch size with repack on.
  Its PPL is not worse: 226.7 vs 236.8, within the ±19% statistical error.
- The divergence is therefore a property of this model/engine pair: any change in accumulation order
  flips about 10% of argmax tokens. It is not a `-nr` defect.
- A likely mechanism is near-tie top-8 expert choices flipping in the MoE router. This is a
  HYPOTHESIS, not measured here.
- For the 31B no batch-size control was run, so its 0.015 has no measured floor. Its PPL moves the other
  way (207.5 vs 201.9, within error).
- On this 3-chunk preview, the pre-registered E2/C2 threshold (KLD < 0.001 and same-top-p > 99.5%) would
  fail for both models. The repacked default at batch 1 would fail it against itself too (see
  Consequences).
- llama-perplexity's "Mean PPL(base)" in KLD mode is biased for this 262k vocabulary: an identical re-run
  shows PPL(Q)/PPL(base) = 1.085 at KLD 0. Compare PPL(Q) with the base run's "Final estimate" instead.

### Desktop memory bandwidth (D5) and derived efficiency

membw in Docker (WSL2 VM, 512 MB, best of 5) gave read 43.4 / **46.9** / 46.5 / 45.8 / 45.8 GB/s at
1 / 2 / 4 / 6 / 12 threads (`results/2026-10-05/membw_desktop.json`).

DERIVED efficiency = tg × bytes/token ÷ 46.9 GB/s. The bytes per token come from `results/map26.json`
(2,380,072,318 B) and `traces/h2/map31.json` (17,635,170,858 B):

| Model, variant | Efficiency |
|---|---|
| 26B @4t, default | 0.83 |
| 26B @4t, norepack | 0.73 |
| 31B @6t, default | 0.88 |
| 31B @6t, norepack | 0.75 |

The in-VM membw may understate the native bandwidth, so these are approximate.

## Verdict vs the pre-registered thresholds

| Threshold | Result |
|---|---|
| 26B tg@4t in 12.14–18.22 and tg@6t in 10.80–16.20 | **PASS** (16.31, 15.55) |
| 31B tg@6t in 1.95–2.93 | **PASS** (2.33) |
| 26B repack gate | **PASS** (13,755.4 MiB mapped, no CPU_REPACK, 1,236 MB private, the control repacks) |
| 31B repack gate | **FAIL**: private peak 2,533 MB ≥ 1,500 MB. The other three checks PASS. The cause is the KV cache plus the compute buffer, not the weights |
| Norepack penalty (reported, "expected 10–40%") | tg: 26B −12 to −17% @4t, −4 to −8% @6t; 31B −10 to −14% @6t. pp128: −30 to −45% |
| Hypothesis "variants differ in speed, not output" | **Not supported as stated.** The `-nr` logits differ: KLD 0.087 (26B), 0.015 (31B). The 26B raw-text greedy text diverged, while the chat-prompt text is identical. For the 26B the size of the difference equals the batch-size-only control |

## Is `-nr` a viable Pi mode?

From the desktop evidence: **yes for the 26B-A4B.**
- It is the only mode that keeps the file page-cache backed: the gate passes with 1.24 GB private.
- Decode costs 12–17% at the Pi's thread count (4) on x86, and prefill costs 30–34%.
- On a chat prompt the output is identical, and the logit changes are no larger than a batch-size change.

The real Pi already runs the QAT Q4_0 26B with `-nr` at 3.4–4.1 tok/s, with 36/36 Q&A by content
(`results/pi-real/README.md`). The ARM penalty is not measured here: ARM uses the 4x4/4x8 dotprod repack
kernels, so the x86 ratio is not transferable. pi/05 (E9) must measure it.

A PROJECTION, if the desktop 4-thread decode ratio (~0.85) held on ARM: the catalog's 0.65 efficiency
would become ~0.55 under `-nr`. That is between the 0.65 and 0.45 rows of HYPOTHESES 4.6.

## Consequences for the next stages (recommendations; the orchestrator and the operator decide)

- **E5/E6 and Pi streaming use the 26B, whose gate PASSES.** The `-nr` flag behaves as verified, so I
  see no flag problem to fix. The literal gate rule ("if the gate fails, stop all streaming work") is
  triggered by the 31B's private-peak FAIL. That failure measures KV and compute memory, not a copied
  weight, so proceeding needs an explicit decision and a deviations-log entry. The memory budget for a
  31B on the Pi must include ~2.2 GB of non-weight anonymous memory at `-c 4096` (f16 KV).
- **E2 (C2):** the pre-registered "KLD < 0.001 and same-top-p > 99.5%" is very likely to fail for both
  models. The repacked default variant fails it against itself when only the batch size changes. Add
  the default-b1-vs-default-b512 control as the measured noise floor and record it as a deviation. The
  threshold itself is not changed here.
- **E3:** C2's fallback applies. Cross-check `-nr` routing on one calibration file traced without `-nr`,
  because about 10% of next-token argmaxes (and plausibly some expert choices) differ between kernel
  paths.
- **E0 completion prompt:** it is raw text without a chat template, so its greedy continuation is
  fragile. A chat-wrapped prompt (`bench completion --chat`) would make the coherence column meaningful.

## Caveats

- The machine was not idle. Background CPU was 12–62% before each run (10 s window); see the envelope
  field `extra.background_cpu`. Default-variant runs drove available RAM down to ~9 GB, with hard page-ins
  of 2–9k pages/s (sampled at 2 s while ABBA block 2 ran). Ratios from interleaved runs are more robust
  than the absolute tok/s.
- llama-bench feeds random token ids, so its numbers say nothing about routing or locality.
- Windows has no per-process page-fault or read-bytes counter in this tool: read_bytes/majflt per token
  are n/a.
- The KLD diagnostic uses 3 chunks (768 tokens), not E2's 6. It is a preview, not the E2 measurement.

## Files

- Gate and tables: `results/2026-10-05/E0_GATE.md`. Envelopes: `results/2026-10-05/E0_*.json` (48).
  Raw logs: `results/2026-10-05/raw/`.
- Driver log: `results/e0-desktop_bench-2026-10-05.log`. Summary: `docs/results/DESKTOP_2026-10-05.md`.
- ABBA: `results/2026-10-05/e0-interleave/` (envelopes, `penalty_summary.py`, `penalty_summary.json`).
- Diagnostics: `results/2026-10-05/e0-diag/`. It holds the KLD logs and envelopes, the chat check,
  `run_kld_diag.sh`, `run_followup.sh` and `write_kld_envelopes.py`.
- membw: `results/2026-10-05/membw_desktop.json`.
