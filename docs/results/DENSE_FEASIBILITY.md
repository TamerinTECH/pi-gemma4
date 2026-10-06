# D1 — dense 31B: activation-aware low-rank base + streamed residual blocks (feasibility, 2026-10-05)

Stage D1 of the decision workflow (brief section 6; UPDATE 2 "dense C"). **Approximate-computation research
track**, desktop only, independent of the exact MoE cache work (track A). Nothing below is evidence for or
against expert pinning, and no result of track A or B carries over to it. No Pi streaming engine was built
and nothing ran on the Pi.

Labels: **MEASURED** (computed on this desktop from captured activations and the real weights),
**PROJECTED** (extrapolated from the three sampled layers to all 60, bytes only), **ASSUMED** (a constant
with its source), **NOT MEASURED**.

**Reference.** Every error is relative to the **31B Google QAT Q4_0 GGUF dequantised to float32**
(`google/gemma-4-31B-it-qat-q4_0-gguf` @ `59dde245`, `gemma-4-31B_q4_0-it.gguf`, 17,651,001,568 B, sha256
`179cfb99…2874b`). The original BF16 checkpoint is **not** used, so the quantisation error of Q4_0 itself
is NOT MEASURED.

## 1. Verdict

**NEGATIVE for this factorization.** The rule was written to `results/dense/preregistration.json` before any
fit or evaluation. The study stopped after step 4 of 6: "few corrections suffice" is false, so the online
selector was not run as a gate (step 5) and no autoregressive generation was run (step 6).

The idea was to keep a small approximate copy of each large feed-forward (FFN) matrix in RAM and to fetch
from storage only the exact pieces that matter for the current token. On the three layers tested, the small
copy is too inaccurate, and when the exact weights are cut into blocks of 128 columns or rows almost all of
the blocks matter. A plain 3-bit quantisation of the same layers has 2.6 to 3.3 times lower error at the
same size, needs no selector and no new engine, and 3-bit files of this model already run on the Pi.

| MEASURED on held-out tokens: error of the FFN output (mean relative L2) | layer 2 | layer 30 | layer 57 |
|---|---|---|---|
| Q4_0 layer itself, 1.000× bytes | 0 (reference) | 0 | 0 |
| **Q3_K** re-quantisation of the same layer, 0.764× bytes | **0.077** | **0.120** | **0.106** |
| Simple 3-bit absmax quantiser, 0.778× bytes | 0.121 | 0.175 | 0.185 |
| Base + residual blocks, **oracle** selection, 0.764× bytes (best rank and block layout) | 0.198 | 0.400 | 0.290 |
| … compared with Q3_K (the rule required at most 0.5×) | 2.6× worse | 3.3× worse | 2.7× worse |
| Base + residual blocks, online norm-bound selector, 0.764× bytes (best rank) | 0.208 | 0.408 | 0.318 |
| Base + one static block set, 0.764× bytes (best layout) | 0.281 | 0.463 | 0.314 |
| Rank-512 base alone, 0.225× bytes | 0.412 | 0.591 | 0.404 |
| Bytes the oracle needs to **match Q3_K's error** (× Q4_0, interpolated) | 0.97 | 1.16 | 1.11 |
| Residual blocks the oracle needs for error ≤ 1e-2 on ≥ 90 % of tokens | 97.7 % | 99.5 % | 99.9 % |
| … total bytes at that point, base included (× Q4_0) | 1.20 | 1.22 | 1.22 |

"× bytes" is the fraction of the layer's Q4_0 FFN bytes (195,084,288 B). The oracle uses the true residual
of each token, so it is not deployable. It shows what a selector with perfect knowledge of each matrix would
reach; it is not a proven optimum.

### Why it failed

1. **The base is too weak for what it costs** (main cause). A rank-512 base costs 0.225× the layer's bytes
   and leaves a per-matrix output error of 0.17 to 0.66. Q3_K leaves 0.04 to 0.13 at 0.764×. A rank-2048
   base already costs more than Q3_K (0.90×) and still has 2.1 to 4.2 times Q3_K's per-matrix error.
   This is not an effect of the small calibration set: four times more calibration tokens change the error
   by 0.01 to 0.03, and a base fitted on the held-out tokens themselves still has 2.2 to 3.3 times Q3_K's
   error (section 4.7, S1).
2. **Cut into blocks of 128, the residual has no structure to exploit** (main cause). For the gate and up
   projections the oracle's error curve lies on the no-structure curve `e_base · sqrt(1 − p)`: reading half
   of the blocks removes about 30 % of the error, as it would if every block carried an equal share.
   Reaching 1e-2 from a base error of 0.2 to 0.6 therefore takes all blocks but a handful.
3. **Selection is not the bottleneck for blocks.** At equal bytes the cheap norm-bound selector is within 2
   to 5 % of the oracle (0.211 vs 0.202, 0.408 vs 0.400, 0.318 vs 0.311 at rank 512, column blocks). There
   is little token-dependent structure to select: the oracle's error is only 8 to 30 % below that of one
   static block set.
4. **Error accumulation was not tested.** Steps 5 and 6 are gated on steps 1 to 4 and were not run.

### What the post-hoc check adds (exploratory, not part of the verdict)

Cause 2 is a property of the block layout, not of the weights. Organised per FFN neuron instead of per
128-wide block, the corrections are concentrated (section 4.7, S2):

| MEASURED, post-hoc, same held-out tokens and metric | layer 2 | layer 30 | layer 57 |
|---|---|---|---|
| Q3_K, 0.764× bytes | 0.077 | 0.120 | 0.106 |
| Bytes an oracle neuron selection needs to match Q3_K's error (× Q4_0) | 0.09 | 0.36 | 0.29 |
| Error at 0.764× bytes: oracle neuron selection + rank-512 base | 0.012 | 0.093 | 0.070 |
| Error at 0.764× bytes: neurons chosen **from the base alone** + base (a deployable selector) | 0.027 | 0.243 | 0.136 |
| Error at 0.764× bytes: one static neuron set + base | 0.157 | 0.339 | 0.172 |

An oracle that knows which neurons matter beats Q3_K at equal bytes on all three layers. The one cheap
selector tested recovers that at the early layer only; at the middle and late layers it is 2.0 and 1.3
times worse than Q3_K. Tight tolerances still need most neurons (75 to 90 % for 1e-2 at layers 30 and 57).
None of this is validated beyond single-layer error with perfect inputs, and a 9 KB neuron is the wrong
unit for microSD or USB streaming. It does not change the recommendation in section 7.

## 2. What was run

| Step of the brief | Status | Result |
|---|---|---|
| 1. Capture FFN activations on calibration and separate held-out text | done | 7,870 calibration tokens; 2,961 held-out tokens, 1,745 evaluated (528 of them generated by the model) |
| 2. One representation, fitted on calibration only | done | whitened low-rank base (ranks 128 to 2048, 8-bit factors) + exact residual blocks of 128 columns or 128 rows; ASVD-style diagonal variant at rank 512 |
| 3. Base + all blocks reconstructs the reference | **PASS** | max relative error 8.1e-7 against a declared tolerance of 1e-5 |
| 4. Offline-informed selection, with storage, runtime and equal-byte comparison against Q4_0, Q3_K and simple 3-bit / 2-bit quantisers | done | **NEGATIVE** (section 1) |
| 4b. Argmax preservation through the rest of the network | not run | needs a patched forward pass; not justified after the layer-level negative |
| 5. Cheap online selector | not run as a gate | measured as a by-product at equal bytes: close to the block oracle, far from Q3_K |
| 6. Accumulated error in autoregressive generation | not run | precondition failed |
| UPDATE 2: dynamic corrections against the quantised baseline, the base alone and a static selection; online selector that does not read the corrections | done | sections 4.5 (blocks: norm bound from resident norms) and 4.7 (neurons: ranking from the base) |
| Post-hoc robustness checks of the two attributions | done | section 4.7; exploratory, cannot change the verdict |

Wall time: capture 64 min, fits and evaluation 66 min, robustness checks 15 min, report about 45 min. The
run was interrupted for about 75 min by an account usage limit between the evaluation and the report.

## 3. Method

**Layers.** 2 (early), 30 (middle), 57 (late) of 60. All three are sliding-attention layers; the FFN has the
same shape in global-attention layers. Matrices: gate and up [21,504 × 5,376], down [5,376 × 21,504], each
65,028,096 B at Q4_0. The FFN computes `W_down(gelu_tanh(W_gate x) · W_up x)`; the 21,504 intermediate
values are the "neurons".

**Data, no overlap.** `data/workloads/*.jsonl` did not exist when the stage started, so the split is by
file (`results/dense/inputs/split.json`).
Calibration, used only for fitting: four corpus files as raw text in eight fresh-context chunks of at most
1,024 tokens (Hebrew discussion 4,140 tokens, C++ 2,519, English technical 785, mixed Q&A 635).
Held-out, used only for evaluation: 11 chat turns with the model's own template and thinking off. Three are
the other corpus files (English narrative, German expository, C++) followed by a summarise or explain
instruction; eight are short English, Hebrew, code and mixed prompts. Each turn is followed by 48 greedy
tokens generated by the model. Evaluated tokens: all 528 generated tokens plus every second prompt token.

**Capture.** The existing probe, `pigemma-probe --mode act-dump` (llama.cpp a7b94df, Docker
`pi-gemma4-build:latest`, default CPU repack, `-t 6 -c 2048 -b 1024 -ub 512`), kinds `ffn_norm` (input of
gate and up), `ffn_geglu` (input of down), `ffn_out` (output of down, before the post-FFN norm).

**Base.** `W_base = U_r U_rᵀ W`, where `U_r` holds the top-r left singular vectors of `W L` and
`L Lᵀ = C + 0.01·mean(diag C)·I`, `C = E[x xᵀ]` over the calibration inputs of that matrix. This is the
rank-r matrix that minimises `‖(W − W_base) X_cal‖_F` (the truncation-aware whitening of SVD-LLM). The 1 %
ridge was declared in advance and not tuned. A second variant replaces `C` by its diagonal (ASVD-style).
The factors `U_r` and `U_rᵀ W` are stored as 8-bit block-32 codes with an f16 scale (1.0625 B per weight) and
all errors use the quantised factors. Base bytes per matrix = r × 26,880 × 1.0625 B: 0.056× (r = 128),
0.112×, 0.225×, 0.450×, 0.899× (r = 2048) of the matrix's Q4_0 bytes.

**Residual blocks.** `R = W_ref − W_base`, cut into blocks of 128 input columns (42 blocks for gate and up,
168 for down) or 128 output rows (168 and 42). A selected block is applied exactly. Its payload is the
reference's own Q4_0 bytes for that block (Q4_0 blocks of 32 inputs tile a 128-column block, so the
re-layout is byte-exact, 0.5625 B per weight); the base is used for the unselected blocks only. Metadata:
12 B per column block resident (spectral norm for the selector and an offset), 2 B per selected block per
token.

**Selection.** Oracle, from the true contributions `C_g = R_g x_g` of the token: column blocks greedily by
largest error reduction, with the option to stop early at a byte budget; row blocks by largest residual
energy, which is optimal because they touch disjoint outputs. Online column selector: the norm bound
`b_g = ‖R_g‖₂ · ‖x_g‖`, which needs only resident norms and the input, not the corrections. Static: one
block order for all tokens, ranked by mean calibration energy. The oracle is a potential study, not a
deployable predictor and not a proven optimum for the composed FFN: the three matrices get the same residual
fraction (equal-byte comparison) or the same per-matrix tolerance (tolerance search), not a jointly optimal
allocation.

**Errors.** Relative L2 per held-out token, reported as mean (p90 is in the JSON files): for one matrix
with perfect inputs ("isolated"), and for the FFN output with all three matrices approximated and the
captured layer input ("composed").

**Quantised baselines on the same layer.** Q4_0 itself (error 0 by definition). **Q3_K** (3.4375 bits per
weight, 0.764×): a numpy port of llama.cpp's `quantize_row_q3_K_impl` with an importance matrix from the
same calibration activations, applied to the Q4_0 reference as `llama-quantize --allow-requantize` would.
Its packing was checked against gguf-py's Q3_K dequantiser (exact match on all checked rows); the scale
search uses pairwise instead of sequential float sums, so it is not guaranteed bit-identical to llama.cpp.
Simple absmax block-32 quantisers at 3 bits (0.778×) and 2 bits (0.556×).

**Decision rule (pre-registered).** PROMISING only if on every sampled layer (a) at Q3_K's bytes the oracle
FFN error is at most half of Q3_K's and (b) FFN error ≤ 1e-2 on at least 90 % of held-out tokens with at most
15 % of the residual and a base of at most 0.25× Q4_0. INCREMENTAL if the oracle beats Q3_K at equal bytes
by 1.2 to 2× on every layer. Otherwise NEGATIVE with an attribution. Choosing the best rank and block layout
on held-out data favours the method; this was accepted because a positive verdict would have been
re-checked in step 6.

## 4. Results

### 4.1 Reconstruction and the separate error sources (step 3)

| Source | Value | Status |
|---|---|---|
| Base + all residual blocks vs the reference, computed in the runtime form `A(Bx) + Σ R_g x_g` | ≤ 8.1e-7 relative over all 54 layer × base × rank × matrix cases (first 48 held-out tokens each); declared tolerance 1e-5 | MEASURED, **PASS** |
| Quantisation error of Q4_0 against the original checkpoint | no higher-precision reference on disk (the BF16 checkpoint is about 61 GB; D: has 13 GB free against a 10 GB floor) | NOT MEASURED |
| llama.cpp's own FFN output against float32 arithmetic on the same Q4_0 weights | median 0.0043 / 0.0071 / 0.0080 at layers 2 / 30 / 57 (p90 0.0083 / 0.0087 / 0.0112) | MEASURED |
| 8-bit quantisation of the base factors | changes the base error by at most 0.001 | MEASURED |
| Approximation error of the base | 0.17 to 0.66 per matrix at rank 512 (section 4.3) | MEASURED |

llama.cpp quantises activations to Q8_0 inside its Q4_0 dot products (`ggml-cpu.c`, `vec_dot_type` of Q4_0),
so the engine itself differs from exact float32 arithmetic by 0.4 to 0.8 %. The brief's tolerances 5e-3 and
1e-3 are at or below this floor: they can only be met by reading essentially everything and are not a
meaningful quality bar for this engine. The 100 % end points of the row-oracle curves in the JSON files read
about 2e-5; that is float32 cancellation in the energy-subtraction formula, not a reconstruction error.

### 4.2 The quantised baselines on these layers (MEASURED, held-out)

| Layer | Format | Bytes / Q4_0 | gate | up | down | FFN mean | FFN p90 |
|---|---|---|---|---|---|---|---|
| 2 | Q3_K, importance-weighted | 0.764 | 0.040 | 0.117 | 0.059 | 0.077 | 0.142 |
| 2 | 3-bit absmax | 0.778 | 0.065 | 0.184 | 0.090 | 0.121 | 0.204 |
| 2 | 2-bit absmax | 0.556 | 0.144 | 0.337 | 0.166 | 0.270 | 0.387 |
| 30 | Q3_K, importance-weighted | 0.764 | 0.076 | 0.121 | 0.088 | 0.120 | 0.141 |
| 30 | 3-bit absmax | 0.778 | 0.110 | 0.173 | 0.128 | 0.175 | 0.204 |
| 30 | 2-bit absmax | 0.556 | 0.218 | 0.319 | 0.243 | 0.347 | 0.394 |
| 57 | Q3_K, importance-weighted | 0.764 | 0.069 | 0.129 | 0.069 | 0.106 | 0.165 |
| 57 | 3-bit absmax | 0.778 | 0.104 | 0.187 | 0.139 | 0.185 | 0.259 |
| 57 | 2-bit absmax | 0.556 | 0.205 | 0.341 | 0.253 | 0.358 | 0.466 |

### 4.3 The base: cost and error (MEASURED, held-out, whitened base, mean error per matrix)

| Rank | Bytes / Q4_0 | L2 gate / up / down | L30 gate / up / down | L57 gate / up / down |
|---|---|---|---|---|
| 128 | 0.056 | 0.250 / 0.714 / 0.383 | 0.497 / 0.770 / 0.560 | 0.325 / 0.625 / 0.407 |
| 256 | 0.112 | 0.209 / 0.627 / 0.360 | 0.459 / 0.723 / 0.539 | 0.302 / 0.586 / 0.374 |
| 512 | 0.225 | 0.169 / 0.518 / 0.327 | 0.414 / 0.663 / 0.507 | 0.274 / 0.536 / 0.335 |
| 1024 | 0.450 | 0.128 / 0.391 / 0.280 | 0.355 / 0.580 / 0.458 | 0.236 / 0.469 / 0.286 |
| 2048 | 0.899 | 0.085 / 0.255 / 0.210 | 0.273 / 0.454 / 0.372 | 0.183 / 0.368 / 0.218 |
| Q3_K | 0.764 | 0.040 / 0.117 / 0.059 | 0.076 / 0.121 / 0.088 | 0.069 / 0.129 / 0.069 |

Composed FFN error of the base alone: rank 512 gives 0.412 / 0.591 / 0.404 at layers 2 / 30 / 57, rank 1024
gives 0.337 / 0.547 / 0.362. The diagonal (ASVD-style) base at rank 512 is worse than the whitened one at
layers 2 and 57 (by up to 0.14, on the up matrix of layer 57) and slightly better on gate and up at layer
30 (0.384 vs 0.414 and 0.632 vs 0.663).

### 4.4 How many corrections are needed (potential study, MEASURED, held-out, oracle)

Composed FFN, rank-512 base, cheapest point of the tolerance search that meets the target on at least 90 %
of tokens. "Residual" is the fraction of the layer's Q4_0 bytes read as corrections.

| FFN target | L2: residual / total | L30: residual / total | L57: residual / total |
|---|---|---|---|
| 1e-2 | 0.977 / 1.202 | 0.995 / 1.220 | 0.999 / 1.224 |
| 5e-3 | 0.993 / 1.217 | 0.997 / 1.222 | 1.000 / 1.225 |
| 1e-3 | 1.000 / 1.224 | 0.997 / 1.222 | 1.000 / 1.225 |

A rank-1024 base does not help: the residual fractions stay at 0.974 to 0.999 and the total rises to 1.42
to 1.45. Looser tolerances, same search (rank 512, column blocks): a mean FFN error of 0.14 / 0.13 / 0.12
costs 0.81 / 1.16 / 1.11 of the Q4_0 bytes, and 0.062 / 0.049 / 0.052 costs 1.03 / 1.21 / 1.19.

Isolated matrices with perfect inputs, rank-512 base, mean fraction of column blocks the oracle reads:

| Tolerance | L2 gate / up / down | L30 gate / up / down | L57 gate / up / down |
|---|---|---|---|
| 1e-1 | 0.54 / 0.97 / 0.26 | 0.93 / 0.97 / 0.88 | 0.92 / 0.99 / 0.74 |
| 1e-2 | 1.00 / 1.00 / 0.93 | 1.00 / 1.00 / 0.995 | 1.00 / 1.00 / 0.998 |
| 5e-3 | 1.00 / 1.00 / 0.98 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| 1e-3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |

Error left after the oracle has read a fraction p of the column blocks, next to `e_base · sqrt(1 − p)`
(what equal, uncorrelated blocks would give), rank 512:

| Layer, matrix | base error | p = 0.25: measured / no-structure | p = 0.50 | p = 0.75 |
|---|---|---|---|---|
| 2 gate | 0.169 | 0.147 / 0.147 | 0.122 / 0.120 | 0.086 / 0.085 |
| 2 up | 0.518 | 0.449 / 0.449 | 0.371 / 0.366 | 0.262 / 0.259 |
| 2 down | 0.327 | 0.108 / 0.283 | 0.066 / 0.231 | 0.039 / 0.163 |
| 30 gate | 0.414 | 0.347 / 0.359 | 0.277 / 0.293 | 0.189 / 0.207 |
| 30 up | 0.663 | 0.561 / 0.574 | 0.450 / 0.469 | 0.306 / 0.331 |
| 30 down | 0.507 | 0.338 / 0.439 | 0.251 / 0.359 | 0.164 / 0.254 |
| 57 gate | 0.274 | 0.255 / 0.237 | 0.226 / 0.193 | 0.168 / 0.137 |
| 57 up | 0.536 | 0.502 / 0.464 | 0.446 / 0.379 | 0.331 / 0.268 |
| 57 down | 0.335 | 0.220 / 0.290 | 0.170 / 0.237 | 0.110 / 0.168 |

The down projection is the exception. Its input is the post-GELU activation, which is heavy-tailed, and
column selection works on it: at layer 2, 25 % of the blocks cut the error from 0.327 to 0.108. In
isolation, with a rank-128 base, the oracle and the online selector even beat Q3_K there at equal bytes
(0.047 and 0.046 vs 0.059). The same comparison loses at layer 30 (0.195 vs 0.088) and layer 57 (0.137 vs
0.069), and the gate and up projections dominate the composed FFN error at all three layers.

At layer 57 the gate and up curves are worse than the no-structure curve. The whitened base is fitted so
that its block errors cancel each other; applying only some exact blocks removes part of that cancellation.
For gate and up at layers 2 and 57, a partial set of corrections is at some point worse than no correction
at all on 18 to 30 % of tokens at rank 512 and on about half of the tokens at rank 2048 (2 to 7 % at layer
30; never with the diagonal base).

### 4.5 Equal bytes and equal error against the quantised layer (MEASURED, held-out, composed FFN)

Total bytes (base + residual + metadata) equal to Q3_K's; the three matrices get the same residual fraction.

| Layer | Rank | Residual fraction | Q3_K | oracle, columns | oracle, rows | online norm bound | static columns | static rows | base alone |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 512 | 0.539 | 0.077 | 0.202 | 0.259 | 0.211 | 0.289 | 0.291 | 0.412 |
| 2 | 1024 | 0.314 | 0.077 | 0.198 | 0.264 | 0.208 | 0.287 | 0.281 | 0.337 |
| 30 | 512 | 0.539 | 0.120 | 0.400 | 0.428 | 0.408 | 0.463 | 0.464 | 0.591 |
| 30 | 1024 | 0.314 | 0.120 | 0.436 | 0.462 | 0.445 | 0.488 | 0.484 | 0.547 |
| 57 | 512 | 0.539 | 0.106 | 0.311 | 0.290 | 0.318 | 0.352 | 0.314 | 0.404 |
| 57 | 1024 | 0.314 | 0.106 | 0.315 | 0.303 | 0.332 | 0.352 | 0.320 | 0.362 |

A rank-2048 base alone already exceeds the Q3_K budget. On generated tokens only, the oracle is worse at
layer 57 (0.438 against 0.311 over all tokens, rank 512, columns) and about the same at layers 2 and 30.

| Other equal-byte and equal-error comparisons, rank 512, best oracle layout | L2 | L30 | L57 |
|---|---|---|---|
| At the 3-bit absmax budget (0.778×): ours / absmax | 0.195 / 0.121 | 0.390 / 0.175 | 0.285 / 0.185 |
| At the 2-bit absmax budget (0.556×): ours / absmax | 0.265 / 0.270 | 0.489 / 0.347 | 0.343 / 0.358 |
| Bytes to match Q3_K's error (× Q4_0; × Q3_K's bytes), interpolated | 0.97; 1.27× | 1.16; 1.52× | 1.11; 1.45× |

With oracle selection the representation is on a par with the naive 2-bit absmax quantiser at layers 2 and
57 and worse at layer 30. It never comes within a factor 1.5 of even the simple 3-bit quantiser.

### 4.6 Storage, metadata, selection overhead, CPU time, data moved per token

| Quantity | Value | Label |
|---|---|---|
| Q4_0 FFN bytes per layer; 60 layers | 195,084,288 B; 11.71 GB | EXACT |
| Q3_K FFN bytes per layer; 60 layers | 149,022,720 B; 8.94 GB | EXACT |
| Base, rank 512, three matrices per layer; 60 layers | 43,868,160 B; 2.63 GB | EXACT; PROJECTED |
| Residual store (all blocks, byte-exact re-layout of the Q4_0 data) | 195,084,288 B per layer | EXACT |
| Metadata resident per layer; per selected block per token | 3,024 B; 2 B | EXACT |
| Total storage on disk, 60 layers (base + full residual store) | 14.34 GB against 11.71 GB for Q4_0 | PROJECTED |
| Residual read per token at FFN error 1e-2, 60 layers | 11.59 GB (99.0 % of the Q4_0 FFN) | PROJECTED |
| Data moved per token at FFN error 1e-2, attention and output head included | 20.15 GB against 17.64 GB for plain Q4_0 | PROJECTED |
| Residual read per token at Q3_K-equal bytes, 60 layers | 6.31 GB, at 2.6 to 3.3× Q3_K's error | PROJECTED |
| Attention + output head, untouched by any FFN scheme | 5.93 GB per token | EXACT |
| Streaming the 1e-2 residual from storage | 13.6 s/token at 850 MB/s (ASSUMED NVMe Gen3); 127 s/token at 91.6 MB/s (the Pi microSD's MEASURED rate) | PROJECTED |

Selection overhead of the norm-bound selector (norms of the input blocks, multiply, top-k): 0.015 ms for 42
blocks and 0.034 to 0.041 ms for 168 blocks per matrix per token, under 6 ms per token for 180 matrices
(MEASURED, torch, 6 threads). The oracle has no meaningful overhead number because it needs every
correction.

CPU time of a float32 torch prototype, layer 30, one token, 6 threads, median of 30 runs. The desktop was
not idle (background CPU 21 % before, 35 % after), so these are indicative only.

| Operation (ms) | gate [21,504 × 5,376] | down [5,376 × 21,504] |
|---|---|---|
| Dense float32 mat-vec | 16.1 | 16.8 |
| Base, rank 256 / 512 / 1024 / 2048 | 2.1 / 3.5 / 4.6 / 7.4 | 1.3 / 2.3 / 4.4 / 8.2 |
| Residual, block loop over 10 % / 25 % / 50 % / 100 % of the column blocks | 4.0 / 10.4 / 22.1 / 42.2 | 4.5 / 11.4 / 23.1 / 47.0 |

At Q3_K-equal bytes (rank 512 plus 54 % of the blocks) the prototype needs about 27 ms for the gate matrix
against 16 ms for the plain dense product, 1.7 times slower. Part of that is prototype overhead: the block
loop over all blocks costs 2.6 to 2.8 times the single dense product. No CPU-time saving was measured at any
residual fraction above about 30 %. The prototype is not comparable in absolute terms with the existing
quantised implementation: llama.cpp's Q4_0 kernels run the whole 60-layer model at 2.33 to 2.44 tok/s on
this desktop (MEASURED earlier, `results/desktop-baseline-2026-10-04.md` and E0), which is about 1.5 ms per
FFN matrix if time is split in proportion to bytes (PROJECTED). No ggml kernel for the base or the block
residual was written; after the quality result there was no case for one.

### 4.7 Robustness checks of the two attributions (post-hoc, exploratory)

Both checks were added after the rule had returned NEGATIVE. They were not pre-registered and cannot change
the verdict for the representation as specified. Same reference, split and held-out tokens; float32 factors;
6 threads; background CPU 12 % before and 23 % after (not a timing run). Script:
`results/dense/d1_robustness.py`. Where it overlaps with the main analysis it reproduces it: the per-matrix
base errors agree to three digits, and the base-alone FFN error is 0.411 / 0.591 / 0.404 against 0.412 /
0.591 / 0.404.

**S1. Is the base weak because the calibration set is small?** No.

Whitened base, rank 512 (0.225× bytes), mean error per matrix on the held-out tokens unless stated.

| Base fitted on | Tokens in the fit | L2 gate / up / down | L30 gate / up / down | L57 gate / up / down |
|---|---|---|---|---|
| a quarter of the calibration tokens | 1,968 | 0.179 / 0.544 / 0.339 | 0.433 / 0.686 / 0.520 | 0.283 / 0.554 / 0.351 |
| half of the calibration tokens | 3,935 | 0.172 / 0.526 / 0.329 | 0.421 / 0.671 / 0.513 | 0.277 / 0.543 / 0.341 |
| all calibration tokens (the main analysis) | 7,870 | 0.169 / 0.518 / 0.327 | 0.414 / 0.663 / 0.507 | 0.274 / 0.536 / 0.335 |
| all calibration tokens, error measured on calibration itself (in-sample) | 7,870 | 0.105 / 0.308 / 0.242 | 0.265 / 0.422 / 0.375 | 0.180 / 0.347 / 0.232 |
| the held-out tokens themselves (in-sample bound, not a legitimate method) | 1,745 | 0.101 / 0.304 / 0.193 | 0.185 / 0.305 / 0.259 | 0.152 / 0.304 / 0.196 |
| nothing: plain SVD of W, no activation statistics | 0 | 0.217 / 0.719 / 0.357 | 0.393 / 0.654 / 0.552 | 0.332 / 0.679 / 0.539 |
| Q3_K for comparison (0.764× bytes) | | 0.040 / 0.117 / 0.059 | 0.076 / 0.121 / 0.088 | 0.069 / 0.129 / 0.069 |

- The learning curve is flat. Four times more calibration tokens lower the held-out error by 0.010 to 0.026.
- Two optimistic bounds stay far above Q3_K. Measured in-sample on the calibration tokens the base has 2.6
  to 4.3 times Q3_K's error; fitted on the held-out tokens themselves it has 2.2 to 3.3 times. Both use
  0.29 of Q3_K's bytes, and neither is achievable on new text.
- A calibration set matched to the evaluation text could at best move the base from the third row towards
  the in-sample rows. Combined with the no-structure curve of section 4.4, base plus 54 % of the blocks
  would still be above Q3_K (layer 2 gate: 0.105 × 0.68 = 0.07 against 0.040; PROJECTED).
- At rank 1024 (0.45×) the in-sample error on calibration is 1.3 to 3.3 times Q3_K's. The fit on the
  held-out tokens is over-parameterised at that rank (1,024 directions from 1,745 tokens) and bounds
  nothing.
- Activation statistics help at layers 2 and 57. At layer 30 a plain SVD of W is slightly better on
  held-out text for gate and up (0.393 and 0.654 against 0.414 and 0.663): the whitening picks up
  calibration-specific directions there.

**S2. Is the residual dense because of the 128-wide blocks?** For an oracle, largely yes. The cheap selector
tested captures the gain at the early layer only.

The same idea with block size 1 on the hidden side: one block is one FFN neuron, that is its gate row, its
up row and its down column (3 × 3,024 B at Q4_0). A fraction of the 21,504 neurons is computed exactly; the
others come from the rank-512 base or are dropped. Four selections:

- **oracle, no base**: neurons ranked by the true `|h_j| · ‖W_down[:, j]‖`, the others dropped. It needs the
  exact hidden activations `h`, so it is not deployable. It is a magnitude ranking, not a proven optimum.
- **oracle + base**: the same ranking, rank-512 base for the others.
- **base-predicted + base**: ranking from the base's own hidden activations. This is an online selector
  that decides without reading or computing any correction. Resident metadata: 43 KB of column norms per
  layer.
- **static set + base**: one neuron set for all tokens, ranked on calibration.

Composed FFN error, mean over the held-out tokens. Bytes are fractions of the layer's Q4_0 FFN bytes: the
neuron fraction alone without base, plus 0.225 with base.

| Layer | Neurons exact | Bytes without / with base | oracle, no base | oracle + base | base-predicted + base | static set + base |
|---|---|---|---|---|---|---|
| 2 | 10 % | 0.100 / 0.325 | 0.070 | 0.050 | 0.088 | 0.310 |
| 2 | 25 % | 0.250 / 0.475 | 0.027 | 0.030 | 0.057 | 0.244 |
| 2 | 50 % | 0.500 / 0.725 | 0.005 | 0.014 | 0.030 | 0.168 |
| 2 | 53.9 % | 0.539 / 0.764 | 0.004 | 0.012 | 0.027 | 0.157 |
| 2 | 76.4 % | 0.764 / 0.989 | 0.0005 | 0.004 | 0.013 | 0.095 |
| 30 | 10 % | 0.100 / 0.325 | 0.374 | 0.254 | 0.419 | 0.525 |
| 30 | 25 % | 0.250 / 0.475 | 0.189 | 0.174 | 0.350 | 0.464 |
| 30 | 50 % | 0.500 / 0.725 | 0.067 | 0.102 | 0.257 | 0.363 |
| 30 | 53.9 % | 0.539 / 0.764 | 0.057 | 0.093 | 0.243 | 0.339 |
| 30 | 76.4 % | 0.764 / 0.989 | 0.016 | 0.047 | 0.160 | 0.234 |
| 57 | 10 % | 0.100 / 0.325 | 0.260 | 0.178 | 0.248 | 0.317 |
| 57 | 25 % | 0.250 / 0.475 | 0.126 | 0.132 | 0.200 | 0.254 |
| 57 | 50 % | 0.500 / 0.725 | 0.043 | 0.077 | 0.144 | 0.182 |
| 57 | 53.9 % | 0.539 / 0.764 | 0.036 | 0.070 | 0.136 | 0.172 |
| 57 | 76.4 % | 0.764 / 0.989 | 0.009 | 0.034 | 0.090 | 0.107 |

| Against Q3_K (0.764× bytes) | L2 | L30 | L57 |
|---|---|---|---|
| Q3_K FFN error | 0.077 | 0.120 | 0.106 |
| Bytes to match Q3_K's error: oracle, no base (interpolated) | 0.09 | 0.36 | 0.29 |
| … oracle + base | 0.25 | 0.65 | 0.58 |
| … base-predicted + base | 0.37 | 1.07 | 0.90 |
| … static set + base | 1.03 | above 1.13 | 0.99 |
| Error at equal bytes: oracle, no base (76.4 % of neurons) | 0.0005 | 0.016 | 0.009 |
| … oracle + base (53.9 %) | 0.012 | 0.093 | 0.070 |
| … base-predicted + base (53.9 %); on generated tokens only | 0.027; 0.029 | 0.243; 0.264 | 0.136; 0.226 |
| … static set + base (53.9 %) | 0.157 | 0.339 | 0.172 |
| Exact gate and up, then the true top 25 % of down columns (0.75× bytes; deployable without a predictor) | 0.027 | 0.189 | 0.126 |
| Neurons that hold 90 % / 99 % of the ranking energy | 0.3 % / 5.7 % | 18 % / 51 % | 6.7 % / 38 % |
| Oracle, no base: tokens with error ≤ 1e-2 at 50 % / 75 % / 90 % of the neurons | 85 / 100 / 100 % | 0 / 3 / 100 % | 0 / 57 / 100 % |

- Organised per neuron, the corrections are concentrated. The oracle matches Q3_K's error with 0.09 to 0.36
  of the Q4_0 bytes, 2.1 to 8.7 times fewer than Q3_K. The 128-wide blocks hid this because the neuron order
  inside a block is arbitrary and because gate and up were selected on their dense input side.
- They are still not "few" for a tight tolerance. Error ≤ 1e-2 on 90 % of tokens needs 50 to 75 % of the
  neurons at layer 2 and 75 to 90 % at layers 30 and 57.
- The deployable selector is the weak part. Ranking from the base works at layer 2 (0.027 at equal bytes,
  0.35 times Q3_K's error). At layer 30 it is 2.0 times worse than Q3_K and at layer 57 1.3 times worse;
  on the generated tokens of layer 57 its error rises from 0.136 to 0.226 (Q3_K was not split by token
  type). To match Q3_K it needs 3.0 to 4.2 times the bytes of the oracle without base. In the terms of
  the pre-registered attribution this is "selection ineffective".
- Exact gate and up followed by a sparse down projection needs no predictor. It beats Q3_K at layer 2 only.
- A static neuron set never works (1.6 to 2.8 times Q3_K's error at equal bytes). The gain is token-dependent.
- Caveats. These are single-layer FFN errors with perfect inputs. At layer 2 a handful of very large
  neurons dominate (0.3 % of the neurons hold 90 % of the ranking energy), so a small relative error
  there says little about the remaining, token-specific part. The byte counts omit the Q4_0
  block scales of individually stored down columns (at most 0.037× of the FFN bytes, resident). One neuron
  is 9 KB: 25 to 54 % of the neurons are 5,400 to 11,600 scattered reads per layer and token, which rules
  out microSD and USB storage unless neurons that fire together are stored together.
- This is the mechanism of Deja Vu and LLM in a Flash (a low-rank predictor chooses FFN neurons). The
  research summary expected usable activation sparsity of at most 40 to 50 % for GELU models without
  retraining. At the middle and late layers, dropping half of the neurons leaves an oracle error of 0.07
  and 0.04, which is of the order of a 3-bit quantisation; the early layer is much sparser.

PROJECTED consequence: if every layer behaved like the worst sampled one (oracle parity with Q3_K at 0.36×),
a perfect selector would move 4.2 GB of FFN data plus 5.93 GB of attention and output head per token, 1.5
times less than a model with a 3-bit FFN (14.9 GB) at the same per-layer error. With the mean of the three
layers the factor is 1.7. The selector tested here reaches none of it beyond the early layer.

## 5. Limitations

- Three of 60 layers, all with sliding attention. The 60-layer byte figures assume they are representative.
- The calibration set is small (7,870 tokens for input widths of 5,376 and 21,504) and differs from the
  held-out set: raw text without chat template or generated tokens, no German. Section 4.7 (S1) measures
  how much of the base's weakness this explains: little.
- The reference is the dequantised Q4_0 file. The error of Q4_0 against the original checkpoint is not
  measured, and the QAT weights were trained for Q4_0, not for 3-bit formats or low-rank truncation.
- All errors are FFN-output errors of single layers with perfect inputs. Relative error in the residual
  stream, logits, argmax agreement and generated text were not measured. A per-layer FFN error of about 0.1
  is known to be usable (3-bit files of this model answer the Pi QA set); whether 0.2 to 0.4 is usable is
  unknown. Quantisation noise and dropped neurons are different kinds of error, and equal L2 error does not
  guarantee equal effect downstream.
- One representation and one fitting method: no training or fine-tuning of the factors, no learned
  predictor. The earlier quick study with a 2-bit uniform base and residual groups reached the same kind of
  result (`docs/results/H2_RESULTS_quick.md`, verdict NO_ADVANTAGE: 96 to 97 % of the residual at 3 %
  tolerance, 23.3 GB moved per token).
- The block oracle is per matrix with a fixed split of the byte budget between gate, up and down. A joint
  allocation could do somewhat better. The tolerance search, which lets the fractions differ per matrix,
  still needs 0.97 to 1.16 of the Q4_0 bytes to match Q3_K.
- Q3_K is our port, verified for format but not bit-identical to llama.cpp in its scale search. If the real
  quantiser is better, the gap widens.
- CPU times come from a float32 torch prototype on a desktop with 21 to 35 % background load. No CPU time
  was measured for the per-neuron variant. Nothing was measured on the Pi.
- The best rank and block layout were chosen on held-out data. This favours the method and does not rescue
  it.
- Section 4.7 is post-hoc. Its fractions and the rank of its base were not chosen in advance, and its
  positive oracle result has not been tested under a rule of its own.

## 6. Reproduce

```bash
# from the repo root, Git Bash; the three scripts hold the path of the activation dumps (C: scratch) in SCR / ACT
python -m pigemma.paths lock --hold -- bash results/dense/capture_acts.sh                       # ~65 min
python -m pigemma.paths lock --hold --min-free-gb 8 -- bash results/dense/run_analysis.sh       # ~70 min, 6 threads
python results/dense/d1_robustness.py                                                           # ~15 min, takes results/.cpu.lock
python results/dense/d1_derive.py                                                               # seconds
```

| Artifact | Content |
|---|---|
| `results/dense/preregistration.json` | representation, metric, baselines and decision rule, written 2026-10-05 18:51 local, before the fits |
| `results/dense/inputs/` | calibration text, held-out prompts, split description |
| `results/dense/L2.json`, `L30.json`, `L57.json` | every measured number per layer (isolated, composed, baselines, curves, fit times) |
| `results/dense/d1_summary.json`, `d1_tables.md` | rule evaluation, projection, full tables for all ranks |
| `results/dense/runtime.json` | prototype CPU times and background load |
| `results/dense/robustness.json`, `robustness.log` | post-hoc checks of section 4.7 (S1, S2) |
| `results/dense/d1_derived.json` | equal-error interpolation, no-structure curve, digest of S1 and S2 |
| `pigemma/h2/d1_lowrank_residual.py`, `pigemma/h2/d1_report.py` | analysis and rule code |
| `results/dense/capture_acts.sh`, `run_analysis.sh`, `d1_robustness.py`, `d1_derive.py` | drivers |
| activation dumps (2.1 GB, fp16) and cached fits (1.5 GB) | session scratch on C: (`…/scratchpad/dense/acts-cal`, `acts-heldout`, `fitcache`); not in the repo because D: is at its free-space floor; regenerate with `capture_acts.sh` |

## 7. Recommendation

**Stop this factorization. Do not build a Pi residual-streaming engine for it.** A low-rank base with
128-wide exact residual blocks loses to a plain 3-bit quantisation of the same layers by a factor 2.6 to 3.3
at equal bytes with oracle selection, and needs more than the full Q4_0 layer to come close to exact. The
earlier 2-bit-base variant failed the same way. The cause is in the representation (weak base, structureless
blocks), not in the selector or in the calibration size.

1. **Dense 31B on the Pi.** The working option remains a 3-bit-class file resident in RAM on the 16 GB
   board (MEASURED earlier: 0.53 tok/s with UD-IQ3_XXS). It is limited by compute. Attention and the output
   head are 5.93 of the 17.64 GB moved per token, and no FFN scheme touches them: even an FFN that cost
   nothing would cut the bytes per token by at most 3.0 times (PROJECTED).
2. **The one dense follow-up with a measured signal** is per-neuron dynamic selection (section 4.7, S2). It
   is not ready for engine work, and its upside for the Pi goal is small: about 1.5 to 1.7 times fewer bytes
   per token than a 3-bit model with a perfect selector (PROJECTED), for a model that is resident in RAM,
   not for one that streams from microSD or USB. If the dense track is continued at all, the next tests
   are, in this order and under their own pre-registered rule: (a) the same measurement on all 60 layers;
   (b) a selector that works at mid-depth, for example a trained predictor, with its bytes and time
   counted; (c) accumulated error in real generation with every patched layer active (step 6 of the brief);
   (d) a storage layout that turns 9 KB neurons into large sequential reads. Stop at the first failure.
3. **For stage X5 (dense additions).** The three items it was asked to add are measured here: the base
   alone, a static selection, and an online selector that does not read the corrections, at block level
   (section 4.5) and at neuron level (section 4.7), each against Q3_K at equal bytes. The layer-level
   result for the block representation is decisive, so no autoregressive test of it is justified.
4. **For the decision report.** Track C verdict: NEGATIVE for "base in RAM + streamed residual blocks" on
   the dense 31B. This says nothing about track A (exact MoE caching) or track B.
