# CF-Engine: integrated A76 backend evaluation

## Outcome and scope

This iteration implements real C++ and separately assembled AArch64 kernels
inside complete CPU-only Gemma 4 26B-A4B Google QAT Q4_0 generation on
`context-friend-1`, a 16GB Raspberry Pi 5. It does not replace the 26B objective
with E2B. The deployment compiler, reversible compact formats, bounded native
adapter, portable kernel, NEON kernel and experimental `.S` backend are executable.
CF owns layer 0 gate/up and down, two of 60 expert projection tensors; the
reference executes the remaining graph. This is Milestone A integration, not
completion of the later fused-block, constrained-memory or algorithmic milestones.

The branch is `cf-engine`, with local uncommitted work in an originally unborn
repository. No commit or push was made. Existing builds and raw measurements
are preserved. Reproduction commands are in [README](../../cf-engine/README.md),
the [format specification](../../cf-engine/FORMAT.md) and
[assembly record](ASSEMBLY.md). [ARCHITECTURE](ARCHITECTURE.md) identifies owned
and reused components. The reference source remains
`a7b94df2c616bc1f62a73b964b4a71cb0dcc488e`.

Final performance trials are complete (section "Final timings" at the end;
added 2026-10-06 12:40 from the saved artefacts, because the session that ran
them ended before writing them down). Assembly is excluded from default builds;
`CF_EXPERIMENTAL_ASM=ON` exists only for explicit reproduction. It did not earn
performance acceptance. No assembly generator or speculative-decoding extension
was started before that decision gate.

## Exact model and execution contract

The original single GGUF SHA256 is
`3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.
The Pi uses the five verified splits recorded in
[split-identities.json](../../results/cf-engine/split-identities.json).
The inventory contains 392 F32, 265 Q4_0 and one Q6_K tensor. The selected
gate/up shape is `[2816,1408,128]`, down is `[704,2816,128]`, with eight routed
experts. Both packed payloads total 428,212,224 bytes, exactly their original
compact payload size. Scales, codes, signs and block semantics are unchanged.

The intended contract is semantics-preserving relative to this quantized GGUF
and pinned reference. No equivalence to an original higher-precision checkpoint
is claimed. Reference routing, GELU, normalization, router weights, residuals,
attention, tokenizer and greedy sampler are reused. Future tokens are never
treated as an available decode batch. Down inputs remain distinct by expert.

The reference already shares input Q8 preparation and groups prefill work by
expert. CF does not claim those ideas as new. Its four-row compact tile permits
reuse of one loaded activation block across four output rows. That changes
logical kernel loads compared with separate row dot products, but does not
establish fewer DRAM transactions. Upstream's packed GEMV already offers similar
reuse and is included in the comparison.

## Correctness: rejected candidate and correction

The first implementation passed operator tolerances and produced the same ten
greedy texts, but failed full-model logit and routing validation. The failure is
retained in [quality-asm-summary.json](../../results/cf-engine/quality-asm-summary.json).
Repeated reference requests were deterministic. The original portable, NEON and
assembly arithmetic reduced integer lanes before accumulating and used fused
operations where this pinned upstream A76 path uses separate FMUL/FADD. These
differences are consistent with the observed downstream routing divergence.
The original sources remain under `cf-engine/research/rejected-v1/`.

The corrected kernels preserve the reference's two parity accumulators and
four float lanes per row, scale/application order, final reduction and odd-tail
FMADD. A real disassembly check caught and fixed a CMake source-property scope
bug that had left the integrated C++ target contracting operations. Only owned
`kernels.cpp` receives `-ffp-contract=off`; upstream flags are unchanged.

| Validation | Result and evidence |
|---|---|
| Compiler/inverse packing | 46 tests, including actual source experts and all 65,536 scale bit patterns |
| Native malformed artifacts/options | 35 passed; [log](../../results/cf-engine/native-artifact-tests-exact.txt) |
| Pi independent numerical checks | 4,704 passed; max relative L2 `1.414576796e-7`, declared limit `2e-5` |
| Pi invalid-boundary detection | 520 guard-page cases, separate weight/input/output mappings at both boundaries |
| Pi calling convention | 312 callee-preserved-register probes passed |
| Actual upstream operator identity | All CF portable/NEON/assembly cases bit-identical on the captured projection at 1/2/4 workers; cache-sized and full selected-expert cases |
| Full autoregressive generation | 20 paired comparisons: ten held-out prompts each against NEON and assembly; all token IDs, text, stops, full logits and ordered routes match |

Fresh Pi tests are in [job-exact-guards](../../results/cf-engine/job-exact-guards/stdout.txt).
They include zero inputs, extreme finite scales/codes, alignments, block tails
and multithreaded output. Odd block counts have independent numerical/guard
coverage; bit identity to upstream for arbitrary odd lengths is not established.
The artifact compiler preserves arbitrary scale bit patterns; native loading
does not reject all NaN/Inf scales. See the explicit trust/validation limits in
the architecture record.

[exact-quality-summary.json](../../results/cf-engine/exact-quality-summary.json)
contains the complete full-model gate. Its limits were declared before the
failed candidate: relative L2 `2e-4`, absolute `2e-3`, KL `1e-5`, identical greedy
IDs and ordered routing. The corrected maximum logit difference is zero.
Independent raw verification compared 122 paired logit records, 31,981,568
float32 words including signed-zero bits, and 7,680 paired routing records.
The ten answers are correct on inspection: short explanation, extraction,
negation, numbers, one-line code, 217-token context and topic change. German and
Hebrew outputs are numbers/place names, not extended-language generation.
This is bounded evidence, not general quality certification or arbitrary
sampling-distribution proof.

## Measurement method

Only the integration owner runs Pi jobs. `pi_job.py` serializes experiments
with the existing lock and monitors services, available RAM, root disk,
temperature and throttling. It uses nice 10 and limits core dumps only for its
own process tree. No service, boot, swap or global cache setting is changed.

The initial real operation profile selected the gate/up family. Its cold-run
storage faults are material: [selection evidence](../../results/cf-engine/initial-projection-selection.json)
is not presented as resident arithmetic dominance. The final operator comparison
uses captured real gate/up inputs and expert IDs. The cache-sized case has
25,344 weight bytes per method; the actual eight-expert case has 17,842,176,
larger than the 2MB L3. No cache flushing or DRAM-counter claim is made.

The comparison executes actual pinned upstream vec_dot, actual upstream
CPU_REPACK 4x4 GEMV, and the pinned KleidiAI v1.24.0 DOTPROD microkernel, including
their coalesced row-range APIs. It includes Q8 preparation and persistent-worker
synchronization; reversible conversion costs and live duplicate buffers are
reported separately. Kleidi receives the exact reference Q8 codes. Its native
quantizer has a different tie rule and is separately labeled. Its whole-model
upstream selector does not support MUL_MAT_ID, so operator success is not a
whole-model Kleidi result. See [KLEIDIAI](KLEIDIAI.md).

Each independent operator run has 61 rotated/reversed iterations per method
and thread count. Full-generation trials alternate reference, NEON and assembly
within one fixed-model process, with three measured repetitions each after
warmup. `-nr` preserves original bytes for the narrow adapter. Timers and source
identity checks are explicit; startup and first-use checks are excluded only
from the labeled warm results, never from product startup costs.

## Resource and product limits

The corrected quality run measured 83.988 seconds of artifact/source identity
initialization and 2.237 seconds of first-use source-tensor hashing, in addition
to model loading and request work. Its peak process RSS was 14,751,536 KiB.
The adapter maps 428,212,224 packed bytes in addition to the source files
(14,439,364,160 bytes), allocates 1,531,904 scratch bytes and records 382,976
active scratch bytes in this run. Source and packed file cache duplication,
KV and runtime allocations remain part of the system cost; the payload equality
does not mean zero additional resident memory.

The mode called `resident` does not pin every page or certify zero I/O. Request
storage bytes, faults, process RSS/swap and health traces accompany results.
No suitable memory-controller byte counters were used. Logical weight sizes,
storage reads, cache traffic and DRAM traffic are distinct. No energy result is
claimed. The Pi has `cgroup_disable=memory`; no valid 8GiB application-budget
experiment was possible without an unauthorized device change.

The CLI adapter reuses bounded output, explicit context limits, exact-prefix
KV retention and cancellation from the existing native probe. It is one fixed
model/tokenizer/configuration per process, not a new concurrent-session server.
First-sample timestamps and sampled-token stalls are recorded; visible-channel
token timestamps are not, so first visible answer and useful-answer token speed
are not claimed. No real classroom calibration data was used.

The historical E2B/26B, bandwidth and parallel-stream corrections and all
competitor statuses are in [AUDIT](AUDIT.md). The 100 tok/s question remains a
traffic-reduction research question. This implementation has not demonstrated
algorithmic work reduction, a 20% matched decode win or 50% matched prefill win.

## Final timings (corrected, bit-identical kernels)

Sources: `results/cf-engine/exact-operator-{2,3,4}.jsonl` (summary
`exact-operator-summary.json`), `results/cf-engine/exact-sweep-t{1,2,4}-summary.json`,
health `results/cf-engine/health-summary.json` (all services active, no
throttling, max 72.7 C, no model swap).

Captured real gate/up projection, eight routed experts, 17,842,176 weight bytes
per method; median full-operator milliseconds of each of three independent
runs (61 alternated iterations each):

| Kernel | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| upstream `vec_dot` (the `-nr` path) | 3.503 / 3.505 / 3.501 | 1.787 / 1.784 / 1.782 | 1.557 / 1.822 / 1.770 |
| upstream CPU_REPACK 4x4 | 1.558 / 1.561 / 1.560 | 1.394 / 1.345 / 1.350 | 1.489 / 1.484 / 1.490 |
| upstream CPU_REPACK range | 1.536 / 1.537 / 1.535 | 1.364 / 1.347 / 1.320 | 1.480 / 1.470 / 1.476 |
| KleidiAI DOTPROD (same Q8 codes) | 1.777 / 1.782 / 1.776 | 1.368 / 1.329 / 1.359 | 1.517 / 1.514 / 1.515 |
| CF NEON | 2.812 / 2.813 / 2.811 | 1.444 / 1.430 / 1.440 | 1.521 / 1.569 / 1.545 |
| CF assembly baseline | 2.949 / 2.947 / 2.947 | 1.504 / 1.496 / 1.502 | 1.490 / 1.497 / 1.581 |
| CF assembly unroll2 | 2.948 / 2.947 / 2.946 | 1.507 / 1.497 / 1.499 | 1.483 / 1.742 / 1.511 |
| CF assembly prefetch512 | 3.164 / 3.144 / 3.158 | 1.650 / 1.632 / 1.648 | 1.517 / 1.529 / 1.570 |
| CF portable | 23.73 / 23.73 / 23.70 | 12.03 / 11.90 / 12.22 | 6.20 / 6.50 / 6.26 |

Full Gemma 4 26B-A4B QAT Q4_0 generation, reference / NEON / assembly
alternated inside one process, medians of three repetitions:

| Threads | Reference decode (prefill) tok/s | CF NEON | CF assembly | Reference share of graph time with CF on |
|---|---|---|---|---|
| 1 | 2.057 (3.294) | 2.062 (3.304) | 2.060 (3.302) | 98.9 % |
| 2 | 3.877 (6.502) | 3.886 (6.518) | 3.884 (6.516) | 98.9 % |
| 4 | 4.608 (11.693) | 4.627 (11.721) | 4.618 (11.706) | 98.7 % |

Decision gates: neither the 20 % matched decode gain nor the 50 % matched
prefill gain was reached (differences are below 0.5 %). Assembly loses to NEON
at one and two threads and has no repeatable win at four.

Interpretation (measured, not assumed): with two or more threads every
competent kernel converges on about 1.48 ms for 17.84 MB, i.e. about 12.0 GB/s,
which is the Pi's measured DRAM read bandwidth (12.5-14.1 GB/s). The operator
is limited by how fast the weights arrive, not by how they are multiplied. At
one thread the upstream repacked kernel is 1.8x faster than CF NEON because its
interleaved layout does more useful work per loaded byte; that advantage
disappears as soon as a second thread saturates the memory bus. Consequence for
the roadmap: Milestone B (a faster fused MoE block) cannot change decode speed
on this board. The levers that remain are fewer weight bytes per pass, several
committed tokens per pass under exact verification, and approximate execution.
