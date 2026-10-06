# CF-Engine v1 architecture record

The implemented boundary is a native executor for selected quantized expert
projections inside the existing complete Gemma 4 inference graph. The primary
model remains the Google Gemma 4 26B-A4B QAT Q4_0 GGUF and the primary device is
the CPU-only Raspberry Pi 5. This record describes the integrated path; current
performance conclusions belong in the experiment report, not in the design.

```text
Original GGUF + full tensor inventory
                 |
        offline compiler.py
                 |
    manifest.json + reversible Q4 payloads
                 |
    existing probe / tokenizer / model graph
                 |
      CF CPU hook for selected MUL_MAT_ID
                 |
    owned tile kernels + reused GGML worker pool
                 |
    reference graph dependencies and sampling
```

## Ownership and preservation

| Component | Implemented ownership and boundary |
|---|---|
| Deployment compiler | CF owns tensor selection, two reversible layouts, complete inventory, hashes, target declaration, decode/prefill descriptions and memory estimates. Existing `pigemma.gguf_map` owns GGUF parsing. Python runs offline. |
| Packed arithmetic | CF owns portable Q4/Q8 multiplication, ARM dotprod intrinsics, a bitplane/LUT path and provisional A76 assembly variants. Kernels consume compact four-row tiles without permanent expanded weights. |
| Executor | CF owns eligibility, explicit dispatch, selected-expert tile scheduling, preparation coordination, bounded activation/LUT buffers, validation and reporting. It uses the existing persistent GGML worker pool and introduces no nested pool. |
| Model/application compatibility | Pinned llama.cpp owns model loading, tokenizer, chat template, graph construction, attention/KV, routing, nonlinear functions, scales and sampling. The existing native probe owns request parsing, context limits, bounded generation, signal handling and request records. |
| Memory boundary | CF maps selected packed files and accounts for their sizes and owned buffers. Original weight mappings remain available for fallback. This is resident policy only, not an adaptive residency manager. |
| External arithmetic comparison | The isolated KleidiAI wrapper calls identified external kernels and discloses their licensing and preparation differences. It is a separate comparison, not the arithmetic in the initial CF full-generation run. See [KLEIDIAI.md](KLEIDIAI.md). |

`prepare.py` checks reference revision
`a7b94df2c616bc1f62a73b964b4a71cb0dcc488e`, creates an independent source overlay,
and inserts the CPU hooks there. Known-good checkouts, model artifacts, old
experiments, raw measurements, routing traces and cache simulator remain intact.
No obsolete experiment was deleted or silently repurposed. The new core replaces
only the selected operation dispatch behind `--engine cf`; `--engine reference`
runs the instrumented pinned reference through the same adapter.

## What actually executes

The initial plan selects these real tensors:

| Tensor | GGUF shape `[K,M,E]` | Packed bytes |
|---|---|---:|
| `blk.0.ffn_gate_up_exps.weight` | `[2816,1408,128]` | 285,474,816 |
| `blk.0.ffn_down_exps.weight` | `[704,2816,128]` | 142,737,408 |

These are two of 60 expert projection tensors across the model's 30 layers.
Other expert projections, shared/dense FFN, router, norms, attention, lookup
tables, output head and remaining graph operations use the reference. Coverage
is a runtime fact: the initial CF-only full-generation run reported 100 selected
projection calls, not 100 tokens or full-model kernel ownership.

The loader checks artifact version, selected/inventory identity consistency,
shape limits, payload bounds, paths, checksums and source file hashes. The first
execution of each selected tensor hashes its actual original weight bytes.
The supported original-byte hook uses `-nr`. Artifact dimensions outside
`K <= 2816`, `M <= 2816`, `E <= 128` reject initialization even if the more general
offline format can represent them. During execution, unsupported graph shapes
or batches beyond eight selected experts and 64 input tokens use a declared
reference path. CF mode with no
selected execution cannot silently succeed as CF evidence: normal completion
returns exit 5. The upstream `--version` handler exits before this check; native
loader tests use it strictly as an initialization control.

`dotprod` uses a guarded ARM implementation and reports portable arithmetic if
unavailable. `asm`, `asm-unroll2` and `asm-prefetch512` explicitly reject an
unavailable assembly path instead of reporting a false assembly result. Assembly
requires opt-in `CF_EXPERIMENTAL_ASM=ON`; it is excluded from default builds.
The corrected tile4 arithmetic follows the actual pinned A76 reference's four
dot-product lanes, separate even/odd block chains and unfused multiply/add
order. `kernels.cpp` is built with `-ffp-contract=off`; reassociative fast math
would invalidate that numerical contract. Captured operator bit identity does
not establish full autoregressive identity or a performance advantage.

The low-level guard checks compiled availability and the runtime dot-product
feature. The full ARM build targets the already verified A76/ASIMD/FP16 device;
the guard is not a general all-instruction compatibility validator for arbitrary
ARM64 hosts. Use a matching build and the recorded board capabilities.

## Preparation, scheduling and dependencies

The adapter reuses the reference Q8_0 quantizer. Gate/up input preparation is
shared only when selected experts receive the same input vector. Each selected
expert's post-nonlinearity down input is prepared separately. LUT preparation is
also per distinct prepared input; tables contain integer subset sums and use the
original weight/input scales during multiplication.

Decode schedules selected experts and four-row tiles for the current token.
Prefill groups the known token/expert work so a tile can be reused over its
assigned inputs. Work size controls parallel participation. Fixed shape and
batch bounds protect preallocated scratch, and unsupported larger shapes or
batches remain reference operations.

The reference already shares activation quantization and groups prefill work by
expert. Its Gemma graph already contains a combined gate/up weight tensor.
Those mechanisms are therefore reused baseline behavior, not a new algorithmic
CF advantage. The research question is whether the new layout, arithmetic and
coordination improve complete execution enough to repay their costs.

Gate/up, nonlinear activation and down currently remain separate graph
operations. The reference graph retains the routing algorithm, routing weights,
scale applications, normalization and quantization boundaries; CF executes every
selected expert. Identical downstream expert IDs still require numerical
evidence, because earlier projection errors can affect routing. **A fused
complete MoE block has not been implemented.**

## Representation and memory

The [v1 format specification](../../cf-engine/FORMAT.md) defines exact byte
ordering and validation. Both formats preserve each source Q4_0 scale bit and
integer code. Tile4 transposes four rows; bitplane4 rearranges the same nibble
bits into groups used by the LUT kernel. No source requantization or persistent
8-bit/float weight expansion occurs.

The offline verifier is stronger than the native loader: it reconstructs the
original tensor bytes from the packed layout, checks their hash, and checks the
complete inventory against source ranges and, with `--model`, the actual GGUF
header. The native loader verifies selected metadata consistency and hashes but
does not perform that inverse transformation or repeat every inventory-range
check. A trusted compiler-produced, offline-verified artifact is the supported
boundary; a self-consistent rewritten checksum manifest is not an authenticity
or arithmetic-correctness certificate.

There is currently **no finite-scale scan** in the compiler or native loader.
Layout conversion intentionally preserves all scale bit patterns, including
NaN/Inf. The `kernels.h` comment assigns finite-scale validation to callers; it
does not mean such validation has been implemented. Diagnostic output checks
and offline logit analysis reject nonfinite results when run, but ordinary
inference is not a general malformed-numerical-input validator.

The manifest's `plans`, `scratch`, `target` and `memory` sections record intent
and estimates. The current adapter consumes selected tensor identities/layouts
and uses fixed shape capacities, owned buffers and explicit CLI scheduling.
It is not yet an interpreter of the compiler's complete execution/memory plan.

The two selected packed files occupy 428,212,224 bytes (about 428 MB decimal).
Because the original source mappings remain, this is potentially **428,212,224
additional resident bytes**, not free layout metadata or a memory reduction.
Actual page-cache residency may differ. Source hashing and payload verification
also touch data and have recorded initialization/first-use costs.

The runtime reports mapped packed bytes, source file bytes, allocated and active
scratch, peak RSS, page faults and process I/O. These measurements have different
meanings and are not interchangeable. KV and other reference allocations are
part of process memory; manifest KV/metadata estimates remain explicitly unknown
until runtime configuration determines them. Summing logical tensor bytes does
not measure DRAM transactions, and zero storage reads do not establish cache or
DRAM traffic.

Only `--cf-memory resident` is supported. The Pi has its memory cgroup controller
disabled. Constrained mode rejects initialization; no boot/global configuration
change was made, and no valid 8 GiB application-budget result exists. Adaptive
expert residency, canonical representation reclamation and file-page ownership
audits remain Milestone C work. An application budget on the 16 GB board would
still not prove whole-device operation on 8 GB.

## Session and product boundary

The executable is an adapter to the existing probe, not a new HTTP service.
It keeps one model/native worker pool alive during a run, supports bounded
generation and explicit context sizing, and uses existing signal cancellation.
The integration workload uses `keep: 0` and clears KV between requests. The
reused probe already supports a per-item `keep` value: it checks exact prefix
token equality, removes positions after the retained prefix, and verifies the
remaining position range. Model, tokenizer and runtime configuration are fixed
within the process. It fails when required SWA state is unavailable; retention
beyond that window requires `--swa-full` rather than silent history loss.

This inherited exact-prefix path has not yet been validated with CF or measured
for reused-context latency. A stable application session API and multi-session
isolation evaluation remain future work. No semantic prompt cache or silent
global-attention history truncation is introduced.

Source identity checks, model startup, new-input processing and generated output
must remain separately visible. Current probe records are not sufficient to
turn all generated model tokens into a visible useful-answer speed claim.
Speech and classroom services are unchanged, and the calibration/integration
prompts are synthetic or public task text.

## Evidence and milestone limits

| Gate | Current evidence and limitation |
|---|---|
| Compiler correctness | 46 passing tests: independent scalar oracle, all scale bit patterns, bounds/corruption and four actual 26B single-expert roundtrips. |
| Native artifact loading | 35 passing tests on the native binary, with tiny synthetic source/payload files; valid initialization and explicit rejection paths, including schedule/LUT incompatibility and unavailable assembly. No model loading occurs in these tests. |
| Initial operator diagnostic | 5,372,928 values compared, zero failures against preregistered relative L2 `2e-5` and scaled max-absolute `2e-4` limits. Diagnostic execution then uses reference outputs, so it does not test accumulated candidate error. |
| Corrected captured operator | Tile4 portable, NEON and all three assembly variants have zero output-bit mismatches against actual upstream on captured gate/up input, cache-sized/full selected-expert cases and 1/2/4 threads. This is a bounded operator result. |
| Corrected kernel guards | 4,704 numerical cases, 520 guard-page cases and 312 assembly ABI cases pass. These do not replace full-generation validation. |
| Historical full generation | Initial 26B generation recorded 100 CF calls and matching token IDs/text on four short requests. The later assembly quality capture found logit and selected-expert-set differences despite matching text. That failed run is retained; it is not the corrected implementation's quality result. |
| Corrected full generation | A 31-request run includes one warmup and reference/NEON/baseline-assembly executions for 10 tasks. All 20 declared candidate/reference pairs pass input, backend, generated token/text, full-logit and ordered-routing checks. This is a bounded autoregressive result on this model, build and workload. |
| Milestone A | Working compiler, artifact, narrow native adapter and real optimized operations in full inference are integrated. |
| Milestone B | Partial: both projections in one MoE layer run through CF; full-block fusion, broad coverage and accepted end-to-end improvement remain unproven. |
| Milestone C | Blocked on an accounted constrained-memory environment; resident accounting exists, adaptive policy does not. |
| Milestone D | No new speculative executor or selective-correction module is claimed. They remain gated on measured execution headroom. |

The [corrected quality summary](../../results/cf-engine/exact-quality-summary.json)
reports zero logit differences and matching routing. An independent sequential
binary parse reaches both files' exact ends: 205 logit records and 12,840 routing
records across 31 requests. Raw float32 payloads match byte for byte for all 122
paired logit records (31,981,568 float32 words compared), including signed-zero
bits. Projection names, shapes and expert-ID payloads match in order across all
7,680 paired routing records. Request IDs are paired explicitly; they naturally
differ between reference and candidate capture headers.

The independent [checker](../../cf-engine/check_quality_bits.py) and
[saved result](../../results/cf-engine/exact-quality-bits.json) preserve this
test. The checker uses bounded 64 KiB payload reads, records source-file SHA256s,
rejects nonfinite logits, and verifies sampled-token coverage and total routing
projection counts against the request/profile records. It compares raw payload
bytes without float conversion or numerical tolerances. Full input/backend/output
validation remains the separate quality analyzer's responsibility.

Inspection of the ten reference answers finds no task-answer error or output
truncation: all stop at EOS. The ice explanation is correct; English extraction
and negation return `7` and `coffee`; German arithmetic/negation return `42` and
`Bonn`; Hebrew arithmetic/extraction return `42` and `3`; code is
`def square(x): return x**2`; the longer inventory returns `16`; the topic-change
cue returns `Berlin`. The Hebrew extraction prompt contains a grammatical
agreement typo, retained from the synthetic input; its numerical answer is still
correct. German/Hebrew output here is limited to numbers and a place name. The
topic-change task is a fresh request, not a test of multi-turn retained state.

No artifact test proves output quality, no short greedy match proves arbitrary
sampling equivalence, and no isolated kernel win is accepted as a full-model
gain. Repeat matched baseline/treatment runs in alternating order, retain
failures and variation, and count initialization, preparation, fallback work and
all live memory. The next kernel investment depends on those measured results.

Build, compiler, run and test commands are in the
[CF-Engine README](../../cf-engine/README.md). The
[focused audit](AUDIT.md) separates historical E2B/26B results, aggregate versus
single-stream timing, modeled traffic and unavailable physical counters.
