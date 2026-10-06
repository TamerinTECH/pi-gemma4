# CF-Engine A76 assembly candidate

The handwritten backend has executed real Q4_0 MoE projections inside 26B
inference. Assembly remains an experimental candidate and is disabled by default.
No assembly backend has been accepted for production. The original arithmetic
passed isolated numerical tests but failed the stricter full-model logit and
routing checks; those implementations and measurements are rejected as evidence
for the semantics-preserving path. Their sources remain in
[research/rejected-v1](../../cf-engine/research/rejected-v1/).

The corrected assembly matches the pinned A76 reference's accumulation order.
Rebuilt actual-shape comparisons report zero bit mismatches for portable,
intrinsics and all assembly entrypoints. Corrected NEON and baseline assembly
also passed [full-model revalidation](../../results/cf-engine/exact-quality-summary.json):
20 paired comparisons over ten held-out English, Hebrew and German requests,
with zero measured logit differences and matching routing and generated output.
Matched performance measurements remain pending; passing correctness does not
establish an end-to-end speed improvement.

## Owned code and execution boundary

[q4_tile4_a76.S](../../cf-engine/q4_tile4_a76.S) is handwritten GNU assembler;
[asm_kernel.h](../../cf-engine/asm_kernel.h) declares its C ABI. It consumes the
existing reversible `q4-tile4-v1` representation. No new weight format, precision
change, permanent expanded copy, tokenizer, router, or sampling implementation
was introduced. The signed integer code remains `nibble - 8`.

Three entrypoints share the signature:

```c
void kernel(const uint8_t *tile, const void *q8, size_t blocks, float *out);
```

The inputs contain `blocks * 72` weight bytes and `blocks * 34` Q8_0 bytes;
the output contains four float32 values. A block contains 32 input features and
four output rows. The caller validates dimensions, artifact identity and complete
input ranges before invoking the kernel. Zero blocks read neither input and
write four zeros. Odd block counts have an explicit tail. Weights can be
unaligned; the C++ Q8Block interface retains its normal two-byte alignment.

| Runtime option | Entry point | Change being tested |
|---|---|---|
| `asm` | `cf_q4_tile4_a76` | Reference-order arithmetic, early activation/scales loads, explicit schedule |
| `asm-unroll2` | `cf_q4_tile4_a76_unroll2` | Alias of corrected baseline; no longer a distinct candidate |
| `asm-prefetch512` | `cf_q4_tile4_a76_prefetch512` | Same arithmetic plus one bounded weight prefetch per block pair |

Only little-endian Linux AArch64 is supported by the assembly. Build integration
defines `CF_ENABLE_ASM=1` when linking its object. `has_asm()` also requires the
running CPU's `HWCAP_ASIMDDP`; big-endian AArch64 is explicitly rejected by the
resolver. `resolve_asm()` returns a verified function pointer or `nullptr`.
The adapter resolves once outside the tile loop and rejects an explicitly
requested but unavailable assembly backend. The standalone `dot4_asm()` helper
has a declared portable fallback for correctness tests.

The assembly uses baseline AArch64 integer/ASIMD/float instructions plus `SDOT`.
`FCVTL` converts stored binary16 scales to binary32; it does not perform FP16
arithmetic. No i8mm or SVE instructions are used. This describes the assembly
object's requirements, not universal portability of the surrounding executable:
the saved Pi build uses `-march=armv8.2-a+dotprod+fp16` for its CPU backend.

Arithmetic entrypoints use caller-saved x0-x6, v0-v7 and v16-v31. They do not
change SP, x19-x29, or the preserved low halves of v8-v15. The separate test-only
ABI probe saves/restores the caller's registers in a 160-byte aligned frame.

## Corrected reference arithmetic and actual instructions

The selected reference is the pinned CPU's no-i8mm, `nrc == 1`
`ggml_vec_dot_q4_0_q8_0` implementation in
[ARM quants.c](../../third_party/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c),
checked against its [actual Pi disassembly](../../results/cf-engine/upstream-q4.disasm.txt).
It keeps four floating-point lanes per output row and independent even/odd
K-block accumulators. SDOT lane `l` sums positions `4*l..4*l+3` and
`16+4*l..19+4*l`. Each lane is converted to float, multiplied by the product of
the two scales, then added to its parity accumulator. The observed reference
uses separate FMUL and FADD instructions in this loop. Each parity is reduced
as `(lane0 + lane1) + (lane2 + lane3)` only after all block pairs, then the two
parity totals are added.

The odd tail differs: it reduces the integer dot first, converts to float,
multiplies by the weight scale, then uses FMADD with the activation scale and
the previously reduced sum. This tail behavior follows the actual reference
instructions, not an assumption that all operations must avoid FMA. The real
projection widths, K=2816 and K=704, contain 88 and 22 blocks respectively and
do not use the odd tail.

Portable, intrinsics, LUT and assembly sources now follow this grouping. The
LUT integer groups `g` and `g+4` contribute to the same SDOT lane. The C++ source
requires `-ffp-contract=off` without floating-point reassociation. This must be
verified on the actual `ggml-cpu` object's compile command and disassembly,
not just on the standalone test target.

The first corrected build exposed exactly that integration issue: the CMake
source property initially applied in the root scope but did not reach the
`ggml-cpu` target created in a subdirectory. The saved
[initial corrected C++ object](../../results/cf-engine/kernels-exact.disasm.txt)
still has paired-loop FMLA at `0x5a4` (portable), `0xb80..0xb8c` and
`0xc34..0xc40` (intrinsics), and `0x185c` (LUT). Its actual-shape
portable/intrinsics bit comparisons failed. Reported standalone passes were
also insufficient because the build helper initially executed an older binary.
These are
failed-build diagnostic artifacts, not proof that the correction was deployed
to the complete runtime. The property now explicitly targets `ggml-cpu` and
`cf-kernel-test` after those targets exist.

The [rebuilt C++ object](../../results/cf-engine/kernels-exact-2.disasm.txt),
copied from the actual Pi `ggml-cpu` object, verifies the change. It contains no
FMLA. Portable's paired FMUL/FADD is at `0x5a0/0x5a4`; intrinsics uses separate
FMUL/FADD in `0xac4..0xc68`, followed by six FADDPs and a final FADD at
`0xc84..0xc9c`; LUT's paired FMUL/FADD is at `0x187c/0x1880`. The remaining
FMADDs at `0x9a8`, `0x1020` and `0x1554` are explicit odd tails in portable,
intrinsics and LUT respectively. The corrected intrinsics loop has 106
instructions per block pair, 15 load instructions and 16 SDOTs. Its wrapper
has a 160-byte frame; the handwritten arithmetic remains stackless. Assembly's
lower load-instruction count and larger total instruction count do not establish
a performance advantage. Both consume the same logical 212 input bytes per
block pair.

The [corrected assembly object](../../results/cf-engine/assembly-exact.disasm.txt)
does not depend on compiler contraction. Its baseline paired loop is
`0x30..0x1e8`: 111 instructions for two blocks, including 12 load instructions,
16 SDOTs, separate FMUL/FADD, and no integer horizontal reductions. The six
FADDPs and final FADD are outside that loop. The only FMLA is the explicit odd
tail at `0x2c4`. The prefetch loop `0x300..0x4c4` executes 114 instructions with
the hint or 113 when the hint is skipped; its only FMLA is likewise the odd tail
at `0x5a0`. `asm-unroll2` is a symbol alias of the baseline, not a second
measurement of a different algorithm. Counts exclude setup, final reduction,
tail and padding.

The arithmetic mismatch with the reference is confirmed. After the arithmetic
and build-flag corrections, full generation revalidation reports zero logit
differences and matching routing on the held-out suite. That supports this
mismatch as the explanation for the previous failures on these requests; it
does not prove correctness for every possible input. The matching output text
in the rejected run was insufficient to accept its changed logits or routing.

## Historical disassembly: REJECTED v1 arithmetic

The following counts describe the archived, rejected implementation, not the
current reference-order correction. They were extracted from saved Pi object
disassembly: [assembly v1](../../results/cf-engine/assembly.disasm.txt) and
[intrinsics v2](../../results/cf-engine/kernels-v2.disasm.txt). Padding NOPs and
function setup/teardown are excluded from loop counts.

| Loop | Address range | Instructions | Load instructions | SDOT | ADDP reductions |
|---|---|---:|---:|---:|---:|
| Intrinsics v2, one block | `0x500..0x5b8` | 47 | 8 | 8 | 3 |
| Assembly baseline, one block | `0x10..0xc8` | 47 | 6 | 8 | 3 |
| Assembly unroll2, two blocks | `0xf8..0x268` | 93 | 12 | 16 | 6 |
| Assembly prefetch, one block with hint | `0x340..0x404` | 50 | 6 plus PRFM | 8 | 3 |

The last seven blocks of the prefetch variant execute 49 instructions per loop,
because the hint is skipped. The unroll2 variant also executes its tail check and
a final float vector addition outside its main loop. Its 93 instructions per two
blocks should not be presented as a substantial reduction in arithmetic work.

The baseline's two paired vector loads replace four individual weight loads.
Extra pointer updates offset the instruction savings: its total loop count is
still 47. Both implementations issue the same logical 106 input bytes per block:
72 bytes of weights and 34 bytes of activations. Pair loads do not demonstrate
fewer DRAM bytes or fewer load micro-operations.

The observed compiler loop loads its two activation vectors at instructions 26
and 30, and its Q8 scale at instruction 37. The handwritten loop issues those
loads at instructions 1, 2 and 3, with the four weight scales at instruction 4.
This may overlap load latency with weight decoding. Both paths still execute
four ANDs, four shifts, eight subtracts, eight SDOTs and three ADDP reductions.
Whether the different schedule helps the A76 is an empirical question.

The saved intrinsics wrapper reserves an 80-byte frame and checks its cached
feature flag for every tile call. Its common path saves/restores 56 bytes of
registers each way. The assembly arithmetic function has no frame, and the
adapter's cached function pointer avoids that per-tile feature check. This is a
dispatch difference as well as a kernel difference; it should not be attributed
entirely to assembly arithmetic. Intrinsics could also use one-time dispatch.

The prefetch hint is issued only with at least eight blocks remaining. Its
address, current weight pointer plus 512, is therefore inside at least 576
remaining weight bytes. It changes fetch timing; it does not reduce required
weights and may be slower or increase traffic. It has no activation prefetch.

The strongest same-artifact comparison must also include upstream's existing
`ggml_gemv_q4_0_4x4_q8_0` in
[ARM repack.cpp](../../third_party/llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp).
Its reversible signed/interleaved representation enables eight shift/AND decode
operations and SDOT-lane accumulation directly into four output lanes, avoiding
the rejected implementation's three ADDP reductions. That is a different packing mechanism. Compare its
actual packer, kernel and integration costs; beating an unpacked row kernel is
insufficient evidence of an advantage over this available alternative.

## Numerical, bounds and ABI evidence

The corrected test binary was explicitly copied to the Pi and run; its saved
output is [job-exact-guards/stdout.txt](../../results/cf-engine/job-exact-guards/stdout.txt).
It reports assembly and ARM dotprod actually executed. An earlier build helper
had run an old test binary, so those intermediate test invocations do not
validate the correction. The fresh run reports:

| Check | Saved result |
|---|---|
| Numerical comparisons | 4,704 passed |
| Maximum relative L2 error | `1.414576796e-7`, limit `2e-5` |
| Maximum absolute error divided by `max(1,maxabs(reference))` | `1.430242702e-7`, limit `2e-4` |
| Separate weights/Q8/output guard-page cases | 520 passed |
| Callee-preserved-register probes | 312 passed |

The oracle decodes original Q4_0 bytes independently into double arithmetic and
uses the same Q8_0 integer codes. Cases include zero and extreme signed codes,
finite half-scale extremes and subnormals, packing round trips, all weight
alignments 0..15, and independent multithreaded outputs. Portable and intrinsics
implementations remain in the same test suite.

Guarded weights, activations and outputs are separate mappings, each placed
against its lower or upper inaccessible page in all eight combinations. Block
counts are `0,1,2,3,7,8,9,21,22,23,87,88,89`, covering zero/odd tails, prefetch
thresholds and the actual projection widths. The ABI probe checks x19-x29 and
the low 64 bits of v8-v15; normal return also exercises the link-register path.

The original [test run](../../results/cf-engine/pi-assembly-build/stdout.txt)
had also passed these categories. Such tests were necessary but insufficient:
all original variants used an arithmetic order that differed from the reference,
and the full-model
[quality result](../../results/cf-engine/quality-asm-summary.json) failed the
declared logit/routing checks despite all ten greedy texts matching. The
preregistered tolerance has not been relaxed. Exact reference operator checks
now supplement, rather than replace, the broader numerical and model checks.

The first corrected [operator run](../../results/cf-engine/exact-operator-1.jsonl)
records zero bit mismatches for every assembly entrypoint on the captured
gate/up projection, including its 11,264-output actual-shape case. In that same
case the incorrectly compiled C++ portable/intrinsics object has 7,089 bit
mismatches. After the build-flag correction,
[operator run 2](../../results/cf-engine/exact-operator-2.jsonl) records zero bit
mismatches for portable, intrinsics and the three assembly options, both for
the cache-sized 16-output case and the 11,264-output actual gate/up case at
1, 2 and 4 workers. LUT was not included in that direct upstream comparison.
These results establish equality only for the measured inputs and shapes, not
arbitrary inputs, all model operations or arbitrary sampling.

The actual supported projection widths have even block counts. Odd counts
`1,3` are included in the numerical suite, and `1,3,7,9,21,23,87,89` in the
guard/ABI suite. The saved upstream odd tail at `0xd78..0xd84` confirms integer
conversion, FMUL by weight scale, then FMADD by activation scale. Corrected
sources follow that order; regrouping the integer products cannot overflow
int32 for a 32-element Q4_0/Q8_0 block. Existing odd-count tests compare against
the independent double oracle at declared tolerances. Direct bit identity
against upstream for odd tails has not been established, and is not claimed.

The corrected [full-model quality report](../../results/cf-engine/exact-quality-summary.json)
passes all 20 requested pairs: NEON and baseline assembly each compared with
reference on ten held-out requests covering English, Hebrew, German, extraction,
numbers, negation, code, a longer prompt and a topic change. Every sampled-token
full-vocabulary comparison has zero relative-L2, maximum-absolute and KL
difference; routed expert IDs, generated token IDs, text and stop conditions
match. Input/backend identity checks pass, and operator diagnostic mode is
disabled, so candidate outputs participate in the autoregressive computation.
This establishes paired numerical agreement for the tested layer-zero path and
requests, not unrestricted task-quality or sampling guarantees. Prefetch, LUT
and the alias option were not separate full-model treatments in this suite.

## Integration evidence and remaining gate

The historical pre-correction
[pi-assembly-capture/profile.json](../../results/cf-engine/pi-assembly-capture/profile.json)
records `kernel_executed: cf-asm`, 44 CF projection calls and 33,682 reference
leading-operation calls during real 26B generation. The enabled tensors are
layer zero's gate/up `[2816,1408,128]` and down `[704,2816,128]` projections.
[Captured gate/up inputs](../../results/cf-engine/pi-assembly-capture/activations/blk.0.ffn_gate_up_exps.weight.json)
have one shared input vector; the
[down inputs](../../results/cf-engine/pi-assembly-capture/activations/blk.0.ffn_down_exps.weight.json)
have eight distinct vectors for the same eight routed expert IDs. The capture
retains the original routing and does not share post-nonlinearity activations.

This is a narrow integrated path, not a complete owned MoE block. The capture
run has operator validation disabled; its zero validation failures therefore
are not validation evidence. Its report includes first-use source hashing and
capture overhead. The saved generated continuation establishes actual execution,
not a matched quality or speed result.

Performance acceptance of the corrected path remains pending the integration
owner's matched operator, block and end-to-end trials, including upstream
repack and applicable external kernels. Full-model correctness has passed for
the measured NEON and baseline assembly treatments described above.
Use identical captured inputs/expert IDs, real model shapes, several thread
counts, working sets larger than the Pi cache and repeated alternating full
requests. Include preparation, packing, synchronization and fallback costs.
Retain intrinsics or the stronger compatible path if assembly does not win
reproducibly. No instruction-count result here is a DRAM-traffic measurement or
a single-stream throughput claim.

Rebuild provenance must distinguish the rejected initial kernels, the first
corrected C++ build with the missing flag, and the fully corrected runtime.
