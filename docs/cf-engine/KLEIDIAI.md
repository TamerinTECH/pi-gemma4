# KleidiAI comparison: exact Q4_0 projection path

Status: **locally tested at operator level; full-model MoE integration remains blocked by the upstream selector**. Three independent Pi runs executed the real external ARM dot-product microkernel on captured Gemma 26B inputs, using both tile and range calls. Raw results are `results/cf-engine/operator-{1,2,3}.jsonl`; all same-code numerical checks passed. No full-model KleidiAI speedup is claimed.

## Pinned source and license

The reference llama.cpp `a7b94df` pins KleidiAI **v1.24.0** in `ggml/src/ggml-cpu/CMakeLists.txt`. This comparison uses its exact release archive:

- URL: https://github.com/ARM-software/kleidiai/releases/download/v1.24.0/kleidiai-v1.24.0-src.tar.gz
- Archive bytes: 2,466,038.
- MD5: `2f02ebe29573d45813e671eb304f2a00` (matches the pinned CMake value).
- SHA256: `9348b969e042d8890a54b01a463dbe71f5a4c074b5329e9c26a85ef3b68aa19b` (measured after download).
- Isolated directory: `third_party/kleidiai-v1.24.0`; original archive retained as `source.tar.gz`.
- All external files remain unmodified. Their SPDX notices and `LICENSES/Apache-2.0.txt` and `LICENSES/BSD-3-Clause.txt` are preserved. The selected three C source files carry Apache-2.0 notices.

The selected external arithmetic is disclosed as KleidiAI, not CF's own arithmetic. CF owns the small comparison wrapper, range scheduling boundary, reversible-byte checks and measurement integration. [Arm's primary repository](https://github.com/ARM-software/kleidiai).

To restore the isolated dependency on Linux, after checking the destination is appropriate:

```sh
mkdir -p third_party/kleidiai-v1.24.0
curl -fL https://github.com/ARM-software/kleidiai/releases/download/v1.24.0/kleidiai-v1.24.0-src.tar.gz \
  -o third_party/kleidiai-v1.24.0/source.tar.gz
echo '9348b969e042d8890a54b01a463dbe71f5a4c074b5329e9c26a85ef3b68aa19b  third_party/kleidiai-v1.24.0/source.tar.gz' | sha256sum -c -
tar -xf third_party/kleidiai-v1.24.0/source.tar.gz -C third_party/kleidiai-v1.24.0 --strip-components=1
```

## Exact operation and representation

The wrapper selects:

```
kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod
```

Its parameters are `mr=1, nr=4, kr=16, sr=2, bl=32`. This is one activation vector and four output rows per tile. The caller may invoke larger output-row ranges, but four-row calls allow an identical persistent-worker schedule to the initial CF path. Main real shape: input K=2816, output M=1408, eight selected experts; cache-resident control may use M=16. This path does not implement multi-token GEMM.

The official RHS packer is `kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0`. Its actual C implementation consumes the GGML Q4_0 scale-first block: two unchanged fp16 scale bytes followed by 16 bytes whose lower/upper nibbles encode positions 0–15/16–31. The original value is `(code - 8) * scale`. Header prose has an inconsistent illustration showing a trailing scale; the C implementation and llama.cpp integration are the governing evidence.

Packing interleaves four rows and eight-byte nibble segments, XORing nibble codes with `0x88` to change to the micro-kernel's signed-nibble representation. It does **not** requantize or change scale bits, including signed scales and signed zeros. For M divisible by four, the packed tensor has exactly the same `M*(K/32)*18` bytes. `prepare()` independently reverses the layout for every source byte and rejects any mismatch; verification cost is reported separately from packing. No permanent int8 or float weight expansion is used.

The wrapper rejects unsupported K/M, overflow, nonfinite weight scales, unavailable CPU features and invalid row ranges. It does not silently use another micro-kernel. Runtime availability requires Linux AArch64 ASIMD, dotprod and scalar/vector FP16. Only the selected C assembly source receives `-march=armv8.2-a+dotprod+fp16`; desktop instruction detection never defines the Pi target.

## Activation semantics and measurement

There are two explicitly different preparation modes:

1. `pack_q8(activation, reference_q8)` copies the already prepared **exact GGML Q8_0 scale bits and int8 codes**. For `mr=1`, Kleidi's input layout is byte-identical: `[fp16 scale, int8 codes[32]]` per block. This is the matched arithmetic comparison against CF/reference. Charge the reference Q8 preparation plus the measured copy; do not claim activation quantization disappeared.
2. `pack_activation(activation, float_input)` calls Kleidi's native `kai_lhs_quant_pack_qsi8d32p_f32`. It computes scale from the absolute maximum and uses `roundf`, whose ties round away from zero. The reference AArch64 GGML Q8_0 quantizer uses `vcvtnq_s32_f32`, ties to even. Therefore identical Q8 codes for arbitrary inputs are **not claimed**; compare native codes and output numerically and report any differences. Native preparation is timed separately, includes input checks, and may be shared only by experts receiving the same input.

The similarly named `kai_lhs_quant_pack_qsi8d32p_f32_neon` is **not an interchangeable replacement**: this pinned implementation requires `kr=4` and stores scales at the end of the row. The generic packer above is the companion registered by llama.cpp for the selected `1x4x32` kernel.

`Activation::pack_ms` measures the last preparation call. `Weights::pack_ms` measures external packing only; allocation and finite-scale scanning are outside that field, so the caller must also time all of `prepare()` for complete deployment cost. `Weights::validation_ms` measures reversible-byte checking. The `run()` function is allocation-free and does not collect time internally; caller wall timing therefore includes checks and external arithmetic without a clock call on each four-row tile.

`Weights::source_bytes`, `.packed.size()`, and `Activation::packed.size()` expose represented payload sizes. Count both original and packed live copies when both remain in the benchmark, plus capacities, output arrays, CF packed weights, metadata and process memory. Equal packed payload size alone does not establish equal total working-set memory.

## Build and caller boundary

Include `cf-engine/kleidi_sources.cmake` and link target `cf-kleidi-compare`. The source location can be set with `CF_KLEIDI_SOURCE_DIR`. On non-Linux/non-AArch64 builds only weight-packing validation is available; `available()` returns false and execution throws. No unavailable path may be counted as a Kleidi result.

```cpp
#include "kleidi_compare.h"
auto weights = cf::kleidi::prepare(original_q4_rows, 2816, 1408);
auto input = cf::kleidi::prepare_activation(2816);
cf::kleidi::pack_q8(input, existing_reference_q8);
// In the caller's existing persistent pool, for disjoint four-row ranges:
cf::kleidi::run(weights, input, whole_output_array, row_start, 4);
```

The pointer passed as output addresses the full M-row vector; `run` applies `row_start`. Packed weights and activations are read-only during multiplication, so expert/row tasks may execute concurrently. One shared activation is appropriate for gate/up. Different experts' down inputs need separate activation preparation.

## Dispatch applicability and current limits

The pinned llama.cpp Kleidi `tensor_traits::work_size` explicitly rejects operations other than `GGML_OP_MUL_MAT`; its buffer selector also requires a two-dimensional source weight tensor. `MUL_MAT_ID` and the Gemma expert stack therefore do not enter this integration. Merely building llama.cpp with Kleidi enabled would not prove that these MoE projections used it. [llama.cpp primary build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#arm-kleidiai).

The direct comparison extracts each selected expert as the valid two-dimensional projection and actually calls the named dotprod micro-kernel. This establishes a relevant arithmetic comparison once run. It is **not a full-model Kleidi integration**. The selected `1x4x32` GEMV appears in a pinned registry entry paired with an i8mm GEMM, so that registry entry's combined feature gate includes i8mm; the GEMV C source itself requires dotprod and no i8mm. Direct selection is explicit and independently guarded. The Pi's upstream default kernel may be another dotprod variant.

Before performance claims, compare same-code outputs on actual model rows against the reference, verify executed micro-kernel identity, and measure tiny-cache and selected-eight-expert working sets including preparation. Full-model timing, complete MoE block integration and end-to-end quality remain separate validation requirements. The integration owner's results are authoritative for locally tested status; this document contains no invented timings.

Implementation check: desktop C++17 syntax-only check passed. ARM compilation, runtime round trips, numerical tests and Pi timing remain pending the serialized integration-owner run.
