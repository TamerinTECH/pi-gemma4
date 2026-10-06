# CF-Engine v1 implementation contract

Integration owner: root. Private experimental runtime, not an upstream submission.
Reference: llama.cpp a7b94df2c616bc1f62a73b964b4a71cb0dcc488e, CPU, no repack.
The first operation is Q4_0 MUL_MAT_ID for Gemma 4 experts; the initial profile
decides gate/up versus down enablement. No numerical or speed claim precedes tests.

## Packed tensors

Compiler runs offline in Python on desktop using existing gguf reader. Output is
a directory containing manifest.json and per-tensor payload files, no executable
code in an artifact. Manifest format `cf-engine-plan`, version 1. Each selected
tensor has name, type `Q4_0`, shape `[K,M,E]`, layout `q4-tile4-v1` or
`q4-bitplane4-v1`, relative file, bytes, sha256, source_sha256 (tensor bytes).
K must be divisible by 32, M by 4; E>=1. Unsupported cases are declared fallback.
Every original tensor is listed in inventory with type, shape, bytes and access
classification. Source file SHA256, size, architecture and full identity recorded.

Both payloads have exactly E*(M/4)*(K/32)*72 bytes. Index order:
expert, row tile, input block. Each 72-byte block starts with four unchanged
little-endian fp16 scale bit patterns (8 bytes), then 4*16 bytes of integer codes.
q4-tile4-v1: four original GGML Q4_0 qs[16] arrays; nibble low means index 0..15,
high means 16..31, signed value = nibble - 8. q4-bitplane4-v1: per row 16 bytes,
byte[group*2+bit/2] holds the four-input bitmask in low/high nibble for bit%2;
group=0..7 contiguous groups of four logical input indices; bit=0..3.
No requantization or permanent expanded weights. Roundtrip must be byte-identical.
Manifest includes separate decode/prefill schedules, bounded scratch lifetimes,
Pi target capabilities (A76 ASIMD/dotprod/fp16, no i8mm/SVE), and memory accounting.
Runtime rejects bad versions, paths, sizes, checksums, types, shapes, or source
tensor identity. SHA256 implementation is reused from llama.cpp vendor/hash.

## Kernel C++ API (namespace cf)

`kernels.h`: `enum class Layout { Tile4, Bitplane4 };`
`struct Q8Block { uint16_t d; int8_t qs[32]; };` sizeof=34, compatible GGML Q8_0.
`bool has_dotprod();`
`void build_lut(const Q8Block * x, size_t blocks, int16_t * lut);`
LUT is [blocks][8 groups][16 subset masks] int16, no scales, from q8 codes.
`void dot4_portable(const uint8_t * tile, const Q8Block * x, size_t blocks, float out[4]);`
`void dot4_arm(const uint8_t * tile, const Q8Block * x, size_t blocks, float out[4]);`
`void dot4_lut(const uint8_t * tile, const Q8Block * x, const int16_t * lut, size_t blocks, float out[4]);`
All functions consume one complete four-row tile (blocks*72 bytes), return raw
projection sums, and allocate nothing. Adapter uses existing Q8 activation
quantizer ONCE per distinct input (reference already does this). Down-projection
activation preparation stays per selected expert. ARM entry dispatches safely
or reports unavailable; portable path always testable. LUT uses bitplane layout.
Operator tolerance preregistration: relative L2 <=2e-5 and max absolute error
<=2e-4*max(1,maxabs(reference)); finite inputs and outputs. Compare to reference
Q4_0 x SAME Q8_0 codes, not float activations. Report float-activation difference
separately. End-to-end tolerances and routing tests are independently reported.

## Integration and profiling

Out-of-tree build generates an isolated source copy from pinned checkout; no
reference source/build mutation. Small CPU hook calls CF for declared supported
MUL_MAT_ID, otherwise original implementation. Existing persistent GGML worker
pool and scratch arena reused, no nested workers. Env CF_MODE=reference|cf;
CF_PLAN path; CF_KERNEL=portable|dotprod|lut; CF_REPORT JSON output. Public
harness option --engine reference|cf sets these explicitly. No plan => CF mode
fails. Runtime records each op/node, CF/fallback, prefill/decode, time, scratch,
preparation separately. Baseline profiling can run without any packed artifact.
Full-block reference normalization/GELU/router/scales remain unchanged initially.
Fusion is added only across proven compatible dependencies.

Only the integration owner runs Pi experiments, taking its bench lock and
checking services, >=2GB root disk and safe RAM/temperature. No global cache,
swap, service, boot or production edits. Pi cgroup-memory absence is a real gate.
