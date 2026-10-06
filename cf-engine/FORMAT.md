# CF deployment artifact v1

The desktop compiler creates a directory containing `manifest.json` and one
`tensor-NNNN.cfq4` file per selected tensor. It reuses the existing
`pigemma.gguf_map.read_gguf_header` parser; it does not instantiate an inference
model. The only weight transformation is a reversible rearrangement of Q4_0
scale bits and integer bits. It does not requantize weights or store expanded
8-bit/floating-point weights.

```powershell
python cf-engine/compiler.py --model models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf --out artifacts/cf-26b-layer0-tile4 --layers 0 --projections gate_up --layout tile4
python cf-engine/compiler.py --verify artifacts/cf-26b-layer0-tile4/manifest.json --model models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf
python -m pytest cf-engine/test_compiler.py -q -m "integration or not integration"
```

`--layers` defaults to `0`; it also accepts `0,2-4` and `all`.
`--projections` is `gate_up`, `down`, or `both`; `gate_up` also selects separate
gate/up expert tensors when present. `--layout` is `tile4` or `bitplane`.
The output must be new or empty. The compiler never replaces an existing
artifact. A failed conversion may leave payloads but does not publish a usable
manifest; use a new output directory after inspecting the failure.

Giving a numbered GGUF split discovers all siblings and checks split count,
index, architecture, and complete tensor names. Every source tensor must have
valid dimensions, aligned file offsets, nonoverlapping byte ranges and known
GGML storage semantics. A known type unsupported by CF remains in the inventory
with a declared reference fallback. An unknown type whose size cannot be safely
determined is rejected. Compilation fails if no executable CF tensor is selected.

## Manifest

The top-level identity is `"format": "cf-engine-plan", "version": 1`. All sizes
and offsets are bytes and fit signed 64-bit addressing. Shapes are in GGUF order,
with input dimension first. Required sections are:

| Field | Meaning |
|---|---|
| `source.architecture` | Original GGUF architecture key. |
| `source.files[]` | Original file name/path, size, SHA256, GGUF version, alignment and tensor-data offset. Paths record provenance; execution must verify the model actually loaded. |
| `source.metadata` | Source metadata; large tokenizer arrays are summarized by type/count and covered by complete source hashes. |
| `inventory[]` | Every source tensor, not only Q4_0 or expert tensors. Includes name, GGML type/type ID, shape, bytes, source file index, absolute source offset, source tensor SHA256, logical access class, supported/selected flags and fallback reason. |
| `tensors[]` | Selected packed tensor records described below. |
| `target` | A76 target, candidate ASIMD/dotprod/FP16 features, excluded i8mm/SVE, cache capacities. Feature verification remains a runtime obligation. |
| `plans.decode` | Dynamic selected-expert row tiles for currently known input; no future-token batch. |
| `plans.prefill` | Known-token/expert work within the reference MUL_MAT_ID routing. This declaration does not claim an implemented weight-reuse speedup. |
| `scratch` | Maximum per-input Q8 and LUT bytes, per-worker accumulators, allocation owners, and operation-scoped lifetimes. Runtime must multiply by actual live input capacity. |
| `memory` | Source/packed sizes, explicitly counted duplicate representations, expert strides, and unknown runtime allocations. These are logical accounting estimates, not measured DRAM traffic. |

The CF adapter owns both Q8 activation and LUT arenas; the reference supplies
the persistent worker pool. `accumulator_bytes_per_worker` is 128 for both
layouts: two block parities, four rows, four lanes, and four bytes per float.
The bitplane/LUT path additionally records 64 integer-temporary bytes for its
four-by-four int32 block sums. These describe register/compiler-local arithmetic
state, not extra arena allocations or a complete stack/register bound. This
corrects older descriptive metadata (16 accumulator bytes and GGML Q8 ownership)
without changing payload layouts, artifact version, or native compatibility.
Previously tested manifests remain preserved as experiment inputs.

For a single-file source, `source.name`, `source.bytes`, and `source.sha256`
duplicate its file identity for convenience. File SHA256 and every original
tensor SHA256 are calculated in one sequential scan. The compiler packs selected
tensors in subsequent bounded reads and checks their hashes again, catching a
source change between inventory and conversion.

Each selected tensor contains these fields:

```json
{
  "name": "blk.0.ffn_gate_up_exps.weight",
  "type": "Q4_0",
  "shape": [2816, 1408, 128],
  "source_file": 0,
  "source_offset": 123456,
  "source_sha256": "<SHA256 of original tensor bytes>",
  "layout": "q4-tile4-v1",
  "file": "tensor-0000.cfq4",
  "bytes": 285474816,
  "sha256": "<SHA256 of packed file bytes>",
  "roundtrip": "byte-identical",
  "layer": 0,
  "projection": "gate_up",
  "kernel": "q4-dotprod-or-portable",
  "quantization": {
    "block_elements": 32,
    "source_block_bytes": 18,
    "scale": "unchanged-little-endian-fp16-bits",
    "integer": "unsigned-nibble-minus-8",
    "requantized": false
  }
}
```

The example offset and hash placeholders are explanatory, not an actual record.
The bitplane layout selects `q4-bitplane-lut`. An execution option may select the
portable arithmetic path; actual dispatch and fallback coverage must come from
runtime reports, never from the manifest alone.

`--alternative-source-identities identities.json` accepts a JSON list of
`{"name": "...", "bytes": N, "sha256": "..."}` records, for example independently
verified Pi split identities. These are explicitly **declared-only provenance**.
They do not replace original source identity or authorize accepting arbitrary
tensor data. Runtime compatibility across split/unsplit files depends on hashing
the actual selected tensor bytes by name/type/shape. Compiler `--verify --model`
verifies the original exact source file set; it deliberately does not treat
alternative file declarations as proof of equivalence.

## Payload order

Both layouts require `shape = [K, M, E]`, `K % 32 == 0`, `M % 4 == 0`, and positive
dimensions. There is no payload header or padding; the manifest owns metadata.
Each file contains exactly:

```text
E * (M / 4) * (K / 32) * 72 bytes
```

Ordering is expert, four-row tile, input block. For expert `e`, row tile `r`, and
input block `b`, its 72-byte unit starts at:

```text
((e * (M / 4) + r) * (K / 32) + b) * 72
```

Bytes 0..7 contain the four original little-endian fp16 scale bit patterns in row
order. A negative scale, zero, NaN bit pattern or any other 16-bit pattern is
copied exactly. No scale is converted through a floating-point value.

For `q4-tile4-v1`, bytes 8..71 contain each row's original 16 `qs` bytes. Low
nibbles represent logical indices 0..15; high nibbles represent indices 16..31.
The signed integer value is `nibble - 8`, exactly as in source GGML Q4_0.

For `q4-bitplane4-v1`, each row still occupies 16 code bytes. Divide its 32 logical
indices into eight contiguous groups of four. For `group=0..7` and `bit=0..3`,
the four-bit mask contains bit `bit` of each of the four unsigned Q4 codes, with
logical group position `j` stored in mask bit `j`. The mask occupies the low/high
nibble of `row_bytes[group*2 + bit/2]` according to `bit%2`. Signed correction
(`-8`) and unchanged scale application belong to the arithmetic kernel.

The compiler reads at most 4 MiB of compact source weights per chunk. NumPy
temporaries for transpose/bitplane assembly are bounded by this chunk, not by
model size. Its complete header and manifest remain resident. Temporary nibble
arrays exist only during offline conversion and are never part of the artifact.
Every chunk is inverted and compared byte for byte before it is accepted.

## Validation and limitations

`--verify` checks format version, known types, shapes, sizes, inventory ranges,
alignment, overlapping ranges, duplicate names/files, payload paths, payload
checksums, and SHA256 of the inverse layout against original tensor hashes.
Absolute paths, traversal paths, Windows drive paths and escaping symlinks are
rejected. `--verify --model` additionally checks original file sizes/hashes,
architecture, every inventory record against the actual GGUF header, and selected
source tensor hashes. A checksum manifest is an integrity format, not a digital
signature or a trust boundary for a hostile model supplier.

The test suite checks an independent scalar packing oracle, all 65,536 fp16 scale
bit patterns, short inputs, invalid/tail dimensions, source identity mismatch,
corrupted bytes/digests, inverse mismatch, unsupported versions, duplicate
records and invalid paths. Integration tests optionally read a single expert from
the actual desktop Google QAT 26B source for both gate/up and down, in both
layouts. They never load the full inference model.

The inspected 26B artifact contains 392 F32 tensors (46,056,568 bytes), 265 Q4_0
tensors (13,771,929,600 bytes), and one Q6_K tensor (605,552,640 bytes).
This inventory does not imply that all stored bytes are read per token. Expert
tensors use router-dependent accesses; lookup tables and output projections have
different access patterns. Measured DRAM traffic requires separate valid device
counters or a clearly labeled execution traffic model.

The compiler does not claim a complete fused MoE block, an adaptive memory policy,
a live KV allocation estimate, E2B optimization coverage, or a speed improvement.
Source and packed representations can both consume page cache; the manifest
counts their potential duplication. Runtime accounting must add actual KV,
lookup state, scratch, metadata and shared physical pages before making a bounded
memory claim.
