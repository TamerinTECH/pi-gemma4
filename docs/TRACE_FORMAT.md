# Trace and data formats (normative, format_version 1)

Frozen 2026-10-05, before any reader or writer was coded, so that the C++ probe
(`tools/probe/probe.cpp`), the C pinner (`tools/pinner/pinner.c`), the bash drivers
(`pc/`, `pi/`) and the Python package (`pigemma/`) agree byte for byte.

Authoritative sources, in this order:

1. `data_formats[]` in [design/manifest.json](design/manifest.json). Every entry is reproduced
   **verbatim** below in a block labelled "Verbatim contract".
2. This document. It refines the contracts: byte offsets, JSON types, units, protocols and
   validator rules that the manifest leaves open.
3. The real llama.cpp sources at commit `a7b94df` (build 11398) in `third_party/llama.cpp`.
   Where the manifest text and the real code disagree, the code wins. Section 0.3 lists every
   such case found while writing this file.

Background facts and measured numbers: [design/FACTS.md](design/FACTS.md),
[research/RESEARCH_SUMMARY.md](research/RESEARCH_SUMMARY.md),
[../results/desktop-baseline-2026-10-04.md](../results/desktop-baseline-2026-10-04.md).

The keywords MUST, MUST NOT, SHOULD and MAY are used in their RFC 2119 sense.

## Contents

- 0. Scope, producers and consumers, verified deviations
- 1. Conventions for every format
- 2. Machine-readable constants
- 3. `.moetrace` (expert routing trace)
- 4. Trace sidecars (`tokens.bin`, `gen.txt`, `meta.json`), the probe summary line, the replay chunk file
- 5. `stats.json` (trace statistics)
- 6. Tensor-name contract (what the probe hooks in llama.cpp)
- 7. Activation dump directory (H2) and the fixed 128-byte `.npy` header
- 8. `map.json` (GGUF byte-range map)
- 9. `hotset.json` and `hotset.ranges.txt`
- 10. Results JSON envelope
- 11. `sim.json` and `sim.csv`
- 12. `validation.json` and `io.csv`
- 13. H2 `oracle.json` and the decomposition `.npz` files
- 14. `facts.json`
- 15. `io_mem.json` and device-profile overrides
- 16. `catalog.json`
- 17. Change control

## 0. Scope, producers and consumers, verified deviations

### 0.1 What is covered

All binary and JSON or text interchange files of the experiment. A file that only one program
writes and reads, such as a log, is out of scope.

### 0.2 Producers and consumers

| Format | Written by | Read by |
|---|---|---|
| `.moetrace` | `pigemma-probe --mode moe-trace`; `pigemma.trace.TraceWriter` (merge, tests, synthetic traces) | `pigemma.trace`, `pigemma.cache_sim` |
| `<trace>.tokens.bin` | probe, `TraceWriter` | `pigemma-probe --replay`, `pigemma.trace` |
| `<trace>.gen.txt` | probe | `pigemma.quality.is_coherent` (through `pc/trace_collect.sh`), humans |
| `<trace>.meta.json` | `pigemma.trace.write_meta` (called by `pc/trace_collect.sh`) | humans, results write-up, `--replay-chunks` |
| `stats.json` | `python -m pigemma.trace stats` | `pigemma.gguf_map hotset`, write-up |
| acts directory (`L<il>.<kind>.npy`, `phase.u8`, `tokens.bin`, `meta.json`) | `pigemma-probe --mode act-dump` | `pigemma.trace.read_acts_dir`, `pigemma.h2.oracle` |
| `map.json` | `python -m pigemma.gguf_map build` | `gguf_map hotset`, `pigemma.cache_sim`, `pigemma.h2.decompose`, `pi/06_pin_hotset.sh` |
| `hotset.json`, `hotset.ranges.txt` | `python -m pigemma.gguf_map hotset` | `tools/pinner`, `pi/06_pin_hotset.sh`, `pc/pagecache_limit.sh` |
| results envelope | `pigemma.results.write_result` (bench, quality, cache_sim, pi scripts) | `python -m pigemma.results summarize` |
| `sim.json`, `sim.csv` | `python -m pigemma.cache_sim` | `pigemma.feasibility --sim`, write-up |
| `validation.json`, `io.csv` | `pc/pagecache_limit.sh` (with `pigemma-probe --io-csv`) | write-up, `tests/test_cache_sim.py` |
| `oracle.json` | `python -m pigemma.h2.oracle` | write-up |
| decomposition `.npz` | `python -m pigemma.h2.decompose` | `pigemma.h2.oracle` |
| `facts.json` | `pi/lib/common.sh` (via `pi/00_feasibility.sh`) | `pigemma.feasibility` (stdlib only) |
| `io_mem.json` | `pi/03_bench_io_mem.sh` | `pigemma.feasibility.load_catalog(io_mem=...)` |
| `catalog.json` | hand-written, refreshed by `gguf_map catalog-refresh` | `pigemma.feasibility`, `pigemma.cache_sim`, `pi/04_fetch_model.sh` |

### 0.3 Verified deviations and resolved ambiguities

These are binding. Implementers follow this list, not the conflicting manifest wording.

| # | Manifest says | Reality / resolution | Evidence |
|---|---|---|---|
| D1 | `ffn_moe_weights_norm` is F32 `[1,K,T]` | At a7b94df its `ne` is **`[K, T]`**: the weights are reshaped to 2-D and `ggml_div` returns that shape. The memory layout is identical (K contiguous floats per token, `nb[1] = 4K`). The probe MUST accept both `ne = [K,T,1,1]` and `ne = [1,K,T,1]`. The record format does not change. | `src/llama-graph.cpp:2153` (reshape_2d), `:2162` (div), `:2163` (callback name). The pre-normalisation `ffn_moe_weights` (`:2141`) is `[1,K,T]`, which is probably where the manifest text came from. |
| D2 | `llama_commit: 'a7b94df'` | `LLAMA_COMMIT` is git's short hash *as compiled*. Its length varies: the official Windows b11398 prints `commit a7b94df2c`. Compare commits by prefix (section 1). | `third_party/llama.cpp-win/llama-cli.exe --version` prints `version: 0.5.0-dev (build 11398, commit a7b94df2c)` |
| D3 | Sidecars are named `<stem>.tokens.bin`, `<stem>.gen.txt`, `<stem>.meta.json` | `<stem>` means **the full trace path including `.moetrace`**. So `traces/26b/all.moetrace` has `traces/26b/all.moetrace.tokens.bin`. This matches the probe spec (`<trace>.tokens.bin`) and the literal command in `pc/pagecache_limit.sh` (`--replay /work/traces/26b/all.moetrace.tokens.bin`). | manifest `files[]` entries for `tools/probe/probe.cpp` and `pc/pagecache_limit.sh` |
| D4 | The `.moetrace` JSON header has `chunks` "patched at close", but the header length is fixed when the first record is written | Writers reserve slack in the header (section 3.5). If the final header still does not fit, they rewrite the file. | this document |
| D5 | `trace.py` lists 18 required header keys; `data_formats` says "all required" (22 keys) | The validator checks **all 22** keys in section 3.4. | this document |
| D6 | `oracle.py` refers to "data_formats 'h2.json'" | The schema is `data_formats` entry "h2 oracle.json" (section 13). The file is `results/h2/oracle.json`. | manifest |
| D7 | `gguf_map.py` stores `expert_layers[].gate_up/down = {offset, nb_expert}` | `data_formats` adds `n_bytes`, and the superset is binding: `{offset, n_bytes, nb_expert}`. | manifest |
| D8 | The optional keys of `results.py` omit `spec_active`, `kld.ppl_q` and `kld.ppl_base` | These are first-class envelope keys per `data_formats` and are accepted at top level, or inside `kld`. Only keys absent from section 10 go under `extra`. | manifest |
| D9 | `sim.json` verdict enum is `FITS_RAM`, `STREAM_GOOD`, `STREAM_WEAK`, `NOT_VIABLE` | `cache_sim.py` also emits **`INFEASIBLE`** when a profile leaves no room for any expert cache (budget <= 0). Readers MUST accept it. | `files[]` entry `pigemma/cache_sim.py` |
| D10 | (not stated) gemma4 prunes the last layer to output rows | Only when `cparams.embeddings_nextn_masked` is true. The default is false (`src/llama-context.cpp:120`), so every layer, including layer `n_layer-1`, sees all T tokens. The probe MUST NOT enable nextn embeddings. Its per-layer column-count assertion (exit 4) detects a violation. | `src/models/gemma4.cpp:279` |

## 1. Conventions for every format

- **Byte order.** Every binary integer and float is little-endian. Both target CPUs (x86-64,
  aarch64) are little-endian. A C/C++ writer MUST check this at compile or run time and refuse to
  run on a big-endian host.
- **No padding inside binary records.** Fields are packed back to back. C/C++ writers serialise
  field by field into a byte buffer with `memcpy`. They never `fwrite` a struct, and never assume
  that an `f16` or `u32` field is aligned.
- **f16** is IEEE 754 binary16. Conversion from f32 rounds to nearest, ties to even:
  `ggml_fp32_to_fp16` / `ggml_fp32_to_fp16_row` in C, `ndarray.astype(np.float16)` in Python. Both
  give identical bits for the same f32 input. Values with magnitude above 65504 become ±inf.
- **Units.**

  | Suffix | Unit |
  |---|---|
  | `*_bytes`, `bytes`, `n_bytes`, `offset`, `length` | exact integer bytes |
  | `*_mb` | 10^6 bytes |
  | `*_mib` | 2^20 bytes |
  | `*_kb` | KiB (1024 bytes), as `/proc` and `/sys` report them (`ram_total_kb`, `rss_anon_kb`, `read_ahead_kb`) |
  | `*_gb` and CLI `--budgets-gb` | 10^9 bytes |
  | `*_gbps` | 10^9 bytes per second |
  | `*_mbps` | 10^6 **bytes** per second (not bits) |
  | `*_ms`, `*_s` | milliseconds, seconds |
  | `tok_s` | tokens per second |
  | `*_bits` | information in bits (log base 2) |

- **JSON.** UTF-8 without BOM. Non-ASCII text is allowed (Hebrew corpus names). `NaN` and
  `Infinity` are forbidden: write `null` (in Python, replace non-finite values with `None` and
  dump with `allow_nan=False`). Counts and byte sizes are JSON integers. Key order has no
  meaning. Readers MUST ignore unknown keys. Writers MUST NOT reuse a key defined here with a
  different meaning.
- **Timestamps** are ISO 8601 UTC with a `Z` suffix, e.g. `2026-10-05T09:12:44Z`. Fractional
  seconds are allowed.
- **Paths** are recorded as the writer saw them. Inside Docker, `/work/` is the repository root
  (the `pc/docker_run.sh` bind mount) and `/models/` is the `pigemma-models` volume
  (`pc/pagecache_limit.sh`). Readers treat `gguf_path` as an identifier: they compare the basename
  plus `gguf_size_bytes` and never assume the path is openable on the reading host. Repo-relative
  paths use forward slashes.
- **Build identity.** `llama_build` is an integer (11398). Two `llama_commit` strings are
  compatible when one is a prefix of the other and the shared prefix is at least 7 hex digits
  (`a7b94df` ~ `a7b94df2c`).
- **Honesty labels.** `kind` is `'measured' | 'projected' | 'assumed'` wherever it appears. A
  number that was not measured carries a `source` string that says where it came from. Catalog
  assumptions contain the word `assumed`.
- **Versioning.** `.moetrace` carries its version twice: in the magic (`...01`) and in
  `format_version: 1`. Any incompatible change bumps both and is written here first (section 17).
  JSON formats are versioned by this document. Adding an optional key is compatible.

## 2. Machine-readable constants

Tests (`tests/test_trace.py`) and implementations can parse the block between the two HTML
comment markers. Every line is `NAME = <Python literal>`, readable with `ast.literal_eval`.

<!-- pigemma-constants:begin -->
```python
MOETRACE_MAGIC = b'MOETRC01'
MOETRACE_FORMAT_VERSION = 1
MOETRACE_OFF_MAGIC = 0
MOETRACE_OFF_HEADER_JSON_LEN = 8
MOETRACE_OFF_RECORD_SIZE = 12
MOETRACE_OFF_N_RECORDS = 16
MOETRACE_OFF_RESERVED = 24
MOETRACE_OFF_HEADER_JSON = 32
MOETRACE_FIXED_HEADER_BYTES = 32
MOETRACE_HEADER_JSON_ALIGN = 8
MOETRACE_HEADER_PAD_BYTE = b' '
MOETRACE_N_RECORDS_OPEN = 0xFFFFFFFFFFFFFFFF
MOETRACE_HEADER_RESERVE_BYTES = 16384
MOETRACE_RECORD_FIXED_BYTES = 16
MOETRACE_REC_OFF_TOKEN_INDEX = 0
MOETRACE_REC_OFF_POS = 4
MOETRACE_REC_OFF_TOKEN_ID = 8
MOETRACE_REC_OFF_PHASE = 12
MOETRACE_REC_OFF_CHUNK_ID = 13
MOETRACE_REC_OFF_RESERVED = 14
MOETRACE_REC_OFF_LAYERS = 16
MOETRACE_BYTES_PER_LAYER_SLOT = 3
MOETRACE_RECORD_SIZE_FORMULA = '16 + 3*L*K + 3*L*P'
MOETRACE_MAX_EXPERTS = 256
MOETRACE_WEIGHT_SUM_TOL = 0.02
MOETRACE_PHASE_PREFILL = 0
MOETRACE_PHASE_GENERATED = 1
MOETRACE_WEIGHTS_TENSOR = 'ffn_moe_weights_norm'
MOETRACE_RECORD_SIZE_26B_P0 = 736
MOETRACE_RECORD_SIZE_26B_P16 = 2176
MOETRACE_BYTES_26B_100K_TOKENS = 73600000
SIDECAR_TOKENS = '.tokens.bin'
SIDECAR_GEN = '.gen.txt'
SIDECAR_META = '.meta.json'
GEN_CHUNK_SEPARATOR = '\n--- chunk {i} ---\n'
NPY_HEADER_BYTES = 128
NPY_MAGIC_AND_VERSION = b'\x93NUMPY\x01\x00'
NPY_HEADER_LEN_FIELD = 118
NPY_DESCR = '<f2'
ACTS_PHASE_FILE = 'phase.u8'
ACTS_TOKENS_FILE = 'tokens.bin'
PROBE_SUMMARY_PREFIX = 'PIGEMMA_SUMMARY '
RANGES_FIRST_LINE_PREFIX = '# pigemma-ranges v1 '
IO_CSV_HEADER = 'token_index,phase,wall_ms,read_bytes_delta,majflt_delta,rss_anon_kb,rss_file_kb'
SHA256_PREFIX_BYTES = 16777216
GEMMA4_26B_DATA_OFFSET = 15823936
GEMMA4_26B_EXPERT_UNIT_BYTES = 3345408
GEMMA4_26B_EXPERT_BYTES = 12846366720
GEMMA4_26B_NONEXPERT_BYTES = 1577172088
GEMMA4_26B_EXPERT_BYTES_PER_TOKEN_WORST = 802897920
GEMMA4_31B_DATA_OFFSET = 15833440
GEMMA4_31B_FFN_MATRIX_BYTES = 65028096
```
<!-- pigemma-constants:end -->

## 3. `.moetrace` (expert routing trace)

### 3.1 Verbatim contract

Verbatim contract: `data_formats[0]`, name ".moetrace (expert routing trace, binary, little-endian)" (copied unchanged from the manifest):

```text
HEADER: bytes 0-7 magic ASCII 'MOETRC01'; 8-11 u32 header_json_len (UTF-8 JSON, space-padded to a multiple of 8); 12-15 u32 record_size; 16-23 u64 n_records (0xFFFFFFFFFFFFFFFF while open, patched at close; readers infer from file size and flag truncated=True if unpatched); 24-31 u64 reserved=0; then the JSON header; records begin at 32 + header_json_len. JSON header keys (all required): format_version:1, model_name, gguf_path, gguf_size_bytes, llama_build:11398, llama_commit:'a7b94df', arch:'gemma4', n_layer:30, n_expert:128, n_expert_used:8, moe_layers:[0..29] (record order), n_embd:2816, probs_top:P (0 = none), weights_tensor:'ffn_moe_weights_norm', corpus_id, created_at (ISO UTC), n_ctx, n_batch, n_ubatch, n_threads, flags:[argv], chunks:[{chunk_id, source, n_prompt, n_gen}] (patched at close). RECORD (fixed size = 16 + L*K*3 + L*P*3 bytes; 26B with P=0: 736 B): u32 token_index (0-based position in this trace's token stream = index into tokens.bin) | u32 pos (position inside its chunk/context) | i32 token_id (the INPUT token at that position) | u8 phase (0 = prefill/teacher-forced, 1 = generated) | u8 chunk_id | u16 reserved=0 | for each il in moe_layers: K x u8 expert_id in ffn_moe_topk order (descending prob) then K x f16 normalised routing weight (ffn_moe_weights_norm; sum ~ 1) | if P>0, for each il: P x (u8 id, f16 prob) descending from softmax(ffn_moe_probs). Invariants enforced by validate(): ids < n_expert, K distinct ids per (token, layer), weights finite and sum within 0.02 of 1, token_index contiguous from 0, n_expert <= 256. Sidecars next to the trace: `<stem>.tokens.bin` (raw i32 LE token ids in stream order, one per record), `<stem>.gen.txt` (generated text per chunk), `<stem>.meta.json` (host fingerprint, gguf sha256_prefix, corpus files + sha256, argv, wall_ms, tok/s, callback_overhead_ms, coherent:bool), `<stem>.stats.json` (see stats.json).
```

### 3.2 File layout

```text
offset 0          32                  32+H                                   32+H+R*N
       +----------+-------------------+----------+----------+-----+----------+
       | fixed    | JSON header       | record 0 | record 1 | ... | record   |
       | 32 bytes | H bytes (incl.    | R bytes  | R bytes  |     | N-1      |
       |          | space padding)    |          |          |     |          |
       +----------+-------------------+----------+----------+-----+----------+
H = header_json_len (multiple of 8)   R = record_size   N = n_records
```

Records start at `32 + H`, which is always a multiple of 8. Nothing follows the last record.

### 3.3 Fixed header (32 bytes)

| Offset | Size | Type | Field | Rule |
|---|---|---|---|---|
| 0 | 8 | `char[8]` | `magic` | ASCII `MOETRC01` = `4D 4F 45 54 52 43 30 31`, no NUL terminator |
| 8 | 4 | `u32` LE | `header_json_len` (H) | byte length of the JSON header **including** its trailing space padding; `H % 8 == 0`; `H >= 2` |
| 12 | 4 | `u32` LE | `record_size` (R) | `16 + 3*L*K + 3*L*P` (section 3.7); MUST match the JSON header |
| 16 | 8 | `u64` LE | `n_records` (N) | `0xFFFFFFFFFFFFFFFF` while the writer is open; the real count after close |
| 24 | 8 | `u64` LE | `reserved` | 0 |
| 32 | H | UTF-8 | JSON header | one JSON object, then 0 or more `0x20` bytes up to H. No NUL, no BOM, no newline required |

Example: the first 32 bytes of a closed 26B smoke trace with 72 records. This is an illustration:
the H value depends on the JSON text; here it is 17,280 = `0x4380`.

```text
00: 4D 4F 45 54 52 43 30 31  80 43 00 00 E0 02 00 00   MOETRC01 H=17280 R=736
10: 48 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00   N=72     reserved=0
```

Parsing the JSON: decode bytes `[32, 32+H)` as UTF-8 and parse with `json.loads` or
`nlohmann::json::parse`. Both accept the trailing spaces as whitespace.

### 3.4 JSON header keys

Every key below is REQUIRED (D5). The example column shows the 26B smoke trace.

| Key | JSON type | Example (26B) | Probe source | Rule |
|---|---|---|---|---|
| `format_version` | int | `1` | constant | MUST be 1 |
| `model_name` | str | value of `general.name` | `llama_model_meta_val_str(model, "general.name")` | `""` if absent |
| `gguf_path` | str | `"/work/models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf"` | `-m` as given | identifier only (section 1) |
| `gguf_size_bytes` | int | `14439363584` | `stat()` | exact file size |
| `llama_build` | int | `11398` | `LLAMA_BUILD_NUMBER` | |
| `llama_commit` | str | `"a7b94df"` | `LLAMA_COMMIT` | prefix-compatible with `a7b94df` (D2) |
| `arch` | str | `"gemma4"` | meta `general.architecture` | |
| `n_layer` | int | `30` | `llama_model_n_layer` | all layers, MoE or not |
| `n_expert` | int | `128` | meta `<arch>.expert_count` | E, `1 <= E <= 256` |
| `n_expert_used` | int | `8` | meta `<arch>.expert_used_count` | K, `1 <= K <= E` |
| `moe_layers` | int[] | `[0, 1, ..., 29]` | layers that emit `ffn_moe_topk-<il>` | L = length, `L >= 1`; strictly increasing; every value in `[0, n_layer)`; this is the record order |
| `n_embd` | int | `2816` | `llama_model_n_embd` | |
| `probs_top` | int | `0` | `--probs-top` | P, `0 <= P <= E`; 0 = no probability block |
| `weights_tensor` | str | `"ffn_moe_weights_norm"` | constant | name of the tensor the weights came from |
| `corpus_id` | str | `"en_tech_moe_streaming"` | `--corpus-id`; else the `-f` basename without extension; else `"prompt"` | `"merged"` after merge; `"replay"` is the default for `--replay` |
| `created_at` | str | `"2026-10-05T09:12:44Z"` | time of open | ISO UTC |
| `n_ctx` | int | `4096` | `llama_n_ctx(ctx)` | effective value, not the requested one |
| `n_batch` | int | `2048` | `llama_n_batch(ctx)` | effective |
| `n_ubatch` | int | `256` | `llama_n_ubatch(ctx)` | effective |
| `n_threads` | int | `12` | `params.cpuparams.n_threads` | |
| `flags` | str[] | `["pigemma-probe", "--mode", "moe-trace", "-m", "..."]` | `argv` | complete argv, including `argv[0]` |
| `chunks` | object[] | `[{"chunk_id": 0, "source": "en_tech_moe_streaming", "n_prompt": 64, "n_gen": 8}]` | built while running, final at close | see below |

`chunks[i]` object:

| Key | Type | Meaning |
|---|---|---|
| `chunk_id` | int | `i`: the 0-based chunk index in stream order, **not** reduced mod 256. Records store `i mod 256`. |
| `source` | str | corpus identifier of the chunk's input: the trace's `corpus_id` for `-f`/`-p` input; for `--gen-prompts FILE` the prompts-file basename without extension; `"replay"` (or the corpus id) for `--replay` |
| `n_prompt` | int | number of phase-0 records in this chunk |
| `n_gen` | int | number of phase-1 records in this chunk |

Optional `chunks[i]` keys: `prompt_index` (int, position in the `--gen-prompts` list),
`prompt_id` (str, the item's `id`), `from_source` (int, index into the header's `sources`, after a
merge) and `pos0` (int, probe 1.1.0).

**Server sessions (probe 1.1.0, stage T1 of the decision workflow).** A `--gen-prompts` item may
carry `tokens` (exact prompt token ids, BOS included), `keep` (number of leading tokens whose KV
state is reused from the previous chunk, llama-server's prompt cache) and `batches` (sizes of the
prefill's consecutive `llama_process` calls). Such a chunk records **only the tokens that were
evaluated**: `n_prompt` = prompt tokens - `keep`, its records start at `pos = keep`, and the header
entry carries `pos0 = keep` (absent or 0 = fresh context; `validate` expects
`pos = pos0 + offset`). With `--server-stop` (header key `server_stop: true`) the token that ends a
reply (EOG, or the `max_tokens`-th token) is sampled but not evaluated, as in llama-server: a reply
of n sampled tokens has n - 1 phase-1 records, and the last sampled token is found only in the
`--requests-out` sidecar. That sidecar (`<trace>.requests.jsonl` by convention) has one JSON line
per chunk: `chunk_id`, `id`, `n_prompt_tokens`, `n_keep`, `n_prefill`, `prefill_calls`,
`max_tokens`, `n_sampled`, `n_gen_evaluated`, `stop` (`eos` / `limit` / `none` / `interrupted`),
`gen_tokens` (every sampled id), `text` (pieces rendered without special tokens, like the server's
`content`), `t_prefill_ms`, `t_total_ms`, `sampled_ms` (wall clock of each sampling relative to the
chunk start) and `trace_first_record`. Prefill ubatches are the consecutive `n_ubatch`-sized pieces
of each `prefill_calls` entry.

Optional header keys (writers MAY add them; readers ignore the ones they do not know):
`sources` (str[]: input traces of a merge, in order), `replay`
(`{"tokens_file": str, "range": [A, B]}`), `chat` (bool, `--chat` used), `gen_max` (int, the
`--gen` value) and `notes` (str[]).

`moe_layers` is known only once the first chunk has run. A probe that wants it from the observed
tensors therefore writes the file header when it flushes the first chunk's records. It MAY instead
derive `moe_layers` from model metadata (the layers that have `blk.<il>.ffn_gate_inp.weight`).
Either way, the list MUST equal the set of layers that emit `ffn_moe_topk-<il>`.

### 3.5 Writing and closing protocol

1. **Header with reserve.** Serialise the header with `"chunks": []` (or with the final chunks
   if already known). Set `H = round_up_8(len(json_bytes) + 16384)`, where 16384 is
   `MOETRACE_HEADER_RESERVE_BYTES`. Write the 32 fixed bytes with `n_records =
   0xFFFFFFFFFFFFFFFF`, then the JSON, then spaces up to H. A writer that knows the final header
   at open (merge, unit-test fixtures) MAY use a reserve of 0.
2. **Records.** Append each record as exactly R bytes. Append the same `token_id` to
   `<trace>.tokens.bin` (4 bytes, i32 LE) in the same order. The probe flushes a chunk's prefill
   records only after `llama_process` returns and every layer has delivered exactly
   `chunk_len` columns. Generated records are flushed one by one.
3. **Close.** Serialise the final header (complete `chunks`).
   - If its length is `<= H`, overwrite bytes `[32, 32+H)` with it plus space padding.
   - If it is longer, rewrite the file: write `<trace>.tmp` with `H' = round_up_8(len + 16384)`,
     copy the records unchanged, then rename it over the original. With the 16 KiB reserve this
     happens only past roughly 150 chunks.
   - Write `n_records` at offset 16 **last**, then flush and close. A crash before this step
     leaves the sentinel, so readers infer N and flag the trace as truncated.
4. **SIGINT/SIGTERM.** The probe's handler sets a flag. The main loop finishes the current
   `llama_process` call, flushes the completed records and closes normally. A handler MAY also
   `pwrite` `n_records` directly (async-signal-safe), accepting that `chunks` can then be stale.
   The validator flags that case with warning W2.

### 3.6 Record layout

| Offset in record | Size | Type | Field |
|---|---|---|---|
| 0 | 4 | `u32` | `token_index`: 0-based position in this trace's stream; record `i` has `token_index == i` |
| 4 | 4 | `u32` | `pos`: position inside its chunk (= KV position; restarts at 0 for every chunk) |
| 8 | 4 | `i32` | `token_id`: the **input** token processed at `pos` |
| 12 | 1 | `u8` | `phase`: 0 = prefill / teacher-forced, 1 = generated |
| 13 | 1 | `u8` | `chunk_id`: chunk index mod 256 |
| 14 | 2 | `u16` | `reserved` = 0 |
| `16 + 3*K*j` | K | `u8[K]` | `ids` of layer `moe_layers[j]`, `j = 0..L-1`, in `ffn_moe_topk` order (descending router probability) |
| `16 + 3*K*j + K` | 2K | `f16[K]` | `weights` of layer `moe_layers[j]`; `weights[k]` belongs to `ids[k]`; values of `ffn_moe_weights_norm` |
| `16 + 3*L*K + 3*(P*j + p)` | 1 | `u8` | only if P > 0: id of the p-th most probable expert of layer `moe_layers[j]`, `p = 0..P-1` |
| `16 + 3*L*K + 3*(P*j + p) + 1` | 2 | `f16` | its probability, the value of `ffn_moe_probs` |

So each layer contributes one contiguous 3K-byte slot `[ids(K) | weights(2K)]`. The optional
probability block follows all L slots as `L*P` interleaved `(u8 id, f16 prob)` triples.

Probabilities are the values of `ffn_moe_probs`, which for gemma4
(`LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`) are already `softmax(ffn_moe_logits)` over all E
experts. Writers MUST NOT apply softmax again. Order is descending probability; ties go to the
lower expert id.

Worked layout for the 26B-A4B (L = 30, K = 8, P = 0, R = 736):

| Bytes | Content |
|---|---|
| 0-15 | `token_index`, `pos`, `token_id`, `phase`, `chunk_id`, `reserved` |
| 16-23 | layer 0: 8 expert ids (u8) |
| 24-39 | layer 0: 8 weights (f16) |
| 40-47 / 48-63 | layer 1: ids / weights |
| `16+24j` - `23+24j` / `24+24j` - `39+24j` | layer j: ids / weights |
| 712-719 / 720-735 | layer 29: ids / weights |

Reference NumPy dtype, which `pigemma.trace` uses. It is packed (`align=False`) and its
`itemsize` equals the formula (verified: 736 for (30,8,0), 2176 for (30,8,16), 112 for (4,8,0),
28 for (2,2,0)):

```python
import numpy as np

def record_dtype(L: int, K: int, P: int) -> np.dtype:
    fields = [("token_index", "<u4"), ("pos", "<u4"), ("token_id", "<i4"),
              ("phase", "u1"), ("chunk_id", "u1"), ("reserved", "<u2"),
              ("moe", [("ids", "u1", (K,)), ("w", "<f2", (K,))], (L,))]
    if P > 0:
        fields.append(("probs", [("id", "u1"), ("p", "<f2")], (L, P)))
    return np.dtype(fields)          # itemsize == 16 + 3*L*K + 3*L*P
```

Reference C++ serialisation, as used by the probe:

```cpp
std::vector<uint8_t> rec(record_size, 0);
std::memcpy(&rec[0], &token_index, 4);
std::memcpy(&rec[4], &pos, 4);
std::memcpy(&rec[8], &token_id, 4);
rec[12] = phase;
rec[13] = (uint8_t)(chunk_index & 0xFF);       // rec[14..15] stay 0
for (int j = 0; j < L; ++j) {
    uint8_t * slot = &rec[16 + 3 * K * j];
    for (int k = 0; k < K; ++k) {
        slot[k] = (uint8_t) ids[j][k];
        ggml_fp16_t h = ggml_fp32_to_fp16(w[j][k]);
        std::memcpy(slot + K + 2 * k, &h, 2);
    }
}
// P > 0: uint8_t * q = &rec[16 + 3*L*K + 3*(P*j + p)]; q[0] = id; memcpy(q + 1, &h_prob, 2);
```

### 3.7 `record_size` and sizes

`record_size = 16 + 3*L*K + 3*L*P`, where L = `len(moe_layers)`, K = `n_expert_used` and
P = `probs_top`.

| Case | L | K | P | record_size | 1,000 tokens | 100,000 tokens |
|---|---|---|---|---|---|---|
| 26B-A4B, routing only | 30 | 8 | 0 | **736 B** | 736,000 B | **73,600,000 B = 73.6 MB** |
| 26B-A4B, `--probs-top 16` | 30 | 8 | 16 | 2,176 B | 2,176,000 B | 217.6 MB |
| 26B smoke golden trace (`tests/fixtures/golden_26b_72tok.moetrace`, 72 records) | 30 | 8 | 0 | 736 B | records = 52,992 B | |
| `tests/synth.py` default (L=4, K=8, E=16) | 4 | 8 | 0 | 112 B | 112,000 B | |
| `tiny_moe_gguf` fixture (2 layers, K=2) | 2 | 2 | 0 | 28 B | | |

Add `32 + H` bytes of header (H is at least about 16.4 KiB with the reserve) and 4 bytes per token
for `<trace>.tokens.bin` (400 KB per 100k tokens).

### 3.8 Stream semantics

- **Stream order.** Records are in processing order and record `i` has `token_index == i`. The
  stream is the concatenation of the chunks. Within a chunk, all prefill records come first
  (phase 0, `pos = 0..n_prompt-1`), followed by the generated records (phase 1,
  `pos = n_prompt..n_prompt+n_gen-1`).
- **Chunks** are fresh contexts: the probe calls `llama_memory_clear` before each one, so `pos`
  restarts at 0. Chunk `i` covers `token_index` in `[S_i, S_i + n_prompt_i + n_gen_i)`, where
  `S_i` is the sum of `n_prompt + n_gen` over the earlier chunks. Records store `chunk_id = i mod 256`;
  the header's `chunks[i]` is authoritative for the full index. Each chunk SHOULD start with the
  model's BOS token at pos 0: when the probe splits one tokenised text into chunks, it inserts BOS
  at the start of every chunk that lacks one, and that BOS is an ordinary phase-0 record.
- **What a record means.** The routing in a record was computed while processing `token_id` at
  `pos`, so it depends on the chunk's tokens at positions `<= pos`. Teacher-forced prefill routing
  therefore equals the routing the model would produce while generating the same text. The only
  differences are floating-point effects from different ubatch sizes (different matmul kernels),
  which is why E5 replays with `-ub 1`, like decode.
- **Generated records.** With `--gen N`, the probe takes the greedy argmax of the previous
  position's logits, feeds that token at the next position, and records it (phase 1,
  `token_id` = the fed token). It produces at most N phase-1 records per chunk. An
  end-of-generation token (`llama_vocab_is_eog`) stops generation and is never fed, recorded or
  written to `gen.txt`.
- **Replay** (`--replay tokens.bin [--replay-range A:B]`) feeds `tokens[A:B]` (half-open, Python
  slice semantics; an empty B means the end). Every replay record is phase 0, because nothing is
  sampled; the source trace still knows the original phase at `token_index + A`. `token_index`
  restarts at 0. Chunk boundaries come from `--chunk-tokens`, counted from A, and/or from
  `--replay-chunks` (section 4.6), given as absolute source-stream indices and intersected with
  `[A, B)`. The header gets the optional `replay` key.

### 3.9 Validator: `pigemma.trace.validate(path) -> list[str]`

`validate` returns **errors only**. Each string starts with its code (`"E10: layer 3 token 17:
id 131 >= n_expert 128"`). Warnings go through `warnings.warn` or logging and never into the
returned list. `python -m pigemma.trace validate` exits 1 if and only if the list is non-empty.

| Code | Error (MUST reject) | Manifest invariant |
|---|---|---|
| E1 | `magic != b'MOETRC01'` | magic |
| E2 | `H % 8 != 0`, `H < 2`, `32 + H > file size`, or the JSON bytes are not UTF-8 / not a JSON object followed only by spaces | |
| E3 | any key of section 3.4 missing or of the wrong JSON type; `format_version != 1` | header keys present |
| E4 | `n_expert` outside `1..256`; `n_expert_used` outside `1..n_expert`; `probs_top` outside `0..n_expert`; `moe_layers` empty, not strictly increasing, or with a value outside `[0, n_layer)` | `n_expert <= 256` |
| E5 | fixed-header `record_size != 16 + 3*L*K + 3*L*P` | record_size formula |
| E6 | `(file_size - 32 - H) % record_size != 0` (truncated record) | truncated record |
| E7 | `n_records` is not the sentinel and `!= (file_size - 32 - H) / record_size` | |
| E8 | `token_index[i] != i` for some i | contiguous from 0 |
| E9 | `phase` not in {0, 1}, or `reserved != 0` | |
| E10 | an id `>= n_expert` | `ids < n_expert` |
| E11 | the K ids of a (token, layer) are not pairwise distinct | K distinct ids |
| E12 | a weight is non-finite, or `abs(sum_k w - 1) > 0.02` for a (token, layer). Applies when `weights_tensor == 'ffn_moe_weights_norm'`; otherwise skipped with W5 | weights finite, sum within 0.02 |
| E13 | P > 0 and a prob id `>= n_expert`, a duplicate prob id within a (token, layer), or a prob that is non-finite or outside `[0, 1]` | |
| E14 | `<trace>.tokens.bin` exists and its size `!= 4 * N` | tokens.bin length |

| Code | Warning (report, do not reject) |
|---|---|
| W1 | `n_records` is the sentinel: the trace was not closed. The reader infers N and sets `truncated=True` |
| W2 | chunk bookkeeping is inconsistent: `sum(n_prompt + n_gen) != N`, a record's `chunk_id != i mod 256`, a phase-1 record precedes a phase-0 record within a chunk, or `pos` is not contiguous from 0 within a chunk |
| W3 | weights of a (token, layer) are not non-increasing within 1e-3 (ids should be in descending-probability order) |
| W4 | `P >= K` and `set(first K prob ids) != set(ids)` for a (token, layer) |
| W5 | `weights_tensor != 'ffn_moe_weights_norm'` (sum check skipped) |
| W6 | `llama_build != 11398` or `llama_commit` is not prefix-compatible with `a7b94df` |
| W7 | `<trace>.tokens.bin` is absent |

### 3.10 Reader rules (`TraceReader`)

- Parse the fixed header and the JSON. Compute `data_start = 32 + H`.
- If `n_records` is the sentinel, set `N = (file_size - data_start) // record_size`, set
  `truncated = True` and warn. A partial trailing record is ignored by the reader and is an E6
  error in the validator.
- Map records with `np.memmap(path, dtype=record_dtype(L, K, P), mode="r", offset=data_start,
  shape=(N,))`. Expose `ids` as uint8 `[N, L, K]`, `weights` as float32 `[N, L, K]`, `phase`,
  `chunk` and `token_index`. Expose `token_ids` from `<trace>.tokens.bin` when present, and
  `probs` as `(id uint8 [N, L, P], p float32 [N, L, P])` or None.
- Unknown header keys are kept in `.header` and otherwise ignored.

### 3.11 Merge (`python -m pigemma.trace merge <out> <in...>`)

- Inputs MUST agree on `arch`, `n_layer`, `n_expert`, `n_expert_used`, `moe_layers`,
  `probs_top`, `weights_tensor` and `gguf_size_bytes`. Otherwise merge refuses with the name of
  the first differing key.
- Output header: a copy of the first input's header, with `created_at` = now, `flags` = the
  merge argv, `corpus_id = "merged"`, `sources` = the input paths in order, and `chunks` = all
  input chunks concatenated, `chunk_id` renumbered `0..` and each entry gaining `from_source`. If
  `n_ctx`, `n_batch`, `n_ubatch` or `n_threads` differ between inputs, a `notes` entry says so.
  An input with an empty `chunks` list contributes one chunk with `source` = its `corpus_id` and
  counts taken from its phases.
- Records: `token_index` is renumbered, `chunk_id` = (global chunk index) mod 256, and every
  other byte is copied unchanged. `tokens.bin` files are concatenated. `gen.txt` files MAY be
  concatenated, with the chunk numbers in the separators renumbered.

## 4. Trace sidecars, probe summary line, replay chunk file

### 4.1 Naming (D3)

| Sidecar | Name | Writer | Found automatically by readers |
|---|---|---|---|
| token stream | `<trace>.tokens.bin` | probe / `TraceWriter` | yes |
| generated text | `<trace>.gen.txt` | probe | yes |
| run metadata | `<trace>.meta.json` | `pigemma.trace.write_meta` | yes |
| statistics | always an explicit path (`--out`, `--stats`); the project convention is `traces/26b/all.stats.json`, used by `pc/trace_collect.sh` and `pi/06_pin_hotset.sh` | `pigemma.trace stats` | no |

`<trace>` is the trace path with its `.moetrace` extension: `traces/26b/en_tech_moe_streaming.moetrace`
→ `traces/26b/en_tech_moe_streaming.moetrace.tokens.bin`.

### 4.2 `<trace>.tokens.bin`

Raw `i32` little-endian token ids with no header, one per record in stream order:
`size == 4 * N` and `tokens[i] == record[i].token_id`. Any i32 stream in this layout is valid
`--replay` input.

### 4.3 `<trace>.gen.txt`

For every chunk where generation was requested (`--gen > 0`), the probe writes the separator
`"\n--- chunk <i> ---\n"` (i = full chunk index), even when no tokens follow. It then writes the
pieces of the generated tokens, from `common_token_to_piece(ctx, tok, /*special=*/true)`, so that
`pigemma.quality` can strip the thinking channel. Pieces are concatenated as raw bytes, so a
multi-byte UTF-8 character split across tokens is reassembled. Readers decode with
`errors="replace"` and split on the regex `^--- chunk (\d+) ---$` (multiline).

### 4.4 `<trace>.meta.json`

| Key | Type | Meaning |
|---|---|---|
| `trace` | str | path of the trace |
| `host` | object | `pigemma.paths.host_fingerprint()`: hostname, os, kernel, cpu, n_logical, n_physical, ram_total_bytes, ram_avail_bytes, page_size, in_container, cgroup_mem_max, python |
| `gguf_path` | str | as in the header |
| `gguf_size_bytes` | int | |
| `gguf_sha256_prefix` | str | lowercase hex SHA-256 of the **first 16,777,216 bytes** of the GGUF (`pigemma.paths.sha256_prefix`) |
| `corpus` | object[] | `[{"path": str, "bytes": int, "sha256": str (full file)}]`, the input text/prompt files |
| `argv` | str[] | the probe argv |
| `wall_ms` | number | wall time of the probe process |
| `tok_s` | object | `{"prompt": number or null, "gen": number or null}` from the probe summary |
| `callback_overhead_ms` | number | total time spent inside the eval callback |
| `coherent` | bool or null | `pigemma.quality.is_coherent` on `gen.txt`; null when nothing was generated |
| `chunks` | object[] | SHOULD: a copy of the header's `chunks` (makes the file usable as `--replay-chunks`) |
| `probe_summary` | object | MAY: the parsed `PIGEMMA_SUMMARY` object |
| `notes` | str[] | MAY |

### 4.5 Probe summary line and `--mode names` output

At exit (also on exit codes 3 and 4 when possible), `pigemma-probe` writes **one** line to
stderr: `PIGEMMA_SUMMARY ` followed by compact single-line JSON. Wrappers find it with
`grep '^PIGEMMA_SUMMARY '`.

| Key | Type | Source |
|---|---|---|
| `mode` | str | `names`, `moe-trace` or `act-dump` |
| `exit_code` | int | 0 ok, 2 usage, 3 load/compute failure, 4 tensor contract violation |
| `n_prompt`, `n_gen` | int | phase-0 / phase-1 tokens processed |
| `n_records` | int | records written (moe-trace), rows written (act-dump), 0 (names) |
| `n_chunks` | int | |
| `t_load_ms`, `t_p_eval_ms`, `t_eval_ms` | number | `llama_perf_context(ctx)` |
| `tok_s_prompt`, `tok_s_gen` | number or null | `1000 * n_p_eval / t_p_eval_ms`, `1000 * n_eval / t_eval_ms` |
| `callback_ms` | number | total time inside `probe_cb` |
| `trace_out`, `act_out`, `io_csv` | str or null | output paths |

`--mode names` prints every tensor name seen by the callback to **stdout**, one per line, each
name once, sorted by byte value. Names are at most 63 characters (`GGML_MAX_NAME` = 64).

### 4.6 `--replay-chunks FILE`

A JSON object with a top-level `chunks` array in the header chunk schema (section 3.4). A
`<trace>.meta.json` that carries `chunks` qualifies. Boundaries are the cumulative
`n_prompt + n_gen` counted from token 0 of the source stream.

## 5. `stats.json` (trace statistics)

### 5.1 Verbatim contract

Verbatim contract: `data_formats[1]`, name "stats.json (trace statistics)" (copied unchanged from the manifest):

```text
{trace, n_tokens, n_prefill, n_gen, train_frac, per_layer:[{layer, freq:[128 ints], weight_mass:[128 floats], entropy_bits, effective_experts, gini, coverage_at:{'8':..,'16':..,'32':..,'48':..,'64':..,'96':..}, coverage_weight_at:{...}, adjacent_reuse, jaccard_next_mean, working_set:{'16':{median,p95},'64':...,'256':...,'1024':...}, prefill_vs_gen_js_bits}], global:{unique_experts_per_token_mean, items_never_used, lru_hit_curve:[[capacity_items, hit_rate], ... for every capacity 1..L*E] (from the stack-distance histogram), stack_distance_hist:[counts], adjacent_reuse_mean, corpus_overlap:{'<a>|<b>': jaccard_top32}}, cross_layer:{precision_at_8, precision_at_16, baseline_static_top16, baseline_same_ids, mutual_info_bits_mean}, probs:{margin_p8_p9:{p10,p50,p90}, eighth_expert_mass_mean} (only if probs_top>0)}.
```

### 5.2 Definitions (normative for `pigemma.trace.stats`)

Notation: N records, L MoE layer slots (`moe_layers`), K experts per token and layer, E experts
per layer. `S[t][j]` is the set of K ids of record t at slot j. An **item** is a pair
`(il, e)`. The list `freq[128 ints]` in the contract means E entries; E = 128 for the 26B.

- **Split.** `n_train = floor(train_frac * N)`. The train split is records `[0, n_train)` and the
  test split is `[n_train, N)`, both in stream order. Statistics use all N records unless they
  say otherwise.
- **Top-level counts.** `trace` (path), `n_tokens = N`, `n_prefill` = count(phase 0),
  `n_gen` = count(phase 1), `train_frac`.

`per_layer[j]` has one entry per slot, in `moe_layers` order:

| Key | Definition |
|---|---|
| `layer` | `il = moe_layers[j]` |
| `freq` | E ints: `freq[e]` = number of records with `e` in `S[t][j]`; the sum is N*K |
| `weight_mass` | E floats: the sum over records of the weight of `e` at this layer (0 when not selected); the sum is about N |
| `entropy_bits` | `-sum p_e log2 p_e` with `p_e = freq[e] / (N*K)` and `0*log 0 = 0` |
| `effective_experts` | `2 ** entropy_bits` |
| `gini` | Gini coefficient of `freq` over all E experts: `sum_i sum_k abs(x_i - x_k) / (2 * E * sum x)`; 0 = uniform |
| `coverage_at["n"]` | for n in {8, 16, 32, 48, 64, 96}: (sum of the n largest `freq`) / (N*K); 1.0 when n >= E; non-decreasing in n |
| `coverage_weight_at["n"]` | the same with `weight_mass`, ranked by `weight_mass` |
| `adjacent_reuse` | mean over t in `[0, N-2]` of `len(S[t][j] & S[t+1][j]) / K`. Consecutive records in stream order, **including across chunk boundaries**, because that is the order a cache sees them |
| `jaccard_next_mean` | the mean of `len(S[t][j] & S[t+1][j]) / len(S[t][j] | S[t+1][j])` (Jaccard index) over the same pairs |
| `working_set["W"]` | for W in {16, 64, 256, 1024}: cut the stream into non-overlapping windows of W records (a trailing partial window is dropped) and count the distinct experts of this layer in each window. Value `{median, p95}` (NumPy linear percentiles); `null` if N < W |
| `prefill_vs_gen_js_bits` | Jensen-Shannon divergence (base 2, range [0, 1]) between the `freq` distributions of phase-0 and phase-1 records; `null` if either phase is empty |

`global`:

| Key | Definition |
|---|---|
| `unique_experts_per_token_mean` | mean over records of the number of distinct ids across all layers of the token (ids compared regardless of layer, so the value lies in `[K, min(E, L*K)]`). It measures id reuse across layers. It is not a byte count: expert e of layer 3 and expert e of layer 4 are different weights |
| `items_never_used` | number of items `(il, e)` with `freq == 0`, out of L*E |
| `stack_distance_hist` | int list of length `L*E + 1` over the **access sequence** `for t in 0..N-1: for j in 0..L-1: for k in 0..K-1: item (moe_layers[j], ids[t][j][k])` (N*L*K accesses, the same sequence `pigemma.cache_sim` replays). `hist[0]` = first-touch (cold) accesses; `hist[d]` for d >= 1 = accesses with LRU stack distance d, where d = 1 + the number of distinct other items accessed since the previous access to the same item. The sum is N*L*K |
| `lru_hit_curve` | `[[C, hit(C)] for C in 1..L*E]` with `hit(C) = sum(hist[1..C]) / (N*L*K)`: the exact **overall** hit rate of an LRU cache holding C items, counting cold misses and with no warm-up exclusion. `pigemma.cache_sim` with policy `lru`, `capacity_items = C` and `--warmup-tokens 0` MUST report the same `overall.hit_rate` within 1e-9. Its steady-state numbers exclude warm-up and do differ |
| `adjacent_reuse_mean` | the mean of `per_layer[j].adjacent_reuse` |
| `corpus_overlap` | for every unordered pair of distinct chunk `source` values a < b (Python string order), the key `"a|b"` holds the mean over layers of `Jaccard(top_a, top_b)`. `top_s` at a layer is the `min(32, E)` experts with the highest `freq` among records of chunks whose source is s (ties to the lower id). `{}` when there are fewer than two sources |

`cross_layer` requires L >= 2 and non-empty train and test splits; otherwise every value is `null`.

| Key | Definition |
|---|---|
| (predictor) | for each consecutive slot pair (j, j+1), count `C_j[e][e']` = train records with `e` in `S[t][j]` and `e'` in `S[t][j+1]`, and set `P(e' given e) = C_j[e][e'] / freq_train_j[e]` (0 if that frequency is 0). For a test record: `score(e') = sum over e in S[t][j] of P(e' given e)`; the prediction is the top N by score (ties to the lower id) |
| `precision_at_8`, `precision_at_16` | mean over test records and slot pairs of `len(pred_N & S[t][j+1]) / K`: the fraction of the true next-layer set that an N-expert prefetch covers. At N = K = 8 this equals classic precision; the wasted-prefetch fraction is `1 - hits/N` |
| `baseline_static_top16` | the same metric, predicting the 16 most frequent train experts of slot j+1 |
| `baseline_same_ids` | the same metric, predicting `S[t][j]` (same ids as the previous layer, N = K) |
| `mutual_info_bits_mean` | mean over slot pairs of the plug-in mutual information `I(top1_j ; top1_{j+1})` in bits over all records, where `top1 = ids[t][j][0]` |

`probs` is present only when `probs_top > 0`:

| Key | Definition |
|---|---|
| `margin_p8_p9` | `{p10, p50, p90}` over (record, layer) of `prob[K-1] - prob[K]` (K-th minus (K+1)-th router probability); `null` unless `P >= K+1` |
| `eighth_expert_mass_mean` | mean over (record, layer) of `weights[K-1]`, the smallest normalised routing weight of the selected K |

The CSV export (`--csv-dir`) is a convenience for plotting and is not an interchange contract.

## 6. Tensor-name contract (what the probe hooks)

llama.cpp names a graph tensor `"<base>-<il>"` for per-layer tensors (`ggml_format_name(cur,
"%s-%d", name, il)`, `src/llama-context.cpp:2639`) and `"<base>"` otherwise. The callback
receives the final name. A tensor renamed twice, such as the dense down output, which `build_ffn`
creates and `gemma4.cpp` renames to `ffn_out`, is seen under its last name. Notation: `ne` is
listed fastest dimension first (`ne[0]` first); T is the number of tokens in the **ubatch** being
computed; E = `n_expert`; K = `n_expert_used`.

| Name (`<base>-<il>`) | Defined at (a7b94df) | Type | `ne` | Strides / layout | Meaning | Probe mode |
|---|---|---|---|---|---|---|
| `ffn_moe_topk` | `src/llama-graph.cpp:2127-2130`; a view made by `ggml_argsort_top_k` (`ggml/src/ggml.c:5450`) | I32 | `[K, T]` | **strided view**: `nb[0] = 4`, `nb[1] = E*4` (= 512 B for E = 128). Element (k, t) at byte `k*nb[0] + t*nb[1]`. Not contiguous | ids of the K selected experts of token t, descending probability | moe-trace |
| `ffn_moe_argsort` | `:2128` | I32 | `[E, T]` | contiguous | full descending order; the view's source | not hooked |
| `ffn_moe_weights_norm` | `:2162-2163` | F32 | `[K, T]` at a7b94df (D1); `[1, K, T]` per the manifest | contiguous: K floats per token, token stride `4K`. Element (k, t) at `k*4 + t*4K` in both shapes | routing weights renormalised over the K selected experts; they sum to 1 (gemma4 passes `norm_w = true`, `gemma4.cpp:331`) | moe-trace |
| `ffn_moe_weights` | `:2141-2142` | F32 | `[1, K, T]` | contiguous | weights **before** renormalisation; the sum is below 1 | not recorded |
| `ffn_moe_probs` | `:2078` | F32 | `[E, T]` | contiguous | softmax over all E router logits | moe-trace with `--probs-top > 0` |
| `ffn_moe_logits` | `src/models/gemma4.cpp:322` | F32 | `[E, T]` | contiguous | router logits | not hooked |
| `ffn_norm` | `gemma4.cpp:350`, **dense layers only** (31B) | F32 | `[n_embd, T]` | contiguous | input of `ffn_gate` and `ffn_up` (after the pre-FFN RMSNorm) | act-dump |
| `ffn_geglu` | `src/llama-graph.cpp:1879` (`build_ffn`, GELU + PAR) | F32 | `[n_ff, T]` | contiguous | `gelu(gate) * up`, the input of `ffn_down` | act-dump |
| `ffn_out` | `gemma4.cpp:358`, dense layers only | F32 | `[n_embd, T]` | contiguous | output of `ffn_down`, **before** `ffn_post_norm` | act-dump (sanity check) |
| `ffn_norm_1`, `ffn_norm_2` | `gemma4.cpp:298`, `:315`, MoE layers only (26B) | F32 | `[n_embd, T]` | contiguous | shared-FFN input / MoE input | `--mode names` check only |

Rules the probe MUST follow:

1. **No warm-up.** Set `params.warmup = false`. During warm-up `n_expert_used` becomes
   `n_expert` (`src/llama-graph.cpp:1473`), so every token "selects" all 128 experts and the
   trace would be polluted.
2. **Honour strides.** Read every hooked tensor through `nb[]`, never assuming contiguity. If
   `ggml_backend_buffer_is_host(t->buffer)` is false, copy the covering byte span (for
   `ffn_moe_topk`: `(T-1)*nb[1] + K*nb[0]` bytes) with `ggml_backend_tensor_get` first.
3. **Check on first sight.** On the first sight of each name, assert the type and shape against
   the table: `ne[0] == K` (topk); `ne[0]*ne[1] == K` per token and `ne[0] in {K, 1}` (weights_norm,
   D1); `ne[0] == E` (probs); `ne[0] == n_embd` or `n_ff` (activations); `nb[0] == 4`. On a
   mismatch, exit 4 with a message naming the tensor.
4. **Ubatch accumulation.** The callback runs once per ubatch with that ubatch's T columns, and
   ubatches arrive in token order. After `llama_process` returns, every wanted layer MUST have
   accumulated exactly `chunk_len` columns; otherwise exit 4 (`layer %d got %d columns, expected
   %d`). D10 explains why the last layer is not pruned.
5. **Gemma 4 shapes.** 26B-A4B: n_embd 2816, E 128, K 8, the shared FFN's `ffn_geglu` has ne0
   2112, and there is no `ffn_norm`/`ffn_out` (all 30 layers are MoE). 31B: n_embd 5376, n_ff
   21504, no MoE tensors (moe-trace refuses with exit 4 because `expert_count` is 0).

## 7. Activation dump directory (H2)

### 7.1 Verbatim contract

Verbatim contract: `data_formats[4]`, name "acts dump directory (H2 activations)" (copied unchanged from the manifest):

```text
Directory traces/h2/acts/<model_tag>/ with meta.json {model, gguf_path, layers:[...], kinds:{'ffn_norm':5376,'ffn_geglu':21504,'ffn_out':5376}, n_tokens, phase_file:'phase.u8', tokens_file:'tokens.bin', chunks:[{chunk_id, source, n_prompt, n_gen}], llama_build, n_ubatch, flags}; `L<il>.<kind>.npy` = NumPy .npy v1.0, fixed 128-byte header, dtype '<f2', shape (n_tokens, ne0), C order, one row per token in stream order (header shape rewritten at close); `phase.u8` (one byte per token: 0 prefill, 1 generated); `tokens.bin` (i32 LE). For the 31B with 512 tokens: 5.5 MB + 22 MB + 5.5 MB per layer.
```

### 7.2 Directory layout

```text
traces/h2/acts/<model_tag>/        e.g. traces/h2/acts/31b/
  meta.json                        see 7.1; n_tokens = actual T after close
  L<il>.<kind>.npy                 il in decimal without padding: L5.ffn_norm.npy, L20.ffn_geglu.npy
  phase.u8                         T bytes, 0 = prefill, 1 = generated, stream order
  tokens.bin                       T x i32 LE, stream order
```

Every `.npy` file in a directory has the same T, and row t of every file belongs to the same token
t. `kinds` maps a kind to its `ne0` (31B: `ffn_norm` 5376, `ffn_geglu` 21504, `ffn_out` 5376).
Optional `meta.json` key: `f16_nonfinite` (`{"L5.ffn_geglu": count}`), the number of values that
overflowed to ±inf in the f16 conversion. Readers (`pigemma.h2.oracle`) MUST check finiteness and
report or drop non-finite rows.

### 7.3 The fixed 128-byte `.npy` header

The file is NumPy format version 1.0 with a header of exactly 128 bytes:

| Bytes | Content |
|---|---|
| 0-5 | `93 4E 55 4D 50 59` (`\x93NUMPY`) |
| 6-7 | `01 00` (version 1.0) |
| 8-9 | `76 00`: HEADER_LEN = 118 (u16 LE) |
| 10-126 | ASCII dict `{'descr': '<f2', 'fortran_order': False, 'shape': (T, NE0), }` followed by spaces up to byte 126 |
| 127 | `0A` (newline) |
| 128... | data: T rows x NE0 `<f2` values, C order (row t = token t) |

Example, verified with NumPy 2.3.3, which writes exactly these 128 bytes for any 2-D `<f2` array
(its own 64-byte alignment rule also produces 128):

```text
b"\x93NUMPY\x01\x00v\x00{'descr': '<f2', 'fortran_order': False, 'shape': (512, 21504), }" + b' ' * 52 + b'\n'
```

The dict text is at most 97 bytes even for 2^64-1 values, so the header never needs to grow.
Writers MUST use the canonical single-quoted form above, byte for byte identical to `np.save`.
Readers that parse it themselves MUST use Python-literal rules (`ast.literal_eval`, as NumPy
does), which also accept double quotes.

**Patching rule.**

1. **Open.** Write the header with `shape = (T_MAX, NE0)`, where T_MAX is the `--tokens` value.
2. **Append.** Append rows in stream order: `NE0` values converted with
   `ggml_fp32_to_fp16_row`, 2*NE0 bytes per row.
3. **Close.** Rewrite bytes 0..127 with the real T, re-padded so the header stays exactly 128
   bytes and byte 127 stays `\n`. Then `ftruncate` the file to `128 + T*NE0*2` (this also covers
   preallocation). Write `phase.u8`, `tokens.bin` and `meta.json` with the same T.

**Reader recovery.** If the file size differs from `128 + T*NE0*2` (the writer died before
close), use `T_file = (size - 128) // (2*NE0)` rows, warn "unpatched npy", and memmap with
`np.memmap(path, dtype='<f2', mode='r', offset=128, shape=(min(T, T_file), NE0))`. A partial row
is ignored.

Sizes for the 31B with T = 512: `ffn_norm` and `ffn_out` 512 x 5376 x 2 = 5,505,024 B each,
`ffn_geglu` 512 x 21504 x 2 = 22,020,096 B, so 33,030,144 B per layer plus 3 x 128 B of headers.

## 8. `map.json` (GGUF byte-range map)

### 8.1 Verbatim contract

Verbatim contract: `data_formats[2]`, name "map.json (GGUF byte-range map)" (copied unchanged from the manifest):

```text
{gguf, file_size, data_offset, alignment, page_size, arch, n_layer, n_expert, n_expert_used, n_embd, n_ff, n_ff_exp, head_count_kv:[...], key_length, key_length_swa, sliding_window, sliding_pattern:[bool], tensors:[{name, type, shape:[ne0,ne1,ne2?], n_bytes, offset (absolute file offset), kind:'token_embd'|'output'|'attn'|'ffn_dense'|'norm'|'router'|'expert_scale'|'expert'|'other', layer:int|null}], expert_layers:[{layer, gate_up:{offset, n_bytes, nb_expert}, down:{offset, n_bytes, nb_expert}}] (slab e = offset + e*nb_expert), summary:{nonexpert_bytes, expert_bytes, expert_unit_bytes, n_moe_layers, expert_bytes_per_token_worst, nonexpert_bytes_per_token, kv_bytes_per_token_f16, kv_bytes_f16_at_4096, kv_bytes_q8_at_4096, bytes_per_kind:{kind: bytes}}}. For the 26B: data_offset 15823936, expert_unit_bytes 3345408, expert_bytes 12846366720, nonexpert_bytes 1577172088, expert_bytes_per_token_worst 802897920.
```

### 8.2 Field rules

- `offset` is the **absolute** file offset of a tensor's data, which is gguf-py's
  `ReaderTensor.data_offset` (verified absolute; see FACTS (6)). `data_offset` at top level is the
  start of the tensor-data section. `alignment` is `general.alignment`, or 32 when absent.
- `type` is the gguf-py `GGMLQuantizationType` name: `"Q4_0"`, `"Q6_K"`, `"F32"`, `"F16"`, ...
- `shape` uses gguf-py order (`ReaderTensor.shape`, `ne[0]` fastest) and has 1 to 3 entries. A
  weight used as `y = W @ x` has `ne = [n_in, n_out]`.
- `layer` is the integer captured by `^blk\.(\d+)\.`, else `null`.
- `kind` classification. The first three rules are normative because byte ranges depend on them;
  the remaining kinds are assigned by `pigemma.gguf_map` and constrained by the verified totals in
  8.3:

  ```text
  expert        ^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$
  expert_scale  ^blk\.(\d+)\.ffn_.*_exps\.scale$
  router        ^blk\.(\d+)\.ffn_gate_inp\.          (weight and scale)
  token_embd    ^token_embd\.
  output        ^output\.weight$                       (absent in the tied Gemma 4 GGUFs)
  attn          blk.N.attn_{q,k,v,output}.weight (+ their q/k norms if gguf_map counts them there)
  ffn_dense     blk.N.ffn_{gate,up,down}.weight
  norm, other   norm weights; everything else
  ```

- **Expert slabs.** For an expert tensor, `n_expert = shape[-1]` and
  `nb_expert = n_bytes / n_expert`, which MUST divide exactly. Expert e occupies
  `[offset + e*nb_expert, offset + (e+1)*nb_expert)`. `expert_layers[i]` holds
  `{layer, gate_up: {offset, n_bytes, nb_expert}, down: {offset, n_bytes, nb_expert}}` (D7).
  A GGUF with separate `ffn_gate_exps` / `ffn_up_exps` MAY use keys `gate` and `up` instead of
  `gate_up`; `expert_unit_bytes` is always the sum over all of a layer's expert tensors for one
  expert.
- **Page alignment.** `aligned(offset, length, page)` = `[floor(offset/page)*page,
  ceil((offset+length)/page)*page)`. `page_size` in the map is the page size the planner
  assumed (default 16384, the Pi kernel). Consumers that pin or read pages re-align with the
  runtime page size.
- **Summary formulas.**
  - `expert_bytes_per_token_worst = n_moe_layers * n_expert_used * expert_unit_bytes`
  - `nonexpert_bytes_per_token = attn + ffn_dense + norm + router + expert_scale + (output if a
    separate output tensor exists, else token_embd, because the tied head is read every token) +
    one embedding row`
  - `kv_bytes_per_token_f16 = sum over layers of n_head_kv[il] * (key_length_swa if sliding else
    key_length) * 2 bytes * 2 (K and V)`, an upper bound
  - `kv_bytes_f16_at_4096 = sum over layers of min(4096, sliding_window if sliding else 4096) *
    per-layer bytes`
  - `kv_bytes_q8_at_4096 = kv_bytes_f16_at_4096 * (1/2 + 1/32)` (q8_0 = 34 bytes per 32 values)

### 8.3 Verified constants (FACTS, design panel 2026-10-05)

| | 26B-A4B Q4_0 | 31B Q4_0 |
|---|---|---|
| file size (stat) | 14,439,363,584 B | 17,651,001,568 B |
| `data_offset` | 15,823,936 | 15,833,440 |
| tensors | 658 | 833 |
| `expert_unit_bytes` | 3,345,408 (gate_up 2,230,272 + down 1,115,136) | n/a (dense) |
| `expert_bytes` | 12,846,366,720 | 0 |
| `nonexpert_bytes` | 1,577,172,088 (attn 624,885,760; tied Q6_K head 605,563,904; dense FFN 301,086,720; norms/router/scales 45,635,704) | FFN 11,705,057,280 (65,028,096 per gate/up/down matrix); attention 4,770,160,640; tied head 1,156,076,544 |
| `expert_bytes_per_token_worst` | 802,897,920 (= 30 x 8 x 3,345,408) | n/a |

## 9. `hotset.json` and `hotset.ranges.txt`

### 9.1 Verbatim contract

Verbatim contract: `data_formats[3]`, name "hotset.json + hotset.ranges.txt" (copied unchanged from the manifest):

```text
hotset.json: {gguf, map_sha256, stats (path or null), page_size, budget_bytes, total_bytes, nonexpert_included:bool, n_experts, expert_coverage_fraction, rank:'count'|'weight', ranges:[{offset, length, kind, layer, expert, hits}] sorted by offset, page-aligned, adjacent ranges merged}. hotset.ranges.txt: line 1 '# pigemma-ranges v1 gguf=<basename> size=<bytes> page=<page> budget=<bytes>'; then one range per line 'offset length' in decimal, optionally followed by '# kind=<kind> layer=<l> expert=<e>'; '#' lines are comments. This is what tools/pinner reads (no JSON parser in C).
```

### 9.2 `hotset.json` rules

- `ranges[]` is sorted by `offset`, page-aligned to `page_size`, non-overlapping, with adjacent
  ranges merged. A merged range has `kind` = the common kind or `"mixed"`, `layer` = the common
  layer or `null`, `expert` = the single expert id or `null`, and `hits` = the sum of the parts'
  `freq` (or weight mass with `rank: "weight"`). It MAY carry `n_parts`.
- `map_sha256` is the hex SHA-256 of the `map.json` file used. `stats` is the path of the
  `stats.json` used, or `null` (non-expert-only pinning).
- `total_bytes` is the sum of range lengths after alignment and merging. It is at most
  `budget_bytes` plus one page per range.
- `expert_coverage_fraction` = (sum of `freq` of the selected experts) / (sum of all `freq`).

### 9.3 `hotset.ranges.txt` grammar (normative for `tools/pinner`)

```text
# pigemma-ranges v1 gguf=gemma-4-26B_q4_0-it.gguf size=14439363584 page=16384 budget=5200000000
# non-expert tensors first, then experts by descending frequency (illustrative values)
15810560 1577189376 # kind=mixed layer=- expert=-
1592999936 3358720 # kind=expert layer=0 expert=102
```

- The file is ASCII with LF line endings. The pinner also strips a trailing `\r`.
- Line 1 MUST start with `# pigemma-ranges v1 ` followed by the space-separated pairs
  `gguf=<basename> size=<bytes> page=<page> budget=<bytes>`. Values contain no spaces.
- A line whose first non-blank character is `#` is a comment. Blank lines are ignored.
- A range line is `<offset> <length>`: two unsigned decimal integers (no sign, no `0x`, no digit
  separators) separated by spaces or tabs, optionally followed by whitespace and a comment of the
  form `# kind=<kind> layer=<int or -> expert=<int or ->`. The pinner uses `kind` only to report
  per-kind totals.
- `length > 0` and `offset + length <= size`. Ranges are sorted and non-overlapping. The pinner
  MUST NOT rely on that, but MAY warn.
- Any other line is malformed: the pinner exits 1 and names the line number.
- The pinner re-aligns every range to `sysconf(_SC_PAGESIZE)` (16384 on the Pi kernel, 4096 in
  Docker and under qemu).

## 10. Results JSON envelope (one file per run)

### 10.1 Verbatim contract

Verbatim contract: `data_formats[5]`, name "results JSON envelope (one file per run)" (copied unchanged from the manifest):

```text
{run_id, experiment:'E0'..'E8'|'adhoc', created_at, kind:'measured'|'projected'|'assumed', host:{hostname, os, kernel, cpu, n_logical, n_physical, ram_total_bytes, page_size, in_container, cgroup_mem_max}, llama_build, llama_commit, model:{id, file, quant, bytes, sha_prefix}, variant, flags:[...], threads, n_ctx, assumptions:{name:{value, unit, source}} (required when kind != measured), inputs:[paths], pp:{tok_s, sd, n}, tg:{tok_s, sd, n}, load_s, rss_peak_mb, private_peak_mb, mapped_mb, buffer_split:{CPU_REPACK_mib, CPU_Mapped_mib, CPU_mib}, read_bytes_per_token, majflt_per_token, io_rbytes_per_token, accept_rate, draft_n, draft_n_accepted, spec_type_effective, spec_active, identical_to_plain, kld:{mean, max, p999, same_top_p, ppl_q, ppl_base}, ppl, qa_score, qa_total, coherent, output_sample, notes:[...], extra:{}}.
```

### 10.2 Field rules

- **Required** (`pigemma.results.write_result` raises `ValueError` listing what is missing):
  `run_id` (uuid4, canonical 36-character string), `experiment` (`"E0"`..`"E8"` or `"adhoc"`; the
  manifest's long experiment ids such as `E0-desktop-baseline-and-repack-gate` map to their
  prefix), `created_at`, `kind`, `host` (`pigemma.paths.host_fingerprint()`; extra keys allowed),
  `llama_build`, `llama_commit`, `model {id, file, quant, bytes, sha_prefix}`, `variant`,
  `flags`, `threads`, `n_ctx`.
- **When `kind != "measured"`,** `assumptions` (non-empty, `{name: {value, unit, source}}`) and
  `inputs` (paths of the results or traces used) are also required.
- `model.id` is the catalog model id. `model.file` is the GGUF basename. `model.bytes` is the
  file size. `model.sha_prefix` is the hex SHA-256 of the first 16,777,216 bytes.
- `pp` and `tg` are `{tok_s: mean, sd: sample standard deviation, n: repetitions}`.
- `buffer_split` comes from the llama.cpp load log lines `CPU_REPACK model buffer size`,
  `CPU_Mapped model buffer size` and `CPU model buffer size`, in MiB.
- `kld` is `{mean, max, p999, same_top_p, ppl_q, ppl_base}` from `llama-perplexity
  --kl-divergence`. `same_top_p` is a **fraction in [0, 1]** (llama.cpp prints a percentage;
  divide by 100).
- Speculative decoding: `draft_n`, `draft_n_accepted` (llama-server `timings`),
  `accept_rate = draft_n_accepted / draft_n` (`null` when `draft_n == 0`),
  `spec_active` (`draft_n > 0` on every request), `spec_type_effective`
  (`"none" | "draft-mtp" | "draft-dflash" | "inactive"`), `identical_to_plain` (greedy output
  identical to the plain run).
- Per-token I/O: `read_bytes_per_token` (from `io.csv` `read_bytes_delta`), `majflt_per_token`,
  `io_rbytes_per_token` (cgroup `io.stat` rbytes).
- `output_sample` has at most 2000 characters. `notes` is a list of strings.
- Metrics that were not run are omitted or `null`; the summariser renders them as "not run".
  Keys not listed in 10.1 go under `extra` (D8).

## 11. `sim.json` and `sim.csv` (cache simulation)

### 11.1 Verbatim contract

Verbatim contract: `data_formats[6]`, name "sim.json (cache simulation output)" (copied unchanged from the manifest):

```text
{trace, map, catalog, token_range, warmup_tokens, read_granularity, train_split, runs:[{policy, budget_bytes, capacity_items, profile:null|name, steady:{hit_rate, weighted_hit_rate, miss_bytes_per_token:{mean,p50,p95,max}, n_miss_per_token, prefetch_bytes_per_token, wasted_prefetch_fraction}, overall:{...}, cold_bytes, per_layer_hit_rate:[30], projection:{resident_bytes_per_token, t_ram_s, t_io_s, tg_serial, tg_overlap, floor_all_in_ram_tok_s, verdict:'FITS_RAM'|'STREAM_GOOD'|'STREAM_WEAK'|'NOT_VIABLE', kind:'projected', assumptions:{mem_bw_gbps:{value,source}, llama_efficiency:{value,source}, rand_read_3mb_mbps:{value,source}, access_ms:{value,source}, os_reserve_bytes:{...}, kv_bytes:{...}}}|null}], dense_31b_line:{bytes_per_token, tok_s_at_mem_bw, s_per_token_at_storage_bw}}; sim.csv columns: policy,budget_gb,profile,hit_rate_ss,weighted_hit_ss,miss_mb_tok_mean,miss_mb_tok_p95,tg_serial,tg_overlap,verdict.
```

### 11.2 Field rules

- `token_range` is `[A, B]` (half-open) of the trace records simulated. `warmup_tokens` are
  excluded from `steady` but included in `overall`. `read_granularity` is in bytes (default
  131072). `train_split` is the fraction used by `static` and `lru_prefetch`.
- `runs[].budget_bytes` is the expert-cache size. `capacity_items` is the number of `(layer,
  expert)` items the cache holds, as computed by `cache_sim`. It is the quantity to compare with
  `stats.json` `lru_hit_curve` (section 5.2). `profile` is a catalog device name, or `null` for a
  raw `--budgets-gb` run, in which case `projection` is `null`.
- `verdict` is one of `FITS_RAM`, `STREAM_GOOD`, `STREAM_WEAK`, `NOT_VIABLE` or `INFEASIBLE` (D9).
  Every `projection` has `kind: "projected"`, and each assumption has `{value, source}`.
- `per_layer_hit_rate` has L entries in `moe_layers` order (30 for the 26B).
- `sim.csv` has the header line exactly
  `policy,budget_gb,profile,hit_rate_ss,weighted_hit_ss,miss_mb_tok_mean,miss_mb_tok_p95,tg_serial,tg_overlap,verdict`.
  `budget_gb` is in 10^9 B and `miss_mb_*` in 10^6 B. Empty cells stand for `null` (raw budgets
  have an empty `profile`, `tg_*` and `verdict`). Floats are written with up to 6 significant
  digits.

## 12. `validation.json` and `io.csv` (E5/E6 page-cache proxy)

### 12.1 Verbatim contract

Verbatim contract: `data_formats[7]`, name "validation.json (E5/E6 proxy) + io.csv" (copied unchanged from the manifest):

```text
io.csv (probe --io-csv): header 'token_index,phase,wall_ms,read_bytes_delta,majflt_delta,rss_anon_kb,rss_file_kb', one row per processed token. validation.json: {gguf, volume:'pigemma-models'|'bind-mount', token_range, warmup_tokens, points:[{mem_cap_bytes, pinned:bool, pin_budget_bytes, pin_locked_bytes, anon_rss_bytes, file_cache_bytes, tier_rand_read_3mb_mbps, measured_bytes_per_token, measured_majflt_per_token, measured_tok_s, sim_budget_bytes, sim_lru_bytes_per_token, sim_belady_bytes_per_token, ratio_sim_lru_over_measured, predicted_tok_s, oom:bool, notes:[...]}], kind:'measured' with a 'projected' sub-block per point for predicted_tok_s and its assumptions}.
```

### 12.2 `io.csv` columns (`pigemma-probe --io-csv F`, Linux only)

The first line is exactly
`token_index,phase,wall_ms,read_bytes_delta,majflt_delta,rss_anon_kb,rss_file_kb`. After it comes
one row per processed ubatch (with `-ub 1`, one row per token), with LF line endings. On
non-Linux hosts `--io-csv` is a usage error (exit 2).

| Column | Type | Meaning |
|---|---|---|
| `token_index` | int | `token_index` (in this run's trace) of the last token the row covers |
| `phase` | int | 0 or 1 (always 0 in replay) |
| `wall_ms` | float (3 decimals) | wall-clock ms since the previous row. The first row counts from the start of the first `llama_process` call after model load, so load I/O is excluded |
| `read_bytes_delta` | int | change in `read_bytes` of `/proc/self/io` since the previous row: bytes the block layer read for this process (page-cache misses including readahead). Reads through the Docker 9p bind mount are **not** counted, so use the `pigemma-models` volume |
| `majflt_delta` | int | change in field 12 (`majflt`) of `/proc/self/stat` |
| `rss_anon_kb` | int | `RssAnon` from `/proc/self/status` at row time (KiB) |
| `rss_file_kb` | int | `RssFile` from `/proc/self/status` at row time (KiB) |

### 12.3 `validation.json` rules

- `measured_bytes_per_token` is the mean `read_bytes_delta` over rows after `warmup_tokens`.
  `measured_tok_s = 1000 / mean(wall_ms)` over the same rows.
- `kind` at the top level is `"measured"`. Each point has a `projected` sub-object
  `{"kind": "projected", "predicted_tok_s": number, "assumptions": {name: {value, unit,
  source}}}`. The point-level `predicted_tok_s` MUST equal `projected.predicted_tok_s`.
- `volume: "bind-mount"` points carry the note `bind-mount proxy: lower-bound bytes, timings not
  representative`. Their bytes per token are estimated as `majflt * page_size` (a lower bound).
- `oom` is true when the cgroup `memory.events` `oom_kill` counter increased during the run.

## 13. H2 `oracle.json` and the decomposition `.npz` files

### 13.1 Verbatim contract (`results/h2/oracle.json`)

Verbatim contract: `data_formats[8]`, name "h2 oracle.json" (copied unchanged from the manifest):

```text
{model, acts_dir, decomp_dir, sampling:{layers, mats, bases, group_cols, res_bits, n_tokens, eps:[0.01,0.03,0.10]}, sanity:[{layer, rel_err_median, rel_err_p95, pass:bool}], per_case:[{layer, mat, base, phase:'prefill'|'gen'|'all', n_tokens, bytes_q4_0, base_bytes, residual_bytes_total, err_base_mean, oracle:{'0.01':{frac_bytes_mean, frac_bytes_p90, k_groups_mean}, '0.03':{...}, '0.10':{...}}, curve:[[bytes_frac, rel_err_mean, rel_err_p95], ...], bound:{'0.01':{certified_frac_bytes_mean, realised_err_mean, soundness_violations, certification_efficiency}, '0.03':{...}, '0.10':{...}}, tightness:{p10,p50,p90}, spearman_bound_vs_actual, iso_byte:[{bytes_per_elem, uniform_err, base_plus_oracle_err, base_plus_certified_err}], seconds}], summary:{frac_0_03_median, cert_eff_median, soundness_violations_total, implied_bytes_per_token:{attention:4770160640, head:1156076544, ffn_base, ffn_residual_at_0_03, total, kind:'projected', note:'extrapolated from sampled layers'}, s_per_token_nvme_850mbps, tok_s_in_ram_10gbps, verdict:'PROMISING'|'WEAK'|'NOT_SUPPORTED'|'NO_ADVANTAGE', rule:'<threshold text>'}}.
```

### 13.2 Field rules

- The keys of the `eps` maps (`oracle`, `bound`) are the eps values formatted with two decimals:
  `"0.01"`, `"0.03"`, `"0.10"`.
- Errors and fractions are measured on real 31B activations. `summary.implied_bytes_per_token`
  is `kind: "projected"` (extrapolated from the sampled layers). Its fixed parts are attention
  4,770,160,640 B and head 1,156,076,544 B. The Q4_0 comparison is 17,635,166,448 B per token.
- `sanity[].pass` is `rel_err_median < 0.03` for `W_down_q4 @ ffn_geglu` vs captured `ffn_out`.
  The oracle aborts when it exceeds 0.1.
- `soundness_violations` MUST be 0. A non-zero value means the certification rule or its
  implementation is wrong, and the verdict is void.

### 13.3 Decomposition `.npz` (from the `pigemma/h2/decompose.py` spec)

`data_formats` has no entry for this file, so this section fixes the interface between
`decompose.py` and `oracle.py`:

- Path: `<decomp_dir>/L<il>.<mat>.<base>.npz`, with `mat` in {`gate`, `up`, `down`} and `base` in
  {`q2`, `q3`, `lowrank256`, `lowrank512`, `q2+lowrank128`}. Example:
  `traces/h2/decomp/L20.down.q2.npz`.
- Written with `np.savez` (uncompressed). Arrays:

  | Array | dtype / shape | Meaning |
  |---|---|---|
  | `W_base_hat` | f16 `[n_out, n_in]` | the base approximation |
  | `R_hat` | f16 `[G, n_out, g]` | residual of input-column group i (`R[:, cols[i]]`), quantised at `res_bits` (exact for 16); a shorter last group is zero-padded on the right |
  | `cols` | int64 `[G, 2]` | `[start, stop)` input-column range of each group |
  | `norm2` | float64 `[G]` | largest singular value of the group's residual |
  | `normF` | float64 `[G]` | Frobenius norm of the group's residual |
  | `bytes_g` | int64 `[G]` | stored bytes of each residual group at `res_bits` |
  | `base_bytes` | int64 scalar | stored bytes of the base |
  | `q4_0_bytes` | int64 scalar | bytes of the original Q4_0 matrix (65,028,096 for every 31B FFN matrix) |
  | `layer`, `group_cols`, `res_bits`, `n_out`, `n_in` | int64 scalars | |
  | `mat`, `base` | 0-d unicode arrays | |

- Orientation is `y = W @ x` with W `[n_out, n_in]`. GGUF stores `ne = [n_in, n_out]`. For the
  31B, `gate` and `up` are `[21504, 5376]` (input `ffn_norm`, 5376) and `down` is
  `[5376, 21504]` (input `ffn_geglu`, 21504). With `group_cols = 128` the groups divide exactly:
  42 groups for 5376 columns and 168 groups for 21504.
- Disk use: one 31B matrix holds 115,605,504 values, so `W_base_hat` and `R_hat` are about 231 MB
  each at f16, roughly 462 MB per file. The full plan of 4 layers x 3 matrices x 2 bases is about
  11 GB. Check free disk space before a full run.

## 14. `facts.json` (Pi feasibility input)

### 14.1 Verbatim contract

Verbatim contract: `data_formats[9]`, name "facts.json (Pi feasibility input)" (copied unchanged from the manifest):

```text
{hostname, kernel, page_size, cpu_model, cpu_flags:[...], nproc, ram_total_kb, ram_available_kb, swap_total_kb, loadavg1, model_dir, storage:{dev, type:'nvme'|'usb-ssd'|'microsd'|'unknown', rotational:bool, pcie_link_speed:'8.0 GT/s PCIe'|'5.0 GT/s PCIe'|null, pcie_link_width, read_ahead_kb}, free_disk_bytes, running_services:[names], hat_product:str|null, pciex1_gen_cfg:int|null, governor, memlock_limit:'unlimited'|int, tools:{cmake, gcc, git, python3}, override:'PIGEMMA_I_KNOW'|null, collected_at}.
```

### 14.2 Field rules

`facts.json` is written by bash (`pi/lib/common.sh`) and read by stdlib-only Python on the Pi.

- `*_kb` fields are KiB from `/proc/meminfo`. `loadavg1` is a float. `page_size` is
  `getconf PAGESIZE` (16384 on the Pi kernel, 4096 under qemu or in containers).
- `cpu_flags` come from the `/proc/cpuinfo` `Features` line on aarch64 (`flags` on x86).
  `cpu_model` comes from `model name` or `Model`.
- `storage.type` is one of `nvme`, `usb-ssd`, `microsd` or `unknown`. Overlay filesystems,
  containers and anything undetectable report `unknown`; consumers then use the catalog's
  conservative default and say so. `pcie_link_speed` is the sysfs `current_link_speed` string, or
  `null`. `pcie_link_width` and `read_ahead_kb` are int or `null`.
- `free_disk_bytes` is the free space of `model_dir`'s filesystem.
- `running_services` lists systemd unit names. It is empty (with a warning in the log) when
  systemd is not PID 1 or `systemctl` is missing.
- `hat_product` comes from `/proc/device-tree/hat/product`, or `null`. `pciex1_gen_cfg` is the
  `dtparam=pciex1_gen=N` value from `/boot/firmware/config.txt`, or `null`.
- `memlock_limit` is `"unlimited"` or an int in KiB (as `ulimit -l` prints it).
- `tools` maps each tool to its version string, or `null` when absent.
- `override` is `"PIGEMMA_I_KNOW"` only when the operator exported it.
- Optional: `emulated` (bool, `PIGEMMA_EMULATED=1`), `in_container` (bool), `notes` (str[]).

## 15. `io_mem.json` (Pi measured constants) and device-profile overrides

### 15.1 Verbatim contract

Verbatim contract: `data_formats[10]`, name "io_mem.json (Pi measured constants) and device profile override" (copied unchanged from the manifest):

```text
{host, device, collected_at, storage:{seq_read_mbps_qd1:{value,source:'measured',cmd}, seq_read_mbps_qd8:{...}, rand_read_3mb_mbps_qd1:{...}, rand_read_3mb_mbps_qd4:{...}, rand_read_16k_iops:{...}, access_ms:{...derived from 16k latency}}, memory:{read_gbps_by_threads:{'1':..,'2':..,'4':..}, copy_gbps, triad_gbps, source:'measured'}}. feasibility.load_catalog(io_mem=...) overwrites the matching device profile fields (storage.*, mem_bw_gbps) and sets source:'measured'; results/pi/calibration.json {llama_efficiency:{value, source:'measured', from:[tg tok/s, bytes/token, membw]}} overrides the 0.65 assumption in projections.
```

### 15.2 Field rules

- Every storage metric is `{"value": number, "unit": str, "source": "measured", "cmd": str}`,
  where `cmd` is the exact fio command line. `*_mbps` values are 10^6 bytes per second, from
  fio's `bw_bytes`. `rand_read_16k_iops` is operations per second. `access_ms` is the mean 16 KiB
  random-read completion latency (fio `clat_ns.mean / 1e6`).
- `memory.read_gbps_by_threads` has the keys `"1"`, `"2"` and `"4"`, in 10^9 B/s from
  `tools/membw`.
- A dd fallback (fio missing) sets `source: "measured"` and `cmd` to the dd command, and adds
  `notes: ["dd fallback: sequential read only"]`. Metrics it cannot measure are omitted.

## 16. `catalog.json`

### 16.1 Verbatim contract

Verbatim contract: `data_formats[11]`, name "catalog.json" (copied unchanged from the manifest):

```text
See pigemma/data/catalog.json spec: models[] (id, family, quant, repo, file, bytes, bytes_exact, sha256?, nonexpert_bytes, expert_bytes, expert_unit_bytes, n_moe_layers, n_expert, n_expert_used, drafts[], estimated?), devices[] (name, ram_bytes, os_reserve_bytes, compute_buffers_bytes, n_ctx, mem_bw_gbps{value,source}, storage{type, seq_read_mbps{value,source}, rand_read_3mb_mbps{value,source}, access_ms{value,source}}), selection{stream_hit_rate_curve_assumed, viable_stream_tok_s, good_stream_tok_s}, protected_hosts[], protected_service_globs[]. Rule: any numeric that was not measured carries a source string containing 'assumed'.
```

### 16.2 Field rules

- Every non-measured number is an object `{value, source}` whose `source` contains `assumed`,
  e.g. `"assumed: published Pi 5 NVMe Gen3 x1 784-894 MB/s, conservative"`. Measured overrides
  set `source: "measured"` (section 15).
- `bytes_exact: false` marks a size copied from a model card. `gguf_map catalog-refresh`
  replaces it with the exact LFS size and `sha256`.
- The byte budgets in the catalog MUST match section 8.3 for the two downloaded Q4_0 GGUFs.

## 17. Change control

- This file is the single source for the formats above. Change it **before** changing any reader
  or writer. Bump `.moetrace` to `MOETRC02` / `format_version: 2` for any incompatible change to
  the binary layout, and keep a reader for version 1.
- After a change, re-run `tests/test_trace.py` (constants, golden trace), `tests/test_docs.py`
  (links) and the probe's `--smoke` integration, then regenerate
  `tests/fixtures/golden_26b_72tok.moetrace`.
- The verbatim blocks MUST stay byte-identical to `data_formats[]` in
  [design/manifest.json](design/manifest.json). If the manifest changes, update both.
