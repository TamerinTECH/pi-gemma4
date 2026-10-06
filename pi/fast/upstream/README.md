# Material for a possible upstream contribution (the operator reviews and submits; nothing here was sent anywhere)

Target: `ggml-org/llama.cpp` master at `51ce9c11a6f2dfa895696c0048c4333e8953b728` (2026-10-06), i.e. after
PR #30017, which replaced the nextn guards by `crop_before_nextn()` / `crop_after_nextn()`.

## Read first: the project's rules for AI-assisted work (CONTRIBUTING.md, AGENTS.md at that revision)

- AI-generated code is allowed, with disclosure, and "you are 100% responsible for every line". The contributor
  must be able to explain every line without AI help; the guide asks for about one hour of manual review per
  200-400 lines.
- "It is strictly prohibited to use AI to write your posts for you (bug reports, feature requests, pull request
  descriptions, Github discussions, responding to humans, ...)". AGENTS.md repeats it for commit messages.
  **So this directory contains patches, facts and commands, and no issue or PR text. The text has to be yours.**
- "Features must begin with an issue, not a PR"; new contributors: "Limit your open PRs to 1".
- "Check for an existing PR addressing the same change; if one exists, comment there to work with its author
  instead of opening a duplicate." This applies to the layer skip (see below).
- If you commit code written with an assistant: trailer `Assisted-by: <assistant name>`, not `Co-authored-by:`.
- Agents must not push or open PRs. Nothing was pushed; the scratch clone used here has no credentials.

## Two separate changes

| File | Touches | Size | What |
|---|---|---|---|
| `gemma4-shared-kv-layer-skip.patch` | `src/models/gemma4.cpp`, `src/llama-graph.{h,cpp}`, `common/speculative.cpp` | +73 / -13 | tokens without output skip the layers that store no KV (the TODO in `gemma4.cpp`) |
| `repack-pad-tail.patch` | `ggml/src/ggml-cpu/repack.cpp` | +37 / -2 | 2-3 rows beyond a multiple of 4 run as one zero-padded 4-row block instead of one full weight sweep per row |

`port_to_master.py layer-skip|pad-tail <tree>` produces the same diffs with checked string replacements (it stops
if the upstream text has moved). Both patches apply to a clean checkout (`git apply --check`), contain ASCII only
and no environment switches. They are independent: different files, different maintainers (model graph vs ggml CPU
backend; ggml changes need two approvals).

Not proposed, on purpose:

- the cache-sized weight chunks (`PIGEMMA_CHUNK_KB`): +14 % only at 8 rows per pass on the Pi, nothing elsewhere,
  and the best value is a property of one CPU;
- the drafter cluster head: it needs two tensors that the upstream converter currently drops, so it is a
  converter + GGUF-key feature and has to start as an issue;
- the test tool as a new test program: the guide asks to extend existing tests, not to add a test system.
  It is evidence for the discussion (see "Evidence").

## 1. Layer skip

### Relation to existing upstream work (checked 2026-10-06; re-read the threads before posting)

- The idea is the maintainers' own: `// TODO @ngxson : strip unused token right after the last KV layer to speed
  up prompt processing` in `src/models/gemma4.cpp`. No novelty claim follows from implementing a TODO.
- **Open draft PR #29189** "Optimizing Gemma4 graph for prompt processing" (Meet91721, opened 2026-09-20). It
  skips the work after the attention only when a batch has no output token at all. The only review comment
  (ngxson, 2026-09-21, on `src/models/gemma4.cpp`): "this is an incorrect fix, you did not see my next TODO".
  By the rule quoted above the right first step is a comment there (or an issue that links it), not a competing PR.
- PR #30017 (merged 2026-10-06) changed exactly the lines this patch edits; the patch is written against the
  new helpers.

### What the patch does differently from #29189

- It removes rows as soon as the first shared-KV layer starts, for any batch (also when the last token has an
  output), by gathering `inpL`, the positions, the attention-mask rows and the per-layer input rows with
  `inp_out_ids`. This is the case that matters for a chat prompt: 1 output row out of N.
- It keeps the reference graph whenever something reads one row per token from those layers: unmasked nextn
  embeddings, exported layer inputs, an eval callback, training, embedding inputs, per-sequence KV streams with
  more than one sequence in the batch.
- With the Gemma 4 assistant (shared KV, `is_mem_shared`) the target switches to the masked nextn export, because
  that drafter never replays the prompt and only reads target rows of tokens that have the output flag. For every
  other drafter nothing changes (unmasked export, no skip).

### Why this is safe (short form; full table in `docs/fast/LAYER_SKIP.md`)

A layer at or above `n_layer_kv_from_start` calls `build_attn(..., Qcur, nullptr, nullptr, ...)`: it stores no
keys or values. A token's state in those layers therefore reaches nothing but its own logits, its own exported
hidden row and callbacks that look at intermediate tensors. Logits and masked hidden rows exist for output tokens
only; the other consumers are the guarded cases.

### Points a reviewer will probably raise

- `build_attn` (shared graph code) gets one optional argument. The alternative would be a second mask input
  built for the output rows; the argument is the smaller change, but it is shared code that only Gemma 4 uses.
- The logits are not bit-identical in one situation: one remaining query row and 512 or more KV cells make ggml's
  CPU flash attention take its single-query path, which sums in another order. The reference shows the same
  size of difference between two ways of splitting the same tokens into batches (control in "Evidence").
- Graph topology now depends on the unified-KV flag and on the number of sequences in the batch (already part of
  the graph-reuse key), not on the number of outputs.
- Only the CPU backend was run. `ggml_get_rows` on a mask and `ggml_cast` to F16 are standard ops, but no GPU
  backend was tested.
- The quantized-KV path and the non-flash path were run (see below); image/audio input falls back and was not run.

### Evidence

Filled in below from `results/desktop-fast/upstream-51ce9c1/` (desktop x86, master + patch vs clean master) and
from the Pi 5 measurements of the same change on a7b94df (`docs/fast/LAYER_SKIP.md`,
`docs/fast/DECISION_2026-10-06.md`).

RESULTS_LAYER_SKIP

### Commands a reviewer can run

```sh
# speed: prompt processing of a Gemma 4 E2B/E4B GGUF, before and after
llama-bench -m gemma-4-E2B_q4_0-it.gguf -t 4 -p 512 -n 32 -r 3
# correctness: same greedy text with and without the patch, long prompt, plain and with the assistant
llama-server -m gemma-4-E2B_q4_0-it.gguf -c 4096 --reasoning off
llama-server -m gemma-4-E2B_q4_0-it.gguf -md mtp-gemma-4-E2B-it.gguf --spec-type draft-mtp -c 4096 --reasoning off
# other paths: -fa off ; -ctk q8_0 -ctv q8_0 ; -b 64 (prompt batches without any output row)
```

## 2. Padded tail in the repacked matmul

### The problem (measured, llama.cpp a7b94df, Pi 5, E2B Q4_0, `llama-bench -p 420 -n 0 -b N`)

| Rows per pass | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| tok/s | 13.1 | 27.2 | 24.9 | 24.8 | 24.3 | 36.6 |
| ms per pass | 229 | 147 | 201 | 242 | 288 | 219 |

`forward_mul_mat_one_chunk` sends `nrows - nrows % 4` rows through `gemm` and every leftover row through its own
`gemv` call, i.e. one more sweep over the weight chunk per leftover row. When the weights do not fit in the cache,
3 rows cost more than 4. Batches of 2-3 rows are exactly what draft verification produces.

### The change

Each thread builds one zero-padded 4-row block from the 2-3 leftover rows (a few kilobytes, quantized like the
other blocks) and runs one `gemm` sweep for it; the padded result rows are dropped. One leftover row keeps the
`gemv` path. No shared state: the block is a local buffer of the calling thread.

Difference to the experiment-local version in `pigemma-fast-v7.patch`: that one keeps the padded block in one
process-wide buffer filled by thread 0. That is fine for `llama-server` (one decode at a time) but would be a data
race for a program that decodes on two contexts at once, so the upstream version uses per-thread buffers.

### Relation to existing upstream work

No PR for the leftover rows was found on 2026-10-06. Nearest neighbours (titles and states read from the GitHub
API that day; read them before writing an issue):

- #17526 "ggml-cpu: repack: Fix chunks being too small with small matrix shapes in REPACK forward_mul_mat"
  (Alcpz, merged, Nov 2025): the chunk arithmetic of the same function, tested on a Raspberry Pi 5.
- #17791 "ggml-cpu: add repack GEMM and GEMV for floating-point" (taimur-10x, open since Dec 2025): renames
  `ggml_quantize_mat_t` to `ggml_repack_mat_t`; this patch calls that function, so the two conflict textually.
- #27851 "ggml-cpu: tiled mul_mat for k-quants" (jbooth, merged, Aug 2026): a different, tiled path for k-quants;
  check whether your quantization type still reaches `forward_mul_mat` of `repack.cpp` on the target CPU.

### What a reviewer will ask for

- "New kernels are expected to come with concrete perf data (throughput across realistic tensor shapes)": the
  table above plus the after-numbers below; other quantization types that use the repacked path (Q4_K, Q5_K,
  Q6_K, Q8_0) were exercised through the model's Q6_K output head only.
- Result rows of a tail token now come from the 4-row kernel instead of the 1-row kernel; on ARM the two differ
  in the last bits (largest logit difference seen 6e-6, no argmax change).
- x86 gains may be small or zero (large caches); the effect is a memory-bandwidth effect.

### Evidence

RESULTS_PAD_TAIL

## Reproduce the check

```sh
# three trees: clean master, master + layer skip, master + padded tail; test tool installed into each
pi/fast/tests/install_into_tree.sh <tree>
pi/fast/upstream/run_upstream_check.sh      # inside the build container, see the header of the script
```
