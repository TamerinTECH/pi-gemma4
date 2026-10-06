# Layer skip for tokens without output (Gemma 4 models with shared-KV layers)

Switch: `PIGEMMA_STRIP_PREFILL=1`. Patch: `pi/fast/pigemma-fast-v7.patch` (base llama.cpp a7b94df), files
`src/models/gemma4.cpp`, `src/llama-graph.{h,cpp}`, `common/speculative.cpp`. Off by default.

## 1. What is removed

Gemma 4 E2B and E4B carry `gemma4.attention.shared_kv_layers` (E2B: 20 of 35 layers, E4B: 18 of 42). llama.cpp
turns this into `hparams.n_layer_kv_from_start` (E2B 15, E4B 24). Layers at or above that index have no key/value
projections of their own; their attention reads the cache cells written by layer `n_layer_kv_from_start - 1`
(global attention) or `- 2` (sliding window) — `src/llama-model.cpp` lines 2796-2805 (`reuse` callback) and
`src/llama-kv-cache.cpp` lines 253-272 (`map_layer_ids[il] = map_layer_ids[il_reuse]`).

With the switch on, the graph gathers the rows of the tokens that have the output flag (`inp_out_ids`) right
before the first shared-KV layer and runs the remaining layers, the final norm and the output head on those rows
only. For a prompt batch that is one row instead of the whole batch. The reference does the same gather, but only
after the last layer (or inside the last layer in masked mode).

## 2. Why no removed computation has a consumer

For a token *without* the output flag, the hidden state above layer `n_layer_kv_from_start - 1` can only flow into
the following places. Each is either impossible for those rows or guarded.

| Possible consumer | Why it does not need the removed rows | Code (a7b94df) |
|---|---|---|
| Persistent KV state | Only layers with `has_kv(il)` pass `k_cur`/`v_cur` to `build_attn`, which is the only place that calls `cpy_k`/`cpy_v`. Shared-KV layers call `build_attn(..., Qcur, nullptr, nullptr, ...)`: nothing is stored | `src/models/gemma4.cpp` 234-275, `src/llama-graph.cpp` 3171-3184, `src/llama-hparams.cpp` 321-331 |
| Other tokens of the same or a later batch | A token influences others only through the KV cache; above the last KV layer every row is computed from its own hidden state plus cache cells written by lower layers | same |
| Logits | Extracted for output rows only (`n_outputs`, `out_ids`) | `src/llama-graph.cpp` 2498-2518 |
| Embeddings (`cparams.embeddings`) | In embedding mode every token is an output (`output_all = cparams.embeddings`), so no row is removed; pooling therefore sees all rows | `src/llama-context.cpp` 1735-1776 |
| Hidden state for a drafter (`embeddings_nextn`), unmasked | **Needs every row** — guarded: no skip unless the export is masked (output rows only). See section 3 | `src/models/gemma4.cpp` 277-282, 410-419 |
| Per-layer inputs (`llama_set_embeddings_layer_inp`) | Buffers are sized one row per token — guarded: no skip if any layer at or above the skip point is exported (v4) | `src/llama-context.cpp` 2278-2300 |
| Eval callback (`cparams.cb_eval`: imatrix, tracing tools) | Would see fewer rows in the upper layers — guarded: no skip with a callback (v5) | `src/llama-cparams.h` |
| Training (`cparams.training`) | Different KV handling — guarded (v5) | `src/llama-graph.cpp` 3165-3171 |
| Per-layer token embeddings (PLE) | Inputs, gathered with the same row indices | `src/models/gemma4.cpp` 368-390 |
| Control vectors, LoRA | Applied per row, no cross-row term | `build_cvec`, `build_lora_mm` |
| In-graph samplers | Attached to the logits of output rows | — |

What the gather preserves: the row order of `inp_out_ids`; the original position of every kept token
(`inp_pos` is gathered with the same indices, so RoPE sees the true positions); the attention mask row of every
kept token (mask `[n_kv, n_tokens]`, gathered along the token axis and cast back to F16 for flash attention), which
carries causal order, sliding window and sequence visibility; the per-layer embedding rows.

Explicit fallback to the reference graph (the condition in `gemma4.cpp`): no `inp_out_ids`; model without
shared-KV layers (26B-A4B, 31B: `n_layer_kv_from_start == n_layer`); several sequences in one batch with
per-sequence KV streams (the mask is then 4-dimensional); more than one position per embedding; embedding inputs
instead of token ids (images, audio); eval callback; training; unmasked drafter export; exported per-layer inputs.

## 3. The drafter (MTP) case

Upstream keeps all rows whenever a drafter is attached (`embeddings_nextn` unmasked, comment "MTP target needs the
hidden state for every token"). That is required for drafters with their own KV cache (Qwen, Step, …): their
`process()` hook replays every prompt token paired with the hidden state of the previous one
(`common/speculative.cpp` 1572-1625).

The Gemma 4 assistant is different: it shares the target's KV cache (`is_mem_shared`, line 1481), and for it the
hook skips that replay ("if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode"). The
only target hidden rows it ever reads are

1. `pending_h` = the row of the **last** token of the batch (the one that has the output flag in a prompt batch), and
2. `verify_h` = the rows of a verification batch, in which **every** token has the output flag.

So for this drafter the rows of non-output tokens are never read. Patch v2 therefore switches the target to the
masked export when `is_mem_shared` and the switch is on, and reads rows by output index. A false output flag alone
is not the argument; the argument is that the one consumer that exists reads output rows only. For any other
drafter the export stays unmasked and the skip stays off.

Evidence that the drafter receives the same inputs: with the skip on, the number of drafted and accepted tokens
per request is identical to the reference drafter run in every comparison made (for example the long prompts:
60/42, 58/43, 51/46 drafted/accepted in both builds), and the generated text is identical.

## 4. Tests

Tool: `pi/fast/tests/fast_check.cpp` (`pigemma-fastcheck`), built inside the patched tree and inside an unpatched
copy of the pinned source. One run writes the full logits of every output row, a digest of the persistent KV
state of each sequence (`llama_state_seq_get_data`), requested embeddings and committed token sequences.
`pi/fast/tests/run_fastcheck.sh` runs it in six modes and compares.

| Scenario | What it protects |
|---|---|
| s1 plain | prompt in one batch, then 12 greedy tokens |
| s2 chunk | the same 45-token prompt in caller batches of 1, 2, 3, 4, 5, 7, 8, 9 tokens (batches with no output at all) |
| s3 boundary | prompts of 511, 512, 513 tokens in one call; 513 also as 511+2 and 512+1 (ubatch limit, sliding window 512) |
| s4 reuse | prompt, 4 generated tokens, context cut back to 40 tokens, other continuation; and the same context built from scratch |
| s5 multiseq | two sequences in one batch, output on the last token of each, then one token each; unified KV buffer and per-sequence streams |
| s6 midrows | output flags on rows 3, 7, 16 and 29 of a 30-token batch (non-final rows) |
| s7 rollback | draft-and-verify by hand with acceptance, rejection and removal of rejected rows from the KV state; committed tokens must equal plain greedy |
| s8 embeddings | per-token embeddings of a 12-token batch |
| s9 widths | batches of 1, 2, 3, 4, 5, 7, 8, 9 tokens with every row flagged (verification shape, tail fix) |

Result on the Pi 5, E2B QAT Q4_0 (`results/pi-real/fastcheck-e2b/`, 140 logit/embedding vectors, 32 KV digests,
3 token sequences per run):

| Comparison with the unpatched reference build | Bit-identical vectors | Argmax changes | Worst KL | KV digests equal | Token sequences equal |
|---|---|---|---|---|---|
| patched, all switches off | 140 / 140 | 0 | 0 | 32 / 32 | 3 / 3 |
| layer skip only | 137 / 140 | 0 | 2.9e-4 | 32 / 32 | 3 / 3 |
| padded tail only | 126 / 140 | 0 | 4.6e-13 | 32 / 32 | 3 / 3 |
| cache-sized chunks only | 140 / 140 | 0 | 0 | 32 / 32 | 3 / 3 |
| all three | 123 / 140 | 0 | 2.9e-4 | 32 / 32 | 3 / 3 |

The three vectors that differ under the layer skip are the long prompts of s3 (511, 512, 511+2 tokens: max logit
difference 0.44-0.58, KL 1.7e-7 to 2.9e-4, same argmax). Cause: with one remaining query row and at least 512
cache cells, ggml's flash attention takes its single-query path (`use_split_kv_path`,
`ggml/src/ggml-cpu/ops.cpp` line 9489), which sums in a different order than the multi-query path the reference
uses for the same row. This is the reference's own sensitivity to how tokens are split into batches, measured in
the same run as a control: the reference against itself, same 513 tokens as one call or as 511+2: KL 2.4e-4;
same 45 tokens in one batch or in chunks of 1, 3 or 7: KL 5.0e-4. The layer skip stays below both.

The s7 scenario committed the same 24 tokens as plain greedy decoding in every mode (12 cycles, 12 accepted
drafts, 9 cycles with a rejection and rollback).

Desktop (x86, same tool, E2B): 137 / 140 bit-identical under the layer skip, 32 / 32 KV digests, the same three
long-prompt vectors differ (KL up to 3.6e-5; reference against itself 5.4e-4).

E4B: see section 6.

Server level (llama-server, greedy, one stream; `results/pi-real/fast/`): 115 of 115 generated texts identical
to the reference build (plain and with the drafter, English, Hebrew, German, code, prompts of 20-374 tokens,
including prompt batches of 64 tokens without any output row).

## 5. Statement of scope

- Supported and tested: Gemma 4 E2B text input; plain decoding; prompt chunking; context reuse after truncation;
  several sequences in a unified KV buffer; non-final output rows; the Gemma 4 assistant drafter with acceptance,
  rejection and rollback; embeddings mode (no row removed).
- Falls back to the reference graph: everything in the list at the end of section 2.
- Not claimed: bit-identical logits. The persistent state is bit-identical in every test; logits are bit-identical
  unless the attention kernel picks a different code path for the smaller number of query rows, and then they
  differ by less than the reference differs from itself across batch splits. Greedy output was identical in
  every test; with sampling at temperature > 0 a difference of this size can change a draw, exactly as a
  different batch size does in the reference.
- Not tested: image/audio input (falls back), LoRA and control vectors (no cross-row term, but no test),
  contexts beyond 513 tokens in the API tool (the server tests go to 374-token prompts plus generation).

## 6. Patch v7 and E4B (Pi 5, 2026-10-06 21:13-21:35)

The same scenarios were repeated with the final patch (v7) for E2B and for E4B (42 layers, 18 of them shared-KV,
hidden size 2560; `results/pi-real/fastcheck-e2b-v7/`, `results/pi-real/fastcheck-e4b-v7/`, full log
`fastcheck-e4b-v7/phase_a.out`). E2B v7 reproduces the table of section 4 number for number.

E4B QAT Q4_0, 140 vectors, 32 KV digests, 3 token sequences per run:

| Comparison with the unpatched reference build | Bit-identical vectors | Argmax changes | Worst KL | KV digests equal | Token sequences equal |
|---|---|---|---|---|---|
| patched, all switches off | 140 / 140 | 0 | 0 | 32 / 32 | 3 / 3 |
| layer skip only | 137 / 140 | 0 | 3.6e-4 | 32 / 32 | 3 / 3 |
| padded tail only | 126 / 140 | 0 | 7.6e-13 | 32 / 32 | 3 / 3 |
| cache-sized chunks only | 140 / 140 | 0 | 0 | 32 / 32 | 3 / 3 |
| all three | 123 / 140 | 0 | 3.6e-4 | 32 / 32 | 3 / 3 |

Controls in the same run, reference build against itself: 513 tokens as one call or as 511+2: KL 1.5e-4
(largest logit difference 0.56); 45 tokens in one batch or in chunks of 1, 3 or 7: KL 9.3e-4 (0.86);
513 as 512+1 and "reused context" against "fresh context": bit-identical. As for E2B, the layer skip changes
logits only for the three long-prompt vectors and by less than the reference's own batch-split variation.

The 26B-A4B and the 31B have no shared-KV layers (`n_layer_kv_from_start == n_layer`): the condition is false
and the graph is the reference graph. Nothing was measured for them with the switch on, because nothing changes.

## 6a. The same change on current llama.cpp master (desktop x86)

`pi/fast/upstream/gemma4-shared-kv-layer-skip.patch` is the same change without the environment switch, written
against master 51ce9c1 (after the upstream refactoring of the nextn guards). Results of
`pi/fast/upstream/run_upstream_check.sh` are in `pi/fast/upstream/README.md` (default settings, flash attention
off, quantized KV cache).

## 7. Reproduce

```sh
# Pi, after ~/pigemma/fast/mount_work.sh and build_fast.sh
~/pigemma/fast/tests/run_fastcheck.sh llama-fast-v6 /mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf e2b
# any Linux machine: copy pi/fast/tests into <tree>/examples/pigemma-fastcheck (install_into_tree.sh), build the
# target pigemma-fastcheck in a patched and in an unpatched tree, then
pigemma-fastcheck -m MODEL.gguf --text corpus.txt --out REF.bin            # unpatched build
PIGEMMA_STRIP_PREFILL=1 pigemma-fastcheck -m MODEL.gguf --text corpus.txt --out S.bin
pigemma-fastcheck --compare REF.bin S.bin --require-kv-equal
```
