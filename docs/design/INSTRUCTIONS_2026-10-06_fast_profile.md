# Brief 3 (the operator, 2026-10-06 evening, pasted verbatim) — fast-profile milestone

Received after the fast-profile report (prompt processing 62 -> 170 tok/s, same text). It replaces the
"at least 100 tok/s" session goal as the near-term objective. Text below is unchanged.

---

Continue from the current fast-profile implementation. Do not restart
the engine or discard the working baseline.

The next milestone is a reproducible, correctness-validated prefill
optimization, followed by the exact drafter vocabulary shortcut and
a fair model quality/latency comparison.

1. Preserve the result now

Pause the current automatic 100 tok/s goal hook. Preserve its previous
configuration and document that the near-term objective has changed.
Do not replace it with another unbounded retry loop.

Commit the patches, build scripts, correctness tests, benchmark
manifests, and exact upstream base revision to a dedicated local branch.
Do not publish private material or push publicly without authorization.

Preserve raw results in durable storage. Record their locations and
checksums. Do not commit model weights, credentials, or sensitive data.

Move the reproducible source/build setup out of /tmp. Keep the reference
binary and ContextFriend services untouched.

Verify a fresh build in a separate directory without rebooting the Pi.

2. Validate the layer-skip dependency rule

Document why each removed computation has no required consumer.

Check not only logits, but also persistent KV state, requested
embeddings, auxiliary hidden states, MTP state, and speculative
verification requirements.

Audit the upstream guard around unmasked nextn embeddings. Demonstrate
why our supported MTP path remains correct rather than assuming that
a false logits flag makes the token unnecessary.

Test plain decoding, prompt chunking, reused context, multiple sequences,
non-final requested output rows, and speculative acceptance/rejection
with rollback.

Include batch-size boundaries around the tail fixes, such as
1, 2, 3, 4, 5, 7, 8, 9, 511, 512, and 513.

Preserve original token positions, masks, sequence identities, and
per-layer-input alignment when gathering rows.

Compare intermediate values where useful, not only decoded strings.

For unsupported consumers or modes, fall back explicitly to the
reference execution path. Document supported scope rather than
claiming universal exactness.

3. Separate performance contributions

Run the four matched configurations:
reference/plain, patched/plain, reference/drafter, patched/drafter.

Keep the model, input tokens, output requirements, reasoning mode,
sampling, context state, and thread settings matched.

Measure layer skipping and the two tail fixes separately in a small
ablation before reporting the combined profile.

Report paired per-request results, repeated-run variation, time to first
visible answer, complete-answer time, committed decode throughput,
peak memory, and any regressions.

Do not attribute the benefit of enabling the drafter to our patch.

Use --reasoning off for the direct-answer profile only after verifying
the actual rendered prompt and generated behavior. Keep thinking-enabled
quality tests as a separately labeled configuration.

4. Implement the built-in assistant vocabulary shortcut next

Identify and use the assistant's intended clustering tensors and token
mapping. Keep the target verification path unchanged.

Measure draft construction time, verification time, accepted draft
tokens, total committed tokens per cycle, and complete request latency.

Compare across English, Hebrew, German, and code. A reduction in draft
cost is not a win if acceptance drops enough to cancel it.

For greedy mode, verify target token-sequence agreement.

For stochastic mode, audit the actual acceptance/resampling algorithm
and the proposal probabilities after vocabulary restriction. Do not
equate same-seed text matching with a distributional guarantee.

Clarify whether the separately proposed approximate output-head change
affects only draft proposals or the target verifier. Defer any target
approximation.

5. Run one fair quality-and-latency screen

Evaluate E2B, E4B, and the existing 26B on the same held-out task content.

Keep correctness testing and model-quality testing separate.

Use task-appropriate output budgets for quality evaluation. Preserve
the previous short-cap scenario as a separate product test, and report
truncated answers explicitly.

Include short factual questions, extraction from supplied material,
negation, numbers, multilingual instructions, and executable code tests.

Use objective scoring where possible. Record both correctness and
complete-answer latency. Do not compare percentages from different sets.

Test E4B with the validated dependency optimization and its matching
assistant if supported. Inspect its architecture instead of hardcoding
the E2B layer counts. Do not assume the same speedup.

Do not build an automatic model router yet. First identify which
configuration is actually suitable for each measured workload.

6. Gate the larger 26B prepacking effort

Profile the existing 26B with and without its assistant.

Identify exactly which cost the packed representation would remove.
Prototype representative operations or a block before implementing
the complete converter and loader.

Include packing cost, peak memory, duplicated mappings, verification
batches, and expert-weight traffic in the analysis.

Keep quantization values and scales unchanged in the exact-layout track.

Treat 4.6 -> 6.5 tok/s as a hypothesis until measured in full generation.

Do not apply the E2B shared-KV skip to a model that lacks the necessary
architecture.

7. Prepare an upstream-ready contribution

Separate the layer-skip change from the two tail fixes.

Prepare a minimal patch, dependency explanation, regression tests,
supported-mode description, benchmark commands, and known limitations.

Check for overlapping upstream work before preparing the submission.
Do not claim that identifying an existing TODO establishes novelty.

Keep this preparation independent of the longer 26B work.

8. Report a decision

Return:
- The durable commit and reproduction instructions.
- The supported scope and correctness evidence for layer skipping.
- The isolated patch gain and combined patch-plus-drafter gain.
- The measured vocabulary-shortcut result.
- A matched E2B/E4B/26B quality-latency table.
- A go/no-go recommendation for full 26B prepacking.
- The next justified CF-Engine component.

Recalculate the current speculative throughput estimate using committed
tokens and all measured cycle costs. Describe it as a configuration-
specific estimate, not a proof that all exact methods share that limit.

Do not spend the next iteration on a full runtime rewrite, broad assembly
autotuning, or approximate target computation.

Prioritize preserving and validating the real gain, then improving the
quality-latency trade-off.
