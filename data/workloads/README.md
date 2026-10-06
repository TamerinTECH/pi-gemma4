# Workloads for the decision milestone (brief 2026-10-05, sections 3, 4, 7)

Three **disjoint** request sets. Each set has exactly one job; using a set for another job
invalidates the comparison.

| File | Only used for | Conversations | Requests | Generation bound (tokens) | First-request prompt tokens (measured) | New prefill per pass (projected upper bound) |
|---|---|---|---|---|---|---|
| `calibration.jsonl` | expert ranking (routing traces of real generations) | 24 | 32 | 2320 | 4435 | 5279 |
| `tuning.jsonl` | pilot runs and the pinning-budget choice (frozen before evaluation) | 8 | 11 | 704 | 1577 | 1850 |
| `heldout.jsonl` | the A/B/C/D evaluation only | 19 | 26 | **1488** (limit 1500 per method-repeat) | 3664 | 4289 |

Token numbers: `token_counts.json` (written by `python -m pigemma.workloads count-tokens`):
`llama-tokenize` b11398 (a7b94df, Windows prebuilt), **vocab-only load of
`gemma-4-26B_q4_0-it.gguf` (no weights, no inference)**, `--no-escape`, BOS included. "Measured" =
the rendered prompt of every request with all earlier replies left empty. "Projected upper
bound" = every earlier reply at its full `max_tokens`; it is not a measurement. Length classes
(first request): short < 80 tokens, medium 80-499, long >= 500. Long prompts: 554-884 tokens
(calibration 5, tuning 2, held-out 4).

## Line format

```json
{"id": "ho-17", "set": "heldout", "lang": "en|he|de|mixed", "category": "...",
 "turns": [{"role": "user", "content": "..."}, {"role": "user", "content": "..."}],
 "max_tokens": 56, "context_reuse": "none|follow_up_same_conversation", "notes": "..."}
```

* `turns` holds only the **input** messages (an optional leading `system`, then `user`
  messages). After every user message the model under test generates the assistant reply
  (greedy, thinking disabled), and that reply is appended to the history before the next user
  message. Follow-up prompts therefore contain real model output; for an exact-caching method
  they are identical to the reference run's, for an approximate method they may differ (record
  them). `pigemma.workloads.request_messages(conv, k, replies)` builds request k.
* `max_tokens` (48-96) is the generation bound per assistant reply. It is sent to llama-server as
  its native `n_predict` (the server treats `max_tokens` as an alias). Replies may end earlier at
  end-of-turn; record the real generated count.
* `category`: chat, instruction_following (several simultaneous constraints), code,
  extraction (from text given in the prompt), summarization (long document), topic_switch
  (multi-turn, abrupt topic and often language changes), follow_up (multi-turn on the same
  document or code), translation.
* `lang`: language of the prompt text (`mixed` = the turns switch language). Code is a category.

## Context reuse (KV cache) - exact procedure

* `context_reuse: "none"` - one request; send `cache_prompt: false` so no KV prefix of an earlier,
  unrelated request is reused (the shared `<bos><|turn>user\n` is 4 tokens, but it is still
  disabled to keep prefill cost exact).
* `context_reuse: "follow_up_same_conversation"` - request 0 of the conversation with
  `cache_prompt: false`; requests 1..n back-to-back on the same slot (`-np 1`, no other request
  in between) with `cache_prompt: true`.
* What can be reused (read from the Gemma 4 GGUF chat template, verified with jinja2 against
  `tests/fixtures/gemma4_chat_template.jinja`): with thinking disabled, the generation prompt
  ends in `<|turn>model\n<|channel>thought\n<channel|>` (the empty thought block = 4 tokens),
  but an earlier reply is re-rendered as `<|turn>model\n{reply, trimmed}<turn|>\n` **without** the
  thought block. The reusable prefix of request k is therefore request k-1's prompt **up to and
  including** its last `<|turn>model\n`; the earlier reply, the new user turn and the new
  generation prompt are prefilled again. `token_counts.json` gives that prefix as
  `reusable_prefix_upper`; the runner must compare it with the server's `timings.cache_n`
  and report differences instead of assuming reuse.
* Gemma 4 uses sliding-window attention (window 1024) in 5 of 6 layers. llama-server keeps
  context checkpoints for SWA models (`--ctx-checkpoints`, default 32 per slot at a7b94df); a
  follow-up whose prefix exceeds the window may fall back to a full re-prefill ("forcing full
  prompt re-processing" in the server log). All conversations here stay below ~1000 tokens,
  but the measured `cache_n` is the evidence, not this note.
* Keep `-c`, `-np`, `--ctx-checkpoints`, `--swa-full` and the KV-cache type identical across
  methods A/B/C/D.

## Disjointness and provenance

`python -m pigemma.workloads validate` (also run by `tests/test_quality_v2.py`) fails on:
schema errors, duplicate ids, a set size outside its range, a held-out generation bound above
1500 tokens, an identical turn in two sets, any shared run of 8 words between two sets, and
any shared 8-word run between tuning/held-out and the older corpus (`data/calibration/*`,
`data/prompts/qa_set.json`, `qa_hard.json`, `gen_prompts.json`) that earlier traces may have
used. `data/prompts/qa_heldout_v2.json` is checked against all three sets the same way.

All long documents are original text written for this repository on 2026-10-05 (fictional
organisations, people and numbers), except two **calibration-only** prompts that reuse the
repository's own corpus: `cal-16` (the opening of
`data/calibration/he_discussion_llm_efficiency.txt`, up to its first question) and `cal-17`
(`data/calibration/de_schule_kleincomputer.txt`). No third-party text.

## Commands

```bash
python -m pigemma.workloads validate            # exit 1 on any problem
python -m pigemma.workloads summary
python -m pigemma.workloads render ho-17 --turn 1   # rendered Gemma 4 prompt, replies as <REPLY k>
python -m pigemma.workloads count-tokens --model 26b-q4   # re-measure token_counts.json
```
