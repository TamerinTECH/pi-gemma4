# pi-gemma4 / CF-Engine — current handover

**Edit this head in `docs/cf-engine/STATUS.md`, then run `python cf-engine/update_handover.py`** (it rewrites the top of
`CLAUDE.md` and keeps everything from "Historical decision brief" down). The handover before CF-Engine is preserved
verbatim in `results/cf-engine/CLAUDE.before.md`.

## CURRENT BRIEF: brief 3 (the operator, 2026-10-06 ~20:15) — read `docs/design/INSTRUCTIONS_2026-10-06_fast_profile.md` (verbatim)

The near-term objective is no longer "100 tok/s". It is: preserve and validate the prefill gain, then the exact
drafter vocabulary shortcut, then a fair E2B / E4B / 26B quality-latency screen, a go/no-go for 26B prepacking
and an upstream-ready patch. The old `/goal` stop hook ("at least 100 tokens / second") cannot be paused from
inside a session: **the operator has to run `/goal clear`**; its text is preserved in the section below.

Progress (update after every step):

| Part | State | Where |
|---|---|---|
| 1 preserve | DONE: local branch `fast-profile` (e2e614c baseline, c7524d1 fast profile, 6eb3b49 + later commits = brief-3 work; nothing pushed); raw results checksummed (`results/pi-real/SHA256SUMS-2026-10-06.txt`); Pi work area out of `/tmp`: ext4 image `/mnt/usb/pigemma/fast/pigemma-work.ext4` loop-mounted at `/mnt/pigemma-fast` (`pi/fast/mount_work.sh`, once per boot); fresh build from pristine source + patch file verified (`llama-fast-v4-fresh`; current tree `llama-fast-v6` = v7 by in-place upgrade, binaries `bin-v7`); reproduction notes `pi/fast/README.md`, `pi/fast/BASE_REVISION` | `pi/fast/` |
| 2 dependency rule | DONE: `docs/fast/LAYER_SKIP.md` (consumer table, drafter case, guards, scope). Tool `pi/fast/tests/fast_check.cpp` (9 scenarios: logits, KV-state digests, token sequences; options `--flash off`, `--kv-type q8_0`), built in the patched tree AND an unpatched reference copy. Pi, patch v7: **E2B** reference vs switches-off 140/140 bit-identical; vs layer skip 137/140, KV digests 32/32, 0 argmax changes, worst KL 2.9e-4; **E4B** 140/140; 137/140, 32/32, 0, worst KL 3.6e-4. The reference against itself (same tokens, other batch split) reaches KL 5.0e-4 (E2B) and 9.3e-4 (E4B), so the skip stays inside the reference's own variation | `results/pi-real/fastcheck-e2b-v7/`, `fastcheck-e4b-v7/` |
| 3 matched benchmark | DONE (Pi, 9 prompts, 3 alternated repetitions, `--reasoning off`, streaming): reference/plain 131.8 s, patched/plain 120.6 s, reference/drafter 85.1 s, patched/drafter 73.1 s for the series; first visible text 0.66 -> 0.30 s (short prompts), 3.5-6.1 -> 1.3-2.2 s (210-367-token prompts); prompt 56.6 -> 147.4 tok/s; decode 7.45 -> 7.46 plain, 12.60 -> 12.82 with drafter; same text 27/27 in every configuration; no request slower; peak RSS not higher. Ablation: layer skip = the whole prompt gain; padded tail +2.9 % decode at 3 drafts and 7.5 -> 13.2 tok/s at 2 drafts; chunks +1.3 % | `docs/fast/DECISION_2026-10-06.md` sections 3 and 3a, `results/pi-real/fast3/{matched,ablation}/` |
| 4 vocabulary shortcut | DONE: exact drafter cluster head (patch v6/v7, `PIGEMMA_MTP_CENTROIDS=1`, drafter file from `pi/fast/make_centroid_gguf.py`). Pi, 50 requests EN/HE/DE/code, two repetitions: draft construction 21.1 -> 16.0 ms per pass (-25 %), verification 148 ms unchanged, accepted drafts 57.4 -> 55.6 %, tokens per pass 2.659 -> 2.608, decode 15.71 -> 15.88 tok/s (+1.1 %; German -4 %), same text 50/50. Sampling audit (desktop): default verification never uses proposal probabilities, both heads commit identical text (6/6 long, 400/400 short = plain sampling with the same seed); rejection mode: digit histograms vs plain p = 0.08-1.0, heads within 2 counts. Verdict: correct, about 1 %, optional switch only. Approximate target head: deferred, nothing approximated | `docs/fast/DECISION_2026-10-06.md` section 4, `results/pi-real/fast3/centroids/`, `results/desktop-fast/sampling/` |
| 5 model screen | DONE (44 held-out tasks, direct answers, task budgets, nothing cut off): **E2B 23/44** (reference plain 464.9 s, fast profile + assistant 200.7 s, 17.0 tok/s); **E4B 32/44** (668.1 s -> 273.7 s, 3.84 -> 10.37 tok/s, layer skip validated, prompt 28 -> 46 tok/s); **26B-A4B 42/44** (`-nr` plain 929 s cold / ~831 s warm, 3.77 tok/s; + its assistant 699.4 s, 5.08 tok/s, 84 % of drafts accepted). Separately labelled: E2B with thinking 32/44 in 1419 s (24 s to the first visible word; code cut by the 768-token budget). One E2B code answer differed between reference/plain and fast/drafter: attribution run shows the UNPATCHED build with drafter gives the same second text, so patched = reference in the same mode; drafter = plain is upstream behaviour (184 of 186 answers) | `docs/fast/DECISION_2026-10-06.md` sections 5, 5a; `results/pi-real/fast3/{screen,think,attrib}/` |
| 6 26B gate | DONE: **no-go** for prepacking the 26B as a whole and as a plain-decode measure (one-row pass reads 2.38 GB = 5.0 tok/s ceiling; dense parts already at 10.4-11.4 GB/s unpacked; packed expert kernels gave 4.61 -> 4.63 in Milestone A; expert traffic is per row). **Go for one bounded prototype only**: pack the dense 1.58 GB (attention, dense feed-forward, router, output matrix) and use the assistant; E2B representative operations at 4 rows per pass: 110.8 ms packed vs 255.4 ms unpacked; projection (not measured on the 26B) 5.1 -> ~6.7 tok/s; acceptance bar +20 % on the 44-task screen, identical output in the same mode, no more swap than today. The 26B runs put 1.3 GB of the server's memory and ~0.7 GB more into swap; services stayed active | `docs/fast/DECISION_2026-10-06.md` section 6; `results/pi-real/fast3/profile26/`, `results/pi-real/26b_bytes_per_token.json` |
| 7 upstream | patches DONE: `pi/fast/upstream/gemma4-shared-kv-layer-skip.patch` (+73/-13, 4 files) and `repack-pad-tail.patch` (+37/-2, per-thread buffer instead of the shared one) against master 51ce9c1, no environment switches, ASCII, both apply to a clean checkout and build; notes `pi/fast/upstream/README.md` (project rules: the operator writes issue/PR text himself, features start with an issue, comment on the open draft PR #29189 first, `Assisted-by:` trailer). Correctness run on master in Docker running (`run_upstream_check.sh`); desktop speed numbers pending | `pi/fast/upstream/` |
| 8 report | pending | |

Patch versions: v1 three switches; v2 layer skip together with the Gemma 4 drafter; v3 pad once per call;
v4 no skip when per-layer inputs are exported; v5 no skip with an eval callback, in training mode or for
embedding inputs; v6 drafter cluster head + `PIGEMMA_SPEC_STATS=1` (draft time per request); v7 loader
registration of the two cluster tensors. Current patch file `pi/fast/pigemma-fast-v7.patch` (sha256 f3fdee94…),
Pi binaries `/mnt/usb/pigemma/fast/bin-v7/`.

## Goal and honest status (2026-10-06)

the operator's session goal (set again 12:20): *"make full gemma4 model work on raspberry pi 5 cpu only with fast results
(at least 100 tokens / second)"*. Status, all measured on the Pi 5 (16 GB, CPU only) with the smallest full Gemma 4
(E2B, Google QAT Q4_0):

- **Prompt processing: 170 tok/s** with the patched llama.cpp copy "pigemma-fast" (reference build: 62). Exact, same
  generated text. This is the only place where 100 tok/s is reached.
- **Generation: 7.5 tok/s plain, 14 tok/s with the MTP drafter on open text, 17–20 on copy-heavy text.** 100 tok/s of
  generated text is not reachable on this board with any exact or verified method (one model pass costs at least
  126 ms of memory traffic and carries about 2.4 accepted tokens; ceiling about 17 tok/s on open text).
- Whole request with a 217–374-token prompt and 64 generated tokens: 12.1–14.9 s (reference, plain) → **4.9–5.5 s**
  (fast profile), same text.
- Several users at once: 33.6 → 37.6 tok/s summed over 16 streams; 2 and 3 streams 9–10 → 12.5 and 17.7.
- Primary large target 26B-A4B QAT Q4_0: 4.6 tok/s decode / 11.7 tok/s prefill (no shared-KV layers, so the layer
  skip does not apply).

The stop hook with this goal is still armed; the operator has to clear it or set a reachable target.

## Who did what on 2026-10-06

- 08:35–09:25 Claude: memory-bandwidth and speed-ceiling measurements (`results/pi-real/speed-ceiling/`), stopped on
  the operator's "now stop, we are going to update our libraries codes".
- 09:27–11:26 **OpenAI Codex session** (transcript `a local transcript kept outside this repository`, branch `cf-engine`, no
  commits): built **CF-Engine Milestone A** from the operator's "Build CF-Engine" brief plus the follow-up "Add an AArch64
  assembly backend". It ran out of quota after the last Pi job, before writing the final numbers.
- 12:20–13:20 Claude: read the transcript and artefacts, closed the Milestone A report below, ran and analysed the
  verified-ceiling run (42 min on the Pi, services healthy throughout, max 73 °C, no throttling).

## CF-Engine Milestone A — final result (Codex, closed by Claude from the saved artefacts)

Code: `cf-engine/` (deployment compiler `compiler.py`, reversible packed format `FORMAT.md`, native adapter
`adapter.cpp` hooked into a patched copy of the pinned llama.cpp a7b94df under `cf-engine/vendor-llama/`, kernels
`kernels.cpp` portable + NEON dot-product, experimental `q4_tile4_a76.S`, operator bench with real upstream and
KleidiAI v1.24.0 kernels). Docs: `cf-engine/README.md`, `docs/cf-engine/{ARCHITECTURE,ASSEMBLY,AUDIT,KLEIDIAI,RESULTS}.md`.
Raw results: `results/cf-engine/`; Pi copies under `/mnt/usb/pigemma/cf-engine/` (binaries in `bin/`, build tree in
`/tmp/cf-engine-work`, lost on reboot).

- **Integration is real**: CF kernels execute layer-0 gate/up and down (2 of 60 expert projection tensors) inside full
  Gemma 4 26B-A4B QAT Q4_0 generation on the Pi; every fallback is counted (98.6–98.9 % of graph time is still
  reference code).
- **Correctness**: the first kernels produced the same ten texts but FAILED the logit/routing gate (different float
  accumulation order, fused multiply-add). After the fix all three (portable, NEON, assembly) are **bit-identical** to
  upstream: 20 paired generations, 31,981,568 logit words, 7,680 routing records; 4,704 numerical cases, 520
  guard-page cases, 312 register-preservation probes on the Pi. Lesson kept: equal text is not a correctness proof.
- **Speed: no gain.** Assembly is excluded from the default build (`-DCF_EXPERIMENTAL_ASM=ON` only for reproduction).

| Real expert projection (8 experts, 17.84 MB of Q4_0), median ms, 3 runs × 61 iterations | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| upstream row dot product (`vec_dot`, what `-nr` uses) | 3.50 | 1.78 | 1.56–1.82 |
| upstream repacked 4x4 kernel | 1.54–1.56 | 1.32–1.39 | 1.47–1.49 |
| KleidiAI DOTPROD | 1.71–1.78 | 1.33–1.38 | 1.49–1.52 |
| CF NEON | 2.81 | 1.43–1.44 | 1.52–1.57 |
| CF assembly (best variant) | 2.95 | 1.50 | 1.48–1.58 |

| Full 26B-A4B QAT Q4_0 generation, same process, alternating | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| reference decode tok/s (prefill tok/s) | 2.057 (3.29) | 3.877 (6.50) | 4.608 (11.69) |
| CF NEON | 2.062 (3.30) | 3.886 (6.52) | 4.627 (11.72) |
| CF assembly | 2.060 (3.30) | 3.884 (6.52) | 4.618 (11.71) |

Costs found: verifying the model identity at start takes 84 s; the packed copy adds 428 MB of mapped file; peak
RSS 14.75 GB. Upstream's own repack of the whole 26B needs anonymous memory and pushed the 16 GB Pi into swap.

### What limits the speed (the conclusion that Milestone A proves)

Every good kernel finishes the real 17.84 MB projection in ~1.48 ms once two or more threads run:
17.84 MB / 1.48 ms = **12.0 GB/s, the Pi's measured DRAM read bandwidth** (12.5–14.1 GB/s,
`results/pi-real/membw_pi_2026-10-06.json`). A faster inner loop cannot beat that, which is why assembly, KleidiAI and
the upstream repack all land on the same number, and why full generation scales 2.06 → 3.88 → 4.61 tok/s from 1 to 2
to 4 threads. Decode speed = read bandwidth ÷ weight bytes read per pass: 26B-A4B ≈ 2.6 GB per token → 4.6 tok/s;
E2B ≈ 1.4 GB → ~8 tok/s. **Kernel work is finished as a lever for decode.** What can still move tokens per second:

1. fewer weight bytes per pass (lower-bit weights, a smaller model) — bounded by quality;
2. several committed tokens per pass (speculative decoding with exact verification) — bounded by drafter
   acceptance and, once passes are batched, by the batched evaluation rate (E2B prompt test: 62 tok/s);
3. approximate execution (skipping work) — a research track with a quality cost, nothing measured yet.

100 tok/s would need ~120 MB of weights read per committed token; E2B reads ~1.4 GB per pass, so it needs ~12
committed tokens per pass AND a batched evaluation rate above 100 tok/s. The second number is measured at 62.

## "Try your best" work after the verdict (2026-10-06 15:20–, the operator: "continue, try your bst")

**Pi was rebooted at 15:12** (not by the agent; nothing of ours was running since 13:16). After a reboot: remount the
stick (`sudo mount -o rw,uid=1000,gid=1000,umask=022,noatime /dev/sda1 /mnt/usb`); `/tmp` builds are gone.

New measurements (raw: `results/pi-real/e2b-profile/`, Pi `~/pigemma/results/{e2b-profile,sampler-ab}/`):

- **Per-operation profile of one E2B decode pass** (CF runner in reference mode, 126 ms per token): output head
  (Q6_K, 330 MB) 27.8 ms = 22 %; feed-forward gate/up/down 24.7 + 24.7 + 25.0 ms = 59 %; attention q/output 12 ms;
  per-layer projection 2.3 ms; everything else < 7 ms. **Every one of these runs at ~11.9 GB/s** = memory speed, so
  no single operation is left to speed up inside a one-token pass. `-nr` is slightly faster for one token
  (122.6 ms) and much slower for batches (prefill 46 vs 20.6 ms per token).
- **Sampler chain is not a cost**: default chain vs top-k 1 vs temperature-only give the same tok/s (7.5–7.7 plain,
  14.0–15.5 with MTP) and the same text.
- **Where a multi-token pass loses time** (174 ms with MTP n_max 3 vs 131 ms plain): the drafter has its OWN small
  vocabulary head (`gemma4-assistant.cpp`, about 5 ms per draft step); each extra GROUP of 4 output columns re-reads
  the whole 330 MB head (chunks are 1/16 of a matrix = 20 MB for the head, the token-group loop is outside the
  weight loop); batches of 5–15 tokens run as several 4-token sweeps, only 16+ use the 16-row kernel
  (that is why 12 per pass is slower than 16).
- **E2B layer structure** (GGUF metadata): 35 layers, the upper 20 (`shared_kv_layers = 20`, the ones with the
  12288-wide feed-forward = 73 % of the layer work) have no keys/values of their own. llama.cpp still sends every
  prompt token through them (`src/models/gemma4.cpp`: "TODO @ngxson : strip unused token right after the last KV
  layer to speed up prompt processing").

**In progress: pigemma-fast** = patched COPY of llama.cpp a7b94df (`pi/fast/apply_patches.py`, build script
`pi/fast/build_fast.sh`; Pi: source `/tmp/llama-fast`, binaries `/mnt/usb/pigemma/fast/bin/`; the reference tree
`~/pigemma/llama.cpp` is untouched). All three changes are off unless their environment variable is set:
`PIGEMMA_STRIP_PREFILL=1` (prompt tokens without output skip the 20 shared-KV layers; expected ~3× prompt speed;
not usable together with the MTP drafter, which needs every token's hidden state), `PIGEMMA_PAD_TAIL=1` (2–3
leftover token rows in one padded sweep), `PIGEMMA_CHUNK_KB=<n>` (cache-sized weight chunks for batches > 4).
Compile-checked on x86 in Docker; Pi build 392 s. Patch versions: v1 (15:50) = the three switches; v2 = layer skip
also with the Gemma 4 drafter (it shares the target's KV and only reads the hidden row of output tokens, so the
target can export masked rows); v3 = pad rows quantized once per call (v1 re-quantized them for every weight chunk,
which cost as much as the saved sweep). `apply_patches.py --upgrade <tree>` updates a patched tree in place.

**Round 1 results (v1, 15:58–16:25; raw `results/pi-real/fast/`)** — E2B QAT Q4_0, 4 threads, `llama-bench -p 512`:

| Tokens per pass (prompt test) | 3 | 8 | 16 | 64 | 512 |
|---|---|---|---|---|---|
| reference a7b94df, tok/s | 13.5 | 36.9 | 58.4 | 56.0 | **61.9** |
| patched, `PIGEMMA_CHUNK_KB=64` (+pad v1) | 14.6 | 42.1 | 58.6 | 56.4 | 60.8 |
| patched, + `PIGEMMA_STRIP_PREFILL=1` | – | 49.6 | 85.0 | 129.6 | **170.2** |
| `PIGEMMA_STRIP_PREFILL=1` alone | – | – | 86.7 | 129.9 | **171.1** |

- **Prompt processing 61.9 → 170–171 tok/s (×2.75), exact.** Decode is unchanged (7.50 reference, 7.44 patched).
- llama-server, three long prompts (217–374 tokens, reasoning off, 64 generated tokens): prompt speed 58.6–61.5 →
  156–169 tok/s, request time 12.1–14.9 s → 10.0–11.1 s, **text identical to the reference build in 6 of 6
  comparisons** (default batch and `-b 64`, i.e. prompt batches with no output token at all).
- Cache-sized chunks: +14 % at 8 tokens per pass (two 4-token sweeps), nothing at 16+ (the 16-row kernel is limited
  by arithmetic, about 12.7 multiply-adds per cycle and core). Best value 64 KiB.
- Pad v1 did not help (2 per pass 10.1, 3 per pass 14.6, 6 per pass 21.5 tok/s) — fixed in v3, measured in round 2.

**Round 2 results (v3, 16:30–16:58; raw `results/pi-real/fast/`)** — incremental rebuild 29 s.

Padding fixed (prompt test, `PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=64`, no layer skip):

| Tokens per pass | 2 | 3 | 6 | 7 | 8 | 12 |
|---|---|---|---|---|---|---|
| reference, tok/s | 10.3 | 13.1–13.5 | 24.8 | 24.3 | 36.6–36.9 | 41.9 |
| patched v3, tok/s | 13.5 | 19.7 | 31.2 | 35.8 | 42.2 | 47.8 |
| patched v3, ms per pass | 148 | 153 | 192 | 196 | 190 | 251 |

Server, one stream, greedy (all switches on for "patched"):

| Workload | Reference plain | Reference + MTP n_max 3 | Patched + MTP n_max 3 |
|---|---|---|---|
| Six standard prompts, 128 tokens: decode tok/s | 7.46 | 14.03 | 14.20 |
| … prompt tok/s (28-token prompts, median) | 51.9 | 51.7 | 113 |
| … MTP n_max 2: decode tok/s | – | 7.0 (earlier run) | 12.55 |
| Long prompts (217–374 tokens) + 64 tokens, reasoning off: prompt tok/s | 58.6–61.5 | 58.9–62.1 | **160–168** |
| … decode tok/s | 7.1–7.3 | 17.1–20.5 | 16.9–19.3 |
| … whole request, seconds | 12.1–14.9 | 7.0–9.2 | **4.9–5.5** |

**Same text everywhere**: patched plain, patched MTP n_max 3 and patched MTP n_max 2 give the reference plain text
on 6 of 6 standard prompts; patched MTP + layer skip gives the reference text on 3 of 3 long prompts (27 comparisons
in both rounds, 27 identical). So the layer skip also works with the drafter (v2).

Parallel streams (generated tok/s summed over all streams, 96 tokens each):

| Streams | 1 | 2 | 3 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| reference | 7.2 | 9.0 | 9.9 | 23.1 | 26.4 | 33.6 |
| patched | 7.4 | 12.5 | 17.7 | 23.7 | 32.3 | 37.6 |

**Fast profile (the configuration to use on this Pi):**
`PIGEMMA_STRIP_PREFILL=1 PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=64 /mnt/usb/pigemma/fast/bin/llama-server
-m /mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf -md …/mtp-gemma-4-E2B-it.gguf --spec-type draft-mtp
-t 4 -c 4096 --reasoning-budget 0 --host 127.0.0.1 --port <free>` — ports 18190 and 18191 on the Pi are taken by
another local service (a reference run failed to bind there; rerun on another port).

**44 held-out questions (EN 21 / HE 13 / DE 10; negation, numbers, multilingual, extraction, code), 96-token cap,
reference build plain vs fast profile** (`pi/fast/run_fast_qa44.sh`, `results/pi-real/fast/Q*_*/ab.json`):

| Server flag for thinking | Same text | Reference plain: decode tok/s, total s | Fast profile: decode tok/s, total s | Graded correct |
|---|---|---|---|---|
| `--reasoning-budget 0` | **44 of 44** | 7.41, 595.8 (13.5 s per request) | 16.37, 266.3 (6.1 s per request) | not gradable |
| `--reasoning off` | **44 of 44** | 7.43, 264.1 (median 1.34 s, p90 14.3 s) | 16.83, 113.0 (median 0.74 s, p90 6.25 s) | 16 of 44 both |

- Same text in 88 of 88 comparisons here; with rounds 1–2 the patched build matched the reference text in
  **115 of 115** comparisons (plain and with drafter, three languages, prompts of 20–374 tokens).
- The drafter carries 2.8 tokens per pass on these tasks (2.4 on free text), hence 16–17 tok/s decode.
- **`--reasoning-budget 0` is the wrong switch for a product**: the thinking channel is closed, but the model then
  writes its reasoning into the visible answer and runs into the token cap. **Use `--reasoning off`** (answers
  come directly, e.g. "91", 35 of 44 stop on their own).
- **Quality at this speed is modest**: E2B without thinking gets 16 of 44 on this hard set (EN 7/21, HE 7/13,
  DE 2/10; negation 4/8, numbers 2/10, multilingual 6/8, extraction 3/8, code 1/10 — nine code answers were cut by
  the 96-token cap, so that row understates it). The 26B-A4B scored 34–35 of 36 on the earlier, different set at
  4.6 tok/s. Speed and answer quality pull in opposite directions on this board.
- Binaries on the Pi are v4 since 17:12 (v4 = guard: no layer skip when a caller exports per-layer inputs).

Generation speed outlook (unchanged physics): one pass ≥ 126 ms and ~2.4 accepted tokens per pass on open text →
about 17 tok/s is the most an exact single stream can reach; the patches mainly help prompt processing (time to
first token), several users at once and copy-heavy output.

## Running now (keep current)

- **Pi: `~/pigemma/fast/run_phase_c.sh`** (started 2026-10-07 00:03, log `~/pigemma/results/fast3/phase_c.out`).
  Steps: `attrib` (done), `profile26` (done), `fresh7` (from-scratch build of patch v7 in
  `/mnt/pigemma-fast/llama-fast-v7-fresh` -> `bin-v7-fresh`, then the correctness scenarios), `upstream`
  (`pi/fast/upstream/run_upstream_pi.sh`: the two upstream-style patches on llama.cpp master, three trees under
  `/mnt/pigemma-fast/up/`, results `~/pigemma/results/fast3/upstream-pi/`), `draftlen` (one bounded sweep:
  7 or 15 drafts per pass gated by drafter confidence, results `~/pigemma/results/fast3/draftlen/`).
  A step can be repeated alone: `~/pigemma/fast/run_phase_c.sh upstream`. Phase B finished 00:02 without an abort.
- When phase C is done: copy `fast3/upstream-pi` and `fast3/draftlen` to `results/pi-real/fast3/`, run
  `python pi/fast/analyze_phase_b.py`, fill `PI_MASTER_*` in `pi/fast/upstream/README.md` and
  `PI_MASTER_REPORT` + section 8 in `docs/fast/DECISION_2026-10-06.md`, run `python pi/fast/make_checksums.py`,
  commit on branch `fast-profile`.
- After a reboot of the Pi: `~/pigemma/fast/mount_work.sh` (mounts the stick and the ext4 work image).
  Never overwrite a script on the Pi while it is running (bash reads it incrementally; a build died that way).
- Ports 18190/18191 on the Pi belong to another local service. The 26B at this board's memory limit pushes
  memory into swap (about 0.7 GB still there after the screen); do not run it more often than needed.
- Models for desktop tests: `a local models directory outside this repository\` (E2B, E4B, drafters, drafters with cluster tensors);
  Docker volumes `pigemma-fastcheck-build6` (x86 build of patch v7 + E2B) and `pigemma-upstream-51ce9c1`
  (master, three builds). The scratch clone of llama.cpp master with the three worktrees is in the session
  scratchpad and can be recreated with `pi/fast/upstream/port_to_master.py`.
  `results/.cpu.lock` still names `cf-engine-root` (Codex, 09:36) — stale.

### Verified-ceiling run 12:26–13:15 — FINAL (raw + `summary.json`: `results/pi-real/verified-ceiling/`)

E2B QAT Q4_0, llama.cpp a7b94df, 4 threads, tokens evaluated per second when N tokens go through the model in one pass
(`llama-bench -p 512 -n 0 -b N`). This is the ceiling for a drafter that is free and always right:

| Tokens per pass | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 512 |
|---|---|---|---|---|---|---|---|---|---|
| tok/s | 7.07 | 10.26 | 25.78 | 34.88 | 57.67 | 61.84 | 55.57 | 62.02 | 60.90 |
| ms per pass | 141 | 195 | 155 | 229 | 277 | 517 | 1152 | 2064 | 8407 |

- **The board evaluates E2B at 62 tok/s at most**, however many tokens share a pass (that plateau is arithmetic speed,
  about 16 ms per token, no longer memory speed). Every committed token must be evaluated once, so 62 tok/s is the
  hard upper bound for any exact or verified scheme with these kernels; 100 is above it.
- 2 tokens per pass cost MORE than 4 (195 vs 155 ms). Cause verified in `ggml-cpu/repack.cpp` lines 4637–4647: token
  rows beyond a multiple of 4 are each sent through `gemv`, i.e. one more full pass over the weights per leftover
  token. This is why MTP n_max 3 (1+3 = 4 tokens per pass) works and n_max 6 (7 tokens) fell to 8 tok/s. Batches must
  be padded to a multiple of 4 (or the tail handled in the same pass) — a concrete small-batch kernel fix.
- QAT 2-bit file (UD-Q2_K_XL, 2.02 GiB): decode 10.66 tok/s (×1.39 vs Q4_0, fewer bytes per pass) but only 41 tok/s
  batched (30.1 / 35.7 / 42.0 at 4 / 8 / 16 per pass): the 2-bit kernel is slower per multiplication.

Steps 3–4 (done 12:57; `results/pi-real/verified-ceiling/{e2b_q4_mtp,e2b_q2_mtp}/ab.json`, summary
`summary_mtp.json`, tool `pi/adhoc/analyze_verified_ceiling.py`). Six prompts (EN ×3, HE, DE, code), 128 tokens each,
greedy, chat endpoint, one stream. "Same text" = content + reasoning identical to the run without drafter.

| E2B configuration | Decode tok/s (range) | Tokens per pass | ms per pass | Draft acceptance | Same text |
|---|---|---|---|---|---|
| Q4_0, no drafter | 6.92 (6.8–7.0) | 1.0 | 145 | – | – |
| Q4_0, MTP n_max 3 | **13.72** (10.9–15.8) | 2.42 | 176 | 49 % | 6 of 6 |
| Q4_0, MTP n_max 2 | 7.00 (5.9–7.9) | 2.10 | 300 | 57 % | 6 of 6 |
| Q4_0, MTP n_max 4 | 9.80 (7.7–11.2) | 2.55 | 260 | 41 % | 6 of 6 |
| Q4_0, MTP n_max 3, synthetic 100 % acceptance | 22.17 | 3.88 | 175 | 100 % | n/a |
| Q4_0, MTP n_max 7, synthetic 100 % | 24.96 | 7.53 | 302 | 100 % | n/a |
| Q4_0, MTP n_max 15, synthetic 100 % | 30.79 | 14.22 | 462 | 100 % | n/a |
| 2-bit, no drafter | 11.36 (11.2–11.6) | 1.0 | 88 | – | – |
| 2-bit, MTP n_max 3 | 13.49 (10.9–16.6) | 2.24 | 166 | 43 % | 2 of 6 |

- Best real single stream: **~14 tok/s** (Q4_0 + MTP n_max 3), and on Q4_0 the drafted output is the same text on
  all six prompts for n_max 2, 3 and 4 (this closes the audit point "identity not shown" for text; logits were not
  compared).
- A drafter that is always right gives 22 tok/s at 3 drafts and 31 tok/s at 15 (the drafter itself costs time);
  a free, perfect one would give the 62 tok/s evaluation limit. **No verified scheme reaches 100 on this board.**
- n_max 2 and 4 are slow for the kernel reason above (3 or 5 tokens per pass = extra full weight passes).
- The 2-bit file is the fastest without drafter (11.4) but gains only ×1.19 from MTP (its batched kernel is slow)
  and its drafted text diverges from the plain text on 4 of 6 prompts (batched and single-token arithmetic differ
  in the K-quant kernels, so near-ties flip).
- All 128 tokens of every Q4_0 answer were hidden reasoning with an empty visible answer: these are model tokens,
  not answer tokens. A product needs `--reasoning-budget 0` (or a limit) for the speed to be visible.

Step 5, copy-heavy tasks (repeat a 300-word text, fix typos, rename a variable in 45 lines of Python), E2B Q4_0,
`--reasoning-budget 0`, 336–420 visible tokens each (`e2b_q4_copy/ab.json`). All drafted outputs are the same text
as the plain run (3 of 3 in every configuration).

| Drafter | Decode tok/s: repeat / fix typos / rename | All three | Tokens per pass | Acceptance |
|---|---|---|---|---|
| none | 7.0 / 7.1 / 7.0 | 7.04 | 1.0 | – |
| n-gram (`ngram-mod`, defaults) | 17.4 / 6.5 / 8.0 | 9.20 | 1.76 | 48 % |
| n-gram, short match (n-match 8, n 4–32) | 22.6 / 7.5 / 13.0 | 12.03 | 2.80 | 47 % |
| MTP n_max 3 | 20.3 / 18.6 / 20.8 | **19.85** | 3.67 | 90 % |
| n-gram + MTP | **24.0** / 16.8 / 13.3 | 17.38 | 5.72 | 65 % |

Follow-up, batch widths around the multiple of 4 (`bench_e2b_batch_tail.md`, `llama-bench -p 420 -n 0 -b N`):

| Tokens per pass | 3 | 4 | 5 | 6 | 7 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| tok/s | 13.09 | 27.22 | 24.92 | 24.81 | 24.32 | 36.56 | 41.94 |
| ms per pass | 229 | 147 | 201 | 242 | 288 | 219 | 286 |

Each token beyond a multiple of 4 costs about 45–55 ms (one more pass over the repacked weights) instead of the
~5–18 ms it costs inside a group of 4. Fixing the tail in `repack.cpp` (pad to 4 or one-pass tail) would remove the
cliff for variable-length drafts; it does not raise the best case, because n_max 3 already lands on 4.

### Verdict on 100 tok/s and what is realistic (measured basis above)

- One pass over E2B's weights takes 141–147 ms (memory speed). Up to 4 tokens ride in that pass almost for free;
  beyond that each token costs ~16 ms of arithmetic. The curve tops out at **62 tok/s**, the rate at which four
  Cortex-A76 cores can evaluate E2B at all. 100 committed tokens per second would need every token evaluated in
  10 ms, 1.6× faster than the board's best batched rate, plus a drafter that is right ~16 tokens ahead every time.
- Real drafters: 2.4 tokens per pass on open text (14 tok/s), 3.7 on copy-heavy text (20 tok/s), best single task 24.
- 2-bit weights: 11.4 tok/s plain, no further gain from drafting, drafted text diverges.
- Bigger models are slower in proportion to bytes per pass: 26B-A4B 4.6 tok/s, 31B 0.5 tok/s.
- Not tried, approximate only (quality cost unknown): output-head shortlist (−23 % bytes per pass), skipping
  feed-forward neurons or layers. Even optimistic estimates stay below 30 tok/s single stream.
- What would reach 100 tok/s: other hardware (the desktop CPU already runs 26B at 15–16 tok/s; a GPU/NPU or a
  machine with ~8× the Pi's memory speed and arithmetic rate), or a model about 10× smaller than E2B, which is no
  longer a Gemma 4.

**Realistic targets for this Pi (CPU only):** E2B Q4_0 + MTP n_max 3, reasoning off: 14–20 tok/s visible text,
same output as without drafter. 26B-A4B for quality at 4.6 tok/s. Next engineering steps that still pay, in order:
(1) ship E2B + MTP n_max 3 with reasoning off as the fast profile and measure time to first visible token;
(2) one-pass tail in the batch kernel (removes the 2/4-draft cliff, enables confidence-gated drafts);
(3) only if a quality study is wanted: output-head shortlist as an approximate profile.

## Corrections of earlier evidence (Codex audit, `docs/cf-engine/AUDIT.md`; all accepted)

- ~8 tok/s is **E2B**, not 26B. The old 26B ~3.3 tok/s used UD-Q3_K_M; QAT Q4_0 from the USB stick is 4.6 tok/s.
- 1.38 GB/token was arithmetic over tensor sizes, not a DRAM measurement; with a missed dense projection it is
  ~1.407 GB. The operator timing above is the direct measurement that confirms the bandwidth limit.
- E2B sixteen streams: 1,536 tokens / 46.01 s = 33.4 tok/s in total; per stream 2.09 tok/s on the common clock. The
  saved sample was reasoning text with an empty visible answer, so it is model-token throughput, not answer speed.
- MTP speed-ups were measured without saving full outputs, so "same text as without drafter" was not shown. The run
  in progress stores them.

## Safety and scope (unchanged, binding)

Keep ContextFriend box, GPU tunnel, Hailo and Whisper services unchanged and active. No swap/cache/boot/production
changes, no memory ballast, one Pi performance job at a time. Builds use `/tmp`; large files use `/mnt/usb/pigemma`
(FAT32, mounted by hand, 4 GiB file limit). Root has only ~2.0 GB free — never write large files there. Abort below
1 GB available RAM, at 78 °C, on throttling or an unhealthy service. Pi access:
`ssh -o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1 pi@PI_HOST`.

The Pi memory cgroup controller is disabled (`cgroup_disable=memory`), so no valid 8 GiB budget result exists; the
earlier caching/pinning workflows stay paused (restart needs the operator's explicit go-ahead, multi-agent mode is off).
Open decisions for the operator: enable the memory cgroup (one reboot) or accept desktop emulation; reformat the FAT32 stick to
ext4; Ethernet for the Pi; commit the `cf-engine` branch (nothing is committed).
