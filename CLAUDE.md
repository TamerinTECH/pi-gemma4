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
| 1 preserve | DONE: local branch `fast-profile` (nothing pushed); raw results checksummed (`results/pi-real/SHA256SUMS-2026-10-06.txt`, regenerate with `python pi/fast/make_checksums.py`); Pi work area out of `/tmp`: ext4 image `/mnt/usb/pigemma/fast/pigemma-work.ext4` loop-mounted at `/mnt/pigemma-fast` (`pi/fast/mount_work.sh`, once per boot); **from-scratch build of the final patch v7** in a new directory (`llama-fast-v7-fresh` -> `bin-v7-fresh`): sources identical to the measured tree, correctness numbers reproduced (binaries are not byte-identical: the build is not byte-reproducible); reproduction notes `pi/fast/README.md`, `pi/fast/BASE_REVISION`. The `/goal` hook can only be cleared by the operator | `pi/fast/`, `results/pi-real/fastcheck-e2b-v7-fresh/` |
| 2 dependency rule | DONE: `docs/fast/LAYER_SKIP.md` (consumer table, drafter case, guards, scope). Tool `pi/fast/tests/fast_check.cpp` (9 scenarios: logits, KV-state digests, token sequences; options `--flash off`, `--kv-type q8_0`), built in the patched tree AND an unpatched reference copy. Pi, patch v7: **E2B** reference vs switches-off 140/140 bit-identical; vs layer skip 137/140, KV digests 32/32, 0 argmax changes, worst KL 2.9e-4; **E4B** 140/140; 137/140, 32/32, 0, worst KL 3.6e-4. The reference against itself (same tokens, other batch split) reaches KL 5.0e-4 (E2B) and 9.3e-4 (E4B), so the skip stays inside the reference's own variation | `results/pi-real/fastcheck-e2b-v7/`, `fastcheck-e4b-v7/` |
| 3 matched benchmark | DONE (Pi, 9 prompts, 3 alternated repetitions, `--reasoning off`, streaming): reference/plain 131.8 s, patched/plain 120.6 s, reference/drafter 85.1 s, patched/drafter 73.1 s for the series; first visible text 0.66 -> 0.30 s (short prompts), 3.5-6.1 -> 1.3-2.2 s (210-367-token prompts); prompt 56.6 -> 147.4 tok/s; decode 7.45 -> 7.46 plain, 12.60 -> 12.82 with drafter; same text 27/27 in every configuration; no request slower; peak RSS not higher. Ablation: layer skip = the whole prompt gain; padded tail +2.9 % decode at 3 drafts and 7.5 -> 13.2 tok/s at 2 drafts; chunks +1.3 % | `docs/fast/DECISION_2026-10-06.md` sections 3 and 3a, `results/pi-real/fast3/{matched,ablation}/` |
| 4 vocabulary shortcut | DONE: exact drafter cluster head (patch v6/v7, `PIGEMMA_MTP_CENTROIDS=1`, drafter file from `pi/fast/make_centroid_gguf.py`). Pi, 50 requests EN/HE/DE/code, two repetitions: draft construction 21.1 -> 16.0 ms per pass (-25 %), verification 148 ms unchanged, accepted drafts 57.4 -> 55.6 %, tokens per pass 2.659 -> 2.608, decode 15.71 -> 15.88 tok/s (+1.1 %; German -4 %), same text 50/50. Sampling audit (desktop): default verification never uses proposal probabilities, both heads commit identical text (6/6 long, 400/400 short = plain sampling with the same seed); rejection mode: digit histograms vs plain p = 0.08-1.0, heads within 2 counts. Verdict: correct, about 1 %, optional switch only. Approximate target head: deferred, nothing approximated | `docs/fast/DECISION_2026-10-06.md` section 4, `results/pi-real/fast3/centroids/`, `results/desktop-fast/sampling/` |
| 5 model screen | DONE (44 held-out tasks, direct answers, task budgets, nothing cut off): **E2B 23/44** (reference plain 464.9 s, fast profile + assistant 200.7 s, 17.0 tok/s); **E4B 32/44** (668.1 s -> 273.7 s, 3.84 -> 10.37 tok/s, layer skip validated, prompt 28 -> 46 tok/s); **26B-A4B 42/44** (`-nr` plain 929 s cold / ~831 s warm, 3.77 tok/s; + its assistant 699.4 s, 5.08 tok/s, 84 % of drafts accepted). Separately labelled: E2B with thinking 32/44 in 1419 s (24 s to the first visible word; code cut by the 768-token budget). One E2B code answer differed between reference/plain and fast/drafter: attribution run shows the UNPATCHED build with drafter gives the same second text, so patched = reference in the same mode; drafter = plain is upstream behaviour (184 of 186 answers) | `docs/fast/DECISION_2026-10-06.md` sections 5, 5a; `results/pi-real/fast3/{screen,think,attrib}/` |
| 6 26B gate | DONE: **no-go** for prepacking the 26B as a whole and as a plain-decode measure (one-row pass reads 2.38 GB = 5.0 tok/s ceiling; dense parts already at 10.4-11.4 GB/s unpacked; packed expert kernels gave 4.61 -> 4.63 in Milestone A; expert traffic is per row). **Go for one bounded prototype only**: pack the dense 1.58 GB (attention, dense feed-forward, router, output matrix) and use the assistant; E2B representative operations at 4 rows per pass: 110.8 ms packed vs 255.4 ms unpacked; projection (not measured on the 26B) 5.1 -> ~6.7 tok/s; acceptance bar +20 % on the 44-task screen, identical output in the same mode, no more swap than today. The 26B runs put 1.3 GB of the server's memory and ~0.7 GB more into swap; services stayed active | `docs/fast/DECISION_2026-10-06.md` section 6; `results/pi-real/fast3/profile26/`, `results/pi-real/26b_bytes_per_token.json` |
| 7 upstream | DONE, nothing sent: two independent patches against llama.cpp master 51ce9c1 in `pi/fast/upstream/` — `gemma4-shared-kv-layer-skip.patch` (+73/-13, 4 files) and `repack-pad-tail.patch` (+37/-2, one file; third version: one thread prepares the padded block in the operation's work buffer, no shared state) — no environment switches, ASCII, both apply to a clean checkout and build. **Measured on the Pi on master:** layer skip 62.7 -> 171.4 tok/s prompt processing, KV state identical, logits 137/140 bit-identical; padded tail 2/3/7 rows per pass 10.9/13.3/24.6 -> 13.6/20.0/31.4 tok/s (letting every thread prepare its own block loses half of that: measured). Desktop x86: both bit-identical or within the reference's own variation, incl. flash attention off and q8_0 KV cache. Notes with the project's rules (`README.md` there): the operator writes issue/PR text himself, features start with an issue, comment on the open draft PR #29189 first, `Assisted-by:` trailer | `pi/fast/upstream/`, `results/pi-real/fast3/upstream-pi*/`, `results/desktop-fast/upstream-51ce9c1/` |
| 8 report | DONE: `docs/fast/DECISION_2026-10-06.md` (sections 1-8; section 8 is the short version). Extra bounded sweep for the throughput estimate (section 3b): up to 7 or 15 drafts per pass with or without a confidence gate never beats 3 drafts (best 15.74 vs 15.71 tok/s) and changes the greedy text in 12-18 % of answers. Next justified component: selective packing of the 26B's dense tensors for multi-row verification passes, only if the 26B ships; otherwise get the layer skip upstream | `docs/fast/DECISION_2026-10-06.md`, `results/pi-real/fast3/draftlen/` |

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

- **Nothing is running.** Brief 3 is finished (2026-10-07 09:15): report `docs/fast/DECISION_2026-10-06.md`
  (section 8 = decision), `docs/fast/LAYER_SKIP.md`, `pi/fast/README.md`, `pi/fast/upstream/README.md`; all
  committed on the local branch `fast-profile`, nothing pushed. Open decisions are the operator's: clear the `/goal` hook;
  which profile ships (E2B fast / E4B fast / 26B + assistant); whether to open the upstream issue; whether to
  fund the one bounded 26B prototype (dense tensors packed + assistant).
- Pi state: services active, no job of ours running, about 0.7 GB in swap since the 26B screen. Trees under
  `/mnt/pigemma-fast/`: `llama-fast-v6` (= v7, measured), `llama-fast-v7-fresh`, `llama-ref-check`, `up/{ref,ls,pt}`
  (llama.cpp master, clean / layer skip / padded tail). Binaries on the stick: `fast/bin-v7`, `bin-v7-fresh`.
  Logs: `results/pi-real/fast3/phase_b.out`, `phase_c*.out`.
- To repeat a step: `~/pigemma/fast/run_phase_b.sh <steps>`, `~/pigemma/fast/run_phase_c.sh <steps>`,
  `~/pigemma/fast/upstream/run_upstream_pi_pt2.sh <tag>` (re-patches `up/pt` with the current
  `repack-pad-tail.patch`). Analysis: `python pi/fast/analyze_phase_b.py`, `pi/fast/analyze_op_profile.py`,
  `pi/fast/analyze_sampling.py`; checksums: `python pi/fast/make_checksums.py`.
- After a reboot of the Pi: `~/pigemma/fast/mount_work.sh` (mounts the stick and the ext4 work image).
  Never overwrite a script on the Pi while it is running (bash reads it incrementally; a build died that way).
- Ports 18190/18191 on the Pi belong to another local service. The 26B at this board's memory limit pushes
  memory into swap; do not run it more often than needed.
- Models for desktop tests: `a local models directory outside this repository\` (E2B, E4B, drafters, drafters with cluster tensors);
  Docker volumes `pigemma-fastcheck-build6` (x86 build of patch v7 + E2B) and `pigemma-upstream-51ce9c1`
  (master, three builds). The scratch clone of llama.cpp master with the three worktrees was in the session
  scratchpad of 2026-10-06 and can be recreated with `pi/fast/upstream/port_to_master.py`.
  `results/.cpu.lock` still names `cf-engine-root` (Codex, 2026-10-06 09:36) — stale.

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

## Historical decision brief (2026-10-05 evening) — read `docs/design/INSTRUCTIONS_2026-10-05_decision.md` (the operator, verbatim)

Goal now: an **evidence-backed decision** on whether expert pinning/caching improves MoE inference when the
model does NOT fit in RAM, at a matched, honestly accounted memory budget (A plain llama.cpp + Linux cache,
B harness with caching off, C static hot-expert pinning, D size-matched random pinning), plus a bounded
dense-residual feasibility study on the desktop, narrow quality/MTP claims, and a final decision report.
The old experiment workflow `wf_844590e3-095` was STOPPED after E0 (its later stages conflicted with this brief).

**Hard limitation on the real Pi (verified 2026-10-05 18:10):** the kernel boots with `cgroup_disable=memory`
(no `memory.*` cgroup files) and PSI is off (`/proc/pressure` absent); io controller exists but is not
enabled in subtree_control; `/dev/kvm` exists, QEMU not installed. A valid cgroup-v2 memory budget on the Pi
therefore needs the operator's decision: (1) append `cgroup_enable=memory cgroup_memory=1 psi=1` to
`/boot/firmware/cmdline.txt` and reboot once (services restart; reversible), or (2) a KVM guest with a fixed
RAM size (needs `apt install qemu-system-arm` + guest image; no reboot), or (3) accept the desktop
Docker cgroup-v2 emulation (x86 CPU, I/O throttled to the Pi microSD's 91.6 MB/s) as the controlled
comparison and keep the Pi as the resident reference. Balloon runs are preliminary stress tests only and
are no longer allowed on the live device. Until the operator decides, the controlled comparison runs on the desktop.

### Desktop isolation decision (orchestrator, 2026-10-05 19:25) — binding for P5 and later stages

P4 found Docker Desktop's VM (WSL2 kernel 5.15.167.4) runs **cgroup v1**, without blk-throttling (no io.max,
`--device-read-bps` ignored) and without PSI. Verified: FUSE works in containers
(`--device /dev/fuse --cap-add SYS_ADMIN --security-opt apparmor:unconfined`; `fuse` in /proc/filesystems;
no libfuse in the current image). Decision:
- **Do NOT reconfigure WSL / Docker Desktop** (a WSL restart would stop the operator's running containers, e.g.
  `contextfriend-box-stt`). Memory isolation = the cgroup **v1** memory controller of the container
  (`--memory` = memory.limit_in_bytes; it charges page cache to the container, so mapped model pages count).
  Record the v1 equivalents of the v2 fields the operator asked for: memory.limit_in_bytes (≈max), memory.soft_limit
  (≈high), memory.usage_in_bytes (≈current), memory.max_usage_in_bytes (≈peak), memory.stat (cache, rss,
  mapped_file, unevictable, pgmajfault, pgscan/pgsteal if present), memory.failcnt + memory.oom_control
  (≈events); swap: `--memory-swap` = `--memory` and record whether memsw accounting exists. PSI: unavailable —
  say so; use reclaim counters from memory.stat / /proc/vmstat deltas instead. Label everything
  "desktop Docker, cgroup v1 memory controller" — a documented deviation from the requested cgroup v2.
- **Storage-rate emulation** = a rate-limited **FUSE passthrough** (small C program with libfuse3, built in a
  derived image `pi-gemma4-build:fuse` with fuse3/libfuse3-dev; token bucket at 91.6 MB/s = the Pi microSD
  measured rate, optional 70 MB/s = USB stick) running INSIDE the same container/cgroup (its memory counts),
  reading the backing GGUF on the ext4 volume with **O_DIRECT** (aligned) so no second copy sits in the
  backing file's page cache; llama-server mmaps the FUSE-exposed file (kernel page cache of the FUSE file is
  the only cache). Record FUSE options (max_read, max_readahead, kernel_cache) and verify by measurement that
  (a) the rate limit holds for mmap page faults, (b) the backing file is not cached (check memory.stat cache
  growth vs bytes served). If FUSE mmap or O_DIRECT fails, fall back to unthrottled runs and label them.

### Brief UPDATE 2 + coordination rules (orchestrator, 2026-10-05 19:45) — binding for every stage from now on

the operator's second brief: `docs/design/INSTRUCTIONS_2026-10-05_v2.md` (verbatim; wins over the first brief where they
conflict; a summary is appended to `INSTRUCTIONS_2026-10-05_decision.md`). Three SEPARATE tracks:
**A** same-model constrained-memory (exact computation), **B** practical local GenAI on the Pi (model may change),
**C** dense research. Never present success in one track as success in another.

- **Engines (track A):** the controlled harness must take the llama-server binary/build as a parameter so the
  same cell can run: our pinned a7b94df build (reference), one RECENT upstream ggml-org/llama.cpp revision
  (separate worktree under `third_party/`), and HotPin (`LozzKappa/hotpin-llm`, AGPLv3 — isolated under
  `third_party/hotpin-llm`, notices preserved) **against its own unmodified parent revision**. The comparisons
  workflow builds these and records them in `results/comparisons/registry.json` (status per target: locally
  tested / documentation-only / blocked / not applicable). M1/M2 must include every engine whose registry
  status is "runnable" for our exact GGUF; if HotPin cannot run Gemma 4 at its parent revision, record "blocked"
  and do not port it.
- **Desktop CPU lock:** any CPU-heavy job (builds, dense numerics, model runs) creates
  `results/.cpu.lock` (JSON: owner, purpose, start time; remove when done; stale after 6 h). Timing stages
  (M1, M2, quality timing, desktop runtime comparisons) need the lock EXCLUSIVELY and wait (poll every 60 s)
  while it is held; background builds use at most `-j 4`.
- **Pi lock:** every Pi performance run takes `flock -n ~/pigemma/.bench.lock` (skip/wait if busy) and checks
  that no other llama-server/llama-completion/llama-cli/powerinfer process is running. Only one Pi
  performance experiment at a time; resident (no memory pressure) runs only.
- **Order:** memory baseline + pinning comparison first (main decision workflow), dense independently on the
  desktop, then runtime/model comparisons (comparisons workflow; its Pi-side product runs may proceed in
  parallel because they use the Pi, not the desktop CPU).
- **Main results table columns** (final report): model/quant; runtime/revision/method; memory budget + measured
  peak; storage device; workload + language; prefill tok/s; decode tok/s; time to first useful answer;
  complete-task latency; tail stalls; physical read bytes; correctness/task score; run count + variation;
  limitations. Include failed builds, unsupported configs, OOMs, regressions; vendor numbers kept separate.

### (RESOLVED 2026-10-06 08:35: Pi back after a power-cycle) Pi was unreachable 2026-10-05 23:36 → 2026-10-06 08:35

`context-friend-1` (PI_HOST) dropped off the network between 23:36 and 23:46 while X2's quality
diagnostic was running (our own a7b94df build, resident, 71.6 °C, MemAvailable 15.4 GB, all four ContextFriend
services active at 23:35). Since then no ARP/ping/SSH answer (retried until 00:51 and again at 03:25); the name
still resolves to a stale address and the current one; OTHER_LAN_HOST is a DIFFERENT Raspberry Pi (other host key) — never touch it.
Cause unknown (Wi-Fi driver drop under sustained load is documented on this board; a hang or power issue is
also possible). **State of the ContextFriend services is unknown.** Nothing can be done remotely.
- the operator: please check the board (power LED / network; a power-cycle is your call). Tell the agent when it is back.
- When it is back, FIRST: `uptime`; `systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort
  whisper-bench@base`; `pgrep -a '^llama-'` (no leftover server); `~/pigemma/.bench.lock` free; `df -B1 /`
  (only ~75 MB above the 2 GB gate — ik_llama.cpp sources/objects were built in /tmp tmpfs and are gone after a
  reboot; binaries are in `~/pigemma/ik_llama.cpp/build/bin`); remount the USB stick (not persistent):
  `sudo mount -o rw,uid=1000,gid=1000,umask=022,noatime /dev/sda1 /mnt/usb`; then retrieve
  `~/pigemma/x2/results/quality` + `quality.out` and run the two x2 scripts listed in
  `results/comparisons/ik_llama_pi/X2_RESULT.json` -> `pi_reachability.to_do_when_the_pi_is_back`.
- Until then Pi stages (X4 resident bottleneck, X3 product track B) are BLOCKED; desktop stages continue.
- Lesson for future Pi runs: keep sessions short, cool-down pauses between blocks, and consider asking the operator to
  plug in Ethernet before long runs (Wi-Fi was the weak point all along).

### Interruption 2026-10-05 ~21:00 and resume 22:30 (orchestrator) — read before continuing any stage

Both workflows were cut by the previous account's WEEKLY usage limit and resumed at 22:30 under a new login
(same run ids: main `wf_c08039a6-95e`, comparisons `wf_02d183f6-341`; finished stages replay from cache).
Stages that were interrupted mid-way left work on disk — **continue from it, do not redo**:
- **P5 harness (nearly complete):** `pigemma/controlled.py`, `controlled_cell.py`, `controlled_summary.py`,
  `tools/fusethrottle/` (+ image `pi-gemma4-build:fuse`), pinner 1.1.0 (`--allow-empty`), `pi/cgroup_cell.sh`,
  `tests/test_controlled*.py`, `docs/results/P5-controlled-harness.md`, smoke cells under
  `results/controlled/20261005-smoke*`, throttle check `results/controlled/throttle-verify-2026-10-05/`.
  Remaining: verify the last smoke cell, the engine parameter (`results/comparisons/engines.json`), final report.
- **D1 dense (analysis computed, report missing):** `results/dense/{L2,L30,L57}.json`, `d1_summary.json`,
  `d1_tables.md`, `runtime.json`, `preregistration.json`, activation dumps under `traces/`. The numbers already
  point to NEGATIVE (e.g. layer 2: at Q3_K-equal bytes the oracle error is 0.20 vs 0.077 for Q3_K; reaching 1e-2
  needs 97-99 % of the residual). Remaining: write `docs/results/DENSE_FEASIBILITY.md` from the existing files.
- **X0 done**; X1a/X1b/X1c/X2/X4/X3/X5/X6 not started. New stage **X1c** builds ik_llama.cpp for x86 and registers
  `ik-plain` and `ik-prefetch-experts` (an existing exact expert prefetcher = a real track-A baseline).
- **M1 freeze rule:** do not freeze the method list until `results/comparisons/engines.json` has a final status
  (runnable / blocked) for upstream-recent, hotpin (+ its parent) and ik (plain + prefetch-experts); poll the
  progress log up to 2 h, then freeze with what exists and record what was missing. M2 runs every runnable engine
  in the same budget; each patched engine is compared against its OWN unmodified parent.
- Pi facts from X0/P1: only ~2.15 GB free on `/` (literal >= 2e9 gate), no clang/mold/libaio; Pi-side binaries
  for ik_llama.cpp / PowerInfer must be cross-built in the desktop arm64 container (`pi-gemma4-pi-emu:trixie`,
  slow) and copied; all earlier server runs used Gemma-3 `<start_of_turn>` text markers instead of the GGUF's
  Gemma 4 chat template — pick the GGUF template (chat endpoint) and never mix formats inside one comparison.
- D: has ~13 GB free (gate 10 GB): no new multi-GB files on D:; use C: (75 GB free) for downloads/splits.

## Rules (from the operator + safety)

- Respond in English. No git commits/pushes unless the operator asks (repo is `git init`ed, nothing committed).
- Never kill processes you did not start; never stop/start servers or services unless the operator says so.
- **Real Pi (`context-friend-1`) runs the operator's ContextFriend services — never stop/restart/kill them**
  (`contextfriend-box`, `contextfriend-gpu-tunnel`, `hailort`, `whisper-bench@base`). All heavy work `nice -n 10`,
  downloads `ionice -c3`, keep **>= 2 GB free on `/`**, never edit `/boot/firmware/config.txt`, never reboot,
  no `/etc/fstab` edits. mlock only for bounded experiments (balloon/pinner, self-expiring).
- Never prune Docker caches/images/volumes you did not create (they belong to other projects).
- Never print the HF token (`HF_TOKEN_FILE`, key `HUGGING_FACE_TOKEN`; repos are ungated anyway).
- Report MEASURED vs PROJECTED vs ASSUMED separately; never claim speed-ups from runs on a loaded desktop.

## Access & environment cheat-sheet

| What | How |
|---|---|
| Real Pi | `ssh -o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1 pi@PI_HOST` (the name also resolves to a a stale address; Wi-Fi only, drops packets under load — use `nohup` for anything > 1 min) |
| Pi work dir | `~/pigemma` (llama.cpp build in `~/pigemma/llama.cpp/build/bin`, models in `~/pigemma/models`, results in `~/pigemma/results`, ad-hoc tools `*.sh *.py`) |
| Pi USB stick | `/dev/sda1` FAT32 (4 GB file limit → split GGUFs), mounted by hand at `/mnt/usb` (NOT persistent): `sudo mount -o rw,uid=1000,gid=1000,umask=022,noatime /dev/sda1 /mnt/usb` |
| Desktop llama.cpp | Windows prebuilt b11398 in `third_party/llama.cpp-win/` (use for timing); Linux build in `third_party/llama.cpp/build-linux-x86/bin` (Docker) |
| Docker (Linux toolchain) | `MSYS_NO_PATHCONV=1 docker run --rm -v "REPO_ROOT:/work" pi-gemma4-build:latest ...` |
| Emulated Pi (arm64 Debian 13) | image `pi-gemma4-pi-emu:trixie`, volume `pigemma-arm64-llama` (aarch64 llama.cpp build) — functional only |
| Bash heredoc gotcha | if a heredoc fails with "unexpected EOF looking for matching quote", write the file with the Write tool |

## Models

| Where | File | Size | Source checkpoint |
|---|---|---|---|
| Desktop `models/` | `google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf` | 14.44 GB | Google QAT Q4_0 |
| Desktop `models/` | `google__gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf` | 17.65 GB | Google QAT Q4_0 |
| Desktop `models/` | MTP drafters 26B (0.25 GB) / 31B (0.28 GB), DFlash 26B (0.47 GB) | | ggml-org / unsloth |
| Pi microSD `~/pigemma/models` | `gemma-4-26B-A4B-it-UD-Q3_K_M.gguf` + `mtp-gemma-4-26B-A4B-it-Q4_0.gguf` | 12.73 + 0.25 GB | unsloth (non-QAT) |
| Pi USB `/mnt/usb/pigemma/models/26b-qat-q4_0/` | QAT Q4_0 26B as 5 split parts (`-m gemma-4-26B_q4_0-it-00001-of-00005.gguf`) + SHA256SUMS | 14.44 GB | Google QAT (copied + sha256-verified 13:47) |
| (deleted from Pi) | `gemma-4-31B-it-UD-IQ3_XXS.gguf` (11.84 GB, sha256 beb8b6e6…) | | unsloth; re-fetch with `fetch_gguf.sh` |

## Results so far (all MEASURED; details + raw logs in the linked files)

| Setup | Model | Gen tok/s | Prompt tok/s | Notes | File |
|---|---|---|---|---|---|
| Desktop Ryzen 5 8400F, idle | 26B-A4B QAT Q4_0 | **15.2** (4 thr) | 56 | memory-bound, best at 4 threads | `results/desktop-baseline-2026-10-04.md` |
| Desktop, idle | 31B QAT Q4_0 | **2.4** | 12.5 | | same |
| Desktop | MTP (`--spec-type draft-mtp` REQUIRED) | engages: 60–69 % accept | | speed-up NOT measured (desktop was loaded) | same, "Update 2026-10-05" |
| Desktop E0 (bg CPU 12–62 %) | 26B QAT Q4_0 | default **16.31** @4t / 15.55 @6t; `-nr` 14.31 / 14.88 | default 56–70; `-nr` 37–46 | E0 speed band PASS; **repack gate PASS** (`-nr`: 99.9 % CPU_Mapped, no CPU_REPACK, private 1.24 GB); **`-nr` penalty: decode 0.83–0.88× @4t, 0.92–0.96× @6t; prefill 0.66–0.70×** | `docs/results/E0-desktop-baseline-and-repack-gate.md`, `results/2026-10-05/` |
| Desktop E0 | 31B QAT Q4_0 | default **2.33** @6t; `-nr` 2.00 | default 11.8; `-nr` 6.5 | speed PASS; **repack gate FAIL** only on private peak 2.53 GB (= f16 KV 1520 MiB + compute 592 MiB at -c 4096; no weight copy); `-nr` decode 0.86–0.90×, prefill 0.55–0.62× | same |
| Desktop E0 diag | both | `-nr` vs default KLD 0.087 (26B) / 0.015 (31B), top-1 90 % / 94 % | | 26B default at batch 1 vs 512 gives the SAME KLD 0.088 → engine sensitivity, not a `-nr` defect; chat greedy output identical; E2's KLD<0.001 bar will likely fail for both | `results/2026-10-05/e0-diag/` |
| Pi 5 16 GB | 31B UD-IQ3_XXS | **0.53** | 0.63 | 138 s cold load, 72 °C, no throttle; EN+HE correct | `results/pi-real/README.md` |
| Pi 5 16 GB | 26B UD-Q3_K_M | **3.2** | **9.2** | 148 s load; **0 storage reads after load**; 821-tok prompt → 89 s to 1st token | same + `series-26b/` |
| Pi 5 16 GB, USB stick | 26B **QAT Q4_0** (`-nr`) | **3.4–4.1** | 9.3–10.2 | 125 s load; at the page-cache limit with services running → 0.05–2.4 GB re-read per request; faster than Q3_K_M when resident | `results/pi-real/README.md`, `qat/` |
| Desktop, track C dense (D1) | 31B QAT Q4_0 dequantised, FFN layers 2/30/57 | — | — | **NEGATIVE** (pre-registered): low-rank base + oracle residual blocks at Q3_K-equal bytes has FFN error 0.198/0.400/0.290 vs Q3_K 0.077/0.120/0.106 (2.6–3.3× worse); online norm-bound, static and base-alone are worse; stop this factorization | `docs/results/DENSE_FEASIBILITY.md`, `results/dense/` |
| Pi, 36-item QA (EN/HE/DE, hard set) | both 26B files | Q3_K_M 35/36, QAT 34/36 strict; **36/36 by content** | | misses = thinking-channel markup glitches; KLD needed to separate quants | `results/pi-real/README.md`, `qa/` |
| Pi, MTP A/B dense | 31B UD-IQ3_XXS | base 0.51 → MTP 0.25–0.28 (**halves**) | 0.62 | IQ3 kernels compute-bound on A76 (pp ≈ tg) so batched verification saves nothing; acceptance 24–31 % (drafter built for QAT) | `results/pi-real/README.md` |
| Pi, MTP A/B | 26B UD-Q3_K_M | code +34 %, EN −2 %, DE −25 %, HE −29 % | | MoE: speculation pays only at ~90 % acceptance | same |
| Pi, memory balloon 8 GiB (≈ 8 GB board) | 26B UD-Q3_K_M | 0.55–0.79 | 0.5–0.7 | 75–108 MB read/token from microSD = the baseline to beat | same |

Pi hardware: Pi 5 Rev 1.1, 16 GB, Cortex-A76×4, Debian 13, 16 KiB pages, microSD 91.6 MB/s, USB stick 70 MB/s
read / 57 MB/s write, Active Cooler, kernel `cgroup_disable=memory` (no memory cgroup → use `pi/adhoc/balloon.py`).

## Running jobs / how to resume (keep this section current)

1. **Decision workflow — RUNNING** since 2026-10-05 ~18:25: run id `wf_c08039a6-95e`, script
   `docs/design/decision-workflow.js` (original in the session scratchpad; no args). Resume after a stop:
   `Workflow({scriptPath: "<decision-workflow.js>", resumeFromRunId: "wf_c08039a6-95e"})` - finished stages replay
   from cache. If the run id is gone (new session), re-run the script: stages read existing artifacts first.
   Stages (each appends one line to the Progress log below):
   Prepare = P1 baseline audit (`docs/results/BASELINE_AUDIT.md`, `results/baselines/registry.json`) | P2 prior art
   (`docs/research/PRIOR_ART_CACHE.md`) | P3 workloads + harder QA (`data/workloads/{calibration,tuning,heldout}.jsonl`,
   `data/prompts/qa_heldout_v2.json`, `pigemma/quality_v2.py`) | P4 stage UD-Q3_K_M on desktop + Docker volume
   `pigemma-models` + map (`results/maps/map26_ud_q3_k_m.json`) -> P5 controlled-memory harness (Docker cgroup v2,
   io.max throttled to 91.6 MB/s, methods A/B/C/D, `pi/cgroup_cell.sh` for the Pi once enabled);
   Dense = D1 (`docs/results/DENSE_FEASIBILITY.md`), parallel to Prepare;
   Traces = T1 real-generation traces + calibration-only hot ranking (`traces/q3km/`, `results/rankings/`);
   Measure = M1 pilot + FROZEN config (`results/controlled/FROZEN_CONFIG.json`) -> M2 held-out A/B/C/D x >=3
   repeats (`docs/results/CONTROLLED_COMPARISON.md`); Analyze = A1 simulator validation, A2 conditional improvement;
   Decide = Q1 quality v2, R1 decision report (`docs/results/DECISION_2026-10-05.md`) + CLAUDE.md update,
   3 adversarial reviewers, fixes, Z final check.
2. **Comparisons workflow — RUNNING** since 2026-10-05 ~19:55 (brief UPDATE 2): run id `wf_02d183f6-341`, script
   `docs/design/comparisons-workflow.js` (no args). Resume: `Workflow({scriptPath: "<comparisons-workflow.js>",
   resumeFromRunId: "wf_02d183f6-341"})`. Stages: X0 registry + compatibility/licensing
   (`results/comparisons/registry.json`, `docs/research/COMPARISON_TARGETS.md`) -> in parallel: desktop engines
   X1a recent upstream llama.cpp worktree, X1b HotPin vs its parent (`results/comparisons/engines.json`, CPU lock) |
   Pi chain X2 ik_llama.cpp same-file check -> X4 resident bottleneck (`docs/results/RESIDENT_BOTTLENECK_PI.md`) ->
   X3 product track B (`docs/results/PRODUCT_TRACK_B.md`) (Pi lock) -> X5 dense additions (after D1) ->
   X6 `docs/results/COMPARISONS.md`.
3. Older workflows (finished/stopped, for reference): build `wf_b92d4ba6-a59` (done), experiments `wf_844590e3-095`
   (stopped after E0; its plan is superseded).

## Next steps (in order)

- [x] Build workflow DONE (2026-10-05 ~16:10): 36 files, Docker build OK (pigemma-probe, pinner, membw in tools/build-linux-x86/bin), pytest 891 passed / 21 skipped / 0 failed, 11 integration tests pass, maps match all constants. Full report: `docs/design/build-workflow-result.json`. Leftovers (handled in E9): catalog still deny-lists context-friend-1 by name (use PIGEMMA_I_KNOW=1), no real-Pi device profile yet, aarch64 probe link untested, LLAMA_DIR default ~/llama.cpp vs real ~/pigemma/llama.cpp, docs/HYPOTHESES.md §3.3 still says the Pi 26B run is unfinished.
- [x] Copy job finished 13:47 (5 parts, sha256 OK).
- [x] QAT Q4_0 26B cold run from USB on the Pi: 3.4–4.1 tok/s, near the page-cache limit (results/pi-real/qat/).
- [x] Pi QA run done (14:45): Q3_K_M 35/36, QAT 34/36 strict, 36/36 by content. Was: Pi QA run (started 14:05): 36 items (qa_set + qa_hard) greedy on BOTH 26B files → `~/pigemma/results/qa-26b-q3km/`, `qa-26b-qat-q4_0/` (gen-<id>.txt each); log `~/pigemma/qa_both.out`; script `pi/adhoc/run_qa_both.sh`. Score on the desktop with `pigemma.quality` (or the kinds contains/regex/number in the qa JSON files).
- [ ] 31B dense + MTP on the Pi — files on the stick (15:28, sha256 OK); A/B RUNNING since 15:30: `~/pigemma/run_31b_mtp_ab.sh`, log `~/pigemma/ab_31b.out`, results `~/pigemma/results/31b-iq3xxs-{base,mtp}/`. Original plan: `pc/adhoc_31b_to_pi.sh` (started 14:55; downloads unsloth 31B UD-IQ3_XXS to the desktop, splits, deletes the single file, copies 4 parts + `mtp-gemma-4-31B-it-Q4_0.gguf` to `/mnt/usb/pigemma/models/31b-iq3xxs/`; log `results/pi-real/copy_31b_iq3xxs.log`, ends with ALL_31B_DONE). Then on the Pi: `serve_measure.py` base vs `--extra "-md /mnt/usb/pigemma/models/31b-iq3xxs/mtp-gemma-4-31B-it-Q4_0.gguf --spec-type draft-mtp"`, n-predict 48, default 4 prompts.
- [x] Experiment workflow launched (wf_844590e3-095).
- [x] **E0 done 2026-10-05 17:50**: speed PASS both; repack gate 26B PASS, 31B FAIL (private peak = KV cache + compute buffer, not weights); `-nr` decode penalty 12–17 % @4t (26B), prefill 30–45 %. DECISION NEEDED (orchestrator/the operator): the literal gate rule says a FAIL stops E5/E6/Pi streaming — recommendation: proceed on the 26B (its gate passed) with a deviations-log entry; add a batch-size noise-floor control to E2 (its KLD<0.001 bar is below the engine's own b1-vs-b512 difference of 0.088).
- [ ] Final summary for the operator: what runs, measured numbers, hardware recommendation, H1/H2 verdicts.
- Open decisions for the operator: reformat the USB stick to ext4 (only factory SanDisk installer files on it) so
  single-file GGUFs can be downloaded directly; enable the memory cgroup (cmdline.txt + reboot) — not needed so far.

## Progress log (append one line per step: date time — what — result/pointer)

- 2026-10-04 22:3x — project created, llama.cpp a7b94df cloned/built (Docker + Windows prebuilt), both QAT GGUFs + drafters downloaded.
- 2026-10-04 23:xx — desktop baselines measured (see table); research workflow (12 agents) → `docs/research/`.
- 2026-10-05 ~02:00 — design panel (3 designs, 3 judges) → `docs/design/manifest.json` (36 files); build workflow started.
- 2026-10-05 08:30 — emulated Pi image + aarch64 llama.cpp build (1321 s) → `results/pi-emu/`.
- 2026-10-05 08:42 — new Pi 5 16 GB probed; native build 410 s; 31B IQ3_XXS measured 0.53 tok/s.
- 2026-10-05 10:46–11:17 — 26B UD-Q3_K_M series on the Pi: 3.2 tok/s, zero I/O after load; MTP A/B; balloon 4/6/8 GiB.
- 2026-10-05 ~11:30 — MTP found to need explicit `--spec-type draft-mtp` (desktop). Usage limit hit; build workflow stopped at 20/36.
- 2026-10-05 ~12:50 — USB stick (FAT32, 58 GB, 70 MB/s) mounted at /mnt/usb; build workflow resumed (now level 3, 23 entries done);
  QAT Q4_0 26B split into 5 parts, copy to the stick started (~5 MB/s Wi-Fi, ~45 min).
- 2026-10-05 13:10 — this handover file created.
- 2026-10-05 13:25 — copy: part 1/5 done (7 MB/s); part 2 lost to a Wi-Fi reset → copy script made resumable (append), restarted.
- 2026-10-05 13:47 — QAT Q4_0 26B on the Pi USB stick (5 split parts, sha256 OK; 5–7 MB/s over Wi-Fi). 13:50 cold logged run started.
- 2026-10-05 14:00 — QAT Q4_0 26B on the Pi (-nr, from USB): 3.4–4.1 tok/s gen, 9.3–10.2 pp when resident; re-reads 0.05–2.4 GB/request (page-cache limit with services).
- 2026-10-05 14:05 — Pi: 36-item QA run started on both 26B files (Q3_K_M on SD, QAT Q4_0 on USB). Build workflow at level 4 (30/36).
- 2026-10-05 14:50 — Pi QA scored: Q3_K_M 35/36, QAT Q4_0 34/36 strict (all misses are thinking-channel markup glitches; 36/36 by content). Lesson: use the chat endpoint for QA.
- 2026-10-05 14:55 — started 31B UD-IQ3_XXS + 31B MTP transfer to the Pi USB stick for a dense MTP A/B (build workflow still at level 4).
- 2026-10-05 15:30 — 31B UD-IQ3_XXS (4 parts) + 31B MTP on the Pi stick; dense base-vs-MTP A/B started on the Pi.
- 2026-10-05 16:05 — dense MTP A/B on the Pi: MTP halves speed (0.51 → 0.25–0.28 tok/s; acceptance 24–31 %); reason: IQ3_XXS pp≈tg on A76. Next for dense: K-quant/Q4_0-class file that fits RAM.
- 2026-10-05 16:15 — build workflow complete (891 tests pass). Launching the experiment workflow next.
- 2026-10-05 16:20 — experiment workflow launched: wf_844590e3-095 (E0 → E3+E8 → E4+E7 → E9 Pi ∥ E1/E2 → write-up → review → fixes → final check).
- 2026-10-05 17:50 — E0 (desktop baseline + repack gate) done: speed PASS (26B 16.31@4t, 31B 2.33@6t); gate 26B PASS, 31B FAIL (private 2.53 GB = KV+compute, no weight copy); `-nr` decode 0.83–0.88×@4t, prefill ~0.66×; `-nr` KLD 0.087 = same as batch-size change; desktop not idle (bg 12–62 %) → `docs/results/E0-desktop-baseline-and-repack-gate.md`.
- 2026-10-05 18:15 — NEW BRIEF from the operator (decision milestone). Stopped wf_844590e3-095 after E0. Verified the Pi has no memory cgroup / PSI (cgroup_disable=memory) → controlled Pi comparison blocked pending the operator; desktop Docker cgroup-v2 emulation planned meanwhile.
- 2026-10-05 18:25 — decision workflow launched: wf_c08039a6-95e (script docs/design/decision-workflow.js).
- 2026-10-05 18:55 - P2-prior-art - Mira found and verified (arXiv 2609.38090, Yadav & Asgari, 29 Sep 2026: single-GPU HOT+STAGE VRAM cache, i->i+2 predictor, INT8 experts; 5.71x is vs MoE-Infinity at mixed precision, prefetch-only ablation 1.08-1.49x); 12 systems compared; pinning/predicting experts and CPU-only storage streaming are NOT new; matched same-engine exact gains in the literature are ~1.07-1.5x; only possibly-new item is the matched exact A/B/C/D measurement on microSD/USB-class storage - docs/research/PRIOR_ART_CACHE.md
- 2026-10-05 19:09 - P3-workloads-and-quality-sets - DONE: 3 disjoint workload sets (calibration 24 conv / tuning 8 / held-out 19, held-out generation bound 1488 tok, prompts 27-884 tok measured with llama-tokenize vocab-only, explicit KV-reuse protocol incl. the Gemma 4 template limit) + qa_heldout_v2.json (44 objective items, selftest 130/130) + pigemma/quality_v2.py (chat endpoint, thinking stripped, sandboxed code checks) and pigemma/workloads.py; tests 27 passed, no model runs - data/workloads/README.md, data/prompts/qa_heldout_v2.json
- 2026-10-05 19:11 - P4-stage-model-and-maps - DONE: UD-Q3_K_M (sha256 811882f6, unsloth@c099eb48; really IQ3_XXS/IQ4_NL experts + Q8_0 attn, no Q3_K) on desktop + Docker volume pigemma-models (/dev/sdc 8:32, whole disk, sha256 OK) + exact per-(layer,expert) byte/page maps for it and the Pi's 5 split QAT parts (byte-verified vs original); BLOCKER for P5: the Docker VM is cgroup v1 with CONFIG_BLK_DEV_THROTTLING and CONFIG_PSI unset (no io.max, --device-read-bps discarded), and default repack would move 4.1 GB (x86) / 6.8 GB (A76) out of the page cache so all methods need -nr - results/maps/map26_ud_q3_k_m.json, docs/results/P4-model-and-maps.md
- 2026-10-05 19:25 — orchestrator: desktop isolation decision recorded (cgroup v1 memory controller + O_DIRECT rate-limited FUSE passthrough at 91.6 MB/s; no WSL reconfiguration). P2/P3/P4 done; P1 audit + D1 dense running.
- 2026-10-05 19:35 - P1-baseline-audit - 190 results audited into results/baselines/registry.json; 0.55-0.79 tok/s resolved = decode of unsloth UD-Q3_K_M (non-QAT, sha256 811882f6) on the Pi microSD under an 8 GiB mlocked balloon (PRELIMINARY stress test, not baseline A); one resident Pi reference rerun with streaming (decode 2.94-3.01 tok/s, ITL median 0.32 s, cold load 150 s; 6-8 % below the morning series, unexplained -> interleave all comparisons); all server runs used non-canonical Gemma-3 prompt markers; desktop sha256 of the two big QAT GGUFs pending (bench lock held by the dense stage) - docs/results/BASELINE_AUDIT.md
- 2026-10-05 19:45 — the operator's brief UPDATE 2 saved (docs/design/INSTRUCTIONS_2026-10-05_v2.md); coordination rules (engine-swappable harness, CPU lock, Pi lock, track separation) added; comparisons workflow next.
- 2026-10-05 19:55 — comparisons workflow launched: wf_02d183f6-341 (script docs/design/comparisons-workflow.js) for brief UPDATE 2 (registry, upstream + HotPin engines, ik_llama.cpp, resident bottleneck, product track B, dense additions).
- 2026-10-05 20:25 - X0-registry-and-compatibility - 11 targets registered, all documentation-only or not applicable, none blocked; key facts: upstream b11430 (8345f33) has no CPU or MoE change since a7b94df, so the expected delta is ~0; HotPin's parent llama.cpp a6647b1 (b9857) already contains Gemma 4 and its loader matches our GGUF; HotPin is AGPLv3 + commercial and needs -nr; its prefetch layer is Windows-only; its mlock log counts requested bytes, not locked bytes; ik_llama.cpp (Gemma 4 since 86e33fd6) covers our types on NEON and ships an exact --prefetch-experts as an existing baseline; SmallThinker is trained mainly on English, and the vendor Pi build has dotprod off; the small model is Gemma 4 E4B QAT Q4_0 (5.15 GB, needs a split for FAT32; E2B 3.35 GB as fallback); picoLLM needs an AccessKey checked online and has no Gemma 4; the Pi / has only 2.15 GB free, so cross-build in the arm64 container - results/comparisons/registry.json, docs/research/COMPARISON_TARGETS.md, docs/research/PRIOR_ART_CACHE.md (X0 extension)
- 2026-10-05 22:30 — both workflows were cut by the weekly usage limit (~21:00); resumed under a new login with the same run ids; partial P5/D1 state documented above; X1c (ik_llama x86 engine incl. --prefetch-experts) added.
- 2026-10-05 22:27 - P3-workloads-and-quality-sets - RE-VERIFIED after the resume, nothing rebuilt or changed (artifacts of 19:09 intact, sha256 of the 3 sets = token_counts.json): validate OK (calibration 24 conv/32 req, tuning 8/11, held-out 19/26, held-out generation bound 1488 <= 1500, 0 shared turns and 0 shared 6-word runs between sets by an independent check), qa_heldout_v2 44 items selftest 130/130, pytest tests/test_quality_v2.py 16 passed (+ tests/test_workloads.py 11), sandbox timeout and import screen probed; no model runs; note: long prompts are 554-884 tok (held-out 642-884), none reaches 1000 - data/workloads/README.md, data/prompts/qa_heldout_v2.json
- 2026-10-05 22:30 - P4-stage-model-and-maps (replay after resume: re-verified, nothing redone) - UD-Q3_K_M sha256 OK again on D: and in Docker volume pigemma-models (/dev/sdc 8:32, whole disk, cgroup v1, no io.max/PSI); map26_ud_q3_k_m.json re-checked with an independent gguf-py parse + per-page brute force (3,840 experts x 3 tensors, 4 KiB + 16 KiB: 0 mismatches) and the 5-part QAT split map reproduced offline from the original's tensor table (part sizes + 90 expert placements exact); 31 tests pass; D: 13.1 GB free (gate passes, no room for more multi-GB files); two cosmetic fixes (one example sentence, one path label) - results/maps/map26_ud_q3_k_m.json, docs/results/P4-model-and-maps.md
- 2026-10-05 22:35 - P1-baseline-audit - COMPLETED (resumed pass): the two pending desktop hashes are done (26B QAT Q4_0 3eca3b8f..., 31B QAT Q4_0 179cfb99..., both == HF LFS; all 5 desktop GGUFs + the P4 UD-Q3_K_M copy verified); registry (190 results, statuses unchanged 59 reproducible / 101 ambiguous / 3 missing_raw / 13 must_rerun / 9 preliminary / 5 functional_only) gained a machine-readable resolution of the 0.55-0.79 tok/s number (decode of unsloth UD-Q3_K_M non-QAT sha256 811882f6 on the Pi microSD under the 8 GiB mlocked balloon = PRELIMINARY STRESS TEST; the README figure 75-108 MB/token is per prompt+generated token, decode-only is 94/60 MB per generated token) and the reference config desktop_docker_resident_ud_q3_k_m (DEFINED, NOT MEASURED - M1 must measure it in-session); raw cross-check found one reporting omission (31B MTP 0.25-0.28 halves leaves out code_short 0.412 tok/s = -19 %); no second Pi rerun, Pi healthy at 22:27 - docs/results/BASELINE_AUDIT.md, results/baselines/registry.json
- 2026-10-05 22:35 - P2-prior-art (resume re-verification) - existing prior-art file re-read against the sources: Mira confirmed (arXiv 2609.38090, Yadav & Asgari, 29 Sep 2026; GPU DRAM-to-VRAM, INT8 experts, prefetch-only ablation 1.07-1.49x); 6 corrections (PR #25294 does have an mmap comparison, 2.4x decode but with experts moved to the GPU, and its bit-exact claim was later qualified; EdgeMoE gains re-attributed, Pi 4B SD-card 2.55-3.25x vs unquantized on-demand loading; Mira TTFT; HOBBIT storage; SmallThinker baseline + Pi 5 in-memory 6.61 tok/s; our setting = UD-Q3_K_M, cgroup v1 + FUSE, 4 KiB pages on the desktop); 2 systems added (routhjim/llama.cpp-lab PR #9 = mlock'd LRU of expert slabs in the page cache, no A/B run; SSD-LLaMA arXiv 2609.18110 = exact GPU+SSD pipeline, gains from I/O path and layout); conclusion unchanged: pinning/predicting experts, CPU-only storage streaming and mlock of expert pages are NOT new, matched exact policy gains in the literature are ~1.0-1.5x decode, only the matched A/B/C/D measurement on 70-92 MB/s storage is possibly new; X0 extension kept byte-identical - docs/research/PRIOR_ART_CACHE.md
- 2026-10-05 22:58 - D1-dense-residual-feasibility - NEGATIVE (pre-registered rule, stopped after step 4 of 6; reference = 31B QAT Q4_0 dequantised, BF16 not used): at Q3_K-equal bytes (0.764x of the layer's Q4_0 FFN) the low-rank base + oracle-selected 128-wide residual blocks has FFN error 0.198 / 0.400 / 0.290 vs Q3_K 0.077 / 0.120 / 0.106 at layers 2 / 30 / 57 (2.6-3.3x worse; online norm bound 0.208 / 0.408 / 0.318, static 0.281 / 0.463 / 0.314, rank-512 base alone 0.41 / 0.59 / 0.40), needs 0.97-1.16x the Q4_0 bytes to match Q3_K and 97.7-99.9 % of the blocks (1.20-1.22x Q4_0) for 1e-2; reconstruction with all blocks 8.1e-7 (PASS); cause = base too weak for its cost (not a calibration-size effect: 4x more tokens change it by 0.01-0.03) + no structure in 128-blocks, selection not the bottleneck, accumulation not tested; post-hoc exploratory check: per-neuron selection has oracle potential (parity with Q3_K at 0.09 / 0.36 / 0.29x bytes) but the base-predicted selector wins only at layer 2 (0.027 vs 0.077) and loses at 30 / 57 (0.243 / 0.136); recommendation: stop this factorization, no Pi residual streamer; X5 items (base alone, static, online selector) are already measured here - docs/results/DENSE_FEASIBILITY.md (+ results/dense/{L2,L30,L57,d1_summary,robustness,d1_derived}.json)
- 2026-10-05 23:01 - X1a-upstream-llama-recent - DONE: upstream llama.cpp 8345f33 (b11430) built unmodified in Docker (198 s, -j 4, CPU lock) with the pinned Linux build flags, which are now known and verified (GGML_NATIVE=ON, BUILD_SHARED_LIBS=OFF; a control rebuild of a7b94df gave 6/6 byte-identical binaries), and on UD-Q3_K_M with -nr and the harness argv its greedy output is bit-identical to a7b94df (2 tuning prompts x 32 tokens: token ids and top-5 logprobs equal), the only measured differences being a +8.51 MiB compute buffer and 3 more graph nodes; engine llama.cpp-upstream-8345f33 registered as runnable (a drift control, not a source of gains: incidental decode 9.3-10.2 vs 9.4-10.3 tok/s is inside the spread, no speed claim); no controlled cell and no Pi build yet (aarch64 command recorded) - results/comparisons/engines.json, results/comparisons/x1a/README.md, results/comparisons/registry.json, third_party/llama.cpp-upstream-8345f33/build-linux-x86/
- 2026-10-05 23:20 — orchestrator check: D1 dense = NEGATIVE (report written); X1a upstream b11430 bit-identical to a7b94df (drift control); HotPin parent + patched runnable, profiling calibration; P5 running method-B smoke cell. Engines so far in results/comparisons/engines.json: llama.cpp-a7b94df, llama.cpp-upstream-8345f33, hotpin-parent, hotpin.
- 2026-10-06 00:12 - P5-controlled-memory-harness - DONE: controlled-memory harness complete on the desktop (Docker VM re-checked: cgroup v1, no io.max / PSI, so budget = v1 memory limit with swap off and storage = in-cell FUSE throttle at 91.6 MB/s, verified for mmap faults) and run end-to-end in 9 FUNCTIONAL cells at 8 GiB (A x3, B, C, D, a 2-cell randomized `series`, the Pi wrapper's config in direct mode; all ok; token ids identical across A/B/C/D/direct; locked == requested on VmLck + mincore + cgroup; 28/28 prompts equal the server's own Gemma 4 template; KV reuse on follow-ups = common prefix - 1 token); resume added the engine parameter (results/comparisons/engines.json + `engine-check`; upstream 8345f33 accepts every cell flag), the desktop CPU lock, `series`, per-request reclaim / background-CPU / physical-read columns and lock accounting; running method B found a real defect (pinner 1.1.0 exited at once on an empty list, so B was A) -> pinner 1.1.1 holds it; 43 unit tests pass (Windows py3.14 + Linux py3.12); pi/cgroup_cell.sh refuses on the Pi (exit 4, read-only check 22:57, nothing written; Pi / is only 78 MB above the 2 GB gate); NO A/B/C/D comparison was run - the smoke numbers are not a baseline (host CPU 21-33 %, synthetic pin ranking) - docs/results/P5-controlled-harness.md
- 2026-10-06 00:20 - X1b-hotpin - DONE, not blocked: HotPin's parent llama.cpp a6647b1 (b9857) loads our exact UD-Q3_K_M, so `hotpin-parent` (unmodified, 242 s) and `hotpin` (its two core patches applied unmodified, 179 s; AGPL-3.0 + commercial, comparison use only, kept under third_party/hotpin-llm-build-a6647b1 with notices) were built with the pinned-build flags and registered as runnable (harness engine-check ok, no flag override), HotPin's own profiling + make_hot_file.py were run on calibration.jsonl only (32 requests, 6,255 prompt + 1,989 generated tokens, 530 s) and gave pin sets for our file (original 1-6 GB tool budgets, which lock only 0.81-0.82 x the stated GiB, and a budget-matched prefix = active/hotpin.txt, PROVISIONAL until M1 freezes method C: re-match with pc/x1b/hotpin_pinset.py), and with -nr it locks exactly the predicted pages (2 GB set: 426,466 unique 4 KiB pages incl. 625 alignment pages, 0 failed requests, all in the GGUF mapping, 78 % of them read during a 64-token generation) with output bit-identical to its parent; measured caveats: its routing dump ignores the stride of the ffn_moe_topk view, so prompt tokens add a uniform count and the ranking is in effect built from generated tokens only (1,140/1,140 prefill layer-evaluations are a permutation of all experts; a one-hunk local fix + its ranking are kept as a labelled local adaptation), without -nr (its README's launch) 40 % of the locked memory is anonymous repacked copies, with a 64 MiB memlock limit 1,246 of 1,294 requests fail while its log still reports 1,659 MiB, and a6647b1 is not numerically identical to a7b94df (tun-02 diverges at token 8), so hotpin is judged only against hotpin-parent; ARM: builds for aarch64 (cross GCC 13.3, 216 s) and the pin path matches the prediction under qemu-user, real Pi not tested and not a drop-in Pi solution (memlock limit, -nr, no memory cgroup, prefetch layer Windows-only); one FUNCTIONAL smoke cell hotpin:A at 8 GiB ran after P5 finished (cgroup unevictable = predicted 1,496,231,936 B, 0 failed locks, no OOM; no speed claim), no measurement cell and no hotpin-parent cell yet - results/comparisons/x1b/README.md, results/comparisons/engines.json, results/comparisons/x1b/hotfiles/, results/comparisons/x1b/lock_verify/, results/comparisons/registry.json, pc/x1b/, third_party/llama.cpp-a6647b1/build-linux-x86/, third_party/hotpin-llm-build-a6647b1/, results/controlled/20261006-x1b-smoke-tuning-hotpin-A-8g-r0/
- 2026-10-06 00:55 - X2-ik-llama-on-pi - DONE for the comparison, with an open incident: ik_llama.cpp 3d27f5bb built natively on the Pi (833 s, in tmpfs) loads our exact UD-Q3_K_M with the weights still mmapped; same file, resident, frozen settings (ours -nr -t 4; ik -t 4 --swa-compress), 4 held-out prompts x 3 alternated repeats: prefill 1.6x faster on 33-40-token prompts and 2.86x on the 884-token prompt (26.5 vs 9.3 tok/s, first token after 33 s vs 95 s), decode 2-3 % slower on short prompts (3.03 vs 3.11 tok/s) and 15 % faster after the long prompt (2.77 vs 2.42), no storage reads, no stall above 1 s, no throttling; outputs differ between the engines after 5-20 tokens and both Hebrew answers of ik had a malformed word, so quality parity is NOT shown; side finding: our build without -nr is not resident on this board (1.1-2.3 GB re-read per request); INCIDENT: the Pi stopped answering on the network at about 23:36 while a 34-item quality diagnostic was running and was still unreachable at 00:54 (cause unknown, service state unknown) - Pi stages X4/X3 cannot start until it is back; / on the Pi has only about 75 MB above the 2 GB gate - results/comparisons/ik_llama_pi/X2_RESULT.json (+ raw/, build/, driver/), results/comparisons/registry.json (ik entry), pi/adhoc/x2/
- 2026-10-06 03:30 — orchestrator: Pi still unreachable (since 23:36); incident block added above. Usage limit hit again ~01:00 and reset ~03:15: T1, X1c, X4 restarted automatically; done so far: P1-P5, D1 (NEGATIVE), X0, X1a, X1b, X2 (ik prefill 1.6-2.9x faster, decode ~equal, quality parity NOT shown).
- 2026-10-06 09:00 — NEW GOAL (>= 100 tok/s on the Pi). Pi back since 08:35. membw on the Pi: read 12.5-14.1 GB/s -> 100 tok/s needs <= 0.13 GB/token (~0.26B params @4 bit); E2B download + speed-ceiling series started; workflows not resumed (403 + Ultracode off).
- 2026-10-06 09:25 — STOPPED on the operator's request (library/code update). Speed ceiling measured on the Pi: E2B 7.7-7.9 tok/s single stream, 15 tok/s with MTP (x2.0), 33 tok/s aggregate @16 streams, prompt 62 tok/s; 26B 3.3 single, 4.9 aggregate @8; 100 tok/s is outside the Pi 5's memory bandwidth for every Gemma 4. Raw: results/pi-real/speed-ceiling/.
- 2026-10-06 09:27-11:26 — OpenAI Codex session (other agent, transcript a local transcript kept outside this repository): CF-Engine Milestone A on branch cf-engine (no commits): reversible packed format + portable/NEON/assembly kernels inside full 26B-A4B QAT Q4_0 generation on the Pi; first kernels failed the logit/routing gate, fixed to bit-identical; NO speed gain (decode 4.61 ref / 4.63 NEON / 4.62 asm tok/s at 4 threads); assembly rejected; real 17.84 MB expert projection = 1.48 ms on every good kernel = 12 GB/s = DRAM read speed. Session ended on quota before final numbers were written.
- 2026-10-06 12:20-12:45 — Claude: read the Codex transcript + artefacts; closed docs/cf-engine/RESULTS.md ("Final timings") and rewrote docs/cf-engine/STATUS.md (head of this file, synced by cf-engine/update_handover.py). the operator re-set the 100 tok/s goal at 12:20.
- 2026-10-06 12:26-13:15 — Claude: verified-ceiling run on the Pi (results/pi-real/verified-ceiling/, summary.json): E2B evaluates at 62 tok/s at most (batch 16-512); plain 6.9-7.1; MTP n_max 3 = 13.7 tok/s open text (same text 6/6), 19.9 on copy-heavy tasks (best task 24.0 with n-gram + MTP); perfect drafter (synthetic) 22/25/31 tok/s at 3/7/15 drafts; 2-bit QAT 11.4 plain, 13.5 with MTP (text diverges 4/6); llama.cpp repack.cpp re-reads the weights per token beyond a multiple of 4 (batch 3 = 229 ms, 4 = 147 ms). VERDICT: 100 tok/s is not reachable on a Pi 5 CPU for any Gemma 4 with exact or verified decoding. Pi idle and healthy afterwards.
- 2026-10-06 15:20-17:00 — Claude ("continue, try your best"): Pi had been rebooted 15:12 (stick remounted). Per-op profile: every op of a one-token E2B pass runs at ~11.9 GB/s (head 22 %, FFN 59 %); sampler chain costs nothing. Found shared-KV structure (upper 20 of 35 layers) + upstream TODO; built pigemma-fast = patched COPY of llama.cpp a7b94df (pi/fast/apply_patches.py v3; Pi /mnt/usb/pigemma/fast/bin): PIGEMMA_STRIP_PREFILL=1 prompt processing 61.9 -> 170 tok/s (server 59-62 -> 156-169), works with the MTP drafter (v2), same text in 27/27 comparisons; PIGEMMA_PAD_TAIL=1 (v3) 2/3/6/7 tokens per pass 10.3/13.5/24.8/24.3 -> 13.5/19.7/31.2/35.8 tok/s, MTP n_max 2 7.0 -> 12.6; PIGEMMA_CHUNK_KB=64 +14 % at 8 per pass; parallel 2/3/8/16 streams 9.0/9.9/26.4/33.6 -> 12.5/17.7/32.3/37.6; request with 217-374-token prompt + 64 tokens 12.1-14.9 s -> 4.9-5.5 s. Generation itself unchanged (7.5 plain, 14 MTP). Raw: results/pi-real/{e2b-profile,fast}/.
- 2026-10-06 17:00-17:20 — Claude: 44 held-out questions, reference plain vs fast profile (patched v4 + MTP n_max 3): same text 44/44 with --reasoning-budget 0 and 44/44 with --reasoning off (115/115 over all comparisons of the day); decode 7.4 -> 16.4-16.8 tok/s, total 595.8 -> 266.3 s and 264.1 -> 113.0 s; E2B without thinking 16/44 correct (code answers cut by the 96-token cap). --reasoning-budget 0 makes the model reason in the visible answer: use --reasoning off. Ports 18190/18191 on the Pi belong to another local service. Raw: results/pi-real/fast/Q*_*/.
- 2026-10-06 19:40-21:40 — Claude, brief 3 (docs/design/INSTRUCTIONS_2026-10-06_fast_profile.md): branch fast-profile; Pi builds moved from /tmp to an ext4 image on the stick (pi/fast/mount_work.sh), fresh build from pristine source + patch file verified; correctness tool pi/fast/tests (9 scenarios, logits + KV-state digests + token sequences) built against patched AND unpatched trees: patch v7 on the Pi, E2B and E4B: switches off = bit-identical (140/140), layer skip 137/140 with identical KV state and argmax, worst KL 2.9e-4 / 3.6e-4 vs the reference's own batch-split variation 5.0e-4 / 9.3e-4 (docs/fast/LAYER_SKIP.md); drafter cluster head implemented (v6/v7, exact proposals-only shortcut; desktop: draft time -23 %, acceptance -2.7 points, same text); upstream-style patches for master 51ce9c1 in pi/fast/upstream/ (layer skip and padded tail separated, per-thread buffer); phase B (matched four configurations, ablation, cluster head, E2B/E4B/26B screen) started on the Pi 21:35.
- 2026-10-06 21:35-22:37 — Claude, brief 3 phase B on the Pi, first three steps: MATCHED four configurations (reference/plain 131.8 s, patched/plain 120.6, reference/drafter 85.1, patched/drafter 73.1; prompt 56.6 -> 147.4 tok/s; first visible text 2.1-2.8x sooner; decode 7.45 -> 7.46 plain, 12.60 -> 12.82 with drafter; same text 27/27 each; no regression); ABLATION (layer skip = all of the prompt gain; padded tail: 2 drafts per pass 7.5 -> 13.2 tok/s); DRAFTER CLUSTER HEAD (draft time -25 %, acceptance 57.4 -> 55.6 %, decode +1.1 %, same text 50/50; sampling audit on the desktop clean). Report: docs/fast/DECISION_2026-10-06.md sections 1-4. Still running: E2B/E4B/26B screen, thinking run, then phase C (26B profile, fresh v7 build, upstream patches on master on the Pi).
- 2026-10-06 22:37 - 2026-10-07 00:15 — Claude, brief 3 phase B finished without abort + first phase-C steps: MODEL SCREEN on 44 held-out tasks (direct answers, task budgets, nothing cut): E2B 23/44 (464.9 s reference plain -> 200.7 s fast profile + assistant), E4B 32/44 (668.1 -> 273.7 s, 3.84 -> 10.37 tok/s), 26B-A4B 42/44 (-nr plain 3.77 tok/s, + its assistant 5.08 tok/s, 699 s); E2B with thinking 32/44 in 1419 s (separately labelled). ATTRIBUTION of the one differing E2B answer: the unpatched build with the drafter gives the same text, so patched = reference in the same mode. 26B GATE: no-go for prepacking the whole model or for plain decode (2.38 GB per pass = 5.0 tok/s ceiling); go for one bounded prototype (pack the dense 1.58 GB + assistant; E2B representative ops 110.8 vs 255.4 ms per 4-row pass; projection 5.1 -> ~6.7 tok/s). The 26B runs pushed memory into swap (0.7 GB left afterwards), services active. Report sections 5, 5a, 6 written.
- 2026-10-07 00:15-00:47 — Claude, brief 3 phase C on the Pi (ran unattended, PC restarted in the morning): FRESH BUILD of patch v7 from pinned source + patch file in a new directory: sources identical to the measured tree, correctness numbers reproduced; DRAFT-LENGTH SWEEP (7 or 15 drafts per pass, with/without confidence gate): no setting beats 3 drafts (15.74 vs 15.71 tok/s), wider passes change the greedy text in 12-18 % of answers; the upstream-on-master step failed at once on a script bug (tar | head under pipefail) and was restarted 07:03. Report docs/fast/DECISION_2026-10-06.md complete (section 8 = decision: E4B fast profile is the new middle point 32/44 at 10.4 tok/s; 26B prepacking no-go except one bounded dense-only prototype; next component = that prototype only if the 26B ships, else upstream the layer skip).
- 2026-10-07 07:00-09:15 — Claude, brief 3 closed: UPSTREAM-STYLE PATCHES MEASURED ON THE PI on llama.cpp master 51ce9c1 (pi/fast/upstream/): layer skip 62.7 -> 171.4 tok/s prompt processing, KV state identical, logits 137/140 bit-identical; padded tail rewritten twice after measurement (every thread preparing its own padded block loses half of the gain; final version: one thread prepares it in the operation's work buffer, no shared state): 2/3/6/7 rows per pass 10.9/13.2/24.8/24.6 -> 13.6/20.0/27.5/31.3 tok/s, bit-identical on x86 at 1/3/6 threads, largest logit difference 5.7e-6 on the Pi. Report, upstream notes, checksums and handover final; branch fast-profile, nothing pushed. Nothing running; Pi healthy, 0.7 GB in swap since the 26B screen.
