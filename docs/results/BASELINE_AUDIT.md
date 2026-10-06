# P1 — Baseline freeze and audit (2026-10-05)

Stage P1 of the decision brief (`docs/design/INSTRUCTIONS_2026-10-05_decision.md`, section 1).
Machine-readable companion: **`results/baselines/registry.json`** (190 entries, one per reported result,
built by `results/baselines/build_registry.py` from the raw files; rerun it to refresh). Besides `results[]` it holds
`artifacts` (model files: repo, revision, sha256, quant, QAT provenance, locations), `builds`, `hosts`,
`server_defaults`, `reference_configurations` and `resolved` (the 0.55–0.79 tok/s resolution, machine-readable).
The audit ran in two passes: 18:30–19:35 (everything below) and a resumed pass at 22:25–22:35 after the workflow
interruption (the two large desktop hashes, cross-checks against the raw files, facts added by stages P4/P5; §8).
Every number below is labelled **MEASURED**, **RECOVERED** (re-derived from existing raw logs),
**DERIVED** (arithmetic on measured numbers) or **ASSUMED**.

## Verdict in six lines

1. **The 0.55–0.79 tok/s "memory-pressure baseline" is resolved** (section 3): it is the **decode** rate of the
   unsloth **UD-Q3_K_M** 26B (non-QAT, sha256 `811882f6…`, HF rev `c099eb48`) read from the Pi's **microSD**, by
   `llama-server -nr -t 4 -c 4096 -np 1`, prompts `en_short` / `he_short`, **48 generated tokens** (4 of them an empty
   thought channel), greedy, with an **mlocked 8 GiB anonymous balloon** on the 16 GB board. It is a
   **PRELIMINARY STRESS TEST** (global pressure, not a cgroup budget, not an 8 GB device) and must not be used as
   the matched baseline A of the cache comparison.
2. **One resident reference rerun was done on the Pi** (the only allowed one), because no result anywhere had an
   inter-token-latency distribution or a prefill/decode I/O split. Same file, same command, same prompts and 48 tokens:
   decode **2.94–3.01 tok/s**, ITL median **0.32 s**, p99 ≤ **0.43 s**, no stall > 1 s, cold load 150 s (MEASURED).
   Its greedy text is byte-identical to the 8 GiB balloon run and to its own repeat.
3. That rerun is **6–8 % slower in decode** than the morning series (3.15–3.27 tok/s) with the identical file and command.
   **Unexplained** (candidates: `stream=true`, thermal/firmware state — the sticky soft-temperature flag `0x80000`
   appeared after the morning runs —, page-cache neighbours). Consequence: **never compare tok/s across sessions;
   baseline and treatment must be interleaved in the same session.**
4. **Every server/completion run so far used a non-canonical prompt format**: Gemma-3 `<start_of_turn>` markers, which
   are *not* Gemma 4 tokens and are tokenized as plain text (52 prompt tokens instead of 39 for `en_short`). The model
   then emits an empty thought channel (`<|channel>thought\n<channel|>`, 4 generated tokens). The QA "markup glitches"
   (e.g. `Jupiter<end_yt_of_turn>`) are consistent with this. All **llama-cli** rows of the 2026-10-04 desktop table were
   the opposite case: thinking ON via the GGUF template, and **all 200 generated tokens were reasoning tokens**.
5. **Desktop timings are ambiguous in absolute terms**: the 2026-10-04 "idle" sweep did not record background load,
   three of its numbers have no raw log, and all E0 timings ran with 12–62 % background CPU. Ratios from E0's
   interleaved (ABBA) runs remain usable.
6. Pinned identities: all desktop files carry their HF commit at download (`.metadata`) == today's HF head; Pi files
   are sha256-verified (Q3_K_M re-hashed today on the Pi; split parts by SHA256SUMS); **all five desktop GGUFs of the
   audited results were hashed in full and equal the HF LFS sha256** (the two large QAT files in the resumed pass at
   22:27, §2.1). Nothing is pending.

## 1. Scope and sources audited

`CLAUDE.md` results table, `results/desktop-baseline-2026-10-04.md`, `results/pi-real/**` (incl. the 31B MTP A/B and
Pi logs that existed only on the Pi — now copied to `results/pi-real/31b-mtp-ab/` and `results/pi-real/pi-logs/`),
`results/e0-desktop_bench-2026-10-05.log` + the E0 envelopes in `results/2026-10-05/**`, and `docs/results/*`
(E0, DESKTOP_2026-10-05, H2_RESULTS_quick). Registry status counts:

| status | entries | meaning |
|---|---|---|
| reproducible | 59 | exact command, file identity and conditions logged; rerunnable as-is |
| ambiguous | 101 | the number exists, but a configuration field or a condition is missing/uncontrolled (mostly: loaded desktop) |
| missing_raw | 3 | reported number without any saved raw log |
| must_rerun | 13 | cannot support the claim it is used for (MTP never engaged; loaded-desktop MTP speeds) |
| preliminary | 9 | the balloon series (3 loads + 6 requests): stress test only |
| functional_only | 5 | proves something runs (smoke tests, llama-debug routing, emulated build, trace smoke) |

## 2. Pinned versions

### Model files

| id | HF repo @ revision | file(s) | bytes | sha256 | quant (actual) | checkpoint provenance | where |
|---|---|---|---|---|---|---|---|
| `26b-qat-q4_0` | google/gemma-4-26B-A4B-it-qat-q4_0-gguf @ `d1c082be` | gemma-4-26B_q4_0-it.gguf | 14,439,363,584 | `3eca3b8f…ca51d` (desktop full-file hash == HF LFS, §2.1) | Q4_0 (token_embd Q6_K) | **Google QAT** | desktop NVMe |
| `26b-qat-q4_0-split5` | same | 5 parts `-0000N-of-00005` (llama-gguf-split 3500M) | 14,439,364,160 | per part in `/mnt/usb/.../SHA256SUMS` (verified 13:47) | Q4_0 | Google QAT; **different file layout/offsets** from the single file | Pi USB stick FAT32 (70 MB/s) |
| `26b-ud-q3_k_m` | unsloth/gemma-4-26B-A4B-it-GGUF @ `c099eb48` | gemma-4-26B-A4B-it-UD-Q3_K_M.gguf | 12,728,497,888 | **`811882f6df81c8ea…cdfa052`** = HF LFS (MEASURED on the Pi today) | "Q3_K_M" label, but experts are **IQ3_XXS (gate_up) + IQ4_NL (down)**; 1 layer IQ4_XS + Q8_0 | **unsloth, NOT QAT** (BF16 base + unsloth imatrix, 141 chunks) | Pi microSD (91.6 MB/s); since stage P4 also desktop NVMe and Docker volume `pigemma-models` (both full sha256 OK, `results/maps/provenance_ud_q3_k_m.json`) |
| `31b-qat-q4_0` | google/gemma-4-31B-it-qat-q4_0-gguf @ `59dde245` | gemma-4-31B_q4_0-it.gguf | 17,651,001,568 | `179cfb99…2874b` (desktop full-file hash == HF LFS, §2.1) | Q4_0 | Google QAT | desktop NVMe |
| `31b-ud-iq3_xxs` | unsloth/gemma-4-31B-it-GGUF @ `c1ac76e9` | gemma-4-31B-it-UD-IQ3_XXS.gguf | 11,837,782,976 | HF LFS `beb8b6e6…d739b8` (verified by fetch_gguf.sh on the Pi) | IQ3_XXS (3.06 bpw) | unsloth, NOT QAT | Pi microSD (deleted); split4 on USB |
| `mtp-26b-q4_0` | ggml-org/gemma-4-26B-A4B-it-GGUF @ `bb4531cd` | mtp-gemma-4-26B-A4B-it-Q4_0.gguf | 251,939,520 | `e00b0ce3…` (Pi re-hash == HF) | Q4_0, arch gemma4-assistant | drafter for the 26B assistant | desktop + Pi SD |
| `dflash-26b-q8_0` | same repo | dflash-gemma-4-26B-A4B-it-Q8_0.gguf | 472,433,824 | HF LFS `b353b64a…` | Q8_0 | | desktop |
| `mtp-31b-q4_0` | unsloth/gemma-4-31B-it-qat-GGUF @ `43cc1aeb` | MTP/mtp-gemma-4-31B-it-Q4_0.gguf | 279,955,968 | `3a5e99fd…` (Pi re-hash == HF) | "smart Q4_0" (header file_type BF16-mostly) | drafter built for the **QAT** 31B | desktop + Pi USB |

Revision evidence: `models/**/.cache/huggingface/download/*.metadata` line 1 (commit at download) equals
`HfApi().model_info(repo).sha` today for every desktop file; Pi downloads used `resolve/main` on 2026-10-05 and the
repos were last modified 2026-07-17, so main == the listed head. Raw: `results/baselines/hf_revisions_2026-10-05.json`.

**Do not attribute Q3_K_M-vs-QAT differences to bit width**: the two 26B files differ in checkpoint (QAT vs non-QAT),
quantizer (Google vs unsloth imatrix), expert kernels (Q4_0 vs IQ3_XXS/IQ4_NL), chat template, file layout (single vs
5-way split) and storage (SD vs FAT32 USB stick). Note also that `results/map26.json` maps the **QAT single file**;
the UD-Q3_K_M has non-uniform expert sizes (~2.63 MB per expert-layer typical vs 3.345 MB in Q4_0; one layer larger)
and a 2.40 GB non-expert part (vs 1.58 GB). A simulator for the Pi file needs its own map.

### 2.1 Desktop full-file sha256

| file | sha256 (MEASURED on the desktop) | == HF LFS |
|---|---|---|
| mtp-gemma-4-26B-A4B-it-Q4_0.gguf | `e00b0ce326d03d8f6f6972ad2320fe90b6ff25a3da5fc51bec15efe283052aa2` | yes |
| dflash-gemma-4-26B-A4B-it-Q8_0.gguf | `b353b64a95f4a498ce036969df21a15191cc85800722a3e1f62f6110b91f01b2` | yes |
| MTP/mtp-gemma-4-31B-it-Q4_0.gguf | `3a5e99fd8d0b23afb1fccd1ee0c9ebd1f571d00399c2dae2292d217feeec0f6b` | yes |
| gemma-4-26B_q4_0-it.gguf (14,439,363,584 B) | `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` | yes |
| gemma-4-31B_q4_0-it.gguf (17,651,001,568 B) | `179cfb99212709597eae5929112cfca677e1bbf566178b479ae1da0c4772874b` | yes |
| gemma-4-26B-A4B-it-UD-Q3_K_M.gguf (desktop copy, staged by P4) | `811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052` (P4, 18:44; not re-read here) | yes |

Raw: `results/baselines/desktop_sha256.json` (all five files of the audited results, one run, 2026-10-05 22:27,
15.2 s + 18.8 s for the two large files, background CPU 20.9 % before the start — irrelevant for a hash) and
`results/baselines/hash_desktop_models.out`. The first pass (19:18) could hash only the three sidecars because
`results/.bench.lock` was held by the parallel dense stage; its record is kept as `desktop_sha256_partial.json`.
The resumed pass ran `python results/baselines/hash_desktop_models.py` with no other model job running (checked:
no llama process, no workflow container, both locks free), under `results/.bench.lock` and `results/.cpu.lock`
(both released). Consequence: every E0 / 2026-10-04 desktop number is now tied to a full-file digest, not only to the
HF commit + size (the E0 envelopes' `sha_prefix` hashes only the first 16 MiB).

### llama.cpp builds (all commit `a7b94df2c616bc1f62a73b964b4a71cb0dcc488e`, build b11398; no local patches)

| id | used for | build |
|---|---|---|
| `win-b11398` | all desktop timings | official Windows CPU prebuilt (`third_party/llama.cpp-win`), Clang 20, dynamic backend → `ggml-cpu-zen4` on the 8400F, OpenMP, REPACK on by default |
| `linux-docker-x86` | llama-debug, probe, membw (Docker) | `third_party/llama.cpp/build-linux-x86/bin`: configure flags **not recorded** (no CMakeCache kept); `tools/build-linux-x86`: Release, static, `GGML_NATIVE=OFF AVX2=ON AVX512=ON` (no VNNI/BF16) |
| `pi-native-a76` | all Pi runs | `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF`, GNU 14.2.0 → `-mcpu=cortex-a76+crc+crypto+dotprod+noi8mm+nosve+nosme`, OpenMP; `git status` clean on the Pi today |
| `emu-arm64` | functional only | qemu-user arm64, `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16` |

Runtime defaults that the logs do not print (verified in `common/common.h` @ a7b94df): `n_batch 2048`,
`n_ubatch 512`, KV cache **f16/f16**, flash attention **auto**, warm-up decode **on** (the server's load time includes a
warm-up pass that routes to all experts, so with `-nr` it faults in the whole file).

## 3. The 0.55–0.79 tok/s baseline — resolved

Registry ids `pi-26b-q3km-balloon8-load`, `-en_short`, `-he_short`; raw `results/pi-real/series-26b/26b-q3km-balloon8/`.

| field | value |
|---|---|
| model / file | `26b-ud-q3_k_m` = unsloth UD-Q3_K_M, 12,728,497,888 B, sha256 `811882f6…` (non-QAT) |
| storage | Pi microSD (`mmcblk0p2`, 91.6 MB/s sequential O_DIRECT) |
| binary / build | `pi-native-a76` llama-server, a7b94df, no patches |
| complete command | `nice -n 10 /home/pi/pigemma/llama.cpp/build/bin/llama-server -m models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf -nr -t 4 -c 4096 -np 1 --host 127.0.0.1 --port 18080` |
| implicit settings | n_batch 2048, n_ubatch 512, KV f16, flash-attn auto, mmap, **no repack** (`-nr`), 1 slot |
| requests | `POST /completion {prompt, n_predict: 48, temperature: 0, cache_prompt: false}` (non-streaming), via `pi/adhoc/serve_measure.py --balloon-gib 8 --n-predict 48 --prompts prompts_short2.json` |
| prompts | `en_short` (52 prompt tokens) then `he_short` (54), legacy Gemma-3 text markers (`results/pi-real/pi-logs/prompts_short2.json`) |
| generated tokens | 48 each = 4 empty-thought markup tokens + 44 answer tokens; greedy; reasoning effectively off (empty channel) |
| memory condition | `balloon.py 8 3000`: 8 GiB anonymous, **mlocked**, global; MemAvailable 15.87 → **7.54 GB** after the balloon; services running; zram swap 2 GiB on |
| cache state | **partially warm, inherited** from the 6 GiB run before it: Cached 9.3 GB at start, 7.56 GB after the balloon; load read 11.31 GB of the 12.73 GB file |
| load | 170.2 s (partially warm, includes the warm-up pass) |
| **decode (the 0.55–0.79)** | **0.545 tok/s (en), 0.794 tok/s (he)** — server `predicted_per_second`, MEASURED |
| prefill | 0.511 / 0.732 tok/s (MEASURED) |
| total request | 188.0 / 133.0 s client wall (MEASURED) |
| storage reads per request | 10.80 / 7.54 GB process `read_bytes`; 83,210 / 62,216 major faults (MEASURED; no device counter existed then) |
| prefill vs decode reads | en: prefill 6.27 GB, decode 4.53 GB = **94 MB / generated token**; he: prefill 4.64 GB, decode 2.90 GB = **60 MB / token** (RECOVERED from 1 Hz samples, ±1 s ≈ ±0.1 GB) |
| the "75–108 MB read/token" of CLAUDE.md / README | = request `read_bytes` / (**prompt + generated** tokens): 10.796 GB / 100 = 108.0 MB (en), 7.539 GB / 102 = 73.9 MB (he). It mixes prefill and decode I/O; the decode-only figure is the 94 / 60 MB per generated token above (DERIVED / RECOVERED) |
| effective decode read rate | 4.53 GB / 86.2 s = **52.5 MB/s** (en), 49 MB/s (he) — about 57 % of the card's sequential rate (DERIVED) |
| llama-server swap | anonymous pages swapped to zram: smaps `Swap` 336–479 MB during the requests (MEASURED) — swap policy was not controlled |
| output | byte-identical to the resident reference run (same 48 tokens, both prompts): the balloon changed timing, not computation |
| matched resident number | P1 reference, same prompts/48 tokens: 2.98 / 2.98 tok/s decode → the balloon run is **0.18× / 0.27×** of resident (DERIVED, cross-session) |

**What it is not.** Not an 8 GB Pi (an 8 GB board has a different kernel reserve, CMA, zram size and no 8 GiB of
pinned anonymous memory competing in reclaim); not an application budget (a balloon is global pressure: services,
kernel and llama all compete; llama's own anonymous memory was swapped); not a controlled cache state (each balloon run
inherited the page cache of the previous one, and he_short inherits en_short's). It must not be compared with any
QAT Q4_0, USB-stick, desktop or Docker number. It cannot be rerun on the live Pi (balloon now forbidden). Baseline A of
the decisive comparison must be re-measured under a real memory budget (brief §2–3: the controlled harness of stage
P5, then M1/M2), with the file, storage, command and prompts frozen in the same session as the treatments.

The other two balloon levels (same caveats; RECOVERED decode reads per token): 4 GiB → 2.07 / 2.63 tok/s, 8.3 / 3.6 MB
per token; 6 GiB → 1.01 / 1.55 tok/s, 38 / 20 MB per token.

## 4. Known-good resident reference configurations

### 4.1 Pi 16 GB, resident, UD-Q3_K_M (the reference for every Pi comparison)

Why the one allowed rerun was done: no existing result had (a) an inter-token latency distribution, (b) a measured
prefill/decode I/O split, (c) device-level read counters, or (d) a resident run with the balloon's exact workload
(the README's "none" row came from `warm-base-r1` with `n_predict 96`). None of these can be recovered from logs.

Run 2026-10-05 18:51–18:59 (`results/pi-real/p1-resident-ref/`; script `p1_run_resident_ref.sh`; driver
`pi/adhoc/serve_measure.py --cold --stream`): services active before/after, MemAvailable ≥ 15.5 GB throughout,
temperature 48 → 68 °C, `throttled=0x80000` (sticky soft-temperature flag, already set before the run; bit 3 "active"
never seen), `/` free 2.146 GB before and after (≥ 2 GB; the run wrote < 1 MB), advisory lock `~/pigemma/.pi_experiment.lock`.

| request (order) | format | prompt tok | gen tok | prefill tok/s | **decode tok/s** | TTFT s | ITL median / p90 / p99 / max s | request s | reads (prefill / decode) |
|---|---|---|---|---|---|---|---|---|---|
| en_short legacy A (1st after cold load) | legacy | 52 | 48 | 7.83 | **2.981** | 6.65 | 0.322 / 0.369 / 0.408 / 0.408 | 22.4 | 37.2 MB / 5.7 MB |
| he_short legacy A | legacy | 54 | 48 | 8.45 | **2.979** | 6.45 | 0.322 / 0.372 / 0.402 / 0.402 | 22.2 | 12.6 MB / 4.9 MB |
| en_short canonical | canonical | 39 | 48 | 8.45 | **2.971** | 4.63 | 0.327 / 0.360 / 0.428 / 0.428 | 20.4 | 1.7 MB / 0.2 MB |
| he_short canonical | canonical | 41 | 48 | 8.64 | **3.014** | 4.75 | 0.320 / 0.360 / 0.400 / 0.400 | 20.3 | 0.8 MB / 1.1 MB |
| en_short legacy B | legacy | 52 | 48 | 8.99 | **2.988** | 5.80 | 0.325 / 0.364 / 0.399 / 0.399 | 21.5 | 0 / 0 |
| he_short legacy B | legacy | 54 | 48 | 8.91 | **2.940** | 6.11 | 0.328 / 0.368 / 0.388 / 0.388 | 22.1 | 0 / 0 |

All MEASURED. Caveat found in the resumed pass: stage P4 read 19.4 MB of this file on the Pi (header + two expert
slices, read-only `dd`, ~18:54) while this run's cold load was starting, so at most 19.4 MB (0.15 %) of the load may
have been page-cache-warm; the load numbers are not corrected for it. Cold load (file evicted with `posix_fadvise(DONTNEED)` on this file only): **150.1 s**, process
`read_bytes` 12.80 GB, **device** (`/sys/dev/block/179:2/stat`) 12.81 GB, `pgpgin` 12.5 GB — process and device counters
agree within 0.04 %. No stall > 1 s, no swap activity in any request, 0 bytes read in the last two requests.
Legacy A and B outputs are byte-identical (determinism) and equal the balloon-8 outputs. The canonical format removes
the 13 text-marker prompt tokens and the 4 thought-markup output tokens (all 48 generated tokens are answer tokens).

Comparison with the morning series (`26b-q3km-cold`, `warm-base-r1/r2`, same file and server command, non-streaming,
n_predict 96): decode 3.15–3.27 tok/s, prefill 9.0–9.35 tok/s, cold load 148.1 s. Today's reference on the same
(legacy) prompts: decode −6 to −8 % (mean en 2.985 vs 3.211, he 2.960 vs 3.176), prefill −1 to −16 % (first requests
after load lowest). **Unexplained** — not adjusted, not averaged away.

Reproduce:
```bash
# on the Pi (nohup; one experiment at a time; services untouched)
python3 ~/pigemma/serve_measure.py --model models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
  --label <label> --n-predict 48 --prompts <prompts.json> --cold --stream
# prompts: results/pi-real/p1-resident-ref/p1_prompts.json (legacy) - use "format": "canonical" for the GGUF turn markers
```

### 4.2 Desktop, unconstrained (engine reference, not a Pi proxy)

`third_party/llama.cpp-win/llama-bench.exe -m models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf -t 4 -p 128 -n 32 -r 5 -o json`
(default repack; add `--repack 0` for the `-nr` path), plus `llama-server.exe ... -t 6 -c 4096 -np 1` for real
generation. Known values: tg32 @4t **16.31 ± 0.22** (E0, n = 3, background CPU 29.8 %; the E0 argv was
`-t 4,6,12 -p 128 -n 32 -r 3 -o json` — the `-t 4 -r 5` form above is the recommended rerun, not what E0 ran),
15.18 ± 0.42 (2026-10-04, n = 2, "idle" not recorded); server generation @6t 16.03 tok/s (legacy prompt, 200 tokens).
No desktop number was taken on a verifiably idle machine; rerun only if an absolute desktop tok/s becomes
decision-relevant (the Pi decision does not need one). This reference is **synthetic** (llama-bench random tokens)
for the tg/pp values and uses the **QAT Q4_0** file on Windows: it is an engine sanity anchor, not a reference for
the controlled comparison.

### 4.3 Desktop Docker, resident, UD-Q3_K_M (reference for the controlled comparison) — DEFINED, NOT MEASURED

The controlled comparison runs the Pi's file (UD-Q3_K_M, sha256 `811882f6…`) with the Linux a7b94df llama-server
inside a Docker container (cgroup v1 memory controller; CLAUDE.md "Desktop isolation decision"). **No audited result
covers that combination**: the file reached the desktop only in stage P4, and the two P5 smoke cells
(`results/controlled/20261005-smoke*-tuning-A-8g-r0`) are 8 GiB method-A harness checks on a loaded desktop that P5
itself labels "not a baseline". The resident reference is therefore defined here and must be measured by M1 in the
same session as the constrained cells:

`python -m pigemma.controlled cell --method A --budget-gib <B> --workload tuning` with `B` well above file +
anonymous memory (file 11.85 GiB; e.g. 20 GiB inside the ~31 GB VM), all other flags identical to the constrained
cells (`-nr -t 4 -tb 4 -c 4096 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa auto -np 1 --cache-ram 0`, `stream=true`).
Acceptance: no storage reads and no limit hits after load in the measured requests. Without it, a constrained
cell's slowdown cannot be separated from the x86-container engine speed.

## 5. Every reported result, with its configuration and status

Latency component: **L** cold/warm load, **P** prompt processing (prefill), **D** decode, **T** total request.
Workload: **S** synthetic (llama-bench random token ids), **G** real autoregressive generation, **F** teacher-forced
scoring. Reasoning: **∅** empty thought channel emitted (4 markup tokens counted as generated), **R** reasoning tokens
generated, **–** n/a.

| reported result (where) | config | wl / comp / reasoning | cold/warm | status | key issue |
|---|---|---|---|---|---|
| Desktop 26B tg 15.2 @4t (CLAUDE.md, 10-04 md) | win-b11398, QAT Q4_0, repack, `-p 0 -n 32 -t 4,6,8,12 -r 2` | S / D / – | warm | ambiguous | idle not recorded; superseded by E0 16.31 |
| Desktop 26B pp 56 (CLAUDE.md "Prompt tok/s") | same, 6 threads | S / P / – | warm | **missing_raw** | taken at **6** threads during a disk-writing download; no log |
| Desktop 31B tg 2.4 / pp 12.5 | QAT Q4_0, 6t | S / D,P / – | warm | **missing_raw** (tg32, pp128); tg16 2.38 has a log | |
| Desktop llama-cli table (11.4 / 1.7 tok/s, "MTP +24 %") | `llama-cli -st -n 200 --temp 0 -t 6 -c 4096 [-md X]` | G / P,D / **R: all 200 tokens reasoning** | warm | ambiguous (base) / **must_rerun** (MTP rows) | thinking never closed; MTP never engaged without `--spec-type`; prompt length not printed |
| Desktop smoke (8.71 / 2.36 tok/s, load 12.7 / 13.3 s) | llama-completion, `-c 2048 -n 160`, legacy markers, CRLF | G / L,P,D / ∅ | warm | functional_only | md says "160-token decode": actually 74 / 55 tokens (EOS); disturbed by a download |
| Desktop server 16.03 / 2.40 tok/s, DFlash 11.83, MTP "never drafted" | `llama-server -t 6 -c 4096 -np 1`, 200 tok, legacy prompt (58 tok), warm-up request first | G / P,D,T / ∅ | warm | ambiguous; MTP rows **must_rerun** | background load not recorded; MTP not engaged (no `--spec-type`) |
| Desktop MTP engages 60–69 % (mtpx, 10-05) | + `--spec-type draft-mtp`, 96 tok | G / D / ∅ | warm, no warm-up | acceptance valid; speeds **must_rerun** | bg ~25 %; same-session baselines 10.2 / 1.7 tok/s |
| E0 matrix + ABBA (26B 16.31 @4t etc.) | logged argv in 87 envelopes (48 matrix, 32 ABBA, 7 diag) | S (+ G for completion) / P,D,L / – | warm (standby list) | ambiguous (absolute), ratios reproducible | bg 12–62 %; repack runs under Windows memory pressure (~9 GB avail, page-ins) |
| E0 `-nr` penalty 0.83–0.88× @4t etc. | ABBA interleaved | S / D,P | warm | reproducible (derived) | x86 only; ARM ratio unmeasured |
| E0 KLD 0.087 / 0.015 | llama-perplexity `--kl-divergence -c 512 --chunks 3 -t 6` | F | – | reproducible | 3 chunks only |
| E0 load 2.1–30.7 s | llama-completion -lv 4 | L | **warm** (standby list) | ambiguous | not cold-disk loads |
| H2 quick (NO_ADVANTAGE, 0.036 tok/s projected) | pigemma.h2.oracle on 2 layers | offline / PROJECTED | – | reproducible | 2 layers, 128 tokens |
| Pi 31B IQ3_XXS 0.53 / 0.63, load 138 s | bench_pi.sh: llama-bench `-p 64 -n 32 -r 2`, llama-completion `-nr -t 4 -c 2048 -n 48` | S + G / L,P,D / ∅ | cold load, then resident | reproducible | legacy markers |
| Pi 26B UD-Q3_K_M 3.2 / 9.2, load 148 s, 0 reads | serve_measure, `-nr -t 4 -c 4096 -np 1`, n_predict 96 | G / L,P,D,T / ∅ | cold load (fadvise), resident | reproducible | no ITL; legacy markers; −6–8 % in today's rerun |
| Pi QAT Q4_0 USB "3.4–4.1 / 9.3–10.2" | serve_measure, split5 on FAT32 USB | G / L,P,D / ∅ | "cold" (method not logged) | **ambiguous** | range omits en_short 2.36/0.95 and long_summary 2.81; 0.05–2.4 GB re-read per request: not resident |
| Pi QA 35/36 vs 34/36 | 36 items, n_predict 64, legacy markers | G / D,P / ∅ (29 of 36) | warm | ambiguous (smoke) | different checkpoint + storage + layout; template glitches |
| Pi MTP A/B 26B (code +34 %, HE −29 %) | `-md mtp-26b --spec-type draft-mtp`, n_max 3, ABAB | G / D / ∅ | warm | reproducible | drafter vs non-QAT target; legacy markers |
| Pi MTP A/B 31B (0.51 → 0.25–0.28) | split4 on USB, base block then MTP block | G / D / ∅ | warm | reproducible | not interleaved; drafter built for QAT; **the reported range omits code_short: 0.412 tok/s (−19 %, 30 of 48 drafts accepted)** — "halves" holds for en/he/de only |
| **Pi balloon 4/6/8 GiB (0.55–0.79 at 8)** | section 3 | G / L,P,D,T / ∅ | inherited partial | **preliminary** | global pressure, swap uncontrolled, inherited cache |
| **P1 resident reference** | section 4.1 | G / L,P,D,T + ITL / ∅ and canonical | cold load, then resident | reproducible | first run with ITL, phase I/O and device counters |

## 6. Cross-cutting findings

1. **No inter-token latency distribution existed anywhere** before P1 (all requests non-streaming; server timings are
   means). Now available only for the resident reference; every later measurement must stream (`serve_measure.py
   --stream` on the Pi; the P5 harness uses `stream=true` and records raw event times).
2. **Prompt format.** Gemma 4 turn tokens are `<|turn>` (105) / `<turn|>` (106); `<start_of_turn>` is not in the
   vocabulary and costs 7 text tokens each time. The GGUF template with thinking off pre-fills
   `<|channel>thought\n<channel|>`. Never mix the two formats in one comparison. **Decided since** (CLAUDE.md resume note;
   P3 workloads and the P5 harness render with the GGUF's Gemma 4 template): the controlled comparison uses the
   canonical format, so its numbers are not comparable with any legacy-format row of this audit except through the
   four canonical requests of §4.1.
3. **Thinking tokens.** Server/completion runs: empty channel, 4 markup tokens inside every `gen_n`. llama-cli runs:
   100 % reasoning tokens. QA runs: 29/36 empty, 1 non-empty, the rest none/unclosed.
4. **Cold vs warm.** Only three runs were truly cold (31B bench_pi, 26B cold series, P1 reference — all by
   `posix_fadvise(DONTNEED)` on the model file only). The QAT USB "cold" run has no logged method; the balloon runs
   inherited cache state; all desktop loads were warm (Windows standby list).
5. **Counters.** Before P1 only process `read_bytes`/`majflt` were logged. `serve_measure.py` now also logs
   block-device read bytes and `/proc/vmstat` deltas (majfault, pgpgin, pswpin/out, pgscan/pgsteal, workingset refaults)
   per request and per phase. Page-fault counts are not byte counts: in balloon-8 decode, 45,554 faults moved 4.53 GB
   (≈ 99 KB per fault through readahead/fault-around).
6. **Cross-session drift of 6–8 %** on the Pi with identical file and command (section 4.1). Interleave A/B/C/D in
   one session and repeat ≥ 3 times; report per-session ratios.
7. **The QAT-on-USB run is not a resident configuration** on this board (14.44 GB at the page-cache limit with services
   running). It is unsuitable as a resident reference; the UD-Q3_K_M on the SD is the resident reference.
8. **E0's "`-nr` penalty" is x86-only.** No Pi measurement compares repack vs `-nr` for the 26B (the 31B IQ3_XXS has no
   repack kernel, so its equality says nothing).
9. **Pi state now:** `/` free is 2.146 GB (just above the 2 GB rule, already below 2 GiB) — any further Pi work that
   writes more than a few MB must free space first or write to the USB stick. Re-checked 22:27: 2.145 GB free,
   MemAvailable 15.9 GB, 46.6 °C, four services active, no llama process.
10. **Configuration fields that could not be established** (null in the registry, not guessed): the complete command,
    batch and context of the three `missing_raw` numbers (26B pp128, 31B tg32, 31B pp128 of 2026-10-04); the context
    size of the llama-bench rows that did not pass `-c` (10 rows: llama-bench sizes it itself; E0 envelopes do log
    `n_ctx`); `n_ubatch` of four llama-completion rows and batch/KV of the 31B cold-load row (the logs print `n_batch`
    at most; the a7b94df defaults 512 / f16 apply but were not logged); the configure flags of
    `third_party/llama.cpp/build-linux-x86`. Rows that are not model runs (membw, H2 quick, emulated build, the E0
    penalty summary, the llama-debug routing check) carry nulls because the field does not apply.

## 7. What is reproducible, what is ambiguous, what must be rerun

**Reproducible as-is** (exact command, verified file, conditions logged): Pi 31B IQ3_XXS bench/gen; Pi 26B UD-Q3_K_M
cold + warm series; Pi MTP A/B (26B, 31B) as configured; E0 KLD diagnostics; E0 ABBA ratios; membw; H2 quick; the P1
resident reference (the anchor for all Pi comparisons).

**Ambiguous or missing:** inter-token latency for every pre-P1 run (cannot be recovered); desktop absolute tok/s (loaded
machine or unrecorded load); three 2026-10-04 desktop numbers with no raw log (26B pp128, 31B tg32, 31B pp128); prompt
token count of all llama-cli runs; the QAT USB cold method; per-phase I/O of pre-P1 runs (recovered only to ±1 s);
configure flags of `third_party/llama.cpp/build-linux-x86`; the cause of the 6–8 % cross-session drift.

**Must be rerun before it can support a decision:**
- **Baseline A under a real memory budget** (replaces the balloon numbers): controlled harness (P5) in M1/M2, same
  file/storage/command/prompts as the treatments, interleaved, ≥ 3 repeats, streaming.
- **The resident reference in the comparison's own environment** (§4.3): never measured so far.
- MTP speed claims on the desktop (E1, idle, interleaved) — only if MTP is to be recommended; MTP stays off for the
  cache experiments.
- Desktop idle absolute numbers — only if they become decision-relevant.
- Not to be rerun: the balloon series (forbidden on the live Pi; superseded by the controlled comparison).

## 8. Changes made in this stage

- `pi/adhoc/serve_measure.py` (also deployed to the Pi): additive, default path unchanged — per-request block-device
  read bytes and `/proc/vmstat` deltas; `--stream` (TTFT, ITL median/p90/p99/max, stalls, prefill/decode I/O split at
  the first-token boundary, `itl-<id>.json` with raw event times); per-prompt `"format": "canonical"`.
- New: `results/baselines/` (`registry.json`, `build_registry.py`, `pi_phase_io.py`, `hash_desktop_models.py`,
  `hf_revisions_2026-10-05.json`, `pi_sha256.json`, `desktop_sha256_partial.json`), `results/pi-real/31b-mtp-ab/`, `results/pi-real/pi-logs/`,
  `results/pi-real/p1-resident-ref/`.
- Nothing was stopped, restarted or reconfigured on the Pi; no balloon, no drop_caches, no swapoff; one experiment
  (8 minutes) at nice 10; temporary files in `/tmp` removed.

### Resumed pass (2026-10-05 22:25–22:35, after the workflow interruption)

- Desktop: full sha256 of the 26B and 31B QAT Q4_0 GGUFs (both == HF LFS) → `results/baselines/desktop_sha256.json`;
  `build_registry.py` re-run. No model was loaded; nothing else ran on the desktop.
- Registry additions: `resolved` (0.55–0.79 tok/s, machine-readable), reference configuration
  `desktop_docker_resident_ud_q3_k_m` (defined, not measured), the desktop + Docker-volume copies of the UD-Q3_K_M
  (from P4's provenance file), the exact E0 argv behind the desktop known-good value, the P4-overlap caveat on the
  P1 cold load. Entry count and status counts unchanged (190; 59 / 101 / 3 / 13 / 9 / 5).
- Cross-checks against raw files (all agree with this document; one reporting omission found and added to §5 and the
  registry: the 31B MTP summary "0.25–0.28, halves" leaves out code_short at 0.412 tok/s): every `run.json` under `results/pi-real/**`
  (decode / prefill / wall / read bytes / ITL of each request), the balloon-8 and resident outputs (sha256 of
  `gen-en_short*.txt` and `gen-he_short*.txt` equal between balloon-8, legacy A and legacy B), the empty thought
  channel in the raw server outputs, `[Start thinking]` in the llama-cli outputs, the E0 16.31 ± 0.22 envelope
  with its 29.8 % background load.
- HF revisions re-queried at 22:32 (`HfApi().model_info(repo).sha`, no token): unchanged for all 8 repos in
  `results/baselines/hf_revisions_2026-10-05.json`.
- Pi: **no second rerun** (the single allowed resident rerun was used in the first pass and no needed field is
  missing); one read-only health check over SSH at 22:27.

## P1 status

- Registry: 190 entries (`results/baselines/registry.json`), all 313 referenced raw files present.
- Desktop sha256: all five GGUFs of the audited results match HF; the desktop UD-Q3_K_M copy was verified by P4.
- Pi: one resident reference run (8 min, first pass), services healthy before and after and again at 22:27; no other
  Pi experiment overlapped (except P4's 19.4 MB of read-only `dd`, §4.1).
- Open, for later stages (not blockers of this audit): baseline A and the §4.3 resident reference must be measured
  by M1/M2; the 6–8 % Pi cross-session drift is unexplained; three 2026-10-04 desktop numbers have no raw log.
