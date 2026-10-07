# Real Raspberry Pi 5 — context-friend-1 (new board, 2026-10-05)

the operator replaced the board behind the hostname `context-friend-1` with a new Pi 5 and authorised testing on
it directly. Everything here was measured on that box, without stopping or modifying its running
ContextFriend services (all heavy work at `nice -n 10`, downloads at `ionice -c3`, no mlock, ≥ 2 GB kept free).

## Hardware (probed read-only, 08:42)

| Item | Value |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1 (revision e04171) |
| RAM | **16 GB** (MemTotal 16,607,552 kB; 16.3 GB available at idle) |
| CPU | Cortex-A76 ×4 @ 2.4 GHz; dotprod + fp16, no i8mm / SVE |
| OS | Debian 13 (trixie), kernel 6.18.50+rpt-rpi-2712, **16 KiB pages** |
| Storage | 30 GB microSD only (SD32G); 15.5 GB free; **91.6 MB/s** raw sequential read (O_DIRECT, 4 MiB) |
| PCIe | free (only the RP1 south bridge; no AI HAT, no NVMe) |
| Cooling | Active Cooler (pwm-fan); idle 24–28 °C; `throttled=0x0` |
| Network | Wi-Fi only (5 GHz ch 100, −67…−70 dBm); HF download 8.3 MB/s idle, ~1 MB/s while compiling |
| Services | contextfriend-box, contextfriend-gpu-tunnel, hailort, whisper-bench@base (~0.4 GB RSS) |

## Fit verdict

* **Gemma 4 26B-A4B (MoE)** fits in RAM. The Google QAT Q4_0 file (14.44 GB) does **not** fit on the SD
  card with a safe margin (0.5 GB would remain), so the Pi tests use the full model at 3-bit
  (`unsloth UD-Q3_K_M`, 12.73 GB). A USB 3 SSD or an NVMe HAT (the PCIe lane is free) would allow Q4_0.
* **Gemma 4 31B (dense)** at Q4_0 (17.65 GB) fits neither RAM nor SD. At 3-bit (`unsloth UD-IQ3_XXS`,
  11.84 GB) it fits both, and runs from RAM after the first load.
* Models are tested one at a time (SD space); the 26B is left installed at the end.

## llama.cpp native build (measured)

Commit a7b94df, `-DGGML_NATIVE=ON` → `-mcpu=cortex-a76+crc+crypto`, dotprod ✓, i8mm ✗;
5 targets (cli, completion, server, bench, perplexity) built in **410 s** with `-j 4` at nice 10;
355 MB on disk after deleting object files. Script: `pi/adhoc/build_llama_native.sh`.

## Results

### Gemma 4 31B dense — `unsloth/gemma-4-31B-it-GGUF` UD-IQ3_XXS (11.84 GB, 3.06 bpw), 4 threads, measured 09:14–09:37

| Measurement | Value |
|---|---|
| Download from Hugging Face over Wi-Fi | 1692 s, ~7 MB/s average (SHA-256 verified) |
| Cold load from microSD (`-nr`, file evicted from page cache first) | **138 s** load (11.84 GB ≈ 86 MB/s, the card's speed); 157 s wall incl. 8 tokens |
| Peak RSS | 12.58 GB (all of it file-backed page cache; anonymous memory stayed ~0.44 GB) |
| llama-bench pp64 / tg32, `--repack 0` | **0.63 / 0.53 tok/s** |
| llama-bench pp64 / tg32, `--repack 1` (default) | 0.63 / 0.53 tok/s (IQ3_XXS has no repacked kernel) |
| Chat EN (51-token prompt, 47 generated) | prompt 0.63 tok/s (81 s), generation **0.53 tok/s** (88 s) |
| Chat HE (53-token prompt, 47 generated) | prompt 0.63 tok/s (85 s), generation 0.53 tok/s (89 s) |
| Thermals over 23 min of load | 42 → 72.5 °C peak, `throttled=0x0` throughout, clock 2.4 GHz |
| Disk free during the run | 3.29 GB |

Output quality (greedy, both answers cut at the 48-token cap): English — *"A mixture-of-experts model only
activates a small subset of its parameters for each token. This significantly reduces the number of
floating-point operations required per forward pass. Consequently, it achieves the performance of a large
model…"*; Hebrew — *"הוא קטן וצורך מעט מאוד חשמל. הוא זול ונגיש למטרות למידה וניסויים. הוא מאפשר לבנות
פרויקטים…"* — correct and fluent in both languages.

Reading: the full dense 31B **runs** on a 16 GB Pi 5 entirely from RAM after a 2.3-minute load, at about
half a token per second — roughly 11 GB of weights per token at ~6 GB/s effective. That is usable for
batch jobs, not for conversation. Raw logs: `31b-iq3xxs/`.

### Gemma 4 26B-A4B MoE — `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q3_K_M (12.73 GB), 4 threads, measured 10:46–10:58

Driver: `pi/adhoc/serve_measure.py` (llama-server, per-request storage bytes from /proc/<pid>/io, smaps_rollup).
Command: `nice -n 10 llama-server -m models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf -nr -t 4 -c 4096 -np 1`.

| Measurement | Value |
|---|---|
| Cold load from microSD | **148 s**; read_bytes = 12,728,500,224 (exactly the file, once) |
| Storage reads during inference | **0 bytes, 0 major faults** on every request (cold run and warm runs) |
| Memory after load / after an 821-token context | file pages 12.42 GB, anonymous 0.52 GB → 0.99 GB, swap 0 |
| Prompt processing | **9.1–9.3 tok/s** (also for an 821-token prompt: 89 s to first token) |
| Generation, short prompts EN / HE / DE / code | **3.17–3.27 tok/s** |
| Generation after an 821-token prompt | 2.50 tok/s |
| Thermals | 46 → 72 °C, `throttled=0x0` |
| vs dense 31B on the same Pi | generation ×6.1, prompt processing ×14.7 |

Quality spot-check (greedy): correct, fluent Hebrew answer; correct 3-sentence summary of the long text.
### Gemma 4 26B-A4B — Google QAT Q4_0 (14.44 GB, 5 split parts on the USB stick), `-nr`, 4 threads, measured 13:50–14:00

The reference-quality file (same as the desktop runs). Parts copied over Wi-Fi to the FAT32 USB stick
(`pc/adhoc_copy_split_to_pi.sh`, sha256-verified) and evicted from the page cache before the run.
Command: `nice -n 10 llama-server -m /mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf -nr -t 4 -c 4096 -np 1`.

| Request | Prompt tok/s | Gen tok/s | Storage read in request | File pages resident after |
|---|---|---|---|---|
| en_short (first after load) | 0.95 | 2.36 | 2.42 GB | 12.57 GB |
| he_short | 5.12 | 3.68 | 0.33 GB | 12.96 GB |
| de_short | 9.33 | **4.14** | 0.05 GB | 13.02 GB |
| code_short | 4.63 | 3.38 | 0.41 GB | 13.69 GB |
| long_summary (821 tok) | 10.20 | 2.81 | 0.27 GB | 13.85 GB |

Load 125 s (read 15.67 GB for a 14.44 GB file: pages were evicted and re-read during the load); ~50 MB of the
services' memory went to zram swap; 58 → 69 °C, no throttling. Hebrew answer correct and fluent.

Reading: (1) **Q4_0 is faster than UD-Q3_K_M when resident** (4.14 vs 3.2 tok/s; prompt 9.3–10.2 vs 9.2) — the
Q4_0 dot-product kernels are cheaper on the Cortex-A76 than Q3_K dequantisation, which outweighs the 13 %
larger file. (2) At 14.44 GB the model sits right at the page-cache limit of a 16 GB board that also runs the
ContextFriend services (MemAvailable 15.8 GB minus ~0.5–1 GB of llama anonymous memory): it never became fully
resident, so each request still re-read 0.05–2.4 GB from the 70 MB/s stick. On a dedicated 16 GB Pi (no other
services) it would fit; on this box UD-IQ4_XS (13.6 GB) or Q3_K_M is the safer choice.

#### Quality on the Pi: 36 fixed questions, both 26B files (greedy, 64 tokens, raw `/completion` with a hand-written Gemma turn template)

Items: `data/prompts/qa_set.json` (16 basic: EN/HE facts, arithmetic) + `data/prompts/qa_hard.json` (20 harder:
negation, multi-step numbers, extraction from a passage, multi-constraint instructions, checkable code;
EN/HE/DE). Scored with `pigemma.quality.grade_outputs` (thinking channel stripped). Raw answers: `qa/`.

| File | Strict score | Basic | Hard | Hebrew | German | Mean gen tok/s | Mean prompt tok/s |
|---|---|---|---|---|---|---|---|
| unsloth UD-Q3_K_M (microSD) | 35/36 | 16/16 | 19/20 | 10/10 | 4/4 | 2.80 | 8.41 |
| Google QAT Q4_0 (USB stick) | 34/36 | 16/16 | 18/20 | 10/10 | 3/4 | **3.60** | **9.36** |

Every strict failure is a markup glitch around Gemma 4's thinking channel, not a wrong answer: Q3_K_M answered
`NO` (correct) after a stray `model` prefix; QAT wrote `16,00` and `4` (both correct) without closing
`<|channel>thought`, and once kept generating a fake next user turn. **By content both files score 36/36**, so
this set cannot separate 3-bit from QAT Q4_0 — a KLD measurement against the Q4_0 reference is needed (E2).
Method note: future QA runs should use the server's chat endpoint so the GGUF's own chat template is applied.

#### MTP speculative decoding on the Pi (interleaved A/B: base, MTP, base, MTP; greedy, 96 tokens)

`-md models/mtp-gemma-4-26B-A4B-it-Q4_0.gguf --spec-type draft-mtp` (n_max 3). Both rounds gave identical
numbers (deterministic greedy decoding). Acceptance from the server log; tok/s from the server timings.

| Prompt | Acceptance | Tokens per verification | Base tok/s | MTP tok/s | Change |
|---|---|---|---|---|---|
| code_short | 90 % | 3.65 | 3.18 | **4.25** | **+34 %** |
| en_short | 58 % | 2.74 | 3.18 | 3.10 | −2 % |
| de_short | 36 % | 2.09 | 3.19 | 2.39 | −25 % |
| he_short | 32 % | 1.97 | 3.16 | 2.25 | −29 % |
| mean of the 4 | – | – | 3.18 | 3.00 | −6 % |

Reading: on a mixture-of-experts model each drafted token routes to its own 8 experts, so verifying a batch
of drafts reads more expert weights than a single decode step. MTP therefore only pays when acceptance is
very high (code, ~90 %); for Hebrew and German it costs ~25–30 %. The drafter (ggml-org MTP sidecar) was
paired with the unsloth UD-Q3_K_M (non-QAT) target; acceptance on Hebrew/German may improve with a matched
drafter. (`serve_measure.py` did not capture draft_n from the JSON on this run; the counts above come from
`server.log`.)

#### MTP on the dense 31B on the Pi (UD-IQ3_XXS split parts on the USB stick, greedy, 48 tokens)

Drafter: `mtp-gemma-4-31B-it-Q4_0.gguf` (unsloth, built for the QAT checkpoint) with `--spec-type draft-mtp`.

| Prompt | Base tok/s | MTP tok/s | Drafted / accepted | Acceptance |
|---|---|---|---|---|
| en_short | 0.508 | 0.265 | 75 / 21 | 28 % |
| he_short | 0.509 | 0.248 | 80 / 19 | 24 % |
| de_short | 0.508 | 0.276 | 72 / 22 | 31 % |
| code_short | 0.508 | _see raw log_ | | |

Base prompt processing is 0.62 tok/s, barely above the 0.51 tok/s generation: on the Cortex-A76 the IQ3_XXS
kernels are **compute-bound**, so verifying a batch of drafted tokens costs almost as much per token as
generating them one by one. Speculation can only pay when batched throughput is far above single-token
throughput (26B Q3_K_M: 9.2 vs 3.2; QAT Q4_0: ~10 vs ~4). Here it halves the speed. Acceptance is also low
(24–31 % vs 63 % on the desktop with the QAT target) — the drafter was trained against the QAT checkpoint,
the target here is the non-QAT 3-bit file. Next for the dense model on the Pi: a K-quant / Q4_0-class file that
fits in RAM (e.g. Q3_K_S 13.2 GB) where batched verification is cheap, plus a matched drafter.

#### Memory pressure: same GGUF, part of the RAM pinned by an mlocked balloon (plain llama.cpp + Linux page cache)

The Pi kernel boots with `cgroup_disable=memory`, so a cgroup limit is impossible without a reboot; an
mlocked anonymous balloon (`pi/adhoc/balloon.py`) removes RAM from the page cache instead. 48 greedy tokens.

| Balloon | MemAvailable after balloon | Load | Request | pp tok/s | gen tok/s | Storage read in request | Major faults |
|---|---|---|---|---|---|---|---|
| none (warm) | 15.8 GB | 4 s | en / he | 9.2 / 9.0 | 3.17 / 3.15 | 0 | 0 |
| 4 GiB | 11.6 GB | 68 s | en / he | 1.55 / 2.99 | 2.07 / 2.63 | 1.71 / 0.70 GB | 39,631 / 15,203 |
| 6 GiB | 9.6 GB | 132 s | en / he | 0.81 / 1.09 | 1.01 / 1.55 | 5.08 / 3.59 GB | 52,928 / 36,044 |
| 8 GiB (≈ 8 GB board) | 7.5 GB | 170 s | en / he | 0.51 / 0.73 | 0.55 / 0.79 | 10.80 / 7.54 GB | 83,210 / 62,216 |

Reading: with 7.5 GB of usable RAM (an 8 GB board's situation) the 12.73 GB model still runs, at 0.55–0.79
tok/s, reading ~75–108 MB from the microSD per token (≈ 0.9 s/token at 91.6 MB/s - the run is storage-bound).
That is the baseline any expert cache / pinning method has to beat **at the same memory budget**. Raw data:
`series-26b/` (run.json, samples.csv, server.log per run; `series_26b.out`).
