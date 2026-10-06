# Hypotheses H1 and H2: falsifiable statements, arithmetic and Pi 5 budgets

> **Status (2026-10-05).** This document was written before experiments E3 to E7 ran, so it serves
> as their pre-registration. The thresholds below are fixed now and are not re-tuned after the data
> arrives. the operator's question is whether a *full* Gemma 4 (26B-A4B MoE or 31B dense) can run on a
> Raspberry Pi 5 with good speed and quality. Neither hypothesis promises that it can. Each one says
> what we would have to measure before believing it.

## 0. How to read this file

Every number carries one of four labels:

| Label | Meaning |
|---|---|
| **MEASURED** | Read from a real run on named hardware. The result file is cited. |
| **PROJECTED** | Computed with the formulas in section 3.4 or 5.8. Every input is listed with its own label. |
| **ASSUMED** | A constant that nobody has measured on the target hardware yet. In [catalog.json](../pigemma/data/catalog.json) its `source` string starts with `assumed:`. |
| **PRIOR** | A literature number for a *different* model. It stands in until our own experiment replaces it. |

The constants come from the catalog and nowhere else. Section 8 holds a machine-readable copy that
`tests/test_docs_hypotheses.py` checks against [catalog.json](../pigemma/data/catalog.json). That test
also re-computes the headline projections with [pigemma/feasibility.py](../pigemma/feasibility.py) and
checks that they appear here.

Every Pi row in sections 3 and 4 is the output of `pigemma.feasibility.verdict()` run with these
catalog constants. Byte counts come from the GGUF headers through
[pigemma/gguf_map.py](../pigemma/gguf_map.py) ([results/map26.json](../results/map26.json),
[traces/h2/map31.json](../traces/h2/map31.json)). Some files override the assumptions once they
exist: `results/pi/io_mem.json` (pi/03), `results/pi/calibration.json` (pi/05) and
`results/h1/sim.json` (E4). After that, feasibility and cache_sim stop using the ASSUMED and PRIOR
values below, and this document becomes the record of what we expected.

Experiment ids (E0 to E8) refer to the procedures in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) and
[docs/design/manifest.json](design/manifest.json).

---

## 1. The two hypotheses on one page

| | H1: MoE "small active set" | H2: dense "staged residuals" |
|---|---|---|
| Model | Gemma 4 26B-A4B (128 experts, top-8, 30 MoE layers), QAT Q4_0, 14.44 GB | Gemma 4 31B dense (60 layers), QAT Q4_0, 17.65 GB |
| Idea | Keep the 1.58 GB of non-expert weights and a cache of expert slabs in RAM. Read missed experts from storage on demand. | Keep a cheap base `W_base` of each FFN matrix in RAM. Read a residual group `R_g` only when it can change the output by more than ε. |
| Falsifiable claim | On a Pi 5 with 8 GB and NVMe Gen3, an LRU expert cache of 5.16 GB (40.2% of the experts) gives **tg_serial ≥ 1.0 and tg_overlap ≥ 2.0 tok/s** (E4 rule, section 4.8). That needs **LRU hit ≥ 0.582** on Gemma 4's own routing, and page-cache streaming must match the simulator within ±30% (E5). | At ε = 3% relative FFN-output error on real activations: the oracle needs **≤ 25% of the residual bytes**, the runtime norm bound certifies **≥ 50%** of what the oracle skips, implied traffic is **< 8.5 GB/token**, and the result beats uniform quantization at equal bytes (E7). |
| Decisive experiments | E3 (routing skew), E4 (cache simulation), E5 (page-cache proxy), E6 (pinning), then pi/05 on real hardware | E7 (act-dump, decompose, oracle) |
| Pre-registered expectation | **Plausible.** With the PRIOR hit curve the 8 GB + NVMe Gen3 Pi projects 1.94 tok/s serial and 2.87 overlapped. The weak links are the page-fault I/O path and the ARM kernel efficiency, not routing skew. | **NOT_SUPPORTED.** Section 5.8 shows that the 8.5 GB criterion cannot be met with a q2 base even when no residual is read (12.43 GB/token). A probe on a real 31B matrix (section 5.9) needed 100% of the residual at ε = 3%. |
| Ceiling even if it works | 2.73 tok/s in RAM on a Pi 5 (10.0 GB/s × 0.65 / 2.38 GB), 4.20 at 100% efficiency | Attention plus head alone are 5.93 GB/token, i.e. ≤ 1.69 tok/s even in RAM at 10.0 GB/s |

---

## 2. Bytes per token: the quantity everything depends on

Decode on a CPU is bound by memory bandwidth: each generated token must read every weight it uses
once. All figures below are **MEASURED** from the GGUF headers (gguf-py reader, 2026-10-05; catalog
`breakdown_bytes`, map.json `summary`).

**Gemma 4 26B-A4B, QAT Q4_0** (`26b-q4`, 14,439,363,584 B file):

| Part | Bytes | Read per token |
|---|---:|---|
| Attention (q, k, v, output + norms) | 624,885,760 | always |
| Tied head = `token_embd` (Q6_K), used as the output projection | 605,563,904 | always (the whole matrix for the logits) |
| Dense FFN, i.e. the "shared expert" | 301,086,720 | always |
| Router, norms, expert scales | 45,635,704 | always |
| **Non-expert total** (`nonexpert_bytes_per_token`) | **1,577,172,088** | **always** |
| One expert in one layer (`expert_unit_bytes`) = gate_up 2,230,272 + down 1,115,136 | 3,345,408 | when routed |
| All experts: 30 layers × 128 = 3840 slabs | 12,846,366,720 | — |
| **Worst-case expert traffic** = 30 layers × 8 used × 3,345,408 (`expert_bytes_per_token_worst`) | **802,897,920** | nothing cached |
| **Total per token with nothing cached** | **2,380,070,008** (2.38 GB) | — |

**Gemma 4 31B dense, QAT Q4_0** (`31b-q4`, 17,651,001,568 B file):

| Part | Bytes |
|---|---:|
| FFN = 60 layers × 3 matrices × 65,028,096 | 11,705,057,280 |
| Attention | 4,770,160,640 |
| Tied head (Q6_K) | 1,156,076,544 |
| Norms and other | 3,871,984 |
| **Total per token** (`nonexpert_bytes_per_token`) | **17,635,166,448** (17.64 GB) |

**KV cache, q8_0, at n_ctx 4096** (derived by gguf_map from the header's head counts and key
lengths, `kv_bytes_q8_at_4096` in the map; the sliding layers stop growing at the 1024-token
window):

- 26B: 155,975,680 B. `pigemma.feasibility` uses 160,000,000 B (`KV_Q8_4K["26b"]`, ~160 MB).
- 31B: 623,902,720 B. Feasibility uses 420,000,000 B, an understatement listed as an open issue.

Attention reads the whole KV cache once per token. At a full 4096-token context that adds 0.16 GB
per token for the 26B, which is not in the time model (section 7).

**Lower-bit quants (catalog, MEASURED from HTTP-range header reads).** A smaller file does not
always mean fewer bytes per token. The unsloth UD quants of the 26B keep attention and embeddings at
Q4_K to Q8_0, so their non-expert part is *larger* than in the QAT Q4_0 file:

| id | file bytes | non-expert B/token | worst expert B/token | total B/token | PROJECTED in RAM, 10.0 GB/s × 0.65 | at × 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| 26b-q4 (QAT Q4_0) | 14,439,363,584 | 1,577,172,088 | 802,897,920 | 2,380,070,008 | 2.73 tok/s | 4.20 |
| 26b-ud-iq2_xxs | 9,922,480,608 | 1,667,534,712 | 514,945,024 | 2,182,479,736 | 2.98 | 4.58 |
| 26b-ud-q2_k_xl | 10,546,934,240 | 1,800,337,272 | 545,673,216 | 2,346,010,488 | 2.77 | 4.26 |
| 26b-ud-iq3_xxs | 11,416,548,832 | 2,059,352,952 | 583,835,648 | 2,643,188,600 | 2.46 | 3.78 |
| 26b-ud-q3_k_m | 12,728,497,888 | 2,399,894,648 | 644,548,608 | 3,044,443,256 | 2.13 | 3.28 |
| 26b-ud-iq4_xs | 13,597,177,568 | 2,578,676,856 | 687,667,200 | 3,266,344,056 | 1.99 | 3.06 |
| 31b-q4 (QAT Q4_0) | 17,651,001,568 | 17,635,166,448 | 0 | 17,635,166,448 | 0.37 | 0.57 |
| 31b-ud-iq2_xxs | 8,534,295,488 | 8,518,459,632 | 0 | 8,518,459,632 | 0.76 | 1.17 |
| 31b-ud-q2_k_xl | 11,774,991,296 | 11,759,155,440 | 0 | 11,759,155,440 | 0.55 | 0.85 |
| 31b-ud-iq3_xxs | 11,837,782,976 | 11,821,947,120 | 0 | 11,821,947,120 | 0.55 | 0.85 |
| 31b-q3_k_s | 13,210,856,384 | 13,195,020,528 | 0 | 13,195,020,528 | 0.49 | 0.76 |
| 31b-iq4_xs | 16,372,460,480 | 16,356,624,624 | 0 | 16,356,624,624 | 0.40 | 0.61 |

The table leads to a falsifiable side prediction for the real 16 GB Pi (section 3.3). Its 26B
UD-Q3_K_M run should decode *slower* than the QAT Q4_0 file would (2.13 vs 2.73 tok/s projected),
even though the file is 1.7 GB smaller. That holds unless decode is compute-bound or the `-nr`
penalty on Q4_0 exceeds 22%.

---

## 3. Pi 5 hardware budget

### 3.1 Physics (sources: [docs/research/RESEARCH_SUMMARY.md](research/RESEARCH_SUMMARY.md) section 2; per-claim URLs in `docs/research/research-2026-10-04.json`)

| Item | Published (others measured) | Catalog constant used here (all ASSUMED) |
|---|---|---|
| RAM | 32-bit LPDDR4X-4267, 17.07 GB/s theoretical. About 10 GB/s effective in llama.cpp decode; STREAM triad 5.6 to 12.4 GB/s. | `mem_bw_gbps` = **10.0 GB/s**, multiplied by `llama_efficiency` **0.65** (`feasibility.LLAMA_EFFICIENCY`) |
| microSD | 89 to 92 MB/s large-block read (SDR104) | seq **85 MB/s**, random 3 MB **60 MB/s**, access **2.0 ms** |
| USB 3.0 SSD | 360 to 416 MB/s | seq **380 MB/s**, random 3 MB **350 MB/s**, access **0.5 ms** |
| NVMe Gen2 x1 (default) | 417 to 454 MB/s | seq **430 MB/s**, random 3 MB **400 MB/s**, access **0.2 ms** |
| NVMe Gen3 x1 (`dtparam=pciex1_gen=3`, uncertified) | 784 to 894 MB/s | seq **850 MB/s**, random 3 MB **700 MB/s**, access **0.2 ms** |
| Unknown storage (containers, overlay root) | — | same as microSD: **85 / 60 MB/s**, **2.0 ms** |
| CPU | Cortex-A76 ×4 at 2.4 GHz, Armv8.2-A. **dotprod and fp16: yes. i8mm, SVE, BF16: no.** `GGML_NATIVE=ON` detects this; a native build was 17.8 vs 1.1 tok/s for a generic build on a 1B model. | — |
| Pages | 16 KiB default kernel page (`kernel_2712.img`). This is the minimum fault unit when streaming. Expert slabs are not page-aligned, so the aligned mean slab is 3,378,142 B (+0.98%, map26 `expert_unit_bytes_aligned_mean`). | runtime `getconf PAGESIZE` |
| PCIe | **One** PCIe 2.0 x1 FPC. An AI HAT (Hailo) and an NVMe HAT exclude each other unless a dual-slot HAT is used. | — |
| Thermals | Throttles from 80 °C. A bare board passes 85 °C under 4-core load; the Active Cooler holds 62 to 66 °C. | — |
| AI accelerators | Hailo-8/8L: no DRAM, no LLM support. Hailo-10H (AI HAT+ 2): at most about 2B models, slower than the Pi CPU for decode, no llama.cpp backend. **No help for 25 to 30B models**, and the HAT takes the PCIe lane the NVMe needs. | — |
| OS reserve | — | **0.7 GB** (4 GB board), **0.8 GB** (8 GB), **0.9 GB** (16 GB) |
| llama.cpp compute buffers | — | **0.3 GB** (0.4 GB on the 16 GB profile), at `-c 4096 -b 256 -ub 64` |

The random 3 MB read is the number that matters for H1 (one expert slab is 3.3 MB). The sequential
read is the number for dense streaming.

### 3.2 Desktop calibration (MEASURED, [results/desktop-baseline-2026-10-04.md](../results/desktop-baseline-2026-10-04.md))

Effective bandwidth = measured tok/s × bytes per token:

| Run (Ryzen 5 8400F, DDR5, llama.cpp b11398 Windows CPU build, repack on) | tok/s | × bytes/token | effective traffic |
|---|---:|---:|---:|
| 26B MoE, llama-bench tg32, 4 threads (best) | 15.18 | 2.38 GB | **36.1 GB/s** |
| 26B MoE, llama-bench tg32, 6 threads | 13.52 | 2.38 GB | 32.2 GB/s |
| 26B MoE, llama-server generation, 6 threads | 16.03 | 2.38 GB | 38.2 GB/s |
| 31B dense, llama-bench tg32, 6 threads | 2.44 | 17.64 GB | **43.0 GB/s** |

Readings:

- The MoE gets about 0.84 of the dense model's effective bandwidth (36.1 vs 43.0 GB/s). Each token
  runs 240 small routed matmul pairs (30 layers × 8 experts) plus routing and thread
  synchronisation, and the MoE peaks at 4 threads.
- `tools/membw` in Docker reported a read plateau of about 38 to 40 GB/s (implementer's self-test,
  2026-10-05; no results file yet). The dense model's 43 GB/s exceeds that, so the in-Docker figure
  is a lower bound for native bandwidth. E0 records both.
- The catalog's desktop `mem_bw_gbps` is **60.0 GB/s** (ASSUMED, dual-channel DDR5). Against it the
  desktop efficiency is 0.60 (MoE) and 0.72 (dense). The Pi's ASSUMED 0.65 falls inside this range.
  pi/05 replaces it with `results/pi/calibration.json`.
- The desktop profile's other catalog constants are also ASSUMED: storage 3000 MB/s sequential,
  2000 MB/s random 3 MB, 0.1 ms access (generic NVMe; drive model not recorded), and an OS
  reserve of 2 GB for Windows 11 plus the Docker Desktop VM. They feed desktop projections only, never a Pi verdict.
- All desktop runs above used the default repack. The `-nr` (no repack) penalty that every Pi
  streaming or FITS_MAPPED mode pays was MEASURED by E0 on 2026-10-05 on a non-idle desktop
  (background CPU 12–62%). Decode tg(norepack)/tg(default): 26B 0.83–0.88 at 4 threads and 0.92–0.96
  at 6; 31B 0.86–0.90 at 6. Prefill pp128: 0.55–0.70. Decode is bandwidth-bound, so the slower
  kernels mostly hit prefill. The ARM ratio is still unmeasured (pi/05). See
  [docs/results/E0-desktop-baseline-and-repack-gate.md](results/E0-desktop-baseline-and-repack-gate.md).

### 3.3 Real Pi calibration point (MEASURED, [results/pi-real/README.md](../results/pi-real/README.md))

The real target is context-friend-1: a Pi 5 with **16 GB** (MemTotal 16,607,552 kB = 17.01 GB),
microSD only (91.6 MB/s raw sequential, O_DIRECT 4 MiB), 16 KiB pages and an Active Cooler.

| 31B dense, unsloth UD-IQ3_XXS (11,837,782,976 B), 4 threads, from RAM after load | MEASURED | PROJECTED by feasibility |
|---|---:|---:|
| tg32 (llama-bench, `--repack 0` and `--repack 1` identical) | **0.53 tok/s** | 0.55 tok/s (10.0 GB/s × 0.65 / 11.82 GB) |
| pp64 | 0.63 tok/s | — |
| Cold load from microSD (`-nr`) | 138 s (about 86 MB/s) | 11.84 GB / 85 MB/s = 139 s |
| Anonymous memory with `-nr` | 0.44 GB (the 12.58 GB working set is file-backed page cache) | — |
| Peak temperature over 23 min | 72.5 °C, `throttled=0x0` | — |

Effective traffic is 0.53 × 11.82 GB = **6.27 GB/s**, which is 0.63 of the ASSUMED 10.0 GB/s. That
agrees with the 0.65 efficiency within 4%. **This agreement is not yet a bandwidth calibration.**
pp64 (0.63) is barely above tg32 (0.53). A batch of 64 tokens shares each weight read, so a
bandwidth-bound kernel would show a much higher pp. The IQ3_XXS dequantisation (grid lookups) is
therefore probably **compute-bound** on the A76. Q4_0 and Q4_K have NEON dotprod kernels and need
their own number (pi/05 with the 26B). The 26B UD-Q3_K_M run on the same board had not finished when
this was written.

### 3.4 The time model (shared by H1, the dense line and feasibility)

For a MoE with LRU hit rate `h` over the routed expert slabs (feasibility, cache_sim):

```
B_ne   = nonexpert_bytes_per_token          (26b-q4: 1,577,172,088 B)
W      = expert_bytes_per_token_worst        (26b-q4:   802,897,920 B)
N      = n_moe_layers × n_expert_used        (26b-q4:   240 slab reads per token)
BW_mem = mem_bw_gbps × 1e9 × llama_efficiency            (10.0 GB/s × 0.65 = 6.5 GB/s)
BW_st  = rand_read_3mb_mbps × 1e6;  a = access_ms / 1e3

t_ram  = (B_ne + h·W) / BW_mem                 resident bytes served from RAM
t_io   = (1 − h)·W / BW_st + (1 − h)·N·a       missed slabs from storage
tg_serial  = 1 / (t_ram + t_io)                no overlap of I/O and compute
tg_overlap = 1 / max(t_ram, t_io)              perfect overlap (upper bound; cache_sim's
                                               lru_prefetch reports the achievable part)
in-RAM ceiling (h = 1)  = BW_mem / (B_ne + W)  = 6.5 GB/s / 2.38 GB = 2.73 tok/s
dense, file larger than RAM: s/token = file_bytes / (seq_read_mbps × 1e6)
```

Two facts follow before any experiment runs:

- **The RAM side caps everything.** Even with zero expert traffic, `t_ram ≥ B_ne / BW_mem` =
  0.243 s, so no 26B configuration on a Pi 5 can exceed 4.12 tok/s at an efficiency of 0.65. With
  every expert in RAM the ceiling is **2.73 tok/s**: 10.0 GB/s × eff / 2.38 GB, which is 2.7 to
  4 tok/s for an efficiency of 0.65 to 0.95. Caching can only approach this ceiling, never exceed
  it.
- The 31B dense line is 17.64 GB per token. Even in RAM that is 0.37 tok/s at 0.65 efficiency and
  0.57 at 1.0, and the file does not fit in 16 GB of RAM.

### 3.5 RAM worksheets per configuration (PROJECTED; inputs ASSUMED unless marked)

Expert cache = RAM − OS reserve − KV (q8_0, 4096 ctx) − compute buffers − non-expert resident.
Feasibility and cache_sim `--profiles` use the same formula:

| | pi5-4gb-microsd | pi5-8gb-usb3 | pi5-8gb-nvme-gen3 | pi5-16gb-nvme-gen3 | context-friend-1 (16 GB, microSD) |
|---|---:|---:|---:|---:|---:|
| RAM | 4.00 GB | 8.00 GB | 8.00 GB | 16.00 GB | 17.01 GB (MEASURED MemTotal) |
| − OS reserve | 0.7 GB | 0.8 GB | 0.8 GB | 0.9 GB | 0.9 GB (tier rule) |
| − KV q8_0 at 4096 (26B) | 0.16 GB | 0.16 GB | 0.16 GB | 0.16 GB | 0.16 GB |
| − compute buffers | 0.3 GB | 0.3 GB | 0.3 GB | 0.4 GB | 0.3 GB (default) |
| = usable | 2.84 GB | 6.74 GB | 6.74 GB | 14.54 GB | 15.65 GB |
| − non-expert resident (26b-q4) | 1.58 GB | 1.58 GB | 1.58 GB | 1.58 GB | 1.58 GB |
| **= expert cache** | **1.26 GB** | **5.16 GB** | **5.16 GB** | 12.96 GB ≥ 12.85 GB: **all** | 14.07 GB: **all** |
| cache / all experts | 9.8% | 40.2% | 40.2% | 100% | 100% |
| expert-layer slabs cached (of 3840) | 377 (12.6 per layer) | 1543 (51.4 per layer) | 1543 (51.4 per layer) | 3840 | 3840 |
| 14.44 GB Q4_0 file fits in RAM? | no | no | no | **only with `-nr`** (14.44 ≤ 14.54; the repack copy needs 14.74 ≤ 13.54, which fails) | yes with `-nr`. The SD card cannot hold it with the 2 GB free-space rule (15.50 GB free). |

The catalog lists `n_ctx` 2048 for the 4 GB profile. Feasibility floors the KV at its 4096 value,
which is conservative. On a 16 GB board the 0.10 GB spare is thin, but the file pages are
reclaimable page cache, so the failure mode is mild re-streaming, not an OOM.

### 3.6 RAM × storage grid (PROJECTED by `feasibility.verdict()` with the PRIOR hit curve)

The table shows 26b-q4 tok/s and the feasibility verdict. Section 4.8 gives the pre-registered
viability line (1.0 tok/s).

| RAM \ storage | microSD (85/60 MB/s, 2.0 ms) | USB 3.0 SSD (380/350 MB/s, 0.5 ms) | NVMe Gen2 (430/400 MB/s, 0.2 ms) | NVMe Gen3 (850/700 MB/s, 0.2 ms) |
|---|---|---|---|---|
| 4 GB | 0.11 NOT_VIABLE | 0.57 STREAM_SLOW | 0.65 STREAM_SLOW | 0.98 STREAM_SLOW |
| 8 GB | 0.44 STREAM_SLOW | 1.46 STREAM_OK | 1.57 STREAM_OK | **1.94 STREAM_OK** |
| 16 GB | 2.73 FITS_MAPPED | 2.73 FITS_MAPPED | 2.73 FITS_MAPPED | **2.73 FITS_MAPPED** |
| 31b-q4 on any of these | 207.7 s/token | 46.5 s/token | 41.0 s/token | 20.8 s/token |

Feasibility verdicts: STREAM_OK means ≥ 1.0 tok/s with a cache of at least 2 GB on NVMe or USB.
STREAM_SLOW means ≥ 0.3 tok/s. FITS_MAPPED means the file fits in RAM only file-backed (`-nr`). The
31B is NEVER_GOOD_SPEED everywhere.

The hardware recommendation follows directly: **16 GB + NVMe Gen3 + Active Cooler**. It is
RAM-bound at 2.7 to 4 tok/s, and the NVMe matters for load time (14.44 GB / 850 MB/s ≈ 17 s, versus
170 s from microSD), not for decode. The 8 GB + NVMe Gen3 board is the configuration where H1 is
actually needed. Multi-Pi (llama.cpp RPC) is slower than one 16 GB Pi in published measurements.

### 3.7 The dense line (PROJECTED, 31b-q4, 17,635,166,448 B/token)

| Where the weights live | s/token | tok/s |
|---|---:|---:|
| RAM, if it fit: 10.0 GB/s × 0.65 | 2.71 | 0.37 |
| RAM, if it fit: 10.0 GB/s × 1.0 | 1.76 | 0.57 |
| NVMe Gen3, 850 MB/s sequential | 20.7 | 0.048 |
| NVMe Gen2, 430 MB/s | 41.0 | 0.024 |
| USB 3.0 SSD, 380 MB/s | 46.4 | 0.022 |
| microSD, 85 MB/s | 207.5 | 0.005 |

The grid in section 3.6 shows 20.8 / 41.0 / 46.5 / 207.7 s because feasibility divides the whole
file (17,651,001,568 B, including the GGUF header and metadata) rather than the tensor bytes.

The access pattern also hurts. A dense model reads its weights in the same cyclic order every token.
With a file larger than the cache, pure LRU on a cyclic scan gets **0%** hits, so only a pinned
(static) subset helps (pi/06). Linux's page cache uses active/inactive lists rather than pure LRU,
so pi/05 measures what actually happens. The fastest *measured* dense 31B on a Pi 5 is the 3-bit
model held in RAM: **0.53 tok/s** (section 3.3).

---

## 4. H1: MoE small active set (Gemma 4 26B-A4B)

### 4.1 Statement

> On a Raspberry Pi 5 with 8 GB of RAM and an NVMe Gen3 drive, Gemma 4 26B-A4B Q4_0 runs at
> **≥ 1.0 tok/s serial and ≥ 2.0 tok/s with I/O overlapped** (verdict STREAM_GOOD). It keeps the
> 1,577,172,088 B of non-expert weights plus an LRU cache of 5.16 GB of expert slabs (40.2% of
> 12,846,366,720 B) in RAM and reads every missed slab from the drive. Its output is unchanged
> (`-nr` vs default: KLD mean < 0.001).

The claim can be checked before a Pi exists because it splits into measurable parts:

| Part | What must hold | Measured by |
|---|---|---|
| H1a, skew | Routing on real EN/HE/code text is skewed and sticky: mean coverage@32 ≥ 0.70 **or** adjacent reuse ≥ 0.40 | E3 (`pigemma.trace stats` on the probe trace) |
| H1b, cache | LRU at the 8 GB budget reaches hit ≥ 0.582 (≤ 336 MB of misses per token) | E4 (`pigemma.cache_sim`) |
| H1c, realism | The real kernel page cache under a memory cap moves within ±30% of the simulated LRU bytes, and the time model predicts tok/s within ±35% | E5 (`pc/pagecache_limit.sh`) |
| H1d, pinning | Pinning non-expert plus hot experts beats plain page-cache LRU by ≥ 15% tok/s or ≥ 20% fewer bytes | E6 |
| H1e, in-RAM tier | A 16 GB Pi runs the whole model in RAM at 2.7 to 4 tok/s (the ceiling of section 3.4) | pi/05 on context-friend-1 (UD-Q3_K_M there, projected 2.13) |
| precondition | `-nr` keeps the file CPU_Mapped (≥ 95% of the file, no CPU_REPACK line, private bytes < 1.5 GB), while `-ot "ffn_(gate_up\|down)_exps=CPU"` still repacks (negative control) | E0 gate. If it fails, all streaming work stops. |

### 4.2 Why `-nr` is part of the hypothesis

At llama.cpp a7b94df the default load copies every Q4_0 tensor out of the mmap into anonymous
CPU_REPACK memory (llama-model.cpp:1849-1870). An `-ot ...=CPU` override re-selects CPU_REPACK
(llama-model-loader.cpp:1240-1242). `-nr` / `--no-repack` is therefore the **only** mode in which
expert pages stay file-backed and can be evicted and re-read: it is the mechanism H1 relies on.
Without it, streaming silently turns into a 14 GB anonymous allocation that OOMs on 8 GB. The cost
is that the non-repacked Q4_0 kernels are slower. E0 measures that penalty on the desktop and pi/05
on the Pi, and section 4.7 shows what it does to the projections.

### 4.3 What the cache must achieve (PROJECTED; 26b-q4, efficiency 0.65)

This analysis is independent of the budget. It gives the hit rate needed for each speed target on
each storage class, and the cache size the PRIOR curve says would deliver it:

| Storage (catalog class) | t_io with nothing cached | hit for tg_serial ≥ 1.0 | hit for tg_overlap ≥ 2.0 | **hit for STREAM_GOOD (both)** | hit for tg_serial ≥ 2.0 | PRIOR cache for STREAM_GOOD |
|---|---:|---:|---:|---:|---:|---|
| microSD (60 MB/s, 2.0 ms) | 13.86 s | 0.954 | 0.964 | **0.964** | 0.990 | unreachable (the prior tops out at 0.88) |
| USB 3.0 SSD (350 MB/s, 0.5 ms) | 2.41 s | 0.723 | 0.793 | **0.793** | 0.942 | 30.3% = 3.89 GB (board RAM ≈ 6.73 GB) |
| NVMe Gen2 (400 MB/s, 0.2 ms) | 2.06 s | 0.672 | 0.757 | **0.757** | 0.931 | 26.7% = 3.43 GB (≈ 6.27 GB) |
| NVMe Gen3 (700 MB/s, 0.2 ms) | 1.20 s | 0.408 | 0.582 | **0.582** | 0.875 | 14.5% = 1.87 GB (≈ 4.71 GB) |

"Board RAM" adds the 8 GB-tier overheads to the cache: non-expert 1.58 + KV 0.16 + buffers 0.3 + OS
0.8 GB. For the NVMe Gen3 row, serial ≥ 2.0 needs hit 0.875, i.e. a 5.62 GB cache on an ≈ 8.46 GB
board. That is just out of reach of an 8 GB Pi under the prior.

Miss traffic at a given hit rate is (1 − h) × 802.9 MB. At 60 / 75 / 85 / 92% hit that is 321 /
201 / 120 / 64 MB per token, matching the research summary's 322 / 201 / 121 / 64 MB.

### 4.4 Priors (PRIOR: literature, other models; replaced by E3 and E4)

| Prior | Value | Source | Replaced by |
|---|---|---|---|
| LRU hit vs fraction of experts resident (SeqMoE, average of 4 MoE models) | **15% → 60%, 25% → 74%, 35% → 84%, 45% → 88%**. Linear to (0, 0) below 15%; held at 0.88 above 45%. | RESEARCH_SUMMARY section 3; catalog `selection.stream_hit_rate_curve_assumed` | E4 `sim.json` (source `simulated`) |
| Next-token expert reuse, Qwen3-30B-A3B (same 128-expert / top-8 shape) | **0.444** (vs 0.223 by frequency-matched chance) | RESEARCH_SUMMARY section 3 | E3 `adjacent_reuse` |
| LRU holding 13.4% of Qwen3-30B-A3B experts | serves 66% of fetches | RESEARCH_SUMMARY section 3 | E4 |
| Qwen3-30B-A3B streamed from NVMe on a 16 GB M1 | 3.8 to 4.7 tok/s at 56 to 88% hits | llama.cpp discussion #27149 | pi/05 |
| Qwen3-30B-A3B on a 16 GB Pi 5 (2.70 bpw, in RAM) | about 8 tok/s | RESEARCH_SUMMARY section 2 | pi/05 (our model reads 2.38 GB/token at Q4_0, far more than that quant) |
| Plain LRU on phone CPUs | still misses 22 to 40% with 30 to 45 experts per layer cached | Qualcomm, RESEARCH_SUMMARY section 3 | E4 |

**Null baselines** (uniform independent routing): coverage@32 = 32/128 = 0.25, adjacent reuse =
8/128 = 0.0625, cross-layer precision@16 = (16 × 8 / 128) / 8 = 0.125, and LRU hit = cache fraction (0.402
at the 8 GB budget). With a uniform hit of 0.402, the 8 GB + NVMe Gen3 Pi would still project
**0.99 tok/s** serial (1.39 overlapped). **Routing skew decides between "slow" and "good"; it is not
what makes the 8 GB tier possible.**

### 4.5 Projections per catalog profile (PROJECTED, PRIOR hit, `feasibility.verdict()`)

| Profile | expert cache | resident fraction | PRIOR hit | miss MB/token | misses per token | t_ram | t_io | **tg_serial** | **tg_overlap** | feasibility | E4 label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pi5-4gb-microsd | 1.26 GB | 9.8% | 0.393 | 487 | 145.6 | 0.291 s | 8.411 s | **0.11** | 0.12 | NOT_VIABLE (exit 3) | NOT_VIABLE |
| pi5-8gb-usb3 | 5.16 GB | 40.2% | 0.861 | 112 | 33.4 | 0.349 s | 0.336 s | **1.46** | 2.87 | STREAM_OK | STREAM_GOOD |
| pi5-8gb-nvme-gen3 | 5.16 GB | 40.2% | 0.861 | 112 | 33.4 | 0.349 s | 0.166 s | **1.94** | 2.87 | STREAM_OK | STREAM_GOOD |
| pi5-16gb-nvme-gen3 | all | 100% | 1 | 0 | 0 | 0.366 s | 0 | **2.73** | 2.73 | FITS_MAPPED | FITS_RAM |

All four rows rest on ASSUMED storage, an ASSUMED efficiency of 0.65, ASSUMED memory bandwidth and
the PRIOR hit curve. Nothing in this table is measured.

### 4.6 What a skeptic should look at first (sensitivity, PROJECTED, 8 GB + NVMe Gen3)

| Efficiency \ effective storage (random 3 MB) | 700 MB/s (catalog) | 350 MB/s (page-fault path at half of fio) | 175 MB/s (quarter) |
|---|---|---|---|
| 0.90 | 2.39 serial / 3.97 overlap | 1.73 / 3.07 | 1.11 / 1.55 |
| **0.65 (catalog)** | **1.94 / 2.87** | 1.48 / 2.87 | 1.01 / 1.55 |
| 0.45 (`-nr` penalty about 30%) | 1.49 / 1.98 | 1.20 / 1.98 | 0.87 / 1.55 |

Each cell assumes the PRIOR hit of 0.861. The same efficiencies move the 16 GB in-RAM ceiling to
3.78 / 2.73 / 1.89 tok/s. The two biggest unknowns are the effective storage bandwidth and the ARM
`-nr` kernel efficiency. The first matters because llama.cpp misses arrive as synchronous 16 KiB page
faults plus readahead, not as fio's queued 3 MB reads. E5, pi/03 and pi/05 measure both; the routing
statistics are a smaller lever.

### 4.7 Known simplifications in the H1 model

- Each missed slab is two tensors (gate_up and down), i.e. two seeks, not one. Doubling the access
  term moves 1.94 to 1.92 tok/s.
- With 16 KiB alignment slabs grow by 0.98% (3,378,142 B mean), which moves 1.94 to 1.93.
- Readahead can inflate the bytes read per miss. cache_sim models this with `--read-granularity`
  (default 131072 B) and E5 measures the real factor.
- A missed slab is read from storage and then again from RAM by the matmul. This adds
  (1 − h)·W / BW_mem, about 0.017 s at h = 0.86.
- KV reads are ignored: up to 0.16 GB per token at a full 4096 context (+0.024 s at 6.5 GB/s), close
  to zero for short chats.
- Compute is ignored: the model is bandwidth-only. The real Pi's pp ≈ tg on IQ3_XXS (section 3.3)
  shows that some kernels are compute-bound on the A76. If pi/05 finds that for Q4_0 / Q4_K too, every
  ceiling here is optimistic.
- `tg_overlap` assumes perfect prefetch. Only `lru_prefetch` in cache_sim says how much overlap a
  real predictor achieves.

### 4.8 Pre-registered thresholds (fixed before the data)

| Exp. | Metric | Threshold and meaning |
|---|---|---|
| E0 | Repack gate | **PASS** if `-nr` logs CPU_Mapped ≥ 95% of the file with no CPU_REPACK line and private peak < 1.5 GB, **and** the `-ot` negative control logs CPU_REPACK. If it fails, stop E5, E6 and the Pi streaming work. Desktop reference within ±20%: 26B tg 15.18 at 4 threads and 13.5 at 6 threads, 31B tg 2.44 at 6 threads. |
| E2 | Numerical neutrality | `-nr` and the negative control: KLD mean < 0.001 and same-top-p > 99.5% |
| E3 | Skew (H1a) | Trace VALID if `validate()` is clean, with ≥ 10k prefill and ≥ 2k generated tokens and coherent text. **Strong skew**: mean coverage@32 ≥ 0.70 *or* adjacent reuse ≥ 0.40. **Moderate**: coverage@32 between 0.5 and 0.7. **Near-uniform**: < 0.5, in which case LRU alone is the plan and static hot sets are dropped. Prefetch is modelled only if predictor precision@16 beats the static baseline by ≥ 0.1. |
| E4 | Cache (H1b) | Per profile and policy, with LRU (never Belady, which is the upper bound only): **STREAM_GOOD** if tg_overlap ≥ 2.0 **and** tg_serial ≥ 1.0. **STREAM_WEAK** if tg_serial is between 0.5 and 1.0. Otherwise **NOT_VIABLE**. A row with tg_serial ≥ 1.0 and tg_overlap < 2.0 is reported as STREAM_WEAK, closing a gap in the literal rule. If LRU bytes/token > 1.5 × Belady, prefetch and static policies are worth building. If static is within 10% of LRU, pinning is the simpler deployment. Any projection whose inputs are still ASSUMED says so in its row. |
| E5 | Realism (H1c) | **PASS** if \|sim_lru / measured − 1\| ≤ 0.30 for caps of 4 to 8 GB **and** measured lies within [0.8 × Belady, 1.3 × LRU]. Time model **PASS** if \|predicted / measured − 1\| ≤ 0.35 on ≥ 3 caps. **Zero** OOM events with `-nr`. |
| E6 | Pinning (H1d) | **PASS** if pinned tg ≥ 1.15 × unpinned at the 5 GB cap, **or** bytes/token falls by ≥ 20%. Otherwise pinning stays optional. |
| Pi | Speed lines | `selection.viable_stream_tok_s` = **1.0** tok/s ("viable") and `selection.good_stream_tok_s` = **2.0** tok/s ("good"). Both are policy choices made here, not measurements. |

**H1 is supported** for the 8 GB tier if E0 passes, E4 gives STREAM_GOOD for pi5-8gb-nvme-gen3 with
LRU on Gemma 4's own trace, and E5 passes. The Pi verdict stays PROJECTED until pi/05 measures the
same configuration. **H1 is falsified** for that tier if E4's LRU hit at the 5.16 GB budget is
below 0.408 (not even viable), or if E5 fails in the direction measured ≫ simulated and the
corrected projection drops below 1.0 tok/s. Between 0.408 and 0.582, H1 survives only as "viable
but slow" (≥ 1.0 serial, < 2.0 overlapped).

### 4.9 What would change our minds about H1

| Observation | Consequence |
|---|---|
| E3 near-uniform (coverage@32 < 0.5) | The prior curve is far too optimistic. Expect hit ≈ 0.40 at 8 GB: 0.99 tok/s on NVMe Gen3, i.e. "slow", not "good". Static hot sets are dropped. |
| E4 LRU hit ≥ 0.875 at 5.16 GB | 8 GB + NVMe Gen3 reaches ≥ 2.0 tok/s *serial*. The model's own locality beats the literature, which would be the best case for H1. |
| E5 measured bytes/token ≫ simulated (readahead) | Record `read_ahead_kb`, feed the inflation factor into `--read-granularity`, and lower the effective storage bandwidth. The table in section 4.6 shows a halving costs 0.46 tok/s. |
| E0 or pi/05 `-nr` penalty > 30% | Even the 16 GB Pi falls below 2 tok/s (1.89 at 0.45 efficiency). The real fix is the follow-up llama.cpp patch: repacked attention and head, mapped experts. |
| pi/03 `memory.read_gbps` < 8 GB/s or pi/05 calibration < 0.5 | Every Pi projection scales down linearly in t_ram. The 2.73 ceiling becomes ≈ 2.1 or less. |
| pi/05: Q4_0 / Q4_K pp ≈ tg on the A76 (compute-bound, as IQ3_XXS was) | The bandwidth model is the wrong model for the Pi. Add a compute term; caching cannot help beyond it. |
| Thermal throttling (`vcgencmd get_throttled` ≠ 0x0) during long decodes | Measured tok/s is a thermal artefact. Re-run with the Active Cooler and report both. |

---

## 5. H2: dense staged residuals (Gemma 4 31B)

### 5.1 Statement

> For each FFN matrix of the 31B (gate and up: 21504 × 5376, down: 5376 × 21504; `y = W x`), write
> `W = W_base + Σ_g R_g`, where the groups g are blocks of **128 input columns** and the residual is
> quantised at 4 bits. Measure on real captured activations at a relative FFN-output error ε = 3%.
> H2 is **PROMISING** if *all* of these hold: (i) the oracle needs **≤ 25%** of the residual bytes;
> (ii) the runtime-checkable norm bound certifies **≥ 50%** of what the oracle skips; (iii) implied
> traffic is **< 8.5 GB/token** (the 31B IQ2_XXS file); (iv) base plus oracle has lower error than
> uniform group quantization at the same bytes.

### 5.2 The decomposition and the bound

```
y_ref = W_q4 · x                         (exact dequant of the Q4_0 matrix: what llama.cpp computes)
W_q4  = W_base + Σ_g R_g                 R_g = residual restricted to input columns g (n_out × 128)
C_g   = R_g · x_g                        contribution of group g to the output
‖C_g‖ ≤ ‖R_g‖₂ · ‖x_g‖₂ =: b_g           (spectral norm, one float stored per group)
```

`‖R_g‖₂` (the largest singular value) is computed once offline. `‖x_g‖₂` is known as soon as the
layer input exists, before any residual byte is read. So `b_g` costs one 128-element norm per group
at run time, and a group can be skipped **without reading it** whenever the bound allows.

### 5.3 Why input-column groups, stored group-major

1. **The bound needs `‖x_g‖`.** A column group's contribution depends only on the inputs in that
   group, so the bound adapts to the input and is cheap. A group of output rows would be bounded by
   `‖R_rows‖·‖x‖`, with the same full `‖x‖` for every group. That bound cannot exploit which inputs
   are small, and every skipped row leaves one output coordinate plainly wrong.
2. **The contributions add.** `y = W_base x + Σ_g C_g`, so the errors of skipped groups add as
   vectors and the triangle inequality gives a sound total.
3. **Contiguous reads.** ggml stores a matrix row-major with `ne[0] = n_in` contiguous, so a slice of
   128 columns is `n_out` scattered 72-byte pieces (4 Q4_0 blocks per row). Each 16 KiB page fault
   would then deliver 72 useful bytes. Stored group-major, each group is one contiguous read:
   **387,072 B** for down (5376 × 128 at 4.5 bits per weight) and **1,548,288 B** for gate and up
   (21504 × 128). That is the same 0.4 to 3 MB range as the MoE expert slabs, so the H1 storage
   numbers apply.
4. Group counts: 5376 / 128 = 42 groups (gate, up) and 21504 / 128 = 168 groups (down). No
   remainder.

### 5.4 How loose is the bound? (expectation and measurement)

For a random unit direction `u` in a group, `E‖R_g u‖² = ‖R_g‖_F² / g`. The per-group tightness
`‖C_g‖ / b_g` is therefore about **√(srank(R_g) / g)**, where `srank = ‖R‖_F² / ‖R‖₂²` is the stable
rank:

- If R_g is close to rank one, tightness is about 1/√g = 0.088 at g = 128. This is the manifest's
  "~1/√g" figure, but it is the **worst case**, not the typical one.
- If R_g is white noise (a tall 5376 × 128 Gaussian), tightness is close to 1.
- **MEASURED** on the real `blk.20.ffn_down.weight` (q2 base, section 5.9): stable rank median 30.8
  of 128, tightness **p10 / p50 / p90 = 0.425 / 0.49 / 0.595**, matching √(30.8/128) = 0.49. The
  bound is therefore about **2× loose per group**, not 11×.

**The larger looseness comes from summation, not from a single group.** Contributions of skipped
groups are close to independent vectors in `R^{n_out}`, so their true sum grows like √k while the
bound's sum grows like k. If the oracle can skip `k_o` comparable groups, the certificate can skip
about `τ·√k_o` of them, i.e. a certification efficiency of about **τ / √k_o**. With τ ≈ 0.49 and
k_o ≈ 17, that is about 0.12. The 50% efficiency in criterion (ii) therefore needs either very few
skippable groups or strongly concentrated, near-parallel contributions. The probe measured an
efficiency of about 10 to 15% at ε = 10% (section 5.9).

### 5.5 The oracle (pigemma.h2.oracle, E7)

Per token, sort the groups by `‖C_g‖` in descending order and add them to `W_base x` until
`‖y_ref − ŷ‖ ≤ ε‖y_ref‖`, for ε in {0.01, 0.03, 0.10}. Record the number of groups `k_oracle` and the
**residual byte fraction** = bytes of the selected groups / all residual bytes. The oracle needs
`C_g`, i.e. the answer, so it **cannot run at inference time**. Up to its greedy ordering, it is an
upper bound on what any selector that must decide before reading could save. The base error
`err_base = ‖y_ref − W_base x‖ / ‖y_ref‖`, the error-vs-bytes curve, and the split between
teacher-forced (phase 0) and generated (phase 1) tokens are reported alongside.

**Floor.** The residual is itself quantised (4 bits by default), so even reading every group leaves
an error of `‖(R − R̂)x‖`. The skippable residual energy is about
`(ε² − φ²) / err_base²`, with φ the floor relative to `‖y‖`. On the real down matrix with a q2 base,
`‖R − R̂‖ / ‖R‖` = 0.0725, so φ ≈ 0.0725 × 0.383 = 0.028. That puts **ε = 1% out of reach** and
leaves ε = 3% with almost no slack: about 0.09% of the residual energy can stay unread. An ε below
the floor must be reported as unreachable, not as "all groups".

### 5.6 The sound certification rule (as in pigemma.h2.oracle)

```
sort groups by b_g ascending; certified-skip set P = the longest prefix with
    B := Σ_{g∈P} b_g  ≤  ε · ‖ŷ‖ / (1 + ε),      ŷ = y_ref − Σ_{g∈P} C_g
proof:  ‖y_ref − ŷ‖ = ‖Σ_P C_g‖ ≤ Σ_P ‖C_g‖ ≤ B   and   ‖y_ref‖ ≥ ‖ŷ‖ − B
        ⇒ ‖y_ref − ŷ‖ / ‖y_ref‖ ≤ B / (‖ŷ‖ − B) ≤ ε    ⇔   B(1 + ε) ≤ ε‖ŷ‖
```

The rule can be decided at run time. The `b_g` are known before any read and `ŷ` is built from the
groups that *were* read: read the complement of a candidate prefix, compute ŷ and check the
inequality, and if it fails, read the prefix's largest-`b_g` group and check again. Reported per
case: certified-skip fraction (bytes), realised error, **soundness violations (must be 0)**,
**certification efficiency** = certified bytes skipped / oracle bytes skipped at the same ε,
tightness p10 / p50 / p90, and the Spearman correlation between `b_g` and `‖C_g‖`. A runtime version
must also bound the quantisation floor of section 5.5 (`Σ_g ‖R_g − R̂_g‖₂ ‖x_g‖`), or use residual
bits high enough that the floor is negligible.

### 5.7 The iso-byte competitor

Cost per weight, block 32, f16 scale: q2 = **0.3125 B**, q3 = **0.4375 B**, q4 = **0.5625 B** (= Q4_0).
A low-rank base of rank r costs `2·r·(n_in + n_out)` bytes: **0.119 B/weight** at rank 256 and 0.238
at rank 512. Base q2 plus a residual fraction f costs `0.3125 + 0.5625·f` B per weight, so:

- it equals uniform q3 at **f = 0.222**;
- it equals Q4_0 at **f = 0.444**;
- at f = 25% it costs 0.453 B per weight, which is 0.81 × Q4_0.

E7 compares, at equal bytes, the relative error of plain uniform group quantization against base
plus oracle and base plus certified. The verdict is **NO_ADVANTAGE** if uniform q3 has error ≤ 1.2 ×
the oracle error at equal bytes. **MEASURED** on the real down matrix: uniform q3 has a relative
Frobenius error of **0.210**, and the q2 base alone has 0.383.

The real competitor on the real Pi is simpler still: the 3-bit 31B held in RAM, **MEASURED at
0.53 tok/s** (section 3.3). H2 matters only if it beats that number at equal or better quality.

### 5.8 Byte arithmetic: what PROMISING would require (PROJECTED, 31b-q4)

The FFN holds 20,808,990,720 weights (180 matrices × 115,605,504). Attention plus head add
**5,926,237,184 B/token (5.93 GB)** that H2 does not touch:

| Base | FFN base bytes | Total with **no** residual read | Residual fraction allowed under 8.5 GB |
|---|---:|---:|---:|
| q2 (0.3125 B/weight) | 6,502,809,600 | **12.43 GB** | **none.** The criterion is unreachable. |
| rank 256 (0.119 B/weight) | 2,477,260,800 | 8.40 GB | ≤ **0.82%** of the 11.71 GB of 4-bit residual |
| q2 base, f = 25% (the PROMISING fraction) | 6,502,809,600 + 2,926,264,320 | 15.36 GB | — |

So criterion (iii) **cannot be met with a q2 base at all**. It needs a base cheaper than
2.57 GB / 20.8 G weights = **0.124 B/weight** that also has a small error. The only such base
in E7 is rank 256, and on the real down matrix it has a **base error of 0.905** (it captures 18.0% of
the energy; rank 512 reaches 0.852 / 27.4%). A base that bad would need almost the whole residual
to reach 3%. The pre-registered expectation is therefore **NOT_SUPPORTED** (or NO_ADVANTAGE). The
experiment still runs, because it measures *how far* H2 is from viability on real activations.
Real activations, which have outlier channels, are the one input the probe below did not use.

### 5.9 Pre-registration probe (MEASURED on real weights with *synthetic* inputs, 2026-10-05)

To ground sections 5.4, 5.5 and 5.8 before E7, one real tensor was read memory-mapped:
`blk.20.ffn_down.weight` of `31b-q4`, at file offset 6,728,959,968, 65,028,096 B, `[n_out 5376,
n_in 21504]`. The probe used a ggml-style symmetric-absmax group quantizer (block 32, f16 scale:
`d = m / −2^(b−1)`, `q = clamp(round(w/d), −2^(b−1), 2^(b−1)−1)`), groups of 128 input columns, a
4-bit residual, and **32 synthetic input vectors** (Gaussian, and a GEGLU-like `gelu(a)·b` proxy).
This is **not** E7. `pigemma.h2.decompose` may use a different quantizer, and real activations may
concentrate energy differently.

| Quantity | Gaussian x | GEGLU-proxy x |
|---|---:|---:|
| rel. Frobenius error q2 base / q3 / rank-256 / rank-512 | 0.383 / 0.210 / 0.905 / 0.852 | (same, weight-space) |
| 4-bit residual floor `‖R − R̂‖ / ‖R‖` | 0.0725 | (same) |
| err_base (output space, median) | 0.383 | 0.382 |
| per-group tightness p10 / p50 / p90 | 0.425 / 0.49 / 0.595 | 0.425 / 0.489 / 0.595 |
| oracle residual fraction, ε = 1% / 3% / 10% (median) | unreachable / **1.00** / 0.92 | unreachable / **1.00** / 0.875 |
| certified-skip fraction, ε = 1% / 3% / 10% (median) | 0 / 0 / 0.012 | 0 / 0 / 0.012 |

Random-direction tightness was 0.421 / 0.492 / 0.612 (p10 / p50 / p90), compared with 1/√128 = 0.088.
The probe took 13 s; no inference ran.

<details><summary>Probe code (numpy only; run from the repo root; E7 supersedes it)</summary>

```python
import numpy as np
GGUF = "models/google__gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf"
OFF, NOUT, NIN, g = 6728959968, 5376, 21504, 128          # blk.20.ffn_down.weight (traces/h2/map31.json)
blk = np.asarray(np.memmap(GGUF, np.uint8, "r", OFF, (NOUT * NIN // 32 * 18,))).reshape(-1, 18)
d = blk[:, :2].copy().view(np.float16).astype(np.float32)
qs = blk[:, 2:]
W = (d * (np.concatenate([qs & 15, qs >> 4], 1).astype(np.float32) - 8)).reshape(NOUT, NIN)
def gq(A, bits):                                           # symmetric absmax, block 32, f16 scale
    B = A.reshape(-1, 32); m = B[np.arange(len(B)), np.abs(B).argmax(1)]
    lo, hi = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    s = (m / lo).astype(np.float16).astype(np.float32)[:, None]; s[s == 0] = 1
    return (np.clip(np.round(B / s), lo, hi) * s).reshape(A.shape)
Wb = gq(W, 2); R = W - Wb; Rh = gq(R, 4); G = NIN // g
s2 = np.array([np.linalg.svd(Rh[:, k*g:(k+1)*g], compute_uv=False)[0] for k in range(G)])
x = np.random.default_rng(0).standard_normal(NIN).astype(np.float32)
y = W @ x; e = y - Wb @ x; xg = x.reshape(G, g)
C = np.stack([Rh[:, k*g:(k+1)*g] @ xg[k] for k in range(G)]); cn = np.linalg.norm(C, axis=1)
b = s2 * np.linalg.norm(xg, axis=1)
print("err_base", np.linalg.norm(e) / np.linalg.norm(y), "tightness p50", np.median(cn / b))
k = 0                                                       # oracle at eps = 0.03
for i in np.argsort(-cn):
    if np.linalg.norm(e) <= 0.03 * np.linalg.norm(y): break
    e = e - C[i]; k += 1
print("oracle groups read", k, "of", G)
```
</details>

### 5.10 Pre-registered thresholds (E7)

| Gate | Rule |
|---|---|
| Pipeline VALID | sanity median rel_err(`W_down_q4 · ffn_geglu`, captured `ffn_out`) **< 0.03** (abort above 0.1: orientation or scale bug) **and** soundness_violations **== 0** |
| **PROMISING** | at ε = 3%: oracle residual fraction **≤ 25%** AND certification efficiency **≥ 50%** AND implied bytes/token **< 8.5 GB** AND base + oracle error < uniform-quant error at equal bytes |
| **WEAK** | fraction ≤ 50%, or certification efficiency between 0.2 and 0.5 |
| **NO_ADVANTAGE** | uniform q3 error ≤ 1.2 × the oracle error at equal bytes |
| **NOT_SUPPORTED** | otherwise |
| Always reported | s/token on a Pi: any traffic above 2 GB/token is < 0.5 tok/s from NVMe Gen3. Attention + head = 5.93 GB/token caps the dense model at **1.69 tok/s** at 10.0 GB/s even if everything else cost nothing (1.10 at 0.65 efficiency). |

### 5.11 The non-promise

Even a perfect H2 (zero FFN bytes) leaves **5.93 GB per token** of attention and tied head. At the
Pi's 10.0 GB/s that is at most 1.69 tok/s with everything in RAM (1.10 at the ASSUMED 0.65), and
from NVMe Gen3 at 850 MB/s it is 7.0 s per token. H2 is a **measurement** of how much of the
residual a decision actually needs. It is not a path to "good speed" for the dense 31B on any Pi 5.
For the dense model the larger proven lever is speculative verification with a resident draft: on
Qwen2.5-32B, SubSpec-style self-speculation took 0.52 to 6.33 tok/s (PRIOR). At b11398 the Gemma 4
MTP sidecars load but never draft (E1, MEASURED on the desktop).

### 5.12 What would change our minds about H2

| Observation | Consequence |
|---|---|
| E7 on real activations: at ε = 3%, ≥ 75% of groups have negligible `‖C_g‖` (real outlier channels concentrate the energy) | Fraction criterion (i) could pass. Criterion (iii) still needs a base ≤ 0.124 B/weight with low error, so look at activation-ordered column permutations and better bases. H2 becomes WEAK rather than NOT_SUPPORTED. |
| Spearman(b_g, ‖C_g‖) ≥ 0.9 and tightness p50 ≥ 0.8 | Certification approaches the oracle. The bound is not the problem, but the bytes are. |
| A base of ≤ 0.124 B/weight with err_base ≤ 0.1 appears (no known post-training method: 2-bit VQ such as QuIP# or AQLM costs about 0.25 to 0.3 B/weight) | (iii) becomes reachable. Extending H2 to attention (4.77 GB) becomes the next bottleneck. |
| A working drafter with high acceptance on Gemma 4 (E1 `spec_active`, identical greedy output) | Amortises the 17.64 GB read over several tokens. This is the more promising dense path, independent of H2. |

---

## 6. Cross-cutting: what is measured, what is projected, what is assumed (2026-10-05)

| Quantity | MEASURED | PROJECTED | ASSUMED |
|---|---|---|---|
| Desktop 26B / 31B decode | 15.18 / 2.44 tok/s (llama-bench, repack on) | — | — |
| Pi 5 memory bandwidth | (pi/03 pending) | — | 10.0 GB/s × 0.65 |
| Pi 5 storage | context-friend-1 microSD 91.6 MB/s sequential (others pending pi/03) | — | 85 / 380 / 430 / 850 MB/s sequential; 60 / 350 / 400 / 700 MB/s random 3 MB |
| Gemma 4 routing locality | (E3 pending) | — | PRIOR curve 0.60 / 0.74 / 0.84 / 0.88 |
| 26B on 8 GB + NVMe Gen3 | — | 1.94 tok/s serial, 2.87 overlapped | all inputs |
| 26B on 16 GB (in RAM) | (pi/05 on context-friend-1 running, UD-Q3_K_M) | 2.73 (Q4_0), 2.13 (UD-Q3_K_M) | efficiency 0.65 |
| 31B on 16 GB (3-bit in RAM) | **0.53 tok/s** (UD-IQ3_XXS, real Pi) | 0.55 | — |
| H2 residual fraction at ε = 3% | 1.00 on one real matrix with synthetic inputs (probe, section 5.9) | — | E7 pending |

---

## 7. Model limits, consolidated

The H1 time model counts bytes, not FLOPs. It ignores the KV reads (≤ 0.16 GB/token at 4096 for the
26B), the second RAM read of a missed slab, the second seek per slab, page rounding, readahead
inflation and thermal throttling. Sections 4.6 and 4.7 quantify each one; together they move the
8 GB + NVMe Gen3 projection by a few hundredths of a tok/s. The two unknowns that move it by tenths
are the effective storage bandwidth of the page-fault path and the ARM `-nr` kernel efficiency.
H2's arithmetic ignores the cost of computing `‖x_g‖` and of the base matmul itself; both are small
next to the bytes.

---

## 8. Constants used in this file (machine-readable; must equal pigemma/data/catalog.json)

The block below is parsed by `tests/test_docs_hypotheses.py` with the regex
`^([A-Za-z0-9_.\-]+) = (.+)$`. Each value is the JSON value of that catalog path: devices and models
are addressed by `name` / `id`. If the catalog changes, update this block and re-run the
projections of sections 3 to 5 with `pigemma.feasibility`. The test fails until the two agree.

<!-- pigemma-catalog-constants:begin -->
```
devices.pi5-4gb-microsd.ram_bytes = 4000000000
devices.pi5-4gb-microsd.os_reserve_bytes = 700000000
devices.pi5-4gb-microsd.compute_buffers_bytes = 300000000
devices.pi5-4gb-microsd.n_ctx = 2048
devices.pi5-4gb-microsd.mem_bw_gbps.value = 10.0
devices.pi5-4gb-microsd.storage.seq_read_mbps.value = 85
devices.pi5-4gb-microsd.storage.rand_read_3mb_mbps.value = 60
devices.pi5-4gb-microsd.storage.access_ms.value = 2.0
devices.pi5-8gb-usb3.ram_bytes = 8000000000
devices.pi5-8gb-usb3.os_reserve_bytes = 800000000
devices.pi5-8gb-usb3.compute_buffers_bytes = 300000000
devices.pi5-8gb-usb3.n_ctx = 4096
devices.pi5-8gb-usb3.mem_bw_gbps.value = 10.0
devices.pi5-8gb-usb3.storage.seq_read_mbps.value = 380
devices.pi5-8gb-usb3.storage.rand_read_3mb_mbps.value = 350
devices.pi5-8gb-usb3.storage.access_ms.value = 0.5
devices.pi5-8gb-nvme-gen3.ram_bytes = 8000000000
devices.pi5-8gb-nvme-gen3.os_reserve_bytes = 800000000
devices.pi5-8gb-nvme-gen3.compute_buffers_bytes = 300000000
devices.pi5-8gb-nvme-gen3.n_ctx = 4096
devices.pi5-8gb-nvme-gen3.mem_bw_gbps.value = 10.0
devices.pi5-8gb-nvme-gen3.storage.seq_read_mbps.value = 850
devices.pi5-8gb-nvme-gen3.storage.rand_read_3mb_mbps.value = 700
devices.pi5-8gb-nvme-gen3.storage.access_ms.value = 0.2
devices.pi5-16gb-nvme-gen3.ram_bytes = 16000000000
devices.pi5-16gb-nvme-gen3.os_reserve_bytes = 900000000
devices.pi5-16gb-nvme-gen3.compute_buffers_bytes = 400000000
devices.pi5-16gb-nvme-gen3.n_ctx = 4096
devices.pi5-16gb-nvme-gen3.mem_bw_gbps.value = 10.0
devices.pi5-16gb-nvme-gen3.storage.seq_read_mbps.value = 850
devices.pi5-16gb-nvme-gen3.storage.rand_read_3mb_mbps.value = 700
devices.pi5-16gb-nvme-gen3.storage.access_ms.value = 0.2
devices.desktop-8400f.ram_bytes = 68262760448
devices.desktop-8400f.os_reserve_bytes = 2000000000
devices.desktop-8400f.compute_buffers_bytes = 500000000
devices.desktop-8400f.n_ctx = 4096
devices.desktop-8400f.mem_bw_gbps.value = 60.0
devices.desktop-8400f.storage.seq_read_mbps.value = 3000
devices.desktop-8400f.storage.rand_read_3mb_mbps.value = 2000
devices.desktop-8400f.storage.access_ms.value = 0.1
storage_classes.microsd.seq_read_mbps.value = 85
storage_classes.microsd.rand_read_3mb_mbps.value = 60
storage_classes.microsd.access_ms.value = 2.0
storage_classes.usb-ssd.seq_read_mbps.value = 380
storage_classes.usb-ssd.rand_read_3mb_mbps.value = 350
storage_classes.usb-ssd.access_ms.value = 0.5
storage_classes.nvme-gen2.seq_read_mbps.value = 430
storage_classes.nvme-gen2.rand_read_3mb_mbps.value = 400
storage_classes.nvme-gen2.access_ms.value = 0.2
storage_classes.nvme-gen3.seq_read_mbps.value = 850
storage_classes.nvme-gen3.rand_read_3mb_mbps.value = 700
storage_classes.nvme-gen3.access_ms.value = 0.2
storage_classes.unknown.seq_read_mbps.value = 85
storage_classes.unknown.rand_read_3mb_mbps.value = 60
storage_classes.unknown.access_ms.value = 2.0
selection.stream_hit_rate_curve_assumed = [[0.15, 0.6], [0.25, 0.74], [0.35, 0.84], [0.45, 0.88]]
selection.viable_stream_tok_s = 1.0
selection.good_stream_tok_s = 2.0
models.26b-q4.bytes = 14439363584
models.26b-q4.nonexpert_bytes = 1577172088
models.26b-q4.expert_bytes = 12846366720
models.26b-q4.expert_unit_bytes = 3345408
models.26b-q4.expert_bytes_per_token_worst = 802897920
models.26b-q4.nonexpert_bytes_per_token = 1577172088
models.26b-q4.n_moe_layers = 30
models.26b-q4.n_expert = 128
models.26b-q4.n_expert_used = 8
models.26b-q4.breakdown_bytes.attn = 624885760
models.26b-q4.breakdown_bytes.head_tied_token_embd_q6_k = 605563904
models.26b-q4.breakdown_bytes.dense_ffn_shared_expert = 301086720
models.26b-q4.breakdown_bytes.router_norms_scales = 45635704
models.26b-q4.breakdown_bytes.expert_gate_up_per_expert_layer = 2230272
models.26b-q4.breakdown_bytes.expert_down_per_expert_layer = 1115136
models.31b-q4.bytes = 17651001568
models.31b-q4.nonexpert_bytes_per_token = 17635166448
models.31b-q4.breakdown_bytes.ffn = 11705057280
models.31b-q4.breakdown_bytes.ffn_per_matrix = 65028096
models.31b-q4.breakdown_bytes.attn = 4770160640
models.31b-q4.breakdown_bytes.head_tied_token_embd_q6_k = 1156076544
models.31b-q4.breakdown_bytes.norms_other = 3871984
models.31b-ud-iq3_xxs.bytes = 11837782976
models.31b-ud-iq3_xxs.nonexpert_bytes_per_token = 11821947120
models.26b-ud-q3_k_m.bytes = 12728497888
models.26b-ud-q3_k_m.nonexpert_bytes_per_token = 2399894648
models.26b-ud-q3_k_m.expert_bytes_per_token_worst = 644548608
```
<!-- pigemma-catalog-constants:end -->

Constants from code, not from the catalog. They are `pigemma.feasibility` module constants, and the
test reads them from the module: `LLAMA_EFFICIENCY` = 0.65 (ASSUMED; `results/pi/calibration.json`
overrides it), `KV_Q8_4K["26b"]` = 160000000 and `KV_Q8_4K["31b"]` = 420000000 (ASSUMED),
`COMPUTE_BUFFERS` = 300000000, `MIN_STREAM_CACHE` = 2000000000 (policy), `SLOW_TOK_S` = 0.3
(policy).

---

## 9. Sources

- [docs/design/FACTS.md](design/FACTS.md): verified facts, hardware options, the real-Pi update.
- [docs/design/manifest.json](design/manifest.json): `experiments[]` (E0 to E8 procedures and
  thresholds) and the design summary (repack facts with llama.cpp line references).
- [docs/research/RESEARCH_SUMMARY.md](research/RESEARCH_SUMMARY.md): Pi 5 numbers, MoE offloading
  priors, dense literature. URLs are in `docs/research/research-2026-10-04.json`.
- [results/desktop-baseline-2026-10-04.md](../results/desktop-baseline-2026-10-04.md): desktop
  measurements and speculative-decoding findings.
- [results/pi-real/README.md](../results/pi-real/README.md): context-friend-1 hardware and the 31B
  UD-IQ3_XXS run.
- [pigemma/data/catalog.json](../pigemma/data/catalog.json),
  [pigemma/feasibility.py](../pigemma/feasibility.py),
  [pigemma/gguf_map.py](../pigemma/gguf_map.py), [results/map26.json](../results/map26.json),
  [traces/h2/map31.json](../traces/h2/map31.json): byte budgets and the projection code.
- [docs/TRACE_FORMAT.md](TRACE_FORMAT.md): definitions of coverage@N, adjacent reuse, stack-distance
  LRU curve and precision@N used by E3 and E4.
- Procedures: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), [PI_RUNBOOK.md](PI_RUNBOOK.md),
  [README.md](../README.md).
