# Research summary (2026-10-04)

Distilled from a 6-topic web sweep with an adversarial fact-check pass (12 agents, 741 tool calls;
full structured output with per-claim URLs in `research-2026-10-04.json`). Numbers below are the
ones the repo's scripts and projections hard-code. "Measured" = read from a primary source that
measured it; "derived" = computed here from measured inputs.

## 1. Gemma 4 (confirmed)

| | 31B dense | 26B-A4B MoE |
|---|---|---|
| Params (total / active) | 30.70B | 25.23B / 3.82B |
| Layers / hidden / FFN | 60 / 5376 / 21504 | 30 / 2816 / dense 2112 + 128 experts x 704 (top-8) |
| Attention pattern | 5 sliding (window 1024, 16 KV heads) : 1 global (4 KV heads, head_dim 512, K==V) | same pattern, 8 / 2 KV heads |
| Vocab / context | 262144 tied / 256K | same |
| Official QAT Q4_0 GGUF | 17.65 GB (`google/gemma-4-31B-it-qat-q4_0-gguf`) | 14.44 GB (`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`) |
| Google's own Q4_0 memory figure (weights +20%) | 17.5 GB | 14.4 GB |
| License | Apache 2.0, repos ungated | same |
| llama.cpp | arch `gemma4`, PR #21309 (2026-04-02); MTP drafter arch `gemma4-assistant` (PR #23398) | same; experts stored as one 3-D tensor per layer, gate+up merged (`ffn_gate_up_exps` [2816,1408,128], `ffn_down_exps` [704,2816,128]); the dense FFN is the "shared expert" (no `*_shexp` tensors) |

MoE byte budget at Q4_0 (derived from the GGUF header): 3.345 MB per expert-layer, 428 MB per layer,
12.85 GB for all 3840 expert-layers (89% of the file), ~1.6 GB non-expert. One token visits 240
expert-layers = **803 MB of expert weights if nothing is cached**.

## 2. Raspberry Pi 5 (measured by others)

| Item | Number | Source type |
|---|---|---|
| RAM bus | 32-bit LPDDR4X-4267, 17.07 GB/s theoretical | official |
| RAM, usable for decode | ~10 GB/s effective in llama.cpp; tinymembench memcpy 4.8-6.0 GB/s; STREAM triad 11.5-12.4 GB/s single-thread, 5.6-7.7 GB/s at 4 threads | measured |
| RAM SKUs | 1 / 2 / 4 / 8 / 16 GB (16 GB since 2025-01; price risen to $305 by 2026-04) | official |
| microSD read | ~89-92 MB/s large-block (SDR104 bus 104 MB/s); `dtparam=sd_cqe` helps 4K IOPS 2-4x, not MB/s | measured |
| USB 3.0 SSD read | 360-416 MB/s | measured |
| NVMe over the FPC | Gen2 x1: 417-454 MB/s; Gen3 x1 (`dtparam=pciex1_gen=3`, uncertified): 784-894 MB/s | measured |
| CPU ISA | Cortex-A76, Armv8.2-A: dotprod + fp16 yes; i8mm / SVE / BF16 no. llama.cpp `GGML_NATIVE=ON` auto-detects; manual: `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`. Native vs generic build: 17.8 vs 1.1 tok/s on a 1B model | measured |
| Page size | 16 KiB default kernel on Pi 5 (`kernel_2712.img`); llama.cpp mmap handles it; 16 KiB is the minimum fault unit when streaming | official |
| Published llama.cpp tok/s (Q4_K_M class) | 1-1.5B: 10-18; 3B: 4.6-6.0; 7-8B: 2.3-2.8 | measured |
| MoE on a 16 GB Pi 5 | Qwen3-30B-A3B at a custom 2.70 bpw quant (13.3 GB class): ~8 tok/s tg, 8-10 pp (`--no-mmap -c 4096`); 3.92 bpw: 5.3 tok/s | measured |
| Thermals | throttles from 80 C; bare board under 4-core load exceeds 85 C; the $5 Active Cooler holds 62-66 C | measured |
| PCIe | ONE PCIe 2.0 x1 FPC: an AI HAT (Hailo) and an NVMe HAT cannot share it without a dual-slot HAT | official |

## 3. MoE expert offloading, what the literature says to expect

* LRU hit rate vs fraction of experts resident (average over 4 models, SeqMoE): 15% -> 60%, 25% -> 74%,
  35% -> 84%, 45% -> 88%. Predictive caches: 85% at 15%, 95% at 35%.
* Qwen3-30B-A3B (same 128-expert / top-8 shape as Gemma 4 26B-A4B): P(expert reused next token) = 0.444
  vs 0.223 chance; an LRU cache holding 13.4% of experts serves 66% of fetches; 95% of traffic goes
  through 52.5% of experts (expert frequency is skewed). llama.cpp discussion #27149 streamed it from
  NVMe on a 16 GB M1 at 3.8-4.7 tok/s with 56-88% hit rates.
* Plain LRU on phone CPUs still misses 22-40% of fetches with 30-45 experts/layer cached (Qualcomm);
  cache-aware re-routing cuts that to 7-21% at 0.1-3% perplexity cost.
* Every GPU-era system (Mixtral offloading, MoE-Infinity, Fiddler, HOBBIT, ExpertFlow, ProMoE,
  ktransformers) assumes a GPU and host-DRAM bandwidth; none streams at edge-storage speeds.
  SmallThinker shows a model co-designed for storage prefetch does 15.5 tok/s on a phone.
* Derived for Gemma 4 26B-A4B: streamed MB per token at hit rate 60/75/85/92% = 322/201/121/64 MB.
  IO-bound tok/s ceilings: microSD 0.28/0.45/0.74/1.4; USB3 1.1/1.7/2.9/5.5; NVMe Gen3 2.5/4.0/6.6/12.5.

## 4. Dense models far larger than RAM, what the literature says

* Nested / progressive precision exists post-training (Any-Precision LLM bit-planes, BitStack ~1-bit
  residual blocks, AQLM codebook prefixes, MatQuant with training). They give the "base resident +
  residual elsewhere" layout at little bit penalty, but **nobody gates residual fetches on the current
  decision** - that part of hypothesis H2 is open.
* 2-bit vector quantization (QuIP#, AQLM, VPTQ, QTIP) is viable at 13-70B at +18-25% perplexity = a 2x
  traffic cut. Post-training low-rank (SVD-LLM, ASVD, SliceGPT) is dominated by 2-bit VQ per byte.
* Training-free activation sparsity for GELU/SiLU dense models caps at ~40-50% (TEAL, CATS); the
  90-97% FFN sparsity behind Deja Vu / PowerInfer / LLM-in-a-Flash needs ReLU models or ~150B tokens
  of continued pretraining.
* Largest published per-token traffic cut for a dense model without training: SubSpec-style
  self-speculation (4-bit resident draft built from the model's own layers, lossless): ~27.5 accepted
  tokens per verification on Qwen2.5-32B, 0.52 -> 6.33 tok/s (12.5x). Requires the draft in RAM.
* Margin guarantees: top-1 survives iff the top-2 logit gap exceeds 2x the maximum rounding error, but
  only proven for the scoring layer; ~15-19% of decode steps are margin-fragile (MarginGate).
* Derived for a 31B at 4 bit (17-19.6 GB per token): 22 s/token at 800 MB/s; 2-bit VQ 10.5 s; 2-bit +
  50% sparsity 5.4 s. Only speculative verification amortises below ~1 s/token, and only with a resident
  draft and high acceptance.

## 5. Accelerators: no

Hailo-8/8L have no DRAM and cannot run LLMs; Hailo-10H (AI HAT+ 2, 8 GB) tops out at 1.7-2B models and is
slower than the Pi 5 CPU for decode; Coral has no LLM path; Axera M.2 cards cap at 7B. For a 25-30B
model the only thing that works on a Pi 5 is the CPU in a 16 GB board, and only for a ~4B-active MoE.

## 6. What this means for the experiment (the honest version)

* **26B-A4B MoE**: the realistic Pi tier. Fits entirely in RAM only on a 16 GB Pi 5 (at Q4_0 it is
  tight; a ~3-bit quant is comfortable). On 8 GB it must stream experts: ~43% of experts fit in a
  ~5.5 GB cache, so expect LRU hit rates in the 75-88% band and 2-4 tok/s from NVMe Gen3, under 1 tok/s
  from microSD. The expert trace + cache simulation in this repo measures Gemma 4's own locality instead
  of borrowing Qwen's.
* **31B dense**: cannot reach "good speed" on any Pi 5 with known post-training techniques; the
  staged-residual experiment here quantifies how much of the residual an oracle actually needs, which is
  the number that decides whether H2 is worth pursuing further.
