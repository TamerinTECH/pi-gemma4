# Prior art: MoE expert caching / offloading vs. our setting (brief §8)

Written 2026-10-05 18:55 (stage P2-prior-art). **Re-verified and corrected 2026-10-05 ~22:40** after the workflow
resume; what changed is listed in the "Re-verification log" below. Every number here is **reported by the cited
authors**. We reproduced none of it.

How the sources were read: each page was fetched on 2026-10-05 and queried through an automated page-to-text
reader. Quoted figures can therefore carry small transcription errors. Items tagged *(new)* were added in the
resume pass and rest on that pass alone.

**Our setting, for comparison (track A, exact caching):**

* **Target device.** Raspberry Pi 5 16 GB, CPU-only (Cortex-A76 ×4), 16 KiB-page kernel, microSD 91.6 MB/s or
  USB stick ~70 MB/s (MEASURED by us).
* **Where the controlled comparison runs today.** On the desktop: Docker, cgroup **v1** memory controller,
  4 CPUs, **4 KiB pages**, storage throttled to 91.6 MB/s by a FUSE passthrough (stage P5). The Pi kernel boots
  with `cgroup_disable=memory`, so no matched-budget run is possible there yet.
* **Model.** Gemma 4 26B-A4B (128 experts, top-8, 30 layers). Primary file: unsloth UD-Q3_K_M (12.73 GB,
  2.63 MB per expert slice; stage P4). Second file: Google QAT Q4_0 (14.44 GB, 3.35 MB per slice).
* **Engine and method.** Unmodified llama.cpp a7b94df with `-nr`, so the GGUF stays one file-backed mmap and the
  kernel page cache is the expert cache. Method C is an external `mlock` daemon (`tools/pinner/pinner.c`) that
  pins the page ranges of chosen experts of that same file. Routing, weights, precision and kernels are
  unchanged (methods A/B/C/D of the brief).

## Comparison table

Setting tags: **GPU, DRAM→VRAM** = experts wait in host RAM and are copied over PCIe into GPU memory.
**CPU, storage→RAM** = experts wait on flash or disk and are read into RAM, and the CPU computes (our setting).

| System (id, date) | Setting | Mechanism | Computation exact? | Reported gain | …measured against |
|---|---|---|---|---|---|
| **MoE-Infinity** (arXiv 2401.14361; Xue, Fu, Lu, Mai, Marina; v1 2024-01-25, v3 2025-03-12) | **GPU, DRAM→VRAM** over PCIe 4.0. Full model in host DRAM (32 GB–1 TB per model); an RTX A5000 24 GB caches experts and computes. No SSD in the evaluation. | **Prediction.** Request-level Expert Activation Matrix (token counts per layer × expert), matched by cosine distance against a stored collection (EAMC, ~120 entries). It drives prefetch and an eviction priority with layer-proximity decay. | Yes as far as described: the router is untouched and every routed expert is fetched and computed. The paper does not state output identity explicitly. | 3.1–16.7× lower per-token latency at batch 1. DeepSeek-V2-Lite TPOT: 155 ms vs 485 (vLLM) / 737 (DeepSpeed) / 1250 (Mixtral-Offloading) / 2590 ms (Ollama). | **Other systems** (vLLM, Ollama, DeepSpeed, Mixtral-Offloading, BrainStorm): different engines, baseline precision not stated. No numeric same-engine LRU/LFU/static ablation. |
| **Mira** (arXiv **2609.38090** v1, **29 Sep 2026**; **Sanjali Yadav, Bahar Asgari**, Univ. of Maryland, College Park; cs.LG) | **GPU, DRAM→VRAM.** All experts sit as INT8 in CPU-pinned host memory and are staged over PCIe into VRAM; the GPU computes. RTX 6000 Ada 48 GB / RTX A5000 24 GB / RTX 3070 8 GB. No SSD. | Adaptive **HOT** set (rebalanced from token-level routing telemetry) plus a **STAGE** prefetch window filled by an offline-trained per-layer **predictor** (layer i → i+2). Custom INT8 expert format. | **Routing exact, weights not.** The router is untouched and a miss is fetched and computed, but experts are INT8 instead of FP16: WikiText-2 ppl 3.9324 → 3.9554, TriviaQA 0.57 → 0.54. | Throughput at 48 GB: 2.86× vs MoE-Infinity, 1.55× vs Fiddler, 1.25× vs HQQ, 10.46× vs DeepSpeed. At 24 GB: **5.71× vs MoE-Infinity**, 3.86× vs HQQ. TTFT at 48 GB: 11.71× vs Fiddler (the abstract's figure), 18.38× vs DeepSpeed, 7.49× vs MoE-Infinity. Beam search: 3.84× vs Fiddler. | **Other systems at other precisions** (HQQ 2-bit, Mixtral-Offloading 4-bit, others not stated), so the headline mixes a byte cut with policy. Own ablation, predictor/prefetch on vs off: 1.07× (48 GB) and 1.32× (24 GB) end to end; 1.08–1.49× across VRAM caps from 48 down to 16 GB. No numeric static-pinning or LRU ablation. |
| **EdgeMoE** (arXiv 2308.14352; Yi, Guo, Wei, Zhou, Wang, Xu; v1 2023-08-28, v2 2025-03-07) | **CPU, storage→RAM** on a Raspberry Pi 4B 8 GB with an SD card (SanDisk Ultra, speed not stated). Also Jetson TX2 (GPU; SSD 550 MB/s, HDD 50 MB/s) and a Xiaomi 14. Models: Switch Transformers (top-1), GPTSAN (top-2), MiniCPM-MoE. | Non-expert weights resident; experts on storage, fetched when activated into a budgeted buffer. **Prediction**: statistical preload of the likely next expert in an I/O-compute pipeline. Eviction by usage frequency and layer index. **Per-expert INT2/4/8 bitwidths.** | Routing exact, **weights not**: experts are re-quantized (loss bounded at 2 %, measured 0.89–1.95 %). | Pi 4B: 2.55–3.25×. TX2: 2.64–3.03×. Both are vs IO-EXP. Also 1.19–2.12× vs IO-QEXP and 1.22–2.77× vs STI. | IO-EXP = plain on-demand expert loading without expert quantization, so the gain mixes a byte cut with policy. IO-QEXP = the same with INT4 experts (the closest to a matched baseline). STI = a different pipeline. |
| **Qualcomm, "Mixture of Cache-Conditional Experts"** (arXiv 2412.00099; Skliar, van Rozendaal, Lepert, Boinovski, van Baalen, Nagel, Whatmough, Bejnordi; TMLR 06/2025) | **CPU, storage→RAM.** Modified llama.cpp with `mlock` on Snapdragon phones (12/16 GB, UFS flash); 4-/8-bit models; 30 or 45 of 60 experts per layer cached. | LRU cache plus **cache-prior re-routing**: in-cache experts get boosted logits when the top-k is chosen. | **No, approximate**: it changes which experts run. LRU misses 22–40 % → 7–21 %, at +0.1–3 % ppl and <0.1 % on MMLU/GSM8K. | ~2× decode on device ("2× speedups on mobile devices"). | The **same engine's plain LRU** at the same cache size (matched). |
| **SmallThinker** (arXiv 2507.20984; Song, Xue, Wei et al., 14 authors; 2025-07-28) | **CPU, storage→RAM** in the PowerInfer framework, with an 8 GiB (21B-A3B) or 1 GiB (4B-A0.6B) memory limit. PC i9-14900K, OnePlus 13, RK3588. Raspberry Pi 5 only with the full model in memory. Storage type and speed not stated. | **Model co-designed and trained** for offload: a pre-attention router lets the expert fetch overlap attention; hot experts in DRAM (LRU), cold on SSD. | Exact for its own model. It needs a new model, so it does not apply to Gemma 4. | 21B-A3B at 8 GiB: 20.30 tok/s (PC), 15.50 (OnePlus 13), 8.56 (RK3588). In memory on a Pi 5: 6.61 tok/s (21B-A3B), 28.77 (4B-A0.6B). | **A different model** in the same framework and quantization ("SmallThinker and other models using PowerInfer framework with Q4_0 quantization"): Qwen3-30B-A3B 10.11 tok/s (PC) and 0.18 tok/s (OnePlus 13) at 8 GiB. Not a cache-policy comparison. |
| **llama.cpp Discussion #27149**, "Expert-Aware SSD Streaming for MoE Models on Memory-Constrained Hardware" (@jerryjokesalot, 2026-08-15; TinyGiant) | **CPU, storage→RAM.** M1 MacBook Pro 16 GB, internal NVMe (2.97 GB/s sequential measured, 6.06 GB/s double-buffered), Qwen3-30B-A3B (128 experts, top-8). | Shared weights resident; only routed experts read. **Text-calibrated static pinning: 42 % hit vs 6 % for random**; calibrated + LRU 56 %. Expert-contiguous relayout (192 vs 6,912 pages per expert). | Routed experts are computed ("verified correct text" from a Python engine); no bit-identity check stated. | 4.7 tok/s (prototype with **simulated compute**, real I/O). 2.40 / 3.81 / 4.10 / 4.45 tok/s at 50 / 75 / 88 / 100 % hit (C + BLAS **expert-layer benchmark**, not end-to-end generation). | Its own simulated full-layer-read baseline (0.9 tok/s). **No comparison with stock llama.cpp mmap.** |
| **Cacheable by Design?** (arXiv 2608.18261; S. R. Suram; 2026-08-18; pre-registered negative result) | GPU desktop streaming from SSD: i9-12900, RTX 3070 8 GB, 32 GB DDR4, NVMe ~2.4 GB/s cold sequential. | Trains routers with a locality loss (137M–340M models) and stacks training-free cache-aware re-routing. | **No, approximate** (changes routing). | Locality loss: ~60 % fewer misses at +2.0–3.1 % ppl, so it **failed the pre-registered gate** (≥30 % fewer misses and ≤1 % ppl: "No weight satisfies both gates"). With re-routing: ~80 % fewer misses at ≤3.4 % ppl. | Same model, original router. Its measurement study on Qwen3-30B-A3B: next-token expert reuse 0.444 vs 0.223 chance; an LRU holding 13.4 % of experts serves 66 % of requests; 95 % of traffic flows through 52.5 % of experts. |
| **FlashMoE** (arXiv 2601.17063; Kim, Lee, Han, Yoo, Kim; 2026-01-22) | **GPU, SSD→VRAM under limited RAM.** RTX 5070 Ti 16 GB, 16 GB DDR5 (about 1 GB left free), PCIe 5 NVMe 7.4 GB/s. Non-expert weights in memory; experts on SSD, loaded on demand. | **Learned replacement policy**: a small feed-forward net on recency and frequency, trained toward Belady's choices. | Exact (policy only). | OLMoE-1B-7B hit rate +21 % vs LRU, +51 % vs LFU. **Speed +22 % (OLMoE) and +7 % (Qwen3-30B-A3B) vs LRU.** "Up to 2.6×" overall. | LRU/LFU inside the same engine (matched). The 2.6× is vs Fiddler and DAOP, which are other, DRAM-based systems. |
| **HOBBIT** (arXiv 2411.01433; Tang, Liu, Hou et al.; 2024-11; built on llama.cpp) | **GPU.** Jetson AGX Orin 32 GB unified; RTX 4090 24 GB + 256 GB host. Expert cache in GPU memory, other experts "offloaded to CPU memory or SSD". The cost it quotes is CPU memory → GPU over PCIe 4.0: ~80 ms to load one Mixtral layer vs ~3 ms to compute it. | Cache-miss experts judged less critical are **replaced by low-precision (int4/int2) copies**; next-layer prefetch; multi-criteria cache. | **No, approximate** (mixed-precision substitution, ≤1 % accuracy loss reported). | Orin (INT8): 13.0× (Mixtral) / 18.9× (Phi-MoE) decode vs llama.cpp; 3.64× / 9.93× vs MoE-Infinity. RTX 4090 (FP16): 3.21× / 3.29× vs MoE-Offloading. | Other systems. No exact-only ablation quoted. |
| **ProMoE** (arXiv 2410.22134; Song, Zhong, Chen, Chen; v3 2025-09-01) | **GPU, DRAM→VRAM.** RTX 4090 24 GB, i9-14900K, 128 GB DRAM, PCIe 4.0. No SSD. | **Learned predictor** with stride prefetch several layers ahead; chunked prefetch, early preemption, reordered inference. | Exact (prefetch only). | vs existing offloading systems: prefill 2.20× avg (up to 3.21×), decode 2.07× avg (up to 5.02×). **vs hand-crafted static/LRU caches: prefill 1.78× (up to 2.48×), decode 1.34× (up to 1.79×).** | Both are given; the static/LRU comparison is the matched one. |
| *Related:* **Who Should Own the Expert Cache?** (arXiv 2608.12103; Si, Lin, Li, Zhang; 2026-08-12, v2 2026-08-30) | Datacenter GH200, replayed traces, 128–896 experts per layer, a 1.45 TB expert pool larger than DRAM. **Kernel page cache vs a user-space arena.** | Kernel recency eviction plus page-cache admission, compared with an oracle static-frequency pinned arena. | Exact ("token-identical outputs"). | Page-cache admission: 1.09–1.10× steady decode. **"kernel recency serves essentially the same demand"** as the oracle static-frequency policy. The oracle-pinned arena stays 1.09–1.11× faster ("the cost of the page-cache hit and reclaim path"). | The same engine; matched. |
| *Related:* **llama.cpp PR #25294** "llama : stream MoE routed experts from disk" (freedomljc, 2026-07-04, **open, unmerged**) | Storage→device slot cache inside llama.cpp. `pread` thread pool, **O_DIRECT (bypasses the page cache)**. | Per-layer cache of `n_slots` expert slabs filled on miss; eviction by decaying route hotness with LRU tiebreak; waved prefill. | Claimed "bit-exact when both paths use the same kernels/ubatch". A tester later withdrew byte-identical parity for long generations (comment of 2026-09-06). | GLM-5.2 UD-Q2_K_XL (~254 GB) on a 128 GB Grace-Blackwell box: 73–79 % hit, decode 1.83–2.20 tok/s. Author's comment (2026-07-08): streaming 5.58 prefill / 2.08 decode tok/s vs mmap + `--cpu-moe` 1.06 / 0.87, i.e. 5.3× / 2.4×. | **Not compute-matched**: the streaming run computes experts on the GPU, the mmap run on the CPU. |
| *Related (new):* **routhjim/llama.cpp-lab PR #9** "ggml-cpu : add a bounded mlock'd cache for CPU-resident expert slabs" (personal fork, merged 2026-08-31, Linux only) | **CPU, page cache.** Experts are forced onto the CPU backend and read from the mmap'd file. | Size-bounded **LRU of `mlock`ed expert slabs in the page cache** (`GGML_EXPERT_CACHE_GIB`), with an asynchronous locking thread and an optional online successor predictor. | Exact by construction ("changes only whether pages are resident, never what the matmul reads"). | Counters only, 8 GiB cache: hit 767,711 / miss 84,641 (9.9 % miss), resident 8.0/8.0 GiB, lockfail 0. **"No runtime measurement in this PR yet"**: no tok/s and no A/B run. | Nothing. Hardware, storage and the exact model are not stated in what we read. |
| *Related (new):* **SSD-LLaMA** (arXiv 2609.18110; Liang, Shen, Hu, Xu, Gao, Xu, Meng; 2026-09-16; built on llama.cpp) | **GPU, SSD→RAM→VRAM.** RTX 5090 32 GB, 16–32 GB host RAM, NVMe up to 9 GiB/s; 143–544 GB 4-bit models. | SSD-native pipeline: O_DIRECT + io_uring into pinned host buffers (not the page cache), a separate contiguous "expert pack" file with lossless compression, a three-tier frequency + recency cache, CPU–GPU hybrid execution. | Exact ("executes every selected expert without pruning or substitution"). | Prefill 1.52–4.19×, decode 2.10–15.58×; above 1 tok/s on a 1T-parameter model. Ablation: the expert-pack layout gives 2.6× / 4.1× over reading the GGUF layout. | llama.cpp (mmap) and KTransformers on the same machine. Baseline placement details are not given in what we read. |

## Notes

* **Mira, found and verified.** "Mira: Memory-Efficient MoE Inference Using Adaptive Caching and Predictive Expert
  Staging", Sanjali Yadav and Bahar Asgari (University of Maryland, College Park), arXiv:2609.38090v1 [cs.LG],
  submitted 29 Sep 2026. The abstract page, the HTML full text and a web-search hit agree on title, authors, id
  and date.
  * **Setting.** Single-GPU VRAM management ("high-capacity MoE inference on a single GPU"). Experts sit in host
    DRAM as INT8 and the GPU computes. Main model Mixtral-8x7B, plus DeepSeek-V2-Lite-Chat. Workload: ShareGPT
    conversations with varied input and output lengths.
  * **Static pinning.** Mira argues that static, Fiddler-style pinning of popular experts fails when the workload
    drifts. It supports this with heatmaps, not with a numeric static-vs-adaptive ablation.
  * **What it does not cover.** Storage-to-RAM paging, the page cache, and CPU-only compute. No code repository
    is named in the paper.
* **MoE-Infinity is GPU-memory offloading too.** Its central observation is that at batch 1, fewer than 5 % of
  experts are activated repeatedly within one request for ~100-expert models (25 % for Mixtral). Each request
  has a small hot set of its own, and MoE-Infinity predicts it by matching against earlier requests. A hot-set
  ranking built from the evaluation requests themselves would inherit this effect, so ours must come from
  separate calibration data.
* **Systems closest to our setting** (the CPU computes and experts come from storage): EdgeMoE, Qualcomm,
  SmallThinker, Discussion #27149, routhjim PR #9, and HotPin and ik_llama.cpp `--prefetch-experts` from the X0
  extension below. Among them:
  * the only large throughput gain with a matched baseline is Qualcomm's ~2×, and it changes routing;
  * the exact ones report hit rates or counters (#27149: calibrated 42 % vs random 6 %; PR #9: 9.9 % miss with
    an 8 GiB mlock LRU), not matched end-to-end speed.
* **Matched, same-engine, exact policy gains are small.** Comparisons that keep the engine, the precision and
  the routing fixed report:
  * FlashMoE: +7 % (Qwen3-30B-A3B, the shape closest to Gemma 4 26B-A4B) and +22 % (OLMoE) over LRU;
  * ProMoE: 1.34× decode and 1.78× prefill over hand-crafted static/LRU caches;
  * Mira: 1.07–1.49× for prediction/prefetch on vs off, largest at the smallest VRAM cap (16 GB);
  * "Who Should Own…": ~1.1× for an oracle-pinned arena over the kernel page cache, with kernel recency serving
    about the same demand as oracle static-frequency pinning;
  * HotPin (X0 extension): flat for Gemma 4 26B-A4B in RAM, +45.5 % for a bigger-than-RAM model on fast NVMe
    (single runs).

  The published range for an exact policy change inside one engine is therefore roughly 1.0–1.5× for decode
  (best single case: ProMoE 1.79×). The brief's 1.5× decode threshold sits at the top of that range.
* **Larger exact gains exist, but they come from the I/O path and the file layout, not from the cache policy.**
  * SSD-LLaMA: 2.1–15.6× decode over its baselines (llama.cpp mmap, KTransformers). The contiguous expert-pack
    layout alone gives 2.6–4.1×.
  * PR #25294: 2.4× decode over mmap, with expert compute moved to the GPU.
  * ik_llama.cpp `--prefetch-experts`: prefill ×1.9–2.7 (X0 extension).
  * Discussion #27149: relayout cuts an expert from 6,912 scattered pages to 192.

  All of these ran on NVMe at 2.4 GB/s or faster, most with a GPU. For us they point at read amplification and
  layout. The brief (§5) allows layout changes only when measured read amplification justifies them, and
  separately from the policy comparison.
* **Where the 2–19× headlines come from.** One or more of: a different baseline system; fewer bytes per expert
  (INT8, 2-bit, mixed precision); approximate routing (re-routing, substitution); a model trained for locality;
  a GPU replacing CPU compute. **None of these is allowed in our exact-caching arms**, which share one GGUF
  file, one engine and the unchanged router.
* **Caveat in the other direction (an expectation, not a result).**
  * The ~1.1× results come from compute-heavy regimes (GPU, 2–9 GB/s NVMe).
  * At 70–92 MB/s, decode is I/O-bound, so tok/s scales roughly with 1 / (miss bytes per token). A hit-rate gain
    that moves the miss fraction from 0.40 to 0.25 would then give ~1.6×, but only if page-cache readahead and
    boundary pages do not dominate.
  * Prior art cannot tell which effect wins on Pi-class storage. That is what the A/B/C/D measurement is for.
* **Dense precedent, for track C only.** "LLM in a flash" (arXiv 2312.11514; Alizadeh et al., Apple; v3
  2024-07-30) runs dense models up to twice the DRAM size from flash. It uses "windowing" (reuse of recently
  activated neurons) and "row-column bundling" (contiguous flash reads), and reports 4–5× (CPU) and 20–25× (GPU)
  over naive loading. It is not MoE and relies on activation sparsity (RESEARCH_SUMMARY §4).

## What would be new in our setting, and what is NOT

**Not new; do not claim it:**
* **Exact routing with caching.** MoE-Infinity, ProMoE, FlashMoE, SSD-LLaMA, PR #25294, PR #9, HotPin and
  ik_llama.cpp all compute every routed expert.
* **Pinning or caching hot experts, and predicting or prefetching them.** Frequency, recency, adaptive and
  learned policies: MoE-Infinity, Mira (HOT set), EdgeMoE, ProMoE, FlashMoE, HOBBIT, Fiddler.
* **Calibrated static pinning beating random pinning** on a top-8-of-128 model, at the hit-rate level:
  Discussion #27149 (42 % vs 6 %).
* **CPU-only, storage-to-RAM expert streaming.** On a Raspberry Pi with an SD card: EdgeMoE (Pi 4B). On phones
  with UFS and `mlock` in llama.cpp: Qualcomm. On a laptop: #27149. By model design: SmallThinker.
* **`mlock` of expert pages in the page cache of the mmap'd GGUF.** routhjim PR #9 (LRU, inside the engine) and
  HotPin (static, inside the engine).
* **Using the OS page cache as the expert cache and comparing it with static pinning.** "Who Should Own the
  Expert Cache?".
* **Exact expert streaming inside llama.cpp, and expert-contiguous relayout.** PR #25294, SSD-LLaMA, #27149.

**Possibly new (narrow, a measurement contribution, valid only once measured):**
1. **The controlled comparison.** No source found has all of the following:
   * exact routing at a matched, honestly accounted memory budget, with setup costs reported;
   * an A/B/C/D design: unmodified llama.cpp with the Linux page cache, instrumented with caching off, static
     hot-expert pinning, and a size-matched random-pinning control (plus HotPin and ik prefetch as engines);
   * a modern top-8-of-128 decoder MoE (Gemma 4 26B-A4B);
   * an edge CPU with **microSD / USB-stick-class storage (70–92 MB/s)**.

   The nearest sources each miss a part. EdgeMoE has the Pi and the SD card, but with top-1 encoder-decoder
   models and re-quantized experts. #27149 has calibrated vs random pinning, but only as hit rates at ~3 GB/s.
   PR #9 has the mlock mechanism without any A/B run. HotPin has single runs without memory accounting. No MoE
   expert-streaming measurement was found on storage slower than ~2.4 GB/s other than EdgeMoE's SD-card and HDD
   runs.
2. **Engine-transparent pinning.** An external `mlock` on `MAP_SHARED` page-cache pages of the same GGUF pins
   experts with zero change to the inference engine, and counts locked pages exactly, including boundary pages
   shared between experts. This is an implementation and accounting detail, close to `vmtouch -l`, and PR #9 and
   HotPin already lock the same pages from inside the engine. It is a weak novelty claim at best.
3. **Gemma 4 26B-A4B locality data.** We found no published routing-locality or hit-rate data for this model.

**On the 16 KiB-page kernel specifically.** A 16 KiB page size is not new in itself: Discussion #27149 ran on
an Apple M1, and macOS on Apple silicon also uses 16 KiB pages (platform knowledge, not stated in the
discussion as we read it). No source read here reports page-exact locked-memory accounting on a 16 KiB
kernel. For us it is so far a map calculation (stage P4: +1.82 % page overhead per expert at 16 KiB for the UD
file), not a measurement under memory pressure, because the controlled runs use x86 with 4 KiB pages. It may be
claimed only after a matched run on the Pi itself.

Even (1) is conditional. The matched-budget comparison currently runs in the desktop emulation (Docker, cgroup
v1 memory controller, FUSE throttle at the Pi's microSD rate) until the operator decides how to enable a memory budget on
the Pi (CLAUDE.md, "Hard limitation"). A positive or negative result would be a data point for this hardware
class, not a new method.

## Sources (fetched 2026-10-05)

- Mira: https://arxiv.org/abs/2609.38090 (abstract), https://arxiv.org/html/2609.38090, https://arxiv.org/pdf/2609.38090
- MoE-Infinity: https://arxiv.org/abs/2401.14361, https://arxiv.org/html/2401.14361
- EdgeMoE: https://arxiv.org/abs/2308.14352, https://arxiv.org/html/2308.14352
- Qualcomm cache-conditional experts: https://arxiv.org/abs/2412.00099, https://arxiv.org/html/2412.00099v2
- SmallThinker: https://arxiv.org/abs/2507.20984, https://arxiv.org/html/2507.20984
- llama.cpp Discussion #27149: https://github.com/ggml-org/llama.cpp/discussions/27149
- Cacheable by Design?: https://arxiv.org/abs/2608.18261, https://arxiv.org/html/2608.18261
- FlashMoE: https://arxiv.org/abs/2601.17063, https://arxiv.org/html/2601.17063
- HOBBIT: https://arxiv.org/abs/2411.01433, https://arxiv.org/html/2411.01433
- ProMoE: https://arxiv.org/abs/2410.22134, https://arxiv.org/html/2410.22134
- Who Should Own the Expert Cache?: https://arxiv.org/abs/2608.12103
- llama.cpp PR #25294: https://github.com/ggml-org/llama.cpp/pull/25294
- routhjim/llama.cpp-lab PR #9: https://github.com/routhjim/llama.cpp-lab/pull/9
- SSD-LLaMA: https://arxiv.org/abs/2609.18110, https://arxiv.org/html/2609.18110
- LLM in a flash: https://arxiv.org/abs/2312.11514
- Also seen, abstract only, not analysed:
  - MoE-CORE (arXiv 2610.01950, 2026-10-01, cs.DC; Ke Yang et al.): host-to-device expert offloading on "a
    compact AI appliance"; staged layer loading, non-uniform cache capacity, routing-informed replacement,
    cross-layer prefetch; 38.0–44.8 ms vs 1268.9–1269.1 ms against vLLM configurations.
  - Cache-Aware Joint Router Adaptation (arXiv 2609.04895, 2026-09-04; Wu et al.): post-trains routers for
    GPU-memory caching, so it is approximate; 4.6–53.3 % less data movement than prefetching baselines.
  - ST-MoE (arXiv 2606.15453, 2026-06-13, cs.AR; Zhao, Bunescu, Louri, Karanth, Wang): spatio-temporal expert
    prediction "that preserves the original routing behavior" with a reconfigurable hardware design.
- Titles seen in search results only, not read: OD-MoE (arXiv 2512.03927), DuoServe-MoE (arXiv 2509.07379),
  AdapMoE (arXiv 2408.10284), FineMoE (EuroSys '26), llama.cpp Discussion #24528 / Issue #20757 / PR #27861 (GPU
  VRAM expert caches), `vpozharski/moe-cache` (search hit; the repository returned HTTP 404 today).

## Re-verification log (resume pass, 2026-10-05 ~22:30–22:40)

* **Confirmed unchanged:** Mira (id, title, authors, date, abstract, hardware, INT8 accuracy figures, ablation
  range), MoE-Infinity, ProMoE, FlashMoE, Cacheable by Design?, Discussion #27149, "Who Should Own…". The
  figures previously tagged *(sweep)* for ProMoE, FlashMoE, Cacheable by Design? and SmallThinker were re-read
  from the papers, and the tag was removed.
* **Corrected:**
  * PR #25294: the earlier text said that no mmap comparison existed and gave the bit-exact claim without
    qualification. Both were wrong (see its row).
  * EdgeMoE: the gains were attributed to the wrong baselines, and the Pi 4B figure was missing.
  * Mira: "TTFT up to 18.38×" is the DeepSpeed figure; the abstract's 11.71× is vs Fiddler.
  * HOBBIT: the paper says "CPU memory or SSD", not CPU memory only.
  * SmallThinker: the baseline ran in the same framework; the Pi 5 in-memory figures were added.
  * "Our setting": the primary file is UD-Q3_K_M (stage P4), and the desktop emulation is cgroup v1 with a FUSE
    throttle and 4 KiB pages, not cgroup v2.
* **Added:** routhjim PR #9, SSD-LLaMA, the "LLM in a flash" note, the I/O-path note, the 16 KiB paragraph.
* **Not re-read in this pass:** the body figures of the Qualcomm paper (miss rates, cache sizes; only its
  abstract was re-read), the 192 vs 6,912 page figure of #27149, and the X0 extension below (HotPin,
  ik_llama.cpp), which belongs to stage X0.

## Extension (stage X0 of the comparisons workflow, 2026-10-05 ~20:20): GPU offloading vs CPU-only storage-to-RAM

Brief UPDATE 2 §2 asks which MoE-Infinity / Mira assumptions are GPU-specific and which transfer to our setting.
Paper numbers stay "reported by the authors". The Pi arithmetic below is PROJECTED from our MEASURED constants:
microSD 91.6 MB/s; UD-Q3_K_M expert slice 2,632,964 B; resident Pi decode ~3.0-3.2 tok/s, i.e. ~10-11 ms per layer.

**How the two settings differ (orders of magnitude):**

* **Fetch link.** One UD expert slice takes ~29 ms to read at 91.6 MB/s on the Pi, assuming sequential speed.
  The same 2.6 MB crosses PCIe 4.0 x16 (~25 GB/s) in ~0.1 ms, about 270× faster. On a GPU, an expert copy is
  comparable to one layer's compute. On the Pi it is about 3 layers of compute, and every missed expert-slice is
  a synchronous page fault with 16 KiB granularity plus readahead, not an asynchronous DMA.
* **Who controls the cache.** GPU systems place and evict whole experts in VRAM slots themselves. Our cache is the
  kernel page cache, which caches pages, not experts, and can hold partially resident experts. We can steer it only
  with `mlock`/`munlock` (pin), `madvise` WILLNEED/COLD/PAGEOUT, `fadvise`, or our own `pread` (prefetch).
* **Spare resources.** GPUs have copy engines that run beside compute. The Pi has 4 cores, all used by decode, so a
  prefetch helper thread competes with compute unless it is purely asynchronous I/O (io_uring / readahead).
* **Workload.** MoE-Infinity and Mira report serving throughput with batching on ShareGPT-like requests. We run
  batch-1 interactive decode plus long prefills.
  * A prefill of ~900 tokens routes to almost every expert. On a non-resident model it streams nearly the whole
    10.3 GB expert set (~113 s at 91.6 MB/s, PROJECTED).
  * GPU papers hide this behind PCIe. On CPU it calls for full-tensor streaming at high queue depth, as in
    ik_llama.cpp's lookahead.

**What does NOT transfer:**

- VRAM capacity planning and PCIe/DMA overlap arithmetic. Mira's 2-layer lookahead hides at most ~20 ms of Pi
  compute, which is less than one slice read, so lookahead alone cannot hide a miss on microSD.
- Mira's INT8 re-quantisation of experts. It cuts bytes, so it is not exact and is excluded from track A.
- Batching-driven throughput gains, and GPU kernel fusion / CUDA graphs.
- MoE-Infinity's GPU-resident expert slots and their eviction mechanics.

**What transfers (as algorithms, at page granularity):**

1. **Hot set + staging split** (Mira HOT+STAGE): reserve part of the RAM budget for pinned hot experts (method C,
   HotPin L1) and leave the rest to the kernel's recency cache. Mira rebalances the HOT set from token telemetry
   with bounded swaps. That maps onto the brief's preferred "cost-aware, layer-aware cache with hysteresis" (an
   `munlock`/`mlock` churn budget per N tokens). Its cost is syscalls plus possible re-reads. Build it only if the
   validated simulator shows headroom beyond static pinning.
2. **Layer-ahead prediction** (Mira i→i+2, ProMoE stride, MoE-Infinity EAMC): a CPU predictor can drive
   `pread`/WILLNEED for predicted slices. On microSD it helps only by raising I/O queue depth and overlapping I/O
   with compute; it cannot cover a whole fetch. Wasted reads and helper-thread CPU must be counted (brief §4).
3. **Priority with layer-proximity decay, and request-level reuse statistics** (MoE-Infinity): MoE-Infinity
   reports that <5 % of experts are reused repeatedly within a request for ~100-expert models. So within-request
   caching gives little, and the cross-request hot set plus recency carries the value. Our calibration-only
   ranking already follows this; data leakage must be avoided.
4. **Telemetry is cheaper on CPU.** Misses per expert can be measured directly, from `mincore()` on the P4 byte
   ranges or from page-fault / `read_bytes` deltas. GPU systems have to estimate them.

**Directly relevant implementations found in X0 (not in the table above; they narrow our novelty further):**

* **HotPin** (LozzKappa/hotpin-llm, AGPLv3 + commercial; July 2026).
  * Mechanism: routing-frequency static pinning of expert slices with `mlock`/`VirtualLock`, inside llama.cpp
    (parent a6647b1), plus Windows-only prefetch hints. Exact computation.
  * Gemma 4 26B-A4B was tested, but only in RAM: flat, +0.0 % / −0.1 % (reported).
  * The bigger-than-RAM gain is +45.5 % decode, for gpt-oss-120b on NVMe faster than 1 GB/s (reported, single runs).
  * Consequence: our method C (external `mlock` of calibrated hot experts) is essentially HotPin L1 done from
    outside the engine. Novelty item 2 above ("engine-transparent pinning") is at most an implementation detail.
* **ik_llama.cpp `--prefetch-experts`** (PR #2101, merged 2026-07-11). It runs a `pread` worker pool that warms
  the page cache for mmap'd experts (Linux). Decode enqueues routed slices once routing ids exist; prefill streams
  the next splits in full, then marks them `MADV_COLD`. Exact. Reported PP ×1.9–2.7 when the model exceeds RAM
  (NVMe, GPU host); TG change "likely noise". Also `--defer-experts` (#1634), which faults expert pages on demand
  at load.
* **llama.cpp `llama_prefetch_rows`** (PR #29599, merged 2026-09-30, already in our a7b94df). It applies
  `MADV_WILLNEED` to per-layer-embedding rows (Gemma 4 E2B/E4B, Qwen4Exp), not to experts.

**Updated "not new" list:**

- routing-frequency static pinning via `mlock` inside llama.cpp, Gemma 4 26B included (HotPin);
- exact `pread`-based expert prefetch into the page cache in a llama.cpp-family engine (ik_llama.cpp).

What remains possibly new is unchanged and still conditional on measurement: the matched, honestly accounted
A/B/C/D(+HotPin, +ik prefetch) comparison on microSD/USB-class storage (70–92 MB/s) with Gemma 4 26B-A4B
locality data.

Sources (fetched 2026-10-05): https://github.com/LozzKappa/hotpin-llm (clone ac52bd9a: README, REPRODUCIBILITY,
patches, docs/paper/main.tex, results/all_benchmarks.json); https://github.com/ikawrakow/ik_llama.cpp/pull/2101,
/pull/1634, clone 3d27f5bb (`ggml/src/ggml-moe-prefetch.h`, `docs/parameters.md`);
https://github.com/ggml-org/llama.cpp/pull/29599; https://github.com/EfficientMoE/MoE-Infinity (README: CUDA GPU
required); https://arxiv.org/abs/2609.38090 (no code repository found by web search).
