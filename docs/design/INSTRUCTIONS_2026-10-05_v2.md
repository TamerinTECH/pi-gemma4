# the operator's second brief, 2026-10-05 ~19:40 (verbatim). Where it conflicts with INSTRUCTIONS_2026-10-05_decision.md, this one wins.

## Instructions for the coding agent

Continue the existing `pi-gemma4` project. Read `pi-gemma4/CLAUDE.md`, inspect the latest commits and raw results, and reuse the existing benchmark infrastructure.

**The next objective is to establish whether our implementation offers a measurable advantage over existing solutions, and identify which improvement is worth developing further.**

Do not restart completed work. Do not expand the framework simply to increase coverage or test counts. Prioritize a small number of decisive, reproducible experiments.

### 1. Keep three different objectives separate

| Track | Question | What counts as progress |
|---|---|---|
| **A. Same-model, constrained-memory inference** | Can we execute the existing Gemma model faster when its working set does not fit in RAM, without changing its computation? | Better performance than the strongest matched baseline, with the same weights, quantization and memory budget. |
| **B. Practical local GenAI** | Which available model and runtime provide the best useful answers on this Pi? | Better task quality, response latency and resource use. Changing the model is allowed, but must be clearly identified. |
| **C. Dense-model research** | Can a cheap base plus selectively evaluated corrections reduce actual computation and data movement? | Measured savings beyond ordinary quantization, with documented quality retention and selection overhead. |

**Do not present success in one track as proof of success in another.** A smaller model answering faster is a product improvement, not evidence that we accelerated Gemma. A cache improvement is not evidence that we reduced the model's arithmetic.

Use our reported results as starting points, not as independently verified constants. In particular, audit the exact configuration behind the approximately 3.4–4.1 tok/s resident-memory result and the 0.55–0.79 tok/s memory-pressure result.

---

### 2. Check these comparison targets

#### Priority comparisons

| Priority | Target | What to check |
|---|---|---|
| **P0** | **Upstream `ggml-org/llama.cpp`** | Preserve our current known-good revision. Also evaluate one recent compatible revision in a separate worktree. This separates our algorithmic gain from upstream improvements. |
| **P0** | **`LozzKappa/hotpin-llm`** | Compare routing-guided expert pinning against our implementation. Its published requirements specify AVX2 or newer, and its patches target a particular llama.cpp revision. Verify ARM compatibility rather than assuming it is a drop-in Pi solution. [GitHub](https://github.com/LozzKappa/hotpin-llm) |
| **P1** | **`ikawrakow/ik_llama.cpp`** | Check whether our exact Gemma GGUF and ARM CPU configuration are supported. The fork lists additional quantizations, performance work and Gemma 4 functionality, but that does not establish Pi compatibility or a speed advantage on our workload. [GitHub](https://github.com/ikawrakow/ik_llama.cpp) |
| **P1** | **SmallThinker with `Tiiny-AI/PowerInfer`** | Start with `SmallThinker-4BA0.6B-Instruct`. Test `SmallThinker-21BA3B-Instruct` afterward if the first comparison is useful and storage permits. Use the documented `smallthinker` runtime and conversion requirements. Treat these as different-model comparisons. The project publishes Pi measurements, which we must reproduce locally rather than quote as our results. [GitHub](https://github.com/Tiiny-AI/PowerInfer/blob/main/smallthinker/README.md) |
| **P1** | **One smaller multilingual model supported by our existing runtime** | Select one current 2B–8B-class candidate after checking its model card, license and language support. Prefer an already available model. Determine whether Gemma's extra capability actually justifies its latency for our tasks. |
| **P2, optional** | **`Picovoice/picollm`** | Check supported Pi models, available model files and deployment restrictions. Its documentation lists Raspberry Pi support, but also requires an AccessKey and internet connectivity for license validation. Do not classify that automatically as fully air-gapped operation. Test only with credentials already authorized for this project. [GitHub](https://github.com/Picovoice/picollm) |

**Do not spend the whole iteration porting competitors.** For each target, first perform a compatibility and licensing check. If substantial porting is required, record the blocker and continue with the other experiments.

For HotPin, distinguish an original implementation benchmark from a locally adapted version. Where different llama.cpp revisions are necessary, benchmark each implementation against its own unmodified parent revision. Do not attribute a revision difference to the caching algorithm.

Inspect licenses before incorporating external code. HotPin currently declares AGPLv3 and offers separate commercial terms. Keep comparison code isolated, preserve notices and flag licensing questions before product integration. [GitHub](https://github.com/LozzKappa/hotpin-llm)

#### Research comparisons, not mandatory ports

Review **MoE-Infinity** and **Mira** for expert-cache allocation, prediction and staging. Record which assumptions concern GPU memory and which could transfer to CPU-only storage-to-RAM operation. Do not treat their published hardware results as Pi baselines. [arXiv](https://arxiv.org/abs/2401.14361)

For the dense track, compare our proposed mechanism with **SVD-LLM**, **TEAL** and **LLM in a Flash**. Specifically distinguish low-rank compression, activation sparsity, and storage-aware weight access from our proposed dynamic residual selection. A conceptual resemblance does not mean their implementations support Gemma directly. [arXiv](https://arxiv.org/abs/2403.07378)

For every external target, record one status: **locally tested, documentation-only, blocked, or not applicable**.

---

### 3. Establish fair comparison conditions

#### Freeze the experiment configuration

Record the model repository and revision, all GGUF shard hashes, exact quantization, QAT provenance, engine revision, patches, compiler flags, thread counts, batch settings, context length, KV-cache settings, chat template, reasoning configuration and storage device.

For same-model comparisons, keep these settings matched wherever possible. If a runtime requires a different quantization or model conversion, place that result in a separate comparison group.

Use a small tuning set to select supported runtime settings, then freeze them before evaluation. Do not tune each method on the final test requests.

Keep speculative decoding disabled for the primary memory experiment. Preserve the existing MTP results as a separate configuration, not another uncontrolled variable.

#### Use an honestly accounted memory budget

Start with the existing unrestricted Pi reference and an **8 GiB experiment budget**. Add 6 GiB only if the first experiment is safe and informative.

Use cgroup v2 for the inference process and every helper holding model data. Launch them inside the group before loading the model. Audit existing file-page ownership, since moving a process does not automatically move its previous memory charges. Set and record the experiment's swap policy explicitly. [Linux Kernel Documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html)

Include model pages, pinned pages, copied or repacked weights, KV cache, buffers, predictors and metadata. Do not double-count the same shared pages.

**An 8 GiB application budget on a 16GB Pi is not proof of operation on a physical 8GB Pi.** Report host headroom separately and reserve whole-device claims for an appropriate validation.

Do not stop or reconfigure ContextFriend services. Do not use global `drop_caches`, `swapoff`, reboots, unrestricted memory ballast, or destructive storage tests. Use reversible experiment-local changes and stop if service health deteriorates.

#### Measure complete requests

Use a small pilot to eliminate broken configurations. Then evaluate finalists on held-out requests in English, Hebrew and German, including short chat, structured extraction, reasoning with numbers and negation, code, and sudden topic changes.

Keep calibration and evaluation data separate. The existing 36 questions remain a smoke test, not evidence of equivalent model quality.

Use the following workload groups.

| Workload | What it establishes |
|---|---|
| Short question with short answer | Basic interactive latency. |
| Approximately 1,000 new input tokens | Whether prefill makes document-based requests impractical. |
| Follow-up with a reusable conversation prefix | The benefit of verified context reuse. |
| Several sequential turns with topic changes | Cache adaptation and growing-context behavior. |
| Longer input, only after the pilot succeeds | Scaling and memory pressure beyond short demonstrations. |

Collect routing traces from genuine model-generated continuations. Synthetic benchmark streams can remain useful for isolated engine measurements, but do not use them alone to infer realistic expert locality.

For different-model comparisons, use the **same task content and scoring rubric**, not merely equal token counts. Report actual token counts because tokenization differs. Output limits are safety limits, not proof of equal work.

Save cold loading time, time to first visible answer content, prefill throughput, decode throughput, complete-task latency, inter-token stalls, memory, physical read volume, temperature and throttling status. Count reasoning tokens and their latency separately.

Use process I/O counters alongside cgroup or device counters where available. `read_bytes` is intended to reflect storage-layer reads, unlike logical read counts. Document unrelated traffic and helper processes. [Linux Kernel Documentation](https://docs.kernel.org/filesystems/proc.html)

Run only one performance benchmark at a time on the Pi. Alternate or randomize baseline and treatment order, and repeat finalist comparisons at least three times. Report variation and failures, not only the best run. Measure energy only if suitable instrumentation exists, otherwise mark it unmeasured.

---

### 4. Finish the memory experiment before adding another complex policy

The first controlled comparison should include:

1. Unmodified llama.cpp and our instrumented build with optimization disabled.
2. Our current static pinning and HotPin, where compatible.
3. A size-matched random pinning control.

Use the same calibration data for frequency-based policies. Generate rankings for the exact model under test rather than reusing an unrelated published hot-expert list.

Verify that the pages being pinned are the pages actually accessed by the active kernels. Check whether weight repacking or copied buffers make the original mapped ranges irrelevant.

Count actual unique locked pages, including alignment overhead. Record failed lock requests. Use matched warm-up procedures and report the cost of profiling, prefaulting and pinning, as well as steady-state performance.

For this track, **every expert selected by the original router must still execute with unchanged weights and precision**. A cache miss triggers fetching, not skipping an expert.

Within a matched engine configuration, validate deterministic outputs and inspect divergences. Across different engines, numerical differences must be investigated and documented rather than automatically labeled as a cache correctness failure.

#### Use the simulator to choose the next algorithm

Map each layer and expert to the actual tensors and file ranges accessed, including split GGUF files, shared pages and repacked representations.

Validate the simulator against observed physical reads and stalls. A plausible hit rate is not sufficient.

Estimate the remaining opportunity using a future-informed replacement reference under explicit assumptions. Do not call an expert-level heuristic a proven optimal latency bound.

If there is substantial headroom beyond static pinning, implement **one** evidence-driven improvement. The preferred candidate is:

> A cost-aware, layer-aware cache that reserves non-expert headroom and adjusts retention according to expected avoided fetch cost per physical byte.

Consider persistent hot experts, recent use, and prefill-versus-decode demand. Include hysteresis to avoid repeatedly changing allocations for small fluctuations.

Prefetching and storage-layout changes should follow only when measured access patterns justify them. Record wasted reads and predictor overhead.

**If normal decoding already generates no storage reads, stop attributing its resident-memory speed limit to storage.** Investigate CPU kernels, memory traffic, thread settings and prefill separately.

If the cache does not outperform the strongest available matched baseline, report that result. Do not keep adding complexity merely to obtain a favorable benchmark.

---

### 5. Run the product comparison independently

Compare our best Gemma configuration with SmallThinker and the selected smaller multilingual model.

The question is not "which model has more parameters?" It is:

> Which configuration completes our actual tasks correctly, within acceptable latency and memory?

Predefine the grading rubric. Use executable tests for code, schema validation for structured output, and expected facts for extraction. Include questions where information is deliberately absent and the model should not invent it.

Measure warm-session chat and first-use requests separately. Verify reused prompt state from runtime evidence rather than assuming it occurred.

As initial **product targets**, evaluate whether a short warm-session request can deliver useful answer content within about three seconds and complete a short answer within about twenty seconds. These are targets, not promised results or universal usability standards.

Report whether a smaller model meets the quality floor. If it does, it may be the better deployment choice, while the large-model research continues separately.

Do not hide long prefill, preprocessing or indexing costs. Show both first-use and amortized costs.

---

### 6. Keep the dense-model experiment small and decisive

Use the desktop for the "base plus streamed corrections" feasibility study. Do not build a full Pi engine yet.

Choose representative early, middle and late feed-forward layers. Test one concrete activation-aware low-rank base plus block-organized residual representation.

Compare the proposed dynamic correction scheme against the existing quantized implementation, the base alone, and a simple static correction selection. Use the same chosen reference weights and calibration split.

First verify that base plus all corrections reconstructs the reference within the declared numerical tolerance. Then examine how many correction bytes are required on held-out activations.

An offline informed selector can estimate potential, but its cost must not disappear from a deployment claim. The online selector must make useful choices **without reading or computing every correction first**.

Account for base storage, residual storage, metadata, selection, arithmetic and actual CPU runtime. A lower matrix error is not enough.

Proceed to autoregressive quality testing only if the layer-level results show plausible savings. This is necessary to expose accumulated errors rather than repeatedly supplying perfect reference activations.

**Compare against our working quantized baseline, not only FP16.** If this factorization fails, identify the failure mechanism and stop that implementation path. Do not generalize one negative factorization result to every possible residual method.

---

### 7. Define what would establish differentiation

For the memory track, a strong next result would be:

**At least 1.5× median paired decode improvement over the strongest available matched same-model baseline at the same constrained-memory budget, with preserved computation and no substantial prefill or tail-latency regression.**

Reaching at least 2 tok/s within an honestly accounted 8 GiB experiment budget is an additional practical target. It is not sufficient by itself if an existing method already achieves it.

These are project decision thresholds. Smaller gains can be useful, but should be described as incremental.

Before making a general claim, reproduce a promising improvement on a second compatible MoE model or architecture. Keep that expansion conditional on a clear first result.

For the product track, differentiation means a better **quality-latency-memory trade-off**. For the dense track, it means actual savings beyond quantization after accounting for the cost of approximation and selection.

Do not claim novelty simply because we use pinning, expert prediction, low-rank factors or sparse computation. State exactly which mechanism differs from the closest prior implementation and which measurement supports its value.

---

### 8. Deliver results and a decision

Reuse existing result directories and add only what is necessary. Produce a comparison registry, frozen experiment manifests, raw measurements, reproduction commands and a concise decision report.

The main results table should contain:

```text
Model / quantization
Runtime / revision / method
Memory budget and measured peak
Storage device
Workload and language
Prefill throughput
Decode throughput
Time to first useful answer
Complete-task latency
Tail stalls
Physical read bytes
Correctness or task score
Run count and variation
Limitations
```

Include failed builds, unsupported configurations, out-of-memory failures and regressions. Keep vendor-reported numbers separate from locally measured results.

Update `pi-gemma4/CLAUDE.md` with completed work, artifacts, limitations and the next justified action.

**Execution order:** finish the matched memory baseline and pinning comparison first, perform the bounded dense feasibility experiment independently on the desktop, then expand the strongest runtime and model comparisons.

The final report must answer:

**What improved, compared with which alternative, at what resource cost, with what quality evidence, and why is the next development step justified?**
