# the operator's instructions, 2026-10-05 (verbatim) — the current authoritative brief

Continue the existing pi-gemma4 project from its current state.

Read pi-gemma4/CLAUDE.md, inspect the implemented experiments and raw
results, and reuse the existing infrastructure. Do not restart the
project, rebuild completed components unnecessarily, or expand the
framework without a specific experimental need.

The next milestone is an evidence-backed decision about whether our
memory-management method improves inference when the model does not
fit in the available RAM.

Successful model loading and unit-test counts are not the research
result we need now.

1. Freeze and audit the current baselines

Associate every reported result with its exact configuration:
model repository and revision, GGUF filenames and hashes, quantization,
QAT provenance, llama.cpp commit, local patches, build flags, thread
counts, batch settings, context size, KV-cache settings, storage device,
and complete command line.

Resolve which exact model file and storage device produced the
0.55-0.79 tok/s memory-pressure baseline. Do not compare this number
against a different quantization, device, or workload.

Record whether each result came from a synthetic benchmark or actual
autoregressive generation. Record prompt length, generated-token
count, and whether reasoning tokens were enabled.

Distinguish cold model loading, prompt processing, decoding, and total
request latency. Identify existing measurements that are already
reproducible and rerun only what is missing or ambiguous.

Keep a known-good resident-memory configuration as a reference.

2. Make the constrained-memory experiment valid and safe

Treat the existing 8 GiB holdback experiment as a preliminary stress
test, not as proof of performance on a physical 8 GB Pi.

Use a dedicated cgroup v2 for the inference process and every helper
that holds model data. Launch them inside it before loading the model.
Start with an 8 GiB experiment memory budget, then test 6 GiB if feasible.

Account for mapped model pages, anonymous copies, KV cache, scratch
buffers, metadata, and predictors without double-counting shared pages.
Audit file-page ownership and prevent unaccounted warm pages outside
the experiment from providing hidden capacity.

Set swap policy explicitly for this experiment and record memory.high,
memory.max, memory.current, memory.peak, memory.stat, memory.events,
and available pressure metrics. Keep settings identical across methods.

An 8 GiB application budget is not an 8 GB whole-device claim.
A physical-device claim must also allow RAM for the OS and services.

Do not stop, reconfigure, or degrade ContextFriend services.
Do not run global drop_caches, swapoff, reboot, or unrestricted memory
pressure tests on the live device. Abort the experiment if service
health or safe resource headroom deteriorates.

Use only reversible experiment-local changes. If safe isolation is
unavailable, report that limitation instead of using a disruptive
workaround. Run only one performance experiment at a time on the Pi.

3. Complete the decisive expert-pinning comparison

Select one existing MoE GGUF and one storage device for the primary
comparison. Keep all inference settings fixed. Disable MTP and other
speculative decoding for this experiment.

Compare:
A. Unmodified llama.cpp using normal Linux caching.
B. The instrumented implementation with its caching changes disabled.
C. Static hot-expert pinning.
D. A size-matched random-expert pinning control.

Use B to detect instrumentation or implementation overhead. Use D to
check whether any improvement is really due to useful expert selection,
rather than extra prefaulting or a different warm-up procedure.

Build the hot-expert ranking from a separate calibration set. Do not
rank experts using the held-out evaluation requests.

Use a small pilot to choose a conservative pinning budget, then freeze
that choice before evaluation. Count actual unique locked pages,
including page-alignment overhead, rather than nominal expert bytes.

Verify that locking actually succeeds. Record requested and effective
locked memory. Include pinning helpers and any copied weights within
the same total memory budget.

Measure the non-expert working set and leave headroom for attention,
shared components, KV cache, and temporary buffers. Do not assume that
all remaining RAM is available for pinned experts.

Use consistent warm-up and cache-state procedures. Report the cost of
prefaulting and pinning separately, as well as amortized performance.
Do not hide setup costs in an untimed preparation stage.

For this exact-caching track, execute every expert selected by the
original router with unchanged weights and precision. A cache miss must
cause a fetch and computation, not expert skipping or substitution.

4. Use realistic workloads and measure the complete cost

Collect routing traces from real model-generated continuations, not
only random-token or fixed synthetic benchmark streams.

Separate calibration, tuning, and held-out evaluation data. Include
English, Hebrew, German, code, instruction following, extraction from
provided text, and abrupt topic changes within a conversation.

Begin with a small pilot. Expand only the baseline and promising
configurations, rather than running a large Cartesian product.

Include short prompts, longer prompts, and follow-up requests with
explicitly documented context reuse. Measure both cold and warm
conditions where they can be established safely.

For each run, save:
- Exact configuration, input/output token counts, and raw output.
- Model-load time, time to first token, prompt throughput, decode
  throughput, and total request time.
- Median and tail inter-token latency, including visible stalls.
- Peak accounted memory, physical read bytes, major faults, reclaim
  pressure, swap activity, temperature, and throttling status.

Separate startup, prefill, and decode I/O. Corroborate process
read_bytes with cgroup and device-level counters where available,
accounting for unrelated device traffic and helper processes.

Do not treat page-fault counts alone as physical byte counts.
Do not infer general full residency from one short zero-read run.

Randomize or alternate baseline/treatment order and repeat finalist
comparisons at least three times. Report per-workload results and
variation, not only the best aggregate throughput.

Measure tracing overhead. Use tracing-off runs for headline performance
once correctness and trace fidelity have been established.

For exact caching, compare deterministic output against the unchanged
reference. Investigate any divergence. Do not call a method exact merely
because its answers still look correct.

5. Validate the simulator, then decide whether a better policy is justified

Map each (layer, expert) to all relevant tensors and actual byte ranges
in the split GGUF files. Inspect the real layout rather than assuming
each expert is a single contiguous block.

Use the device's actual page size and account for shared boundary pages,
readahead, non-expert accesses, runtime repacking, and duplicate buffers.

Validate predicted read volume and stall behavior against real Pi
measurements. A plausible expert hit rate is not simulator validation.
Report unexplained discrepancies rather than adjusting parameters to
make the simulator appear accurate.

Compare simple recency/frequency policies with static pinning and an
offline future-informed replacement reference. For equal-size pages,
an optimal replacement simulation can bound replacement-policy
potential under its stated assumptions. Do not present it as an optimal
latency bound for arbitrary expert sizes and asynchronous I/O.

Use memory-capacity curves to identify where the opportunity exists.

If static pinning loses but the traces indicate substantial remaining
potential, implement one evidence-driven improvement: for example,
layer-aware allocation or an adaptive cache combining persistent hot
experts with recently useful experts.

Rank retention candidates by expected avoided fetch cost per retained
byte, not frequency alone. Use only information available at runtime.
Measure prediction, bookkeeping, and wasted-prefetch costs.

Consider storage-layout changes only when measured read amplification
justifies them. Keep those results separate from the policy comparison.

If the same-representation simulation leaves little headroom, stop
tuning that cache design and document the limit. This does not invalidate
other representations or the dense-model research track.

6. Run a bounded dense-model feasibility experiment on the desktop

Keep this separate from the exact MoE caching work. It is an approximate
computation research track unless stronger guarantees are demonstrated.

Use existing model files where possible. Clearly state whether the
reference is the original checkpoint or a particular quantized GGUF.

Start with representative early, middle, and late feed-forward layers.
Test one concrete representation first: an activation-aware low-rank
base plus block-organized residual corrections.

Fit the representation on calibration data only.

First verify that the complete base-plus-residual computation
reconstructs the chosen reference within the declared numerical
tolerance. Account separately for quantization and approximation error.

Then use offline informed residual selection to estimate how many
correction bytes and operations are needed on held-out activations.
Label this as a potential study, not a deployable predictor or a proven
optimal selection algorithm.

Measure base storage, residual storage, metadata, selection overhead,
actual CPU runtime, and data movement. Compare against the existing
quantized implementation, not only FP16.

If few corrections are sufficient, test whether a cheap online selector
can recover enough of that benefit. Then evaluate accumulating errors
with actual autoregressive generation, not only isolated layers receiving
perfect reference activations.

Do not build a full Pi residual-streaming engine before there is evidence
of both useful quality retention and plausible end-to-end savings.

A negative result for this factorization is a valid outcome. Explain
whether it failed because the base was too expensive, residuals were too
dense, selection was ineffective, or errors accumulated.

7. Keep quality and speculative decoding claims narrow

The existing 36 correct answers are a smoke test, not evidence that
3-bit and Q4_0 have equivalent quality.

Extend testing with harder held-out questions involving negation,
numbers, multilingual instructions, structured extraction, and
executable code checks. Use objective grading where possible and keep
model quality comparisons separate from exact-cache correctness checks.

For quantization comparisons, record checkpoint and QAT provenance.
Do not attribute differences solely to bit width when the sources differ.

Leave MTP off for the primary cache experiments. Preserve the measured
code-specific benefit as an optional configuration. Before recommending
it, account for draft-model memory, startup, verification time, accepted
tokens per cycle, and end-to-end latency.

Do not add an automatic MTP policy until the core memory experiment has
a clear result.

8. Deliver a decision, not another infrastructure progress report

Use these as project decision thresholds, not universal scientific rules.

A promising cache result should show at least 1.5x median paired decode
speedup over ordinary Linux caching on held-out requests at the same
memory budget, with unchanged computation and no substantial prefill
or tail-latency regression.

A stronger milestone is at least 2 tok/s and at least 2x the matched
constrained-memory baseline, within an honestly accounted budget.
This remains an experimental target, not a promised outcome.

Smaller gains can still be worthwhile, but label them incremental.
Report failures and workload regressions alongside wins.

Produce:
- A compact results table and raw machine-readable measurements.
- Exact reproduction commands and pinned code/model versions.
- Memory-budget versus throughput results.
- Physical-read volume versus performance results.
- Simulator-versus-device validation.
- A dense-residual feasibility verdict.
- A clear recommendation: continue, revise the method, or stop this
  particular approach, supported by the measured bottleneck.

Update pi-gemma4/CLAUDE.md with completed work, commands, artifacts,
limitations, and the next justified action.

Keep a brief prior-art comparison, including MoE-Infinity and the
Mira preprint of September 29, 2026. Distinguish GPU-memory offloading
from our CPU-only storage-to-RAM setting. Do not claim novelty merely
because we pin or predict experts.

Prioritize completing the controlled Pi cache comparison. Run the
bounded dense feasibility work independently on the desktop.

The final report must answer:
What improved, against which matched baseline, at what memory cost,
with what correctness or quality evidence, and what should we do next?


---

# UPDATE 2 — the operator's second brief, 2026-10-05 ~19:40 (verbatim below). Where it conflicts with the brief above, UPDATE 2 wins.

Key deltas to apply (orchestrator summary; the verbatim text follows): three SEPARATE tracks (A same-model constrained-memory; B practical local GenAI = product comparison where the model may change; C dense research) — never present success in one as success in another; external comparison targets with a status each (locally tested / documentation-only / blocked / not applicable): P0 upstream ggml-org/llama.cpp recent revision in a separate worktree + our pinned a7b94df, P0 LozzKappa/hotpin-llm (AGPLv3, AVX2, patches a specific llama.cpp revision; benchmark against ITS OWN unmodified parent revision; keep code isolated), P1 ikawrakow/ik_llama.cpp (Gemma GGUF + ARM support check), P1 SmallThinker-4BA0.6B (then 21BA3B) with Tiiny-AI/PowerInfer smallthinker runtime on the Pi (different-model group), P1 one smaller 2B-8B multilingual model on our runtime, P2 optional Picovoice picoLLM (AccessKey + internet licensing: documentation-only unless authorised credentials exist); research-only: MoE-Infinity, Mira, SVD-LLM, TEAL, LLM in a Flash. Track A first comparison = unmodified llama.cpp, our instrumented build with optimisation disabled, our static pinning, HotPin where compatible, size-matched random pinning; verify pinned pages are the pages the active kernels read (repacking!); if resident decoding has no storage reads, investigate CPU kernels / memory traffic / threads / prefill instead of storage. Workload groups: short Q&A; ~1,000 new input tokens; follow-up with reusable prefix (verify reuse from runtime evidence); multi-turn topic changes; longer input only after the pilot. Product track B: Gemma best config vs SmallThinker vs a smaller multilingual model, predefined rubric (executable code tests, schema validation, expected facts, deliberately-absent-information questions), first-use vs warm-session, targets ~3 s to useful content and ~20 s to complete a short answer (targets, not promises). Dense C: compare dynamic corrections vs the quantized baseline, base alone, and simple STATIC correction selection; the online selector must decide WITHOUT reading/computing every correction first. Main results table columns: model/quant; runtime/revision/method; memory budget + measured peak; storage device; workload + language; prefill tok/s; decode tok/s; time to first useful answer; complete-task latency; tail stalls; physical read bytes; correctness/task score; run count + variation; limitations. Include failed builds, unsupported configs, OOMs, regressions; keep vendor-reported numbers separate. Execution order: matched memory baseline + pinning comparison first; dense independently on the desktop; then runtime/model comparisons. Final question: What improved, compared with which alternative, at what resource cost, with what quality evidence, and why is the next development step justified?

(verbatim text: see docs/design/INSTRUCTIONS_2026-10-05_v2.md)
