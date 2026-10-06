# Experiment plan (E0-E8)

**Status:** pre-registered on 2026-10-05, before E1-E8 produced any data. llama.cpp is pinned at
commit a7b94df (build b11398) on every host.
**Authoritative sources:** [docs/design/FACTS.md](design/FACTS.md) (verified facts and measured
numbers), [docs/design/manifest.json](design/manifest.json) (`experiments`, `run_order_today`, `risks`,
`data_formats`), [docs/TRACE_FORMAT.md](TRACE_FORMAT.md) (file formats).
**Companion documents:** [docs/HYPOTHESES.md](HYPOTHESES.md) (the arithmetic behind H1 and H2),
[docs/PI_RUNBOOK.md](PI_RUNBOOK.md) (the Raspberry Pi procedure), [README.md](../README.md).

This plan fixes, before the numbers exist, what each experiment runs, what it measures, which result
counts as a pass, and what would change the conclusion. Each experiment id E0-E8 is also the
`experiment` field of the results envelopes those runs write (`EXPERIMENTS` in
[pigemma/results.py](../pigemma/results.py)). Each experiment heading below is exactly the heading that
`python -m pigemma.results summarize` renders for that experiment (`## E<n>: <EXPERIMENT_TITLES[E<n>]>`),
so a rendered results file and this plan line up section by section.

## Contents

- [How to read this plan](#how-to-read-this-plan)
- [Ground rules for every run](#ground-rules-for-every-run)
- [Overview](#overview)
- [Run order](#run-order)
- [Controls](#controls)
- [E0: desktop baseline and repack gate](#e0-desktop-baseline-and-repack-gate)
- [E1: speculative decoding via llama-server](#e1-speculative-decoding-via-llama-server)
- [E2: quality (KLD, PPL, QA)](#e2-quality-kld-ppl-qa)
- [E3: MoE routing trace](#e3-moe-routing-trace)
- [E4: cache simulation](#e4-cache-simulation)
- [E5: page-cache validation](#e5-page-cache-validation)
- [E6: hot-set pinning proxy](#e6-hot-set-pinning-proxy)
- [E7: H2 dense oracle](#e7-h2-dense-oracle)
- [E8: Pi runtime dry run](#e8-pi-runtime-dry-run)
- [Results template and rendered outputs](#results-template-and-rendered-outputs)
- [Deviations log](#deviations-log)

## How to read this plan

### Three kinds of number

| Kind | Meaning | Section title in rendered results |
|---|---|---|
| `measured` | Observed on the host named in the row, with the result file that holds it. | MEASURED (kind=measured) |
| `projected` | Computed from measured inputs plus listed assumptions; not observed on hardware. | PROJECTED (kind=projected) |
| `assumed` | Taken as given (literature, catalog, vendor numbers). A calculation whose inputs are all assumed is also `assumed`, not `projected` (the rule in pigemma/results.py). | ASSUMED (kind=assumed) |

- An **emulated** host (qemu arm64, `PIGEMMA_EMULATED=1`) gives functional evidence only. Its speed
  numbers are never Pi measurements.
- A **Docker proxy** (E5, E6) measures a Linux page cache on a virtual disk inside Docker Desktop's VM.
  It validates the cache model, not a Pi's NVMe.
- Numbers measured **before this plan was frozen** are MEASURED reference values, always cited with
  their files: [results/desktop-baseline-2026-10-04.md](../results/desktop-baseline-2026-10-04.md),
  `results/specsrv-*.json` and [results/pi-real/README.md](../results/pi-real/README.md). They are not
  results of E0-E8 until a driver that writes envelopes re-runs them.
- Simulated quantities (cache-simulator hit rates from a measured trace) appear in the PROJECTED
  column. The catalog calls their source `simulated`.

### Experiment ids and files

- The `experiment` field of every envelope is one of `E0`..`E8` or `adhoc`. The manifest's long ids
  (for example `E0-desktop-baseline-and-repack-gate`) normalise to the short ones
  (`results.normalize_experiment`).
- Where outputs are written:

| Path | Written by | Contents | In git |
|---|---|---|---|
| `results/<date>/` | [pc/desktop_bench.ps1](../pc/desktop_bench.ps1) via [pigemma/bench.py](../pigemma/bench.py) and [pigemma/quality.py](../pigemma/quality.py) | E0-E2 envelopes, `E0_GATE.md`, `membw_desktop.json` (D5) | yes |
| `results/quality/` | pigemma/quality.py | `ref_<model>.logits` (1.61 GB each), `qa_<model>_<variant>.json` | logits: no |
| `traces/26b/` | [pc/trace_collect.sh](../pc/trace_collect.sh) | `*.moetrace` plus `.tokens.bin` sidecars (no), `.gen.txt`, `.meta.json`, `all.stats.json` (yes) | partly |
| `results/h1/` | [pigemma/cache_sim.py](../pigemma/cache_sim.py) (including `io-point`), [pc/pagecache_limit.sh](../pc/pagecache_limit.sh), `trace stats --csv-dir` | `sim.json`, `sim.csv`, `E4_*.json` envelopes, `validation*.json`, `io_*.csv`, `hotset_*.json`, `*.ranges.txt`, stats CSVs | yes |
| `traces/h2/` | [pc/h2_run.ps1](../pc/h2_run.ps1) | `map31.json` (yes), `acts/31b/*.npy` and `decomp/*.npz` (no) | partly |
| `results/h2/` | [pigemma/h2/oracle.py](../pigemma/h2/oracle.py) | `oracle.json` | yes |
| `results/pi/` | `pi/03`, `pi/05`, `pi/06` on the Pi | `io_mem.json`, `calibration.json`, `bench_*.json`, `pin_ab.json` | yes |
| `docs/results/` | `results summarize`, `h2.oracle --md` | `DESKTOP_<date>.md`, `H1_RESULTS.md`, `H2_RESULTS.md`, `PI_<date>.md` | yes |

### Pre-registration rule

- The blockquotes under **Hypothesis**, **Procedure** and **Pre-registered thresholds**, and the left
  column of each **Metrics** table, are copied verbatim from `manifest.json` `experiments` (frozen on
  2026-10-05). Everything else is this plan's operational detail.
- A threshold marked **(added by this plan)** is not in the manifest. It is pre-registered here.
- A change made after data exists is only allowed by appending a row to the
  [Deviations log](#deviations-log): date, reason, and whether the affected numbers had already been
  seen. A threshold is never edited in place.

## Ground rules for every run

1. **One model-loading job at a time on the desktop.** Every model run, Docker runs included, holds
   the bench lock `results/.bench.lock`, which refuses while another job holds it or when less than
   20 GB of RAM is free. The drivers take the lock themselves. By hand:

   ```powershell
   python -m pigemma.paths lock --status
   python -m pigemma.paths lock --hold --min-free-gb 20 -- <command...>
   ```

2. **Idle machine for every speed number.** Before a speed run, check that no other llama process is
   running (`tasklist /FI "IMAGENAME eq llama*"` and `docker ps`). A run that overlapped another job
   is labelled in `notes` and never used against a threshold.
3. **Disk is printed before every heavy step:**
   `python -m pigemma.paths free-disk D:/ --gb`. At planning time D: had about 32 GB free and C: about
   26 GB (the Docker VM disk lives on C:). The KLD reference logits (1,610,612,736 B per model) need
   2x free space. Optional quant downloads need 1.3x. Docker caches are never pruned; that is the operator's
   decision.
4. **`-nr` is the only page-cache streaming mode at a7b94df** (`--repack 0` in llama-bench). The
   default repack copies every Q4_0 tensor out of the mmap into anonymous memory
   (llama-model.cpp:1849-1870). `-ot "ffn_(gate_up|down)_exps=CPU"` still repacks
   (llama-model-loader.cpp:1240-1242), so it is only ever a negative control. No streaming experiment
   (E5, E6, Pi streaming) starts before the E0 repack gate has passed.
5. **Greedy decoding** (temperature 0) for every quality and losslessness comparison. The prompt and
   corpus files ([data/prompts/](../data/prompts/qa_set.json), [data/calibration/](../data/calibration/README.md))
   are fixed and never rewritten.
6. **Honesty labels:** every envelope carries `kind`. Container and emulated hosts are tagged in the
   rendered host column. A projection whose inputs are still assumed prints the word `assumed`.
7. **Pinned build:** every envelope records `llama_build` and `llama_commit`. A different llama.cpp
   build invalidates the E0 gate until E0 is re-run.
8. **The real Pi `context-friend-1`** runs production ContextFriend services. Work happens there only
   with `PIGEMMA_I_KNOW=1` and under these safety rules:
   - builds, downloads and inference run under `nice -n 10`, and downloads also under `ionice -c3`;
   - never stop, restart or kill the ContextFriend or Hailo services;
   - no mlock of model weights;
   - at least 2 GB stays free on the root filesystem;
   - one model on disk at a time;
   - no edits to `/boot/firmware/config.txt`, no reboot and no apt without the operator's explicit approval.

   See E8 Stage B and D1.
9. **Re-runs:** drivers skip a stage whose result file exists, unless `-Force` is given. Result files
   are never deleted to "start clean": a re-run gets a new `run_id`, and the summariser shows both.

## Overview

| Id | Question | Driver | Runtime | Needs | Main output |
|---|---|---|---|---|---|
| E0 | Do both models run, how fast, and does `-nr` keep the file mapped? | `pc/desktop_bench.ps1 -Stage bench` | 35 min | models | `results/<date>/E0_GATE.md` |
| E1 | Do MTP, DFlash or n-gram drafts engage, help, and stay lossless? | `pc/desktop_bench.ps1 -Stage spec` | 30 min | E0 | `results/<date>/E1_*.json` |
| E2 | Is `-nr` numerically neutral, and do the models answer EN, HE and arithmetic? | `pc/desktop_bench.ps1 -Stage quality` | 40 min | E0 | `results/<date>/E2_*.json` |
| E3 | Is real expert routing skewed and sticky enough to cache? | `pc/trace_collect.sh` | 30-35 min | probe build | `traces/26b/all.stats.json` |
| E4 | Which hit rate and bytes per token at each Pi budget, and what tok/s does that project to? | `python -m pigemma.cache_sim` | under 10 min | E3 | `results/h1/sim.json` |
| E5 | Does the simulator predict real page-cache bytes per token under a memory cap? | `pc/pagecache_limit.sh` | 45 min plus a one-time copy | E0 gate, E3, E4 | `results/h1/validation.json` |
| E6 | Does mlock-pinning the hot set beat plain page-cache LRU? | `pc/pagecache_limit.sh --pin` | 20 min | E5 | `results/h1/validation_pin.json` |
| E7 | Can staged residuals cut the dense model's bytes per token usefully? | `pc/h2_run.ps1` | 45-55 min | probe build | `results/h2/oracle.json` |
| E8 | Are the Pi scripts correct, and what does the real Pi do? | pytest, `pi/*.sh` | 10 min (A), about 1 h (B) | feasibility, pi scripts | test run, `results/pi/` |

## Run order

### From the design manifest (verbatim)

- 0\. Orchestrator (10 min): freeze docs/TRACE_FORMAT.md, pigemma/data/catalog.json and the results envelope from this manifest; commit the existing results/ and data/calibration; create results/&lt;date>/.
- 1\. Group 0 in parallel (5 agents, ~45 min): pyproject + paths + results | trace.py | gguf_map.py (run `build` on both real GGUFs immediately: expect 802,897,920 B/token and the verified constants) | catalog.json + prompts + TRACE_FORMAT.md | tests/conftest + synth + tools/CMakeLists + membw.c.
- 2\. Group 1 in parallel (5 agents, ~60 min): LANE A probe.cpp + pc/docker_run.sh build-tools (run `--mode names` on the 26B first, then --smoke trace, commit the golden fixture) | LANE B bench.py and start E0 on the 26B natively right away (E0 gate decides everything downstream) | LANE C cache_sim.py validated on synthetic traces | LANE D h2/decompose.py + a dequant sanity run on one real 31B tensor | LANE E feasibility.py + pi/lib/common.sh + fixtures (E8 tests).
- 3\. Group 2 (~45 min): quality.py (then E2 KLD base for both models runs in the background under the bench lock, serialized after E0) | h2/oracle.py on synthetic acts | pc/trace_collect.sh --quick on the 26B in Docker (E3, ~25 min, bench lock) | pc/pagecache_limit.sh (volume copy can start as soon as E0's repack gate passed and the Docker VM disk check passes) | pi/\* scripts.
- 4\. Group 3 (~60 min): desktop_bench.ps1 -Stage spec,quality (E1, E2) natively while E3 finishes in Docker only if free RAM >= 20 GB (bench lock enforces) | pi/00-05 + h2_run.ps1 (E7 act-dump on the 31B in Docker runs after E3 releases the lock; decompose/oracle run natively afterwards, ~35 min).
- 5\. Integration (~60 min): E4 cache_sim + stats -> docs/results/H1_RESULTS.md; E5 pagecache_limit --mem 3,4,5,6,8 then E6 --pin at 5,6 GB (Docker, bench lock); E7 oracle -> docs/results/H2_RESULTS.md; pi/06 + E8 checks.
- 6\. Write-up (~30 min): python -m pigemma.results summarize results/&lt;date> -> docs/results/DESKTOP_&lt;date>.md; README + HYPOTHESES + EXPERIMENT_PLAN + PI_RUNBOOK finalized with MEASURED / PROJECTED / ASSUMED tables, the repack finding with line references, the Pi recommendation (16 GB + NVMe Gen3) and the plain dense-31B statement; commit results JSON + md (traces/npy/logits stay untracked).
- Rules: never two model-loading jobs at once (results/.bench.lock + 20 GB free-RAM check, Docker runs included); the 31B-dependent steps (E0-31B, E7 act-dump) are scheduled last in their lanes; every heavy script prints free disk first (D: 32 GB, C: 26 GB); nothing is ever run on context-friend-1.

### Adjustments on 2026-10-05

- **Already done before this plan:**
  - the idle-machine desktop baseline (2026-10-04) and the ad-hoc speculative-decoding server run;
  - llama.cpp a7b94df binaries for Windows and Linux x86, plus the emulated arm64 build
    ([results/pi-emu/README.md](../results/pi-emu/README.md));
  - `membw` and `pinner` built in Docker (`tools/build-linux-x86/bin`);
  - the probe compiled, with its 72-token golden trace checked in
    (`tests/fixtures/golden_26b_72tok.moetrace`);
  - `results/map26.json` and `traces/h2/map31.json` built.
- **`context-friend-1` is now the real target Pi** (16 GB RAM, microSD only, Wi-Fi). The manifest's
  rule that nothing is ever run there is superseded for that host (D1). E8 Stage B runs there under the
  safety rules in ground rule 8. One run is already measured: the 31B UD-IQ3_XXS at 0.53 tok/s
  ([results/pi-real/README.md](../results/pi-real/README.md)).
- **Critical path** (each step that loads a model holds the bench lock; the Pi is a separate lane with
  one model job at a time):
  1. E8 Stage A unit tests. They need no model and can run at any time.
  2. E0 on the 26B, whose repack gate decides E5, E6 and Pi streaming. Then E0 on the 31B.
  3. Probe `--mode names` on the 26B, then `trace_collect.sh --smoke`, then `--quick` (E3, about
     25 min).
  4. E1 and E2 on the native Windows binaries. Run E2's `kld-base` early, because E7 also needs
     `data/calibration/all.txt`.
  5. E7: capture activations from the 31B in Docker, then run decompose and the oracle natively.
  6. E4, as soon as E3 has finished.
  7. E5, then E6, in Docker.
  8. E8 Stage B on the Pi, once the `pi/*.sh` scripts exist. It is independent of the desktop lane.
  9. Write-up: [Results template and rendered outputs](#results-template-and-rendered-outputs).

## Controls

Each control guards against one specific way an experiment could produce a confident wrong answer. A
failed control stops the conclusion it protects, not the whole plan.

### C1: hybrid_negctrl (E0, E2)

- **What:** the variant `hybrid_negctrl` adds `-ot "ffn_(gate_up|down)_exps=CPU"`.
- **Why:** this looks like the obvious way to repack attention and keep the experts mapped, but at
  a7b94df it is not:
  - an override to the CPU buffer type re-runs weight-buffer selection over `buft_list_cpu`, which
    still contains CPU_REPACK (llama-model-loader.cpp:1240-1242);
  - `-ot` can only name default device buffer types (arg.cpp:255-260).

  The control proves this on the build in use, so no streaming result silently depends on it.
- **Pass:** in E0, the hybrid_negctrl completion run logs a
  `load_tensors: CPU_REPACK model buffer size` line. In E2, its KLD against the default run is
  essentially zero, because it computes the same thing.
- **If it fails:** with no CPU_REPACK line, the loader no longer behaves as verified. Re-read
  `create_tensor` in llama-model-loader.cpp at the commit actually built before using `-ot` for
  anything. Until then it stays a control.

### C2: KLD of -nr vs default (E2, protects E3, E5, E6 and the Pi)

- **What:** `python -m pigemma.quality kld --variant norepack` against default-variant reference
  logits, on 6 chunks of 512 tokens of `data/calibration/all.txt`.
- **Why:** `-nr` switches to the non-repacked Q4_0 kernels. Every streaming result assumes this changes
  speed, not output. The E3 routing traces are also collected with `-nr`: if the logits differed,
  near-tie expert choices could flip.
- **Pass:** KLD mean < 0.001 and same-top-p > 99.5% for both models (manifest E2).
- **If it fails:** every streaming number carries the caveat. E3's routing is cross-checked on one
  calibration file traced without `-nr`. In Docker that costs the 14 GB anonymous repack copy, which
  fits in the 31 GB VM.

### C3: teacher-forced vs generated routing (E3, E4, E7)

- **What:** the probe records phase 0 (prefill of the calibration text, teacher-forced) and phase 1
  (the model's own generated tokens). E3 compares them with `per_layer[].prefill_vs_gen_js_bits` in
  `stats.json`. E7 reports the oracle per phase (`prefill`, `gen`, `all`).
- **Why:** the routing of token t depends only on its prefix, so teacher-forced routing equals
  generation routing for the same text. But generated text is a different distribution from the
  corpus, and a Pi generates.
- **Pass (added by this plan):** the mean over layers of `prefill_vs_gen_js_bits` is at most 0.10 bits,
  with at least 2,000 generated tokens in the trace. Jensen-Shannon divergence between empirical
  distributions is biased upward on small samples: the 8 generated tokens of the golden trace give
  0.40-0.49 bits. So the rule does not apply to tiny traces.
- **If it fails:**
  - E4 also simulates the `gen_prompts` input of the merged trace alone (`--token-range` over that
    segment, which is mostly generated tokens), and the H1 verdict must hold there too.
  - E7's verdict must hold on the `gen` phase as well as on `all`.

### C4: stack-distance vs simulator (E3, E4)

- **What:** two independent implementations of the same LRU hit rate:
  - `pigemma.trace stats` derives `global.lru_hit_curve` from a stack-distance histogram;
  - `pigemma.cache_sim` runs an OrderedDict LRU.
- **Pass:** at capacity C items with zero warm-up, `overall.hit_rate` equals `lru_hit_curve[C-1][1]`
  within 1e-9 ([docs/TRACE_FORMAT.md](TRACE_FORMAT.md) section 5.2). The overall block counts every
  token, so the run's warm-up setting does not matter. This is unit-tested on synthetic traces
  (`tests/test_cache_sim.py`). cache_sim also performs the check on every uniform-size LRU run: the
  result is `sim.json` `checks.lru_vs_stack_distance`, and a mismatch exits 1 (`--no-stack-check` is
  never used for E4). E4 re-checks it independently from `sim.json` (command there).
- **If it fails:** one of the two implementations is wrong. No E4 number is used until it is fixed.

### C5: iso-byte competitor for H2 (E7)

- **What:** at byte budgets equal to a uniform q2 read, a uniform q3 read, and the oracle's bytes at
  eps 0.03, compare three things:
  - plain uniform group quantisation;
  - base plus oracle-selected residuals;
  - base plus certified residuals.
- **Why:** staged residuals only matter if they beat simply storing a smaller quant at the same number
  of bytes.
- **Rule:** NO_ADVANTAGE if the uniform q3 error is at most 1.2x the oracle error at equal bytes
  (manifest E7).

### C6: ffn_out sanity (E7)

- **What:** the relative error of `W_down_q4 @ ffn_geglu_t` against the captured `ffn_out_t`, median
  over tokens, for each sampled layer.
- **Why:** it catches orientation, scale and dequantisation bugs (wrong `ne` order, wrong nibble
  order) before any H2 number is trusted. llama.cpp quantises activations to Q8 for this matmul, so a
  small error is expected.
- **Pass:** median < 0.03. The pipeline is VALID only together with `soundness_violations == 0`. Above
  0.1, the run aborts.

### C7: standing controls

- **Belady (E4):** the offline optimum. It is reported only as the upper bound of any cache policy,
  never as a projection.
- **Warm-up exclusion (E4: 256 tokens, E5: 200 tokens):** the cold start and the
  MAP_POPULATE/WILLNEED load storm (llama-mmap.cpp:493-515) stay out of the steady-state bytes per
  token. Overall numbers are reported as well.
- **Idle machine and one model job (all speed runs):** ground rules 1 and 2.
- **Golden trace (E3):** `tests/fixtures/golden_26b_72tok.moetrace`, written by the real probe, is
  validated by the Python reader in the unit tests. Format drift between C++ and Python is caught
  before E3.
- **Emulation is not speed (E8):** qemu arm64 runs prove the scripts and the aarch64 build. Their
  tok/s are never compared with a threshold.

## E0: desktop baseline and repack gate

| Item | Value |
|---|---|
| Manifest id | `E0-desktop-baseline-and-repack-gate` |
| Results `experiment` | `E0` |
| Driver | [pc/desktop_bench.ps1](../pc/desktop_bench.ps1) `-Stage bench`, which runs [pigemma/bench.py](../pigemma/bench.py) |
| Host and models | desktop, native Windows binaries in `third_party/llama.cpp-win`; `26b-q4`, `31b-q4` |
| Runtime (manifest) | ~35 min (31B dominates) |
| Needs | idle machine, free bench lock, at least 20 GB RAM free |
| Writes | `results/<date>/E0_*.json` (one envelope per model, variant, thread count and pp/tg; one per completion run), `results/<date>/E0_GATE.md`, `results/<date>/membw_desktop.json` |

### Hypothesis

> Both Q4_0 models run end-to-end on the 8400F; the MoE is bandwidth-bound and peaks at 4 threads (~15 tok/s), the dense 31B at ~2.4 tok/s; `-nr` keeps the whole file CPU_Mapped with anon RSS < 1.5 GB at a measurable kernel penalty, while `-ot "ffn_(gate_up|down)_exps=CPU"` still repacks (negative control). Variants differ in speed, not output.

### Procedure

> .\pc\desktop_bench.ps1 -Stage bench: llama-bench.exe -t 4,6,12 -p 128 -n 32 -r 3 -o json for variants default / --repack 0 / -ot hybrid_negctrl / -lm none on both GGUFs (31B: -t 6,12, -n 16, -r 2); then llama-completion.exe -n 64 --temp 0 per variant capturing load time, working-set and private-bytes peaks and the 'load_tensors: &lt;buft> model buffer size' lines.

Commands (PowerShell 5.1, from the repo root):

```powershell
python -m pigemma.paths lock --status
.\pc\desktop_bench.ps1 -Stage bench -Models 26b,31b -DryRun
.\pc\desktop_bench.ps1 -Stage bench -Models 26b,31b
```

What the driver runs, for debugging one step by hand (`--model` takes a registry id or a path):

```powershell
python -m pigemma.bench matrix --model 26b-q4 --threads 4,6,12 --pp 128 --tg 32 -r 3 --variants default,norepack,hybrid_negctrl,nommap --out results/<date>/
python -m pigemma.bench matrix --model 31b-q4 --threads 6,12 --pp 128 --tg 16 -r 2 --variants default,norepack,hybrid_negctrl,nommap --out results/<date>/
python -m pigemma.bench completion --model 26b-q4 --prompt-file data/calibration/en_tech_moe_streaming.txt -n 64 --threads 6 --variant norepack --out results/<date>/
```

The raw llama.cpp equivalent of one matrix variant and one completion run:

```powershell
third_party\llama.cpp-win\llama-bench.exe -m models\google__gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf -t 4,6,12 -p 128 -n 32 -r 3 -o json --repack 0
third_party\llama.cpp-win\llama-completion.exe -m models\google__gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf -f data\calibration\en_tech_moe_streaming.txt -n 64 --temp 0 -t 6 -c 4096 -nr
```

Desktop memory bandwidth (added by this plan, D5). This is the denominator of `llama_efficiency`. It
is measured in Docker, because there is no Windows C compiler. Store it next to the E0 envelopes:
`python -m pigemma.cache_sim --e0-dir results/<date>` derives the desktop efficiency from a measured E0
`tg` and the best membw read rate it finds under that directory.

```bash
bash pc/docker_run.sh --no-stdin -- tools/build-linux-x86/bin/membw --size-mb 512 --threads 1,2,4,6,12 --json > results/<date>/membw_desktop.json
```

Variant to llama.cpp flags (bench.py):

| Variant | llama-bench | llama-completion, llama-server, llama-perplexity | Role |
|---|---|---|---|
| `default` | none | none | repacked baseline |
| `norepack` | `--repack 0` | `-nr` | the only page-cache streaming mode |
| `hybrid_negctrl` | the `-ot` override below | the same | negative control C1 |
| `nommap` | `-lm none` | `-lm none` | weights read into anonymous memory |
| `mlock` | `-lm mmap+mlock` | the same | the Pi's FITS_MAPPED mode; not part of E0 |
| `kvq8` | `-fa on -ctk q8_0 -ctv q8_0` | the same | KV cache in q8_0; not part of E0 |

The `hybrid_negctrl` override, identical for every binary, is `-ot "ffn_(gate_up|down)_exps=CPU"`.
`python -m pigemma.bench variants` prints this table from the code.

### Metrics

| Metric (manifest) | Recorded as |
|---|---|
| pp128 and tg tok/s per thread count and variant (mean, sd) | envelope `pp` and `tg` = {tok_s, sd, n}, with `variant` and `threads` (`bench matrix`) |
| norepack penalty = tg(norepack)/tg(default) per model | computed at write-up from two `tg` envelopes with the same model and thread count; below 1 means slower |
| load_s | envelope `load_s` (from llama's `load time` line, `bench completion`) |
| rss_peak_mb vs private_peak_mb vs mapped_mb | envelope `rss_peak_mb` (Windows working-set peak), `private_peak_mb` (private-bytes peak), `mapped_mb` |
| buffer_split: CPU_REPACK vs CPU_Mapped MiB per variant | envelope `buffer_split` {CPU_REPACK_mib, CPU_Mapped_mib, CPU_mib}, parsed from `load_tensors: <buft> model buffer size = X MiB` |

### Pre-registered thresholds

> PASS if 26B tg@4t within +-20% of 15.18 and tg@6t within +-20% of 13.5 (idle-machine reference), 31B tg@6t within +-20% of 2.44; REPACK GATE PASS if the norepack variant logs CPU_Mapped >= 95% of the file size with no CPU_REPACK line and private peak < 1.5 GB, AND hybrid_negctrl logs a CPU_REPACK line (confirms -ot does not stream). If the gate fails, stop all streaming work (E5/E6/Pi) until the flags are fixed. The norepack penalty is reported as the cost of streaming mode (expected 10-40%).

| Check | Pass condition | Derived numbers |
|---|---|---|
| 26B speed (default variant) | tg@4t within ±20% of 15.18; tg@6t within ±20% of 13.5 | 12.14-18.22 tok/s; 10.80-16.20 tok/s |
| 31B speed (default variant) | tg@6t within ±20% of 2.44 | 1.95-2.93 tok/s |
| Repack gate, 26B | `norepack`: CPU_Mapped ≥ 95% of the file, no CPU_REPACK line, private peak < 1.5 GB | file 14,439,363,584 B = 13,770.5 MiB, so CPU_Mapped ≥ 13,081.9 MiB |
| Repack gate, 31B | the same | file 17,651,001,568 B = 16,833.3 MiB, so CPU_Mapped ≥ 15,991.6 MiB |
| Negative control (C1) | `hybrid_negctrl` logs a CPU_REPACK line | presence of the line |
| Norepack penalty | reported, not pass/fail; expected 10-40% | tg(norepack)/tg(default) of 0.60-0.90 |
| Gate failure | stop E5, E6 and all Pi streaming work until the flags are fixed | none |

The speed references are the idle-machine numbers in
[results/desktop-baseline-2026-10-04.md](../results/desktop-baseline-2026-10-04.md): llama-bench tg32
gave 15.18 ± 0.42 tok/s at 4 threads, 13.52 ± 1.77 at 6, 12.50 ± 0.95 at 8 and 12.79 ± 0.31 at 12; the
31B gave 2.44 ± 0.04 at 6 threads. They are reference values for the thresholds, not E0 results.

### Results template (E0)

| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value, source) |
|---|---|---|---|
| 26B tg32 tok/s, default, 4 / 6 / 12 threads | 16.31 / 15.55 / 13.89 (desktop, 2026-10-05, background CPU 12–62%, `results/2026-10-05/`) | — | — |
| 26B pp128 tok/s, default, 6 threads | 69.68 (same) | — | — |
| 26B tg32 tok/s, norepack, 4 / 6 / 12 threads | 14.31 / 14.88 / 12.93 (same) | — | — |
| 26B norepack penalty, tg(norepack)/tg(default) at 4 and 6 threads | 0.83–0.88 at 4 threads, 0.92–0.96 at 6 (sequential matrix plus ABBA repeats in `results/2026-10-05/e0-interleave/`). pp128: 0.66–0.70 ([E0 results](results/E0-desktop-baseline-and-repack-gate.md)) | — | 0.60-0.90, the manifest's expected 10-40% penalty (an expectation, not a measurement) |
| 26B load_s, default / norepack | 9.92 / 2.08 s (warm page cache) | — | — |
| 26B private peak MB and RSS peak MB, norepack | 1,236 / 13,821 MB (default: 15,034 / 28,831) | — | — |
| 26B buffer_split (MiB), default / norepack / hybrid_negctrl | REPACK 13,133.9 + Mapped 13,755.4 / Mapped 13,755.4 / REPACK 13,133.9 + Mapped 13,755.4 | — | — |
| 31B tg16 tok/s, default and norepack, 6 / 12 threads | default 2.33 / 2.16; norepack 2.00 / 2.18 | — | — |
| 31B norepack penalty at 6 threads | 0.86–0.90 (matrix 0.859, ABBA 0.902). pp128: 0.55–0.62 | — | 0.60-0.90 (as above) |
| 31B buffer_split (MiB), norepack / hybrid_negctrl | Mapped 16,818.2 / REPACK 15,710.6 + Mapped 16,818.2 | — | — |
| Desktop memory bandwidth, read GB/s (membw in Docker, best thread count) | 46.90 at 2 threads (`results/2026-10-05/membw_desktop.json`, inside the WSL2 VM) | — | 60 GB/s, catalog `desktop-8400f` (assumed: DDR5 dual channel; module speed not recorded) |
| Desktop llama_efficiency = tg × bytes/token ÷ membw read (derived from measurements) | DERIVED: 0.83 (26B @4t default; 0.73 norepack), 0.88 (31B @6t default; 0.75 norepack) | — | 0.65 (the catalog and feasibility default, assumed) |

**Verdict vs pre-registered threshold:** measured on 2026-10-05 (`results/2026-10-05/E0_GATE.md`).

- Speed: **PASS** for both models (26B 16.31 @4t and 15.55 @6t; 31B 2.33 @6t).
- REPACK GATE 26B: **PASS** (13,755.4 MiB mapped, no CPU_REPACK, private peak 1,236 MB).
- REPACK GATE 31B: **FAIL** on the private peak (2,533 MB ≥ 1,500 MB). The f16 KV cache at `-c 4096` is 1,520 MiB and the compute buffer 592 MiB. The mapped and no-repack checks pass, so no weight was copied.
- Negative control C1: confirmed for both models (`-ot` repacks 13,133.9 / 15,710.6 MiB).
- Not supported: "variants differ in speed, not output". `-nr` vs default KLD is 0.087 (26B) and 0.015 (31B). On the 26B, a batch-size-only change with repack on gives the same 0.088.

Details: [E0 results](results/E0-desktop-baseline-and-repack-gate.md).

**What would change this conclusion:**
- A re-run on an idle machine that lands outside the ±20% band. E0 is then repeated before anything
  downstream.
- Any llama.cpp update. The `-nr` and CPU_Mapped behaviour is code, not physics, so the gate is
  re-verified on every build.
- A buffer_split parser that misses a `load_tensors:` line. Always cross-check against the raw stderr
  kept with the result.

## E1: speculative decoding via llama-server

| Item | Value |
|---|---|
| Manifest id | `E1-speculative-via-server` |
| Results `experiment` | `E1` |
| Driver | [pc/desktop_bench.ps1](../pc/desktop_bench.ps1) `-Stage spec`, which runs `python -m pigemma.bench server-spec` |
| Host and models | desktop, native `llama-server.exe`. `26b-q4` with the drafts `26b-mtp` (0.25 GB) and `26b-dflash` (0.47 GB); `31b-q4` with `31b-mtp` (0.28 GB) |
| Runtime (manifest) | ~30 min |
| Needs | E0 finished (same lock); port 18080 free; [data/prompts/gen_prompts.json](../data/prompts/gen_prompts.json) |
| Writes | `results/<date>/E1_*.json` (one envelope per model and variant) |

### Hypothesis

> At b11398 the Gemma 4 MTP sidecars load but do not draft in the server slot (ad-hoc evidence), DFlash drafts at ~28% acceptance and is slower on the CPU-bound MoE; an explicit `--spec-type` may activate MTP; any active drafter must be greedy-lossless.

### Procedure

> .\pc\desktop_bench.ps1 -Stage spec: pigemma.bench server-spec on the 26B for plain, mtp, mtp-explicit (--spec-type draft-mtp), dflash, dflash-explicit, ngram (--spec-type ngram-simple) and on the 31B for plain, mtp, mtp-explicit; 8 prompts, -n 200, temperature 0, -t 6, -c 4096; record timings.draft_n / draft_n_accepted per request and compare visible text with the plain run.

Commands:

```powershell
.\pc\desktop_bench.ps1 -Stage spec -Models 26b,31b -DryRun
.\pc\desktop_bench.ps1 -Stage spec -Models 26b,31b
```

The equivalent commands by hand. `--auto-explicit` adds `mtp-explicit` and `dflash-explicit` only when
the inferred variant never drafted; `plain` always runs first:

```powershell
python -m pigemma.bench server-spec --model 26b-q4 --variant plain,mtp,dflash,ngram --auto-explicit --prompts data/prompts/gen_prompts.json -n 200 --threads 6 --port 18080 --out results/<date>/
python -m pigemma.bench server-spec --model 31b-q4 --variant plain,mtp,mtp-explicit --prompts data/prompts/gen_prompts.json -n 200 --threads 6 --port 18080 --out results/<date>/
python -m pigemma.bench variants
```

The last command prints the variant-to-flags table below from the code itself.

The raw server command for one variant (bench.py starts and stops its own server process and never
touches any other):

```powershell
third_party\llama.cpp-win\llama-server.exe -m models\google__gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf -t 6 -c 4096 -np 1 --host 127.0.0.1 --port 18080 -md models\ggml-org__gemma-4-26B-A4B-it-GGUF\mtp-gemma-4-26B-A4B-it-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 6
```

| Variant | Server flags | Note |
|---|---|---|
| `plain` | none | reference for the speedup and for identical_to_plain |
| `mtp` | `-md` with the family's MTP sidecar | type inferred from the draft GGUF |
| `mtp-explicit` | `-md` MTP plus `--spec-type draft-mtp` | run when `mtp` shows no drafts |
| `dflash` | `-md` with the DFlash sidecar | 26B only |
| `dflash-explicit` | `-md` DFlash plus `--spec-type draft-dflash` | 26B only |
| `ngram` | `--spec-type ngram-simple` | no draft model; 26B only |

### Metrics

| Metric (manifest) | Recorded as |
|---|---|
| tg tok/s per variant and speedup vs plain | envelope `tg` (server `timings.predicted_per_second` over the 8 prompts); the speedup is computed against `plain` at write-up |
| accept_rate = sum(draft_n_accepted)/sum(draft_n) | envelope `accept_rate`, `draft_n`, `draft_n_accepted` (sums over the 8 requests) |
| spec_active (draft_n > 0 on every request) and spec_type_effective | envelope `spec_active` (true only if draft_n > 0 on every request) and `spec_type_effective` (`inferred`, `explicit:<type>` or `none`) |
| identical_to_plain fraction | envelope `identical_to_plain`: the fraction of prompts whose visible text (thinking channel stripped) equals the plain run's |
| load_time_s and peak private bytes with the draft loaded | envelope `load_s` (time to a healthy `/health`) and `private_peak_mb` |

### Pre-registered thresholds

> A variant counts as ACTIVE only if spec_active; report 'useful' if speedup >= 1.2x with identical_to_plain >= 0.95 on >= 6 of 8 prompts; 'inactive' (MTP never drafting) and 'slower' (DFlash) are recorded as findings, not failures; any non-identical greedy output with an active drafter is a bug noted in results and that draft is excluded from the Pi recommendation.

How the rule is applied (added by this plan):

- **ACTIVE** means `spec_active`.
- **Useful** means a per-prompt speedup of at least 1.2 on at least 6 of the 8 prompts, AND
  `identical_to_plain` ≥ 0.95. With 8 prompts, that last condition means all 8 visible texts must be
  identical.
- **Inactive** and **slower** are findings, not failures.
- An ACTIVE drafter with any non-identical greedy output is a bug. That draft is excluded from the Pi
  recommendation.

Reference values, measured ad hoc before this plan (llama-server, 6 threads, 200 greedy tokens;
[results/desktop-baseline-2026-10-04.md](../results/desktop-baseline-2026-10-04.md),
`results/specsrv-*.json`):

- **26B:** plain gave 16.03 tok/s. MTP loaded but never drafted (draft n_max 3 and 6). DFlash drafted
  321 tokens and 91 were accepted (28%), which slowed generation to 11.83 tok/s.
- **31B:** plain gave 2.40 tok/s. MTP loaded but never drafted.

### Results template (E1)

| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value, source) |
|---|---|---|---|
| 26B plain tg tok/s | _pending_ | — | — |
| 26B mtp: spec_active, accept_rate, tg, speedup | _pending_ | — | — |
| 26B mtp-explicit: spec_active, accept_rate, tg, speedup | _pending_ | — | — |
| 26B dflash: spec_active, accept_rate, tg, speedup | _pending_ | — | — |
| 26B dflash-explicit: spec_active, accept_rate, tg, speedup | _pending_ | — | — |
| 26B ngram: spec_active, accept_rate, tg, speedup | _pending_ | — | — |
| 26B identical_to_plain per ACTIVE variant | _pending_ | — | — |
| 31B plain tg tok/s | _pending_ | — | — |
| 31B mtp and mtp-explicit: spec_active, accept_rate, tg, speedup, identical_to_plain | _pending_ | — | — |
| load_s and private peak MB with the draft loaded, per model | _pending_ | — | — |
| Speedup carried into the Pi projections | — | _pending_ (only for a variant that is ACTIVE, useful and lossless; otherwise "none") | — |

**Verdict vs pre-registered threshold:** _pending_. Give each variant one word (useful, inactive, slower, or bug) with its numbers and result file.

**What would change this conclusion:**
- A drafter that drafts on every request with high acceptance (roughly 70% or more on this
  bandwidth-bound CPU) and produces identical greedy output. Candidates: a fixed MTP path in a later
  llama.cpp, or a better-matched draft model.
- On the dense 31B, where each verified token amortises a 17.6 GB weight read, even moderate
  acceptance could pay off. E1 would then be repeated on the Pi.

## E2: quality (KLD, PPL, QA)

| Item | Value |
|---|---|
| Manifest id | `E2-quality` |
| Results `experiment` | `E2` |
| Driver | [pc/desktop_bench.ps1](../pc/desktop_bench.ps1) `-Stage quality`, which runs [pigemma/quality.py](../pigemma/quality.py) (`kld-base`, `kld`, `qa`) |
| Host and models | desktop, native binaries; `26b-q4`, `31b-q4`; lower quants optional |
| Runtime (manifest) | ~40 min for both models (+30 min per optional quant) |
| Needs | E0 finished (same lock); at least 2 × 1,610,612,736 B free per model for the reference logits |
| Writes | `results/quality/ref_<model>.logits` (untracked), `results/<date>/E2_*.json`, `results/quality/qa_<model>_<variant>.json`, `data/calibration/all.txt` (built by `quality.build_corpus()`) |

### Hypothesis

> -nr and the -ot negative control are numerically identical to the default run (KLD ~ 0); both Q4_0 models answer the fixed EN/HE/arithmetic set; lower quants (optional) degrade measurably.

### Procedure

> .\pc\desktop_bench.ps1 -Stage quality: llama-perplexity.exe --save-all-logits on data/calibration/all.txt (-c 512 --chunks 6 = 1.61 GB per model, disk-gated) for each Q4_0 model; --kl-divergence --kl-divergence-base for variants norepack and hybrid_negctrl (and a lower quant only with -AllowDownload and >= 1.3x free disk); pigemma.quality qa (16 items) per model via llama-server, plus mtp/dflash variants only when E1 found them active.

Commands:

```powershell
.\pc\desktop_bench.ps1 -Stage quality -Models 26b,31b -DryRun
.\pc\desktop_bench.ps1 -Stage quality -Models 26b,31b
```

What the driver runs for one model (the same again for `31b-q4`):

```powershell
python -m pigemma.quality kld-base --model 26b-q4 --text data/calibration/all.txt -c 512 --chunks 6 --out results/quality/ref_26b-q4.logits
python -m pigemma.quality kld --model 26b-q4 --base results/quality/ref_26b-q4.logits --variant norepack --out results/<date>/
python -m pigemma.quality kld --model 26b-q4 --base results/quality/ref_26b-q4.logits --variant hybrid_negctrl --out results/<date>/
python -m pigemma.quality qa --model 26b-q4 --threads 6 --server-variant plain -n 160 --out results/quality/qa_26b-q4_plain.json
```

The raw perplexity commands. At a7b94df, `--save-all-logits` and `--kl-divergence-base` are two names
of the same option. With `--kl-divergence` set, perplexity reads that file instead of writing it.

```powershell
third_party\llama.cpp-win\llama-perplexity.exe -m models\google__gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf -f data\calibration\all.txt -c 512 --chunks 6 -t 6 --save-all-logits results\quality\ref_26b-q4.logits
third_party\llama.cpp-win\llama-perplexity.exe -m models\google__gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf -f data\calibration\all.txt -c 512 --chunks 6 -t 6 --kl-divergence --kl-divergence-base results\quality\ref_26b-q4.logits -nr
```

The quants the real Pi runs (D2) are optional and disk-gated:

- **26b-ud-q3_k_m:** `-AllowDownload` fetches this one (12.73 GB) when free disk is at least 1.3x its
  size.
- **31b-ud-iq3_xxs** (11.84 GB): this plan adds it, but only with at least 15.4 GB free on D:. The
  file is deleted afterwards.

```powershell
.\pc\desktop_bench.ps1 -Stage quality -Models 26b -AllowDownload
hf download unsloth/gemma-4-31B-it-GGUF gemma-4-31B-it-UD-IQ3_XXS.gguf --local-dir models/unsloth__gemma-4-31B-it-GGUF
python -m pigemma.quality kld --model models/unsloth__gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-IQ3_XXS.gguf --base results/quality/ref_31b-q4.logits --variant 31b-ud-iq3_xxs --out results/<date>/
```

The QA set ([data/prompts/qa_set.json](../data/prompts/qa_set.json)) has 16 items: 7 English
(`en-01`..`en-07`), 6 Hebrew (`he-01`..`he-06`) and 3 numeric (`num-01`..`num-03`). Answers are graded
on the visible text after the thinking channel is stripped, with Hebrew-aware normalisation.

### Metrics

| Metric (manifest) | Recorded as |
|---|---|
| KLD mean / max / p99.9 and same-top-p per variant | envelope `kld` {mean, max, p999, same_top_p, ppl_q, ppl_base}, one per variant |
| PPL per model | envelope `ppl` (also `kld.ppl_base`) |
| qa_score /16 overall and per language | envelope `qa_score` and `qa_total`; per-language scores in `results/quality/qa_<model>_<variant>.json` |
| coherent flag and thinking_chars share | envelope `coherent`; `thinking_chars` per item in the QA file |
| exact_match_vs_plain for speculative variants | `exact_match_vs_plain` in the QA file (only for E1 variants that were ACTIVE) |

### Pre-registered thresholds

> PASS: norepack and hybrid_negctrl KLD mean < 0.001 and same-top-p > 99.5% (proves the streaming mode is numerically neutral); qa_score >= 13/16 for both models and Hebrew subset >= 4/6 (else document a Hebrew caveat). Quant tiers for the Pi recommendation: KLD mean < 0.05 'acceptable', < 0.15 'degraded', else 'not recommended'.

| Check | Pass condition |
|---|---|
| `norepack` and `hybrid_negctrl`, both models | KLD mean < 0.001 and same-top-p > 99.5% (control C2) |
| QA, each model | qa_score ≥ 13/16 |
| Hebrew subset, each model | ≥ 4/6, otherwise a documented Hebrew caveat |
| Quant tier for the Pi recommendation | KLD mean < 0.05 is "acceptable", < 0.15 is "degraded", otherwise "not recommended" |

Reference observations, made ad hoc before this plan; these are functional, not graded:

- **Desktop:** both Q4_0 models answered a 3-sentence English question correctly and wrote "שלום"
  (`results/smoke-26b-moe.stdout.txt`, `results/smoke-31b-dense.stdout.txt`).
- **Real Pi:** the 31B UD-IQ3_XXS answered fluently in English and Hebrew
  ([results/pi-real/README.md](../results/pi-real/README.md)).

### Results template (E2)

| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value, source) |
|---|---|---|---|
| 26B KLD norepack: mean / max / p99.9 / same-top-p | _pending_ | — | — |
| 26B KLD hybrid_negctrl: mean / max / p99.9 / same-top-p | _pending_ | — | — |
| 31B KLD norepack: mean / max / p99.9 / same-top-p | _pending_ | — | — |
| 31B KLD hybrid_negctrl: mean / max / p99.9 / same-top-p | _pending_ | — | — |
| PPL 26B / 31B (6 × 512 tokens of data/calibration/all.txt) | _pending_ | — | — |
| QA 26B: score /16; en /7, he /6, num /3 | _pending_ | — | — |
| QA 31B: score /16; en /7, he /6, num /3 | _pending_ | — | — |
| Coherent outputs and thinking-character share, per model | _pending_ | — | — |
| 26b-ud-q3_k_m vs 26b-q4: KLD mean, same-top-p, tier (optional; the real-Pi quant) | _pending_ | — | — |
| 31b-ud-iq3_xxs vs 31b-q4: KLD mean, same-top-p, tier (optional; the real-Pi quant) | _pending_ | — | — |
| exact_match_vs_plain for ACTIVE speculative variants (only if E1 found any) | _pending_ | — | — |

**Verdict vs pre-registered threshold:** _pending_. Report the C2 and C1 KLD results as PASS/FAIL, QA as PASS/FAIL per model with the Hebrew subset, and a tier for each optional quant.

**What would change this conclusion:**
- A `-nr` KLD above 0.001. That would mean the non-repacked kernels compute something different. Then
  re-run E2 with more chunks and inspect same-top-p before trusting any streaming number.
- A grading bug. Read the visible text stored in the QA file by eye for every failed item.
- For the quant tiers: KLD on a longer corpus, or on a different one.

## E3: MoE routing trace

| Item | Value |
|---|---|
| Manifest id | `E3-routing-trace` |
| Results `experiment` | `E3` |
| Driver | [pc/trace_collect.sh](../pc/trace_collect.sh), which runs `pigemma-probe --mode moe-trace` ([tools/probe/probe.cpp](../tools/probe/probe.cpp)) in Docker, then [pigemma/trace.py](../pigemma/trace.py) |
| Host and model | desktop Docker (`pi-gemma4-build`, 12 CPUs, 31 GB VM); `26b-q4` through the `/work` bind mount, with `-nr` |
| Runtime (manifest) | build 5-8 min; smoke 3 min (first bind-mount read of the GGUF dominates); quick ~20-25 min |
| Needs | the probe built (`bash pc/docker_run.sh build-tools`); a free bench lock; 14.4 GB of VM page cache for the model |
| Writes | `traces/26b/<stem>.moetrace` with `.tokens.bin`, `.gen.txt` and `.meta.json`; `traces/26b/all.moetrace`; `traces/26b/all.stats.json`; `results/h1/all.*.csv` |

### Hypothesis

> Expert routing on real mixed EN/HE/code text is skewed and temporally sticky: per layer, the 32 hottest of 128 experts cover >= 70% of selections and >= 40% of a token's experts were used by the previous token in the same layer; Hebrew and code use partly different experts than English.

### Procedure

> bash pc/docker_run.sh build-tools; bash pc/trace_collect.sh --smoke (validate 72 records, commit as golden fixture); bash pc/trace_collect.sh --quick (all calibration files, chunk 1024, --gen 128 per chunk, plus 8 gen prompts x 256, -nr -t 12 -ub 256, warmup off); merge; python -m pigemma.trace stats.

Commands (Git Bash, from the repo root):

```bash
bash pc/docker_run.sh build-tools
bash pc/docker_run.sh --no-stdin -- python -m pigemma.paths lock --hold -- tools/build-linux-x86/bin/pigemma-probe --mode names -m /work/models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf -nr -t 12 -c 512 > results/<date>/probe_names_26b.txt
bash pc/trace_collect.sh --smoke
bash pc/trace_collect.sh --quick
```

What `trace_collect.sh --quick` runs. There is one probe process per corpus file, plus one for the
generation prompts:

```bash
bash pc/docker_run.sh -- python -m pigemma.paths lock --hold -- tools/build-linux-x86/bin/pigemma-probe --mode moe-trace -m /work/models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf -f /work/data/calibration/en_tech_moe_streaming.txt -nr -t 12 -c 4096 -b 2048 -ub 256 --chunk-tokens 1024 --gen 128 --corpus-id en_tech_moe_streaming --trace-out /work/traces/26b/en_tech_moe_streaming.moetrace
bash pc/docker_run.sh -- python -m pigemma.paths lock --hold -- tools/build-linux-x86/bin/pigemma-probe --mode moe-trace -m /work/models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf --gen-prompts /work/data/prompts/gen_prompts.json --chat -nr -t 12 -c 4096 -b 2048 -ub 256 --corpus-id gen_prompts --trace-out /work/traces/26b/gen_prompts.moetrace
python -m pigemma.trace validate traces/26b/en_tech_moe_streaming.moetrace
python -m pigemma.trace merge traces/26b/all.moetrace traces/26b/en_tech_moe_streaming.moetrace traces/26b/gen_prompts.moetrace
python -m pigemma.trace stats traces/26b/all.moetrace --out traces/26b/all.stats.json --csv-dir results/h1/
```

List the merge inputs explicitly (one per corpus file, as above). A glob such as
`traces/26b/*.moetrace` would also match an older `all.moetrace`.

To read the headline numbers (Git Bash):

```bash
python -c "import json,statistics as st; s=json.load(open('traces/26b/all.stats.json',encoding='utf-8')); pl=s['per_layer']; g=s['global']; c=s['cross_layer']; print('n_prefill',s['n_prefill'],'n_gen',s['n_gen']); print('coverage@32 mean',st.mean(l['coverage_at']['32'] for l in pl)); print('adjacent_reuse_mean',g['adjacent_reuse_mean']); print('p@16',c['precision_at_16'],'static',c['baseline_static_top16'],'same_ids',c['baseline_same_ids']); print('JS prefill-vs-gen mean',st.mean(l['prefill_vs_gen_js_bits'] for l in pl if l['prefill_vs_gen_js_bits'] is not None))"
```

### Metrics

| Metric (manifest) | Recorded as (`stats.json` keys, [docs/TRACE_FORMAT.md](TRACE_FORMAT.md) section 5.2) |
|---|---|
| n_prefill, n_gen, all 30 layers present, validate() clean | `n_prefill`, `n_gen`; header `moe_layers` (30 entries); `python -m pigemma.trace validate` exits 0 |
| per-layer coverage@8/16/32/48/64 (count and weight mass), entropy, Gini | `per_layer[].coverage_at`, `coverage_weight_at`, `entropy_bits`, `effective_experts`, `gini` |
| adjacent reuse and Jaccard, working sets | `per_layer[].adjacent_reuse`, `jaccard_next_mean`, `working_set`; `global.adjacent_reuse_mean` |
| LRU stack-distance hit curve | `global.stack_distance_hist`, `global.lru_hit_curve` (control C4) |
| cross-layer predictor precision@16 vs static and same-ids baselines | `cross_layer.precision_at_16`, `baseline_static_top16`, `baseline_same_ids`, `mutual_info_bits_mean` against `mutual_info_bits_null_mean` |
| per-corpus top-32 overlap (EN vs HE vs code) | `global.corpus_overlap`, keyed by two chunk sources joined with a vertical bar (TRACE_FORMAT section 5.2) |
| probe tok/s and callback overhead | `<trace>.meta.json` `tok_s` {prompt, gen} and `callback_overhead_ms`; the probe's `PIGEMMA_SUMMARY` line |

### Pre-registered thresholds

> Trace VALID if validate() passes, >= 10k prefill + >= 2k generated tokens, generated text coherent. H1 precondition MET ('strong skew') if mean coverage@32 >= 0.70 or adjacent reuse >= 0.40; 'moderate' 0.5-0.7; 'near-uniform' < 0.5 (then LRU alone is the plan and static hot sets are dropped). Prefetch worth modelling only if predictor precision@16 beats the static baseline by >= 0.1.

| Outcome | Rule |
|---|---|
| Trace VALID | `validate()` clean, ≥ 10,000 prefill and ≥ 2,000 generated tokens, generated text coherent |
| Strong skew (H1 precondition MET) | mean coverage@32 ≥ 0.70 **or** adjacent reuse ≥ 0.40 |
| Moderate | coverage@32 between 0.5 and 0.7, and no strong-skew condition met |
| Near-uniform | coverage@32 < 0.5: LRU alone is the plan, and static hot sets are dropped (E6 then cannot become a recommendation) |
| Prefetch worth modelling | `precision_at_16 − baseline_static_top16` ≥ 0.10 |
| C3 (added by this plan) | mean `prefill_vs_gen_js_bits` ≤ 0.10 bits |

For scale, these are the values uniform routing would give. They are computed, not assumed:
- coverage@32 = 32/128 = 0.25;
- adjacent reuse = K/E = 8/128 = 0.0625;
- precision@16 of a random 16-set = 16/128 = 0.125;
- entropy 7 bits, Gini 0.

### Results template (E3)

| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value, source) |
|---|---|---|---|
| Trace size: n_prefill / n_gen / chunks / MoE layers | _pending_ | — | — |
| validate() errors (must be none) | _pending_ | — | — |
| Coherent generated chunks (gen.txt) | _pending_ | — | — |
| Mean coverage@32, by count / by weight | _pending_ | — | 0.25 for uniform routing (computed, for scale) |
| Mean coverage@8 / @16 / @64 | _pending_ | — | 0.0625 / 0.125 / 0.5 for uniform routing |
| Mean entropy bits / effective experts / Gini | _pending_ | — | 7 / 128 / 0 for uniform routing |
| adjacent_reuse_mean / jaccard_next_mean | _pending_ | — | 0.0625 uniform. Literature prior: P(expert reused by the next token) = 0.444 for Qwen3-30B-A3B ([docs/research/RESEARCH_SUMMARY.md](research/RESEARCH_SUMMARY.md) section 3; definition differs slightly) |
| Working set per layer, W = 64 tokens, median / p95 | _pending_ | — | — |
| LRU overall hit rate at 384 / 768 / 1,536 items (10% / 20% / 40% of 3,840) | _pending_ | — | 0.40 / 0.67 / 0.86 from the catalog prior curve (`stream_hit_rate_curve_assumed`; steady state, not overall) |
| Cross-layer precision@16 vs static top-16 vs same ids | _pending_ | — | 0.125 for a random prediction |
| Corpus overlap, top-32 Jaccard, EN vs HE vs code | _pending_ | — | — |
| C3: mean prefill-vs-gen JS (bits) | _pending_ | — | — |
| Probe tok/s, prompt / gen; callback overhead (ms) | _pending_ | — | — |

**Verdict vs pre-registered threshold:** _pending_. Give the trace VALID/INVALID, the skew class (strong, moderate or near-uniform), whether prefetch is worth modelling, and the C3 result, with `traces/26b/all.stats.json`.

**What would change this conclusion:**
- A larger or more varied corpus (the `--full` tier, other topics or languages) that moves coverage@32
  across 0.5 or 0.7.
- A probe bug caught by the golden fixture or by `validate()`.
- Routing on the 3-bit quant (`26b-ud-q3_k_m`, the real-Pi file) differing materially from Q4_0. If
  E4 projections are to be used for that file, re-trace it.

## E4: cache simulation

| Item | Value |
|---|---|
| Manifest id | `E4-cache-simulation` |
| Results `experiment` | `E4` |
| Driver | [pigemma/cache_sim.py](../pigemma/cache_sim.py) |
| Host | desktop, native Python; no model load (it reads the trace and `map.json`) |
| Runtime (manifest) | < 10 min |
| Needs | the E3 trace, `results/map26.json`, [pigemma/data/catalog.json](../pigemma/data/catalog.json); E0 envelopes plus `membw_desktop.json` for the desktop `llama_efficiency` (else 0.65 assumed) |
| Writes | `results/h1/sim.json`, `results/h1/sim.csv`, `results/h1/E4_*.json` (one `kind: projected` envelope per profile and policy, plus the dense line), `docs/results/H1_RESULTS.md` |

### Hypothesis

> At the 8 GB + NVMe Gen3 budget (~5.2 GB expert cache, ~40% of experts) LRU reaches >= 60% hit rate on real text (miss bytes <= ~320 MB/token) giving a projected 1.5-2.5 tok/s streaming; 4 GB / microSD is < 0.3 tok/s under every policy; 16 GB holds everything and is RAM-bandwidth-bound at ~3-4 tok/s.

### Procedure

> python -m pigemma.cache_sim traces/26b/all.moetrace --map results/map26.json --policies lru,lfu,static,belady,lru_prefetch,lru_perlayer --budgets-gb 0.5,1,2,3,4,5,6,8,12.85 --profiles all --train-split 0.3 --warmup-tokens 256 --out results/h1/sim.json; projections use the catalog device profiles (assumed) and the desktop-calibrated llama_efficiency; python -m pigemma.results summarize -> docs/results/H1_RESULTS.md.

Commands:

```powershell
python -m pigemma.gguf_map build 26b-q4 --out results/map26.json --reuse
python -m pigemma.cache_sim traces/26b/all.moetrace --map results/map26.json --catalog pigemma/data/catalog.json --policies lru,lfu,static,belady,lru_prefetch,lru_perlayer --budgets-gb 0.5,1,2,3,4,5,6,8,12.85 --profiles all --train-split 0.3 --warmup-tokens 256 --e0-dir results/<date> --out results/h1/sim.json --csv results/h1/sim.csv
python -m pigemma.results summarize results/h1 --experiment E3,E4,E5,E6 --md docs/results/H1_RESULTS.md --title "H1 results: MoE expert streaming"
```

`--io-mem` and `--calibration` default to `auto`. Once `results/pi/io_mem.json` (pi/03) and
`results/pi/calibration.json` (pi/05) exist, the same command projects with the measured Pi constants
and labels them `measured`.

Independent re-check of control C4 against the stats file (Git Bash). Every printed line must end in
`True`; capacities above 3,840 items compare with the last point of the curve:

```bash
python -c "import json; s=json.load(open('results/h1/sim.json',encoding='utf-8')); h=json.load(open('traces/26b/all.stats.json',encoding='utf-8'))['global']['lru_hit_curve']; [print(r['capacity_items'], r['overall']['hit_rate'], h[min(int(r['capacity_items']),len(h))-1][1], abs(r['overall']['hit_rate']-h[min(int(r['capacity_items']),len(h))-1][1]) < 1e-9) for r in s['runs'] if r['policy']=='lru' and r['capacity_items']>0]"
```

The Pi verdicts of `pigemma.feasibility` use this simulation. `--sim results/h1/sim.json` replaces the
assumed hit-rate curve (`hit_source: simulated`).

**Projection model**, as implemented in `pigemma/cache_sim.py` (every symbol is listed in
`projection.assumptions` of `sim.json`):

- `expert_cache_bytes = ram − os_reserve − non_expert − KV(n_ctx, q8_0) − compute_buffers`. A value
  ≤ 0 gives INFEASIBLE.
- `resident_bytes_per_token = non_expert + hit_bytes`.
- `t_ram = resident_bytes_per_token / (mem_bw × llama_efficiency)`.
- `t_io = miss_bytes / rand_read_3mb + n_miss × access_ms`, plus the same terms for prefetch reads.
- `tg_serial = 1 / (t_ram + t_io − overlapped)`, where prefetch hides behind at most one layer of
  compute per MoE slot.
- `tg_overlap = 1 / max(t_ram, t_io)`.
- For policies without prefetch, `overlapped` = 0, and both reduce to the manifest formulas.
- Verdict:
  - INFEASIBLE;
  - FITS_RAM (the cache holds every expert);
  - STREAM_GOOD (tg_overlap ≥ 2.0 and tg_serial ≥ 1.0);
  - STREAM_WEAK (tg_serial ≥ 0.5, including tg_serial ≥ 1.0 with tg_overlap < 2.0);
  - NOT_VIABLE.

Budgets that follow from the formula and the catalog profiles. Every input is the catalog's
**assumed** value. The non-expert size and KV size are exact, from `results/map26.json`: 1,577,172,088 B
and 155,975,680 B at 4,096 tokens in q8_0.

| Profile (catalog) | RAM | OS reserve | KV q8_0 (n_ctx) | Compute | Expert cache | Share of 12.85 GB | Items of 3,840 |
|---|---|---|---|---|---|---|---|
| `pi5-4gb-microsd` | 4.0 GB | 0.7 GB | 0.078 GB (2,048) | 0.3 GB | 1.34 GB | 10.5% | 401 |
| `pi5-8gb-usb3` | 8.0 GB | 0.8 GB | 0.156 GB (4,096) | 0.3 GB | 5.17 GB | 40.2% | 1,544 |
| `pi5-8gb-nvme-gen3` | 8.0 GB | 0.8 GB | 0.156 GB (4,096) | 0.3 GB | 5.17 GB | 40.2% | 1,544 |
| `pi5-16gb-nvme-gen3` | 16.0 GB | 0.9 GB | 0.156 GB (4,096) | 0.4 GB | 12.97 GB (≥ 12.85) | 100% | all |

**Worked example with prior inputs only (ASSUMED).** This uses the catalog's prior hit-rate curve and
the assumed device numbers:

- memory: 10 GB/s × llama_efficiency 0.65;
- random 3 MB reads: 700 / 350 / 60 MB/s, with access latencies of 0.2 / 0.5 / 2.0 ms (NVMe Gen3 /
  USB3 SSD / microSD);
- 240 expert reads of 3,345,408 B per token, 802,897,920 B in total.

cache_sim additionally rounds each slab read (gate_up, down) up to a multiple of `--read-granularity`
(131,072 B) to model readahead, so its miss bytes come out a few percent higher.

| Profile | Hit rate used | Miss MB/token | t_ram (s) | t_io (s) | tg_serial | tg_overlap | Verdict (cache_sim rule) |
|---|---|---|---|---|---|---|---|
| `pi5-4gb-microsd` | 0.419 (prior at 10.5%) | 466.7 | 0.294 | 8.057 | 0.12 | 0.12 | NOT_VIABLE |
| `pi5-8gb-usb3` | 0.861 (prior at 40.2%) | 111.7 | 0.349 | 0.336 | 1.46 | 2.87 | STREAM_GOOD |
| `pi5-8gb-usb3` | 0.60 (the hypothesis' floor) | 321.2 | 0.317 | 0.966 | 0.78 | 1.04 | STREAM_WEAK |
| `pi5-8gb-nvme-gen3` | 0.861 (prior at 40.2%) | 111.7 | 0.349 | 0.166 | 1.94 | 2.87 | STREAM_GOOD |
| `pi5-8gb-nvme-gen3` | 0.60 (the hypothesis' floor) | 321.2 | 0.317 | 0.478 | 1.26 | 2.09 | STREAM_GOOD (borderline) |
| `pi5-16gb-nvme-gen3` | 1.0 (everything fits) | 0 | 0.366 | 0 | 2.73 | 2.73 | FITS_RAM |

What follows from the worked example:
- On 8 GB, the choice between USB3 and NVMe only matters if the real hit rate is near the
  hypothesis' floor. That is exactly what E4 measures.
- The 16 GB in-RAM floor is 10 GB/s × 0.65 ÷ 2,380,072,318 B = 2.73 tok/s, or 4.2 tok/s at an
  efficiency of 1.0.
- The dense 31B line (17,635,166,448 B per token) gives 0.57 tok/s at 10 GB/s with an efficiency of
  1.0 (0.37 at 0.65), and 20.7 s per token streamed at 850 MB/s.

The two tools use different verdict words. This plan reports each tool's own word; the mapping below
is approximate:

| cache_sim | pigemma.feasibility |
|---|---|
| FITS_RAM | FITS_MAPPED or FITS_REPACKED |
| STREAM_GOOD | STREAM_OK, which only needs ≥ 1 tok/s, a cache of at least 2 GB, and NVMe or USB SSD storage |
| STREAM_WEAK | STREAM_SLOW, which needs ≥ 0.3 tok/s |
| NOT_VIABLE | NOT_VIABLE, with different cut-offs |
| INFEASIBLE | NOT_VIABLE (the non-expert weights do not fit) |

### Metrics

| Metric (manifest) | Recorded as (`sim.json`, [docs/TRACE_FORMAT.md](TRACE_FORMAT.md) section 11) |
|---|---|
| hit_rate, weighted_hit_rate, miss MB/token (mean, p95) per policy x budget | `runs[].steady.{hit_rate, weighted_hit_rate, miss_bytes_per_token.mean, .p95}`, plus the same under `overall` |
| Belady gap = LRU miss bytes / Belady miss bytes | ratio of `steady.miss_bytes_per_token.mean` for `lru` and `belady` at the same budget |
| per-layer hit rate | `runs[].per_layer_hit_rate` (30 values) |
| projected tg_serial and tg_overlap per Pi profile with assumptions listed (kind=projected) | `runs[].projection.{tg_serial, tg_overlap, verdict, assumptions}` |
| stack-distance curve vs simulator LRU agreement | control C4: `sim.json` `checks.lru_vs_stack_distance`, plus the independent re-check above against `lru_hit_curve` |

### Pre-registered thresholds

> Decision rule for the Pi recommendation: a configuration is 'viable for streaming' if LRU (no oracle) gives tg_overlap >= 2.0 and tg_serial >= 1.0 tok/s; 'weak' if tg_serial 0.5-1.0; else not viable. Belady reported as the upper bound only. If LRU bytes/token > 1.5x Belady, prefetch/static policies are worth implementing on the Pi; if static is within 10% of LRU, hot-set pinning is the simpler deployment. Any projection whose inputs are still 'assumed' is printed with that word.

| Question | Rule |
|---|---|
| Is a Pi configuration viable for streaming? | LRU (no oracle): tg_overlap ≥ 2.0 and tg_serial ≥ 1.0 |
| Weak | tg_serial between 0.5 and 1.0 |
| Not viable | otherwise |
| Belady | upper bound only |
| Are prefetch or static policies worth implementing on the Pi? | LRU bytes per token > 1.5 × Belady |
| Is hot-set pinning the simpler deployment? | static within 10% of LRU |
| Labelling | any projection with an assumed input says "assumed" |

### Results template (E4)

| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value, source) |
|---|---|---|---|
| LRU steady hit rate / weighted hit rate at 5.17 GB (the 8 GB profiles) | — | _pending_ (inputs: E3 trace, map26; simulated) | 0.861 (catalog prior curve at 40.2% resident) |
| LRU miss MB/token mean / p95 at 5.17 GB | — | _pending_ | 111.7 MB mean (prior: (1 − 0.861) × 802.9 MB) |
| Belady gap (LRU ÷ Belady miss bytes) at 5.17 GB | — | _pending_ | — |
| static vs LRU miss bytes at 5.17 GB | — | _pending_ | — |
| lru_prefetch: stall bytes and wasted-prefetch fraction at 5.17 GB | — | _pending_ | — |
| Per-layer LRU hit rate, min / median / max at 5.17 GB | — | _pending_ | — |
| `pi5-4gb-microsd`: tg_serial / tg_overlap / verdict (LRU) | — | _pending_ | 0.12 / 0.12, NOT_VIABLE (worked example) |
| `pi5-8gb-usb3`: tg_serial / tg_overlap / verdict (LRU) | — | _pending_ | 1.46 / 2.87, STREAM_GOOD (prior); 0.78 / 1.04, STREAM_WEAK at a 0.60 hit rate |
| `pi5-8gb-nvme-gen3`: tg_serial / tg_overlap / verdict (LRU) | — | _pending_ | 1.94 / 2.87, STREAM_GOOD (prior); 1.26 / 2.09 at a 0.60 hit rate |
| `pi5-16gb-nvme-gen3`: verdict and in-RAM floor | — | _pending_ | FITS_RAM, 2.73 tok/s (10 GB/s × 0.65 ÷ 2.38 GB) |
| Dense 31B line: tok/s at mem_bw, s/token at storage bandwidth | — | _pending_ | 0.57 tok/s at 10 GB/s (efficiency 1.0); 20.7 s/token at 850 MB/s |
| C4: `checks.lru_vs_stack_distance` and the largest abs difference between sim LRU and lru_hit_curve | _pending_ (an exact check, not a projection) | — | — |

**Verdict vs pre-registered threshold:** _pending_. Give each Pi profile its verdict word with tg_serial and tg_overlap, and answer the prefetch, static and pinning questions, citing `results/h1/sim.json` and naming which inputs are still assumed.

**What would change this conclusion:**
- E5 finding the simulator off by more than ±30%. Projections are then scaled by the measured ratio,
  and the result says so.
- Measured Pi constants from pi/03 (`results/pi/io_mem.json`) and pi/05 (`results/pi/calibration.json`)
  replacing the assumed bandwidth, latency and efficiency.
- A trace that is generation-heavy or in another language (see C3).

## E5: page-cache validation

| Item | Value |
|---|---|
| Manifest id | `E5-pagecache-validation` |
| Results `experiment` | `E5` |
| Driver | [pc/pagecache_limit.sh](../pc/pagecache_limit.sh) (a host side, plus `--inner` inside the container) |
| Host and model | desktop Docker; the 26B GGUF copied once into the ext4 volume `pigemma-models`; cgroup memory caps of 3, 4, 5, 6 and 8 GB |
| Runtime (manifest) | volume copy ~10-15 min once; ~5 min per cap (load storm under the cap + 1500 tokens at -ub 1) -> ~45 min |
| Needs | the E0 repack gate PASSED; the E3 trace (`all.moetrace.tokens.bin`); E4; VM root at least 16 GB free and C: at least 18 GB free. Otherwise `--allow-bind-mount` gives lower-bound bytes only, flagged as not representative |
| Writes | `results/h1/validation.json` (points appended by `python -m pigemma.cache_sim io-point`), `results/h1/io_<cap>.csv`, `results/h1/sim_e5_<cap>g.json` |

### Hypothesis

> With `-nr`, the simulator's LRU bytes/token predicts real page-cache behaviour within +-30% when the cache budget equals the cgroup limit minus anon RSS minus the resident non-expert bytes, and the time model predicts tok/s within +-35% using the measured proxy tier bandwidth.

### Procedure

> bash pc/pagecache_limit.sh --mem 3,4,5,6,8 --range 0:1500: 26B GGUF copied once into the ext4 Docker volume (VM/C: disk gated), per cap: vmtouch -e, fio randread bs=3345k on the volume (tier bandwidth), pigemma-probe --replay all.moetrace.tokens.bin -ub 1 -nr -t 6 --io-csv under --memory=&lt;N>g --memory-swap=&lt;N>g --cpus 6, record /proc/self/io read_bytes and majflt per token, cgroup memory.stat and memory.events (OOM detection), plus a plain llama-completion -nr -n 64 tok/s; compare with cache_sim lru/belady at budget = cap - anon_rss - 1.577 GB - 0.1 GB on the same token range after a 200-token warm-up.

Commands (Git Bash):

```bash
bash pc/docker_run.sh vm-df
bash pc/pagecache_limit.sh --mem 3,4,5,6,8 --tokens traces/26b/all.moetrace --range 0:1500 --out results/h1/validation.json
```

What the inner step runs in each `--memory=<cap>g --memory-swap=<cap>g --cpus 6` container (abridged
from the manifest):

```bash
vmtouch -e /models/gemma-4-26B_q4_0-it.gguf
fio --name=tier --filename=/models/gemma-4-26B_q4_0-it.gguf --rw=randread --bs=3345k --direct=1 --readonly --iodepth=1 --runtime=15 --time_based --output-format=json
tools/build-linux-x86/bin/pigemma-probe --mode moe-trace -m /models/gemma-4-26B_q4_0-it.gguf --replay /work/traces/26b/all.moetrace.tokens.bin --replay-range 0:1500 -ub 1 -nr -t 6 -c 2048 --io-csv /tmp/io.csv --trace-out /tmp/replay.moetrace
```

On the host, the simulator then runs at the matching budget, which is
`cap − measured anon RSS − 1.577 GB − 0.1 GB`, over the same token range with a 200-token warm-up.
Below are the budgets planned with an anonymous RSS of 0.6 GB; the measured value replaces it.

| Cap | Simulator budget | Items | Share of the experts |
|---|---|---|---|
| 3 GB | 0.72 GB | 216 | 5.6% |
| 4 GB | 1.72 GB | 514 | 13.4% |
| 5 GB | 2.72 GB | 813 | 21.2% |
| 6 GB | 3.72 GB | 1,112 | 29.0% |
| 8 GB | 5.72 GB | 1,710 | 44.5% |

The comparison for one cap (5 GB shown). pagecache_limit.sh runs it for each cap; `io-point` appends
one point to `validation.json`:

```powershell
python -m pigemma.cache_sim traces/26b/all.moetrace --map results/map26.json --policies lru,belady --budgets-gb 2.72 --token-range 0:1500 --warmup-tokens 200 --no-envelopes --out results/h1/sim_e5_5g.json
python -m pigemma.cache_sim io-point --io-csv results/h1/io_5.csv --sim results/h1/sim_e5_5g.json --mem-cap-bytes 5000000000 --anon-rss-bytes <measured> --tier-mbps <fio> --ram-tok-s <uncapped tok/s> --warmup-tokens 200 --page-size 4096 --token-range 0:1500 --validation results/h1/validation.json
```

`io-point` computes:
- measured bytes per token = sum(`read_bytes_delta`) ÷ tokens, after the warm-up;
- measured tok/s = 1000 × tokens ÷ sum(`wall_ms`);
- predicted tok/s = 1 / (1/ram_tok_s + sim_lru_bytes/tier + n_miss × access_ms), with `access_ms`
  defaulting to 0 because the QD1 fio 3 MB reads already contain the per-request latency.

On the bind-mount fallback, bytes per token are `majflt × page_size`, a lower bound.

### Metrics

| Metric (manifest) | Recorded as (`validation.json` `points[]`, [docs/TRACE_FORMAT.md](TRACE_FORMAT.md) section 12) |
|---|---|
| measured read bytes per token and majflt per token per cap | `measured_bytes_per_token`, `measured_majflt_per_token` (from `io.csv` after the warm-up) |
| measured tok/s (replay -ub 1 and natural generation) per cap | `measured_tok_s` (replay at `-ub 1`); natural-generation tok/s from the plain `llama-completion -n 64` run |
| tier bandwidth | `tier_rand_read_3mb_mbps` (fio) |
| sim LRU and Belady bytes/token at the matching budget and the ratio | `sim_lru_bytes_per_token`, `sim_belady_bytes_per_token`, `ratio_sim_lru_over_measured`, `sim_budget_bytes` |
| predicted vs measured tok/s | `predicted_tok_s` (the projected sub-block, with its assumptions) against `measured_tok_s` |
| OOM events (must be 0 with -nr) | `oom` (from `memory.events` oom_kill) |

### Pre-registered thresholds

> PASS if |sim_lru/measured - 1| <= 0.30 for caps 4-8 GB and measured lies within [0.8 x Belady, 1.3 x LRU]; time model PASS if |predicted/measured - 1| <= 0.35 on >= 3 caps. If measured >> sim: readahead inflation (record the VM device read_ahead_kb, feed the inflation factor into --read-granularity and document). If measured << sim: the kernel keeps more than the limit charged (compare memory.stat file vs anon) -> adjust the budget formula and document. Any OOM means repack leaked (check buffer_split) -> fix before continuing.

| Check | Pass condition |
|---|---|
| Simulator bytes | abs(sim_lru / measured − 1) ≤ 0.30 for caps 4-8 GB, and 0.8 × Belady ≤ measured ≤ 1.3 × LRU |
| Time model | abs(predicted / measured − 1) ≤ 0.35 on at least 3 caps |
| OOM | no OOM events; any OOM means the repack leaked, which is fixed before continuing |
| Measured far above the simulator | readahead inflation: record `read_ahead_kb`, feed the factor into `--read-granularity`, document it |
| Measured far below the simulator | the kernel keeps more than the limit charges: compare `memory.stat` file against anon, adjust the budget formula, document it |

### Results template (E5)

| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value, source) |
|---|---|---|---|
| Measured read MB/token, caps 3 / 4 / 5 / 6 / 8 GB | _pending_ per cap | — | — |
| Measured majflt/token, per cap | _pending_ per cap | — | — |
| Measured tok/s, replay at -ub 1, per cap | _pending_ per cap | — | — |
| Measured tok/s, natural generation (-n 64), per cap | _pending_ per cap | — | — |
| Tier bandwidth, fio randread 3,345 KiB (MB/s) | _pending_ (a Docker VM disk, not a Pi number) | — | — |
| Anonymous RSS (MB), per cap | _pending_ per cap | — | — |
| Simulator LRU / Belady bytes per token at the matching budget | — | _pending_ per cap (simulated) | — |
| ratio sim_lru / measured, per cap | _pending_ (derived) | — | — |
| Predicted vs measured tok/s, per cap | — | _pending_ (inputs: tier MB/s, uncapped RAM bandwidth) | — |
| OOM events | _pending_ | — | 0 expected with `-nr` |

**Verdict vs pre-registered threshold:** _pending_. Give the simulator-bytes check and the time-model check as PASS/FAIL, with each cap's ratio, the OOM count and `results/h1/validation.json`.

**What would change this conclusion:**
- Readahead inflation (the VM's `read_ahead_kb`).
- The kernel charging page cache differently from what the budget formula assumes.
- E5 validates the cache model, not Pi NVMe speed, because the proxy tier is a virtual disk. The Pi
  repeat (`pi/05_run.sh stream-bench` and `pi/06_pin_hotset.sh` on a Pi that is not
  `context-friend-1`) has the final word.

## E6: hot-set pinning proxy

| Item | Value |
|---|---|
| Manifest id | `E6-hotset-pinning-proxy` |
| Results `experiment` | `E6` |
| Driver | [pc/pagecache_limit.sh](../pc/pagecache_limit.sh) `--pin`, with `python -m pigemma.gguf_map hotset` and [tools/pinner/pinner.c](../tools/pinner/pinner.c) |
| Host and model | desktop Docker; the same volume and container as E5; caps of 5 and 6 GB |
| Runtime (manifest) | ~20 min |
| Needs | E5's unpinned points at 5 and 6 GB; E3 `all.stats.json`; `results/map26.json` |
| Writes | `results/h1/hotset_5.json`, `results/h1/hotset_6.json` and their `.ranges.txt` files; `results/h1/validation_pin.json` |

### Hypothesis

> mlock-pinning the non-expert weights plus the hottest experts beats plain page-cache LRU at the same memory cap by >= 15% tok/s or >= 20% fewer bytes/token, because LRU otherwise evicts always-needed attention/head/dense pages under pressure.

### Procedure

> Same as E5 at caps 5 and 6 GB with --pin: hotset from results/map26.json + all.stats.json at budget = cap - 0.6 GB anon - 0.3 GB slack (non-expert 1.577 GB first, then hottest experts), pinner --verify in the same container/cgroup before the run, compared against the unpinned run of E5.

Commands. The page size is 4096 inside Docker Desktop's x86 VM; the Pi kernel uses 16384.

```powershell
python -m pigemma.gguf_map hotset --map results/map26.json --stats traces/26b/all.stats.json --budget-bytes 4.1e9 --page-size 4096 --out results/h1/hotset_5.json --ranges-out results/h1/hotset_5.ranges.txt
python -m pigemma.gguf_map hotset --map results/map26.json --stats traces/26b/all.stats.json --budget-bytes 5.1e9 --page-size 4096 --out results/h1/hotset_6.json --ranges-out results/h1/hotset_6.ranges.txt
```

```bash
bash pc/pagecache_limit.sh --mem 5,6 --pin --skip-copy --tokens traces/26b/all.moetrace --range 0:1500 --out results/h1/validation_pin.json
```

Inside the capped container, the pinner runs in the same cgroup as the probe, so locked pages count
against the cap. It is stopped with SIGTERM to its own pidfile only:

```bash
tools/build-linux-x86/bin/pinner --gguf /models/gemma-4-26B_q4_0-it.gguf --ranges /work/results/h1/hotset_5.ranges.txt --verify --json --pidfile /tmp/pinner.pid
```

The pinned point is recorded the same way as in E5, with the pin fields set:

```powershell
python -m pigemma.cache_sim io-point --io-csv results/h1/io_5_pin.csv --sim results/h1/sim_e5_5g.json --mem-cap-bytes 5000000000 --pinned --pin-budget-bytes 4100000000 --pin-locked-bytes <pinner locked_bytes> --anon-rss-bytes <measured> --tier-mbps <fio> --ram-tok-s <uncapped tok/s> --warmup-tokens 200 --page-size 4096 --token-range 0:1500 --validation results/h1/validation_pin.json
```

Planned pin budgets (cap − 0.6 GB anonymous − 0.3 GB slack), before the page-alignment overhead:

| Cap | Pin budget | Non-expert part | Expert part | Experts pinned | Share of the experts |
|---|---|---|---|---|---|
| 5 GB | 4.1 GB | 1.577 GB | 2.52 GB | about 754 | 19.6% |
| 6 GB | 5.1 GB | 1.577 GB | 3.52 GB | about 1,053 | 27.4% |

There is no pinning on `context-friend-1`: the safety rule keeps the page cache reclaimable for its
services, and `ulimit -l` there is 8192 kB (D4). A Pi A/B run (`pi/06_pin_hotset.sh ab`) needs a
non-production Pi.

### Metrics

| Metric (manifest) | Recorded as |
|---|---|
| tok/s pinned vs unpinned | `validation_pin.json` points with `pinned: true`, against the E5 points (`pinned: false`) at the same cap |
| read bytes/token pinned vs unpinned | `measured_bytes_per_token` pinned against unpinned |
| pinner resident fraction (mincore) before and after the run | the pinner's `--json` summary `resident_bytes`, from `--verify-only` before and after the run |
| locked bytes vs requested | the pinner's `locked_bytes` against `requested_bytes`, and its exit code (0 = all, 2 = partial, 3 = refused) |

### Pre-registered thresholds

> PASS if pinned tg >= 1.15x unpinned at 5 GB OR bytes/token reduced >= 20%. If no gain, document that the kernel LRU already keeps the hot pages and make pinning optional (not default) in pi/05; if the pinner cannot lock the full budget (exit 2), report the achieved fraction.

| Outcome | Rule |
|---|---|
| PASS | pinned tg ≥ 1.15 × unpinned at 5 GB, OR bytes per token reduced by ≥ 20% |
| No gain | the kernel LRU already keeps the hot pages: pinning becomes optional (not the default) in pi/05 |
| Partial lock (pinner exit 2) | report the achieved fraction, `locked_bytes / requested_bytes` |

### Results template (E6)

| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value, source) |
|---|---|---|---|
| tok/s pinned / unpinned at 5 GB (ratio) | _pending_ | — | — |
| tok/s pinned / unpinned at 6 GB (ratio) | _pending_ | — | — |
| Read MB/token pinned / unpinned at 5 GB and 6 GB | _pending_ | — | — |
| Pinner resident fraction (mincore), before / after | _pending_ | — | — |
| Locked bytes / requested bytes (pinner exit code) | _pending_ | — | — |
| Expert-access coverage of the pinned set (hotset `expert_coverage_fraction`) | — | _pending_ (from E3 stats) | — |

**Verdict vs pre-registered threshold:** _pending_. Give PASS or no gain at 5 GB and at 6 GB, with the ratios and `results/h1/validation_pin.json`.

**What would change this conclusion:**
- Pinning can hurt by stealing page cache from the misses at small caps. That is a finding, not a
  bug.
- A real Pi can behave differently: it has 16 KiB pages, different readahead and microSD or NVMe
  latency. A Pi A/B on a non-production board is the final word.

## E7: H2 dense oracle

| Item | Value |
|---|---|
| Manifest id | `E7-h2-dense-oracle` |
| Results `experiment` | `E7` |
| Driver | [pc/h2_run.ps1](../pc/h2_run.ps1): probe `--mode act-dump` in Docker, then [pigemma/h2/decompose.py](../pigemma/h2/decompose.py) and [pigemma/h2/oracle.py](../pigemma/h2/oracle.py) natively |
| Host and model | desktop (Docker for the capture, native for the numerics); `31b-q4` |
| Runtime (manifest) | act-dump ~20 min (17.65 GB first read through the bind mount + 512 tokens at ~10 tok/s prefill); decompose + oracle ~25-35 min for 4 layers x 3 matrices x 2 bases |
| Needs | the probe built; the bench lock (the capture loads the 31B, and the numerics are serialised behind the lock); `data/calibration/all.txt` (built in E2) |
| Writes | `traces/h2/acts/31b/` (about 33 MB per layer at 512 tokens), `traces/h2/decomp/`, `results/h2/oracle.json`, `docs/results/H2_RESULTS.md` |

### Hypothesis

> For the dense 31B, W = W_base + sum R_g with a cheap base (2-bit group quant or rank-256) needs only a minority of residual bytes per token to keep FFN output error <= 3%, the norm bound certifies a useful share of the skippable groups, and this beats plain lower-bit quantisation at equal bytes. Expected honest outcome: the bound is loose (~1/sqrt(g)) and the bytes saved do not approach what a Pi needs.

### Procedure

> .\pc\h2_run.ps1 -Layers 5,20,35,50 -Tokens 512 -Group 128 -Bases q2,lowrank256: act-dump of ffn_norm / ffn_geglu / ffn_out in Docker (-nr, 384 prefill + 128 generated per chunk); decompose (q2 block-32, randomized SVD rank 256, groups of 128 input columns, spectral norms, residual at 4 bits); oracle at eps 1/3/10% with the sound certification rule, iso-byte control, teacher-forced vs generated split, extrapolation to 60 layers.

Commands. Run `-Quick` first (layers 20 and 45, 128 tokens, q2 only), then the full sample:

```powershell
.\pc\h2_run.ps1 -Quick
.\pc\h2_run.ps1 -Layers 5,20,35,50 -Tokens 512 -Group 128 -Bases q2,lowrank256
```

What the driver runs:

```bash
bash pc/docker_run.sh -- python -m pigemma.paths lock --hold -- tools/build-linux-x86/bin/pigemma-probe --mode act-dump -m /work/models/google__gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf -f /work/data/calibration/all.txt --chunk-tokens 384 --gen 128 -nr -t 12 -c 1024 -b 512 -ub 128 --layers 5,20,35,50 --kinds ffn_norm,ffn_geglu,ffn_out --tokens 512 --act-out /work/traces/h2/acts/31b
```

```powershell
python -m pigemma.gguf_map build 31b-q4 --out traces/h2/map31.json --reuse
python -m pigemma.h2.decompose --map traces/h2/map31.json --layers 5,20,35,50 --mats gate,up,down --bases q2,lowrank256 --group 128 --res-bits 4 --out traces/h2/decomp/
python -m pigemma.h2.oracle --decomp traces/h2/decomp/ --acts traces/h2/acts/31b/ --tokens 512 --eps 0.01,0.03,0.10 --out results/h2/oracle.json --md docs/results/H2_RESULTS.md
```

**Pre-registered arithmetic.** This follows from the byte counts alone, before any activation is seen.
Each FFN matrix of the 31B has 5,376 × 21,504 = 115,605,504 weights, and there are 60 layers × 3
matrices = 180 of them. Attention takes 4,770,160,640 B and the tied head 1,156,076,544 B per token,
5,926,237,184 B together (design-panel constants; `map31.json` files 1.4 MB of attention norms under
`norm`).

| FFN representation | FFN bytes per token | Plus attention and head | Total per token | Against 8.5 GB (31B IQ2_XXS file) |
|---|---|---|---|---|
| Q4_0 (today: 0.5625 B/weight) | 11.705 GB | 5.926 GB | 17.63 GB | above |
| q2 base only (0.3125 B/weight), zero residual | 6.503 GB | 5.926 GB | 12.43 GB | above: PROMISING is unreachable with the q2 base |
| q3 base only (0.4375 B/weight), zero residual | 9.104 GB | 5.926 GB | 15.03 GB | above |
| lowrank256 base only (2 × 256 × 26,880 B per matrix), zero residual | 2.477 GB | 5.926 GB | 8.40 GB | 0.097 GB of room, which is 0.82% of a 4-bit residual per matrix |

So with these two bases, the `implied bytes/token < 8.5 GB` clause of PROMISING can only be met with the
lowrank256 base and a residual fraction under about 0.8% at eps 3%. Independently of H2, attention and
the head alone (5.93 GB per token) cap the dense model at 1.69 tok/s at 10 GB/s with an efficiency of
1.0 (1.10 at 0.65), even entirely in RAM. Q4_0 streamed at 850 MB/s takes 20.7 s per token. The
measured real-Pi point is in E8: the 31B UD-IQ3_XXS, held entirely in RAM, ran at 0.53 tok/s.

### Metrics

| Metric (manifest) | Recorded as (`oracle.json`, [docs/TRACE_FORMAT.md](TRACE_FORMAT.md) section 13) |
|---|---|
| sanity rel_err of W_down_q4 @ ffn_geglu vs captured ffn_out (median) | `sanity[].rel_err_median` (control C6) |
| oracle residual byte fraction at eps 1/3/10% (mean, p90) per layer x matrix x base x phase | `per_case[].oracle["0.01"/"0.03"/"0.10"].frac_bytes_mean`, `.frac_bytes_p90` |
| error-vs-bytes curve | `per_case[].curve` |
| certified-skip fraction, certification efficiency, soundness violations, tightness p10/p50/p90, Spearman | `per_case[].bound[eps].{certified_frac_bytes_mean, certification_efficiency, soundness_violations}`, `tightness`, `spearman_bound_vs_actual` |
| iso-byte: uniform q2/q3 error vs base+oracle vs base+certified | `per_case[].iso_byte[]` (control C5) |
| implied bytes/token (attention 4.77 GB + head 1.16 GB + FFN) vs 17.64 GB (Q4_0) and 8.5 GB (IQ2_XXS), s/token at 850 MB/s, tok/s at 10 GB/s | `summary.implied_bytes_per_token` (kind projected), `s_per_token_nvme_850mbps`, `tok_s_in_ram_10gbps` |

### Pre-registered thresholds

> Pipeline VALID only if sanity median rel_err < 0.03 and soundness_violations == 0. PROMISING only if at eps 3% the oracle residual fraction <= 25% AND certification efficiency >= 50% AND implied bytes/token < 8.5 GB AND base+oracle error < uniform-quant error at equal bytes; WEAK if fraction <= 50% or efficiency 0.2-0.5; NO_ADVANTAGE if uniform q3 error <= 1.2x the oracle error at equal bytes; NOT_SUPPORTED otherwise. Regardless: report s/token on a Pi (any number > 2 GB/token is < 0.5 tok/s from NVMe) and that attention + head alone (5.9 GB/token) cap the dense model at ~1.7 tok/s even if it were entirely in RAM.

| Verdict | Rule |
|---|---|
| Pipeline VALID | sanity median rel_err < 0.03 and soundness_violations == 0 (C6) |
| PROMISING | at eps 3%: residual fraction ≤ 25% AND certification efficiency ≥ 50% AND implied bytes per token < 8.5 GB AND base+oracle error < uniform-quant error at equal bytes |
| WEAK | residual fraction ≤ 50%, or certification efficiency 0.2-0.5 |
| NO_ADVANTAGE | uniform q3 error ≤ 1.2 × the oracle error at equal bytes (C5) |
| NOT_SUPPORTED | otherwise |
| C3 (added by this plan) | the verdict must hold on the `gen` phase as well as on `all` |

### Results template (E7)

| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value, source) |
|---|---|---|---|
| Sanity: median rel_err of W_down_q4 @ ffn_geglu vs ffn_out, per layer (C6) | _pending_ | — | — |
| Soundness violations (must be 0) | _pending_ | — | — |
| Oracle residual byte fraction at eps 3%, mean / p90, q2 and lowrank256 | _pending_ (sampled layers) | — | — |
| Oracle residual byte fraction at eps 1% and 10% | _pending_ | — | — |
| Certified-skip fraction and certification efficiency at eps 3% | _pending_ | — | — |
| Bound tightness p10 / p50 / p90; Spearman(bound, actual) | _pending_ | — | about 1/sqrt(g) = 0.088 for random directions at g = 128 (the manifest's expectation) |
| Iso-byte: uniform q2 / q3 error vs base+oracle vs base+certified (C5) | _pending_ | — | — |
| Prefill vs gen phase: fraction at eps 3% (C3) | _pending_ | — | — |
| Implied bytes per token over 60 layers | — | _pending_ (extrapolated from 4 sampled layers) | 17.64 GB for Q4_0 (exact, from the file) |
| s/token from NVMe Gen3 at 850 MB/s | — | _pending_ | 20.7 s for Q4_0; 850 MB/s is the catalog's NVMe Gen3 sequential read (assumed) |
| tok/s in RAM at 10 GB/s | — | _pending_ | 1.69 tok/s ceiling from attention and head alone (efficiency 1.0) |
| Verdict word (PROMISING / WEAK / NOT_SUPPORTED / NO_ADVANTAGE) | — | _pending_ | — |

**Verdict vs pre-registered threshold:** _pending_. Give pipeline VALID/INVALID, then the verdict word with the four PROMISING clauses checked one by one, citing `results/h2/oracle.json` and `docs/results/H2_RESULTS.md`.

**What would change this conclusion:**
- A base that is much cheaper per byte than q2 or rank 256, such as vector quantisation.
- Compressing attention as well. Its 5.93 GB per token is outside H2.
- A certification bound tighter than the spectral norm.
- Activations from longer, free-running generations.
- Even then, the attention-and-head ceiling above stays.

## E8: Pi runtime dry run

| Item | Value |
|---|---|
| Manifest id | `E8-pi-runtime-dry-run` |
| Results `experiment` | `E8` |
| Driver | Stage A: pytest, `bash -n`, [pc/docker_run.sh](../pc/docker_run.sh) with the `pi/*.sh` scripts, [pigemma/feasibility.py](../pigemma/feasibility.py). Stage B: `pi/00`..`pi/05` on `context-friend-1` (added by this plan, D1) |
| Host | Stage A: desktop, Docker and emulated arm64. Stage B: the real Raspberry Pi 5 16 GB |
| Runtime (manifest) | 10 min (Stage A); Stage B about 1 h, dominated by a 12.7 GB download over Wi-Fi at about 7-8 MB/s and a 2.5 min cold load |
| Needs | Stage A: the `pi/*.sh` scripts and their tests. Stage B: `PIGEMMA_I_KNOW=1`, the safety rules (ground rule 8) and [docs/PI_RUNBOOK.md](PI_RUNBOOK.md) |
| Writes | Stage A: test output only. Stage B: `results/pi/*.json`, `pi/out/*.json` |

### Hypothesis

> The Pi scripts are correct before a Pi exists: verdict logic, flag selection, safety refusals, and the native build tree configure on generic Linux.

### Procedure

> pytest tests/test_feasibility.py tests/test_pi_scripts.py tests/test_catalog.py (fixtures: 4GB+microSD/context-friend-1, 8GB+USB3, 8GB+NVMe, 16GB+NVMe, busy host); bash -n and shellcheck on pi/\*.sh and pc/\*.sh; `pc/docker_run.sh -- bash pi/00_feasibility.sh` and `-- bash pi/01_tune_os.sh` (dry run) inside the image; pinner integration on a 64 MiB synthetic file in Docker; no command is ever run on context-friend-1.

**Stage A** (desktop; no model inference):

```powershell
python -m pytest tests/test_feasibility.py tests/test_pi_scripts.py tests/test_catalog.py
python -m pytest -m integration tests/test_pi_scripts.py
```

```bash
for f in pi/*.sh pi/lib/*.sh pc/*.sh; do bash -n "$f" || echo "LINT FAIL: $f"; done
if command -v shellcheck >/dev/null; then shellcheck pi/*.sh pi/lib/*.sh pc/*.sh; else echo "shellcheck not installed: skipped"; fi
bash pc/docker_run.sh -- bash pi/00_feasibility.sh
bash pc/docker_run.sh -- bash pi/01_tune_os.sh
bash pc/docker_run.sh --pi-emu --memory 16g -- bash pi/00_feasibility.sh
```

| Fixture | Expected verdict | Expected exit |
|---|---|---|
| 4 GB + microSD (the old `context-friend-1` probe) | NOT_VIABLE | 3, or 4 with `contextfriend-*` services running |
| 8 GB + USB3 SSD | STREAM_SLOW or STREAM_OK, as hand arithmetic computes | 2 or 0 |
| 8 GB + NVMe Gen3 | STREAM_OK, with flags containing `-nr` | 0 |
| 16 GB + NVMe Gen3 | FITS_MAPPED with `-nr -lm mmap+mlock`, FITS_REPACKED false | 0 |
| Busy host | refused | 4 |

**Stage B** (the real Pi, `context-friend-1`, added by this plan, D1). Run as the user `pi` from the
repository checkout, one model on disk at a time, at `nice -n 10`:

```bash
export PIGEMMA_I_KNOW=1
bash pi/00_feasibility.sh --want 26b --json pi/out/feasibility.json
bash pi/02_build_llama.sh --with-tools
bash pi/03_bench_io_mem.sh --quick
bash pi/04_fetch_model.sh --model 26b --quant UD-Q3_K_M
bash pi/05_run.sh bench --variants norepack
bash pi/05_run.sh stream-bench
bash pi/00_feasibility.sh --want 26b
```

Notes for Stage B:
- **pi/02:** llama.cpp a7b94df is already built in `~/llama.cpp` (by `pi/adhoc/build_llama_native.sh`,
  410 s). `--with-tools` adds `pigemma-probe`, `pinner` and `membw`.
- **pi/03:** fio is not installed. pi/03 falls back to dd for sequential reads, and installing fio
  needs the operator's approval.
- **The second pi/00 run** picks up `results/pi/io_mem.json` and `results/pi/calibration.json`
  automatically, so the verdict is recomputed with measured inputs.
- **No mlock (D4):** the FITS_MAPPED flags contain `-lm mmap+mlock`, but on `context-friend-1` they are
  used without `+mlock`. `pi/pi_config.env` has `ALLOW_MLOCK=auto`, which forbids mlock of model
  weights on protected hosts. Check that the effective llama flags pi/05 prints contain no `mlock`. If
  they do, stop and run the benchmark by hand instead, then log it as a deviation:

```bash
nice -n 10 ~/llama.cpp/build/bin/llama-bench -m models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf -t 4 -p 64 -n 32 -r 3 --repack 0 -lm mmap -o json > results/pi/llama-bench_26b-ud-q3_k_m_norepack.json
python3 -m pigemma.bench parse-bench results/pi/llama-bench_26b-ud-q3_k_m_norepack.json --model models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf --variant norepack --flags "-lm mmap" --experiment E8 --out results/pi/
```

`parse-bench` turns the raw llama-bench JSON into results envelopes. The raw file is kept as evidence;
`results summarize` skips it because it is not an envelope. On aarch64 hosts, bench.py's matrix,
completion and parse subcommands default to `experiment: E8`, so pi/05 runs land in E8 as well.

Expectations registered before Stage B (PROJECTED unless marked otherwise):

| Quantity | Expected | Inputs |
|---|---|---|
| `26b-ud-q3_k_m` tg tok/s, 4 threads, `-nr`, file in RAM | 1.7-3.0 (central 2.1) | 3,044,443,256 B per token (catalog: 2,399,894,648 non-expert + 644,548,608 experts). Lower bound: the 6.27 GB/s effective weight traffic measured here for the 31B IQ3_XXS × 0.84, the desktop MoE/dense ratio (36.1 vs 43.0 GB/s). Central: 10 GB/s × 0.65 (assumed). Upper: about 9 GB/s, which the published Qwen3-30B-A3B result on a 16 GB Pi 5 implies |
| `26b-ud-q3_k_m` cold load from the microSD | about 148 s | 12,728,497,888 B at 85.8 MB/s (the measured 31B load: 11.84 GB in 138 s) |
| pi/00 verdict for `26b-ud-q3_k_m` | FITS_REPACKED (corrected from FITS_MAPPED, see D6); pi/05 still runs it with `-nr` on this protected host | 12.73 GB file against MemTotal 16,607,552 kB |
| `26b-q4` (Q4_0, 14.44 GB) on this SD card | not allowed | it would leave less than the 2 GB the safety rule requires (FACTS: 0.5 GB) |
| Any 31B quant | NEVER_GOOD_SPEED | dense: every weight byte is read for every token |

Rule (added by this plan):
- **"Projection holds"** if the measured 26B tg lies inside 1.7-3.0 tok/s.
- **Below 1.7:** the MoE kernel and routing overheads on the A76 are larger than on the desktop.
- **Above 3.0:** the effective bandwidth is better than assumed.

In both of the last two cases, `llama_efficiency` from `results/pi/calibration.json` replaces the 0.65
assumption, and E4's projections are recomputed. The speed words come from the catalog: ≥ 2.0 tok/s is
"good" and ≥ 1.0 is "viable". One caveat on the lower bound: IQ3_XXS kernels are compute-heavier than
Q3_K and Q4_K on the A76, so the 31B's 6.27 GB/s may be compute-limited. It is a lower bound on the
bandwidth the A76 can actually sustain.

### Metrics

| Metric (manifest) | Recorded as |
|---|---|
| tests green | pytest exit 0 for the three test modules (plus the integration run) |
| lint clean | `bash -n` prints nothing for every `pi/*.sh` and `pc/*.sh`; shellcheck when installed |
| verdict table renders for every fixture with the expected codes | `python -m pigemma.feasibility --json` per fixture: `rows[].verdict`, `exit_code`, `llama_flags` |
| pinner locks and verifies the synthetic file | the pinner's `--json` summary `status: ok`, `locked_bytes` equal to `requested_bytes`, `vm_lck_kb` |

### Pre-registered thresholds

> PASS: 4 GB + microSD -> NOT_VIABLE (exit 3; exit 4 with the protected services present); 8 GB + USB3 -> STREAM_SLOW or STREAM_OK as computed (asserted against hand arithmetic); 8 GB + NVMe -> STREAM_OK with flags containing -nr; 16 GB -> FITS_MAPPED with -nr -lm mmap+mlock and FITS_REPACKED false; busy -> exit 4; all scripts lint clean. On the real Pi later: pi/03 and pi/05 replace 'assumed' inputs and E4 projections are recomputed.

Stage B thresholds are the projected band and the rule above (added by this plan).

### Results template (E8)

| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value, source) |
|---|---|---|---|
| Stage A: pytest (feasibility, pi_scripts, catalog) | _pending_ | — | — |
| Stage A: lint (`bash -n`; shellcheck if installed) | _pending_ | — | — |
| Stage A: verdict and exit code per fixture | _pending_ | — | — |
| Stage A: pinner on a 64 MiB synthetic file in Docker | _pending_ | — | — |
| Stage A: pi/00 inside the emulated arm64 container (functional only) | _pending_ | — | — |
| Stage B: pi/00 verdict and exit code on context-friend-1 | _pending_ | FITS_REPACKED for 26b-ud-q3_k_m (D6) | — |
| Stage B: microSD sequential read MB/s (pi/03) | 91.6 MB/s raw O_DIRECT 4 MiB (context-friend-1, 2026-10-05, results/pi-real/README.md) | — | 85 MB/s (catalog `microsd`, assumed) |
| Stage B: microSD random read 3 MB (MB/s) and access ms (pi/03) | _pending_ | — | 60 MB/s, 2.0 ms (catalog `microsd`, assumed) |
| Stage B: memory bandwidth, read GB/s at 1 / 2 / 4 threads (membw) | _pending_ | — | 10 GB/s (catalog, assumed) |
| Stage B: 26b-ud-q3_k_m pp64 / tg32, `-nr`, 4 threads | _pending_ | tg 1.7-3.0, central 2.1 (inputs above) | — |
| Stage B: 26b-ud-q3_k_m repacked default variant (only if pi/00 says FITS_REPACKED) | _pending_ | — | — |
| Stage B: 26b-ud-q3_k_m cold load (s) | _pending_ | about 148 s | — |
| Stage B: 26b-ud-q3_k_m stream-bench read MB/token after warm-up | _pending_ | about 0 (the file fits in RAM) | — |
| Stage B: 26b-ud-q3_k_m EN and HE sample answers correct and coherent | _pending_ | — | — |
| Stage B: 31b-ud-iq3_xxs pp64 / tg32, 4 threads | 0.63 / 0.53 tok/s, `--repack 0` and `1` identical (context-friend-1, 2026-10-05, results/pi-real/31b-iq3xxs/bench-nr.md; ad hoc, before pi/05) | — | — |
| Stage B: 31b-ud-iq3_xxs cold load from microSD | 138 s for 11.84 GB (context-friend-1, 2026-10-05, results/pi-real/README.md) | — | — |
| Stage B: thermals during load | 72.5 °C peak, `throttled=0x0` over 23 min of the 31B run (results/pi-real/README.md) | — | — |
| Stage B: effective weight traffic, 31B (0.53 tok/s × 11.82 GB) | 6.27 GB/s (derived from the ad hoc measurement) | — | — |
| Stage B: llama_efficiency (`results/pi/calibration.json`) | _pending_ | — | 0.65 (assumed) |

**Verdict vs pre-registered threshold:** _pending_. Give Stage A PASS/FAIL per fixture and for lint. For Stage B, say whether the projection holds or is wrong (and in which direction), with the result files.

**What would change this conclusion:**
- A real-Pi verdict that contradicts a fixture verdict.
- A measured constant from pi/03 or pi/05 that moves a profile across a verdict boundary.
- New hardware on the free PCIe lane, such as an NVMe HAT or a USB 3 SSD. That makes the QAT Q4_0 26B
  possible: re-run pi/00.

## Results template and rendered outputs

### Column rules

- **MEASURED (host, date, result file):** `value, hostname [tags], YYYY-MM-DD, repo-relative file`.
  - Every number needs a file.
  - The tags are `container` and `emulated`, as in the rendered host column.
  - A value derived only from measurements (a ratio, an efficiency) is marked `derived`.
- **PROJECTED (inputs, assumptions):** `value; inputs: files; assumptions: name = value (source)`. The
  envelope with `kind: projected` carries the same assumptions.
- **ASSUMED (value, source):** `value (source)`. The source is the catalog's source string or a section
  of [docs/research/RESEARCH_SUMMARY.md](research/RESEARCH_SUMMARY.md). A calculation from assumed
  inputs only also goes here.
- **Cell markers:**
  - `_pending_`: not run yet.
  - `—`: this column does not apply.
  - `not run` and `n/a`: the same meaning as in the rendered summaries (`not run` = the metric is
    absent; `n/a` = present but null).
- **Each experiment ends with two lines.** The verdict line quotes the rule, the numbers and the files.
  The "what would change this conclusion" line was pre-filled here, and is amended (never deleted) at
  write-up.

Skeleton for an additional table:

```markdown
| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value, source) |
|---|---|---|---|
| <metric> | <value>, <host>, <date>, <file> | <value>; inputs: <files>; assumptions: <name = value> | <value> (<source>) |

**Verdict vs pre-registered threshold:** <word>: <numbers against the rule>, <files>

**What would change this conclusion:** <condition>
```

### Rendered outputs (docs/results/)

`python -m pigemma.results summarize` renders only results envelopes, in the same E0-E8 sections and
the same MEASURED / PROJECTED / ASSUMED sub-sections as this plan. `sim.json`, `validation.json` and
`oracle.json` have formats of their own ([docs/TRACE_FORMAT.md](TRACE_FORMAT.md) sections 11-13).

- cache_sim writes one E4 envelope (`kind: projected`) per profile and policy next to `sim.json`, so E4
  appears in the summary by itself.
- If pagecache_limit.sh does not also write E5/E6 envelopes (`kind: measured`), copy the headline
  numbers of `validation.json` and `validation_pin.json` into `H1_RESULTS.md` by hand, under the same
  three headings and each with its source file.

| File | Produced by | Command |
|---|---|---|
| `docs/results/DESKTOP_<date>.md` | `results.summarize` over `results/<date>/` (E0-E2); the last step of desktop_bench.ps1 | `python -m pigemma.results summarize results/<date> --md docs/results/DESKTOP_<date>.md --title "Desktop results <date>"` |
| `docs/results/H1_RESULTS.md` | `results.summarize` over `results/h1/` (E3-E6), plus the sim.csv and validation.json tables | `python -m pigemma.results summarize results/h1 --experiment E3,E4,E5,E6 --md docs/results/H1_RESULTS.md --title "H1 results: MoE expert streaming"` |
| `docs/results/H2_RESULTS.md` | `pigemma.h2.oracle --md` (E7) | the E7 oracle command |
| `docs/results/PI_<date>.md` | `results.summarize` over `results/pi/` (E8 Stage B) | `python -m pigemma.results summarize results/pi --experiment E8 --md docs/results/PI_<date>.md --title "Raspberry Pi 5 results <date>"` |

### Write-up checklist

1. Validate every envelope (exit 1 means a schema error; fix it, don't delete the file):

   ```powershell
   python -m pigemma.results validate results/<date> results/h1 results/pi
   ```

2. Render the four files in the table above.
3. Fill the results templates in this plan from the rendered files, keeping each number's file. Write
   the verdict lines. Amend the "what would change" lines where the data showed something new.
4. Update the MEASURED / PROJECTED / ASSUMED tables in [README.md](../README.md),
   [docs/HYPOTHESES.md](HYPOTHESES.md) and [docs/PI_RUNBOOK.md](PI_RUNBOOK.md) from the same files.
5. Whenever pi/03 or pi/05 produce measured constants, re-run the projections. Both
   `pigemma.feasibility` and `pigemma.cache_sim` read `results/pi/io_mem.json` and
   `results/pi/calibration.json` by default (`--io-mem auto --calibration auto`), so re-running E4's
   command and `pi/00_feasibility.sh` is enough. Any device number still taken from the catalog stays
   labelled assumed.
6. Results JSON and Markdown stay tracked. Traces, `.npy`/`.npz` files and logits stay untracked
   (`.gitignore`). the operator decides when to commit.

## Deviations log

Rows are only ever appended; existing rows are not edited.

| # | Date | Change | Reason | Numbers seen before the change? |
|---|---|---|---|---|
| D1 | 2026-10-05 | E8 gains Stage B on the real Pi `context-friend-1`, with `PIGEMMA_I_KNOW=1` and the safety rules. The manifest's "no command is ever run on context-friend-1" (E8 procedure and run-order rules) no longer applies to that host. | The board behind that hostname was replaced by a 16 GB Pi 5, and the operator authorised testing on it (FACTS, update of 2026-10-05). | Yes: the ad hoc 31B UD-IQ3_XXS run (0.53 tok/s) existed when this plan was frozen. No threshold tests it; it is an input to the E8 band. |
| D2 | 2026-10-05 | The real-Pi quants are the 3-bit `26b-ud-q3_k_m` and `31b-ud-iq3_xxs`, so E2's optional quant tier now matters for the Pi claim. A KLD for `31b-ud-iq3_xxs` is added as optional. | The 30 GB microSD cannot hold the Q4_0 files with at least 2 GB free. | No |
| D3 | 2026-10-05 | Thresholds added by this plan: the operational reading of "useful" in E1, C3's JS ≤ 0.10 bits, and E8 Stage B's projected band of 1.7-3.0 tok/s with its rule. | The manifest left them open. | No (the D1 number is an input to the band, not a test of it) |
| D4 | 2026-10-05 | No mlock of model weights on `context-friend-1`: FITS_MAPPED runs there use `-lm mmap`, not `-lm mmap+mlock`. E6-style pinning (pi/06) is not run on that host. | Safety rule: the production services need reclaimable page cache. `ulimit -l` is 8192 kB there. | No |
| D5 | 2026-10-05 | E0 also records the desktop memory bandwidth (`results/<date>/membw_desktop.json`, next to the E0 envelopes) as the input to `llama_efficiency`. E4 passes `--e0-dir results/<date>`. | cache_sim derives the desktop efficiency from a measured E0 `tg` and membw found under `--e0-dir` (default `results/e0`, which desktop_bench.ps1 does not write), but the manifest's E0 metrics did not list membw. | No |
| D6 | 2026-10-05 | The E8 Stage B expectation "pi/00 verdict for `26b-ud-q3_k_m`" is corrected from FITS_MAPPED to FITS_REPACKED. | `pigemma.feasibility` on the real-Pi profile (16.6 GB MemTotal, 0.9 GB OS reserve) computes FITS_REPACKED for the 12.73 GB file (FITS_MAPPED for `26b-q4`); the earlier cell was a transcription slip. The run itself is unchanged: pi/05 forces `-nr` on a protected host. | No (Stage B pi/00 has not run) |
| D7 | 2026-10-05 | E0 gains diagnostics outside the manifest procedure: (a) ABBA-interleaved llama-bench repeats of default vs norepack (26B at 4 and 6 threads, 31B at 6) into `results/2026-10-05/e0-interleave/`, with one run excluded as disturbed by a rule written after seeing it (last sample = minimum and < 50% of the first); (b) a 3-chunk llama-perplexity KLD preview of `-nr`, the default variant at batch 1 and a default re-run against the default logits, plus a chat-prompt greedy check, into `results/2026-10-05/e0-diag/`. No threshold was changed. | The desktop was not idle (background CPU 12–62%), so one sequential matrix could not resolve the `-nr` penalty. The 26B `-nr` raw-text completion diverged from default and needed an explanation. | Yes: the sequential matrix and the completion outputs had been seen. |
