# CF-Engine v1

CF-Engine runs owned C++ quantized expert projections inside complete Gemma 4
26B-A4B generation on the CPU-only Raspberry Pi 5. The first integration replaces
layer 0 gate/up and down: **two of the model's 60 MoE projection tensors**. The
remaining operations execute through the pinned reference and are reported as
fallbacks. E2B remains a separate product baseline; it has not replaced the 26B
objective or gained CF kernel coverage in this iteration.

The initial Pi run recorded 100 actual CF calls and identical greedy token IDs
for four short requests, including a repeated English prompt, German and Hebrew.
A separate diagnostic compared 5,372,928 projection output values with zero
tolerance failures. A later broader assembly capture nevertheless found logit
and downstream routing differences despite matching generated text. These
historical results establish integration, not general numerical equivalence.

The corrected tile4 implementations preserve the pinned A76 upstream reduction
order. Portable, NEON and assembly paths have zero output-bit mismatches in the
captured gate/up operator comparison at 1/2/4 threads. Corrected full generation
also passes **20 paired comparisons across 10 tasks**: NEON and baseline assembly
each match reference token IDs, text, full captured logits and ordered routing.
An independent raw-byte check confirms identical float32 logit bits, including
signed zeros, for all 122 paired logit records. See the
[corrected quality summary](../results/cf-engine/exact-quality-summary.json) and
[raw-bit check](../results/cf-engine/exact-quality-bits.json). Reproduce the
independent standard-library check without loading a model:

```powershell
python cf-engine/check_quality_bits.py --directory results/cf-engine/exact-quality --pairs results/cf-engine/exact-suite/quality-pairs.json --out results/cf-engine/exact-quality-bits.json
```

All ten reference answers are correct on inspection and finish at EOS. This is
limited evidence on synthetic/public tasks, not broad model quality, arbitrary
sampling equivalence or a universal bit-identity guarantee. Assembly remains
excluded from default builds and available as an opt-in experiment; performance
acceptance is separate from this correctness result.

Read [the architecture record](../docs/cf-engine/ARCHITECTURE.md),
[artifact format](FORMAT.md), and [baseline audit](../docs/cf-engine/AUDIT.md).
`CLAUDE.md` at the repository root records the current integration state.

## Build an isolated reference overlay

The reference revision is
`a7b94df2c616bc1f62a73b964b4a71cb0dcc488e`. `prepare.py` copies that exact revision
and adds narrow CPU hooks without changing the reference checkout or builds.
Use a new output path; it refuses to replace an existing overlay.

From the repository root on the desktop:

```powershell
python cf-engine/prepare.py --source third_party/llama.cpp --out cf-engine/vendor-llama-repro
docker run --rm -v REPO_ROOT:/work -w /work pi-gemma4-build:latest cmake -S cf-engine -B cf-engine/build-repro -DCF_LLAMA_DIR=/work/cf-engine/vendor-llama-repro -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF
docker run --rm -v REPO_ROOT:/work -w /work pi-gemma4-build:latest cmake --build cf-engine/build-repro -j 2 --target cf-run cf-kernel-test
docker run --rm -v REPO_ROOT:/work pi-gemma4-build:latest /work/cf-engine/build-repro/bin/cf-kernel-test
```

The Docker commands use the project's existing `pi-gemma4-build:latest` build
image. The executable is Linux x86 on that desktop; it is a correctness build,
not a Pi benchmark. Preparation of packed weights does not depend on desktop ISA.

On the Pi, from the staged source root `/tmp/cf-engine-work`, build for the
verified board capabilities:

```sh
cmake -S cf-engine -B cf-engine/build-pi-repro \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
cmake --build cf-engine/build-pi-repro -j 2 --target cf-run cf-kernel-test
cf-engine/build-pi-repro/bin/cf-kernel-test
```

This uses the already prepared `cf-engine/vendor-llama` overlay staged with the
source. The build does not enable i8mm or SVE. Runtime feature checks still govern
kernel dispatch. `pi_build.sh` and `pi_rebuild.sh` record the original staging
workflow; do not overwrite preserved binaries or result directories when making
a new comparison.

For assembly reproduction only, add `-DCF_EXPERIMENTAL_ASM=ON` to CMake.
For the actual upstream/Kleidi operator comparison, restore the pinned dependency
as described in [KLEIDIAI.md](../docs/cf-engine/KLEIDIAI.md), also add
`-DCF_BUILD_COMPARISONS=ON`, and build target `cf-operator-bench`. Without the
experimental assembly option, an explicit assembly request fails; it never
silently substitutes another kernel.

```sh
cf-engine/build-pi-repro/bin/cf-operator-bench \
  --plan /mnt/usb/pigemma/cf-engine/artifacts/layer0-tile4/manifest.json \
  --capture /mnt/usb/pigemma/cf-engine/results/capture/activations/blk.0.ffn_gate_up_exps.weight.json \
  --out /mnt/usb/pigemma/cf-engine/results/operator-repro.jsonl --iterations 61
```

The operator executable uses real captured activations and selected experts,
upstream's actual Q8 quantizer, actual native upstream repacker/GEMV and the
disclosed KleidiAI microkernel. It includes preparation and persistent-pool
synchronization. Its cache-sized and larger-than-L3 cases are not DRAM-counter
measurements. Full-generation alternating runs are generated by
`make_experiments.py` and executed by `pi_sweep.sh` under `pi_job.py`.

## Compile the two selected projections

The desktop source is the original Google QAT GGUF. It contains F32 and Q6_K
tensors as well as Q4_0; every tensor is inventoried. Both compact formats
preserve all scale bits and quantized codes and have byte-identical inverses.

```powershell
python cf-engine/compiler.py --model models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf --out cf-engine/artifacts/repro-layer0-tile4 --layers 0 --projections both --layout tile4 --alternative-source-identities results/cf-engine/split-identities.json
python cf-engine/compiler.py --verify cf-engine/artifacts/repro-layer0-tile4/manifest.json --model models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf
scp -r cf-engine/artifacts/repro-layer0-tile4 pi@context-friend-1:/mnt/usb/pigemma/cf-engine/artifacts/
```

Use a new artifact directory for each conversion. `--layout bitplane` creates a
separate artifact for `--cf-kernel lut`. The tile4 format is required by portable,
intrinsic dotprod and assembly candidates. Both selected payloads together occupy
428,212,224 bytes. The source mapping remains available, so up to that many bytes
of additional packed/source duplication must be accounted for.

Split identities are provenance supplied to the compiler. The native loader
hashes the actual selected source file set, then checks each selected tensor's
actual original bytes on first use. Compiler validation against `--model` uses
the original file set, not the alternative split declarations.

The native adapter is narrower than the layout format: selected tensors require
`K <= 2816`, `M <= 2816`, `E <= 128`, with the format's divisibility rules.
Initialization rejects an artifact outside that envelope. During inference,
graph cases beyond eight selected experts or 64 input tokens use the reported
reference fallback. The supported integration uses `-nr` so original source
tensor bytes remain available for identity checking.

Artifact integrity is not numerical input sanitization. The compiler deliberately
preserves every fp16 scale bit pattern, including NaN/Inf; neither it nor the
native loader currently scans weight scales for finiteness. The loader also
does not repeat the offline inverse-layout check. Use compiler-produced,
offline-verified artifacts from the chosen reference model; checksums alone do
not certify arbitrary supplied weights or a manually rewritten manifest.

## Run real generation with explicit coverage

On `pi@context-friend-1`, from `/tmp/cf-engine-work`, create a fresh evidence
directory. The monitor takes the shared benchmark lock and stops its own job if
service health, memory headroom, temperature, throttling or root disk space fails.
It does not change services, swap, caches or boot configuration.

```sh
base=/mnt/usb/pigemma/cf-engine
model=/mnt/usb/pigemma/models/26b-qat-q4_0/gemma-4-26B_q4_0-it-00001-of-00005.gguf
run_dir="$base/results/repro-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$run_dir"
python3 cf-engine/pi_job.py --out "$run_dir/job-cf" --timeout 1800 -- \
  cf-engine/build-pi-repro/bin/cf-run --engine cf \
  --cf-plan "$base/artifacts/repro-layer0-tile4/manifest.json" \
  --cf-kernel dotprod --cf-memory resident --cf-report "$run_dir/cf-report.json" \
  --mode run -m "$model" -nr -t 4 -c 512 -b 64 -ub 64 \
  --gen-prompts "$base/integration-prompts.json" --chat --server-stop \
  --requests-out "$run_dir/cf-requests.jsonl"
```

For the matched instrumented reference, use `--engine reference`, omit
`--cf-plan`, and choose fresh `reference` report/request/job names. Keep model,
prompt file, threads, context and batch flags identical. For the diagnostic
operator comparison, add `--cf-validate` to CF mode and use fresh `validate`
output names. That mode also executes the reference and retains its outputs;
its timings are not standalone CF performance and its continuation does not
test accumulated CF numerical error.

Available kernel options are `portable`, `dotprod`, `lut`, `asm`, `asm-unroll2`
and `asm-prefetch512`. Assembly is built only for Linux AArch64 and explicitly
fails initialization when unavailable. `dotprod` reports its portable fallback
when the guarded ARM path is unavailable. `CF_REPORT` records the executed
kernel, CF/fallback operation counts, operation times, preparation/multiplication
times, scratch, source/packed sizes, process memory and I/O. A normal CF run with
zero CF calls exits 5. `--version` exits inside the reused argument parser before
coverage reporting and proves initialization only.

The existing probe supplies bounded generation, explicit `-c` context limits,
signal cancellation and request output. It keeps the loaded model and worker
pool for the run. The integration requests use `keep: 0` and clear KV. Existing
per-item `keep` support can instead retain an exact token prefix after checking
token equality and retained positions; it fails if the required SWA state is
unavailable, with `--swa-full` needed beyond that retained window. Model,
tokenizer and runtime configuration are fixed within the process. This inherited
reuse path has not yet been tested with CF, and no reused-context speed gain or
new application session API is claimed.

## Verify and interpret results

```powershell
python -m pytest cf-engine/test_compiler.py -q -m "integration or not integration"
docker run --rm -v REPO_ROOT:/work -e CF_NATIVE_BINARY=/work/cf-engine/build-repro/bin/cf-run pi-gemma4-build:latest python3 /work/cf-engine/test_native_artifact.py -v
```

Recorded compiler validation: **46 passing tests**, including four actual 26B
single-expert roundtrips. Recorded native loader validation: **35 passing tests**
using a 128-byte synthetic source and 72-byte payload, without model loading.
The corrected Pi kernel suite also passes 4,704 numerical cases, 520 guard-page
cases and 312 assembly ABI cases. `analyze.py` compares saved CF reports and
probe requests; `analyze_sweep.py` joins explicitly alternating request records
and checks paired full logits and ordered routing captures. Inspect output
tokens, logits, routing and execution coverage together; matching text alone is
insufficient. Numerical agreement on these cases is not a general bit-identity
claim for all kernels, shapes, devices or sampling policies.

This iteration implements Milestone A and part of B. It does not implement a
fused complete MoE block, full-model CF coverage, verified multi-token decoding
or selective corrections. Shared activation quantization and expert-grouped
prefill already exist in the reference, so their presence is not a new CF result.
Only resident memory policy is enabled: this Pi's memory cgroup controller is
disabled, and no valid 8 GiB application-budget experiment has been run. Logical
weight accesses are modeled quantities, not measured DRAM traffic or energy.

The corrected 10-task suite covers short explanation, extraction, numbers,
negation, code, a longer synthetic inventory and an isolated topic-change cue.
Its German/Hebrew answers are numbers or a place name; extended multilingual
generation and multi-turn reuse remain untested. Neither these exact matches nor
the earlier diagnostic establishes visible-answer speed or the end-to-end
improvement gate. Performance requires repeated, alternated, matched experiments
including startup, source verification, preparation and fallback costs.
