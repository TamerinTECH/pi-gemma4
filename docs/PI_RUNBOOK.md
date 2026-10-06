# Raspberry Pi 5 runbook (pi/00 to pi/06)

This is the ordered procedure for running Gemma 4 on a Raspberry Pi 5 once the board is known. For
each `pi/` script it says what the script does, what it refuses, what it needs and what it writes. It
also covers how to fill `pi/pi_config.env` and how measured values replace the assumed ones.

Status: written 2026-10-05 against the scripts in `pi/` (all at version 1.0.0) and llama.cpp a7b94df
(build b11398). The real target board is **`context-friend-1`**: a Pi 5 with 16 GB RAM and a microSD
card, which also runs production services. Its rules are in [section 2.3](#23-safety-rules-on-context-friend-1).

Every number is labelled the same way as in [HYPOTHESES.md](HYPOTHESES.md):

- **MEASURED**: read off a run on real hardware; the result file is named.
- **PROJECTED**: computed by `pigemma.feasibility` or `pigemma.cache_sim` from inputs.
- **ASSUMED**: a catalog constant ([pigemma/data/catalog.json](../pigemma/data/catalog.json)) that has
  not been measured on that hardware yet.

Related documents:

- [HYPOTHESES.md](HYPOTHESES.md): the arithmetic behind every verdict (sections 3.1 to 3.7).
- [EXPERIMENT_PLAN.md, E8](EXPERIMENT_PLAN.md#e8-pi-runtime-dry-run): the pre-registered Pi
  expectations. [E6](EXPERIMENT_PLAN.md#e6-hot-set-pinning-proxy) is the pinning threshold.
- [results/pi-real/README.md](../results/pi-real/README.md): everything measured on
  `context-friend-1` so far.
- [README.md](../README.md): the project front page.

## Contents

1. [The short version](#1-the-short-version)
2. [Rules every script enforces](#2-rules-every-script-enforces)
3. [Hardware options](#3-hardware-options)
4. [Prerequisites](#4-prerequisites)
5. [Putting the repository on the Pi](#5-putting-the-repository-on-the-pi)
6. [Filling pi/pi_config.env](#6-filling-pipi_configenv)
7. [Steps 00 to 06](#7-steps-00-to-06): [00](#step-00-feasibility-verdict-read-only), [01](#step-01-os-tuning-print-first),
   [02](#step-02-build-llamacpp-and-the-tools), [03](#step-03-measure-storage-and-memory),
   [04](#step-04-fetch-the-model), [05](#step-05-run-bench-stream-bench-chat-server),
   [06](#step-06-pin-the-hot-set-and-ab-test)
8. [llama.cpp flags per verdict, and why repack is refused](#8-llamacpp-flags-per-verdict-and-why-repack-is-refused)
9. [Desktop artefacts to the Pi, Pi results to the desktop](#9-desktop-artefacts-to-the-pi-pi-results-to-the-desktop)
10. [How measured values replace assumed ones](#10-how-measured-values-replace-assumed-ones)
11. [Smoke tests](#11-smoke-tests)
12. [Troubleshooting](#12-troubleshooting)
13. [Status on the Pi: measured, projected, assumed](#13-status-on-the-pi-measured-projected-assumed)
14. [Usage lines (reference card)](#14-usage-lines-reference-card)
15. [Deviations from the manifest spec](#15-deviations-from-the-manifest-spec)

## 1. The short version

On a Pi you own, run these from the repository checkout. Every script re-checks the host and the
verdict before doing anything heavy.

```bash
bash pi/00_feasibility.sh                    # read-only verdict: exit 0 or 2 = go on, 3 = stop, 4 = refused
bash pi/01_tune_os.sh                        # dry run: read the plan
bash pi/01_tune_os.sh --apply                # apply it (root or passwordless sudo); log in again for memlock
nohup bash pi/02_build_llama.sh --with-tools > pi/out/build.out 2>&1 &
bash pi/03_bench_io_mem.sh --quick           # measured storage + memory -> results/pi/io_mem.json
bash pi/04_fetch_model.sh --dry-run          # what would be fetched, how big, how long
bash pi/04_fetch_model.sh --bg               # the download itself, detached
bash pi/05_run.sh bench --variants norepack
bash pi/05_run.sh stream-bench
bash pi/05_run.sh chat -p 'What is the capital of France? Answer with one word.'
bash pi/06_pin_hotset.sh ab --budget-bytes 5.2e9 --rounds 2
bash pi/00_feasibility.sh                    # again: io_mem.json and calibration.json are now merged
```

On `context-friend-1` (protected: production services, microSD only, no mlock), the sequence is
shorter. Section 2.3 explains why:

```bash
export PIGEMMA_I_KNOW=1                      # the operator authorised testing on this box (2026-10-05)
export MODEL_DIR=~/pigemma/models LLAMA_DIR=~/pigemma/llama.cpp
bash pi/00_feasibility.sh --want 26b
bash pi/01_tune_os.sh                        # dry run only; --apply needs the operator's go-ahead
nohup bash pi/02_build_llama.sh --with-tools > pi/out/build.out 2>&1 &
bash pi/03_bench_io_mem.sh --quick           # fio is missing there: sequential read via dd only
bash pi/04_fetch_model.sh --model 26b --quant UD-Q3_K_M --dry-run   # already there: FETCH_STATE=complete
bash pi/05_run.sh bench --variants norepack
bash pi/05_run.sh stream-bench
bash pi/05_run.sh chat -p 'מהי בירת צרפת? ענה במילה אחת.'
bash pi/00_feasibility.sh --want 26b
```

`pi/06_pin_hotset.sh` refuses on `context-friend-1` (exit 4), and that refusal is intended. See step 06.

## 2. Rules every script enforces

### 2.1 Exit codes (one convention for all pi scripts and `pigemma.feasibility`)

| Code | Meaning | What to do |
|---|---|---|
| 0 | ok: the step ran, or for pi/00 a model fits in RAM (FITS_REPACKED, FITS_MAPPED) or streams well (STREAM_OK) | go on |
| 1 | usage error, internal error, or a failed step (git, configure, build, download, verification, a failed run whose results file is still written with a `failed:` note) | read the last stderr lines, fix, re-run; every step is idempotent |
| 2 | only slow operation is possible (STREAM_SLOW, or a dense model that fits in RAM but stays below 1 tok/s). pi/06 start: partial pin | pi/05 accepts it; pi/04 warns and fetches (`--strict` turns it into exit 2) |
| 3 | not viable, or a precondition failed: RAM, disk, missing tool, missing model, binary or input | the message names the missing piece and the fix (apt hint, a smaller quant, more storage) |
| 4 | refused: protected host, busy host, bench lock held, port in use, mlock not allowed | do not work around it; see 2.2 |

`pi/05_run.sh chat` and `server` return the llama.cpp tool's own exit code. `pi/06_pin_hotset.sh status`
returns 3 when no pinner is running.

### 2.2 Refusal rules

| Rule | Checked by | Result | Override |
|---|---|---|---|
| **Deny-listed hostname.** `DENY_HOSTS` in `pi/pi_config.env` is empty by default since 2026-10-05. The catalog's `protected_hosts` still lists `context-friend-1`, so that hostname is refused | `assert_safe_host` in [pi/lib/common.sh](../pi/lib/common.sh) (pi/01 to pi/06) and `pigemma.feasibility` (pi/00, pi/04, pi/05) | exit 4 | `PIGEMMA_I_KNOW=1`, only with the host owner's explicit authorisation. Results then carry a note |
| **Protected services.** A running systemd unit matches `PROTECTED_SERVICE_GLOBS` (`contextfriend* context-friend*`) or the catalog's `protected_service_globs` (same globs). Matching is case-insensitive, with or without `.service`. Without systemd (a container), process names are scanned instead | same | exit 4 | same |
| **Busy host.** loadavg1 > 2.0 × max(1, nproc/4), which is 2.0 on a Pi 5 (`PIGEMMA_MAX_LOADAVG` overrides), or effective MemAvailable < 1 GB | `assert_not_busy` (pi/03, pi/05) and `pigemma.feasibility` (pi/00, pi/04, pi/05) | exit 4 | none: `PIGEMMA_I_KNOW=1` never lifts it. Wait until the box is idle |
| **Bench lock held.** `results/.bench.lock` belongs to another model-loading job. Only one model job runs at a time | pi/02, pi/03, pi/05, pi/06, `pigemma.bench`, `pigemma.quality` | exit 4 | wait; `python3 -m pigemma.paths lock --status` names the holder. A stale lock (dead pid) is replaced automatically. pi/02 has `--ignore-lock` |
| **Not enough RAM.** pi/05: effective MemAvailable ≥ file bytes + 0.5 GB when the model fits in RAM, or ≥ non-expert bytes + 0.5 GB when it streams, plus the draft's bytes. pi/02: ≥ `--min-mem-gb` (1.5). pi/06: hot set + KV + compute + headroom. membw: buffer + 256 MiB | each script before loading | exit 3 | free memory, use a smaller quant or a smaller `-c` |
| **Not enough disk.** `require_free_disk`: free ≥ max(`MIN_FREE_DISK_FACTOR` (1.1) × bytes, bytes + `KEEP_FREE_GB` (2) GB when the path is on `/`). pi/04 applies it to the download, pi/02 to its 3 GB work budget, pi/03 to its test file (1.2 × size) | pi/02, pi/03, pi/04 | exit 3. In auto mode pi/04 first falls back to the best model that fits | more storage, or delete an old model by hand (the scripts never delete) |
| **mlock not allowed.** `ALLOW_MLOCK=auto` forbids mlock of model weights on every protected host, even with `PIGEMMA_I_KNOW=1`: the page cache must stay reclaimable for the services | pi/06 and `pi/05 --pin` refuse. `safe_llama_flags` turns `-lm mmap+mlock` into `-lm mmap` in every llama command | exit 4, or flags changed with a note | `ALLOW_MLOCK=1`, only with the owner's approval. pi/06 also needs `--allow-protected-host` |
| **AI HAT on the PCIe lane.** A Hailo or AI HAT shows up in `/proc/device-tree/hat/product`, in `lspci`, as PCI vendor 0x1e60 or as `/dev/hailo*` | pi/01 item (a) is skipped with `PCIe lane used by the AI HAT: NVMe impossible without a dual-slot HAT`. pi/00 and `pigemma.feasibility` print a note | no exit code change; the NVMe Gen3 item is skipped | remove the HAT or use a dual-slot HAT. The HAT does not help 25 to 30B models anyway |
| **Container, non-Linux or fake root** | `pi/01 --apply` | exit 3 | none needed: the dry run still prints the plan |
| **Split GGUF** (parts `-00001-of-0000N.gguf`) | pi/06 (the map, the hot set and the pinner address byte offsets of one file) | exit 3 | use a single-file GGUF |

These rules were checked on the desktop with the fake system root of `context-friend-1`
(`PIGEMMA_FAKE_PROC=tests/fixtures/pi_proc_cf1_16gb`). pi/00 exits 4 and names both reasons (the
deny-listed hostname and the two `contextfriend-*` services). With `PIGEMMA_I_KNOW=1` it exits 0 and
prints `PROTECTED HOST 'context-friend-1' - OVERRIDE ACTIVE (PIGEMMA_I_KNOW=1)`.

### 2.3 Safety rules on context-friend-1

These rules hold even with `PIGEMMA_I_KNOW=1`. They come from the operator's authorisation of 2026-10-05
([docs/design/FACTS.md](design/FACTS.md)), and the scripts enforce the parts they can.

- **Services.** Never stop, restart or kill `contextfriend-box`, `contextfriend-gpu-tunnel`, `hailort`
  or `whisper-bench@base`. No script does; pi/06 `stop` signals only the pid in `pi/out/pinner.pid`,
  and only after checking that it is a pinner.
- **Priority.** Builds, downloads and inference run at `nice -n 10` (`NICE_LEVEL`). Downloads and fio
  also run at `ionice -c3`.
- **No mlock of model weights** (`ALLOW_MLOCK=auto`). pi/06 and `pi/05 --pin` refuse, and `-lm mmap+mlock`
  becomes `-lm mmap`.
- **No anonymous repack copy.** `-nr` is forced for chat, server and stream-bench, and the bench `default`
  (repack) variant is refused unless `--allow-repack` is given. An anonymous copy is no more
  reclaimable for the services than mlocked pages are.
- **Free space.** Keep ≥ 2 GB free on `/` at all times (`KEEP_FREE_GB=2`). Models are swapped one at a
  time, and you delete the old one by hand.
- **No boot changes.** No edit of `/boot/firmware/config.txt` and no reboot without the operator's explicit
  approval. On a protected host pi/01 is runtime-only and needs `--allow-boot-edit` for item (a).
- **Network.** Reach the box as `ssh -o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1 pi@PI_HOST`.
  The name resolves to two addresses, one of them stale. The Wi-Fi drops packets under load, so run
  anything longer than a minute with `nohup` (or `--bg`) and poll its log. A dropped SSH session does
  not mean the job died.

## 3. Hardware options

The question is which Pi 5 configuration runs which Gemma 4 model, and how fast. 26B means the
26B-A4B MoE (QAT Q4_0, 14.44 GB). 31B means the dense model (QAT Q4_0, 17.65 GB). The tok/s values are
**PROJECTED** by `pigemma.feasibility` with **ASSUMED** inputs: 10 GB/s memory bandwidth × 0.65 llama
efficiency, the catalog storage speeds below, and the prior LRU hit curve. They are the same grid as
[HYPOTHESES.md](HYPOTHESES.md) section 3.6. The dense column is the whole file divided by the
sequential read speed, because the file does not fit in RAM. **MEASURED** values come from
`context-friend-1` ([results/pi-real/README.md](../results/pi-real/README.md)).

| Pi 5 RAM | Model storage (catalog: sequential / random 3 MB read, ASSUMED) | 26B-A4B Q4_0: tok/s and verdict | 31B dense Q4_0 | Verdict |
|---|---|---|---|---|
| 4 GB | microSD (85 / 60 MB/s) | 0.11 NOT_VIABLE | 207.7 s/token | not viable for either model |
| 4 GB | USB 3.0 SSD (380 MB/s), NVMe Gen2 (430 MB/s), NVMe Gen3 (850 MB/s) | 0.57, 0.65, 0.98 STREAM_SLOW | 46.5, 41.0, 20.8 s/token | not viable: about 1.3 GB of expert cache is too small |
| 8 GB | microSD | 0.44 STREAM_SLOW | 207.7 s/token | not viable. MEASURED proxy: the 12.73 GB UD-Q3_K_M with 7.5 GB usable RAM (an 8 GiB balloon on `context-friend-1`) ran at 0.55–0.79 tok/s, storage-bound |
| 8 GB | USB 3.0 SSD (380 / 350 MB/s) | 1.46 STREAM_OK | 46.5 s/token | the MoE streams experts, slower |
| 8 GB | NVMe Gen2 x1 (430 / 400 MB/s, the default link) | 1.57 STREAM_OK | 41.0 s/token | the MoE streams experts |
| 8 GB | NVMe Gen3 x1 (850 / 700 MB/s; `dtparam=pciex1_gen=3`, uncertified) | 1.94 STREAM_OK | 20.8 s/token | **the MoE streams experts with about 40% of them cached.** This is where H1 matters |
| 16 GB | microSD (85 / 60 MB/s; 91.6 MB/s MEASURED on `context-friend-1`) | 2.73 FITS_MAPPED. MEASURED with the 3-bit file: 3.17–3.27 tok/s | 3-bit file in RAM: 0.53 tok/s MEASURED | works today. The Q4_0 26B file does not fit a 30 GB card with 2 GB kept free, so 3-bit quants are used |
| 16 GB | USB 3.0 SSD | 2.73 FITS_MAPPED | 46.5 s/token (Q4_0 does not fit in RAM) | good; the SSD only shortens the load (about 38 s) |
| 16 GB | NVMe Gen3 x1 + Active Cooler | 2.73 FITS_MAPPED | 20.8 s/token for Q4_0; a 3-bit 31B in RAM runs at about 0.5 tok/s | **recommended**: the 26B-A4B fits in RAM with `-nr` and loads in about 17 s |
| any | + AI HAT (Hailo-8 or 8L; AI HAT+ 2 with Hailo-10H) | no change | no change | **does not help.** Hailo-8/8L have no DRAM and no LLM support. Hailo-10H handles at most about 2B models, decodes more slowly than the Pi CPU, and has no llama.cpp backend. The HAT also takes the only PCIe lane, so an NVMe then needs a dual-slot HAT |
| 2 or more Pis | llama.cpp RPC | — | — | slower than one 16 GB Pi in published measurements |

Notes on the table:

- **The dense 31B never reaches good speed on any Pi 5.** It reads every weight for every token: 17.6 GB
  per token at Q4_0, and about 11.8 GB at 3 bits.
- **The 16 GB rows are RAM-bound.** Storage only changes the load time: 14.44 GB at 850 MB/s is about
  17 s, at 380 MB/s about 38 s, and at 85 MB/s about 170 s, all PROJECTED. MEASURED: 148 s for the
  12.73 GB file from the microSD.
- **The measured 26B numbers beat the projection.** `26b-ud-q3_k_m` is projected at 2.14 tok/s and
  measured at 3.17–3.27 tok/s. The ASSUMED 0.65 efficiency is too low, or the catalog's bytes per token
  overcount what llama.cpp reads. Steps 03 and 05 measure which (section 10).
- **Price.** The Pi 5 16 GB cost $120 at launch and about $305 by 2026-04, after memory prices rose.

## 4. Prerequisites

| Item | Why | How | context-friend-1 (probed 2026-10-05) |
|---|---|---|---|
| Debian 13 / Raspberry Pi OS, aarch64 | the scripts and the native build | stock image; the 16 KiB-page kernel is fine (the scripts read the page size at runtime) | Debian 13, kernel 6.18.50+rpt-rpi-2712, 16 KiB pages |
| Packages | build, measure, download | `sudo apt-get install -y build-essential cmake git fio python3` (plus `curl`, and `rsync` for `--from-desktop`). The scripts never install anything; a missing tool exits 3 with this hint | gcc 14.2, cmake 3.31.6, git, Python 3.13, curl present; **fio missing** (installing it needs the operator's approval; pi/03 falls back to dd) |
| Python ≥ 3.10 | `pigemma.feasibility`, `pigemma.bench`, `pigemma.gguf_map` run as `python3 -m ...` from the checkout | stdlib only on the Pi; numpy and psutil are optional | Python 3.13 + numpy 2.2.4 |
| Active Cooler | decode keeps 4 cores busy for minutes; the Pi 5 throttles from 80 °C | fit the official Active Cooler | fitted: 72.5 °C peak MEASURED over 23 min of the 31B run, `throttled=0x0` |
| 27 W power supply | under-voltage slows the CPU and can corrupt the SD card | official 5 V / 5 A supply | — |
| NVMe HAT (or a USB 3.0 SSD on a blue port) | fast model storage; 16 GB + NVMe Gen3 is the recommendation | fit the HAT, then `TUNE_NVME_GEN3=1 bash pi/01_tune_os.sh --apply` for Gen3 and reboot **by hand**. Point `MODEL_DIR` at the mount | none: microSD only; the PCIe lane is free |
| `ulimit -l unlimited` | `-lm mmap+mlock` (FITS_MAPPED flags) and the pinner (pi/06) lock pages | `bash pi/01_tune_os.sh --apply --only e`, then **log out and in again** (a new SSH session) and check that `ulimit -l` prints `unlimited`. systemd services pick it up after a reboot | 8192 kB, deliberately left alone: no mlock on this host |
| Disk | GGUF plus 10% (plus 2 GB kept free on `/`); about 3 GB for the build | pi/00's summary line `disk (pi/04)` says whether the recommended file fits | 15,501,496,320 B free on the 30 GB card |
| Network | 12.7 to 14.4 GB per model | Hugging Face over Wi-Fi: about 7 to 8 MB/s MEASURED (1692 s for 11.84 GB) | Wi-Fi only; drops under load |

## 5. Putting the repository on the Pi

The repository is not on a git remote yet, so copy it. Git Bash on the desktop has `ssh`, `scp` and
`tar`, but no `rsync`. A tar pipe over ssh is the simplest route. It leaves out the models, the
llama.cpp sources (pi/02 fetches its own copy, about 40 MB), the traces and all build directories. On
2026-10-05 the stream was about 4.9 MB (MEASURED).

```bash
cd REPO_ROOT
PI="-o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1"
tar -cf - --exclude=./models --exclude=./third_party --exclude=./traces --exclude=./results \
    --exclude=./tools/build --exclude=./tools/build-linux-x86 --exclude=./pi/out --exclude=./.git \
    --exclude=__pycache__ --exclude=./.pytest_cache . \
  | ssh $PI pi@PI_HOST 'mkdir -p ~/pi-gemma4 && tar -xf - -C ~/pi-gemma4'
```

From a Linux machine with rsync on both ends, the same copy is
`rsync -a --exclude models --exclude third_party --exclude traces --exclude results --exclude 'tools/build*' --exclude pi/out --exclude .git ./ pi@<pi>:pi-gemma4/`.

The desktop artefacts (map, routing statistics, simulation) follow separately (section 9). Re-running
the copy overwrites the scripts, including `pi/pi_config.env`. Keep per-host settings outside the
repository (section 6).

## 6. Filling pi/pi_config.env

[pi/pi_config.env](../pi/pi_config.env) is the single parameter file. Every script sources it through
`pi/lib/common.sh` and prints the effective values on stderr. Precedence:

1. **The environment wins.** `THREADS=3 bash pi/05_run.sh bench` works without editing anything.
2. **Explicitly empty values.** Variables written as `${VAR-default}` (`DENY_HOSTS`,
   `PROTECTED_SERVICE_GLOBS`) keep an explicitly empty value, for example `DENY_HOSTS=`.
3. **Replacing the file.** `PIGEMMA_CONFIG=/path/other.env` sources another file instead. Missing
   variables then take the built-in defaults.

Keep host settings in a small file outside the checkout, for example `~/.pigemma.env`, and source it
in every session. Re-copying the repository then never loses them:

```bash
# ~/.pigemma.env on context-friend-1 (source it: . ~/.pigemma.env)
export PIGEMMA_I_KNOW=1                     # authorised by the operator, 2026-10-05; results record it
export MODEL_DIR=~/pigemma/models           # where the 26B UD-Q3_K_M already is
export LLAMA_DIR=~/pigemma/llama.cpp        # the native build from 2026-10-05 (410 s, -j 4)
export THREADS=4 CTX=4096 NICE_LEVEL=10 KEEP_FREE_GB=2 ALLOW_MLOCK=auto
```

| Variable | Default | Set it when | context-friend-1 |
|---|---|---|---|
| `MODEL_DIR` | `$PIGEMMA_HOME/models` | the models live on an NVMe or SSD mount: storage type, link speed and free disk are measured on this path | `~/pigemma/models` |
| `LLAMA_DIR` | `~/llama.cpp` | an existing build lives elsewhere (binaries in `$LLAMA_DIR/build/bin`; `PIGEMMA_LLAMA_BIN` overrides) | `~/pigemma/llama.cpp` |
| `LLAMA_COMMIT` | `a7b94df` | never, unless all desktop numbers are re-measured at the new commit | default |
| `PIGEMMA_OUT` | `pi/out` | several boards share one checkout | default |
| `THREADS` | `4` | rarely: the Pi 5 has 4 cores. bench measures 2, 3 and 4 | 4 |
| `CTX` | `4096` | a longer context costs KV RAM; feasibility includes it | 4096 |
| `QUANT_OVERRIDE` | empty | you want a specific quant instead of the recommendation (pi/04, pi/05) | empty: pass `--quant` instead |
| `OS_RESERVE_GIB` | empty (tier rule: 0.7 / 0.8 / 0.9 GB) | the board runs other services; set it to their resident size plus a margin | empty; the services use about 0.4 GB RSS |
| `MIN_FREE_DISK_FACTOR` | `1.1` | rarely | default |
| `KEEP_FREE_GB` | `2` | never lower it on a shared box | 2 |
| `NICE_LEVEL` | `10` | the board is dedicated (0 is a little faster) | 10 |
| `DENY_HOSTS` | empty | another board must never run heavy work | empty; the catalog lists `context-friend-1` |
| `PROTECTED_SERVICE_GLOBS` | `contextfriend* context-friend*` | the board runs other production services | default |
| `PIGEMMA_I_KNOW` | `0` | the owner authorised work on a protected host | 1 |
| `ALLOW_MLOCK` | `auto` | `1` only on a dedicated board, or with the owner's approval | auto, so no mlock |
| `TUNE_APPLY`, `TUNE_NVME_GEN3`, `TUNE_GOVERNOR`, `TUNE_SWAPPINESS`, `TUNE_READAHEAD_KB`, `TUNE_MEMLOCK` | `0`, `0`, `performance`, `10`, `2048`, `1` | step 01 (Gen3 only with an NVMe and no AI HAT) | dry run only |
| `BUILD_JOBS` | `auto` = min(nproc, floor(MemAvailable_GiB / 1.2), 4) | a build runs out of memory: use a lower number | auto (4) |
| `HF_LIMIT_RATE` | empty | the download must leave bandwidth for other traffic, e.g. `5M` | consider `5M` while the services are busy |
| `PIN_BUDGET_BYTES` | `auto` = MemAvailable − KV − 0.3e9 − 0.5e9 | you want a fixed hot-set size (step 06) | not used |

Environment-only knobs: `PIGEMMA_DEVICE` names the catalog device profile for facts and io_mem.
`PIGEMMA_EMULATED=1` labels runs as emulated and writes to `results/pi-emu/`. `PIGEMMA_MAX_LOADAVG`
changes the busy threshold. `PIGEMMA_PYTHON` picks the interpreter. `PIGEMMA_LOG_FILE` appends log
lines to a file, and `PIGEMMA_QUIET=1` hides info logs.

## 7. Steps 00 to 06

Each step lists the command, what the script does, what it needs, what it writes, its exit codes and
the output to expect. Run the steps in order: each one re-checks the host and the verdict, so a
skipped step shows up as a refusal or a missing input, never as silent damage.

### Step 00: feasibility verdict (read-only)

```bash
bash pi/00_feasibility.sh
bash pi/00_feasibility.sh --want 26b --json pi/out/feasibility.json
bash pi/00_feasibility.sh --sim results/h1/sim.json      # with the desktop cache simulation (section 9)
```

- **What it does.** Collects the host facts read-only (`/proc`, `/sys`, `/boot`, command output). Runs
  `python3 -m pigemma.feasibility` and prints a report: host facts, a banner for protected or busy
  hosts, one verdict row per model and quant with projected tok/s, and a summary. The summary gives the
  recommended model, its flags as used on this host after the mlock rule, the pi/04 disk check, the
  next steps and the fixed hardware line. pi/00 changes nothing, so it runs on a protected host without
  the override (so do `pi/06_pin_hotset.sh stop` and `status`, which lock nothing).
- **Needs.** bash and Python ≥ 3.10. Nothing else.
- **Writes.** `pi/out/facts.json`, `pi/out/feasibility.json` (moved into place only when a verdict
  exists) and `pi/out/feasibility.txt` (a copy of the report).
- **Exit codes.** 0 = FITS_* or STREAM_OK. 2 = only slow operation. 3 = nothing viable, or no
  Python ≥ 3.10. 4 = refused (protected host without `PIGEMMA_I_KNOW=1`, or busy). 1 = usage or input
  error. If the bash gate refuses but `pigemma.feasibility` did not, the code is raised to 4.

Expected output on `context-friend-1` with `PIGEMMA_I_KNOW=1`. This is an abridged real run on the
desktop against the fixture `tests/fixtures/pi_proc_cf1_16gb` with `--io-mem none --calibration none`,
2026-10-05; paths shortened to `pi/out`:

```text
==== summary ====
  verdict         exit 0 = OK - a model fits in RAM (FITS_*) or streams well (STREAM_OK)
  recommended     26b-q4 (Q4_0, 14.44 GB) FITS_MAPPED, projected 2.73 tok/s [assumed inputs]
  llama flags     -nr -lm mmap+mlock -t 4 -c 4096 -fa on -ctk q8_0 -ctv q8_0
  on this host    -nr -lm mmap -t 4 -c 4096 -fa on -ctk q8_0 -ctv q8_0   (mlock removed: ALLOW_MLOCK=auto on a protected host: ...)
  disk (pi/04)    NOT ENOUGH: 26b-q4 needs max(1.1 x 14.44 GB, 14.44 GB + keep-free 2.00 GB) = 16.44 GB free on /; free 15.50 GB -> pi/04_fetch_model.sh will refuse this file
                  best row that also fits the disk: 26b-ud-q3_k_m (UD-Q3_K_M, 12.73 GB) FITS_REPACKED, projected 2.14 tok/s -> pi/04_fetch_model.sh --model 26b --quant UD-Q3_K_M
  projections     all tok/s are PROJECTED. hit rates: assumed curve (catalog.selection)
                  mem_bw_gbps 10 GB/s [assumed], llama_efficiency 0.65 [assumed], seq_read_mbps 85 MB/s [assumed], ...
```

Without the override the same run exits 4. The summary then reads
`verdict exit 4 = REFUSED - protected or busy host; the rows are informational only`, and a banner
says `EVERY OTHER pi/*.sh SCRIPT REFUSES ON THIS HOST (exit 4)`.

The rows that matter on `context-friend-1` (PROJECTED, same fixture run):

| Row | Verdict | Projected tok/s | pi/04 disk rule (15.50 GB free on `/`) |
|---|---|---|---|
| `26b-q4` (QAT Q4_0, 14.44 GB) | FITS_MAPPED | 2.73 | needs 16.44 GB: refused, pi/04 falls back |
| `26b-ud-iq4_xs` (13.60 GB) | FITS_REPACKED | 1.99 | needs 15.60 GB: refused |
| `26b-ud-q3_k_m` (12.73 GB) | FITS_REPACKED | 2.14 | needs 14.73 GB: fits. **This is the file used on this board** |
| `31b-ud-iq3_xxs` (11.84 GB) | NEVER_GOOD_SPEED (runnable) | 0.55 | needs 13.84 GB: fits (only one model at a time) |
| `31b-q4` (17.65 GB) | NEVER_GOOD_SPEED | 0.005 | does not fit in RAM either |

**What to do next.**

| pi/00 exit | Next step |
|---|---|
| 0 | step 01 |
| 2 | go on, knowing the result is a measurement and not a usable speed. Only `--want 31b`, or boards with ≤ 8 GB on slow storage, end up here |
| 3 | change the hardware (section 3) or the storage. Nothing else will run |
| 4 | stop. On `context-friend-1`, continue only with the authorisation in section 2.3 |

### Step 01: OS tuning (print-first)

```bash
bash pi/01_tune_os.sh                         # dry run: current vs proposed value per item, the exact commands
bash pi/01_tune_os.sh --apply                 # execute; one 'sudo -n' per privileged command
TUNE_NVME_GEN3=1 bash pi/01_tune_os.sh --apply    # also the PCIe Gen3 line (NVMe present, no AI HAT); reboot by hand
bash pi/01_tune_os.sh --revert --apply        # undo an earlier --apply
```

- **What it does.** The default is a dry run. For each item it prints the current value, the proposed
  value, the status (CHANGE, OK, SKIP, NOT SELECTED or ADVICE) and the commands.
  - (a) PCIe Gen3 x1: `dtparam=pciex1_gen=3`. Only with `TUNE_NVME_GEN3=1`, an NVMe present and no
    AI HAT on the lane.
  - (b) cpufreq governor `performance`.
  - (c) `vm.swappiness=10` and `vm.vfs_cache_pressure=50`.
  - (d) `read_ahead_kb=2048` on the disk that holds `MODEL_DIR`.
  - (e) memlock unlimited (limits.d plus systemd `DefaultLimitMEMLOCK`; log in again afterwards).
  - (f) advice only: swap and zram, cooling, power, storage.

  `--only` and `--skip` select items (a to f). It never reboots, never restarts a service and never
  installs anything.
- **Needs.** Root or passwordless `sudo -n` for `--apply`.
- **Writes (only with `--apply`).** The marked block in the boot config, plus a
  `.pigemma-<UTC>.bak` copy. Also `/etc/systemd/system/pigemma-governor.service`,
  `/etc/sysctl.d/90-pigemma.conf`, `/etc/udev/rules.d/90-pigemma-readahead.rules`,
  `/etc/security/limits.d/90-pigemma.conf`, `/etc/systemd/system.conf.d/pigemma.conf` and
  `pi/out/tune_os.state` (the original values, used by `--revert`). Every `/etc` file carries the
  marker `Managed by pi-gemma4 pi/01_tune_os.sh`.
- **Exit codes.** 0 = plan printed, applied or nothing to do. 1 = usage error or a failed step
  (completed items stay applied). 3 = `--apply` precondition failed: not root and no passwordless sudo,
  inside a container, non-Linux, or a fake root. 4 = protected host without the override.
- **On a protected host with `PIGEMMA_I_KNOW=1`.** The script is runtime-only (no persistence files
  unless `--persist`). Item (a) also needs `--allow-boot-edit`. Item (e) is skipped while mlock is not
  allowed. On `context-friend-1`, record the dry run and apply nothing without the operator's go-ahead: the
  2026-10-05 measurements were taken with the stock settings.

Expected dry-run output on the `context-friend-1` fixture with the override (abridged):

```text
(a) ... status  : SKIP - no NVMe present (model storage: microsd): nothing to tune. Recommended upgrade: NVMe HAT + PCIe Gen3 x1 ...
(b) ... status  : CHANGE        (ondemand -> performance, live only)
(c) ... status  : CHANGE        (vm.swappiness=10, vm.vfs_cache_pressure=50, live only)
(d) ... status  : CHANGE        (read_ahead_kb=2048 on mmcblk0, live only; mmcblk0 also holds the root filesystem)
(e) ... status  : SKIP - mlock of model weights is not allowed on this host (ALLOW_MLOCK=auto on a protected host, deviation D4) ...
(f) ... status  : ADVICE
summary : 3 to change, 0 already OK, 2 skipped / not selected (tune plan)
```

### Step 02: build llama.cpp and the tools

```bash
nohup bash pi/02_build_llama.sh --with-tools > pi/out/build.out 2>&1 &
tail -n 5 pi/out/build.out                    # poll; progress every 30 s
bash pi/02_build_llama.sh --dry-run           # the checks and commands only
```

- **What it does.** Gets llama.cpp at `LLAMA_COMMIT` (a7b94df) into `LLAMA_DIR`. It uses an existing
  checkout as is, otherwise a depth-1 fetch of the full hash (about 40 MB), with `git clone
  --filter=blob:none` as the fallback. `--from-desktop SRC` copies a tree instead.
- **Configure and build.** Configures `-DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF`,
  falling back once to `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16` (the Pi 5 ISA:
  dotprod and fp16, no i8mm or SVE). Builds llama-cli, llama-completion, llama-bench, llama-perplexity
  and llama-server at `nice -n 10` with `pick_jobs` jobs. `--with-tools` also builds `pigemma-probe`,
  `pinner` and `membw` into `tools/build/bin` from the same CMake tree.
- **Skips work.** Up-to-date binaries are not rebuilt (`--rebuild` forces it).
- **Needs.** cmake, gcc, g++, make (or ninja) and git. Free disk ≥ 3 GB (`--min-free-gb`), plus
  `KEEP_FREE_GB` on `/`. MemAvailable ≥ 1.5 GB.
- **Writes.** `$LLAMA_DIR/build/bin/*`, `tools/build/bin/*`, `pi/out/build.json` (host, commit, arch
  flags, targets, seconds, disk use; also written with status `failed`) and `pi/out/build-logs/`.
- **Stdout.** `== llama.cpp <commit> in <bin_dir> ==`, one path per target, then version, arch and
  build. Next comes `llama-bench --help | head -1`, then (with `--with-tools`) `== pigemma tools: ... ==`
  with the membw, pinner and pigemma-probe version lines, and finally `build.json <path>`.
- **Exit codes.** 0 = built, up to date, configured or dry run. 1 = usage error, or a failed step
  (git, rsync, configure, build, a binary that does not run). 3 = a tool, disk, RAM or source problem
  (foreign or modified `LLAMA_DIR`, commit missing in a shallow checkout, desktop tree at another
  commit). 4 = protected host, bench lock held, or another build running.
- **MEASURED on `context-friend-1`.** The 5 targets built in 410 s with `-j 4` at nice 10, and used
  355 MB after deleting the object files (`pi/adhoc/build_llama_native.sh`). pi/02 judges that build up
  to date and only compiles the tools, which builds llama and ggml a second time.
- **Never build the Pi binaries elsewhere with `GGML_NATIVE=ON`.** Under qemu, native detection adds
  i8mm, SVE and SME, which crash a real Pi 5 with SIGILL (MEASURED in `pi-gemma4-pi-emu:trixie`). pi/02
  uses the Pi 5 ISA under emulation.

### Step 03: measure storage and memory

```bash
bash pi/03_bench_io_mem.sh --quick            # 5 fio jobs x 5 s + membw 256 MiB x 3 reps (about a minute, estimate)
bash pi/03_bench_io_mem.sh                    # full: 5 fio jobs x 20 s + membw 512 MiB x 5 reps
bash pi/03_bench_io_mem.sh --dir /mnt/nvme/models    # measure the storage that holds the models
```

- **Storage.** fio with O_DIRECT and `--readonly` runs five jobs:
  - `seq_read_qd1` and `seq_read_qd8` (1 MiB blocks);
  - `rand_read_3mb_qd1` and `rand_read_3mb_qd4` (one expert slab: 3,345,408 B rounded up to the page,
    3,358,720 B on 16 KiB pages);
  - `rand_read_16k_qd1` (one Pi page fault: IOPS and the access latency).
- **Memory.** `membw` read, copy and triad at 1, 2 and 4 threads.
- **Test file.** `--file auto` reads the largest GGUF of at least 1 GiB in `MODEL_DIR`, read-only, so
  nothing is written to the SD card. Only when there is none is a test file written and deleted
  afterwards.
- **Needs.** fio (without it: a sequential read with `dd iflag=direct` only, flagged, `complete:false`;
  `--require-fio` turns this into exit 3). `tools/build/bin/membw` from step 02. Not busy, bench lock
  free.
- **Writes.** `results/pi/io_mem.json` (kind `measured`; `results/pi-emu/io_mem.json` when emulated) and
  the evidence in `results/pi/raw/io_mem-<UTC>/`. Unmeasured metrics are **absent**, so the catalog
  keeps its assumed value for them. After writing, the script checks that the file merges through
  `pigemma.feasibility.load_catalog`, and exits 1 if it does not.
- **Stdout.** The measured table, the catalog values before and after the merge, and derived numbers
  (PROJECTED in-RAM ceilings and the minimum expert-cache hit rate for 1 and 2 tok/s).
- **Exit codes.** 0 = written (possibly partial). 1 = usage, internal or merge-check error.
  3 = precondition failed or nothing measured. 4 = protected host, busy host, or bench lock held.
- **Device name.** On a Pi 5 the device is the catalog profile with the same RAM tier and storage class,
  otherwise `pi5-<tier>gb-<class>`, which feasibility synthesizes. `context-friend-1` becomes
  `pi5-16gb-microsd`. `--device` or `PIGEMMA_DEVICE` force a name.
- **On `context-friend-1`.** fio is missing, so only the sequential read is measured (the raw `dd`
  number already MEASURED there: 91.6 MB/s with O_DIRECT and 4 MiB blocks). The random 3 MB read and
  the access latency stay ASSUMED (60 MB/s, 2.0 ms) until the operator approves `sudo apt-get install -y fio`.

### Step 04: fetch the model

```bash
bash pi/04_fetch_model.sh --dry-run                    # selection, disk budget, ETA; no network, no writes
bash pi/04_fetch_model.sh --bg                         # download detached; log in pi/out/fetch.log
tail -n 3 pi/out/fetch.log                             # progress every 30 s
bash pi/04_fetch_model.sh --quant UD-Q3_K_M --draft    # explicit quant + the family's MTP sidecar
```

- **What it does.** In auto mode it collects fresh facts (with `MODEL_DIR` as the destination) and
  fetches `pigemma.feasibility`'s recommendation. With `--quant`, a catalog id or `QUANT_OVERRIDE`, it
  does a plain catalog lookup instead.
- **Disk rule.** need = max(1.1 × remaining bytes, remaining + 2 GB when the destination is on `/`). If
  the recommendation does not fit, auto mode re-runs the verdict with the disk budget and fetches the
  best model that fits (`--no-disk-fallback` exits 3 instead).
- **Download.**
  - Checks `X-Linked-Size` and `X-Linked-ETag` (sha256) against the catalog first.
  - Downloads into `<file>.part` with repeated, separate `curl -C -` calls. Each call resumes from the
    current size; curl's own `--retry` truncated a 4.3 GB partial on the real Pi.
  - Gives up after 30 attempts without progress.
  - Verifies the size and the sha256, then renames the file and writes a `<file>.sha256` stamp.
  - Runs at `nice -n 10 ionice -c3`.
- **Needs.** curl and sha256sum, the disk budget, not busy, one fetch at a time (`pi/out/.fetch.lock`).
  A token is optional (`HF_TOKEN`, or `--token-file`); the Gemma 4 GGUFs are ungated.
- **Writes.** `<MODEL_DIR>/<catalog file>` (the 31B MTP sidecar lands in `MTP/`), the `.sha256` stamp,
  `pi/out/fetch.json` and, with `--bg`, `pi/out/fetch.log`. It never deletes or overwrites a file it
  did not create.
- **Stdout.** Shell-safe `FETCH_*=value` lines, so `eval "$(bash pi/04_fetch_model.sh --dry-run)"` works.
- **Exit codes.** 0 = fetched, already present and verified, or a dry run. 1 = usage, network or
  verification error (a wrong-size file is never deleted: the message prints the `rm` to run).
  2 = slow only, with `--strict`. 3 = not viable, not enough disk, or a missing tool. 4 = protected or
  busy host, or another fetch running.

Expected `--dry-run` result on the `context-friend-1` fixture with the override (a real run, 2026-10-05,
abridged): the Q4_0 recommendation does not fit the card, so pi/04 falls back to the 3-bit file.

```text
WARNING: not enough free disk for 26b-q4: free 15.50 GB (...), need 16.44 GB = max(1.1 x 14.44 GB remaining, 14.44 GB + keep-free 2.00 GB)
WARNING: DISK FALLBACK: fetching 26b-ud-q3_k_m (UD-Q3_K_M, 12.73 GB, verdict FITS_REPACKED, projected 2.135 tok/s) instead of 26b-q4
FETCH_SELECTION=auto-disk-fallback
FETCH_MODEL_ID=26b-ud-q3_k_m
FETCH_URL=https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf
FETCH_NEED_BYTES=14728497888
FETCH_ETA_S=1273
```

To swap models on a full card, delete the old file and its stamp **by hand** first, for example
`rm ~/pigemma/models/<old>.gguf ~/pigemma/models/<old>.gguf.sha256`.

### Step 05: run (bench, stream-bench, chat, server)

```bash
bash pi/05_run.sh bench --variants norepack              # llama-bench -t 2,3,4 -p 64 -n 32 -r 3, results envelopes (E8)
bash pi/05_run.sh bench --variants default,norepack      # adds the repack variant where FITS_REPACKED allows it
bash pi/05_run.sh stream-bench                           # llama-completion -n 64 + read_bytes/majflt per token + HE sample
bash pi/05_run.sh chat -p 'Explain a mixture-of-experts model in two sentences.'
bash pi/05_run.sh server --host 127.0.0.1 --port 8080    # reach it through: ssh -L 8080:127.0.0.1:8080 ...
bash pi/05_run.sh server --draft auto                    # MTP sidecar, --spec-type draft-mtp passed explicitly
bash pi/05_run.sh stream-bench --dry-run                 # the exact command lines only
```

- **Gates.** Every subcommand runs `assert_safe_host`, then `assert_not_busy`, then
  `require_verdict_ok --allow-slow`, then model selection and the RAM gate.
- **Model selection.** `--model` (a GGUF path or catalog id), else `QUANT_OVERRIDE`, else the verdict's
  recommendation. If that file is not in `MODEL_DIR`, the best runnable row whose file is there.
- **Flags.** They come from that row in `pi/out/feasibility.json`, through `safe_llama_flags`.
- **Outputs.**
  - stdout: one results-envelope path per line (`results/pi/E8_*.json`, written by `pigemma.bench`).
  - `results/pi/bench_<model>_<variant>_<UTC>.json`: the raw llama-bench JSON.
  - `results/pi/raw/*.vmstat.log`.
  - `pi/out/last_run.json` and `pi/out/last_<subcommand>.json`.
  - `results/pi/calibration.json`, after an in-RAM bench or stream-bench (section 10).
- **Envelope notes.** Every envelope has `extra.pi05`: verdict, mode, effective flags, RAM gate,
  thermals before and after, pinner state and vmstat summary.
- **Repack.** The bench `default` variant is refused whenever the verdict says FITS_REPACKED is false,
  and on a protected host. If every requested variant is refused, the exit is 3.
- **Drafts.** `--draft` is for chat and server only: `-md` is not wired into llama-bench or
  llama-completion at a7b94df.
- **Exit codes.** 0 = ok. 1 = usage error, internal error, or a failed run (the envelope is still
  written with a `failed:` note). 3 = not viable, RAM gate, missing model, draft, binary or prompt,
  size mismatch, or every bench variant refused. 4 = protected or busy host, bench lock held, port in
  use, or `--pin` where mlock is not allowed. A slow verdict (2) is accepted.

Stdout of a bench run is one envelope path per line. The filename has the form
`E8_<model>_<variant>_<threads>_<UTC>_<run>.json`, for example
`results/pi/E8_26b-ud-q3_k_m_norepack_4_<UTC>_<run>.json`. This is the format, not a real file. Read the
numbers with `python3 -m pigemma.results summarize results/pi --experiment E8`.

What to expect on `context-friend-1` with `26b-ud-q3_k_m`. These numbers are **MEASURED** with the ad-hoc
drivers that preceded pi/05 (`pi/adhoc/serve_measure.py`, llama-bench), `-nr`, 4 threads; pi/05 itself
has not run on the real Pi yet.

| Quantity | MEASURED on context-friend-1 | PROJECTED (pi/00) | Source |
|---|---|---|---|
| Cold load from the microSD | 148 s, read exactly the file once | about 150 s (12.73 GB / 85 MB/s) | results/pi-real/README.md |
| Generation, short prompts EN / HE / DE / code | 3.17–3.27 tok/s | 2.14 tok/s (pre-registered E8 band 1.7 to 3.0) | results/pi-real/README.md |
| Prompt processing | 9.1–9.3 tok/s | — | results/pi-real/README.md |
| Generation after an 821-token prompt | 2.50 tok/s | — | results/pi-real/README.md |
| Storage reads during inference (model in RAM) | 0 bytes, 0 major faults | about 0 | results/pi-real/README.md |
| QAT Q4_0 26B (split parts on a USB stick), short prompts | 2.36 to 4.14 tok/s; never fully resident next to the services | 2.73 tok/s | results/pi-real/README.md |
| 31B UD-IQ3_XXS, llama-bench tg32 | 0.53 tok/s (repack on and off identical) | 0.55 tok/s | results/pi-real/31b-iq3xxs/ |

The flags pi/05 uses there: the row is FITS_REPACKED with `-lm mmap -t 4 -c 4096 -fa on -ctk q8_0 -ctv q8_0`.
On a protected host pi/05 forces `-nr`, and the bench variant list is `norepack` only.

**Speculative decoding.** It only engages with `--spec-type draft-mtp`, which pi/05 passes for MTP
sidecars. On `context-friend-1` the 26B MTP sidecar MEASURED +34 % on code, −2 % on English, −25 % on
German and −29 % on Hebrew: −6 % mean over the four prompts. Leave `--draft` off for general chat; it
pays only for code.

### Step 06: pin the hot set and A/B-test

```bash
bash pi/06_pin_hotset.sh start --model 26b-q4 --stats traces/26b/all.stats.json
bash pi/06_pin_hotset.sh status
bash pi/06_pin_hotset.sh stop
bash pi/06_pin_hotset.sh ab --model 26b-q4 --budget-bytes 5.2e9 --rounds 2
bash pi/06_pin_hotset.sh start --dry-run --model 26b-ud-q3_k_m
```

- **start.** Plans the hot set with `python3 -m pigemma.gguf_map hotset`: the non-expert tensors first,
  then the most-used experts from the desktop routing statistics, page-aligned with
  `--page-size $(getconf PAGESIZE)`. Then it starts `tools/build/bin/pinner` as a daemon at `nice -n 10`
  and waits for its one-line JSON summary. The map is `results/map26.json` when it describes the local
  GGUF; otherwise one is built from the local header into `pi/out/map_<stem>.json`. Without stats, only
  the non-expert tensors are pinned.
- **stop.** Sends SIGTERM to the pid in `pi/out/pinner.pid` only, and only when that process is a pinner.
- **status.** Reports VmLck and the hot set's page-cache residency (`pinner --verify-only`). Exit 3 when
  no pinner runs.
- **ab.** Runs `pi/05_run.sh stream-bench` unpinned and pinned per round (ABBA with `--rounds 2`) under
  one bench lock and writes `results/pi/pin_ab.json`.
- **Budget.** `--budget-bytes`, else `PIN_BUDGET_BYTES`, else auto: MemAvailable − KV(ctx) − 0.3e9
  compute − 0.5e9 headroom, capped at the file size. 5200000000, 5.2e9, 5.2GB and 2GiB are all accepted.
- **Needs.** mlock allowed (`mlock_allowed`) and `ulimit -l` ≥ the hot set (or `unlimited`). The pinner
  binary from `pi/02 --with-tools`. Python. A single-file GGUF.
- **Writes.** `pi/out/hotset.json`, `pi/out/hotset.ranges.txt`, `pi/out/pinner.pid`,
  `pi/out/pinner.log` (exactly the pinner's JSON line), `pi/out/pinner.stderr.log` and
  `pi/out/pin_state.json`. `ab` also writes `results/pi/pin_ab.json` and a timestamped copy, plus
  per-run evidence in `pi/out/pin_ab_<UTC>/`.
- **Exit codes.** 0 = ok. 1 = usage error, internal error, failed run, pinner failure or start timeout.
  2 = partial pin (start; the daemon keeps what it locked, and pi/05 accepts it). 3 = precondition:
  model, pinner, map, stats or Python missing, memlock limit too small, RAM, a budget below the
  non-expert set, a split GGUF; also `status` with no pinner. 4 = protected host, mlock not allowed,
  a pinner already running, or the bench lock held by another job.

`pin_ab.json` carries `threshold_check` with the pre-registered E6 rule: PASS needs a pinned/unpinned
tok/s ratio ≥ 1.15 **and** a bytes-per-token reduction ≥ 0.20. When the whole model fits in RAM
(`in_ram: true`) the result is INCONCLUSIVE: there is nothing left to stream. On a 16 GB board,
emulate a smaller one first with the bounded balloon, `sudo python3 pi/adhoc/balloon.py 8 3600 &`.
It refuses to leave less than 2.5 GiB available and frees its memory after at most 3600 s. Without
pinning, that 7.5 GB situation MEASURED 0.55–0.79 tok/s on `context-friend-1`. That is the baseline
the pinner must beat at the same memory budget.

**On `context-friend-1` step 06 refuses (exit 4)**, and that is intended: mlock of model weights is not
allowed there, and `ulimit -l` is 8192 kB. Running it would need the operator's explicit approval for all of
`ALLOW_MLOCK=1`, `bash pi/01_tune_os.sh --apply --only e --persist` with a new login, and
`--allow-protected-host`. The QAT Q4_0 parts on its USB stick are a split GGUF, which step 06 refuses
in any case.

## 8. llama.cpp flags per verdict, and why repack is refused

The flag strings below come from `pigemma/feasibility.py` (`FLAGS_*`), shown at `THREADS=4` and
`CTX=4096`. pi/05 uses the recommended row's flags. On a host where mlock is not allowed,
`safe_llama_flags` replaces `-lm mmap+mlock` with `-lm mmap`.

| Verdict | Flags | When | Notes |
|---|---|---|---|
| FITS_REPACKED | `-lm mmap -t 4 -c 4096 -fa on -ctk q8_0 -ctv q8_0` | file + 0.3 GB fits in the usable RAM − 1.0 GB headroom | default load: llama.cpp repacks Q4_0 into faster kernels in **anonymous** memory |
| FITS_MAPPED | `-nr -lm mmap+mlock -t 4 -c 4096 -fa on -ctk q8_0 -ctv q8_0` | the file fits in RAM only file-backed | `-nr` keeps every weight in the page cache (`CPU_Mapped`). mlock pins it and needs `ulimit -l unlimited`; without it llama.cpp warns and continues unlocked |
| STREAM_OK, STREAM_SLOW | `-nr -lm mmap -t 4 -c 4096 -ctk q8_0 -ctv q8_0 -b 256 -ub 64` | MoE: non-expert weights + KV + buffers fit, experts stream from storage | the smaller batch keeps compute buffers small and leaves the rest of RAM to the page cache as expert cache |
| NEVER_GOOD_SPEED (runnable) | the FITS flags of its RAM fit | a dense model that fits in RAM at ≥ 0.3 tok/s | a measurement, not a usable speed |
| NOT_VIABLE, NO_DISK | none | — | pi/05 refuses (exit 3) |

Why the default (repack) mode is refused when the anonymous copy would not fit:

- **Repack copies the weights out of the mmap.** By default llama.cpp moves every Q4_0 tensor into a
  `CPU_REPACK` buffer. That is anonymous memory, almost the size of the file (about 14 GB for the 26B
  Q4_0). Only the default buffer type is served from the mmap (`src/llama-model.cpp`, the
  `is_default_buft` condition at line 1856). Anonymous pages cannot be dropped and re-read like page
  cache: if they do not fit, the load ends in swap or the OOM killer, and on a shared board the
  services lose memory first.
- **`-nr` / `--no-repack` (`--repack 0` in llama-bench) is the only mode that keeps the weights
  file-backed** at a7b94df. The page cache can then shrink under pressure, and `-nr` is also the
  only page-cache streaming mode.
- **`-ot "ffn_(gate_up|down)_exps=CPU"` does not help.** An override to the CPU buffer type re-runs the
  buffer selection over a list that still contains `CPU_REPACK` (`src/llama-model-loader.cpp` lines
  1240 to 1242). It is kept only as the negative control `hybrid_negctrl`.
- **`-lm none` (no mmap)** reads the whole file into anonymous memory, with the same OOM risk.
- **The arithmetic.** Feasibility calls a file FITS_REPACKED only if file + 0.3 GB ≤ usable − 1.0 GB.
  For `26b-q4` on a 16 GB board that is 14.74 GB ≤ 14.65 GB, which fails, so it is FITS_MAPPED with `-nr`.
- **What `-nr` costs and what it saves.** It costs kernel speed, which E0 measures (the Q4_0 repack
  kernels are the faster ones). Control C2 checks that it does not change the output (KLD of `-nr` vs
  default, [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md#controls)). MEASURED with `-nr`: 0.44 GB anonymous
  memory for the 31B, and 0.52 GB rising to 0.99 GB after an 821-token context for the 26B UD-Q3_K_M.
  Everything else stayed reclaimable page cache.
- **How to check which mode a run used.** `llama-completion ... -lv 4` prints `load_tensors: CPU_REPACK
  model buffer size = ...` for repacked tensors and `CPU_Mapped model buffer size = ...` for mapped
  ones. `pigemma.bench` stores these as `buffer_split` in each completion envelope.

## 9. Desktop artefacts to the Pi, Pi results to the desktop

| Artefact | Produced on the desktop by | Used on the Pi by | Note |
|---|---|---|---|
| `results/map26.json` (131 KB) | `python -m pigemma.gguf_map build 26b-q4 --out results/map26.json --reuse` | pi/06 `--map` (the default) | describes the QAT Q4_0 file only. For any other quant pi/06 builds `pi/out/map_<stem>.json` from the local header |
| `traces/26b/all.stats.json` | `bash pc/trace_collect.sh --quick` (E3) | pi/06 `--stats` (the default): ranking of the hot experts | valid for every 26B quant: same 30 layers × 128 experts. That the routing of a 3-bit file matches the Q4_0 trace is ASSUMED |
| `results/h1/sim.json` | E4: `python -m pigemma.cache_sim ...` | `bash pi/00_feasibility.sh --sim results/h1/sim.json` (also pi/04 and pi/05 `--sim`) | replaces the prior LRU hit curve with simulated hit rates |
| hot set (`results/h1/hotset_<cap>.json` + `.ranges.txt`) | E6 or `python -m pigemma.gguf_map hotset --page-size 16384 ...` | `bash pi/06_pin_hotset.sh start --hotset pi/out/hotset_desktop.json` | only for the **identical** GGUF (the pinner checks `size=` in the ranges header) and the Pi's 16 KiB page. Otherwise let pi/06 plan on the Pi |
| GGUF files | `models/` | step 04 | prefer `pi/04 --from-desktop` (needs sshd and rsync on the sender) or the Hugging Face download |

From Git Bash (no rsync there) with scp:

```bash
cd REPO_ROOT
PI="-o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1"
ssh $PI pi@PI_HOST 'mkdir -p ~/pi-gemma4/results/h1 ~/pi-gemma4/traces/26b'
scp $PI results/map26.json pi@PI_HOST:pi-gemma4/results/
scp $PI traces/26b/all.stats.json pi@PI_HOST:pi-gemma4/traces/26b/
scp $PI results/h1/sim.json pi@PI_HOST:pi-gemma4/results/h1/
```

With rsync on both ends (a Linux desktop, WSL with rsync, or from the Pi pulling), one command keeps
the relative paths:

```bash
rsync -av --relative -e "ssh $PI" results/map26.json traces/26b/all.stats.json results/h1/sim.json pi@PI_HOST:pi-gemma4/
```

To bring the Pi results back for the write-up and the re-projection (section 10):

```bash
scp $PI -r pi@PI_HOST:pi-gemma4/results/pi results/
```

## 10. How measured values replace assumed ones

The catalog file is **never rewritten**. Measured files are merged in memory when they are present
(`--io-mem auto` and `--calibration auto` are the defaults). Every merged field's `source` then starts
with `measured:`, and every field that was not measured keeps its `assumed:` source.

| Quantity | Catalog value (ASSUMED) | Measured by | File and field | Merge rule | Read by |
|---|---|---|---|---|---|
| sequential read | microSD 85, USB 3.0 SSD 380, NVMe Gen2 430, NVMe Gen3 850 MB/s | pi/03 fio `seq_read_qd1` / `seq_read_qd8` (dd fallback) | `results/pi/io_mem.json` `storage.seq_read_mbps_qd1`, `storage.seq_read_mbps_qd8` | max(qd1, qd8) | feasibility (dense streaming, load time), cache_sim |
| random 3 MB read | 60, 350, 400, 700 MB/s | pi/03 fio `rand_read_3mb_qd1` / `_qd4` | `storage.rand_read_3mb_mbps_qd1` | qd1 (falls back to qd4) | feasibility and cache_sim t_io (expert misses) |
| access latency | 2.0, 0.5, 0.2, 0.2 ms | pi/03 fio `rand_read_16k_qd1` mean latency | `storage.access_ms` | as measured | t_io per missed slab |
| memory bandwidth | 10.0 GB/s | pi/03 membw read at 1, 2 and 4 threads | `memory.read_gbps_by_threads` | best thread count (falls back to copy, then triad) | t_ram, in-RAM ceiling |
| llama efficiency | 0.65 | pi/05 bench or stream-bench with the model in RAM: tg tok/s × bytes per token / memory bandwidth | `results/pi/calibration.json` `llama_efficiency` | used as given (0.05 to 1.5; not written for emulated runs or runs that read > 2% of the bytes per token from storage) | feasibility, cache_sim `--calibration` |
| expert-cache hit rate | prior curve: 0.15 → 0.60, 0.25 → 0.74, 0.35 → 0.84, 0.45 → 0.88 | E3 trace + E4 simulation on the desktop | `results/h1/sim.json` (LRU `steady.hit_rate`) | source `simulated` | `pigemma.feasibility --sim` |

The loop, in order:

1. **On the Pi**, run `bash pi/03_bench_io_mem.sh`, then `bash pi/05_run.sh bench --variants norepack`
   (writes `calibration.json` when the model is in RAM), then `bash pi/00_feasibility.sh`. The pi/00
   report's `projections` line now shows `[measured]` next to `mem_bw_gbps`, `seq_read_mbps`, and so on.
   Its `inputs` line names `results/pi/io_mem.json` and `results/pi/calibration.json`. To read the
   sources directly:
   `python3 -m pigemma.feasibility --facts pi/out/facts.json --json - | python3 -c "import json,sys; d=json.load(sys.stdin)['device']; print(d['mem_bw_gbps'], d['storage'])"`.
2. **Copy `results/pi/` to the desktop** (section 9) and recompute the E4 projections with the Pi's
   measured constants. Writing to a separate directory keeps the desktop E4 envelopes untouched:

   ```bash
   python -m pigemma.cache_sim traces/26b/all.moetrace --map results/map26.json --profiles pi5 --io-mem results/pi/io_mem.json --calibration results/pi/calibration.json --read-granularity 2097152 --out results/h1/pi-measured/sim.json
   python -m pigemma.results summarize results/pi --experiment E8 --md docs/results/PI_2026-10-05.md
   ```

   Set `--read-granularity` to the Pi's `read_ahead_kb` × 1024. 2097152 matches pi/01's 2048 KB; the
   stock value is 128 KB = 131072, which is also cache_sim's default. With `--profiles pi5`, the device
   that io_mem.json names is simulated even when the catalog does not list it (`pi5-16gb-microsd` is
   synthesized).
3. **Copy `results/h1/sim.json` to the Pi** and run `bash pi/00_feasibility.sh --sim results/h1/sim.json`.
   The streaming rows now use simulated hit rates instead of the prior curve.
4. **Report.** The E8 results template in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md#e8-pi-runtime-dry-run)
   lists which rows each number fills. A measured constant that moves a profile across a verdict
   boundary goes into its "What would change this conclusion" line.

Emulated runs (`PIGEMMA_EMULATED=1`, the arm64 container on the desktop) write to `results/pi-emu/` and
are never merged automatically: qemu speed numbers are not Pi numbers.

## 11. Smoke tests

Run these after steps 00 to 05, in this order. Each one has a pass criterion fixed before the run.

**1. Normal chat in English and Hebrew** (greedy, one turn each):

```bash
bash pi/05_run.sh chat -p 'What is the capital of France? Answer with one word.'
bash pi/05_run.sh chat -p 'מהי בירת צרפת? ענה במילה אחת.'
```

- **Pass.** The visible answer contains `Paris` and `פריז`. A thinking block
  `<|channel>thought ... <channel|>` may come first; it is not part of the answer. The sentences
  are coherent, and the Hebrew is in Hebrew script.
- **Load time.** Each `chat -p` loads the model again. Warm (file still in the page cache) that takes
  about 4 s. Cold, it took 148 s from the microSD and 125 s for the QAT file from the USB stick (MEASURED).
- **Already MEASURED on `context-friend-1`** (ad hoc, 2026-10-05): both 26B files answered all 16 basic
  questions and all 10 Hebrew questions correctly; strict scores were 35/36 (UD-Q3_K_M) and 34/36 (QAT
  Q4_0) on the 36-item set. The strict failures were markup glitches around the thinking channel,
  not wrong answers.
- **Optional: the scored set** through `pigemma.quality` (one llama-server load, 16 items):

```bash
LLAMA_DIR=~/pigemma/llama.cpp python3 -m pigemma.quality qa --model ~/pigemma/models/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf --threads 4 --load-variant norepack --nice 10 --min-free-gb 13.3 --out results/pi/
```

`--min-free-gb` is the bench lock's RAM gate. Its default (20) is for the desktop; 13.3 is the file
size + 0.5 GB rule. The pre-registered pass is ≥ 13/16 overall and ≥ 4/6 Hebrew.

**2. stream-bench numbers:**

```bash
bash pi/05_run.sh stream-bench
python3 -m pigemma.results summarize results/pi --experiment E8
```

- **Pass when the model fits in RAM.** After the first, cold run, `read_bytes_per_token` is about 0 and
  `majflt_per_token` is about 0 (MEASURED 0 bytes and 0 major faults with the 26B on
  `context-friend-1`). tg lies inside or above the pre-registered E8 band of 1.7 to 3.0 tok/s for
  `26b-ud-q3_k_m`.
- **Above the band.** The ad-hoc 3.17–3.27 tok/s already is, so the projection was wrong in the
  optimistic direction. `calibration.json` then replaces the 0.65.
- **Pass when it streams** (8 GB boards). `read_bytes_per_token` is at most the worst-case expert bytes
  per token (`26b-q4`: 802,897,920 B; `26b-ud-q3_k_m`: 644,548,608 B), and tok/s is reported next to
  pi/00's PROJECTED row. More bytes than the worst case means read-ahead waste or thrash (section 12).
- **Output.** The EN and HE outputs are printed; they must be coherent (`coherent: true` in the
  envelope).

**3. Pin A/B** (only on a board where mlock is allowed, see step 06):

```bash
bash pi/06_pin_hotset.sh ab --model 26b-q4 --budget-bytes 5.2e9 --rounds 2
```

Pass is `threshold_check.result` = PASS in `results/pi/pin_ab.json`: tok/s ratio ≥ 1.15 and
bytes-per-token reduction ≥ 0.20, both pre-registered for E6. INCONCLUSIVE with `in_ram: true` means
the board holds the whole model; use the balloon, or a smaller board.

## 12. Troubleshooting

### OOM on load (repack mode without room)

- **Symptoms.**
  - The process dies during "loading model" with `Killed`, exit 137.
  - `journalctl -k` or `sudo dmesg` shows `oom-kill` or `Out of memory: Killed process ... (llama-...)`.
  - RssAnon climbs towards the file size, and services may get swapped to zram.
- **Cause.** The default repack load (no `-nr`) allocates an anonymous copy of the weights (section 8),
  or `-lm none` reads the whole file into anonymous memory.
- **Fix.**
  - Use `-nr` (`--repack 0` for llama-bench). pi/05 adds it whenever the row is not FITS_REPACKED and
    on every protected host. For hand-written commands, copy `llama flags` / `on this host` from pi/00.
  - Check the mode with `-lv 4` (`CPU_REPACK` vs `CPU_Mapped` buffer lines).
  - If even `-nr` fails, the model does not fit at all: use a smaller quant, a smaller `-c`, or close
    other memory users. On a shared box, never retry a repack load to "see if it fits".

### Downloads time out (ETIMEDOUT) at about 10 MB/s

- **Symptoms.** `curl: (28) Operation too slow` or `Connection timed out` in `pi/out/fetch.log`; long
  stalls; the SSH session drops.
- **Cause.** The Pi's Wi-Fi drops packets under load (66 % loss for about 15 min observed on 2026-10-05).
  Hugging Face delivers about 7 to 8 MB/s at best.
- **Fix.**
  - Use `--bg` (or `nohup`) and poll the log. pi/04 already retries with separate `curl -C -` calls that
    resume from the current `.part` size, and re-running pi/04 resumes from where it stopped.
  - Never use `curl --retry` by hand: it resumes from the first attempt's offset and truncated a 4.3 GB
    partial.
  - `HF_LIMIT_RATE=5M` leaves room for SSH and the services.
  - The alternatives are `--from-desktop` (rsync with `--partial --append-verify`), or the resumable
    ad-hoc copy `pc/adhoc_copy_split_to_pi.sh`. Its split parts suit a FAT32 stick, but pi/06
    cannot use them.
  - A wrong-size file is never deleted automatically: the message prints the `rm` to run.

### memlock: EPERM or ENOMEM

- **Symptoms.**
  - llama.cpp prints `warning: failed to mlock ... -byte buffer (after previously locking ... bytes):`
    followed by `Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).` and continues unlocked.
  - The pinner exits 3 with `pinner: REFUSED:` and a `pinner: hint:` line, or exits 2 (partial) with
    `--best-effort`.
- **Cause.** `ulimit -l` is finite (8192 kB on `context-friend-1`), or the process lacks
  `CAP_IPC_LOCK`.
- **Fix.** On a board where mlock is allowed, run `bash pi/01_tune_os.sh --apply --only e`, open a
  **new** login session, and check that `ulimit -l` prints `unlimited`. systemd-started processes only
  see `DefaultLimitMEMLOCK` after a reboot. On a protected host this is refused by design
  (`ALLOW_MLOCK=auto`): run without mlock, which pi/05 already does.

### read_ahead thrash

- **Symptoms.**
  - When streaming, stream-bench's `read_bytes_per_token` is larger than the worst-case expert bytes
    per token (section 11), or `majflt_per_token` keeps rising run after run.
  - tok/s falls far below pi/00's projection while the storage LED never stops.
- **Cause.**
  - Each 16 KiB page fault makes the kernel read around the faulting page (`read_ahead_kb`). Too large
    a window reads neighbouring experts nobody needs and evicts hot pages. Too small a window turns a
    2.2 MB `gate_up` slab into many latency-bound reads.
  - Separately, a page cache that is too small for the working set (non-experts + the hot experts)
    thrashes whatever the window size.
- **Fix.**
  - Compare 2 or 3 values: `TUNE_READAHEAD_KB=512 bash pi/01_tune_os.sh --apply --only d`, re-run
    `bash pi/05_run.sh stream-bench`, and compare `read_bytes_per_token` and tok/s. The slabs are 1.1 MB
    (`down`) and 2.2 MB (`gate_up`).
  - Keep cache_sim's `--read-granularity` equal to the value used.
  - Give the page cache more room: a smaller `-c`, fewer other processes, a smaller quant.
  - Pin the hot set (step 06) so the hottest pages cannot be evicted.

### Thermal throttling

- **Symptoms.** tok/s drifts down during a long run; `vcgencmd get_throttled` is not `0x0`;
  `vcgencmd measure_temp` is near 80 °C; `vcgencmd measure_clock arm` is below 2.4 GHz.
- **Reading `get_throttled`.** 0x4 = throttled now, 0x2 = frequency capped now, 0x8 = soft temperature
  limit now, 0x1 = under-voltage now. 0x40000, 0x20000, 0x80000 and 0x10000 mean the same has happened
  since boot.
- **Fix.** Fit the Active Cooler and use the official 27 W supply, then re-run. pi/05 records the
  thermals before and after every run in `extra.pi05.thermal`. MEASURED reference with the Active
  Cooler: 72.5 °C peak over 23 min of the 31B, and 72 °C for the 26B, `throttled=0x0`.

### Every script exits 4 on context-friend-1

This is intended: the hostname is in the catalog's `protected_hosts`, and `contextfriend-*` services are
running. Export `PIGEMMA_I_KNOW=1` (section 6) **only** under the operator's authorisation. A busy refusal
(loadavg or MemAvailable) is never lifted; wait until the services are idle.

### Illegal instruction (SIGILL)

The binaries were built for a CPU with i8mm, SVE or SME: for example `GGML_NATIVE=ON` under qemu, or
copied from another board. Rebuild on the Pi with `bash pi/02_build_llama.sh --rebuild`. Its `--arch auto`
falls back to the Pi 5 ISA (`armv8.2-a+dotprod+fp16`) when a binary dies with SIGILL.

### llama.cpp reports "build 1"

A shallow clone has no commit count. pi/02 passes `-DLLAMA_BUILD_NUMBER=11398` when HEAD is exactly the
pinned commit, and `pigemma.bench` adds a note when a tool reports build 1. The commit hash is what
identifies the build.

### The MTP draft loads but never drafts

At a7b94df the MTP drafter only drafts with an explicit `--spec-type draft-mtp`; auto-detection only
recognises DFlash. pi/05 `--draft auto` passes it. The server log shows the acceptance.

### The USB stick is treated as a USB 3.0 SSD

A flash stick has transport `usb` and is classified `usb-ssd` (380 MB/s ASSUMED). The stick on
`context-friend-1` read at about 70 MB/s ([results/pi-real/README.md](../results/pi-real/README.md)).
Run `bash pi/03_bench_io_mem.sh --dir /mnt/usb/pigemma/models` so the measured number replaces the
assumption before trusting any streaming verdict for it.

### pi/06: stats inconsistent with the map

The stats must describe the same layer and expert layout as the map (30 MoE layers × 128 experts for
every 26B quant). An explicit `--stats` that does not fit exits 1. The default stats file is optional:
if it is missing or does not fit, pi/06 pins only the non-expert tensors and warns.

### A stale bench lock

`python3 -m pigemma.paths lock --status` shows the holder. A lock whose pid is dead on this host (or
whose heartbeat stopped) is replaced automatically by the next job. Never delete a lock whose holder
is alive.

## 13. Status on the Pi: measured, projected, assumed

| Quantity | MEASURED (host, date, file) | PROJECTED | ASSUMED (catalog) |
|---|---|---|---|
| microSD sequential read | 91.6 MB/s raw, O_DIRECT 4 MiB (context-friend-1, 2026-10-05, results/pi-real/README.md) | — | 85 MB/s |
| microSD random 3 MB read, access latency | not yet (fio missing) | — | 60 MB/s, 2.0 ms |
| Memory bandwidth | not yet (pi/03 membw) | — | 10.0 GB/s |
| llama efficiency | not yet (pi/05 calibration) | — | 0.65 |
| 26B UD-Q3_K_M generation, in RAM | 3.17–3.27 tok/s (results/pi-real/README.md) | 2.14 tok/s | — |
| 26B QAT Q4_0 generation (split, USB stick, not fully resident) | 2.36 to 4.14 tok/s, load 125 s (results/pi-real/README.md) | 2.73 tok/s | — |
| 26B UD-Q3_K_M at 7.5 GB usable RAM (balloon) | 0.55–0.79 tok/s (results/pi-real/README.md) | — | — |
| 31B UD-IQ3_XXS tg32 | 0.53 tok/s (results/pi-real/31b-iq3xxs/) | 0.55 tok/s | — |
| Cold load 26B UD-Q3_K_M from microSD | 148 s | about 150 s | — |
| Cold load 31B UD-IQ3_XXS from microSD (`-nr`) | 138 s, about 86 MB/s (results/pi-real/README.md) | 139 s (11.84 GB / 85 MB/s) | — |
| llama.cpp native build | 410 s, 5 targets, `-j 4` | — | — |
| Peak temperature with the Active Cooler | 72.5 °C, `throttled=0x0` | — | — |
| 26B on the 16 GB + NVMe Gen3 recommendation | not yet (no such board) | 2.73 tok/s, about 17 s load | 850 / 700 MB/s, 0.2 ms |
| 26B on 8 GB + NVMe Gen3 (expert streaming, H1) | not yet | 1.94 tok/s with the prior hit curve | 850 / 700 MB/s, 0.2 ms |
| Hot-set pinning gain (E6 on a Pi) | not yet (refused on context-friend-1) | — | — |

## 14. Usage lines (reference card)

These are the scripts' own usage lines, copied verbatim. `tests/test_pi_runbook.py` checks that they
still match the scripts. Run any script with `--help` for every option.

```text
pi/00_feasibility.sh [options] [-- extra pigemma.feasibility args]
pi/01_tune_os.sh [--apply] [--no-pcie]
pi/01_tune_os.sh [--apply] [--revert] [--dry-run] [--only LIST] [--skip LIST]
                 [--persist | --runtime-only] [--allow-boot-edit]
pi/02_build_llama.sh [options]
bash pi/03_bench_io_mem.sh [--file auto|new|PATH] [--dir DIR] [--size-mb 2048] [--seconds 20] [--quick]
                           [--out FILE] [--device NAME] [--threads 1,2,4] [--membw-size-mb 512]
                           [--membw-reps 5] [--ioengine libaio] [--slab-bytes N] [--skip-io] [--skip-mem]
                           [--require-fio] [--no-busy-check] [--dry-run]
bash pi/03_bench_io_mem.sh --check-jobs [options]
bash pi/03_bench_io_mem.sh --from-raw DIR [--out FILE] [--device NAME]
pi/04_fetch_model.sh [options] [-- extra pigemma.feasibility args]
pi/05_run.sh bench|stream-bench|chat|server [--model <gguf>] [--draft <gguf>] [--variants default,norepack] [--pin] [-p 'text']
pi/06_pin_hotset.sh start|stop|status|ab [--model <gguf>] [--map results/map26.json] [--stats traces/26b/all.stats.json] [--budget-bytes N]
bash pi/lib/common.sh <command> [args]
```

`pi/lib/common.sh` can be run directly for quick checks. Its commands are `facts`, `write-facts [OUT]`,
`safe-host`, `not-busy`, `verdict [opts]`, `jobs [--auto]`, `config`, `storage [DIR]`,
`fmt-gb BYTES`, `mem-available-kb`, `free-disk [PATH]`, `version` and `help`. For example,
`bash pi/lib/common.sh safe-host` answers "would every script refuse here?" without running anything
heavy, and `bash pi/lib/common.sh facts` prints the facts JSON.

## 15. Deviations from the manifest spec

- **The context-friend-1 deny-listing has moved.** The spec calls `context-friend-1` a deny-listed
  hostname. Since 2026-10-05 it is the real target Pi, so `DENY_HOSTS` is empty by default. The hostname
  is still refused through the catalog's `protected_hosts` and through its running `contextfriend-*`
  services, and `PIGEMMA_I_KNOW=1` is the authorised override ([docs/design/FACTS.md](design/FACTS.md)).
- **Copying uses tar or scp before rsync.** The spec says rsync for shipping artefacts. Git Bash on the
  desktop has no rsync, so tar over ssh and scp are given first; rsync works where both ends have it.
- **The pin A/B smoke test cannot run on context-friend-1.** mlock is not allowed there (D4 in
  [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md#deviations-log)). It runs on a dedicated board, or with
  the owner's explicit approval as described in step 06.
- **The 3-bit 26B is FITS_REPACKED, not FITS_MAPPED.** On `context-friend-1` pi/00 computes
  FITS_REPACKED for `26b-ud-q3_k_m`, while the E8 expectation table lists FITS_MAPPED for it. pi/05
  forces `-nr` on that protected host either way, so the flags used are the same.
