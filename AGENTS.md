# Agents

Read this file before you change the repository or touch the Raspberry Pi.
Humans use [README.md](README.md). The measured handover is [CLAUDE.md](CLAUDE.md).
Its top is generated: edit `docs/cf-engine/STATUS.md`, then run
`python cf-engine/update_handover.py`. That rewrite keeps the text from
"Historical decision brief" downward. Append a progress-log line there when you
finish a step.

Developed by TamerinTECH GmbH. The public repository is
https://github.com/TamerinTECH/pi-gemma4 on branch `main`. This repository is
MIT ([LICENSE](LICENSE)). Upstream checkouts keep their own licences. Commits
and pushes only when asked.

## Install

Desktop, from an empty directory. Python 3.12 or newer.

```powershell
git clone https://github.com/TamerinTECH/pi-gemma4.git
cd pi-gemma4
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest tests/test_no_secrets.py
.\.venv\Scripts\python.exe -m pigemma.paths info
```

On the Pi the same install is `python3` and `requirements.txt` (or the apt
packages named in that file). A checkout can run `pytest` with no model and no
llama.cpp. `pytest -m integration` needs real GGUFs, Docker, or native binaries.

Do not `pip install` into the global interpreter on the desktop. Do not put a
token in a clone URL.

## What this repository is

A measurement project: can a full Gemma 4 run on a Raspberry Pi 5, CPU only.
Three tracks stay separate. Do not report a win in one as a win in another.

| Track | Question | Where the answer lives |
|---|---|---|
| A | Same model, tight memory, exact arithmetic | `docs/results/`, `pigemma/controlled.py` |
| B | A practical model on this Pi | `docs/fast/DECISION_2026-10-06.md` |
| C | Dense residual factorisation | `docs/results/DENSE_FEASIBILITY.md` (stopped, negative) |

Label every number **MEASURED**, **PROJECTED**, or **ASSUMED**. A speed claim
needs a result file. A loaded desktop is not a benchmark.

The original harness (`pc/`, `pi/00`–`pi/06`) runs stock llama.cpp `a7b94df`.
The fast profile is a patched copy of that same commit. Every patch is off
unless its environment variable is set. With none set, the patched build matches
the reference. Details: [pi/fast/README.md](pi/fast/README.md) and
[docs/fast/LAYER_SKIP.md](docs/fast/LAYER_SKIP.md).

Current decision (measured 2026-10-06, Pi 5, 16 GB, CPU only): prompt processing
on E2B reaches about 170 tok/s with the layer skip. Open-text generation is
about 7.5 tok/s plain and about 14 tok/s with the MTP drafter. 100 tok/s of
generated text is above what this board can do for any full Gemma 4. The short
version is section 8 of the decision doc.

## Safety

The measured board is `context-friend-1`. It runs production services. Never
stop, restart, or kill `contextfriend-box`, `contextfriend-gpu-tunnel`,
`hailort`, or `whisper-bench@base`. Never kill a process you did not start.

Do not commit that board's address, a home directory, or a token. Scripts that
need SSH read `PI_HOST`. Set `PI_HOST=user@host` outside this repository. The
hostname can also resolve to a different machine. Do not use that one. The
host-key alias below is what selects the measured board.

```text
ssh -o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1 "$PI_HOST"
```

On that Pi:

- One performance job at a time. Take `flock -n ~/pigemma/.bench.lock`.
- Heavy work is `nice -n 10`. Downloads are `ionice -c3`.
- Keep at least 2 GB free on `/`. Do not write large files there. Use the USB
  stick at `/mnt/usb` (mounted by hand; FAT32, 4 GB file limit).
- Do not edit `/boot/firmware/config.txt` or `/etc/fstab`. Do not reboot.
- No mlock of model weights on this host. `pi/06` refuses it.
- Abort under 1 GB available RAM, at 78 °C, on throttling, or if a service is
  unhealthy.
- Ports 18190 and 18191 belong to another local service.
- After a reboot, run `~/pigemma/fast/mount_work.sh` before a fast-profile build.
  Do not overwrite a shell script on the Pi while it is running.
- Wi-Fi drops under load. Anything longer than a minute goes under `nohup`.

Desktop: one model-loading job holds `results/.bench.lock`. Do not prune Docker
images, volumes, or caches you did not create. Background builds use at most
`-j 4`. A CPU-heavy job writes `results/.cpu.lock` and removes it when finished.

The Hugging Face token is not in this repository. The Gemma files used here are
ungated. Scripts that need a token read `HF_TOKEN` or `HUGGING_FACE_TOKEN` from
the environment, or a file passed with `--token-file`. `pc/adhoc_fetch_26b_ud_q3km.sh`
and `pc/adhoc_31b_to_pi.sh` exit if that variable is unset. Never print the
value, and never commit it.

## What must not be committed

`git add -A` will try to pick up local result JSON. That tree is hundreds of
megabytes and includes files GitHub rejects. Add files by name.

| Leave it out | Why |
|---|---|
| `.env`, `*.pem`, private keys, anything matching a token shape | `tests/test_no_secrets.py` fails the commit check |
| `models/`, `*.gguf` | Weights. Multi-gigabyte, not source |
| `third_party/` | Upstream checkouts and their binaries. Clone them |
| `*.exe`, `*.dll`, `*.so`, `*.bin`, `*.o`, build trees, `*.ext4` | Build products. One logits file is 205 MB |
| `results/.cpu.lock`, `llama.log`, `results/**/*.log`, `results/**/*.csv` | Locks, logs, bulky traces |
| Docker volumes and Pi binaries under `/mnt/usb` | They are not in the working tree |

JSON and Markdown already in history stay. New raw dumps stay on the machine
that produced them unless a report needs one named file and the operator asks for the commit.

## Third-party checkouts

Clone these beside the package, under `third_party/`. Do not copy their code
into `pigemma/`, `tools/`, or a product.

| Directory | Upstream | Commit | Licence |
|---|---|---|---|
| `llama.cpp/` | https://github.com/ggml-org/llama.cpp | `a7b94df2c616bc1f62a73b964b4a71cb0dcc488e` | MIT |
| `llama.cpp-upstream-8345f33/` | same | `8345f333951c661d166b00e6f9362e553768f292` | MIT |
| `llama.cpp-a6647b1/` | same | `a6647b1a32f2ee456abff98ee2db15ed6e957c74` | MIT |
| `ik_llama.cpp/` | https://github.com/ikawrakow/ik_llama.cpp | `3d27f5bb6348c4d79edded63b03e9a93efb2f80e` | MIT |
| `hotpin-llm/` | https://github.com/LozzKappa/hotpin-llm | `ac52bd9a136129bc4895af4715d5fa9dfeb03dc5` | AGPL-3.0 plus a commercial licence |

HotPin is comparison-only. Do not copy it into product code. gguf-py is imported
from the pinned llama.cpp tree (`third_party/llama.cpp/gguf-py`), not from pip.

Fast profile, on a clean checkout of the pinned commit:

```bash
patch -p1 < /path/to/pi-gemma4/pi/fast/pigemma-fast-v7.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build -j 4 --target llama-server llama-bench
```

Run the server only with the variables documented in `pi/fast/README.md`.
For a product-style server, use `--reasoning off`. `--reasoning-budget 0` makes
the model write its reasoning into the visible answer.

Linux tool builds (probe, pinner, membw) use `bash pc/docker_run.sh build-tools`
and [docker/Dockerfile.build](docker/Dockerfile.build). On Windows, prefix Docker
paths with `MSYS_NO_PATHCONV=1` when Git Bash rewrites them.

## Where to look

| Need | File |
|---|---|
| Install, for a person | [README.md](README.md) |
| Pi procedure, step by step | [docs/PI_RUNBOOK.md](docs/PI_RUNBOOK.md) |
| Fast-profile decision | [docs/fast/DECISION_2026-10-06.md](docs/fast/DECISION_2026-10-06.md) |
| Layer-skip rule | [docs/fast/LAYER_SKIP.md](docs/fast/LAYER_SKIP.md) |
| Upstream-style patches | [pi/fast/upstream/README.md](pi/fast/upstream/README.md) |
| Hypotheses and thresholds | [docs/HYPOTHESES.md](docs/HYPOTHESES.md) |
| Experiment commands | [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) |
| Pi measurements already in git | [results/pi-real/README.md](results/pi-real/README.md) |

## Before you finish

- Run `python -m pytest` for a code change that the unit tests cover.
- Run `python -m pytest tests/test_no_secrets.py` before any commit.
- Do not claim a command succeeded unless you saw the output.
- Do not start, stop, or kill servers unless the operator says so.
- Leave files you did not mean to change as they are.
