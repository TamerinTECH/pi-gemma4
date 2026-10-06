# tests/fixtures: provenance

Checked-in inputs for the unit tests. Nothing here is a project measurement. The sample outputs
show the exact *format* that the parsers in `pigemma.bench`, `pigemma.quality` and the `pi/`
scripts must handle. The numbers in them come from whatever the capture ran on (a tiny synthetic
model or the desktop), and must never be quoted as results.

Line endings: every text fixture is stored with LF. The Windows `.exe` builds write CRLF; the
fixtures were normalised to LF, which is what the Linux/Pi builds emit and what Python's text mode
(`read_text()`, `subprocess(..., text=True)`) produces from CRLF anyway. Byte content is otherwise
unchanged.

## Real binary output (llama.cpp b11398 = a7b94df, captured 2026-10-05)

"tiny model" means the synthetic llama-architecture GGUFs from `tests/synth.py`
(`write_tiny_llama_gguf`, seed 0: 2 layers, n_embd 64, 354-token byte-level SentencePiece vocab,
random weights). F32 is the base model and Q4_0 its quantisation. The binaries are the official
Windows CPU builds in `third_party/llama.cpp-win/`, run on the desktop (Ryzen 5 8400F), with the
model files in the working directory so no user path is recorded. The capture ran under
`python -m pigemma.paths lock --hold`.

| File | Command | Notes |
|---|---|---|
| `llama_bench_sample.json` | `llama-bench -m tiny-llama-q4_0.gguf -t 1,2 -p 32 -n 16 -r 3 --repack 0,1 -o json` (stdout) | 8 rows: pp32/tg16 x threads 1,2 x repack false/true. Every key of this build's JSON is present (`build_commit` "a7b94df2c", `build_number` 11398, `avg_ts`, `stddev_ts`, `samples_ts`, `repack`, `load_mode`, ...). The tok/s values are for a 119 K-parameter model and are meaningless. |
| `completion_stderr_sample.txt` | `llama-completion -m tiny-llama-q4_0.gguf -f prompt.txt -n 32 --temp 0 -t 2 -c 256 -lv 4 --log-colors off` (stderr) | Default load mode (repack on). Has both `load_tensors:   CPU_Mapped model buffer size =     0.05 MiB` and `load_tensors:   CPU_REPACK model buffer size =     0.04 MiB`, plus the `common_perf_print:` load / prompt eval / eval lines. **At this build the `load_tensors:` buffer lines are only printed at `-lv 4` (verbose)**; the default verbosity (3) prints only the perf lines. |
| `completion_stderr_norepack_sample.txt` | the same plus `-nr` | `-nr`: only a `CPU_Mapped` buffer line, no `CPU_REPACK`. |
| `perplexity_base_sample.txt` | `llama-perplexity -m tiny-llama-f32.gguf -f en_tech_moe_streaming.txt -c 64 --chunks 4 -t 2 --kl-divergence-base base.kld` (stdout+stderr merged, `2>&1`) | Saves the base logits; the last line is `Final estimate: PPL = 1165.0598 +/- 189.16265`. The text is `data/calibration/en_tech_moe_streaming.txt`. |
| `perplexity_kld_sample.txt` | `llama-perplexity -m tiny-llama-q4_0.gguf -f en_tech_moe_streaming.txt -c 64 --chunks 4 -t 2 --kl-divergence --kl-divergence-base base.kld` (stdout+stderr merged) | Contains `Mean PPL(Q)`, `Mean PPL(base)`, `Mean    KLD:`, `Maximum KLD:`, `99.9%   KLD:`, `Mean    Δp:`, `RMS Δp`, and `Same top p: 77.419 ± 3.770 %`. Same top p is a **percentage** here; the results envelope wants `kld.same_top_p` as a fraction (0.77419). The file holds UTF-8 (`±`, `Δ`). |

## Real Gemma 4 output (orchestrator runs, copied from `results/`)

| File | Source | Notes |
|---|---|---|
| `completion_stderr_gemma26b_sample.txt` | `results/smoke-26b-moe.stderr.txt` (llama-completion.exe, 26B-A4B QAT Q4_0, 6 threads, default verbosity, 2026-10-04) | Real perf lines (`load time =   12747.86 ms`, prompt 50.92 tok/s, eval 8.71 tok/s) and **no** buffer lines, because the default verbosity hides them. |
| `server_timings_sample.json` | `results/specsrv-moe-dflash.json` (llama-server `/completion` response, 26B + DFlash draft, 2026-10-05) | `timings.draft_n` 321 and `draft_n_accepted` 91 (28 %). `content` begins with the empty thinking block `<|channel>thought\n<channel|>`. |
| `server_timings_nodraft_sample.json` | `results/specsrv-moe-mtp.json` (26B started with `-md` MTP sidecar) | No `draft_n` key at all and `generation_settings["speculative.types"] == "none"`: the MTP sidecar never drafted. Use it for spec_active = False. |
| `thinking_sample.txt` | `results/smoke-26b-moe.stdout.txt` (llama-completion stdout, 26B) | Echoed prompt with `<start_of_turn>`/`<end_of_turn>` template tokens, an **empty** thinking block `<|channel>thought\n<channel|>`, the English answer, `שלום`, then ` [end of text]`. |
| `thinking_pi31b_he_sample.txt` | `results/pi-real/31b-iq3xxs/gen-he.stdout` (real Pi 5 16 GB, 31B UD-IQ3_XXS, 2026-10-05) | Hebrew prompt and answer. `<|channel>thought` follows `model` on the same line, and the answer is cut at the 48-token cap with no end marker. |

No captured sample has a **non-empty** thinking block yet. Tests of that case build the string inline.

## Real tool output from the Docker build image (desktop, not the Pi)

| File | Command | Notes |
|---|---|---|
| `membw_sample.json` | `bash pc/docker_run.sh --no-stdin -- tools/build-linux-x86/bin/membw --size-mb 256 --threads 1,2,4 --json` | Desktop DDR5 inside the Docker Desktop VM (kernel `5.15.167.4-microsoft-standard-WSL2`). The numbers are not the Pi's. |
| `fio_sample.json` | `bash pc/docker_run.sh --no-stdin -- fio --output-format=json tests/fixtures/fio_sample.fio` (the job file is checked in next to it) | fio-3.36, five jobs separated by `stonewall`, each running 2 s, `direct=1`, `ioengine=libaio` on a 256 MiB file in the container's `/tmp`: `seq_read_qd1` and `seq_read_qd8` (bs 1M), `rand_read_3mb_qd1` and `rand_read_3mb_qd4` (bs 3345k = one expert slab), `rand_read_16k_qd1` (bs 16k). These are the pi/03 metrics: read `jobs[i].read.bw_bytes`, `.iops` and `.clat_ns.mean`. Desktop SSD through the VM, not the Pi. |
| `fio_sample.fio` | the job file behind `fio_sample.json` | Job names are the io_mem.json metric names. The file name is `/tmp/pigemma-fio-test.bin`; `size=256m` and `runtime=2` are short on purpose. pi/03 uses its own `--size-mb` / `--seconds`. |

## Synthetic Pi facts (`facts_<name>.json`)

These were written by hand in the data_formats `facts.json` shape (TRACE_FORMAT section 14), with
the optional `notes` key recording what each file models. Values that are quoted from
`docs/design/FACTS.md` or `results/pi-real/` are stated in the notes. Every other value is
illustrative. Load them with `fake_facts(name)` (conftest) or `synth.load_facts(name)`.

| Name | Models | Verdict with the current catalog |
|---|---|---|
| `pi5-4gb-microsd` | the originally probed context-friend-1: 4 GB (MemTotal 4,039,000 kB), microSD, Hailo AI HAT, ContextFriend services running | exit 4 (protected services), every 26B row NOT_VIABLE |
| `pi5-8gb-usb3` | 8 GB + USB 3.0 SSD | 26b-q4 STREAM_OK, exit 0 |
| `pi5-8gb-nvme-gen3` | 8 GB + NVMe on M.2 HAT+, `pciex1_gen=3`, link 8.0 GT/s | 26b-q4 STREAM_OK, exit 0 |
| `pi5-16gb-nvme-gen3` | 16 GB (MemTotal 16,607,552 kB) + NVMe Gen3 | 26b-q4 FITS_MAPPED, exit 0 |
| `generic-linux-nodisk` | x86-64 container: storage `unknown`, 1.5 GB free, no systemd | NO_DISK rows, exit 3 |
| `pi5-16gb-microsd-real` | the real 16 GB board behind context-friend-1 since 2026-10-05 (measured RAM, disk, loadavg, services) | exit 4 unless `override` is `"PIGEMMA_I_KNOW"`, then 26b-q4 FITS_MAPPED, exit 0 |
| `pi5-8gb-nvme-busy` | as `pi5-8gb-nvme-gen3` with load average 3.6 | exit 4 (busy) |

## Not owned by tests/conftest.py

- `golden_26b_72tok.moetrace` (+ `.tokens.bin`, `.gen.txt`): the probe's real 26B smoke trace,
  written by the tools/probe/probe.cpp author and validated by tests/test_trace.py.
- `pi_proc_*` directories, if present, belong to pi/lib/common.sh (`PIGEMMA_FAKE_PROC`).
- Any other sub-directory (e.g. `bench/`) belongs to the module whose tests use it; see that test file.
