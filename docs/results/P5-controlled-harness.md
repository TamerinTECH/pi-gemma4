# P5 — controlled-memory measurement harness (2026-10-05, finished 2026-10-06 00:25)

Stage P5 of the decision workflow (brief sections 2, 3, 4). Deliverable: a harness that runs **one cell =
one method × one memory budget × one workload set × one repeat** in a fresh memory cgroup, with every
helper inside the cgroup, an identical cache-state procedure for A/B/C/D, and full per-request and
per-phase accounting.

Status:
- The harness is complete and was run end-to-end in nine cells: methods A, B, C and D on the throttled
  FUSE storage, a two-cell `series`, and the Pi wrapper's own config in `direct` storage mode. All
  cells finished `ok`.
- Running method B found a real defect: pinner 1.1.0 exited immediately when its list was empty, so B
  was not "the same helper, nothing locked". Fixed in pinner 1.1.1 (section 8).
- **No A/B/C/D comparison was run in this stage.** The smoke cells are functional tests on a desktop that
  was not idle, with a synthetic pin ranking. Their numbers are **not** a baseline and **not** a result.
- The work was interrupted at 21:00 and resumed at 22:36. Sections marked *(resume)* were added then.

Labels: **MEASURED** (read from this stage's runs), **SOURCE** (read from code), **ASSUMED**, **PROJECTED**.

## 1. What was built

| File | Role |
|---|---|
| [pigemma/controlled.py](../../pigemma/controlled.py) | Host side (Windows). `cell` (exact `docker run` flags, bench lock, **CPU lock**, refuses while other model jobs run, disk gates, host CPU sampling), `series` (randomized blocks), `make-ranges` (page-exact C and D pin sets), `pin-budget`, `engines`, `engine-check`, `cpu-lock`, `docker-cmd`, `verify-throttle` |
| [pigemma/controlled_cell.py](../../pigemma/controlled_cell.py) | In-cgroup runner (stdlib; Python 3.12 in Docker, 3.13 on the Pi). Starts the FUSE throttle, pinner and llama-server inside the cgroup, runs the procedure of section 4, writes all outputs. Reads cgroup v1 **and** v2 |
| [pigemma/controlled_summary.py](../../pigemma/controlled_summary.py) | Paired analysis: median paired decode speed-up with a cluster-bootstrap CI (requests are the clusters), per category and per request; memory budget vs throughput; read bytes vs throughput; setup costs and amortized tok/s; exactness; background CPU next to the paired numbers; engine variants |
| [tools/fusethrottle/fusethrottle.c](../../tools/fusethrottle/fusethrottle.c) | Read-only FUSE passthrough of one file. Serial-device bandwidth model at 91.6 MB/s, O_DIRECT backing reads, a `.stats` counter file, an optional per-read trace |
| [tools/pinner/pinner.c](../../tools/pinner/pinner.c) **1.1.1** | `--allow-empty` (method B: same process, same whole-file mapping and reporting, nothing locked). 1.1.1 *(resume)*: the daemon now **holds** an empty set until SIGTERM, as it holds a real pin set |
| [docker/Dockerfile.fuse](../../docker/Dockerfile.fuse) | Image `pi-gemma4-build:fuse` = `pi-gemma4-build:latest` + fuse3, libfuse3-dev 3.14.0 |
| [pi/cgroup_cell.sh](../../pi/cgroup_cell.sh) | Pi wrapper: `systemd-run --scope` with MemoryMax / MemorySwapMax=0 / MemoryHigh=infinity / optional IOReadBandwidthMax. It **refuses while the memory controller is unavailable** |
| [tests/test_controlled.py](../../tests/test_controlled.py), [tests/test_controlled_summary.py](../../tests/test_controlled_summary.py) | 45 unit tests (26 + 19). They pass on Windows Python 3.14 (pytest) and inside the cell image on Python 3.12 |
| `results/controlled/_dev/` | Dev aids: `pi_wrapper_dryrun_test.sh`, `pi_config_direct_smoke.sh`, `pinner_hold_test.sh`, `cell_digest.py`, synthetic smoke pin plans |

## 2. Requested (brief) vs implemented on the desktop

The orchestrator decided on 2026-10-05 not to reconfigure WSL or Docker Desktop. Re-checked at 22:40:
the Docker VM (kernel 5.15.167.4-microsoft-standard-WSL2) runs **cgroup v1**, `CONFIG_BLK_DEV_THROTTLING`
and `CONFIG_PSI` are not set, and the blkio directory holds only `blkio.reset_stats`. Everything below is
labelled **"desktop Docker, cgroup v1 memory controller"**, a documented deviation from the cgroup v2
the brief asks for.

| Brief asks for | Implemented (MEASURED that it works) |
|---|---|
| dedicated cgroup v2, every helper inside, started before the model loads | A fresh container per cell. The runner is its first process; fusethrottle, pinner and llama-server are its children. Order: storage, evict, method setup, server |
| `memory.max` = budget | `--memory=<B>b` gives `memory.limit_in_bytes` = 8,589,934,592. The runner refuses unless limit == budget |
| explicit swap policy | `--memory-swap` = `--memory` (so `memory.memsw.limit_in_bytes` = budget) and `--memory-swappiness=0`. The runner refuses if memsw ≠ budget. Swap in/out counters are recorded per phase |
| `memory.high` (recorded) | No v1 equivalent. `soft_limit_in_bytes` (unlimited) is recorded; `high` is stored as `null` |
| `memory.current` / `peak` | `usage_in_bytes` / `max_usage_in_bytes`. v1 cannot reset the peak, so per-phase peaks are **sampled every 0.5 s** |
| `memory.stat` (anon, file, file_mapped, unevictable) | v1 `rss`, `cache`, `mapped_file`, `unevictable`, plus `pgmajfault`, `swap` and kmem usage |
| `memory.events` (max, oom, oom_kill) | `failcnt` + **`memsw.failcnt`** (with memsw = mem the memsw counter is the one that trips), `oom_control` |
| `memory.pressure` (PSI) | **Unavailable** (no PSI in this kernel). Reclaim evidence per phase *(resume)*: `cg_pgpgin` / `cg_pgpgout` = pages charged to / uncharged from the cgroup (v1 `memory.stat`), the limit-hit counter, and VM-wide `workingset_refault`. On cgroup v2 the same columns carry `pgscan`, `pgsteal`, `workingset_refault_file`, `memory.events high` and the PSI `some`/`full` stall seconds |
| `io.max` throttle to 91.6 MB/s | **Unavailable** (no blk-throttle; `--device-read-bps` is discarded). `fusethrottle` runs inside the cell. Verified in section 3 |
| cgroup `io.stat` rbytes | **Unavailable** on v1. One labelled figure per phase, `device_read_bytes` *(resume)*, from the most exact source present: the emulated device's counter (FUSE mode) > cgroup `io.stat` (v2) > `/proc/<pid>/io read_bytes` of server + pinner > `/proc/diskstats`. All raw sources stay in `io.csv` |
| `--cpus 4`, 4 physical cores | `--cpus=4.0 --cpuset-cpus=2,4,6,8`: one SMT sibling of guest cores 1-4. Hyper-V can still move vCPUs between host cores; a VM cannot pin to host cores |
| `IPC_LOCK`, memlock unlimited | `--cap-add IPC_LOCK --ulimit memlock=-1:-1`, plus `SYS_ADMIN --device /dev/fuse --security-opt apparmor:unconfined` for FUSE and `--network none` |
| temperature, throttling status | No sensor in the VM (recorded as `null`). On the Pi the runner samples `thermal_zone0` and records `vcgencmd get_throttled` at start and end *(resume)* |

Fixed llama-server flags, identical for every method (`SERVER_FLAGS` in the runner; SOURCE + MEASURED in
`config.json effective.server_argv`):
`-nr -t 4 -tb 4 -c 4096 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa auto -np 1 --cache-ram 0 --ctx-checkpoints 32 --seed 0 -lv 4 --no-webui --warmup`.

- Speculative flags (`-md`, `--spec-type`, `--spec-draft-n-max`, …) are rejected. The server log
  confirms "no implementations specified for speculative decoding" in every cell.
- No tracing callbacks: the cell runs the stock `llama-server` binary. Its sha256, size and `--version`
  text are recorded per cell (a7b94df: `5f60d6845a68…`, 15,679,072 B).
- `--cache-ram 0` is required. The default host prompt cache (8192 MiB) would be anonymous memory inside
  the budget and would carry KV state across `cache_prompt=false` requests.
- Requests go to **`/completion`** with `stream=true`, `return_tokens=true`, temperature 0, seed 0. The
  endpoint is recorded per request.
- Prompts are rendered with `pigemma.workloads.render_gemma4(..., bos=False)`; the server adds BOS.
  *(resume)* Before every request the runner asks the server's `/apply-template` (thinking disabled) for
  its own rendering and compares it with the prompt it is about to send. MEASURED: **28 of 28 prompts
  identical** across the seven cells run after this check was added. So `/completion` receives exactly
  what the GGUF's own chat template produces.

## 3. Storage-rate emulation verified (MEASURED, 2026-10-05 19:51; not repeated)

Command: `python -m pigemma.controlled verify-throttle --out results/controlled/throttle-verify-2026-10-05 --region-mib 512 --random-pages 2000`.
Container: 4 GiB, 2 CPUs, FUSE `max_readahead` 131072 (matching `read_ahead_kb` 128 on the VM's `sdc` and on
the Pi's `mmcblk0`). Results: [throttle_verify.json](../../results/controlled/throttle-verify-2026-10-05/throttle_verify.json).

| Case | Bytes served by the device | Rate | Page-cache growth / served | Backing file cached |
|---|---|---|---|---|
| ext4 direct mmap, sequential (reference, unthrottled, buffered) | 537 MB | 480 MB/s (app) | 1.00 | yes (537 MB: the second copy that FUSE avoids) |
| FUSE unthrottled, sequential `read()` | 537.0 MB | 450.9 MB/s | 1.000 | **0 B** |
| FUSE 91.6 MB/s, sequential `read()` | 536.9 MB | **91.57 MB/s** (0.9997×) | 1.000 | **0 B** |
| FUSE 91.6 MB/s, **mmap page faults**, sequential | 537.1 MB | **91.61 MB/s** (1.0001×) | 1.000 | **0 B** |
| FUSE 91.6 MB/s, mmap faults at 2000 random pages | 259.3 MB (1984 reads ≤ 128 KiB) | 79.2 MB/s (0.865×; one fault at a time) | 1.000 | **0 B** |

- **The rate limit holds for mmap page-cache reads.**
- The backing file is never cached: O_DIRECT is active, and cache growth equals the bytes served.
- Every random fault reads a ~128 KiB read-around window, so 2000 touched pages cost 259 MB.
- In the cells of this stage the model load ran at 87.8–89.1 MB/s effective
  (15.34 GB / 172.1 s, 10.01 GB / 113.8 s, 9.78 GB / 111.3 s).
- `sdc` device reads agree with the FUSE backing reads per phase (smoke 3, tun-01 prefill:
  2,817,404,928 B on both counters).

This is a bandwidth-only device model: no IOPS, latency or queue-depth model. The Pi's microSD delivered
~52 MB/s effective during random-ish decode in the balloon stress test (P1, RECOVERED), so this emulation
is likely **optimistic** for the miss path. `--op-latency-us` exists for a later calibration; it was not
tuned (0 µs). `--storage fuse-unthrottled` and `--storage direct` are separately labelled conditions; the
summarizer never pairs cells of different storage conditions.

## 4. Cell procedure (identical for all methods)

1. Preflight checks.
   - The cgroup limit equals the budget, swap is off, and MemAvailable ≥ budget + 2 GiB (Pi: + 3 GB).
   - Record caps, memlock, cpuset, page size, and the engine identity (binary sha256, `--version`).
2. **Storage up**: fusethrottle mounts `/mnt/pigemma-thr/<gguf>` (or, in `direct` mode, the file itself).
3. **Evict**. A targeted `posix_fadvise(DONTNEED)` on the backing and the served file (never a global
   drop_caches). `mincore` must then show ≤ 64 MiB cached, else the cell fails. MEASURED in all cells:
   0 B served, 0 B backing. This is also the file-page ownership audit: a page that another cgroup still
   maps cannot be dropped, and the cell then refuses instead of using it.
4. **Method setup** (timed; its reads are counted).
   - A: nothing.
   - B: `pinner --allow-empty` with an empty list.
   - C/D: `pinner --json --verify --pidfile` with the plan's ranges. The cell fails unless status is `ok`
     and locked == requested.
5. **Server load** (timed, reads counted). Effective buffers are parsed from the log (CPU_Mapped / CPU_REPACK / KV / compute).
6. **Fixed warm-up request**, the same for every method: "Count from one to five, separated by commas.",
   n_predict 16.
7. **Workload requests** in set order, from `data/workloads/<set>.jsonl`.
   - Per request: input tokens (new + reused `cache_n`), output tokens, raw text, token ids, per-event
     client timestamps, TTFT, prefill and decode tok/s (client and server), total time, ITL
     median/p90/p99/max, stalls (gaps > 3× the median ITL).
   - I/O and memory deltas are recorded separately for prefill (send → first token) and decode
     (first token → end).
   - Follow-ups *(resume)*: before the request the runner counts the tokens of the longest common text
     prefix with the previous prompt (server `/tokenize`) and afterwards compares it with
     `timings.cache_n`. Reuse is reported from this evidence, not assumed.
   - Background *(resume)*: CPU cores busy in the VM outside the cell during prefill and decode, and the
     request's start time so the Windows host CPU samples can be joined to it.
8. **Residency snapshots** (`mincore` of the served file: compressed bitmap plus pages per category:
   header / non-expert / experts by layer) cold, after setup, after load, after warm-up, after every
   request, and at the end. These feed simulator validation (A1).
9. **Lock accounting** *(resume)* and **teardown**. `pinner --verify-only` reports the mincore residency of
   the pinned ranges; the server and pinner receive SIGTERM; FUSE is unmounted.

Outputs: `results/controlled/<cell_id>/`
- `config.json`: request + `effective` (what really ran) + host CPU + status
- `requests.jsonl`; `outputs/<rid>.json` (text, token ids, timestamps, timings)
- `io.csv`: 46 columns per phase and per request (`IO_FIELDS`); `samples.csv` every 0.5 s
- `cgroup_start.json`, `cgroup_phases.json`, `cgroup_end.json`
- `server.log`, `pinner.log`, `fusethrottle.log`, `fuse_trace.csv` (every device read)
- `residency/`, `host_cpu.csv`, `runner.log`

## 5. Pin budget, C/D planning, lock accounting, exactness

- **`pin-budget`**. Expert pin bytes = budget − anon (measured under A) − kernel (measured) − non-expert
  pinned bytes (exact: unique 4 KiB pages from the map, 2,400,129,024 B) − page-cache reserve for misses −
  safety margin.
  - Defaults: the miss reserve is the worst-case expert bytes of one token (644,549,568 B, ASSUMED) and
    the safety margin is 5 % of the budget (ASSUMED).
  - Example from smoke 3 (MEASURED inputs): anon 0.585 GB + kernel 0.057 GB leave **4.47 GB for expert
    pins at 8 GiB**.
  - This is an illustration only: the smoke requests generated 16 tokens. M1 must measure anon with
    full-length requests.
  - Anonymous memory **grows during a conversation** (MEASURED, smoke 4): 0.562 GB after load, 0.661 GB
    after three short turns (129 tokens of context). The KV buffer is touched as the context fills, and
    the server keeps context checkpoints (5 created in those turns, 4.5–24.4 MiB each, up to 32 per
    slot; server.log). `pin-budget --from-cell` uses the peak over the whole cell, so the source cell
    must be an A cell that ran the full workload.
- **`make-ranges`**.
  - **C** takes a strict prefix of the calibration ranking, adding experts until their new unique pages
    exceed the expert pin budget. Boundary pages shared with the non-expert set or with other experts are
    counted once.
  - **D** draws uniformly random experts (seeded) until its total unique pages (non-expert included)
    match C's total.
  - Both write a pinner ranges file and a `plan.json`. The plan now also carries the nominal tensor bytes
    and the page-alignment overhead *(resume)*.
  - A ranking whose metadata names a set other than `calibration` is refused.
  - The smoke plans (synthetic ranking, `results/controlled/_dev/smoke-plans/`): C 558 experts,
    951,607 pages; D (seed 1) 560 experts, 951,503 pages: 104 pages = 0.4 MB fewer than C. Alignment
    overhead of the C plan at 4 KiB: 272,264 B.
- **Requested vs effective locked memory** *(resume)*, `effective.lock_accounting` of every cell. Four
  independent views must agree: the pinner's own summary, its `VmLck`, the cgroup `unevictable` counter
  and `mincore` of the ranges at the end.
  - MEASURED, smoke 4 (C): requested 3,897,782,272 B (951,607 unique pages), locked 3,897,782,272,
    VmLck 3,897,782,272, unevictable 3,897,782,272, mincore 3,897,782,272, 0 failed ranges.
  - MEASURED, smoke 5 (D): 3,897,356,288 B (951,503 pages) on all four views.
  - MEASURED, smoke 7 (C, final code): the same four views, and the plan's total equals what the pinner
    was asked to lock (`plan_total_eq_pinner_requested: true`).
  - The cgroup counter can trail by a few pages right after `mlock` (per-CPU LRU batches: 4–14 pages
    seen after setup, 0 at the end of every cell), so that one view is checked with a 1 MiB tolerance.
    VmLck and mincore are exact.
- **Setup is not free and differs between C and D** (MEASURED, n = 1 each).
  - C (61 contiguous segments): 43.4 s, 3.920 GB read. D (1035 scattered ranges): 49.2 s, 4.277 GB read.
  - D read 380 MB more than it locked: readahead around scattered ranges. Those extra pages are cached
    but not locked. A real calibration ranking will be scattered like D.
- **Exactness**. `controlled_summary exactness` compares token ids and output text per request against a
  reference cell that ran the same requests, and prints the first diverging index. It also compares
  prompt hashes, because follow-up prompts contain earlier replies.
  - MEASURED: the warm-up has **identical token ids** in every cell of this stage. Requests tun-01 and
    tun-02 have identical token ids and text in all eight cells that ran them: A three times on FUSE
    (19:57, 20:06, 23:01), B twice, C and D with 3.9 GB locked, and A in `direct` storage mode. The
    three turns of tun-07 (two follow-ups) are identical in the three cells that ran them (C, B, C).

## 6. Engines *(resume)* — brief UPDATE 2

`--engine ID` selects the llama-server build of a cell. `a7b94df` is built in. Every other id comes from
`results/comparisons/engines.json`, which the comparison stages X1a/X1b/X1c write.
`python -m pigemma.controlled engines --schema` prints the schema.

State of the real file at 00:15 (X1a + X1b; X1c's two ik engines still pending), as the harness reads it:

| engine id | runnable | parent | notes |
|---|---|---|---|
| `a7b94df` (built in) | yes | – | reference |
| `llama.cpp-a7b94df` | yes | – | the same binary under another name: resolved to the built-in engine, so its cells are plain variants A–D |
| `llama.cpp-upstream-8345f33` | yes | – | drift control; compared with reference A |
| `hotpin-parent` | yes | – | HotPin's own unmodified parent (a6647b1) |
| `hotpin` | yes | `hotpin-parent` | `env LLAMA_HOT_EXPERTS=…`; registered for method A only |

- The reader was fitted to that file after it appeared: nested `harness.extra_server_args` and
  `harness.server_flag_overrides`, `parent_engine_id`, `harness.methods` (a cell with a method the engine
  is not registered for is refused), and the alias of the reference binary. A unit test loads the real
  file and builds a command line for every runnable engine.
- MEASURED by X1b's own cell `results/controlled/20261006-x1b-smoke-tuning-hotpin-A-8g-r0` (run through
  this harness, 00:12–00:17, status ok): the profile's binary and environment reach the cell, the
  variant is `hotpin:A`, the engine identity is recorded, the a6647b1 build accepts every canonical
  flag, returns token ids and renders the same template (2 of 2). Its in-process locking is visible in
  the cell's marks: server VmLck 1,461,164 kB, cgroup unevictable 1,496,231,936 B. `lock_accounting`
  now also carries the server's VmLck for such engines (added after that cell).

- Keys read (aliases accepted): `engine_id`, `binary_path_in_container`, `revision`, `build_flags`,
  `notes`, `status`, `parent`, `server_flag_overrides`, `extra_args`, `env`, `api`, `docker_volumes`,
  `docker_image`. A list, `{"engines": [...]}` and `{id: profile}` are all accepted.
- A non-runnable status (`blocked`, `documentation-only`, …) or a missing `/work/...` binary is refused.
- `server_flag_overrides` respells or drops single canonical flags for a build that lacks them. Everything
  else stays identical, and the effective argv is recorded.
- `python -m pigemma.controlled engine-check --engine ID` runs the binary's `--version` and `--help` in
  the cell image (no model) and lists the canonical flags its help text does not mention.
  - MEASURED: `a7b94df` and X1a's `third_party/llama.cpp-upstream-8345f33` build both accept **all**
    canonical flags (reports in `results/controlled/_dev/engine-check-*.json`). No override is needed for
    upstream 8345f33.
  - SOURCE (static read of `third_party/ik_llama.cpp` at 3d27f5b, not run): it has no `-nr`, `-lv`,
    `--no-webui`, `--warmup` and no `return_tokens`. A starting profile for X1c to verify:
    `"server_flag_overrides": {"no_repack": null, "verbosity": null, "no_webui": null, "warmup_on": null}`.
- A cell of another engine is labelled `<engine>:<method>` (its **variant**). The summarizer compares
  every variant with the reference method A, and a variant whose engine names a `parent` also with the
  same method on that unmodified parent. Exactness uses the same engine's method A first, then the
  parent, then the reference engine (labelled cross-engine).
- An engine that returns no token ids is timed per content-bearing SSE event. Its ids are then
  re-tokenized from the output text and labelled as such; they are not the sampled ids.

This stage itself ran no cell of a non-reference engine; the evidence above is from X1b's cell.

## 7. Desktop CPU lock and series *(resume)*

- `cell` takes `results/.cpu.lock` for its duration and refuses (exit 4) while another owner holds it.
  `--cpu-lock-wait-s N` waits; `--cpu-lock-owner NAME` runs under a lock that a stage already holds;
  `--no-cpu-lock` is for functional tests and is recorded in `config.json`.
- `python -m pigemma.controlled cpu-lock status|acquire|release --owner NAME`. The file format is the one
  `pc/x1a/cpu_lock.sh` writes. A lock older than 6 h is reported and never removed automatically.
- `--max-bg-cpu-pct X` turns the idle probe into a gate; `--bg-wait-s S` keeps probing before refusing.
- `series` runs variants × repeats as **randomized complete blocks** (block = repeat, every variant once
  per block, order from a recorded seed) under one lock hold, and writes
  `results/controlled/series/<id>.json`. It stops at the first failed or refused cell unless
  `--keep-going`. `--dry-run` prints the plan.
- MEASURED (functional): `series --variants B,C --repeats 1 --seed 7` waited 720 s for the CPU lock,
  ran B then C, recorded start, end and exit code of both cells in
  `results/controlled/series/20261005-smoke7-series-BC.json`, and released the lock.

## 8. Smoke tests (MEASURED; functional only, not a baseline)

Configuration of every cell: UD-Q3_K_M (sha256 811882f6…), 8 GiB, swap off, 4 vCPUs (2,4,6,8), `-nr`,
flags as in section 2, `--n-predict-cap 16`. Smoke 3–6 ran under one CPU-lock hold (22:56–23:15); my own
light docker checks ran next to smoke 3. Smoke 1 and 2 (19:57 and 20:06, before the interruption) ran the
same command as smoke 3 with host CPU 23 % and 70 %; they are not repeated in the table. The series
(smoke 7) and the method-B rerun (smoke 8) follow the table.

| | smoke 3: A | smoke 4: C (synthetic plan) | smoke 5: D (synthetic plan) | smoke 6: A, `direct` storage, Pi wrapper config |
|---|---|---|---|---|
| requests | tun-01, tun-02 | tun-07 (3 requests, 2 follow-ups) | tun-01, tun-02 | tun-01, tun-02 |
| storage | FUSE 91.6 MB/s | FUSE 91.6 MB/s | FUSE 91.6 MB/s | ext4 volume, unthrottled |
| host CPU during the cell, mean / p95 | 21.5 % / 40.1 % | 20.9 % / 34.4 % | 27.8 % / 73.5 % | not recorded (manual `docker run`) |
| setup (pin + prefault) | 0 s | 43.4 s, 3.920 GB read | 49.2 s, 4.277 GB read | 0 s |
| locked bytes (4 views equal) | – | 3,897,782,272 | 3,897,356,288 | – |
| server load | 172.1 s, 15.34 GB | 113.8 s, 10.01 GB | 111.3 s, 9.78 GB | 38.4 s, 15.34 GB |
| cgroup after load: file / anon | 7.978 / 0.555 GB | 7.963 / 0.562 GB | 7.966 / 0.555 GB | 7.979 / 0.554 GB |
| resident after load: non-expert / experts | 2.400 / 5.567 GB | 2.400 / 5.553 GB | 2.400 / 5.555 GB | 2.400 / 5.569 GB |
| warm-up: TTFT / decode / reads | 37.2 s / 2.70 tok/s / 3.60 GB | 36.7 s / 1.84 / 3.76 GB | 31.7 s / 1.73 / 3.27 GB | 10.7 s / 5.27 / 3.37 GB |
| request 1: TTFT / decode / ITL median / decode reads per token | 32.3 s / 1.53 / 0.60 s / 47.9 MB | 26.1 s / 0.97 / 1.06 s / 82.3 MB | 31.5 s / 1.10 / 0.86 s / 64.7 MB | 8.9 s / 3.61 / 0.25 s / 48.8 MB |
| request 2 | 45.1 s / 1.70 / 0.60 s / 42.3 MB | 38.6 s / 1.36 / 0.81 s / 55.5 MB | 38.9 s / 0.85 / 0.79 s / 56.2 MB | 12.4 s / 3.63 / 0.29 s / 46.0 MB |
| request 3 | – | 40.2 s / 1.12 / 0.81 s / 69.7 MB | – | – |
| prefill reads per request | 2.82, 3.99 GB | 2.28, 3.37, 3.51 GB | 2.75, 3.39 GB | 2.61, 3.78 GB |
| KV reuse on follow-ups (`cache_n` vs expected) | – | 23 vs 24, 75 vs 76 | – | – |
| prompt = server template | 3 of 3 | 4 of 4 | 3 of 3 | 3 of 3 |
| token ids vs smoke 3 | reference | other requests | identical (2 of 2) | identical (2 of 2) |

Buffers in every cell (server log): CPU_Mapped 12,123.75 MiB, **no CPU_REPACK**, KV 380 MiB, compute
101.5 MiB. Peak cgroup usage (sampled): 8.590 GB = the limit, in every cell.

What the smoke cells show about the harness (MEASURED):
- **The pinner path works in a cell.** Locked == requested on four independent views (section 5). All
  2.40 GB of non-expert data stays resident under A as well, without any pinning.
- **KV reuse is real and one token short of the common prefix.** The server restores a context checkpoint
  and re-evaluates its last token (server.log: "restored context checkpoint … n_tokens = 24, n_past = 23").
  A `cache_prompt=false` request reuses nothing ("cached n_tokens = 0").
- **Device counters corroborate each other.** In `direct` mode the server's `/proc/<pid>/io read_bytes`
  and `/proc/diskstats sdc` agree within 0.004 % per phase (2,606,227,456 vs 2,606,137,344 B). In FUSE
  mode the FUSE counter and `sdc` are equal.
- **Faults are not bytes.** One prefill: server `majflt` 48,319, cgroup `pgmajfault` 5,952, device reads
  2.82 GB. Faults are reported only as fault counts.
- **Reclaim is visible without PSI.** In the same prefill the cgroup charged 707,299 pages and uncharged
  705,109: at the limit every page read evicts another.
- **The summarizer runs on real cells.** `results/controlled/_dev/smoke_summary_A3_D5.md` pairs smoke 5
  with smoke 3: median paired decode ratio 0.61 (0.72 and 0.50), host CPU 37 % vs 21 % over the requests.
  Two more real-cell summaries are described below (C vs B on five requests; A, B, D on two).

What must **not** be concluded from them:
- The D-vs-A ratio is n = 2 pairs from one cell each, with a synthetic plan and unequal host load. In
  D's second request the cell burned 35.8 CPU-seconds in 17.7 s (the others use about 1 core) and had a
  2.9 s stall. That is exactly the contamination the idle gate exists for.
- C used different requests than A; its numbers are not comparable at all.

### Smoke 7: functional series B, C on five requests (23:33–23:48)

`python -m pigemma.controlled series --variants B,C --repeats 1 --seed 7 --budget-gib 8 --workload tuning --conversations tun-01,tun-02,tun-07 --n-predict-cap 16 --ranges-c results/controlled/_dev/smoke-plans/smokeC2.ranges.txt --purpose smoke --series-id 20261005-smoke7-series-BC --cpu-lock-wait-s 2700`

| | B (pinner 1.1.0, see below) | C (synthetic plan, 3.898 GB locked) |
|---|---|---|
| host CPU during the cell, mean / p95 | 32.6 % / 51.2 % | 27.6 % / 39.6 % |
| setup | 0.02 s, 0 GB | 43.4 s, 3.920 GB |
| load | 173.7 s, 15.43 GB | 113.7 s, 9.99 GB |
| decode tok/s, 5 requests | 1.30, 1.47, 1.25, 1.84, 1.25 | 1.15, 1.38, 1.05, 1.48, 1.03 |
| decode reads per token, MB | 57.4, 47.7, 59.5, 36.3, 59.1 | 67.3, 54.2, 74.3, 49.5, 75.8 |
| KV reuse on the 2 follow-ups | 23 of 24, 75 of 76 (2 checkpoint restores) | the same |
| prompt = server template | 6 of 6 | 6 of 6 |
| `io.csv` | 22 rows, 0 malformed | 22 rows, 0 malformed |

- Summarizer on these two cells (`results/controlled/_dev/smoke_summary_series7_C_vs_B.md`, baseline B):
  median paired decode ratio C/B 0.84, CI [0.80, 0.94], all 5 pairs below 1; per category chat 0.91,
  topic_switch 0.82; token ids identical in 5 of 5.
- This is one cell per method with a pin set that was **not** chosen from traces (experts in index
  order). It says nothing about calibration-ranked pinning. It does show the mechanism M1 must expect:
  locked pages are taken out of the kernel's evictable pool, so a pin set that is not better than the
  kernel's own choice costs decode speed and read volume. The same holds for the random control D.

### The method-B defect and smoke 8

- In smoke 7 the B pinner printed its summary and exited ("nothing was locked; not holding"). The cell
  shows it: no `VmLck` for the pinner at any mark, no pinner exit code at teardown. B was therefore
  method A plus a 25 ms process launch.
- The earlier component test had used `--once`, where exiting is correct, so the daemon case was never
  exercised before a cell ran it.
- Fix: pinner 1.1.1 holds an empty set accepted with `--allow-empty` until SIGTERM. Component test
  `results/controlled/_dev/pinner_hold_test.sh` (FUSE file, 2 GiB container): the B pinner is alive and
  sleeping 2 s after its summary, holds the whole-file mapping, VmLck 0 kB, exits 0 on SIGTERM; a 64 MiB
  pin still gives requested = locked = resident = VmLck. 15 checks pass.
- **Smoke 8** (00:04–00:10, `cell --method B --conversations tun-01,tun-02 --n-predict-cap 16`, pinner
  1.1.1, after waiting 840 s for the CPU lock): the pinner is alive at every mark of the cell (VmLck 0,
  RSS 1.8 MB), receives SIGTERM at teardown and exits 0. Load 173.1 s / 15.34 GB; decode 1.48 and
  1.63 tok/s; 49.3 and 43.9 MB read per token; host CPU 31.7 % mean. Token ids identical to smoke 3 (A).
- Summarizer on smoke 3 (A), 8 (B) and 5 (D), `results/controlled/_dev/smoke_summary_A3_B8_D5.md`:
  B/A median paired decode ratio 0.96 (0.958 and 0.966), D/A 0.61. Two pairs each, host CPU 32 % and
  28 % against 21 % for A: not evidence of an overhead, and not a result.

Two observations for M1 to test properly (MEASURED here, n small):
- At 8 GiB and 91.6 MB/s the emulated device is busy for 95–97 % of prefill wall time and 78–87 % of
  decode wall time (device bytes / rate / wall, `io.csv`; 52 % in the one D request with the CPU
  episode). The same two requests on the unthrottled volume decoded 2.1–2.4× faster while reading about
  the same amount per token (46–49 MB vs 42–48 MB).
- Even a 28–39-token prefill reads 2.3–4.0 GB: a few dozen tokens route to most of the 3,840 experts.

## 9. Pi readiness (refusal re-verified read-only on the Pi, 2026-10-05 22:57)

Nothing was written on the Pi (no `results/controlled` directory afterwards). Raw output:
`results/controlled/pi-readiness-2026-10-05/`.

- `ssh … pi@PI_HOST 'bash -s -- --check' < pi/cgroup_cell.sh` → **exit 4** with
  `"cgroup_v2":1,"memory_controller":0,"cmdline_cgroup_disable_memory":1,"psi":0,"io_controller":1,"systemd_run":1,"ready":false`,
  all four ContextFriend services `active`, MemAvailable 16.2 GB, `/` free 2,078,724,096 B.
- A run request (`--method A --budget-gib 8 --workload tuning`) → the same refusal, exit 4.
- The refusal text names the three options (cmdline.txt + reboot, KVM guest, desktop emulation) and says
  a balloon is not a substitute.

Changes *(resume)*, found by testing the run path off the Pi:
- `results/controlled/_dev/pi_wrapper_dryrun_test.sh` runs the wrapper in the cell image: refusal on the
  VM's kernel, then `--dry-run` against a fake cgroup-v2 tree. 18 checks pass. The fake tree is honoured
  **only together with `--dry-run`**, which starts nothing; verified on the Pi too (exit 4 without it).
- Bugs fixed: `--io-read-bps` aborted when the device had no parent; the unit name ended in a dash.
- The Pi lock is now `~/pigemma/.bench.lock` (the lock every stage uses), and the wrapper refuses while
  another `llama-server` / `llama-completion` / `llama-cli` / `powerinfer` process runs.
- `--engine-id`, `--engine-parent`, `--server-bin`, `--server-extra` select another build.
- Methods B/C/D need `--allow-pinner-on-protected-host`. The pinner's own guard refuses to mlock on a
  host that runs ContextFriend services, and that flag needs the operator's explicit approval. Without it the
  cell fails at setup and nothing is locked.
- Smoke 6 executed the wrapper's generated config through the real runner (desktop, cgroup v1, `direct`
  storage). The config, the Pi safety thresholds and the `direct` path therefore work; `systemd-run`
  itself and cgroup v2 readings remain untested.

Once the operator enables the controller, the wrapper checks MemAvailable ≥ budget + 3 GB, `/` ≥ 2 GB free and
service health, takes the lock, and runs
`sudo systemd-run --scope -p MemoryMax=B -p MemorySwapMax=0 -p MemoryHigh=infinity … nice -n 10 python3 -m pigemma.controlled_cell run`.
A watchdog stops only its own scope if a service turns unhealthy; the runner stops its own server if
MemAvailable < 3 GB. It needs the repo's `pigemma/` + `data/workloads` + `results/maps` under
`~/pigemma/code`, a Pi build of the pinner, and pin plans made with `--page-size 16384`. None of this is
deployed. The Pi's `/` is 78 MB above the 2 GB gate, so nothing can be copied there before space is freed.

## 10. Reproduce

```bash
# build (once)
docker build -t pi-gemma4-build:fuse -f docker/Dockerfile.fuse docker
MSYS_NO_PATHCONV=1 docker run --rm -v "REPO_ROOT:/work" pi-gemma4-build:fuse bash -c \
  'cd /work && gcc -std=c11 -D_GNU_SOURCE -Wall -Wextra -Werror -O2 -o tools/build-linux-x86/bin/fusethrottle tools/fusethrottle/fusethrottle.c $(pkg-config --cflags --libs fuse3) -lpthread && gcc -std=c11 -D_GNU_SOURCE -Wall -Wextra -Werror -O2 -o tools/build-linux-x86/bin/pinner tools/pinner/pinner.c'

# the smoke cells of this stage
python -m pigemma.controlled cpu-lock acquire --owner P5-smoke --purpose "functional smoke" --wait-s 600
python -m pigemma.controlled cell --method A --budget-gib 8 --workload tuning --conversations tun-01,tun-02 --n-predict-cap 16 --purpose smoke --cpu-lock-owner P5-smoke
python -m pigemma.controlled make-ranges --method C --ranking results/controlled/_dev/smoke-plans/synthetic_ranking.json --expert-pin-bytes 1500000000 --out results/controlled/_dev/smoke-plans/smokeC
python -m pigemma.controlled make-ranges --method D --match results/controlled/_dev/smoke-plans/smokeC.plan.json --seed 1 --out results/controlled/_dev/smoke-plans/smokeD
python -m pigemma.controlled cell --method C --ranges results/controlled/_dev/smoke-plans/smokeC.ranges.txt --budget-gib 8 --workload tuning --conversations tun-07 --n-predict-cap 16 --purpose smoke --cpu-lock-owner P5-smoke
python -m pigemma.controlled cell --method D --ranges results/controlled/_dev/smoke-plans/smokeD.ranges.txt --budget-gib 8 --workload tuning --conversations tun-01,tun-02 --n-predict-cap 16 --purpose smoke --cpu-lock-owner P5-smoke
python -m pigemma.controlled cpu-lock release --owner P5-smoke
python -m pigemma.controlled_summary summarize --include-smoke --cells <A cell>,<D cell> --out s.json --md s.md
python results/controlled/_dev/cell_digest.py results/controlled/<cell>

# a measurement (M1/M2)
python -m pigemma.controlled pin-budget --budget-gib 8 --from-cell results/controlled/<A cell>
python -m pigemma.controlled make-ranges --method C --ranking results/rankings/<calibration ranking>.json --expert-pin-bytes N --out results/controlled/plans/C-8g
python -m pigemma.controlled make-ranges --method D --match results/controlled/plans/C-8g.plan.json --seed 1 --out results/controlled/plans/D-8g
python -m pigemma.controlled series --variants A,B,C,D --repeats 3 --seed <fixed in advance> --budget-gib 8 --workload heldout \
    --ranges-c results/controlled/plans/C-8g.ranges.txt --ranges-d results/controlled/plans/D-8g.ranges.txt \
    --max-bg-cpu-pct 10 --bg-wait-s 600 --cpu-lock-wait-s 7200
python -m pigemma.controlled_summary summarize --out results/controlled/summary.json --md results/controlled/summary.md
python -m pigemma.controlled_summary exactness

# engines
python -m pigemma.controlled engines            # what can run; --schema prints the file format
python -m pigemma.controlled engine-check --engine <id>
python -m pigemma.controlled cell --engine <id> --method A --budget-gib 8 --workload tuning

# tests
python -m pytest tests/test_controlled.py tests/test_controlled_summary.py -q     # 45 passed
MSYS_NO_PATHCONV=1 docker run --rm --network none --cpus=1 -v pigemma-models:/models:ro -v "REPO_ROOT:/work" -w /work pi-gemma4-build:fuse bash results/controlled/_dev/pi_wrapper_dryrun_test.sh
MSYS_NO_PATHCONV=1 docker run --rm --network none --cpus=2 --memory=2g --memory-swap=2g --cap-add IPC_LOCK --ulimit memlock=-1:-1 --cap-add SYS_ADMIN --device /dev/fuse --security-opt apparmor:unconfined -v pigemma-models:/models:ro -v "REPO_ROOT:/work" -w /work pi-gemma4-build:fuse bash results/controlled/_dev/pinner_hold_test.sh
ssh -o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1 pi@PI_HOST 'bash -s -- --check' < pi/cgroup_cell.sh   # exit 4
```

Cell duration, PROJECTED from the smoke rates (not measured): load about 3 min, each short prompt 30–45 s
of prefill, decode about 1.5 tok/s under A at 8 GiB. A full tuning cell (11 requests, 704 tokens bound) is
about 20 min; a full held-out cell (26 requests, 1488 tokens bound) about 50–55 min; 12 held-out cells about
10–11 h. M1 should measure this before M2 is planned.

## 11. Limitations and open items

- **Platform**: x86 Zen 4 compute (AVX-512) with an emulated device, not an A76 Pi.
  - Compute-bound phases are faster than on the Pi, so device-bound shares look larger.
  - The device model covers bandwidth only.
  - This is cgroup v1 (no memory.high, no PSI, no io.stat) inside a Hyper-V VM whose vCPUs float over
    host cores.
- **FUSE is not the block layer**. Request sizes differ (4–256 KiB reads, most of them 256 KiB), there is
  a user-space round-trip per read, and the FUSE daemon's CPU shares the cell's 4 vCPUs. Its memory
  (6–9 MB RSS) is inside the budget.
- **Load reads more than the file under A**. MAP_POPULATE reads all 12.73 GB, and llama's all-expert
  warm-up re-reads what was evicted: 15.3 GB in about 172 s. C/D read 9.8–10.0 GB at load after 3.9–4.3 GB
  at setup. `--server-warmup off` exists as a recorded option, but it is not the stock behaviour.
- **Peak memory per phase is sampled every 0.5 s**. The v1 `max_usage_in_bytes` covers the whole
  container life and is recorded as well.
- **Not exercised in a cell**: a budget other than 8 GiB, a series with more than one repeat or with a
  non-reference engine, the `--max-bg-cpu-pct` gate, a failing cell inside a series, an engine that needs
  flag overrides or returns no token ids (ik_llama.cpp), and the summarizer's own-parent comparison on
  real cells (unit-tested with synthetic cells only).
- **Defects in the earlier smoke cells' files** (fixed in the code, visible in the data):
  - In the `io.csv` of smoke 3–6 the last column (`device_read_source`) is split at a comma, and the
    `load` row of smoke 6 names `/proc/diskstats` as its source. `requests.jsonl` is unaffected.
    Smoke 7 and 8 have valid CSV.
  - The reuse verdict in smoke 4 reads "partial reuse" for what is the complete reuse (expected − 1).
  - The B cell of smoke 7 ran with pinner 1.1.0 and is in effect an A cell.
  - C/D cells before smoke 7 carry `alignment_overhead_bytes` in `lock_accounting`; that field compared
    the wrong quantities and was replaced by the plan's own figure.
- **Smoke timings are invalid as measurements**: host CPU 21–33 % mean with peaks to 73 %.
- **`vm_other` CPU** counts everything in the VM outside the cell's cgroup, including kernel work done
  on the cell's behalf. It was about 0.3 cores during prefill in every cell; treat it as an upper bound
  on foreign load.
- **Pi**: `systemd-run`, cgroup v2 readings and the 16 KiB page path are untested. Nothing is deployed.
