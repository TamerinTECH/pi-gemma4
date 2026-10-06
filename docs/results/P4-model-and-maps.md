# P4 — primary MoE file staged on the desktop + exact expert byte/page maps (2026-10-05)

Stage P4 of the decision workflow (brief sections 3 and 5: one MoE file, one storage device, exact layout).
Everything below is **MEASURED** (read from the files/headers/kernel) unless labelled PROJECTED or ASSUMED.

## Artifacts

| What | Where |
|---|---|
| Primary comparison file on the desktop | `models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf` (+ `SHA256SUMS`) |
| Same file in Docker volume `pigemma-models` | `/models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf` (mount `-v pigemma-models:/models`) |
| Map of that file (gguf_map map.json + `expert_map` section) | [results/maps/map26_ud_q3_k_m.json](../../results/maps/map26_ud_q3_k_m.json) |
| Map of the Pi's 5 split QAT Q4_0 parts | [results/maps/map26_qat_q4_0_split5.json](../../results/maps/map26_qat_q4_0_split5.json) |
| Provenance + verification records | [provenance_ud_q3_k_m.json](../../results/maps/provenance_ud_q3_k_m.json), [provenance_qat_q4_0_split5.json](../../results/maps/provenance_qat_q4_0_split5.json) |
| Logs | [fetch log](../../results/maps/fetch_26b_ud_q3km.log), [volume copy log](../../results/maps/volume_copy_26b_ud_q3km.log) |
| New code + tests | [pigemma/expert_map.py](../../pigemma/expert_map.py), [tests/test_expert_map.py](../../tests/test_expert_map.py) (31 tests incl. 1 integration, all pass) |
| Scripts | [pc/adhoc_fetch_26b_ud_q3km.sh](../../pc/adhoc_fetch_26b_ud_q3km.sh), [pc/adhoc_copy_26b_q3km_to_volume.sh](../../pc/adhoc_copy_26b_q3km_to_volume.sh) |

## The file (identity is the point of this stage)

`unsloth/gemma-4-26B-A4B-it-GGUF` @ `c099eb48e663fd284577b04978a94ffccb261841`, `gemma-4-26B-A4B-it-UD-Q3_K_M.gguf`,
12,728,497,888 B, sha256 `811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052`, **non-QAT**.
Three copies, all checked:

- desktop: full sha256 OK (download 153 s, 18:44 CEST); D: 14 GB free afterwards (gate ≥ 10 GB: pass);
- Docker volume `pigemma-models`: `sha256sum -c` OK inside the VM (18:48);
- Pi microSD (`/home/pi/pigemma/models/…`, the file behind the Pi resident 3.2 tok/s and the 0.55–0.79 tok/s
  balloon stress test): same size, same sha256 of the header and of two expert slices (full sha256 was checked
  by `pi/adhoc/fetch_gguf.sh` at download).

**"UD-Q3_K_M" contains no Q3_K tensor.** Real types: experts `gate_up` IQ3_XXS + `down` IQ4_NL in layers 0–28,
IQ4_XS + Q8_0 in layer 29; attention and the dense FFN are Q8_0; the tied embedding/head is Q6_K. Any
"3-bit vs Q4_0" comparison is a comparison of these mixed types and of a non-QAT vs a QAT checkpoint (brief §7).

## Docker VM facts that change the P5 plan

The volume is ext4 on the VM's `/dev/sdc`, **major:minor 8:32**, a whole disk (no partition; `read_ahead_kb`
128, scheduler none, 4 KiB pages). But:

- the Docker Desktop VM (kernel 5.15.167.4-microsoft-standard-WSL2) runs **cgroup v1 (hybrid)**, not v2:
  no `memory.high`, `memory.peak`, `memory.events`; v1 offers `memory.limit_in_bytes`,
  `max_usage_in_bytes`, `usage_in_bytes`, `stat`, `failcnt`, `memsw.*`, `swappiness`;
- **`CONFIG_BLK_DEV_THROTTLING` is not set**: there is no `io.max` and no `blkio.throttle.*`, under any cgroup
  version. `docker run --device-read-bps /dev/sdc:91mb` is silently discarded ("Your kernel does not support
  BPS Block I/O read limit"). Throttling the model reads to the microSD's 91.6 MB/s needs another mechanism
  (FUSE is compiled in; dm-delay is not);
- **`CONFIG_PSI` is not set** (no `/proc/pressure`);
- the VM has 8 GiB swap (`/dev/sdb`, 6.75 GB in use at 18:45 by other containers) and Docker's default
  memsw limit is 2× `--memory`: swap must be set explicitly (`--memory-swap` = `--memory`, `--memory-swappiness 0`);
- the VM disk image is `F:\Docker\DockerDesktopWSL\disk\docker_data.vhdx` (Docker setting `CustomWslDistroDir`),
  so the "C: holds the Docker VM disk" gate is really an F: gate (F: 732 GB free; C: 83 GB untouched).

## How the map is built (`pigemma.expert_map`, on top of `pigemma.gguf_map`)

For every MoE layer L and expert e: **all** tensors `blk.L.ffn_<x>_exps.<y>` whose last dim is 128
(`ffn_gate_up_exps.weight`, `ffn_down_exps.weight`, and the per-expert F32 vector `ffn_down_exps.scale`),
each sliced as `[offset + e·nb, offset + (e+1)·nb)` with its own type and size (no uniformity assumed).
Offsets are file-relative; pages are per file. For 4096 B (desktop VM) and 16384 B (Pi) pages every page's
owners are resolved exactly (headers, non-expert tensors, each expert), giving per expert:
`pages` (unique pages of the union of its slices), `exclusive`, `shared_expert`, `shared_nonexpert`.
`ExpertMap.pages_for(set, page)` returns the exact unique page count of any set of experts (boundary pages
counted once) and `ranges_for` the page-aligned merged ranges a pinner must lock. Unit tests compare every
number against a per-page brute force at page sizes 32…4096 B, including split files.

## Key numbers (MEASURED from the headers)

| | UD-Q3_K_M (desktop, VM, Pi microSD) | QAT Q4_0, 5 split parts (Pi USB) |
|---|---|---|
| expert slice per (layer, expert) | 2,632,964 B (L0–28), 4,212,740 B (L29) | 3,345,412 B (all layers) |
| all experts / non-expert tensors / headers | 10,312,793,088 / 2,399,879,288 / 15,824,672 B | 12,846,382,080 / 1,577,156,728 / 15,824,512 B |
| worst-case expert bytes per token (8 × 30) | 644,549,568 B | 802,898,880 B |
| pages per expert @4 KiB, mean (overhead vs nominal) | 658.6 (+0.45 %) | 819.7 (+0.36 %) |
| pages per expert @16 KiB, mean (overhead) | 166.9 = 2.73 MB (+1.82 %) | 207.2 = 3.39 MB (+1.46 %) |
| all experts, union @16 KiB | 629,503 pages = 10.314 GB (+0.009 %) | 784,141 pages = 12.847 GB |
| pages shared by ≥ 2 experts @16 KiB | 7,637 (≈ 255 per layer incl. the scale page shared by all 128) | 7,634 |
| expert/non-expert mixed pages | 120 | 121 |
| non-expert working set, union @16 KiB | 146,538 pages = 2.401 GB | 96,326 pages = 1.578 GB |

Example (UD, layer 0, expert 0, 4 KiB): 644 pages = 640 exclusive + 4 boundary pages (scale page shared with all
128 experts and `ffn_down.weight`; last `down` page shared with expert 1; first `gate_up` page shared with the
router; last `gate_up` page shared with expert 1). Pinning experts 0–2 of layer 0 at 16 KiB costs 484 unique
pages, 6 fewer than the three single-expert counts added up (163 + 163 + 164 = 490): the pin budget must use
`pages_for`, never nominal bytes.

**Split check (QAT):** the 5 part headers (streamed read-only from the Pi) hold the same 658 tensors with
identical names/types/shapes/sizes in the original order; six expert slices in four different parts
(incl. a 4-byte scale element) have the same sha256 on the Pi as in the desktop original. The parts'
`data_offset`s (15,792,320 / 9,728 / 9,568 / 9,568 / 3,328) are not page multiples, so their page boundaries
differ from the unsplit file's; use this map, not `results/map26.json`, for the USB-stick runs.

## Runtime repacking (static, from llama.cpp a7b94df `repack.cpp`, NOT measured)

With repack on (llama.cpp's default), eligible weights are copied out of the page cache into anonymous
`CPU_REPACK` memory: UD-Q3_K_M → **4.14 GB on x86** (all IQ4_NL `down` experts) and **6.16 GB + a 0.61 GB
tied-head duplicate on the Pi's A76** (Q8_0 attention/dense FFN, IQ4_NL + Q8_0 experts, Q6_K head). QAT Q4_0 →
13.77 GB on both. So every method in the A/B/C/D comparison must run with `-nr`, otherwise "Linux caching" is
partly anonymous memory. The real split must be read from each run's load log (CPU_Mapped / CPU_REPACK).

## Budget arithmetic for P5/M1 (PROJECTED; an upper bound, not a pin budget)

ASSUMED: ~1.24 GB anonymous memory per process (E0 measured 1,236 MB private peak for the 26B QAT, `-nr`,
`-c 4096`, Windows build: KV 380 MiB + compute 538 MiB + output 1 MiB; not yet measured for UD-Q3_K_M).
Memory left for expert pages (page cache or pinned) after the 2.40 GB non-expert set:
8 GiB → ≈ 4.95 GB ≈ 48 % of expert bytes (≈ 1,810 of 3,840 experts at 16 KiB, ≈ 1,840 at 4 KiB);
6 GiB → ≈ 2.81 GB ≈ 27 % (≈ 1,030 experts). Headroom for temporaries is not yet subtracted (brief §3).

## Reproduce

```bash
bash pc/adhoc_fetch_26b_ud_q3km.sh                 # download + full sha256 (token from HF_TOKEN_FILE, never printed)
bash pc/adhoc_copy_26b_q3km_to_volume.sh           # docker volume pigemma-models + sha256 inside the VM
python -m pigemma.expert_map build models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf --embed-base-map --page-sizes 4096,16384 --provenance results/maps/provenance_ud_q3_k_m.json --out results/maps/map26_ud_q3_k_m.json
python -m pigemma.expert_map pages --map results/maps/map26_ud_q3_k_m.json --experts 0:0,0:1,0:2 --page-size 16384
```

The split-part map command (header prefixes streamed with `head -c` from the Pi, `MSYS_NO_PATHCONV=1` so Git
Bash does not rewrite the `/mnt/usb/...` labels) is recorded in `provenance_qat_q4_0_split5.json`.

## Caveats

- Repack eligibility is source-derived; read the real CPU_REPACK size from each run's log.
- The per-step timings in the volume-copy log are missing (`bc` is not in the image; the script now uses awk).
- At ~18:54 CEST my read-only check of the Pi's UD file (19.4 MB: header + 2 slices) overlapped the start of
  another stage's `llama-server` on that file (pid 26856, `-nr -t 4 -c 4096`): up to 19.4 MB of that run's
  cold load may have been page-cache-warm. No other Pi I/O by this stage; nothing was written on the Pi
  (its `/` had exactly 2.0 GB free).

## Re-verification after the workflow resume (2026-10-05 22:25–22:30 CEST) — all MEASURED

The stage was replayed after the usage-limit interruption. Nothing was re-downloaded, re-copied or rebuilt; the
existing artifacts were checked again:

- desktop copy: `sha256sum -c SHA256SUMS` OK (34 s; size 12,728,497,888 B, mtime 18:44:40 unchanged);
- Docker volume `pigemma-models`: `sha256sum -c` OK inside the VM (28 s, read-only mount); still ext4 on
  `/dev/sdc`, **8:32**, whole disk, `read_ahead_kb` 128, scheduler none, 4 KiB pages, kernel
  5.15.167.4-microsoft-standard-WSL2, cgroup v1 (tmpfs root, no `io.max`, no `/proc/pressure`);
- `map26_ud_q3_k_m.json` against the file with an **independent parser** (gguf-py `GGUFReader`, not
  `pigemma.gguf_map`) and a per-page brute force over the whole file: all 3,840 (layer, expert) range lists equal
  (3 tensors each: `ffn_down_exps.scale`, `ffn_down_exps.weight`, `ffn_gate_up_exps.weight`); no other `*exps*`
  tensor exists; the only other tensors with a last dim of 128 are the 30 routers `ffn_gate_inp.weight`
  (non-expert, read for every token); expert / non-expert / header byte totals equal; no tensor overlap; the six
  page totals and all 3,840 per-expert (pages, exclusive, shared_expert, shared_nonexpert) tuples equal at
  4,096 B and 16,384 B — 0 mismatches;
- `map26_qat_q4_0_split5.json` offline against the desktop original's tensor table: packing the 658 tensors in
  the original order at alignment 32 into parts of 143 / 157 / 153 / 153 / 52 tensors from the recorded
  `data_offset`s reproduces all five part sizes exactly and all 90 expert-tensor placements; expert bytes equal
  (12,846,382,080). No Pi read was made this time (the byte-level Pi check of 18:5x stands);
- `tests/test_expert_map.py`: 31 passed (30 unit + 1 integration on the real file);
- gates: D: 13.10 GB = 12.2 GiB free (≥ 10: pass, but no room for another multi-GB file), C: 80.5 GB = 75 GiB,
  F: (Docker VM disk) 784.8 GB = 731 GiB. No other model job was running; `results/.cpu.lock` was held for the two hashes and released.

Two corrections made in this pass: the three-expert example above said "5 fewer than 3 × 163" (the union of
484 pages is right; the single counts are 163 + 163 + 164 = 490, so it is 6 fewer), and one provenance label in
the split map had been rewritten by Git Bash to `C:/Program Files/Git/mnt/usb/...` (now `/mnt/usb/...`; no
offset or count changed).
