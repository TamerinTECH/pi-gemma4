# pigemma-fast — patched copy of llama.cpp for Gemma 4 on the Raspberry Pi 5

Base: `ggml-org/llama.cpp` at `a7b94df2c616bc1f62a73b964b4a71cb0dcc488e` (see `BASE_REVISION`).
The reference tree and its binaries are never modified; the patches go into a copy.

## What the patches do

Every change is OFF unless its environment variable is set when the binary starts. With no variable set the
patched build is bit-identical to the reference build (140 of 140 logit vectors, 32 of 32 KV-state digests).

| Variable | Effect | Result is |
|---|---|---|
| `PIGEMMA_STRIP_PREFILL=1` | Gemma 4 models with shared-KV layers (E2B, E4B): a token without the output flag is not sent through the layers that reuse the KV cache of earlier layers | same persistent state bit for bit; logits equal up to the reference's own batch-split rounding. Rule, guards and tests: `docs/fast/LAYER_SKIP.md` |
| `PIGEMMA_PAD_TAIL=1` | repacked matmul: 2 or 3 token rows beyond a multiple of 4 run in one zero-padded 4-row sweep instead of one sweep per row | each row now computed by the 4-row kernel (logit differences up to 6e-6) |
| `PIGEMMA_CHUNK_KB=64` | repacked matmul: weight chunks of at most 64 KiB when more than 4 token rows are evaluated | scheduling only, bit-identical |
| `PIGEMMA_MTP_CENTROIDS=1` | Gemma 4 assistant (drafter): its vocabulary head scores 2048 cluster centroids, keeps the best 32 clusters and computes logits for their 4096 tokens only (`=n` with n > 1 keeps n clusters). Needs a drafter file that contains the two cluster tensors (`make_centroid_gguf.py`) | proposals only; the target model verifies every token exactly as before |
| `PIGEMMA_SPEC_STATS=1` | prints draft calls, drafted tokens and draft time per request to the server log | diagnostics |

## Files

- `pigemma-fast-v7.patch` — the complete change as one unified diff (`patch -p1` in a clean checkout of the base).
- `apply_patches.py` — the same change as checked string replacements; `--upgrade` brings an older patched tree
  up to date in place (the marker file `PIGEMMA_FAST_PATCHED` lists the applied versions v1 … v7).
- `mount_work.sh` — Pi, once per boot: mounts the stick and the ext4 work image `/mnt/pigemma-fast`
  (source copies and build trees live there, no longer in `/tmp`).
- `build_fast.sh [tree] [bin-dir]` — Pi: fresh copy of the pinned source (`git archive`) + patch file, or an
  in-place upgrade of an existing patched tree; builds `llama-server`, `llama-bench`, `llama-completion`.
- `tests/` — `pigemma-fastcheck` (nine API-level scenarios; full logits, KV-state digests, token sequences) and
  `run_fastcheck.sh`, which builds it in the patched tree and in an unpatched reference copy and compares.
- `make_centroid_gguf.py` — adds the cluster tensors from Google's public assistant checkpoint to a drafter GGUF.
- `run_phase_a.sh` (v7 build + correctness), `run_phase_b.sh` (matched four configurations, ablation, drafter
  head, model screen), `analyze_phase_b.py`, `prompts_*.json` — the measurements of brief 3.
- `run_fast_bench.sh`, `run_fast_bench2.sh`, `run_fast_qa44.sh` — the earlier measurements of 2026-10-06.
- `upstream/` — the layer skip and the tail fix as two separate changes for current llama.cpp master, with notes.
- `../adhoc/mtp_ab.py` — request driver (streaming, stores full texts, time to first visible text, peak RSS).

## Reproduce on the Pi

```sh
~/pigemma/fast/mount_work.sh                         # once per boot
# fresh build from the pinned source + patch file (reference tree ~/pigemma/llama.cpp stays untouched)
~/pigemma/fast/build_fast.sh llama-fast-v7 bin-v7    # about 7 minutes; binaries -> /mnt/usb/pigemma/fast/bin-v7
# correctness scenarios (about 7 minutes for E2B)
~/pigemma/fast/tests/run_fastcheck.sh llama-fast-v7 /mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf e2b
# fast profile
D=/mnt/usb/pigemma/models/e2b-qat-q4_0
PIGEMMA_STRIP_PREFILL=1 PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=64 \
  /mnt/usb/pigemma/fast/bin-v7/llama-server -m $D/gemma-4-E2B_q4_0-it.gguf \
  -md $D/mtp-gemma-4-E2B-it.gguf --spec-type draft-mtp -t 4 -c 4096 --reasoning off \
  --host 127.0.0.1 --port 18400      # ports 18190/18191 are used by another local service
```

## Reproduce elsewhere

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a7b94df2c616bc1f62a73b964b4a71cb0dcc488e
patch -p1 < /path/to/pigemma-fast-v7.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build -j 4 --target llama-server llama-bench
build/bin/llama-bench -m gemma-4-E2B_q4_0-it.gguf -t 4 -p 512 -n 64                           # reference behaviour
PIGEMMA_STRIP_PREFILL=1 build/bin/llama-bench -m gemma-4-E2B_q4_0-it.gguf -t 4 -p 512 -n 64   # layer skip
```

Models (not in this repository; sizes and sha256 prefixes of the files used):

| File | Bytes | sha256 | Source |
|---|---|---|---|
| `gemma-4-E2B_q4_0-it.gguf` | 3,349,516,256 | fa401b55… | `google/gemma-4-E2B-it-qat-q4_0-gguf` |
| `mtp-gemma-4-E2B-it.gguf` | 59,235,648 | 586f2460… | `unsloth/gemma-4-E2B-it-qat-GGUF` |
| `mtp-gemma-4-E2B-it-centroids.gguf` | 62,381,632 | c0e39e5c… | the file above + `make_centroid_gguf.py` (tensors from `google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant`) |
| `gemma-4-E4B_q4_0-it.gguf` | 5,154,941,280 | 676c3507… | `google/gemma-4-E4B-it-qat-q4_0-gguf` (split in two parts on the stick) |
| `mtp-gemma-4-E4B-it.gguf` | 59,678,016 | 423074e5… | `unsloth/gemma-4-E4B-it-qat-GGUF` |
| `mtp-gemma-4-E4B-it-centroids.gguf` | 62,824,000 | bbdce8df… | as for E2B, from the E4B assistant checkpoint |

## Results and their checksums

Raw results: `results/pi-real/{fast,fast3,fastcheck-e2b,fastcheck-e2b-v7,fastcheck-e4b-v7,verified-ceiling,e2b-profile,speed-ceiling}/`,
checksums in `results/pi-real/SHA256SUMS-2026-10-06.txt`. The same files exist on the Pi under
`~/pigemma/results/`. Summary tables: `docs/cf-engine/STATUS.md` (copied into the top of `CLAUDE.md`), decision
report: `docs/fast/DECISION_2026-10-06.md`.
