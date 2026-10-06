# pigemma-fast — patched copy of llama.cpp for Gemma 4 on the Raspberry Pi 5

Base: `ggml-org/llama.cpp` at `a7b94df2c616bc1f62a73b964b4a71cb0dcc488e` (see `BASE_REVISION`).
The reference tree and its binaries are never modified; the patches go into a copy.

## What the patches do

All three are OFF unless the environment variable is set when the binary starts.

| Variable | Effect | Scope |
|---|---|---|
| `PIGEMMA_STRIP_PREFILL=1` | Gemma 4 models with shared-KV layers (E2B, E4B): a token without the output flag is not sent through the layers that reuse the KV cache of earlier layers | exact up to batch-split rounding; see `docs/fast/LAYER_SKIP.md` for the dependency rule and supported modes |
| `PIGEMMA_PAD_TAIL=1` | repacked matmul: 2 or 3 token rows beyond a multiple of 4 run in one zero-padded 4-row sweep instead of one sweep per row | arithmetic of each row unchanged, but a tail row is now computed by the 4-row kernel |
| `PIGEMMA_CHUNK_KB=64` | repacked matmul: weight chunks of at most 64 KiB when more than 4 token rows are evaluated | scheduling only |

## Files

- `pigemma-fast-v4.patch` — the complete change as one unified diff (`patch -p1` in a clean checkout of the base).
- `apply_patches.py` — the same change as checked string replacements; `--upgrade` brings an older patched tree
  up to date in place.
- `build_fast.sh` — Pi: copy the reference tree, patch, build `llama-server`, `llama-bench`, `llama-completion`.
- `run_fast_bench.sh`, `run_fast_bench2.sh`, `run_fast_qa44.sh` — the measurements of 2026-10-06.
- `../adhoc/mtp_ab.py` — request driver (stores full texts), `../adhoc/par_throughput.py`, `../adhoc/sampler_ab.py`.

## Reproduce on the Pi

```sh
# once per boot: the stick is not mounted automatically
sudo mount -o rw,uid=1000,gid=1000,umask=022,noatime /dev/sda1 /mnt/usb
# build (reference tree ~/pigemma/llama.cpp must be at the base revision, unmodified)
~/pigemma/fast/build_fast.sh            # about 7 minutes; binaries -> /mnt/usb/pigemma/fast/bin
# fast profile
D=/mnt/usb/pigemma/models/e2b-qat-q4_0
PIGEMMA_STRIP_PREFILL=1 PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=64 \
  /mnt/usb/pigemma/fast/bin/llama-server -m $D/gemma-4-E2B_q4_0-it.gguf \
  -md $D/mtp-gemma-4-E2B-it.gguf --spec-type draft-mtp -t 4 -c 4096 --reasoning off \
  --host 127.0.0.1 --port 18400      # ports 18190/18191 are used by another local service
```

## Reproduce elsewhere

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a7b94df2c616bc1f62a73b964b4a71cb0dcc488e
patch -p1 < /path/to/pigemma-fast-v4.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build -j 4 --target llama-server llama-bench
PIGEMMA_STRIP_PREFILL=1 build/bin/llama-bench -m gemma-4-E2B_q4_0-it.gguf -t 4 -p 512 -n 64
```

Models (not in this repository): `google/gemma-4-E2B-it-qat-q4_0-gguf` (`gemma-4-E2B_q4_0-it.gguf`,
3,349,516,256 bytes, sha256 fa401b55…) and the drafter `mtp-gemma-4-E2B-it.gguf` (59,235,648 bytes,
sha256 586f2460…) from `unsloth/gemma-4-E2B-it-qat-GGUF`.

## Results and their checksums

Raw results of 2026-10-06: `results/pi-real/{fast,verified-ceiling,e2b-profile,speed-ceiling}/`, checksums in
`results/pi-real/SHA256SUMS-2026-10-06.txt`. The same files exist on the Pi under `~/pigemma/results/`.
Summary tables: `docs/cf-engine/STATUS.md` (copied into the top of `CLAUDE.md`).
