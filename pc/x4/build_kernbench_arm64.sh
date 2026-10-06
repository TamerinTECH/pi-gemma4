#!/usr/bin/env bash
# X4: build tools/kernbench for aarch64 (Debian 13, same OS as the Pi) inside the emulated-Pi image,
# linking the static ggml libs of the llama.cpp a7b94df aarch64 build in Docker volume pigemma-arm64-llama
# (built with -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16: the same NEON+dotprod kernels the
# Pi's native build selects; only the -mcpu=cortex-a76 tuning of the native build is missing).
# Output: tools/build-pi-arm64/kernbench (+ BUILD_INFO.txt). One C++ file under qemu-user: ~1-3 min, 1 core.
# Usage (Git Bash on the desktop): pc/x4/build_kernbench_arm64.sh [--test]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WIN_REPO="$(cd "$REPO" && pwd -W 2>/dev/null || echo "$REPO")"
TEST="${1:-}"
MSYS_NO_PATHCONV=1 docker run --rm --platform linux/arm64 \
  -v pigemma-arm64-llama:/opt/llama:ro -v "$WIN_REPO:/work" pi-gemma4-pi-emu:trixie bash -c '
set -euo pipefail
OUT=/work/tools/build-pi-arm64
mkdir -p "$OUT"
L=/opt/llama/build/ggml/src
FLAGS="-O2 -std=c++17 -fopenmp -march=armv8.2-a+dotprod+fp16 -Wall -Wextra"
LIBS="-Wl,--start-group $L/libggml.a $L/libggml-cpu.a $L/libggml-base.a -Wl,--end-group -lpthread -lm -ldl"
t0=$(date +%s)
if g++ $FLAGS -static -I/opt/llama/src/ggml/include /work/tools/kernbench/kernbench.cpp -o "$OUT/kernbench" $LIBS 2> "$OUT/kernbench_build.log"; then
  LINK=static
else
  echo "static link failed, retrying dynamic" >> "$OUT/kernbench_build.log"
  g++ $FLAGS -I/opt/llama/src/ggml/include /work/tools/kernbench/kernbench.cpp -o "$OUT/kernbench" $LIBS 2>> "$OUT/kernbench_build.log"
  LINK=dynamic
fi
t1=$(date +%s)
{
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ) in $((t1 - t0)) s (qemu-user aarch64, image pi-gemma4-pi-emu:trixie)"
  echo "compiler: $(g++ --version | head -1)"
  echo "flags: $FLAGS ; link: $LINK"
  echo "ggml libs: $L (llama.cpp $(cd /opt/llama/src && git rev-parse HEAD 2>/dev/null || echo a7b94df), GGML_NATIVE=OFF, GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16, GGML_OPENMP=ON)"
  echo "file: $(file "$OUT/kernbench" 2>/dev/null || true)"
  echo "sha256: $(sha256sum "$OUT/kernbench")"
  echo "source sha256: $(sha256sum /work/tools/kernbench/kernbench.cpp)"
} > "$OUT/BUILD_INFO.txt"
cat "$OUT/BUILD_INFO.txt"
cat "$OUT/kernbench_build.log" | tail -20
if [ "'"$TEST"'" = "--test" ]; then
  echo "--- functional test under qemu-user (tiny sizes; speeds are NOT representative)"
  "$OUT/kernbench" --types Q8_0,Q6_K,IQ3_XXS,IQ4_NL,IQ4_NL@704,IQ4_XS,Q4_0 --threads 1,2 --stream-mb 4 --cache-kb 64 \
     --cache-total-mb 1 --batch-cols 4 --reps 1 --min-avail-mb 0 --json > "$OUT/kernbench_qemu_selftest.json"
  head -c 3000 "$OUT/kernbench_qemu_selftest.json"
fi
'
