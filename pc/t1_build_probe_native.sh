#!/usr/bin/env bash
# Stage T1: build pigemma-probe with the SAME CPU flags as the harness's llama-server
# (GGML_NATIVE=ON, see results/comparisons/x1a/README.md), so that the probe's kernels - and with
# them its greedy token ids - equal the server's. Output: tools/build-linux-x86-native/bin/pigemma-probe
#
# RECONSTRUCTED on 2026-10-06 from tools/build-linux-x86-native/logs/{configure,build}.log and
# bin/pigemma-tools-build.json of the binary that produced the T1 traces (probe 1.1.0, sha256
# 94e282571cab9cb93ea014d56437e3633643fdde3c181b6ad2f937b70e8f1da5). The original command line was
# not saved; this script was NOT re-run in the resumed pass (the binary already existed and a
# rebuild needs the desktop CPU lock). The standard build (pc/docker_run.sh build-tools) uses
# -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=ON and writes tools/build-linux-x86/bin.
#
# Usage (Git Bash, repo root; take results/.cpu.lock first):  bash pc/t1_build_probe_native.sh [jobs]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && (pwd -W 2>/dev/null || pwd))"
JOBS="${1:-4}"
OUT="tools/build-linux-x86-native"
export MSYS_NO_PATHCONV=1
docker run --rm --network none -v "${REPO}:/work" -w /work pi-gemma4-build:latest bash -c "
  set -euo pipefail
  mkdir -p /work/${OUT}/bin /work/${OUT}/logs
  cmake -S /work/tools -B /build -G Ninja -DLLAMA_DIR=/work/third_party/llama.cpp \
        -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DPIGEMMA_REQUIRE_ALL_TOOLS=OFF 2>&1 | tee /work/${OUT}/logs/configure.log
  cmake --build /build -j ${JOBS} --target pigemma-probe 2>&1 | tee /work/${OUT}/logs/build.log
  cp /build/bin/pigemma-probe /build/bin/pigemma-tools-build.json /work/${OUT}/bin/
  sha256sum /work/${OUT}/bin/pigemma-probe
  /work/${OUT}/bin/pigemma-probe --version
"
