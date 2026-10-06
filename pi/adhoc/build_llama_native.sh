#!/usr/bin/env bash
# pi-gemma4 ad-hoc: native llama.cpp build on the real Pi 5 at the pinned commit a7b94df.
# Run with: nohup nice -n 10 ./build_llama_native.sh > build_llama.out 2>&1 &
# Does not touch any service. Deletes object files afterwards to keep the SD card free.
set -euo pipefail
cd ~/pigemma
if [ ! -d llama.cpp/.git ]; then
  git init -q llama.cpp
  git -C llama.cpp remote add origin https://github.com/ggml-org/llama.cpp
  git -C llama.cpp fetch -q --depth 1 origin a7b94df2c616bc1f62a73b964b4a71cb0dcc488e
  git -C llama.cpp checkout -q FETCH_HEAD
fi
echo "commit $(git -C llama.cpp rev-parse --short HEAD)"
start=$(date +%s)
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF > build-cmake.log 2>&1
grep -iE "ARM|GGML_CPU|dotprod|fp16|i8mm|mcpu|native" build-cmake.log | head -15 || true
cmake --build llama.cpp/build -j 4 --target llama-cli llama-completion llama-server llama-bench llama-perplexity > build.log 2>&1
end=$(date +%s)
echo "build_seconds=$((end-start))"
ls llama.cpp/build/bin
llama.cpp/build/bin/llama-cli --version 2>&1 | tail -2
find llama.cpp/build -name '*.o' -delete
du -sh llama.cpp
echo "free_bytes=$(df -B1 --output=avail / | tail -1)"
echo BUILD_DONE
