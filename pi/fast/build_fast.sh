#!/usr/bin/env bash
# pigemma-fast: build a PATCHED COPY of the pinned llama.cpp (a7b94df) on the Pi in /tmp (tmpfs); the reference tree
# ~/pigemma/llama.cpp and its binaries are not touched. Binaries are copied to the USB stick.
# Run: nohup nice -n 10 ./build_fast.sh > build_fast.out 2>&1 &
set -euo pipefail
src=~/pigemma/llama.cpp
dst=/tmp/llama-fast
out=/mnt/usb/pigemma/fast/bin
[ "$(git -C "$src" rev-parse HEAD)" = "a7b94df2c616bc1f62a73b964b4a71cb0dcc488e" ] || { echo "reference tree is not a7b94df"; exit 3; }
[ -z "$(git -C "$src" status --porcelain --untracked-files=no)" ] || { echo "reference tree has local changes"; exit 3; }
findmnt /mnt/usb > /dev/null || { echo "/mnt/usb is not mounted"; exit 3; }
start=$(date +%s)
if [ ! -f "$dst/PIGEMMA_FAST_PATCHED" ]; then
  rm -rf "$dst"; mkdir -p "$dst"
  git -C "$src" archive HEAD | tar -x -C "$dst"
  python3 ~/pigemma/fast/apply_patches.py "$dst"
else
  echo "reusing patched tree $dst ($(cat "$dst/PIGEMMA_FAST_PATCHED"))"
  python3 ~/pigemma/fast/apply_patches.py --upgrade "$dst"
fi
cmake -S "$dst" -B "$dst/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF > "$dst/cmake.log" 2>&1 || { tail -20 "$dst/cmake.log"; exit 4; }
cmake --build "$dst/build" -j 3 --target llama-server llama-bench llama-completion > "$dst/build.log" 2>&1 || { grep -E "error|FAILED" -A6 "$dst/build.log" | head -60; exit 5; }
mkdir -p "$out"
cp "$dst/build/bin/llama-server" "$dst/build/bin/llama-bench" "$dst/build/bin/llama-completion" "$out/"
cp ~/pigemma/fast/apply_patches.py "$out/apply_patches.py.used"
end=$(date +%s)
echo "build_seconds=$((end-start))"
ls -la "$out"
echo FAST_BUILD_DONE
