#!/usr/bin/env bash
# pigemma-fast: build a PATCHED COPY of the pinned llama.cpp (a7b94df) on the Pi. The reference tree
# ~/pigemma/llama.cpp and its binaries are not touched.
#   build_fast.sh [tree-name] [bin-dir-name]
#     tree-name     directory under the work area (default llama-fast)      -> /mnt/pigemma-fast/<tree-name>
#     bin-dir-name  directory under /mnt/usb/pigemma/fast (default bin)     -> binaries are copied there
# A new tree is created from `git archive` of the reference tree plus the patch file (patch -p1);
# an existing tree is brought up to date with apply_patches.py --upgrade and rebuilt incrementally.
# Run: nohup nice -n 10 ./build_fast.sh > build_fast.out 2>&1 &
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=~/pigemma/llama.cpp
work=${PIGEMMA_FAST_WORK:-/mnt/pigemma-fast}
dst=$work/${1:-llama-fast}
out=/mnt/usb/pigemma/fast/${2:-bin}
base=a7b94df2c616bc1f62a73b964b4a71cb0dcc488e
patchfile=$here/${PIGEMMA_PATCH:-pigemma-fast-v7.patch}
versions="v1 v2 v3 v4 v5 v6 v7"
[ "$(git -C "$src" rev-parse HEAD)" = "$base" ] || { echo "reference tree is not $base"; exit 3; }
[ -z "$(git -C "$src" status --porcelain --untracked-files=no)" ] || { echo "reference tree has local changes"; exit 3; }
findmnt /mnt/usb > /dev/null || { echo "/mnt/usb is not mounted (run mount_work.sh)"; exit 3; }
findmnt "$work" > /dev/null || [ "$work" != /mnt/pigemma-fast ] || { echo "$work is not mounted (run mount_work.sh)"; exit 3; }
start=$(date +%s)
if [ ! -f "$dst/PIGEMMA_FAST_PATCHED" ]; then
  rm -rf "$dst"; mkdir -p "$dst"
  git -C "$src" archive "$base" | tar -x -C "$dst"
  (cd "$dst" && patch -p1 -s < "$patchfile")
  echo "$versions" > "$dst/PIGEMMA_FAST_PATCHED"
  echo "new tree $dst from $base + $(basename "$patchfile") ($(sha256sum "$patchfile" | cut -c1-16))"
else
  echo "reusing patched tree $dst ($(cat "$dst/PIGEMMA_FAST_PATCHED"))"
  python3 "$here/apply_patches.py" --upgrade "$dst"
fi
cmake -S "$dst" -B "$dst/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF > "$dst/cmake.log" 2>&1 || { tail -20 "$dst/cmake.log"; exit 4; }
cmake --build "$dst/build" -j 3 --target llama-server llama-bench llama-completion > "$dst/build.log" 2>&1 || { grep -E "error|FAILED" -A6 "$dst/build.log" | head -60; exit 5; }
mkdir -p "$out"
cp "$dst/build/bin/llama-server" "$dst/build/bin/llama-bench" "$dst/build/bin/llama-completion" "$out/"
cp "$patchfile" "$out/"
(cd "$out" && sha256sum llama-server llama-bench llama-completion "$(basename "$patchfile")" > SHA256SUMS)
end=$(date +%s)
echo "build_seconds=$((end-start))"
ls -la "$out"
echo FAST_BUILD_DONE
