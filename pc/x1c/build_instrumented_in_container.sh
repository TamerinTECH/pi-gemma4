#!/usr/bin/env bash
# X1c: EVIDENCE build of ik_llama.cpp 3d27f5b with trace-only instrumentation of the MoE expert
# prefetcher (pc/x1c/instrument_prefetch.py). NOT an engine: it is never registered in engines.json.
#
#   build_instrumented_in_container.sh <VOL> <OUT_DIR> <EXPECTED_SHA>
#
# <VOL>/src        the LF clone the engine was built from (left untouched)
# <VOL>/src-instr  copy of it + the instrumentation (one file changed)
# <VOL>/build-instr build tree (same CMake flags as the engine build, only llama-server)
set -euo pipefail
VOL="${1:?vol}"; OUT="${2:?out}"; EXPECT="${3:?sha}"
JOBS="${JOBS:-4}"
SRC="$VOL/src"; ISRC="$VOL/src-instr"; BUILD="$VOL/build-instr"
CMAKE_FLAGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DGGML_NATIVE=ON
  -DGGML_OPENMP=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_CURL=OFF
  -DLLAMA_SERVER_SSL=OFF
  -DGGML_LTO=OFF
  -DGGML_CCACHE=OFF
)
mkdir -p "$OUT/bin" "$OUT/logs"
git config --global --add safe.directory '*'
git config --global core.autocrlf false
[ "$(git -C "$SRC" rev-parse HEAD)" = "$EXPECT" ] || { echo "REVISION MISMATCH"; exit 8; }
[ "$(git -C "$SRC" status --porcelain | wc -l)" = 0 ] || { echo "ENGINE SOURCE TREE NOT CLEAN"; exit 7; }
rm -rf "$ISRC"
cp -a "$SRC" "$ISRC"
python3 /x1c/instrument_prefetch.py "$ISRC/ggml/src/ggml-moe-prefetch.cpp"
git -C "$ISRC" diff > "$OUT/INSTRUMENTATION.diff"
git -C "$ISRC" status --porcelain > "$OUT/logs/git_status_instrumented.txt"
echo "changed files: $(cat "$OUT/logs/git_status_instrumented.txt" | tr '\n' ' ')"
[ "$(wc -l < "$OUT/logs/git_status_instrumented.txt")" = 1 ] || { echo "EXPECTED EXACTLY ONE CHANGED FILE"; exit 6; }
t0=$(date +%s)
cmake -S "$ISRC" -B "$BUILD" "${CMAKE_FLAGS[@]}" > "$OUT/logs/configure.log" 2>&1 || { tail -40 "$OUT/logs/configure.log"; echo "CONFIGURE FAILED"; exit 10; }
nice -n 10 cmake --build "$BUILD" -j "$JOBS" --target llama-server > "$OUT/logs/build.log" 2>&1 || { tail -60 "$OUT/logs/build.log"; echo "BUILD FAILED"; exit 11; }
t1=$(date +%s)
cp -f "$BUILD/bin/llama-server" "$OUT/bin/llama-server-x1c-instrumented"
cp -f "$SRC/LICENSE" "$OUT/LICENSE.ik_llama.cpp.txt"
# the engine tree must still be clean
[ "$(git -C "$SRC" status --porcelain | wc -l)" = 0 ] || { echo "ENGINE SOURCE TREE CHANGED"; exit 7; }
sha256sum "$OUT/bin/llama-server-x1c-instrumented" | tee "$OUT/logs/sha256.txt"
"$OUT/bin/llama-server-x1c-instrumented" --version 2>&1 | tee "$OUT/logs/version.txt" || true
echo "INSTRUMENTED_BUILD_DONE seconds=$((t1-t0)) diff_lines=$(wc -l < "$OUT/INSTRUMENTATION.diff")"
