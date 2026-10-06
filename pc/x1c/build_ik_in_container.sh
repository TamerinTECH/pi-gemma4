#!/usr/bin/env bash
# X1c (comparisons workflow): build ikawrakow/ik_llama.cpp as a Linux x86-64 CPU engine INSIDE the
# pi-gemma4-build container.
#
#   build_ik_in_container.sh <CLONE_RO> <VOL> <OUT_DIR> <EXPECTED_SHA> <LABEL>
#
# <CLONE_RO>  X0's clone (third_party/ik_llama.cpp), mounted READ-ONLY. Its working tree has CRLF line
#             endings (cloned on Windows with core.autocrlf=true), which would end up inside raw string
#             literals, so the build does NOT use that working tree: it makes an LF clone of the same
#             commit on the Docker volume (<VOL>/src, core.autocrlf=false) and builds that.
# <VOL>       Docker volume (ext4 in the Docker Desktop VM): <VOL>/src + <VOL>/build
# <OUT_DIR>   bin/, logs/, BUILD_INFO.json are copied here (repo: third_party/ik_llama.cpp/build-linux-x86)
#
# Flags: the subset of the pinned llama.cpp Linux build flags (pc/x1a) that ik_llama.cpp's CMake has,
# plus the ik-only switches that correspond to the ones it lacks, all at ik's defaults:
#   ours: -G Ninja Release BUILD_SHARED_LIBS=OFF GGML_NATIVE=ON GGML_OPENMP=ON LLAMA_BUILD_TESTS=OFF
#         LLAMA_OPENSSL=OFF LLAMA_BUILD_UI=OFF LLAMA_USE_PREBUILT_UI=OFF
#   ik  : LLAMA_OPENSSL -> LLAMA_SERVER_SSL=OFF + LLAMA_CURL=OFF (ik defaults); LLAMA_BUILD_UI /
#         LLAMA_USE_PREBUILT_UI do not exist (ik always embeds its web UI assets; it is switched off at
#         run time with `--webui none`); GGML_LTO=OFF and GGML_CCACHE=OFF are ik defaults / no-ops here.
set -euo pipefail
CLONE="${1:?clone}"; VOL="${2:?vol}"; OUT="${3:?out}"; EXPECT="${4:?sha}"; LABEL="${5:?label}"
JOBS="${JOBS:-4}"
SRC="$VOL/src"; BUILD="$VOL/build"
TARGETS=(llama-server llama-cli llama-bench llama-tokenize llama-eval-callback llama-perplexity)
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
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)
mkdir -p "$OUT/bin" "$OUT/logs"
LOGDIR="$OUT/logs"
git config --global --add safe.directory '*'
git config --global core.autocrlf false
if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git -c core.autocrlf=false clone --quiet --no-local --depth 1 "file://$CLONE" "$SRC" > "$LOGDIR/clone.log" 2>&1 \
    || { cat "$LOGDIR/clone.log"; echo "CLONE FAILED"; exit 9; }
fi
REV=$(git -C "$SRC" rev-parse HEAD)
[ "$REV" = "$EXPECT" ] || { echo "REVISION MISMATCH: $REV != $EXPECT"; exit 8; }
DIRTY=$(git -C "$SRC" status --porcelain | wc -l)
CRLF_FILES=$(git -C "$SRC" ls-files --eol | awk '$1 == "i/lf" && $2 == "w/crlf"' | wc -l)
echo "== $LABEL: rev=$REV dirty_files=$DIRTY files_with_crlf_added_by_checkout=$CRLF_FILES build=$BUILD out=$OUT jobs=$JOBS"
[ "$DIRTY" = 0 ] || { git -C "$SRC" status --porcelain | head; echo "SOURCE TREE NOT CLEAN"; exit 7; }
[ "$CRLF_FILES" = 0 ] || { echo "LF CLONE STILL HAS CRLF CONVERSION"; exit 7; }
if [ "${CLEAN:-0}" = 1 ]; then rm -rf "$BUILD"; fi
mkdir -p "$BUILD"
t0=$(date +%s)
cmake -S "$SRC" -B "$BUILD" "${CMAKE_FLAGS[@]}" > "$LOGDIR/configure.log" 2>&1 || { tail -60 "$LOGDIR/configure.log"; echo "CONFIGURE FAILED"; exit 10; }
t1=$(date +%s)
nice -n 10 cmake --build "$BUILD" -j "$JOBS" --target "${TARGETS[@]}" > "$LOGDIR/build.log" 2>&1 || { tail -80 "$LOGDIR/build.log"; echo "BUILD FAILED"; exit 11; }
t2=$(date +%s)
for t in "${TARGETS[@]}"; do cp -f "$BUILD/bin/$t" "$OUT/bin/$t"; chmod +x "$OUT/bin/$t" || true; done
cp -f "$SRC/LICENSE" "$OUT/LICENSE.ik_llama.cpp.txt"
(grep -c "warning:" "$LOGDIR/build.log" || true) > "$LOGDIR/warning_count.txt"
python3 /x1c/build_info.py "$SRC" "$BUILD" "$OUT" "$LABEL" "$REV" "$((t1-t0))" "$((t2-t1))" "$JOBS" "${TARGETS[*]}" "${CMAKE_FLAGS[*]}" "$CLONE"
echo "== $LABEL DONE: configure $((t1-t0)) s, build $((t2-t1)) s"
ls -la "$OUT/bin"
