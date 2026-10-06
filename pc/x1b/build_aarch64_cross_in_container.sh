#!/usr/bin/env bash
# X1b: does llama.cpp a6647b1 + HotPin patches BUILD for aarch64? (runs INSIDE pi-gemma4-build:latest, x86)
# Cross-compiles the already patched tree (/hp/src on volume pigemma-x1b-hotpin-a6647b1) with Ubuntu's
# aarch64 cross GCC, installed with apt INSIDE this throw-away container (no image is modified).
# ISA flags = the ones of the existing emulated-Pi a7b94df build: armv8.2-a+dotprod+fp16 (Cortex-A76 class).
#   build_aarch64_cross_in_container.sh <SRC_DIR> <BUILD_DIR> <OUT_DIR>
set -euo pipefail
SRC="${1:?src}"; BUILD="${2:?build}"; OUT="${3:?out}"
JOBS="${JOBS:-4}"
read -r -a TARGETS <<< "${TARGETS:-llama-server llama-simple-chat}"
mkdir -p "$OUT/bin" "$OUT/logs" "$BUILD"
export DEBIAN_FRONTEND=noninteractive
t0=$(date +%s)
( apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc-aarch64-linux-gnu g++-aarch64-linux-gnu ) > "$OUT/logs/apt.log" 2>&1 \
  || { tail -20 "$OUT/logs/apt.log"; echo "APT FAILED (cross toolchain not installable)"; exit 9; }
t1=$(date +%s)
aarch64-linux-gnu-g++ --version | head -1
git config --global --add safe.directory '*'
REV=$(git -C "$SRC" rev-parse HEAD)
DIRTY=$(git -C "$SRC" -c core.fileMode=false status --short | wc -l)
CMAKE_FLAGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DCMAKE_SYSTEM_NAME=Linux
  -DCMAKE_SYSTEM_PROCESSOR=aarch64
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
  -DGGML_NATIVE=OFF
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
  -DGGML_OPENMP=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_OPENSSL=OFF
  -DLLAMA_BUILD_UI=OFF
  -DLLAMA_USE_PREBUILT_UI=OFF
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)
echo "== aarch64 cross build: src=$SRC rev=$REV modified_files=$DIRTY build=$BUILD jobs=$JOBS targets=${TARGETS[*]}"
cmake -S "$SRC" -B "$BUILD" "${CMAKE_FLAGS[@]}" > "$OUT/logs/configure.log" 2>&1 || { tail -40 "$OUT/logs/configure.log"; echo "CONFIGURE FAILED"; exit 10; }
t2=$(date +%s)
nice -n 10 cmake --build "$BUILD" -j "$JOBS" --target "${TARGETS[@]}" > "$OUT/logs/build.log" 2>&1 || { tail -60 "$OUT/logs/build.log"; echo "BUILD FAILED"; exit 11; }
t3=$(date +%s)
for t in "${TARGETS[@]}"; do cp -f "$BUILD/bin/$t" "$OUT/bin/$t"; chmod +x "$OUT/bin/$t" || true; done
{
  echo "{"
  echo " \"schema\": \"pigemma-x1b-aarch64-cross-build v1\","
  echo " \"revision\": \"$REV\", \"files_modified_vs_revision\": $DIRTY,"
  echo " \"patches\": \"hotpin_l1_pin.diff + dump_routing_prefetch.diff (see ../build-linux-x86/PATCHED_TREE.diff)\","
  echo " \"compiler\": \"$(aarch64-linux-gnu-g++ --version | head -1)\","
  echo " \"host_image\": \"pi-gemma4-build:latest (x86-64) + apt gcc-aarch64-linux-gnu g++-aarch64-linux-gnu, installed in the throw-away container\","
  echo " \"cmake_configure_flags\": \"${CMAKE_FLAGS[*]}\","
  echo " \"targets\": \"${TARGETS[*]}\", \"jobs\": $JOBS,"
  echo " \"apt_seconds\": $((t1-t0)), \"configure_seconds\": $((t2-t1)), \"build_seconds\": $((t3-t2)),"
  echo " \"warnings_in_build_log\": $(grep -c 'warning:' "$OUT/logs/build.log" || true),"
  echo " \"file_llama_server\": \"$(file -b "$OUT/bin/llama-server" 2>/dev/null | sed 's/"/\\"/g' || echo 'file(1) not installed')\","
  echo " \"sha256\": {"
  first=1
  for t in "${TARGETS[@]}"; do
    [ $first = 1 ] || echo ","
    printf '  "%s": {"bytes": %s, "sha256": "%s"}' "$t" "$(stat -c %s "$OUT/bin/$t")" "$(sha256sum "$OUT/bin/$t" | cut -d' ' -f1)"
    first=0
  done
  echo ""
  echo " },"
  echo " \"compile_flags_ggml_cpu\": \"$(grep -m1 -o '\-march=[^ ]*' "$BUILD/compile_commands.json" || true)\""
  echo "}"
} > "$OUT/BUILD_INFO.json"
cat "$OUT/BUILD_INFO.json"
aarch64-linux-gnu-readelf -h "$OUT/bin/llama-server" | grep -E "Class|Machine" || true
aarch64-linux-gnu-readelf -d "$OUT/bin/llama-server" | grep NEEDED || true
echo "AARCH64_CROSS_BUILD_DONE apt $((t1-t0)) s, configure $((t2-t1)) s, build $((t3-t2)) s"
