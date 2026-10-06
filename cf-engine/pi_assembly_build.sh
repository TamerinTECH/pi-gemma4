#!/usr/bin/env bash
set -euo pipefail
root=/tmp/cf-engine-work
base=/mnt/usb/pigemma/cf-engine
mkdir -p "$base/bin/v2"
cp -n "$base/bin/cf-run" "$base/bin/cf-kernel-test" "$base/bin/v2/"
tar -xzf "$base/assembly-update.tar.gz" -C "$root"
cmake -S "$root/cf-engine" -B "$root/cf-engine/build-pi" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DCF_EXPERIMENTAL_ASM=ON
cmake --build "$root/cf-engine/build-pi" -j 2 --target cf-run cf-kernel-test
cp "$root/cf-engine/build-pi/bin/cf-run" "$root/cf-engine/build-pi/bin/cf-kernel-test" "$base/bin/"
"$base/bin/cf-kernel-test"
cp "$root/cf-engine/build-pi/CMakeCache.txt" "$base/CMakeCache-assembly.txt"
objdump -d -C "$root/cf-engine/build-pi/llama/ggml/src/CMakeFiles/ggml-cpu.dir/__/__/__/q4_tile4_a76.S.o" > "$base/assembly.disasm.txt"
