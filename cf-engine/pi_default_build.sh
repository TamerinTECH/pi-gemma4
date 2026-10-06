#!/usr/bin/env bash
set -euo pipefail
root=/tmp/cf-engine-work/cf-engine
base=/mnt/usb/pigemma/cf-engine
mkdir "$base/bin/exact-experimental"
cp "$base/bin/cf-run" "$base/bin/cf-kernel-test" "$base/bin/cf-operator-bench" "$base/bin/exact-experimental/"
cp "$root/build-pi/CMakeCache.txt" "$base/CMakeCache-exact-experimental.txt"
cmake -S "$root" -B "$root/build-pi" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DCF_BUILD_COMPARISONS=ON -DCF_EXPERIMENTAL_ASM=OFF
cmake --build "$root/build-pi" -j 2 --target cf-run cf-kernel-test
cp "$root/build-pi/bin/cf-run" "$root/build-pi/bin/cf-kernel-test" "$base/bin/"
cp "$root/build-pi/CMakeCache.txt" "$base/CMakeCache-default.txt"
"$base/bin/cf-kernel-test"
