#!/usr/bin/env bash
set -euo pipefail
root=/tmp/cf-engine-work
base=/mnt/usb/pigemma/cf-engine
mkdir -p "$base/bin/v1"
cp -n "$base/bin/cf-run" "$base/bin/cf-kernel-test" "$base/bin/v1/"
cmake -S "$root/cf-engine" -B "$root/cf-engine/build-pi" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
cmake --build "$root/cf-engine/build-pi" -j 2 --target cf-run cf-kernel-test
cp "$root/cf-engine/build-pi/bin/cf-run" "$root/cf-engine/build-pi/bin/cf-kernel-test" "$base/bin/"
"$base/bin/cf-kernel-test" --bench --iterations 3
