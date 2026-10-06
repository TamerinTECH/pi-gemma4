#!/usr/bin/env bash
set -euo pipefail
root=/tmp/cf-engine-work
base=/mnt/usb/pigemma/cf-engine
tar -xzf "$base/comparison-update.tar.gz" -C "$root"
cmake -S "$root/cf-engine" -B "$root/cf-engine/build-pi" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DCF_BUILD_COMPARISONS=ON -DCF_EXPERIMENTAL_ASM=ON
cmake --build "$root/cf-engine/build-pi" -j 2 --target cf-run cf-kernel-test cf-operator-bench
cp "$root/cf-engine/build-pi/bin/cf-run" "$root/cf-engine/build-pi/bin/cf-kernel-test" "$root/cf-engine/build-pi/bin/cf-operator-bench" "$base/bin/"
"$base/bin/cf-kernel-test"
