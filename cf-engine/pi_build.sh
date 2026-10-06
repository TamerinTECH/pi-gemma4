#!/usr/bin/env bash
set -euo pipefail
root=/tmp/cf-engine-work
mkdir -p "$root"
tar -xzf /mnt/usb/pigemma/cf-engine-source.tar.gz -C "$root"
cmake -S "$root/cf-engine" -B "$root/cf-engine/build-pi" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
cmake --build "$root/cf-engine/build-pi" -j 2 --target cf-run cf-kernel-test
mkdir -p /mnt/usb/pigemma/cf-engine/bin
cp "$root/cf-engine/build-pi/bin/cf-run" "$root/cf-engine/build-pi/bin/cf-kernel-test" /mnt/usb/pigemma/cf-engine/bin/
cp "$root/cf-engine/build-pi/CMakeCache.txt" /mnt/usb/pigemma/cf-engine/CMakeCache.txt
/mnt/usb/pigemma/cf-engine/bin/cf-kernel-test
