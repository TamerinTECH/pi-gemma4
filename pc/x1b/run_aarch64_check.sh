#!/usr/bin/env bash
# X1b host wrapper (Git Bash): ARM status of HotPin.
#   step 1  cross-build llama.cpp a6647b1 + HotPin patches for aarch64 (x86 container, apt cross GCC, -j 4)
#   step 2  run the aarch64 llama-server under qemu-user in the emulated-Pi image (Debian 13 arm64) with
#           LLAMA_HOT_EXPERTS + -nr on our exact GGUF: load + lock accounting only (no generation, no
#           warm-up - emulation speed is not representative and a warm-up would touch every expert)
# Nothing runs on the real Pi. Holds results/.cpu.lock; step 2 also holds results/.bench.lock.
set -u
REPO_W="REPO_ROOT"
REPO="REPO_ROOT"
OWNER="X1b-hotpin (wf_02d183f6-341)"
HPOUT="hotpin-llm-build-a6647b1"
LOG="$REPO/results/comparisons/x1b/aarch64_check.log"
export MSYS_NO_PATHCONV=1
cd "$REPO" || exit 1
exec >> "$LOG" 2>&1
echo "=== $(date -Is) run_aarch64_check.sh start"
for d in /d /c; do
  free_kb=$(df -k "$d" | awk 'NR==2{print $4}')
  echo "disk $d free: $((free_kb/1024/1024)) GB"
  if [ "$free_kb" -lt $((10*1024*1024)) ]; then echo "REFUSED: $d below the 10 GB gate"; exit 4; fi
done
mkdir -p "$REPO/third_party/$HPOUT/build-linux-aarch64-cross" "$REPO/results/comparisons/x1b/lock_verify/aarch64-emu"
bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "HotPin ARM status: aarch64 cross-build of a6647b1 + HotPin patches (-j 4), then an emulated load + lock-accounting check (no generation)" 7200 || exit 4
trap 'bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"; echo "=== $(date -Is) run_aarch64_check.sh end"' EXIT
docker run --rm --name pigemma-x1b-aarch64-cross \
  -e JOBS=4 -e TARGETS="llama-server llama-simple-chat" \
  -v "pigemma-x1b-hotpin-a6647b1:/hp" \
  -v "$REPO_W/third_party/$HPOUT/build-linux-aarch64-cross:/out" \
  -v "$REPO_W/pc/x1b:/x1b:ro" \
  pi-gemma4-build:latest \
  bash /x1b/build_aarch64_cross_in_container.sh /hp/src /hp/build-aarch64-cross /out
rc=$?
echo "docker exit: $rc (aarch64 cross build)"
if [ $rc -ne 0 ]; then echo "AARCH64_CHECK_DONE build_failed"; exit 0; fi
cat > "$REPO/results/comparisons/x1b/lock_verify/cases_aarch64_emu.json" <<JSON
[
 {"case_id": "hotpin-aarch64-emu-nr-original2gb",
  "bin": "/work/third_party/$HPOUT/build-linux-aarch64-cross/bin/llama-server", "nr": true, "warmup": false,
  "env": {"LLAMA_HOT_EXPERTS": "/work/results/comparisons/x1b/hotfiles/hotpin_original_2gb.txt"},
  "note": "aarch64 cross build run under qemu-user (Debian 13 arm64 container, 4 KiB pages): load + lock accounting only"}
]
JSON
python -m pigemma.paths lock --hold --min-free-gb 16 -- \
  docker run --rm --name pigemma-x1b-aarch64-emu --platform linux/arm64 \
    --cpuset-cpus 2,4,6,8 --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
    -v "$REPO_W:/work" -v pigemma-models:/models:ro \
    -w /work pi-gemma4-pi-emu:trixie \
    python3 /work/pc/x1b/hotpin_lock_verify.py \
      --cases /work/results/comparisons/x1b/lock_verify/cases_aarch64_emu.json \
      --model /models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
      --out /work/results/comparisons/x1b/lock_verify/aarch64-emu --skip-generation
echo "emulated lock verify exit: $?"
echo "AARCH64_CHECK_DONE"
