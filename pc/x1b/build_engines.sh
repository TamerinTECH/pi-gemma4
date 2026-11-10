#!/usr/bin/env bash
# X1b host wrapper (Git Bash): builds HotPin's parent llama.cpp a6647b1 UNMODIFIED (engine hotpin-parent)
# and the same revision with HotPin's two core patches applied (engine hotpin) in the pi-gemma4-build
# container, -j 4, under results/.cpu.lock. Flags identical to pc/x1a (the pinned-build flags).
#   parent source : third_party/llama.cpp-a6647b1 (mounted read-only), build tree on volume pigemma-x1b-build-a6647b1
#   hotpin source : copy of the parent + `git apply` of third_party/hotpin-llm/patches/{hotpin_l1_pin,dump_routing_prefetch}.diff
#                   on volume pigemma-x1b-hotpin-a6647b1 (/hp/src, /hp/build) - AGPL-3.0 combined work
set -u
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
OWNER="X1b-hotpin (wf_02d183f6-341)"
LOG="$REPO/results/comparisons/x1b/build.log"
PARENT="llama.cpp-a6647b1"
HPOUT="hotpin-llm-build-a6647b1"
export MSYS_NO_PATHCONV=1
exec >> "$LOG" 2>&1
echo "=== $(date -Is) build_engines.sh start"
for d in /d /c; do
  free_kb=$(df -k "$d" | awk 'NR==2{print $4}')
  echo "disk $d free: $((free_kb/1024/1024)) GB"
  if [ "$free_kb" -lt $((10*1024*1024)) ]; then echo "REFUSED: $d below the 10 GB gate"; exit 4; fi
done
bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "docker builds: llama.cpp a6647b1 unmodified (hotpin-parent) + a6647b1 with HotPin patches (hotpin), -j 4" 7200 || exit 4
trap 'bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"; echo "=== $(date -Is) build_engines.sh end"' EXIT
mkdir -p "$REPO/third_party/$PARENT/build-linux-x86" "$REPO/third_party/$HPOUT/build-linux-x86"
docker volume create pigemma-x1b-build-a6647b1 >/dev/null
docker volume create pigemma-x1b-hotpin-a6647b1 >/dev/null
TARGETS="llama-server llama-tokenize llama-bench llama-simple-chat"
# 1) unmodified parent
docker run --rm --name pigemma-x1b-build-parent \
  -e JOBS=4 -e PIGEMMA_IMAGE=pi-gemma4-build:latest -e TARGETS="$TARGETS" \
  -v "$REPO_W/third_party/$PARENT:/work/third_party/$PARENT:ro" \
  -v "pigemma-x1b-build-a6647b1:/work/third_party/$PARENT/build-linux-x86" \
  -v "$REPO_W/third_party/$PARENT/build-linux-x86:/out" \
  -v "$REPO_W/pc/x1b:/x1b:ro" \
  pi-gemma4-build:latest \
  bash /x1b/build_in_container.sh "/work/third_party/$PARENT" "/work/third_party/$PARENT/build-linux-x86" /out hotpin-parent-a6647b1
echo "docker exit: $? (hotpin-parent)"
# 2) HotPin-patched tree (core patches only: L1 pin + routing dump/prefetch; gptoss/glm patches are not needed for Gemma 4)
docker run --rm --name pigemma-x1b-build-hotpin \
  -e JOBS=4 -e PIGEMMA_IMAGE=pi-gemma4-build:latest -e TARGETS="$TARGETS" \
  -e PATCH_FROM=/parent -e PATCHES="/hotpin/patches/hotpin_l1_pin.diff /hotpin/patches/dump_routing_prefetch.diff" \
  -v "$REPO_W/third_party/$PARENT:/parent:ro" \
  -v "$REPO_W/third_party/hotpin-llm:/hotpin:ro" \
  -v "pigemma-x1b-hotpin-a6647b1:/hp" \
  -v "$REPO_W/third_party/$HPOUT/build-linux-x86:/out" \
  -v "$REPO_W/pc/x1b:/x1b:ro" \
  pi-gemma4-build:latest \
  bash /x1b/build_in_container.sh /hp/src /hp/build /out hotpin-a6647b1
echo "docker exit: $? (hotpin)"
echo "ALL_BUILDS_DONE"
