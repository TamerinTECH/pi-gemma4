#!/usr/bin/env bash
# X1c host wrapper (Git Bash): builds ik_llama.cpp 3d27f5b for x86-64 Linux in the pi-gemma4-build
# container at -j 4 under results/.cpu.lock. X0's clone is mounted read-only; the LF source tree and
# the build tree live on the Docker volume pigemma-x1c-ik-3d27f5b (created by this script).
set -u
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
OWNER="X1c-ik-llama-x86-engine (wf_02d183f6-341)"
SHA="3d27f5bb6348c4d79edded63b03e9a93efb2f80e"
VOL="pigemma-x1c-ik-3d27f5b"
LOG="$REPO/results/comparisons/x1c/build_ik.log"
export MSYS_NO_PATHCONV=1
mkdir -p "$REPO/results/comparisons/x1c"
exec >> "$LOG" 2>&1
echo "=== $(date -Is) build_ik.sh start"
for d in /d /c; do
  free_kb=$(df -k "$d" | awk 'NR==2{print $4}')
  echo "disk $d free: $((free_kb/1024/1024)) GB"
  if [ "$free_kb" -lt $((10*1024*1024)) ]; then echo "REFUSED: $d below the 10 GB gate"; exit 4; fi
done
bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "docker build: ik_llama.cpp 3d27f5b x86-64 (-j 4)" 7200 || exit 4
trap 'bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"; echo "=== $(date -Is) build_ik.sh end"' EXIT
mkdir -p "$REPO/third_party/ik_llama.cpp/build-linux-x86"
docker volume create "$VOL" >/dev/null
docker run --rm --name "pigemma-x1c-build-ik" \
  -e JOBS=4 -e CLEAN="${CLEAN:-0}" -e PIGEMMA_IMAGE=pi-gemma4-build:latest \
  -v "$REPO_W/third_party/ik_llama.cpp:/clone:ro" \
  -v "$VOL:/vol" \
  -v "$REPO_W/third_party/ik_llama.cpp/build-linux-x86:/out" \
  -v "$REPO_W/pc/x1c:/x1c:ro" \
  pi-gemma4-build:latest \
  bash /x1c/build_ik_in_container.sh /clone /vol /out "$SHA" ik-3d27f5b
rc=$?
echo "docker exit: $rc"
if [ "$rc" = 0 ]; then echo "IK_BUILD_DONE"; else echo "IK_BUILD_FAILED"; fi
exit $rc
