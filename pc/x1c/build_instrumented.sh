#!/usr/bin/env bash
# X1c host wrapper (Git Bash): evidence build of the instrumented ik_llama.cpp server (-j 4, CPU lock).
set -u
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
OWNER="X1c-ik-llama-x86-engine (wf_02d183f6-341)"
SHA="3d27f5bb6348c4d79edded63b03e9a93efb2f80e"
VOL="pigemma-x1c-ik-3d27f5b"
OUTNAME="ik_llama.cpp-x1c-instrumented"
LOG="$REPO/results/comparisons/x1c/build_instrumented.log"
export MSYS_NO_PATHCONV=1
exec >> "$LOG" 2>&1
echo "=== $(date -Is) build_instrumented.sh start"
for d in /d /c; do
  free_kb=$(df -k "$d" | awk 'NR==2{print $4}')
  echo "disk $d free: $((free_kb/1024/1024)) GB"
  if [ "$free_kb" -lt $((10*1024*1024)) ]; then echo "REFUSED: $d below the 10 GB gate"; exit 4; fi
done
if [ "${NO_LOCK:-0}" != 1 ]; then
  bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "docker build: instrumented ik_llama.cpp llama-server (evidence only, -j 4)" 7200 || exit 4
  trap 'bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"; echo "=== $(date -Is) build_instrumented.sh end"' EXIT
fi
mkdir -p "$REPO/third_party/$OUTNAME"
docker run --rm --name "pigemma-x1c-build-instr" \
  -e JOBS=4 \
  -v "$VOL:/vol" \
  -v "$REPO_W/third_party/$OUTNAME:/out" \
  -v "$REPO_W/pc/x1c:/x1c:ro" \
  pi-gemma4-build:latest \
  bash /x1c/build_instrumented_in_container.sh /vol /out "$SHA"
rc=$?
echo "docker exit: $rc"
exit $rc
