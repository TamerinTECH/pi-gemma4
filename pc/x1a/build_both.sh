#!/usr/bin/env bash
# X1a host wrapper (Git Bash): builds upstream llama.cpp 8345f33 and an identical-flags control rebuild of
# a7b94df in the pi-gemma4-build container, at -j 4, under results/.cpu.lock. Source trees are mounted
# read-only; build trees live on the Docker volumes pigemma-x1a-build-<rev> (created by this script).
set -u
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
OWNER="X1a-upstream-llama-recent (wf_02d183f6-341)"
LOG="$REPO/results/comparisons/x1a/build_both.log"
export MSYS_NO_PATHCONV=1
exec >> "$LOG" 2>&1
echo "=== $(date -Is) build_both.sh start"
# disk gates
for d in /d /c; do
  free_kb=$(df -k "$d" | awk 'NR==2{print $4}')
  echo "disk $d free: $((free_kb/1024/1024)) GB"
  if [ "$free_kb" -lt $((10*1024*1024)) ]; then echo "REFUSED: $d below the 10 GB gate"; exit 4; fi
done
bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "docker builds: llama.cpp upstream 8345f33 + a7b94df control rebuild (-j 4)" 7200 || exit 4
trap 'bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"; echo "=== $(date -Is) build_both.sh end"' EXIT

build_one() { # build_one <host src dir name under third_party> <host out dir name under third_party> <volume> <label>
  local srcname="$1" outname="$2" vol="$3" label="$4"
  mkdir -p "$REPO/third_party/$outname/build-linux-x86"
  docker volume create "$vol" >/dev/null
  docker run --rm --name "pigemma-x1a-build-$label" \
    -e JOBS=4 -e PIGEMMA_IMAGE=pi-gemma4-build:latest \
    -v "$REPO_W/third_party/$srcname:/work/third_party/$srcname:ro" \
    -v "$vol:/work/third_party/$srcname/build-linux-x86" \
    -v "$REPO_W/third_party/$outname/build-linux-x86:/out" \
    -v "$REPO_W/pc/x1a:/x1a:ro" \
    pi-gemma4-build:latest \
    bash /x1a/build_engine_in_container.sh "/work/third_party/$srcname" "/work/third_party/$srcname/build-linux-x86" /out "$label"
  echo "docker exit: $? ($label)"
}
# 1) upstream (needs its own build-linux-x86 mountpoint in the clone; llama.cpp's .gitignore ignores build*)
mkdir -p "$REPO/third_party/llama.cpp-upstream-8345f33/build-linux-x86"
build_one llama.cpp-upstream-8345f33 llama.cpp-upstream-8345f33 pigemma-x1a-build-8345f33 upstream-8345f33
# 2) control: a7b94df rebuilt with the identical script/flags (pinned tree mounted read-only, nothing written to it)
build_one llama.cpp llama.cpp-a7b94df-rebuild-x1a pigemma-x1a-build-a7b94df a7b94df-rebuild
echo "ALL_BUILDS_DONE"
