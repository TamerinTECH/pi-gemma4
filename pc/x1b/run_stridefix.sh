#!/usr/bin/env bash
# X1b host wrapper (Git Bash): LOCALLY ADAPTED HotPin routing dump (stride fix) - build the extra binary,
# then profile data/workloads/calibration.jsonl with it (tag "stridefix"). Not HotPin's original procedure.
set -u
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
OWNER="X1b-hotpin (wf_02d183f6-341)"
HPOUT="hotpin-llm-build-a6647b1"
LOG="$REPO/results/comparisons/x1b/stridefix_build.log"
export MSYS_NO_PATHCONV=1
cd "$REPO" || exit 1
{
echo "=== $(date -Is) run_stridefix.sh start"
bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "HotPin local adaptation: rebuild llama-simple-chat with the routing-dump stride fix (one object + link)" 7200 || exit 4
docker run --rm --name pigemma-x1b-stridefix \
  -v "pigemma-x1b-hotpin-a6647b1:/hp" \
  -v "$REPO_W/third_party/$HPOUT:/out" \
  -v "$REPO_W/pc/x1b:/x1b:ro" \
  pi-gemma4-build:latest \
  bash /x1b/build_stridefix_in_container.sh /hp/src /hp/build /out
rc=$?
echo "docker exit: $rc (stridefix build)"
bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"
echo "=== $(date -Is) run_stridefix.sh build end"
} >> "$LOG" 2>&1
[ "${rc:-1}" -eq 0 ] || exit 2
bash "$REPO/pc/x1b/run_routing_dump.sh" stridefix "/work/third_party/$HPOUT/build-linux-x86/bin/llama-simple-chat-stridefix"
