#!/usr/bin/env bash
# X1b host wrapper (Git Bash): lock accounting / repack / referenced-pages check for the HotPin engines.
#   run_lock_verify.sh <run id> <cases json (repo-relative)> [memlock ulimit, default -1:-1] [ipc_lock yes|no]
# Holds results/.cpu.lock and results/.bench.lock. Functional check, not a benchmark.
set -u
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
OWNER="X1b-hotpin (wf_02d183f6-341)"
RUN_ID="${1:?run id}"; CASES="${2:?cases json}"; MEMLOCK="${3:--1:-1}"; IPC="${4:-yes}"
OUT_REL="results/comparisons/x1b/lock_verify/$RUN_ID"
LOG="$REPO/results/comparisons/x1b/lock_verify_${RUN_ID}.log"
export MSYS_NO_PATHCONV=1
cd "$REPO" || exit 1
exec >> "$LOG" 2>&1
echo "=== $(date -Is) run_lock_verify.sh $RUN_ID start (memlock=$MEMLOCK ipc_lock=$IPC)"
for d in /d /c; do
  free_kb=$(df -k "$d" | awk 'NR==2{print $4}')
  echo "disk $d free: $((free_kb/1024/1024)) GB"
  if [ "$free_kb" -lt $((10*1024*1024)) ]; then echo "REFUSED: $d below the 10 GB gate"; exit 4; fi
done
mkdir -p "$REPO/$OUT_REL"
CAPS=()
if [ "$IPC" = yes ]; then CAPS=(--cap-add IPC_LOCK); fi
bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "HotPin lock verification $RUN_ID: llama-server starts on UD-Q3_K_M (4 threads) + 2 tuning prompts x 32 tokens per case; not a timing run" 7200 || exit 4
trap 'bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"; echo "=== $(date -Is) run_lock_verify.sh end"' EXIT
python -m pigemma.paths lock --hold --min-free-gb 16 -- \
  docker run --rm --name "pigemma-x1b-lockverify-$RUN_ID" \
    --cpuset-cpus 2,4,6,8 "${CAPS[@]}" --ulimit "memlock=$MEMLOCK" \
    -v "$REPO_W:/work" -v pigemma-models:/models:ro \
    -w /work pi-gemma4-build:fuse \
    python3 /work/pc/x1b/hotpin_lock_verify.py \
      --cases "/work/$CASES" \
      --model /models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
      --out "/work/$OUT_REL"
echo "lock verify exit: $?"
echo "LOCK_VERIFY_DONE"
