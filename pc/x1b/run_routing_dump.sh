#!/usr/bin/env bash
# X1b host wrapper (Git Bash): HotPin's own routing profiling on data/workloads/calibration.jsonl ONLY.
#   run_routing_dump.sh <tag> <llama-simple-chat path inside the container> [extra driver args]
# Holds results/.cpu.lock (CPU-heavy) and results/.bench.lock (model-loading job). 4 threads, cpuset 2,4,6,8.
set -u
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
OWNER="X1b-hotpin (wf_02d183f6-341)"
TAG="${1:?tag}"; EXE="${2:?exe}"; shift 2
OUT_REL="results/comparisons/x1b/routing"
LOG="$REPO/results/comparisons/x1b/routing_dump_${TAG}.log"
export MSYS_NO_PATHCONV=1
cd "$REPO" || exit 1
exec >> "$LOG" 2>&1
echo "=== $(date -Is) run_routing_dump.sh $TAG start"
for d in /d /c; do
  free_kb=$(df -k "$d" | awk 'NR==2{print $4}')
  echo "disk $d free: $((free_kb/1024/1024)) GB"
  if [ "$free_kb" -lt $((10*1024*1024)) ]; then echo "REFUSED: $d below the 10 GB gate"; exit 4; fi
done
mkdir -p "$REPO/$OUT_REL"
bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "HotPin routing profiling ($TAG): patched llama-simple-chat --dump-routing on calibration.jsonl (32 requests, 4 threads; not a timing run)" 7200 || exit 4
trap 'bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"; echo "=== $(date -Is) run_routing_dump.sh end"' EXIT
python -m pigemma.paths lock --hold --min-free-gb 16 -- \
  docker run --rm --name "pigemma-x1b-routing-$TAG" \
    --cpuset-cpus 2,4,6,8 \
    -v "$REPO_W:/work" -v pigemma-models:/models:ro \
    -w /work pi-gemma4-build:fuse \
    python3 /work/pc/x1b/hotpin_routing_dump.py \
      --exe "$EXE" \
      --model /models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
      --out "/work/$OUT_REL" --tag "$TAG" --threads 4 --ctx 4096 "$@"
echo "routing dump exit: $?"
echo "ROUTING_DUMP_DONE"
