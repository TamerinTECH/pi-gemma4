#!/usr/bin/env bash
# X1c host wrapper (Git Bash): run pc/x1c/ik_check.py in a container under results/.cpu.lock and
# results/.bench.lock (model-loading job).
#
#   run_check.sh <RUN_ID> <ENGINES_JSON (repo-relative)> [options]
#     MEM=8g            memory cgroup limit of the container (default: none = unconstrained)
#     IMAGE=...         default pi-gemma4-build:fuse ; pi-gemma4-build:x1c-tools for strace runs
#     MODES=a,b,stream  request modes
#     CONVS=tun-01,tun-02
#     NPRED=32
#     SKIP_SHA=1        do not re-hash the model (hashing also warms the page cache)
#     MIN_AVAIL=16      refuse below this MemAvailable (GiB) in the Docker VM
#     KIND="..."        label stored in check.json
#
# The model is read from the Docker volume pigemma-models (read-only; same sha256 as
# models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf on D:).
set -u
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
OWNER="X1c-ik-llama-x86-engine (wf_02d183f6-341)"
RUN_ID="${1:?run id}"; ENGINES_REL="${2:?engines json}"
OUT_REL="results/comparisons/x1c/$RUN_ID"
IMAGE="${IMAGE:-pi-gemma4-build:fuse}"
MODES="${MODES:-a,b,stream}"; CONVS="${CONVS:-tun-01,tun-02}"; NPRED="${NPRED:-32}"
MIN_AVAIL="${MIN_AVAIL:-16}"
KIND="${KIND:-FUNCTIONAL CHECK - not a benchmark (no memory budget, no storage throttle, desktop not verified idle)}"
LOG="$REPO/results/comparisons/x1c/${RUN_ID}.log"
export MSYS_NO_PATHCONV=1
cd "$REPO" || exit 1
mkdir -p "$REPO/$OUT_REL"
exec >> "$LOG" 2>&1
echo "=== $(date -Is) run_check.sh $RUN_ID start (image=$IMAGE mem=${MEM:-none} modes=$MODES convs=$CONVS)"
for d in /d /c; do
  free_kb=$(df -k "$d" | awk 'NR==2{print $4}')
  echo "disk $d free: $((free_kb/1024/1024)) GB"
  if [ "$free_kb" -lt $((10*1024*1024)) ]; then echo "REFUSED: $d below the 10 GB gate"; exit 4; fi
done
python -c "
import json
from pigemma import controlled as c
cfg = c.default_config(method='A', budget_bytes=8*c.GIB, workload_set='tuning', repeat=0, cell_id='x1c-check', purpose='smoke')
json.dump(cfg['server'], open(r'$OUT_REL/server_cfg.json', 'w', encoding='utf-8'), indent=1)
print('server cfg written; harness default bin =', cfg['server']['bin'])
" || { echo "server cfg generation failed"; exit 1; }
cp -f "$REPO/$ENGINES_REL" "$REPO/$OUT_REL/engines_in.json"
MEMARGS=()
if [ -n "${MEM:-}" ]; then MEMARGS=(--memory="$MEM" --memory-swap="$MEM" --memory-swappiness=0); fi
SHAARGS=(--model-sha256 811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052)
if [ "${SKIP_SHA:-0}" = 1 ]; then SHAARGS=(--skip-sha256); fi
if [ "${NO_LOCK:-0}" != 1 ]; then   # NO_LOCK=1: the caller (pc/x1c/run_all.sh) already holds results/.cpu.lock
  bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "ik_llama.cpp check $RUN_ID: server starts on UD-Q3_K_M (4 threads), $CONVS x $NPRED greedy tokens" 7200 || exit 4
  trap 'bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"; echo "=== $(date -Is) run_check.sh end"' EXIT
fi
python -m pigemma.paths lock --hold --min-free-gb 10 -- \
  docker run --rm --name "pigemma-x1c-check-$RUN_ID" \
    --cpuset-cpus 2,4,6,8 --cpus=4 "${MEMARGS[@]}" \
    --cap-add SYS_PTRACE --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
    --network none \
    -v "$REPO_W:/work" -v pigemma-models:/models:ro ${EXTRA_VOL:+-v "$EXTRA_VOL"} \
    -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1 -w /work "$IMAGE" \
    python3 /work/pc/x1c/ik_check.py \
      --engines "/work/$OUT_REL/engines_in.json" \
      --server-cfg "/work/$OUT_REL/server_cfg.json" \
      --model /models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
      "${SHAARGS[@]}" \
      --conversations "$CONVS" --n-predict "$NPRED" --modes "$MODES" \
      --min-avail-gib "$MIN_AVAIL" --kind "$KIND" \
      --out "/work/$OUT_REL"
rc=$?
echo "check exit: $rc"
echo "CHECK_DONE"
exit $rc
