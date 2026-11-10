#!/usr/bin/env bash
# X1a host wrapper (Git Bash): desktop-only functional check of the llama.cpp engines on the exact track-A GGUF.
# Holds results/.cpu.lock (CPU-heavy) and results/.bench.lock (model-loading job) for the duration.
# The model is read from the Docker volume pigemma-models (mounted READ-ONLY; byte-identical copy of
# models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf, sha256 checked in the run).
set -u
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
REPO_W="$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))"
OWNER="X1a-upstream-llama-recent (wf_02d183f6-341)"
OUT_REL="results/comparisons/x1a/functional"
RUN_ID="${1:-run1}"
LOG="$REPO/results/comparisons/x1a/functional_${RUN_ID}.log"
export MSYS_NO_PATHCONV=1
cd "$REPO" || exit 1
exec >> "$LOG" 2>&1
echo "=== $(date -Is) run_functional_check.sh $RUN_ID start"
for d in /d /c; do
  free_kb=$(df -k "$d" | awk 'NR==2{print $4}')
  echo "disk $d free: $((free_kb/1024/1024)) GB"
  if [ "$free_kb" -lt $((10*1024*1024)) ]; then echo "REFUSED: $d below the 10 GB gate"; exit 4; fi
done
mkdir -p "$REPO/$OUT_REL/$RUN_ID"
# the harness's own server configuration (pigemma.controlled.default_config) -> exact argv parity
python -c "
import json
from pigemma import controlled as c
cfg = c.default_config(method='A', budget_bytes=8*c.GIB, workload_set='tuning', repeat=0, cell_id='x1a-funccheck', purpose='smoke')
json.dump(cfg['server'], open(r'$OUT_REL/$RUN_ID/server_cfg.json', 'w', encoding='utf-8'), indent=1)
print('server cfg written; harness default bin =', cfg['server']['bin'])
" || { echo "server cfg generation failed"; exit 1; }
cat > "$REPO/$OUT_REL/$RUN_ID/engines_in.json" <<JSON
[
 {"engine_id": "llama.cpp-a7b94df-pinned",  "bin": "/work/third_party/llama.cpp/build-linux-x86/bin/llama-server"},
 {"engine_id": "llama.cpp-upstream-8345f33", "bin": "/work/third_party/llama.cpp-upstream-8345f33/build-linux-x86/bin/llama-server"},
 {"engine_id": "llama.cpp-a7b94df-pinned-restart", "bin": "/work/third_party/llama.cpp/build-linux-x86/bin/llama-server"}
]
JSON
bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "functional check: a7b94df, upstream 8345f33, a7b94df again (3 server starts) x 2 tuning prompts x 32 greedy tokens (UD-Q3_K_M, -nr, 4 threads)" 7200 || exit 4
trap 'bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"; echo "=== $(date -Is) run_functional_check.sh end"' EXIT
python -m pigemma.paths lock --hold --min-free-gb 16 -- \
  docker run --rm --name "pigemma-x1a-funccheck-$RUN_ID" \
    --cpuset-cpus 2,4,6,8 \
    -v "$REPO_W:/work" -v pigemma-models:/models:ro \
    -w /work pi-gemma4-build:fuse \
    python3 /work/pc/x1a/functional_check.py \
      --engines "/work/$OUT_REL/$RUN_ID/engines_in.json" \
      --server-cfg "/work/$OUT_REL/$RUN_ID/server_cfg.json" \
      --model /models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
      --model-sha256 811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052 \
      --conversations tun-01,tun-02 --n-predict 32 \
      --out "/work/$OUT_REL/$RUN_ID"
echo "functional check exit: $?"
echo "FUNCTIONAL_CHECK_DONE"
