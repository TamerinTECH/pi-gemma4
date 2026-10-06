#!/usr/bin/env bash
# X1c pipeline (Git Bash): every CPU-heavy step of the stage under ONE hold of results/.cpu.lock, so the
# stage does not have to queue behind a timing stage several times.
#
#   1. E1  functional check, unconstrained: a7b94df reference, ik-plain, ik-prefetch-experts,
#          ik-plain without --swa-compress (2 tuning prompts x 32 greedy tokens, 3 runs each)
#   2.     evidence build of the trace-instrumented ik_llama.cpp server (-j 4)
#   3. E2  behaviour of --prefetch-experts under an 8 GiB cgroup-v1 limit (unthrottled volume):
#          instrumented build, unmodified ik-prefetch-experts under strace, ik-plain under strace
#   4.     ONE functional smoke cell in the controlled harness: ik-prefetch-experts:A, 8 GiB, FUSE 91.6 MB/s
#
# Steps 2-4 are skipped when E1 shows that the ik build cannot run the exact GGUF.
set -u
REPO="REPO_ROOT"
OWNER="X1c-ik-llama-x86-engine (wf_02d183f6-341)"
X="$REPO/results/comparisons/x1c"
LOG="$X/run_all.log"
export MSYS_NO_PATHCONV=1
cd "$REPO" || exit 1
exec >> "$LOG" 2>&1
echo "=== $(date -Is) run_all.sh start (steps: ${STEPS:=1,2,3,4})"
bash "$REPO/pc/x1a/cpu_lock.sh" acquire "$OWNER" "X1c: ik_llama.cpp functional check (4 server starts), instrumented evidence build (-j 4), prefetch behaviour check under 8 GiB (3 server starts), one smoke cell ik-prefetch-experts:A; about 35 min" 14400 || exit 4
trap 'bash "$REPO/pc/x1a/cpu_lock.sh" release "$OWNER"; echo "=== $(date -Is) run_all.sh end"' EXIT
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }

if has 1; then
  echo "--- $(date -Is) step 1: E1 functional check"
  NO_LOCK=1 bash "$REPO/pc/x1c/run_check.sh" e1-functional results/comparisons/x1c/engines_e1.json
  echo "step 1 exit: $?"
  python "$REPO/pc/x1c/compare_outputs.py" "$X/e1-functional/check.json" "$X/e1-functional/comparison.json" \
    --pairs "ik-plain:ik-prefetch-experts,ik-plain:ik-plain-no-swa-compress" || echo "compare failed"
fi
python - <<'PY'
import json, sys
try:
    d = json.load(open("results/comparisons/x1c/e1-functional/check.json", encoding="utf-8"))
    ok = {e["engine_id"]: bool(e.get("ok")) for e in d["engines"]}
except Exception as e:
    print("E1 result unreadable:", repr(e)); sys.exit(3)
print("E1 ok:", ok)
sys.exit(0 if ok.get("ik-plain") and ok.get("ik-prefetch-experts") else 3)
PY
if [ $? != 0 ]; then echo "STOP: ik engines did not pass E1 - steps 2-4 skipped"; exit 3; fi

if has 2; then
  echo "--- $(date -Is) step 2: instrumented evidence build"
  NO_LOCK=1 bash "$REPO/pc/x1c/build_instrumented.sh"
  echo "step 2 exit: $?"
fi

if has 3; then
  echo "--- $(date -Is) step 3: E2 prefetch behaviour under an 8 GiB cgroup limit"
  if [ -x "$REPO/third_party/ik_llama.cpp-x1c-instrumented/bin/llama-server-x1c-instrumented" ] || [ -f "$REPO/third_party/ik_llama.cpp-x1c-instrumented/bin/llama-server-x1c-instrumented" ]; then
    E2ENG=results/comparisons/x1c/engines_e2.json
  else
    echo "instrumented binary missing: running the strace engines only"
    python -c "
import json
d = json.load(open('results/comparisons/x1c/engines_e2.json'))
json.dump([e for e in d if 'instrumented' not in e['engine_id']], open('results/comparisons/x1c/engines_e2_strace_only.json', 'w'), indent=1)"
    E2ENG=results/comparisons/x1c/engines_e2_strace_only.json
  fi
  NO_LOCK=1 MEM=8g IMAGE=pi-gemma4-build:x1c-tools MODES=a,stream SKIP_SHA=1 MIN_AVAIL=10 \
    KIND="BEHAVIOUR CHECK under an 8 GiB cgroup-v1 memory limit (swap off), UNTHROTTLED storage (Docker volume), with instrumentation / strace overhead - not a benchmark, no speed claim" \
    bash "$REPO/pc/x1c/run_check.sh" e2-prefetch-8g "$E2ENG"
  echo "step 3 exit: $?"
  python "$REPO/pc/x1c/analyze_prefetch.py" --check "$X/e2-prefetch-8g/check.json" --map "$REPO/results/maps/map26_ud_q3_k_m.json" \
    --instr ik-prexp-instrumented --strace ik-prexp-strace,ik-plain-strace --out "$X/e2-prefetch-8g/prefetch_analysis.json" \
    > "$X/e2-prefetch-8g/prefetch_analysis.stdout.txt" 2>&1 || echo "analysis failed (see prefetch_analysis.stdout.txt)"
fi

if has 4; then
  echo "--- $(date -Is) step 4: one functional smoke cell ik-prefetch-experts:A (8 GiB, FUSE 91.6 MB/s)"
  python -m pigemma.controlled cell --method A --engine ik-prefetch-experts \
    --engines-file results/comparisons/x1c/engines_draft.json \
    --budget-gib 8 --workload tuning --conversations tun-01,tun-02 --n-predict-cap 16 \
    --purpose smoke --tag x1c-ik-prexp-smoke --cpu-lock-owner "$OWNER"
  echo "step 4 exit: $?"
fi
echo "ALL_STEPS_DONE"
