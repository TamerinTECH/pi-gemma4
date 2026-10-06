#!/usr/bin/env bash
# Phase C (after phase B): 26B per-operation profile, from-scratch build of patch v7 in a new directory with the
# correctness scenarios, and the upstream-style patches on llama.cpp master on the Pi.
#   run_phase_c.sh [steps]     steps = comma list out of: attrib,profile26,fresh7,upstream (default: all)
# Every step takes the bench lock itself, so they run one after the other.
set -uo pipefail
cd ~/pigemma/fast
steps=",${1:-attrib,profile26,fresh7,upstream},"
want() { case "$steps" in *",$1,"*) return 0;; *) return 1;; esac; }
E2B=/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
if want attrib; then
  echo "=== attribution of the one differing answer $(date -Is)"
  ./run_attrib.sh
fi
if want profile26; then
  echo "=== profile26 $(date -Is)"
  ./run_26b_profile.sh
fi
if want fresh7; then
  echo "=== fresh build of patch v7 from the pinned source + patch file $(date -Is)"
  # room for the new trees: the v4 fresh-build tree is superseded (its binaries stay in bin-v4-fresh)
  rm -rf /mnt/pigemma-fast/llama-fast-v4-fresh /mnt/pigemma-fast/fastcheck-e2b /mnt/pigemma-fast/fastcheck-e2b-v7 /mnt/pigemma-fast/fastcheck-e4b-v7
  df -h --output=avail /mnt/pigemma-fast | tail -1
  nice -n 10 ./build_fast.sh llama-fast-v7-fresh bin-v7-fresh 2>&1 | tail -6
  echo "--- binaries: in-place upgraded tree (bin-v7) vs fresh tree (bin-v7-fresh)"
  (cd /mnt/usb/pigemma/fast && sha256sum bin-v7/llama-server bin-v7-fresh/llama-server bin-v7/llama-bench bin-v7-fresh/llama-bench | cut -c1-100)
  diff <(cd /mnt/pigemma-fast/llama-fast-v6 && find common src ggml/src tools/server -name "*.cpp" -o -name "*.h" | sort | xargs sha256sum) \
       <(cd /mnt/pigemma-fast/llama-fast-v7-fresh && find common src ggml/src tools/server -name "*.cpp" -o -name "*.h" | sort | xargs sha256sum) > /dev/null \
    && echo "SOURCES_IDENTICAL: upgraded tree == fresh tree" || echo "SOURCES_DIFFER"
  ./tests/run_fastcheck.sh llama-fast-v7-fresh "$E2B" e2b-v7-fresh 2>&1 | grep -E "SUMMARY|=== compare|FASTCHECK_DONE|REFUSED|FAILED|error" | cut -c1-260
  rm -f /mnt/pigemma-fast/fastcheck-e2b-v7-fresh/*.bin
fi
if want upstream; then
  echo "=== upstream-style patches on master $(date -Is)"
  ./upstream/run_upstream_pi.sh
fi
echo "=== PHASE_C_DONE $(date -Is)"
