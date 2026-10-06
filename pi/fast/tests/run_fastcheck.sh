#!/usr/bin/env bash
# pigemma-fast: API-level correctness scenarios on the Pi (E2B QAT Q4_0 by default).
# Builds pigemma-fastcheck twice: inside the patched tree and inside an UNPATCHED copy of the pinned source
# (the true reference), runs the scenarios in every mode and compares.
#   run_fastcheck.sh [patched-tree-name] [model.gguf] [tag]
# Not a timing run, but it still takes the bench lock (one job at a time). nice 10.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=/mnt/pigemma-fast
tree=$work/${1:-llama-fast-v4-fresh}
model=${2:-/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf}
tag=${3:-e2b}
ref=$work/llama-ref-check
out=$work/fastcheck-$tag; mkdir -p "$out"
src=~/pigemma/llama.cpp
base=a7b94df2c616bc1f62a73b964b4a71cb0dcc488e
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi job holds the lock"; exit 4; }
h() { echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') $(vcgencmd measure_temp) $(vcgencmd get_throttled) free=$(df -h --output=avail "$work" | tail -1)"; }
h
cfg="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF"
echo "=== build the tool in the patched tree $(date -Is)"
bash "$here/install_into_tree.sh" "$tree"
nice -n 10 cmake -S "$tree" -B "$tree/build" $cfg > "$out/cmake_fast.log" 2>&1 || { tail -5 "$out/cmake_fast.log"; exit 5; }
nice -n 10 cmake --build "$tree/build" -j 3 --target pigemma-fastcheck > "$out/build_fast.log" 2>&1 || { grep -E "error" -A5 "$out/build_fast.log" | head -30; exit 5; }
echo "=== build the tool in an unpatched reference copy $(date -Is)"
if [ ! -f "$ref/CMakeLists.txt" ]; then
  mkdir -p "$ref"; git -C "$src" archive "$base" | tar -x -C "$ref"
fi
[ ! -f "$ref/PIGEMMA_FAST_PATCHED" ] || { echo "reference copy is patched?!"; exit 3; }
bash "$here/install_into_tree.sh" "$ref"
nice -n 10 cmake -S "$ref" -B "$ref/build" $cfg > "$out/cmake_ref.log" 2>&1 || { tail -5 "$out/cmake_ref.log"; exit 5; }
nice -n 10 cmake --build "$ref/build" -j 3 --target pigemma-fastcheck > "$out/build_ref.log" 2>&1 || { grep -E "error" -A5 "$out/build_ref.log" | head -30; exit 5; }
FC_FAST=$tree/build/bin/pigemma-fastcheck
FC_REF=$ref/build/bin/pigemma-fastcheck
sha256sum "$FC_FAST" "$FC_REF" | cut -c1-90
h
run() {  # run <name> <binary> [VAR=value ...]
  local name=$1 bin=$2; shift 2
  echo "--- run $name ($*) $(date +%T)"
  env "$@" nice -n 10 "$bin" -m "$model" --text "$here/corpus.txt" --out "$out/$name.bin" -t 4 2> "$out/$name.err" || { tail -3 "$out/$name.err"; return 1; }
  tail -1 "$out/$name.err"
}
run REF  "$FC_REF"  PIGEMMA_NONE=0
run OFF  "$FC_FAST" PIGEMMA_NONE=0
run S    "$FC_FAST" PIGEMMA_STRIP_PREFILL=1
run P    "$FC_FAST" PIGEMMA_PAD_TAIL=1
run C    "$FC_FAST" PIGEMMA_CHUNK_KB=64
run SPC  "$FC_FAST" PIGEMMA_STRIP_PREFILL=1 PIGEMMA_PAD_TAIL=1 PIGEMMA_CHUNK_KB=64
h
cmp() {  # cmp <a> <b> [extra args]
  local a=$1 b=$2; shift 2
  echo "=== compare $a vs $b $*"
  "$FC_FAST" --compare "$out/$a.bin" "$out/$b.bin" --json "$out/${a}_vs_${b}.json" "$@" | grep -E "SUMMARY|max\|d\||DIFFER|MISSING|s7\." | cut -c1-230
}
cmp REF OFF --require-kv-equal --tol-kl 0 --tol-abs 0
cmp REF S   --require-kv-equal
cmp REF P
cmp REF C   --require-kv-equal
cmp REF SPC
echo "=== controls: the reference against itself"
for pair in "s3.n513.513.last s3.n513.511+2.last" "s3.n513.513.last s3.n513.512+1.last" "s2.b45.last s2.b1.last" "s2.b45.last s2.b3.last" "s2.b45.last s2.b7.last" "s4.reused.last s4.fresh.last"; do
  set -- $pair
  "$FC_FAST" --pair "$out/REF.bin" "$1" "$out/REF.bin" "$2"
done
echo "=== the same controls with all switches on"
for pair in "s3.n513.513.last s3.n513.511+2.last" "s2.b45.last s2.b3.last" "s2.b45.last s2.b7.last" "s4.reused.last s4.fresh.last"; do
  set -- $pair
  "$FC_FAST" --pair "$out/SPC.bin" "$1" "$out/SPC.bin" "$2"
done
mkdir -p ~/pigemma/results/fastcheck-$tag
cp "$out"/*.json ~/pigemma/results/fastcheck-$tag/ 2>/dev/null
(cd "$out" && sha256sum *.bin > SHA256SUMS-bin.txt && cp SHA256SUMS-bin.txt ~/pigemma/results/fastcheck-$tag/)
h
echo "=== FASTCHECK_DONE $(date -Is)"
