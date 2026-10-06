#!/usr/bin/env bash
# Desktop check of the two upstream-style patches against llama.cpp master 51ce9c1 (x86, inside the build container).
#   /src/ref  clean master + the test tool        /src/ls  master + gemma4-shared-kv-layer-skip.patch + tool
#   /src/pt   master + repack-pad-tail.patch + tool      /b  build volume      /m  models      /work  this repository
# Builds the three trees (incremental), runs the correctness scenarios (default, flash attention off, quantized KV),
# compares against the clean build, then measures prompt speed (desktop numbers, not Pi numbers).
set -uo pipefail
M=${MODEL:-/m/gemma-4-E2B_q4_0-it.gguf}
T=${THREADS:-6}
out=/b/check; mkdir -p "$out"
res=/work/results/desktop-fast/upstream-51ce9c1; mkdir -p "$res"
cfg="-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF"
for t in ref ls pt; do
  [ "${SKIP_BUILD:-0}" = 1 ] && continue
  t0=$(date +%s)
  cmake -S /src/$t -B /b/$t $cfg > /b/cmake_$t.log 2>&1 || { echo "cmake $t FAILED"; tail -15 /b/cmake_$t.log; exit 5; }
  cmake --build /b/$t -j 6 --target pigemma-fastcheck llama-server llama-bench > /b/build_$t.log 2>&1 \
    || { echo "build $t FAILED"; grep -E "error" -B2 -A8 /b/build_$t.log | head -60; exit 5; }
  echo "build $t OK $(( $(date +%s) - t0 )) s"
done
run() {  # run <name> <tree> [tool args]
  local name=$1 tree=$2; shift 2
  /b/$tree/bin/pigemma-fastcheck -m "$M" --text /work/pi/fast/tests/corpus.txt --out "$out/$name.bin" -t "$T" "$@" 2> "$out/$name.err" \
    || { echo "run $name FAILED"; tail -4 "$out/$name.err"; return 1; }
  echo "run $name: $(tail -1 "$out/$name.err" | cut -c1-120)"
}
cmp() {  # cmp <a> <b> [args]
  local a=$1 b=$2; shift 2
  echo "=== compare $a vs $b $*"
  /b/ls/bin/pigemma-fastcheck --compare "$out/$a.bin" "$out/$b.bin" --json "$res/${a}_vs_${b}.json" "$@" | grep -E "SUMMARY|DIFFER|MISSING" | cut -c1-260
}
if [ "${SKIP_RUNS:-0}" != 1 ]; then
run REF      ref
run LS       ls
run PT       pt
run REF_nofa ref --flash off
run LS_nofa  ls  --flash off
run REF_q8   ref --flash on --kv-type q8_0
run LS_q8    ls  --flash on --kv-type q8_0
cmp REF      LS      --require-kv-equal
cmp REF      PT
cmp REF_nofa LS_nofa --require-kv-equal
cmp REF_q8   LS_q8   --require-kv-equal
echo "=== controls: the clean build against itself (same tokens, other batch split)"
for pair in "s3.n513.513.last s3.n513.511+2.last" "s2.b45.last s2.b3.last" "s2.b45.last s2.b7.last" "s4.reused.last s4.fresh.last"; do
  set -- $pair
  /b/ls/bin/pigemma-fastcheck --pair "$out/REF.bin" "$1" "$out/REF.bin" "$2" | cut -c1-200
done
(cd "$out" && sha256sum *.bin > "$res/SHA256SUMS-bin.txt")
fi
[ "${SKIP_BENCH:-0}" = 1 ] && { echo "=== UPSTREAM_CHECK_DONE (correctness only)"; exit 0; }
echo "=== prompt speed, desktop x86, $T threads (llama-bench -p 512 -n 32 -r 3)"
for t in ref ls; do
  /b/$t/bin/llama-bench -m "$M" -t "$T" -p 512 -n 32 -r 3 -o md 2> /dev/null | tee "$res/bench_p512_$t.md" | grep -E "pp512|tg32" | sed "s/^/$t /" | cut -c1-200
done
echo "=== small batches, desktop x86 (llama-bench -p 420 -n 0 -b N)"
for t in ref pt; do
  /b/$t/bin/llama-bench -m "$M" -t "$T" -p 420 -n 0 -b 2,3,4,6,7,8 -r 2 -o md 2> /dev/null | tee "$res/bench_tail_$t.md" | grep -E "pp420" | sed "s/^/$t /" | cut -c1-200
done
echo "=== UPSTREAM_CHECK_DONE"
