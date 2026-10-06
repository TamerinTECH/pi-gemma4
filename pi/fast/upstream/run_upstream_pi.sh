#!/usr/bin/env bash
# Pi 5: the two upstream-style patches on llama.cpp master 51ce9c1 (ARM numbers for the notes in README.md).
# Downloads the source archive of the revision from GitHub if /mnt/pigemma-fast/up/master.tar.gz is missing; needs
# the two patch files in ~/pigemma/fast/upstream and the test tool in ~/pigemma/fast/tests. Three trees: ref (clean), ls (layer skip), pt (padded tail).
# Builds only the test tool and llama-bench. One job at a time (bench lock), nice 10.
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi job holds the lock"; exit 4; }
work=/mnt/pigemma-fast/up
here=~/pigemma/fast/upstream
M=/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
out=$work/check; res=~/pigemma/results/fast3/upstream-pi; mkdir -p "$out" "$res"
h() { echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') $(vcgencmd measure_temp) $(vcgencmd get_throttled) free=$(df -h --output=avail "$work" | tail -1)"; }
h
cfg="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF"
rev=51ce9c11a6f2dfa895696c0048c4333e8953b728
if [ ! -s "$work/master.tar.gz" ]; then
  mkdir -p "$work"
  curl -sSL --max-time 600 -o "$work/master.tar.gz" "https://github.com/ggml-org/llama.cpp/archive/$rev.tar.gz" || { echo "download FAILED"; exit 5; }
fi
sha256sum "$work/master.tar.gz" | cut -c1-64
strip=0; tar -tzf "$work/master.tar.gz" | head -1 | grep -q "^llama.cpp-" && strip=1
for t in ref ls pt; do
  if [ ! -f "$work/$t/CMakeLists.txt" ]; then
    mkdir -p "$work/$t" && tar -xzf "$work/master.tar.gz" -C "$work/$t" --strip-components=$strip || { echo "extract $t FAILED"; exit 5; }
    [ "$t" = ls ] && { (cd "$work/$t" && patch -p1 -s < "$here/gemma4-shared-kv-layer-skip.patch") || { echo "patch ls FAILED"; exit 5; }; }
    [ "$t" = pt ] && { (cd "$work/$t" && patch -p1 -s < "$here/repack-pad-tail.patch") || { echo "patch pt FAILED"; exit 5; }; }
    bash ~/pigemma/fast/tests/install_into_tree.sh "$work/$t" > /dev/null
  fi
  t0=$(date +%s)
  nice -n 10 cmake -S "$work/$t" -B "$work/$t/build" $cfg > "$out/cmake_$t.log" 2>&1 || { echo "cmake $t FAILED"; tail -5 "$out/cmake_$t.log"; exit 5; }
  nice -n 10 cmake --build "$work/$t/build" -j 3 --target pigemma-fastcheck llama-bench > "$out/build_$t.log" 2>&1 \
    || { echo "build $t FAILED"; grep -E "error" -A6 "$out/build_$t.log" | head -40; exit 5; }
  echo "build $t OK $(( $(date +%s) - t0 )) s"
  h
done
run() {  # run <name> <tree> [tool args]
  local name=$1 tree=$2; shift 2
  nice -n 10 "$work/$tree/build/bin/pigemma-fastcheck" -m "$M" --text ~/pigemma/fast/tests/corpus.txt --out "$out/$name.bin" -t 4 "$@" 2> "$out/$name.err" \
    || { echo "run $name FAILED"; tail -4 "$out/$name.err"; return 1; }
  echo "run $name: $(tail -1 "$out/$name.err" | cut -c1-100)"
}
cmp() {  # cmp <a> <b> [args]
  local a=$1 b=$2; shift 2
  echo "=== compare $a vs $b $*"
  "$work/ls/build/bin/pigemma-fastcheck" --compare "$out/$a.bin" "$out/$b.bin" --json "$res/${a}_vs_${b}.json" "$@" | grep -E "SUMMARY|DIFFER|MISSING" | cut -c1-260
}
run REF ref
run LS  ls
run PT  pt
cmp REF LS --require-kv-equal
cmp REF PT
echo "=== controls: the clean build against itself"
for pair in "s3.n513.513.last s3.n513.511+2.last" "s2.b45.last s2.b3.last"; do
  set -- $pair
  "$work/ls/build/bin/pigemma-fastcheck" --pair "$out/REF.bin" "$1" "$out/REF.bin" "$2" | cut -c1-200
done
(cd "$out" && sha256sum *.bin > "$res/SHA256SUMS-bin.txt"; rm -f *.bin)
h
echo "=== prompt speed and decode (llama-bench -p 512 -n 32 -r 3), clean master vs layer skip"
for t in ref ls; do
  nice -n 10 "$work/$t/build/bin/llama-bench" -m "$M" -t 4 -p 512 -n 32 -r 3 -o md 2> /dev/null | tee "$res/bench_p512_$t.md" | grep -E "pp512|tg32" | sed "s/^/$t /" | cut -c1-200
done
h
echo "=== small batches (llama-bench -p 420 -n 0 -b N), clean master vs padded tail"
for t in ref pt; do
  nice -n 10 "$work/$t/build/bin/llama-bench" -m "$M" -t 4 -p 420 -n 0 -b 2,3,4,5,6,7,8 -r 2 -o md 2> /dev/null | tee "$res/bench_tail_$t.md" | grep -E "pp420" | sed "s/^/$t /" | cut -c1-200
done
h
echo "=== UPSTREAM_PI_DONE $(date -Is)"
