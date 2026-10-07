#!/usr/bin/env bash
# Pi 5: a new version of the padded-tail patch on llama.cpp master. Re-patches the existing tree
# /mnt/pigemma-fast/up/pt, rebuilds it, repeats the correctness comparison against the clean tree and the
# small-batch benchmark for both trees.   run_upstream_pi_pt2.sh [tag]   (results: fast3/upstream-pi-<tag>)
#   needs ~/pigemma/fast/upstream/repack-pad-tail.applied.patch (what the tree has now) and repack-pad-tail.patch (new)
set -uo pipefail
cd ~/pigemma
exec 9> ~/pigemma/.bench.lock
flock -n 9 || { echo "REFUSED: another Pi job holds the lock"; exit 4; }
work=/mnt/pigemma-fast/up
here=~/pigemma/fast/upstream
M=/mnt/usb/pigemma/models/e2b-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
tag=${1:-pt3}
out=$work/check; res=~/pigemma/results/fast3/upstream-pi-$tag; mkdir -p "$out" "$res"
h() { echo "[health $(date +%T)] $(systemctl is-active contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base | tr '\n' ' ') $(vcgencmd measure_temp) $(vcgencmd get_throttled)"; }
h
cd "$work/pt" || exit 5
# the tree carries the patch saved as repack-pad-tail.applied.patch; replace it by the current repack-pad-tail.patch
if ! cmp -s "$here/repack-pad-tail.applied.patch" "$here/repack-pad-tail.patch"; then
  patch -R -p1 -s < "$here/repack-pad-tail.applied.patch" || { echo "reverse patch FAILED"; exit 5; }
  cmp -s ggml/src/ggml-cpu/repack.cpp "$work/ref/ggml/src/ggml-cpu/repack.cpp" || { echo "tree is not clean after the reverse patch"; exit 5; }
  patch -p1 -s < "$here/repack-pad-tail.patch" || { echo "patch FAILED"; exit 5; }
  cp "$here/repack-pad-tail.patch" "$here/repack-pad-tail.applied.patch"
fi
sha256sum "$here/repack-pad-tail.patch" | cut -c1-64
t0=$(date +%s)
nice -n 10 cmake --build "$work/pt/build" -j 3 --target pigemma-fastcheck llama-bench > "$out/build_pt2.log" 2>&1 \
  || { echo "build pt FAILED"; grep -E "error" -A6 "$out/build_pt2.log" | head -40; exit 5; }
echo "build pt OK $(( $(date +%s) - t0 )) s"
cd ~/pigemma
for t in ref pt; do
  nice -n 10 "$work/$t/build/bin/pigemma-fastcheck" -m "$M" --text ~/pigemma/fast/tests/corpus.txt --out "$out/${t}2.bin" -t 4 2> "$out/${t}2.err" \
    || { echo "run $t FAILED"; tail -4 "$out/${t}2.err"; exit 5; }
done
echo "=== compare clean master vs padded tail ($tag)"
"$work/pt/build/bin/pigemma-fastcheck" --compare "$out/ref2.bin" "$out/pt2.bin" --json "$res/REF_vs_PT2.json" | grep -E "SUMMARY|DIFFER|MISSING" | cut -c1-260
(cd "$out" && sha256sum ref2.bin pt2.bin > "$res/SHA256SUMS-bin.txt"; rm -f ref2.bin pt2.bin)
h
echo "=== small batches (llama-bench -p 420 -n 0 -b N), clean master vs padded tail, two alternated rounds"
for round in 1 2; do
  for t in ref pt; do
    nice -n 10 "$work/$t/build/bin/llama-bench" -m "$M" -t 4 -p 420 -n 0 -b 2,3,4,5,6,7,8 -r 2 -o md 2> /dev/null | tee "$res/bench_tail_${t}_r$round.md" | grep -E "pp420" | sed "s/^/$t r$round /" | cut -c1-200
  done
done
h
echo "=== UPSTREAM_PI_PT2_DONE $(date -Is)"
