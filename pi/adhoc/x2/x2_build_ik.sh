#!/usr/bin/env bash
# X2 (comparisons workflow, 2026-10-05): native build of ik_llama.cpp @3d27f5bb on the real Pi 5.
# Source + build tree live in /tmp (tmpfs = RAM) because / has only ~2.15 GB free (gate: >= 2 GB);
# only the two binaries (llama-server, llama-bench) are installed to ~/pigemma/ik_llama.cpp/build/bin.
# Flags equivalent to our a7b94df Pi build: Release, GGML_NATIVE=ON (gcc 14.2 resolves -march=native to
# -mcpu=cortex-a76+crc+crypto on this board), static libs, no curl, no tests. nice -n 10, -j 3.
# Holds ~/pigemma/.bench.lock for the whole build (no Pi benchmark may run concurrently).
# Guard: aborts (killing only its own build) if MemAvailable < 3 GB or a ContextFriend service is not active.
# Run: nohup flock -n ~/pigemma/.bench.lock bash x2_build_ik.sh > ~/pigemma/x2_build_ik.out 2>&1 &
set -uo pipefail
COMMIT=3d27f5bb6348c4d79edded63b03e9a93efb2f80e
WORK=/tmp/x2_ik_build
DEST=~/pigemma/ik_llama.cpp
SERVICES="contextfriend-box contextfriend-gpu-tunnel hailort whisper-bench@base"
echo "X2_BUILD_START $(date -Is) pid $$"
avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo); free=$(df -B1 --output=avail / | tail -1)
tmpfree=$(df -B1 --output=avail /tmp | tail -1)
echo "pre: MemAvailable_kB=$avail root_free=$free tmp_free=$tmpfree $(vcgencmd get_throttled) $(vcgencmd measure_temp)"
for s in $SERVICES; do echo "pre: $s $(systemctl is-active $s)"; done
[ "$avail" -gt 6291456 ] || { echo "ABORT: MemAvailable < 6 GiB before build"; exit 5; }
[ "$free" -gt 2000000000 ] || { echo "ABORT: root free < 2 GB"; exit 5; }
# match on the process NAME (comm), not the full command line, so wrapper shells that merely mention these names do not count
if pgrep "^(llama-|powerinfer)" >/dev/null; then echo "ABORT: other llama process running"; pgrep -a "^(llama-|powerinfer)"; exit 6; fi
mkdir -p "$WORK"
cd "$WORK"
if [ ! -d src/.git ]; then
  git init -q src
  git -C src remote add origin https://github.com/ikawrakow/ik_llama.cpp
fi
ok=0
for i in 1 2 3 4 5 6; do
  if ionice -c3 nice -n 10 git -C src fetch -q --depth 1 origin $COMMIT; then ok=1; break; fi
  echo "fetch attempt $i failed, retrying"; sleep 20
done
[ $ok = 1 ] || { echo "ABORT: git fetch failed"; exit 7; }
git -C src checkout -q FETCH_HEAD
echo "commit $(git -C src rev-parse HEAD) $(git -C src log -1 --format=%cI)"
du -sh src
# guard loop (kills only this script's build processes)
(
  while sleep 10; do
    a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    t=$(vcgencmd measure_temp | tr -d "temp='C")
    echo "mon $(date +%T) MemAvailable_kB=$a temp=$t tmp_used=$(du -sm $WORK 2>/dev/null | cut -f1)MB"
    bad=0; for s in $SERVICES; do [ "$(systemctl is-active $s)" = active ] || bad=1; done
    if [ "$a" -lt 3145728 ] || [ $bad = 1 ]; then
      echo "GUARD_ABORT MemAvailable_kB=$a services_bad=$bad"; pkill -TERM -P $$ -f "cmake --build" ; pkill -TERM -f "$WORK/build" ; touch $WORK/GUARD_ABORT; exit 0
    fi
    [ -e $WORK/BUILD_FINISHED ] && exit 0
  done
) &
MON=$!
echo "native flags: $(gcc -march=native -Q --help=target 2>/dev/null | grep -E "^[[:space:]]+-m(arch|cpu|tune)=" | tr -s "[:space:]" " ")"
start=$(date +%s)
nice -n 10 cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > cmake.log 2>&1
rc_cfg=$?
grep -iE "ARM|OpenMP|native|mcpu|march|IQK|compiler identification|BUILD_TYPE|warning" cmake.log | head -30
if [ $rc_cfg = 0 ]; then
  nice -n 10 cmake --build build -j 3 --target llama-server llama-bench > build.log 2>&1
  rc_build=$?
else
  rc_build=99
fi
end=$(date +%s)
touch $WORK/BUILD_FINISHED
wait $MON 2>/dev/null
echo "cmake_rc=$rc_cfg build_rc=$rc_build build_seconds=$((end-start))"
tail -15 build.log
grep -m1 -E '"command".*ggml\.c"?' build/compile_commands.json 2>/dev/null | cut -c1-900
grep -m1 -E '"command".*iqk_mul_mat\.cpp' build/compile_commands.json 2>/dev/null | cut -c1-900
# exact compile flags of one ggml source
grep -rh "C_FLAGS\|CXX_FLAGS" build/ggml/src/CMakeFiles/ggml.dir/flags.make 2>/dev/null | head -4
mkdir -p "$DEST/build/bin"
cp cmake.log build.log "$DEST/" 2>/dev/null
if [ $rc_build = 0 ] && [ ! -e $WORK/GUARD_ABORT ]; then
  ls -la build/bin/llama-server build/bin/llama-bench
  binsz=$(( $(stat -c %s build/bin/llama-server) + $(stat -c %s build/bin/llama-bench) ))
  rootfree=$(df -B1 --output=avail / | tail -1)
  if [ $((rootfree - binsz)) -lt 2050000000 ]; then
    echo "binaries ($binsz B) would leave < 2.05 GB on / -> stripping"; strip build/bin/llama-server build/bin/llama-bench
  fi
  cp build/bin/llama-server build/bin/llama-bench "$DEST/build/bin/"
  ls -la "$DEST/build/bin"
  sha256sum "$DEST"/build/bin/* > "$DEST/BIN_SHA256SUMS"
  cat "$DEST/BIN_SHA256SUMS"
  "$DEST/build/bin/llama-server" --version 2>&1 | tail -3
  {
    echo "ik_llama.cpp commit: $(git -C src rev-parse HEAD) ($(git -C src log -1 --format=%cI))"
    echo "built: $(date -Is) on $(hostname) $(uname -r), $(gcc --version | head -1), $(cmake --version | head -1)"
    echo "cmake: -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    echo "gcc -march=native resolves to: $(gcc -march=native -Q --help=target 2>/dev/null | grep -E '^\s+-m(arch|cpu|tune)=' | tr -s ' 	' ' ' | tr '
' ';')"
    echo "compile command (ggml.c):"; grep -m1 -E '"command".*ggml\.c"?' build/compile_commands.json 2>/dev/null
    echo "compile command (iqk_mul_mat.cpp):"; grep -m1 -E '"command".*iqk_mul_mat\.cpp' build/compile_commands.json 2>/dev/null
    echo "targets: llama-server llama-bench (-j 3, nice -n 10); build_seconds=$((end-start))"
    echo "source + objects were in /tmp/x2_ik_build (tmpfs) and were deleted after the build"
    echo "flags.make:"; grep -h "FLAGS" build/ggml/src/CMakeFiles/ggml.dir/flags.make 2>/dev/null
  } > "$DEST/BUILD_INFO.txt"
fi
cd /
rm -rf "$WORK"
echo "post: MemAvailable_kB=$(awk '/MemAvailable/{print $2}' /proc/meminfo) root_free=$(df -B1 --output=avail / | tail -1) $(vcgencmd get_throttled) $(vcgencmd measure_temp)"
for s in $SERVICES; do echo "post: $s $(systemctl is-active $s)"; done
echo "X2_BUILD_DONE rc=$rc_build $(date -Is)"
