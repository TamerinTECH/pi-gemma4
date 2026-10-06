#!/usr/bin/env bash
# X1a (comparisons workflow): build one llama.cpp revision as a Linux x86-64 CPU engine INSIDE the
# pi-gemma4-build container. Runs with the source tree mounted READ-ONLY, the build directory on a
# Docker volume (fast ext4, no D: usage) and only the selected binaries copied to /out.
#
#   build_engine_in_container.sh <SRC_DIR> <BUILD_DIR> <OUT_DIR> <LABEL>
#
# Flags are identical for every revision built by this script (recorded in <OUT_DIR>/BUILD_INFO.json).
# They mirror the observable properties of the pre-existing a7b94df Linux build
# (third_party/llama.cpp/build-linux-x86/bin, whose CMakeCache was not kept): Release, libllama/ggml
# linked statically into each tool (BUILD_SHARED_LIBS=OFF; glibc/libstdc++/libgomp stay dynamic),
# CPU backend only, GGML_NATIVE=ON (-march=native = znver4 on this desktop: AVX-512 incl. VBMI/VNNI/BF16),
# OpenMP on, no OpenSSL (libssl-dev is not in the image), no embedded web UI.
set -euo pipefail
SRC="${1:?src}"; BUILD="${2:?build}"; OUT="${3:?out}"; LABEL="${4:?label}"
JOBS="${JOBS:-4}"
TARGETS=(llama-server llama-cli llama-completion llama-bench llama-perplexity llama-tokenize)
CMAKE_FLAGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DGGML_NATIVE=ON
  -DGGML_OPENMP=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_OPENSSL=OFF
  -DLLAMA_BUILD_UI=OFF
  -DLLAMA_USE_PREBUILT_UI=OFF
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)
mkdir -p "$OUT/bin" "$BUILD"
LOGDIR="$OUT/logs"; mkdir -p "$LOGDIR"
git config --global --add safe.directory '*'
REV=$(git -C "$SRC" rev-parse HEAD)
echo "== $LABEL: src=$SRC rev=$REV build=$BUILD out=$OUT jobs=$JOBS"
if [ "${CLEAN:-0}" = 1 ]; then rm -rf "${BUILD:?}"/* "${BUILD:?}"/.[!.]* 2>/dev/null || true; fi
t0=$(date +%s)
cmake -S "$SRC" -B "$BUILD" "${CMAKE_FLAGS[@]}" > "$LOGDIR/configure.log" 2>&1 || { tail -40 "$LOGDIR/configure.log"; echo "CONFIGURE FAILED"; exit 10; }
t1=$(date +%s)
nice -n 10 cmake --build "$BUILD" -j "$JOBS" --target "${TARGETS[@]}" > "$LOGDIR/build.log" 2>&1 || { tail -60 "$LOGDIR/build.log"; echo "BUILD FAILED"; exit 11; }
t2=$(date +%s)
for t in "${TARGETS[@]}"; do cp -f "$BUILD/bin/$t" "$OUT/bin/$t"; chmod +x "$OUT/bin/$t" || true; done
# evidence: effective compile command of one ggml-cpu kernel file + cache values
python3 - "$SRC" "$BUILD" "$OUT" "$LABEL" "$REV" "$((t1-t0))" "$((t2-t1))" "$JOBS" "${TARGETS[*]}" "${CMAKE_FLAGS[*]}" <<'PY'
import hashlib, json, os, re, subprocess, sys
src, build, out, label, rev, t_cfg, t_bld, jobs, targets, flags = sys.argv[1:11]
def sh(*a):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERR {e!r}"
cache = {}
for line in open(os.path.join(build, "CMakeCache.txt"), encoding="utf-8", errors="replace"):
    m = re.match(r"([A-Za-z0-9_]+):[A-Z]+=(.*)", line.rstrip("\r\n"))
    if m and re.match(r"(GGML_|LLAMA_|CMAKE_BUILD_TYPE|CMAKE_C_FLAGS|CMAKE_CXX_FLAGS|BUILD_SHARED_LIBS|CMAKE_C_COMPILER$|CMAKE_CXX_COMPILER$|CMAKE_GENERATOR$|OpenMP_C_FLAGS|OpenMP_CXX_FLAGS|CMAKE_INTERPROCEDURAL)", m.group(1)):
        cache[m.group(1)] = m.group(2)
cc = json.load(open(os.path.join(build, "compile_commands.json"), encoding="utf-8"))
def cmd_for(suffix):
    for e in cc:
        if e["file"].replace("\\", "/").endswith(suffix):
            c = e.get("command") or " ".join(e.get("arguments", []))
            return re.sub(r"\s-o\s+\S+\s+-c\s+\S+", " -o <obj> -c <src>", c)
    return None
bins = {}
for t in targets.split():
    p = os.path.join(out, "bin", t)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    bins[t] = {"bytes": os.path.getsize(p), "sha256": h}
info = {
    "schema": "pigemma-x1a-build-info v1",
    "label": label,
    "source_dir_in_container": src,
    "source_mounted_read_only": True,
    "revision": rev,
    "git_describe_note": "shallow clone: llama.cpp reports 'build 1'",
    "build_dir_in_container": build,
    "build_dir_storage": "Docker volume (ext4 in the Docker Desktop VM); only bin/ + logs copied to the repo",
    "cmake_configure_flags": flags.split(),
    "cmake_build": f"nice -n 10 cmake --build <build> -j {jobs} --target {targets}",
    "targets": targets.split(),
    "configure_seconds": int(t_cfg), "build_seconds": int(t_bld), "jobs": int(jobs),
    "toolchain": {"gcc": sh("gcc", "--version").splitlines()[0], "gxx": sh("g++", "--version").splitlines()[0],
                  "cmake": sh("cmake", "--version").splitlines()[0], "ninja": sh("ninja", "--version"),
                  "image": os.environ.get("PIGEMMA_IMAGE", "pi-gemma4-build:latest"),
                  "march_native_resolves_to": sh("bash", "-c", "gcc -march=native -Q --help=target 2>/dev/null | grep -E '^\s+-march=' | awk '{print $2}'"),
                  "cpu_model": sh("bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-")},
    "cmake_cache_selected": cache,
    "compile_command_ggml_cpu_c": cmd_for("ggml/src/ggml-cpu/ggml-cpu.c"),
    "compile_command_sgemm": cmd_for("ggml/src/ggml-cpu/llamafile/sgemm.cpp"),
    "compile_command_llama_model": cmd_for("src/llama-model.cpp"),
    "binaries": bins,
    "version_output": sh(os.path.join(out, "bin", "llama-server"), "--version") or subprocess.run([os.path.join(out, "bin", "llama-server"), "--version"], capture_output=True, text=True).stderr.strip(),
    "ldd_llama_server": sh("ldd", os.path.join(out, "bin", "llama-server")).splitlines(),
}
json.dump(info, open(os.path.join(out, "BUILD_INFO.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: info[k] for k in ("label", "revision", "configure_seconds", "build_seconds", "version_output")}, indent=1))
PY
echo "== $LABEL DONE: configure $((t1-t0)) s, build $((t2-t1)) s"
ls -la "$OUT/bin"
