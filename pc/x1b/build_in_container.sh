#!/usr/bin/env bash
# X1b (comparisons workflow): build ONE llama.cpp tree as a Linux x86-64 CPU engine INSIDE the
# pi-gemma4-build container. Same configure flags as pc/x1a/build_engine_in_container.sh (the flags
# that reproduce the pinned a7b94df Linux build byte-for-byte), so hotpin-parent and hotpin differ
# only by the HotPin patches.
#
#   build_in_container.sh <SRC_DIR> <BUILD_DIR> <OUT_DIR> <LABEL>
#
# env: JOBS (default 4), TARGETS (space separated),
#      PATCH_FROM=<pristine parent tree, read-only>  + PATCHES="<a.diff> <b.diff>":
#        SRC_DIR is then (re)created as a copy of PATCH_FROM with the patches applied by `git apply`
#        (HotPin patches are AGPL-3.0: the patched tree stays on its Docker volume, only binaries,
#        logs and the resulting diff are copied to <OUT_DIR>, which carries the notices).
set -euo pipefail
SRC="${1:?src}"; BUILD="${2:?build}"; OUT="${3:?out}"; LABEL="${4:?label}"
JOBS="${JOBS:-4}"
read -r -a TARGETS <<< "${TARGETS:-llama-server llama-tokenize llama-bench llama-simple-chat}"
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
mkdir -p "$OUT/bin" "$OUT/logs" "$BUILD"
LOGDIR="$OUT/logs"
git config --global --add safe.directory '*'
PATCH_NOTE="none"
if [ -n "${PATCH_FROM:-}" ]; then
  echo "== $LABEL: creating patched tree $SRC from $PATCH_FROM"
  mkdir -p "$SRC"
  find "$SRC" -mindepth 1 -maxdepth 1 ! -name "$(basename "$BUILD")" -exec rm -rf {} +
  tar -C "$PATCH_FROM" --exclude=./build-linux-x86 -cf - . | tar -C "$SRC" -xf -
  git -C "$SRC" config core.fileMode false
  git -C "$SRC" config core.autocrlf false
  echo "pristine status lines before patching: $(git -C "$SRC" status --short | wc -l)" | tee "$LOGDIR/patch.log"
  for p in ${PATCHES:?PATCHES}; do
    echo "--- git apply $p (sha256 $(sha256sum "$p" | cut -d' ' -f1))" | tee -a "$LOGDIR/patch.log"
    git -C "$SRC" apply --verbose "$p" 2>&1 | tee -a "$LOGDIR/patch.log"
  done
  git -C "$SRC" status --short | tee -a "$LOGDIR/patch.log"
  git -C "$SRC" diff --stat | tee -a "$LOGDIR/patch.log"
  git -C "$SRC" diff > "$OUT/PATCHED_TREE.diff"
  PATCH_NOTE="$(for p in $PATCHES; do printf '%s sha256=%s; ' "$(basename "$p")" "$(sha256sum "$p" | cut -d' ' -f1)"; done)"
fi
REV=$(git -C "$SRC" rev-parse HEAD)
DIRTY=$(git -C "$SRC" -c core.fileMode=false status --short | wc -l)
echo "== $LABEL: src=$SRC rev=$REV dirty_files=$DIRTY build=$BUILD out=$OUT jobs=$JOBS targets=${TARGETS[*]}"
if [ "${CLEAN:-0}" = 1 ]; then rm -rf "${BUILD:?}"/* "${BUILD:?}"/.[!.]* 2>/dev/null || true; fi
t0=$(date +%s)
cmake -S "$SRC" -B "$BUILD" "${CMAKE_FLAGS[@]}" > "$LOGDIR/configure.log" 2>&1 || { tail -40 "$LOGDIR/configure.log"; echo "CONFIGURE FAILED"; exit 10; }
t1=$(date +%s)
nice -n 10 cmake --build "$BUILD" -j "$JOBS" --target "${TARGETS[@]}" > "$LOGDIR/build.log" 2>&1 || { tail -60 "$LOGDIR/build.log"; echo "BUILD FAILED"; exit 11; }
t2=$(date +%s)
for t in "${TARGETS[@]}"; do cp -f "$BUILD/bin/$t" "$OUT/bin/$t"; chmod +x "$OUT/bin/$t" || true; done
python3 - "$SRC" "$BUILD" "$OUT" "$LABEL" "$REV" "$((t1-t0))" "$((t2-t1))" "$JOBS" "${TARGETS[*]}" "${CMAKE_FLAGS[*]}" "$DIRTY" "$PATCH_NOTE" <<'PY'
import hashlib, json, os, re, subprocess, sys
src, build, out, label, rev, t_cfg, t_bld, jobs, targets, flags, dirty, patch_note = sys.argv[1:13]
def sh(*a):
    try:
        r = subprocess.run(a, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERR {e!r}"
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
    bins[t] = {"bytes": os.path.getsize(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
info = {
    "schema": "pigemma-x1b-build-info v1",
    "label": label,
    "source_dir_in_container": src,
    "revision": rev,
    "files_modified_vs_revision": int(dirty),
    "patches_applied": patch_note,
    "build_dir_in_container": build,
    "build_dir_storage": "Docker volume (ext4 in the Docker Desktop VM); only bin/, logs and (for the patched tree) PATCHED_TREE.diff are copied to the repo",
    "cmake_configure_flags": flags.split(),
    "cmake_build": f"nice -n 10 cmake --build <build> -j {jobs} --target {targets}",
    "targets": targets.split(),
    "configure_seconds": int(t_cfg), "build_seconds": int(t_bld), "jobs": int(jobs),
    "toolchain": {"gcc": sh("gcc", "--version").splitlines()[0], "gxx": sh("g++", "--version").splitlines()[0],
                  "cmake": sh("cmake", "--version").splitlines()[0], "ninja": sh("ninja", "--version"),
                  "image": os.environ.get("PIGEMMA_IMAGE", "pi-gemma4-build:latest"),
                  "march_native_resolves_to": sh("bash", "-c", "gcc -march=native -Q --help=target 2>/dev/null | grep -E '^[[:space:]]+-march=' | awk '{print $2}'"),
                  "cpu_model": sh("bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-")},
    "cmake_cache_selected": cache,
    "compile_command_ggml_cpu_c": cmd_for("ggml/src/ggml-cpu/ggml-cpu.c"),
    "compile_command_llama_model": cmd_for("src/llama-model.cpp"),
    "binaries": bins,
    "version_output": sh(os.path.join(out, "bin", "llama-server"), "--version"),
    "ldd_llama_server": sh("ldd", os.path.join(out, "bin", "llama-server")).splitlines(),
}
json.dump(info, open(os.path.join(out, "BUILD_INFO.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: info[k] for k in ("label", "revision", "files_modified_vs_revision", "configure_seconds", "build_seconds", "version_output")}, indent=1))
PY
echo "== $LABEL DONE: configure $((t1-t0)) s, build $((t2-t1)) s"
ls -la "$OUT/bin"
