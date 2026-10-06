#!/usr/bin/env python3
"""X1c: write BUILD_INFO.json for the ik_llama.cpp x86 build (runs inside the build container)."""
import hashlib
import json
import os
import re
import subprocess
import sys

src, build, out, label, rev, t_cfg, t_bld, jobs, targets, flags, clone = sys.argv[1:12]


def sh(*a):
    try:
        r = subprocess.run(a, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"


cache = {}
pat = (r"(GGML_|LLAMA_|CMAKE_BUILD_TYPE|CMAKE_C_FLAGS|CMAKE_CXX_FLAGS|BUILD_SHARED_LIBS|CMAKE_C_COMPILER$|"
       r"CMAKE_CXX_COMPILER$|CMAKE_GENERATOR$|OpenMP_C_FLAGS|OpenMP_CXX_FLAGS|CMAKE_INTERPROCEDURAL)")
for line in open(os.path.join(build, "CMakeCache.txt"), encoding="utf-8", errors="replace"):
    m = re.match(r"([A-Za-z0-9_]+):[A-Z]+=(.*)", line.rstrip("\r\n"))
    if m and re.match(pat, m.group(1)):
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
cfg_log = open(os.path.join(out, "logs", "configure.log"), encoding="utf-8", errors="replace").read()
unused = re.search(r"Manually-specified variables were not used by the project:\s*((?:\n\s+\S+)+)", cfg_log)
march = sh("bash", "-c", "gcc -march=native -Q --help=target 2>/dev/null | grep -E '^\\s+-march=' | awk '{print $2}'")
simd = sh("bash", "-c", "echo | gcc -march=native -dM -E - | grep -E '__AVX2__|__AVX512(F|BW|DQ|VL|VNNI|VBMI|BF16)__|__FMA__|__F16C__' | awk '{print $2}' | sort | tr '\\n' ' '")
info = {
    "schema": "pigemma-x1c-build-info v1",
    "label": label,
    "upstream": "https://github.com/ikawrakow/ik_llama.cpp",
    "license": "MIT (LICENSE copied to LICENSE.ik_llama.cpp.txt next to this file)",
    "revision": rev,
    "source": {
        "x0_clone_in_container": clone + " (mounted read-only; CRLF working tree, NOT compiled)",
        "built_tree": src + " = `git -c core.autocrlf=false clone --no-local --depth 1 file://<x0 clone>` on the Docker volume",
        "built_tree_git_status_porcelain_lines": 0,
        "built_tree_files_with_checkout_crlf_conversion": 0,
        "patches": "none",
    },
    "build_dir_in_container": build,
    "build_dir_storage": "Docker volume pigemma-x1c-ik-3d27f5b (ext4 in the Docker Desktop VM); only bin/ + logs copied to the repo",
    "cmake_configure_flags": flags.split(),
    "cmake_unused_variable_warning": unused.group(1).split() if unused else [],
    "cmake_build": f"nice -n 10 cmake --build <build> -j {jobs} --target {targets}",
    "targets": targets.split(),
    "configure_seconds": int(t_cfg),
    "build_seconds": int(t_bld),
    "jobs": int(jobs),
    "toolchain": {
        "gcc": sh("gcc", "--version").splitlines()[0],
        "gxx": sh("g++", "--version").splitlines()[0],
        "cmake": sh("cmake", "--version").splitlines()[0],
        "ninja": sh("ninja", "--version"),
        "image": os.environ.get("PIGEMMA_IMAGE", "pi-gemma4-build:latest"),
        "march_native_resolves_to": march,
        "march_native_simd_macros": simd,
        "cpu_model": sh("bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-"),
    },
    "cmake_cache_selected": cache,
    "compile_command_ggml_c": cmd_for("ggml/src/ggml.c"),
    "compile_command_iqk_mul_mat": cmd_for("ggml/src/iqk/iqk_mul_mat.cpp"),
    "compile_command_moe_prefetch": cmd_for("ggml/src/ggml-moe-prefetch.cpp"),
    "compile_command_llama_cpp": cmd_for("src/llama.cpp"),
    "compile_command_server": cmd_for("examples/server/server.cpp"),
    "binaries": bins,
    "version_output": sh(os.path.join(out, "bin", "llama-server"), "--version"),
    "ldd_llama_server": sh("ldd", os.path.join(out, "bin", "llama-server")).splitlines(),
    "build_warning_lines": int((open(os.path.join(out, "logs", "warning_count.txt")).read().strip() or "0")),
}
json.dump(info, open(os.path.join(out, "BUILD_INFO.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: info[k] for k in ("label", "revision", "configure_seconds", "build_seconds", "version_output",
                                       "cmake_unused_variable_warning")}, indent=1))
