# =============================================================================
# pigemma-tools: out-of-tree CMake project for the three small native tools of
# the pi-gemma4 experiment, built against the PINNED llama.cpp checkout without
# modifying it (llama.cpp is pulled in with add_subdirectory(... EXCLUDE_FROM_ALL),
# so only the libraries our tools link are compiled).
#
#   pigemma-probe  (probe/probe.cpp,  C++17)  cb_eval recorder: names | moe-trace | act-dump
#                                             links llama-common + llama (same as
#                                             llama.cpp examples/eval-callback)
#   pinner         (pinner/pinner.c,  C11)    hot-set mlock daemon - Linux only
#   membw          (membw/membw.c,    C11)    memory-bandwidth micro-bench (pthreads)
#
# All binaries land in ${CMAKE_BINARY_DIR}/bin. A provenance file
# ${CMAKE_BINARY_DIR}/bin/pigemma-tools-build.json is written at configure time
# (llama.cpp commit + build number, compiler, ggml arch flags, configured targets).
#
# -----------------------------------------------------------------------------
# Invocation 1 - desktop, inside Docker (x86-64; image pi-gemma4-build:latest).
# GGML_NATIVE is OFF so the binaries do not depend on the build host's CPU;
# AVX2 + AVX-512 match the Ryzen 5 8400F test box. pc/docker_run.sh build-tools
# runs exactly this (from Git Bash; MSYS_NO_PATHCONV=1 stops path mangling):
#
#   MSYS_NO_PATHCONV=1 docker run --rm -v "REPO_ROOT:/work" \
#     pi-gemma4-build:latest bash -c '
#       cmake -S /work/tools -B /work/tools/build-linux-x86 -G Ninja \
#             -DLLAMA_DIR=/work/third_party/llama.cpp \
#             -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=ON \
#             -DCMAKE_BUILD_TYPE=Release &&
#       cmake --build /work/tools/build-linux-x86 -j 12 --target pigemma-probe pinner membw'
#
# Invocation 2 - Raspberry Pi 5 (Debian 13 aarch64, gcc 14, cmake 3.31); this is
# what pi/02_build_llama.sh --with-tools runs (LLAMA_DIR = its own checkout at the
# pinned commit). GGML_NATIVE=ON lets ggml detect dotprod + fp16 on the Cortex-A76:
#
#   cmake -S "$PIGEMMA_HOME/tools" -B "$PIGEMMA_HOME/tools/build" \
#         -DLLAMA_DIR="$LLAMA_DIR" -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
#   nice -n 10 cmake --build "$PIGEMMA_HOME/tools/build" -j 2 --target pigemma-probe pinner membw
#
#   Fallback when native arch detection fails (or for a generic/emulated build):
#   -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
#   (the A76 has dotprod + fp16 but NOT i8mm / SVE / BF16 - do not add those).
#
#   Emulated Pi (docker --platform linux/arm64, qemu-user): MEASURED 2026-10-05 in
#   pi-gemma4-pi-emu:trixie - GGML_NATIVE=ON detects qemu's "max" CPU and adds
#   +i8mm+sve+sme, i.e. a binary that would SIGILL on a real Pi 5. Under
#   PIGEMMA_EMULATED=1 use the fallback flags above so the emulated build has
#   exactly the Pi 5 ISA.
#
# -----------------------------------------------------------------------------
# Options / cache variables of this project (all optional):
#   LLAMA_DIR                   path to the llama.cpp source tree
#                               (default: ../third_party/llama.cpp next to this file)
#   PIGEMMA_PINNED_COMMIT       short hash the repo is pinned to (default a7b94df)
#   PIGEMMA_PINNED_BUILD_NUMBER upstream build number of that commit (default 11398)
#   PIGEMMA_REQUIRE_ALL_TOOLS   OFF (default): a tool whose source file does not
#                               exist yet is skipped with a WARNING (lets the
#                               tools be developed one at a time); ON: hard error.
#   LLAMA_BUILD_NUMBER / LLAMA_BUILD_COMMIT
#                               override the build info compiled into the probe
#                               (it records them in every .moetrace header).
#
# Build-info choice (documented decision): the repo's llama.cpp is a SHALLOW
# clone, so `git rev-list --count HEAD` returns 1 and upstream's build-info would
# report build 1. When the checked-out commit is the pinned commit we use the
# known upstream build number (b11398, matching the official binaries'
# "build : b11398-a7b94df2c"); for any other shallow commit the number is 0 and a
# WARNING says so. git is invoked with -c safe.directory=* because the Docker
# bind mount is owned by a different uid than the container's root.
#
# Flag choices (documented decisions):
#   * pinner: -std=c11 -Wall -Wextra -Werror -O2 -D_GNU_SOURCE (spec). Because
#     _GNU_SOURCE comes from the command line, pinner.c must NOT #define it again
#     unguarded (redefinition warning -> -Werror failure); use #ifndef _GNU_SOURCE.
#   * membw: same warning set (-Wall -Wextra -Werror) but -std=gnu11 and no -D:
#     glibc then exposes POSIX.1-2008 (pthread_barrier_t, clock_gettime, uname)
#     whether or not membw.c defines its own feature-test macro, and -O3 is forced
#     so the measured GB/s does not depend on CMAKE_BUILD_TYPE.
#   * pigemma-probe: -Wall -Wextra without -Werror (it includes llama.cpp's
#     headers, whose warning cleanliness we do not control across gcc versions).
#   * LLAMA_OPENSSL is forced OFF in addition to the spec's list: the probe never
#     talks HTTPS and the Pi need not have libssl-dev. LLAMA_BUILD_APP (new at
#     this llama.cpp commit) is forced OFF for the same "build only what we link"
#     reason.
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(pigemma-tools C CXX)

# ---------------------------------------------------------------------------
# Build type: default Release for single-config generators (Ninja/Makefiles).
# Set before llama.cpp is added so its own default does not apply differently.
# ---------------------------------------------------------------------------
get_property(_pigemma_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _pigemma_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ---------------------------------------------------------------------------
# Project options
# ---------------------------------------------------------------------------
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp" CACHE PATH
    "Path to the llama.cpp source tree (pinned commit)")
get_filename_component(LLAMA_DIR "${LLAMA_DIR}" ABSOLUTE)

set(PIGEMMA_PINNED_COMMIT "a7b94df" CACHE STRING
    "llama.cpp commit (short hash) this repository is pinned to")
set(PIGEMMA_PINNED_BUILD_NUMBER "11398" CACHE STRING
    "Upstream llama.cpp build number of PIGEMMA_PINNED_COMMIT")
option(PIGEMMA_REQUIRE_ALL_TOOLS
    "Fail configuration if a tool source file is missing (default: skip with a warning)" OFF)

if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt" OR NOT EXISTS "${LLAMA_DIR}/include/llama.h")
    message(FATAL_ERROR
        "pigemma-tools: LLAMA_DIR='${LLAMA_DIR}' is not a llama.cpp source tree "
        "(CMakeLists.txt / include/llama.h missing).\n"
        "Pass -DLLAMA_DIR=/path/to/llama.cpp (Docker: /work/third_party/llama.cpp; "
        "Pi: the checkout made by pi/02_build_llama.sh).")
endif()

# ---------------------------------------------------------------------------
# llama.cpp configuration: libraries only, static, no network/TLS, nothing extra.
# FORCE so a stale cache or a stray -D cannot turn on tests/examples/server.
# ---------------------------------------------------------------------------
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "llama: build common utils library" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "llama: build tests"                FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama: build examples"             FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "llama: build tools"                FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "llama: build server example"       FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "llama: build the unified binary"   FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "llama: use libcurl (deprecated)"   FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "llama: use openssl for HTTPS"      FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "build shared libraries"            FORCE)

# ---------------------------------------------------------------------------
# Build info (LLAMA_BUILD_NUMBER / LLAMA_BUILD_COMMIT) - see header comment.
# Normal (non-cache) variables are inherited by the llama.cpp subdirectory,
# whose CMakeLists only computes them when they are NOT DEFINED.
# ---------------------------------------------------------------------------
set(PIGEMMA_LLAMA_SHALLOW "unknown")
set(PIGEMMA_BUILD_INFO_SOURCE "command line")
if(NOT DEFINED LLAMA_BUILD_COMMIT OR NOT DEFINED LLAMA_BUILD_NUMBER)
    find_package(Git QUIET)
    set(_pg_full_hash "")
    set(_pg_short_hash "unknown")
    set(_pg_count "")
    if(GIT_EXECUTABLE)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -c safe.directory=* -C "${LLAMA_DIR}" rev-parse HEAD
            OUTPUT_VARIABLE _pg_full_hash
            RESULT_VARIABLE _pg_res
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT _pg_res EQUAL 0)
            set(_pg_full_hash "")
        endif()
        if(_pg_full_hash)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -c safe.directory=* -C "${LLAMA_DIR}" rev-parse --short=7 HEAD
                OUTPUT_VARIABLE _pg_short_hash
                RESULT_VARIABLE _pg_res
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT _pg_res EQUAL 0 OR NOT _pg_short_hash)
                string(SUBSTRING "${_pg_full_hash}" 0 7 _pg_short_hash)
            endif()
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -c safe.directory=* -C "${LLAMA_DIR}" rev-parse --is-shallow-repository
                OUTPUT_VARIABLE PIGEMMA_LLAMA_SHALLOW
                RESULT_VARIABLE _pg_res
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT _pg_res EQUAL 0 OR NOT PIGEMMA_LLAMA_SHALLOW)
                set(PIGEMMA_LLAMA_SHALLOW "unknown")
            endif()
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -c safe.directory=* -C "${LLAMA_DIR}" rev-list --count HEAD
                OUTPUT_VARIABLE _pg_count
                RESULT_VARIABLE _pg_res
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT _pg_res EQUAL 0)
                set(_pg_count "")
            endif()
        endif()
    endif()

    set(_pg_number 0)
    if(NOT _pg_full_hash)
        set(PIGEMMA_BUILD_INFO_SOURCE "unavailable")
        message(WARNING
            "pigemma-tools: cannot read the git commit of ${LLAMA_DIR} (no git, or no .git "
            "directory). The probe will record commit 'unknown', build 0. Pass "
            "-DLLAMA_BUILD_COMMIT=<hash> -DLLAMA_BUILD_NUMBER=<n> to record them.")
    else()
        set(_pg_pos -1)
        if(PIGEMMA_PINNED_COMMIT)
            string(FIND "${_pg_full_hash}" "${PIGEMMA_PINNED_COMMIT}" _pg_pos)
        endif()
        if(_pg_pos EQUAL 0)
            set(_pg_number "${PIGEMMA_PINNED_BUILD_NUMBER}")
            set(PIGEMMA_BUILD_INFO_SOURCE "git commit == pin; build number from PIGEMMA_PINNED_BUILD_NUMBER")
            if(PIGEMMA_LLAMA_SHALLOW STREQUAL "false" AND _pg_count AND NOT _pg_count STREQUAL PIGEMMA_PINNED_BUILD_NUMBER)
                message(WARNING
                    "pigemma-tools: full clone reports rev-list count ${_pg_count} for the pinned "
                    "commit, PIGEMMA_PINNED_BUILD_NUMBER says ${PIGEMMA_PINNED_BUILD_NUMBER}; "
                    "using the pinned value.")
            endif()
        else()
            message(WARNING
                "pigemma-tools: ${LLAMA_DIR} is at ${_pg_short_hash}, but this repository is "
                "pinned to ${PIGEMMA_PINNED_COMMIT}. Results from this build are NOT comparable "
                "with the pinned baseline; the probe will record ${_pg_short_hash}.")
            if(PIGEMMA_LLAMA_SHALLOW STREQUAL "false" AND _pg_count)
                set(_pg_number "${_pg_count}")
                set(PIGEMMA_BUILD_INFO_SOURCE "git rev-list --count (full clone, unpinned commit)")
            else()
                set(PIGEMMA_BUILD_INFO_SOURCE "unpinned commit in a shallow/unknown clone; build number unknown (0)")
                message(WARNING
                    "pigemma-tools: shallow clone of an unpinned commit - build number unknown, "
                    "recording 0. Pass -DLLAMA_BUILD_NUMBER=<n> to set it.")
            endif()
        endif()
    endif()

    if(NOT DEFINED LLAMA_BUILD_COMMIT)
        set(LLAMA_BUILD_COMMIT "${_pg_short_hash}")
    endif()
    if(NOT DEFINED LLAMA_BUILD_NUMBER)
        set(LLAMA_BUILD_NUMBER "${_pg_number}")
    endif()
endif()

# ---------------------------------------------------------------------------
# llama.cpp itself (only the targets we link get built).
# ---------------------------------------------------------------------------
add_subdirectory("${LLAMA_DIR}" llama EXCLUDE_FROM_ALL)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(PIGEMMA_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
set(PIGEMMA_C_WARNINGS -Wall -Wextra -Werror)
set(PIGEMMA_TARGETS "")

# Returns TRUE in OUT_VAR when SRC exists; otherwise warns (or fails when
# PIGEMMA_REQUIRE_ALL_TOOLS) and returns FALSE.
function(pigemma_source_present TARGET_NAME SRC OUT_VAR)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}")
        set(${OUT_VAR} TRUE PARENT_SCOPE)
    elseif(PIGEMMA_REQUIRE_ALL_TOOLS)
        message(FATAL_ERROR "pigemma-tools: ${TARGET_NAME}: source ${SRC} does not exist.")
    else()
        message(WARNING "pigemma-tools: ${TARGET_NAME}: source ${SRC} does not exist yet - target skipped.")
        set(${OUT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# pigemma-probe (C++17, llama-common + llama), linked exactly like
# llama.cpp/examples/eval-callback.
# ---------------------------------------------------------------------------
pigemma_source_present(pigemma-probe probe/probe.cpp _have_probe)
if(_have_probe)
    add_executable(pigemma-probe probe/probe.cpp)
    target_link_libraries(pigemma-probe PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(pigemma-probe PRIVATE cxx_std_17)
    set_target_properties(pigemma-probe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PIGEMMA_BIN_DIR}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(pigemma-probe PRIVATE -Wall -Wextra)
    endif()
    list(APPEND PIGEMMA_TARGETS pigemma-probe)
endif()

# ---------------------------------------------------------------------------
# pinner (C11, Linux only: mmap/mlock/mincore, /proc).
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    pigemma_source_present(pinner pinner/pinner.c _have_pinner)
    if(_have_pinner)
        add_executable(pinner pinner/pinner.c)
        set_target_properties(pinner PROPERTIES
            C_STANDARD 11
            C_STANDARD_REQUIRED ON
            C_EXTENSIONS OFF
            RUNTIME_OUTPUT_DIRECTORY "${PIGEMMA_BIN_DIR}")
        target_compile_definitions(pinner PRIVATE _GNU_SOURCE)
        target_compile_options(pinner PRIVATE ${PIGEMMA_C_WARNINGS} -O2)
        list(APPEND PIGEMMA_TARGETS pinner)
    endif()
else()
    message(STATUS "pigemma-tools: pinner skipped (Linux only; CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME})")
endif()

# ---------------------------------------------------------------------------
# membw (C11 + pthreads; any POSIX-threads platform).
# ---------------------------------------------------------------------------
if(CMAKE_USE_PTHREADS_INIT)
    pigemma_source_present(membw membw/membw.c _have_membw)
    if(_have_membw)
        add_executable(membw membw/membw.c)
        set_target_properties(membw PROPERTIES
            C_STANDARD 11
            C_STANDARD_REQUIRED ON
            C_EXTENSIONS ON
            RUNTIME_OUTPUT_DIRECTORY "${PIGEMMA_BIN_DIR}")
        target_compile_options(membw PRIVATE ${PIGEMMA_C_WARNINGS} -O3)
        target_link_libraries(membw PRIVATE Threads::Threads)
        list(APPEND PIGEMMA_TARGETS membw)
    endif()
else()
    message(STATUS "pigemma-tools: membw skipped (needs POSIX threads)")
endif()

# Convenience aggregate: cmake --build <dir> --target pigemma-tools
if(PIGEMMA_TARGETS)
    add_custom_target(pigemma-tools DEPENDS ${PIGEMMA_TARGETS})
endif()

# ---------------------------------------------------------------------------
# Provenance file + configure summary.
# ---------------------------------------------------------------------------
function(pigemma_json_str OUT_VAR VALUE)
    string(REPLACE "\\" "\\\\" _v "${VALUE}")
    string(REPLACE "\"" "\\\"" _v "${_v}")
    string(REPLACE "\n" " " _v "${_v}")
    set(${OUT_VAR} "\"${_v}\"" PARENT_SCOPE)
endfunction()

# Arch-specific ggml flags are only meaningful on their own architecture
# (ggml defaults GGML_AVX2=ON whenever GGML_NATIVE=OFF, even on aarch64).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86|x86)$")
    set(_pg_avx2 "${GGML_AVX2}")
    set(_pg_avx512 "${GGML_AVX512}")
    set(_pg_arm_arch "n/a")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|armv.*)$")
    set(_pg_avx2 "n/a")
    set(_pg_avx512 "n/a")
    set(_pg_arm_arch "${GGML_CPU_ARM_ARCH}")
else()
    set(_pg_avx2 "${GGML_AVX2}")
    set(_pg_avx512 "${GGML_AVX512}")
    set(_pg_arm_arch "${GGML_CPU_ARM_ARCH}")
endif()

set(_pg_fields "")
foreach(_pg_pair
        "llama_dir=${LLAMA_DIR}"
        "llama_commit=${LLAMA_BUILD_COMMIT}"
        "pinned_commit=${PIGEMMA_PINNED_COMMIT}"
        "llama_shallow_clone=${PIGEMMA_LLAMA_SHALLOW}"
        "build_info_source=${PIGEMMA_BUILD_INFO_SOURCE}"
        "system_name=${CMAKE_SYSTEM_NAME}"
        "system_processor=${CMAKE_SYSTEM_PROCESSOR}"
        "c_compiler=${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}"
        "cxx_compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        "build_type=${CMAKE_BUILD_TYPE}"
        "generator=${CMAKE_GENERATOR}"
        "ggml_native=${GGML_NATIVE}"
        "ggml_cpu_arm_arch=${_pg_arm_arch}"
        "ggml_avx2=${_pg_avx2}"
        "ggml_avx512=${_pg_avx512}"
        "bin_dir=${PIGEMMA_BIN_DIR}")
    string(FIND "${_pg_pair}" "=" _pg_eq)
    string(SUBSTRING "${_pg_pair}" 0 ${_pg_eq} _pg_key)
    math(EXPR _pg_vstart "${_pg_eq} + 1")
    string(SUBSTRING "${_pg_pair}" ${_pg_vstart} -1 _pg_val)
    pigemma_json_str(_pg_jv "${_pg_val}")
    string(APPEND _pg_fields "  \"${_pg_key}\": ${_pg_jv},\n")
endforeach()
if(LLAMA_BUILD_NUMBER MATCHES "^[0-9]+$")
    set(_pg_num_json "${LLAMA_BUILD_NUMBER}")
else()
    pigemma_json_str(_pg_num_json "${LLAMA_BUILD_NUMBER}")
endif()
set(_pg_targets_json "")
foreach(_pg_t IN LISTS PIGEMMA_TARGETS)
    if(_pg_targets_json)
        string(APPEND _pg_targets_json ", ")
    endif()
    string(APPEND _pg_targets_json "\"${_pg_t}\"")
endforeach()
string(TIMESTAMP _pg_now "%Y-%m-%dT%H:%M:%SZ" UTC)
file(MAKE_DIRECTORY "${PIGEMMA_BIN_DIR}")
file(WRITE "${PIGEMMA_BIN_DIR}/pigemma-tools-build.json"
    "{\n"
    "  \"format_version\": 1,\n"
    "  \"configured_at\": \"${_pg_now}\",\n"
    "  \"llama_build_number\": ${_pg_num_json},\n"
    "${_pg_fields}"
    "  \"targets\": [${_pg_targets_json}]\n"
    "}\n")

message(STATUS "pigemma-tools: LLAMA_DIR          = ${LLAMA_DIR}")
message(STATUS "pigemma-tools: llama build info   = b${LLAMA_BUILD_NUMBER}-${LLAMA_BUILD_COMMIT} (${PIGEMMA_BUILD_INFO_SOURCE})")
message(STATUS "pigemma-tools: system / processor = ${CMAKE_SYSTEM_NAME} / ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "pigemma-tools: GGML_NATIVE=${GGML_NATIVE} GGML_CPU_ARM_ARCH='${_pg_arm_arch}' GGML_AVX2=${_pg_avx2} GGML_AVX512=${_pg_avx512}")
message(STATUS "pigemma-tools: targets            = ${PIGEMMA_TARGETS} (output: ${PIGEMMA_BIN_DIR})")
