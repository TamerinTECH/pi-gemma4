# Narrow external comparison, not the full KleidiAI/llama backend integration.
# Included from CF CMakeLists or another project; target: cf-kleidi-compare.
set(CF_KLEIDI_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/kleidiai-v1.24.0"
    CACHE PATH "Unmodified KleidiAI v1.24.0 source")
set(CF_KLEIDI_PACK "${CF_KLEIDI_SOURCE_DIR}/kai/ukernels/matmul/pack")
set(CF_KLEIDI_MATMUL "${CF_KLEIDI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p")
if(NOT EXISTS "${CF_KLEIDI_PACK}/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c")
    message(FATAL_ERROR "KleidiAI v1.24.0 source missing; see docs/cf-engine/KLEIDIAI.md")
endif()
add_library(cf-kleidi-compare STATIC
    "${CMAKE_CURRENT_LIST_DIR}/kleidi_compare.cpp"
    "${CF_KLEIDI_PACK}/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c")
target_compile_features(cf-kleidi-compare PUBLIC cxx_std_17)
target_include_directories(cf-kleidi-compare PUBLIC "${CMAKE_CURRENT_LIST_DIR}"
    PRIVATE "${CF_KLEIDI_SOURCE_DIR}")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(CF_KLEIDI_DOTPROD "${CF_KLEIDI_MATMUL}/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c")
    target_sources(cf-kleidi-compare PRIVATE
        "${CF_KLEIDI_PACK}/kai_lhs_quant_pack_qsi8d32p_f32.c" "${CF_KLEIDI_DOTPROD}")
    set_source_files_properties("${CF_KLEIDI_DOTPROD}" PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+fp16")
    target_compile_definitions(cf-kleidi-compare PRIVATE CF_KLEIDI_ENABLED=1)
else()
    message(STATUS "Kleidi comparison: packing validation only; dotprod execution unavailable")
endif()
if(UNIX)
    target_link_libraries(cf-kleidi-compare PRIVATE m)
endif()
