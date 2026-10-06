# pigemma-fastcheck: API-level regression scenarios for the pigemma-fast patches.
# Installed into a llama.cpp tree as examples/pigemma-fastcheck by install_into_tree.sh.
set(TARGET pigemma-fastcheck)
add_executable(${TARGET} fast_check.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
