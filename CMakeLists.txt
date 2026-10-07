cmake_minimum_required(VERSION 3.18)
project(kvmem_engine_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    add_compile_options(-O3 -Wall -Wextra -Werror -pedantic -march=armv8.5-a+simd)
else()
    add_compile_options(-O3 -Wall -Wextra -Werror -pedantic)
endif()

include_directories(csrc)

add_executable(kvmem_engine_bench
    csrc/ubba_solver.cpp
    csrc/ifr_lse.cpp
    csrc/main_bench.cpp
)

set_target_properties(kvmem_engine_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build"
)
