cmake_minimum_required(VERSION 3.14)

project(pennylane_lightning_benchmarks)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "ENABLE_GB is ON. Find GBenchmark.")
find_package(benchmark QUIET)

if (NOT benchmark_FOUND)
    message(STATUS "GBenchmark is not found. Fetch GBenchmark.")

    include(FetchContent)

    # Fetch GTest; required for GBenchmark
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.11.0 # latest
    )
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
    endif()

    # Fetch GBenchmark and surpress internal tests.
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.6.1 # latest
    )
    set(BENCHMARK_ENABLE_TESTING off)
    FetchContent_GetProperties(googlebenchmark)
    if(NOT googlebenchmark_POPULATED)
        FetchContent_Populate(googlebenchmark)
        add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
    endif()
else()
    message(STATUS "GBenchmark found.")
endif()

################################################################################
# Define a library for dependencies
################################################################################

add_library(lightning_benchmarks_dependency INTERFACE)
target_link_libraries(lightning_benchmarks_dependency INTERFACE lightning_utils
                                                    lightning_external_libs
                                                    benchmark::benchmark_main)

################################################################################
# Add benchmark_utils exe
################################################################################

set(GBENCH_SRC  Bench_BitUtil.cpp
                Bench_LinearAlgebra.cpp)

add_executable(utils ${GBENCH_SRC})

add_compile_definitions(utils INTERFACE _ENABLE_BLAS=1)

target_link_libraries(utils PRIVATE lightning_benchmarks_dependency)

################################################################################
# Add benchmark_apply_operations exe
################################################################################

add_executable(apply_operations Bench_ApplyOperations.cpp)

target_link_libraries(apply_operations PRIVATE lightning_compile_options
                                            lightning_gates
                                            lightning_simulator
                                            lightning_benchmarks_dependency)

################################################################################
# Add benchmark_apply_multirz exe
################################################################################

add_executable(apply_multirz Bench_ApplyMultiRZ.cpp)

target_link_libraries(apply_multirz PRIVATE lightning_compile_options
                                            lightning_gates
                                            lightning_simulator
                                            lightning_benchmarks_dependency)