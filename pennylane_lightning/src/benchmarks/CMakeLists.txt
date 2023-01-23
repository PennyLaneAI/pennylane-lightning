cmake_minimum_required(VERSION 3.14)

project(pennylane_lightning_benchmarks)

set(CMAKE_CXX_STANDARD 20) # At least C++20 is required

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
    set(BUILD_GMOCK OFF CACHE INTERNAL "")
    set(INSTALL_GTEST OFF CACHE INTERNAL "")
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
    set(BENCHMARK_ENABLE_TESTING OFF)
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
target_link_libraries(lightning_benchmarks_dependency INTERFACE lightning_compile_options
                                                                lightning_external_libs
                                                                lightning_gates
                                                                lightning_simulator
                                                                lightning_utils)

################################################################################
# Add benchmark_utils exe
################################################################################

set(GBENCH_SRC  Bench_BitUtil.cpp
                Bench_LinearAlgebra.cpp)

add_executable(utils ${GBENCH_SRC})

target_link_libraries(utils PRIVATE lightning_benchmarks_dependency
                                    benchmark::benchmark_main)

################################################################################
# Add pennylane_lightning_bench_operations exe
################################################################################

add_executable(pennylane_lightning_bench_operations Bench_ApplyOperations.cpp)

target_link_libraries(pennylane_lightning_bench_operations PRIVATE lightning_benchmarks_dependency
                                               benchmark::benchmark_main)

################################################################################
# Add pennylane_lightning_bench_kernels
################################################################################

add_executable(pennylane_lightning_bench_kernels Bench_Kernels.cpp)
target_link_libraries(pennylane_lightning_bench_kernels PRIVATE lightning_benchmarks_dependency
                                            benchmark::benchmark)


add_custom_command(TARGET pennylane_lightning_bench_kernels POST_BUILD 
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                           ${PROJECT_SOURCE_DIR}/benchmark_all.sh
                           ${PROJECT_BINARY_DIR}/benchmark_all.sh
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                           ${PROJECT_SOURCE_DIR}/plot_gate_benchmark.py
                           ${PROJECT_BINARY_DIR}/plot_gate_benchmark.py
)

install(TARGETS pennylane_lightning_bench_kernels pennylane_lightning_bench_operations DESTINATION bin)
install(FILES 
    ${PROJECT_SOURCE_DIR}/benchmark_all.sh 
    ${PROJECT_SOURCE_DIR}/plot_gate_benchmark.py 
    DESTINATION bin
)