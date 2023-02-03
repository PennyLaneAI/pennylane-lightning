cmake_minimum_required(VERSION 3.16)

project(lightning_components LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20) # At least C++20 is required

option(ENABLE_WARNINGS "Enable warnings" ON)
option(ENABLE_OPENMP "Enable OpenMP" ON)
option(ENABLE_GATE_DISPATCHER "Enable gate kernel dispatching on AVX/AVX2/AVX512" ON)

if(ENABLE_CLANG_TIDY)
    if(NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    message(STATUS "Using CLANG_TIDY_BINARY=${CLANG_TIDY_BINARY}")
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                             -extra-arg=-std=c++20;
    )
endif()


###############################################################################
# Process options
###############################################################################
include("${PROJECT_SOURCE_DIR}/../../cmake/process_options.cmake")


###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS      algorithms;
                           gates;
                           simulator;
                           util;
)
foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests" "tests")
endif()

if (BUILD_BENCHMARKS)
    add_subdirectory("benchmarks" "benchmarks")
endif()
