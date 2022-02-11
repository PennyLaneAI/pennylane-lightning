cmake_minimum_required(VERSION 3.14)

project(lightning_components LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

option(ENABLE_WARNINGS "Enable warnings" ON)
option(ENABLE_OPENMP "Enable OpenMP" ON)

if(ENABLE_CLANG_TIDY)
    if(NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                            -extra-arg=-std=c++17;
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

if (BUILD_EXAMPLES)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    add_subdirectory("examples" "examples")
endif()
