cmake_minimum_required(VERSION 3.14)

project(pennylane_lightning_tests)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(ENABLE_NATIVE "Enable native CPU build tuning" OFF)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.1
)

FetchContent_MakeAvailable(Catch2)

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)


################################################################################
# Define compile options and library
################################################################################

add_library(lightning_tests_dependency INTERFACE)
target_link_libraries(lightning_tests_dependency INTERFACE lightning_simulator lightning_algorithms lightning_utils Catch2::Catch2)
target_compile_options(lightning_tests_dependency INTERFACE "$<$<CONFIG:DEBUG>:-Wall>")
target_sources(lightning_tests_dependency INTERFACE runner_main.cpp)

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
    target_compile_options(lightning_tests_dependency INTERFACE -march=native)
endif()

################################################################################
# Define targets
################################################################################

set(TEST_SOURCES Test_AdjDiff.cpp
                 Test_Bindings.cpp
                 Test_DynamicDispatcher.cpp
                 Test_GateOperations_Nonparam.cpp
                 Test_GateOperations_Param.cpp
                 Test_IndicesUtil.cpp
                 Test_Measures.cpp
                 Test_SelectGateOps.cpp
                 Test_StateVectorManaged.cpp
                 Test_StateVectorRaw.cpp
                 Test_Util.cpp
                 Test_VectorJacobianProduct.cpp)


add_executable(runner ${TEST_SOURCES})
target_link_libraries(runner PRIVATE lightning_tests_dependency)
catch_discover_tests(runner)
