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

add_executable(runner runner_main.cpp)

# TODO: Recover before push
#target_link_libraries(runner lightning_simulator lightning_utils lightning_algorithms Catch2::Catch2)
target_link_libraries(runner lightning_utils Catch2::Catch2)

target_sources(runner PRIVATE   #Test_AdjDiff.cpp
                                #Test_Bindings.cpp
                                #Test_StateVector_Nonparam.cpp 
                                #Test_StateVector_Param.cpp 
                                #Test_StateVectorManaged_Nonparam.cpp 
                                #Test_StateVectorManaged_Param.cpp 
                                Test_Util.cpp
                                #Test_Measures.cpp
                                #Test_VectorJacobianProduct.cpp
)

target_compile_options(runner PRIVATE "$<$<CONFIG:DEBUG>:-Wall>")

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
    target_compile_options(runner PRIVATE -march=native)
endif()

catch_discover_tests(runner)
