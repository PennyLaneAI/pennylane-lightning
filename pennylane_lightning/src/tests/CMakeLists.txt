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
target_link_libraries(lightning_tests_dependency INTERFACE Catch2::Catch2)
target_compile_options(lightning_tests_dependency INTERFACE "$<$<CONFIG:DEBUG>:-Wall>")
target_sources(lightning_tests_dependency INTERFACE runner_main.cpp)

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
    target_compile_options(lightning_tests_dependency PRIVATE -march=native)
endif()

################################################################################
# Define targets
################################################################################

add_executable(Test_Util Test_Util.cpp)
target_link_libraries(Test_Util PRIVATE lightning_tests_dependency 
                                        lightning_utils)
catch_discover_tests(Test_Util)

add_executable(Test_IndicesUtil Test_IndicesUtil.cpp)
target_link_libraries(Test_IndicesUtil PRIVATE lightning_tests_dependency 
                                               lightning_utils 
                                               lightning_simulator)
catch_discover_tests(Test_IndicesUtil)

add_executable(Test_GateOperations Test_GateOperations_Single_Nonparam.cpp
                                   Test_GateOperations_Single_Param.cpp
                                   Test_GateOperations_Two_Nonparam.cpp
                                   Test_GateOperations_Two_Param.cpp)

target_link_libraries(Test_GateOperations PRIVATE lightning_tests_dependency 
                                                  lightning_utils 
                                                  lightning_simulator)
catch_discover_tests(Test_GateOperations)


add_executable(Test_SelectGateOps Test_SelectGateOps.cpp)
target_link_libraries(Test_SelectGateOps PRIVATE lightning_tests_dependency 
                                                 lightning_utils 
                                                 lightning_simulator)
catch_discover_tests(Test_SelectGateOps)



add_executable(Test_AdjDiff Test_AdjDiff.cpp)
target_link_libraries(Test_AdjDiff PRIVATE lightning_tests_dependency 
                                           lightning_utils 
                                           lightning_simulator
                                           lightning_algorithms)
catch_discover_tests(Test_AdjDiff)


# As default make test does not build test executables
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS Test_Util Test_IndicesUtil Test_GateOperations Test_SelectGateOps Test_AdjDiff)

# TODO: Recover before push
#target_link_libraries(runner lightning_simulator lightning_utils lightning_algorithms Catch2::Catch2)
