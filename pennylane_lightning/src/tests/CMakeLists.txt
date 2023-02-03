cmake_minimum_required(VERSION 3.16)

project(pennylane_lightning_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(ENABLE_NATIVE "Enable native CPU build tuning" OFF)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.9
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
target_link_libraries(lightning_tests_dependency INTERFACE lightning_algorithms
                                                           lightning_gates
                                                           lightning_simulator
                                                           lightning_utils
                                                           Catch2::Catch2)
if(ENABLE_WARNINGS)
    message(STATUS "ENABLE_WARNINGS is ON.")
    if(MSVC)
        target_compile_options(lightning_tests_dependency INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4;/WX>)
    else()
        target_compile_options(lightning_tests_dependency INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
    endif()
else()
    if(MSVC)
        target_compile_options(lightning_tests_dependency INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
    else()
        target_compile_options(lightning_tests_dependency INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wno-unused>)
    endif()
endif()

target_sources(lightning_tests_dependency INTERFACE runner_main.cpp)

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
    target_compile_options(lightning_tests_dependency INTERFACE -march=native)
endif()

################################################################################
# Define targets
################################################################################

add_executable(compile_time_tests compile_time_tests.cpp)
target_link_libraries(compile_time_tests lightning_compile_options lightning_gates lightning_utils lightning_simulator)

set(TEST_SOURCES CreateAllWires.cpp
                 Test_AdjDiff.cpp
                 Test_AlgUtil.cpp
                 Test_AVXSingleQubitGateHelpers.cpp
                 Test_AVXTwoQubitGateHelpers.cpp
                 #Test_Bindings.cpp
                 Test_CompilerSupport.cpp
                 Test_DynamicDispatcher.cpp
                 Test_Error.cpp
                 Test_GateImplementations_CompareKernels.cpp
                 Test_GateImplementations_Generator.cpp
                 Test_GateImplementations_Inverse.cpp
                 Test_GateImplementations_Matrix.cpp
                 Test_GateImplementations_Nonparam.cpp
                 Test_GateImplementations_Param.cpp
                 Test_GateUtil.cpp
                 Test_Internal.cpp
                 Test_KernelMap.cpp
                 Test_Kokkos_Sparse.cpp
                 Test_LinearAlgebra.cpp
                 Test_Measures.cpp
                 Test_Measures_Sparse.cpp
                 Test_Observables.cpp
                 Test_OpToMemberFuncPtr.cpp
                 Test_RuntimeInfo.cpp
                 Test_StateVecAdjDiff.cpp
                 Test_StateVectorManagedCPU.cpp
                 Test_StateVectorRawCPU.cpp
                 Test_TypeTraits.cpp
                 Test_Util.cpp)

add_executable(pennylane_lightning_test_runner ${TEST_SOURCES})
target_link_libraries(pennylane_lightning_test_runner PRIVATE lightning_tests_dependency
                                     lightning_compile_options
                                     lightning_external_libs)
catch_discover_tests(pennylane_lightning_test_runner)

# We build compile time tests before the runtime tests as build error messages
# are horrible if compile time constants are not well defined.
add_dependencies(pennylane_lightning_test_runner compile_time_tests)

install(TARGETS pennylane_lightning_test_runner DESTINATION bin)
