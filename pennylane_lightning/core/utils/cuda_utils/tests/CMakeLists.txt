cmake_minimum_required(VERSION 3.20)

project(cuda_utils_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(cuda_utils_tests INTERFACE)
target_link_libraries(cuda_utils_tests INTERFACE     Catch2::Catch2
                                                                lightning_utils
                                                                cuda_utils
                                                                )

ProcessTestOptions(cuda_utils_tests)

target_sources(cuda_utils_tests INTERFACE runner_cuda_utils.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES 
                        Test_LinearAlgebra.cpp)

add_executable(cuda_utils_test_runner ${TEST_SOURCES})
target_link_libraries(cuda_utils_test_runner PRIVATE cuda_utils_tests)
catch_discover_tests(cuda_utils_test_runner)

install(TARGETS cuda_utils_test_runner DESTINATION bin)
