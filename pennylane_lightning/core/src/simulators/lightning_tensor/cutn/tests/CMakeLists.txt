cmake_minimum_required(VERSION 3.20)

project(${PL_BACKEND}_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(${PL_TENSOR}_tests INTERFACE)
target_link_libraries(${PL_TENSOR}_tests INTERFACE     Catch2::Catch2
                                                       ${PL_TENSOR})

ProcessTestOptions(${PL_TENSOR}_tests)

target_sources(${PL_TENSOR}_tests INTERFACE runner_lightning_tensor_CutnMPS.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Tests_CutnMPS.cpp
)

add_executable(${PL_TENSOR}_test_runner ${TEST_SOURCES})
target_link_libraries(${PL_TENSOR}_test_runner PRIVATE ${PL_TENSOR}_tests)
catch_discover_tests(${PL_TENSOR}_test_runner)

install(TARGETS ${PL_TENSOR}_test_runner DESTINATION bin)
