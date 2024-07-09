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

add_library(${PL_BACKEND}_tests INTERFACE)
target_link_libraries(${PL_BACKEND}_tests INTERFACE     Catch2::Catch2
                                                        ${PL_BACKEND})

ProcessTestOptions(${PL_BACKEND}_tests)

target_sources(${PL_BACKEND}_tests INTERFACE runner_lightning_tensor_MPSTNCuda.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Tests_MPSTNCuda.cpp
)

add_executable(${PL_BACKEND}_test_runner ${TEST_SOURCES})
target_link_libraries(${PL_BACKEND}_test_runner PRIVATE ${PL_BACKEND}_tests)
catch_discover_tests(${PL_BACKEND}_test_runner)

install(TARGETS ${PL_BACKEND}_test_runner DESTINATION bin)
