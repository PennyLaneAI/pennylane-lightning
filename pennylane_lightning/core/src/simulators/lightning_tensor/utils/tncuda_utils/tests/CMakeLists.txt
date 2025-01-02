cmake_minimum_required(VERSION 3.20)

project(tncuda_utils_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(tncuda_utils_tests INTERFACE)
target_link_libraries(tncuda_utils_tests INTERFACE Catch2::Catch2 tncuda_utils)

ProcessTestOptions(tncuda_utils_tests)

target_sources(tncuda_utils_tests INTERFACE runner_${PL_BACKEND}_tncuda_utils.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_TNCuda_utils.cpp)

add_executable(tncuda_utils_test_runner ${TEST_SOURCES})
target_link_libraries(tncuda_utils_test_runner PRIVATE tncuda_utils_tests)
catch_discover_tests(tncuda_utils_test_runner)

install(TARGETS tncuda_utils_test_runner DESTINATION bin)
