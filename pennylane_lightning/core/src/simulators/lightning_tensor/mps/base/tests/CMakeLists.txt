cmake_minimum_required(VERSION 3.20)

project(mps_base_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(mps_base_tests INTERFACE)
target_link_libraries(mps_base_tests INTERFACE  Catch2::Catch2)

ProcessTestOptions(mps_base_tests)

# Create dependency on the dynamically defined simulator/backend target.
target_link_libraries(mps_base_tests INTERFACE ${PL_TENSOR} ${PL_TENSOR}_utils)

target_sources(mps_base_tests INTERFACE runner_lightning_tensor_mps_base.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Tests_mps_base.cpp
                    )

add_executable(mps_base_test_runner ${TEST_SOURCES})
target_link_libraries(mps_base_test_runner PRIVATE  mps_base_tests)

catch_discover_tests(mps_base_test_runner)

install(TARGETS mps_base_test_runner DESTINATION bin)
