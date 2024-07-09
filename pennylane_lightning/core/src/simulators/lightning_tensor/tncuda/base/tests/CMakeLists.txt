cmake_minimum_required(VERSION 3.20)

project(tensornet_base_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(tensornet_base_tests INTERFACE)
target_link_libraries(tensornet_base_tests INTERFACE  Catch2::Catch2)

ProcessTestOptions(tensornet_base_tests)

# Create dependency on the dynamically defined simulator/backend target.
target_link_libraries(tensornet_base_tests INTERFACE ${PL_BACKEND} ${PL_BACKEND}_tncuda_utils)

target_sources(tensornet_base_tests INTERFACE runner_lightning_tensor_tensornetBase.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Tests_tensornetBase.cpp
                    )

add_executable(tensornet_base_test_runner ${TEST_SOURCES})
target_link_libraries(tensornet_base_test_runner PRIVATE  tensornet_base_tests)

catch_discover_tests(tensornet_base_test_runner)

install(TARGETS tensornet_base_test_runner DESTINATION bin)
