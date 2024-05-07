cmake_minimum_required(VERSION 3.20)

project(TensorCuda_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(TensorCuda_tests INTERFACE)
target_link_libraries(TensorCuda_tests INTERFACE  Catch2::Catch2)

ProcessTestOptions(TensorCuda_tests)

# Create dependency on the dynamically defined simulator/backend target.
target_link_libraries(TensorCuda_tests INTERFACE lightning_utils tensor_cuda)

target_sources(TensorCuda_tests INTERFACE runner_lightning_tensor_TensorCuda.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Tests_TensorCuda.cpp
                    )

add_executable(TensorCuda_test_runner ${TEST_SOURCES})
target_link_libraries(TensorCuda_test_runner PRIVATE  TensorCuda_tests)

catch_discover_tests(TensorCuda_test_runner)

install(TARGETS TensorCuda_test_runner DESTINATION bin)
