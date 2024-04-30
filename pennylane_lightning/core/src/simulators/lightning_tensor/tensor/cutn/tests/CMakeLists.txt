cmake_minimum_required(VERSION 3.20)

project(CudaTensor_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(CudaTensor_tests INTERFACE)
target_link_libraries(CudaTensor_tests INTERFACE  Catch2::Catch2)

ProcessTestOptions(CudaTensor_tests)

# Create dependency on the dynamically defined simulator/backend target.
target_link_libraries(CudaTensor_tests INTERFACE lightning_utils cuda_tensor)

target_sources(CudaTensor_tests INTERFACE runner_lightning_tensor_CudaTensor.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Tests_CudaTensor.cpp
                    )

add_executable(CudaTensor_test_runner ${TEST_SOURCES})
target_link_libraries(CudaTensor_test_runner PRIVATE  CudaTensor_tests)

catch_discover_tests(CudaTensor_test_runner)

install(TARGETS CudaTensor_test_runner DESTINATION bin)
