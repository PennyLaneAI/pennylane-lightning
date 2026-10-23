cmake_minimum_required(VERSION 3.26)

project(lightning_tensornet_catalyst_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

add_library(lightning_tensornet_catalyst_tests INTERFACE)
target_link_libraries(lightning_tensornet_catalyst_tests INTERFACE Catch2::Catch2
                                                               lightning_tensornet_catalyst
)

# DenseDensityMatrix.hpp (exact reference, shared with the MPS tests)
target_include_directories(lightning_tensornet_catalyst_tests INTERFACE
    ${pennylane_lightning_SOURCE_DIR}/pennylane_lightning/core/simulators/lightning_tensornet/tests)

ProcessTestOptions(lightning_tensornet_catalyst_tests)

target_sources(lightning_tensornet_catalyst_tests INTERFACE runner_lightning_tensornet_catalyst.cpp)

set(TEST_SOURCES Test_LightningTensornetSimulator.cpp)

add_executable(lightning_tensornet_catalyst_tests_runner ${TEST_SOURCES})
target_link_libraries(lightning_tensornet_catalyst_tests_runner PRIVATE lightning_tensornet_catalyst_tests)

catch_discover_tests(lightning_tensornet_catalyst_tests_runner)

install(TARGETS lightning_tensornet_catalyst_tests_runner DESTINATION bin)
