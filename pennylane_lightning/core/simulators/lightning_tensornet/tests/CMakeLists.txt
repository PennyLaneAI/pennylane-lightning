cmake_minimum_required(VERSION 3.26)

project(lightning_tensornet_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

add_library(lightning_tensornet_tests INTERFACE)
target_link_libraries(lightning_tensornet_tests INTERFACE Catch2::Catch2 lightning_tensornet)
# DenseDensityMatrix.hpp (exact reference)
target_include_directories(lightning_tensornet_tests INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

ProcessTestOptions(lightning_tensornet_tests)

target_sources(lightning_tensornet_tests INTERFACE runner_lightning_tensornet.cpp)

set(TEST_SOURCES Test_MPSLinAlg.cpp Test_MPSKokkos.cpp)

add_executable(lightning_tensornet_test_runner ${TEST_SOURCES})
target_link_libraries(lightning_tensornet_test_runner PRIVATE lightning_tensornet_tests)
catch_discover_tests(lightning_tensornet_test_runner)

install(TARGETS lightning_tensornet_test_runner DESTINATION bin)
