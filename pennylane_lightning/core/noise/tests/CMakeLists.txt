cmake_minimum_required(VERSION 3.26)

project(noise_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

add_library(noise_tests INTERFACE)
target_link_libraries(noise_tests INTERFACE Catch2::Catch2 lightning_noise)

ProcessTestOptions(noise_tests)

target_sources(noise_tests INTERFACE runner_noise.cpp)

set(TEST_SOURCES Test_NoiseModel.cpp Test_ObservableTerms.cpp Test_HermitianEigen.cpp)

add_executable(noise_test_runner ${TEST_SOURCES})
target_link_libraries(noise_test_runner PRIVATE noise_tests)
catch_discover_tests(noise_test_runner)

install(TARGETS noise_test_runner DESTINATION bin)
