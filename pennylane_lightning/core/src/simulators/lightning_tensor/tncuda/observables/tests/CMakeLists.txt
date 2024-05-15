cmake_minimum_required(VERSION 3.20)

project(TNCuda_observables_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(TNCuda_observables_tests INTERFACE)
target_link_libraries(TNCuda_observables_tests INTERFACE     Catch2::Catch2
${PL_TENSOR}_gates
                                                                TNCuda_observables
                                                                ${PL_TENSOR}
                                                                )

ProcessTestOptions(TNCuda_observables_tests)

target_sources(TNCuda_observables_tests INTERFACE runner_TNCuda_observables.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES Test_Observables_TNCuda.cpp)

add_executable(TNCuda_observables_test_runner ${TEST_SOURCES})
target_link_libraries(TNCuda_observables_test_runner PRIVATE TNCuda_observables_tests)
catch_discover_tests(TNCuda_observables_test_runner)

install(TARGETS TNCuda_observables_test_runner DESTINATION bin)
