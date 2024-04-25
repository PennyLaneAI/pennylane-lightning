cmake_minimum_required(VERSION 3.20)

project(${PL_BACKEND}_observables_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(${PL_BACKEND}_obs_tests INTERFACE)
target_link_libraries(${PL_BACKEND}_obs_tests INTERFACE     Catch2::Catch2
                                                                ${PL_BACKEND}_gates
                                                                ${PL_BACKEND}_observables
                                                                ${PL_BACKEND}
                                                                )

ProcessTestOptions(${PL_BACKEND}_obs_tests)

target_sources(${PL_BACKEND}_obs_tests INTERFACE runner_${PL_BACKEND}_obs.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_MPSCutn_Observables.cpp

)

add_executable(${PL_BACKEND}_obs_test_runner ${TEST_SOURCES})
target_link_libraries(${PL_BACKEND}_obs_test_runner PRIVATE ${PL_BACKEND}_obs_tests)
catch_discover_tests(${PL_BACKEND}_obs_test_runner)

install(TARGETS ${PL_BACKEND}_obs_test_runner DESTINATION bin)
