cmake_minimum_required(VERSION 3.20)

project(observables_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${CMAKE_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(observables_tests INTERFACE)
target_link_libraries(observables_tests INTERFACE   Catch2::Catch2
                                                    lightning_observables
                                                    )

ProcessTestOptions(observables_tests)

target_sources(observables_tests INTERFACE runner_observables.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_Observables.cpp
                    )

add_executable(observables_test_runner ${TEST_SOURCES})
target_link_libraries(observables_test_runner PRIVATE  observables_tests)

# Create dependency on the dynamically defined simulator/backend target.
target_link_libraries(observables_test_runner PRIVATE   ${PL_BACKEND}
                                                        )

catch_discover_tests(observables_test_runner)

install(TARGETS observables_test_runner DESTINATION bin)
