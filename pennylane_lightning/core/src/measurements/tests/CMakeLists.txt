cmake_minimum_required(VERSION 3.20)

project(measurements_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${CMAKE_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(measurements_tests INTERFACE)
target_link_libraries(measurements_tests INTERFACE  Catch2::Catch2)

# Create dependencies on the dynamically defined simulator/backend targets.
target_link_libraries(measurements_tests INTERFACE  ${PL_BACKEND}
                                                    "${PL_BACKEND}_measurements"
                                                    "${PL_BACKEND}_observables"
                                                    )

ProcessTestOptions(measurements_tests)

target_sources(measurements_tests INTERFACE runner_measurements.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_MeasurementsBase.cpp
                    )

add_executable(measurements_test_runner ${TEST_SOURCES})
target_link_libraries(measurements_test_runner PRIVATE  measurements_tests)

catch_discover_tests(measurements_test_runner)

install(TARGETS measurements_test_runner DESTINATION bin)
