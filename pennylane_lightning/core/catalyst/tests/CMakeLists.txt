cmake_minimum_required(VERSION 3.20)

project(catalyst_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(catalyst_tests INTERFACE)
target_link_libraries(catalyst_tests INTERFACE Catch2::Catch2)

foreach(BACKEND ${PL_BACKEND})
    target_link_libraries(catalyst_tests INTERFACE  ${BACKEND}
                                                        "${BACKEND}_catalyst"
                                                        )
endforeach()

ProcessTestOptions(catalyst_tests)

target_sources(catalyst_tests INTERFACE runner_catalyst.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES
    Test_LightningDriver.cpp
    Test_LightningQubitManager.cpp
)

add_executable(catalyst_tests_runner ${TEST_SOURCES})
target_link_libraries(catalyst_tests_runner PRIVATE  catalyst_tests)

catch_discover_tests(catalyst_tests_runner)

install(TARGETS catalyst_tests_runner DESTINATION bin)
