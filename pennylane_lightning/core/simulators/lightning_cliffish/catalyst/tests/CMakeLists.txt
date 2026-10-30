cmake_minimum_required(VERSION 3.26)

project(lightning_cliffish_catalyst_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(lightning_cliffish_catalyst_tests INTERFACE)
target_link_libraries(lightning_cliffish_catalyst_tests INTERFACE Catch2::Catch2
                                                                  lightning_cliffish_catalyst
)

ProcessTestOptions(lightning_cliffish_catalyst_tests)

target_sources(lightning_cliffish_catalyst_tests INTERFACE runner_lightning_cliffish_catalyst.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_LightningCliffishSimulator.cpp
)

add_executable(lightning_cliffish_catalyst_tests_runner ${TEST_SOURCES})
target_link_libraries(lightning_cliffish_catalyst_tests_runner PRIVATE  lightning_cliffish_catalyst_tests)

catch_discover_tests(lightning_cliffish_catalyst_tests_runner)

install(TARGETS lightning_cliffish_catalyst_tests_runner DESTINATION bin)
