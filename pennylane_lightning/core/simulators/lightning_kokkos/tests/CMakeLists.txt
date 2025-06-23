cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(lightning_kokkos_tests INTERFACE)
target_link_libraries(lightning_kokkos_tests INTERFACE Catch2::Catch2
                                                    lightning_kokkos)

ProcessTestOptions(lightning_kokkos_tests)

target_sources(lightning_kokkos_tests INTERFACE runner_lightning_kokkos.cpp)

################################################################################
# Define targets
################################################################################

set(TEST_SOURCES Test_StateVectorLKokkos.cpp)

add_executable(lightning_kokkos_test_runner ${TEST_SOURCES})
target_link_libraries(lightning_kokkos_test_runner PRIVATE lightning_kokkos_tests)

catch_discover_tests(lightning_kokkos_test_runner)

install(TARGETS lightning_kokkos_test_runner DESTINATION bin)

if(ENABLE_MPI)
    add_library(${PL_BACKEND}_tests_mpi INTERFACE)
    target_link_libraries(${PL_BACKEND}_tests_mpi INTERFACE     Catch2::Catch2
                                                                ${PL_BACKEND}
                                                                lightning_external_libs
                                                                )

    ProcessTestOptions(${PL_BACKEND}_tests_mpi)

    target_sources(${PL_BACKEND}_tests_mpi INTERFACE 
                                                        ./mpi/runner_${PL_BACKEND}_mpi.cpp
                                                        )

    ################################################################################
    # Define targets
    ################################################################################
    set(TEST_SOURCES 
                            ./mpi/Test_StateVectorKokkosMPI.cpp
                            )

    add_executable(${PL_BACKEND}_test_runner_mpi ${TEST_SOURCES})
    target_link_libraries(${PL_BACKEND}_test_runner_mpi PRIVATE ${PL_BACKEND}_tests_mpi)
    catch_discover_tests(${PL_BACKEND}_test_runner_mpi)

endif()

