cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_observables_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(lightning_kokkos_observables_tests INTERFACE)
target_link_libraries(lightning_kokkos_observables_tests INTERFACE     Catch2::Catch2
                                                                lightning_kokkos_gates
                                                                lightning_kokkos_observables
                                                                lightning_kokkos
                                                                )

ProcessTestOptions(lightning_kokkos_observables_tests)

target_sources(lightning_kokkos_observables_tests INTERFACE runner_lightning_kokkos_observables.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES Test_ObservablesKokkos.cpp)

add_executable(lightning_kokkos_observables_test_runner ${TEST_SOURCES})
target_link_libraries(lightning_kokkos_observables_test_runner PRIVATE lightning_kokkos_observables_tests)
catch_discover_tests(lightning_kokkos_observables_test_runner)

install(TARGETS lightning_kokkos_observables_test_runner DESTINATION bin)

if(ENABLE_MPI)
    add_library(lightning_kokkos_observables_tests_mpi INTERFACE)
    target_link_libraries(lightning_kokkos_observables_tests_mpi INTERFACE   Catch2::Catch2
                                                                        lightning_kokkos
                                                                        lightning_kokkos_observables
                                                                        )

    ProcessTestOptions(lightning_kokkos_observables_tests_mpi)

    target_sources(lightning_kokkos_observables_tests_mpi INTERFACE ./mpi/runner_lightning_kokkos_observables_mpi.cpp)

    ################################################################################
    # Define targets
    ################################################################################
    set(TEST_SOURCES    
                                        ./mpi/Test_ObservablesKokkosMPI.cpp)

    add_executable(lightning_kokkos_observables_test_runner_mpi ${TEST_SOURCES})
    target_link_libraries(lightning_kokkos_observables_test_runner_mpi PRIVATE  lightning_kokkos_observables_tests_mpi)

    catch_discover_tests(lightning_kokkos_observables_test_runner_mpi)

    install(TARGETS lightning_kokkos_observables_test_runner_mpi DESTINATION bin)

endif()
