cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_measurements_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(lightning_kokkos_measurements_tests INTERFACE)
target_link_libraries(lightning_kokkos_measurements_tests INTERFACE     Catch2::Catch2
                                                                lightning_kokkos_gates
                                                                lightning_kokkos_utils
                                                                lightning_kokkos_measurements
                                                                lightning_kokkos_observables
                                                                lightning_kokkos
                                                                )

ProcessTestOptions(lightning_kokkos_measurements_tests)

target_sources(lightning_kokkos_measurements_tests INTERFACE runner_lightning_kokkos_measurements.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    
                    Test_StateVectorKokkos_Expval.cpp  
                    Test_StateVectorKokkos_Measure.cpp  
                    Test_StateVectorKokkos_Var.cpp
)

add_executable(lightning_kokkos_measurements_test_runner ${TEST_SOURCES})
target_link_libraries(lightning_kokkos_measurements_test_runner PRIVATE lightning_kokkos_measurements_tests)
catch_discover_tests(lightning_kokkos_measurements_test_runner)

install(TARGETS lightning_kokkos_measurements_test_runner DESTINATION bin)


if(ENABLE_MPI)
    add_library(${PL_BACKEND}_measurements_tests_mpi INTERFACE)
    target_link_libraries(${PL_BACKEND}_measurements_tests_mpi INTERFACE     Catch2::Catch2
                                                                ${PL_BACKEND}
                                                                ${PL_BACKEND}_gates
                                                                ${PL_BACKEND}_utils
                                                                ${PL_BACKEND}_measurements
                                                                ${PL_BACKEND}_observables
                                                                lightning_external_libs
                                                                )

    ProcessTestOptions(${PL_BACKEND}_measurements_tests_mpi)

    target_sources(${PL_BACKEND}_measurements_tests_mpi INTERFACE ./mpi/runner_${PL_BACKEND}_measurements_mpi.cpp)

    ################################################################################
    # Define targets
    ################################################################################
    set(TEST_SOURCES 
                        ./mpi/Test_StateVectorKokkosMPI_Measure.cpp
                        )

    add_executable(${PL_BACKEND}_measurements_test_runner_mpi ${TEST_SOURCES})
    target_link_libraries(${PL_BACKEND}_measurements_test_runner_mpi PRIVATE ${PL_BACKEND}_measurements_tests_mpi)
    catch_discover_tests(${PL_BACKEND}_measurements_test_runner_mpi)

endif()

