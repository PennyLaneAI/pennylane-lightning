cmake_minimum_required(VERSION 3.20)

project(lightning_gpu_gates_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(lightning_gpu_gates_tests INTERFACE)
target_link_libraries(lightning_gpu_gates_tests INTERFACE     Catch2::Catch2
                                                                lightning_gpu_algorithms
                                                                lightning_gpu_gates
                                                                lightning_gpu_measurements
                                                                lightning_gpu
                                                                )

ProcessTestOptions(lightning_gpu_gates_tests)

target_sources(lightning_gpu_gates_tests INTERFACE runner_lightning_gpu_gates.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_StateVectorCudaManaged_Generators.cpp
                    Test_StateVectorCudaManaged_NonParam.cpp
                    Test_StateVectorCudaManaged_Param.cpp
)

add_executable(lightning_gpu_gates_test_runner ${TEST_SOURCES})
target_link_libraries(lightning_gpu_gates_test_runner PRIVATE lightning_gpu_gates_tests)
catch_discover_tests(lightning_gpu_gates_test_runner)

install(TARGETS lightning_gpu_gates_test_runner DESTINATION bin)

if(ENABLE_MPI)
    ################################################################################
    # Define library
    ################################################################################
    add_library(lightning_gpu_gates_tests_mpi INTERFACE)
    target_link_libraries(lightning_gpu_gates_tests_mpi INTERFACE     Catch2::Catch2
                                                                lightning_gpu_gates
                                                                lightning_gpu_utils
                                                                lightning_gpu
                                                                )

    ProcessTestOptions(lightning_gpu_gates_tests_mpi)

    target_sources(lightning_gpu_gates_tests_mpi INTERFACE ./mpi/mpi_runner_lightning_gpu_gates.cpp)

    ################################################################################
    # Define targets
    ################################################################################
    set(TEST_MPI_SOURCES    ./mpi/Test_StateVectorCudaMPI_NonParam.cpp
                            ./mpi/Test_StateVectorCudaMPI_Param.cpp
                            ./mpi/Test_StateVectorCudaMPI_Generators.cpp
    )

    add_executable(lightning_gpu_gates_test_runner_mpi ${TEST_MPI_SOURCES})
    target_link_libraries(lightning_gpu_gates_test_runner_mpi PRIVATE lightning_gpu_gates_tests_mpi)

    catch_discover_tests(lightning_gpu_gates_test_runner_mpi)
endif()
