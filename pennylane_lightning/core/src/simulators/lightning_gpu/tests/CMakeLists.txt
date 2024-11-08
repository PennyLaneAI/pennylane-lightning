cmake_minimum_required(VERSION 3.20)

project(lightning_gpu_tests
    LANGUAGES CXX C CUDA
)

findCUDATK(lightning_external_libs)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(lightning_gpu_tests INTERFACE)
target_link_libraries(lightning_gpu_tests INTERFACE Catch2::Catch2
                                                    lightning_gpu)

ProcessTestOptions(lightning_gpu_tests)

target_sources(lightning_gpu_tests INTERFACE runner_lightning_gpu.cpp)

################################################################################
# Define targets
################################################################################

set(TEST_SOURCES Test_StateVectorCudaManaged.cpp)

add_executable(lightning_gpu_test_runner ${TEST_SOURCES})
target_link_libraries(lightning_gpu_test_runner PRIVATE lightning_gpu_tests
                                                        lightning_external_libs
)

catch_discover_tests(lightning_gpu_test_runner)

install(TARGETS lightning_gpu_test_runner DESTINATION bin)

if(ENABLE_MPI)
    add_library(lightning_gpu_tests_mpi INTERFACE)
    target_link_libraries(lightning_gpu_tests_mpi INTERFACE     Catch2::Catch2
                                                                lightning_gpu
                                                                lightning_external_libs
                                                                )

    ProcessTestOptions(lightning_gpu_tests_mpi)

    target_sources(lightning_gpu_tests_mpi INTERFACE ./mpi/runner_lightning_gpu_mpi.cpp)

    ################################################################################
    # Define targets
    ################################################################################
    set(TEST_SOURCES ./mpi/Test_StateVectorCudaMPI.cpp)

    add_executable(lightning_gpu_test_runner_mpi ${TEST_SOURCES})
    target_link_libraries(lightning_gpu_test_runner_mpi PRIVATE lightning_gpu_tests_mpi)
    catch_discover_tests(lightning_gpu_test_runner_mpi)

endif()

