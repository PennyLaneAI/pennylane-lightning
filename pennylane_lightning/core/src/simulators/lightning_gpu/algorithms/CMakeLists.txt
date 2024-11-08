cmake_minimum_required(VERSION 3.20)

project(lightning_gpu_algorithms LANGUAGES CXX)

set(ALGORITHMS_FILES AlgorithmsGPU.cpp CACHE INTERNAL "" FORCE)

if(ENABLE_MPI)
    list(APPEND ALGORITHMS_FILES AlgorithmsGPUMPI.cpp)
endif()

add_library(lightning_gpu_algorithms STATIC ${ALGORITHMS_FILES})

target_include_directories(lightning_gpu_algorithms INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_gpu_algorithms PRIVATE      lightning_compile_options
                                                            lightning_external_libs
                                                            )

target_link_libraries(lightning_gpu_algorithms PUBLIC       lightning_gpu_utils
                                                            lightning_algorithms
                                                            lightning_utils
                                                            lightning_gpu
                                                            lightning_gpu_observables
                                                            )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()