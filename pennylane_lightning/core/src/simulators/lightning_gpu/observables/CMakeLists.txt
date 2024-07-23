cmake_minimum_required(VERSION 3.20)

project(lightning_gpu_observables LANGUAGES CXX)

set(OBSERVABLES_FILES ObservablesGPU.cpp CACHE INTERNAL "" FORCE)

if(ENABLE_MPI)
    list(APPEND OBSERVABLES_FILES ObservablesGPUMPI.cpp)
endif()

add_library(lightning_gpu_observables STATIC ${OBSERVABLES_FILES})

target_include_directories(lightning_gpu_observables INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_gpu_observables PRIVATE   lightning_compile_options
                                                            lightning_external_libs
                                                            )

target_link_libraries(lightning_gpu_observables PUBLIC  lightning_utils
                                                        lightning_gates
                                                        lightning_observables
                                                        lightning_gpu_utils
                                                        lightning_gpu_gates
                                                        lightning_gpu
                                                        )

set_property(TARGET lightning_gpu_observables PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()