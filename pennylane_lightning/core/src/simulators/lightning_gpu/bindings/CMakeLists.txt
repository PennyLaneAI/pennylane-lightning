cmake_minimum_required(VERSION 3.20)

project(lightning_gpu_bindings LANGUAGES CXX)

add_library(lightning_gpu_bindings INTERFACE)

target_include_directories(lightning_gpu_bindings INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_gpu_bindings INTERFACE  lightning_bindings
                                                        lightning_utils
                                                        lightning_gpu
                                                        lightning_gpu_gates
                                                        lightning_gpu_measurements
                                                        lightning_gpu_utils
                                                        )

set_property(TARGET lightning_gpu_bindings PROPERTY POSITION_INDEPENDENT_CODE ON)
