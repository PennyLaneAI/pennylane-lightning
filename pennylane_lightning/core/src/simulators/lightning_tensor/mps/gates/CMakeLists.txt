cmake_minimum_required(VERSION 3.20)
project(lightning_tensor_gates LANGUAGES CXX CUDA)

add_library(lightning_tensor_gates INTERFACE)

target_include_directories(lightning_tensor_gates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_tensor_gates INTERFACE lightning_tensor_utils )

set_property(TARGET lightning_tensor_gates PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()