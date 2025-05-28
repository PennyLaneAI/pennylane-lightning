cmake_minimum_required(VERSION 3.20)
project(${PL_BACKEND}_gates LANGUAGES CXX CUDA)

add_library(${PL_BACKEND}_gates INTERFACE)

target_include_directories(${PL_BACKEND}_gates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PL_BACKEND}_gates INTERFACE ${PL_BACKEND}_utils ${PL_BACKEND}_tensor lightning_gates)

set_property(TARGET ${PL_BACKEND}_gates PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
