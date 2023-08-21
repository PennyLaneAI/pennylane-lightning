cmake_minimum_required(VERSION 3.20)

project(${PL_BACKEND}_utils LANGUAGES CXX)

add_library(${PL_BACKEND}_utils INTERFACE)

target_include_directories(${PL_BACKEND}_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PL_BACKEND}_utils INTERFACE lightning_gates lightning_utils)

set_property(TARGET ${PL_BACKEND}_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
