cmake_minimum_required(VERSION 3.20)

project(lightning_catalyst LANGUAGES CXX)

add_library(lightning_catalyst INTERFACE)

target_include_directories(lightning_catalyst INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (BUILD_TESTS AND NOT DEFINED PL_TENSOR)
    enable_testing()
    add_subdirectory("tests")
endif()
