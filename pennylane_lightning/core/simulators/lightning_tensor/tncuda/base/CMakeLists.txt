cmake_minimum_required(VERSION 3.20)

project(${PL_BACKEND}_tensornetBase LANGUAGES CXX)

add_library(${PL_BACKEND}_tensornetBase INTERFACE)

target_include_directories(${PL_BACKEND}_tensornetBase INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PL_BACKEND}_tensornetBase INTERFACE  lightning_compile_options
                                                lightning_external_libs
                                                )

set_property(TARGET ${PL_BACKEND}_tensornetBase PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
