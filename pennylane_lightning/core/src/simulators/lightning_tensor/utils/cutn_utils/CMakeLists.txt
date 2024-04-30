cmake_minimum_required(VERSION 3.20)
project(cutn_utils LANGUAGES CXX)

add_library(cutn_utils INTERFACE)

target_include_directories(cutn_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cutn_utils INTERFACE lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET cutn_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
    enable_testing()
    #add_subdirectory("tests")
endif()
