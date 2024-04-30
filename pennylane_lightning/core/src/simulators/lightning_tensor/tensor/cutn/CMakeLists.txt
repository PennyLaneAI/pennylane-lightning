cmake_minimum_required(VERSION 3.20)
project(cutn_tensor LANGUAGES CXX)

add_library(cutn_tensor INTERFACE)

target_include_directories(cutn_tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cutn_tensor INTERFACE lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET cutn_tensor PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
    enable_testing()
    #add_subdirectory("tests")
endif()
