cmake_minimum_required(VERSION 3.20)
project(tensor_cuda LANGUAGES CXX)

add_library(tensor_cuda INTERFACE)

target_include_directories(tensor_cuda INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tensor_cuda INTERFACE tensorBase lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET tensor_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
