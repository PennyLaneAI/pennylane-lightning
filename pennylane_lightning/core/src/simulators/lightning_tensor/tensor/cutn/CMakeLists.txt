cmake_minimum_required(VERSION 3.20)
project(cuda_tensor LANGUAGES CXX)

add_library(cuda_tensor INTERFACE)

target_include_directories(cuda_tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cuda_tensor INTERFACE tensorBase lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET cuda_tensor PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
