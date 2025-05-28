cmake_minimum_required(VERSION 3.20)
project(tncuda_utils LANGUAGES CXX)

add_library(tncuda_utils INTERFACE)

target_include_directories(tncuda_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tncuda_utils INTERFACE lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET tncuda_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()
