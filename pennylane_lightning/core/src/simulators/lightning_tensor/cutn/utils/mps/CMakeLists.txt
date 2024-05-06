cmake_minimum_required(VERSION 3.20)
project(mps_cutn_utils LANGUAGES CXX)

add_library(mps_cutn_utils INTERFACE)

target_include_directories(mps_cutn_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mps_cutn_utils INTERFACE lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET mps_cutn_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
