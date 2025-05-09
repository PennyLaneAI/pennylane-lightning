cmake_minimum_required(VERSION 3.20)
project(tensorBase LANGUAGES CXX)

add_library(tensorBase INTERFACE)


target_include_directories(tensorBase INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tensorBase INTERFACE lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET tensorBase PROPERTY POSITION_INDEPENDENT_CODE ON)
