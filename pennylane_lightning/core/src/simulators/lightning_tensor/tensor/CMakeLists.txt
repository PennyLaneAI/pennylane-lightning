cmake_minimum_required(VERSION 3.20)
project(${PL_BACKEND}_tensor LANGUAGES CXX)

add_library(${PL_BACKEND}_tensor INTERFACE)

add_subdirectory(base)

set(TNCUDA_UTILS_ADDED FALSE CACHE BOOL "Add tncuda_utils header files")

if("${PL_TENSOR_BACKEND}" STREQUAL "cutensornet" AND NOT TNCUDA_UTILS_ADDED)
    add_subdirectory(tncuda)
    target_include_directories(${PL_BACKEND}_tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tncuda)
    target_link_libraries(${PL_BACKEND}_tensor INTERFACE tensor_cuda)
    set(TNCUDA_UTILS_ADDED TRUE)
endif()

target_include_directories(${PL_BACKEND}_tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/base)
target_link_libraries(${PL_BACKEND}_tensor INTERFACE tensorBase lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET ${PL_BACKEND}_tensor PROPERTY POSITION_INDEPENDENT_CODE ON)
