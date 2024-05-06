cmake_minimum_required(VERSION 3.20)
project(${PL_TENSOR}_utils LANGUAGES CXX)

add_library(${PL_TENSOR}_utils INTERFACE)

set(CUTN_UTILS_ADDED FALSE CACHE BOOL "Add cutn_utils header files")

foreach(BACKEND ${PL_BACKEND})
    if("${PL_TENSOR_METHOD}" STREQUAL "mps" AND NOT CUTN_UTILS_ADDED)
        add_subdirectory(mps)
        target_include_directories(${PL_TENSOR}_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/mps)
        target_link_libraries(${PL_TENSOR}_utils INTERFACE mps_cutn_utils)
        set(CUDA_UTILS_ADDED TRUE)
    endif()
endforeach()

target_include_directories(${PL_TENSOR}_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PL_TENSOR}_utils INTERFACE lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET ${PL_TENSOR}_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
