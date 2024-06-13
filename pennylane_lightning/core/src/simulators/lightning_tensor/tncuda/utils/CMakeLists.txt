cmake_minimum_required(VERSION 3.20)
project(${PL_BACKEND}_tncuda_utils LANGUAGES CXX)

add_library(${PL_BACKEND}_tncuda_utils INTERFACE)

set(TNCUDA_UTILS_ADDED FALSE CACHE BOOL "Add tncuda_utils header files")

foreach(BACKEND ${PL_BACKEND})
    if("${PL_TENSOR_METHOD}" STREQUAL "mps" AND NOT TNCUDA_UTILS_ADDED)
        add_subdirectory(mps)
        target_include_directories(${PL_BACKEND}_tncuda_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/mps)
        target_link_libraries(${PL_BACKEND}_tncuda_utils INTERFACE mps_tncuda_utils)
        set(TNCUDA_UTILS_ADDED TRUE)
    endif()
endforeach()

target_include_directories(${PL_BACKEND}_tncuda_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PL_BACKEND}_tncuda_utils INTERFACE lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET ${PL_BACKEND}_tncuda_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
