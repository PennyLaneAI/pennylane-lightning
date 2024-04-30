cmake_minimum_required(VERSION 3.20)
project(lightning_tensor_utils LANGUAGES CXX)

add_library(lightning_tensor_utils INTERFACE)

set(CUTN_UTILS_ADDED FALSE CACHE BOOL "Add cutn_utils header files")


foreach(BACKEND ${PL_BACKEND})
    if("${PL_TENSOR_BACKEND}" STREQUAL "cutensornet" AND NOT CUTN_UTILS_ADDED)
        add_subdirectory(cutn_utils)
        target_include_directories(lightning_tensor_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cutn_utils)
        target_link_libraries(lightning_tensor_utils INTERFACE cutn_utils)
        set(CUDA_UTILS_ADDED TRUE)
    endif()
endforeach()

target_include_directories(lightning_tensor_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_tensor_utils INTERFACE lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET lightning_tensor_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
    enable_testing()
    #add_subdirectory("tests")
endif()