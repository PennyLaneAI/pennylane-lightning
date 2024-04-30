cmake_minimum_required(VERSION 3.20)
project(${PL_BACKEND}_tensor LANGUAGES CXX)

add_library(lightning_tensor_tensor INTERFACE)

add_subdirectory(base)

set(CUTN_UTILS_ADDED FALSE CACHE BOOL "Add cutn_utils header files")

if("${PL_TENSOR_BACKEND}" STREQUAL "cutensornet" AND NOT CUTN_UTILS_ADDED)
    add_subdirectory(cutn)
    target_include_directories(lightning_tensor_tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cutn)
    target_link_libraries(lightning_tensor_tensor INTERFACE cuda_tensor)
    set(CUDA_UTILS_ADDED TRUE)
endif()

target_include_directories(lightning_tensor_tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/base)
target_link_libraries(lightning_tensor_tensor INTERFACE tensorBase lightning_utils lightning_compile_options lightning_external_libs)

set_property(TARGET lightning_tensor_tensor PROPERTY POSITION_INDEPENDENT_CODE ON)