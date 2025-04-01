cmake_minimum_required(VERSION 3.20)

project(${PL_BACKEND}_measurements LANGUAGES CXX C CUDA)

if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
endif()

set(LTENSOR_MPS_FILES cuda_kernels_measures.cu CACHE INTERNAL "" FORCE)

add_library(${PL_BACKEND}_measurements STATIC ${LTENSOR_MPS_FILES})

target_include_directories(${PL_BACKEND}_measurements INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PL_BACKEND}_measurements PUBLIC  lightning_compile_options
                                                            lightning_external_libs
                                                            ${PL_BACKEND}
                                                            ${PL_BACKEND}_utils
                                                            ${PL_BACKEND}_observables
                                                            )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
