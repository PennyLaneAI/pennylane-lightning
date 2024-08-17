cmake_minimum_required(VERSION 3.20)

project(lightning_utils LANGUAGES CXX)

set(LQUBIT_UTILS_FILES RuntimeInfo.cpp CACHE INTERNAL "" FORCE)


add_library(lightning_utils STATIC ${LQUBIT_UTILS_FILES})

list(APPEND LQUBIT_UTILS_FILES UtilLinearAlg.cpp)

set(CUDA_UTILS_ADDED FALSE CACHE BOOL "Add cuda_utils header files")

foreach(BACKEND ${PL_BACKEND})
    if("${BACKEND}" STREQUAL "lightning_gpu" OR "${PL_TENSOR_BACKEND}" STREQUAL "cutensornet"  AND NOT CUDA_UTILS_ADDED)
        add_subdirectory(cuda_utils)
        target_include_directories(lightning_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_utils)
        target_link_libraries(lightning_utils INTERFACE cuda_utils)
        set(CUDA_UTILS_ADDED TRUE)
    endif()
endforeach()

target_include_directories(lightning_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_utils INTERFACE lightning_compile_options
                                                lightning_external_libs)


if(ENABLE_PYTHON)
    add_compile_definitions(_ENABLE_PYTHON)
else()
    target_link_libraries(lightning_utils INTERFACE pybind11::module
                                                  pybind11::pybind11
                                                  pybind11::embed)
endif()

if(USE_PYTHON_HEADERS)
    add_compile_definitions(_USE_PYTHON_HEADERS=1)
    target_include_directories(lightning_utils INTERFACE ${PYTHON_INCLUDE_DIR})
    target_link_libraries(lightning_utils INTERFACE ${PYTHON_LIBRARY})
endif()

set_property(TARGET lightning_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()