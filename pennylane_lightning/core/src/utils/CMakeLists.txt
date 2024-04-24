cmake_minimum_required(VERSION 3.20)

project(lightning_utils LANGUAGES CXX)

set(LQUBIT_UTILS_FILES RuntimeInfo.cpp CACHE INTERNAL "" FORCE)


add_library(lightning_utils STATIC ${LQUBIT_UTILS_FILES})

if(ENABLE_LAPACK)
    list(APPEND LQUBIT_UTILS_FILES UtilLinearAlg.cpp)
endif()

set(CUDA_UTILS_ADDED FALSE CACHE BOOL "Add cuda_utils header files")

foreach(BACKEND ${PL_BACKEND})
    if(${BACKEND} STREQUAL "lightning_gpu" AND NOT CUDA_UTILS_ADDED)
        add_subdirectory(cuda_utils)
        target_include_directories(lightning_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_utils)
        target_link_libraries(lightning_utils INTERFACE cuda_utils)
        set(CUDA_UTILS_ADDED TRUE)
    endif()
endforeach()


target_include_directories(lightning_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_utils INTERFACE lightning_compile_options
                                                lightning_external_libs
                                                )


set_property(TARGET lightning_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()