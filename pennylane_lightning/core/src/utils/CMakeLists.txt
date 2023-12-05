cmake_minimum_required(VERSION 3.20)

project(lightning_utils LANGUAGES CXX)

set(LQUBIT_UTILS_FILES RuntimeInfo.cpp CACHE INTERNAL "" FORCE)
add_library(lightning_utils STATIC ${LQUBIT_UTILS_FILES})

target_include_directories(lightning_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} "/opt/cray/pe/libsci/23.09.1.1/gnu/103/x86_64/include/")
target_link_directories(lightning_utils INTERFACE "/opt/cray/pe/libsci/23.09.1.1/gnu/103/x86_64/lib/")
target_link_libraries(lightning_utils INTERFACE lightning_compile_options
                                                lightning_external_libs
                                                ${LAPACK_LIBRARIES}
                                                sci_gnu
                                                )
set_property(TARGET lightning_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()