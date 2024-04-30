cmake_minimum_required(VERSION 3.20)

project(lightning_tensor_mps_base LANGUAGES CXX)

add_library(lightning_tensor_mps_base INTERFACE)

target_include_directories(lightning_tensor_mps_base INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_tensor_mps_base INTERFACE  lightning_compile_options
                                                lightning_external_libs
                                                )

set_property(TARGET lightning_tensor_mps_base PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
