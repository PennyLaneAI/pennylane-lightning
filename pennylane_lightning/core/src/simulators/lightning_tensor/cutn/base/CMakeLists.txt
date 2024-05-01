cmake_minimum_required(VERSION 3.20)

project(lightning_tensor_tensornetBase LANGUAGES CXX)

add_library(lightning_tensor_tensornetBase INTERFACE)

target_include_directories(lightning_tensor_tensornetBase INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_tensor_tensornetBase INTERFACE  lightning_compile_options
                                                lightning_external_libs
                                                )

set_property(TARGET lightning_tensor_tensornetBase PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
