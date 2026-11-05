cmake_minimum_required(VERSION 3.26)

project(lightning_cliffish_tableau LANGUAGES CXX)

add_library(lightning_cliffish_tableau INTERFACE)

target_include_directories(lightning_cliffish_tableau INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_cliffish_tableau INTERFACE lightning_compile_options
                                                         lightning_external_libs
                                                         lightning_utils
                                                         lightning_cliffish_pauli
                                                         )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
