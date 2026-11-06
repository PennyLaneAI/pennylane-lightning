cmake_minimum_required(VERSION 3.26)

project(lightning_cliffish_planner LANGUAGES CXX)

add_library(lightning_cliffish_planner INTERFACE)

target_include_directories(lightning_cliffish_planner INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_cliffish_planner INTERFACE lightning_compile_options
                                                         lightning_external_libs
                                                         lightning_utils
                                                         lightning_cliffish_program
                                                         lightning_cliffish_tableau
                                                         lightning_cliffish_dense
                                                         )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
