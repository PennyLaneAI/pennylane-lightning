cmake_minimum_required(VERSION 3.26)

project(lightning_cliffish_testing LANGUAGES CXX)

# Test helpers shared by the Lightning-Cliffish component tests.
add_library(lightning_cliffish_testing INTERFACE)

target_include_directories(lightning_cliffish_testing INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_cliffish_testing INTERFACE lightning_cliffish_pauli)
