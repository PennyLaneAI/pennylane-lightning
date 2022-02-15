#############################
## I. Set project details
#############################
cmake_minimum_required(VERSION 3.14)

project("gate_benchmark"
        VERSION 0.1.0
        DESCRIPTION "Benchmark of parametric & non-parametric gates."
        LANGUAGES CXX
)

# add_executable(gate_benchmark gate_benchmark.cpp)
# target_link_libraries(gate_benchmark lightning_utils lightning_simulator
#                                      lightning_compile_options
#                                      lightning_external_libs)

add_library(lightning_examples INTERFACE)
target_link_libraries(lightning_examples INTERFACE lightning_compile_options
                                                   lightning_external_libs
                                                   lightning_gates
                                                   lightning_simulator
                                                   lightning_utils)

add_executable(gate_benchmark_oplist gate_benchmark_oplist.cpp)
target_link_libraries(gate_benchmark_oplist PRIVATE lightning_examples)
add_executable(benchmark_multi_rz benchmark_multi_rz.cpp)
target_link_libraries(benchmark_multi_rz PRIVATE lightning_examples)

configure_file("compiler_info.in" "compiler_info.txt")

add_custom_command(TARGET gate_benchmark_oplist POST_BUILD 
                   COMMAND ${CMAKE_COMMAND} -E copy
                           ${PROJECT_SOURCE_DIR}/run_gate_benchmark.sh
                           ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run_gate_benchmark.sh
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                           ${PROJECT_SOURCE_DIR}/plot_gate_benchmark.py
                           ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plot_gate_benchmark.py
                   COMMAND ${CMAKE_COMMAND} -E rename
                           ${CMAKE_CURRENT_BINARY_DIR}/compiler_info.txt
                           ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/compiler_info.txt)
