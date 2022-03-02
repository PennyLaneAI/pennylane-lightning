#############################
## I. Set project details
#############################
cmake_minimum_required(VERSION 3.14)

project("gate_benchmark"
        VERSION 0.1.0
        DESCRIPTION "Benchmark of parametric & non-parametric gates."
        LANGUAGES CXX
)

################################################################################
# II. Set dependencies
################################################################################

add_library(lightning_examples INTERFACE)
target_link_libraries(lightning_examples INTERFACE lightning_compile_options
                                                   lightning_external_libs
                                                   lightning_gates
                                                   lightning_simulator
                                                   lightning_utils)


################################################################################
# III. set executables
################################################################################

add_executable(benchmark_gate benchmark_gate.cpp)
target_link_libraries(benchmark_gate PRIVATE lightning_examples)

add_executable(benchmark_generator benchmark_generator.cpp)
target_link_libraries(benchmark_generator PRIVATE lightning_examples)

add_executable(benchmark_matrix benchmark_matrix.cpp)
target_link_libraries(benchmark_matrix PRIVATE lightning_examples)

# add_executable(benchmark_operation_float benchmark_operation.cpp)
# target_compile_options(benchmark_operation_float PRIVATE "-DUSE_SINGLE_PRECISION")
# target_link_libraries(benchmark_operation_float PRIVATE lightning_examples)

configure_file("compiler_info.in" "compiler_info.txt")

add_custom_command(TARGET benchmark_gate POST_BUILD 
                   COMMAND ${CMAKE_COMMAND} -E copy
                           ${PROJECT_SOURCE_DIR}/run_benchmark.py
                           ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run_benchmark.py
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                           ${PROJECT_SOURCE_DIR}/plot_benchmark.py
                           ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plot_benchmark.py
                   COMMAND ${CMAKE_COMMAND} -E rename
                           ${CMAKE_CURRENT_BINARY_DIR}/compiler_info.txt
                           ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/compiler_info.txt)
