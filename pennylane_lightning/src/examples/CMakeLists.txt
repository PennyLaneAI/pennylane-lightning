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
