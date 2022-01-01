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
#     pennylane_lightning_compile_options pennylane_lightning_external_libs)

add_executable(gate_benchmark_oplist gate_benchmark_oplist.cpp)
target_link_libraries(gate_benchmark_oplist lightning_utils lightning_simulator
    pennylane_lightning_compile_options pennylane_lightning_external_libs)
