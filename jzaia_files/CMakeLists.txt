project(lightning_demo LANGUAGES CXX)
add_executable(lq_grover main.cpp)
target_include_directories(lq_grover PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lq_grover lightning_qubit
                                lightning_qubit_observables
                                lightning_qubit_measurements
                                lightning_qubit_gates
                                lightning_gates
                                lq_gates_kernel_map
                                lq_gates_register_kernels_x64
                        )

# Unsure if this is the best way to include the necessary gate implementations
target_include_directories(lq_grover PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../pennylane_lightning/core/src/simulators/lightning_qubit/gates/cpu_kernels/)
