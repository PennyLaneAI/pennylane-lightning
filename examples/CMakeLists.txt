#############################
## I. Set project details
#############################
cmake_minimum_required(VERSION 3.14)
#set(CMAKE_POLICY_DEFAULT_CMP0127 NEW)

project("gate_benchmark"
        VERSION 0.1.0
        DESCRIPTION "Benchmark of parametric & non-parametric gates."
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)

option(ENABLE_CLANG_TIDY "Enable clang-tidy build checks" OFF)

if(ENABLE_CLANG_TIDY)
    if (NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                            -extra-arg=-std=c++17;
    )
endif()

add_executable(gate_benchmark gate_benchmark.cpp)
target_link_libraries(gate_benchmark lightning_utils lightning_simulator
    pennylane_lightning_compile_options pennylane_lightning_external_libs)

add_executable(gate_benchmark_oplist gate_benchmark_oplist.cpp)
target_link_libraries(gate_benchmark_oplist lightning_utils lightning_simulator
    pennylane_lightning_compile_options pennylane_lightning_external_libs)
