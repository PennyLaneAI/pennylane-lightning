# cmake_minimum_required(VERSION 3.14)
# project(gate_benchmark)
#############################
## I. Set project details
#############################
cmake_minimum_required(VERSION 3.14)

project("gate_benchmark"
        VERSION 0.1.0
        DESCRIPTION "Benchmark of parametric & non-parametric gates."
        LANGUAGES CXX
)

#############################
## II. Fetch project
#############################

Include(FetchContent)

FetchContent_Declare(
    Pennylane-lightning
    GIT_REPOSITORY  https://github.com/PennyLaneAI/pennylane-lightning
    GIT_TAG         master
)
FetchContent_MakeAvailable(Pennylane-lightning)

#############################
## III. Create project target
#############################

set(CMAKE_CXX_STANDARD 17) # At least C++17 is required
set(CMAKE_CXX_FLAGS "-O3") # Some optimisation

add_executable(gate_benchmark gate_benchmark.cpp)
target_link_libraries(gate_benchmark pennylane_lightning)
