#############################
## I. Set project details
#############################
cmake_minimum_required(VERSION 3.14)
set(CMAKE_POLICY_DEFAULT_CMP0127 NEW)

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
# set(CMAKE_CXX_FLAGS "-O3") # Not including flags here as I pass them via commandline args in bash script (I would normally set it here)

add_executable(gate_benchmark gate_benchmark.cpp)
target_link_libraries(gate_benchmark pennylane_lightning)