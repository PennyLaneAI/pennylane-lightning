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

option(ENABLE_WARNINGS "Enable warnings" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy build checks" OFF)

if(ENABLE_CLANG_TIDY)
    if (NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                            -extra-arg=-std=c++17;
    )
endif()

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

add_executable(gate_benchmark gate_benchmark.cpp)
target_link_libraries(gate_benchmark pennylane_lightning)