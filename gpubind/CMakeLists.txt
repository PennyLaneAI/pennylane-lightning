cmake_minimum_required(VERSION 3.19...3.27)
project(gpu_binding_example_project LANGUAGES CXX CUDA)

# Find the Python interpreter and development libraries
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Fetch nanobind, which is required for the bindings
include(FetchContent)
FetchContent_Declare(nanobind
GIT_REPOSITORY https://github.com/wjakob/nanobind.git
GIT_TAG        v2.7.0
)
FetchContent_MakeAvailable(nanobind)

find_package(CUDAToolkit REQUIRED)

nanobind_add_module(
    gpu_binding_example_ext
    SHARED
    src/bindings.cpp
    src/tensor.cpp
    src/kernels.cu
)

# Link the CUDA runtime libraries to the module.
target_link_libraries(gpu_binding_example_ext PRIVATE CUDA::cudart)

