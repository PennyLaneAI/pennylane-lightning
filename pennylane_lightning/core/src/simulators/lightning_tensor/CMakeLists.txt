cmake_minimum_required(VERSION 3.21)
set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░█▀▀░█▀█░█░█
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░█░█░█▀▀░█░█
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░▀▀▀░▀░░░▀▀▀
]=])
message(${LOGO})

project(${PL_BACKEND}
    DESCRIPTION "Lightning-Tensor bindings for PennyLane. Backed by NVIDIA cuQuantum SDK."
    LANGUAGES CXX C CUDA
)

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_pllgpu.cmake")
findCUDATK(lightning_external_libs)
findCutensornet(lightning_external_libs)

target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_PLTENSOR=1")


#########################
## Set Default Options ##
#########################

add_executable(test_tensor_explore test6.cpp)

target_link_libraries(test_tensor_explore PRIVATE lightning_utils lightning_compile_options lightning_external_libs)