project(lightning_gates)

# We make a static library for default kernels using explicit template instantiations.
# We thus ensure that only a single instantiation exists during the build process.

add_library(lightning_gates INTERFACE)
target_include_directories(lightning_gates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (ENABLE_GATE_DISPATCHER AND UNIX AND (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(AMD64)|(X64)|(x64)|(x86_64)"))
    message(STATUS "Compiling for x86. Enable AVX2/AVX512 kernels (runtime enabled).")

    add_library(lightning_gates_register_kernels_avx2 STATIC RegisterKernels_AVX2.cpp)
    target_compile_options(lightning_gates_register_kernels_avx2 PRIVATE "-mavx2;-mfma")
    target_link_libraries(lightning_gates_register_kernels_avx2 PRIVATE lightning_external_libs lightning_compile_options lightning_utils)
    target_include_directories(lightning_gates_register_kernels_avx2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(lightning_gates_register_kernels_avx2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(lightning_gates_register_kernels_avx512 STATIC RegisterKernels_AVX512.cpp)
    target_compile_options(lightning_gates_register_kernels_avx512 PRIVATE "-mavx512f")
    target_link_libraries(lightning_gates_register_kernels_avx512 PRIVATE lightning_external_libs lightning_compile_options lightning_utils)
    target_include_directories(lightning_gates_register_kernels_avx512 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(lightning_gates_register_kernels_avx512 PROPERTIES POSITION_INDEPENDENT_CODE ON)
    
    add_library(lightning_gates_register_kernels_x64 STATIC RegisterKernels_x64.cpp cpu_kernels/GateImplementationsLM.cpp cpu_kernels/GateImplementationsPI.cpp GateUtil.cpp)
    target_link_libraries(lightning_gates_register_kernels_x64 PRIVATE lightning_external_libs lightning_compile_options lightning_utils)
    target_include_directories(lightning_gates_register_kernels_x64 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(lightning_gates_register_kernels_x64 PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(lightning_gates INTERFACE lightning_gates_register_kernels_x64 lightning_gates_register_kernels_avx2 lightning_gates_register_kernels_avx512)
else()
    add_library(lightning_gates_register_kernels_default STATIC RegisterKernels_Default.cpp cpu_kernels/GateImplementationsLM.cpp cpu_kernels/GateImplementationsPI.cpp GateUtil.cpp)
    target_include_directories(lightning_gates_register_kernels_default PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(lightning_gates_register_kernels_default PRIVATE lightning_external_libs lightning_compile_options lightning_utils)
    set_target_properties(lightning_gates_register_kernels_default PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(lightning_gates INTERFACE lightning_gates_register_kernels_default)
endif()
