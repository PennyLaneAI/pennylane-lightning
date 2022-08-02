project(lightning_gates)

# We make a static library for default kernels using explicit template instantiations.
# We thus ensure that only a single instantiation exists during the build process.
add_library(lightning_gates_register_kernels_default STATIC RegisterKernels_Default.cpp cpu_kernels/GateImplementationsLM.cpp cpu_kernels/GateImplementationsPI.cpp)
target_include_directories(lightning_gates_register_kernels_default PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_gates_register_kernels_default PRIVATE lightning_external_libs lightning_compile_options lightning_utils)
set_target_properties(lightning_gates_register_kernels_default PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(lightning_gates_util STATIC GateUtil.cpp)
target_include_directories(lightning_gates_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_gates_util PRIVATE lightning_external_libs lightning_compile_options lightning_utils)
set_target_properties(lightning_gates_util PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(lightning_gates INTERFACE)
target_link_libraries(lightning_gates INTERFACE lightning_gates_register_kernels_default
                                                lightning_gates_util)
target_include_directories(lightning_gates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
