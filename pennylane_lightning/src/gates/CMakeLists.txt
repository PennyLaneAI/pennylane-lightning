project(lightning_gates)

set(GATES_FILES GateUtil.cpp CACHE INTERNAL "" FORCE)

# We make a static library for default kernels using explicit template instantiations.
# We thus ensure that only a single instantiation exists during the build process.
add_library(lightning_gates_register_kernels_default STATIC RegisterKernels_Default.cpp cpu_kernels/GateImplementationsLM.cpp cpu_kernels/GateImplementationsPI.cpp)
target_include_directories(lightning_gates_register_kernels_default PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_gates_register_kernels_default PRIVATE lightning_external_libs lightning_compile_options lightning_utils)

add_library(lightning_gates STATIC ${GATES_FILES})
target_link_libraries(lightning_gates PRIVATE lightning_compile_options
                                              lightning_external_libs
                                              lightning_utils
                                              lightning_gates_register_kernels_default)
target_include_directories(lightning_gates PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_property(TARGET lightning_gates PROPERTY POSITION_INDEPENDENT_CODE ON)
