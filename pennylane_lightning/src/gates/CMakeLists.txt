project(lightning_gates)

set(GATES_FILES GateUtil.cpp CACHE INTERNAL "" FORCE)

add_library(lightning_gates STATIC ${GATES_FILES})
target_link_libraries(lightning_gates PRIVATE lightning_compile_options
                                              lightning_external_libs
                                              lightning_utils)

if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(AMD64)|(X64)|(x64)|(x86_64)"))
    message(STATUS "Compiling for x86. Enable AVX2/AVX512 kernels (runtime enabled).")

    add_library(lightning_gates_register_kernels_avx2 STATIC RegisterKernels_AVX2.cpp GateUtil.cpp)
    target_compile_options(lightning_gates_register_kernels_avx2 PRIVATE "-mavx2")
    target_link_libraries(lightning_gates_register_kernels_avx2 PRIVATE lightning_external_libs lightning_compile_options lightning_utils)
    target_include_directories(lightning_gates_register_kernels_avx2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(lightning_gates_register_kernels_avx2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(lightning_gates_register_kernels_avx512 STATIC RegisterKernels_AVX512.cpp GateUtil.cpp)
    target_compile_options(lightning_gates_register_kernels_avx512 PRIVATE "-mavx512f")
    target_link_libraries(lightning_gates_register_kernels_avx512 PRIVATE lightning_external_libs lightning_compile_options lightning_utils)
    target_include_directories(lightning_gates_register_kernels_avx512 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(lightning_gates_register_kernels_avx512 PROPERTIES POSITION_INDEPENDENT_CODE ON)
    
    add_library(lightning_gates_register_kernels_x64 STATIC RegisterKernels_x64.cpp GateUtil.cpp)
    target_link_libraries(lightning_gates_register_kernels_x64 PRIVATE lightning_external_libs lightning_compile_options lightning_utils)
    target_include_directories(lightning_gates_register_kernels_x64 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(lightning_gates_register_kernels_x64 PRIVATE lightning_gates_register_kernels_avx2 lightning_gates_register_kernels_avx512)
    set_target_properties(lightning_gates_register_kernels_x64 PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(lightning_gates PRIVATE lightning_gates_register_kernels_x64)
else()
    add_library(lightning_gates_register_kernels_default STATIC RegisterKernels_Default.cpp GateUtil.cpp)
    target_include_directories(lightning_gates_register_kernels_default PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(lightning_gates_register_kernels_default PRIVATE lightning_external_libs lightning_compile_options lightning_utils)
    set_target_properties(lightning_gates_register_kernels_default PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(lightning_gates PRIVATE lightning_gates_register_kernels_default)
endif()

target_include_directories(lightning_gates PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_property(TARGET lightning_gates PROPERTY POSITION_INDEPENDENT_CODE ON)
