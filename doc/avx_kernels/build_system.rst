AVX2/512 kernel build system
############################


In PennyLane-Lightning, a kernel is registered to :cpp:class`Pennylane::DynamicDispatcher` when the library is loaded, and it is used in runtime when the library consider it is the most suitable kernel under the given input.

To support AVX2 and AVX512 kernels, we always compile those kernels if the target system is UNIX on x86-64. Specifically, we made separate C++ files for AVX2 and AVX512 kernels and build them with corresponding compile options. This is handles by CMake. Following is a relevant code block from ``gates/CMakeLists.txt`` file.

.. code-block:: cmake

   if (UNIX AND (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(AMD64)|(X64)|(x64)|(x86_64)"))
       message(STATUS "Compiling for x86. Enable AVX2/AVX512 kernels (runtime enabled).")

       add_library(lightning_gates_register_kernels_avx2 STATIC RegisterKernels_AVX2.cpp GateUtil.cpp)
       target_compile_options(lightning_gates_register_kernels_avx2 PRIVATE "-mavx2;-mfma")
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


Thus AVX2/512 kernels are always included in the binary when compiled for UNIX compatible OSs on x86-64 architecture. However, we register these kernels to `DynamicDispatcher` only when runtime environment supports these architecture set.
