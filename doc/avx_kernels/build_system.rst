AVX2/512 kernel build system
############################


In PennyLane-Lightning, a kernel is registered to :cpp:class:`Pennylane::DynamicDispatcher` when the library is loaded, and it is used at runtime when it is the most suitable kernel for the given input.

To support AVX2 and AVX512 kernels, we always compile those kernels if the target system is UNIX on x86-64. Specifically, we made separate C++ files for AVX2 and AVX512 kernels and build them as a static library with the corresponding compile options. This is handled by CMake. One can check ``gates/CMakeLists.txt`` file for details.

One caveat is that we want to make sure that default kernels (``KernelType::PI`` and ``KernelType::LM``) are only instantiated once with specific compiler flags during the compile process. This is important as the linker sometimes cannot choose the right instantiation when there are multiple instantiations of the same template class. 
This problem does not arise when all instantiations are compiled with the same options, but with the AVX2/512 kernels we use different compile options for each translation unit. We solve this problem by adding explicit instantiation declarations in the header files for these kernels (:ref:`file_pennylane_lightning_src_gates_cpu_kernels_GateImplementationsLM.hpp` and :ref:`file_pennylane_lightning_src_gates_cpu_kernels_GateImplementationsPI.hpp`) and compile them as a separate static library.

With this the AVX2/512 kernels are always included in the binary when compiled for UNIX compatible OSs on x86-64 architecture. However, we register these kernels to :cpp:class:`Pennylane::DynamicDispatcher` only when runtime environment supports these architecture sets.

.. literalinclude:: ../../pennylane_lightning/src/gates/RegisterKernels_x64.cpp
   :lines: 27-53

Likewise, we also inform :cpp:class:`Pennylane::KernelMap::OperationKernelMap` to use AVX2/512 kernels when aligned memory is used.

.. literalinclude:: ../../pennylane_lightning/src/simulator/KernelMap_X64.cpp
   :lines: 30-43

