# C++ tests for PennyLane-Lightning

Gate implementations (kernels) are tested in `Test_GateImplementations_*.cpp` files.

As some of the kernels (AVX2 and AVX512) are only runnable in the corresponding architecture, we cannot test all kernels.
Even though it is possible to test available kernels by detecting the architecture, we currently use the approach below to simplify the test codes:

In `Test_GateImplementations_(Param|Nonparam|Matrix).cpp` files only test default kernels (`LM` and `PI`) which are independent to the architecture.

In `Test_GateImplementations_(Inverse|CompareKernels|Generator).cpp` files run tests registered to `DynamicDispatcher`. As we register kernels to `DynamicDispatcher` by detecting the runtime architecture, these files test all accessible kernels.
