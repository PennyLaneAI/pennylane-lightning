# C++ tests for PennyLane-Lightning

Gate implementations (kernels) are tested in `Test_GateImplementations_*.cpp` files.
In `Test_GateImplementations_(Param|Nonparam|Matrix).cpp` files test `LM` and `PI` kernels directly.

Other kernels (AVX2/AVX512) are tested by comparing their results to `LM` and `PI` kernels.
