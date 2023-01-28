# Implementation of PennyLane-Lightning AVX2/512 kernels

Each gate operation is implemented in a class with a corresponding name. For example, SWAP operation is implemented in `ApplySwap` class defined in [ApplySwap.cpp](ApplySwap.cpp) file. 

Depending on the wires gates apply to, we use two (for single-qubit operations), three (for symmetric two-qubit operators), and four (for non-symmetric two-qubit operators) functions to implement each gate.
For single-qubit operations, functions `applyInternal` corresponds to intra-register gate operations  and `applyExternal` corresponds to inter-register gate operations.
For two-qubit operations, we have `applyInternalInternal` (both wires act internally), `applyInternalExternal` (control wire acts internally whereas target wire acts externally), `applyExternalInternal` (target wire acts internally whereas control wire acts externally), and `applyExternalExternal` (both wires act externally).


Most cases, we implement a gate operation by splitting it into permutations, multiplications, and summations. These operations are translated into intrinsics in the compile time using C++ template mechanism.
Permutations and factors for multiplications are often obtained from functions. Those functions are named by concatenating the function name it is called with `Permutation` or `Factor`. For example, `applyInternalInternalPermutation` returns a permutation that is required for `applyInternalInternal` function.


See [the document](https://docs.pennylane.ai/projects/lightning/en/stable/avx_kernels/implementation.html) for details of the implementation.
