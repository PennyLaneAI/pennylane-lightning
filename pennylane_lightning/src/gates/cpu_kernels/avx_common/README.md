# Implementation of PennyLane-Lightning AVX2/512 kernels

Each operation is defined using a class with a corresponding name. For example, SWAP operation is implemented in `ApplySwap` class defined in [ApplySwap.cpp](ApplySwap.cpp) file. 


Implementation of each single qubit operation is divided by two cases depending on the wire the operation applies to. First, if the operation acts within packed data, we load data to a AVX register and process them within the register. Functions named `applyInternal` in each file implement these operations. For example, AVX512 register packs 512 bits (4 complex double-precision numbers), so applying a single-qubit gate with `rev_wire` (defined by `num_qubits - 1 - wire`) equals 0 or 1 (less than $log_2 (4)$) calls these functions.
Second, if the operation acts between packed data, we load data to AVX registers and apply inter-register computations. Functions named `applyExternal` in each file do this job.
As in the previous example, calling a gate with `rev_wire` larger than or equals to 2 will use these function.
More detailed implementation of single-qubit gates can be found [here](https://docs.pennylane.ai/projects/lightning/en/stable/avx_kernels/index.html).


We also implement two-qubit gates that have two wires (`control` and `target`).
As there are four cases that `control` or `target` wires can act, we implement them using functions named
`applyInternalInternal` (both wires act on the packed data), `applyInternalExternal` (only the control wire acts on the packed data), `applyExternalInternal` (only the target wire acts on the packed data), `applyExternalExternal` (both wires acts outside of the packed data).
When the gate is symmetric (`control` and `target` wires act the same when they are swapped), we discard one cases and set `symmetric=true` in the class.

Most cases, we implement a gate operation by splitting it into permutations, multiplications, and summations. These operations are translated into intrinsics in the compile time using C++ template mechanism.
