// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Defines
 */
#include "DynamicDispatcher.hpp"

namespace {
    using Pennylane::KernelType;
    using Pennylane::GateOperations;
    using Pennylane::SelectGateOps;
    using Pennylane::DynamicDispatcher;

    using Pennylane::AVAILABLE_KERNELS;
    using Pennylane::GATE_NAMES;
}

#define PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA_PARAMS0(GATE_NAME)                              \
    case GateOperations::GATE_NAME:                                                        \
        return [](std::complex<fp_t>* data, size_t num_qubits,                             \
                    const std::vector<size_t>& wires, bool inverse,                        \
                    [[maybe_unused]] const std::vector<fp_t>& params) {                    \
        assert(params.empty());                                                            \
        SelectGateOps<fp_t, kernel>::apply##GATE_NAME(data, num_qubits, wires, inverse);   \
    };
#define PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA_PARAMS1(GATE_NAME)                              \
    case GateOperations::GATE_NAME:                                                        \
        return [](std::complex<fp_t>* data, size_t num_qubits,                             \
                  const std::vector<size_t>& wires, bool inverse,                          \
                  [[maybe_unused]] const std::vector<fp_t>& params) {                      \
        assert(params.size() == 1);                                                        \
        SelectGateOps<fp_t, kernel>::apply##GATE_NAME(data, num_qubits, wires, inverse,    \
                                                      params[0]);                          \
    };
#define PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA_PARAMS3(GATE_NAME)                              \
    case GateOperations::GATE_NAME:                                                        \
        return [](std::complex<fp_t>* data, size_t num_qubits,                             \
                    const std::vector<size_t>& wires, bool inverse,                        \
                    [[maybe_unused]] const std::vector<fp_t>& params) {                    \
        assert(params.size() == 3);                                                        \
        SelectGateOps<fp_t, kernel>::apply##GATE_NAME(data, num_qubits, wires, inverse,    \
                                                      params[0], params[1], params[2]);    \
    };
#define PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(GATE_NAME, NUM_PARAMS)                          \
        PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA_PARAMS##NUM_PARAMS(GATE_NAME)

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
template<class fp_t, KernelType kernel> constexpr auto createFunctor(GateOperations op)
    -> typename DynamicDispatcher<fp_t>::Func {
    switch(op) { 
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(PauliX, 0)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(PauliY, 0)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(PauliZ, 0)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(Hadamard, 0)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(S, 0)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(T, 0)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(RX, 1)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(RY, 1)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(RZ, 1)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(PhaseShift, 1)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(Rot, 3)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(ControlledPhaseShift, 1)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(CNOT, 0)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(CZ, 0)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(SWAP, 0)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(CRX, 1)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(CRY, 1)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(CRZ, 1)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(CRot, 3)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(Toffoli, 0)
    PENNYLANE_KERNEL_CASE_OP_TO_LAMBDA(CSWAP, 0)

    default:
        return nullptr;
    }
}

template <class fp_t, KernelType kernel>
void registerAllImplementedGateOps() {
    auto &dispatcher = DynamicDispatcher<fp_t>::getInstance();

    for (const auto gate_op: SelectGateOps<fp_t, kernel>::implemented_gates) {
        const auto name = std::string(lookup(GATE_NAMES, gate_op));
        dispatcher.registerGateOperation(name, kernel, createFunctor<fp_t, kernel>(gate_op));
    }
}

template <class fp_t, size_t idx>
void registerKernelIter() {
    if constexpr (idx == AVAILABLE_KERNELS.size()) {
        return;
    } else {
        registerAllImplementedGateOps<fp_t, std::get<0>(AVAILABLE_KERNELS[idx])>();
        registerKernelIter<fp_t, idx+1>();
    }
}

template<class fp_t>
constexpr auto registerAllAvailableKernels() -> int {
    registerKernelIter<fp_t, 0>();
    return 0;
}


template <class fp_t> 
struct registerBeforeMain {
    static const int dummy;
};

template <>
const int registerBeforeMain<float>::dummy =
    registerAllAvailableKernels<float>();

template <>
const int registerBeforeMain<double>::dummy =
    registerAllAvailableKernels<double>();
