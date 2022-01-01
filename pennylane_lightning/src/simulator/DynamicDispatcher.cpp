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

    using Pennylane::GATE_NAMES;
}

#define PENNYLANE_SPECIALIZE_TO_FUNCTOR_PARAMS0(GATE_NAME)                                 \
template <class fp_t, KernelType kernel>                                                   \
struct toFunctor<fp_t, kernel, GateOperations::GATE_NAME> {                                \
    void operator()(std::complex<fp_t>* data, size_t num_qubits,                           \
                    const std::vector<size_t>& wires, bool inverse,                        \
                    [[maybe_unused]] const std::vector<fp_t>& params) {                    \
        assert(params.empty());                                                            \
        SelectGateOps<fp_t, kernel>::apply##GATE_NAME(data, num_qubits, wires, inverse);   \
    }                                                                                      \
};
#define PENNYLANE_SPECIALIZE_TO_FUNCTOR_PARAMS1(GATE_NAME)                                 \
template <class fp_t, KernelType kernel>                                                   \
struct toFunctor<fp_t, kernel, GateOperations::GATE_NAME> {                                \
    void operator()(std::complex<fp_t>* data, size_t num_qubits,                           \
                    const std::vector<size_t>& wires, bool inverse,                        \
                    const std::vector<fp_t>& params) {                                     \
        assert(params.size() == 1);                                                        \
        SelectGateOps<fp_t, kernel>::apply##GATE_NAME(data, num_qubits, wires, inverse,    \
                                                      params[0]);                          \
    }                                                                                      \
};
#define PENNYLANE_SPECIALIZE_TO_FUNCTOR_PARAMS3(GATE_NAME)                                 \
template <class fp_t, KernelType kernel>                                                   \
struct toFunctor<fp_t, kernel, GateOperations::GATE_NAME> {                                \
    void operator()(std::complex<fp_t>* data, size_t num_qubits,                           \
                    const std::vector<size_t>& wires, bool inverse,                        \
                    const std::vector<fp_t>& params) {                                     \
        assert(params.size() == 3);                                                        \
        SelectGateOps<fp_t, kernel>::apply##GATE_NAME(data, num_qubits, wires, inverse,    \
                                                      params[0], params[1], params[2]);    \
    }                                                                                      \
};
#define PENNYLANE_SPECIALIZE_TO_FUNCTOR(GATE_NAME, NUM_PARAMS) \
        PENNYLANE_SPECIALIZE_TO_FUNCTOR_PARAMS##NUM_PARAMS(GATE_NAME)

template <class fp_t, KernelType kernel, GateOperations op>
struct toFunctor {
    void operator()(std::complex<fp_t>* data, size_t num_qubits, 
                    const std::vector<size_t>& wires, bool inverse, 
                    const std::vector<fp_t>& params);
};

PENNYLANE_SPECIALIZE_TO_FUNCTOR(PauliX, 0)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(PauliY, 0)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(PauliZ, 0)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(Hadamard, 0)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(S, 0)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(T, 0)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(RX, 1)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(RY, 1)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(RZ, 1)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(PhaseShift, 1)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(Rot, 3)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(ControlledPhaseShift, 1)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(CNOT, 0)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(CZ, 0)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(SWAP, 0)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(CRX, 1)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(CRY, 1)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(CRZ, 1)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(CRot, 3)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(Toffoli, 0)
PENNYLANE_SPECIALIZE_TO_FUNCTOR(CSWAP, 0)

template <class fp_t, KernelType kernel, GateOperations op>
constexpr auto toFunctorTuple() -> 
std::tuple<std::string_view, KernelType, typename DynamicDispatcher<fp_t>::FunctionType>
{
    return std::make_tuple(lookup(GATE_NAMES, op), kernel, toFunctor<fp_t, kernel, op>());
}

template <class fp_t, KernelType kernel, size_t idx>
void registerImplementedGateIter(DynamicDispatcher<fp_t>& dispatcher) {
    if constexpr (idx == SelectGateOps<fp_t, kernel>::implemented_gates.size()) {
        return;
    }
    else {
        constexpr auto op = SelectGateOps<fp_t, kernel>::implemented_gates[idx];
        const auto name = std::string(lookup(GATE_NAMES, op));
        dispatcher.registerGateOperation(name,
                                         kernel, toFunctor<fp_t, kernel, op>());

        registerImplementedGateIter<fp_t, kernel, idx + 1>(dispatcher);
    }
}

template <class fp_t, KernelType kernel> int registerAllImplementedGateOps() {
    auto &dispatcher = DynamicDispatcher<fp_t>::getInstance();
    registerImplementedGateIter<fp_t, kernel, 0>(dispatcher);
    return 0;
}

template <class fp_t, KernelType kernel> struct registerBeforeMain {
    static const int dummy;
};

/* Explicit instantiations */
template struct registerBeforeMain<float, KernelType::PI>;
template struct registerBeforeMain<float, KernelType::LM>;
template struct registerBeforeMain<double, KernelType::PI>;
template struct registerBeforeMain<double, KernelType::LM>;

template <>
const int registerBeforeMain<float, KernelType::PI>::dummy =
    registerAllImplementedGateOps<float, KernelType::PI>();

template <>
const int registerBeforeMain<float, KernelType::LM>::dummy =
    registerAllImplementedGateOps<float, KernelType::LM>();

template <>
const int registerBeforeMain<double, KernelType::PI>::dummy =
    registerAllImplementedGateOps<double, KernelType::PI>();

template <>
const int registerBeforeMain<double, KernelType::LM>::dummy =
    registerAllImplementedGateOps<double, KernelType::LM>();
