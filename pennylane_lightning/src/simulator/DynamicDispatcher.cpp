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

#include <unordered_set>

/// @cond DEV
namespace {
using Pennylane::DynamicDispatcher;
using Pennylane::GateOperations;
using Pennylane::KernelType;
using Pennylane::SelectGateOps;

using Pennylane::static_lookup;

using Pennylane::Constant::available_kernels;
using Pennylane::Constant::gate_names;
using Pennylane::Constant::gate_num_params;
} // namespace

#define PENNYLANE_APPLY_OPS_TO_LAMBDA(GATE_NAME)                               \
    template <class fp_t, KernelType kernel, int num_params>                   \
    struct Apply##GATE_NAME##ToLambda {                                        \
        static_assert((num_params <= 1) || (num_params == 3),                  \
                      "The given number of parameters is not supported.");     \
    };                                                                         \
    template <class fp_t, KernelType kernel>                                   \
    struct Apply##GATE_NAME##ToLambda<fp_t, kernel, 0> {                       \
        static auto createFunctor() ->                                         \
            typename DynamicDispatcher<fp_t>::Func {                           \
            return [](std::complex<fp_t> *data, size_t num_qubits,             \
                      const std::vector<size_t> &wires, bool inverse,          \
                      [[maybe_unused]] const std::vector<fp_t> &params) {      \
                assert(params.empty());                                        \
                SelectGateOps<fp_t, kernel>::apply##GATE_NAME(                 \
                    data, num_qubits, wires, inverse);                         \
            };                                                                 \
        }                                                                      \
    };                                                                         \
    template <class fp_t, KernelType kernel>                                   \
    struct Apply##GATE_NAME##ToLambda<fp_t, kernel, 1> {                       \
        static auto createFunctor() ->                                         \
            typename DynamicDispatcher<fp_t>::Func {                           \
            return [](std::complex<fp_t> *data, size_t num_qubits,             \
                      const std::vector<size_t> &wires, bool inverse,          \
                      [[maybe_unused]] const std::vector<fp_t> &params) {      \
                assert(params.size() == 1);                                    \
                SelectGateOps<fp_t, kernel>::apply##GATE_NAME(                 \
                    data, num_qubits, wires, inverse, params[0]);              \
            };                                                                 \
        }                                                                      \
    };                                                                         \
    template <class fp_t, KernelType kernel>                                   \
    struct Apply##GATE_NAME##ToLambda<fp_t, kernel, 3> {                       \
        static auto createFunctor() ->                                         \
            typename DynamicDispatcher<fp_t>::Func {                           \
            return [](std::complex<fp_t> *data, size_t num_qubits,             \
                      const std::vector<size_t> &wires, bool inverse,          \
                      [[maybe_unused]] const std::vector<fp_t> &params) {      \
                assert(params.size() == 3);                                    \
                SelectGateOps<fp_t, kernel>::apply##GATE_NAME(                 \
                    data, num_qubits, wires, inverse, params[0], params[1],    \
                    params[2]);                                                \
            };                                                                 \
        }                                                                      \
    };

PENNYLANE_APPLY_OPS_TO_LAMBDA(PauliX)
PENNYLANE_APPLY_OPS_TO_LAMBDA(PauliY)
PENNYLANE_APPLY_OPS_TO_LAMBDA(PauliZ)
PENNYLANE_APPLY_OPS_TO_LAMBDA(Hadamard)
PENNYLANE_APPLY_OPS_TO_LAMBDA(S)
PENNYLANE_APPLY_OPS_TO_LAMBDA(T)
PENNYLANE_APPLY_OPS_TO_LAMBDA(RX)
PENNYLANE_APPLY_OPS_TO_LAMBDA(RY)
PENNYLANE_APPLY_OPS_TO_LAMBDA(RZ)
PENNYLANE_APPLY_OPS_TO_LAMBDA(PhaseShift)
PENNYLANE_APPLY_OPS_TO_LAMBDA(Rot)
PENNYLANE_APPLY_OPS_TO_LAMBDA(ControlledPhaseShift)
PENNYLANE_APPLY_OPS_TO_LAMBDA(CNOT)
PENNYLANE_APPLY_OPS_TO_LAMBDA(CZ)
PENNYLANE_APPLY_OPS_TO_LAMBDA(SWAP)
PENNYLANE_APPLY_OPS_TO_LAMBDA(CRX)
PENNYLANE_APPLY_OPS_TO_LAMBDA(CRY)
PENNYLANE_APPLY_OPS_TO_LAMBDA(CRZ)
PENNYLANE_APPLY_OPS_TO_LAMBDA(CRot)
PENNYLANE_APPLY_OPS_TO_LAMBDA(Toffoli)
PENNYLANE_APPLY_OPS_TO_LAMBDA(CSWAP)
PENNYLANE_APPLY_OPS_TO_LAMBDA(GeneratorPhaseShift)
PENNYLANE_APPLY_OPS_TO_LAMBDA(GeneratorCRX)
PENNYLANE_APPLY_OPS_TO_LAMBDA(GeneratorCRY)
PENNYLANE_APPLY_OPS_TO_LAMBDA(GeneratorCRZ)
PENNYLANE_APPLY_OPS_TO_LAMBDA(GeneratorControlledPhaseShift)

#define PENNYLANE_GATE_OP_FUNCTOR_PAIR(GATE_NAME)                              \
    {                                                                          \
        GateOperations::GATE_NAME,                                             \
            Apply##GATE_NAME##ToLambda<                                        \
                fp_t, kernel,                                                  \
                static_lookup<GateOperations::GATE_NAME>(                      \
                    gate_num_params)>::createFunctor()                         \
    }

template <class fp_t, KernelType kernel> void registerAllImplementedGateOps() {
    auto &dispatcher = DynamicDispatcher<fp_t>::getInstance();

    const std::unordered_map<GateOperations,
                             typename DynamicDispatcher<fp_t>::Func>
        all_gate_ops{
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(PauliX),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(PauliY),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(PauliZ),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(Hadamard),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(S),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(T),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(RX),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(RY),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(RZ),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(PhaseShift),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(Rot),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(ControlledPhaseShift),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(CNOT),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(CZ),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(SWAP),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(CRX),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(CRY),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(CRZ),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(CRot),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(Toffoli),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(CSWAP),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(GeneratorPhaseShift),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(GeneratorCRX),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(GeneratorCRY),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(GeneratorCRZ),
            PENNYLANE_GATE_OP_FUNCTOR_PAIR(GeneratorControlledPhaseShift),
        };

    for (const auto gate_op : SelectGateOps<fp_t, kernel>::implemented_gates) {
        if (gate_op == GateOperations::Matrix) {
            // applyMatrix is not supported by this dynamic dispatcher
            continue;
        }
        auto iter = all_gate_ops.find(gate_op);
        std::string op_name = std::string(lookup(gate_names, gate_op));
        if (iter == all_gate_ops.cend()) {
            // implemented gates is not in all_gate_ops; something wrong.
            PL_ABORT("An implemented gate " + op_name +
                     " is not found in all_gate_ops. "
                     "Check this variable.");
        }
        dispatcher.registerGateOperation(op_name, kernel, iter->second);
    }
}

template <class fp_t, size_t idx> void registerKernelIter() {
    if constexpr (idx == available_kernels.size()) {
        return;
    } else {
        registerAllImplementedGateOps<fp_t,
                                      std::get<0>(available_kernels[idx])>();
        registerKernelIter<fp_t, idx + 1>();
    }
}

template <class fp_t> constexpr auto registerAllAvailableKernels() -> int {
    registerKernelIter<fp_t, 0>();
    return 0;
}

template <class fp_t> struct registerBeforeMain { static const int dummy; };

template <>
const int
    registerBeforeMain<float>::dummy = registerAllAvailableKernels<float>();

template <>
const int
    registerBeforeMain<double>::dummy = registerAllAvailableKernels<double>();
/// @endcond
