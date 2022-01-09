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

using Pennylane::Internal::callGateOps;
using Pennylane::Internal::GateOpsFuncPtrPairs;

using Pennylane::Constant::available_kernels;
using Pennylane::Constant::gate_names;
using Pennylane::Constant::gate_num_params;
} // namespace

/**
 * @brief return a lambda function for the given kernel and gate operation
 *
 * As we want the lamba function to be stateless, kernel and gate_op are
 * template paramters. In C++20, one may use a template lambda function instead.
 */
template <class fp_t, class ParamT, KernelType kernel, GateOperations gate_op>
constexpr auto gateOpToFunctor() {
    return [](std::complex<fp_t> *data, size_t num_qubits,
              const std::vector<size_t> &wires, bool inverse,
              const std::vector<fp_t> &params) {
        constexpr size_t num_params = static_lookup<gate_op>(gate_num_params);
        auto &&func = static_lookup<gate_op>(
            GateOpsFuncPtrPairs<fp_t, ParamT, kernel, num_params>::value);
        callGateOps(std::forward<decltype(func)>(func), data, num_qubits, wires,
                    inverse, params);
    };
}

/// @cond DEV
template <class fp_t, class ParamT, KernelType kernel, size_t gate_idx>
constexpr auto constructGateOpsFunctorTupleIter() {
    if constexpr (gate_idx ==
                  SelectGateOps<fp_t, kernel>::implemented_gates.size()) {
        return std::tuple{};
    } else if (gate_idx <
               SelectGateOps<fp_t, kernel>::implemented_gates.size()) {
        constexpr auto gate_op =
            SelectGateOps<fp_t, kernel>::implemented_gates[gate_idx];
        if constexpr (gate_op == GateOperations::Matrix) {
            /* GateOperations::Matrix is not supported for dynamic dispatch now */
            return constructGateOpsFunctorTupleIter<fp_t, ParamT, kernel, gate_idx + 1>();
        } else {
            return prepend_to_tuple(
                std::pair{gate_op, gateOpToFunctor<fp_t, ParamT, kernel, gate_op>()},
                constructGateOpsFunctorTupleIter<fp_t, ParamT, kernel, gate_idx + 1>());
        }
    }
}
/// @endcond

/**
 * @brief Generate array of all functors
 *
 * TODO: use std::vector and std::transform in C++20 which become constexpr
 */
template <class fp_t, class ParamT, KernelType kernel>
constexpr auto constructGateOpsFunctorTuple() {
    return constructGateOpsFunctorTupleIter<fp_t, ParamT, kernel, 0>();
};

/**
 * @brief Register all implemented gates for a given kernel
 */
template <class fp_t, class ParamT, KernelType kernel>
void registerAllImplementedGateOps() {
    auto &dispatcher = DynamicDispatcher<fp_t>::getInstance();

    constexpr auto gateFunctorPairs = constructGateOpsFunctorTuple<fp_t, ParamT, kernel>();

    auto registerGateToDispatcher = [&dispatcher] (auto&& gate_op_func_pair) {
        const auto& [gate_op, func] = gate_op_func_pair;
        std::string op_name = std::string(lookup(gate_names, gate_op));
        dispatcher.registerGateOperation(op_name, kernel, func);
        return gate_op;
    };

    std::apply([&registerGateToDispatcher](auto... elt) {
        std::make_tuple(registerGateToDispatcher(elt)...);
    }, gateFunctorPairs);
}

template <class fp_t, class ParamT, size_t idx> void registerKernelIter() {
    if constexpr (idx == available_kernels.size()) {
        return;
    } else {
        registerAllImplementedGateOps<fp_t, ParamT,
                                      std::get<0>(available_kernels[idx])>();
        registerKernelIter<fp_t, ParamT, idx + 1>();
    }
}

template <class fp_t, class ParamT> auto registerAllAvailableKernels() -> int {
    registerKernelIter<fp_t, ParamT, 0>();
    return 0;
}

template <class fp_t> struct registerBeforeMain { static const int dummy; };

template <>
const int registerBeforeMain<float>::dummy =
    registerAllAvailableKernels<float, float>();

template <>
const int registerBeforeMain<double>::dummy =
    registerAllAvailableKernels<double, double>();
/// @endcond
