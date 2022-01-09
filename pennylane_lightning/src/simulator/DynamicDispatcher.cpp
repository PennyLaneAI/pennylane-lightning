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
using Pennylane::Internal::GateFuncPtrPairs;

using Pennylane::Constant::available_kernels;
using Pennylane::Constant::gate_names;
using Pennylane::Constant::gate_num_params;
} // namespace

/**
 * @brief return a lambda function for the given gate operation
 *
 * TODO: gate_op can be a function parameter (instead of template parameter).
 */
template <class fp_t, class ParamT, KernelType kernel, GateOperations gate_op>
constexpr auto getFunctor() {
    return [](std::complex<fp_t> *data, size_t num_qubits,
              const std::vector<size_t> &wires, bool inverse,
              const std::vector<fp_t> &params) {
        constexpr size_t num_params = static_lookup<gate_op>(gate_num_params);
        auto &&func = static_lookup<gate_op>(
            GateFuncPtrPairs<fp_t, ParamT, kernel, num_params>::value);
        callGateOps<fp_t, ParamT, kernel>(
            std::forward<decltype(func)>(func), data, num_qubits, wires,
            inverse, params);
    };
}

/**
 * TODO: This function can be changed into usual constexpr function in C++20
 *
 * std::transform from implementaed gates to getFunctor will work
 * */
template <class fp_t, class ParamT, KernelType kernel> struct FunctorArray {
    static inline std::array<
        typename DynamicDispatcher<fp_t>::Func,
        SelectGateOps<fp_t, kernel>::implemented_gates.size()>
        value;

    template <size_t idx> static void constexpr constructIter() {
        constexpr auto gates = SelectGateOps<fp_t, kernel>::implemented_gates;
        if constexpr (idx < gates.size()) {
            value[idx] = getFunctor<fp_t, ParamT, kernel, gates[idx]>();
            constructIter<idx + 1>();
        }
    }

    static void constexpr construct() { constructIter<0>(); }
};

template <class fp_t, KernelType kernel> void registerAllImplementedGateOps() {
    auto &dispatcher = DynamicDispatcher<fp_t>::getInstance();

    FunctorArray<fp_t, fp_t, kernel>::construct();

    constexpr auto num_gates =
        SelectGateOps<fp_t, kernel>::implemented_gates.size();
    for (size_t i = 0; i < num_gates; i++) {
        const auto gate_op = SelectGateOps<fp_t, kernel>::implemented_gates[i];
        if (gate_op == GateOperations::Matrix) {
            // applyMatrix is not supported by this dynamic dispatcher
            continue;
        }
        std::string op_name = std::string(lookup(gate_names, gate_op));
        dispatcher.registerGateOperation(
            op_name, kernel, FunctorArray<fp_t, fp_t, kernel>::value[i]);
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
