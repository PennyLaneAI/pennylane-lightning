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
 * @file DynamicDispatcher.cpp
 * Register gate operations to dynamic dispatcher.
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
 * template paramters (or the functions can be consteval in C++20).
 * In C++20, one also may use a template lambda function instead.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * @tparam ParamT Floating point type for parameters
 * @tparam kernel Kernel for the gate operation
 * @tparam gate_op Gate operation to make a functor
 */
template <class PrecisionT, class ParamT, KernelType kernel,
          GateOperations gate_op>
constexpr auto gateOpToFunctor() {
    return [](std::complex<PrecisionT> *data, size_t num_qubits,
              const std::vector<size_t> &wires, bool inverse,
              const std::vector<PrecisionT> &params) {
        constexpr size_t num_params = static_lookup<gate_op>(gate_num_params);
        constexpr auto func_ptr = static_lookup<gate_op>(
            GateOpsFuncPtrPairs<PrecisionT, ParamT, kernel, num_params>::value);
        // This line is added as static_lookup cannnot raise exception
        // statically in GCC 9.
        static_assert(func_ptr != nullptr,
                      "Function pointer for the gate is not "
                      "included in GateOpsFuncPtrPairs.");
        callGateOps(func_ptr, data, num_qubits, wires, inverse, params);
    };
}

/// @cond DEV
/**
 * @brief Internal recursion function for constructGateOpsFunctorTuple
 */
template <class PrecisionT, class ParamT, KernelType kernel, size_t gate_idx>
constexpr auto constructGateOpsFunctorTupleIter() {
    if constexpr (gate_idx ==
                  SelectGateOps<PrecisionT, kernel>::implemented_gates.size()) {
        return std::tuple{};
    } else if (gate_idx <
               SelectGateOps<PrecisionT, kernel>::implemented_gates.size()) {
        constexpr auto gate_op =
            SelectGateOps<PrecisionT, kernel>::implemented_gates[gate_idx];
        if constexpr (gate_op == GateOperations::Matrix) {
            /* GateOperations::Matrix is not supported for dynamic dispatch now
             */
            return constructGateOpsFunctorTupleIter<PrecisionT, ParamT, kernel,
                                                    gate_idx + 1>();
        } else {
            return prepend_to_tuple(
                std::pair{
                    gate_op,
                    gateOpToFunctor<PrecisionT, ParamT, kernel, gate_op>()},
                constructGateOpsFunctorTupleIter<PrecisionT, ParamT, kernel,
                                                 gate_idx + 1>());
        }
    }
}
/// @endcond

/**
 * @brief Generate a tuple of gate operation and function pointer pairs.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam kernel Kernel to construct tuple
 */
template <class PrecisionT, class ParamT, KernelType kernel>
constexpr auto constructGateOpsFunctorTuple() {
    return constructGateOpsFunctorTupleIter<PrecisionT, ParamT, kernel, 0>();
};

/**
 * @brief Register all implemented gates for a given kernel
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam kernel Kernel to construct tuple
 */
template <class PrecisionT, class ParamT, KernelType kernel>
void registerAllImplementedGateOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    constexpr auto gateFunctorPairs =
        constructGateOpsFunctorTuple<PrecisionT, ParamT, kernel>();

    auto registerGateToDispatcher = [&dispatcher](auto &&gate_op_func_pair) {
        const auto &[gate_op, func] = gate_op_func_pair;
        std::string op_name = std::string(lookup(gate_names, gate_op));
        dispatcher.registerGateOperation(op_name, kernel, func);
        return gate_op;
    };

    std::apply(
        [&registerGateToDispatcher](auto... elt) {
            std::make_tuple(registerGateToDispatcher(elt)...);
        },
        gateFunctorPairs);
}

/// @cond DEV
/**
 * @brief Internal function to iterate over all available kerenls in
 * the compile time
 */
template <class PrecisionT, class ParamT, size_t idx>
void registerKernelIter() {
    if constexpr (idx == available_kernels.size()) {
        return;
    } else {
        registerAllImplementedGateOps<PrecisionT, ParamT,
                                      std::get<0>(available_kernels[idx])>();
        registerKernelIter<PrecisionT, ParamT, idx + 1>();
    }
}
/// @endcond

/**
 * @brief Register all implemented gates for all available kernels.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * @tparam ParamT Floating point type for parameters
 */
template <class PrecisionT, class ParamT>
auto registerAllAvailableKernels() -> int {
    registerKernelIter<PrecisionT, ParamT, 0>();
    return 0;
}

template <class PrecisionT> struct registerBeforeMain {
    static const int dummy;
};

template <>
const int registerBeforeMain<float>::dummy =
    registerAllAvailableKernels<float, float>();

template <>
const int registerBeforeMain<double>::dummy =
    registerAllAvailableKernels<double, double>();
/// @endcond
