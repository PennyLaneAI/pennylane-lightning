// Copyright 2022 Xanadu Quantum Technologies Inc.

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
 * Register all gate and generator implementations
 */
#include "DynamicDispatcher.hpp"
#include "AvailableKernels.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "GateUtil.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "SelectKernel.hpp"

using namespace Pennylane;
using namespace Pennylane::Util;

/// @cond DEV
namespace {
/**
 * @brief return a lambda function for the given kernel and gate operation
 *
 * As we want the lambda function to be stateless, kernel and gate_op are
 * template parameters (or the functions can be consteval in C++20).
 * In C++20, one also may use a template lambda function instead.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * @tparam ParamT Floating point type for parameters>
 * @tparam GateImplementation Gate implementation class.
 * @tparam gate_op Gate operation to make a functor.
 */
template <class PrecisionT, class ParamT, class GateImplementation,
          Gates::GateOperation gate_op>
constexpr auto gateOpToFunctor() {
    return [](std::complex<PrecisionT> *data, size_t num_qubits,
              const std::vector<size_t> &wires, bool inverse,
              const std::vector<PrecisionT> &params) {
        constexpr auto func_ptr =
            Gates::GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                         gate_op>::value;
        assert(params.size() ==
               Gates::static_lookup<gate_op>(Gates::Constant::gate_num_params));
        Gates::callGateOps(func_ptr, data, num_qubits, wires, inverse, params);
    };
}
/// @endcond

/// @cond DEV
/**
 * @brief Internal recursion function for constructGateOpsFunctorTuple
 *
 * @return Tuple of gate operations and corresponding GateImplementation member
 * function pointers.
 */
template <class PrecisionT, class ParamT, class GateImplementation,
          size_t gate_idx>
constexpr auto constructGateOpsFunctorTupleIter() {
    if constexpr (gate_idx == GateImplementation::implemented_gates.size()) {
        return std::tuple{};
    } else if (gate_idx < GateImplementation::implemented_gates.size()) {
        constexpr auto gate_op =
            GateImplementation::implemented_gates[gate_idx];
        if constexpr (gate_op == Gates::GateOperation::Matrix) {
            /* GateOperation::Matrix is not supported for dynamic dispatch now
             */
            return constructGateOpsFunctorTupleIter<
                PrecisionT, ParamT, GateImplementation, gate_idx + 1>();
        } else {
            return prepend_to_tuple(
                std::pair{gate_op,
                          gateOpToFunctor<PrecisionT, ParamT,
                                          GateImplementation, gate_op>()},
                constructGateOpsFunctorTupleIter<
                    PrecisionT, ParamT, GateImplementation, gate_idx + 1>());
        }
    }
}
/**
 * @brief Internal recursion function for constructGateOpsFunctorTuple
 */
template <class PrecisionT, class GateImplementation, size_t gntr_idx>
constexpr auto constructGeneratorOpsFunctorTupleIter() {
    if constexpr (gntr_idx ==
                  GateImplementation::implemented_generators.size()) {
        return std::tuple{};
    } else if (gntr_idx < GateImplementation::implemented_generators.size()) {
        constexpr auto gntr_op =
            GateImplementation::implemented_generators[gntr_idx];
        return prepend_to_tuple(
            std::pair{gntr_op,
                      Gates::GeneratorOpToMemberFuncPtr<
                          PrecisionT, GateImplementation, gntr_op>::value},
            constructGeneratorOpsFunctorTupleIter<
                PrecisionT, GateImplementation, gntr_idx + 1>());
    }
}
/// @endcond

/**
 * @brief Tuple of gate operation and function pointer pairs.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class ParamT, class GateImplementation>
constexpr auto gate_op_functor_tuple = constructGateOpsFunctorTupleIter<
    PrecisionT, ParamT, GateImplementation, 0>();

/**
 * @brief Tuple of gate operation and function pointer pairs.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class GateImplementation>
constexpr auto generator_op_functor_tuple =
    constructGeneratorOpsFunctorTupleIter<PrecisionT, GateImplementation, 0>();

/**
 * @brief Register all implemented gates for a given kernel
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class ParamT, class GateImplementation>
void registerAllImplementedGateOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto registerGateToDispatcher = [&dispatcher](
                                        const auto &gate_op_func_pair) {
        const auto &[gate_op, func] = gate_op_func_pair;
        std::string op_name =
            std::string(lookup(Gates::Constant::gate_names, gate_op));
        dispatcher.registerGateOperation(op_name, GateImplementation::kernel_id,
                                         func);
        return gate_op;
    };

    [[maybe_unused]] const auto registerd_gate_ops = std::apply(
        [&registerGateToDispatcher](auto... elt) {
            return std::make_tuple(registerGateToDispatcher(elt)...);
        },
        gate_op_functor_tuple<PrecisionT, ParamT, GateImplementation>);
}
/**
 * @brief Register all implemented generators for a given kernel
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class GateImplementation>
void registerAllImplementedGeneratorOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto registerGeneratorToDispatcher =
        [&dispatcher](const auto &gntr_op_func_pair) {
            const auto &[gntr_op, func] = gntr_op_func_pair;
            std::string op_name =
                std::string(lookup(Gates::Constant::generator_names, gntr_op));
            dispatcher.registerGeneratorOperation(
                op_name, GateImplementation::kernel_id, func);
            return gntr_op;
        };

    [[maybe_unused]] const auto registerd_gate_ops = std::apply(
        [&registerGeneratorToDispatcher](auto... elt) {
            return std::make_tuple(registerGeneratorToDispatcher(elt)...);
        },
        generator_op_functor_tuple<PrecisionT, GateImplementation>);
}

/// @cond DEV
/**
 * @brief Internal function to iterate over all available kernels in
 * the compile time
 */
template <class PrecisionT, class ParamT, class TypeList>
void registerKernelIter() {
    if constexpr (std::is_same_v<TypeList, void>) {
        return;
    } else {
        registerAllImplementedGateOps<PrecisionT, ParamT,
                                      typename TypeList::Type>();
        registerAllImplementedGeneratorOps<PrecisionT,
                                           typename TypeList::Type>();
        registerKernelIter<PrecisionT, ParamT, typename TypeList::Next>();
    }
}
/// @endcond
} // namespace

/// @cond DEV
namespace Pennylane::Internal {
template <class PrecisionT, class ParamT> int registerAllAvailableKernels() {
    registerKernelIter<PrecisionT, ParamT, AvailableKernels>();
    return 1;
}
/// @endcond

// explicit instantiations
template int registerAllAvailableKernels<float, float>();
template int registerAllAvailableKernels<double, double>();

} // namespace Pennylane::Internal
