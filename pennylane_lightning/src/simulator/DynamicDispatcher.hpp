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
 * @file DynamicDispatcher.hpp
 * Defines DynamicDispatcher class. Can be used to call a gate operation by
 * string.
 */

#pragma once

#include "Error.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "SelectGateOps.hpp"

#include <complex>
#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace Pennylane::Internal {
struct PairHash {
    size_t operator()(const std::pair<std::string, KernelType> &p) const {
        return std::hash<std::string>()(p.first) ^
               std::hash<int>()(static_cast<int>(p.second));
    }
};

} // namespace Pennylane::Internal

namespace Pennylane {

/**
 * @brief DynamicDispatcher class
 *
 * This class is used to call a gate operation dynamically
 */
template <typename PrecisionT> class DynamicDispatcher {
  public:
    using CFP_t = std::complex<PrecisionT>;

    using GateFunc = std::function<void(
        std::complex<PrecisionT> * /*data*/, size_t /*num_qubits*/,
        const std::vector<size_t> & /*wires*/, bool /*inverse*/,
        const std::vector<PrecisionT> & /*params*/)>;

    using GeneratorFunc = PrecisionT (*)(std::complex<PrecisionT> * /*data*/,
                                         size_t /*num_qubits*/,
                                         const std::vector<size_t> & /*wires*/,
                                         bool /*adjoint*/);

  private:
    std::unordered_map<std::string, size_t> gate_wires_;

    std::unordered_map<std::string, KernelType> gate_kernel_map_;
    std::unordered_map<std::string, KernelType> generator_kernel_map_;

    std::unordered_map<std::pair<std::string, KernelType>, GateFunc,
                       Internal::PairHash>
        gates_;

    std::unordered_map<std::pair<std::string, KernelType>, GeneratorFunc,
                       Internal::PairHash>
        generators_;

    std::string removeGeneratorPrefix(const std::string &op_name) {
        constexpr std::string_view prefix = "Generator";
        // TODO: change to string::starts_with in C++20
        if (op_name.rfind(prefix) != 0) {
            return op_name;
        }
        return op_name.substr(prefix.size());
    }
    std::string_view removeGeneratorPrefix(std::string_view op_name) {
        constexpr std::string_view prefix = "Generator";
        // TODO: change to string::starts_with in C++20
        if (op_name.rfind(prefix) != 0) {
            return op_name;
        }
        return op_name.substr(prefix.size());
    }

    DynamicDispatcher() {
        for (const auto &[gate_op, n_wires] : Constant::gate_wires) {
            gate_wires_.emplace(lookup(Constant::gate_names, gate_op), n_wires);
        }

        for (const auto &[gate_op, gate_name] : Constant::gate_names) {
            KernelType kernel =
                lookup(Constant::default_kernel_for_gates, gate_op);
            const auto implemented_gates = implementedGatesForKernel(kernel);
            if (std::find(std::cbegin(implemented_gates),
                          std::cend(implemented_gates),
                          gate_op) == std::cend(implemented_gates)) {
                PL_ABORT("Default kernel for " + std::string(gate_name) +
                         " does not implement the gate.");
            }
            gate_kernel_map_.emplace(gate_name, kernel);
        }

        for (const auto &[gntr_op, gntr_name] : Constant::generator_names) {
            KernelType kernel =
                lookup(Constant::default_kernel_for_generators, gntr_op);
            const auto implemented_generators =
                implementedGeneratorsForKernel(kernel);
            if (std::find(std::cbegin(implemented_generators),
                          std::cend(implemented_generators),
                          gntr_op) == std::cend(implemented_generators)) {
                PL_ABORT("Default kernel for " + std::string(gntr_name) +
                         " does not implement the generator.");
            }
            generator_kernel_map_.emplace(removeGeneratorPrefix(gntr_name),
                                          kernel);
        }
    }

  public:
    static DynamicDispatcher &getInstance() {
        static DynamicDispatcher singleton;
        return singleton;
    }

    /**
     * @brief Register a new gate operation for the operation. Can pass a custom
     * kernel
     */
    template <typename FunctionType>
    void registerGateOperation(const std::string &op_name, KernelType kernel,
                               FunctionType &&func) {
        // TODO: Add mutex when we go to multithreading
        gates_.emplace(std::make_pair(op_name, kernel),
                       std::forward<FunctionType>(func));
    }

    /**
     * @brief Register a new gate generator for the operation. Can pass a custom
     * kernel
     */
    template <typename FunctionType>
    void registerGeneratorOperation(const std::string &op_name,
                                    KernelType kernel, FunctionType &&func) {
        // TODO: Add mutex when we go to multithreading
        generators_.emplace(
            std::make_pair(removeGeneratorPrefix(op_name), kernel),
            std::forward<FunctionType>(func));
    }

    /**
     * @brief Apply a single gate to the state-vector using the given kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param op_name Gate operation name.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(KernelType kernel, CFP_t *data, size_t num_qubits,
                        const std::string &op_name,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<PrecisionT> &params = {}) const {
        const auto iter = gates_.find(std::make_pair(op_name, kernel));
        if (iter == gates_.cend()) {
            throw std::invalid_argument(
                "Cannot find a gate with a given name \"" + op_name + "\".");
        }
        const auto gate_wire_iter = gate_wires_.find(op_name);
        if ((gate_wire_iter != gate_wires_.end()) &&
            (gate_wire_iter->second != wires.size())) {
            throw std::invalid_argument(
                std::string("The supplied gate requires ") +
                std::to_string(gate_wire_iter->second) + " wires, but " +
                std::to_string(wires.size()) + " were supplied.");
            // TODO: change to std::format in C++20
        }
        (iter->second)(data, num_qubits, wires, inverse, params);
    }

    /**
     * @brief Apply a single gate to the state-vector using a registered kernel
     *
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param op_name Gate operation name.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    inline void
    applyOperation(CFP_t *data, size_t num_qubits, const std::string &op_name,
                   const std::vector<size_t> &wires, bool inverse,
                   const std::vector<PrecisionT> &params = {}) const {
        const auto kernel_iter = gate_kernel_map_.find(op_name);
        if (kernel_iter == gate_kernel_map_.end()) {
            PL_ABORT("Kernel for gate " + op_name + " is not registered.");
        }

        applyOperation(kernel_iter->second, data, num_qubits, op_name, wires,
                       inverse, params);
    }

    /**
     * @brief Apply multiple gates to the state-vector using a registered kernel
     *
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param ops List of Gate operation names.
     * @param wires List of wires to apply each gate to.
     * @param inverse List of inverses
     * @param params List of parameters
     */
    void applyOperations(CFP_t *data, size_t num_qubits,
                         const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &wires,
                         const std::vector<bool> &inverse,
                         const std::vector<std::vector<PrecisionT>> &params) {
        const size_t numOperations = ops.size();
        if (numOperations != wires.size() || numOperations != params.size()) {
            throw std::invalid_argument(
                "Invalid arguments: number of operations, wires, and "
                "parameters must all be equal");
        }

        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(data, num_qubits, ops[i], wires[i], inverse[i],
                           params[i]);
        }
    }

    /**
     * @brief Apply multiple (non-paramterized) gates to the state-vector
     * using a registered kernel
     *
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param ops List of Gate operation names.
     * @param wires List of wires to apply each gate to.
     * @param inverse List of inverses
     * @param params List of parameters
     */
    void applyOperations(CFP_t *data, size_t num_qubits,
                         const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &wires,
                         const std::vector<bool> &inverse) {
        const size_t numOperations = ops.size();
        if (numOperations != wires.size()) {
            throw std::invalid_argument(
                "Invalid arguments: number of operations, wires, and "
                "parameters must all be equal");
        }

        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(data, num_qubits, ops[i], wires[i], inverse[i], {});
        }
    }

    /**
     * @brief Apply a single generator to the state-vector using the given
     * kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param op_name Gate operation name.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of gate.
     */
    auto applyGenerator(KernelType kernel, CFP_t *data, size_t num_qubits,
                        const std::string &op_name,
                        const std::vector<size_t> &wires, bool adj) const
        -> PrecisionT {
        const auto iter = generators_.find(std::make_pair(op_name, kernel));
        if (iter == generators_.cend()) {
            throw std::invalid_argument(
                "Cannot find a gate with a given name \"" + op_name + "\".");
        }
        return (iter->second)(data, num_qubits, wires, adj);
    }

    /**
     * @brief Apply a single gate to the state-vector using a registered kernel
     *
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param op_name Gate operation name.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     */
    inline auto applyGenerator(CFP_t *data, size_t num_qubits,
                               const std::string &op_name,
                               const std::vector<size_t> &wires, bool adj) const
        -> PrecisionT {
        const auto kernel_iter = generator_kernel_map_.find(op_name);
        if (kernel_iter == generator_kernel_map_.end()) {
            PL_ABORT("Kernel for gate " + op_name + " is not registered.");
        }

        return applyGenerator(kernel_iter->second, data, num_qubits, op_name,
                              wires, adj);
    }
};
/*******************************************************************************
 * The functions below are only used for register kernels to the dynamic
 * dispatcher.
 ******************************************************************************/

namespace Internal {

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
template <class PrecisionT, class ParamT, class GateImplementation,
          GateOperation gate_op>
constexpr auto gateOpToFunctor() {
    return [](std::complex<PrecisionT> *data, size_t num_qubits,
              const std::vector<size_t> &wires, bool inverse,
              const std::vector<PrecisionT> &params) {
        constexpr auto func_ptr =
            GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                  gate_op>::value;
        callGateOps(func_ptr, data, num_qubits, wires, inverse, params);
    };
}

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
        if constexpr (gate_op == GateOperation::Matrix) {
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
                      GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                                 gntr_op>::value},
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
 * @tparam kernel Kernel to construct tuple
 */
template <class PrecisionT, class ParamT, class GateImplementation>
constexpr static auto gate_op_functor_tuple = constructGateOpsFunctorTupleIter<
    PrecisionT, ParamT, GateImplementation, 0>();

/**
 * @brief Tuple of gate operation and function pointer pairs.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam kernel Kernel to construct tuple
 */
template <class PrecisionT, class GateImplementation>
constexpr static auto generator_op_functor_tuple =
    constructGeneratorOpsFunctorTupleIter<PrecisionT, GateImplementation, 0>();

/**
 * @brief Register all implemented gates for a given kernel
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam kernel Kernel to construct tuple
 */
template <class PrecisionT, class ParamT, class GateImplementation>
void registerAllImplementedGateOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto registerGateToDispatcher = [&dispatcher](
                                        const auto &gate_op_func_pair) {
        const auto &[gate_op, func] = gate_op_func_pair;
        std::string op_name =
            std::string(lookup(Constant::gate_names, gate_op));
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
 * @tparam ParamT Floating point type of gate parameters
 * @tparam kernel Kernel to construct tuple
 */
template <class PrecisionT, class GateImplementation>
void registerAllImplementedGeneratorOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto registerGeneratorToDispatcher =
        [&dispatcher](const auto &gntr_op_func_pair) {
            const auto &[gntr_op, func] = gntr_op_func_pair;
            std::string op_name =
                std::string(lookup(Constant::generator_names, gntr_op));
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
 * @brief Internal function to iterate over all available kerenls in
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

/**
 * @brief Register all implemented gates for all available kernels.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * @tparam ParamT Floating point type for parameters
 */
template <class PrecisionT, class ParamT>
auto registerAllAvailableKernels() -> int {
    registerKernelIter<PrecisionT, ParamT, AvailableKernels>();
    return 0;
}
} // namespace Internal

template <class PrecisionT, class ParamT> struct registerBeforeMain {};

template <> struct registerBeforeMain<float, float> {
    static inline int dummy =
        Internal::registerAllAvailableKernels<float, float>();
};

template <> struct registerBeforeMain<double, double> {
    static inline int dummy =
        Internal::registerAllAvailableKernels<double, double>();
};

} // namespace Pennylane
