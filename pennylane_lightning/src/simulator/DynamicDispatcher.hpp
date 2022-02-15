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

#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "GateUtil.hpp"
#include "KernelType.hpp"

#include <cassert>
#include <complex>
#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

/// @cond DEV
namespace Pennylane::Internal {
struct PairHash {
    size_t
    operator()(const std::pair<std::string, Gates::KernelType> &p) const {
        return std::hash<std::string>()(p.first) ^
               std::hash<int>()(static_cast<int>(p.second));
    }
};
/**
 * @brief Register all implemented gates for all available kernels.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * @tparam ParamT Floating point type for parameters
 */
template <class PrecisionT, class ParamT> int registerAllAvailableKernels();
} // namespace Pennylane::Internal
/// @endcond

namespace Pennylane {
/**
 * @brief These functions are only used to register kernels to the dynamic
 * dispatcher.
 */
template <class PrecisionT, class ParamT> struct registerBeforeMain;

template <> struct registerBeforeMain<float, float> {
    static inline int dummy =
        Internal::registerAllAvailableKernels<float, float>();
};

template <> struct registerBeforeMain<double, double> {
    static inline int dummy =
        Internal::registerAllAvailableKernels<double, double>();
};

/**
 * @brief DynamicDispatcher class
 *
 * This class calls a gate/generator operation dynamically
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

    std::unordered_map<std::string, Gates::KernelType> gate_kernel_map_;
    std::unordered_map<std::string, Gates::KernelType> generator_kernel_map_;

    std::unordered_map<std::pair<std::string, Gates::KernelType>, GateFunc,
                       Internal::PairHash>
        gates_;

    std::unordered_map<std::pair<std::string, Gates::KernelType>, GeneratorFunc,
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
        using Gates::KernelType;
        for (const auto &[gate_op, n_wires] : Gates::Constant::gate_wires) {
            gate_wires_.emplace(
                Util::lookup(Gates::Constant::gate_names, gate_op), n_wires);
        }

        for (const auto &[gate_op, gate_name] : Gates::Constant::gate_names) {
            KernelType kernel = Util::lookup(
                Gates::Constant::default_kernel_for_gates, gate_op);
            const auto implemented_gates = implementedGatesForKernel(kernel);
            if (std::find(std::cbegin(implemented_gates),
                          std::cend(implemented_gates),
                          gate_op) == std::cend(implemented_gates)) {
                PL_ABORT("Default kernel for " + std::string(gate_name) +
                         " does not implement the gate.");
            }
            gate_kernel_map_.emplace(gate_name, kernel);
        }

        for (const auto &[gntr_op, gntr_name] :
             Gates::Constant::generator_names) {
            KernelType kernel = Util::lookup(
                Gates::Constant::default_kernel_for_generators, gntr_op);
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
    void registerGateOperation(const std::string &op_name,
                               Gates::KernelType kernel, FunctionType &&func) {
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
                                    Gates::KernelType kernel,
                                    FunctionType &&func) {
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
    void applyOperation(Gates::KernelType kernel, CFP_t *data,
                        size_t num_qubits, const std::string &op_name,
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
     * @brief Apply multiple (non-parameterized) gates to the state-vector
     * using a registered kernel
     *
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param ops List of Gate operation names.
     * @param wires List of wires to apply each gate to.
     * @param inverse List of inverses
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
    auto applyGenerator(Gates::KernelType kernel, CFP_t *data,
                        size_t num_qubits, const std::string &op_name,
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
} // namespace Pennylane
