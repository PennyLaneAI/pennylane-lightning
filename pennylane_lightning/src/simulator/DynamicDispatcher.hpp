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

#pragma once

#include "Error.hpp"
#include "SelectGateOps.hpp"

#include <complex>
#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace Pennylane {

struct PairHash {
    size_t operator()(const std::pair<std::string, KernelType> &p) const {
        return std::hash<std::string>()(p.first) ^
               std::hash<int>()(static_cast<int>(p.second));
    }
};
/**
 * @brief DynamicDispatcher class
 *
 * This class controls all dynamic <-> static conversions.
 */
template <typename fp_t> class DynamicDispatcher {
  public:
    using scalar_type_t = fp_t;
    using CFP_t = std::complex<scalar_type_t>;

  private:
    using Func = KernelFuncType<fp_t>;
    const std::unordered_map<std::string, size_t> gate_wires_{
        {"PauliX", 1},     {"PauliY", 1},  {"PauliZ", 1},
        {"Hadamard", 1},   {"S", 1},       {"T", 1},
        {"RX", 1},         {"RY", 1},      {"RZ", 1},
        {"PhaseShift", 1}, {"Rot", 1},     {"ControlledPhaseShift", 2},
        {"CNOT", 2},       {"SZ", 2},      {"SWAP", 2},
        {"CRX", 2},        {"CRY", 2},     {"CRZ", 2},
        {"CRot", 2},       {"Toffoli", 3}, {"CSWAP", 3}};
    const std::vector<std::string> gate_names_{
        /* Single-qubit gates */
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "T",
        "RX",
        "RY",
        "RZ",
        "PhaseShift",
        "Rot",
        /* Two-qubit gates */
        "ControlledPhaseShift",
        "CNOT",
        "CZ",
        "SWAP",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        /* Three-qubit gates */
        "Toffoli",
        "CSWAP",
    };

    std::unordered_map<std::pair<std::string, KernelType>, Func, PairHash>
        gates_;

    std::unordered_map<std::string, KernelType> kernel_map_;

    DynamicDispatcher() {
        for (const auto &[gate_op, gate_name] : GATE_NAMES) {
            kernel_map_.emplace(
                gate_name, dynamic_lookup(DEFAULT_KERNEL_FOR_OPS, gate_op));
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
        gates_.emplace(std::make_pair(op_name, kernel), func);
    }

    template <typename FunctionType>
    void updateKernelForOps(const std::string &op_name, KernelType kernel) {
        kernel_map_.emplace(op_name, kernel);
    }

    /**
     * @brief Apply a single gate to the state-vector using the given kernel.
     *
     * @param kernel Lernel to run the gate operation.
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
                        const std::vector<fp_t> &params) const {
        const auto iter = gates_.find(std::make_pair(op_name, kernel));
        if (iter == gates_.cend()) {
            throw std::invalid_argument(
                "Cannot find a gate with a given name \"" + op_name + "\".");
        }

        if (const auto requiredWires = gate_wires_.at(op_name);
            requiredWires != wires.size()) {
            throw std::invalid_argument(
                std::string("The supplied gate requires ") +
                std::to_string(requiredWires) + " wires, but " +
                std::to_string(wires.size()) + " were supplied.");
        }
        (iter->second)(data, num_qubits, wires, inverse, params);
    }

    /**
     */
    inline void applyOperation(CFP_t *data, size_t num_qubits,
                               const std::string &op_name,
                               const std::vector<size_t> &wires, bool inverse,
                               const std::vector<fp_t> &params) const {
        const auto kernel_iter = kernel_map_.find(op_name);
        if (kernel_iter == kernel_map_.end()) {
            PL_ABORT("Kernel for gate " + op_name + " is not registered.");
        }

        applyOperation(kernel_iter->second, data, num_qubits, op_name, wires,
                       inverse, params);
    }

    void applyOperations(CFP_t *data, size_t num_qubits,
                         const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &wires,
                         const std::vector<bool> &inverse,
                         const std::vector<std::vector<fp_t>> &params) {
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
};
} // namespace Pennylane
