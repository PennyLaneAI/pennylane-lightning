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

#include "SelectGateOps.hpp"

#include <complex>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <functional>

namespace Pennylane {

struct PairHash {
    size_t operator()(const std::pair<std::string, KernelType>& p) const {
        return std::hash<std::string>()(p.first) ^ 
               std::hash<int>()(static_cast<int>(p.second));
    }
};
/**
 * @brief DynamicDispatcher class
 *
 * This class controls all dynamic <-> static conversions.
 */
template<typename fp_t>
class DynamicDispatcher {
  public:
    using scalar_type_t = fp_t;
    using CFP_t = std::complex<scalar_type_t>;
    
  private:
    using Func = KernelFuncType<fp_t>;
    const std::unordered_map<std::string, size_t> gate_wires_{
        {"PauliX", 1},    {"PauliY", 1},   {"PauliZ", 1},
        {"Hadamard", 1},  {"S", 1},        {"T", 1},
        {"RX", 1},        {"RY", 1},       {"RZ", 1},
        {"PhaseShift", 1},{"Rot", 1},      {"ControlledPhaseShift", 2},
        {"CNOT", 2},      {"SZ", 2}  ,     {"SWAP", 2},
        {"CRX", 2},       {"CRY", 2},      {"CRZ", 2},
        {"CRot", 2},      {"Toffoli", 3},  {"CSWAP", 3}
    };
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

    std::unordered_map<std::pair<std::string, KernelType>, Func, PairHash> gates_;
    
    std::unordered_map<std::string, KernelType> kernel_for_ops_;

    DynamicDispatcher() {
        for(int idx = 0; idx < static_cast<int>(GateOperations::END); ++idx) {
            kernel_for_ops_[gate_names_[idx]] = DEFAULT_KERNEL_FOR_OPS[idx];
        }
    }

  public:
    static DynamicDispatcher& getInstance() {
        static DynamicDispatcher singleton;
        return singleton;
    }

    /**
     * @brief Register a new gate operation for the operation. Can pass a custom kernel
     */
    template<typename FunctionType>
    void registerGateOperation(std::string op_name, KernelType kernel_type, 
                               FunctionType&& func) {
        gates_.emplace(std::make_pair(std::move(op_name), kernel_type), func);
    }

    template<typename FunctionType>
    void updateKernelForOps(std::string op_name, KernelType kernel_type) {
        kernel_for_ops_.emplace(std::move(op_name), kernel_type);
    }

    /**
     * @brief call the corresponding operation function from GateOperations
     */
    void applyOperation(CFP_t* data, size_t num_qubits,
                        const std::string& op_name, const std::vector<size_t>& wires,
                        bool inverse, const std::vector<fp_t>& params) {

        const auto iter = gates_.find(std::make_pair(op_name, kernel_for_ops_[op_name]));
        if (iter == gates_.end()) {
            throw std::invalid_argument("Cannot find a gate with a given name \"" 
                    + op_name + "\".");
        }

        if (const auto requiredWires = gate_wires_.at(op_name); requiredWires != wires.size())
        {
            throw std::invalid_argument(std::string("The supplied gate requires ") +
                                        std::to_string(requiredWires) + " wires, but " +
                                        std::to_string(wires.size()) +
                                        " were supplied.");
        }
        (iter->second)(data, num_qubits, wires, inverse, params);
    }

    template<class Derived>
    void applyOperations(CFP_t* data, size_t num_qubits,
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
            applyOperation(data, num_qubits, ops[i], wires[i], inverse[i], params[i]);
        }
    }

    template<class Derived>
    void applyOperations(CFP_t* data, size_t num_qubits,
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
