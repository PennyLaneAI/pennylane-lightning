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
 * Defines a template class for choosing a Gate operations
 */
#pragma once

#include "GateOperationsPI.hpp"
#include "GateOperationsLM.hpp"

#include <array>
#include <functional>

namespace Pennylane {
/**
 * @brief enum class for all gate operations
 *
 * When you add a gate in this enum, please sure that GATE_NUM_PARAMS is also updated 
 * accordingly.
 * */
enum class GateOperations: int {
    /* Single-qubit gates */
    PauliX = 0,
    PauliY,
    PauliZ,
    Hadamard,
    S,
    T,
    RX,
    RY,
    RZ,
    PhaseShift,
    Rot,
    /* Two-qubit gates */
    ControlledPhaseShift,
    CNOT,
    CZ,
    SWAP,
    CRX,
    CRY,
    CRZ,
    CRot,
    /* Three-qubit gates */
    Toffoli,
    CSWAP,
    /* General matrix */
    Matrix,
    /* END */
    END
};

constexpr std::array<int, static_cast<int>(GateOperations::END) - 1>
GATE_NUM_PARAMS = {
    /* PuliX                = */ 0,
    /* PuliY                = */ 0,
    /* PuliZ                = */ 0,
    /* Hadamard             = */ 0,
    /* S                    = */ 0,
    /* T                    = */ 0,
    /* RX                   = */ 1,
    /* RY                   = */ 1,
    /* RZ                   = */ 1,
    /* PhaseShift           = */ 1,
    /* Rot                  = */ 3,
    /* ControlledPhaseShift = */ 1,
    /* CNOT                 = */ 0,
    /* CZ                   = */ 0,
    /* SWAP                 = */ 0,
    /* CRX                  = */ 1,
    /* CRY                  = */ 1,
    /* CRZ                  = */ 1,
    /* CRot                 = */ 3,
    /* Toffoli              = */ 0,
    /* CSWAP                = */ 0
};

enum class KernelType {PI, LM};

/**
 * @brief Define which kernel to use for each operation
 *
 * This value is used for:
 *   1) StateVector apply##GATE_NAME methods. The kernel function is statically binded to the 
 *   given kernel and cannot be modified.
 *   2) Default kernel functions of StateVector applyOperation(opName, ...) methods. The 
 *   kernel function is dynamically binded and can be changed using DynamicDispatcher singleton
 *   class.
 *   3) Python binding. 
 */
constexpr std::array<KernelType, static_cast<int>(GateOperations::END)>
DEFAULT_KERNEL_FOR_OPS = {
    /* PuliX                = */ KernelType::PI,
    /* PuliY                = */ KernelType::PI,
    /* PuliZ                = */ KernelType::PI,
    /* Hadamard             = */ KernelType::PI,
    /* S                    = */ KernelType::PI,
    /* T                    = */ KernelType::PI,
    /* RX                   = */ KernelType::PI,
    /* RY                   = */ KernelType::PI,
    /* RZ                   = */ KernelType::PI,
    /* PhaseShift           = */ KernelType::PI,
    /* Rot                  = */ KernelType::PI,
    /* ControlledPhaseShift = */ KernelType::PI,
    /* CNOT                 = */ KernelType::PI,
    /* CZ                   = */ KernelType::PI,
    /* SWAP                 = */ KernelType::PI,
    /* CRX                  = */ KernelType::PI,
    /* CRY                  = */ KernelType::PI,
    /* CRZ                  = */ KernelType::PI,
    /* CRot                 = */ KernelType::PI,
    /* Toffoli              = */ KernelType::PI,
    /* CSWAP                = */ KernelType::PI,
    /* Matrix               = */ KernelType::PI,
};


template<class fp_t>
using KernelFuncType = std::function<void(std::complex<fp_t>* /*data*/, size_t /*num_qubits*/,
                                          const std::vector<size_t>& /*wires*/, 
                                          bool /*inverse*/,
                                          const std::vector<fp_t>& /*params*/)>;

template<class fp_t, KernelType kernel_type>
class SelectGateOps {};

template<class fp_t>
class SelectGateOps<fp_t, KernelType::PI> : public GateOperationsPI<fp_t> {};
template<class fp_t>
class SelectGateOps<fp_t, KernelType::LM> : public GateOperationsLM<fp_t> {};

} // namespace Pennylane

template<>
struct std::hash<Pennylane::GateOperations> {
    size_t operator()(Pennylane::GateOperations gate_operation) {
        return std::hash<int>()(static_cast<int>(gate_operation));
    }
};

