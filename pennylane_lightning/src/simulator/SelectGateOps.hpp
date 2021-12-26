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

namespace Pennylane {
/**
 * @brief enum class for all gate operations
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

enum class KernelType {PI, LM};


/**
 * @brief Define which kernel to use for each operation
 *
 * This value is used for:
 *   1) StateVector apply##GATE_NAME methods. The kernel function is statically binded to the 
 *   given kernel and cannot be modified.
 *   2) Default kernel functions of StateVector applyOperation(opName, ...) methods. The 
 *   kernel function is dynamically binded and can be changed using ApplyOperation singleton
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

template<class fp_t, KernelType kernel_type>
class SelectGateOps {};

template<class fp_t>
class SelectGateOps<fp_t, KernelType::PI> : public GateOperationsPI<fp_t> {};
template<class fp_t>
class SelectGateOps<fp_t, KernelType::LM> : public GateOperationsLM<fp_t> {};

} // namespace Pennylane
