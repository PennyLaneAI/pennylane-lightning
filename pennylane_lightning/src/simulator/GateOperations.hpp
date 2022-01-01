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
#include <array>
#include <string>
#include <utility>

namespace Pennylane {
/**
 * @brief enum class for all gate operations
 *
 * When you add a gate in this enum, please sure that GATE_NUM_PARAMS is also
 * updated accordingly.
 * */
enum class GateOperations : int {
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
constexpr std::array<std::pair<GateOperations, std::string_view>,
                     static_cast<int>(GateOperations::END)>
    GATE_NAMES = {
        std::pair{GateOperations::PauliX, "PauliX"},
        std::pair{GateOperations::PauliY, "PauliY"},
        std::pair{GateOperations::PauliZ, "PauliZ"},
        std::pair{GateOperations::Hadamard, "Hadamard"},
        std::pair{GateOperations::S, "S"},
        std::pair{GateOperations::T, "T"},
        std::pair{GateOperations::RX, "RX"},
        std::pair{GateOperations::RY, "RY"},
        std::pair{GateOperations::RZ, "RZ"},
        std::pair{GateOperations::PhaseShift, "PhaseShift"},
        std::pair{GateOperations::Rot, "Rot"},
        std::pair{GateOperations::ControlledPhaseShift, "ControlledPhaseShift"},
        std::pair{GateOperations::CNOT, "CNOT"},
        std::pair{GateOperations::CZ, "CZ"},
        std::pair{GateOperations::SWAP, "SWAP"},
        std::pair{GateOperations::CRX, "CRX"},
        std::pair{GateOperations::CRY, "CRY"},
        std::pair{GateOperations::CRZ, "CRZ"},
        std::pair{GateOperations::CRot, "CRot"},
        std::pair{GateOperations::Toffoli, "Toffoli"},
        std::pair{GateOperations::CSWAP, "CSWAP"},
        std::pair{GateOperations::Matrix, "Matrix"}};
} // namespace Pennylane
