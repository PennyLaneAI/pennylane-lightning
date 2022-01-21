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
 * When you add a gate in this enum, please sure that gate_num_params is also
 * updated accordingly.
 * */
enum class GateOperations : int {
    BEGIN = 0,
    /* Single-qubit gates */
    PauliX = 0,
    PauliY,
    PauliZ,
    Hadamard,
    S,
    T,
    PhaseShift,
    RX,
    RY,
    RZ,
    Rot,
    /* Two-qubit gates */
    CNOT,
    CY,
    CZ,
    SWAP,
    ControlledPhaseShift,
    CRX,
    CRY,
    CRZ,
    CRot,
    /* Three-qubit gates */
    Toffoli,
    CSWAP,
    /* Gate generators (only used internally for adjoint diff) */
    GeneratorPhaseShift,
    GeneratorCRX,
    GeneratorCRY,
    GeneratorCRZ,
    GeneratorControlledPhaseShift,
    /* General matrix */
    Matrix,
    /* END (placeholder) */
    END
};
} // namespace Pennylane

namespace Pennylane::Constant {
constexpr std::array<std::pair<GateOperations, std::string_view>,
                     static_cast<int>(GateOperations::END)>
    gate_names = {
        std::pair{GateOperations::PauliX, "PauliX"},
        std::pair{GateOperations::PauliY, "PauliY"},
        std::pair{GateOperations::PauliZ, "PauliZ"},
        std::pair{GateOperations::Hadamard, "Hadamard"},
        std::pair{GateOperations::S, "S"},
        std::pair{GateOperations::T, "T"},
        std::pair{GateOperations::PhaseShift, "PhaseShift"},
        std::pair{GateOperations::RX, "RX"},
        std::pair{GateOperations::RY, "RY"},
        std::pair{GateOperations::RZ, "RZ"},
        std::pair{GateOperations::Rot, "Rot"},
        std::pair{GateOperations::CNOT, "CNOT"},
        std::pair{GateOperations::CY, "CY"},
        std::pair{GateOperations::CZ, "CZ"},
        std::pair{GateOperations::SWAP, "SWAP"},
        std::pair{GateOperations::ControlledPhaseShift, "ControlledPhaseShift"},
        std::pair{GateOperations::CRX, "CRX"},
        std::pair{GateOperations::CRY, "CRY"},
        std::pair{GateOperations::CRZ, "CRZ"},
        std::pair{GateOperations::CRot, "CRot"},
        std::pair{GateOperations::Toffoli, "Toffoli"},
        std::pair{GateOperations::CSWAP, "CSWAP"},
        std::pair{GateOperations::GeneratorPhaseShift, "GeneratrorPhaseShift"},
        std::pair{GateOperations::GeneratorCRX, "GeneratrorCRX"},
        std::pair{GateOperations::GeneratorCRY, "GeneratrorCRY"},
        std::pair{GateOperations::GeneratorCRZ, "GeneratrorCRZ"},
        std::pair{GateOperations::GeneratorControlledPhaseShift,
                  "GeneratrorControlledPhaseShift"},
        std::pair{GateOperations::Matrix, "Matrix"},
};

constexpr std::array<std::pair<GateOperations, std::size_t>,
                     static_cast<int>(GateOperations::END) - 1>
    gate_wires = {
        std::pair{GateOperations::PauliX, 1},
        std::pair{GateOperations::PauliY, 1},
        std::pair{GateOperations::PauliZ, 1},
        std::pair{GateOperations::Hadamard, 1},
        std::pair{GateOperations::S, 1},
        std::pair{GateOperations::T, 1},
        std::pair{GateOperations::PhaseShift, 1},
        std::pair{GateOperations::RX, 1},
        std::pair{GateOperations::RY, 1},
        std::pair{GateOperations::RZ, 1},
        std::pair{GateOperations::Rot, 1},
        std::pair{GateOperations::CNOT, 2},
        std::pair{GateOperations::CY, 2},
        std::pair{GateOperations::CZ, 2},
        std::pair{GateOperations::SWAP, 2},
        std::pair{GateOperations::ControlledPhaseShift, 2},
        std::pair{GateOperations::CRX, 2},
        std::pair{GateOperations::CRY, 2},
        std::pair{GateOperations::CRZ, 2},
        std::pair{GateOperations::CRot, 2},
        std::pair{GateOperations::Toffoli, 3},
        std::pair{GateOperations::CSWAP, 3},
        std::pair{GateOperations::GeneratorPhaseShift, 1},
        std::pair{GateOperations::GeneratorCRX, 2},
        std::pair{GateOperations::GeneratorCRY, 2},
        std::pair{GateOperations::GeneratorCRZ, 2},
        std::pair{GateOperations::GeneratorControlledPhaseShift, 2},
};

constexpr std::array<std::pair<GateOperations, size_t>,
                     static_cast<int>(GateOperations::END) - 1>
    gate_num_params = {
        std::pair{GateOperations::PauliX, 0},
        std::pair{GateOperations::PauliY, 0},
        std::pair{GateOperations::PauliZ, 0},
        std::pair{GateOperations::Hadamard, 0},
        std::pair{GateOperations::S, 0},
        std::pair{GateOperations::T, 0},
        std::pair{GateOperations::PhaseShift, 1},
        std::pair{GateOperations::RX, 1},
        std::pair{GateOperations::RY, 1},
        std::pair{GateOperations::RZ, 1},
        std::pair{GateOperations::Rot, 3},
        std::pair{GateOperations::CNOT, 0},
        std::pair{GateOperations::CY, 0},
        std::pair{GateOperations::CZ, 0},
        std::pair{GateOperations::SWAP, 0},
        std::pair{GateOperations::ControlledPhaseShift, 1},
        std::pair{GateOperations::CRX, 1},
        std::pair{GateOperations::CRY, 1},
        std::pair{GateOperations::CRZ, 1},
        std::pair{GateOperations::CRot, 3},
        std::pair{GateOperations::Toffoli, 0},
        std::pair{GateOperations::CSWAP, 0},
        std::pair{GateOperations::GeneratorPhaseShift, 0},
        std::pair{GateOperations::GeneratorCRX, 0},
        std::pair{GateOperations::GeneratorCRY, 0},
        std::pair{GateOperations::GeneratorCRZ, 0},
        std::pair{GateOperations::GeneratorControlledPhaseShift, 0},
};
} // namespace Pennylane::Constant
