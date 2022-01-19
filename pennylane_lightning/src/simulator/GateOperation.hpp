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
enum class GateOperation : uint32_t {
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
    /* General matrix */
    Matrix,
    /* END (placeholder) */
    END
};
enum class GeneratorOperation : uint32_t {
    BEGIN = 0,
    /* Gate generators (only used internally for adjoint diff) */
    PhaseShift = 0,
    CRX,
    CRY,
    CRZ,
    ControlledPhaseShift,
    /* END (placeholder) */
    END
};
} // namespace Pennylane

namespace Pennylane::Constant {

/**
 * @brief Gate names
 */
constexpr std::array<std::pair<GateOperation, std::string_view>,
                     static_cast<size_t>(GateOperation::END)>
    gate_names = {
        std::pair{GateOperation::PauliX, "PauliX"},
        std::pair{GateOperation::PauliY, "PauliY"},
        std::pair{GateOperation::PauliZ, "PauliZ"},
        std::pair{GateOperation::Hadamard, "Hadamard"},
        std::pair{GateOperation::S, "S"},
        std::pair{GateOperation::T, "T"},
        std::pair{GateOperation::PhaseShift, "PhaseShift"},
        std::pair{GateOperation::RX, "RX"},
        std::pair{GateOperation::RY, "RY"},
        std::pair{GateOperation::RZ, "RZ"},
        std::pair{GateOperation::Rot, "Rot"},
        std::pair{GateOperation::CNOT, "CNOT"},
        std::pair{GateOperation::CY, "CY"},
        std::pair{GateOperation::CZ, "CZ"},
        std::pair{GateOperation::SWAP, "SWAP"},
        std::pair{GateOperation::ControlledPhaseShift, "ControlledPhaseShift"},
        std::pair{GateOperation::CRX, "CRX"},
        std::pair{GateOperation::CRY, "CRY"},
        std::pair{GateOperation::CRZ, "CRZ"},
        std::pair{GateOperation::CRot, "CRot"},
        std::pair{GateOperation::Toffoli, "Toffoli"},
        std::pair{GateOperation::CSWAP, "CSWAP"},
        std::pair{GateOperation::Matrix, "Matrix"},
    };
/**
 * @brief Generator names
 */
constexpr std::array<std::pair<GeneratorOperation, std::string_view>,
          static_cast<size_t>(GeneratorOperation::END)>
    generator_names = {
        std::pair{GeneratorOperation::PhaseShift, "GeneratrorPhaseShift"},
        std::pair{GeneratorOperation::CRX, "GeneratrorCRX"},
        std::pair{GeneratorOperation::CRY, "GeneratrorCRY"},
        std::pair{GeneratorOperation::CRZ, "GeneratrorCRZ"},
        std::pair{GeneratorOperation::ControlledPhaseShift,
                  "GeneratrorControlledPhaseShift"},
    };

/**
 * @brief Number of wires for gates besides Matrix
 */
constexpr std::array<std::pair<GateOperation, std::size_t>,
                     static_cast<size_t>(GateOperation::END) - 1> // besides Matrix
    gate_wires = {
        std::pair{GateOperation::PauliX, 1},
        std::pair{GateOperation::PauliY, 1},
        std::pair{GateOperation::PauliZ, 1},
        std::pair{GateOperation::Hadamard, 1},
        std::pair{GateOperation::S, 1},
        std::pair{GateOperation::T, 1},
        std::pair{GateOperation::PhaseShift, 1},
        std::pair{GateOperation::RX, 1},
        std::pair{GateOperation::RY, 1},
        std::pair{GateOperation::RZ, 1},
        std::pair{GateOperation::Rot, 1},
        std::pair{GateOperation::CNOT, 2},
        std::pair{GateOperation::CY, 2},
        std::pair{GateOperation::CZ, 2},
        std::pair{GateOperation::SWAP, 2},
        std::pair{GateOperation::ControlledPhaseShift, 2},
        std::pair{GateOperation::CRX, 2},
        std::pair{GateOperation::CRY, 2},
        std::pair{GateOperation::CRZ, 2},
        std::pair{GateOperation::CRot, 2},
        std::pair{GateOperation::Toffoli, 3},
        std::pair{GateOperation::CSWAP, 3},
    };
constexpr std::array<std::pair<GeneratorOperation, std::size_t>, 
                     static_cast<size_t>(GeneratorOperation::END)>
    generator_wires = {
        std::pair{GeneratorOperation::PhaseShift, 1},
        std::pair{GeneratorOperation::CRX, 2},
        std::pair{GeneratorOperation::CRY, 2},
        std::pair{GeneratorOperation::CRZ, 2},
        std::pair{GeneratorOperation::ControlledPhaseShift, 2},
    };

/**
 * @brief Number of parameters for gates
 */
constexpr std::array<std::pair<GateOperation, size_t>,
                     static_cast<size_t>(GateOperation::END) - 1> // Besides Matrix
    gate_num_params = {
        std::pair{GateOperation::PauliX, 0},
        std::pair{GateOperation::PauliY, 0},
        std::pair{GateOperation::PauliZ, 0},
        std::pair{GateOperation::Hadamard, 0},
        std::pair{GateOperation::S, 0},
        std::pair{GateOperation::T, 0},
        std::pair{GateOperation::PhaseShift, 1},
        std::pair{GateOperation::RX, 1},
        std::pair{GateOperation::RY, 1},
        std::pair{GateOperation::RZ, 1},
        std::pair{GateOperation::Rot, 3},
        std::pair{GateOperation::CNOT, 0},
        std::pair{GateOperation::CY, 0},
        std::pair{GateOperation::CZ, 0},
        std::pair{GateOperation::SWAP, 0},
        std::pair{GateOperation::ControlledPhaseShift, 1},
        std::pair{GateOperation::CRX, 1},
        std::pair{GateOperation::CRY, 1},
        std::pair{GateOperation::CRZ, 1},
        std::pair{GateOperation::CRot, 3},
        std::pair{GateOperation::Toffoli, 0},
        std::pair{GateOperation::CSWAP, 0},
};
} // namespace Pennylane::Constant
