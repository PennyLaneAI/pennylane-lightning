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
 * @brief Enum class for all gate operations
 */
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
    /* Mutli-qubit gates */
    MultiRZ,
    /* General matrix */
    Matrix,
    /* END (placeholder) */
    END
};
/**
 * @brief Enum class of all gate generators
 */
enum class GeneratorOperation : uint32_t {
    BEGIN = 0,
    /* Gate generators (only used internally for adjoint diff) */
    PhaseShift = 0,
    RX,
    RY,
    RZ,
    CRX,
    CRY,
    CRZ,
    ControlledPhaseShift,
    MultiRZ,
    /* END (placeholder) */
    END
};
} // namespace Pennylane

namespace Pennylane::Constant {
/**
 * @brief List of multi-qubit gates
 */
constexpr std::array multi_qubit_gates {
    GateOperation::MultiRZ,
    GateOperation::Matrix
};
/**
 * @brief List of multi-qubit generators
 */
constexpr std::array multi_qubit_generators {
    GeneratorOperation::MultiRZ,
};

/**
 * @brief Gate names
 */
constexpr std::array gate_names = {
    std::pair<GateOperation, std::string_view>{GateOperation::PauliX, "PauliX"},
    std::pair<GateOperation, std::string_view>{GateOperation::PauliY, "PauliY"},
    std::pair<GateOperation, std::string_view>{GateOperation::PauliZ, "PauliZ"},
    std::pair<GateOperation, std::string_view>{GateOperation::Hadamard, "Hadamard"},
    std::pair<GateOperation, std::string_view>{GateOperation::S, "S"},
    std::pair<GateOperation, std::string_view>{GateOperation::T, "T"},
    std::pair<GateOperation, std::string_view>{GateOperation::PhaseShift, "PhaseShift"},
    std::pair<GateOperation, std::string_view>{GateOperation::RX, "RX"},
    std::pair<GateOperation, std::string_view>{GateOperation::RY, "RY"},
    std::pair<GateOperation, std::string_view>{GateOperation::RZ, "RZ"},
    std::pair<GateOperation, std::string_view>{GateOperation::Rot, "Rot"},
    std::pair<GateOperation, std::string_view>{GateOperation::CNOT, "CNOT"},
    std::pair<GateOperation, std::string_view>{GateOperation::CY, "CY"},
    std::pair<GateOperation, std::string_view>{GateOperation::CZ, "CZ"},
    std::pair<GateOperation, std::string_view>{GateOperation::SWAP, "SWAP"},
    std::pair<GateOperation, std::string_view>{GateOperation::ControlledPhaseShift, "ControlledPhaseShift"},
    std::pair<GateOperation, std::string_view>{GateOperation::CRX, "CRX"},
    std::pair<GateOperation, std::string_view>{GateOperation::CRY, "CRY"},
    std::pair<GateOperation, std::string_view>{GateOperation::CRZ, "CRZ"},
    std::pair<GateOperation, std::string_view>{GateOperation::CRot, "CRot"},
    std::pair<GateOperation, std::string_view>{GateOperation::Toffoli, "Toffoli"},
    std::pair<GateOperation, std::string_view>{GateOperation::CSWAP, "CSWAP"},
    std::pair<GateOperation, std::string_view>{GateOperation::MultiRZ, "MultiRZ"},
    std::pair<GateOperation, std::string_view>{GateOperation::Matrix, "Matrix"},
};
/**
 * @brief Generator names
 */
constexpr std::array generator_names = {
    std::pair<GeneratorOperation, std::string_view>
            {GeneratorOperation::PhaseShift, "GeneratorPhaseShift"},
    std::pair<GeneratorOperation, std::string_view>
            {GeneratorOperation::RX, "GeneratorRX"},
    std::pair<GeneratorOperation, std::string_view>
            {GeneratorOperation::RY, "GeneratorRY"},
    std::pair<GeneratorOperation, std::string_view>
            {GeneratorOperation::RZ, "GeneratorRZ"},
    std::pair<GeneratorOperation, std::string_view>
            {GeneratorOperation::CRX, "GeneratorCRX"},
    std::pair<GeneratorOperation, std::string_view>
            {GeneratorOperation::CRY, "GeneratorCRY"},
    std::pair<GeneratorOperation, std::string_view>
            {GeneratorOperation::CRZ, "GeneratorCRZ"},
    std::pair<GeneratorOperation, std::string_view>
            {GeneratorOperation::ControlledPhaseShift,"GeneratorControlledPhaseShift"},
    std::pair<GeneratorOperation, std::string_view>
            {GeneratorOperation::MultiRZ, "GeneratorMultiRZ"},
};

/**
 * @brief Number of wires for gates besides Matrix
 */
constexpr std::array gate_wires = {
    std::pair<GateOperation, size_t>{GateOperation::PauliX, 1},
    std::pair<GateOperation, size_t>{GateOperation::PauliY, 1},
    std::pair<GateOperation, size_t>{GateOperation::PauliZ, 1},
    std::pair<GateOperation, size_t>{GateOperation::Hadamard, 1},
    std::pair<GateOperation, size_t>{GateOperation::S, 1},
    std::pair<GateOperation, size_t>{GateOperation::T, 1},
    std::pair<GateOperation, size_t>{GateOperation::PhaseShift, 1},
    std::pair<GateOperation, size_t>{GateOperation::RX, 1},
    std::pair<GateOperation, size_t>{GateOperation::RY, 1},
    std::pair<GateOperation, size_t>{GateOperation::RZ, 1},
    std::pair<GateOperation, size_t>{GateOperation::Rot, 1},
    std::pair<GateOperation, size_t>{GateOperation::CNOT, 2},
    std::pair<GateOperation, size_t>{GateOperation::CY, 2},
    std::pair<GateOperation, size_t>{GateOperation::CZ, 2},
    std::pair<GateOperation, size_t>{GateOperation::SWAP, 2},
    std::pair<GateOperation, size_t>{GateOperation::ControlledPhaseShift, 2},
    std::pair<GateOperation, size_t>{GateOperation::CRX, 2},
    std::pair<GateOperation, size_t>{GateOperation::CRY, 2},
    std::pair<GateOperation, size_t>{GateOperation::CRZ, 2},
    std::pair<GateOperation, size_t>{GateOperation::CRot, 2},
    std::pair<GateOperation, size_t>{GateOperation::Toffoli, 3},
    std::pair<GateOperation, size_t>{GateOperation::CSWAP, 3},
};

constexpr std::array generator_wires = {
        std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::PhaseShift, 1},
        std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::RX, 1},
        std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::RY, 1},
        std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::RZ, 1},
        std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::CRX, 2},
        std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::CRY, 2},
        std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::CRZ, 2},
        std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::ControlledPhaseShift, 2},
    };

/**
 * @brief Number of parameters for gates
 */
constexpr std::array gate_num_params = {
        std::pair<GateOperation, size_t>{GateOperation::PauliX, 0},
        std::pair<GateOperation, size_t>{GateOperation::PauliY, 0},
        std::pair<GateOperation, size_t>{GateOperation::PauliZ, 0},
        std::pair<GateOperation, size_t>{GateOperation::Hadamard, 0},
        std::pair<GateOperation, size_t>{GateOperation::S, 0},
        std::pair<GateOperation, size_t>{GateOperation::T, 0},
        std::pair<GateOperation, size_t>{GateOperation::PhaseShift, 1},
        std::pair<GateOperation, size_t>{GateOperation::RX, 1},
        std::pair<GateOperation, size_t>{GateOperation::RY, 1},
        std::pair<GateOperation, size_t>{GateOperation::RZ, 1},
        std::pair<GateOperation, size_t>{GateOperation::Rot, 3},
        std::pair<GateOperation, size_t>{GateOperation::CNOT, 0},
        std::pair<GateOperation, size_t>{GateOperation::CY, 0},
        std::pair<GateOperation, size_t>{GateOperation::CZ, 0},
        std::pair<GateOperation, size_t>{GateOperation::SWAP, 0},
        std::pair<GateOperation, size_t>{GateOperation::ControlledPhaseShift, 1},
        std::pair<GateOperation, size_t>{GateOperation::CRX, 1},
        std::pair<GateOperation, size_t>{GateOperation::CRY, 1},
        std::pair<GateOperation, size_t>{GateOperation::CRZ, 1},
        std::pair<GateOperation, size_t>{GateOperation::CRot, 3},
        std::pair<GateOperation, size_t>{GateOperation::Toffoli, 0},
        std::pair<GateOperation, size_t>{GateOperation::CSWAP, 0},
};
} // namespace Pennylane::Constant
