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
 * @file GateOperation.hpp
 * Defines possible operations.
 */
#pragma once
#include <array>
#include <string>
#include <utility>

namespace Pennylane::Gates {
/**
 * @brief Enum class for all gate operations
 */
enum class GateOperation : uint32_t {
    BEGIN = 0,
    /* Single-qubit gates */
    Identity = 0,
    PauliX,
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
    IsingXX,
    IsingYY,
    IsingZZ,
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
    IsingXX,
    IsingYY,
    IsingZZ,
    CRX,
    CRY,
    CRZ,
    ControlledPhaseShift,
    MultiRZ,
    /* END (placeholder) */
    END
};
} // namespace Pennylane::Gates
