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
 * @file Constant.hpp
 * Defines all constants for statevector
 */
#pragma once

#include "GateOperation.hpp"
#include "KernelType.hpp"
#include "TypeList.hpp"

namespace Pennylane::Gates::Constant {
/**
 * @brief List of multi-qubit gates
 */
[[maybe_unused]] constexpr std::array multi_qubit_gates{GateOperation::MultiRZ};
/**
 * @brief List of multi-qubit generators
 */
[[maybe_unused]] constexpr std::array multi_qubit_generators{
    GeneratorOperation::MultiRZ,
};
/**
 * @brief List of multi-qubit matrix operation
 */
[[maybe_unused]] constexpr std::array multi_qubit_matrix_ops{
    MatrixOperation::MultiQubitOp,
};

/**
 * @brief Gate names
 */
[[maybe_unused]] constexpr std::array gate_names = {
    std::pair<GateOperation, std::string_view>{GateOperation::Identity,
                                               "Identity"},
    std::pair<GateOperation, std::string_view>{GateOperation::PauliX, "PauliX"},
    std::pair<GateOperation, std::string_view>{GateOperation::PauliY, "PauliY"},
    std::pair<GateOperation, std::string_view>{GateOperation::PauliZ, "PauliZ"},
    std::pair<GateOperation, std::string_view>{GateOperation::Hadamard,
                                               "Hadamard"},
    std::pair<GateOperation, std::string_view>{GateOperation::S, "S"},
    std::pair<GateOperation, std::string_view>{GateOperation::T, "T"},
    std::pair<GateOperation, std::string_view>{GateOperation::PhaseShift,
                                               "PhaseShift"},
    std::pair<GateOperation, std::string_view>{GateOperation::RX, "RX"},
    std::pair<GateOperation, std::string_view>{GateOperation::RY, "RY"},
    std::pair<GateOperation, std::string_view>{GateOperation::RZ, "RZ"},
    std::pair<GateOperation, std::string_view>{GateOperation::Rot, "Rot"},
    std::pair<GateOperation, std::string_view>{GateOperation::CNOT, "CNOT"},
    std::pair<GateOperation, std::string_view>{GateOperation::CY, "CY"},
    std::pair<GateOperation, std::string_view>{GateOperation::CZ, "CZ"},
    std::pair<GateOperation, std::string_view>{GateOperation::IsingXX,
                                               "IsingXX"},
    std::pair<GateOperation, std::string_view>{GateOperation::IsingXY,
                                               "IsingXY"},
    std::pair<GateOperation, std::string_view>{GateOperation::IsingYY,
                                               "IsingYY"},
    std::pair<GateOperation, std::string_view>{GateOperation::IsingZZ,
                                               "IsingZZ"},
    std::pair<GateOperation, std::string_view>{GateOperation::SWAP, "SWAP"},
    std::pair<GateOperation, std::string_view>{
        GateOperation::ControlledPhaseShift, "ControlledPhaseShift"},
    std::pair<GateOperation, std::string_view>{GateOperation::CRX, "CRX"},
    std::pair<GateOperation, std::string_view>{GateOperation::CRY, "CRY"},
    std::pair<GateOperation, std::string_view>{GateOperation::CRZ, "CRZ"},
    std::pair<GateOperation, std::string_view>{GateOperation::CRot, "CRot"},
    std::pair<GateOperation, std::string_view>{GateOperation::SingleExcitation,
                                               "SingleExcitation"},
    std::pair<GateOperation, std::string_view>{
        GateOperation::SingleExcitationMinus, "SingleExcitationMinus"},
    std::pair<GateOperation, std::string_view>{
        GateOperation::SingleExcitationPlus, "SingleExcitationPlus"},
    std::pair<GateOperation, std::string_view>{GateOperation::Toffoli,
                                               "Toffoli"},
    std::pair<GateOperation, std::string_view>{GateOperation::CSWAP, "CSWAP"},
    std::pair<GateOperation, std::string_view>{GateOperation::DoubleExcitation,
                                               "DoubleExcitation"},
    std::pair<GateOperation, std::string_view>{
        GateOperation::DoubleExcitationMinus, "DoubleExcitationMinus"},
    std::pair<GateOperation, std::string_view>{
        GateOperation::DoubleExcitationPlus, "DoubleExcitationPlus"},
    std::pair<GateOperation, std::string_view>{GateOperation::MultiRZ,
                                               "MultiRZ"}};
/**
 * @brief Generator names.
 *
 * Note that a name of generators must be "Generator" +
 * the name of the corresponding gate.
 */
[[maybe_unused]] constexpr std::array generator_names = {
    std::pair<GeneratorOperation, std::string_view>{
        GeneratorOperation::PhaseShift, "GeneratorPhaseShift"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::RX,
                                                    "GeneratorRX"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::RY,
                                                    "GeneratorRY"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::RZ,
                                                    "GeneratorRZ"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::CRX,
                                                    "GeneratorCRX"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::CRY,
                                                    "GeneratorCRY"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::CRZ,
                                                    "GeneratorCRZ"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::IsingXX,
                                                    "GeneratorIsingXX"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::IsingXY,
                                                    "GeneratorIsingXY"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::IsingYY,
                                                    "GeneratorIsingYY"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::IsingZZ,
                                                    "GeneratorIsingZZ"},
    std::pair<GeneratorOperation, std::string_view>{
        GeneratorOperation::ControlledPhaseShift,
        "GeneratorControlledPhaseShift"},
    std::pair<GeneratorOperation, std::string_view>{
        GeneratorOperation::SingleExcitation, "GeneratorSingleExcitation"},
    std::pair<GeneratorOperation, std::string_view>{
        GeneratorOperation::SingleExcitationMinus,
        "GeneratorSingleExcitationMinus"},
    std::pair<GeneratorOperation, std::string_view>{
        GeneratorOperation::SingleExcitationPlus,
        "GeneratorSingleExcitationPlus"},
    std::pair<GeneratorOperation, std::string_view>{GeneratorOperation::MultiRZ,
                                                    "GeneratorMultiRZ"},
    std::pair<GeneratorOperation, std::string_view>{
        GeneratorOperation::DoubleExcitation, "GeneratorDoubleExcitation"},
    std::pair<GeneratorOperation, std::string_view>{
        GeneratorOperation::DoubleExcitationMinus,
        "GeneratorDoubleExcitationMinus"},
    std::pair<GeneratorOperation, std::string_view>{
        GeneratorOperation::DoubleExcitationPlus,
        "GeneratorDoubleExcitationPlus"},
};
/**
 * @brief Matrix names.
 */
[[maybe_unused]] constexpr std::array matrix_names = {
    std::pair<MatrixOperation, std::string_view>{MatrixOperation::SingleQubitOp,
                                                 "SingleQubitOp"},
    std::pair<MatrixOperation, std::string_view>{MatrixOperation::TwoQubitOp,
                                                 "TwoQubitOp"},
    std::pair<MatrixOperation, std::string_view>{MatrixOperation::MultiQubitOp,
                                                 "MultiQubitOp"},
};

/**
 * @brief Number of wires for gates besides multi-qubit gates
 */
[[maybe_unused]] constexpr std::array gate_wires = {
    std::pair<GateOperation, size_t>{GateOperation::Identity, 1},
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
    std::pair<GateOperation, size_t>{GateOperation::IsingXX, 2},
    std::pair<GateOperation, size_t>{GateOperation::IsingXY, 2},
    std::pair<GateOperation, size_t>{GateOperation::IsingYY, 2},
    std::pair<GateOperation, size_t>{GateOperation::IsingZZ, 2},
    std::pair<GateOperation, size_t>{GateOperation::ControlledPhaseShift, 2},
    std::pair<GateOperation, size_t>{GateOperation::CRX, 2},
    std::pair<GateOperation, size_t>{GateOperation::CRY, 2},
    std::pair<GateOperation, size_t>{GateOperation::CRZ, 2},
    std::pair<GateOperation, size_t>{GateOperation::CRot, 2},
    std::pair<GateOperation, size_t>{GateOperation::SingleExcitation, 2},
    std::pair<GateOperation, size_t>{GateOperation::SingleExcitationMinus, 2},
    std::pair<GateOperation, size_t>{GateOperation::SingleExcitationPlus, 2},
    std::pair<GateOperation, size_t>{GateOperation::Toffoli, 3},
    std::pair<GateOperation, size_t>{GateOperation::CSWAP, 3},
    std::pair<GateOperation, size_t>{GateOperation::DoubleExcitation, 4},
    std::pair<GateOperation, size_t>{GateOperation::DoubleExcitationMinus, 4},
    std::pair<GateOperation, size_t>{GateOperation::DoubleExcitationPlus, 4},
};

/**
 * @brief Number of wires for generators besides multi-qubit gates
 */
[[maybe_unused]] constexpr std::array generator_wires = {
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::PhaseShift,
                                               1},
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::RX, 1},
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::RY, 1},
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::RZ, 1},
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::IsingXX, 2},
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::IsingXY, 2},
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::IsingYY, 2},
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::IsingZZ, 2},
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::CRX, 2},
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::CRY, 2},
    std::pair<GeneratorOperation, std::size_t>{GeneratorOperation::CRZ, 2},
    std::pair<GeneratorOperation, std::size_t>{
        GeneratorOperation::SingleExcitation, 2},
    std::pair<GeneratorOperation, std::size_t>{
        GeneratorOperation::SingleExcitationMinus, 2},
    std::pair<GeneratorOperation, std::size_t>{
        GeneratorOperation::SingleExcitationPlus, 2},
    std::pair<GeneratorOperation, std::size_t>{
        GeneratorOperation::ControlledPhaseShift, 2},
    std::pair<GeneratorOperation, std::size_t>{
        GeneratorOperation::DoubleExcitation, 4},
    std::pair<GeneratorOperation, std::size_t>{
        GeneratorOperation::DoubleExcitationMinus, 4},
    std::pair<GeneratorOperation, std::size_t>{
        GeneratorOperation::DoubleExcitationPlus, 4},
};

/**
 * @brief Number of parameters for gates
 */
[[maybe_unused]] constexpr std::array gate_num_params = {
    std::pair<GateOperation, size_t>{GateOperation::Identity, 0},
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
    std::pair<GateOperation, size_t>{GateOperation::IsingXX, 1},
    std::pair<GateOperation, size_t>{GateOperation::IsingXY, 1},
    std::pair<GateOperation, size_t>{GateOperation::IsingYY, 1},
    std::pair<GateOperation, size_t>{GateOperation::IsingZZ, 1},
    std::pair<GateOperation, size_t>{GateOperation::ControlledPhaseShift, 1},
    std::pair<GateOperation, size_t>{GateOperation::CRX, 1},
    std::pair<GateOperation, size_t>{GateOperation::CRY, 1},
    std::pair<GateOperation, size_t>{GateOperation::CRZ, 1},
    std::pair<GateOperation, size_t>{GateOperation::SingleExcitation, 1},
    std::pair<GateOperation, size_t>{GateOperation::SingleExcitationMinus, 1},
    std::pair<GateOperation, size_t>{GateOperation::SingleExcitationPlus, 1},
    std::pair<GateOperation, size_t>{GateOperation::CRot, 3},
    std::pair<GateOperation, size_t>{GateOperation::Toffoli, 0},
    std::pair<GateOperation, size_t>{GateOperation::DoubleExcitation, 1},
    std::pair<GateOperation, size_t>{GateOperation::DoubleExcitationMinus, 1},
    std::pair<GateOperation, size_t>{GateOperation::DoubleExcitationPlus, 1},
    std::pair<GateOperation, size_t>{GateOperation::CSWAP, 0},
    std::pair<GateOperation, size_t>{GateOperation::MultiRZ, 1},
};
} // namespace Pennylane::Gates::Constant
