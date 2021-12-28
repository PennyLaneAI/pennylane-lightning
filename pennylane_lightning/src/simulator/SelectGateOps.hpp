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

#include "GateOperationsLM.hpp"
#include "GateOperationsPI.hpp"
#include "KernelType.hpp"

#include <array>
#include <functional>

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

/**
 * This variable is only used in runtime. Thus constructing std::map in a runtim
 * is sufficient and do not need constexpr Map.
 */
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

/**
 * @brief Define which kernel to use for each operation
 *
 * This value is used for:
 *   1) StateVector apply##GATE_NAME methods. The kernel function is statically
 * binded to the given kernel and cannot be modified. 2) Default kernel
 * functions of StateVector applyOperation(opName, ...) methods. The kernel
 * function is dynamically binded and can be changed using DynamicDispatcher
 * singleton class. 3) Python binding.
 *
 * TODO: Change to constexpr
 * Map(https://www.youtube.com/watch?v=INn3xa4pMfg&list=WL&index=9) in C++20?
 */
constexpr std::array<std::pair<GateOperations, KernelType>,
                     static_cast<int>(GateOperations::END)>
    DEFAULT_KERNEL_FOR_OPS = {
        std::pair{GateOperations::PauliX, KernelType::LM},
        std::pair{GateOperations::PauliY, KernelType::LM},
        std::pair{GateOperations::PauliZ, KernelType::LM},
        std::pair{GateOperations::Hadamard, KernelType::LM},
        std::pair{GateOperations::S, KernelType::LM},
        std::pair{GateOperations::T, KernelType::LM},
        std::pair{GateOperations::RX, KernelType::LM},
        std::pair{GateOperations::RY, KernelType::LM},
        std::pair{GateOperations::RZ, KernelType::LM},
        std::pair{GateOperations::PhaseShift, KernelType::LM},
        std::pair{GateOperations::Rot, KernelType::PI},
        std::pair{GateOperations::ControlledPhaseShift, KernelType::PI},
        std::pair{GateOperations::CNOT, KernelType::PI},
        std::pair{GateOperations::CZ, KernelType::PI},
        std::pair{GateOperations::SWAP, KernelType::PI},
        std::pair{GateOperations::CRX, KernelType::PI},
        std::pair{GateOperations::CRY, KernelType::PI},
        std::pair{GateOperations::CRZ, KernelType::PI},
        std::pair{GateOperations::CRot, KernelType::PI},
        std::pair{GateOperations::Toffoli, KernelType::PI},
        std::pair{GateOperations::CSWAP, KernelType::PI},
        std::pair{GateOperations::Matrix, KernelType::PI},
};

template <GateOperations op, typename T, size_t size>
constexpr auto
static_lookup(const std::array<std::pair<GateOperations, T>, size> &arr) -> T {
    for (size_t idx = 0; idx < size; ++idx) {
        if (std::get<0>(arr[idx]) == op) {
            return std::get<1>(arr[idx]);
        }
    }
    static_assert("The given key (gate operation) does not exists.");
    return static_cast<KernelType>(0);
};

/**
 * @brief lookup value for key op
 *
 * This function is only for very small array
 */
template <typename T, size_t size>
auto dynamic_lookup(const std::array<std::pair<GateOperations, T>, size> &arr,
                    GateOperations op) -> T {
    for (const auto &[k, v] : arr) {
        if (k == op) {
            return v;
        }
    }
    PL_ABORT("Dynamic lookup failed");
    return T{};
}

template <class fp_t>
using KernelFuncType =
    std::function<void(std::complex<fp_t> * /*data*/, size_t /*num_qubits*/,
                       const std::vector<size_t> & /*wires*/, bool /*inverse*/,
                       const std::vector<fp_t> & /*params*/)>;

template <class fp_t, KernelType kernel> class SelectGateOps {};

template <class fp_t>
class SelectGateOps<fp_t, KernelType::PI> : public GateOperationsPI<fp_t> {};
template <class fp_t>
class SelectGateOps<fp_t, KernelType::LM> : public GateOperationsLM<fp_t> {};

} // namespace Pennylane

template <> struct std::hash<Pennylane::GateOperations> {
    size_t operator()(Pennylane::GateOperations gate_operation) {
        return std::hash<int>()(static_cast<int>(gate_operation));
    }
};
