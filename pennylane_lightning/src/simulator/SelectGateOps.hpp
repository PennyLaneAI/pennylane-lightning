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

#include "GateOperations.hpp"
#include "GateOperationsLM.hpp"
#include "GateOperationsPI.hpp"
#include "KernelType.hpp"

#include <array>
#include <functional>

namespace Pennylane {
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
/**
 * @brief For lookup from any array of pair whose first elements are GateOperations. 
 *
 * Note that Util::lookup can be used in constexpr context, thus this function is redundant 
 * (by the standard). But GCC 9 still does not accept Util::lookup in constexpr.
 */
template<GateOperations op, class T, size_t size>
constexpr auto 
static_lookup(const std::array<std::pair<GateOperations, T>, size>& arr) -> T {
    for (size_t idx = 0; idx < size; ++idx) {
        if (std::get<0>(arr[idx]) == op) {
            return std::get<1>(arr[idx]);
        }
    }
    return T{};
}

template <class fp_t, KernelType kernel>
class SelectGateOps {};

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
