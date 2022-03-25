// Copyright 2022 Xanadu Quantum Technologies Inc.

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
 * Defines default kernels for operations
 */
#pragma once

#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "KernelType.hpp"

namespace Pennylane {
/**
 *
 * @brief Define which kernel to use for each gate operation.
 *
 * @rst
 * Check
 * `this repository
 * <https://github.com/PennyLaneAI/pennylane-lightning-compare-kernels>`_ to see
 * the benchmark results for each gate
 * @endrst
 *
 * This value is used for:
 * 1. StateVector `apply##GATE_NAME` methods. The kernel function is statically
 * binded to the given kernel and cannot be modified.
 * 2. Default kernel functions for DynamicDispatcher. The kernel function is
 * dynamically binded and can be changed using DynamicDispatcher singleton
 * class.
 * 3. For the Python binding.
 */
[[maybe_unused]] constexpr std::array default_kernel_for_gates = {
    std::pair{Gates::GateOperation::PauliX, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::PauliY, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::PauliZ, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::Hadamard, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::S, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::T, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::RX, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::RY, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::RZ, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::PhaseShift, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::Rot, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::ControlledPhaseShift,
              Gates::KernelType::PI},
    std::pair{Gates::GateOperation::CNOT, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::CY, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::CZ, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::SWAP, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::IsingXX, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::IsingYY, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::IsingZZ, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::CRX, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::CRY, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::CRZ, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::CRot, Gates::KernelType::LM},
    std::pair{Gates::GateOperation::Toffoli, Gates::KernelType::PI},
    std::pair{Gates::GateOperation::CSWAP, Gates::KernelType::PI},
    std::pair{Gates::GateOperation::MultiRZ, Gates::KernelType::LM},
};
/**
 * @brief Define which kernel to use for each generator operation.
 */
[[maybe_unused]] constexpr std::array default_kernel_for_generators = {
    std::pair{Gates::GeneratorOperation::PhaseShift, Gates::KernelType::PI},
    std::pair{Gates::GeneratorOperation::RX, Gates::KernelType::LM},
    std::pair{Gates::GeneratorOperation::RY, Gates::KernelType::LM},
    std::pair{Gates::GeneratorOperation::RZ, Gates::KernelType::LM},
    std::pair{Gates::GeneratorOperation::IsingXX, Gates::KernelType::LM},
    std::pair{Gates::GeneratorOperation::IsingYY, Gates::KernelType::LM},
    std::pair{Gates::GeneratorOperation::IsingZZ, Gates::KernelType::LM},
    std::pair{Gates::GeneratorOperation::CRX, Gates::KernelType::LM},
    std::pair{Gates::GeneratorOperation::CRY, Gates::KernelType::LM},
    std::pair{Gates::GeneratorOperation::CRZ, Gates::KernelType::LM},
    std::pair{Gates::GeneratorOperation::ControlledPhaseShift,
              Gates::KernelType::LM},
    std::pair{Gates::GeneratorOperation::MultiRZ, Gates::KernelType::LM},
};

/**
 * @brief Define which kernel to use for each generator operation.
 */
[[maybe_unused]] constexpr std::array default_kernel_for_matrices = {
    std::pair{Gates::MatrixOperation::SingleQubitOp, Gates::KernelType::LM},
    std::pair{Gates::MatrixOperation::TwoQubitOp, Gates::KernelType::LM},
    std::pair{Gates::MatrixOperation::MultiQubitOp, Gates::KernelType::PI},
};

/**
 * @brief Return default kernel for gate operation
 *
 * @param gate_op Gate operation
 */
constexpr auto getDefaultKernelForGate(Gates::GateOperation gate_op)
    -> Gates::KernelType {
    return Util::lookup(default_kernel_for_gates, gate_op);
}

/**
 * @brief Return default kernel for generator operation
 *
 * @param gntr_op Generator operation
 */
constexpr auto getDefaultKernelForGenerator(Gates::GeneratorOperation gntr_op)
    -> Gates::KernelType {
    return Util::lookup(default_kernel_for_generators, gntr_op);
}

/**
 * @brief Return default kernel for matrix operation
 *
 * @param mat_op Matrix operation
 */
constexpr auto getDefaultKernelForMatrix(Gates::MatrixOperation mat_op)
    -> Gates::KernelType {
    return Util::lookup(default_kernel_for_matrices, mat_op);
}
} // namespace Pennylane
