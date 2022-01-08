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
#include "Macros.hpp"

#include <array>
#include <functional>
#include <variant>

namespace Pennylane {

namespace Constant {
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
    default_kernel_for_ops = {
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
        std::pair{GateOperations::Rot, KernelType::LM},
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
        std::pair{GateOperations::GeneratorPhaseShift, KernelType::PI},
        std::pair{GateOperations::GeneratorCRX, KernelType::PI},
        std::pair{GateOperations::GeneratorCRY, KernelType::PI},
        std::pair{GateOperations::GeneratorCRZ, KernelType::PI},
        std::pair{GateOperations::GeneratorControlledPhaseShift,
                  KernelType::PI},
};
} // namespace Constant

/**
 * @brief For lookup from any array of pair whose first elements are
 * GateOperations.
 *
 * As Util::lookup can be used in constexpr context, this function is redundant
 * (by the standard). But GCC 9 still does not accept Util::lookup in constexpr
 * some cases.
 */
template <GateOperations op, class T, size_t size>
constexpr auto
static_lookup(const std::array<std::pair<GateOperations, T>, size> &arr) -> T {
    for (size_t idx = 0; idx < size; ++idx) {
        if (std::get<0>(arr[idx]) == op) {
            return std::get<1>(arr[idx]);
        }
    }
    return T{};
}

/**
 * @brief This class is for choosing a gate implementation at the compile time.
 *
 * When someone wants to add another gate implementation, one needs to add a key
 * in KernelType and assign it to SelectGateOps by template specialization.
 */
template <class fp_t, KernelType kernel> class SelectGateOps {};

template <class fp_t>
class SelectGateOps<fp_t, KernelType::PI> : public GateOperationsPI<fp_t> {};
template <class fp_t>
class SelectGateOps<fp_t, KernelType::LM> : public GateOperationsLM<fp_t> {};

} // namespace Pennylane

namespace Pennylane::Internal {

template <class PrecisionT, class ParamT, KernelType kernel, size_t num_params>
struct GateFuncPtrPairs {
    static_assert(num_params < 2 || num_params == 3,
                  "The given num_params is not supported.");
};

/**
 * @brief List of all gate operation and funciont pointer pairs without
 * parameters
 */
template <class PrecisionT, class ParamT, KernelType kernel>
struct GateFuncPtrPairs<PrecisionT, ParamT, kernel, 0> {
    constexpr static std::array value = {
        std::pair{GateOperations::PauliX,
                  &SelectGateOps<PrecisionT, kernel>::applyPauliX},
        std::pair{GateOperations::PauliY,
                  &SelectGateOps<PrecisionT, kernel>::applyPauliY},
        std::pair{GateOperations::PauliZ,
                  &SelectGateOps<PrecisionT, kernel>::applyPauliZ},
        std::pair{GateOperations::Hadamard,
                  &SelectGateOps<PrecisionT, kernel>::applyHadamard},
        std::pair{GateOperations::S,
                  &SelectGateOps<PrecisionT, kernel>::applyS},
        std::pair{GateOperations::T,
                  &SelectGateOps<PrecisionT, kernel>::applyT},
        std::pair{GateOperations::CNOT,
                  &SelectGateOps<PrecisionT, kernel>::applyCNOT},
        std::pair{GateOperations::CZ,
                  &SelectGateOps<PrecisionT, kernel>::applyCZ},
        std::pair{GateOperations::SWAP,
                  &SelectGateOps<PrecisionT, kernel>::applySWAP},
        std::pair{GateOperations::Toffoli,
                  &SelectGateOps<PrecisionT, kernel>::applyToffoli},
        std::pair{GateOperations::CSWAP,
                  &SelectGateOps<PrecisionT, kernel>::applyCSWAP},
        std::pair{GateOperations::GeneratorPhaseShift,
                  &SelectGateOps<PrecisionT, kernel>::applyGeneratorPhaseShift},
        std::pair{GateOperations::GeneratorCRX,
                  &SelectGateOps<PrecisionT, kernel>::applyGeneratorCRX},
        std::pair{GateOperations::GeneratorCRY,
                  &SelectGateOps<PrecisionT, kernel>::applyGeneratorCRY},
        std::pair{GateOperations::GeneratorCRZ,
                  &SelectGateOps<PrecisionT, kernel>::applyGeneratorCRZ},
        std::pair{GateOperations::GeneratorControlledPhaseShift,
                  &SelectGateOps<PrecisionT,
                                 kernel>::applyGeneratorControlledPhaseShift}};
};

/**
 * @brief List of all gate operation and funciont pointer pairs with a single
 * paramter
 */
template <class PrecisionT, class ParamT, KernelType kernel>
struct GateFuncPtrPairs<PrecisionT, ParamT, kernel, 1> {
    constexpr static std::array value = {
        std::pair{GateOperations::RX,
                  &SelectGateOps<PrecisionT, kernel>::template applyRX<ParamT>},
        std::pair{GateOperations::RY,
                  &SelectGateOps<PrecisionT, kernel>::template applyRY<ParamT>},
        std::pair{GateOperations::RZ,
                  &SelectGateOps<PrecisionT, kernel>::template applyRZ<ParamT>},
        std::pair{GateOperations::PhaseShift,
                  &SelectGateOps<PrecisionT,
                                 kernel>::template applyPhaseShift<ParamT>},
        std::pair{
            GateOperations::CRX,
            &SelectGateOps<PrecisionT, kernel>::template applyCRX<ParamT>},
        std::pair{
            GateOperations::CRY,
            &SelectGateOps<PrecisionT, kernel>::template applyCRY<ParamT>},
        std::pair{
            GateOperations::CRZ,
            &SelectGateOps<PrecisionT, kernel>::template applyCRZ<ParamT>},
        std::pair{GateOperations::ControlledPhaseShift,
                  &SelectGateOps<PrecisionT, kernel>::
                      template applyControlledPhaseShift<ParamT>}};
};

/**
 * @brief List of all gate operation and funciont pointer pairs with three
 * paramters
 */
template <class PrecisionT, class ParamT, KernelType kernel>
struct GateFuncPtrPairs<PrecisionT, ParamT, kernel, 3> {
    constexpr static std::array value = {
        std::pair{
            GateOperations::Rot,
            &SelectGateOps<PrecisionT, kernel>::template applyRot<ParamT>},
        std::pair{
            GateOperations::CRot,
            &SelectGateOps<PrecisionT, kernel>::template applyCRot<ParamT>}};
};

template <class fp_t, class ParamT, int num_params> struct CallKernelFunc {
    static_assert(num_params < 2 || num_params == 3,
                  "Unsupported number of parameters.");
};

template <class fp_t, class ParamT> struct CallKernelFunc<fp_t, ParamT, 0> {
    template <class Func>
    inline static void call(Func &&func, std::complex<fp_t> *data,
                            size_t num_qubits, const std::vector<size_t> &wires,
                            bool inverse,
                            [[maybe_unused]] const std::vector<fp_t> &params) {
        assert(params.empty());
        func(data, num_qubits, wires, inverse);
    }
};

template <class fp_t, class ParamT> struct CallKernelFunc<fp_t, ParamT, 1> {
    template <class Func>
    inline static void call(Func &&func, std::complex<fp_t> *data,
                            size_t num_qubits, const std::vector<size_t> &wires,
                            bool inverse, const std::vector<fp_t> &params) {
        assert(params.size() == 1);
        func(data, num_qubits, wires, inverse, params[0]);
    }
};

template <class fp_t, class ParamT> struct CallKernelFunc<fp_t, ParamT, 3> {
    template <class Func>
    inline static void call(Func &&func, std::complex<fp_t> *data,
                            size_t num_qubits, const std::vector<size_t> &wires,
                            bool inverse, const std::vector<fp_t> &params) {
        assert(params.size() == 3);
        func(data, num_qubits, wires, inverse, params[0], params[1], params[2]);
    }
};

template <typename fp_t, size_t idx>
std::vector<GateOperations> implementedGatesForKernelIter(KernelType kernel) {
    if constexpr (idx == Constant::available_kernels.size()) {
        return {};
    } else if (kernel == std::get<0>(Constant::available_kernels[idx])) {
        const auto &arr =
            SelectGateOps<fp_t, std::get<0>(Constant::available_kernels[idx])>::
                implemented_gates;
        return std::vector(arr.begin(), arr.end());
    } else {
        return implementedGatesForKernelIter<fp_t, idx + 1>(kernel);
    }
}

/**
 * @brief return implemented_gates constexpr member variables for a given kernel
 *
 * This function interfaces the runtime variable kernel with the constant time
 * variable implemented_gates
 *
 * TODO: Change to constexpr function in C++20
 */
template <class fp_t>
std::vector<GateOperations> implementedGatesForKernel(KernelType kernel) {
    return Internal::implementedGatesForKernelIter<fp_t, 0>(kernel);
}

/********************************************************************
 * Functions below are only used in a compile time to check
 * consistency.
 ********************************************************************/
template <typename fp_t>
constexpr auto check_default_kernels_are_available() -> bool {
    // TODO: change to constexpr std::all_of in C++20
    // which is not constexpr in C++17.
    // NOLINTNEXTLINE (readability-use-anyofallof)
    for (const auto &[gate_op, kernel] : Constant::default_kernel_for_ops) {
        if (!is_available_kernel(kernel)) {
            return false;
        }
    }
    return true;
}

static_assert(check_default_kernels_are_available<double>(),
              "default_kernel_for_ops contains an unavailable kernel");
} // namespace Pennylane::Internal

template <> struct std::hash<Pennylane::GateOperations> {
    size_t operator()(Pennylane::GateOperations gate_operation) const {
        return std::hash<int>()(static_cast<int>(gate_operation));
    }
};
