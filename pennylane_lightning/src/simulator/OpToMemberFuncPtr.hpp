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
 * @file OperationToMemberFuncPtr.hpp
 */

#pragma once
#include "GateOperation.hpp"

#include <complex>
#include <vector>

namespace Pennylane {

/**
 * @brief Return a specific member function pointer for a given gate operation.
 * See speicalized classes.
 */
template <class PrecisionT, class ParamT, class GateImplOrSVType,
          GateOperation gate_op>
struct GateOpToMemberFuncPtr {
    static_assert(
        gate_op != GateOperation::Matrix,
        "GateOpToMemberFuncPtr is not defined for GateOperation::Matrix.");
    constexpr static auto value = nullptr;
};

template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::PauliX> {
    constexpr static auto value =
        &GateImplOrSVType::template applyPauliX<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::PauliY> {
    constexpr static auto value =
        &GateImplOrSVType::template applyPauliY<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::PauliZ> {
    constexpr static auto value =
        &GateImplOrSVType::template applyPauliZ<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::Hadamard> {
    constexpr static auto value =
        &GateImplOrSVType::template applyHadamard<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::S> {
    constexpr static auto value =
        &GateImplOrSVType::template applyS<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::T> {
    constexpr static auto value =
        &GateImplOrSVType::template applyT<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::PhaseShift> {
    constexpr static auto value =
        &GateImplOrSVType::template applyPhaseShift<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::RX> {
    constexpr static auto value =
        &GateImplOrSVType::template applyRX<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::RY> {
    constexpr static auto value =
        &GateImplOrSVType::template applyRY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::RZ> {
    constexpr static auto value =
        &GateImplOrSVType::template applyRZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::Rot> {
    constexpr static auto value =
        &GateImplOrSVType::template applyRot<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::CNOT> {
    constexpr static auto value =
        &GateImplOrSVType::template applyCNOT<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::CY> {
    constexpr static auto value =
        &GateImplOrSVType::template applyCY<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::CZ> {
    constexpr static auto value =
        &GateImplOrSVType::template applyCZ<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::SWAP> {
    constexpr static auto value =
        &GateImplOrSVType::template applySWAP<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::IsingXX> {
    constexpr static auto value =
        &GateImplOrSVType::template applyIsingXX<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::IsingYY> {
    constexpr static auto value =
        &GateImplOrSVType::template applyIsingYY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::IsingZZ> {
    constexpr static auto value =
        &GateImplOrSVType::template applyIsingZZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::ControlledPhaseShift> {
    constexpr static auto value =
        &GateImplOrSVType::template applyControlledPhaseShift<PrecisionT,
                                                              ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::CRX> {
    constexpr static auto value =
        &GateImplOrSVType::template applyCRX<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::CRY> {
    constexpr static auto value =
        &GateImplOrSVType::template applyCRY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::CRZ> {
    constexpr static auto value =
        &GateImplOrSVType::template applyCRZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::CRot> {
    constexpr static auto value =
        &GateImplOrSVType::template applyCRot<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::Toffoli> {
    constexpr static auto value =
        &GateImplOrSVType::template applyToffoli<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::CSWAP> {
    constexpr static auto value =
        &GateImplOrSVType::template applyCSWAP<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplOrSVType,
                             GateOperation::MultiRZ> {
    constexpr static auto value =
        &GateImplOrSVType::template applyMultiRZ<PrecisionT, ParamT>;
};

/**
 * @brief Return a specific member function pointer for a given generator
 * operation. See speicalized classes.
 */
template <class PrecisionT, class GateImplOrSVType, GeneratorOperation gntr_op>
struct GeneratorOpToMemberFuncPtr; // Link error when used

template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType,
                                  GeneratorOperation::RX> {
    constexpr static auto value =
        &GateImplOrSVType::template applyGeneratorRX<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType,
                                  GeneratorOperation::RY> {
    constexpr static auto value =
        &GateImplOrSVType::template applyGeneratorRY<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType,
                                  GeneratorOperation::RZ> {
    constexpr static auto value =
        &GateImplOrSVType::template applyGeneratorRZ<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType,
                                  GeneratorOperation::PhaseShift> {
    constexpr static auto value =
        &GateImplOrSVType::template applyGeneratorPhaseShift<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType,
                                  GeneratorOperation::CRX> {
    constexpr static auto value =
        &GateImplOrSVType::template applyGeneratorCRX<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType,
                                  GeneratorOperation::CRY> {
    constexpr static auto value =
        &GateImplOrSVType::template applyGeneratorCRY<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType,
                                  GeneratorOperation::CRZ> {
    constexpr static auto value =
        &GateImplOrSVType::template applyGeneratorCRZ<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType,
                                  GeneratorOperation::ControlledPhaseShift> {
    constexpr static auto value =
        &GateImplOrSVType::template applyGeneratorControlledPhaseShift<
            PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType,
                                  GeneratorOperation::MultiRZ> {
    constexpr static auto value =
        &GateImplOrSVType::template applyGeneratorMultiRZ<PrecisionT>;
};

//
namespace Internal {
/**
 * @brief Gate operation pointer type for a statevector. See all specialized
 * types.
 */
template <class SVType, class ParamT, size_t num_params> struct GateMemFuncPtr {
    static_assert(num_params < 2 || num_params == 3,
                  "The given num_params is not supported.");
};
/**
 * @brief Function pointer type for a gate operation without parameters.
 */
template <class SVType, class ParamT> struct GateMemFuncPtr<SVType, ParamT, 0> {
    using Type = void (SVType::*)(const std::vector<size_t> &, bool);
};
/**
 * @brief Function pointer type for a gate operation with a single parameter.
 */
template <class SVType, class ParamT> struct GateMemFuncPtr<SVType, ParamT, 1> {
    using Type = void (SVType::*)(const std::vector<size_t> &, bool, ParamT);
};
/**
 * @brief Function pointer type for a gate operation with three parameters.
 */
template <class SVType, class ParamT> struct GateMemFuncPtr<SVType, ParamT, 3> {
    using Type = void (SVType::*)(const std::vector<size_t> &, bool, ParamT,
                                  ParamT, ParamT);
};

template <class SVType, class ParamT, size_t num_params>
using GateMemFuncPtrT =
    typename GateMemFuncPtr<SVType, ParamT, num_params>::Type;

/**
 * @brief Gate operation pointer type. See all specialized types.
 */
template <class PrecisionT, class ParamT, size_t num_params>
struct GateFuncPtr {
    static_assert(num_params < 2 || num_params == 3,
                  "The given num_params is not supported.");
};

/**
 * @brief Pointer type for a gate operation without parameters.
 */
template <class PrecisionT, class ParamT>
struct GateFuncPtr<PrecisionT, ParamT, 0> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &, bool);
};
/**
 * @brief Pointer type for a gate operation with a single parameter
 */
template <class PrecisionT, class ParamT>
struct GateFuncPtr<PrecisionT, ParamT, 1> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &, bool, ParamT);
};
/**
 * @brief Pointer type for a gate operation with three paramters
 */
template <class PrecisionT, class ParamT>
struct GateFuncPtr<PrecisionT, ParamT, 3> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &, bool, ParamT, ParamT,
                          ParamT);
};

/**
 * @brief Pointer type for a generator operation
 */
template <class PrecisionT> struct GeneratorFuncPtr {
    using Type = PrecisionT (*)(std::complex<PrecisionT> *, size_t,
                                const std::vector<size_t> &, bool);
};
} // namespace Internal

/**
 * @brief Convinient type alias for GateFuncPtr. See GateFuncPtr for details.
 */
template <class PrecisionT, class ParamT, size_t num_params>
using GateFuncPtrT =
    typename Internal::GateFuncPtr<PrecisionT, ParamT, num_params>::Type;

template <class PrecisionT>
using GeneratorFuncPtrT = typename Internal::GeneratorFuncPtr<PrecisionT>::Type;

/**
 * @defgroup Call gate operation with provided arguments
 *
 * @tparam PrecisionT Floating point type for the state-vector.
 * @tparam ParamT Floating point type for the gate paramters.
 * @param func Function pointer for the gate operation.
 * @param num_qubits The number of qubits of the state-vector.
 * @param wires Wires the gate applies to.
 * @param inverse If true, we apply the inverse of the gate.
 * @param params The list of gate paramters.
 */
/// @{
/**
 * @brief Overload for a gate operation without parameters
 */
template <class PrecisionT, class ParamT>
inline void callGateOps(GateFuncPtrT<PrecisionT, ParamT, 0> func,
                        std::complex<PrecisionT> *data, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        [[maybe_unused]] const std::vector<ParamT> &params) {
    assert(params.empty());
    func(data, num_qubits, wires, inverse);
}

/**
 * @brief Overload for a gate operation for a single paramter
 */
template <class PrecisionT, class ParamT>
inline void callGateOps(GateFuncPtrT<PrecisionT, ParamT, 1> func,
                        std::complex<PrecisionT> *data, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<ParamT> &params) {
    assert(params.size() == 1);
    func(data, num_qubits, wires, inverse, params[0]);
}

/**
 * @brief Overload for a gate operation for three paramters
 */
template <class PrecisionT, class ParamT>
inline void callGateOps(GateFuncPtrT<PrecisionT, ParamT, 3> func,
                        std::complex<PrecisionT> *data, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<ParamT> &params) {
    assert(params.size() == 3);
    func(data, num_qubits, wires, inverse, params[0], params[1], params[2]);
}
/// @}
/**
 * @brief Call a generator operation.
 *
 * @tparam PrecisionT Floating point type for the state-vector.
 * @return Scaling factor
 */
template <class PrecisionT>
inline PrecisionT callGeneratorOps(GeneratorFuncPtrT<PrecisionT> func,
                                   std::complex<PrecisionT> *data,
                                   size_t num_qubits,
                                   const std::vector<size_t> &wires, bool adj) {
    return func(data, num_qubits, wires, adj);
}
} // namespace Pennylane
