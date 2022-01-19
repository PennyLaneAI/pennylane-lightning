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

namespace Pennylane {

/**
 * @brief Return a specific member function pointer for a given gate operation. See
 * speicalized classes.
 */
template <class PrecisionT, class ParamT, class GateImplOrSVType, GateOperation gate_op>
struct GateOpToMemberFuncPtr;

template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::PauliX> {
    constexpr static auto value = &GateImplOrSVType::template applyPauliX<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::PauliY> {
    constexpr static auto value = &GateImplOrSVType::template applyPauliY<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::PauliZ> {
    constexpr static auto value = &GateImplOrSVType::template applyPauliZ<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::Hadamard> {
    constexpr static auto value = &GateImplOrSVType::template applyHadamard<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::S> {
    constexpr static auto value = &GateImplOrSVType::template applyS<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::T> {
    constexpr static auto value = &GateImplOrSVType::template applyT<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::PhaseShift> {
    constexpr static auto value = &GateImplOrSVType::template applyPhaseShift<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::RX> {
    constexpr static auto value = &GateImplOrSVType::template applyRX<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::RY> {
    constexpr static auto value = &GateImplOrSVType::template applyRY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::RZ> {
    constexpr static auto value = &GateImplOrSVType::template applyRZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::Rot> {
    constexpr static auto value = &GateImplOrSVType::template applyRot<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::CNOT> {
    constexpr static auto value = &GateImplOrSVType::template applyCNOT<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::CY> {
    constexpr static auto value = &GateImplOrSVType::template applyCY<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::CZ> {
    constexpr static auto value = &GateImplOrSVType::template applyCZ<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::SWAP> {
    constexpr static auto value = &GateImplOrSVType::template applySWAP<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::ControlledPhaseShift> {
    constexpr static auto value = &GateImplOrSVType::template applyControlledPhaseShift<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::CRX> {
    constexpr static auto value = &GateImplOrSVType::template applyCRX<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::CRY> {
    constexpr static auto value = &GateImplOrSVType::template applyCRY<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::CRZ> {
    constexpr static auto value = &GateImplOrSVType::template applyCRZ<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::CRot> {
    constexpr static auto value = &GateImplOrSVType::template applyCRot<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::Toffoli> {
    constexpr static auto value = &GateImplOrSVType::template applyToffoli<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::CSWAP> {
    constexpr static auto value = &GateImplOrSVType::template applyCSWAP<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplOrSVType>
struct GateOpToMemberFuncPtr <PrecisionT, ParamT, GateImplOrSVType, GateOperation::Matrix> {
    constexpr static auto value = &GateImplOrSVType::template applyPauliX<PrecisionT>;
};

/**
 * @brief Return a specific member function pointer for a given generator operation. See
 * speicalized classes.
 */
template <class PrecisionT, class GateImplOrSVType, GeneratorOperation gntr_op>
struct GeneratorOpToMemberFuncPtr;
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType, GeneratorOperation::PhaseShift> {
    constexpr static auto value = &GateImplOrSVType::template applyGeneratorPhaseShift<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType, GeneratorOperation::CRX> {
    constexpr static auto value = &GateImplOrSVType::template applyGeneratorCRX<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType, GeneratorOperation::CRY> {
    constexpr static auto value = &GateImplOrSVType::template applyGeneratorCRY<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType, GeneratorOperation::CRZ> {
    constexpr static auto value = &GateImplOrSVType::template applyGeneratorCRZ<PrecisionT>;
};
template <class PrecisionT, class GateImplOrSVType>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplOrSVType, GeneratorOperation::ControlledPhaseShift> {
    constexpr static auto value = &GateImplOrSVType::template applyGeneratorCRX<PrecisionT>;
};
} // namespace Pennylane
