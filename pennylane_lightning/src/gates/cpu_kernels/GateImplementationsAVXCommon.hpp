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
 * Defines kernel functions for all AVX
 */
#pragma once

// General implementations
#include "Macros.hpp"

#ifdef PL_USE_AVX2
#include "avx_common/AVX2Concept.hpp"
#endif
#ifdef PL_USE_AVX512F
#include "avx_common/AVX512Concept.hpp"
#endif
#include "avx_common/ApplyCNOT.hpp"
#include "avx_common/ApplyCZ.hpp"
#include "avx_common/ApplyHadamard.hpp"
#include "avx_common/ApplyIsingXX.hpp"
#include "avx_common/ApplyIsingYY.hpp"
#include "avx_common/ApplyIsingZZ.hpp"
#include "avx_common/ApplyPauliX.hpp"
#include "avx_common/ApplyPauliY.hpp"
#include "avx_common/ApplyPauliZ.hpp"
#include "avx_common/ApplyPhaseShift.hpp"
#include "avx_common/ApplyRX.hpp"
#include "avx_common/ApplyRY.hpp"
#include "avx_common/ApplyRZ.hpp"
#include "avx_common/ApplyS.hpp"
#include "avx_common/ApplySWAP.hpp"
#include "avx_common/ApplySingleQubitOp.hpp"
#include "avx_common/ApplyT.hpp"
#include "avx_common/SingleQubitGateHelper.hpp"
#include "avx_common/TwoQubitGateHelper.hpp"

#include "Error.hpp"
#include "GateImplementationsLM.hpp"
#include "GateOperation.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"

#include <immintrin.h>

#include <complex>
#include <vector>

namespace Pennylane::Gates {
template <class Derived>
class GateImplementationsAVXCommon
    : public PauliGenerator<GateImplementationsAVXCommon<Derived>> {
  public:
    constexpr static std::array implemented_gates = {
        GateOperation::PauliX,     GateOperation::PauliY,
        GateOperation::PauliZ,     GateOperation::Hadamard,
        GateOperation::S,          GateOperation::T,
        GateOperation::PhaseShift, GateOperation::RX,
        GateOperation::RY,         GateOperation::RZ,
        GateOperation::Rot,        GateOperation::CNOT,
        GateOperation::CZ,         GateOperation::SWAP,
        GateOperation::IsingXX,    GateOperation::IsingYY,
        GateOperation::IsingZZ,
        /* CY, IsingXY, ControlledPhaseShift, CRX, CRY, CRZ, CRot */
    };

    constexpr static std::array implemented_generators = {
        GeneratorOperation::RX, GeneratorOperation::RY, GeneratorOperation::RZ};

    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliXAVX =
            AVX::ApplyPauliX<PrecisionT,
                             Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        assert(wires.size() == 1);
        auto helper = AVX::SingleQubitGateWithoutParamHelper<ApplyPauliXAVX>(
            &GateImplementationsLM::applyPauliX);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void applyPauliY(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliYAVX =
            AVX::ApplyPauliY<PrecisionT,
                             Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        assert(wires.size() == 1);
        auto helper = AVX::SingleQubitGateWithoutParamHelper<ApplyPauliYAVX>(
            &GateImplementationsLM::applyPauliY);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliZAVX =
            AVX::ApplyPauliZ<PrecisionT,
                             Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        assert(wires.size() == 1);
        auto helper = AVX::SingleQubitGateWithoutParamHelper<ApplyPauliZAVX>(
            &GateImplementationsLM::applyPauliZ);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void applyS(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        using ApplySAVX =
            AVX::ApplyS<PrecisionT, Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        auto helper = AVX::SingleQubitGateWithoutParamHelper<ApplySAVX>(
            &GateImplementationsLM::applyS);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void applyT(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        using ApplyTAVX =
            AVX::ApplyT<PrecisionT, Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        assert(wires.size() == 1);
        auto helper = AVX::SingleQubitGateWithoutParamHelper<ApplyTAVX>(
            &GateImplementationsLM::applyT);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyPhaseShift(std::complex<PrecisionT> *arr,
                                const size_t num_qubits,
                                const std::vector<size_t> &wires, bool inverse,
                                ParamT angle) {
        using ApplyPhaseShiftAVX =
            AVX::ApplyPhaseShift<PrecisionT,
                                 Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        assert(wires.size() == 1);
        auto helper =
            AVX::SingleQubitGateWithParamHelper<ApplyPhaseShiftAVX, ParamT>(
                &GateImplementationsLM::applyPhaseShift);
        helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT>
    static void applyHadamard(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        using ApplyHadamardAVX =
            AVX::ApplyHadamard<PrecisionT,
                               Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");
        assert(wires.size() == 1);
        auto helper = AVX::SingleQubitGateWithoutParamHelper<ApplyHadamardAVX>(
            &GateImplementationsLM::applyHadamard);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRX(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        using ApplyRXAVX = AVX::ApplyRX<PrecisionT, Derived::packed_bytes /
                                                        sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");
        assert(wires.size() == 1);
        auto helper = AVX::SingleQubitGateWithParamHelper<ApplyRXAVX, ParamT>(
            &GateImplementationsLM::applyRX);
        helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRY(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        using ApplyRYAVX = AVX::ApplyRY<PrecisionT, Derived::packed_bytes /
                                                        sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");
        assert(wires.size() == 1);
        auto helper = AVX::SingleQubitGateWithParamHelper<ApplyRYAVX, ParamT>(
            &GateImplementationsLM::applyRY);
        helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        using ApplyRZAVX = AVX::ApplyRZ<PrecisionT, Derived::packed_bytes /
                                                        sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");
        assert(wires.size() == 1);
        auto helper = AVX::SingleQubitGateWithParamHelper<ApplyRZAVX, ParamT>(
            &GateImplementationsLM::applyRZ);
        helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRot(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT phi, ParamT theta, ParamT omega) {
        assert(wires.size() == 1);

        const auto rotMat =
            (inverse) ? Gates::getRot<PrecisionT>(-omega, -theta, -phi)
                      : Gates::getRot<PrecisionT>(phi, theta, omega);

        Derived::applySingleQubitOp(arr, num_qubits, rotMat.data(), wires);
    }

    /* Two-qubit gates*/
    template <class PrecisionT>
    static void applyCZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        using ApplyCZAVX = AVX::ApplyCZ<PrecisionT, Derived::packed_bytes /
                                                        sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        assert(wires.size() == 2);

        const AVX::TwoQubitGateWithoutParamHelper<ApplyCZAVX> gate_helper(
            &GateImplementationsLM::applyCZ<PrecisionT>);

        gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void
    applySWAP(std::complex<PrecisionT> *arr, const size_t num_qubits,
              const std::vector<size_t> &wires, [[maybe_unused]] bool inverse) {
        using ApplySWAPAVX = AVX::ApplySWAP<PrecisionT, Derived::packed_bytes /
                                                            sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        assert(wires.size() == 2);

        const AVX::TwoQubitGateWithoutParamHelper<ApplySWAPAVX> gate_helper(
            &GateImplementationsLM::applySWAP<PrecisionT>);

        gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void
    applyCNOT(std::complex<PrecisionT> *arr, const size_t num_qubits,
              const std::vector<size_t> &wires, [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);

        using ApplyCNOTAVX = AVX::ApplyCNOT<PrecisionT, Derived::packed_bytes /
                                                            sizeof(PrecisionT)>;

        static_assert(AVX::AsymmetricTwoQubitGateWithoutParam<ApplyCNOTAVX>);

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        assert(wires.size() == 2);

        const AVX::TwoQubitGateWithoutParamHelper<ApplyCNOTAVX> gate_helper(
            &GateImplementationsLM::applyCNOT<PrecisionT>);

        gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingXX(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        assert(wires.size() == 2);

        using ApplyIsingXXAVX =
            AVX::ApplyIsingXX<PrecisionT,
                              Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        const AVX::TwoQubitGateWithParamHelper<ApplyIsingXXAVX, ParamT>
            gate_helper(
                &GateImplementationsLM::applyIsingXX<PrecisionT, ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingYY(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        assert(wires.size() == 2);

        using ApplyIsingYYAVX =
            AVX::ApplyIsingYY<PrecisionT,
                              Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        const AVX::TwoQubitGateWithParamHelper<ApplyIsingYYAVX, ParamT>
            gate_helper(
                &GateImplementationsLM::applyIsingYY<PrecisionT, ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingZZ(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        using ApplyIsingZZAVX =
            AVX::ApplyIsingZZ<PrecisionT,
                              Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        assert(wires.size() == 2);

        const AVX::TwoQubitGateWithParamHelper<ApplyIsingZZAVX, ParamT>
            gate_helper(
                &GateImplementationsLM::applyIsingZZ<PrecisionT, ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }
};

} // namespace Pennylane::Gates
