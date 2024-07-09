// Copyright 2018-2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "BasicGateFunctors.hpp"
#include "BitUtil.hpp"
#include "GateOperation.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using Kokkos::kokkos_swap;
using Pennylane::Gates::GeneratorOperation;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Functors {

template <class ExecutionSpace, class PrecisionT>
void applyGenPhaseShift(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0, const std::size_t i1) {
            [[maybe_unused]] auto i1_ = i1;
            arr(i0) = 0.0;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenControlledPhaseShift(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            [[maybe_unused]] const auto i11_ = i11;

            arr(i00) = 0.0;
            arr(i01) = 0.0;
            arr(i10) = 0.0;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenCRX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 const std::size_t num_qubits,
                 const std::vector<std::size_t> &wires,
                 [[maybe_unused]] const bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            arr(i00) = 0.0;
            arr(i01) = 0.0;
            kokkos_swap(arr(i10), arr(i11));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenCRY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 const std::size_t num_qubits,
                 const std::vector<std::size_t> &wires,
                 [[maybe_unused]] const bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            arr(i00) = 0.0;
            arr(i01) = 0.0;
            const auto v0 = arr(i10);
            arr(i10) =
                Kokkos::complex<PrecisionT>{imag(arr(i11)), -real(arr(i11))};
            arr(i11) = Kokkos::complex<PrecisionT>{-imag(v0), real(v0)};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenCRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 const std::size_t num_qubits,
                 const std::vector<std::size_t> &wires,
                 [[maybe_unused]] const bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            [[maybe_unused]] const auto i10_ = i10;
            arr(i00) = 0.0;
            arr(i01) = 0.0;
            arr(i11) *= -1;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingXX(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            kokkos_swap(arr(i00), arr(i11));
            kokkos_swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingXY(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            kokkos_swap(arr(i10), arr(i01));
            arr(i00) = 0.0;
            arr(i11) = 0.0;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingYY(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            const auto v00 = arr(i00);
            arr(i00) = -arr(i11);
            arr(i11) = -v00;
            kokkos_swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingZZ(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            [[maybe_unused]] const auto i00_ = i00;
            [[maybe_unused]] const auto i11_ = i11;

            arr(i10) *= -1;
            arr(i01) *= -1;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitation(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            arr(i00) = 0.0;
            arr(i01) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
            arr(i10) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
            arr(i11) = 0.0;
            kokkos_swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            [[maybe_unused]] const auto i00_ = i00;
            [[maybe_unused]] const auto i11_ = i11;

            arr(i01) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
            arr(i10) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
            kokkos_swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            [[maybe_unused]] const auto i00_ = i00;
            [[maybe_unused]] const auto i11_ = i11;

            arr(i00) *= -1;
            arr(i01) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
            arr(i10) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
            arr(i11) *= -1;
            kokkos_swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitation(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC4Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0000, const std::size_t i0001,
                      const std::size_t i0010, const std::size_t i0011,
                      const std::size_t i0100, const std::size_t i0101,
                      const std::size_t i0110, const std::size_t i0111,
                      const std::size_t i1000, const std::size_t i1001,
                      const std::size_t i1010, const std::size_t i1011,
                      const std::size_t i1100, const std::size_t i1101,
                      const std::size_t i1110, const std::size_t i1111) {
            const Kokkos::complex<PrecisionT> v3 = arr(i0011);
            const Kokkos::complex<PrecisionT> v12 = arr(i1100);
            arr(i0000) = 0.0;
            arr(i0001) = 0.0;
            arr(i0010) = 0.0;
            arr(i0011) = v12 * Kokkos::complex<PrecisionT>{0.0, -1.0};
            arr(i0100) = 0.0;
            arr(i0101) = 0.0;
            arr(i0110) = 0.0;
            arr(i0111) = 0.0;
            arr(i1000) = 0.0;
            arr(i1001) = 0.0;
            arr(i1010) = 0.0;
            arr(i1011) = 0.0;
            arr(i1100) = v3 * Kokkos::complex<PrecisionT>{0.0, 1.0};
            arr(i1101) = 0.0;
            arr(i1110) = 0.0;
            arr(i1111) = 0.0;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC4Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0000, const std::size_t i0001,
                      const std::size_t i0010, const std::size_t i0011,
                      const std::size_t i0100, const std::size_t i0101,
                      const std::size_t i0110, const std::size_t i0111,
                      const std::size_t i1000, const std::size_t i1001,
                      const std::size_t i1010, const std::size_t i1011,
                      const std::size_t i1100, const std::size_t i1101,
                      const std::size_t i1110, const std::size_t i1111) {
            [[maybe_unused]] const auto i0000_ = i0000;
            [[maybe_unused]] const auto i0001_ = i0001;
            [[maybe_unused]] const auto i0010_ = i0010;
            [[maybe_unused]] const auto i0100_ = i0100;
            [[maybe_unused]] const auto i0101_ = i0101;
            [[maybe_unused]] const auto i0110_ = i0110;
            [[maybe_unused]] const auto i0111_ = i0111;
            [[maybe_unused]] const auto i1000_ = i1000;
            [[maybe_unused]] const auto i1001_ = i1001;
            [[maybe_unused]] const auto i1010_ = i1010;
            [[maybe_unused]] const auto i1011_ = i1011;
            [[maybe_unused]] const auto i1101_ = i1101;
            [[maybe_unused]] const auto i1110_ = i1110;
            [[maybe_unused]] const auto i1111_ = i1111;
            arr(i0011) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
            arr(i1100) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
            kokkos_swap(arr(i1100), arr(i0011));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC4Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0000, const std::size_t i0001,
                      const std::size_t i0010, const std::size_t i0011,
                      const std::size_t i0100, const std::size_t i0101,
                      const std::size_t i0110, const std::size_t i0111,
                      const std::size_t i1000, const std::size_t i1001,
                      const std::size_t i1010, const std::size_t i1011,
                      const std::size_t i1100, const std::size_t i1101,
                      const std::size_t i1110, const std::size_t i1111) {
            [[maybe_unused]] const auto i0000_ = i0000;
            [[maybe_unused]] const auto i0001_ = i0001;
            [[maybe_unused]] const auto i0010_ = i0010;
            [[maybe_unused]] const auto i0100_ = i0100;
            [[maybe_unused]] const auto i0101_ = i0101;
            [[maybe_unused]] const auto i0110_ = i0110;
            [[maybe_unused]] const auto i0111_ = i0111;
            [[maybe_unused]] const auto i1000_ = i1000;
            [[maybe_unused]] const auto i1001_ = i1001;
            [[maybe_unused]] const auto i1010_ = i1010;
            [[maybe_unused]] const auto i1011_ = i1011;
            [[maybe_unused]] const auto i1101_ = i1101;
            [[maybe_unused]] const auto i1110_ = i1110;
            [[maybe_unused]] const auto i1111_ = i1111;
            arr(i0011) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
            arr(i1100) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
            kokkos_swap(arr(i1100), arr(i0011));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenMultiRZ(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    std::size_t wires_parity = static_cast<std::size_t>(0U);
    for (std::size_t wire : wires) {
        wires_parity |=
            (static_cast<std::size_t>(1U) << (num_qubits - wire - 1));
    }
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, exp2(num_qubits)),
        KOKKOS_LAMBDA(const std::size_t k) {
            arr_(k) *= static_cast<PrecisionT>(
                1 - 2 * int(Kokkos::Impl::bit_count(k & wires_parity) % 2));
        });
}

template <class ExecutionSpace, class PrecisionT>
PrecisionT applyNamedGenerator(const GeneratorOperation generator_op,
                               Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                               const std::size_t num_qubits,
                               const std::vector<std::size_t> &wires,
                               const bool inverse = false,
                               const std::vector<PrecisionT> &params = {}) {
    switch (generator_op) {
    case GeneratorOperation::RX:
        applyNamedOperation<ExecutionSpace>(GateOperation::PauliX, arr_,
                                            num_qubits, wires, inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::RY:
        applyNamedOperation<ExecutionSpace>(GateOperation::PauliY, arr_,
                                            num_qubits, wires, inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::RZ:
        applyNamedOperation<ExecutionSpace>(GateOperation::PauliZ, arr_,
                                            num_qubits, wires, inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::PhaseShift:
        applyGenPhaseShift<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                           params);
        return static_cast<PrecisionT>(1.0);
    case GeneratorOperation::ControlledPhaseShift:
        applyGenControlledPhaseShift<ExecutionSpace>(arr_, num_qubits, wires,
                                                     inverse, params);
        return static_cast<PrecisionT>(1);
    case GeneratorOperation::CRX:
        applyGenCRX<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::CRY:
        applyGenCRY<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::CRZ:
        applyGenCRZ<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingXX:
        applyGenIsingXX<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingXY:
        applyGenIsingXY<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        return static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingYY:
        applyGenIsingYY<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingZZ:
        applyGenIsingZZ<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::SingleExcitation:
        applyGenSingleExcitation<ExecutionSpace>(arr_, num_qubits, wires,
                                                 inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::SingleExcitationMinus:
        applyGenSingleExcitationMinus<ExecutionSpace>(arr_, num_qubits, wires,
                                                      inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::SingleExcitationPlus:
        applyGenSingleExcitationPlus<ExecutionSpace>(arr_, num_qubits, wires,
                                                     inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::DoubleExcitation:
        applyGenDoubleExcitation<ExecutionSpace>(arr_, num_qubits, wires,
                                                 inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::DoubleExcitationMinus:
        applyGenDoubleExcitationMinus<ExecutionSpace>(arr_, num_qubits, wires,
                                                      inverse, params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::DoubleExcitationPlus:
        applyGenDoubleExcitationPlus<ExecutionSpace>(arr_, num_qubits, wires,
                                                     inverse, params);
        return static_cast<PrecisionT>(0.5);
    case GeneratorOperation::MultiRZ:
        applyGenMultiRZ<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::GlobalPhase:
        return static_cast<PrecisionT>(-1.0);
    /// LCOV_EXCL_START
    default:
        PL_ABORT("Generator operation does not exist.");
        /// LCOV_EXCL_STOP
    }
}

} // namespace Pennylane::LightningKokkos::Functors
