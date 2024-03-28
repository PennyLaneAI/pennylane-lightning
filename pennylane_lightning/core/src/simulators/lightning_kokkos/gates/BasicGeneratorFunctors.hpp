// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
using Kokkos::Experimental::swap;
using Pennylane::Gates::GeneratorOperation;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Functors {

template <class ExecutionSpace, class PrecisionT>
void applyGenPhaseShift(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0,
                      [[maybe_unused]] const std::size_t i1) {
            arr(i0) = Kokkos::complex<PrecisionT>{0.0, 0.0};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenControlledPhaseShift(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10,
                      [[maybe_unused]] const std::size_t i11) {
            arr(i00) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i01) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i10) = Kokkos::complex<PrecisionT>{0.0, 0.0};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenCRX(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                 const std::size_t num_qubits, const std::vector<size_t> &wires,
                 [[maybe_unused]] const bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            arr(i00) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i01) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            swap(arr(i10), arr(i11));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenCRY(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                 const std::size_t num_qubits, const std::vector<size_t> &wires,
                 [[maybe_unused]] const bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            arr(i00) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i01) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            const auto v0 = arr(i10);
            arr(i10) =
                Kokkos::complex<PrecisionT>{imag(arr(i11)), -real(arr(i11))};
            arr(i11) = Kokkos::complex<PrecisionT>{-imag(v0), real(v0)};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenCRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                 const std::size_t num_qubits, const std::vector<size_t> &wires,
                 [[maybe_unused]] const bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      [[maybe_unused]] const std::size_t i10,
                      const std::size_t i11) {
            arr(i00) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i01) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i11) *= -1;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingXX(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            swap(arr(i00), arr(i11));
            swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingXY(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            swap(arr(i10), arr(i01));
            arr(i00) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i11) = Kokkos::complex<PrecisionT>{0.0, 0.0};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingYY(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
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
            swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingZZ(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      const std::size_t i01, const std::size_t i10,
                      [[maybe_unused]] const std::size_t i11) {
            arr(i10) *= -1;
            arr(i01) *= -1;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitation(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            arr(i00) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i01) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
            arr(i10) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
            arr(i11) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      const std::size_t i01, const std::size_t i10,
                      [[maybe_unused]] const std::size_t i11) {
            arr(i01) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
            arr(i10) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
            swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      const std::size_t i01, const std::size_t i10,
                      [[maybe_unused]] const std::size_t i11) {
            arr(i00) *= -1;
            arr(i01) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
            arr(i10) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
            arr(i11) *= -1;
            swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitation(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
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
            arr(i0000) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i0001) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i0010) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i0011) = v12 * Kokkos::complex<PrecisionT>{0.0, -1.0};
            arr(i0100) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i0101) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i0110) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i0111) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i1000) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i1001) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i1010) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i1011) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i1100) = v3 * Kokkos::complex<PrecisionT>{0.0, 1.0};
            arr(i1101) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i1110) = Kokkos::complex<PrecisionT>{0.0, 0.0};
            arr(i1111) = Kokkos::complex<PrecisionT>{0.0, 0.0};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC4Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(
            Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
            [[maybe_unused]] const std::size_t i0000,
            [[maybe_unused]] const std::size_t i0001,
            [[maybe_unused]] const std::size_t i0010, const std::size_t i0011,
            [[maybe_unused]] const std::size_t i0100,
            [[maybe_unused]] const std::size_t i0101,
            [[maybe_unused]] const std::size_t i0110,
            [[maybe_unused]] const std::size_t i0111,
            [[maybe_unused]] const std::size_t i1000,
            [[maybe_unused]] const std::size_t i1001,
            [[maybe_unused]] const std::size_t i1010,
            [[maybe_unused]] const std::size_t i1011, const std::size_t i1100,
            [[maybe_unused]] const std::size_t i1101,
            [[maybe_unused]] const std::size_t i1110,
            [[maybe_unused]] const std::size_t i1111) {
            arr(i0011) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
            arr(i1100) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
            swap(arr(i1100), arr(i0011));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC4Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(
            Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
            [[maybe_unused]] const std::size_t i0000,
            [[maybe_unused]] const std::size_t i0001,
            [[maybe_unused]] const std::size_t i0010, const std::size_t i0011,
            [[maybe_unused]] const std::size_t i0100,
            [[maybe_unused]] const std::size_t i0101,
            [[maybe_unused]] const std::size_t i0110,
            [[maybe_unused]] const std::size_t i0111,
            [[maybe_unused]] const std::size_t i1000,
            [[maybe_unused]] const std::size_t i1001,
            [[maybe_unused]] const std::size_t i1010,
            [[maybe_unused]] const std::size_t i1011, const std::size_t i1100,
            [[maybe_unused]] const std::size_t i1101,
            [[maybe_unused]] const std::size_t i1110,
            [[maybe_unused]] const std::size_t i1111) {
            arr(i0011) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
            arr(i1100) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
            swap(arr(i1100), arr(i0011));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyGenMultiRZ(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    std::size_t wires_parity = static_cast<size_t>(0U);
    for (size_t wire : wires) {
        wires_parity |= (static_cast<size_t>(1U) << (num_qubits - wire - 1));
    }
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, exp2(num_qubits)),
        KOKKOS_LAMBDA(const std::size_t k) {
            arr_(k) *= static_cast<PrecisionT>(
                1 - 2 * int(Kokkos::Impl::bit_count(k & wires_parity) % 2));
        });
}

template <class PrecisionT>
PrecisionT namedGeneratorFactor(const GeneratorOperation generator_op) {
    switch (generator_op) {
    case GeneratorOperation::RX:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::RY:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::RZ:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::PhaseShift:
        return static_cast<PrecisionT>(1.0);
    case GeneratorOperation::ControlledPhaseShift:
        return static_cast<PrecisionT>(1);
    case GeneratorOperation::CRX:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::CRY:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::CRZ:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingXX:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingXY:
        return static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingYY:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingZZ:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::SingleExcitation:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::SingleExcitationMinus:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::SingleExcitationPlus:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::DoubleExcitation:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::DoubleExcitationMinus:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::DoubleExcitationPlus:
        return static_cast<PrecisionT>(0.5);
    case GeneratorOperation::MultiRZ:
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::GlobalPhase:
        return static_cast<PrecisionT>(-1.0);
    /// LCOV_EXCL_START
    default:
        PL_ABORT("Generator operation does not exist.");
        /// LCOV_EXCL_STOP
    }
}

template <class ExecutionSpace, class PrecisionT>
PrecisionT applyNamedGenerator(
    const GeneratorOperation generator_op,
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    const bool inverse = false, const std::vector<PrecisionT> &params = {}) {
    switch (generator_op) {
    case GeneratorOperation::RX:
        applyNamedOperation<ExecutionSpace>(GateOperation::PauliX, arr_,
                                            num_qubits, wires, inverse, params);
        break;
    case GeneratorOperation::RY:
        applyNamedOperation<ExecutionSpace>(GateOperation::PauliY, arr_,
                                            num_qubits, wires, inverse, params);
        break;
    case GeneratorOperation::RZ:
        applyNamedOperation<ExecutionSpace>(GateOperation::PauliZ, arr_,
                                            num_qubits, wires, inverse, params);
        break;
    case GeneratorOperation::PhaseShift:
        applyGenPhaseShift<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                           params);
        break;
    case GeneratorOperation::ControlledPhaseShift:
        applyGenControlledPhaseShift<ExecutionSpace>(arr_, num_qubits, wires,
                                                     inverse, params);
        break;
    case GeneratorOperation::CRX:
        applyGenCRX<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        break;
    case GeneratorOperation::CRY:
        applyGenCRY<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        break;
    case GeneratorOperation::CRZ:
        applyGenCRZ<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        break;
    case GeneratorOperation::IsingXX:
        applyGenIsingXX<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        break;
    case GeneratorOperation::IsingXY:
        applyGenIsingXY<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        break;
    case GeneratorOperation::IsingYY:
        applyGenIsingYY<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        break;
    case GeneratorOperation::IsingZZ:
        applyGenIsingZZ<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        break;
    case GeneratorOperation::SingleExcitation:
        applyGenSingleExcitation<ExecutionSpace>(arr_, num_qubits, wires,
                                                 inverse, params);
        break;
    case GeneratorOperation::SingleExcitationMinus:
        applyGenSingleExcitationMinus<ExecutionSpace>(arr_, num_qubits, wires,
                                                      inverse, params);
        break;
    case GeneratorOperation::SingleExcitationPlus:
        applyGenSingleExcitationPlus<ExecutionSpace>(arr_, num_qubits, wires,
                                                     inverse, params);
        break;
    case GeneratorOperation::DoubleExcitation:
        applyGenDoubleExcitation<ExecutionSpace>(arr_, num_qubits, wires,
                                                 inverse, params);
        break;
    case GeneratorOperation::DoubleExcitationMinus:
        applyGenDoubleExcitationMinus<ExecutionSpace>(arr_, num_qubits, wires,
                                                      inverse, params);
        break;
    case GeneratorOperation::DoubleExcitationPlus:
        applyGenDoubleExcitationPlus<ExecutionSpace>(arr_, num_qubits, wires,
                                                     inverse, params);
        break;
    case GeneratorOperation::MultiRZ:
        applyGenMultiRZ<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        break;
    case GeneratorOperation::GlobalPhase:
        break;
    /// LCOV_EXCL_START
    default:
        PL_ABORT("Generator operation does not exist.");
        /// LCOV_EXCL_STOP
    }
    return namedGeneratorFactor<PrecisionT>(generator_op);
}

} // namespace Pennylane::LightningKokkos::Functors