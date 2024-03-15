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

#include "BitUtil.hpp"
#include "Gates.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using Kokkos::Experimental::swap;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Functors {

template <class PrecisionT, class FuncT> class applyNC1Functor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    const FuncT core_function;
    const std::size_t rev_wire;
    const std::size_t rev_wire_shift;
    const std::size_t wire_parity;
    const std::size_t wire_parity_inv;

  public:
    template <class ExecutionSpace>
    applyNC1Functor([[maybe_unused]] ExecutionSpace exec,
                    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                    std::size_t num_qubits, const std::vector<size_t> &wires,
                    FuncT core_function_)
        : arr(arr_), core_function(core_function_),
          rev_wire(num_qubits - wires[0] - 1),
          rev_wire_shift((static_cast<size_t>(1U) << rev_wire)),
          wire_parity(fillTrailingOnes(rev_wire)),
          wire_parity_inv(fillLeadingOnes(rev_wire + 1)) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(0, exp2(num_qubits - 1)),
            *this);
    }
    KOKKOS_FUNCTION void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        core_function(arr, i0, i1);
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyPauliX(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                 const std::size_t num_qubits, const std::vector<size_t> &wires,
                 [[maybe_unused]] const bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0, const std::size_t i1) {
            Kokkos::Experimental::swap(arr(i0), arr(i1));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyPauliY(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                 const std::size_t num_qubits, const std::vector<size_t> &wires,
                 [[maybe_unused]] const bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0, const std::size_t i1) {
            const auto v0 = arr(i0);
            const auto v1 = arr(i1);
            arr(i0) = Kokkos::complex<PrecisionT>{imag(v1), -real(v1)};
            arr(i1) = Kokkos::complex<PrecisionT>{-imag(v0), real(v0)};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyPauliZ(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                 const std::size_t num_qubits, const std::vector<size_t> &wires,
                 [[maybe_unused]] const bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i0,
                      const std::size_t i1) { arr(i1) *= -1.0; });
}

template <class ExecutionSpace, class PrecisionT>
void applyHadamard(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    const bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i0,
                      const std::size_t i1) {
            const Kokkos::complex<PrecisionT> v0 = arr(i0);
            const Kokkos::complex<PrecisionT> v1 =
                (inverse) ? -arr(i1) : arr(i1);
            arr(i0) = M_SQRT1_2 * v0 +
                      M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
            arr(i1) = M_SQRT1_2 * v0 +
                      -M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyS(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
            const std::size_t num_qubits, const std::vector<size_t> &wires,
            const bool inverse = false,
            [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    const Kokkos::complex<PrecisionT> shift =
        (inverse) ? -Kokkos::complex{0.0, 1.0} : Kokkos::complex{0.0, 1.0};
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i0,
                      const std::size_t i1) { arr(i1) *= shift; });
}

template <class ExecutionSpace, class PrecisionT>
void applyT(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
            const std::size_t num_qubits, const std::vector<size_t> &wires,
            const bool inverse = false,
            [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    const Kokkos::complex<PrecisionT> shift =
        (inverse) ? conj(exp(Kokkos::complex<PrecisionT>(
                        0, static_cast<PrecisionT>(M_PI / 4))))
                  : exp(Kokkos::complex<PrecisionT>(
                        0, static_cast<PrecisionT>(M_PI / 4)));
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i0,
                      const std::size_t i1) { arr(i1) *= shift; });
}

template <class ExecutionSpace, class PrecisionT>
void applyPhaseShift(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                     const std::size_t num_qubits,
                     const std::vector<size_t> &wires,
                     const bool inverse = false,
                     const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const Kokkos::complex<PrecisionT> shift =
        (inverse) ? exp(-Kokkos::complex<PrecisionT>(0, angle))
                  : exp(Kokkos::complex<PrecisionT>(0, angle));
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i0,
                      const std::size_t i1) { arr(i1) *= shift; });
}

template <class ExecutionSpace, class PrecisionT>
void applyRX(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
             const std::size_t num_qubits, const std::vector<size_t> &wires,
             const bool inverse = false,
             const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const PrecisionT c = cos(angle * static_cast<PrecisionT>(0.5));
    const PrecisionT s = (inverse) ? sin(angle * static_cast<PrecisionT>(0.5))
                                   : sin(-angle * static_cast<PrecisionT>(0.5));
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0, const std::size_t i1) {
            const auto v0 = arr(i0);
            const auto v1 = arr(i1);
            arr(i0) = c * v0 +
                      Kokkos::complex<PrecisionT>{-imag(v1) * s, real(v1) * s};
            arr(i1) = Kokkos::complex<PrecisionT>{-imag(v0) * s, real(v0) * s} +
                      c * v1;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyRY(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
             const std::size_t num_qubits, const std::vector<size_t> &wires,
             const bool inverse = false,
             const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const PrecisionT c = cos(angle * static_cast<PrecisionT>(0.5));
    const PrecisionT s = (inverse) ? -sin(angle * static_cast<PrecisionT>(0.5))
                                   : sin(angle * static_cast<PrecisionT>(0.5));
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0, const std::size_t i1) {
            const auto v0 = arr(i0);
            const auto v1 = arr(i1);
            arr(i0) = Kokkos::complex<PrecisionT>{c * real(v0) - s * real(v1),
                                                  c * imag(v0) - s * imag(v1)};
            arr(i1) = Kokkos::complex<PrecisionT>{s * real(v0) + c * real(v1),
                                                  s * imag(v0) + c * imag(v1)};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
             const std::size_t num_qubits, const std::vector<size_t> &wires,
             const bool inverse = false,
             const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const PrecisionT cos_angle = cos(angle * static_cast<PrecisionT>(0.5));
    const PrecisionT sin_angle = sin(angle * static_cast<PrecisionT>(0.5));
    const Kokkos::complex<PrecisionT> shift_0{
        cos_angle, (inverse) ? sin_angle : -sin_angle};
    const Kokkos::complex<PrecisionT> shift_1 = Kokkos::conj(shift_0);
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0, const std::size_t i1) {
            arr(i0) *= shift_0;
            arr(i1) *= shift_1;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyRot(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              const std::size_t num_qubits, const std::vector<size_t> &wires,
              const bool inverse = false,
              const std::vector<PrecisionT> &params = {}) {
    const PrecisionT phi = (inverse) ? -params[2] : params[0];
    const PrecisionT theta = (inverse) ? -params[1] : params[1];
    const PrecisionT omega = (inverse) ? -params[0] : params[2];
    const auto mat = Pennylane::Gates::getRot<Kokkos::complex, PrecisionT>(
        phi, theta, omega);
    const Kokkos::complex<PrecisionT> mat_0b00 = mat[0b00];
    const Kokkos::complex<PrecisionT> mat_0b01 = mat[0b01];
    const Kokkos::complex<PrecisionT> mat_0b10 = mat[0b10];
    const Kokkos::complex<PrecisionT> mat_0b11 = mat[0b11];
    applyNC1Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0, const std::size_t i1) {
            const Kokkos::complex<PrecisionT> v0 = arr(i0);
            const Kokkos::complex<PrecisionT> v1 = arr(i1);
            arr(i0) = mat_0b00 * v0 + mat_0b01 * v1;
            arr(i1) = mat_0b10 * v0 + mat_0b11 * v1;
        });
}

template <class PrecisionT, class FuncT> class applyNC2Functor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    const FuncT core_function;
    const std::size_t rev_wire0;
    const std::size_t rev_wire1;
    const std::size_t rev_wire0_shift;
    const std::size_t rev_wire1_shift;
    const std::size_t rev_wire_min;
    const std::size_t rev_wire_max;
    const std::size_t parity_low;
    const std::size_t parity_high;
    const std::size_t parity_middle;

  public:
    template <class ExecutionSpace>
    applyNC2Functor([[maybe_unused]] ExecutionSpace exec,
                    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                    std::size_t num_qubits, const std::vector<size_t> &wires,
                    FuncT core_function_)
        : arr(arr_), core_function(core_function_),
          rev_wire0(num_qubits - wires[1] - 1),
          rev_wire1(num_qubits - wires[0] - 1),
          rev_wire0_shift(static_cast<size_t>(1U) << rev_wire0),
          rev_wire1_shift(static_cast<size_t>(1U) << rev_wire1),
          rev_wire_min(std::min(rev_wire0, rev_wire1)),
          rev_wire_max(std::max(rev_wire0, rev_wire1)),
          parity_low(fillTrailingOnes(rev_wire_min)),
          parity_high(fillLeadingOnes(rev_wire_max + 1)),
          parity_middle(fillLeadingOnes(rev_wire_min + 1) &
                        fillTrailingOnes(rev_wire_max)) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(0, exp2(num_qubits - 2)),
            *this);
    }
    KOKKOS_FUNCTION void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;
        core_function(arr, i00, i01, i10, i11);
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyCNOT(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               const std::size_t num_qubits, const std::vector<size_t> &wires,
               [[maybe_unused]] const bool inverse = false,
               [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      [[maybe_unused]] const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            Kokkos::Experimental::swap(arr(i10), arr(i11));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyCY(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
             const std::size_t num_qubits, const std::vector<size_t> &wires,
             [[maybe_unused]] const bool inverse = false,
             [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      [[maybe_unused]] const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            Kokkos::complex<PrecisionT> v10 = arr(i10);
            arr(i10) =
                Kokkos::complex<PrecisionT>{imag(arr(i11)), -real(arr(i11))};
            arr(i11) = Kokkos::complex<PrecisionT>{-imag(v10), real(v10)};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyCZ(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
             const std::size_t num_qubits, const std::vector<size_t> &wires,
             [[maybe_unused]] const bool inverse = false,
             [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      [[maybe_unused]] const std::size_t i01,
                      [[maybe_unused]] const std::size_t i10,
                      const std::size_t i11) { arr(i11) *= -1; });
}

template <class ExecutionSpace, class PrecisionT>
void applySWAP(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               const std::size_t num_qubits, const std::vector<size_t> &wires,
               [[maybe_unused]] const bool inverse = false,
               [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      const std::size_t i01, const std::size_t i10,
                      [[maybe_unused]] const std::size_t i11) {
            swap(arr(i10), arr(i01));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyControlledPhaseShift(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    const bool inverse = false, const std::vector<PrecisionT> &params = {}) {

    const PrecisionT &angle = params[0];
    const Kokkos::complex<PrecisionT> s =
        (inverse) ? exp(-Kokkos::complex<PrecisionT>(0, angle))
                  : exp(Kokkos::complex<PrecisionT>(0, angle));
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      [[maybe_unused]] const std::size_t i01,
                      [[maybe_unused]] const std::size_t i10,
                      const std::size_t i11) { arr(i11) *= s; });
}

template <class ExecutionSpace, class PrecisionT>
void applyCRX(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              const std::size_t num_qubits, const std::vector<size_t> &wires,
              const bool inverse = false,
              const std::vector<PrecisionT> &params = {}) {

    const PrecisionT &angle = params[0];
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT js =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      [[maybe_unused]] const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            const Kokkos::complex<PrecisionT> v10 = arr(i10);
            const Kokkos::complex<PrecisionT> v11 = arr(i11);
            arr(i10) = Kokkos::complex<PrecisionT>{
                c * real(v10) + js * imag(v11), c * imag(v10) - js * real(v11)};
            arr(i11) = Kokkos::complex<PrecisionT>{
                c * real(v11) + js * imag(v10), c * imag(v11) - js * real(v10)};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyCRY(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              const std::size_t num_qubits, const std::vector<size_t> &wires,
              const bool inverse = false,
              const std::vector<PrecisionT> &params = {}) {

    const PrecisionT &angle = params[0];
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      [[maybe_unused]] const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            const Kokkos::complex<PrecisionT> v10 = arr(i10);
            const Kokkos::complex<PrecisionT> v11 = arr(i11);
            arr(i10) = c * v10 - s * v11;
            arr(i11) = s * v10 + c * v11;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyCRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              const std::size_t num_qubits, const std::vector<size_t> &wires,
              const bool inverse = false,
              const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const PrecisionT cos_angle = cos(angle * static_cast<PrecisionT>(0.5));
    const PrecisionT sin_angle = sin(angle * static_cast<PrecisionT>(0.5));
    const Kokkos::complex<PrecisionT> shift_0{
        cos_angle, (inverse) ? sin_angle : -sin_angle};
    const Kokkos::complex<PrecisionT> shift_1 = Kokkos::conj(shift_0);
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      [[maybe_unused]] const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            arr(i10) *= shift_0;
            arr(i11) *= shift_1;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyCRot(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               const std::size_t num_qubits, const std::vector<size_t> &wires,
               const bool inverse = false,
               const std::vector<PrecisionT> &params = {}) {
    const PrecisionT phi = (inverse) ? -params[2] : params[0];
    const PrecisionT theta = (inverse) ? -params[1] : params[1];
    const PrecisionT omega = (inverse) ? -params[0] : params[2];
    const auto mat = Pennylane::Gates::getRot<Kokkos::complex, PrecisionT>(
        phi, theta, omega);
    const Kokkos::complex<PrecisionT> mat_0b00 = mat[0b00];
    const Kokkos::complex<PrecisionT> mat_0b01 = mat[0b01];
    const Kokkos::complex<PrecisionT> mat_0b10 = mat[0b10];
    const Kokkos::complex<PrecisionT> mat_0b11 = mat[0b11];
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      [[maybe_unused]] const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            const Kokkos::complex<PrecisionT> v0 = arr(i10);
            const Kokkos::complex<PrecisionT> v1 = arr(i11);
            arr(i10) = mat_0b00 * v0 + mat_0b01 * v1;
            arr(i11) = mat_0b10 * v0 + mat_0b11 * v1;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyIsingXX(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  const std::size_t num_qubits,
                  const std::vector<size_t> &wires, const bool inverse = false,
                  const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            const Kokkos::complex<PrecisionT> v00 = arr(i00);
            const Kokkos::complex<PrecisionT> v01 = arr(i01);
            const Kokkos::complex<PrecisionT> v10 = arr(i10);
            const Kokkos::complex<PrecisionT> v11 = arr(i11);
            arr(i00) =
                Kokkos::complex<PrecisionT>{cr * real(v00) + sj * imag(v11),
                                            cr * imag(v00) - sj * real(v11)};
            arr(i01) =
                Kokkos::complex<PrecisionT>{cr * real(v01) + sj * imag(v10),
                                            cr * imag(v01) - sj * real(v10)};
            arr(i10) =
                Kokkos::complex<PrecisionT>{cr * real(v10) + sj * imag(v01),
                                            cr * imag(v10) - sj * real(v01)};
            arr(i11) =
                Kokkos::complex<PrecisionT>{cr * real(v11) + sj * imag(v00),
                                            cr * imag(v11) - sj * real(v00)};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyIsingXY(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  const std::size_t num_qubits,
                  const std::vector<size_t> &wires, const bool inverse = false,
                  const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            const Kokkos::complex<PrecisionT> v00 = arr(i00);
            const Kokkos::complex<PrecisionT> v01 = arr(i01);
            const Kokkos::complex<PrecisionT> v10 = arr(i10);
            const Kokkos::complex<PrecisionT> v11 = arr(i11);
            arr(i00) = Kokkos::complex<PrecisionT>{real(v00), imag(v00)};
            arr(i01) =
                Kokkos::complex<PrecisionT>{cr * real(v01) - sj * imag(v10),
                                            cr * imag(v01) + sj * real(v10)};
            arr(i10) =
                Kokkos::complex<PrecisionT>{cr * real(v10) - sj * imag(v01),
                                            cr * imag(v10) + sj * real(v01)};
            arr(i11) = Kokkos::complex<PrecisionT>{real(v11), imag(v11)};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyIsingYY(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  const std::size_t num_qubits,
                  const std::vector<size_t> &wires, const bool inverse = false,
                  const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            const Kokkos::complex<PrecisionT> v00 = arr(i00);
            const Kokkos::complex<PrecisionT> v01 = arr(i01);
            const Kokkos::complex<PrecisionT> v10 = arr(i10);
            const Kokkos::complex<PrecisionT> v11 = arr(i11);
            arr(i00) =
                Kokkos::complex<PrecisionT>{cr * real(v00) - sj * imag(v11),
                                            cr * imag(v00) + sj * real(v11)};
            arr(i01) =
                Kokkos::complex<PrecisionT>{cr * real(v01) + sj * imag(v10),
                                            cr * imag(v01) - sj * real(v10)};
            arr(i10) =
                Kokkos::complex<PrecisionT>{cr * real(v10) + sj * imag(v01),
                                            cr * imag(v10) - sj * real(v01)};
            arr(i11) =
                Kokkos::complex<PrecisionT>{cr * real(v11) - sj * imag(v00),
                                            cr * imag(v11) + sj * real(v00)};
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyIsingZZ(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  const std::size_t num_qubits,
                  const std::vector<size_t> &wires, const bool inverse = false,
                  const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const Kokkos::complex<PrecisionT> shift_0 = Kokkos::complex<PrecisionT>{
        std::cos(angle / 2),
        (inverse) ? std::sin(angle / 2) : -std::sin(angle / 2)};
    const Kokkos::complex<PrecisionT> shift_1 = conj(shift_0);
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            arr(i00) *= shift_0;
            arr(i01) *= shift_1;
            arr(i10) *= shift_1;
            arr(i11) *= shift_0;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applySingleExcitation(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                           const std::size_t num_qubits,
                           const std::vector<size_t> &wires,
                           const bool inverse = false,
                           const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      const std::size_t i01, const std::size_t i10,
                      [[maybe_unused]] const std::size_t i11) {
            const Kokkos::complex<PrecisionT> v01 = arr(i01);
            const Kokkos::complex<PrecisionT> v10 = arr(i10);
            arr(i01) = cr * v01 - sj * v10;
            arr(i10) = sj * v01 + cr * v10;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applySingleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    const bool inverse = false, const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const Kokkos::complex<PrecisionT> e =
        inverse ? exp(Kokkos::complex<PrecisionT>(0, angle / 2))
                : exp(Kokkos::complex<PrecisionT>(0, -angle / 2));
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i00, const std::size_t i01,
                      const std::size_t i10, const std::size_t i11) {
            const Kokkos::complex<PrecisionT> v01 = arr(i01);
            const Kokkos::complex<PrecisionT> v10 = arr(i10);
            arr(i00) *= e;
            arr(i01) = cr * v01 - sj * v10;
            arr(i10) = sj * v01 + cr * v10;
            arr(i11) *= e;
        });
}

template <class ExecutionSpace, class PrecisionT>
void applySingleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
    const std::size_t num_qubits, const std::vector<size_t> &wires,
    const bool inverse = false, const std::vector<PrecisionT> &params = {}) {
    const PrecisionT &angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const Kokkos::complex<PrecisionT> e =
        inverse ? exp(Kokkos::complex<PrecisionT>(0, -angle / 2))
                : exp(Kokkos::complex<PrecisionT>(0, angle / 2));
    applyNC2Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i00,
                      const std::size_t i01, const std::size_t i10,
                      [[maybe_unused]] const std::size_t i11) {
            const Kokkos::complex<PrecisionT> v01 = arr(i01);
            const Kokkos::complex<PrecisionT> v10 = arr(i10);
            arr(i00) *= e;
            arr(i01) = cr * v01 - sj * v10;
            arr(i10) = sj * v01 + cr * v10;
            arr(i11) *= e;
        });
}

template <typename T>
T get_wire_mid(const T wire0, const T wire1, const T wire2) {
    const T wire_min = std::min({wire0, wire1, wire2});
    return (wire_min == wire0) ? std::min(wire1, wire2)
                               : ((wire_min == wire1) ? std::min(wire0, wire2)
                                                      : std::min(wire0, wire1));
}

template <class PrecisionT, class FuncT> class applyNC3Functor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    const FuncT core_function;
    const std::size_t rev_wire0;
    const std::size_t rev_wire1;
    const std::size_t rev_wire2;
    const std::size_t rev_wire0_shift;
    const std::size_t rev_wire1_shift;
    const std::size_t rev_wire2_shift;
    const std::size_t rev_wire_min;
    const std::size_t rev_wire_max;
    const std::size_t rev_wire_mid;
    const std::size_t parity_low;
    const std::size_t parity_high;
    const std::size_t parity_hmiddle;
    const std::size_t parity_lmiddle;

  public:
    template <class ExecutionSpace>
    applyNC3Functor([[maybe_unused]] ExecutionSpace exec,
                    Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                    std::size_t num_qubits, const std::vector<size_t> &wires,
                    FuncT core_function_)
        : arr(arr_), core_function(core_function_),
          rev_wire0(num_qubits - wires[2] - 1),
          rev_wire1(num_qubits - wires[1] - 1),
          rev_wire2(num_qubits - wires[0] - 1),
          rev_wire0_shift(static_cast<size_t>(1U) << rev_wire0),
          rev_wire1_shift(static_cast<size_t>(1U) << rev_wire1),
          rev_wire2_shift(static_cast<size_t>(1U) << rev_wire2),
          rev_wire_min(std::min({rev_wire0, rev_wire1, rev_wire2})),
          rev_wire_max(std::max({rev_wire0, rev_wire1, rev_wire2})),
          rev_wire_mid(get_wire_mid(rev_wire0, rev_wire1, rev_wire2)),
          parity_low(fillTrailingOnes(rev_wire_min)),
          parity_high(fillLeadingOnes(rev_wire_max + 1)),
          parity_hmiddle(fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_mid)),
          parity_lmiddle(fillLeadingOnes(rev_wire_mid + 1) &
                         fillTrailingOnes(rev_wire_max)) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(0, exp2(num_qubits - 3)),
            *this);
    }
    KOKKOS_FUNCTION void operator()(const std::size_t k) const {
        const std::size_t i000 =
            ((k << 3U) & parity_high) | ((k << 2U) & parity_hmiddle) |
            ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const std::size_t i001 = i000 | rev_wire0_shift;
        const std::size_t i010 = i000 | rev_wire1_shift;
        const std::size_t i011 = i000 | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i100 = i000 | rev_wire2_shift;
        const std::size_t i101 = i000 | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;
        const std::size_t i111 =
            i000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        core_function(arr, i000, i001, i010, i011, i100, i101, i110, i111);
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyCSWAP(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                const std::size_t num_qubits, const std::vector<size_t> &wires,
                [[maybe_unused]] const bool inverse = false,
                [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC3Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i000,
                      [[maybe_unused]] const std::size_t i001,
                      [[maybe_unused]] const std::size_t i010,
                      [[maybe_unused]] const std::size_t i011,
                      [[maybe_unused]] const std::size_t i100,
                      const std::size_t i101, const std::size_t i110,
                      [[maybe_unused]] const std::size_t i111) {
            swap(arr(i101), arr(i110));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyToffoli(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  const std::size_t num_qubits,
                  const std::vector<size_t> &wires,
                  [[maybe_unused]] const bool inverse = false,
                  [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC3Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      [[maybe_unused]] const std::size_t i000,
                      [[maybe_unused]] const std::size_t i001,
                      [[maybe_unused]] const std::size_t i010,
                      [[maybe_unused]] const std::size_t i011,
                      [[maybe_unused]] const std::size_t i100,
                      [[maybe_unused]] const std::size_t i101,
                      const std::size_t i110,
                      const std::size_t i111) { swap(arr(i111), arr(i110)); });
}

// template <class PrecisionT, class FuncT> class applyNC4Functor {
//     Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
//     const FuncT core_function;
//     const std::size_t rev_wire0;
//     const std::size_t rev_wire1;
//     const std::size_t rev_wire2;
//     const std::size_t rev_wire3;
//     const std::size_t rev_wire0_shift;
//     const std::size_t rev_wire1_shift;
//     const std::size_t rev_wire2_shift;
//     const std::size_t rev_wire3_shift;
//     const std::size_t rev_wire_min;
//     const std::size_t rev_wire_max;
//     const std::size_t rev_wire_min_mid;
//     const std::size_t rev_wire_max_mid;
//     const std::size_t parity_low;
//     const std::size_t parity_high;
//     const std::size_t parity_lmiddle;
//     const std::size_t parity_hmiddle;
//     const std::size_t parity_middle;

//   public:
//     template <class ExecutionSpace>
//     applyNC4Functor([[maybe_unused]] ExecutionSpace exec,
//                     Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
//                     std::size_t num_qubits, const std::vector<size_t> &wires,
//                     FuncT core_function_)
//         : arr(arr_), core_function(core_function_),
//           rev_wire0(num_qubits - wires[2] - 1),
//           rev_wire1(num_qubits - wires[1] - 1),
//           rev_wire2(num_qubits - wires[0] - 1),
//           rev_wire3(num_qubits - wires[0] - 1),
//           rev_wire0_shift(static_cast<size_t>(1U) << rev_wire0),
//           rev_wire1_shift(static_cast<size_t>(1U) << rev_wire1),
//           rev_wire2_shift(static_cast<size_t>(1U) << rev_wire2),
//           rev_wire3_shift(static_cast<size_t>(1U) << rev_wire2),
//           rev_wire_min(std::min({rev_wire0, rev_wire1, rev_wire2,
//           rev_wire3})), rev_wire_max(std::max({rev_wire0, rev_wire1,
//           rev_wire2, rev_wire3})), rev_wire_min_mid((rev_wire_min ==
//           rev_wire0)
//                            ? std::min(rev_wire1, rev_wire2)
//                            : ((rev_wire_min == rev_wire1)
//                                   ? std::min(rev_wire0, rev_wire2)
//                                   : std::min(rev_wire0, rev_wire1))),
//           parity_low(fillTrailingOnes(rev_wire_min)),
//           parity_high(fillLeadingOnes(rev_wire_max + 1)),
//           parity_hmiddle(fillLeadingOnes(rev_wire_min + 1) &
//                          fillTrailingOnes(rev_wire_mid)),
//           parity_lmiddle(fillLeadingOnes(rev_wire_mid + 1) &
//                          fillTrailingOnes(rev_wire_max)) {
//         Kokkos::parallel_for(
//             Kokkos::RangePolicy<ExecutionSpace>(0, exp2(num_qubits - 4)),
//             *this);
//     }
//     KOKKOS_FUNCTION void operator()(const std::size_t k) const {
//         const std::size_t i000 =
//             ((k << 3U) & parity_high) | ((k << 2U) & parity_hmiddle) |
//             ((k << 1U) & parity_lmiddle) | (k & parity_low);
//         const std::size_t i001 = i000 | rev_wire0_shift;
//         const std::size_t i010 = i000 | rev_wire1_shift;
//         const std::size_t i011 = i000 | rev_wire1_shift | rev_wire0_shift;
//         const std::size_t i100 = i000 | rev_wire2_shift;
//         const std::size_t i101 = i000 | rev_wire2_shift | rev_wire0_shift;
//         const std::size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;
//         const std::size_t i111 =
//             i000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
//         core_function(arr, i000, i001, i010, i011, i100, i101, i110, i111);
//     }
// };

template <class PrecisionT, bool inverse = false> struct hadamardFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    hadamardFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                    std::size_t num_qubits, const std::vector<size_t> &wires,
                    [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        if constexpr (inverse) {
            const std::size_t i0 =
                ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const std::size_t i1 = i0 | rev_wire_shift;
            const Kokkos::complex<PrecisionT> v0 = arr(i0);
            const Kokkos::complex<PrecisionT> v1 = arr(i1);
            arr(i0) = M_SQRT1_2 * v0 +
                      M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
            arr(i1) = M_SQRT1_2 * v0 +
                      (-M_SQRT1_2) * v1; // NOLINT(readability-magic-numbers)
                                         // }
        } else {
            const std::size_t i0 =
                ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const std::size_t i1 = i0 | rev_wire_shift;
            const Kokkos::complex<PrecisionT> v0 = arr(i0);
            const Kokkos::complex<PrecisionT> v1 = arr(i1);
            arr(i0) = M_SQRT1_2 * v0 +
                      M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
            arr(i1) = M_SQRT1_2 * v0 +
                      -M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
                                       // }
        }
    }
};

template <class PrecisionT, bool inverse = false> struct pauliXFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    pauliXFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  std::size_t num_qubits, const std::vector<size_t> &wires,
                  [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        swap(arr(i0), arr(i1));
    }
};

template <class PrecisionT, bool inverse = false> struct pauliYFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    pauliYFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  std::size_t num_qubits, const std::vector<size_t> &wires,
                  [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr(i0);
        const auto v1 = arr(i1);
        arr(i0) = Kokkos::complex<PrecisionT>{imag(v1), -real(v1)};
        arr(i1) = Kokkos::complex<PrecisionT>{-imag(v0), real(v0)};
    }
};

template <class PrecisionT, bool inverse = false> struct pauliZFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    pauliZFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  std::size_t num_qubits, const std::vector<size_t> &wires,
                  [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr(i1) *= -1;
    }
};

template <class PrecisionT, bool inverse = false> struct sFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;
    Kokkos::complex<PrecisionT> shift;

    sFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
             std::size_t num_qubits, const std::vector<size_t> &wires,
             [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        shift =
            (inverse) ? -Kokkos::complex{0.0, 1.0} : Kokkos::complex{0.0, 1.0};
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr(i1) *= shift;
    }
};

template <class PrecisionT, bool inverse = false> struct tFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;
    Kokkos::complex<PrecisionT> shift;

    tFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
             std::size_t num_qubits, const std::vector<size_t> &wires,
             [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        shift = (inverse) ? conj(exp(Kokkos::complex<PrecisionT>(
                                0, static_cast<PrecisionT>(M_PI / 4))))
                          : exp(Kokkos::complex<PrecisionT>(
                                0, static_cast<PrecisionT>(M_PI / 4)));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr(i1) *= shift;
    }
};

template <class PrecisionT, bool inverse = false> struct cnotFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    cnotFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                std::size_t num_qubits, const std::vector<size_t> &wires,
                [[maybe_unused]] const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;

        swap(arr(i10), arr(i11));
    }
};

template <class PrecisionT, bool inverse = false> struct cyFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    cyFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              std::size_t num_qubits, const std::vector<size_t> &wires,
              [[maybe_unused]] const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;
        Kokkos::complex<PrecisionT> v10 = arr(i10);
        arr(i10) = Kokkos::complex<PrecisionT>{imag(arr(i11)), -real(arr(i11))};
        arr(i11) = Kokkos::complex<PrecisionT>{-imag(v10), real(v10)};
    }
};

template <class PrecisionT, bool inverse = false> struct czFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    czFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              std::size_t num_qubits, const std::vector<size_t> &wires,
              [[maybe_unused]] const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;
        arr(i11) *= -1;
    }
};

template <class PrecisionT, bool inverse = false> struct swapFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    swapFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                std::size_t num_qubits, const std::vector<size_t> &wires,
                [[maybe_unused]] const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        swap(arr(i10), arr(i01));
    }
};

template <class PrecisionT, bool inverse = false> struct cSWAPFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;
    Kokkos::complex<PrecisionT> shifts_2;

    cSWAPFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                 std::size_t num_qubits, const std::vector<size_t> &wires,
                 [[maybe_unused]] const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[2] - 1;
        rev_wire1 = num_qubits - wires[1] - 1;
        rev_wire2 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        if (rev_wire2 < rev_wire_min) {
            rev_wire_mid = rev_wire_min;
            rev_wire_min = rev_wire2;
        } else if (rev_wire2 > rev_wire_max) {
            rev_wire_mid = rev_wire_max;
            rev_wire_max = rev_wire2;
        } else {
            rev_wire_mid = rev_wire2;
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_mid);
        parity_hmiddle =
            fillLeadingOnes(rev_wire_mid + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i000 =
            ((k << 3U) & parity_high) | ((k << 2U) & parity_hmiddle) |
            ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const std::size_t i101 = i000 | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;

        swap(arr(i101), arr(i110));
    }
};

template <class PrecisionT, bool inverse = false> struct toffoliFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;
    Kokkos::complex<PrecisionT> shifts_2;

    toffoliFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   std::size_t num_qubits, const std::vector<size_t> &wires,
                   [[maybe_unused]] const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[2] - 1;
        rev_wire1 = num_qubits - wires[1] - 1;
        rev_wire2 = num_qubits - wires[0] - 1; // Control qubit
        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);
        if (rev_wire2 < rev_wire_min) {
            rev_wire_mid = rev_wire_min;
            rev_wire_min = rev_wire2;
        } else if (rev_wire2 > rev_wire_max) {
            rev_wire_mid = rev_wire_max;
            rev_wire_max = rev_wire2;
        } else {
            rev_wire_mid = rev_wire2;
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_mid);
        parity_hmiddle =
            fillLeadingOnes(rev_wire_mid + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i000 =
            ((k << 3U) & parity_high) | ((k << 2U) & parity_hmiddle) |
            ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const std::size_t i111 =
            i000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;

        swap(arr(i111), arr(i110));
    }
};

} // namespace Pennylane::LightningKokkos::Functors