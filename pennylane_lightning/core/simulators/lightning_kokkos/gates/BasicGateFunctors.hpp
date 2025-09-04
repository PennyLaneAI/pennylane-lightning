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

#include "BitUtil.hpp"
#include "GateOperation.hpp"
#include "Gates.hpp"
#include "Util.hpp" // exp2, INVSQRT2
#include "UtilKokkos.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using Kokkos::kokkos_swap;
using Pennylane::Gates::GateOperation;
using Pennylane::LightningKokkos::Util::controlBitPatterns;
using Pennylane::LightningKokkos::Util::generateBitPatterns;
using Pennylane::LightningKokkos::Util::parity_2_offset;
using Pennylane::LightningKokkos::Util::reverseWires;
using Pennylane::LightningKokkos::Util::vector2view;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Functors {
template <class PrecisionT, class FuncT> class applyNCNFunctor {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;
    using MemberType = Kokkos::TeamPolicy<>::member_type;

    KokkosComplexVector arr;
    const FuncT core_function;
    KokkosIntVector indices;
    KokkosIntVector parity;
    KokkosIntVector rev_wires;
    KokkosIntVector rev_wire_shifts;
    std::size_t dim;

  public:
    template <class ExecutionSpace>
    applyNCNFunctor([[maybe_unused]] ExecutionSpace exec,
                    KokkosComplexVector arr_, std::size_t num_qubits,
                    const std::vector<std::size_t> &controlled_wires,
                    const std::vector<bool> &controlled_values,
                    const std::vector<std::size_t> &wires, FuncT core_function_)
        : arr(arr_), core_function(core_function_) {
        const std::size_t two2N = Pennylane::Util::exp2(
            num_qubits - wires.size() - controlled_wires.size());
        dim = Pennylane::Util::exp2(wires.size());
        const auto &[parity_, rev_wires_] =
            reverseWires(num_qubits, wires, controlled_wires);
        parity = parity_;
        std::vector<std::size_t> indices_ =
            generateBitPatterns(wires, num_qubits);
        controlBitPatterns(indices_, num_qubits, controlled_wires,
                           controlled_values);
        indices = vector2view(indices_);
        Kokkos::parallel_for(Kokkos::TeamPolicy(two2N, Kokkos::AUTO, dim),
                             *this);
    }
    // TODO: Runtime selection for copying indices to scratch level 0/shmem
    KOKKOS_FUNCTION void operator()(const MemberType &teamMember) const {
        const std::size_t k = teamMember.league_rank();
        const std::size_t offset = Util::parity_2_offset(parity, k);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, dim),
            [&](std::size_t i) { core_function(arr, i, indices, offset); });
    }
};

template <class PrecisionT, class FuncT, bool has_controls>
class applyNC1Functor {};

template <class PrecisionT, class FuncT>
class applyNC1Functor<PrecisionT, FuncT, true> {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    const FuncT core_function;
    KokkosIntVector indices;
    KokkosIntVector parity;
    KokkosIntVector rev_wires;
    KokkosIntVector rev_wire_shifts;

  public:
    template <class ExecutionSpace>
    applyNC1Functor([[maybe_unused]] ExecutionSpace exec,
                    KokkosComplexVector arr_, std::size_t num_qubits,
                    const std::vector<std::size_t> &controlled_wires,
                    const std::vector<bool> &controlled_values,
                    const std::vector<std::size_t> &wires, FuncT core_function_)
        : arr(arr_), core_function(core_function_) {
        const auto &[parity_, rev_wires_] =
            reverseWires(num_qubits, wires, controlled_wires);
        parity = parity_;
        std::vector<std::size_t> indices_ =
            generateBitPatterns(wires, num_qubits);
        controlBitPatterns(indices_, num_qubits, controlled_wires,
                           controlled_values);
        indices = vector2view(indices_);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(
                0, Pennylane::Util::exp2(num_qubits - controlled_wires.size() -
                                         wires.size())),
            *this);
    }
    KOKKOS_FUNCTION void operator()(std::size_t k) const {
        const std::size_t offset = Util::parity_2_offset(parity, k);
        std::size_t i0 = indices(0B00);
        std::size_t i1 = indices(0B01);

        core_function(arr, i0 + offset, i1 + offset);
    }
};

template <class PrecisionT, class FuncT>
class applyNC1Functor<PrecisionT, FuncT, false> {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;

    KokkosComplexVector arr;
    const FuncT core_function;
    const std::size_t rev_wire;
    const std::size_t rev_wire_shift;
    const std::size_t wire_parity;
    const std::size_t wire_parity_inv;

  public:
    template <class ExecutionSpace>
    applyNC1Functor([[maybe_unused]] ExecutionSpace exec,
                    KokkosComplexVector arr_, std::size_t num_qubits,
                    const std::vector<std::size_t> &wires, FuncT core_function_)
        : arr(arr_), core_function(core_function_),
          rev_wire(num_qubits ? num_qubits - wires[0] - 1 : 0),
          rev_wire_shift((static_cast<std::size_t>(1U) << rev_wire)),
          wire_parity(fillTrailingOnes(rev_wire)),
          wire_parity_inv(fillLeadingOnes(rev_wire + 1)) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(
                0, num_qubits ? Pennylane::Util::exp2(num_qubits - 1) : 1),
            *this);
    }
    KOKKOS_FUNCTION void operator()(std::size_t k) const {
        std::size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        std::size_t i1 = i0 | rev_wire_shift;
        core_function(arr, i0, i1);
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyNCPauliX(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        kokkos_swap(arr(i0), arr(i1));
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyPauliX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 std::size_t num_qubits, const std::vector<std::size_t> &wires,
                 bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNCPauliX<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                              inverse);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCPauliY(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        const auto v0 = arr(i0);
        const auto v1 = arr(i1);
        arr(i0) = Kokkos::complex<PrecisionT>{imag(v1), -real(v1)};
        arr(i1) = Kokkos::complex<PrecisionT>{-imag(v0), real(v0)};
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyPauliY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 std::size_t num_qubits, const std::vector<std::size_t> &wires,
                 bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNCPauliY<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                              inverse);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCPauliZ(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        [[maybe_unused]] const auto i0_ = i0;
        // Note: this is to avoid Clang complain [[maybe_unused]]
        // attribute for lambda function arguments
        arr(i1) *= -1.0;
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyPauliZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 std::size_t num_qubits, const std::vector<std::size_t> &wires,
                 bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNCPauliZ<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                              inverse);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCHadamard(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        const Kokkos::complex<PrecisionT> v0 = arr(i0);
        const Kokkos::complex<PrecisionT> v1 = arr(i1);
        arr(i0) = M_SQRT1_2 * (v0 + v1); // M_SQRT1_2 * v0 + M_SQRT1_2 * v1
        arr(i1) = M_SQRT1_2 * (v0 - v1); // M_SQRT1_2 * v0 - M_SQRT1_2 * v1
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyHadamard(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &wires, bool inverse = false,
    [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNCHadamard<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                                inverse);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCS(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
              std::size_t num_qubits,
              const std::vector<std::size_t> &controlled_wires,
              const std::vector<bool> &controlled_values,
              const std::vector<std::size_t> &wires, bool inverse = false,
              [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    const Kokkos::complex<PrecisionT> shift =
        (inverse) ? Kokkos::complex<PrecisionT>{0.0, -1.0}
                  : Kokkos::complex<PrecisionT>{0.0, 1.0};
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        [[maybe_unused]] const auto i0_ = i0;
        arr(i1) *= shift;
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyS(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
            std::size_t num_qubits, const std::vector<std::size_t> &wires,
            bool inverse = false,
            [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNCS<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                         inverse);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCSX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
               std::size_t num_qubits,
               const std::vector<std::size_t> &controlled_wires,
               const std::vector<bool> &controlled_values,
               const std::vector<std::size_t> &wires, bool inverse = false,
               [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    constexpr PrecisionT half = 0.5;
    const Kokkos::complex<PrecisionT> z0{half, (inverse) ? -half : half};
    const Kokkos::complex<PrecisionT> z1 = Kokkos::conj(z0);

    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        const Kokkos::complex<PrecisionT> v0 = arr(i0);
        const Kokkos::complex<PrecisionT> v1 = arr(i1);

        arr[i0] = z0 * v0 + z1 * v1;
        arr[i1] = z1 * v0 + z0 * v1;
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applySX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
             std::size_t num_qubits, const std::vector<std::size_t> &wires,
             bool inverse = false,
             [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNCSX<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                          inverse);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCT(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
              std::size_t num_qubits,
              const std::vector<std::size_t> &controlled_wires,
              const std::vector<bool> &controlled_values,
              const std::vector<std::size_t> &wires, bool inverse = false,
              [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    const Kokkos::complex<PrecisionT> shift =
        (inverse) ? Kokkos::conj(Kokkos::exp(Kokkos::complex<PrecisionT>(
                        0, static_cast<PrecisionT>(M_PI_4))))
                  : Kokkos::exp(Kokkos::complex<PrecisionT>(
                        0, static_cast<PrecisionT>(M_PI_4)));
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        [[maybe_unused]] const auto i0_ = i0;
        arr(i1) *= shift;
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyT(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
            std::size_t num_qubits, const std::vector<std::size_t> &wires,
            bool inverse = false,
            [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNCT<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                         inverse);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCPhaseShift(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                       std::size_t num_qubits,
                       const std::vector<std::size_t> &controlled_wires,
                       const std::vector<bool> &controlled_values,
                       const std::vector<std::size_t> &wires,
                       bool inverse = false,
                       const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const Kokkos::complex<PrecisionT> shift =
        (inverse) ? Kokkos::exp(-Kokkos::complex<PrecisionT>(0, angle))
                  : Kokkos::exp(Kokkos::complex<PrecisionT>(0, angle));
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        [[maybe_unused]] const auto i0_ = i0;
        arr(i1) *= shift;
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyPhaseShift(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                     std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     bool inverse = false,
                     const std::vector<PrecisionT> &params = {}) {
    applyNCPhaseShift<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {},
                                                  wires, inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCRX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
               std::size_t num_qubits,
               const std::vector<std::size_t> &controlled_wires,
               const std::vector<bool> &controlled_values,
               const std::vector<std::size_t> &wires, bool inverse = false,
               const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT c = std::cos(angle * static_cast<PrecisionT>(0.5));
    const PrecisionT s = (inverse)
                             ? std::sin(angle * static_cast<PrecisionT>(0.5))
                             : std::sin(-angle * static_cast<PrecisionT>(0.5));
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        const auto v0 = arr(i0);
        const auto v1 = arr(i1);
        arr(i0) =
            c * v0 + Kokkos::complex<PrecisionT>{-imag(v1) * s, real(v1) * s};
        arr(i1) =
            Kokkos::complex<PrecisionT>{-imag(v0) * s, real(v0) * s} + c * v1;
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyRX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
             std::size_t num_qubits, const std::vector<std::size_t> &wires,
             bool inverse = false, const std::vector<PrecisionT> &params = {}) {
    applyNCRX<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                          inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCRY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
               std::size_t num_qubits,
               const std::vector<std::size_t> &controlled_wires,
               const std::vector<bool> &controlled_values,
               const std::vector<std::size_t> &wires, bool inverse = false,
               const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT c = std::cos(angle * static_cast<PrecisionT>(0.5));
    const PrecisionT s = (inverse)
                             ? -std::sin(angle * static_cast<PrecisionT>(0.5))
                             : std::sin(angle * static_cast<PrecisionT>(0.5));
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        const auto v0 = arr(i0);
        const auto v1 = arr(i1);
        arr(i0) = Kokkos::complex<PrecisionT>{c * real(v0) - s * real(v1),
                                              c * imag(v0) - s * imag(v1)};
        arr(i1) = Kokkos::complex<PrecisionT>{s * real(v0) + c * real(v1),
                                              s * imag(v0) + c * imag(v1)};
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyRY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
             std::size_t num_qubits, const std::vector<std::size_t> &wires,
             bool inverse = false, const std::vector<PrecisionT> &params = {}) {
    applyNCRY<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                          inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
               std::size_t num_qubits,
               const std::vector<std::size_t> &controlled_wires,
               const std::vector<bool> &controlled_values,
               const std::vector<std::size_t> &wires, bool inverse = false,
               const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cos_angle = std::cos(angle * static_cast<PrecisionT>(0.5));
    const PrecisionT sin_angle = std::sin(angle * static_cast<PrecisionT>(0.5));
    const Kokkos::complex<PrecisionT> shift_0{
        cos_angle, (inverse) ? sin_angle : -sin_angle};
    const Kokkos::complex<PrecisionT> shift_1 = Kokkos::conj(shift_0);
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        arr(i0) *= shift_0;
        arr(i1) *= shift_1;
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
             std::size_t num_qubits, const std::vector<std::size_t> &wires,
             bool inverse = false, const std::vector<PrecisionT> &params = {}) {
    applyNCRZ<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                          inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCRot(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                std::size_t num_qubits,
                const std::vector<std::size_t> &controlled_wires,
                const std::vector<bool> &controlled_values,
                const std::vector<std::size_t> &wires, bool inverse = false,
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
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        const Kokkos::complex<PrecisionT> v0 = arr(i0);
        const Kokkos::complex<PrecisionT> v1 = arr(i1);
        arr(i0) = mat_0b00 * v0 + mat_0b01 * v1;
        arr(i1) = mat_0b10 * v0 + mat_0b11 * v1;
    };
    if (controlled_wires.empty()) {
        applyNC1Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC1Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyRot(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
              std::size_t num_qubits, const std::vector<std::size_t> &wires,
              bool inverse = false,
              const std::vector<PrecisionT> &params = {}) {
    applyNCRot<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                           inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGlobalPhase(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                        std::size_t num_qubits,
                        const std::vector<std::size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        [[maybe_unused]] const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
    const Kokkos::complex<PrecisionT> phase = Kokkos::exp(
        Kokkos::complex<PrecisionT>{0, (inverse) ? params[0] : -params[0]});

    std::size_t target{0U};
    if (!controlled_wires.empty()) {
        for (std::size_t i = 0; i < num_qubits; i++) {
            if (std::find(controlled_wires.begin(), controlled_wires.end(),
                          i) == controlled_wires.end()) {
                target = i;
                break;
            }
        }
    }

    if (num_qubits) [[likely]] {
        auto core_function =
            KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                          std::size_t i0, std::size_t i1) {
            arr(i1) *= phase;
            arr(i0) *= phase;
        };

        if (controlled_wires.empty()) {
            applyNC1Functor<PrecisionT, decltype(core_function), false>(
                ExecutionSpace{}, arr_, num_qubits, {target}, core_function);
        } else {
            applyNC1Functor<PrecisionT, decltype(core_function), true>(
                ExecutionSpace{}, arr_, num_qubits, controlled_wires,
                controlled_values, {target}, core_function);
        }
    } else {
        auto core_function =
            KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                          std::size_t i0, std::size_t i1) {
            [[maybe_unused]] const auto i1_ = i1;
            arr(i0) *= phase;
        };

        if (controlled_wires.empty()) {
            applyNC1Functor<PrecisionT, decltype(core_function), false>(
                ExecutionSpace{}, arr_, num_qubits, {target}, core_function);
        } else {
            applyNC1Functor<PrecisionT, decltype(core_function), true>(
                ExecutionSpace{}, arr_, num_qubits, controlled_wires,
                controlled_values, {target}, core_function);
        }
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyGlobalPhase(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                      std::size_t num_qubits,
                      [[maybe_unused]] const std::vector<std::size_t> &wires,
                      bool inverse = false,
                      const std::vector<PrecisionT> &params = {}) {
    applyNCGlobalPhase<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {},
                                                   wires, inverse, params);
}

template <class PrecisionT, class FuncT, bool has_controls>
class applyNC2Functor {};

template <class PrecisionT, class FuncT>
class applyNC2Functor<PrecisionT, FuncT, true> {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    const FuncT core_function;
    KokkosIntVector indices;
    KokkosIntVector parity;
    KokkosIntVector rev_wires;
    KokkosIntVector rev_wire_shifts;

  public:
    template <class ExecutionSpace>
    applyNC2Functor([[maybe_unused]] ExecutionSpace exec,
                    KokkosComplexVector arr_, std::size_t num_qubits,
                    const std::vector<std::size_t> &controlled_wires,
                    const std::vector<bool> &controlled_values,
                    const std::vector<std::size_t> &wires, FuncT core_function_)
        : arr(arr_), core_function(core_function_) {
        const auto &[parity_, rev_wires_] =
            reverseWires(num_qubits, wires, controlled_wires);
        parity = parity_;
        std::vector<std::size_t> indices_ =
            generateBitPatterns(wires, num_qubits);
        controlBitPatterns(indices_, num_qubits, controlled_wires,
                           controlled_values);
        indices = vector2view(indices_);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(
                0, Pennylane::Util::exp2(num_qubits - controlled_wires.size() -
                                         wires.size())),
            *this);
    }
    KOKKOS_FUNCTION void operator()(std::size_t k) const {
        const std::size_t offset = Util::parity_2_offset(parity, k);
        std::size_t i00 = indices(0B00);
        std::size_t i01 = indices(0B01);
        std::size_t i10 = indices(0B10);
        std::size_t i11 = indices(0B11);

        core_function(arr, i00 + offset, i01 + offset, i10 + offset,
                      i11 + offset);
    }
};

template <class PrecisionT, class FuncT>
class applyNC2Functor<PrecisionT, FuncT, false> {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;
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
                    KokkosComplexVector arr_, std::size_t num_qubits,
                    const std::vector<std::size_t> &wires, FuncT core_function_)
        : arr(arr_), core_function(core_function_),
          rev_wire0(num_qubits - wires[1] - 1),
          rev_wire1(num_qubits - wires[0] - 1),
          rev_wire0_shift(static_cast<std::size_t>(1U) << rev_wire0),
          rev_wire1_shift(static_cast<std::size_t>(1U) << rev_wire1),
          rev_wire_min(std::min(rev_wire0, rev_wire1)),
          rev_wire_max(std::max(rev_wire0, rev_wire1)),
          parity_low(fillTrailingOnes(rev_wire_min)),
          parity_high(fillLeadingOnes(rev_wire_max + 1)),
          parity_middle(fillLeadingOnes(rev_wire_min + 1) &
                        fillTrailingOnes(rev_wire_max)) {
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(
                                 0, Pennylane::Util::exp2(num_qubits - 2)),
                             *this);
    }
    KOKKOS_FUNCTION void operator()(std::size_t k) const {
        std::size_t i00 = ((k << 2U) & parity_high) |
                          ((k << 1U) & parity_middle) | (k & parity_low);
        std::size_t i01 = i00 | rev_wire0_shift;
        std::size_t i10 = i00 | rev_wire1_shift;
        std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;
        core_function(arr, i00, i01, i10, i11);
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyCNOT(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
               std::size_t num_qubits, const std::vector<std::size_t> &wires,
               [[maybe_unused]] bool inverse = false,
               [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i01_ = i01;

        kokkos_swap(arr(i10), arr(i11));
    };

    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyCY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
             std::size_t num_qubits, const std::vector<std::size_t> &wires,
             [[maybe_unused]] bool inverse = false,
             [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i01_ = i01;
        Kokkos::complex<PrecisionT> v10 = arr(i10);
        arr(i10) = Kokkos::complex<PrecisionT>{imag(arr(i11)), -real(arr(i11))};
        arr(i11) = Kokkos::complex<PrecisionT>{-imag(v10), real(v10)};
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyCZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
             std::size_t num_qubits, const std::vector<std::size_t> &wires,
             [[maybe_unused]] bool inverse = false,
             [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i01_ = i01;
        [[maybe_unused]] const auto i10_ = i10;
        arr(i11) *= -1;
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCSWAP(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 std::size_t num_qubits,
                 const std::vector<std::size_t> &controlled_wires,
                 const std::vector<bool> &controlled_values,
                 const std::vector<std::size_t> &wires,
                 [[maybe_unused]] bool inverse = false,
                 [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i11_ = i11;

        kokkos_swap(arr(i10), arr(i01));
    };
    if (controlled_wires.empty()) {
        applyNC2Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC2Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applySWAP(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
               std::size_t num_qubits, const std::vector<std::size_t> &wires,
               bool inverse = false,
               [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNCSWAP<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                            inverse);
}

template <class ExecutionSpace, class PrecisionT>
void applyControlledPhaseShift(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                               std::size_t num_qubits,
                               const std::vector<std::size_t> &wires,
                               bool inverse = false,
                               const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const Kokkos::complex<PrecisionT> s =
        (inverse) ? Kokkos::exp(-Kokkos::complex<PrecisionT>(0, angle))
                  : Kokkos::exp(Kokkos::complex<PrecisionT>(0, angle));
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i01_ = i01;
        [[maybe_unused]] const auto i10_ = i10;
        arr(i11) *= s;
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyCRX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
              std::size_t num_qubits, const std::vector<std::size_t> &wires,
              bool inverse = false,
              const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT js =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i01_ = i01;
        const Kokkos::complex<PrecisionT> v10 = arr(i10);
        const Kokkos::complex<PrecisionT> v11 = arr(i11);
        arr(i10) = Kokkos::complex<PrecisionT>{c * real(v10) + js * imag(v11),
                                               c * imag(v10) - js * real(v11)};
        arr(i11) = Kokkos::complex<PrecisionT>{c * real(v11) + js * imag(v10),
                                               c * imag(v11) - js * real(v10)};
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyCRY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
              std::size_t num_qubits, const std::vector<std::size_t> &wires,
              bool inverse = false,
              const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i01_ = i01;
        const Kokkos::complex<PrecisionT> v10 = arr(i10);
        const Kokkos::complex<PrecisionT> v11 = arr(i11);
        arr(i10) = c * v10 - s * v11;
        arr(i11) = s * v10 + c * v11;
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyCRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
              std::size_t num_qubits, const std::vector<std::size_t> &wires,
              bool inverse = false,
              const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cos_angle = std::cos(angle * static_cast<PrecisionT>(0.5));
    const PrecisionT sin_angle = std::sin(angle * static_cast<PrecisionT>(0.5));
    const Kokkos::complex<PrecisionT> shift_0{
        cos_angle, (inverse) ? sin_angle : -sin_angle};
    const Kokkos::complex<PrecisionT> shift_1 = Kokkos::conj(shift_0);
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i01_ = i01;
        arr(i10) *= shift_0;
        arr(i11) *= shift_1;
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyCRot(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
               std::size_t num_qubits, const std::vector<std::size_t> &wires,
               bool inverse = false,
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
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i01_ = i01;
        const Kokkos::complex<PrecisionT> v0 = arr(i10);
        const Kokkos::complex<PrecisionT> v1 = arr(i11);
        arr(i10) = mat_0b00 * v0 + mat_0b01 * v1;
        arr(i11) = mat_0b10 * v0 + mat_0b11 * v1;
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCIsingXX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                    std::size_t num_qubits,
                    const std::vector<std::size_t> &controlled_wires,
                    const std::vector<bool> &controlled_values,
                    const std::vector<std::size_t> &wires, bool inverse = false,
                    const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        const Kokkos::complex<PrecisionT> v00 = arr(i00);
        const Kokkos::complex<PrecisionT> v01 = arr(i01);
        const Kokkos::complex<PrecisionT> v10 = arr(i10);
        const Kokkos::complex<PrecisionT> v11 = arr(i11);
        arr(i00) = Kokkos::complex<PrecisionT>{cr * real(v00) + sj * imag(v11),
                                               cr * imag(v00) - sj * real(v11)};
        arr(i01) = Kokkos::complex<PrecisionT>{cr * real(v01) + sj * imag(v10),
                                               cr * imag(v01) - sj * real(v10)};
        arr(i10) = Kokkos::complex<PrecisionT>{cr * real(v10) + sj * imag(v01),
                                               cr * imag(v10) - sj * real(v01)};
        arr(i11) = Kokkos::complex<PrecisionT>{cr * real(v11) + sj * imag(v00),
                                               cr * imag(v11) - sj * real(v00)};
    };
    if (controlled_wires.empty()) {
        applyNC2Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC2Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyIsingXX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  std::size_t num_qubits, const std::vector<std::size_t> &wires,
                  bool inverse = false,
                  const std::vector<PrecisionT> &params = {}) {
    applyNCIsingXX<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                               inverse, params);
}
template <class ExecutionSpace, class PrecisionT>
void applyNCIsingXY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                    std::size_t num_qubits,
                    const std::vector<std::size_t> &controlled_wires,
                    const std::vector<bool> &controlled_values,
                    const std::vector<std::size_t> &wires, bool inverse = false,
                    const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        const Kokkos::complex<PrecisionT> v00 = arr(i00);
        const Kokkos::complex<PrecisionT> v01 = arr(i01);
        const Kokkos::complex<PrecisionT> v10 = arr(i10);
        const Kokkos::complex<PrecisionT> v11 = arr(i11);
        arr(i00) = Kokkos::complex<PrecisionT>{real(v00), imag(v00)};
        arr(i01) = Kokkos::complex<PrecisionT>{cr * real(v01) - sj * imag(v10),
                                               cr * imag(v01) + sj * real(v10)};
        arr(i10) = Kokkos::complex<PrecisionT>{cr * real(v10) - sj * imag(v01),
                                               cr * imag(v10) + sj * real(v01)};
        arr(i11) = Kokkos::complex<PrecisionT>{real(v11), imag(v11)};
    };
    if (controlled_wires.empty()) {
        applyNC2Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC2Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyIsingXY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  std::size_t num_qubits, const std::vector<std::size_t> &wires,
                  bool inverse = false,
                  const std::vector<PrecisionT> &params = {}) {
    applyNCIsingXY<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                               inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCIsingYY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                    std::size_t num_qubits,
                    const std::vector<std::size_t> &controlled_wires,
                    const std::vector<bool> &controlled_values,
                    const std::vector<std::size_t> &wires, bool inverse = false,
                    const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        const Kokkos::complex<PrecisionT> v00 = arr(i00);
        const Kokkos::complex<PrecisionT> v01 = arr(i01);
        const Kokkos::complex<PrecisionT> v10 = arr(i10);
        const Kokkos::complex<PrecisionT> v11 = arr(i11);
        arr(i00) = Kokkos::complex<PrecisionT>{cr * real(v00) - sj * imag(v11),
                                               cr * imag(v00) + sj * real(v11)};
        arr(i01) = Kokkos::complex<PrecisionT>{cr * real(v01) + sj * imag(v10),
                                               cr * imag(v01) - sj * real(v10)};
        arr(i10) = Kokkos::complex<PrecisionT>{cr * real(v10) + sj * imag(v01),
                                               cr * imag(v10) - sj * real(v01)};
        arr(i11) = Kokkos::complex<PrecisionT>{cr * real(v11) - sj * imag(v00),
                                               cr * imag(v11) + sj * real(v00)};
    };
    if (controlled_wires.empty()) {
        applyNC2Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC2Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyIsingYY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  std::size_t num_qubits, const std::vector<std::size_t> &wires,
                  bool inverse = false,
                  const std::vector<PrecisionT> &params = {}) {
    applyNCIsingYY<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                               inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCIsingZZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                    std::size_t num_qubits,
                    const std::vector<std::size_t> &controlled_wires,
                    const std::vector<bool> &controlled_values,
                    const std::vector<std::size_t> &wires, bool inverse = false,
                    const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const Kokkos::complex<PrecisionT> shift_0 = Kokkos::complex<PrecisionT>{
        std::cos(angle / 2),
        (inverse) ? std::sin(angle / 2) : -std::sin(angle / 2)};
    const Kokkos::complex<PrecisionT> shift_1 = Kokkos::conj(shift_0);
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        arr(i00) *= shift_0;
        arr(i01) *= shift_1;
        arr(i10) *= shift_1;
        arr(i11) *= shift_0;
    };
    if (controlled_wires.empty()) {
        applyNC2Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC2Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyIsingZZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  std::size_t num_qubits, const std::vector<std::size_t> &wires,
                  bool inverse = false,
                  const std::vector<PrecisionT> &params = {}) {
    applyNCIsingZZ<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                               inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCSingleExcitation(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<std::size_t> &wires,
                             bool inverse = false,
                             const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i11_ = i11;

        const Kokkos::complex<PrecisionT> v01 = arr(i01);
        const Kokkos::complex<PrecisionT> v10 = arr(i10);
        arr(i01) = cr * v01 - sj * v10;
        arr(i10) = sj * v01 + cr * v10;
    };
    if (controlled_wires.empty()) {
        applyNC2Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC2Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applySingleExcitation(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                           std::size_t num_qubits,
                           const std::vector<std::size_t> &wires,
                           bool inverse = false,
                           const std::vector<PrecisionT> &params = {}) {
    applyNCSingleExcitation<ExecutionSpace, PrecisionT>(
        arr_, num_qubits, {}, {}, wires, inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCSingleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires, bool inverse = false,
    const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    const Kokkos::complex<PrecisionT> e =
        (inverse) ? Kokkos::exp(Kokkos::complex<PrecisionT>(0, angle / 2))
                  : Kokkos::exp(Kokkos::complex<PrecisionT>(0, -angle / 2));
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        const Kokkos::complex<PrecisionT> v01 = arr(i01);
        const Kokkos::complex<PrecisionT> v10 = arr(i10);
        arr(i00) *= e;
        arr(i01) = cr * v01 - sj * v10;
        arr(i10) = sj * v01 + cr * v10;
        arr(i11) *= e;
    };
    if (controlled_wires.empty()) {
        applyNC2Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC2Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applySingleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &wires, bool inverse = false,
    const std::vector<PrecisionT> &params = {}) {
    applyNCSingleExcitationMinus<ExecutionSpace, PrecisionT>(
        arr_, num_qubits, {}, {}, wires, inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCSingleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires, bool inverse = false,
    const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    const Kokkos::complex<PrecisionT> e =
        (inverse) ? Kokkos::exp(Kokkos::complex<PrecisionT>(0, -angle / 2))
                  : Kokkos::exp(Kokkos::complex<PrecisionT>(0, angle / 2));
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i11_ = i11;

        const Kokkos::complex<PrecisionT> v01 = arr(i01);
        const Kokkos::complex<PrecisionT> v10 = arr(i10);
        arr(i00) *= e;
        arr(i01) = cr * v01 - sj * v10;
        arr(i10) = sj * v01 + cr * v10;
        arr(i11) *= e;
    };
    if (controlled_wires.empty()) {
        applyNC2Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC2Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applySingleExcitationPlus(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                               std::size_t num_qubits,
                               const std::vector<std::size_t> &wires,
                               bool inverse = false,
                               const std::vector<PrecisionT> &params = {}) {
    applyNCSingleExcitationPlus<ExecutionSpace, PrecisionT>(
        arr_, num_qubits, {}, {}, wires, inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCPSWAP(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  std::size_t num_qubits,
                  const std::vector<std::size_t> &controlled_wires,
                  const std::vector<bool> &controlled_values,
                  const std::vector<std::size_t> &wires, bool inverse = false,
                  const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const Kokkos::complex<PrecisionT> shift =
        Kokkos::exp(Kokkos::complex<PrecisionT>(0, inverse ? -angle : angle));
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i11_ = i11;

        arr(i01) *= shift;
        arr(i10) *= shift;
        kokkos_swap(arr(i10), arr(i01));
    };

    if (controlled_wires.empty()) {
        applyNC2Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC2Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyPSWAP(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                std::size_t num_qubits, const std::vector<std::size_t> &wires,
                bool inverse = false,
                const std::vector<PrecisionT> &params = {}) {
    applyNCPSWAP<ExecutionSpace, PrecisionT>(arr_, num_qubits, {}, {}, wires,
                                             inverse, params);
}

template <class PrecisionT, class FuncT> class applyNC3Functor {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;

    KokkosComplexVector arr;
    const FuncT core_function;
    const std::size_t rev_wire0;
    const std::size_t rev_wire1;
    const std::size_t rev_wire2;
    const std::size_t rev_wire0_shift;
    const std::size_t rev_wire1_shift;
    const std::size_t rev_wire2_shift;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

  public:
    template <class ExecutionSpace>
    applyNC3Functor([[maybe_unused]] ExecutionSpace exec,
                    KokkosComplexVector arr_, std::size_t num_qubits,
                    const std::vector<std::size_t> &wires, FuncT core_function_)
        : arr(arr_), core_function(core_function_),
          rev_wire0(num_qubits - wires[2] - 1),
          rev_wire1(num_qubits - wires[1] - 1),
          rev_wire2(num_qubits - wires[0] - 1),
          rev_wire0_shift(static_cast<std::size_t>(1U) << rev_wire0),
          rev_wire1_shift(static_cast<std::size_t>(1U) << rev_wire1),
          rev_wire2_shift(static_cast<std::size_t>(1U) << rev_wire2) {
        std::size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        std::size_t rev_wire_max = std::max(rev_wire0, rev_wire1);
        std::size_t rev_wire_mid{0U};
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
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(
                                 0, Pennylane::Util::exp2(num_qubits - 3)),
                             *this);
    }
    KOKKOS_FUNCTION void operator()(std::size_t k) const {
        std::size_t i000 = ((k << 3U) & parity_high) |
                           ((k << 2U) & parity_hmiddle) |
                           ((k << 1U) & parity_lmiddle) | (k & parity_low);
        std::size_t i001 = i000 | rev_wire0_shift;
        std::size_t i010 = i000 | rev_wire1_shift;
        std::size_t i011 = i000 | rev_wire1_shift | rev_wire0_shift;
        std::size_t i100 = i000 | rev_wire2_shift;
        std::size_t i101 = i000 | rev_wire2_shift | rev_wire0_shift;
        std::size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;
        std::size_t i111 =
            i000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        core_function(arr, i000, i001, i010, i011, i100, i101, i110, i111);
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyCSWAP(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                std::size_t num_qubits, const std::vector<std::size_t> &wires,
                [[maybe_unused]] bool inverse = false,
                [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC3Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i000, std::size_t i001, std::size_t i010,
                      std::size_t i011, std::size_t i100, std::size_t i101,
                      std::size_t i110, std::size_t i111) {
            [[maybe_unused]] const auto i000_ = i000;
            [[maybe_unused]] const auto i001_ = i001;
            [[maybe_unused]] const auto i010_ = i010;
            [[maybe_unused]] const auto i011_ = i011;
            [[maybe_unused]] const auto i100_ = i100;
            [[maybe_unused]] const auto i111_ = i111;

            kokkos_swap(arr(i101), arr(i110));
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyToffoli(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  std::size_t num_qubits, const std::vector<std::size_t> &wires,
                  [[maybe_unused]] bool inverse = false,
                  [[maybe_unused]] const std::vector<PrecisionT> &params = {}) {
    applyNC3Functor(
        ExecutionSpace{}, arr_, num_qubits, wires,
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i000, std::size_t i001, std::size_t i010,
                      std::size_t i011, std::size_t i100, std::size_t i101,
                      std::size_t i110, std::size_t i111) {
            [[maybe_unused]] const auto i000_ = i000;
            [[maybe_unused]] const auto i001_ = i001;
            [[maybe_unused]] const auto i010_ = i010;
            [[maybe_unused]] const auto i011_ = i011;
            [[maybe_unused]] const auto i100_ = i100;
            [[maybe_unused]] const auto i101_ = i101;

            kokkos_swap(arr(i111), arr(i110));
        });
}

template <class PrecisionT, class FuncT, bool has_controls>
class applyNC4Functor {};

template <class PrecisionT, class FuncT>
class applyNC4Functor<PrecisionT, FuncT, true> {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    const FuncT core_function;
    KokkosIntVector indices;
    KokkosIntVector parity;
    KokkosIntVector rev_wires;
    KokkosIntVector rev_wire_shifts;

  public:
    template <class ExecutionSpace>
    applyNC4Functor([[maybe_unused]] ExecutionSpace exec,
                    KokkosComplexVector arr_, std::size_t num_qubits,
                    const std::vector<std::size_t> &controlled_wires,
                    const std::vector<bool> &controlled_values,
                    const std::vector<std::size_t> &wires, FuncT core_function_)
        : arr(arr_), core_function(core_function_) {
        const auto &[parity_, rev_wires_] =
            reverseWires(num_qubits, wires, controlled_wires);
        parity = parity_;
        std::vector<std::size_t> indices_ =
            generateBitPatterns(wires, num_qubits);
        controlBitPatterns(indices_, num_qubits, controlled_wires,
                           controlled_values);
        indices = vector2view(indices_);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(
                0, Pennylane::Util::exp2(num_qubits - controlled_wires.size() -
                                         wires.size())),
            *this);
    }
    KOKKOS_FUNCTION void operator()(std::size_t k) const {
        const std::size_t offset = Util::parity_2_offset(parity, k);
        std::size_t i0000 = indices(0B0000);
        std::size_t i0001 = indices(0B0001);
        std::size_t i0010 = indices(0B0010);
        std::size_t i0011 = indices(0B0011);
        std::size_t i0100 = indices(0B0100);
        std::size_t i0101 = indices(0B0101);
        std::size_t i0110 = indices(0B0110);
        std::size_t i0111 = indices(0B0111);
        std::size_t i1000 = indices(0B1000);
        std::size_t i1001 = indices(0B1001);
        std::size_t i1010 = indices(0B1010);
        std::size_t i1011 = indices(0B1011);
        std::size_t i1100 = indices(0B1100);
        std::size_t i1101 = indices(0B1101);
        std::size_t i1110 = indices(0B1110);
        std::size_t i1111 = indices(0B1111);

        core_function(
            arr, i0000 + offset, i0001 + offset, i0010 + offset, i0011 + offset,
            i0100 + offset, i0101 + offset, i0110 + offset, i0111 + offset,
            i1000 + offset, i1001 + offset, i1010 + offset, i1011 + offset,
            i1100 + offset, i1101 + offset, i1110 + offset, i1111 + offset);
    }
};

template <class PrecisionT, class FuncT>
class applyNC4Functor<PrecisionT, FuncT, false> {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;

    KokkosComplexVector arr;
    const FuncT core_function;
    const std::size_t rev_wire0;
    const std::size_t rev_wire1;
    const std::size_t rev_wire2;
    const std::size_t rev_wire3;
    const std::size_t rev_wire0_shift;
    const std::size_t rev_wire1_shift;
    const std::size_t rev_wire2_shift;
    const std::size_t rev_wire3_shift;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_lmiddle;
    std::size_t parity_hmiddle;
    std::size_t parity_middle;

  public:
    template <class ExecutionSpace>
    applyNC4Functor([[maybe_unused]] ExecutionSpace exec,
                    KokkosComplexVector arr_, std::size_t num_qubits,
                    const std::vector<std::size_t> &wires, FuncT core_function_)
        : arr(arr_), core_function(core_function_),
          rev_wire0(num_qubits - wires[3] - 1),
          rev_wire1(num_qubits - wires[2] - 1),
          rev_wire2(num_qubits - wires[1] - 1),
          rev_wire3(num_qubits - wires[0] - 1),
          rev_wire0_shift(static_cast<std::size_t>(1U) << rev_wire0),
          rev_wire1_shift(static_cast<std::size_t>(1U) << rev_wire1),
          rev_wire2_shift(static_cast<std::size_t>(1U) << rev_wire2),
          rev_wire3_shift(static_cast<std::size_t>(1U) << rev_wire3) {
        std::size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        std::size_t rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        std::size_t rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        std::size_t rev_wire_max = std::max(rev_wire2, rev_wire3);
        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }
        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(
                                 0, Pennylane::Util::exp2(num_qubits - 4)),
                             *this);
    }
    KOKKOS_FUNCTION void operator()(std::size_t k) const {
        std::size_t i0000 = ((k << 4U) & parity_high) |
                            ((k << 3U) & parity_hmiddle) |
                            ((k << 2U) & parity_middle) |
                            ((k << 1U) & parity_lmiddle) | (k & parity_low);
        std::size_t i0001 = i0000 | rev_wire0_shift;
        std::size_t i0010 = i0000 | rev_wire1_shift;
        std::size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        std::size_t i0100 = i0000 | rev_wire2_shift;
        std::size_t i0101 = i0000 | rev_wire2_shift | rev_wire0_shift;
        std::size_t i0110 = i0000 | rev_wire2_shift | rev_wire1_shift;
        std::size_t i0111 =
            i0000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        std::size_t i1000 = i0000 | rev_wire3_shift;
        std::size_t i1001 = i0000 | rev_wire3_shift | rev_wire0_shift;
        std::size_t i1010 = i0000 | rev_wire3_shift | rev_wire1_shift;
        std::size_t i1011 =
            i0000 | rev_wire3_shift | rev_wire1_shift | rev_wire0_shift;
        std::size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;
        std::size_t i1101 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire0_shift;
        std::size_t i1110 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire1_shift;
        std::size_t i1111 = i0000 | rev_wire3_shift | rev_wire2_shift |
                            rev_wire1_shift | rev_wire0_shift;
        core_function(arr, i0000, i0001, i0010, i0011, i0100, i0101, i0110,
                      i0111, i1000, i1001, i1010, i1011, i1100, i1101, i1110,
                      i1111);
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyNCDoubleExcitation(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<std::size_t> &wires,
                             bool inverse = false,
                             const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj =
        (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i0000,
        std::size_t i0001, std::size_t i0010, std::size_t i0011,
        std::size_t i0100, std::size_t i0101, std::size_t i0110,
        std::size_t i0111, std::size_t i1000, std::size_t i1001,
        std::size_t i1010, std::size_t i1011, std::size_t i1100,
        std::size_t i1101, std::size_t i1110, std::size_t i1111) {
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
        const Kokkos::complex<PrecisionT> v3 = arr(i0011);
        const Kokkos::complex<PrecisionT> v12 = arr(i1100);
        arr(i0011) = cr * v3 - sj * v12;
        arr(i1100) = sj * v3 + cr * v12;
    };
    if (controlled_wires.empty()) {
        applyNC4Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC4Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyDoubleExcitation(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                           std::size_t num_qubits,
                           const std::vector<std::size_t> &wires,
                           bool inverse = false,
                           const std::vector<PrecisionT> &params = {}) {
    applyNCDoubleExcitation<ExecutionSpace, PrecisionT>(
        arr_, num_qubits, {}, {}, wires, inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCDoubleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires, bool inverse = false,
    const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const Kokkos::complex<PrecisionT> e =
        inverse ? Kokkos::exp(Kokkos::complex<PrecisionT>(0, angle / 2))
                : Kokkos::exp(Kokkos::complex<PrecisionT>(0, -angle / 2));
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i0000,
        std::size_t i0001, std::size_t i0010, std::size_t i0011,
        std::size_t i0100, std::size_t i0101, std::size_t i0110,
        std::size_t i0111, std::size_t i1000, std::size_t i1001,
        std::size_t i1010, std::size_t i1011, std::size_t i1100,
        std::size_t i1101, std::size_t i1110, std::size_t i1111) {
        const Kokkos::complex<PrecisionT> v3 = arr(i0011);
        const Kokkos::complex<PrecisionT> v12 = arr(i1100);
        arr(i0000) *= e;
        arr(i0001) *= e;
        arr(i0010) *= e;
        arr(i0011) = cr * v3 - sj * v12;
        arr(i0100) *= e;
        arr(i0101) *= e;
        arr(i0110) *= e;
        arr(i0111) *= e;
        arr(i1000) *= e;
        arr(i1001) *= e;
        arr(i1010) *= e;
        arr(i1011) *= e;
        arr(i1100) = sj * v3 + cr * v12;
        arr(i1101) *= e;
        arr(i1110) *= e;
        arr(i1111) *= e;
    };
    if (controlled_wires.empty()) {
        applyNC4Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC4Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyDoubleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &wires, bool inverse = false,
    const std::vector<PrecisionT> &params = {}) {
    applyNCDoubleExcitationMinus<ExecutionSpace, PrecisionT>(
        arr_, num_qubits, {}, {}, wires, inverse, params);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCDoubleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires, bool inverse = false,
    const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const PrecisionT cr = std::cos(angle / 2);
    const PrecisionT sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const Kokkos::complex<PrecisionT> e =
        inverse ? Kokkos::exp(Kokkos::complex<PrecisionT>(0, -angle / 2))
                : Kokkos::exp(Kokkos::complex<PrecisionT>(0, angle / 2));
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i0000,
        std::size_t i0001, std::size_t i0010, std::size_t i0011,
        std::size_t i0100, std::size_t i0101, std::size_t i0110,
        std::size_t i0111, std::size_t i1000, std::size_t i1001,
        std::size_t i1010, std::size_t i1011, std::size_t i1100,
        std::size_t i1101, std::size_t i1110, std::size_t i1111) {
        const Kokkos::complex<PrecisionT> v3 = arr(i0011);
        const Kokkos::complex<PrecisionT> v12 = arr(i1100);
        arr(i0000) *= e;
        arr(i0001) *= e;
        arr(i0010) *= e;
        arr(i0011) = cr * v3 - sj * v12;
        arr(i0100) *= e;
        arr(i0101) *= e;
        arr(i0110) *= e;
        arr(i0111) *= e;
        arr(i1000) *= e;
        arr(i1001) *= e;
        arr(i1010) *= e;
        arr(i1011) *= e;
        arr(i1100) = sj * v3 + cr * v12;
        arr(i1101) *= e;
        arr(i1110) *= e;
        arr(i1111) *= e;
    };
    if (controlled_wires.empty()) {
        applyNC4Functor<PrecisionT, decltype(core_function), false>(
            ExecutionSpace{}, arr_, num_qubits, wires, core_function);
    } else {
        applyNC4Functor<PrecisionT, decltype(core_function), true>(
            ExecutionSpace{}, arr_, num_qubits, controlled_wires,
            controlled_values, wires, core_function);
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyDoubleExcitationPlus(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                               std::size_t num_qubits,
                               const std::vector<std::size_t> &wires,
                               bool inverse = false,
                               const std::vector<PrecisionT> &params = {}) {
    applyNCDoubleExcitationPlus<ExecutionSpace, PrecisionT>(
        arr_, num_qubits, {}, {}, wires, inverse, params);
}

// Note: Directly using parallel_for with KOKKOS_LAMBDA
// may cause segmentation faults for CUDA with RelWithDebugInfo.
// Therefore we use a functor class to encapsulate the logic
template <typename PrecisionT> class applyMultiRZFunctor {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;
    using MemberType = Kokkos::TeamPolicy<>::member_type;

    KokkosComplexVector arr;
    Kokkos::complex<PrecisionT> shift_0;
    Kokkos::complex<PrecisionT> shift_1;
    std::size_t wires_parity;

  public:
    template <class ExecutionSpace>
    applyMultiRZFunctor([[maybe_unused]] ExecutionSpace exec,
                        KokkosComplexVector arr_, std::size_t num_qubits,
                        const std::vector<std::size_t> &wires, bool inverse,
                        PrecisionT angle)
        : arr(arr_) {
        shift_0 = Kokkos::complex<PrecisionT>{std::cos(angle / 2),
                                              (inverse) ? std::sin(angle / 2)
                                                        : -std::sin(angle / 2)};
        shift_1 = Kokkos::conj(shift_0);
        wires_parity = 0U;
        for (std::size_t wire : wires) {
            wires_parity |=
                (static_cast<std::size_t>(1U) << (num_qubits - wire - 1));
        }

        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(
                                 0, Pennylane::Util::exp2(num_qubits)),
                             *this);
    }

    KOKKOS_FUNCTION void operator()(const std::size_t k) const {
        arr(k) *= (Kokkos::Impl::bit_count(k & wires_parity) % 2 == 0)
                      ? shift_0
                      : shift_1;
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyMultiRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  std::size_t num_qubits, const std::vector<std::size_t> &wires,
                  bool inverse = false,
                  const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];

    applyMultiRZFunctor(ExecutionSpace{}, arr_, num_qubits, wires, inverse,
                        angle);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCMultiRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                    std::size_t num_qubits,
                    const std::vector<std::size_t> &controlled_wires,
                    const std::vector<bool> &controlled_values,
                    const std::vector<std::size_t> &wires, bool inverse = false,
                    const std::vector<PrecisionT> &params = {}) {
    const PrecisionT angle = params[0];
    const Kokkos::complex<PrecisionT> shift_0 = Kokkos::complex<PrecisionT>{
        std::cos(angle / 2),
        (inverse) ? std::sin(angle / 2) : -std::sin(angle / 2)};
    const Kokkos::complex<PrecisionT> shift_1 = Kokkos::conj(shift_0);
    std::size_t wires_parity = 0U;
    wires_parity =
        std::accumulate(wires.begin(), wires.end(), std::size_t{0},
                        [num_qubits](std::size_t acc, std::size_t wire) {
                            return acc | (static_cast<std::size_t>(1U)
                                          << (num_qubits - wire - 1));
                        });
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i,
        Kokkos::View<std::size_t *> indices, std::size_t offset) {
        std::size_t index = indices(i);
        arr(index + offset) *=
            (Kokkos::Impl::bit_count((index + offset) & wires_parity) % 2 == 0)
                ? shift_0
                : shift_1;
    };

    applyNCNFunctor(ExecutionSpace{}, arr_, num_qubits, controlled_wires,
                    controlled_values, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyPauliRot(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                   std::size_t num_qubits,
                   const std::vector<std::size_t> &wires, bool inverse,
                   const PrecisionT angle, const std::string &word) {
    using ComplexT = Kokkos::complex<PrecisionT>;
    constexpr auto IMAG = Pennylane::Util::IMAG<PrecisionT>();
    PL_ABORT_IF_NOT(wires.size() == word.size(),
                    "wires and word have incompatible dimensions.")
    if (std::find_if_not(word.begin(), word.end(),
                         [](const int w) { return w == 'Z'; }) == word.end()) {
        applyMultiRZ<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                     std::vector<PrecisionT>{angle});
        return;
    }
    const PrecisionT c = std::cos(angle / 2);
    const ComplexT s = ((inverse) ? IMAG : -IMAG) * std::sin(angle / 2);
    const std::vector<ComplexT> sines = {s, IMAG * s, -s, -IMAG * s};
    auto d_sines = vector2view(sines);
    auto get_mask =
        [num_qubits, &wires](
            [[maybe_unused]] const std::function<bool(const int)> &condition) {
            std::size_t mask{0U};
            for (std::size_t iw = 0; iw < wires.size(); iw++) {
                const auto bit = static_cast<std::size_t>(condition(iw));
                mask |= bit << (num_qubits - 1 - wires[iw]);
            }
            return mask;
        };
    const std::size_t mask_xy =
        get_mask([&word](const int a) { return word[a] != 'Z'; });
    const std::size_t mask_y =
        get_mask([&word](const int a) { return word[a] == 'Y'; });
    const std::size_t mask_z =
        get_mask([&word](const int a) { return word[a] == 'Z'; });
    const auto count_mask_y = std::popcount(mask_y);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0,
                                            Pennylane::Util::exp2(num_qubits)),
        KOKKOS_LAMBDA(std::size_t i0) {
            std::size_t i1 = i0 ^ mask_xy;
            if (i0 <= i1) {
                const auto count_y = Kokkos::Impl::bit_count(i0 & mask_y) * 2;
                const auto count_z = Kokkos::Impl::bit_count(i0 & mask_z) * 2;
                const auto sign_i0 = count_z + count_mask_y * 3 - count_y;
                const auto sign_i1 = count_z + count_mask_y + count_y;
                const ComplexT v0 = arr_(i0);
                const ComplexT v1 = arr_(i1);
                arr_(i0) = c * v0 + d_sines(sign_i0 % 4) * v1;
                arr_(i1) = c * v1 + d_sines(sign_i1 % 4) * v0;
            }
        });
}

template <class ExecutionSpace, class PrecisionT>
void applyNamedOperation(const GateOperation gateop,
                         Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                         std::size_t num_qubits,
                         const std::vector<std::size_t> &wires,
                         bool inverse = false,
                         const std::vector<PrecisionT> &params = {}) {
    switch (gateop) {
    case GateOperation::PauliX:
        applyPauliX<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::PauliY:
        applyPauliY<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::PauliZ:
        applyPauliZ<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::Hadamard:
        applyHadamard<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::S:
        applyS<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::SX:
        applySX<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::T:
        applyT<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::PhaseShift:
        applyPhaseShift<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                        params);
        return;
    case GateOperation::RX:
        applyRX<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::RY:
        applyRY<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::RZ:
        applyRZ<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::Rot:
        applyRot<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::CNOT:
        applyCNOT<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::CY:
        applyCY<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::CZ:
        applyCZ<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::SWAP:
        applySWAP<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::ControlledPhaseShift:
        applyControlledPhaseShift<ExecutionSpace>(arr_, num_qubits, wires,
                                                  inverse, params);
        return;
    case GateOperation::CRX:
        applyCRX<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::CRY:
        applyCRY<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::CRZ:
        applyCRZ<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::CRot:
        applyCRot<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::IsingXX:
        applyIsingXX<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::IsingXY:
        applyIsingXY<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::IsingYY:
        applyIsingYY<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::IsingZZ:
        applyIsingZZ<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::SingleExcitation:
        applySingleExcitation<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                              params);
        return;
    case GateOperation::SingleExcitationMinus:
        applySingleExcitationMinus<ExecutionSpace>(arr_, num_qubits, wires,
                                                   inverse, params);
        return;
    case GateOperation::SingleExcitationPlus:
        applySingleExcitationPlus<ExecutionSpace>(arr_, num_qubits, wires,
                                                  inverse, params);
        return;
    case GateOperation::CSWAP:
        applyCSWAP<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::Toffoli:
        applyToffoli<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::DoubleExcitation:
        applyDoubleExcitation<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                              params);
        return;
    case GateOperation::DoubleExcitationMinus:
        applyDoubleExcitationMinus<ExecutionSpace>(arr_, num_qubits, wires,
                                                   inverse, params);
        return;
    case GateOperation::DoubleExcitationPlus:
        applyDoubleExcitationPlus<ExecutionSpace>(arr_, num_qubits, wires,
                                                  inverse, params);
        return;
    case GateOperation::GlobalPhase:
        applyGlobalPhase<ExecutionSpace>(arr_, num_qubits, wires, inverse,
                                         params);
        return;
    case GateOperation::MultiRZ:
        applyMultiRZ<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    case GateOperation::PSWAP:
        applyPSWAP<ExecutionSpace>(arr_, num_qubits, wires, inverse, params);
        return;
    default:
        PL_ABORT("Gate operation does not exist.");
    }
}

template <class ExecutionSpace, class PrecisionT>
void applyNCNamedOperation(const ControlledGateOperation gateop,
                           Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                           std::size_t num_qubits,
                           const std::vector<std::size_t> &controlled_wires,
                           const std::vector<bool> &controlled_values,
                           const std::vector<std::size_t> &wires,
                           bool inverse = false,
                           const std::vector<PrecisionT> &params = {}) {
    switch (gateop) {
    case ControlledGateOperation::PauliX:
        applyNCPauliX<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                      controlled_values, wires, inverse,
                                      params);
        return;
    case ControlledGateOperation::PauliY:
        applyNCPauliY<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                      controlled_values, wires, inverse,
                                      params);
        return;
    case ControlledGateOperation::PauliZ:
        applyNCPauliZ<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                      controlled_values, wires, inverse,
                                      params);
        return;
    case ControlledGateOperation::Hadamard:
        applyNCHadamard<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                        controlled_values, wires, inverse,
                                        params);
        return;
    case ControlledGateOperation::S:
        applyNCS<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                 controlled_values, wires, inverse, params);
        return;
    case ControlledGateOperation::SX:
        applyNCSX<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                  controlled_values, wires, inverse, params);
        return;
    case ControlledGateOperation::T:
        applyNCT<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                 controlled_values, wires, inverse, params);
        return;
    case ControlledGateOperation::PhaseShift:
        applyNCPhaseShift<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                          controlled_values, wires, inverse,
                                          params);
        return;
    case ControlledGateOperation::RX:
        applyNCRX<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                  controlled_values, wires, inverse, params);
        return;
    case ControlledGateOperation::RY:
        applyNCRY<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                  controlled_values, wires, inverse, params);
        return;
    case ControlledGateOperation::RZ:
        applyNCRZ<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                  controlled_values, wires, inverse, params);
        return;
    case ControlledGateOperation::Rot:
        applyNCRot<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                   controlled_values, wires, inverse, params);
        return;
    case ControlledGateOperation::SWAP:
        applyNCSWAP<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                    controlled_values, wires, inverse, params);
        return;
    case ControlledGateOperation::IsingXX:
        applyNCIsingXX<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                       controlled_values, wires, inverse,
                                       params);
        return;
    case ControlledGateOperation::IsingXY:
        applyNCIsingXY<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                       controlled_values, wires, inverse,
                                       params);
        return;
    case ControlledGateOperation::IsingYY:
        applyNCIsingYY<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                       controlled_values, wires, inverse,
                                       params);
        return;
    case ControlledGateOperation::IsingZZ:
        applyNCIsingZZ<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                       controlled_values, wires, inverse,
                                       params);
        return;
    case ControlledGateOperation::SingleExcitation:
        applyNCSingleExcitation<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse, params);
        return;
    case ControlledGateOperation::SingleExcitationMinus:
        applyNCSingleExcitationMinus<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse, params);
        return;
    case ControlledGateOperation::SingleExcitationPlus:
        applyNCSingleExcitationPlus<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse, params);
        return;
    case ControlledGateOperation::DoubleExcitation:
        applyNCDoubleExcitation<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse, params);
        return;
    case ControlledGateOperation::DoubleExcitationMinus:
        applyNCDoubleExcitationMinus<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse, params);
        return;
    case ControlledGateOperation::DoubleExcitationPlus:
        applyNCDoubleExcitationPlus<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse, params);
        return;
    case ControlledGateOperation::GlobalPhase:
        applyNCGlobalPhase<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                           controlled_values, wires, inverse,
                                           params);
        return;
    case ControlledGateOperation::MultiRZ:
        applyNCMultiRZ<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                       controlled_values, wires, inverse,
                                       params);
        return;
    case ControlledGateOperation::PSWAP:
        applyNCPSWAP<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                     controlled_values, wires, inverse, params);
        return;
    default:
        PL_ABORT("Controlled gate operation does not exist.");
    }
}
} // namespace Pennylane::LightningKokkos::Functors
