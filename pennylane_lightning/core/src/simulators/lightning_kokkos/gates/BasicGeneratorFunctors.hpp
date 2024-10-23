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
void applyGenPhaseShift(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                        const std::size_t num_qubits,
                        const std::vector<std::size_t> &wires,
                        [[maybe_unused]] const bool inverse = false) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0, const std::size_t i1) {
        [[maybe_unused]] auto i1_ = i1;
        arr(i0) = 0.0;
    };
    applyNC1Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenControlledPhaseShift(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        [[maybe_unused]] const auto i11_ = i11;

        arr(i00) = 0.0;
        arr(i01) = 0.0;
        arr(i10) = 0.0;
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenCRX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 const std::size_t num_qubits,
                 const std::vector<std::size_t> &wires,
                 [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        arr(i00) = 0.0;
        arr(i01) = 0.0;
        kokkos_swap(arr(i10), arr(i11));
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenCRY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 const std::size_t num_qubits,
                 const std::vector<std::size_t> &wires,
                 [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        arr(i00) = 0.0;
        arr(i01) = 0.0;
        const auto v0 = arr(i10);
        arr(i10) = Kokkos::complex<PrecisionT>{imag(arr(i11)), -real(arr(i11))};
        arr(i11) = Kokkos::complex<PrecisionT>{-imag(v0), real(v0)};
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenCRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 const std::size_t num_qubits,
                 const std::vector<std::size_t> &wires,
                 [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        [[maybe_unused]] const auto i10_ = i10;
        arr(i00) = 0.0;
        arr(i01) = 0.0;
        arr(i11) *= -1;
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingXX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                     const std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        kokkos_swap(arr(i00), arr(i11));
        kokkos_swap(arr(i10), arr(i01));
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingXY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                     const std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        kokkos_swap(arr(i10), arr(i01));
        arr(i00) = 0.0;
        arr(i11) = 0.0;
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingYY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                     const std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        const auto v00 = arr(i00);
        arr(i00) = -arr(i11);
        arr(i11) = -v00;
        kokkos_swap(arr(i10), arr(i01));
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingZZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                     const std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i11_ = i11;

        arr(i10) *= -1;
        arr(i01) *= -1;
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitation(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                              const std::size_t num_qubits,
                              const std::vector<std::size_t> &wires,
                              [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        arr(i00) = 0.0;
        arr(i01) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
        arr(i10) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
        arr(i11) = 0.0;
        kokkos_swap(arr(i10), arr(i01));
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i11_ = i11;

        arr(i01) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
        arr(i10) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
        kokkos_swap(arr(i10), arr(i01));
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i00,
        const std::size_t i01, const std::size_t i10, const std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i11_ = i11;

        arr(i00) *= -1;
        arr(i01) *= Kokkos::complex<PrecisionT>{0.0, 1.0};
        arr(i10) *= Kokkos::complex<PrecisionT>{0.0, -1.0};
        arr(i11) *= -1;
        kokkos_swap(arr(i10), arr(i01));
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitation(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                              const std::size_t num_qubits,
                              const std::vector<std::size_t> &wires,
                              [[maybe_unused]] const bool inverse = false) {
    auto core_function =
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
    };
    applyNC4Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false) {
    auto core_function =
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
    };
    applyNC4Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false) {
    auto core_function =
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
    };
    applyNC4Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenMultiRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                     const std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] const bool inverse = false) {
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

template <class PrecisionT, class FuncT> class applyNCGenerator1Functor {
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    const FuncT core_function;
    KokkosIntVector indices;
    KokkosIntVector parity;
    KokkosIntVector rev_wires;
    KokkosIntVector rev_wire_shifts;
    std::size_t mask{0U};
    static constexpr std::size_t one{1U};
    std::size_t i0;
    std::size_t i1;

  public:
    template <class ExecutionSpace>
    applyNCGenerator1Functor([[maybe_unused]] ExecutionSpace exec,
                             Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<std::size_t> &wires,
                             FuncT core_function_)
        : arr(arr_), core_function(core_function_) {
        const std::size_t n_contr = controlled_wires.size();
        const std::size_t n_wires = wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(n_wires == 1);
        PL_ASSERT(num_qubits >= nw_tot);

        std::vector<std::size_t> all_wires;
        all_wires.reserve(nw_tot);
        all_wires.insert(all_wires.begin(), wires.begin(), wires.end());
        all_wires.insert(all_wires.begin(), controlled_wires.begin(),
                         controlled_wires.end());

        std::tie(parity, rev_wires) =
            reverseWires(num_qubits, wires, controlled_wires);
        std::vector<std::size_t> indices_ =
            generateBitPatterns(all_wires, num_qubits);
        for (std::size_t k = 0; k < controlled_values.size(); k++) {
            mask |= static_cast<std::size_t>(controlled_values[n_contr - 1 - k])
                    << k;
        }
        i0 = indices_[mask << one];
        i1 = indices_[(mask << one) | one];
        indices = vector2view(indices_);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(
                0, exp2(num_qubits - controlled_wires.size() - wires.size())),
            *this);
    }
    KOKKOS_FUNCTION void operator()(const std::size_t k) const {
        const std::size_t offset = parity_2_offset(parity, k);
        for (std::size_t i = 0; i < indices.size(); i++) {
            if ((i >> one) == mask) {
                continue;
            }
            arr(indices(i) + offset) = 0.0;
        }
        core_function(arr, i0 + offset, i1 + offset);
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyNCGenRX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  const std::size_t num_qubits,
                  const std::vector<std::size_t> &controlled_wires,
                  const std::vector<bool> &controlled_values,
                  const std::vector<std::size_t> &wires,
                  [[maybe_unused]] const bool inverse = false) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0, const std::size_t i1) {
        kokkos_swap(arr(i0), arr(i1));
    };
    applyNCGenerator1Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenRY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  const std::size_t num_qubits,
                  const std::vector<std::size_t> &controlled_wires,
                  const std::vector<bool> &controlled_values,
                  const std::vector<std::size_t> &wires,
                  [[maybe_unused]] const bool inverse = false) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      const std::size_t i0, const std::size_t i1) {
        const auto v0 = arr(i0);
        const auto v1 = arr(i1);
        arr(i0) = Kokkos::complex<PrecisionT>{imag(v1), -real(v1)};
        arr(i1) = Kokkos::complex<PrecisionT>{-imag(v0), real(v0)};
    };
    applyNCGenerator1Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  const std::size_t num_qubits,
                  const std::vector<std::size_t> &controlled_wires,
                  const std::vector<bool> &controlled_values,
                  const std::vector<std::size_t> &wires,
                  [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
        [[maybe_unused]] const std::size_t i0, const std::size_t i1) {
        arr(i1) *= -1;
    };
    applyNCGenerator1Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenPhaseShift(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                          const std::size_t num_qubits,
                          const std::vector<std::size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<std::size_t> &wires,
                          [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, const std::size_t i0,
        [[maybe_unused]] const std::size_t i1) {
        arr[i0] = 0.0;
    };
    applyNCGenerator1Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenGlobalPhase(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
    const std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    [[maybe_unused]] const std::vector<std::size_t> &wires,
    [[maybe_unused]] const bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        [[maybe_unused]] Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
        [[maybe_unused]] const std::size_t i0,
        [[maybe_unused]] const std::size_t i1){};
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
    applyNCGenerator1Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        {target}, core_function);
}

template <class ExecutionSpace, class PrecisionT>
PrecisionT applyNamedGenerator(const GeneratorOperation generator_op,
                               Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                               const std::size_t num_qubits,
                               const std::vector<std::size_t> &wires,
                               const bool inverse = false) {
    switch (generator_op) {
    case GeneratorOperation::RX:
        applyNamedOperation<ExecutionSpace>(GateOperation::PauliX, arr_,
                                            num_qubits, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::RY:
        applyNamedOperation<ExecutionSpace>(GateOperation::PauliY, arr_,
                                            num_qubits, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::RZ:
        applyNamedOperation<ExecutionSpace>(GateOperation::PauliZ, arr_,
                                            num_qubits, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::PhaseShift:
        applyGenPhaseShift<ExecutionSpace>(arr_, num_qubits, wires, inverse);
        return static_cast<PrecisionT>(1.0);
    case GeneratorOperation::ControlledPhaseShift:
        applyGenControlledPhaseShift<ExecutionSpace>(arr_, num_qubits, wires,
                                                     inverse);
        return static_cast<PrecisionT>(1);
    case GeneratorOperation::CRX:
        applyGenCRX<ExecutionSpace>(arr_, num_qubits, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::CRY:
        applyGenCRY<ExecutionSpace>(arr_, num_qubits, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::CRZ:
        applyGenCRZ<ExecutionSpace>(arr_, num_qubits, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingXX:
        applyGenIsingXX<ExecutionSpace>(arr_, num_qubits, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingXY:
        applyGenIsingXY<ExecutionSpace>(arr_, num_qubits, wires, inverse);
        return static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingYY:
        applyGenIsingYY<ExecutionSpace>(arr_, num_qubits, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::IsingZZ:
        applyGenIsingZZ<ExecutionSpace>(arr_, num_qubits, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::SingleExcitation:
        applyGenSingleExcitation<ExecutionSpace>(arr_, num_qubits, wires,
                                                 inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::SingleExcitationMinus:
        applyGenSingleExcitationMinus<ExecutionSpace>(arr_, num_qubits, wires,
                                                      inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::SingleExcitationPlus:
        applyGenSingleExcitationPlus<ExecutionSpace>(arr_, num_qubits, wires,
                                                     inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::DoubleExcitation:
        applyGenDoubleExcitation<ExecutionSpace>(arr_, num_qubits, wires,
                                                 inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::DoubleExcitationMinus:
        applyGenDoubleExcitationMinus<ExecutionSpace>(arr_, num_qubits, wires,
                                                      inverse);
        return -static_cast<PrecisionT>(0.5);
    case GeneratorOperation::DoubleExcitationPlus:
        applyGenDoubleExcitationPlus<ExecutionSpace>(arr_, num_qubits, wires,
                                                     inverse);
        return static_cast<PrecisionT>(0.5);
    case GeneratorOperation::MultiRZ:
        applyGenMultiRZ<ExecutionSpace>(arr_, num_qubits, wires, inverse);
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
PrecisionT
applyNCNamedGenerator(const ControlledGeneratorOperation generator_op,
                      Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                      const std::size_t num_qubits,
                      const std::vector<std::size_t> &controlled_wires,
                      const std::vector<bool> &controlled_values,
                      const std::vector<std::size_t> &wires,
                      const bool inverse = false) {
    switch (generator_op) {
    case ControlledGeneratorOperation::RX:
        applyNCGenRX<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                     controlled_values, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::RY:
        applyNCGenRY<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                     controlled_values, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::RZ:
        applyNCGenRZ<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                     controlled_values, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::PhaseShift:
        applyNCGenPhaseShift<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                             controlled_values, wires, inverse);
        return static_cast<PrecisionT>(1.0);
    case ControlledGeneratorOperation::GlobalPhase:
        applyNCGenGlobalPhase<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse);
        return static_cast<PrecisionT>(-1.0);
    /// LCOV_EXCL_START
    default:
        PL_ABORT("Controlled Generator operation does not exist.");
        /// LCOV_EXCL_STOP
    }
}

} // namespace Pennylane::LightningKokkos::Functors
