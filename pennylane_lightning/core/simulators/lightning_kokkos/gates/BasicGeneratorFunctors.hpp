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
                        std::size_t num_qubits,
                        const std::vector<std::size_t> &wires,
                        [[maybe_unused]] bool inverse = false) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        [[maybe_unused]] auto i1_ = i1;
        arr(i0) = 0.0;
    };
    applyNC1Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenControlledPhaseShift(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
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
                 std::size_t num_qubits, const std::vector<std::size_t> &wires,
                 [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        arr(i00) = 0.0;
        arr(i01) = 0.0;
        kokkos_swap(arr(i10), arr(i11));
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenCRY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                 std::size_t num_qubits, const std::vector<std::size_t> &wires,
                 [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
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
                 std::size_t num_qubits, const std::vector<std::size_t> &wires,
                 [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
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
                     std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        kokkos_swap(arr(i00), arr(i11));
        kokkos_swap(arr(i10), arr(i01));
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingXY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                     std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        kokkos_swap(arr(i10), arr(i01));
        arr(i00) = 0.0;
        arr(i11) = 0.0;
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenIsingYY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                     std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
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
                     std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
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
                              std::size_t num_qubits,
                              const std::vector<std::size_t> &wires,
                              [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        arr(i00) = 0.0;
        arr(i11) = 0.0;
        const auto v01 = arr(i01);
        const auto v10 = arr(i10);
        arr(i10) = Kokkos::complex<PrecisionT>{-imag(v01), real(v01)};
        arr(i01) = Kokkos::complex<PrecisionT>{imag(v10), -real(v10)};
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i11_ = i11;
        const auto v01 = arr(i01);
        const auto v10 = arr(i10);
        arr(i10) = Kokkos::complex<PrecisionT>{-imag(v01), real(v01)};
        arr(i01) = Kokkos::complex<PrecisionT>{imag(v10), -real(v10)};
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenSingleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        arr(i00) *= -1;
        arr(i11) *= -1;
        const auto v01 = arr(i01);
        const auto v10 = arr(i10);
        arr(i10) = Kokkos::complex<PrecisionT>{-imag(v01), real(v01)};
        arr(i01) = Kokkos::complex<PrecisionT>{imag(v10), -real(v10)};
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenPSWAP(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                   std::size_t num_qubits,
                   const std::vector<std::size_t> &wires,
                   [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        arr(i00) = 0.0;
        arr(i11) = 0.0;
        kokkos_swap(arr(i01), arr(i10));
    };
    applyNC2Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitation(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                              std::size_t num_qubits,
                              const std::vector<std::size_t> &wires,
                              [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i0000,
        std::size_t i0001, std::size_t i0010, std::size_t i0011,
        std::size_t i0100, std::size_t i0101, std::size_t i0110,
        std::size_t i0111, std::size_t i1000, std::size_t i1001,
        std::size_t i1010, std::size_t i1011, std::size_t i1100,
        std::size_t i1101, std::size_t i1110, std::size_t i1111) {
        const Kokkos::complex<PrecisionT> v0011 = arr(i0011);
        const Kokkos::complex<PrecisionT> v1100 = arr(i1100);
        arr(i0000) = 0.0;
        arr(i0001) = 0.0;
        arr(i0010) = 0.0;
        arr(i0011) = Kokkos::complex<PrecisionT>{imag(v1100), -real(v1100)};
        arr(i0100) = 0.0;
        arr(i0101) = 0.0;
        arr(i0110) = 0.0;
        arr(i0111) = 0.0;
        arr(i1000) = 0.0;
        arr(i1001) = 0.0;
        arr(i1010) = 0.0;
        arr(i1011) = 0.0;
        arr(i1100) = Kokkos::complex<PrecisionT>{-imag(v0011), real(v0011)};
        arr(i1101) = 0.0;
        arr(i1110) = 0.0;
        arr(i1111) = 0.0;
    };
    applyNC4Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
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
        const auto v0011 = arr(i0011);
        const auto v1100 = arr(i1100);
        arr(i0011) = Kokkos::complex<PrecisionT>{imag(v1100), -real(v1100)};
        arr(i1100) = Kokkos::complex<PrecisionT>{-imag(v0011), real(v0011)};
    };
    applyNC4Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenDoubleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
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
        const auto v0011 = arr(i0011);
        const auto v1100 = arr(i1100);
        arr(i0011) = Kokkos::complex<PrecisionT>{-imag(v1100), real(v1100)};
        arr(i1100) = Kokkos::complex<PrecisionT>{imag(v0011), -real(v0011)};
    };
    applyNC4Functor<PrecisionT, decltype(core_function), false>(
        ExecutionSpace{}, arr_, num_qubits, wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyGenMultiRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                     std::size_t num_qubits,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] bool inverse = false) {
    std::size_t wires_parity = static_cast<std::size_t>(0U);
    for (std::size_t wire : wires) {
        wires_parity |= Pennylane::Util::exp2(num_qubits - wire - 1);
    }
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0,
                                            Pennylane::Util::exp2(num_qubits)),
        KOKKOS_LAMBDA(std::size_t k) {
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
        const std::size_t n_wires = wires.size();
        const std::size_t n_contr = controlled_wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(n_wires == 1);
        PL_ASSERT(num_qubits >= nw_tot);

        std::vector<std::size_t> all_wires;
        all_wires.reserve(nw_tot);
        all_wires.insert(all_wires.begin(), controlled_wires.begin(),
                         controlled_wires.end());
        all_wires.insert(all_wires.begin() + n_contr, wires.begin(),
                         wires.end());

        const auto &[parity_, rev_wires_] =
            reverseWires(num_qubits, wires, controlled_wires);
        parity = parity_;
        const std::vector<std::size_t> indices_ =
            generateBitPatterns(all_wires, num_qubits);

        std::size_t k = 0;
        mask = std::accumulate(
            controlled_values.rbegin(), controlled_values.rend(),
            std::size_t{0}, [&k](std::size_t acc, std::size_t value) {
                return acc | (static_cast<std::size_t>(value) << k++);
            });

        i0 = indices_[mask << one];
        i1 = indices_[(mask << one) | one];
        indices = vector2view(indices_);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(
                0, Pennylane::Util::exp2(num_qubits - controlled_wires.size() -
                                         wires.size())),
            *this);
    }
    KOKKOS_FUNCTION void operator()(std::size_t k) const {
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
                  std::size_t num_qubits,
                  const std::vector<std::size_t> &controlled_wires,
                  const std::vector<bool> &controlled_values,
                  const std::vector<std::size_t> &wires,
                  [[maybe_unused]] bool inverse = false) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        kokkos_swap(arr(i0), arr(i1));
    };
    applyNCGenerator1Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenRY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                  std::size_t num_qubits,
                  const std::vector<std::size_t> &controlled_wires,
                  const std::vector<bool> &controlled_values,
                  const std::vector<std::size_t> &wires,
                  [[maybe_unused]] bool inverse = false) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
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
                  std::size_t num_qubits,
                  const std::vector<std::size_t> &controlled_wires,
                  const std::vector<bool> &controlled_values,
                  const std::vector<std::size_t> &wires,
                  [[maybe_unused]] bool inverse = false) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        [[maybe_unused]] const auto i0_ = i0;
        arr(i1) *= -1;
    };
    applyNCGenerator1Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenPhaseShift(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                          std::size_t num_qubits,
                          const std::vector<std::size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<std::size_t> &wires,
                          [[maybe_unused]] bool inverse = false) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        [[maybe_unused]] const auto i1_ = i1;
        arr[i0] = 0.0;
    };
    applyNCGenerator1Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenGlobalPhase(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    [[maybe_unused]] const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
    auto core_function =
        KOKKOS_LAMBDA(Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
                      std::size_t i0, std::size_t i1) {
        [[maybe_unused]] const auto &arr_ = arr;
        [[maybe_unused]] const auto i0_ = i0;
        [[maybe_unused]] const auto i1_ = i1;
    };
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

template <class PrecisionT, class FuncT> class applyNCGenerator2Functor {
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    const FuncT core_function;
    KokkosIntVector indices;
    KokkosIntVector parity;
    KokkosIntVector rev_wires;
    KokkosIntVector rev_wire_shifts;
    std::size_t mask{0U};
    static constexpr std::size_t one{1U};
    static constexpr std::size_t two{2U};
    std::size_t i00;
    std::size_t i01;
    std::size_t i10;
    std::size_t i11;

  public:
    template <class ExecutionSpace>
    applyNCGenerator2Functor([[maybe_unused]] ExecutionSpace exec,
                             Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<std::size_t> &wires,
                             FuncT core_function_)
        : arr(arr_), core_function(core_function_) {
        const std::size_t n_wires = wires.size();
        const std::size_t n_contr = controlled_wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(n_wires == 2);
        PL_ASSERT(num_qubits >= nw_tot);

        std::vector<std::size_t> all_wires;
        all_wires.reserve(nw_tot);
        all_wires.insert(all_wires.begin(), controlled_wires.begin(),
                         controlled_wires.end());
        all_wires.insert(all_wires.begin() + n_contr, wires.begin(),
                         wires.end());

        const auto &[parity_, rev_wires_] =
            reverseWires(num_qubits, wires, controlled_wires);
        parity = parity_;
        const std::vector<std::size_t> indices_ =
            generateBitPatterns(all_wires, num_qubits);

        std::size_t k = 0;
        mask = std::accumulate(
            controlled_values.rbegin(), controlled_values.rend(),
            std::size_t{0}, [&k](std::size_t acc, std::size_t value) {
                return acc | (static_cast<std::size_t>(value) << k++);
            });
        i00 = indices_[mask << two];
        i01 = indices_[(mask << two) | one];
        i10 = indices_[(mask << two) | two];
        i11 = indices_[(mask << two) | two | one];
        indices = vector2view(indices_);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(
                0, Pennylane::Util::exp2(num_qubits - controlled_wires.size() -
                                         wires.size())),
            *this);
    }
    KOKKOS_FUNCTION void operator()(std::size_t k) const {
        const std::size_t offset = parity_2_offset(parity, k);
        for (std::size_t i = 0; i < indices.size(); i++) {
            if ((i >> two) == mask) {
                continue;
            }
            arr(indices(i) + offset) = 0.0;
        }
        core_function(arr, i00 + offset, i01 + offset, i10 + offset,
                      i11 + offset);
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyNCGenIsingXX(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                       std::size_t num_qubits,
                       const std::vector<std::size_t> &controlled_wires,
                       const std::vector<bool> &controlled_values,
                       const std::vector<std::size_t> &wires,
                       [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        kokkos_swap(arr(i00), arr(i11));
        kokkos_swap(arr(i10), arr(i01));
    };
    applyNCGenerator2Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenIsingXY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                       std::size_t num_qubits,
                       const std::vector<std::size_t> &controlled_wires,
                       const std::vector<bool> &controlled_values,
                       const std::vector<std::size_t> &wires,
                       [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        arr(i00) = 0.0;
        arr(i11) = 0.0;
        kokkos_swap(arr(i10), arr(i01));
    };
    applyNCGenerator2Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenIsingYY(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                       std::size_t num_qubits,
                       const std::vector<std::size_t> &controlled_wires,
                       const std::vector<bool> &controlled_values,
                       const std::vector<std::size_t> &wires,
                       [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        const auto v00 = arr(i00);
        arr(i00) = -arr(i11);
        arr(i11) = -v00;
        kokkos_swap(arr(i10), arr(i01));
    };
    applyNCGenerator2Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenIsingZZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                       std::size_t num_qubits,
                       const std::vector<std::size_t> &controlled_wires,
                       const std::vector<bool> &controlled_values,
                       const std::vector<std::size_t> &wires,
                       [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i11_ = i11;
        arr(i10) *= -1;
        arr(i01) *= -1;
    };
    applyNCGenerator2Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenSingleExcitation(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        arr(i00) = 0.0;
        arr(i11) = 0.0;
        const auto v01 = arr(i01);
        const auto v10 = arr(i10);
        arr(i10) = Kokkos::complex<PrecisionT>{-imag(v01), real(v01)};
        arr(i01) = Kokkos::complex<PrecisionT>{imag(v10), -real(v10)};
    };
    applyNCGenerator2Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenSingleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        [[maybe_unused]] const auto i00_ = i00;
        [[maybe_unused]] const auto i11_ = i11;
        const auto v01 = arr(i01);
        const auto v10 = arr(i10);
        arr(i10) = Kokkos::complex<PrecisionT>{-imag(v01), real(v01)};
        arr(i01) = Kokkos::complex<PrecisionT>{imag(v10), -real(v10)};
    };
    applyNCGenerator2Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenSingleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        arr(i00) *= -1;
        arr(i11) *= -1;
        const auto v01 = arr(i01);
        const auto v10 = arr(i10);
        arr(i10) = Kokkos::complex<PrecisionT>{-imag(v01), real(v01)};
        arr(i01) = Kokkos::complex<PrecisionT>{imag(v10), -real(v10)};
    };
    applyNCGenerator2Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenPSWAP(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                     std::size_t num_qubits,
                     const std::vector<std::size_t> &controlled_wires,
                     const std::vector<bool> &controlled_values,
                     const std::vector<std::size_t> &wires,
                     [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i00,
        std::size_t i01, std::size_t i10, std::size_t i11) {
        arr(i00) = 0.0;
        arr(i11) = 0.0;
        kokkos_swap(arr(i01), arr(i10));
    };
    applyNCGenerator2Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class PrecisionT, class FuncT> class applyNCGenerator4Functor {
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    const FuncT core_function;
    KokkosIntVector indices;
    KokkosIntVector parity;
    KokkosIntVector rev_wires;
    KokkosIntVector rev_wire_shifts;
    std::size_t mask{0U};
    static constexpr std::size_t one{1U};
    static constexpr std::size_t two{2U};
    std::size_t i0011;
    std::size_t i1100;

  public:
    template <class ExecutionSpace>
    applyNCGenerator4Functor([[maybe_unused]] ExecutionSpace exec,
                             Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<std::size_t> &wires,
                             FuncT core_function_)
        : arr(arr_), core_function(core_function_) {
        const std::size_t n_wires = wires.size();
        const std::size_t n_contr = controlled_wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(n_wires == 4);
        PL_ASSERT(num_qubits >= nw_tot);

        std::vector<std::size_t> all_wires;
        all_wires.reserve(nw_tot);
        all_wires.insert(all_wires.begin(), controlled_wires.begin(),
                         controlled_wires.end());
        all_wires.insert(all_wires.begin() + n_contr, wires.begin(),
                         wires.end());

        const auto &[parity_, rev_wires_] =
            reverseWires(num_qubits, wires, controlled_wires);
        parity = parity_;
        const std::vector<std::size_t> indices_ =
            generateBitPatterns(all_wires, num_qubits);

        std::size_t k = 0;
        mask = std::accumulate(
            controlled_values.rbegin(), controlled_values.rend(),
            std::size_t{0}, [&k](std::size_t acc, std::size_t value) {
                return acc | (static_cast<std::size_t>(value) << k++);
            });
        i0011 = indices_[(mask << 4U) + 3U];
        i1100 = indices_[(mask << 4U) + 12U];
        indices = vector2view(indices_);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecutionSpace>(
                0, Pennylane::Util::exp2(num_qubits - controlled_wires.size() -
                                         wires.size())),
            *this);
    }
    KOKKOS_FUNCTION void operator()(std::size_t k) const {
        const std::size_t offset = parity_2_offset(parity, k);
        for (std::size_t i = 0; i < indices.size(); i++) {
            if ((i >> 4U) == mask) {
                continue;
            }
            arr(indices(i) + offset) = 0.0;
        }
        core_function(arr, i0011 + offset, i1100 + offset, indices, offset);
    }
};

template <class ExecutionSpace, class PrecisionT>
void applyNCGenDoubleExcitation(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i0011,
        std::size_t i1100, const KokkosIntVector &indices, std::size_t offset) {
        const auto v0011 = arr(i0011);
        const auto v1100 = arr(i1100);
        for (std::size_t i = 0; i < indices.size(); i++) {
            arr(indices(i) + offset) = 0.0;
        }
        arr(i0011) = Kokkos::complex<PrecisionT>{imag(v1100), -real(v1100)};
        arr(i1100) = Kokkos::complex<PrecisionT>{-imag(v0011), real(v0011)};
    };
    applyNCGenerator4Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenDoubleExcitationMinus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i0011,
        std::size_t i1100, const KokkosIntVector &indices, std::size_t offset) {
        [[maybe_unused]] const auto &indices_ = indices;
        [[maybe_unused]] const auto offset_ = offset;
        const auto v0011 = arr(i0011);
        const auto v1100 = arr(i1100);
        arr(i0011) = Kokkos::complex<PrecisionT>{imag(v1100), -real(v1100)};
        arr(i1100) = Kokkos::complex<PrecisionT>{-imag(v0011), real(v0011)};
    };
    applyNCGenerator4Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenDoubleExcitationPlus(
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires,
    [[maybe_unused]] bool inverse = false) {
    auto core_function = KOKKOS_LAMBDA(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr, std::size_t i0011,
        std::size_t i1100, const KokkosIntVector &indices, std::size_t offset) {
        [[maybe_unused]] const auto &indices_ = indices;
        [[maybe_unused]] const auto offset_ = offset;
        const auto v0011 = arr(i0011);
        const auto v1100 = arr(i1100);
        arr(i0011) = Kokkos::complex<PrecisionT>{-imag(v1100), real(v1100)};
        arr(i1100) = Kokkos::complex<PrecisionT>{imag(v0011), -real(v0011)};
    };
    applyNCGenerator4Functor<PrecisionT, decltype(core_function)>(
        ExecutionSpace{}, arr_, num_qubits, controlled_wires, controlled_values,
        wires, core_function);
}

template <class ExecutionSpace, class PrecisionT>
void applyNCGenMultiRZ(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                       std::size_t num_qubits,
                       const std::vector<std::size_t> &controlled_wires,
                       const std::vector<bool> &controlled_values,
                       const std::vector<std::size_t> &wires,
                       [[maybe_unused]] bool inverse = false) {
    auto ctrls_mask = static_cast<std::size_t>(0U);
    for (std::size_t i = 0; i < controlled_wires.size(); i++) {
        ctrls_mask |= (static_cast<std::size_t>(controlled_values[i])
                       << (num_qubits - controlled_wires[i] - 1));
    }
    std::size_t ctrls_parity = std::accumulate(
        controlled_wires.begin(), controlled_wires.end(), std::size_t{0},
        [num_qubits](std::size_t acc, std::size_t wire) {
            return acc | Pennylane::Util::exp2(num_qubits - wire - 1);
        });
    std::size_t wires_parity = std::accumulate(
        wires.begin(), wires.end(), std::size_t{0},
        [num_qubits](std::size_t acc, std::size_t wire) {
            return acc | Pennylane::Util::exp2(num_qubits - wire - 1);
        });

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0,
                                            Pennylane::Util::exp2(num_qubits)),
        KOKKOS_LAMBDA(std::size_t k) {
            if (ctrls_mask == (ctrls_parity & k)) {
                arr_(k) *= static_cast<PrecisionT>(
                    1 - 2 * int(Kokkos::Impl::bit_count(k & wires_parity) % 2));
            } else {
                arr_(k) = 0.0;
            }
        });
}

template <class ExecutionSpace, class PrecisionT>
PrecisionT applyNamedGenerator(const GeneratorOperation generator_op,
                               Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                               std::size_t num_qubits,
                               const std::vector<std::size_t> &wires,
                               bool inverse = false) {
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
    case GeneratorOperation::PSWAP:
        applyGenPSWAP<ExecutionSpace>(arr_, num_qubits, wires, inverse);
        return static_cast<PrecisionT>(1.0);
    default:
        PL_ABORT("Generator operation does not exist.");
    }
}

template <class ExecutionSpace, class PrecisionT>
PrecisionT applyNCNamedGenerator(
    const ControlledGeneratorOperation generator_op,
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr_, std::size_t num_qubits,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires, bool inverse = false) {
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
    case ControlledGeneratorOperation::IsingXX:
        applyNCGenIsingXX<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                          controlled_values, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::IsingXY:
        applyNCGenIsingXY<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                          controlled_values, wires, inverse);
        return static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::IsingYY:
        applyNCGenIsingYY<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                          controlled_values, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::IsingZZ:
        applyNCGenIsingZZ<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                          controlled_values, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::SingleExcitation:
        applyNCGenSingleExcitation<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::SingleExcitationMinus:
        applyNCGenSingleExcitationMinus<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::SingleExcitationPlus:
        applyNCGenSingleExcitationPlus<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::DoubleExcitation:
        applyNCGenDoubleExcitation<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::DoubleExcitationMinus:
        applyNCGenDoubleExcitationMinus<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::DoubleExcitationPlus:
        applyNCGenDoubleExcitationPlus<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse);
        return static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::MultiRZ:
        applyNCGenMultiRZ<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                          controlled_values, wires, inverse);
        return -static_cast<PrecisionT>(0.5);
    case ControlledGeneratorOperation::GlobalPhase:
        applyNCGenGlobalPhase<ExecutionSpace>(
            arr_, num_qubits, controlled_wires, controlled_values, wires,
            inverse);
        return static_cast<PrecisionT>(-1.0);
    case ControlledGeneratorOperation::PSWAP:
        applyNCGenPSWAP<ExecutionSpace>(arr_, num_qubits, controlled_wires,
                                        controlled_values, wires, inverse);
        return static_cast<PrecisionT>(1.0);
    default:
        PL_ABORT("Controlled generator operation does not exist.");
    }
}

} // namespace Pennylane::LightningKokkos::Functors
