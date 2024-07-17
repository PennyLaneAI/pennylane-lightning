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
#include "UtilKokkos.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using Kokkos::Experimental::swap;
using Pennylane::LightningKokkos::Util::one;
using Pennylane::LightningKokkos::Util::vector2view;
using Pennylane::LightningKokkos::Util::wires2Parity;
using std::size_t;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Functors {

template <class Precision> struct multiQubitOpFunctor {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<Precision> *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;
    using ScratchViewComplex =
        Kokkos::View<Kokkos::complex<Precision> *,
                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ScratchViewSizeT =
        Kokkos::View<std::size_t *,
                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using MemberType = Kokkos::TeamPolicy<>::member_type;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    KokkosIntVector parity;
    KokkosIntVector rev_wire_shifts;
    std::size_t dim;
    std::size_t num_qubits;

    multiQubitOpFunctor(KokkosComplexVector arr_, std::size_t num_qubits_,
                        const KokkosComplexVector &matrix_,
                        const std::vector<std::size_t> &wires_) {
        wires = vector2view(wires_);
        dim = one << wires_.size();
        num_qubits = num_qubits_;
        arr = arr_;
        matrix = matrix_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &teamMember) const {
        const std::size_t k = teamMember.league_rank();
        ScratchViewComplex coeffs_in(teamMember.team_scratch(0), dim);
        ScratchViewSizeT indices(teamMember.team_scratch(0), dim);
        if (teamMember.team_rank() == 0) {
            std::size_t idx = (k & parity(0));
            for (std::size_t i = 1; i < parity.size(); i++) {
                idx |= ((k << i) & parity(i));
            }
            indices(0) = idx;
            coeffs_in(0) = arr(idx);

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, 1, dim),
                                 [&](const std::size_t inner_idx) {
                                     std::size_t index = indices(0);
                                     for (std::size_t i = 0; i < wires.size();
                                          i++) {
                                         if ((inner_idx & (one << i)) != 0) {
                                             index |= rev_wire_shifts(i);
                                         }
                                     }
                                     indices(inner_idx) = index;
                                     coeffs_in(inner_idx) = arr(index);
                                 });
        }
        teamMember.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, dim), [&](const std::size_t i) {
                const auto idx = indices(i);
                arr(idx) = 0.0;
                const std::size_t base_idx = i * dim;

                for (std::size_t j = 0; j < dim; j++) {
                    arr(idx) += matrix(base_idx + j) * coeffs_in(j);
                }
            });
    }
};

template <class PrecisionT> struct apply1QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    const std::size_t n_wires = 1;
    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;
    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    apply1QubitOpFunctor(KokkosComplexVector arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;

        rev_wire = num_qubits - wires_[0] - 1;
        rev_wire_shift = (static_cast<std::size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const Kokkos::complex<PrecisionT> v0 = arr[i0];
        const Kokkos::complex<PrecisionT> v1 = arr[i1];

        arr(i0) = matrix(0B00) * v0 + matrix(0B01) * v1;
        arr(i1) = matrix(0B10) * v0 + matrix(0B11) * v1;
    }
};

template <class PrecisionT> struct apply2QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    const std::size_t n_wires = 2;
    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;
    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    apply2QubitOpFunctor(KokkosComplexVector arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;

        rev_wire0 = num_qubits - wires_[1] - 1;
        rev_wire1 = num_qubits - wires_[0] - 1; // Control qubit
        rev_wire0_shift = static_cast<std::size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<std::size_t>(1U) << rev_wire1;
        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);
        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v00 = arr[i00];
        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr(i00) = matrix(0B0000) * v00 + matrix(0B0001) * v01 +
                   matrix(0B0010) * v10 + matrix(0B0011) * v11;
        arr(i01) = matrix(0B0100) * v00 + matrix(0B0101) * v01 +
                   matrix(0B0110) * v10 + matrix(0B0111) * v11;
        arr(i10) = matrix(0B1000) * v00 + matrix(0B1001) * v01 +
                   matrix(0B1010) * v10 + matrix(0B1011) * v11;
        arr(i11) = matrix(0B1100) * v00 + matrix(0B1101) * v01 +
                   matrix(0B1110) * v10 + matrix(0B1111) * v11;
    }
};

#define GATEENTRY3(xx, yy) xx << 3 | yy
#define GATETERM3(xx, yy, vyy) matrix(GATEENTRY3(xx, yy)) * vyy
#define GATESUM3(xx)                                                           \
    GATETERM3(xx, 0B000, v000) + GATETERM3(xx, 0B001, v001) +                  \
        GATETERM3(xx, 0B010, v010) + GATETERM3(xx, 0B011, v011) +              \
        GATETERM3(xx, 0B100, v100) + GATETERM3(xx, 0B101, v101) +              \
        GATETERM3(xx, 0B110, v110) + GATETERM3(xx, 0B111, v111)

template <class PrecisionT> struct apply3QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    KokkosIntVector parity;
    KokkosIntVector rev_wire_shifts;
    const std::size_t n_wires = 3;
    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;

    apply3QubitOpFunctor(KokkosComplexVector arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        wires = vector2view(wires_);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        std::size_t i000 = (k & parity(0));
        for (std::size_t i = 1; i < parity.size(); i++) {
            i000 |= ((k << i) & parity(i));
        }
        ComplexT v000 = arr(i000);

        std::size_t i001 = i000 | rev_wire_shifts(0);
        ComplexT v001 = arr(i001);
        std::size_t i010 = i000 | rev_wire_shifts(1);
        ComplexT v010 = arr(i010);
        std::size_t i011 = i000 | rev_wire_shifts(0) | rev_wire_shifts(1);
        ComplexT v011 = arr(i011);
        std::size_t i100 = i000 | rev_wire_shifts(2);
        ComplexT v100 = arr(i100);
        std::size_t i101 = i000 | rev_wire_shifts(0) | rev_wire_shifts(2);
        ComplexT v101 = arr(i101);
        std::size_t i110 = i000 | rev_wire_shifts(1) | rev_wire_shifts(2);
        ComplexT v110 = arr(i110);
        std::size_t i111 =
            i000 | rev_wire_shifts(0) | rev_wire_shifts(1) | rev_wire_shifts(2);
        ComplexT v111 = arr(i111);
        arr(i000) = GATESUM3(0B000);
        arr(i001) = GATESUM3(0B001);
        arr(i010) = GATESUM3(0B010);
        arr(i011) = GATESUM3(0B011);
        arr(i100) = GATESUM3(0B100);
        arr(i101) = GATESUM3(0B101);
        arr(i110) = GATESUM3(0B110);
        arr(i111) = GATESUM3(0B111);
    }
};

#define GATEENTRY4(xx, yy) xx << 4 | yy
#define GATETERM4(xx, yy, vyy) matrix(GATEENTRY4(xx, yy)) * vyy
#define GATESUM4(xx)                                                           \
    GATETERM4(xx, 0B0000, v0000) + GATETERM4(xx, 0B0001, v0001) +              \
        GATETERM4(xx, 0B0010, v0010) + GATETERM4(xx, 0B0011, v0011) +          \
        GATETERM4(xx, 0B0100, v0100) + GATETERM4(xx, 0B0101, v0101) +          \
        GATETERM4(xx, 0B0110, v0110) + GATETERM4(xx, 0B0111, v0111) +          \
        GATETERM4(xx, 0B1000, v1000) + GATETERM4(xx, 0B1001, v1001) +          \
        GATETERM4(xx, 0B1010, v1010) + GATETERM4(xx, 0B1011, v1011) +          \
        GATETERM4(xx, 0B1100, v1100) + GATETERM4(xx, 0B1101, v1101) +          \
        GATETERM4(xx, 0B1110, v1110) + GATETERM4(xx, 0B1111, v1111)

template <class PrecisionT> struct apply4QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    KokkosIntVector parity;
    KokkosIntVector rev_wire_shifts;
    const std::size_t n_wires = 4;
    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;

    apply4QubitOpFunctor(KokkosComplexVector arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        wires = vector2view(wires_);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        std::size_t i0000 = (k & parity(0));
        for (std::size_t i = 1; i < parity.size(); i++) {
            i0000 |= ((k << i) & parity(i));
        }
        ComplexT v0000 = arr(i0000);

        std::size_t i0001 = i0000 | rev_wire_shifts(0);
        ComplexT v0001 = arr(i0001);
        std::size_t i0010 = i0000 | rev_wire_shifts(1);
        ComplexT v0010 = arr(i0010);
        std::size_t i0011 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1);
        ComplexT v0011 = arr(i0011);
        std::size_t i0100 = i0000 | rev_wire_shifts(2);
        ComplexT v0100 = arr(i0100);
        std::size_t i0101 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(2);
        ComplexT v0101 = arr(i0101);
        std::size_t i0110 = i0000 | rev_wire_shifts(1) | rev_wire_shifts(2);
        ComplexT v0110 = arr(i0110);
        std::size_t i0111 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                            rev_wire_shifts(2);
        ComplexT v0111 = arr(i0111);
        std::size_t i1000 = i0000 | rev_wire_shifts(3);
        ComplexT v1000 = arr(i1000);
        std::size_t i1001 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(3);
        ComplexT v1001 = arr(i1001);
        std::size_t i1010 = i0000 | rev_wire_shifts(1) | rev_wire_shifts(3);
        ComplexT v1010 = arr(i1010);
        std::size_t i1011 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                            rev_wire_shifts(3);
        ComplexT v1011 = arr(i1011);
        std::size_t i1100 = i0000 | rev_wire_shifts(2) | rev_wire_shifts(3);
        ComplexT v1100 = arr(i1100);
        std::size_t i1101 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                            rev_wire_shifts(3);
        ComplexT v1101 = arr(i1101);
        std::size_t i1110 = i0000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                            rev_wire_shifts(3);
        ComplexT v1110 = arr(i1110);
        std::size_t i1111 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                            rev_wire_shifts(2) | rev_wire_shifts(3);
        ComplexT v1111 = arr(i1111);

        arr(i0000) = GATESUM4(0B0000);
        arr(i0001) = GATESUM4(0B0001);
        arr(i0010) = GATESUM4(0B0010);
        arr(i0011) = GATESUM4(0B0011);
        arr(i0100) = GATESUM4(0B0100);
        arr(i0101) = GATESUM4(0B0101);
        arr(i0110) = GATESUM4(0B0110);
        arr(i0111) = GATESUM4(0B0111);
        arr(i1000) = GATESUM4(0B1000);
        arr(i1001) = GATESUM4(0B1001);
        arr(i1010) = GATESUM4(0B1010);
        arr(i1011) = GATESUM4(0B1011);
        arr(i1100) = GATESUM4(0B1100);
        arr(i1101) = GATESUM4(0B1101);
        arr(i1110) = GATESUM4(0B1110);
        arr(i1111) = GATESUM4(0B1111);
    }
};

#define GATEENTRY5(xx, yy) xx << 5 | yy
#define GATETERM5(xx, yy, vyy) matrix(GATEENTRY5(xx, yy)) * vyy
#define GATESUM5(xx)                                                           \
    GATETERM5(xx, 0B00000, v00000) + GATETERM5(xx, 0B00001, v00001) +          \
        GATETERM5(xx, 0B00010, v00010) + GATETERM5(xx, 0B00011, v00011) +      \
        GATETERM5(xx, 0B00100, v00100) + GATETERM5(xx, 0B00101, v00101) +      \
        GATETERM5(xx, 0B00110, v00110) + GATETERM5(xx, 0B00111, v00111) +      \
        GATETERM5(xx, 0B01000, v01000) + GATETERM5(xx, 0B01001, v01001) +      \
        GATETERM5(xx, 0B01010, v01010) + GATETERM5(xx, 0B01011, v01011) +      \
        GATETERM5(xx, 0B01100, v01100) + GATETERM5(xx, 0B01101, v01101) +      \
        GATETERM5(xx, 0B01110, v01110) + GATETERM5(xx, 0B01111, v01111) +      \
        GATETERM5(xx, 0B10000, v10000) + GATETERM5(xx, 0B10001, v10001) +      \
        GATETERM5(xx, 0B10010, v10010) + GATETERM5(xx, 0B10011, v10011) +      \
        GATETERM5(xx, 0B10100, v10100) + GATETERM5(xx, 0B10101, v10101) +      \
        GATETERM5(xx, 0B10110, v10110) + GATETERM5(xx, 0B10111, v10111) +      \
        GATETERM5(xx, 0B11000, v11000) + GATETERM5(xx, 0B11001, v11001) +      \
        GATETERM5(xx, 0B11010, v11010) + GATETERM5(xx, 0B11011, v11011) +      \
        GATETERM5(xx, 0B11100, v11100) + GATETERM5(xx, 0B11101, v11101) +      \
        GATETERM5(xx, 0B11110, v11110) + GATETERM5(xx, 0B11111, v11111)
template <class PrecisionT> struct apply5QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    KokkosIntVector parity;
    KokkosIntVector rev_wire_shifts;
    const std::size_t n_wires = 5;
    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;

    apply5QubitOpFunctor(KokkosComplexVector arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        wires = vector2view(wires_);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        std::size_t i00000 = (k & parity(0));
        for (std::size_t i = 1; i < parity.size(); i++) {
            i00000 |= ((k << i) & parity(i));
        }
        ComplexT v00000 = arr(i00000);

        std::size_t i00001 = i00000 | rev_wire_shifts(0);
        ComplexT v00001 = arr(i00001);
        std::size_t i00010 = i00000 | rev_wire_shifts(1);
        ComplexT v00010 = arr(i00010);
        std::size_t i00011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1);
        ComplexT v00011 = arr(i00011);
        std::size_t i00100 = i00000 | rev_wire_shifts(2);
        ComplexT v00100 = arr(i00100);
        std::size_t i00101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2);
        ComplexT v00101 = arr(i00101);
        std::size_t i00110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2);
        ComplexT v00110 = arr(i00110);
        std::size_t i00111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2);
        ComplexT v00111 = arr(i00111);
        std::size_t i01000 = i00000 | rev_wire_shifts(3);
        ComplexT v01000 = arr(i01000);
        std::size_t i01001 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(3);
        ComplexT v01001 = arr(i01001);
        std::size_t i01010 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(3);
        ComplexT v01010 = arr(i01010);
        std::size_t i01011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(3);
        ComplexT v01011 = arr(i01011);
        std::size_t i01100 = i00000 | rev_wire_shifts(2) | rev_wire_shifts(3);
        ComplexT v01100 = arr(i01100);
        std::size_t i01101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                             rev_wire_shifts(3);
        ComplexT v01101 = arr(i01101);
        std::size_t i01110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                             rev_wire_shifts(3);
        ComplexT v01110 = arr(i01110);
        std::size_t i01111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2) | rev_wire_shifts(3);
        ComplexT v01111 = arr(i01111);
        std::size_t i10000 = i00000 | rev_wire_shifts(4);
        ComplexT v10000 = arr(i10000);
        std::size_t i10001 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(4);
        ComplexT v10001 = arr(i10001);
        std::size_t i10010 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(4);
        ComplexT v10010 = arr(i10010);
        std::size_t i10011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(4);
        ComplexT v10011 = arr(i10011);
        std::size_t i10100 = i00000 | rev_wire_shifts(2) | rev_wire_shifts(4);
        ComplexT v10100 = arr(i10100);
        std::size_t i10101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                             rev_wire_shifts(4);
        ComplexT v10101 = arr(i10101);
        std::size_t i10110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                             rev_wire_shifts(4);
        ComplexT v10110 = arr(i10110);
        std::size_t i10111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2) | rev_wire_shifts(4);
        ComplexT v10111 = arr(i10111);
        std::size_t i11000 = i00000 | rev_wire_shifts(3) | rev_wire_shifts(4);
        ComplexT v11000 = arr(i11000);
        std::size_t i11001 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        ComplexT v11001 = arr(i11001);
        std::size_t i11010 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        ComplexT v11010 = arr(i11010);
        std::size_t i11011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(3) | rev_wire_shifts(4);
        ComplexT v11011 = arr(i11011);
        std::size_t i11100 = i00000 | rev_wire_shifts(2) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        ComplexT v11100 = arr(i11100);
        std::size_t i11101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                             rev_wire_shifts(3) | rev_wire_shifts(4);
        ComplexT v11101 = arr(i11101);
        std::size_t i11110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                             rev_wire_shifts(3) | rev_wire_shifts(4);
        ComplexT v11110 = arr(i11110);
        std::size_t i11111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        ComplexT v11111 = arr(i11111);

        arr(i00000) = GATESUM5(0B00000);
        arr(i00001) = GATESUM5(0B00001);
        arr(i00010) = GATESUM5(0B00010);
        arr(i00011) = GATESUM5(0B00011);
        arr(i00100) = GATESUM5(0B00100);
        arr(i00101) = GATESUM5(0B00101);
        arr(i00110) = GATESUM5(0B00110);
        arr(i00111) = GATESUM5(0B00111);
        arr(i01000) = GATESUM5(0B01000);
        arr(i01001) = GATESUM5(0B01001);
        arr(i01010) = GATESUM5(0B01010);
        arr(i01011) = GATESUM5(0B01011);
        arr(i01100) = GATESUM5(0B01100);
        arr(i01101) = GATESUM5(0B01101);
        arr(i01110) = GATESUM5(0B01110);
        arr(i01111) = GATESUM5(0B01111);
        arr(i10000) = GATESUM5(0B10000);
        arr(i10001) = GATESUM5(0B10001);
        arr(i10010) = GATESUM5(0B10010);
        arr(i10011) = GATESUM5(0B10011);
        arr(i10100) = GATESUM5(0B10100);
        arr(i10101) = GATESUM5(0B10101);
        arr(i10110) = GATESUM5(0B10110);
        arr(i10111) = GATESUM5(0B10111);
        arr(i11000) = GATESUM5(0B11000);
        arr(i11001) = GATESUM5(0B11001);
        arr(i11010) = GATESUM5(0B11010);
        arr(i11011) = GATESUM5(0B11011);
        arr(i11100) = GATESUM5(0B11100);
        arr(i11101) = GATESUM5(0B11101);
        arr(i11110) = GATESUM5(0B11110);
        arr(i11111) = GATESUM5(0B11111);
    }
};

} // namespace Pennylane::LightningKokkos::Functors
