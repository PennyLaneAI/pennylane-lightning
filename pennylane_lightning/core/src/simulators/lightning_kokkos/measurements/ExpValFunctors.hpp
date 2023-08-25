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

#include "BitUtil.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Functors {
template <class PrecisionT> struct getExpectationValueIdentityFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    getExpectationValueIdentityFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
        [[maybe_unused]] std::size_t num_qubits,
        [[maybe_unused]] const std::vector<size_t> &wires) {
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        expval += real(conj(arr[k]) * arr[k]);
    }
};

template <class PrecisionT> struct getExpectationValuePauliXFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValuePauliXFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;

        expval += real(conj(arr[i0]) * arr[i1]);
        expval += real(conj(arr[i1]) * arr[i0]);
    }
};

template <class PrecisionT> struct getExpectationValuePauliYFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValuePauliYFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];

        expval += real(conj(arr[i0]) *
                       Kokkos::complex<PrecisionT>{imag(v1), -real(v1)});
        expval += real(conj(arr[i1]) *
                       Kokkos::complex<PrecisionT>{-imag(v0), real(v0)});
    }
};

template <class PrecisionT> struct getExpectationValuePauliZFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValuePauliZFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        expval += real(conj(arr[i1]) * (-arr[i1]));
        expval += real(conj(arr[i0]) * (arr[i0]));
    }
};

template <class PrecisionT> struct getExpectationValueHadamardFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValueHadamardFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const Kokkos::complex<PrecisionT> v0 = arr[i0];
        const Kokkos::complex<PrecisionT> v1 = arr[i1];

        expval += real(M_SQRT1_2 *
                       (conj(arr[i0]) * (v0 + v1) + conj(arr[i1]) * (v0 - v1)));
    }
};

template <class PrecisionT> struct getExpectationValueSingleQubitOpFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    Kokkos::View<Kokkos::complex<PrecisionT> *> matrix;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValueSingleQubitOpFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
        std::size_t num_qubits,
        const Kokkos::View<Kokkos::complex<PrecisionT> *> &matrix_,
        const std::vector<size_t> &wires) {
        arr = arr_;
        matrix = matrix_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;

        expval += real(
            conj(arr[i0]) * (matrix[0B00] * arr[i0] + matrix[0B01] * arr[i1]) +
            conj(arr[i1]) * (matrix[0B10] * arr[i0] + matrix[0B11] * arr[i1]));
    }
};

template <class PrecisionT> struct getExpectationValueTwoQubitOpFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    Kokkos::View<Kokkos::complex<PrecisionT> *> matrix;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    getExpectationValueTwoQubitOpFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
        std::size_t num_qubits,
        const Kokkos::View<Kokkos::complex<PrecisionT> *> &matrix_,
        const std::vector<size_t> &wires) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1;

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
        matrix = matrix_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        expval +=
            real(conj(arr[i00]) *
                     (matrix[0B0000] * arr[i00] + matrix[0B0001] * arr[i01] +
                      matrix[0B0010] * arr[i10] + matrix[0B0011] * arr[i11]) +
                 conj(arr[i01]) *
                     (matrix[0B0100] * arr[i00] + matrix[0B0101] * arr[i01] +
                      matrix[0B0110] * arr[i10] + matrix[0B0111] * arr[i11]) +
                 conj(arr[i10]) *
                     (matrix[0B1000] * arr[i00] + matrix[0B1001] * arr[i01] +
                      matrix[0B1010] * arr[i10] + matrix[0B1011] * arr[i11]) +
                 conj(arr[i11]) *
                     (matrix[0B1100] * arr[i00] + matrix[0B1101] * arr[i01] +
                      matrix[0B1110] * arr[i10] + matrix[0B1111] * arr[i11]));
    }
};

template <class PrecisionT> struct getExpectationValueMultiQubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;
    using ScratchViewComplex =
        Kokkos::View<ComplexT *,
                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using MemberType = Kokkos::TeamPolicy<>::member_type;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    std::size_t dim;
    std::size_t num_qubits;

    getExpectationValueMultiQubitOpFunctor(const KokkosComplexVector &arr_,
                                           std::size_t num_qubits_,
                                           const KokkosComplexVector &matrix_,
                                           KokkosIntVector &wires_) {
        dim = 1U << wires_.size();
        num_qubits = num_qubits_;
        wires = wires_;
        arr = arr_;
        matrix = matrix_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &teamMember, PrecisionT &expval) const {
        const std::size_t k = teamMember.league_rank() * dim;
        PrecisionT tempExpVal = 0.0;
        ScratchViewComplex coeffs_in(teamMember.team_scratch(0), dim);
        if (teamMember.team_rank() == 0) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(teamMember, dim),
                [&](const std::size_t inner_idx) {
                    std::size_t idx = k | inner_idx;
                    const std::size_t n_wires = wires.size();
                    for (std::size_t pos = 0; pos < n_wires; pos++) {
                        std::size_t x =
                            ((idx >> (n_wires - pos - 1)) ^
                             (idx >> (num_qubits - wires(pos) - 1))) &
                            1U;
                        idx = idx ^ ((x << (n_wires - pos - 1)) |
                                     (x << (num_qubits - wires(pos) - 1)));
                    }
                    coeffs_in(inner_idx) = arr(idx);
                });
        }
        teamMember.team_barrier();
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(teamMember, dim),
            [&](const std::size_t i, PrecisionT &innerExpVal) {
                const std::size_t base_idx = i * dim;
                ComplexT tmp{0.0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(teamMember, dim),
                    [&](const std::size_t j, ComplexT &isum) {
                        isum = isum + matrix(base_idx + j) * coeffs_in(j);
                    },
                    tmp);
                innerExpVal += real(conj(coeffs_in(i)) * tmp);
            },
            tempExpVal);
        if (teamMember.team_rank() == 0) {
            expval += tempExpVal;
        }
    }
};

template <class PrecisionT> struct getExpectationValueSparseFunctor {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;
    using KokkosSizeTVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector data;
    KokkosSizeTVector indices;
    KokkosSizeTVector indptr;
    std::size_t length;

    getExpectationValueSparseFunctor(KokkosComplexVector arr_,
                                     const KokkosComplexVector data_,
                                     const KokkosSizeTVector indices_,
                                     const KokkosSizeTVector indptr_) {
        length = indices_.size();
        indices = indices_;
        indptr = indptr_;
        data = data_;
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t row, PrecisionT &expval) const {
        for (size_t j = indptr[row]; j < indptr[row + 1]; j++) {
            expval += real(conj(arr[row]) * data[j] * arr[indices[j]]);
        }
    }
};

} // namespace Pennylane::LightningKokkos::Functors