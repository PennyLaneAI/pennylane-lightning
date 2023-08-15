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

namespace {
using namespace Pennylane::Util;
using Kokkos::Experimental::swap;
using std::size_t;
} // namespace

namespace Pennylane::LightningKokkos::Functors {
template <class PrecisionT, bool inverse = false> struct singleQubitOpFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    Kokkos::View<Kokkos::complex<PrecisionT> *> matrix;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

    singleQubitOpFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
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
    void operator()(const size_t k) const {
        if constexpr (inverse) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            const Kokkos::complex<PrecisionT> v0 = arr[i0];
            const Kokkos::complex<PrecisionT> v1 = arr[i1];

            arr[i0] =
                conj(matrix[0B00]) * v0 +
                conj(matrix[0B10]) * v1; // NOLINT(readability-magic-numbers)
            arr[i1] =
                conj(matrix[0B01]) * v0 +
                conj(matrix[0B11]) * v1; // NOLINT(readability-magic-numbers)
                                         // }
        } else {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            const Kokkos::complex<PrecisionT> v0 = arr[i0];
            const Kokkos::complex<PrecisionT> v1 = arr[i1];
            arr[i0] = matrix[0B00] * v0 +
                      matrix[0B01] * v1; // NOLINT(readability-magic-numbers)
            arr[i1] = matrix[0B10] * v0 +
                      matrix[0B11] * v1; // NOLINT(readability-magic-numbers)
        }
    }
};

/**
 * @brief Apply a two qubit gate to the statevector.
 *
 * @param arr Pointer to the statevector.
 * @param num_qubits Number of qubits.
 * @param matrix Perfect square matrix in row-major order.
 * @param wires Wires the gate applies to.
 * @param inverse Indicate whether inverse should be taken.
 */
template <class PrecisionT, bool inverse = false> struct twoQubitOpFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    Kokkos::View<Kokkos::complex<PrecisionT> *> matrix;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    twoQubitOpFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const Kokkos::View<Kokkos::complex<PrecisionT> *> &matrix_,
        const std::vector<size_t> &wires) {
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
        matrix = matrix_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        if constexpr (inverse) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const Kokkos::complex<PrecisionT> v00 = arr[i00];
            const Kokkos::complex<PrecisionT> v01 = arr[i01];
            const Kokkos::complex<PrecisionT> v10 = arr[i10];
            const Kokkos::complex<PrecisionT> v11 = arr[i11];

            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i00] = conj(matrix[0b0000]) * v00 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b0100]) * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1000]) * v10 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1100]) * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i01] = conj(matrix[0b0001]) * v00 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b0101]) * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1001]) * v10 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1101]) * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i10] = conj(matrix[0b0010]) * v00 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b0110]) * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1010]) * v10 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1110]) * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i11] = conj(matrix[0b0011]) * v00 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b0111]) * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1011]) * v10 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       conj(matrix[0b1111]) * v11;
        } else {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const Kokkos::complex<PrecisionT> v00 = arr[i00];
            const Kokkos::complex<PrecisionT> v01 = arr[i01];
            const Kokkos::complex<PrecisionT> v10 = arr[i10];
            const Kokkos::complex<PrecisionT> v11 = arr[i11];

            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i00] = matrix[0b0000] * v00 + matrix[0b0001] * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       matrix[0b0010] * v10 + matrix[0b0011] * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i01] = matrix[0b0100] * v00 + matrix[0b0101] * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       matrix[0b0110] * v10 + matrix[0b0111] * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i10] = matrix[0b1000] * v00 + matrix[0b1001] * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       matrix[0b1010] * v10 + matrix[0b1011] * v11;
            // NOLINTNEXTLINE(readability-magic-numbers)
            arr[i11] = matrix[0b1100] * v00 + matrix[0b1101] * v01 +
                       // NOLINTNEXTLINE(readability-magic-numbers)
                       matrix[0b1110] * v10 + matrix[0b1111] * v11;
            // }
        }
    }
};

template <class Precision, bool inverse = false> struct multiQubitOpFunctor {

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
    std::size_t dim;
    std::size_t num_qubits;

    multiQubitOpFunctor(KokkosComplexVector &arr_, std::size_t num_qubits_,
                        const KokkosComplexVector &matrix_,
                        KokkosIntVector &wires_) {
        dim = 1U << wires_.size();
        num_qubits = num_qubits_;
        wires = wires_;
        arr = arr_;
        matrix = matrix_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &teamMember) const {
        const std::size_t k = teamMember.league_rank() * dim;
        ScratchViewComplex coeffs_in(teamMember.team_scratch(0), dim);
        ScratchViewSizeT indices(teamMember.team_scratch(0), dim);
        if constexpr (inverse) {

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

                        indices(inner_idx) = idx;
                        coeffs_in(inner_idx) = arr(idx);
                    });
            }
            teamMember.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, dim),
                [&](const std::size_t i) {
                    const auto idx = indices[i];
                    arr(idx) = 0.0;

                    for (size_t j = 0; j < dim; j++) {
                        const std::size_t base_idx = j * dim;
                        arr(idx) +=
                            Kokkos::conj(matrix[base_idx + i]) * coeffs_in[j];
                    }
                });
        } else {
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

                        indices(inner_idx) = idx;
                        coeffs_in(inner_idx) = arr(idx);
                    });
            }
            teamMember.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, dim),
                                 [&](const std::size_t i) {
                                     const auto idx = indices[i];
                                     arr(idx) = 0.0;
                                     const std::size_t base_idx = i * dim;

                                     for (std::size_t j = 0; j < dim; j++) {
                                         arr(idx) += matrix(base_idx + j) *
                                                     coeffs_in(j);
                                     }
                                 });
        }
    }
};

template <class PrecisionT, bool inverse = false> struct phaseShiftFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;
    Kokkos::complex<PrecisionT> s;

    phaseShiftFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                      size_t num_qubits, const std::vector<size_t> &wires,
                      const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const PrecisionT &angle = params[0];

        s = inverse ? exp(-Kokkos::complex<PrecisionT>(0, angle))
                    : exp(Kokkos::complex<PrecisionT>(0, angle));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        arr[i1] *= s;
    }
};

template <class PrecisionT, bool inverse = false> struct rxFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;
    PrecisionT c;
    PrecisionT s;
    rxFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              size_t num_qubits, const std::vector<size_t> &wires,
              const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const PrecisionT &angle = params[0];
        c = cos(angle * static_cast<PrecisionT>(0.5));
        s = (inverse) ? sin(angle * static_cast<PrecisionT>(0.5))
                      : sin(-angle * static_cast<PrecisionT>(0.5));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];

        arr[i0] =
            c * v0 + Kokkos::complex<PrecisionT>{-imag(v1) * s, real(v1) * s};
        arr[i1] =
            Kokkos::complex<PrecisionT>{-imag(v0) * s, real(v0) * s} + c * v1;
    }
};

template <class PrecisionT, bool inverse = false> struct ryFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;
    PrecisionT c;
    PrecisionT s;
    ryFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              size_t num_qubits, const std::vector<size_t> &wires,
              const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const PrecisionT &angle = params[0];
        c = cos(angle * static_cast<PrecisionT>(0.5));
        s = (inverse) ? -sin(angle * static_cast<PrecisionT>(0.5))
                      : sin(angle * static_cast<PrecisionT>(0.5));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];

        arr[i0] = Kokkos::complex<PrecisionT>{c * real(v0) - s * real(v1),
                                              c * imag(v0) - s * imag(v1)};
        arr[i1] = Kokkos::complex<PrecisionT>{s * real(v0) + c * real(v1),
                                              s * imag(v0) + c * imag(v1)};
    }
};

template <class PrecisionT, bool inverse = false> struct rzFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;
    Kokkos::complex<PrecisionT> shift_0;
    Kokkos::complex<PrecisionT> shift_1;

    rzFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              size_t num_qubits, const std::vector<size_t> &wires,
              const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const PrecisionT &angle = params[0];
        PrecisionT cos_angle = cos(angle * static_cast<PrecisionT>(0.5));
        PrecisionT sin_angle = sin(angle * static_cast<PrecisionT>(0.5));
        Kokkos::complex<PrecisionT> first{cos_angle, -sin_angle};
        Kokkos::complex<PrecisionT> second{cos_angle, sin_angle};
        shift_0 = (inverse) ? Kokkos::conj(first) : first;
        shift_1 = (inverse) ? Kokkos::conj(second) : second;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        arr[i0] *= shift_0;
        arr[i1] *= shift_1;
    }
};

template <class PrecisionT, bool inverse = false> struct cRotFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    Kokkos::complex<PrecisionT> rot_mat_0b00;
    Kokkos::complex<PrecisionT> rot_mat_0b10;
    Kokkos::complex<PrecisionT> rot_mat_0b01;
    Kokkos::complex<PrecisionT> rot_mat_0b11;

    cRotFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                size_t num_qubits, const std::vector<size_t> &wires,
                const std::vector<PrecisionT> &params) {

        const PrecisionT phi = (inverse) ? -params[2] : params[0];
        const PrecisionT theta = (inverse) ? -params[1] : params[1];
        const PrecisionT omega = (inverse) ? -params[0] : params[2];
        const PrecisionT c = std::cos(theta / 2);
        const PrecisionT s = std::sin(theta / 2);
        const PrecisionT p{phi + omega};
        const PrecisionT m{phi - omega};

        auto imag = Kokkos::complex<PrecisionT>(0, 1);
        rot_mat_0b00 =
            Kokkos::exp(static_cast<PrecisionT>(p / 2) * (-imag)) * c;
        rot_mat_0b01 = -Kokkos::exp(static_cast<PrecisionT>(m / 2) * imag) * s;
        rot_mat_0b10 =
            Kokkos::exp(static_cast<PrecisionT>(m / 2) * (-imag)) * s;
        rot_mat_0b11 = Kokkos::exp(static_cast<PrecisionT>(p / 2) * imag) * c;

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
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const auto v0 = arr[i10];
        const auto v1 = arr[i11];

        arr[i10] = rot_mat_0b00 * v0 + rot_mat_0b01 * v1;
        arr[i11] = rot_mat_0b10 * v0 + rot_mat_0b11 * v1;
    }
};

template <class PrecisionT, bool inverse = false> struct isingXXFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;

    isingXXFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<PrecisionT> &params) {
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

        const PrecisionT &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v00 = arr[i00];
        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr[i00] = Kokkos::complex<PrecisionT>{cr * real(v00) + sj * imag(v11),
                                               cr * imag(v00) - sj * real(v11)};
        arr[i01] = Kokkos::complex<PrecisionT>{cr * real(v01) + sj * imag(v10),
                                               cr * imag(v01) - sj * real(v10)};
        arr[i10] = Kokkos::complex<PrecisionT>{cr * real(v10) + sj * imag(v01),
                                               cr * imag(v10) - sj * real(v01)};
        arr[i11] = Kokkos::complex<PrecisionT>{cr * real(v11) + sj * imag(v00),
                                               cr * imag(v11) - sj * real(v00)};
    }
};

template <class PrecisionT, bool inverse = false> struct isingXYFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;

    isingXYFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<PrecisionT> &params) {
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

        const PrecisionT &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v00 = arr[i00];
        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr[i00] = Kokkos::complex<PrecisionT>{real(v00), imag(v00)};
        arr[i01] = Kokkos::complex<PrecisionT>{cr * real(v01) - sj * imag(v10),
                                               cr * imag(v01) + sj * real(v10)};
        arr[i10] = Kokkos::complex<PrecisionT>{cr * real(v10) - sj * imag(v01),
                                               cr * imag(v10) + sj * real(v01)};
        arr[i11] = Kokkos::complex<PrecisionT>{real(v11), imag(v11)};
    }
};

template <class PrecisionT, bool inverse = false> struct isingYYFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;

    isingYYFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
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

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v00 = arr[i00];
        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr[i00] = Kokkos::complex<PrecisionT>{cr * real(v00) - sj * imag(v11),
                                               cr * imag(v00) + sj * real(v11)};
        arr[i01] = Kokkos::complex<PrecisionT>{cr * real(v01) + sj * imag(v10),
                                               cr * imag(v01) - sj * real(v10)};
        arr[i10] = Kokkos::complex<PrecisionT>{cr * real(v10) + sj * imag(v01),
                                               cr * imag(v10) - sj * real(v01)};
        arr[i11] = Kokkos::complex<PrecisionT>{cr * real(v11) - sj * imag(v00),
                                               cr * imag(v11) + sj * real(v00)};
    }
};

template <class PrecisionT, bool inverse = false> struct isingZZFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    Kokkos::complex<PrecisionT> first;
    Kokkos::complex<PrecisionT> second;
    Kokkos::complex<PrecisionT> shift_0;
    Kokkos::complex<PrecisionT> shift_1;

    isingZZFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];

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

        first = Kokkos::complex<PrecisionT>{std::cos(angle / 2),
                                            -std::sin(angle / 2)};
        second = Kokkos::complex<PrecisionT>{std::cos(angle / 2),
                                             std::sin(angle / 2)};

        shift_0 = (inverse) ? conj(first) : first;
        shift_1 = (inverse) ? conj(second) : second;

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] *= shift_0;
        arr[i01] *= shift_1;
        arr[i10] *= shift_1;
        arr[i11] *= shift_0;
    }
};
template <class PrecisionT, bool inverse = false>
struct singleExcitationFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;

    singleExcitationFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                            size_t num_qubits, const std::vector<size_t> &wires,
                            const std::vector<PrecisionT> &params) {
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

        const PrecisionT &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;

        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];

        arr[i01] = cr * v01 - sj * v10;
        arr[i10] = sj * v01 + cr * v10;
    }
};

template <class PrecisionT, bool inverse = false>
struct singleExcitationMinusFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;
    Kokkos::complex<PrecisionT> e;

    singleExcitationMinusFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const std::vector<size_t> &wires,
        const std::vector<PrecisionT> &params) {
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

        const PrecisionT &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<PrecisionT>(0, angle / 2))
                    : exp(Kokkos::complex<PrecisionT>(0, -angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];

        arr[i00] *= e;
        arr[i01] = cr * v01 - sj * v10;
        arr[i10] = sj * v01 + cr * v10;
        arr[i11] *= e;
    }
};

template <class PrecisionT, bool inverse = false>
struct singleExcitationPlusFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;
    Kokkos::complex<PrecisionT> e;

    singleExcitationPlusFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const std::vector<size_t> &wires,
        const std::vector<PrecisionT> &params) {
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

        const PrecisionT &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<PrecisionT>(0, -angle / 2))
                    : exp(Kokkos::complex<PrecisionT>(0, angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];

        arr[i00] *= e;
        arr[i01] = cr * v01 - sj * v10;
        arr[i10] = sj * v01 + cr * v10;
        arr[i11] *= e;
    }
};

template <class PrecisionT, bool inverse = false>
struct doubleExcitationFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire2;
    size_t rev_wire3;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire2_shift;
    size_t rev_wire3_shift;
    size_t rev_wire_min;
    size_t rev_wire_min_mid;
    size_t rev_wire_max_mid;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;
    size_t parity_hmiddle;
    size_t parity_lmiddle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;
    Kokkos::complex<PrecisionT> shifts_2;
    Kokkos::complex<PrecisionT> shifts_3;

    PrecisionT cr;
    PrecisionT sj;

    doubleExcitationFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                            size_t num_qubits, const std::vector<size_t> &wires,
                            const std::vector<PrecisionT> &params) {

        const PrecisionT &angle = params[0];
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                size_t tmp = rev_wire_min_mid;
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

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0000 = ((k << 4U) & parity_high) |
                             ((k << 3U) & parity_hmiddle) |
                             ((k << 2U) & parity_middle) |
                             ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;

        const Kokkos::complex<PrecisionT> v3 = arr[i0011];
        const Kokkos::complex<PrecisionT> v12 = arr[i1100];

        arr[i0011] = cr * v3 - sj * v12;
        arr[i1100] = sj * v3 + cr * v12;
    }
};

template <class PrecisionT, bool inverse = false>
struct doubleExcitationMinusFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire2;
    size_t rev_wire3;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire2_shift;
    size_t rev_wire3_shift;
    size_t rev_wire_min;
    size_t rev_wire_min_mid;
    size_t rev_wire_max_mid;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;
    size_t parity_hmiddle;
    size_t parity_lmiddle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;
    Kokkos::complex<PrecisionT> shifts_2;
    Kokkos::complex<PrecisionT> shifts_3;

    PrecisionT cr;
    PrecisionT sj;
    Kokkos::complex<PrecisionT> e;

    doubleExcitationMinusFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const std::vector<size_t> &wires,
        const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                size_t tmp = rev_wire_min_mid;
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

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<PrecisionT>(0, angle / 2))
                    : exp(Kokkos::complex<PrecisionT>(0, -angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0000 = ((k << 4U) & parity_high) |
                             ((k << 3U) & parity_hmiddle) |
                             ((k << 2U) & parity_middle) |
                             ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const size_t i0001 = i0000 | rev_wire0_shift;
        const size_t i0010 = i0000 | rev_wire1_shift;
        const size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const size_t i0100 = i0000 | rev_wire2_shift;
        const size_t i0101 = i0000 | rev_wire2_shift | rev_wire0_shift;
        const size_t i0110 = i0000 | rev_wire2_shift | rev_wire1_shift;
        const size_t i0111 =
            i0000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const size_t i1000 = i0000 | rev_wire3_shift;
        const size_t i1001 = i0000 | rev_wire3_shift | rev_wire0_shift;
        const size_t i1010 = i0000 | rev_wire3_shift | rev_wire1_shift;
        const size_t i1011 =
            i0000 | rev_wire3_shift | rev_wire1_shift | rev_wire0_shift;
        const size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;
        const size_t i1101 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire0_shift;
        const size_t i1110 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire1_shift;
        const size_t i1111 = i0000 | rev_wire3_shift | rev_wire2_shift |
                             rev_wire1_shift | rev_wire0_shift;

        const Kokkos::complex<PrecisionT> v3 = arr[i0011];
        const Kokkos::complex<PrecisionT> v12 = arr[i1100];

        arr[i0000] *= e;
        arr[i0001] *= e;
        arr[i0010] *= e;
        arr[i0011] = cr * v3 - sj * v12;
        arr[i0100] *= e;
        arr[i0101] *= e;
        arr[i0110] *= e;
        arr[i0111] *= e;
        arr[i1000] *= e;
        arr[i1001] *= e;
        arr[i1010] *= e;
        arr[i1011] *= e;
        arr[i1100] = sj * v3 + cr * v12;
        arr[i1101] *= e;
        arr[i1110] *= e;
        arr[i1111] *= e;
    }
};

template <class PrecisionT, bool inverse = false>
struct doubleExcitationPlusFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire2;
    size_t rev_wire3;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire2_shift;
    size_t rev_wire3_shift;
    size_t rev_wire_min;
    size_t rev_wire_min_mid;
    size_t rev_wire_max_mid;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;
    size_t parity_hmiddle;
    size_t parity_lmiddle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;
    Kokkos::complex<PrecisionT> shifts_2;
    Kokkos::complex<PrecisionT> shifts_3;

    PrecisionT cr;
    PrecisionT sj;
    Kokkos::complex<PrecisionT> e;

    doubleExcitationPlusFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const std::vector<size_t> &wires,
        const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                size_t tmp = rev_wire_min_mid;
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

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<PrecisionT>(0, -angle / 2))
                    : exp(Kokkos::complex<PrecisionT>(0, angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {

        const size_t i0000 = ((k << 4U) & parity_high) |
                             ((k << 3U) & parity_hmiddle) |
                             ((k << 2U) & parity_middle) |
                             ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const size_t i0001 = i0000 | rev_wire0_shift;
        const size_t i0010 = i0000 | rev_wire1_shift;
        const size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const size_t i0100 = i0000 | rev_wire2_shift;
        const size_t i0101 = i0000 | rev_wire2_shift | rev_wire0_shift;
        const size_t i0110 = i0000 | rev_wire2_shift | rev_wire1_shift;
        const size_t i0111 =
            i0000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const size_t i1000 = i0000 | rev_wire3_shift;
        const size_t i1001 = i0000 | rev_wire3_shift | rev_wire0_shift;
        const size_t i1010 = i0000 | rev_wire3_shift | rev_wire1_shift;
        const size_t i1011 =
            i0000 | rev_wire3_shift | rev_wire1_shift | rev_wire0_shift;
        const size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;
        const size_t i1101 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire0_shift;
        const size_t i1110 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire1_shift;
        const size_t i1111 = i0000 | rev_wire3_shift | rev_wire2_shift |
                             rev_wire1_shift | rev_wire0_shift;

        const Kokkos::complex<PrecisionT> v3 = arr[i0011];
        const Kokkos::complex<PrecisionT> v12 = arr[i1100];

        arr[i0000] *= e;
        arr[i0001] *= e;
        arr[i0010] *= e;
        arr[i0011] = cr * v3 - sj * v12;
        arr[i0100] *= e;
        arr[i0101] *= e;
        arr[i0110] *= e;
        arr[i0111] *= e;
        arr[i1000] *= e;
        arr[i1001] *= e;
        arr[i1010] *= e;
        arr[i1011] *= e;
        arr[i1100] = sj * v3 + cr * v12;
        arr[i1101] *= e;
        arr[i1110] *= e;
        arr[i1111] *= e;
    }
};

template <class PrecisionT, bool inverse = false>
struct controlledPhaseShiftFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    Kokkos::complex<PrecisionT> s;

    controlledPhaseShiftFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const std::vector<size_t> &wires,
        const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
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

        s = inverse ? exp(-Kokkos::complex<PrecisionT>(0, angle))
                    : exp(Kokkos::complex<PrecisionT>(0, angle));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;

        arr[i11] *= s;
    }
};

template <class PrecisionT, bool inverse = false> struct crxFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT c;
    PrecisionT js;

    crxFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               size_t num_qubits, const std::vector<size_t> &wires,
               const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
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

        c = std::cos(angle / 2);
        js = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr[i10] = Kokkos::complex<PrecisionT>{
            c * Kokkos::real(v10) + js * Kokkos::imag(v11),
            c * Kokkos::imag(v10) - js * Kokkos::real(v11)};
        arr[i11] = Kokkos::complex<PrecisionT>{
            c * Kokkos::real(v11) + js * Kokkos::imag(v10),
            c * Kokkos::imag(v11) - js * Kokkos::real(v10)};
    }
};

template <class PrecisionT, bool inverse = false> struct cryFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT c;
    PrecisionT s;

    cryFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               size_t num_qubits, const std::vector<size_t> &wires,
               const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
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

        c = std::cos(angle / 2);
        s = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr[i10] = c * v10 - s * v11;
        arr[i11] = s * v10 + c * v11;
    }
};

template <class PrecisionT, bool inverse = false> struct crzFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;

    crzFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               size_t num_qubits, const std::vector<size_t> &wires,
               const std::vector<PrecisionT> &params) {
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

        const PrecisionT &angle = params[0];

        const Kokkos::complex<PrecisionT> first = Kokkos::complex<PrecisionT>{
            std::cos(angle / 2), -std::sin(angle / 2)};
        const Kokkos::complex<PrecisionT> second = Kokkos::complex<PrecisionT>{
            std::cos(angle / 2), std::sin(angle / 2)};

        shifts_0 = (inverse) ? conj(first) : first;
        shifts_1 = (inverse) ? conj(second) : second;

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i10] *= shifts_0;
        arr[i11] *= shifts_1;
    }
};

template <class PrecisionT, bool inverse = false> struct multiRZFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t wires_parity;

    Kokkos::complex<PrecisionT> shift_0;
    Kokkos::complex<PrecisionT> shift_1;

    multiRZFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<PrecisionT> &params) {

        const PrecisionT &angle = params[0];
        const Kokkos::complex<PrecisionT> first = Kokkos::complex<PrecisionT>{
            std::cos(angle / 2), -std::sin(angle / 2)};
        const Kokkos::complex<PrecisionT> second = Kokkos::complex<PrecisionT>{
            std::cos(angle / 2), std::sin(angle / 2)};

        shift_0 = (inverse) ? conj(first) : first;
        shift_1 = (inverse) ? conj(second) : second;

        size_t wires_parity_ = 0U;
        for (size_t wire : wires) {
            wires_parity_ |=
                (static_cast<size_t>(1U) << (num_qubits - wire - 1));
        }

        wires_parity = wires_parity_;
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        arr[k] *= (Kokkos::Impl::bit_count(k & wires_parity) % 2 == 0)
                      ? shift_0
                      : shift_1;
    }
};

template <class PrecisionT, bool inverse = false> struct rotFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    Kokkos::complex<PrecisionT> rot_mat_0b00;
    Kokkos::complex<PrecisionT> rot_mat_0b10;
    Kokkos::complex<PrecisionT> rot_mat_0b01;
    Kokkos::complex<PrecisionT> rot_mat_0b11;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

    rotFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               size_t num_qubits, const std::vector<size_t> &wires,
               const std::vector<PrecisionT> &params) {
        const PrecisionT phi = (inverse) ? -params[2] : params[0];
        const PrecisionT theta = (inverse) ? -params[1] : params[1];
        const PrecisionT omega = (inverse) ? -params[0] : params[2];
        const PrecisionT c = std::cos(theta / 2);
        const PrecisionT s = std::sin(theta / 2);
        const PrecisionT p{phi + omega};
        const PrecisionT m{phi - omega};

        auto imag = Kokkos::complex<PrecisionT>(0, 1);
        rot_mat_0b00 =
            Kokkos::exp(static_cast<PrecisionT>(p / 2) * (-imag)) * c;
        rot_mat_0b01 = -Kokkos::exp(static_cast<PrecisionT>(m / 2) * imag) * s;
        rot_mat_0b10 =
            Kokkos::exp(static_cast<PrecisionT>(m / 2) * (-imag)) * s;
        rot_mat_0b11 = Kokkos::exp(static_cast<PrecisionT>(p / 2) * imag) * c;

        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        const Kokkos::complex<PrecisionT> v0 = arr[i0];
        const Kokkos::complex<PrecisionT> v1 = arr[i1];
        arr[i0] = rot_mat_0b00 * v0 +
                  rot_mat_0b01 * v1; // NOLINT(readability-magic-numbers)
        arr[i1] = rot_mat_0b10 * v0 +
                  rot_mat_0b11 * v1; // NOLINT(readability-magic-numbers)
    }
};

} // namespace Pennylane::LightningKokkos::Functors
