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

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using Kokkos::Experimental::swap;
using std::size_t;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Functors {

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
                        const std::vector<std::size_t> &wires_) {
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_host.size());
        Kokkos::deep_copy(wires, wires_host);
        dim = 1U << wires_.size();
        num_qubits = num_qubits_;
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

template <class PrecisionT, std::size_t n_wires, bool inverse = false>
struct apply1QubitOpFunctor {

    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    std::size_t dim;
    std::size_t num_qubits;
    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

    apply1QubitOpFunctor(
        KokkosComplexVector &arr_, std::size_t num_qubits_,
        const KokkosComplexVector &matrix_,
        [[maybe_unused]] const std::vector<std::size_t> &wires_) {
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        dim = 1U << n_wires;

        rev_wire = num_qubits - wires_[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        const Kokkos::complex<PrecisionT> v0 = arr[i0];
        const Kokkos::complex<PrecisionT> v1 = arr[i1];

        if constexpr (inverse) {
            arr(i0) = conj(matrix(0B00)) * v0 + conj(matrix(0B10)) * v1;
            arr(i1) = conj(matrix(0B01)) * v0 + conj(matrix(0B11)) * v1;
        } else {
            arr(i0) = matrix(0B00) * v0 + matrix(0B01) * v1;
            arr(i1) = matrix(0B10) * v0 + matrix(0B11) * v1;
        }
    }
};

template <class PrecisionT, std::size_t n_wires, bool inverse = false>
struct apply2QubitOpFunctor {

    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    std::size_t dim;
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

    apply2QubitOpFunctor(
        KokkosComplexVector &arr_, std::size_t num_qubits_,
        const KokkosComplexVector &matrix_,
        [[maybe_unused]] const std::vector<std::size_t> &wires_) {
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        dim = 1U << n_wires;

        rev_wire0 = num_qubits - wires_[1] - 1;
        rev_wire1 = num_qubits - wires_[0] - 1; // Control qubit
        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
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

        if constexpr (inverse) {
            arr(i00) = conj(matrix(0B0000)) * v00 + conj(matrix(0B0100)) * v01 +
                       conj(matrix(0B1000)) * v10 + conj(matrix(0B1100)) * v11;
            arr(i01) = conj(matrix(0B0001)) * v00 + conj(matrix(0B0101)) * v01 +
                       conj(matrix(0B1001)) * v10 + conj(matrix(0B1101)) * v11;
            arr(i10) = conj(matrix(0B0010)) * v00 + conj(matrix(0B0110)) * v01 +
                       conj(matrix(0B1010)) * v10 + conj(matrix(0B1110)) * v11;
            arr(i11) = conj(matrix(0B0011)) * v00 + conj(matrix(0B0111)) * v01 +
                       conj(matrix(0B1011)) * v10 + conj(matrix(0B1111)) * v11;
        } else {
            arr(i00) = matrix(0B0000) * v00 + matrix(0B0001) * v01 +
                       matrix(0B0010) * v10 + matrix(0B0011) * v11;
            arr(i01) = matrix(0B0100) * v00 + matrix(0B0101) * v01 +
                       matrix(0B0110) * v10 + matrix(0B0111) * v11;
            arr(i10) = matrix(0B1000) * v00 + matrix(0B1001) * v01 +
                       matrix(0B1010) * v10 + matrix(0B1011) * v11;
            arr(i11) = matrix(0B1100) * v00 + matrix(0B1101) * v01 +
                       matrix(0B1110) * v10 + matrix(0B1111) * v11;
        }
    }
};

template <class PrecisionT, std::size_t n_wires, bool inverse = false>
struct apply3QubitOpFunctor {

    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    std::size_t dim;
    std::size_t num_qubits;

    apply3QubitOpFunctor(KokkosComplexVector &arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_host.size());
        Kokkos::deep_copy(wires, wires_host);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        dim = 1U << n_wires;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t kdim = k * dim;

        std::size_t i000 = kdim | 0;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i000 >> (n_wires - pos - 1)) ^
                             (i000 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i000 = i000 ^ ((x << (n_wires - pos - 1)) |
                           (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v000 = arr(i000);

        std::size_t i001 = kdim | 1;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i001 >> (n_wires - pos - 1)) ^
                             (i001 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i001 = i001 ^ ((x << (n_wires - pos - 1)) |
                           (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v001 = arr(i001);

        std::size_t i010 = kdim | 2;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i010 >> (n_wires - pos - 1)) ^
                             (i010 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i010 = i010 ^ ((x << (n_wires - pos - 1)) |
                           (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v010 = arr(i010);

        std::size_t i011 = kdim | 3;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i011 >> (n_wires - pos - 1)) ^
                             (i011 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i011 = i011 ^ ((x << (n_wires - pos - 1)) |
                           (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v011 = arr(i011);

        std::size_t i100 = kdim | 4;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i100 >> (n_wires - pos - 1)) ^
                             (i100 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i100 = i100 ^ ((x << (n_wires - pos - 1)) |
                           (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v100 = arr(i100);

        std::size_t i101 = kdim | 5;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i101 >> (n_wires - pos - 1)) ^
                             (i101 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i101 = i101 ^ ((x << (n_wires - pos - 1)) |
                           (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v101 = arr(i101);

        std::size_t i110 = kdim | 6;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i110 >> (n_wires - pos - 1)) ^
                             (i110 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i110 = i110 ^ ((x << (n_wires - pos - 1)) |
                           (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v110 = arr(i110);

        std::size_t i111 = kdim | 7;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i111 >> (n_wires - pos - 1)) ^
                             (i111 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i111 = i111 ^ ((x << (n_wires - pos - 1)) |
                           (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v111 = arr(i111);

        if constexpr (inverse) {
            arr(i000) =
                conj(matrix(0B000000)) * v000 + conj(matrix(0B001000)) * v001 +
                conj(matrix(0B010000)) * v010 + conj(matrix(0B011000)) * v011 +
                conj(matrix(0B100000)) * v100 + conj(matrix(0B101000)) * v101 +
                conj(matrix(0B110000)) * v110 + conj(matrix(0B111000)) * v111;
            arr(i001) =
                conj(matrix(0B000001)) * v000 + conj(matrix(0B001001)) * v001 +
                conj(matrix(0B010001)) * v010 + conj(matrix(0B011001)) * v011 +
                conj(matrix(0B100001)) * v100 + conj(matrix(0B101001)) * v101 +
                conj(matrix(0B110001)) * v110 + conj(matrix(0B111001)) * v111;
            arr(i010) =
                conj(matrix(0B000010)) * v000 + conj(matrix(0B001010)) * v001 +
                conj(matrix(0B010010)) * v010 + conj(matrix(0B011010)) * v011 +
                conj(matrix(0B100010)) * v100 + conj(matrix(0B101010)) * v101 +
                conj(matrix(0B110010)) * v110 + conj(matrix(0B111010)) * v111;
            arr(i011) =
                conj(matrix(0B000011)) * v000 + conj(matrix(0B001011)) * v001 +
                conj(matrix(0B010011)) * v010 + conj(matrix(0B011011)) * v011 +
                conj(matrix(0B100011)) * v100 + conj(matrix(0B101011)) * v101 +
                conj(matrix(0B110011)) * v110 + conj(matrix(0B111011)) * v111;
            arr(i100) =
                conj(matrix(0B000100)) * v000 + conj(matrix(0B001100)) * v001 +
                conj(matrix(0B010100)) * v010 + conj(matrix(0B011100)) * v011 +
                conj(matrix(0B100100)) * v100 + conj(matrix(0B101100)) * v101 +
                conj(matrix(0B110100)) * v110 + conj(matrix(0B111100)) * v111;
            arr(i101) =
                conj(matrix(0B000101)) * v000 + conj(matrix(0B001101)) * v001 +
                conj(matrix(0B010101)) * v010 + conj(matrix(0B011101)) * v011 +
                conj(matrix(0B100101)) * v100 + conj(matrix(0B101101)) * v101 +
                conj(matrix(0B110101)) * v110 + conj(matrix(0B111101)) * v111;
            arr(i110) =
                conj(matrix(0B000110)) * v000 + conj(matrix(0B001110)) * v001 +
                conj(matrix(0B010110)) * v010 + conj(matrix(0B011110)) * v011 +
                conj(matrix(0B100110)) * v100 + conj(matrix(0B101110)) * v101 +
                conj(matrix(0B110110)) * v110 + conj(matrix(0B111110)) * v111;
            arr(i111) =
                conj(matrix(0B000111)) * v000 + conj(matrix(0B001111)) * v001 +
                conj(matrix(0B010111)) * v010 + conj(matrix(0B011111)) * v011 +
                conj(matrix(0B100111)) * v100 + conj(matrix(0B101111)) * v101 +
                conj(matrix(0B110111)) * v110 + conj(matrix(0B111111)) * v111;
        } else {
            arr(i000) = matrix(0B000000) * v000 + matrix(0B000001) * v001 +
                        matrix(0B000010) * v010 + matrix(0B000011) * v011 +
                        matrix(0B000100) * v100 + matrix(0B000101) * v101 +
                        matrix(0B000110) * v110 + matrix(0B000111) * v111;
            arr(i001) = matrix(0B001000) * v000 + matrix(0B001001) * v001 +
                        matrix(0B001010) * v010 + matrix(0B001011) * v011 +
                        matrix(0B001100) * v100 + matrix(0B001101) * v101 +
                        matrix(0B001110) * v110 + matrix(0B001111) * v111;
            arr(i010) = matrix(0B010000) * v000 + matrix(0B010001) * v001 +
                        matrix(0B010010) * v010 + matrix(0B010011) * v011 +
                        matrix(0B010100) * v100 + matrix(0B010101) * v101 +
                        matrix(0B010110) * v110 + matrix(0B010111) * v111;
            arr(i011) = matrix(0B011000) * v000 + matrix(0B011001) * v001 +
                        matrix(0B011010) * v010 + matrix(0B011011) * v011 +
                        matrix(0B011100) * v100 + matrix(0B011101) * v101 +
                        matrix(0B011110) * v110 + matrix(0B011111) * v111;
            arr(i100) = matrix(0B100000) * v000 + matrix(0B100001) * v001 +
                        matrix(0B100010) * v010 + matrix(0B100011) * v011 +
                        matrix(0B100100) * v100 + matrix(0B100101) * v101 +
                        matrix(0B100110) * v110 + matrix(0B100111) * v111;
            arr(i101) = matrix(0B101000) * v000 + matrix(0B101001) * v001 +
                        matrix(0B101010) * v010 + matrix(0B101011) * v011 +
                        matrix(0B101100) * v100 + matrix(0B101101) * v101 +
                        matrix(0B101110) * v110 + matrix(0B101111) * v111;
            arr(i110) = matrix(0B110000) * v000 + matrix(0B110001) * v001 +
                        matrix(0B110010) * v010 + matrix(0B110011) * v011 +
                        matrix(0B110100) * v100 + matrix(0B110101) * v101 +
                        matrix(0B110110) * v110 + matrix(0B110111) * v111;
            arr(i111) = matrix(0B111000) * v000 + matrix(0B111001) * v001 +
                        matrix(0B111010) * v010 + matrix(0B111011) * v011 +
                        matrix(0B111100) * v100 + matrix(0B111101) * v101 +
                        matrix(0B111110) * v110 + matrix(0B111111) * v111;
        }
    }
};

template <class PrecisionT, std::size_t n_wires, bool inverse = false>
struct apply4QubitOpFunctor {

    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    std::size_t dim;
    std::size_t num_qubits;

    apply4QubitOpFunctor(KokkosComplexVector &arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_host.size());
        Kokkos::deep_copy(wires, wires_host);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        dim = 1U << n_wires;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t kdim = k * dim;

        std::size_t i0000 = kdim | 0;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i0000 >> (n_wires - pos - 1)) ^
                             (i0000 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i0000 = i0000 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v0000 = arr(i0000);

        std::size_t i0001 = kdim | 1;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i0001 >> (n_wires - pos - 1)) ^
                             (i0001 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i0001 = i0001 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v0001 = arr(i0001);

        std::size_t i0010 = kdim | 2;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i0010 >> (n_wires - pos - 1)) ^
                             (i0010 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i0010 = i0010 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v0010 = arr(i0010);

        std::size_t i0011 = kdim | 3;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i0011 >> (n_wires - pos - 1)) ^
                             (i0011 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i0011 = i0011 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v0011 = arr(i0011);

        std::size_t i0100 = kdim | 4;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i0100 >> (n_wires - pos - 1)) ^
                             (i0100 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i0100 = i0100 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v0100 = arr(i0100);

        std::size_t i0101 = kdim | 5;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i0101 >> (n_wires - pos - 1)) ^
                             (i0101 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i0101 = i0101 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v0101 = arr(i0101);

        std::size_t i0110 = kdim | 6;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i0110 >> (n_wires - pos - 1)) ^
                             (i0110 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i0110 = i0110 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v0110 = arr(i0110);

        std::size_t i0111 = kdim | 7;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i0111 >> (n_wires - pos - 1)) ^
                             (i0111 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i0111 = i0111 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v0111 = arr(i0111);

        std::size_t i1000 = kdim | 8;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i1000 >> (n_wires - pos - 1)) ^
                             (i1000 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i1000 = i1000 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v1000 = arr(i1000);

        std::size_t i1001 = kdim | 9;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i1001 >> (n_wires - pos - 1)) ^
                             (i1001 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i1001 = i1001 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v1001 = arr(i1001);

        std::size_t i1010 = kdim | 10;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i1010 >> (n_wires - pos - 1)) ^
                             (i1010 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i1010 = i1010 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v1010 = arr(i1010);

        std::size_t i1011 = kdim | 11;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i1011 >> (n_wires - pos - 1)) ^
                             (i1011 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i1011 = i1011 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v1011 = arr(i1011);

        std::size_t i1100 = kdim | 12;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i1100 >> (n_wires - pos - 1)) ^
                             (i1100 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i1100 = i1100 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v1100 = arr(i1100);

        std::size_t i1101 = kdim | 13;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i1101 >> (n_wires - pos - 1)) ^
                             (i1101 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i1101 = i1101 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v1101 = arr(i1101);

        std::size_t i1110 = kdim | 14;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i1110 >> (n_wires - pos - 1)) ^
                             (i1110 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i1110 = i1110 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v1110 = arr(i1110);

        std::size_t i1111 = kdim | 15;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i1111 >> (n_wires - pos - 1)) ^
                             (i1111 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i1111 = i1111 ^ ((x << (n_wires - pos - 1)) |
                             (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v1111 = arr(i1111);

        if constexpr (inverse) {
            arr(i0000) = conj(matrix(0B00000000)) * v0000 +
                         conj(matrix(0B00010000)) * v0001 +
                         conj(matrix(0B00100000)) * v0010 +
                         conj(matrix(0B00110000)) * v0011 +
                         conj(matrix(0B01000000)) * v0100 +
                         conj(matrix(0B01010000)) * v0101 +
                         conj(matrix(0B01100000)) * v0110 +
                         conj(matrix(0B01110000)) * v0111 +
                         conj(matrix(0B10000000)) * v1000 +
                         conj(matrix(0B10010000)) * v1001 +
                         conj(matrix(0B10100000)) * v1010 +
                         conj(matrix(0B10110000)) * v1011 +
                         conj(matrix(0B11000000)) * v1100 +
                         conj(matrix(0B11010000)) * v1101 +
                         conj(matrix(0B11100000)) * v1110 +
                         conj(matrix(0B11110000)) * v1111;
            arr(i0001) = conj(matrix(0B00000001)) * v0000 +
                         conj(matrix(0B00010001)) * v0001 +
                         conj(matrix(0B00100001)) * v0010 +
                         conj(matrix(0B00110001)) * v0011 +
                         conj(matrix(0B01000001)) * v0100 +
                         conj(matrix(0B01010001)) * v0101 +
                         conj(matrix(0B01100001)) * v0110 +
                         conj(matrix(0B01110001)) * v0111 +
                         conj(matrix(0B10000001)) * v1000 +
                         conj(matrix(0B10010001)) * v1001 +
                         conj(matrix(0B10100001)) * v1010 +
                         conj(matrix(0B10110001)) * v1011 +
                         conj(matrix(0B11000001)) * v1100 +
                         conj(matrix(0B11010001)) * v1101 +
                         conj(matrix(0B11100001)) * v1110 +
                         conj(matrix(0B11110001)) * v1111;
            arr(i0010) = conj(matrix(0B00000010)) * v0000 +
                         conj(matrix(0B00010010)) * v0001 +
                         conj(matrix(0B00100010)) * v0010 +
                         conj(matrix(0B00110010)) * v0011 +
                         conj(matrix(0B01000010)) * v0100 +
                         conj(matrix(0B01010010)) * v0101 +
                         conj(matrix(0B01100010)) * v0110 +
                         conj(matrix(0B01110010)) * v0111 +
                         conj(matrix(0B10000010)) * v1000 +
                         conj(matrix(0B10010010)) * v1001 +
                         conj(matrix(0B10100010)) * v1010 +
                         conj(matrix(0B10110010)) * v1011 +
                         conj(matrix(0B11000010)) * v1100 +
                         conj(matrix(0B11010010)) * v1101 +
                         conj(matrix(0B11100010)) * v1110 +
                         conj(matrix(0B11110010)) * v1111;
            arr(i0011) = conj(matrix(0B00000011)) * v0000 +
                         conj(matrix(0B00010011)) * v0001 +
                         conj(matrix(0B00100011)) * v0010 +
                         conj(matrix(0B00110011)) * v0011 +
                         conj(matrix(0B01000011)) * v0100 +
                         conj(matrix(0B01010011)) * v0101 +
                         conj(matrix(0B01100011)) * v0110 +
                         conj(matrix(0B01110011)) * v0111 +
                         conj(matrix(0B10000011)) * v1000 +
                         conj(matrix(0B10010011)) * v1001 +
                         conj(matrix(0B10100011)) * v1010 +
                         conj(matrix(0B10110011)) * v1011 +
                         conj(matrix(0B11000011)) * v1100 +
                         conj(matrix(0B11010011)) * v1101 +
                         conj(matrix(0B11100011)) * v1110 +
                         conj(matrix(0B11110011)) * v1111;
            arr(i0100) = conj(matrix(0B00000100)) * v0000 +
                         conj(matrix(0B00010100)) * v0001 +
                         conj(matrix(0B00100100)) * v0010 +
                         conj(matrix(0B00110100)) * v0011 +
                         conj(matrix(0B01000100)) * v0100 +
                         conj(matrix(0B01010100)) * v0101 +
                         conj(matrix(0B01100100)) * v0110 +
                         conj(matrix(0B01110100)) * v0111 +
                         conj(matrix(0B10000100)) * v1000 +
                         conj(matrix(0B10010100)) * v1001 +
                         conj(matrix(0B10100100)) * v1010 +
                         conj(matrix(0B10110100)) * v1011 +
                         conj(matrix(0B11000100)) * v1100 +
                         conj(matrix(0B11010100)) * v1101 +
                         conj(matrix(0B11100100)) * v1110 +
                         conj(matrix(0B11110100)) * v1111;
            arr(i0101) = conj(matrix(0B00000101)) * v0000 +
                         conj(matrix(0B00010101)) * v0001 +
                         conj(matrix(0B00100101)) * v0010 +
                         conj(matrix(0B00110101)) * v0011 +
                         conj(matrix(0B01000101)) * v0100 +
                         conj(matrix(0B01010101)) * v0101 +
                         conj(matrix(0B01100101)) * v0110 +
                         conj(matrix(0B01110101)) * v0111 +
                         conj(matrix(0B10000101)) * v1000 +
                         conj(matrix(0B10010101)) * v1001 +
                         conj(matrix(0B10100101)) * v1010 +
                         conj(matrix(0B10110101)) * v1011 +
                         conj(matrix(0B11000101)) * v1100 +
                         conj(matrix(0B11010101)) * v1101 +
                         conj(matrix(0B11100101)) * v1110 +
                         conj(matrix(0B11110101)) * v1111;
            arr(i0110) = conj(matrix(0B00000110)) * v0000 +
                         conj(matrix(0B00010110)) * v0001 +
                         conj(matrix(0B00100110)) * v0010 +
                         conj(matrix(0B00110110)) * v0011 +
                         conj(matrix(0B01000110)) * v0100 +
                         conj(matrix(0B01010110)) * v0101 +
                         conj(matrix(0B01100110)) * v0110 +
                         conj(matrix(0B01110110)) * v0111 +
                         conj(matrix(0B10000110)) * v1000 +
                         conj(matrix(0B10010110)) * v1001 +
                         conj(matrix(0B10100110)) * v1010 +
                         conj(matrix(0B10110110)) * v1011 +
                         conj(matrix(0B11000110)) * v1100 +
                         conj(matrix(0B11010110)) * v1101 +
                         conj(matrix(0B11100110)) * v1110 +
                         conj(matrix(0B11110110)) * v1111;
            arr(i0111) = conj(matrix(0B00000111)) * v0000 +
                         conj(matrix(0B00010111)) * v0001 +
                         conj(matrix(0B00100111)) * v0010 +
                         conj(matrix(0B00110111)) * v0011 +
                         conj(matrix(0B01000111)) * v0100 +
                         conj(matrix(0B01010111)) * v0101 +
                         conj(matrix(0B01100111)) * v0110 +
                         conj(matrix(0B01110111)) * v0111 +
                         conj(matrix(0B10000111)) * v1000 +
                         conj(matrix(0B10010111)) * v1001 +
                         conj(matrix(0B10100111)) * v1010 +
                         conj(matrix(0B10110111)) * v1011 +
                         conj(matrix(0B11000111)) * v1100 +
                         conj(matrix(0B11010111)) * v1101 +
                         conj(matrix(0B11100111)) * v1110 +
                         conj(matrix(0B11110111)) * v1111;
            arr(i1000) = conj(matrix(0B00001000)) * v0000 +
                         conj(matrix(0B00011000)) * v0001 +
                         conj(matrix(0B00101000)) * v0010 +
                         conj(matrix(0B00111000)) * v0011 +
                         conj(matrix(0B01001000)) * v0100 +
                         conj(matrix(0B01011000)) * v0101 +
                         conj(matrix(0B01101000)) * v0110 +
                         conj(matrix(0B01111000)) * v0111 +
                         conj(matrix(0B10001000)) * v1000 +
                         conj(matrix(0B10011000)) * v1001 +
                         conj(matrix(0B10101000)) * v1010 +
                         conj(matrix(0B10111000)) * v1011 +
                         conj(matrix(0B11001000)) * v1100 +
                         conj(matrix(0B11011000)) * v1101 +
                         conj(matrix(0B11101000)) * v1110 +
                         conj(matrix(0B11111000)) * v1111;
            arr(i1001) = conj(matrix(0B00001001)) * v0000 +
                         conj(matrix(0B00011001)) * v0001 +
                         conj(matrix(0B00101001)) * v0010 +
                         conj(matrix(0B00111001)) * v0011 +
                         conj(matrix(0B01001001)) * v0100 +
                         conj(matrix(0B01011001)) * v0101 +
                         conj(matrix(0B01101001)) * v0110 +
                         conj(matrix(0B01111001)) * v0111 +
                         conj(matrix(0B10001001)) * v1000 +
                         conj(matrix(0B10011001)) * v1001 +
                         conj(matrix(0B10101001)) * v1010 +
                         conj(matrix(0B10111001)) * v1011 +
                         conj(matrix(0B11001001)) * v1100 +
                         conj(matrix(0B11011001)) * v1101 +
                         conj(matrix(0B11101001)) * v1110 +
                         conj(matrix(0B11111001)) * v1111;
            arr(i1010) = conj(matrix(0B00001010)) * v0000 +
                         conj(matrix(0B00011010)) * v0001 +
                         conj(matrix(0B00101010)) * v0010 +
                         conj(matrix(0B00111010)) * v0011 +
                         conj(matrix(0B01001010)) * v0100 +
                         conj(matrix(0B01011010)) * v0101 +
                         conj(matrix(0B01101010)) * v0110 +
                         conj(matrix(0B01111010)) * v0111 +
                         conj(matrix(0B10001010)) * v1000 +
                         conj(matrix(0B10011010)) * v1001 +
                         conj(matrix(0B10101010)) * v1010 +
                         conj(matrix(0B10111010)) * v1011 +
                         conj(matrix(0B11001010)) * v1100 +
                         conj(matrix(0B11011010)) * v1101 +
                         conj(matrix(0B11101010)) * v1110 +
                         conj(matrix(0B11111010)) * v1111;
            arr(i1011) = conj(matrix(0B00001011)) * v0000 +
                         conj(matrix(0B00011011)) * v0001 +
                         conj(matrix(0B00101011)) * v0010 +
                         conj(matrix(0B00111011)) * v0011 +
                         conj(matrix(0B01001011)) * v0100 +
                         conj(matrix(0B01011011)) * v0101 +
                         conj(matrix(0B01101011)) * v0110 +
                         conj(matrix(0B01111011)) * v0111 +
                         conj(matrix(0B10001011)) * v1000 +
                         conj(matrix(0B10011011)) * v1001 +
                         conj(matrix(0B10101011)) * v1010 +
                         conj(matrix(0B10111011)) * v1011 +
                         conj(matrix(0B11001011)) * v1100 +
                         conj(matrix(0B11011011)) * v1101 +
                         conj(matrix(0B11101011)) * v1110 +
                         conj(matrix(0B11111011)) * v1111;
            arr(i1100) = conj(matrix(0B00001100)) * v0000 +
                         conj(matrix(0B00011100)) * v0001 +
                         conj(matrix(0B00101100)) * v0010 +
                         conj(matrix(0B00111100)) * v0011 +
                         conj(matrix(0B01001100)) * v0100 +
                         conj(matrix(0B01011100)) * v0101 +
                         conj(matrix(0B01101100)) * v0110 +
                         conj(matrix(0B01111100)) * v0111 +
                         conj(matrix(0B10001100)) * v1000 +
                         conj(matrix(0B10011100)) * v1001 +
                         conj(matrix(0B10101100)) * v1010 +
                         conj(matrix(0B10111100)) * v1011 +
                         conj(matrix(0B11001100)) * v1100 +
                         conj(matrix(0B11011100)) * v1101 +
                         conj(matrix(0B11101100)) * v1110 +
                         conj(matrix(0B11111100)) * v1111;
            arr(i1101) = conj(matrix(0B00001101)) * v0000 +
                         conj(matrix(0B00011101)) * v0001 +
                         conj(matrix(0B00101101)) * v0010 +
                         conj(matrix(0B00111101)) * v0011 +
                         conj(matrix(0B01001101)) * v0100 +
                         conj(matrix(0B01011101)) * v0101 +
                         conj(matrix(0B01101101)) * v0110 +
                         conj(matrix(0B01111101)) * v0111 +
                         conj(matrix(0B10001101)) * v1000 +
                         conj(matrix(0B10011101)) * v1001 +
                         conj(matrix(0B10101101)) * v1010 +
                         conj(matrix(0B10111101)) * v1011 +
                         conj(matrix(0B11001101)) * v1100 +
                         conj(matrix(0B11011101)) * v1101 +
                         conj(matrix(0B11101101)) * v1110 +
                         conj(matrix(0B11111101)) * v1111;
            arr(i1110) = conj(matrix(0B00001110)) * v0000 +
                         conj(matrix(0B00011110)) * v0001 +
                         conj(matrix(0B00101110)) * v0010 +
                         conj(matrix(0B00111110)) * v0011 +
                         conj(matrix(0B01001110)) * v0100 +
                         conj(matrix(0B01011110)) * v0101 +
                         conj(matrix(0B01101110)) * v0110 +
                         conj(matrix(0B01111110)) * v0111 +
                         conj(matrix(0B10001110)) * v1000 +
                         conj(matrix(0B10011110)) * v1001 +
                         conj(matrix(0B10101110)) * v1010 +
                         conj(matrix(0B10111110)) * v1011 +
                         conj(matrix(0B11001110)) * v1100 +
                         conj(matrix(0B11011110)) * v1101 +
                         conj(matrix(0B11101110)) * v1110 +
                         conj(matrix(0B11111110)) * v1111;
            arr(i1111) = conj(matrix(0B00001111)) * v0000 +
                         conj(matrix(0B00011111)) * v0001 +
                         conj(matrix(0B00101111)) * v0010 +
                         conj(matrix(0B00111111)) * v0011 +
                         conj(matrix(0B01001111)) * v0100 +
                         conj(matrix(0B01011111)) * v0101 +
                         conj(matrix(0B01101111)) * v0110 +
                         conj(matrix(0B01111111)) * v0111 +
                         conj(matrix(0B10001111)) * v1000 +
                         conj(matrix(0B10011111)) * v1001 +
                         conj(matrix(0B10101111)) * v1010 +
                         conj(matrix(0B10111111)) * v1011 +
                         conj(matrix(0B11001111)) * v1100 +
                         conj(matrix(0B11011111)) * v1101 +
                         conj(matrix(0B11101111)) * v1110 +
                         conj(matrix(0B11111111)) * v1111;
        } else {
            arr(i0000) =
                matrix(0B00000000) * v0000 + matrix(0B00000001) * v0001 +
                matrix(0B00000010) * v0010 + matrix(0B00000011) * v0011 +
                matrix(0B00000100) * v0100 + matrix(0B00000101) * v0101 +
                matrix(0B00000110) * v0110 + matrix(0B00000111) * v0111 +
                matrix(0B00001000) * v1000 + matrix(0B00001001) * v1001 +
                matrix(0B00001010) * v1010 + matrix(0B00001011) * v1011 +
                matrix(0B00001100) * v1100 + matrix(0B00001101) * v1101 +
                matrix(0B00001110) * v1110 + matrix(0B00001111) * v1111;
            arr(i0001) =
                matrix(0B00010000) * v0000 + matrix(0B00010001) * v0001 +
                matrix(0B00010010) * v0010 + matrix(0B00010011) * v0011 +
                matrix(0B00010100) * v0100 + matrix(0B00010101) * v0101 +
                matrix(0B00010110) * v0110 + matrix(0B00010111) * v0111 +
                matrix(0B00011000) * v1000 + matrix(0B00011001) * v1001 +
                matrix(0B00011010) * v1010 + matrix(0B00011011) * v1011 +
                matrix(0B00011100) * v1100 + matrix(0B00011101) * v1101 +
                matrix(0B00011110) * v1110 + matrix(0B00011111) * v1111;
            arr(i0010) =
                matrix(0B00100000) * v0000 + matrix(0B00100001) * v0001 +
                matrix(0B00100010) * v0010 + matrix(0B00100011) * v0011 +
                matrix(0B00100100) * v0100 + matrix(0B00100101) * v0101 +
                matrix(0B00100110) * v0110 + matrix(0B00100111) * v0111 +
                matrix(0B00101000) * v1000 + matrix(0B00101001) * v1001 +
                matrix(0B00101010) * v1010 + matrix(0B00101011) * v1011 +
                matrix(0B00101100) * v1100 + matrix(0B00101101) * v1101 +
                matrix(0B00101110) * v1110 + matrix(0B00101111) * v1111;
            arr(i0011) =
                matrix(0B00110000) * v0000 + matrix(0B00110001) * v0001 +
                matrix(0B00110010) * v0010 + matrix(0B00110011) * v0011 +
                matrix(0B00110100) * v0100 + matrix(0B00110101) * v0101 +
                matrix(0B00110110) * v0110 + matrix(0B00110111) * v0111 +
                matrix(0B00111000) * v1000 + matrix(0B00111001) * v1001 +
                matrix(0B00111010) * v1010 + matrix(0B00111011) * v1011 +
                matrix(0B00111100) * v1100 + matrix(0B00111101) * v1101 +
                matrix(0B00111110) * v1110 + matrix(0B00111111) * v1111;
            arr(i0100) =
                matrix(0B01000000) * v0000 + matrix(0B01000001) * v0001 +
                matrix(0B01000010) * v0010 + matrix(0B01000011) * v0011 +
                matrix(0B01000100) * v0100 + matrix(0B01000101) * v0101 +
                matrix(0B01000110) * v0110 + matrix(0B01000111) * v0111 +
                matrix(0B01001000) * v1000 + matrix(0B01001001) * v1001 +
                matrix(0B01001010) * v1010 + matrix(0B01001011) * v1011 +
                matrix(0B01001100) * v1100 + matrix(0B01001101) * v1101 +
                matrix(0B01001110) * v1110 + matrix(0B01001111) * v1111;
            arr(i0101) =
                matrix(0B01010000) * v0000 + matrix(0B01010001) * v0001 +
                matrix(0B01010010) * v0010 + matrix(0B01010011) * v0011 +
                matrix(0B01010100) * v0100 + matrix(0B01010101) * v0101 +
                matrix(0B01010110) * v0110 + matrix(0B01010111) * v0111 +
                matrix(0B01011000) * v1000 + matrix(0B01011001) * v1001 +
                matrix(0B01011010) * v1010 + matrix(0B01011011) * v1011 +
                matrix(0B01011100) * v1100 + matrix(0B01011101) * v1101 +
                matrix(0B01011110) * v1110 + matrix(0B01011111) * v1111;
            arr(i0110) =
                matrix(0B01100000) * v0000 + matrix(0B01100001) * v0001 +
                matrix(0B01100010) * v0010 + matrix(0B01100011) * v0011 +
                matrix(0B01100100) * v0100 + matrix(0B01100101) * v0101 +
                matrix(0B01100110) * v0110 + matrix(0B01100111) * v0111 +
                matrix(0B01101000) * v1000 + matrix(0B01101001) * v1001 +
                matrix(0B01101010) * v1010 + matrix(0B01101011) * v1011 +
                matrix(0B01101100) * v1100 + matrix(0B01101101) * v1101 +
                matrix(0B01101110) * v1110 + matrix(0B01101111) * v1111;
            arr(i0111) =
                matrix(0B01110000) * v0000 + matrix(0B01110001) * v0001 +
                matrix(0B01110010) * v0010 + matrix(0B01110011) * v0011 +
                matrix(0B01110100) * v0100 + matrix(0B01110101) * v0101 +
                matrix(0B01110110) * v0110 + matrix(0B01110111) * v0111 +
                matrix(0B01111000) * v1000 + matrix(0B01111001) * v1001 +
                matrix(0B01111010) * v1010 + matrix(0B01111011) * v1011 +
                matrix(0B01111100) * v1100 + matrix(0B01111101) * v1101 +
                matrix(0B01111110) * v1110 + matrix(0B01111111) * v1111;
            arr(i1000) =
                matrix(0B10000000) * v0000 + matrix(0B10000001) * v0001 +
                matrix(0B10000010) * v0010 + matrix(0B10000011) * v0011 +
                matrix(0B10000100) * v0100 + matrix(0B10000101) * v0101 +
                matrix(0B10000110) * v0110 + matrix(0B10000111) * v0111 +
                matrix(0B10001000) * v1000 + matrix(0B10001001) * v1001 +
                matrix(0B10001010) * v1010 + matrix(0B10001011) * v1011 +
                matrix(0B10001100) * v1100 + matrix(0B10001101) * v1101 +
                matrix(0B10001110) * v1110 + matrix(0B10001111) * v1111;
            arr(i1001) =
                matrix(0B10010000) * v0000 + matrix(0B10010001) * v0001 +
                matrix(0B10010010) * v0010 + matrix(0B10010011) * v0011 +
                matrix(0B10010100) * v0100 + matrix(0B10010101) * v0101 +
                matrix(0B10010110) * v0110 + matrix(0B10010111) * v0111 +
                matrix(0B10011000) * v1000 + matrix(0B10011001) * v1001 +
                matrix(0B10011010) * v1010 + matrix(0B10011011) * v1011 +
                matrix(0B10011100) * v1100 + matrix(0B10011101) * v1101 +
                matrix(0B10011110) * v1110 + matrix(0B10011111) * v1111;
            arr(i1010) =
                matrix(0B10100000) * v0000 + matrix(0B10100001) * v0001 +
                matrix(0B10100010) * v0010 + matrix(0B10100011) * v0011 +
                matrix(0B10100100) * v0100 + matrix(0B10100101) * v0101 +
                matrix(0B10100110) * v0110 + matrix(0B10100111) * v0111 +
                matrix(0B10101000) * v1000 + matrix(0B10101001) * v1001 +
                matrix(0B10101010) * v1010 + matrix(0B10101011) * v1011 +
                matrix(0B10101100) * v1100 + matrix(0B10101101) * v1101 +
                matrix(0B10101110) * v1110 + matrix(0B10101111) * v1111;
            arr(i1011) =
                matrix(0B10110000) * v0000 + matrix(0B10110001) * v0001 +
                matrix(0B10110010) * v0010 + matrix(0B10110011) * v0011 +
                matrix(0B10110100) * v0100 + matrix(0B10110101) * v0101 +
                matrix(0B10110110) * v0110 + matrix(0B10110111) * v0111 +
                matrix(0B10111000) * v1000 + matrix(0B10111001) * v1001 +
                matrix(0B10111010) * v1010 + matrix(0B10111011) * v1011 +
                matrix(0B10111100) * v1100 + matrix(0B10111101) * v1101 +
                matrix(0B10111110) * v1110 + matrix(0B10111111) * v1111;
            arr(i1100) =
                matrix(0B11000000) * v0000 + matrix(0B11000001) * v0001 +
                matrix(0B11000010) * v0010 + matrix(0B11000011) * v0011 +
                matrix(0B11000100) * v0100 + matrix(0B11000101) * v0101 +
                matrix(0B11000110) * v0110 + matrix(0B11000111) * v0111 +
                matrix(0B11001000) * v1000 + matrix(0B11001001) * v1001 +
                matrix(0B11001010) * v1010 + matrix(0B11001011) * v1011 +
                matrix(0B11001100) * v1100 + matrix(0B11001101) * v1101 +
                matrix(0B11001110) * v1110 + matrix(0B11001111) * v1111;
            arr(i1101) =
                matrix(0B11010000) * v0000 + matrix(0B11010001) * v0001 +
                matrix(0B11010010) * v0010 + matrix(0B11010011) * v0011 +
                matrix(0B11010100) * v0100 + matrix(0B11010101) * v0101 +
                matrix(0B11010110) * v0110 + matrix(0B11010111) * v0111 +
                matrix(0B11011000) * v1000 + matrix(0B11011001) * v1001 +
                matrix(0B11011010) * v1010 + matrix(0B11011011) * v1011 +
                matrix(0B11011100) * v1100 + matrix(0B11011101) * v1101 +
                matrix(0B11011110) * v1110 + matrix(0B11011111) * v1111;
            arr(i1110) =
                matrix(0B11100000) * v0000 + matrix(0B11100001) * v0001 +
                matrix(0B11100010) * v0010 + matrix(0B11100011) * v0011 +
                matrix(0B11100100) * v0100 + matrix(0B11100101) * v0101 +
                matrix(0B11100110) * v0110 + matrix(0B11100111) * v0111 +
                matrix(0B11101000) * v1000 + matrix(0B11101001) * v1001 +
                matrix(0B11101010) * v1010 + matrix(0B11101011) * v1011 +
                matrix(0B11101100) * v1100 + matrix(0B11101101) * v1101 +
                matrix(0B11101110) * v1110 + matrix(0B11101111) * v1111;
            arr(i1111) =
                matrix(0B11110000) * v0000 + matrix(0B11110001) * v0001 +
                matrix(0B11110010) * v0010 + matrix(0B11110011) * v0011 +
                matrix(0B11110100) * v0100 + matrix(0B11110101) * v0101 +
                matrix(0B11110110) * v0110 + matrix(0B11110111) * v0111 +
                matrix(0B11111000) * v1000 + matrix(0B11111001) * v1001 +
                matrix(0B11111010) * v1010 + matrix(0B11111011) * v1011 +
                matrix(0B11111100) * v1100 + matrix(0B11111101) * v1101 +
                matrix(0B11111110) * v1110 + matrix(0B11111111) * v1111;
        }
    }
};

template <class PrecisionT, std::size_t n_wires, bool inverse = false>
struct apply5QubitOpFunctor {

    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    std::size_t dim;
    std::size_t num_qubits;

    apply5QubitOpFunctor(KokkosComplexVector &arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_host.size());
        Kokkos::deep_copy(wires, wires_host);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        dim = 1U << n_wires;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t kdim = k * dim;

        std::size_t i00000 = kdim | 0;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i00000 >> (n_wires - pos - 1)) ^
                             (i00000 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i00000 = i00000 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v00000 = arr(i00000);

        std::size_t i00001 = kdim | 1;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i00001 >> (n_wires - pos - 1)) ^
                             (i00001 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i00001 = i00001 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v00001 = arr(i00001);

        std::size_t i00010 = kdim | 2;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i00010 >> (n_wires - pos - 1)) ^
                             (i00010 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i00010 = i00010 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v00010 = arr(i00010);

        std::size_t i00011 = kdim | 3;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i00011 >> (n_wires - pos - 1)) ^
                             (i00011 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i00011 = i00011 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v00011 = arr(i00011);

        std::size_t i00100 = kdim | 4;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i00100 >> (n_wires - pos - 1)) ^
                             (i00100 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i00100 = i00100 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v00100 = arr(i00100);

        std::size_t i00101 = kdim | 5;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i00101 >> (n_wires - pos - 1)) ^
                             (i00101 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i00101 = i00101 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v00101 = arr(i00101);

        std::size_t i00110 = kdim | 6;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i00110 >> (n_wires - pos - 1)) ^
                             (i00110 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i00110 = i00110 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v00110 = arr(i00110);

        std::size_t i00111 = kdim | 7;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i00111 >> (n_wires - pos - 1)) ^
                             (i00111 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i00111 = i00111 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v00111 = arr(i00111);

        std::size_t i01000 = kdim | 8;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i01000 >> (n_wires - pos - 1)) ^
                             (i01000 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i01000 = i01000 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v01000 = arr(i01000);

        std::size_t i01001 = kdim | 9;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i01001 >> (n_wires - pos - 1)) ^
                             (i01001 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i01001 = i01001 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v01001 = arr(i01001);

        std::size_t i01010 = kdim | 10;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i01010 >> (n_wires - pos - 1)) ^
                             (i01010 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i01010 = i01010 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v01010 = arr(i01010);

        std::size_t i01011 = kdim | 11;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i01011 >> (n_wires - pos - 1)) ^
                             (i01011 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i01011 = i01011 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v01011 = arr(i01011);

        std::size_t i01100 = kdim | 12;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i01100 >> (n_wires - pos - 1)) ^
                             (i01100 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i01100 = i01100 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v01100 = arr(i01100);

        std::size_t i01101 = kdim | 13;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i01101 >> (n_wires - pos - 1)) ^
                             (i01101 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i01101 = i01101 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v01101 = arr(i01101);

        std::size_t i01110 = kdim | 14;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i01110 >> (n_wires - pos - 1)) ^
                             (i01110 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i01110 = i01110 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v01110 = arr(i01110);

        std::size_t i01111 = kdim | 15;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i01111 >> (n_wires - pos - 1)) ^
                             (i01111 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i01111 = i01111 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v01111 = arr(i01111);

        std::size_t i10000 = kdim | 16;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i10000 >> (n_wires - pos - 1)) ^
                             (i10000 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i10000 = i10000 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v10000 = arr(i10000);

        std::size_t i10001 = kdim | 17;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i10001 >> (n_wires - pos - 1)) ^
                             (i10001 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i10001 = i10001 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v10001 = arr(i10001);

        std::size_t i10010 = kdim | 18;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i10010 >> (n_wires - pos - 1)) ^
                             (i10010 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i10010 = i10010 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v10010 = arr(i10010);

        std::size_t i10011 = kdim | 19;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i10011 >> (n_wires - pos - 1)) ^
                             (i10011 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i10011 = i10011 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v10011 = arr(i10011);

        std::size_t i10100 = kdim | 20;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i10100 >> (n_wires - pos - 1)) ^
                             (i10100 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i10100 = i10100 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v10100 = arr(i10100);

        std::size_t i10101 = kdim | 21;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i10101 >> (n_wires - pos - 1)) ^
                             (i10101 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i10101 = i10101 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v10101 = arr(i10101);

        std::size_t i10110 = kdim | 22;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i10110 >> (n_wires - pos - 1)) ^
                             (i10110 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i10110 = i10110 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v10110 = arr(i10110);

        std::size_t i10111 = kdim | 23;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i10111 >> (n_wires - pos - 1)) ^
                             (i10111 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i10111 = i10111 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v10111 = arr(i10111);

        std::size_t i11000 = kdim | 24;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i11000 >> (n_wires - pos - 1)) ^
                             (i11000 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i11000 = i11000 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v11000 = arr(i11000);

        std::size_t i11001 = kdim | 25;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i11001 >> (n_wires - pos - 1)) ^
                             (i11001 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i11001 = i11001 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v11001 = arr(i11001);

        std::size_t i11010 = kdim | 26;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i11010 >> (n_wires - pos - 1)) ^
                             (i11010 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i11010 = i11010 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v11010 = arr(i11010);

        std::size_t i11011 = kdim | 27;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i11011 >> (n_wires - pos - 1)) ^
                             (i11011 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i11011 = i11011 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v11011 = arr(i11011);

        std::size_t i11100 = kdim | 28;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i11100 >> (n_wires - pos - 1)) ^
                             (i11100 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i11100 = i11100 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v11100 = arr(i11100);

        std::size_t i11101 = kdim | 29;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i11101 >> (n_wires - pos - 1)) ^
                             (i11101 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i11101 = i11101 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v11101 = arr(i11101);

        std::size_t i11110 = kdim | 30;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i11110 >> (n_wires - pos - 1)) ^
                             (i11110 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i11110 = i11110 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v11110 = arr(i11110);

        std::size_t i11111 = kdim | 31;
        for (std::size_t pos = 0; pos < n_wires; pos++) {
            std::size_t x = ((i11111 >> (n_wires - pos - 1)) ^
                             (i11111 >> (num_qubits - wires(pos) - 1))) &
                            1U;
            i11111 = i11111 ^ ((x << (n_wires - pos - 1)) |
                               (x << (num_qubits - wires(pos) - 1)));
        }
        ComplexT v11111 = arr(i11111);

        if constexpr (inverse) {
            arr(i00000) = conj(matrix(0B0000000000)) * v00000 +
                          conj(matrix(0B0000100000)) * v00001 +
                          conj(matrix(0B0001000000)) * v00010 +
                          conj(matrix(0B0001100000)) * v00011 +
                          conj(matrix(0B0010000000)) * v00100 +
                          conj(matrix(0B0010100000)) * v00101 +
                          conj(matrix(0B0011000000)) * v00110 +
                          conj(matrix(0B0011100000)) * v00111 +
                          conj(matrix(0B0100000000)) * v01000 +
                          conj(matrix(0B0100100000)) * v01001 +
                          conj(matrix(0B0101000000)) * v01010 +
                          conj(matrix(0B0101100000)) * v01011 +
                          conj(matrix(0B0110000000)) * v01100 +
                          conj(matrix(0B0110100000)) * v01101 +
                          conj(matrix(0B0111000000)) * v01110 +
                          conj(matrix(0B0111100000)) * v01111 +
                          conj(matrix(0B1000000000)) * v10000 +
                          conj(matrix(0B1000100000)) * v10001 +
                          conj(matrix(0B1001000000)) * v10010 +
                          conj(matrix(0B1001100000)) * v10011 +
                          conj(matrix(0B1010000000)) * v10100 +
                          conj(matrix(0B1010100000)) * v10101 +
                          conj(matrix(0B1011000000)) * v10110 +
                          conj(matrix(0B1011100000)) * v10111 +
                          conj(matrix(0B1100000000)) * v11000 +
                          conj(matrix(0B1100100000)) * v11001 +
                          conj(matrix(0B1101000000)) * v11010 +
                          conj(matrix(0B1101100000)) * v11011 +
                          conj(matrix(0B1110000000)) * v11100 +
                          conj(matrix(0B1110100000)) * v11101 +
                          conj(matrix(0B1111000000)) * v11110 +
                          conj(matrix(0B1111100000)) * v11111;
            arr(i00001) = conj(matrix(0B0000000001)) * v00000 +
                          conj(matrix(0B0000100001)) * v00001 +
                          conj(matrix(0B0001000001)) * v00010 +
                          conj(matrix(0B0001100001)) * v00011 +
                          conj(matrix(0B0010000001)) * v00100 +
                          conj(matrix(0B0010100001)) * v00101 +
                          conj(matrix(0B0011000001)) * v00110 +
                          conj(matrix(0B0011100001)) * v00111 +
                          conj(matrix(0B0100000001)) * v01000 +
                          conj(matrix(0B0100100001)) * v01001 +
                          conj(matrix(0B0101000001)) * v01010 +
                          conj(matrix(0B0101100001)) * v01011 +
                          conj(matrix(0B0110000001)) * v01100 +
                          conj(matrix(0B0110100001)) * v01101 +
                          conj(matrix(0B0111000001)) * v01110 +
                          conj(matrix(0B0111100001)) * v01111 +
                          conj(matrix(0B1000000001)) * v10000 +
                          conj(matrix(0B1000100001)) * v10001 +
                          conj(matrix(0B1001000001)) * v10010 +
                          conj(matrix(0B1001100001)) * v10011 +
                          conj(matrix(0B1010000001)) * v10100 +
                          conj(matrix(0B1010100001)) * v10101 +
                          conj(matrix(0B1011000001)) * v10110 +
                          conj(matrix(0B1011100001)) * v10111 +
                          conj(matrix(0B1100000001)) * v11000 +
                          conj(matrix(0B1100100001)) * v11001 +
                          conj(matrix(0B1101000001)) * v11010 +
                          conj(matrix(0B1101100001)) * v11011 +
                          conj(matrix(0B1110000001)) * v11100 +
                          conj(matrix(0B1110100001)) * v11101 +
                          conj(matrix(0B1111000001)) * v11110 +
                          conj(matrix(0B1111100001)) * v11111;
            arr(i00010) = conj(matrix(0B0000000010)) * v00000 +
                          conj(matrix(0B0000100010)) * v00001 +
                          conj(matrix(0B0001000010)) * v00010 +
                          conj(matrix(0B0001100010)) * v00011 +
                          conj(matrix(0B0010000010)) * v00100 +
                          conj(matrix(0B0010100010)) * v00101 +
                          conj(matrix(0B0011000010)) * v00110 +
                          conj(matrix(0B0011100010)) * v00111 +
                          conj(matrix(0B0100000010)) * v01000 +
                          conj(matrix(0B0100100010)) * v01001 +
                          conj(matrix(0B0101000010)) * v01010 +
                          conj(matrix(0B0101100010)) * v01011 +
                          conj(matrix(0B0110000010)) * v01100 +
                          conj(matrix(0B0110100010)) * v01101 +
                          conj(matrix(0B0111000010)) * v01110 +
                          conj(matrix(0B0111100010)) * v01111 +
                          conj(matrix(0B1000000010)) * v10000 +
                          conj(matrix(0B1000100010)) * v10001 +
                          conj(matrix(0B1001000010)) * v10010 +
                          conj(matrix(0B1001100010)) * v10011 +
                          conj(matrix(0B1010000010)) * v10100 +
                          conj(matrix(0B1010100010)) * v10101 +
                          conj(matrix(0B1011000010)) * v10110 +
                          conj(matrix(0B1011100010)) * v10111 +
                          conj(matrix(0B1100000010)) * v11000 +
                          conj(matrix(0B1100100010)) * v11001 +
                          conj(matrix(0B1101000010)) * v11010 +
                          conj(matrix(0B1101100010)) * v11011 +
                          conj(matrix(0B1110000010)) * v11100 +
                          conj(matrix(0B1110100010)) * v11101 +
                          conj(matrix(0B1111000010)) * v11110 +
                          conj(matrix(0B1111100010)) * v11111;
            arr(i00011) = conj(matrix(0B0000000011)) * v00000 +
                          conj(matrix(0B0000100011)) * v00001 +
                          conj(matrix(0B0001000011)) * v00010 +
                          conj(matrix(0B0001100011)) * v00011 +
                          conj(matrix(0B0010000011)) * v00100 +
                          conj(matrix(0B0010100011)) * v00101 +
                          conj(matrix(0B0011000011)) * v00110 +
                          conj(matrix(0B0011100011)) * v00111 +
                          conj(matrix(0B0100000011)) * v01000 +
                          conj(matrix(0B0100100011)) * v01001 +
                          conj(matrix(0B0101000011)) * v01010 +
                          conj(matrix(0B0101100011)) * v01011 +
                          conj(matrix(0B0110000011)) * v01100 +
                          conj(matrix(0B0110100011)) * v01101 +
                          conj(matrix(0B0111000011)) * v01110 +
                          conj(matrix(0B0111100011)) * v01111 +
                          conj(matrix(0B1000000011)) * v10000 +
                          conj(matrix(0B1000100011)) * v10001 +
                          conj(matrix(0B1001000011)) * v10010 +
                          conj(matrix(0B1001100011)) * v10011 +
                          conj(matrix(0B1010000011)) * v10100 +
                          conj(matrix(0B1010100011)) * v10101 +
                          conj(matrix(0B1011000011)) * v10110 +
                          conj(matrix(0B1011100011)) * v10111 +
                          conj(matrix(0B1100000011)) * v11000 +
                          conj(matrix(0B1100100011)) * v11001 +
                          conj(matrix(0B1101000011)) * v11010 +
                          conj(matrix(0B1101100011)) * v11011 +
                          conj(matrix(0B1110000011)) * v11100 +
                          conj(matrix(0B1110100011)) * v11101 +
                          conj(matrix(0B1111000011)) * v11110 +
                          conj(matrix(0B1111100011)) * v11111;
            arr(i00100) = conj(matrix(0B0000000100)) * v00000 +
                          conj(matrix(0B0000100100)) * v00001 +
                          conj(matrix(0B0001000100)) * v00010 +
                          conj(matrix(0B0001100100)) * v00011 +
                          conj(matrix(0B0010000100)) * v00100 +
                          conj(matrix(0B0010100100)) * v00101 +
                          conj(matrix(0B0011000100)) * v00110 +
                          conj(matrix(0B0011100100)) * v00111 +
                          conj(matrix(0B0100000100)) * v01000 +
                          conj(matrix(0B0100100100)) * v01001 +
                          conj(matrix(0B0101000100)) * v01010 +
                          conj(matrix(0B0101100100)) * v01011 +
                          conj(matrix(0B0110000100)) * v01100 +
                          conj(matrix(0B0110100100)) * v01101 +
                          conj(matrix(0B0111000100)) * v01110 +
                          conj(matrix(0B0111100100)) * v01111 +
                          conj(matrix(0B1000000100)) * v10000 +
                          conj(matrix(0B1000100100)) * v10001 +
                          conj(matrix(0B1001000100)) * v10010 +
                          conj(matrix(0B1001100100)) * v10011 +
                          conj(matrix(0B1010000100)) * v10100 +
                          conj(matrix(0B1010100100)) * v10101 +
                          conj(matrix(0B1011000100)) * v10110 +
                          conj(matrix(0B1011100100)) * v10111 +
                          conj(matrix(0B1100000100)) * v11000 +
                          conj(matrix(0B1100100100)) * v11001 +
                          conj(matrix(0B1101000100)) * v11010 +
                          conj(matrix(0B1101100100)) * v11011 +
                          conj(matrix(0B1110000100)) * v11100 +
                          conj(matrix(0B1110100100)) * v11101 +
                          conj(matrix(0B1111000100)) * v11110 +
                          conj(matrix(0B1111100100)) * v11111;
            arr(i00101) = conj(matrix(0B0000000101)) * v00000 +
                          conj(matrix(0B0000100101)) * v00001 +
                          conj(matrix(0B0001000101)) * v00010 +
                          conj(matrix(0B0001100101)) * v00011 +
                          conj(matrix(0B0010000101)) * v00100 +
                          conj(matrix(0B0010100101)) * v00101 +
                          conj(matrix(0B0011000101)) * v00110 +
                          conj(matrix(0B0011100101)) * v00111 +
                          conj(matrix(0B0100000101)) * v01000 +
                          conj(matrix(0B0100100101)) * v01001 +
                          conj(matrix(0B0101000101)) * v01010 +
                          conj(matrix(0B0101100101)) * v01011 +
                          conj(matrix(0B0110000101)) * v01100 +
                          conj(matrix(0B0110100101)) * v01101 +
                          conj(matrix(0B0111000101)) * v01110 +
                          conj(matrix(0B0111100101)) * v01111 +
                          conj(matrix(0B1000000101)) * v10000 +
                          conj(matrix(0B1000100101)) * v10001 +
                          conj(matrix(0B1001000101)) * v10010 +
                          conj(matrix(0B1001100101)) * v10011 +
                          conj(matrix(0B1010000101)) * v10100 +
                          conj(matrix(0B1010100101)) * v10101 +
                          conj(matrix(0B1011000101)) * v10110 +
                          conj(matrix(0B1011100101)) * v10111 +
                          conj(matrix(0B1100000101)) * v11000 +
                          conj(matrix(0B1100100101)) * v11001 +
                          conj(matrix(0B1101000101)) * v11010 +
                          conj(matrix(0B1101100101)) * v11011 +
                          conj(matrix(0B1110000101)) * v11100 +
                          conj(matrix(0B1110100101)) * v11101 +
                          conj(matrix(0B1111000101)) * v11110 +
                          conj(matrix(0B1111100101)) * v11111;
            arr(i00110) = conj(matrix(0B0000000110)) * v00000 +
                          conj(matrix(0B0000100110)) * v00001 +
                          conj(matrix(0B0001000110)) * v00010 +
                          conj(matrix(0B0001100110)) * v00011 +
                          conj(matrix(0B0010000110)) * v00100 +
                          conj(matrix(0B0010100110)) * v00101 +
                          conj(matrix(0B0011000110)) * v00110 +
                          conj(matrix(0B0011100110)) * v00111 +
                          conj(matrix(0B0100000110)) * v01000 +
                          conj(matrix(0B0100100110)) * v01001 +
                          conj(matrix(0B0101000110)) * v01010 +
                          conj(matrix(0B0101100110)) * v01011 +
                          conj(matrix(0B0110000110)) * v01100 +
                          conj(matrix(0B0110100110)) * v01101 +
                          conj(matrix(0B0111000110)) * v01110 +
                          conj(matrix(0B0111100110)) * v01111 +
                          conj(matrix(0B1000000110)) * v10000 +
                          conj(matrix(0B1000100110)) * v10001 +
                          conj(matrix(0B1001000110)) * v10010 +
                          conj(matrix(0B1001100110)) * v10011 +
                          conj(matrix(0B1010000110)) * v10100 +
                          conj(matrix(0B1010100110)) * v10101 +
                          conj(matrix(0B1011000110)) * v10110 +
                          conj(matrix(0B1011100110)) * v10111 +
                          conj(matrix(0B1100000110)) * v11000 +
                          conj(matrix(0B1100100110)) * v11001 +
                          conj(matrix(0B1101000110)) * v11010 +
                          conj(matrix(0B1101100110)) * v11011 +
                          conj(matrix(0B1110000110)) * v11100 +
                          conj(matrix(0B1110100110)) * v11101 +
                          conj(matrix(0B1111000110)) * v11110 +
                          conj(matrix(0B1111100110)) * v11111;
            arr(i00111) = conj(matrix(0B0000000111)) * v00000 +
                          conj(matrix(0B0000100111)) * v00001 +
                          conj(matrix(0B0001000111)) * v00010 +
                          conj(matrix(0B0001100111)) * v00011 +
                          conj(matrix(0B0010000111)) * v00100 +
                          conj(matrix(0B0010100111)) * v00101 +
                          conj(matrix(0B0011000111)) * v00110 +
                          conj(matrix(0B0011100111)) * v00111 +
                          conj(matrix(0B0100000111)) * v01000 +
                          conj(matrix(0B0100100111)) * v01001 +
                          conj(matrix(0B0101000111)) * v01010 +
                          conj(matrix(0B0101100111)) * v01011 +
                          conj(matrix(0B0110000111)) * v01100 +
                          conj(matrix(0B0110100111)) * v01101 +
                          conj(matrix(0B0111000111)) * v01110 +
                          conj(matrix(0B0111100111)) * v01111 +
                          conj(matrix(0B1000000111)) * v10000 +
                          conj(matrix(0B1000100111)) * v10001 +
                          conj(matrix(0B1001000111)) * v10010 +
                          conj(matrix(0B1001100111)) * v10011 +
                          conj(matrix(0B1010000111)) * v10100 +
                          conj(matrix(0B1010100111)) * v10101 +
                          conj(matrix(0B1011000111)) * v10110 +
                          conj(matrix(0B1011100111)) * v10111 +
                          conj(matrix(0B1100000111)) * v11000 +
                          conj(matrix(0B1100100111)) * v11001 +
                          conj(matrix(0B1101000111)) * v11010 +
                          conj(matrix(0B1101100111)) * v11011 +
                          conj(matrix(0B1110000111)) * v11100 +
                          conj(matrix(0B1110100111)) * v11101 +
                          conj(matrix(0B1111000111)) * v11110 +
                          conj(matrix(0B1111100111)) * v11111;
            arr(i01000) = conj(matrix(0B0000001000)) * v00000 +
                          conj(matrix(0B0000101000)) * v00001 +
                          conj(matrix(0B0001001000)) * v00010 +
                          conj(matrix(0B0001101000)) * v00011 +
                          conj(matrix(0B0010001000)) * v00100 +
                          conj(matrix(0B0010101000)) * v00101 +
                          conj(matrix(0B0011001000)) * v00110 +
                          conj(matrix(0B0011101000)) * v00111 +
                          conj(matrix(0B0100001000)) * v01000 +
                          conj(matrix(0B0100101000)) * v01001 +
                          conj(matrix(0B0101001000)) * v01010 +
                          conj(matrix(0B0101101000)) * v01011 +
                          conj(matrix(0B0110001000)) * v01100 +
                          conj(matrix(0B0110101000)) * v01101 +
                          conj(matrix(0B0111001000)) * v01110 +
                          conj(matrix(0B0111101000)) * v01111 +
                          conj(matrix(0B1000001000)) * v10000 +
                          conj(matrix(0B1000101000)) * v10001 +
                          conj(matrix(0B1001001000)) * v10010 +
                          conj(matrix(0B1001101000)) * v10011 +
                          conj(matrix(0B1010001000)) * v10100 +
                          conj(matrix(0B1010101000)) * v10101 +
                          conj(matrix(0B1011001000)) * v10110 +
                          conj(matrix(0B1011101000)) * v10111 +
                          conj(matrix(0B1100001000)) * v11000 +
                          conj(matrix(0B1100101000)) * v11001 +
                          conj(matrix(0B1101001000)) * v11010 +
                          conj(matrix(0B1101101000)) * v11011 +
                          conj(matrix(0B1110001000)) * v11100 +
                          conj(matrix(0B1110101000)) * v11101 +
                          conj(matrix(0B1111001000)) * v11110 +
                          conj(matrix(0B1111101000)) * v11111;
            arr(i01001) = conj(matrix(0B0000001001)) * v00000 +
                          conj(matrix(0B0000101001)) * v00001 +
                          conj(matrix(0B0001001001)) * v00010 +
                          conj(matrix(0B0001101001)) * v00011 +
                          conj(matrix(0B0010001001)) * v00100 +
                          conj(matrix(0B0010101001)) * v00101 +
                          conj(matrix(0B0011001001)) * v00110 +
                          conj(matrix(0B0011101001)) * v00111 +
                          conj(matrix(0B0100001001)) * v01000 +
                          conj(matrix(0B0100101001)) * v01001 +
                          conj(matrix(0B0101001001)) * v01010 +
                          conj(matrix(0B0101101001)) * v01011 +
                          conj(matrix(0B0110001001)) * v01100 +
                          conj(matrix(0B0110101001)) * v01101 +
                          conj(matrix(0B0111001001)) * v01110 +
                          conj(matrix(0B0111101001)) * v01111 +
                          conj(matrix(0B1000001001)) * v10000 +
                          conj(matrix(0B1000101001)) * v10001 +
                          conj(matrix(0B1001001001)) * v10010 +
                          conj(matrix(0B1001101001)) * v10011 +
                          conj(matrix(0B1010001001)) * v10100 +
                          conj(matrix(0B1010101001)) * v10101 +
                          conj(matrix(0B1011001001)) * v10110 +
                          conj(matrix(0B1011101001)) * v10111 +
                          conj(matrix(0B1100001001)) * v11000 +
                          conj(matrix(0B1100101001)) * v11001 +
                          conj(matrix(0B1101001001)) * v11010 +
                          conj(matrix(0B1101101001)) * v11011 +
                          conj(matrix(0B1110001001)) * v11100 +
                          conj(matrix(0B1110101001)) * v11101 +
                          conj(matrix(0B1111001001)) * v11110 +
                          conj(matrix(0B1111101001)) * v11111;
            arr(i01010) = conj(matrix(0B0000001010)) * v00000 +
                          conj(matrix(0B0000101010)) * v00001 +
                          conj(matrix(0B0001001010)) * v00010 +
                          conj(matrix(0B0001101010)) * v00011 +
                          conj(matrix(0B0010001010)) * v00100 +
                          conj(matrix(0B0010101010)) * v00101 +
                          conj(matrix(0B0011001010)) * v00110 +
                          conj(matrix(0B0011101010)) * v00111 +
                          conj(matrix(0B0100001010)) * v01000 +
                          conj(matrix(0B0100101010)) * v01001 +
                          conj(matrix(0B0101001010)) * v01010 +
                          conj(matrix(0B0101101010)) * v01011 +
                          conj(matrix(0B0110001010)) * v01100 +
                          conj(matrix(0B0110101010)) * v01101 +
                          conj(matrix(0B0111001010)) * v01110 +
                          conj(matrix(0B0111101010)) * v01111 +
                          conj(matrix(0B1000001010)) * v10000 +
                          conj(matrix(0B1000101010)) * v10001 +
                          conj(matrix(0B1001001010)) * v10010 +
                          conj(matrix(0B1001101010)) * v10011 +
                          conj(matrix(0B1010001010)) * v10100 +
                          conj(matrix(0B1010101010)) * v10101 +
                          conj(matrix(0B1011001010)) * v10110 +
                          conj(matrix(0B1011101010)) * v10111 +
                          conj(matrix(0B1100001010)) * v11000 +
                          conj(matrix(0B1100101010)) * v11001 +
                          conj(matrix(0B1101001010)) * v11010 +
                          conj(matrix(0B1101101010)) * v11011 +
                          conj(matrix(0B1110001010)) * v11100 +
                          conj(matrix(0B1110101010)) * v11101 +
                          conj(matrix(0B1111001010)) * v11110 +
                          conj(matrix(0B1111101010)) * v11111;
            arr(i01011) = conj(matrix(0B0000001011)) * v00000 +
                          conj(matrix(0B0000101011)) * v00001 +
                          conj(matrix(0B0001001011)) * v00010 +
                          conj(matrix(0B0001101011)) * v00011 +
                          conj(matrix(0B0010001011)) * v00100 +
                          conj(matrix(0B0010101011)) * v00101 +
                          conj(matrix(0B0011001011)) * v00110 +
                          conj(matrix(0B0011101011)) * v00111 +
                          conj(matrix(0B0100001011)) * v01000 +
                          conj(matrix(0B0100101011)) * v01001 +
                          conj(matrix(0B0101001011)) * v01010 +
                          conj(matrix(0B0101101011)) * v01011 +
                          conj(matrix(0B0110001011)) * v01100 +
                          conj(matrix(0B0110101011)) * v01101 +
                          conj(matrix(0B0111001011)) * v01110 +
                          conj(matrix(0B0111101011)) * v01111 +
                          conj(matrix(0B1000001011)) * v10000 +
                          conj(matrix(0B1000101011)) * v10001 +
                          conj(matrix(0B1001001011)) * v10010 +
                          conj(matrix(0B1001101011)) * v10011 +
                          conj(matrix(0B1010001011)) * v10100 +
                          conj(matrix(0B1010101011)) * v10101 +
                          conj(matrix(0B1011001011)) * v10110 +
                          conj(matrix(0B1011101011)) * v10111 +
                          conj(matrix(0B1100001011)) * v11000 +
                          conj(matrix(0B1100101011)) * v11001 +
                          conj(matrix(0B1101001011)) * v11010 +
                          conj(matrix(0B1101101011)) * v11011 +
                          conj(matrix(0B1110001011)) * v11100 +
                          conj(matrix(0B1110101011)) * v11101 +
                          conj(matrix(0B1111001011)) * v11110 +
                          conj(matrix(0B1111101011)) * v11111;
            arr(i01100) = conj(matrix(0B0000001100)) * v00000 +
                          conj(matrix(0B0000101100)) * v00001 +
                          conj(matrix(0B0001001100)) * v00010 +
                          conj(matrix(0B0001101100)) * v00011 +
                          conj(matrix(0B0010001100)) * v00100 +
                          conj(matrix(0B0010101100)) * v00101 +
                          conj(matrix(0B0011001100)) * v00110 +
                          conj(matrix(0B0011101100)) * v00111 +
                          conj(matrix(0B0100001100)) * v01000 +
                          conj(matrix(0B0100101100)) * v01001 +
                          conj(matrix(0B0101001100)) * v01010 +
                          conj(matrix(0B0101101100)) * v01011 +
                          conj(matrix(0B0110001100)) * v01100 +
                          conj(matrix(0B0110101100)) * v01101 +
                          conj(matrix(0B0111001100)) * v01110 +
                          conj(matrix(0B0111101100)) * v01111 +
                          conj(matrix(0B1000001100)) * v10000 +
                          conj(matrix(0B1000101100)) * v10001 +
                          conj(matrix(0B1001001100)) * v10010 +
                          conj(matrix(0B1001101100)) * v10011 +
                          conj(matrix(0B1010001100)) * v10100 +
                          conj(matrix(0B1010101100)) * v10101 +
                          conj(matrix(0B1011001100)) * v10110 +
                          conj(matrix(0B1011101100)) * v10111 +
                          conj(matrix(0B1100001100)) * v11000 +
                          conj(matrix(0B1100101100)) * v11001 +
                          conj(matrix(0B1101001100)) * v11010 +
                          conj(matrix(0B1101101100)) * v11011 +
                          conj(matrix(0B1110001100)) * v11100 +
                          conj(matrix(0B1110101100)) * v11101 +
                          conj(matrix(0B1111001100)) * v11110 +
                          conj(matrix(0B1111101100)) * v11111;
            arr(i01101) = conj(matrix(0B0000001101)) * v00000 +
                          conj(matrix(0B0000101101)) * v00001 +
                          conj(matrix(0B0001001101)) * v00010 +
                          conj(matrix(0B0001101101)) * v00011 +
                          conj(matrix(0B0010001101)) * v00100 +
                          conj(matrix(0B0010101101)) * v00101 +
                          conj(matrix(0B0011001101)) * v00110 +
                          conj(matrix(0B0011101101)) * v00111 +
                          conj(matrix(0B0100001101)) * v01000 +
                          conj(matrix(0B0100101101)) * v01001 +
                          conj(matrix(0B0101001101)) * v01010 +
                          conj(matrix(0B0101101101)) * v01011 +
                          conj(matrix(0B0110001101)) * v01100 +
                          conj(matrix(0B0110101101)) * v01101 +
                          conj(matrix(0B0111001101)) * v01110 +
                          conj(matrix(0B0111101101)) * v01111 +
                          conj(matrix(0B1000001101)) * v10000 +
                          conj(matrix(0B1000101101)) * v10001 +
                          conj(matrix(0B1001001101)) * v10010 +
                          conj(matrix(0B1001101101)) * v10011 +
                          conj(matrix(0B1010001101)) * v10100 +
                          conj(matrix(0B1010101101)) * v10101 +
                          conj(matrix(0B1011001101)) * v10110 +
                          conj(matrix(0B1011101101)) * v10111 +
                          conj(matrix(0B1100001101)) * v11000 +
                          conj(matrix(0B1100101101)) * v11001 +
                          conj(matrix(0B1101001101)) * v11010 +
                          conj(matrix(0B1101101101)) * v11011 +
                          conj(matrix(0B1110001101)) * v11100 +
                          conj(matrix(0B1110101101)) * v11101 +
                          conj(matrix(0B1111001101)) * v11110 +
                          conj(matrix(0B1111101101)) * v11111;
            arr(i01110) = conj(matrix(0B0000001110)) * v00000 +
                          conj(matrix(0B0000101110)) * v00001 +
                          conj(matrix(0B0001001110)) * v00010 +
                          conj(matrix(0B0001101110)) * v00011 +
                          conj(matrix(0B0010001110)) * v00100 +
                          conj(matrix(0B0010101110)) * v00101 +
                          conj(matrix(0B0011001110)) * v00110 +
                          conj(matrix(0B0011101110)) * v00111 +
                          conj(matrix(0B0100001110)) * v01000 +
                          conj(matrix(0B0100101110)) * v01001 +
                          conj(matrix(0B0101001110)) * v01010 +
                          conj(matrix(0B0101101110)) * v01011 +
                          conj(matrix(0B0110001110)) * v01100 +
                          conj(matrix(0B0110101110)) * v01101 +
                          conj(matrix(0B0111001110)) * v01110 +
                          conj(matrix(0B0111101110)) * v01111 +
                          conj(matrix(0B1000001110)) * v10000 +
                          conj(matrix(0B1000101110)) * v10001 +
                          conj(matrix(0B1001001110)) * v10010 +
                          conj(matrix(0B1001101110)) * v10011 +
                          conj(matrix(0B1010001110)) * v10100 +
                          conj(matrix(0B1010101110)) * v10101 +
                          conj(matrix(0B1011001110)) * v10110 +
                          conj(matrix(0B1011101110)) * v10111 +
                          conj(matrix(0B1100001110)) * v11000 +
                          conj(matrix(0B1100101110)) * v11001 +
                          conj(matrix(0B1101001110)) * v11010 +
                          conj(matrix(0B1101101110)) * v11011 +
                          conj(matrix(0B1110001110)) * v11100 +
                          conj(matrix(0B1110101110)) * v11101 +
                          conj(matrix(0B1111001110)) * v11110 +
                          conj(matrix(0B1111101110)) * v11111;
            arr(i01111) = conj(matrix(0B0000001111)) * v00000 +
                          conj(matrix(0B0000101111)) * v00001 +
                          conj(matrix(0B0001001111)) * v00010 +
                          conj(matrix(0B0001101111)) * v00011 +
                          conj(matrix(0B0010001111)) * v00100 +
                          conj(matrix(0B0010101111)) * v00101 +
                          conj(matrix(0B0011001111)) * v00110 +
                          conj(matrix(0B0011101111)) * v00111 +
                          conj(matrix(0B0100001111)) * v01000 +
                          conj(matrix(0B0100101111)) * v01001 +
                          conj(matrix(0B0101001111)) * v01010 +
                          conj(matrix(0B0101101111)) * v01011 +
                          conj(matrix(0B0110001111)) * v01100 +
                          conj(matrix(0B0110101111)) * v01101 +
                          conj(matrix(0B0111001111)) * v01110 +
                          conj(matrix(0B0111101111)) * v01111 +
                          conj(matrix(0B1000001111)) * v10000 +
                          conj(matrix(0B1000101111)) * v10001 +
                          conj(matrix(0B1001001111)) * v10010 +
                          conj(matrix(0B1001101111)) * v10011 +
                          conj(matrix(0B1010001111)) * v10100 +
                          conj(matrix(0B1010101111)) * v10101 +
                          conj(matrix(0B1011001111)) * v10110 +
                          conj(matrix(0B1011101111)) * v10111 +
                          conj(matrix(0B1100001111)) * v11000 +
                          conj(matrix(0B1100101111)) * v11001 +
                          conj(matrix(0B1101001111)) * v11010 +
                          conj(matrix(0B1101101111)) * v11011 +
                          conj(matrix(0B1110001111)) * v11100 +
                          conj(matrix(0B1110101111)) * v11101 +
                          conj(matrix(0B1111001111)) * v11110 +
                          conj(matrix(0B1111101111)) * v11111;
            arr(i10000) = conj(matrix(0B0000010000)) * v00000 +
                          conj(matrix(0B0000110000)) * v00001 +
                          conj(matrix(0B0001010000)) * v00010 +
                          conj(matrix(0B0001110000)) * v00011 +
                          conj(matrix(0B0010010000)) * v00100 +
                          conj(matrix(0B0010110000)) * v00101 +
                          conj(matrix(0B0011010000)) * v00110 +
                          conj(matrix(0B0011110000)) * v00111 +
                          conj(matrix(0B0100010000)) * v01000 +
                          conj(matrix(0B0100110000)) * v01001 +
                          conj(matrix(0B0101010000)) * v01010 +
                          conj(matrix(0B0101110000)) * v01011 +
                          conj(matrix(0B0110010000)) * v01100 +
                          conj(matrix(0B0110110000)) * v01101 +
                          conj(matrix(0B0111010000)) * v01110 +
                          conj(matrix(0B0111110000)) * v01111 +
                          conj(matrix(0B1000010000)) * v10000 +
                          conj(matrix(0B1000110000)) * v10001 +
                          conj(matrix(0B1001010000)) * v10010 +
                          conj(matrix(0B1001110000)) * v10011 +
                          conj(matrix(0B1010010000)) * v10100 +
                          conj(matrix(0B1010110000)) * v10101 +
                          conj(matrix(0B1011010000)) * v10110 +
                          conj(matrix(0B1011110000)) * v10111 +
                          conj(matrix(0B1100010000)) * v11000 +
                          conj(matrix(0B1100110000)) * v11001 +
                          conj(matrix(0B1101010000)) * v11010 +
                          conj(matrix(0B1101110000)) * v11011 +
                          conj(matrix(0B1110010000)) * v11100 +
                          conj(matrix(0B1110110000)) * v11101 +
                          conj(matrix(0B1111010000)) * v11110 +
                          conj(matrix(0B1111110000)) * v11111;
            arr(i10001) = conj(matrix(0B0000010001)) * v00000 +
                          conj(matrix(0B0000110001)) * v00001 +
                          conj(matrix(0B0001010001)) * v00010 +
                          conj(matrix(0B0001110001)) * v00011 +
                          conj(matrix(0B0010010001)) * v00100 +
                          conj(matrix(0B0010110001)) * v00101 +
                          conj(matrix(0B0011010001)) * v00110 +
                          conj(matrix(0B0011110001)) * v00111 +
                          conj(matrix(0B0100010001)) * v01000 +
                          conj(matrix(0B0100110001)) * v01001 +
                          conj(matrix(0B0101010001)) * v01010 +
                          conj(matrix(0B0101110001)) * v01011 +
                          conj(matrix(0B0110010001)) * v01100 +
                          conj(matrix(0B0110110001)) * v01101 +
                          conj(matrix(0B0111010001)) * v01110 +
                          conj(matrix(0B0111110001)) * v01111 +
                          conj(matrix(0B1000010001)) * v10000 +
                          conj(matrix(0B1000110001)) * v10001 +
                          conj(matrix(0B1001010001)) * v10010 +
                          conj(matrix(0B1001110001)) * v10011 +
                          conj(matrix(0B1010010001)) * v10100 +
                          conj(matrix(0B1010110001)) * v10101 +
                          conj(matrix(0B1011010001)) * v10110 +
                          conj(matrix(0B1011110001)) * v10111 +
                          conj(matrix(0B1100010001)) * v11000 +
                          conj(matrix(0B1100110001)) * v11001 +
                          conj(matrix(0B1101010001)) * v11010 +
                          conj(matrix(0B1101110001)) * v11011 +
                          conj(matrix(0B1110010001)) * v11100 +
                          conj(matrix(0B1110110001)) * v11101 +
                          conj(matrix(0B1111010001)) * v11110 +
                          conj(matrix(0B1111110001)) * v11111;
            arr(i10010) = conj(matrix(0B0000010010)) * v00000 +
                          conj(matrix(0B0000110010)) * v00001 +
                          conj(matrix(0B0001010010)) * v00010 +
                          conj(matrix(0B0001110010)) * v00011 +
                          conj(matrix(0B0010010010)) * v00100 +
                          conj(matrix(0B0010110010)) * v00101 +
                          conj(matrix(0B0011010010)) * v00110 +
                          conj(matrix(0B0011110010)) * v00111 +
                          conj(matrix(0B0100010010)) * v01000 +
                          conj(matrix(0B0100110010)) * v01001 +
                          conj(matrix(0B0101010010)) * v01010 +
                          conj(matrix(0B0101110010)) * v01011 +
                          conj(matrix(0B0110010010)) * v01100 +
                          conj(matrix(0B0110110010)) * v01101 +
                          conj(matrix(0B0111010010)) * v01110 +
                          conj(matrix(0B0111110010)) * v01111 +
                          conj(matrix(0B1000010010)) * v10000 +
                          conj(matrix(0B1000110010)) * v10001 +
                          conj(matrix(0B1001010010)) * v10010 +
                          conj(matrix(0B1001110010)) * v10011 +
                          conj(matrix(0B1010010010)) * v10100 +
                          conj(matrix(0B1010110010)) * v10101 +
                          conj(matrix(0B1011010010)) * v10110 +
                          conj(matrix(0B1011110010)) * v10111 +
                          conj(matrix(0B1100010010)) * v11000 +
                          conj(matrix(0B1100110010)) * v11001 +
                          conj(matrix(0B1101010010)) * v11010 +
                          conj(matrix(0B1101110010)) * v11011 +
                          conj(matrix(0B1110010010)) * v11100 +
                          conj(matrix(0B1110110010)) * v11101 +
                          conj(matrix(0B1111010010)) * v11110 +
                          conj(matrix(0B1111110010)) * v11111;
            arr(i10011) = conj(matrix(0B0000010011)) * v00000 +
                          conj(matrix(0B0000110011)) * v00001 +
                          conj(matrix(0B0001010011)) * v00010 +
                          conj(matrix(0B0001110011)) * v00011 +
                          conj(matrix(0B0010010011)) * v00100 +
                          conj(matrix(0B0010110011)) * v00101 +
                          conj(matrix(0B0011010011)) * v00110 +
                          conj(matrix(0B0011110011)) * v00111 +
                          conj(matrix(0B0100010011)) * v01000 +
                          conj(matrix(0B0100110011)) * v01001 +
                          conj(matrix(0B0101010011)) * v01010 +
                          conj(matrix(0B0101110011)) * v01011 +
                          conj(matrix(0B0110010011)) * v01100 +
                          conj(matrix(0B0110110011)) * v01101 +
                          conj(matrix(0B0111010011)) * v01110 +
                          conj(matrix(0B0111110011)) * v01111 +
                          conj(matrix(0B1000010011)) * v10000 +
                          conj(matrix(0B1000110011)) * v10001 +
                          conj(matrix(0B1001010011)) * v10010 +
                          conj(matrix(0B1001110011)) * v10011 +
                          conj(matrix(0B1010010011)) * v10100 +
                          conj(matrix(0B1010110011)) * v10101 +
                          conj(matrix(0B1011010011)) * v10110 +
                          conj(matrix(0B1011110011)) * v10111 +
                          conj(matrix(0B1100010011)) * v11000 +
                          conj(matrix(0B1100110011)) * v11001 +
                          conj(matrix(0B1101010011)) * v11010 +
                          conj(matrix(0B1101110011)) * v11011 +
                          conj(matrix(0B1110010011)) * v11100 +
                          conj(matrix(0B1110110011)) * v11101 +
                          conj(matrix(0B1111010011)) * v11110 +
                          conj(matrix(0B1111110011)) * v11111;
            arr(i10100) = conj(matrix(0B0000010100)) * v00000 +
                          conj(matrix(0B0000110100)) * v00001 +
                          conj(matrix(0B0001010100)) * v00010 +
                          conj(matrix(0B0001110100)) * v00011 +
                          conj(matrix(0B0010010100)) * v00100 +
                          conj(matrix(0B0010110100)) * v00101 +
                          conj(matrix(0B0011010100)) * v00110 +
                          conj(matrix(0B0011110100)) * v00111 +
                          conj(matrix(0B0100010100)) * v01000 +
                          conj(matrix(0B0100110100)) * v01001 +
                          conj(matrix(0B0101010100)) * v01010 +
                          conj(matrix(0B0101110100)) * v01011 +
                          conj(matrix(0B0110010100)) * v01100 +
                          conj(matrix(0B0110110100)) * v01101 +
                          conj(matrix(0B0111010100)) * v01110 +
                          conj(matrix(0B0111110100)) * v01111 +
                          conj(matrix(0B1000010100)) * v10000 +
                          conj(matrix(0B1000110100)) * v10001 +
                          conj(matrix(0B1001010100)) * v10010 +
                          conj(matrix(0B1001110100)) * v10011 +
                          conj(matrix(0B1010010100)) * v10100 +
                          conj(matrix(0B1010110100)) * v10101 +
                          conj(matrix(0B1011010100)) * v10110 +
                          conj(matrix(0B1011110100)) * v10111 +
                          conj(matrix(0B1100010100)) * v11000 +
                          conj(matrix(0B1100110100)) * v11001 +
                          conj(matrix(0B1101010100)) * v11010 +
                          conj(matrix(0B1101110100)) * v11011 +
                          conj(matrix(0B1110010100)) * v11100 +
                          conj(matrix(0B1110110100)) * v11101 +
                          conj(matrix(0B1111010100)) * v11110 +
                          conj(matrix(0B1111110100)) * v11111;
            arr(i10101) = conj(matrix(0B0000010101)) * v00000 +
                          conj(matrix(0B0000110101)) * v00001 +
                          conj(matrix(0B0001010101)) * v00010 +
                          conj(matrix(0B0001110101)) * v00011 +
                          conj(matrix(0B0010010101)) * v00100 +
                          conj(matrix(0B0010110101)) * v00101 +
                          conj(matrix(0B0011010101)) * v00110 +
                          conj(matrix(0B0011110101)) * v00111 +
                          conj(matrix(0B0100010101)) * v01000 +
                          conj(matrix(0B0100110101)) * v01001 +
                          conj(matrix(0B0101010101)) * v01010 +
                          conj(matrix(0B0101110101)) * v01011 +
                          conj(matrix(0B0110010101)) * v01100 +
                          conj(matrix(0B0110110101)) * v01101 +
                          conj(matrix(0B0111010101)) * v01110 +
                          conj(matrix(0B0111110101)) * v01111 +
                          conj(matrix(0B1000010101)) * v10000 +
                          conj(matrix(0B1000110101)) * v10001 +
                          conj(matrix(0B1001010101)) * v10010 +
                          conj(matrix(0B1001110101)) * v10011 +
                          conj(matrix(0B1010010101)) * v10100 +
                          conj(matrix(0B1010110101)) * v10101 +
                          conj(matrix(0B1011010101)) * v10110 +
                          conj(matrix(0B1011110101)) * v10111 +
                          conj(matrix(0B1100010101)) * v11000 +
                          conj(matrix(0B1100110101)) * v11001 +
                          conj(matrix(0B1101010101)) * v11010 +
                          conj(matrix(0B1101110101)) * v11011 +
                          conj(matrix(0B1110010101)) * v11100 +
                          conj(matrix(0B1110110101)) * v11101 +
                          conj(matrix(0B1111010101)) * v11110 +
                          conj(matrix(0B1111110101)) * v11111;
            arr(i10110) = conj(matrix(0B0000010110)) * v00000 +
                          conj(matrix(0B0000110110)) * v00001 +
                          conj(matrix(0B0001010110)) * v00010 +
                          conj(matrix(0B0001110110)) * v00011 +
                          conj(matrix(0B0010010110)) * v00100 +
                          conj(matrix(0B0010110110)) * v00101 +
                          conj(matrix(0B0011010110)) * v00110 +
                          conj(matrix(0B0011110110)) * v00111 +
                          conj(matrix(0B0100010110)) * v01000 +
                          conj(matrix(0B0100110110)) * v01001 +
                          conj(matrix(0B0101010110)) * v01010 +
                          conj(matrix(0B0101110110)) * v01011 +
                          conj(matrix(0B0110010110)) * v01100 +
                          conj(matrix(0B0110110110)) * v01101 +
                          conj(matrix(0B0111010110)) * v01110 +
                          conj(matrix(0B0111110110)) * v01111 +
                          conj(matrix(0B1000010110)) * v10000 +
                          conj(matrix(0B1000110110)) * v10001 +
                          conj(matrix(0B1001010110)) * v10010 +
                          conj(matrix(0B1001110110)) * v10011 +
                          conj(matrix(0B1010010110)) * v10100 +
                          conj(matrix(0B1010110110)) * v10101 +
                          conj(matrix(0B1011010110)) * v10110 +
                          conj(matrix(0B1011110110)) * v10111 +
                          conj(matrix(0B1100010110)) * v11000 +
                          conj(matrix(0B1100110110)) * v11001 +
                          conj(matrix(0B1101010110)) * v11010 +
                          conj(matrix(0B1101110110)) * v11011 +
                          conj(matrix(0B1110010110)) * v11100 +
                          conj(matrix(0B1110110110)) * v11101 +
                          conj(matrix(0B1111010110)) * v11110 +
                          conj(matrix(0B1111110110)) * v11111;
            arr(i10111) = conj(matrix(0B0000010111)) * v00000 +
                          conj(matrix(0B0000110111)) * v00001 +
                          conj(matrix(0B0001010111)) * v00010 +
                          conj(matrix(0B0001110111)) * v00011 +
                          conj(matrix(0B0010010111)) * v00100 +
                          conj(matrix(0B0010110111)) * v00101 +
                          conj(matrix(0B0011010111)) * v00110 +
                          conj(matrix(0B0011110111)) * v00111 +
                          conj(matrix(0B0100010111)) * v01000 +
                          conj(matrix(0B0100110111)) * v01001 +
                          conj(matrix(0B0101010111)) * v01010 +
                          conj(matrix(0B0101110111)) * v01011 +
                          conj(matrix(0B0110010111)) * v01100 +
                          conj(matrix(0B0110110111)) * v01101 +
                          conj(matrix(0B0111010111)) * v01110 +
                          conj(matrix(0B0111110111)) * v01111 +
                          conj(matrix(0B1000010111)) * v10000 +
                          conj(matrix(0B1000110111)) * v10001 +
                          conj(matrix(0B1001010111)) * v10010 +
                          conj(matrix(0B1001110111)) * v10011 +
                          conj(matrix(0B1010010111)) * v10100 +
                          conj(matrix(0B1010110111)) * v10101 +
                          conj(matrix(0B1011010111)) * v10110 +
                          conj(matrix(0B1011110111)) * v10111 +
                          conj(matrix(0B1100010111)) * v11000 +
                          conj(matrix(0B1100110111)) * v11001 +
                          conj(matrix(0B1101010111)) * v11010 +
                          conj(matrix(0B1101110111)) * v11011 +
                          conj(matrix(0B1110010111)) * v11100 +
                          conj(matrix(0B1110110111)) * v11101 +
                          conj(matrix(0B1111010111)) * v11110 +
                          conj(matrix(0B1111110111)) * v11111;
            arr(i11000) = conj(matrix(0B0000011000)) * v00000 +
                          conj(matrix(0B0000111000)) * v00001 +
                          conj(matrix(0B0001011000)) * v00010 +
                          conj(matrix(0B0001111000)) * v00011 +
                          conj(matrix(0B0010011000)) * v00100 +
                          conj(matrix(0B0010111000)) * v00101 +
                          conj(matrix(0B0011011000)) * v00110 +
                          conj(matrix(0B0011111000)) * v00111 +
                          conj(matrix(0B0100011000)) * v01000 +
                          conj(matrix(0B0100111000)) * v01001 +
                          conj(matrix(0B0101011000)) * v01010 +
                          conj(matrix(0B0101111000)) * v01011 +
                          conj(matrix(0B0110011000)) * v01100 +
                          conj(matrix(0B0110111000)) * v01101 +
                          conj(matrix(0B0111011000)) * v01110 +
                          conj(matrix(0B0111111000)) * v01111 +
                          conj(matrix(0B1000011000)) * v10000 +
                          conj(matrix(0B1000111000)) * v10001 +
                          conj(matrix(0B1001011000)) * v10010 +
                          conj(matrix(0B1001111000)) * v10011 +
                          conj(matrix(0B1010011000)) * v10100 +
                          conj(matrix(0B1010111000)) * v10101 +
                          conj(matrix(0B1011011000)) * v10110 +
                          conj(matrix(0B1011111000)) * v10111 +
                          conj(matrix(0B1100011000)) * v11000 +
                          conj(matrix(0B1100111000)) * v11001 +
                          conj(matrix(0B1101011000)) * v11010 +
                          conj(matrix(0B1101111000)) * v11011 +
                          conj(matrix(0B1110011000)) * v11100 +
                          conj(matrix(0B1110111000)) * v11101 +
                          conj(matrix(0B1111011000)) * v11110 +
                          conj(matrix(0B1111111000)) * v11111;
            arr(i11001) = conj(matrix(0B0000011001)) * v00000 +
                          conj(matrix(0B0000111001)) * v00001 +
                          conj(matrix(0B0001011001)) * v00010 +
                          conj(matrix(0B0001111001)) * v00011 +
                          conj(matrix(0B0010011001)) * v00100 +
                          conj(matrix(0B0010111001)) * v00101 +
                          conj(matrix(0B0011011001)) * v00110 +
                          conj(matrix(0B0011111001)) * v00111 +
                          conj(matrix(0B0100011001)) * v01000 +
                          conj(matrix(0B0100111001)) * v01001 +
                          conj(matrix(0B0101011001)) * v01010 +
                          conj(matrix(0B0101111001)) * v01011 +
                          conj(matrix(0B0110011001)) * v01100 +
                          conj(matrix(0B0110111001)) * v01101 +
                          conj(matrix(0B0111011001)) * v01110 +
                          conj(matrix(0B0111111001)) * v01111 +
                          conj(matrix(0B1000011001)) * v10000 +
                          conj(matrix(0B1000111001)) * v10001 +
                          conj(matrix(0B1001011001)) * v10010 +
                          conj(matrix(0B1001111001)) * v10011 +
                          conj(matrix(0B1010011001)) * v10100 +
                          conj(matrix(0B1010111001)) * v10101 +
                          conj(matrix(0B1011011001)) * v10110 +
                          conj(matrix(0B1011111001)) * v10111 +
                          conj(matrix(0B1100011001)) * v11000 +
                          conj(matrix(0B1100111001)) * v11001 +
                          conj(matrix(0B1101011001)) * v11010 +
                          conj(matrix(0B1101111001)) * v11011 +
                          conj(matrix(0B1110011001)) * v11100 +
                          conj(matrix(0B1110111001)) * v11101 +
                          conj(matrix(0B1111011001)) * v11110 +
                          conj(matrix(0B1111111001)) * v11111;
            arr(i11010) = conj(matrix(0B0000011010)) * v00000 +
                          conj(matrix(0B0000111010)) * v00001 +
                          conj(matrix(0B0001011010)) * v00010 +
                          conj(matrix(0B0001111010)) * v00011 +
                          conj(matrix(0B0010011010)) * v00100 +
                          conj(matrix(0B0010111010)) * v00101 +
                          conj(matrix(0B0011011010)) * v00110 +
                          conj(matrix(0B0011111010)) * v00111 +
                          conj(matrix(0B0100011010)) * v01000 +
                          conj(matrix(0B0100111010)) * v01001 +
                          conj(matrix(0B0101011010)) * v01010 +
                          conj(matrix(0B0101111010)) * v01011 +
                          conj(matrix(0B0110011010)) * v01100 +
                          conj(matrix(0B0110111010)) * v01101 +
                          conj(matrix(0B0111011010)) * v01110 +
                          conj(matrix(0B0111111010)) * v01111 +
                          conj(matrix(0B1000011010)) * v10000 +
                          conj(matrix(0B1000111010)) * v10001 +
                          conj(matrix(0B1001011010)) * v10010 +
                          conj(matrix(0B1001111010)) * v10011 +
                          conj(matrix(0B1010011010)) * v10100 +
                          conj(matrix(0B1010111010)) * v10101 +
                          conj(matrix(0B1011011010)) * v10110 +
                          conj(matrix(0B1011111010)) * v10111 +
                          conj(matrix(0B1100011010)) * v11000 +
                          conj(matrix(0B1100111010)) * v11001 +
                          conj(matrix(0B1101011010)) * v11010 +
                          conj(matrix(0B1101111010)) * v11011 +
                          conj(matrix(0B1110011010)) * v11100 +
                          conj(matrix(0B1110111010)) * v11101 +
                          conj(matrix(0B1111011010)) * v11110 +
                          conj(matrix(0B1111111010)) * v11111;
            arr(i11011) = conj(matrix(0B0000011011)) * v00000 +
                          conj(matrix(0B0000111011)) * v00001 +
                          conj(matrix(0B0001011011)) * v00010 +
                          conj(matrix(0B0001111011)) * v00011 +
                          conj(matrix(0B0010011011)) * v00100 +
                          conj(matrix(0B0010111011)) * v00101 +
                          conj(matrix(0B0011011011)) * v00110 +
                          conj(matrix(0B0011111011)) * v00111 +
                          conj(matrix(0B0100011011)) * v01000 +
                          conj(matrix(0B0100111011)) * v01001 +
                          conj(matrix(0B0101011011)) * v01010 +
                          conj(matrix(0B0101111011)) * v01011 +
                          conj(matrix(0B0110011011)) * v01100 +
                          conj(matrix(0B0110111011)) * v01101 +
                          conj(matrix(0B0111011011)) * v01110 +
                          conj(matrix(0B0111111011)) * v01111 +
                          conj(matrix(0B1000011011)) * v10000 +
                          conj(matrix(0B1000111011)) * v10001 +
                          conj(matrix(0B1001011011)) * v10010 +
                          conj(matrix(0B1001111011)) * v10011 +
                          conj(matrix(0B1010011011)) * v10100 +
                          conj(matrix(0B1010111011)) * v10101 +
                          conj(matrix(0B1011011011)) * v10110 +
                          conj(matrix(0B1011111011)) * v10111 +
                          conj(matrix(0B1100011011)) * v11000 +
                          conj(matrix(0B1100111011)) * v11001 +
                          conj(matrix(0B1101011011)) * v11010 +
                          conj(matrix(0B1101111011)) * v11011 +
                          conj(matrix(0B1110011011)) * v11100 +
                          conj(matrix(0B1110111011)) * v11101 +
                          conj(matrix(0B1111011011)) * v11110 +
                          conj(matrix(0B1111111011)) * v11111;
            arr(i11100) = conj(matrix(0B0000011100)) * v00000 +
                          conj(matrix(0B0000111100)) * v00001 +
                          conj(matrix(0B0001011100)) * v00010 +
                          conj(matrix(0B0001111100)) * v00011 +
                          conj(matrix(0B0010011100)) * v00100 +
                          conj(matrix(0B0010111100)) * v00101 +
                          conj(matrix(0B0011011100)) * v00110 +
                          conj(matrix(0B0011111100)) * v00111 +
                          conj(matrix(0B0100011100)) * v01000 +
                          conj(matrix(0B0100111100)) * v01001 +
                          conj(matrix(0B0101011100)) * v01010 +
                          conj(matrix(0B0101111100)) * v01011 +
                          conj(matrix(0B0110011100)) * v01100 +
                          conj(matrix(0B0110111100)) * v01101 +
                          conj(matrix(0B0111011100)) * v01110 +
                          conj(matrix(0B0111111100)) * v01111 +
                          conj(matrix(0B1000011100)) * v10000 +
                          conj(matrix(0B1000111100)) * v10001 +
                          conj(matrix(0B1001011100)) * v10010 +
                          conj(matrix(0B1001111100)) * v10011 +
                          conj(matrix(0B1010011100)) * v10100 +
                          conj(matrix(0B1010111100)) * v10101 +
                          conj(matrix(0B1011011100)) * v10110 +
                          conj(matrix(0B1011111100)) * v10111 +
                          conj(matrix(0B1100011100)) * v11000 +
                          conj(matrix(0B1100111100)) * v11001 +
                          conj(matrix(0B1101011100)) * v11010 +
                          conj(matrix(0B1101111100)) * v11011 +
                          conj(matrix(0B1110011100)) * v11100 +
                          conj(matrix(0B1110111100)) * v11101 +
                          conj(matrix(0B1111011100)) * v11110 +
                          conj(matrix(0B1111111100)) * v11111;
            arr(i11101) = conj(matrix(0B0000011101)) * v00000 +
                          conj(matrix(0B0000111101)) * v00001 +
                          conj(matrix(0B0001011101)) * v00010 +
                          conj(matrix(0B0001111101)) * v00011 +
                          conj(matrix(0B0010011101)) * v00100 +
                          conj(matrix(0B0010111101)) * v00101 +
                          conj(matrix(0B0011011101)) * v00110 +
                          conj(matrix(0B0011111101)) * v00111 +
                          conj(matrix(0B0100011101)) * v01000 +
                          conj(matrix(0B0100111101)) * v01001 +
                          conj(matrix(0B0101011101)) * v01010 +
                          conj(matrix(0B0101111101)) * v01011 +
                          conj(matrix(0B0110011101)) * v01100 +
                          conj(matrix(0B0110111101)) * v01101 +
                          conj(matrix(0B0111011101)) * v01110 +
                          conj(matrix(0B0111111101)) * v01111 +
                          conj(matrix(0B1000011101)) * v10000 +
                          conj(matrix(0B1000111101)) * v10001 +
                          conj(matrix(0B1001011101)) * v10010 +
                          conj(matrix(0B1001111101)) * v10011 +
                          conj(matrix(0B1010011101)) * v10100 +
                          conj(matrix(0B1010111101)) * v10101 +
                          conj(matrix(0B1011011101)) * v10110 +
                          conj(matrix(0B1011111101)) * v10111 +
                          conj(matrix(0B1100011101)) * v11000 +
                          conj(matrix(0B1100111101)) * v11001 +
                          conj(matrix(0B1101011101)) * v11010 +
                          conj(matrix(0B1101111101)) * v11011 +
                          conj(matrix(0B1110011101)) * v11100 +
                          conj(matrix(0B1110111101)) * v11101 +
                          conj(matrix(0B1111011101)) * v11110 +
                          conj(matrix(0B1111111101)) * v11111;
            arr(i11110) = conj(matrix(0B0000011110)) * v00000 +
                          conj(matrix(0B0000111110)) * v00001 +
                          conj(matrix(0B0001011110)) * v00010 +
                          conj(matrix(0B0001111110)) * v00011 +
                          conj(matrix(0B0010011110)) * v00100 +
                          conj(matrix(0B0010111110)) * v00101 +
                          conj(matrix(0B0011011110)) * v00110 +
                          conj(matrix(0B0011111110)) * v00111 +
                          conj(matrix(0B0100011110)) * v01000 +
                          conj(matrix(0B0100111110)) * v01001 +
                          conj(matrix(0B0101011110)) * v01010 +
                          conj(matrix(0B0101111110)) * v01011 +
                          conj(matrix(0B0110011110)) * v01100 +
                          conj(matrix(0B0110111110)) * v01101 +
                          conj(matrix(0B0111011110)) * v01110 +
                          conj(matrix(0B0111111110)) * v01111 +
                          conj(matrix(0B1000011110)) * v10000 +
                          conj(matrix(0B1000111110)) * v10001 +
                          conj(matrix(0B1001011110)) * v10010 +
                          conj(matrix(0B1001111110)) * v10011 +
                          conj(matrix(0B1010011110)) * v10100 +
                          conj(matrix(0B1010111110)) * v10101 +
                          conj(matrix(0B1011011110)) * v10110 +
                          conj(matrix(0B1011111110)) * v10111 +
                          conj(matrix(0B1100011110)) * v11000 +
                          conj(matrix(0B1100111110)) * v11001 +
                          conj(matrix(0B1101011110)) * v11010 +
                          conj(matrix(0B1101111110)) * v11011 +
                          conj(matrix(0B1110011110)) * v11100 +
                          conj(matrix(0B1110111110)) * v11101 +
                          conj(matrix(0B1111011110)) * v11110 +
                          conj(matrix(0B1111111110)) * v11111;
            arr(i11111) = conj(matrix(0B0000011111)) * v00000 +
                          conj(matrix(0B0000111111)) * v00001 +
                          conj(matrix(0B0001011111)) * v00010 +
                          conj(matrix(0B0001111111)) * v00011 +
                          conj(matrix(0B0010011111)) * v00100 +
                          conj(matrix(0B0010111111)) * v00101 +
                          conj(matrix(0B0011011111)) * v00110 +
                          conj(matrix(0B0011111111)) * v00111 +
                          conj(matrix(0B0100011111)) * v01000 +
                          conj(matrix(0B0100111111)) * v01001 +
                          conj(matrix(0B0101011111)) * v01010 +
                          conj(matrix(0B0101111111)) * v01011 +
                          conj(matrix(0B0110011111)) * v01100 +
                          conj(matrix(0B0110111111)) * v01101 +
                          conj(matrix(0B0111011111)) * v01110 +
                          conj(matrix(0B0111111111)) * v01111 +
                          conj(matrix(0B1000011111)) * v10000 +
                          conj(matrix(0B1000111111)) * v10001 +
                          conj(matrix(0B1001011111)) * v10010 +
                          conj(matrix(0B1001111111)) * v10011 +
                          conj(matrix(0B1010011111)) * v10100 +
                          conj(matrix(0B1010111111)) * v10101 +
                          conj(matrix(0B1011011111)) * v10110 +
                          conj(matrix(0B1011111111)) * v10111 +
                          conj(matrix(0B1100011111)) * v11000 +
                          conj(matrix(0B1100111111)) * v11001 +
                          conj(matrix(0B1101011111)) * v11010 +
                          conj(matrix(0B1101111111)) * v11011 +
                          conj(matrix(0B1110011111)) * v11100 +
                          conj(matrix(0B1110111111)) * v11101 +
                          conj(matrix(0B1111011111)) * v11110 +
                          conj(matrix(0B1111111111)) * v11111;
        } else {
            arr(i00000) =
                matrix(0B0000000000) * v00000 + matrix(0B0000000001) * v00001 +
                matrix(0B0000000010) * v00010 + matrix(0B0000000011) * v00011 +
                matrix(0B0000000100) * v00100 + matrix(0B0000000101) * v00101 +
                matrix(0B0000000110) * v00110 + matrix(0B0000000111) * v00111 +
                matrix(0B0000001000) * v01000 + matrix(0B0000001001) * v01001 +
                matrix(0B0000001010) * v01010 + matrix(0B0000001011) * v01011 +
                matrix(0B0000001100) * v01100 + matrix(0B0000001101) * v01101 +
                matrix(0B0000001110) * v01110 + matrix(0B0000001111) * v01111 +
                matrix(0B0000010000) * v10000 + matrix(0B0000010001) * v10001 +
                matrix(0B0000010010) * v10010 + matrix(0B0000010011) * v10011 +
                matrix(0B0000010100) * v10100 + matrix(0B0000010101) * v10101 +
                matrix(0B0000010110) * v10110 + matrix(0B0000010111) * v10111 +
                matrix(0B0000011000) * v11000 + matrix(0B0000011001) * v11001 +
                matrix(0B0000011010) * v11010 + matrix(0B0000011011) * v11011 +
                matrix(0B0000011100) * v11100 + matrix(0B0000011101) * v11101 +
                matrix(0B0000011110) * v11110 + matrix(0B0000011111) * v11111;
            arr(i00001) =
                matrix(0B0000100000) * v00000 + matrix(0B0000100001) * v00001 +
                matrix(0B0000100010) * v00010 + matrix(0B0000100011) * v00011 +
                matrix(0B0000100100) * v00100 + matrix(0B0000100101) * v00101 +
                matrix(0B0000100110) * v00110 + matrix(0B0000100111) * v00111 +
                matrix(0B0000101000) * v01000 + matrix(0B0000101001) * v01001 +
                matrix(0B0000101010) * v01010 + matrix(0B0000101011) * v01011 +
                matrix(0B0000101100) * v01100 + matrix(0B0000101101) * v01101 +
                matrix(0B0000101110) * v01110 + matrix(0B0000101111) * v01111 +
                matrix(0B0000110000) * v10000 + matrix(0B0000110001) * v10001 +
                matrix(0B0000110010) * v10010 + matrix(0B0000110011) * v10011 +
                matrix(0B0000110100) * v10100 + matrix(0B0000110101) * v10101 +
                matrix(0B0000110110) * v10110 + matrix(0B0000110111) * v10111 +
                matrix(0B0000111000) * v11000 + matrix(0B0000111001) * v11001 +
                matrix(0B0000111010) * v11010 + matrix(0B0000111011) * v11011 +
                matrix(0B0000111100) * v11100 + matrix(0B0000111101) * v11101 +
                matrix(0B0000111110) * v11110 + matrix(0B0000111111) * v11111;
            arr(i00010) =
                matrix(0B0001000000) * v00000 + matrix(0B0001000001) * v00001 +
                matrix(0B0001000010) * v00010 + matrix(0B0001000011) * v00011 +
                matrix(0B0001000100) * v00100 + matrix(0B0001000101) * v00101 +
                matrix(0B0001000110) * v00110 + matrix(0B0001000111) * v00111 +
                matrix(0B0001001000) * v01000 + matrix(0B0001001001) * v01001 +
                matrix(0B0001001010) * v01010 + matrix(0B0001001011) * v01011 +
                matrix(0B0001001100) * v01100 + matrix(0B0001001101) * v01101 +
                matrix(0B0001001110) * v01110 + matrix(0B0001001111) * v01111 +
                matrix(0B0001010000) * v10000 + matrix(0B0001010001) * v10001 +
                matrix(0B0001010010) * v10010 + matrix(0B0001010011) * v10011 +
                matrix(0B0001010100) * v10100 + matrix(0B0001010101) * v10101 +
                matrix(0B0001010110) * v10110 + matrix(0B0001010111) * v10111 +
                matrix(0B0001011000) * v11000 + matrix(0B0001011001) * v11001 +
                matrix(0B0001011010) * v11010 + matrix(0B0001011011) * v11011 +
                matrix(0B0001011100) * v11100 + matrix(0B0001011101) * v11101 +
                matrix(0B0001011110) * v11110 + matrix(0B0001011111) * v11111;
            arr(i00011) =
                matrix(0B0001100000) * v00000 + matrix(0B0001100001) * v00001 +
                matrix(0B0001100010) * v00010 + matrix(0B0001100011) * v00011 +
                matrix(0B0001100100) * v00100 + matrix(0B0001100101) * v00101 +
                matrix(0B0001100110) * v00110 + matrix(0B0001100111) * v00111 +
                matrix(0B0001101000) * v01000 + matrix(0B0001101001) * v01001 +
                matrix(0B0001101010) * v01010 + matrix(0B0001101011) * v01011 +
                matrix(0B0001101100) * v01100 + matrix(0B0001101101) * v01101 +
                matrix(0B0001101110) * v01110 + matrix(0B0001101111) * v01111 +
                matrix(0B0001110000) * v10000 + matrix(0B0001110001) * v10001 +
                matrix(0B0001110010) * v10010 + matrix(0B0001110011) * v10011 +
                matrix(0B0001110100) * v10100 + matrix(0B0001110101) * v10101 +
                matrix(0B0001110110) * v10110 + matrix(0B0001110111) * v10111 +
                matrix(0B0001111000) * v11000 + matrix(0B0001111001) * v11001 +
                matrix(0B0001111010) * v11010 + matrix(0B0001111011) * v11011 +
                matrix(0B0001111100) * v11100 + matrix(0B0001111101) * v11101 +
                matrix(0B0001111110) * v11110 + matrix(0B0001111111) * v11111;
            arr(i00100) =
                matrix(0B0010000000) * v00000 + matrix(0B0010000001) * v00001 +
                matrix(0B0010000010) * v00010 + matrix(0B0010000011) * v00011 +
                matrix(0B0010000100) * v00100 + matrix(0B0010000101) * v00101 +
                matrix(0B0010000110) * v00110 + matrix(0B0010000111) * v00111 +
                matrix(0B0010001000) * v01000 + matrix(0B0010001001) * v01001 +
                matrix(0B0010001010) * v01010 + matrix(0B0010001011) * v01011 +
                matrix(0B0010001100) * v01100 + matrix(0B0010001101) * v01101 +
                matrix(0B0010001110) * v01110 + matrix(0B0010001111) * v01111 +
                matrix(0B0010010000) * v10000 + matrix(0B0010010001) * v10001 +
                matrix(0B0010010010) * v10010 + matrix(0B0010010011) * v10011 +
                matrix(0B0010010100) * v10100 + matrix(0B0010010101) * v10101 +
                matrix(0B0010010110) * v10110 + matrix(0B0010010111) * v10111 +
                matrix(0B0010011000) * v11000 + matrix(0B0010011001) * v11001 +
                matrix(0B0010011010) * v11010 + matrix(0B0010011011) * v11011 +
                matrix(0B0010011100) * v11100 + matrix(0B0010011101) * v11101 +
                matrix(0B0010011110) * v11110 + matrix(0B0010011111) * v11111;
            arr(i00101) =
                matrix(0B0010100000) * v00000 + matrix(0B0010100001) * v00001 +
                matrix(0B0010100010) * v00010 + matrix(0B0010100011) * v00011 +
                matrix(0B0010100100) * v00100 + matrix(0B0010100101) * v00101 +
                matrix(0B0010100110) * v00110 + matrix(0B0010100111) * v00111 +
                matrix(0B0010101000) * v01000 + matrix(0B0010101001) * v01001 +
                matrix(0B0010101010) * v01010 + matrix(0B0010101011) * v01011 +
                matrix(0B0010101100) * v01100 + matrix(0B0010101101) * v01101 +
                matrix(0B0010101110) * v01110 + matrix(0B0010101111) * v01111 +
                matrix(0B0010110000) * v10000 + matrix(0B0010110001) * v10001 +
                matrix(0B0010110010) * v10010 + matrix(0B0010110011) * v10011 +
                matrix(0B0010110100) * v10100 + matrix(0B0010110101) * v10101 +
                matrix(0B0010110110) * v10110 + matrix(0B0010110111) * v10111 +
                matrix(0B0010111000) * v11000 + matrix(0B0010111001) * v11001 +
                matrix(0B0010111010) * v11010 + matrix(0B0010111011) * v11011 +
                matrix(0B0010111100) * v11100 + matrix(0B0010111101) * v11101 +
                matrix(0B0010111110) * v11110 + matrix(0B0010111111) * v11111;
            arr(i00110) =
                matrix(0B0011000000) * v00000 + matrix(0B0011000001) * v00001 +
                matrix(0B0011000010) * v00010 + matrix(0B0011000011) * v00011 +
                matrix(0B0011000100) * v00100 + matrix(0B0011000101) * v00101 +
                matrix(0B0011000110) * v00110 + matrix(0B0011000111) * v00111 +
                matrix(0B0011001000) * v01000 + matrix(0B0011001001) * v01001 +
                matrix(0B0011001010) * v01010 + matrix(0B0011001011) * v01011 +
                matrix(0B0011001100) * v01100 + matrix(0B0011001101) * v01101 +
                matrix(0B0011001110) * v01110 + matrix(0B0011001111) * v01111 +
                matrix(0B0011010000) * v10000 + matrix(0B0011010001) * v10001 +
                matrix(0B0011010010) * v10010 + matrix(0B0011010011) * v10011 +
                matrix(0B0011010100) * v10100 + matrix(0B0011010101) * v10101 +
                matrix(0B0011010110) * v10110 + matrix(0B0011010111) * v10111 +
                matrix(0B0011011000) * v11000 + matrix(0B0011011001) * v11001 +
                matrix(0B0011011010) * v11010 + matrix(0B0011011011) * v11011 +
                matrix(0B0011011100) * v11100 + matrix(0B0011011101) * v11101 +
                matrix(0B0011011110) * v11110 + matrix(0B0011011111) * v11111;
            arr(i00111) =
                matrix(0B0011100000) * v00000 + matrix(0B0011100001) * v00001 +
                matrix(0B0011100010) * v00010 + matrix(0B0011100011) * v00011 +
                matrix(0B0011100100) * v00100 + matrix(0B0011100101) * v00101 +
                matrix(0B0011100110) * v00110 + matrix(0B0011100111) * v00111 +
                matrix(0B0011101000) * v01000 + matrix(0B0011101001) * v01001 +
                matrix(0B0011101010) * v01010 + matrix(0B0011101011) * v01011 +
                matrix(0B0011101100) * v01100 + matrix(0B0011101101) * v01101 +
                matrix(0B0011101110) * v01110 + matrix(0B0011101111) * v01111 +
                matrix(0B0011110000) * v10000 + matrix(0B0011110001) * v10001 +
                matrix(0B0011110010) * v10010 + matrix(0B0011110011) * v10011 +
                matrix(0B0011110100) * v10100 + matrix(0B0011110101) * v10101 +
                matrix(0B0011110110) * v10110 + matrix(0B0011110111) * v10111 +
                matrix(0B0011111000) * v11000 + matrix(0B0011111001) * v11001 +
                matrix(0B0011111010) * v11010 + matrix(0B0011111011) * v11011 +
                matrix(0B0011111100) * v11100 + matrix(0B0011111101) * v11101 +
                matrix(0B0011111110) * v11110 + matrix(0B0011111111) * v11111;
            arr(i01000) =
                matrix(0B0100000000) * v00000 + matrix(0B0100000001) * v00001 +
                matrix(0B0100000010) * v00010 + matrix(0B0100000011) * v00011 +
                matrix(0B0100000100) * v00100 + matrix(0B0100000101) * v00101 +
                matrix(0B0100000110) * v00110 + matrix(0B0100000111) * v00111 +
                matrix(0B0100001000) * v01000 + matrix(0B0100001001) * v01001 +
                matrix(0B0100001010) * v01010 + matrix(0B0100001011) * v01011 +
                matrix(0B0100001100) * v01100 + matrix(0B0100001101) * v01101 +
                matrix(0B0100001110) * v01110 + matrix(0B0100001111) * v01111 +
                matrix(0B0100010000) * v10000 + matrix(0B0100010001) * v10001 +
                matrix(0B0100010010) * v10010 + matrix(0B0100010011) * v10011 +
                matrix(0B0100010100) * v10100 + matrix(0B0100010101) * v10101 +
                matrix(0B0100010110) * v10110 + matrix(0B0100010111) * v10111 +
                matrix(0B0100011000) * v11000 + matrix(0B0100011001) * v11001 +
                matrix(0B0100011010) * v11010 + matrix(0B0100011011) * v11011 +
                matrix(0B0100011100) * v11100 + matrix(0B0100011101) * v11101 +
                matrix(0B0100011110) * v11110 + matrix(0B0100011111) * v11111;
            arr(i01001) =
                matrix(0B0100100000) * v00000 + matrix(0B0100100001) * v00001 +
                matrix(0B0100100010) * v00010 + matrix(0B0100100011) * v00011 +
                matrix(0B0100100100) * v00100 + matrix(0B0100100101) * v00101 +
                matrix(0B0100100110) * v00110 + matrix(0B0100100111) * v00111 +
                matrix(0B0100101000) * v01000 + matrix(0B0100101001) * v01001 +
                matrix(0B0100101010) * v01010 + matrix(0B0100101011) * v01011 +
                matrix(0B0100101100) * v01100 + matrix(0B0100101101) * v01101 +
                matrix(0B0100101110) * v01110 + matrix(0B0100101111) * v01111 +
                matrix(0B0100110000) * v10000 + matrix(0B0100110001) * v10001 +
                matrix(0B0100110010) * v10010 + matrix(0B0100110011) * v10011 +
                matrix(0B0100110100) * v10100 + matrix(0B0100110101) * v10101 +
                matrix(0B0100110110) * v10110 + matrix(0B0100110111) * v10111 +
                matrix(0B0100111000) * v11000 + matrix(0B0100111001) * v11001 +
                matrix(0B0100111010) * v11010 + matrix(0B0100111011) * v11011 +
                matrix(0B0100111100) * v11100 + matrix(0B0100111101) * v11101 +
                matrix(0B0100111110) * v11110 + matrix(0B0100111111) * v11111;
            arr(i01010) =
                matrix(0B0101000000) * v00000 + matrix(0B0101000001) * v00001 +
                matrix(0B0101000010) * v00010 + matrix(0B0101000011) * v00011 +
                matrix(0B0101000100) * v00100 + matrix(0B0101000101) * v00101 +
                matrix(0B0101000110) * v00110 + matrix(0B0101000111) * v00111 +
                matrix(0B0101001000) * v01000 + matrix(0B0101001001) * v01001 +
                matrix(0B0101001010) * v01010 + matrix(0B0101001011) * v01011 +
                matrix(0B0101001100) * v01100 + matrix(0B0101001101) * v01101 +
                matrix(0B0101001110) * v01110 + matrix(0B0101001111) * v01111 +
                matrix(0B0101010000) * v10000 + matrix(0B0101010001) * v10001 +
                matrix(0B0101010010) * v10010 + matrix(0B0101010011) * v10011 +
                matrix(0B0101010100) * v10100 + matrix(0B0101010101) * v10101 +
                matrix(0B0101010110) * v10110 + matrix(0B0101010111) * v10111 +
                matrix(0B0101011000) * v11000 + matrix(0B0101011001) * v11001 +
                matrix(0B0101011010) * v11010 + matrix(0B0101011011) * v11011 +
                matrix(0B0101011100) * v11100 + matrix(0B0101011101) * v11101 +
                matrix(0B0101011110) * v11110 + matrix(0B0101011111) * v11111;
            arr(i01011) =
                matrix(0B0101100000) * v00000 + matrix(0B0101100001) * v00001 +
                matrix(0B0101100010) * v00010 + matrix(0B0101100011) * v00011 +
                matrix(0B0101100100) * v00100 + matrix(0B0101100101) * v00101 +
                matrix(0B0101100110) * v00110 + matrix(0B0101100111) * v00111 +
                matrix(0B0101101000) * v01000 + matrix(0B0101101001) * v01001 +
                matrix(0B0101101010) * v01010 + matrix(0B0101101011) * v01011 +
                matrix(0B0101101100) * v01100 + matrix(0B0101101101) * v01101 +
                matrix(0B0101101110) * v01110 + matrix(0B0101101111) * v01111 +
                matrix(0B0101110000) * v10000 + matrix(0B0101110001) * v10001 +
                matrix(0B0101110010) * v10010 + matrix(0B0101110011) * v10011 +
                matrix(0B0101110100) * v10100 + matrix(0B0101110101) * v10101 +
                matrix(0B0101110110) * v10110 + matrix(0B0101110111) * v10111 +
                matrix(0B0101111000) * v11000 + matrix(0B0101111001) * v11001 +
                matrix(0B0101111010) * v11010 + matrix(0B0101111011) * v11011 +
                matrix(0B0101111100) * v11100 + matrix(0B0101111101) * v11101 +
                matrix(0B0101111110) * v11110 + matrix(0B0101111111) * v11111;
            arr(i01100) =
                matrix(0B0110000000) * v00000 + matrix(0B0110000001) * v00001 +
                matrix(0B0110000010) * v00010 + matrix(0B0110000011) * v00011 +
                matrix(0B0110000100) * v00100 + matrix(0B0110000101) * v00101 +
                matrix(0B0110000110) * v00110 + matrix(0B0110000111) * v00111 +
                matrix(0B0110001000) * v01000 + matrix(0B0110001001) * v01001 +
                matrix(0B0110001010) * v01010 + matrix(0B0110001011) * v01011 +
                matrix(0B0110001100) * v01100 + matrix(0B0110001101) * v01101 +
                matrix(0B0110001110) * v01110 + matrix(0B0110001111) * v01111 +
                matrix(0B0110010000) * v10000 + matrix(0B0110010001) * v10001 +
                matrix(0B0110010010) * v10010 + matrix(0B0110010011) * v10011 +
                matrix(0B0110010100) * v10100 + matrix(0B0110010101) * v10101 +
                matrix(0B0110010110) * v10110 + matrix(0B0110010111) * v10111 +
                matrix(0B0110011000) * v11000 + matrix(0B0110011001) * v11001 +
                matrix(0B0110011010) * v11010 + matrix(0B0110011011) * v11011 +
                matrix(0B0110011100) * v11100 + matrix(0B0110011101) * v11101 +
                matrix(0B0110011110) * v11110 + matrix(0B0110011111) * v11111;
            arr(i01101) =
                matrix(0B0110100000) * v00000 + matrix(0B0110100001) * v00001 +
                matrix(0B0110100010) * v00010 + matrix(0B0110100011) * v00011 +
                matrix(0B0110100100) * v00100 + matrix(0B0110100101) * v00101 +
                matrix(0B0110100110) * v00110 + matrix(0B0110100111) * v00111 +
                matrix(0B0110101000) * v01000 + matrix(0B0110101001) * v01001 +
                matrix(0B0110101010) * v01010 + matrix(0B0110101011) * v01011 +
                matrix(0B0110101100) * v01100 + matrix(0B0110101101) * v01101 +
                matrix(0B0110101110) * v01110 + matrix(0B0110101111) * v01111 +
                matrix(0B0110110000) * v10000 + matrix(0B0110110001) * v10001 +
                matrix(0B0110110010) * v10010 + matrix(0B0110110011) * v10011 +
                matrix(0B0110110100) * v10100 + matrix(0B0110110101) * v10101 +
                matrix(0B0110110110) * v10110 + matrix(0B0110110111) * v10111 +
                matrix(0B0110111000) * v11000 + matrix(0B0110111001) * v11001 +
                matrix(0B0110111010) * v11010 + matrix(0B0110111011) * v11011 +
                matrix(0B0110111100) * v11100 + matrix(0B0110111101) * v11101 +
                matrix(0B0110111110) * v11110 + matrix(0B0110111111) * v11111;
            arr(i01110) =
                matrix(0B0111000000) * v00000 + matrix(0B0111000001) * v00001 +
                matrix(0B0111000010) * v00010 + matrix(0B0111000011) * v00011 +
                matrix(0B0111000100) * v00100 + matrix(0B0111000101) * v00101 +
                matrix(0B0111000110) * v00110 + matrix(0B0111000111) * v00111 +
                matrix(0B0111001000) * v01000 + matrix(0B0111001001) * v01001 +
                matrix(0B0111001010) * v01010 + matrix(0B0111001011) * v01011 +
                matrix(0B0111001100) * v01100 + matrix(0B0111001101) * v01101 +
                matrix(0B0111001110) * v01110 + matrix(0B0111001111) * v01111 +
                matrix(0B0111010000) * v10000 + matrix(0B0111010001) * v10001 +
                matrix(0B0111010010) * v10010 + matrix(0B0111010011) * v10011 +
                matrix(0B0111010100) * v10100 + matrix(0B0111010101) * v10101 +
                matrix(0B0111010110) * v10110 + matrix(0B0111010111) * v10111 +
                matrix(0B0111011000) * v11000 + matrix(0B0111011001) * v11001 +
                matrix(0B0111011010) * v11010 + matrix(0B0111011011) * v11011 +
                matrix(0B0111011100) * v11100 + matrix(0B0111011101) * v11101 +
                matrix(0B0111011110) * v11110 + matrix(0B0111011111) * v11111;
            arr(i01111) =
                matrix(0B0111100000) * v00000 + matrix(0B0111100001) * v00001 +
                matrix(0B0111100010) * v00010 + matrix(0B0111100011) * v00011 +
                matrix(0B0111100100) * v00100 + matrix(0B0111100101) * v00101 +
                matrix(0B0111100110) * v00110 + matrix(0B0111100111) * v00111 +
                matrix(0B0111101000) * v01000 + matrix(0B0111101001) * v01001 +
                matrix(0B0111101010) * v01010 + matrix(0B0111101011) * v01011 +
                matrix(0B0111101100) * v01100 + matrix(0B0111101101) * v01101 +
                matrix(0B0111101110) * v01110 + matrix(0B0111101111) * v01111 +
                matrix(0B0111110000) * v10000 + matrix(0B0111110001) * v10001 +
                matrix(0B0111110010) * v10010 + matrix(0B0111110011) * v10011 +
                matrix(0B0111110100) * v10100 + matrix(0B0111110101) * v10101 +
                matrix(0B0111110110) * v10110 + matrix(0B0111110111) * v10111 +
                matrix(0B0111111000) * v11000 + matrix(0B0111111001) * v11001 +
                matrix(0B0111111010) * v11010 + matrix(0B0111111011) * v11011 +
                matrix(0B0111111100) * v11100 + matrix(0B0111111101) * v11101 +
                matrix(0B0111111110) * v11110 + matrix(0B0111111111) * v11111;
            arr(i10000) =
                matrix(0B1000000000) * v00000 + matrix(0B1000000001) * v00001 +
                matrix(0B1000000010) * v00010 + matrix(0B1000000011) * v00011 +
                matrix(0B1000000100) * v00100 + matrix(0B1000000101) * v00101 +
                matrix(0B1000000110) * v00110 + matrix(0B1000000111) * v00111 +
                matrix(0B1000001000) * v01000 + matrix(0B1000001001) * v01001 +
                matrix(0B1000001010) * v01010 + matrix(0B1000001011) * v01011 +
                matrix(0B1000001100) * v01100 + matrix(0B1000001101) * v01101 +
                matrix(0B1000001110) * v01110 + matrix(0B1000001111) * v01111 +
                matrix(0B1000010000) * v10000 + matrix(0B1000010001) * v10001 +
                matrix(0B1000010010) * v10010 + matrix(0B1000010011) * v10011 +
                matrix(0B1000010100) * v10100 + matrix(0B1000010101) * v10101 +
                matrix(0B1000010110) * v10110 + matrix(0B1000010111) * v10111 +
                matrix(0B1000011000) * v11000 + matrix(0B1000011001) * v11001 +
                matrix(0B1000011010) * v11010 + matrix(0B1000011011) * v11011 +
                matrix(0B1000011100) * v11100 + matrix(0B1000011101) * v11101 +
                matrix(0B1000011110) * v11110 + matrix(0B1000011111) * v11111;
            arr(i10001) =
                matrix(0B1000100000) * v00000 + matrix(0B1000100001) * v00001 +
                matrix(0B1000100010) * v00010 + matrix(0B1000100011) * v00011 +
                matrix(0B1000100100) * v00100 + matrix(0B1000100101) * v00101 +
                matrix(0B1000100110) * v00110 + matrix(0B1000100111) * v00111 +
                matrix(0B1000101000) * v01000 + matrix(0B1000101001) * v01001 +
                matrix(0B1000101010) * v01010 + matrix(0B1000101011) * v01011 +
                matrix(0B1000101100) * v01100 + matrix(0B1000101101) * v01101 +
                matrix(0B1000101110) * v01110 + matrix(0B1000101111) * v01111 +
                matrix(0B1000110000) * v10000 + matrix(0B1000110001) * v10001 +
                matrix(0B1000110010) * v10010 + matrix(0B1000110011) * v10011 +
                matrix(0B1000110100) * v10100 + matrix(0B1000110101) * v10101 +
                matrix(0B1000110110) * v10110 + matrix(0B1000110111) * v10111 +
                matrix(0B1000111000) * v11000 + matrix(0B1000111001) * v11001 +
                matrix(0B1000111010) * v11010 + matrix(0B1000111011) * v11011 +
                matrix(0B1000111100) * v11100 + matrix(0B1000111101) * v11101 +
                matrix(0B1000111110) * v11110 + matrix(0B1000111111) * v11111;
            arr(i10010) =
                matrix(0B1001000000) * v00000 + matrix(0B1001000001) * v00001 +
                matrix(0B1001000010) * v00010 + matrix(0B1001000011) * v00011 +
                matrix(0B1001000100) * v00100 + matrix(0B1001000101) * v00101 +
                matrix(0B1001000110) * v00110 + matrix(0B1001000111) * v00111 +
                matrix(0B1001001000) * v01000 + matrix(0B1001001001) * v01001 +
                matrix(0B1001001010) * v01010 + matrix(0B1001001011) * v01011 +
                matrix(0B1001001100) * v01100 + matrix(0B1001001101) * v01101 +
                matrix(0B1001001110) * v01110 + matrix(0B1001001111) * v01111 +
                matrix(0B1001010000) * v10000 + matrix(0B1001010001) * v10001 +
                matrix(0B1001010010) * v10010 + matrix(0B1001010011) * v10011 +
                matrix(0B1001010100) * v10100 + matrix(0B1001010101) * v10101 +
                matrix(0B1001010110) * v10110 + matrix(0B1001010111) * v10111 +
                matrix(0B1001011000) * v11000 + matrix(0B1001011001) * v11001 +
                matrix(0B1001011010) * v11010 + matrix(0B1001011011) * v11011 +
                matrix(0B1001011100) * v11100 + matrix(0B1001011101) * v11101 +
                matrix(0B1001011110) * v11110 + matrix(0B1001011111) * v11111;
            arr(i10011) =
                matrix(0B1001100000) * v00000 + matrix(0B1001100001) * v00001 +
                matrix(0B1001100010) * v00010 + matrix(0B1001100011) * v00011 +
                matrix(0B1001100100) * v00100 + matrix(0B1001100101) * v00101 +
                matrix(0B1001100110) * v00110 + matrix(0B1001100111) * v00111 +
                matrix(0B1001101000) * v01000 + matrix(0B1001101001) * v01001 +
                matrix(0B1001101010) * v01010 + matrix(0B1001101011) * v01011 +
                matrix(0B1001101100) * v01100 + matrix(0B1001101101) * v01101 +
                matrix(0B1001101110) * v01110 + matrix(0B1001101111) * v01111 +
                matrix(0B1001110000) * v10000 + matrix(0B1001110001) * v10001 +
                matrix(0B1001110010) * v10010 + matrix(0B1001110011) * v10011 +
                matrix(0B1001110100) * v10100 + matrix(0B1001110101) * v10101 +
                matrix(0B1001110110) * v10110 + matrix(0B1001110111) * v10111 +
                matrix(0B1001111000) * v11000 + matrix(0B1001111001) * v11001 +
                matrix(0B1001111010) * v11010 + matrix(0B1001111011) * v11011 +
                matrix(0B1001111100) * v11100 + matrix(0B1001111101) * v11101 +
                matrix(0B1001111110) * v11110 + matrix(0B1001111111) * v11111;
            arr(i10100) =
                matrix(0B1010000000) * v00000 + matrix(0B1010000001) * v00001 +
                matrix(0B1010000010) * v00010 + matrix(0B1010000011) * v00011 +
                matrix(0B1010000100) * v00100 + matrix(0B1010000101) * v00101 +
                matrix(0B1010000110) * v00110 + matrix(0B1010000111) * v00111 +
                matrix(0B1010001000) * v01000 + matrix(0B1010001001) * v01001 +
                matrix(0B1010001010) * v01010 + matrix(0B1010001011) * v01011 +
                matrix(0B1010001100) * v01100 + matrix(0B1010001101) * v01101 +
                matrix(0B1010001110) * v01110 + matrix(0B1010001111) * v01111 +
                matrix(0B1010010000) * v10000 + matrix(0B1010010001) * v10001 +
                matrix(0B1010010010) * v10010 + matrix(0B1010010011) * v10011 +
                matrix(0B1010010100) * v10100 + matrix(0B1010010101) * v10101 +
                matrix(0B1010010110) * v10110 + matrix(0B1010010111) * v10111 +
                matrix(0B1010011000) * v11000 + matrix(0B1010011001) * v11001 +
                matrix(0B1010011010) * v11010 + matrix(0B1010011011) * v11011 +
                matrix(0B1010011100) * v11100 + matrix(0B1010011101) * v11101 +
                matrix(0B1010011110) * v11110 + matrix(0B1010011111) * v11111;
            arr(i10101) =
                matrix(0B1010100000) * v00000 + matrix(0B1010100001) * v00001 +
                matrix(0B1010100010) * v00010 + matrix(0B1010100011) * v00011 +
                matrix(0B1010100100) * v00100 + matrix(0B1010100101) * v00101 +
                matrix(0B1010100110) * v00110 + matrix(0B1010100111) * v00111 +
                matrix(0B1010101000) * v01000 + matrix(0B1010101001) * v01001 +
                matrix(0B1010101010) * v01010 + matrix(0B1010101011) * v01011 +
                matrix(0B1010101100) * v01100 + matrix(0B1010101101) * v01101 +
                matrix(0B1010101110) * v01110 + matrix(0B1010101111) * v01111 +
                matrix(0B1010110000) * v10000 + matrix(0B1010110001) * v10001 +
                matrix(0B1010110010) * v10010 + matrix(0B1010110011) * v10011 +
                matrix(0B1010110100) * v10100 + matrix(0B1010110101) * v10101 +
                matrix(0B1010110110) * v10110 + matrix(0B1010110111) * v10111 +
                matrix(0B1010111000) * v11000 + matrix(0B1010111001) * v11001 +
                matrix(0B1010111010) * v11010 + matrix(0B1010111011) * v11011 +
                matrix(0B1010111100) * v11100 + matrix(0B1010111101) * v11101 +
                matrix(0B1010111110) * v11110 + matrix(0B1010111111) * v11111;
            arr(i10110) =
                matrix(0B1011000000) * v00000 + matrix(0B1011000001) * v00001 +
                matrix(0B1011000010) * v00010 + matrix(0B1011000011) * v00011 +
                matrix(0B1011000100) * v00100 + matrix(0B1011000101) * v00101 +
                matrix(0B1011000110) * v00110 + matrix(0B1011000111) * v00111 +
                matrix(0B1011001000) * v01000 + matrix(0B1011001001) * v01001 +
                matrix(0B1011001010) * v01010 + matrix(0B1011001011) * v01011 +
                matrix(0B1011001100) * v01100 + matrix(0B1011001101) * v01101 +
                matrix(0B1011001110) * v01110 + matrix(0B1011001111) * v01111 +
                matrix(0B1011010000) * v10000 + matrix(0B1011010001) * v10001 +
                matrix(0B1011010010) * v10010 + matrix(0B1011010011) * v10011 +
                matrix(0B1011010100) * v10100 + matrix(0B1011010101) * v10101 +
                matrix(0B1011010110) * v10110 + matrix(0B1011010111) * v10111 +
                matrix(0B1011011000) * v11000 + matrix(0B1011011001) * v11001 +
                matrix(0B1011011010) * v11010 + matrix(0B1011011011) * v11011 +
                matrix(0B1011011100) * v11100 + matrix(0B1011011101) * v11101 +
                matrix(0B1011011110) * v11110 + matrix(0B1011011111) * v11111;
            arr(i10111) =
                matrix(0B1011100000) * v00000 + matrix(0B1011100001) * v00001 +
                matrix(0B1011100010) * v00010 + matrix(0B1011100011) * v00011 +
                matrix(0B1011100100) * v00100 + matrix(0B1011100101) * v00101 +
                matrix(0B1011100110) * v00110 + matrix(0B1011100111) * v00111 +
                matrix(0B1011101000) * v01000 + matrix(0B1011101001) * v01001 +
                matrix(0B1011101010) * v01010 + matrix(0B1011101011) * v01011 +
                matrix(0B1011101100) * v01100 + matrix(0B1011101101) * v01101 +
                matrix(0B1011101110) * v01110 + matrix(0B1011101111) * v01111 +
                matrix(0B1011110000) * v10000 + matrix(0B1011110001) * v10001 +
                matrix(0B1011110010) * v10010 + matrix(0B1011110011) * v10011 +
                matrix(0B1011110100) * v10100 + matrix(0B1011110101) * v10101 +
                matrix(0B1011110110) * v10110 + matrix(0B1011110111) * v10111 +
                matrix(0B1011111000) * v11000 + matrix(0B1011111001) * v11001 +
                matrix(0B1011111010) * v11010 + matrix(0B1011111011) * v11011 +
                matrix(0B1011111100) * v11100 + matrix(0B1011111101) * v11101 +
                matrix(0B1011111110) * v11110 + matrix(0B1011111111) * v11111;
            arr(i11000) =
                matrix(0B1100000000) * v00000 + matrix(0B1100000001) * v00001 +
                matrix(0B1100000010) * v00010 + matrix(0B1100000011) * v00011 +
                matrix(0B1100000100) * v00100 + matrix(0B1100000101) * v00101 +
                matrix(0B1100000110) * v00110 + matrix(0B1100000111) * v00111 +
                matrix(0B1100001000) * v01000 + matrix(0B1100001001) * v01001 +
                matrix(0B1100001010) * v01010 + matrix(0B1100001011) * v01011 +
                matrix(0B1100001100) * v01100 + matrix(0B1100001101) * v01101 +
                matrix(0B1100001110) * v01110 + matrix(0B1100001111) * v01111 +
                matrix(0B1100010000) * v10000 + matrix(0B1100010001) * v10001 +
                matrix(0B1100010010) * v10010 + matrix(0B1100010011) * v10011 +
                matrix(0B1100010100) * v10100 + matrix(0B1100010101) * v10101 +
                matrix(0B1100010110) * v10110 + matrix(0B1100010111) * v10111 +
                matrix(0B1100011000) * v11000 + matrix(0B1100011001) * v11001 +
                matrix(0B1100011010) * v11010 + matrix(0B1100011011) * v11011 +
                matrix(0B1100011100) * v11100 + matrix(0B1100011101) * v11101 +
                matrix(0B1100011110) * v11110 + matrix(0B1100011111) * v11111;
            arr(i11001) =
                matrix(0B1100100000) * v00000 + matrix(0B1100100001) * v00001 +
                matrix(0B1100100010) * v00010 + matrix(0B1100100011) * v00011 +
                matrix(0B1100100100) * v00100 + matrix(0B1100100101) * v00101 +
                matrix(0B1100100110) * v00110 + matrix(0B1100100111) * v00111 +
                matrix(0B1100101000) * v01000 + matrix(0B1100101001) * v01001 +
                matrix(0B1100101010) * v01010 + matrix(0B1100101011) * v01011 +
                matrix(0B1100101100) * v01100 + matrix(0B1100101101) * v01101 +
                matrix(0B1100101110) * v01110 + matrix(0B1100101111) * v01111 +
                matrix(0B1100110000) * v10000 + matrix(0B1100110001) * v10001 +
                matrix(0B1100110010) * v10010 + matrix(0B1100110011) * v10011 +
                matrix(0B1100110100) * v10100 + matrix(0B1100110101) * v10101 +
                matrix(0B1100110110) * v10110 + matrix(0B1100110111) * v10111 +
                matrix(0B1100111000) * v11000 + matrix(0B1100111001) * v11001 +
                matrix(0B1100111010) * v11010 + matrix(0B1100111011) * v11011 +
                matrix(0B1100111100) * v11100 + matrix(0B1100111101) * v11101 +
                matrix(0B1100111110) * v11110 + matrix(0B1100111111) * v11111;
            arr(i11010) =
                matrix(0B1101000000) * v00000 + matrix(0B1101000001) * v00001 +
                matrix(0B1101000010) * v00010 + matrix(0B1101000011) * v00011 +
                matrix(0B1101000100) * v00100 + matrix(0B1101000101) * v00101 +
                matrix(0B1101000110) * v00110 + matrix(0B1101000111) * v00111 +
                matrix(0B1101001000) * v01000 + matrix(0B1101001001) * v01001 +
                matrix(0B1101001010) * v01010 + matrix(0B1101001011) * v01011 +
                matrix(0B1101001100) * v01100 + matrix(0B1101001101) * v01101 +
                matrix(0B1101001110) * v01110 + matrix(0B1101001111) * v01111 +
                matrix(0B1101010000) * v10000 + matrix(0B1101010001) * v10001 +
                matrix(0B1101010010) * v10010 + matrix(0B1101010011) * v10011 +
                matrix(0B1101010100) * v10100 + matrix(0B1101010101) * v10101 +
                matrix(0B1101010110) * v10110 + matrix(0B1101010111) * v10111 +
                matrix(0B1101011000) * v11000 + matrix(0B1101011001) * v11001 +
                matrix(0B1101011010) * v11010 + matrix(0B1101011011) * v11011 +
                matrix(0B1101011100) * v11100 + matrix(0B1101011101) * v11101 +
                matrix(0B1101011110) * v11110 + matrix(0B1101011111) * v11111;
            arr(i11011) =
                matrix(0B1101100000) * v00000 + matrix(0B1101100001) * v00001 +
                matrix(0B1101100010) * v00010 + matrix(0B1101100011) * v00011 +
                matrix(0B1101100100) * v00100 + matrix(0B1101100101) * v00101 +
                matrix(0B1101100110) * v00110 + matrix(0B1101100111) * v00111 +
                matrix(0B1101101000) * v01000 + matrix(0B1101101001) * v01001 +
                matrix(0B1101101010) * v01010 + matrix(0B1101101011) * v01011 +
                matrix(0B1101101100) * v01100 + matrix(0B1101101101) * v01101 +
                matrix(0B1101101110) * v01110 + matrix(0B1101101111) * v01111 +
                matrix(0B1101110000) * v10000 + matrix(0B1101110001) * v10001 +
                matrix(0B1101110010) * v10010 + matrix(0B1101110011) * v10011 +
                matrix(0B1101110100) * v10100 + matrix(0B1101110101) * v10101 +
                matrix(0B1101110110) * v10110 + matrix(0B1101110111) * v10111 +
                matrix(0B1101111000) * v11000 + matrix(0B1101111001) * v11001 +
                matrix(0B1101111010) * v11010 + matrix(0B1101111011) * v11011 +
                matrix(0B1101111100) * v11100 + matrix(0B1101111101) * v11101 +
                matrix(0B1101111110) * v11110 + matrix(0B1101111111) * v11111;
            arr(i11100) =
                matrix(0B1110000000) * v00000 + matrix(0B1110000001) * v00001 +
                matrix(0B1110000010) * v00010 + matrix(0B1110000011) * v00011 +
                matrix(0B1110000100) * v00100 + matrix(0B1110000101) * v00101 +
                matrix(0B1110000110) * v00110 + matrix(0B1110000111) * v00111 +
                matrix(0B1110001000) * v01000 + matrix(0B1110001001) * v01001 +
                matrix(0B1110001010) * v01010 + matrix(0B1110001011) * v01011 +
                matrix(0B1110001100) * v01100 + matrix(0B1110001101) * v01101 +
                matrix(0B1110001110) * v01110 + matrix(0B1110001111) * v01111 +
                matrix(0B1110010000) * v10000 + matrix(0B1110010001) * v10001 +
                matrix(0B1110010010) * v10010 + matrix(0B1110010011) * v10011 +
                matrix(0B1110010100) * v10100 + matrix(0B1110010101) * v10101 +
                matrix(0B1110010110) * v10110 + matrix(0B1110010111) * v10111 +
                matrix(0B1110011000) * v11000 + matrix(0B1110011001) * v11001 +
                matrix(0B1110011010) * v11010 + matrix(0B1110011011) * v11011 +
                matrix(0B1110011100) * v11100 + matrix(0B1110011101) * v11101 +
                matrix(0B1110011110) * v11110 + matrix(0B1110011111) * v11111;
            arr(i11101) =
                matrix(0B1110100000) * v00000 + matrix(0B1110100001) * v00001 +
                matrix(0B1110100010) * v00010 + matrix(0B1110100011) * v00011 +
                matrix(0B1110100100) * v00100 + matrix(0B1110100101) * v00101 +
                matrix(0B1110100110) * v00110 + matrix(0B1110100111) * v00111 +
                matrix(0B1110101000) * v01000 + matrix(0B1110101001) * v01001 +
                matrix(0B1110101010) * v01010 + matrix(0B1110101011) * v01011 +
                matrix(0B1110101100) * v01100 + matrix(0B1110101101) * v01101 +
                matrix(0B1110101110) * v01110 + matrix(0B1110101111) * v01111 +
                matrix(0B1110110000) * v10000 + matrix(0B1110110001) * v10001 +
                matrix(0B1110110010) * v10010 + matrix(0B1110110011) * v10011 +
                matrix(0B1110110100) * v10100 + matrix(0B1110110101) * v10101 +
                matrix(0B1110110110) * v10110 + matrix(0B1110110111) * v10111 +
                matrix(0B1110111000) * v11000 + matrix(0B1110111001) * v11001 +
                matrix(0B1110111010) * v11010 + matrix(0B1110111011) * v11011 +
                matrix(0B1110111100) * v11100 + matrix(0B1110111101) * v11101 +
                matrix(0B1110111110) * v11110 + matrix(0B1110111111) * v11111;
            arr(i11110) =
                matrix(0B1111000000) * v00000 + matrix(0B1111000001) * v00001 +
                matrix(0B1111000010) * v00010 + matrix(0B1111000011) * v00011 +
                matrix(0B1111000100) * v00100 + matrix(0B1111000101) * v00101 +
                matrix(0B1111000110) * v00110 + matrix(0B1111000111) * v00111 +
                matrix(0B1111001000) * v01000 + matrix(0B1111001001) * v01001 +
                matrix(0B1111001010) * v01010 + matrix(0B1111001011) * v01011 +
                matrix(0B1111001100) * v01100 + matrix(0B1111001101) * v01101 +
                matrix(0B1111001110) * v01110 + matrix(0B1111001111) * v01111 +
                matrix(0B1111010000) * v10000 + matrix(0B1111010001) * v10001 +
                matrix(0B1111010010) * v10010 + matrix(0B1111010011) * v10011 +
                matrix(0B1111010100) * v10100 + matrix(0B1111010101) * v10101 +
                matrix(0B1111010110) * v10110 + matrix(0B1111010111) * v10111 +
                matrix(0B1111011000) * v11000 + matrix(0B1111011001) * v11001 +
                matrix(0B1111011010) * v11010 + matrix(0B1111011011) * v11011 +
                matrix(0B1111011100) * v11100 + matrix(0B1111011101) * v11101 +
                matrix(0B1111011110) * v11110 + matrix(0B1111011111) * v11111;
            arr(i11111) =
                matrix(0B1111100000) * v00000 + matrix(0B1111100001) * v00001 +
                matrix(0B1111100010) * v00010 + matrix(0B1111100011) * v00011 +
                matrix(0B1111100100) * v00100 + matrix(0B1111100101) * v00101 +
                matrix(0B1111100110) * v00110 + matrix(0B1111100111) * v00111 +
                matrix(0B1111101000) * v01000 + matrix(0B1111101001) * v01001 +
                matrix(0B1111101010) * v01010 + matrix(0B1111101011) * v01011 +
                matrix(0B1111101100) * v01100 + matrix(0B1111101101) * v01101 +
                matrix(0B1111101110) * v01110 + matrix(0B1111101111) * v01111 +
                matrix(0B1111110000) * v10000 + matrix(0B1111110001) * v10001 +
                matrix(0B1111110010) * v10010 + matrix(0B1111110011) * v10011 +
                matrix(0B1111110100) * v10100 + matrix(0B1111110101) * v10101 +
                matrix(0B1111110110) * v10110 + matrix(0B1111110111) * v10111 +
                matrix(0B1111111000) * v11000 + matrix(0B1111111001) * v11001 +
                matrix(0B1111111010) * v11010 + matrix(0B1111111011) * v11011 +
                matrix(0B1111111100) * v11100 + matrix(0B1111111101) * v11101 +
                matrix(0B1111111110) * v11110 + matrix(0B1111111111) * v11111;
        }
    }
};

} // namespace Pennylane::LightningKokkos::Functors