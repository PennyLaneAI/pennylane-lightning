// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Defines PhaseShift gate
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct ApplyPhaseShift {
    using Precision = PrecisionT;
    using PrecisionAVXConcept =
        typename AVXConcept<PrecisionT, packed_size>::Type;

    constexpr static size_t packed_size_ = packed_size;

    static constexpr auto createPermutation(size_t rev_wire) {
        std::array<uint8_t, packed_size> perm = {
            0,
        };

        for (size_t n = 0; n < packed_size / 2; n++) {
            if (((n >> rev_wire) & 1U) == 0) {
                perm[2 * n + 0] = 2 * n + 0;
                perm[2 * n + 1] = 2 * n + 1;
            } else {
                perm[2 * n + 0] = 2 * n + 1;
                perm[2 * n + 1] = 2 * n + 0;
            }
        }

        return Permutation::compilePermutation<PrecisionT>(perm);
    }

    static auto cosFactor(size_t rev_wire, PrecisionT cos)
        -> AVXIntrinsicType<PrecisionT, packed_size> {
        std::array<PrecisionT, packed_size> data = {
            0,
        };
        for (size_t n = 0; n < packed_size / 2; n++) {
            if (((n >> rev_wire) & 1U) == 0) {
                data[2 * n + 0] = 1.0;
                data[2 * n + 1] = 1.0;
            } else {
                data[2 * n + 0] = cos;
                data[2 * n + 1] = cos;
            }
        }
        return PrecisionAVXConcept::loadu(data.data());
    }

    static auto isinFactor(size_t rev_wire, PrecisionT isin)
        -> AVXIntrinsicType<PrecisionT, packed_size> {
        std::array<PrecisionT, packed_size> data = {
            0,
        };
        for (size_t n = 0; n < packed_size / 2; n++) {
            if (((n >> rev_wire) & 1U) == 0) {
                data[2 * n + 0] = 0.0;
                data[2 * n + 1] = 0.0;
            } else {
                data[2 * n + 0] = -isin;
                data[2 * n + 1] = isin;
            }
        }
        return PrecisionAVXConcept::loadu(data.data());
    }

    template <size_t rev_wire, typename ParamT>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, bool inverse,
                              ParamT angle) {
        constexpr static auto perm = createPermutation(rev_wire);
        const auto cos_factor =
            cosFactor(rev_wire, static_cast<PrecisionT>(cos(angle)));
        const auto isin_factor =
            isinFactor(rev_wire, static_cast<PrecisionT>(inverse ? -1.0 : 1.0) *
                                     static_cast<PrecisionT>(sin(angle)));

        for (size_t k = 0; k < (1U << num_qubits); k += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + k);
            const auto w =
                cos_factor * v + isin_factor * Permutation::permute<perm>(v);
            PrecisionAVXConcept::store(arr + k, w);
        }
    }

    template <typename ParamT>
    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, const size_t rev_wire,
                              bool inverse, ParamT angle) {
        using namespace Permutation;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const auto cos_factor =
            set1<PrecisionT, packed_size>(static_cast<PrecisionT>(cos(angle)));
        const auto isin_factor =
            set1<PrecisionT, packed_size>(inverse ? -1.0 : 1.0) *
            imagFactor<PrecisionT, packed_size>(
                static_cast<PrecisionT>(sin(angle)));
        constexpr static auto perm = compilePermutation<PrecisionT>(
            swapRealImag(identity<packed_size>()));

        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v1 = PrecisionAVXConcept::load(arr + i1);
            const auto w1 = cos_factor * v1 + isin_factor * permute<perm>(v1);
            PrecisionAVXConcept::store(arr + i1, w1);
        }
    }
};
} // namespace Pennylane::Gates::AVXCommon
