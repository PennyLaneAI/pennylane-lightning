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
 * Defines common utility functions for all AVX
 */
#pragma once
#include "BitUtil.hpp"
#include "Macros.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <cstdlib>

namespace Pennylane::Gates::AVX {
using Pennylane::Util::exp2;
using Pennylane::Util::fillLeadingOnes;
using Pennylane::Util::fillTrailingOnes;

template <typename PrecisionT, size_t packed_size> struct AVXIntrinsic {
    static_assert((sizeof(PrecisionT) * packed_size == 32) ||
                  (sizeof(PrecisionT) * packed_size == 64));
};
template <typename T, size_t size>
using AVXIntrinsicType = typename AVXIntrinsic<T, size>::Type;

template <class PrecisionT, size_t packed_size> struct AVXConcept;

template <class PrecisionT, size_t packed_size>
using AVXConceptType = typename AVXConcept<PrecisionT, packed_size>::Type;

/**
 * @brief @rst
 * For a function :math:`f(x)` with binary output, this function create 
 * an AVX intrinsic floating-point type with values :math:`(-1)^{f(x)}`
 * where :math:`x` is index of an array (viewed as a complex-valued array).
 * @endrst
 *
 * @rst
 * For example, when :math:`f(x) = x % 2`, this returns a packed array
 * with values [1, 1, -1, -1, 1, 1, -1, -1]. Note that each value is repeated
 * twice as it applies to the both real and imaginary parts. This function is
 * used e.g. in CZ gate.
 * @endrst
 *
 * @tparam PrecisionT Floating point precision type
 * @tparam packed_size Number of packed values for a AVX intrinsic type
 * @tparam Func Type of a function
 * @param func Binary output function
 */
template <typename PrecisionT, size_t packed_size, typename Func>
auto toParity(Func &&func) -> AVXIntrinsicType<PrecisionT, packed_size> {
    std::array<PrecisionT, packed_size> data = {};
    for (size_t idx = 0; idx < packed_size / 2; idx++) {
        data[2 * idx + 0] = static_cast<PrecisionT>(1.0) -
                            2 * static_cast<PrecisionT>(func(idx));
        data[2 * idx + 1] = static_cast<PrecisionT>(1.0) -
                            2 * static_cast<PrecisionT>(func(idx));
    }
    return AVXConceptType<PrecisionT, packed_size>::loadu(data.data());
}

/**
 * @brief @rst
 *
 */
template <typename PrecisionT, size_t packed_size, typename Func>
auto setValueOneTwo(Func &&func) -> AVXIntrinsicType<PrecisionT, packed_size> {
    std::array<PrecisionT, packed_size> data = {
        0,
    };
    for (size_t idx = 0; idx < packed_size / 2; idx++) {
        data[2 * idx + 0] = func(idx);
        data[2 * idx + 1] = func(idx);
    }
    return AVXConceptType<PrecisionT, packed_size>::loadu(data.data());
}

/**
 * @brief one or minus one parity for reverse wire in packed data.
 *
 * All specializations are defined in AVX2Concept.hpp and AVX512Concept.hpp
 * files.
 */
template <typename PrecisionT, size_t packed_size>
constexpr auto internalParity(size_t rev_wire)
    -> AVXIntrinsicType<PrecisionT, packed_size>;

/**
 * @brief Factor that is applied to the intrinsic type for product of
 * pure imaginary value.
 *
 * Template specializations are defined in each AVX(2|512)Concept.hpp file.
 */
template <typename PrecisionT, size_t packed_size> struct ImagFactor;

template <typename PrecisionT, size_t packed_size>
constexpr auto imagFactor(PrecisionT val = 1.0) {
    return ImagFactor<PrecisionT, packed_size>::create(val);
}

template <typename PrecisionT, size_t packed_size> struct Set1;

template <typename PrecisionT, size_t packed_size>
constexpr auto set1(PrecisionT val) {
    return Set1<PrecisionT, packed_size>::create(val);
}

template <size_t packed_size> struct InternalWires {
    constexpr static auto value = Util::log2PerfectPower(packed_size / 2);
};
template <size_t packed_size>
constexpr auto internal_wires_v = InternalWires<packed_size>::value;

// clang-format off
#ifdef PL_USE_AVX2
constexpr __m256i setr256i(int32_t  e0, int32_t  e1, int32_t  e2, int32_t  e3,
		                   int32_t  e4, int32_t  e5, int32_t  e6, int32_t  e7) {
    return __m256i{(int64_t(e1) << 32) | e0,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e3) << 32) | e2,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e5) << 32) | e4,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e7) << 32) | e6}; // NOLINT(hicpp-signed-bitwise)
}
#endif
#ifdef PL_USE_AVX512F
constexpr __m512i setr512i(int32_t  e0, int32_t  e1, int32_t  e2, int32_t  e3,
		                   int32_t  e4, int32_t  e5, int32_t  e6, int32_t  e7, 
		                   int32_t  e8, int32_t  e9, int32_t e10, int32_t e11, 
		                   int32_t e12, int32_t e13, int32_t e14, int32_t e15) {
    return __m512i{(int64_t(e1) << 32)  |  e0,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e3) << 32)  |  e2,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e5) << 32)  |  e4,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e7) << 32)  |  e6,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e9) << 32)  |  e8,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e11) << 32) | e10,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e13) << 32) | e12,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e15) << 32) | e14}; // NOLINT(hicpp-signed-bitwise)
}
constexpr __m512i setr512i(int64_t  e0, int64_t  e1, int64_t  e2, int64_t  e3,
		                   int64_t  e4, int64_t  e5, int64_t  e6, int64_t  e7) {
    return __m512i{e0, e1, e2, e3, e4, e5, e6, e7};
}
#endif
// clang-format on

} // namespace Pennylane::Gates::AVX
