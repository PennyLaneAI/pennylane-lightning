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
// function aliases
[[maybe_unused]] constexpr static auto &fillLeadingOnes =
    Pennylane::Util::fillLeadingOnes<size_t>;
[[maybe_unused]] constexpr static auto &fillTrailingOnes =
    Pennylane::Util::fillTrailingOnes<size_t>;
[[maybe_unused]] constexpr static auto &exp2 = Pennylane::Util::exp2;

template <typename PrecisionT, size_t packed_size> struct AVXIntrinsic {
    static_assert((sizeof(PrecisionT) * packed_size == 32) ||
                  (sizeof(PrecisionT) * packed_size == 64));
};
template <typename T, size_t size>
using AVXIntrinsicType = typename AVXIntrinsic<T, size>::Type;

template <class PrecisionT, size_t packed_size> struct AVXConcept;

template <class PrecisionT, size_t packed_size>
using AVXConceptType = typename AVXConcept<PrecisionT, packed_size>::Type;

template <typename PrecisionT, size_t packed_size, typename Func>
auto toParity(Func &&func) -> decltype(auto) {
    std::array<PrecisionT, packed_size> data = {};
    for (size_t idx = 0; idx < packed_size / 2; idx++) {
        data[2 * idx + 0] = static_cast<PrecisionT>(1.0) -
                            2 * static_cast<PrecisionT>(func(idx));
        data[2 * idx + 1] = static_cast<PrecisionT>(1.0) -
                            2 * static_cast<PrecisionT>(func(idx));
    }
    return AVXConceptType<PrecisionT, packed_size>::loadu(data.data());
}
template <typename PrecisionT, size_t packed_size, typename Func>
auto setValueOneTwo(Func &&func) -> decltype(auto) {
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

} // namespace Pennylane::Gates::AVX
