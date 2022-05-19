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
 * Contains uncategorised utility functions.
 */
#pragma once
#include <bit>
#include <climits>
#include <cstdint>
#include <cstdlib>

#if defined(_MSC_VER)
#include <intrin.h> // for __lzcnt64 and __popcount
#endif

namespace Pennylane::Util {
/**
 * @brief Faster log2 when the value is the perfect power of 2.
 *
 * If the value is the perfect power of 2, using a system provided bit operation
 * is much faster than std::log2
 */
inline auto constexpr log2PerfectPower(uint64_t val) -> size_t {
    return static_cast<size_t>(std::countr_zero(val));
}

/**
 * @brief Check if there is a positive integer n such that value == 2^n.
 *
 * @param value Value to calculate for.
 * @return bool
 */
inline auto constexpr isPerfectPowerOf2(size_t value) -> bool {
#if __cpp_lib_int_pow2
    return std::has_single_bit(value);
#else
    return std::popcount(value) == 1;
#endif
}
/**
 * @brief Fill ones from LSB to rev_wire
 */
inline auto constexpr fillTrailingOnes(size_t pos) -> size_t {
    return (pos == 0) ? 0 : (~size_t(0) >> (CHAR_BIT * sizeof(size_t) - pos));
}
/**
 * @brief Fill ones from MSB to pos
 */
inline auto constexpr fillLeadingOnes(size_t pos) -> size_t {
    return (~size_t(0)) << pos;
}

/**
 * @brief Swap bits in i-th and j-th position in place
 */
inline auto constexpr bitswap(size_t bits, const size_t i, const size_t j)
    -> size_t {
    size_t x = ((bits >> i) ^ (bits >> j)) & 1U;
    return bits ^ ((x << i) | (x << j));
}
} // namespace Pennylane::Util
