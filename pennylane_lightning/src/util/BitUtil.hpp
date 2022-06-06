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
#include <type_traits>

#if __has_include(<version>)
#include <version>
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
 * @brief Compute log2 of value in a compile-time.
 *
 * @param value Number to compute log2
 */
inline auto constexpr isPerfectPowerOf2(size_t value) -> bool {
#if __cpp_lib_int_pow2 >= 202002L
    return std::has_single_bit(value);
#else
    return std::popcount(value) == 1;
#endif
}

/**
 * @brief Fill ones from LSB to nbits. Runnable in a compile-time and for any
 * integer type.
 *
 * @tparam IntegerType Integer type to use
 * @param nbits Number of bits to fill
 */
template <class IntegerType = size_t>
inline auto constexpr fillTrailingOnes(size_t nbits) -> IntegerType {
    static_assert(std::is_integral_v<IntegerType> &&
                  std::is_unsigned_v<IntegerType>);

    return (nbits == 0) ? 0
                        : static_cast<IntegerType>(~IntegerType(0)) >>
                              static_cast<IntegerType>(
                                  CHAR_BIT * sizeof(IntegerType) - nbits);
}
/**
 * @brief Fill ones from MSB to pos
 *
 * @tparam IntegerType Integer type to use
 * @param pos Position up to which bit one is filled.
 */
template <class IntegerType = size_t>
inline auto constexpr fillLeadingOnes(size_t pos) -> size_t {
    static_assert(std::is_integral_v<IntegerType> &&
                  std::is_unsigned_v<IntegerType>);

    return (~IntegerType{0}) << pos;
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
