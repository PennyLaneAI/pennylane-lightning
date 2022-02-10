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
#include <climits>
#include <cstdint>
#include <cstdlib>

#if defined(_MSC_VER)
#include <intrin.h> // for __lzcnt64 and __popcount
#endif

/// @cond DEV
namespace Pennylane::Util::Internal {
/**
 * @brief Count the number of 1s in the binary representation of n.
 *
 * @param n Unsigned 32 bit integer
 */
constexpr auto countBit1(uint32_t n) -> size_t {
    n = (n & 0x55555555U) +         // NOLINT(readability-magic-numbers)
        ((n >> 1U) & 0x55555555U);  // NOLINT(readability-magic-numbers)
    n = (n & 0x33333333U) +         // NOLINT(readability-magic-numbers)
        ((n >> 2U) & 0x33333333U);  // NOLINT(readability-magic-numbers)
    n = (n & 0x0F0F0F0FU) +         // NOLINT(readability-magic-numbers)
        ((n >> 4U) & 0x0F0F0F0FU);  // NOLINT(readability-magic-numbers)
    n = (n & 0X00FF00FFU) +         // NOLINT(readability-magic-numbers)
        ((n >> 8U) & 0x00FF00FFU);  // NOLINT(readability-magic-numbers)
    n = (n & 0X0000FFFFU) +         // NOLINT(readability-magic-numbers)
        ((n >> 16U) & 0x0000FFFFU); // NOLINT(readability-magic-numbers)
    return n;
}

/**
 * @brief Count the number of 1s in the binary representation of n.
 *
 * @param n Unsigned 64 bit integer
 */
constexpr auto countBit1(uint64_t n) -> size_t {
    return countBit1(static_cast<uint32_t>(
               n & 0xFFFFFFFFU)) + // NOLINT(readability-magic-numbers)
           countBit1(static_cast<uint32_t>(
               n >> 32U)); // NOLINT(readability-magic-numbers)
}

/**
 * @brief Lookup table for number of trailing zeros in the binary
 * representation for 0 to 255 (8 bit integers).
 */
// NOLINTNEXTLINE (readability-magic-numbers)
constexpr uint8_t TRAILING_ZERO_LOOKUP_TABLE[256] = {
    0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};

/**
 * @brief Number of trailing zeros (starting from LSB) in the binary
 * representation of n.
 *
 * @param n Unsigned 8 bit integer
 */
constexpr auto countTrailing0(uint8_t n) -> size_t {
    return TRAILING_ZERO_LOOKUP_TABLE[n];
}

/**
 * @brief Number of trailing zeros (starting from LSB) in the binary
 * representation of n.
 *
 * @param n Unsigned 16 bit integer
 */
constexpr auto countTrailing0(uint16_t n) -> size_t {
    // NOLINTNEXTLINE (readability-magic-numbers)
    if (const auto mod = (n & 0xFFU); mod != 0) {
        return countTrailing0(static_cast<uint8_t>(mod));
    }
    // NOLINTNEXTLINE (readability-magic-numbers)
    return countTrailing0(static_cast<uint8_t>(n >> 8U)) + 8U;
}

/**
 * @brief Number of trailing zeros (starting from LSB) in the binary
 * representation of n.
 *
 * @param n Unsigned 32 bit integer
 */
constexpr auto countTrailing0(uint32_t n) -> size_t {
    // NOLINTNEXTLINE (readability-magic-numbers)
    if (const auto mod = (n & 0xFFFFU); mod != 0) {
        return countTrailing0(static_cast<uint16_t>(mod));
    }
    // NOLINTNEXTLINE (readability-magic-numbers)
    return countTrailing0(static_cast<uint16_t>(n >> 16U)) + 16U;
}
constexpr auto countTrailing0(uint64_t n) -> size_t {
    // NOLINTNEXTLINE (readability-magic-numbers)
    if (const auto mod = (n & 0xFFFFFFFFU); mod != 0) {
        return countTrailing0(static_cast<uint32_t>(mod));
    }
    // NOLINTNEXTLINE (readability-magic-numbers)
    return countTrailing0(static_cast<uint32_t>(n >> 32U)) + 32U;
}
} // namespace Pennylane::Util::Internal
/// @endcond

namespace Pennylane::Util {
/**
 * @brief Define popcount for multiple compilers as well as different types.
 *
 * TODO: change to std::popcount in C++20
 */
///@{
#if defined(_MSC_VER)
inline auto popcount(uint64_t val) -> size_t {
    return static_cast<size_t>(__popcnt64(val));
}
#elif defined(__GNUC__) || defined(__clang__)
inline auto popcount(unsigned long val) -> size_t {
    return static_cast<size_t>(__builtin_popcountl(val));
}
#else
inline auto popcount(unsigned long val) -> size_t {
    return static_cast<size_t>(Internal::countBit1(val));
}
#endif
///@}

/**
 * @brief Faster log2 when the value is the perfect power of 2.
 *
 * If the value is the perfect power of 2, using a system provided bit operation
 * is much faster than std::log2
 *
 * TODO: change to std::count_zero in C++20
 */
///@{
#if defined(_MSC_VER)
inline auto log2PerfectPower(uint64_t val) -> size_t {
    return static_cast<size_t>(63 - __lzcnt64(val));
}
#elif defined(__GNUC__) || defined(__clang__)
inline auto log2PerfectPower(unsigned long val) -> size_t {
    return static_cast<size_t>(__builtin_ctzl(val));
}
#else
inline auto log2PerfectPower(unsigned long val) -> size_t {
    return Internal::countTrailing0(val);
}
#endif
///@}

/**
 * @brief Check if there is a positive integer n such that value == 2^n.
 *
 * @param value Value to calculate for.
 * @return bool
 */
inline auto isPerfectPowerOf2(size_t value) -> bool {
    return popcount(value) == 1;
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
inline void constexpr bitswap(size_t bits, const size_t i, const size_t j) {
    size_t x = ((bits >> i) ^ (bits >> j)) & 1U;
    bits ^= ((x << i) | (x << j));
}
} // namespace Pennylane::Util
