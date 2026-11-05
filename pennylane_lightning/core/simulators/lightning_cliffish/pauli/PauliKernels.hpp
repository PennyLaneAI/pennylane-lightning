// Copyright 2026 Xanadu Quantum Technologies Inc.

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
 * @file PauliKernels.hpp
 * Bit-packed Pauli-word arithmetic shared by every Lightning-Cliffish
 * component.
 *
 * A Pauli word on n qubits is stored as two bit vectors, x and z, packed into
 * 64-bit words, plus a phase i^log_i. Qubit q is bit q % 64 of word q / 64.
 * The single-qubit Pauli at q is I, X, Z or Y for (x, z) = (0, 0), (1, 0),
 * (0, 1) or (1, 1); Y is the Hermitian Y, not XZ.
 *
 * Every function here works on raw word pointers, so it runs on the host and
 * inside Kokkos kernels alike.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Macros.hpp>

namespace Pennylane::LightningCliffish::Pauli {

/// Storage word of a bit-packed Pauli word.
using Word = std::uint64_t;

/// Number of bits in a storage word.
inline constexpr std::size_t word_bits = 64;

/// Words are padded to a multiple of this count: 8 x 64 = 512 bits, the
/// widest native SIMD width. Padding bits are always zero.
inline constexpr std::size_t words_per_pad = 8;

/**
 * @brief Number of padded storage words for a Pauli word on `num_qubits`.
 */
KOKKOS_INLINE_FUNCTION constexpr auto numWords(std::size_t num_qubits)
    -> std::size_t {
    const std::size_t words = (num_qubits + word_bits - 1) / word_bits;
    return ((words + words_per_pad - 1) / words_per_pad) * words_per_pad;
}

/**
 * @brief Read the bit of qubit `qubit`.
 */
KOKKOS_INLINE_FUNCTION auto getBit(const Word *words, std::size_t qubit)
    -> bool {
    return ((words[qubit / word_bits] >> (qubit % word_bits)) & Word{1}) != 0;
}

/**
 * @brief Set the bit of qubit `qubit` to `value`.
 */
KOKKOS_INLINE_FUNCTION void setBit(Word *words, std::size_t qubit, bool value) {
    const Word mask = Word{1} << (qubit % word_bits);
    Word &word = words[qubit / word_bits];
    word = value ? (word | mask) : (word & ~mask);
}

/**
 * @brief Phase contribution, as a signed count of factors of i, of the
 * product a * b on the qubits of one word.
 *
 * For single-qubit Paulis, a * b = i^g(a, b) (a xor b) with g = +1 for XY, YZ
 * and ZX, g = -1 for YX, ZY and XZ, and g = 0 otherwise.
 */
KOKKOS_INLINE_FUNCTION auto productPhaseWord(Word x_a, Word z_a, Word x_b,
                                             Word z_b) -> int {
    const Word pauli_x_a = x_a & ~z_a;
    const Word pauli_y_a = x_a & z_a;
    const Word pauli_z_a = ~x_a & z_a;
    const Word pauli_x_b = x_b & ~z_b;
    const Word pauli_y_b = x_b & z_b;
    const Word pauli_z_b = ~x_b & z_b;

    const Word plus_i = (pauli_x_a & pauli_y_b) | (pauli_y_a & pauli_z_b) |
                        (pauli_z_a & pauli_x_b);
    const Word minus_i = (pauli_y_a & pauli_x_b) | (pauli_z_a & pauli_y_b) |
                         (pauli_x_a & pauli_z_b);

    return Kokkos::popcount(plus_i) - Kokkos::popcount(minus_i);
}

/**
 * @brief Reduce a signed count of factors of i to an exponent in [0, 4).
 */
KOKKOS_INLINE_FUNCTION constexpr auto logIMod4(int count) -> std::uint8_t {
    return static_cast<std::uint8_t>(((count % 4) + 4) % 4);
}

/**
 * @brief Whether the Pauli words a and b anticommute.
 */
KOKKOS_INLINE_FUNCTION auto anticommutes(const Word *x_a, const Word *z_a,
                                         const Word *x_b, const Word *z_b,
                                         std::size_t num_words) -> bool {
    Word parity = 0;
    for (std::size_t w = 0; w < num_words; w++) {
        parity ^= (x_a[w] & z_b[w]) ^ (z_a[w] & x_b[w]);
    }
    return (Kokkos::popcount(parity) & 1) != 0;
}

/**
 * @brief Overwrite a with the product a * b, ignoring both phases.
 *
 * @return The phase of the product of the bare words, as a signed count of
 * factors of i. The caller adds the phases of a and b.
 */
KOKKOS_INLINE_FUNCTION auto multiplyInPlace(Word *x_a, Word *z_a,
                                            const Word *x_b, const Word *z_b,
                                            std::size_t num_words) -> int {
    int count = 0;
    for (std::size_t w = 0; w < num_words; w++) {
        count += productPhaseWord(x_a[w], z_a[w], x_b[w], z_b[w]);
        x_a[w] ^= x_b[w];
        z_a[w] ^= z_b[w];
    }
    return count;
}

} // namespace Pennylane::LightningCliffish::Pauli
