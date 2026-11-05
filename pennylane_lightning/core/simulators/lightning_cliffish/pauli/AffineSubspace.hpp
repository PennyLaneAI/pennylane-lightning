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
 * @file AffineSubspace.hpp
 * Affine subspaces of GF(2)^n, and uniform sampling from them.
 *
 * The computational-basis outcomes of a stabilizer state are uniform on an
 * affine subspace, so sampling a stabilizer state is sampling one of these.
 */

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Error.hpp"
#include "PauliKernels.hpp"
#include "PauliRows.hpp"

namespace Pennylane::LightningCliffish::GF2 {

using Pauli::Word;
using Pauli::word_bits;

/// A bit vector, bit k in bit k % 64 of word k / 64.
using Bits = std::vector<Word>;

/// Number of words for `num_bits` bits, unpadded.
inline auto numBitWords(std::size_t num_bits) -> std::size_t {
    return (num_bits + word_bits - 1) / word_bits;
}

inline auto getBit(const Bits &bits, std::size_t k) -> bool {
    return Pauli::getBit(bits.data(), k);
}

inline void setBit(Bits &bits, std::size_t k, bool value) {
    Pauli::setBit(bits.data(), k, value);
}

inline void xorInto(Bits &target, const Bits &source) {
    for (std::size_t w = 0; w < target.size(); w++) {
        target[w] ^= source[w];
    }
}

/// The lowest set bit, or nothing for the zero vector.
inline auto lowestSetBit(const Bits &bits) -> std::optional<std::size_t> {
    for (std::size_t w = 0; w < bits.size(); w++) {
        if (bits[w] != 0) {
            return w * word_bits +
                   static_cast<std::size_t>(std::countr_zero(bits[w]));
        }
    }
    return std::nullopt;
}

/**
 * @brief Index of an outcome among 2^num_bits, with bit 0 the most
 * significant, as in PennyLane.
 */
inline auto outcomeIndex(const Bits &bits, std::size_t num_bits)
    -> std::uint64_t {
    PL_ABORT_IF(num_bits > 64, "An outcome index holds at most 64 bits");
    std::uint64_t index = 0;
    for (std::size_t k = 0; k < num_bits; k++) {
        index = (index << 1U) | (getBit(bits, k) ? 1U : 0U);
    }
    return index;
}

/**
 * @brief The set offset + span(basis) in GF(2)^num_bits.
 *
 * The basis is kept in reduced echelon form: basis vector i has its lowest
 * set bit at pivots[i], and no other basis vector, nor the offset, has that
 * bit set. This makes the offset and the basis canonical.
 */
class AffineSubspace {
  private:
    std::size_t num_bits_;
    Bits offset_;
    std::vector<Bits> basis_{};
    std::vector<std::size_t> pivots_{};

    void reduce(Bits &vector) const {
        for (std::size_t i = 0; i < basis_.size(); i++) {
            if (getBit(vector, pivots_[i])) {
                xorInto(vector, basis_[i]);
            }
        }
    }

    void addVector(Bits vector) {
        reduce(vector);
        const auto pivot = lowestSetBit(vector);
        if (!pivot.has_value()) {
            return;
        }
        for (auto &other : basis_) {
            if (getBit(other, *pivot)) {
                xorInto(other, vector);
            }
        }
        if (getBit(offset_, *pivot)) {
            xorInto(offset_, vector);
        }
        basis_.push_back(std::move(vector));
        pivots_.push_back(*pivot);
    }

  public:
    /**
     * @brief offset + span(spanning) in GF(2)^num_bits.
     */
    AffineSubspace(std::size_t num_bits, Bits offset,
                   const std::vector<Bits> &spanning = {})
        : num_bits_{num_bits}, offset_{std::move(offset)} {
        PL_ABORT_IF_NOT(offset_.size() == numBitWords(num_bits),
                        "Offset has the wrong number of words");
        reduce(offset_);
        for (const auto &vector : spanning) {
            PL_ABORT_IF_NOT(vector.size() == numBitWords(num_bits),
                            "Spanning vector has the wrong number of words");
            addVector(vector);
        }
    }

    /**
     * @brief The solutions x of A x = rhs, or nothing when there are none.
     *
     * @param num_bits Length of x.
     * @param rows The rows of A.
     * @param rhs Bit i is the right-hand side of row i.
     */
    static auto solve(std::size_t num_bits, std::vector<Bits> rows,
                      std::vector<bool> rhs) -> std::optional<AffineSubspace> {
        PL_ABORT_IF_NOT(rows.size() == rhs.size(),
                        "Each row needs a right-hand side");

        // Reduced row echelon form, one pivot column at a time.
        std::vector<std::size_t> pivot_columns;
        std::size_t rank = 0;
        for (std::size_t column = 0; column < num_bits && rank < rows.size();
             column++) {
            std::size_t pivot = rank;
            while (pivot < rows.size() && !getBit(rows[pivot], column)) {
                pivot++;
            }
            if (pivot == rows.size()) {
                continue;
            }
            std::swap(rows[rank], rows[pivot]);
            std::vector<bool>::swap(rhs[rank], rhs[pivot]);
            for (std::size_t r = 0; r < rows.size(); r++) {
                if (r != rank && getBit(rows[r], column)) {
                    xorInto(rows[r], rows[rank]);
                    rhs[r] = rhs[r] != rhs[rank];
                }
            }
            pivot_columns.push_back(column);
            rank++;
        }
        for (std::size_t r = rank; r < rows.size(); r++) {
            if (rhs[r]) {
                return std::nullopt;
            }
        }

        // Free variables set to zero give one solution; setting one free
        // variable to one gives each null-space basis vector.
        std::vector<bool> is_pivot(num_bits, false);
        Bits offset(numBitWords(num_bits), 0);
        for (std::size_t i = 0; i < rank; i++) {
            is_pivot[pivot_columns[i]] = true;
            setBit(offset, pivot_columns[i], rhs[i]);
        }
        std::vector<Bits> null_space;
        for (std::size_t free = 0; free < num_bits; free++) {
            if (is_pivot[free]) {
                continue;
            }
            Bits vector(numBitWords(num_bits), 0);
            setBit(vector, free, true);
            for (std::size_t i = 0; i < rank; i++) {
                if (getBit(rows[i], free)) {
                    setBit(vector, pivot_columns[i], true);
                }
            }
            null_space.push_back(std::move(vector));
        }
        return AffineSubspace(num_bits, std::move(offset), null_space);
    }

    [[nodiscard]] auto numBits() const -> std::size_t { return num_bits_; }
    [[nodiscard]] auto dimension() const -> std::size_t {
        return basis_.size();
    }
    [[nodiscard]] auto offset() const -> const Bits & { return offset_; }
    [[nodiscard]] auto basis() const -> const std::vector<Bits> & {
        return basis_;
    }

    [[nodiscard]] auto contains(const Bits &point) const -> bool {
        PL_ABORT_IF_NOT(point.size() == offset_.size(),
                        "Point has the wrong number of words");
        Bits difference = point;
        xorInto(difference, offset_);
        reduce(difference);
        return !lowestSetBit(difference).has_value();
    }

    /**
     * @brief The image under x -> (x[positions[0]], x[positions[1]], ...).
     */
    [[nodiscard]] auto project(const std::vector<std::size_t> &positions) const
        -> AffineSubspace {
        auto projected = [&positions](const Bits &vector) {
            Bits result(numBitWords(positions.size()), 0);
            for (std::size_t k = 0; k < positions.size(); k++) {
                setBit(result, k, getBit(vector, positions[k]));
            }
            return result;
        };
        for (auto position : positions) {
            PL_ABORT_IF_NOT(position < num_bits_, "Position out of range");
        }
        std::vector<Bits> basis;
        basis.reserve(basis_.size());
        for (const auto &vector : basis_) {
            basis.push_back(projected(vector));
        }
        return AffineSubspace(positions.size(), projected(offset_), basis);
    }
};

/**
 * @brief Bit-packed outcomes, one row of words per shot.
 */
struct Samples {
    std::size_t num_shots;
    std::size_t num_bits;
    std::size_t words_per_shot;
    std::vector<Word> data;

    [[nodiscard]] auto bit(std::size_t shot, std::size_t k) const -> bool {
        return Pauli::getBit(data.data() + shot * words_per_shot, k);
    }

    [[nodiscard]] auto shot(std::size_t shot) const -> Bits {
        const auto first =
            data.begin() + static_cast<std::ptrdiff_t>(shot * words_per_shot);
        return {first, first + static_cast<std::ptrdiff_t>(words_per_shot)};
    }
};

/**
 * @brief SplitMix64: one step of a counter-based generator.
 */
KOKKOS_INLINE_FUNCTION auto splitMix64(std::uint64_t &state) -> std::uint64_t {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

/**
 * @brief The generator state of shot `shot`: every shot draws from its own
 * stream, so samples do not depend on how shots are split over threads.
 */
KOKKOS_INLINE_FUNCTION auto shotState(std::uint64_t seed, std::uint64_t shot)
    -> std::uint64_t {
    std::uint64_t mix = shot;
    return seed ^ splitMix64(mix);
}

/**
 * @brief `num_shots` independent uniform samples from `space`.
 */
inline auto sampleUniform(const AffineSubspace &space, std::size_t num_shots,
                          std::uint64_t seed) -> Samples {
    Pauli::ensureKokkosInitialized();
    const std::size_t words_per_shot = numBitWords(space.numBits());
    const std::size_t dimension = space.dimension();

    std::vector<Word> basis;
    basis.reserve(dimension * words_per_shot);
    for (const auto &vector : space.basis()) {
        basis.insert(basis.end(), vector.begin(), vector.end());
    }

    Samples samples{num_shots, space.numBits(), words_per_shot,
                    std::vector<Word>(num_shots * words_per_shot, 0)};
    const Word *offset = space.offset().data();
    const Word *basis_data = basis.data();
    Word *out = samples.data.data();

    using HostSpace = Kokkos::DefaultHostExecutionSpace;
    Kokkos::parallel_for(
        "GF2::sampleUniform", Kokkos::RangePolicy<HostSpace>(0, num_shots),
        [=](std::size_t shot) {
            Word *row = out + shot * words_per_shot;
            for (std::size_t w = 0; w < words_per_shot; w++) {
                row[w] = offset[w];
            }
            std::uint64_t state = shotState(seed, shot);
            std::uint64_t random = 0;
            for (std::size_t j = 0; j < dimension; j++) {
                if (j % word_bits == 0) {
                    random = splitMix64(state);
                }
                if ((random >> (j % word_bits)) & 1U) {
                    const Word *vector = basis_data + j * words_per_shot;
                    for (std::size_t w = 0; w < words_per_shot; w++) {
                        row[w] ^= vector[w];
                    }
                }
            }
        });
    Kokkos::fence();
    return samples;
}

/**
 * @brief The probability of each outcome index, for a uniform distribution
 * on `space`.
 */
inline auto probabilities(const AffineSubspace &space) -> std::vector<double> {
    // Each output entry is a double, so 2^24 entries already take 128 MiB.
    static constexpr std::size_t max_bits = 24;
    PL_ABORT_IF(space.numBits() > max_bits,
                "Probabilities of more than 24 wires are not supported");
    std::vector<double> result(std::size_t{1} << space.numBits(), 0.0);
    const std::size_t dimension = space.dimension();
    const double probability =
        1.0 / static_cast<double>(std::size_t{1} << dimension);

    // Visit every point once, in Gray-code order.
    Bits point = space.offset();
    result[outcomeIndex(point, space.numBits())] = probability;
    for (std::size_t step = 1; step < (std::size_t{1} << dimension); step++) {
        xorInto(
            point,
            space.basis()[static_cast<std::size_t>(std::countr_zero(step))]);
        result[outcomeIndex(point, space.numBits())] = probability;
    }
    return result;
}

/**
 * @brief Number of shots of each outcome index.
 */
inline auto counts(const Samples &samples)
    -> std::map<std::uint64_t, std::size_t> {
    PL_ABORT_IF(samples.num_bits > 64, "Counts hold at most 64 wires");
    std::map<std::uint64_t, std::size_t> result;
    for (std::size_t shot = 0; shot < samples.num_shots; shot++) {
        result[outcomeIndex(samples.shot(shot), samples.num_bits)]++;
    }
    return result;
}

} // namespace Pennylane::LightningCliffish::GF2
