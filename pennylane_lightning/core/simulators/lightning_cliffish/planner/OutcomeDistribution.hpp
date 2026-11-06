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
 * @file OutcomeDistribution.hpp
 * The distribution of computational-basis outcomes of the hybrid state, and
 * sampling from it.
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "AffineSubspace.hpp"
#include "Error.hpp"

namespace Pennylane::LightningCliffish::Planner {

/**
 * @brief Outcomes u + sum_i o_i d_i: u uniform on an affine subspace, and the
 * bits o drawn from a table built from the dense register.
 *
 * Measured wires whose outcome a dormant qubit decides are uniform and
 * independent of the rest, so the table covers only the bits the dense
 * register decides. With no dense bits this is the stabilizer case.
 */
struct OutcomeDistribution {
    GF2::AffineSubspace uniform;
    /// The outcome vector d_i that dense bit i adds.
    std::vector<GF2::Bits> dense_columns{};
    /// Probability of each dense outcome o, with o_i as bit i of the index.
    std::vector<double> dense_probabilities{};
};

/**
 * @brief `num_shots` independent samples from `distribution`.
 */
inline auto sample(const OutcomeDistribution &distribution,
                   std::size_t num_shots, std::uint64_t seed) -> GF2::Samples {
    auto samples = GF2::sampleUniform(distribution.uniform, num_shots, seed);
    if (distribution.dense_columns.empty()) {
        return samples;
    }

    std::vector<double> cumulative(distribution.dense_probabilities.size());
    double total = 0.0;
    for (std::size_t o = 0; o < cumulative.size(); o++) {
        total += distribution.dense_probabilities[o];
        cumulative[o] = total;
    }
    const std::size_t words_per_shot = samples.words_per_shot;
    std::vector<GF2::Word> columns;
    for (const auto &column : distribution.dense_columns) {
        columns.insert(columns.end(), column.begin(), column.end());
    }

    // The dense draw takes its own stream, so the uniform part of each shot
    // is the one sampleUniform gives for the same seed.
    std::uint64_t dense_seed = seed;
    dense_seed = GF2::splitMix64(dense_seed);
    const double *cdf = cumulative.data();
    const std::size_t num_outcomes = cumulative.size();
    const GF2::Word *column_data = columns.data();
    const std::size_t num_columns = distribution.dense_columns.size();
    GF2::Word *out = samples.data.data();

    using HostSpace = Kokkos::DefaultHostExecutionSpace;
    Kokkos::parallel_for(
        "Planner::sampleDense", Kokkos::RangePolicy<HostSpace>(0, num_shots),
        [=](std::size_t shot) {
            std::uint64_t state = GF2::shotState(dense_seed, shot);
            const double u =
                static_cast<double>(GF2::splitMix64(state) >> 11U) * 0x1.0p-53 *
                total;
            const auto outcome = static_cast<std::size_t>(
                std::upper_bound(cdf, cdf + num_outcomes, u) - cdf);
            const std::size_t o = std::min(outcome, num_outcomes - 1);
            GF2::Word *row = out + shot * words_per_shot;
            for (std::size_t i = 0; i < num_columns; i++) {
                if ((o >> i) & 1U) {
                    const GF2::Word *column = column_data + i * words_per_shot;
                    for (std::size_t w = 0; w < words_per_shot; w++) {
                        row[w] ^= column[w];
                    }
                }
            }
        });
    Kokkos::fence();
    return samples;
}

/**
 * @brief The probability of each outcome index of `distribution`, with bit 0
 * of an outcome the most significant.
 */
inline auto probabilities(const OutcomeDistribution &distribution)
    -> std::vector<double> {
    if (distribution.dense_columns.empty()) {
        return GF2::probabilities(distribution.uniform);
    }
    static constexpr std::size_t max_bits = 24;
    const auto &uniform = distribution.uniform;
    const std::size_t num_bits = uniform.numBits();
    PL_ABORT_IF(num_bits > max_bits,
                "Probabilities of more than 24 wires are not supported");

    // outcomeIndex is linear, so a sum of vectors indexes as the XOR of
    // their indices.
    std::vector<std::uint64_t> uniform_indices;
    uniform_indices.reserve(std::size_t{1} << uniform.dimension());
    GF2::Bits point = uniform.offset();
    uniform_indices.push_back(GF2::outcomeIndex(point, num_bits));
    for (std::size_t step = 1; step < (std::size_t{1} << uniform.dimension());
         step++) {
        GF2::xorInto(
            point,
            uniform.basis()[static_cast<std::size_t>(std::countr_zero(step))]);
        uniform_indices.push_back(GF2::outcomeIndex(point, num_bits));
    }
    std::vector<std::uint64_t> column_indices;
    for (const auto &column : distribution.dense_columns) {
        column_indices.push_back(GF2::outcomeIndex(column, num_bits));
    }

    std::vector<double> result(std::size_t{1} << num_bits, 0.0);
    const double weight =
        1.0 / static_cast<double>(std::size_t{1} << uniform.dimension());
    for (std::size_t o = 0; o < distribution.dense_probabilities.size(); o++) {
        const double probability = distribution.dense_probabilities[o] * weight;
        if (probability == 0.0) {
            continue;
        }
        std::uint64_t dense_index = 0;
        for (std::size_t i = 0; i < column_indices.size(); i++) {
            if ((o >> i) & 1U) {
                dense_index ^= column_indices[i];
            }
        }
        for (auto index : uniform_indices) {
            result[index ^ dense_index] += probability;
        }
    }
    return result;
}

} // namespace Pennylane::LightningCliffish::Planner
