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

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "AffineSubspace.hpp"

namespace {
using namespace Pennylane::LightningCliffish::GF2;
using Catch::Matchers::ContainsSubstring;

auto randomBits(std::size_t num_bits, std::mt19937 &gen) -> Bits {
    Bits bits(numBitWords(num_bits), 0);
    for (std::size_t k = 0; k < num_bits; k++) {
        setBit(bits, k, std::bernoulli_distribution(0.5)(gen));
    }
    return bits;
}

auto dot(const Bits &a, const Bits &b) -> bool {
    std::size_t count = 0;
    for (std::size_t w = 0; w < a.size(); w++) {
        count += static_cast<std::size_t>(std::popcount(a[w] & b[w]));
    }
    return count % 2 == 1;
}

/// The bits of `index` as an outcome, with bit 0 the most significant.
auto outcomeBits(std::uint64_t index, std::size_t num_bits) -> Bits {
    Bits bits(numBitWords(num_bits), 0);
    for (std::size_t k = 0; k < num_bits; k++) {
        setBit(bits, k, (index >> (num_bits - 1 - k)) & 1U);
    }
    return bits;
}

/// A random affine subspace of dimension at most `max_dimension`.
auto randomSpace(std::size_t num_bits, std::size_t max_dimension,
                 std::mt19937 &gen) -> AffineSubspace {
    std::vector<Bits> spanning;
    for (std::size_t i = 0; i < max_dimension; i++) {
        spanning.push_back(randomBits(num_bits, gen));
    }
    return {num_bits, randomBits(num_bits, gen), spanning};
}
} // namespace

TEST_CASE("AffineSubspace keeps an independent reduced basis",
          "[AffineSubspace]") {
    Bits a(1, 0b0011);
    Bits b(1, 0b0110);
    Bits c(1, 0b0101); // a + b
    const AffineSubspace space(4, Bits(1, 0b1000), {a, b, c});
    CHECK(space.dimension() == 2);

    CHECK(space.contains(Bits(1, 0b1000)));
    CHECK(space.contains(Bits(1, 0b1011)));
    CHECK(space.contains(Bits(1, 0b1110)));
    CHECK(space.contains(Bits(1, 0b1101)));
    CHECK_FALSE(space.contains(Bits(1, 0b0000)));
    CHECK_FALSE(space.contains(Bits(1, 0b1001)));

    // The offset and the set of basis vectors are canonical: the same space
    // from another offset and spanning order is equal.
    const AffineSubspace same(4, Bits(1, 0b1101), {b, a});
    CHECK(same.offset() == space.offset());
    auto sorted = [](std::vector<Bits> basis) {
        std::sort(basis.begin(), basis.end());
        return basis;
    };
    CHECK(sorted(same.basis()) == sorted(space.basis()));
}

TEST_CASE("AffineSubspace::solve finds every solution", "[AffineSubspace]") {
    std::mt19937 gen{12};
    for (std::size_t num_bits : {1, 5, 64, 130}) {
        for (std::size_t num_rows : {std::size_t{0}, num_bits / 2, num_bits}) {
            std::vector<Bits> rows;
            for (std::size_t r = 0; r < num_rows; r++) {
                rows.push_back(randomBits(num_bits, gen));
            }
            const Bits solution = randomBits(num_bits, gen);
            std::vector<bool> rhs;
            for (const auto &row : rows) {
                rhs.push_back(dot(row, solution));
            }

            const auto space = AffineSubspace::solve(num_bits, rows, rhs);
            REQUIRE(space.has_value());
            CHECK(space->contains(solution));
            CHECK(space->dimension() >= num_bits - num_rows);
            for (const auto &vector : space->basis()) {
                for (const auto &row : rows) {
                    CHECK_FALSE(dot(row, vector));
                }
            }
            for (std::size_t r = 0; r < rows.size(); r++) {
                CHECK(dot(rows[r], space->offset()) == rhs[r]);
            }
        }
    }
}

TEST_CASE("AffineSubspace::solve detects inconsistent systems",
          "[AffineSubspace]") {
    // x0 + x1 = 0 and x0 + x1 = 1 have no solution.
    const auto space =
        AffineSubspace::solve(2, {Bits(1, 0b11), Bits(1, 0b11)}, {false, true});
    CHECK_FALSE(space.has_value());
}

TEST_CASE("AffineSubspace::project keeps the image of every point",
          "[AffineSubspace]") {
    std::mt19937 gen{13};
    const std::size_t num_bits = 100;
    const auto space = randomSpace(num_bits, 30, gen);
    const std::vector<std::size_t> positions{99, 3, 64, 3, 0, 63};
    const auto projected = space.project(positions);
    CHECK(projected.numBits() == positions.size());
    CHECK(projected.dimension() <= positions.size());

    const auto samples = sampleUniform(space, 200, 1);
    for (std::size_t shot = 0; shot < samples.num_shots; shot++) {
        const Bits point = samples.shot(shot);
        Bits image(numBitWords(positions.size()), 0);
        for (std::size_t k = 0; k < positions.size(); k++) {
            setBit(image, k, getBit(point, positions[k]));
        }
        CHECK(projected.contains(image));
    }
    REQUIRE_THROWS_WITH(space.project({num_bits}),
                        ContainsSubstring("out of range"));
}

TEST_CASE("sampleUniform draws reproducible points of the space",
          "[AffineSubspace]") {
    std::mt19937 gen{14};
    const auto space = randomSpace(130, 40, gen);

    const auto samples = sampleUniform(space, 500, 99);
    CHECK(samples.num_shots == 500);
    CHECK(samples.num_bits == 130);
    for (std::size_t shot = 0; shot < samples.num_shots; shot++) {
        CHECK(space.contains(samples.shot(shot)));
    }
    CHECK(sampleUniform(space, 500, 99).data == samples.data);
    CHECK(sampleUniform(space, 500, 100).data != samples.data);

    const auto none = sampleUniform(space, 0, 99);
    CHECK(none.num_shots == 0);
    CHECK(none.data.empty());
}

TEST_CASE("sampleUniform is uniform on the space", "[AffineSubspace]") {
    // A 3-dimensional space in 5 bits: 8 equally likely points.
    const AffineSubspace space(
        5, Bits(1, 0b10000),
        {Bits(1, 0b00011), Bits(1, 0b00110), Bits(1, 0b01100)});
    const std::size_t shots = 80000;
    const auto histogram = counts(sampleUniform(space, shots, 7));
    CHECK(histogram.size() == 8);
    const double expected = shots / 8.0;
    const double tolerance = 5 * std::sqrt(expected);
    for (const auto &[index, count] : histogram) {
        CHECK(space.contains(outcomeBits(index, 5)));
        CHECK(std::abs(static_cast<double>(count) - expected) < tolerance);
    }
}

TEST_CASE("probabilities are uniform on the space and zero elsewhere",
          "[AffineSubspace]") {
    std::mt19937 gen{15};
    for (std::size_t num_bits : {0, 1, 6, 10}) {
        const auto space = randomSpace(num_bits, 4, gen);
        const auto result = probabilities(space);
        REQUIRE(result.size() == (std::size_t{1} << num_bits));
        const double probability =
            1.0 / static_cast<double>(std::size_t{1} << space.dimension());
        double total = 0.0;
        for (std::uint64_t index = 0; index < result.size(); index++) {
            const bool inside = space.contains(outcomeBits(index, num_bits));
            CHECK(result[index] == (inside ? probability : 0.0));
            total += result[index];
        }
        CHECK(std::abs(total - 1.0) < 1e-12);
    }
    REQUIRE_THROWS_WITH(probabilities(AffineSubspace(25, Bits(1, 0))),
                        ContainsSubstring("more than 24 wires"));
}

TEST_CASE("Outcome indices put bit 0 first", "[AffineSubspace]") {
    CHECK(outcomeIndex(Bits(1, 0b001), 3) == 0b100);
    CHECK(outcomeIndex(Bits(1, 0b110), 3) == 0b011);

    const AffineSubspace point(3, Bits(1, 0b001));
    const auto histogram = counts(sampleUniform(point, 10, 3));
    CHECK(histogram == std::map<std::uint64_t, std::size_t>{{0b100, 10}});
    REQUIRE_THROWS_WITH(outcomeIndex(Bits(2, 0), 65),
                        ContainsSubstring("at most 64 bits"));
}
