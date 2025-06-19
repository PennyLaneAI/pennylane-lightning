// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <cmath>
#include <random>
#include <unordered_set>
#include <vector>

#include "TestHelpers.hpp"
#include "Util.hpp"
#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Util;
} // namespace
/// @endcond

/**
 * @brief This tests the compile-time calculation of a given scalar
 * multiplication.
 */
TEMPLATE_TEST_CASE("Util::ConstMult", "[Util]", float, double) {
    constexpr TestType r_val{0.679};
    constexpr std::complex<TestType> c0_val{TestType{1.321}, TestType{-0.175}};
    constexpr std::complex<TestType> c1_val{TestType{0.579}, TestType{1.334}};

    SECTION("Real times Complex") {
        constexpr std::complex<TestType> result =
            Util::ConstMult(r_val, c0_val);
        const std::complex<TestType> expected = r_val * c0_val;
        CHECK(isApproxEqual(result, expected));
    }
    SECTION("Complex times Complex") {
        constexpr std::complex<TestType> result =
            Util::ConstMult(c0_val, c1_val);
        const std::complex<TestType> expected = c0_val * c1_val;
        CHECK(isApproxEqual(result, expected));
    }
}

TEMPLATE_TEST_CASE("Constant values", "[Util]", float, double) {
    SECTION("One") {
        CHECK(Util::ONE<TestType>() == std::complex<TestType>{1, 0});
    }
    SECTION("Zero") {
        CHECK(Util::ZERO<TestType>() == std::complex<TestType>{0, 0});
    }
    SECTION("Imag") {
        CHECK(Util::IMAG<TestType>() == std::complex<TestType>{0, 1});
    }
    SECTION("Sqrt2") {
        CHECK(Util::SQRT2<TestType>() == std::sqrt(static_cast<TestType>(2)));
    }
    SECTION("Inverse Sqrt2") {
        CHECK(Util::INVSQRT2<TestType>() ==
              static_cast<TestType>(1 / std::sqrt(2)));
    }
}

// NOLINTNEXTLINE: Avoid complexity errors
TEMPLATE_TEST_CASE("Utility math functions", "[Util]", float, double) {
    SECTION("exp2: 2^n") {
        for (std::size_t i = 0; i < 10; i++) {
            CHECK(Util::exp2(i) == static_cast<std::size_t>(std::pow(2, i)));
        }
    }
    SECTION("maxDecimalForQubit") {
        for (std::size_t num_qubits = 0; num_qubits < 4; num_qubits++) {
            for (std::size_t index = 0; index < num_qubits; index++) {
                CHECK(Util::maxDecimalForQubit(index, num_qubits) ==
                      static_cast<std::size_t>(
                          std::pow(2, num_qubits - index - 1)));
            }
        }
    }
}

TEST_CASE("Test AlignedAllocator", "[Util][Memory]") {
    AlignedAllocator<double> allocator(8);
    REQUIRE(allocator.allocate(0) == nullptr);
/* Allocate 1 PiB */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
    REQUIRE_THROWS_AS(std::unique_ptr<double>(allocator.allocate(
                          std::size_t{1024} * std::size_t{1024 * 1024} *
                          std::size_t{1024 * 1024})),
                      std::bad_alloc);
#pragma GCC diagnostic pop
}

TEST_CASE("Test tensor transposition", "[Util]") {
    // Transposition axes and expected result.
    std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>>
        input = {{{0, 1, 2}, {0, 1, 2, 3, 4, 5, 6, 7}},
                 {{0, 2, 1}, {0, 2, 1, 3, 4, 6, 5, 7}},
                 {{1, 0, 2}, {0, 1, 4, 5, 2, 3, 6, 7}},
                 {{1, 2, 0}, {0, 4, 1, 5, 2, 6, 3, 7}},
                 {{2, 0, 1}, {0, 2, 4, 6, 1, 3, 5, 7}},
                 {{2, 1, 0}, {0, 4, 2, 6, 1, 5, 3, 7}},
                 {{0, 1, 2, 3},
                  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
                 {{0, 1, 3, 2},
                  {0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15}},
                 {{0, 2, 1, 3},
                  {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}},
                 {{0, 2, 3, 1},
                  {0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15}},
                 {{0, 3, 1, 2},
                  {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15}},
                 {{0, 3, 2, 1},
                  {0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15}},
                 {{1, 0, 2, 3},
                  {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15}},
                 {{1, 0, 3, 2},
                  {0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15}},
                 {{1, 2, 0, 3},
                  {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
                 {{1, 2, 3, 0},
                  {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}},
                 {{1, 3, 0, 2},
                  {0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15}},
                 {{1, 3, 2, 0},
                  {0, 8, 2, 10, 1, 9, 3, 11, 4, 12, 6, 14, 5, 13, 7, 15}},
                 {{2, 0, 1, 3},
                  {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15}},
                 {{2, 0, 3, 1},
                  {0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15}},
                 {{2, 1, 0, 3},
                  {0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7, 14, 15}},
                 {{2, 1, 3, 0},
                  {0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15}},
                 {{2, 3, 0, 1},
                  {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}},
                 {{2, 3, 1, 0},
                  {0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15}},
                 {{3, 0, 1, 2},
                  {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}},
                 {{3, 0, 2, 1},
                  {0, 4, 2, 6, 8, 12, 10, 14, 1, 5, 3, 7, 9, 13, 11, 15}},
                 {{3, 1, 0, 2},
                  {0, 2, 8, 10, 4, 6, 12, 14, 1, 3, 9, 11, 5, 7, 13, 15}},
                 {{3, 1, 2, 0},
                  {0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15}},
                 {{3, 2, 0, 1},
                  {0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15}},
                 {{3, 2, 1, 0},
                  {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}}};

    SECTION("Looping over different wire configurations:") {
        for (const auto &term : input) {
            // Defining a tensor to be transposed.
            std::vector<std::size_t> indices(1U << term.first.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::vector<std::size_t> result_transposed_indices =
                transpose_state_tensor(indices, term.first);
            REQUIRE(term.second == result_transposed_indices);
        }
    }
}

TEMPLATE_TEST_CASE("Util::squaredNorm", "[Util][LinearAlgebra]", float,
                   double) {
    SECTION("For real type") {
        std::vector<TestType> vec{0.0, 1.0, 3.0, 10.0};
        CHECK(squaredNorm(vec) == Approx(110.0));
    }

    SECTION("For complex type") {
        std::vector<std::complex<TestType>> vec{{0.0, 1.0}, {3.0, 10.0}};
        CHECK(squaredNorm(vec) == Approx(110.0));
    }
}

TEMPLATE_TEST_CASE("Util::is_Hermitian", "[Util][LinearAlgebra]", float,
                   double) {
    SECTION("Test for a Hermitian matrix") {
        std::vector<std::complex<TestType>> A{
            {-6.0, 0.0}, {2.0, 1.0}, {2.0, -1.0}, {0.0, 0.0}};

        REQUIRE(is_Hermitian(2, 2, A) == true);
    }

    SECTION("Test for a non-Hermitian matrix") {
        std::vector<std::complex<TestType>> A{
            {-6.0, 0.0}, {2.0, 1.0}, {2.0, 1.0}, {0.0, 0.0}};

        REQUIRE(is_Hermitian(2, 2, A) == false);
    }
}

TEMPLATE_TEST_CASE("Util::kronProd", "[Util][LinearAlgebra]", float, double) {
    SECTION("For -1, 1 values") {
        std::vector<TestType> vec0{1, -1};
        std::vector<TestType> vec1{1, -1};
        auto vec = kronProd(vec0, vec1);
        std::vector<TestType> expected = {1, -1, -1, 1};

        CHECK(vec == expected);
    }

    SECTION("For NON -1, 1 values") {
        std::vector<TestType> vec0{3, -2};
        std::vector<TestType> vec1{-4, 5};
        auto vec = kronProd(vec0, vec1);
        std::vector<TestType> expected = {-12, 15, 8, -10};

        CHECK(vec == expected);
    }
}

TEST_CASE("Util::areVecsDisjoint", "[Util][LinearAlgebra]") {
    SECTION("Test for disjoint vectors") {
        std::vector<std::size_t> vec0{0, 1, 2};
        std::vector<std::size_t> vec1{3, 4, 5};

        REQUIRE(areVecsDisjoint(vec0, vec1) == true);
    }

    SECTION("Test for joint vectors") {
        std::vector<std::size_t> vec0{0, 1, 2};
        std::vector<std::size_t> vec1{2, 4, 5};

        REQUIRE(areVecsDisjoint(vec0, vec1) == false);
    }
}

TEST_CASE("Util::createRandomWiresSubset", "[createRandomWiresSubset]") {
    using IndexT = std::size_t;
    std::mt19937 re{1337};

    SECTION(
        "createRandomWiresSubset generates correct number of unique wires") {
        IndexT sv_num_qubits = 10;
        IndexT unitary_num_qubits = 5;

        auto wires =
            createRandomWiresSubset(re, sv_num_qubits, unitary_num_qubits);

        REQUIRE(wires.size() == unitary_num_qubits);

        std::unordered_set<std::size_t> unique_wires(wires.begin(),
                                                     wires.end());
        REQUIRE(unique_wires.size() == unitary_num_qubits);
    }

    SECTION("createRandomWiresSubset throws if unitary_num_qubits > "
            "sv_num_qubits") {
        IndexT sv_num_qubits = 5;
        IndexT unitary_num_qubits = 10;

        REQUIRE_THROWS_WITH(
            createRandomWiresSubset(re, sv_num_qubits, unitary_num_qubits),
            Catch::Matchers::Contains(
                "If unitary_num_qubits > sv_num_qubits, the "
                "internal while loop will go on forever."));
    }

    SECTION("createRandomWiresSubset generates wires within correct range") {
        IndexT sv_num_qubits = 10;
        IndexT unitary_num_qubits = 5;

        auto wires =
            createRandomWiresSubset(re, sv_num_qubits, unitary_num_qubits);

        for (auto wire : wires) {
            REQUIRE(wire < sv_num_qubits);
        }
    }
}

TEST_CASE("Util::isElementInVector", "[isElementInVector]") {
    std::vector<std::size_t> wires = {0, 1, 4, 5};
    REQUIRE(isElementInVector(wires, 0UL));
    REQUIRE(isElementInVector(wires, 1UL));
    REQUIRE(isElementInVector(wires, 4UL));
    REQUIRE(isElementInVector(wires, 5UL));
    REQUIRE(!isElementInVector(wires, 2UL));
}

TEST_CASE("Util::findElementInVector", "[findElementInVector]") {
    std::vector<std::size_t> wires = {0, 1, 4};
    auto it = findElementInVector(wires, 0UL);
    REQUIRE(it != wires.end());
    REQUIRE(*it == static_cast<std::size_t>(0));
    it = findElementInVector(wires, 1UL);
    REQUIRE(it != wires.end());
    REQUIRE(*it == static_cast<std::size_t>(1));
    it = findElementInVector(wires, 4UL);
    REQUIRE(it != wires.end());
    REQUIRE(*it == static_cast<std::size_t>(4));
    it = findElementInVector(wires, 2UL);
    REQUIRE(it == wires.end());

    wires = {};
    it = findElementInVector(wires, 7UL);
    REQUIRE(it == wires.end());
}

TEST_CASE("Util::getElementIndexInVector", "[getElementIndexInVector]") {
    std::vector<std::size_t> wires = {0, 1, 4, 5};
    REQUIRE(getElementIndexInVector(wires, 0UL) == 0);
    REQUIRE(getElementIndexInVector(wires, 1UL) == 1);
    REQUIRE(getElementIndexInVector(wires, 4UL) == 2);
    REQUIRE(getElementIndexInVector(wires, 5UL) == 3);

    PL_REQUIRE_THROWS_MATCHES(
        getElementIndexInVector(wires, static_cast<std::size_t>(2)),
        LightningException, "Element not in vector");
}

TEST_CASE("Util::getRevWireIndex", "[getRevWireIndex]") {
    std::vector<std::size_t> wires = {0, 1, 4, 5};
    REQUIRE(getRevWireIndex(wires, 0UL) == 3);
    REQUIRE(getRevWireIndex(wires, 1UL) == 2);
    REQUIRE(getRevWireIndex(wires, 2UL) == 1);
    REQUIRE(getRevWireIndex(wires, 3UL) == 0);
    PL_REQUIRE_THROWS_MATCHES(getRevWireIndex(wires, 4UL), LightningException,
                              "out of bounds");
}
