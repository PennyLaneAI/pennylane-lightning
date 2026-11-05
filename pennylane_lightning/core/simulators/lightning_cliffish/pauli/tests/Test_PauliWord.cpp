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

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "PauliKernels.hpp"
#include "PauliWord.hpp"

namespace {
using namespace Pennylane::LightningCliffish::Pauli;

/**
 * @brief Reference single-qubit product: a * b = i^phase c.
 */
auto referenceProduct(char a, char b) -> std::pair<int, char> {
    if (a == 'I') {
        return {0, b};
    }
    if (b == 'I') {
        return {0, a};
    }
    if (a == b) {
        return {0, 'I'};
    }
    // XY = iZ, YZ = iX, ZX = iY; the reversed orders give -i.
    const std::string cycle = "XYZ";
    const auto ia = cycle.find(a);
    const auto ib = cycle.find(b);
    const char c = cycle[3 - ia - ib];
    return {(ib == (ia + 1) % 3) ? 1 : 3, c};
}

auto randomWord(std::size_t num_qubits, std::mt19937 &gen) -> std::string {
    static constexpr std::array<char, 4> paulis{'I', 'X', 'Y', 'Z'};
    std::uniform_int_distribution<std::size_t> pick(0, 3);
    std::string word;
    for (std::size_t q = 0; q < num_qubits; q++) {
        word += paulis[pick(gen)];
    }
    return word;
}

/**
 * @brief Reference multi-qubit product of bare words, as a phased string.
 */
auto referenceProduct(const std::string &a, const std::string &b)
    -> std::string {
    int log_i = 0;
    std::string word;
    for (std::size_t q = 0; q < a.size(); q++) {
        const auto [phase, pauli] = referenceProduct(a[q], b[q]);
        log_i += phase;
        word += pauli;
    }
    static constexpr std::array<const char *, 4> phases{"+", "+i", "-", "-i"};
    return phases[log_i % 4] + word;
}

auto referenceAnticommutes(const std::string &a, const std::string &b) -> bool {
    std::size_t count = 0;
    for (std::size_t q = 0; q < a.size(); q++) {
        count += (a[q] != 'I' && b[q] != 'I' && a[q] != b[q]) ? 1 : 0;
    }
    return count % 2 == 1;
}
} // namespace

TEST_CASE("numWords pads to a multiple of words_per_pad", "[Pauli]") {
    CHECK(numWords(0) == 0);
    CHECK(numWords(1) == words_per_pad);
    CHECK(numWords(word_bits * words_per_pad) == words_per_pad);
    CHECK(numWords(word_bits * words_per_pad + 1) == 2 * words_per_pad);
}

TEST_CASE("PauliWord string round trip", "[Pauli]") {
    CHECK(PauliWord::fromString("XYZI").toString() == "+XYZI");
    CHECK(PauliWord::fromString("+XYZI").toString() == "+XYZI");
    CHECK(PauliWord::fromString("-ZZ").toString() == "-ZZ");
    CHECK(PauliWord::fromString("iX").toString() == "+iX");
    CHECK(PauliWord::fromString("+iX").toString() == "+iX");
    CHECK(PauliWord::fromString("-iY").toString() == "-iY");
    CHECK(PauliWord::fromString("").toString() == "+");

    const auto word = PauliWord::fromString("IXYZ");
    CHECK(word.numQubits() == 4);
    CHECK(word.weight() == 3);
    CHECK(word.get(0) == 'I');
    CHECK(word.get(2) == 'Y');
    CHECK(word.isHermitian());
    CHECK_FALSE(PauliWord::fromString("iX").isHermitian());
}

TEST_CASE("PauliWord rejects invalid input", "[Pauli]") {
    using Catch::Matchers::ContainsSubstring;
    REQUIRE_THROWS_WITH(PauliWord::fromString("XA"),
                        ContainsSubstring("only contain I, X, Y and Z"));
    REQUIRE_THROWS_WITH(PauliWord(2).get(2), ContainsSubstring("out of range"));
    REQUIRE_THROWS_WITH(PauliWord(2) * PauliWord(3),
                        ContainsSubstring("different numbers of qubits"));
}

TEST_CASE("PauliWord single-qubit products", "[Pauli]") {
    for (char a : std::string{"IXYZ"}) {
        for (char b : std::string{"IXYZ"}) {
            const auto product = PauliWord::fromString(std::string{a}) *
                                 PauliWord::fromString(std::string{b});
            CHECK(product.toString() ==
                  referenceProduct(std::string{a}, std::string{b}));
        }
    }
}

TEST_CASE("PauliWord products and commutation match the reference", "[Pauli]") {
    std::mt19937 gen{1729};
    // Sizes cover a partial word, a full word and more than one pad.
    for (std::size_t num_qubits : {1, 7, 64, 65, 600}) {
        for (int trial = 0; trial < 20; trial++) {
            const auto a = randomWord(num_qubits, gen);
            const auto b = randomWord(num_qubits, gen);
            const auto word_a = PauliWord::fromString(a);
            const auto word_b = PauliWord::fromString(b);

            CHECK((word_a * word_b).toString() == referenceProduct(a, b));
            CHECK(word_a.commutes(word_b) == !referenceAnticommutes(a, b));
        }
    }
}

TEST_CASE("PauliWord product phases include operand phases", "[Pauli]") {
    // (-X)(iY) = -i XY = -i (i Z) = +Z
    CHECK((PauliWord::fromString("-X") * PauliWord::fromString("iY"))
              .toString() == "+Z");
    // Products of commuting Hermitian words are Hermitian.
    const auto product =
        PauliWord::fromString("XX") * PauliWord::fromString("-ZZ");
    CHECK(product.toString() == "+YY");
}

TEST_CASE("PauliWord padding bits stay zero", "[Pauli]") {
    auto word = PauliWord::fromString(std::string(70, 'Y'));
    word *= PauliWord::fromString(std::string(70, 'X'));
    for (std::size_t w = 0; w < word.numWords(); w++) {
        const std::size_t first_qubit = w * word_bits;
        if (first_qubit >= 70) {
            CHECK(word.x()[w] == 0);
            CHECK(word.z()[w] == 0);
        }
    }
    // YX = -iZ on each qubit, and (-i)^70 = -1.
    CHECK(word.toString() == "-" + std::string(70, 'Z'));
}
