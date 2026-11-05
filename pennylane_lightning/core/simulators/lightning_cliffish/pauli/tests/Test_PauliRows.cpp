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

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "PauliRows.hpp"
#include "PauliWord.hpp"

namespace {
using namespace Pennylane::LightningCliffish::Pauli;

auto randomWord(std::size_t num_qubits, std::mt19937 &gen) -> PauliWord {
    static const std::string paulis{"IXYZ"};
    std::uniform_int_distribution<std::size_t> pick(0, 3);
    std::string word;
    for (std::size_t q = 0; q < num_qubits; q++) {
        word += paulis[pick(gen)];
    }
    return PauliWord::fromString(word);
}
} // namespace

TEST_CASE("PauliRows starts as identity rows", "[PauliRows]") {
    PauliRows<> rows(3, 5);
    CHECK(rows.numRows() == 3);
    CHECK(rows.numQubits() == 5);
    CHECK(rows.numWords() == numWords(5));
    for (std::size_t row = 0; row < 3; row++) {
        CHECK(rows.getRow(row) == PauliWord(5));
    }
}

TEST_CASE("PauliRows round trips words", "[PauliRows]") {
    std::mt19937 gen{42};
    for (std::size_t num_qubits : {0, 3, 64, 130}) {
        PauliRows<> rows(4, num_qubits);
        std::vector<PauliWord> words;
        for (std::size_t row = 0; row < 4; row++) {
            auto word = randomWord(num_qubits, gen);
            word.setLogI(static_cast<std::uint8_t>(row));
            rows.setRow(row, word);
            words.push_back(word);
        }
        for (std::size_t row = 0; row < 4; row++) {
            CHECK(rows.getRow(row) == words[row]);
        }
    }
}

TEST_CASE("PauliRows::multiplyRow matches PauliWord products", "[PauliRows]") {
    std::mt19937 gen{7};
    for (std::size_t num_qubits : {1, 65, 600}) {
        for (int trial = 0; trial < 10; trial++) {
            auto target = randomWord(num_qubits, gen);
            auto source = randomWord(num_qubits, gen);
            source.setLogI(2);

            PauliRows<> rows(2, num_qubits);
            rows.setRow(0, target);
            rows.setRow(1, source);
            rows.multiplyRow(0, 1);

            CHECK(rows.getRow(0) == target * source);
            CHECK(rows.getRow(1) == source);
        }
    }
}

TEST_CASE("PauliRows::anticommutingRows matches PauliWord::commutes",
          "[PauliRows]") {
    std::mt19937 gen{11};
    const std::size_t num_rows = 9;
    const std::size_t num_qubits = 70;
    PauliRows<> rows(num_rows, num_qubits);
    std::vector<PauliWord> words;
    for (std::size_t row = 0; row < num_rows; row++) {
        words.push_back(randomWord(num_qubits, gen));
        rows.setRow(row, words.back());
    }

    for (std::size_t row = 0; row < num_rows; row++) {
        const auto flags = rows.anticommutingRows(row);
        for (std::size_t other = 0; other < num_rows; other++) {
            CHECK(flags(other) == !words[row].commutes(words[other]));
        }
    }
}

TEST_CASE("PauliRows rejects invalid rows", "[PauliRows]") {
    using Catch::Matchers::ContainsSubstring;
    PauliRows<> rows(2, 3);
    REQUIRE_THROWS_WITH(rows.getRow(2), ContainsSubstring("out of range"));
    REQUIRE_THROWS_WITH(rows.setRow(0, PauliWord(4)),
                        ContainsSubstring("different number of qubits"));
    REQUIRE_THROWS_WITH(rows.multiplyRow(1, 1), ContainsSubstring("by itself"));
}
