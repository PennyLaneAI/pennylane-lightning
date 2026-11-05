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
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DenseReference.hpp"
#include "NativeTableau.hpp"

namespace {
using namespace Pennylane::LightningCliffish;
using namespace Pennylane::LightningCliffish::Testing;
using Tableau::EngineOp;
using Tableau::NativeTableau;

/**
 * @brief Expectation values of the tableau and the dense state agree on
 * `num_checks` random Pauli words.
 */
void requireSameState(const NativeTableau<> &tableau, const State &state,
                      std::mt19937 &gen, std::size_t num_checks = 100) {
    const std::size_t num_qubits = tableau.numQubits();
    for (std::size_t check = 0; check < num_checks; check++) {
        const auto pauli = randomPauli(num_qubits, gen);
        const double expected = expectation(state, pauli);
        INFO("Pauli word " << pauli.toString());
        REQUIRE(std::abs(tableau.expectation(pauli) - expected) < 1e-9);
    }
}

/**
 * @brief Apply `num_rotations` random rotations to the tableau and the dense
 * state.
 */
void applyRandomRotations(NativeTableau<> &tableau, State &state,
                          std::size_t num_rotations, std::mt19937 &gen) {
    std::uniform_int_distribution<unsigned> turns(1, 3);
    for (std::size_t i = 0; i < num_rotations; i++) {
        const auto pauli = randomPauli(tableau.numQubits(), gen);
        const auto quarter_turns = static_cast<std::uint8_t>(turns(gen));
        tableau.applyBlock({EngineOp::rotation(pauli, quarter_turns)}, gen);
        state = rotate(state, pauli, quarter_turns);
    }
}
} // namespace

TEST_CASE("NativeTableau starts in |0...0>", "[NativeTableau]") {
    NativeTableau<> tableau(3);
    CHECK(tableau.numQubits() == 3);
    CHECK(tableau.name() == "native");
    for (std::size_t q = 0; q < 3; q++) {
        CHECK(tableau.expectation(singleQubitPauli(3, q, 'Z')) == 1);
        CHECK(tableau.expectation(singleQubitPauli(3, q, 'X')) == 0);
        CHECK(tableau.expectation(singleQubitPauli(3, q, 'Y')) == 0);
    }
    CHECK(tableau.expectation(PauliWord::fromString("-ZZI")) == -1);
}

TEST_CASE("NativeTableau rotations match a dense state", "[NativeTableau]") {
    std::mt19937 gen{2026};
    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (int trial = 0; trial < 5; trial++) {
            NativeTableau<> tableau(num_qubits);
            State state = zeroState(num_qubits);
            applyRandomRotations(tableau, state, 30, gen);
            requireSameState(tableau, state, gen);
        }
    }
}

TEST_CASE("NativeTableau measurements match a dense state", "[NativeTableau]") {
    using Catch::Matchers::ContainsSubstring;
    std::mt19937 gen{99};
    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (int trial = 0; trial < 10; trial++) {
            NativeTableau<> tableau(num_qubits);
            State state = zeroState(num_qubits);
            applyRandomRotations(tableau, state, 20, gen);

            for (int measurement = 0; measurement < 5; measurement++) {
                const auto pauli = randomPauli(num_qubits, gen);
                const double value = expectation(state, pauli);
                if (std::abs(value) > 0.5) {
                    // Deterministic: the outcome is fixed and the other one
                    // cannot be projected onto.
                    const bool outcome = value < 0;
                    CHECK(tableau.applyBlock({EngineOp::measurement(pauli)},
                                             gen) ==
                          std::vector<bool>{outcome});
                    REQUIRE_THROWS_WITH(
                        tableau.applyBlock(
                            {EngineOp::measurement(pauli, !outcome)}, gen),
                        ContainsSubstring("zero probability"));
                } else {
                    const bool outcome = std::bernoulli_distribution(0.5)(gen);
                    CHECK(tableau.applyBlock(
                              {EngineOp::measurement(pauli, outcome)}, gen) ==
                          std::vector<bool>{outcome});
                    state = project(state, pauli, outcome);
                }
                requireSameState(tableau, state, gen, 50);
            }
        }
    }
}

TEST_CASE("NativeTableau random outcomes are unbiased", "[NativeTableau]") {
    std::mt19937 gen{5};
    int ones = 0;
    const int trials = 2000;
    for (int trial = 0; trial < trials; trial++) {
        NativeTableau<> tableau(1);
        const auto outcomes = tableau.applyBlock(
            {EngineOp::measurement(PauliWord::fromString("X"))}, gen);
        ones += outcomes[0] ? 1 : 0;
    }
    CHECK(std::abs(ones - trials / 2) < 150);
}

TEST_CASE("NativeTableau pulls Paulis back through the state's Clifford",
          "[NativeTableau]") {
    std::mt19937 gen{17};

    SECTION("exp(-i pi/4 Z) maps X back to -Y") {
        NativeTableau<> tableau(1);
        tableau.applyBlock({EngineOp::rotation(PauliWord::fromString("Z"), 1)},
                           gen);
        CHECK(tableau.pullBack(PauliWord::fromString("X")).toString() == "-Y");
        CHECK(tableau.pullBack(PauliWord::fromString("Y")).toString() == "+X");
        CHECK(tableau.pullBack(PauliWord::fromString("Z")).toString() == "+Z");
    }

    SECTION("pull-back agrees with expectation values and products") {
        for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
            NativeTableau<> tableau(num_qubits);
            State state = zeroState(num_qubits);
            applyRandomRotations(tableau, state, 30, gen);

            for (int check = 0; check < 30; check++) {
                const auto pauli = randomPauli(num_qubits, gen);
                const auto pulled = tableau.pullBack(pauli);
                CHECK(pulled.isHermitian());
                CHECK(pulled.weight() > 0);
                // <0| C^† P C |0> = <psi| P |psi>
                CHECK(std::abs(expectation(zeroState(num_qubits), pulled) -
                               expectation(state, pauli)) < 1e-9);

                const auto other = randomPauli(num_qubits, gen);
                if (pauli.commutes(other)) {
                    CHECK(tableau.pullBack(pauli * other) ==
                          pulled * tableau.pullBack(other));
                }
            }
        }
    }
}

TEST_CASE("NativeTableau::addQubits keeps the state and adds |0>",
          "[NativeTableau]") {
    std::mt19937 gen{3};
    NativeTableau<> tableau(2);
    State state = zeroState(2);
    applyRandomRotations(tableau, state, 20, gen);

    tableau.addQubits(2);
    CHECK(tableau.numQubits() == 4);
    // The dense state on 4 qubits: the old state on qubits 0 and 1, |00> on
    // qubits 2 and 3.
    State extended(16, 0.0);
    for (std::size_t i = 0; i < 4; i++) {
        extended[i] = state[i];
    }
    requireSameState(tableau, extended, gen);

    applyRandomRotations(tableau, extended, 20, gen);
    requireSameState(tableau, extended, gen);
}

TEST_CASE("NativeTableau handles more than one storage pad",
          "[NativeTableau]") {
    // A GHZ state on 600 qubits: H on qubit 0, then a CNOT chain written as
    // quarter-turn rotations.
    const std::size_t num_qubits = 600;
    std::mt19937 gen{1};
    NativeTableau<> tableau(num_qubits);

    auto word = [num_qubits](std::size_t q0, char p0, std::size_t q1, char p1) {
        PauliWord pauli(num_qubits);
        pauli.set(q0, p0);
        if (q1 != q0) {
            pauli.set(q1, p1);
        }
        return pauli;
    };
    // H = exp(-i pi/4 Y) Z, up to a phase.
    Tableau::EngineBlock block{EngineOp::rotation(word(0, 'Z', 0, 'Z'), 2),
                               EngineOp::rotation(word(0, 'Y', 0, 'Y'), 1)};
    for (std::size_t q = 0; q + 1 < num_qubits; q++) {
        // CNOT = exp(-i pi/4 Z_c) exp(-i pi/4 X_t) exp(i pi/4 Z_c X_t)
        block.push_back(EngineOp::rotation(word(q, 'Z', q, 'Z'), 1));
        block.push_back(EngineOp::rotation(word(q + 1, 'X', q + 1, 'X'), 1));
        block.push_back(EngineOp::rotation(word(q, 'Z', q + 1, 'X'), 3));
    }
    tableau.applyBlock(block, gen);

    CHECK(tableau.expectation(word(0, 'Z', num_qubits - 1, 'Z')) == 1);
    CHECK(tableau.expectation(word(0, 'Z', 0, 'Z')) == 0);
    CHECK(tableau.expectation(
              PauliWord::fromString(std::string(num_qubits, 'X'))) == 1);

    const auto outcomes = tableau.applyBlock(
        {EngineOp::measurement(word(0, 'Z', 0, 'Z')),
         EngineOp::measurement(word(num_qubits - 1, 'Z', num_qubits - 1, 'Z'))},
        gen);
    CHECK(outcomes[0] == outcomes[1]);
}

TEST_CASE("NativeTableau rejects invalid ops", "[NativeTableau]") {
    using Catch::Matchers::ContainsSubstring;
    std::mt19937 gen{0};
    NativeTableau<> tableau(2);
    REQUIRE_THROWS_WITH(
        tableau.applyBlock({EngineOp::rotation(PauliWord::fromString("X"), 1)},
                           gen),
        ContainsSubstring("different number of qubits"));
    REQUIRE_THROWS_WITH(
        tableau.applyBlock(
            {EngineOp::rotation(PauliWord::fromString("iXX"), 1)}, gen),
        ContainsSubstring("Hermitian"));
    REQUIRE_THROWS_WITH(
        tableau.applyBlock({EngineOp::rotation(PauliWord::fromString("XX"), 0)},
                           gen),
        ContainsSubstring("1, 2 or 3 quarter turns"));
}

TEST_CASE("NativeTableau::computationalSupport matches dense probabilities",
          "[NativeTableau]") {
    std::mt19937 gen{21};
    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (int trial = 0; trial < 10; trial++) {
            NativeTableau<> tableau(num_qubits);
            State state = zeroState(num_qubits);
            applyRandomRotations(tableau, state, 25, gen);

            // All qubits in order, then a random ordered subset.
            std::vector<std::size_t> wires(num_qubits);
            for (std::size_t q = 0; q < num_qubits; q++) {
                wires[q] = q;
            }
            std::vector<std::vector<std::size_t>> wire_sets{wires};
            std::shuffle(wires.begin(), wires.end(), gen);
            wires.resize(
                std::uniform_int_distribution<std::size_t>(1, num_qubits)(gen));
            wire_sets.push_back(wires);

            for (const auto &subset : wire_sets) {
                const auto expected = probabilities(state, subset);
                const auto actual =
                    Pennylane::LightningCliffish::GF2::probabilities(
                        tableau.computationalSupport(subset));
                REQUIRE(actual.size() == expected.size());
                for (std::size_t i = 0; i < expected.size(); i++) {
                    CHECK(std::abs(actual[i] - expected[i]) < 1e-9);
                }
            }
            // Reading the support leaves the state unchanged.
            requireSameState(tableau, state, gen, 20);
        }
    }
}

TEST_CASE("NativeTableau samples are possible outcomes on many qubits",
          "[NativeTableau]") {
    namespace GF2 = Pennylane::LightningCliffish::GF2;
    const std::size_t num_qubits = 200;
    std::mt19937 gen{22};
    NativeTableau<> tableau(num_qubits);
    std::uniform_int_distribution<unsigned> turns(1, 3);
    for (int i = 0; i < 600; i++) {
        tableau.applyBlock(
            {EngineOp::rotation(randomPauli(num_qubits, gen),
                                static_cast<std::uint8_t>(turns(gen)))},
            gen);
    }

    std::vector<std::size_t> qubits(num_qubits);
    for (std::size_t q = 0; q < num_qubits; q++) {
        qubits[q] = q;
    }
    const auto support = tableau.computationalSupport(qubits);
    const auto samples = GF2::sampleUniform(support, 10, 5);

    // Projecting a copy onto every bit of a sample never hits a
    // zero-probability outcome.
    for (std::size_t shot = 0; shot < samples.num_shots; shot++) {
        const auto copy = tableau.clone();
        Tableau::EngineBlock block;
        for (std::size_t q = 0; q < num_qubits; q++) {
            block.push_back(EngineOp::measurement(
                singleQubitPauli(num_qubits, q, 'Z'), samples.bit(shot, q)));
        }
        REQUIRE_NOTHROW(copy->applyBlock(block, gen));
    }
}

TEST_CASE("NativeTableau::clone is independent of the original",
          "[NativeTableau]") {
    std::mt19937 gen{23};
    NativeTableau<> tableau(3);
    State state = zeroState(3);
    applyRandomRotations(tableau, state, 20, gen);

    const auto copy = tableau.clone();
    CHECK(copy->name() == "native");
    CHECK(copy->numQubits() == 3);
    copy->applyBlock({EngineOp::rotation(randomPauli(3, gen), 1),
                      EngineOp::measurement(randomPauli(3, gen))},
                     gen);
    requireSameState(tableau, state, gen);
}
