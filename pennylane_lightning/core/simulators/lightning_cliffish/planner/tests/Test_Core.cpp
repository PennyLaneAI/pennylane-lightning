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
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "CircuitHelpers.hpp"
#include "Core.hpp"
#include "DenseReference.hpp"
#include "GateExpansions.hpp"
#include "HybridMethod.hpp"

namespace {
using namespace Pennylane::LightningCliffish;
using namespace Pennylane::LightningCliffish::Testing;
using Catch::Matchers::ContainsSubstring;
} // namespace

TEST_CASE("Core prepares a Bell state", "[Core]") {
    std::mt19937 gen{1};
    Core core;
    core.setPRNG(&gen);
    core.allocate(2);
    core.append(gate("Hadamard", {0}));
    core.append(gate("CNOT", {0, 1}));

    CHECK(core.expectation("ZZ", {0, 1}) == 1);
    CHECK(core.expectation("XX", {0, 1}) == 1);
    CHECK(core.expectation("YY", {0, 1}) == -1);
    CHECK(core.expectation("Z", {0}) == 0);
    CHECK(core.expectation("Z", {1}) == 0);
}

TEST_CASE("Core measurements are correlated and forced on demand", "[Core]") {
    std::mt19937 gen{2};
    int ones = 0;
    for (int trial = 0; trial < 200; trial++) {
        Core core;
        core.setPRNG(&gen);
        core.allocate(2);
        core.append(gate("Hadamard", {0}));
        core.append(gate("CNOT", {0, 1}));
        const auto first = core.measure(measure(0));
        const auto second = core.measure(measure(1));
        // Asking for a value forces a flush of everything buffered.
        const bool value = core.value(first);
        CHECK(core.value(second) == value);
        ones += value ? 1 : 0;
    }
    CHECK(ones > 50);
    CHECK(ones < 150);
}

TEST_CASE("Core projects onto postselected outcomes", "[Core]") {
    std::mt19937 gen{3};
    Core core;
    core.setPRNG(&gen);
    core.allocate(2);
    core.append(gate("Hadamard", {0}));
    const auto result = core.measure(measure(0, true));
    CHECK(core.value(result));
    CHECK(core.expectation("Z", {0}) == -1);

    // Qubit 1 is |0>, so outcome 1 has zero probability.
    const auto impossible = core.measure(measure(1, true));
    REQUIRE_THROWS_WITH(core.value(impossible),
                        ContainsSubstring("zero probability"));
}

TEST_CASE("Core releases and resets qubits", "[Core]") {
    std::mt19937 gen{4};

    SECTION("A released qubit's slot is reused in |0>") {
        Planner::HybridMethod method;
        Program::Program program;
        Planner::Results results;
        program.allocate(2);
        program.append(gate("PauliX", {1}));
        program.append(gate("Hadamard", {0}));
        program.release(1);
        const auto reused = program.allocate(1);
        CHECK(reused == std::vector<QubitId>{2});
        method.run(method.lower(program.takeWindow()), gen, results);

        CHECK(method.engine().numQubits() == 2);
        CHECK(method.expectation("Z", {2}) == 1);
        CHECK(method.expectation("X", {0}) == 1);
    }

    SECTION("Reset returns one qubit to |0>") {
        Core core;
        core.setPRNG(&gen);
        core.allocate(2);
        core.append(gate("Hadamard", {0}));
        core.append(gate("CNOT", {0, 1}));
        core.append(Op{.kind = OpKind::Reset, .wires = {1}});
        CHECK(core.expectation("Z", {1}) == 1);
        // Qubit 0 is left in a basis state, by the measurement in the reset.
        CHECK(std::abs(core.expectation("Z", {0})) == 1);
    }
}

TEST_CASE("Core matches a dense state on random Clifford circuits", "[Core]") {
    std::mt19937 gen{2027};
    for (std::size_t num_qubits = 1; num_qubits <= 4; num_qubits++) {
        for (int trial = 0; trial < 10; trial++) {
            Core core;
            core.setPRNG(&gen);
            core.allocate(num_qubits);
            State state = zeroState(num_qubits);

            for (int step = 0; step < 40; step++) {
                if (step % 10 == 9) {
                    // A mid-circuit measurement, forced when it is random.
                    const QubitId wire = std::uniform_int_distribution<QubitId>(
                        0, num_qubits - 1)(gen);
                    const auto z = singleQubitPauli(num_qubits, wire, 'Z');
                    const double value = expectation(state, z);
                    if (std::abs(value) > 0.5) {
                        const auto result = core.measure(measure(wire));
                        CHECK(core.value(result) == (value < 0));
                    } else {
                        const bool outcome =
                            std::bernoulli_distribution(0.5)(gen);
                        const auto result =
                            core.measure(measure(wire, outcome));
                        CHECK(core.value(result) == outcome);
                        state = project(state, z, outcome);
                    }
                    continue;
                }
                const Op op = randomGate(num_qubits, gen);
                core.append(op);
                applyDense(state, op, num_qubits);
            }

            for (int check = 0; check < 30; check++) {
                auto pauli = randomPauli(num_qubits, gen);
                pauli.setLogI(0);
                const std::string word = pauli.toString().substr(1);
                INFO("Pauli word " << word);
                CHECK(std::abs(core.expectation(word, allWires(num_qubits)) -
                               expectation(state, pauli)) < 1e-9);
            }
        }
    }
}

TEST_CASE("Core runs a 1000-qubit GHZ circuit", "[Core]") {
    const std::size_t num_qubits = 1000;
    std::mt19937 gen{5};
    Core core;
    core.setPRNG(&gen);
    core.allocate(num_qubits);
    core.append(gate("Hadamard", {0}));
    for (QubitId q = 0; q + 1 < num_qubits; q++) {
        core.append(gate("CNOT", {q, q + 1}));
    }
    CHECK(core.expectation("ZZ", {0, num_qubits - 1}) == 1);
    CHECK(core.expectation(std::string(num_qubits, 'X'),
                           allWires(num_qubits)) == 1);

    const auto first = core.measure(measure(0));
    const auto last = core.measure(measure(num_qubits - 1));
    CHECK(core.value(first) == core.value(last));
}

TEST_CASE("Core fails loudly on what it cannot run", "[Core]") {
    std::mt19937 gen{6};

    SECTION("A dense register over its limit") {
        Core core(Planner::HybridOptions{.max_dense_qubits = 2});
        core.setPRNG(&gen);
        core.allocate(3);
        for (QubitId q = 0; q < 3; q++) {
            core.append(gate("Hadamard", {q}));
            core.append(gate("T", {q}));
        }
        REQUIRE_THROWS_WITH(core.flush(),
                            ContainsSubstring("dense register of 3 qubits") &&
                                ContainsSubstring("max_dense_qubits = 2") &&
                                ContainsSubstring("lightning.qubit"));
    }

    SECTION("Conditional ops") {
        Core core;
        core.setPRNG(&gen);
        core.allocate(1);
        const auto result = core.measure(measure(0));
        auto op = gate("PauliX", {0});
        op.predicate = result;
        core.append(op);
        REQUIRE_THROWS_WITH(core.flush(), ContainsSubstring("conditional ops"));
    }

    SECTION("Unknown results") {
        Core core;
        REQUIRE_THROWS_WITH(core.value(7), ContainsSubstring("Unknown"));
    }
}
