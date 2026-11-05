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
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "CircuitHelpers.hpp"
#include "Core.hpp"
#include "DenseReference.hpp"
#include "Observables.hpp"

namespace {
using namespace Pennylane::LightningCliffish;
using namespace Pennylane::LightningCliffish::Testing;
using Catch::Matchers::ContainsSubstring;
using Planner::Observable;
using Planner::PauliTerm;

/// Apply the same random Clifford circuit to a core and a dense state.
void prepareRandomState(Core &core, State &state, std::size_t num_qubits,
                        std::size_t num_gates, std::mt19937 &gen) {
    core.allocate(num_qubits);
    state = zeroState(num_qubits);
    for (std::size_t i = 0; i < num_gates; i++) {
        const Op op = randomGate(num_qubits, gen);
        core.append(op);
        applyDense(state, op, num_qubits);
    }
}

/// A random observable on all wires, sometimes with an identity term.
auto randomObservable(std::size_t num_qubits, std::size_t num_terms,
                      std::mt19937 &gen) -> Observable {
    std::uniform_real_distribution<double> coefficient(-2.0, 2.0);
    Observable observable;
    for (std::size_t t = 0; t < num_terms; t++) {
        auto pauli = randomPauli(num_qubits, gen);
        pauli.setLogI(0);
        observable.push_back(PauliTerm{coefficient(gen),
                                       pauli.toString().substr(1),
                                       allWires(num_qubits)});
    }
    if (std::bernoulli_distribution(0.5)(gen)) {
        observable.push_back(PauliTerm{coefficient(gen), "", {}});
    }
    return observable;
}

/// The observable's terms as dense Pauli words.
auto densePaulis(const Observable &observable, std::size_t num_qubits)
    -> std::pair<std::vector<double>, std::vector<PauliWord>> {
    std::vector<double> coefficients;
    std::vector<PauliWord> paulis;
    for (const auto &term : observable) {
        PauliWord pauli(num_qubits);
        for (std::size_t k = 0; k < term.wires.size(); k++) {
            pauli.set(term.wires[k], term.pauli[k]);
        }
        coefficients.push_back(term.coefficient);
        paulis.push_back(pauli);
    }
    return {coefficients, paulis};
}

auto denseExpval(const State &state, const Observable &observable,
                 std::size_t num_qubits) -> double {
    const auto [coefficients, paulis] = densePaulis(observable, num_qubits);
    double value = 0.0;
    for (std::size_t k = 0; k < paulis.size(); k++) {
        value += coefficients[k] * expectation(state, paulis[k]);
    }
    return value;
}

auto denseVar(const State &state, const Observable &observable,
              std::size_t num_qubits) -> double {
    const auto [coefficients, paulis] = densePaulis(observable, num_qubits);
    const State applied = applySum(state, coefficients, paulis);
    double second_moment = 0.0;
    for (const auto &amplitude : applied) {
        second_moment += std::norm(amplitude);
    }
    const double mean = denseExpval(state, observable, num_qubits);
    return second_moment - mean * mean;
}

auto bellCore(std::mt19937 &gen) -> Core {
    Core core;
    core.setPRNG(&gen);
    core.allocate(2);
    core.append(gate("Hadamard", {0}));
    core.append(gate("CNOT", {0, 1}));
    return core;
}
} // namespace

TEST_CASE("Core::expval and Core::var are exact", "[Outputs]") {
    std::mt19937 gen{31};
    for (std::size_t num_qubits = 1; num_qubits <= 4; num_qubits++) {
        for (int trial = 0; trial < 10; trial++) {
            Core core;
            core.setPRNG(&gen);
            State state;
            prepareRandomState(core, state, num_qubits, 30, gen);

            for (std::size_t num_terms : {1, 2, 5}) {
                const auto observable =
                    randomObservable(num_qubits, num_terms, gen);
                CHECK(std::abs(core.expval(observable) -
                               denseExpval(state, observable, num_qubits)) <
                      1e-9);
                CHECK(std::abs(core.var(observable) -
                               denseVar(state, observable, num_qubits)) < 1e-9);
            }
        }
    }
}

TEST_CASE("Core observables cover named, tensor and sum observables",
          "[Outputs]") {
    std::mt19937 gen{32};
    Core core = bellCore(gen);

    // The empty observable is zero, and the identity is one.
    CHECK(core.expval(Observable{}) == 0.0);
    CHECK(core.expval(Planner::pauliObservable("", {})) == 1.0);
    CHECK(core.var(Planner::pauliObservable("", {})) == 0.0);

    // Tensor products and Hamiltonians of the Bell state.
    CHECK(core.expval(Planner::pauliObservable("ZZ", {0, 1})) == 1.0);
    CHECK(core.var(Planner::pauliObservable("ZZ", {0, 1})) == 0.0);
    CHECK(core.var(Planner::pauliObservable("Z", {0}, 3.0)) == 9.0);
    const Observable hamiltonian{
        {0.5, "ZZ", {0, 1}}, {-2.0, "XX", {0, 1}}, {1.5, "Z", {1}}};
    CHECK(core.expval(hamiltonian) == -1.5);
    // ZZ and XX are fixed; only Z_1 varies: 1.5^2 * 1.
    CHECK(std::abs(core.var(hamiltonian) - 2.25) < 1e-12);

    // A Hadamard observable is (X + Z) / sqrt(2): on |0>, <H> = 1/sqrt(2).
    Core zero;
    zero.allocate(1);
    const double r = 1.0 / std::sqrt(2.0);
    const Observable hadamard{{r, "X", {0}}, {r, "Z", {0}}};
    CHECK(std::abs(zero.expval(hadamard) - r) < 1e-12);
    CHECK(std::abs(zero.var(hadamard) - 0.5) < 1e-12);
}

TEST_CASE("Core::probabilities match a dense state", "[Outputs]") {
    std::mt19937 gen{33};
    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (int trial = 0; trial < 10; trial++) {
            Core core;
            core.setPRNG(&gen);
            State state;
            prepareRandomState(core, state, num_qubits, 30, gen);

            auto wires = allWires(num_qubits);
            std::shuffle(wires.begin(), wires.end(), gen);
            wires.resize(
                std::uniform_int_distribution<std::size_t>(1, num_qubits)(gen));
            const auto expected = probabilities(state, wires);
            const auto actual = core.probabilities(wires);
            REQUIRE(actual.size() == expected.size());
            for (std::size_t i = 0; i < expected.size(); i++) {
                CHECK(std::abs(actual[i] - expected[i]) < 1e-9);
            }
        }
    }
}

TEST_CASE("Core::sample draws from the state's distribution", "[Outputs]") {
    std::mt19937 gen{34};
    const std::size_t shots = 40000;
    for (int trial = 0; trial < 5; trial++) {
        const std::size_t num_qubits = 3;
        Core core;
        core.setPRNG(&gen);
        State state;
        prepareRandomState(core, state, num_qubits, 30, gen);

        auto wires = allWires(num_qubits);
        std::shuffle(wires.begin(), wires.end(), gen);
        const auto expected = probabilities(state, wires);
        const auto samples = core.sample(wires, shots);
        REQUIRE(samples.num_shots == shots);
        REQUIRE(samples.num_bits == num_qubits);

        std::vector<std::size_t> histogram(expected.size(), 0);
        for (std::size_t shot = 0; shot < shots; shot++) {
            std::size_t index = 0;
            for (std::size_t k = 0; k < num_qubits; k++) {
                index = (index << 1U) | (samples.bit(shot, k) ? 1U : 0U);
            }
            histogram[index]++;
        }
        for (std::size_t i = 0; i < expected.size(); i++) {
            const double frequency =
                static_cast<double>(histogram[i]) / static_cast<double>(shots);
            if (expected[i] == 0.0) {
                CHECK(histogram[i] == 0);
            } else {
                const double sigma =
                    std::sqrt(expected[i] * (1 - expected[i]) / shots);
                CHECK(std::abs(frequency - expected[i]) < 5 * sigma);
            }
        }
    }
}

TEST_CASE("Core::counts use wires[0] as the most significant bit",
          "[Outputs]") {
    std::mt19937 gen{35};
    Core core;
    core.setPRNG(&gen);
    core.allocate(3);
    core.append(gate("PauliX", {1}));

    using Counts = std::map<std::uint64_t, std::size_t>;
    CHECK(core.counts({0, 1, 2}, 50) == Counts{{0b010, 50}});
    CHECK(core.counts({1, 2}, 50) == Counts{{0b10, 50}});
    CHECK(core.counts({2, 1}, 50) == Counts{{0b01, 50}});

    Core bell = bellCore(gen);
    const auto histogram = bell.counts({0, 1}, 10000);
    REQUIRE(histogram.size() == 2);
    CHECK(histogram.at(0b00) + histogram.at(0b11) == 10000);
    CHECK(std::abs(static_cast<double>(histogram.at(0b00)) - 5000.0) < 250.0);
}

TEST_CASE("Core sampled expectations estimate from shots", "[Outputs]") {
    std::mt19937 gen{36};
    const std::size_t shots = 10000;
    Core core = bellCore(gen);

    // Deterministic terms are exact at any shot count.
    CHECK(core.expval(Planner::pauliObservable("ZZ", {0, 1}), 1) == 1.0);
    CHECK(core.expval(Planner::pauliObservable("YY", {0, 1}), 7) == -1.0);

    // A term with mean zero averages fair +1/-1 outcomes.
    const double z0 = core.expval(Planner::pauliObservable("Z", {0}), shots);
    CHECK(std::abs(z0) < 5.0 / std::sqrt(static_cast<double>(shots)));

    const Observable hamiltonian{{0.5, "ZZ", {0, 1}}, {2.0, "Z", {0}}};
    const double estimate = core.expval(hamiltonian, shots);
    CHECK(std::abs(estimate - 0.5) <
          2.0 * 5.0 / std::sqrt(static_cast<double>(shots)));

    // The sampled variance of one term uses the same estimate.
    CHECK(core.var(Planner::pauliObservable("ZZ", {0, 1}, 2.0), shots) == 0.0);
    const double var_z = core.var(Planner::pauliObservable("Z", {0}), shots);
    CHECK(var_z <= 1.0);
    CHECK(var_z > 1.0 - 25.0 / static_cast<double>(shots));
}

TEST_CASE("Core sampled outputs are reproducible from the PRNG seed",
          "[Outputs]") {
    auto run = [](std::uint32_t seed) {
        std::mt19937 gen{seed};
        Core core;
        core.setPRNG(&gen);
        State state;
        std::mt19937 circuit_gen{77};
        prepareRandomState(core, state, 4, 30, circuit_gen);
        const auto samples = core.sample(allWires(4), 1000);
        const double estimate =
            core.expval(Planner::pauliObservable("XZ", {0, 3}), 1000);
        return std::make_pair(samples.data, estimate);
    };
    CHECK(run(5) == run(5));
    CHECK(run(5).first != run(6).first);
}

TEST_CASE("Core outputs leave the state unchanged", "[Outputs]") {
    std::mt19937 gen{37};
    Core core;
    core.setPRNG(&gen);
    State state;
    prepareRandomState(core, state, 3, 30, gen);
    const auto observable = randomObservable(3, 3, gen);
    const double before = core.expval(observable);

    core.sample(allWires(3), 100);
    core.counts({0, 2}, 100);
    core.probabilities({1});
    core.expval(observable, 100);
    CHECK(core.expval(observable) == before);

    // The program continues after an output.
    const Op op = randomGate(3, gen);
    core.append(op);
    applyDense(state, op, 3);
    CHECK(std::abs(core.expval(observable) -
                   denseExpval(state, observable, 3)) < 1e-9);
}

TEST_CASE("Core samples a 1000-qubit GHZ state", "[Outputs]") {
    const std::size_t num_qubits = 1000;
    std::mt19937 gen{38};
    Core core;
    core.setPRNG(&gen);
    core.allocate(num_qubits);
    core.append(gate("Hadamard", {0}));
    for (QubitId q = 0; q + 1 < num_qubits; q++) {
        core.append(gate("CNOT", {q, q + 1}));
    }

    const std::size_t shots = 10000;
    const auto samples = core.sample(allWires(num_qubits), shots);
    std::size_t ones = 0;
    for (std::size_t shot = 0; shot < shots; shot++) {
        const bool first = samples.bit(shot, 0);
        bool all_equal = true;
        for (std::size_t q = 1; q < num_qubits; q++) {
            all_equal = all_equal && samples.bit(shot, q) == first;
        }
        CHECK(all_equal);
        ones += first ? 1 : 0;
    }
    CHECK(std::abs(static_cast<double>(ones) - shots / 2.0) < 250.0);
    CHECK(core.probabilities({0, 500, 999}) ==
          std::vector<double>{0.5, 0, 0, 0, 0, 0, 0, 0.5});
}

TEST_CASE("Core outputs fail loudly on invalid requests", "[Outputs]") {
    std::mt19937 gen{39};
    Core core;
    core.setPRNG(&gen);
    core.allocate(70);
    core.release(69);

    REQUIRE_THROWS_WITH(core.expval(Planner::pauliObservable("Z", {0}), 0),
                        ContainsSubstring("at least one shot"));
    REQUIRE_THROWS_WITH(
        core.var(Observable{{1.0, "Z", {0}}, {1.0, "X", {1}}}, 10),
        ContainsSubstring("one Pauli term"));
    REQUIRE_THROWS_WITH(core.probabilities(allWires(25)),
                        ContainsSubstring("more than 24 wires"));
    REQUIRE_THROWS_WITH(core.counts(allWires(65), 10),
                        ContainsSubstring("at most 64 wires"));
    REQUIRE_THROWS_WITH(core.sample({69}, 10),
                        ContainsSubstring("no engine slot"));
    REQUIRE_THROWS_WITH(core.expval(Planner::pauliObservable("ZZ", {0})),
                        ContainsSubstring("one character per wire"));
}
