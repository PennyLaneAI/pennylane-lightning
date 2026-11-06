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
 * Tests of the hybrid method with a non-empty dense register: Clifford+T and
 * arbitrary-angle circuits against the dense reference.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "CircuitHelpers.hpp"
#include "Core.hpp"
#include "DenseReference.hpp"

namespace {
using namespace Pennylane::LightningCliffish;
using namespace Pennylane::LightningCliffish::Testing;
using Catch::Matchers::ContainsSubstring;
using Planner::HybridOptions;
using Planner::Recompression;

/// A random non-Clifford gate: T, its inverse, or a rotation at a random
/// angle.
auto randomNonCliffordGate(std::size_t num_qubits, std::mt19937 &gen) -> Op {
    static const std::vector<std::tuple<std::string, std::size_t>> gates{
        {"T", 1},          {"RX", 1},      {"RY", 1},      {"RZ", 1},
        {"PhaseShift", 1}, {"IsingXX", 2}, {"IsingZZ", 2}, {"PauliRot", 3},
    };
    std::vector<QubitId> wires = allWires(num_qubits);
    std::shuffle(wires.begin(), wires.end(), gen);
    std::uniform_real_distribution<double> angle(-M_PI, M_PI);
    while (true) {
        const auto &[name, num_wires] =
            gates[std::uniform_int_distribution<std::size_t>(0, gates.size() -
                                                                    1)(gen)];
        if (num_wires > num_qubits) {
            continue;
        }
        const std::vector<QubitId> op_wires{
            wires.begin(),
            wires.begin() + static_cast<std::ptrdiff_t>(num_wires)};
        if (name == "T") {
            return gate("T", op_wires, {},
                        std::bernoulli_distribution(0.5)(gen));
        }
        if (name == "PauliRot") {
            static const std::string paulis{"IXYZ"};
            std::string word;
            for (std::size_t k = 0; k < num_wires; k++) {
                word += paulis[std::uniform_int_distribution<std::size_t>(0, 3)(
                    gen)];
            }
            return Op{.kind = OpKind::PauliRot,
                      .params = {angle(gen)},
                      .wires = op_wires,
                      .pauli_word = word};
        }
        return gate(name, op_wires, {angle(gen)});
    }
}

/// A random circuit with about `non_clifford_fraction` non-Clifford gates.
auto randomCircuit(std::size_t num_qubits, std::size_t depth,
                   double non_clifford_fraction, std::mt19937 &gen)
    -> std::vector<Op> {
    std::vector<Op> circuit;
    std::bernoulli_distribution non_clifford(non_clifford_fraction);
    for (std::size_t i = 0; i < depth; i++) {
        circuit.push_back(non_clifford(gen)
                              ? randomNonCliffordGate(num_qubits, gen)
                              : randomGate(num_qubits, gen));
    }
    return circuit;
}

/// Apply an op to the dense reference: a gate, a PauliRot, or nothing.
void applyReference(State &state, const Op &op, std::size_t num_qubits) {
    applyDense(state, op, num_qubits);
}

/// The Pauli word with `pauli[k]` on `wires[k]`, on `num_qubits` qubits.
auto wordOn(const std::string &pauli, const std::vector<QubitId> &wires,
            std::size_t num_qubits) -> PauliWord {
    PauliWord word(num_qubits);
    for (std::size_t k = 0; k < wires.size(); k++) {
        word.set(wires[k], pauli[k]);
    }
    return word;
}

/**
 * @brief The core's state and the dense state agree: probabilities of every
 * wire order tried, and Pauli expectations (all of them on few qubits,
 * which fixes the state).
 */
void requireSameState(Core &core, const State &state, std::size_t num_qubits,
                      std::mt19937 &gen) {
    const auto wires = allWires(num_qubits);
    const auto probabilities = core.probabilities(wires);
    const auto expected = Testing::probabilities(state, wires);
    for (std::size_t i = 0; i < expected.size(); i++) {
        INFO("Outcome " << i);
        REQUIRE(std::abs(probabilities[i] - expected[i]) < 1e-10);
    }

    auto reversed = wires;
    std::reverse(reversed.begin(), reversed.end());
    reversed.pop_back();
    const auto partial = core.probabilities(reversed);
    const auto expected_partial = Testing::probabilities(state, reversed);
    for (std::size_t i = 0; i < expected_partial.size(); i++) {
        INFO("Partial outcome " << i);
        REQUIRE(std::abs(partial[i] - expected_partial[i]) < 1e-10);
    }

    auto check = [&](const std::string &word) {
        INFO("Pauli word " << word);
        REQUIRE(std::abs(core.expectation(word, wires) -
                         expectation(state, PauliWord::fromString(word))) <
                1e-10);
    };
    if (num_qubits <= 4) {
        static const std::string paulis{"IXYZ"};
        std::size_t total = 1;
        for (std::size_t q = 0; q < num_qubits; q++) {
            total *= 4;
        }
        for (std::size_t index = 1; index < total; index++) {
            std::string word;
            for (std::size_t q = 0, rest = index; q < num_qubits;
                 q++, rest /= 4) {
                word += paulis[rest % 4];
            }
            check(word);
        }
    } else {
        for (std::size_t i = 0; i < 60; i++) {
            check(randomPauli(num_qubits, gen).toString().substr(1));
        }
    }
}

/// Gate-level QFT, with each controlled phase as PhaseShifts and CNOTs.
auto qft(std::size_t num_qubits) -> std::vector<Op> {
    std::vector<Op> circuit;
    for (QubitId i = 0; i < num_qubits; i++) {
        circuit.push_back(gate("Hadamard", {i}));
        for (QubitId j = i + 1; j < num_qubits; j++) {
            const double phi = M_PI / std::ldexp(1.0, static_cast<int>(j - i));
            circuit.push_back(gate("PhaseShift", {j}, {phi / 2}));
            circuit.push_back(gate("CNOT", {j, i}));
            circuit.push_back(gate("PhaseShift", {i}, {-phi / 2}));
            circuit.push_back(gate("CNOT", {j, i}));
            circuit.push_back(gate("PhaseShift", {i}, {phi / 2}));
        }
    }
    return circuit;
}

auto runOn(const std::vector<Op> &circuit, std::size_t num_qubits,
           HybridOptions options, std::mt19937 &gen) -> Core {
    Core core(options);
    core.setPRNG(&gen);
    core.allocate(num_qubits);
    for (const auto &op : circuit) {
        core.append(op);
    }
    return core;
}

const std::vector<HybridOptions> all_options{
    {.recompression = Recompression::None, .dense_checks = false},
    {.recompression = Recompression::T1, .dense_checks = false},
    {.recompression = Recompression::T2, .dense_checks = false},
    {.recompression = Recompression::None},
    {.recompression = Recompression::T2},
    {},
};

auto describe(const HybridOptions &options) -> std::string {
    return (options.recompression.has_value()
                ? Planner::recompressionName(*options.recompression)
                : std::string{"auto"}) +
           (options.dense_checks ? " with checks" : " without checks");
}
} // namespace

TEST_CASE("Random Clifford+T and rotation circuits match the dense state",
          "[Hybrid]") {
    std::mt19937 gen{101};
    for (const auto &options : all_options) {
        for (std::size_t num_qubits : {1, 2, 4, 6}) {
            for (double fraction : {0.1, 0.4}) {
                INFO(describe(options) << ", " << num_qubits << " qubits, "
                                       << fraction << " non-Clifford");
                const auto circuit =
                    randomCircuit(num_qubits, 50, fraction, gen);
                auto core = runOn(circuit, num_qubits, options, gen);
                State state = zeroState(num_qubits);
                for (const auto &op : circuit) {
                    applyReference(state, op, num_qubits);
                }
                requireSameState(core, state, num_qubits, gen);
            }
        }
    }
}

TEST_CASE("Mid-circuit measurements follow one branch of the dense state",
          "[Hybrid]") {
    std::mt19937 gen{102};
    const std::size_t num_qubits = 4;
    for (const auto &options : all_options) {
        INFO(describe(options));
        for (std::size_t trial = 0; trial < 6; trial++) {
            Core core(options);
            core.setPRNG(&gen);
            core.allocate(num_qubits);
            State state = zeroState(num_qubits);
            for (std::size_t round = 0; round < 5; round++) {
                for (const auto &op : randomCircuit(num_qubits, 12, 0.4, gen)) {
                    core.append(op);
                    applyReference(state, op, num_qubits);
                }
                // A Z measurement or a random Pauli measurement.
                const QubitId wire = std::uniform_int_distribution<QubitId>(
                    0, num_qubits - 1)(gen);
                if (round % 2 == 0) {
                    const auto result = core.measure(measure(wire));
                    const bool outcome = core.value(result);
                    state = project(state, wordOn("Z", {wire}, num_qubits),
                                    outcome);
                } else {
                    const auto word = randomPauli(num_qubits, gen);
                    const auto pauli = word.toString().substr(1);
                    const auto result =
                        core.measure(Op{.kind = OpKind::PauliMeasure,
                                        .wires = allWires(num_qubits),
                                        .pauli_word = pauli});
                    const bool outcome = core.value(result);
                    state =
                        project(state, PauliWord::fromString(pauli), outcome);
                }
            }
            requireSameState(core, state, num_qubits, gen);
        }
    }
}

TEST_CASE("Postselection projects the dense register", "[Hybrid]") {
    std::mt19937 gen{103};
    Core core;
    core.setPRNG(&gen);
    core.allocate(2);
    core.append(gate("Hadamard", {0}));
    core.append(gate("T", {0}));
    core.append(gate("RY", {0}, {0.7}));
    core.append(gate("CNOT", {0, 1}));
    State state = zeroState(2);
    for (const auto &op : {gate("Hadamard", {0}), gate("T", {0}),
                           gate("RY", {0}, {0.7}), gate("CNOT", {0, 1})}) {
        applyReference(state, op, 2);
    }
    const auto result = core.measure(measure(1, true));
    CHECK(core.value(result));
    state = project(state, wordOn("Z", {1}, 2), true);
    requireSameState(core, state, 2, gen);

    // The other outcome now has zero probability.
    core.measure(measure(1, false));
    REQUIRE_THROWS_WITH(core.flush(), ContainsSubstring("zero probability"));
}

TEST_CASE("Released qubits leave the dense register and are reused",
          "[Hybrid]") {
    std::mt19937 gen{104};
    Core core;
    core.setPRNG(&gen);
    const auto qubits = core.allocate(3);
    for (const auto &op :
         {gate("Hadamard", {0}), gate("T", {0}), gate("CNOT", {0, 1}),
          gate("Hadamard", {2}), gate("T", {2})}) {
        core.append(op);
    }
    core.release(qubits[2]);
    const auto reused = core.allocate(1);
    core.append(gate("Hadamard", {reused[0]}));
    // Qubits 0 and 1 hold (|00> + e^{i pi/4}|11>)/sqrt(2); the new qubit |+>.
    State state = zeroState(3);
    for (const auto &op : {gate("Hadamard", {0}), gate("T", {0}),
                           gate("CNOT", {0, 1}), gate("Hadamard", {2})}) {
        applyReference(state, op, 3);
    }
    const std::vector<QubitId> wires{0, 1, reused[0]};
    const auto probabilities = core.probabilities(wires);
    const auto expected = Testing::probabilities(state, {0, 1, 2});
    for (std::size_t i = 0; i < expected.size(); i++) {
        CHECK(std::abs(probabilities[i] - expected[i]) < 1e-12);
    }
    CHECK(std::abs(core.expectation("XXX", wires) -
                   expectation(state, PauliWord::fromString("XXX"))) < 1e-12);
}

TEST_CASE("Variances of Pauli sums on the dense register", "[Hybrid]") {
    std::mt19937 gen{105};
    const std::size_t num_qubits = 3;
    const auto circuit = randomCircuit(num_qubits, 40, 0.4, gen);
    auto core = runOn(circuit, num_qubits, {}, gen);
    State state = zeroState(num_qubits);
    for (const auto &op : circuit) {
        applyReference(state, op, num_qubits);
    }
    const Planner::Observable observable{
        {0.5, "XZ", {0, 2}}, {-1.5, "Y", {1}}, {0.25, "ZZ", {0, 1}}};
    std::vector<double> coefficients;
    std::vector<PauliWord> words;
    for (const auto &term : observable) {
        coefficients.push_back(term.coefficient);
        words.push_back(wordOn(term.pauli, term.wires, num_qubits));
    }
    const State applied = applySum(state, coefficients, words);
    std::complex<double> mean = 0.0;
    std::complex<double> second = 0.0;
    for (std::size_t i = 0; i < state.size(); i++) {
        mean += std::conj(state[i]) * applied[i];
        second += std::conj(applied[i]) * applied[i];
    }
    CHECK(std::abs(core.expval(observable) - mean.real()) < 1e-10);
    CHECK(std::abs(core.var(observable) -
                   (second.real() - mean.real() * mean.real())) < 1e-10);
}

TEST_CASE("Samples of the dense register follow its probabilities",
          "[Hybrid]") {
    std::mt19937 gen{106};
    const std::size_t num_qubits = 4;
    const std::size_t shots = 40000;
    for (std::size_t trial = 0; trial < 4; trial++) {
        const auto circuit = randomCircuit(num_qubits, 30, 0.4, gen);
        auto core = runOn(circuit, num_qubits, {}, gen);
        const std::vector<QubitId> wires{2, 0, 3};
        const auto probabilities = core.probabilities(wires);
        const auto counts = core.counts(wires, shots);
        for (std::size_t outcome = 0; outcome < probabilities.size();
             outcome++) {
            const double p = probabilities[outcome];
            const double observed =
                counts.contains(outcome)
                    ? static_cast<double>(counts.at(outcome)) / shots
                    : 0.0;
            const double sigma = std::sqrt(p * (1 - p) / shots);
            INFO("Trial " << trial << ", outcome " << outcome << ", p = " << p);
            CHECK(std::abs(observed - p) <= 5 * sigma + 1e-12);
        }

        // A sampled expectation is close to the exact one.
        const Planner::Observable observable{{1.0, "XYZ", {0, 1, 2}}};
        const double exact = core.expval(observable);
        const double sampled = core.expval(observable, shots);
        CHECK(std::abs(sampled - exact) <=
              5 * std::sqrt((1 - exact * exact) / shots) + 1e-12);
    }
}

TEST_CASE("Seeded samples of the dense register are reproducible", "[Hybrid]") {
    auto samples = [](std::uint32_t seed) {
        std::mt19937 circuit_gen{107};
        const auto circuit = randomCircuit(3, 30, 0.4, circuit_gen);
        std::mt19937 gen{seed};
        auto core = runOn(circuit, 3, {}, gen);
        return core.sample(allWires(3), 500).data;
    };
    CHECK(samples(1) == samples(1));
    CHECK(samples(1) != samples(2));
}

TEST_CASE("Fusion reaches the true register size on gate-level QFT",
          "[Hybrid]") {
    std::mt19937 gen{108};
    for (std::size_t num_qubits : {4, 6, 8}) {
        INFO(num_qubits << " qubits");
        const auto circuit = qft(num_qubits);
        State state = zeroState(num_qubits);
        for (const auto &op : circuit) {
            applyReference(state, op, num_qubits);
        }

        auto none = runOn(
            circuit, num_qubits,
            {.recompression = Recompression::None, .dense_checks = false}, gen);
        requireSameState(none, state, num_qubits, gen);
        CHECK(none.report().peak_dense_qubits == num_qubits - 1);

        auto t2 = runOn(
            circuit, num_qubits,
            {.recompression = Recompression::T2, .dense_checks = false}, gen);
        requireSameState(t2, state, num_qubits, gen);
        CHECK(t2.report().peak_dense_qubits == 1);
        CHECK(t2.report().dense_qubits == 0);
        CHECK(t2.report().fusion_demotions > 0);

        // The dry run predicts t2's sizes exactly and picks it.
        auto chosen = runOn(circuit, num_qubits, {.dense_checks = false}, gen);
        chosen.flush();
        const auto report = chosen.report();
        CHECK(report.recompression == Recompression::T2);
        CHECK(report.predicted_peak == std::optional<std::size_t>{1});
        CHECK(report.predicted_final == std::optional<std::size_t>{0});
        CHECK(report.peak_dense_qubits == 1);
        CHECK(report.dense_qubits == 0);
    }
}

TEST_CASE("t1 fuses pi/8 rotations only", "[Hybrid]") {
    std::mt19937 gen{109};
    const std::vector<Op> t_pair{gate("Hadamard", {0}), gate("T", {0}),
                                 gate("T", {0}, {}, true)};
    const std::vector<Op> rz_pair{gate("Hadamard", {0}), gate("RZ", {0}, {0.3}),
                                  gate("RZ", {0}, {-0.3})};
    for (const auto &[circuit, t1_final] :
         {std::pair{t_pair, std::size_t{0}},
          std::pair{rz_pair, std::size_t{1}}}) {
        auto none = runOn(
            circuit, 1,
            {.recompression = Recompression::None, .dense_checks = false}, gen);
        none.flush();
        CHECK(none.report().dense_qubits == 1);
        auto t1 = runOn(
            circuit, 1,
            {.recompression = Recompression::T1, .dense_checks = false}, gen);
        t1.flush();
        CHECK(t1.report().dense_qubits == t1_final);
        auto t2 = runOn(
            circuit, 1,
            {.recompression = Recompression::T2, .dense_checks = false}, gen);
        t2.flush();
        CHECK(t2.report().dense_qubits == 0);
    }
}

TEST_CASE("Fusion follows the sign of the pulled-back axis", "[Hybrid]") {
    std::mt19937 gen{111};
    // After Y then H, Z pulls back to -X: the rotations act on the promoted
    // qubit with the opposite sign of their angles.
    for (const auto &prefix :
         {std::vector<Op>{gate("Hadamard", {0})},
          std::vector<Op>{gate("PauliY", {0}), gate("Hadamard", {0})}}) {
        for (const auto &[first, second] :
             {std::pair{gate("T", {0}), gate("T", {0}, {}, true)},
              std::pair{gate("RZ", {0}, {0.3}),
                        gate("RZ", {0}, {M_PI / 2 - 0.3})},
              std::pair{gate("RZ", {0}, {-0.3}), gate("RZ", {0}, {0.3})}}) {
            auto circuit = prefix;
            circuit.push_back(first);
            circuit.push_back(gate("CNOT", {0, 1}));
            circuit.push_back(gate("CNOT", {0, 1}));
            circuit.push_back(second);
            State state = zeroState(2);
            for (const auto &op : circuit) {
                applyReference(state, op, 2);
            }
            auto core = runOn(
                circuit, 2,
                {.recompression = Recompression::T2, .dense_checks = false},
                gen);
            requireSameState(core, state, 2, gen);
            CHECK(core.report().dense_qubits == 0);
            CHECK(core.report().fusion_demotions == 1);
        }
    }
}

TEST_CASE("Dense checks fold stabilizers back into the tableau", "[Hybrid]") {
    std::mt19937 gen{110};

    SECTION("The end check before an output") {
        // A rotation on two qubits and its inverse after a Clifford that
        // commutes with it: no policy fuses them, and the end check finds
        // the register back in a stabilizer state.
        const std::vector<Op> circuit{
            gate("Hadamard", {0}), gate("Hadamard", {1}),
            gate("IsingZZ", {0, 1}, {0.4}), gate("CZ", {0, 1}),
            gate("IsingZZ", {0, 1}, {-0.4})};
        State state = zeroState(2);
        for (const auto &op : circuit) {
            applyReference(state, op, 2);
        }
        auto unchecked = runOn(
            circuit, 2,
            {.recompression = Recompression::None, .dense_checks = false}, gen);
        requireSameState(unchecked, state, 2, gen);
        CHECK(unchecked.report().dense_qubits == 1);

        auto checked =
            runOn(circuit, 2, {.recompression = Recompression::None}, gen);
        requireSameState(checked, state, 2, gen);
        CHECK(checked.report().dense_qubits == 0);
        CHECK(checked.report().successful_dense_checks >= 1);
        CHECK(checked.report().check_demotions >= 1);
    }

    SECTION("Trigger 4 between rotations") {
        // Each qubit gets magic and then loses it again by a rotation that
        // only a dense check can see, so the register would grow to n
        // without checks.
        const std::size_t num_qubits = 8;
        std::vector<Op> circuit;
        for (QubitId q = 0; q < num_qubits; q++) {
            circuit.push_back(gate("Hadamard", {q}));
            circuit.push_back(gate("RZ", {q}, {0.5}));
            circuit.push_back(gate("RX", {q}, {0.3}));
            circuit.push_back(gate("RX", {q}, {-0.3}));
            circuit.push_back(gate("RZ", {q}, {-0.5}));
        }
        State state = zeroState(num_qubits);
        for (const auto &op : circuit) {
            applyReference(state, op, num_qubits);
        }
        auto unchecked = runOn(
            circuit, num_qubits,
            {.recompression = Recompression::None, .dense_checks = false}, gen);
        requireSameState(unchecked, state, num_qubits, gen);
        CHECK(unchecked.report().peak_dense_qubits == num_qubits);

        auto checked = runOn(circuit, num_qubits,
                             {.recompression = Recompression::None}, gen);
        requireSameState(checked, state, num_qubits, gen);
        CHECK(checked.report().peak_dense_qubits < num_qubits);
        CHECK(checked.report().dense_qubits == 0);
    }

    SECTION("A check never folds a near-identity rotation") {
        // exp(-i 1e-6 X) |+> is 1e-6 away from a stabilizer state: the
        // linear residual rejects it, where an expectation test would not.
        const std::vector<Op> circuit{gate("Hadamard", {0}),
                                      gate("RY", {0}, {2e-6})};
        auto core =
            runOn(circuit, 1, {.recompression = Recompression::None}, gen);
        State state = zeroState(1);
        for (const auto &op : circuit) {
            applyReference(state, op, 1);
        }
        requireSameState(core, state, 1, gen);
        CHECK(core.report().dense_qubits == 1);
    }
}

TEST_CASE("Hybrid method options are checked", "[Hybrid]") {
    REQUIRE_THROWS_WITH(
        Planner::HybridMethod(HybridOptions{.max_dense_qubits = 64}),
        ContainsSubstring("max_dense_qubits"));
    REQUIRE_THROWS_WITH(Planner::parseRecompression("t3"),
                        ContainsSubstring("Unknown recompression"));
    CHECK(Planner::parseRecompression("auto") == std::nullopt);
    CHECK(Planner::parseRecompression("t1") == Recompression::T1);
}
