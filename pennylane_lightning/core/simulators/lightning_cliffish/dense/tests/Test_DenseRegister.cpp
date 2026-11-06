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

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DenseReference.hpp"
#include "DenseRegister.hpp"

namespace {
using namespace Pennylane::LightningCliffish;
using namespace Pennylane::LightningCliffish::Testing;
using Catch::Matchers::ContainsSubstring;
using Dense::DensePauli;
using Dense::DenseRegister;

auto toDense(const PauliWord &pauli) -> DensePauli {
    DensePauli dense{.log_i = pauli.logI()};
    for (std::size_t q = 0; q < pauli.numQubits(); q++) {
        const char single = pauli.get(q);
        if (single == 'X' || single == 'Y') {
            dense.x |= std::uint64_t{1} << q;
        }
        if (single == 'Z' || single == 'Y') {
            dense.z |= std::uint64_t{1} << q;
        }
    }
    return dense;
}

void requireSameState(const DenseRegister<> &reg, const State &state) {
    const auto amplitudes = reg.hostAmplitudes();
    REQUIRE(amplitudes.size() == state.size());
    for (std::size_t k = 0; k < state.size(); k++) {
        INFO("Index " << k);
        REQUIRE(std::abs(amplitudes[k].real() - state[k].real()) < 1e-12);
        REQUIRE(std::abs(amplitudes[k].imag() - state[k].imag()) < 1e-12);
    }
}

/// Random rotations on both, at random and at Clifford angles.
void applyRandomRotations(DenseRegister<> &reg, State &state,
                          std::size_t num_rotations, std::mt19937 &gen) {
    std::uniform_real_distribution<double> angle(-M_PI, M_PI);
    std::uniform_int_distribution<unsigned> turns(1, 3);
    for (std::size_t i = 0; i < num_rotations; i++) {
        const auto pauli = randomPauli(reg.numQubits(), gen);
        if (i % 2 == 0) {
            const double half_angle = angle(gen);
            reg.rotate(toDense(pauli), half_angle);
            state = rotateBy(state, pauli, half_angle);
        } else {
            const auto quarter_turns = static_cast<std::uint8_t>(turns(gen));
            reg.rotateQuarterTurns(toDense(pauli), quarter_turns);
            state = rotate(state, pauli, quarter_turns);
        }
    }
}
} // namespace

TEST_CASE("DenseRegister starts in |0...0>", "[DenseRegister]") {
    for (std::size_t num_qubits : {0, 1, 3}) {
        const DenseRegister<> reg(num_qubits);
        CHECK(reg.numQubits() == num_qubits);
        requireSameState(reg, zeroState(num_qubits));
    }
    REQUIRE_THROWS_WITH(DenseRegister<>(Dense::max_register_qubits + 1),
                        ContainsSubstring("too large"));
}

TEST_CASE("DenseRegister rotations match the dense reference",
          "[DenseRegister]") {
    std::mt19937 gen{11};
    for (std::size_t num_qubits : {1, 2, 5}) {
        INFO("Qubits " << num_qubits);
        DenseRegister<> reg(num_qubits);
        State state = zeroState(num_qubits);
        applyRandomRotations(reg, state, 40, gen);
        requireSameState(reg, state);
    }
}

TEST_CASE("DenseRegister expectations and residuals", "[DenseRegister]") {
    std::mt19937 gen{12};
    const std::size_t num_qubits = 4;
    DenseRegister<> reg(num_qubits);
    State state = zeroState(num_qubits);
    applyRandomRotations(reg, state, 30, gen);
    for (std::size_t check = 0; check < 50; check++) {
        const auto pauli = randomPauli(num_qubits, gen);
        INFO("Pauli " << pauli.toString());
        CHECK(std::abs(reg.expectation(toDense(pauli)) -
                       expectation(state, pauli)) < 1e-12);
    }

    // On |0...0>, Z_0 stabilizes, -Z_0 does not, and X_0 is far from it.
    const DenseRegister<> zero(num_qubits);
    CHECK(zero.residual(toDense(PauliWord::fromString("ZIII"))) == 0.0);
    CHECK(zero.residual(toDense(PauliWord::fromString("-ZIII"))) == 1.0);
    CHECK(zero.residual(toDense(PauliWord::fromString("XIII"))) > 0.5);

    // The residual is linear in a small rotation away from a stabilizer.
    DenseRegister<> tilted(1);
    tilted.rotate(toDense(PauliWord::fromString("X")), 1e-6);
    const double residual =
        tilted.residual(toDense(PauliWord::fromString("Z")));
    CHECK(std::abs(residual - std::sin(1e-6)) < 1e-12);
}

TEST_CASE("DenseRegister projects with applyLinear", "[DenseRegister]") {
    std::mt19937 gen{13};
    const std::size_t num_qubits = 3;
    DenseRegister<> reg(num_qubits);
    State state = zeroState(num_qubits);
    applyRandomRotations(reg, state, 20, gen);
    const auto pauli = PauliWord::fromString("XZY");
    const double p0 = (1 + reg.expectation(toDense(pauli))) / 2;
    const double scale = 1 / (2 * std::sqrt(p0));
    reg.applyLinear(toDense(pauli), {scale, 0.0}, {scale, 0.0});
    requireSameState(reg, project(state, pauli, false));
}

TEST_CASE("DenseRegister adds and drops qubits", "[DenseRegister]") {
    std::mt19937 gen{14};
    DenseRegister<> reg(2);
    State state = zeroState(2);
    applyRandomRotations(reg, state, 10, gen);

    // A new qubit is the highest bit, in |0>.
    reg.addQubit();
    State grown(8, 0.0);
    for (std::size_t k = 0; k < 4; k++) {
        grown[k] = state[k];
    }
    requireSameState(reg, grown);

    // Entangle the new qubit, then undo it, and drop it.
    const auto pauli = PauliWord::fromString("XIY");
    reg.rotate(toDense(pauli), 0.4);
    reg.rotate(toDense(pauli), -0.4);
    CHECK(reg.dropQubit(2) < 1e-24);
    requireSameState(reg, state);

    // Dropping a middle qubit moves the higher ones down.
    DenseRegister<> wide(3);
    State narrow = zeroState(2);
    for (std::size_t i = 0; i < 10; i++) {
        const auto pauli = randomPauli(2, gen);
        auto spread = PauliWord(3);
        spread.set(0, pauli.get(0));
        spread.set(2, pauli.get(1));
        spread.setLogI(pauli.logI());
        wide.rotate(toDense(spread), 0.1 * static_cast<double>(i + 1));
        narrow = rotateBy(narrow, pauli, 0.1 * static_cast<double>(i + 1));
    }
    DenseRegister<> flipped = wide.clone();
    flipped.rotateQuarterTurns(toDense(PauliWord::fromString("IXI")), 2);
    CHECK(flipped.dropQubit(1) > 0.99);
    CHECK(wide.dropQubit(1) < 1e-24);
    requireSameState(wide, narrow);
}

TEST_CASE("DenseRegister rejects words outside it", "[DenseRegister]") {
    DenseRegister<> reg(2);
    REQUIRE_THROWS_WITH(reg.expectation({.x = 4}),
                        ContainsSubstring("outside the dense register"));
}
