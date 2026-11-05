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
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "GateExpansions.hpp"

namespace {
using namespace Pennylane::LightningCliffish;
using namespace Pennylane::LightningCliffish::Planner;
using Catch::Matchers::ContainsSubstring;
using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;

const Complex imag{0, 1};

auto identity(std::size_t dim) -> Matrix {
    Matrix m(dim, std::vector<Complex>(dim, 0.0));
    for (std::size_t i = 0; i < dim; i++) {
        m[i][i] = 1.0;
    }
    return m;
}

auto multiply(const Matrix &a, const Matrix &b) -> Matrix {
    const std::size_t dim = a.size();
    Matrix m(dim, std::vector<Complex>(dim, 0.0));
    for (std::size_t i = 0; i < dim; i++) {
        for (std::size_t k = 0; k < dim; k++) {
            for (std::size_t j = 0; j < dim; j++) {
                m[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return m;
}

auto kron(const Matrix &a, const Matrix &b) -> Matrix {
    const std::size_t da = a.size();
    const std::size_t db = b.size();
    Matrix m(da * db, std::vector<Complex>(da * db, 0.0));
    for (std::size_t i = 0; i < da; i++) {
        for (std::size_t j = 0; j < da; j++) {
            for (std::size_t k = 0; k < db; k++) {
                for (std::size_t l = 0; l < db; l++) {
                    m[i * db + k][j * db + l] = a[i][j] * b[k][l];
                }
            }
        }
    }
    return m;
}

/// The matrix of a Pauli word; character 0 is the most significant qubit,
/// as in PennyLane.
auto pauliMatrix(const std::string &word) -> Matrix {
    Matrix m = identity(1);
    for (char pauli : word) {
        Matrix single;
        switch (pauli) {
        case 'X':
            single = {{0, 1}, {1, 0}};
            break;
        case 'Y':
            single = {{0, -imag}, {imag, 0}};
            break;
        case 'Z':
            single = {{1, 0}, {0, -1}};
            break;
        default:
            single = identity(2);
        }
        m = kron(m, single);
    }
    return m;
}

/// The unitary of the rotations, applied in order.
auto rotationsMatrix(const std::vector<LocalRotation> &rotations,
                     std::size_t num_wires) -> Matrix {
    Matrix u = identity(std::size_t{1} << num_wires);
    for (const auto &rotation : rotations) {
        const double half_angle = rotation.quarter_turns * M_PI / 4;
        const Matrix p = pauliMatrix(rotation.pauli);
        Matrix r = identity(p.size());
        for (std::size_t i = 0; i < p.size(); i++) {
            for (std::size_t j = 0; j < p.size(); j++) {
                r[i][j] = std::cos(half_angle) * r[i][j] -
                          imag * std::sin(half_angle) * p[i][j];
            }
        }
        u = multiply(r, u);
    }
    return u;
}

/// Whether a and b are equal up to a global phase.
auto equalUpToPhase(const Matrix &a, const Matrix &b) -> bool {
    std::size_t row = 0;
    std::size_t col = 0;
    for (std::size_t i = 0; i < a.size(); i++) {
        for (std::size_t j = 0; j < a.size(); j++) {
            if (std::abs(a[i][j]) > std::abs(a[row][col])) {
                row = i;
                col = j;
            }
        }
    }
    if (std::abs(b[row][col]) < 1e-12) {
        return false;
    }
    const Complex phase = a[row][col] / b[row][col];
    for (std::size_t i = 0; i < a.size(); i++) {
        for (std::size_t j = 0; j < a.size(); j++) {
            if (std::abs(a[i][j] - phase * b[i][j]) > 1e-9) {
                return false;
            }
        }
    }
    return std::abs(std::abs(phase) - 1.0) < 1e-9;
}

auto gate(const std::string &name, std::size_t num_wires,
          std::vector<double> params = {}, bool inverse = false)
    -> Program::Op {
    std::vector<Program::QubitId> wires(num_wires);
    for (std::size_t i = 0; i < num_wires; i++) {
        wires[i] = i;
    }
    return Program::Op{.kind = Program::OpKind::Gate,
                       .name = name,
                       .params = std::move(params),
                       .wires = wires,
                       .inverse = inverse};
}
} // namespace

TEST_CASE("cliffordQuarterTurns snaps only near non-identity Clifford angles",
          "[GateExpansions]") {
    CHECK(cliffordQuarterTurns(0.0) == 0);
    CHECK(cliffordQuarterTurns(M_PI / 2) == 1);
    CHECK(cliffordQuarterTurns(M_PI) == 2);
    CHECK(cliffordQuarterTurns(-M_PI / 2) == 3);
    CHECK(cliffordQuarterTurns(3 * M_PI / 2) == 3);
    CHECK(cliffordQuarterTurns(2 * M_PI) == 0);
    CHECK(cliffordQuarterTurns(M_PI / 2 + 1e-13) == 1);
    CHECK_FALSE(cliffordQuarterTurns(M_PI / 2 + 1e-9).has_value());
    CHECK_FALSE(cliffordQuarterTurns(M_PI / 4).has_value());
    // Near-identity rotations are never snapped to the identity.
    CHECK_FALSE(cliffordQuarterTurns(1e-15).has_value());
    CHECK_FALSE(cliffordQuarterTurns(2 * M_PI + 1e-13).has_value());
}

TEST_CASE("Fixed gate expansions match PennyLane's matrices",
          "[GateExpansions]") {
    const double r = 1.0 / std::sqrt(2.0);
    const std::vector<std::pair<std::string, Matrix>> gates{
        {"Identity", identity(2)},
        {"PauliX", {{0, 1}, {1, 0}}},
        {"PauliY", {{0, -imag}, {imag, 0}}},
        {"PauliZ", {{1, 0}, {0, -1}}},
        {"Hadamard", {{r, r}, {r, -r}}},
        {"S", {{1, 0}, {0, imag}}},
        {"SX",
         {{(1.0 + imag) / 2.0, (1.0 - imag) / 2.0},
          {(1.0 - imag) / 2.0, (1.0 + imag) / 2.0}}},
        {"CNOT", {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}},
        {"CY", {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, -imag}, {0, 0, imag, 0}}},
        {"CZ", {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}}},
        {"SWAP", {{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}},
        {"ISWAP",
         {{1, 0, 0, 0}, {0, 0, imag, 0}, {0, imag, 0, 0}, {0, 0, 0, 1}}},
    };
    for (const auto &[name, matrix] : gates) {
        INFO("Gate " << name);
        const std::size_t num_wires = matrix.size() == 2 ? 1 : 2;
        const auto rotations = expandGate(gate(name, num_wires));
        CHECK(equalUpToPhase(matrix, rotationsMatrix(rotations, num_wires)));

        // The adjoint expansion inverts the gate.
        const auto inverse = expandGate(gate(name, num_wires, {}, true));
        auto all = rotations;
        all.insert(all.end(), inverse.begin(), inverse.end());
        CHECK(equalUpToPhase(identity(matrix.size()),
                             rotationsMatrix(all, num_wires)));
    }
}

TEST_CASE("Parametric gates expand at Clifford angles", "[GateExpansions]") {
    for (double angle : {M_PI / 2, M_PI, -M_PI / 2}) {
        INFO("Angle " << angle);
        const double c = std::cos(angle / 2);
        const double s = std::sin(angle / 2);
        const Matrix rx{{c, -imag * s}, {-imag * s, c}};
        const Matrix ry{{c, -s}, {s, c}};
        const Matrix rz{{std::exp(-imag * angle / 2.0), 0},
                        {0, std::exp(imag * angle / 2.0)}};
        const Matrix phase_shift{{1, 0}, {0, std::exp(imag * angle)}};

        CHECK(equalUpToPhase(
            rx, rotationsMatrix(expandGate(gate("RX", 1, {angle})), 1)));
        CHECK(equalUpToPhase(
            ry, rotationsMatrix(expandGate(gate("RY", 1, {angle})), 1)));
        CHECK(equalUpToPhase(
            rz, rotationsMatrix(expandGate(gate("RZ", 1, {angle})), 1)));
        CHECK(equalUpToPhase(
            phase_shift,
            rotationsMatrix(expandGate(gate("PhaseShift", 1, {angle})), 1)));
        const Complex minus = std::exp(-imag * angle / 2.0);
        const Complex plus = std::exp(imag * angle / 2.0);
        const Matrix ising_zz{{minus, 0, 0, 0},
                              {0, plus, 0, 0},
                              {0, 0, plus, 0},
                              {0, 0, 0, minus}};
        CHECK(equalUpToPhase(
            ising_zz,
            rotationsMatrix(expandGate(gate("IsingZZ", 2, {angle})), 2)));

        const Matrix ising_xx{{c, 0, 0, -imag * s},
                              {0, c, -imag * s, 0},
                              {0, -imag * s, c, 0},
                              {-imag * s, 0, 0, c}};
        CHECK(equalUpToPhase(
            ising_xx,
            rotationsMatrix(expandGate(gate("IsingXX", 2, {angle})), 2)));
    }

    const auto multi_rz = expandGate(gate("MultiRZ", 3, {M_PI}));
    CHECK(multi_rz == std::vector<LocalRotation>{{"ZZZ", 2}});

    const auto pauli_rot =
        expandGate(Program::Op{.kind = Program::OpKind::PauliRot,
                               .params = {-M_PI / 2},
                               .wires = {0, 1},
                               .pauli_word = "XY"});
    CHECK(pauli_rot == std::vector<LocalRotation>{{"XY", 3}});

    // An exactly-zero angle is the identity.
    CHECK(expandGate(gate("RX", 1, {0.0})).empty());
}

TEST_CASE("Gate expansion fails loudly on unsupported ops",
          "[GateExpansions]") {
    REQUIRE_THROWS_WITH(expandGate(gate("T", 1)),
                        ContainsSubstring("cannot apply 'T'") &&
                            ContainsSubstring("lightning.qubit"));
    REQUIRE_THROWS_WITH(expandGate(gate("RZ", 1, {M_PI / 4})),
                        ContainsSubstring("not a Clifford operation"));
    REQUIRE_THROWS_WITH(expandGate(gate("RZ", 1, {1e-15})),
                        ContainsSubstring("not a Clifford operation"));
    REQUIRE_THROWS_WITH(expandGate(gate("QubitUnitary", 1)),
                        ContainsSubstring("does not support the operation"));
    REQUIRE_THROWS_WITH(expandGate(gate("CNOT", 1)),
                        ContainsSubstring("acts on 2 wires"));
    REQUIRE_THROWS_WITH(expandGate(gate("RX", 1)),
                        ContainsSubstring("exactly one parameter"));

    auto controlled = gate("PauliX", 1);
    controlled.controlled_wires = {1};
    controlled.controlled_values = {true};
    REQUIRE_THROWS_WITH(expandGate(controlled),
                        ContainsSubstring("controlled operations"));
}
