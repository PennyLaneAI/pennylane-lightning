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
 * @file CircuitHelpers.hpp
 * Ops and random Clifford circuits for the planner tests.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "DenseReference.hpp"
#include "GateExpansions.hpp"
#include "Program.hpp"

namespace Pennylane::LightningCliffish::Testing {

using Program::Op;
using Program::OpKind;
using Program::QubitId;

inline auto gate(const std::string &name, std::vector<QubitId> wires,
                 std::vector<double> params = {}, bool inverse = false) -> Op {
    return Op{.kind = OpKind::Gate,
              .name = name,
              .params = std::move(params),
              .wires = std::move(wires),
              .inverse = inverse};
}

inline auto measure(QubitId wire, std::optional<bool> postselect = std::nullopt)
    -> Op {
    return Op{
        .kind = OpKind::Measure, .wires = {wire}, .postselect = postselect};
}

inline auto allWires(std::size_t num_qubits) -> std::vector<QubitId> {
    std::vector<QubitId> wires(num_qubits);
    for (std::size_t i = 0; i < num_qubits; i++) {
        wires[i] = i;
    }
    return wires;
}

/// Apply a gate op to the dense state through its rotation expansion.
inline void applyDense(State &state, const Op &op, std::size_t num_qubits) {
    for (const auto &rotation : Planner::expandGate(op)) {
        PauliWord pauli(num_qubits);
        for (std::size_t k = 0; k < op.wires.size(); k++) {
            pauli.set(op.wires[k], rotation.pauli[k]);
        }
        state = rotateBy(state, pauli, rotation.halfAngle());
    }
}

/// A random Clifford gate on random wires.
inline auto randomGate(std::size_t num_qubits, std::mt19937 &gen) -> Op {
    static const std::vector<std::tuple<std::string, std::size_t, bool>> gates{
        {"Hadamard", 1, false}, {"S", 1, false},      {"SX", 1, false},
        {"PauliX", 1, false},   {"PauliY", 1, false}, {"PauliZ", 1, false},
        {"RX", 1, true},        {"RY", 1, true},      {"PhaseShift", 1, true},
        {"CNOT", 2, false},     {"CY", 2, false},     {"CZ", 2, false},
        {"SWAP", 2, false},     {"ISWAP", 2, false},  {"IsingXX", 2, true},
    };
    static const std::vector<double> angles{M_PI / 2, M_PI, -M_PI / 2};

    std::vector<QubitId> wires = allWires(num_qubits);
    std::shuffle(wires.begin(), wires.end(), gen);
    while (true) {
        const auto &[name, num_wires, parametric] =
            gates[std::uniform_int_distribution<std::size_t>(0, gates.size() -
                                                                    1)(gen)];
        if (num_wires > num_qubits) {
            continue;
        }
        std::vector<double> params;
        if (parametric) {
            params.push_back(angles[std::uniform_int_distribution<std::size_t>(
                0, angles.size() - 1)(gen)]);
        }
        return gate(name,
                    {wires.begin(),
                     wires.begin() + static_cast<std::ptrdiff_t>(num_wires)},
                    params, std::bernoulli_distribution(0.3)(gen));
    }
}
} // namespace Pennylane::LightningCliffish::Testing
