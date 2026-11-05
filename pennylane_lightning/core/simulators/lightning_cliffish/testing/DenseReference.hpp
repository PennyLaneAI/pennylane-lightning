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
 * @file DenseReference.hpp
 * A small dense state-vector reference for testing Lightning-Cliffish.
 * Qubit q is bit q of a basis-state index.
 */

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "PauliWord.hpp"

namespace Pennylane::LightningCliffish::Testing {

using Pauli::PauliWord;
using State = std::vector<std::complex<double>>;

/// The state |0...0> on `num_qubits` qubits.
inline auto zeroState(std::size_t num_qubits) -> State {
    State state(std::size_t{1} << num_qubits, 0.0);
    state[0] = 1.0;
    return state;
}

/// P |state> for a Pauli word P with phase.
inline auto applyPauli(const State &state, const PauliWord &pauli) -> State {
    static const std::complex<double> phases[] = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    const std::size_t num_qubits = pauli.numQubits();
    State result(state.size(), 0.0);
    for (std::size_t index = 0; index < state.size(); index++) {
        std::complex<double> amplitude = state[index] * phases[pauli.logI()];
        std::size_t target = index;
        for (std::size_t q = 0; q < num_qubits; q++) {
            const bool bit = (index >> q) & 1U;
            switch (pauli.get(q)) {
            case 'X':
                target ^= std::size_t{1} << q;
                break;
            case 'Y':
                // Y|0> = i|1>, Y|1> = -i|0>
                target ^= std::size_t{1} << q;
                amplitude *= bit ? std::complex<double>{0, -1}
                                 : std::complex<double>{0, 1};
                break;
            case 'Z':
                amplitude *= bit ? -1.0 : 1.0;
                break;
            default:
                break;
            }
        }
        result[target] += amplitude;
    }
    return result;
}

/// exp(-i quarter_turns pi/4 P) |state>.
inline auto rotate(const State &state, const PauliWord &pauli,
                   unsigned quarter_turns) -> State {
    const double half_angle = quarter_turns * M_PI / 4;
    const State pauli_state = applyPauli(state, pauli);
    State result(state.size());
    for (std::size_t i = 0; i < state.size(); i++) {
        result[i] =
            std::cos(half_angle) * state[i] -
            std::complex<double>{0, 1} * std::sin(half_angle) * pauli_state[i];
    }
    return result;
}

/// <state| P |state> for a Hermitian Pauli word P.
inline auto expectation(const State &state, const PauliWord &pauli) -> double {
    const State pauli_state = applyPauli(state, pauli);
    std::complex<double> value = 0.0;
    for (std::size_t i = 0; i < state.size(); i++) {
        value += std::conj(state[i]) * pauli_state[i];
    }
    return value.real();
}

/// Project onto outcome `outcome` of measuring P, and normalize.
inline auto project(const State &state, const PauliWord &pauli, bool outcome)
    -> State {
    const State pauli_state = applyPauli(state, pauli);
    const double sign = outcome ? -1.0 : 1.0;
    State result(state.size());
    double norm = 0.0;
    for (std::size_t i = 0; i < state.size(); i++) {
        result[i] = (state[i] + sign * pauli_state[i]) / 2.0;
        norm += std::norm(result[i]);
    }
    for (auto &amplitude : result) {
        amplitude /= std::sqrt(norm);
    }
    return result;
}

/**
 * @brief Probability of each outcome index of measuring `wires`, with
 * wires[0] the most significant bit.
 */
inline auto probabilities(const State &state,
                          const std::vector<std::size_t> &wires)
    -> std::vector<double> {
    std::vector<double> result(std::size_t{1} << wires.size(), 0.0);
    for (std::size_t index = 0; index < state.size(); index++) {
        std::size_t outcome = 0;
        for (auto wire : wires) {
            outcome = (outcome << 1U) | ((index >> wire) & 1U);
        }
        result[outcome] += std::norm(state[index]);
    }
    return result;
}

/// sum_k coefficients[k] paulis[k] |state>.
inline auto applySum(const State &state,
                     const std::vector<double> &coefficients,
                     const std::vector<PauliWord> &paulis) -> State {
    State result(state.size(), 0.0);
    for (std::size_t k = 0; k < paulis.size(); k++) {
        const State term = applyPauli(state, paulis[k]);
        for (std::size_t i = 0; i < state.size(); i++) {
            result[i] += coefficients[k] * term[i];
        }
    }
    return result;
}

/// A uniformly random Hermitian Pauli word, not the identity.
inline auto randomPauli(std::size_t num_qubits, std::mt19937 &gen)
    -> PauliWord {
    static const std::string paulis{"IXYZ"};
    std::uniform_int_distribution<std::size_t> pick(0, 3);
    std::string word;
    do {
        word.clear();
        for (std::size_t q = 0; q < num_qubits; q++) {
            word += paulis[pick(gen)];
        }
    } while (word.find_first_not_of('I') == std::string::npos);
    auto pauli = PauliWord::fromString(word);
    pauli.setLogI(std::bernoulli_distribution(0.5)(gen) ? 2 : 0);
    return pauli;
}

/// The Pauli word with `pauli` on `qubit` and I elsewhere.
inline auto singleQubitPauli(std::size_t num_qubits, std::size_t qubit,
                             char pauli) -> PauliWord {
    PauliWord word(num_qubits);
    word.set(qubit, pauli);
    return word;
}

} // namespace Pennylane::LightningCliffish::Testing
