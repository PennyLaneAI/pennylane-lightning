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
 * @file TableauEngine.hpp
 * The tableau engine interface: the seam between the planner and the code
 * that holds a stabilizer state.
 *
 * Engines take planned op blocks. A block is unconditional and belongs to one
 * branch: engines never see a predicate, an outcome dependence or a result id.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "PauliWord.hpp"

namespace Pennylane::LightningCliffish::Tableau {

using Pauli::PauliWord;

/**
 * @brief One planned primitive on the engine's qubits.
 */
struct EngineOp {
    enum class Kind : std::uint8_t {
        /// exp(-i quarter_turns pi/4 pauli), quarter_turns in {1, 2, 3}.
        Rotation,
        /// Projective measurement of pauli; outcome 1 is eigenvalue -1.
        Measurement,
    };

    Kind kind;
    /// Hermitian Pauli word on all of the engine's qubits.
    PauliWord pauli;
    std::uint8_t quarter_turns{0};
    /// Measurement only: the outcome to project onto, or random when empty.
    std::optional<bool> forced_outcome{std::nullopt};

    static auto rotation(PauliWord pauli, std::uint8_t quarter_turns)
        -> EngineOp {
        return {Kind::Rotation, std::move(pauli), quarter_turns, std::nullopt};
    }

    static auto measurement(PauliWord pauli,
                            std::optional<bool> forced_outcome = std::nullopt)
        -> EngineOp {
        return {Kind::Measurement, std::move(pauli), 0, forced_outcome};
    }
};

/// Unconditional ops of one branch, applied in order.
using EngineBlock = std::vector<EngineOp>;

/**
 * @brief What an engine supports. A missing capability is rejected at plan
 * time.
 */
struct Capabilities {
    /// Fork and merge of outcome branches.
    bool fork_merge{false};
    /// Coupling to a dense register for non-Clifford rotations.
    bool dense_register{false};
};

/**
 * @brief A stabilizer state on a growing set of qubits.
 */
class TableauEngine {
  public:
    virtual ~TableauEngine() = default;

    [[nodiscard]] virtual auto name() const -> std::string = 0;
    [[nodiscard]] virtual auto capabilities() const -> Capabilities = 0;
    [[nodiscard]] virtual auto numQubits() const -> std::size_t = 0;

    /// Append `count` qubits in the state |0>.
    virtual void addQubits(std::size_t count) = 0;

    /**
     * @brief Apply a block in order.
     *
     * @param block Planned ops on numQubits() qubits.
     * @param gen Source of random measurement outcomes.
     * @return The outcome of each measurement in the block, in order.
     */
    virtual auto applyBlock(const EngineBlock &block, std::mt19937 &gen)
        -> std::vector<bool> = 0;

    /// Expectation value of a Hermitian Pauli word: -1, 0 or +1.
    [[nodiscard]] virtual auto expectation(const PauliWord &pauli) const
        -> int = 0;

    /// Pull a Hermitian Pauli word back through the state's Clifford: C^† P C.
    [[nodiscard]] virtual auto pullBack(const PauliWord &pauli) const
        -> PauliWord = 0;
};

} // namespace Pennylane::LightningCliffish::Tableau
