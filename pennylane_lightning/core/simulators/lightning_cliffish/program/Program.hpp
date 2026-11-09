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
 * @file Program.hpp
 * The program as data: ops as the device received them, waiting to be
 * simulated.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Error.hpp"

namespace Pennylane::LightningCliffish::Program {

/// A qubit as the program names it. Ids are never reused.
using QubitId = std::size_t;

/// A measurement result as the program names it.
using ResultId = std::uint32_t;

/// The constant-true result: the predicate of every top-level op.
inline constexpr ResultId result_true = 0;

/// Marks an op that defines no result.
inline constexpr ResultId no_result = std::numeric_limits<ResultId>::max();

enum class OpKind : std::uint8_t {
    /// A named gate, expanded to Clifford rotations only when planned.
    Gate,
    /// exp(-i params[0]/2 P) for the Pauli word P on the wires.
    PauliRot,
    /// Measurement of the Pauli word on the wires.
    PauliMeasure,
    /// Computational-basis measurement of one wire.
    Measure,
    Allocate,
    Release,
    /// Return one wire to |0>.
    Reset,
    /// A result that holds params[0] != 0 in every shot.
    ResultConstant,
    /// operands[0] XOR operands[1].
    ResultXor,
    /// NOT operands[0].
    ResultNot,
    /// operands[1] where operands[0] holds, operands[2] elsewhere.
    ResultSelect,
    /// The program no longer reads operands[0].
    ResultRelease,
    /// Opens a span that implements exp(-i params[0]/2 P), for the Pauli word P
    /// on the wires; params[1] is the compiler's verdict.
    GadgetBegin,
    /// Closes the open gadget span.
    GadgetEnd,
};

/// The compiler's verdict on a gadget, as in Catalyst's CatalystGadgetVerdict.
enum class GadgetVerdict : std::uint8_t {
    Unchecked = 0,
    Verified = 1,
    Mismatch = 2,
};

/**
 * @brief One op, held as received.
 */
struct Op {
    OpKind kind;
    std::string name{};
    std::vector<double> params{};
    std::vector<QubitId> wires{};
    /// PauliRot and PauliMeasure only: one character per wire.
    std::string pauli_word{};
    bool inverse{false};
    std::vector<QubitId> controlled_wires{};
    std::vector<bool> controlled_values{};
    /// Measurements only: the outcome to keep, after the sign is applied.
    std::optional<bool> postselect{std::nullopt};
    /// PauliMeasure only: measure minus the Pauli word.
    bool negated{false};
    /// Result ops only: the results they read.
    std::vector<ResultId> operands{};
    /// The op applies only when this result is true.
    ResultId predicate{result_true};
    /// Measurements only: the result the op defines.
    ResultId result{no_result};
};

/**
 * @brief The ops received since the last flush, and the ids they use.
 *
 * Qubit and result ids persist across flushes; the ops do not.
 */
class Program {
  private:
    std::vector<Op> window_{};
    std::size_t num_allocated_{0};
    std::unordered_set<QubitId> live_{};
    ResultId next_result_{result_true + 1};

    void checkWires(const std::vector<QubitId> &wires) const {
        std::unordered_set<QubitId> seen;
        for (auto wire : wires) {
            PL_ABORT_IF_NOT(live_.contains(wire),
                            "Op acts on a qubit that is not allocated");
            PL_ABORT_IF_NOT(seen.insert(wire).second,
                            "Op acts on the same qubit twice");
        }
    }

    void checkResults(const std::vector<ResultId> &results) const {
        for (auto result : results) {
            PL_ABORT_IF_NOT(result < next_result_,
                            "Unknown measurement result");
        }
    }

  public:
    /**
     * @brief Allocate `count` qubits in |0>.
     *
     * @return The new qubit ids.
     */
    auto allocate(std::size_t count) -> std::vector<QubitId> {
        std::vector<QubitId> ids(count);
        for (auto &id : ids) {
            id = num_allocated_++;
            live_.insert(id);
        }
        window_.push_back(Op{.kind = OpKind::Allocate, .wires = ids});
        return ids;
    }

    /**
     * @brief Release a qubit. Its state is discarded.
     */
    void release(QubitId qubit) {
        checkWires({qubit});
        live_.erase(qubit);
        window_.push_back(Op{.kind = OpKind::Release, .wires = {qubit}});
    }

    /**
     * @brief Append a gate, rotation or reset.
     */
    void append(Op op) {
        PL_ABORT_IF_NOT(op.kind == OpKind::Gate ||
                            op.kind == OpKind::PauliRot ||
                            op.kind == OpKind::Reset,
                        "Use allocate, release or appendMeasurement for this "
                        "op");
        checkWires(op.wires);
        checkWires(op.controlled_wires);
        PL_ABORT_IF(op.kind == OpKind::PauliRot &&
                        op.pauli_word.size() != op.wires.size(),
                    "The Pauli word must have one character per wire");
        op.result = no_result;
        window_.push_back(std::move(op));
    }

    /**
     * @brief Append a measurement.
     *
     * @return The result the measurement defines.
     */
    auto appendMeasurement(Op op) -> ResultId {
        PL_ABORT_IF_NOT(op.kind == OpKind::Measure ||
                            op.kind == OpKind::PauliMeasure,
                        "Op is not a measurement");
        checkWires(op.wires);
        PL_ABORT_IF(op.kind == OpKind::Measure && op.wires.size() != 1,
                    "Measure acts on exactly one qubit");
        PL_ABORT_IF(op.kind == OpKind::PauliMeasure &&
                        op.pauli_word.size() != op.wires.size(),
                    "The Pauli word must have one character per wire");
        op.result = next_result_++;
        const ResultId result = op.result;
        window_.push_back(std::move(op));
        return result;
    }

    /**
     * @brief Append an op that combines results.
     *
     * @return The result the op defines.
     */
    auto appendResultOp(Op op) -> ResultId {
        const std::size_t arity = op.kind == OpKind::ResultConstant ? 0
                                  : op.kind == OpKind::ResultNot    ? 1
                                  : op.kind == OpKind::ResultXor    ? 2
                                  : op.kind == OpKind::ResultSelect ? 3
                                                                    : SIZE_MAX;
        PL_ABORT_IF(arity == SIZE_MAX, "Op is not a result op");
        PL_ABORT_IF_NOT(op.operands.size() == arity,
                        "Wrong number of results for a result op");
        checkResults(op.operands);
        PL_ABORT_IF(op.kind == OpKind::ResultConstant && op.params.size() != 1,
                    "A constant result takes its value as a parameter");
        op.result = next_result_++;
        const ResultId result = op.result;
        window_.push_back(std::move(op));
        return result;
    }

    /**
     * @brief Append a marker that defines nothing: a result release or a
     * gadget bracket.
     */
    void appendMarker(Op op) {
        PL_ABORT_IF_NOT(op.kind == OpKind::ResultRelease ||
                            op.kind == OpKind::GadgetBegin ||
                            op.kind == OpKind::GadgetEnd,
                        "Op is not a marker");
        checkResults(op.operands);
        if (op.kind == OpKind::GadgetBegin) {
            checkWires(op.wires);
            PL_ABORT_IF(op.pauli_word.size() != op.wires.size() ||
                            op.params.size() != 2,
                        "A gadget names its Pauli word, angle and verdict");
        }
        op.result = no_result;
        window_.push_back(std::move(op));
    }

    /// The ops received since the last flush.
    [[nodiscard]] auto window() const -> const std::vector<Op> & {
        return window_;
    }

    /// Hand the window to a flush and start a new one.
    auto takeWindow() -> std::vector<Op> { return std::exchange(window_, {}); }

    [[nodiscard]] auto isLive(QubitId qubit) const -> bool {
        return live_.contains(qubit);
    }

    /// Number of qubits currently allocated.
    [[nodiscard]] auto numQubits() const -> std::size_t { return live_.size(); }

    /// Whether `result` was defined by a measurement or a result op.
    [[nodiscard]] auto isResult(ResultId result) const -> bool {
        return result != result_true && result < next_result_;
    }
};

} // namespace Pennylane::LightningCliffish::Program
