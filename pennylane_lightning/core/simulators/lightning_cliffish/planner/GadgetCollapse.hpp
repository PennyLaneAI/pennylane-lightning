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
 * @file GadgetCollapse.hpp
 * Gadget collapse (broly ADR 0012): apply a bracketed gadget that merges
 * exactly to one rotation as that rotation.
 */

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "GateExpansions.hpp"
#include "Program.hpp"

namespace Pennylane::LightningCliffish::Planner {

/// What to do with a gadget, by the device option `gadget_collapse`.
enum class GadgetCollapse : std::uint8_t {
    /// Apply the rotation the device's own check measured. Fail if it is not
    /// the gadget's source rotation, or disagrees with the compiler's verdict.
    Measured,
    /// Apply the gadget's source rotation, and record every substitution.
    Source,
    /// Run gadgets as given.
    Off,
};

inline auto parseGadgetCollapse(std::string_view name) -> GadgetCollapse {
    if (name == "measured") {
        return GadgetCollapse::Measured;
    }
    if (name == "source") {
        return GadgetCollapse::Source;
    }
    if (name == "off") {
        return GadgetCollapse::Off;
    }
    PL_ABORT("Unknown gadget_collapse mode '" + std::string{name} +
             "': expected measured, source or off");
}

/// What gadget collapse did.
struct GadgetReport {
    std::size_t gadgets{0};
    std::size_t collapsed{0};
    /// Distinct gadget shapes checked.
    std::size_t shapes{0};
    /// Source-mode gadgets whose check did not find their source rotation.
    std::size_t substitutions{0};
    /// One line per gadget that runs as given, with the reason.
    std::vector<std::string> uncollapsed{};
};

namespace Detail {

using Complex = std::complex<double>;

/// The tolerance on how far a branch may be from the merged unitary.
inline constexpr double collapse_tolerance = 1e-12;

/// The largest check, in qubits: twice the data qubits plus auxiliaries.
inline constexpr std::size_t max_check_qubits = 20;

/// A gadget op on the check's local qubits and results.
struct CheckOp {
    enum class Kind : std::uint8_t {
        Rotation,
        Measure,
        Allocate,
        Release,
        Result
    };
    Kind kind;
    /// Rotation and Measure: (local qubit, Pauli) pairs.
    std::vector<std::pair<std::size_t, char>> paulis{};
    /// Rotation: exp(-i half_angle P).
    double half_angle{0.0};
    bool negated{false};
    /// Allocate and Release: the local qubit.
    std::size_t qubit{0};
    /// Result: what it computes.
    Program::OpKind result_kind{Program::OpKind::ResultConstant};
    bool constant{false};
    std::vector<std::size_t> operands{};
    /// Measure and Result: the local result defined.
    std::size_t result{0};
    /// The local result the op is conditioned on, if any.
    std::optional<std::size_t> predicate{};
};

/// A state vector on a few qubits; qubit i is bit i of the index.
class CheckState {
  public:
    explicit CheckState(std::size_t num_qubits)
        : num_qubits_{num_qubits}, amplitudes_(std::size_t{1} << num_qubits) {}

    [[nodiscard]] auto numQubits() const -> std::size_t { return num_qubits_; }
    [[nodiscard]] auto amplitudes() -> std::vector<Complex> & {
        return amplitudes_;
    }

    /// psi <- a psi + b P psi, with P on the given qubits.
    void applyLinear(const std::vector<std::pair<std::size_t, char>> &paulis,
                     Complex a, Complex b) {
        std::size_t flip = 0;
        for (auto [qubit, pauli] : paulis) {
            if (pauli == 'X' || pauli == 'Y') {
                flip |= std::size_t{1} << qubit;
            }
        }
        std::vector<Complex> result(amplitudes_.size());
        for (std::size_t k = 0; k < amplitudes_.size(); k++) {
            Complex phase{1.0, 0.0};
            for (auto [qubit, pauli] : paulis) {
                const bool bit = ((k >> qubit) & 1U) != 0;
                if (pauli == 'Z') {
                    phase *= bit ? -1.0 : 1.0;
                } else if (pauli == 'Y') {
                    phase *= bit ? Complex{0.0, -1.0} : Complex{0.0, 1.0};
                }
            }
            result[k ^ flip] += b * phase * amplitudes_[k];
        }
        for (std::size_t k = 0; k < amplitudes_.size(); k++) {
            amplitudes_[k] = a * amplitudes_[k] + result[k];
        }
    }

    [[nodiscard]] auto norm() const -> double {
        double total = 0.0;
        for (const auto &amplitude : amplitudes_) {
            total += std::norm(amplitude);
        }
        return total;
    }

    /// Append a qubit in |0> as the highest qubit.
    auto addQubit() -> std::size_t {
        amplitudes_.resize(2 * amplitudes_.size());
        return num_qubits_++;
    }

    /// Remove `qubit` if it is unentangled with the rest; higher qubits move
    /// down by one.
    auto removeQubit(std::size_t qubit) -> bool {
        const std::size_t bit = std::size_t{1} << qubit;
        Complex rho[2][2] = {};
        for (std::size_t k = 0; k < amplitudes_.size(); k++) {
            if ((k & bit) != 0) {
                continue;
            }
            const Complex v[2] = {amplitudes_[k], amplitudes_[k | bit]};
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 2; b++) {
                    rho[a][b] += v[a] * std::conj(v[b]);
                }
            }
        }
        const double trace = (rho[0][0] + rho[1][1]).real();
        const double det =
            std::abs(rho[0][0] * rho[1][1] - rho[0][1] * rho[1][0]);
        if (det > collapse_tolerance * trace * trace) {
            return false;
        }
        const int column = rho[0][0].real() >= rho[1][1].real() ? 0 : 1;
        Complex phi[2] = {rho[0][column], rho[1][column]};
        const double scale = std::sqrt(std::norm(phi[0]) + std::norm(phi[1]));
        phi[0] /= scale;
        phi[1] /= scale;
        std::vector<Complex> next(amplitudes_.size() / 2);
        for (std::size_t t = 0; t < next.size(); t++) {
            const std::size_t low = t & (bit - 1);
            const std::size_t k = ((t ^ low) << 1U) | low;
            next[t] = std::conj(phi[0]) * amplitudes_[k] +
                      std::conj(phi[1]) * amplitudes_[k | bit];
        }
        amplitudes_ = std::move(next);
        num_qubits_--;
        return true;
    }

  private:
    std::size_t num_qubits_;
    std::vector<Complex> amplitudes_;
};

/// The outcome of checking one gadget shape.
struct ShapeCheck {
    bool merges{false};
    std::string reason{};
    /// When it merges: the d x d unitary U[i][j], up to a global phase.
    std::vector<Complex> unitary{};
};

/// The unitary of one branch, or a reason it has none.
struct BranchRun {
    bool needs_outcome{false};
    std::string failure{};
    CheckState state{0};
    std::vector<std::size_t> slots{};
};

/**
 * @brief Run every outcome branch of a gadget on a Choi state of its
 * `num_data` data qubits, and check that all branches merge to one unitary.
 */
inline auto checkShape(std::size_t num_data, std::size_t num_local_qubits,
                       std::size_t num_results, const std::vector<CheckOp> &ops)
    -> ShapeCheck {
    if (2 * num_data > max_check_qubits) {
        return {.reason = "the gadget has too many data qubits to check"};
    }
    const std::size_t d = std::size_t{1} << num_data;

    auto runBranch = [&](const std::vector<bool> &outcomes) -> BranchRun {
        BranchRun run{.state = CheckState(2 * num_data)};
        for (std::size_t i = 0; i < d; i++) {
            run.state.amplitudes()[i | (i << num_data)] =
                1.0 / std::sqrt(static_cast<double>(d));
        }
        // Local qubit -> state qubit; data qubit i is state qubit i.
        constexpr std::size_t unset = SIZE_MAX;
        run.slots.assign(num_local_qubits, unset);
        for (std::size_t i = 0; i < num_data; i++) {
            run.slots[i] = i;
        }
        std::vector<bool> bits(num_results, false);
        std::size_t measured = 0;
        auto mapped = [&](const CheckOp &op) {
            std::vector<std::pair<std::size_t, char>> paulis;
            for (auto [qubit, pauli] : op.paulis) {
                paulis.emplace_back(run.slots[qubit], pauli);
            }
            return paulis;
        };
        for (const auto &op : ops) {
            const bool active =
                !op.predicate.has_value() || bits[*op.predicate];
            switch (op.kind) {
            case CheckOp::Kind::Rotation:
                if (active) {
                    run.state.applyLinear(
                        mapped(op), std::cos(op.half_angle),
                        Complex{0.0, -std::sin(op.half_angle)});
                }
                break;
            case CheckOp::Kind::Measure: {
                if (!active) {
                    bits[op.result] = false;
                    break;
                }
                if (measured == outcomes.size()) {
                    run.needs_outcome = true;
                    return run;
                }
                const bool outcome = outcomes[measured++];
                // Outcome 0 is the +1 eigenvalue of the signed word.
                const double sign = (op.negated != outcome) ? -0.5 : 0.5;
                run.state.applyLinear(mapped(op), 0.5, sign);
                bits[op.result] = outcome;
                break;
            }
            case CheckOp::Kind::Allocate:
                if (run.state.numQubits() >= max_check_qubits) {
                    run.failure = "the gadget is too large to check";
                    return run;
                }
                run.slots[op.qubit] = run.state.addQubit();
                break;
            case CheckOp::Kind::Release: {
                const std::size_t slot = run.slots[op.qubit];
                if (!run.state.removeQubit(slot)) {
                    run.failure = "entangled-release: an auxiliary qubit is "
                                  "released while entangled";
                    return run;
                }
                run.slots[op.qubit] = unset;
                for (auto &other : run.slots) {
                    if (other != unset && other > slot) {
                        other--;
                    }
                }
                break;
            }
            case CheckOp::Kind::Result: {
                using Program::OpKind;
                const auto &in = op.operands;
                bool value = op.constant;
                if (op.result_kind == OpKind::ResultXor) {
                    value = bits[in[0]] != bits[in[1]];
                } else if (op.result_kind == OpKind::ResultNot) {
                    value = !bits[in[0]];
                } else if (op.result_kind == OpKind::ResultSelect) {
                    value = bits[in[0]] ? bits[in[1]] : bits[in[2]];
                }
                bits[op.result] = value;
                break;
            }
            }
        }
        if (run.state.numQubits() != 2 * num_data) {
            run.failure = "an auxiliary qubit is not released in the gadget";
        }
        return run;
    };

    ShapeCheck check;
    std::vector<std::vector<bool>> pending{{}};
    double total_weight = 0.0;
    while (!pending.empty()) {
        std::vector<bool> outcomes = std::move(pending.back());
        pending.pop_back();
        BranchRun run = runBranch(outcomes);
        if (!run.failure.empty()) {
            return {.reason = run.failure};
        }
        if (run.needs_outcome) {
            for (bool outcome : {false, true}) {
                auto longer = outcomes;
                longer.push_back(outcome);
                pending.push_back(std::move(longer));
            }
            continue;
        }
        const double weight = run.state.norm();
        if (weight < collapse_tolerance * collapse_tolerance) {
            continue;
        }
        total_weight += weight;
        // U[i][j] = sqrt(d / weight) <i|_data <j|_ref |psi>
        std::vector<Complex> unitary(d * d);
        for (std::size_t i = 0; i < d; i++) {
            for (std::size_t j = 0; j < d; j++) {
                std::size_t index = j << num_data;
                for (std::size_t q = 0; q < num_data; q++) {
                    index |= ((i >> q) & 1U) << run.slots[q];
                }
                unitary[i * d + j] = run.state.amplitudes()[index] *
                                     std::sqrt(static_cast<double>(d) / weight);
            }
        }
        if (check.unitary.empty()) {
            check.unitary = std::move(unitary);
            continue;
        }
        // Every branch must be the first one, up to a global phase.
        Complex overlap{};
        for (std::size_t e = 0; e < d * d; e++) {
            overlap += std::conj(check.unitary[e]) * unitary[e];
        }
        const Complex phase =
            std::abs(overlap) > 0 ? overlap / std::abs(overlap) : Complex{1.0};
        double distance = 0.0;
        for (std::size_t e = 0; e < d * d; e++) {
            distance += std::norm(unitary[e] - phase * check.unitary[e]);
        }
        if (std::sqrt(distance / static_cast<double>(d)) > collapse_tolerance) {
            return {.reason = "the outcome branches do not merge to one "
                              "unitary"};
        }
    }
    if (check.unitary.empty() || std::abs(total_weight - 1.0) > 1e-9) {
        return {.reason = "the branch weights do not sum to one"};
    }
    check.merges = true;
    return check;
}

/// exp(-i theta/2 P) as a d x d matrix, for P on `num_data` qubits.
inline auto rotationMatrix(const std::string &word, double theta)
    -> std::vector<Complex> {
    const std::size_t num_data = word.size();
    const std::size_t d = std::size_t{1} << num_data;
    std::vector<std::pair<std::size_t, char>> paulis;
    for (std::size_t q = 0; q < num_data; q++) {
        if (word[q] != 'I') {
            paulis.emplace_back(q, word[q]);
        }
    }
    std::vector<Complex> matrix(d * d);
    for (std::size_t j = 0; j < d; j++) {
        CheckState column(num_data);
        column.amplitudes()[0] = 0.0;
        column.amplitudes()[j] = 1.0;
        column.applyLinear(paulis, std::cos(theta / 2),
                           Complex{0.0, -std::sin(theta / 2)});
        for (std::size_t i = 0; i < d; i++) {
            matrix[i * d + j] = column.amplitudes()[i];
        }
    }
    return matrix;
}

/// Distance between two unitaries, minimised over a global phase.
inline auto unitaryDistance(const std::vector<Complex> &a,
                            const std::vector<Complex> &b) -> double {
    Complex overlap{};
    for (std::size_t e = 0; e < a.size(); e++) {
        overlap += std::conj(a[e]) * b[e];
    }
    const Complex phase =
        std::abs(overlap) > 0 ? overlap / std::abs(overlap) : Complex{1.0};
    double distance = 0.0;
    for (std::size_t e = 0; e < a.size(); e++) {
        distance += std::norm(b[e] - phase * a[e]);
    }
    const auto dim = static_cast<double>(
        std::llround(std::sqrt(static_cast<double>(a.size()))));
    return std::sqrt(distance / dim);
}

} // namespace Detail

/**
 * @brief Gadget collapse over a window of the program as data.
 *
 * A gadget is a bracketed span. It collapses when every outcome branch, run on
 * a Choi state of its data qubits, merges exactly to one rotation, with every
 * auxiliary qubit released unentangled, and no outcome of the span is read
 * after it. The check runs once per gadget shape.
 *
 * In a PBC gadget, every op acts on the data qubits as the identity or as the
 * source word P. Such a gadget is checked on one data qubit, with P mapped to
 * Z, whatever the weight of P.
 */
class GadgetCollapser {
  public:
    explicit GadgetCollapser(GadgetCollapse mode = GadgetCollapse::Measured)
        : mode_{mode} {}

    [[nodiscard]] auto mode() const -> GadgetCollapse { return mode_; }
    [[nodiscard]] auto report() const -> const GadgetReport & {
        return report_;
    }

    /**
     * @brief Replace each collapsible gadget of `window` with one PauliRot,
     * and drop the gadget brackets.
     */
    auto run(std::vector<Program::Op> window) -> std::vector<Program::Op> {
        using Program::OpKind;
        std::vector<Program::Op> result;
        result.reserve(window.size());
        for (std::size_t i = 0; i < window.size(); i++) {
            if (window[i].kind == OpKind::GadgetEnd) {
                PL_ABORT("A gadget bracket closes without opening");
            }
            if (window[i].kind != OpKind::GadgetBegin) {
                result.push_back(std::move(window[i]));
                continue;
            }
            std::size_t end = i + 1;
            while (end < window.size() &&
                   window[end].kind != OpKind::GadgetEnd) {
                PL_ABORT_IF(window[end].kind == OpKind::GadgetBegin,
                            "Gadget brackets do not nest");
                end++;
            }
            PL_ABORT_IF(end == window.size(),
                        "A gadget is still open at a flush");
            report_.gadgets++;
            const Program::Op &begin = window[i];
            std::vector<Program::Op> span(
                std::make_move_iterator(window.begin() + i + 1),
                std::make_move_iterator(window.begin() + end));
            auto replacement = collapse(begin, span, window, end + 1);
            if (replacement.has_value()) {
                report_.collapsed++;
                if (replacement->has_value()) {
                    result.push_back(std::move(**replacement));
                }
            } else {
                for (auto &op : span) {
                    result.push_back(std::move(op));
                }
            }
            i = end;
        }
        return result;
    }

  private:
    GadgetCollapse mode_;
    GadgetReport report_{};
    std::unordered_map<std::string, Detail::ShapeCheck> shapes_{};

    struct Shape {
        std::size_t num_data{0};
        std::size_t num_qubits{0};
        std::size_t num_results{0};
        std::vector<Detail::CheckOp> ops{};
        std::string key{};
        /// The data qubits, in the order of the check's data qubits.
        bool reduced{false};
    };

    static auto describe(const Program::Op &begin) -> std::string {
        std::ostringstream text;
        text << "exp(-i " << begin.params[0] << "/2 " << begin.pauli_word
             << ")";
        return text.str();
    }

    void uncollapsed(const Program::Op &begin, const std::string &reason) {
        report_.uncollapsed.push_back("gadget " + describe(begin) + ": " +
                                      reason);
    }

    /**
     * @brief The shape of a span, on local qubits and results, or a reason it
     * cannot be checked. With `reduce`, the data qubits become one qubit on
     * which the source word is Z.
     */
    static auto shapeOf(const Program::Op &begin,
                        const std::vector<Program::Op> &span, bool reduce)
        -> std::optional<Shape> {
        using Detail::CheckOp;
        using Program::OpKind;
        Shape shape;
        shape.reduced = reduce;
        shape.num_data = reduce ? 1 : begin.wires.size();
        std::unordered_map<Program::QubitId, std::size_t> data;
        for (std::size_t q = 0; q < begin.wires.size(); q++) {
            data[begin.wires[q]] = q;
        }
        std::unordered_map<Program::QubitId, std::size_t> aux;
        std::unordered_map<Program::ResultId, std::size_t> results;
        std::size_t next_qubit = shape.num_data;

        auto localResult =
            [&](Program::ResultId id) -> std::optional<std::size_t> {
            const auto it = results.find(id);
            if (it == results.end()) {
                return std::nullopt;
            }
            return it->second;
        };
        // The op's Pauli word on local qubits, or nothing when it does not
        // fit the shape.
        auto localPaulis = [&](const std::string &word,
                               const std::vector<Program::QubitId> &wires)
            -> std::optional<std::vector<std::pair<std::size_t, char>>> {
            std::vector<std::pair<std::size_t, char>> paulis;
            std::string on_data(begin.wires.size(), 'I');
            for (std::size_t k = 0; k < wires.size(); k++) {
                if (const auto it = data.find(wires[k]); it != data.end()) {
                    on_data[it->second] = word[k];
                    if (!reduce && word[k] != 'I') {
                        paulis.emplace_back(it->second, word[k]);
                    }
                } else if (const auto a = aux.find(wires[k]); a != aux.end()) {
                    if (word[k] != 'I') {
                        paulis.emplace_back(a->second, word[k]);
                    }
                } else {
                    return std::nullopt;
                }
            }
            if (reduce) {
                const bool identity =
                    on_data.find_first_not_of('I') == std::string::npos;
                if (!identity) {
                    if (on_data != begin.pauli_word) {
                        return std::nullopt;
                    }
                    paulis.emplace_back(0, 'Z');
                }
            }
            return paulis;
        };

        std::ostringstream key;
        key.precision(17);
        key << (reduce ? "R" : "F") << begin.wires.size() << ';';
        for (const auto &op : span) {
            CheckOp check{.kind = CheckOp::Kind::Result};
            if (op.predicate != Program::result_true) {
                check.predicate = localResult(op.predicate);
                if (!check.predicate.has_value()) {
                    return std::nullopt;
                }
            }
            switch (op.kind) {
            case OpKind::Gate:
            case OpKind::PauliRot: {
                if (op.wires.empty()) {
                    // A global phase.
                    continue;
                }
                std::vector<LocalRotation> rotations;
                try {
                    rotations = expandGate(op);
                } catch (const std::exception &) {
                    return std::nullopt;
                }
                for (const auto &rotation : rotations) {
                    auto paulis = localPaulis(rotation.pauli, op.wires);
                    if (!paulis.has_value()) {
                        return std::nullopt;
                    }
                    CheckOp step = check;
                    step.kind = CheckOp::Kind::Rotation;
                    step.paulis = std::move(*paulis);
                    step.half_angle = rotation.halfAngle();
                    shape.ops.push_back(std::move(step));
                }
                break;
            }
            case OpKind::Measure:
            case OpKind::PauliMeasure: {
                if (op.postselect.has_value()) {
                    return std::nullopt;
                }
                const std::string word = op.kind == OpKind::Measure
                                             ? std::string{"Z"}
                                             : op.pauli_word;
                auto paulis = localPaulis(word, op.wires);
                if (!paulis.has_value()) {
                    return std::nullopt;
                }
                check.kind = CheckOp::Kind::Measure;
                check.paulis = std::move(*paulis);
                check.negated = op.negated;
                check.result = results.size();
                results[op.result] = check.result;
                shape.ops.push_back(std::move(check));
                break;
            }
            case OpKind::Allocate:
                for (auto qubit : op.wires) {
                    CheckOp step = check;
                    step.kind = CheckOp::Kind::Allocate;
                    step.qubit = next_qubit;
                    aux[qubit] = next_qubit++;
                    shape.ops.push_back(std::move(step));
                }
                break;
            case OpKind::Release: {
                const auto it = aux.find(op.wires[0]);
                if (it == aux.end()) {
                    return std::nullopt;
                }
                check.kind = CheckOp::Kind::Release;
                check.qubit = it->second;
                shape.ops.push_back(std::move(check));
                break;
            }
            case OpKind::ResultConstant:
            case OpKind::ResultXor:
            case OpKind::ResultNot:
            case OpKind::ResultSelect: {
                check.result_kind = op.kind;
                check.constant = !op.params.empty() && op.params[0] != 0.0;
                for (auto operand : op.operands) {
                    auto local = localResult(operand);
                    if (!local.has_value()) {
                        return std::nullopt;
                    }
                    check.operands.push_back(*local);
                }
                check.result = results.size();
                results[op.result] = check.result;
                shape.ops.push_back(std::move(check));
                break;
            }
            case OpKind::ResultRelease:
                continue;
            default:
                return std::nullopt;
            }
        }
        shape.num_qubits = next_qubit;
        shape.num_results = results.size();
        for (const auto &op : shape.ops) {
            key << static_cast<int>(op.kind) << '|';
            for (auto [qubit, pauli] : op.paulis) {
                key << qubit << pauli;
            }
            key << '|' << op.half_angle << '|' << op.negated << '|' << op.qubit
                << '|' << static_cast<int>(op.result_kind) << op.constant
                << '|';
            for (auto operand : op.operands) {
                key << operand << ',';
            }
            key << '|'
                << (op.predicate ? static_cast<long>(*op.predicate) : -1L)
                << ';';
        }
        shape.key = key.str();
        return shape;
    }

    auto checked(const Shape &shape) -> const Detail::ShapeCheck & {
        auto it = shapes_.find(shape.key);
        if (it == shapes_.end()) {
            report_.shapes++;
            it = shapes_
                     .emplace(shape.key, Detail::checkShape(
                                             shape.num_data, shape.num_qubits,
                                             shape.num_results, shape.ops))
                     .first;
        }
        return it->second;
    }

    /**
     * @brief The op that replaces a gadget (none for the identity), or nothing
     * when the gadget runs as given.
     */
    auto collapse(const Program::Op &begin, std::vector<Program::Op> &span,
                  const std::vector<Program::Op> &window, std::size_t after)
        -> std::optional<std::optional<Program::Op>> {
        using Program::OpKind;
        if (mode_ == GadgetCollapse::Off) {
            uncollapsed(begin, "gadget collapse is off");
            return std::nullopt;
        }
        if (begin.predicate != Program::result_true) {
            uncollapsed(begin, "the gadget is conditional");
            return std::nullopt;
        }
        // The span's outcomes must not be read after it.
        std::unordered_set<Program::ResultId> defined;
        for (const auto &op : span) {
            if (op.result != Program::no_result) {
                defined.insert(op.result);
            }
        }
        for (std::size_t k = after; k < window.size(); k++) {
            const auto &op = window[k];
            if (op.kind == OpKind::ResultRelease) {
                continue;
            }
            bool reads = defined.contains(op.predicate);
            for (auto operand : op.operands) {
                reads = reads || defined.contains(operand);
            }
            if (reads) {
                uncollapsed(begin, "an outcome of the gadget is read after it");
                return std::nullopt;
            }
        }

        std::optional<Shape> shape = shapeOf(begin, span, /*reduce=*/true);
        if (!shape.has_value()) {
            shape = shapeOf(begin, span, /*reduce=*/false);
        }
        if (!shape.has_value()) {
            uncollapsed(begin, "the span holds an op the check cannot follow");
            return std::nullopt;
        }
        const Detail::ShapeCheck &check = checked(*shape);

        const double source_theta = begin.params[0];
        const auto verdict = static_cast<Program::GadgetVerdict>(
            static_cast<int>(begin.params[1]));
        const std::string source_word =
            shape->reduced ? std::string{"Z"} : begin.pauli_word;
        const bool is_source =
            check.merges &&
            Detail::unitaryDistance(
                Detail::rotationMatrix(source_word, source_theta),
                check.unitary) <= Detail::collapse_tolerance;

        if (mode_ == GadgetCollapse::Source) {
            if (!is_source) {
                report_.substitutions++;
                uncollapsed(begin,
                            "applied the source rotation, but the "
                            "device check found " +
                                (check.merges ? std::string{"another unitary"}
                                              : check.reason));
            }
            return rotation(begin, source_theta);
        }

        if (!check.merges) {
            uncollapsed(begin, check.reason);
            return std::nullopt;
        }
        PL_ABORT_IF_NOT(is_source,
                        "lightning.cliffish: gadget " + describe(begin) +
                            " merges to a different rotation than its source. "
                            "The compiled program is wrong; run with "
                            "gadget_collapse='source' to apply the source "
                            "rotation and report it.");
        PL_ABORT_IF(verdict == Program::GadgetVerdict::Mismatch,
                    "lightning.cliffish: gadget " + describe(begin) +
                        " implements its source rotation, but the compiler's "
                        "verdict is 'mismatch'.");
        return rotation(begin, source_theta);
    }

    static auto rotation(const Program::Op &begin, double theta)
        -> std::optional<Program::Op> {
        if (!cliffordQuarterTurns(theta).has_value() ||
            *cliffordQuarterTurns(theta) != 0) {
            return Program::Op{.kind = Program::OpKind::PauliRot,
                               .params = {theta},
                               .wires = begin.wires,
                               .pauli_word = begin.pauli_word};
        }
        return std::nullopt;
    }
};

} // namespace Pennylane::LightningCliffish::Planner
