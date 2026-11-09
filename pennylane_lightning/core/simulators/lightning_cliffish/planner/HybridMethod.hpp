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
 * @file HybridMethod.hpp
 * The hybrid method: a tableau plus a dense register.
 *
 * The state is C (|phi> (x) |0...0>): C is the tableau's Clifford, |phi> is
 * the dense register on the active virtual qubits, and every other virtual
 * qubit is dormant in |0>. A non-Clifford rotation pulled back through C
 * either acts on the register or promotes one dormant qubit into it.
 * Measurements, fused rotations (ADR 0007) and dense checks (ADR 0008) demote
 * qubits back.
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "AffineSubspace.hpp"
#include "DenseRegister.hpp"
#include "Error.hpp"
#include "GateExpansions.hpp"
#include "NativeTableau.hpp"
#include "Observables.hpp"
#include "OutcomeDistribution.hpp"
#include "PauliWord.hpp"
#include "Program.hpp"
#include "TableauEngine.hpp"

namespace Pennylane::LightningCliffish::Planner {

using Pauli::PauliWord;
using Program::QubitId;
using Program::ResultId;

/// Grow the engine by `count` qubits in |0>.
struct AddQubitsStep {
    std::size_t count;
};

/// A non-Clifford rotation exp(-i angle/2 P) on the engine's qubits.
struct RotationStep {
    PauliWord pauli;
    double angle;
};

/// Measure and write the outcome to `result`.
struct MeasureStep {
    PauliWord pauli;
    ResultId result;
    std::optional<bool> forced_outcome;
};

/// Return engine qubit `slot` to |0>.
struct ResetStep {
    std::size_t slot;
};

/**
 * @brief One step of a plan: an engine block, or a step the planner takes
 * itself.
 */
using PlanStep = std::variant<Tableau::EngineBlock, RotationStep, AddQubitsStep,
                              MeasureStep, ResetStep>;

using Plan = std::vector<PlanStep>;

/**
 * @brief Measurement outcomes by result id.
 */
using Results = std::unordered_map<ResultId, bool>;

/**
 * @brief How the register shrinks without reading amplitudes (ADR 0007).
 */
enum class Recompression : std::uint8_t {
    /// Promote on every non-Clifford rotation; demote only at measurements.
    None,
    /// Fuse pi/8 rotations about one axis of a freshly promoted qubit.
    T1,
    /// Fuse any rotations about one axis of a freshly promoted qubit, and
    /// demote it once their sum is a Clifford angle.
    T2,
};

inline auto recompressionName(Recompression recompression) -> std::string {
    switch (recompression) {
    case Recompression::None:
        return "none";
    case Recompression::T1:
        return "t1";
    default:
        return "t2";
    }
}

/**
 * @brief The policy named `name`, or nothing for "auto".
 */
inline auto parseRecompression(const std::string &name)
    -> std::optional<Recompression> {
    if (name == "auto") {
        return std::nullopt;
    }
    for (auto recompression :
         {Recompression::None, Recompression::T1, Recompression::T2}) {
        if (name == recompressionName(recompression)) {
            return recompression;
        }
    }
    PL_ABORT("Unknown recompression policy '" + name +
             "'; use auto, none, t1 or t2");
}

struct HybridOptions {
    /// The largest dense register; a program that needs more fails.
    std::size_t max_dense_qubits{24};
    /// The re-compression policy, or empty to choose it by a dry run.
    std::optional<Recompression> recompression{};
    /// Whether to run dense checks: trigger 4 and the check before outputs.
    bool dense_checks{true};
};

/**
 * @brief What the hybrid method did, for tests and reports.
 */
struct HybridReport {
    /// The policy in use.
    Recompression recompression{Recompression::T2};
    /// The dry run's predicted peak and final register sizes, for the last
    /// window it chose a policy for.
    std::optional<std::size_t> predicted_peak{};
    std::optional<std::size_t> predicted_final{};
    std::size_t dense_qubits{0};
    std::size_t peak_dense_qubits{0};
    std::size_t promotions{0};
    std::size_t fusion_demotions{0};
    std::size_t measurement_demotions{0};
    std::size_t dense_checks{0};
    std::size_t successful_dense_checks{0};
    std::size_t check_demotions{0};
};

/// A dense check accepts a stabilizer only this close to one (ADR 0008).
inline constexpr double stabilizer_tolerance = 1e-12;

/// A demoted qubit may hold at most this much weight outside |0>.
inline constexpr double demotion_leak_tolerance = 1e-6;

/**
 * @brief Conjugate `word` by exp(-i quarter_turns pi/4 R).
 *
 * A word that anticommutes with R maps to exp(-i quarter_turns pi/2 R) word.
 */
inline void conjugate(PauliWord &word, const PauliWord &rotation,
                      std::uint8_t quarter_turns) {
    if (word.commutes(rotation)) {
        return;
    }
    if (quarter_turns == 2) {
        word.setLogI(word.logI() + 2);
        return;
    }
    word = rotation * word;
    word.setLogI(word.logI() + (quarter_turns == 1 ? 3 : 1));
}

class HybridMethod {
  private:
    /// Rotations on virtual qubits, applied in order.
    using VirtualGate = Tableau::EngineBlock;

    struct CleanQubit {
        /// The axis, X, Y or Z, of every rotation on the qubit since it was
        /// promoted.
        char axis;
        /// The qubit is exp(-i half_angle axis)|0>.
        double half_angle;
    };

    HybridOptions options_{};
    std::unique_ptr<Tableau::TableauEngine> engine_;

    // Lowering state.

    /// Engine qubit of each live program qubit.
    std::unordered_map<QubitId, std::size_t> slots_{};
    /// Engine qubits in |0> that no program qubit holds.
    std::vector<std::size_t> free_slots_{};
    /// Engine width once every plan lowered so far has run.
    std::size_t planned_width_{0};

    // Run state.

    /// A dry run tracks the register's size only, with no amplitudes.
    bool frame_only_{false};
    Dense::DenseRegister<> register_{};
    /// The virtual qubit of each register qubit.
    std::vector<std::size_t> active_{};
    Recompression recompression_{Recompression::T2};
    /// Register qubits still in a product state with the rest.
    std::unordered_map<std::size_t, CleanQubit> clean_{};
    std::size_t rotations_since_check_{0};
    std::size_t check_budget_{1};
    /// Whether the register is unchanged since the last dense check.
    bool checked_{true};
    HybridReport report_{};

    struct DryRunTag {};

    HybridMethod(DryRunTag /*tag*/,
                 std::unique_ptr<Tableau::TableauEngine> engine,
                 HybridOptions options)
        : options_{options}, engine_{std::move(engine)} {}

    [[nodiscard]] auto slotOf(QubitId qubit) const -> std::size_t {
        const auto it = slots_.find(qubit);
        PL_ABORT_IF(it == slots_.end(), "Qubit has no engine slot");
        return it->second;
    }

    /// The word on the engine's planned width with `pauli[k]` on `wires[k]`.
    [[nodiscard]] auto engineWord(const std::string &pauli,
                                  const std::vector<QubitId> &wires) const
        -> PauliWord {
        PauliWord word(planned_width_);
        for (std::size_t k = 0; k < wires.size(); k++) {
            word.set(slotOf(wires[k]), pauli[k]);
        }
        return word;
    }

    /// `pauli` on virtual qubit `slot`.
    [[nodiscard]] auto singleWord(std::size_t slot, char pauli) const
        -> PauliWord {
        PauliWord word(engine_->numQubits());
        word.set(slot, pauli);
        return word;
    }

    /// The gate `name` on virtual qubits, as rotations.
    [[nodiscard]] auto virtualGate(const std::string &name,
                                   const std::vector<std::size_t> &slots,
                                   bool inverse = false) const -> VirtualGate {
        const Program::Op op{.kind = Program::OpKind::Gate,
                             .name = name,
                             .wires = {slots.begin(), slots.end()},
                             .inverse = inverse};
        VirtualGate gate;
        for (const auto &rotation : expandGate(op)) {
            PauliWord word(engine_->numQubits());
            for (std::size_t k = 0; k < slots.size(); k++) {
                word.set(slots[k], rotation.pauli[k]);
            }
            gate.push_back(
                Tableau::EngineOp::rotation(word, rotation.quarter_turns));
        }
        return gate;
    }

    [[nodiscard]] auto isActive(std::size_t slot) const -> bool {
        return std::find(active_.begin(), active_.end(), slot) != active_.end();
    }

    /**
     * @brief Drop the Z bits of `word` on dormant qubits where it has no X:
     * they act as +1 on |0>.
     *
     * @return The dormant qubits where `word` has an X or a Y.
     */
    auto reduce(PauliWord &word) const -> std::vector<std::size_t> {
        std::vector<Pauli::Word> active(word.numWords(), 0);
        for (auto slot : active_) {
            Pauli::setBit(active.data(), slot, true);
        }
        std::vector<std::size_t> dormant_x;
        for (std::size_t w = 0; w < word.numWords(); w++) {
            const Pauli::Word dormant = ~active[w];
            word.z()[w] &= ~(dormant & ~word.x()[w]);
            for (Pauli::Word bits = dormant & word.x()[w]; bits != 0;
                 bits &= bits - 1) {
                dormant_x.push_back(
                    w * Pauli::word_bits +
                    static_cast<std::size_t>(std::countr_zero(bits)));
            }
        }
        return dormant_x;
    }

    /// The register qubits `word` acts on, in register order.
    [[nodiscard]] auto activeSupport(const PauliWord &word) const
        -> std::vector<std::size_t> {
        std::vector<std::size_t> support;
        for (auto slot : active_) {
            if (word.get(slot) != 'I') {
                support.push_back(slot);
            }
        }
        return support;
    }

    /// `word`, which acts on register qubits only, as a register word.
    [[nodiscard]] auto densePauli(const PauliWord &word) const
        -> Dense::DensePauli {
        Dense::DensePauli pauli{.log_i = word.logI()};
        std::size_t covered = 0;
        for (std::size_t p = 0; p < active_.size(); p++) {
            const char single = word.get(active_[p]);
            if (single == 'X' || single == 'Y') {
                pauli.x |= std::uint64_t{1} << p;
            }
            if (single == 'Z' || single == 'Y') {
                pauli.z |= std::uint64_t{1} << p;
            }
            covered += (single != 'I') ? 1 : 0;
        }
        PL_ABORT_IF(covered != word.weight(),
                    "The word acts outside the dense register");
        return pauli;
    }

    /**
     * @brief Apply a gate on virtual qubits: the frame changes by its
     * inverse, so the state is unchanged.
     *
     * @param on_register Apply the gate to the register too. Without it, the
     * gate must fix the register and the dormant |0>, or the caller means to
     * change the state.
     * @param tracked Words to conjugate by the gate.
     */
    void applyVirtual(const VirtualGate &gate, bool on_register,
                      const std::vector<PauliWord *> &tracked = {}) {
        engine_->applyVirtualBlock(gate);
        for (const auto &op : gate) {
            if (on_register) {
                if (!frame_only_) {
                    register_.rotateQuarterTurns(densePauli(op.pauli),
                                                 op.quarter_turns);
                }
                for (auto slot : activeSupport(op.pauli)) {
                    clean_.erase(slot);
                }
            }
            for (auto *word : tracked) {
                conjugate(*word, op.pauli, op.quarter_turns);
            }
        }
    }

    /**
     * @brief Make `word` an X on dormant qubit d and the identity on the
     * other dormant qubits, by gates that fix the dormant |0>.
     *
     * @return d.
     */
    auto isolateDormant(PauliWord &word,
                        const std::vector<std::size_t> &dormant_x)
        -> std::size_t {
        const std::size_t d = dormant_x.front();
        for (std::size_t k = 1; k < dormant_x.size(); k++) {
            applyVirtual(virtualGate("CNOT", {d, dormant_x[k]}), false,
                         {&word});
        }
        reduce(word);
        if (word.get(d) == 'Y') {
            applyVirtual(virtualGate("S", {d}, true), false, {&word});
        }
        return d;
    }

    [[nodiscard]] auto overflowMessage(std::size_t needed) const
        -> std::string {
        const double gib = 16.0 * std::ldexp(1.0, static_cast<int>(needed)) /
                           std::ldexp(1.0, 30);
        return "lightning.cliffish needs a dense register of " +
               std::to_string(needed) +
               " qubits for this program, more than max_dense_qubits = " +
               std::to_string(options_.max_dense_qubits) +
               ". The program carries more non-Clifford magic than the "
               "hybrid method can hold. Run it on lightning.qubit or "
               "lightning.kokkos instead, or raise max_dense_qubits if " +
               std::to_string(gib) + " GiB of amplitudes fit in memory.";
    }

    void promote(std::size_t slot) {
        if (!frame_only_) {
            PL_ABORT_IF(active_.size() + 1 > options_.max_dense_qubits,
                        overflowMessage(active_.size() + 1));
            register_.addQubit();
        }
        active_.push_back(slot);
        report_.promotions++;
        report_.peak_dense_qubits =
            std::max(report_.peak_dense_qubits, active_.size());
    }

    /// Remove register qubit `slot`, which is in |0>, making it dormant.
    void drop(std::size_t slot) {
        const auto it = std::find(active_.begin(), active_.end(), slot);
        PL_ABORT_IF(it == active_.end(), "Qubit is not in the dense register");
        if (!frame_only_) {
            const double leaked = register_.dropQubit(
                static_cast<std::size_t>(it - active_.begin()));
            PL_ABORT_IF(leaked > demotion_leak_tolerance,
                        "Internal error: a demoted qubit is not in |0>");
        }
        active_.erase(it);
        clean_.erase(slot);
    }

    /**
     * @brief Map `word`, a register word, to plus or minus Z_q for a register
     * qubit q not in `excluded`, by Clifford gates on the register.
     *
     * @param apply Applies each gate, conjugating `word` and any other words
     * the caller tracks.
     * @return q.
     */
    template <class Apply>
    auto mapToZ(PauliWord &word, const std::vector<std::size_t> &excluded,
                Apply &&apply) const -> std::size_t {
        auto withPauli = [this, &word](auto test) {
            std::vector<std::size_t> slots;
            for (auto slot : active_) {
                if (test(word.get(slot))) {
                    slots.push_back(slot);
                }
            }
            return slots;
        };
        const auto x_slots =
            withPauli([](char p) { return p == 'X' || p == 'Y'; });
        std::size_t q = 0;
        if (!x_slots.empty()) {
            q = x_slots.front();
            for (std::size_t k = 1; k < x_slots.size(); k++) {
                apply(virtualGate("CNOT", {q, x_slots[k]}));
            }
            if (word.get(q) == 'Y') {
                apply(virtualGate("S", {q}, true));
            }
            apply(virtualGate("Hadamard", {q}));
        } else {
            const auto z_slots = withPauli([](char p) { return p == 'Z'; });
            const auto it =
                std::find_if(z_slots.begin(), z_slots.end(), [&](auto slot) {
                    return std::find(excluded.begin(), excluded.end(), slot) ==
                           excluded.end();
                });
            PL_ABORT_IF(it == z_slots.end(),
                        "Internal error: the register words are dependent");
            q = *it;
        }
        for (auto slot : withPauli([](char p) { return p == 'Z'; })) {
            if (slot != q) {
                apply(virtualGate("CNOT", {slot, q}));
            }
        }
        PL_ABORT_IF_NOT(word.weight() == 1 && word.get(q) == 'Z',
                        "Internal error: the word did not map to Z");
        return q;
    }

    /**
     * @brief Demote the register qubit that `word` stabilizes: `word` is a
     * register word and the register is its +1 eigenstate.
     */
    void demote(PauliWord &word, const std::vector<PauliWord *> &others = {}) {
        std::vector<PauliWord *> tracked{&word};
        tracked.insert(tracked.end(), others.begin(), others.end());
        const std::size_t q =
            mapToZ(word, {}, [this, &tracked](const VirtualGate &gate) {
                applyVirtual(gate, true, tracked);
            });
        if (word.logI() == 2) {
            applyVirtual(virtualGate("PauliX", {q}), true, tracked);
        }
        drop(q);
    }

    /// One fair bit, drawn as the tableau draws its random outcomes.
    static auto randomBit(std::mt19937 &gen) -> bool {
        return ((gen() >> 31U) & 1U) != 0;
    }

    /// A uniform double in [0, 1) with 53 random bits.
    static auto uniform(std::mt19937 &gen) -> double {
        const auto high = static_cast<double>(gen() >> 5U);
        const auto low = static_cast<double>(gen() >> 6U);
        return (high * 67108864.0 + low) / 9007199254740992.0;
    }

    void applyNonClifford(const RotationStep &step) {
        PauliWord word = engine_->pullBack(step.pauli);
        const auto dormant_x = reduce(word);
        std::optional<std::size_t> promoted{};
        if (!dormant_x.empty()) {
            promoted = isolateDormant(word, dormant_x);
            promote(*promoted);
        }
        const auto support = activeSupport(word);
        if (support.empty()) {
            return; // A global phase.
        }
        const double half_angle = step.angle / 2;
        const double sign = word.logI() == 0 ? 1.0 : -1.0;
        trackFusion(word, support, promoted, half_angle, sign);
        if (!frame_only_) {
            register_.rotate(densePauli(word), half_angle);
            checked_ = false;
        }

        if (support.size() == 1) {
            const std::size_t q = support.front();
            if (const auto it = clean_.find(q); it != clean_.end()) {
                const auto quarter_turns =
                    cliffordQuarterTurns(2 * it->second.half_angle);
                if (quarter_turns.has_value()) {
                    demoteClean(q, *quarter_turns);
                }
            }
        }

        if (options_.dense_checks && !active_.empty() &&
            ++rotations_since_check_ >= check_budget_) {
            rotations_since_check_ = 0;
            const std::size_t found = denseCheck();
            check_budget_ = (found > 0)
                                ? std::max<std::size_t>(1, active_.size())
                                : 2 * check_budget_;
        }
    }

    /**
     * @brief Update the clean qubits for a rotation exp(-i half_angle word)
     * on the register qubits `support`.
     */
    void trackFusion(const PauliWord &word,
                     const std::vector<std::size_t> &support,
                     std::optional<std::size_t> promoted, double half_angle,
                     double sign) {
        if (recompression_ == Recompression::None || support.size() != 1) {
            for (auto slot : support) {
                clean_.erase(slot);
            }
            return;
        }
        const std::size_t q = support.front();
        const char axis = word.get(q);
        const bool fusable =
            recompression_ == Recompression::T2 ||
            std::abs(std::abs(half_angle) - M_PI / 8) <= clifford_snap;
        const auto it = clean_.find(q);
        if (promoted == q && fusable) {
            clean_[q] = {axis, sign * half_angle};
        } else if (it != clean_.end() && it->second.axis == axis && fusable) {
            it->second.half_angle += sign * half_angle;
        } else {
            clean_.erase(q);
        }
    }

    /// Undo the Clifford rotation on clean qubit `slot` and demote it.
    void demoteClean(std::size_t slot, std::uint8_t quarter_turns) {
        const char axis = clean_.at(slot).axis;
        if (quarter_turns != 0) {
            applyVirtual({Tableau::EngineOp::rotation(
                             singleWord(slot, axis),
                             static_cast<std::uint8_t>(4 - quarter_turns))},
                         true);
        }
        drop(slot);
        report_.fusion_demotions++;
    }

    auto measure(const PauliWord &pauli, std::optional<bool> forced_outcome,
                 std::mt19937 &gen) -> bool {
        if (active_.empty()) {
            return engine_->applyBlock(
                {Tableau::EngineOp::measurement(pauli, forced_outcome)},
                gen)[0];
        }
        PauliWord word = engine_->pullBack(pauli);
        const auto dormant_x = reduce(word);
        if (!dormant_x.empty()) {
            // The word is Q_A X_d. (1 + m Q_A X_d)(|phi> (x) |0>_d) / sqrt(2)
            // is U (|phi> (x) |0>_d) with U = exp(-i pi/4 m Q_A Y_d), so the
            // outcome is fair and only the frame changes: C -> C U.
            const std::size_t d = isolateDormant(word, dormant_x);
            const bool outcome = forced_outcome.value_or(randomBit(gen));
            PauliWord generator = word;
            Pauli::setBit(generator.z(), d, true);
            if (outcome) {
                generator.setLogI(generator.logI() + 2);
            }
            applyVirtual({Tableau::EngineOp::rotation(generator, 3)}, false);
            return outcome;
        }

        const auto support = activeSupport(word);
        if (support.empty()) {
            const bool outcome = word.logI() == 2;
            PL_ABORT_IF(forced_outcome.has_value() &&
                            *forced_outcome != outcome,
                        "Cannot project onto a measurement outcome with zero "
                        "probability");
            return outcome;
        }
        bool outcome = forced_outcome.value_or(false);
        if (!frame_only_) {
            const auto dense_word = densePauli(word);
            const double p1 = std::clamp(
                (1.0 - register_.expectation(dense_word)) / 2, 0.0, 1.0);
            if (!forced_outcome.has_value()) {
                outcome = uniform(gen) < p1;
            }
            const double probability = outcome ? p1 : 1.0 - p1;
            PL_ABORT_IF(probability < stabilizer_tolerance,
                        "Cannot project onto a measurement outcome with zero "
                        "probability");
            const double scale = 1.0 / (2 * std::sqrt(probability));
            register_.applyLinear(dense_word, {scale, 0.0},
                                  {outcome ? -scale : scale, 0.0});
            checked_ = false;
        }
        if (outcome) {
            word.setLogI(word.logI() + 2);
        }
        demote(word);
        report_.measurement_demotions++;
        return outcome;
    }

    /**
     * @brief The vectors v with v . (s xor s0) = 0 for every index s in the
     * support of `amplitudes`: the Z^v that stabilize the register up to a
     * sign.
     */
    static auto hullNullSpace(
        const std::vector<Dense::DenseRegister<>::Complex> &amplitudes,
        std::size_t num_qubits) -> std::vector<std::uint64_t> {
        double largest = 0.0;
        for (const auto &amplitude : amplitudes) {
            largest = std::max(largest, Kokkos::abs(amplitude));
        }
        // Reduced row echelon form: row i has its highest bit at pivots[i],
        // and no other row has that bit.
        std::vector<std::uint64_t> rows;
        std::vector<std::uint64_t> pivots;
        std::optional<std::uint64_t> first{};
        for (std::uint64_t k = 0; k < amplitudes.size(); k++) {
            if (Kokkos::abs(amplitudes[k]) <= 1e-13 * largest) {
                continue;
            }
            if (!first.has_value()) {
                first = k;
                continue;
            }
            std::uint64_t vector = k ^ *first;
            for (std::size_t i = 0; i < rows.size(); i++) {
                if (vector & pivots[i]) {
                    vector ^= rows[i];
                }
            }
            if (vector == 0) {
                continue;
            }
            const std::uint64_t pivot = std::uint64_t{1}
                                        << (63 - std::countl_zero(vector));
            for (auto &row : rows) {
                if (row & pivot) {
                    row ^= vector;
                }
            }
            rows.push_back(vector);
            pivots.push_back(pivot);
            if (rows.size() == num_qubits) {
                return {};
            }
        }
        std::uint64_t pivot_mask = 0;
        for (auto pivot : pivots) {
            pivot_mask |= pivot;
        }
        std::vector<std::uint64_t> null_space;
        for (std::size_t bit = 0; bit < num_qubits; bit++) {
            const std::uint64_t free = std::uint64_t{1} << bit;
            if (free & pivot_mask) {
                continue;
            }
            std::uint64_t vector = free;
            for (std::size_t i = 0; i < rows.size(); i++) {
                if (rows[i] & free) {
                    vector |= pivots[i];
                }
            }
            null_space.push_back(vector);
        }
        return null_space;
    }

    /**
     * @brief Find Paulis that stabilize the register and fold them back into
     * the tableau (ADR 0008).
     *
     * @return The number of register qubits demoted.
     */
    auto denseCheck() -> std::size_t {
        checked_ = true;
        if (active_.empty() || frame_only_ || !options_.dense_checks) {
            return 0;
        }
        report_.dense_checks++;
        const std::size_t num_active = active_.size();

        // Accepted stabilizers, signed so that the register is their +1
        // eigenstate, with their (x, z) bits kept in a reduced basis.
        std::vector<PauliWord> found;
        std::vector<std::uint64_t> basis_x;
        std::vector<std::uint64_t> basis_z;
        std::vector<std::pair<bool, std::size_t>> basis_pivots;
        auto reduceBits = [&](std::uint64_t &x, std::uint64_t &z) {
            for (std::size_t i = 0; i < basis_x.size(); i++) {
                const auto [is_z, bit] = basis_pivots[i];
                if (((is_z ? z : x) >> bit) & 1U) {
                    x ^= basis_x[i];
                    z ^= basis_z[i];
                }
            }
        };
        auto offer = [&](PauliWord word) {
            word.setLogI(0);
            auto dense_word = densePauli(word);
            std::uint64_t x = dense_word.x;
            std::uint64_t z = dense_word.z;
            reduceBits(x, z);
            if (x == 0 && z == 0) {
                return;
            }
            if (register_.residual(dense_word) > stabilizer_tolerance) {
                dense_word.log_i = 2;
                if (register_.residual(dense_word) > stabilizer_tolerance) {
                    return;
                }
                word.setLogI(2);
            }
            const bool is_z = (x == 0);
            const auto bit =
                static_cast<std::size_t>(std::countr_zero(is_z ? z : x));
            for (std::size_t i = 0; i < basis_x.size(); i++) {
                if ((((is_z ? basis_z[i] : basis_x[i]) >> bit) & 1U) != 0) {
                    basis_x[i] ^= x;
                    basis_z[i] ^= z;
                }
            }
            basis_x.push_back(x);
            basis_z.push_back(z);
            basis_pivots.emplace_back(is_z, bit);
            found.push_back(std::move(word));
        };

        // Single virtual-qubit Paulis on the register.
        for (auto slot : std::vector<std::size_t>(active_)) {
            for (char pauli : {'Z', 'X', 'Y'}) {
                offer(singleWord(slot, pauli));
            }
        }
        // Images of single physical-qubit Paulis that act on the register
        // alone: a physical qubit back in a product state.
        for (std::size_t qubit = 0; qubit < engine_->numQubits(); qubit++) {
            for (char pauli : {'X', 'Z', 'Y'}) {
                PauliWord word = engine_->pullBack(singleWord(qubit, pauli));
                if (!reduce(word).empty() || activeSupport(word).empty()) {
                    continue;
                }
                offer(std::move(word));
            }
        }
        // Z-type and X-type stabilizers from the affine hull of the support.
        auto offerHull = [&](const Dense::DenseRegister<> &state, char pauli) {
            for (auto vector :
                 hullNullSpace(state.hostAmplitudes(), num_active)) {
                PauliWord word(engine_->numQubits());
                for (std::size_t p = 0; p < num_active; p++) {
                    if ((vector >> p) & 1U) {
                        word.set(active_[p], pauli);
                    }
                }
                offer(std::move(word));
            }
        };
        offerHull(register_, 'Z');
        auto transformed = register_.clone();
        for (std::size_t p = 0; p < num_active; p++) {
            for (const auto &op : virtualGate("Hadamard", {active_[p]})) {
                transformed.rotateQuarterTurns(densePauli(op.pauli),
                                               op.quarter_turns);
            }
        }
        offerHull(transformed, 'X');

        if (found.empty()) {
            return 0;
        }
        report_.successful_dense_checks++;
        for (std::size_t i = 0; i < found.size(); i++) {
            // Earlier demotions leave Z bits on now-dormant qubits.
            PL_ABORT_IF_NOT(reduce(found[i]).empty(),
                            "Internal error: a stabilizer left the register");
            std::vector<PauliWord *> rest;
            for (std::size_t j = i + 1; j < found.size(); j++) {
                rest.push_back(&found[j]);
            }
            demote(found[i], rest);
            report_.check_demotions++;
        }
        return found.size();
    }

    void setRecompression(Recompression recompression) {
        if (recompression != recompression_) {
            clean_.clear();
        }
        recompression_ = recompression;
        report_.recompression = recompression;
    }

    /**
     * @brief The peak and final register sizes of running `plan` with
     * `recompression`, without amplitudes.
     */
    [[nodiscard]] auto dryRun(const Plan &plan,
                              Recompression recompression) const
        -> std::pair<std::size_t, std::size_t> {
        HybridOptions options = options_;
        options.dense_checks = false;
        HybridMethod copy(DryRunTag{}, engine_->clone(), options);
        copy.frame_only_ = true;
        copy.active_ = active_;
        copy.recompression_ = recompression;
        if (recompression == recompression_) {
            copy.clean_ = clean_;
        }
        copy.report_.peak_dense_qubits = active_.size();
        std::mt19937 gen{0};
        Results results;
        copy.run(plan, gen, results);
        return {copy.report_.peak_dense_qubits, copy.active_.size()};
    }

    /**
     * @brief Pick the policy for `plan` by a dry run of each (ADR 0007).
     *
     * A dry run cannot follow a measurement on the register without its
     * amplitudes, so a plan with measurements runs t2.
     */
    void chooseRecompression(const Plan &plan) {
        if (options_.recompression.has_value()) {
            setRecompression(*options_.recompression);
            return;
        }
        bool has_rotation = false;
        bool has_measurement = false;
        for (const auto &step : plan) {
            has_rotation |= std::holds_alternative<RotationStep>(step);
            has_measurement |= std::holds_alternative<MeasureStep>(step) ||
                               std::holds_alternative<ResetStep>(step);
        }
        if (!has_rotation) {
            return;
        }
        if (has_measurement) {
            setRecompression(Recompression::T2);
            report_.predicted_peak.reset();
            report_.predicted_final.reset();
            return;
        }
        // Ties go to the cheaper policy.
        std::optional<std::pair<std::size_t, std::size_t>> best{};
        Recompression best_recompression = Recompression::None;
        for (auto recompression :
             {Recompression::None, Recompression::T1, Recompression::T2}) {
            const auto predicted = dryRun(plan, recompression);
            if (!best.has_value() || predicted < *best) {
                best = predicted;
                best_recompression = recompression;
            }
        }
        setRecompression(best_recompression);
        report_.predicted_peak = best->first;
        report_.predicted_final = best->second;
    }

  public:
    explicit HybridMethod(HybridOptions options = {})
        : HybridMethod(std::make_unique<Tableau::NativeTableau<>>(), options) {}

    explicit HybridMethod(std::unique_ptr<Tableau::TableauEngine> engine,
                          HybridOptions options = {})
        : options_{options}, engine_{std::move(engine)} {
        PL_ABORT_IF(engine_->numQubits() != 0,
                    "The hybrid method needs an engine with no qubits");
        PL_ABORT_IF_NOT(engine_->capabilities().dense_register,
                        "The hybrid method needs an engine that couples to a "
                        "dense register");
        PL_ABORT_IF(options_.max_dense_qubits > Dense::max_register_qubits,
                    "max_dense_qubits must be at most " +
                        std::to_string(Dense::max_register_qubits));
        recompression_ = options_.recompression.value_or(Recompression::T2);
        report_.recompression = recompression_;
    }

    [[nodiscard]] auto engine() const -> const Tableau::TableauEngine & {
        return *engine_;
    }

    [[nodiscard]] auto report() const -> HybridReport {
        HybridReport report = report_;
        report.dense_qubits = active_.size();
        return report;
    }

    /**
     * @brief Lower a window to a plan.
     *
     * Lowering assigns engine qubits, so plans must run in the order they are
     * lowered.
     */
    auto lower(const std::vector<Program::Op> &window) -> Plan {
        using Program::OpKind;
        Plan plan;
        auto current_block = [&plan]() -> Tableau::EngineBlock & {
            if (plan.empty() ||
                !std::holds_alternative<Tableau::EngineBlock>(plan.back())) {
                plan.emplace_back(Tableau::EngineBlock{});
            }
            return std::get<Tableau::EngineBlock>(plan.back());
        };

        for (const auto &op : window) {
            PL_ABORT_IF_NOT(op.predicate == Program::result_true,
                            "lightning.cliffish does not support conditional "
                            "ops yet");
            switch (op.kind) {
            case OpKind::Allocate: {
                std::size_t new_slots = 0;
                for (auto qubit : op.wires) {
                    if (free_slots_.empty()) {
                        slots_[qubit] = planned_width_ + new_slots++;
                    } else {
                        slots_[qubit] = free_slots_.back();
                        free_slots_.pop_back();
                    }
                }
                if (new_slots > 0) {
                    plan.emplace_back(AddQubitsStep{new_slots});
                    planned_width_ += new_slots;
                }
                break;
            }
            case OpKind::Release: {
                const std::size_t slot = slotOf(op.wires[0]);
                plan.emplace_back(ResetStep{slot});
                slots_.erase(op.wires[0]);
                free_slots_.push_back(slot);
                break;
            }
            case OpKind::Reset:
                plan.emplace_back(ResetStep{slotOf(op.wires[0])});
                break;
            case OpKind::Gate:
            case OpKind::PauliRot:
                for (const auto &rotation : expandGate(op)) {
                    auto word = engineWord(rotation.pauli, op.wires);
                    if (rotation.angle.has_value()) {
                        plan.emplace_back(
                            RotationStep{std::move(word), *rotation.angle});
                    } else {
                        current_block().push_back(Tableau::EngineOp::rotation(
                            std::move(word), rotation.quarter_turns));
                    }
                }
                break;
            case OpKind::Measure:
                plan.emplace_back(MeasureStep{engineWord("Z", op.wires),
                                              op.result, op.postselect});
                break;
            case OpKind::PauliMeasure: {
                auto word = engineWord(op.pauli_word, op.wires);
                if (op.negated) {
                    word.setLogI(
                        static_cast<std::uint8_t>((word.logI() + 2) % 4));
                }
                plan.emplace_back(
                    MeasureStep{std::move(word), op.result, op.postselect});
                break;
            }
            case OpKind::ResultRelease:
                break;
            case OpKind::ResultConstant:
            case OpKind::ResultXor:
            case OpKind::ResultNot:
            case OpKind::ResultSelect:
            case OpKind::GadgetBegin:
            case OpKind::GadgetEnd:
                PL_ABORT("lightning.cliffish plans result ops and gadget "
                         "brackets only on a delegated program");
            }
        }
        return plan;
    }

    /**
     * @brief Run a plan, writing measurement outcomes to `results`.
     */
    void run(const Plan &plan, std::mt19937 &gen, Results &results) {
        if (!frame_only_) {
            chooseRecompression(plan);
        }
        for (const auto &step : plan) {
            if (const auto *block = std::get_if<Tableau::EngineBlock>(&step)) {
                engine_->applyBlock(*block, gen);
            } else if (const auto *rotation =
                           std::get_if<RotationStep>(&step)) {
                applyNonClifford(*rotation);
            } else if (const auto *grow = std::get_if<AddQubitsStep>(&step)) {
                engine_->addQubits(grow->count);
            } else if (const auto *step_measure =
                           std::get_if<MeasureStep>(&step)) {
                results[step_measure->result] = measure(
                    step_measure->pauli, step_measure->forced_outcome, gen);
            } else {
                const auto &reset = std::get<ResetStep>(step);
                const PauliWord z = singleWord(reset.slot, 'Z');
                if (measure(z, std::nullopt, gen)) {
                    engine_->applyBlock({Tableau::EngineOp::rotation(
                                            singleWord(reset.slot, 'X'), 2)},
                                        gen);
                }
            }
        }
    }

    /**
     * @brief Run the dense check that precedes an output, unless the
     * register is unchanged since the last one (ADR 0008).
     */
    void prepareOutput() {
        if (!checked_) {
            denseCheck();
        }
    }

    /**
     * @brief Exact expectation of a Hermitian word on the engine's qubits.
     */
    [[nodiscard]] auto expectationOf(const PauliWord &pauli) const -> double {
        PL_ABORT_IF_NOT(planned_width_ == engine_->numQubits(),
                        "Every lowered plan must run before an output");
        if (active_.empty()) {
            return engine_->expectation(pauli);
        }
        PauliWord word = engine_->pullBack(pauli);
        if (!reduce(word).empty()) {
            return 0.0;
        }
        if (activeSupport(word).empty()) {
            return word.logI() == 0 ? 1.0 : -1.0;
        }
        return register_.expectation(densePauli(word));
    }

    /**
     * @brief Expectation of the Pauli word with `pauli[k]` on `wires[k]`.
     */
    [[nodiscard]] auto expectation(const std::string &pauli,
                                   const std::vector<QubitId> &wires) const
        -> double {
        PL_ABORT_IF_NOT(pauli.size() == wires.size(),
                        "The Pauli word must have one character per wire");
        return expectationOf(engineWord(pauli, wires));
    }

    /// Exact expectation value of `observable`.
    [[nodiscard]] auto expval(const Observable &observable) const -> double {
        double value = 0.0;
        for (const auto &term : observable) {
            value += term.coefficient * expectation(term.pauli, term.wires);
        }
        return value;
    }

    /**
     * @brief Exact variance of `observable`: <O^2> - <O>^2.
     *
     * With O = sum_i c_i P_i, <O^2> = sum_i c_i^2 + 2 sum_{i<j} c_i c_j
     * <P_i P_j>, where only commuting pairs contribute.
     */
    [[nodiscard]] auto variance(const Observable &observable) const -> double {
        PL_ABORT_IF_NOT(planned_width_ == engine_->numQubits(),
                        "Every lowered plan must run before an output");
        std::vector<PauliWord> words;
        words.reserve(observable.size());
        for (const auto &term : observable) {
            PL_ABORT_IF_NOT(term.pauli.size() == term.wires.size(),
                            "The Pauli word must have one character per wire");
            words.push_back(engineWord(term.pauli, term.wires));
        }

        double mean = 0.0;
        double second_moment = 0.0;
        for (std::size_t i = 0; i < words.size(); i++) {
            const double c_i = observable[i].coefficient;
            mean += c_i * expectationOf(words[i]);
            second_moment += c_i * c_i;
            for (std::size_t j = i + 1; j < words.size(); j++) {
                if (words[i].commutes(words[j])) {
                    second_moment += 2 * c_i * observable[j].coefficient *
                                     expectationOf(words[i] * words[j]);
                }
            }
        }
        return std::max(0.0, second_moment - mean * mean);
    }

    /**
     * @brief Support of the computational-basis outcomes of `wires` while
     * the register is empty; bit k is the outcome of wires[k].
     */
    [[nodiscard]] auto
    computationalSupport(const std::vector<QubitId> &wires) const
        -> GF2::AffineSubspace {
        PL_ABORT_IF_NOT(planned_width_ == engine_->numQubits(),
                        "Every lowered plan must run before an output");
        PL_ABORT_IF_NOT(active_.empty(),
                        "The outcomes are uniform only with an empty register");
        std::vector<std::size_t> slots;
        slots.reserve(wires.size());
        for (auto wire : wires) {
            slots.push_back(slotOf(wire));
        }
        return engine_->computationalSupport(slots);
    }

    /**
     * @brief The distribution of the computational-basis outcomes of
     * `wires`; bit k is the outcome of wires[k].
     *
     * The pulled-back Z of each wire is reduced over GF(2): combinations
     * with an X on a dormant qubit are fair and independent, combinations
     * that act on nothing are fixed, and the rest act on the register alone.
     * Those commute, so one Clifford on a copy of the register makes them
     * single Zs, and their joint distribution is a marginal of the copy.
     */
    [[nodiscard]] auto
    outcomeDistribution(const std::vector<QubitId> &wires) const
        -> OutcomeDistribution {
        if (active_.empty()) {
            return {computationalSupport(wires)};
        }
        PL_ABORT_IF_NOT(planned_width_ == engine_->numQubits(),
                        "Every lowered plan must run before an output");
        const std::size_t num_wires = wires.size();
        const std::size_t width = engine_->numQubits();
        const std::size_t bit_words = GF2::numBitWords(num_wires);

        // Row k is the pull-back of a product of wire Zs, and combos[k] says
        // which wires: its outcome is combos[k] . b for the outcomes b.
        std::vector<PauliWord> rows;
        std::vector<GF2::Bits> combos;
        for (std::size_t k = 0; k < num_wires; k++) {
            rows.push_back(
                engine_->pullBack(singleWord(slotOf(wires[k]), 'Z')));
            GF2::Bits combo(bit_words, 0);
            GF2::setBit(combo, k, true);
            combos.push_back(std::move(combo));
        }

        // Columns in pivot order: X on each dormant qubit, then X and Z on
        // each register qubit.
        std::vector<std::pair<std::size_t, bool>> columns;
        for (std::size_t slot = 0; slot < width; slot++) {
            if (!isActive(slot)) {
                columns.emplace_back(slot, false);
            }
        }
        const std::size_t num_dormant_columns = columns.size();
        for (auto slot : active_) {
            columns.emplace_back(slot, false);
        }
        for (auto slot : active_) {
            columns.emplace_back(slot, true);
        }

        enum class Kind : std::uint8_t { Fixed, Uniform, Dense };
        std::vector<Kind> kinds(num_wires, Kind::Fixed);
        std::size_t rank = 0;
        for (std::size_t c = 0; c < columns.size() && rank < num_wires; c++) {
            const auto [slot, is_z] = columns[c];
            auto has_bit = [&, slot = slot, is_z = is_z](const PauliWord &row) {
                return Pauli::getBit(is_z ? row.z() : row.x(), slot);
            };
            std::size_t pivot = rank;
            while (pivot < num_wires && !has_bit(rows[pivot])) {
                pivot++;
            }
            if (pivot == num_wires) {
                continue;
            }
            std::swap(rows[rank], rows[pivot]);
            std::swap(combos[rank], combos[pivot]);
            for (std::size_t r = 0; r < num_wires; r++) {
                if (r != rank && has_bit(rows[r])) {
                    rows[r] *= rows[rank];
                    GF2::xorInto(combos[r], combos[rank]);
                }
            }
            kinds[rank] =
                (c < num_dormant_columns) ? Kind::Uniform : Kind::Dense;
            rank++;
        }

        // b = T^-1 o for the matrix T of combos: invert T by Gauss-Jordan.
        std::vector<GF2::Bits> inverse(num_wires, GF2::Bits(bit_words, 0));
        for (std::size_t k = 0; k < num_wires; k++) {
            GF2::setBit(inverse[k], k, true);
        }
        {
            auto matrix = combos;
            for (std::size_t column = 0; column < num_wires; column++) {
                std::size_t pivot = column;
                while (pivot < num_wires &&
                       !GF2::getBit(matrix[pivot], column)) {
                    pivot++;
                }
                PL_ABORT_IF(pivot == num_wires,
                            "Internal error: the outcome map is singular");
                std::swap(matrix[column], matrix[pivot]);
                std::swap(inverse[column], inverse[pivot]);
                for (std::size_t r = 0; r < num_wires; r++) {
                    if (r != column && GF2::getBit(matrix[r], column)) {
                        GF2::xorInto(matrix[r], matrix[column]);
                        GF2::xorInto(inverse[r], inverse[column]);
                    }
                }
            }
        }
        auto inverseColumn = [&](std::size_t j) {
            GF2::Bits column(bit_words, 0);
            for (std::size_t i = 0; i < num_wires; i++) {
                GF2::setBit(column, i, GF2::getBit(inverse[i], j));
            }
            return column;
        };

        GF2::Bits offset(bit_words, 0);
        std::vector<GF2::Bits> uniform_columns;
        std::vector<GF2::Bits> dense_columns;
        std::vector<PauliWord> dense_words;
        for (std::size_t j = 0; j < num_wires; j++) {
            if (kinds[j] == Kind::Uniform) {
                uniform_columns.push_back(inverseColumn(j));
                continue;
            }
            PauliWord word = rows[j];
            PL_ABORT_IF_NOT(reduce(word).empty(),
                            "Internal error: a register outcome depends on a "
                            "dormant qubit");
            if (kinds[j] == Kind::Fixed) {
                if (word.logI() == 2) {
                    GF2::xorInto(offset, inverseColumn(j));
                }
                continue;
            }
            dense_columns.push_back(inverseColumn(j));
            dense_words.push_back(std::move(word));
        }

        OutcomeDistribution distribution{
            GF2::AffineSubspace(num_wires, std::move(offset), uniform_columns),
            std::move(dense_columns)};
        if (distribution.dense_columns.empty()) {
            return distribution;
        }

        auto copy = register_.clone();
        std::vector<PauliWord *> tracked;
        for (auto &word : dense_words) {
            tracked.push_back(&word);
        }
        auto apply = [&](const VirtualGate &gate) {
            for (const auto &op : gate) {
                copy.rotateQuarterTurns(densePauli(op.pauli), op.quarter_turns);
                for (auto *word : tracked) {
                    conjugate(*word, op.pauli, op.quarter_turns);
                }
            }
        };
        std::vector<std::size_t> used;
        std::vector<std::size_t> positions;
        for (auto &word : dense_words) {
            const std::size_t q = mapToZ(word, used, apply);
            used.push_back(q);
            positions.push_back(static_cast<std::size_t>(
                std::find(active_.begin(), active_.end(), q) -
                active_.begin()));
        }

        // Dense bit i is the bit of its Z, flipped when the Z is negated.
        std::uint64_t flips = 0;
        for (std::size_t i = 0; i < dense_words.size(); i++) {
            flips |= std::uint64_t{dense_words[i].logI() == 2} << i;
        }
        auto &table = distribution.dense_probabilities;
        table.assign(std::size_t{1} << dense_words.size(), 0.0);
        const auto amplitudes = copy.hostAmplitudes();
        for (std::size_t k = 0; k < amplitudes.size(); k++) {
            std::uint64_t outcome = 0;
            for (std::size_t i = 0; i < positions.size(); i++) {
                outcome |= ((k >> positions[i]) & 1U) << i;
            }
            const double weight = Kokkos::abs(amplitudes[k]);
            table[outcome ^ flips] += weight * weight;
        }
        return distribution;
    }
};

} // namespace Pennylane::LightningCliffish::Planner
