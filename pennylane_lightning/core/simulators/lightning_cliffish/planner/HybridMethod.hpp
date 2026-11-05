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
 * The hybrid method: a tableau plus a dense register. The dense register is
 * not implemented yet, so this is the hybrid method with an empty register,
 * which is exact for Clifford programs.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Error.hpp"
#include "GateExpansions.hpp"
#include "NativeTableau.hpp"
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
using PlanStep =
    std::variant<Tableau::EngineBlock, AddQubitsStep, MeasureStep, ResetStep>;

using Plan = std::vector<PlanStep>;

/**
 * @brief Measurement outcomes by result id.
 */
using Results = std::unordered_map<ResultId, bool>;

class HybridMethod {
  private:
    std::unique_ptr<Tableau::TableauEngine> engine_;

    /// Engine qubit of each live program qubit.
    std::unordered_map<QubitId, std::size_t> slots_{};
    /// Engine qubits in |0> that no program qubit holds.
    std::vector<std::size_t> free_slots_{};
    /// Engine width once every plan lowered so far has run.
    std::size_t planned_width_{0};

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

  public:
    HybridMethod() : engine_{std::make_unique<Tableau::NativeTableau<>>()} {}

    explicit HybridMethod(std::unique_ptr<Tableau::TableauEngine> engine)
        : engine_{std::move(engine)} {
        PL_ABORT_IF(engine_->numQubits() != 0,
                    "The hybrid method needs an engine with no qubits");
    }

    [[nodiscard]] auto engine() const -> const Tableau::TableauEngine & {
        return *engine_;
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
                    current_block().push_back(Tableau::EngineOp::rotation(
                        engineWord(rotation.pauli, op.wires),
                        rotation.quarter_turns));
                }
                break;
            case OpKind::Measure:
                plan.emplace_back(MeasureStep{engineWord("Z", op.wires),
                                              op.result, op.postselect});
                break;
            case OpKind::PauliMeasure:
                plan.emplace_back(
                    MeasureStep{engineWord(op.pauli_word, op.wires), op.result,
                                op.postselect});
                break;
            }
        }
        return plan;
    }

    /**
     * @brief Run a plan, writing measurement outcomes to `results`.
     */
    void run(const Plan &plan, std::mt19937 &gen, Results &results) {
        for (const auto &step : plan) {
            if (const auto *block = std::get_if<Tableau::EngineBlock>(&step)) {
                engine_->applyBlock(*block, gen);
            } else if (const auto *grow = std::get_if<AddQubitsStep>(&step)) {
                engine_->addQubits(grow->count);
            } else if (const auto *measure = std::get_if<MeasureStep>(&step)) {
                const auto outcomes = engine_->applyBlock(
                    {Tableau::EngineOp::measurement(measure->pauli,
                                                    measure->forced_outcome)},
                    gen);
                results[measure->result] = outcomes[0];
            } else {
                const auto &reset = std::get<ResetStep>(step);
                PauliWord z(engine_->numQubits());
                z.set(reset.slot, 'Z');
                const auto outcomes = engine_->applyBlock(
                    {Tableau::EngineOp::measurement(z)}, gen);
                if (outcomes[0]) {
                    PauliWord x(engine_->numQubits());
                    x.set(reset.slot, 'X');
                    engine_->applyBlock({Tableau::EngineOp::rotation(x, 2)},
                                        gen);
                }
            }
        }
    }

    /**
     * @brief Expectation of the Pauli word with `pauli[k]` on `wires[k]`.
     */
    [[nodiscard]] auto expectation(const std::string &pauli,
                                   const std::vector<QubitId> &wires) const
        -> int {
        PL_ABORT_IF_NOT(pauli.size() == wires.size(),
                        "The Pauli word must have one character per wire");
        PL_ABORT_IF_NOT(planned_width_ == engine_->numQubits(),
                        "Every lowered plan must run before an output");
        return engine_->expectation(engineWord(pauli, wires));
    }
};

} // namespace Pennylane::LightningCliffish::Planner
