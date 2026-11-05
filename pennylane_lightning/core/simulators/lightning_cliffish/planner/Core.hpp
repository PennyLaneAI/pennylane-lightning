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
 * @file Core.hpp
 * The Lightning-Cliffish core: append ops, flush, and request outputs.
 */

#pragma once

#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "HybridMethod.hpp"
#include "Program.hpp"

namespace Pennylane::LightningCliffish {

/**
 * @brief The simulator core behind the device adapter.
 *
 * Ops are buffered as received. A flush lowers the buffered window to a plan
 * and runs it. A flush is forced when the program demands a value: a
 * measurement value, or an output.
 */
class Core {
  private:
    Program::Program program_{};
    Planner::HybridMethod method_{};
    Planner::Results results_{};
    std::mt19937 default_gen_{std::random_device{}()};
    std::mt19937 *gen_{&default_gen_};

  public:
    Core() = default;

    explicit Core(Planner::HybridMethod method) : method_{std::move(method)} {}

    /**
     * @brief Draw measurement outcomes from `gen`, or from the core's own
     * generator when `gen` is null.
     */
    void setPRNG(std::mt19937 *gen) {
        gen_ = (gen != nullptr) ? gen : &default_gen_;
    }

    auto allocate(std::size_t count) -> std::vector<Program::QubitId> {
        return program_.allocate(count);
    }

    void release(Program::QubitId qubit) { program_.release(qubit); }

    void append(Program::Op op) { program_.append(std::move(op)); }

    auto measure(Program::Op op) -> Program::ResultId {
        return program_.appendMeasurement(std::move(op));
    }

    [[nodiscard]] auto numQubits() const -> std::size_t {
        return program_.numQubits();
    }

    /**
     * @brief Simulate every op received so far.
     */
    void flush() {
        const auto plan = method_.lower(program_.takeWindow());
        method_.run(plan, *gen_, results_);
    }

    /**
     * @brief The value of a measurement result, flushing if it is not
     * simulated yet.
     */
    auto value(Program::ResultId result) -> bool {
        PL_ABORT_IF_NOT(program_.isResult(result),
                        "Unknown measurement result");
        if (!results_.contains(result)) {
            flush();
        }
        return results_.at(result);
    }

    /**
     * @brief Expectation value of the Pauli word with `pauli[k]` on
     * `wires[k]`. An output is a forced flush.
     */
    auto expectation(const std::string &pauli,
                     const std::vector<Program::QubitId> &wires) -> int {
        flush();
        return method_.expectation(pauli, wires);
    }
};

} // namespace Pennylane::LightningCliffish
