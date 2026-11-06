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

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "AffineSubspace.hpp"
#include "Error.hpp"
#include "HybridMethod.hpp"
#include "Observables.hpp"
#include "OutcomeDistribution.hpp"
#include "Program.hpp"

namespace Pennylane::LightningCliffish {

/**
 * @brief The simulator core behind the device adapter.
 *
 * Ops are buffered as received. A flush lowers the buffered window to a plan
 * and runs it. A flush is forced when the program demands a value: a
 * measurement value, or an output. Before an output, the hybrid method runs
 * its end check.
 */
class Core {
  private:
    Program::Program program_{};
    Planner::HybridMethod method_{};
    Planner::Results results_{};
    std::mt19937 default_gen_{std::random_device{}()};
    /// The device's generator, or null to use default_gen_.
    std::mt19937 *gen_{nullptr};

    auto generator() -> std::mt19937 & {
        return (gen_ != nullptr) ? *gen_ : default_gen_;
    }

    /// A 64-bit seed for one output, drawn from the core's generator.
    auto drawSeed() -> std::uint64_t {
        auto &gen = generator();
        return (static_cast<std::uint64_t>(gen()) << 32U) |
               static_cast<std::uint64_t>(gen());
    }

    /// Mean of `shots` outcomes of a Pauli word with exact expectation
    /// `exact`.
    static auto sampledMean(double exact, std::size_t shots, std::uint64_t seed)
        -> double {
        if (exact == 1.0 || exact == -1.0) {
            return exact;
        }
        const auto num_shots = static_cast<double>(shots);
        std::uint64_t state = seed;
        if (exact == 0.0) {
            // Each outcome is +1 or -1 with probability 1/2.
            std::size_t ones = 0;
            for (std::size_t done = 0; done < shots; done += 64) {
                std::uint64_t bits = GF2::splitMix64(state);
                if (shots - done < 64) {
                    bits &= (std::uint64_t{1} << (shots - done)) - 1;
                }
                ones += static_cast<std::size_t>(std::popcount(bits));
            }
            return (num_shots - 2.0 * static_cast<double>(ones)) / num_shots;
        }
        // Each outcome is +1 with probability (1 + exact) / 2.
        const double plus_probability = (1.0 + exact) / 2;
        std::size_t plus = 0;
        for (std::size_t shot = 0; shot < shots; shot++) {
            const double u =
                static_cast<double>(GF2::splitMix64(state) >> 11U) * 0x1.0p-53;
            plus += (u < plus_probability) ? 1 : 0;
        }
        return (2.0 * static_cast<double>(plus) - num_shots) / num_shots;
    }

    /// Flush, and run the check that precedes an output.
    void flushForOutput() {
        flush();
        method_.prepareOutput();
    }

  public:
    Core() = default;

    explicit Core(Planner::HybridOptions options) : method_{options} {}

    explicit Core(Planner::HybridMethod method) : method_{std::move(method)} {}

    /// What the hybrid method did so far.
    [[nodiscard]] auto report() const -> Planner::HybridReport {
        return method_.report();
    }

    /**
     * @brief Draw measurement outcomes from `gen`, or from the core's own
     * generator when `gen` is null.
     */
    void setPRNG(std::mt19937 *gen) { gen_ = gen; }

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
        method_.run(plan, generator(), results_);
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
                     const std::vector<Program::QubitId> &wires) -> double {
        flushForOutput();
        return method_.expectation(pauli, wires);
    }

    // ------------------------------------------------------------------
    // Outputs. Each one is a forced flush, and the program may continue.
    // ------------------------------------------------------------------

    /// Exact expectation value of `observable`.
    auto expval(const Planner::Observable &observable) -> double {
        flushForOutput();
        return method_.expval(observable);
    }

    /// Exact variance of `observable`.
    auto var(const Planner::Observable &observable) -> double {
        flushForOutput();
        return method_.variance(observable);
    }

    /**
     * @brief Estimate the expectation value of `observable` from `shots`
     * measurements of each of its terms.
     */
    auto expval(const Planner::Observable &observable, std::size_t shots)
        -> double {
        PL_ABORT_IF(shots == 0,
                    "A sampled expectation needs at least one shot");
        flushForOutput();
        double value = 0.0;
        for (const auto &term : observable) {
            const double exact = method_.expectation(term.pauli, term.wires);
            value += term.coefficient * sampledMean(exact, shots, drawSeed());
        }
        return value;
    }

    /**
     * @brief Estimate the variance of a single Pauli term from `shots`
     * measurements.
     */
    auto var(const Planner::Observable &observable, std::size_t shots)
        -> double {
        PL_ABORT_IF(shots == 0, "A sampled variance needs at least one shot");
        PL_ABORT_IF_NOT(observable.size() == 1,
                        "A sampled variance needs an observable with one "
                        "Pauli term");
        flushForOutput();
        const auto &term = observable.front();
        const double mean = sampledMean(
            method_.expectation(term.pauli, term.wires), shots, drawSeed());
        // The outcomes are +c or -c, so their variance is c^2 (1 - mean^2).
        return term.coefficient * term.coefficient * (1.0 - mean * mean);
    }

    /**
     * @brief `shots` computational-basis samples of `wires`; bit k of a shot is
     * the outcome of wires[k]. The state is unchanged.
     */
    auto sample(const std::vector<Program::QubitId> &wires, std::size_t shots)
        -> GF2::Samples {
        flushForOutput();
        return Planner::sample(method_.outcomeDistribution(wires), shots,
                               drawSeed());
    }

    /**
     * @brief Shots of each outcome index of `wires`, with wires[0] the most
     * significant bit.
     */
    auto counts(const std::vector<Program::QubitId> &wires, std::size_t shots)
        -> std::map<std::uint64_t, std::size_t> {
        PL_ABORT_IF(wires.size() > 64, "Counts hold at most 64 wires");
        return GF2::counts(sample(wires, shots));
    }

    /**
     * @brief Exact probability of each outcome index of `wires`, with
     * wires[0] the most significant bit.
     */
    auto probabilities(const std::vector<Program::QubitId> &wires)
        -> std::vector<double> {
        flushForOutput();
        return Planner::probabilities(method_.outcomeDistribution(wires));
    }
};

} // namespace Pennylane::LightningCliffish
