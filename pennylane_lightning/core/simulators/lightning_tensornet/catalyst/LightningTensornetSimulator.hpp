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
 * @file LightningTensornetSimulator.hpp
 * Catalyst runtime device for lightning.tensornet (Kokkos MPS / MPDO).
 *
 * Device kwargs (from the Python device): max_bond_dim (int), cutoff
 * (float), cutoff_mode ("abs" | "rel") and noise_model (serialized native
 * noise model). With a non-empty noise model the state is simulated as a
 * matrix product density operator from the start.
 */
#pragma once

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "MPSKokkos.hpp"
#include "NoisySimulator.hpp"

namespace Catalyst::Runtime::Simulator {

class LightningTensornetSimulator final
    : public NoisySimulator<Pennylane::LightningTensornet::MPSKokkos<double>> {
  public:
    using StateT = Pennylane::LightningTensornet::MPSKokkos<double>;

    explicit LightningTensornetSimulator(const std::string &kwargs = "{}")
        : NoisySimulator<StateT>(kwargs) {
        const auto &opts = getOptions();
        try {
            if (auto it = opts.find("max_bond_dim"); it != opts.end()) {
                const long long v = std::stoll(it->second);
                RT_FAIL_IF(v <= 0, "max_bond_dim must be a positive integer.");
                max_bond_dim_ = static_cast<std::size_t>(v);
            }
            if (auto it = opts.find("cutoff"); it != opts.end()) {
                cutoff_ = std::stod(it->second);
            }
        } catch (const std::logic_error &) {
            // std::invalid_argument / std::out_of_range from stoll / stod
            RT_FAIL("Invalid max_bond_dim or cutoff for lightning.tensornet.");
        }
        RT_FAIL_IF(cutoff_ < 0, "cutoff must be non-negative.");
        if (auto it = opts.find("cutoff_mode"); it != opts.end()) {
            cutoff_mode_ = it->second;
            RT_FAIL_IF(cutoff_mode_ != "abs" && cutoff_mode_ != "rel",
                       "cutoff_mode must be 'abs' or 'rel'.");
        }
        mixed_ = !getNoiseModel().empty();
    }
    ~LightningTensornetSimulator() override = default;

    LightningTensornetSimulator(const LightningTensornetSimulator &) = delete;
    LightningTensornetSimulator &
    operator=(const LightningTensornetSimulator &) = delete;
    LightningTensornetSimulator(LightningTensornetSimulator &&) = delete;
    LightningTensornetSimulator &
    operator=(LightningTensornetSimulator &&) = delete;

    [[nodiscard]] std::size_t getMaxBondDim() const { return max_bond_dim_; }
    [[nodiscard]] double getCutoff() const { return cutoff_; }
    [[nodiscard]] const std::string &getCutoffMode() const {
        return cutoff_mode_;
    }

  protected:
    [[nodiscard]] std::unique_ptr<StateT>
    makeState(std::size_t n) const override {
        return std::make_unique<StateT>(n, max_bond_dim_, cutoff_, cutoff_mode_,
                                        mixed_);
    }

  private:
    std::size_t max_bond_dim_{128};
    double cutoff_{0};
    std::string cutoff_mode_{"abs"};
    bool mixed_{false};
};

} // namespace Catalyst::Runtime::Simulator
