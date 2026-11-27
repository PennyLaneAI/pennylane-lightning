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
 * @file LightningCliffishSimulator.hpp
 */

#pragma once

#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "LightningQubitManager.hpp"
#include "QuantumDevice.hpp"

namespace Catalyst::Runtime::Simulator {
/**
 * @brief Catalyst device for lightning.cliffish.
 *
 * The device manages qubits, shots and the PRNG. Every operation and
 * measurement fails loudly until the simulator core is implemented.
 * More info:
 * https://github.com/PennyLaneAI/catalyst/blob/main/runtime/include/QuantumDevice.hpp
 */
class LightningCliffishSimulator final
    : public Catalyst::Runtime::QuantumDevice {
  private:
    QubitManager<QubitIdType, std::size_t> qubit_manager{};
    std::size_t next_device_id{0};

    std::size_t device_shots{0};

    std::mt19937 *gen{nullptr};

  public:
    explicit LightningCliffishSimulator(
        [[maybe_unused]] const std::string &kwargs = "{}") noexcept {}
    ~LightningCliffishSimulator() noexcept = default;

    LightningCliffishSimulator(const LightningCliffishSimulator &) = delete;
    LightningCliffishSimulator &
    operator=(const LightningCliffishSimulator &) = delete;
    LightningCliffishSimulator(LightningCliffishSimulator &&) = delete;
    LightningCliffishSimulator &
    operator=(LightningCliffishSimulator &&) = delete;

    auto AllocateQubit() -> QubitIdType override;
    auto AllocateQubits(std::size_t num_qubits)
        -> std::vector<QubitIdType> override;
    void ReleaseQubit(QubitIdType q) override;
    void ReleaseQubits(const std::vector<QubitIdType> &ids) override;
    [[nodiscard]] auto GetNumQubits() const -> std::size_t override;
    void SetDeviceShots(std::size_t shots) override;
    [[nodiscard]] auto GetDeviceShots() const -> std::size_t override;
    void SetDevicePRNG(std::mt19937 *gen) override;

    void NamedOperation(
        const std::string &name, const std::vector<double> &params,
        const std::vector<QubitIdType> &wires, bool inverse = false,
        const std::vector<QubitIdType> &controlled_wires = {},
        const std::vector<bool> &controlled_values = {},
        const std::vector<std::string> &optional_params = {}) override;
    auto Measure(QubitIdType wire,
                 std::optional<int32_t> postselect = std::nullopt)
        -> Result override;
};

} // namespace Catalyst::Runtime::Simulator
