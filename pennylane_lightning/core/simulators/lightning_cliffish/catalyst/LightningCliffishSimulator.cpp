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

#include "LightningCliffishSimulator.hpp"

#include "Exception.hpp"

namespace Catalyst::Runtime::Simulator {

auto LightningCliffishSimulator::AllocateQubit() -> QubitIdType {
    return this->qubit_manager.Allocate(this->next_device_id++);
}

auto LightningCliffishSimulator::AllocateQubits(std::size_t num_qubits)
    -> std::vector<QubitIdType> {
    auto ids =
        this->qubit_manager.AllocateRange(this->next_device_id, num_qubits);
    this->next_device_id += num_qubits;
    return ids;
}

void LightningCliffishSimulator::ReleaseQubit(QubitIdType q) {
    this->qubit_manager.Release(q);
}

void LightningCliffishSimulator::ReleaseQubits(
    const std::vector<QubitIdType> &ids) {
    for (auto id : ids) {
        this->qubit_manager.Release(id);
    }
}

auto LightningCliffishSimulator::GetNumQubits() const -> std::size_t {
    return this->qubit_manager.getNumQubits();
}

void LightningCliffishSimulator::SetDeviceShots(std::size_t shots) {
    this->device_shots = shots;
}

auto LightningCliffishSimulator::GetDeviceShots() const -> std::size_t {
    return this->device_shots;
}

void LightningCliffishSimulator::SetDevicePRNG(std::mt19937 *gen) {
    this->gen = gen;
}

void LightningCliffishSimulator::NamedOperation(
    const std::string &name, [[maybe_unused]] const std::vector<double> &params,
    [[maybe_unused]] const std::vector<QubitIdType> &wires,
    [[maybe_unused]] bool inverse,
    [[maybe_unused]] const std::vector<QubitIdType> &controlled_wires,
    [[maybe_unused]] const std::vector<bool> &controlled_values,
    [[maybe_unused]] const std::vector<std::string> &optional_params) {
    const std::string message = "lightning.cliffish cannot apply '" + name +
                                "' yet: the simulator core is not implemented";
    RT_FAIL(message.c_str());
}

auto LightningCliffishSimulator::Measure(
    [[maybe_unused]] QubitIdType wire,
    [[maybe_unused]] std::optional<int32_t> postselect) -> Result {
    RT_FAIL("lightning.cliffish cannot measure yet: the simulator core is not "
            "implemented");
}

} // namespace Catalyst::Runtime::Simulator

/// LCOV_EXCL_START
GENERATE_DEVICE_FACTORY(
    LightningCliffishSimulator,
    Catalyst::Runtime::Simulator::LightningCliffishSimulator);
/// LCOV_EXCL_STOP
