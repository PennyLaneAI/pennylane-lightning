// Copyright 2022-2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "Exception.hpp"
#include "Types.h"

namespace Catalyst::Runtime::Simulator {

/**
 * Qubit Manager
 *
 * @brief Class to maintain a mapping from qubit IDs between the Catalyst
 * runtime (i.e. the program) and the device internals. Generally, program
 * qubits always receive new IDs upon allocation and never reuse the same ID.
 * This is done for safety to detect any use-after-free or other anomalies.
 *
 * Additionally, the class maintains a collection of qubits that are inactive
 * (i.e. have been freed) but whose memory is not physically released to the OS.
 * This mechanism is an optimization to allow efficient reuse of simulator
 * resources.
 */
template <typename ProgramQubitID = QubitIdType,
          typename DeviceQubitID = size_t>
class QubitManager final {
  private:
    std::unordered_map<ProgramQubitID, DeviceQubitID> qubit_id_map{};
    std::unordered_set<DeviceQubitID> free_device_qubits;
    ProgramQubitID next_program_id{0};

  public:
    QubitManager() = default;
    ~QubitManager() = default;

    QubitManager(const QubitManager &) = delete;
    QubitManager &operator=(const QubitManager &) = delete;
    QubitManager(QubitManager &&) = delete;
    QubitManager &operator=(QubitManager &&) = delete;

    auto getNumQubits() const -> size_t { return this->qubit_id_map.size(); }

    auto isValidQubitId(ProgramQubitID program_id) const -> bool {
        return this->qubit_id_map.contains(program_id);
    }

    auto isValidQubitId(const std::vector<ProgramQubitID> &program_ids) const
        -> bool {
        return std::all_of(program_ids.begin(), program_ids.end(),
                           [this](ProgramQubitID program_id) {
                               return isValidQubitId(program_id);
                           });
    }

    auto getDeviceId(ProgramQubitID program_id) const -> DeviceQubitID {
        RT_FAIL_IF(!isValidQubitId(program_id), "Invalid program qubit ID");

        return this->qubit_id_map.at(program_id);
    }

    auto getDeviceId(const std::vector<ProgramQubitID> &program_ids) const
        -> std::vector<DeviceQubitID> {
        std::vector<DeviceQubitID> device_ids;
        device_ids.reserve(program_ids.size());

        for (ProgramQubitID id : program_ids) {
            device_ids.push_back(getDeviceId(id));
        }

        return device_ids;
    }

    auto getProgramId(DeviceQubitID device_id) const -> ProgramQubitID {
        auto program_id_iter = std::find_if(
            this->qubit_id_map.begin(), this->qubit_id_map.end(),
            [&device_id](auto &&kv) { return kv.second == device_id; });

        RT_FAIL_IF(program_id_iter == this->qubit_id_map.end(),
                   "Invalid device qubit ID");

        return program_id_iter->first;
    }

    auto getProgramId(const std::vector<DeviceQubitID> &device_ids) const
        -> std::vector<ProgramQubitID> {
        std::vector<ProgramQubitID> program_ids;
        program_ids.reserve(program_ids.size());

        for (DeviceQubitID id : device_ids) {
            program_ids.push_back(getProgramId(id));
        }

        return program_ids;
    }

    auto getAllQubitIds() -> std::vector<ProgramQubitID> const {
        std::vector<ProgramQubitID> program_ids;
        program_ids.reserve(getNumQubits());

        for (const auto &it : this->qubit_id_map) {
            program_ids.push_back(it.first);
        }

        return program_ids;
    }

    auto popFreeQubit() -> std::optional<DeviceQubitID> {
        if (this->free_device_qubits.size() == 0) {
            return std::nullopt;
        }

        DeviceQubitID device_id = *this->free_device_qubits.begin();
        this->free_device_qubits.erase(device_id);

        return device_id;
    }

    auto Allocate(DeviceQubitID device_id) -> ProgramQubitID {
        this->qubit_id_map[this->next_program_id] = device_id;
        return this->next_program_id++;
    }

    auto AllocateRange(DeviceQubitID start_device_id, size_t num_qubits)
        -> std::vector<ProgramQubitID> {
        std::vector<ProgramQubitID> program_ids;
        program_ids.reserve(num_qubits);

        for (size_t i = 0; i < num_qubits; i++) {
            program_ids.push_back(this->next_program_id);
            this->qubit_id_map[this->next_program_id++] = start_device_id + i;
        }

        return program_ids;
    }

    void Release(ProgramQubitID program_id) {
        auto device_id_iter = this->qubit_id_map.find(program_id);
        RT_FAIL_IF(device_id_iter == this->qubit_id_map.end(),
                   "Cannot release qubit, the given ID is invalid");

        DeviceQubitID device_id = device_id_iter->second;
        this->qubit_id_map.erase(device_id_iter);
        this->free_device_qubits.insert(device_id);
    }

    void ReleaseAll() {
        this->qubit_id_map.clear();
        this->free_device_qubits.clear();
    }
};
} // namespace Catalyst::Runtime::Simulator
