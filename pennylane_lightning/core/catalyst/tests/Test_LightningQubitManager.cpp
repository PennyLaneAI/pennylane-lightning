// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <optional>
#include <set>

#include "catch2/catch.hpp"

#include "LightningQubitManager.hpp"

using namespace Catalyst::Runtime::Simulator;

TEST_CASE("Test lightning qubit manager initilization", "[Utils]") {
    QubitManager<QubitIdType, size_t> manager{};
}

TEST_CASE("Test lightning qubit manager allocate and release", "[Utils]") {
    QubitManager<QubitIdType, size_t> manager{};
    size_t device_qubit_id = 0;

    QubitIdType q = manager.Allocate(device_qubit_id);
    CHECK(manager.getNumQubits() == 1);
    CHECK(manager.isValidQubitId(q));
    CHECK(manager.getProgramId(device_qubit_id) == q);
    CHECK(manager.getDeviceId(q) == device_qubit_id);

    manager.Release(q);
    CHECK(manager.getNumQubits() == 0);
    CHECK(!manager.isValidQubitId(q));
}

TEST_CASE("Test lightning qubit manager batched allocate and release",
          "[Utils]") {
    QubitManager<QubitIdType, size_t> manager{};
    std::vector<size_t> device_qubit_ids = {0, 1, 2, 3};
    size_t start_device_qubit = device_qubit_ids[0];
    size_t num_qubits = device_qubit_ids.size();

    std::vector<QubitIdType> qs =
        manager.AllocateRange(start_device_qubit, num_qubits);
    CHECK(manager.getNumQubits() == num_qubits);
    CHECK(manager.isValidQubitId(qs));

    for (size_t i = 0; i < num_qubits; i++) {
        CHECK(manager.getProgramId(device_qubit_ids[i]) == qs[i]);
        CHECK(manager.getDeviceId(qs[i]) == device_qubit_ids[i]);
    }

    std::vector<QubitIdType> programIds = manager.getAllQubitIds();
    CHECK(std::set(programIds.begin(), programIds.end()) ==
          std::set(qs.begin(), qs.end()));

    for (QubitIdType q : qs) {
        manager.Release(q);
    }
    CHECK(manager.getNumQubits() == 0);
    CHECK(!manager.isValidQubitId(qs));
}

TEST_CASE("Test lightning qubit manager release all", "[Utils]") {
    QubitManager<QubitIdType, size_t> manager{};

    std::vector<QubitIdType> qs = manager.AllocateRange(0, 100);
    CHECK(manager.getNumQubits() == 100);

    manager.ReleaseAll();
    CHECK(manager.getNumQubits() == 0);
    CHECK(!manager.isValidQubitId(qs));
}

TEST_CASE("Test lightning qubit manager pop free qubit", "[Utils]") {
    QubitManager<QubitIdType, size_t> manager{};

    std::vector<QubitIdType> qs = manager.AllocateRange(0, 100);
    CHECK(manager.getNumQubits() == 100);
    CHECK(manager.popFreeQubit() == std::nullopt);

    size_t free_target_device_id = manager.getDeviceId(qs[0]);
    manager.Release(qs[0]);
    CHECK(manager.getNumQubits() == 99);
    CHECK(manager.popFreeQubit() == free_target_device_id);
}

TEST_CASE("Test lightning qubit manager invalid IDs", "[Utils]") {
    QubitManager<QubitIdType, size_t> manager{};
    size_t device_qubit_id = 0;

    QubitIdType q = manager.Allocate(device_qubit_id);
    REQUIRE_THROWS_WITH(manager.getProgramId(device_qubit_id + 42),
                        Catch::Contains("Invalid device qubit ID"));
    REQUIRE_THROWS_WITH(manager.getDeviceId(q + 42),
                        Catch::Contains("Invalid program qubit ID"));

    REQUIRE_THROWS_WITH(
        manager.Release(q + 42),
        Catch::Contains("Cannot release qubit, the given ID is invalid"));
}
