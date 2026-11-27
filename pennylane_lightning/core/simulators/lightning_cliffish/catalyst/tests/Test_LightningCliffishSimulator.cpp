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

#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "LightningCliffishSimulator.hpp"
#include "QuantumDevice.hpp"

namespace {
using namespace Catalyst::Runtime::Simulator;
using LCSimulator = LightningCliffishSimulator;
using QDevice = Catalyst::Runtime::QuantumDevice;

GENERATE_DEVICE_FACTORY(
    LightningCliffishSimulator,
    Catalyst::Runtime::Simulator::LightningCliffishSimulator);
} // namespace

TEST_CASE("LightningCliffishSimulator::constructor", "[constructibility]") {
    SECTION("LightningCliffishSimulator") {
        REQUIRE(std::is_constructible<LCSimulator>::value);
    }
    SECTION("LightningCliffishSimulator(string)") {
        REQUIRE(std::is_constructible<LCSimulator, std::string>::value);
    }
}

TEST_CASE("Test the device factory method", "[constructibility]") {
    std::unique_ptr<QDevice> LCsim(LightningCliffishSimulatorFactory(""));
    REQUIRE(LCsim->GetNumQubits() == 0);
}

TEST_CASE("LightningCliffishSimulator::qubit management", "[qubits]") {
    std::unique_ptr<LCSimulator> LCsim = std::make_unique<LCSimulator>();

    auto Qs = LCsim->AllocateQubits(3);
    REQUIRE(Qs == std::vector<QubitIdType>{0, 1, 2});
    REQUIRE(LCsim->GetNumQubits() == 3);

    auto q = LCsim->AllocateQubit();
    REQUIRE(q == 3);
    REQUIRE(LCsim->GetNumQubits() == 4);

    LCsim->ReleaseQubit(q);
    REQUIRE(LCsim->GetNumQubits() == 3);

    LCsim->ReleaseQubits(Qs);
    REQUIRE(LCsim->GetNumQubits() == 0);

    REQUIRE_THROWS_WITH(LCsim->ReleaseQubit(q),
                        Catch::Matchers::ContainsSubstring("invalid"));
}

TEST_CASE("LightningCliffishSimulator::shots and PRNG", "[shots]") {
    std::unique_ptr<LCSimulator> LCsim = std::make_unique<LCSimulator>();
    REQUIRE(LCsim->GetDeviceShots() == 0);

    LCsim->SetDeviceShots(500);
    REQUIRE(LCsim->GetDeviceShots() == 500);

    std::mt19937 gen{37};
    REQUIRE_NOTHROW(LCsim->SetDevicePRNG(&gen));
}

TEST_CASE("LightningCliffishSimulator::operations fail loudly",
          "[unsupported]") {
    std::unique_ptr<LCSimulator> LCsim = std::make_unique<LCSimulator>();
    auto Qs = LCsim->AllocateQubits(2);

    REQUIRE_THROWS_WITH(
        LCsim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}),
        Catch::Matchers::ContainsSubstring("cannot apply 'CNOT'"));
    REQUIRE_THROWS_WITH(LCsim->Measure(Qs[0]),
                        Catch::Matchers::ContainsSubstring("cannot measure"));
    REQUIRE_THROWS_WITH(
        LCsim->PauliMeasure("ZZ", {Qs[0], Qs[1]}),
        Catch::Matchers::ContainsSubstring("PauliMeasure is unsupported"));
}
