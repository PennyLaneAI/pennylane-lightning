// Copyright 2022-2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "LightningKokkosSimulator.hpp"
#include "catch2/catch.hpp"

/// @cond DEV
namespace {
// MemRef type definition (Helper)
template <typename T, size_t R> struct MemRefT {
    T *data_allocated;
    T *data_aligned;
    size_t offset;
    size_t sizes[R];
    size_t strides[R];
};
using namespace Catalyst::Runtime::Simulator;
using LKSimulator = LightningKokkosSimulator;
} // namespace
/// @endcond

TEST_CASE("Zero qubits. Zero parameters", "[Gradient]") {
    std::unique_ptr<LKSimulator> LKsim = std::make_unique<LKSimulator>();

    std::vector<DataView<double, 1>> gradients;
    std::vector<intptr_t> Qs = LKsim->AllocateQubits(0);
    REQUIRE_NOTHROW(LKsim->Gradient(gradients, {}));
}

TEST_CASE("Test Gradient with zero number of obs", "[Gradient]")
{
    std::unique_ptr<LKSimulator> sim = std::make_unique<LKSimulator>();

    std::vector<double> buffer(1);
    std::vector<DataView<double, 1>> gradients;
    gradients.emplace_back(buffer);

    std::vector<size_t> trainParams{0};

    const auto q = sim->AllocateQubit();

    sim->StartTapeRecording();

    sim->NamedOperation("S", {}, {q}, false);
    sim->NamedOperation("T", {}, {q}, false);
    
    REQUIRE_NOTHROW(sim->Gradient(gradients, trainParams));

    sim->StopTapeRecording();
}

TEST_CASE("Test Gradient with Var", "[Gradient]")
{
    std::unique_ptr<LKSimulator> sim = std::make_unique<LKSimulator>();

    std::vector<double> buffer(1);
    std::vector<DataView<double, 1>> gradients;
    gradients.emplace_back(buffer);

    std::vector<size_t> trainParams{0};

    const auto q = sim->AllocateQubit();

    sim->StartTapeRecording();

    sim->NamedOperation("RX", {-M_PI / 7}, {q}, false);
    auto pz = sim->Observable(ObsId::PauliZ, {}, {q});
    sim->Var(pz);

    REQUIRE_THROWS_WITH(sim->Gradient(gradients, trainParams),
        Catch::Contains("Unsupported measurements to compute gradient"));

    REQUIRE_THROWS_WITH(sim->Gradient(gradients, {}),
        Catch::Contains("Unsupported measurements to compute gradient"));

    sim->StopTapeRecording();
}

TEST_CASE("Test Gradient with Op=RX, Obs=Z", "[Gradient]")
{
    std::unique_ptr<LKSimulator> sim = std::make_unique<LKSimulator>();

    std::vector<double> buffer(1);
    std::vector<DataView<double, 1>> gradients;
    gradients.emplace_back(buffer);

    std::vector<size_t> trainParams{0};

    const auto q = sim->AllocateQubit();

    sim->StartTapeRecording();

    sim->NamedOperation("RX", {-M_PI / 7}, {q}, false);
    auto pz = sim->Observable(ObsId::PauliZ, {}, {q});
    sim->Expval(pz);

    sim->Gradient(gradients, trainParams);
    CHECK(-sin(-M_PI / 7) == Approx(buffer[0]).margin(1e-5));


    // Update buffer
    buffer[0] = 0.0;
    sim->Gradient(gradients, {});
    CHECK(-sin(-M_PI / 7) == Approx(buffer[0]).margin(1e-5));

    sim->StopTapeRecording();
}
