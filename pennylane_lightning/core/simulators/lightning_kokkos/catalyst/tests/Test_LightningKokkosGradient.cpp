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
template <typename T, std::size_t R> struct MemRefT {
    T *data_allocated;
    T *data_aligned;
    std::size_t offset;
    std::size_t sizes[R];
    std::size_t strides[R];
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

TEST_CASE("Test Gradient with zero number of obs", "[Gradient]") {
    std::unique_ptr<LKSimulator> sim = std::make_unique<LKSimulator>();

    std::vector<double> buffer(1);
    std::vector<DataView<double, 1>> gradients;
    gradients.emplace_back(buffer);

    const std::vector<std::size_t> trainParams{0};

    const auto q = sim->AllocateQubit();

    sim->StartTapeRecording();

    sim->NamedOperation("S", {}, {q}, false);
    sim->NamedOperation("T", {}, {q}, false);

    REQUIRE_NOTHROW(sim->Gradient(gradients, trainParams));

    sim->StopTapeRecording();
}

TEST_CASE("Test Gradient with Var", "[Gradient]") {
    std::unique_ptr<LKSimulator> sim = std::make_unique<LKSimulator>();

    std::vector<double> buffer(1);
    std::vector<DataView<double, 1>> gradients;
    gradients.emplace_back(buffer);

    const std::vector<std::size_t> trainParams{0};

    const auto q = sim->AllocateQubit();

    sim->StartTapeRecording();

    sim->NamedOperation("RX", {-M_PI / 7}, {q}, false);
    auto pz = sim->Observable(ObsId::PauliZ, {}, {q});
    sim->Var(pz);

    REQUIRE_THROWS_WITH(
        sim->Gradient(gradients, trainParams),
        Catch::Contains("Unsupported measurements to compute gradient"));

    REQUIRE_THROWS_WITH(
        sim->Gradient(gradients, {}),
        Catch::Contains("Unsupported measurements to compute gradient"));

    sim->StopTapeRecording();
}

TEST_CASE("Test Gradient with Op=RX, Obs=Z", "[Gradient]") {
    std::unique_ptr<LKSimulator> sim = std::make_unique<LKSimulator>();

    std::vector<double> buffer(1);
    std::vector<DataView<double, 1>> gradients;
    gradients.emplace_back(buffer);

    const std::vector<std::size_t> trainParams{0};

    const auto q = sim->AllocateQubit();

    sim->StartTapeRecording();

    sim->NamedOperation("RX", {-M_PI / 7}, {q}, false);
    auto obs = sim->Observable(ObsId::PauliZ, {}, {q});
    sim->Expval(obs);

    sim->Gradient(gradients, trainParams);
    CHECK(-sin(-M_PI / 7) == Approx(buffer[0]).margin(1e-5));

    // Update buffer
    buffer[0] = 0.0;

    sim->Gradient(gradients, {});
    CHECK(-sin(-M_PI / 7) == Approx(buffer[0]).margin(1e-5));

    sim->StopTapeRecording();
}

TEST_CASE("Test Gradient with Op=RX, Obs=Hermitian", "[Gradient]") {
    std::unique_ptr<LKSimulator> sim = std::make_unique<LKSimulator>();

    std::vector<double> buffer(1);
    std::vector<DataView<double, 1>> gradients;
    gradients.emplace_back(buffer);

    const std::vector<std::size_t> trainParams{0};

    constexpr double expected{0.2169418696};

    const auto q = sim->AllocateQubit();

    sim->StartTapeRecording();

    sim->NamedOperation("RX", {-M_PI / 7}, {q}, false);

    std::vector<std::complex<double>> mat{
        {1.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}};

    auto obs = sim->Observable(ObsId::Hermitian, mat, {q});

    sim->Expval(obs);

    sim->Gradient(gradients, trainParams);
    CHECK(expected == Approx(buffer[0]).margin(1e-5));

    // Update buffer
    buffer[0] = 0.0;

    sim->Gradient(gradients, {});
    CHECK(expected == Approx(buffer[0]).margin(1e-5));

    sim->StopTapeRecording();
}

TEST_CASE("Test Gradient with Op=[RX,RX,RX,CZ], Obs=[Z,Z,Z]", "[Gradient]") {
    std::unique_ptr<LKSimulator> sim = std::make_unique<LKSimulator>();

    constexpr std::size_t num_parms = 3;

    std::vector<double> buffer_p0(num_parms);
    std::vector<double> buffer_p1(num_parms);
    std::vector<double> buffer_p2(num_parms);
    std::vector<DataView<double, 1>> gradients;
    gradients.emplace_back(buffer_p0);
    gradients.emplace_back(buffer_p1);
    gradients.emplace_back(buffer_p2);

    const std::vector<std::size_t> trainParams{0, 1, 2};

    const std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<double> expected{-sin(param[0]), -sin(param[1]),
                                       -sin(param[2])};

    const auto Qs = sim->AllocateQubits(num_parms);

    sim->StartTapeRecording();

    sim->NamedOperation("RX", {param[0]}, {Qs[0]}, false);
    sim->NamedOperation("RX", {param[1]}, {Qs[1]}, false);
    sim->NamedOperation("RX", {param[2]}, {Qs[2]}, false);
    sim->NamedOperation("CZ", {}, {Qs[0], Qs[2]}, false);

    std::vector<std::complex<double>> mat{
        {1.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}};

    auto obs0 = sim->Observable(ObsId::PauliZ, {}, {Qs[0]});
    auto obs1 = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    auto obs2 = sim->Observable(ObsId::PauliZ, {}, {Qs[2]});

    sim->Expval(obs0);
    sim->Expval(obs1);
    sim->Expval(obs2);

    sim->Gradient(gradients, trainParams);
    CHECK(expected[0] == Approx(buffer_p0[0]).margin(1e-5));
    CHECK(expected[1] == Approx(buffer_p1[1]).margin(1e-5));
    CHECK(expected[2] == Approx(buffer_p2[2]).margin(1e-5));

    sim->StopTapeRecording();
}

TEST_CASE("Test Gradient with Op=Mixed, Obs=Hamiltonian([Z@Z, H], {0.2, 0.6})",
          "[Gradient]") {
    std::unique_ptr<LKSimulator> sim = std::make_unique<LKSimulator>();

    constexpr std::size_t num_parms = 6;

    std::vector<double> buffer(num_parms);
    std::vector<DataView<double, 1>> gradients;
    gradients.emplace_back(buffer);

    const std::vector<std::size_t> trainParams{0, 1, 2, 3, 4, 5};

    const std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<double> expected{0.0, -0.2493761627, 0.0,
                                       0.0, -0.1175570505, 0.0};

    const auto Qs = sim->AllocateQubits(3);

    sim->StartTapeRecording();

    sim->NamedOperation("RZ", {param[0]}, {Qs[0]}, false);
    sim->NamedOperation("RY", {param[1]}, {Qs[0]}, false);
    sim->NamedOperation("RZ", {param[2]}, {Qs[0]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[1], Qs[2]}, false);
    sim->NamedOperation("RZ", {param[0]}, {Qs[1]}, false);
    sim->NamedOperation("RY", {param[1]}, {Qs[1]}, false);
    sim->NamedOperation("RZ", {param[2]}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat{
        {1.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}};

    auto obs0 = sim->Observable(ObsId::PauliZ, {}, {Qs[0]});
    auto obs1 = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    auto obs2 = sim->TensorObservable({obs0, obs1});
    auto obs3 = sim->Observable(ObsId::Hadamard, {}, {Qs[2]});
    auto obs4 = sim->HamiltonianObservable({0.2, 0.6}, {obs2, obs3});

    sim->Expval(obs4);

    sim->Gradient(gradients, trainParams);

    for (std::size_t i = 0; i < num_parms; i++) {
        CAPTURE(i);
        CHECK(expected[i] == Approx(buffer[i]).margin(1e-5));
        buffer[i] = 0.0;
    }

    sim->Gradient(gradients, {});

    for (std::size_t i = 0; i < num_parms; i++) {
        CAPTURE(i);
        CHECK(expected[i] == Approx(buffer[i]).margin(1e-5));
    }

    sim->StopTapeRecording();
}

TEST_CASE("Test Gradient with QubitUnitary", "[Gradient]") {
    std::unique_ptr<LKSimulator> sim = std::make_unique<LKSimulator>();

    std::vector<double> buffer(1);
    std::vector<DataView<double, 1>> gradients;
    gradients.emplace_back(buffer);

    const std::vector<std::size_t> trainParams{0};

    constexpr double expected{-0.8611041863};

    const std::vector<std::complex<double>> matrix{
        {-0.6709485262524046, -0.6304426335363695},
        {-0.14885403153998722, 0.3608498832392019},
        {-0.2376311670004963, 0.3096798175687841},
        {-0.8818365947322423, -0.26456390390903695},
    };

    const auto Qs = sim->AllocateQubits(1);

    sim->StartTapeRecording();

    sim->NamedOperation("RX", {-M_PI / 7}, {Qs[0]}, false);
    sim->MatrixOperation(matrix, {Qs[0]}, false);

    auto obs = sim->Observable(ObsId::PauliY, {}, {Qs[0]});
    sim->Expval(obs);

    sim->Gradient(gradients, trainParams);
    CHECK(expected == Approx(buffer[0]).margin(1e-5));

    // Update buffer
    buffer[0] = 0.0;

    sim->Gradient(gradients, {});
    CHECK(expected == Approx(buffer[0]).margin(1e-5));

    sim->StopTapeRecording();
}
