// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "TestHelpers.hpp"
#include "catch2/catch.hpp"
#include <algorithm>
#include <numeric>
#include <string>

#ifdef _ENABLE_PLQUBIT

constexpr bool BACKEND_FOUND = true;
#include "LightningSimulator.hpp"
using LSimulator = Catalyst::Runtime::Simulator::LightningSimulator;

#elif _ENABLE_PLKOKKOS == 1
constexpr bool BACKEND_FOUND = true;
#include "LightningKokkosSimulator.hpp"
using LSimulator = Catalyst::Runtime::Simulator::LightningKokkosSimulator;

#elif _ENABLE_PLGPU == 1
constexpr bool BACKEND_FOUND = true;
#include "LightningGPUSimulator.hpp"
using LSimulator = Catalyst::Runtime::Simulator::LightningGPUSimulator;

#else
constexpr bool BACKEND_FOUND = false;
using LSimulator = Pennylane::Util::TypeList<void>;
#endif

using namespace Catalyst::Runtime::Simulator;
using namespace Catalyst::Runtime;

TEST_CASE("Test parse_kwargs coverage", "[Utils]") {
    std::string case1;
    CHECK(parse_kwargs(case1).empty());

    std::string case2{"{my_attr : 1000}"};
    std::string case3{"my_attr : 1000"};
    std::string case4{"'my_attr':'1000'"};
    CHECK(parse_kwargs(case2) == parse_kwargs(case3));
    CHECK(parse_kwargs(case3) == parse_kwargs(case4));

    std::string case5{"{'A':'B', 'C':'D', 'E':'F'}"};
    auto res5 = parse_kwargs(case5);
    CHECK(res5.size() == 3);
    CHECK((res5.contains("A") && res5["A"] == "B"));
    CHECK((res5.contains("C") && res5["C"] == "D"));
    CHECK((res5.contains("E") && res5["E"] == "F"));

    std::string case6{
        "device_type : braket.aws.qubit,{'device_arn': 'sv1', "
        "'s3_destination_folder': \"('catalyst-op3-s3', 'prefix')\"}"};
    auto res6 = parse_kwargs(case6);
    CHECK(res6.size() == 3);
    CHECK((res6.contains("device_type") &&
           res6["device_type"] == "braket.aws.qubit"));
    CHECK((res6.contains("device_arn") && res6["device_arn"] == "sv1"));
    CHECK((res6.contains("s3_destination_folder") &&
           res6["s3_destination_folder"] == "('catalyst-op3-s3', 'prefix')"));
}

TEST_CASE("lightning Basis vector", "[Driver]") {
    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

    [[maybe_unused]] QubitIdType q1 = sim->AllocateQubit();
    [[maybe_unused]] QubitIdType q2 = sim->AllocateQubit();
    [[maybe_unused]] QubitIdType q3 = sim->AllocateQubit();

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(view(0).real() == Approx(1.0).epsilon(1e-5));
    CHECK(view(0).imag() == Approx(0.0).epsilon(1e-5));
    CHECK(view(1).real() == Approx(0.0).epsilon(1e-5));
    CHECK(view(1).imag() == Approx(0.0).epsilon(1e-5));
    CHECK(view(2).real() == Approx(0.0).epsilon(1e-5));
    CHECK(view(2).imag() == Approx(0.0).epsilon(1e-5));
    CHECK(view(3).real() == Approx(0.0).epsilon(1e-5));
    CHECK(view(3).imag() == Approx(0.0).epsilon(1e-5));
}

TEST_CASE("test AllocateQubits", "[Driver]") {
    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

    CHECK(sim->AllocateQubits(0).empty());

    auto &&q = sim->AllocateQubits(2);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state[0].real() == Approx(1.0).epsilon(1e-5));
}

TEST_CASE("test multiple AllocateQubits", "[Driver]") {
    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

    auto &&q1 = sim->AllocateQubits(2);
    CHECK(q1[0] == 0);
    CHECK(q1[1] == 1);

    auto &&q2 = sim->AllocateQubits(3);
    CHECK(q2.size() == 3);
    CHECK(q2[0] == 2);
    CHECK(q2[2] == 4);
}

TEST_CASE("test DeviceShots", "[Driver]") {
    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

    CHECK(sim->GetDeviceShots() == 0);

    sim->SetDeviceShots(500);

    CHECK(sim->GetDeviceShots() == 500);
}

TEST_CASE("compute register tests", "[Driver]") {
    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

    constexpr size_t n = 10;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    // allocate a few qubits
    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    // release some of them
    sim->ReleaseQubit(n - 1);
    sim->ReleaseQubit(n - 2);

    const size_t new_n = n - 2;

    // check the correctness
    std::vector<QubitIdType> Qs_expected(new_n);
    std::iota(Qs_expected.begin(), Qs_expected.end(),
              static_cast<QubitIdType>(0));

    for (size_t i = 0; i < new_n; i++) {
        CHECK(Qs_expected[i] == Qs[i]);
    }
}

TEST_CASE("Check an unsupported operation", "[Driver]") {
    REQUIRE_THROWS_WITH(
        Lightning::lookup_gates(Lightning::simulator_gate_info,
                                "UnsupportedGateName"),
        Catch::Contains(
            "The given operation is not supported by the simulator"));
}

TEST_CASE("Check re-AllocateQubit", "[Driver]") {
    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

    sim->AllocateQubit();
    sim->NamedOperation("Hadamard", {}, {0}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);
    CHECK(state[0].real() == Approx(0.707107).epsilon(1e-5));
    CHECK(state[1].real() == Approx(0.707107).epsilon(1e-5));

    sim->AllocateQubit();
    sim->AllocateQubit();
    sim->AllocateQubit();

    state = std::vector<std::complex<double>>(1U << sim->GetNumQubits());
    view = DataView<std::complex<double>, 1>(state);
    sim->State(view);
    CHECK(state[0].real() == Approx(0.707107).epsilon(1e-5));
    CHECK(state[8].real() == Approx(0.707107).epsilon(1e-5));
}

TEST_CASE("Check dynamic qubit reuse", "[Driver]") {
    SECTION("Two total device qubits with PauliX gate and deterministic "
            "measurements") {
        std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

        std::vector<intptr_t> Qs = sim->AllocateQubits(1);
        CHECK(Qs[0] == 0);
        std::vector<intptr_t> tempQs1 = sim->AllocateQubits(1);
        CHECK(tempQs1[0] == 1);
        sim->NamedOperation("PauliX", {}, {tempQs1[0]}, false);
        sim->ReleaseQubits(tempQs1);

        // Check that internal device state vector is still |01> after release
        std::vector<std::complex<double>> state(4);
        DataView<std::complex<double>, 1> view(state);
        sim->State(view);
        CHECK(state[0b00].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b10].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b01].real() == Approx(1.).epsilon(1e-5));
        CHECK(state[0b11].real() == Approx(0.).epsilon(1e-5));

        std::vector<intptr_t> tempQs2 = sim->AllocateQubits(1);
        // Check that program ID is different for every allocation
        CHECK(tempQs2[0] == 2);

        // Check that the second allocation reuses bit 1, and device sv resets
        // back to |00>
        sim->State(view);
        CHECK(state[0b00].real() == Approx(1.).epsilon(1e-5));
        CHECK(state[0b10].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b01].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b11].real() == Approx(0.).epsilon(1e-5));
        sim->ReleaseQubits(tempQs2);
    }

    SECTION("Two total device qubits with Hadamard gate and non deterministic "
            "measurements") {
        std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

        auto qubits = sim->AllocateQubits(2);
        sim->NamedOperation("PauliX", {}, {qubits[0]}, false);
        sim->NamedOperation("Hadamard", {}, {qubits[1]}, false);

        std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        sim->State(view);
        CHECK(state[0b00].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b10].real() == Approx(0.707107).epsilon(1e-5));
        CHECK(state[0b01].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b11].real() == Approx(0.707107).epsilon(1e-5));

        sim->ReleaseQubit(qubits[1]);

        sim->State(view);
        CHECK(state[0b00].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b10].real() == Approx(0.707107).epsilon(1e-5));
        CHECK(state[0b01].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b11].real() == Approx(0.707107).epsilon(1e-5));

        auto new_qubit = sim->AllocateQubit();
        CHECK(new_qubit != qubits[1]);

        sim->State(view);
        CHECK(state[0b00].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b10].real() == Approx(1.).epsilon(1e-5));
        CHECK(state[0b01].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b11].real() == Approx(0.).epsilon(1e-5));
    }

    SECTION("Multi qubit gates on dynamically and statically allocated qubits "
            "together") {
        std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();
        std::vector<std::complex<double>> state(16);
        DataView<std::complex<double>, 1> state_view(state);

        std::vector<intptr_t> Qs = sim->AllocateQubits(3); // |000>

        std::vector<intptr_t> tempQs1 = sim->AllocateQubits(1); // |000> and |0>
        sim->NamedOperation("PauliX", {}, {tempQs1[0]},
                            false); // |000> and |1>
        sim->NamedOperation("CNOT", {}, {tempQs1[0], Qs[1]},
                            false); // |010> and |1>

        sim->State(state_view);
        CHECK(state[0b0101].real() == Approx(1.).epsilon(1e-5));
        CHECK(std::accumulate(state.begin(), state.end(),
                              std::complex<double>{0.0, 0.0}) ==
              PLApproxComplex(std::complex<double>{1.0, 0.0}).epsilon(1e-5));

        sim->ReleaseQubits(tempQs1); // |010>

        std::vector<intptr_t> tempQs2 = sim->AllocateQubits(1); // |010> and |0>
        sim->State(state_view);
        CHECK(state[0b0100].real() == Approx(1.).epsilon(1e-5));
        CHECK(std::accumulate(state.begin(), state.end(),
                              std::complex<double>{0.0, 0.0}) ==
              PLApproxComplex(std::complex<double>{1.0, 0.0}).epsilon(1e-5));

        sim->NamedOperation("PauliX", {}, {tempQs2[0]},
                            false); // |010> and |1>
        sim->NamedOperation("CNOT", {}, {tempQs2[0], Qs[1]},
                            false);  // |000> and |1>
        sim->ReleaseQubits(tempQs2); // |000>

        std::vector<intptr_t> tempQs3 = sim->AllocateQubits(1); // |000> and |0>
        sim->State(state_view);
        CHECK(state[0b0000].real() == Approx(1.).epsilon(1e-5));
        CHECK(std::accumulate(state.begin(), state.end(),
                              std::complex<double>{0.0, 0.0}) ==
              PLApproxComplex(std::complex<double>{1.0, 0.0}).epsilon(1e-5));

        sim->NamedOperation("PauliX", {}, {tempQs3[0]},
                            false); // |000> and |1>
        sim->NamedOperation("CNOT", {}, {tempQs3[0], Qs[1]},
                            false);  // |010> and |1>
        sim->ReleaseQubits(tempQs3); // |010>

        sim->State(state_view);
        CHECK(state[0b0101].real() == Approx(1.).epsilon(1e-5));
        CHECK(std::accumulate(state.begin(), state.end(),
                              std::complex<double>{0.0, 0.0}) ==
              PLApproxComplex(std::complex<double>{1.0, 0.0}).epsilon(1e-5));

        std::vector<double> probs(8);
        DataView<double, 1> view(probs);
        sim->PartialProbs(view, Qs);

        std::vector<double> expected_probs = {0.0, 0.0, 1.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0};

        CHECK(probs == PLApprox(expected_probs).margin(1e-5));
    }
}

TEST_CASE("Release Qubits", "[Driver]") {
    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

    auto qubits = sim->AllocateQubits(4);

    sim->ReleaseQubits({qubits[1], qubits[2]});

    CHECK(sim->GetNumQubits() == 2);
}
