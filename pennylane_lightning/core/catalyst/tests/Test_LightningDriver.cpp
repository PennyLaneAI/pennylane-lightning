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

[[maybe_unused]]
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

        // Original state was |01>, but after releasing qubit 1, only qubit 0
        // remains
        std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
        DataView<std::complex<double>, 1> view(state);
        sim->State(view);
        CHECK(state.size() == 2);
        CHECK(state[0b0].real() == Approx(1.).epsilon(1e-5)); // |0>
        CHECK(state[0b1].real() == Approx(0.).epsilon(1e-5)); // |1>

        std::vector<intptr_t> tempQs2 = sim->AllocateQubits(1);
        // Check that program ID is different for every allocation
        CHECK(tempQs2[0] == 2);

        // After allocating new qubit, state vector expands back to 2 qubits
        // The new qubit is initialized to |0>, so state is |00>
        state.resize(1U << sim->GetNumQubits());
        DataView<std::complex<double>, 1> view_after_alloc(state);
        sim->State(view_after_alloc);
        CHECK(state.size() == 4);
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

        // Only qubit 0 remains and in |1> state
        state.resize(1U << sim->GetNumQubits());
        DataView<std::complex<double>, 1> view_after_release(state);
        sim->State(view_after_release);
        CHECK(state.size() == 2);
        CHECK(state[0b0].real() == Approx(0.).epsilon(1e-5)); // |0>
        CHECK(state[0b1].real() == Approx(1.).epsilon(1e-5)); // |1>

        auto new_qubit = sim->AllocateQubit();
        CHECK(new_qubit != qubits[1]);

        // After allocating new qubit, state vector expands back to 2 qubits
        // qubit 0 is |1>, new qubit is |0>, so state is |10>
        state.resize(1U << sim->GetNumQubits());

        DataView<std::complex<double>, 1> view_after_alloc(state);
        sim->State(view_after_alloc);
        CHECK(state.size() == 4);
        CHECK(state[0b00].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b10].real() == Approx(1.).epsilon(1e-5));
        CHECK(state[0b01].real() == Approx(0.).epsilon(1e-5));
        CHECK(state[0b11].real() == Approx(0.).epsilon(1e-5));
    }

    SECTION("Multi qubit gates on dynamically and statically allocated qubits "
            "together") {
        std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();
        std::vector<intptr_t> Qs = sim->AllocateQubits(3); // |000>

        std::vector<intptr_t> tempQs1 = sim->AllocateQubits(1); // |000> and |0>
        sim->NamedOperation("PauliX", {}, {tempQs1[0]},
                            false); // |000> and |1>
        sim->NamedOperation("CNOT", {}, {tempQs1[0], Qs[1]},
                            false); // |010> and |1>

        std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
        DataView<std::complex<double>, 1> state_view(state);
        sim->State(state_view);
        CHECK(state[0b0101].real() == Approx(1.).epsilon(1e-5));
        CHECK(std::accumulate(state.begin(), state.end(),
                              std::complex<double>{0.0, 0.0}) ==
              PLApproxComplex(std::complex<double>{1.0, 0.0}).epsilon(1e-5));

        sim->ReleaseQubits(tempQs1); // |010>

        std::vector<intptr_t> tempQs2 = sim->AllocateQubits(1); // |010> and |0>
        state.resize(1U << sim->GetNumQubits());
        DataView<std::complex<double>, 1> view_after_alloc_1(state);
        sim->State(view_after_alloc_1);
        CHECK(state.size() == 16); // 2^4 = 16
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
        state.resize(1U << sim->GetNumQubits());
        DataView<std::complex<double>, 1> view_after_alloc_2(state);
        sim->State(view_after_alloc_2);
        CHECK(state.size() == 16); // 2^4 = 16
        CHECK(state[0b0000].real() == Approx(1.).epsilon(1e-5));
        CHECK(std::accumulate(state.begin(), state.end(),
                              std::complex<double>{0.0, 0.0}) ==
              PLApproxComplex(std::complex<double>{1.0, 0.0}).epsilon(1e-5));

        sim->NamedOperation("PauliX", {}, {tempQs3[0]},
                            false); // |000> and |1>
        sim->NamedOperation("CNOT", {}, {tempQs3[0], Qs[1]},
                            false);  // |010> and |1>
        sim->ReleaseQubits(tempQs3); // |010>

        state.resize(1U << sim->GetNumQubits());
        DataView<std::complex<double>, 1> view_after_release_3(state);
        sim->State(view_after_release_3);
        CHECK(state[0b010].real() == Approx(1.).epsilon(1e-5));
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

TEST_CASE("Sample after dynamic qubit release", "[Driver]") {
    // This test mirrors the Python code:
    // @qjit
    // @qml.qnode(qml.device("lightning.qubit", wires=3, shots=10))
    // def circuit():
    //     with qml.allocate(2) as qs:
    //         qml.X(qs[1])
    //     return qml.sample(wires=[0, 1])

    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

    // Allocate 3 static qubits (wires 0, 1, 2) : all in |0>
    std::vector<intptr_t> static_qubits = sim->AllocateQubits(3);

    // Dynamically allocate 2 qubits
    std::vector<intptr_t> dynamic_qubits = sim->AllocateQubits(2);

    // Apply PauliX to dynamic_qubits[1]
    sim->NamedOperation("PauliX", {}, {dynamic_qubits[1]}, false);

    // Release the dynamic qubits
    sim->ReleaseQubits(dynamic_qubits);

    // Sample on static wires [0, 1]
    // Since static qubits were never modified, they should all be |0>
    constexpr size_t num_shots = 10;
    constexpr size_t num_wires = 2;
    sim->SetDeviceShots(num_shots);

    std::vector<double> samples(num_shots * num_wires);
    const size_t sizes[2] = {num_shots, num_wires};
    const size_t strides[2] = {num_wires, 1}; // row-major: stride[0]=num_wires
    DataView<double, 2> samples_view(samples.data(), 0, sizes, strides);

    sim->PartialSample(samples_view, {static_qubits[0], static_qubits[1]});

    for (size_t i = 0; i < num_shots * num_wires; i++) {
        CHECK(samples[i] == 0.);
    }
}

TEST_CASE("Sample after releasing middle qubit (triggers remap)", "[Driver]") {
    // Scenario:
    // 1. Allocate 3 qubits -> device IDs: 0, 1, 2
    // 2. Apply X to qubit 2 (device ID 2) -> state |001>
    // 3. Release qubit 1 (device ID 1) -> remaining device IDs: 0, 2
    // 4. reduceStateVector remaps: device ID 2 -> 1
    // 5. Sample qubit 2 (now device ID 1) -> should get |1>

    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

    // Allocate 3 qubits: device IDs 0, 1, 2
    std::vector<intptr_t> qubits = sim->AllocateQubits(3);

    // Apply X to qubit[2] (device ID 2): state becomes |001>
    sim->NamedOperation("PauliX", {}, {qubits[2]}, false);

    // Release qubit[1] (device ID 1), this creates a gap in device IDs
    // Remaining: qubit[0] (device 0), qubit[2] (device 2)
    // After reduceStateVector: qubit[0] -> device 0, qubit[2] -> device 1
    sim->ReleaseQubit(qubits[1]);

    // Sample on qubit[0] and qubit[2]
    // qubit[0] should be |0>, qubit[2] should be |1>
    constexpr size_t num_shots = 10;
    constexpr size_t num_wires = 2;
    sim->SetDeviceShots(num_shots);

    std::vector<double> samples(num_shots * num_wires);
    const size_t sizes[2] = {num_shots, num_wires};
    const size_t strides[2] = {num_wires, 1}; // row-major: stride[0]=num_wires
    DataView<double, 2> samples_view(samples.data(), 0, sizes, strides);

    sim->PartialSample(samples_view, {qubits[0], qubits[2]});

    // each shot should be [0, 1] (qubit[0]=0, qubit[2]=1)
    for (size_t shot = 0; shot < num_shots; shot++) {
        CHECK(samples[shot * num_wires + 0] == 0.); // qubit[0] is |0>
        CHECK(samples[shot * num_wires + 1] == 1.); // qubit[2] is |1>
    }
}

TEST_CASE("Cannot reuse entangled qubit when allocating", "[Driver]") {
    // 1. Allocate 2 qubits and entangle them
    // 2. Release one entangled qubit
    // 3. AllocateQubit() tries to reuse the released qubit -> should fail
    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();

    std::vector<intptr_t> qubits = sim->AllocateQubits(2);
    sim->NamedOperation("Hadamard", {}, {qubits[0]}, false);
    sim->NamedOperation("CNOT", {}, {qubits[0], qubits[1]}, false);

    sim->ReleaseQubit(qubits[1]);

    REQUIRE_THROWS_WITH(
        sim->AllocateQubit(),
        Catch::Contains("Cannot reuse qubit: qubit is entangled with remaining "
                        "qubits. Release qubits must be disentangled."));
}

TEST_CASE("Release one qubit from entangled qubits", "[Driver]") {
    // 1. Allocate 2 static qubits (wires 0, 1) and apply PauliX to each
    // 2. Dynamically allocate 2 qubits and apply CNOT to create entanglement
    // 3. Try to release only 1 entangled qubit
    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();
    std::vector<intptr_t> static_qubits = sim->AllocateQubits(2);

    sim->NamedOperation("PauliX", {}, {static_qubits[0]}, false);
    sim->NamedOperation("PauliX", {}, {static_qubits[1]}, false);

    std::vector<intptr_t> dynamic_qubits = sim->AllocateQubits(2);

    sim->NamedOperation("Hadamard", {}, {dynamic_qubits[0]}, false);
    sim->NamedOperation("CNOT", {}, {dynamic_qubits[0], dynamic_qubits[1]},
                        false);

    // Try to release only one entangled qubit
    sim->ReleaseQubit(dynamic_qubits[0]);

    constexpr size_t num_shots = 10;
    sim->SetDeviceShots(num_shots);

    std::vector<double> probs(4); // 2^2 for remaining 2 qubits
    DataView<double, 1> probs_view(probs);

    REQUIRE_THROWS_WITH(
        sim->PartialProbs(probs_view, {static_qubits[0], static_qubits[1]}),
        Catch::Contains("Cannot release qubits: released qubits are entangled "
                        "with remaining qubits"));
}

TEST_CASE("Release all entangled qubits", "[Driver]") {
    // 1. Allocate 2 static qubits (wires 0, 1) and apply PauliX to each
    // 2. Dynamically allocate 2 qubits and apply CNOT to create entanglement
    // 3. Release all entangled qubits
    std::unique_ptr<LSimulator> sim = std::make_unique<LSimulator>();
    std::vector<intptr_t> static_qubits = sim->AllocateQubits(2);

    sim->NamedOperation("PauliX", {}, {static_qubits[0]}, false); // |10>
    sim->NamedOperation("PauliX", {}, {static_qubits[1]}, false); // |11>

    std::vector<intptr_t> dynamic_qubits = sim->AllocateQubits(2); // |1100>

    // (|1100> + |1110>)/sqrt(2)
    sim->NamedOperation("Hadamard", {}, {dynamic_qubits[0]}, false);

    // (|1100> + |1111>)/sqrt(2)
    sim->NamedOperation("CNOT", {}, {dynamic_qubits[0], dynamic_qubits[1]},
                        false);

    // Release all entangled qubits
    sim->ReleaseQubits(dynamic_qubits);

    constexpr size_t num_shots = 10;
    sim->SetDeviceShots(num_shots);

    std::vector<double> probs(4); // 2^2 = 4 for 2 qubits
    DataView<double, 1> probs_view(probs);

    // <00|11><11|00> = 0
    // <01|11><11|01> = 0
    // <10|11><11|10> = 0
    // <11|11><11|11> = 1
    sim->PartialProbs(probs_view, {static_qubits[0], static_qubits[1]});
    CHECK(probs[0] == Approx(0.0).margin(1e-6));
    CHECK(probs[1] == Approx(0.0).margin(1e-6));
    CHECK(probs[2] == Approx(0.0).margin(1e-6));
    CHECK(probs[3] == Approx(1.0).margin(1e-6));
}
