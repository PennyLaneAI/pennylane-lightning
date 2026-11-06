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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
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
using Catch::Matchers::ContainsSubstring;
using LCSimulator = LightningCliffishSimulator;
using QDevice = Catalyst::Runtime::QuantumDevice;

GENERATE_DEVICE_FACTORY(
    LightningCliffishSimulator,
    Catalyst::Runtime::Simulator::LightningCliffishSimulator);

/// A 2-D view over `buffer`, shaped rows x cols.
auto view2d(std::vector<double> &buffer, std::size_t rows, std::size_t cols)
    -> DataView<double, 2> {
    const std::size_t sizes[2] = {rows, cols};
    const std::size_t strides[2] = {cols, 1};
    return DataView<double, 2>(buffer.data(), 0, sizes, strides);
}

/// A device in the Bell state (|00> + |11>) / sqrt(2), with `shots` shots.
auto bellDevice(std::mt19937 &gen, std::size_t shots = 0)
    -> std::unique_ptr<LCSimulator> {
    auto device = std::make_unique<LCSimulator>();
    device->SetDevicePRNG(&gen);
    device->SetDeviceShots(shots);
    const auto qs = device->AllocateQubits(2);
    device->NamedOperation("Hadamard", {}, {qs[0]});
    device->NamedOperation("CNOT", {}, {qs[0], qs[1]});
    return device;
}

/// The key of a tensor product of named Pauli observables.
auto pauliProduct(LCSimulator &device, const std::string &paulis,
                  const std::vector<QubitIdType> &wires) -> ObsIdType {
    std::vector<ObsIdType> factors;
    for (std::size_t k = 0; k < paulis.size(); k++) {
        const ObsId id = paulis[k] == 'X'   ? ObsId::PauliX
                         : paulis[k] == 'Y' ? ObsId::PauliY
                                            : ObsId::PauliZ;
        factors.push_back(device.Observable(id, {}, {wires[k]}));
    }
    return device.TensorObservable(factors);
}
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
                        ContainsSubstring("not allocated"));
}

TEST_CASE("LightningCliffishSimulator::shots and PRNG", "[shots]") {
    std::unique_ptr<LCSimulator> LCsim = std::make_unique<LCSimulator>();
    REQUIRE(LCsim->GetDeviceShots() == 0);

    LCsim->SetDeviceShots(500);
    REQUIRE(LCsim->GetDeviceShots() == 500);

    std::mt19937 gen{37};
    REQUIRE_NOTHROW(LCsim->SetDevicePRNG(&gen));
}

TEST_CASE("LightningCliffishSimulator exact Expval and Var", "[measurements]") {
    std::mt19937 gen{1};
    auto device = bellDevice(gen);

    CHECK(device->Expval(pauliProduct(*device, "ZZ", {0, 1})) == 1.0);
    CHECK(device->Expval(pauliProduct(*device, "XX", {0, 1})) == 1.0);
    CHECK(device->Expval(pauliProduct(*device, "YY", {0, 1})) == -1.0);
    CHECK(device->Expval(device->Observable(ObsId::PauliZ, {}, {0})) == 0.0);
    CHECK(device->Var(device->Observable(ObsId::PauliZ, {}, {0})) == 1.0);
    CHECK(device->Expval(device->Observable(ObsId::Identity, {}, {0})) == 1.0);

    // 0.5 ZZ - 2 XX + 1.5 Z_1
    const auto hamiltonian = device->HamiltonianObservable(
        {0.5, -2.0, 1.5}, {pauliProduct(*device, "ZZ", {0, 1}),
                           pauliProduct(*device, "XX", {0, 1}),
                           device->Observable(ObsId::PauliZ, {}, {1})});
    CHECK(device->Expval(hamiltonian) == -1.5);
    CHECK(std::abs(device->Var(hamiltonian) - 2.25) < 1e-12);

    // A Hadamard observable on |0>: <H> = 1/sqrt(2), Var(H) = 1/2.
    LCSimulator zero;
    zero.AllocateQubits(1);
    const auto hadamard = zero.Observable(ObsId::Hadamard, {}, {0});
    CHECK(std::abs(zero.Expval(hadamard) - 1.0 / std::sqrt(2.0)) < 1e-12);
    CHECK(std::abs(zero.Var(hadamard) - 0.5) < 1e-12);

    // A tensor product with a Hadamard factor expands into two terms.
    const auto mixed = zero.TensorObservable(
        {hadamard, zero.Observable(ObsId::Identity, {}, {0})});
    CHECK(std::abs(zero.Expval(mixed) - 1.0 / std::sqrt(2.0)) < 1e-12);
}

TEST_CASE("LightningCliffishSimulator sampled Expval and Var",
          "[measurements]") {
    std::mt19937 gen{2};
    const std::size_t shots = 10000;
    auto device = bellDevice(gen, shots);
    const double tolerance = 5.0 / std::sqrt(static_cast<double>(shots));

    CHECK(device->Expval(pauliProduct(*device, "ZZ", {0, 1})) == 1.0);
    const double z0 =
        device->Expval(device->Observable(ObsId::PauliZ, {}, {0}));
    CHECK(std::abs(z0) < tolerance);
    const double var_z0 =
        device->Var(device->Observable(ObsId::PauliZ, {}, {0}));
    CHECK(var_z0 > 1.0 - tolerance * tolerance);
    CHECK(var_z0 <= 1.0);
}

TEST_CASE("LightningCliffishSimulator Sample, Counts and Probs",
          "[measurements]") {
    std::mt19937 gen{3};
    const std::size_t shots = 4000;
    auto device = bellDevice(gen, shots);

    SECTION("Sample and PartialSample") {
        std::vector<double> buffer(shots * 2);
        auto samples = view2d(buffer, shots, 2);
        device->Sample(samples);
        std::size_t ones = 0;
        for (std::size_t shot = 0; shot < shots; shot++) {
            CHECK(samples(shot, 0) == samples(shot, 1));
            ones += samples(shot, 0) == 1.0 ? 1 : 0;
        }
        CHECK(std::abs(static_cast<double>(ones) - shots / 2.0) < 200.0);

        std::vector<double> partial_buffer(shots);
        auto partial = view2d(partial_buffer, shots, 1);
        device->PartialSample(partial, {1});
        CHECK(std::all_of(partial_buffer.begin(), partial_buffer.end(),
                          [](double bit) { return bit == 0.0 || bit == 1.0; }));
    }

    SECTION("Counts and PartialCounts") {
        std::vector<double> eigvals_buffer(4);
        std::vector<int64_t> counts_buffer(4);
        DataView<double, 1> eigvals(eigvals_buffer);
        DataView<int64_t, 1> counts(counts_buffer);
        device->Counts(eigvals, counts);
        CHECK(eigvals_buffer == std::vector<double>{0, 1, 2, 3});
        CHECK(counts_buffer[1] == 0);
        CHECK(counts_buffer[2] == 0);
        CHECK(counts_buffer[0] + counts_buffer[3] ==
              static_cast<int64_t>(shots));

        std::vector<double> small_eigvals(2);
        std::vector<int64_t> small_counts(2);
        DataView<double, 1> partial_eigvals(small_eigvals);
        DataView<int64_t, 1> partial_counts(small_counts);
        device->PartialCounts(partial_eigvals, partial_counts, {1});
        CHECK(small_counts[0] + small_counts[1] == static_cast<int64_t>(shots));

        REQUIRE_THROWS_WITH(device->Counts(partial_eigvals, partial_counts),
                            ContainsSubstring("pre-allocated counts"));
    }

    SECTION("Probs and PartialProbs") {
        std::vector<double> probs_buffer(4);
        DataView<double, 1> probs(probs_buffer);
        device->Probs(probs);
        CHECK(probs_buffer == std::vector<double>{0.5, 0.0, 0.0, 0.5});

        std::vector<double> partial_buffer(2);
        DataView<double, 1> partial(partial_buffer);
        device->PartialProbs(partial, {1});
        CHECK(partial_buffer == std::vector<double>{0.5, 0.5});
    }
}

TEST_CASE("LightningCliffishSimulator outcomes put wires[0] first",
          "[measurements]") {
    std::mt19937 gen{4};
    LCSimulator device;
    device.SetDevicePRNG(&gen);
    device.SetDeviceShots(10);
    device.AllocateQubits(2);
    device.NamedOperation("PauliX", {}, {1});

    std::vector<double> eigvals_buffer(4);
    std::vector<int64_t> counts_buffer(4);
    DataView<double, 1> eigvals(eigvals_buffer);
    DataView<int64_t, 1> counts(counts_buffer);
    device.Counts(eigvals, counts);
    CHECK(counts_buffer == std::vector<int64_t>{0, 10, 0, 0});

    device.PartialCounts(eigvals, counts, {1, 0});
    CHECK(counts_buffer == std::vector<int64_t>{0, 0, 10, 0});

    std::vector<double> probs_buffer(4);
    DataView<double, 1> probs(probs_buffer);
    device.PartialProbs(probs, {1, 0});
    CHECK(probs_buffer == std::vector<double>{0.0, 0.0, 1.0, 0.0});
}

TEST_CASE("LightningCliffishSimulator mid-circuit measurements",
          "[measurements]") {
    std::mt19937 gen{5};

    SECTION("Measure outcomes are correlated") {
        for (int trial = 0; trial < 20; trial++) {
            auto device = bellDevice(gen);
            const bool first = *device->Measure(0);
            CHECK(*device->Measure(1) == first);
            const double z =
                device->Expval(device->Observable(ObsId::PauliZ, {}, {0}));
            CHECK(z == (first ? -1.0 : 1.0));
        }
    }

    SECTION("Postselection projects onto the requested outcome") {
        auto device = bellDevice(gen);
        CHECK(*device->Measure(0, 1));
        CHECK(*device->Measure(1));
        REQUIRE_THROWS_WITH(device->Measure(1, 0),
                            ContainsSubstring("zero probability"));
    }

    SECTION("PauliMeasure measures Pauli words") {
        auto device = bellDevice(gen);
        CHECK_FALSE(*device->PauliMeasure("XX", {0, 1}));
        CHECK_FALSE(*device->PauliMeasure("ZZ", {0, 1}));
        CHECK(*device->PauliMeasure("YY", {0, 1}));
    }

    SECTION("Shots after a measurement come from its branch") {
        // Single-branch statistics: the program runs once, so every shot
        // after a mid-circuit measurement shares its outcome.
        const std::size_t shots = 500;
        auto device = bellDevice(gen, shots);
        const double outcome = *device->Measure(0) ? 1.0 : 0.0;
        std::vector<double> buffer(shots * 2);
        auto samples = view2d(buffer, shots, 2);
        device->Sample(samples);
        CHECK(std::all_of(buffer.begin(), buffer.end(),
                          [outcome](double bit) { return bit == outcome; }));
    }
}

TEST_CASE("LightningCliffishSimulator applies PauliRot and adjoints",
          "[operations]") {
    std::mt19937 gen{6};
    LCSimulator device;
    device.SetDevicePRNG(&gen);
    device.AllocateQubits(2);

    // exp(-i pi/4 X)|0> has <Y> = -1; its adjoint gives +1.
    device.NamedOperation("PauliRot", {M_PI / 2}, {0}, false, {}, {}, {"X"});
    CHECK(device.Expval(device.Observable(ObsId::PauliY, {}, {0})) == -1.0);
    device.NamedOperation("PauliRot", {M_PI / 2}, {0}, true, {}, {}, {"X"});
    device.NamedOperation("PauliRot", {M_PI / 2}, {0}, true, {}, {}, {"X"});
    CHECK(device.Expval(device.Observable(ObsId::PauliY, {}, {0})) == 1.0);

    // S S^dagger is the identity.
    device.NamedOperation("Hadamard", {}, {1});
    device.NamedOperation("S", {}, {1});
    device.NamedOperation("S", {}, {1}, true);
    CHECK(device.Expval(device.Observable(ObsId::PauliX, {}, {1})) == 1.0);
}

TEST_CASE("LightningCliffishSimulator samples a 1000-qubit GHZ state",
          "[measurements]") {
    const std::size_t num_qubits = 1000;
    const std::size_t shots = 1000;
    std::mt19937 gen{7};
    LCSimulator device;
    device.SetDevicePRNG(&gen);
    device.SetDeviceShots(shots);
    device.AllocateQubits(num_qubits);
    device.NamedOperation("Hadamard", {}, {0});
    for (QubitIdType q = 0; q + 1 < static_cast<QubitIdType>(num_qubits); q++) {
        device.NamedOperation("CNOT", {}, {q, q + 1});
    }

    std::vector<double> buffer(shots * num_qubits);
    auto samples = view2d(buffer, shots, num_qubits);
    device.Sample(samples);
    for (std::size_t shot = 0; shot < shots; shot++) {
        bool all_equal = true;
        for (std::size_t q = 1; q < num_qubits; q++) {
            all_equal = all_equal && samples(shot, q) == samples(shot, 0);
        }
        CHECK(all_equal);
    }
    CHECK(device.Expval(pauliProduct(device, "ZZ", {0, 999})) == 1.0);
}

TEST_CASE("LightningCliffishSimulator fails loudly on what it cannot run",
          "[unsupported]") {
    std::mt19937 gen{8};
    LCSimulator device;
    device.SetDevicePRNG(&gen);
    device.AllocateQubits(2);
    const auto z0 = device.Observable(ObsId::PauliZ, {}, {0});

    SECTION("A dense register over max_dense_qubits fails when the program "
            "is simulated") {
        LCSimulator small("{'max_dense_qubits': 1}");
        small.SetDevicePRNG(&gen);
        small.AllocateQubits(2);
        for (QubitIdType q = 0; q < 2; q++) {
            small.NamedOperation("Hadamard", {}, {q});
            small.NamedOperation("T", {}, {q});
        }
        const auto z = small.Observable(ObsId::PauliZ, {}, {0});
        REQUIRE_THROWS_WITH(small.Expval(z),
                            ContainsSubstring("dense register of 2 qubits"));
    }

    SECTION("An unknown recompression policy fails at construction") {
        REQUIRE_THROWS_WITH(LCSimulator("{'recompression': 'eager'}"),
                            ContainsSubstring("Unknown recompression"));
    }

    SECTION("Controlled gates fail when the program is simulated") {
        device.NamedOperation("PauliX", {}, {0}, false, {1}, {true});
        REQUIRE_THROWS_WITH(device.Expval(z0),
                            ContainsSubstring("controlled operations"));
    }

    SECTION("Unsupported observables and outputs fail at once") {
        REQUIRE_THROWS_WITH(
            device.Observable(ObsId::Hermitian, {1, 0, 0, 1}, {0}),
            ContainsSubstring("only Pauli observables"));
        REQUIRE_THROWS_WITH(device.TensorObservable({z0, z0}),
                            ContainsSubstring("distinct wires"));
        REQUIRE_THROWS_WITH(device.Expval(42),
                            ContainsSubstring("Invalid observable key"));

        std::vector<std::complex<double>> state_buffer(4);
        DataView<std::complex<double>, 1> state(state_buffer);
        REQUIRE_THROWS_WITH(device.State(state),
                            ContainsSubstring("cannot return the state"));
        REQUIRE_THROWS_WITH(device.MatrixOperation({1, 0, 0, 1}, {0}),
                            ContainsSubstring("unsupported by device"));
        REQUIRE_THROWS_WITH(device.NamedOperation("PauliRot", {0.0}, {0}),
                            ContainsSubstring("Pauli word"));
        REQUIRE_THROWS_WITH(device.Measure(0, 2),
                            ContainsSubstring("must be 0 or 1"));
    }
}
