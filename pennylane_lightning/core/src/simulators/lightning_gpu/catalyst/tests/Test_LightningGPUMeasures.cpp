// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <random>

#include "CacheManager.hpp"
#include "LightningGPUSimulator.hpp"
#include "QuantumDevice.hpp"
#include "Types.h"
#include "Utils.hpp"
#include "catch2/catch.hpp"
#include "cmath"

/// @cond DEV
namespace {
// MemRef type definition (Helper)
// TODO: Move this to a common header file
template <typename T, std::size_t R> struct MemRefT {
    T *data_allocated;
    T *data_aligned;
    std::size_t offset;
    std::size_t sizes[R];
    std::size_t strides[R];
};
using namespace Catalyst::Runtime::Simulator;
using LGPUSimulator = LightningGPUSimulator;
} // namespace
/// @endcond

TEST_CASE("NameObs test with invalid number of wires", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    REQUIRE_THROWS_WITH(sim->Observable(ObsId::PauliX, {}, {1}),
                        Catch::Contains("Invalid number of wires"));
}

TEST_CASE("NameObs test with invalid given wires for NamedObs", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    sim->AllocateQubit();

    REQUIRE_THROWS_WITH(sim->Observable(ObsId::PauliX, {}, {1}),
                        Catch::Contains("Invalid given wires"));
}

TEST_CASE("HermitianObs test with invalid number of wires", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    REQUIRE_THROWS_WITH(sim->Observable(ObsId::Hermitian, {}, {1}),
                        Catch::Contains("Invalid number of wires"));
}

TEST_CASE("HermitianObs test with invalid given wires for HermitianObs",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();
    sim->AllocateQubit();

    REQUIRE_THROWS_WITH(sim->Observable(ObsId::Hermitian, {}, {1}),
                        Catch::Contains("Invalid given wires"));
}

TEST_CASE("Check an unsupported observable", "[Measures]") {
    REQUIRE_THROWS_WITH(
        Lightning::lookup_obs<Lightning::simulator_observable_support_size>(
            Lightning::simulator_observable_support, static_cast<ObsId>(10)),
        Catch::Contains(
            "The given observable is not supported by the simulator"));
}

TEST_CASE("Measurement collapse test with 2 wires", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    constexpr std::size_t n = 2;
    std::vector<intptr_t> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    auto m = sim->Measure(Qs[0]);
    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    // LCOV_EXCL_START
    // This is conditional over the measurement result
    if (*m) {
        CHECK(pow(std::abs(std::real(state[2])), 2) +
                  pow(std::abs(std::imag(state[2])), 2) ==
              Approx(1.0).margin(1e-5));
    } else {
        CHECK(pow(std::abs(std::real(state[0])), 2) +
                  pow(std::abs(std::imag(state[0])), 2) ==
              Approx(1.0).margin(1e-5));
    }
    // LCOV_EXCL_STOP
}

TEST_CASE("Measurement collapse concrete logical qubit difference",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    constexpr std::size_t n = 1;
    // The first time an array is allocated, logical and concrete qubits
    // are the same.
    std::vector<intptr_t> Qs = sim->AllocateQubits(n);
    sim->ReleaseAllQubits();

    // Now in this the concrete qubits are shifted by n.
    Qs = sim->AllocateQubits(n);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->Measure(Qs[0]);
    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    // LCOV_EXCL_START
    bool is_zero = pow(std::abs(std::real(state[0])), 2) +
                       pow(std::abs(std::imag(state[0])), 2) ==
                   Approx(1.0).margin(1e-5);
    bool is_one = pow(std::abs(std::real(state[1])), 2) +
                      pow(std::abs(std::imag(state[1])), 2) ==
                  Approx(1.0).margin(1e-5);
    bool is_valid = is_zero ^ is_one;
    CHECK(is_valid);
    // LCOV_EXCL_STOP
}

TEST_CASE("Mid-circuit measurement naive test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    intptr_t q;

    q = sim->AllocateQubit();

    sim->NamedOperation("PauliX", {}, {q}, false);

    auto m = sim->Measure(q);

    CHECK(*m);
}

TEST_CASE("Mid-circuit measurement test with postselect = 0", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    intptr_t q;

    q = sim->AllocateQubit();

    sim->NamedOperation("Hadamard", {}, {q}, false);

    auto m = sim->Measure(q, 0);

    CHECK(*m == 0);
}

TEST_CASE("Mid-circuit measurement test with postselect = 1", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    intptr_t q;

    q = sim->AllocateQubit();

    sim->NamedOperation("Hadamard", {}, {q}, false);

    auto m = sim->Measure(q, 1);

    CHECK(*m == 1);
}

TEST_CASE("Mid-circuit measurement test with invalid postselect value",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    intptr_t q;

    q = sim->AllocateQubit();

    sim->NamedOperation("Hadamard", {}, {q}, false);

    REQUIRE_THROWS_WITH(sim->Measure(q, 2),
                        Catch::Contains("Invalid postselect value"));
}

TEST_CASE("Expval(ObsT) test with invalid key for cached observables",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    REQUIRE_THROWS_WITH(sim->Expval(0),
                        Catch::Contains("Invalid key for cached observables"));
}

TEST_CASE("Expval(NamedObs) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    CHECK(sim->Expval(px) == Approx(1.0).margin(1e-5));
    CHECK(sim->Expval(py) == Approx(.0).margin(1e-5));
    CHECK(sim->Expval(pz) == Approx(-1.0).margin(1e-5));
}

TEST_CASE("Expval(NamedObs) shots test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    constexpr std::size_t num_shots = 10000;
    sim->SetDeviceShots(num_shots);

    CHECK(sim->Expval(px) == Approx(0.0).margin(5e-2));
    CHECK(sim->Expval(py) == Approx(0.0).margin(5e-2));
    CHECK(sim->Expval(pz) == Approx(-1.0).margin(5e-2));
}

TEST_CASE("Expval(HermitianObs) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 2;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});
    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});

    CHECK(sim->Expval(h1) == Approx(.0).margin(1e-5));
    CHECK(sim->Expval(h2) == Approx(.0).margin(1e-5));
}

TEST_CASE("Expval(HermitianObs) shots test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 2;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    constexpr std::size_t num_shots = 10000;
    sim->SetDeviceShots(num_shots);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});

#ifndef PL_USE_LAPACK
    REQUIRE_THROWS_WITH(
        sim->Expval(h1),
        Catch::Contains(
            "Hermitian observables with shot measurement are not supported"));
#else
    CHECK(sim->Expval(h1) == Approx(0.0).margin(1e-5));
#endif
}

TEST_CASE("Var(HermitianObs) shots test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 2;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    constexpr std::size_t num_shots = 10000;
    sim->SetDeviceShots(num_shots);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
#ifndef PL_USE_LAPACK
    REQUIRE_THROWS_WITH(
        sim->Var(h1),
        Catch::Contains(
            "Hermitian observables with shot measurement are not supported"));
#else
    CHECK(sim->Var(h1) == Approx(0.0).margin(1e-5));
#endif
}

TEST_CASE("Expval(TensorProd(NamedObs)) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpx = sim->TensorObservable({px});
    ObsIdType tpy = sim->TensorObservable({py});
    ObsIdType tpz = sim->TensorObservable({pz});

    CHECK(sim->Expval(tpx) == Approx(1.0).margin(1e-5));
    CHECK(sim->Expval(tpy) == Approx(.0).margin(1e-5));
    CHECK(sim->Expval(tpz) == Approx(-1.0).margin(1e-5));
}

TEST_CASE("Expval(TensorProd(NamedObs)) shots test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpx = sim->TensorObservable({px});
    ObsIdType tpy = sim->TensorObservable({py});
    ObsIdType tpz = sim->TensorObservable({pz});

    constexpr std::size_t num_shots = 10000;
    sim->SetDeviceShots(num_shots);

    CHECK(sim->Expval(tpx) == Approx(1.0).margin(5e-2));
    CHECK(sim->Expval(tpy) == Approx(.0).margin(5e-2));
    CHECK(sim->Expval(tpz) == Approx(-1.0).margin(5e-2));
}

TEST_CASE("Expval(TensorProd(NamedObs[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpxy = sim->TensorObservable({px, py});
    ObsIdType tpxz = sim->TensorObservable({px, pz});

    REQUIRE_THROWS_WITH(
        sim->TensorObservable({px, py, pz}),
        Catch::Contains("All wires in observables must be disjoint."));

    CHECK(sim->Expval(tpxy) == Approx(0.0).margin(1e-5));
    CHECK(sim->Expval(tpxz) == Approx(-1.0).margin(1e-5));
}

TEST_CASE("Expval(TensorProd(NamedObs[])) shots test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();
    std::unique_ptr<LGPUSimulator> sim0 = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
        Qs.push_back(sim0->AllocateQubit());
    }

    constexpr std::size_t num_shots = 10000;
    sim->SetDeviceShots(num_shots);

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    sim0->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim0->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim0->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType tpxy = sim->TensorObservable({px, py});

    ObsIdType px0 = sim0->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py0 = sim0->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType tpxy0 = sim0->TensorObservable({px0, py0});

    CHECK(sim->Expval(tpxy) == Approx(sim0->Expval(tpxy0)).margin(5e-2));
}

TEST_CASE("Expval(TensorProd(HermitianObs))", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 2;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});
    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tph1 = sim->TensorObservable({h1});
    ObsIdType tph2 = sim->TensorObservable({h2});

    CHECK(sim->Expval(tph1) == Approx(.0).margin(1e-5));
    CHECK(sim->Expval(tph2) == Approx(.0).margin(1e-5));
}

TEST_CASE("Expval(TensorProd(HermitianObs[]))", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 2;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(4, {1.0, 0});
    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tp = sim->TensorObservable({h1, h2});

    CHECK(sim->Expval(tp) == Approx(.0).margin(1e-5));
}

TEST_CASE("Expval(TensorProd(Obs[]))", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};

    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tp = sim->TensorObservable({px, h, pz});

    CHECK(sim->Expval(tp) == Approx(-3.0).margin(1e-5));
}

TEST_CASE("Expval(Tensor(Hamiltonian(NamedObs[]), NamedObs)) test",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[0]});
    ObsIdType hxy = sim->HamiltonianObservable({0.4, 0.8}, {px, py});
    ObsIdType thz = sim->TensorObservable({hxy, pz});

    CHECK(sim->Expval(thz) == Approx(-0.4).margin(1e-5));
}

TEST_CASE("Expval(Tensor(HermitianObs, Hamiltonian()) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 3;
    std::vector<intptr_t> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType her = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[1]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[2]});
    ObsIdType hxy = sim->HamiltonianObservable({0.4, 0.8}, {px, py});
    ObsIdType ten = sim->TensorObservable({her, hxy});

    CHECK(sim->Expval(ten) == Approx(0.0).margin(1e-5));
}

TEST_CASE("Expval(Hamiltonian(NamedObs[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType hxyz = sim->HamiltonianObservable({0.4, 0.8, 0.2}, {px, py, pz});

    CHECK(sim->Expval(hxyz) == Approx(0.2).margin(1e-5));
}

TEST_CASE("Expval(Hamiltonian(NamedObs[])) shots test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType hxyz = sim->HamiltonianObservable({0.4, 0.8, 0.2}, {px, py, pz});

    constexpr std::size_t num_shots = 10000;
    sim->SetDeviceShots(num_shots);

    CHECK(sim->Expval(hxyz) == Approx(0.2).margin(5e-2));
}

TEST_CASE("Expval(Hamiltonian(TensorObs[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpxy = sim->TensorObservable({px, py});
    ObsIdType tpxz = sim->TensorObservable({px, pz});
    ObsIdType hxyz = sim->HamiltonianObservable({0.2, 0.6}, {tpxy, tpxz});

    CHECK(sim->Expval(hxyz) == Approx(-.6).margin(1e-5));
}

TEST_CASE("Expval(Hamiltonian(Hermitian[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hxhz = sim->HamiltonianObservable({0.2, 0.3, 0.6}, {px, h, pz});

    CHECK(sim->Expval(hxhz) == Approx(0.5).margin(1e-5));
}

TEST_CASE("Expval(Hamiltonian({TensorProd, Hermitian}[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tp = sim->TensorObservable({px, pz});

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hhtp = sim->HamiltonianObservable({0.5, 0.3}, {h, tp});

    CHECK(sim->Expval(hhtp) == Approx(1.2).margin(1e-5));
}

TEST_CASE("Expval(Hamiltonian({Hamiltonian, Hermitian}[])) test",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType hp = sim->HamiltonianObservable({0.2, 0.6}, {px, pz});

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hhtp = sim->HamiltonianObservable({0.5, 0.3}, {h, hp});

    CHECK(sim->Expval(hhtp) == Approx(1.38).margin(1e-5));
}

TEST_CASE("Expval(Hamiltonian({Hamiltonian(Hamiltonian), Hermitian}[])) test",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType hp = sim->HamiltonianObservable({0.2, 0.6}, {px, pz});
    ObsIdType hhp = sim->HamiltonianObservable({1}, {hp});

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hhtp = sim->HamiltonianObservable({0.5, 0.3}, {hhp, h});

    CHECK(sim->Expval(hhtp) == Approx(0.7).margin(1e-5));
}

TEST_CASE("Var(NamedObs) test with numWires=4", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[0]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[3]});

    CHECK(sim->Var(px) == Approx(.0).margin(1e-5));
    CHECK(sim->Var(py) == Approx(1.0).margin(1e-5));
    CHECK(sim->Var(pz) == Approx(.0).margin(1e-5));
}

TEST_CASE("Var(NamedObs) shots test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 2;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    constexpr std::size_t num_shots = 5000;
    sim->SetDeviceShots(num_shots);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[0]});

    CHECK(sim->Var(py) == Approx(1.0).margin(5e-2));
}

TEST_CASE("Var(HermitianObs) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 2;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});
    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});

    CHECK(sim->Var(h1) == Approx(.0).margin(1e-5));
    CHECK(sim->Var(h2) == Approx(1.0).margin(1e-5));
}

TEST_CASE("Var(TensorProd(NamedObs)) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpx = sim->TensorObservable({px});
    ObsIdType tpy = sim->TensorObservable({py});
    ObsIdType tpz = sim->TensorObservable({pz});

    CHECK(sim->Var(tpx) == Approx(.0).margin(1e-5));
    CHECK(sim->Var(tpy) == Approx(1.0).margin(1e-5));
    CHECK(sim->Var(tpz) == Approx(.0).margin(1e-5));
}

TEST_CASE("Var(TensorProd(NamedObs)) shots test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    constexpr std::size_t num_shots = 10000;
    sim->SetDeviceShots(num_shots);

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpx = sim->TensorObservable({px});
    ObsIdType tpz = sim->TensorObservable({pz});

    CHECK(sim->Var(tpx) == Approx(.0).margin(5e-2));
    CHECK(sim->Var(tpz) == Approx(.0).margin(5e-2));
}

TEST_CASE("Var(TensorProd(NamedObs[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpxy = sim->TensorObservable({px, py});
    ObsIdType tpxz = sim->TensorObservable({px, pz});

    CHECK(sim->Var(tpxy) == Approx(1.0).margin(1e-5));
    CHECK(sim->Var(tpxz) == Approx(0.0).margin(1e-5));
}

TEST_CASE("Var(TensorProd(HermitianObs)) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 2;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});
    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tph1 = sim->TensorObservable({h1});
    ObsIdType tph2 = sim->TensorObservable({h2});

    CHECK(sim->Var(tph1) == Approx(.0).margin(1e-5));
    CHECK(sim->Var(tph2) == Approx(1.0).margin(1e-5));
}

TEST_CASE("Var(TensorProd(HermitianObs[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 2;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(4, {1.0, 0});
    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tp = sim->TensorObservable({h1, h2});

    CHECK(sim->Var(tp) == Approx(2.0).margin(1e-5));
}

TEST_CASE("Var(TensorProd(Obs[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};

    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tp = sim->TensorObservable({px, h, pz});

    CHECK(sim->Var(tp) == Approx(4.0).margin(1e-5));
}

TEST_CASE("Var(Tensor(Hamiltonian(NamedObs[]), NamedObs)) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[0]});
    ObsIdType hxy = sim->HamiltonianObservable({0.4, 0.8}, {px, py});
    ObsIdType thz = sim->TensorObservable({hxy, pz});

    CHECK(sim->Var(thz) == Approx(0.64).margin(1e-5));
}

TEST_CASE("Var(Tensor(NamedObs[])) shots test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    constexpr std::size_t num_shots = 5000;
    sim->SetDeviceShots(num_shots);

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[0]});
    ObsIdType thz = sim->TensorObservable({px, py, pz});

    CHECK(sim->Var(thz) == Approx(0.99998976).margin(5e-2));
}

TEST_CASE("Var(Tensor(NamedObs[])) shots test without gates "
          "(influenced from a bug in Lightning)",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 3;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    constexpr std::size_t num_shots = 5000;
    sim->SetDeviceShots(num_shots);

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[0]});
    ObsIdType thz = sim->TensorObservable({px, py, pz});

    CHECK(sim->Var(thz) == Approx(0.99966144).margin(5e-2));
}

TEST_CASE("Var(Tensor(HermitianObs, Hamiltonian()) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 3;
    std::vector<intptr_t> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType her = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[1]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[2]});
    ObsIdType hxy = sim->HamiltonianObservable({0.4, 0.8}, {px, py});
    ObsIdType ten = sim->TensorObservable({her, hxy});

    CHECK(sim->Var(ten) == Approx(0.8).margin(1e-5));
}

TEST_CASE("Var(Tensor(HermitianObs, Hamiltonian()) shots test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 3;
    std::vector<intptr_t> Qs = sim->AllocateQubits(n);

    constexpr std::size_t num_shots = 5000;
    sim->SetDeviceShots(num_shots);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[1]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[2]});
    ObsIdType hxy = sim->HamiltonianObservable({0.4, 0.8}, {px, py});

    CHECK(sim->Var(hxy) == Approx(0.8).margin(5e-2));
}

TEST_CASE("Var(Hamiltonian(NamedObs[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType hxyz = sim->HamiltonianObservable({0.4, 0.8, 0.2}, {px, py, pz});

    CHECK(sim->Var(hxyz) == Approx(0.64).margin(1e-5));
}

TEST_CASE("Var(Hamiltonian(TensorObs[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpxy = sim->TensorObservable({px, py});
    ObsIdType tpxz = sim->TensorObservable({px, pz});
    ObsIdType hxyz = sim->HamiltonianObservable({0.2, 0.6}, {tpxy, tpxz});

    CHECK(sim->Var(hxyz) == Approx(0.04).margin(1e-5));
}

TEST_CASE("Var(Hamiltonian(Hermitian[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hxhz = sim->HamiltonianObservable({0.2, 0.3, 0.6}, {px, h, pz});

    CHECK(sim->Var(hxhz) == Approx(0.36).margin(1e-5));
}

TEST_CASE("Var(Hamiltonian({TensorProd, Hermitian}[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tp = sim->TensorObservable({px, pz});

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hhtp = sim->HamiltonianObservable({0.5, 0.3}, {h, tp});

    CHECK(sim->Var(hhtp) == Approx(1.0).margin(1e-5));
}

TEST_CASE("Var(Hamiltonian({Hamiltonian, Hermitian}[])) test", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType hp = sim->HamiltonianObservable({0.2, 0.6}, {px, pz});

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hhtp = sim->HamiltonianObservable({0.5, 0.3}, {h, hp});

    CHECK(sim->Var(hhtp) == Approx(1.0).margin(1e-5));
}

TEST_CASE("Var(Hamiltonian({Hamiltonian(Hamiltonian), Hermitian}[])) test",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType hp = sim->HamiltonianObservable({0.2, 0.6}, {px, pz});
    ObsIdType hhp = sim->HamiltonianObservable({1}, {hp});

    std::vector<std::complex<double>> mat2{
        {1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hhtp = sim->HamiltonianObservable({0.5, 0.3}, {hhp, h});

    CHECK(sim->Var(hhtp) == Approx(0.36).margin(1e-5));
}

TEST_CASE("State test with incorrect size", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs = sim->AllocateQubits(n);

    std::vector<std::complex<double>> state(1U << (n - 1));
    DataView<std::complex<double>, 1> view(state);
    REQUIRE_THROWS_WITH(
        sim->State(view),
        Catch::Contains("Invalid size for the pre-allocated state vector"));
}

TEST_CASE("State test with numWires=4", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    for (std::size_t i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 12 || i == 14) {
            CHECK(std::real(state[i]) == Approx(0.).margin(1e-5));
            CHECK(std::imag(state[i]) == Approx(0.5).margin(1e-5));
        } else {
            CHECK(std::real(state[i]) == Approx(0.).margin(1e-5));
            CHECK(std::imag(state[i]) == Approx(0.).margin(1e-5));
        }
    }
}

TEST_CASE("PartialProbs test with incorrect numWires and numAlloc",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    std::vector<double> probs_vec(1);
    DataView<double, 1> probs_view(probs_vec);

    REQUIRE_THROWS_WITH(
        sim->PartialProbs(probs_view, {Qs[0], Qs[1], Qs[2], Qs[3], Qs[0]}),
        Catch::Contains("Invalid number of wires"));

    REQUIRE_THROWS_WITH(
        sim->PartialProbs(probs_view, {Qs[0]}),
        Catch::Contains(
            "Invalid size for the pre-allocated partial-probabilities"));

    REQUIRE_THROWS_WITH(
        sim->Probs(probs_view),
        Catch::Contains("Invalid size for the pre-allocated probabilities"));

    sim->ReleaseQubit(Qs[0]);

    REQUIRE_THROWS_WITH(sim->PartialProbs(probs_view, {Qs[0]}),
                        Catch::Contains("Invalid given wires to measure"));
}

TEST_CASE("Probs and PartialProbs tests with numWires=0-4", "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    std::vector<double> probs0(1);
    DataView<double, 1> view0(probs0);
    sim->PartialProbs(view0, std::vector<intptr_t>{});

    std::vector<double> probs1(2);
    DataView<double, 1> view1(probs1);
    sim->PartialProbs(view1, std::vector<intptr_t>{Qs[2]});

    std::vector<double> probs2(4);
    DataView<double, 1> view2(probs2);
    sim->PartialProbs(view2, std::vector<intptr_t>{Qs[0], Qs[3]});

    std::vector<double> probs3(16);
    DataView<double, 1> view3(probs3);
    sim->PartialProbs(view3, Qs);

    std::vector<double> probs4(16);
    DataView<double, 1> view4(probs4);
    sim->Probs(view4);

    CHECK(probs0.size() == 1);
    CHECK(probs0[0] == Approx(1.0));
    CHECK(probs1[0] == Approx(0.5).margin(1e-5));
    CHECK(probs1[1] == Approx(0.5).margin(1e-5));
    for (std::size_t i = 0; i < 4; i++) {
        if (i == 0 || i == 2) {
            CHECK(probs2[i] == Approx(0.5).margin(1e-5));
        } else {
            CHECK(probs2[i] == Approx(0.).margin(1e-5));
        }
    }
    for (std::size_t i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 12 || i == 14) {
            CHECK(probs3[i] == Approx(0.25).margin(1e-5));
            CHECK(probs4[i] == Approx(0.25).margin(1e-5));
        } else {
            CHECK(probs3[i] == Approx(0.).margin(1e-5));
            CHECK(probs4[i] == Approx(0.).margin(1e-5));
        }
    }
}

TEST_CASE("Probs and PartialProbs shots tests with numWires=0-4",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    constexpr std::size_t num_shots = 10000;
    sim->SetDeviceShots(num_shots);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    std::vector<double> probs0(1);
    DataView<double, 1> view0(probs0);
    sim->PartialProbs(view0, std::vector<intptr_t>{});

    std::vector<double> probs1(2);
    DataView<double, 1> view1(probs1);
    sim->PartialProbs(view1, std::vector<intptr_t>{Qs[2]});

    std::vector<double> probs2(4);
    DataView<double, 1> view2(probs2);
    sim->PartialProbs(view2, std::vector<intptr_t>{Qs[0], Qs[3]});

    std::vector<double> probs3(16);
    DataView<double, 1> view3(probs3);
    sim->PartialProbs(view3, Qs);

    std::vector<double> probs4(16);
    DataView<double, 1> view4(probs4);
    sim->Probs(view4);

    CHECK(probs0.size() == 1);
    CHECK(probs0[0] == Approx(1.0).margin(5e-2));
    CHECK(probs1[0] == Approx(0.5).margin(5e-2));
    CHECK(probs1[1] == Approx(0.5).margin(5e-2));
    for (std::size_t i = 0; i < 4; i++) {
        if (i == 0 || i == 2) {
            CHECK(probs2[i] == Approx(0.5).margin(5e-2));
        } else {
            CHECK(probs2[i] == Approx(0.).margin(5e-2));
        }
    }
    for (std::size_t i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 12 || i == 14) {
            CHECK(probs3[i] == Approx(0.25).margin(5e-2));
            CHECK(probs4[i] == Approx(0.25).margin(5e-2));
        } else {
            CHECK(probs3[i] == Approx(0.).margin(5e-2));
            CHECK(probs4[i] == Approx(0.).margin(5e-2));
        }
    }
}

TEST_CASE("PartialSample test with incorrect numWires and numAlloc",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    std::vector<double> samples_vec(1);
    MemRefT<double, 2> samples{samples_vec.data(),
                               samples_vec.data(),
                               0,
                               {samples_vec.size(), 1},
                               {1, 1}};
    DataView<double, 2> view(samples.data_aligned, samples.offset,
                             samples.sizes, samples.strides);

    REQUIRE_THROWS_WITH(
        sim->PartialSample(view, {Qs[0], Qs[1], Qs[2], Qs[3], Qs[0]}, 4),
        Catch::Contains("Invalid number of wires"));

    REQUIRE_THROWS_WITH(
        sim->PartialSample(view, {Qs[0], Qs[1]}, 2),
        Catch::Contains("Invalid size for the pre-allocated partial-samples"));

    REQUIRE_THROWS_WITH(
        sim->Sample(view, 2),
        Catch::Contains("Invalid size for the pre-allocated samples"));

    sim->ReleaseQubit(Qs[0]);

    REQUIRE_THROWS_WITH(sim->PartialSample(view, {Qs[0]}, 4),
                        Catch::Contains("Invalid given wires to measure"));
}

TEST_CASE("PartialCounts test with incorrect numWires and numAlloc",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    std::vector<double> eigvals_vec(1);
    DataView<double, 1> eigvals_view(eigvals_vec);

    std::vector<int64_t> counts_vec(1);
    DataView<int64_t, 1> counts_view(counts_vec);

    REQUIRE_THROWS_WITH(sim->PartialCounts(eigvals_view, counts_view,
                                           {Qs[0], Qs[1], Qs[2], Qs[3], Qs[0]},
                                           4),
                        Catch::Contains("Invalid number of wires"));

    REQUIRE_THROWS_WITH(
        sim->PartialCounts(eigvals_view, counts_view, {Qs[0]}, 1),
        Catch::Contains("Invalid size for the pre-allocated partial-counts"));

    REQUIRE_THROWS_WITH(
        sim->Counts(eigvals_view, counts_view, 1),
        Catch::Contains("Invalid size for the pre-allocated counts"));

    sim->ReleaseQubit(Qs[0]);

    REQUIRE_THROWS_WITH(
        sim->PartialCounts(eigvals_view, counts_view, {Qs[0]}, 4),
        Catch::Contains("Invalid given wires to measure"));
}

TEST_CASE("Sample and PartialSample tests with numWires=0-4 shots=100",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("RX", {0.5}, {Qs[0]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[1]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

    std::size_t shots = 100;

    std::vector<double> samples1(shots * 1);
    MemRefT<double, 2> buffer1{
        samples1.data(), samples1.data(), 0, {shots, 1}, {1, 1}};
    DataView<double, 2> view1(buffer1.data_aligned, buffer1.offset,
                              buffer1.sizes, buffer1.strides);
    sim->PartialSample(view1, std::vector<intptr_t>{Qs[2]}, shots);

    std::vector<double> samples2(shots * 2);
    MemRefT<double, 2> buffer2{
        samples2.data(), samples2.data(), 0, {shots, 2}, {1, 1}};
    DataView<double, 2> view2(buffer2.data_aligned, buffer2.offset,
                              buffer2.sizes, buffer2.strides);
    sim->PartialSample(view2, std::vector<intptr_t>{Qs[0], Qs[3]}, shots);

    std::vector<double> samples3(shots * 4);
    MemRefT<double, 2> buffer3{
        samples3.data(), samples3.data(), 0, {shots, 4}, {1, 1}};
    DataView<double, 2> view3(buffer3.data_aligned, buffer3.offset,
                              buffer3.sizes, buffer3.strides);
    sim->PartialSample(view3, Qs, shots);

    std::vector<double> samples4(shots * 4);
    MemRefT<double, 2> buffer4{
        samples4.data(), samples4.data(), 0, {shots, 4}, {1, 1}};
    DataView<double, 2> view4(buffer4.data_aligned, buffer4.offset,
                              buffer4.sizes, buffer4.strides);
    sim->Sample(view4, shots);

    for (std::size_t i = 0; i < shots * 1; i++)
        CHECK((samples1[i] == 0. || samples1[i] == 1.));
    for (std::size_t i = 0; i < shots * 2; i++)
        CHECK((samples2[i] == 0. || samples2[i] == 1.));
    for (std::size_t i = 0; i < shots * 4; i++)
        CHECK((samples3[i] == 0. || samples3[i] == 1.));
    for (std::size_t i = 0; i < shots * 4; i++)
        CHECK((samples4[i] == 0. || samples4[i] == 1.));
}

TEST_CASE("Sample and PartialSample tests with numWires=0-4 "
          "shots=1000 mcmc=True num_burnin=200",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim =
        std::make_unique<LGPUSimulator>("{mcmc : True, num_burnin : 200}");

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs;
    Qs.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("RX", {0.5}, {Qs[0]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[1]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

    std::size_t shots = 100;

    std::vector<double> samples1(shots * 1);
    MemRefT<double, 2> buffer1{
        samples1.data(), samples1.data(), 0, {shots, 1}, {1, 1}};
    DataView<double, 2> view1(buffer1.data_aligned, buffer1.offset,
                              buffer1.sizes, buffer1.strides);
    sim->PartialSample(view1, std::vector<intptr_t>{Qs[2]}, shots);

    std::vector<double> samples2(shots * 2);
    MemRefT<double, 2> buffer2{
        samples2.data(), samples2.data(), 0, {shots, 2}, {1, 1}};
    DataView<double, 2> view2(buffer2.data_aligned, buffer2.offset,
                              buffer2.sizes, buffer2.strides);
    sim->PartialSample(view2, std::vector<intptr_t>{Qs[0], Qs[3]}, shots);

    std::vector<double> samples3(shots * 4);
    MemRefT<double, 2> buffer3{
        samples3.data(), samples3.data(), 0, {shots, 4}, {1, 1}};
    DataView<double, 2> view3(buffer3.data_aligned, buffer3.offset,
                              buffer3.sizes, buffer3.strides);
    sim->PartialSample(view3, Qs, shots);

    std::vector<double> samples4(shots * 4);
    MemRefT<double, 2> buffer4{
        samples4.data(), samples4.data(), 0, {shots, 4}, {1, 1}};
    DataView<double, 2> view4(buffer4.data_aligned, buffer4.offset,
                              buffer4.sizes, buffer4.strides);
    sim->Sample(view4, shots);

    for (std::size_t i = 0; i < shots * 1; i++)
        CHECK((samples1[i] == 0. || samples1[i] == 1.));
    for (std::size_t i = 0; i < shots * 2; i++)
        CHECK((samples2[i] == 0. || samples2[i] == 1.));
    for (std::size_t i = 0; i < shots * 4; i++)
        CHECK((samples3[i] == 0. || samples3[i] == 1.));
    for (std::size_t i = 0; i < shots * 4; i++)
        CHECK((samples4[i] == 0. || samples4[i] == 1.));
}

TEST_CASE("Counts and PartialCounts tests with numWires=0-4 shots=100",
          "[Measures]") {
    std::unique_ptr<LGPUSimulator> sim = std::make_unique<LGPUSimulator>();

    // state-vector with #qubits = n
    constexpr std::size_t n = 4;
    std::vector<intptr_t> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("RX", {0.5}, {Qs[0]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[1]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

    std::size_t shots = 100;

    std::vector<double> eigvals0(1);
    std::vector<int64_t> counts0(1);
    DataView<double, 1> eview0(eigvals0);
    DataView<int64_t, 1> cview0(counts0);
    sim->PartialCounts(eview0, cview0, std::vector<intptr_t>{}, shots);

    std::vector<double> eigvals1(2);
    std::vector<int64_t> counts1(2);
    DataView<double, 1> eview1(eigvals1);
    DataView<int64_t, 1> cview1(counts1);
    sim->PartialCounts(eview1, cview1, std::vector<intptr_t>{Qs[2]}, shots);

    std::vector<double> eigvals2(4);
    std::vector<int64_t> counts2(4);
    DataView<double, 1> eview2(eigvals2);
    DataView<int64_t, 1> cview2(counts2);
    sim->PartialCounts(eview2, cview2, std::vector<intptr_t>{Qs[0], Qs[3]},
                       shots);

    std::vector<double> eigvals3(16);
    std::vector<int64_t> counts3(16);
    DataView<double, 1> eview3(eigvals3);
    DataView<int64_t, 1> cview3(counts3);
    sim->PartialCounts(eview3, cview3, Qs, shots);

    std::vector<double> eigvals4(16);
    std::vector<int64_t> counts4(16);
    DataView<double, 1> eview4(eigvals4);
    DataView<int64_t, 1> cview4(counts4);
    sim->Counts(eview4, cview4, shots);

    CHECK(eigvals0.size() == 1);
    CHECK(eigvals0[0] == 0.0);
    CHECK(counts0.size() == 1);
    CHECK(counts0[0] == static_cast<int64_t>(shots));
    CHECK((eigvals1[0] == 0. && eigvals1[1] == 1.));
    CHECK((eigvals2[0] == 0. && eigvals2[1] == 1. && eigvals2[2] == 2. &&
           eigvals2[3] == 3.));
    for (std::size_t i = 0; i < 16; i++) {
        CHECK(eigvals3[i] == static_cast<double>(i));
        CHECK(eigvals4[i] == static_cast<double>(i));
    }

    CHECK(counts1[0] + counts1[1] == static_cast<int64_t>(shots));
    CHECK(counts2[0] + counts2[1] + counts2[2] + counts2[3] ==
          static_cast<int64_t>(shots));
    std::size_t sum3 = 0, sum4 = 0;
    for (std::size_t i = 0; i < 16; i++) {
        sum3 += counts3[i];
        sum4 += counts4[i];
    }
    CHECK(sum3 == shots);
    CHECK(sum4 == shots);
}

TEST_CASE("Measurement with a seeded device", "[Measures]") {
    std::array<std::unique_ptr<LGPUSimulator>, 2> sims;
    std::vector<std::mt19937> gens{std::mt19937{37}, std::mt19937{37}};

    auto circuit = [](LGPUSimulator &sim, std::mt19937 &gen) {
        sim.SetDevicePRNG(&gen);
        std::vector<intptr_t> Qs;
        Qs.reserve(1);
        Qs.push_back(sim.AllocateQubit());
        sim.NamedOperation("Hadamard", {}, {Qs[0]}, false);
        auto m = sim.Measure(Qs[0]);
        return m;
    };

    for (std::size_t trial = 0; trial < 5; trial++) {
        sims[0] = std::make_unique<LGPUSimulator>();
        sims[1] = std::make_unique<LGPUSimulator>();

        auto m0 = circuit(*(sims[0]), gens[0]);
        auto m1 = circuit(*(sims[1]), gens[1]);

        CHECK(*m0 == *m1);
    }
}

TEST_CASE("Sample with a seeded device", "[Measures]") {
    std::size_t shots = 100;
    std::array<std::unique_ptr<LGPUSimulator>, 2> sims;
    std::vector<std::vector<double>> sample_vec(2,
                                                std::vector<double>(shots * 4));

    std::vector<MemRefT<double, 2>> buffers{
        MemRefT<double, 2>{
            sample_vec[0].data(), sample_vec[0].data(), 0, {shots, 1}, {1, 1}},
        MemRefT<double, 2>{
            sample_vec[1].data(), sample_vec[1].data(), 0, {shots, 1}, {1, 1}},
    };
    std::vector<DataView<double, 2>> views{
        DataView<double, 2>(buffers[0].data_aligned, buffers[0].offset,
                            buffers[0].sizes, buffers[0].strides),
        DataView<double, 2>(buffers[1].data_aligned, buffers[1].offset,
                            buffers[1].sizes, buffers[1].strides)};

    std::vector<std::mt19937> gens{std::mt19937{37}, std::mt19937{37}};

    auto circuit = [shots](LGPUSimulator &sim, DataView<double, 2> &view,
                           std::mt19937 &gen) {
        sim.SetDevicePRNG(&gen);
        std::vector<intptr_t> Qs;
        Qs.reserve(1);
        Qs.push_back(sim.AllocateQubit());
        sim.NamedOperation("Hadamard", {}, {Qs[0]}, false);
        sim.NamedOperation("RX", {0.5}, {Qs[0]}, false);
        sim.Sample(view, shots);
    };

    for (std::size_t trial = 0; trial < 5; trial++) {
        sims[0] = std::make_unique<LGPUSimulator>();
        sims[1] = std::make_unique<LGPUSimulator>();

        for (std::size_t sim_idx = 0; sim_idx < sims.size(); sim_idx++) {
            circuit(*(sims[sim_idx]), views[sim_idx], gens[sim_idx]);
        }

        for (std::size_t i = 0; i < sample_vec[0].size(); i++) {
            CHECK((sample_vec[0][i] == sample_vec[1][i]));
        }
    }
}
