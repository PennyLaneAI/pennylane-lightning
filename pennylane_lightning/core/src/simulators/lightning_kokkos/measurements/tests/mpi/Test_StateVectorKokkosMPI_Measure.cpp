// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <complex>
#include <limits> // numeric_limits
#include <random>
#include <type_traits>
#include <vector>

#include <catch2/catch.hpp>

#include "MeasurementsKokkosMPI.hpp"
#include "ObservablesKokkosMPI.hpp"
#include "StateVectorKokkosMPI.hpp"
#include "TestHelpers.hpp" // createRandomStateVectorData

/**
 * @file
 *  Tests for functionality for the class StateVectorKokkosMPI.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::Util;

using Pennylane::Util::isApproxEqual;
using Pennylane::Util::randomUnitary;

std::mt19937_64 re{1337};
} // namespace
/// @endcond

// TODO: Add error cases

// expval
TEMPLATE_TEST_CASE("Expval - named string", "[LKMPI_Expval]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});

    const std::string ob_name =
        GENERATE("Identity", "PauliX", "PauliY", "PauliZ", "Hadamard");
    const std::size_t wire = GENERATE(0, 1, 2, 3);

    auto res = m.expval(ob_name, {wire});
    auto res_ref = m_ref.expval(ob_name, {wire});
    CHECK(res == Approx(res_ref).margin(EP));
}

// expval matrix
TEMPLATE_TEST_CASE("Expval - 1-wire matrix", "[LKMPI_Expval]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliX", {1});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});
    sv.applyOperation("RX", {1}, false, {0.5});
    sv.applyOperation("RX", {2}, false, {0.5});
    sv.applyOperation("RY", {1}, true, {0.3});
    sv.applyOperation("RZ", {3}, true, {0.2});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliX", {1});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});
    sv_ref.applyOperation("RX", {1}, false, {0.5});
    sv_ref.applyOperation("RX", {2}, false, {0.5});
    sv_ref.applyOperation("RY", {1}, true, {0.3});
    sv_ref.applyOperation("RZ", {3}, true, {0.2});

    std::size_t num_wires = 1;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire = GENERATE(0, 1, 2, 3);

    auto res = m.expval(mat_ob, {wire});
    auto res_ref = m_ref.expval(mat_ob, {wire});
    CHECK(res == Approx(res_ref).margin(EP));
}

TEMPLATE_TEST_CASE("Expval - 2-wire matrix", "[LKMPI_Expval]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliX", {1});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});
    sv.applyOperation("RX", {1}, false, {0.5});
    sv.applyOperation("RX", {2}, false, {0.5});
    sv.applyOperation("RY", {1}, true, {0.3});
    sv.applyOperation("RZ", {3}, true, {0.2});
    sv.applyOperation("RX", {4}, false, {0.5});
    sv.applyOperation("RY", {5}, true, {0.3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliX", {1});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});
    sv_ref.applyOperation("RX", {1}, false, {0.5});
    sv_ref.applyOperation("RX", {2}, false, {0.5});
    sv_ref.applyOperation("RY", {1}, true, {0.3});
    sv_ref.applyOperation("RZ", {3}, true, {0.2});
    sv_ref.applyOperation("RX", {4}, false, {0.5});
    sv_ref.applyOperation("RY", {5}, true, {0.3});

    std::size_t num_wires = 2;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire_0 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t wire_1 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1};

    if (wires.size() == num_wires) {
        auto res = m.expval(mat_ob, {wire_0, wire_1});
        auto res_ref = m_ref.expval(mat_ob, {wire_0, wire_1});
        CHECK(res == Approx(res_ref).margin(EP));
    }
}

// expval Hermitian obs
TEMPLATE_TEST_CASE("Expval - 1-wire matrix Hermitian obs", "[LKMPI_Expval]",
                   float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliX", {1});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});
    sv.applyOperation("RX", {1}, false, {0.5});
    sv.applyOperation("RX", {2}, false, {0.5});
    sv.applyOperation("RY", {1}, true, {0.3});
    sv.applyOperation("RZ", {3}, true, {0.2});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliX", {1});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});
    sv_ref.applyOperation("RX", {1}, false, {0.5});
    sv_ref.applyOperation("RX", {2}, false, {0.5});
    sv_ref.applyOperation("RY", {1}, true, {0.3});
    sv_ref.applyOperation("RZ", {3}, true, {0.2});

    std::size_t num_wires = 1;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire = GENERATE(0, 1, 2, 3);

    auto ob = HermitianObsMPI<StateVectorKokkosMPI<TestType>>(mat_ob, {wire});
    auto ob_ref = HermitianObs<StateVectorKokkos<TestType>>(mat_ob, {wire});
    auto res = m.expval(ob);
    auto res_ref = m_ref.expval(ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}

TEMPLATE_TEST_CASE("Expval - 2-wire matrix Hermitian obs", "[LKMPI_Expval]",
                   float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliX", {1});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});
    sv.applyOperation("RX", {1}, false, {0.5});
    sv.applyOperation("RX", {2}, false, {0.5});
    sv.applyOperation("RY", {1}, true, {0.3});
    sv.applyOperation("RZ", {3}, true, {0.2});
    sv.applyOperation("RX", {4}, false, {0.5});
    sv.applyOperation("RY", {5}, true, {0.3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliX", {1});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});
    sv_ref.applyOperation("RX", {1}, false, {0.5});
    sv_ref.applyOperation("RX", {2}, false, {0.5});
    sv_ref.applyOperation("RY", {1}, true, {0.3});
    sv_ref.applyOperation("RZ", {3}, true, {0.2});
    sv_ref.applyOperation("RX", {4}, false, {0.5});
    sv_ref.applyOperation("RY", {5}, true, {0.3});

    std::size_t num_wires = 2;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire_0 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t wire_1 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1};

    if (wires.size() == num_wires) {

        auto ob = HermitianObsMPI<StateVectorKokkosMPI<TestType>>(
            mat_ob, {wire_0, wire_1});
        auto ob_ref =
            HermitianObs<StateVectorKokkos<TestType>>(mat_ob, {wire_0, wire_1});
        auto res = m.expval(ob);
        auto res_ref = m_ref.expval(ob_ref);
        CHECK(res == Approx(res_ref).margin(EP));
    }
}

// expval NamedObs
TEMPLATE_TEST_CASE("Expval - NamedObs", "[LKMPI_Expval]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});

    const std::string ob_name =
        GENERATE("Identity", "PauliX", "PauliY", "PauliZ", "Hadamard");
    const std::size_t wire = GENERATE(0, 1, 2, 3);

    auto ob = NamedObsMPI<StateVectorKokkosMPI<TestType>>(ob_name, {wire});
    auto ob_ref = NamedObs<StateVectorKokkos<TestType>>(ob_name, {wire});

    auto res = m.expval(ob);
    auto res_ref = m_ref.expval(ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}

// expval TensorProdObs

TEMPLATE_TEST_CASE("Expval - TensorProdobs", "[LKMPI_Expval]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});

    auto X0_ref = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
        "PauliX", std::vector<std::size_t>{0});
    auto Z1_ref = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
        "PauliZ", std::vector<std::size_t>{1});
    auto X0 = std::make_shared<NamedObs<StateVectorKokkosMPI<TestType>>>(
        "PauliX", std::vector<std::size_t>{0});
    auto Z1 = std::make_shared<NamedObs<StateVectorKokkosMPI<TestType>>>(
        "PauliZ", std::vector<std::size_t>{1});

    auto ob_ref =
        TensorProdObs<StateVectorKokkos<TestType>>::create({X0_ref, Z1_ref});
    auto ob =
        TensorProdObsMPI<StateVectorKokkosMPI<TestType>>::create({X0, Z1});

    auto res = m.expval(*ob);
    auto res_ref = m_ref.expval(*ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}

// expval HamiltonianObs
TEMPLATE_TEST_CASE("Test expectation value of HamiltonianObs", "[LKMPI_Expval]",
                   // float,
                   double) {
    using ComplexT = StateVectorKokkosMPI<TestType>::ComplexT;
    SECTION("Using expval") {
        std::vector<ComplexT> init_sv{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                      {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                      {0.3, 0.4}, {0.4, 0.5}};
        const TestType EP = 1e-4;
        const std::size_t num_qubits = 3;
        StateVectorKokkosMPI<TestType> sv{num_qubits};
        // Only run with 4 mpi ranks:
        REQUIRE(sv.getMPISize() == 4);

        auto m = MeasurementsMPI(sv);

        // Set initial data
        std::size_t block_size = sv.getLocalBlockSize();
        std::vector<Kokkos::complex<TestType>> init_subsv(block_size,
                                                          {0.0, 0.0});
        for (std::size_t element = 0; element < block_size; element++) {
            init_subsv[element] =
                init_sv[sv.getMPIRank() * block_size + element];
        }
        sv.updateData(init_subsv);

        auto X0 = std::make_shared<NamedObsMPI<StateVectorKokkosMPI<TestType>>>(
            "PauliX", std::vector<std::size_t>{0});
        auto Z1 = std::make_shared<NamedObsMPI<StateVectorKokkosMPI<TestType>>>(
            "PauliZ", std::vector<std::size_t>{1});

        auto ob = HamiltonianMPI<StateVectorKokkosMPI<TestType>>::create(
            {0.3, 0.5}, {X0, Z1});
        auto res = m.expval(*ob);
        auto expected = TestType(-0.086);
        CHECK(expected == Approx(res));
    }
}

// expval pauli word
// This test takes a long time - float 2.5min, double 2.8min
TEMPLATE_TEST_CASE("Expval - pauli word - 4 wires", "[LKMPI_Expval]", float,
                   double) {
    const TestType EP = std::is_same_v<TestType, float> ? 1e-3 : 1e-6;
    const std::size_t num_qubits = 6;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliY", {1});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("Rot", {0}, false, {0.1, 0.2, 0.3});
    sv.applyOperation("Rot", {1}, false, {0.2, 0.3, 0.4});
    sv.applyOperation("Rot", {2}, false, {0.3, 0.4, 0.5});
    sv.applyOperation("Rot", {3}, false, {0.4, 0.5, 0.6});
    sv.applyOperation("Rot", {4}, false, {0.5, 0.6, 0.7});
    sv.applyOperation("Rot", {5}, false, {0.6, 0.7, 0.8});
    sv.applyOperation("RX", {0}, false, {0.1});
    sv.applyOperation("CNOT", {1, 3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliY", {1});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("Rot", {0}, false, {0.1, 0.2, 0.3});
    sv_ref.applyOperation("Rot", {1}, false, {0.2, 0.3, 0.4});
    sv_ref.applyOperation("Rot", {2}, false, {0.3, 0.4, 0.5});
    sv_ref.applyOperation("Rot", {3}, false, {0.4, 0.5, 0.6});
    sv_ref.applyOperation("Rot", {4}, false, {0.5, 0.6, 0.7});
    sv_ref.applyOperation("Rot", {5}, false, {0.6, 0.7, 0.8});
    sv_ref.applyOperation("RX", {0}, false, {0.1});
    sv_ref.applyOperation("CNOT", {1, 3});

    const std::string P_0 = GENERATE("I", "X", "Y", "Z");
    const std::string P_1 = GENERATE("I", "X", "Y", "Z");
    const std::string P_2 = GENERATE("I", "X", "Y", "Z");
    const std::string P_3 = GENERATE("I", "X", "Y", "Z");
    const std::string ob = P_0 + P_1 + P_2 + P_3;
    // const std::string ob = "ZIII";

    const std::size_t wire_0 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t wire_1 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t wire_2 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t wire_3 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1, wire_2, wire_3};

    DYNAMIC_SECTION("Pauli word = " << ob << " on wires " << wire_0 << ", "
                                    << wire_1 << ", " << wire_2 << ", "
                                    << wire_3) {
        if (wires.size() == 4) {

            auto res =
                m.expval({ob}, {{wire_0, wire_1, wire_2, wire_3}}, {0.1});
            auto res_ref =
                m_ref.expval({ob}, {{wire_0, wire_1, wire_2, wire_3}}, {0.1});
            CHECK(res == Approx(res_ref).margin(EP));
        }
    }
}

TEMPLATE_TEST_CASE("Expval - pauli word - 4 wires linear combin",
                   "[LKMPI_Expval]", float, double) {
    const TestType EP = 1e-6;
    const std::size_t num_qubits = 6;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliY", {1});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("Rot", {0}, false, {0.1, 0.2, 0.3});
    sv.applyOperation("Rot", {1}, false, {0.2, 0.3, 0.4});
    sv.applyOperation("Rot", {2}, false, {0.3, 0.4, 0.5});
    sv.applyOperation("Rot", {3}, false, {0.4, 0.5, 0.6});
    sv.applyOperation("Rot", {4}, false, {0.5, 0.6, 0.7});
    sv.applyOperation("Rot", {5}, false, {0.6, 0.7, 0.8});
    sv.applyOperation("RX", {0}, false, {0.1});
    sv.applyOperation("CNOT", {1, 3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliY", {1});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("Rot", {0}, false, {0.1, 0.2, 0.3});
    sv_ref.applyOperation("Rot", {1}, false, {0.2, 0.3, 0.4});
    sv_ref.applyOperation("Rot", {2}, false, {0.3, 0.4, 0.5});
    sv_ref.applyOperation("Rot", {3}, false, {0.4, 0.5, 0.6});
    sv_ref.applyOperation("Rot", {4}, false, {0.5, 0.6, 0.7});
    sv_ref.applyOperation("Rot", {5}, false, {0.6, 0.7, 0.8});
    sv_ref.applyOperation("RX", {0}, false, {0.1});
    sv_ref.applyOperation("CNOT", {1, 3});

    const std::string ob_0 = "ZZYYX";
    const std::string ob_1 = "XYIZX";

    const std::vector<std::size_t> wires_0{0, 1, 2, 4, 5};
    const std::vector<std::size_t> wires_1{5, 1, 4, 3, 0};

    auto res = m.expval({ob_0, ob_1}, {wires_0, wires_1}, {0.1, 0.2});
    auto res_ref = m_ref.expval({ob_0, ob_1}, {wires_0, wires_1}, {0.1, 0.2});
    CHECK(res == Approx(res_ref).margin(EP));
}

TEMPLATE_TEST_CASE("Expval - pauli word - error", "[LKMPI_Expval]", float,
                   double) {
    const std::size_t num_qubits = 6;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);

    PL_REQUIRE_THROWS_MATCHES(m.expval({"XXXXX"}, {{0, 1, 2, 3, 4}}, {0.1}),
                              LightningException,
                              "Number of PauliX and PauliY in Pauli String "
                              "exceeds the number of local wires.");

    PL_REQUIRE_THROWS_MATCHES(m.expval({"YYYYY"}, {{0, 1, 2, 3, 4}}, {0.1}),
                              LightningException,
                              "Number of PauliX and PauliY in Pauli String "
                              "exceeds the number of local wires.");
}

// var named string
TEMPLATE_TEST_CASE("Var - named string", "[LKMPI_Var]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});

    const std::string ob_name =
        GENERATE("Identity", "PauliX", "PauliY", "PauliZ", "Hadamard");
    const std::size_t wire = GENERATE(0, 1, 2, 3);

    auto res = m.var(ob_name, {wire});
    auto res_ref = m_ref.var(ob_name, {wire});
    CHECK(res == Approx(res_ref).margin(EP));
}

// var matrix
TEMPLATE_TEST_CASE("Var - 1-wire matrix", "[LKMPI_Var]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliX", {1});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});
    sv.applyOperation("RX", {1}, false, {0.5});
    sv.applyOperation("RX", {2}, false, {0.5});
    sv.applyOperation("RY", {1}, true, {0.3});
    sv.applyOperation("RZ", {3}, true, {0.2});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliX", {1});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});
    sv_ref.applyOperation("RX", {1}, false, {0.5});
    sv_ref.applyOperation("RX", {2}, false, {0.5});
    sv_ref.applyOperation("RY", {1}, true, {0.3});
    sv_ref.applyOperation("RZ", {3}, true, {0.2});

    std::size_t num_wires = 1;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire = GENERATE(0, 1, 2, 3);

    auto res = m.var(mat_ob, {wire});
    auto res_ref = m_ref.var(mat_ob, {wire});
    CHECK(res == Approx(res_ref).margin(EP));
}

TEMPLATE_TEST_CASE("Var - 2-wire matrix", "[LKMPI_Var]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliX", {1});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});
    sv.applyOperation("RX", {1}, false, {0.5});
    sv.applyOperation("RX", {2}, false, {0.5});
    sv.applyOperation("RY", {1}, true, {0.3});
    sv.applyOperation("RZ", {3}, true, {0.2});
    sv.applyOperation("RX", {4}, false, {0.5});
    sv.applyOperation("RY", {5}, true, {0.3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliX", {1});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});
    sv_ref.applyOperation("RX", {1}, false, {0.5});
    sv_ref.applyOperation("RX", {2}, false, {0.5});
    sv_ref.applyOperation("RY", {1}, true, {0.3});
    sv_ref.applyOperation("RZ", {3}, true, {0.2});
    sv_ref.applyOperation("RX", {4}, false, {0.5});
    sv_ref.applyOperation("RY", {5}, true, {0.3});

    std::size_t num_wires = 2;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire_0 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t wire_1 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1};

    if (wires.size() == num_wires) {
        auto res = m.var(mat_ob, {wire_0, wire_1});
        auto res_ref = m_ref.var(mat_ob, {wire_0, wire_1});
        CHECK(res == Approx(res_ref).margin(EP));
    }
}

// var NamedObs
TEMPLATE_TEST_CASE("Var - named obs", "[LKMPI_Var]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});

    const std::string ob_name =
        GENERATE("Identity", "PauliX", "PauliY", "PauliZ", "Hadamard");
    const std::size_t wire = GENERATE(0, 1, 2, 3);

    auto ob = NamedObsMPI<StateVectorKokkosMPI<TestType>>(ob_name, {wire});
    auto ob_ref = NamedObs<StateVectorKokkos<TestType>>(ob_name, {wire});

    auto res = m.var(ob);
    auto res_ref = m_ref.var(ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}

// var Hermitian obs
TEMPLATE_TEST_CASE("Var - 1-wire matrix Hermitian obs", "[LKMPI_Var]", float,
                   double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliX", {1});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});
    sv.applyOperation("RX", {1}, false, {0.5});
    sv.applyOperation("RX", {2}, false, {0.5});
    sv.applyOperation("RY", {1}, true, {0.3});
    sv.applyOperation("RZ", {3}, true, {0.2});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliX", {1});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});
    sv_ref.applyOperation("RX", {1}, false, {0.5});
    sv_ref.applyOperation("RX", {2}, false, {0.5});
    sv_ref.applyOperation("RY", {1}, true, {0.3});
    sv_ref.applyOperation("RZ", {3}, true, {0.2});

    std::size_t num_wires = 1;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire = GENERATE(0, 1, 2, 3);

    auto ob = HermitianObsMPI<StateVectorKokkosMPI<TestType>>(mat_ob, {wire});
    auto ob_ref = HermitianObs<StateVectorKokkos<TestType>>(mat_ob, {wire});
    auto res = m.var(ob);
    auto res_ref = m_ref.var(ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}

TEMPLATE_TEST_CASE("Var - 2-wire matrix Hermitian obs", "[LKMPI_Var]", float,
                   double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliX", {1});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});
    sv.applyOperation("RX", {1}, false, {0.5});
    sv.applyOperation("RX", {2}, false, {0.5});
    sv.applyOperation("RY", {1}, true, {0.3});
    sv.applyOperation("RZ", {3}, true, {0.2});
    sv.applyOperation("RX", {4}, false, {0.5});
    sv.applyOperation("RY", {5}, true, {0.3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliX", {1});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});
    sv_ref.applyOperation("RX", {1}, false, {0.5});
    sv_ref.applyOperation("RX", {2}, false, {0.5});
    sv_ref.applyOperation("RY", {1}, true, {0.3});
    sv_ref.applyOperation("RZ", {3}, true, {0.2});
    sv_ref.applyOperation("RX", {4}, false, {0.5});
    sv_ref.applyOperation("RY", {5}, true, {0.3});

    std::size_t num_wires = 2;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire_0 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t wire_1 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1};

    if (wires.size() == num_wires) {

        auto ob = HermitianObsMPI<StateVectorKokkosMPI<TestType>>(
            mat_ob, {wire_0, wire_1});
        auto ob_ref =
            HermitianObs<StateVectorKokkos<TestType>>(mat_ob, {wire_0, wire_1});
        auto res = m.var(ob);
        auto res_ref = m_ref.var(ob_ref);
        CHECK(res == Approx(res_ref).margin(EP));
    }
}

// var Tensorprod op
TEMPLATE_TEST_CASE("Var - TensorProdobs", "[LKMPI_Expval]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    StateVectorKokkosMPI<TestType> sv{num_qubits};
    StateVectorKokkos<TestType> sv_ref{num_qubits};
    // Only run with 4 mpi ranks:
    REQUIRE(sv.getMPISize() == 4);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] = init_sv[sv.getMPIRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("CNOT", {1, 3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("CNOT", {1, 3});

    auto X0_ref = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
        "PauliX", std::vector<std::size_t>{0});
    auto Z1_ref = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
        "PauliZ", std::vector<std::size_t>{1});
    auto X0 = std::make_shared<NamedObs<StateVectorKokkosMPI<TestType>>>(
        "PauliX", std::vector<std::size_t>{0});
    auto Z1 = std::make_shared<NamedObs<StateVectorKokkosMPI<TestType>>>(
        "PauliZ", std::vector<std::size_t>{1});

    auto ob_ref =
        TensorProdObs<StateVectorKokkos<TestType>>::create({X0_ref, Z1_ref});
    auto ob =
        TensorProdObsMPI<StateVectorKokkosMPI<TestType>>::create({X0, Z1});

    auto res = m.var(*ob);
    auto res_ref = m_ref.var(*ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}
