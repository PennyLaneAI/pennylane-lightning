// Copyright 2025 Xanadu Quantum Technologies Inc.

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
#include "TestHelpers.hpp"             // createRandomStateVectorData
#include "TestHelpersStateVectors.hpp" // initializeLKTestSV, applyNonTrivialOperations

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
TEMPLATE_TEST_CASE("Expval - raise error", "[LKMPI_Expval]", float, double) {
    const std::size_t num_qubits = 4;
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);
    auto m = MeasurementsMPI(sv);

    std::size_t num_wires = 3;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    REQUIRE_THROWS_WITH(
        m.expval(mat_ob, {0, 1, 2}),
        Catch::Contains("Not enough local wires to swap with global wires."));
}

TEMPLATE_TEST_CASE("Expval - named string", "[LKMPI_Expval]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);

    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    std::size_t num_wires = 2;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire_0 = GENERATE(0, 2, 4);
    const std::size_t wire_1 = GENERATE(1, 3, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1};

    if (wires.size() == num_wires) {
        auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
        auto m = MeasurementsMPI(sv);
        auto m_ref = Measurements(sv_ref);

        applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    std::size_t num_wires = 2;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire_0 = GENERATE(0, 2, 4);
    const std::size_t wire_1 = GENERATE(1, 3, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1};

    if (wires.size() == num_wires) {
        auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
        auto m = MeasurementsMPI(sv);
        auto m_ref = Measurements(sv_ref);

        applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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
        const std::size_t num_qubits = 3;
        MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 4);

        StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);

        auto m = MeasurementsMPI(sv);

        // Set initial data
        std::size_t block_size = sv.getLocalBlockSize();
        std::vector<Kokkos::complex<TestType>> init_subsv(block_size,
                                                          {0.0, 0.0});
        for (std::size_t element = 0; element < block_size; element++) {
            init_subsv[element] =
                init_sv[mpi_manager.getRank() * block_size + element];
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
TEMPLATE_TEST_CASE("Expval - pauli word - 4 wires", "[LKMPI_Expval]", float,
                   double) {
    const TestType EP = std::is_same_v<TestType, float> ? 1e-3 : 1e-6;
    const std::size_t num_qubits = 6;

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

    const std::string P_0 = GENERATE("I", "X", "Y", "Z");
    const std::string P_1 = GENERATE("I", "X", "Y", "Z");
    const std::string P_2 = GENERATE("I", "X", "Y", "Z");
    const std::string P_3 = GENERATE("I", "X", "Y", "Z");
    const std::string ob = P_0 + P_1 + P_2 + P_3;

    const std::size_t wire_0 = 2;
    const std::size_t wire_1 = 4;
    const std::size_t wire_2 = GENERATE(0, 1, 3, 5);
    const std::size_t wire_3 = GENERATE(0, 1, 3, 5);
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

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);

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

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    std::size_t num_wires = 2;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire_0 = GENERATE(0, 2, 4);
    const std::size_t wire_1 = GENERATE(1, 3, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1};

    if (wires.size() == num_wires) {

        auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
        auto m = MeasurementsMPI(sv);
        auto m_ref = Measurements(sv_ref);

        applyNonTrivialOperations(num_qubits, sv, sv_ref);
        auto res = m.var(mat_ob, {wire_0, wire_1});
        auto res_ref = m_ref.var(mat_ob, {wire_0, wire_1});
        CHECK(res == Approx(res_ref).margin(EP));
    }
}

// var NamedObs
TEMPLATE_TEST_CASE("Var - named obs", "[LKMPI_Var]", float, double) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    std::size_t num_wires = 2;
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2),
                                                  {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire_0 = GENERATE(0, 2, 4);
    const std::size_t wire_1 = GENERATE(1, 3, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1};

    if (wires.size() == num_wires) {

        auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
        auto m = MeasurementsMPI(sv);
        auto m_ref = Measurements(sv_ref);

        applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

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

// probs
TEMPLATE_TEST_CASE("probs - 1 wires", "[LKMPI_Expval]", float, double) {
    const std::size_t num_qubits = 5;

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);
    std::size_t wire_0 = GENERATE(0, 1, 2, 3, 4);
    std::vector<std::size_t> wires = {wire_0};

    DYNAMIC_SECTION("Wires " << wire_0) {
        auto res = m.probs(wires);
        auto res_ref = m_ref.probs(wires);

        int local_probs_size = res.size();
        auto local_probs_sizes = sv.getMPIManager().allgather(local_probs_size);

        std::vector<TestType> res_gather(exp2(wires.size()));
        std::vector<int> displacements(local_probs_sizes.size());
        displacements[0] = 0;
        for (std::size_t i = 1; i < local_probs_sizes.size(); i++) {
            displacements[i] = displacements[i - 1] + local_probs_sizes[i - 1];
        }
        sv.getMPIManager().GatherV(res, res_gather, local_probs_sizes, 0,
                                   displacements);

        if (sv.getMPIManager().getRank() == 0) {
            CHECK_THAT(res_gather, Catch::Approx(res_ref).margin(1e-7));
        }
    }
}

TEMPLATE_TEST_CASE("probs - 2 wires", "[LKMPI_Expval]", float, double) {
    const std::size_t num_qubits = 5;

    std::size_t wire_0 = GENERATE(0, 1, 2, 3);
    std::size_t wire_1 = GENERATE(1, 2, 3, 4);

    std::vector<std::size_t> wires = {wire_0, wire_1};
    std::set<std::size_t> wires_set(wires.begin(), wires.end());

    DYNAMIC_SECTION("Wires " << wire_0 << ", " << wire_1) {
        if (wires_set.size() == 2 &&
            std::is_sorted(wires.begin(), wires.end())) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            auto m = MeasurementsMPI(sv);
            auto m_ref = Measurements(sv_ref);

            applyNonTrivialOperations(num_qubits, sv, sv_ref);

            auto res = m.probs(wires);
            auto res_ref = m_ref.probs(wires);

            int local_probs_size = res.size();
            auto local_probs_sizes =
                sv.getMPIManager().allgather(local_probs_size);

            std::vector<TestType> res_gather(exp2(wires.size()));
            std::vector<int> displacements(local_probs_sizes.size());
            displacements[0] = 0;
            for (std::size_t i = 1; i < local_probs_sizes.size(); i++) {
                displacements[i] =
                    displacements[i - 1] + local_probs_sizes[i - 1];
            }
            sv.getMPIManager().GatherV(res, res_gather, local_probs_sizes, 0,
                                       displacements);

            if (sv.getMPIManager().getRank() == 0) {
                CHECK_THAT(res_gather, Catch::Approx(res_ref).margin(1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("probs - 3 wires", "[LKMPI_Expval]", float, double) {
    const std::size_t num_qubits = 5;

    std::size_t wire_0 = GENERATE(0, 1, 2);
    std::size_t wire_1 = GENERATE(1, 2, 3);
    std::size_t wire_2 = GENERATE(2, 3, 4);

    std::vector<std::size_t> wires = {wire_0, wire_1, wire_2};
    std::set<std::size_t> wires_set(wires.begin(), wires.end());

    DYNAMIC_SECTION("Wires " << wire_0 << ", " << wire_1 << ", " << wire_2) {
        if (wires_set.size() == 3 &&
            std::is_sorted(wires.begin(), wires.end())) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            auto m = MeasurementsMPI(sv);
            auto m_ref = Measurements(sv_ref);

            applyNonTrivialOperations(num_qubits, sv, sv_ref);
            auto res = m.probs(wires);
            auto res_ref = m_ref.probs(wires);

            int local_probs_size = res.size();
            auto local_probs_sizes =
                sv.getMPIManager().allgather(local_probs_size);

            std::vector<TestType> res_gather(exp2(wires.size()));
            std::vector<int> displacements(local_probs_sizes.size());
            displacements[0] = 0;
            for (std::size_t i = 1; i < local_probs_sizes.size(); i++) {
                displacements[i] =
                    displacements[i - 1] + local_probs_sizes[i - 1];
            }
            sv.getMPIManager().GatherV(res, res_gather, local_probs_sizes, 0,
                                       displacements);

            if (sv.getMPIManager().getRank() == 0) {
                CHECK_THAT(res_gather, Catch::Approx(res_ref).margin(1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("probs - 4 wires", "[LKMPI_Expval]", float, double) {
    const std::size_t num_qubits = 5;

    std::size_t wire_0 = GENERATE(0, 1);
    std::size_t wire_1 = GENERATE(1, 2);
    std::size_t wire_2 = GENERATE(2, 3);
    std::size_t wire_3 = GENERATE(3, 4);

    std::vector<std::size_t> wires = {wire_0, wire_1, wire_2, wire_3};
    std::set<std::size_t> wires_set(wires.begin(), wires.end());

    DYNAMIC_SECTION("Wires " << wire_0 << ", " << wire_1 << ", " << wire_2
                             << ", " << wire_3) {
        if (wires_set.size() == 4 &&
            std::is_sorted(wires.begin(), wires.end())) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            auto m = MeasurementsMPI(sv);
            auto m_ref = Measurements(sv_ref);

            applyNonTrivialOperations(num_qubits, sv, sv_ref);
            auto res = m.probs(wires);
            auto res_ref = m_ref.probs(wires);

            int local_probs_size = res.size();
            auto local_probs_sizes =
                sv.getMPIManager().allgather(local_probs_size);

            std::vector<TestType> res_gather(exp2(wires.size()));
            std::vector<int> displacements(local_probs_sizes.size());
            displacements[0] = 0;
            for (std::size_t i = 1; i < local_probs_sizes.size(); i++) {
                displacements[i] =
                    displacements[i - 1] + local_probs_sizes[i - 1];
            }
            sv.getMPIManager().GatherV(res, res_gather, local_probs_sizes, 0,
                                       displacements);

            if (sv.getMPIManager().getRank() == 0) {
                CHECK_THAT(res_gather, Catch::Approx(res_ref).margin(1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("probs - 5 wires", "[LKMPI_Expval]", float, double) {
    const std::size_t num_qubits = 5;

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

    std::size_t wire_0 = 0;
    std::size_t wire_1 = 1;
    std::size_t wire_2 = 2;
    std::size_t wire_3 = 3;
    std::size_t wire_4 = 4;

    std::vector<std::size_t> wires = {wire_0, wire_1, wire_2, wire_3, wire_4};
    std::set<std::size_t> wires_set(wires.begin(), wires.end());

    DYNAMIC_SECTION("Wires " << wire_0 << ", " << wire_1 << ", " << wire_2
                             << ", " << wire_3 << ", " << wire_4) {
        if (wires_set.size() == 5 &&
            std::is_sorted(wires.begin(), wires.end())) {
            auto res = m.probs(wires);
            auto res_ref = m_ref.probs(wires);

            int local_probs_size = res.size();
            auto local_probs_sizes =
                sv.getMPIManager().allgather(local_probs_size);

            std::vector<TestType> res_gather(exp2(wires.size()));
            std::vector<int> displacements(local_probs_sizes.size());
            displacements[0] = 0;
            for (std::size_t i = 1; i < local_probs_sizes.size(); i++) {
                displacements[i] =
                    displacements[i - 1] + local_probs_sizes[i - 1];
            }
            sv.getMPIManager().GatherV(res, res_gather, local_probs_sizes, 0,
                                       displacements);

            if (sv.getMPIManager().getRank() == 0) {
                CHECK_THAT(res_gather, Catch::Approx(res_ref).margin(1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("probs - raise wire error", "[LKMPI_Expval]", float,
                   double) {
    const std::size_t num_qubits = 5;

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);

    REQUIRE_THROWS_WITH(m.probs({0, 3, 2}),
                        Catch::Contains("out-of-order wire"));
}

TEMPLATE_TEST_CASE("probs - all wires", "[LKMPI_Expval]", float, double) {
    const std::size_t num_qubits = 5;

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    applyNonTrivialOperations(num_qubits, sv, sv_ref);

    auto res = m.probs();
    auto res_ref = m_ref.probs();

    std::vector<TestType> res_gather(exp2(num_qubits));
    sv.getMPIManager().Gather(res, res_gather, 0);

    if (sv.getMPIManager().getRank() == 0) {
        CHECK_THAT(res_gather, Catch::Approx(res_ref).margin(1e-7));
    }
}

TEMPLATE_TEST_CASE("Generate Samples", "[LKMPI_Expval]", float, double) {
    const std::size_t num_qubits = 5;
    constexpr std::size_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    applyNonTrivialOperations(num_qubits, sv, sv_ref);
    auto m = MeasurementsMPI(sv);
    auto m_ref = Measurements(sv_ref);

    sv.normalize();
    sv_ref.normalize();

    std::size_t N = exp2(num_qubits);
    std::size_t num_samples = 100000;

    auto samples = m.generate_samples(num_samples);
    auto prob_ref = m_ref.probs();

    std::vector<std::size_t> counts(N, 0);
    std::vector<std::size_t> samples_decimal(num_samples, 0);

    // convert samples to decimal and then bin them in counts
    for (std::size_t i = 0; i < num_samples; i++) {
        for (std::size_t j = 0; j < num_qubits; j++) {
            if (samples[i * num_qubits + j] != 0) {
                samples_decimal[i] += twos[(num_qubits - 1 - j)];
            }
        }
        counts[samples_decimal[i]] += 1;
    }

    // compute estimated probabilities from histogram
    std::vector<TestType> probabilities(counts.size());
    for (std::size_t i = 0; i < counts.size(); i++) {
        probabilities[i] = counts[i] / (TestType)num_samples;
    }
    REQUIRE_THAT(probabilities, Catch::Approx(prob_ref).margin(.05));
}
