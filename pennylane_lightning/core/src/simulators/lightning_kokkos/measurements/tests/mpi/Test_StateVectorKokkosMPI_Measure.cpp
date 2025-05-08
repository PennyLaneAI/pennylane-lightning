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
#include "StateVectorKokkosMPI.hpp"
#include "ObservablesKokkos.hpp"
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
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2), {0.0, 0.0});
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
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2), {0.0, 0.0});
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
TEMPLATE_TEST_CASE("Expval - 1-wire matrix Hermitian obs", "[LKMPI_Expval]", float, double) {
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
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2), {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire = GENERATE(0, 1, 2, 3);

    auto ob = HermitianObs<StateVectorKokkosMPI<TestType>>(mat_ob, {wire});
    auto ob_ref = HermitianObs<StateVectorKokkos<TestType>>(mat_ob, {wire});
    auto res = m.expval(ob);
    auto res_ref = m_ref.expval(ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}


TEMPLATE_TEST_CASE("Expval - 2-wire matrix Hermitian obs", "[LKMPI_Expval]", float, double) {
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
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2), {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire_0 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t wire_1 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1};
    
    if (wires.size() == num_wires) {

    auto ob = HermitianObs<StateVectorKokkosMPI<TestType>>(mat_ob,  {wire_0, wire_1});
    auto ob_ref = HermitianObs<StateVectorKokkos<TestType>>(mat_ob, {wire_0, wire_1});
    auto res = m.expval(ob);
    auto res_ref = m_ref.expval(ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}
}

// expval namedobs
TEMPLATE_TEST_CASE("Expval - named obs", "[LKMPI_Expval]", float, double) {
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

    auto ob = NamedObs<StateVectorKokkosMPI<TestType>>(ob_name, {wire});
    auto ob_ref = NamedObs<StateVectorKokkos<TestType>>(ob_name, {wire});

    auto res = m.expval(ob);
    auto res_ref = m_ref.expval(ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}


//expval op list

// expval pauli word

// expval tensorprod op

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



//var matrix 
TEMPLATE_TEST_CASE("Var - 1-wire matrix", "[LKMPI_Var]",  double, float) {
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
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2), {0.0, 0.0});
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
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2), {0.0, 0.0});
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


//var op list

// var pauli word

// var Named obs
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

    auto ob = NamedObs<StateVectorKokkosMPI<TestType>>(ob_name, {wire});
    auto ob_ref = NamedObs<StateVectorKokkos<TestType>>(ob_name, {wire});

    auto res = m.var(ob);
    auto res_ref = m_ref.var(ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}

// var Hermitian obs
TEMPLATE_TEST_CASE("Var - 1-wire matrix Hermitian obs", "[LKMPI_Var]", float, double) {
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
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2), {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire = GENERATE(0, 1, 2, 3);

    auto ob = HermitianObs<StateVectorKokkosMPI<TestType>>(mat_ob, {wire});
    auto ob_ref = HermitianObs<StateVectorKokkos<TestType>>(mat_ob, {wire});
    auto res = m.var(ob);
    auto res_ref = m_ref.var(ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}


TEMPLATE_TEST_CASE("Var - 2-wire matrix Hermitian obs", "[LKMPI_Var]", float, double) {
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
    std::vector<Kokkos::complex<TestType>> mat_ob(exp2(num_wires * 2), {0.0, 0.0});
    for (std::size_t i = 0; i < mat_ob.size(); i++) {
        mat_ob[i] = i * Kokkos::complex<TestType>(0.2, 1.1);
    }

    const std::size_t wire_0 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t wire_1 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::set<std::size_t> wires = {wire_0, wire_1};
    
    if (wires.size() == num_wires) {

    auto ob = HermitianObs<StateVectorKokkosMPI<TestType>>(mat_ob,  {wire_0, wire_1});
    auto ob_ref = HermitianObs<StateVectorKokkos<TestType>>(mat_ob, {wire_0, wire_1});
    auto res = m.var(ob);
    auto res_ref = m_ref.var(ob_ref);
    CHECK(res == Approx(res_ref).margin(EP));
}
}


// var Tensorprod op
