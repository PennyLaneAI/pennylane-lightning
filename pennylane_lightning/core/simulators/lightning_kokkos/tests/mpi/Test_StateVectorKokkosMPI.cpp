// Copyright 2022-2025 Xanadu Quantum Technologies Inc.

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

#include "StateVectorKokkosMPI.hpp"
#include "TestHelpers.hpp"             // createRandomStateVectorData
#include "TestHelpersStateVectors.hpp" // initializeLKTestSV, applyNonTrivialOperations
#include "mpi.h"

/**
 * @file
 *  Tests for functionality for the class StateVectorKokkosMPI.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::Util;

using Pennylane::Util::isApproxEqual;
using Pennylane::Util::randomUnitary;

std::mt19937_64 re{1337};
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorKokkosMPI::Constructibility",
                   "[Default Constructibility]", StateVectorKokkosMPI<>) {
    SECTION("StateVectorBackend<>") {
        REQUIRE(!std::is_constructible_v<TestType>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorKokkosMPI::Constructibility",
                           "[General Constructibility]", (StateVectorKokkosMPI),
                           (float, double)) {
    using StateVectorT = TestType;
    using CFP_t = typename StateVectorT::CFP_t;

    SECTION("StateVectorBackend<TestType> {StateVectorT, MPIManagerKokkos, "
            "std::size_t, std::size_t, Kokkos::InitializationSettings}") {
        REQUIRE(std::is_constructible_v<StateVectorT, MPIManagerKokkos,
                                        std::size_t, std::size_t,
                                        Kokkos::InitializationSettings>);
    }

    SECTION("StateVectorBackend<TestType> {StateVectorT, MPIManagerKokkos, "
            "std::size_t,Kokkos::InitializationSettings}") {
        REQUIRE(
            std::is_constructible_v<StateVectorT, MPIManagerKokkos, std::size_t,
                                    Kokkos::InitializationSettings>);
    }

    SECTION("StateVectorBackend<TestType> {StateVectorT, "
            "std::size_t,Kokkos::InitializationSettings}") {
        REQUIRE(std::is_constructible_v<StateVectorT, std::size_t,
                                        Kokkos::InitializationSettings>);
    }

    SECTION("StateVectorBackend<TestType> {StateVectorT, std::size_t, "
            "std::size_t, Kokkos::InitializationSettings}") {
        REQUIRE(std::is_constructible_v<StateVectorT, std::size_t, std::size_t,
                                        Kokkos::InitializationSettings>);
    }

    SECTION("StateVectorBackend<TestType> {StateVectorT, MPI_Comm, "
            "std::size_t, std::size_t, Kokkos::InitializationSettings}") {
        REQUIRE(std::is_constructible_v<StateVectorT, MPI_Comm, std::size_t,
                                        std::size_t,
                                        Kokkos::InitializationSettings>);
    }

    SECTION(
        "StateVectorBackend<TestType> {StateVectorT, std::size_t, std::size_t, "
        "std::vector<Kokkos::complex>, Kokkos::InitializationSettings, "
        "MPI_Comm}") {
        REQUIRE(
            std::is_constructible_v<StateVectorT, std::size_t, std::size_t,
                                    std::vector<Kokkos::complex<CFP_t>>,
                                    Kokkos::InitializationSettings, MPI_Comm>);
    }

    SECTION("StateVectorBackend<TestType> {StateVectorT, MPI_Comm, "
            "std::size_t,Kokkos::InitializationSettings}") {
        REQUIRE(std::is_constructible_v<StateVectorT, MPI_Comm, std::size_t,
                                        Kokkos::InitializationSettings>);
    }
    SECTION(
        "StateVectorBackend<TestType> {const StateVectorBackend<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorT>);
    }
}

/**
 * Test StateVectorKokkosMPI wire-related helper methods
 */
TEMPLATE_TEST_CASE("Local/Global wire helpers", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 5;
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);

    // Test getNumGlobalWires and getNumLocalWires
    REQUIRE(sv.getNumGlobalWires() == 2);
    REQUIRE(sv.getNumLocalWires() == 3);

    // Test getGlobalWiresIndex
    REQUIRE(sv.getRevGlobalWireIndex(0) == 1);
    REQUIRE(sv.getRevGlobalWireIndex(1) == 0);

    PL_REQUIRE_THROWS_MATCHES(sv.getRevGlobalWireIndex(2), LightningException,
                              "Element not in vector");
    PL_REQUIRE_THROWS_MATCHES(sv.getRevLocalWireIndex(1), LightningException,
                              "Element not in vector");

    // Test getLocalWiresIndex
    REQUIRE(sv.getRevLocalWireIndex(2) == 2);
    REQUIRE(sv.getRevLocalWireIndex(3) == 1);
    REQUIRE(sv.getRevLocalWireIndex(4) == 0);

    // Test getGlobalWires and getLocalWires
    REQUIRE(sv.getGlobalWires() == std::vector<std::size_t>{0, 1});
    REQUIRE(sv.getLocalWires() == std::vector<std::size_t>{2, 3, 4});

    // Test getLocalWireIndices and getGlobalWiresIndices
    REQUIRE(sv.getLocalWireIndices({2, 4}) == std::vector<std::size_t>{0, 2});
    REQUIRE(sv.getLocalWireIndices({3, 2}) == std::vector<std::size_t>{1, 0});
    PL_REQUIRE_THROWS_MATCHES(sv.getLocalWireIndices({3, 0}),
                              LightningException, "Element not in vector");

    REQUIRE(sv.getGlobalWiresIndices({0, 1}) == std::vector<std::size_t>{0, 1});
    REQUIRE(sv.getGlobalWiresIndices({1, 0}) == std::vector<std::size_t>{1, 0});
    PL_REQUIRE_THROWS_MATCHES(sv.getGlobalWiresIndices({3, 0}),
                              LightningException, "Element not in vector");

    // Test findGlobalWires
    REQUIRE(sv.findGlobalWires({1, 3}) == std::vector<std::size_t>{1});
    REQUIRE(sv.findGlobalWires({1, 3, 0}) == std::vector<std::size_t>{1, 0});
    REQUIRE(sv.findGlobalWires({1, 0}) == std::vector<std::size_t>{1, 0});
    REQUIRE(sv.findGlobalWires({1, 0, 5}) == std::vector<std::size_t>{1, 0});
    REQUIRE(sv.findGlobalWires({5, 0, 1}) == std::vector<std::size_t>{0, 1});
    REQUIRE(sv.findGlobalWires({4, 3, 5}) == std::vector<std::size_t>{});
    REQUIRE(sv.findGlobalWires({1, 2, 0}) == std::vector<std::size_t>{1, 0});

    // Test isWiresLocal and isWiresGlobal
    REQUIRE(sv.isWiresLocal({2, 3}) == true);
    REQUIRE(sv.isWiresLocal({0, 1}) == false);
    REQUIRE(sv.isWiresLocal({2, 0}) == false);
    REQUIRE(sv.isWiresGlobal({0, 1}) == true);
    REQUIRE(sv.isWiresGlobal({2, 3}) == false);
    REQUIRE(sv.isWiresGlobal({2, 0}) == false);
    REQUIRE(sv.isWiresGlobal({2, 10}) == false);

    // Test getLocalWireIndex
    REQUIRE(sv.getLocalBlockSize() == 8);

    // Test getGlobal2LocalIndex
    REQUIRE(sv.global2localIndex(0) ==
            std::pair<std::size_t, std::size_t>{0, 0});
    REQUIRE(sv.global2localIndex(1) ==
            std::pair<std::size_t, std::size_t>{0, 1});
    REQUIRE(sv.global2localIndex(7) ==
            std::pair<std::size_t, std::size_t>{0, 7});
    REQUIRE(sv.global2localIndex(8) ==
            std::pair<std::size_t, std::size_t>{1, 0});
    REQUIRE(sv.global2localIndex(9) ==
            std::pair<std::size_t, std::size_t>{1, 1});
    REQUIRE(sv.global2localIndex(30) ==
            std::pair<std::size_t, std::size_t>{3, 6});
    REQUIRE(sv.global2localIndex(31) ==
            std::pair<std::size_t, std::size_t>{3, 7});

    PL_REQUIRE_THROWS_MATCHES(sv.global2localIndex(33), LightningException,
                              "out of bounds");
}

TEMPLATE_TEST_CASE("getDataVector", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 5;
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);
    REQUIRE(sv.getNumGlobalWires() == 2);
    REQUIRE(sv.getNumLocalWires() == 3);
    std::vector<Kokkos::complex<TestType>> reference_data(exp2(num_qubits),
                                                          {0.0, 0.0});
    reference_data[0] = 1.0;

    auto data = sv.getDataVector(0);
    if (mpi_manager.getRank() == 0) {
        for (std::size_t j = 0; j < reference_data.size(); j++) {
            CHECK(real(data[j]) == Approx(real(reference_data[j])));
            CHECK(imag(data[j]) == Approx(imag(reference_data[j])));
        }
    }
}

TEMPLATE_TEST_CASE("setBasisState", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 5;
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);

    for (std::size_t i = 0; i < exp2(num_qubits); i++) {

        std::vector<Kokkos::complex<TestType>> reference_data(32, {0.0, 0.0});
        reference_data[i] = 1.0;
        sv.setBasisState(i);
        auto data = sv.getDataVector(0);
        if (mpi_manager.getRank() == 0) {
            for (std::size_t j = 0; j < reference_data.size(); j++) {
                CHECK(real(data[j]) == Approx(real(reference_data[j])));
                CHECK(imag(data[j]) == Approx(imag(reference_data[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("updateData/getData/getDataVector", "[LKMPI]", double,
                   float) {
    const std::size_t num_qubits = 5;
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);

    // Set initial data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set divided initial data for initialization
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] =
            init_sv[mpi_manager.getRank() * block_size + element];
    }

    // Update the state vector with the initial data with updateData()
    sv.updateData(init_subsv);

    // Check getData()
    for (std::size_t j = 0; j < init_subsv.size(); j++) {
        CHECK(real(sv.getData()[j]) == Approx(real(init_subsv[j])));
        CHECK(imag(sv.getData()[j]) == Approx(imag(init_subsv[j])));
    }

    // Check getDataVector()
    auto data = sv.getDataVector(0);
    if (mpi_manager.getRank() == 0) {
        for (std::size_t j = 0; j < init_sv.size(); j++) {
            CHECK(real(data[j]) == Approx(real(init_sv[j])));
            CHECK(imag(data[j]) == Approx(imag(init_sv[j])));
        }
    }
}

TEMPLATE_TEST_CASE("resetIndices/initZeros", "[LKMPI]", double,
                   float) {
    const std::size_t num_qubits = 5;
    auto [sv,sv_ref] = initializeLKTestSV<TestType>(num_qubits);
    applyNonTrivialOperations(num_qubits, sv, sv_ref);

    sv.initZeros();
    CHECK(sv.getGlobalWires() == std::vector<std::size_t>{0, 1});
    CHECK(sv.getLocalWires() == std::vector<std::size_t>{2,3,4});
    CHECK(sv.getMPIRankToGlobalIndexMap() ==
          std::vector<std::size_t>{0, 1, 2, 3});
    
          Kokkos::complex<TestType> zero{0.0, 0.0};
    for (std::size_t j = 0; j < sv.getLocalBlockSize(); j++) {
        CHECK(real(sv.getData()[j]) == Approx(real(zero)));
        CHECK(imag(sv.getData()[j]) == Approx(imag(zero)));
    }
}

TEMPLATE_TEST_CASE("getLocalSV", "[LKMPI]", double,
                   float) {
    const std::size_t num_qubits = 5;
    StateVectorKokkosMPI<TestType> sv(num_qubits);
    auto local_sv = sv.getLocalSV();
    REQUIRE(local_sv.getNumQubits() == sv.getNumLocalWires());
}

TEMPLATE_TEST_CASE("setBasisState - explicit state", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 5;
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);

    std::vector<std::size_t> state = {1, 1};
    std::vector<std::vector<std::size_t>> wires = {{3, 4}, {0, 1}, {3, 0}};
    std::vector<std::size_t> reference_location = {3, 24, 18};

    for (std::size_t i = 0; i < wires.size(); i++) {
        std::vector<Kokkos::complex<TestType>> reference_data(exp2(num_qubits),
                                                              {0.0, 0.0});
        reference_data[reference_location[i]] = 1.0;
        sv.setBasisState(state, wires[i]);
        auto data = sv.getDataVector(0);
        if (mpi_manager.getRank() == 0) {
            for (std::size_t j = 0; j < reference_data.size(); j++) {
                CHECK(real(data[j]) == Approx(real(reference_data[j])));
                CHECK(imag(data[j]) == Approx(imag(reference_data[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("resetStateVector", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 5;
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);

    std::vector<std::size_t> state = {1, 1};
    std::vector<std::size_t> wires = {3, 4};
    sv.setBasisState(state, wires);
    sv.resetStateVector();

    std::vector<Kokkos::complex<TestType>> reference_data(exp2(num_qubits),
                                                            {0.0, 0.0});
    reference_data[0] = 1.0;
    auto data = sv.getDataVector(0);
    if (mpi_manager.getRank() == 0) {
        for (std::size_t j = 0; j < reference_data.size(); j++) {
            CHECK(real(data[j]) == Approx(real(reference_data[j])));
            CHECK(imag(data[j]) == Approx(imag(reference_data[j])));
        }
        
    }
}

TEMPLATE_TEST_CASE("Swap global local wires", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 5;
    std::vector<std::vector<Kokkos::complex<TestType>>> swapped_sv(18);
    swapped_sv[0] = {
        {0.0, 0.0},  {1.0, 0.0},  {2.0, 0.0},  {3.0, 0.0},  {16.0, 0.0},
        {17.0, 0.0}, {18.0, 0.0}, {19.0, 0.0}, {8.0, 0.0},  {9.0, 0.0},
        {10.0, 0.0}, {11.0, 0.0}, {24.0, 0.0}, {25.0, 0.0}, {26.0, 0.0},
        {27.0, 0.0}, {4.0, 0.0},  {5.0, 0.0},  {6.0, 0.0},  {7.0, 0.0},
        {20.0, 0.0}, {21.0, 0.0}, {22.0, 0.0}, {23.0, 0.0}, {12.0, 0.0},
        {13.0, 0.0}, {14.0, 0.0}, {15.0, 0.0}, {28.0, 0.0}, {29.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[1] = {
        {0.0, 0.0},  {1.0, 0.0},  {16.0, 0.0}, {17.0, 0.0}, {4.0, 0.0},
        {5.0, 0.0},  {20.0, 0.0}, {21.0, 0.0}, {8.0, 0.0},  {9.0, 0.0},
        {24.0, 0.0}, {25.0, 0.0}, {12.0, 0.0}, {13.0, 0.0}, {28.0, 0.0},
        {29.0, 0.0}, {2.0, 0.0},  {3.0, 0.0},  {18.0, 0.0}, {19.0, 0.0},
        {6.0, 0.0},  {7.0, 0.0},  {22.0, 0.0}, {23.0, 0.0}, {10.0, 0.0},
        {11.0, 0.0}, {26.0, 0.0}, {27.0, 0.0}, {14.0, 0.0}, {15.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[2] = {
        {0.0, 0.0},  {16.0, 0.0}, {2.0, 0.0},  {18.0, 0.0}, {4.0, 0.0},
        {20.0, 0.0}, {6.0, 0.0},  {22.0, 0.0}, {8.0, 0.0},  {24.0, 0.0},
        {10.0, 0.0}, {26.0, 0.0}, {12.0, 0.0}, {28.0, 0.0}, {14.0, 0.0},
        {30.0, 0.0}, {1.0, 0.0},  {17.0, 0.0}, {3.0, 0.0},  {19.0, 0.0},
        {5.0, 0.0},  {21.0, 0.0}, {7.0, 0.0},  {23.0, 0.0}, {9.0, 0.0},
        {25.0, 0.0}, {11.0, 0.0}, {27.0, 0.0}, {13.0, 0.0}, {29.0, 0.0},
        {15.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[3] = {
        {0.0, 0.0},  {1.0, 0.0},  {2.0, 0.0},  {3.0, 0.0},  {8.0, 0.0},
        {9.0, 0.0},  {10.0, 0.0}, {11.0, 0.0}, {4.0, 0.0},  {5.0, 0.0},
        {6.0, 0.0},  {7.0, 0.0},  {12.0, 0.0}, {13.0, 0.0}, {14.0, 0.0},
        {15.0, 0.0}, {16.0, 0.0}, {17.0, 0.0}, {18.0, 0.0}, {19.0, 0.0},
        {24.0, 0.0}, {25.0, 0.0}, {26.0, 0.0}, {27.0, 0.0}, {20.0, 0.0},
        {21.0, 0.0}, {22.0, 0.0}, {23.0, 0.0}, {28.0, 0.0}, {29.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[4] = {
        {0.0, 0.0},  {1.0, 0.0},  {8.0, 0.0},  {9.0, 0.0},  {4.0, 0.0},
        {5.0, 0.0},  {12.0, 0.0}, {13.0, 0.0}, {2.0, 0.0},  {3.0, 0.0},
        {10.0, 0.0}, {11.0, 0.0}, {6.0, 0.0},  {7.0, 0.0},  {14.0, 0.0},
        {15.0, 0.0}, {16.0, 0.0}, {17.0, 0.0}, {24.0, 0.0}, {25.0, 0.0},
        {20.0, 0.0}, {21.0, 0.0}, {28.0, 0.0}, {29.0, 0.0}, {18.0, 0.0},
        {19.0, 0.0}, {26.0, 0.0}, {27.0, 0.0}, {22.0, 0.0}, {23.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[5] = {
        {0.0, 0.0},  {8.0, 0.0},  {2.0, 0.0},  {10.0, 0.0}, {4.0, 0.0},
        {12.0, 0.0}, {6.0, 0.0},  {14.0, 0.0}, {1.0, 0.0},  {9.0, 0.0},
        {3.0, 0.0},  {11.0, 0.0}, {5.0, 0.0},  {13.0, 0.0}, {7.0, 0.0},
        {15.0, 0.0}, {16.0, 0.0}, {24.0, 0.0}, {18.0, 0.0}, {26.0, 0.0},
        {20.0, 0.0}, {28.0, 0.0}, {22.0, 0.0}, {30.0, 0.0}, {17.0, 0.0},
        {25.0, 0.0}, {19.0, 0.0}, {27.0, 0.0}, {21.0, 0.0}, {29.0, 0.0},
        {23.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[6] = {
        {0.0, 0.0},  {1.0, 0.0},  {8.0, 0.0},  {9.0, 0.0},  {16.0, 0.0},
        {17.0, 0.0}, {24.0, 0.0}, {25.0, 0.0}, {2.0, 0.0},  {3.0, 0.0},
        {10.0, 0.0}, {11.0, 0.0}, {18.0, 0.0}, {19.0, 0.0}, {26.0, 0.0},
        {27.0, 0.0}, {4.0, 0.0},  {5.0, 0.0},  {12.0, 0.0}, {13.0, 0.0},
        {20.0, 0.0}, {21.0, 0.0}, {28.0, 0.0}, {29.0, 0.0}, {6.0, 0.0},
        {7.0, 0.0},  {14.0, 0.0}, {15.0, 0.0}, {22.0, 0.0}, {23.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[7] = {
        {0.0, 0.0},  {1.0, 0.0},  {16.0, 0.0}, {17.0, 0.0}, {8.0, 0.0},
        {9.0, 0.0},  {24.0, 0.0}, {25.0, 0.0}, {4.0, 0.0},  {5.0, 0.0},
        {20.0, 0.0}, {21.0, 0.0}, {12.0, 0.0}, {13.0, 0.0}, {28.0, 0.0},
        {29.0, 0.0}, {2.0, 0.0},  {3.0, 0.0},  {18.0, 0.0}, {19.0, 0.0},
        {10.0, 0.0}, {11.0, 0.0}, {26.0, 0.0}, {27.0, 0.0}, {6.0, 0.0},
        {7.0, 0.0},  {22.0, 0.0}, {23.0, 0.0}, {14.0, 0.0}, {15.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[8] = {
        {0.0, 0.0},  {8.0, 0.0},  {2.0, 0.0},  {10.0, 0.0}, {16.0, 0.0},
        {24.0, 0.0}, {18.0, 0.0}, {26.0, 0.0}, {1.0, 0.0},  {9.0, 0.0},
        {3.0, 0.0},  {11.0, 0.0}, {17.0, 0.0}, {25.0, 0.0}, {19.0, 0.0},
        {27.0, 0.0}, {4.0, 0.0},  {12.0, 0.0}, {6.0, 0.0},  {14.0, 0.0},
        {20.0, 0.0}, {28.0, 0.0}, {22.0, 0.0}, {30.0, 0.0}, {5.0, 0.0},
        {13.0, 0.0}, {7.0, 0.0},  {15.0, 0.0}, {21.0, 0.0}, {29.0, 0.0},
        {23.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[9] = {
        {0.0, 0.0},  {16.0, 0.0}, {2.0, 0.0},  {18.0, 0.0}, {8.0, 0.0},
        {24.0, 0.0}, {10.0, 0.0}, {26.0, 0.0}, {4.0, 0.0},  {20.0, 0.0},
        {6.0, 0.0},  {22.0, 0.0}, {12.0, 0.0}, {28.0, 0.0}, {14.0, 0.0},
        {30.0, 0.0}, {1.0, 0.0},  {17.0, 0.0}, {3.0, 0.0},  {19.0, 0.0},
        {9.0, 0.0},  {25.0, 0.0}, {11.0, 0.0}, {27.0, 0.0}, {5.0, 0.0},
        {21.0, 0.0}, {7.0, 0.0},  {23.0, 0.0}, {13.0, 0.0}, {29.0, 0.0},
        {15.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[10] = {
        {0.0, 0.0},  {8.0, 0.0},  {16.0, 0.0}, {24.0, 0.0}, {4.0, 0.0},
        {12.0, 0.0}, {20.0, 0.0}, {28.0, 0.0}, {1.0, 0.0},  {9.0, 0.0},
        {17.0, 0.0}, {25.0, 0.0}, {5.0, 0.0},  {13.0, 0.0}, {21.0, 0.0},
        {29.0, 0.0}, {2.0, 0.0},  {10.0, 0.0}, {18.0, 0.0}, {26.0, 0.0},
        {6.0, 0.0},  {14.0, 0.0}, {22.0, 0.0}, {30.0, 0.0}, {3.0, 0.0},
        {11.0, 0.0}, {19.0, 0.0}, {27.0, 0.0}, {7.0, 0.0},  {15.0, 0.0},
        {23.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[11] = {
        {0.0, 0.0},  {16.0, 0.0}, {8.0, 0.0},  {24.0, 0.0}, {4.0, 0.0},
        {20.0, 0.0}, {12.0, 0.0}, {28.0, 0.0}, {2.0, 0.0},  {18.0, 0.0},
        {10.0, 0.0}, {26.0, 0.0}, {6.0, 0.0},  {22.0, 0.0}, {14.0, 0.0},
        {30.0, 0.0}, {1.0, 0.0},  {17.0, 0.0}, {9.0, 0.0},  {25.0, 0.0},
        {5.0, 0.0},  {21.0, 0.0}, {13.0, 0.0}, {29.0, 0.0}, {3.0, 0.0},
        {19.0, 0.0}, {11.0, 0.0}, {27.0, 0.0}, {7.0, 0.0},  {23.0, 0.0},
        {15.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[12] = {
        {0.0, 0.0},  {1.0, 0.0},  {16.0, 0.0}, {17.0, 0.0}, {8.0, 0.0},
        {9.0, 0.0},  {24.0, 0.0}, {25.0, 0.0}, {4.0, 0.0},  {5.0, 0.0},
        {20.0, 0.0}, {21.0, 0.0}, {12.0, 0.0}, {13.0, 0.0}, {28.0, 0.0},
        {29.0, 0.0}, {2.0, 0.0},  {3.0, 0.0},  {18.0, 0.0}, {19.0, 0.0},
        {10.0, 0.0}, {11.0, 0.0}, {26.0, 0.0}, {27.0, 0.0}, {6.0, 0.0},
        {7.0, 0.0},  {22.0, 0.0}, {23.0, 0.0}, {14.0, 0.0}, {15.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[13] = {
        {0.0, 0.0},  {1.0, 0.0},  {8.0, 0.0},  {9.0, 0.0},  {16.0, 0.0},
        {17.0, 0.0}, {24.0, 0.0}, {25.0, 0.0}, {2.0, 0.0},  {3.0, 0.0},
        {10.0, 0.0}, {11.0, 0.0}, {18.0, 0.0}, {19.0, 0.0}, {26.0, 0.0},
        {27.0, 0.0}, {4.0, 0.0},  {5.0, 0.0},  {12.0, 0.0}, {13.0, 0.0},
        {20.0, 0.0}, {21.0, 0.0}, {28.0, 0.0}, {29.0, 0.0}, {6.0, 0.0},
        {7.0, 0.0},  {14.0, 0.0}, {15.0, 0.0}, {22.0, 0.0}, {23.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[14] = {
        {0.0, 0.0},  {16.0, 0.0}, {2.0, 0.0},  {18.0, 0.0}, {8.0, 0.0},
        {24.0, 0.0}, {10.0, 0.0}, {26.0, 0.0}, {4.0, 0.0},  {20.0, 0.0},
        {6.0, 0.0},  {22.0, 0.0}, {12.0, 0.0}, {28.0, 0.0}, {14.0, 0.0},
        {30.0, 0.0}, {1.0, 0.0},  {17.0, 0.0}, {3.0, 0.0},  {19.0, 0.0},
        {9.0, 0.0},  {25.0, 0.0}, {11.0, 0.0}, {27.0, 0.0}, {5.0, 0.0},
        {21.0, 0.0}, {7.0, 0.0},  {23.0, 0.0}, {13.0, 0.0}, {29.0, 0.0},
        {15.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[15] = {
        {0.0, 0.0},  {8.0, 0.0},  {2.0, 0.0},  {10.0, 0.0}, {16.0, 0.0},
        {24.0, 0.0}, {18.0, 0.0}, {26.0, 0.0}, {1.0, 0.0},  {9.0, 0.0},
        {3.0, 0.0},  {11.0, 0.0}, {17.0, 0.0}, {25.0, 0.0}, {19.0, 0.0},
        {27.0, 0.0}, {4.0, 0.0},  {12.0, 0.0}, {6.0, 0.0},  {14.0, 0.0},
        {20.0, 0.0}, {28.0, 0.0}, {22.0, 0.0}, {30.0, 0.0}, {5.0, 0.0},
        {13.0, 0.0}, {7.0, 0.0},  {15.0, 0.0}, {21.0, 0.0}, {29.0, 0.0},
        {23.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[16] = {
        {0.0, 0.0},  {16.0, 0.0}, {8.0, 0.0},  {24.0, 0.0}, {4.0, 0.0},
        {20.0, 0.0}, {12.0, 0.0}, {28.0, 0.0}, {2.0, 0.0},  {18.0, 0.0},
        {10.0, 0.0}, {26.0, 0.0}, {6.0, 0.0},  {22.0, 0.0}, {14.0, 0.0},
        {30.0, 0.0}, {1.0, 0.0},  {17.0, 0.0}, {9.0, 0.0},  {25.0, 0.0},
        {5.0, 0.0},  {21.0, 0.0}, {13.0, 0.0}, {29.0, 0.0}, {3.0, 0.0},
        {19.0, 0.0}, {11.0, 0.0}, {27.0, 0.0}, {7.0, 0.0},  {23.0, 0.0},
        {15.0, 0.0}, {31.0, 0.0},
    };
    swapped_sv[17] = {
        {0.0, 0.0},  {8.0, 0.0},  {16.0, 0.0}, {24.0, 0.0}, {4.0, 0.0},
        {12.0, 0.0}, {20.0, 0.0}, {28.0, 0.0}, {1.0, 0.0},  {9.0, 0.0},
        {17.0, 0.0}, {25.0, 0.0}, {5.0, 0.0},  {13.0, 0.0}, {21.0, 0.0},
        {29.0, 0.0}, {2.0, 0.0},  {10.0, 0.0}, {18.0, 0.0}, {26.0, 0.0},
        {6.0, 0.0},  {14.0, 0.0}, {22.0, 0.0}, {30.0, 0.0}, {3.0, 0.0},
        {11.0, 0.0}, {19.0, 0.0}, {27.0, 0.0}, {7.0, 0.0},  {15.0, 0.0},
        {23.0, 0.0}, {31.0, 0.0},
    };
    std::vector<std::vector<std::size_t>> global_wires_to_swap = {
        {0},    {0},    {0},    {1},    {1},    {1},    {0, 1}, {0, 1}, {0, 1},
        {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}};
    std::vector<std::vector<std::size_t>> local_wires_to_swap = {
        {2},    {3},    {4},    {2},    {3},    {4},    {2, 3}, {3, 2}, {2, 4},
        {4, 2}, {3, 4}, {4, 3}, {2, 3}, {3, 2}, {2, 4}, {4, 2}, {3, 4}, {4, 3}};

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    for (std::size_t i = 0; i < swapped_sv.size(); i++) {
        // Initialize the state vector
        MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 4);
        StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);
        // Set initial data
        std::size_t block_size = sv.getLocalBlockSize();
        std::vector<Kokkos::complex<TestType>> init_subsv(block_size,
                                                          {0.0, 0.0});
        for (std::size_t element = 0; element < block_size; element++) {
            init_subsv[element] =
                init_sv[mpi_manager.getRank() * block_size + element];
        }

        // Update the state vector with the initial data with updateData()
        sv.updateData(init_subsv);
        // Swap Global Local wires
        sv.swapGlobalLocalWires(global_wires_to_swap[i],
                                local_wires_to_swap[i]);
        // Check getData()
        for (std::size_t j = 0; j < init_subsv.size(); j++) {
            CHECK(real(sv.getData()[j]) ==
                  Approx(real(
                      swapped_sv[i][mpi_manager.getRank() * block_size + j])));
            CHECK(imag(sv.getData()[j]) ==
                  Approx(imag(
                      swapped_sv[i][mpi_manager.getRank() * block_size + j])));
        }
        // Check getDataVector()
        auto data = sv.getDataVector(0);
        if (mpi_manager.getRank() == 0) {
            for (std::size_t j = 0; j < init_sv.size(); j++) {
                CHECK(real(data[j]) == Approx(real(init_sv[j])));
                CHECK(imag(data[j]) == Approx(imag(init_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Match local wires", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 5;

    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);

    StateVectorKokkosMPI<TestType> sv_1(sv);

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
        init_subsv[element] =
            init_sv[mpi_manager.getRank() * block_size + element];
    }

    // Update the state vector with the initial data with updateData()
    sv.updateData(init_subsv);
    sv_1.updateData(init_subsv);

    // Swap wires so local wires are out of order
    sv_1.swapGlobalLocalWires({0}, {2});
    sv_1.swapGlobalLocalWires({2}, {3});
    sv_1.swapGlobalLocalWires({3}, {0});

    // Match global wires and index
    sv.matchLocalWires(sv_1.getLocalWires());

    // Final global wires = {0, 1}
    // Final local wires = {3, 2, 4}
    std::vector<Kokkos::complex<TestType>> swapped_sv = {
        {0.0, 0.0},  {1.0, 0.0},  {4.0, 0.0},  {5.0, 0.0},  {2.0, 0.0},
        {3.0, 0.0},  {6.0, 0.0},  {7.0, 0.0},  {8.0, 0.0},  {9.0, 0.0},
        {12.0, 0.0}, {13.0, 0.0}, {10.0, 0.0}, {11.0, 0.0}, {14.0, 0.0},
        {15.0, 0.0}, {16.0, 0.0}, {17.0, 0.0}, {20.0, 0.0}, {21.0, 0.0},
        {18.0, 0.0}, {19.0, 0.0}, {22.0, 0.0}, {23.0, 0.0}, {24.0, 0.0},
        {25.0, 0.0}, {28.0, 0.0}, {29.0, 0.0}, {26.0, 0.0}, {27.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };

    for (std::size_t j = 0; j < init_subsv.size(); j++) {
        CHECK(real(sv.getData()[j]) ==
              Approx(real(swapped_sv[mpi_manager.getRank() * block_size + j])));
        CHECK(real(sv.getData()[j]) == Approx(real(sv_1.getData()[j])));
        CHECK(imag(sv.getData()[j]) ==
              Approx(imag(swapped_sv[mpi_manager.getRank() * block_size + j])));
        CHECK(imag(sv.getData()[j]) == Approx(imag(sv_1.getData()[j])));
    }
}

TEMPLATE_TEST_CASE("Match global wires", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 5;

    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);
    StateVectorKokkosMPI<TestType> sv_1(sv);

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
        init_subsv[element] =
            init_sv[mpi_manager.getRank() * block_size + element];
    }

    // Update the state vector with the initial data with updateData()
    sv.updateData(init_subsv);
    sv_1.updateData(init_subsv);

    // Swap wires so global wires are out of order
    sv_1.swapGlobalLocalWires({0}, {2});
    sv_1.swapGlobalLocalWires({1}, {0});
    sv_1.swapGlobalLocalWires({2}, {1});

    // Match global wires and index
    sv.matchGlobalWiresAndIndex(sv_1);
    ;

    // Final global wires = {1, 0}
    // Final local wires = {2, 3, 4}
    std::vector<Kokkos::complex<TestType>> swapped_sv = {
        {0.0, 0.0},  {1.0, 0.0},  {2.0, 0.0},  {3.0, 0.0},  {4.0, 0.0},
        {5.0, 0.0},  {6.0, 0.0},  {7.0, 0.0},  {16.0, 0.0}, {17.0, 0.0},
        {18.0, 0.0}, {19.0, 0.0}, {20.0, 0.0}, {21.0, 0.0}, {22.0, 0.0},
        {23.0, 0.0}, {8.0, 0.0},  {9.0, 0.0},  {10.0, 0.0}, {11.0, 0.0},
        {12.0, 0.0}, {13.0, 0.0}, {14.0, 0.0}, {15.0, 0.0}, {24.0, 0.0},
        {25.0, 0.0}, {26.0, 0.0}, {27.0, 0.0}, {28.0, 0.0}, {29.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };

    for (std::size_t j = 0; j < init_subsv.size(); j++) {
        CHECK(real(sv.getData()[j]) ==
              Approx(real(swapped_sv[mpi_manager.getRank() * block_size + j])));
        CHECK(real(sv.getData()[j]) == Approx(real(sv_1.getData()[j])));
        CHECK(imag(sv.getData()[j]) ==
              Approx(imag(swapped_sv[mpi_manager.getRank() * block_size + j])));
        CHECK(imag(sv.getData()[j]) == Approx(imag(sv_1.getData()[j])));
    }
}

TEMPLATE_TEST_CASE("Match wires", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 5;
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);
    StateVectorKokkosMPI<TestType> sv_1(sv);

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
        init_subsv[element] =
            init_sv[mpi_manager.getRank() * block_size + element];
    }

    // Update the state vector with the initial data with updateData()
    sv.updateData(init_subsv);
    sv_1.updateData(init_subsv);

    // Swap wires so global and local wires are out of order
    sv_1.swapGlobalLocalWires({0}, {2});
    sv_1.swapGlobalLocalWires({1}, {0});

    // Match global wires and index
    sv.matchWires(sv_1);

    // Final global wires = {2, 0}
    // Final local wires = {1, 3, 4}
    std::vector<Kokkos::complex<TestType>> swapped_sv = {
        {0.0, 0.0},  {1.0, 0.0},  {2.0, 0.0},  {3.0, 0.0},  {8.0, 0.0},
        {9.0, 0.0},  {10.0, 0.0}, {11.0, 0.0}, {16.0, 0.0}, {17.0, 0.0},
        {18.0, 0.0}, {19.0, 0.0}, {24.0, 0.0}, {25.0, 0.0}, {26.0, 0.0},
        {27.0, 0.0}, {4.0, 0.0},  {5.0, 0.0},  {6.0, 0.0},  {7.0, 0.0},
        {12.0, 0.0}, {13.0, 0.0}, {14.0, 0.0}, {15.0, 0.0}, {20.0, 0.0},
        {21.0, 0.0}, {22.0, 0.0}, {23.0, 0.0}, {28.0, 0.0}, {29.0, 0.0},
        {30.0, 0.0}, {31.0, 0.0},
    };

    for (std::size_t j = 0; j < init_subsv.size(); j++) {
        CHECK(real(sv.getData()[j]) ==
              Approx(real(swapped_sv[mpi_manager.getRank() * block_size + j])));
        CHECK(real(sv.getData()[j]) == Approx(real(sv_1.getData()[j])));
        CHECK(imag(sv.getData()[j]) ==
              Approx(imag(swapped_sv[mpi_manager.getRank() * block_size + j])));
        CHECK(imag(sv.getData()[j]) == Approx(imag(sv_1.getData()[j])));
    }
}
