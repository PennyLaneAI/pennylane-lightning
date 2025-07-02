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

#include "StateVectorKokkosMPI.hpp"
#include "TestHelpers.hpp"             // createRandomStateVectorData
#include "TestHelpersStateVectors.hpp" // initializeLKTestSV
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

TEMPLATE_TEST_CASE("Apply op with wires more than local wires", "[LKMPI]",
                   double, float) {
    const std::size_t num_qubits = 4;
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);
    REQUIRE_THROWS_WITH(
        sv.applyOperation("XXX", {0, 1, 2}),
        Catch::Contains("smaller than or equal to the number of local wires."));

    std::vector<Kokkos::complex<TestType>> matrix(64, {0.0, 0.0});
    REQUIRE_THROWS_WITH(
        sv.applyMatrix(matrix, {0, 1, 2}),
        Catch::Contains("smaller than or equal to the number of local wires."));
}

TEMPLATE_TEST_CASE("Apply non-trivial op", "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);

    sv.applyOperation("PauliX", {0});
    sv.applyOperation("PauliX", {1});
    sv.applyOperation("PauliX", {2});
    sv.applyOperation("PauliX", {3});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("Hadamard", {1});
    sv.applyOperation("Hadamard", {2});
    sv.applyOperation("Hadamard", {3});
    sv.applyOperation("PauliY", {0});
    sv.applyOperation("PauliY", {1});
    sv.applyOperation("PauliY", {2});
    sv.applyOperation("PauliY", {3});
    sv.applyOperation("PauliZ", {0});
    sv.applyOperation("PauliZ", {1});
    sv.applyOperation("PauliZ", {2});
    sv.applyOperation("PauliZ", {3});
    sv.applyOperation("Hadamard", {0});
    sv.applyOperation("Hadamard", {1});
    sv.applyOperation("Hadamard", {2});
    sv.applyOperation("Hadamard", {3});

    sv_ref.applyOperation("PauliX", {0});
    sv_ref.applyOperation("PauliX", {1});
    sv_ref.applyOperation("PauliX", {2});
    sv_ref.applyOperation("PauliX", {3});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("Hadamard", {1});
    sv_ref.applyOperation("Hadamard", {2});
    sv_ref.applyOperation("Hadamard", {3});
    sv_ref.applyOperation("PauliY", {0});
    sv_ref.applyOperation("PauliY", {1});
    sv_ref.applyOperation("PauliY", {2});
    sv_ref.applyOperation("PauliY", {3});
    sv_ref.applyOperation("PauliZ", {0});
    sv_ref.applyOperation("PauliZ", {1});
    sv_ref.applyOperation("PauliZ", {2});
    sv_ref.applyOperation("PauliZ", {3});
    sv_ref.applyOperation("Hadamard", {0});
    sv_ref.applyOperation("Hadamard", {1});
    sv_ref.applyOperation("Hadamard", {2});
    sv_ref.applyOperation("Hadamard", {3});

    auto reference = sv_ref.getDataVector();
    auto data = getFullDataVector(sv, 0);

    if (sv.getMPIManager().getRank() == 0) {
        for (std::size_t j = 0; j < data.size(); j++) {
            CHECK(real(data[j]) == Approx(real(reference[j])).margin(EP));
            CHECK(imag(data[j]) == Approx(imag(reference[j])).margin(EP));
        }
    }
}

TEMPLATE_TEST_CASE("Apply Operation - non-param 1 wire", "[LKMPI]", double,
                   float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE(
        "Identity", "PauliX", "PauliY", "PauliZ", "Hadamard", "S", "SX", "T");
    const std::size_t wire = GENERATE(0, 1, 2, 3);

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Wire = " << wire) {
        sv.applyOperation(gate_name, {wire}, inverse);
        sv_ref.applyOperation(gate_name, {wire}, inverse);

        auto reference = sv_ref.getDataVector();
        auto data = getFullDataVector(sv, 0);

        if (sv.getMPIManager().getRank() == 0) {
            for (std::size_t j = 0; j < data.size(); j++) {
                CHECK(real(data[j]) == Approx(real(reference[j])).margin(EP));
                CHECK(imag(data[j]) == Approx(imag(reference[j])).margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Apply Operation - non-param 2 wires", "[LKMPI]", double,
                   float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE("CNOT", "SWAP", "CY", "CZ");
    const std::size_t wire_0 = GENERATE(0, 2, 4);
    const std::size_t wire_1 = GENERATE(1, 3, 5);

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Wire 0 = " << wire_0
                              << " Wire 1 =" << wire_1) {
        if (wire_0 != wire_1) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation(gate_name, {wire_0, wire_1}, inverse);
            sv_ref.applyOperation(gate_name, {wire_0, wire_1}, inverse);

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Apply Operation - non-param 3 wires", "[LKMPI]", double,
                   float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE("CSWAP", "Toffoli");
    const std::size_t wire_0 = GENERATE(0, 3);
    const std::size_t wire_1 = GENERATE(1, 4);
    const std::size_t wire_2 = GENERATE(2, 5);

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Wire 0 = " << wire_0 << " Wire 1 =" << wire_1
                              << " Wire 2 =" << wire_2) {
        if (wire_0 != wire_1 && wire_0 != wire_2 && wire_1 != wire_2) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation(gate_name, {wire_0, wire_1, wire_2}, inverse);
            sv_ref.applyOperation(gate_name, {wire_0, wire_1, wire_2}, inverse);

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "Apply Controlled Operation - non-param 1 control 1 target wire", "[LKMPI]",
    double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name =
        GENERATE("PauliX", "PauliY", "PauliZ", "Hadamard", "S", "SX", "T");
    const bool control_value = GENERATE(false, true);
    const std::size_t control_wire = GENERATE(0, 2, 4);
    const std::size_t target_wire = GENERATE(1, 3, 5);
    const std::set<std::size_t> wires = {target_wire, control_wire};

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Target Wire = " << target_wire
                              << " Control Wire = " << control_wire
                              << " Control Value = " << control_value) {
        if (wires.size() == 2) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation(gate_name, {control_wire}, {control_value},
                              {target_wire}, inverse, {});
            sv_ref.applyOperation(gate_name, {control_wire}, {control_value},
                                  {target_wire}, inverse, {});

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "Apply Controlled Operation - non-param 1 control 2 target wires",
    "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE("SWAP");
    const bool control_value = GENERATE(false, true);
    const std::size_t control_wire = GENERATE(0, 3);
    const std::size_t target_wire_0 = GENERATE(1, 4);
    const std::size_t target_wire_1 = GENERATE(2, 5);
    const std::set<std::size_t> wires = {target_wire_0, target_wire_1,
                                         control_wire};

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Target Wire 0 = " << target_wire_0
                              << " Target Wire 1 = " << target_wire_1
                              << " Control Wire = " << control_wire
                              << " Control Value = " << control_value) {
        if (wires.size() == 3) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation(gate_name, {control_wire}, {control_value},
                              {target_wire_0, target_wire_1}, inverse, {});
            sv_ref.applyOperation(gate_name, {control_wire}, {control_value},
                                  {target_wire_0, target_wire_1}, inverse, {});

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "Apply Controlled Operation - non-param 2 control 1 target wire", "[LKMPI]",
    double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name =
        GENERATE("PauliX", "PauliY", "PauliZ", "Hadamard", "S", "SX", "T");
    const bool control_value_0 = GENERATE(false, true);
    const std::size_t control_wire_0 = GENERATE(0, 3);
    const bool control_value_1 = GENERATE(false, true);
    const std::size_t control_wire_1 = GENERATE(1, 4);
    const std::size_t target_wire = GENERATE(0, 2, 5);
    const std::set<std::size_t> wires = {target_wire, control_wire_0,
                                         control_wire_1};

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Target Wire = " << target_wire
                              << " Control Wire 0 = " << control_wire_0
                              << " Control Value 0 = " << control_value_0
                              << " Control Wire 1 = " << control_wire_1
                              << " Control Value 1 = " << control_value_1) {
        if (wires.size() == 3) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation(gate_name, {control_wire_0, control_wire_1},
                              {control_value_0, control_value_1}, {target_wire},
                              inverse, {});
            sv_ref.applyOperation(gate_name, {control_wire_0, control_wire_1},
                                  {control_value_0, control_value_1},
                                  {target_wire}, inverse, {});

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE(
    "Apply Controlled Operation - non-param 2 control 2 target wire", "[LKMPI]",
    double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE("SWAP");
    const bool control_value_0 = GENERATE(false, true);
    const std::size_t control_wire_0 = GENERATE(0, 2, 4);
    const bool control_value_1 = GENERATE(false, true);
    const std::size_t control_wire_1 = GENERATE(1, 3, 5);
    const std::size_t target_wire_0 = GENERATE(0, 2, 4);
    const std::size_t target_wire_1 = GENERATE(1, 3, 5);
    const std::set<std::size_t> wires = {target_wire_0, target_wire_1,
                                         control_wire_0, control_wire_1};

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Target Wire 0 = " << target_wire_0
                              << " Target Wire 1 = " << target_wire_1
                              << " Control Wire 0 = " << control_wire_0
                              << " Control Value 0 = " << control_value_0
                              << " Control Wire 1 = " << control_wire_1
                              << " Control Value 1 = " << control_value_1) {
        if (wires.size() == 4) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation(gate_name, {control_wire_0, control_wire_1},
                              {control_value_0, control_value_1},
                              {target_wire_0, target_wire_1}, inverse, {});
            sv_ref.applyOperation(gate_name, {control_wire_0, control_wire_1},
                                  {control_value_0, control_value_1},
                                  {target_wire_0, target_wire_1}, inverse, {});

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Match wires after apply PauliX", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 4;
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);
    StateVectorKokkosMPI<TestType> sv_1(sv);

    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] =
            init_sv[mpi_manager.getRank() * block_size + element];
    }

    sv.updateData(init_subsv);
    sv_1.updateData(init_subsv);
    sv.applyOperation("PauliX", {0});

    std::vector<Kokkos::complex<TestType>> reference = {
        {8.0, 0.0},  {9.0, 0.0},  {10.0, 0.0}, {11.0, 0.0},
        {12.0, 0.0}, {13.0, 0.0}, {14.0, 0.0}, {15.0, 0.0},
        {0.0, 0.0},  {1.0, 0.0},  {2.0, 0.0},  {3.0, 0.0},
        {4.0, 0.0},  {5.0, 0.0},  {6.0, 0.0},  {7.0, 0.0},
    };
    sv_1.matchWires(sv);

    auto local_sv_1_data = sv_1.getLocalSV().getDataVector();

    for (std::size_t j = 0; j < init_subsv.size(); j++) {
        CHECK(real(local_sv_1_data[j]) ==
              Approx(real(reference[mpi_manager.getRank() * 4 + j])));
    }

    auto data = getFullDataVector(sv_1, 0);
    if (mpi_manager.getRank() == 0) {
        for (std::size_t j = 0; j < init_sv.size(); j++) {
            CHECK(real(data[j]) == Approx(real(init_sv[j])));
            CHECK(imag(data[j]) == Approx(imag(init_sv[j])));
        }
    }

    StateVectorKokkosMPI<TestType> sv_new(mpi_manager, num_qubits);
    sv.matchWires(sv_new);

    auto local_sv_data = sv.getLocalSV().getDataVector();
    for (std::size_t j = 0; j < init_subsv.size(); j++) {
        CHECK(real(local_sv_data[j]) ==
              Approx(real(reference[mpi_manager.getRank() * 4 + j])));
    }

    data = getFullDataVector(sv, 0);
    if (mpi_manager.getRank() == 0) {
        for (std::size_t j = 0; j < init_sv.size(); j++) {
            CHECK(real(data[j]) == Approx(real(reference[j])));
            CHECK(imag(data[j]) == Approx(imag(reference[j])));
        }
    }
}

// Apply matrix

TEMPLATE_TEST_CASE("Apply matrix - 1 wire", "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;
    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);

    const bool inverse = GENERATE(false, true);
    const std::size_t wire = GENERATE(0, 1, 2, 3);
    std::vector<Kokkos::complex<TestType>> matrix = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0},
        {5.0, 5.0}, {6.0, 6.0}, {7.0, 7.0}, {8.0, 8.0},
    };

    DYNAMIC_SECTION(" Inverse = " << inverse << " Wire = " << wire) {
        sv.applyOperation("matrix", {wire}, inverse, {}, matrix);
        sv_ref.applyOperation("matrix", {wire}, inverse, {}, matrix);

        auto reference = sv_ref.getDataVector();
        auto data = getFullDataVector(sv, 0);

        if (sv.getMPIManager().getRank() == 0) {
            for (std::size_t j = 0; j < data.size(); j++) {
                CHECK(real(data[j]) == Approx(real(reference[j])).margin(EP));
                CHECK(imag(data[j]) == Approx(imag(reference[j])).margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Apply matrix - 2 wires", "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::vector<Kokkos::complex<TestType>> matrix = {
        {1.0, 1.0},   {2.0, 2.0},   {3.0, 3.0},   {4.0, 4.0},
        {5.0, 5.0},   {6.0, 6.0},   {7.0, 7.0},   {8.0, 8.0},
        {9.0, 9.0},   {10.0, 10.0}, {11.0, 11.0}, {12.0, 12.0},
        {13.0, 13.0}, {14.0, 14.0}, {15.0, 15.0}, {16.0, 16.0},
    };

    const std::size_t wire_0 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t wire_1 = GENERATE(0, 1, 2, 3, 4, 5);

    DYNAMIC_SECTION(" Inverse = " << inverse << " Wire 0 = " << wire_0
                                  << " Wire 1 = " << wire_1) {
        if (wire_0 != wire_1) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation("matrix", {wire_0, wire_1}, inverse, {}, matrix);
            sv_ref.applyOperation("matrix", {wire_0, wire_1}, inverse, {},
                                  matrix);

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}
// Apply controlled matrix

TEMPLATE_TEST_CASE("Apply Controlled matrix - 1 control 1 target wire",
                   "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const bool control_value = GENERATE(false, true);
    const std::size_t control_wire = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t target_wire = GENERATE(0, 1, 2, 3, 4, 5);
    const std::set<std::size_t> wires = {target_wire, control_wire};
    const std::vector<Kokkos::complex<TestType>> matrix = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0},
        {5.0, 5.0}, {6.0, 6.0}, {7.0, 7.0}, {8.0, 8.0},
    };

    DYNAMIC_SECTION("Inverse = " << inverse << " Target Wire = " << target_wire
                                 << " Control Wire = " << control_wire
                                 << " Control Value = " << control_value) {
        if (wires.size() == 2) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation("matrix", {control_wire}, {control_value},
                              {target_wire}, inverse, {}, matrix);
            sv_ref.applyOperation("matrix", {control_wire}, {control_value},
                                  {target_wire}, inverse, {}, matrix);

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Apply Controlled matrix - 1 control 2 target wires",
                   "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::vector<Kokkos::complex<TestType>> matrix = {
        {1.0, 1.0},   {2.0, 2.0},   {3.0, 3.0},   {4.0, 4.0},
        {5.0, 5.0},   {6.0, 6.0},   {7.0, 7.0},   {8.0, 8.0},
        {9.0, 9.0},   {10.0, 10.0}, {11.0, 11.0}, {12.0, 12.0},
        {13.0, 13.0}, {14.0, 14.0}, {15.0, 15.0}, {16.0, 16.0},
    };
    const bool control_value = GENERATE(false, true);
    const std::size_t control_wire = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t target_wire_0 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::size_t target_wire_1 = GENERATE(0, 1, 2, 3, 4, 5);
    const std::set<std::size_t> wires = {target_wire_0, target_wire_1,
                                         control_wire};

    DYNAMIC_SECTION("Inverse = " << inverse
                                 << " Target Wire 0 = " << target_wire_0
                                 << " Target Wire 1 = " << target_wire_1
                                 << " Control Wire = " << control_wire
                                 << " Control Value = " << control_value) {

        if (wires.size() == 3) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation("matrix", {control_wire}, {control_value},
                              {target_wire_0, target_wire_1}, inverse, {},
                              matrix);
            sv_ref.applyOperation("matrix", {control_wire}, {control_value},
                                  {target_wire_0, target_wire_1}, inverse, {},
                                  matrix);

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Apply Controlled matrix - 2 control 1 target wire",
                   "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::vector<Kokkos::complex<TestType>> matrix = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0},
        {5.0, 5.0}, {6.0, 6.0}, {7.0, 7.0}, {8.0, 8.0},
    };
    const bool control_value_0 = GENERATE(false, true);
    const std::size_t control_wire_0 = GENERATE(0, 2, 4);
    const bool control_value_1 = GENERATE(false, true);
    const std::size_t control_wire_1 = GENERATE(1, 3, 5);
    const std::size_t target_wire = GENERATE(0, 2, 4, 5);
    const std::set<std::size_t> wires = {target_wire, control_wire_0,
                                         control_wire_1};

    DYNAMIC_SECTION("Inverse = " << inverse << " Target Wire = " << target_wire
                                 << " Control Wire 0 = " << control_wire_0
                                 << " Control Value 0 = " << control_value_0
                                 << " Control Wire 1 = " << control_wire_1
                                 << " Control Value 1 = " << control_value_1) {
        if (wires.size() == 3) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation("matrix", {control_wire_0, control_wire_1},
                              {control_value_0, control_value_1}, {target_wire},
                              inverse, {}, matrix);
            sv_ref.applyOperation("matrix", {control_wire_0, control_wire_1},
                                  {control_value_0, control_value_1},
                                  {target_wire}, inverse, {}, matrix);

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Apply Controlled matrix - 2 control 2 target wire",
                   "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::vector<Kokkos::complex<TestType>> matrix = {
        {1.0, 1.0},   {2.0, 2.0},   {3.0, 3.0},   {4.0, 4.0},
        {5.0, 5.0},   {6.0, 6.0},   {7.0, 7.0},   {8.0, 8.0},
        {9.0, 9.0},   {10.0, 10.0}, {11.0, 11.0}, {12.0, 12.0},
        {13.0, 13.0}, {14.0, 14.0}, {15.0, 15.0}, {16.0, 16.0},
    };
    const bool control_value_0 = GENERATE(false, true);
    const std::size_t control_wire_0 = GENERATE(0, 2, 4);
    const bool control_value_1 = GENERATE(false, true);
    const std::size_t control_wire_1 = GENERATE(1, 3, 5);
    const std::size_t target_wire_0 = GENERATE(0, 2, 4);
    const std::size_t target_wire_1 = GENERATE(1, 3, 5);
    const std::set<std::size_t> wires = {target_wire_0, target_wire_1,
                                         control_wire_0, control_wire_1};

    DYNAMIC_SECTION("Inverse = " << inverse
                                 << " Target Wire 0 = " << target_wire_0
                                 << " Target Wire 1 = " << target_wire_1
                                 << " Control Wire 0 = " << control_wire_0
                                 << " Control Value 0 = " << control_value_0
                                 << " Control Wire 1 = " << control_wire_1
                                 << " Control Value 1 = " << control_value_1) {
        if (wires.size() == 4) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation("matrix", {control_wire_0, control_wire_1},
                              {control_value_0, control_value_1},
                              {target_wire_0, target_wire_1}, inverse, {},
                              matrix);
            sv_ref.applyOperation("matrix", {control_wire_0, control_wire_1},
                                  {control_value_0, control_value_1},
                                  {target_wire_0, target_wire_1}, inverse, {},
                                  matrix);

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(data[j]) ==
                          Approx(real(reference[j])).margin(EP));
                    CHECK(imag(data[j]) ==
                          Approx(imag(reference[j])).margin(EP));
                }
            }
        }
    }
}
