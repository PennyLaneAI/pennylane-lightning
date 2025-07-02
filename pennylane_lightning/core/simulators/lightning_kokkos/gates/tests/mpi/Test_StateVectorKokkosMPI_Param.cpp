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

TEMPLATE_TEST_CASE("Apply Operation - param 1 wire", "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 4;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name =
        GENERATE("PhaseShift", "RX", "RY", "RZ", "Rot");

    const TestType param = 0.12342;
    const std::size_t wire = GENERATE(0, 1, 2, 3);
    auto gate_op =
        reverse_lookup(Constant::gate_names, std::string_view{gate_name});
    auto num_params = lookup(Constant::gate_num_params, gate_op);
    auto params = std::vector<TestType>(num_params, param);

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Wire = " << wire) {
        auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
        sv.applyOperation(gate_name, {wire}, inverse, params);
        sv_ref.applyOperation(gate_name, {wire}, inverse, params);

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

TEMPLATE_TEST_CASE("Apply Operation - param 2 wires", "[LKMPI]", double,
                   float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE(
        "ControlledPhaseShift", "CRX", "CRY", "CRZ", "IsingXX", "IsingXY",
        "IsingYY", "IsingZZ", "SingleExcitation", "SingleExcitationMinus",
        "SingleExcitationPlus", "CRot", "PSWAP", "MultiRZ");

    const TestType param = 0.12342;
    const std::size_t wire_0 = GENERATE(0, 2, 4);
    const std::size_t wire_1 = GENERATE(1, 3, 5);
    auto gate_op =
        reverse_lookup(Constant::gate_names, std::string_view{gate_name});
    auto num_params = lookup(Constant::gate_num_params, gate_op);
    auto params = std::vector<TestType>(num_params, param);

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Wire 0 = " << wire_0
                              << " Wire 1 = " << wire_1) {
        if (wire_0 != wire_1) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation(gate_name, {wire_0, wire_1}, inverse, params);
            sv_ref.applyOperation(gate_name, {wire_0, wire_1}, inverse, params);

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

TEMPLATE_TEST_CASE("Apply Operation - param 4 wires", "[LKMPI]", double,
                   float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 7;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE(
        "DoubleExcitation", "DoubleExcitationMinus", "DoubleExcitationPlus");

    const TestType param = 0.12342;
    const std::size_t wire_0 = GENERATE(0, 2, 4);
    const std::size_t wire_1 = GENERATE(1, 3, 5);
    const std::size_t wire_2 = GENERATE(0, 2, 4);
    const std::size_t wire_3 = GENERATE(1, 3, 5);
    std::set<std::size_t> wires = {wire_0, wire_1, wire_2, wire_3};

    auto gate_op =
        reverse_lookup(Constant::gate_names, std::string_view{gate_name});
    auto num_params = lookup(Constant::gate_num_params, gate_op);
    auto params = std::vector<TestType>(num_params, param);

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Wire 0 = " << wire_0 << " Wire 1 = "
                              << wire_1 << " Wire 2 = " << wire_2
                              << " Wire 3 = " << wire_3) {
        if (wires.size() == 4) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            sv.applyOperation(gate_name, {wire_0, wire_1, wire_2, wire_3},
                              inverse, params);
            sv_ref.applyOperation(gate_name, {wire_0, wire_1, wire_2, wire_3},
                                  inverse, params);

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

TEMPLATE_TEST_CASE("Apply Controlled Operation - param 1 control 1 target wire",
                   "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE("RX", "RY", "RZ");
    const TestType param = 0.1232;
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
                              {target_wire}, inverse, {param});
            sv_ref.applyOperation(gate_name, {control_wire}, {control_value},
                                  {target_wire}, inverse, {param});

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
    "Apply Controlled Operation - param 1 control 2 target wires", "[LKMPI]",
    double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name =
        GENERATE("IsingXX", "IsingXY", "IsingYY", "IsingZZ", "SingleExcitation",
                 "SingleExcitationMinus", "SingleExcitationPlus", "MultiRZ");
    const TestType param = 0.1232;
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
                              {target_wire_0, target_wire_1}, inverse, {param});
            sv_ref.applyOperation(gate_name, {control_wire}, {control_value},
                                  {target_wire_0, target_wire_1}, inverse,
                                  {param});

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

TEMPLATE_TEST_CASE("Apply Controlled Operation - param 2 control 1 target wire",
                   "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE("RX");
    const TestType param = 0.1232;
    const bool control_value_0 = GENERATE(false, true);
    const std::size_t control_wire_0 = GENERATE(0, 3);
    const bool control_value_1 = GENERATE(false, true);
    const std::size_t control_wire_1 = GENERATE(1, 4);
    const std::size_t target_wire = GENERATE(0, 1, 2, 5);
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
                              inverse, {param});
            sv_ref.applyOperation(gate_name, {control_wire_0, control_wire_1},
                                  {control_value_0, control_value_1},
                                  {target_wire}, inverse, {param});

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

TEMPLATE_TEST_CASE("Apply Controlled Operation - param 2 control 2 target wire",
                   "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE("IsingXX");
    const TestType param = 0.1232;
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
                              {target_wire_0, target_wire_1}, inverse, {param});
            sv_ref.applyOperation(gate_name, {control_wire_0, control_wire_1},
                                  {control_value_0, control_value_1},
                                  {target_wire_0, target_wire_1}, inverse,
                                  {param});

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

TEMPLATE_TEST_CASE("Apply PauliRot - 1 wire", "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 5;

    const bool inverse = GENERATE(false, true);

    const std::size_t wire = GENERATE(0, 1, 2, 3, 4);
    const std::string op = GENERATE("X", "Y", "Z");
    const std::string word = op;
    const TestType param = 0.12342;

    DYNAMIC_SECTION("word = " << word << " Inverse = " << inverse
                              << " Wire = " << wire) {
        auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
        sv.applyPauliRot({wire}, inverse, {param}, word);
        sv_ref.applyPauliRot({wire}, inverse, {param}, word);

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

TEMPLATE_TEST_CASE("Apply PauliRot - 2 wires", "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 5;

    const bool inverse = GENERATE(false, true);

    const std::size_t wire_0 = GENERATE(0, 2, 4);
    const std::size_t wire_1 = GENERATE(1, 3);
    const std::string op_0 = GENERATE("X", "Y", "Z");
    const std::string op_1 = GENERATE("X", "Y", "Z");
    const std::string word = op_0 + op_1;
    const std::vector<TestType> param = {0.12342, 0.23436};

    DYNAMIC_SECTION("word = " << word << " Inverse = " << inverse
                              << " Wire_0 = " << wire_0
                              << " Wire_1 = " << wire_1) {
        auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
        if (wire_0 != wire_1) {
            sv.applyPauliRot({wire_0, wire_1}, inverse, {param}, word);
            sv_ref.applyPauliRot({wire_0, wire_1}, inverse, {param}, word);

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

TEMPLATE_TEST_CASE("Apply PauliRot - error", "[LKMPI]", double, float) {
    const std::size_t num_qubits = 5;
    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);

    PL_REQUIRE_THROWS_MATCHES(
        sv.applyPauliRot({0, 1, 3, 4}, true, {0.1, 0.2, 0.3, 0.4}, "XYXY"),
        LightningException, "Number of wires must be smaller than");
    PL_REQUIRE_THROWS_MATCHES(
        sv.applyPauliRot({0, 1, 3}, true, {0.1, 0.2, 0.3, 0.4}, "XYXY"),
        LightningException, "wires and word have incompatible dimensions.");
}
