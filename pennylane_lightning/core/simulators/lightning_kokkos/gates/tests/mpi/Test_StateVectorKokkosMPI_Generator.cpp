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
#include <cstdlib>
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

TEMPLATE_TEST_CASE("Apply Generator - 1 wire", "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;
    auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE("PhaseShift", "RX", "RY", "RZ");
    const std::size_t wire = GENERATE(0, 1, 2, 3, 4, 5);

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Wire = " << wire) {
        auto scale = sv.applyGenerator(gate_name, {wire}, inverse);
        auto scale_ref = sv_ref.applyGenerator(gate_name, {wire}, inverse);

        auto reference = sv_ref.getDataVector();
        auto data = getFullDataVector(sv, 0);

        if (sv.getMPIManager().getRank() == 0) {
            for (std::size_t j = 0; j < data.size(); j++) {
                CHECK(real(data[j]) == Approx(real(reference[j])).margin(EP));
                CHECK(imag(data[j]) == Approx(imag(reference[j])).margin(EP));
            }
        }
        CHECK(scale == scale_ref);
    }
}

TEMPLATE_TEST_CASE("Apply Generator - 2 wires", "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE(
        "ControlledPhaseShift", "CRX", "CRY", "CRZ", "IsingXX", "IsingXY",
        "IsingYY", "IsingZZ", "SingleExcitation", "SingleExcitationMinus",
        "SingleExcitationPlus", "PSWAP", "GlobalPhase", "MultiRZ");

    const std::size_t wire_0 = GENERATE(0, 1, 2);
    const std::size_t wire_1 = GENERATE(3, 4, 5);

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Wire 0 = " << wire_0
                              << " Wire 1 = " << wire_1) {
        if (wire_0 != wire_1) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);

            auto scale =
                sv.applyGenerator(gate_name, {wire_0, wire_1}, inverse);
            auto scale_ref =
                sv_ref.applyGenerator(gate_name, {wire_0, wire_1}, inverse);

            auto reference = sv_ref.getDataVector();
            auto data = getFullDataVector(sv, 0);

            if (sv.getMPIManager().getRank() == 0) {
                for (std::size_t j = 0; j < data.size(); j++) {
                    CHECK(real(reference[j]) ==
                          Approx(real(data[j])).margin(EP));
                    CHECK(imag(reference[j]) ==
                          Approx(imag(data[j])).margin(EP));
                }
            }
            CHECK(scale == scale_ref);
        }
    }
}

TEMPLATE_TEST_CASE("Apply Generator - 4 wires", "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 7;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name = GENERATE(
        "DoubleExcitation", "DoubleExcitationMinus", "DoubleExcitationPlus");

    const std::size_t wire_0 = GENERATE(0, 1, 2);
    const std::size_t wire_1 = GENERATE(1, 2, 3);
    const std::size_t wire_2 = GENERATE(3, 4);
    const std::size_t wire_3 = GENERATE(5, 6);
    std::set<std::size_t> wires = {wire_0, wire_1, wire_2, wire_3};

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Wire 0 = " << wire_0 << " Wire 1 = "
                              << wire_1 << " Wire 2 = " << wire_2
                              << " Wire 3 = " << wire_3) {
        if (wires.size() == 4) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            auto scale = sv.applyGenerator(
                gate_name, {wire_0, wire_1, wire_2, wire_3}, inverse);
            auto scale_ref = sv_ref.applyGenerator(
                gate_name, {wire_0, wire_1, wire_2, wire_3}, inverse);

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
            CHECK(scale == scale_ref);
        }
    }
}

// Apply controlled generator

TEMPLATE_TEST_CASE("Apply Controlled Generator - 1 control 1 target wire",
                   "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name =
        GENERATE("RX", "RY", "RZ", "PhaseShift", "GlobalPhase");
    const bool control_value = GENERATE(false, true);
    const std::size_t control_wire = GENERATE(0, 1, 4, 5);
    const std::size_t target_wire = GENERATE(0, 1, 2, 3);
    const std::set<std::size_t> wires = {target_wire, control_wire};

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Target Wire = " << target_wire
                              << " Control Wire = " << control_wire
                              << " Control Value = " << control_value) {
        if (wires.size() == 2) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            auto scale = sv.applyControlledGenerator(gate_name, {control_wire},
                                                     {control_value},
                                                     {target_wire}, inverse);
            auto scale_ref = sv_ref.applyControlledGenerator(
                gate_name, {control_wire}, {control_value}, {target_wire},
                inverse);

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
            CHECK(scale == scale_ref);
        }
    }
}

TEMPLATE_TEST_CASE("Apply Controlled Generator - 1 control 2 target wires",
                   "[LKMPI]", double, float) {
    const TestType EP = 1e-4;
    const std::size_t num_qubits = 6;

    const bool inverse = GENERATE(false, true);
    const std::string gate_name =
        GENERATE("IsingXX", "IsingXY", "IsingYY", "IsingZZ", "SingleExcitation",
                 "SingleExcitationMinus", "SingleExcitationPlus", "MultiRZ");
    const bool control_value = GENERATE(false, true);
    const std::size_t control_wire = GENERATE(0, 1, 4, 5);
    const std::size_t target_wire_0 = GENERATE(0, 1, 3);
    const std::size_t target_wire_1 = GENERATE(0, 1, 4, 5);
    const std::set<std::size_t> wires = {target_wire_0, target_wire_1,
                                         control_wire};

    DYNAMIC_SECTION("Gate = " << gate_name << " Inverse = " << inverse
                              << " Target Wire 0 = " << target_wire_0
                              << " Target Wire 1 = " << target_wire_1
                              << " Control Wire = " << control_wire
                              << " Control Value = " << control_value) {
        if (wires.size() == 3) {
            auto [sv, sv_ref] = initializeLKTestSV<TestType>(num_qubits);
            auto scale = sv.applyControlledGenerator(
                gate_name, {control_wire}, {control_value},
                {target_wire_0, target_wire_1}, inverse);
            auto scale_ref = sv_ref.applyControlledGenerator(
                gate_name, {control_wire}, {control_value},
                {target_wire_0, target_wire_1}, inverse);

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
            CHECK(scale == scale_ref);
        }
    }
}
