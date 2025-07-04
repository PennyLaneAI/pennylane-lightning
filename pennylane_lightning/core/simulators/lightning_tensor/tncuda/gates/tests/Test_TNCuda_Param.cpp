// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <complex>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "ExactTNCuda.hpp"
#include "Gates.hpp"
#include "MPSTNCuda.hpp"
#include "TNCudaGateCache.hpp"

#include "TestHelpers.hpp"
#include "TestHelpersTNCuda.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::LightningTensor::TNCuda::Gates;
using namespace Pennylane::Util;
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
/// @endcond

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::PhaseShift", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;

    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles{0.3, 0.8, 2.4};
    const Precision_T sign = (inverse) ? -1.0 : 1.0;
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(
            Pennylane::Gates::getPhaseShift<std::complex, Precision_T>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][3], ps_data[0][3], ps_data[0][3], ps_data[0][3]},
        {ps_data[1][0], ps_data[1][0], ps_data[1][3], ps_data[1][3],
         ps_data[1][0], ps_data[1][0], ps_data[1][3], ps_data[1][3]},
        {ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3],
         ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("PhaseShift", {index}, inverse,
                                 {sign * angles[index]});

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[index]).margin(1e-5));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::RX", "[TNCuda_Param]", TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles{0.3, 0.8, 2.4};

    // Results from default.qubit
    std::vector<cp_t> results = {{0.34958337, -0.05283436},
                                 {0.32564424, -0.13768018},
                                 {0.12811281, -0.32952558}};

    for (auto &val : results) {
        val = inverse ? std::conj(val) : val;
    }

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(std::size_t{1} << num_qubits, results[0]),
        std::vector<cp_t>(std::size_t{1} << num_qubits, results[1]),
        std::vector<cp_t>(std::size_t{1} << num_qubits, results[2]),
    };

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("RX", {index}, inverse, {angles[index]});

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }
}

/* python code to get the reference values for a single parameter
import pennylane as qml

qubits = 3
dev = qml.device('default.qubit', wires=qubits)

gate = qml.RY

invert = False
gate = qml.adjoint(gate) if invert else gate
wires=0

@qml.qnode(dev)
def circuit():
    [qml.H(i) for i in range(qubits)]
    gate(0.3, wires=wires)
    return qml.state()

result = circuit()
[print(f"{i:3d}  :  ",r) for i,r in enumerate(result)]
*/

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::RY", "[MPSTNCuda_Nonparam]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles{0.3, 0.8, 2.4};

    // Results from default.qubit
    std::vector<std::vector<cp_t>> expected_results{{{0.29674901, 0},
                                                     {0.29674901, 0},
                                                     {0.29674901, 0},
                                                     {0.29674901, 0},
                                                     {0.40241773, 0},
                                                     {0.40241773, 0},
                                                     {0.40241773, 0},
                                                     {0.40241773, 0}},
                                                    {{0.18796406, 0},
                                                     {0.18796406, 0},
                                                     {0.46332441, 0},
                                                     {0.46332441, 0},
                                                     {0.18796406, 0},
                                                     {0.18796406, 0},
                                                     {0.46332441, 0},
                                                     {0.46332441, 0}},
                                                    {{-0.20141277, 0},
                                                     {0.45763839, 0},
                                                     {-0.20141277, 0},
                                                     {0.45763839, 0},
                                                     {-0.20141277, 0},
                                                     {0.45763839, 0},
                                                     {-0.20141277, 0},
                                                     {0.45763839, 0}}};

    if (inverse) {
        std::swap(expected_results[0][4], expected_results[0][0]);
        std::swap(expected_results[0][5], expected_results[0][1]);
        std::swap(expected_results[0][6], expected_results[0][2]);
        std::swap(expected_results[0][7], expected_results[0][3]);

        std::swap(expected_results[1][2], expected_results[1][0]);
        std::swap(expected_results[1][3], expected_results[1][1]);
        std::swap(expected_results[1][6], expected_results[1][4]);
        std::swap(expected_results[1][7], expected_results[1][5]);

        std::swap(expected_results[2][1], expected_results[2][0]);
        std::swap(expected_results[2][3], expected_results[2][2]);
        std::swap(expected_results[2][5], expected_results[2][4]);
        std::swap(expected_results[2][7], expected_results[2][6]);
    }

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("RY", {index}, inverse, {angles[index]});

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::RZ", "[TNCuda_Param]", TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles{0.3, 0.8, 2.4};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{{{0.34958337, -0.05283436},
                                                     {0.34958337, -0.05283436},
                                                     {0.34958337, -0.05283436},
                                                     {0.34958337, -0.05283436},
                                                     {0.34958337, 0.05283436},
                                                     {0.34958337, 0.05283436},
                                                     {0.34958337, 0.05283436},
                                                     {0.34958337, 0.05283436}},

                                                    {{0.32564424, -0.13768018},
                                                     {0.32564424, -0.13768018},
                                                     {0.32564424, 0.13768018},
                                                     {0.32564424, 0.13768018},
                                                     {0.32564424, -0.13768018},
                                                     {0.32564424, -0.13768018},
                                                     {0.32564424, 0.13768018},
                                                     {0.32564424, 0.13768018}},

                                                    {{0.12811281, -0.32952558},
                                                     {0.12811281, 0.32952558},
                                                     {0.12811281, -0.32952558},
                                                     {0.12811281, 0.32952558},
                                                     {0.12811281, -0.32952558},
                                                     {0.12811281, 0.32952558},
                                                     {0.12811281, -0.32952558},
                                                     {0.12811281, 0.32952558}}};

    for (auto &vec : expected_results) {
        for (auto &val : vec) {
            val = inverse ? std::conj(val) : val;
        }
    }

    SECTION("Apply different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("RZ", {index}, inverse, {angles[index]});

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::ControlledPhaseShift", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles{0.3, 2.4};
    const Precision_T sign = (inverse) ? -1.0 : 1.0;
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(
            Pennylane::Gates::getPhaseShift<std::complex, Precision_T>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][0], ps_data[0][0], ps_data[0][3], ps_data[0][3]},
        {ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][0],
         ps_data[1][0], ps_data[1][3], ps_data[1][0], ps_data[1][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    SECTION("Apply adjacent wire indices") {
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});
        tn_state->applyOperation("ControlledPhaseShift", {0, 1}, inverse,
                                 {sign * angles[0]});

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[0]).margin(1e-6));
    }

    SECTION("Apply non-adjacent wire indices") {
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("ControlledPhaseShift", {0, 2}, inverse,
                                 {sign * angles[1]});
        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[1]).margin(1e-6));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::Rot", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<std::vector<Precision_T>> angles{
        std::vector<Precision_T>{0.3, 0.8, 2.4},
        std::vector<Precision_T>{0.5, 1.1, 3.0},
        std::vector<Precision_T>{2.3, 0.1, 0.4}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits)};

    for (std::size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat =
            Pennylane::Gates::getRot<std::complex, Precision_T>(
                angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = inverse ? std::conj(rot_mat[0]) : rot_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] =
            inverse ? -rot_mat[2] : rot_mat[2];
    }

    SECTION("Apply at different wire indices") {
        const std::size_t index = GENERATE(0, 1, 2);

        tn_state->applyOperation("Rot", {index}, inverse, angles[index]);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::CRot", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles =
        std::vector<Precision_T>{0.3, 0.8, 2.4};

    std::vector<cp_t> expected_results = std::vector<cp_t>(0b1 << num_qubits);

    SECTION("Apply adjacent wires") {
        tn_state->applyOperation("CRot", {0, 1}, inverse, angles);

        expected_results[0] = cp_t{1, 0};

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results));
    }

    SECTION("Apply non-adjacent wires") {
        tn_state->applyOperation("CRot", {0, 2}, inverse, angles);

        expected_results[0] = cp_t{1, 0};

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results));
    }
}

/* python code to get the reference values for a single parameter with 2 target
wires import pennylane as qml

qubits = 3
dev = qml.device('default.qubit', wires=qubits)

gate = qml.IsingXX

invert = False
gate = qml.adjoint(gate) if invert else gate
adjacent = True
wires = [0, 1] if adjacent  else [0,2]

@qml.qnode(dev)
def circuit():
    # [qml.H(i) for i in range(qubits)]
    gate(0.3, wires=wires)
    return qml.state()

result = circuit()
[print(f"{i:3d}  :  ",r) for i,r in enumerate(result)]
*/

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::IsingXX", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles{0.3, 0.8};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};

    expected_results[0][0] = {0.9887710779360422, 0.0};
    expected_results[0][6] = {0.0, -0.14943813247359922};

    expected_results[1][0] = {0.9210609940028851, 0.0};
    expected_results[1][6] = {0.0, -0.3894183423086505};

    expected_results[2][0] = {0.9887710779360422, 0.0};
    expected_results[2][5] = {0.0, -0.14943813247359922};

    expected_results[3][0] = {0.9210609940028851, 0.0};
    expected_results[3][5] = {0.0, -0.3894183423086505};

    for (auto &vec : expected_results) {
        for (auto &val : vec) {
            val = inverse ? std::conj(val) : val;
        }
    }

    SECTION("Apply adjacent wires") {
        const std::size_t index = GENERATE(0, 1);

        tn_state->applyOperation("IsingXX", {0, 1}, inverse, {angles[index]});

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[index]));
    }

    SECTION("Apply non-adjacent wires") {
        const std::size_t index = GENERATE(0, 1);

        tn_state->applyOperation("IsingXX", {0, 2}, inverse, {angles[index]});

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[index + angles.size()]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::IsingXY", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles = {0.3};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
        std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
    };

    expected_results[0][2] = {0.34958337, 0.05283436};
    expected_results[0][3] = {0.34958337, 0.05283436};
    expected_results[0][4] = {0.34958337, 0.05283436};
    expected_results[0][5] = {0.34958337, 0.05283436};

    expected_results[1][1] = {0.34958337, 0.05283436};
    expected_results[1][3] = {0.34958337, 0.05283436};
    expected_results[1][4] = {0.34958337, 0.05283436};
    expected_results[1][6] = {0.34958337, 0.05283436};

    for (auto &vec : expected_results) {
        for (auto &val : vec) {
            val = inverse ? std::conj(val) : val;
        }
    }

    SECTION("Apply adjacent wires") {
        tn_state->reset();

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("IsingXY", {0, 1}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[0]).margin(1e-6));
    }

    SECTION("Apply non-adjacent wires") {
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});
        tn_state->applyOperation("IsingXY", {0, 2}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[1]).margin(1e-6));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::IsingYY", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles = {0.3};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0.34958337, 0.05283436}),
        std::vector<cp_t>(1 << num_qubits, {0.34958337, 0.05283436}),
    };

    expected_results[0][2] = {0.34958337, -0.05283436};
    expected_results[0][3] = {0.34958337, -0.05283436};
    expected_results[0][4] = {0.34958337, -0.05283436};
    expected_results[0][5] = {0.34958337, -0.05283436};

    expected_results[1][1] = {0.34958337, -0.05283436};
    expected_results[1][3] = {0.34958337, -0.05283436};
    expected_results[1][4] = {0.34958337, -0.05283436};
    expected_results[1][6] = {0.34958337, -0.05283436};

    for (auto &vec : expected_results) {
        for (auto &val : vec) {
            val = inverse ? std::conj(val) : val;
        }
    }

    SECTION("Apply adjacent wires") {
        tn_state->reset();

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("IsingYY", {0, 1}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[0]));
    }

    SECTION("Apply non-adjacent wires") {
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("IsingYY", {0, 2}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[1]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::IsingZZ", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles = {0.3};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0.34958337, 0.05283436}),
        std::vector<cp_t>(1 << num_qubits, {0.34958337, 0.05283436}),
    };

    expected_results[0][0] = {0.34958337, -0.05283436};
    expected_results[0][1] = {0.34958337, -0.05283436};
    expected_results[0][6] = {0.34958337, -0.05283436};
    expected_results[0][7] = {0.34958337, -0.05283436};

    expected_results[1][0] = {0.34958337, -0.05283436};
    expected_results[1][2] = {0.34958337, -0.05283436};
    expected_results[1][5] = {0.34958337, -0.05283436};
    expected_results[1][7] = {0.34958337, -0.05283436};

    for (auto &vec : expected_results) {
        for (auto &val : vec) {
            val = inverse ? std::conj(val) : val;
        }
    }

    SECTION("Apply adjacent wires") {
        tn_state->reset();

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("IsingZZ", {0, 1}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[0]));
    }

    SECTION("Apply non-adjacent wires") {
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("IsingZZ", {0, 2}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[1]));
    }
}

/* python code to get the reference values for a single parameter with 2 target
for CRX import pennylane as qml

qubits = 3
dev = qml.device('default.qubit', wires=qubits)

gate = qml.CRX

invert = False
gate = qml.adjoint(gate) if invert else gate
adjacent = True
wires = [0,1] if adjacent else [0,2]

@qml.qnode(dev)
def circuit():
    [qml.H(i) for i in range(qubits-1)]
    gate(0.3, wires=wires)

    return qml.state()

result = circuit()
[print(f"{i:3d}  :  ",r) for i,r in enumerate(result)]
*/
TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::CRX", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles = {0.3};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0.0, 0.0}),
        std::vector<cp_t>(1 << num_qubits, {0.0, 0.0}),
    };

    // adjacent wires
    expected_results[0][0] = {0.5, 0.0};
    expected_results[0][2] = {0.5, 0.0};
    expected_results[0][4] = {0.49438553, -0.07471906};
    expected_results[0][6] = {0.49438553, -0.07471906};
    // non - adjacent wires
    expected_results[1][0] = {0.5, 0.0};
    expected_results[1][2] = {0.5, 0.0};
    expected_results[1][4] = {0.49438553, 0.0};
    expected_results[1][6] = {0.49438553, 0.0};
    expected_results[1][5] = {0.0, -0.07471906};
    expected_results[1][7] = {0.0, -0.07471906};

    for (auto &vec : expected_results) {
        for (auto &val : vec) {
            val = inverse ? std::conj(val) : val;
        }
    }

    SECTION("Apply adjacent wires") {
        tn_state->reset();

        tn_state->applyOperations({"Hadamard", "Hadamard"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("CRX", {0, 1}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[0]).margin(1e-6));
    }

    SECTION("Apply non-adjacent wires") {
        tn_state->applyOperations({"Hadamard", "Hadamard"}, {{0}, {1}},
                                  {false, false});

        tn_state->applyOperation("CRX", {0, 2}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[1]).margin(1e-6));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::CRY", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles = {0.3};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
        std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
    };

    expected_results[0][4] = {0.29674901, 0.0};
    expected_results[0][5] = {0.29674901, 0.0};
    expected_results[0][6] = {0.40241773, 0.0};
    expected_results[0][7] = {0.40241773, 0.0};

    if (inverse) {
        std::swap(expected_results[0][4], expected_results[0][6]);
        std::swap(expected_results[0][5], expected_results[0][7]);
    }

    expected_results[1][4] = {0.29674901, 0.0};
    expected_results[1][5] = {0.40241773, 0.0};
    expected_results[1][6] = {0.29674901, 0.0};
    expected_results[1][7] = {0.40241773, 0.0};

    if (inverse) {
        std::swap(expected_results[1][4], expected_results[1][5]);
        std::swap(expected_results[1][6], expected_results[1][7]);
    }

    SECTION("Apply adjacent wires") {
        tn_state->reset();

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("CRY", {0, 1}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[0]));
    }

    SECTION("Apply non-adjacent wires") {
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("CRY", {0, 2}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[1]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::CRZ", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles = {0.3};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
        std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
    };

    expected_results[0][4] = {0.34958337, -0.05283436};
    expected_results[0][5] = {0.34958337, -0.05283436};
    expected_results[0][6] = {0.34958337, 0.05283436};
    expected_results[0][7] = {0.34958337, 0.05283436};

    expected_results[1][4] = {0.34958337, -0.05283436};
    expected_results[1][5] = {0.34958337, 0.05283436};
    expected_results[1][6] = {0.34958337, -0.05283436};
    expected_results[1][7] = {0.34958337, 0.05283436};

    for (auto &vec : expected_results) {
        for (auto &val : vec) {
            val = inverse ? std::conj(val) : val;
        }
    }

    SECTION("Apply adjacent wires") {
        tn_state->reset();

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("CRZ", {0, 1}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[0]).margin(1e-6));
    }

    SECTION("Apply non-adjacent wires") {
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});
        tn_state->applyOperation("CRZ", {0, 2}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[1]).margin(1e-6));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::SingleExcitation", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles = {0.3};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
        std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
    };

    expected_results[0][2] = {0.29674901, 0.0};
    expected_results[0][3] = {0.29674901, 0.0};
    expected_results[0][4] = {0.40241773, 0.0};
    expected_results[0][5] = {0.40241773, 0.0};

    expected_results[1][1] = {0.29674901, 0.0};
    expected_results[1][3] = {0.29674901, 0.0};
    expected_results[1][4] = {0.40241773, 0.0};
    expected_results[1][6] = {0.40241773, 0.0};

    if (inverse) {
        std::swap(expected_results[0][2], expected_results[0][4]);
        std::swap(expected_results[0][3], expected_results[0][5]);
        std::swap(expected_results[1][1], expected_results[1][6]);
        std::swap(expected_results[1][3], expected_results[1][4]);
    }

    SECTION("Apply adjacent wires") {
        tn_state->reset();

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("SingleExcitation", {0, 1}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[0]));
    }

    SECTION("Apply non-adjacent wires") {
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("SingleExcitation", {0, 2}, inverse, angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results == Pennylane::Util::approx(expected_results[1]));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::SingleExcitationMinus",
                        "[TNCuda_Param]", TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles = {0.3};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0.34958337, -0.05283436}),
        std::vector<cp_t>(1 << num_qubits, {0.34958337, -0.05283436}),
    };

    for (auto &vec : expected_results) {
        for (auto &val : vec) {
            val = inverse ? std::conj(val) : val;
        }
    }

    expected_results[0][2] = {0.29674901, 0.0};
    expected_results[0][3] = {0.29674901, 0.0};
    expected_results[0][4] = {0.40241773, 0.0};
    expected_results[0][5] = {0.40241773, 0.0};

    expected_results[1][1] = {0.29674901, 0.0};
    expected_results[1][3] = {0.29674901, 0.0};
    expected_results[1][4] = {0.40241773, 0.0};
    expected_results[1][6] = {0.40241773, 0.0};

    if (inverse) {
        std::swap(expected_results[0][2], expected_results[0][4]);
        std::swap(expected_results[0][3], expected_results[0][5]);
        std::swap(expected_results[1][1], expected_results[1][6]);
        std::swap(expected_results[1][3], expected_results[1][4]);
    }

    SECTION("Apply adjacent wires") {
        tn_state->reset();

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("SingleExcitationMinus", {0, 1}, inverse,
                                 angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[0]).margin(1e-7));
    }

    SECTION("Apply non-adjacent wires") {
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("SingleExcitationMinus", {0, 2}, inverse,
                                 angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[1]).margin(1e-7));
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::SingleExcitationPlus", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 3;
    constexpr std::size_t maxExtent = 2;

    std::unique_ptr<TNDevice_T> tn_state =
        createTNState<TNDevice_T>(num_qubits, maxExtent);

    const std::vector<Precision_T> angles = {0.3};

    // Results collected from `default.qubit`
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0.34958337, 0.05283436}),
        std::vector<cp_t>(1 << num_qubits, {0.34958337, 0.05283436}),
    };

    for (auto &vec : expected_results) {
        for (auto &val : vec) {
            val = inverse ? std::conj(val) : val;
        }
    }

    expected_results[0][2] = {0.29674901, 0.0};
    expected_results[0][3] = {0.29674901, 0.0};
    expected_results[0][4] = {0.40241773, 0.0};
    expected_results[0][5] = {0.40241773, 0.0};

    expected_results[1][1] = {0.29674901, 0.0};
    expected_results[1][3] = {0.29674901, 0.0};
    expected_results[1][4] = {0.40241773, 0.0};
    expected_results[1][6] = {0.40241773, 0.0};

    if (inverse) {
        std::swap(expected_results[0][2], expected_results[0][4]);
        std::swap(expected_results[0][3], expected_results[0][5]);
        std::swap(expected_results[1][1], expected_results[1][6]);
        std::swap(expected_results[1][3], expected_results[1][4]);
    }

    SECTION("Apply adjacent wires") {
        tn_state->reset();

        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});

        tn_state->applyOperation("SingleExcitationPlus", {0, 1}, inverse,
                                 angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[0]).margin(1e-5));
    }

    SECTION("Apply non-adjacent wires") {
        tn_state->applyOperations({"Hadamard", "Hadamard", "Hadamard"},
                                  {{0}, {1}, {2}}, {false, false, false});
        tn_state->applyOperation("SingleExcitationPlus", {0, 2}, inverse,
                                 angles);

        tn_state_append_mps_final_state(tn_state);

        auto results = tn_state->getDataVector();

        CHECK(results ==
              Pennylane::Util::approx(expected_results[1]).margin(1e-5));
    }
}

/* python code to get the reference values for a single parameter with 4 target
   wires


    import pennylane as qml

    qubits = 5
    dev = qml.device('default.qubit', wires=qubits)

    gate = qml.DoubleExcitationPlus

    invert = True
    gate = qml.adjoint(gate) if invert else gate
    adjacent = True
    wires = [0,1,2,3] if adjacent else [0,1,3,4]

    @qml.qnode(dev)
    def circuit():
        [qml.H(i) for i in range(qubits)]
        gate(0.3, wires=wires)

        return qml.state()
    result = circuit()
    [print(f"{i:3d}  :  ",r) for i,r in enumerate(result)]
*/

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::DoubleExcitation", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 5;
    constexpr std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state;

    if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                  std::is_same_v<TestType, MPSTNCuda<float>>) {
        // Create the object for MPSTNCuda
        tn_state = std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);

        REQUIRE_THROWS_AS(
            tn_state->applyOperation("DoubleExcitation", {0, 1, 2, 3}, inverse),
            LightningException);
    } else {
        // Create the object for ExactTNCuda
        tn_state = std::make_unique<TNDevice_T>(num_qubits, dev_tag);

        const std::vector<Precision_T> angles = {0.3};

        // Results collected from `default.qubit`
        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits, {0.17677669, 0.0}),
            std::vector<cp_t>(1 << num_qubits, {0.17677669, 0.0}),
        };

        expected_results[0][6] = {0.14837450, 0.0};
        expected_results[0][7] = {0.14837450, 0.0};
        expected_results[0][24] = {0.20120886, 0.0};
        expected_results[0][25] = {0.20120886, 0.0};

        expected_results[1][3] = {0.14837450, 0.0};
        expected_results[1][7] = {0.14837450, 0.0};
        expected_results[1][24] = {0.20120886, 0.0};
        expected_results[1][28] = {0.20120886, 0.0};

        if (inverse) {
            std::swap(expected_results[0][6], expected_results[0][24]);
            std::swap(expected_results[0][7], expected_results[0][25]);
            std::swap(expected_results[1][3], expected_results[1][24]);
            std::swap(expected_results[1][7], expected_results[1][28]);
        }

        SECTION("Apply adjacent wires") {
            tn_state->reset();

            tn_state->applyOperations(
                {"Hadamard", "Hadamard", "Hadamard", "Hadamard", "Hadamard"},
                {{0}, {1}, {2}, {3}, {4}}, {false, false, false, false, false});

            tn_state->applyOperation("DoubleExcitation", {0, 1, 2, 3}, inverse,
                                     angles);

            auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent wires") {
            tn_state->reset();

            tn_state->applyOperations(
                {"Hadamard", "Hadamard", "Hadamard", "Hadamard", "Hadamard"},
                {{0}, {1}, {2}, {3}, {4}}, {false, false, false, false, false});
            tn_state->applyOperation("DoubleExcitation", {0, 1, 3, 4}, inverse,
                                     angles);

            auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::DoubleExcitationMinus",
                        "[TNCuda_Param]", TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 5;
    constexpr std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state;

    if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                  std::is_same_v<TestType, MPSTNCuda<float>>) {
        // Create the object for MPSTNCuda
        tn_state = std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);

        REQUIRE_THROWS_AS(tn_state->applyOperation("DoubleMinusExcitationMinus",
                                                   {0, 1, 2, 3}, inverse),
                          LightningException);
    } else {
        // Create the object for ExactTNCuda
        tn_state = std::make_unique<TNDevice_T>(num_qubits, dev_tag);

        const std::vector<Precision_T> angles = {0.3};

        // Results collected from `default.qubit`
        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits, {0.17479168, -0.02641717}),
            std::vector<cp_t>(1 << num_qubits, {0.17479168, -0.02641717}),
        };

        expected_results[0][6] = {0.14837450, 0.0};
        expected_results[0][7] = {0.14837450, 0.0};
        expected_results[0][24] = {0.20120886, 0.0};
        expected_results[0][25] = {0.20120886, 0.0};

        expected_results[1][3] = {0.14837450, 0.0};
        expected_results[1][7] = {0.14837450, 0.0};
        expected_results[1][24] = {0.20120886, 0.0};
        expected_results[1][28] = {0.20120886, 0.0};

        if (inverse) {
            for (size_t i = 0; i < expected_results[0].size(); i++) {
                expected_results[0][i] = std::conj(expected_results[0][i]);
                expected_results[1][i] = std::conj(expected_results[1][i]);
            }

            std::swap(expected_results[0][6], expected_results[0][24]);
            std::swap(expected_results[0][7], expected_results[0][25]);
            std::swap(expected_results[1][3], expected_results[1][24]);
            std::swap(expected_results[1][7], expected_results[1][28]);
        }

        SECTION("Apply adjacent wires") {
            tn_state->reset();

            tn_state->applyOperations(
                {"Hadamard", "Hadamard", "Hadamard", "Hadamard", "Hadamard"},
                {{0}, {1}, {2}, {3}, {4}}, {false, false, false, false, false});

            tn_state->applyOperation("DoubleExcitationMinus", {0, 1, 2, 3},
                                     inverse, angles);

            auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent wires") {
            tn_state->reset();

            tn_state->applyOperations(
                {"Hadamard", "Hadamard", "Hadamard", "Hadamard", "Hadamard"},
                {{0}, {1}, {2}, {3}, {4}}, {false, false, false, false, false});
            tn_state->applyOperation("DoubleExcitationMinus", {0, 1, 3, 4},
                                     inverse, angles);

            auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("TNCuda::Gates::DoubleExcitationPlus", "[TNCuda_Param]",
                        TestTNBackends) {
    const bool inverse = GENERATE(false, true);
    using TNDevice_T = TestType;
    using cp_t = typename TNDevice_T::ComplexT;
    using Precision_T = typename TNDevice_T::PrecisionT;
    constexpr std::size_t num_qubits = 5;
    constexpr std::size_t maxExtent = 2;
    DevTag<int> dev_tag{0, 0};

    std::unique_ptr<TNDevice_T> tn_state;

    if constexpr (std::is_same_v<TestType, MPSTNCuda<double>> ||
                  std::is_same_v<TestType, MPSTNCuda<float>>) {
        // Create the object for MPSTNCuda
        tn_state = std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);

        REQUIRE_THROWS_AS(tn_state->applyOperation("DoubleMinusExcitationPlus",
                                                   {0, 1, 2, 3}, inverse),
                          LightningException);
    } else {
        // Create the object for ExactTNCuda
        tn_state = std::make_unique<TNDevice_T>(num_qubits, dev_tag);

        const std::vector<Precision_T> angles = {0.3};

        // Results collected from `default.qubit`
        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits, {0.17479168, 0.02641717}),
            std::vector<cp_t>(1 << num_qubits, {0.17479168, 0.02641717}),
        };

        expected_results[0][6] = {0.14837450, 0.0};
        expected_results[0][7] = {0.14837450, 0.0};
        expected_results[0][24] = {0.20120886, 0.0};
        expected_results[0][25] = {0.20120886, 0.0};

        expected_results[1][3] = {0.14837450, 0.0};
        expected_results[1][7] = {0.14837450, 0.0};
        expected_results[1][24] = {0.20120886, 0.0};
        expected_results[1][28] = {0.20120886, 0.0};

        if (inverse) {
            for (auto &vec : expected_results) {
                for (auto &val : vec) {
                    val = std::conj(val);
                }
            }

            std::swap(expected_results[0][6], expected_results[0][24]);
            std::swap(expected_results[0][7], expected_results[0][25]);
            std::swap(expected_results[1][3], expected_results[1][24]);
            std::swap(expected_results[1][7], expected_results[1][28]);
        }

        SECTION("Apply adjacent wires") {
            tn_state->reset();

            tn_state->applyOperations(
                {"Hadamard", "Hadamard", "Hadamard", "Hadamard", "Hadamard"},
                {{0}, {1}, {2}, {3}, {4}}, {false, false, false, false, false});

            tn_state->applyOperation("DoubleExcitationPlus", {0, 1, 2, 3},
                                     inverse, angles);

            tn_state_append_mps_final_state(tn_state);

            auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent wires") {
            tn_state->reset();

            tn_state->applyOperations(
                {"Hadamard", "Hadamard", "Hadamard", "Hadamard", "Hadamard"},
                {{0}, {1}, {2}, {3}, {4}}, {false, false, false, false, false});
            tn_state->applyOperation("DoubleExcitationPlus", {0, 1, 3, 4},
                                     inverse, angles);

            tn_state_append_mps_final_state(tn_state);

            auto results = tn_state->getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}
