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

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "Gates.hpp"
#include "MPSTNCuda.hpp"
#include "TNCudaGateCache.hpp"
#include "TestHelpers.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Util;
using namespace Pennylane;

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace

TEMPLATE_TEST_CASE("MPSTNCuda::applyPhaseShift", "[MPSTNCuda_Param]", float,
                   double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles{0.3, 0.8, 2.4};
        const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

        std::vector<std::vector<cp_t>> ps_data;
        ps_data.reserve(angles.size());
        for (auto &a : angles) {
            ps_data.push_back(Gates::getPhaseShift<std::complex, TestType>(a));
        }

        std::vector<std::vector<cp_t>> expected_results = {
            {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
             ps_data[0][3], ps_data[0][3], ps_data[0][3], ps_data[0][3]},
            {
                ps_data[1][0],
                ps_data[1][0],
                ps_data[1][3],
                ps_data[1][3],
                ps_data[1][0],
                ps_data[1][0],
                ps_data[1][3],
                ps_data[1][3],
            },
            {ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3],
             ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3]}};

        for (auto &vec : expected_results) {
            scaleVector(vec, coef);
        }

        SECTION("Apply different wire indices using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("PhaseShift", {index}, inverse,
                                        {angles[index]});

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyRX", "[MPSTNCuda_Param]", float, double) {
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles{0.3, 0.8, 2.4};

        // Results from default.qubit
        std::vector<cp_t> results = {{0.34958337, -0.05283436},
                                     {0.32564424, -0.13768018},
                                     {0.12811281, -0.32952558}};

        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(std::size_t{1} << num_qubits, results[0]),
            std::vector<cp_t>(std::size_t{1} << num_qubits, results[1]),
            std::vector<cp_t>(std::size_t{1} << num_qubits, results[2]),
        };

        SECTION("Apply different wire indices using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("RX", {index}, inverse,
                                        {angles[index]});

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyRY", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles{0.3, 0.8, 2.4};

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

        SECTION("Apply different wire indices using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("RY", {index}, inverse,
                                        {angles[index]});

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyRZ", "[MPSTNCuda_Param]", float, double) {
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles{0.3, 0.8, 2.4};

        // Results from default.qubit

        std::vector<std::vector<cp_t>> expected_results{
            {{0.34958337, -0.05283436},
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

        SECTION("Apply different wire indices using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("RZ", {index}, inverse,
                                        {angles[index]});

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyControlledPhaseShift", "[MPSTNCuda_Param]",
                   float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles{0.3, 2.4};
        const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

        std::vector<std::vector<cp_t>> ps_data;
        ps_data.reserve(angles.size());
        for (auto &a : angles) {
            ps_data.push_back(Gates::getPhaseShift<std::complex, TestType>(a));
        }

        std::vector<std::vector<cp_t>> expected_results = {
            {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
             ps_data[0][0], ps_data[0][0], ps_data[0][3], ps_data[0][3]},
            {ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][0],
             ps_data[1][0], ps_data[1][3], ps_data[1][0], ps_data[1][3]}};

        for (auto &vec : expected_results) {
            scaleVector(vec, coef);
        }

        SECTION("Apply adjacent wire indices using dispatcher") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("ControlledPhaseShift", {0, 1}, inverse,
                                        {angles[0]});

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent wire indices using dispatcher") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("ControlledPhaseShift", {0, 2}, inverse,
                                        {angles[1]});

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyRot", "[MPSTNCuda_param]", float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<std::vector<TestType>> angles{
            std::vector<TestType>{0.3, 0.8, 2.4},
            std::vector<TestType>{0.5, 1.1, 3.0},
            std::vector<TestType>{2.3, 0.1, 0.4}};

        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(0b1 << num_qubits),
            std::vector<cp_t>(0b1 << num_qubits),
            std::vector<cp_t>(0b1 << num_qubits)};

        for (size_t i = 0; i < angles.size(); i++) {
            const auto rot_mat =
                (inverse) ? Gates::getRot<std::complex, TestType>(
                                -angles[i][0], -angles[i][1], -angles[i][2])
                          : Gates::getRot<std::complex, TestType>(
                                angles[i][0], angles[i][1], angles[i][2]);
            expected_results[i][0] = rot_mat[0];
            expected_results[i][0b1 << (num_qubits - i - 1)] = rot_mat[2];
        }

        SECTION("Apply using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Rot", {index}, inverse, angles[index]);
            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyCRot", "[MPSTNCuda_param]", float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles =
            std::vector<TestType>{0.3, 0.8, 2.4};

        std::vector<cp_t> expected_results =
            std::vector<cp_t>(0b1 << num_qubits);

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("CRot", {0, 1}, inverse, angles);

            expected_results[0] = cp_t{1, 0};
            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(expected_results));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("CRot", {0, 2}, inverse, angles);

            expected_results[0] = cp_t{1, 0};
            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyIsingXX", "[MPSTNCuda_param]", float,
                   double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles{0.3, 0.8};

        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits),
            std::vector<cp_t>(1 << num_qubits),
            std::vector<cp_t>(1 << num_qubits),
            std::vector<cp_t>(1 << num_qubits)};
        expected_results[0][0] = {0.9887710779360422, 0.0};
        expected_results[0][6] = {0.0, -0.14943813247359922};

        expected_results[1][0] = {0.9210609940028851, 0.0};
        expected_results[1][6] = {0.0, -0.3894183423086505};

        expected_results[2][0] = {0.9887710779360422, 0.0};
        expected_results[2][5] = {0.0, -0.14943813247359922};

        expected_results[3][0] = {0.9210609940028851, 0.0};
        expected_results[3][5] = {0.0, -0.3894183423086505};

        SECTION("Apply adjacent sites") {
            const std::size_t index = GENERATE(0, 1);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("IsingXX", {0, 1}, inverse,
                                        {angles[index]});

            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }

        SECTION("Apply non-adjacent sites") {
            const std::size_t index = GENERATE(0, 1);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("IsingXX", {0, 2}, inverse,
                                        {angles[index]});

            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(
                      expected_results[index + angles.size()]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyIsingXY", "[MPSTNCuda_param]", float,
                   double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

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

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("IsingXY", {0, 1}, inverse, angles);

            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("IsingXY", {0, 2}, inverse, angles);

            auto results = sv.getDataVector();

            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyIsingYY", "[MPSTNCuda_param]", float,
                   double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

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

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("IsingYY", {0, 1}, inverse, angles);

            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("IsingYY", {0, 2}, inverse, angles);

            auto results = sv.getDataVector();

            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyIsingZZ", "[MPSTNCuda_param]", float,
                   double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

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

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("IsingZZ", {0, 1}, inverse, angles);

            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("IsingZZ", {0, 2}, inverse, angles);

            auto results = sv.getDataVector();

            CHECK(sv.getDataVector() ==
                  Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyCRX", "[MPSTNCuda_param]", float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
            std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
        };

        expected_results[0][4] = {0.34958337, -0.05283436};
        expected_results[0][5] = {0.34958337, -0.05283436};
        expected_results[0][6] = {0.34958337, -0.05283436};
        expected_results[0][7] = {0.34958337, -0.05283436};

        expected_results[1][4] = {0.34958337, -0.05283436};
        expected_results[1][5] = {0.34958337, -0.05283436};
        expected_results[1][6] = {0.34958337, -0.05283436};
        expected_results[1][7] = {0.34958337, -0.05283436};

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("CRX", {0, 1}, inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("CRX", {0, 2}, inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyCRY", "[MPSTNCuda_param]", float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
            std::vector<cp_t>(1 << num_qubits, {0.35355339, 0.0}),
        };

        expected_results[0][4] = {0.29674901, 0.0};
        expected_results[0][5] = {0.29674901, 0.0};
        expected_results[0][6] = {0.40241773, 0.0};
        expected_results[0][7] = {0.40241773, 0.0};

        expected_results[1][4] = {0.29674901, 0.0};
        expected_results[1][5] = {0.40241773, 0.0};
        expected_results[1][6] = {0.29674901, 0.0};
        expected_results[1][7] = {0.40241773, 0.0};

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("CRY", {0, 1}, inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("CRY", {0, 2}, inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyCRZ", "[MPSTNCuda_param]", float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

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

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("CRZ", {0, 1}, inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("CRZ", {0, 2}, inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applySingleExcitation", "[MPSTNCuda_param]",
                   float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

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

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("SingleExcitation", {0, 1}, inverse,
                                        angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("SingleExcitation", {0, 2}, inverse,
                                        angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applySingleExcitationMinus", "[MPSTNCuda_param]",
                   float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits, {0.34958337, -0.05283436}),
            std::vector<cp_t>(1 << num_qubits, {0.34958337, -0.05283436}),
        };

        expected_results[0][2] = {0.29674901, 0.0};
        expected_results[0][3] = {0.29674901, 0.0};
        expected_results[0][4] = {0.40241773, 0.0};
        expected_results[0][5] = {0.40241773, 0.0};

        expected_results[1][1] = {0.29674901, 0.0};
        expected_results[1][3] = {0.29674901, 0.0};
        expected_results[1][4] = {0.40241773, 0.0};
        expected_results[1][6] = {0.40241773, 0.0};

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("SingleExcitationMinus", {0, 1},
                                        inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("SingleExcitationMinus", {0, 2},
                                        inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applySingleExcitationPlus", "[MPSTNCuda_param]",
                   float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits, {0.34958337, 0.05283436}),
            std::vector<cp_t>(1 << num_qubits, {0.34958337, 0.05283436}),
        };

        expected_results[0][2] = {0.29674901, 0.0};
        expected_results[0][3] = {0.29674901, 0.0};
        expected_results[0][4] = {0.40241773, 0.0};
        expected_results[0][5] = {0.40241773, 0.0};

        expected_results[1][1] = {0.29674901, 0.0};
        expected_results[1][3] = {0.29674901, 0.0};
        expected_results[1][4] = {0.40241773, 0.0};
        expected_results[1][6] = {0.40241773, 0.0};

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("SingleExcitationPlus", {0, 1}, inverse,
                                        angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);

            sv.appendGateTensorOperator("SingleExcitationPlus", {0, 2}, inverse,
                                        angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyDoubleExcitation", "[MPSTNCuda_param]",
                   float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 5;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits, {0.1767767, 0.0}),
            std::vector<cp_t>(1 << num_qubits, {0.1767767, 0.0}),
        };

        expected_results[0][6] = {0.1483745, 0.0};
        expected_results[0][7] = {0.1483745, 0.0};
        expected_results[0][24] = {0.20120886, 0.0};
        expected_results[0][25] = {0.20120886, 0.0};

        expected_results[1][5] = {0.1483745, 0.0};
        expected_results[1][7] = {0.1483745, 0.0};
        expected_results[1][24] = {0.20120886, 0.0};
        expected_results[1][26] = {0.20120886, 0.0};

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("Hadamard", {3}, false);
            sv.appendGateTensorOperator("Hadamard", {4}, false);

            sv.appendGateTensorOperator("DoubleExcitation", {0, 1, 2, 3},
                                        inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("Hadamard", {3}, false);
            sv.appendGateTensorOperator("Hadamard", {4}, false);

            sv.appendGateTensorOperator("DoubleExcitation", {0, 1, 2, 4},
                                        inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyDoubleExcitationMinus", "[MPSTNCuda_param]",
                   float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 5;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits, {0.17479168, -0.02641718}),
            std::vector<cp_t>(1 << num_qubits, {0.17479168, -0.02641718}),
        };

        expected_results[0][6] = {0.1483745, 0.0};
        expected_results[0][7] = {0.1483745, 0.0};
        expected_results[0][24] = {0.20120886, 0.0};
        expected_results[0][25] = {0.20120886, 0.0};

        expected_results[1][5] = {0.1483745, 0.0};
        expected_results[1][7] = {0.1483745, 0.0};
        expected_results[1][24] = {0.20120886, 0.0};
        expected_results[1][26] = {0.20120886, 0.0};

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("Hadamard", {3}, false);
            sv.appendGateTensorOperator("Hadamard", {4}, false);

            sv.appendGateTensorOperator("DoubleExcitationMinus", {0, 1, 2, 3},
                                        inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("Hadamard", {3}, false);
            sv.appendGateTensorOperator("Hadamard", {4}, false);

            sv.appendGateTensorOperator("DoubleExcitationMinus", {0, 1, 2, 4},
                                        inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyDoubleExcitationPlus", "[MPSTNCuda_param]",
                   float, double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 5;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};

        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>(1 << num_qubits, {0.17479168, 0.02641718}),
            std::vector<cp_t>(1 << num_qubits, {0.17479168, 0.02641718}),
        };

        expected_results[0][6] = {0.1483745, 0.0};
        expected_results[0][7] = {0.1483745, 0.0};
        expected_results[0][24] = {0.20120886, 0.0};
        expected_results[0][25] = {0.20120886, 0.0};

        expected_results[1][5] = {0.1483745, 0.0};
        expected_results[1][7] = {0.1483745, 0.0};
        expected_results[1][24] = {0.20120886, 0.0};
        expected_results[1][26] = {0.20120886, 0.0};

        SECTION("Apply adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};
            sv.reset();

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("Hadamard", {3}, false);
            sv.appendGateTensorOperator("Hadamard", {4}, false);

            sv.appendGateTensorOperator("DoubleExcitationPlus", {0, 1, 2, 3},
                                        inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[0]));
        }

        SECTION("Apply non-adjacent sites") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("Hadamard", {3}, false);
            sv.appendGateTensorOperator("Hadamard", {4}, false);

            sv.appendGateTensorOperator("DoubleExcitationPlus", {0, 1, 2, 4},
                                        inverse, angles);

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[1]));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyMultiRZ", "[MPSTNCuda_param]", float,
                   double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        std::size_t num_qubits = 5;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<TestType> angles = {0.3};
        SECTION("Throw errors") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            REQUIRE_THROWS_WITH(sv.appendGateTensorOperator(
                                    "MultiRZ", {0, 1}, inverse, {angles[0]}),
                                Catch::Contains("Unsupported gate."));
        }
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::applyMatrix", "[MPSTNCuda_param]", float,
                   double) {
    // TODO only support inverse = false now
    const bool inverse = GENERATE(false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        const std::vector<cp_t> x_gate{cuUtil::ZERO<cp_t>(),
                                       cuUtil::ONE<cp_t>(), cuUtil::ONE<cp_t>(),
                                       cuUtil::ZERO<cp_t>()};

        SECTION("Append tensor operator with host data and wires") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("applyMatrix", {index}, inverse, {},
                                        x_gate);

            auto results = sv.getDataVector();

            CHECK(results[0] == cuUtil::ZERO<std::complex<TestType>>());
            CHECK(results[0b1 << (num_qubits - index - 1)] ==
                  cuUtil::ONE<std::complex<TestType>>());
        }
    }
}