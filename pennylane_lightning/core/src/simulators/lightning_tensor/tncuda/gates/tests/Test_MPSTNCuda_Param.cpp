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

TEMPLATE_TEST_CASE("MPSTNCuda::applyPhaseShift", "[MPSTNCuda_Nonparam]", float,
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

TEMPLATE_TEST_CASE("MPSTNCuda::applyRX", "[MPSTNCuda_Nonparam]", float,
                   double) {
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

TEMPLATE_TEST_CASE("MPSTNCuda::applyRZ", "[MPSTNCuda_Nonparam]", float,
                   double) {
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

TEMPLATE_TEST_CASE("MPSTNCuda::applyControlledPhaseShift",
                   "[MPSTNCuda_Nonparam]", float, double) {
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
            {ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3],
             ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3]}};

        for (auto &vec : expected_results) {
            scaleVector(vec, coef);
        }

        SECTION("Apply adjacent wire indices using dispatcher") {
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {0}, false);
            sv.appendGateTensorOperator("Hadamard", {1}, false);
            sv.appendGateTensorOperator("Hadamard", {2}, false);
            sv.appendGateTensorOperator("ControlledPhaseShift", {index},
                                        inverse, {angles[index]});

            auto results = sv.getDataVector();

            CHECK(results == Pennylane::Util::approx(expected_results[index]));
        }
    }
}
