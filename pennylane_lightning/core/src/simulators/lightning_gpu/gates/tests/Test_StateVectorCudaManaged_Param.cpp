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
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "Gates.hpp"
#include "TestHelpers.hpp"

#include "LinearAlg.hpp"
#include "StateVectorCudaManaged.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::LightningGPU;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("LightningGPU::applyRX", "[LightningGPU_Param]", double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 1;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{{0.1}, {0.6}};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.9987502603949663, 0.0},
                          {0.0, -0.04997916927067834}},
        std::vector<cp_t>{{0.9553364891256061, 0.0}, {0, -0.2955202066613395}},
        std::vector<cp_t>{{0.49757104789172696, 0.0}, {0, -0.867423225594017}}};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>{{0.9987502603949663, 0.0},
                          {0.0, 0.04997916927067834}},
        std::vector<cp_t>{{0.9553364891256061, 0.0}, {0, 0.2955202066613395}},
        std::vector<cp_t>{{0.49757104789172696, 0.0}, {0, 0.867423225594017}}};

    const auto init_state = sv.getDataVector();
    SECTION("adj = false") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyRX({0}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                             init_state.size()};
                sv_dispatch.applyOperation("RX", {0}, false, {angles[index]});
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
    SECTION("adj = true") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyRX({0}, true, {angles[index]});
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                             init_state.size()};
                sv_dispatch.applyOperation("RX", {0}, true, {angles[index]});
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyRY", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;

    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>{{0.8731983044562817, 0.04786268954660339},
                          {0.0876120655431924, -0.47703040785184303}},
        std::vector<cp_t>{{0.8243771119105122, 0.16439396602553008},
                          {0.3009211363333468, -0.45035926880694604}},
        std::vector<cp_t>{{0.10575112905629831, 0.47593196040758534},
                          {0.8711876098966215, -0.0577721051072477}}};
    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>{{0.8731983044562817, -0.04786268954660339},
                          {-0.0876120655431924, -0.47703040785184303}},
        std::vector<cp_t>{{0.8243771119105122, -0.16439396602553008},
                          {-0.3009211363333468, -0.45035926880694604}},
        std::vector<cp_t>{{0.10575112905629831, -0.47593196040758534},
                          {-0.8711876098966215, -0.0577721051072477}}};

    const std::vector<cp_t> init_state{{0.8775825618903728, 0.0},
                                       {0.0, -0.47942553860420306}};
    SECTION("adj = false") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyRY({0}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                             init_state.size()};
                sv_dispatch.applyOperation("RY", {0}, false, {angles[index]});
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
    SECTION("adj = true") {
        SECTION("Apply directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyRY({0}, true, {angles[index]});
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                             init_state.size()};
                sv_dispatch.applyOperation("RY", {0}, true, {angles[index]});
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyRZ", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    // Test using |+++> state
    sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                      {{0}, {1}, {2}}, {{false}, {false}, {false}});
    const std::vector<TestType> angles{0.2, 0.7, 2.9};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> rz_data;
    rz_data.reserve(angles.size());
    for (auto &a : angles) {
        rz_data.push_back(Gates::getRZ<std::complex, TestType>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {rz_data[0][0], rz_data[0][0], rz_data[0][0], rz_data[0][0],
         rz_data[0][3], rz_data[0][3], rz_data[0][3], rz_data[0][3]},
        {
            rz_data[1][0],
            rz_data[1][0],
            rz_data[1][3],
            rz_data[1][3],
            rz_data[1][0],
            rz_data[1][0],
            rz_data[1][3],
            rz_data[1][3],
        },
        {rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3],
         rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                       init_state.size()};

            sv_direct.applyRZ({index}, false, {angles[index]});
            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                         init_state.size()};
            sv_dispatch.applyOperation("RZ", {index}, false, {angles[index]});

            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyPhaseShift", "[LightningGPU_Param]",
                   float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    // Test using |+++> state
    sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                      {{0}, {1}, {2}}, {{false}, {false}, {false}});

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

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                       init_state.size()};

            sv_direct.applyPhaseShift({index}, false, {angles[index]});
            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                         init_state.size()};
            sv_dispatch.applyOperation("PhaseShift", {index}, false,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyControlledPhaseShift",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    // Test using |+++> state
    sv.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                      {{0}, {1}, {2}}, {{false}, {false}, {false}});

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

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly") {
        StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                   init_state.size()};

        sv_direct.applyControlledPhaseShift({0, 1}, false, {angles[0]});
        CHECK(sv_direct.getDataVector() ==
              Pennylane::Util::approx(expected_results[0]));
    }
    SECTION("Apply using dispatcher") {
        StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                     init_state.size()};
        sv_dispatch.applyOperation("ControlledPhaseShift", {1, 2}, false,
                                   {angles[1]});
        CHECK(sv_dispatch.getDataVector() ==
              Pennylane::Util::approx(expected_results[1]));
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyRot", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;

    const std::vector<std::vector<TestType>> angles{
        std::vector<TestType>{0.3, 0.8, 2.4},
        std::vector<TestType>{0.5, 1.1, 3.0},
        std::vector<TestType>{2.3, 0.1, 0.4}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat = Gates::getRot<std::complex, TestType>(
            angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rot_mat[2];
    }

    SECTION("Apply directly") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_direct{num_qubits};
            sv_direct.initSV();

            sv_direct.applyRot({index}, false, angles[index][0],
                               angles[index][1], angles[index][2]);
            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.initSV();

            sv_dispatch.applyOperation("Rot", {index}, false, angles[index]);
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyCRot", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8, 2.4};

    std::vector<cp_t> expected_results(8);
    const auto rot_mat =
        Gates::getRot<std::complex, TestType>(angles[0], angles[1], angles[2]);
    expected_results[0b1 << (num_qubits - 1)] = rot_mat[0];
    expected_results[(0b1 << num_qubits) - 2] = rot_mat[2];

    const auto init_state = sv.getDataVector();

    SECTION("Apply directly") {
        SECTION("CRot0,1 |000> -> |000>") {
            StateVectorCudaManaged<TestType> sv_direct{num_qubits};
            sv_direct.initSV();

            sv_direct.applyCRot({0, 1}, false, angles[0], angles[1], angles[2]);

            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(init_state));
        }
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            StateVectorCudaManaged<TestType> sv_direct{num_qubits};
            sv_direct.initSV();

            sv_direct.applyOperation("PauliX", {0});
            sv_direct.applyCRot({0, 1}, false, angles[0], angles[1], angles[2]);
            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results));
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            StateVectorCudaManaged<TestType> sv_direct{num_qubits};
            sv_direct.initSV();

            sv_direct.applyOperation("PauliX", {0});
            sv_direct.applyOperation("CRot", {0, 1}, false, angles);

            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyIsingXX", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8};

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

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.0};
    expected_results_adj[0][6] = {0.0, 0.14943813247359922};

    expected_results_adj[1][0] = {0.9210609940028851, 0.0};
    expected_results_adj[1][6] = {0.0, 0.3894183423086505};

    expected_results_adj[2][0] = {0.9887710779360422, 0.0};
    expected_results_adj[2][5] = {0.0, 0.14943813247359922};

    expected_results_adj[3][0] = {0.9210609940028851, 0.0};
    expected_results_adj[3][5] = {0.0, 0.3894183423086505};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("IsingXX 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyIsingXX({0, 1}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("IsingXX 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyIsingXX({0, 2}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(
                          expected_results[index + angles.size()]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("IsingXX 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyIsingXX({0, 1}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("IsingXX 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingXX({0, 2}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(
                          expected_results_adj[index + angles.size()]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                         init_state.size()};

            sv_dispatch.applyOperation("IsingXX", {0, 1}, true,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyIsingYY", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, 0.0};
    expected_results[0][6] = {0.0, 0.14943813247359922};

    expected_results[1][0] = {0.9210609940028851, 0.0};
    expected_results[1][6] = {0.0, 0.3894183423086505};

    expected_results[2][0] = {0.9887710779360422, 0.0};
    expected_results[2][5] = {0.0, 0.14943813247359922};

    expected_results[3][0] = {0.9210609940028851, 0.0};
    expected_results[3][5] = {0.0, 0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits),
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.0};
    expected_results_adj[0][6] = {0.0, -0.14943813247359922};

    expected_results_adj[1][0] = {0.9210609940028851, 0.0};
    expected_results_adj[1][6] = {0.0, -0.3894183423086505};

    expected_results_adj[2][0] = {0.9887710779360422, 0.0};
    expected_results_adj[2][5] = {0.0, -0.14943813247359922};

    expected_results_adj[3][0] = {0.9210609940028851, 0.0};
    expected_results_adj[3][5] = {0.0, -0.3894183423086505};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("IsingYY 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingYY({0, 1}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("IsingYY 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingYY({0, 2}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(
                          expected_results[index + angles.size()]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("IsingYY 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingYY({0, 1}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("IsingYY 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingYY({0, 2}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(
                          expected_results_adj[index + angles.size()]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.initSV();

            sv_dispatch.applyOperation("IsingYY", {0, 1}, true,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyIsingZZ", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits, {0, 0}),
        std::vector<cp_t>(1 << num_qubits, {0, 0})};
    expected_results[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, 0.3894183423086505};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("IsingZZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingZZ({0, 1}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("IsingZZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingZZ({0, 2}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("IsingZZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingZZ({0, 1}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("IsingZZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingZZ({0, 2}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.initSV();

            sv_dispatch.applyOperation("IsingZZ", {0, 1}, true,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applySingleExcitation",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<cp_t> expected_results(1 << num_qubits);
    expected_results[0] = {1.0, 0.0};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly") {
        SECTION("SingleExcitation 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitation({0, 1}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results));
            }
        }
        SECTION("SingleExcitation 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitation({0, 2}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.initSV();
            sv_dispatch.applyOperation("SingleExcitation", {0, 1}, false,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applySingleExcitationMinus",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, 0.3894183423086505};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("SingleExcitationMinus 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationMinus({0, 1}, false,
                                                     angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("SingleExcitationMinus 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationMinus({0, 2}, false,
                                                     angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("SingleExcitationMinus 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationMinus({0, 1}, true,
                                                     angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("SingleExcitationMinus 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationMinus({0, 2}, true,
                                                     angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.initSV();

            sv_dispatch.applyOperation("SingleExcitationMinus", {0, 1}, true,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applySingleExcitationPlus",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, 0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, -0.3894183423086505};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("SingleExcitationPlus 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationPlus({0, 1}, false,
                                                    angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("SingleExcitationPlus 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationPlus({0, 2}, false,
                                                    angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("SingleExcitationPlus 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationPlus({0, 1}, true,
                                                    angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("SingleExcitationPlus 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationPlus({0, 2}, true,
                                                    angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.initSV();

            sv_dispatch.applyOperation("SingleExcitationPlus", {0, 1}, true,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyDoubleExcitation",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 4;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8, 2.4};

    std::vector<cp_t> expected_results(1 << num_qubits);
    expected_results[0] = {1.0, 0.0};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly") {
        SECTION("DoubleExcitation 0,1,2,3") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyDoubleExcitation({0, 1, 2, 3}, false,
                                                angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.initSV();

            sv_dispatch.applyOperation("DoubleExcitation", {0, 1, 2, 3}, false,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyDoubleExcitationMinus",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 4;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, 0.3894183423086505};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("DoubleExcitationMinus 0,1,2,3") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyDoubleExcitationMinus({0, 1, 2, 3}, false,
                                                     angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("DoubleExcitationMinus 0,1,2,3") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyDoubleExcitationMinus({0, 1, 2, 3}, true,
                                                     angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.initSV();

            sv_dispatch.applyOperation("DoubleExcitationMinus", {0, 1, 2, 3},
                                       true, {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyDoubleExcitationPlus",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 4;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, 0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, -0.3894183423086505};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("DoubleExcitationPlus 0,1,2,3") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyDoubleExcitationPlus({0, 1, 2, 3}, false,
                                                    angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("DoubleExcitationPlus 0,1,2,3") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyDoubleExcitationPlus({0, 1, 2, 3}, true,
                                                    angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.initSV();
            sv_dispatch.applyOperation("DoubleExcitationPlus", {0, 1, 2, 3},
                                       true, {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyMultiRZ", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};
    sv.initSV();

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results[0][0] = {0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = {0.9210609940028851, -0.3894183423086505};

    std::vector<std::vector<cp_t>> expected_results_adj{
        std::vector<cp_t>(1 << num_qubits), std::vector<cp_t>(1 << num_qubits)};
    expected_results_adj[0][0] = {0.9887710779360422, 0.14943813247359922};
    expected_results_adj[1][0] = {0.9210609940028851, 0.3894183423086505};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly adjoint=false") {
        SECTION("MultiRZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyMultiRZ({0, 1}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("MultiRZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyMultiRZ({0, 2}, false, angles[index]);

                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
    SECTION("Apply directly adjoint=true") {
        SECTION("MultiRZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyMultiRZ({0, 1}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("MultiRZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyMultiRZ({0, 2}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.initSV();

            sv_dispatch.applyOperation("MultiRZ", {0, 1}, true,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results_adj[index]));
        }
    }
}

// NOLINTNEXTLINE: Avoid complexity errors
TEMPLATE_TEST_CASE("LightningGPU::applyOperation 1 wire",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 5;

    // Note: gates are defined as right-to-left order

    SECTION("Apply XZ gate") {
        const std::vector<cp_t> xz_gate{
            cuUtil::ZERO<cp_t>(), cuUtil::ONE<cp_t>(), -cuUtil::ONE<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            StateVectorCudaManaged<TestType> sv{num_qubits};
            sv.initSV();

            StateVectorCudaManaged<TestType> sv_expected{num_qubits};
            sv_expected.initSV();

            for (size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperation({{"PauliX"}, {"PauliZ"}},
                                           {{index}, {index}}, {false, false});

                sv.applyOperation_std("XZ", {index}, false, {0.0}, xz_gate);
            }

            CHECK(sv.getDataVector() == sv_expected.getDataVector());
        }
    }
    SECTION("Apply ZX gate") {
        const std::vector<cp_t> zx_gate{
            cuUtil::ZERO<cp_t>(), -cuUtil::ONE<cp_t>(), cuUtil::ONE<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            StateVectorCudaManaged<TestType> sv{num_qubits};
            sv.initSV();
            StateVectorCudaManaged<TestType> sv_expected{num_qubits};
            sv_expected.initSV();

            for (size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperation({{"PauliZ"}, {"PauliX"}},
                                           {{index}, {index}}, {false, false});
                sv.applyOperation_std("ZX", {index}, false, {0.0}, zx_gate);
            }
            CHECK(sv.getDataVector() == sv_expected.getDataVector());
        }
    }
    SECTION("Apply XY gate") {
        const std::vector<cp_t> xy_gate{
            -cuUtil::IMAG<cp_t>(), cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(),
            cuUtil::IMAG<cp_t>()};

        SECTION("Apply using dispatcher") {
            StateVectorCudaManaged<TestType> sv{num_qubits};
            sv.initSV();
            StateVectorCudaManaged<TestType> sv_expected{num_qubits};
            sv_expected.initSV();

            for (size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperation({{"PauliX"}, {"PauliY"}},
                                           {{index}, {index}}, {false, false});
                sv.applyOperation_std("XY", {index}, false, {0.0}, xy_gate);
            }
            CHECK(sv.getDataVector() == sv_expected.getDataVector());
        }
    }
    SECTION("Apply YX gate") {
        const std::vector<cp_t> yx_gate{
            cuUtil::IMAG<cp_t>(), cuUtil::ZERO<cp_t>(), cuUtil::ZERO<cp_t>(),
            -cuUtil::IMAG<cp_t>()};

        SECTION("Apply using dispatcher") {
            StateVectorCudaManaged<TestType> sv{num_qubits};
            sv.initSV();
            StateVectorCudaManaged<TestType> sv_expected{num_qubits};
            sv_expected.initSV();

            for (size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperation({{"PauliY"}, {"PauliX"}},
                                           {{index}, {index}}, {false, false});
                sv.applyOperation_std("YX", {index}, false, {0.0}, yx_gate);
            }
            CHECK(sv.getDataVector() == sv_expected.getDataVector());
        }
    }
    SECTION("Apply YZ gate") {
        const std::vector<cp_t> yz_gate{
            cuUtil::ZERO<cp_t>(), -cuUtil::IMAG<cp_t>(), -cuUtil::IMAG<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            StateVectorCudaManaged<TestType> sv{num_qubits};
            sv.initSV();
            StateVectorCudaManaged<TestType> sv_expected{num_qubits};
            sv_expected.initSV();

            for (size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperation({{"PauliY"}, {"PauliZ"}},
                                           {{index}, {index}}, {false, false});
                sv.applyOperation_std("YZ", {index}, false, {0.0}, yz_gate);
            }
            CHECK(sv.getDataVector() == sv_expected.getDataVector());
        }
    }
    SECTION("Apply ZY gate") {
        const std::vector<cp_t> zy_gate{
            cuUtil::ZERO<cp_t>(), cuUtil::IMAG<cp_t>(), cuUtil::IMAG<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            StateVectorCudaManaged<TestType> sv{num_qubits};
            sv.initSV();
            StateVectorCudaManaged<TestType> sv_expected{num_qubits};
            sv_expected.initSV();

            for (size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperation({{"PauliZ"}, {"PauliY"}},
                                           {{index}, {index}}, {false, false});
                sv.applyOperation_std("ZY", {index}, false, {0.0}, zy_gate);
            }
            CHECK(sv.getDataVector() == sv_expected.getDataVector());
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyOperation multiple wires",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const size_t num_qubits = 3;

    StateVectorCudaManaged<TestType> sv_init{num_qubits};
    sv_init.initSV();

    sv_init.applyOperation({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                           {{0}, {1}, {2}}, {false, false, false});

    const auto cz_gate = cuGates::getCZ<cp_t>();
    const auto tof_gate = cuGates::getToffoli<cp_t>();
    const auto arb_gate = cuGates::getToffoli<cp_t>();

    SECTION("Apply CZ gate") {
        StateVectorCudaManaged<TestType> sv{sv_init.getDataVector().data(),
                                            sv_init.getDataVector().size()};
        StateVectorCudaManaged<TestType> sv_expected{
            sv_init.getDataVector().data(), sv_init.getDataVector().size()};

        sv_expected.applyOperation({{"Hadamard"}, {"CNOT"}, {"Hadamard"}},
                                   {{1}, {0, 1}, {1}}, {false, false, false});

        sv.applyOperation_std("CZmat", {0, 1}, false, {0.0}, cz_gate);
        CHECK(sv.getDataVector() ==
              Pennylane::Util::approx(sv_expected.getDataVector()));
    }
}

TEMPLATE_TEST_CASE("Sample", "[LightningGPU_Param]", float, double) {
    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

    // Defining the State Vector that will be measured.
    size_t num_qubits = 3;
    size_t data_size = std::pow(2, num_qubits);

    std::vector<std::complex<TestType>> init_state(data_size, 0);
    init_state[0] = 1;
    StateVectorCudaManaged<TestType> sv{init_state.data(), init_state.size()};
    TestType alpha = 0.7;
    TestType beta = 0.5;
    TestType gamma = 0.2;
    sv.applyOperations({"RX", "RY", "RX", "RY", "RX", "RY"},
                       {{0}, {0}, {1}, {1}, {2}, {2}},
                       {false, false, false, false, false, false},
                       {{alpha}, {alpha}, {beta}, {beta}, {gamma}, {gamma}});

    std::vector<TestType> expected_probabilities = {
        0.687573, 0.013842, 0.089279, 0.001797,
        0.180036, 0.003624, 0.023377, 0.000471};

    size_t N = std::pow(2, num_qubits);
    size_t num_samples = 100000;
    auto &&samples = sv.generate_samples(num_samples);

    std::vector<size_t> counts(N, 0);
    std::vector<size_t> samples_decimal(num_samples, 0);

    // convert samples to decimal and then bin them in counts
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < num_qubits; j++) {
            if (samples[i * num_qubits + j] != 0) {
                samples_decimal[i] += twos[(num_qubits - 1 - j)];
            }
        }
        counts[samples_decimal[i]] += 1;
    }

    // compute estimated probabilities from histogram
    std::vector<TestType> probabilities(counts.size());
    for (size_t i = 0; i < counts.size(); i++) {
        probabilities[i] = counts[i] / (TestType)num_samples;
    }

    REQUIRE_THAT(probabilities,
                 Catch::Approx(expected_probabilities).margin(.05));
}
