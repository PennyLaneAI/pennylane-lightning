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
#include <random>
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

TEMPLATE_TEST_CASE("LightningGPU:applyOperation", "[LightningGPU_Param]",
                   double) {
    const std::size_t num_qubits = 1;
    StateVectorCudaManaged<TestType> sv{num_qubits};

    SECTION("Catch failures caused by unsupported named gates") {
        std::string obs = "paulix";
        PL_CHECK_THROWS_MATCHES(sv.applyOperation(obs, {0}), LightningException,
                                "Currently unsupported gate: paulix");
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyRX", "[LightningGPU_Param]", double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 1;
    StateVectorCudaManaged<TestType> sv{num_qubits};

    const std::vector<TestType> angles{{0.1}, {0.6}};

    const auto init_state = sv.getDataVector();
    SECTION("adj = false") {
        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>{{0.9987502603949663, 0.0},
                              {0.0, -0.04997916927067834}},
            std::vector<cp_t>{{0.9553364891256061, 0.0},
                              {0, -0.2955202066613395}},
            std::vector<cp_t>{{0.49757104789172696, 0.0},
                              {0, -0.867423225594017}}};

        SECTION("Apply directly") {
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyRX({0}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                             init_state.size()};
                sv_dispatch.applyOperation("RX", {0}, false, {angles[index]});
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
    SECTION("adj = true") {
        std::vector<std::vector<cp_t>> expected_results_adj{
            std::vector<cp_t>{{0.9987502603949663, 0.0},
                              {0.0, 0.04997916927067834}},
            std::vector<cp_t>{{0.9553364891256061, 0.0},
                              {0, 0.2955202066613395}},
            std::vector<cp_t>{{0.49757104789172696, 0.0},
                              {0, 0.867423225594017}}};

        SECTION("Apply directly") {
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyRX({0}, true, {angles[index]});
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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

    const std::vector<cp_t> init_state{{0.8775825618903728, 0.0},
                                       {0.0, -0.47942553860420306}};
    SECTION("adj = false") {
        std::vector<std::vector<cp_t>> expected_results{
            std::vector<cp_t>{{0.8731983044562817, 0.04786268954660339},
                              {0.0876120655431924, -0.47703040785184303}},
            std::vector<cp_t>{{0.8243771119105122, 0.16439396602553008},
                              {0.3009211363333468, -0.45035926880694604}},
            std::vector<cp_t>{{0.10575112905629831, 0.47593196040758534},
                              {0.8711876098966215, -0.0577721051072477}}};
        SECTION("Apply directly") {
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyRY({0}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                             init_state.size()};
                sv_dispatch.applyOperation("RY", {0}, false, {angles[index]});
                CHECK(sv_dispatch.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
    }
    SECTION("adj = true") {
        std::vector<std::vector<cp_t>> expected_results_adj{
            std::vector<cp_t>{{0.8731983044562817, -0.04786268954660339},
                              {-0.0876120655431924, -0.47703040785184303}},
            std::vector<cp_t>{{0.8243771119105122, -0.16439396602553008},
                              {-0.3009211363333468, -0.45035926880694604}},
            std::vector<cp_t>{{0.10575112905629831, -0.47593196040758534},
                              {-0.8711876098966215, -0.0577721051072477}}};
        SECTION("Apply directly") {
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyRY({0}, true, {angles[index]});
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

    // Test using |+++> state
    sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
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
        for (std::size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                       init_state.size()};

            sv_direct.applyRZ({index}, false, {angles[index]});
            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < num_qubits; index++) {
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
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

    // Test using |+++> state
    sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
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
        for (std::size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                       init_state.size()};

            sv_direct.applyPhaseShift({index}, false, {angles[index]});
            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < num_qubits; index++) {
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
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

    // Test using |+++> state
    sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
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
    const bool adjoint = GENERATE(true, false);

    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;

    const std::vector<std::vector<TestType>> angles{
        std::vector<TestType>{0.3, 0.8, 2.4},
        std::vector<TestType>{0.5, 1.1, 3.0},
        std::vector<TestType>{2.3, 0.1, 0.4}};

    std::vector<std::vector<cp_t>> expected_results{
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits),
        std::vector<cp_t>(0b1 << num_qubits)};

    for (std::size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat =
            (adjoint) ? Gates::getRot<std::complex, TestType>(
                            -angles[i][0], -angles[i][1], -angles[i][2])
                      : Gates::getRot<std::complex, TestType>(
                            angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rot_mat[2];
    }

    SECTION("Apply directly") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_direct{num_qubits};

            sv_direct.applyRot({index}, adjoint, angles[index][0],
                               angles[index][1], angles[index][2]);
            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
        for (std::size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_direct{num_qubits};

            sv_direct.applyRot({index}, adjoint, angles[index]);
            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < num_qubits; index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};

            sv_dispatch.applyOperation("Rot", {index}, adjoint, angles[index]);
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyCRot", "[LightningGPU_Param]", float,
                   double) {
    const bool adjoint = GENERATE(true, false);

    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8, 2.4};

    std::vector<cp_t> expected_results(8);
    const auto rot_mat = (adjoint) ? Gates::getRot<std::complex, TestType>(
                                         -angles[0], -angles[1], -angles[2])
                                   : Gates::getRot<std::complex, TestType>(
                                         angles[0], angles[1], angles[2]);

    expected_results[0b1 << (num_qubits - 1)] = rot_mat[0];
    expected_results[(0b1 << num_qubits) - 2] = rot_mat[2];

    const auto init_state = sv.getDataVector();

    SECTION("Apply directly") {
        SECTION("CRot0,1 |000> -> |000>") {
            {
                StateVectorCudaManaged<TestType> sv_direct{num_qubits};

                sv_direct.applyCRot({0, 1}, adjoint, angles[0], angles[1],
                                    angles[2]);

                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(init_state));
            }
            {
                StateVectorCudaManaged<TestType> sv_direct{num_qubits};

                sv_direct.applyCRot({0, 1}, adjoint, angles);

                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(init_state));
            }
        }
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            StateVectorCudaManaged<TestType> sv_direct{num_qubits};

            sv_direct.applyOperation("PauliX", {0});
            sv_direct.applyCRot({0, 1}, adjoint, angles[0], angles[1],
                                angles[2]);
            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results));
        }
    }
    SECTION("Apply using dispatcher") {
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            StateVectorCudaManaged<TestType> sv_direct{num_qubits};

            sv_direct.applyOperation("PauliX", {0});
            sv_direct.applyOperation("CRot", {0, 1}, adjoint, angles);

            CHECK(sv_direct.getDataVector() ==
                  Pennylane::Util::approx(expected_results));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyIsingXX", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyIsingXX({0, 1}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("IsingXX 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyIsingXX({0, 1}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("IsingXX 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
        for (std::size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{init_state.data(),
                                                         init_state.size()};

            sv_dispatch.applyOperation("IsingXX", {0, 1}, true,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyIsingXY", "[LightningGPU_Param]", float,
                   double) {
    using ComplexT = StateVectorCudaManaged<TestType>::ComplexT;

    std::vector<ComplexT> ini_st{
        ComplexT{0.267462841882, 0.010768564798},
        ComplexT{0.228575129706, 0.010564590956},
        ComplexT{0.099492749900, 0.260849823392},
        ComplexT{0.093690204310, 0.189847108173},
        ComplexT{0.033390732374, 0.203836830144},
        ComplexT{0.226979395737, 0.081852150975},
        ComplexT{0.031235505729, 0.176933497281},
        ComplexT{0.294287602843, 0.145156781198},
        ComplexT{0.152742706049, 0.111628061129},
        ComplexT{0.012553863703, 0.120027860480},
        ComplexT{0.237156555364, 0.154658769755},
        ComplexT{0.117001120872, 0.228059505033},
        ComplexT{0.041495873225, 0.065934827444},
        ComplexT{0.089653239407, 0.221581340372},
        ComplexT{0.217892322429, 0.291261296999},
        ComplexT{0.292993251871, 0.186570798697},
    };

    std::vector<ComplexT> expected{
        ComplexT{0.267462849617, 0.010768564418},
        ComplexT{0.228575125337, 0.010564590804},
        ComplexT{0.099492751062, 0.260849833488},
        ComplexT{0.093690201640, 0.189847111702},
        ComplexT{0.015641822883, 0.225092900621},
        ComplexT{0.205574608177, 0.082808663337},
        ComplexT{0.006827173322, 0.211631480575},
        ComplexT{0.255280800811, 0.161572331669},
        ComplexT{0.119218164572, 0.115460377284},
        ComplexT{-0.000315789761, 0.153835664378},
        ComplexT{0.206786872079, 0.157633689097},
        ComplexT{0.093027614553, 0.271012980118},
        ComplexT{0.041495874524, 0.065934829414},
        ComplexT{0.089653238654, 0.221581339836},
        ComplexT{0.217892318964, 0.291261285543},
        ComplexT{0.292993247509, 0.186570793390},
    };

    SECTION("Apply directly") {
        StateVectorCudaManaged<TestType> sv(ini_st.data(), ini_st.size());
        sv.applyIsingXY({0, 1}, false, {0.312});
        CHECK(sv.getDataVector() == Pennylane::Util::approx(expected));
    }

    SECTION("Apply using dispatcher") {
        StateVectorCudaManaged<TestType> sv(ini_st.data(), ini_st.size());
        sv.applyOperation("IsingXY", {0, 1}, false, {0.312});
        CHECK(sv.getDataVector() == Pennylane::Util::approx(expected));
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyIsingYY", "[LightningGPU_Param]", float,
                   double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingYY({0, 1}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("IsingYY 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingYY({0, 1}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("IsingYY 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
        for (std::size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
            sv_dispatch.resetStateVector();
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
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingZZ({0, 1}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("IsingZZ 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingZZ({0, 1}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("IsingZZ 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};
                sv_direct.applyIsingZZ({0, 2}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};

            sv_dispatch.applyOperation("IsingZZ", {0, 1}, true,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results_adj[index]));
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyCRX", "[LightningGPU_Param]", float,
                   double) {
    using ComplexT = StateVectorCudaManaged<TestType>::ComplexT;

    std::vector<ComplexT> ini_st{
        ComplexT{0.188018120185, 0.267344585187},
        ComplexT{0.172684792903, 0.187465336044},
        ComplexT{0.218892658302, 0.241508557821},
        ComplexT{0.107094509452, 0.233123916768},
        ComplexT{0.144398681914, 0.102112687699},
        ComplexT{0.266641428689, 0.096286886834},
        ComplexT{0.037126289559, 0.047222166486},
        ComplexT{0.136865047634, 0.203178369592},
        ComplexT{0.001562711889, 0.224933454573},
        ComplexT{0.009933412610, 0.080866505038},
        ComplexT{0.000948295069, 0.280652963863},
        ComplexT{0.109817299553, 0.150776413412},
        ComplexT{0.297480913626, 0.232588348025},
        ComplexT{0.247386444054, 0.077608200535},
        ComplexT{0.192650977126, 0.054764192471},
        ComplexT{0.033093927690, 0.243038790593},
    };

    std::vector<ComplexT> expected{
        ComplexT{0.188018120185, 0.267344585187},
        ComplexT{0.172684792903, 0.187465336044},
        ComplexT{0.218892658302, 0.241508557821},
        ComplexT{0.107094509452, 0.233123916768},
        ComplexT{0.144398681914, 0.102112687699},
        ComplexT{0.266641428689, 0.096286886834},
        ComplexT{0.037126289559, 0.047222166486},
        ComplexT{0.136865047634, 0.203178369592},
        ComplexT{0.037680529583, 0.175982985869},
        ComplexT{0.021870621269, 0.041448569986},
        ComplexT{0.009445384485, 0.247313095111},
        ComplexT{0.146244209335, 0.143803745197},
        ComplexT{0.328815969263, 0.229521152393},
        ComplexT{0.256946415396, 0.075122442730},
        ComplexT{0.233916049255, 0.053951837341},
        ComplexT{0.056117891609, 0.223025389250},
    };

    SECTION("Apply directly") {
        StateVectorCudaManaged<TestType> sv(ini_st.data(), ini_st.size());

        sv.applyCRX({0, 1}, false, {0.312});

        CHECK(sv.getDataVector() == Pennylane::Util::approx(expected));
    }

    SECTION("Apply using dispatcher") {
        StateVectorCudaManaged<TestType> sv(ini_st.data(), ini_st.size());

        sv.applyOperation("CRX", {0, 1}, false, {0.312});

        CHECK(sv.getDataVector() == Pennylane::Util::approx(expected));
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyCRY", "[LightningGPU_Param]", float,
                   double) {
    using ComplexT = StateVectorCudaManaged<TestType>::ComplexT;

    std::vector<ComplexT> ini_st{
        ComplexT{0.024509081663, 0.005606762650},
        ComplexT{0.261792037054, 0.259257414596},
        ComplexT{0.168380715455, 0.096012484887},
        ComplexT{0.169761107379, 0.042890935442},
        ComplexT{0.012169527484, 0.082631086139},
        ComplexT{0.155790166500, 0.292998574950},
        ComplexT{0.150529463310, 0.282021216715},
        ComplexT{0.097100202708, 0.134938013786},
        ComplexT{0.062640753523, 0.251735121160},
        ComplexT{0.121654204141, 0.116964600258},
        ComplexT{0.152865184550, 0.084800955456},
        ComplexT{0.300145205424, 0.101098965771},
        ComplexT{0.288274703880, 0.038180155037},
        ComplexT{0.041378441702, 0.206525491532},
        ComplexT{0.033201995261, 0.096777018650},
        ComplexT{0.303210250465, 0.300817738868},
    };

    std::vector<ComplexT> expected{
        ComplexT{0.024509081663, 0.005606762650},
        ComplexT{0.261792037054, 0.259257414596},
        ComplexT{0.168380715455, 0.096012484887},
        ComplexT{0.169761107379, 0.042890935442},
        ComplexT{0.012169527484, 0.082631086139},
        ComplexT{0.155790166500, 0.292998574950},
        ComplexT{0.150529463310, 0.282021216715},
        ComplexT{0.097100202708, 0.134938013786},
        ComplexT{0.017091411508, 0.242746239557},
        ComplexT{0.113748028260, 0.083456799483},
        ComplexT{0.145850361424, 0.068735133269},
        ComplexT{0.249391258812, 0.053133825802},
        ComplexT{0.294506455875, 0.076828111036},
        ComplexT{0.059777143539, 0.222190141515},
        ComplexT{0.056549175144, 0.108777179774},
        ComplexT{0.346161234622, 0.312872353290},
    };

    SECTION("Apply directly") {
        StateVectorCudaManaged<TestType> sv(ini_st.data(), ini_st.size());

        sv.applyCRY({0, 1}, false, {0.312});

        CHECK(sv.getDataVector() == Pennylane::Util::approx(expected));
    }

    SECTION("Apply using dispatcher") {
        StateVectorCudaManaged<TestType> sv(ini_st.data(), ini_st.size());

        sv.applyOperation("CRY", {0, 1}, false, {0.312});

        CHECK(sv.getDataVector() == Pennylane::Util::approx(expected));
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyCRZ", "[LightningGPU_Param]", float,
                   double) {
    using ComplexT = StateVectorCudaManaged<TestType>::ComplexT;

    std::vector<ComplexT> ini_st{
        ComplexT{0.264968228755, 0.059389110312},
        ComplexT{0.004927738580, 0.117198819444},
        ComplexT{0.192517901751, 0.061524928233},
        ComplexT{0.285160768924, 0.013212111581},
        ComplexT{0.278645646186, 0.212116779981},
        ComplexT{0.171786665640, 0.141260537212},
        ComplexT{0.199480649113, 0.218261452113},
        ComplexT{0.071007710848, 0.294720535623},
        ComplexT{0.169589173252, 0.010528306669},
        ComplexT{0.061973371011, 0.033143783035},
        ComplexT{0.177570977662, 0.116785656786},
        ComplexT{0.070266502325, 0.084338553411},
        ComplexT{0.053744021753, 0.146932844792},
        ComplexT{0.254428637803, 0.138916780809},
        ComplexT{0.260354050166, 0.267004004472},
        ComplexT{0.008910554792, 0.316282675508},
    };

    std::vector<ComplexT> expected{
        ComplexT{0.264968228755, 0.059389110312},
        ComplexT{0.004927738580, 0.117198819444},
        ComplexT{0.192517901751, 0.061524928233},
        ComplexT{0.285160768924, 0.013212111581},
        ComplexT{0.278645646186, 0.212116779981},
        ComplexT{0.171786665640, 0.141260537212},
        ComplexT{0.199480649113, 0.218261452113},
        ComplexT{0.071007710848, 0.294720535623},
        ComplexT{0.169165556003, -0.015948278519},
        ComplexT{0.066370291483, 0.023112625918},
        ComplexT{0.193559430151, 0.087778634862},
        ComplexT{0.082516747253, 0.072397233118},
        ComplexT{0.030262722499, 0.153498691785},
        ComplexT{0.229755796458, 0.176759943762},
        ComplexT{0.215708594452, 0.304212379961},
        ComplexT{-0.040337866447, 0.313826361773},
    };

    SECTION("Apply directly") {
        StateVectorCudaManaged<TestType> sv(ini_st.data(), ini_st.size());

        sv.applyCRZ({0, 1}, false, {0.312});

        CHECK(sv.getDataVector() == Pennylane::Util::approx(expected));
    }

    SECTION("Apply using dispatcher") {
        StateVectorCudaManaged<TestType> sv(ini_st.data(), ini_st.size());

        sv.applyOperation("CRZ", {0, 1}, false, {0.312});

        CHECK(sv.getDataVector() == Pennylane::Util::approx(expected));
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applySingleExcitation",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = std::complex<TestType>;
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<cp_t> expected_results(1 << num_qubits);
    expected_results[0] = {1.0, 0.0};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly") {
        SECTION("SingleExcitation 0,1") {
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitation({0, 1}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results));
            }
        }
        SECTION("SingleExcitation 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitation({0, 2}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
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
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationMinus({0, 1}, false,
                                                     angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("SingleExcitationMinus 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationMinus({0, 1}, true,
                                                     angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("SingleExcitationMinus 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
        for (std::size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};

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
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationPlus({0, 1}, false,
                                                    angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("SingleExcitationPlus 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applySingleExcitationPlus({0, 1}, true,
                                                    angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("SingleExcitationPlus 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
        for (std::size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};

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
    const std::size_t num_qubits = 4;
    StateVectorCudaManaged<TestType> sv{num_qubits};

    const std::vector<TestType> angles{0.3, 0.8, 2.4};

    std::vector<cp_t> expected_results(1 << num_qubits);
    expected_results[0] = {1.0, 0.0};

    const auto init_state = sv.getDataVector();
    SECTION("Apply directly") {
        SECTION("DoubleExcitation 0,1,2,3") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
        for (std::size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};

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
    const std::size_t num_qubits = 4;
    StateVectorCudaManaged<TestType> sv{num_qubits};

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
            for (std::size_t index = 0; index < angles.size(); index++) {
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
            for (std::size_t index = 0; index < angles.size(); index++) {
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
        for (std::size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};

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
    const std::size_t num_qubits = 4;
    StateVectorCudaManaged<TestType> sv{num_qubits};

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
            for (std::size_t index = 0; index < angles.size(); index++) {
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
            for (std::size_t index = 0; index < angles.size(); index++) {
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
        for (std::size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};
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
    const std::size_t num_qubits = 3;
    StateVectorCudaManaged<TestType> sv{num_qubits};

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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyMultiRZ({0, 1}, false, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results[index]));
            }
        }
        SECTION("MultiRZ 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
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
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyMultiRZ({0, 1}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
        SECTION("MultiRZ 0,2") {
            for (std::size_t index = 0; index < angles.size(); index++) {
                StateVectorCudaManaged<TestType> sv_direct{init_state.data(),
                                                           init_state.size()};

                sv_direct.applyMultiRZ({0, 2}, true, angles[index]);
                CHECK(sv_direct.getDataVector() ==
                      Pennylane::Util::approx(expected_results_adj[index]));
            }
        }
    }
    SECTION("Apply using dispatcher") {
        for (std::size_t index = 0; index < angles.size(); index++) {
            StateVectorCudaManaged<TestType> sv_dispatch{num_qubits};

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
    using cp_t = StateVectorCudaManaged<TestType>::CFP_t;
    const std::size_t num_qubits = 5;

    // Note: gates are defined as right-to-left order

    SECTION("Apply XZ gate") {
        const std::vector<cp_t> xz_gate{
            cuUtil::ZERO<cp_t>(), cuUtil::ONE<cp_t>(), -cuUtil::ONE<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            StateVectorCudaManaged<TestType> sv{num_qubits};

            StateVectorCudaManaged<TestType> sv_expected{num_qubits};

            for (std::size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperations({{"PauliX"}, {"PauliZ"}},
                                            {{index}, {index}}, {false, false});

                sv.applyOperation("XZ", {index}, false, {0.0}, xz_gate);
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
            StateVectorCudaManaged<TestType> sv_expected{num_qubits};

            for (std::size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperations({{"PauliZ"}, {"PauliX"}},
                                            {{index}, {index}}, {false, false});
                sv.applyOperation("ZX", {index}, false, {0.0}, zx_gate);
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
            StateVectorCudaManaged<TestType> sv_expected{num_qubits};

            for (std::size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperations({{"PauliX"}, {"PauliY"}},
                                            {{index}, {index}}, {false, false});
                sv.applyOperation("XY", {index}, false, {0.0}, xy_gate);
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
            StateVectorCudaManaged<TestType> sv_expected{num_qubits};

            for (std::size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperations({{"PauliY"}, {"PauliX"}},
                                            {{index}, {index}}, {false, false});
                sv.applyOperation("YX", {index}, false, {0.0}, yx_gate);
            }
            CHECK(sv.getDataVector() == sv_expected.getDataVector());
        }
    }
    // LCOV_EXCL_START
    SECTION("Apply YZ gate") {
        const std::vector<cp_t> yz_gate{
            cuUtil::ZERO<cp_t>(), -cuUtil::IMAG<cp_t>(), -cuUtil::IMAG<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            StateVectorCudaManaged<TestType> sv{num_qubits};
            StateVectorCudaManaged<TestType> sv_expected{num_qubits};

            for (std::size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperations({{"PauliY"}, {"PauliZ"}},
                                            {{index}, {index}}, {false, false});
                sv.applyOperation("YZ", {index}, false, {0.0}, yz_gate);
            }
            CHECK(sv.getDataVector() == sv_expected.getDataVector());
        }
    }
    // LCOV_EXCL_STOP
    SECTION("Apply ZY gate") {
        const std::vector<cp_t> zy_gate{
            cuUtil::ZERO<cp_t>(), cuUtil::IMAG<cp_t>(), cuUtil::IMAG<cp_t>(),
            cuUtil::ZERO<cp_t>()};

        SECTION("Apply using dispatcher") {
            StateVectorCudaManaged<TestType> sv{num_qubits};
            StateVectorCudaManaged<TestType> sv_expected{num_qubits};

            for (std::size_t index = 0; index < num_qubits; index++) {
                sv_expected.applyOperations({{"PauliZ"}, {"PauliY"}},
                                            {{index}, {index}}, {false, false});
                sv.applyOperation("ZY", {index}, false, {0.0}, zy_gate);
            }
            CHECK(sv.getDataVector() == sv_expected.getDataVector());
        }
    }
}

TEMPLATE_TEST_CASE("LightningGPU::applyOperation multiple wires",
                   "[LightningGPU_Param]", float, double) {
    using cp_t = StateVectorCudaManaged<TestType>::CFP_t;
    const std::size_t num_qubits = 3;

    StateVectorCudaManaged<TestType> sv_init{num_qubits};

    sv_init.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                            {{0}, {1}, {2}}, {false, false, false});

    const auto cz_gate = cuGates::getCZ<cp_t>();
    const auto tof_gate = cuGates::getToffoli<cp_t>();
    const auto arb_gate = cuGates::getToffoli<cp_t>();

    SECTION("Apply CZ gate") {
        StateVectorCudaManaged<TestType> sv{sv_init.getDataVector().data(),
                                            sv_init.getDataVector().size()};
        StateVectorCudaManaged<TestType> sv_expected{
            sv_init.getDataVector().data(), sv_init.getDataVector().size()};

        sv_expected.applyOperations({{"Hadamard"}, {"CNOT"}, {"Hadamard"}},
                                    {{1}, {0, 1}, {1}}, {false, false, false});

        sv.applyOperation("CZmat", {0, 1}, false, {0.0}, cz_gate);
        CHECK(sv.getDataVector() ==
              Pennylane::Util::approx(sv_expected.getDataVector()));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyGlobalPhase",
                   "[StateVectorCudaManaged_Param]", double) {
    using ComplexT = StateVectorCudaManaged<TestType>::ComplexT;
    std::mt19937_64 re{1337};
    const std::size_t num_qubits = 3;
    const bool inverse = GENERATE(false, true);
    const std::size_t index = GENERATE(0, 1, 2);
    const TestType param = 0.234;
    const ComplexT phase = std::exp(ComplexT{0, (inverse) ? param : -param});

    auto sv_data = createRandomStateVectorData<TestType>(re, num_qubits);
    StateVectorCudaManaged<TestType> sv(
        reinterpret_cast<ComplexT *>(sv_data.data()), sv_data.size());
    sv.applyOperation("GlobalPhase", {index}, inverse, {param});
    auto result_sv = sv.getDataVector();
    for (std::size_t j = 0; j < exp2(num_qubits); j++) {
        ComplexT tmp = phase * ComplexT(sv_data[j]);
        CHECK((real(result_sv[j])) == Approx(real(tmp)));
        CHECK((imag(result_sv[j])) == Approx(imag(tmp)));
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorCudaManaged::applyOperation param one-qubit with controls",
    "[StateVectorCudaManaged]", float, double) {
    using PrecisionT = TestType;
    std::mt19937 re{1337};
    const int num_qubits = 4;
    const auto margin = PrecisionT{1e-5};
    const std::size_t control = GENERATE(0, 1, 2, 3);
    const std::size_t wire = GENERATE(0, 1, 2, 3);
    bool inverse = GENERATE(false, true);
    PrecisionT param = GENERATE(-0.5, 1.5);

    auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);

    StateVectorCudaManaged<PrecisionT> sv0(num_qubits);
    StateVectorCudaManaged<PrecisionT> sv1(num_qubits);

    sv0.CopyHostDataToGpu(st0.data(), st0.size());
    sv1.CopyHostDataToGpu(st0.data(), st0.size());

    DYNAMIC_SECTION("N-controlled PhaseShift - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            sv0.applyOperation("ControlledPhaseShift", {control, wire}, inverse,
                               {param});
            sv1.applyOperation("PhaseShift", std::vector<std::size_t>{control},
                               std::vector<bool>{true},
                               std::vector<std::size_t>{wire}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled RX - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            sv0.applyOperation("CRX", {control, wire}, inverse, {param});
            sv1.applyOperation("RX", std::vector<std::size_t>{control},
                               std::vector<bool>{true},
                               std::vector<std::size_t>{wire}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled RY - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            sv0.applyOperation("CRY", {control, wire}, inverse, {param});
            sv1.applyOperation("RY", std::vector<std::size_t>{control},
                               std::vector<bool>{true},
                               std::vector<std::size_t>{wire}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled RZ - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            sv0.applyOperation("CRZ", {control, wire}, inverse, {param});
            sv1.applyOperation("RZ", std::vector<std::size_t>{control},
                               std::vector<bool>{true},
                               std::vector<std::size_t>{wire}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled Rot - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            const std::vector<PrecisionT> params = {
                param, static_cast<PrecisionT>(2.0) * param,
                static_cast<PrecisionT>(3.0) * param};
            sv0.applyOperation("CRot", {control, wire}, inverse, params);
            sv1.applyOperation("Rot", std::vector<std::size_t>{control},
                               std::vector<bool>{true},
                               std::vector<std::size_t>{wire}, inverse, params);
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorCudaManaged::applyOperation param two-qubits with controls",
    "[StateVectorCudaManaged]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<TestType>;
    std::mt19937 re{1337};
    const int num_qubits = 4;
    const auto margin = PrecisionT{1e-5};
    const std::size_t control = GENERATE(0, 1, 2, 3);
    const std::size_t wire0 = GENERATE(0, 1, 2, 3);
    const std::size_t wire1 = GENERATE(0, 1, 2, 3);
    bool inverse = GENERATE(false, true);
    PrecisionT param = GENERATE(-0.5, 1.5);

    auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
    StateVectorCudaManaged<PrecisionT> sv0(num_qubits);
    StateVectorCudaManaged<PrecisionT> sv1(num_qubits);

    sv0.CopyHostDataToGpu(st0.data(), st0.size());
    sv1.CopyHostDataToGpu(st0.data(), st0.size());

    auto getControlledGate = [](std::vector<ComplexT> matrix) {
        std::vector<ComplexT> cmatrix(matrix.size() * 4);
        for (std::size_t i = 0; i < 4; i++) {
            cmatrix[i * 8 + i] = ComplexT{1.0};
        }
        for (std::size_t i = 0; i < 4; i++) {
            for (std::size_t j = 0; j < 4; j++) {
                cmatrix[(i + 4) * 8 + j + 4] = matrix[i * 4 + j];
            }
        }
        return cmatrix;
    };

    DYNAMIC_SECTION("N-controlled IsingXX - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix = getIsingXX<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation("IsingXX", std::vector<std::size_t>{control},
                               std::vector<bool>{true},
                               std::vector<std::size_t>{wire0, wire1}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled IsingXY - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix = getIsingXY<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation("IsingXY", std::vector<std::size_t>{control},
                               std::vector<bool>{true},
                               std::vector<std::size_t>{wire0, wire1}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled IsingYY - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix = getIsingYY<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation("IsingYY", std::vector<std::size_t>{control},
                               std::vector<bool>{true},
                               std::vector<std::size_t>{wire0, wire1}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled IsingZZ - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix = getIsingZZ<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation("IsingZZ", std::vector<std::size_t>{control},
                               std::vector<bool>{true},
                               std::vector<std::size_t>{wire0, wire1}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled SingleExcitation - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix = getSingleExcitation<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation(
                "SingleExcitation", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire0, wire1},
                inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled SingleExcitationMinus - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix =
                getSingleExcitationMinus<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation(
                "SingleExcitationMinus", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire0, wire1},
                inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled SingleExcitationPlus - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix =
                getSingleExcitationPlus<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation(
                "SingleExcitationPlus", std::vector<std::size_t>{control},
                std::vector<bool>{true}, std::vector<std::size_t>{wire0, wire1},
                inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorCudaManaged::applyOperation param four-qubits with controls",
    "[StateVectorCudaManaged]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<TestType>;
    std::mt19937 re{1337};
    const int num_qubits = 5;
    const auto margin = PrecisionT{1e-5};
    const std::size_t control = GENERATE(0, 1, 2, 3, 4);
    const std::size_t wire0 = GENERATE(0, 1, 2, 3, 4);
    const std::size_t wire1 = GENERATE(0, 1, 2, 3, 4);
    const std::size_t wire2 = GENERATE(0, 1, 2, 3, 4);
    const std::size_t wire3 = GENERATE(0, 1, 2, 3, 4);
    bool inverse = GENERATE(false, true);
    PrecisionT param = GENERATE(-0.5, 1.5);

    StateVectorCudaManaged<PrecisionT> sv0(num_qubits);
    StateVectorCudaManaged<PrecisionT> sv1(num_qubits);

    auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
    sv0.CopyHostDataToGpu(st0.data(), st0.size());
    sv1.CopyHostDataToGpu(st0.data(), st0.size());

    auto getControlledGate = [](std::vector<ComplexT> matrix) {
        std::vector<ComplexT> cmatrix(matrix.size() * 4);
        for (std::size_t i = 0; i < 16; i++) {
            cmatrix[i * 32 + i] = ComplexT{1.0};
        }
        for (std::size_t i = 0; i < 16; i++) {
            for (std::size_t j = 0; j < 16; j++) {
                cmatrix[(i + 16) * 32 + j + 16] = matrix[i * 16 + j];
            }
        }
        return cmatrix;
    };

    DYNAMIC_SECTION("N-controlled DoubleExcitation - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << ", " << wire2
                    << ", " << wire3 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<std::size_t> wires = {control, wire0, wire1, wire2, wire3};
        std::sort(wires.begin(), wires.end());
        if (std::adjacent_find(wires.begin(), wires.end()) == wires.end()) {
            auto matrix = getDoubleExcitation<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1, wire2, wire3},
                            inverse);
            sv1.applyOperation(
                "DoubleExcitation", std::vector<std::size_t>{control},
                std::vector<bool>{true},
                std::vector<std::size_t>{wire0, wire1, wire2, wire3}, inverse,
                {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled DoubleExcitationMinus - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << ", " << wire2
                    << ", " << wire3 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<std::size_t> wires = {control, wire0, wire1, wire2, wire3};
        std::sort(wires.begin(), wires.end());
        if (std::adjacent_find(wires.begin(), wires.end()) == wires.end()) {
            auto matrix =
                getDoubleExcitationMinus<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1, wire2, wire3},
                            inverse);
            sv1.applyOperation(
                "DoubleExcitationMinus", std::vector<std::size_t>{control},
                std::vector<bool>{true},
                std::vector<std::size_t>{wire0, wire1, wire2, wire3}, inverse,
                {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled DoubleExcitationPlus - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << ", " << wire2
                    << ", " << wire3 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<std::size_t> wires = {control, wire0, wire1, wire2, wire3};
        std::sort(wires.begin(), wires.end());
        if (std::adjacent_find(wires.begin(), wires.end()) == wires.end()) {
            auto matrix =
                getDoubleExcitationPlus<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1, wire2, wire3},
                            inverse);
            sv1.applyOperation(
                "DoubleExcitationPlus", std::vector<std::size_t>{control},
                std::vector<bool>{true},
                std::vector<std::size_t>{wire0, wire1, wire2, wire3}, inverse,
                {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled MultiRZ - "
                    << "controls = {" << control << ", " << wire0 << ", "
                    << wire1 << "} "
                    << ", wires = {" << wire2 << ", " << wire3 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<std::size_t> wires = {control, wire0, wire1, wire2, wire3};
        std::sort(wires.begin(), wires.end());
        const ComplexT e = std::exp(ComplexT{0, -0.5} * param);
        std::vector<ComplexT> matrix(16, 0.0);
        matrix[0] = e;
        matrix[5] = std::conj(e);
        matrix[10] = std::conj(e);
        matrix[15] = e;
        if (std::adjacent_find(wires.begin(), wires.end()) == wires.end()) {
            sv0.applyControlledMatrix(matrix.data(), {control, wire0, wire1},
                                      std::vector<bool>{true, false, true},
                                      {wire2, wire3}, inverse);
            sv1.applyOperation(
                "MultiRZ", std::vector<std::size_t>{control, wire0, wire1},
                std::vector<bool>{true, false, true},
                std::vector<std::size_t>{wire2, wire3}, inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyControlledGlobalPhase",
                   "[StateVectorCudaManaged_Param]", double) {
    using ComplexT = StateVectorCudaManaged<TestType>::ComplexT;
    std::mt19937_64 re{1337};
    const std::size_t num_qubits = 3;
    const bool inverse = GENERATE(false, true);
    /* The `phase` array contains the diagonal entries of the controlled-phase
       operator. It can be created in Python using the following command

       ```
       global_phase_diagonal(-np.pi/2, wires=[0, 1, 2], controls=[0, 1],
       control_values=[0, 1])
       ```

       where the phase angle is chosen as `-np.pi/2` for simplicity.
    */
    const std::vector<ComplexT> phase = {{1.0, 0.}, {1.0, 0.}, {0.0, 1.},
                                         {0.0, 1.}, {1.0, 0.}, {1.0, 0.},
                                         {1.0, 0.}, {1.0, 0.}};

    auto sv_data = createRandomStateVectorData<TestType>(re, num_qubits);
    StateVectorCudaManaged<TestType> sv(
        reinterpret_cast<ComplexT *>(sv_data.data()), sv_data.size());
    std::vector<std::size_t> ctrls = {0, 1};
    std::vector<bool> ctrl_vals = {0, 1};
    std::vector<std::size_t> tgts = {2};
    const TestType param = -M_PI_2;
    sv.applyOperation("GlobalPhase", ctrls, ctrl_vals, tgts, inverse, {param});
    auto result_sv = sv.getDataVector();
    for (std::size_t j = 0; j < exp2(num_qubits); j++) {
        ComplexT tmp = (inverse) ? conj(phase[j]) : phase[j];
        tmp *= ComplexT(sv_data[j]);
        CHECK((real(result_sv[j])) == Approx(real(tmp)));
        CHECK((imag(result_sv[j])) == Approx(imag(tmp)));
    }
}
