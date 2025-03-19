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
#include <array>
#include <complex>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "ExactTNCuda.hpp"
#include "MPSTNCuda.hpp"
#include "MeasurementsTNCuda.hpp"
#include "TNCudaGateCache.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpersTNCuda.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda::Measures;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
} // namespace
/// @endcond

TEMPLATE_LIST_TEST_CASE("Test variance of NamedObs", "[TNCuda_Var]",
                        TestTNBackends) {
    using TNDeviceT = TestType;
    using PrecisionT = typename TNDeviceT::PrecisionT;
    using NamedObsT = NamedObsTNCuda<TNDeviceT>;

    const std::size_t bondDim = GENERATE(2, 3, 4, 5);
    constexpr std::size_t num_qubits = 2;
    const std::size_t maxBondDim = bondDim;

    std::unique_ptr<TNDeviceT> tn_state =
        createTNState<TNDeviceT>(num_qubits, maxBondDim);

    auto measure = MeasurementsTNCuda<TNDeviceT>(*tn_state);

    SECTION("var(Identity[0])") {
        tn_state->applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        tn_state_append_mps_final_state(tn_state);
        auto ob = NamedObsT("Identity", {0});
        auto res = measure.var(ob);
        auto expected = PrecisionT(0);
        CHECK(res == Approx(expected).margin(1e-7));
    }

    SECTION("var(PauliX[0])") {
        tn_state->applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        tn_state_append_mps_final_state(tn_state);
        auto ob = NamedObsT("PauliX", {0});
        auto res = measure.var(ob);
        auto expected = PrecisionT(0.75722220);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliY[0])") {
        tn_state->applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        tn_state_append_mps_final_state(tn_state);
        auto ob = NamedObsT("PauliY", {0});
        auto res = measure.var(ob);
        auto expected = PrecisionT(0.58498357);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliZ[1])") {
        tn_state->applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        tn_state_append_mps_final_state(tn_state);
        auto ob = NamedObsT("PauliZ", {1});
        auto res = measure.var(ob);
        auto expected = PrecisionT(0.40686720);
        CHECK(res == Approx(expected));
    }

    SECTION("var(Hadamard[1])") {
        tn_state->applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        tn_state_append_mps_final_state(tn_state);
        auto ob = NamedObsT("Hadamard", {1});
        auto res = measure.var(ob);
        auto expected = PrecisionT(0.29089449);
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_LIST_TEST_CASE("Test variance of HermitianObs", "[TNCuda_Var]",
                        TestTNBackends) {
    using TNDeviceT = TestType;
    using PrecisionT = typename TNDeviceT::PrecisionT;
    using ComplexT = typename TNDeviceT::ComplexT;
    using HermitianObsT = HermitianObsTNCuda<TNDeviceT>;

    const std::size_t bondDim = GENERATE(2, 3, 4, 5);
    constexpr std::size_t num_qubits = 3;
    const std::size_t maxBondDim = bondDim;

    std::unique_ptr<TNDeviceT> tn_state =
        createTNState<TNDeviceT>(num_qubits, maxBondDim);

    auto measure = MeasurementsTNCuda<TNDeviceT>(*tn_state);

    tn_state->applyOperations(
        {{"RX"}, {"RY"}, {"RX"}, {"RY"}, {"RX"}, {"RY"}},
        {{0}, {0}, {1}, {1}, {2}, {2}},
        {{false}, {false}, {false}, {false}, {false}, {false}},
        {{0.7}, {0.7}, {0.5}, {0.5}, {0.3}, {0.3}});

    tn_state_append_mps_final_state(tn_state);

    SECTION("Target at 1 wire") {
        std::vector<ComplexT> matrix = {
            {2.5, 0.0}, {1.0, 1.0}, {1.0, -1.0}, {3.8, 0.0}};

        auto ob = HermitianObsT(matrix, {0});
        auto res = measure.var(ob);
        auto expected = PrecisionT(1.8499002); // from default.qubit
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_LIST_TEST_CASE("Test variance of TensorProdObs", "[TNCuda_Var]",
                        TestTNBackends) {
    using TNDeviceT = TestType;
    using PrecisionT = typename TNDeviceT::PrecisionT;
    using NamedObsT = NamedObsTNCuda<TNDeviceT>;
    using TensorProdObsT = TensorProdObsTNCuda<TNDeviceT>;

    const std::size_t bondDim = GENERATE(2, 3, 4, 5);
    constexpr std::size_t num_qubits = 3;
    const std::size_t maxBondDim = bondDim;

    std::unique_ptr<TNDeviceT> tn_state =
        createTNState<TNDeviceT>(num_qubits, maxBondDim);

    auto measure = MeasurementsTNCuda<TNDeviceT>(*tn_state);

    SECTION("Using var") {
        tn_state->applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        tn_state_append_mps_final_state(tn_state);

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto Z1 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1});

        auto ob = TensorProdObsT::create({X0, Z1});
        auto res = measure.var(*ob);
        auto expected = PrecisionT(0.83667953);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_LIST_TEST_CASE("Test var value of HamiltonianObs", "[TNCuda_Var]",
                        TestTNBackends) {
    using TNDeviceT = TestType;
    using PrecisionT = typename TNDeviceT::PrecisionT;
    using ComplexT = typename TNDeviceT::ComplexT;
    using NamedObsT = NamedObsTNCuda<TNDeviceT>;
    using HermitianObsT = HermitianObsTNCuda<TNDeviceT>;
    using TensorProdObsT = TensorProdObsTNCuda<TNDeviceT>;
    using HamiltonianObsT = HamiltonianTNCuda<TNDeviceT>;

    SECTION("Using TensorProd") {
        const std::size_t bondDim = GENERATE(2, 3, 4, 5);
        constexpr std::size_t num_qubits = 5;
        constexpr std::size_t num_paulis = 5;
        constexpr std::size_t num_obs_terms = 6;
        const std::size_t maxBondDim = bondDim;

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        tn_state->applyOperations(
            {{"RX"},
             {"RY"},
             {"RX"},
             {"RY"},
             {"RX"},
             {"RY"},
             {"RX"},
             {"RY"},
             {"RX"},
             {"RY"}},
            {{0}, {0}, {1}, {1}, {2}, {2}, {3}, {3}, {4}, {4}},
            {{false},
             {false},
             {false},
             {false},
             {false},
             {false},
             {false},
             {false},
             {false},
             {false}},
            {{0.5},
             {0.5},
             {0.2},
             {0.2},
             {0.5},
             {0.5},
             {0.2},
             {0.2},
             {0.5},
             {0.5}});
        tn_state_append_mps_final_state(tn_state);

        auto m = MeasurementsTNCuda<TNDeviceT>(*tn_state);

        std::array<std::string, num_paulis> paulis = {
            "Identity", "PauliX", "PauliY", "PauliZ", "Hadamard"};

        std::array<std::array<std::shared_ptr<NamedObsT>, num_qubits>,
                   num_paulis>
            named_obs;
        // create Paulis(0)-Paulis(num_qubits-1) for each Pauli
        for (std::size_t i = 0; i < num_paulis; i++) {
            for (std::size_t j = 0; j < num_qubits; j++) {
                named_obs[i][j] = std::make_shared<NamedObsT>(
                    paulis[i], std::vector<std::size_t>{j});
            }
        }

        std::array<std::shared_ptr<TensorProdObsT>, num_obs_terms>
            obs_tensor_prod;

        // Create the tensor product of the observables
        // I(i % num_qubits)@X((i + 1) % num_qubits)@Y((i + 2) %
        // num_qubits)@Z((i + 3) % num_qubits)@H((i + 4) % num_qubits)
        for (std::size_t i = 0; i < num_obs_terms - 1; i++) {
            obs_tensor_prod[i + 1] =
                TensorProdObsT::create({named_obs[0][i % num_qubits],
                                        named_obs[1][(i + 1) % num_qubits],
                                        named_obs[2][(i + 2) % num_qubits],
                                        named_obs[3][(i + 3) % num_qubits],
                                        named_obs[4][(i + 4) % num_qubits]});
        }

        obs_tensor_prod[0] = obs_tensor_prod[1];

        std::initializer_list<PrecisionT> coeffs{0.3, 0.5, 0.3, 0.5, 0.3, 0.5};
        std::initializer_list<std::shared_ptr<ObservableTNCuda<TNDeviceT>>> obs{
            obs_tensor_prod[0], obs_tensor_prod[1], obs_tensor_prod[2],
            obs_tensor_prod[3], obs_tensor_prod[4], obs_tensor_prod[5]};

        auto ob = HamiltonianObsT::create(coeffs, obs);

        auto res = m.var(*ob);
        CHECK(res == Approx(1.0830144)); // results from default.qubit
    }

    SECTION("Using 1 Hermitian") {
        const std::size_t bondDim = GENERATE(2, 3, 4, 5);
        constexpr std::size_t num_qubits = 3;
        const std::size_t maxBondDim = bondDim;

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        tn_state->applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        tn_state_append_mps_final_state(tn_state);

        auto m = MeasurementsTNCuda<TNDeviceT>(*tn_state);

        std::vector<ComplexT> matrix = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

        auto Herm0 = std::make_shared<HermitianObsT>(
            matrix, std::vector<std::size_t>{0});

        auto Herm1 = std::make_shared<HermitianObsT>(
            matrix, std::vector<std::size_t>{1});

        auto Z0 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{0});
        auto Z1 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1});

        auto obs0 = TensorProdObsT::create({Z0, Z1});
        auto obs1 = TensorProdObsT::create({Herm0, Herm1});

        auto ob = HamiltonianObsT::create({PrecisionT{0.3}, PrecisionT{0.5}},
                                          {obs0, obs1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.24231647));
    }

    SECTION("Using 2 Hermitians") {
        const std::size_t bondDim = GENERATE(2, 3, 4, 5);
        constexpr std::size_t num_qubits = 3;
        const std::size_t maxBondDim = bondDim;

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        tn_state->applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        tn_state_append_mps_final_state(tn_state);

        auto m = MeasurementsTNCuda<TNDeviceT>(*tn_state);

        std::vector<ComplexT> matrix = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

        std::vector<ComplexT> matrix_z = {
            {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}};

        auto X0 = std::make_shared<HermitianObsT>(matrix,
                                                  std::vector<std::size_t>{0});
        auto Z1 = std::make_shared<HermitianObsT>(matrix_z,
                                                  std::vector<std::size_t>{1});

        auto ob = HamiltonianObsT::create({PrecisionT{0.3}, PrecisionT{0.5}},
                                          {X0, Z1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.09341363));
    }
}
