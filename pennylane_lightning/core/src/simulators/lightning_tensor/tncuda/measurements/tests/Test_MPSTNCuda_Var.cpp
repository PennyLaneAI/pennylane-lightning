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

#include "MPSTNCuda.hpp"
#include "MeasurementsTNCuda.hpp"
#include "TNCudaGateCache.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda::Measures;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Test variance of NamedObs", "[MPSTNCuda_Var]", float,
                   double) {
    using TensorNetT = MPSTNCuda<TestType>;
    using NamedObsT = NamedObsTNCuda<TensorNetT>;

    std::size_t bondDim = GENERATE(2, 3, 4, 5);
    std::size_t num_qubits = 2;
    std::size_t maxBondDim = bondDim;

    TensorNetT mps_state{num_qubits, maxBondDim};

    auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

    SECTION("var(Identity[0])") {
        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        mps_state.append_mps_final_state();
        auto ob = NamedObsT("Identity", {0});
        auto res = measure.var(ob);
        auto expected = TestType(0);
        CHECK(res == Approx(expected).margin(1e-7));
    }

    SECTION("var(PauliX[0])") {
        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        mps_state.append_mps_final_state();
        auto ob = NamedObsT("PauliX", {0});
        auto res = measure.var(ob);
        auto expected = TestType(0.75722220);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliY[0])") {
        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        mps_state.append_mps_final_state();
        auto ob = NamedObsT("PauliY", {0});
        auto res = measure.var(ob);
        auto expected = TestType(0.58498357);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliZ[1])") {
        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        mps_state.append_mps_final_state();
        auto ob = NamedObsT("PauliZ", {1});
        auto res = measure.var(ob);
        auto expected = TestType(0.40686720);
        CHECK(res == Approx(expected));
    }

    SECTION("var(Hadamard[1])") {
        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        mps_state.append_mps_final_state();
        auto ob = NamedObsT("Hadamard", {1});
        auto res = measure.var(ob);
        auto expected = TestType(0.29089449);
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_TEST_CASE("Test variance of HermitianObs", "[MPSTNCuda_Var]", float,
                   double) {
    using TensorNetT = MPSTNCuda<TestType>;
    using ComplexT = typename TensorNetT::ComplexT;
    using HermitianObsT = HermitianObsTNCuda<TensorNetT>;

    std::size_t bondDim = GENERATE(2, 3, 4, 5);
    std::size_t num_qubits = 3;
    std::size_t maxBondDim = bondDim;

    TensorNetT mps_state{num_qubits, maxBondDim};

    auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

    mps_state.applyOperations(
        {{"RX"}, {"RY"}, {"RX"}, {"RY"}, {"RX"}, {"RY"}},
        {{0}, {0}, {1}, {1}, {2}, {2}},
        {{false}, {false}, {false}, {false}, {false}, {false}},
        {{0.7}, {0.7}, {0.5}, {0.5}, {0.3}, {0.3}});
    mps_state.append_mps_final_state();

    SECTION("Target at 1 wire") {
        std::vector<ComplexT> matrix = {
            {2.5, 0.0}, {1.0, 1.0}, {1.0, -1.0}, {3.8, 0.0}};

        auto ob = HermitianObsT(matrix, {0});
        auto res = measure.var(ob);
        auto expected = TestType(1.8499002); // from default.qubit
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_TEST_CASE("Test variance of TensorProdObs", "[MPSTNCuda_Var]", float,
                   double) {
    using TensorNetT = MPSTNCuda<TestType>;
    using NamedObsT = NamedObsTNCuda<TensorNetT>;
    using TensorProdObsT = TensorProdObsTNCuda<TensorNetT>;

    std::size_t bondDim = GENERATE(2, 3, 4, 5);
    std::size_t num_qubits = 3;
    std::size_t maxBondDim = bondDim;

    TensorNetT mps_state{num_qubits, maxBondDim};

    auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

    SECTION("Using var") {
        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        mps_state.append_mps_final_state();

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto Z1 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1});

        auto ob = TensorProdObsT::create({X0, Z1});
        auto res = measure.var(*ob);
        auto expected = TestType(0.83667953);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("Test var value of HamiltonianObs", "[MPSTNCuda_Var]", float,
                   double) {
    using TensorNetT = MPSTNCuda<TestType>;
    using ComplexT = typename TensorNetT::ComplexT;
    using NamedObsT = NamedObsTNCuda<TensorNetT>;
    using HermitianObsT = HermitianObsTNCuda<TensorNetT>;
    using TensorProdObsT = TensorProdObsTNCuda<TensorNetT>;
    using HamiltonianObsT = HamiltonianTNCuda<TensorNetT>;

    SECTION("Using TensorProd") {
        std::size_t bondDim = GENERATE(2);
        constexpr std::size_t num_qubits = 5;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
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
        mps_state.append_mps_final_state();

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        std::array<std::shared_ptr<NamedObsT>, num_qubits> I;
        std::array<std::shared_ptr<NamedObsT>, num_qubits> X;
        std::array<std::shared_ptr<NamedObsT>, num_qubits> Y;
        std::array<std::shared_ptr<NamedObsT>, num_qubits> Z;
        std::array<std::shared_ptr<NamedObsT>, num_qubits> H;

        for (std::size_t i = 0; i < num_qubits; i++) {
            I[i] = std::make_shared<NamedObsT>("Identity",
                                               std::vector<std::size_t>{i});

            X[i] = std::make_shared<NamedObsT>("PauliX",
                                               std::vector<std::size_t>{i});

            Y[i] = std::make_shared<NamedObsT>("PauliY",
                                               std::vector<std::size_t>{i});

            Z[i] = std::make_shared<NamedObsT>("PauliZ",
                                               std::vector<std::size_t>{i});

            H[i] = std::make_shared<NamedObsT>("Hadamard",
                                               std::vector<std::size_t>{i});
        }

        auto ob0 = TensorProdObsT::create({I[0], X[1], Y[2], Z[3], H[4]});
        auto ob1 = TensorProdObsT::create({I[0], X[1], Y[2], Z[3], H[4]});
        auto ob2 = TensorProdObsT::create({I[1], X[2], Y[3], Z[4], H[0]});
        auto ob3 = TensorProdObsT::create({I[2], X[3], Y[4], Z[0], H[1]});
        auto ob4 = TensorProdObsT::create({I[3], X[4], Y[0], Z[1], H[2]});
        auto ob5 = TensorProdObsT::create({I[4], X[0], Y[1], Z[2], H[3]});

        auto ob = HamiltonianObsT::create({TestType{0.3}, TestType{0.5},
                                           TestType{0.3}, TestType{0.5},
                                           TestType{0.3}, TestType{0.5}},
                                          {ob0, ob1, ob2, ob3, ob4, ob5});

        auto res = m.var(*ob);
        CHECK(res == Approx(1.0830144));
    }

    SECTION("Using 1 Hermitian") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        mps_state.append_mps_final_state();

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

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

        auto ob = HamiltonianObsT::create({TestType{0.3}, TestType{0.5}},
                                          {obs0, obs1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.24231647));
    }

    SECTION("Using 2 Hermitians") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        mps_state.append_mps_final_state();

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        std::vector<ComplexT> matrix = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

        std::vector<ComplexT> matrix_z = {
            {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}};

        auto X0 = std::make_shared<HermitianObsT>(matrix,
                                                  std::vector<std::size_t>{0});
        auto Z1 = std::make_shared<HermitianObsT>(matrix_z,
                                                  std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {X0, Z1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.09341363));
    }
}
