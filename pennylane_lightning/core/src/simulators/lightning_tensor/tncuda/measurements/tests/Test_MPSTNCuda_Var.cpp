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
        auto expected = TestType(0.7572222074);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliY[0])") {
        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        mps_state.append_mps_final_state();
        auto ob = NamedObsT("PauliY", {0});
        auto res = measure.var(ob);
        auto expected = TestType(0.5849835715);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliZ[1])") {
        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        mps_state.append_mps_final_state();
        auto ob = NamedObsT("PauliZ", {1});
        auto res = measure.var(ob);
        auto expected = TestType(0.4068672016);
        CHECK(res == Approx(expected));
    }

    SECTION("var(Hadamard[1])") {
        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});
        mps_state.append_mps_final_state();
        auto ob = NamedObsT("Hadamard", {1});
        auto res = measure.var(ob);
        auto expected = TestType(0.2908944989);
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
        auto expected = TestType(1.8499002205); // from default.qubit
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
        auto expected = TestType(0.836679);
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
    SECTION("Using XI") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto I1 = std::make_shared<NamedObsT>("Identity",
                                              std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {X0, I1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.07406834808884483));
    }

    SECTION("Using XX") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto X1 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {X0, X1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.3145904327559435));
    }

    SECTION("Using XY") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto Y1 =
            std::make_shared<NamedObsT>("PauliY", std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {X0, Y1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.3142009723392052));
    }

    SECTION("Using XZ") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto Z1 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {X0, Z1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.093413));
    }

    SECTION("Using XH") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto H1 = std::make_shared<NamedObsT>("Hadamard",
                                              std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {X0, H1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.15724601172876368));
    }

    SECTION("Using YI") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto Y0 =
            std::make_shared<NamedObsT>("PauliY", std::vector<std::size_t>{0});
        auto I1 = std::make_shared<NamedObsT>("Identity",
                                              std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {Y0, I1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.06931360376406634));
    }

    SECTION("Using YX") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto Y0 =
            std::make_shared<NamedObsT>("PauliY", std::vector<std::size_t>{0});
        auto X1 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {Y0, X1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.30983568843116516));
    }

    SECTION("Using YY") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto Y0 =
            std::make_shared<NamedObsT>("PauliY", std::vector<std::size_t>{0});
        auto Y1 =
            std::make_shared<NamedObsT>("PauliY", std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {Y0, Y1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.3094462280144269));
    }

    SECTION("Using YZ") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto Y0 =
            std::make_shared<NamedObsT>("PauliY", std::vector<std::size_t>{0});
        auto Z1 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {Y0, Z1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.08865889484660677));
    }

    SECTION("Using YH") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto H0 = std::make_shared<NamedObsT>("Hadamard",
                                              std::vector<std::size_t>{0});
        auto Y1 =
            std::make_shared<NamedObsT>("PauliY", std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {H0, Y1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.26631312915836103));
    }

    SECTION("Using ZI") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto Z0 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{0});
        auto I1 = std::make_shared<NamedObsT>("Identity",
                                              std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {Z0, I1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.03661804814708902));
    }

    SECTION("Using HI") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto H0 = std::make_shared<NamedObsT>("Hadamard",
                                              std::vector<std::size_t>{0});
        auto I1 = std::make_shared<NamedObsT>("Identity",
                                              std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {H0, I1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.02618050490800028));
    }

    SECTION("Using ZY") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto Z0 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{0});
        auto Y1 =
            std::make_shared<NamedObsT>("PauliY", std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {Z0, Y1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.27675067239744966));
    }

    SECTION("Using XY") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto Y1 =
            std::make_shared<NamedObsT>("PauliY", std::vector<std::size_t>{1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {X0, Y1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.314201));
    }

    SECTION("Using 1 Hermitian") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        std::vector<ComplexT> matrix = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

        auto X0 = std::make_shared<HermitianObsT>(matrix,
                                                  std::vector<std::size_t>{0});
        auto Z1 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1});

        auto ob_term = TensorProdObsT::create({X0, Z1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {X0, Z1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.093413));
    }

    SECTION("Using 2 Hermitians") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        std::vector<ComplexT> matrix = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

        std::vector<ComplexT> matrix_z = {
            {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}};

        auto X0 = std::make_shared<HermitianObsT>(matrix,
                                                  std::vector<std::size_t>{0});
        auto Z1 = std::make_shared<HermitianObsT>(matrix_z,
                                                  std::vector<std::size_t>{1});

        auto ob_term = TensorProdObsT::create({X0, Z1});

        auto ob =
            HamiltonianObsT::create({TestType{0.3}, TestType{0.5}}, {X0, Z1});

        auto res = m.var(*ob);
        CHECK(res == Approx(0.093413));
    }
}
