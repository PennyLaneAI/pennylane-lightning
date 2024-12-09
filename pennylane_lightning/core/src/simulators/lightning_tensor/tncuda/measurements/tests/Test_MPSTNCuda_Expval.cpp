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

TEMPLATE_TEST_CASE("[Identity]", "[MPSTNCuda_Expval]", float, double) {
    using TensorNetT = MPSTNCuda<TestType>;
    using NamedObsT = NamedObsTNCuda<TensorNetT>;
    auto ONE = TestType(1);

    std::size_t bondDim = GENERATE(2, 3, 4, 5);
    std::size_t num_qubits = 3;
    std::size_t maxBondDim = bondDim;

    TensorNetT mps_state{num_qubits, maxBondDim};

    auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

    SECTION("Using expval") {
        mps_state.applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                  {{0}, {0, 1}, {1, 2}},
                                  {{false}, {false}, {false}});
        mps_state.append_mps_final_state();
        auto ob = NamedObsT("Identity", {0});
        auto res = measure.expval(ob);
        CHECK(res == Approx(ONE));
    }
}

TEMPLATE_TEST_CASE("[PauliX]", "[MPSTNCuda_Expval]", float, double) {
    {
        using TensorNetT = MPSTNCuda<TestType>;
        using NamedObsT = NamedObsTNCuda<TensorNetT>;

        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto ZERO = TestType(0);
        auto ONE = TestType(1);

        SECTION("Using expval") {
            mps_state.applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                      {{0}, {0, 1}, {1, 2}},
                                      {{false}, {false}, {false}});
            mps_state.append_mps_final_state();
            auto ob = NamedObsT("PauliX", {0});
            auto res = measure.expval(ob);
            CHECK(res == ZERO);
        }

        SECTION("Using expval: Plus states") {
            mps_state.applyOperations(
                {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                {{false}, {false}, {false}});
            mps_state.append_mps_final_state();
            auto ob = NamedObsT("PauliX", {0});
            auto res = measure.expval(ob);
            CHECK(res == Approx(ONE));
        }

        SECTION("Using expval: Minus states") {
            mps_state.applyOperations(
                {{"PauliX"},
                 {"Hadamard"},
                 {"PauliX"},
                 {"Hadamard"},
                 {"PauliX"},
                 {"Hadamard"}},
                {{0}, {0}, {1}, {1}, {2}, {2}},
                {{false}, {false}, {false}, {false}, {false}, {false}});
            mps_state.append_mps_final_state();
            auto ob = NamedObsT("PauliX", {0});
            auto res = measure.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("[PauliY]", "[MPSTNCuda_Expval]", float, double) {
    {
        using TensorNetT = MPSTNCuda<TestType>;
        using NamedObsT = NamedObsTNCuda<TensorNetT>;

        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto ZERO = TestType(0);
        auto ONE = TestType(1);
        auto PI = TestType(M_PI);

        SECTION("Using expval") {
            mps_state.applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                      {{0}, {0, 1}, {1, 2}},
                                      {{false}, {false}, {false}});
            auto ob = NamedObsT("PauliY", {0});
            auto res = measure.expval(ob);
            CHECK(res == ZERO);
        }

        SECTION("Using expval: Plus i states") {
            mps_state.applyOperations({{"RX"}, {"RX"}, {"RX"}}, {{0}, {1}, {2}},
                                      {{false}, {false}, {false}},
                                      {{-PI / 2}, {-PI / 2}, {-PI / 2}});
            auto ob = NamedObsT("PauliY", {0});
            auto res = measure.expval(ob);
            CHECK(res == Approx(ONE));
        }

        SECTION("Using expval: Minus i states") {
            mps_state.applyOperations({{"RX"}, {"RX"}, {"RX"}}, {{0}, {1}, {2}},
                                      {{false}, {false}, {false}},
                                      {{PI / 2}, {PI / 2}, {PI / 2}});
            auto ob = NamedObsT("PauliY", {0});
            auto res = measure.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("[PauliZ]", "[MPSTNCuda_Expval]", float, double) {
    {
        using TensorNetT = MPSTNCuda<TestType>;
        using PrecisionT = TensorNetT::PrecisionT;
        using TensorNetT = MPSTNCuda<TestType>;
        using NamedObsT = NamedObsTNCuda<TensorNetT>;

        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        SECTION("Using expval") {
            mps_state.applyOperations(
                {{"RX"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                {{false}, {false}, {false}}, {{0.5}, {}, {}});
            auto m = MeasurementsTNCuda<TensorNetT>(mps_state);
            auto ob = NamedObsT("PauliZ", {0});
            auto res = m.expval(ob);
            PrecisionT ref = 0.8775825618903724;
            REQUIRE(res == Approx(ref).margin(1e-6));
        }

        SECTION("Using expval mps with cutoff") {
            double cutoff = GENERATE(1e-1, 1e-2);
            std::string cutoff_mode = GENERATE("rel", "abs");
            mps_state.applyOperations(
                {{"Hadamard"},
                 {"Hadamard"},
                 {"Hadamard"},
                 {"SingleExcitation"},
                 {"IsingXX"},
                 {"IsingXY"}},
                {{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}},
                {{false}, {false}, {false}, {false}, {false}, {false}},
                {{}, {}, {}, {0.5}, {0.6}, {0.7}});
            mps_state.append_mps_final_state(cutoff, cutoff_mode);
            auto m = MeasurementsTNCuda<TensorNetT>(mps_state);
            auto ob = NamedObsT("PauliZ", {0});
            auto res = m.expval(ob);
            // ref is from default.qubit
            PrecisionT ref = -0.2115276040475712;

            REQUIRE_THAT(res, Catch::Matchers::WithinRel(
                                  ref, static_cast<PrecisionT>(0.1)));
        }
    }
}

TEMPLATE_TEST_CASE("[Hadamard]", "[MPSTNCuda_Expval]", float, double) {
    {
        using TensorNetT = MPSTNCuda<TestType>;
        using NamedObsT = NamedObsTNCuda<TensorNetT>;

        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto INVSQRT2 = TestType(0.707106781186547524401);

        auto ONE = TestType(1);

        // NOTE: Following tests show that the current design can be measured
        // multiple times with different observables
        SECTION("Using expval") {
            mps_state.applyOperation("PauliX", {0});
            mps_state.append_mps_final_state();

            auto ob = NamedObsT("Hadamard", {0});
            auto res = measure.expval(ob);
            CHECK(res == Approx(-INVSQRT2).epsilon(1e-7));

            auto ob1 = NamedObsT("Identity", {0});
            auto res1 = measure.expval(ob1);
            CHECK(res1 == Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("[Parametric_obs]", "[MPSTNCuda_Expval]", float, double) {
    {
        using TensorNetT = MPSTNCuda<TestType>;
        using NamedObsT = NamedObsTNCuda<TensorNetT>;

        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);
        auto ONE = TestType(1);

        SECTION("Using expval") {
            mps_state.applyOperation("PauliX", {0});
            mps_state.append_mps_final_state();

            auto ob = NamedObsT("RX", {0}, {0});
            auto res = measure.expval(ob);
            CHECK(res == Approx(ONE).epsilon(1e-7));
        }
    }
}

TEMPLATE_TEST_CASE("[Hermitian]", "[MPSTNCuda_Expval]", float, double) {
    {
        using TensorNetT = MPSTNCuda<TestType>;
        using ComplexT = typename TensorNetT::ComplexT;
        using HermitianObsT = HermitianObsTNCuda<TensorNetT>;

        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto ZERO = TestType(0);
        auto ONE = TestType(1);

        std::vector<ComplexT> mat = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

        SECTION("Using expval") {
            mps_state.applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                      {{0}, {0, 1}, {1, 2}},
                                      {{false}, {false}, {false}});
            mps_state.append_mps_final_state();
            auto ob = HermitianObsT(mat, std::vector<std::size_t>{0});
            auto res = measure.expval(ob);
            CHECK(res == ZERO);
        }

        SECTION("Using expval: Plus states") {
            mps_state.applyOperations(
                {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                {{false}, {false}, {false}});
            mps_state.append_mps_final_state();
            auto ob = HermitianObsT(mat, {0});
            auto res = measure.expval(ob);
            CHECK(res == Approx(ONE));
        }

        SECTION("Using expval: Minus states") {
            mps_state.applyOperations(
                {{"PauliX"},
                 {"Hadamard"},
                 {"PauliX"},
                 {"Hadamard"},
                 {"PauliX"},
                 {"Hadamard"}},
                {{0}, {0}, {1}, {1}, {2}, {2}},
                {{false}, {false}, {false}, {false}, {false}, {false}});
            mps_state.append_mps_final_state();
            auto ob = HermitianObsT(mat, {0});
            auto res = measure.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("Test expectation value of TensorProdObs",
                   "[MPSTNCuda_Expval]", float, double) {
    using TensorNetT = MPSTNCuda<TestType>;
    using NamedObsT = NamedObsTNCuda<TensorNetT>;
    using TensorProdObsT = TensorProdObsTNCuda<TensorNetT>;
    auto ZERO = TestType(0);
    auto INVSQRT2 = TestType(0.707106781186547524401);
    SECTION("Using XZ") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto Z1 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1});

        auto ob = TensorProdObsT::create({X0, Z1});
        auto res = m.expval(*ob);
        CHECK(res == Approx(ZERO));
    }

    SECTION("Using HHH") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto H0 = std::make_shared<NamedObsT>("Hadamard",
                                              std::vector<std::size_t>{0});
        auto H1 = std::make_shared<NamedObsT>("Hadamard",
                                              std::vector<std::size_t>{1});
        auto H2 = std::make_shared<NamedObsT>("Hadamard",
                                              std::vector<std::size_t>{2});

        auto ob = TensorProdObsT::create({H0, H1, H2});
        auto res = m.expval(*ob);
        CHECK(res == Approx(INVSQRT2 / 2));
    }
}

TEMPLATE_TEST_CASE("Test expectation value of HamiltonianObs",
                   "[MPSTNCuda_Expval]", float, double) {
    using TensorNetT = MPSTNCuda<TestType>;
    using NamedObsT = NamedObsTNCuda<TensorNetT>;
    using HamiltonianObsT = HamiltonianTNCuda<TensorNetT>;
    auto ONE = TestType(1);
    SECTION("Using XZ") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        auto m = MeasurementsTNCuda<TensorNetT>(mps_state);

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto Z1 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1});

        auto ob = HamiltonianObsT::create({TestType{1}, TestType{1}}, {X0, Z1});
        auto res = m.expval(*ob);
        CHECK(res == Approx(ONE));
    }
}
