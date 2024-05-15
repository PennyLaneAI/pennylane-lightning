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
#include <limits>
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
    using StateTensorT = MPSTNCuda<TestType>;
    auto ONE = TestType(1);

    std::size_t bondDim = GENERATE(2, 3, 4, 5);
    std::size_t num_qubits = 3;
    std::size_t maxBondDim = bondDim;

    StateTensorT mps_state{num_qubits, maxBondDim};

    auto measure = Measurements<StateTensorT>();

    SECTION("Using expval") {
        mps_state.applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                  {{0}, {0, 1}, {1, 2}},
                                  {{false}, {false}, {false}});
        mps_state.get_final_state();
        auto ob = NamedObs<StateTensorT>("Identity", {0});
        auto res = measure.expval(ob, mps_state);
        CHECK(res == Approx(ONE));
    }
}

TEMPLATE_TEST_CASE("[PauliX]", "[MPSTNCuda_Expval]", float, double) {
    {
        using StateTensorT = MPSTNCuda<TestType>;

        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        StateTensorT mps_state{num_qubits, maxBondDim};

        auto measure = Measurements<StateTensorT>();

        auto ZERO = TestType(0);
        auto ONE = TestType(1);

        SECTION("Using expval") {
            mps_state.applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                      {{0}, {0, 1}, {1, 2}},
                                      {{false}, {false}, {false}});
            mps_state.get_final_state();
            auto ob = NamedObs<StateTensorT>("PauliX", {0});
            auto res = measure.expval(ob, mps_state);
            CHECK(res == ZERO);
        }

        SECTION("Using expval: Plus states") {
            mps_state.applyOperations(
                {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                {{false}, {false}, {false}});
            mps_state.get_final_state();
            auto ob = NamedObs<StateTensorT>("PauliX", {0});
            auto res = measure.expval(ob, mps_state);
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
            mps_state.get_final_state();
            auto ob = NamedObs<StateTensorT>("PauliX", {0});
            auto res = measure.expval(ob, mps_state);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("[PauliY]", "[MPSTNCuda_Expval]", float, double) {
    {
        using StateTensorT = MPSTNCuda<TestType>;

        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        StateTensorT mps_state{num_qubits, maxBondDim};

        auto measure = Measurements<StateTensorT>();

        auto ZERO = TestType(0);
        auto ONE = TestType(1);
        auto PI = TestType(M_PI);

        SECTION("Using expval") {
            mps_state.applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                      {{0}, {0, 1}, {1, 2}},
                                      {{false}, {false}, {false}});
            auto ob = NamedObs<StateTensorT>("PauliY", {0});
            auto res = measure.expval(ob, mps_state);
            CHECK(res == ZERO);
        }

        SECTION("Using expval: Plus i states") {
            mps_state.applyOperations({{"RX"}, {"RX"}, {"RX"}}, {{0}, {1}, {2}},
                                      {{false}, {false}, {false}},
                                      {{-PI / 2}, {-PI / 2}, {-PI / 2}});
            auto ob = NamedObs<StateTensorT>("PauliY", {0});
            auto res = measure.expval(ob, mps_state);
            CHECK(res == Approx(ONE));
        }

        SECTION("Using expval: Minus i states") {
            mps_state.applyOperations({{"RX"}, {"RX"}, {"RX"}}, {{0}, {1}, {2}},
                                      {{false}, {false}, {false}},
                                      {{PI / 2}, {PI / 2}, {PI / 2}});
            auto ob = NamedObs<StateTensorT>("PauliY", {0});
            auto res = measure.expval(ob, mps_state);
            CHECK(res == -Approx(ONE));
        }
    }
}
/*
TEMPLATE_TEST_CASE("[PauliZ]", "[MPSTNCuda_Expval]", float,
                   double) {
    {
        using StateTensorT = MPSTNCuda<TestType>;
        using PrecisionT = StateVectorT::PrecisionT;

        // Defining the statevector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT sv(statevector_data.data(), statevector_data.size());

        SECTION("Using expval") {
            auto m = Measurements(sv);
            auto ob = NamedObs<StateVectorT>("PauliZ", {1});
            auto res = m.expval(ob);
            PrecisionT ref = 0.77015115;
            REQUIRE(res == Approx(ref).margin(1e-6));
        }
    }
}
*/

TEMPLATE_TEST_CASE("[Hadamard]", "[MPSTNCuda_Expval]", float, double) {
    {
        using StateTensorT = MPSTNCuda<TestType>;

        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        StateTensorT mps_state{num_qubits, maxBondDim};

        auto measure = Measurements<StateTensorT>();

        auto INVSQRT2 = TestType(0.707106781186547524401);

        SECTION("Using expval") {
            mps_state.applyOperation("PauliX", {0});
            mps_state.get_final_state();

            auto ob = NamedObs<StateTensorT>("Hadamard", {0});
            auto res = measure.expval(ob, mps_state);
            CHECK(res == Approx(-INVSQRT2).epsilon(1e-7));
        }
    }
}