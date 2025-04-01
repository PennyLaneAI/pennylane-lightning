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
using namespace Pennylane::LightningTensor::TNCuda::Util;

} // namespace
/// @endcond

TEMPLATE_LIST_TEST_CASE("[Identity]", "[TNCuda_Expval]", TestTNBackends) {
    using TNDeviceT = TestType;
    using PrecisionT = typename TNDeviceT::PrecisionT;
    using NamedObsT = NamedObsTNCuda<TNDeviceT>;
    auto ONE = PrecisionT(1);

    const std::size_t bondDim = GENERATE(2, 3, 4, 5);
    constexpr std::size_t num_qubits = 3;
    const std::size_t maxBondDim = bondDim;

    std::unique_ptr<TNDeviceT> tn_state =
        createTNState<TNDeviceT>(num_qubits, maxBondDim);

    auto measure = MeasurementsTNCuda<TNDeviceT>(*tn_state);

    SECTION("Using expval") {
        tn_state->applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                  {{0}, {0, 1}, {1, 2}},
                                  {{false}, {false}, {false}});

        tn_state_append_mps_final_state(tn_state);

        auto ob = NamedObsT("Identity", {0});
        auto res = measure.expval(ob);
        CHECK(res == Approx(ONE));
    }
}

TEMPLATE_LIST_TEST_CASE("[PauliX]", "[TNCuda_Expval]", TestTNBackends) {
    {
        using TNDeviceT = TestType;
        using PrecisionT = typename TNDeviceT::PrecisionT;
        using NamedObsT = NamedObsTNCuda<TNDeviceT>;

        const std::size_t bondDim = GENERATE(2, 3, 4, 5);
        constexpr std::size_t num_qubits = 3;
        const std::size_t maxBondDim = bondDim;

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        auto measure = MeasurementsTNCuda<TNDeviceT>(*tn_state);

        auto ZERO = PrecisionT(0);
        auto ONE = PrecisionT(1);

        SECTION("Using expval") {
            tn_state->applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                      {{0}, {0, 1}, {1, 2}},
                                      {{false}, {false}, {false}});
            tn_state_append_mps_final_state(tn_state);
            auto ob = NamedObsT("PauliX", {0});
            auto res = measure.expval(ob);
            CHECK(res == ZERO);
        }

        SECTION("Using expval: Plus states") {
            tn_state->applyOperations(
                {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                {{false}, {false}, {false}});
            tn_state_append_mps_final_state(tn_state);
            auto ob = NamedObsT("PauliX", {0});
            auto res = measure.expval(ob);
            CHECK(res == Approx(ONE));
        }

        SECTION("Using expval: Minus states") {
            tn_state->applyOperations(
                {{"PauliX"},
                 {"Hadamard"},
                 {"PauliX"},
                 {"Hadamard"},
                 {"PauliX"},
                 {"Hadamard"}},
                {{0}, {0}, {1}, {1}, {2}, {2}},
                {{false}, {false}, {false}, {false}, {false}, {false}});
            tn_state_append_mps_final_state(tn_state);
            auto ob = NamedObsT("PauliX", {0});
            auto res = measure.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("[PauliY]", "[TNCuda_Expval]", TestTNBackends) {
    {
        using TNDeviceT = TestType;
        using PrecisionT = typename TNDeviceT::PrecisionT;
        using NamedObsT = NamedObsTNCuda<TNDeviceT>;

        const std::size_t bondDim = GENERATE(2, 3, 4, 5);
        constexpr std::size_t num_qubits = 3;
        const std::size_t maxBondDim = bondDim;

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        auto measure = MeasurementsTNCuda<TNDeviceT>(*tn_state);

        auto ZERO = PrecisionT(0);
        auto ONE = PrecisionT(1);
        auto PI = PrecisionT(M_PI);

        SECTION("Using expval") {
            tn_state->applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                      {{0}, {0, 1}, {1, 2}},
                                      {{false}, {false}, {false}});
            auto ob = NamedObsT("PauliY", {0});
            auto res = measure.expval(ob);
            CHECK(res == ZERO);
        }

        SECTION("Using expval: Plus i states") {
            tn_state->applyOperations({{"RX"}, {"RX"}, {"RX"}}, {{0}, {1}, {2}},
                                      {{false}, {false}, {false}},
                                      {{-PI / 2}, {-PI / 2}, {-PI / 2}});
            auto ob = NamedObsT("PauliY", {0});
            auto res = measure.expval(ob);
            CHECK(res == Approx(ONE));
        }

        SECTION("Using expval: Minus i states") {
            tn_state->applyOperations({{"RX"}, {"RX"}, {"RX"}}, {{0}, {1}, {2}},
                                      {{false}, {false}, {false}},
                                      {{PI / 2}, {PI / 2}, {PI / 2}});
            auto ob = NamedObsT("PauliY", {0});
            auto res = measure.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("[PauliZ]", "[TNCuda_Expval]", TestTNBackends) {
    {
        using TNDeviceT = TestType;
        using PrecisionT = typename TNDeviceT::PrecisionT;

        using NamedObsT = NamedObsTNCuda<TNDeviceT>;

        const std::size_t bondDim = GENERATE(2, 3, 4, 5);
        constexpr std::size_t num_qubits = 3;
        const std::size_t maxBondDim = bondDim;

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        SECTION("Using expval") {
            tn_state->applyOperations(
                {{"RX"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                {{false}, {false}, {false}}, {{0.5}, {}, {}});
            auto m = MeasurementsTNCuda<TNDeviceT>(*tn_state);
            auto ob = NamedObsT("PauliZ", {0});
            auto res = m.expval(ob);
            PrecisionT ref = 0.8775825618903724;
            REQUIRE(res == Approx(ref).margin(1e-6));
        }

        SECTION("Using expval mps with cutoff") {
            double cutoff = GENERATE(1e-1, 1e-2);
            std::string cutoff_mode = GENERATE("rel", "abs");
            tn_state->applyOperations(
                {{"Hadamard"},
                 {"Hadamard"},
                 {"Hadamard"},
                 {"SingleExcitation"},
                 {"IsingXX"},
                 {"IsingXY"}},
                {{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}},
                {{false}, {false}, {false}, {false}, {false}, {false}},
                {{}, {}, {}, {0.5}, {0.6}, {0.7}});

            if constexpr (std::is_same_v<TNDeviceT, MPSTNCuda<double>> ||
                          std::is_same_v<TNDeviceT, MPSTNCuda<float>>) {
                tn_state->append_mps_final_state(cutoff, cutoff_mode);
            }

            auto m = MeasurementsTNCuda<TNDeviceT>(*tn_state);
            auto ob = NamedObsT("PauliZ", {0});
            auto res = m.expval(ob);
            // ref is from default.qubit
            PrecisionT ref = -0.2115276040475712;

            REQUIRE_THAT(res, Catch::Matchers::WithinRel(
                                  ref, static_cast<PrecisionT>(0.1)));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("[Hadamard]", "[TNCuda_Expval]", TestTNBackends) {
    {
        using TNDeviceT = TestType;
        using PrecisionT = typename TNDeviceT::PrecisionT;
        using NamedObsT = NamedObsTNCuda<TNDeviceT>;

        const std::size_t bondDim = GENERATE(2, 3, 4, 5);
        constexpr std::size_t num_qubits = 3;
        const std::size_t maxBondDim = bondDim;

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        auto measure = MeasurementsTNCuda<TNDeviceT>(*tn_state);

        auto ONE = PrecisionT(1);

        // NOTE: Following tests show that the current design can be measured
        // multiple times with different observables
        SECTION("Using expval") {
            tn_state->applyOperation("PauliX", {0});
            tn_state_append_mps_final_state(tn_state);

            auto ob = NamedObsT("Hadamard", {0});
            auto res = measure.expval(ob);
            CHECK(
                res ==
                Approx(-Pennylane::Util::INVSQRT2<PrecisionT>()).epsilon(1e-7));

            auto ob1 = NamedObsT("Identity", {0});
            auto res1 = measure.expval(ob1);
            CHECK(res1 == Approx(ONE));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("[Parametric_obs]", "[TNCuda_Expval]", TestTNBackends) {
    {
        using TNDeviceT = TestType;
        using PrecisionT = typename TNDeviceT::PrecisionT;
        using NamedObsT = NamedObsTNCuda<TNDeviceT>;

        const std::size_t bondDim = GENERATE(2, 3, 4, 5);
        constexpr std::size_t num_qubits = 3;
        const std::size_t maxBondDim = bondDim;

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        auto measure = MeasurementsTNCuda<TNDeviceT>(*tn_state);

        auto ONE = PrecisionT(1);

        SECTION("Using expval") {
            tn_state->applyOperation("PauliX", {0});
            tn_state_append_mps_final_state(tn_state);

            auto ob = NamedObsT("RX", {0}, {0});
            auto res = measure.expval(ob);
            CHECK(res == Approx(ONE).epsilon(1e-7));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("[Hermitian]", "[TNCuda_Expval]", TestTNBackends) {
    {
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

        auto ZERO = PrecisionT(0);
        auto ONE = PrecisionT(1);

        std::vector<ComplexT> mat = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

        SECTION("Using expval") {
            tn_state->applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                                      {{0}, {0, 1}, {1, 2}},
                                      {{false}, {false}, {false}});
            tn_state_append_mps_final_state(tn_state);
            auto ob = HermitianObsT(mat, std::vector<std::size_t>{0});
            auto res = measure.expval(ob);
            CHECK(res == ZERO);
        }

        SECTION("Using expval: Plus states") {
            tn_state->applyOperations(
                {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                {{false}, {false}, {false}});
            tn_state_append_mps_final_state(tn_state);
            auto ob = HermitianObsT(mat, {0});
            auto res = measure.expval(ob);
            CHECK(res == Approx(ONE));
        }

        SECTION("Using expval: Minus states") {
            tn_state->applyOperations(
                {{"PauliX"},
                 {"Hadamard"},
                 {"PauliX"},
                 {"Hadamard"},
                 {"PauliX"},
                 {"Hadamard"}},
                {{0}, {0}, {1}, {1}, {2}, {2}},
                {{false}, {false}, {false}, {false}, {false}, {false}});
            tn_state_append_mps_final_state(tn_state);
            auto ob = HermitianObsT(mat, {0});
            auto res = measure.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("Test expectation value of TensorProdObs",
                        "[TNCuda_Expval]", TestTNBackends) {
    using TNDeviceT = TestType;
    using PrecisionT = typename TNDeviceT::PrecisionT;
    using NamedObsT = NamedObsTNCuda<TNDeviceT>;
    using TensorProdObsT = TensorProdObsTNCuda<TNDeviceT>;
    auto ZERO = PrecisionT(0);
    SECTION("Using XZ") {
        std::size_t bondDim = GENERATE(2);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        tn_state->applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        auto m = MeasurementsTNCuda<TNDeviceT>(*tn_state);

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

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        tn_state->applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        auto m = MeasurementsTNCuda<TNDeviceT>(*tn_state);

        auto H0 = std::make_shared<NamedObsT>("Hadamard",
                                              std::vector<std::size_t>{0});
        auto H1 = std::make_shared<NamedObsT>("Hadamard",
                                              std::vector<std::size_t>{1});
        auto H2 = std::make_shared<NamedObsT>("Hadamard",
                                              std::vector<std::size_t>{2});

        auto ob = TensorProdObsT::create({H0, H1, H2});
        auto res = m.expval(*ob);
        CHECK(res == Approx(Pennylane::Util::INVSQRT2<PrecisionT>() / 2));
    }
}

TEMPLATE_LIST_TEST_CASE("Test expectation value of HamiltonianObs",
                        "[TNCuda_Expval]", TestTNBackends) {
    using TNDeviceT = TestType;
    using PrecisionT = typename TNDeviceT::PrecisionT;
    using NamedObsT = NamedObsTNCuda<TNDeviceT>;
    using HamiltonianObsT = HamiltonianTNCuda<TNDeviceT>;
    auto ONE = PrecisionT(1);
    SECTION("Using XZ") {
        const std::size_t bondDim = GENERATE(2);
        constexpr std::size_t num_qubits = 3;
        const std::size_t maxBondDim = bondDim;

        std::unique_ptr<TNDeviceT> tn_state =
            createTNState<TNDeviceT>(num_qubits, maxBondDim);

        tn_state->applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        auto m = MeasurementsTNCuda<TNDeviceT>(*tn_state);

        auto X0 =
            std::make_shared<NamedObsT>("PauliX", std::vector<std::size_t>{0});
        auto Z1 =
            std::make_shared<NamedObsT>("PauliZ", std::vector<std::size_t>{1});

        auto ob = HamiltonianObsT::create({ONE, ONE}, {X0, Z1});
        auto res = m.expval(*ob);
        CHECK(res == Approx(ONE));
    }
}
