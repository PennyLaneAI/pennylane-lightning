// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

#include "MeasurementsLQubit.hpp"
#include "ObservablesLQubit.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "Util.hpp"

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

/// @cond DEV
namespace {
using namespace Pennylane::Util;

using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Observables;
using namespace Pennylane::LightningQubit::Measures;

}; // namespace
/// @endcond

TEMPLATE_PRODUCT_TEST_CASE("Expected Values", "[Measurements]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    // Defining the statevector that will be measured.
    auto statevector_data = createNonTrivialState<StateVectorT>();
    StateVectorT statevector(statevector_data.data(), statevector_data.size());

    // Initializing the Measurements class.
    // This object attaches to the statevector allowing several measures.
    Measurements<StateVectorT> Measurer(statevector);

    SECTION("Testing single operation defined by a matrix:") {
        std::vector<ComplexT> PauliX = {0, 1, 1, 0};
        std::vector<size_t> wires_single = {0};
        PrecisionT exp_value = Measurer.expval(PauliX, wires_single);
        PrecisionT exp_values_ref = 0.492725;
        REQUIRE(exp_value == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        std::vector<size_t> wires_single = {0};
        PrecisionT exp_value = Measurer.expval("PauliX", wires_single);
        PrecisionT exp_values_ref = 0.492725;
        REQUIRE(exp_value == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by a matrix:") {
        std::vector<ComplexT> PauliX = {0, 1, 1, 0};
        std::vector<ComplexT> PauliY = {0, {0, -1}, {0, 1}, 0};
        std::vector<ComplexT> PauliZ = {1, 0, 0, -1};

        std::vector<PrecisionT> exp_values;
        std::vector<PrecisionT> exp_values_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::vector<ComplexT>> operations_list;

        operations_list = {PauliX, PauliX, PauliX};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.49272486, 0.42073549, 0.28232124};
        CHECK_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliY, PauliY, PauliY};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {-0.64421768, -0.47942553, -0.29552020};
        CHECK_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliZ, PauliZ, PauliZ};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.58498357, 0.77015115, 0.91266780};
        CHECK_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by its name:") {
        std::vector<PrecisionT> exp_values;
        std::vector<PrecisionT> exp_values_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> operations_list;

        operations_list = {"PauliX", "PauliX", "PauliX"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.49272486, 0.42073549, 0.28232124};
        CHECK_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliY", "PauliY", "PauliY"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {-0.64421768, -0.47942553, -0.29552020};
        CHECK_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliZ", "PauliZ", "PauliZ"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.58498357, 0.77015115, 0.91266780};
        CHECK_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Variances", "[Measurements]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    // Defining the State Vector that will be measured.
    auto statevector_data = createNonTrivialState<StateVectorT>();
    StateVectorT statevector(statevector_data.data(), statevector_data.size());

    // Initializing the measurements class.
    // This object attaches to the statevector allowing several measurements.
    Measurements<StateVectorT> Measurer(statevector);

    SECTION("Testing single operation defined by a matrix:") {
        std::vector<ComplexT> PauliX = {0, 1, 1, 0};
        std::vector<size_t> wires_single = {0};
        PrecisionT variance = Measurer.var(PauliX, wires_single);
        PrecisionT variances_ref = 0.7572222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        std::vector<size_t> wires_single = {0};
        PrecisionT variance = Measurer.var("PauliX", wires_single);
        PrecisionT variances_ref = 0.7572222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by a matrix:") {
        std::vector<ComplexT> PauliX = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
        std::vector<ComplexT> PauliY = {{0, 0}, {0, -1}, {0, 1}, {0, 0}};
        std::vector<ComplexT> PauliZ = {{1, 0}, {0, 0}, {0, 0}, {-1, 0}};

        std::vector<PrecisionT> variances;
        std::vector<PrecisionT> variances_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::vector<ComplexT>> operations_list;

        operations_list = {PauliX, PauliX, PauliX};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.7572222, 0.8229816, 0.9202947};
        CHECK_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {PauliY, PauliY, PauliY};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.5849835, 0.7701511, 0.9126678};
        CHECK_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {PauliZ, PauliZ, PauliZ};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.6577942, 0.4068672, 0.1670374};
        CHECK_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by its name:") {
        std::vector<PrecisionT> variances;
        std::vector<PrecisionT> variances_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> operations_list;

        operations_list = {"PauliX", "PauliX", "PauliX"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.7572222, 0.8229816, 0.9202947};
        CHECK_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {"PauliY", "PauliY", "PauliY"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.5849835, 0.7701511, 0.9126678};
        CHECK_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {"PauliZ", "PauliZ", "PauliZ"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.6577942, 0.4068672, 0.1670374};
        CHECK_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Probabilities", "[Measurements]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    SECTION("1 qubit") {
        auto statevector_data = std::vector<ComplexT>{{1.0, 0.0}, {0.0, 0.0}};
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        Measurements<StateVectorT> Measurer(statevector);

        SECTION("Testing probs()") {
            auto p0 = Measurer.probs();
            statevector.applyOperation("Hadamard", {0}, false);
            auto p1 = Measurer.probs();

            CHECK_THAT(
                p0,
                Catch::Approx(std::vector<PrecisionT>{1.0, 0.0}).margin(1e-7));
            CHECK_THAT(
                p1,
                Catch::Approx(std::vector<PrecisionT>{0.5, 0.5}).margin(1e-7));
        }
        SECTION("Testing probs(NamedObs)") {
            const auto obs1 = Observables::NamedObs<StateVectorT>(
                {"PauliX"}, std::vector<size_t>{0});
            const auto obs2 = Observables::NamedObs<StateVectorT>(
                {"PauliZ"}, std::vector<size_t>{0});
            const auto obs3 = Observables::NamedObs<StateVectorT>(
                {"Hadamard"}, std::vector<size_t>{0});
            auto p0_obs1 = Measurer.probs(obs1);
            auto p0_obs2 = Measurer.probs(obs2);
            auto p0_obs3 = Measurer.probs(obs3);

            CHECK_THAT(
                p0_obs1,
                Catch::Approx(std::vector<PrecisionT>{0.5, 0.5}).margin(1e-7));
            CHECK_THAT(
                p0_obs2,
                Catch::Approx(std::vector<PrecisionT>{1.0, 0.0}).margin(1e-7));
            CHECK_THAT(p0_obs3, Catch::Approx(std::vector<PrecisionT>{
                                                  0.85355339, 0.14644661})
                                    .margin(1e-7));

            statevector.applyOperation("Hadamard", {0}, false);
            auto p1_obs1 = Measurer.probs(obs1);
            auto p1_obs2 = Measurer.probs(obs2);
            auto p1_obs3 = Measurer.probs(obs3);

            CHECK_THAT(
                p1_obs1,
                Catch::Approx(std::vector<PrecisionT>{1.0, 0.0}).margin(1e-7));
            CHECK_THAT(
                p1_obs2,
                Catch::Approx(std::vector<PrecisionT>{0.5, 0.5}).margin(1e-7));
            CHECK_THAT(p1_obs3, Catch::Approx(std::vector<PrecisionT>{
                                                  0.85355339, 0.14644661})
                                    .margin(1e-7));

            statevector.applyOperation("Hadamard", {0}, false);
            auto p2_obs1 = Measurer.probs(obs1);
            auto p2_obs2 = Measurer.probs(obs2);
            auto p2_obs3 = Measurer.probs(obs3);

            CHECK_THAT(
                p0_obs1,
                Catch::Approx(std::vector<PrecisionT>{0.5, 0.5}).margin(1e-7));
            CHECK_THAT(
                p0_obs2,
                Catch::Approx(std::vector<PrecisionT>{1.0, 0.0}).margin(1e-7));
            CHECK_THAT(p0_obs3, Catch::Approx(std::vector<PrecisionT>{
                                                  0.85355339, 0.14644661})
                                    .margin(1e-7));
        }
    }
    SECTION("n-qubit") {
        SECTION("2 qubits") {
            SECTION("|00> state") {
                constexpr std::size_t num_qubits = 2;
                auto statevector_data =
                    std::vector<ComplexT>((1UL << num_qubits), {0.0, 0.0});
                const std::vector<std::size_t> wires{0, 1};
                statevector_data[0] = {1.0, 0.0};

                StateVectorT statevector(statevector_data.data(),
                                         statevector_data.size());
                Measurements<StateVectorT> Measurer(statevector);

                auto p0_full = Measurer.probs();
                auto p0_0 = Measurer.probs({0}, wires);
                auto p0_1 = Measurer.probs({1}, wires);
                auto p0_perm0 = Measurer.probs({1, 0}, wires);

                CHECK_THAT(p0_full, Catch::Approx(std::vector<PrecisionT>{
                                                      1.0, 0.0, 0.0, 0.0})
                                        .margin(1e-7));
                CHECK_THAT(p0_0,
                           Catch::Approx(std::vector<PrecisionT>{1.0, 0.0})
                               .margin(1e-7));
                CHECK_THAT(p0_1,
                           Catch::Approx(std::vector<PrecisionT>{1.0, 0.0})
                               .margin(1e-7));

                CHECK_THAT(p0_perm0, Catch::Approx(std::vector<PrecisionT>{
                                                       1.0, 0.0, 0.0, 0.0})
                                         .margin(1e-7));

                statevector.applyOperation("Hadamard", {0}, false);
                auto p1_full = Measurer.probs();
                auto p1_0 = Measurer.probs({0}, wires);
                auto p1_1 = Measurer.probs({1}, wires);
                auto p1_perm0 = Measurer.probs({1, 0}, wires);

                CHECK_THAT(p1_full, Catch::Approx(std::vector<PrecisionT>{
                                                      0.5, 0.0, 0.5, 0.0})
                                        .margin(1e-7));
                CHECK_THAT(p1_0,
                           Catch::Approx(std::vector<PrecisionT>{0.5, 0.5})
                               .margin(1e-7));
                CHECK_THAT(p1_1,
                           Catch::Approx(std::vector<PrecisionT>{1.0, 0.0})
                               .margin(1e-7));
                CHECK_THAT(p1_perm0, Catch::Approx(std::vector<PrecisionT>{
                                                       0.5, 0.5, 0.0, 0.0})
                                         .margin(1e-7));

                statevector.applyOperation("Hadamard", {1}, false);
                auto p2_full = Measurer.probs();
                auto p2_0 = Measurer.probs({0}, wires);
                auto p2_1 = Measurer.probs({1}, wires);
                auto p2_perm0 = Measurer.probs({1, 0}, wires);

                CHECK_THAT(p2_full, Catch::Approx(std::vector<PrecisionT>{
                                                      0.25, 0.25, 0.25, 0.25})
                                        .margin(1e-7));
                CHECK_THAT(p2_0,
                           Catch::Approx(std::vector<PrecisionT>{0.5, 0.5})
                               .margin(1e-7));
                CHECK_THAT(p2_1,
                           Catch::Approx(std::vector<PrecisionT>{0.5, 0.5})
                               .margin(1e-7));
                CHECK_THAT(p2_perm0, Catch::Approx(std::vector<PrecisionT>{
                                                       0.25, 0.25, 0.25, 0.25})
                                         .margin(1e-7));
            }
        }
        SECTION("3 qubits") {
            SECTION("|000> state") {
                constexpr std::size_t num_qubits = 3;
                auto statevector_data =
                    std::vector<ComplexT>((1UL << num_qubits), {0.0, 0.0});
                const std::vector<std::size_t> wires{0, 1, 2};
                statevector_data[0] = {1.0, 0.0};

                StateVectorT statevector(statevector_data.data(),
                                         statevector_data.size());
                Measurements<StateVectorT> Measurer(statevector);

                auto p0_full = Measurer.probs();
                auto p0_0 = Measurer.probs({0}, wires);
                auto p0_1 = Measurer.probs({1}, wires);
                auto p0_2 = Measurer.probs({2}, wires);

                auto p0_01 = Measurer.probs({0, 1}, wires);
                auto p0_02 = Measurer.probs({0, 2}, wires);
                auto p0_12 = Measurer.probs({1, 2}, wires);

                auto p0_10 = Measurer.probs({1, 0}, wires);
                auto p0_20 = Measurer.probs({2, 0}, wires);
                auto p0_21 = Measurer.probs({2, 1}, wires);

                auto p0_012 = Measurer.probs({0, 1, 2}, wires);
                auto p0_021 = Measurer.probs({0, 2, 1}, wires);
                auto p0_102 = Measurer.probs({1, 0, 2}, wires);
                auto p0_120 = Measurer.probs({1, 2, 0}, wires);
                auto p0_201 = Measurer.probs({2, 0, 1}, wires);
                auto p0_210 = Measurer.probs({2, 1, 0}, wires);

                CHECK_THAT(p0_full, Catch::Approx(std::vector<PrecisionT>{
                                                      1.0, 0.0, 0.0, 0.0, 0.0,
                                                      0.0, 0.0, 0.0})
                                        .margin(1e-7));
                CHECK_THAT(p0_0,
                           Catch::Approx(std::vector<PrecisionT>{1.0, 0.0})
                               .margin(1e-7));
                CHECK_THAT(p0_1,
                           Catch::Approx(std::vector<PrecisionT>{1.0, 0.0})
                               .margin(1e-7));
                CHECK_THAT(p0_2,
                           Catch::Approx(std::vector<PrecisionT>{1.0, 0.0})
                               .margin(1e-7));

                CHECK_THAT(p0_01, Catch::Approx(std::vector<PrecisionT>{
                                                    1.0, 0.0, 0.0, 0.0})
                                      .margin(1e-7));
                CHECK_THAT(p0_02, Catch::Approx(std::vector<PrecisionT>{
                                                    1.0, 0.0, 0.0, 0.0})
                                      .margin(1e-7));
                CHECK_THAT(p0_12, Catch::Approx(std::vector<PrecisionT>{
                                                    1.0, 0.0, 0.0, 0.0})
                                      .margin(1e-7));

                CHECK_THAT(p0_10, Catch::Approx(std::vector<PrecisionT>{
                                                    1.0, 0.0, 0.0, 0.0})
                                      .margin(1e-7));
                CHECK_THAT(p0_20, Catch::Approx(std::vector<PrecisionT>{
                                                    1.0, 0.0, 0.0, 0.0})
                                      .margin(1e-7));
                CHECK_THAT(p0_21, Catch::Approx(std::vector<PrecisionT>{
                                                    1.0, 0.0, 0.0, 0.0})
                                      .margin(1e-7));

                CHECK_THAT(p0_012, Catch::Approx(std::vector<PrecisionT>{
                                                     1.0, 0.0, 0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));
                CHECK_THAT(p0_021, Catch::Approx(std::vector<PrecisionT>{
                                                     1.0, 0.0, 0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));
                CHECK_THAT(p0_102, Catch::Approx(std::vector<PrecisionT>{
                                                     1.0, 0.0, 0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));

                CHECK_THAT(p0_120, Catch::Approx(std::vector<PrecisionT>{
                                                     1.0, 0.0, 0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));
                CHECK_THAT(p0_201, Catch::Approx(std::vector<PrecisionT>{
                                                     1.0, 0.0, 0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));
                CHECK_THAT(p0_210, Catch::Approx(std::vector<PrecisionT>{
                                                     1.0, 0.0, 0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));

                statevector.applyOperation("Hadamard", {0}, false);

                auto p1_full = Measurer.probs();
                auto p1_0 = Measurer.probs({0}, wires);
                auto p1_1 = Measurer.probs({1}, wires);
                auto p1_2 = Measurer.probs({2}, wires);

                auto p1_01 = Measurer.probs({0, 1}, wires);
                auto p1_02 = Measurer.probs({0, 2}, wires);
                auto p1_12 = Measurer.probs({1, 2}, wires);

                auto p1_10 = Measurer.probs({1, 0}, wires);
                auto p1_20 = Measurer.probs({2, 0}, wires);
                auto p1_21 = Measurer.probs({2, 1}, wires);

                auto p1_012 = Measurer.probs({0, 1, 2}, wires);
                auto p1_021 = Measurer.probs({0, 2, 1}, wires);
                auto p1_102 = Measurer.probs({1, 0, 2}, wires);
                auto p1_120 = Measurer.probs({1, 2, 0}, wires);
                auto p1_201 = Measurer.probs({2, 0, 1}, wires);
                auto p1_210 = Measurer.probs({2, 1, 0}, wires);

                CHECK_THAT(p1_full, Catch::Approx(std::vector<PrecisionT>{
                                                      0.5, 0.0, 0.0, 0.0, 0.5,
                                                      0.0, 0.0, 0.0})
                                        .margin(1e-7));
                CHECK_THAT(p1_0,
                           Catch::Approx(std::vector<PrecisionT>{0.5, 0.5})
                               .margin(1e-7));
                CHECK_THAT(p1_1,
                           Catch::Approx(std::vector<PrecisionT>{1.0, 0.0})
                               .margin(1e-7));
                CHECK_THAT(p1_2,
                           Catch::Approx(std::vector<PrecisionT>{1.0, 0.0})
                               .margin(1e-7));

                CHECK_THAT(p1_01, Catch::Approx(std::vector<PrecisionT>{
                                                    0.5, 0.0, 0.5, 0.0})
                                      .margin(1e-7));
                CHECK_THAT(p1_02, Catch::Approx(std::vector<PrecisionT>{
                                                    0.5, 0.0, 0.5, 0.0})
                                      .margin(1e-7));
                CHECK_THAT(p1_12, Catch::Approx(std::vector<PrecisionT>{
                                                    1.0, 0.0, 0.0, 0.0})
                                      .margin(1e-7));

                CHECK_THAT(p1_10, Catch::Approx(std::vector<PrecisionT>{
                                                    0.5, 0.5, 0.0, 0.0})
                                      .margin(1e-7));
                CHECK_THAT(p1_20, Catch::Approx(std::vector<PrecisionT>{
                                                    0.5, 0.5, 0.0, 0.0})
                                      .margin(1e-7));
                CHECK_THAT(p1_21, Catch::Approx(std::vector<PrecisionT>{
                                                    1.0, 0.0, 0.0, 0.0})
                                      .margin(1e-7));

                CHECK_THAT(p1_012, Catch::Approx(std::vector<PrecisionT>{
                                                     0.5, 0.0, 0.0, 0.0, 0.5,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));
                CHECK_THAT(p1_021, Catch::Approx(std::vector<PrecisionT>{
                                                     0.5, 0.0, 0.0, 0.0, 0.5,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));
                CHECK_THAT(p1_102, Catch::Approx(std::vector<PrecisionT>{
                                                     0.5, 0.0, 0.5, 0.0, 0.0,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));

                CHECK_THAT(p1_120, Catch::Approx(std::vector<PrecisionT>{
                                                     0.5, 0.5, 0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));
                CHECK_THAT(p1_201, Catch::Approx(std::vector<PrecisionT>{
                                                     0.5, 0.0, 0.5, 0.0, 0.0,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));
                CHECK_THAT(p1_210, Catch::Approx(std::vector<PrecisionT>{
                                                     0.5, 0.5, 0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0})
                                       .margin(1e-7));
            }
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Sample with Metropolis (Local Kernel)",
                           "[Measurements][MCMC]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;

    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

    // Defining the statevector that will be measured.
    auto statevector_data = createNonTrivialState<StateVectorT>();
    StateVectorT statevector(statevector_data.data(), statevector_data.size());

    // Initializing the measurements class.
    // This object attaches to the statevector allowing several measurements.
    Measurements<StateVectorT> Measurer(statevector);

    std::vector<PrecisionT> expected_probabilities = {
        0.67078706, 0.03062806, 0.0870997,  0.00397696,
        0.17564072, 0.00801973, 0.02280642, 0.00104134};

    size_t num_qubits = 3;
    size_t N = std::pow(2, num_qubits);
    size_t num_samples = 100000;
    size_t num_burnin = 1000;

    std::string kernel = "Local";
    auto &&samples =
        Measurer.generate_samples_metropolis(kernel, num_burnin, num_samples);

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
    std::vector<PrecisionT> probabilities(counts.size());
    for (size_t i = 0; i < counts.size(); i++) {
        probabilities[i] = counts[i] / (PrecisionT)num_samples;
    }

    // compare estimated probabilities to real probabilities
    SECTION("No wires provided:") {
        CHECK_THAT(probabilities,
                   Catch::Approx(expected_probabilities).margin(.05));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Sample with Metropolis (NonZeroRandom Kernel)",
                           "[Measurements][MCMC]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;

    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

    // Defining the State Vector that will be measured.
    auto statevector_data = createNonTrivialState<StateVectorT>();
    StateVectorT statevector(statevector_data.data(), statevector_data.size());

    // Initializing the measurements class.
    // This object attaches to the statevector allowing several measurements.
    Measurements<StateVectorT> Measurer(statevector);

    std::vector<PrecisionT> expected_probabilities = {
        0.67078706, 0.03062806, 0.0870997,  0.00397696,
        0.17564072, 0.00801973, 0.02280642, 0.00104134};

    size_t num_qubits = 3;
    size_t N = std::pow(2, num_qubits);
    size_t num_samples = 100000;
    size_t num_burnin = 1000;

    const std::string kernel = "NonZeroRandom";
    auto &&samples =
        Measurer.generate_samples_metropolis(kernel, num_burnin, num_samples);

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
    std::vector<PrecisionT> probabilities(counts.size());
    for (size_t i = 0; i < counts.size(); i++) {
        probabilities[i] = counts[i] / (PrecisionT)num_samples;
    }

    // compare estimated probabilities to real probabilities
    SECTION("No wires provided:") {
        CHECK_THAT(probabilities,
                   Catch::Approx(expected_probabilities).margin(.05));
    }
}
