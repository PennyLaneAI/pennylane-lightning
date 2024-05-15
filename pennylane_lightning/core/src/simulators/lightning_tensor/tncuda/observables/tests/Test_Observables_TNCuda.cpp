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
#include "MPSTNCuda.hpp"
#include "ObservablesTNCuda.hpp"

#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using Pennylane::Util::LightningException;
} // namespace
/// @endcond

TEMPLATE_PRODUCT_TEST_CASE("NamedObs", "[Observables]", (MPSTNCuda),
                           (float, double)) {
    using StateTensorT = TestType;
    using PrecisionT = typename StateTensorT::PrecisionT;
    using NamedObsT = NamedObs<StateTensorT>;

    SECTION("Non-Default constructibility") {
        REQUIRE(!std::is_constructible_v<NamedObsT>);
    }

    SECTION("Constructibility") {
        REQUIRE(std::is_constructible_v<NamedObsT, std::string,
                                        std::vector<std::size_t>>);
    }

    SECTION("Constructibility - optional parameters") {
        REQUIRE(std::is_constructible_v<NamedObsT, std::string,
                                        std::vector<std::size_t>,
                                        std::vector<PrecisionT>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<NamedObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<NamedObsT>);
    }

    SECTION("NamedObs only accepts correct arguments") {
        REQUIRE_THROWS_AS(NamedObsT("PauliX", {}), LightningException);
        REQUIRE_THROWS_AS(NamedObsT("PauliX", {0, 3}), LightningException);

        REQUIRE_THROWS_AS(NamedObsT("RX", {0}), LightningException);
        REQUIRE_THROWS_AS(NamedObsT("RX", {0, 1, 2, 3}), LightningException);
        REQUIRE_THROWS_AS(
            NamedObsT("RX", {0}, std::vector<PrecisionT>{0.3, 0.4}),
            LightningException);
        REQUIRE_NOTHROW(
            NamedObsT("Rot", {0}, std::vector<PrecisionT>{0.3, 0.4, 0.5}));
    }

    SECTION("Throw errors for applyInPlace() method") {
        auto obs = NamedObsT("PauliX", {0});
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = 2;

        StateTensorT mps_state{num_qubits, maxBondDim};

        REQUIRE_THROWS_WITH(
            obs.applyInPlace(mps_state),
            Catch::Matchers::Contains(
                "Lightning.Tensor doesn't support the applyInPlace() method."));
    }

    SECTION("Throw errors for applyInPlaceShots() method") {
        auto obs = NamedObsT("PauliX", {0});
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = 2;

        StateTensorT mps_state{num_qubits, maxBondDim};

        REQUIRE_THROWS_WITH(
            obs.applyInPlaceShots(mps_state),
            Catch::Matchers::Contains(
                "Lightning.Tensor doesn't support the applyInPlace() method."));
    }
}