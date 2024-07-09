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
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "MPSTNCuda.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Util;
using namespace Pennylane::LightningTensor::TNCuda;

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("MPSTNCuda::Constructibility", "[Default Constructibility]",
                   float, double) {
    SECTION("MPSTNCuda<>") {
        REQUIRE(!std::is_constructible_v<MPSTNCuda<TestType>()>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("MPSTNCuda::Constructibility",
                           "[General Constructibility]", (MPSTNCuda),
                           (float, double)) {
    using MPST = TestType;

    SECTION("MPST<TestType>") { REQUIRE(!std::is_constructible_v<MPST>); }
    SECTION(
        "MPST<TestType> {const std::size_t, const std::size_t, DevTag<int> }") {
        REQUIRE(std::is_constructible_v<MPST, const std::size_t,
                                        const std::size_t, DevTag<int>>);
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::SetBasisStates() & reset()", "[MPSTNCuda]",
                   float, double) {
    std::vector<std::vector<std::size_t>> basisStates = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
        {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};

    SECTION("Failure for wrong basisState size") {
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = 3;
        std::vector<std::size_t> basisState = {0, 0, 0, 0};

        MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim};

        REQUIRE_THROWS_WITH(
            mps_state.setBasisState(basisState),
            Catch::Matchers::Contains("The size of a basis state should be "
                                      "equal to the number of qubits."));
    }

    SECTION("Failure for wrong basisState input") {
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = 3;
        std::vector<std::size_t> basisState = {0, 0, 2};

        MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim};

        REQUIRE_THROWS_WITH(
            mps_state.setBasisState(basisState),
            Catch::Matchers::Contains("Please ensure all elements of a basis "
                                      "state should be either 0 or 1."));
    }

    SECTION("Set reset on device with data on the host") {
        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        MPSTNCuda<TestType> mps_state{num_qubits, bondDim};

        mps_state.reset();

        std::vector<std::complex<TestType>> expected_state(
            std::size_t{1} << num_qubits, std::complex<TestType>({0.0, 0.0}));

        std::size_t index = 0;

        expected_state[index] = {1.0, 0.0};

        CHECK(mps_state.getMaxBondDim() == maxBondDim);

        CHECK(expected_state ==
              Pennylane::Util::approx(mps_state.getDataVector()));
    }

    SECTION("Test different bondDim and different basisstate") {
        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t stateIdx = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim};

        mps_state.setBasisState(basisStates[stateIdx]);

        std::vector<std::complex<TestType>> expected_state(
            std::size_t{1} << num_qubits, std::complex<TestType>({0.0, 0.0}));

        std::size_t index = 0;

        for (std::size_t i = 0; i < basisStates[stateIdx].size(); i++) {
            index += (std::size_t{1} << (num_qubits - i - 1)) *
                     basisStates[stateIdx][i];
        }

        expected_state[index] = {1.0, 0.0};

        CHECK(expected_state ==
              Pennylane::Util::approx(mps_state.getDataVector()));
    }

    SECTION("Test different bondDim and different basisstate & reset()") {
        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t stateIdx = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim};

        mps_state.setBasisState(basisStates[stateIdx]);

        mps_state.reset();

        std::vector<std::complex<TestType>> expected_state(
            std::size_t{1} << num_qubits, std::complex<TestType>({0.0, 0.0}));

        std::size_t index = 0;

        expected_state[index] = {1.0, 0.0};

        CHECK(expected_state ==
              Pennylane::Util::approx(mps_state.getDataVector()));
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::getDataVector()", "[MPSTNCuda]", float, double) {
    std::size_t num_qubits = 10;
    std::size_t maxBondDim = 2;
    DevTag<int> dev_tag{0, 0};

    MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim, dev_tag};

    SECTION("Get zero state") {
        std::vector<std::complex<TestType>> expected_state(
            std::size_t{1} << num_qubits, std::complex<TestType>({0.0, 0.0}));

        expected_state[0] = {1.0, 0.0};

        CHECK(expected_state ==
              Pennylane::Util::approx(mps_state.getDataVector()));
    }
}
