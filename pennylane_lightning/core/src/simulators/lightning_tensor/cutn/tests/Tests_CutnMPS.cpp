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
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "MPSCutn.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Util;
using namespace Pennylane::LightningTensor::Cutn;

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace

TEMPLATE_TEST_CASE("MPSCutn::Constructibility", "[Default Constructibility]",
                   float, double) {
    SECTION("MPSCutn<>") {
        REQUIRE(!std::is_constructible_v<MPSCutn<TestType>()>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("MPSCutn::Constructibility",
                           "[General Constructibility]", (MPSCutn),
                           (float, double)) {
    using MPST = TestType;

    SECTION("MPST<TestType>") { REQUIRE(!std::is_constructible_v<MPST>); }
    SECTION("MPST<TestType> {const size_t, const size_t, DevTag<int> &}") {
        REQUIRE(std::is_constructible_v<MPST, const size_t, const size_t,
                                        DevTag<int> &>);
    }
}

TEMPLATE_TEST_CASE("MPSCutn::SetBasisStates() & reset()", "[MPSCutn]", float,
                   double) {
    std::vector<std::vector<size_t>> basisStates = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
        {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};

    SECTION("Failure for wrong basisState size") {
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = 3;
        std::vector<size_t> basisState = {0, 0, 0, 0};

        MPSCutn<TestType> mps_state{num_qubits, maxBondDim};

        REQUIRE_THROWS_WITH(
            mps_state.setBasisState(basisState),
            Catch::Matchers::Contains("The size of a basis state should be "
                                      "equal to the number of qubits."));
    }

    SECTION("Failure for wrong basisState input") {
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = 3;
        std::vector<size_t> basisState = {0, 0, 2};

        MPSCutn<TestType> mps_state{num_qubits, maxBondDim};

        REQUIRE_THROWS_WITH(
            mps_state.setBasisState(basisState),
            Catch::Matchers::Contains("Please ensure all elements of a basis "
                                      "state should be either 0 or 1."));
    }

    SECTION("Set reset on device with data on the host") {
        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        MPSCutn<TestType> mps_state{num_qubits, bondDim};

        mps_state.reset();

        std::vector<std::complex<TestType>> expected_state(
            size_t{1} << num_qubits, std::complex<TestType>({0.0, 0.0}));

        std::size_t index = 0;

        expected_state[index] = {1.0, 0.0};

        CHECK(mps_state.getMaxBondDim() == maxBondDim);

        CHECK(expected_state ==
              Pennylane::Util::approx(mps_state.getDataVector()));
    }

    SECTION("Test different bondDim and different basisstate") {
        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        for (auto &basisState : basisStates) {
            std::size_t num_qubits = 3;
            std::size_t maxBondDim = bondDim;

            MPSCutn<TestType> mps_state{num_qubits, maxBondDim};

            mps_state.setBasisState(basisState);

            std::vector<std::complex<TestType>> expected_state(
                size_t{1} << num_qubits, std::complex<TestType>({0.0, 0.0}));

            std::size_t index = 0;

            for (size_t i = 0; i < basisState.size(); i++) {
                index += (size_t{1} << (num_qubits - i - 1)) * basisState[i];
            }

            expected_state[index] = {1.0, 0.0};

            CHECK(expected_state ==
                  Pennylane::Util::approx(mps_state.getDataVector()));
        }
    }
}

TEMPLATE_TEST_CASE("MPSCutn::getDataVector()", "[MPSCutn]", float, double) {
    std::size_t num_qubits = 3;
    std::size_t maxBondDim = 2;
    DevTag<int> dev_tag{0, 0};

    MPSCutn<TestType> mps_state{num_qubits, maxBondDim, dev_tag};

    SECTION("Get zero state") {
        std::vector<std::complex<TestType>> expected_state(
            size_t{1} << num_qubits, std::complex<TestType>({0.0, 0.0}));

        expected_state[0] = {1.0, 0.0};

        CHECK(expected_state ==
              Pennylane::Util::approx(mps_state.getDataVector()));
    }
}
