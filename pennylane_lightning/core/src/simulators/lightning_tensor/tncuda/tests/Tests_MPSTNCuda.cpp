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
#include "MPOTNCuda.hpp"
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

TEMPLATE_TEST_CASE("MPSTNCuda::setIthMPSSite", "[MPSTNCuda]", float, double) {
    SECTION("Set MPS site with wrong site index") {
        const std::size_t num_qubits = 3;
        const std::size_t maxBondDim = 3;
        const std::size_t siteIdx = 3;

        MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim};

        std::vector<std::complex<TestType>> site_data(1, {0.0, 0.0});

        REQUIRE_THROWS_WITH(
            mps_state.updateMPSSiteData(siteIdx, site_data.data(),
                                        site_data.size()),
            Catch::Matchers::Contains(
                "The site index should be less than the number of qubits."));
    }

    SECTION("Set MPS site with wrong site data size") {
        const std::size_t num_qubits = 3;
        const std::size_t maxBondDim = 3;
        const std::size_t siteIdx = 0;

        MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim};

        std::vector<std::complex<TestType>> site_data(1, {0.0, 0.0});

        REQUIRE_THROWS_WITH(
            mps_state.updateMPSSiteData(siteIdx, site_data.data(),
                                        site_data.size()),
            Catch::Matchers::Contains("The length of the host data should "
                                      "match its copy on the device."));
    }

    SECTION("Set MPS sites") {
        const std::size_t num_qubits = 2;
        const std::size_t maxBondDim = 3;

        MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim};

        mps_state.reset(); // Reset the state to zero state

        std::vector<std::complex<TestType>> site0_data(4, {0.0, 0.0}); // MSB
        std::vector<std::complex<TestType>> site1_data(4, {0.0, 0.0}); // LSB

        site0_data[2] = {1.0, 0.0};
        site1_data[1] = {1.0, 0.0};

        mps_state.updateMPSSiteData(0, site0_data.data(), site0_data.size());
        mps_state.updateMPSSiteData(1, site1_data.data(), site1_data.size());

        auto results = mps_state.getDataVector();

        std::vector<std::complex<TestType>> expected_state(
            std::size_t{1} << num_qubits, std::complex<TestType>({0.0, 0.0}));

        expected_state[3] = {1.0, 0.0};

        CHECK(expected_state == Pennylane::Util::approx(results));
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

        auto results = mps_state.getDataVector();

        CHECK(expected_state == Pennylane::Util::approx(results));
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

        auto results = mps_state.getDataVector();

        CHECK(expected_state == Pennylane::Util::approx(results));
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

        auto results = mps_state.getDataVector();

        CHECK(expected_state == Pennylane::Util::approx(results));
    }
}

TEMPLATE_TEST_CASE("MPSTNCuda::getDataVector()", "[MPSTNCuda]", float, double) {
    using cp_t = std::complex<TestType>;
    SECTION("Get zero state") {
        std::size_t num_qubits = 10;
        std::size_t maxBondDim = 2;
        DevTag<int> dev_tag{0, 0};

        MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim, dev_tag};
        std::vector<std::complex<TestType>> expected_state(
            std::size_t{1} << num_qubits, std::complex<TestType>({0.0, 0.0}));

        expected_state[0] = {1.0, 0.0};

        auto results = mps_state.getDataVector();

        CHECK(expected_state == Pennylane::Util::approx(results));
    }

    SECTION("Throw error for getData() on device") {
        std::size_t num_qubits = 50;
        std::size_t maxBondDim = 2;
        DevTag<int> dev_tag{0, 0};

        MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim, dev_tag};

        const std::size_t length = std::size_t{1} << num_qubits;
        std::vector<cp_t> results(1);

        REQUIRE_THROWS_WITH(
            mps_state.getData(results.data(), length),
            Catch::Matchers::Contains(
                "State tensor size exceeds the available GPU memory!"));
    }

    SECTION("Throw wrong size for getData() on device") {
        std::size_t num_qubits = 50;
        std::size_t maxBondDim = 2;
        DevTag<int> dev_tag{0, 0};

        MPSTNCuda<TestType> mps_state{num_qubits, maxBondDim, dev_tag};

        const std::size_t length = 1;
        std::vector<cp_t> results(1);

        REQUIRE_THROWS_WITH(mps_state.getData(results.data(), length),
                            Catch::Matchers::Contains(
                                "The size of the result vector should be equal "
                                "to the dimension of the quantum state."));
    }

    SECTION("Throw error for 0 an 1 qubit circuit") {
        std::size_t num_qubits = GENERATE(0, 1);
        std::size_t maxBondDim = 2;
        DevTag<int> dev_tag{0, 0};

        REQUIRE_THROWS_WITH(
            MPSTNCuda<TestType>(num_qubits, maxBondDim, dev_tag),
            Catch::Matchers::Contains(
                "The number of qubits should be greater than 1."));
    }
}

TEMPLATE_TEST_CASE("MPOTNCuda::getBondDims()", "[MPOTNCuda]", float, double) {
    using cp_t = std::complex<TestType>;
    SECTION("Check if bondDims is correctly set") {
        const std::size_t num_qubits = 3;
        const std::size_t maxBondDim = 128;
        const DevTag<int> dev_tag{0, 0};

        MPSTNCuda<TestType> mps{num_qubits, maxBondDim, dev_tag};

        std::vector<std::vector<cp_t>> tensors; //([2,2,3], [3,2,2,3], [3,2,2])
        const std::vector<std::size_t> wires = {0, 1, 2};
        const std::size_t maxMPOBondDim = 3;

        tensors.emplace_back(std::vector<cp_t>(12, {0.0, 0.0}));
        tensors.emplace_back(std::vector<cp_t>(36, {0.0, 0.0}));
        tensors.emplace_back(std::vector<cp_t>(12, {0.0, 0.0}));

        const auto tensors_const = tensors;

        MPOTNCuda<TestType> mpo{tensors_const,
                                wires,
                                maxMPOBondDim,
                                num_qubits,
                                mps.getTNCudaHandle(),
                                mps.getCudaDataType(),
                                dev_tag};

        auto bondDims = mpo.getBondDims();

        std::vector<std::size_t> expected_bondDims = {maxMPOBondDim,
                                                      maxMPOBondDim};

        CHECK(bondDims == expected_bondDims);
    }
}