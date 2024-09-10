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
#include <complex>
#include <vector>

#include <catch2/catch.hpp>

#include "TypeList.hpp"

#include "TestHelpersMPSTNCuda.hpp"

using namespace Pennylane::LightningTensor::TNCuda::Util;

/**
 * @file
 *  Tests for functionality defined in the MPSBase class.
 */

template <typename TypeList> void testTensornetBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using MPS_T = typename TypeList::Type;

        const std::size_t num_qubits = 4;
        const std::size_t maxBondDim = 2;
        std::vector<std::size_t> qubitDims = {2, 2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        MPS_T mps_state{num_qubits, maxBondDim, dev_tag};

        DYNAMIC_SECTION("Methods implemented in the base class - "
                        << MPSToName<MPS_T>::name) {
            REQUIRE(mps_state.getNumQubits() == 4);
            REQUIRE(mps_state.getQubitDims() == qubitDims);
        }
        testTensornetBase<typename TypeList::Next>();
    }
}

TEST_CASE("testTensornetBase", "[TensornetBase]") {
    testTensornetBase<TestMPSBackends>();
}
