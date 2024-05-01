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

#include "TestHelpersMPSCutn.hpp"

using namespace Pennylane::LightningTensor::Cutn::Util;

/**
 * @file
 *  Tests for functionality defined in the MPSBase class.
 */

template <typename TypeList> void testMPSBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using MPST = typename TypeList::Type;

        const size_t num_qubits = 4;
        const std::size_t maxBondDim = 2;
        std::vector<size_t> qubitDims = {2, 2, 2, 2};
        DevTag<int> dev_tag{0, 0};

        MPST sv{num_qubits, maxBondDim, dev_tag};

        DYNAMIC_SECTION("Methods implemented in the base class - "
                        << MPSToName<MPST>::name) {
            REQUIRE(sv.getNumQubits() == 4);
            REQUIRE(sv.getMaxBondDim() == 2);
            REQUIRE(sv.getQubitDims() == qubitDims);
        }
        testMPSBase<typename TypeList::Next>();
    }
}

TEST_CASE("testMPSBase", "[MPSBase]") { testMPSBase<TestMPSBackends>(); }