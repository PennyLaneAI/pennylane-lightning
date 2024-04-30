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
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include <iostream>

#include "DevTag.hpp"
#include "MPSCutn.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Util;
using namespace Pennylane::LightningTensor::MPS::Cutn;

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace

TEMPLATE_TEST_CASE("MPSCutn::Constructibility", "[Default Constructibility]",
                   float, double) {
    SECTION("MPSCutn<>") {
        REQUIRE(!std::is_constructible_v<MPSCutn<TestType>()>);
    }
}

TEMPLATE_TEST_CASE("MPSCutn::SetIthStates", "[MPSCutn]", float, double) {
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    std::vector<size_t> qubitDims = {2, 2, 2};
    DevTag<int> dev_tag{0, 0};

    SECTION("Set basis state on device with data on the host") {
        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

        std::vector<size_t> basisState = {0, 1, 1};
        sv.setBasisState(basisState);

        std::vector<std::complex<TestType>> expected_state(
            size_t{1} << num_qubits, std::complex<TestType>({0.0, 0.0}));

        std::size_t index = 0;

        for (size_t i = 0; i < basisState.size(); i++) {
            index += (size_t{1} << (num_qubits - i - 1)) * basisState[i];
        }

        expected_state[index] = {1.0, 0.0};

        CHECK(expected_state == Pennylane::Util::approx(sv.getDataVector()));
    }
}
