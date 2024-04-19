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

#include "cuGateTensorCache.hpp"
#include "cuGates_host.hpp"
#include "cuMPS.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Util;

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace

TEMPLATE_TEST_CASE("cuMPS::applyHadamard", "[cuMPS_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        std::vector<size_t> qubitDims = {2, 2, 2};
        Pennylane::LightningGPU::DevTag<int> dev_tag{0, 0};

        SECTION("Apply using dispatcher") {
            for (std::size_t index = 0; index < num_qubits; index++) {
                cuMPS<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

                CHECK(sv.getDataVector()[0] == cp_t{1, 0});
                sv.applyOperation("Hadamard", {index}, inverse);
                cp_t expected(1.0 / std::sqrt(2), 0);

                CHECK(expected.real() == Approx(sv.getDataVector()[0].real()));
                CHECK(expected.imag() == Approx(sv.getDataVector()[0].imag()));

                CHECK(expected.real() ==
                      Approx(sv.getDataVector()[0b1 << (index)].real()));
                CHECK(expected.imag() ==
                      Approx(sv.getDataVector()[0b1 << (index)].imag()));
            }
        }
    }
}