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
#include "MPSTNCuda.hpp"
#include "TNCudaGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Util;

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace

TEMPLATE_TEST_CASE("MPSTNCuda::applyHadamard", "[MPSTNCuda_Nonparam]", float,
                   double) {
    const bool inverse = GENERATE(true, false);
    {
        using cp_t = std::complex<TestType>;
        std::size_t num_qubits = 3;
        std::size_t maxExtent = 2;
        DevTag<int> dev_tag{0, 0};

        SECTION("Apply using dispatcher") {
            const std::size_t index = GENERATE(0, 1, 2);
            MPSTNCuda<TestType> sv{num_qubits, maxExtent, dev_tag};

            sv.appendGateTensorOperator("Hadamard", {index}, inverse);
            cp_t expected(1.0 / std::sqrt(2), 0);

            auto results = sv.getDataVector();

            CHECK(expected.real() ==
                  Approx(results[0b1 << ((num_qubits - 1 - index))].real()));
            CHECK(expected.imag() ==
                  Approx(results[0b1 << ((num_qubits - index - 1))].imag()));
        }
    }
}
