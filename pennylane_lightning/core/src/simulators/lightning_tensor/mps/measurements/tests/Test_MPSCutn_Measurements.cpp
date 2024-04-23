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

#include "MPSCutn.hpp"
#include "MeasurementsMPSCutn.hpp"
#include "cuGateTensorCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningTensor;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::Util;

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::Measures;
using namespace Pennylane::LightningTensor::Observables;
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("[Identity]", "[MPSCutn_Expval]", float, double) {
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    std::vector<size_t> qubitDims(num_qubits, 2);
    DevTag<int> dev_tag{0, 0};

    SECTION("Using expval") {
        MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};
        auto m = Measurements(sv);

        auto ob = NamedObsMPSCutn<TestType>("Identity", {0});
        auto res = m.expval(ob);
        CHECK(std::real(res) == 1.0);
    }
}