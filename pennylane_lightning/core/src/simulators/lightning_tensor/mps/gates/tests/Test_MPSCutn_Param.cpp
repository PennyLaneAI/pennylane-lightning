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
#include "Gates.hpp"
#include "MPSCutn.hpp"
#include "cuGateTensorCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Util;

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane;
using namespace Pennylane::LightningGPU;
} // namespace

TEMPLATE_TEST_CASE("LightningGPU::applyPhaseShift", "[LightningGPU_Param]",
                   float, double) {
    using cp_t = std::complex<TestType>;
    std::size_t num_qubits = 3;
    std::size_t maxExtent = 2;
    std::vector<size_t> qubitDims = {2, 2, 2};
    DevTag<int> dev_tag{0, 0};

    MPSCutn<TestType> sv{num_qubits, maxExtent, qubitDims, dev_tag};

    // Test using |+++> state
    sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                       {{0}, {1}, {2}}, {{false}, {false}, {false}});

    const std::vector<TestType> angles{0.3, 0.8, 2.4};
    const cp_t coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<cp_t>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(Gates::getPhaseShift<std::complex, TestType>(a));
    }

    std::vector<std::vector<cp_t>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][3], ps_data[0][3], ps_data[0][3], ps_data[0][3]},
        {
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
        },
        {ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3],
         ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    const auto init_state = sv.getDataVector();

    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            MPSCutn<TestType> sv_dispatch(sv);
            sv_dispatch.applyOperation("PhaseShift", {index}, false,
                                       {angles[index]});
            CHECK(sv_dispatch.getDataVector() ==
                  Pennylane::Util::approx(expected_results[index]));
        }
    }
}