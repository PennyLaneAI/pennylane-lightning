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
#include <array>
#include <complex>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "MPSTNCuda.hpp"
#include "MeasurementsTNCuda.hpp"
#include "TNCudaGateCache.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda::Measures;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using namespace Pennylane::LightningTensor::TNCuda;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Probabilities", "[Measures]", float, double) {
    using TensorNetT = MPSTNCuda<TestType>;
    // Probabilities calculated with Pennylane default.qubit:
    std::vector<std::pair<std::vector<std::size_t>, std::vector<TestType>>>
        input = {
            {{2, 1, 0},
             {7.67899385e-01, 9.97094446e-02, 1.54593908e-02, 2.00735578e-03,
              9.97094446e-02, 1.29469740e-02, 2.00735578e-03, 2.60649160e-04}}};

    // Defining the State Vector that will be measured.
    std::size_t bondDim = GENERATE(2, 3, 4, 5);
    std::size_t num_qubits = 3;
    std::size_t maxBondDim = bondDim;

    TensorNetT mps_state{num_qubits, maxBondDim};

    mps_state.applyOperations(
        {{"RX"}, {"RY"}, {"RX"}, {"RY"}, {"RX"}, {"RY"}},
        {{0}, {0}, {1}, {1}, {2}, {2}},
        {{false}, {false}, {false}, {false}, {false}, {false}},
        {{0.5}, {0.5}, {0.2}, {0.2}, {0.5}, {0.5}});
    mps_state.append_mps_final_state();

    auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

    SECTION("Looping over different wire configurations:") {
        for (const auto &term : input) {
            auto probabilities = measure.probs(term.first);
            REQUIRE_THAT(term.second,
                         Catch::Approx(probabilities).margin(1e-6));
        }
    }
}