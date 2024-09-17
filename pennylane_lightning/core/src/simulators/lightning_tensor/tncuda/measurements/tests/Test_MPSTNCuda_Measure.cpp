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
#include "TestHelpers.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda::Measures;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Probabilities", "[Measures]", float, double) {
    using TensorNetT = MPSTNCuda<TestType>;

    SECTION("Looping over different wire configurations:") {
        // Probabilities calculated with Pennylane default.qubit:
        std::vector<std::pair<std::vector<std::size_t>, std::vector<TestType>>>
            input = {
                {{0, 1, 2},
                 {0.65473791, 0.08501576, 0.02690407, 0.00349341, 0.19540418,
                  0.02537265, 0.00802942, 0.0010426}},
                {{0, 1}, {0.73975367, 0.03039748, 0.22077683, 0.00907202}},
                {{0, 2}, {0.68164198, 0.08850918, 0.2034336, 0.02641525}},
                {{1, 2}, {0.85014208, 0.11038841, 0.03493349, 0.00453601}},
                {{0}, {0.77015115, 0.22984885}},
                {{1}, {0.9605305, 0.0394695}},
                {{2}, {0.88507558, 0.11492442}}}; // data from default.qubit

        // Defining the State Vector that will be measured.
        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RX"}, {"RY"}, {"RY"}, {"RX"}, {"RY"}},
            {{0}, {0}, {1}, {1}, {2}, {2}},
            {{false}, {false}, {false}, {false}, {false}, {false}},
            {{0.5}, {0.5}, {0.2}, {0.2}, {0.5}, {0.5}});
        mps_state.append_mps_final_state();

        auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

        for (const auto &term : input) {
            auto probabilities = measure.probs(term.first);
            REQUIRE_THAT(term.second,
                         Catch::Approx(probabilities).margin(1e-6));
        }
    }

    SECTION("Test TNCudaOperator ctor failures") {
        // Defining the State Vector that will be measured.
        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations({{"RX"}, {"RY"}}, {{0}, {0}},
                                  {{false}, {false}}, {{0.5}, {0.5}});
        mps_state.append_mps_final_state();

        auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);
        REQUIRE_THROWS_AS(measure.probs({2, 1}), LightningException);
    }

    SECTION("Test excessive projected wires failure") {
        // Defining the State Vector that will be measured.
        std::size_t bondDim = GENERATE(2, 3, 4, 5);
        std::size_t num_qubits = 100;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);
        REQUIRE_THROWS_AS(measure.probs({0, 1, 2, 3}), LightningException);
    }
}

TEMPLATE_TEST_CASE("Samples", "[Measures]", float, double) {
    using TensorNetT = MPSTNCuda<TestType>;

    SECTION("Looping over different wire configurations:") {
        // Probabilities calculated with Pennylane default.qubit:
        std::vector<TestType> expected_probabilities = {
            0.67078706, 0.03062806, 0.0870997,  0.00397696,
            0.17564072, 0.00801973, 0.02280642, 0.00104134};

        // Defining the State Vector that will be measured.
        std::size_t bondDim = GENERATE(4, 5);
        std::size_t num_qubits = 3;
        std::size_t maxBondDim = bondDim;

        TensorNetT mps_state{num_qubits, maxBondDim};

        mps_state.applyOperations(
            {{"RX"}, {"RX"}, {"RY"}, {"RY"}, {"RX"}, {"RY"}},
            {{0}, {0}, {1}, {1}, {2}, {2}},
            {{false}, {false}, {false}, {false}, {false}, {false}},
            {{0.5}, {0.5}, {0.2}, {0.2}, {0.5}, {0.5}});
        mps_state.append_mps_final_state();

        auto measure = MeasurementsTNCuda<TensorNetT>(mps_state);

        std::size_t num_samples = 100000;
        const std::vector<std::size_t> wires = {0, 1, 2};
        auto samples = measure.generate_samples(wires, num_samples);
        auto counts = samples_to_decimal(samples, num_qubits, num_samples);

        // compute estimated probabilities from histogram
        std::vector<TestType> probabilities(counts.size());
        for (std::size_t i = 0; i < counts.size(); i++) {
            probabilities[i] = counts[i] / static_cast<TestType>(num_samples);
        }

        // compare estimated probabilities to real probabilities
        SECTION("No wires provided:") {
            REQUIRE_THAT(probabilities,
                         Catch::Approx(expected_probabilities).margin(.1));
        }
    }
}
