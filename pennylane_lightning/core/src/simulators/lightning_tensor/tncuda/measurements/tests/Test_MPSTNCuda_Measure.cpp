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
}

TEMPLATE_TEST_CASE("Samples", "[Measures]", float, double) {
    using TensorNetT = MPSTNCuda<TestType>;

    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

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

        std::size_t N = std::pow(2, num_qubits);
        std::size_t num_samples = 100000;

        auto samples = measure.generate_samples(num_samples);

        std::vector<std::size_t> counts(N, 0);
        std::vector<std::size_t> samples_decimal(num_samples, 0);

        // convert samples to decimal and then bin them in counts
        for (std::size_t i = 0; i < num_samples; i++) {
            for (std::size_t j = 0; j < num_qubits; j++) {
                if (samples[i * num_qubits + j] != 0) {
                    samples_decimal[i] += twos[(num_qubits - 1 - j)];
                }
            }
            counts[samples_decimal[i]] += 1;
        }

        // compute estimated probabilities from histogram
        std::vector<TestType> probabilities(counts.size());
        for (std::size_t i = 0; i < counts.size(); i++) {
            probabilities[i] = counts[i] / (TestType)num_samples;
        }

        // compare estimated probabilities to real probabilities
        SECTION("No wires provided:") {
            REQUIRE_THAT(probabilities,
                         Catch::Approx(expected_probabilities).margin(.1));
        }
    }
}