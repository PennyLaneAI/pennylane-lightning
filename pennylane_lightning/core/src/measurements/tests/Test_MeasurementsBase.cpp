// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using Pennylane::Util::isApproxEqual;
} // namespace
/// @endcond

#ifdef _ENABLE_PLQUBIT
constexpr bool BACKEND_FOUND = true;

#include "MeasurementsLQubit.hpp"
#include "ObservablesLQubit.hpp"
#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName
#include "TestHelpersWires.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit::Measures;
using namespace Pennylane::LightningQubit::Observables;
using namespace Pennylane::LightningQubit::Util;
} // namespace
/// @endcond

#elif _ENABLE_PLKOKKOS == 1
constexpr bool BACKEND_FOUND = true;

#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName
#include "TestHelpersWires.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::LightningKokkos::Util;
} // namespace
  /// @endcond

#elif _ENABLE_PLGPU == 1
constexpr bool BACKEND_FOUND = true;
#include "MeasurementsGPU.hpp"
#include "ObservablesGPU.hpp"
#include "TestHelpersStateVectors.hpp"
#include "TestHelpersWires.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Observables;
} // namespace
  /// @endcond

#else
constexpr bool BACKEND_FOUND = false;
using TestStateVectorBackends = Pennylane::Util::TypeList<void>;

template <class StateVector> struct StateVectorToName {};
#endif

template <typename TypeList> void testProbabilities() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Expected results calculated with Pennylane default.qubit:
        std::vector<std::pair<std::vector<size_t>, std::vector<PrecisionT>>>
            input = {
#ifdef _ENABLE_PLGPU
                // Bit index reodering conducted in the python layer
                // for L-GPU. Also L-GPU backend doesn't support
                // out of order wires for probability calculation
                {{2, 1, 0},
                 {0.67078706, 0.03062806, 0.0870997, 0.00397696, 0.17564072,
                  0.00801973, 0.02280642, 0.00104134}}
#else
                {{0, 1, 2},
                 {0.67078706, 0.03062806, 0.0870997, 0.00397696, 0.17564072,
                  0.00801973, 0.02280642, 0.00104134}},
                {{0, 2, 1},
                 {0.67078706, 0.0870997, 0.03062806, 0.00397696, 0.17564072,
                  0.02280642, 0.00801973, 0.00104134}},
                {{1, 0, 2},
                 {0.67078706, 0.03062806, 0.17564072, 0.00801973, 0.0870997,
                  0.00397696, 0.02280642, 0.00104134}},
                {{1, 2, 0},
                 {0.67078706, 0.0870997, 0.17564072, 0.02280642, 0.03062806,
                  0.00397696, 0.00801973, 0.00104134}},
                {{2, 0, 1},
                 {0.67078706, 0.17564072, 0.03062806, 0.00801973, 0.0870997,
                  0.02280642, 0.00397696, 0.00104134}},
                {{2, 1, 0},
                 {0.67078706, 0.17564072, 0.0870997, 0.02280642, 0.03062806,
                  0.00801973, 0.00397696, 0.00104134}},
                {{0, 1}, {0.70141512, 0.09107666, 0.18366045, 0.02384776}},
                {{0, 2}, {0.75788676, 0.03460502, 0.19844714, 0.00906107}},
                {{1, 2}, {0.84642778, 0.0386478, 0.10990612, 0.0050183}},
                {{2, 1}, {0.84642778, 0.10990612, 0.0386478, 0.0050183}},
                {{0}, {0.79249179, 0.20750821}},
                {{1}, {0.88507558, 0.11492442}},
                {{2}, {0.9563339, 0.0436661}}
#endif
            };

        // Defining the Statevector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measurements class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<PrecisionT> probabilities;

        DYNAMIC_SECTION("Looping over different wire configurations - "
                        << StateVectorToName<StateVectorT>::name) {
            for (const auto &term : input) {
                probabilities = Measurer.probs(term.first);
                REQUIRE_THAT(term.second,
                             Catch::Approx(probabilities).margin(1e-6));
            }
        }
        testProbabilities<typename TypeList::Next>();
    }
}

TEST_CASE("Probabilities", "[MeasurementsBase]") {
    if constexpr (BACKEND_FOUND) {
        testProbabilities<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testNamedObsExpval() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> obs_name = {"PauliX", "PauliY", "PauliZ"};
        // Expected results calculated with Pennylane default.qubit:
        std::vector<std::vector<PrecisionT>> exp_values_ref = {
            {0.49272486, 0.42073549, 0.28232124},
            {-0.64421768, -0.47942553, -0.29552020},
            {0.58498357, 0.77015115, 0.91266780}};

        for (size_t ind_obs = 0; ind_obs < obs_name.size(); ind_obs++) {
            DYNAMIC_SECTION(obs_name[ind_obs]
                            << " - Varying wires"
                            << StateVectorToName<StateVectorT>::name) {
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObs<StateVectorT> obs(obs_name[ind_obs],
                                               wires_list[ind_wires]);
                    PrecisionT expected = exp_values_ref[ind_obs][ind_wires];
                    PrecisionT result = Measurer.expval(obs);
                    REQUIRE(expected == Approx(result).margin(1e-6));
                }
            }
        }
        testNamedObsExpval<typename TypeList::Next>();
    }
}

TEST_CASE("Expval - NamedObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testNamedObsExpval<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testHermitianObsExpval() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using MatrixT = std::vector<ComplexT>;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        const PrecisionT theta = M_PI / 2;
        const PrecisionT real_term = std::cos(theta);
        const PrecisionT imag_term = std::sin(theta);

        DYNAMIC_SECTION("Varying wires - 2x2 matrix - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.644217687237691, 0.4794255386042027, 0.29552020666133955};

            MatrixT Hermitian_matrix{real_term, ComplexT{0, imag_term},
                                     ComplexT{0, -imag_term}, real_term};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObs<StateVectorT> obs(Hermitian_matrix,
                                               wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.expval(obs);
                REQUIRE(expected == Approx(result).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Varying wires - 4x4 matrix - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {
                {0, 1}, {0, 2}, {1, 2}, {2, 1}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.5874490024807637, 0.44170554255359035, 0.3764821318486682,
                0.5021569932};

            MatrixT Hermitian_matrix(16);
            Hermitian_matrix[0] = real_term;
            Hermitian_matrix[1] = ComplexT{0, imag_term};
            Hermitian_matrix[4] = ComplexT{0, -imag_term};
            Hermitian_matrix[5] = real_term;
            Hermitian_matrix[10] = ComplexT{1.0, 0};
            Hermitian_matrix[15] = ComplexT{1.0, 0};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObs<StateVectorT> obs(Hermitian_matrix,
                                               wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.expval(obs);
                REQUIRE(expected == Approx(result).margin(1e-6));
            }
        }

        testHermitianObsExpval<typename TypeList::Next>();
    }
}

TEST_CASE("Expval - HermitianObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testHermitianObsExpval<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testNamedObsVar() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> obs_name = {"PauliX", "PauliY", "PauliZ"};
        // Expected results calculated with Pennylane default.qubit:
        std::vector<std::vector<PrecisionT>> exp_values_ref = {
            {0.7572222, 0.8229816, 0.9202947},
            {0.5849835, 0.7701511, 0.9126678},
            {0.6577942, 0.4068672, 0.1670374}};

        for (size_t ind_obs = 0; ind_obs < obs_name.size(); ind_obs++) {
            DYNAMIC_SECTION(obs_name[ind_obs]
                            << " - Varying wires"
                            << StateVectorToName<StateVectorT>::name) {
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObs<StateVectorT> obs(obs_name[ind_obs],
                                               wires_list[ind_wires]);
                    PrecisionT expected = exp_values_ref[ind_obs][ind_wires];
                    PrecisionT result = Measurer.var(obs);
                    REQUIRE(expected == Approx(result).margin(1e-6));
                }
            }
        }
        testNamedObsVar<typename TypeList::Next>();
    }
}

TEST_CASE("Var - NamedObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testNamedObsVar<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testHermitianObsVar() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using MatrixT = std::vector<ComplexT>;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        const PrecisionT theta = M_PI / 2;
        const PrecisionT real_term = std::cos(theta);
        const PrecisionT imag_term = std::sin(theta);

        DYNAMIC_SECTION("Varying wires - 2x2 matrix - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.5849835714501204, 0.7701511529340699, 0.9126678074548389};

            MatrixT Hermitian_matrix{real_term, ComplexT{0, imag_term},
                                     ComplexT{0, -imag_term}, real_term};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObs<StateVectorT> obs(Hermitian_matrix,
                                               wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.var(obs);
                REQUIRE(expected == Approx(result).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Varying wires - 4x4 matrix - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {
                {0, 1}, {0, 2}, {1, 2}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.6549036423585175, 0.8048961865516002, 0.8582611741038356};

            MatrixT Hermitian_matrix(16);
            Hermitian_matrix[0] = real_term;
            Hermitian_matrix[1] = ComplexT{0, imag_term};
            Hermitian_matrix[4] = ComplexT{0, -imag_term};
            Hermitian_matrix[5] = real_term;
            Hermitian_matrix[10] = ComplexT{1.0, 0};
            Hermitian_matrix[15] = ComplexT{1.0, 0};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObs<StateVectorT> obs(Hermitian_matrix,
                                               wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.var(obs);
                REQUIRE(expected == Approx(result).margin(1e-6));
            }
        }

        testHermitianObsVar<typename TypeList::Next>();
    }
}

TEST_CASE("Var - HermitianObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testHermitianObsVar<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testSamples() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        constexpr size_t twos[] = {
            1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
            1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
            1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
            1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
            1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
            1U << 30U, 1U << 31U};

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measurements class.
        // This object attaches to the statevector allowing several
        // measurements.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<PrecisionT> expected_probabilities = {
            0.67078706, 0.03062806, 0.0870997,  0.00397696,
            0.17564072, 0.00801973, 0.02280642, 0.00104134};

        size_t num_qubits = 3;
        size_t N = std::pow(2, num_qubits);
        size_t num_samples = 100000;
        auto &&samples = Measurer.generate_samples(num_samples);

        std::vector<size_t> counts(N, 0);
        std::vector<size_t> samples_decimal(num_samples, 0);

        // convert samples to decimal and then bin them in counts
        for (size_t i = 0; i < num_samples; i++) {
            for (size_t j = 0; j < num_qubits; j++) {
                if (samples[i * num_qubits + j] != 0) {
                    samples_decimal[i] += twos[(num_qubits - 1 - j)];
                }
            }
            counts[samples_decimal[i]] += 1;
        }

        // compute estimated probabilities from histogram
        std::vector<PrecisionT> probabilities(counts.size());
        for (size_t i = 0; i < counts.size(); i++) {
            probabilities[i] = counts[i] / (PrecisionT)num_samples;
        }

        DYNAMIC_SECTION("No wires provided - "
                        << StateVectorToName<StateVectorT>::name) {
            REQUIRE_THAT(probabilities,
                         Catch::Approx(expected_probabilities).margin(.05));
        }
        testSamples<typename TypeList::Next>();
    }
}

TEST_CASE("Samples", "[MeasurementsBase]") {
    if constexpr (BACKEND_FOUND) {
        testSamples<TestStateVectorBackends>();
    }
}