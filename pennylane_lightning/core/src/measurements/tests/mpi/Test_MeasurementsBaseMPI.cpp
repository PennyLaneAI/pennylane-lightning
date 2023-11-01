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
#include <catch2/catch.hpp>

#include "TestHelpers.hpp"
/// @cond DEV
namespace {
using Pennylane::Util::isApproxEqual;
} // namespace
/// @endcond

#ifdef _ENABLE_PLGPU
constexpr bool BACKEND_FOUND = true;
#include "MPIManager.hpp"
#include "MeasurementsGPU.hpp"
#include "MeasurementsGPUMPI.hpp"
#include "ObservablesGPU.hpp"
#include "ObservablesGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"
#include "TestHelpersStateVectorsMPI.hpp"
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
using TestStateVectorMPIBackends = Pennylane::Util::TypeList<void>;

template <class StateVector> struct StateVectorMPIToName {};
#endif

template <typename TypeList> void testProbabilities() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Expected results calculated with Pennylane default.qubit:
        std::vector<std::pair<std::vector<size_t>, std::vector<PrecisionT>>>
            input = {// Bit index reodering conducted in the python layer
                     // for L-GPU. Also L-GPU backend doesn't support
                     // out of order wires for probability calculation
                     {{2, 1, 0},
                      {0.67078706, 0.03062806, 0.0870997, 0.00397696,
                       0.17564072, 0.00801973, 0.02280642, 0.00104134}}};

        // Defining the Statevector that will be measured.
        auto statevector_data =
            createNonTrivialState<StateVectorCudaManaged<PrecisionT>>();

        size_t num_qubits = 3;

        MPIManager mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 2);

        size_t mpi_buffersize = 1;

        size_t nGlobalIndexBits =
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
        size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;

        int nDevices = 0;
        cudaGetDeviceCount(&nDevices);
        REQUIRE(nDevices >= 2);
        int deviceId = mpi_manager.getRank() % nDevices;
        cudaSetDevice(deviceId);
        DevTag<int> dt_local(deviceId, 0);
        mpi_manager.Barrier();

        auto sv_data_local = mpi_manager.scatter(statevector_data, 0);

        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();
        // Initializing the measurements class.
        // This object attaches to the statevector allowing several measures.
        MeasurementsMPI<StateVectorT> Measurer(sv);

        std::vector<PrecisionT> expected_prob = {
            0.67078706, 0.03062806, 0.0870997,  0.00397696,
            0.17564072, 0.00801973, 0.02280642, 0.00104134};
        auto prob_local = mpi_manager.scatter(expected_prob, 0);

        DYNAMIC_SECTION("Looping over different wire configurations - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            for (const auto &term : input) {
                auto probabilities = Measurer.probs(term.first);
                REQUIRE_THAT(prob_local,
                             Catch::Approx(probabilities).margin(1e-6));
            }
        }
        testProbabilities<typename TypeList::Next>();
    }
}

TEST_CASE("Probabilities", "[MeasurementsBase]") {
    if constexpr (BACKEND_FOUND) {
        testProbabilities<TestStateVectorMPIBackends>();
    }
}

template <typename TypeList> void testNamedObsExpval() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Defining the State Vector that will be measured.
        auto statevector_data =
            createNonTrivialState<StateVectorCudaManaged<PrecisionT>>();

        size_t num_qubits = 3;

        MPIManager mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 2);

        size_t mpi_buffersize = 1;

        size_t nGlobalIndexBits =
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
        size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;

        int nDevices = 0;
        cudaGetDeviceCount(&nDevices);
        REQUIRE(nDevices >= 2);
        int deviceId = mpi_manager.getRank() % nDevices;
        cudaSetDevice(deviceId);
        DevTag<int> dt_local(deviceId, 0);
        mpi_manager.Barrier();

        auto sv_data_local = mpi_manager.scatter(statevector_data, 0);

        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();
        // Initializing the measurements class.
        // This object attaches to the statevector allowing several measures.
        MeasurementsMPI<StateVectorT> Measurer(sv);

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
                            << StateVectorMPIToName<StateVectorT>::name) {
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObsMPI<StateVectorT> obs(obs_name[ind_obs],
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
        testNamedObsExpval<TestStateVectorMPIBackends>();
    }
}

template <typename TypeList> void testHermitianObsExpval() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using MatrixT = std::vector<ComplexT>;

        // Defining the State Vector that will be measured.
        auto statevector_data =
            createNonTrivialState<StateVectorCudaManaged<PrecisionT>>();

        size_t num_qubits = 3;

        MPIManager mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 2);

        size_t mpi_buffersize = 1;

        size_t nGlobalIndexBits =
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
        size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;

        int nDevices = 0;
        cudaGetDeviceCount(&nDevices);
        REQUIRE(nDevices >= 2);
        int deviceId = mpi_manager.getRank() % nDevices;
        cudaSetDevice(deviceId);
        DevTag<int> dt_local(deviceId, 0);
        mpi_manager.Barrier();

        auto sv_data_local = mpi_manager.scatter(statevector_data, 0);

        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        MeasurementsMPI<StateVectorT> Measurer(sv);

        const PrecisionT theta = M_PI / 2;
        const PrecisionT real_term = std::cos(theta);
        const PrecisionT imag_term = std::sin(theta);

        DYNAMIC_SECTION("Varying wires - 2x2 matrix - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.644217687237691, 0.4794255386042027, 0.29552020666133955};

            MatrixT Hermitian_matrix{real_term, ComplexT{0, imag_term},
                                     ComplexT{0, -imag_term}, real_term};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObsMPI<StateVectorT> obs(Hermitian_matrix,
                                                  wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.expval(obs);
                REQUIRE(expected == Approx(result).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Varying wires - 4x4 matrix - "
                        << StateVectorMPIToName<StateVectorT>::name) {
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
                HermitianObsMPI<StateVectorT> obs(Hermitian_matrix,
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
        testHermitianObsExpval<TestStateVectorMPIBackends>();
    }
}

template <typename TypeList> void testNamedObsVar() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Defining the State Vector that will be measured.
        auto statevector_data =
            createNonTrivialState<StateVectorCudaManaged<PrecisionT>>();

        size_t num_qubits = 3;

        MPIManager mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 2);

        size_t mpi_buffersize = 1;

        size_t nGlobalIndexBits =
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
        size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;

        int nDevices = 0;
        cudaGetDeviceCount(&nDevices);
        REQUIRE(nDevices >= 2);
        int deviceId = mpi_manager.getRank() % nDevices;
        cudaSetDevice(deviceId);
        DevTag<int> dt_local(deviceId, 0);
        mpi_manager.Barrier();

        auto sv_data_local = mpi_manager.scatter(statevector_data, 0);

        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        MeasurementsMPI<StateVectorT> Measurer(sv);

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
                            << StateVectorMPIToName<StateVectorT>::name) {
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObsMPI<StateVectorT> obs(obs_name[ind_obs],
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
        testNamedObsVar<TestStateVectorMPIBackends>();
    }
}

template <typename TypeList> void testHermitianObsVar() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using MatrixT = std::vector<ComplexT>;

        // Defining the State Vector that will be measured.
        auto statevector_data =
            createNonTrivialState<StateVectorCudaManaged<PrecisionT>>();

        size_t num_qubits = 3;

        MPIManager mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 2);

        size_t mpi_buffersize = 1;

        size_t nGlobalIndexBits =
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
        size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;

        int nDevices = 0;
        cudaGetDeviceCount(&nDevices);
        REQUIRE(nDevices >= 2);
        int deviceId = mpi_manager.getRank() % nDevices;
        cudaSetDevice(deviceId);
        DevTag<int> dt_local(deviceId, 0);
        mpi_manager.Barrier();

        auto sv_data_local = mpi_manager.scatter(statevector_data, 0);

        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        MeasurementsMPI<StateVectorT> Measurer(sv);

        const PrecisionT theta = M_PI / 2;
        const PrecisionT real_term = std::cos(theta);
        const PrecisionT imag_term = std::sin(theta);

        DYNAMIC_SECTION("Varying wires - 2x2 matrix - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.5849835714501204, 0.7701511529340699, 0.9126678074548389};

            MatrixT Hermitian_matrix{real_term, ComplexT{0, imag_term},
                                     ComplexT{0, -imag_term}, real_term};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObsMPI<StateVectorT> obs(Hermitian_matrix,
                                                  wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.var(obs);
                REQUIRE(expected == Approx(result).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Varying wires - 4x4 matrix - "
                        << StateVectorMPIToName<StateVectorT>::name) {
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
                HermitianObsMPI<StateVectorT> obs(Hermitian_matrix,
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
        testHermitianObsVar<TestStateVectorMPIBackends>();
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
        auto statevector_data =
            createNonTrivialState<StateVectorCudaManaged<PrecisionT>>();

        size_t num_qubits = 3;

        MPIManager mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 2);

        size_t mpi_buffersize = 1;

        size_t nGlobalIndexBits =
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
        size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;

        int nDevices = 0;
        cudaGetDeviceCount(&nDevices);
        REQUIRE(nDevices >= 2);
        int deviceId = mpi_manager.getRank() % nDevices;
        cudaSetDevice(deviceId);
        DevTag<int> dt_local(deviceId, 0);
        mpi_manager.Barrier();

        auto sv_data_local = mpi_manager.scatter(statevector_data, 0);

        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        MeasurementsMPI<StateVectorT> Measurer(sv);

        std::vector<PrecisionT> expected_probabilities = {
            0.67078706, 0.03062806, 0.0870997,  0.00397696,
            0.17564072, 0.00801973, 0.02280642, 0.00104134};

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
                        << StateVectorMPIToName<StateVectorT>::name) {
            REQUIRE_THAT(probabilities,
                         Catch::Approx(expected_probabilities).margin(.05));
        }
        testSamples<typename TypeList::Next>();
    }
}

TEST_CASE("Samples", "[MeasurementsBase]") {
    if constexpr (BACKEND_FOUND) {
        testSamples<TestStateVectorMPIBackends>();
    }
}