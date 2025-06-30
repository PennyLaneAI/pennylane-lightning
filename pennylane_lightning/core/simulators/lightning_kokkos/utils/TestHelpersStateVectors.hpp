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
#pragma once
/**
 * @file
 * This file defines the necessary functionality to test over LQubit State
 * Vectors.
 */
#if _ENABLE_PLKOKKOS_MPI == 1
#include "StateVectorKokkosMPI.hpp"
#endif
#include "StateVectorKokkos.hpp"
#include "TypeList.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
} // namespace

namespace Pennylane::LightningKokkos::Util {
template <class StateVector> struct StateVectorToName;

template <> struct StateVectorToName<StateVectorKokkos<float>> {
    constexpr static auto name = "StateVectorKokkos<float>";
};
template <> struct StateVectorToName<StateVectorKokkos<double>> {
    constexpr static auto name = "StateVectorKokkos<double>";
};

using TestStateVectorBackends =
    Pennylane::Util::TypeList<StateVectorKokkos<float>,
                              StateVectorKokkos<double>, void>;

#if _ENABLE_PLKOKKOS_MPI == 1
template <typename TestType>
std::pair<StateVectorKokkosMPI<TestType>, StateVectorKokkos<TestType>>
initializeLKTestSV(const std::size_t num_qubits) {
    MPIManagerKokkos mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 4);

    StateVectorKokkosMPI<TestType> sv(mpi_manager, num_qubits);
    StateVectorKokkos<TestType> sv_ref{num_qubits};

    // Set the reference data
    std::vector<Kokkos::complex<TestType>> init_sv(exp2(num_qubits),
                                                   {0.0, 0.0});
    for (std::size_t i = 0; i < init_sv.size(); i++) {
        init_sv[i] = i;
    }

    // Set initial data
    std::size_t block_size = sv.getLocalBlockSize();
    std::vector<Kokkos::complex<TestType>> init_subsv(block_size, {0.0, 0.0});
    for (std::size_t element = 0; element < block_size; element++) {
        init_subsv[element] =
            init_sv[mpi_manager.getRank() * block_size + element];
    }
    sv.updateData(init_subsv);
    sv_ref.updateData(init_sv);

    return std::make_pair(std::move(sv), std::move(sv_ref));
}

template <typename TestType>
void applyNonTrivialOperations(const std::size_t num_qubits,
                               StateVectorKokkosMPI<TestType> &sv,
                               StateVectorKokkos<TestType> &sv_ref) {
    for (std::size_t i = 0; i < num_qubits; i++) {
        sv.applyOperation("PauliX", {i});
        sv_ref.applyOperation("PauliX", {i});
        sv.applyOperation("PauliY", {i});
        sv_ref.applyOperation("PauliY", {i});
        sv.applyOperation("PauliZ", {i});
        sv_ref.applyOperation("PauliZ", {i});
        sv.applyOperation("Hadamard", {i});
        sv_ref.applyOperation("Hadamard", {i});
        sv.applyOperation("RX", {i}, false, {0.5});
        sv_ref.applyOperation("RX", {i}, false, {0.5});
        sv.applyOperation("RY", {i}, true, {0.3});
        sv_ref.applyOperation("RY", {i}, true, {0.3});
        sv.applyOperation("RZ", {i}, true, {0.2});
        sv_ref.applyOperation("RZ", {i}, true, {0.2});
    }
    sv.applyOperation("CNOT", {1, 3});
    sv_ref.applyOperation("CNOT", {1, 3});
}

/**
 * @brief Get combined in-order data vector to root rank
 */
template <typename TestType>
auto getFullDataVector(StateVectorKokkosMPI<TestType> &sv,
                       const std::size_t root = 0)
    -> std::vector<Kokkos::complex<TestType>> {
    sv.reorderGlobalLocalWires();
    sv.reorderLocalWires();
    auto mpi_manager = sv.getMPIManager();
    std::vector<Kokkos::complex<TestType>> data(
        (mpi_manager.getRank() == root) ? sv.getLength() : 0);

    std::vector<Kokkos::complex<TestType>> local(sv.getLocalBlockSize());
    sv.DeviceToHost(local.data(), local.size());

    auto global_wires = sv.getGlobalWires();

    std::vector<int> displacements(mpi_manager.getSize(), 0);
    for (std::size_t rank = 0; rank < mpi_manager.getSize(); rank++) {
        for (std::size_t i = 0; i < sv.getNumGlobalWires(); i++) {
            std::size_t temp =
                ((sv.getGlobalIndexFromMPIRank(rank) >>
                  (sv.getNumGlobalWires() - 1 - i)) &
                 1)
                << (sv.getNumGlobalWires() - 1 - global_wires[i]);
            displacements[rank] += temp;
        }
        displacements[rank] *= local.size();
    }
    mpi_manager.GatherV(local, data, root, displacements);
    return data;
}

#endif

} // namespace Pennylane::LightningKokkos::Util
/// @endcond
