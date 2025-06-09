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
#endif

} // namespace Pennylane::LightningKokkos::Util
/// @endcond
