// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file
 * Defines utility functions for Bitwise operations.
 */
#pragma once
#include <Kokkos_Core.hpp>

#include "BitUtil.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using KokkosIntVector = Kokkos::View<std::size_t *>;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Util {

inline constexpr std::size_t one{1};

/**
 * @brief Faster log2 when the value is a power of 2.
 *
 * @param val Size of the state vector. Expected to be a power of 2.
 * @return size_t Log2(val), or the state vector's number of qubits.
 */
std::pair<KokkosIntVector, KokkosIntVector>
wires2Parity(const std::size_t num_qubits,
             const std::vector<std::size_t> &wires_) {
    constexpr std::size_t one{1};
    KokkosIntVector parity;
    KokkosIntVector rev_wire_shifts;

    std::vector<std::size_t> rev_wires_(wires_.size());
    std::vector<std::size_t> rev_wire_shifts_(wires_.size());
    for (std::size_t k = 0; k < wires_.size(); k++) {
        rev_wires_[k] = (num_qubits - 1) - wires_[(wires_.size() - 1) - k];
        rev_wire_shifts_[k] = (one << rev_wires_[k]);
    }
    const std::vector<std::size_t> parity_ = revWireParity(rev_wires_);

    Kokkos::View<const size_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        rev_wire_shifts_host(rev_wire_shifts_.data(), rev_wire_shifts_.size());
    Kokkos::resize(rev_wire_shifts, rev_wire_shifts_host.size());
    Kokkos::deep_copy(rev_wire_shifts, rev_wire_shifts_host);

    Kokkos::View<const size_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        parity_host(parity_.data(), parity_.size());
    Kokkos::resize(parity, parity_host.size());
    Kokkos::deep_copy(parity, parity_host);

    return {parity, rev_wire_shifts};
}

} // namespace Pennylane::LightningKokkos::Util