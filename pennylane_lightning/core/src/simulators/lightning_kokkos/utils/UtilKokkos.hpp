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

constexpr std::size_t one{1};

/**
 * @brief Copy the content of a Kokkos view to an `std::vector`.
 *
 * @tparam T View data type.
 * @param view Kokkos view.
 * @return `std::vector<T>` containing a copy of the view.
 */
template <typename T>
inline auto view2vector(const Kokkos::View<T *> view) -> std::vector<T> {
    using UnmanagedHostView =
        Kokkos::View<T *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    std::vector<T> vec(view.size());
    Kokkos::deep_copy(UnmanagedHostView(vec.data(), vec.size()), view);
    return vec;
}

/**
 * @brief Copy the content of a pointer to a Kokkos view.
 *
 * @tparam T Pointer data type.
 * @param vec Pointer.
 * @return Kokkos view pointing to a copy of the pointer.
 */
template <typename T>
inline auto pointer2view(const T *vec, const std::size_t num)
    -> Kokkos::View<T *> {
    using UnmanagedView = Kokkos::View<const T *, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::View<T *> view("vec", num);
    Kokkos::deep_copy(view, UnmanagedView(vec, num));
    return view;
}

/**
 * @brief Copy the content of an `std::vector` to a Kokkos view.
 *
 * @tparam T Vector data type.
 * @param vec Vector.
 * @return Kokkos view pointing to a copy of the vector.
 */
template <typename T>
inline auto vector2view(const std::vector<T> &vec) -> Kokkos::View<T *> {
    return pointer2view(vec.data(), vec.size());
}

/**
 * @brief Compute the parities and shifts for multi-qubit operations.
 *
 * @param num_qubits Number of qubits in the state vector.
 * @param wires List of target wires.
 * @return std::pair<KokkosIntVector, KokkosIntVector> Parities and shifts for
 * multi-qubit operations.
 */
inline auto wires2Parity(const std::size_t num_qubits,
                         const std::vector<std::size_t> &wires)
    -> std::pair<KokkosIntVector, KokkosIntVector> {
    KokkosIntVector parity;
    KokkosIntVector rev_wire_shifts;

    std::vector<std::size_t> rev_wires_(wires.size());
    std::vector<std::size_t> rev_wire_shifts_(wires.size());
    for (std::size_t k = 0; k < wires.size(); k++) {
        rev_wires_[k] = (num_qubits - 1) - wires[(wires.size() - 1) - k];
        rev_wire_shifts_[k] = (one << rev_wires_[k]);
    }
    const std::vector<std::size_t> parity_ = revWireParity(rev_wires_);

    Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        rev_wire_shifts_host(rev_wire_shifts_.data(), rev_wire_shifts_.size());
    Kokkos::resize(rev_wire_shifts, rev_wire_shifts_host.size());
    Kokkos::deep_copy(rev_wire_shifts, rev_wire_shifts_host);

    Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        parity_host(parity_.data(), parity_.size());
    Kokkos::resize(parity, parity_host.size());
    Kokkos::deep_copy(parity, parity_host);

    return {parity, rev_wire_shifts};
}

} // namespace Pennylane::LightningKokkos::Util