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
#include "BitUtil.hpp"
#include "Util.hpp"
#include <Kokkos_Core.hpp>

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using KokkosIntVector = Kokkos::View<std::size_t *>;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Util {
/// @cond DEV
constexpr std::size_t one{1};
/// @endcond

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

/**
 * @brief Compute parity and reverse wires for multi-qubit control operations
 *
 * @param num_qubits Number of qubits in the state vector.
 * @param wires List of target wires.
 * @param controlled_wires List of control wires.
 * @return std::pair<KokkosIntVector, KokkosIntVector> Parities and reverse
 * wires for control multi-qubit operations
 */
inline auto reverseWires(const std::size_t num_qubits,
                         const std::vector<std::size_t> &wires,
                         const std::vector<std::size_t> &controlled_wires)
    -> std::pair<KokkosIntVector, KokkosIntVector> {
    KokkosIntVector parity;
    KokkosIntVector rev_wires;

    const std::size_t n_contr = controlled_wires.size();
    const std::size_t n_wires = wires.size();
    const std::size_t nw_tot = n_contr + n_wires;
    std::vector<std::size_t> all_wires;
    all_wires.reserve(nw_tot);
    all_wires.insert(all_wires.begin(), wires.begin(), wires.end());
    all_wires.insert(all_wires.begin() + n_wires, controlled_wires.begin(),
                     controlled_wires.end());

    std::vector<std::size_t> rev_wires_(nw_tot, (num_qubits - 1));
    std::transform(rev_wires_.begin(), rev_wires_.end(), all_wires.rbegin(),
                   rev_wires_.begin(), std::minus<>{});
    const std::vector<std::size_t> parity_ = revWireParity(rev_wires_);

    Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        rev_wires_host(rev_wires_.data(), rev_wires_.size());
    Kokkos::resize(rev_wires, rev_wires_host.size());
    Kokkos::deep_copy(rev_wires, rev_wires_host);

    Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        parity_host(parity_.data(), parity_.size());
    Kokkos::resize(parity, parity_host.size());
    Kokkos::deep_copy(parity, parity_host);

    return {parity, rev_wires};
}

/**
 * @brief Generate bit patterns for multi-qubit operations
 * TODO: parallelize with LK
 *
 * @param wires List of target wires.
 * @param num_qubits
 * @return std::vector List of indices containing control bit patterns
 */
inline auto generateBitPatterns(const std::vector<std::size_t> &wires,
                                const std::size_t num_qubits)
    -> std::vector<std::size_t> {
    std::vector<std::size_t> indices;
    indices.reserve(Pennylane::Util::exp2(wires.size()));
    indices.emplace_back(0);

    for (auto index_it = wires.rbegin(); index_it != wires.rend(); index_it++) {
        const std::size_t value =
            Pennylane::Util::maxDecimalForQubit(*index_it, num_qubits);
        const std::size_t currentSize = indices.size();
        for (std::size_t j = 0; j < currentSize; j++) {
            indices.emplace_back(indices[j] + value);
        }
    }
    return indices;
}

/**
 * @brief Introduce quantum controls in indices generated by
 * generateBitPatterns.
 *
 * @param indices Indices for the operation.
 * @param num_qubits Number of qubits in register.
 * @param controlled_wires Control wires.
 * @param controlled_values Control values (false or true).
 */
inline void controlBitPatterns(std::vector<std::size_t> &indices,
                               const std::size_t num_qubits,
                               const std::vector<std::size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values) {
    if (controlled_wires.empty()) {
        return;
    }
    std::vector<std::size_t> masks(controlled_wires.size());
    std::vector<std::size_t> values(controlled_wires.size());
    for (std::size_t k = 0; k < controlled_wires.size(); k++) {
        const std::size_t rev_wire = num_qubits - 1 - controlled_wires[k];
        masks[k] = ~(one << rev_wire);
        values[k] = static_cast<std::size_t>(controlled_values[k]) << rev_wire;
    }

    std::for_each(indices.begin(), indices.end(),
                  [&masks, &values](std::size_t &i) {
                      for (std::size_t k = 0; k < masks.size(); k++) {
                          i = (i & masks[k]) | values[k];
                      }
                  });
}

/**
 * @brief Compute index offset from parity for control operations
 *
 * @param parity List of parities for control operation.
 * @param k Iteration index for applying control operation.
 * @return std::size_t Index offset.
 */
KOKKOS_INLINE_FUNCTION std::size_t
parity_2_offset(const KokkosIntVector &parity, const std::size_t k) {
    std::size_t offset{0U};
    for (std::size_t i = 0; i < parity.size(); i++) {
        offset |= ((k << i) & parity(i));
    }
    return offset;
}

} // namespace Pennylane::LightningKokkos::Util
