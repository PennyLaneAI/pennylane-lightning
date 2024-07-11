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
 * Defines a class for the measurement of observables in quantum states
 * represented by a Lightning Qubit StateVector class.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>
#include <stack>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "LinearAlgebra.hpp"
#include "MeasurementsBase.hpp"
#include "NDPermuter.hpp"
#include "Observables.hpp"
#include "SparseLinAlg.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TransitionKernels.hpp"
#include "Util.hpp" //transpose_state_tensor, sorting_indices

/// @cond DEV
namespace {
using namespace Pennylane::Measures;
using namespace Pennylane::Observables;
using Pennylane::LightningQubit::StateVectorLQubitManaged;
using Pennylane::LightningQubit::Util::innerProdC;
namespace PUtil = Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Measures {
/**
 * @brief Observable's Measurement Class.
 *
 * This class couples with a statevector to performs measurements.
 * Observables are defined by its operator(matrix), the observable class,
 * or through a string-based function dispatch.
 *
 * @tparam StateVectorT type of the statevector to be measured.
 */
template <class StateVectorT>
class Measurements final
    : public MeasurementsBase<StateVectorT, Measurements<StateVectorT>> {
  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using BaseType = MeasurementsBase<StateVectorT, Measurements<StateVectorT>>;

  public:
    explicit Measurements(const StateVectorT &statevector)
        : BaseType{statevector} {};

    /**
     * @brief Probabilities of each computational basis state.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    auto probs() -> std::vector<PrecisionT> {
        const ComplexT *arr_data = this->_statevector.getData();
        std::vector<PrecisionT> basis_probs(this->_statevector.getLength(), 0);

        std::transform(
            arr_data, arr_data + this->_statevector.getLength(),
            basis_probs.begin(),
            [](const ComplexT &z) -> PrecisionT { return std::norm(z); });
        return basis_probs;
    };

#define PROBS_SPECIAL_CASE(n)                                                  \
    if (n_wires == n) {                                                        \
        return probs_core<n>(arr_data, num_qubits, wires);                     \
    }

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    auto
    probs(const std::vector<std::size_t> &wires,
          [[maybe_unused]] const std::vector<std::size_t> &device_wires = {})
        -> std::vector<PrecisionT> {
        constexpr std::size_t one{1};

        // Determining index that would sort the vector.
        // This information is needed later.
        const std::size_t n_wires = wires.size();
        const std::size_t num_qubits = this->_statevector.getNumQubits();

        // If all wires are requested, dispatch to `this->probs()`
        bool is_all_wires = n_wires == num_qubits;
        for (std::size_t k = 0; k < n_wires; k++) {
            if (!is_all_wires) {
                break;
            }
            is_all_wires = wires[k] == k;
        }
        if (is_all_wires) {
            return this->probs();
        }

        const ComplexT *arr_data = this->_statevector.getData();
        PROBS_SPECIAL_CASE(1)
        PROBS_SPECIAL_CASE(2)
        PROBS_SPECIAL_CASE(3)
        PROBS_SPECIAL_CASE(4)
        PROBS_SPECIAL_CASE(5)
        PROBS_SPECIAL_CASE(6)
        PROBS_SPECIAL_CASE(7)
        PROBS_SPECIAL_CASE(8)
        // PROBS_SPECIAL_CASE(9)
        // PROBS_SPECIAL_CASE(10)
        // Determining probabilities for the sorted wires.
        std::vector<std::size_t> rev_wires(n_wires);
        std::vector<std::size_t> rev_wire_shifts(n_wires);
        for (std::size_t k = 0; k < n_wires; k++) {
            rev_wires[k] = (num_qubits - 1) - wires[k];
            rev_wire_shifts[k] = one << rev_wires[k];
        }
        const std::vector<std::size_t> parity =
            Pennylane::Util::revWireParity(rev_wires);
        std::vector<PrecisionT> probabilities(PUtil::exp2(n_wires), 0);
        for (std::size_t k = 0; k < exp2(num_qubits - n_wires); k++) {
            const auto indices = parity2indices(k, parity, rev_wire_shifts);
            for (std::size_t i = 0; i < probabilities.size(); i++) {
                probabilities[i] += std::norm(arr_data[indices[i]]);
            }
        }
        return probabilities;
    }

    auto parity2indices(const std::size_t k, std::vector<std::size_t> parity,
                        std::vector<std::size_t> rev_wire_shifts)
        -> std::vector<std::size_t> {
        constexpr std::size_t one{1};
        const std::size_t dim = one << rev_wire_shifts.size();
        std::vector<std::size_t> indices(dim);
        std::size_t idx = (k & parity[0]);
        for (std::size_t i = 1; i < parity.size(); i++) {
            idx |= ((k << i) & parity[i]);
        }
        indices[0] = idx;
        for (std::size_t inner_idx = 1; inner_idx < dim; inner_idx++) {
            idx = indices[0];
            for (std::size_t i = 0; i < rev_wire_shifts.size(); i++) {
                if ((inner_idx & (one << i)) != 0) {
                    idx |= rev_wire_shifts[i];
                }
            }
            indices[inner_idx] = idx;
        }
        return indices;
    }

#define PROBS_CORE_DECLARE_NW(n)                                               \
    std::size_t rev_wires_##n;                                                 \
    if constexpr (n_wires > n) {                                               \
        rev_wires_##n = rev_wires[n];                                          \
    }
#define PROBS_CORE_DECLARE_P(n)                                                \
    std::size_t parity_##n;                                                    \
    if constexpr (n_wires >= n) {                                              \
        parity_##n = parity[n];                                                \
    }

#define PROBS_CORE_DEF_I0 ((k << 0U) & parity_0) | ((k << 1U) & parity_1)
#define PROBS_CORE_DEF_I00 PROBS_CORE_DEF_I0 | ((k << 2U) & parity_2)
#define PROBS_CORE_DEF_I000 PROBS_CORE_DEF_I00 | ((k << 3U) & parity_3)
#define PROBS_CORE_DEF_I0000 PROBS_CORE_DEF_I000 | ((k << 4U) & parity_4)
#define PROBS_CORE_DEF_I00000 PROBS_CORE_DEF_I0000 | ((k << 5U) & parity_5)
#define PROBS_CORE_DEF_I000000 PROBS_CORE_DEF_I00000 | ((k << 6U) & parity_6)
#define PROBS_CORE_DEF_I0000000 PROBS_CORE_DEF_I000000 | ((k << 7U) & parity_7)
#define PROBS_CORE_DEF_I00000000                                               \
    PROBS_CORE_DEF_I0000000 | ((k << 8U) & parity_8)
#define PROBS_CORE_DEF_I000000000                                              \
    PROBS_CORE_DEF_I00000000 | ((k << 9U) & parity_9)
#define PROBS_CORE_DEF_I0000000000                                             \
    PROBS_CORE_DEF_I000000000 | ((k << 10U) & parity_10)

#define PROBS_CORE_DEF_IF1(var, x0)                                            \
    if constexpr (x0 == 1) {                                                   \
        var##x0 |= (one << rev_wires_0);                                       \
    }
#define PROBS_CORE_DEF_IF2(var, x0, x1)                                        \
    if constexpr (x0 == 1) {                                                   \
        var##x0##x1 |= (one << rev_wires_1);                                   \
    }                                                                          \
    PROBS_CORE_DEF_IF1(var##x0, x1)
#define PROBS_CORE_DEF_IF3(var, x0, x1, x2)                                    \
    if constexpr (x0 == 1) {                                                   \
        var##x0##x1##x2 |= (one << rev_wires_2);                               \
    }                                                                          \
    PROBS_CORE_DEF_IF2(var##x0, x1, x2)
#define PROBS_CORE_DEF_IF4(var, x0, x1, x2, x3)                                \
    if constexpr (x0 == 1) {                                                   \
        var##x0##x1##x2##x3 |= (one << rev_wires_3);                           \
    }                                                                          \
    PROBS_CORE_DEF_IF3(var##x0, x1, x2, x3)
#define PROBS_CORE_DEF_IF5(var, x0, x1, x2, x3, x4)                            \
    if constexpr (x0 == 1) {                                                   \
        var##x0##x1##x2##x3##x4 |= (one << rev_wires_4);                       \
    }                                                                          \
    PROBS_CORE_DEF_IF4(var##x0, x1, x2, x3, x4)
#define PROBS_CORE_DEF_IF6(var, x0, x1, x2, x3, x4, x5)                        \
    if constexpr (x0 == 1) {                                                   \
        var##x0##x1##x2##x3##x4##x5 |= (one << rev_wires_5);                   \
    }                                                                          \
    PROBS_CORE_DEF_IF5(var##x0, x1, x2, x3, x4, x5)
#define PROBS_CORE_DEF_IF7(var, x0, x1, x2, x3, x4, x5, x6)                    \
    if constexpr (x0 == 1) {                                                   \
        var##x0##x1##x2##x3##x4##x5##x6 |= (one << rev_wires_6);               \
    }                                                                          \
    PROBS_CORE_DEF_IF6(var##x0, x1, x2, x3, x4, x5, x6)
#define PROBS_CORE_DEF_IF8(var, x0, x1, x2, x3, x4, x5, x6, x7)                \
    if constexpr (x0 == 1) {                                                   \
        var##x0##x1##x2##x3##x4##x5##x6##x7 |= (one << rev_wires_7);           \
    }                                                                          \
    PROBS_CORE_DEF_IF7(var##x0, x1, x2, x3, x4, x5, x6, x7)
#define PROBS_CORE_DEF_IF9(var, x0, x1, x2, x3, x4, x5, x6, x7, x8)            \
    if constexpr (x0 == 1) {                                                   \
        var##x0##x1##x2##x3##x4##x5##x6##x7##x8 |= (one << rev_wires_8);       \
    }                                                                          \
    PROBS_CORE_DEF_IF8(var##x0, x1, x2, x3, x4, x5, x6, x7, x8)
#define PROBS_CORE_DEF_IF10(var, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)       \
    if constexpr (x0 == 1) {                                                   \
        var##x0##x1##x2##x3##x4##x5##x6##x7##x8##x9 |= (one << rev_wires_9);   \
    }                                                                          \
    PROBS_CORE_DEF_IF9(var##x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)

#define PROBS_CORE_DEF_Ix(x0)                                                  \
    {                                                                          \
        std::size_t j##x0 = i0;                                                \
        PROBS_CORE_DEF_IF1(j, x0)                                              \
        probs[0B##x0] += std::norm(arr[j##x0]);                                \
    }
#define PROBS_CORE_DEF_Ixy(x0, x1)                                             \
    {                                                                          \
        std::size_t j##x0##x1 = i0;                                            \
        PROBS_CORE_DEF_IF2(j, x0, x1)                                          \
        probs[0B##x0##x1] += std::norm(arr[j##x0##x1]);                        \
    }
#define PROBS_CORE_DEF_Ixyz(x0, x1, x2)                                        \
    {                                                                          \
        std::size_t j##x0##x1##x2 = i0;                                        \
        PROBS_CORE_DEF_IF3(j, x0, x1, x2)                                      \
        probs[0B##x0##x1##x2] += std::norm(arr[j##x0##x1##x2]);                \
    }
#define PROBS_CORE_DEF_Ixy4(x0, x1, x2, x3)                                    \
    {                                                                          \
        std::size_t j##x0##x1##x2##x3 = i0;                                    \
        PROBS_CORE_DEF_IF4(j, x0, x1, x2, x3)                                  \
        probs[0B##x0##x1##x2##x3] += std::norm(arr[j##x0##x1##x2##x3]);        \
    }
#define PROBS_CORE_DEF_Ixy5(x0, x1, x2, x3, x4)                                \
    {                                                                          \
        std::size_t j##x0##x1##x2##x3##x4 = i0;                                \
        PROBS_CORE_DEF_IF5(j, x0, x1, x2, x3, x4)                              \
        probs[0B##x0##x1##x2##x3##x4] +=                                       \
            std::norm(arr[j##x0##x1##x2##x3##x4]);                             \
    }
#define PROBS_CORE_DEF_Ixy6(x0, x1, x2, x3, x4, x5)                            \
    {                                                                          \
        std::size_t j##x0##x1##x2##x3##x4##x5 = i0;                            \
        PROBS_CORE_DEF_IF6(j, x0, x1, x2, x3, x4, x5)                          \
        probs[0B##x0##x1##x2##x3##x4##x5] +=                                   \
            std::norm(arr[j##x0##x1##x2##x3##x4##x5]);                         \
    }
#define PROBS_CORE_DEF_Ixy7(x0, x1, x2, x3, x4, x5, x6)                        \
    {                                                                          \
        std::size_t j##x0##x1##x2##x3##x4##x5##x6 = i0;                        \
        PROBS_CORE_DEF_IF7(j, x0, x1, x2, x3, x4, x5, x6)                      \
        probs[0B##x0##x1##x2##x3##x4##x5##x6] +=                               \
            std::norm(arr[j##x0##x1##x2##x3##x4##x5##x6]);                     \
    }
#define PROBS_CORE_DEF_Ixy8(x0, x1, x2, x3, x4, x5, x6, x7)                    \
    {                                                                          \
        std::size_t j##x0##x1##x2##x3##x4##x5##x6##x7 = i0;                    \
        PROBS_CORE_DEF_IF8(j, x0, x1, x2, x3, x4, x5, x6, x7)                  \
        probs[0B##x0##x1##x2##x3##x4##x5##x6##x7] +=                           \
            std::norm(arr[j##x0##x1##x2##x3##x4##x5##x6##x7]);                 \
    }
#define PROBS_CORE_DEF_Ixy9(x0, x1, x2, x3, x4, x5, x6, x7, x8)                \
    {                                                                          \
        std::size_t j##x0##x1##x2##x3##x4##x5##x6##x7##x8 = i0;                \
        PROBS_CORE_DEF_IF9(j, x0, x1, x2, x3, x4, x5, x6, x7, x8)              \
        probs[0B##x0##x1##x2##x3##x4##x5##x6##x7##x8] +=                       \
            std::norm(arr[j##x0##x1##x2##x3##x4##x5##x6##x7##x8]);             \
    }
#define PROBS_CORE_DEF_Ixy10(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)           \
    {                                                                          \
        std::size_t j##x0##x1##x2##x3##x4##x5##x6##x7##x8##x9 = i0;            \
        PROBS_CORE_DEF_IF10(j, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)         \
        probs[0B##x0##x1##x2##x3##x4##x5##x6##x7##x8##x9] +=                   \
            std::norm(arr[j##x0##x1##x2##x3##x4##x5##x6##x7##x8##x9]);         \
    }

#define PROBS_CORE_SUM_1                                                       \
    if constexpr (n_wires == 1) {                                              \
        i0 = PROBS_CORE_DEF_I0;                                                \
        PROBS_CORE_DEF_Ix(0);                                                  \
        PROBS_CORE_DEF_Ix(1);                                                  \
    }
#define PROBS_CORE_SUM_2_2(x) PROBS_CORE_DEF_Ixy(0, x) PROBS_CORE_DEF_Ixy(1, x)
#define PROBS_CORE_SUM_2                                                       \
    if constexpr (n_wires == 2) {                                              \
        i0 = PROBS_CORE_DEF_I00;                                               \
        PROBS_CORE_SUM_2_2(0);                                                 \
        PROBS_CORE_SUM_2_2(1)                                                  \
    }
#define PROBS_CORE_SUM_3_2(x, y)                                               \
    PROBS_CORE_DEF_Ixyz(0, x, y) PROBS_CORE_DEF_Ixyz(1, x, y)
#define PROBS_CORE_SUM_3_4(y) PROBS_CORE_SUM_3_2(0, y) PROBS_CORE_SUM_3_2(1, y)
#define PROBS_CORE_SUM_3                                                       \
    if constexpr (n_wires == 3) {                                              \
        i0 = PROBS_CORE_DEF_I000;                                              \
        PROBS_CORE_SUM_3_4(0);                                                 \
        PROBS_CORE_SUM_3_4(1)                                                  \
    }
#define PROBS_CORE_SUM_4_2(x1, x2, x3)                                         \
    PROBS_CORE_DEF_Ixy4(0, x1, x2, x3) PROBS_CORE_DEF_Ixy4(1, x1, x2, x3)
#define PROBS_CORE_SUM_4_4(x2, x3)                                             \
    PROBS_CORE_SUM_4_2(0, x2, x3) PROBS_CORE_SUM_4_2(1, x2, x3)
#define PROBS_CORE_SUM_4_8(x3)                                                 \
    PROBS_CORE_SUM_4_4(0, x3) PROBS_CORE_SUM_4_4(1, x3)
#define PROBS_CORE_SUM_4                                                       \
    if constexpr (n_wires == 4) {                                              \
        i0 = PROBS_CORE_DEF_I0000;                                             \
        PROBS_CORE_SUM_4_8(0);                                                 \
        PROBS_CORE_SUM_4_8(1)                                                  \
    }
#define PROBS_CORE_SUM_5_2(x1, x2, x3, x4)                                     \
    PROBS_CORE_DEF_Ixy5(0, x1, x2, x3, x4)                                     \
        PROBS_CORE_DEF_Ixy5(1, x1, x2, x3, x4)
#define PROBS_CORE_SUM_5_4(x2, x3, x4)                                         \
    PROBS_CORE_SUM_5_2(0, x2, x3, x4) PROBS_CORE_SUM_5_2(1, x2, x3, x4)
#define PROBS_CORE_SUM_5_8(x3, x4)                                             \
    PROBS_CORE_SUM_5_4(0, x3, x4) PROBS_CORE_SUM_5_4(1, x3, x4)
#define PROBS_CORE_SUM_5_16(x4)                                                \
    PROBS_CORE_SUM_5_8(0, x4) PROBS_CORE_SUM_5_8(1, x4)
#define PROBS_CORE_SUM_5                                                       \
    if constexpr (n_wires == 5) {                                              \
        i0 = PROBS_CORE_DEF_I00000;                                            \
        PROBS_CORE_SUM_5_16(0);                                                \
        PROBS_CORE_SUM_5_16(1)                                                 \
    }
#define PROBS_CORE_SUM_6_2(x1, x2, x3, x4, x5)                                 \
    PROBS_CORE_DEF_Ixy6(0, x1, x2, x3, x4, x5)                                 \
        PROBS_CORE_DEF_Ixy6(1, x1, x2, x3, x4, x5)
#define PROBS_CORE_SUM_6_4(x2, x3, x4, x5)                                     \
    PROBS_CORE_SUM_6_2(0, x2, x3, x4, x5) PROBS_CORE_SUM_6_2(1, x2, x3, x4, x5)
#define PROBS_CORE_SUM_6_8(x3, x4, x5)                                         \
    PROBS_CORE_SUM_6_4(0, x3, x4, x5) PROBS_CORE_SUM_6_4(1, x3, x4, x5)
#define PROBS_CORE_SUM_6_16(x4, x5)                                            \
    PROBS_CORE_SUM_6_8(0, x4, x5) PROBS_CORE_SUM_6_8(1, x4, x5)
#define PROBS_CORE_SUM_6_32(x5)                                                \
    PROBS_CORE_SUM_6_16(0, x5) PROBS_CORE_SUM_6_16(1, x5)
#define PROBS_CORE_SUM_6                                                       \
    if constexpr (n_wires == 6) {                                              \
        i0 = PROBS_CORE_DEF_I000000;                                           \
        PROBS_CORE_SUM_6_32(0);                                                \
        PROBS_CORE_SUM_6_32(1);                                                \
    }
#define PROBS_CORE_SUM_7_2(x1, x2, x3, x4, x5, x6)                             \
    PROBS_CORE_DEF_Ixy7(0, x1, x2, x3, x4, x5, x6)                             \
        PROBS_CORE_DEF_Ixy7(1, x1, x2, x3, x4, x5, x6)
#define PROBS_CORE_SUM_7_4(x2, x3, x4, x5, x6)                                 \
    PROBS_CORE_SUM_7_2(0, x2, x3, x4, x5, x6)                                  \
    PROBS_CORE_SUM_7_2(1, x2, x3, x4, x5, x6)
#define PROBS_CORE_SUM_7_8(x3, x4, x5, x6)                                     \
    PROBS_CORE_SUM_7_4(0, x3, x4, x5, x6) PROBS_CORE_SUM_7_4(1, x3, x4, x5, x6)
#define PROBS_CORE_SUM_7_16(x4, x5, x6)                                        \
    PROBS_CORE_SUM_7_8(0, x4, x5, x6) PROBS_CORE_SUM_7_8(1, x4, x5, x6)
#define PROBS_CORE_SUM_7_32(x5, x6)                                            \
    PROBS_CORE_SUM_7_16(0, x5, x6) PROBS_CORE_SUM_7_16(1, x5, x6)
#define PROBS_CORE_SUM_7_64(x6)                                                \
    PROBS_CORE_SUM_7_32(0, x6) PROBS_CORE_SUM_7_32(1, x6)
#define PROBS_CORE_SUM_7                                                       \
    if constexpr (n_wires == 7) {                                              \
        i0 = PROBS_CORE_DEF_I0000000;                                          \
        PROBS_CORE_SUM_7_64(0);                                                \
        PROBS_CORE_SUM_7_64(1);                                                \
    }
#define PROBS_CORE_SUM_8_2(x1, x2, x3, x4, x5, x6, x7)                         \
    PROBS_CORE_DEF_Ixy8(0, x1, x2, x3, x4, x5, x6, x7)                         \
        PROBS_CORE_DEF_Ixy8(1, x1, x2, x3, x4, x5, x6, x7)
#define PROBS_CORE_SUM_8_4(x2, x3, x4, x5, x6, x7)                             \
    PROBS_CORE_SUM_8_2(0, x2, x3, x4, x5, x6, x7)                              \
    PROBS_CORE_SUM_8_2(1, x2, x3, x4, x5, x6, x7)
#define PROBS_CORE_SUM_8_8(x3, x4, x5, x6, x7)                                 \
    PROBS_CORE_SUM_8_4(0, x3, x4, x5, x6, x7)                                  \
    PROBS_CORE_SUM_8_4(1, x3, x4, x5, x6, x7)
#define PROBS_CORE_SUM_8_16(x4, x5, x6, x7)                                    \
    PROBS_CORE_SUM_8_8(0, x4, x5, x6, x7) PROBS_CORE_SUM_8_8(1, x4, x5, x6, x7)
#define PROBS_CORE_SUM_8_32(x5, x6, x7)                                        \
    PROBS_CORE_SUM_8_16(0, x5, x6, x7) PROBS_CORE_SUM_8_16(1, x5, x6, x7)
#define PROBS_CORE_SUM_8_64(x6, x7)                                            \
    PROBS_CORE_SUM_8_32(0, x6, x7) PROBS_CORE_SUM_8_32(1, x6, x7)
#define PROBS_CORE_SUM_8_128(x6)                                               \
    PROBS_CORE_SUM_8_64(0, x6) PROBS_CORE_SUM_8_64(1, x6)
#define PROBS_CORE_SUM_8                                                       \
    if constexpr (n_wires == 8) {                                              \
        i0 = PROBS_CORE_DEF_I00000000;                                         \
        PROBS_CORE_SUM_8_128(0);                                               \
        PROBS_CORE_SUM_8_128(1);                                               \
    }
    template <std::size_t n_wires>
    auto probs_core(const std::complex<PrecisionT> *arr,
                    const std::size_t num_qubits,
                    const std::vector<std::size_t> &wires)
        -> std::vector<PrecisionT> {
        constexpr std::size_t one{1};
        std::vector<std::size_t> rev_wires(n_wires);
        for (std::size_t k = 0; k < n_wires; k++) {
            rev_wires[n_wires - 1 - k] = (num_qubits - 1) - wires[k];
        }
        const std::vector<std::size_t> parity =
            Pennylane::Util::revWireParity(rev_wires);
        PROBS_CORE_DECLARE_NW(0)
        PROBS_CORE_DECLARE_NW(1)
        PROBS_CORE_DECLARE_NW(2)
        PROBS_CORE_DECLARE_NW(3)
        PROBS_CORE_DECLARE_NW(4)
        PROBS_CORE_DECLARE_NW(5)
        PROBS_CORE_DECLARE_NW(6)
        PROBS_CORE_DECLARE_NW(7)
        PROBS_CORE_DECLARE_NW(8)
        PROBS_CORE_DECLARE_NW(9)
        const std::size_t parity_0 = parity[0];
        PROBS_CORE_DECLARE_P(1)
        PROBS_CORE_DECLARE_P(2)
        PROBS_CORE_DECLARE_P(3)
        PROBS_CORE_DECLARE_P(4)
        PROBS_CORE_DECLARE_P(5)
        PROBS_CORE_DECLARE_P(6)
        PROBS_CORE_DECLARE_P(7)
        PROBS_CORE_DECLARE_P(8)
        PROBS_CORE_DECLARE_P(9)
        PROBS_CORE_DECLARE_P(10)
        std::vector<PrecisionT> probs(PUtil::exp2(n_wires), 0);
        for (std::size_t k = 0; k < exp2(num_qubits - n_wires); k++) {
            std::size_t i0;
            PROBS_CORE_SUM_1
            PROBS_CORE_SUM_2
            PROBS_CORE_SUM_3
            PROBS_CORE_SUM_4
            PROBS_CORE_SUM_5
            PROBS_CORE_SUM_6
            PROBS_CORE_SUM_7
            PROBS_CORE_SUM_8
            // PROBS_CORE_SUM_9
            // PROBS_CORE_SUM_10
        }
        return probs;
    }

    /**
     * @brief Probabilities to measure rotated basis states.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots (Optional). If specified with a non-zero
     * number, shot-noise will be added to return probabilities
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    auto probs(const Observable<StateVectorT> &obs, std::size_t num_shots = 0)
        -> std::vector<PrecisionT> {
        return BaseType::probs(obs, num_shots);
    }

    /**
     * @brief Probabilities with shot-noise.
     *
     * @param num_shots Number of shots.
     *
     * @return Floating point std::vector with probabilities.
     */
    auto probs(size_t num_shots) -> std::vector<PrecisionT> {
        return BaseType::probs(num_shots);
    }

    /**
     * @brief Probabilities with shot-noise for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * @param num_shots Number of shots.
     * of the full system.
     *
     * @return Floating point std::vector with probabilities.
     */

    auto probs(const std::vector<std::size_t> &wires, std::size_t num_shots)
        -> std::vector<PrecisionT> {
        return BaseType::probs(wires, num_shots);
    }

    /**
     * @brief Expected value of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::vector<ComplexT> &matrix,
                const std::vector<std::size_t> &wires) -> PrecisionT {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorLQubitManaged<PrecisionT> operator_statevector(
            this->_statevector);

        operator_statevector.applyMatrix(matrix, wires);

        ComplexT expected_value = innerProdC(this->_statevector.getData(),
                                             operator_statevector.getData(),
                                             this->_statevector.getLength());
        return std::real(expected_value);
    };

    /**
     * @brief Expected value of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::string &operation,
                const std::vector<std::size_t> &wires) -> PrecisionT {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorLQubitManaged<PrecisionT> operator_statevector(
            this->_statevector);

        operator_statevector.applyOperation(operation, wires);

        ComplexT expected_value = innerProdC(this->_statevector.getData(),
                                             operator_statevector.getData(),
                                             this->_statevector.getLength());
        return std::real(expected_value);
    };

    /**
     * @brief Expected value of a Sparse Hamiltonian.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros
     above
     * row j.
     * @param row_map_size  row_map array size.
     * @param entries_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return Floating point expected value of the observable.
     */
    template <class index_type>
    auto expval(const index_type *row_map_ptr, const index_type row_map_size,
                const index_type *entries_ptr, const ComplexT *values_ptr,
                const index_type numNNZ) -> PrecisionT {
        PL_ABORT_IF(
            (this->_statevector.getLength() != (size_t(row_map_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");
        auto operator_vector = Util::apply_Sparse_Matrix(
            this->_statevector.getData(),
            static_cast<index_type>(this->_statevector.getLength()),
            row_map_ptr, row_map_size, entries_ptr, values_ptr, numNNZ);

        ComplexT expected_value =
            innerProdC(this->_statevector.getData(), operator_vector.data(),
                       this->_statevector.getLength());
        return std::real(expected_value);
    };

    /**
     * @brief Expected value for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with expected values for the
     * observables.
     */
    template <typename op_type>
    auto expval(const std::vector<op_type> &operations_list,
                const std::vector<std::vector<std::size_t>> &wires_list)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");
        std::vector<PrecisionT> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                expval(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    }

    /**
     * @brief Expectation value for a general Observable
     *
     * @param obs An observable object.
     * @return Floating point expected value of the observable.
     */
    auto expval(const Observable<StateVectorT> &obs) -> PrecisionT {
        PrecisionT result{};

        if constexpr (std::is_same_v<typename StateVectorT::MemoryStorageT,
                                     MemoryStorageLocation::Internal>) {
            StateVectorT sv(this->_statevector);
            result = calculateObsExpval(sv, obs, this->_statevector);
        } else if constexpr (std::is_same_v<
                                 typename StateVectorT::MemoryStorageT,
                                 MemoryStorageLocation::External>) {
            std::vector<ComplexT> data_storage(
                this->_statevector.getData(),
                this->_statevector.getData() + this->_statevector.getLength());
            StateVectorT sv(data_storage.data(), data_storage.size());
            result = calculateObsExpval(sv, obs, this->_statevector);
        } else {
            /// LCOV_EXCL_START
            PL_ABORT("Undefined memory storage location for StateVectorT.");
            /// LCOV_EXCL_STOP
        }

        return result;
    }

    /**
     * @brief Expectation value for a Observable with shots
     *
     * @param obs An observable object.
     * @param num_shots Number of shots.
     * @param shot_range Vector of shot number to measurement
     * @return Floating point expected value of the observable.
     */

    auto expval(const Observable<StateVectorT> &obs,
                const std::size_t &num_shots,
                const std::vector<std::size_t> &shot_range) -> PrecisionT {
        return BaseType::expval(obs, num_shots, shot_range);
    }

    /**
     * @brief Calculate the variance for an observable with the number of shots.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots.
     *
     * @return Variance of the given observable.
     */

    auto var(const Observable<StateVectorT> &obs, const std::size_t &num_shots)
        -> PrecisionT {
        return BaseType::var(obs, num_shots);
    }

    /**
     * @brief Variance value for a general Observable
     *
     * @param obs An observable object.
     * @return Floating point with the variance of the observable.
     */
    auto var(const Observable<StateVectorT> &obs) -> PrecisionT {
        PrecisionT result{};
        if constexpr (std::is_same_v<typename StateVectorT::MemoryStorageT,
                                     MemoryStorageLocation::Internal>) {
            StateVectorT sv(this->_statevector);
            result = calculateObsVar(sv, obs, this->_statevector);

        } else if constexpr (std::is_same_v<
                                 typename StateVectorT::MemoryStorageT,
                                 MemoryStorageLocation::External>) {
            std::vector<ComplexT> data_storage(
                this->_statevector.getData(),
                this->_statevector.getData() + this->_statevector.getLength());
            StateVectorT sv(data_storage.data(), data_storage.size());
            result = calculateObsVar(sv, obs, this->_statevector);
        } else {
            /// LCOV_EXCL_START
            PL_ABORT("Undefined memory storage location for StateVectorT.");
            /// LCOV_EXCL_STOP
        }
        return result;
    }

    /**
     * @brief Variance of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    auto var(const std::string &operation,
             const std::vector<std::size_t> &wires) -> PrecisionT {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorLQubitManaged<PrecisionT> operator_statevector(
            this->_statevector);

        operator_statevector.applyOperation(operation, wires);

        const std::complex<PrecisionT> *opsv_data =
            operator_statevector.getData();
        std::size_t orgsv_len = this->_statevector.getLength();

        PrecisionT mean_square =
            std::real(innerProdC(opsv_data, opsv_data, orgsv_len));
        PrecisionT squared_mean = std::real(
            innerProdC(this->_statevector.getData(), opsv_data, orgsv_len));
        squared_mean = static_cast<PrecisionT>(std::pow(squared_mean, 2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance of a Hermitian matrix.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    auto var(const std::vector<ComplexT> &matrix,
             const std::vector<std::size_t> &wires) -> PrecisionT {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorLQubitManaged<PrecisionT> operator_statevector(
            this->_statevector);

        operator_statevector.applyMatrix(matrix, wires);

        const std::complex<PrecisionT> *opsv_data =
            operator_statevector.getData();
        std::size_t orgsv_len = this->_statevector.getLength();

        PrecisionT mean_square =
            std::real(innerProdC(opsv_data, opsv_data, orgsv_len));
        PrecisionT squared_mean = std::real(
            innerProdC(this->_statevector.getData(), opsv_data, orgsv_len));
        squared_mean = static_cast<PrecisionT>(std::pow(squared_mean, 2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * Square matrix in row-major order or string with the operator name.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with the variance of the
     observables.
     */
    template <typename op_type>
    auto var(const std::vector<op_type> &operations_list,
             const std::vector<std::vector<std::size_t>> &wires_list)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");

        std::vector<PrecisionT> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                var(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    };

    /**
     * @brief Generate samples using the Metropolis-Hastings method.
     * Reference: Numerical Recipes, NetKet paper
     *
     * @param transition_kernel User-defined functor for producing
     transitions
     * between metropolis states.
     * @param num_burnin Number of Metropolis burn-in steps.
     * @param num_samples The number of samples to generate.
     * @return 1-D vector of samples in binary, each sample is
     * separated by a stride equal to the number of qubits.
     */
    std::vector<std::size_t>
    generate_samples_metropolis(const std::string &kernelname,
                                std::size_t num_burnin,
                                std::size_t num_samples) {
        std::size_t num_qubits = this->_statevector.getNumQubits();
        std::uniform_real_distribution<PrecisionT> distrib(0.0, 1.0);
        std::vector<std::size_t> samples(num_samples * num_qubits, 0);
        std::unordered_map<size_t, std::size_t> cache;
        this->setRandomSeed();

        TransitionKernelType transition_kernel = TransitionKernelType::Local;
        if (kernelname == "NonZeroRandom") {
            transition_kernel = TransitionKernelType::NonZeroRandom;
        }

        auto tk =
            kernel_factory(transition_kernel, this->_statevector.getData(),
                           this->_statevector.getNumQubits());
        std::size_t idx = 0;

        // Burn In
        for (size_t i = 0; i < num_burnin; i++) {
            idx = metropolis_step(this->_statevector, tk, this->rng, distrib,
                                  idx); // Burn-in.
        }

        // Sample
        for (size_t i = 0; i < num_samples; i++) {
            idx = metropolis_step(this->_statevector, tk, this->rng, distrib,
                                  idx);

            if (cache.contains(idx)) {
                std::size_t cache_id = cache[idx];
                auto it_temp = samples.begin() + cache_id * num_qubits;
                std::copy(it_temp, it_temp + num_qubits,
                          samples.begin() + i * num_qubits);
            }

            // If not cached, compute
            else {
                for (size_t j = 0; j < num_qubits; j++) {
                    samples[i * num_qubits + (num_qubits - 1 - j)] =
                        (idx >> j) & 1U;
                }
                cache[idx] = i;
            }
        }
        return samples;
    }

    /**
     * @brief Variance of a sparse Hamiltonian.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros
     above
     * row j.
     * @param row_map_size  row_map array size.
     * @param entries_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return Floating point with the variance of the sparse Hamiltonian.
     */
    template <class index_type>
    PrecisionT var(const index_type *row_map_ptr, const index_type row_map_size,
                   const index_type *entries_ptr, const ComplexT *values_ptr,
                   const index_type numNNZ) {
        PL_ABORT_IF(
            (this->_statevector.getLength() != (size_t(row_map_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");
        auto operator_vector = Util::apply_Sparse_Matrix(
            this->_statevector.getData(),
            static_cast<index_type>(this->_statevector.getLength()),
            row_map_ptr, row_map_size, entries_ptr, values_ptr, numNNZ);

        const PrecisionT mean_square =
            std::real(innerProdC(operator_vector.data(), operator_vector.data(),
                                 operator_vector.size()));
        const auto squared_mean = static_cast<PrecisionT>(
            std::pow(std::real(innerProdC(operator_vector.data(),
                                          this->_statevector.getData(),
                                          operator_vector.size())),
                     2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Generate samples using the alias method.
     * Reference: https://en.wikipedia.org/wiki/Alias_method
     *
     * @param num_samples The number of samples to generate.
     * @return 1-D vector of samples in binary, each sample is
     * separated by a stride equal to the number of qubits.
     */
    std::vector<std::size_t> generate_samples(size_t num_samples) {
        const std::size_t num_qubits = this->_statevector.getNumQubits();
        auto &&probabilities = probs();

        std::vector<std::size_t> samples(num_samples * num_qubits, 0);
        std::uniform_real_distribution<PrecisionT> distribution(0.0, 1.0);
        std::unordered_map<size_t, std::size_t> cache;
        this->setRandomSeed();

        const std::size_t N = probabilities.size();
        std::vector<double> bucket(N);
        std::vector<std::size_t> bucket_partner(N);
        std::stack<std::size_t> overfull_bucket_ids;
        std::stack<std::size_t> underfull_bucket_ids;

        for (size_t i = 0; i < N; i++) {
            bucket[i] = N * probabilities[i];
            bucket_partner[i] = i;
            if (bucket[i] > 1.0) {
                overfull_bucket_ids.push(i);
            }
            if (bucket[i] < 1.0) {
                underfull_bucket_ids.push(i);
            }
        }

        // Run alias algorithm
        while (!underfull_bucket_ids.empty() && !overfull_bucket_ids.empty()) {
            // get an overfull bucket
            std::size_t i = overfull_bucket_ids.top();

            // get an underfull bucket
            std::size_t j = underfull_bucket_ids.top();
            underfull_bucket_ids.pop();

            // underfull bucket is partned with an overfull bucket
            bucket_partner[j] = i;
            bucket[i] = bucket[i] + bucket[j] - 1;

            // if overfull bucket is now underfull
            // put in underfull stack
            if (bucket[i] < 1) {
                overfull_bucket_ids.pop();
                underfull_bucket_ids.push(i);
            }

            // if overfull bucket is full -> remove
            else if (bucket[i] == 1.0) {
                overfull_bucket_ids.pop();
            }
        }

        // Pick samples
        for (size_t i = 0; i < num_samples; i++) {
            PrecisionT pct = distribution(this->rng) * N;
            auto idx = static_cast<std::size_t>(pct);
            if (pct - idx > bucket[idx]) {
                idx = bucket_partner[idx];
            }
            // If cached, retrieve sample from cache
            if (cache.contains(idx)) {
                std::size_t cache_id = cache[idx];
                auto it_temp = samples.begin() + cache_id * num_qubits;
                std::copy(it_temp, it_temp + num_qubits,
                          samples.begin() + i * num_qubits);
            }
            // If not cached, compute
            else {
                for (size_t j = 0; j < num_qubits; j++) {
                    samples[i * num_qubits + (num_qubits - 1 - j)] =
                        (idx >> j) & 1U;
                }
                cache[idx] = i;
            }
        }
        return samples;
    }

  private:
    /**
     * @brief Support function that calculates <bra|obs|ket> to obtain the
     * observable's expectation value.
     *
     * @param bra Reference to the statevector where the observable will be
     * applied, must be mutable.
     * @param obs Constant reference to an observable.
     * @param ket Constant reference to the base statevector.
     * @return PrecisionT
     */
    auto inline calculateObsExpval(StateVectorT &bra,
                                   const Observable<StateVectorT> &obs,
                                   const StateVectorT &ket) -> PrecisionT {
        obs.applyInPlace(bra);
        return std::real(
            innerProdC(bra.getData(), ket.getData(), ket.getLength()));
    }

    /**
     * @brief Support function that calculates <bra|obs^2|ket> and
     * (<bra|obs|ket>)^2 to obtain the observable's variance.
     *
     * @param bra Reference to the statevector where the observable will be
     * applied, must be mutable.
     * @param obs Constant reference to an observable.
     * @param ket Constant reference to the base statevector.
     * @return PrecisionT
     */
    auto inline calculateObsVar(StateVectorT &bra,
                                const Observable<StateVectorT> &obs,
                                const StateVectorT &ket) -> PrecisionT {
        obs.applyInPlace(bra);
        PrecisionT mean_square = std::real(
            innerProdC(bra.getData(), bra.getData(), bra.getLength()));
        auto squared_mean = static_cast<PrecisionT>(
            std::pow(std::real(innerProdC(bra.getData(), ket.getData(),
                                          ket.getLength())),
                     2));
        return (mean_square - squared_mean);
    }

    /**
     * @brief Complete a single Metropolis-Hastings step.
     *
     * @param sv state vector
     * @param tk User-defined functor for producing transitions
     * between metropolis states.
     * @param gen Random number generator.
     * @param distrib Random number distribution.
     * @param init_idx Init index of basis state.
     */
    std::size_t
    metropolis_step(const StateVectorT &sv,
                    const std::unique_ptr<TransitionKernel<PrecisionT>> &tk,
                    std::mt19937 &gen,
                    std::uniform_real_distribution<PrecisionT> &distrib,
                    std::size_t init_idx) {
        auto init_plog = std::log(
            (sv.getData()[init_idx] * std::conj(sv.getData()[init_idx]))
                .real());

        auto init_qratio = tk->operator()(init_idx);

        // transition kernel outputs these two
        auto &trans_idx = init_qratio.first;
        auto &trans_qratio = init_qratio.second;

        auto trans_plog = std::log(
            (sv.getData()[trans_idx] * std::conj(sv.getData()[trans_idx]))
                .real());

        auto alph = std::min<PrecisionT>(
            1., trans_qratio * std::exp(trans_plog - init_plog));
        auto ran = distrib(gen);

        if (ran < alph) {
            return trans_idx;
        }
        return init_idx;
    }
}; // namespace Pennylane::LightningQubit::Measures
} // namespace Pennylane::LightningQubit::Measures