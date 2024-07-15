// Copyright 2024 Xanadu Quantum Technologies Inc.

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
 * @file MeasurementsKernels.hpp
 * Defines macros and methods to support Lightning-Qubit's probs(wires) using bitshift implementation.
 */
#pragma once

#include <complex>
#include <vector>

#include "BitUtil.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
namespace PUtil = Pennylane::Util;
} // namespace
/// @endcond

#define PROBS_SPECIAL_CASE(n)                                                  \
    if (n_wires == n) {                                                        \
        return Pennylane::LightningQubit::Measures::probs_bitshift<PrecisionT, \
                                                                   n>(         \
            arr_data, num_qubits, wires);                                      \
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

namespace Pennylane::LightningQubit::Measures {

template <class PrecisionT>
auto probs_bitshift_generic(const std::complex<PrecisionT> *arr,
                            const std::size_t num_qubits,
                            const std::vector<std::size_t> &wires) {
    constexpr std::size_t one{1};
    const std::size_t n_wires = wires.size();
    std::vector<std::size_t> rev_wires(n_wires);
    for (std::size_t k = 0; k < n_wires; k++) {
        rev_wires[n_wires - 1 - k] = (num_qubits - 1) - wires[k];
    }
    const std::vector<std::size_t> parity =
        Pennylane::Util::revWireParity(rev_wires);
    const std::size_t n_probs = PUtil::exp2(n_wires);
    std::vector<PrecisionT> probabilities(n_probs, 0);
    for (std::size_t k = 0; k < exp2(num_qubits - n_wires); k++) {
        std::size_t idx = (k & parity[0]);
        for (std::size_t i = 1; i < n_wires + 1; i++) {
            idx |= ((k << i) & parity[i]);
        }
        probabilities[0] += std::norm(arr[idx]);
        const std::size_t i0 = idx;
        for (std::size_t inner_idx = 1; inner_idx < n_probs; inner_idx++) {
            idx = i0;
            for (std::size_t i = 0; i < n_wires; i++) {
                idx |= ((inner_idx & (one << i)) >> i) << rev_wires[i];
            }
            probabilities[inner_idx] += std::norm(arr[idx]);
        }
    }
    return probabilities;
}

// NOLINTBEGIN(hicpp-function-size,readability-function-size)
template <class PrecisionT, std::size_t n_wires>
auto probs_bitshift(const std::complex<PrecisionT> *arr,
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
    const std::size_t parity_0 = parity[0];
    PROBS_CORE_DECLARE_P(1)
    PROBS_CORE_DECLARE_P(2)
    PROBS_CORE_DECLARE_P(3)
    PROBS_CORE_DECLARE_P(4)
    PROBS_CORE_DECLARE_P(5)
    PROBS_CORE_DECLARE_P(6)
    PROBS_CORE_DECLARE_P(7)
    PROBS_CORE_DECLARE_P(8)
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
    }
    return probs;
}
// NOLINTEND(hicpp-function-size,readability-function-size)
} // namespace Pennylane::LightningQubit::Measures