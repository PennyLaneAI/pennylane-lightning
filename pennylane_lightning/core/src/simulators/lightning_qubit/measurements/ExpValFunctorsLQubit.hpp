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

/**
 * @file ExpValFunctorsQubit.hpp
 * Define functors for in-place computation of expectation value of
 * Identity, PauliX, PauliY, PauliZ, and Hadamard operators.
 */

#pragma once

#include "BitUtil.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
namespace PUtil = Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Measures {

template <class ParamT, class FuncT>
void applyExpVal1(const std::complex<ParamT> *arr, const std::size_t num_qubits,
                  const std::vector<size_t> &wires, FuncT &core_function,
                  [[maybe_unused]] ParamT &expected_value) {
    const std::size_t rev_wire = num_qubits - wires[0] - 1;
    const std::size_t rev_wire_shift =
        (static_cast<std::size_t>(1) << rev_wire);
    const auto parities =
        PUtil::revWireParity(std::array<std::size_t, 1>{rev_wire});
    const auto parity_high = parities[1];
    const auto parity_low = parities[0];
#pragma omp parallel for collapse(1) reduction(+ : expected_value)
    for (std::size_t k = 0; k < PUtil::exp2(num_qubits - 1); k++) {
        const std::size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        core_function(arr, i0, i1);
    }
}

template <class ParamT>
void applyExpValMat1(const std::complex<ParamT> *arr,
                     const std::size_t num_qubits,
                     const std::vector<size_t> &wires,
                     const std::vector<std::complex<ParamT>> &matrix,
                     [[maybe_unused]] ParamT &expected_value) {
    const std::size_t rev_wire = num_qubits - wires[0] - 1;
    const std::size_t rev_wire_shift =
        (static_cast<std::size_t>(1U) << rev_wire);
    const std::size_t wire_parity = PUtil::fillTrailingOnes(rev_wire);
    const std::size_t wire_parity_inv = PUtil::fillLeadingOnes(rev_wire + 1);
#pragma omp parallel for collapse(1) reduction(+ : expected_value)
    for (std::size_t k = 0; k < PUtil::exp2(num_qubits - 1); k++) {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;

        expected_value += real(
            conj(arr[i0]) * (matrix[0B00] * arr[i0] + matrix[0B01] * arr[i1]));
        expected_value += real(
            conj(arr[i1]) * (matrix[0B10] * arr[i0] + matrix[0B11] * arr[i1]));
    }
}

#define EXPVALENTRY2(xx, yy) xx << 2 | yy
#define EXPVALTERM2(xx, yy, iyy) matrix[EXPVALENTRY2(xx, yy)] * arr[iyy]
#define EXPVAL2(ixx, xx)                                                       \
    std::conj(arr[ixx]) *                                                      \
        (EXPVALTERM2(xx, 0B00, i00) + EXPVALTERM2(xx, 0B01, i01) +             \
         EXPVALTERM2(xx, 0B10, i10) + EXPVALTERM2(xx, 0B11, i11))

template <class ParamT>
void applyExpValMat2(const std::complex<ParamT> *arr,
                     const std::size_t num_qubits,
                     const std::vector<size_t> &wires,
                     const std::vector<std::complex<ParamT>> &matrix,
                     [[maybe_unused]] ParamT &expected_value) {

    const std::size_t rev_wire0 = num_qubits - wires[1] - 1;
    const std::size_t rev_wire1 = num_qubits - wires[0] - 1;
    const std::size_t rev_wire0_shift = static_cast<std::size_t>(1U)
                                        << rev_wire0;
    const std::size_t rev_wire1_shift = static_cast<std::size_t>(1U)
                                        << rev_wire1;
    const std::size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
    const std::size_t rev_wire_max = std::max(rev_wire0, rev_wire1);
    const std::size_t parity_low = PUtil::fillTrailingOnes(rev_wire_min);
    const std::size_t parity_high = PUtil::fillLeadingOnes(rev_wire_max + 1);
    const std::size_t parity_middle = PUtil::fillLeadingOnes(rev_wire_min + 1) &
                                      PUtil::fillTrailingOnes(rev_wire_max);

#pragma omp parallel for collapse(1) reduction(+ : expected_value)
    for (std::size_t k = 0; k < PUtil::exp2(num_qubits - 2); k++) {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        expected_value += real(EXPVAL2(i00, 0B00));
        expected_value += real(EXPVAL2(i10, 0B10));
        expected_value += real(EXPVAL2(i01, 0B01));
        expected_value += real(EXPVAL2(i11, 0B11));
    }
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
    -> std::pair<std::vector<std::size_t>, std::vector<std::size_t>> {

    std::vector<std::size_t> rev_wires_(wires.size());
    std::vector<std::size_t> rev_wire_shifts_(wires.size());
    for (std::size_t k = 0; k < wires.size(); k++) {
        rev_wires_[k] = (num_qubits - 1) - wires[(wires.size() - 1) - k];
        rev_wire_shifts_[k] = (static_cast<std::size_t>(1) << rev_wires_[k]);
    }
    const std::vector<std::size_t> parity_ = PUtil::revWireParity(rev_wires_);

    return {parity_, rev_wire_shifts_};
}

#define EXPVALENTRY3(xx, yy) xx << 3 | yy
#define EXPVALTERM3(xx, yy, iyy) matrix[EXPVALENTRY3(xx, yy)] * arr[iyy]
#define EXPVAL3(ixx, xx)                                                       \
    std::conj(arr[ixx]) *                                                      \
        (EXPVALTERM3(xx, 0B000, i000) + EXPVALTERM3(xx, 0B001, i001) +         \
         EXPVALTERM3(xx, 0B010, i010) + EXPVALTERM3(xx, 0B011, i011) +         \
         EXPVALTERM3(xx, 0B100, i100) + EXPVALTERM3(xx, 0B101, i101) +         \
         EXPVALTERM3(xx, 0B110, i110) + EXPVALTERM3(xx, 0B111, i111))
template <class ParamT>
void applyExpValMat3(const std::complex<ParamT> *arr,
                     const std::size_t num_qubits,
                     const std::vector<size_t> &wires,
                     const std::vector<std::complex<ParamT>> &matrix,
                     [[maybe_unused]] ParamT &expected_value) {
    std::vector<std::size_t> parity;
    std::vector<std::size_t> rev_wire_shifts;
    std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits, wires);

#pragma omp parallel for collapse(1) reduction(+ : expected_value)
    for (std::size_t k = 0; k < PUtil::exp2(num_qubits - 3); k++) {
        std::size_t i000 = (k & parity[0]);
        for (std::size_t i = 1; i < parity.size(); i++) {
            i000 |= ((k << i) & parity[i]);
        }

        std::size_t i001 = i000 | rev_wire_shifts[0];
        std::size_t i010 = i000 | rev_wire_shifts[1];
        std::size_t i011 = i000 | rev_wire_shifts[0] | rev_wire_shifts[1];
        std::size_t i100 = i000 | rev_wire_shifts[2];
        std::size_t i101 = i000 | rev_wire_shifts[0] | rev_wire_shifts[2];
        std::size_t i110 = i000 | rev_wire_shifts[1] | rev_wire_shifts[2];
        std::size_t i111 =
            i000 | rev_wire_shifts[0] | rev_wire_shifts[1] | rev_wire_shifts[2];
        expected_value += real(EXPVAL3(i000, 0B000));
        expected_value += real(EXPVAL3(i001, 0B001));
        expected_value += real(EXPVAL3(i010, 0B010));
        expected_value += real(EXPVAL3(i011, 0B011));
        expected_value += real(EXPVAL3(i100, 0B100));
        expected_value += real(EXPVAL3(i101, 0B101));
        expected_value += real(EXPVAL3(i110, 0B110));
        expected_value += real(EXPVAL3(i111, 0B111));
    }
};

} // namespace Pennylane::LightningQubit::Measures
