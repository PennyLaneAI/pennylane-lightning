// Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
 * @file ExpValFunc.hpp
 * Define functions and functor for in-place computation of expectation value of
 * general matrix, and named Identity, PauliX, PauliY, PauliZ, and Hadamard
 * operators.
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
/**
 * @brief Compute the parities and shifts for multi-qubit operations.
 *
 * @param num_qubits Number of qubits in the state vector.
 * @param wires List of target wires.
 * @return std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
 * Parities and shifts for multi-qubit operations.
 */
inline auto wires2Parity(std::size_t num_qubits,
                         const std::vector<std::size_t> &wires)
    -> std::pair<std::vector<std::size_t>, std::vector<std::size_t>> {
    std::vector<std::size_t> rev_wires_(wires.size());
    std::vector<std::size_t> rev_wire_shifts_(wires.size());
    for (std::size_t k = 0; k < wires.size(); k++) {
        rev_wires_[k] = (num_qubits - 1) - wires[(wires.size() - 1) - k];
        rev_wire_shifts_[k] = (static_cast<std::size_t>(1U) << rev_wires_[k]);
    }
    const std::vector<std::size_t> parity_ = PUtil::revWireParity(rev_wires_);

    return {parity_, rev_wire_shifts_};
}

template <class ParamT, class FuncT>
void applyExpVal1(const std::complex<ParamT> *arr, std::size_t num_qubits,
                  const std::vector<size_t> &wires, const FuncT &core_function,
                  ParamT &expected_value) {
    const std::size_t rev_wire = num_qubits - wires[0] - 1;
    const std::size_t rev_wire_shift =
        (static_cast<std::size_t>(1U) << rev_wire);
    const std::size_t wire_parity = PUtil::fillTrailingOnes(rev_wire);
    const std::size_t wire_parity_inv = PUtil::fillLeadingOnes(rev_wire + 1);

#pragma omp parallel for reduction(+ : expected_value)
    for (std::size_t k = 0; k < PUtil::exp2(num_qubits - 1); k++) {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        core_function(arr, i0, i1, expected_value);
    }
}

template <class ParamT>
void applyExpValMat1(const std::complex<ParamT> *arr, std::size_t num_qubits,
                     const std::vector<size_t> &wires,
                     const std::vector<std::complex<ParamT>> &matrix,
                     ParamT &expected_value) {
    const std::size_t rev_wire = num_qubits - wires[0] - 1;
    const std::size_t rev_wire_shift =
        (static_cast<std::size_t>(1U) << rev_wire);
    const std::size_t wire_parity = PUtil::fillTrailingOnes(rev_wire);
    const std::size_t wire_parity_inv = PUtil::fillLeadingOnes(rev_wire + 1);

#pragma omp parallel for reduction(+ : expected_value)
    for (std::size_t k = 0; k < PUtil::exp2(num_qubits - 1); k++) {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;

        expected_value +=
            std::real(std::conj(arr[i0]) *
                      (matrix[0B00] * arr[i0] + matrix[0B01] * arr[i1]));
        expected_value +=
            std::real(std::conj(arr[i1]) *
                      (matrix[0B10] * arr[i0] + matrix[0B11] * arr[i1]));
    }
}

#define EXPVALENTRY2(xx, yy) xx << 2 | yy
#define EXPVALTERM2(xx, yy, iyy) matrix[EXPVALENTRY2(xx, yy)] * arr[iyy]
#define EXPVAL2(ixx, xx)                                                       \
    std::conj(arr[ixx]) *                                                      \
        (EXPVALTERM2(xx, 0B00, i00) + EXPVALTERM2(xx, 0B01, i01) +             \
         EXPVALTERM2(xx, 0B10, i10) + EXPVALTERM2(xx, 0B11, i11))

template <class ParamT>
void applyExpValMat2(const std::complex<ParamT> *arr, std::size_t num_qubits,
                     const std::vector<size_t> &wires,
                     const std::vector<std::complex<ParamT>> &matrix,
                     ParamT &expected_value) {
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

#pragma omp parallel for reduction(+ : expected_value)
    for (std::size_t k = 0; k < PUtil::exp2(num_qubits - 2); k++) {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        expected_value += std::real(EXPVAL2(i00, 0B00));
        expected_value += std::real(EXPVAL2(i10, 0B10));
        expected_value += std::real(EXPVAL2(i01, 0B01));
        expected_value += std::real(EXPVAL2(i11, 0B11));
    }
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
void applyExpValMat3(const std::complex<ParamT> *arr, std::size_t num_qubits,
                     const std::vector<size_t> &wires,
                     const std::vector<std::complex<ParamT>> &matrix,
                     ParamT &expected_value) {
    const auto [parity, rev_wire_shifts] = wires2Parity(num_qubits, wires);

#pragma omp parallel for reduction(+ : expected_value)
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
        expected_value += std::real(EXPVAL3(i000, 0B000));
        expected_value += std::real(EXPVAL3(i001, 0B001));
        expected_value += std::real(EXPVAL3(i010, 0B010));
        expected_value += std::real(EXPVAL3(i011, 0B011));
        expected_value += std::real(EXPVAL3(i100, 0B100));
        expected_value += std::real(EXPVAL3(i101, 0B101));
        expected_value += std::real(EXPVAL3(i110, 0B110));
        expected_value += std::real(EXPVAL3(i111, 0B111));
    }
};

#define EXPVALENTRY4(xx, yy) xx << 4 | yy
#define EXPVALTERM4(xx, yy, iyy) matrix[EXPVALENTRY4(xx, yy)] * arr[iyy]
#define EXPVAL4(ixx, xx)                                                       \
    std::conj(arr[ixx]) *                                                      \
        (EXPVALTERM4(xx, 0B0000, i0000) + EXPVALTERM4(xx, 0B0001, i0001) +     \
         EXPVALTERM4(xx, 0B0010, i0010) + EXPVALTERM4(xx, 0B0011, i0011) +     \
         EXPVALTERM4(xx, 0B0100, i0100) + EXPVALTERM4(xx, 0B0101, i0101) +     \
         EXPVALTERM4(xx, 0B0110, i0110) + EXPVALTERM4(xx, 0B0111, i0111) +     \
         EXPVALTERM4(xx, 0B1000, i1000) + EXPVALTERM4(xx, 0B1001, i1001) +     \
         EXPVALTERM4(xx, 0B1010, i1010) + EXPVALTERM4(xx, 0B1011, i1011) +     \
         EXPVALTERM4(xx, 0B1100, i1100) + EXPVALTERM4(xx, 0B1101, i1101) +     \
         EXPVALTERM4(xx, 0B1110, i1110) + EXPVALTERM4(xx, 0B1111, i1111))

template <class ParamT>
void applyExpValMat4(const std::complex<ParamT> *arr, std::size_t num_qubits,
                     const std::vector<size_t> &wires,
                     const std::vector<std::complex<ParamT>> &matrix,
                     ParamT &expected_value) {
    const auto [parity, rev_wire_shifts] = wires2Parity(num_qubits, wires);

#pragma omp parallel for reduction(+ : expected_value)
    for (std::size_t k = 0; k < PUtil::exp2(num_qubits - 4); k++) {
        std::size_t i0000 = (k & parity[0]);
        for (std::size_t i = 1; i < parity.size(); i++) {
            i0000 |= ((k << i) & parity[i]);
        }

        std::size_t i0001 = i0000 | rev_wire_shifts[0];
        std::size_t i0010 = i0000 | rev_wire_shifts[1];
        std::size_t i0011 = i0000 | rev_wire_shifts[0] | rev_wire_shifts[1];
        std::size_t i0100 = i0000 | rev_wire_shifts[2];
        std::size_t i0101 = i0000 | rev_wire_shifts[0] | rev_wire_shifts[2];
        std::size_t i0110 = i0000 | rev_wire_shifts[1] | rev_wire_shifts[2];
        std::size_t i0111 = i0000 | rev_wire_shifts[0] | rev_wire_shifts[1] |
                            rev_wire_shifts[2];
        std::size_t i1000 = i0000 | rev_wire_shifts[3];
        std::size_t i1001 = i0000 | rev_wire_shifts[0] | rev_wire_shifts[3];
        std::size_t i1010 = i0000 | rev_wire_shifts[1] | rev_wire_shifts[3];
        std::size_t i1011 = i0000 | rev_wire_shifts[0] | rev_wire_shifts[1] |
                            rev_wire_shifts[3];
        std::size_t i1100 = i0000 | rev_wire_shifts[2] | rev_wire_shifts[3];
        std::size_t i1101 = i0000 | rev_wire_shifts[0] | rev_wire_shifts[2] |
                            rev_wire_shifts[3];
        std::size_t i1110 = i0000 | rev_wire_shifts[1] | rev_wire_shifts[2] |
                            rev_wire_shifts[3];
        std::size_t i1111 = i0000 | rev_wire_shifts[0] | rev_wire_shifts[1] |
                            rev_wire_shifts[2] | rev_wire_shifts[3];
        expected_value += std::real(EXPVAL4(i0000, 0B0000));
        expected_value += std::real(EXPVAL4(i0001, 0B0001));
        expected_value += std::real(EXPVAL4(i0010, 0B0010));
        expected_value += std::real(EXPVAL4(i0011, 0B0011));
        expected_value += std::real(EXPVAL4(i0100, 0B0100));
        expected_value += std::real(EXPVAL4(i0101, 0B0101));
        expected_value += std::real(EXPVAL4(i0110, 0B0110));
        expected_value += std::real(EXPVAL4(i0111, 0B0111));
        expected_value += std::real(EXPVAL4(i1000, 0B1000));
        expected_value += std::real(EXPVAL4(i1001, 0B1001));
        expected_value += std::real(EXPVAL4(i1010, 0B1010));
        expected_value += std::real(EXPVAL4(i1011, 0B1011));
        expected_value += std::real(EXPVAL4(i1100, 0B1100));
        expected_value += std::real(EXPVAL4(i1101, 0B1101));
        expected_value += std::real(EXPVAL4(i1110, 0B1110));
        expected_value += std::real(EXPVAL4(i1111, 0B1111));
    };
};

#define EXPVALENTRY5(xx, yy) xx << 5 | yy
#define EXPVALTERM5(xx, yy, iyy) matrix[EXPVALENTRY5(xx, yy)] * arr[iyy]
#define EXPVAL5(ixx, xx)                                                       \
    std::conj(arr[ixx]) *                                                      \
        (EXPVALTERM5(xx, 0B00000, i00000) + EXPVALTERM5(xx, 0B00001, i00001) + \
         EXPVALTERM5(xx, 0B00010, i00010) + EXPVALTERM5(xx, 0B00011, i00011) + \
         EXPVALTERM5(xx, 0B00100, i00100) + EXPVALTERM5(xx, 0B00101, i00101) + \
         EXPVALTERM5(xx, 0B00110, i00110) + EXPVALTERM5(xx, 0B00111, i00111) + \
         EXPVALTERM5(xx, 0B01000, i01000) + EXPVALTERM5(xx, 0B01001, i01001) + \
         EXPVALTERM5(xx, 0B01010, i01010) + EXPVALTERM5(xx, 0B01011, i01011) + \
         EXPVALTERM5(xx, 0B01100, i01100) + EXPVALTERM5(xx, 0B01101, i01101) + \
         EXPVALTERM5(xx, 0B01110, i01110) + EXPVALTERM5(xx, 0B01111, i01111) + \
         EXPVALTERM5(xx, 0B10000, i10000) + EXPVALTERM5(xx, 0B10001, i10001) + \
         EXPVALTERM5(xx, 0B10010, i10010) + EXPVALTERM5(xx, 0B10011, i10011) + \
         EXPVALTERM5(xx, 0B10100, i10100) + EXPVALTERM5(xx, 0B10101, i10101) + \
         EXPVALTERM5(xx, 0B10110, i10110) + EXPVALTERM5(xx, 0B10111, i10111) + \
         EXPVALTERM5(xx, 0B11000, i11000) + EXPVALTERM5(xx, 0B11001, i11001) + \
         EXPVALTERM5(xx, 0B11010, i11010) + EXPVALTERM5(xx, 0B11011, i11011) + \
         EXPVALTERM5(xx, 0B11100, i11100) + EXPVALTERM5(xx, 0B11101, i11101) + \
         EXPVALTERM5(xx, 0B11110, i11110) + EXPVALTERM5(xx, 0B11111, i11111))

template <class ParamT>
void applyExpValMat5(const std::complex<ParamT> *arr, std::size_t num_qubits,
                     const std::vector<size_t> &wires,
                     const std::vector<std::complex<ParamT>> &matrix,
                     ParamT &expected_value) {
    const auto [parity, rev_wire_shifts] = wires2Parity(num_qubits, wires);

#pragma omp parallel for reduction(+ : expected_value)
    for (std::size_t k = 0; k < PUtil::exp2(num_qubits - 5); k++) {
        std::size_t i00000 = (k & parity[0]);
        for (std::size_t i = 1; i < parity.size(); i++) {
            i00000 |= ((k << i) & parity[i]);
        }

        std::size_t i00001 = i00000 | rev_wire_shifts[0];
        std::size_t i00010 = i00000 | rev_wire_shifts[1];
        std::size_t i00011 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[1];
        std::size_t i00100 = i00000 | rev_wire_shifts[2];
        std::size_t i00101 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[2];
        std::size_t i00110 = i00000 | rev_wire_shifts[1] | rev_wire_shifts[2];
        std::size_t i00111 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[1] |
                             rev_wire_shifts[2];
        std::size_t i01000 = i00000 | rev_wire_shifts[3];
        std::size_t i01001 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[3];
        std::size_t i01010 = i00000 | rev_wire_shifts[1] | rev_wire_shifts[3];
        std::size_t i01011 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[1] |
                             rev_wire_shifts[3];
        std::size_t i01100 = i00000 | rev_wire_shifts[2] | rev_wire_shifts[3];
        std::size_t i01101 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[2] |
                             rev_wire_shifts[3];
        std::size_t i01110 = i00000 | rev_wire_shifts[1] | rev_wire_shifts[2] |
                             rev_wire_shifts[3];
        std::size_t i01111 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[1] |
                             rev_wire_shifts[2] | rev_wire_shifts[3];
        std::size_t i10000 = i00000 | rev_wire_shifts[4];
        std::size_t i10001 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[4];
        std::size_t i10010 = i00000 | rev_wire_shifts[1] | rev_wire_shifts[4];
        std::size_t i10011 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[1] |
                             rev_wire_shifts[4];
        std::size_t i10100 = i00000 | rev_wire_shifts[2] | rev_wire_shifts[4];
        std::size_t i10101 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[2] |
                             rev_wire_shifts[4];
        std::size_t i10110 = i00000 | rev_wire_shifts[1] | rev_wire_shifts[2] |
                             rev_wire_shifts[4];
        std::size_t i10111 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[1] |
                             rev_wire_shifts[2] | rev_wire_shifts[4];
        std::size_t i11000 = i00000 | rev_wire_shifts[3] | rev_wire_shifts[4];
        std::size_t i11001 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[3] |
                             rev_wire_shifts[4];
        std::size_t i11010 = i00000 | rev_wire_shifts[1] | rev_wire_shifts[3] |
                             rev_wire_shifts[4];
        std::size_t i11011 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[1] |
                             rev_wire_shifts[3] | rev_wire_shifts[4];
        std::size_t i11100 = i00000 | rev_wire_shifts[2] | rev_wire_shifts[3] |
                             rev_wire_shifts[4];
        std::size_t i11101 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[2] |
                             rev_wire_shifts[3] | rev_wire_shifts[4];
        std::size_t i11110 = i00000 | rev_wire_shifts[1] | rev_wire_shifts[2] |
                             rev_wire_shifts[3] | rev_wire_shifts[4];
        std::size_t i11111 = i00000 | rev_wire_shifts[0] | rev_wire_shifts[1] |
                             rev_wire_shifts[2] | rev_wire_shifts[3] |
                             rev_wire_shifts[4];

        expected_value += std::real(EXPVAL5(i00000, 0B00000));
        expected_value += std::real(EXPVAL5(i00001, 0B00001));
        expected_value += std::real(EXPVAL5(i00010, 0B00010));
        expected_value += std::real(EXPVAL5(i00011, 0B00011));
        expected_value += std::real(EXPVAL5(i00100, 0B00100));
        expected_value += std::real(EXPVAL5(i00101, 0B00101));
        expected_value += std::real(EXPVAL5(i00110, 0B00110));
        expected_value += std::real(EXPVAL5(i00111, 0B00111));
        expected_value += std::real(EXPVAL5(i01000, 0B01000));
        expected_value += std::real(EXPVAL5(i01001, 0B01001));
        expected_value += std::real(EXPVAL5(i01010, 0B01010));
        expected_value += std::real(EXPVAL5(i01011, 0B01011));
        expected_value += std::real(EXPVAL5(i01100, 0B01100));
        expected_value += std::real(EXPVAL5(i01101, 0B01101));
        expected_value += std::real(EXPVAL5(i01110, 0B01110));
        expected_value += std::real(EXPVAL5(i01111, 0B01111));
        expected_value += std::real(EXPVAL5(i10000, 0B10000));
        expected_value += std::real(EXPVAL5(i10001, 0B10001));
        expected_value += std::real(EXPVAL5(i10010, 0B10010));
        expected_value += std::real(EXPVAL5(i10011, 0B10011));
        expected_value += std::real(EXPVAL5(i10100, 0B10100));
        expected_value += std::real(EXPVAL5(i10101, 0B10101));
        expected_value += std::real(EXPVAL5(i10110, 0B10110));
        expected_value += std::real(EXPVAL5(i10111, 0B10111));
        expected_value += std::real(EXPVAL5(i11000, 0B11000));
        expected_value += std::real(EXPVAL5(i11001, 0B11001));
        expected_value += std::real(EXPVAL5(i11010, 0B11010));
        expected_value += std::real(EXPVAL5(i11011, 0B11011));
        expected_value += std::real(EXPVAL5(i11100, 0B11100));
        expected_value += std::real(EXPVAL5(i11101, 0B11101));
        expected_value += std::real(EXPVAL5(i11110, 0B11110));
        expected_value += std::real(EXPVAL5(i11111, 0B11111));
    };
};

template <class ParamT>
void applyExpValMatMultiQubit(const std::complex<ParamT> *arr,
                              std::size_t num_qubits,
                              const std::vector<size_t> &wires,
                              const std::vector<std::complex<ParamT>> &matrix,
                              ParamT &expected_value) {
    const auto [parity, rev_wire_shifts] = wires2Parity(num_qubits, wires);

    const std::size_t dim = PUtil::exp2(wires.size());
    const std::size_t two2N = PUtil::exp2(num_qubits - wires.size());
#pragma omp parallel for reduction(+ : expected_value)
    for (std::size_t k = 0; k < two2N; ++k) {
        ParamT innerExpVal = 0.0;
        std::vector<std::complex<ParamT>> coeffs_in(dim);

        std::size_t idx = (k & parity[0]);
        for (std::size_t i = 1; i < parity.size(); i++) {
            idx |= (((k << i) & parity[i]));
        }
        coeffs_in[0] = arr[idx];

        for (std::size_t inner_idx = 1; inner_idx < dim; ++inner_idx) {
            std::size_t index = idx;
            for (std::size_t i = 0; i < wires.size(); i++) {
                if ((inner_idx & (static_cast<std::size_t>(1U) << i)) != 0) {
                    index |= rev_wire_shifts[i];
                }
            }
            coeffs_in[inner_idx] = arr[index];
        }

        for (std::size_t i = 0; i < dim; ++i) {
            std::complex<ParamT> tmp(0.0);
            for (std::size_t j = 0; j < dim; ++j) {
                tmp += matrix[i * dim + j] * coeffs_in[j];
            }
            innerExpVal += std::real(std::conj(coeffs_in[i]) * tmp);
        }
        expected_value += innerExpVal;
    }
};
} // namespace Pennylane::LightningQubit::Measures
