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
 * Defines kernel functions with less memory (and fast)
 */
#pragma once
#include <algorithm>
#include <bit>
#include <complex>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "BitUtil.hpp" // revWireParity
#include "Error.hpp"
#include "GateIndices.hpp"
#include "GateOperation.hpp"
#include "GatePragmas.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"
#include "PauliGenerator.hpp"
#include "Util.hpp" // exp2, INVSQRT2

/// @cond DEV
namespace {
using namespace Pennylane::Gates;
using namespace Pennylane::LightningQubit::Gates::Pragmas;
using Pennylane::Util::exp2;
using Pennylane::Util::INVSQRT2;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Gates {
/**
 * @brief A gate operation implementation with less memory.
 *
 * We use a bitwise operation to calculate the indices where the gate
 * applies to on the fly.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 */
class GateImplementationsLM : public PauliGenerator<GateImplementationsLM> {
  private:
    /* Alias utility functions */
    static std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
    reverseWires(const std::size_t num_qubits,
                 const std::vector<std::size_t> &all_wires,
                 const std::vector<bool> &controlled_values) {
        constexpr std::size_t one{1U};
        const std::size_t nw_tot = all_wires.size();
        const std::size_t n_contr = controlled_values.size();
        std::vector<std::size_t> rev_wires(nw_tot);
        std::vector<std::size_t> rev_wire_shifts(nw_tot);
        for (std::size_t k = 0; k < nw_tot; k++) {
            rev_wires[k] = (num_qubits - 1) - all_wires[(nw_tot - 1) - k];
            const std::size_t value =
                (k < n_contr) ? static_cast<std::size_t>(
                                    controlled_values[(n_contr - 1) - k])
                              : one;
            rev_wire_shifts[k] = (value << rev_wires[k]);
        }
        return {rev_wires, rev_wire_shifts};
    }

    static std::pair<std::size_t, std::size_t>
    revWireParity(std::size_t rev_wire) {
        const auto parity = Pennylane::Util::revWireParity(
            std::array<std::size_t, 1>{rev_wire});
        return {parity[1], parity[0]};
    }
    static std::tuple<std::size_t, std::size_t, std::size_t>
    revWireParity(std::size_t rev_wire0, std::size_t rev_wire1) {
        const auto parity = Pennylane::Util::revWireParity(
            std::array<std::size_t, 2>{rev_wire0, rev_wire1});
        return {parity[2], parity[1], parity[0]};
    }
    template <const std::size_t wire_size = 3>
    static constexpr auto revWireParity(std::size_t rev_wire0,
                                        std::size_t rev_wire1,
                                        std::size_t rev_wire2)
        -> std::array<std::size_t, wire_size + 1> {
        return Pennylane::Util::revWireParity(
            std::array<std::size_t, wire_size>{rev_wire0, rev_wire1,
                                               rev_wire2});
    }
    template <const std::size_t wire_size = 4>
    static constexpr auto
    revWireParity(std::size_t rev_wire0, std::size_t rev_wire1,
                  std::size_t rev_wire2, std::size_t rev_wire3)
        -> std::array<std::size_t, wire_size + 1> {
        return Pennylane::Util::revWireParity(
            std::array<std::size_t, wire_size>{rev_wire0, rev_wire1, rev_wire2,
                                               rev_wire3});
    }

  public:
    constexpr static KernelType kernel_id = KernelType::LM;
    constexpr static std::string_view name = "LM";
    template <typename PrecisionT>
    constexpr static std::size_t required_alignment =
        std::alignment_of_v<PrecisionT>;
    template <typename PrecisionT>
    constexpr static std::size_t packed_bytes = sizeof(PrecisionT);

    constexpr static std::array implemented_gates = {
        GateOperation::Identity,
        GateOperation::PauliX,
        GateOperation::PauliY,
        GateOperation::PauliZ,
        GateOperation::Hadamard,
        GateOperation::S,
        GateOperation::SX,
        GateOperation::T,
        GateOperation::PhaseShift,
        GateOperation::RX,
        GateOperation::RY,
        GateOperation::RZ,
        GateOperation::Rot,
        GateOperation::CNOT,
        GateOperation::CY,
        GateOperation::CZ,
        GateOperation::SWAP,
        GateOperation::CSWAP,
        GateOperation::Toffoli,
        GateOperation::IsingXX,
        GateOperation::IsingXY,
        GateOperation::IsingYY,
        GateOperation::IsingZZ,
        GateOperation::ControlledPhaseShift,
        GateOperation::CRX,
        GateOperation::CRY,
        GateOperation::CRZ,
        GateOperation::CRot,
        GateOperation::SingleExcitation,
        GateOperation::SingleExcitationMinus,
        GateOperation::SingleExcitationPlus,
        GateOperation::DoubleExcitation,
        GateOperation::DoubleExcitationMinus,
        GateOperation::DoubleExcitationPlus,
        GateOperation::MultiRZ,
        GateOperation::GlobalPhase,
        GateOperation::PSWAP,
        GateOperation::PCPhase,
    };

    constexpr static std::array implemented_controlled_gates = {
        ControlledGateOperation::PauliX,
        ControlledGateOperation::PauliY,
        ControlledGateOperation::PauliZ,
        ControlledGateOperation::Hadamard,
        ControlledGateOperation::S,
        ControlledGateOperation::SX,
        ControlledGateOperation::T,
        ControlledGateOperation::PhaseShift,
        ControlledGateOperation::RX,
        ControlledGateOperation::RY,
        ControlledGateOperation::RZ,
        ControlledGateOperation::Rot,
        ControlledGateOperation::SWAP,
        ControlledGateOperation::IsingXX,
        ControlledGateOperation::IsingXY,
        ControlledGateOperation::IsingYY,
        ControlledGateOperation::IsingZZ,
        ControlledGateOperation::SingleExcitation,
        ControlledGateOperation::SingleExcitationMinus,
        ControlledGateOperation::SingleExcitationPlus,
        ControlledGateOperation::DoubleExcitation,
        ControlledGateOperation::DoubleExcitationMinus,
        ControlledGateOperation::DoubleExcitationPlus,
        ControlledGateOperation::MultiRZ,
        ControlledGateOperation::GlobalPhase,
        ControlledGateOperation::PSWAP,
        ControlledGateOperation::PCPhase,
    };

    constexpr static std::array implemented_generators = {
        GeneratorOperation::PhaseShift,
        GeneratorOperation::RX,
        GeneratorOperation::RY,
        GeneratorOperation::RZ,
        GeneratorOperation::IsingXX,
        GeneratorOperation::IsingXY,
        GeneratorOperation::IsingYY,
        GeneratorOperation::IsingZZ,
        GeneratorOperation::CRX,
        GeneratorOperation::CRY,
        GeneratorOperation::CRZ,
        GeneratorOperation::ControlledPhaseShift,
        GeneratorOperation::SingleExcitation,
        GeneratorOperation::SingleExcitationMinus,
        GeneratorOperation::SingleExcitationPlus,
        GeneratorOperation::DoubleExcitation,
        GeneratorOperation::DoubleExcitationMinus,
        GeneratorOperation::DoubleExcitationPlus,
        GeneratorOperation::PSWAP,
        GeneratorOperation::MultiRZ,
        GeneratorOperation::GlobalPhase,
    };

    constexpr static std::array implemented_controlled_generators = {
        ControlledGeneratorOperation::PhaseShift,
        ControlledGeneratorOperation::RX,
        ControlledGeneratorOperation::RY,
        ControlledGeneratorOperation::RZ,
        ControlledGeneratorOperation::IsingXX,
        ControlledGeneratorOperation::IsingXY,
        ControlledGeneratorOperation::IsingYY,
        ControlledGeneratorOperation::IsingZZ,
        ControlledGeneratorOperation::SingleExcitation,
        ControlledGeneratorOperation::SingleExcitationMinus,
        ControlledGeneratorOperation::SingleExcitationPlus,
        ControlledGeneratorOperation::DoubleExcitation,
        ControlledGeneratorOperation::DoubleExcitationMinus,
        ControlledGeneratorOperation::DoubleExcitationPlus,
        ControlledGeneratorOperation::PSWAP,
        ControlledGeneratorOperation::MultiRZ,
        ControlledGeneratorOperation::GlobalPhase,
    };

    constexpr static std::array implemented_matrices = {
        MatrixOperation::SingleQubitOp,
        MatrixOperation::TwoQubitOp,
        MatrixOperation::MultiQubitOp,
    };

    constexpr static std::array implemented_controlled_matrices = {
        ControlledMatrixOperation::NCSingleQubitOp,
        ControlledMatrixOperation::NCTwoQubitOp,
        ControlledMatrixOperation::NCMultiQubitOp,
    };

    constexpr static std::array implemented_sparse_matrices = {
        SparseMatrixOperation::SparseMultiQubitOp,
    };

    constexpr static std::array implemented_controlled_sparse_matrices = {
        ControlledSparseMatrixOperation::NCSparseMultiQubitOp,
    };

    static std::size_t parity_2_offset(const std::vector<std::size_t> &parity,
                                       const std::size_t k) {
        std::size_t offset{0U};
        for (std::size_t i = 0; i < parity.size(); i++) {
            offset |= ((k << i) & parity[i]);
        }
        return offset;
    }

    /* Matrix gates */

    /**
     * @brief Apply a single qubit gate to the statevector.
     *
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wire A wire the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT>
    static void
    applyNCSingleQubitOp(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                         const std::complex<PrecisionT> *matrix,
                         const std::vector<std::size_t> &controlled_wires,
                         const std::vector<bool> &controlled_values,
                         const std::vector<std::size_t> &wires,
                         bool inverse = false) {
        constexpr std::size_t one{1U};
        constexpr std::size_t dim = one << 1U;
        std::vector<std::complex<PrecisionT>> mat(matrix, matrix + dim * dim);
        if (inverse) {
            for (std::size_t i = 0; i < dim; i++) {
                for (std::size_t j = 0; j < dim; j++) {
                    mat[j * dim + i] = std::conj(matrix[i * dim + j]);
                }
            }
        }
        auto core_function = [&mat](std::complex<PrecisionT> *arr,
                                    const std::size_t i0,
                                    const std::size_t i1) {
            const std::complex<PrecisionT> v0 = arr[i0];
            const std::complex<PrecisionT> v1 = arr[i1];
            arr[i0] = mat[0B00] * v0 +
                      mat[0B01] * v1; // NOLINT(readability-magic-numbers)
            arr[i1] = mat[0B10] * v0 +
                      mat[0B11] * v1; // NOLINT(readability-magic-numbers)
        };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, PrecisionT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, PrecisionT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT>
    static void applySingleQubitOp(std::complex<PrecisionT> *arr,
                                   std::size_t num_qubits,
                                   const std::complex<PrecisionT> *matrix,
                                   const std::vector<std::size_t> &wires,
                                   bool inverse = false) {
        applyNCSingleQubitOp(arr, num_qubits, matrix, {}, {}, wires, inverse);
    }

    /**
     * @brief Apply a two qubit gate to the statevector.
     *
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT>
    static void
    applyNCTwoQubitOp(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                      const std::complex<PrecisionT> *matrix,
                      const std::vector<std::size_t> &controlled_wires,
                      const std::vector<bool> &controlled_values,
                      const std::vector<std::size_t> &wires,
                      bool inverse = false) {
        constexpr std::size_t one{1U};
        constexpr std::size_t dim = one << 2U;
        std::vector<std::complex<PrecisionT>> mat(matrix, matrix + dim * dim);
        if (inverse) {
            for (std::size_t i = 0; i < dim; i++) {
                for (std::size_t j = 0; j < dim; j++) {
                    mat[j * dim + i] = std::conj(matrix[i * dim + j]);
                }
            }
        }
        auto core_function =
            [&mat](std::complex<PrecisionT> *arr, const std::size_t i00,
                   const std::size_t i01, const std::size_t i10,
                   const std::size_t i11) {
                const std::complex<PrecisionT> v00 = arr[i00];
                const std::complex<PrecisionT> v01 = arr[i01];
                const std::complex<PrecisionT> v10 = arr[i10];
                const std::complex<PrecisionT> v11 = arr[i11];
                // NOLINTBEGIN(readability-magic-numbers)
                arr[i00] = mat[0B0000] * v00 + mat[0B0001] * v01 +
                           mat[0B0010] * v10 + mat[0B0011] * v11;
                arr[i01] = mat[0B0100] * v00 + mat[0B0101] * v01 +
                           mat[0B0110] * v10 + mat[0B0111] * v11;
                arr[i10] = mat[0B1000] * v00 + mat[0B1001] * v01 +
                           mat[0B1010] * v10 + mat[0B1011] * v11;
                arr[i11] = mat[0B1100] * v00 + mat[0B1101] * v01 +
                           mat[0B1110] * v10 + mat[0B1111] * v11;
                // NOLINTEND(readability-magic-numbers)
            };
        if (controlled_wires.empty()) {
            applyNC2<PrecisionT, PrecisionT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC2<PrecisionT, PrecisionT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT>
    static void applyTwoQubitOp(std::complex<PrecisionT> *arr,
                                std::size_t num_qubits,
                                const std::complex<PrecisionT> *matrix,
                                const std::vector<std::size_t> &wires,
                                bool inverse = false) {
        applyNCTwoQubitOp(arr, num_qubits, matrix, {}, {}, wires, inverse);
    }

    /**
     * @brief Apply a matrix with controls to the statevector.
     *
     * @tparam PrecisionT Floating point precision of underlying statevector
     * data
     * @tparam FuncT Gate function type.
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param core_function Gate function returning the matrix-vector product
     * for a wire block.
     */
    template <class PrecisionT, class FuncT>
    static void applyNCN(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                         const std::vector<std::size_t> &controlled_wires,
                         const std::vector<bool> &controlled_values,
                         const std::vector<std::size_t> &wires,
                         FuncT core_function) {
        const std::size_t n_contr = controlled_wires.size();
        const std::size_t n_wires = wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(num_qubits >= nw_tot);
        PL_ABORT_IF_NOT(controlled_values.size() == n_contr,
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");

        std::vector<std::size_t> all_wires;
        all_wires.reserve(nw_tot);
        all_wires.insert(all_wires.begin(), wires.begin(), wires.end());
        all_wires.insert(all_wires.begin() + wires.size(),
                         controlled_wires.begin(), controlled_wires.end());

        const auto revs = reverseWires(num_qubits, all_wires, {});
        const auto &rev_wires = revs.first;
        const std::vector<std::size_t> parity =
            Pennylane::Util::revWireParity(rev_wires);
        std::vector<std::size_t> indices =
            Gates::generateBitPatterns(wires, num_qubits);
        Gates::controlBitPatterns(indices, num_qubits, controlled_wires,
                                  controlled_values);
        PL_LOOP_PARALLEL(1)
        for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
            const std::size_t offset = parity_2_offset(parity, k);
            core_function(arr, indices, offset);
        }
    }

    /**
     * @brief Apply a matrix with controls to the statevector.
     *
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT>
    static void
    applyNCMultiQubitOp(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                        const std::complex<PrecisionT> *matrix,
                        const std::vector<std::size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<std::size_t> &wires, bool inverse) {
        constexpr std::size_t one{1U};
        const std::size_t n_wires = wires.size();
        const std::size_t dim = one << n_wires;
        std::vector<std::complex<PrecisionT>> mat(matrix, matrix + dim * dim);
        if (inverse) {
            for (std::size_t i = 0; i < dim; i++) {
                for (std::size_t j = 0; j < dim; j++) {
                    mat[j * dim + i] = std::conj(matrix[i * dim + j]);
                }
            }
        }
        auto core_function = [dim,
                              &mat](std::complex<PrecisionT> *arr,
                                    const std::vector<std::size_t> &indices,
                                    const std::size_t offset) {
            std::vector<std::complex<PrecisionT>> coeffs(dim);
            for (std::size_t i = 0; i < dim; i++) {
                coeffs[i] = arr[indices[i] + offset];
            }
            for (std::size_t i = 0; i < dim; i++) {
                const auto index = indices[i] + offset;
                const std::size_t base_idx = i * dim;
                arr[index] = 0.0;
                for (std::size_t j = 0; j < dim; j++) {
                    arr[index] += mat[base_idx + j] * coeffs[j];
                }
            }
        };
        applyNCN(arr, num_qubits, controlled_wires, controlled_values, wires,
                 core_function);
    }

    template <class PrecisionT>
    static void
    applyMultiQubitOp(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                      const std::complex<PrecisionT> *matrix,
                      const std::vector<std::size_t> &wires, bool inverse) {
        applyNCMultiQubitOp(arr, num_qubits, matrix, {}, {}, wires, inverse);
    }

    /**
     * @brief Apply a sparse matrix with controls to the statevector.
     * @tparam PrecisionT Floating point precision of underlying statevector
     * data
     * @tparam IndexT Index type
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param row_map_ptr Pointer to the row map.
     * @param col_idx_ptr Pointer to the columns indexes
     * @param values_ptr Pointer to the values.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT, class IndexT = std::size_t>
    static void applyNCMultiQubitSparseOp(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const IndexT *row_map_ptr, const IndexT *col_idx_ptr,
        const std::complex<PrecisionT> *values_ptr,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, bool inverse = false) {
        constexpr std::size_t one{1U};
        const std::size_t n_wires = wires.size();
        const std::size_t dim = one << n_wires;

        PL_ABORT_IF(inverse, "Inverse not implemented for sparse ops.");

        auto core_function = [dim, &row_map_ptr, &col_idx_ptr, &values_ptr](
                                 std::complex<PrecisionT> *arr,
                                 const std::vector<std::size_t> &indices,
                                 const std::size_t offset) {
            std::vector<std::complex<PrecisionT>> coeffs(dim);
            for (std::size_t i = 0; i < dim; i++) {
                coeffs[i] = arr[indices[i] + offset];
            }
            std::size_t count = 0;
            for (IndexT i = 0; i < dim; i++) {
                const auto index = indices[i] + offset;
                arr[index] = 0.0;
                auto row_range = row_map_ptr[i + 1] - row_map_ptr[i];
                for (IndexT j = 0; j < row_range; j++) {
                    arr[index] +=
                        values_ptr[count] * coeffs[col_idx_ptr[count]];
                    count++;
                }
            }
        };
        applyNCN(arr, num_qubits, controlled_wires, controlled_values, wires,
                 core_function);
    }

    /**
     * @brief Apply a sparse matrix to the statevector.
     * @tparam PrecisionT Floating point precision of underlying statevector
     * data
     * @tparam IndexT Index type
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param row_map_ptr Pointer to the row map.
     * @param col_idx_ptr Pointer to the columns indexes
     * @param values_ptr Pointer to the values.
     * @param wires Wires the gate applies to.
     */
    template <class PrecisionT, class IndexT = std::size_t>
    static void applyMultiQubitSparseOp(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const IndexT *row_map_ptr, const IndexT *col_idx_ptr,
        const std::complex<PrecisionT> *values_ptr,
        const std::vector<std::size_t> &wires, bool inverse = false) {
        applyNCMultiQubitSparseOp(arr, num_qubits, row_map_ptr, col_idx_ptr,
                                  values_ptr, {}, {}, wires, inverse);
    }

    template <class PrecisionT>
    static void
    applyPauliRot(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                  const std::vector<std::size_t> &wires, const bool inverse,
                  PrecisionT angle, const std::string &word) {
        using ComplexT = std::complex<PrecisionT>;
        constexpr auto IMAG = Pennylane::Util::IMAG<PrecisionT>();
        PL_ABORT_IF_NOT(wires.size() == word.size(),
                        "wires and word have incompatible dimensions.")
        if (std::find_if_not(word.begin(), word.end(), [](const int w) {
                return w == 'Z';
            }) == word.end()) {
            applyMultiRZ(arr, num_qubits, wires, inverse, angle);
            return;
        }
        const PrecisionT c = std::cos(angle / 2);
        const ComplexT s = ((inverse) ? IMAG : -IMAG) * std::sin(angle / 2);
        const std::array<ComplexT, 4> sines{s, IMAG * s, -s, -IMAG * s};

        auto get_mask =
            [num_qubits,
             &wires]([[maybe_unused]] const std::function<bool(const int)>
                         &condition) {
                std::size_t mask{0U};
                for (std::size_t iw = 0; iw < wires.size(); iw++) {
                    const auto bit = static_cast<std::size_t>(condition(iw));
                    mask |= bit << (num_qubits - 1 - wires[iw]);
                }
                return mask;
            };
        const std::size_t mask_xy =
            get_mask([&word](const int a) { return word[a] != 'Z'; });
        const std::size_t mask_y =
            get_mask([&word](const int a) { return word[a] == 'Y'; });
        const std::size_t mask_z =
            get_mask([&word](const int a) { return word[a] == 'Z'; });
        const auto count_mask_y = std::popcount(mask_y);
        const std::size_t sv_length = exp2(num_qubits);

        PL_LOOP_PARALLEL(1)
        for (std::size_t i0 = 0; i0 < sv_length; i0++) {
            const std::size_t i1 = i0 ^ mask_xy;
            if (i0 > i1) {
                continue;
            }
            const auto count_y = std::popcount(i0 & mask_y) * 2;
            const auto count_z = std::popcount(i0 & mask_z) * 2;
            const auto sign_i0 = count_z + count_mask_y * 3 - count_y;
            const auto sign_i1 = count_z + count_mask_y + count_y;
            const ComplexT v0 = arr[i0];
            const ComplexT v1 = arr[i1];
            arr[i0] = c * v0 + sines[sign_i0 % 4] * v1;
            arr[i1] = c * v1 + sines[sign_i1 % 4] * v0;
        }
    }

    /* One-qubit gates */

    /**
     * @brief Apply a single-qubit gate with controls to the statevector.
     *
     * @tparam PrecisionT Floating point precision of underlying statevector
     * data
     * @tparam ParamT Floating point precision of the gate parameters
     * @tparam FuncT Gate function type.
     * @tparam has_controls True if the gate has control wires and false
     * otherwise (for optimization)
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param core_function Gate function returning the matrix-vector
     * product for a wire block.
     */
    template <class PrecisionT, class ParamT = PrecisionT, class FuncT,
              bool has_controls = true>
    static void
    applyNC1(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
             const std::vector<std::size_t> &controlled_wires,
             const std::vector<bool> &controlled_values,
             const std::vector<std::size_t> &wires, FuncT core_function) {
        constexpr std::size_t one{1U};
        const std::size_t n_contr = controlled_wires.size();
        const std::size_t n_wires = wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(n_wires == 1);
        PL_ASSERT(num_qubits >= nw_tot);

        if constexpr (has_controls) {
            PL_ABORT_IF_NOT(controlled_values.size() == n_contr,
                            "`controlled_wires` must have the same size as "
                            "`controlled_values`.");

            std::vector<std::size_t> all_wires;
            all_wires.reserve(nw_tot);
            all_wires.insert(all_wires.begin(), wires.begin(), wires.end());
            all_wires.insert(all_wires.begin() + wires.size(),
                             controlled_wires.begin(), controlled_wires.end());
            const auto revs = reverseWires(num_qubits, all_wires, {});
            const auto &rev_wires = revs.first;
            const std::vector<std::size_t> parity =
                Pennylane::Util::revWireParity(rev_wires);
            std::vector<std::size_t> indices =
                Gates::generateBitPatterns(wires, num_qubits);
            Gates::controlBitPatterns(indices, num_qubits, controlled_wires,
                                      controlled_values);
            const std::size_t i0 = indices[0];
            const std::size_t i1 = indices[1];
            PL_LOOP_PARALLEL(1)
            for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
                const std::size_t offset = parity_2_offset(parity, k);
                core_function(arr, i0 + offset, i1 + offset);
            }
        } else {
            const std::size_t rev_wire = num_qubits - wires[0] - 1;
            const std::size_t rev_wire_shift = (one << rev_wire);
            const auto parities = revWireParity(rev_wire);
            const auto parity_high = parities.first;
            const auto parity_low = parities.second;
            PL_LOOP_PARALLEL(1)
            for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
                const std::size_t i0 =
                    ((k << 1U) & parity_high) | (parity_low & k);
                const std::size_t i1 = i0 | rev_wire_shift;
                core_function(arr, i0, i1);
            }
        }
    }

    template <class PrecisionT>
    static void applyNCPauliX(std::complex<PrecisionT> *arr,
                              const std::size_t num_qubits,
                              const std::vector<std::size_t> &controlled_wires,
                              const std::vector<bool> &controlled_values,
                              const std::vector<std::size_t> &wires,
                              [[maybe_unused]] const bool inverse) {
        using ParamT = PrecisionT;
        auto core_function = [](std::complex<PrecisionT> *arr,
                                const std::size_t i0, const std::size_t i1) {
            std::swap(arr[i0], arr[i1]);
        };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT>
    static void
    applyPauliX(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
                const std::vector<std::size_t> &wires, const bool inverse) {
        applyNCPauliX(arr, num_qubits, {}, {}, wires, inverse);
    }

    template <class PrecisionT>
    static void applyCNOT(std::complex<PrecisionT> *arr,
                          const std::size_t num_qubits,
                          const std::vector<std::size_t> &wires,
                          [[maybe_unused]] const bool inverse) {
        using ParamT = PrecisionT;
        auto core_function = [](std::complex<PrecisionT> *arr,
                                [[maybe_unused]] const std::size_t i00,
                                [[maybe_unused]] const std::size_t i01,
                                const std::size_t i10, const std::size_t i11) {
            std::swap(arr[i10], arr[i11]);
        };
        applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
            arr, num_qubits, {}, {}, wires, core_function);
    }

    template <class PrecisionT>
    static void applyToffoli(std::complex<PrecisionT> *arr,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &wires,
                             [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 3);
        applyNCPauliX(arr, num_qubits, {wires[0], wires[1]}, {true, true},
                      {wires[2]}, inverse);
    }

    template <class PrecisionT>
    static void applyNCPauliY(std::complex<PrecisionT> *arr,
                              const std::size_t num_qubits,
                              const std::vector<std::size_t> &controlled_wires,
                              const std::vector<bool> &controlled_values,
                              const std::vector<std::size_t> &wires,
                              [[maybe_unused]] const bool inverse) {
        using ParamT = PrecisionT;
        auto core_function = [](std::complex<PrecisionT> *arr,
                                const std::size_t i0, const std::size_t i1) {
            const auto v0 = arr[i0];
            const auto v1 = arr[i1];
            arr[i0] = {std::imag(v1), -std::real(v1)};
            arr[i1] = {-std::imag(v0), std::real(v0)};
        };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT>
    static void
    applyPauliY(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
                const std::vector<std::size_t> &wires, const bool inverse) {
        applyNCPauliY(arr, num_qubits, {}, {}, wires, inverse);
    }

    template <class PrecisionT>
    static void applyCY(std::complex<PrecisionT> *arr,
                        const std::size_t num_qubits,
                        const std::vector<std::size_t> &wires,
                        [[maybe_unused]] const bool inverse) {
        using ParamT = PrecisionT;
        auto core_function = [](std::complex<PrecisionT> *arr,
                                [[maybe_unused]] const std::size_t i00,
                                [[maybe_unused]] const std::size_t i01,
                                const std::size_t i10, const std::size_t i11) {
            const auto v0 = arr[i10];
            const auto v1 = arr[i11];
            arr[i10] = {std::imag(v1), -std::real(v1)};
            arr[i11] = {-std::imag(v0), std::real(v0)};
        };
        applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
            arr, num_qubits, {}, {}, wires, core_function);
    }

    template <class PrecisionT>
    static void applyNCPauliZ(std::complex<PrecisionT> *arr,
                              const std::size_t num_qubits,
                              const std::vector<std::size_t> &controlled_wires,
                              const std::vector<bool> &controlled_values,
                              const std::vector<std::size_t> &wires,
                              [[maybe_unused]] const bool inverse) {
        using ParamT = PrecisionT;
        auto core_function = [](std::complex<PrecisionT> *arr,
                                [[maybe_unused]] const std::size_t i0,
                                const std::size_t i1) { arr[i1] *= -1; };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT>
    static void
    applyPauliZ(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
                const std::vector<std::size_t> &wires, const bool inverse) {
        applyNCPauliZ(arr, num_qubits, {}, {}, wires, inverse);
    }

    template <class PrecisionT>
    static void applyCZ(std::complex<PrecisionT> *arr,
                        const std::size_t num_qubits,
                        const std::vector<std::size_t> &wires,
                        [[maybe_unused]] const bool inverse) {
        using ParamT = PrecisionT;
        auto core_function = [](std::complex<PrecisionT> *arr,
                                [[maybe_unused]] const std::size_t i00,
                                [[maybe_unused]] const std::size_t i01,
                                [[maybe_unused]] const std::size_t i10,
                                const std::size_t i11) { arr[i11] *= -1; };
        applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
            arr, num_qubits, {}, {}, wires, core_function);
    }

    template <class PrecisionT>
    static void
    applyNCHadamard(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
                    const std::vector<std::size_t> &controlled_wires,
                    const std::vector<bool> &controlled_values,
                    const std::vector<std::size_t> &wires,
                    [[maybe_unused]] const bool inverse) {
        using ParamT = PrecisionT;
        constexpr static auto isqrt2 = INVSQRT2<PrecisionT>();
        auto core_function = [](std::complex<PrecisionT> *arr,
                                const std::size_t i0, const std::size_t i1) {
            const std::complex<PrecisionT> v0 = arr[i0];
            const std::complex<PrecisionT> v1 = arr[i1];
            arr[i0] = isqrt2 * v0 + isqrt2 * v1;
            arr[i1] = isqrt2 * v0 - isqrt2 * v1;
        };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT>
    static void
    applyHadamard(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
                  const std::vector<std::size_t> &wires, const bool inverse) {
        applyNCHadamard(arr, num_qubits, {}, {}, wires, inverse);
    }

    template <class PrecisionT>
    static void
    applyNCS(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
             const std::vector<std::size_t> &controlled_wires,
             const std::vector<bool> &controlled_values,
             const std::vector<std::size_t> &wires, const bool inverse) {
        using ParamT = PrecisionT;
        const std::complex<PrecisionT> shift =
            (inverse) ? -Pennylane::Util::IMAG<PrecisionT>()
                      : Pennylane::Util::IMAG<PrecisionT>();

        auto core_function = [&shift](std::complex<PrecisionT> *arr,
                                      [[maybe_unused]] const std::size_t i0,
                                      const std::size_t i1) {
            arr[i1] *= shift;
        };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT>
    static void
    applyS(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
           const std::vector<std::size_t> &wires, const bool inverse) {
        applyNCS(arr, num_qubits, {}, {}, wires, inverse);
    }

    template <class PrecisionT>
    static void
    applyNCT(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
             const std::vector<std::size_t> &controlled_wires,
             const std::vector<bool> &controlled_values,
             const std::vector<std::size_t> &wires, const bool inverse) {
        using ParamT = PrecisionT;
        constexpr static auto isqrt2 = INVSQRT2<PrecisionT>();
        const std::complex<PrecisionT> shift = {isqrt2,
                                                inverse ? -isqrt2 : isqrt2};
        auto core_function = [&shift](std::complex<PrecisionT> *arr,
                                      [[maybe_unused]] const std::size_t i0,
                                      const std::size_t i1) {
            arr[i1] *= shift;
        };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT>
    static void
    applyT(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
           const std::vector<std::size_t> &wires, const bool inverse) {
        applyNCT(arr, num_qubits, {}, {}, wires, inverse);
    }

    template <class PrecisionT>
    static void
    applyNCSX(std::complex<PrecisionT> *arr, const size_t num_qubits,
              const std::vector<std::size_t> &controlled_wires,
              const std::vector<bool> &controlled_values,
              const std::vector<std::size_t> &wires, const bool inverse) {
        using ParamT = PrecisionT;

        const PrecisionT shift = (inverse) ? -1.0 : 1.0;

        auto core_function = [shift](std::complex<PrecisionT> *arr,
                                     const std::size_t i0,
                                     const std::size_t i1) {
            const std::complex<PrecisionT> v0 = arr[i0];
            const std::complex<PrecisionT> v1 = arr[i1];

            const PrecisionT vr_plus = v0.real() + v1.real();
            const PrecisionT vi_plus = v0.imag() + v1.imag();

            const PrecisionT vr_minus = (v0.real() - v1.real()) * shift;
            const PrecisionT vi_minus = (-v0.imag() + v1.imag()) * shift;

            arr[i0] = std::complex<PrecisionT>(vr_plus + vi_minus,
                                               vi_plus + vr_minus) *
                      PrecisionT(0.5);
            arr[i1] = std::complex<PrecisionT>(vr_plus - vi_minus,
                                               vi_plus - vr_minus) *
                      PrecisionT(0.5);
        };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT>
    static void applySX(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, const bool inverse) {
        applyNCSX(arr, num_qubits, {}, {}, wires, inverse);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void
    applyNCPhaseShift(std::complex<PrecisionT> *arr,
                      const std::size_t num_qubits,
                      const std::vector<std::size_t> &controlled_wires,
                      const std::vector<bool> &controlled_values,
                      const std::vector<std::size_t> &wires, const bool inverse,
                      ParamT angle) {
        const std::complex<PrecisionT> s =
            inverse ? std::exp(-std::complex<PrecisionT>(0, angle))
                    : std::exp(std::complex<PrecisionT>(0, angle));

        auto core_function = [&s](std::complex<PrecisionT> *arr,
                                  [[maybe_unused]] const std::size_t i0,
                                  const std::size_t i1) { arr[i1] *= s; };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyPhaseShift(std::complex<PrecisionT> *arr,
                                const std::size_t num_qubits,
                                const std::vector<std::size_t> &wires,
                                const bool inverse, ParamT angle) {
        applyNCPhaseShift(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyControlledPhaseShift(std::complex<PrecisionT> *arr,
                                          const std::size_t num_qubits,
                                          const std::vector<std::size_t> &wires,
                                          [[maybe_unused]] bool inverse,
                                          ParamT angle) {
        const std::complex<PrecisionT> s =
            inverse ? std::exp(-std::complex<PrecisionT>(0, angle))
                    : std::exp(std::complex<PrecisionT>(0, angle));
        auto core_function = [&s](std::complex<PrecisionT> *arr,
                                  [[maybe_unused]] const std::size_t i00,
                                  [[maybe_unused]] const std::size_t i01,
                                  [[maybe_unused]] const std::size_t i10,
                                  const std::size_t i11) { arr[i11] *= s; };
        applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
            arr, num_qubits, {}, {}, wires, core_function);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyNCRX(std::complex<PrecisionT> *arr,
                          const std::size_t num_qubits,
                          const std::vector<std::size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<std::size_t> &wires,
                          const bool inverse, ParamT angle) {
        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);
        auto core_function = [c, js](std::complex<PrecisionT> *arr,
                                     const std::size_t i0,
                                     const std::size_t i1) {
            const std::complex<PrecisionT> v0 = arr[i0];
            const std::complex<PrecisionT> v1 = arr[i1];
            arr[i0] = c * v0 +
                      std::complex<PrecisionT>{-imag(v1) * js, real(v1) * js};
            arr[i1] = std::complex<PrecisionT>{-imag(v0) * js, real(v0) * js} +
                      c * v1;
        };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRX(std::complex<PrecisionT> *arr,
                        const std::size_t num_qubits,
                        const std::vector<std::size_t> &wires,
                        const bool inverse, ParamT angle) {
        applyNCRX(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyCRX(std::complex<PrecisionT> *arr,
                         const std::size_t num_qubits,
                         const std::vector<std::size_t> &wires,
                         const bool inverse, ParamT angle) {
        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);
        auto core_function = [c, js](std::complex<PrecisionT> *arr,
                                     [[maybe_unused]] const std::size_t i00,
                                     [[maybe_unused]] const std::size_t i01,
                                     const std::size_t i10,
                                     const std::size_t i11) {
            const std::complex<PrecisionT> v0 = arr[i10];
            const std::complex<PrecisionT> v1 = arr[i11];
            arr[i10] = c * v0 +
                       std::complex<PrecisionT>{-imag(v1) * js, real(v1) * js};
            arr[i11] = std::complex<PrecisionT>{-imag(v0) * js, real(v0) * js} +
                       c * v1;
        };
        applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
            arr, num_qubits, {}, {}, wires, core_function);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyNCRY(std::complex<PrecisionT> *arr,
                          const std::size_t num_qubits,
                          const std::vector<std::size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<std::size_t> &wires,
                          const bool inverse, ParamT angle) {
        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        auto core_function = [c, s](std::complex<PrecisionT> *arr,
                                    const std::size_t i0,
                                    const std::size_t i1) {
            const std::complex<PrecisionT> v0 = arr[i0];
            const std::complex<PrecisionT> v1 = arr[i1];
            arr[i0] = std::complex<PrecisionT>{c * real(v0) - s * real(v1),
                                               c * imag(v0) - s * imag(v1)};
            arr[i1] = std::complex<PrecisionT>{s * real(v0) + c * real(v1),
                                               s * imag(v0) + c * imag(v1)};
        };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRY(std::complex<PrecisionT> *arr,
                        const std::size_t num_qubits,
                        const std::vector<std::size_t> &wires,
                        const bool inverse, ParamT angle) {
        applyNCRY(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyCRY(std::complex<PrecisionT> *arr,
                         const std::size_t num_qubits,
                         const std::vector<std::size_t> &wires,
                         const bool inverse, ParamT angle) {
        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
        auto core_function = [c, s](std::complex<PrecisionT> *arr,
                                    [[maybe_unused]] const std::size_t i00,
                                    [[maybe_unused]] const std::size_t i01,
                                    const std::size_t i10,
                                    const std::size_t i11) {
            const std::complex<PrecisionT> v0 = arr[i10];
            const std::complex<PrecisionT> v1 = arr[i11];
            arr[i10] = std::complex<PrecisionT>{c * real(v0) - s * real(v1),
                                                c * imag(v0) - s * imag(v1)};
            arr[i11] = std::complex<PrecisionT>{s * real(v0) + c * real(v1),
                                                s * imag(v0) + c * imag(v1)};
        };
        applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
            arr, num_qubits, {}, {}, wires, core_function);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyNCRZ(std::complex<PrecisionT> *arr,
                          const std::size_t num_qubits,
                          const std::vector<std::size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<std::size_t> &wires,
                          const bool inverse, ParamT angle) {
        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};
        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};

        auto core_function = [&shifts](std::complex<PrecisionT> *arr,
                                       const std::size_t i0,
                                       const std::size_t i1) {
            arr[i0] *= shifts[0];
            arr[i1] *= shifts[1];
        };
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC1<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr,
                        const std::size_t num_qubits,
                        const std::vector<std::size_t> &wires,
                        const bool inverse, ParamT angle) {
        applyNCRZ(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyCRZ(std::complex<PrecisionT> *arr,
                         const std::size_t num_qubits,
                         const std::vector<std::size_t> &wires,
                         const bool inverse, ParamT angle) {
        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};
        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};
        auto core_function = [&shifts](std::complex<PrecisionT> *arr,
                                       [[maybe_unused]] const std::size_t i00,
                                       [[maybe_unused]] const std::size_t i01,
                                       const std::size_t i10,
                                       const std::size_t i11) {
            arr[i10] *= shifts[0];
            arr[i11] *= shifts[1];
        };
        applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
            arr, num_qubits, {}, {}, wires, core_function);
    }

    template <class PrecisionT>
    static void applyIdentity(std::complex<PrecisionT> *arr,
                              const std::size_t num_qubits,
                              const std::vector<std::size_t> &wires,
                              [[maybe_unused]] const bool inverse) {
        static_cast<void>(arr);        // No-op
        static_cast<void>(num_qubits); // No-op
        static_cast<void>(wires);      // No-op
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void
    applyNCRot(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
               const std::vector<std::size_t> &controlled_wires,
               const std::vector<bool> &controlled_values,
               const std::vector<std::size_t> &wires, const bool inverse,
               ParamT phi, ParamT theta, ParamT omega) {
        PL_ASSERT(wires.size() == 1);
        const auto rotMat = getRot<std::complex, PrecisionT>(phi, theta, omega);
        applyNCSingleQubitOp(arr, num_qubits, rotMat.data(), controlled_wires,
                             controlled_values, wires, inverse);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRot(std::complex<PrecisionT> *arr,
                         const std::size_t num_qubits,
                         const std::vector<std::size_t> &wires, bool inverse,
                         ParamT phi, ParamT theta, ParamT omega) {
        PL_ASSERT(wires.size() == 1);
        const auto rotMat =
            (inverse) ? getRot<std::complex, PrecisionT>(-omega, -theta, -phi)
                      : getRot<std::complex, PrecisionT>(phi, theta, omega);
        applySingleQubitOp(arr, num_qubits, rotMat.data(), wires);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRot(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                          const std::vector<std::size_t> &wires, bool inverse,
                          ParamT phi, ParamT theta, ParamT omega) {
        const auto rotMat =
            (inverse) ? getRot<std::complex, PrecisionT>(-omega, -theta, -phi)
                      : getRot<std::complex, PrecisionT>(phi, theta, omega);
        auto core_function = [&rotMat](std::complex<PrecisionT> *arr,
                                       [[maybe_unused]] const std::size_t i00,
                                       [[maybe_unused]] const std::size_t i01,
                                       const std::size_t i10,
                                       const std::size_t i11) {
            const std::complex<PrecisionT> v0 = arr[i10];
            const std::complex<PrecisionT> v1 = arr[i11];
            arr[i10] = rotMat[0] * v0 + rotMat[1] * v1;
            arr[i11] = rotMat[2] * v0 + rotMat[3] * v1;
        };
        applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
            arr, num_qubits, {}, {}, wires, core_function);
    }

    /* Two-qubit gates */

    /**
     * @brief Apply a two-qubit gate with controls to the statevector.
     *
     * @tparam PrecisionT Floating point precision of underlying statevector
     * data
     * @tparam ParamT Floating point precision of the gate parameters
     * @tparam FuncT Gate function type.
     * @tparam has_controls True if the gate has control wires and false
     * otherwise (for optimization)
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param core_function Gate function returning the matrix-vector
     * product for a wire block.
     */
    template <class PrecisionT, class ParamT = PrecisionT, class FuncT,
              bool has_controls = true>
    static void
    applyNC2(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
             const std::vector<std::size_t> &controlled_wires,
             const std::vector<bool> &controlled_values,
             const std::vector<std::size_t> &wires, FuncT core_function) {
        const std::size_t n_contr = controlled_wires.size();
        const std::size_t n_wires = wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(n_wires == 2);
        PL_ASSERT(num_qubits >= nw_tot);
        if constexpr (has_controls) {
            PL_ABORT_IF_NOT(controlled_values.size() == n_contr,
                            "`controlled_wires` must have the same size as "
                            "`controlled_values`.");

            std::vector<std::size_t> all_wires;
            all_wires.reserve(nw_tot);
            all_wires.insert(all_wires.begin(), wires.begin(), wires.end());
            all_wires.insert(all_wires.begin() + wires.size(),
                             controlled_wires.begin(), controlled_wires.end());
            const auto revs = reverseWires(num_qubits, all_wires, {});
            const auto &rev_wires = revs.first;
            const std::vector<std::size_t> parity =
                Pennylane::Util::revWireParity(rev_wires);
            std::vector<std::size_t> indices =
                Gates::generateBitPatterns(wires, num_qubits);
            Gates::controlBitPatterns(indices, num_qubits, controlled_wires,
                                      controlled_values);
            const std::size_t i00 = indices[0B00];
            const std::size_t i01 = indices[0B01];
            const std::size_t i10 = indices[0B10];
            const std::size_t i11 = indices[0B11];
            PL_LOOP_PARALLEL(1)
            for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
                const std::size_t offset = parity_2_offset(parity, k);
                core_function(arr, i00 + offset, i01 + offset, i10 + offset,
                              i11 + offset);
            }
        } else {
            const std::size_t rev_wire0 = num_qubits - wires[1] - 1;
            const std::size_t rev_wire1 =
                num_qubits - wires[0] - 1; // Control qubit
            const std::size_t rev_wire0_shift = static_cast<std::size_t>(1U)
                                                << rev_wire0;
            const std::size_t rev_wire1_shift = static_cast<std::size_t>(1U)
                                                << rev_wire1;
            const auto parities = revWireParity(rev_wire0, rev_wire1);
            const auto parity_high = std::get<0>(parities);
            const auto parity_middle = std::get<1>(parities);
            const auto parity_low = std::get<2>(parities);
            PL_LOOP_PARALLEL(1)
            for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
                const std::size_t i00 = ((k << 2U) & parity_high) |
                                        ((k << 1U) & parity_middle) |
                                        (k & parity_low);
                const std::size_t i10 = i00 | rev_wire1_shift;
                const std::size_t i01 = i00 | rev_wire0_shift;
                const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;
                core_function(arr, i00, i01, i10, i11);
            }
        }
    }

    template <class PrecisionT>
    static void applyNCSWAP(std::complex<PrecisionT> *arr,
                            std::size_t num_qubits,
                            const std::vector<std::size_t> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<std::size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ParamT = PrecisionT;
        auto core_function = [](std::complex<PrecisionT> *arr,
                                [[maybe_unused]] const std::size_t i00,
                                const std::size_t i01, const std::size_t i10,
                                [[maybe_unused]] const std::size_t i11) {
            std::swap(arr[i10], arr[i01]);
        };
        if (controlled_wires.empty()) {
            applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC2<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT>
    static void applySWAP(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                          const std::vector<std::size_t> &wires,
                          [[maybe_unused]] bool inverse) {
        applyNCSWAP(arr, num_qubits, {}, {}, wires, inverse);
    }

    template <class PrecisionT>
    static void applyCSWAP(std::complex<PrecisionT> *arr,
                           std::size_t num_qubits,
                           const std::vector<std::size_t> &wires,
                           [[maybe_unused]] bool inverse) {
        applyNCSWAP(arr, num_qubits, {wires[0]}, {true}, {wires[1], wires[2]},
                    inverse);
    }

    template <class PrecisionT, class ParamT>
    static void applyNCIsingXX(std::complex<PrecisionT> *arr,
                               std::size_t num_qubits,
                               const std::vector<std::size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values,
                               const std::vector<std::size_t> &wires,
                               bool inverse, ParamT angle) {
        using ComplexT = std::complex<PrecisionT>;
        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        auto core_function =
            [cr, sj](std::complex<PrecisionT> *arr, const std::size_t i00,
                     const std::size_t i01, const std::size_t i10,
                     const std::size_t i11) {
                const ComplexT v00 = arr[i00];
                const ComplexT v01 = arr[i01];
                const ComplexT v10 = arr[i10];
                const ComplexT v11 = arr[i11];
                arr[i00] = ComplexT{cr * v00.real() + sj * v11.imag(),
                                    cr * v00.imag() - sj * v11.real()};
                arr[i01] = ComplexT{cr * v01.real() + sj * v10.imag(),
                                    cr * v01.imag() - sj * v10.real()};
                arr[i10] = ComplexT{cr * v10.real() + sj * v01.imag(),
                                    cr * v10.imag() - sj * v01.real()};
                arr[i11] = ComplexT{cr * v11.real() + sj * v00.imag(),
                                    cr * v11.imag() - sj * v00.real()};
            };
        if (controlled_wires.empty()) {
            applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC2<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyIsingXX(std::complex<PrecisionT> *arr,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &wires,
                             bool inverse, ParamT angle) {
        applyNCIsingXX(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyNCIsingXY(std::complex<PrecisionT> *arr,
                               const std::size_t num_qubits,
                               const std::vector<std::size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values,
                               const std::vector<std::size_t> &wires,
                               bool inverse, ParamT angle) {
        using ComplexT = std::complex<PrecisionT>;
        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        auto core_function =
            [cr, sj](std::complex<PrecisionT> *arr, const std::size_t i00,
                     const std::size_t i01, const std::size_t i10,
                     const std::size_t i11) {
                const ComplexT v00 = arr[i00];
                const ComplexT v01 = arr[i01];
                const ComplexT v10 = arr[i10];
                const ComplexT v11 = arr[i11];
                arr[i00] = ComplexT{v00.real(), v00.imag()};
                arr[i01] = ComplexT{cr * v01.real() - sj * v10.imag(),
                                    cr * v01.imag() + sj * v10.real()};
                arr[i10] = ComplexT{cr * v10.real() - sj * v01.imag(),
                                    cr * v10.imag() + sj * v01.real()};
                arr[i11] = ComplexT{v11.real(), v11.imag()};
            };
        if (controlled_wires.empty()) {
            applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC2<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyIsingXY(std::complex<PrecisionT> *arr,
                             const std::size_t num_qubits,
                             const std::vector<std::size_t> &wires,
                             bool inverse, ParamT angle) {
        applyNCIsingXY(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyNCIsingYY(std::complex<PrecisionT> *arr,
                               std::size_t num_qubits,
                               const std::vector<std::size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values,
                               const std::vector<std::size_t> &wires,
                               bool inverse, ParamT angle) {
        using ComplexT = std::complex<PrecisionT>;
        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        auto core_function =
            [cr, sj](std::complex<PrecisionT> *arr, const std::size_t i00,
                     const std::size_t i01, const std::size_t i10,
                     const std::size_t i11) {
                const ComplexT v00 = arr[i00];
                const ComplexT v01 = arr[i01];
                const ComplexT v10 = arr[i10];
                const ComplexT v11 = arr[i11];
                arr[i00] = ComplexT{cr * v00.real() - sj * v11.imag(),
                                    cr * v00.imag() + sj * v11.real()};
                arr[i01] = ComplexT{cr * v01.real() + sj * v10.imag(),
                                    cr * v01.imag() - sj * v10.real()};
                arr[i10] = ComplexT{cr * v10.real() + sj * v01.imag(),
                                    cr * v10.imag() - sj * v01.real()};
                arr[i11] = ComplexT{cr * v11.real() - sj * v00.imag(),
                                    cr * v11.imag() + sj * v00.real()};
            };
        if (controlled_wires.empty()) {
            applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC2<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyIsingYY(std::complex<PrecisionT> *arr,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &wires,
                             bool inverse, ParamT angle) {
        applyNCIsingYY(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyNCIsingZZ(std::complex<PrecisionT> *arr,
                               const std::size_t num_qubits,
                               const std::vector<std::size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values,
                               const std::vector<std::size_t> &wires,
                               bool inverse, ParamT angle) {
        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};
        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};

        auto core_function =
            [shifts](std::complex<PrecisionT> *arr, const std::size_t i00,
                     const std::size_t i01, const std::size_t i10,
                     const std::size_t i11) {
                arr[i00] *= shifts[0];
                arr[i01] *= shifts[1];
                arr[i10] *= shifts[1];
                arr[i11] *= shifts[0];
            };
        if (controlled_wires.empty()) {
            applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC2<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyIsingZZ(std::complex<PrecisionT> *arr,
                             const std::size_t num_qubits,
                             const std::vector<std::size_t> &wires,
                             bool inverse, ParamT angle) {
        applyNCIsingZZ(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyNCSingleExcitation(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, bool inverse, ParamT angle) {
        const PrecisionT c = std::cos(angle / 2);

        const PrecisionT s =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        auto core_function = [c, s](std::complex<PrecisionT> *arr,
                                    [[maybe_unused]] const std::size_t i00,
                                    const std::size_t i01,
                                    const std::size_t i10,
                                    [[maybe_unused]] const std::size_t i11) {
            const std::complex<PrecisionT> v01 = arr[i01];
            const std::complex<PrecisionT> v10 = arr[i10];
            arr[i01] = c * v01 - s * v10;
            arr[i10] = s * v01 + c * v10;
        };
        if (controlled_wires.empty()) {
            applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC2<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applySingleExcitation(std::complex<PrecisionT> *arr,
                                      std::size_t num_qubits,
                                      const std::vector<std::size_t> &wires,
                                      bool inverse, ParamT angle) {
        applyNCSingleExcitation(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyNCSingleExcitationMinus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, bool inverse, ParamT angle) {
        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        const std::complex<PrecisionT> e =
            inverse ? std::exp(std::complex<PrecisionT>(0, angle / 2))
                    : std::exp(-std::complex<PrecisionT>(0, angle / 2));
        auto core_function =
            [c, s, &e](std::complex<PrecisionT> *arr, const std::size_t i00,
                       const std::size_t i01, const std::size_t i10,
                       const std::size_t i11) {
                const std::complex<PrecisionT> v01 = arr[i01];
                const std::complex<PrecisionT> v10 = arr[i10];
                arr[i00] *= e;
                arr[i01] = c * v01 - s * v10;
                arr[i10] = s * v01 + c * v10;
                arr[i11] *= e;
            };
        if (controlled_wires.empty()) {
            applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC2<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applySingleExcitationMinus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &wires, bool inverse, ParamT angle) {
        applyNCSingleExcitationMinus(arr, num_qubits, {}, {}, wires, inverse,
                                     angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyNCSingleExcitationPlus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, bool inverse, ParamT angle) {
        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        const std::complex<PrecisionT> e =
            inverse ? std::exp(-std::complex<PrecisionT>(0, angle / 2))
                    : std::exp(std::complex<PrecisionT>(0, angle / 2));
        auto core_function =
            [c, s, &e](std::complex<PrecisionT> *arr, const std::size_t i00,
                       const std::size_t i01, const std::size_t i10,
                       const std::size_t i11) {
                const std::complex<PrecisionT> v01 = arr[i01];
                const std::complex<PrecisionT> v10 = arr[i10];
                arr[i00] *= e;
                arr[i01] = c * v01 - s * v10;
                arr[i10] = s * v01 + c * v10;
                arr[i11] *= e;
            };
        if (controlled_wires.empty()) {
            applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC2<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applySingleExcitationPlus(std::complex<PrecisionT> *arr,
                                          std::size_t num_qubits,
                                          const std::vector<std::size_t> &wires,
                                          bool inverse, ParamT angle) {
        applyNCSingleExcitationPlus(arr, num_qubits, {}, {}, wires, inverse,
                                    angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyNCPSWAP(std::complex<PrecisionT> *arr,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<std::size_t> &wires,
                             bool inverse, ParamT angle) {
        const std::complex<PrecisionT> phase =
            std::exp(std::complex<PrecisionT>(0, inverse ? -angle : angle));
        auto core_function = [phase](std::complex<PrecisionT> *arr,
                                     [[maybe_unused]] const std::size_t i00,
                                     const std::size_t i01,
                                     const std::size_t i10,
                                     [[maybe_unused]] const std::size_t i11) {
            arr[i10] *= phase;
            arr[i01] *= phase;
            std::swap(arr[i10], arr[i01]);
        };
        if (controlled_wires.empty()) {
            applyNC2<PrecisionT, ParamT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC2<PrecisionT, ParamT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyPSWAP(std::complex<PrecisionT> *arr,
                           std::size_t num_qubits,
                           const std::vector<std::size_t> &wires, bool inverse,
                           ParamT angle) {
        applyNCPSWAP(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    /* Four-qubit gates*/

    /**
     * @brief Apply a four-qubit gate with controls to the statevector.
     *
     * @tparam PrecisionT Floating point precision of underlying statevector
     * data
     * @tparam ParamT Floating point precision of the gate parameters
     * @tparam FuncT Gate function type.
     * @tparam has_controls True if the gate has control wires and false
     * otherwise (for optimization)
     * @tparam compute_indices True if the controlled gate requires the
     * computation of all statevector indices, otherwise only the 3rd and
     * 12th entries are calculated (for optimization)
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param core_function Gate function returning the matrix-vector
     * product for a wire block.
     */
    template <class PrecisionT, class ParamT = PrecisionT, class FuncT,
              bool has_controls = true, bool compute_indices = true>
    static void
    applyNC4(std::complex<PrecisionT> *arr, const std::size_t num_qubits,
             const std::vector<std::size_t> &controlled_wires,
             const std::vector<bool> &controlled_values,
             const std::vector<std::size_t> &wires, FuncT core_function) {
        const std::size_t n_contr = controlled_wires.size();
        const std::size_t n_wires = wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(n_wires == 4);
        PL_ASSERT(num_qubits >= nw_tot);
        if (has_controls || compute_indices) {
            std::vector<std::size_t> all_wires;
            all_wires.reserve(nw_tot);
            all_wires.insert(all_wires.begin(), wires.begin(), wires.end());
            all_wires.insert(all_wires.begin() + wires.size(),
                             controlled_wires.begin(), controlled_wires.end());
            const auto revs = reverseWires(num_qubits, all_wires, {});
            const auto &rev_wires = revs.first;
            const std::vector<std::size_t> parity =
                Pennylane::Util::revWireParity(rev_wires);
            std::vector<std::size_t> indices =
                Gates::generateBitPatterns(wires, num_qubits);
            if constexpr (has_controls) {
                Gates::controlBitPatterns(indices, num_qubits, controlled_wires,
                                          controlled_values);
            }
            const std::size_t i0011 = indices[0B0011];
            const std::size_t i1100 = indices[0B1100];
            PL_LOOP_PARALLEL(1)
            for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
                const std::size_t offset = parity_2_offset(parity, k);
                core_function(arr, i0011 + offset, i1100 + offset, indices,
                              offset);
            }
        } else {
            constexpr std::size_t offset{0U};
            constexpr std::size_t one{1U};
            const std::array<std::size_t, 4> rev_wires{
                num_qubits - wires[3] - 1, num_qubits - wires[2] - 1,
                num_qubits - wires[1] - 1, num_qubits - wires[0] - 1};
            const std::array<std::size_t, 4> rev_wire_shifts{
                one << rev_wires[0], one << rev_wires[1], one << rev_wires[2],
                one << rev_wires[3]};
            const auto parity = Pennylane::Util::revWireParity(rev_wires);
            const std::vector<std::size_t> indices{};
            PL_LOOP_PARALLEL(1)
            for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
                const std::size_t i0000 =
                    ((k << 4U) & parity[4]) | ((k << 3U) & parity[3]) |
                    ((k << 2U) & parity[2]) | ((k << 1U) & parity[1]) |
                    (k & parity[0]);
                const std::size_t i0011 =
                    i0000 | rev_wire_shifts[1] | rev_wire_shifts[0];
                const std::size_t i1100 =
                    i0000 | rev_wire_shifts[3] | rev_wire_shifts[2];
                core_function(arr, i0011, i1100, indices, offset);
            }
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyNCDoubleExcitation(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, bool inverse, ParamT angle) {
        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        auto core_function =
            [cr, sj](std::complex<PrecisionT> *arr, const std::size_t i0011,
                     const std::size_t i1100,
                     [[maybe_unused]] const std::vector<std::size_t> &indices,
                     [[maybe_unused]] const std::size_t offset) {
                const std::complex<PrecisionT> v3 = arr[i0011];
                const std::complex<PrecisionT> v12 = arr[i1100];
                arr[i0011] = cr * v3 - sj * v12;
                arr[i1100] = sj * v3 + cr * v12;
            };
        if (controlled_wires.empty()) {
            applyNC4<PrecisionT, ParamT, decltype(core_function), false, false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC4<PrecisionT, ParamT, decltype(core_function), true, false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyNCDoubleExcitationMinus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, bool inverse, ParamT angle) {
        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        const std::complex<PrecisionT> e =
            inverse ? std::exp(std::complex<PrecisionT>{0, angle / 2})
                    : std::exp(std::complex<PrecisionT>{0, -angle / 2});
        auto core_function = [cr, sj,
                              &e](std::complex<PrecisionT> *arr,
                                  [[maybe_unused]] const std::size_t i0011,
                                  [[maybe_unused]] const std::size_t i1100,
                                  const std::vector<std::size_t> &indices,
                                  const std::size_t offset) {
            const std::complex<PrecisionT> v3 = arr[indices[0B0011] + offset];
            const std::complex<PrecisionT> v12 = arr[indices[0B1100] + offset];
            for (const auto &i : indices) {
                arr[i + offset] *= e;
            }
            arr[indices[0B0011] + offset] = cr * v3 - sj * v12;
            arr[indices[0B1100] + offset] = sj * v3 + cr * v12;
        };
        if (controlled_wires.empty()) {
            applyNC4<PrecisionT, ParamT, decltype(core_function), false, true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC4<PrecisionT, ParamT, decltype(core_function), true, true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyNCDoubleExcitationPlus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, bool inverse, ParamT angle) {
        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        const std::complex<PrecisionT> e =
            inverse ? std::exp(std::complex<PrecisionT>{0, -angle / 2})
                    : std::exp(std::complex<PrecisionT>{0, angle / 2});
        auto core_function = [cr, sj,
                              &e](std::complex<PrecisionT> *arr,
                                  [[maybe_unused]] const std::size_t i0011,
                                  [[maybe_unused]] const std::size_t i1100,
                                  const std::vector<std::size_t> &indices,
                                  const std::size_t offset) {
            const std::complex<PrecisionT> v3 = arr[indices[0B0011] + offset];
            const std::complex<PrecisionT> v12 = arr[indices[0B1100] + offset];
            for (const auto &i : indices) {
                arr[i + offset] *= e;
            }
            arr[indices[0B0011] + offset] = cr * v3 - sj * v12;
            arr[indices[0B1100] + offset] = sj * v3 + cr * v12;
        };
        if (controlled_wires.empty()) {
            applyNC4<PrecisionT, ParamT, decltype(core_function), false, true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNC4<PrecisionT, ParamT, decltype(core_function), true, true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyDoubleExcitation(std::complex<PrecisionT> *arr,
                                      std::size_t num_qubits,
                                      const std::vector<std::size_t> &wires,
                                      bool inverse, ParamT angle) {
        applyNCDoubleExcitation(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyDoubleExcitationMinus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &wires, bool inverse, ParamT angle) {
        applyNCDoubleExcitationMinus(arr, num_qubits, {}, {}, wires, inverse,
                                     angle);
    }

    template <class PrecisionT, class ParamT>
    static void applyDoubleExcitationPlus(std::complex<PrecisionT> *arr,
                                          std::size_t num_qubits,
                                          const std::vector<std::size_t> &wires,
                                          bool inverse, ParamT angle) {
        applyNCDoubleExcitationPlus(arr, num_qubits, {}, {}, wires, inverse,
                                    angle);
    }

    /* Multi-qubit gates */

    template <class PrecisionT, class ParamT>
    static void applyMultiRZ(std::complex<PrecisionT> *arr,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &wires,
                             bool inverse, ParamT angle) {
        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};
        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};

        std::size_t wires_parity = 0U;
        for (std::size_t wire : wires) {
            wires_parity |=
                (static_cast<std::size_t>(1U) << (num_qubits - wire - 1));
        }
        PL_LOOP_PARALLEL(1)
        for (std::size_t k = 0; k < exp2(num_qubits); k++) {
            arr[k] *= shifts[std::popcount(k & wires_parity) % 2];
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyNCMultiRZ(std::complex<PrecisionT> *arr,
                               std::size_t num_qubits,
                               const std::vector<std::size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values,
                               const std::vector<std::size_t> &wires,
                               bool inverse, ParamT angle) {
        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};
        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};
        std::size_t wires_parity = 0U;
        for (std::size_t wire : wires) {
            wires_parity |=
                (static_cast<std::size_t>(1U) << (num_qubits - wire - 1));
        }
        auto core_function = [wires_parity,
                              &shifts](std::complex<PrecisionT> *arr,
                                       const std::vector<std::size_t> &indices,
                                       const std::size_t offset) {
            for (auto k : indices) {
                arr[(k + offset)] *=
                    shifts[std::popcount((k + offset) & wires_parity) % 2];
            }
        };
        applyNCN(arr, num_qubits, controlled_wires, controlled_values, wires,
                 core_function);
    }

    template <class PrecisionT, class ParamT>
    static void
    applyGlobalPhase(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                     [[maybe_unused]] const std::vector<std::size_t> &wires,
                     [[maybe_unused]] bool inverse,
                     [[maybe_unused]] ParamT angle) {
        applyNCGlobalPhase(arr, num_qubits, {}, {}, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT>
    static void
    applyNCGlobalPhase(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                       const std::vector<std::size_t> &controlled_wires,
                       const std::vector<bool> &controlled_values,
                       [[maybe_unused]] const std::vector<std::size_t> &wires,
                       bool inverse, ParamT angle) {
        const std::complex<PrecisionT> phase =
            std::exp(std::complex<PrecisionT>(0, inverse ? angle : -angle));
        auto core_function = [&phase](std::complex<PrecisionT> *arr,
                                      const std::size_t i0,
                                      const std::size_t i1) {
            arr[i0] *= phase;
            arr[i1] *= phase;
        };
        std::size_t target{0U};
        if (!controlled_wires.empty()) {
            for (std::size_t i = 0; i < num_qubits; i++) {
                if (std::find(controlled_wires.begin(), controlled_wires.end(),
                              i) == controlled_wires.end()) {
                    target = i;
                    break;
                }
            }
        }
        if (controlled_wires.empty()) {
            applyNC1<PrecisionT, PrecisionT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, {target},
                core_function);
        } else {
            applyNC1<PrecisionT, PrecisionT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, {target},
                core_function);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyPCPhase(std::complex<PrecisionT> *arr,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &wires,
                             bool inverse, ParamT angle, ParamT dim) {
        applyNCPCPhase(arr, num_qubits, {}, {}, wires, inverse, angle, dim);
    }

    template <class PrecisionT, class ParamT>
    static void applyNCPCPhase(std::complex<PrecisionT> *arr,
                               std::size_t num_qubits,
                               const std::vector<std::size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values,
                               const std::vector<std::size_t> &wires,
                               bool inverse, ParamT angle, ParamT dim) {
        const auto dim_size_t = static_cast<std::size_t>(std::round(dim));

        PL_ABORT_IF(dim_size_t < 0 || dim_size_t > 1U << num_qubits,
                    "The dimension of the PCPhase gate must be a positive "
                    "integer and less than or equal to statevector size.");

        const PrecisionT phase = inverse ? -angle : angle;
        const std::complex<PrecisionT> upper_complex = {std::cos(phase),
                                                        std::sin(phase)};
        const std::complex<PrecisionT> lower_complex = std::conj(upper_complex);

        // Matrix
        const std::size_t matrix_dim = Pennylane::Util::exp2(wires.size());

        std::vector<std::complex<PrecisionT>> matrixPCPhase(matrix_dim *
                                                            matrix_dim);

        // Fill diagonal
        for (std::size_t i = 0; i < dim_size_t; i++) {
            matrixPCPhase[i * matrix_dim + i] = upper_complex;
        }
        for (std::size_t i = dim_size_t; i < matrix_dim; i++) {
            matrixPCPhase[i * matrix_dim + i] = lower_complex;
        }

        if (controlled_wires.empty()) {
            applyMultiQubitOp<PrecisionT>(arr, num_qubits, matrixPCPhase.data(),
                                          wires, false);

        } else {
            applyNCMultiQubitOp<PrecisionT>(
                arr, num_qubits, matrixPCPhase.data(), controlled_wires,
                controlled_values, wires, false);
        }
    }

    /* Generators */

    /**
     * @brief Apply a single-qubit generator with controls to the
     * statevector.
     *
     * @tparam PrecisionT Floating point precision of underlying statevector
     * data
     * @tparam FuncT Gate function type.
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param core_function Gate function returning the matrix-vector
     * product for a wire block.
     */
    template <class PrecisionT, class FuncT>
    static void
    applyNCGenerator1(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                      const std::vector<std::size_t> &controlled_wires,
                      const std::vector<bool> &controlled_values,
                      const std::vector<std::size_t> &wires,
                      FuncT core_function) {
        constexpr std::size_t one{1U};
        constexpr std::complex<PrecisionT> zero{0.0};

        const std::size_t n_contr = controlled_wires.size();
        const std::size_t n_wires = wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(n_wires == 1);
        PL_ASSERT(num_qubits >= nw_tot);

        std::vector<std::size_t> all_wires;
        all_wires.reserve(nw_tot);
        all_wires.insert(all_wires.begin(), controlled_wires.begin(),
                         controlled_wires.end());
        all_wires.insert(all_wires.begin() + n_contr, wires.begin(),
                         wires.end());
        const auto revs = reverseWires(num_qubits, all_wires, {});
        const auto &rev_wires = revs.first;
        const std::vector<std::size_t> parity =
            Pennylane::Util::revWireParity(rev_wires);
        const std::vector<std::size_t> indices =
            Gates::generateBitPatterns(all_wires, num_qubits);

        std::size_t mask{0U};
        for (std::size_t k = 0; k < controlled_values.size(); k++) {
            mask |= static_cast<std::size_t>(controlled_values[n_contr - 1 - k])
                    << k;
        }
        const std::size_t i0 = indices[mask << one];
        const std::size_t i1 = indices[(mask << one) | one];
        PL_LOOP_PARALLEL(1)
        for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
            const std::size_t offset = parity_2_offset(parity, k);
            for (std::size_t i = 0; i < indices.size(); i++) {
                if ((i >> one) == mask) {
                    continue;
                }
                arr[indices[i] + offset] = zero;
            }
            core_function(arr, i0 + offset, i1 + offset);
        }
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyNCGeneratorGlobalPhase(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        [[maybe_unused]] const std::vector<std::size_t> &wires,
        [[maybe_unused]] const bool adj) -> PrecisionT {
        auto core_function = []([[maybe_unused]] std::complex<PrecisionT> *arr,
                                [[maybe_unused]] const std::size_t i0,
                                [[maybe_unused]] const std::size_t i1) {};
        std::size_t target{0U};
        if (!controlled_wires.empty()) {
            for (std::size_t i = 0; i < num_qubits; i++) {
                if (std::find(controlled_wires.begin(), controlled_wires.end(),
                              i) == controlled_wires.end()) {
                    target = i;
                    break;
                }
            }
        }
        applyNCGenerator1<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, {target},
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(-1.0);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorGlobalPhase(
        [[maybe_unused]] std::complex<PrecisionT> *arr,
        [[maybe_unused]] std::size_t num_qubits,
        [[maybe_unused]] const std::vector<std::size_t> &wires,
        [[maybe_unused]] bool adj) -> PrecisionT {
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(-1.0);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyNCGeneratorPhaseShift(std::complex<PrecisionT> *arr,
                               std::size_t num_qubits,
                               const std::vector<std::size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values,
                               const std::vector<std::size_t> &wires,
                               [[maybe_unused]] const bool adj) -> PrecisionT {
        constexpr std::complex<PrecisionT> zero{0.0};
        auto core_function =
            [zero](std::complex<PrecisionT> *arr, const std::size_t i0,
                   [[maybe_unused]] const std::size_t i1) { arr[i0] = zero; };
        applyNCGenerator1<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(1.0);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorPhaseShift(std::complex<PrecisionT> *arr,
                             std::size_t num_qubits,
                             const std::vector<std::size_t> &wires,
                             [[maybe_unused]] bool adj) -> PrecisionT {
        return applyNCGeneratorPhaseShift(arr, num_qubits, {}, {}, wires, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorControlledPhaseShift(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        return applyNCGeneratorPhaseShift(arr, num_qubits, {wires[0]}, {true},
                                          {wires[1]}, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyNCGeneratorRX(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                       const std::vector<std::size_t> &controlled_wires,
                       const std::vector<bool> &controlled_values,
                       const std::vector<std::size_t> &wires,
                       [[maybe_unused]] const bool adj) -> PrecisionT {
        auto core_function = [](std::complex<PrecisionT> *arr,
                                const std::size_t i0, const std::size_t i1) {
            std::swap(arr[i0], arr[i1]);
        };
        applyNCGenerator1<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRX(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                      const std::vector<std::size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        return applyNCGeneratorRX(arr, num_qubits, {wires[0]}, {true},
                                  {wires[1]}, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyNCGeneratorRY(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                       const std::vector<std::size_t> &controlled_wires,
                       const std::vector<bool> &controlled_values,
                       const std::vector<std::size_t> &wires,
                       [[maybe_unused]] const bool adj) -> PrecisionT {
        auto core_function = [](std::complex<PrecisionT> *arr,
                                const std::size_t i0, const std::size_t i1) {
            const auto v0 = arr[i0];
            const auto v1 = arr[i1];
            arr[i0] = {std::imag(v1), -std::real(v1)};
            arr[i1] = {-std::imag(v0), std::real(v0)};
        };
        applyNCGenerator1<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRY(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                      const std::vector<std::size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        return applyNCGeneratorRY(arr, num_qubits, {wires[0]}, {true},
                                  {wires[1]}, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyNCGeneratorRZ(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                       const std::vector<std::size_t> &controlled_wires,
                       const std::vector<bool> &controlled_values,
                       const std::vector<std::size_t> &wires,
                       [[maybe_unused]] const bool adj) -> PrecisionT {
        auto core_function = [](std::complex<PrecisionT> *arr,
                                [[maybe_unused]] const std::size_t i0,
                                const std::size_t i1) { arr[i1] *= -1; };
        applyNCGenerator1<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRZ(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                      const std::vector<std::size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        return applyNCGeneratorRZ(arr, num_qubits, {wires[0]}, {true},
                                  {wires[1]}, adj);
    }

    /**
     * @brief Apply a two-qubit generator with controls to the statevector.
     *
     * @tparam PrecisionT Floating point precision of underlying statevector
     * data
     * @tparam FuncT Gate function type.
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param core_function Gate function returning the matrix-vector
     * product for a wire block.
     */
    template <class PrecisionT, class FuncT>
    static void
    applyNCGenerator2(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                      const std::vector<std::size_t> &controlled_wires,
                      const std::vector<bool> &controlled_values,
                      const std::vector<std::size_t> &wires,
                      FuncT core_function) {
        constexpr std::size_t one{1U};
        constexpr std::size_t two{2U};
        constexpr std::complex<PrecisionT> zero{0.0};

        const std::size_t n_contr = controlled_wires.size();
        const std::size_t n_wires = wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(n_wires == 2);
        PL_ASSERT(num_qubits >= nw_tot);

        std::vector<std::size_t> all_wires;
        all_wires.reserve(nw_tot);
        all_wires.insert(all_wires.begin(), controlled_wires.begin(),
                         controlled_wires.end());
        all_wires.insert(all_wires.begin() + n_contr, wires.begin(),
                         wires.end());
        const auto revs = reverseWires(num_qubits, all_wires, {});
        const auto &rev_wires = revs.first;
        const std::vector<std::size_t> parity =
            Pennylane::Util::revWireParity(rev_wires);
        const std::vector<std::size_t> indices =
            Gates::generateBitPatterns(all_wires, num_qubits);

        std::size_t mask{0U};
        for (std::size_t k = 0; k < controlled_values.size(); k++) {
            mask |= static_cast<std::size_t>(controlled_values[n_contr - 1 - k])
                    << k;
        }
        const std::size_t i00 = indices[mask << two];
        const std::size_t i01 = indices[(mask << two) | one];
        const std::size_t i10 = indices[(mask << two) | two];
        const std::size_t i11 = indices[(mask << two) | two | one];

        PL_LOOP_PARALLEL(1)
        for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
            const std::size_t offset = parity_2_offset(parity, k);
            for (std::size_t i = 0; i < indices.size(); i++) {
                if ((i >> two) == mask) {
                    continue;
                }
                arr[indices[i] + offset] = zero;
            }
            core_function(arr, i00 + offset, i01 + offset, i10 + offset,
                          i11 + offset);
        }
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyNCGeneratorIsingXX(std::complex<PrecisionT> *arr,
                            std::size_t num_qubits,
                            const std::vector<std::size_t> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<std::size_t> &wires,
                            [[maybe_unused]] bool adj) -> PrecisionT {
        auto core_function = [](std::complex<PrecisionT> *arr,
                                const std::size_t i00, const std::size_t i01,
                                const std::size_t i10, const std::size_t i11) {
            std::swap(arr[i00], arr[i11]);
            std::swap(arr[i10], arr[i01]);
        };
        applyNCGenerator2<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingXX(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                          const std::vector<std::size_t> &wires, bool adj)
        -> PrecisionT {
        return applyNCGeneratorIsingXX(arr, num_qubits, {}, {}, wires, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyNCGeneratorIsingXY(std::complex<PrecisionT> *arr,
                            std::size_t num_qubits,
                            const std::vector<std::size_t> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<std::size_t> &wires,
                            [[maybe_unused]] bool adj) -> PrecisionT {
        constexpr std::complex<PrecisionT> zero{0.0};
        auto core_function =
            [zero](std::complex<PrecisionT> *arr, const std::size_t i00,
                   const std::size_t i01, const std::size_t i10,
                   const std::size_t i11) {
                std::swap(arr[i10], arr[i01]);
                arr[i00] = zero;
                arr[i11] = zero;
            };
        applyNCGenerator2<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingXY(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                          const std::vector<std::size_t> &wires, bool adj)
        -> PrecisionT {
        return applyNCGeneratorIsingXY(arr, num_qubits, {}, {}, wires, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyNCGeneratorIsingYY(std::complex<PrecisionT> *arr,
                            std::size_t num_qubits,
                            const std::vector<std::size_t> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<std::size_t> &wires,
                            [[maybe_unused]] bool adj) -> PrecisionT {
        auto core_function = [](std::complex<PrecisionT> *arr,
                                const std::size_t i00, const std::size_t i01,
                                const std::size_t i10, const std::size_t i11) {
            const auto v00 = arr[i00];
            arr[i00] = -arr[i11];
            arr[i11] = -v00;
            std::swap(arr[i10], arr[i01]);
        };
        applyNCGenerator2<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingYY(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                          const std::vector<std::size_t> &wires, bool adj)
        -> PrecisionT {
        return applyNCGeneratorIsingYY(arr, num_qubits, {}, {}, wires, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyNCGeneratorIsingZZ(std::complex<PrecisionT> *arr,
                            std::size_t num_qubits,
                            const std::vector<std::size_t> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<std::size_t> &wires,
                            [[maybe_unused]] bool adj) -> PrecisionT {
        auto core_function = [](std::complex<PrecisionT> *arr,
                                [[maybe_unused]] const std::size_t i00,
                                const std::size_t i01, const std::size_t i10,
                                [[maybe_unused]] const std::size_t i11) {
            arr[i10] *= -1;
            arr[i01] *= -1;
        };
        applyNCGenerator2<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingZZ(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                          const std::vector<std::size_t> &wires, bool adj)
        -> PrecisionT {
        return applyNCGeneratorIsingZZ(arr, num_qubits, {}, {}, wires, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyNCGeneratorSingleExcitation(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        auto core_function = [](std::complex<PrecisionT> *arr,
                                const std::size_t i00, const std::size_t i01,
                                const std::size_t i10, const std::size_t i11) {
            arr[i00] = std::complex<PrecisionT>{};
            arr[i01] *= Pennylane::Util::IMAG<PrecisionT>();
            arr[i10] *= -Pennylane::Util::IMAG<PrecisionT>();
            arr[i11] = std::complex<PrecisionT>{};
            std::swap(arr[i10], arr[i01]);
        };
        applyNCGenerator2<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorSingleExcitation(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &wires, bool adj) -> PrecisionT {
        return applyNCGeneratorSingleExcitation(arr, num_qubits, {}, {}, wires,
                                                adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyNCGeneratorSingleExcitationMinus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        auto core_function = [](std::complex<PrecisionT> *arr,
                                [[maybe_unused]] const std::size_t i00,
                                const std::size_t i01, const std::size_t i10,
                                [[maybe_unused]] const std::size_t i11) {
            arr[i01] *= Pennylane::Util::IMAG<PrecisionT>();
            arr[i10] *= -Pennylane::Util::IMAG<PrecisionT>();
            std::swap(arr[i10], arr[i01]);
        };
        applyNCGenerator2<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorSingleExcitationMinus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &wires, bool adj) -> PrecisionT {
        return applyNCGeneratorSingleExcitationMinus(arr, num_qubits, {}, {},
                                                     wires, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyNCGeneratorSingleExcitationPlus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        auto core_function = [](std::complex<PrecisionT> *arr,
                                const std::size_t i00, const std::size_t i01,
                                const std::size_t i10, const std::size_t i11) {
            arr[i00] *= -1;
            arr[i01] *= Pennylane::Util::IMAG<PrecisionT>();
            arr[i10] *= -Pennylane::Util::IMAG<PrecisionT>();
            arr[i11] *= -1;
            std::swap(arr[i10], arr[i01]);
        };
        applyNCGenerator2<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorSingleExcitationPlus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &wires, bool adj) -> PrecisionT {
        return applyNCGeneratorSingleExcitationPlus(arr, num_qubits, {}, {},
                                                    wires, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyNCGeneratorPSWAP(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                          const std::vector<std::size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<std::size_t> &wires,
                          [[maybe_unused]] const bool adj) -> PrecisionT {
        using ComplexT = std::complex<PrecisionT>;
        constexpr ComplexT zero{0.0};
        auto core_function =
            [zero](ComplexT *arr, const std::size_t i00, const std::size_t i01,
                   const std::size_t i10, const std::size_t i11) {
                arr[i00] = zero;
                arr[i11] = zero;
                std::swap(arr[i10], arr[i01]);
            };
        applyNCGenerator2<PrecisionT, decltype(core_function)>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(1.0);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorPSWAP(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                        const std::vector<std::size_t> &wires, bool adj)
        -> PrecisionT {
        return applyNCGeneratorPSWAP(arr, num_qubits, {}, {}, wires, adj);
    }

    /**
     * @brief Apply a four-qubit generator with controls to the
     * statevector.
     *
     * @tparam PrecisionT Floating point precision of underlying statevector
     * data
     * @tparam FuncT Gate function type.
     * @tparam compute_indices True if the controlled gate requires the
     * computation of all statevector indices, otherwise only the 3rd and
     * 12th entries are calculated (for optimization)
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param core_function Gate function returning the matrix-vector
     * product for a wire block.
     */
    template <class PrecisionT, class FuncT, bool compute_indices = true>
    static void
    applyNCGenerator4(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                      const std::vector<std::size_t> &controlled_wires,
                      const std::vector<bool> &controlled_values,
                      const std::vector<std::size_t> &wires,
                      FuncT core_function) {
        constexpr std::complex<PrecisionT> zero{0.0};

        const std::size_t n_contr = controlled_wires.size();
        const std::size_t n_wires = wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        PL_ASSERT(n_wires == 4);
        PL_ASSERT(num_qubits >= nw_tot);

        std::vector<std::size_t> all_wires;
        all_wires.reserve(nw_tot);
        all_wires.insert(all_wires.begin(), controlled_wires.begin(),
                         controlled_wires.end());
        all_wires.insert(all_wires.begin() + n_contr, wires.begin(),
                         wires.end());
        const auto revs = reverseWires(num_qubits, all_wires, {});
        const auto &rev_wires = revs.first;
        const std::vector<std::size_t> parity =
            Pennylane::Util::revWireParity(rev_wires);
        const std::vector<std::size_t> indices =
            Gates::generateBitPatterns(all_wires, num_qubits);

        std::size_t mask{0U};
        for (std::size_t k = 0; k < controlled_values.size(); k++) {
            mask |= static_cast<std::size_t>(controlled_values[n_contr - 1 - k])
                    << k;
        }

        const std::size_t i0011 = indices[(mask << 4U) + 3U];
        const std::size_t i1100 = indices[(mask << 4U) + 12U];

        PL_LOOP_PARALLEL(1)
        for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
            const std::size_t offset = parity_2_offset(parity, k);
            if constexpr (compute_indices) {
                for (std::size_t i = 0; i < indices.size(); i++) {
                    if ((i >> 4U) == mask) {
                        continue;
                    }
                    arr[indices[i] + offset] = zero;
                }
                core_function(arr, i0011 + offset, i1100 + offset, indices,
                              offset);
            } else {
                core_function(arr, i0011 + offset, i1100 + offset, indices,
                              offset);
            }
        }
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyNCGeneratorDoubleExcitation(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        using ComplexT = std::complex<PrecisionT>;
        constexpr ComplexT zero{};
        constexpr ComplexT imag{0, 1};
        auto core_function =
            [&zero, &imag](std::complex<PrecisionT> *arr,
                           const std::size_t i0011, const std::size_t i1100,
                           const std::vector<std::size_t> &indices,
                           const std::size_t offset) {
                const ComplexT v3 = arr[i0011];
                const ComplexT v12 = arr[i1100];
                for (const auto &i : indices) {
                    arr[i + offset] = zero;
                }
                arr[i0011] = -v12 * imag;
                arr[i1100] = v3 * imag;
            };
        applyNCGenerator4<PrecisionT, decltype(core_function), true>(
            arr, num_qubits, controlled_wires, controlled_values, wires,
            core_function);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorDoubleExcitation(std::complex<PrecisionT> *arr,
                                   std::size_t num_qubits,
                                   const std::vector<std::size_t> &wires,
                                   [[maybe_unused]] bool adj) -> PrecisionT {
        return applyNCGeneratorDoubleExcitation(arr, num_qubits, {}, {}, wires,
                                                adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyNCGeneratorDoubleExcitationMinus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        using ComplexT = std::complex<PrecisionT>;
        constexpr ComplexT imag{0, 1};
        auto core_function =
            [imag](std::complex<PrecisionT> *arr, const std::size_t i0011,
                   const std::size_t i1100,
                   [[maybe_unused]] const std::vector<std::size_t> &indices,
                   [[maybe_unused]] const std::size_t offset) {
                arr[i0011] *= imag;
                arr[i1100] *= -imag;
                swap(arr[i1100], arr[i0011]);
            };
        if (controlled_wires.empty()) {
            applyNCGenerator4<PrecisionT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNCGenerator4<PrecisionT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
        // NOLINTNEXTLINE(readability - magic - numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorDoubleExcitationMinus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        return applyNCGeneratorDoubleExcitationMinus(arr, num_qubits, {}, {},
                                                     wires, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyNCGeneratorDoubleExcitationPlus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        using ComplexT = std::complex<PrecisionT>;
        constexpr ComplexT imag{0, 1};
        auto core_function =
            [imag](std::complex<PrecisionT> *arr, const std::size_t i0011,
                   const std::size_t i1100,
                   [[maybe_unused]] const std::vector<std::size_t> &indices,
                   [[maybe_unused]] const std::size_t offset) {
                arr[i0011] *= -imag;
                arr[i1100] *= imag;
                swap(arr[i1100], arr[i0011]);
            };
        if (controlled_wires.empty()) {
            applyNCGenerator4<PrecisionT, decltype(core_function), false>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        } else {
            applyNCGenerator4<PrecisionT, decltype(core_function), true>(
                arr, num_qubits, controlled_wires, controlled_values, wires,
                core_function);
        }
        // NOLINTNEXTLINE(readability - magic - numbers)
        return static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorDoubleExcitationPlus(
        std::complex<PrecisionT> *arr, std::size_t num_qubits,
        const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        return applyNCGeneratorDoubleExcitationPlus(arr, num_qubits, {}, {},
                                                    wires, adj);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorMultiRZ(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                          const std::vector<std::size_t> &wires,
                          [[maybe_unused]] bool adj) -> PrecisionT {
        auto wires_parity = static_cast<std::size_t>(0U);
        for (std::size_t wire : wires) {
            wires_parity |=
                (static_cast<std::size_t>(1U) << (num_qubits - wire - 1));
        }
        PL_LOOP_PARALLEL(1)
        for (std::size_t k = 0; k < exp2(num_qubits); k++) {
            arr[k] *= static_cast<PrecisionT>(
                1 - 2 * int(std::popcount(k & wires_parity) % 2));
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyNCGeneratorMultiRZ(std::complex<PrecisionT> *arr,
                            std::size_t num_qubits,
                            const std::vector<std::size_t> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<std::size_t> &wires,
                            [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        constexpr std::size_t one{1U};
        constexpr std::complex<PrecisionT> zero{0.0};
        auto ctrls_mask = static_cast<std::size_t>(0U);
        for (std::size_t i = 0; i < controlled_wires.size(); i++) {
            ctrls_mask |= (static_cast<std::size_t>(controlled_values[i])
                           << (num_qubits - controlled_wires[i] - 1));
        }
        auto ctrls_parity = static_cast<std::size_t>(0U);
        for (std::size_t wire : controlled_wires) {
            ctrls_parity |= (one << (num_qubits - wire - 1));
        }
        auto wires_parity = static_cast<std::size_t>(0U);
        for (std::size_t wire : wires) {
            wires_parity |= (one << (num_qubits - wire - 1));
        }
        PL_LOOP_PARALLEL(1)
        for (std::size_t k = 0; k < exp2(num_qubits); k++) {
            if (ctrls_mask == (ctrls_parity & k)) {
                arr[k] *= static_cast<PrecisionT>(
                    1 - 2 * int(std::popcount(k & wires_parity) % 2));
            } else {
                arr[k] = zero;
            }
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }
};
} // namespace Pennylane::LightningQubit::Gates
