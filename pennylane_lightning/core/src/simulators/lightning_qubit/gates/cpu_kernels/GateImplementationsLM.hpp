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
#include <bit>
#include <complex>
#include <tuple>
#include <vector>

#include "BitUtil.hpp" // fillLeadingOnes, fillTrailingOnes, bitswap
#include "Error.hpp"
#include "GateOperation.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"
#include "PauliGenerator.hpp"
#include "Util.hpp" // exp2, INVSQRT2

/// @cond DEV
namespace {
using namespace Pennylane::Gates;
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
    static std::pair<size_t, size_t> revWireParity(size_t rev_wire) {
        const auto parity = Pennylane::Util::revWireParity(
            std::array<std::size_t, 1>{rev_wire});
        return {parity[1], parity[0]};
    }
    static std::tuple<size_t, size_t, size_t> revWireParity(size_t rev_wire0,
                                                            size_t rev_wire1) {
        const auto parity = Pennylane::Util::revWireParity(
            std::array<std::size_t, 2>{rev_wire0, rev_wire1});
        return {parity[2], parity[1], parity[0]};
    }
    template <const size_t wire_size = 3>
    static constexpr auto revWireParity(size_t rev_wire0, size_t rev_wire1,
                                        size_t rev_wire2)
        -> std::array<size_t, wire_size + 1> {
        return Pennylane::Util::revWireParity(
            std::array<std::size_t, wire_size>{rev_wire0, rev_wire1,
                                               rev_wire2});
    }
    template <const size_t wire_size = 4>
    static constexpr auto revWireParity(size_t rev_wire0, size_t rev_wire1,
                                        size_t rev_wire2, size_t rev_wire3)
        -> std::array<size_t, wire_size + 1> {
        return Pennylane::Util::revWireParity(
            std::array<std::size_t, wire_size>{rev_wire0, rev_wire1, rev_wire2,
                                               rev_wire3});
    }

  public:
    constexpr static KernelType kernel_id = KernelType::LM;
    constexpr static std::string_view name = "LM";
    template <typename PrecisionT>
    constexpr static size_t required_alignment =
        std::alignment_of_v<PrecisionT>;
    template <typename PrecisionT>
    constexpr static size_t packed_bytes = sizeof(PrecisionT);

    constexpr static std::array implemented_gates = {
        GateOperation::Identity,
        GateOperation::PauliX,
        GateOperation::PauliY,
        GateOperation::PauliZ,
        GateOperation::Hadamard,
        GateOperation::S,
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
        GateOperation::MultiRZ};

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
        GeneratorOperation::MultiRZ,
    };

    constexpr static std::array implemented_matrices = {
        MatrixOperation::SingleQubitOp, MatrixOperation::TwoQubitOp,
        MatrixOperation::MultiQubitOp};

    /**
     * @brief Computes the array of indices to apply the gate corresponding to
     * the k-th state vector block.
     *
     * @param k State vector block index.
     * @param parity Leading/trailing masks.
     * @param rev_wire_shifts Single-bit masks at wire positions.
     * @return State vector indices corresponding to the k-th state vector
     * block.
     */
    template <std::size_t length>
    static auto parity2indices(const std::size_t k,
                               std::array<std::size_t, length + 1> parity,
                               std::array<std::size_t, length> rev_wire_shifts)
        -> std::array<std::size_t, static_cast<std::size_t>(1) << length> {
        constexpr std::size_t one{1};
        const std::size_t dim = one << length;
        std::array<std::size_t, dim> indices{};
        std::size_t idx = (k & parity[0]);
        for (std::size_t i = 1; i < parity.size(); i++) {
            idx |= ((k << i) & parity[i]);
        }
        indices[0] = idx;
        for (std::size_t inner_idx = 1; inner_idx < dim; inner_idx++) {
            idx = indices[0];
            for (std::size_t i = 0; i < length; i++) {
                if ((inner_idx & (one << i)) != 0) {
                    idx |= rev_wire_shifts[i];
                }
            }
            indices[inner_idx] = idx;
        }
        return indices;
    }

    /**
     * @brief Computes the vector of indices to apply the gate corresponding to
     * the k-th state vector block.
     *
     * @param k State vector block index.
     * @param parity Leading/trailing masks.
     * @param rev_wire_shifts Single-bit masks at wire positions.
     * @return State vector indices corresponding to the k-th state vector
     * block.
     */
    static auto parity2indices(const std::size_t k,
                               std::vector<std::size_t> parity,
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

    /**
     * @brief Apply a single qubit gate to the statevector.
     *
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wire A wire the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT>
    static void
    applySingleQubitOp(std::complex<PrecisionT> *arr, size_t num_qubits,
                       const std::complex<PrecisionT> *matrix,
                       const std::vector<size_t> &wires, bool inverse = false) {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        if (inverse) {
            for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
                const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
                const size_t i1 = i0 | rev_wire_shift;
                const std::complex<PrecisionT> v0 = arr[i0];
                const std::complex<PrecisionT> v1 = arr[i1];
                arr[i0] = std::conj(matrix[0B00]) * v0 +
                          std::conj(matrix[0B10]) *
                              v1; // NOLINT(readability-magic-numbers)
                arr[i1] = std::conj(matrix[0B01]) * v0 +
                          std::conj(matrix[0B11]) *
                              v1; // NOLINT(readability-magic-numbers)
            }
        } else {
            for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
                const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
                const size_t i1 = i0 | rev_wire_shift;
                const std::complex<PrecisionT> v0 = arr[i0];
                const std::complex<PrecisionT> v1 = arr[i1];
                arr[i0] =
                    matrix[0B00] * v0 +
                    matrix[0B01] * v1; // NOLINT(readability-magic-numbers)
                arr[i1] =
                    matrix[0B10] * v0 +
                    matrix[0B11] * v1; // NOLINT(readability-magic-numbers)
            }
        }
    }

    /**
     * @brief Apply a two qubit gate to the statevector.
     *
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT>
    static void
    applyTwoQubitOp(std::complex<PrecisionT> *arr, size_t num_qubits,
                    const std::complex<PrecisionT> *matrix,
                    const std::vector<size_t> &wires, bool inverse = false) {
        PL_ASSERT(wires.size() == 2);
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        if (inverse) {
            for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
                const size_t i00 = ((k << 2U) & parity_high) |
                                   ((k << 1U) & parity_middle) |
                                   (k & parity_low);
                const size_t i10 = i00 | rev_wire1_shift;
                const size_t i01 = i00 | rev_wire0_shift;
                const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

                const std::complex<PrecisionT> v00 = arr[i00];
                const std::complex<PrecisionT> v01 = arr[i01];
                const std::complex<PrecisionT> v10 = arr[i10];
                const std::complex<PrecisionT> v11 = arr[i11];

                // NOLINTBEGIN(readability-magic-numbers)
                arr[i00] = std::conj(matrix[0b0000]) * v00 +
                           std::conj(matrix[0b0100]) * v01 +
                           std::conj(matrix[0b1000]) * v10 +
                           std::conj(matrix[0b1100]) * v11;
                arr[i01] = std::conj(matrix[0b0001]) * v00 +
                           std::conj(matrix[0b0101]) * v01 +
                           std::conj(matrix[0b1001]) * v10 +
                           std::conj(matrix[0b1101]) * v11;
                arr[i10] = std::conj(matrix[0b0010]) * v00 +
                           std::conj(matrix[0b0110]) * v01 +
                           std::conj(matrix[0b1010]) * v10 +
                           std::conj(matrix[0b1110]) * v11;
                arr[i11] = std::conj(matrix[0b0011]) * v00 +
                           std::conj(matrix[0b0111]) * v01 +
                           std::conj(matrix[0b1011]) * v10 +
                           std::conj(matrix[0b1111]) * v11;
                // NOLINTEND(readability-magic-numbers)
            }
        } else {
            for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
                const size_t i00 = ((k << 2U) & parity_high) |
                                   ((k << 1U) & parity_middle) |
                                   (k & parity_low);
                const size_t i10 = i00 | rev_wire1_shift;
                const size_t i01 = i00 | rev_wire0_shift;
                const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

                const std::complex<PrecisionT> v00 = arr[i00];
                const std::complex<PrecisionT> v01 = arr[i01];
                const std::complex<PrecisionT> v10 = arr[i10];
                const std::complex<PrecisionT> v11 = arr[i11];

                // NOLINTBEGIN(readability-magic-numbers)
                arr[i00] = matrix[0b0000] * v00 + matrix[0b0001] * v01 +
                           matrix[0b0010] * v10 + matrix[0b0011] * v11;
                arr[i01] = matrix[0b0100] * v00 + matrix[0b0101] * v01 +
                           matrix[0b0110] * v10 + matrix[0b0111] * v11;
                arr[i10] = matrix[0b1000] * v00 + matrix[0b1001] * v01 +
                           matrix[0b1010] * v10 + matrix[0b1011] * v11;
                arr[i11] = matrix[0b1100] * v00 + matrix[0b1101] * v01 +
                           matrix[0b1110] * v10 + matrix[0b1111] * v11;
                // NOLINTEND(readability-magic-numbers)
            }
        }
    }

    template <class PrecisionT>
    static void
    applyMultiQubitOp(std::complex<PrecisionT> *arr, std::size_t num_qubits,
                      const std::complex<PrecisionT> *matrix,
                      const std::vector<std::size_t> &wires, bool inverse) {
        using Pennylane::Util::bitswap;
        constexpr std::size_t one{1};
        PL_ASSERT(num_qubits >= wires.size());

        const std::size_t dim = one << wires.size();
        std::vector<std::size_t> indices(dim);
        std::vector<std::complex<PrecisionT>> coeffs_in(dim, 0.0);

        std::vector<std::size_t> rev_wires(wires.size());
        std::vector<std::size_t> rev_wire_shifts(wires.size());
        for (std::size_t k = 0; k < wires.size(); k++) {
            rev_wires[k] = (num_qubits - 1) - wires[(wires.size() - 1) - k];
            rev_wire_shifts[k] = (one << rev_wires[k]);
        }
        const std::vector<std::size_t> parity =
            Pennylane::Util::revWireParity(rev_wires);
        PL_ASSERT(wires.size() == parity.size() - 1);

        if (inverse) {
            for (std::size_t k = 0; k < exp2(num_qubits - wires.size()); k++) {
                indices = parity2indices(k, parity, rev_wire_shifts);
                for (std::size_t i = 0; i < dim; i++) {
                    coeffs_in[i] = arr[indices[i]];
                }
                for (size_t i = 0; i < dim; i++) {
                    const auto idx = indices[i];
                    arr[idx] = 0.0;

                    for (size_t j = 0; j < dim; j++) {
                        const size_t base_idx = j * dim;
                        arr[idx] +=
                            std::conj(matrix[base_idx + i]) * coeffs_in[j];
                    }
                }
            }
        } else {
            for (std::size_t k = 0; k < exp2(num_qubits - wires.size()); k++) {
                indices = parity2indices(k, parity, rev_wire_shifts);
                for (std::size_t i = 0; i < dim; i++) {
                    coeffs_in[i] = arr[indices[i]];
                }
                for (std::size_t i = 0; i < dim; i++) {
                    const auto index = indices[i];
                    arr[index] = 0.0;
                    const std::size_t base_idx = i * dim;
                    for (std::size_t j = 0; j < dim; j++) {
                        arr[index] += matrix[base_idx + j] * coeffs_in[j];
                    }
                }
            }
        }
    }

    template <class PrecisionT>
    static void applyIdentity(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        static_cast<void>(arr);        // No-op
        static_cast<void>(num_qubits); // No-op
        static_cast<void>(wires);      // No-op
    }

    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);

        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);

        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            const size_t i1 = i0 | rev_wire_shift;
            std::swap(arr[i0], arr[i1]);
        }
    }

    template <class PrecisionT>
    static void applyPauliY(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);

        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            const size_t i1 = i0 | rev_wire_shift;
            const auto v0 = arr[i0];
            const auto v1 = arr[i1];
            arr[i0] = {std::imag(v1), -std::real(v1)};
            arr[i1] = {-std::imag(v0), std::real(v0)};
        }
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            const size_t i1 = i0 | rev_wire_shift;
            arr[i1] *= -1;
        }
    }

    template <class PrecisionT>
    static void applyHadamard(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        constexpr static auto isqrt2 = INVSQRT2<PrecisionT>();
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            const size_t i1 = i0 | rev_wire_shift;
            const std::complex<PrecisionT> v0 = arr[i0];
            const std::complex<PrecisionT> v1 = arr[i1];
            arr[i0] = isqrt2 * v0 + isqrt2 * v1;
            arr[i1] = isqrt2 * v0 - isqrt2 * v1;
        }
    }

    template <class PrecisionT>
    static void applyS(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        const std::complex<PrecisionT> shift =
            (inverse) ? -Pennylane::Util::IMAG<PrecisionT>()
                      : Pennylane::Util::IMAG<PrecisionT>();

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            const size_t i1 = i0 | rev_wire_shift;
            arr[i1] *= shift;
        }
    }

    template <class PrecisionT>
    static void applyT(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        constexpr static auto isqrt2 = INVSQRT2<PrecisionT>();

        const std::complex<PrecisionT> shift = {isqrt2,
                                                inverse ? -isqrt2 : isqrt2};

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            const size_t i1 = i0 | rev_wire_shift;
            arr[i1] *= shift;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyPhaseShift(std::complex<PrecisionT> *arr,
                                const size_t num_qubits,
                                const std::vector<size_t> &wires, bool inverse,
                                ParamT angle) {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        const std::complex<PrecisionT> s =
            inverse ? std::exp(-std::complex<PrecisionT>(0, angle))
                    : std::exp(std::complex<PrecisionT>(0, angle));

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            const size_t i1 = i0 | rev_wire_shift;
            arr[i1] *= s;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRX(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            const size_t i1 = i0 | rev_wire_shift;
            const std::complex<PrecisionT> v0 = arr[i0];
            const std::complex<PrecisionT> v1 = arr[i1];
            arr[i0] = c * v0 +
                      std::complex<PrecisionT>{-imag(v1) * js, real(v1) * js};
            arr[i1] = std::complex<PrecisionT>{-imag(v0) * js, real(v0) * js} +
                      c * v1;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRY(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            const size_t i1 = i0 | rev_wire_shift;
            const std::complex<PrecisionT> v0 = arr[i0];
            const std::complex<PrecisionT> v1 = arr[i1];
            arr[i0] = std::complex<PrecisionT>{c * real(v0) - s * real(v1),
                                               c * imag(v0) - s * imag(v1)};
            arr[i1] = std::complex<PrecisionT>{s * real(v0) + c * real(v1),
                                               s * imag(v0) + c * imag(v1)};
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        PL_ASSERT(wires.size() == 1);

        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};

        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            const size_t i1 = i0 | rev_wire_shift;
            arr[i0] *= shifts[0];
            arr[i1] *= shifts[1];
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRot(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT phi, ParamT theta, ParamT omega) {
        PL_ASSERT(wires.size() == 1);

        const auto rotMat =
            (inverse) ? getRot<std::complex, PrecisionT>(-omega, -theta, -phi)
                      : getRot<std::complex, PrecisionT>(phi, theta, omega);

        applySingleQubitOp(arr, num_qubits, rotMat.data(), wires);
    }

    /* Two-qubit gates */

    template <class PrecisionT>
    static void
    applyCNOT(std::complex<PrecisionT> *arr, const size_t num_qubits,
              const std::vector<size_t> &wires, [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        /* This is faster than iterate over all indices */
        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;

            std::swap(arr[i10], arr[i11]);
        }
    }

    template <class PrecisionT>
    static void applyCY(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        /* This is faster than iterate over all indices */
        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;
            std::complex<PrecisionT> v10 = arr[i10];
            arr[i10] = std::complex<PrecisionT>{std::imag(arr[i11]),
                                                -std::real(arr[i11])};
            arr[i11] =
                std::complex<PrecisionT>{-std::imag(v10), std::real(v10)};
        }
    }

    template <class PrecisionT>
    static void applyCZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;
            arr[i11] *= -1;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRot(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires, bool inverse,
                          ParamT phi, ParamT theta, ParamT omega) {
        PL_ASSERT(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        const auto rotMat =
            (inverse) ? getRot<std::complex, PrecisionT>(-omega, -theta, -phi)
                      : getRot<std::complex, PrecisionT>(phi, theta, omega);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const std::complex<PrecisionT> v0 = arr[i10];
            const std::complex<PrecisionT> v1 = arr[i11];
            arr[i10] = rotMat[0] * v0 + rotMat[1] * v1;
            arr[i11] = rotMat[2] * v0 + rotMat[3] * v1;
        }
    }

    template <class PrecisionT>
    static void applySWAP(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            std::swap(arr[i10], arr[i01]);
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyIsingXX(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        using ComplexT = std::complex<PrecisionT>;
        using std::imag;
        using std::real;
        PL_ASSERT(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const ComplexT v00 = arr[i00];
            const ComplexT v01 = arr[i01];
            const ComplexT v10 = arr[i10];
            const ComplexT v11 = arr[i11];

            arr[i00] = ComplexT{cr * real(v00) + sj * imag(v11),
                                cr * imag(v00) - sj * real(v11)};
            arr[i01] = ComplexT{cr * real(v01) + sj * imag(v10),
                                cr * imag(v01) - sj * real(v10)};
            arr[i10] = ComplexT{cr * real(v10) + sj * imag(v01),
                                cr * imag(v10) - sj * real(v01)};
            arr[i11] = ComplexT{cr * real(v11) + sj * imag(v00),
                                cr * imag(v11) - sj * real(v00)};
        }
    }

    template <class PrecisionT, class ParamT>
    static void
    applyIsingXY(std::complex<PrecisionT> *arr, const size_t num_qubits,
                 const std::vector<size_t> &wires, bool inverse, ParamT angle) {
        using ComplexT = std::complex<PrecisionT>;
        using std::imag;
        using std::real;
        PL_ASSERT(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const ComplexT v00 = arr[i00];
            const ComplexT v01 = arr[i01];
            const ComplexT v10 = arr[i10];
            const ComplexT v11 = arr[i11];

            arr[i00] = ComplexT{real(v00), imag(v00)};
            arr[i01] = ComplexT{cr * real(v01) - sj * imag(v10),
                                cr * imag(v01) + sj * real(v10)};
            arr[i10] = ComplexT{cr * real(v10) - sj * imag(v01),
                                cr * imag(v10) + sj * real(v01)};
            arr[i11] = ComplexT{real(v11), imag(v11)};
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyIsingYY(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        using ComplexT = std::complex<PrecisionT>;
        using std::imag;
        using std::real;
        PL_ASSERT(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const ComplexT v00 = arr[i00];
            const ComplexT v01 = arr[i01];
            const ComplexT v10 = arr[i10];
            const ComplexT v11 = arr[i11];

            arr[i00] = ComplexT{cr * real(v00) - sj * imag(v11),
                                cr * imag(v00) + sj * real(v11)};
            arr[i01] = ComplexT{cr * real(v01) + sj * imag(v10),
                                cr * imag(v01) - sj * real(v10)};
            arr[i10] = ComplexT{cr * real(v10) + sj * imag(v01),
                                cr * imag(v10) - sj * real(v01)};
            arr[i11] = ComplexT{cr * real(v11) - sj * imag(v00),
                                cr * imag(v11) + sj * real(v00)};
        }
    }

    template <class PrecisionT, class ParamT>
    static void
    applyIsingZZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                 const std::vector<size_t> &wires, bool inverse, ParamT angle) {
        PL_ASSERT(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};

        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            arr[i00] *= shifts[0];
            arr[i01] *= shifts[1];
            arr[i10] *= shifts[1];
            arr[i11] *= shifts[0];
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyControlledPhaseShift(std::complex<PrecisionT> *arr,
                                          const size_t num_qubits,
                                          const std::vector<size_t> &wires,
                                          [[maybe_unused]] bool inverse,
                                          ParamT angle) {
        PL_ASSERT(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        const std::complex<PrecisionT> s =
            inverse ? std::exp(-std::complex<PrecisionT>(0, angle))
                    : std::exp(std::complex<PrecisionT>(0, angle));

        /* This is faster than iterate over all indices */
        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;

            arr[i11] *= s;
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyCRX(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        PL_ASSERT(wires.size() == 2);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT js =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const std::complex<PrecisionT> v10 = arr[i10];
            const std::complex<PrecisionT> v11 = arr[i11];

            arr[i10] = std::complex<PrecisionT>{
                c * std::real(v10) + js * std::imag(v11),
                c * std::imag(v10) - js * std::real(v11)};
            arr[i11] = std::complex<PrecisionT>{
                c * std::real(v11) + js * std::imag(v10),
                c * std::imag(v11) - js * std::real(v10)};
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyCRY(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        PL_ASSERT(wires.size() == 2);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const std::complex<PrecisionT> v10 = arr[i10];
            const std::complex<PrecisionT> v11 = arr[i11];

            arr[i10] = std::complex<PrecisionT>{c * real(v10) - s * real(v11),
                                                c * imag(v10) - s * imag(v11)};
            arr[i11] = std::complex<PrecisionT>{s * real(v10) + c * real(v11),
                                                s * imag(v10) + c * imag(v11)};
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyCRZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        PL_ASSERT(wires.size() == 2);

        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};
        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            arr[i10] *= shifts[0];
            arr[i11] *= shifts[1];
        }
    }

    template <class PrecisionT>
    static void applyCSWAP(std::complex<PrecisionT> *arr, size_t num_qubits,
                           const std::vector<size_t> &wires,
                           [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 3);

        const size_t rev_wire0 = num_qubits - wires[2] - 1;
        const size_t rev_wire1 = num_qubits - wires[1] - 1;
        const size_t rev_wire2 = num_qubits - wires[0] - 1;

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const size_t rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;

        auto parity = revWireParity(rev_wire0, rev_wire1, rev_wire2);

        for (size_t k = 0; k < exp2(num_qubits - 3); k++) {
            const size_t i000 = ((k << 3U) & parity[3]) |
                                ((k << 2U) & parity[2]) |
                                ((k << 1U) & parity[1]) | (k & parity[0]);
            const size_t i101 = i000 | rev_wire2_shift | rev_wire0_shift;
            const size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;
            std::swap(arr[i101], arr[i110]);
        }
    }

    template <class PrecisionT>
    static void applyToffoli(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 3);

        const size_t rev_wire0 = num_qubits - wires[2] - 1;
        const size_t rev_wire1 = num_qubits - wires[1] - 1;
        const size_t rev_wire2 = num_qubits - wires[0] - 1;

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const size_t rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;

        auto parity = revWireParity(rev_wire0, rev_wire1, rev_wire2);

        for (size_t k = 0; k < exp2(num_qubits - 3); k++) {
            const size_t i000 = ((k << 3U) & parity[3]) |
                                ((k << 2U) & parity[2]) |
                                ((k << 1U) & parity[1]) | (k & parity[0]);
            const size_t i111 =
                i000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
            const size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;
            std::swap(arr[i111], arr[i110]);
        }
    }

    /* Defined in the .cpp file */
    template <class PrecisionT, class ParamT>
    static void applySingleExcitation(std::complex<PrecisionT> *arr,
                                      size_t num_qubits,
                                      const std::vector<size_t> &wires,
                                      bool inverse, ParamT angle);

    /* Defined in the .cpp file */
    template <class PrecisionT, class ParamT>
    static void applySingleExcitationMinus(std::complex<PrecisionT> *arr,
                                           size_t num_qubits,
                                           const std::vector<size_t> &wires,
                                           bool inverse, ParamT angle);

    /* Defined in the .cpp file */
    template <class PrecisionT, class ParamT>
    static void applySingleExcitationPlus(std::complex<PrecisionT> *arr,
                                          size_t num_qubits,
                                          const std::vector<size_t> &wires,
                                          bool inverse, ParamT angle);

    /* Four-qubit gates*/
    template <class PrecisionT, class ParamT>
    static void applyDoubleExcitation(std::complex<PrecisionT> *arr,
                                      size_t num_qubits,
                                      const std::vector<size_t> &wires,
                                      bool inverse, ParamT angle) {
        PL_ASSERT(wires.size() == 4);
        constexpr std::size_t one{1};
        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        const std::array<std::size_t, 4> rev_wires{
            num_qubits - wires[3] - 1, num_qubits - wires[2] - 1,
            num_qubits - wires[1] - 1, num_qubits - wires[0] - 1};

        const std::array<std::size_t, 4> rev_wire_shifts{
            one << rev_wires[0], one << rev_wires[1], one << rev_wires[2],
            one << rev_wires[3]};

        const auto parity = Pennylane::Util::revWireParity(rev_wires);

        for (size_t k = 0; k < exp2(num_qubits - 4); k++) {
            const std::size_t i0000 = ((k << 4U) & parity[4]) |
                                      ((k << 3U) & parity[3]) |
                                      ((k << 2U) & parity[2]) |
                                      ((k << 1U) & parity[1]) | (k & parity[0]);
            const size_t i0011 =
                i0000 | rev_wire_shifts[1] | rev_wire_shifts[0];
            const size_t i1100 =
                i0000 | rev_wire_shifts[3] | rev_wire_shifts[2];

            const std::complex<PrecisionT> v3 = arr[i0011];
            const std::complex<PrecisionT> v12 = arr[i1100];

            arr[i0011] = cr * v3 - sj * v12;
            arr[i1100] = sj * v3 + cr * v12;
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyDoubleExcitationMinus(std::complex<PrecisionT> *arr,
                                           size_t num_qubits,
                                           const std::vector<size_t> &wires,
                                           bool inverse, ParamT angle) {
        PL_ASSERT(wires.size() == 4);
        constexpr std::size_t one{1};
        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        const std::complex<PrecisionT> e =
            inverse ? std::exp(std::complex<PrecisionT>{0, angle / 2})
                    : std::exp(std::complex<PrecisionT>{0, -angle / 2});

        const std::array<std::size_t, 4> rev_wires{
            num_qubits - wires[3] - 1, num_qubits - wires[2] - 1,
            num_qubits - wires[1] - 1, num_qubits - wires[0] - 1};

        const std::array<std::size_t, 4> rev_wire_shifts{
            one << rev_wires[0], one << rev_wires[1], one << rev_wires[2],
            one << rev_wires[3]};

        const auto parity = Pennylane::Util::revWireParity(rev_wires);

        for (size_t k = 0; k < exp2(num_qubits - 4); k++) {
            const auto indices = parity2indices(k, parity, rev_wire_shifts);
            const std::complex<PrecisionT> v3 = arr[indices[0B0011]];
            const std::complex<PrecisionT> v12 = arr[indices[0B1100]];
            for (const auto &i : indices) {
                arr[i] *= e;
            }
            arr[indices[0B0011]] = cr * v3 - sj * v12;
            arr[indices[0B1100]] = sj * v3 + cr * v12;
        }
    }

    template <class PrecisionT, class ParamT>
    static void applyDoubleExcitationPlus(std::complex<PrecisionT> *arr,
                                          size_t num_qubits,
                                          const std::vector<size_t> &wires,
                                          bool inverse, ParamT angle) {
        PL_ASSERT(wires.size() == 4);
        constexpr std::size_t one{1};
        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        const std::complex<PrecisionT> e =
            inverse ? std::exp(std::complex<PrecisionT>{0, -angle / 2})
                    : std::exp(std::complex<PrecisionT>{0, angle / 2});

        const std::array<std::size_t, 4> rev_wires{
            num_qubits - wires[3] - 1, num_qubits - wires[2] - 1,
            num_qubits - wires[1] - 1, num_qubits - wires[0] - 1};

        const std::array<std::size_t, 4> rev_wire_shifts{
            one << rev_wires[0], one << rev_wires[1], one << rev_wires[2],
            one << rev_wires[3]};

        const auto parity = Pennylane::Util::revWireParity(rev_wires);

        for (size_t k = 0; k < exp2(num_qubits - 4); k++) {
            const auto indices = parity2indices(k, parity, rev_wire_shifts);
            const std::complex<PrecisionT> v3 = arr[indices[0B0011]];
            const std::complex<PrecisionT> v12 = arr[indices[0B1100]];
            for (const auto &i : indices) {
                arr[i] *= e;
            }
            arr[indices[0B0011]] = cr * v3 - sj * v12;
            arr[indices[0B1100]] = sj * v3 + cr * v12;
        }
    }

    /* Multi-qubit gates */
    template <class PrecisionT, class ParamT>
    static void applyMultiRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};
        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};

        size_t wires_parity = 0U;
        for (size_t wire : wires) {
            wires_parity |=
                (static_cast<size_t>(1U) << (num_qubits - wire - 1));
        }

        for (size_t k = 0; k < exp2(num_qubits); k++) {
            arr[k] *= shifts[std::popcount(k & wires_parity) % 2];
        }
    }

    /* Define generators */
    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorPhaseShift(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const auto [parity_high, parity_low] = revWireParity(rev_wire);

        for (size_t k = 0; k < exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & parity_high) | (parity_low & k);
            arr[i0] = std::complex<PrecisionT>{0.0, 0.0};
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(1.0);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingXX(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            std::swap(arr[i00], arr[i11]);
            std::swap(arr[i10], arr[i01]);
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingXY(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            std::swap(arr[i10], arr[i01]);
            arr[i00] = std::complex<PrecisionT>{0.0, 0.0};
            arr[i11] = std::complex<PrecisionT>{0.0, 0.0};
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingYY(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const auto v00 = arr[i00];
            arr[i00] = -arr[i11];
            arr[i11] = -v00;
            std::swap(arr[i10], arr[i01]);
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingZZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i10 = i00 | rev_wire1_shift;

            arr[i10] *= -1;
            arr[i01] *= -1;
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRX(std::complex<PrecisionT> *arr, size_t num_qubits,
                      const std::vector<size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        using ComplexT = std::complex<PrecisionT>;
        PL_ASSERT(wires.size() == 2);
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            arr[i00] = ComplexT{};
            arr[i01] = ComplexT{};

            std::swap(arr[i10], arr[i11]);
        }

        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRY(std::complex<PrecisionT> *arr, size_t num_qubits,
                      const std::vector<size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        using ComplexT = std::complex<PrecisionT>;
        PL_ASSERT(wires.size() == 2);
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            arr[i00] = ComplexT{};
            arr[i01] = ComplexT{};

            const auto v0 = arr[i10];

            arr[i10] = ComplexT{std::imag(arr[i11]), -std::real(arr[i11])};
            arr[i11] = ComplexT{-std::imag(v0), std::real(v0)};
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                      const std::vector<size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        using ComplexT = std::complex<PrecisionT>;
        PL_ASSERT(wires.size() == 2);
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            arr[i00] = ComplexT{};
            arr[i01] = ComplexT{};
            arr[i11] *= -1;
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorControlledPhaseShift(
        std::complex<PrecisionT> *arr, size_t num_qubits,
        const std::vector<size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        using ComplexT = std::complex<PrecisionT>;
        PL_ASSERT(wires.size() == 2);
        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        const auto [parity_high, parity_middle, parity_low] =
            revWireParity(rev_wire0, rev_wire1);

        for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i10 = i00 | rev_wire1_shift;

            arr[i00] = ComplexT{};
            arr[i01] = ComplexT{};
            arr[i10] = ComplexT{};
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(1);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorSingleExcitation(std::complex<PrecisionT> *arr,
                                   size_t num_qubits,
                                   const std::vector<size_t> &wires,
                                   [[maybe_unused]] bool adj) -> PrecisionT;

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorSingleExcitationMinus(
        std::complex<PrecisionT> *arr, size_t num_qubits,
        const std::vector<size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT;

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorSingleExcitationPlus(std::complex<PrecisionT> *arr,
                                       size_t num_qubits,
                                       const std::vector<size_t> &wires,
                                       [[maybe_unused]] bool adj) -> PrecisionT;

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorDoubleExcitation(std::complex<PrecisionT> *arr,
                                   size_t num_qubits,
                                   const std::vector<size_t> &wires,
                                   [[maybe_unused]] bool adj) -> PrecisionT;

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorDoubleExcitationMinus(
        std::complex<PrecisionT> *arr, size_t num_qubits,
        const std::vector<size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT;

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorDoubleExcitationPlus(std::complex<PrecisionT> *arr,
                                       size_t num_qubits,
                                       const std::vector<size_t> &wires,
                                       [[maybe_unused]] bool adj) -> PrecisionT;

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorMultiRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool adj) -> PrecisionT {
        auto wires_parity = static_cast<size_t>(0U);
        for (size_t wire : wires) {
            wires_parity |=
                (static_cast<size_t>(1U) << (num_qubits - wire - 1));
        }

        for (size_t k = 0; k < exp2(num_qubits); k++) {
            arr[k] *= static_cast<PrecisionT>(
                1 - 2 * int(std::popcount(k & wires_parity) % 2));
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }
};

// Matrix operations
extern template void GateImplementationsLM::applySingleQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
extern template void GateImplementationsLM::applySingleQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);
extern template void GateImplementationsLM::applyTwoQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
extern template void GateImplementationsLM::applyTwoQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);
extern template void GateImplementationsLM::applyMultiQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
extern template void GateImplementationsLM::applyMultiQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

// Single-qubit gates
extern template void
GateImplementationsLM::applyIdentity<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyIdentity<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

extern template void
GateImplementationsLM::applyPauliX<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyPauliX<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

extern template void
GateImplementationsLM::applyPauliY<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyPauliY<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

extern template void
GateImplementationsLM::applyPauliZ<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyPauliZ<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

extern template void
GateImplementationsLM::applyHadamard<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyHadamard<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

extern template void
GateImplementationsLM::applyS<float>(std::complex<float> *, size_t,
                                     const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyS<double>(std::complex<double> *, size_t,
                                      const std::vector<size_t> &, bool);

extern template void
GateImplementationsLM::applyT<float>(std::complex<float> *, size_t,
                                     const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyT<double>(std::complex<double> *, size_t,
                                      const std::vector<size_t> &, bool);

extern template void GateImplementationsLM::applyPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsLM::applyRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsLM::applyRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsLM::applyRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void
GateImplementationsLM::applyRot<float, float>(std::complex<float> *, size_t,
                                              const std::vector<size_t> &, bool,
                                              float, float, float);
extern template void
GateImplementationsLM::applyRot<double, double>(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool, double, double, double);

// Two-qubit gates
extern template void
GateImplementationsLM::applyCNOT<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyCNOT<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

extern template void
GateImplementationsLM::applyCY<float>(std::complex<float> *, size_t,
                                      const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyCY<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

extern template void
GateImplementationsLM::applyCZ<float>(std::complex<float> *, size_t,
                                      const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyCZ<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

extern template void
GateImplementationsLM::applySWAP<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applySWAP<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

extern template void GateImplementationsLM::applyIsingXX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyIsingXX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsLM::applyIsingXY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyIsingXY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsLM::applyIsingYY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyIsingYY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsLM::applyIsingZZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyIsingZZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void
GateImplementationsLM::applyControlledPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void
GateImplementationsLM::applyControlledPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsLM::applyCRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyCRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsLM::applyCRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyCRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsLM::applyCRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyCRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void
GateImplementationsLM::applyCRot<float, float>(std::complex<float> *, size_t,
                                               const std::vector<size_t> &,
                                               bool, float, float, float);
extern template void
GateImplementationsLM::applyCRot<double, double>(std::complex<double> *, size_t,
                                                 const std::vector<size_t> &,
                                                 bool, double, double, double);

extern template void GateImplementationsLM::applyMultiRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsLM::applyMultiRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

// Three-qubit gates
extern template void
GateImplementationsLM::applyCSWAP<float>(std::complex<float> *, size_t,
                                         const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyCSWAP<double>(std::complex<double> *, size_t,
                                          const std::vector<size_t> &, bool);

extern template void
GateImplementationsLM::applyToffoli<float>(std::complex<float> *, size_t,
                                           const std::vector<size_t> &, bool);
extern template void
GateImplementationsLM::applyToffoli<double>(std::complex<double> *, size_t,
                                            const std::vector<size_t> &, bool);

// Four-qubit gates
extern template void GateImplementationsLM::applyDoubleExcitation<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

extern template void
GateImplementationsLM::applyDoubleExcitation<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

extern template void
GateImplementationsLM::applyDoubleExcitationMinus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

extern template void
GateImplementationsLM::applyDoubleExcitationMinus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

extern template void
GateImplementationsLM::applyDoubleExcitationPlus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

extern template void
GateImplementationsLM::applyDoubleExcitationPlus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

/* Generators */
extern template auto GateImplementationsLM::applyGeneratorPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsLM::applyGeneratorPhaseShift(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool) -> double;

extern template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRX(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

extern template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRY(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

extern template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRZ(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

extern template auto GateImplementationsLM::applyGeneratorIsingXX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsLM::applyGeneratorIsingXX(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;

extern template auto
GateImplementationsLM::applyGeneratorIsingXY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsLM::applyGeneratorIsingXY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto GateImplementationsLM::applyGeneratorIsingYY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsLM::applyGeneratorIsingYY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;

extern template auto
GateImplementationsLM::applyGeneratorIsingZZ(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsLM::applyGeneratorIsingZZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto GateImplementationsLM::applyGeneratorCRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsLM::applyGeneratorCRX(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

extern template auto GateImplementationsLM::applyGeneratorCRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsLM::applyGeneratorCRY(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

extern template auto GateImplementationsLM::applyGeneratorCRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsLM::applyGeneratorCRZ(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

extern template auto GateImplementationsLM::applyGeneratorControlledPhaseShift(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsLM::applyGeneratorControlledPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto GateImplementationsLM::applyGeneratorSingleExcitation(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsLM::applyGeneratorSingleExcitation(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto GateImplementationsLM::applyGeneratorSingleExcitationMinus(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsLM::applyGeneratorSingleExcitationMinus(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto GateImplementationsLM::applyGeneratorSingleExcitationPlus(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsLM::applyGeneratorSingleExcitationPlus(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto GateImplementationsLM::applyGeneratorDoubleExcitation(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsLM::applyGeneratorDoubleExcitation(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto GateImplementationsLM::applyGeneratorDoubleExcitationMinus(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsLM::applyGeneratorDoubleExcitationMinus(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto GateImplementationsLM::applyGeneratorDoubleExcitationPlus(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsLM::applyGeneratorDoubleExcitationPlus(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto
GateImplementationsLM::applyGeneratorMultiRZ(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsLM::applyGeneratorMultiRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
} // namespace Pennylane::LightningQubit::Gates
