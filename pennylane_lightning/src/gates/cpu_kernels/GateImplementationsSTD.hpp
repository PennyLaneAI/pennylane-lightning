// Copyright 2021 Xanadu Quantum Technologies Inc.

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
#include "PauliGenerator.hpp"

#include "BitUtil.hpp"
#include "Error.hpp"
#include "GateOperation.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"

#include <thrust/iterator/counting_iterator.h>

#include <complex>
#include <execution>
#include <vector>

namespace Pennylane::Gates {
/**
 * @brief A gate operation implementation with less memory.
 *
 * We use a bitwise operation to calculate the indices where the gate
 * applies to on the fly.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 */
class GateImplementationsSTD : public PauliGenerator<GateImplementationsSTD> {
  private:
    /* Alias utility functions */
    static constexpr auto fillLeadingOnes = Util::fillLeadingOnes;
    static constexpr auto fillTrailingOnes = Util::fillTrailingOnes;
    static constexpr auto bitswap = Util::bitswap;

  public:
    constexpr static KernelType kernel_id = KernelType::STD;
    constexpr static std::string_view name = "STD";
    template <typename PrecisionT>
    constexpr static size_t required_alignment =
        std::alignment_of_v<PrecisionT>;
    template <typename PrecisionT>
    constexpr static size_t packed_bytes = sizeof(PrecisionT);

    constexpr static std::array implemented_gates = {
        GateOperation::Identity,
        GateOperation::PauliX};

    constexpr static std::array implemented_generators = {
        GeneratorOperation::RX,
        GeneratorOperation::RY,
        GeneratorOperation::RZ,
    };

    constexpr static std::array implemented_matrices = {
        MatrixOperation::SingleQubitOp};

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
    static inline void
    applySingleQubitOp(std::complex<PrecisionT> *arr, size_t num_qubits,
                       const std::complex<PrecisionT> *matrix,
                       const std::vector<size_t> &wires, bool inverse = false) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        if (inverse) {
            
            std::for_each_n(
                std::execution::par,
                thrust::counting_iterator(0),
                Util::exp2(num_qubits - 1),
                [=](size_t k) {
                    const size_t i0 =
                        ((k << 1U) & wire_parity_inv) | (wire_parity & k);
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
            );
        } else {
            std::for_each_n(
                std::execution::par,
                thrust::counting_iterator(0),
                Util::exp2(num_qubits - 1),
                [=](size_t k) {
                    const size_t i0 =
                        ((k << 1U) & wire_parity_inv) | (wire_parity & k);
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
            );
        }
    }

    template <class PrecisionT>
    static void applyIdentity(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        static_cast<void>(arr);        // No-op
        static_cast<void>(num_qubits); // No-op
        static_cast<void>(wires);      // No-op
    }

    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);

        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);


        std::for_each_n(
            std::execution::par,
            thrust::counting_iterator(0),
            Util::exp2(num_qubits - 1),
            [=](size_t k) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            std::swap(arr[i0], arr[i1]);
        });
    }
};
} // namespace Pennylane::Gates
