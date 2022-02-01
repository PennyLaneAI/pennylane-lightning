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
 * @file GateOperationsPI.hpp
 * Defines gate operations with precomputed indicies
 */
#pragma once

/// @cond DEV
// Required for compilation with MSVC
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // for C++
#endif
/// @endcond

#include "GateOperations.hpp"
#include "Gates.hpp"
#include "IndicesUtil.hpp"
#include "KernelType.hpp"

#include <complex>
#include <vector>

namespace Pennylane {
/**
 * @brief Kernel functions for gate operations with precomputed indices
 *
 * For given wires, we first compute the indices the gate applies to and use
 * the computed indices to apply the operation.
 *
 * @tparam fp_t Floating point precision of underlying statevector data.
 * */
template <class fp_t> class GateOperationsPI {
  private:
    using GateIndices = IndicesUtil::GateIndices;

  public:
    using scalar_type_t = fp_t;
    using CFP_t = std::complex<fp_t>;

    constexpr static KernelType kernel_id = KernelType::PI;

    constexpr static std::array implemented_gates = {
        GateOperations::PauliX,
        GateOperations::PauliY,
        GateOperations::PauliZ,
        GateOperations::Hadamard,
        GateOperations::S,
        GateOperations::T,
        GateOperations::RX,
        GateOperations::RY,
        GateOperations::RZ,
        GateOperations::PhaseShift,
        GateOperations::Rot,
        GateOperations::ControlledPhaseShift,
        GateOperations::CNOT,
        GateOperations::CY,
        GateOperations::CZ,
        GateOperations::SWAP,
        GateOperations::CRX,
        GateOperations::CRY,
        GateOperations::CRZ,
        GateOperations::CRot,
        GateOperations::Toffoli,
        GateOperations::CSWAP,
        GateOperations::Matrix,
        GateOperations::GeneratorPhaseShift,
        GateOperations::GeneratorCRX,
        GateOperations::GeneratorCRY,
        GateOperations::GeneratorCRZ,
        GateOperations::GeneratorControlledPhaseShift};

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Perfect square matrix in row-major order.
     * @param indices Internal indices participating in the operation.
     * @param externalIndices External indices unaffected by the operation.
     * @param inverse Indicate whether inverse should be taken.
     */
    static void applyMatrix(CFP_t *arr, size_t num_qubits, const CFP_t *matrix,
                            const std::vector<size_t> &wires, bool inverse) {
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        std::vector<CFP_t> v(indices.size());
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            // Gather
            size_t pos = 0;
            for (const size_t &index : indices) {
                v[pos] = shiftedState[index];
                pos++;
            }

            // Apply + scatter
            if (inverse) {
                for (size_t i = 0; i < indices.size(); i++) {
                    size_t index = indices[i];
                    shiftedState[index] = 0;

                    for (size_t j = 0; j < indices.size(); j++) {
                        const size_t baseIndex = j * indices.size();
                        shiftedState[index] +=
                            std::conj(matrix[baseIndex + i]) * v[j];
                    }
                }
            } else {
                for (size_t i = 0; i < indices.size(); i++) {
                    size_t index = indices[i];
                    shiftedState[index] = 0;

                    const size_t baseIndex = i * indices.size();
                    for (size_t j = 0; j < indices.size(); j++) {
                        shiftedState[index] += matrix[baseIndex + j] * v[j];
                    }
                }
            }
        }
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Perfect square matrix in row-major order.
     * @param indices Internal indices participating in the operation.
     * @param externalIndices External indices unaffected by the operation.
     * @param inverse Indicate whether inverse should be taken.
     */
    static void applyMatrix(CFP_t *arr, size_t num_qubits,
                            const std::vector<CFP_t> &matrix,
                            const std::vector<size_t> &wires, bool inverse) {
        if (matrix.size() != Util::exp2(2 * wires.size())) {
            throw std::invalid_argument(
                "The size of matrix does not match with the given "
                "number of wires");
        }
        applyMatrix(arr, num_qubits, matrix.data(), wires, inverse);
    }

    /* Single qubit operators */
    static void applyPauliX(CFP_t *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[0]], shiftedState[indices[1]]);
        }
    }

    static void applyPauliY(CFP_t *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            CFP_t v0 = shiftedState[indices[0]];
            shiftedState[indices[0]] = CFP_t{shiftedState[indices[1]].imag(),
                                             -shiftedState[indices[1]].real()};
            shiftedState[indices[1]] = CFP_t{-v0.imag(), v0.real()};
        }
    }

    static void applyPauliZ(CFP_t *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] = -shiftedState[indices[1]];
        }
    }

    static void applyHadamard(CFP_t *arr, size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;

            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];

            shiftedState[indices[0]] = Util::INVSQRT2<fp_t>() * (v0 + v1);
            shiftedState[indices[1]] = Util::INVSQRT2<fp_t>() * (v0 - v1);
        }
    }

    static void applyS(CFP_t *arr, size_t num_qubits,
                       const std::vector<size_t> &wires, bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const CFP_t shift =
            (inverse) ? -Util::IMAG<fp_t>() : Util::IMAG<fp_t>();

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    static void applyT(CFP_t *arr, size_t num_qubits,
                       const std::vector<size_t> &wires, bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const CFP_t shift =
            (inverse)
                ? std::conj(std::exp(CFP_t(0, static_cast<fp_t>(M_PI / 4))))
                : std::exp(CFP_t(0, static_cast<fp_t>(M_PI / 4)));

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    /* Single qubit operators with a parameter */
    template <class Param_t = fp_t>
    static void applyRX(CFP_t *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        Param_t angle) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const fp_t c = std::cos(angle / 2);
        const fp_t js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] =
                c * v0 + js * CFP_t{-v1.imag(), v1.real()};
            shiftedState[indices[1]] =
                js * CFP_t{-v0.imag(), v0.real()} + c * v1;
        }
    }

    template <class Param_t = fp_t>
    static void applyRY(CFP_t *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        Param_t angle) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const fp_t c = std::cos(angle / 2);
        const fp_t s = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 - s * v1;
            shiftedState[indices[1]] = s * v0 + c * v1;
        }
    }

    template <class Param_t = fp_t>
    static void applyRZ(CFP_t *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        Param_t angle) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const CFP_t first = CFP_t(std::cos(angle / 2), -std::sin(angle / 2));
        const CFP_t second = CFP_t(std::cos(angle / 2), std::sin(angle / 2));
        const CFP_t shift1 = (inverse) ? std::conj(first) : first;
        const CFP_t shift2 = (inverse) ? std::conj(second) : second;

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] *= shift1;
            shiftedState[indices[1]] *= shift2;
        }
    }

    template <class Param_t = fp_t>
    static void applyPhaseShift(CFP_t *arr, size_t num_qubits,
                                const std::vector<size_t> &wires, bool inverse,
                                Param_t angle) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const CFP_t s = inverse ? std::conj(std::exp(CFP_t(0, angle)))
                                : std::exp(CFP_t(0, angle));
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= s;
        }
    }

    template <class Param_t = fp_t>
    static void applyRot(CFP_t *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         Param_t phi, Param_t theta, Param_t omega) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::vector<CFP_t> rot = Gates::getRot<fp_t>(phi, theta, omega);

        const CFP_t t1 = (inverse) ? std::conj(rot[0]) : rot[0];
        const CFP_t t2 = (inverse) ? -rot[1] : rot[1];
        const CFP_t t3 = (inverse) ? -rot[2] : rot[2];
        const CFP_t t4 = (inverse) ? std::conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = t1 * v0 + t2 * v1;
            shiftedState[indices[1]] = t3 * v0 + t4 * v1;
        }
    }

    /* Two qubit operators */
    static void applyCNOT(CFP_t *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
    }

    static void applySWAP(CFP_t *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[1]], shiftedState[indices[2]]);
        }
    }

    static void applyCY(CFP_t *arr, size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            CFP_t v2 = shiftedState[indices[2]];
            shiftedState[indices[2]] = CFP_t{shiftedState[indices[3]].imag(),
                                             -shiftedState[indices[3]].real()};
            shiftedState[indices[3]] = CFP_t{-v2.imag(), v2.real()};
        }
    }

    static void applyCZ(CFP_t *arr, size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[3]] *= -1;
        }
    }

    /* Two qubit operators with a parameter */
    template <class Param_t = fp_t>
    static void applyControlledPhaseShift(CFP_t *arr, size_t num_qubits,
                                          const std::vector<size_t> &wires,
                                          bool inverse, Param_t angle) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const CFP_t s = inverse ? std::conj(std::exp(CFP_t(0, angle)))
                                : std::exp(CFP_t(0, angle));
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[3]] *= s;
        }
    }

    template <class Param_t = fp_t>
    static void applyCRX(CFP_t *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         Param_t angle) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const fp_t c = std::cos(angle / 2);
        const fp_t js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] =
                c * v0 + js * CFP_t{-v1.imag(), v1.real()};
            shiftedState[indices[3]] =
                js * CFP_t{-v0.imag(), v0.real()} + c * v1;
        }
    }

    template <class Param_t = fp_t>
    static void applyCRY(CFP_t *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         Param_t angle) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const fp_t c = std::cos(angle / 2);
        const fp_t s = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 - s * v1;
            shiftedState[indices[3]] = s * v0 + c * v1;
        }
    }

    template <class Param_t = fp_t>
    static void applyCRZ(CFP_t *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         Param_t angle) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const CFP_t m00 =
            (inverse) ? CFP_t(std::cos(angle / 2), std::sin(angle / 2))
                      : CFP_t(std::cos(angle / 2), -std::sin(angle / 2));
        const CFP_t m11 = (inverse)
                              ? CFP_t(std::cos(angle / 2), -std::sin(angle / 2))
                              : CFP_t(std::cos(angle / 2), std::sin(angle / 2));
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[2]] *= m00;
            shiftedState[indices[3]] *= m11;
        }
    }

    template <class Param_t = fp_t>
    static void applyCRot(CFP_t *arr, size_t num_qubits,
                          const std::vector<size_t> &wires, bool inverse,
                          Param_t phi, Param_t theta, Param_t omega) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const auto rot = Gates::getRot<fp_t>(phi, theta, omega);

        const CFP_t t1 = (inverse) ? std::conj(rot[0]) : rot[0];
        const CFP_t t2 = (inverse) ? -rot[1] : rot[1];
        const CFP_t t3 = (inverse) ? -rot[2] : rot[2];
        const CFP_t t4 = (inverse) ? std::conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = t1 * v0 + t2 * v1;
            shiftedState[indices[3]] = t3 * v0 + t4 * v1;
        }
    }

    /* Three-qubit gate */
    static void applyToffoli(CFP_t *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse) {
        assert(wires.size() == 3);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        // Participating swapped indices
        static const size_t op_idx0 = 6;
        static const size_t op_idx1 = 7;
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[op_idx0]],
                      shiftedState[indices[op_idx1]]);
        }
    }

    static void applyCSWAP(CFP_t *arr, size_t num_qubits,
                           const std::vector<size_t> &wires,
                           [[maybe_unused]] bool inverse) {
        assert(wires.size() == 3);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        // Participating swapped indices
        static const size_t op_idx0 = 5;
        static const size_t op_idx1 = 6;
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[op_idx0]],
                      shiftedState[indices[op_idx1]]);
        }
    }

    /* Gate generators */
    static void applyGeneratorPhaseShift(CFP_t *arr, size_t num_qubits,
                                         const std::vector<size_t> &wires,
                                         [[maybe_unused]] bool adj) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = CFP_t{0.0, 0.0};
        }
    }

    static void applyGeneratorCRX(CFP_t *arr, size_t num_qubits,
                                  const std::vector<size_t> &wires,
                                  [[maybe_unused]] bool adj) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = Util::ZERO<fp_t>();
            shiftedState[indices[1]] = Util::ZERO<fp_t>();

            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
    }

    static void applyGeneratorCRY(CFP_t *arr, size_t num_qubits,
                                  const std::vector<size_t> &wires,
                                  [[maybe_unused]] bool adj) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            const auto v0 = shiftedState[indices[2]];
            shiftedState[indices[0]] = Util::ZERO<fp_t>();
            shiftedState[indices[1]] = Util::ZERO<fp_t>();
            shiftedState[indices[2]] = -IMAG<fp_t>() * shiftedState[indices[3]];
            shiftedState[indices[3]] = IMAG<fp_t>() * v0;
        }
    }

    static void applyGeneratorCRZ(CFP_t *arr, size_t num_qubits,
                                  const std::vector<size_t> &wires,
                                  [[maybe_unused]] bool adj) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = Util::ZERO<fp_t>();
            shiftedState[indices[1]] = Util::ZERO<fp_t>();
            shiftedState[indices[3]] *= -1;
        }
    }

    static void
    applyGeneratorControlledPhaseShift(CFP_t *arr, size_t num_qubits,
                                       const std::vector<size_t> &wires,
                                       [[maybe_unused]] bool adj) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = 0;
            shiftedState[indices[1]] = 0;
            shiftedState[indices[2]] = 0;
        }
    }
};
} // namespace Pennylane
