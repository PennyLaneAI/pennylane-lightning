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
 * Defines gate operations with precomputed indices
 */
#pragma once

/// @cond DEV
// Required for compilation with MSVC
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // for C++
#endif
/// @endcond

#include "BitUtil.hpp"
#include "GateOperation.hpp"
#include "GateUtil.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"
#include "PauliGenerator.hpp"

#include <complex>
#include <vector>

namespace Pennylane::Gates {
/**
 * @brief Kernel functions for gate operations with precomputed indices
 *
 * For given wires, we first compute the indices the gate applies to and use
 * the computed indices to apply the operation.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * */
class GateImplementationsPI : public PauliGenerator<GateImplementationsPI> {
  public:
    constexpr static KernelType kernel_id = KernelType::PI;
    constexpr static std::string_view name = "PI";
    constexpr static uint32_t data_alignment_in_bytes = 1;

    constexpr static std::array implemented_gates = {
        GateOperation::Identity,
        GateOperation::PauliX,
        GateOperation::PauliY,
        GateOperation::PauliZ,
        GateOperation::Hadamard,
        GateOperation::S,
        GateOperation::T,
        GateOperation::RX,
        GateOperation::RY,
        GateOperation::RZ,
        GateOperation::PhaseShift,
        GateOperation::Rot,
        GateOperation::ControlledPhaseShift,
        GateOperation::CNOT,
        GateOperation::CY,
        GateOperation::CZ,
        GateOperation::SWAP,
        GateOperation::IsingXX,
        GateOperation::IsingYY,
        GateOperation::IsingZZ,
        GateOperation::CRX,
        GateOperation::CRY,
        GateOperation::CRZ,
        GateOperation::CRot,
        GateOperation::Toffoli,
        GateOperation::CSWAP,
        GateOperation::MultiRZ,
        GateOperation::Matrix};
    constexpr static std::array implemented_generators = {
        GeneratorOperation::RX,  GeneratorOperation::RY,
        GeneratorOperation::RZ,  GeneratorOperation::PhaseShift,
        GeneratorOperation::CRX, GeneratorOperation::CRY,
        GeneratorOperation::CRZ, GeneratorOperation::ControlledPhaseShift};

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT>
    static void applyMatrix(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::complex<PrecisionT> *matrix,
                            const std::vector<size_t> &wires, bool inverse) {
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        std::vector<std::complex<PrecisionT>> v(indices.size());
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
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
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT>
    static void applyMatrix(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<std::complex<PrecisionT>> &matrix,
                            const std::vector<size_t> &wires, bool inverse) {
        if (matrix.size() != Util::exp2(2 * wires.size())) {
            throw std::invalid_argument(
                "The size of matrix does not match with the given "
                "number of wires");
        }
        applyMatrix(arr, num_qubits, matrix.data(), wires, inverse);
    }

    /* Single qubit operators */
    template <class PrecisionT>
    static void applyIdentity(std::complex<PrecisionT> *arr, size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        static_cast<void>(arr);        // No-op
        static_cast<void>(num_qubits); // No-op
        static_cast<void>(wires);      // No-op
    }

    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[0]], shiftedState[indices[1]]);
        }
    }

    template <class PrecisionT>
    static void applyPauliY(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            shiftedState[indices[0]] =
                std::complex<PrecisionT>{shiftedState[indices[1]].imag(),
                                         -shiftedState[indices[1]].real()};
            shiftedState[indices[1]] =
                std::complex<PrecisionT>{-v0.imag(), v0.real()};
        }
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] = -shiftedState[indices[1]];
        }
    }

    template <class PrecisionT>
    static void applyHadamard(std::complex<PrecisionT> *arr, size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];

            shiftedState[indices[0]] = Util::INVSQRT2<PrecisionT>() * (v0 + v1);
            shiftedState[indices[1]] = Util::INVSQRT2<PrecisionT>() * (v0 - v1);
        }
    }

    template <class PrecisionT>
    static void applyS(std::complex<PrecisionT> *arr, size_t num_qubits,
                       const std::vector<size_t> &wires, bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const std::complex<PrecisionT> shift =
            (inverse) ? -Util::IMAG<PrecisionT>() : Util::IMAG<PrecisionT>();

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    template <class PrecisionT>
    static void applyT(std::complex<PrecisionT> *arr, size_t num_qubits,
                       const std::vector<size_t> &wires, bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> shift =
            (inverse) ? std::conj(std::exp(std::complex<PrecisionT>(
                            0, static_cast<PrecisionT>(M_PI / 4))))
                      : std::exp(std::complex<PrecisionT>(
                            0, static_cast<PrecisionT>(M_PI / 4)));

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    /* Single qubit operators with a parameter */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyPhaseShift(std::complex<PrecisionT> *arr,
                                size_t num_qubits,
                                const std::vector<size_t> &wires, bool inverse,
                                ParamT angle) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const std::complex<PrecisionT> s =
            inverse ? std::conj(std::exp(std::complex<PrecisionT>(0, angle)))
                    : std::exp(std::complex<PrecisionT>(0, angle));
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= s;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRX(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] =
                c * v0 + js * std::complex<PrecisionT>{-v1.imag(), v1.real()};
            shiftedState[indices[1]] =
                js * std::complex<PrecisionT>{-v0.imag(), v0.real()} + c * v1;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRY(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 - s * v1;
            shiftedState[indices[1]] = s * v0 + c * v1;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>(std::cos(angle / 2), -std::sin(angle / 2));
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>(std::cos(angle / 2), std::sin(angle / 2));
        const std::complex<PrecisionT> shift1 =
            (inverse) ? std::conj(first) : first;
        const std::complex<PrecisionT> shift2 =
            (inverse) ? std::conj(second) : second;

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] *= shift1;
            shiftedState[indices[1]] *= shift2;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRot(std::complex<PrecisionT> *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT phi, ParamT theta, ParamT omega) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::vector<std::complex<PrecisionT>> rot =
            Gates::getRot<PrecisionT>(phi, theta, omega);

        const std::complex<PrecisionT> t1 =
            (inverse) ? std::conj(rot[0]) : rot[0];
        const std::complex<PrecisionT> t2 = (inverse) ? -rot[1] : rot[1];
        const std::complex<PrecisionT> t3 = (inverse) ? -rot[2] : rot[2];
        const std::complex<PrecisionT> t4 =
            (inverse) ? std::conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = t1 * v0 + t2 * v1;
            shiftedState[indices[1]] = t3 * v0 + t4 * v1;
        }
    }

    /* Two qubit operators */
    template <class PrecisionT>
    static void applyCNOT(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
    }

    template <class PrecisionT>
    static void applyCY(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::complex<PrecisionT> v2 = shiftedState[indices[2]];
            shiftedState[indices[2]] =
                std::complex<PrecisionT>{shiftedState[indices[3]].imag(),
                                         -shiftedState[indices[3]].real()};
            shiftedState[indices[3]] =
                std::complex<PrecisionT>{-v2.imag(), v2.real()};
        }
    }

    template <class PrecisionT>
    static void applyCZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[3]] *= -1;
        }
    }

    template <class PrecisionT>
    static void applySWAP(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[1]], shiftedState[indices[2]]);
        }
    }

    /* Two qubit operators with a parameter */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingXX(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        using ComplexPrecisionT = std::complex<PrecisionT>;
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            const auto v0 = shiftedState[indices[0]];
            const auto v1 = shiftedState[indices[1]];
            const auto v2 = shiftedState[indices[2]];
            const auto v3 = shiftedState[indices[3]];

            shiftedState[indices[0]] = ComplexPrecisionT{
                cr * real(v0) + sj * imag(v3), cr * imag(v0) - sj * real(v3)};
            shiftedState[indices[1]] = ComplexPrecisionT{
                cr * real(v1) + sj * imag(v2), cr * imag(v1) - sj * real(v2)};
            shiftedState[indices[2]] = ComplexPrecisionT{
                cr * real(v2) + sj * imag(v1), cr * imag(v2) - sj * real(v1)};
            shiftedState[indices[3]] = ComplexPrecisionT{
                cr * real(v3) + sj * imag(v0), cr * imag(v3) - sj * real(v0)};
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingYY(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        using ComplexPrecisionT = std::complex<PrecisionT>;
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            const auto v0 = shiftedState[indices[0]];
            const auto v1 = shiftedState[indices[1]];
            const auto v2 = shiftedState[indices[2]];
            const auto v3 = shiftedState[indices[3]];

            shiftedState[indices[0]] = ComplexPrecisionT{
                cr * real(v0) - sj * imag(v3), cr * imag(v0) + sj * real(v3)};
            shiftedState[indices[1]] = ComplexPrecisionT{
                cr * real(v1) + sj * imag(v2), cr * imag(v1) - sj * real(v2)};
            shiftedState[indices[2]] = ComplexPrecisionT{
                cr * real(v2) + sj * imag(v1), cr * imag(v2) - sj * real(v1)};
            shiftedState[indices[3]] = ComplexPrecisionT{
                cr * real(v3) - sj * imag(v0), cr * imag(v3) + sj * real(v0)};
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingZZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};

        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            shiftedState[indices[0]] *= shifts[0];
            shiftedState[indices[1]] *= shifts[1];
            shiftedState[indices[2]] *= shifts[1];
            shiftedState[indices[3]] *= shifts[0];
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyControlledPhaseShift(std::complex<PrecisionT> *arr,
                                          size_t num_qubits,
                                          const std::vector<size_t> &wires,
                                          bool inverse, ParamT angle) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> s =
            inverse ? std::conj(std::exp(std::complex<PrecisionT>(0, angle)))
                    : std::exp(std::complex<PrecisionT>(0, angle));
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[3]] *= s;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRX(std::complex<PrecisionT> *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[2]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] =
                c * v0 + js * std::complex<PrecisionT>{-v1.imag(), v1.real()};
            shiftedState[indices[3]] =
                js * std::complex<PrecisionT>{-v0.imag(), v0.real()} + c * v1;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRY(std::complex<PrecisionT> *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[2]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 - s * v1;
            shiftedState[indices[3]] = s * v0 + c * v1;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const std::complex<PrecisionT> m00 =
            (inverse) ? std::complex<PrecisionT>(std::cos(angle / 2),
                                                 std::sin(angle / 2))
                      : std::complex<PrecisionT>(std::cos(angle / 2),
                                                 -std::sin(angle / 2));
        const std::complex<PrecisionT> m11 =
            (inverse) ? std::complex<PrecisionT>(std::cos(angle / 2),
                                                 -std::sin(angle / 2))
                      : std::complex<PrecisionT>(std::cos(angle / 2),
                                                 std::sin(angle / 2));
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[2]] *= m00;
            shiftedState[indices[3]] *= m11;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRot(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires, bool inverse,
                          ParamT phi, ParamT theta, ParamT omega) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const auto rot = Gates::getRot<PrecisionT>(phi, theta, omega);

        const std::complex<PrecisionT> t1 =
            (inverse) ? std::conj(rot[0]) : rot[0];
        const std::complex<PrecisionT> t2 = (inverse) ? -rot[1] : rot[1];
        const std::complex<PrecisionT> t3 = (inverse) ? -rot[2] : rot[2];
        const std::complex<PrecisionT> t4 =
            (inverse) ? std::conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[2]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = t1 * v0 + t2 * v1;
            shiftedState[indices[3]] = t3 * v0 + t4 * v1;
        }
    }

    /* Three-qubit gate */
    template <class PrecisionT>
    static void applyToffoli(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse) {
        assert(wires.size() == 3);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        // Participating swapped indices
        static const size_t op_idx0 = 6;
        static const size_t op_idx1 = 7;
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[op_idx0]],
                      shiftedState[indices[op_idx1]]);
        }
    }

    template <class PrecisionT>
    static void applyCSWAP(std::complex<PrecisionT> *arr, size_t num_qubits,
                           const std::vector<size_t> &wires,
                           [[maybe_unused]] bool inverse) {
        assert(wires.size() == 3);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        // Participating swapped indices
        static const size_t op_idx0 = 5;
        static const size_t op_idx1 = 6;
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[op_idx0]],
                      shiftedState[indices[op_idx1]]);
        }
    }

    /* Multi-qubit gates */
    template <class PrecisionT, class ParamT>
    static void applyMultiRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};

        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};

        // Participating swapped indices
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            for (size_t k = 0; k < indices.size(); k++) {
                shiftedState[indices[k]] *= shifts[Util::popcount(k) % 2];
            }
        }
    }

    /* Gate generators */
    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorPhaseShift(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool adj) -> PrecisionT {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = std::complex<PrecisionT>{0.0, 0.0};
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(1.0);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRX(std::complex<PrecisionT> *arr, size_t num_qubits,
                      const std::vector<size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = Util::ZERO<PrecisionT>();
            shiftedState[indices[1]] = Util::ZERO<PrecisionT>();

            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRY(std::complex<PrecisionT> *arr, size_t num_qubits,
                      const std::vector<size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const auto v0 = shiftedState[indices[2]];
            shiftedState[indices[0]] = Util::ZERO<PrecisionT>();
            shiftedState[indices[1]] = Util::ZERO<PrecisionT>();
            shiftedState[indices[2]] =
                -Util::IMAG<PrecisionT>() * shiftedState[indices[3]];
            shiftedState[indices[3]] = Util::IMAG<PrecisionT>() * v0;
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                      const std::vector<size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = Util::ZERO<PrecisionT>();
            shiftedState[indices[1]] = Util::ZERO<PrecisionT>();
            shiftedState[indices[3]] *= -1;
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorControlledPhaseShift(
        std::complex<PrecisionT> *arr, size_t num_qubits,
        const std::vector<size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = 0;
            shiftedState[indices[1]] = 0;
            shiftedState[indices[2]] = 0;
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(1);
    }
};
} // namespace Pennylane::Gates
