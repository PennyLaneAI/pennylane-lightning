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
 * Defines the class representation for quantum state vectors.
 */

#pragma once

/// @cond DEV
// Required for compilation with MSVC
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // for C++
#endif
/// @endcond

#include <cmath>
#include <complex>
#include <functional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "Gates.hpp"
#include "Util.hpp"
#include "ApplyOperations.hpp"

#include <iostream>

namespace Pennylane {

/**
 * @brief State-vector base class.
 *
 * This class combines a data array managed by a derived class (CRTP) and an implementation
 * of gate operations proviede by GateOperationType (Policy-based design).
 * The bound data is assumed to be complex, and is required to be in either 32-bit (64-bit
 * `complex<float>`) or 64-bit (128-bit `complex<double>`) floating point
 * representation.
 *
 * @tparam fp_t Floating point precision of underlying statevector data.
 */
template <class fp_t, template<class> class GateOperationType, class Derived>
class StateVectorBase {
  public:
    using scalar_type_t = fp_t;
    /**
     * @brief StateVector complex precision type.
     */
    using CFP_t = std::complex<fp_t>;

  private:
    size_t num_qubits_{0};

  protected:
    StateVectorBase() = default;
    StateVectorBase(size_t num_qubits)
        : num_qubits_{num_qubits} 
    {}

  public:
    /**
     * @brief Redefine the number of qubits in the statevector and number of
     * elements.
     *
     * @param qubits New number of qubits represented by statevector.
     */
    void setNumQubits(size_t qubits) {
        num_qubits_ = qubits;
    }

    /**
     * @brief Get the number of qubits represented by the statevector data.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getNumQubits() const -> std::size_t {
        return num_qubits_;
    }

    size_t getLength() const {
        return static_cast<size_t>(Util::exp2(num_qubits_));
    }

    inline auto getData() -> CFP_t {
        return static_cast<Derived*>(this)->getData();
    }

    inline auto getData() const -> const CFP_t* {
        return static_cast<const Derived*>(this)->getData();
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(const std::string &opName, const std::vector<size_t> &wires,
                        bool inverse = false, const std::vector<fp_t> &params = {}) {

        ApplyOperations<fp_t, GateOperationType>::getInstance().
            applyOperation(*this, opName, wires, inverse, params);
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param matrix Arbitrary unitary gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(const std::vector<CFP_t> &matrix,
                        const std::vector<size_t> &wires, bool inverse = false,
                        [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        auto dim = Util::dimSize(matrix);

        if (dim != wires.size()) {
            throw std::invalid_argument(std::string("The supplied gate requires ") +
                                        std::to_string(dim) + " wires, but " +
                                        std::to_string(wires.size()) +
                                        " were supplied."); // TODO: change to std::format in C++20
        }

        GateOperationType<fp_t>::applyOperation(this->getData(), num_qubits_, matrix,
                wires, inverse, params);
    }

    /**
     * @brief Apply multiple gates to the state-vector.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param wires Vector of wires on which to apply index-matched gate name.
     * @param inverse Indicates whether gate at matched index is to be inverted.
     * @param params Optional parameter data for index matched gates.
     */
    void applyOperations(const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &wires,
                         const std::vector<bool> &inverse,
                         const std::vector<std::vector<fp_t>> &params) {
        ApplyOperations<fp_t, GateOperationType>::getInstance()
            .applyOperation(*this, num_qubits_, wires, inverse, params);
    }

    /**
     * @brief Apply multiple gates to the state-vector.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param wires Vector of wires on which to apply index-matched gate name.
     * @param inverse Indicates whether gate at matched index is to be inverted.
     */
    void applyOperations(const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &wires,
                         const std::vector<bool> &inverse) {
        ApplyOperations<fp_t, GateOperationType>::getInstance()
            .applyOperation(*this, num_qubits_, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Perfect square matrix in row-major order.
     * @param indices Internal indices participating in the operation.
     * @param externalIndices External indices unaffected by the operation.
     * @param inverse Indicate whether inverse should be taken.
     */
    void applyMatrix(const std::vector<CFP_t> &matrix, const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        if (static_cast<size_t>(1ULL << (Util::log2(indices.size()) +
                                         Util::log2(externalIndices.size()))) !=
            length_) {
            throw std::out_of_range(
                "The given indices do not match the state-vector length.");
        }

        std::vector<CFP_t> v(indices.size());
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            // Gather
            size_t pos = 0;
            for (const size_t &index : indices) {
                v[pos] = shiftedState[index];
                pos++;
            }

            // Apply + scatter
            for (size_t i = 0; i < indices.size(); i++) {
                size_t index = indices[i];
                shiftedState[index] = 0;

                if (inverse) {
                    for (size_t j = 0; j < indices.size(); j++) {
                        const size_t baseIndex = j * indices.size();
                        shiftedState[index] +=
                            std::conj(matrix[baseIndex + i]) * v[j];
                    }
                } else {
                    const size_t baseIndex = i * indices.size();
                    for (size_t j = 0; j < indices.size(); j++) {
                        shiftedState[index] += matrix[baseIndex + j] * v[j];
                    }
                }
            }
        }
    }

    /**
     * @brief Apply a given matrix directly to the statevector read directly
     * from numpy data. Data can be in 1D or 2D format.
     *
     * @param matrix Pointer from numpy data.
     * @param indices Internal indices participating in the operation.
     * @param externalIndices External indices unaffected by the operation.
     * @param inverse Indicate whether inverse should be taken.
     */
    void applyMatrix(const CFP_t *matrix, const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        if (static_cast<size_t>(1ULL << (Util::log2(indices.size()) +
                                         Util::log2(externalIndices.size()))) !=
            length_) {
            throw std::out_of_range(
                "The given indices do not match the state-vector length.");
        }

        std::vector<CFP_t> v(indices.size());
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            // Gather
            size_t pos = 0;
            for (const size_t &index : indices) {
                v[pos] = shiftedState[index];
                pos++;
            }

            // Apply + scatter
            for (size_t i = 0; i < indices.size(); i++) {
                size_t index = indices[i];
                shiftedState[index] = 0;

                if (inverse) {
                    for (size_t j = 0; j < indices.size(); j++) {
                        const size_t baseIndex = j * indices.size();
                        shiftedState[index] +=
                            std::conj(matrix[baseIndex + i]) * v[j];
                    }
                } else {
                    const size_t baseIndex = i * indices.size();
                    for (size_t j = 0; j < indices.size(); j++) {
                        shiftedState[index] += matrix[baseIndex + j] * v[j];
                    }
                }
            }
        }
    }

    /**
     * @brief Apply PauliX gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyPauliX(const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices,
                     [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[0]], shiftedState[indices[1]]);
        }
    }

    /**
     * @brief Apply PauliY gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyPauliY(const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices,
                     [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            CFP_t v0 = shiftedState[indices[0]];
            shiftedState[indices[0]] = CFP_t{shiftedState[indices[1]].imag(),
                                             -shiftedState[indices[1]].real()};
            shiftedState[indices[1]] = CFP_t{-v0.imag(), v0.real()};
        }
    }

    /**
     * @brief Apply PauliZ gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyPauliZ(const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices,
                     [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] = -shiftedState[indices[1]];
        }
    }

    /**
     * @brief Apply Hadamard gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyHadamard(const std::vector<size_t> &indices,
                       const std::vector<size_t> &externalIndices,
                       [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;

            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];

            shiftedState[indices[0]] = Util::INVSQRT2<fp_t>() * (v0 + v1);
            shiftedState[indices[1]] = Util::INVSQRT2<fp_t>() * (v0 - v1);
        }
    }

    /**
     * @brief Apply S gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyS(const std::vector<size_t> &indices,
                const std::vector<size_t> &externalIndices, bool inverse) {
        const CFP_t shift =
            (inverse) ? -Util::IMAG<fp_t>() : Util::IMAG<fp_t>();

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    /**
     * @brief Apply T gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyT(const std::vector<size_t> &indices,
                const std::vector<size_t> &externalIndices, bool inverse) {
        const CFP_t shift =
            (inverse)
                ? std::conj(std::exp(CFP_t(0, static_cast<fp_t>(M_PI / 4))))
                : std::exp(CFP_t(0, static_cast<fp_t>(M_PI / 4)));

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    /**
     * @brief Apply RX gate operation to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyRX(const std::vector<size_t> &indices,
                 const std::vector<size_t> &externalIndices, bool inverse,
                 Param_t angle) {
        const Param_t c = std::cos(angle / 2);
        const Param_t js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] =
                c * v0 + js * CFP_t{-v1.imag(), v1.real()};
            shiftedState[indices[1]] =
                js * CFP_t{-v0.imag(), v0.real()} + c * v1;
        }
    }
    /**
     * @brief Apply RY gate operation to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyRY(const std::vector<size_t> &indices,
                 const std::vector<size_t> &externalIndices, bool inverse,
                 Param_t angle) {
        const Param_t c = std::cos(angle / 2);
        const Param_t s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 - s * v1;
            shiftedState[indices[1]] = s * v0 + c * v1;
        }
    }
    /**
     * @brief Apply RZ gate operation to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyRZ(const std::vector<size_t> &indices,
                 const std::vector<size_t> &externalIndices, bool inverse,
                 Param_t angle) {
        const CFP_t first = CFP_t(std::cos(angle / 2), -std::sin(angle / 2));
        const CFP_t second = CFP_t(std::cos(angle / 2), std::sin(angle / 2));
        const CFP_t shift1 = (inverse) ? std::conj(first) : first;
        const CFP_t shift2 = (inverse) ? std::conj(second) : second;

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[0]] *= shift1;
            shiftedState[indices[1]] *= shift2;
        }
    }
    /**
     * @brief Apply phase shift gate operation to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Phase shift angle.
     */
    template <typename Param_t = fp_t>
    void applyPhaseShift(const std::vector<size_t> &indices,
                         const std::vector<size_t> &externalIndices, bool inverse,
                         Param_t angle) {
        const CFP_t s = inverse ? std::conj(std::exp(CFP_t(0, angle)))
                                : std::exp(CFP_t(0, angle));
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= s;
        }
    }

    /**
     * @brief Apply controlled phase shift gate operation to given indices of
     * statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Phase shift angle.
     */
    template <typename Param_t = fp_t>
    void applyControlledPhaseShift(const std::vector<size_t> &indices,
                                   const std::vector<size_t> &externalIndices,
                                   bool inverse, Param_t angle) {
        const CFP_t s = inverse ? std::conj(std::exp(CFP_t(0, angle)))
                                : std::exp(CFP_t(0, angle));
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[3]] *= s;
        }
    }

    /**
     * @brief Apply Rot gate \f$RZ(\omega)RY(\theta)RZ(\phi)\f$ to given indices
     * of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param phi Gate rotation parameter \f$\phi\f$.
     * @param theta Gate rotation parameter \f$\theta\f$.
     * @param omega Gate rotation parameter \f$\omega\f$.
     */
    template <typename Param_t = fp_t>
    void applyRot(const std::vector<size_t> &indices,
                  const std::vector<size_t> &externalIndices, bool inverse,
                  Param_t phi, Param_t theta, Param_t omega) {
        const std::vector<CFP_t> rot = Gates::getRot<fp_t>(phi, theta, omega);

        const CFP_t t1 = (inverse) ? std::conj(rot[0]) : rot[0];
        const CFP_t t2 = (inverse) ? -rot[1] : rot[1];
        const CFP_t t3 = (inverse) ? -rot[2] : rot[2];
        const CFP_t t4 = (inverse) ? std::conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = t1 * v0 + t2 * v1;
            shiftedState[indices[1]] = t3 * v0 + t4 * v1;
        }
    }

    /**
     * @brief Apply CNOT (CX) gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyCNOT(const std::vector<size_t> &indices,
                   const std::vector<size_t> &externalIndices,
                   [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
    }

    /**
     * @brief Apply SWAP gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applySWAP(const std::vector<size_t> &indices,
                   const std::vector<size_t> &externalIndices,
                   [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[1]], shiftedState[indices[2]]);
        }
    }
    /**
     * @brief Apply CZ gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyCZ(const std::vector<size_t> &indices,
                 const std::vector<size_t> &externalIndices,
                 [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[3]] *= -1;
        }
    }

    /**
     * @brief Apply CRX gate to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyCRX(const std::vector<size_t> &indices,
                  const std::vector<size_t> &externalIndices, bool inverse,
                  Param_t angle) {
        const Param_t c = std::cos(angle / 2);
        const Param_t js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] =
                c * v0 + js * CFP_t{-v1.imag(), v1.real()};
            shiftedState[indices[3]] =
                js * CFP_t{-v0.imag(), v0.real()} + c * v1;
        }
    }

    /**
     * @brief Apply CRY gate to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyCRY(const std::vector<size_t> &indices,
                  const std::vector<size_t> &externalIndices, bool inverse,
                  Param_t angle) {
        const Param_t c = std::cos(angle / 2);
        const Param_t s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 - s * v1;
            shiftedState[indices[3]] = s * v0 + c * v1;
        }
    }

    /**
     * @brief Apply CRZ gate to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyCRZ(const std::vector<size_t> &indices,
                  const std::vector<size_t> &externalIndices, bool inverse,
                  Param_t angle) {
        const CFP_t m00 =
            (inverse) ? CFP_t(std::cos(angle / 2), std::sin(angle / 2))
                      : CFP_t(std::cos(angle / 2), -std::sin(angle / 2));
        const CFP_t m11 = (inverse)
                              ? CFP_t(std::cos(angle / 2), -std::sin(angle / 2))
                              : CFP_t(std::cos(angle / 2), std::sin(angle / 2));
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[2]] *= m00;
            shiftedState[indices[3]] *= m11;
        }
    }

    /**
     * @brief Apply CRot gate (controlled \f$RZ(\omega)RY(\theta)RZ(\phi)\f$) to
     * given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param phi Gate rotation parameter \f$\phi\f$.
     * @param theta Gate rotation parameter \f$\theta\f$.
     * @param omega Gate rotation parameter \f$\omega\f$.
     */
    template <typename Param_t = fp_t>
    void applyCRot(const std::vector<size_t> &indices,
                   const std::vector<size_t> &externalIndices, bool inverse,
                   Param_t phi, Param_t theta, Param_t omega) {
        const auto rot = Gates::getRot<fp_t>(phi, theta, omega);

        const CFP_t t1 = (inverse) ? std::conj(rot[0]) : rot[0];
        const CFP_t t2 = (inverse) ? -rot[1] : rot[1];
        const CFP_t t3 = (inverse) ? -rot[2] : rot[2];
        const CFP_t t4 = (inverse) ? std::conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = t1 * v0 + t2 * v1;
            shiftedState[indices[3]] = t3 * v0 + t4 * v1;
        }
    }

    /**
     * @brief Apply Toffoli (CCX) gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyToffoli(const std::vector<size_t> &indices,
                      const std::vector<size_t> &externalIndices,
                      [[maybe_unused]] bool inverse) {
        // Participating swapped indices
        static const size_t op_idx0 = 6;
        static const size_t op_idx1 = 7;
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[op_idx0]],
                      shiftedState[indices[op_idx1]]);
        }
    }

    /**
     * @brief Apply CSWAP gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyCSWAP(const std::vector<size_t> &indices,
                    const std::vector<size_t> &externalIndices,
                    [[maybe_unused]] bool inverse) {
        // Participating swapped indices
        static const size_t op_idx0 = 5;
        static const size_t op_idx1 = 6;
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[op_idx0]],
                      shiftedState[indices[op_idx1]]);
        }
    }
};

/**
 * @brief Streaming operator for StateVector data.
 *
 * @tparam T StateVector data precision.
 * @param out Output stream.
 * @param sv StateVector to stream.
 * @return std::ostream&
 */
template <class T, class Derived>
inline auto operator<<(std::ostream &out, const StateVectorBase<T, Derived>& sv)
    -> std::ostream & {
    const auto num_qubits = sv.getNumQubits();
    const auto data = sv.getData();
    const auto length = 1U << num_qubits;
    out << "num_qubits=" << num_qubits << std::endl;
    out << "data=[";
    out << data[0];
    for (size_t i = 1; i < length - 1; i++) {
        out << "," << data[i];
    }
    out << "," << data[length - 1] << "]";

    return out;
}

} // namespace Pennylane
