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

#include "DynamicDispatcher.hpp"
#include "Error.hpp"
#include "Gates.hpp"
#include "Util.hpp"
#include "SelectGateOps.hpp"

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
#include <iostream>

/**
 * @brief This macro defines methods for State-vector class. The kernel_type template
 * argument choose the kernel to run.
 */
#define PENNYLANE_STATEVECTOR_DEFINE_OPS(GATE_NAME)                                \
    template<KernelType kernel_type, typename... Ts>                               \
    inline void apply##GATE_NAME##_(const std::vector<size_t>& wires,                \
                                               bool inverse, Ts... args) {         \
        auto* arr = getData();                                                     \
        SelectGateOps<fp_t, kernel_type>::apply##GATE_NAME(                        \
                    arr, num_qubits_, wires, inverse, std::forward<Ts>(args)...);  \
    }

#define PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(GATE_NAME)                         \
    template<typename... Ts>                                                        \
    inline void apply##GATE_NAME(const std::vector<size_t>& wires, bool inverse,    \
            Ts... args) {                                                           \
        apply##GATE_NAME##_<DEFAULT_KERNEL_FOR_OPS[ \
        static_cast<int>(GateOperations::GATE_NAME)]>(wires, \
                inverse, std::forward<Ts>(args)...);                                \
    }

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
template <class fp_t, class Derived>
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

    [[nodiscard]] size_t getLength() const {
        return static_cast<size_t>(Util::exp2(num_qubits_));
    }

    [[nodiscard]] inline auto getData() -> CFP_t* {
        return static_cast<Derived*>(this)->getData();
    }

    [[nodiscard]] inline auto getData() const -> const CFP_t* {
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
        
        auto* arr = getData();
        DynamicDispatcher<fp_t>::getInstance().
            applyOperation(arr, num_qubits_, opName, wires, inverse, params);
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
        auto* arr = getData();
        DynamicDispatcher<fp_t>::applyOperation(this->getData(), num_qubits_, matrix,
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
        DynamicDispatcher<fp_t>::getInstance()
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
        DynamicDispatcher<fp_t>::getInstance()
            .applyOperation(*this, num_qubits_, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector read directly
     * from numpy data. Data can be in 1D or 2D format.
     *
     * @param matrix Pointer to the array data.
     * @param indices Internal indices participating in the operation.
     * @param externalIndices External indices unaffected by the operation.
     * @param inverse Indicate whether inverse should be taken.
     */
    template<KernelType kernel_type>
    inline void applyMatrix_(const CFP_t* matrix, const std::vector<size_t>& wires, bool inverse) {
        auto* arr = getData();
        SelectGateOps<fp_t, kernel_type>::applyMatrix(arr, num_qubits_, matrix, wires, inverse);
    }
    inline void applyMatrix(const CFP_t* matrix, const std::vector<size_t>& wires, bool inverse) {
        applyMatrix_<DEFAULT_KERNEL_FOR_OPS[static_cast<int>(GateOperations::Matrix)]>(matrix, wires, inverse);
    }
    template<KernelType kernel_type>
    inline void applyMatrix_(const std::vector<CFP_t>& matrix, 
                             const std::vector<size_t>& wires, bool inverse) {
        auto* arr = getData();
        SelectGateOps<fp_t, kernel_type>::applyMatrix(arr, num_qubits_, matrix, wires, inverse);
    }
    inline void applyMatrix(const std::vector<CFP_t>& matrix, 
                            const std::vector<size_t>& wires, bool inverse) {
        applyMatrix_<DEFAULT_KERNEL_FOR_OPS[static_cast<int>(GateOperations::Matrix)]>(matrix, wires, inverse);
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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(PauliX)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(PauliX)

    /**
     * @brief Apply PauliY gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    PENNYLANE_STATEVECTOR_DEFINE_OPS(PauliY)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(PauliY)

    /**
     * @brief Apply PauliZ gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    PENNYLANE_STATEVECTOR_DEFINE_OPS(PauliZ)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(PauliZ)

    /**
     * @brief Apply Hadamard gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    PENNYLANE_STATEVECTOR_DEFINE_OPS(Hadamard)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(Hadamard)

    /**
     * @brief Apply S gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    PENNYLANE_STATEVECTOR_DEFINE_OPS(S)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(S)

    /**
     * @brief Apply T gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    PENNYLANE_STATEVECTOR_DEFINE_OPS(T)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(T)

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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(RX)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(RX)

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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(RY)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(RY)

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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(RZ)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(RZ)

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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(PhaseShift)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(PhaseShift)

    /*
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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(Rot)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(Rot)

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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(ControlledPhaseShift)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(ControlledPhaseShift)

    /**
     * @brief Apply CNOT (CX) gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    PENNYLANE_STATEVECTOR_DEFINE_OPS(CNOT)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(CNOT)

    /**
     * @brief Apply CZ gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    PENNYLANE_STATEVECTOR_DEFINE_OPS(CZ)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(CZ)

    /**
     * @brief Apply SWAP gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    PENNYLANE_STATEVECTOR_DEFINE_OPS(SWAP)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(SWAP)

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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(CRX)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(CRX)

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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(CRY)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(CRY)

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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(CRZ)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(CRZ)

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
    PENNYLANE_STATEVECTOR_DEFINE_OPS(CRot)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(CRot)

    /**
     * @brief Apply Toffoli (CCX) gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    PENNYLANE_STATEVECTOR_DEFINE_OPS(Toffoli)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(Toffoli)

    /**
     * @brief Apply CSWAP gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    PENNYLANE_STATEVECTOR_DEFINE_OPS(CSWAP)
    PENNYLANE_STATEVECTOR_DEFINE_DEFAULT_OPS(CSWAP)
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
