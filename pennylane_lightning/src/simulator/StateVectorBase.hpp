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
 * @file StateVectorBase.hpp
 * Defines the class representation for quantum state vectors.
 */

#pragma once

#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "DynamicDispatcher.hpp"
#include "Error.hpp"
#include "Util.hpp"

/// @cond DEV
// Required for compilation with MSVC
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // for C++
#endif
/// @endcond

#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Pennylane {
/**
 * @brief State-vector base class.
 *
 * This class combines a data array managed by a derived class (CRTP) and
 * implementations of gate operations. The bound data is assumed to be complex,
 * and is required to be in either 32-bit (64-bit `complex<float>`) or
 * 64-bit (128-bit `complex<double>`) floating point representation.
 * As this is the base class, we do not add default template arguments.
 *
 * @tparam T Floating point precision of underlying statevector data.
 * @tparam Derived Type of a derived class
 */
template <class T, class Derived> class StateVectorBase {
  public:
    /**
     * @brief StateVector complex precision type.
     */
    using PrecisionT = T;
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    size_t num_qubits_{0};

  protected:
    /**
     * @brief Constructor used by derived classes.
     *
     * @param num_qubits Number of qubits
     */
    explicit StateVectorBase(size_t num_qubits) : num_qubits_{num_qubits} {}

    /**
     * @brief Redefine the number of qubits in the statevector and number of
     * elements.
     *
     * @param qubits New number of qubits represented by statevector.
     */
    void setNumQubits(size_t qubits) { num_qubits_ = qubits; }

  public:
    /**
     * @brief Get the number of qubits represented by the statevector data.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getNumQubits() const -> std::size_t {
        return num_qubits_;
    }

    /**
     * @brief Get the size of the statvector
     *
     * @return The size of the statevector
     */
    [[nodiscard]] size_t getLength() const {
        return static_cast<size_t>(Util::exp2(num_qubits_));
    }

    /**
     * @brief Get the data pointer of the statevector
     *
     * @return The pointer to the data of statevector
     */
    [[nodiscard]] inline auto getData() -> decltype(auto) {
        return static_cast<Derived *>(this)->getData();
    }

    [[nodiscard]] inline auto getData() const -> decltype(auto) {
        return static_cast<const Derived *>(this)->getData();
    }

    [[nodiscard]] inline auto
    getKernelForGate(Gates::GateOperation gate_op) const -> Gates::KernelType {
        return static_cast<const Derived *>(this)->getKernelForGate(gate_op);
    }

    [[nodiscard]] inline auto
    getKernelForGenerator(Gates::GeneratorOperation gntr_op) const
        -> Gates::KernelType {
        return static_cast<const Derived *>(this)->getKernelForGenerator(
            gntr_op);
    }

    [[nodiscard]] inline auto
    getKernelForMatrix(Gates::MatrixOperation mat_op) const
        -> Gates::KernelType {
        return static_cast<const Derived *>(this)->getKernelForMatrix(mat_op);
    }

    /**
     * @brief Compare two statevectors.
     *
     * @tparam RhsDerived The derived class for another statevector.
     * @param rhs Another statevector to compare.
     * @return bool
     */
    template <class RhsDerived>
    bool operator==(const StateVectorBase<PrecisionT, RhsDerived> &rhs) {
        if (num_qubits_ != rhs.getNumQubits()) {
            return false;
        }
        const ComplexPrecisionT *data1 = getData();
        const ComplexPrecisionT *data2 = rhs.getData();
        for (size_t k = 0; k < getLength(); k++) {
            if (data1[k] != data2[k]) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Apply a single gate to the state-vector using a given kernel.
     *
     * @param kernel Kernel to run the operation.
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(Gates::KernelType kernel, const std::string &opName,
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        auto *arr = getData();
        DynamicDispatcher<PrecisionT>::getInstance().applyOperation(
            kernel, arr, num_qubits_, opName, wires, inverse, params);
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        auto *arr = getData();
        auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gate_op = dispatcher.strToGateOp(opName);
        dispatcher.applyOperation(getKernelForGate(gate_op), arr, num_qubits_,
                                  gate_op, wires, inverse, params);
    }

    /**
     * @brief Apply multiple gates to the state-vector.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_inverse Indicates whether gate at matched index is to be
     * inverted.
     * @param ops_params Optional parameter data for index matched gates.
     */
    void
    applyOperations(const std::vector<std::string> &ops,
                    const std::vector<std::vector<size_t>> &ops_wires,
                    const std::vector<bool> &ops_inverse,
                    const std::vector<std::vector<PrecisionT>> &ops_params) {
        const size_t numOperations = ops.size();
        PL_ABORT_IF(
            numOperations != ops_wires.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        PL_ABORT_IF(
            numOperations != ops_inverse.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        PL_ABORT_IF(
            numOperations != ops_params.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], ops_wires[i], ops_inverse[i], ops_params[i]);
        }
    }

    /**
     * @brief Apply multiple gates to the state-vector.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_inverse Indicates whether gate at matched index is to be
     * inverted.
     */
    void applyOperations(const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &ops_wires,
                         const std::vector<bool> &ops_inverse) {
        const size_t numOperations = ops.size();
        if (numOperations != ops_wires.size()) {
            PL_ABORT(
                "Invalid arguments: number of operations, wires, and inverses "
                "must all be equal");
        }
        if (numOperations != ops_inverse.size()) {
            PL_ABORT(
                "Invalid arguments: number of operations, wires and inverses"
                "must all be equal");
        }
        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], ops_wires[i], ops_inverse[i], {});
        }
    }

    /**
     * @brief Apply a single generator to the state-vector using a given kernel.
     *
     * @param kernel Kernel to run the operation.
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] inline auto applyGenerator(Gates::KernelType kernel,
                                             const std::string &opName,
                                             const std::vector<size_t> &wires,
                                             bool adj = false) -> PrecisionT {
        auto *arr = getData();
        return DynamicDispatcher<PrecisionT>::getInstance().applyGenerator(
            kernel, arr, num_qubits_, opName, wires, adj);
    }

    /**
     * @brief Apply a single generator to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires the gate applies to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] auto applyGenerator(const std::string &opName,
                                      const std::vector<size_t> &wires,
                                      bool adj = false) -> PrecisionT {
        auto *arr = getData();
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gntr_op = dispatcher.strToGeneratorOp(opName);
        return dispatcher.applyGenerator(getKernelForGenerator(gntr_op), arr,
                                         num_qubits_, opName, wires, adj);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a given
     * kernel.
     *
     * @param kernel Kernel to run the operation
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(Gates::KernelType kernel,
                            const ComplexPrecisionT *matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        using Gates::MatrixOperation;

        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        auto *arr = getData();

        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");

        dispatcher.applyMatrix(kernel, arr, num_qubits_, matrix, wires,
                               inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a given
     * kernel.
     *
     * @param kernel Kernel to run the operation
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(Gates::KernelType kernel,
                            const std::vector<ComplexPrecisionT> &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        using Gates::MatrixOperation;

        PL_ABORT_IF(matrix.size() != Util::exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");

        applyMatrix(kernel, matrix.data(), wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(const ComplexPrecisionT *matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        using Gates::MatrixOperation;

        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");

        const auto kernel = [n_wires = wires.size(), this]() {
            switch (n_wires) {
            case 1:
                return getKernelForMatrix(MatrixOperation::SingleQubitOp);
            case 2:
                return getKernelForMatrix(MatrixOperation::TwoQubitOp);
            default:
                return getKernelForMatrix(MatrixOperation::MultiQubitOp);
            }
        }();
        applyMatrix(kernel, matrix, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <typename Alloc>
    inline void applyMatrix(const std::vector<ComplexPrecisionT, Alloc> &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(matrix.size() != Util::exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");

        applyMatrix(matrix.data(), wires, inverse);
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
inline auto operator<<(std::ostream &out, const StateVectorBase<T, Derived> &sv)
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
