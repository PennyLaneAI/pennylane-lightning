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
 * Minimal class for the Lightning qubit state vector interfacing with the
 * dynamic dispatcher and threading functionalities. This class is a bridge
 * between the base (agnostic) class and specializations for distinct data
 * storage types.
 */

#pragma once
#include <complex>
#include <unordered_map>

#include "CPUMemoryModel.hpp"
#include "GateOperation.hpp"
#include "KernelMap.hpp"
#include "KernelType.hpp"
#include "StateVectorBase.hpp"
#include "Threading.hpp"
#include "cpu_kernels/GateImplementationsLM.hpp"

/// @cond DEV
namespace {
using Pennylane::LightningQubit::Util::Threading;
using Pennylane::Util::CPUMemoryModel;
using Pennylane::Util::exp2;
using Pennylane::Util::squaredNorm;
using namespace Pennylane::LightningQubit::Gates;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit {
/**
 * @brief Lightning qubit state vector class.
 *
 * Minimal class, without data storage, for the Lightning qubit state vector.
 * This class interfaces with the dynamic dispatcher and threading
 * functionalities and is a bridge between the base (agnostic) class and
 * specializations for distinct data storage types.
 *
 * @tparam PrecisionT Floating point precision of underlying state vector data.
 * @tparam Derived Derived class for CRTP.
 */
template <class PrecisionT, class Derived>
class StateVectorLQubit : public StateVectorBase<PrecisionT, Derived> {
  public:
    using ComplexT = std::complex<PrecisionT>;
    using MemoryStorageT = Pennylane::Util::MemoryStorageLocation::Undefined;

  protected:
    const Threading threading_;
    const CPUMemoryModel memory_model_;

  private:
    using BaseType = StateVectorBase<PrecisionT, Derived>;
    using GateKernelMap = std::unordered_map<GateOperation, KernelType>;
    using GeneratorKernelMap =
        std::unordered_map<GeneratorOperation, KernelType>;
    using MatrixKernelMap = std::unordered_map<MatrixOperation, KernelType>;
    using ControlledGateKernelMap =
        std::unordered_map<ControlledGateOperation, KernelType>;
    using ControlledGeneratorKernelMap =
        std::unordered_map<ControlledGeneratorOperation, KernelType>;
    using ControlledMatrixKernelMap =
        std::unordered_map<ControlledMatrixOperation, KernelType>;

    GateKernelMap kernel_for_gates_;
    GeneratorKernelMap kernel_for_generators_;
    MatrixKernelMap kernel_for_matrices_;
    ControlledGateKernelMap kernel_for_controlled_gates_;
    ControlledGeneratorKernelMap kernel_for_controlled_generators_;
    ControlledMatrixKernelMap kernel_for_controlled_matrices_;

    /**
     * @brief Internal function to set kernels for all operations depending on
     * provided dispatch options.
     *
     * @param num_qubits Number of qubits of the statevector
     * @param threading Threading option
     * @param memory_model Memory model
     */
    void setKernels(std::size_t num_qubits, Threading threading,
                    CPUMemoryModel memory_model) {
        using KernelMap::OperationKernelMap;
        kernel_for_gates_ =
            OperationKernelMap<GateOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
        kernel_for_generators_ =
            OperationKernelMap<GeneratorOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
        kernel_for_matrices_ =
            OperationKernelMap<MatrixOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
        kernel_for_controlled_gates_ =
            OperationKernelMap<ControlledGateOperation>::getInstance()
                .getKernelMap(num_qubits, threading, memory_model);
        kernel_for_controlled_generators_ =
            OperationKernelMap<ControlledGeneratorOperation>::getInstance()
                .getKernelMap(num_qubits, threading, memory_model);
        kernel_for_controlled_matrices_ =
            OperationKernelMap<ControlledMatrixOperation>::getInstance()
                .getKernelMap(num_qubits, threading, memory_model);
    }

    /**
     * @brief Get a kernel for a gate operation.
     *
     * @param gate_op Gate operation
     * @return KernelType
     */
    [[nodiscard]] inline auto getKernelForGate(GateOperation gate_op) const
        -> KernelType {
        return kernel_for_gates_.at(gate_op);
    }

    /**
     * @brief Get a kernel for a controlled gate operation.
     *
     * @param gate_op Gate operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForControlledGate(ControlledGateOperation gate_op) const
        -> KernelType {
        return kernel_for_controlled_gates_.at(gate_op);
    }

    /**
     * @brief Get a kernel for a generator operation.
     *
     * @param gen_op Generator operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForGenerator(GeneratorOperation gen_op) const -> KernelType {
        return kernel_for_generators_.at(gen_op);
    }

    /**
     * @brief Get a kernel for a controlled generator operation.
     *
     * @param gen_op Generator operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForControlledGenerator(ControlledGeneratorOperation gen_op) const
        -> KernelType {
        return kernel_for_controlled_generators_.at(gen_op);
    }

    /**
     * @brief Get a kernel for a matrix operation.
     *
     * @param mat_op Matrix operation
     * @return KernelType
     */
    [[nodiscard]] inline auto getKernelForMatrix(MatrixOperation mat_op) const
        -> KernelType {
        return kernel_for_matrices_.at(mat_op);
    }

    /**
     * @brief Get a kernel for a controlled matrix operation.
     *
     * @param mat_op Controlled matrix operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForControlledMatrix(ControlledMatrixOperation mat_op) const
        -> KernelType {
        return kernel_for_controlled_matrices_.at(mat_op);
    }

    /**
     * @brief Get kernels for all gate operations.
     */
    [[nodiscard]] inline auto
    getGateKernelMap() const & -> const GateKernelMap & {
        return kernel_for_gates_;
    }

    [[nodiscard]] inline auto getGateKernelMap() && -> GateKernelMap {
        return kernel_for_gates_;
    }

    /**
     * @brief Get kernels for all controlled gate operations.
     */
    [[nodiscard]] inline auto
    getControlledGateKernelMap() const & -> const ControlledGateKernelMap & {
        return kernel_for_controlled_gates_;
    }

    [[nodiscard]] inline auto
    getControlledGateKernelMap() && -> ControlledGateKernelMap {
        return kernel_for_controlled_gates_;
    }

    /**
     * @brief Get kernels for all generator operations.
     */
    [[nodiscard]] inline auto
    getGeneratorKernelMap() const & -> const GeneratorKernelMap & {
        return kernel_for_generators_;
    }

    [[nodiscard]] inline auto getGeneratorKernelMap() && -> GeneratorKernelMap {
        return kernel_for_generators_;
    }

    /**
     * @brief Get kernels for all controlled generator operations.
     */
    [[nodiscard]] inline auto getControlledGeneratorKernelMap() const & -> const
        ControlledGeneratorKernelMap & {
        return kernel_for_controlled_generators_;
    }

    [[nodiscard]] inline auto
    getControlledGeneratorKernelMap() && -> ControlledGeneratorKernelMap {
        return kernel_for_controlled_generators_;
    }

    /**
     * @brief Get kernels for all matrix operations.
     */
    [[nodiscard]] inline auto
    getMatrixKernelMap() const & -> const MatrixKernelMap & {
        return kernel_for_matrices_;
    }

    [[nodiscard]] inline auto getMatrixKernelMap() && -> MatrixKernelMap {
        return kernel_for_matrices_;
    }

    /**
     * @brief Get kernels for all controlled matrix operations.
     */
    [[nodiscard]] inline auto getControlledMatrixKernelMap() const & -> const
        ControlledMatrixKernelMap & {
        return kernel_for_controlled_matrices_;
    }

    [[nodiscard]] inline auto
    getControlledMatrixKernelMap() && -> ControlledMatrixKernelMap {
        return kernel_for_controlled_matrices_;
    }

  protected:
    explicit StateVectorLQubit(std::size_t num_qubits, Threading threading,
                               CPUMemoryModel memory_model)
        : BaseType(num_qubits), threading_{threading},
          memory_model_{memory_model} {
        setKernels(num_qubits, threading, memory_model);
    }

  public:
    /**
     * @brief Get the statevector's memory model.
     */
    [[nodiscard]] inline CPUMemoryModel memoryModel() const {
        return memory_model_;
    }

    /**
     * @brief Get the statevector's threading mode.
     */
    [[nodiscard]] inline Threading threading() const { return threading_; }

    /**
     *  @brief Returns a tuple containing the gate, generator, and controlled
     * matrix kernel maps respectively.
     */
    [[nodiscard]] auto getSupportedKernels() const & -> std::tuple<
        const GateKernelMap &, const GeneratorKernelMap &,
        const MatrixKernelMap &, const ControlledGateKernelMap &,
        const ControlledGeneratorKernelMap &,
        const ControlledMatrixKernelMap &> {
        return {
            getGateKernelMap(),
            getGeneratorKernelMap(),
            getMatrixKernelMap(),
            getControlledGateKernelMap(),
            getControlledGeneratorKernelMap(),
            getControlledMatrixKernelMap(),
        };
    }

    [[nodiscard]] auto getSupportedKernels() && -> std::tuple<
        GateKernelMap &&, GeneratorKernelMap &&, MatrixKernelMap &&,
        ControlledGateKernelMap &&, ControlledGeneratorKernelMap &&,
        ControlledMatrixKernelMap &&> {
        return {
            getGateKernelMap(),
            getGeneratorKernelMap(),
            getMatrixKernelMap(),
            getControlledGateKernelMap(),
            getControlledGeneratorKernelMap(),
            getControlledMatrixKernelMap(),
        };
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
    void applyOperation(Pennylane::Gates::KernelType kernel,
                        const std::string &opName,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        auto *arr = BaseType::getData();
        DynamicDispatcher<PrecisionT>::getInstance().applyOperation(
            kernel, arr, BaseType::getNumQubits(), opName, wires, inverse,
            params);
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
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        auto *arr = BaseType::getData();
        auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gate_op = dispatcher.strToGateOp(opName);
        dispatcher.applyOperation(getKernelForGate(gate_op), arr,
                                  BaseType::getNumQubits(), gate_op, wires,
                                  inverse, params);
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<std::size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        PL_ABORT_IF_NOT(
            areVecsDisjoint<std::size_t>(controlled_wires, wires),
            "`controlled_wires` and `target wires` must be disjoint.");

        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        auto *arr = BaseType::getData();
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gate_op = dispatcher.strToControlledGateOp(opName);
        const auto kernel = getKernelForControlledGate(gate_op);
        dispatcher.applyControlledGate(
            kernel, arr, BaseType::getNumQubits(), opName, controlled_wires,
            controlled_values, wires, inverse, params);
    }
    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     * @param matrix Matrix data (in row-major format).
     */
    template <typename Alloc>
    void applyOperation(const std::string &opName,
                        const std::vector<std::size_t> &wires, bool inverse,
                        const std::vector<PrecisionT> &params,
                        const std::vector<ComplexT, Alloc> &matrix) {
        auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        if (dispatcher.hasGateOp(opName)) {
            applyOperation(opName, wires, inverse, params);
        } else {
            applyMatrix(matrix, wires, inverse);
        }
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     * @param matrix Matrix data (in row-major format).
     */
    template <typename Alloc>
    void applyOperation(const std::string &opName,
                        const std::vector<std::size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<std::size_t> &wires, bool inverse,
                        const std::vector<PrecisionT> &params,
                        const std::vector<ComplexT, Alloc> &matrix) {
        PL_ABORT_IF_NOT(
            areVecsDisjoint<std::size_t>(controlled_wires, wires),
            "`controlled_wires` and `target wires` must be disjoint.");

        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        if (!controlled_wires.empty()) {
            applyOperation(opName, controlled_wires, controlled_values, wires,
                           inverse, params);
            return;
        }
        auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        if (dispatcher.hasGateOp(opName)) {
            applyOperation(opName, wires, inverse, params);
        } else {
            applyMatrix(matrix, wires, inverse);
        }
    }

    /**
     * @brief Apply a PauliRot gate to the state-vector.
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Rotation angle.
     * @param word A Pauli word (e.g. "XYYX").
     */
    void applyPauliRot(const std::vector<std::size_t> &wires,
                       const bool inverse,
                       const std::vector<PrecisionT> &params,
                       const std::string &word) {
        PL_ABORT_IF_NOT(wires.size() == word.size(),
                        "wires and word have incompatible dimensions.");
        GateImplementationsLM::applyPauliRot<PrecisionT>(
            BaseType::getData(), BaseType::getNumQubits(), wires, inverse,
            params[0], word);
    }

    /**
     * @brief Apply a single generator to the state-vector using a given
     * kernel.
     *
     * @param kernel Kernel to run the operation.
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] inline auto applyGenerator(
        Pennylane::Gates::KernelType kernel, const std::string &opName,
        const std::vector<std::size_t> &wires, bool adj = false) -> PrecisionT {
        auto *arr = BaseType::getData();
        return DynamicDispatcher<PrecisionT>::getInstance().applyGenerator(
            kernel, arr, BaseType::getNumQubits(), opName, wires, adj);
    }

    /**
     * @brief Apply a single generator to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires the gate applies to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] auto applyGenerator(const std::string &opName,
                                      const std::vector<std::size_t> &wires,
                                      bool adj = false) -> PrecisionT {
        auto *arr = BaseType::getData();
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gen_op = dispatcher.strToGeneratorOp(opName);
        return dispatcher.applyGenerator(getKernelForGenerator(gen_op), arr,
                                         BaseType::getNumQubits(), opName,
                                         wires, adj);
    }

    /**
     * @brief Apply a single generator to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] auto
    applyGenerator(const std::string &opName,
                   const std::vector<std::size_t> &controlled_wires,
                   const std::vector<bool> &controlled_values,
                   const std::vector<std::size_t> &wires, bool adj = false)
        -> PrecisionT {
        auto *arr = BaseType::getData();
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto generator_op = dispatcher.strToControlledGeneratorOp(opName);
        const auto kernel = getKernelForControlledGenerator(generator_op);
        return dispatcher.applyControlledGenerator(
            kernel, arr, BaseType::getNumQubits(), opName, controlled_wires,
            controlled_values, wires, adj);
    }

    /**
     * @brief Apply a given controlled-matrix directly to the statevector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void
    applyControlledMatrix(const ComplexT *matrix,
                          const std::vector<std::size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<std::size_t> &wires,
                          bool inverse = false) {
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        auto *arr = BaseType::getData();
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        const auto kernel = [n_wires = wires.size(), this]() {
            switch (n_wires) {
            case 1:
                return getKernelForControlledMatrix(
                    ControlledMatrixOperation::NCSingleQubitOp);
            case 2:
                return getKernelForControlledMatrix(
                    ControlledMatrixOperation::NCTwoQubitOp);
            default:
                return getKernelForControlledMatrix(
                    ControlledMatrixOperation::NCMultiQubitOp);
            }
        }();
        dispatcher.applyControlledMatrix(kernel, arr, BaseType::getNumQubits(),
                                         matrix, controlled_wires,
                                         controlled_values, wires, inverse);
    }

    /**
     * @brief Apply a given controlled-matrix directly to the statevector.
     *
     * @param matrix Vector containing the statevector data (in row-major
     * format).
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void
    applyControlledMatrix(const std::vector<ComplexT> matrix,
                          const std::vector<std::size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<std::size_t> &wires,
                          bool inverse = false) {
        PL_ABORT_IF_NOT(
            areVecsDisjoint<std::size_t>(controlled_wires, wires),
            "`controlled_wires` and `target wires` must be disjoint.");
        applyControlledMatrix(matrix.data(), controlled_wires,
                              controlled_values, wires, inverse);
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
    inline void applyMatrix(Pennylane::Gates::KernelType kernel,
                            const ComplexT *matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false) {
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        auto *arr = BaseType::getData();

        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");

        dispatcher.applyMatrix(kernel, arr, BaseType::getNumQubits(), matrix,
                               wires, inverse);
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
    inline void applyMatrix(Pennylane::Gates::KernelType kernel,
                            const std::vector<ComplexT> &matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
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
    inline void applyMatrix(const ComplexT *matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false) {
        using Pennylane::Gates::MatrixOperation;

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
    inline void applyMatrix(const std::vector<ComplexT, Alloc> &matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");

        applyMatrix(matrix.data(), wires, inverse);
    }

    /**
     * @brief Collapse the state vector as after having measured one of the
     * qubits.
     *
     * The branch parameter imposes the measurement result on the given wire.
     *
     * @param wire Wire to collapse.
     * @param branch Branch 0 or 1.
     */
    void collapse(const std::size_t wire, const bool branch) {
        auto *arr = BaseType::getData();
        const std::size_t stride =
            pow(2, BaseType::getNumQubits() - (1 + wire));
        const std::size_t vec_size = pow(2, BaseType::getNumQubits());
        const auto section_size = vec_size / stride;
        const auto half_section_size = section_size / 2;

        // zero half the entries
        // the "half" entries depend on the stride
        // *_*_*_*_ for stride 1
        // **__**__ for stride 2
        // ****____ for stride 4
        const std::size_t k = branch ? 0 : 1;
        for (std::size_t idx = 0; idx < half_section_size; idx++) {
            const std::size_t offset = stride * (k + 2 * idx);
            for (std::size_t ids = 0; ids < stride; ids++) {
                arr[offset + ids] = {0., 0.};
            }
        }

        normalize();
    }

    /**
     * @brief Normalize vector (to have norm 1).
     */
    void normalize() {
        auto *arr = BaseType::getData();
        PrecisionT norm = std::sqrt(squaredNorm(arr, BaseType::getLength()));

        PL_ABORT_IF(norm < std::numeric_limits<PrecisionT>::epsilon() * 1e2,
                    "vector has norm close to zero and can't be normalized");

        ComplexT inv_norm = 1. / norm;
        for (std::size_t k = 0; k < BaseType::getLength(); k++) {
            arr[k] *= inv_norm;
        }
    }

    /**
     * @brief Prepares a single computational basis state.
     *
     * @param index Index of the target element.
     */
    void setBasisState(const std::size_t index) {
        auto length = BaseType::getLength();
        PL_ABORT_IF(index > length - 1, "Invalid index");

        auto *arr = BaseType::getData();
        std::fill(arr, arr + length, 0.0);
        arr[index] = {1.0, 0.0};
    }

    /**
     * @brief Prepares a single computational basis state.
     *
     * @param state Binary number representing the index
     * @param wires Wires.
     */
    void setBasisState(const std::vector<std::size_t> &state,
                       const std::vector<std::size_t> &wires) {
        const auto n_wires = wires.size();
        const auto num_qubits = BaseType::getNumQubits();
        std::size_t index{0U};
        for (std::size_t k = 0; k < n_wires; k++) {
            const auto bit = static_cast<std::size_t>(state[k]);
            index |= bit << (num_qubits - 1 - wires[k]);
        }
        setBasisState(index);
    }

    /**
     * @brief Reset the data back to the \f$\ket{0}\f$ state.
     *
     */
    void resetStateVector() {
        if (BaseType::getLength() > 0) {
            setBasisState(0U);
        }
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param values Values to be set for the target elements.
     * @param indices Indices of the target elements.
     */
    void setStateVector(const std::vector<std::size_t> &indices,
                        const std::vector<ComplexT> &values) {
        const auto num_indices = indices.size();
        PL_ABORT_IF(num_indices != values.size(),
                    "Indices and values length must match");

        auto *arr = BaseType::getData();
        const auto length = BaseType::getLength();
        std::fill(arr, arr + length, 0.0);
        for (std::size_t i = 0; i < num_indices; i++) {
            PL_ABORT_IF(i >= length, "Invalid index");
            arr[indices[i]] = values[i];
        }
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param state State.
     * @param wires Wires.
     */
    void setStateVector(const std::vector<ComplexT> &state,
                        const std::vector<std::size_t> &wires) {
        PL_ABORT_IF_NOT(state.size() == exp2(wires.size()),
                        "Inconsistent state and wires dimensions.")
        setStateVector(state.data(), wires);
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param state State.
     * @param wires Wires.
     */
    void setStateVector(const ComplexT *state,
                        const std::vector<std::size_t> &wires) {
        const std::size_t num_state = exp2(wires.size());
        const auto total_wire_count = BaseType::getNumQubits();

        std::vector<std::size_t> reversed_sorted_wires(wires);
        std::sort(reversed_sorted_wires.begin(), reversed_sorted_wires.end());
        std::reverse(reversed_sorted_wires.begin(),
                     reversed_sorted_wires.end());
        std::vector<std::size_t> controlled_wires(total_wire_count);
        std::iota(std::begin(controlled_wires), std::end(controlled_wires), 0);
        for (auto wire : reversed_sorted_wires) {
            // Reverse guarantees that we start erasing at the end of the array.
            // Maybe this can be optimized.
            controlled_wires.erase(controlled_wires.begin() + wire);
        }

        const std::vector<bool> controlled_values(controlled_wires.size(),
                                                  false);
        auto core_function = [num_state,
                              &state](ComplexT *arr,
                                      const std::vector<std::size_t> &indices,
                                      const std::size_t offset) {
            for (std::size_t i = 0; i < num_state; i++) {
                const std::size_t index = indices[i] + offset;
                arr[index] = state[i];
            }
        };
        GateImplementationsLM::applyNCN(BaseType::getData(), total_wire_count,
                                        controlled_wires, controlled_values,
                                        wires, core_function);
    }
};
} // namespace Pennylane::LightningQubit
