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
#pragma once

#include "BitUtil.hpp"
#include "Gates.hpp"
#include "KernelMap.hpp"
#include "KernelType.hpp"
#include "Memory.hpp"
#include "StateVectorBase.hpp"
#include "Threading.hpp"
#include "Util.hpp"

namespace Pennylane {

/**
 * @brief StateVector class where data resides in CPU memory.
 *
 * @tparam PrecisionT Data floating point type
 * @tparam Derived Derived class for CRTP.
 */
template <class PrecisionT, class Derived>
class StateVectorCPU : public StateVectorBase<PrecisionT, Derived> {
  public:
    using ComplexPrecisionT = std::complex<PrecisionT>;

  protected:
    const Threading threading_;
    const CPUMemoryModel memory_model_;

  private:
    using BaseType = StateVectorBase<PrecisionT, Derived>;

    std::unordered_map<Gates::GateOperation, Gates::KernelType>
        kernel_for_gates_;
    std::unordered_map<Gates::GeneratorOperation, Gates::KernelType>
        kernel_for_generators_;
    std::unordered_map<Gates::MatrixOperation, Gates::KernelType>
        kernel_for_matrices_;

    /**
     * @brief Internal function set kernels for all operations depending on
     * provided dispatch options.
     *
     * @param num_qubits Number of qubits of the statevector
     * @param threading Threading option
     * @param memory_model Memory model
     */
    void setKernels(size_t num_qubits, Threading threading,
                    CPUMemoryModel memory_model) {
        using KernelMap::OperationKernelMap;
        kernel_for_gates_ =
            OperationKernelMap<Gates::GateOperation>::getInstance()
                .getKernelMap(num_qubits, threading, memory_model);
        kernel_for_generators_ =
            OperationKernelMap<Gates::GeneratorOperation>::getInstance()
                .getKernelMap(num_qubits, threading, memory_model);
        kernel_for_matrices_ =
            OperationKernelMap<Gates::MatrixOperation>::getInstance()
                .getKernelMap(num_qubits, threading, memory_model);
    }

  protected:
    explicit StateVectorCPU(size_t num_qubits, Threading threading,
                            CPUMemoryModel memory_model)
        : BaseType(num_qubits), threading_{threading}, memory_model_{
                                                           memory_model} {
        setKernels(num_qubits, threading, memory_model);
    }

  public:
    /**
     * @brief Get a kernel for a gate operation.
     *
     * @param gate_op Gate operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForGate(Gates::GateOperation gate_op) const -> Gates::KernelType {
        return kernel_for_gates_.at(gate_op);
    }

    /**
     * @brief Get a kernel for a gate operation.
     *
     * @param gntr_op Generator operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForGenerator(Gates::GeneratorOperation gntr_op) const
        -> Gates::KernelType {
        return kernel_for_generators_.at(gntr_op);
    }

    /**
     * @brief Get a kernel for a gate operation.
     *
     * @param mat_op Matrix operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForMatrix(Gates::MatrixOperation mat_op) const
        -> Gates::KernelType {
        return kernel_for_matrices_.at(mat_op);
    }

    /**
     * @brief Get memory model of the statevector
     */
    [[nodiscard]] inline CPUMemoryModel memoryModel() const {
        return memory_model_;
    }

    /**
     * @brief Get threading of the statevector
     */
    [[nodiscard]] inline Threading threading() const { return threading_; }

    /**
     * @brief Get kernels for all gate operations.
     */
    [[nodiscard]] inline auto getGateKernelMap() const & -> const
        std::unordered_map<Gates::GateOperation, Gates::KernelType> & {
        return kernel_for_gates_;
    }

    [[nodiscard]] inline auto getGateKernelMap()
        && -> std::unordered_map<Gates::GateOperation, Gates::KernelType> {
        return kernel_for_gates_;
    }

    /**
     * @brief Get kernels for all generator operations.
     */
    [[nodiscard]] inline auto getGeneratorKernelMap() const & -> const
        std::unordered_map<Gates::GeneratorOperation, Gates::KernelType> & {
        return kernel_for_generators_;
    }

    [[nodiscard]] inline auto getGeneratorKernelMap()
        && -> std::unordered_map<Gates::GeneratorOperation, Gates::KernelType> {
        return kernel_for_generators_;
    }

    /**
     * @brief Get kernels for all matrix operations.
     */
    [[nodiscard]] inline auto getMatrixKernelMap() const & -> const
        std::unordered_map<Gates::MatrixOperation, Gates::KernelType> & {
        return kernel_for_matrices_;
    }

    [[nodiscard]] inline auto getMatrixKernelMap()
        && -> std::unordered_map<Gates::MatrixOperation, Gates::KernelType> {
        return kernel_for_matrices_;
    }
};
} // namespace Pennylane
