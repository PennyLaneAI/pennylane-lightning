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

#include <bitset>
#include <iostream>

#include "CPUMemoryModel.hpp"
#include "GateOperation.hpp"
#include "KernelMap.hpp"
#include "KernelType.hpp"
#include "StateVectorBase.hpp"
#include "Threading.hpp"

#include "BitUtil.hpp"

/// @cond DEV
namespace {
using Pennylane::LightningQubit::Util::Threading;
using Pennylane::Util::CPUMemoryModel;
using Pennylane::Util::exp2;
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

    GateKernelMap kernel_for_gates_;
    GeneratorKernelMap kernel_for_generators_;
    MatrixKernelMap kernel_for_matrices_;

    /**
     * @brief Internal function to set kernels for all operations depending on
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
            OperationKernelMap<GateOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
        kernel_for_generators_ =
            OperationKernelMap<GeneratorOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
        kernel_for_matrices_ =
            OperationKernelMap<MatrixOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
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
     * @brief Get kernels for all matrix operations.
     */
    [[nodiscard]] inline auto
    getMatrixKernelMap() const & -> const MatrixKernelMap & {
        return kernel_for_matrices_;
    }

    [[nodiscard]] inline auto getMatrixKernelMap() && -> MatrixKernelMap {
        return kernel_for_matrices_;
    }

  protected:
    explicit StateVectorLQubit(size_t num_qubits, Threading threading,
                               CPUMemoryModel memory_model)
        : BaseType(num_qubits), threading_{threading}, memory_model_{
                                                           memory_model} {
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
     *  @brief Returns a tuple containing the gate, generator, and matrix kernel
     * maps respectively.
     */
    [[nodiscard]] auto getSupportedKernels()
        const & -> std::tuple<const GateKernelMap &, const GeneratorKernelMap &,
                              const MatrixKernelMap &> {
        return {getGateKernelMap(), getGeneratorKernelMap(),
                getMatrixKernelMap()};
    }

    [[nodiscard]] auto getSupportedKernels() && -> std::tuple<
        GateKernelMap &&, GeneratorKernelMap &&, MatrixKernelMap &&> {
        return {getGateKernelMap(), getGeneratorKernelMap(),
                getMatrixKernelMap()};
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
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        auto *arr = this->getData();
        DynamicDispatcher<PrecisionT>::getInstance().applyOperation(
            kernel, arr, this->getNumQubits(), opName, wires, inverse, params);
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
        auto *arr = this->getData();
        auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gate_op = dispatcher.strToGateOp(opName);
        dispatcher.applyOperation(getKernelForGate(gate_op), arr,
                                  this->getNumQubits(), gate_op, wires, inverse,
                                  params);
    }

    /**
     * @brief Apply a single generator to the state-vector using a given kernel.
     *
     * @param kernel Kernel to run the operation.
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] inline auto
    applyGenerator(Pennylane::Gates::KernelType kernel,
                   const std::string &opName, const std::vector<size_t> &wires,
                   bool adj = false) -> PrecisionT {
        auto *arr = this->getData();
        return DynamicDispatcher<PrecisionT>::getInstance().applyGenerator(
            kernel, arr, this->getNumQubits(), opName, wires, adj);
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
        auto *arr = this->getData();
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gen_op = dispatcher.strToGeneratorOp(opName);
        return dispatcher.applyGenerator(getKernelForGenerator(gen_op), arr,
                                         this->getNumQubits(), opName, wires,
                                         adj);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a given
     * kernel.
     *
     * @param kernel Kernel to run the operation
     * @param matrix Pointer to the array data (in row-major format).
     * @param controlled_wires Control wires.
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyControlledMatrix(
        const ComplexT *matrix, const std::vector<size_t> &controlled_wires,
        const std::vector<size_t> &wires, bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        ComplexT *arr = this->getData();

        // const auto kernel = getKernelForMatrix(MatrixOperation::NQubitOp);
        // const auto &dispatcher =
        // DynamicDispatcher<PrecisionT>::getInstance();
        // dispatcher.applyControlledMatrix(arr, this->getNumQubits(), matrix,
        //                                  controlled_wires, wires, inverse);

        using size_t = std::size_t;
        constexpr std::size_t one{1};
        constexpr std::size_t zero{0};
        const size_t n_contr = controlled_wires.size();
        const size_t n_wires = wires.size();
        const std::size_t nw_tot = n_contr + n_wires;
        const std::size_t num_qubits = this->getNumQubits();
        PL_ASSERT(num_qubits >= nw_tot);

        std::vector<std::size_t> all_wires = controlled_wires;
        all_wires.insert(all_wires.begin(), wires.begin(), wires.end());

        std::vector<std::size_t> rev_wires(nw_tot);
        std::vector<std::size_t> rev_wire_shifts(nw_tot);
        for (std::size_t k = 0; k < nw_tot; k++) {
            rev_wires[k] = (num_qubits - 1) - all_wires[(nw_tot - 1) - k];
            rev_wire_shifts[k] = (one << rev_wires[k]);
        }
        const std::vector<std::size_t> parity =
            Pennylane::Util::revWireParity(rev_wires);
        PL_ASSERT(nw_tot == parity.size() - 1);

        printf("\n=================\n");
        printf("num_qubits = %ld\n", num_qubits);
        printf("nw_tot = %ld\n", nw_tot);
        printf("controlled_wires.size() = %ld\n", n_contr);
        printf("wires.size() = %ld\n", n_wires);
        {
            int count = 0;
            for (const auto &i : all_wires) {
                std::cout << "all_wires(" << count++ << ") = " << i << " = "
                          << std::bitset<8>(i) << std::endl;
            }
        }

        {
            int count = 0;
            for (const auto &i : rev_wires) {
                std::cout << "rev_wires(" << count++ << ") = " << i << " = "
                          << std::bitset<8>(i) << std::endl;
            }
        }

        {
            int count = 0;
            for (const auto &i : rev_wire_shifts) {
                std::cout << "rev_wire_shifts(" << count++ << ") = " << i
                          << " = " << std::bitset<8>(i) << std::endl;
            }
        }

        const size_t step = one << nw_tot;
        const size_t dim = one << n_wires;
        std::vector<size_t> indices(dim);
        std::vector<std::complex<PrecisionT>> coeffs_in(dim, 0.0);

        for (std::size_t k = 0; k < exp2(num_qubits - nw_tot); k++) {
            std::size_t idx = (k & parity[0]);
            for (std::size_t i = 1; i < parity.size(); i++) {
                idx |= ((k << i) & parity[i]);
            }
            std::cout << "index(base) = " << std::bitset<8>(idx) << std::endl;
            for (std::size_t i = 0; i < n_contr; i++) {
                idx |= rev_wire_shifts[i];
                std::cout << "index(" << i << ") = " << std::bitset<8>(idx)
                          << std::endl;
            }
            indices[0] = idx;
            coeffs_in[0] = arr[idx];
            for (std::size_t inner_idx = 1; inner_idx < dim; inner_idx++) {
                idx = indices[0];
                for (std::size_t i = 0; i < n_wires; i++) {
                    if ((inner_idx & (one << i)) != 0) {
                        idx |= rev_wire_shifts[i + n_contr];
                    }
                }
                indices[inner_idx] = idx;
                coeffs_in[inner_idx] = arr[idx];
            }
            {
                int count = 0;
                for (const auto &i : indices) {
                    std::cout << "index(" << count++
                              << ") = " << std::bitset<8>(i) << std::endl;
                }
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
        printf("=================\n");
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
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        auto *arr = this->getData();

        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");

        dispatcher.applyMatrix(kernel, arr, this->getNumQubits(), matrix, wires,
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
    inline void applyMatrix(Pennylane::Gates::KernelType kernel,
                            const std::vector<ComplexT> &matrix,
                            const std::vector<size_t> &wires,
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
                            const std::vector<size_t> &wires,
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
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");

        applyMatrix(matrix.data(), wires, inverse);
    }
};
} // namespace Pennylane::LightningQubit