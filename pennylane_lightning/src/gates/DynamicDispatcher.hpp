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
 * @file DynamicDispatcher.hpp
 * Defines DynamicDispatcher class. Can be used to call a gate operation by
 * string.
 */

#pragma once

#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "GateUtil.hpp"
#include "KernelType.hpp"
#include "Macros.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "Util.hpp"

#include <cassert>
#include <complex>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

/// @cond DEV
namespace Pennylane::Internal {
constexpr auto generatorNamesWithoutPrefix() {
    constexpr std::string_view prefix = "Generator";
    namespace GateConstant = Gates::Constant;
    std::array<std::pair<Gates::GeneratorOperation, std::string_view>,
               GateConstant::generator_names.size()>
        res;
    for (size_t i = 0; i < GateConstant::generator_names.size(); i++) {
        const auto [gntr_op, gntr_name] = GateConstant::generator_names[i];
        res[i].first = gntr_op;
        res[i].second = gntr_name.substr(prefix.size());
    }
    return res;
}

} // namespace Pennylane::Internal
/// @endcond

namespace Pennylane {

/**
 * @brief DynamicDispatcher class
 *
 * This is a singleton class that can call a gate/generator operation
 * dynamically. Currently, all gate operations (gates/generators/matrices) are
 * registered to this class when the library is loaded. As all functions besides
 * registration functions are already thread-safe, we can use this class
 * in multithreading environment without any problem.
 * In addition, adding mutex is not required unless kernel functions are
 * registered in multiple threads.
 */
template <typename PrecisionT> class DynamicDispatcher {
  public:
    using CFP_t = std::complex<PrecisionT>;

    using GateFunc = std::function<void(
        std::complex<PrecisionT> * /*data*/, size_t /*num_qubits*/,
        const std::vector<size_t> & /*wires*/, bool /*inverse*/,
        const std::vector<PrecisionT> & /*params*/)>;

    using GeneratorFunc = Gates::GeneratorFuncPtrT<PrecisionT>;
    using MatrixFunc = Gates::MatrixFuncPtrT<PrecisionT>;

  private:
    std::unordered_map<std::string, Gates::GateOperation> str_to_gates_;
    std::unordered_map<std::string, Gates::GeneratorOperation> str_to_gntrs_;

    std::unordered_map<std::pair<Gates::GateOperation, Gates::KernelType>,
                       GateFunc, Util::PairHash>
        gate_kernels_;

    std::unordered_map<std::pair<Gates::GeneratorOperation, Gates::KernelType>,
                       GeneratorFunc, Util::PairHash>
        generator_kernels_;

    std::unordered_map<std::pair<Gates::MatrixOperation, Gates::KernelType>,
                       MatrixFunc, Util::PairHash>
        matrix_kernels_;

    std::unordered_map<Gates::KernelType, std::string> kernel_names_;

    DynamicDispatcher() {
        using Gates::KernelType;
        constexpr static auto gntr_names_without_prefix =
            Internal::generatorNamesWithoutPrefix();

        for (const auto &[gate_op, gate_name] : Gates::Constant::gate_names) {
            str_to_gates_.emplace(gate_name, gate_op);
        }
        for (const auto &[gntr_op, gntr_name] : gntr_names_without_prefix) {
            str_to_gntrs_.emplace(gntr_name, gntr_op);
        }
    }

  public:
    DynamicDispatcher(const DynamicDispatcher &) = delete;
    DynamicDispatcher(DynamicDispatcher &&) = delete;
    DynamicDispatcher &operator=(const DynamicDispatcher &) = delete;
    DynamicDispatcher &operator=(DynamicDispatcher &&) = delete;
    ~DynamicDispatcher() = default;

    /**
     * @brief Get the singleton instance
     */
    static DynamicDispatcher &getInstance() {
        static DynamicDispatcher singleton;
        return singleton;
    }

    /**
     * @brief Get all registered kernels
     */
    [[nodiscard]] auto registeredKernels() const
        -> std::vector<Gates::KernelType> {
        std::vector<Gates::KernelType> kernels;

        for (const auto &[kernel, name] : kernel_names_) {
            kernels.emplace_back(kernel);
        }
        return kernels;
    }

    /**
     * @brief Check whether the kernel is registered to a dispatcher
     */
    [[nodiscard]] auto isRegisteredKernel(Gates::KernelType kernel) const {
        return kernel_names_.contains(kernel);
    }

    /**
     * @brief Register kernel name
     *
     * @param kernel Kernel
     * @param name Name of the kernel
     */
    void registerKernelName(Gates::KernelType kernel, std::string name) {
        kernel_names_.emplace(kernel, std::move(name));
    }

    /**
     * @brief Get registered name of the kernel
     *
     * @param kernel Kernel
     */
    [[nodiscard]] auto getKernelName(Gates::KernelType kernel) const
        -> std::string {
        return kernel_names_.at(kernel);
    }

    /**
     * @brief Get registered gates for the given kernel
     *
     * @param kernel Kernel
     */
    [[nodiscard]] auto registeredGatesForKernel(Gates::KernelType kernel) const
        -> std::unordered_set<Gates::GateOperation> {
        std::unordered_set<Gates::GateOperation> gates;

        for (const auto &[key, val] : gate_kernels_) {
            if (key.second == kernel) {
                gates.emplace(key.first);
            }
        }
        return gates;
    }

    [[nodiscard]] auto
    registeredGeneratorsForKernel(Gates::KernelType kernel) const
        -> std::unordered_set<Gates::GeneratorOperation> {
        std::unordered_set<Gates::GeneratorOperation> gntrs;

        for (const auto &[key, val] : generator_kernels_) {
            if (key.second == kernel) {
                gntrs.emplace(key.first);
            }
        }
        return gntrs;
    }

    [[nodiscard]] auto
    registeredMatricesForKernel(Gates::KernelType kernel) const
        -> std::unordered_set<Gates::MatrixOperation> {
        std::unordered_set<Gates::MatrixOperation> matrices;

        for (const auto &[key, val] : matrix_kernels_) {
            if (key.second == kernel) {
                matrices.emplace(key.first);
            }
        }
        return matrices;
    }

    /**
     * @brief Gate name to gate operation
     *
     * @param gate_name Gate name
     */
    [[nodiscard]] auto strToGateOp(const std::string &gate_name) const
        -> Gates::GateOperation {
        return str_to_gates_.at(gate_name);
    }

    /**
     * @brief Generator name to generator operation
     *
     * @param gntr_name Generator name without "Generator" prefix
     */
    [[nodiscard]] auto strToGeneratorOp(const std::string &gntr_name) const
        -> Gates::GeneratorOperation {
        return str_to_gntrs_.at(gntr_name);
    }

    /**
     * @brief Register a new gate operation for the operation. Can pass a custom
     * kernel
     */
    template <typename FunctionType>
    void registerGateOperation(Gates::GateOperation gate_op,
                               Gates::KernelType kernel, FunctionType &&func) {
        gate_kernels_.emplace(std::make_pair(gate_op, kernel),
                              std::forward<FunctionType>(func));
    }

    /**
     * @brief Register a new gate generator for the operation. Can pass a custom
     * kernel
     */
    template <typename FunctionType>
    void registerGeneratorOperation(Gates::GeneratorOperation gntr_op,
                                    Gates::KernelType kernel,
                                    FunctionType &&func) {
        generator_kernels_.emplace(std::make_pair(gntr_op, kernel),
                                   std::forward<FunctionType>(func));
    }

    /**
     * @brief Register a new matrix operation. Can pass a custom
     * kernel
     */
    void registerMatrixOperation(Gates::MatrixOperation mat_op,
                                 Gates::KernelType kernel, MatrixFunc func) {
        matrix_kernels_.emplace(std::make_pair(mat_op, kernel), func);
    }

    /**
     * @brief Check if a kernel function is registered for the given
     * gate operation and kernel.
     *
     * @param gate_op Gate operation
     * @param kernel Kernel
     */
    bool isRegistered(Gates::GateOperation gate_op,
                      Gates::KernelType kernel) const {
        return gate_kernels_.find(std::make_pair(gate_op, kernel)) !=
               gate_kernels_.cend();
    }

    /**
     * @brief Check if a kernel function is registered for the given
     * generator operation and kernel.
     *
     * @param gntr_op Generator operation
     * @param kernel Kernel
     */
    bool isRegistered(Gates::GeneratorOperation gntr_op,
                      Gates::KernelType kernel) const {
        return generator_kernels_.find(std::make_pair(gntr_op, kernel)) !=
               generator_kernels_.cend();
    }

    /**
     * @brief Check if a kernel function is registered for the given
     * matrix operation and kernel.
     *
     * @param mat_op Matrix operation
     * @param kernel Kernel
     */
    bool isRegistered(Gates::MatrixOperation mat_op,
                      Gates::KernelType kernel) const {
        return matrix_kernels_.find(std::make_pair(mat_op, kernel)) !=
               matrix_kernels_.cend();
    }

    /**
     * @brief Apply a single gate to the state-vector using the given kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param op_name Gate operation name.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(Gates::KernelType kernel, CFP_t *data,
                        size_t num_qubits, const std::string &op_name,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<PrecisionT> &params = {}) const {
        const auto iter =
            gate_kernels_.find(std::make_pair(strToGateOp(op_name), kernel));
        if (iter == gate_kernels_.cend()) {
            throw std::invalid_argument(
                "Cannot find a registered kernel for a given gate "
                "and kernel pair");
        }
        (iter->second)(data, num_qubits, wires, inverse, params);
    }

    /**
     * @brief Apply a single gate to the state-vector using the given kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param gate_op Gate operation.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(Gates::KernelType kernel, CFP_t *data,
                        size_t num_qubits, Gates::GateOperation gate_op,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<PrecisionT> &params = {}) const {
        const auto iter = gate_kernels_.find(std::make_pair(gate_op, kernel));
        if (iter == gate_kernels_.cend()) {
            throw std::invalid_argument(
                "Cannot find a registered kernel for a given gate "
                "and kernel pair");
        }
        (iter->second)(data, num_qubits, wires, inverse, params);
    }

    /**
     * @brief Apply multiple gates to the state-vector using a registered kernel
     *
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param ops List of Gate operation names.
     * @param wires List of wires to apply each gate to.
     * @param inverse List of inverses
     * @param params List of parameters
     */
    void
    applyOperations(Gates::KernelType kernel, CFP_t *data, size_t num_qubits,
                    const std::vector<std::string> &ops,
                    const std::vector<std::vector<size_t>> &wires,
                    const std::vector<bool> &inverse,
                    const std::vector<std::vector<PrecisionT>> &params) const {
        const size_t numOperations = ops.size();
        if (numOperations != wires.size() || numOperations != params.size()) {
            throw std::invalid_argument(
                "Invalid arguments: number of operations, wires, and "
                "parameters must all be equal");
        }

        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(kernel, data, num_qubits, ops[i], wires[i],
                           inverse[i], params[i]);
        }
    }

    /**
     * @brief Apply multiple (non-parameterized) gates to the state-vector
     * using a registered kernel
     *
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param ops List of Gate operation names.
     * @param wires List of wires to apply each gate to.
     * @param inverse List of inverses
     */
    void applyOperations(Gates::KernelType kernel, CFP_t *data,
                         size_t num_qubits, const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &wires,
                         const std::vector<bool> &inverse) const {
        const size_t numOperations = ops.size();
        if (numOperations != wires.size()) {
            throw std::invalid_argument(
                "Invalid arguments: number of operations, wires, and "
                "parameters must all be equal");
        }

        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(kernel, data, num_qubits, ops[i], wires[i],
                           inverse[i], {});
        }
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param kernel Kernel to use for this operation
     * @param data Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    void applyMatrix(Gates::KernelType kernel, CFP_t *data, size_t num_qubits,
                     const std::complex<PrecisionT> *matrix,
                     const std::vector<size_t> &wires, bool inverse) const {
        using Gates::MatrixOperation;
        assert(num_qubits >= wires.size());

        const auto mat_op = [n_wires = wires.size()]() {
            switch (n_wires) {
            case 1:
                return MatrixOperation::SingleQubitOp;
            case 2:
                return MatrixOperation::TwoQubitOp;
            default:
                return MatrixOperation::MultiQubitOp;
            }
        }();

        const auto iter = matrix_kernels_.find(std::make_pair(mat_op, kernel));

        if (iter == matrix_kernels_.end()) {
            throw std::invalid_argument(
                std::string(
                    Util::lookup(Gates::Constant::matrix_names, mat_op)) +
                " is not registered for the given kernel");
        }
        (iter->second)(data, num_qubits, matrix, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param kernel Kernel to use for this operation
     * @param data Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    void applyMatrix(Gates::KernelType kernel, CFP_t *data, size_t num_qubits,
                     const std::vector<std::complex<PrecisionT>> &matrix,
                     const std::vector<size_t> &wires, bool inverse) const {
        if (matrix.size() != Util::exp2(2 * wires.size())) {
            throw std::invalid_argument(
                "The size of matrix does not match with the given "
                "number of wires");
        }
        applyMatrix(kernel, data, num_qubits, matrix.data(), wires, inverse);
    }

    /**
     * @brief Apply a single generator to the state-vector using the given
     * kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param gntr_op Generator operation.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of gate.
     */
    auto applyGenerator(Gates::KernelType kernel, CFP_t *data,
                        size_t num_qubits, Gates::GeneratorOperation gntr_op,
                        const std::vector<size_t> &wires, bool adj) const
        -> PrecisionT {
        using Gates::Constant::generator_names;
        const auto iter =
            generator_kernels_.find(std::make_pair(gntr_op, kernel));
        if (iter == generator_kernels_.cend()) {
            throw std::invalid_argument(
                "Cannot find a registered kernel for a given generator "
                "and kernel pair.");
        }
        return (iter->second)(data, num_qubits, wires, adj);
    }
    /**
     * @brief Apply a single generator to the state-vector using the given
     * kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param op_name Gate operation name.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of gate.
     */
    auto applyGenerator(Gates::KernelType kernel, CFP_t *data,
                        size_t num_qubits, const std::string &op_name,
                        const std::vector<size_t> &wires, bool adj) const
        -> PrecisionT {
        const auto iter = generator_kernels_.find(
            std::make_pair(strToGeneratorOp(op_name), kernel));
        if (iter == generator_kernels_.cend()) {
            throw std::invalid_argument(
                "Cannot find a registered kernel for a given generator "
                "and kernel pair.");
        }
        return (iter->second)(data, num_qubits, wires, adj);
    }
};
} // namespace Pennylane

/// @cond DEV
namespace Pennylane::Internal {
int registerAllAvailableKernels_Float();
int registerAllAvailableKernels_Double();

/**
 * @brief These functions are only used to register kernels to the dynamic
 * dispatcher.
 */
struct RegisterBeforeMain_Float {
    const static inline int dummy = registerAllAvailableKernels_Float();
};

struct RegisterBeforeMain_Double {
    const static inline int dummy = registerAllAvailableKernels_Double();
};
} // namespace Pennylane::Internal
/// @endcond
