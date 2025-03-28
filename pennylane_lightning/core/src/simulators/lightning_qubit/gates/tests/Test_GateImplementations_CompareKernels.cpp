// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <complex>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "ConstantUtil.hpp" // lookup, array_has_elem
#include "DynamicDispatcher.hpp"
#include "KernelMap.hpp"
#include "KernelType.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "TestHelpers.hpp"       // PrecisionToName
#include "TestHelpersSparse.hpp" // SparseMatrixCSR
#include "TestHelpersWires.hpp"
#include "TestKernels.hpp"
#include "Util.hpp" // for_each_enum

/**
 * @file Test_GateImplementations_Nonparam.cpp
 *
 * This file tests all gate operations (besides matrix) by comparing results
 * between different kernels (gate implementations).
 */
/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Util;
using namespace Pennylane::LightningQubit::Gates;
using Pennylane::Util::randomUnitary;
} // namespace
/// @endcond

template <typename PrecisionT>
auto kernelsImplementingGate(GateOperation gate_op) -> std::vector<KernelType> {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    auto kernels = dispatcher.registeredKernels();

    std::vector<KernelType> res;

    std::copy_if(
        kernels.begin(), kernels.end(), std::back_inserter(res),
        [&dispatcher, gate_op](KernelType kernel) {
            return dispatcher.registeredGatesForKernel(kernel).contains(
                gate_op);
        });
    return res;
}

template <typename PrecisionT>
auto kernelsImplementingMatrix(MatrixOperation mat_op)
    -> std::vector<KernelType> {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    auto kernels = dispatcher.registeredKernels();

    std::vector<KernelType> res;

    std::copy_if(
        kernels.begin(), kernels.end(), std::back_inserter(res),
        [&dispatcher, mat_op](KernelType kernel) {
            return dispatcher.registeredMatricesForKernel(kernel).contains(
                mat_op);
        });
    return res;
}

template <typename PrecisionT>
auto kernelsImplementingSparseMatrix(SparseMatrixOperation mat_op)
    -> std::vector<KernelType> {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    auto kernels = dispatcher.registeredKernels();

    std::vector<KernelType> res;

    std::copy_if(kernels.begin(), kernels.end(), std::back_inserter(res),
                 [&dispatcher, mat_op](KernelType kernel) {
                     return dispatcher.registeredSparseMatricesForKernel(kernel)
                         .contains(mat_op);
                 });
    return res;
}

template <typename PrecisionT>
auto kernelsImplementingControlledSparseMatrix(
    ControlledSparseMatrixOperation mat_op) -> std::vector<KernelType> {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    auto kernels = dispatcher.registeredKernels();

    std::vector<KernelType> res;

    std::copy_if(kernels.begin(), kernels.end(), std::back_inserter(res),
                 [&dispatcher, mat_op](KernelType kernel) {
                     return dispatcher
                         .registeredControlledSparseMatricesForKernel(kernel)
                         .contains(mat_op);
                 });
    return res;
}

/**
 * @brief Apply the given gate using all implementing kernels and compare
 * the results.
 */
template <typename PrecisionT, class RandomEngine>
void testApplyGate(RandomEngine &re, GateOperation gate_op,
                   std::size_t num_qubits) {
    using Pennylane::Gates::Constant::gate_names;

    const auto implementing_kernels =
        kernelsImplementingGate<PrecisionT>(gate_op);
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    std::ostringstream ss;
    ss << "Kernels implementing " << lookup(gate_names, gate_op) << " are ";
    for (KernelType kernel : implementing_kernels) {
        ss << dispatcher.getKernelName(kernel) << ", ";
    }

    INFO(ss.str() << "PrecisionT = " << PrecisionToName<PrecisionT>::value);

    const auto ini = createRandomStateVectorData<PrecisionT>(re, num_qubits);

    const auto params = createParams<PrecisionT>(gate_op);
    const auto all_wires = createAllWires(num_qubits, gate_op, true);

    const bool inverse = GENERATE(true, false);
    for (const auto &wires : all_wires) {
        std::ostringstream ss;
        ss << wires;
        auto wires_str = ss.str();
        INFO(wires_str);

        std::vector<TestVector<std::complex<PrecisionT>>> res;

        // Collect results from all implementing kernels
        for (auto kernel : implementing_kernels) {
            auto st = ini;
            dispatcher.applyOperation(kernel, st.data(), num_qubits, gate_op,
                                      wires, inverse, params);
            res.emplace_back(std::move(st));
        }

        // And compare them
        for (std::size_t i = 0; i < res.size() - 1; i++) {
            REQUIRE(res[i] ==
                    approx(res[i + 1]).margin(static_cast<PrecisionT>(1e-5)));
        }
    }
}

TEMPLATE_TEST_CASE("Test all kernels give the same results for gates",
                   "[GateImplementations_CompareKernels]", float, double) {
    /* We test all gate operations up to the number of qubits we give */
    constexpr std::size_t max_num_qubits = 5;
    std::mt19937 re{1337};
    for_each_enum<GateOperation>([&](GateOperation gate_op) {
        const std::size_t min_num_qubits = [=] {
            if (array_has_elem(Pennylane::Gates::Constant::multi_qubit_gates,
                               gate_op)) {
                return std::size_t{1};
            }
            return lookup(Pennylane::Gates::Constant::gate_wires, gate_op);
        }();
        for (std::size_t num_qubits = min_num_qubits;
             num_qubits <= max_num_qubits; num_qubits++) {
            testApplyGate<TestType>(re, gate_op, num_qubits);
        }
    });
}

template <typename PrecisionT, class RandomEngine>
void testMatrixOp(RandomEngine &re, std::size_t num_qubits,
                  std::size_t num_wires, bool inverse) {
    using Pennylane::Gates::Constant::matrix_names;
    PL_ASSERT(num_wires > 0);

    const auto mat_op = [num_wires]() -> MatrixOperation {
        switch (num_wires) {
        case 1:
            return MatrixOperation::SingleQubitOp;
        case 2:
            return MatrixOperation::TwoQubitOp;
        default:
            return MatrixOperation::MultiQubitOp;
        }
    }();

    const auto implementing_kernels =
        kernelsImplementingMatrix<PrecisionT>(mat_op);
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    const auto op_name = lookup(matrix_names, mat_op);

    std::ostringstream ss;
    ss << "Test " << op_name << " with kernels: ";
    for (KernelType kernel : implementing_kernels) {
        ss << dispatcher.getKernelName(kernel) << ", ";
    }
    ss << "inverse: " << inverse;
    ss << ", num_wires: " << num_wires;
    ss << ", num_qubits: " << num_qubits;

    const auto ini_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);
    const auto matrix = randomUnitary<PrecisionT>(re, num_wires);

    DYNAMIC_SECTION(ss.str()) {
        const auto all_wires =
            CombinationGenerator(num_qubits, num_wires).all_perms();
        for (const auto &wires : all_wires) {
            std::vector<TestVector<std::complex<PrecisionT>>> res;

            // Record result from each kernel
            for (KernelType kernel : implementing_kernels) {
                auto st = ini_st;
                dispatcher.applyMatrix(kernel, st.data(), num_qubits,
                                       matrix.data(), wires, inverse);
                res.emplace_back(std::move(st));
            }

            // And compare them
            for (std::size_t idx = 0; idx < implementing_kernels.size() - 1;
                 idx++) {
                REQUIRE(res[idx] == approx(res[idx + 1]).margin(1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Test all kernels give the same results for matrices",
                   "[Test_GateImplementations_CompareKernels]", float, double) {
    std::mt19937 re{1337};

    const std::size_t num_qubits = 5;

    for (bool inverse : {true, false}) {
        for (std::size_t num_wires = 1; num_wires <= 5; num_wires++) {
            testMatrixOp<TestType>(re, num_qubits, num_wires, inverse);
        }
    }
}

/**
 * @brief Apply the given sparse matrix using all implementing kernels and
 * compare results with dense methods.
 * @tparam PrecisionT Precision of the complex data type.
 * @param num_qubits Number of qubits.
 * @param unit_num_wires Number of wires the matrix applies to.
 * @param sparsity Sparsity of the matrix.
 * @param inverse If inverse is required.
 */
template <typename PrecisionT>
void testSparseMatrixOp(std::size_t num_qubits, std::size_t unit_num_wires,
                        PrecisionT sparsity, bool inverse = false) {
    PL_ASSERT(unit_num_wires > 0);

    using ComplexT = std::complex<PrecisionT>;
    std::mt19937 re{1337};

    using Pennylane::Gates::Constant::matrix_names;
    const auto mat_op = [unit_num_wires]() -> MatrixOperation {
        switch (unit_num_wires) {
        case 1:
            return MatrixOperation::SingleQubitOp;
        case 2:
            return MatrixOperation::TwoQubitOp;
        default:
            return MatrixOperation::MultiQubitOp;
        }
    }();
    const auto op_name = lookup(matrix_names, mat_op);

    using Pennylane::Gates::Constant::sparse_matrix_names;
    const auto sparse_mat_op = [unit_num_wires]() -> SparseMatrixOperation {
        // for future expansion:
        [[maybe_unused]] auto unused_unit_num_wires = unit_num_wires;
        return SparseMatrixOperation::SparseMultiQubitOp;
    }();
    const auto sparse_op_name = lookup(sparse_matrix_names, sparse_mat_op);

    const auto implementing_sparse_kernels =
        kernelsImplementingSparseMatrix<PrecisionT>(sparse_mat_op);
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    std::ostringstream ss;
    ss << "Test " << sparse_op_name << " vs " << op_name << " with kernels: ";
    for (KernelType kernel : implementing_sparse_kernels) {
        ss << dispatcher.getKernelName(kernel) << ", ";
    }
    ss << "num_qubits: " << num_qubits;
    ss << ", unit_num_wires: " << unit_num_wires;
    ss << ", sparsity: " << sparsity;
    ss << ", inverse: " << inverse;

    const auto ini_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);

    SparseMatrixCSR<ComplexT> sparse_unitary;
    sparse_unitary.makeSparseUnitary(re, 1U << unit_num_wires, sparsity);

    const auto matrix = sparse_unitary.toDenseMatrix();

    DYNAMIC_SECTION(ss.str()) {
        const auto all_wires =
            CombinationGenerator(num_qubits, unit_num_wires).all_perms();
        for (const auto &wires : all_wires) {
            std::vector<TestVector<std::complex<PrecisionT>>> res;

            // Calculate with dense and sparse kernels and compare results.
            for (KernelType kernel : implementing_sparse_kernels) {
                auto st_dense = ini_st;
                dispatcher.applyMatrix(kernel, st_dense.data(), num_qubits,
                                       matrix.data(), wires, inverse);

                auto st_sparse = ini_st;
                dispatcher.applySparseMatrix(
                    kernel, st_sparse.data(), num_qubits,
                    sparse_unitary.row_map.data(),
                    sparse_unitary.col_idx.data(), sparse_unitary.values.data(),
                    wires, inverse);

                REQUIRE(st_dense == approx(st_sparse).margin(1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Compare sparse and dense apply matrix kernels",
                   "[Test_GateImplementations_CompareKernels]", float, double) {
    using PrecisionT = TestType;

    const std::size_t num_qubits = GENERATE(4, 8, 10);
    const std::size_t unit_num_wires = GENERATE(1, 2, 3);
    const PrecisionT sparsity = GENERATE(0.1, 0.5, 0.9);

    testSparseMatrixOp<TestType>(num_qubits, unit_num_wires, sparsity, false);
}

/**
 * @brief Apply the given controlled sparse matrix using all implementing
 * kernels and compare results with dense methods.
 * @tparam PrecisionT Precision of the complex data type.
 * @param num_qubits Number of qubits.
 * @param unit_num_wires Number of wires the matrix applies to.
 * @param inverse If inverse is required.
 */
template <typename PrecisionT>
void testControlledSparseMatrixOp(std::size_t num_qubits,
                                  std::size_t unit_num_wires,
                                  std::size_t control_wires,
                                  PrecisionT sparsity, bool inverse = false) {
    PL_ASSERT(unit_num_wires > 0);

    using ComplexT = std::complex<PrecisionT>;
    std::mt19937 re{1337};

    using Pennylane::Gates::Constant::controlled_matrix_names;
    const auto mat_op = [unit_num_wires]() -> ControlledMatrixOperation {
        switch (unit_num_wires) {
        case 1:
            return ControlledMatrixOperation::NCSingleQubitOp;
        case 2:
            return ControlledMatrixOperation::NCTwoQubitOp;
        default:
            return ControlledMatrixOperation::NCMultiQubitOp;
        }
    }();
    const auto op_name = lookup(controlled_matrix_names, mat_op);

    using Pennylane::Gates::Constant::controlled_sparse_matrix_names;
    const auto sparse_mat_op =
        [unit_num_wires]() -> ControlledSparseMatrixOperation {
        // for future expansion:
        [[maybe_unused]] auto unused_unit_num_wires = unit_num_wires;
        return ControlledSparseMatrixOperation::NCSparseMultiQubitOp;
    }();
    const auto sparse_op_name =
        lookup(controlled_sparse_matrix_names, sparse_mat_op);

    const auto implementing_sparse_kernels =
        kernelsImplementingControlledSparseMatrix<PrecisionT>(sparse_mat_op);
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    std::ostringstream ss;
    ss << "Test Controlled " << sparse_op_name << " vs " << op_name
       << " with kernels: ";
    for (KernelType kernel : implementing_sparse_kernels) {
        ss << dispatcher.getKernelName(kernel) << ", ";
    }
    ss << ", num_qubits: " << num_qubits;
    ss << ", unit_num_wires: " << unit_num_wires;
    ss << ", control_wires: " << control_wires;
    ss << ", sparsity: " << sparsity;
    ss << ", inverse: " << inverse;

    const auto ini_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);

    SparseMatrixCSR<ComplexT> sparse_unitary;
    sparse_unitary.makeSparseUnitary(re, 1U << unit_num_wires, sparsity);

    const auto matrix = sparse_unitary.toDenseMatrix();

    DYNAMIC_SECTION(ss.str()) {
        const auto all_wires =
            CombinationGenerator(num_qubits, unit_num_wires + control_wires)
                .all_perms();
        for (const auto &wires : all_wires) {
            std::vector<std::size_t> unitary_wires(
                wires.begin(), wires.begin() + unit_num_wires);
            std::vector<std::size_t> control_wires(
                wires.begin() + unit_num_wires, wires.end());

            std::vector<bool> control_values(control_wires.size());
            std::generate(control_values.begin(), control_values.end(),
                          [&re]() -> bool { return re() % 2; });

            // Calculate with dense and sparse kernels and compare results.
            for (KernelType kernel : implementing_sparse_kernels) {
                auto st_dense = ini_st;
                dispatcher.applyControlledMatrix(
                    kernel, st_dense.data(), num_qubits, matrix.data(),
                    control_wires, control_values, unitary_wires, inverse);

                auto st_sparse = ini_st;
                dispatcher.applyControlledSparseMatrix(
                    kernel, st_sparse.data(), num_qubits,
                    sparse_unitary.row_map.data(),
                    sparse_unitary.col_idx.data(), sparse_unitary.values.data(),
                    control_wires, control_values, unitary_wires, inverse);

                REQUIRE(st_dense == approx(st_sparse).margin(1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Compare sparse and dense apply controlled matrix kernels",
                   "[Test_GateImplementations_CompareKernels]", float, double) {
    using PrecisionT = TestType;

    const std::size_t num_qubits = GENERATE(6, 8, 10);
    const std::size_t unit_num_wires = GENERATE(1, 2, 3);
    const std::size_t control_wires = GENERATE(1, 2, 3);
    const PrecisionT sparsity = GENERATE(0.1, 0.9);

    testControlledSparseMatrixOp<TestType>(num_qubits, unit_num_wires,
                                           control_wires, sparsity, false);
}
