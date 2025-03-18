// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <catch2/catch.hpp>

#include "ConstantUtil.hpp"  // array_has_elem
#include "Gates.hpp"         // getPauliX, getPauliY
#include "LinearAlgebra.hpp" // randomUnitary
#include "TestHelpers.hpp"   // PrecisionToName
#include "TestHelpersSparse.hpp"
#include "TestHelpersWires.hpp"
#include "TestKernels.hpp"
#include "Util.hpp"
#include "cpu_kernels/GateImplementationsLM.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::Util;
using namespace Pennylane::LightningQubit::Gates;

// Function signature for the gate operation
template <typename PrecisionT>
using GateOperationT = void (*)(std::complex<PrecisionT> *, std::size_t,
                                const std::vector<std::size_t> &, bool);
} // namespace
/// @endcond

TEST_CASE("GateImplementationsLM::applyNCMultiQubitSparseOp throws if inverse "
          "is true",
          "[GateImplementationsLM]") {
    using namespace Pennylane::LightningQubit::Gates;
    using PrecisionT = double;
    using ComplexT = std::complex<PrecisionT>;
    using IndexT = std::size_t;

    const std::size_t num_qubits = 4;
    const std::vector<IndexT> controlled_wires = {0};
    const std::vector<bool> controlled_values = {true};
    const std::vector<IndexT> wires = {1, 2};
    const bool inverse = true;

    std::vector<ComplexT> state(1U << num_qubits, 0);
    state[0] = 1.0;

    std::vector<IndexT> row_map = {0, 1, 2, 3, 4};
    std::vector<IndexT> col_idx = {0, 1, 2, 3};
    std::vector<ComplexT> values = {1.0, 1.0, 1.0, 1.0};

    REQUIRE_THROWS_WITH(
        GateImplementationsLM::applyNCMultiQubitSparseOp(
            state.data(), num_qubits, row_map.data(), col_idx.data(),
            values.data(), controlled_wires, controlled_values, wires, inverse),
        Catch::Matchers::Contains("Inverse not implemented for sparse ops."));
}

// Gate operation as a parameter
template <typename PrecisionT>
void applyGateOperation(GateOperationT<PrecisionT> gateOp,
                        std::complex<PrecisionT> *state, std::size_t num_qubits,
                        const std::vector<std::size_t> &wires, bool inverse) {
    gateOp(state, num_qubits, wires, inverse);
}

// Encapsulates a single run of the applyNCMultiQubitSparseOp and checks the
// result.
/**
 * @brief Encapsulates a single run of the applyNCMultiQubitSparseOp and checks
 * the result.
 * @tparam ComplexT Precision of the complex data type.
 * @tparam VectorT State vector data type.
 * @param ref_st Reference state vector.
 * @param dense_matrix Dense matrix to be converted to a sparse matrix.
 * @param st State vector to apply the operation.
 * @param num_qubits Number of qubits.
 * @param control_wires Control wires.
 * @param control_values Control values.
 * @param target_wires Target wires.
 * @param inverse If inverse is required.
 * @param margin Margin for comparison.
 */
template <typename ComplexT, typename VectorT>
void applySparseNCMultiQubitOpRun(
    VectorT &ref_st, const std::vector<ComplexT> &dense_matrix, VectorT &st,
    const std::size_t num_qubits, const std::vector<std::size_t> &control_wires,
    const std::vector<bool> &control_values,
    const std::vector<std::size_t> &target_wires, const bool inverse,
    const typename ComplexT::value_type margin = 1e-5) {
    SparseMatrixCSR<ComplexT> sparse_matrix_from_dense(
        dense_matrix, std::sqrt(dense_matrix.size()));
    auto row_map_ptr = sparse_matrix_from_dense.row_map.data();
    auto col_idx_ptr = sparse_matrix_from_dense.col_idx.data();
    auto values_ptr = sparse_matrix_from_dense.values.data();

    GateImplementationsLM::applyNCMultiQubitSparseOp(
        st.data(), num_qubits, row_map_ptr, col_idx_ptr, values_ptr,
        control_wires, control_values, target_wires, inverse);

    REQUIRE(st == approx(ref_st).margin(margin));
}

/**
 * @brief Encapsulates five runs of applyGateOperation and uses
 * applySparseNCMultiQubitOpRun for running the second part and to check the
 * results.
 * @tparam PrecisionT Precision of the complex data type.
 * @tparam MatrixT Matrix data type.
 * @param gateOp Gate operation to be applied.
 * @param matrix Matrix to be applied.
 * @param num_qubits Number of qubits.
 * @param inverse If inverse is required.
 */
template <typename PrecisionT, typename MatrixT>
void testControlledOperation(GateOperationT<PrecisionT> gateOp,
                             const MatrixT matrix, const std::size_t num_qubits,
                             const bool inverse = false) {
    std::mt19937 re{1337};
    auto n = GENERATE(range(0, 5));
    re.seed(1337 + n); // changing seed for each iteration.

    std::vector<std::size_t> unitary_wires =
        createRandomWiresSubset(re, num_qubits, std::size_t{2});

    auto ref_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);
    auto st(ref_st);

    applyGateOperation(gateOp, ref_st.data(), num_qubits, unitary_wires,
                       inverse);

    applySparseNCMultiQubitOpRun(ref_st, matrix, st, num_qubits,
                                 {unitary_wires[0]}, {true}, {unitary_wires[1]},
                                 inverse);
}

/**
 * @brief Encapsulates five runs of applyNCMultiQubitOp and
 * applyNCMultiQubitSparseOp and checks the result.
 * @tparam PrecisionT Precision of the complex data type.
 * @tparam MatrixT Matrix data type.
 * @param matrix Matrix to be applied.
 * @param num_ctr_wires Number of control wires.
 * @param num_qubits Number of qubits.
 * @param inverse If inverse is required.
 */
template <typename PrecisionT, typename MatrixT>
void testControlledMatrix(const MatrixT matrix, const std::size_t num_ctr_wires,
                          const std::size_t num_qubits,
                          const bool inverse = false) {
    std::mt19937 re{1337};
    auto n = GENERATE(range(0, 5)); // number of tests.
    re.seed(1337 + n);              // changing seed for each iteration.

    // number of qubits in the unitary matrix.
    std::size_t unit_w_size = std::log2(std::sqrt(matrix.size()));

    std::vector<std::size_t> all_wires = createRandomWiresSubset(
        re, num_qubits, std::size_t{num_ctr_wires + unit_w_size});

    std::vector<std::size_t> unitary_wires(all_wires.begin(),
                                           all_wires.begin() + unit_w_size);
    std::vector<std::size_t> control_wires(all_wires.begin() + unit_w_size,
                                           all_wires.end());

    std::vector<bool> control_values(control_wires.size());
    std::generate(control_values.begin(), control_values.end(),
                  [&re]() -> bool { return re() % 2; });

    auto ref_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);
    auto st(ref_st);

    GateImplementationsLM::applyNCMultiQubitOp(
        ref_st.data(), num_qubits, matrix.data(), control_wires, control_values,
        unitary_wires, inverse);

    applySparseNCMultiQubitOpRun(ref_st, matrix, st, num_qubits, control_wires,
                                 control_values, unitary_wires, inverse);
}

template <typename PrecisionT> void testApplySparseNCMultiQubitOp() {
    std::mt19937 re{1337};
    const std::size_t num_qubits = 4;
    using ComplexT = std::complex<PrecisionT>;
    using MatrixT = typename std::vector<ComplexT>;
    SECTION("CNOT") {
        const MatrixT matrix = getPauliX<std::complex, PrecisionT>();
        testControlledOperation<PrecisionT>(GateImplementationsLM::applyCNOT,
                                            matrix, num_qubits);
    }
    SECTION("CY") {
        const MatrixT matrix = getPauliY<std::complex, PrecisionT>();
        testControlledOperation<PrecisionT>(GateImplementationsLM::applyCY,
                                            matrix, num_qubits);
    }
    SECTION("CZ") {
        const MatrixT matrix = getPauliZ<std::complex, PrecisionT>();
        testControlledOperation<PrecisionT>(GateImplementationsLM::applyCZ,
                                            matrix, num_qubits);
    }

    SECTION("Random Controlled Unitary - Varying number of qubits, control "
            "wires and unitary dimension") {
        const std::size_t num_qubits = GENERATE(6, 8, 10);
        const std::size_t control_wires = GENERATE(1, 2, 3);
        const std::size_t unitary_num_qubits = GENERATE(1, 2, 3);

        SparseMatrixCSR<ComplexT> sparse_unitary;
        sparse_unitary.makeSparseUnitary(re, 1U << unitary_num_qubits, 0.5);

        testControlledMatrix<PrecisionT>(sparse_unitary.toDenseMatrix(),
                                         control_wires, num_qubits);
    }
}

TEMPLATE_TEST_CASE(
    "GateImplementationsLM::applyNCMultiQubitSparseOp, inverse = false",
    "[GateImplementations_Matrix]", float, double) {
    using PrecisionT = TestType;

    testApplySparseNCMultiQubitOp<PrecisionT>();
}

/**
 * @brief Encapsulates a single run of the applyMultiQubitSparseOp and
 * checks the result.
 * @tparam ComplexT Precision of the complex data type.
 * @tparam VectorT State vector data type.
 * @param ref_st Reference state vector.
 * @param dense_matrix Dense matrix to be converted to a sparse matrix.
 * @param st State vector to apply the operation.
 * @param num_qubits Number of qubits.
 * @param wires Wires to apply the operation.
 * @param inverse If inverse is required.
 * @param margin Margin for comparison.
 */
template <typename ComplexT, typename VectorT>
void applySparseMultiQubitOpRun(
    VectorT &ref_st, const std::vector<ComplexT> &dense_matrix, VectorT &st,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    const bool inverse, const typename ComplexT::value_type margin = 1e-5) {
    SparseMatrixCSR<ComplexT> sparse_matrix_from_dense(
        dense_matrix, std::sqrt(dense_matrix.size()));

    auto row_map_ptr = sparse_matrix_from_dense.row_map.data();
    auto col_idx_ptr = sparse_matrix_from_dense.col_idx.data();
    auto values_ptr = sparse_matrix_from_dense.values.data();
    GateImplementationsLM::applyMultiQubitSparseOp(st.data(), num_qubits,
                                                   row_map_ptr, col_idx_ptr,
                                                   values_ptr, wires, inverse);

    REQUIRE(st == approx(ref_st).margin(margin));
}

template <typename PrecisionT, typename MatrixT>
void testPauliOperation(GateOperationT<PrecisionT> gateOp, const MatrixT matrix,
                        const std::size_t num_qubits,
                        const bool inverse = false) {
    std::mt19937 re{1337};
    auto n = GENERATE(range(0, 5));
    re.seed(1337 + n); // changing seed for each iteration.
    std::vector<std::size_t> unitary_wires =
        createRandomWiresSubset(re, num_qubits, std::size_t{1});

    auto ref_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);
    auto st(ref_st);

    applyGateOperation(gateOp, ref_st.data(), num_qubits, unitary_wires,
                       inverse);

    applySparseMultiQubitOpRun(ref_st, matrix, st, num_qubits, unitary_wires,
                               inverse);
}

template <typename ComplexT, typename VectorT>
void applyMultiQubitOpRunSparseUnitary(
    VectorT &ref_st, SparseMatrixCSR<ComplexT> &sparse_unitary, VectorT &st,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires,
    const bool inverse, const float margin = 1e-5) {

    GateImplementationsLM::applyMultiQubitSparseOp(
        st.data(), num_qubits, sparse_unitary.row_map.data(),
        sparse_unitary.col_idx.data(), sparse_unitary.values.data(), wires,
        inverse);

    auto dense_unitary = sparse_unitary.toDenseMatrix();

    GateImplementationsLM::applyMultiQubitOp(
        ref_st.data(), num_qubits, dense_unitary.data(), wires, inverse);

    REQUIRE(st == approx(ref_st).margin(margin));
}

template <typename PrecisionT> void testApplySparseMultiQubitOp() {
    std::mt19937 re{1337};
    bool inverse = false;
    using ComplexT = std::complex<PrecisionT>;
    using MatrixT = typename std::vector<ComplexT>;
    SECTION("PauliX") {
        const std::size_t num_qubits = 4;
        const MatrixT matrix = getPauliX<std::complex, PrecisionT>();
        testPauliOperation<PrecisionT>(GateImplementationsLM::applyPauliX,
                                       matrix, num_qubits);
    }
    SECTION("PauliY") {
        const std::size_t num_qubits = 4;
        const MatrixT matrix = getPauliY<std::complex, PrecisionT>();
        testPauliOperation<PrecisionT>(GateImplementationsLM::applyPauliY,
                                       matrix, num_qubits);
    }
    SECTION("PauliZ") {
        const std::size_t num_qubits = 4;
        const MatrixT matrix = getPauliZ<std::complex, PrecisionT>();
        testPauliOperation<PrecisionT>(GateImplementationsLM::applyPauliZ,
                                       matrix, num_qubits);
    }
    SECTION("Hadamard") {
        const std::size_t num_qubits = 4;
        const MatrixT matrix = getHadamard<std::complex, PrecisionT>();
        testPauliOperation<PrecisionT>(GateImplementationsLM::applyHadamard,
                                       matrix, num_qubits);
    }
    SECTION("Random Unitary - Varying Sparsity") {
        const PrecisionT sparsity = GENERATE(0.0, 0.1, 0.5, 0.9);
        const std::size_t num_qubits = 4;
        const std::size_t unitary_num_qubits = 4;

        std::vector<std::size_t> unitary_wires(unitary_num_qubits);
        std::iota(unitary_wires.begin(), unitary_wires.end(), 0);

        SparseMatrixCSR<ComplexT> sparse_unitary;
        sparse_unitary.makeSparseUnitary(re, 1U << num_qubits, sparsity);

        auto ref_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);
        auto st(ref_st);

        applyMultiQubitOpRunSparseUnitary(ref_st, sparse_unitary, st,
                                          num_qubits, unitary_wires, inverse);
    }

    SECTION("Random Unitary - Varying number of qubits") {
        const PrecisionT sparsity = 0.5;
        const std::size_t num_qubits = GENERATE(4, 8, 10);
        const std::size_t unitary_num_qubits = 4;
        std::vector<std::size_t> unitary_wires(unitary_num_qubits);
        std::iota(unitary_wires.begin(), unitary_wires.end(), 0);

        SparseMatrixCSR<ComplexT> sparse_unitary;
        sparse_unitary.makeSparseUnitary(re, 1U << unitary_num_qubits,
                                         sparsity);

        auto ref_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);
        auto st(ref_st);

        applyMultiQubitOpRunSparseUnitary(ref_st, sparse_unitary, st,
                                          num_qubits, unitary_wires, inverse);
    }
    SECTION("Random Unitary - Varying number of qubits - Random unitary "
            "wires") {
        const PrecisionT sparsity = 0.5;
        const std::size_t num_qubits = GENERATE(4, 8, 10);
        const std::size_t unitary_num_qubits = 4;
        std::vector<std::size_t> unitary_wires =
            createRandomWiresSubset(re, num_qubits, unitary_num_qubits);

        SparseMatrixCSR<ComplexT> sparse_unitary;
        sparse_unitary.makeSparseUnitary(re, 1U << unitary_num_qubits,
                                         sparsity);

        auto ref_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);
        auto st(ref_st);

        applyMultiQubitOpRunSparseUnitary(ref_st, sparse_unitary, st,
                                          num_qubits, unitary_wires, inverse);
    }
    SECTION("Random Unitary - Varying unitary dimension") {
        const PrecisionT sparsity = 0.5;
        const std::size_t num_qubits = 12;
        const std::size_t unitary_num_qubits = GENERATE(4, 8, 10);
        std::vector<std::size_t> unitary_wires(unitary_num_qubits);
        std::iota(unitary_wires.begin(), unitary_wires.end(), 0);

        SparseMatrixCSR<ComplexT> sparse_unitary;
        sparse_unitary.makeSparseUnitary(re, 1U << unitary_num_qubits,
                                         sparsity);

        auto ref_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);
        auto st(ref_st);

        applyMultiQubitOpRunSparseUnitary(ref_st, sparse_unitary, st,
                                          num_qubits, unitary_wires, inverse);
    }
}

TEMPLATE_TEST_CASE(
    "GateImplementationsLM::applyMultiQubitSparseOp, inverse = false",
    "[GateImplementations_Matrix]", float, double) {
    using PrecisionT = TestType;

    testApplySparseMultiQubitOp<PrecisionT>();
}
