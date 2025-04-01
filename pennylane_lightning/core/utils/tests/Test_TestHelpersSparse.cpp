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
#include <random>

#include "TestHelpers.hpp"
#include "TestHelpersSparse.hpp"

using namespace Pennylane::Util;

TEMPLATE_TEST_CASE("SparseMatrixCSR data structure", "[SparseMatrixCSR]", float,
                   double) {
    using ComplexT = std::complex<TestType>;
    using IndexT = std::size_t;

    std::random_device rd;
    std::mt19937 gen(rd());
    SECTION("Default Constructor") {
        SparseMatrixCSR<ComplexT> matrix;

        REQUIRE(matrix.num_rows == 0);
        REQUIRE(matrix.num_cols == 0);
        REQUIRE(matrix.row_map.empty());
        REQUIRE(matrix.col_idx.empty());
        REQUIRE(matrix.values.empty());
    }

    SECTION("SparseMatrixCSR: Construct with sparse data") {
        std::vector<IndexT> row_map = {0, 2, 4};
        std::vector<IndexT> col_idx = {0, 1, 0, 1};
        std::vector<ComplexT> values = {2.0, 4.0, 6.0, 7.0};
        IndexT num_rows = 2;
        IndexT num_cols = 2;

        SparseMatrixCSR<ComplexT> matrix(row_map, col_idx, values, num_rows,
                                         num_cols);

        REQUIRE(matrix.num_rows == num_rows);
        REQUIRE(matrix.num_cols == num_cols);
        REQUIRE(matrix.row_map == row_map);
        REQUIRE(matrix.col_idx == col_idx);
        REQUIRE(matrix.values == values);
    }

    SECTION("SparseMatrixCSR: Construct from dense matrix") {
        std::vector<ComplexT> dense_matrix = {1.0, 3.0, 4.0, 0.0};
        IndexT num_rows = 2;

        SparseMatrixCSR<ComplexT> matrix(dense_matrix, num_rows);

        REQUIRE(matrix.num_rows == num_rows);
        REQUIRE(matrix.num_cols == num_rows);
        REQUIRE(matrix.row_map.size() == num_rows + 1);
        REQUIRE(matrix.col_idx.size() == 3);
        REQUIRE(matrix.values.size() == 3);
    }

    SECTION("SparseMatrixCSR: Convert to dense matrix") {
        std::vector<IndexT> row_map = {0, 2, 4};
        std::vector<IndexT> col_idx = {0, 1, 0, 1};
        std::vector<ComplexT> values = {2.0, 3.0, 0.0, 4.0};
        IndexT num_rows = 2;
        IndexT num_cols = 2;

        SparseMatrixCSR<ComplexT> matrix(row_map, col_idx, values, num_rows,
                                         num_cols);
        auto dense_matrix = matrix.toDenseMatrix();

        REQUIRE(dense_matrix.size() == num_rows * num_cols);
        for (IndexT i = 0; i < num_rows * num_cols; ++i) {
            REQUIRE(dense_matrix[i] == PLApproxComplex(values[i]).margin(1e-7));
        }
    }

    SECTION("SparseMatrixCSR: makeSparseUnitary dimension check") {
        SparseMatrixCSR<ComplexT> matrix;

        REQUIRE_THROWS_WITH(
            matrix.makeSparseUnitary(gen, 0, 0.5),
            Catch::Matchers::Contains("Dimension must be greater than 0."));
    }

    SECTION("SparseMatrixCSR: makeSparseUnitary sparsity check") {
        SparseMatrixCSR<ComplexT> matrix;

        REQUIRE_THROWS_WITH(
            matrix.makeSparseUnitary(gen, 4, -0.1),
            Catch::Matchers::Contains("Sparsity must be between 0 and 1."));
        REQUIRE_THROWS_WITH(
            matrix.makeSparseUnitary(gen, 4, 1.1),
            Catch::Matchers::Contains("Sparsity must be between 0 and 1."));
    }

    SECTION("SparseMatrixCSR: Make sparse unitary") {
        IndexT dimension = 4;
        double sparsity = 0.5;

        SparseMatrixCSR<ComplexT> matrix;
        matrix.makeSparseUnitary(gen, dimension, sparsity);

        REQUIRE(matrix.num_rows == dimension);
        REQUIRE(matrix.num_cols == dimension);
        REQUIRE(matrix.row_map.size() == dimension + 1);
        REQUIRE(matrix.col_idx.size() <= dimension * dimension);
        REQUIRE(matrix.values.size() <= dimension * dimension);
    }

    SECTION("SparseMatrixCSR: Output operator") {
        std::vector<IndexT> row_map = {0, 2, 4};
        std::vector<IndexT> col_idx = {0, 1, 0, 1};
        std::vector<ComplexT> values = {2.0, 5.0, 3.0, 4.0};
        IndexT num_rows = 2;
        IndexT num_cols = 2;

        SparseMatrixCSR<ComplexT> matrix(row_map, col_idx, values, num_rows,
                                         num_cols);

        std::ostringstream oss;
        oss << matrix;

        std::string expected_output = "Sparse Matrix (CSR):\n"
                                      "(0, 0): {2, 0}\n"
                                      "(0, 1): {5, 0}\n"
                                      "(1, 0): {3, 0}\n"
                                      "(1, 1): {4, 0}\n";
        REQUIRE(oss.str() == expected_output);
    }
}

TEMPLATE_TEST_CASE("SparseMatrixCSR: Write CSR vectors", "[write_CSR_vectors]",
                   float, double) {
    using ComplexT = std::complex<TestType>;
    using IndexT = std::size_t;

    IndexT num_rows = 3;
    SparseMatrixCSR<ComplexT> matrix;
    write_CSR_vectors(matrix, num_rows);

    REQUIRE(matrix.num_rows == num_rows);
    REQUIRE(matrix.row_map.size() == num_rows + 1);
    REQUIRE(matrix.col_idx.size() == 9);
    REQUIRE(matrix.values.size() == 9);
}
