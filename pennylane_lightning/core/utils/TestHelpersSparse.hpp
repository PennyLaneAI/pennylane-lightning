// Copyright 2025 Xanadu Quantum Technologies Inc.

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
 * Defines sparse helper methods for PennyLane Lightning.
 * @brief Utility functions for testing with Sparse Matrix CSR.
 */
#pragma once

#include <catch2/catch.hpp>
#include <complex>
#include <iostream>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

/// @cond DEV
namespace Pennylane::Util {

template <typename ComplexT, typename IndexT = std::size_t>
struct SparseMatrixCSR {
    // Define PrecisionT as the floating-point type corresponding to ComplexT
    using PrecisionT = typename std::remove_reference<
        decltype(std::declval<ComplexT>().real())>::type;

    std::vector<IndexT>
        row_map; // j element encodes the total number of non-zeros above row j
    std::vector<IndexT> col_idx;  // column indices
    std::vector<ComplexT> values; // matrix non-zero elements
    IndexT num_rows;              // matrix number of rows
    IndexT num_cols;              // matrix number of columns

    /**
     * @brief Default Constructor. Construct an empty Sparse Matrix CSR object.
     */
    SparseMatrixCSR() : num_rows{0}, num_cols{0} {}

    /**
     * @brief Construct a new Sparse Matrix CSR object with the provided sparse
     * data.
     *
     * @param row_map_ the j element encodes the total number of non-zeros above
     * row j.
     * @param col_idx_ column indices.
     * @param values_ matrix non-zero elements.
     * @param num_rows_ matrix number of rows.
     * @param num_cols_ matrix number of columns.
     */
    SparseMatrixCSR(const std::vector<IndexT> &row_map_,
                    const std::vector<IndexT> &col_idx_,
                    const std::vector<ComplexT> &values_, IndexT num_rows_,
                    IndexT num_cols_)
        : row_map{row_map_}, col_idx{col_idx_}, values{values_},
          num_rows{num_rows_}, num_cols{num_cols_} {}

    /**
     * @brief Construct a new Square Sparse Matrix CSR object with the provided
     * sparse data.
     *
     * @param row_map_ the j element encodes the total number of non-zeros above
     * row j.
     * @param col_idx_ column indices.
     * @param values_ matrix non-zero elements.
     * @param num_rows_ matrix number of rows.
     */
    SparseMatrixCSR(const std::vector<IndexT> &row_map_,
                    const std::vector<IndexT> &col_idx_,
                    const std::vector<ComplexT> &values_, IndexT num_rows_)
        : SparseMatrixCSR(row_map_, col_idx_, values_, num_rows_, num_rows_) {}

    /**
     * @brief Construct a new Square Sparse Matrix CSR object from a dense
     * matrix.
     *
     * @param dense_matrix dense matrix data.
     * @param num_rows_ matrix number of rows.
     * @param tolerance tolerance for non-zero elements.
     */
    SparseMatrixCSR(const std::vector<ComplexT> &dense_matrix,
                    IndexT num_rows_) {
        fromDenseMatrix(dense_matrix, num_rows_);
    }

    /**
     * @brief Construct a new Sparse Matrix CSR object from a dense matrix.
     *
     * @param dense_matrix dense matrix data.
     * @param num_rows_ matrix number of rows.
     * @param num_cols_ matrix number of columns.
     * @param tolerance tolerance for non-zero elements.
     */
    void fromDenseMatrix(const std::vector<ComplexT> &dense_matrix,
                         IndexT _num_rows, IndexT _num_cols,
                         PrecisionT tolerance = 1e-5) {
        num_rows = _num_rows;
        num_cols = _num_cols;
        row_map.reserve(num_rows + 1);
        // Estimate the number of non-zero elements based on tolerance and
        // reserve space
        IndexT est_NNZ = std::count_if(dense_matrix.begin(), dense_matrix.end(),
                                       [tolerance](const ComplexT &val) {
                                           return std::abs(val) > tolerance;
                                       });
        col_idx.reserve(est_NNZ);
        values.reserve(est_NNZ);

        row_map.push_back(0);
        for (IndexT rowIdx = 0; rowIdx < num_rows; ++rowIdx) {
            for (IndexT colIdx = 0; colIdx < num_cols; ++colIdx) {
                ComplexT val = dense_matrix[rowIdx * num_cols + colIdx];
                if (std::abs(val) > tolerance) {
                    col_idx.push_back(colIdx);
                    values.push_back(val);
                }
            }
            row_map.push_back(col_idx.size());
        }
    }

    /**
     * @brief Construct a new Square Square Sparse Matrix CSR object from a
     * dense matrix.
     *
     * @param dense_matrix dense matrix data.
     * @param num_rows_ matrix number of rows.
     * @param tolerance tolerance for non-zero elements.
     */
    void fromDenseMatrix(const std::vector<ComplexT> &dense_matrix,
                         IndexT _num_rows, PrecisionT tolerance = 1e-5) {
        fromDenseMatrix(dense_matrix, _num_rows, _num_rows, tolerance);
    }

    /**
     * @brief Generate a random unitary CSR sparse square matrix with a given
     * sparsity. This function will wipe out any existing data in the matrix,
     * and assumes a square matrix filled with ComplexT numbers. The
     * matrix is generated by filling the non-zero elements with random complex
     * numbers and then orthogonalizing the rows. The matrix is then normalized
     * to be unitary.
     * @param gen the random number generator.
     * @param dimension the dimension of the matrix.
     * @param sparsity the sparsity of the matrix.
     */
    template <typename RandomEngine>
    void makeSparseUnitary(RandomEngine &gen, IndexT dimension,
                           PrecisionT sparsity) {
        if (dimension <= 0) {
            PL_ABORT("Dimension must be greater than 0.");
        }
        if (sparsity < 0.0 || sparsity > 1.0) {
            PL_ABORT("Sparsity must be between 0 and 1.");
        }

        num_rows = dimension;
        num_cols = dimension;
        row_map.assign(1, 0);
        col_idx.clear();
        values.clear();

        std::uniform_real_distribution<PrecisionT> dist(0.0, 1.0);

        auto num_nonzero =
            static_cast<IndexT>((1 - sparsity) * dimension * dimension);

        // Custom comparator for the set
        auto tuple_comparator =
            [](const std::tuple<IndexT, IndexT, ComplexT> &a,
               const std::tuple<IndexT, IndexT, ComplexT> &b) {
                if (std::get<0>(a) < std::get<0>(b)) {
                    return true;
                }
                if (std::get<0>(a) > std::get<0>(b)) {
                    return false;
                }
                return std::get<1>(a) <
                       std::get<1>(b); // Compare columns if rows are equal
            };

        std::set<std::tuple<IndexT, IndexT, ComplexT>,
                 decltype(tuple_comparator)>
            nonzeros;

        while (nonzeros.size() < num_nonzero) {
            auto row = static_cast<IndexT>(dist(gen) * dimension);
            auto col = static_cast<IndexT>(dist(gen) * dimension);
            ComplexT val = ComplexT{dist(gen), dist(gen)};

            if (row == col) {
                nonzeros.insert({row, col, std::real(val)});
            } else if (row > col) {
                nonzeros.insert({row, col, val});
                nonzeros.insert({col, row, std::conj(val)});
            }
        }

        // fill the CSR format
        IndexT current_row = 0;
        for (const auto &[row, col, val] : nonzeros) {
            while (row > current_row) {
                row_map.push_back(values.size());
                current_row++;
            }
            col_idx.push_back(col);
            values.push_back(val);
        }

        // Add remaining row map values (for empty rows at the end)
        while (current_row < num_rows) {
            row_map.push_back(values.size());
            current_row++;
        }

        // Approximate unitary by normalizing rows.
        for (IndexT i = 0; i < num_rows; ++i) {
            PrecisionT norm = 0;
            for (IndexT j = row_map[i]; j < row_map[i + 1]; ++j) {
                norm += std::norm(values[j]);
            }
            norm = std::sqrt(norm);
            // There is always the possibility of a zero row.
            // But for testing purposes, we will just leave it as is.
            if (norm > 0) {
                for (IndexT j = row_map[i]; j < row_map[i + 1]; ++j) {
                    values[j] /= norm;
                }
            }
        }
    }

    /**
     * @brief Convert the Sparse Matrix CSR object to a dense matrix.
     * @return std::vector<ComplexT> the matrix dense row-major representation.
     * @note Only for testing purposes.
     */
    [[nodiscard]] auto toDenseMatrix() const -> std::vector<ComplexT> {
        std::vector<ComplexT> dense_matrix(num_rows * num_cols, 0.0);
        for (IndexT i = 0; i < num_rows; ++i) {
            for (IndexT j = row_map[i]; j < row_map[i + 1]; ++j) {
                dense_matrix[i * num_cols + col_idx[j]] = values[j];
            }
        }
        return dense_matrix;
    }

    /**
     * Define the << operator for the SparseMatrixCSR class.
     */
    friend std::ostream &
    operator<<(std::ostream &os,
               const SparseMatrixCSR<ComplexT, IndexT> &matrix) {
        os << "Sparse Matrix (CSR):" << std::endl;
        for (IndexT i = 0; i < matrix.num_rows; ++i) {
            for (IndexT j = matrix.row_map[i]; j < matrix.row_map[i + 1]; ++j) {
                auto value = matrix.values[j];
                os << "(" << i << ", " << matrix.col_idx[j] << "): {"
                   << std::real(value) << ", " << std::imag(value) << "}"
                   << std::endl;
            }
        }
        return os;
    }
};

/**
 * @brief Fills the empty vectors with the CSR (Compressed Sparse Row) sparse
 * matrix representation for a tri-diagonal + periodic boundary conditions
 * Hamiltonian.
 *
 * @tparam ComplexT data float point precision.
 * @tparam IndexT integer type used as indices of the sparse matrix.
 * @param row_map the j element encodes the total number of non-zeros above
 * row j.
 * @param col_idx column indices.
 * @param values  matrix non-zero elements.
 * @param num_rows matrix number of rows.
 */
template <class ComplexT, class IndexT>
void write_CSR_vectors(std::vector<IndexT> &row_map,
                       std::vector<IndexT> &col_idx,
                       std::vector<ComplexT> &values, IndexT num_rows) {
    const ComplexT SC_ONE = 1.0;

    row_map.resize(num_rows + 1);
    for (IndexT rowIdx = 1; rowIdx < static_cast<IndexT>(row_map.size());
         ++rowIdx) {
        row_map[rowIdx] = row_map[rowIdx - 1] + 3;
    };
    const IndexT numNNZ = row_map[num_rows];

    col_idx.resize(numNNZ);
    values.resize(numNNZ);
    for (IndexT rowIdx = 0; rowIdx < num_rows; ++rowIdx) {
        std::size_t idx = row_map[rowIdx];
        if (rowIdx == 0) {
            col_idx[0] = rowIdx;
            col_idx[1] = rowIdx + 1;
            col_idx[2] = num_rows - 1;

            values[0] = SC_ONE;
            values[1] = -SC_ONE;
            values[2] = -SC_ONE;
        } else if (rowIdx == num_rows - 1) {
            col_idx[idx] = 0;
            col_idx[idx + 1] = rowIdx - 1;
            col_idx[idx + 2] = rowIdx;

            values[idx] = -SC_ONE;
            values[idx + 1] = -SC_ONE;
            values[idx + 2] = SC_ONE;
        } else {
            col_idx[idx] = rowIdx - 1;
            col_idx[idx + 1] = rowIdx;
            col_idx[idx + 2] = rowIdx + 1;

            values[idx] = -SC_ONE;
            values[idx + 1] = SC_ONE;
            values[idx + 2] = -SC_ONE;
        }
    }
}

template <class ComplexT, class IndexT>
void write_CSR_vectors(SparseMatrixCSR<ComplexT, IndexT> &matrix,
                       IndexT num_rows) {
    matrix.num_rows = num_rows;
    write_CSR_vectors(matrix.row_map, matrix.col_idx, matrix.values, num_rows);
}

}; // namespace Pennylane::Util
/// @endcond
