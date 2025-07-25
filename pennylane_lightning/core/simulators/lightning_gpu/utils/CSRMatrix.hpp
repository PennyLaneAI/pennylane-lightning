// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include <algorithm>
#include <bit>
#include <complex>
#include <vector>

#include "MPIManagerGPU.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU::Util;
} // namespace
/// @endcond
namespace Pennylane::LightningGPU::MPI {
/**
 * @brief Manage memory of Compressed Sparse Row (CSR) sparse matrix. CSR format
 * represents a matrix M by three (one-dimensional) arrays, that respectively
 * contain nonzero values, row offsets, and column indices.
 *
 * @tparam Precision Floating-point precision type.
 * @tparam IndexT Integer type.
 */
template <class Precision, class IndexT> class CSRMatrix {
  private:
    std::vector<IndexT> columns_;
    std::vector<IndexT> csrOffsets_;
    std::vector<std::complex<Precision>> values_;

  public:
    CSRMatrix(std::size_t num_rows, std::size_t nnz)
        : columns_(nnz, 0), csrOffsets_(num_rows + 1, 0), values_(nnz) {};

    CSRMatrix(std::size_t num_rows, std::size_t nnz, IndexT *column_ptr,
              IndexT *csrOffsets_ptr, std::complex<Precision> *value_ptr)
        : columns_(column_ptr, column_ptr + nnz),
          csrOffsets_(csrOffsets_ptr, csrOffsets_ptr + num_rows + 1),
          values_(value_ptr, value_ptr + nnz) {};

    CSRMatrix() = default;

    /**
     * @brief Get the CSR format index vector of the matrix.
     */
    auto getColumns() -> std::vector<IndexT> & { return columns_; }

    /**
     * @brief Get CSR format offset vector of the matrix.
     */
    auto getCsrOffsets() -> std::vector<IndexT> & { return csrOffsets_; }

    /**
     * @brief Get CSR format data vector of the matrix.
     */
    auto getValues() -> std::vector<std::complex<Precision>> & {
        return values_;
    }
};

/**
 * @brief Convert a global CSR (Compressed Sparse Row) format matrix into
 * local blocks. This operation should be conducted on the rank 0.
 *
 * @tparam Precision Floating-point precision type.
 * @tparam IndexT Integer type used as indices of the sparse matrix.
 * @param mpi_manager MPIManagerGPU object.
 * @param num_rows Number of rows of the CSR matrix.
 * @param csrOffsets_ptr Pointer to the array of row offsets of the sparse
 * matrix. Array of size csrOffsets_size.
 * @param columns_ptr Pointer to the array of column indices of the sparse
 * matrix. Array of size numNNZ
 * @param values_ptr Pointer to the array of the non-zero elements
 *
 * @return auto A vector of vector of CSRMatrix.
 */
template <class Precision, class IndexT>
auto splitCSRMatrix(MPIManagerGPU &mpi_manager, const std::size_t &num_rows,
                    const IndexT *csrOffsets_ptr, const IndexT *columns_ptr,
                    const std::complex<Precision> *values_ptr)
    -> std::vector<std::vector<CSRMatrix<Precision, IndexT>>> {
    std::size_t num_row_blocks = mpi_manager.getSize();
    std::size_t num_col_blocks = num_row_blocks;

    std::vector<std::vector<CSRMatrix<Precision, IndexT>>> splitSparseMatrix(
        num_row_blocks,
        std::vector<CSRMatrix<Precision, IndexT>>(num_col_blocks));

    std::size_t row_block_size = num_rows / num_row_blocks;
    std::size_t col_block_size = row_block_size;

    // Add OpenMP support here later. Need to pay attention to
    // race condition.
    std::size_t current_global_row, current_global_col;
    std::size_t block_row_id, block_col_id;
    std::size_t local_row_id, local_col_id;
    for (std::size_t row = 0; row < num_rows; row++) {
        for (std::size_t col_idx =
                 static_cast<std::size_t>(csrOffsets_ptr[row]);
             col_idx < static_cast<std::size_t>(csrOffsets_ptr[row + 1]);
             col_idx++) {
            current_global_row = row;
            current_global_col = columns_ptr[col_idx];
            std::complex<Precision> current_val = values_ptr[col_idx];

            block_row_id = current_global_row / row_block_size;
            block_col_id = current_global_col / col_block_size;

            local_row_id = current_global_row % row_block_size;
            local_col_id = current_global_col % col_block_size;

            if (splitSparseMatrix[block_row_id][block_col_id]
                    .getCsrOffsets()
                    .size() == 0) {
                splitSparseMatrix[block_row_id][block_col_id].getCsrOffsets() =
                    std::vector<IndexT>(row_block_size + 1, 0);
            }

            splitSparseMatrix[block_row_id][block_col_id]
                .getCsrOffsets()[local_row_id + 1]++;
            splitSparseMatrix[block_row_id][block_col_id]
                .getColumns()
                .push_back(local_col_id);
            splitSparseMatrix[block_row_id][block_col_id].getValues().push_back(
                current_val);
        }
    }

    // Add OpenMP support here later.
    for (std::size_t block_row_id = 0; block_row_id < num_row_blocks;
         block_row_id++) {
        for (std::size_t block_col_id = 0; block_col_id < num_col_blocks;
             block_col_id++) {
            auto &localSpMat = splitSparseMatrix[block_row_id][block_col_id];
            std::size_t local_csr_offset_size =
                localSpMat.getCsrOffsets().size();
            for (std::size_t i0 = 1; i0 < local_csr_offset_size; i0++) {
                localSpMat.getCsrOffsets()[i0] +=
                    localSpMat.getCsrOffsets()[i0 - 1];
            }
        }
    }

    return splitSparseMatrix;
}

/**
 * @brief Scatter a CSR (Compressed Sparse Row) format matrix.
 *
 * @tparam Precision Floating-point precision type.
 * @tparam IndexT Integer type used as indices of the sparse matrix.
 * @param mpi_manager MPIManagerGPU object.
 * @param matrix CSR (Compressed Sparse Row) format matrix vector.
 * @param local_num_rows Number of rows of local CSR matrix.
 * @param root Root rank of the scatter operation.
 */
template <class Precision, class IndexT>
auto scatterCSRMatrix(MPIManagerGPU &mpi_manager,
                      std::vector<CSRMatrix<Precision, IndexT>> &matrix,
                      std::size_t local_num_rows, std::size_t root)
    -> CSRMatrix<Precision, IndexT> {
    std::size_t num_col_blocks = mpi_manager.getSize();

    std::vector<std::size_t> nnzs;

    if (mpi_manager.getRank() == root) {
        nnzs.reserve(matrix.size());
        for (std::size_t j = 0; j < matrix.size(); j++) {
            nnzs.push_back(matrix[j].getValues().size());
        }
    }

    std::size_t local_nnz = mpi_manager.scatter<std::size_t>(nnzs, 0)[0];

    CSRMatrix<Precision, IndexT> localCSRMatrix(local_num_rows, local_nnz);

    if (mpi_manager.getRank() == root) {
        localCSRMatrix.getValues() = matrix[0].getValues();
        localCSRMatrix.getCsrOffsets() = matrix[0].getCsrOffsets();
        localCSRMatrix.getColumns() = matrix[0].getColumns();
    }

    for (std::size_t k = 1; k < num_col_blocks; k++) {
        std::size_t dest = k;
        std::size_t source = root;

        if (mpi_manager.getRank() == 0 && matrix[k].getValues().size()) {
            mpi_manager.Send<std::complex<Precision>>(matrix[k].getValues(),
                                                      dest);
            mpi_manager.Send<IndexT>(matrix[k].getCsrOffsets(), dest);
            mpi_manager.Send<IndexT>(matrix[k].getColumns(), dest);
        } else if (mpi_manager.getRank() == k && local_nnz) {
            mpi_manager.Recv<std::complex<Precision>>(
                localCSRMatrix.getValues(), source);
            mpi_manager.Recv<IndexT>(localCSRMatrix.getCsrOffsets(), source);
            mpi_manager.Recv<IndexT>(localCSRMatrix.getColumns(), source);
        }
    }
    return localCSRMatrix;
}
} // namespace Pennylane::LightningGPU::MPI
