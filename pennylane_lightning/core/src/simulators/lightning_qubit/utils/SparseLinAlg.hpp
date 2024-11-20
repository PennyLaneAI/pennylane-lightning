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
 * @file SparseLinAlg.hpp
 * @brief Contains sparse linear algebra utility functions.
 */

#pragma once
#include <complex>
#include <vector>
#include <thread>

namespace Pennylane::LightningQubit::Util {

/**
 * @brief Worker function to compute a segment of the matrix-vector multiplication for a sparse matrix.
 *
 * @tparam fp_precision data float point precision.
 * @tparam index_type integer type used as indices of the sparse matrix.
 * @param vector_ptr pointer to the vector.
 * @param row_map_ptr Pointer to the row_map array. Elements of this array
 * return the number of non-zero terms in all rows before it.
 * @param entries_ptr pointer to the column indices of the non-zero elements.
 * @param values_ptr non-zero elements.
 * @param result Reference to the output vector where results are stored.
 * @param start Index of the first row to process.
 * @param end Index of the last row (exclusive) to process.
 */
template <class fp_precision, class index_type>
void sparse_worker(const std::complex<fp_precision> *vector_ptr,
                   const index_type *row_map_ptr, 
                   const index_type *entries_ptr,
                   const std::complex<fp_precision> *values_ptr,
                   std::vector<std::complex<fp_precision>> &result,
                   index_type start, index_type end) {
    for (index_type i = start; i < end; i++) {
        std::complex<fp_precision> temp = 0.0;
        // Loop through all non-zero elements in row `i`
        for (index_type j = row_map_ptr[i]; j < row_map_ptr[i + 1]; j++) {
            temp += values_ptr[j] * vector_ptr[entries_ptr[j]];
        }
        result[i] = temp; // Store the computed value in the result vector
    }
}

/**
 * @brief Applies a sparse matrix to a vector using multi-threading.
 *
 * @tparam fp_precision data float point precision.
 * @tparam index_type integer type used as indices of the sparse matrix.
 * @param vector_ptr    pointer to the vector.
 * @param vector_size   size of the vector.
 * @param row_map_ptr   Pointer to the row_map array. Elements of this array
 * return the number of non-zero terms in all rows before it.
 * @param row_map_size  number of elements in the row_map.
 * @param entries_ptr   pointer to the column indices of the non-zero elements.
 * @param values_ptr    non-zero elements.
 * @param numNNZ        number of non-zero elements.
 * @return result       result of the matrix vector multiplication.
 */
template <class fp_precision, class index_type>
std::vector<std::complex<fp_precision>>
apply_Sparse_Matrix(const std::complex<fp_precision> *vector_ptr,
                            const index_type vector_size, 
                            const index_type *row_map_ptr,
                            [[maybe_unused]] const index_type row_map_size,
                            const index_type *entries_ptr,
                            const std::complex<fp_precision> *values_ptr,
                            [[maybe_unused]] const index_type numNNZ,
                            index_type num_threads = 0) {
    // Output vector initialized to zero
    std::vector<std::complex<fp_precision>> result(vector_size, std::complex<fp_precision>(0.0));

    // Determine the number of threads to use
    if (num_threads <= 0) {
        const int max_threads = std::thread::hardware_concurrency();
        num_threads = std::min(vector_size, static_cast<index_type>(max_threads));
    }
    
    // Divide the rows approximately evenly among the threads    
    index_type chunk_size = (vector_size + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    // Create and launch threads
    for (index_type t = 0; t < num_threads; ++t) {
        index_type start = t * chunk_size;
        index_type end = std::min(start + chunk_size, vector_size);

        // Only launch threads if there are rows to process
        if (start < vector_size) {
            threads.emplace_back(sparse_worker<fp_precision, index_type>,
                                 vector_ptr, row_map_ptr, entries_ptr, values_ptr,
                                 std::ref(result), start, end);
        }
    }

    // Wait for all threads to complete
    for (auto &th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    return result;
};

} // namespace Pennylane::LightningQubit::Util