// Copyright 2022 Xanadu Quantum Technologies Inc.

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
 * Contains Kokkos Sparse (linear algebra) utility functions.
 */

#pragma once

#include "Error.hpp"

#ifdef _ENABLE_KOKKOS

#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"
#include "Kokkos_Core.hpp"

constexpr bool USE_KOKKOS = true;

// Implementing Kokkos Sparse operations.
#include <complex>
#include <vector>

namespace Pennylane::Util {
using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;

using index_type = long int;
using index_view_type =
    typename Kokkos::View<index_type *, default_layout, device_type,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using const_index_view_type =
    typename Kokkos::View<const index_type *, default_layout, device_type,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class fp_precision> using data_type = Kokkos::complex<fp_precision>;

template <class fp_precision>
using crs_matrix_type =
    typename KokkosSparse::CrsMatrix<data_type<fp_precision>, index_type,
                                     device_type, void, index_type>;
template <class fp_precision>
using graph_type = typename crs_matrix_type<fp_precision>::staticcrsgraph_type;

template <class fp_precision>
using const_crs_matrix_type =
    typename KokkosSparse::CrsMatrix<const data_type<fp_precision>,
                                     const index_type, device_type, void,
                                     const index_type>;
template <class fp_precision>
using const_graph_type =
    typename const_crs_matrix_type<fp_precision>::staticcrsgraph_type;

template <class fp_precision>
using data_view_type =
    typename Kokkos::View<data_type<fp_precision> *, default_layout,
                          device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class fp_precision>
using const_data_view_type =
    typename Kokkos::View<const data_type<fp_precision> *, default_layout,
                          device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/**
 * @brief Create a Kokkos Sparse Matrix object with unmanaged views.
 *
 * @param row_map_ptr   Pointer to the row_map array.
 *                      The j element encodes the number of non-zeros above row
 * j.
 * @param numRows       Matrix total number or rows.
 * @param entries_ptr   Pointer to the array with the non-zero elements column
 * indices.
 * @param values_ptr    Pointer to the array with the non-zero elements.
 * @param numNNZ        Number of non-zero elements.
 * @return crs_matrix_type
 */
template <class fp_precision>
const_crs_matrix_type<fp_precision> create_Kokkos_Sparse_Matrix(
    const index_type *row_map_ptr, const index_type numRows,
    const index_type *entries_ptr, const std::complex<fp_precision> *values_ptr,
    const index_type numNNZ) {
    const_index_view_type row_map(row_map_ptr, numRows + 1);
    const_index_view_type entries(entries_ptr, numNNZ);
    const_data_view_type<fp_precision> values(
        reinterpret_cast<const Kokkos::complex<fp_precision> *>(values_ptr),
        numNNZ);

    const_graph_type<fp_precision> myGraph(entries, row_map);
    const_crs_matrix_type<fp_precision> SparseMatrix("matrix", numRows, values,
                                                     myGraph);
    return SparseMatrix;
}

/**
 * @brief Apply a sparse matrix to a vector with Kokkos.
 *
 * @tparam fp_precision
 * @param vector_ptr    pointer to the vector.
 * @param vector_size   size of the vector.
 * @param row_map_ptr   pointer to the row_map.
 *                      The j element encodes the number of non-zeros above row
 * j.
 * @param row_map_size  number of elements in the row_map.
 * @param entries_ptr   pointer to the column indices of the non-zero elements.
 * @param values_ptr    non-zero elements.
 * @param numNNZ        number of non-zero elements.
 * @return std::vector<data_type<fp_precision>>
 */
template <class fp_precision, class index_type>
void apply_Sparse_Matrix_Kokkos(
    const std::complex<fp_precision> *vector_ptr, const index_type vector_size,
    const index_type *row_map_ptr, const index_type row_map_size,
    const index_type *entries_ptr, const std::complex<fp_precision> *values_ptr,
    const index_type numNNZ, std::vector<std::complex<fp_precision>> &result) {

    Kokkos::initialize();
    {
        const_data_view_type<fp_precision> vector_view(
            reinterpret_cast<const Kokkos::complex<fp_precision> *>(vector_ptr),
            vector_size);
        Kokkos::View<data_type<fp_precision> *> result_view("rhs", vector_size);

        const_crs_matrix_type<fp_precision> myMatrix =
            create_Kokkos_Sparse_Matrix(row_map_ptr, row_map_size - 1,
                                        entries_ptr, values_ptr, numNNZ);

        const data_type<fp_precision> alpha(1.0);
        const data_type<fp_precision> beta;
        KokkosSparse::spmv("N", alpha, myMatrix, vector_view, beta,
                           result_view);

        result = std::move(std::vector<std::complex<fp_precision>>(
            reinterpret_cast<std::complex<fp_precision> *>(result_view.data()),
            reinterpret_cast<std::complex<fp_precision> *>(result_view.data()) +
                result_view.size()));
    }
    Kokkos::finalize();
};

} // namespace Pennylane::Util
#else
constexpr bool USE_KOKKOS = false;
namespace Pennylane::Util {

/**
 * @brief Apply a sparse matrix to a vector with Kokkos.
 *  Abort, because there is no Kokkos and Kokkos Kernels installation.
 *
 * @tparam fp_precision
 * @return std::vector<data_type<fp_precision>>
 */
template <class fp_precision, class index_type>
void apply_Sparse_Matrix_Kokkos(
    [[maybe_unused]] const std::complex<fp_precision> *vector_ptr,
    [[maybe_unused]] const index_type vector_size,
    [[maybe_unused]] const index_type *row_map_ptr,
    [[maybe_unused]] const index_type row_map_size,
    [[maybe_unused]] const index_type *entries_ptr,
    [[maybe_unused]] const std::complex<fp_precision> *values_ptr,
    [[maybe_unused]] const index_type numNNZ,
    [[maybe_unused]] std::vector<std::complex<fp_precision>> &result) {
    PL_ABORT("Executing the product of a Sparse matrix and a vector needs "
             "Kokkos and Kokkos Kernels installation.");
}; // pragma: no cover
} // namespace Pennylane::Util
#endif

namespace Pennylane::Util {

template <class fp_precision, class index_type>
std::vector<std::complex<fp_precision>> apply_Sparse_Matrix(
    [[maybe_unused]] const std::complex<fp_precision> *vector_ptr,
    [[maybe_unused]] const index_type vector_size,
    [[maybe_unused]] const index_type *row_map_ptr,
    [[maybe_unused]] const index_type row_map_size,
    [[maybe_unused]] const index_type *entries_ptr,
    [[maybe_unused]] const std::complex<fp_precision> *values_ptr,
    [[maybe_unused]] const index_type numNNZ) {
    std::vector<std::complex<fp_precision>> result;
    apply_Sparse_Matrix_Kokkos(vector_ptr, vector_size, row_map_ptr,
                               row_map_size, entries_ptr, values_ptr, numNNZ,
                               result);
    return result;
}
} // namespace Pennylane::Util