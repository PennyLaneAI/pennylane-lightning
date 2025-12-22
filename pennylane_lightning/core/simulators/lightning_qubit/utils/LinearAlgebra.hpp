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
 * Contains linear algebra utility functions.
 */
#pragma once

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "Macros.hpp"
#include "TypeTraits.hpp" // remove_complex_t
#include "Util.hpp"       // ConstSum, ConstMult, ConstMultConj

/// @cond DEV

#if __has_include(<cblas.h>) && defined _ENABLE_BLAS
#include <cblas.h>
constexpr bool USE_CBLAS = true;
#else
constexpr bool USE_CBLAS = false;
#ifndef CBLAS_TRANSPOSE
using CBLAS_TRANSPOSE = enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};
#endif

#ifndef CBLAS_LAYOUT
using CBLAS_LAYOUT = enum CBLAS_LAYOUT {
    CblasRowMajor = 101,
    CblasColMajor = 102
};
#endif
#endif

namespace {
using namespace Pennylane::Util;
} // namespace

/// @endcond

namespace Pennylane::LightningQubit::Util {
/// @cond DEV
/**
 * @brief Transpose enum class
 */
enum class Trans : int {
    NoTranspose = CblasNoTrans,
    Transpose = CblasTrans,
    Adjoint = CblasConjTrans
};
/// @endcond

/**
 * @brief Calculates the inner-product using OpenMP.
 * with the first dataset conjugated.
 *
 * @tparam T Floating point precision type.
 * @tparam NTERMS Number of terms proceeds by each thread
 * @param v1 Complex data array 1.
 * @param v2 Complex data array 2.
 * @param result Calculated inner-product of v1 and v2.
 * @param data_size Size of data arrays.
 */
template <class T,
          std::size_t NTERMS = (1U << 19U)> // NOLINT(readability-magic-numbers)
inline static void
omp_innerProdC(const std::complex<T> *v1, const std::complex<T> *v2,
               std::complex<T> &result, const std::size_t data_size) {
#if defined(_OPENMP)
#pragma omp declare reduction(sm : std::complex<T> : omp_out =                 \
                                  ConstSum(omp_out, omp_in))                   \
    initializer(omp_priv = std::complex<T>{0, 0})
#endif

#if defined(_OPENMP)
    std::size_t nthreads = data_size / NTERMS;
    if (nthreads < 1) {
        nthreads = 1;
    }
#endif

#if defined(_OPENMP)
#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(v1, v2, data_size) reduction(sm : result)
#endif
    for (std::size_t i = 0; i < data_size; i++) {
        result = ConstSum(result, ConstMultConj(*(v1 + i), *(v2 + i)));
    }
}

/**
 * @brief Calculates the inner-product using the best available method
 * with the first dataset conjugated.
 *
 * @tparam T Floating point precision type.
 * @tparam STD_CROSSOVER Threshold for using OpenMP method
 * @param v1 Complex data array 1; conjugated before application.
 * @param v2 Complex data array 2.
 * @param data_size Size of data arrays.
 * @return std::complex<T> Result of inner product operation.
 */
template <class T,
          std::size_t STD_CROSSOVER =
              (1U << 20U)> // NOLINT(readability-magic-numbers)
inline auto innerProdC(const std::complex<T> *v1, const std::complex<T> *v2,
                       const std::size_t data_size) -> std::complex<T> {
    std::complex<T> result(0, 0);

    if constexpr (USE_CBLAS) {
        if constexpr (std::is_same_v<T, float>) {
            cblas_cdotc_sub(data_size, v1, 1, v2, 1, &result);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zdotc_sub(data_size, v1, 1, v2, 1, &result);
        }
    } else {
        if (data_size < STD_CROSSOVER) {
            result =
                std::inner_product(v1, v1 + data_size, v2, std::complex<T>(),
                                   ConstSum<T>, ConstMultConj<T>);
        } else {
            omp_innerProdC(v1, v2, result, data_size);
        }
    }
    return result;
}

/**
 * @brief Calculates the inner-product using the best available method with the
 * first dataset conjugated.
 *
 * @see innerProdC(const std::complex<T> *v1, const std::complex<T> *v2,
 * const std::size_t data_size)
 */
template <class T, class AllocA, class AllocB>
inline auto innerProdC(const std::vector<std::complex<T>, AllocA> &v1,
                       const std::vector<std::complex<T>, AllocB> &v2)
    -> std::complex<T> {
    return innerProdC(v1.data(), v2.data(), v1.size());
}

/**
 * @brief Calculates transpose of a matrix recursively and Cache-Friendly
 * using blocking and Cache-optimized techniques.
 *
 * @tparam T Floating point precision type.
 * @tparam BLOCKSIZE Size of submatrices in the blocking technique.
 * @param mat Data array repr. a flatten (row-wise) matrix m * n.
 * @param mat_t Pre-allocated data array to store the transpose of `mat`.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @param m1 Index of the first row.
 * @param m2 Index of the last row.
 * @param n1 Index of the first column.
 * @param n2 Index of the last column.
 */
template <class T,
          std::size_t BLOCKSIZE = 16> // NOLINT(readability-magic-numbers)
inline static void CFTranspose(const T *mat, T *mat_t, std::size_t m,
                               std::size_t n, std::size_t m1, std::size_t m2,
                               std::size_t n1, std::size_t n2) {
    std::size_t r;
    std::size_t s;

    std::size_t r1;
    std::size_t s1;
    std::size_t r2;
    std::size_t s2;

    r1 = m2 - m1;
    s1 = n2 - n1;

    if (r1 >= s1 && r1 > BLOCKSIZE) {
        r2 = (m1 + m2) / 2;
        CFTranspose(mat, mat_t, m, n, m1, r2, n1, n2);
        m1 = r2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, n2);
    } else if (s1 > BLOCKSIZE) {
        s2 = (n1 + n2) / 2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, s2);
        n1 = s2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, n2);
    } else {
        for (r = m1; r < m2; r++) {
            for (s = n1; s < n2; s++) {
                mat_t[s * m + r] = mat[r * n + s];
            }
        }
    }
}

/**
 * @brief Calculates transpose of a matrix recursively and Cache-Friendly
 * using blocking and Cache-optimized techniques.
 *
 * @tparam T Floating point precision type.
 * @tparam BLOCKSIZE Size of submatrices in the blocking techinque.
 * @param mat Data array repr. a flatten (row-wise) matrix m * n.
 * @param mat_t Pre-allocated data array to store the transpose of `mat`.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @param m1 Index of the first row.
 * @param m2 Index of the last row.
 * @param n1 Index of the first column.
 * @param n2 Index of the last column.
 */
template <class T,
          std::size_t BLOCKSIZE = 16> // NOLINT(readability-magic-numbers)
inline static void CFTranspose(const std::complex<T> *mat,
                               std::complex<T> *mat_t, std::size_t m,
                               std::size_t n, std::size_t m1, std::size_t m2,
                               std::size_t n1, std::size_t n2) {
    std::size_t r;
    std::size_t s;

    std::size_t r1;
    std::size_t s1;
    std::size_t r2;
    std::size_t s2;

    r1 = m2 - m1;
    s1 = n2 - n1;

    if (r1 >= s1 && r1 > BLOCKSIZE) {
        r2 = (m1 + m2) / 2;
        CFTranspose(mat, mat_t, m, n, m1, r2, n1, n2);
        m1 = r2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, n2);
    } else if (s1 > BLOCKSIZE) {
        s2 = (n1 + n2) / 2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, s2);
        n1 = s2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, n2);
    } else {
        for (r = m1; r < m2; r++) {
            for (s = n1; s < n2; s++) {
                mat_t[s * m + r] = mat[r * n + s];
            }
        }
    }
}

/**
 * @brief Transpose a matrix of shape m * n to n * m using the
 * best available method.
 *
 * @tparam T Floating point precision type.
 * @param mat Row-wise flatten matrix of shape m * n.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @return mat transpose of shape n * m.
 */
template <class T, class Allocator = std::allocator<T>>
inline auto Transpose(std::span<const T> mat, std::size_t m, std::size_t n,
                      Allocator allocator = std::allocator<T>())
    -> std::vector<T, Allocator> {
    if (mat.size() != m * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input matrix");
    }

    std::vector<T, Allocator> mat_t(n * m, allocator);
    CFTranspose(mat.data(), mat_t.data(), m, n, 0, m, 0, n);
    return mat_t;
}

/**
 * @brief Transpose a matrix of shape m * n to n * m using the
 * best available method.
 *
 * This version may be merged with the above one when std::ranges is well
 * supported.
 *
 * @tparam T Floating point precision type.
 * @param mat Row-wise flatten matrix of shape m * n.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @return mat transpose of shape n * m.
 */
template <class T, class Allocator>
inline auto Transpose(const std::vector<T, Allocator> &mat, std::size_t m,
                      std::size_t n) -> std::vector<T, Allocator> {
    return Transpose(std::span<const T>{mat}, m, n, mat.get_allocator());
}

/**
 * @brief @rst
 * Calculate :math:`y += a*x` for a scalar :math:`a` and a vector :math:`x`
 * using OpenMP
 * @endrst
 *
 * @tparam STD_CROSSOVER The number of dimension after which OpenMP version
 * outperforms the standard method.
 *
 * @param dim Dimension of data
 * @param a Scalar to scale x
 * @param x Vector to add
 * @param y Vector to be added
 */
template <
    class T,
    std::size_t STD_CROSSOVER = 1U << 12U> // NOLINT(readability-magic-numbers)
void omp_scaleAndAdd(std::size_t dim, std::complex<T> a,
                     const std::complex<T> *x, std::complex<T> *y) {
    if (dim < STD_CROSSOVER) {
        for (std::size_t i = 0; i < dim; i++) {
            y[i] += a * x[i];
        }
    } else {
#if defined(_OPENMP)
#pragma omp parallel for default(none) firstprivate(a, dim, x, y)
#endif
        for (std::size_t i = 0; i < dim; i++) {
            y[i] += a * x[i];
        }
    }
}

/**
 * @brief @rst
 * Calculate :math:`y += a*x` for a scalar :math:`a` and a vector :math:`x`
 * using BLAS.
 * @endrst
 *
 * @param dim Dimension of data
 * @param a Scalar to scale x
 * @param x Vector to add
 * @param y Vector to be added
 */
template <class T>
void blas_scaleAndAdd(std::size_t dim, std::complex<T> a,
                      const std::complex<T> *x, std::complex<T> *y) {
    if constexpr (std::is_same_v<T, float>) {
        cblas_caxpy(dim, &a, x, 1, y, 1);
    } else if (std::is_same_v<T, double>) {
        cblas_zaxpy(dim, &a, x, 1, y, 1);
    } else {
        static_assert(
            std::is_same_v<T, float> || std::is_same_v<T, double>,
            "This procedure only supports a single or double precision "
            "floating point types.");
    }
}

/**
 * @brief @rst
 * Calculate :math:`y += a*x` for a scalar :math:`a` and a vector :math:`x`
 * using the best available method.
 * @endrst
 *
 *
 * @param dim Dimension of data
 * @param a Scalar to scale x
 * @param x Vector to add
 * @param y Vector to be added
 */
template <class T>
void scaleAndAdd(std::size_t dim, std::complex<T> a, const std::complex<T> *x,
                 std::complex<T> *y) {
    if constexpr (USE_CBLAS) {
        blas_scaleAndAdd(dim, a, x, y);
    } else {
        omp_scaleAndAdd(dim, a, x, y);
    }
}
/**
 * @brief @rst
 * Calculate :math:`y += a*x` for a scalar :math:`a` and a vector :math:`x`.
 * @endrst
 *
 * @param dim Dimension of data
 * @param a Scalar to scale x
 * @param x Vector to add
 * @param y Vector to be added
 */
template <class T>
void scaleAndAdd(std::complex<T> a, const std::vector<std::complex<T>> &x,
                 std::vector<std::complex<T>> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Dimensions of vectors mismatch");
    }
    scaleAndAdd(x.size(), a, x.data(), y.data());
}
} // namespace Pennylane::LightningQubit::Util
