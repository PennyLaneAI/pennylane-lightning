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
 * @file Util.hpp
 * Contains uncategorised utility functions.
 */
#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

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
/// @endcond

namespace Pennylane::Util::Internal {
/**
 * @brief Count the number of 1s in the binary representation of n.
 *
 * @param n Unsigned 32 bit integer
 */
constexpr auto countBit1(uint32_t n) -> size_t {
    n = (n & 0x55555555U) +         // NOLINT(readability-magic-numbers)
        ((n >> 1U) & 0x55555555U);  // NOLINT(readability-magic-numbers)
    n = (n & 0x33333333U) +         // NOLINT(readability-magic-numbers)
        ((n >> 2U) & 0x33333333U);  // NOLINT(readability-magic-numbers)
    n = (n & 0x0F0F0F0FU) +         // NOLINT(readability-magic-numbers)
        ((n >> 4U) & 0x0F0F0F0FU);  // NOLINT(readability-magic-numbers)
    n = (n & 0X00FF00FFU) +         // NOLINT(readability-magic-numbers)
        ((n >> 8U) & 0x00FF00FFU);  // NOLINT(readability-magic-numbers)
    n = (n & 0X0000FFFFU) +         // NOLINT(readability-magic-numbers)
        ((n >> 16U) & 0x0000FFFFU); // NOLINT(readability-magic-numbers)
    return n;
}

/**
 * @brief Count the number of 1s in the binary representation of n.
 *
 * @param n Unsigned 64 bit integer
 */
constexpr auto countBit1(uint64_t n) -> size_t {
    return countBit1(static_cast<uint32_t>(
               n & 0xFFFFFFFFU)) + // NOLINT(readability-magic-numbers)
           countBit1(static_cast<uint32_t>(
               n >> 32U)); // NOLINT(readability-magic-numbers)
}

/**
 * @brief Lookup table for number of trailing zeros in the binary
 * representation for 0 to 255 (8 bit integers).
 */
// NOLINTNEXTLINE (readability-magic-numbers)
constexpr uint8_t TRAILING_ZERO_LOOKUP_TABLE[256] = {
    0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};

/**
 * @brief Number of trailing zeros (starting from LSB) in the binary
 * representation of n.
 *
 * @param n Unsigned 8 bit integer
 */
constexpr auto countTrailing0(uint8_t n) -> size_t {
    return TRAILING_ZERO_LOOKUP_TABLE[n];
}

/**
 * @brief Number of trailing zeros (starting from LSB) in the binary
 * representation of n.
 *
 * @param n Unsigned 16 bit integer
 */
constexpr auto countTrailing0(uint16_t n) -> size_t {
    // NOLINTNEXTLINE (readability-magic-numbers)
    if (const auto mod = (n & 0xFFU); mod != 0) {
        return countTrailing0(static_cast<uint8_t>(mod));
    }
    // NOLINTNEXTLINE (readability-magic-numbers)
    return countTrailing0(static_cast<uint8_t>(n >> 8U)) + 8U;
}

/**
 * @brief Number of trailing zeros (starting from LSB) in the binary
 * representation of n.
 *
 * @param n Unsigned 32 bit integer
 */
constexpr auto countTrailing0(uint32_t n) -> size_t {
    // NOLINTNEXTLINE (readability-magic-numbers)
    if (const auto mod = (n & 0xFFFFU); mod != 0) {
        return countTrailing0(static_cast<uint16_t>(mod));
    }
    // NOLINTNEXTLINE (readability-magic-numbers)
    return countTrailing0(static_cast<uint16_t>(n >> 16U)) + 16U;
}
constexpr auto countTrailing0(uint64_t n) -> size_t {
    // NOLINTNEXTLINE (readability-magic-numbers)
    if (const auto mod = (n & 0xFFFFFFFFU); mod != 0) {
        return countTrailing0(static_cast<uint32_t>(mod));
    }
    // NOLINTNEXTLINE (readability-magic-numbers)
    return countTrailing0(static_cast<uint32_t>(n >> 32U)) + 32U;
}

} // namespace Pennylane::Util::Internal

namespace Pennylane::Util {

/**
 * @brief Compile-time scalar real times complex number.
 *
 * @tparam U Precision of real value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Real scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstMult(U a, std::complex<T> b)
    -> std::complex<T> {
    return {a * b.real(), a * b.imag()};
}

/**
 * @brief Compile-time scalar complex times complex.
 *
 * @tparam U Precision of complex value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstMult(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}
template <class T, class U = T>
inline static constexpr auto ConstMultConj(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return {a.real() * b.real() + a.imag() * b.imag(),
            -a.imag() * b.real() + a.real() * b.imag()};
}

/**
 * @brief Compile-time scalar complex summation.
 *
 * @tparam T Precision of complex value `a` and result.
 * @tparam U Precision of complex value `b`.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstSum(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return a + b;
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <class T> inline static constexpr auto ONE() -> std::complex<T> {
    return {1, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <class T> inline static constexpr auto ZERO() -> std::complex<T> {
    return {0, 0};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,1}
 */
template <class T> inline static constexpr auto IMAG() -> std::complex<T> {
    return {0, 1};
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <class T> inline static constexpr auto SQRT2() -> T {
    if constexpr (std::is_same_v<T, float>) {
        return 0x1.6a09e6p+0F; // NOLINT: To be replaced in C++20
    } else {
        return 0x1.6a09e667f3bcdp+0; // NOLINT: To be replaced in C++20
    }
}

/**
 * @brief Returns inverse sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T 1/sqrt(2)
 */
template <class T> inline static constexpr auto INVSQRT2() -> T {
    return {1 / SQRT2<T>()};
}

/**
 * @brief Calculates 2^n for some integer n > 0 using bitshifts.
 *
 * @param n the exponent
 * @return value of 2^n
 */
inline auto exp2(const size_t &n) -> size_t {
    return static_cast<size_t>(1) << n;
}

/**
 * @brief Log2 calculation.
 *
 * @param value Value to calculate for.
 * @return size_t
 */
inline auto log2(size_t value) -> size_t {
    return static_cast<size_t>(std::log2(value));
}

/**
 * @brief Define popcount for multiple compilers as well as different types.
 *
 * TODO: change to std::popcount in C++20
 */
///@{
#if defined(_MSC_VER)
inline auto popcount(uint64_t val) -> size_t {
    return static_cast<size_t>(__popcnt64(val));
}
#elif defined(__GNUC__) || defined(__clang__)
inline auto popcount(unsigned long val) -> size_t {
    return static_cast<size_t>(__builtin_popcountl(val));
}
#else
inline auto popcount(unsigned long val) -> size_t {
    return static_cast<size_t>(Internal::countBit1(val));
}
#endif
///@}

/**
 * @brief Faster log2 when the value is the perferct power of 2.
 *
 * If the value is the perfect power of 2, using a system provided bit operation
 * is much faster than std::log2
 *
 * TODO: change to std::countr_zero in C++20
 */
///@{
#if defined(_MSC_VER)
inline auto log2PerfectPower(uint64_t val) -> size_t {
    return static_cast<size_t>(63 - __lzcnt64(val));
}
#elif defined(__GNUC__) || defined(__clang__)
inline auto log2PerfectPower(unsigned long val) -> size_t {
    return static_cast<size_t>(__builtin_ctzl(val));
}
#else
inline auto log2PerfectPower(unsigned long val) -> size_t {
    return Internal::countTrailing0(val);
}
#endif
///@}

/**
 * @brief Check if there is a positive integer n such that value == 2^n.
 *
 * @param value Value to calculate for.
 * @return bool
 */
inline auto isPerfectPowerOf2(size_t value) -> bool {
    return popcount(value) == 1;
}

/**
 * @brief Calculates the decimal value for a qubit, assuming a big-endian
 * convention.
 *
 * @param qubitIndex the index of the qubit in the range [0, qubits)
 * @param qubits the number of qubits in the circuit
 * @return decimal value for the qubit at specified index
 */
inline auto maxDecimalForQubit(size_t qubitIndex, size_t qubits) -> size_t {
    assert(qubitIndex < qubits);
    return exp2(qubits - qubitIndex - 1);
}

/**
 * @brief Returns the number of wires supported by a given qubit gate.
 *
 * @tparam T Floating point precision type.
 * @param data Gate matrix data.
 * @return size_t Number of wires.
 */
template <class T> inline auto dimSize(const std::vector<T> &data) -> size_t {
    const size_t s = data.size();
    const auto s_sqrt = static_cast<size_t>(std::floor(std::sqrt(s)));

    if (s < 4) {
        throw std::invalid_argument("The dataset must be at least 2x2");
    }
    if (((s == 0) || (s & (s - 1)))) {
        throw std::invalid_argument("The dataset must be a power of 2");
    }
    if (s_sqrt * s_sqrt != s) {
        throw std::invalid_argument("The dataset must be a perfect square");
    }

    return static_cast<size_t>(log2(s_sqrt));
}

/**
 * @brief Calculates the inner-product using OpenMP.
 *
 * @tparam T Floating point precision type.
 * @tparam NTERMS Number of terms proceeds by each thread
 * @param v1 Complex data array 1.
 * @param v2 Complex data array 2.
 * @param result Calculated inner-product of v1 and v2.
 * @param data_size Size of data arrays.
 */
template <class T,
          size_t NTERMS = (1U << 19U)> // NOLINT(readability-magic-numbers)
inline static void
omp_innerProd(const std::complex<T> *v1, const std::complex<T> *v2,
              std::complex<T> &result, const size_t data_size) {
#if defined(_OPENMP)
#pragma omp declare \
            reduction (sm:std::complex<T>:omp_out=ConstSum(omp_out, omp_in)) \
            initializer(omp_priv=std::complex<T> {0, 0})
#endif

#if defined(_OPENMP)
    size_t nthreads = data_size / NTERMS;
    if (nthreads < 1) {
        nthreads = 1;
    }
#endif

#if defined(_OPENMP)
#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(v1, v2, data_size) reduction(sm                                     \
                                        : result)
#endif
    for (size_t i = 0; i < data_size; i++) {
        result = ConstSum(result, ConstMult(*(v1 + i), *(v2 + i)));
    }
}

/**
 * @brief Calculates the inner-product using the best available method.
 *
 * @tparam T Floating point precision type.
 * @tparam STD_CROSSOVER Threshold for using OpenMP method
 * @param v1 Complex data array 1.
 * @param v2 Complex data array 2.
 * @param data_size Size of data arrays.
 * @return std::complex<T> Result of inner product operation.
 */
template <class T,
          size_t STD_CROSSOVER = (1U
                                  << 20U)> // NOLINT(readability-magic-numbers)
inline auto innerProd(const std::complex<T> *v1, const std::complex<T> *v2,
                      const size_t data_size) -> std::complex<T> {
    std::complex<T> result(0, 0);

    if constexpr (USE_CBLAS) {
        if constexpr (std::is_same_v<T, float>) {
            cblas_cdotu_sub(data_size, v1, 1, v2, 1, &result);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zdotu_sub(data_size, v1, 1, v2, 1, &result);
        }
    } else {
        if (data_size < STD_CROSSOVER) {
            result = std::inner_product(
                v1, v1 + data_size, v2, std::complex<T>(), ConstSum<T>,
                static_cast<std::complex<T> (*)(
                    std::complex<T>, std::complex<T>)>(&ConstMult<T>));
        } else {
            omp_innerProd(v1, v2, result, data_size);
        }
    }
    return result;
}

/**
 * @brief Calculates the inner-product using OpenMP.
 * with the the first dataset conjugated.
 *
 * @tparam T Floating point precision type.
 * @tparam NTERMS Number of terms proceeds by each thread
 * @param v1 Complex data array 1.
 * @param v2 Complex data array 2.
 * @param result Calculated inner-product of v1 and v2.
 * @param data_size Size of data arrays.
 */
template <class T,
          size_t NTERMS = (1U << 19U)> // NOLINT(readability-magic-numbers)
inline static void
omp_innerProdC(const std::complex<T> *v1, const std::complex<T> *v2,
               std::complex<T> &result, const size_t data_size) {
#if defined(_OPENMP)
#pragma omp declare \
            reduction (sm:std::complex<T>:omp_out=ConstSum(omp_out, omp_in)) \
            initializer(omp_priv=std::complex<T> {0, 0})
#endif

#if defined(_OPENMP)
    size_t nthreads = data_size / NTERMS;
    if (nthreads < 1) {
        nthreads = 1;
    }
#endif

#if defined(_OPENMP)
#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(v1, v2, data_size) reduction(sm                                     \
                                        : result)
#endif
    for (size_t i = 0; i < data_size; i++) {
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
          size_t STD_CROSSOVER = (1U
                                  << 20U)> // NOLINT(readability-magic-numbers)
inline auto innerProdC(const std::complex<T> *v1, const std::complex<T> *v2,
                       const size_t data_size) -> std::complex<T> {
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
 * @brief Calculates the inner-product using the best available method.
 *
 * @see innerProd(const std::complex<T> *v1, const std::complex<T> *v2,
 * const size_t data_size)
 */
template <class T>
inline auto innerProd(const std::vector<std::complex<T>> &v1,
                      const std::vector<std::complex<T>> &v2)
    -> std::complex<T> {
    return innerProd(v1.data(), v2.data(), v1.size());
}

/**
 * @brief Calculates the inner-product using the best available method with the
 * first dataset conjugated.
 *
 * @see innerProdC(const std::complex<T> *v1, const std::complex<T> *v2,
 * const size_t data_size)
 */
template <class T>
inline auto innerProdC(const std::vector<std::complex<T>> &v1,
                       const std::vector<std::complex<T>> &v2)
    -> std::complex<T> {
    return innerProdC(v1.data(), v2.data(), v1.size());
}

/**
 * @brief Calculates the matrix-vector product using OpenMP.
 *
 * @tparam T Floating point precision type.
 * @param mat Complex data array repr. a flatten (row-wise) matrix m * n.
 * @param v_in Complex data array repr. a vector of shape n * 1.
 * @param v_out Pre-allocated complex data array to store the result.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @param transpose If `true`, considers transposed version of `mat`.
 * row-wise.
 */
template <class T>
inline static void omp_matrixVecProd(const std::complex<T> *mat,
                                     const std::complex<T> *v_in,
                                     std::complex<T> *v_out, size_t m, size_t n,
                                     bool transpose = false) {
    if (!v_out) {
        return;
    }

    size_t row;
    size_t col;

#if defined(_OPENMP)
#pragma omp parallel default(none) private(row, col)
#endif
    {
        if (transpose) {
#if defined(_OPENMP)
#pragma omp for
#endif
            for (row = 0; row < m; row++) {
                for (col = 0; col < n; col++) {
                    v_out[row] += mat[col * m + row] * v_in[col];
                }
            }
        } else {
#if defined(_OPENMP)
#pragma omp for
#endif
            for (row = 0; row < m; row++) {
                for (col = 0; col < n; col++) {
                    v_out[row] += mat[row * n + col] * v_in[col];
                }
            }
        }
    }
}

/**
 * @brief Calculates the matrix-vector product using the best available method.
 *
 * @tparam T Floating point precision type.
 * @param mat Complex data array repr. a flatten (row-wise) matrix m * n.
 * @param v_in Complex data array repr. a vector of shape n * 1.
 * @param v_out Pre-allocated complex data array to store the result.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @param transpose If `true`, considers transposed version of `mat`.
 */
template <class T>
inline void matrixVecProd(const std::complex<T> *mat,
                          const std::complex<T> *v_in, std::complex<T> *v_out,
                          size_t m, size_t n, bool transpose = false) {
    if (!v_out) {
        return;
    }

    if constexpr (USE_CBLAS) {
        constexpr std::complex<T> co{1, 0};
        constexpr std::complex<T> cz{0, 0};
        const auto tr = (transpose) ? CblasTrans : CblasNoTrans;
        if constexpr (std::is_same_v<T, float>) {
            cblas_cgemv(CblasRowMajor, tr, m, n, &co, mat, m, v_in, 1, &cz,
                        v_out, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zgemv(CblasRowMajor, tr, m, n, &co, mat, m, v_in, 1, &cz,
                        v_out, 1);
        }
    } else {
        omp_matrixVecProd(mat, v_in, v_out, m, n, transpose);
    }
}

/**
 * @brief Calculates the matrix-vector product using the best available method.
 *
 * @see void matrixVecProd(const std::complex<T> *mat, const
 * std::complex<T> *v_in, std::complex<T> *v_out, size_t m, size_t n, size_t
 * nthreads = 1, bool transpose = false)
 */
template <class T>
inline auto matrixVecProd(const std::vector<std::complex<T>> mat,
                          const std::vector<std::complex<T>> v_in, size_t m,
                          size_t n, bool transpose = false)
    -> std::vector<std::complex<T>> {
    if (mat.size() != m * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input matrix");
    }
    if (v_in.size() != n) {
        throw std::invalid_argument("Invalid size for the input vector");
    }

    std::vector<std::complex<T>> v_out(m);
    matrixVecProd(mat.data(), v_in.data(), v_out.data(), m, n, transpose);
    return v_out;
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
template <class T, size_t BLOCKSIZE = 32> // NOLINT(readability-magic-numbers)
inline static void CFTranspose(const T *mat, T *mat_t, size_t m, size_t n,
                               size_t m1, size_t m2, size_t n1, size_t n2) {
    size_t r;
    size_t s;

    size_t r1;
    size_t s1;
    size_t r2;
    size_t s2;

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
 * @brief Calculates vector-matrix product.
 *
 * @tparam T Floating point precision type.
 * @param v_in Data array repr. a vector of shape m * 1.
 * @param mat Data array repr. a flatten (row-wise) matrix m * n.
 * @param v_out Pre-allocated data array to store the result that is
 *              `mat_t \times v_in` where `mat_t` is transposed of `mat`.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 */
template <class T>
inline void vecMatrixProd(const T *v_in, const T *mat, T *v_out, size_t m,
                          size_t n) {
    if (!v_out) {
        return;
    }

    size_t i;
    size_t j;

    constexpr T z = static_cast<T>(0.0);
    bool allzero = true;
    for (j = 0; j < m; j++) {
        if (v_in[j] != z) {
            allzero = false;
            break;
        }
    }
    if (allzero) {
        return;
    }

    std::vector<T> mat_t(m * n);
    CFTranspose(mat, mat_t.data(), m, n, 0, m, 0, n);

    for (i = 0; i < n; i++) {
        T t = z;
        for (j = 0; j < m; j++) {
            t += mat_t[i * m + j] * v_in[j];
        }
        v_out[i] = t;
    }
}

/**
 * @brief Calculates the vactor-matrix product using the best available method.
 *
 * @see inline void vecMatrixProd(const T *v_in,
 * const T *mat, T *v_out, size_t m, size_t n)
 */
template <class T>
inline auto vecMatrixProd(const std::vector<T> &v_in, const std::vector<T> &mat,
                          size_t m, size_t n) -> std::vector<T> {
    if (v_in.size() != m) {
        throw std::invalid_argument("Invalid size for the input vector");
    }
    if (mat.size() != m * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input matrix");
    }

    std::vector<T> v_out(n);
    vecMatrixProd(v_in.data(), mat.data(), v_out.data(), m, n);

    return v_out;
}

/**
 * @brief Calculates the vactor-matrix product using the best available method.
 *
 * @see inline void vecMatrixProd(const T *v_in, const T *mat, T *v_out, size_t
 * m, size_t n)
 */
template <class T>
inline void vecMatrixProd(std::vector<T> &v_out, const std::vector<T> &v_in,
                          const std::vector<T> &mat, size_t m, size_t n) {
    if (mat.size() != m * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input matrix");
    }
    if (v_in.size() != m) {
        throw std::invalid_argument("Invalid size for the input vector");
    }
    if (v_out.size() != n) {
        throw std::invalid_argument("Invalid preallocated size for the result");
    }

    vecMatrixProd(v_in.data(), mat.data(), v_out.data(), m, n);
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
template <class T, size_t BLOCKSIZE = 32> // NOLINT(readability-magic-numbers)
inline static void CFTranspose(const std::complex<T> *mat,
                               std::complex<T> *mat_t, size_t m, size_t n,
                               size_t m1, size_t m2, size_t n1, size_t n2) {
    size_t r;
    size_t s;

    size_t r1;
    size_t s1;
    size_t r2;
    size_t s2;

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
template <class T>
inline auto Transpose(const std::vector<std::complex<T>> mat, size_t m,
                      size_t n) -> std::vector<std::complex<T>> {
    if (mat.size() != m * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input matrix");
    }

    std::vector<std::complex<T>> mat_t(n * m);
    CFTranspose(mat.data(), mat_t.data(), m, n, 0, m, 0, n);
    return mat_t;
}

/**
 * @brief Calculates matrix-matrix product using OpenMP.
 *
 * @tparam T Floating point precision type.
 * @tparam STRIDE Size of stride in the cache-blocking technique
 * @param m_left Row-wise flatten matrix of shape m * k.
 * @param m_right Row-wise flatten matrix of shape k * n.
 * @param m_out Pre-allocated row-wise flatten matrix of shape m * n.
 * @param m Number of rows of `m_left`.
 * @param n Number of columns of `m_right`.
 * @param k Number of rows of `m_right`.
 * @param transpose If `true`, requires transposed version of `m_right`.
 *
 * @note Consider transpose=true, to get a better performance.
 *  To transpose a matrix efficiently, check Util::Transpose
 */
template <class T, size_t STRIDE = 2> // NOLINT(readability-magic-numbers)
inline void omp_matrixMatProd(const std::complex<T> *m_left,
                              const std::complex<T> *m_right,
                              std::complex<T> *m_out, size_t m, size_t n,
                              size_t k, bool transpose = false) {
    if (!m_out) {
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel default(none)
#endif
    {
        size_t row;
        size_t col;
        size_t blk;
        if (transpose) {
#if defined(_OPENMP)
#pragma omp for
#endif
            for (row = 0; row < m; row++) {
                for (col = 0; col < n; col++) {
                    for (blk = 0; blk < k; blk++) {
                        m_out[row * n + col] +=
                            m_left[row * k + blk] * m_right[col * n + blk];
                    }
                }
            }
        } else {
            size_t i;
            size_t j;
            size_t l;
            std::complex<T> t;
#if defined(_OPENMP)
#pragma omp for
#endif
            for (row = 0; row < m; row += STRIDE) {
                for (col = 0; col < n; col += STRIDE) {
                    for (blk = 0; blk < k; blk += STRIDE) {
                        // cache-blocking:
                        for (i = row; i < std::min(row + STRIDE, m); i++) {
                            for (j = col; j < std::min(col + STRIDE, n); j++) {
                                t = 0;
                                for (l = blk; l < std::min(blk + STRIDE, k);
                                     l++) {
                                    t += m_left[i * k + l] * m_right[l * n + j];
                                }
                                m_out[i * n + j] += t;
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Calculates matrix-matrix product using the best avaiable method.
 *
 * @tparam T Floating point precision type.
 * @param m_left Row-wise flatten matrix of shape m * k.
 * @param m_right Row-wise flatten matrix of shape k * n.
 * @param m_out Pre-allocated row-wise flatten matrix of shape m * n.
 * @param m Number of rows of `m_left`.
 * @param n Number of columns of `m_right`.
 * @param k Number of rows of `m_right`.
 * @param transpose If `true`, requires transposed version of `m_right`.
 *
 * @note Consider transpose=true, to get a better performance.
 *  To transpose a matrix efficiently, check Util::Transpose
 */
template <class T>
inline void matrixMatProd(const std::complex<T> *m_left,
                          const std::complex<T> *m_right,
                          std::complex<T> *m_out, size_t m, size_t n, size_t k,
                          bool transpose = false) {
    if (!m_out) {
        return;
    }
    if constexpr (USE_CBLAS) {
        constexpr std::complex<T> co{1, 0};
        constexpr std::complex<T> cz{0, 0};
        const auto tr = (transpose) ? CblasTrans : CblasNoTrans;
        if constexpr (std::is_same_v<T, float>) {
            cblas_cgemm(CblasRowMajor, tr, CblasNoTrans, m, n, k, &co, m_left,
                        k, m_right, n, &cz, m_out, n);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zgemm(CblasRowMajor, tr, CblasNoTrans, m, n, k, &co, m_left,
                        k, m_right, n, &cz, m_out, n);
        }
    } else {
        omp_matrixMatProd(m_left, m_right, m_out, m, n, k, transpose);
    }
}

/**
 * @brief Calculates the matrix-matrix product using the best available method.
 *
 * @see void matrixMatProd(const std::complex<T> *m_left, const std::complex<T>
 * *m_right, std::complex<T> *m_out, size_t m, size_t n, size_t k, size_t
 * nthreads = 1, bool transpose = false)
 *
 * @note consider transpose=true, to get a better performance.
 *  To transpose a matrix efficiently, check Util::Transpose
 */
template <class T>
inline auto matrixMatProd(const std::vector<std::complex<T>> m_left,
                          const std::vector<std::complex<T>> m_right, size_t m,
                          size_t n, size_t k, bool transpose = false)
    -> std::vector<std::complex<T>> {
    if (m_left.size() != m * k) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input left matrix");
    }
    if (m_right.size() != k * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input right matrix");
    }

    std::vector<std::complex<T>> m_out(m * n);
    matrixMatProd(m_left.data(), m_right.data(), m_out.data(), m, n, k,
                  transpose);

    return m_out;
}

/**
 * @brief Streaming operator for vector data.
 *
 * @tparam T Vector data type.
 * @param os Output stream.
 * @param vec Vector data.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &os, const std::vector<T> &vec)
    -> std::ostream & {
    os << '[';
    if (!vec.empty()) {
        for (size_t i = 0; i < vec.size() - 1; i++) {
            os << vec[i] << ", ";
        }
        os << vec.back();
    }
    os << ']';
    return os;
}

/**
 * @brief Streaming operator for set data.
 *
 * @tparam T Vector data type.
 * @param os Output stream.
 * @param s Set data.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &os, const std::set<T> &s)
    -> std::ostream & {
    os << '{';
    for (const auto &e : s) {
        os << e << ",";
    }
    os << '}';
    return os;
}

/**
 * @brief Define linearly spaced data [start, end]
 *
 * @tparam T Data type.
 * @param start Start position.
 * @param end End position.
 * @param num_points Number of data-points in range.
 * @return std::vector<T>
 */
template <class T>
auto linspace(T start, T end, size_t num_points) -> std::vector<T> {
    std::vector<T> data(num_points);
    T step = (end - start) / (num_points - 1);
    for (size_t i = 0; i < num_points; i++) {
        data[i] = start + (step * i);
    }
    return data;
}

/**
 * @brief Determines the indices that would sort an array.
 *
 * @tparam T Vector data type.
 * @param arr Array to be inspected.
 * @return a vector with indices that would sort the array.
 */
template <typename T>
inline auto sorting_indices(const T &arr, size_t length)
    -> std::vector<size_t> {
    std::vector<size_t> indices(length);
    iota(indices.begin(), indices.end(), 0);

    // indices will be sorted in accordance to the array provided.
    sort(indices.begin(), indices.end(),
         [&arr](size_t i1, size_t i2) { return arr[i1] < arr[i2]; });

    return indices;
}

/**
 * @brief Determines the indices that would sort a vector.
 *
 * @tparam T Array data type.
 * @param vec Vector to be inspected.
 * @return a vector with indices that would sort the vector.
 */
template <typename T>
inline auto sorting_indices(const std::vector<T> &vec) -> std::vector<size_t> {
    return sorting_indices(vec.data(), vec.size());
}

/**
 * @brief Determines the transposed index of a tensor stored linearly.
 *  This function assumes each axis will have a length of 2 (|0>, |1>).
 *
 * @param ind index after transposition.
 * @param new_axes new axes distribution.
 * @return unsigned int with the new transposed index.
 */
inline auto transposed_state_index(size_t ind,
                                   const std::vector<size_t> &new_axes)
    -> size_t {
    size_t new_index = 0;
    for (size_t axis : new_axes) {
        new_index += (ind % 2) << axis;
        ind /= 2;
    }
    return new_index;
}

/**
 * @brief Template for the transposition of state tensors,
 * axes are assumed to have a length of 2 (|0>, |1>).
 *
 * @tparam T Tensor data type.
 * @param tensor Tensor to be transposed.
 * @param new_axes new axes distribution.
 * @return Transposed Tensor.
 */
template <typename T>
auto transpose_state_tensor(const std::vector<T> &tensor,
                            const std::vector<size_t> &new_axes)
    -> std::vector<T> {
    std::vector<T> transposed_tensor(tensor.size());
    for (size_t ind = 0; ind < tensor.size(); ind++) {
        transposed_tensor[transposed_state_index(ind, new_axes)] = tensor[ind];
    }
    return transposed_tensor;
}

/**
 * @brief Exception for functions that are not yet implemented.
 *
 */
class NotImplementedException : public std::logic_error {
  public:
    /**
     * @brief Construct a NotImplementedException exception object.
     *
     * @param fname Function name to indicate not implemented.
     */
    explicit NotImplementedException(const std::string &fname)
        : std::logic_error(std::string("Function is not implemented. ") +
                           fname){};
};

// Enable until C++20 support is explicitly allowed
template <class T> struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

/**
 * @brief Lookup key in array of pairs. For a constexpr map-like behavior.
 *
 * @tparam Key Type of keys
 * @tparam Value Type of values
 * @tparam size Size of std::array
 * @param arr Array to lookup
 * @param key Key to find
 */
template <typename Key, typename Value, size_t size>
constexpr auto lookup(const std::array<std::pair<Key, Value>, size> &arr,
                      const Key &key) -> Value {
    for (size_t idx = 0; idx < size; idx++) {
        if (std::get<0>(arr[idx]) == key) {
            return std::get<1>(arr[idx]);
        }
    }
    throw std::range_error("The given key does not exist.");
};

/**
 * @brief Check an array has an element.
 *
 * @tparam U Type of array elements.
 * @tparam size Size of array.
 * @param arr Array to check.
 * @param elt Element to find.
 */
template <typename U, size_t size>
constexpr auto array_has_elt(const std::array<U, size> &arr, const U &elt)
    -> bool {
    for (size_t idx = 0; idx < size; idx++) {
        if (arr[idx] == elt) {
            return true;
        }
    }
    return false;
};

/**
 * @brief Extract first elements from the array of pairs.
 *
 * @tparam T Type of the first elements.
 * @tparam U Type of the second elements.
 * @tparam size Size of the array.
 * @param arr Array to extract.
 */
template <typename T, typename U, size_t size>
constexpr std::array<T, size>
first_elts_of(const std::array<std::pair<T, U>, size> &arr) {
    // TODO: change to std::transform in C++20
    std::array<T, size> res = {
        T{},
    };
    for (size_t i = 0; i < size; i++) {
        res[i] = std::get<0>(arr[i]);
    }
    return res;
}
/**
 * @brief Extract second elements from the array of pairs.
 *
 * @tparam T Type of the first elements.
 * @tparam U Type of the second elements.
 * @tparam size Size of the array.
 * @param arr Array to extract.
 */
template <typename T, typename U, size_t size>
constexpr std::array<T, size>
second_elts_of(const std::array<std::pair<T, U>, size> &arr) {
    // TODO: change to std::transform in C++20
    std::array<T, size> res = {
        T{},
    };
    for (size_t i = 0; i < size; i++) {
        res[i] = std::get<0>(arr[i]);
    }
    return res;
}

/**
 * @brief Count the number of unique elements in the array.
 *
 * Current runtime is O(n^2).
 * TODO: count using sorted array in C++20 using constexpr std::sort.
 *
 * @tparam T Type of array elements
 * @tparam size Size of the array
 * @return size_t
 */
template <typename T, size_t size>
constexpr size_t count_unique(const std::array<T, size> &arr) {
    size_t res = 0;

    for (size_t i = 0; i < size; i++) {
        bool counted = false;
        for (size_t j = 0; j < i; j++) {
            if (arr[j] == arr[i]) {
                counted = true;
                break;
            }
        }
        if (!counted) {
            res++;
        }
    }
    return res;
}

/// @cond DEV
namespace Internal {
/**
 * @brief Helper function for prepend_to_tuple
 */
template <class T, class Tuple, std::size_t... I>
constexpr auto
prepend_to_tuple_helper(T &&elt, Tuple &&t,
                        [[maybe_unused]] std::index_sequence<I...> dummy) {
    return std::make_tuple(elt, std::get<I>(std::forward<Tuple>(t))...);
}
} // namespace Internal
/// @endcond

/**
 * @brief Prepent an element to a tuple
 * @tparam T Type of element
 * @tparam Tuple Type of the tuple (usually std::tuple)
 *
 * @param elt Element to prepend
 * @param t Tuple to add an element
 */
template <class T, class Tuple>
constexpr auto prepend_to_tuple(T &&elt, Tuple &&t) {
    return Internal::prepend_to_tuple_helper(
        std::forward<T>(elt), std::forward<Tuple>(t),
        std::make_index_sequence<
            std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

/**
 * @brief Transform a tuple to an array
 *
 * This function only works when all elements of the tuple are the same
 * type or convertible to the same type.
 *
 * @tparam T Type of the elements. This type usually needs to be specified.
 * @tparam Tuple Type of the tuple.
 * @param tuple Tuple to transform
 */
template <class T, class Tuple> constexpr auto tuple_to_array(Tuple &&tuple) {
    return std::apply(
        [](auto... n) { return std::array<T, sizeof...(n)>{n...}; },
        std::forward<Tuple>(tuple));
}
} // namespace Pennylane::Util
