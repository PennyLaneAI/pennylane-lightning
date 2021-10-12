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
 * @file
 * Contains uncategorised utility functions.
 */
#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <type_traits>

#include <iostream>

/// @cond DEV
#if __has_include(<cblas.h>) && defined _ENABLE_BLAS
#include <cblas.h>
#define USE_CBLAS 1
#else
#define USE_CBLAS 0
#endif
/// @endcond

namespace Pennylane {
namespace Util {

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
 * Calculates 2^n for some integer n > 0 using bitshifts.
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
 * Calculates the decimal value for a qubit, assuming a big-endian convention.
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
 * @brief Calculates the inner-product using the best available method.
 *
 * @tparam T Floating point precision type.
 * @param data_1 Complex data array 1.
 * @param data_2 Complex data array 2.
 * @param data_size Size of data arrays.
 * @return std::complex<T> Result of inner product operation.
 */
template <class T>
auto innerProd(const std::complex<T> *data_1, const std::complex<T> *data_2,
               const size_t data_size) -> std::complex<T> {
    std::complex<T> result(0, 0);

    if constexpr (USE_CBLAS) {
        if constexpr (std::is_same_v<T, float>) {
            cblas_cdotu_sub(data_size, data_1, 1, data_2, 1, &result);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zdotu_sub(data_size, data_1, 1, data_2, 1, &result);
        }
    } else {
        result = std::inner_product(
            data_1, data_1 + data_size, data_2, std::complex<T>(), ConstSum<T>,
            static_cast<std::complex<T> (*)(std::complex<T>, std::complex<T>)>(
                &ConstMult<T>));
    }
    return result;
}

/**
 * @brief Calculates the inner-product using the best available method with the
 * first dataset conjugated.
 *
 * @tparam T Floating point precision type.
 * @param data_1 Complex data array 1; conjugated before application.
 * @param data_2 Complex data array 2.
 * @param data_size Size of data arrays.
 * @return std::complex<T> Result of inner product operation.
 */
template <class T>
auto innerProdC(const std::complex<T> *data_1, const std::complex<T> *data_2,
                const size_t data_size) -> std::complex<T> {
    std::complex<T> result(0, 0);

    if constexpr (USE_CBLAS) {
        if constexpr (std::is_same_v<T, float>) {
            cblas_cdotc_sub(data_size, data_1, 1, data_2, 1, &result);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zdotc_sub(data_size, data_1, 1, data_2, 1, &result);
        }
    } else {
        result = std::inner_product(data_1, data_1 + data_size, data_2,
                                    std::complex<T>(), ConstSum<T>,
                                    ConstMultConj<T>);
    }
    return result;
}

/**
 * @brief Calculates the inner-product using the best available method.
 *
 * @see innerProd(const std::complex<T> *data_1, const std::complex<T> *data_2,
 * const size_t data_size)
 */
template <class T>
inline auto innerProd(const std::vector<std::complex<T>> &data_1,
                      const std::vector<std::complex<T>> &data_2)
    -> std::complex<T> {
    return innerProd(data_1.data(), data_2.data(), data_1.size());
}

/**
 * @brief Calculates the inner-product using the best available method with the
 * first dataset conjugated.
 *
 * @see innerProdC(const std::complex<T> *data_1, const std::complex<T> *data_2,
 * const size_t data_size)
 */
template <class T>
inline auto innerProdC(const std::vector<std::complex<T>> &data_1,
                       const std::vector<std::complex<T>> &data_2)
    -> std::complex<T> {
    return innerProdC(data_1.data(), data_2.data(), data_1.size());
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
    for (size_t i = 0; i < vec.size(); i++) {
        os << vec[i] << ",";
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
 * @brief Exception for functions that are not yet implemented.
 *
 */
class NotImplementedException : public std::logic_error {
  public:
    /**
     * @brief Construct a NotImplementedException exception object.
     *
     * @param fname Function name to indicate not imeplemented.
     */
    explicit NotImplementedException(const std::string &fname)
        : std::logic_error(std::string("Function is not implemented. ") +
                           fname){};
};

// Enable until C++20 support is explicitly allowed
template <class T> struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

} // namespace Util
} // namespace Pennylane
