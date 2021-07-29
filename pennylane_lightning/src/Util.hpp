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
#include <stdexcept>
#include <type_traits>

namespace Pennylane {

namespace Util {

/**
 * @brief Compile-time scalar real times complex number.
 *
 * @tparam T Precision of complex value and result.
 * @tparam U Precision of real value.
 * @param a Real scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr std::complex<T> ConstMult(U a, std::complex<T> b) {
    return {a * b.real(), a * b.imag()};
}
template <class T, class U = T>
inline static constexpr std::complex<T> ConstMult(std::complex<U> a,
                                                  std::complex<T> b) {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <class T> inline static constexpr std::complex<T> ONE() {
    return {1, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <class T> inline static constexpr std::complex<T> ZERO() {
    return {0, 0};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,1}
 */
template <class T> inline static constexpr std::complex<T> IMAG() {
    return {0, 1};
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <class T> inline static constexpr T SQRT2() {
    if constexpr (std::is_same_v<T, float>) {
        return {0x1.6a09e6p+0f};
    } else {
        return {0x1.6a09e667f3bcdp+0};
    }
}

/**
 * @brief Returns inverse sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T 1/sqrt(2)
 */
template <class T> inline static constexpr T INVSQRT2() {
    return {1 / SQRT2<T>()};
}

/**
 * Calculates 2^n for some integer n > 0 using bitshifts.
 *
 * @param n the exponent
 * @return value of 2^n
 */
inline size_t exp2(const size_t &n) { return static_cast<size_t>(1) << n; }

/**
 * @brief
 *
 * @param value
 * @return size_t
 */
inline size_t log2(size_t value) {
    return static_cast<size_t>(std::log2(value));
}

/**
 * Calculates the decimal value for a qubit, assuming a big-endian convention.
 *
 * @param qubitIndex the index of the qubit in the range [0, qubits)
 * @param qubits the number of qubits in the circuit
 * @return decimal value for the qubit at specified index
 */
inline size_t maxDecimalForQubit(size_t qubitIndex, size_t qubits) {
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
template <class T> inline size_t dimSize(const std::vector<T> &data) {
    const size_t s = data.size();
    const size_t s_sqrt = std::sqrt(s);

    if (s < 4)
        throw std::invalid_argument("The dataset must be at least 2x2.");
    if (((s == 0) || (s & (s - 1))))
        throw std::invalid_argument("The dataset must be a power of 2");
    if (s_sqrt * s_sqrt != s)
        throw std::invalid_argument("The dataset must be a perfect square");

    return log2(sqrt(data.size()));
}

/**
 * @brief Multiply the given gate data.
 *
 * @tparam T Floating point precision type.
 * @param left Left matrix.
 * @param right Right matrix.
 * @param out Output matrix data.
 */
template <typename T>
void GateMult(const std::vector<T> &left, const std::vector<T> &right,
              std::vector<T> &out) {
    if (left.size() != right.size())
        throw std::invalid_argument(
            "The supplied gates have incompatible sizes");

    T alpha = ONE<T>();
    T beta = ZERO<T>();

    auto A_data = left.data();
    auto B_data = right.data();
    auto C_data = out.data();

    // The following are assumed to be equal.
    size_t m = dimSize(left);
    size_t n = dimSize(right);
    size_t k = m;

    if constexpr (std::is_same_v<T, std::complex<float>>)
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                    A_data, m, B_data, n, &beta, C_data, k);
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                    A_data, m, B_data, n, &beta, C_data, k);
}

/**
 * @brief Multiply the given gate data.
 *
 * @tparam T Floating point precision type.
 * @param left Left matrix.
 * @param right Right matrix.
 */
template <class T>
inline static std::vector<std::complex<T>>
GateMult(const std::vector<std::complex<T>> &left,
         const std::vector<std::complex<T>> &right) {
    std::vector<std::complex<T>> out(left.size());
    GateMult<T>(left, right, out);
    return out;
}

} // namespace Util
} // namespace Pennylane

// Exception for functions that aren't implemented
class NotImplementedException : public std::logic_error {
  public:
    explicit NotImplementedException(const std::string &fname)
        : std::logic_error(std::string("Function is not implemented. ") +
                           fname){};
};
