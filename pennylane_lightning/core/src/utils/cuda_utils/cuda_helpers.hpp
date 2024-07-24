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

// Adapted from JET: https://github.com/XanaduAI/jet.git

/**
 * @file cuda_helpers.hpp
 */

#pragma once
#include <algorithm>
#include <complex>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cusparse_v2.h>

#include "DevTag.hpp"
#include "cuError.hpp"

namespace Pennylane::LightningGPU::Util {

// SFINAE check for existence of real() method in complex type
template <typename ComplexT>
constexpr auto is_cxx_complex(const ComplexT &t) -> decltype(t.real(), bool()) {
    return true;
}

// Catch-all fallback for CUDA complex types
constexpr bool is_cxx_complex(...) { return false; }

inline cuFloatComplex operator-(const cuFloatComplex &a) {
    return {-a.x, -a.y};
}
inline cuDoubleComplex operator-(const cuDoubleComplex &a) {
    return {-a.x, -a.y};
}

template <class ComplexT_T, class ComplexT_U = ComplexT_T>
inline static auto Div(const ComplexT_T &a, const ComplexT_U &b) -> ComplexT_T {
    if constexpr (std::is_same_v<ComplexT_T, cuComplex> ||
                  std::is_same_v<ComplexT_T, float2>) {
        return cuCdivf(a, b);
    } else if (std::is_same_v<ComplexT_T, cuDoubleComplex> ||
               std::is_same_v<ComplexT_T, double2>) {
        return cuCdiv(a, b);
    }
}

/**
 * @brief Conjugate function for CXX & CUDA complex types
 *
 * @tparam ComplexT Complex data type. Supports std::complex<float>,
 * std::complex<double>, cuFloatComplex, cuDoubleComplex
 * @param a The given complex number
 * @return ComplexT The conjugated complex number
 */
template <class ComplexT>
__host__ __device__ inline static constexpr auto Conj(ComplexT a) -> ComplexT {
    if constexpr (std::is_same_v<ComplexT, cuComplex> ||
                  std::is_same_v<ComplexT, float2>) {
        return cuConjf(a);
    } else {
        return cuConj(a);
    }
}

/**
 * @brief Multiplies two numbers for CXX & CUDA complex types
 *
 * @tparam ComplexT Complex data type. Supports std::complex<float>,
 * std::complex<double>, cuFloatComplex, cuDoubleComplex
 * @param a Complex number
 * @param b Complex number
 * @return ComplexT The multiplication result
 */
template <class ComplexT>
__host__ __device__ inline static constexpr auto Cmul(ComplexT a, ComplexT b)
    -> ComplexT {
    if constexpr (std::is_same_v<ComplexT, cuComplex> ||
                  std::is_same_v<ComplexT, float2>) {
        return cuCmulf(a, b);
    } else {
        return cuCmul(a, b);
    }
}

/**
 * @brief Compile-time scalar real times complex number.
 *
 * @tparam U Precision of real value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Real scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class Real_t, class ComplexT = cuDoubleComplex>
inline static constexpr auto ConstMultSC(Real_t a, ComplexT b) -> ComplexT {
    if constexpr (std::is_same_v<ComplexT, cuDoubleComplex>) {
        return make_cuDoubleComplex(a * b.x, a * b.y);
    } else {
        return make_cuFloatComplex(a * b.x, a * b.y);
    }
}

/**
 * @brief Utility to convert cuComplex types to std::complex types
 *
 * @tparam ComplexT cuFloatComplex or cuDoubleComplex types.
 * @param a CUDA compatible complex type.
 * @return std::complex converted a
 */
template <class ComplexT = cuDoubleComplex>
inline static constexpr auto cuToComplex(ComplexT a)
    -> std::complex<decltype(a.x)> {
    return std::complex<decltype(a.x)>{a.x, a.y};
}

/**
 * @brief Utility to convert std::complex types to cuComplex types
 *
 * @tparam ComplexT std::complex types.
 * @param a A std::complex type.
 * @return cuComplex converted a
 */
template <class ComplexT = std::complex<double>>
inline static constexpr auto complexToCu(ComplexT a) {
    if constexpr (std::is_same_v<ComplexT, std::complex<double>>) {
        return make_cuDoubleComplex(a.real(), a.imag());
    } else {
        return make_cuFloatComplex(a.real(), a.imag());
    }
}

/**
 * @brief Utility to convert a vector of std::complex types to cuComplex types
 *
 * @tparam ComplexT std::complex types.
 * @param vec A std::vector<std::complex> type.
 * @return a vector of cuComplex converted vec
 */
template <class ComplexT = std::complex<double>>
inline auto complexToCu(const std::vector<ComplexT> &vec) {
    using cuComplexT = decltype(complexToCu(ComplexT{}));
    std::vector<cuComplexT> cast_vector(vec.size());
    std::transform(vec.begin(), vec.end(), cast_vector.begin(),
                   [&](ComplexT x) { return complexToCu<ComplexT>(x); });
    return cast_vector;
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
template <class ComplexT_T, class ComplexT_U = ComplexT_T>
inline static constexpr auto ConstMult(ComplexT_T a, ComplexT_U b)
    -> ComplexT_T {
    if constexpr (is_cxx_complex(b)) {
        return {a.real() * b.real() - a.imag() * b.imag(),
                a.real() * b.imag() + a.imag() * b.real()};
    } else {
        return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    }
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
template <class ComplexT_T, class ComplexT_U = ComplexT_T>
inline static constexpr auto ConstSum(ComplexT_T a, ComplexT_U b)
    -> ComplexT_T {
    if constexpr (std::is_same_v<ComplexT_T, cuComplex> ||
                  std::is_same_v<ComplexT_T, float2>) {
        return cuCaddf(a, b);
    } else {
        return cuCadd(a, b);
    }
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <class ComplexT> inline static constexpr auto ONE() -> ComplexT {
    return {1, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <class ComplexT> inline static constexpr auto ZERO() -> ComplexT {
    return {0, 0};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,1}
 */
template <class ComplexT> inline static constexpr auto IMAG() -> ComplexT {
    return {0, 1};
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <class ComplexT> inline static constexpr auto SQRT2() {
    if constexpr (std::is_same_v<ComplexT, float2> ||
                  std::is_same_v<ComplexT, cuFloatComplex>) {
        return ComplexT{0x1.6a09e6p+0F, 0}; // NOLINT: To be replaced in C++20
    } else if constexpr (std::is_same_v<ComplexT, double2> ||
                         std::is_same_v<ComplexT, cuDoubleComplex>) {
        return ComplexT{0x1.6a09e667f3bcdp+0,
                        0}; // NOLINT: To be replaced in C++20
    } else if constexpr (std::is_same_v<ComplexT, double>) {
        return 0x1.6a09e667f3bcdp+0; // NOLINT: To be replaced in C++20
    } else {
        return 0x1.6a09e6p+0F; // NOLINT: To be replaced in C++20
    }
}

/**
 * @brief Returns inverse sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T 1/sqrt(2)
 */
template <class ComplexT> inline static constexpr auto INVSQRT2() -> ComplexT {
    if constexpr (std::is_same_v<ComplexT, std::complex<float>> ||
                  std::is_same_v<ComplexT, std::complex<double>>) {
        return ComplexT(1 / M_SQRT2, 0);
    } else {
        return Div(ComplexT{1, 0}, SQRT2<ComplexT>());
    }
}

/**
 * If T is a supported data type for gates, this expression will
 * evaluate to `true`. Otherwise, it will evaluate to `false`.
 *
 * Supported data types are `float2`, `double2`, and their aliases.
 *
 * @tparam T candidate data type
 */
template <class T>
constexpr bool is_supported_data_type =
    std::is_same_v<T, cuComplex> || std::is_same_v<T, float2> ||
    std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, double2>;

/**
 * @brief Simple overloaded method to define CUDA data type.
 *
 * @param t
 * @return cuDoubleComplex
 */
inline cuDoubleComplex getCudaType(const double &t) {
    static_cast<void>(t);
    return {};
}
/**
 * @brief Simple overloaded method to define CUDA data type.
 *
 * @param t
 * @return cuFloatComplex
 */
inline cuFloatComplex getCudaType(const float &t) {
    static_cast<void>(t);
    return {};
}

/**
 * @brief Return the number of supported CUDA capable GPU devices.
 *
 * @return std::size_t
 */
inline int getGPUCount() {
    int result;
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&result));
    return result;
}

/**
 * @brief Return the current GPU device.
 *
 * @return int
 */
inline int getGPUIdx() {
    int result;
    PL_CUDA_IS_SUCCESS(cudaGetDevice(&result));
    return result;
}

inline static void deviceReset() { PL_CUDA_IS_SUCCESS(cudaDeviceReset()); }

/**
 * @brief Checks to see if the given GPU supports the
 * PennyLane-Lightning-GPU device. Minimum supported architecture is SM 7.0.
 *
 * @param device_number GPU device index
 * @return bool
 */
static bool isCuQuantumSupported(int device_number = 0) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_number);
    return deviceProp.major >= 7;
}

/**
 * @brief Get current GPU major.minor support
 *
 * @param device_number
 * @return std::pair<int,int>
 */
static std::pair<int, int> getGPUArch(int device_number = 0) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_number);
    return std::make_pair(deviceProp.major, deviceProp.minor);
}

/**
 * @brief Get free memory size on GPU device
 *
 * @return size_t
 */
inline std::size_t getFreeMemorySize() {
    std::size_t freeBytes{0}, totalBytes{0};
    PL_CUDA_IS_SUCCESS(cudaMemGetInfo(&freeBytes, &totalBytes));
    return freeBytes;
}

/**
 * Utility hash function for complex vectors representing matrices.
 */
struct MatrixHasher {
    template <class Precision = double>
    std::size_t
    operator()(const std::vector<std::complex<Precision>> &matrix) const {
        std::size_t hash_val = matrix.size();
        for (const auto &c_val : matrix) {
            hash_val ^= std::hash<Precision>()(c_val.real()) ^
                        std::hash<Precision>()(c_val.imag());
        }
        return hash_val;
    }
};

/**
 * @brief Normalize/Cast the index ordering to match PennyLane.
 *
 * @tparam IndexTypeIn Integer value type.
 * @tparam IndexTypeOut Integer value type.
 * @param indices Given indices to transform.
 * @param num_qubits Number of qubits.
 */
template <typename IndexTypeIn, typename IndexTypeOut>
inline auto NormalizeCastIndices(const std::vector<IndexTypeIn> &indices,
                                 const std::size_t &num_qubits)
    -> std::vector<IndexTypeOut> {
    std::vector<IndexTypeOut> t_indices(indices.size());
    std::transform(indices.begin(), indices.end(), t_indices.begin(),
                   [&](IndexTypeIn i) {
                       return static_cast<IndexTypeOut>(num_qubits - 1 - i);
                   });
    return t_indices;
}

} // namespace Pennylane::LightningGPU::Util
