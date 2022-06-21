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
 * Defines common AVX256 concept
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Macros.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <type_traits>

namespace Pennylane::Gates::AVX2 {
template <class PrecisionT> struct Intrinsic {
    static_assert(std::is_same_v<PrecisionT, float> ||
                      std::is_same_v<PrecisionT, double>,
                  "Data type for AVX256 must be float or double");
};

template <> struct Intrinsic<float> { using Type = __m256; };

template <> struct Intrinsic<double> { using Type = __m256d; };

template <class PrecisionT>
using IntrinsicType = typename Intrinsic<PrecisionT>::Type;
} // namespace Pennylane::Gates::AVX2

namespace Pennylane::Gates::AVX {
// clang-format off
constexpr __m256i setr256i(int32_t  e0, int32_t  e1, int32_t  e2, int32_t  e3,
		                   int32_t  e4, int32_t  e5, int32_t  e6, int32_t  e7) {
    return __m256i{(int64_t(e1) << 32) | e0,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e3) << 32) | e2,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e5) << 32) | e4,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e7) << 32) | e6}; // NOLINT(hicpp-signed-bitwise)
}
// clang-format on

template <> struct AVXIntrinsic<float, 8> {
    // AVX2
    using Type = __m256;
};
template <> struct AVXIntrinsic<double, 4> {
    // AVX2
    using Type = __m256d;
};

template <typename T> struct AVX2Concept {
    using PrecisionT = T;
    using IntrinsicType = AVX2::IntrinsicType<PrecisionT>;

    PL_FORCE_INLINE
    static auto load(const std::complex<PrecisionT> *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_load_ps(reinterpret_cast<const PrecisionT *>(p));
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_load_pd(reinterpret_cast<const PrecisionT *>(p));
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto loadu(const std::complex<PrecisionT> *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_loadu_ps(reinterpret_cast<const PrecisionT *>(p));
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_loadu_pd(reinterpret_cast<const PrecisionT *>(p));
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }
    PL_FORCE_INLINE
    static auto loadu(PrecisionT *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_loadu_ps(p);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_loadu_pd(p);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static void store(std::complex<PrecisionT> *p, IntrinsicType value) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            _mm256_store_ps(reinterpret_cast<PrecisionT *>(p), value);
        } else if (std::is_same_v<PrecisionT, double>) {
            _mm256_store_pd(reinterpret_cast<PrecisionT *>(p), value);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto mul(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_mul_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_mul_pd(v0, v1);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto add(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_add_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_add_pd(v0, v1);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }
};
template <> struct AVXConcept<float, 8> { using Type = AVX2Concept<float>; };
template <> struct AVXConcept<double, 4> { using Type = AVX2Concept<double>; };

template <> constexpr auto internalParity<float, 8>(size_t rev_wire) -> __m256 {
    switch (rev_wire) {
    case 0:
        return __m256{1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F};
    case 1:
        return __m256{1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F};
    default:
        PL_UNREACHABLE;
    }
    return _mm256_setzero_ps();
}
template <>
constexpr auto internalParity<double, 4>(size_t rev_wire) -> __m256d {
    switch (rev_wire) {
    case 0:
        return __m256d{1.0, 1.0, -1.0, -1.0};
    case 1:
        return __m256d{1.0, 1.0, 1.0, 1.0};
    default:
        PL_UNREACHABLE;
    }
    return _mm256_setzero_pd();
}

template <> struct ImagFactor<float, 8> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 8> {
        return __m256{-val, val, -val, val, -val, val, -val, val};
    };
};
template <> struct ImagFactor<double, 4> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 4> {
        return __m256d{-val, val, -val, val};
    };
};
template <> struct Set1<float, 8> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 8> {
        return __m256{val, val, val, val, val, val, val, val};
    }
};
template <> struct Set1<double, 4> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 4> {
        return __m256d{val, val, val, val};
    }
};
} // namespace Pennylane::Gates::AVX
