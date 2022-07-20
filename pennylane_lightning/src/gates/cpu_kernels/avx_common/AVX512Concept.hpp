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
 * Defines common AVX512 concept
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Macros.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <type_traits>

namespace Pennylane::Gates::AVX512 {
template <class PrecisionT> struct Intrinsic {
    static_assert(std::is_same_v<PrecisionT, float> ||
                      std::is_same_v<PrecisionT, double>,
                  "Data type for AVX512 must be float or double");
};

template <> struct Intrinsic<float> { using Type = __m512; };

template <> struct Intrinsic<double> { using Type = __m512d; };

template <class PrecisionT>
using IntrinsicType = typename Intrinsic<PrecisionT>::Type;
} // namespace Pennylane::Gates::AVX512

namespace Pennylane::Gates::AVX {
template <> struct AVXIntrinsic<float, 16> {
    // AVX512
    using Type = __m512;
};
template <> struct AVXIntrinsic<double, 8> {
    // AVX512
    using Type = __m512d;
};

template <typename T> struct AVX512Concept {
    using PrecisionT = T;
    using IntrinsicType = AVX512::IntrinsicType<PrecisionT>;

    PL_FORCE_INLINE
    static auto load(std::complex<PrecisionT> *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_load_ps(p);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_load_pd(p);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto loadu(std::complex<PrecisionT> *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_loadu_ps(p);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_loadu_pd(p);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto loadu(PrecisionT *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_loadu_ps(p);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_loadu_pd(p);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static void store(std::complex<PrecisionT> *p, IntrinsicType value) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            _mm512_store_ps(p, value);
        } else if (std::is_same_v<PrecisionT, double>) {
            _mm512_store_pd(p, value);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto mul(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_mul_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_mul_pd(v0, v1);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto add(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_add_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_add_pd(v0, v1);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }
};

template <> struct AVXConcept<float, 16> { using Type = AVX512Concept<float>; };
template <> struct AVXConcept<double, 8> {
    using Type = AVX512Concept<double>;
};

template <>
constexpr auto internalParity<float, 16>(size_t rev_wire) -> __m512 {
    // AVX512 with float
    // clang-format off
    switch(rev_wire) {
    case 0:
        // When Z is applied to 0th qubit
        return __m512{1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F,
                      1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F};
    case 1:
        // When Z is applied to 1st qubit
        return __m512{1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
                      1.0F, 1.0F, 1.0F, 1.0F, -1.0F,- 1.0F, -1.0F, -1.0F};
    case 2:
        // When Z is applied to 2nd qubit
        return __m512{ 1.0F,  1.0F,  1.0F,  1.0F,
                       1.0F,  1.0F,  1.0F,  1.0F,
                      -1.0F, -1.0F, -1.0F, -1.0F,
                      -1.0F,- 1.0F, -1.0F, -1.0F};
    default:
        PL_UNREACHABLE;
    }
    // clang-format on
    return __m512{
        0,
    };
};

template <>
constexpr auto internalParity<double, 8>(size_t rev_wire) -> __m512d {
    // AVX512 with double
    switch (rev_wire) {
    case 0:
        // When Z is applied to 0th qubit
        return __m512d{1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0};
    case 1:
        // When Z is applied to 1st qubit
        return __m512d{1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0};
    default:
        PL_UNREACHABLE;
    }
    return __m512d{
        0,
    };
}

template <> struct ImagFactor<float, 16> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 16> {
        return __m512{-val, val, -val, val, -val, val, -val, val,
                      -val, val, -val, val, -val, val, -val, val};
    };
};
template <> struct ImagFactor<double, 8> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 8> {
        return __m512d{-val, val, -val, val, -val, val, -val, val};
    };
};

template <> struct Set1<float, 16> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 16> {
        return __m512{val, val, val, val, val, val, val, val,
                      val, val, val, val, val, val, val, val};
    }
};
template <> struct Set1<double, 8> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 8> {
        return __m512d{val, val, val, val, val, val, val, val};
    }
};
} // namespace Pennylane::Gates::AVX
