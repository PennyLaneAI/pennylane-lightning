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
 * Define some builtin alternatives
 */
#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define PL_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#define PL_UNREACHABLE __assume(false)
#else // Unsupported compiler
#define PL_UNREACHABLE
#endif

#if defined(__AVX2__)
#define PL_USE_AVX2 1
[[maybe_unused]] static constexpr bool use_avx2 = true;
#else
[[maybe_unused]] static constexpr bool use_avx2 = false;
#endif

#if defined(__AVX512F__)
#define PL_USE_AVX512F 1
[[maybe_unused]] static constexpr bool use_avx512f = true;
#else
[[maybe_unused]] static constexpr bool use_avx512f = false;
#endif

#if defined(__AVX512DQ__)
#define PL_USE_AVX512DQ 1
[[maybe_unused]] static constexpr bool use_avx512dq = true;
#else
[[maybe_unused]] static constexpr bool use_avx512dq = false;
#endif

#if defined(__AVX512VL__)
#define PL_USE_AVX512VL 1
[[maybe_unused]] static constexpr bool use_avx512vl = true;
#else
[[maybe_unused]] static constexpr bool use_avx512vl = false;
#endif

#if defined(_OPENMP)
#define PL_USE_OMP 1
[[maybe_unused]] static constexpr bool use_openmp = true;
#else
[[maybe_unused]] static constexpr bool use_openmp = false;
#endif

#if (_OPENMP >= 202011)
#define PL_UNROLL_LOOP __Pragma("omp unroll(8)")
#elif defined(__GNUC__)
#define PL_UNROLL_LOOP _Pragma("GCC unroll 8")
#elif defined(__clang__)
#define PL_UNROLL_LOOP _Pragma("unroll(8)")
#else
#define PL_UNROLL_LOOP
#endif

// Define force inline
#if defined(__GNUC__) || defined(__clang__)
#if NDEBUG
#define PL_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define PL_FORCE_INLINE
#endif
#elif defined(_MSC_VER)
#if NDEBUG
#define PL_FORCE_INLINE __forceinline
#else
#define PL_FORCE_INLINE
#endif
#else
#if NDEBUG
#define PL_FORCE_INLINE inline
#else
#define PL_FORCE_INLINE
#endif
#endif
