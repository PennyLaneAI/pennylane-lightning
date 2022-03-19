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
#else
#define PL_UNREACHABLE __assume(false)
#endif

#if (_OPENMP >= 202011)
#define PL_UNROLL_LOOP _Pragma("omp unroll(8)")
#elif defined(__GNUC__)
#define PL_UNROLL_LOOP _Pragma("GCC unroll 8")
#elif defined(__clang__)
#define PL_UNROLL_LOOP _Pragma("unroll(8)")
#else
#define PL_UNROLL_LOOP
#endif

#if defined(_OPENMP)
#define PL_USE_OMP 1
[[maybe_unused]] static constexpr bool use_openmp = true;
#else
[[maybe_unused]] static constexpr bool use_openmp = false;
#endif
