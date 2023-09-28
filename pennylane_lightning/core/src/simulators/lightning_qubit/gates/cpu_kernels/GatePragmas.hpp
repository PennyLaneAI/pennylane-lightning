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
 * @file GatePragmas.hpp
 * Defines macros for enabling various OpenMP options in the gate kernel
 * definitions.
 */
#pragma once

namespace Pennylane::LightningQubit::Gates::Pragmas {

// Utility pragma to ensure "#" can be generated in macro output
#define M_Hash #

// Utility to ensure macro names are processable when back-to-back
#define F(x) x

// Defines utility macros to annotate gate-kernel loops with OpenMP parallel-for
// and OpenMP SIMD pragmas. Selectable at compile time.
#ifdef PL_LQ_KERNEL_OMP
#define PL_LOOP_PARALLEL(N) F(M_Hash)pragma omp parallel for collapse(N)
#define PL_LOOP_SIMD F(M_Hash) pragma omp simd
#else
#define PL_LOOP_PARALLEL(N)
#define PL_LOOP_SIMD
#endif

}; // namespace Pennylane::LightningQubit::Gates::Pragmas