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
/**
 * @file PCPhase.hpp
 */
#pragma once

#include <cstddef>

#include <cuComplex.h>
#include <cuda.h>

namespace Pennylane::LightningGPU {

void applyPCPhase_CUDA(cuComplex *sv, std::size_t sv_length, const int *tgts,
                       std::size_t num_tgts, std::size_t dimension,
                       float phase, std::size_t thread_per_block,
                       int device_id, cudaStream_t stream_id);
void applyPCPhase_CUDA(cuDoubleComplex *sv, std::size_t sv_length,
                       const int *tgts, std::size_t num_tgts,
                       std::size_t dimension, double phase,
                       std::size_t thread_per_block, int device_id,
                       cudaStream_t stream_id);

void applyControlledPCPhase_CUDA(cuComplex *sv, std::size_t sv_length,
                                 const int *ctrls, const int *ctrl_values,
                                 std::size_t num_ctrls, const int *tgts,
                                 std::size_t num_tgts, std::size_t dimension,
                                 float phase, std::size_t thread_per_block,
                                 int device_id, cudaStream_t stream_id);
void applyControlledPCPhase_CUDA(cuDoubleComplex *sv, std::size_t sv_length,
                                 const int *ctrls, const int *ctrl_values,
                                 std::size_t num_ctrls, const int *tgts,
                                 std::size_t num_tgts, std::size_t dimension,
                                 double phase, std::size_t thread_per_block,
                                 int device_id, cudaStream_t stream_id);

} // namespace Pennylane::LightningGPU
