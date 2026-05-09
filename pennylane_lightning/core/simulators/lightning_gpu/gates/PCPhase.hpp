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

template <class GPUDataT, class PrecisionT>
void applyPCPhase_CUDA(GPUDataT *sv, std::size_t sv_length, const int *ctrls,
                       const int *ctrl_values, std::size_t num_ctrls,
                       const int *tgts, std::size_t num_tgts,
                       std::size_t dimension, PrecisionT phase, int device_id,
                       cudaStream_t stream_id);

extern template void applyPCPhase_CUDA<cuComplex, float>(
    cuComplex *, std::size_t, const int *, const int *, std::size_t,
    const int *, std::size_t, std::size_t, float, int, cudaStream_t);
extern template void applyPCPhase_CUDA<cuDoubleComplex, double>(
    cuDoubleComplex *, std::size_t, const int *, const int *, std::size_t,
    const int *, std::size_t, std::size_t, double, int, cudaStream_t);

template <class GPUDataT>
void applyDiag_CUDA(GPUDataT *sv, std::size_t sv_length, const int *ctrls,
                    const int *ctrl_values, std::size_t num_ctrls,
                    const int *tgts, std::size_t num_tgts,
                    const GPUDataT *diag, int device_id,
                    cudaStream_t stream_id);

extern template void applyDiag_CUDA<cuComplex>(
    cuComplex *, std::size_t, const int *, const int *, std::size_t,
    const int *, std::size_t, const cuComplex *, int, cudaStream_t);
extern template void applyDiag_CUDA<cuDoubleComplex>(
    cuDoubleComplex *, std::size_t, const int *, const int *, std::size_t,
    const int *, std::size_t, const cuDoubleComplex *, int, cudaStream_t);

} // namespace Pennylane::LightningGPU
