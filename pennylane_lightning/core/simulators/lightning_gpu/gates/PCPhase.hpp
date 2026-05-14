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
#include <driver_types.h>
#include <vector>

#include <cuComplex.h>
#include <cuda.h>

namespace Pennylane::LightningGPU {

template <class GPUDataT, class PrecisionT>
void applyPCPhase_CUDA(GPUDataT *sv, std::size_t sv_length,
                       const std::vector<int> &ctrl_wires,
                       const std::vector<int> &ctrl_values,
                       const std::vector<int> &tgt_wires, std::size_t dimension,
                       PrecisionT phase, int device_id, cudaStream_t stream_id);

extern template void applyPCPhase_CUDA<cuComplex, float>(
    cuComplex *, std::size_t, const std::vector<int> &,
    const std::vector<int> &, const std::vector<int> &, std::size_t, float, int,
    cudaStream_t);
extern template void applyPCPhase_CUDA<cuDoubleComplex, float>(
    cuDoubleComplex *, std::size_t, const std::vector<int> &,
    const std::vector<int> &, const std::vector<int> &, std::size_t, float, int,
    cudaStream_t);

template <class GPUDataT>
void applyDiag_CUDA(GPUDataT *sv, std::size_t sv_length,
                    const std::vector<int> &ctrl_wires,
                    const std::vector<int> &ctrl_vals,
                    const std::vector<int> &tgt_wires, const GPUDataT *diag,
                    int device_id, cudaStream_t stream_id);

extern template void
applyDiag_CUDA<cuComplex>(cuComplex *, std::size_t, const std::vector<int> &,
                          const std::vector<int> &, const std::vector<int> &,
                          const cuComplex *, int, cudaStream_t);
extern template void applyDiag_CUDA<cuDoubleComplex>(
    cuDoubleComplex *, std::size_t, const std::vector<int> &,
    const std::vector<int> &, const std::vector<int> &, const cuDoubleComplex *,
    int, cudaStream_t);

} // namespace Pennylane::LightningGPU
