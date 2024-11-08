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
 * @file initSV.cu
 */
#include "cuError.hpp"
#include <cuComplex.h>

#include "cuda_helpers.hpp"
namespace {
using Pennylane::LightningGPU::Util::Cmul;
using Pennylane::LightningGPU::Util::Conj;
} // namespace

namespace Pennylane::LightningGPU {

/**
 * @brief Explicitly set state vector data on GPU device from the input values
 * (on device) and their corresponding indices (on device) information.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param num_indices Number of elements of the value array.
 * @param value Complex data pointer of input values (on device).
 * @param indices Integer data pointer of the indices (on device) of sv elements
 * to be set with corresponding elements in values.
 * @param thread_per_block Number of threads set per block.
 * @param stream_id Stream id of CUDA calls
 */
void setStateVector_CUDA(cuComplex *sv, int &num_indices, cuComplex *value,
                         int *indices, std::size_t thread_per_block,
                         cudaStream_t stream_id);
void setStateVector_CUDA(cuDoubleComplex *sv, long &num_indices,
                         cuDoubleComplex *value, long *indices,
                         std::size_t thread_per_block, cudaStream_t stream_id);

/**
 * @brief Explicitly set basis state data on GPU device from the input values
 * (on device) and their corresponding indices (on device) information.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param value Complex data of the input value.
 * @param index Integer data of the sv index to be set with the value.
 * @param async Use an asynchronous memory copy.
 * @param stream_id Stream id of CUDA calls
 */
void setBasisState_CUDA(cuComplex *sv, cuComplex &value,
                        const std::size_t index, bool async,
                        cudaStream_t stream_id);
void setBasisState_CUDA(cuDoubleComplex *sv, cuDoubleComplex &value,
                        const std::size_t index, bool async,
                        cudaStream_t stream_id);

/**
 * @brief The CUDA kernel that sets state vector data on GPU device from the
 * input values (on device) and their corresponding indices (on device)
 * information.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param num_indices Number of elements of the value array.
 * @param value Complex data pointer of input values (on device).
 * @param indices Integer data pointer of the sv indices (on device) to be set
 * with corresponding elements in values.
 */
template <class GPUDataT, class index_type>
__global__ void setStateVectorkernel(GPUDataT *sv, index_type num_indices,
                                     GPUDataT *value, index_type *indices) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices) {
        sv[indices[i]] = value[i];
    }
}

/**
 * @brief The CUDA kernel call wrapper.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param num_indices Number of elements of the value array.
 * @param value Complex data pointer of input values (on device).
 * @param indices Integer data pointer of the sv indices (on device) to be set
 * by corresponding elements in values.
 * @param thread_per_block Number of threads set per block.
 * @param stream_id Stream id of CUDA calls
 */
template <class GPUDataT, class index_type>
void setStateVector_CUDA_call(GPUDataT *sv, index_type &num_indices,
                              GPUDataT *value, index_type *indices,
                              std::size_t thread_per_block,
                              cudaStream_t stream_id) {
    auto dv = std::div(num_indices, thread_per_block);
    std::size_t num_blocks = dv.quot + (dv.rem == 0 ? 0 : 1);
    const std::size_t block_per_grid = (num_blocks == 0 ? 1 : num_blocks);
    dim3 blockSize(thread_per_block, 1, 1);
    dim3 gridSize(block_per_grid, 1);

    setStateVectorkernel<GPUDataT, index_type>
        <<<gridSize, blockSize, 0, stream_id>>>(sv, num_indices, value,
                                                indices);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}

/**
 * @brief CUDA runtime API call wrapper.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param value Complex data of the input value.
 * @param index Integer data of the sv index to be set by value.
 * @param async Use an asynchronous memory copy.
 * @param stream_id Stream id of CUDA calls
 */
template <class GPUDataT>
void setBasisState_CUDA_call(GPUDataT *sv, GPUDataT &value,
                             const std::size_t index, bool async,
                             cudaStream_t stream_id) {
    if (!async) {
        PL_CUDA_IS_SUCCESS(cudaMemcpy(&sv[index], &value, sizeof(GPUDataT),
                                      cudaMemcpyHostToDevice));
    } else {
        PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(&sv[index], &value, sizeof(GPUDataT),
                                           cudaMemcpyHostToDevice, stream_id));
    }
}

// Definitions
void setStateVector_CUDA(cuComplex *sv, int &num_indices, cuComplex *value,
                         int *indices, std::size_t thread_per_block,
                         cudaStream_t stream_id) {
    setStateVector_CUDA_call(sv, num_indices, value, indices, thread_per_block,
                             stream_id);
}
void setStateVector_CUDA(cuDoubleComplex *sv, long &num_indices,
                         cuDoubleComplex *value, long *indices,
                         std::size_t thread_per_block, cudaStream_t stream_id) {
    setStateVector_CUDA_call(sv, num_indices, value, indices, thread_per_block,
                             stream_id);
}

void setBasisState_CUDA(cuComplex *sv, cuComplex &value,
                        const std::size_t index, bool async,
                        cudaStream_t stream_id) {
    setBasisState_CUDA_call(sv, value, index, async, stream_id);
}
void setBasisState_CUDA(cuDoubleComplex *sv, cuDoubleComplex &value,
                        const std::size_t index, bool async,
                        cudaStream_t stream_id) {
    setBasisState_CUDA_call(sv, value, index, async, stream_id);
}
} // namespace Pennylane::LightningGPU
