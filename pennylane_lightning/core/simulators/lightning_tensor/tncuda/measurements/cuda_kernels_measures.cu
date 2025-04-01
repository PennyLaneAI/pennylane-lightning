// Copyright 2024 Xanadu Quantum Technologies Inc.
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
 * @file cuda_kernels_measures.cu
 */
#include <cuComplex.h>

#include "cuError.hpp"
#include "cuda_helpers.hpp"

namespace Pennylane::LightningTensor::TNCuda::Measures {

/**
 * @brief The CUDA kernel that calculates the probability from a given state
 * tensor data on GPU device.
 *
 * @tparam GPUDataT cuComplex data type (cuComplex or cuDoubleComplex).
 * @tparam PrecisionT Floating data type.
 *
 * @param state Complex data pointer of state tensor on device.
 * @param probs The probability result on device.
 * @param data_size The length of state tensor on device.
 */
template <class GPUDataT, class PrecisionT>
__global__ void getProbsKernel(GPUDataT *state, PrecisionT *probs,
                               const int data_size) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < data_size) {
        PrecisionT real = state[i].x;
        PrecisionT imag = state[i].y;
        probs[i] = real * real + imag * imag;
    }
}

/**
 * @brief The CUDA kernel that normalize the probability from a given state
 * tensor data on GPU device.
 *
 * @tparam PrecisionT Floating data type.
 *
 * @param probs The probability to be normalized.
 * @param data_size The length of state tensor on device.
 * @param sum The sum of all probabilities.
 */
template <class PrecisionT>
__global__ void normalizeProbsKernel(PrecisionT *probs, const int data_size,
                                     const PrecisionT sum) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < data_size) {
        probs[i] /= sum;
    }
}

/**
 * @brief The CUDA kernel call wrapper.
 *
 * @tparam GPUDataT cuComplex data type (cuComplex or cuDoubleComplex).
 * @tparam PrecisionT Floating data type.
 *
 * @param state Complex data pointer of state tensor on device.
 * @param probs The probability result on device.
 * @param data_size The length of state tensor on device.
 * @param thread_per_block Number of threads set per block.
 * @param stream_id Stream id of CUDA calls
 */
template <class GPUDataT, class PrecisionT>
void getProbs_CUDA_call(GPUDataT *state, PrecisionT *probs, const int data_size,
                        std::size_t thread_per_block, cudaStream_t stream_id) {
    auto dv = std::div(data_size, thread_per_block);
    const std::size_t num_blocks = dv.quot + (dv.rem == 0 ? 0 : 1);
    const std::size_t block_per_grid = (num_blocks == 0 ? 1 : num_blocks);
    dim3 blockSize(thread_per_block, 1, 1);
    dim3 gridSize(block_per_grid, 1);

    getProbsKernel<GPUDataT, PrecisionT>
        <<<gridSize, blockSize, 0, stream_id>>>(state, probs, data_size);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}

/**
 * @brief The CUDA kernel call wrapper.
 *
 * @tparam PrecisionT Floating data type.
 *
 * @param probs The probability to be normalized.
 * @param data_size The length of state tensor on device.
 * @param thread_per_block Number of threads set per block.
 * @param stream_id Stream id of CUDA calls
 */
template <class PrecisionT>
void normalizeProbs_CUDA_call(PrecisionT *probs, const int data_size,
                              const PrecisionT sum,
                              std::size_t thread_per_block,
                              cudaStream_t stream_id) {
    auto dv = std::div(data_size, thread_per_block);
    const std::size_t num_blocks = dv.quot + (dv.rem == 0 ? 0 : 1);
    const std::size_t block_per_grid = (num_blocks == 0 ? 1 : num_blocks);
    dim3 blockSize(thread_per_block, 1, 1);
    dim3 gridSize(block_per_grid, 1);

    normalizeProbsKernel<PrecisionT>
        <<<gridSize, blockSize, 0, stream_id>>>(probs, data_size, sum);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}

// Definitions
/**
 * @brief Explicitly get the probability of given state tensor data on GPU
 * device.
 *
 * @param state Complex data pointer of state tensor on device.
 * @param probs The probability result on device.
 * @param data_size The length of state tensor on device.
 * @param thread_per_block Number of threads set per block.
 * @param stream_id Stream id of CUDA calls
 */
void getProbs_CUDA(cuComplex *state, float *probs, const int data_size,
                   const std::size_t thread_per_block, cudaStream_t stream_id) {
    getProbs_CUDA_call<cuComplex, float>(state, probs, data_size,
                                         thread_per_block, stream_id);
}

void getProbs_CUDA(cuDoubleComplex *state, double *probs, const int data_size,
                   const std::size_t thread_per_block, cudaStream_t stream_id) {
    getProbs_CUDA_call<cuDoubleComplex, double>(state, probs, data_size,
                                                thread_per_block, stream_id);
}

/**
 * @brief Explicitly get the probability of given state tensor data on GPU
 * device.
 *
 * @param probs The probability to be normalized.
 * @param data_size The length of state tensor on device.
 * @param thread_per_block Number of threads set per block.
 * @param stream_id Stream id of CUDA calls
 */
void normalizeProbs_CUDA(float *probs, const int data_size, const float sum,
                         const std::size_t thread_per_block,
                         cudaStream_t stream_id) {
    normalizeProbs_CUDA_call<float>(probs, data_size, sum, thread_per_block,
                                    stream_id);
}

void normalizeProbs_CUDA(double *probs, const int data_size, const double sum,
                         const std::size_t thread_per_block,
                         cudaStream_t stream_id) {
    normalizeProbs_CUDA_call<double>(probs, data_size, sum, thread_per_block,
                                     stream_id);
}
} // namespace Pennylane::LightningTensor::TNCuda::Measures
