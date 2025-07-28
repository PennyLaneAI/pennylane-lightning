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

#pragma once

#include "DevTag.hpp"
#include "cuError.hpp"
#include "cuda.h"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU {

/**
 * @brief Data storage class for CUDA memory. Maintains an associated stream and
 * device ID taken during time of allocation.
 *
 * @tparam GPUDataT GPU data type.
 * @tparam DevTagT Device tag index type.
 */
template <class GPUDataT, class DevTagT = int> class DataBuffer {
  public:
    /**
     * @brief Construct a new DataBuffer object
     *
     * @param length Number of elements in data buffer.
     * @param device_id Associated device ID. Must be `cudaSetDevice`
     * compatible.
     * @param stream_id Associated stream ID. Must be `cudaSetStream`
     * compatible.
     * @param alloc_memory Indicate whether to allocate the memory for the
     * buffer. Defaults to `true`
     */
    using type = GPUDataT;

    DataBuffer(std::size_t length, int device_id = 0,
               cudaStream_t stream_id = 0, bool alloc_memory = true,
               std::size_t num_batches = 1)
        : num_batches_{num_batches}, length_{length},
          dev_tag_{device_id, stream_id}, gpu_buffer_{nullptr} {
        if (alloc_memory && (length > 0)) {
            dev_tag_.refresh();
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_),
                           sizeof(GPUDataT) * length * num_batches));
        }
    }

    DataBuffer(std::size_t length, const DevTag<DevTagT> &dev,
               bool alloc_memory = true, std::size_t num_batches = 1)
        : num_batches_{num_batches}, length_{length}, dev_tag_{dev},
          gpu_buffer_{nullptr} {
        if (alloc_memory && (length > 0)) {
            dev_tag_.refresh();
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_),
                           sizeof(GPUDataT) * length * num_batches));
        }
    }

    DataBuffer(std::size_t length, DevTag<DevTagT> &&dev,
               bool alloc_memory = true, std::size_t num_batches = 1)
        : num_batches_{num_batches}, length_{length}, dev_tag_{std::move(dev)},
          gpu_buffer_{nullptr} {
        if (alloc_memory && (length > 0)) {
            dev_tag_.refresh();
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_),
                           sizeof(GPUDataT) * length * num_batches));
        }
    }

    // Buffer should never be default initialized
    DataBuffer() = delete;

    DataBuffer &operator=(const DataBuffer &other) {
        if (this != &other) {
            int local_dev_id = -1;
            PL_CUDA_IS_SUCCESS(cudaGetDevice(&local_dev_id));

            num_batches_ = other.num_batches_;
            length_ = other.length_;
            dev_tag_ =
                DevTag<DevTagT>{local_dev_id, other.dev_tag_.getStreamID()};
            dev_tag_.refresh();
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_),
                           sizeof(GPUDataT) * length_ * num_batches_));
            CopyGpuDataToGpu(other.gpu_buffer_,
                             other.length_ * other.num_batches_);
        }
        return *this;
    }

    DataBuffer &operator=(DataBuffer &&other) {
        if (this != &other) {
            int local_dev_id = -1;
            PL_CUDA_IS_SUCCESS(cudaGetDevice(&local_dev_id));
            num_batches_ = other.num_batches_;
            length_ = other.length_;
            if (local_dev_id == other.dev_tag_.getDeviceID()) {
                dev_tag_ = std::move(other.dev_tag_);
                dev_tag_.refresh();

                gpu_buffer_ = other.gpu_buffer_;
            } else {
                dev_tag_ =
                    DevTag<DevTagT>{local_dev_id, other.dev_tag_.getStreamID()};
                dev_tag_.refresh();

                PL_CUDA_IS_SUCCESS(
                    cudaMalloc(reinterpret_cast<void **>(&gpu_buffer_),
                               sizeof(GPUDataT) * length_ * num_batches_));
                CopyGpuDataToGpu(other.gpu_buffer_,
                                 other.length_ * other.num_batches_);
                PL_CUDA_IS_SUCCESS(cudaFree(other.gpu_buffer_));
                other.dev_tag_ = {};
            }
            other.num_batches_ = 0;
            other.length_ = 0;
            other.gpu_buffer_ = nullptr;
        }
        return *this;
    };

    virtual ~DataBuffer() {
        if (gpu_buffer_ != nullptr) {
            PL_CUDA_IS_SUCCESS(cudaFree(gpu_buffer_));
        }
    };

    /**
     * @brief Zero-initialize the GPU buffer.
     *
     */
    void zeroInit(std::size_t batch_index = 0) {
        PL_CUDA_IS_SUCCESS(cudaMemset(
            gpu_buffer_, 0, num_batches_ * length_ * sizeof(GPUDataT)));
    }

    auto getData(std::size_t batch_dim = 0) -> GPUDataT * {
        return gpu_buffer_ + length_ * batch_dim;
    }
    auto getData(std::size_t batch_dim = 0) const -> const GPUDataT * {
        return gpu_buffer_ + length_ * batch_dim;
        ;
    }

    auto getLength() const { return length_; }
    auto getBatchSize() const { return num_batches_; }

    /**
     * @brief Get the CUDA stream for the given object.
     *
     * @return const cudaStream_t&
     */
    inline auto getStream() const -> cudaStream_t {
        return dev_tag_.getStreamID();
    }

    inline auto getDevice() const -> int { return dev_tag_.getDeviceID(); }

    inline auto getDevTag() const -> const DevTag<DevTagT> & {
        return dev_tag_;
    }

    /**
     * @brief Copy data from another GPU memory block to here.
     *
     */
    void CopyGpuDataToGpu(const GPUDataT *gpu_in, std::size_t length,
                          bool async = false, bool batch_copy = false) {
        PL_ABORT_IF_NOT(
            getLength() == length,
            "Sizes do not match for GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (async) {
            if (batch_copy) {
                for (std::size_t b_dim = 0; b_dim < num_batches_; b_dim++) {
                    PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                        getData(b_dim), gpu_in, sizeof(GPUDataT) * getLength(),
                        cudaMemcpyDeviceToDevice, getStream()));
                }
            } else {
                PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                    getData(), gpu_in, sizeof(GPUDataT) * getLength(),
                    cudaMemcpyDeviceToDevice, getStream()));
            }
        } else {
            if (batch_copy) {
                for (std::size_t b_dim = 0; b_dim < num_batches_; b_dim++) {
                    PL_CUDA_IS_SUCCESS(cudaMemcpy(
                        getData(b_dim), gpu_in, sizeof(GPUDataT) * getLength(),
                        cudaMemcpyDeviceToDevice));
                }
            } else {
                PL_CUDA_IS_SUCCESS(cudaMemcpy(getData(), gpu_in,
                                              sizeof(GPUDataT) * getLength(),
                                              cudaMemcpyDefault));
            }
        }
    }

    /**
     * @brief Copy data from another GPU memory block to here.
     *
     */
    void CopyGpuDataToGpu(const DataBuffer &buffer, bool async = false) {
        CopyGpuDataToGpu(buffer.getData(),
                         buffer.getLength() * buffer.getBatchSize(), async);
    }

    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
     */
    template <class HostDataT = GPUDataT>
    void CopyHostDataToGpu(const HostDataT *host_in, std::size_t length,
                           bool async = false, bool batch_copy = false) {
        PL_ABORT_IF_NOT(
            (getLength() * sizeof(GPUDataT)) == (length * sizeof(HostDataT)),
            "Sizes do not match for host & GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");
        if (async) {
            if (batch_copy) {
                PL_ABORT("Currently CopyHostDataToGpu with batch_copy = true "
                         "is unsupported.");
            } else {
                PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                    getData(), host_in, sizeof(GPUDataT) * getLength(),
                    cudaMemcpyHostToDevice, getStream()));
            }

        } else {
            if (batch_copy) {
                PL_ABORT("Currently CopyHostDataToGpu with batch_copy = true "
                         "is unsupported.");
            } else {
                PL_CUDA_IS_SUCCESS(cudaMemcpy(getData(), host_in,
                                              sizeof(GPUDataT) * getLength(),
                                              cudaMemcpyDefault));
            }
        }
    }

    /**
     * @brief Explicitly copy data from host memory to GPU device with an
     * offset.
     *
     * @tparam HostDataT Host data type.
     *
     * @param host_in Host data buffer.
     * @param length Number of elements to copy.
     * @param offset Offset in the GPU buffer.
     * @param async Asynchronous copy flag.
     *
     */
    template <class HostDataT = GPUDataT>
    void CopyHostDataToGpu(const HostDataT *host_in, std::size_t length,
                           std::size_t offset, bool async = false,
                           bool batch_copy = false) {
        PL_ABORT_IF(
            (getLength() * sizeof(GPUDataT)) <
                ((offset + length) * sizeof(HostDataT)),
            "Sizes do not match for host & GPU data. Please ensure the source "
            "buffer is out of bounds of the destination buffer");

        if (async) {
            if (batch_copy) {
                PL_ABORT("Currently CopyHostDataToGpu with batch_copy = true "
                         "is unsupported.");
            } else {
                PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                    getData() + offset, host_in, sizeof(GPUDataT) * length,
                    cudaMemcpyHostToDevice, getStream()));
            }

        } else {
            if (batch_copy) {
                PL_ABORT("Currently CopyHostDataToGpu with batch_copy = true "
                         "is unsupported.");
            } else {
                PL_CUDA_IS_SUCCESS(cudaMemcpy(getData() + offset, host_in,
                                              sizeof(GPUDataT) * length,
                                              cudaMemcpyDefault));
            }
        }
    }

    /**
     * @brief Explicitly copy data from host memory to GPU device with a stride.
     *
     * @tparam HostDataT Host data type.
     *
     * @param host_in Host data buffer.
     * @param length Number of elements to copy.
     * @param stride Stride in the GPU buffer.
     * @param async Asynchronous copy flag.
     *
     */
    template <class HostDataT = GPUDataT>
    void CopyHostDataToGpuWithStride(const HostDataT *host_in,
                                     std::size_t length, std::size_t stride,
                                     bool async = false,
                                     bool batch_copy = false) {
        PL_ABORT_IF(
            (getLength() * sizeof(GPUDataT)) <
                ((stride * length) * sizeof(HostDataT)),
            "Sizes do not match for host & GPU data. Please ensure the source "
            "buffer is out of bounds of the destination buffer or the stride "
            "is too large");

        if (async) {
            if (batch_copy) {
                PL_ABORT("Currently CopyHostDataToGpu with batch_copy = true "
                         "is unsupported.");
            } else {
                PL_CUDA_IS_SUCCESS(cudaMemcpy2DAsync(
                    getData(), sizeof(GPUDataT) * stride, host_in,
                    sizeof(HostDataT), sizeof(HostDataT), length,
                    cudaMemcpyHostToDevice, getStream()));
            }

        } else {
            if (batch_copy) {
                PL_ABORT("Currently CopyHostDataToGpu with batch_copy = true "
                         "is unsupported.");
            } else {
                PL_CUDA_IS_SUCCESS(
                    cudaMemcpy2D(getData(), sizeof(GPUDataT) * stride, host_in,
                                 sizeof(HostDataT), sizeof(HostDataT), length,
                                 cudaMemcpyHostToDevice));
            }
        }
    }

    /**
     * @brief Explicitly copy data from GPU device to host memory.
     *
     */
    template <class HostDataT = GPUDataT>
    inline void CopyGpuDataToHost(HostDataT *host_out, std::size_t length,
                                  bool async = false) const {
        PL_ABORT_IF_NOT(
            (getLength() * sizeof(GPUDataT)) == (length * sizeof(HostDataT)),
            "Sizes do not match for host & GPU data. Please ensure the source "
            "buffer is not larger than the destination buffer");

        if (!async) {
            PL_CUDA_IS_SUCCESS(cudaMemcpy(host_out, getData(),
                                          sizeof(GPUDataT) * getLength(),
                                          cudaMemcpyDefault));
        } else {
            PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(
                host_out, getData(), sizeof(GPUDataT) * getLength(),
                cudaMemcpyDeviceToHost, getStream()));
        }
    }

  private:
    std::size_t num_batches_;
    std::size_t length_;
    DevTag<DevTagT> dev_tag_;
    GPUDataT *gpu_buffer_;
};
} // namespace Pennylane::LightningGPU
