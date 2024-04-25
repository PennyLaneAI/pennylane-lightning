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
 * @file cuDeviceTensor.hpp
 */

#pragma once

#include <complex>
#include <memory>

#include "DataBuffer.hpp"
#include "Error.hpp"
#include "TensorBase.hpp"
#include "cuda_helpers.hpp"

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
} // namespace

namespace Pennylane::LightningTensor {

/**
 * @brief CRTP-enabled class for CUDA-capable Tensor.
 *
 * @tparam Precision Floating point precision.
 */

template <class PrecisionT>
class cuDeviceTensor
    : public TensorBase<PrecisionT, cuDeviceTensor<PrecisionT>> {
  public:
    using BaseType = TensorBase<PrecisionT, cuDeviceTensor>;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));

    cuDeviceTensor(size_t rank, std::vector<size_t> &modes,
                   std::vector<size_t> &extents, int device_id = 0,
                   cudaStream_t stream_id = 0, bool device_alloc = true)
        : TensorBase<PrecisionT, cuDeviceTensor<PrecisionT>>(rank, modes,
                                                             extents),
          data_buffer_{std::make_shared<DataBuffer<CFP_t>>(
              BaseType::getLength(), device_id, stream_id, device_alloc)} {}

    cuDeviceTensor(size_t rank, std::vector<size_t> &modes,
                   std::vector<size_t> &extents, DevTag<int> dev_tag,
                   bool device_alloc = true)
        : TensorBase<PrecisionT, cuDeviceTensor<PrecisionT>>(rank, modes,
                                                             extents),
          data_buffer_{std::make_shared<DataBuffer<CFP_t>>(
              BaseType::getLength(), dev_tag, device_alloc)} {}

    ~cuDeviceTensor() {}

    /**
     * @brief Return a pointer to the GPU data.
     *
     * @return const CFP_t* Complex device pointer.
     */
    [[nodiscard]] auto getData() const -> const CFP_t * {
        return data_buffer_->getData();
    }
    /**
     * @brief Return a pointer to the GPU data.
     *
     * @return CFP_t* Complex device pointer.
     */
    [[nodiscard]] auto getData() -> CFP_t * { return data_buffer_->getData(); }

    /**
     * @brief Get the CUDA stream for the given object.
     *
     * @return cudaStream_t&
     */
    inline auto getStream() -> cudaStream_t {
        return data_buffer_->getStream();
    }
    /**
     * @brief Get the CUDA stream for the given object.
     *
     * @return const cudaStream_t&
     */
    inline auto getStream() const -> cudaStream_t {
        return data_buffer_->getStream();
    }

    void setStream(const cudaStream_t &s) { data_buffer_->setStream(s); }

    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
     * @param sv StateVector host data class.
     */
    inline void
    CopyHostDataToGpu(const std::vector<std::complex<PrecisionT>> &sv,
                      bool async = false) {
        PL_ABORT_IF_NOT(BaseType::getLength() == sv.size(),
                        "Sizes do not match for Host and GPU data");
        data_buffer_->CopyHostDataToGpu(sv.data(), sv.size(), async);
    }

    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
     * @param host_sv Complex data pointer to array.
     * @param length Number of complex elements.
     */
    inline void CopyGpuDataToGpuIn(const CFP_t *gpu_sv, std::size_t length,
                                   bool async = false) {
        PL_ABORT_IF_NOT(BaseType::getLength() == length,
                        "Sizes do not match for Host and GPU data");
        data_buffer_->CopyGpuDataToGpu(gpu_sv, length, async);
    }

    /**
     * @brief Explicitly copy data from host memory to GPU device.
     *
     * @param host_sv Complex data pointer to array.
     * @param length Number of complex elements.
     */
    inline void CopyHostDataToGpu(const std::complex<PrecisionT> *host_sv,
                                  std::size_t length, bool async = false) {
        PL_ABORT_IF_NOT(BaseType::getLength() == length,
                        "Sizes do not match for Host and GPU data");
        data_buffer_->CopyHostDataToGpu(
            reinterpret_cast<const CFP_t *>(host_sv), length, async);
    }

    /**
     * @brief Explicitly copy data from GPU device to host memory.
     *
     * @param sv Complex data pointer to receive data from device.
     */
    inline void CopyGpuDataToHost(std::complex<PrecisionT> *host_sv,
                                  size_t length, bool async = false) const {
        PL_ABORT_IF_NOT(BaseType::getLength() == length,
                        "Sizes do not match for Host and GPU data");
        data_buffer_->CopyGpuDataToHost(host_sv, length, async);
    }

    const DataBuffer<CFP_t> &getDataBuffer() const { return *data_buffer_; }

    DataBuffer<CFP_t> &getDataBuffer() { return *data_buffer_; }

    /**
     * @brief Move and replace DataBuffer for statevector.
     *
     * @param other Source data to copy from.
     */
    void updateData(std::unique_ptr<DataBuffer<CFP_t>> &&other) {
        data_buffer_ = std::move(other);
    }

  private:
    std::shared_ptr<DataBuffer<CFP_t>> data_buffer_;
};
} // namespace Pennylane::LightningTensor