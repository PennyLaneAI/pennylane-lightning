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
 * @file CudaTensor.hpp
 * CUDA-capable tensor class for cutensornet backends.
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

namespace Pennylane::LightningTensor::Cutn {

/**
 * @brief CRTP-enabled class for CUDA-capable Tensor.
 *
 * @tparam Precision Floating point precision.
 */

template <class PrecisionT>
class CudaTensor final : public TensorBase<PrecisionT, CudaTensor<PrecisionT>> {
  public:
    using BaseType = TensorBase<PrecisionT, CudaTensor>;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));

    CudaTensor(const size_t rank, const std::vector<size_t> &modes,
               const std::vector<size_t> &extents, const DevTag<int> &dev_tag,
               bool device_alloc = true)
        : TensorBase<PrecisionT, CudaTensor<PrecisionT>>(rank, modes, extents),
          data_buffer_{std::make_shared<DataBuffer<CFP_t>>(
              BaseType::getLength(), dev_tag, device_alloc)} {}

    CudaTensor() = delete;

    ~CudaTensor() = default;

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

    DataBuffer<CFP_t> &getDataBuffer() { return *data_buffer_; }

  private:
    std::shared_ptr<DataBuffer<CFP_t>> data_buffer_;
};
} // namespace Pennylane::LightningTensor::Cutn
