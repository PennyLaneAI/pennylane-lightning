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
 * @file TensorCuda.hpp
 * CUDA-capable tensor class for cuTensorNet backends.
 */

#pragma once

#include <complex>
#include <memory>

#include "DataBuffer.hpp"
#include "Error.hpp"
#include "TensorBase.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda {

/**
 * @brief CRTP-enabled class for CUDA-capable Tensor.
 *
 * @tparam Precision Floating point precision.
 */

template <class Precision>
class TensorCuda final : public TensorBase<Precision, TensorCuda<Precision>> {
  public:
    using BaseType = TensorBase<Precision, TensorCuda>;
    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));

    explicit TensorCuda(const std::size_t rank,
                        const std::vector<std::size_t> &modes,
                        const std::vector<std::size_t> &extents,
                        const DevTag<int> &dev_tag, bool device_alloc = true)
        : TensorBase<Precision, TensorCuda<Precision>>(rank, modes, extents),
          data_buffer_{std::make_shared<DataBuffer<CFP_t>>(
              BaseType::getLength(), dev_tag, device_alloc)} {}

    TensorCuda() = delete;

    ~TensorCuda() = default;

    /**
     * @brief Explicitly copy data from GPU device to host memory.
     *
     * @param host_tensor Complex data pointer to receive data from device.
     * @param length Number of elements to copy.
     * @param async If true, the copy is asynchronous.
     */
    inline void CopyGpuDataToHost(std::complex<Precision> *host_tensor,
                                  std::size_t length,
                                  bool async = false) const {
        PL_ABORT_IF_NOT(BaseType::getLength() == length,
                        "Sizes do not match for Host and GPU data");
        data_buffer_->CopyGpuDataToHost(host_tensor, length, async);
    }

    DataBuffer<CFP_t> &getDataBuffer() { return *data_buffer_; }

  private:
    std::shared_ptr<DataBuffer<CFP_t>> data_buffer_;
};
} // namespace Pennylane::LightningTensor::TNCuda
