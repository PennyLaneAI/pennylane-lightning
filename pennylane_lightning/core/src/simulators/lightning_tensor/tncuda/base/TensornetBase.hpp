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
 * @file TensornetBase.hpp
 * Base class for all cuTensorNet backends.
 */

#pragma once

#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include "Error.hpp"
#include "cuda_helpers.hpp"
#include "tncudaError.hpp"
#include "tncuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
///@endcond

namespace Pennylane::LightningTensor::TNCuda {
/**
 * @brief CRTP-enabled base class for cutensornet.
 *
 * @tparam PrecisionT Floating point precision.
 * @tparam Derived Derived class to instantiate using CRTP.
 */
template <class PrecisionT, class Derived> class TensornetBase {
  private:
    DevTag<int> dev_tag_;

    std::size_t numQubits_;
    std::vector<std::size_t> qubitDims_;

    cudaDataType_t typeData_;
    cutensornetComputeType_t typeCompute_;

    SharedTNCudaHandle handle_;

    cutensornetState_t quantumState_;
    cutensornetStatePurity_t purity_ =
        CUTENSORNET_STATE_PURITY_PURE; // Only supports pure tensor network
                                       // states as v24.03

  public:
    TensornetBase() = delete;

    explicit TensornetBase(const std::size_t numQubits, int device_id = 0,
                           cudaStream_t stream_id = 0)
        : dev_tag_({device_id, stream_id}), numQubits_(numQubits) {
        PL_ABORT_IF(numQubits < 2,
                    "The number of qubits should be greater than 1.");

        //Ensure device is set before creating the state
        dev_tag_.refresh();

        qubitDims_ = std::vector<std::size_t>(numQubits, std::size_t{2});

        if constexpr (std::is_same_v<PrecisionT, double>) {
            typeData_ = CUDA_C_64F;
            typeCompute_ = CUTENSORNET_COMPUTE_64F;
        } else {
            typeData_ = CUDA_C_32F;
            typeCompute_ = CUTENSORNET_COMPUTE_32F;
        }

        handle_ = make_shared_tncuda_handle();

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateState(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetStatePurity_t */ purity_,
            /* int32_t numStateModes */
            static_cast<int32_t>(getNumQubits()),
            /* const int64_t *stateModeExtents */
            reinterpret_cast<int64_t *>(getQubitDims().data()),
            /* cudaDataType_t */ typeData_,
            /* cutensornetState_t * */ &quantumState_));
    }

    ~TensornetBase() {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyState(quantumState_));
    }

    /**
     * @brief Get dimension of each qubit
     *
     * @return const std::vector<std::size_t> &
     */
    [[nodiscard]] auto getQubitDims() const
        -> const std::vector<std::size_t> & {
        return qubitDims_;
    };

    /**
     * @brief Get dimension of each qubit
     *
     * @return std::vector<std::size_t> &
     */
    [[nodiscard]] auto getQubitDims() -> std::vector<std::size_t> & {
        return qubitDims_;
    };

    /**
     * @brief Get the number of qubits of the simulated system.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getNumQubits() const -> std::size_t {
        return numQubits_;
    };

    /**
     * @brief Get the CUDA data type.
     *
     * @return cudaDataType_t
     */
    [[nodiscard]] auto getCudaDataType() const -> cudaDataType_t {
        return typeData_;
    }

    /**
     * @brief Get the cutensornet handle that the object is using.
     *
     * @return cutensornetHandle_t
     */
    [[nodiscard]] auto getTNCudaHandle() const -> cutensornetHandle_t {
        return handle_.get();
    }

    /**
     * @brief Get the quantum state pointer.
     *
     * @return cutensornetState_t
     */
    [[nodiscard]] auto getQuantumState() const -> cutensornetState_t {
        return quantumState_;
    };

    /**
     * @brief Get device and Cuda stream information (device ID and the
     * associated Cuda stream ID).
     *
     * @return DevTag
     */
    [[nodiscard]] auto getDevTag() const -> const DevTag<int> & {
        return dev_tag_;
    }
};
} // namespace Pennylane::LightningTensor::TNCuda
