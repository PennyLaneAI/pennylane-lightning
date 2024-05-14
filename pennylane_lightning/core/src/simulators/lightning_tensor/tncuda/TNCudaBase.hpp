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
 * @file TNCudaBase.hpp
 * Base class for cuTensorNet-backed tensor networks.
 */

#pragma once

#include <complex>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include "TensorBase.hpp"
#include "TensorCuda.hpp"
#include "TensornetBase.hpp"
#include "cuda_helpers.hpp"
#include "tncudaError.hpp"
#include "tncuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
///@endcond

namespace Pennylane::LightningTensor::TNCuda {

template <class Precision, class Derived>
class TNCudaBase : public TensornetBase<Precision, Derived> {
  private:
    using BaseType = TensornetBase<Precision, Derived>;
    SharedTNCudaHandle handle_;
    cudaDataType_t typeData_;
    DevTag<int> dev_tag_;
    cutensornetComputeType_t typeCompute_;
    cutensornetState_t quantumState_;
    cutensornetStatePurity_t purity_ =
        CUTENSORNET_STATE_PURITY_PURE; // Only supports pure tensor network
                                       // states as v24.03

  public:
    TNCudaBase() = delete;

    explicit TNCudaBase(const std::size_t numQubits, int device_id = 0,
                        cudaStream_t stream_id = 0)
        : BaseType(numQubits), handle_(make_shared_tncuda_handle()),
          dev_tag_({device_id, stream_id}) {
        // TODO this code block could be moved to base class and need to revisit
        // when working on copy ctor
        if constexpr (std::is_same_v<Precision, double>) {
            typeData_ = CUDA_C_64F;
            typeCompute_ = CUTENSORNET_COMPUTE_64F;
        } else {
            typeData_ = CUDA_C_32F;
            typeCompute_ = CUTENSORNET_COMPUTE_32F;
        }

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateState(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetStatePurity_t */ purity_,
            /* int32_t numStateModes */
            static_cast<int32_t>(BaseType::getNumQubits()),
            /* const int64_t *stateModeExtents */
            reinterpret_cast<int64_t *>(BaseType::getQubitDims().data()),
            /* cudaDataType_t */ typeData_,
            /*  cutensornetState_t * */ &quantumState_));
    }

    explicit TNCudaBase(const std::size_t numQubits, DevTag<int> dev_tag)
        : BaseType(numQubits), handle_(make_shared_tncuda_handle()),
          dev_tag_(dev_tag) {
        // TODO this code block could be moved to base class and need to revisit
        // when working on copy ctor
        if constexpr (std::is_same_v<Precision, double>) {
            typeData_ = CUDA_C_64F;
            typeCompute_ = CUTENSORNET_COMPUTE_64F;
        } else {
            typeData_ = CUDA_C_32F;
            typeCompute_ = CUTENSORNET_COMPUTE_32F;
        }

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateState(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetStatePurity_t */ purity_,
            /* int32_t numStateModes */
            static_cast<int32_t>(BaseType::getNumQubits()),
            /* const int64_t *stateModeExtents */
            reinterpret_cast<int64_t *>(BaseType::getQubitDims().data()),
            /* cudaDataType_t */ typeData_,
            /*  cutensornetState_t * */ &quantumState_));
    }

    ~TNCudaBase() {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyState(quantumState_));
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

  protected:
    /**
     * @brief Returns the workspace size.
     *
     * @return std::size_t
     */
    std::size_t
    getWorkSpaceMemorySize(cutensornetWorkspaceDescriptor_t &workDesc) {
        int64_t worksize{0};

        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceGetMemorySize(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cutensornetWorksizePref_t */
            CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
            /* cutensornetMemspace_t*/ CUTENSORNET_MEMSPACE_DEVICE,
            /* cutensornetWorkspaceKind_t */ CUTENSORNET_WORKSPACE_SCRATCH,
            /*  int64_t * */ &worksize));

        // Ensure data is aligned by 256 bytes
        worksize += int64_t{256} - worksize % int64_t{256};

        return static_cast<std::size_t>(worksize);
    }

    /**
     * @brief Set memory for a workspace.
     *
     * @param workDesc cutensornet work space descriptor
     * @param scratchPtr Pointer to scratch memory
     * @param worksize Memory size of a work space
     */
    void setWorkSpaceMemory(cutensornetWorkspaceDescriptor_t &workDesc,
                            void *scratchPtr, std::size_t worksize) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceSetMemory(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cutensornetMemspace_t*/ CUTENSORNET_MEMSPACE_DEVICE,
            /* cutensornetWorkspaceKind_t */ CUTENSORNET_WORKSPACE_SCRATCH,
            /* void *const */ scratchPtr,
            /* int64_t */ static_cast<int64_t>(worksize)));
    }

    /**
     * @brief Save quantumState information to data provided by a user
     *
     * @param tensorPtr Pointer to tensors provided by a user
     */
    void computeState(void **tensorPtr) {
        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetCreateWorkspaceDescriptor(getTNCudaHandle(), &workDesc));

        // TODO we assign half (magic number is) of free memory size to the
        // maximum memory usage.
        const std::size_t scratchSize = cuUtil::getFreeMemorySize() / 2;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStatePrepare(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetState_t */ getQuantumState(),
            /* size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /*  cudaStream_t unused in v24.03*/ 0x0));

        std::size_t worksize = getWorkSpaceMemorySize(workDesc);

        PL_ABORT_IF(std::size_t(worksize) > scratchSize,
                    "Insufficient workspace size on Device!");

        const std::size_t d_scratch_length = worksize / sizeof(std::size_t);
        DataBuffer<std::size_t, int> d_scratch(d_scratch_length, getDevTag(),
                                               true);

        setWorkSpaceMemory(
            workDesc, reinterpret_cast<void *>(d_scratch.getData()), worksize);

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateCompute(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetState_t */ getQuantumState(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* int64_t * */ nullptr,
            /* int64_t *stridesOut */ nullptr,
            /* void * */ tensorPtr,
            /* cudaStream_t */ getDevTag().getStreamID()));

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
    }
};
} // namespace Pennylane::LightningTensor::TNCuda
