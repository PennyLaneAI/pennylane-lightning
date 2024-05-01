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
 * @file CutnBase.hpp
 * Base class for cutensornet backend.
 */

#pragma once

#include <complex>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include "CudaTensor.hpp"
#include "TensorBase.hpp"
#include "TensornetBase.hpp"
#include "cuda_helpers.hpp"
#include "cutnError.hpp"
#include "cutn_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::Cutn;
using namespace Pennylane::LightningTensor::Cutn::Util;
} // namespace
///@endcond

namespace Pennylane::LightningTensor::Cutn {

template <class Precision, class Derived>
class CutnBase : public TensornetBase<Precision, Derived> {
  protected:
    bool MPSInitialized_ = false;
    bool MPSFinalized_ = false;

  private:
    using BaseType = TensornetBase<Precision, Derived>;
    SharedCutnHandle handle_;
    cudaDataType_t typeData_;
    DevTag<int> dev_tag_;
    cutensornetComputeType_t typeCompute_;
    cutensornetState_t quantumState_;
    cutensornetStatePurity_t purity_ =
        CUTENSORNET_STATE_PURITY_PURE; // Only supports pure tensor network
                                       // states as v24.03

  public:
    CutnBase() = delete;

    CutnBase(const size_t numQubits, DevTag<int> &dev_tag)
        : BaseType(numQubits), handle_(make_shared_cutn_handle()),
          dev_tag_(dev_tag) {
        initHelper_();
    }

    CutnBase(const size_t numQubits, int device_id = 0,
             cudaStream_t stream_id = 0)
        : BaseType(numQubits), handle_(make_shared_cutn_handle()),
          dev_tag_({device_id, stream_id}) {
        initHelper_();
    }

    virtual ~CutnBase() {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyState(quantumState_));
    }

    /**
     * @brief Get the cutensornet handle that the object is using.
     *
     * @return cutensornetHandle_t
     */
    [[nodiscard]] auto getCutnHandle() const -> cutensornetHandle_t {
        return handle_.get();
    }

    /**
     * @brief Get the quantum state pointer.
     *
     * @return cutensornetState_t
     */
    [[nodiscard]] auto getQuantumState() -> cutensornetState_t {
        return quantumState_;
    };

    /**
     * @brief Get device and Cuda stream information (device ID and the
     * associated Cuda stream ID).
     *
     * @return DevTag
     */
    [[nodiscard]] auto getDevTag() -> DevTag<int> & { return dev_tag_; }

    /**
     * @brief Get the memory size used for a work space
     *
     * @return size_t
     */
    size_t getWorkSpaceMemorySize(cutensornetWorkspaceDescriptor_t &workDesc) {
        int64_t worksize{0};

        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceGetMemorySize(
            /* const cutensornetHandle_t */ getCutnHandle(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cutensornetWorksizePref_t */
            CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
            /* cutensornetMemspace_t*/ CUTENSORNET_MEMSPACE_DEVICE,
            /* cutensornetWorkspaceKind_t */ CUTENSORNET_WORKSPACE_SCRATCH,
            /*  int64_t * */ &worksize));

        return worksize;
    }

    /**
     * @brief Set the memory for a work space
     *
     * @param workDesc cutensornet work space descriptor
     * @param scratchPtr Pointer to scratch memory
     * @param worksize Memory size of a work space
     */
    void setWorkSpaceMemory(cutensornetWorkspaceDescriptor_t &workDesc,
                            void *scratchPtr, int64_t &worksize) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceSetMemory(
            /* const cutensornetHandle_t */ getCutnHandle(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cutensornetMemspace_t*/ CUTENSORNET_MEMSPACE_DEVICE,
            /* cutensornetWorkspaceKind_t */ CUTENSORNET_WORKSPACE_SCRATCH,
            /* void *const */ scratchPtr,
            /* int64_t */ worksize));
    }

    /**
     * @brief Save quantumState information to data provided by a user
     *
     * @param extentsPtr Extents of each sites
     * @param tensorPtr Pointer to tensors provided by a user
     */
    void computeState(std::vector<int64_t *> &extentsPtr,
                      std::vector<void *> &tensorPtr) {
        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetCreateWorkspaceDescriptor(getCutnHandle(), &workDesc));

        // TODO we assign half (magic number is) of free memory size to the
        // maximum memory usage.
        const std::size_t scratchSize = cuUtil::getFreeMemorySize() / 2;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStatePrepare(
            /* const cutensornetHandle_t */ getCutnHandle(),
            /* cutensornetState_t */ getQuantumState(),
            /* size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /*  cudaStream_t unused in v24.03*/ 0x0));

        int64_t worksize = getWorkSpaceMemorySize(workDesc);

        // Ensure data is aligned by 256 bytes
        worksize += int64_t{256} - worksize % int64_t{256};

        PL_ABORT_IF(static_cast<std::size_t>(worksize) > scratchSize,
                    "Insufficient workspace size on Device!");

        const std::size_t d_scratch_length = worksize / sizeof(size_t);
        DataBuffer<size_t, int> d_scratch(d_scratch_length, getDevTag(), true);

        setWorkSpaceMemory(
            workDesc, reinterpret_cast<void *>(d_scratch.getData()), worksize);

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateCompute(
            /* const cutensornetHandle_t */ getCutnHandle(),
            /* cutensornetState_t */ getQuantumState(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* int64_t * */ extentsPtr.data(),
            /* int64_t *stridesOut */ nullptr,
            /* void * */ tensorPtr.data(),
            /* cudaStream_t */ getDevTag().getStreamID()));

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
    }

  private:
    /**
     * @brief helper function for constuctor.
     */
    void initHelper_() {
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
};
} // namespace Pennylane::LightningTensor::Cutn
