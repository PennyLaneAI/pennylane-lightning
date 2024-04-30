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
 * @file MPSCutn.hpp
 * MPS class with cutensornet backend.
 */

#pragma once

#include <complex>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include "CudaTensor.hpp"
#include "DataBuffer.hpp"
#include "DevTag.hpp"
#include "MPSCutnBase.hpp"
#include "TensorBase.hpp"
#include "cuda_helpers.hpp"
#include "cutnError.hpp"
#include "cutn_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::MPS;
using namespace Pennylane::LightningTensor::Cutn;
using namespace Pennylane::LightningTensor::Cutn::Util;

/**
 * @brief Get scratch memory size
 *
 * @return  Scratch memory size size_t
 */
std::size_t getScratchMemorySize() {
    std::size_t freeBytes{0}, totalBytes{0};
    PL_CUDA_IS_SUCCESS(cudaMemGetInfo(&freeBytes, &totalBytes));
    // Set scratchSize as half of freeBytes
    // TODO this magic number here should be optimized in the future
    std::size_t scratchSize = freeBytes / 2;
    return scratchSize;
}
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::MPS::Cutn {

/**
 * @brief Managed memory CUDA MPS class using cutensornet high-level APIs
 * backed.
 *
 * @tparam Precision Floating-point precision type.
 */

template <class Precision>
class MPSCutn : public MPSCutnBase<Precision, MPSCutn<Precision>> {
  private:
    using BaseType = MPSCutnBase<Precision, MPSCutn>;
    bool FinalMPSFactorization_flag = false;

  public:
    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;

  public:
    MPSCutn() = delete;

    explicit MPSCutn(const size_t numQubits, const size_t maxExtent,
                     const std::vector<size_t> &qubitDims, DevTag<int> &dev_tag)
        : BaseType(numQubits, maxExtent, qubitDims, dev_tag) {}

    ~MPSCutn() = default;

    /**
     * @brief Set a zero state
     */
    void reset() {
        const std::vector<size_t> zeroState(BaseType::getNumQubits(), 0);
        this->setBasisState(zeroState);
    }

    /**
     * @brief Set basis state
     * NOTE: This API assumes the bond vector is [1,0,0,......]
     * @param basisState Vector representation of a basis state.
     */
    void setBasisState(const std::vector<size_t> &basisState) {
        PL_ABORT_IF(BaseType::getNumQubits() != basisState.size(),
                    "The size of a basis state should be equal to the number "
                    "of qubits.");

        CFP_t value_cu =
            Pennylane::LightningGPU::Util::complexToCu<ComplexT>({1.0, 0.0});

        for (size_t i = 0; i < BaseType::getNumQubits(); i++) {
            BaseType::getIthSiteTensor(i).getDataBuffer().zeroInit();
            size_t target = 0;
            size_t idx = BaseType::getNumQubits() - 1 - i;

            // Rightmost site
            if (i == 0) {
                target = basisState[idx];
            } else {
                target = basisState[idx] == 0 ? 0 : BaseType::getMaxExtent();
            }

            PL_CUDA_IS_SUCCESS(cudaMemcpy(&BaseType::getIthSiteTensor(i)
                                               .getDataBuffer()
                                               .getData()[target],
                                          &value_cu, sizeof(CFP_t),
                                          cudaMemcpyHostToDevice));
        }

        updateQuantumState_(BaseType::getSitesExtentsPtr().data(),
                            BaseType::getTensorsDataPtr().data());
    };

    /**
     * @brief Get the full state vector representation of MPS quantum state.
     *
     * NOTE: This API is only for MPS unit tests purpose only, given that the
     * full statevector system is small. Attempt to apply this method to largr
     * systems will lead to memory issue.
     *
     * @return Full state vector representation of MPS quantum state on host
     * std::vector<ComplexT>
     */
    auto getDataVector() -> std::vector<ComplexT> {
        PL_ABORT_IF_NOT(FinalMPSFactorization_flag == false,
                        "getDataVector() method to return the full state "
                        "vector can't be called "
                        "after cutensornetStateFinalizeMPS is called");
        // 1D representation
        std::vector<size_t> output_modes(1, size_t{1});
        std::vector<size_t> output_extent(1, size_t{1}
                                                 << BaseType::getNumQubits());
        CudaTensor<Precision> output_tensor(output_modes.size(), output_modes,
                                            output_extent,
                                            BaseType::getDevTag());

        std::vector<void *> output_tensorPtr(
            1, static_cast<void *>(output_tensor.getDataBuffer().getData()));

        std::vector<int64_t *> output_extentsPtr;
        std::vector<int64_t> extent_int64(1, size_t{1}
                                                 << BaseType::getNumQubits());
        output_extentsPtr.emplace_back(extent_int64.data());

        computeState_(output_extentsPtr, output_tensorPtr);

        std::vector<ComplexT> results(output_extent.front());
        output_tensor.CopyGpuDataToHost(results.data(), results.size());

        return results;
    }

  private:
    /**
     * @brief Get the memory size used for a work space
     *
     * @return size_t Memory size
     */
    size_t getWorkSpaceMemorySize_(cutensornetWorkspaceDescriptor_t &workDesc) {
        int64_t worksize{0};

        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceGetMemorySize(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
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
    void setWorkSpaceMemory_(cutensornetWorkspaceDescriptor_t &workDesc,
                             void *scratchPtr, int64_t &worksize) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceSetMemory(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cutensornetMemspace_t*/ CUTENSORNET_MEMSPACE_DEVICE,
            /* cutensornetWorkspaceKind_t */ CUTENSORNET_WORKSPACE_SCRATCH,
            /* void *const */ scratchPtr,
            /* int64_t */ worksize));
    }

    /**
     * @brief Update quantumState (cutensornetState_t) with data provided by a
     * user
     *
     * @param extentsIn Extents of each sites
     * @param tensorsIn Pointer to tensors provided by a user
     */
    void updateQuantumState_(const int64_t *const *extentsIn,
                             void **tensorsIn) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateInitializeMPS(
            /*const cutensornetHandle_t*/ BaseType::getCutnHandle(),
            /*cutensornetState_t*/ BaseType::getQuantumState(),
            /*cutensornetBoundaryCondition_t*/
            CUTENSORNET_BOUNDARY_CONDITION_OPEN,
            /*const int64_t *const*/ extentsIn,
            /*const int64_t *const*/ nullptr,
            /*void **/ tensorsIn));
    }

    /**
     * @brief Save quantumState information to data provided by a user
     *
     * @param extentsPtr Extents of each sites
     * @param tensorPtr Pointer to tensors provided by a user
     */
    void computeState_(std::vector<int64_t *> &extentsPtr,
                       std::vector<void *> &tensorPtr) {
        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateWorkspaceDescriptor(
            BaseType::getCutnHandle(), &workDesc));

        const std::size_t scratchSize = getScratchMemorySize();

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStatePrepare(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /*  cudaStream_t unused in v24.03*/ 0x0));

        int64_t worksize = this->getWorkSpaceMemorySize_(workDesc);

        // Ensure data is aligned by 256 bytes
        worksize += 256 - worksize % 256;

        PL_ABORT_IF(static_cast<std::size_t>(worksize) > scratchSize,
                    "Insufficient workspace size on Device!");

        const std::size_t d_scratch_length = worksize / sizeof(size_t);
        DataBuffer<size_t, int> d_scratch(d_scratch_length,
                                          BaseType::getDevTag(), true);

        this->setWorkSpaceMemory_(
            workDesc, reinterpret_cast<void *>(d_scratch.getData()), worksize);

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateCompute(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* int64_t * */ extentsPtr.data(),
            /* int64_t *stridesOut */ nullptr,
            /* void * */ tensorPtr.data(),
            /* cudaStream_t */ BaseType::getDevTag().getStreamID()));

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
    }
};
} // namespace Pennylane::LightningTensor::MPS::Cutn
