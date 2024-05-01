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

#include <algorithm>
#include <complex>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include <iostream>

#include "CudaTensor.hpp"
#include "DataBuffer.hpp"
#include "DevTag.hpp"
#include "CutnBase.hpp"
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

namespace Pennylane::LightningTensor::Cutn {

/**
 * @brief Managed memory CUDA MPS class using cutensornet high-level APIs
 * backed.
 *
 * @tparam Precision Floating-point precision type.
 */

template <class Precision>
class MPSCutn final : public CutnBase<Precision, MPSCutn<Precision>> {
  private:
    using BaseType = CutnBase<Precision, MPSCutn>;
    bool FinalMPSFactorization_flag = false;

    size_t maxExtent_;

    std::vector<std::vector<size_t>> sitesModes_;
    std::vector<std::vector<size_t>> sitesExtents_;

    std::vector<std::vector<int64_t>> sitesExtents_int64_;
    std::vector<int64_t *> sitesExtentsPtr_int64_;
    std::vector<void *> tensorsDataPtr_;
    std::vector<CudaTensor<Precision>> tensors_;

  public:
    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;

  public:
    MPSCutn() = delete;

    explicit MPSCutn(const size_t numQubits, const size_t maxExtent)
        : BaseType(numQubits), maxExtent_(maxExtent) {
            initHelper_();
        }

    explicit MPSCutn(const size_t numQubits, const size_t maxExtent,
                     DevTag<int> &dev_tag)
        : BaseType(numQubits, dev_tag), maxExtent_(maxExtent) {
            initHelper_();
        }

    ~MPSCutn() final = default;


    /**
     * @brief Get the max bond dimension.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getMaxExtent() const -> size_t { return maxExtent_; };

    /**
     * @brief Get modes of each sites
     *
     * @return const std::vector<std::vector<size_t>> &
     */
    [[nodiscard]] auto getSitesModes() const
        -> const std::vector<std::vector<size_t>> & {
        return sitesModes_;
    };

    /**
     * @brief Get extents of each sites
     *
     * @return const std::vector<std::vector<size_t>> &
     */
    [[nodiscard]] auto getSitesExtents() const
        -> const std::vector<std::vector<size_t>> & {
        return sitesExtents_;
    };

    /**
     * @brief Get a vector of pointers to extents of each site
     *
     * @return sitesExtentsPtr_int64_ std::vector<int64_t *> Note int64_t is
     * required by cutensornet backend.
     */
    [[nodiscard]] auto getSitesExtentsPtr() -> std::vector<int64_t *> & {
        return sitesExtentsPtr_int64_;
    }

    /**
     * @brief Get reference to the tensor of ith site
     *
     * @return std::vector<CudaTensor<Precision>> &.
     */
    [[nodiscard]] auto getSitesTensors()
        -> std::vector<CudaTensor<Precision>> & {
        return tensors_;
    }

    /**
     * @brief Get a vector of pointers to tensor data of each site
     *
     * @return tensorsDataPtr_ std::vector<void *> Note void is required by
     * cutensornet backend.
     */
    [[nodiscard]] auto getTensorsDataPtr() -> std::vector<void *> & {
        return tensorsDataPtr_;
    }


    /**
     * @brief Set a zero state
     */
    void reset() {
        const std::vector<size_t> zeroState(BaseType::getNumQubits(), 0);
        this->setBasisState(zeroState);
    }

    /**
     * @brief Set basis state
     * NOTE: This API assumes the bond vector is [1,0,0,......] and current
     * implementation only works for qubits.
     * @param basisState Vector representation of a basis state.
     */
    void setBasisState(const std::vector<size_t> &basisState) {
        PL_ABORT_IF(BaseType::getNumQubits() != basisState.size(),
                    "The size of a basis state should be equal to the number "
                    "of qubits.");

        bool allZeroOrOne = std::all_of(
            basisState.begin(), basisState.end(),
            [](size_t bitVal) { return bitVal == 0 || bitVal == 1; });

        PL_ABORT_IF(allZeroOrOne == false,
                    "Please ensure all elements of a basis state should be "
                    "either 0 or 1.");

        CFP_t value_cu =
            Pennylane::LightningGPU::Util::complexToCu<ComplexT>({1.0, 0.0});
        
        for (size_t i = 0; i < BaseType::getNumQubits(); i++) {
            tensors_[i].getDataBuffer().zeroInit();
            size_t target = 0;
            size_t idx = BaseType::getNumQubits() - size_t{1} - i;

            // Rightmost site
            if (i == 0) {
                target = basisState[idx];
            } else {
                target = basisState[idx] == 0 ? 0 : maxExtent_;
            }

            PL_CUDA_IS_SUCCESS(cudaMemcpy(&tensors_[i]
                                               .getDataBuffer()
                                               .getData()[target],
                                          &value_cu, sizeof(CFP_t),
                                          cudaMemcpyHostToDevice));
        }

        updateQuantumStateMPS_(getSitesExtentsPtr().data(), getTensorsDataPtr().data());
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
        std::vector<size_t> output_modes(size_t{1}, size_t{1});
        std::vector<size_t> output_extent(
            size_t{1}, size_t{1} << BaseType::getNumQubits());
        CudaTensor<Precision> output_tensor(output_modes.size(), output_modes,
                                            output_extent,
                                            BaseType::getDevTag());

        std::vector<void *> output_tensorPtr(
            size_t{1},
            static_cast<void *>(output_tensor.getDataBuffer().getData()));

        std::vector<int64_t *> output_extentsPtr;
        std::vector<int64_t> extent_int64(
            size_t{1},
            static_cast<int64_t>(size_t{1} << BaseType::getNumQubits()));
        output_extentsPtr.emplace_back(extent_int64.data());

        computeState_(output_extentsPtr, output_tensorPtr);

        std::vector<ComplexT> results(output_extent.front());
        output_tensor.CopyGpuDataToHost(results.data(), results.size());

        return results;
    }

  private:
    void initHelper_() {
        // Configure extents for each sites
        for (size_t i = 0; i < BaseType::getNumQubits(); i++) {
            std::vector<size_t> localSiteModes;
            std::vector<size_t> localSiteExtents;
            if (i == 0) {
                // Leftmost site (state mode, shared mode)
                localSiteModes = std::vector<size_t>({i, i + BaseType::getNumQubits()});
                localSiteExtents =
                    std::vector<size_t>({BaseType::getQubitDims()[i], maxExtent_});
            } else if (i == BaseType::getNumQubits() - 1) {
                // Rightmost site (shared mode, state mode)
                localSiteModes = std::vector<size_t>({i + BaseType::getNumQubits() - 1, i});
                localSiteExtents =
                    std::vector<size_t>({maxExtent_, BaseType::getQubitDims()[i]});
            } else {
                // Interior sites (state mode, state mode, shared mode)
                localSiteModes = std::vector<size_t>(
                    {i + BaseType::getNumQubits() - 1, i, i + BaseType::getNumQubits()});
                localSiteExtents = std::vector<size_t>(
                    {maxExtent_, BaseType::getQubitDims()[i], maxExtent_});
            }
            sitesExtents_.push_back(localSiteExtents);
            sitesModes_.push_back(localSiteModes);
        }

        for (size_t i = 0; i < BaseType::getNumQubits(); i++) {
            // Convert datatype of sitesExtents to int64 as required by
            // cutensornet backend
            std::vector<int64_t> siteExtents_int64(
                sitesExtents_[i].size());
            std::transform(sitesExtents_[i].begin(),
                           sitesExtents_[i].end(),
                           siteExtents_int64.begin(),
                           [](size_t x) { return static_cast<int64_t>(x); });

            sitesExtents_int64_.push_back(siteExtents_int64);
            sitesExtentsPtr_int64_.push_back(sitesExtents_int64_.back().data());

            // construct mps tensors reprensentation
            tensors_.emplace_back(sitesModes_[i].size(),
                                  sitesModes_[i],
                                  sitesExtents_[i], BaseType::getDevTag());

            tensorsDataPtr_.push_back(
                static_cast<void *>(tensors_[i].getDataBuffer().getData()));
        }
    }

    /**
     * @brief Update quantumState (cutensornetState_t) with data provided by a
     * user
     *
     * @param extentsIn Extents of each sites
     * @param tensorsIn Pointer to tensors provided by a user
     */
    void updateQuantumStateMPS_(const int64_t *const *extentsIn,
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

        int64_t worksize = BaseType::getWorkSpaceMemorySize(workDesc);

        // Ensure data is aligned by 256 bytes
        worksize += int64_t{256} - worksize % int64_t{256};

        PL_ABORT_IF(static_cast<std::size_t>(worksize) > scratchSize,
                    "Insufficient workspace size on Device!");

        const std::size_t d_scratch_length = worksize / sizeof(size_t);
        DataBuffer<size_t, int> d_scratch(d_scratch_length,
                                          BaseType::getDevTag(), true);

        BaseType::setWorkSpaceMemory(
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
