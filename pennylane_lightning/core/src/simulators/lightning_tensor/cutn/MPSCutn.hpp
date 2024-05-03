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
 * MPS class with cuTensorNet backend.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include "CudaTensor.hpp"
#include "CutnBase.hpp"
#include "DataBuffer.hpp"
#include "DevTag.hpp"
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
/// @endcond

namespace Pennylane::LightningTensor::Cutn {

/**
 * @brief Managed memory MPS class using cutensornet high-level APIs
 * backed.
 *
 * @tparam Precision Floating-point precision type.
 */

template <class Precision>
class MPSCutn final : public CutnBase<Precision, MPSCutn<Precision>> {
  private:
    using BaseType = CutnBase<Precision, MPSCutn>;

    const std::size_t maxBondDim_;

    const std::vector<std::vector<std::size_t>> sitesModes_;
    const std::vector<std::vector<std::size_t>> sitesExtents_;
    const std::vector<std::vector<int64_t>> sitesExtents_int64_;

    std::vector<CudaTensor<Precision>> tensors_;

  public:
    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;

  public:
    MPSCutn() = delete;

    explicit MPSCutn(const std::size_t numQubits, const std::size_t maxBondDim)
        : BaseType(numQubits), maxBondDim_(maxBondDim),
          sitesModes_(setSitesModes_()), sitesExtents_(setSitesExtents_()),
          sitesExtents_int64_(setSitesExtents_int64_()) {
        initTensors_();
    }

    explicit MPSCutn(const std::size_t numQubits, const std::size_t maxBondDim,
                     DevTag<int> &dev_tag)
        : BaseType(numQubits, dev_tag), maxBondDim_(maxBondDim),
          sitesModes_(setSitesModes_()), sitesExtents_(setSitesExtents_()),
          sitesExtents_int64_(setSitesExtents_int64_()) {
        initTensors_();
    }

    ~MPSCutn() = default;

    /**
     * @brief Get the max bond dimension.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getMaxBondDim() const -> std::size_t {
        return maxBondDim_;
    };

    /**
     * @brief Get a vector of pointers to extents of each site
     *
     * @return std::vector<int64_t const *> Note int64_t is
     * required by cutensornet backend.
     */
    [[nodiscard]] auto getSitesExtentsPtr() -> std::vector<int64_t const *> {
        std::vector<int64_t const *> sitesExtentsPtr_int64(
            BaseType::getNumQubits());
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            sitesExtentsPtr_int64[i] = sitesExtents_int64_[i].data();
        }
        return sitesExtentsPtr_int64;
    }

    /**
     * @brief Get a vector of pointers to tensor data of each site
     *
     * @return std::vector<void *> Note void is required by
     * cutensornet backend.
     */
    [[nodiscard]] auto getTensorsDataPtr() -> std::vector<uint64_t *> {
        std::vector<uint64_t *> tensorsDataPtr(BaseType::getNumQubits());
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            tensorsDataPtr[i] = reinterpret_cast<uint64_t *>(
                tensors_[i].getDataBuffer().getData());
        }
        return tensorsDataPtr;
    }

    /**
     * @brief Set a zero state
     */
    void reset() {
        const std::vector<std::size_t> zeroState(BaseType::getNumQubits(), 0);
        setBasisState(zeroState);
    }

    /**
     * @brief Set basis state
     * NOTE: This API assumes the bond vector is [1,0,0,......] and current
     * implementation only works for qubits.
     * @param basisState Vector representation of a basis state.
     */
    void setBasisState(const std::vector<std::size_t> &basisState) {
        PL_ABORT_IF(BaseType::getNumQubits() != basisState.size(),
                    "The size of a basis state should be equal to the number "
                    "of qubits.");

        bool allZeroOrOne = std::all_of(
            basisState.begin(), basisState.end(),
            [](std::size_t bitVal) { return bitVal == 0 || bitVal == 1; });

        PL_ABORT_IF(allZeroOrOne == false,
                    "Please ensure all elements of a basis state should be "
                    "either 0 or 1.");

        PL_ABORT_IF(this->MPSInitialized_ == true,
                    "setBasisState() can be called only once.");

        this->MPSInitialized_ = true;

        CFP_t value_cu =
            Pennylane::LightningGPU::Util::complexToCu<ComplexT>({1.0, 0.0});

        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            tensors_[i].getDataBuffer().zeroInit();
            std::size_t target = 0;
            std::size_t idx = BaseType::getNumQubits() - std::size_t{1} - i;

            // Rightmost site
            if (i == 0) {
                target = basisState[idx];
            } else {
                target = basisState[idx] == 0 ? 0 : maxBondDim_;
            }

            PL_CUDA_IS_SUCCESS(
                cudaMemcpy(&tensors_[i].getDataBuffer().getData()[target],
                           &value_cu, sizeof(CFP_t), cudaMemcpyHostToDevice));
        }

        updateQuantumStateMPS_(getSitesExtentsPtr().data(),
                               getTensorsDataPtr().data());
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
        PL_ABORT_IF_NOT(this->MPSFinalized_ == false,
                        "getDataVector() method to return the full state "
                        "vector can't be called "
                        "after cutensornetStateFinalizeMPS is called");

        this->MPSFinalized_ = true;

        // 1D representation
        std::vector<std::size_t> output_modes(std::size_t{1}, std::size_t{1});
        std::vector<std::size_t> output_extent(
            std::size_t{1}, std::size_t{1} << BaseType::getNumQubits());
        CudaTensor<Precision> output_tensor(output_modes.size(), output_modes,
                                            output_extent,
                                            BaseType::getDevTag());

        std::vector<void *> output_tensorPtr(
            std::size_t{1},
            static_cast<void *>(output_tensor.getDataBuffer().getData()));

        std::vector<int64_t *> output_extentsPtr;
        std::vector<int64_t> extent_int64(
            std::size_t{1},
            static_cast<int64_t>(std::size_t{1} << BaseType::getNumQubits()));
        output_extentsPtr.emplace_back(extent_int64.data());

        this->computeState(output_extentsPtr, output_tensorPtr);

        std::vector<ComplexT> results(output_extent.front());
        output_tensor.CopyGpuDataToHost(results.data(), results.size());

        return results;
    }

  private:
    /**
     * @brief Return siteModes to the member initializer
     *
     * @return const std::vector<std::vector<std::size_t>>
     */
    const std::vector<std::vector<std::size_t>> setSitesModes_() {
        std::vector<std::vector<std::size_t>> localSitesModes;
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            std::vector<std::size_t> localSiteModes;
            if (i == 0) {
                // Leftmost site (state mode, shared mode)
                localSiteModes =
                    std::vector<std::size_t>({i, i + BaseType::getNumQubits()});
            } else if (i == BaseType::getNumQubits() - 1) {
                // Rightmost site (shared mode, state mode)
                localSiteModes = std::vector<std::size_t>(
                    {i + BaseType::getNumQubits() - 1, i});
            } else {
                // Interior sites (state mode, state mode, shared mode)
                localSiteModes =
                    std::vector<std::size_t>({i + BaseType::getNumQubits() - 1,
                                              i, i + BaseType::getNumQubits()});
            }
            localSitesModes.push_back(std::move(localSiteModes));
        }
        return localSitesModes;
    }

    /**
     * @brief Return sitesExtents to the member initializer
     *
     * @return const std::vector<std::vector<std::size_t>>
     */
    const std::vector<std::vector<std::size_t>> setSitesExtents_() {
        std::vector<std::vector<std::size_t>> localSitesExtents;

        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            std::vector<std::size_t> localSiteExtents;
            if (i == 0) {
                // Leftmost site (state mode, shared mode)
                localSiteExtents = std::vector<std::size_t>(
                    {BaseType::getQubitDims()[i], maxBondDim_});
            } else if (i == BaseType::getNumQubits() - 1) {
                // Rightmost site (shared mode, state mode)
                localSiteExtents = std::vector<std::size_t>(
                    {maxBondDim_, BaseType::getQubitDims()[i]});
            } else {
                // Interior sites (state mode, state mode, shared mode)
                localSiteExtents = std::vector<std::size_t>(
                    {maxBondDim_, BaseType::getQubitDims()[i], maxBondDim_});
            }
            localSitesExtents.push_back(std::move(localSiteExtents));
        }
        return localSitesExtents;
    }

    /**
     * @brief Return siteExtents_int64 to the member initializer
     *
     * @return const std::vector<std::vector<int64_t>>
     */
    const std::vector<std::vector<int64_t>> setSitesExtents_int64_() {
        std::vector<std::vector<int64_t>> localSitesExtents_int64;

        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            // Convert datatype of sitesExtents to int64 as required by
            // cutensornet backend
            std::vector<int64_t> siteExtents_int64(sitesExtents_[i].size());
            std::transform(sitesExtents_[i].begin(), sitesExtents_[i].end(),
                           siteExtents_int64.begin(), [](std::size_t x) {
                               return static_cast<int64_t>(x);
                           });

            localSitesExtents_int64.push_back(std::move(siteExtents_int64));
        }
        return localSitesExtents_int64;
    }

    /**
     * @brief The tensors init helper function for ctor.
     */
    void initTensors_() {
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            // construct mps tensors reprensentation
            tensors_.emplace_back(sitesModes_[i].size(), sitesModes_[i],
                                  sitesExtents_[i], BaseType::getDevTag());
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
                                uint64_t **tensorsIn) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateInitializeMPS(
            /*const cutensornetHandle_t*/ BaseType::getCutnHandle(),
            /*cutensornetState_t*/ BaseType::getQuantumState(),
            /*cutensornetBoundaryCondition_t*/
            CUTENSORNET_BOUNDARY_CONDITION_OPEN,
            /*const int64_t *const*/ extentsIn,
            /*const int64_t *const*/ nullptr,
            /*void **/ reinterpret_cast<void **>(tensorsIn)));
    }
};
} // namespace Pennylane::LightningTensor::Cutn
