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
 * @file MPSCutnBase.hpp
 * Base class for MPS cuTensorNetwork backend.
 */

#pragma once

#include <complex>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

// TODO to remove iostream
#include <iostream>

#include <cuda.h>
#include <cutensornet.h>

#include "CudaTensor.hpp"
#include "MPSBase.hpp"
#include "TensorBase.hpp"
#include "cutnError.hpp"
#include "cutn_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::MPS;
using namespace Pennylane::LightningTensor::Cutn;
using namespace Pennylane::LightningTensor::Cutn::Util;
} // namespace
///@endcond

namespace Pennylane::LightningTensor::MPS::Cutn {

template <class Precision, class Derived>
class MPSCutnBase : public MPSBase<Precision, Derived> {
  public:
    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;
    using PrecisionT = Precision;

  private:
    using BaseType = MPSBase<Precision, Derived>;

  private:
    SharedCutnHandle handle_;
    cudaDataType_t typeData_;
    DevTag<int> dev_tag_;
    cutensornetComputeType_t typeCompute_;
    cutensornetState_t quantumState_;
    cutensornetStatePurity_t purity_ =
        CUTENSORNET_STATE_PURITY_PURE; // Only supports pure tensor network
                                       // states as v24.03

    std::vector<std::vector<int64_t>> sitesExtents_int64_;
    std::vector<int64_t *> sitesExtentsPtr_int64_;
    std::vector<void *> tensorsDataPtr_;
    std::vector<CudaTensor<Precision>> tensors_;

  public:
    MPSCutnBase(size_t numQubits, size_t maxExtent,
                std::vector<size_t> qubitDims, DevTag<int> dev_tag)
        : BaseType(numQubits, maxExtent, qubitDims),
          handle_(make_shared_cutn_handle()), dev_tag_(dev_tag) {
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

        for (size_t i = 0; i < BaseType::getNumQubits(); i++) {
            // Convert datatype of sitesExtents to int64 as required by
            // cutensornet backend
            std::vector<int64_t> siteExtents_int64(
                BaseType::getSitesExtents()[i].size());
            std::transform(BaseType::getSitesExtents()[i].begin(),
                           BaseType::getSitesExtents()[i].end(),
                           siteExtents_int64.begin(),
                           [](size_t x) { return static_cast<int64_t>(x); });

            sitesExtents_int64_.push_back(siteExtents_int64);
            sitesExtentsPtr_int64_.push_back(sitesExtents_int64_.back().data());

            // construct mps tensors reprensentation
            tensors_.emplace_back(BaseType::getIthSiteModes(i).size(),
                                  BaseType::getIthSiteModes(i),
                                  BaseType::getIthSiteExtents(i), dev_tag_);

            tensorsDataPtr_.push_back(
                static_cast<void *>(tensors_[i].getDataBuffer().getData()));
        }
    }

    virtual ~MPSCutnBase() {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyState(quantumState_));
    }

    /**
     * @brief Get the cutensornet handle that the object is using.
     *
     * @return cutensornetHandle_t returns the cutensornet handle.
     */
    [[nodiscard]] auto getCutnHandle() const -> cutensornetHandle_t {
        return handle_.get();
    }

    /**
     * @brief Get the quantum state pointer.
     *
     * @return cutensornetState_t returns pointer to quantum state.
     */
    [[nodiscard]] auto getQuantumState() -> cutensornetState_t {
        return quantumState_;
    };

    /**
     * @brief Get Cuda data type.
     *
     * @return Cuda data type
     */
    [[nodiscard]] auto getDataType() -> cudaDataType_t { return typeData_; };

    /**
     * @brief Get device and Cuda stream information (device ID and the
     * associated Cuda stream ID).
     *
     * @return dev_tag_ DevTag object that contains both device and Cuda steam
     * ID.
     */
    [[nodiscard]] auto getDevTag() -> DevTag<int> { return dev_tag_; }

    /**
     * @brief Get a vector of pointers to extents of each site
     *
     * @return sitesExtentsPtr_int64_ std::vector<int64_t *> Note int64_t is
     * required by cutensornet backend.
     */
    [[nodiscard]] auto getSitesExtentsPtr() -> std::vector<int64_t *> {
        return sitesExtentsPtr_int64_;
    }

    /**
     * @brief Get reference to the tensor of ith site
     *
     * @return tensors_[index] CudaTensor<Precision> &.
     */
    [[nodiscard]] auto getIthSiteTensor(size_t index)
        -> CudaTensor<Precision> & {
        PL_ABORT_IF(index >= BaseType::getNumQubits(),
                    "The site index value should be less than qubit number.")
        return tensors_[index];
    }

    /**
     * @brief Get a vector of pointers to tensor data of each site
     *
     * @return tensorsDataPtr_ std::vector<void *> Note void is required by
     * cutensornet backend.
     */
    [[nodiscard]] auto getTensorsDataPtr() -> std::vector<void *> {
        return tensorsDataPtr_;
    }
};
} // namespace Pennylane::LightningTensor::MPS::Cutn
