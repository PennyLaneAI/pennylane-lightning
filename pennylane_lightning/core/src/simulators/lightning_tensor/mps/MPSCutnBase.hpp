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
 * @file MPSCutnBase.hpp
 * Base class for MPS cuTensorNetwork backend.
 */

#pragma once

#include <complex>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include "TensorBase.hpp"
#include "cuDeviceTensor.hpp"
#include "cuGateTensorCache.hpp"
#include "cuTensorNetError.hpp"
#include "cuTensorNet_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::Util;
} // namespace
///@endcond

namespace Pennylane::LightningTensor {

template <class Precision, class Derived> class MPSCutnBase {
  public:
    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;
    using PrecisionT = Precision;

  private:
    SharedCutnHandle handle_;
    cudaDataType_t typeData_;
    cutensornetComputeType_t typeCompute_;
    cutensornetState_t quantumState_;
    cutensornetStatePurity_t purity_ =
        CUTENSORNET_STATE_PURITY_PURE; // Only supports pure tensor network
                                       // states as v24.03

    std::vector<std::vector<int64_t>> siteExtents_;
    std::vector<int64_t *> siteExtentsPtr_;
    std::vector<void *> mpsTensorsDataPtr_;

    size_t numQubits_;
    size_t maxExtent_;
    std::vector<size_t> qubitDims_;

    DevTag<int> dev_tag_;

    std::vector<cuDeviceTensor<Precision>> d_mpsTensors_;

    std::shared_ptr<GateTensorCache<Precision>> gate_cache_;

  public:
    MPSCutnBase(size_t numQubits, size_t maxExtent,
                std::vector<size_t> qubitDims, DevTag<int> dev_tag)
        : handle_(make_shared_cutn_handle()), numQubits_(numQubits),
          maxExtent_(maxExtent), qubitDims_(qubitDims), dev_tag_(dev_tag),
          gate_cache_(
              std::make_shared<GateTensorCache<Precision>>(true, dev_tag)) {

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
            /* int32_t numStateModes */ static_cast<int32_t>(numQubits_),
            /* const int64_t *stateModeExtents */
            reinterpret_cast<int64_t *>(qubitDims_.data()),
            /* cudaDataType_t */ typeData_,
            /*  cutensornetState_t * */ &quantumState_));

        for (size_t i = 0; i < numQubits_; i++) {
            std::vector<size_t> modes;
            std::vector<size_t> siteExtents;
            if (i == 0) {
                // L
                modes = std::vector<size_t>({i, i + numQubits_});
                siteExtents = std::vector<size_t>({qubitDims[i], maxExtent_});
            } else if (i == numQubits_ - 1) {
                // R
                modes = std::vector<size_t>({i + numQubits_, i});
                siteExtents = std::vector<size_t>({qubitDims[i], maxExtent_});
            } else {
                // M
                modes = std::vector<size_t>(
                    {i + numQubits_ - 1, i, i + numQubits_});
                siteExtents =
                    std::vector<size_t>({maxExtent_, qubitDims[i], maxExtent_});
            }
            d_mpsTensors_.emplace_back(modes.size(), modes, siteExtents,
                                       dev_tag_);

            std::vector<int64_t> siteExtents_int64(siteExtents.size());

            std::transform(siteExtents.begin(), siteExtents.end(),
                           siteExtents_int64.begin(),
                           [](size_t x) { return static_cast<int64_t>(x); });

            siteExtents_.push_back(siteExtents_int64);

            siteExtentsPtr_.push_back(siteExtents_.back().data());
        }

        for (size_t i = 0; i < numQubits_; i++) {
            mpsTensorsDataPtr_.push_back(static_cast<void *>(
                d_mpsTensors_[i].getDataBuffer().getData()));
        }
    }

    virtual ~MPSCutnBase() {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyState(quantumState_));
    }

    /**
     * @brief Get the cuTensorNet handle that the object is using.
     *
     * @return cutensornetHandle_t returns the cuTensorNet handle.
     */
    auto getCutnHandle() const -> cutensornetHandle_t { return handle_.get(); }

    auto getQuantumState() -> cutensornetState_t { return quantumState_; };

    auto getGateCache() -> std::shared_ptr<GateTensorCache<Precision>> {
        return gate_cache_;
    }

    auto getDataType() -> cudaDataType_t { return typeData_; };

    auto getQubitDims() const -> std::vector<size_t> { return qubitDims_; };

    auto getMaxExtent() const -> size_t { return maxExtent_; };

    auto getNumQubits() const -> size_t { return numQubits_; };

    auto getSiteExtentsPtr() -> std::vector<int64_t *> {
        return siteExtentsPtr_;
    }

    auto getMPSTensorDataPtr() -> std::vector<void *> {
        return mpsTensorsDataPtr_;
    }

    auto getMPSTensors() const -> std::vector<cuDeviceTensor<Precision>> {
        return d_mpsTensors_;
    }

    auto getMPSTensorData() const -> const cuDeviceTensor<Precision> * {
        return d_mpsTensors_.data();
    }

    auto getMPSTensorData() -> cuDeviceTensor<Precision> * {
        return d_mpsTensors_.data();
    }

    auto getDevTag() const -> DevTag<int> { return dev_tag_; }

    auto getCtrlMap() -> const std::unordered_map<std::string, std::size_t> & {
        return ctrl_map_;
    }

    auto getHostDataCopy() -> std::vector<std::vector<ComplexT>> {
        std::vector<std::vector<ComplexT>> results;

        for (size_t i = 0; i < getNumQubits(); i++) {
            std::vector<ComplexT> results_local(d_mpsTensors_[i].getLength());
            d_mpsTensors_[i].CopyGpuDataToHost(results_local.data(),
                                               results_local.size());
            results.push_back(results_local);
        }
        return results;
    }

  private:
    const std::unordered_set<std::string> const_gates_{
        "Identity", "PauliX", "PauliY", "PauliZ", "Hadamard", "T",      "S",
        "CNOT",     "SWAP",   "CY",     "CZ",     "CSWAP",    "Toffoli"};
    const std::unordered_map<std::string, std::size_t> ctrl_map_{
        // Add mapping from function name to required wires.
        {"Identity", 0},
        {"PauliX", 0},
        {"PauliY", 0},
        {"PauliZ", 0},
        {"Hadamard", 0},
        {"T", 0},
        {"S", 0},
        {"RX", 0},
        {"RY", 0},
        {"RZ", 0},
        {"Rot", 0},
        {"PhaseShift", 0},
        {"ControlledPhaseShift", 1},
        {"CNOT", 1},
        {"SWAP", 0},
        {"CY", 1},
        {"CZ", 1},
        {"CRX", 1},
        {"CRY", 1},
        {"CRZ", 1},
        {"CRot", 1},
        {"CSWAP", 1},
        {"Toffoli", 2}};
};
} // namespace Pennylane::LightningTensor