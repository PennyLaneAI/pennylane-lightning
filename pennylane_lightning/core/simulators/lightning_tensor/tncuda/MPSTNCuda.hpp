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
 * @file MPSTNCuda.hpp
 * MPS class with cuTensorNet backend. Note that current implementation only
 * support the open boundary condition.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include "DataBuffer.hpp"
#include "DevTag.hpp"
#include "MPOTNCuda.hpp"
#include "TNCuda.hpp"
#include "TNCudaBase.hpp"
#include "TensorCuda.hpp"
#include "Util.hpp"
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
/// @endcond

namespace Pennylane::LightningTensor::TNCuda {

/**
 * @brief Managed memory MPS class using cutensornet high-level APIs
 * backed.
 *
 * @tparam Precision Floating-point precision type.
 */

template <class Precision>
class MPSTNCuda final : public TNCuda<Precision, MPSTNCuda<Precision>> {
  private:
    using BaseType = TNCuda<Precision, MPSTNCuda>;

    MPSStatus MPSInitialized_ = MPSStatus::MPSInitNotSet;

    std::vector<std::shared_ptr<MPOTNCuda<Precision>>> mpos_;
    std::vector<std::size_t> mpo_ids_;

  public:
    constexpr static auto method = "mps";

    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;
    using PrecisionT = Precision;

  public:
    MPSTNCuda() = delete;

    explicit MPSTNCuda(const std::size_t numQubits,
                       const std::size_t maxBondDim)
        : BaseType(numQubits, maxBondDim) {}

    explicit MPSTNCuda(const std::size_t numQubits,
                       const std::size_t maxBondDim, DevTag<int> dev_tag)
        : BaseType(numQubits, dev_tag, maxBondDim) {}

    ~MPSTNCuda() = default;

    /**
     * @brief Get the max bond dimension.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getMaxBondDim() const -> std::size_t {
        return BaseType::maxBondDim_;
    };

    /**
     * @brief Get the bond dimensions.
     *
     * @return std::vector<std::size_t>
     */
    [[nodiscard]] auto getBondDims(std::size_t idx) const -> std::size_t {
        return BaseType::bondDims_[idx];
    }

    /**
     * @brief Apply an MPO operator with the gate's MPO decomposition data
     * provided by the user to the compute graph.
     *
     * This API only works for the MPS backend.
     *
     * @param tensors The MPO representation of a gate. Each element in the
     * outer vector represents a MPO tensor site.
     * @param wires The wire indices of the gate acts on. The size of this
     * vector should match the size of the `tensors` vector.
     * @param max_mpo_bond_dim The maximum bond dimension of the MPO operator.
     */
    void applyMPOOperation(const std::vector<std::vector<ComplexT>> &tensors,
                           const std::vector<std::size_t> &wires,
                           const std::size_t max_mpo_bond_dim) {
        PL_ABORT_IF_NOT(
            tensors.size() == wires.size(),
            "The number of tensors should be equal to the number of "
            "wires.");

        // Create a MPO object based on the host data from the user
        mpos_.emplace_back(std::make_shared<MPOTNCuda<Precision>>(
            tensors, wires, max_mpo_bond_dim, BaseType::getNumQubits(),
            BaseType::getTNCudaHandle(), BaseType::getCudaDataType(),
            BaseType::getDevTag()));

        // Append the MPO operator to the compute graph
        int64_t operatorId;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateApplyNetworkOperator(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* cutensornetNetworkOperator_t */ mpos_.back()->getMPOOperator(),
            /* const int32_t immutable */ 1,
            /* const int32_t adjoint */ 0,
            /* const int32_t unitary */ 1,
            /* int64_t * operatorId*/ &operatorId));

        mpo_ids_.push_back(static_cast<std::size_t>(operatorId));
    }

    /**
     * @brief Append MPS final state to the quantum circuit.
     *
     * @param cutoff Cutoff value for SVD decomposition. Default is 0.
     * @param cutoff_mode Cutoff mode for SVD decomposition. Default is "abs".
     */
    void append_mps_final_state(double cutoff = 0,
                                std::string cutoff_mode = "abs") {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateFinalizeMPS(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* cutensornetBoundaryCondition_t */
            CUTENSORNET_BOUNDARY_CONDITION_OPEN,
            /* const int64_t *const extentsOut[] */
            BaseType::getSitesExtentsPtr().data(),
            /*strides=*/nullptr));

        // Optional: SVD
        cutensornetTensorSVDAlgo_t algo =
            CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ; // default option

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateConfigure(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* cutensornetStateAttributes_t */
            CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO,
            /* const void * */ &algo,
            /* std::size_t */ sizeof(algo)));

        PL_ABORT_IF_NOT(cutoff_mode == "rel" || cutoff_mode == "abs",
                        "cutoff_mode should either 'rel' or 'abs'.");

        cutensornetStateAttributes_t svd_cutoff_mode =
            (cutoff_mode == "abs")
                ? CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF
                : CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateConfigure(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* cutensornetStateAttributes_t */ svd_cutoff_mode,
            /* const void * */ &cutoff,
            /* std::size_t */ sizeof(cutoff)));

        // MPO configurations
        // Note that CUTENSORNET_STATE_MPO_APPLICATION_INEXACT is applied if the
        // `cutoff` value is not set to 0 for the MPO application.
        cutensornetStateMPOApplication_t mpo_attribute =
            (cutoff == 0) ? CUTENSORNET_STATE_MPO_APPLICATION_EXACT
                          : CUTENSORNET_STATE_MPO_APPLICATION_INEXACT;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateConfigure(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* cutensornetStateAttributes_t */
            CUTENSORNET_STATE_CONFIG_MPS_MPO_APPLICATION,
            /* const void * */ &mpo_attribute,
            /* std::size_t */ sizeof(mpo_attribute)));

        BaseType::computeState(
            const_cast<int64_t **>(BaseType::getSitesExtentsPtr().data()),
            reinterpret_cast<void **>(BaseType::getTensorsOutDataPtr().data()));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateCaptureMPS(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState()));
    }
};
} // namespace Pennylane::LightningTensor::TNCuda
