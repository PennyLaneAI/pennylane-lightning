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
#include "TNCudaBase.hpp"
#include "TensorCuda.hpp"
#include "TensornetBase.hpp"
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

// TODO check if CRTP is required by the end of project.
template <class Precision>
class MPSTNCuda final : public TNCudaBase<Precision, MPSTNCuda<Precision>> {
  private:
    using BaseType = TNCudaBase<Precision, MPSTNCuda>;

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
    [[nodiscard]] auto getBondDims(const size_t idx) const -> std::size_t {
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

        // Create a queue of wire pairs to apply SWAP gates and MPO local target
        // wires
        const auto [local_wires, swap_wires_queue] =
            create_swap_wire_pair_queue(wires);

        // Apply SWAP gates to ensure the following MPO operator targeting at
        // local wires
        if (swap_wires_queue.size() > 0) {
            for_each(swap_wires_queue.begin(), swap_wires_queue.end(),
                     [this](const auto &swap_wires) {
                         for_each(swap_wires.begin(), swap_wires.end(),
                                  [this](const auto &wire_pair) {
                                      BaseType::applyOperation(
                                          "SWAP", wire_pair, false);
                                  });
                     });
        }

        // Create a MPO object based on the host data from the user
        mpos_.emplace_back(std::make_shared<MPOTNCuda<Precision>>(
            tensors, local_wires, max_mpo_bond_dim, BaseType::getNumQubits(),
            BaseType::getTNCudaHandle(), BaseType::getCudaDataType(),
            BaseType::getDevTag()));

        // Append the MPO operator to the compute graph
        // Note MPO operator only works for local target wires as of v24.08
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

        // Apply SWAP gates to restore the original wire order
        if (swap_wires_queue.size() > 0) {
            for_each(swap_wires_queue.rbegin(), swap_wires_queue.rend(),
                     [this](const auto &swap_wires) {
                         for_each(swap_wires.rbegin(), swap_wires.rend(),
                                  [this](const auto &wire_pair) {
                                      BaseType::applyOperation(
                                          "SWAP", wire_pair, false);
                                  });
                     });
        }
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

        // TODO: This is a dummy tensor update to allow multiple calls to the
        // `append_mps_final_state` method as well as appending additional
        // operations to the graph. This is a temporary solution and this line
        // can be removed in the future when the `cutensornet` backend allows
        // multiple calls to the `cutensornetStateFinalizeMPS` method. For more
        // details, please see the `cutensornet` high-level API workflow logic
        // [here]
        // (https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/api/functions.html#high-level-tensor-network-api).
        // In order to proceed with the following gate operations or
        // measurements after calling the `cutensornetStateCompute()` API, we
        // have to call the `cutensornetStateUpdateTensor()` API, which is
        // wrapped inside the `dummy_tensor_update()` method.
        //
        BaseType::dummy_tensor_update();
    }
};
} // namespace Pennylane::LightningTensor::TNCuda
