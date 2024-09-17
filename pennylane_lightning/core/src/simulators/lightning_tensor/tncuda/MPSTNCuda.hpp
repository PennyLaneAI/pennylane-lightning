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

    const std::size_t maxBondDim_;
    const std::vector<std::size_t> bondDims_;
    const std::vector<std::vector<std::size_t>> sitesModes_;
    const std::vector<std::vector<std::size_t>> sitesExtents_;
    const std::vector<std::vector<int64_t>> sitesExtents_int64_;

    std::vector<TensorCuda<Precision>> tensors_;

    std::vector<TensorCuda<Precision>> tensors_out_;

    std::vector<std::shared_ptr<MPOTNCuda<Precision>>> mpos_;
    std::vector<std::size_t> mpo_ids_;

  public:
    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;
    using PrecisionT = Precision;

  public:
    MPSTNCuda() = delete;

    // TODO: Add method to the constructor to allow users to select methods at
    // runtime in the C++ layer
    explicit MPSTNCuda(const std::size_t numQubits,
                       const std::size_t maxBondDim)
        : BaseType(numQubits), maxBondDim_(maxBondDim),
          bondDims_(setBondDims_()), sitesModes_(setSitesModes_()),
          sitesExtents_(setSitesExtents_()),
          sitesExtents_int64_(setSitesExtents_int64_()) {
        initTensors_();
        reset();
        appendInitialMPSState_();
    }

    // TODO: Add method to the constructor to allow users to select methods at
    // runtime in the C++ layer
    explicit MPSTNCuda(const std::size_t numQubits,
                       const std::size_t maxBondDim, DevTag<int> dev_tag)
        : BaseType(numQubits, dev_tag), maxBondDim_(maxBondDim),
          bondDims_(setBondDims_()), sitesModes_(setSitesModes_()),
          sitesExtents_(setSitesExtents_()),
          sitesExtents_int64_(setSitesExtents_int64_()) {
        initTensors_();
        reset();
        appendInitialMPSState_();
    }

    ~MPSTNCuda() = default;

    /**
     * @brief Get the max bond dimension.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getMaxBondDim() const -> std::size_t {
        return maxBondDim_;
    };

    /**
     * @brief Get a vector of pointers to extents of each site.
     *
     * @return std::vector<int64_t const *> Note int64_t const* is
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
     * @brief Get a vector of pointers to tensor data of each site.
     *
     * @return std::vector<uint64_t *>
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
     * @brief Get a vector of pointers to tensor data of each site.
     *
     * @return std::vector<CFP_t *>
     */
    [[nodiscard]] auto getTensorsOutDataPtr() -> std::vector<CFP_t *> {
        std::vector<CFP_t *> tensorsOutDataPtr(BaseType::getNumQubits());
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            tensorsOutDataPtr[i] = tensors_out_[i].getDataBuffer().getData();
        }
        return tensorsOutDataPtr;
    }

    /**
     * @brief Set current quantum state as zero state.
     */
    void reset() {
        const std::vector<std::size_t> zeroState(BaseType::getNumQubits(), 0);
        setBasisState(zeroState);
    }

    /**
     * @brief Update the ith MPS site data.
     *
     * @param site_idx Index of the MPS site.
     * @param host_data Pointer to the data on host.
     * @param host_data_size Length of the data.
     */
    void updateMPSSiteData(const std::size_t site_idx,
                           const ComplexT *host_data,
                           std::size_t host_data_size) {
        PL_ABORT_IF_NOT(
            site_idx < BaseType::getNumQubits(),
            "The site index should be less than the number of qubits.");

        const std::size_t idx = BaseType::getNumQubits() - site_idx - 1;
        PL_ABORT_IF_NOT(
            host_data_size == tensors_[idx].getDataBuffer().getLength(),
            "The length of the host data should match its copy on the device.");

        tensors_[idx].getDataBuffer().zeroInit();

        tensors_[idx].getDataBuffer().CopyHostDataToGpu(host_data,
                                                        host_data_size);
    }

    /**
     * @brief Update quantum state with a basis state.
     * NOTE: This API assumes the bond vector is a standard basis vector
     * ([1,0,0,......]) and current implementation only works for qubit systems.
     * @param basisState Vector representation of a basis state.
     */
    void setBasisState(const std::vector<std::size_t> &basisState) {
        PL_ABORT_IF(BaseType::getNumQubits() != basisState.size(),
                    "The size of a basis state should be equal to the number "
                    "of qubits.");

        bool allZeroOrOne = std::all_of(
            basisState.begin(), basisState.end(),
            [](std::size_t bitVal) { return bitVal == 0 || bitVal == 1; });

        PL_ABORT_IF_NOT(allZeroOrOne,
                        "Please ensure all elements of a basis state should be "
                        "either 0 or 1.");

        CFP_t value_cu = cuUtil::complexToCu<ComplexT>(ComplexT{1.0, 0.0});

        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            tensors_[i].getDataBuffer().zeroInit();
            std::size_t target = 0;
            std::size_t idx = BaseType::getNumQubits() - std::size_t{1} - i;

            // Rightmost site
            if (i == 0) {
                target = basisState[idx];
            } else {
                target = basisState[idx] == 0 ? 0 : bondDims_[i - 1];
            }

            PL_CUDA_IS_SUCCESS(
                cudaMemcpy(&tensors_[i].getDataBuffer().getData()[target],
                           &value_cu, sizeof(CFP_t), cudaMemcpyHostToDevice));
        }
    };

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
            getSitesExtentsPtr().data(),
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
            const_cast<int64_t **>(getSitesExtentsPtr().data()),
            reinterpret_cast<void **>(getTensorsOutDataPtr().data()));

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

    /**
     * @brief Get the full state vector representation of a MPS quantum state.
     * Note that users/developers should be responsible to ensure that there is
     * sufficient memory on the host to store the full state vector.
     *
     * @param res Pointer to the host memory to store the full state vector
     * @param res_length Length of the result vector
     */
    void getData(ComplexT *res, const std::size_t res_length) {
        PL_ABORT_IF(log2(res_length) != BaseType::getNumQubits(),
                    "The size of the result vector should be equal to the "
                    "dimension of the quantum state.");

        std::size_t avail_gpu_memory = getFreeMemorySize();

        PL_ABORT_IF(log2(avail_gpu_memory) < BaseType::getNumQubits(),
                    "State tensor size exceeds the available GPU memory!");
        BaseType::get_state_tensor(res);
    }

    /**
     * @brief Get the full state vector representation of a MPS quantum state.
     *
     *
     * @return std::vector<ComplexT> Full state vector representation of MPS
     * quantum state on host
     */
    auto getDataVector() -> std::vector<ComplexT> {
        std::size_t length = std::size_t{1} << BaseType::getNumQubits();
        std::vector<ComplexT> results(length);

        getData(results.data(), results.size());

        return results;
    }

  private:
    /**
     * @brief Return bondDims to the member initializer
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::size_t>
     */
    std::vector<std::size_t> setBondDims_() {
        std::vector<std::size_t> localBondDims(BaseType::getNumQubits() - 1,
                                               maxBondDim_);

        const std::size_t ubDim = log2(maxBondDim_);
        for (std::size_t i = 0; i < localBondDims.size(); i++) {
            const std::size_t bondDim =
                std::min(i + 1, BaseType::getNumQubits() - i - 1);

            if (bondDim <= ubDim) {
                localBondDims[i] = std::size_t{1} << bondDim;
            }
        }
        return localBondDims;
    }

    /**
     * @brief Return siteModes to the member initializer
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::vector<std::size_t>>
     */
    std::vector<std::vector<std::size_t>> setSitesModes_() {
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
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::vector<std::size_t>>
     */
    std::vector<std::vector<std::size_t>> setSitesExtents_() {
        std::vector<std::vector<std::size_t>> localSitesExtents;

        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            std::vector<std::size_t> localSiteExtents;
            if (i == 0) {
                // Leftmost site (state mode, shared mode)
                localSiteExtents = std::vector<std::size_t>(
                    {BaseType::getQubitDims()[i], bondDims_[i]});
            } else if (i == BaseType::getNumQubits() - 1) {
                // Rightmost site (shared mode, state mode)
                localSiteExtents = std::vector<std::size_t>(
                    {bondDims_[i - 1], BaseType::getQubitDims()[i]});
            } else {
                // Interior sites (state mode, state mode, shared mode)
                localSiteExtents = std::vector<std::size_t>(
                    {bondDims_[i - 1], BaseType::getQubitDims()[i],
                     bondDims_[i]});
            }
            localSitesExtents.push_back(std::move(localSiteExtents));
        }
        return localSitesExtents;
    }

    /**
     * @brief Return siteExtents_int64 to the member initializer
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::vector<int64_t>>
     */
    std::vector<std::vector<int64_t>> setSitesExtents_int64_() {
        std::vector<std::vector<int64_t>> localSitesExtents_int64;

        for (const auto &siteExtents : sitesExtents_) {
            localSitesExtents_int64.push_back(
                std::move(Pennylane::Util::cast_vector<std::size_t, int64_t>(
                    siteExtents)));
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

            tensors_out_.emplace_back(sitesModes_[i].size(), sitesModes_[i],
                                      sitesExtents_[i], BaseType::getDevTag());
        }
    }

    /**
     * @brief Append initial MPS sites to the compute graph with data provided
     * by a user
     *
     */
    void appendInitialMPSState_() {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateInitializeMPS(
            /*const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /*cutensornetState_t*/ BaseType::getQuantumState(),
            /*cutensornetBoundaryCondition_t */
            CUTENSORNET_BOUNDARY_CONDITION_OPEN,
            /*const int64_t *const* */ getSitesExtentsPtr().data(),
            /*const int64_t *const* */ nullptr,
            /*void ** */
            reinterpret_cast<void **>(getTensorsDataPtr().data())));
    }
};
} // namespace Pennylane::LightningTensor::TNCuda
