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

    const std::vector<std::vector<std::size_t>> sitesModes_;
    const std::vector<std::vector<std::size_t>> sitesExtents_;
    const std::vector<std::vector<int64_t>> sitesExtents_int64_;

    std::vector<TensorCuda<Precision>> tensors_;

    std::vector<TensorCuda<Precision>> tensors_out_;

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
          sitesModes_(setSitesModes_()), sitesExtents_(setSitesExtents_()),
          sitesExtents_int64_(setSitesExtents_int64_()) {
        initTensors_();
        setZeroState();
        appendInitialMPSState_();
    }

    // TODO: Add method to the constructor to allow users to select methods at
    // runtime in the C++ layer
    explicit MPSTNCuda(const std::size_t numQubits,
                       const std::size_t maxBondDim, DevTag<int> dev_tag)
        : BaseType(numQubits, dev_tag), maxBondDim_(maxBondDim),
          sitesModes_(setSitesModes_()), sitesExtents_(setSitesExtents_()),
          sitesExtents_int64_(setSitesExtents_int64_()) {
        initTensors_();
        setZeroState();
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

    void setZeroState() { reset(); }

    /**
     * @brief Set the ith MPS site.
     *
     * @param i Index of the MPS site.
     * @param data Pointer to the data on host.
     * @param length Length of the data.
     */

    void setIthMPSSite(const std::size_t i, const ComplexT *data,
                       std::size_t length) {
        PL_ABORT_IF(BaseType::getNumQubits() < i,
                    "The size of a basis state should be equal to the number "
                    "of qubits.");

        const std::size_t idx = BaseType::getNumQubits() - i - 1;
        PL_ABORT_IF(length > tensors_[idx].getDataBuffer().getLength(),
                    "The length of the data should be equal to the dimension "
                    "of the qubit.");

        tensors_[idx].getDataBuffer().zeroInit();

        PL_CUDA_IS_SUCCESS(cudaMemcpy(tensors_[idx].getDataBuffer().getData(),
                                      data, sizeof(CFP_t) * length,
                                      cudaMemcpyHostToDevice));
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

        // TODO: Refactor this part to set bondDims as a data member variable
        std::vector<std::size_t> bondDims(BaseType::getNumQubits() - 1,
                                          maxBondDim_);

        for (std::size_t i = 0; i < bondDims.size(); i++) {
            std::size_t bondDim =
                std::min(i + 1, BaseType::getNumQubits() - i - 1);
            if (bondDim > log2(maxBondDim_)) {
                bondDims[i] = maxBondDim_;
            } else {
                bondDims[i] = std::size_t{1} << bondDim;
            }
        }

        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            tensors_[i].getDataBuffer().zeroInit();
            std::size_t target = 0;
            std::size_t idx = BaseType::getNumQubits() - std::size_t{1} - i;

            // Rightmost site
            if (i == 0) {
                target = basisState[idx];
            } else {
                target = basisState[idx] == 0 ? 0 : bondDims[i - 1];
            }

            PL_CUDA_IS_SUCCESS(
                cudaMemcpy(&tensors_[i].getDataBuffer().getData()[target],
                           &value_cu, sizeof(CFP_t), cudaMemcpyHostToDevice));
        }
    };

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
            CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ; // default

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateConfigure(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* cutensornetStateAttributes_t */
            CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO,
            /* const void * */ &algo,
            /* size_t */ sizeof(algo)));

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
            /* size_t */ sizeof(cutoff)));

        BaseType::computeState(
            const_cast<int64_t **>(getSitesExtentsPtr().data()),
            reinterpret_cast<void **>(getTensorsOutDataPtr().data()));

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
        // TODO: Refactor this part to set bondDims as a data member variable
        std::vector<std::size_t> bondDims(BaseType::getNumQubits() - 1,
                                          maxBondDim_);

        for (std::size_t i = 0; i < bondDims.size(); i++) {
            std::size_t bondDim =
                std::min(i + 1, BaseType::getNumQubits() - i - 1);
            if (bondDim > log2(maxBondDim_)) {
                bondDims[i] = maxBondDim_;
            } else {
                bondDims[i] = std::size_t{1} << bondDim;
            }
        }

        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            std::vector<std::size_t> localSiteExtents;

            if (i == 0) {
                // Leftmost site (state mode, shared mode)
                localSiteExtents = std::vector<std::size_t>(
                    {BaseType::getQubitDims()[i], bondDims[i]});
            } else if (i == BaseType::getNumQubits() - 1) {
                // Rightmost site (shared mode, state mode)
                localSiteExtents = std::vector<std::size_t>(
                    {bondDims[i - 1], BaseType::getQubitDims()[i]});
            } else {
                // Interior sites (state mode, state mode, shared mode)
                localSiteExtents = std::vector<std::size_t>(
                    {bondDims[i - 1], BaseType::getQubitDims()[i],
                     bondDims[i]});
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
     * @brief Update quantumState (cutensornetState_t) with data provided by a
     * user
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
