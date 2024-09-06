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
 * @file MPOTNCuda.hpp
 * Base class for cuTensorNet-backed MPO.
 */

#pragma once

#include <algorithm>
#include <cuComplex.h>
#include <cutensornet.h>
#include <queue>
#include <variant>
#include <vector>

#include <iostream>

#include "TNCudaMPOCache.hpp"
#include "TensorCuda.hpp"
#include "cuda_helpers.hpp"
#include "tncudaError.hpp"
#include "tncuda_helpers.hpp"

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningTensor::TNCuda::MPO;
} // namespace

namespace Pennylane::LightningTensor::TNCuda {

/**
 * @brief Class for MPO tensor network in cuTensorNet backend.
 * Any gate tensor can be represented as a MPO tensor network in the context of
 MPS.
 * The gate tensor has to be decomposed respect to its target wires. If the
 target wires
 * are not adjacent, Identity tensors are inserted between the MPO tensors.
 * 1. The MPO tensors' modes order in an open boundary condition are:
   2              3              2
   |              |              |
   X--1--....--0--X--2--....--0--X
   |              |              |
   0              1              1

 * 2. The extents of the left side bound MPO tensor are [2, bondR, 2].
   The extents of the right side bound MPO tensor are [bondL, 2, 2].
   The extents of the middle MPO tensors are [bondL, 2, bondR, 2].

 * MPO tensor modes with connecting Identity tensors in an open boundary
 condition are:

   X--I--...--I--X--I--...--I--X
 * Note that extents of mode 0 and 2 of I are equal to the bond dimension of
 * the nearest MPO tensor. The shape of a connecting Identity tensor is
 *[bond, 2, bond, 2]. If the Identity tensor is flatten, its 0th and
 * (2*bind+1)th element are complex{1, 0}, and the rest are complex{0,0}.
 * Also note that the life time of the tensor data is designed to be aligned
 with
 * the life time of the tensor network it's applied to.
 * @tparam PrecisionT Floating point type.
 */
template <class PrecisionT> class MPOTNCudaOpt {
  private:
    using ComplexT = std::complex<PrecisionT>;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));

    std::vector<std::size_t> wires_; // pennylane  wires convention

    // To buuld a MPO tensor network, we need: 1. a cutensornetHandle; 2. a
    // cutensornetNetworkOperator_t object; 3. Complex coefficient associated
    // with the appended operator component. 4. Number of MPO sites; 5. MPO
    // tensor mode extents for each MPO tensor;
    // 6. Boundary conditions;
    cutensornetNetworkOperator_t MPOOperator_;
    cuDoubleComplex coeff_ =
        make_cuDoubleComplex(1.0, 0.0); // default coefficient
    cutensornetBoundaryCondition_t boundaryCondition_{
        CUTENSORNET_BOUNDARY_CONDITION_OPEN}; // open boundary condition
    int64_t componentIdx_;

    std::size_t maxBondDim_;
    std::vector<std::size_t> bondDims_;

    std::size_t numMPOSites_;
    std::vector<std::size_t> modes_;
    std::vector<int32_t> MPO_modes_int32_;

    std::vector<std::vector<int64_t>> modesExtents_int64_;
    // TODO: move tensors_ to MPSTNCuda to allow sharing tensors across
    // different MPO
    //  operators. A singleton class managing MPO tensors is a better solution.
    //  These tensors can be stored in an hashtable and be access with a
    //  std::pair key storing tensors's name and wires-order. Consider storing
    //  connecting Identity tensors in a separte singleton class. This would
    //  avoid excessive MPO decompositions for the same gate.
    // std::vector<std::shared_ptr<TensorCuda<PrecisionT>>> tensors_;

    /**
     * @brief Get a vector of pointers to extents of each site.
     *
     * @return std::vector<int64_t const *> Note int64_t const* is
     * required by cutensornet backend.
     */
    [[nodiscard]] auto getModeExtentsPtr_() -> std::vector<int64_t const *> {
        std::vector<int64_t const *> modeExtentsPtr_int64(numMPOSites_);
        for (std::size_t i = 0; i < numMPOSites_; i++) {
            modeExtentsPtr_int64[i] = modesExtents_int64_[i].data();
        }
        return modeExtentsPtr_int64;
    }

    /**
     * @brief Get a vector of pointers to tensor data of each site.
     *
     * @return std::vector<void *>
     */
    //[[nodiscard]] auto getTensorsDataPtr() -> std::vector<void *> {
    //    std::vector<void *> tensorsDataPtr(numMPOSites_);
    //    for (std::size_t i = 0; i < numMPOSites_; i++) {
    //        tensorsDataPtr[i] = reinterpret_cast<void *>(
    //            tensors_[i]->getDataBuffer().getData());
    //    }
    //    return tensorsDataPtr;
    //}

  public:
    explicit MPOTNCudaOpt(const std::string &opsName,
                          const std::vector<PrecisionT> &param,
                          const std::vector<std::size_t> &wires,
                          const std::vector<std::size_t> &wires_order,
                          const std::size_t maxMPOBondDim,
                          const std::vector<ComplexT> &matrix_data,
                          const std::size_t numQubits,
                          const cutensornetHandle_t &cutensornetHandle,
                          const cudaDataType_t &cudaDataType) {
        // Create an empty MPO tensor network operator. Note that the state
        // extents are aligned with the quantum state.
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /* const cutensornetHandle_t */ cutensornetHandle,
            /* int32_t */ static_cast<int32_t>(numQubits),
            /* const int64_t stateModeExtents */
            std::vector<int64_t>(numQubits, 2).data(),
            /* cudaDataType_t */ cudaDataType,
            /* cutensornetNetworkOperator_t */ &MPOOperator_));

        wires_ = wires;

        numMPOSites_ = wires.back() - wires.front() + 1;

        MPO_modes_int32_.resize(numMPOSites_);

        std::iota(MPO_modes_int32_.begin(), MPO_modes_int32_.end(),
                  wires.front());

        std::transform(MPO_modes_int32_.begin(), MPO_modes_int32_.end(),
                       MPO_modes_int32_.begin(), [numQubits](std::size_t mode) {
                           return static_cast<int32_t>(numQubits - 1 - mode);
                       });

        // Ensure the modes are in ascending order
        std::reverse(MPO_modes_int32_.begin(), MPO_modes_int32_.end());

        // set up max bond dimensions
        maxBondDim_ = maxMPOBondDim;

        // set up target bond dimensions
        std::vector<std::size_t> BondDims(wires.size() - 1, maxMPOBondDim);
        for (std::size_t i = 0; i < BondDims.size(); i++) {
            std::size_t bondDim = std::min(i + 1, BondDims.size() - i) *
                                  2; // 1+1 (1 for bra and 1 for ket)
            if (bondDim <= log2(maxBondDim_)) {
                BondDims[i] = std::size_t{1} << bondDim;
            }
        }

        bondDims_ = BondDims;
        // Insert bond dimensions of Identity tensors
        if (wires.size() != numMPOSites_) {
            for (std::size_t i = 0; i < wires.size() - 1; i++) {
                const std::size_t numISites = wires[i + 1] - wires[i] - 1;
                if (numISites > 0) {
                    std::vector<std::size_t> ISites(numISites, BondDims[i]);
                    bondDims_.insert(bondDims_.begin() + i + 1, ISites.begin(),
                                     ISites.end());
                }
            }
        }

        // set up MPO tensor mode extents and initialize MPO tensors
        for (std::size_t i = 0; i < numMPOSites_; i++) {
            std::vector<std::size_t> localModesExtents;
            if (i == 0) {
                localModesExtents = {2, bondDims_[i], 2};
            } else if (i == numMPOSites_ - 1) {
                localModesExtents = {bondDims_[i - 1], 2, 2};
            } else {
                localModesExtents = {bondDims_[i - 1], 2, bondDims_[i], 2};
            }

            modesExtents_int64_.emplace_back(
                Pennylane::Util::cast_vector<std::size_t, int64_t>(
                    localModesExtents));
        }

        auto &mpo_cache = TNCudaMPOCache<PrecisionT>::getInstance();

        auto mpo_data_ptr = mpo_cache.get_MPO_device_Ptr(
            opsName, param, wires_order, maxMPOBondDim, matrix_data);

        std::vector<std::size_t> mpo_site_tag(numMPOSites_, numMPOSites_);

        for (std::size_t i = 0; i < wires.size(); i++) {
            auto idx = wires[i] - wires[0];
            mpo_site_tag[idx] = i;
        }

        std::vector<void *> tensors_data_ptr(numMPOSites_, nullptr);

        // Update MPO tensors
        for (std::size_t i = 0; i < numMPOSites_; i++) {
            auto idx = mpo_site_tag[i];
            if (idx < numMPOSites_) {
                tensors_data_ptr[i] = mpo_data_ptr[idx];
            } else {
                tensors_data_ptr[i] = mpo_cache.get_Identity_MPO_device_Ptr(
                    bondDims_[idx - 1], bondDims_[idx]);
            }
        }

        // append MPO tensor network operator components
        PL_CUTENSORNET_IS_SUCCESS(cutensornetNetworkOperatorAppendMPO(
            /* const cutensornetHandle_t */ cutensornetHandle,
            /* cutensornetNetworkOperator_t */ MPOOperator_,
            /* const cuDoubleComplex */ coeff_,
            /* int32_t numStateModes */ static_cast<int32_t>(numMPOSites_),
            /* const int32_t stateModes[] */ MPO_modes_int32_.data(),
            /* const int64_t *stateModeExtents[] */
            getModeExtentsPtr_().data(),
            /* const int64_t *tensorModeStrides[] */ nullptr,
            /* const void * */
            std::vector<const void *>(tensors_data_ptr.cbegin(),
                                      tensors_data_ptr.cend())
                .data(),
            /* cutensornetBoundaryCondition_t */ boundaryCondition_,
            /* int64_t * */ &componentIdx_));
    }

    auto getMPOOperator() const -> const cutensornetNetworkOperator_t & {
        return MPOOperator_;
    }

    ~MPOTNCudaOpt() {
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyNetworkOperator(MPOOperator_));
    };
};
} // namespace Pennylane::LightningTensor::TNCuda