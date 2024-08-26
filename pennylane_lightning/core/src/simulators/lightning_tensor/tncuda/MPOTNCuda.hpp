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

#include "TensorCuda.hpp"
#include "tncudaError.hpp"
#include "tncuda_helpers.hpp"

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
}

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
template <class PrecisionT> class MPOTNCuda {
  private:
    using ComplexT = std::complex<PrecisionT>;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));

    std::vector<std::size_t> wires_; // pennylane  wires convention

    // To buuld a MPO tensor network, we need: 1. a cutensornetHandle; 2. a
    // cutensornetNetworkOperator_t object; 3. Complex coefficient associated
    // with the appended operator component. 4. Number of MPO sites; 5. MPO
    // tensor mode extents for each MPO tensor;
    // 6. Boundary conditions;
    cutensornetNetworkOperator_t networkOperator_;
    cuDoubleComplex coeff_{1.0, 0.0}; // default coefficient
    cutensornetBoundaryCondition_t boundaryCondition_{
        CUTENSORNET_BOUNDARY_CONDITION_OPEN}; // open boundary condition
    int64_t componentIdx_;

    std::size_t maxBondDim_;
    std::size_t numSites_;
    std::vector<std::size_t> stateModeExtents_;
    std::vector<int64_t> stateModeExtents_int64_;
    std::vector<std::size_t> modes_;
    std::vector<int32_t> modes_int32_;

    std::vector<std::size_t> bondDims_;

    std::vector<std::vector<std::size_t>> modesExtents_;
    std::vector<std::vector<int64_t>> modesExtents_int64_;
    std::vector<TensorCuda<PrecisionT>> tensors_;

    /**
     * @brief Get a vector of pointers to extents of each site.
     *
     * @return std::vector<int64_t const *> Note int64_t const* is
     * required by cutensornet backend.
     */
    [[nodiscard]] auto getModeExtentsPtr_() -> std::vector<int64_t const *> {
        std::vector<int64_t const *> modeExtentsPtr_int64(numSites_);
        for (std::size_t i = 0; i < numSites_; i++) {
            modeExtentsPtr_int64[i] = modesExtents_int64_[i].data();
        }
        return modeExtentsPtr_int64;
    }

    /**
     * @brief Get a vector of pointers to tensor data of each site.
     *
     * @return std::vector<uint64_t *>
     */
    [[nodiscard]] auto getTensorsDataPtr() -> std::vector<void *> {
        std::vector<void *> tensorsDataPtr(numSites_);
        for (std::size_t i = 0; i < numSites_; i++) {
            tensorsDataPtr[i] =
                reinterpret_cast<void *>(tensors_[i].getDataBuffer().getData());
        }
        return tensorsDataPtr;
    }

  public:
    explicit MPOTNCuda(const std::vector<std::vector<ComplexT>> &tensors,
                       const std::vector<std::size_t> &wires,
                       const std::size_t maxBondDim,
                       const std::size_t numQubits,
                       const cutensornetHandle_t &cutensornetHandle,
                       const cudaDataType_t &cudaDataType,
                       const DevTag<int> &dev_tag) {
        PL_ABORT_IF_NOT(tensors.size() == wires.size(),
                        "Number of tensors and wires must match.");

        PL_ABORT_IF(maxBondDim < 2,
                    "Max MPO bond dimension must be at least 2.");

        PL_ABORT_IF_NOT(std::is_sorted(wires.begin(), wires.end()),
                        "Only sorted target wires is accepeted.");

        wires_ = wires;

        // set up max bond dimensions and number of MPO sites
        maxBondDim_ = maxBondDim;
        numSites_ = wires.back() - wires.front() + 1;

        stateModeExtents_int64_ = std::vector<int64_t>(numQubits, 2);

        // set up MPO tensor network operator
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /* const cutensornetHandle_t */ cutensornetHandle,
            /* int32_t */ static_cast<int32_t>(numQubits),
            /* const int64_t stateModeExtents */
            stateModeExtents_int64_.data(),
            /* cudaDataType_t */ cudaDataType,
            /* cutensornetNetworkOperator_t */ &networkOperator_));

        // set up MPO target modes
        modes_.resize(numSites_);
        std::iota(modes_.begin(), modes_.end(), wires[0]);
        modes_int32_ = cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
            modes_, numQubits);

        // set up target bond dimensions
        std::vector<std::size_t> targetSitesBondDims(wires.size() - 1,
                                                     maxBondDim);
        for (std::size_t i = 0; i < targetSitesBondDims.size(); i++) {
            std::size_t bondDim =
                std::min(i + 1, targetSitesBondDims.size() - i) *
                2; // 1+1 (1 for bra and 1 for ket)
            if (bondDim <= log2(maxBondDim_)) {
                targetSitesBondDims[i] = (std::size_t{1} << bondDim);
                std::cout << "bondDim: " << bondDim << std::endl;
            }
        }

        bondDims_ = targetSitesBondDims;

        PL_ABORT_IF_NOT(bondDims_.size() == numSites_ - 1,
                        "Number of bond dimensions must match the number of "
                        "MPO sites.");

        // set up MPO tensor mode extents and initialize MPO tensors
        for (std::size_t i = 0; i < numSites_; i++) {
            if (i == 0) {
                modesExtents_.push_back({2, bondDims_[i], 2});
            } else if (i == numSites_ - 1) {
                modesExtents_.push_back({bondDims_[i - 1], 2, 2});
            } else {
                modesExtents_.push_back({bondDims_[i - 1], 2, bondDims_[i], 2});
            }

            modesExtents_int64_.emplace_back(
                Pennylane::Util::cast_vector<std::size_t, int64_t>(
                    modesExtents_.back()));

            tensors_.emplace_back(modesExtents_.back().size(),
                                  modesExtents_.back(), modesExtents_.back(),
                                  dev_tag);
            tensors_.back().getDataBuffer().zeroInit();
        }

        for (std::size_t i = 0; i < numSites_; i++) {
            auto tensor_cu =
                cuUtil::complexToCu<ComplexT>(tensors[numSites_ - 1 - i]);
            std::cout << "tensor_cu: ";
            for (auto t : tensor_cu) {
                std::cout << " " << t.x << " + " << t.y << "j; ";
            }
            std::cout << std::endl;

            tensors_[i].getDataBuffer().CopyHostDataToGpu(tensor_cu.data(),
                                                          tensor_cu.size());
        }

        /*

        // set up MPO target modes
        for (std::size_t i = 0; i < numSites_; i++) {
            modes_.push_back(wires.front() + i);
            modes_int32_.push_back(
                static_cast<int32_t>(numQubits - 1 - modes_.back()));
        }

        // set up target bond dimensions
        std::vector<std::size_t> targetSitesBondDims(wires.size() - 1,
                                                     maxBondDim);
        for (std::size_t i = 0; i < targetSitesBondDims.size(); i++) {
            std::size_t bondDim =
                std::min(i + 1, targetSitesBondDims.size() - i) *
                2; // 1+1 (1 for bra and 1 for ket)
            if (bondDim <= log2(maxBondDim_)) {
                targetSitesBondDims[i] = (std::size_t{1} << bondDim);
            }
        }

        std::vector<std::size_t> bondDims_orderC = targetSitesBondDims;

        // Insert bond dimensions of Identity tensors
        if (wires.size() != numSites_) {
            for (std::size_t i = 0; i < wires.size() - 1; i++) {
                const std::size_t numIdentitySites =
                    wires[i + 1] - wires[i] - 1;
                if (numIdentitySites > 0) {
                    std::vector<std::size_t> identitySites(
                        numIdentitySites, targetSitesBondDims[i]);
                    bondDims_orderC.insert(bondDims_orderC.begin() + i + 1,
                                           identitySites.begin(),
                                           identitySites.end());
                }
            }
        }

        bondDims_ = bondDims_orderC;

        PL_ABORT_IF_NOT(bondDims_.size() == numSites_ - 1,
                        "Number of bond dimensions must match the number of "
                        "MPO sites.");

        // set up MPO tensor mode extents and initialize MPO tensors
        for (std::size_t i = 0; i < numSites_; i++) {
            if (i == 0) {
                modesExtents_.push_back({2, bondDims_[i], 2});
            } else if (i == numSites_ - 1) {
                modesExtents_.push_back({bondDims_[i - 1], 2, 2});
            } else {
                modesExtents_.push_back({bondDims_[i - 1], 2, bondDims_[i], 2});
            }

            modesExtents_int64_.emplace_back(
                Pennylane::Util::cast_vector<std::size_t, int64_t>(
                    modesExtents_.back()));

            tensors_.emplace_back(modesExtents_.back().size(),
                                  modesExtents_.back(), modesExtents_.back(),
                                  dev_tag);
            tensors_.back().getDataBuffer().zeroInit();
        }

        // set up MPO tensor data
        std::vector<std::size_t> target_map(numSites_, numSites_);

        for (std::size_t i = 0; i < wires.size(); i++) {
            auto idx = wires[i] - wires[0];
            target_map[idx] = i;
        }

        for (std::size_t i = 0; i < numSites_; i++) {
            if (target_map[i] == numSites_) {
                CFP_t value_cu{1.0, 0.0};
                std::size_t target_idx = 0;
                PL_CUDA_IS_SUCCESS(cudaMemcpy(
                    &tensors_[i].getDataBuffer().getData()[target_idx],
                    &value_cu, sizeof(CFP_t), cudaMemcpyHostToDevice));

                target_idx = 2 * bondDims_[i - 1] + 1;

                PL_CUDA_IS_SUCCESS(cudaMemcpy(
                    &tensors_[i].getDataBuffer().getData()[target_idx],
                    &value_cu, sizeof(CFP_t), cudaMemcpyHostToDevice));
            } else {
                const std::size_t wire_idx = target_map[i];
                auto tensor_cu = cuUtil::complexToCu<ComplexT>(
                    tensors[wires.size() - 1 - wire_idx]);

                tensors_[i].getDataBuffer().CopyHostDataToGpu(tensor_cu.data(),
                                                              tensor_cu.size());
            }
        }
        */

        // append MPO tensor network operator components
        PL_CUTENSORNET_IS_SUCCESS(cutensornetNetworkOperatorAppendMPO(
            /* const cutensornetHandle_t */ cutensornetHandle,
            /* cutensornetNetworkOperator_t */ networkOperator_,
            /* const cuDoubleComplex */ coeff_,
            /* int32_t numStateModes */ static_cast<int32_t>(numSites_),
            /* const int32_t stateModes[] */ modes_int32_.data(),
            /* const int64_t *stateModeExtents[] */
            getModeExtentsPtr_().data(),
            /* const int64_t *tensorModeStrides[] */ nullptr,
            /* const void * */
            const_cast<const void **>(getTensorsDataPtr().data()),
            /* cutensornetBoundaryCondition_t */ boundaryCondition_,
            /* int64_t * */ &componentIdx_));
    }

    auto getMPOOperator() const -> const cutensornetNetworkOperator_t & {
        return networkOperator_;
    }

    ~MPOTNCuda() {
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyNetworkOperator(networkOperator_));
    };
};
} // namespace Pennylane::LightningTensor::TNCuda