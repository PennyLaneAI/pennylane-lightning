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
 * Class for cuTensorNet-backed Matrix Product Operator.
 */

#pragma once

#include <algorithm>
#include <cuComplex.h>
#include <cutensornet.h>
#include <vector>

#include "TensorCuda.hpp"
#include "tncudaError.hpp"
#include "tncuda_helpers.hpp"

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
}

namespace Pennylane::LightningTensor::TNCuda {

/**
 * @brief Class representing an Matrix Product Operator (MPO) object for the MPS
 backend.
 * Any gate tensor can be represented as an MPO tensor network in the context of
 MPS. The gate tensor must be decomposed with respect to its target wires. If
 the target wires are not adjacent, identity tensors are inserted between the
 MPO tensors.
 1. The MPO tensors' modes order in an open boundary condition are:
   2              3              2
   |              |              |
   X--1--....--0--X--2--....--0--X
   |              |              |
   0              1              1

 2. The extents of the left side bound MPO tensor are [2, bondR, 2].
   The extents of the right side bound MPO tensor are [bondL, 2, 2].
   The extents of the middle MPO tensors are [bondL, 2, bondR, 2].

 Note that the gate tensor should be permuted to ascending order and decomposed
 into MPO sites before passing to this class. Preprocess and postprocess with
 SWAP operations are required to ensure MPOs target at adjacent wires and the
 target wires are correct.

 * @tparam PrecisionT Floating point type.
 */
template <class PrecisionT> class MPOTNCuda {
  private:
    using ComplexT = std::complex<PrecisionT>;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));

    cutensornetNetworkOperator_t MPOOperator_;
    cuDoubleComplex coeff_ =
        make_cuDoubleComplex(1.0, 0.0); // default coefficient
    cutensornetBoundaryCondition_t boundaryCondition_{
        CUTENSORNET_BOUNDARY_CONDITION_OPEN}; // open boundary condition
    int64_t componentIdx_;

    std::vector<std::size_t> bondDims_;

    std::size_t numMPOSites_;
    std::vector<std::size_t> modes_;
    std::vector<int32_t> MPO_modes_int32_;

    std::vector<std::vector<int64_t>> modesExtents_int64_;
    // TODO: Explore if tensors_ can be stored in a separate memory manager
    // class
    std::vector<std::shared_ptr<TensorCuda<PrecisionT>>> tensors_;

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
    [[nodiscard]] auto getTensorsDataPtr_() -> std::vector<void *> {
        std::vector<void *> tensorsDataPtr(numMPOSites_);
        for (std::size_t i = 0; i < numMPOSites_; i++) {
            tensorsDataPtr[i] = reinterpret_cast<void *>(
                tensors_[i]->getDataBuffer().getData());
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

        PL_ABORT_IF_NOT(wires.size() == wires.back() - wires.front() + 1,
                        "Only support local target wires.");

        // Create an empty MPO tensor network operator. Note that the state
        // extents are aligned with the quantum state.
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /* const cutensornetHandle_t */ cutensornetHandle,
            /* int32_t */ static_cast<int32_t>(numQubits),
            /* const int64_t stateModeExtents */
            std::vector<int64_t>(numQubits, 2).data(),
            /* cudaDataType_t */ cudaDataType,
            /* cutensornetNetworkOperator_t */ &MPOOperator_));

        numMPOSites_ = wires.size();

        MPO_modes_int32_.resize(numMPOSites_);

        std::iota(MPO_modes_int32_.begin(), MPO_modes_int32_.end(),
                  wires.front());

        std::transform(MPO_modes_int32_.begin(), MPO_modes_int32_.end(),
                       MPO_modes_int32_.begin(), [numQubits](std::size_t mode) {
                           return static_cast<int32_t>(numQubits - 1 - mode);
                       });

        // Ensure the modes are in ascending order
        std::reverse(MPO_modes_int32_.begin(), MPO_modes_int32_.end());

        std::vector<std::size_t> BondDims(wires.size() - 1, maxBondDim);
        for (std::size_t i = 0; i < BondDims.size(); i++) {
            std::size_t bondDim = std::min(i + 1, BondDims.size() - i) *
                                  2; // 1+1 (1 for bra and 1 for ket)
            if (bondDim <= log2(maxBondDim)) {
                BondDims[i] = std::size_t{1} << bondDim;
            }
        }

        bondDims_ = BondDims;

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

            tensors_.emplace_back(std::make_shared<TensorCuda<PrecisionT>>(
                localModesExtents.size(), localModesExtents, localModesExtents,
                dev_tag));
            tensors_[i]->getDataBuffer().zeroInit();
        }

        for (std::size_t i = 0; i < numMPOSites_; i++) {
            auto tensor_cu = cuUtil::complexToCu<ComplexT>(tensors[i]);
            tensors_[i]->getDataBuffer().CopyHostDataToGpu(tensor_cu.data(),
                                                           tensor_cu.size());
        }

        // Append MPO tensor network operator components
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
            const_cast<const void **>(getTensorsDataPtr_().data()),
            /* cutensornetBoundaryCondition_t */ boundaryCondition_,
            /* int64_t * */ &componentIdx_));
    }

    auto getMPOOperator() const -> const cutensornetNetworkOperator_t & {
        return MPOOperator_;
    }

    ~MPOTNCuda() {
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyNetworkOperator(MPOOperator_));
    };
};
} // namespace Pennylane::LightningTensor::TNCuda
