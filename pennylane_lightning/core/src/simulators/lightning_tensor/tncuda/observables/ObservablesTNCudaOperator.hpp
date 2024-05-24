// Copyright 2024 Xanadu Quantum Technologies Inc. and contributors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cutensornet.h>
#include <vector>

#include "ObservablesTNCuda.hpp"

#include "TensorCuda.hpp"
#include "Util.hpp"
#include "tncudaError.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::LightningTensor::TNCuda;
template <class T> using vector1D = std::vector<T>;
template <class T> using vector2D = std::vector<vector1D<T>>;
template <class T> using vector3D = std::vector<vector2D<T>>;

} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::Observables {

// Current design allows multiple measurements to be performed for a
// circuit.
/**
 * @brief ObservableTNCudaOperator Class.
 *
 * This class creates custatenetTensorNetwork Operator from ObservablesTNCuda
 * objects for measurement purpose. Since the NamedObs, HermitianObs,
 * TensorProdObs and Hamiltionain objects can be encapsulated in a
 * cutensornetNetworkOperator_t instance, only one ObservableTNCudaOperator
 * class is designed here. Note that a cutensornetNetworkOperator_t object can
 * only be created and destroyed by creating a new ObservableTNCudaOperator
 * object, which ensures its lifetime is aligned with that of data associated to
 * it.
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT> class ObservableTNCudaOperator {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;

  private:
    // cutensornetNetworkOperator operator
    cutensornetNetworkOperator_t obsOperator_{nullptr};

    // Quatum state to be measured
    const StateTensorT &state_tensor_;
    const size_t numObsTerms_;

    // ids for each term in the graph
    std::vector<int64_t> ids_;

    // coefficients for each term
    vector1D<cuDoubleComplex> coeffs_;

    // number of tensors in each term
    vector1D<size_t> numTensors_;

    // number of state modes of each tensor in each term
    vector2D<int32_t> numModes_;

    vector3D<int32_t> modes_;

    vector2D<const int32_t *> modesPtr_;

    vector2D<const void *> tensorDataPtr_;

    vector2D<TensorCuda<PrecisionT>> tensorData_;

  public:
    ObservableTNCudaOperator(const StateTensorT &state_tensor,
                             ObservableTNCuda<StateTensorT> &obs)
        : state_tensor_{state_tensor},
          numObsTerms_(obs.getNumTensors().size()) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /* const cutensornetHandle_t */ state_tensor.getTNCudaHandle(),
            /* int32_t */ static_cast<int32_t>(state_tensor.getNumQubits()),
            /* const int64_t stateModeExtents */
            reinterpret_cast<int64_t *>(
                const_cast<size_t *>(state_tensor.getQubitDims().data())),
            /* cudaDataType_t */ state_tensor.getCudaDataType(),
            /* cutensornetNetworkOperator_t */ &obsOperator_));

        numTensors_ = obs.getNumTensors(); // number of tensors in each term

        for (std::size_t term_idx = 0; term_idx < numObsTerms_; term_idx++) {
            auto coeff = cuDoubleComplex{
                static_cast<double>(obs.getCoeffsPerTerm()[term_idx]), 0.0};
            auto numTensors = numTensors_[term_idx];

            coeffs_.emplace_back(coeff);

            // number of state modes of each tensor in each term
            numModes_.emplace_back(cast_vector<std::size_t, int32_t>(
                obs.getNumStateModes()[term_idx]));

            // modes initialization
            vector2D<int32_t> modes_per_term;
            for (std::size_t tensor_idx = 0; tensor_idx < numTensors;
                 tensor_idx++) {
                modes_per_term.emplace_back(
                    cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
                        obs.getStateModes()[term_idx][tensor_idx],
                        state_tensor.getNumQubits()));
            }
            modes_.emplace_back(modes_per_term);

            // modes pointer initialization
            vector1D<const int32_t *> modesPtrPerTerm;
            for (size_t tensor_idx = 0; tensor_idx < modes_.back().size();
                 tensor_idx++) {
                modesPtrPerTerm.emplace_back(modes_.back()[tensor_idx].data());
            }
            modesPtr_.emplace_back(modesPtrPerTerm);

            // tensor data initialization
            vector1D<TensorCuda<PrecisionT>> tensorDataPerTerm_;
            for (std::size_t tensor_idx = 0; tensor_idx < numTensors;
                 tensor_idx++) {
                auto rank = Pennylane::Util::log2(
                    obs.getData()[term_idx][tensor_idx].size());
                tensorDataPerTerm_.emplace_back(
                    std::vector<std::size_t>(rank, 2),
                    obs.getData()[term_idx][tensor_idx],
                    state_tensor.getDevTag());
            }

            tensorData_.emplace_back(tensorDataPerTerm_);

            vector1D<const void *> tensorDataPtrPerTerm_;
            for (size_t tensor_idx = 0; tensor_idx < tensorData_.back().size();
                 tensor_idx++) {
                tensorDataPtrPerTerm_.emplace_back(
                    tensorData_.back()[tensor_idx].getDataBuffer().getData());
            }

            tensorDataPtr_.emplace_back(tensorDataPtrPerTerm_);

            appendTNOperator_(coeff, numTensors, numModes_.back().data(),
                              modesPtr_.back().data(),
                              tensorDataPtr_.back().data());
        }
    }

    virtual ~ObservableTNCudaOperator() {
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyNetworkOperator(obsOperator_));
    }

    [[nodiscard]] auto getTNOperator() -> cutensornetNetworkOperator_t {
        return obsOperator_;
    }

  private:
    void appendTNOperator_(const cuDoubleComplex &coeff,
                           const std::size_t numTensors,
                           const int32_t *numStateModes,
                           const int32_t **stateModes,
                           const void **tensorDataPtr) {
        int64_t id;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetNetworkOperatorAppendProduct(
            /* const cutensornetHandle_t */ state_tensor_.getTNCudaHandle(),
            /* cutensornetNetworkOperator_t */ getTNOperator(),
            /* cuDoubleComplex coefficient*/ coeff,
            /* int32_t numTensors */ static_cast<int32_t>(numTensors),
            /* const int32_t numStateModes[] */ numStateModes,
            /* const int32_t *stateModes[] */ stateModes,
            /* const int64_t *tensorModeStrides[] */ nullptr,
            /* const void *tensorData[] */ tensorDataPtr,
            /* int64_t* */ &id));
        ids_.push_back(id);
    }
};
} // namespace Pennylane::LightningTensor::TNCuda::Observables
