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
#include <tuple>
#include <vector>

#include "ObservablesTNCuda.hpp"
#include "cuGates_host.hpp"

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
// given circuit.
/**
 * @brief ObservableTNCudaOperator Class.
 *
 * This class creates `custatenetTensorNetwork` Operator from
 * `ObservablesTNCuda` objects for measurement purpose. Since the NamedObs,
 * HermitianObs, TensorProdObs and Hamiltionain objects can be encapsulated in a
 * `cutensornetNetworkOperator_t` instance, only one ObservableTNCudaOperator
 * class is designed here. Note that a `cutensornetNetworkOperator_t object can
 * only be created and destroyed by creating a new ObservableTNCudaOperator
 * object, which ensures its lifetime is aligned with that of data associated to
 * it.
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT> class ObservableTNCudaOperator {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;

    using CFP_t = typename StateTensorT::CFP_t;

    using ComplexT = typename StateTensorT::ComplexT;

    using obs_key =
        std::tuple<std::string, std::vector<PrecisionT>, std::size_t>;

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

    struct ObsKeyHaser {
        std::size_t operator()(
            const std::tuple<std::string, std::vector<PrecisionT>, std::size_t>
                &obsKey) const {
            std::size_t hash_val = 0;
            hash_val ^= std::hash<std::string>{}(std::get<0>(obsKey));
            for (const auto &param : std::get<1>(obsKey)) {
                hash_val ^= std::hash<PrecisionT>{}(param);
            }
            hash_val ^= std::hash<std::size_t>{}(std::get<2>(obsKey));
            return hash_val;
        }
    };

    std::unordered_map<obs_key, TensorCuda<PrecisionT>, ObsKeyHaser>
        device_obs_cache_;

    /**
     * @brief Add gate numerical value to the cache, indexed by the id of gate
     * tensor operator in the graph and its name and parameter value are
     * recorded as well.
     *
     * @param obs_name String representing the name of the given gate.
     * @param obs_param Vector of parameter values. `{}` if non-parametric
     * gate.
     */
    void add_obs_(const std::string &obs_name,
                  [[maybe_unused]] std::vector<PrecisionT> obs_param = {}) {
        auto obsKey = std::make_tuple(obs_name, obs_param, std::size_t{0});

        auto &gateMap =
            cuGates::DynamicGateDataAccess<PrecisionT>::getInstance();

        add_obs_(obsKey, gateMap.getGateData(obs_name, obs_param));
    }
    /**
     * @brief Add gate numerical value to the cache, indexed by the id of gate
     * tensor operator in the graph and its name and parameter value as well as
     * the gate data on host.
     *
     * @param gate_id The id of gate tensor operator in the computate graph.
     * @param gate_key String representing the name of the given gate as well as
     * its associated parameter value.
     * @param gate_data_host Vector of complex floating point values
     * representing the gate data on host.
     */

    void add_obs_(const obs_key &obsKey,
                  const std::vector<CFP_t> &gate_data_host) {
        const std::size_t rank = Pennylane::Util::log2(gate_data_host.size());
        auto modes = std::vector<std::size_t>(rank, 0);
        auto extents = std::vector<std::size_t>(rank, 2);

        auto &&tensor = TensorCuda<PrecisionT>(rank, modes, extents,
                                               state_tensor_.getDevTag());

        device_obs_cache_.emplace(std::piecewise_construct,
                                  std::forward_as_tuple(obsKey),
                                  std::forward_as_tuple(std::move(tensor)));

        device_obs_cache_.at(obsKey).getDataBuffer().CopyHostDataToGpu(
            gate_data_host.data(), gate_data_host.size());
    }

    /**
     * @brief Returns a pointer to the GPU device memory where the gate is
     * stored.
     *
     * @param gate_id The id of gate tensor operator in the computate graph.
     * @return const CFP_t* Pointer to gate values on device.
     */
    CFP_t *get_obs_device_ptr_(const obs_key &obsKey) {
        return device_obs_cache_.at(obsKey).getDataBuffer().getData();
    }

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
            vector1D<const void *> tensorDataPtrPerTerm_;
            for (std::size_t tensor_idx = 0; tensor_idx < numTensors;
                 tensor_idx++) {
                auto metaData = obs.getMetaData()[term_idx][tensor_idx];

                auto obsName = std::get<0>(metaData);
                auto param = std::get<1>(metaData);
                auto hermitianMatrix = std::get<2>(metaData);
                std::size_t hash_val = 0;

                if (!hermitianMatrix.empty()) {
                    hash_val = MatrixHasher()(hermitianMatrix);
                }

                auto obsKey = std::make_tuple(obsName, param, hash_val);

                if (device_obs_cache_.find(obsKey) ==
                        device_obs_cache_.end()) {
                     if(hermitianMatrix.empty()){
                        add_obs_(obsName, param);
                     }else{
                        auto hermitianMatrix_cu =
                            cuUtil::complexToCu<ComplexT>(hermitianMatrix);
                        add_obs_(obsKey, hermitianMatrix_cu);
                     }
                }
                tensorDataPtrPerTerm_.emplace_back(
                        get_obs_device_ptr_(obsKey));
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
