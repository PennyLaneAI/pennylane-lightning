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
#include "TensorCuda.hpp"
#include "Util.hpp"
#include "cuGates_host.hpp"
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
 * This class creates `custatenetTensorNetwork`  from
 * `ObservablesTNCuda` objects for measurement purpose. Since the NamedObs,
 * HermitianObs, TensorProdObs and Hamiltionian objects can be encapsulated in a
 * `cutensornetNetworkOperator_t` instance, only one ObservableTNCudaOperator
 * class is designed here. Note that a `cutensornetNetworkOperator_t` object can
 * only be created and destroyed by creating a new ObservableTNCudaOperator
 * object, which ensures its lifetime is aligned with that of data associated to
 * it.
 *
 * @tparam TensorNetT tensor network class.
 */
template <class TensorNetT> class ObservableTNCudaOperator {
  public:
    using PrecisionT = typename TensorNetT::PrecisionT;
    using CFP_t = typename TensorNetT::CFP_t;
    using ComplexT = typename TensorNetT::ComplexT;
    using obs_key =
        std::tuple<std::string, std::vector<PrecisionT>, std::size_t>;

  private:
    cutensornetNetworkOperator_t obsOperator_{
        nullptr}; // cutensornetNetworkOperator operator

    const TensorNetT &tensor_network_; // quantum state to be measured

    const std::size_t numObsTerms_;    // number of observable terms
    vector1D<cuDoubleComplex> coeffs_; // coefficients for each term
    vector1D<std::size_t> numTensors_; // number of tensors in each term

    vector2D<int32_t>
        numModes_; // number of state modes of each tensor in each term

    vector3D<int32_t> modes_; // modes for each tensor in each term

    vector2D<const int32_t *>
        modesPtr_; // pointers for modes of each tensor in each term

    vector2D<const void *>
        tensorDataPtr_; // pointers for each tensor data in each term

    std::vector<int64_t> ids_; // ids for each term in the graph

  private:
    /**
     * @brief Hasher for observable key.
     */
    struct ObsKeyHasher {
        std::size_t operator()(
            const std::tuple<std::string, std::vector<PrecisionT>, std::size_t>
                &obsKey) const {
            std::size_t hash_val =
                std::hash<std::string>{}(std::get<0>(obsKey));
            for (const auto &param : std::get<1>(obsKey)) {
                hash_val ^= std::hash<PrecisionT>{}(param);
            }
            hash_val ^= std::hash<std::size_t>{}(std::get<2>(obsKey));
            return hash_val;
        }
    };

    /**
     * @brief Cache for observable data on device.
     */
    std::unordered_map<obs_key, TensorCuda<PrecisionT>, ObsKeyHasher>
        device_obs_cache_;

    /**
     * @brief Add an observable numerical value to the cached map, indexed by
     * the name, parameters and hash value(default as 0 for named observables).
     *
     * @param obs_name String representing the name of the given observable.
     * @param obs_param Vector of parameter values. `{}` if non-parametric
     * gate.
     */
    void add_obs_(const std::string &obs_name,
                  [[maybe_unused]] std::vector<PrecisionT> &obs_param = {}) {
        auto obsKey = std::make_tuple(obs_name, obs_param, std::size_t{0});

        auto &gateMap =
            cuGates::DynamicGateDataAccess<PrecisionT>::getInstance();

        add_obs_(obsKey, gateMap.getGateData(obs_name, obs_param));
    }

    /**
     * @brief Add observable numerical value to the cache map, the name,
     * parameters and hash value(default as 0 for named observables).
     *
     * @param obsKey obs_key tuple representing the name, parameters and hash
     * value(default as 0 for named observables).
     * @param obs_data_host Vector of complex floating point values
     * representing the observable data on host.
     */
    void add_obs_(const obs_key &obsKey,
                  const std::vector<CFP_t> &obs_data_host) {
        const std::size_t rank = Pennylane::Util::log2(obs_data_host.size());
        auto modes = std::vector<std::size_t>(rank, 0);
        auto extents = std::vector<std::size_t>(rank, 2);

        auto &&tensor = TensorCuda<PrecisionT>(rank, modes, extents,
                                               tensor_network_.getDevTag());

        device_obs_cache_.emplace(std::piecewise_construct,
                                  std::forward_as_tuple(obsKey),
                                  std::forward_as_tuple(std::move(tensor)));

        device_obs_cache_.at(obsKey).getDataBuffer().CopyHostDataToGpu(
            obs_data_host.data(), obs_data_host.size());
    }

    /**
     * @brief Returns a pointer to the GPU device memory where the observable is
     * stored.
     *
     * @param obsKey The key of observable tensor operator.
     * @return const CFP_t* Pointer to gate values on device.
     */
    const CFP_t *get_obs_device_ptr_(const obs_key &obsKey) {
        return device_obs_cache_.at(obsKey).getDataBuffer().getData();
    }

  public:
    ObservableTNCudaOperator(const TensorNetT &tensor_network,
                             ObservableTNCuda<TensorNetT> &obs)
        : tensor_network_{tensor_network},
          numObsTerms_(obs.getNumTensors().size()) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /* const cutensornetHandle_t */ tensor_network.getTNCudaHandle(),
            /* int32_t */ static_cast<int32_t>(tensor_network.getNumQubits()),
            /* const int64_t stateModeExtents */
            reinterpret_cast<int64_t *>(const_cast<std::size_t *>(
                tensor_network.getQubitDims().data())),
            /* cudaDataType_t */ tensor_network.getCudaDataType(),
            /* cutensornetNetworkOperator_t */ &obsOperator_));

        numTensors_ = obs.getNumTensors(); // number of tensors in each term

        for (std::size_t term_idx = 0; term_idx < numObsTerms_; term_idx++) {
            auto coeff = cuDoubleComplex{
                static_cast<double>(obs.getCoeffs()[term_idx]), 0.0};
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
                        tensor_network.getNumQubits()));
            }
            modes_.emplace_back(modes_per_term);

            // modes pointer initialization
            vector1D<const int32_t *> modesPtrPerTerm;
            for (std::size_t tensor_idx = 0; tensor_idx < modes_.back().size();
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

                if (device_obs_cache_.find(obsKey) == device_obs_cache_.end()) {
                    if (hermitianMatrix.empty()) {
                        add_obs_(obsName, param);
                    } else {
                        auto hermitianMatrix_cu =
                            cuUtil::complexToCu<ComplexT>(hermitianMatrix);
                        add_obs_(obsKey, hermitianMatrix_cu);
                    }
                }
                tensorDataPtrPerTerm_.emplace_back(get_obs_device_ptr_(obsKey));
            }

            tensorDataPtr_.emplace_back(tensorDataPtrPerTerm_);

            appendTNOperator_(coeff, numTensors, numModes_.back().data(),
                              modesPtr_.back().data(),
                              tensorDataPtr_.back().data());
        }
    }

    ~ObservableTNCudaOperator() {
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyNetworkOperator(obsOperator_));
    }

    /**
     * @brief Get the `cutensornetNetworkOperator_t` object.
     *
     * @return cutensornetNetworkOperator_t
     */
    [[nodiscard]] auto getTNOperator() const -> cutensornetNetworkOperator_t {
        return obsOperator_;
    }

  private:
    /**
     * @brief Append a product of tensors to the `cutensornetNetworkOperator_t`
     *
     * @param coeff Coefficient of the product.
     * @param numTensors Number of tensors in the product.
     * @param numStateModes Number of state modes of each tensor in the product.
     * @param stateModes State modes of each tensor in the product.
     * @param tensorDataPtr Pointer to the data of each tensor in the product.
     */
    void appendTNOperator_(const cuDoubleComplex &coeff,
                           const std::size_t numTensors,
                           const int32_t *numStateModes,
                           const int32_t **stateModes,
                           const void **tensorDataPtr) {
        int64_t id;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetNetworkOperatorAppendProduct(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
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
