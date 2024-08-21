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

/**
 * @file ObservablesTNCudaOperator.hpp
 * Class for appending a ObservablesTNCuda object to a tensor network object.
 */

#pragma once

#include <cutensornet.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "LinearAlg.hpp"
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

    using MetaDataT = std::tuple<std::string, std::vector<PrecisionT>,
                                 std::vector<ComplexT>>; // name, params, matrix
    using obs_key =
        std::tuple<std::string, std::vector<PrecisionT>, std::size_t>;

  private:
    static inline std::unordered_map<std::string, std::string> pauli_map_{
        {"Identity", "I"},
        {"PauliX", "X"},
        {"PauliY", "Y"},
        {"PauliZ", "Z"},
        {"Hadamard", "H"}};

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

    const bool var_cal_{false};

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
     * @brief Create a map of modes to observable metadata.
     *
     * @param obs An observableTNCuda object.
     * @param modes Modes of all observable terms.
     * @param term_idx Index of the first observable term.
     * @param term_idy Index of the second observable term.
     *
     * @return std::unordered_map<int32_t, std::vector<MetaDataT>> Map of modes
     * to observable meta data.
     */
    auto create_modes_obsname_map_(ObservableTNCuda<TensorNetT> &obs,
                                   const vector3D<int32_t> &modes,
                                   const std::size_t term_idx,
                                   const std::size_t term_idy)
        -> std::unordered_map<int32_t, std::vector<MetaDataT>> {
        std::unordered_map<int32_t, std::vector<MetaDataT>> modes_obsname_map;

        auto &&modes_termx = modes[term_idx];
        auto &&modes_termy = modes[term_idy];

        for (std::size_t tensor_idx = 0; tensor_idx < modes_termx.size();
             tensor_idx++) {
            PL_ABORT_IF_NOT(modes_termx[tensor_idx].size() == 1,
                            "Only one-wire observables are "
                            "supported for cutensornet v24.03");

            modes_obsname_map[modes_termx[tensor_idx][0]] = {
                obs.getMetaData()[term_idx][tensor_idx]};
        }

        for (std::size_t tensor_idy = 0; tensor_idy < modes_termy.size();
             tensor_idy++) {
            auto &&termy = modes_termy[tensor_idy];
            auto it = modes_obsname_map.find(termy.front());
            if (it == modes_obsname_map.end()) {
                modes_obsname_map[termy.front()] = {
                    obs.getMetaData()[term_idy][tensor_idy]};
            } else {
                modes_obsname_map[termy.front()].push_back(
                    obs.getMetaData()[term_idy][tensor_idy]);
            }
        }

        return modes_obsname_map;
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
     * @brief Add metadata of an observable.
     *
     * @param metaData Metadata of the observable.
     * @return obs_key The key of observable tensor operator.
     */
    auto add_meta_data_(const MetaDataT &metaData) -> obs_key {
        auto obsName = std::get<0>(metaData);
        auto param = std::get<1>(metaData);
        auto hermitianMatrix = std::get<2>(metaData);
        std::size_t hash_val =
            hermitianMatrix.empty() ? 0 : MatrixHasher()(hermitianMatrix);

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

        return obsKey;
    }

    /**
     * @brief Add metadata of product of two observables at the same target.
     *
     * @param metaData0 Metadata of the first observable.
     * @param metaData1 Metadata of the second observable.
     * @param cublas CublasCaller object.
     *
     * @return obs_key The key of observable tensor operator.
     */
    auto add_meta_data_(const MetaDataT &metaData0, const MetaDataT &metaData1,
                        const CublasCaller &cublascaller) -> obs_key {
        auto obsName0 = std::get<0>(metaData0);
        auto obsName1 = std::get<0>(metaData1);
        // Branch for two Pauli observables
        if (pauli_map_.find(obsName0) != pauli_map_.end() &&
            pauli_map_.find(obsName1) != pauli_map_.end()) {
            auto obsName = pauli_map_[obsName0] + "@" + pauli_map_[obsName1];
            return add_meta_data_(MetaDataT{obsName, {}, {}});
        }

        auto obsName = obsName0 + "@" + obsName1;

        auto obsMat0 = std::get<2>(metaData0);
        auto obsMat1 = std::get<2>(metaData1);

        auto hermitianMatrix = obsMat0;
        hermitianMatrix.insert(hermitianMatrix.end(), obsMat1.begin(),
                               obsMat1.end());

        std::size_t hash_val = MatrixHasher()(hermitianMatrix);

        auto obsKey =
            std::make_tuple(obsName, std::vector<PrecisionT>{}, hash_val);

        if (device_obs_cache_.find(obsKey) == device_obs_cache_.end()) {
            std::vector<CFP_t> hermitianMatrix_cu(
                obsMat0.empty() ? obsMat1.size() : obsMat0.size());

            add_obs_(obsKey, hermitianMatrix_cu);

            auto &&obsKey0 = add_meta_data_(metaData0);
            auto &&obsKey1 = add_meta_data_(metaData1);

            // update the matrix data with MM operation
            CFP_t *mat0 = const_cast<CFP_t *>(get_obs_device_ptr_(obsKey0));
            CFP_t *mat1 = const_cast<CFP_t *>(get_obs_device_ptr_(obsKey1));
            CFP_t *res = const_cast<CFP_t *>(get_obs_device_ptr_(obsKey));
            const std::size_t m =
                Pennylane::Util::log2(hermitianMatrix_cu.size());

            GEMM_CUDA_device(mat0, mat1, res, m, m, m,
                             tensor_network_.getDevTag().getDeviceID(),
                             tensor_network_.getDevTag().getStreamID(),
                             cublascaller);
        }

        return obsKey;
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

    /**
     * @brief Initialize the observable tensor operator for expectation value
     * calculation.
     *
     * @param tensor_network Tensor network object.
     * @param obs ObservableTNCuda object.
     */
    void initHelper_expval_(const TensorNetT &tensor_network,
                            ObservableTNCuda<TensorNetT> &obs) {
        for (std::size_t term_idx = 0; term_idx < numObsTerms_; term_idx++) {
            PrecisionT coeff_real = obs.getCoeffs()[term_idx];
            auto numTensors = obs.getNumTensors()[term_idx];

            coeffs_.emplace_back(
                cuDoubleComplex{static_cast<double>(coeff_real), 0.0});
            numTensors_.emplace_back(numTensors);

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

            // Not required for var calculation below
            //  modes pointer initialization
            vector1D<const int32_t *> modesPtrPerTerm;
            for (std::size_t tensor_idx = 0;
                 tensor_idx < modes_[term_idx].size(); tensor_idx++) {
                modesPtrPerTerm.emplace_back(
                    modes_[term_idx][tensor_idx].data());
            }
            modesPtr_.emplace_back(modesPtrPerTerm);

            // tensor data initialization
            vector1D<const void *> tensorDataPtrPerTerm_;
            for (std::size_t tensor_idx = 0; tensor_idx < numTensors;
                 tensor_idx++) {
                auto obsKey =
                    add_meta_data_(obs.getMetaData()[term_idx][tensor_idx]);

                tensorDataPtrPerTerm_.emplace_back(get_obs_device_ptr_(obsKey));
            }
            tensorDataPtr_.emplace_back(tensorDataPtrPerTerm_);
        }
    }

    /**
     * @brief Initialize the observable tensor operator for variance
     * calculation.
     *
     * @param tensor_network Tensor network object.
     * @param obs ObservableTNCuda object.
     */
    void initHelper_var_(const TensorNetT &tensor_network,
                         ObservableTNCuda<TensorNetT> &obs) {
        // convert obs modes to cutensornet compatible format/order
        vector3D<int32_t> modes;
        for (std::size_t term_idx = 0; term_idx < numObsTerms_; term_idx++) {
            vector2D<int32_t> modes_per_term;
            for (std::size_t tensor_idx = 0;
                 tensor_idx < obs.getNumTensors()[term_idx]; tensor_idx++) {
                modes_per_term.emplace_back(
                    cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
                        obs.getStateModes()[term_idx][tensor_idx],
                        tensor_network.getNumQubits()));
            }
            modes.emplace_back(modes_per_term);
        }

        for (std::size_t term_idx = 0; term_idx < numObsTerms_; term_idx++) {
            for (std::size_t term_idy = 0; term_idy < numObsTerms_;
                 term_idy++) {
                PrecisionT coeff_real =
                    obs.getCoeffs()[term_idx] * obs.getCoeffs()[term_idy];

                coeffs_.emplace_back(
                    cuDoubleComplex{static_cast<double>(coeff_real), 0.0});

                auto modes_obsname_map =
                    create_modes_obsname_map_(obs, modes, term_idx, term_idy);

                auto numTensorsPerTerm = modes_obsname_map.size();

                numTensors_.emplace_back(numTensorsPerTerm);

                vector2D<int32_t> modes_per_term;
                vector1D<const void *> tensorDataPtrPerTerm_;
                vector1D<int32_t> num_modes_per_term;

                for (const auto &tensors_info : modes_obsname_map) {
                    modes_per_term.emplace_back(
                        std::vector<int32_t>{tensors_info.first});

                    num_modes_per_term.emplace_back(
                        modes_per_term.back().size());

                    auto metaDataArr = tensors_info.second;

                    obs_key obsKey;

                    if (metaDataArr.size() == 1) {
                        obsKey = std::move(add_meta_data_(metaDataArr[0]));
                    } else if (metaDataArr.size() == 2) {
                        obsKey = std::move(
                            add_meta_data_(metaDataArr[0], metaDataArr[1],
                                           tensor_network.getCublasCaller()));
                    } else {
                        PL_ABORT("Only one wire observables are supported "
                                 "for cutensornet v24.03");
                    }

                    tensorDataPtrPerTerm_.emplace_back(
                        static_cast<const void *>(get_obs_device_ptr_(obsKey)));
                }

                modes_.emplace_back(modes_per_term);

                numModes_.emplace_back(num_modes_per_term);

                // modes pointer initialization
                vector1D<const int32_t *> modesPtrPerTerm;
                for (std::size_t tensor_idx = 0;
                     tensor_idx < modes_.back().size(); tensor_idx++) {
                    modesPtrPerTerm.emplace_back(
                        modes_.back()[tensor_idx].data());
                }
                modesPtr_.emplace_back(modesPtrPerTerm);
                tensorDataPtr_.emplace_back(tensorDataPtrPerTerm_);
            }
        }
    }

  public:
    /**
     * @brief Construct a new ObservableTNCudaOperator object.
     *
     * @param tensor_network Tensor network object.
     * @param obs ObservableTNCuda object.
     * @param var_cal If true, calculate the variance of the observable.
     */
    ObservableTNCudaOperator(const TensorNetT &tensor_network,
                             ObservableTNCuda<TensorNetT> &obs,
                             const bool var_cal = false)
        : tensor_network_{tensor_network},
          numObsTerms_(obs.getNumTensors().size()), var_cal_{var_cal} {
        if (var_cal) {
            initHelper_var_(tensor_network, obs);
        } else {
            initHelper_expval_(tensor_network, obs);
        }

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /* const cutensornetHandle_t */ tensor_network.getTNCudaHandle(),
            /* int32_t */
            static_cast<int32_t>(tensor_network.getNumQubits()),
            /* const int64_t stateModeExtents */
            reinterpret_cast<int64_t *>(const_cast<std::size_t *>(
                tensor_network.getQubitDims().data())),
            /* cudaDataType_t */ tensor_network.getCudaDataType(),
            /* cutensornetNetworkOperator_t */ &obsOperator_));

        const std::size_t numObsTerms =
            var_cal ? (numObsTerms_ * numObsTerms_) : numObsTerms_;
        for (std::size_t term_idx = 0; term_idx < numObsTerms; term_idx++) {
            appendTNOperator_(coeffs_[term_idx], numTensors_[term_idx],
                              numModes_[term_idx].data(),
                              modesPtr_[term_idx].data(),
                              tensorDataPtr_[term_idx].data());
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
     * @brief Append a product of tensors to the
     * `cutensornetNetworkOperator_t`
     *
     * @param coeff Coefficient of the product.
     * @param numTensors Number of tensors in the product.
     * @param numStateModes Number of state modes of each tensor in the
     * product.
     * @param stateModes State modes of each tensor in the product.
     * @param tensorDataPtr Pointer to the data of each tensor in the
     * product.
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
