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
    std::unordered_map<std::string, std::string> pauli_map_{{"Identity", "I"},
                                                            {"PauliX", "X"},
                                                            {"PauliY", "Y"},
                                                            {"PauliZ", "Z"},
                                                            {"Hadamard", "H"}};

  private:
    SharedCublasCaller cublascaller_;
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

    /*--------------------Var Support Below------------------------*/
    std::size_t numObsTerms2_;          // number of observable terms
    vector1D<cuDoubleComplex> coeffs2_; // coefficients for each term
    vector1D<std::size_t> numTensors2_; // number of tensors in each term

    vector2D<int32_t>
        numModes2_; // number of state modes of each tensor in each term

    vector3D<int32_t> modes2_; // modes for each tensor in each term

    vector2D<const int32_t *>
        modesPtr2_; // pointers for modes of each tensor in each term

    vector2D<const void *>
        tensorDataPtr2_; // pointers for each tensor data in each term
    /*--------------------Var Support Above------------------------*/
    const bool var_cal_ = false;

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
                             ObservableTNCuda<TensorNetT> &obs,
                             const bool var_cal = false)
        : tensor_network_{tensor_network},
          numObsTerms_(obs.getNumTensors().size()), var_cal_{var_cal} {
        if (var_cal) {
            // PL_ABORT_IF_NOT(
            //     numObsTerms_ == 1,
            //     "Only one observable term is allowed for variance
            //     calculation");
            cublascaller_ = make_shared_cublas_caller();
        }

        numTensors_ = obs.getNumTensors(); // number of tensors in each term

        for (std::size_t term_idx = 0; term_idx < numObsTerms_; term_idx++) {
            PrecisionT coeff_real = obs.getCoeffs()[term_idx];
            // if (var_cal) {
            //     coeff_real = coeff_real * coeff_real;
            // }
            auto coeff = cuDoubleComplex{static_cast<double>(coeff_real), 0.0};
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
            // Not required for var calculation below
            //  modes pointer initialization
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
                // if (var_cal) {
                //     obsName = obsName + "_squared";
                // }
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
                        /*
                        if (var_cal) {
                            square_matrix_CUDA_device(
                                const_cast<CFP_t *>(
                                    get_obs_device_ptr_(obsKey)),
                                Pennylane::Util::log2(
                                    hermitianMatrix_cu.size()),
                                Pennylane::Util::log2(
                                    hermitianMatrix_cu.size()),
                                tensor_network_.getDevTag().getDeviceID(),
                                tensor_network_.getDevTag().getStreamID(),
                                *cublascaller_);
                        }
                        */
                    }
                }
                tensorDataPtrPerTerm_.emplace_back(get_obs_device_ptr_(obsKey));
            }

            tensorDataPtr_.emplace_back(tensorDataPtrPerTerm_);
            // Not required for var calculation above
        }

        if (var_cal) {
            numObsTerms2_ = numObsTerms_ * numObsTerms_;

            for (std::size_t term_idx = 0; term_idx < numObsTerms_;
                 term_idx++) {
                for (std::size_t term_idy = 0; term_idy < numObsTerms_;
                     term_idy++) {
                    auto coeff = cuDoubleComplex{
                        static_cast<double>(obs.getCoeffs()[term_idx] *
                                            obs.getCoeffs()[term_idy]),
                        0.0};
                    coeffs2_.emplace_back(coeff);

                    auto modes_termx = modes_[term_idx];
                    auto modes_termy = modes_[term_idy];

                    std::unordered_map<int32_t,
                                       std::vector<MetaDataT>>
                        modes_obsname_map; // Note that one-wire observables are
                                           // supported as cutensornet v24.03

                    for (std::size_t tensor_idx = 0;
                         tensor_idx < modes_termx.size(); tensor_idx++) {
                        PL_ABORT_IF_NOT(modes_termx[tensor_idx].size() == 1,
                                        "Only one-wire observables are "
                                        "supported for cutensornet v24.03");

                        modes_obsname_map[modes_termx[tensor_idx][0]] = {
                            obs.getMetaData()[term_idx][tensor_idx]};
                    }

                    for (std::size_t tensor_idy = 0;
                         tensor_idy < modes_termy.size(); tensor_idy++) {
                        auto it = modes_obsname_map.find(
                            modes_termy[tensor_idy].front());
                        if (it != modes_obsname_map.end()) {
                            modes_obsname_map[modes_termy[tensor_idy].front()]
                                .push_back(
                                    obs.getMetaData()[term_idy][tensor_idy]);
                        } else {
                            modes_obsname_map[modes_termy[tensor_idy].front()] =
                                {obs.getMetaData()[term_idy][tensor_idy]};
                        }
                    }

                    auto numTensorsPerTerm = modes_obsname_map.size();

                    numTensors2_.emplace_back(numTensorsPerTerm);

                    vector2D<int32_t> modes_per_term;
                    vector1D<const void *> tensorDataPtrPerTerm_;
                    vector1D<int32_t> num_modes_per_term;

                    for (const auto &tensors_info : modes_obsname_map) {
                        modes_per_term.emplace_back(
                            std::vector<int32_t>{tensors_info.first});

                        num_modes_per_term.emplace_back(
                            modes_per_term.back().size());
                        auto metaDataArr = tensors_info.second;
                        if (metaDataArr.size() == 1) {
                            auto metaData = metaDataArr[0];
                            auto obsName = std::get<0>(metaData);
                            auto param = std::get<1>(metaData);
                            auto hermitianMatrix = std::get<2>(metaData);
                            std::size_t hash_val = 0;

                            if (!hermitianMatrix.empty()) {
                                hash_val = MatrixHasher()(hermitianMatrix);
                            }

                            auto obsKey =
                                std::make_tuple(obsName, param, hash_val);

                            if (device_obs_cache_.find(obsKey) ==
                                device_obs_cache_.end()) {
                                if (hermitianMatrix.empty()) {
                                    add_obs_(obsName, param);
                                } else {
                                    auto hermitianMatrix_cu =
                                        cuUtil::complexToCu<ComplexT>(
                                            hermitianMatrix);
                                    add_obs_(obsKey, hermitianMatrix_cu);
                                }
                            }

                            tensorDataPtrPerTerm_.emplace_back(
                                get_obs_device_ptr_(obsKey));

                        } else {
                            PL_ABORT_IF(metaDataArr.size() > 2,
                                        "DEBUG PURPOSE ONLY");
                            auto metaData0 = metaDataArr[0];
                            auto metaData1 = metaDataArr[1];

                            auto param0 = std::get<1>(metaData0);
                            auto param1 = std::get<1>(metaData1);

                            auto obsName0 = std::get<0>(metaData0);
                            auto obsName1 = std::get<0>(metaData1);

                            std::string obsName = obsName0 + "@" + obsName1;

                            // Branch for Pauli strings
                            if (pauli_map_.find(obsName0) != pauli_map_.end() &&
                                pauli_map_.find(obsName1) != pauli_map_.end()) {
                                obsName0 = pauli_map_[obsName0];
                                obsName1 = pauli_map_[obsName1];
                                obsName = obsName0 + "@" + obsName1;

                                auto obsKey = std::make_tuple(
                                    obsName, std::vector<PrecisionT>{},
                                    std::size_t{0});
                                if (device_obs_cache_.find(obsKey) ==
                                    device_obs_cache_.end()) {
                                    add_obs_(obsName, param0);
                                }
                                tensorDataPtrPerTerm_.emplace_back(
                                    static_cast<const void *>(
                                        get_obs_device_ptr_(obsKey)));
                            }
                            // Hermitian below to be tidy up
                            else {
                                // Branch for Hermtian involving Pauli strings
                                // add both observables matrix to GPU cache
                                auto hermitianMatrix0 =
                                    std::get<2>(metaDataArr[0]);
                                auto hermitianMatrix1 =
                                    std::get<2>(metaDataArr[1]);
                                std::size_t hash_val0 = 0;
                                std::size_t hash_val1 = 0;
                                if (!hermitianMatrix0.empty()) {
                                    hash_val0 =
                                        MatrixHasher()(hermitianMatrix0);
                                }
                                if (!hermitianMatrix1.empty()) {
                                    hash_val1 =
                                        MatrixHasher()(hermitianMatrix1);
                                }
                                auto obsKey0 = std::make_tuple(
                                    obsName0, std::vector<PrecisionT>{},
                                    hash_val0);
                                auto obsKey1 = std::make_tuple(
                                    obsName1, std::vector<PrecisionT>{},
                                    hash_val1);

                                if (device_obs_cache_.find(obsKey0) ==
                                    device_obs_cache_.end()) {
                                    if (hermitianMatrix0.empty()) {
                                        add_obs_(obsName0, param0);
                                    } else {
                                        auto hermitianMatrix_cu =
                                            cuUtil::complexToCu<ComplexT>(
                                                hermitianMatrix0);
                                        add_obs_(obsKey0, hermitianMatrix_cu);
                                    }
                                }

                                if (device_obs_cache_.find(obsKey1) ==
                                    device_obs_cache_.end()) {
                                    if (hermitianMatrix1.empty()) {
                                        add_obs_(obsName1, param1);
                                    } else {
                                        auto hermitianMatrix_cu =
                                            cuUtil::complexToCu<ComplexT>(
                                                hermitianMatrix1);
                                        add_obs_(obsKey1, hermitianMatrix_cu);
                                    }
                                }

                                // add the combined observable matrix together
                                auto obsName = obsName0 + "@" + obsName1;

                                PL_ABORT_IF(hermitianMatrix0.size() !=
                                                hermitianMatrix1.size(),
                                            "DEBUG PURPOSE ONLY");

                                std::size_t hash_val = 0;

                                auto hermitianMatrix = hermitianMatrix0;
                                if (!hermitianMatrix1.empty()) {
                                    hermitianMatrix.insert(
                                        hermitianMatrix.end(),
                                        hermitianMatrix1.begin(),
                                        hermitianMatrix1.end());
                                }

                                if (!hermitianMatrix.empty()) {
                                    hash_val = MatrixHasher()(hermitianMatrix);
                                }

                                auto obsKey = std::make_tuple(
                                    obsName, std::vector<PrecisionT>{},
                                    hash_val);

                                if (device_obs_cache_.find(obsKey) ==
                                    device_obs_cache_.end()) {

                                    auto hermitianMatrix_cu =
                                        cuUtil::complexToCu<ComplexT>(
                                            hermitianMatrix0);
                                    add_obs_(obsKey, hermitianMatrix_cu);
                                    // update the matrix data with MM operation
                                    MM_CUDA_device(
                                        const_cast<CFP_t *>(
                                            get_obs_device_ptr_(obsKey0)),
                                        const_cast<CFP_t *>(
                                            get_obs_device_ptr_(obsKey1)),
                                        const_cast<CFP_t *>(
                                            get_obs_device_ptr_(obsKey)),
                                        Pennylane::Util::log2(
                                            hermitianMatrix_cu.size()),
                                        Pennylane::Util::log2(
                                            hermitianMatrix_cu.size()),
                                        Pennylane::Util::log2(
                                            hermitianMatrix_cu.size()),
                                        tensor_network_.getDevTag()
                                            .getDeviceID(),
                                        tensor_network_.getDevTag()
                                            .getStreamID(),
                                        *cublascaller_);
                                }
                                tensorDataPtrPerTerm_.emplace_back(
                                    get_obs_device_ptr_(obsKey));
                            }
                        }
                        // Hermitian above
                    }
                    modes2_.emplace_back(modes_per_term);
                    numModes2_.emplace_back(num_modes_per_term);

                    // modes pointer initialization
                    vector1D<const int32_t *> modesPtrPerTerm;
                    for (std::size_t tensor_idx = 0;
                         tensor_idx < modes2_.back().size(); tensor_idx++) {
                        modesPtrPerTerm.emplace_back(
                            modes2_.back()[tensor_idx].data());
                    }
                    modesPtr2_.emplace_back(modesPtrPerTerm);
                    tensorDataPtr2_.emplace_back(tensorDataPtrPerTerm_);
                }
            }
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

        if (var_cal) {
            for (std::size_t term_idx = 0; term_idx < numObsTerms2_;
                 term_idx++) {
                appendTNOperator_(coeffs2_[term_idx], numTensors2_[term_idx],
                                  numModes2_[term_idx].data(),
                                  modesPtr2_[term_idx].data(),
                                  tensorDataPtr2_[term_idx].data());
            }

        } else {
            for (std::size_t term_idx = 0; term_idx < numObsTerms_;
                 term_idx++) {
                appendTNOperator_(coeffs_[term_idx], numTensors_[term_idx],
                                  numModes_[term_idx].data(),
                                  modesPtr_[term_idx].data(),
                                  tensorDataPtr_[term_idx].data());
            }
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
