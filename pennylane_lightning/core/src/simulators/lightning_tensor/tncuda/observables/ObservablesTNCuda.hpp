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
#include <functional>
#include <vector>

#include <iostream>

#include "ObservablesTNCuda_host.hpp"

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Observables.hpp"
#include "TensorCuda.hpp"
#include "Util.hpp"
#include "cuError.hpp"
#include "tncudaError.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Observables;

template <class T> using vector1D = std::vector<T>;
template <class T> using vector2D = std::vector<std::vector<T>>;
template <class T> using vector3D = std::vector<std::vector<std::vector<T>>>;

} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::Observables {

/**
 * @brief A base class for all observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT> class ObservableTNCudaOperator {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;

  private:
    cutensornetNetworkOperator_t obsOperator_{nullptr};

    const StateTensorT &state_tensor_;
    const size_t numObsTerms_;

    std::vector<int64_t> ids_;

    vector1D<cuDoubleComplex> coeffs_;

    vector1D<size_t> numTensors_;

    vector2D<int32_t> numModes_;

    vector3D<int32_t> modes_;

    vector2D<const int32_t *> modesPtr_;

    vector2D<const void *> tensorDataPtr_;

    vector2D<TensorCuda<PrecisionT>> tensorData_;

  public:
    ObservableTNCudaOperator(const StateTensorT &state_tensor,
                             Observable<StateTensorT> &obs)
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

        for (std::size_t term_idx = 0; term_idx < numObsTerms_; term_idx++) {
            // coeffs initialization
            cuDoubleComplex coeff = obs.getCoeffs()[term_idx];
            coeffs_.push_back(coeff);

            // number of tensors in each term
            auto numTensors = obs.getNumTensors()[term_idx];
            numTensors_.push_back(numTensors);

            // number of state modes of each tensor in each term
            vector1D<int32_t> local_num_modes_int32(
                obs.getNumStateModes()[term_idx].size());
            std::transform(obs.getNumStateModes()[term_idx].begin(),
                           obs.getNumStateModes()[term_idx].end(),
                           local_num_modes_int32.begin(),
                           [](size_t x) { return static_cast<int32_t>(x); });
            numModes_.push_back(local_num_modes_int32);

            // modes initialization
            vector2D<int32_t> modes_per_term;
            for (std::size_t tensor_idx = 0; tensor_idx < numTensors;
                 tensor_idx++) {
                auto modes_per_tensor =
                    obs.getStateModes()[term_idx][tensor_idx];
                vector1D<int32_t> modes_per_tensor_int32;
                std::transform(modes_per_tensor.begin(), modes_per_tensor.end(),
                               std::back_inserter(modes_per_tensor_int32),
                               [&](size_t x) {
                                   return static_cast<int32_t>(
                                       state_tensor.getNumQubits() - 1 - x);
                               });
                modes_per_term.push_back(modes_per_tensor_int32);
            }
            modes_.push_back(modes_per_term);

            // modes pointer initialization
            vector1D<const int32_t *> modesPtrPerTerm;
            for (size_t tensor_idx = 0; tensor_idx < modes_.back().size();
                 tensor_idx++) {
                modesPtrPerTerm.push_back(modes_.back()[tensor_idx].data());
            }
            modesPtr_.push_back(modesPtrPerTerm);

            // tensor data initialization
            vector1D<TensorCuda<PrecisionT>> tensorDataPerTerm_;
            for (std::size_t tensor_idx = 0; tensor_idx < numTensors;
                 tensor_idx++) {
                tensorDataPerTerm_.emplace_back(
                    std::vector<std::size_t>(
                        Pennylane::Util::log2(
                            obs.getData()[term_idx][tensor_idx].size()),
                        2),
                    obs.getData()[term_idx][tensor_idx],
                    state_tensor.getDevTag());
            }

            tensorData_.push_back(tensorDataPerTerm_);

            vector1D<const void *> tensorDataPtrPerTerm_;
            for (size_t tensor_idx = 0; tensor_idx < tensorData_.back().size();
                 tensor_idx++) {
                tensorDataPtrPerTerm_.push_back(
                    tensorData_.back()[tensor_idx].getDataBuffer().getData());
            }

            tensorDataPtr_.push_back(tensorDataPtrPerTerm_);

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

/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateTensorT State tensor class.
 */

/*
template <typename StateTensorT>
class NamedObsTNCuda final : public ObservableTNCuda<StateTensorT> {
  private:
    using BaseType = ObservableTNCuda<StateTensorT>;
    using PrecisionT = typename StateTensorT::PrecisionT;

    NamedObs<StateTensorT> obs_;

    std::size_t numQubits_;
    cuDoubleComplex coeff_{1, 0.0};
    std::size_t numTensors_{1};
    std::vector<int32_t> wires_int_;
    std::vector<int32_t> numStateModes_;
    std::vector<const int32_t *> stateModes_;
    std::vector<const void *> tensorDataPtr_;
    std::vector<TensorCuda<PrecisionT>> tensorData_;

  public:
    NamedObsTNCuda(const StateTensorT &state_tensor,
                   Observable<StateTensorT> &obs)
        : BaseType(state_tensor) {
        numQubits_ = state_tensor.getNumQubits();

        wires_int_ = std::vector<int32_t>(wires.size());

        numStateModes_.push_back(static_cast<int32_t>(wires.size()));

        std::cout << getObsName() << std::endl;

        std::transform(
            wires.begin(), wires.end(), wires_int_.begin(),
            [&](size_t x) { return static_cast<int32_t>(numQubits_ - x - 1); });

        stateModes_.push_back(wires_int_.data());

        auto &&par = (params.empty()) ? std::vector<PrecisionT>{0.0} : params;

        auto &gateMap =
            cuGates::DynamicGateDataAccess<PrecisionT>::getInstance();

        tensorData_.emplace_back(
            std::vector<std::size_t>(2 * wires_int_.size(), 2),
            gateMap.getGateData(obs_name, par), state_tensor.getDevTag());

        tensorDataPtr_.push_back(tensorData_.back().getDataBuffer().getData());

        BaseType::appendTNOperator(coeff_, numTensors_, numStateModes_.data(),
                                   stateModes_.data(), tensorDataPtr_.data());
    }

    ~NamedObsTNCuda() {}

    [[nodiscard]] auto getObsName() const -> std::string {
        return obs_.getObsName();
    }

    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> {
        return obs_.getWires();
    }
};
*/

/*
template <typename StateTensorT>
class NamedObs final : public ObservableTNCuda<StateTensorT> {
  private:
    using BaseType = ObservableTNCuda<StateTensorT>;
    using PrecisionT = typename StateTensorT::PrecisionT;

  private:
    std::string obs_name_;
    std::vector<std::size_t> wires_;
    std::vector<PrecisionT> params_;

  private:
    std::size_t numQubits_;
    cuDoubleComplex coeff_{1, 0.0};
    std::size_t numTensors_{1};
    std::vector<int32_t> wires_int_;
    std::vector<int32_t> numStateModes_;
    std::vector<const int32_t *> stateModes_;
    std::vector<const void *> tensorDataPtr_;
    std::vector<TensorCuda<PrecisionT>> tensorData_;

  private:
    [[nodiscard]] auto
    isEqual(const ObservableTNCuda<StateTensorT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const NamedObs<StateTensorT> &>(other);

        return (obs_name_ == other_cast.obs_name_) &&
               (wires_ == other_cast.wires_) && (params_ == other_cast.params_);
    }

  public:
    /-**
     * @brief Construct a NamedObs object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     *-/
    NamedObs(const StateTensorT &state_tensor, std::string obs_name,
             std::vector<std::size_t> wires,
             std::vector<PrecisionT> params = {})
        : BaseType(state_tensor), obs_name_{std::move(obs_name)}, wires_{wires},
          params_{std::move(params)} {
        using Pennylane::Gates::Constant::gate_names;
        using Pennylane::Gates::Constant::gate_num_params;
        using Pennylane::Gates::Constant::gate_wires;

        const auto gate_op = lookup(reverse_pairs(gate_names),
                                    std::string_view{this->obs_name_});
        PL_ASSERT(lookup(gate_wires, gate_op) == this->wires_.size());
        PL_ASSERT(lookup(gate_num_params, gate_op) == this->params_.size());

        numQubits_ = state_tensor.getNumQubits();

        wires_int_ = std::vector<int32_t>(wires_.size());

        numStateModes_.push_back(static_cast<int32_t>(wires_.size()));

        std::transform(
            wires_.begin(), wires_.end(), wires_int_.begin(),
            [&](size_t x) { return static_cast<int32_t>(numQubits_ - x - 1); });

        stateModes_.push_back(wires_int_.data());

        auto &&par = (params_.empty()) ? std::vector<PrecisionT>{0.0} : params_;


        tensorData_.emplace_back(
            std::vector<std::size_t>(2 * wires_int_.size(), 2),
            state_tensor.getGateCache()->get_gate_host_vector(obs_name_, par),
            state_tensor.getDevTag());

        tensorDataPtr_.push_back(tensorData_.back().getDataBuffer().getData());

        BaseType::appendTNOperator(coeff_, numTensors_, numStateModes_.data(),
                                   stateModes_.data(), tensorDataPtr_.data());
    }

    ~NamedObs() {}

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << obs_name_ << wires_;
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> override {
        return wires_;
    }
};*/
} // namespace Pennylane::LightningTensor::TNCuda::Observables
