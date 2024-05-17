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

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Observables.hpp"
#include "TNCudaGateCache.hpp"
#include "TensorCuda.hpp"
#include "Util.hpp"
#include "cuError.hpp"
#include "tncudaError.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Observables;
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
template <class StateTensorT> class ObservableTNCuda {
  private:
    cutensornetNetworkOperator_t obsOperator_{nullptr};
    const StateTensorT &state_tensor_;
    int64_t id_;

  public:
    using PrecisionT = typename StateTensorT::PrecisionT;

    ObservableTNCuda(const StateTensorT &state_tensor)
        : state_tensor_{state_tensor} {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /* const cutensornetHandle_t */ state_tensor.getTNCudaHandle(),
            /* int32_t */ static_cast<int32_t>(state_tensor.getNumQubits()),
            /* const int64_t stateModeExtents */
            reinterpret_cast<int64_t *>(
                const_cast<size_t *>(state_tensor.getQubitDims().data())),
            /* cudaDataType_t */ state_tensor.getCudaDataType(),
            /* cutensornetNetworkOperator_t */ &obsOperator_));
    }

    // private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param Another instance of subclass of Observable<StateTensorT> to
     * compare.
     */
    //[[nodiscard]] virtual bool
    // isEqual(const ObservableTNCuda<StateTensorT> &other) const = 0;

  public:
    virtual ~ObservableTNCuda() {
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyNetworkOperator(obsOperator_));
    }

    cutensornetNetworkOperator_t getTNOperator() { return obsOperator_; }

    void appendTNOperator(cuDoubleComplex coefficient, std::size_t numTensors,
                          const int32_t *numStateModes,
                          const int32_t **stateModes,
                          const void **tensorDataPtr) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetNetworkOperatorAppendProduct(
            /* const cutensornetHandle_t */ state_tensor_.getTNCudaHandle(),
            /* cutensornetNetworkOperator_t */ getTNOperator(),
            /* cuDoubleComplex coefficient*/ coefficient,
            /* int32_t numTensors */ static_cast<int32_t>(numTensors),
            /* const int32_t numStateModes[] */ numStateModes,
            /* const int32_t *stateModes[] */ stateModes,
            /* const int64_t *tensorModeStrides[] */ nullptr,
            /* const void *tensorData[] */ tensorDataPtr,
            /* int64_t* */ &id_));
    }

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] virtual auto getObsName() const -> std::string = 0;

    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] virtual auto getWires() const -> std::vector<std::size_t> = 0;

    /**
     * @brief Get the observable data.
     *
     */
    [[nodiscard]] virtual auto getObs() const
        -> std::vector<std::shared_ptr<ObservableTNCuda<StateTensorT>>> {
        return {};
    };

    /**
     * @brief Get the coefficients of a Hamiltonian observable.
     */
    [[nodiscard]] virtual auto getCoeffs() const -> std::vector<PrecisionT> {
        return {};
    };

    /**
     * @brief Test whether this object is equal to another object
     */
    [[nodiscard]] auto operator==(const Observable<StateTensorT> &other) const
        -> bool {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] auto operator!=(const Observable<StateTensorT> &other) const
        -> bool {
        return !(*this == other);
    }
};

/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateTensorT State tensor class.
 */

template <typename StateTensorT>
class NamedObsTNCuda final : public ObservableTNCuda<StateTensorT> {
  private:
    using BaseType = ObservableTNCuda<StateTensorT>;
    using PrecisionT = typename StateTensorT::PrecisionT;

    NamedObsBase<StateTensorT> obs_;

    std::size_t numQubits_;
    cuDoubleComplex coeff_{1, 0.0};
    std::size_t numTensors_{1};
    std::vector<int32_t> wires_int_;
    std::vector<int32_t> numStateModes_;
    std::vector<const int32_t *> stateModes_;
    std::vector<const void *> tensorDataPtr_;
    std::vector<TensorCuda<PrecisionT>> tensorData_;

  public:
    NamedObsTNCuda(const StateTensorT &state_tensor, std::string obs_name,
                   std::vector<std::size_t> wires,
                   std::vector<PrecisionT> params = {})
        : BaseType(state_tensor), obs_{obs_name, wires, params} {
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
